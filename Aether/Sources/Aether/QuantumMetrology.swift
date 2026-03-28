// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate
import Foundation

/// Threshold for safe normalization division.
private let normalizationEpsilon: Double = 1e-15
/// Threshold below which phase angles are treated as zero.
private let phaseEpsilon: Double = 1e-10

/// Prior probability distribution for Bayesian phase estimation.
///
/// Specifies initial knowledge about the unknown phase before measurements.
/// The prior is updated to a posterior via Bayes' theorem after each measurement.
/// Supports uniform (no prior knowledge), Gaussian (peaked around expected value),
/// and custom discretized distributions.
///
/// **Example:**
/// ```swift
/// let uniform = PhasePrior.uniform
/// let gaussian = PhasePrior.gaussian(mean: .pi / 4, stdDev: 0.3)
/// let custom = PhasePrior.custom([0.1, 0.2, 0.4, 0.2, 0.1])
/// ```
///
/// - SeeAlso: ``BayesianPhaseEstimation``
/// - SeeAlso: ``BayesianPhaseEstimation/Result``
@frozen
public enum PhasePrior: Sendable {
    case uniform
    case gaussian(mean: Double, stdDev: Double)
    case custom([Double])
}

/// Adaptive Bayesian phase estimation for high-precision unitary eigenvalue determination.
///
/// Implements adaptive Bayesian inference for estimating the eigenphase phi of a unitary
/// operator U where U|psi> = exp(i*phi)|psi>. Uses controlled-U^k operations with adaptively
/// chosen k to maximize information gain per measurement. Achieves Heisenberg-limited
/// precision scaling: sigma proportional to 1/N (vs 1/sqrt(N) standard quantum limit).
///
/// The algorithm maintains a discretized posterior distribution over [0, 2pi) and updates
/// it after each measurement using Bayes' theorem. Adaptive selection of the power k
/// maximizes expected information gain based on current posterior variance.
///
/// **Example:**
/// ```swift
/// let estimator = BayesianPhaseEstimation(
///     unitary: .rotationZ(.pi / 4),
///     prior: .uniform,
///     discretizationBins: 256
/// )
///
/// let result = await estimator.run(
///     maxMeasurements: 100,
///     targetPrecision: 0.01
/// ) { iteration, stdDev in
///     print("Iteration \(iteration): sigma = \(stdDev)")
/// }
///
/// print("Estimated phase: \(result.mapEstimate)")
/// ```
///
/// - SeeAlso: ``PhasePrior``
/// - SeeAlso: ``BayesianPhaseEstimation/Result``
/// - SeeAlso: ``QuantumCircuit/phaseEstimation(unitary:precisionQubits:eigenstateQubits:)``
public actor BayesianPhaseEstimation {
    private let unitary: QuantumGate
    private let discretizationBins: Int
    private var posterior: [Double]
    private let phaseGrid: [Double]
    private let sinGrid: [Double]
    private let cosGrid: [Double]
    private let deltaPhase: Double
    private var posteriorBuffer: [Double]

    /// Result of Bayesian phase estimation containing posterior distribution and statistics.
    ///
    /// Provides maximum a posteriori (MAP) estimate, posterior mean, standard deviation,
    /// and full discretized posterior distribution. The posterior precision typically
    /// achieves Heisenberg scaling: sigma proportional to 1/N measurements (vs 1/sqrt(N) classical).
    ///
    /// **Example:**
    /// ```swift
    /// let result = await estimator.run(maxMeasurements: 100)
    /// print("Phase estimate: \(result.mapEstimate)")
    /// print("Uncertainty: \(result.posteriorStdDev)")
    /// print("Measurements used: \(result.measurementCount)")
    /// ```
    ///
    /// - SeeAlso: ``BayesianPhaseEstimation``
    /// - SeeAlso: ``PhasePrior``
    @frozen
    public struct Result: Sendable, CustomStringConvertible {
        /// Maximum a posteriori phase estimate (mode of posterior).
        public let mapEstimate: Double

        /// Expected value of posterior distribution.
        public let posteriorMean: Double

        /// Standard deviation of posterior distribution.
        public let posteriorStdDev: Double

        /// Full discretized posterior probability distribution over [0, 2pi).
        public let posterior: [Double]

        /// Total number of measurements performed.
        public let measurementCount: Int

        /// Multi-line formatted summary of estimation results.
        @inlinable
        public var description: String {
            """
            BayesianPhaseEstimation.Result:
              MAP Estimate: \(String(format: "%.6f", mapEstimate)) rad
              Posterior Mean: \(String(format: "%.6f", posteriorMean)) rad
              Posterior StdDev: \(String(format: "%.6f", posteriorStdDev)) rad
              Measurements: \(measurementCount)
            """
        }
    }

    /// Creates Bayesian phase estimator for given unitary gate.
    ///
    /// Initializes posterior distribution from specified prior. The phase space [0, 2pi)
    /// is discretized into the specified number of bins for numerical integration.
    /// Higher bin counts improve precision but increase memory and computation.
    ///
    /// **Example:**
    /// ```swift
    /// let estimator = BayesianPhaseEstimation(
    ///     unitary: .pauliZ,
    ///     prior: .gaussian(mean: .pi, stdDev: 0.5),
    ///     discretizationBins: 512
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - unitary: Single-qubit unitary gate whose eigenphase to estimate
    ///   - prior: Prior distribution over phase (default: uniform)
    ///   - discretizationBins: Number of bins for phase discretization (default: 256)
    /// - Precondition: unitary must be a single-qubit gate
    /// - Precondition: discretizationBins must be positive
    /// - Precondition: stdDev must be positive (for gaussian prior)
    /// - Precondition: custom PDF must be non-empty (for custom prior)
    /// - Complexity: O(discretizationBins)
    public init(
        unitary: QuantumGate,
        prior: PhasePrior = .uniform,
        discretizationBins: Int = 256,
    ) {
        ValidationUtilities.validateControlledGateIsSingleQubit(unitary.qubitsRequired)
        ValidationUtilities.validatePositiveInt(discretizationBins, name: "discretizationBins")

        self.unitary = unitary
        self.discretizationBins = discretizationBins
        let step = 2.0 * .pi / Double(discretizationBins)
        deltaPhase = step

        let grid = [Double](unsafeUninitializedCapacity: discretizationBins) { buffer, count in
            for i in 0 ..< discretizationBins {
                buffer[i] = Double(i) * step
            }
            count = discretizationBins
        }
        phaseGrid = grid
        sinGrid = grid.map { sin($0) }
        cosGrid = grid.map { cos($0) }

        switch prior {
        case .uniform:
            let uniformValue = 1.0 / Double(discretizationBins)
            posterior = [Double](repeating: uniformValue, count: discretizationBins)

        case let .gaussian(mean, stdDev):
            ValidationUtilities.validatePositiveDouble(stdDev, name: "stdDev")
            var priorValues = [Double](unsafeUninitializedCapacity: discretizationBins) {
                buffer, count in
                let normFactor = 1.0 / (stdDev * sqrt(2.0 * .pi))
                let halfInvVar = -0.5 / (stdDev * stdDev)
                for i in 0 ..< discretizationBins {
                    let diff = grid[i] - mean
                    buffer[i] = normFactor * exp(diff * diff * halfInvVar)
                }
                count = discretizationBins
            }
            var sum = 0.0
            vDSP_sveD(priorValues, 1, &sum, vDSP_Length(priorValues.count))
            if sum > normalizationEpsilon {
                var invSum = 1.0 / sum
                vDSP_vsmulD(priorValues, 1, &invSum, &priorValues, 1, vDSP_Length(priorValues.count))
            }
            posterior = priorValues

        case let .custom(pdf):
            ValidationUtilities.validateNonEmpty(pdf, name: "custom PDF")
            var normalized: [Double]
            if pdf.count < discretizationBins {
                normalized = pdf
                let padValue = 1.0 / Double(discretizationBins)
                normalized.reserveCapacity(discretizationBins)
                while normalized.count < discretizationBins {
                    normalized.append(padValue)
                }
            } else if pdf.count > discretizationBins {
                normalized = Array(pdf.prefix(discretizationBins))
            } else {
                normalized = pdf
            }
            var sum = 0.0
            vDSP_sveD(normalized, 1, &sum, vDSP_Length(normalized.count))
            if sum > normalizationEpsilon {
                var invSum = 1.0 / sum
                vDSP_vsmulD(normalized, 1, &invSum, &normalized, 1, vDSP_Length(normalized.count))
            }
            posterior = normalized
        }

        posteriorBuffer = [Double](repeating: 0.0, count: discretizationBins)
    }

    /// Runs adaptive Bayesian phase estimation until convergence or measurement limit.
    ///
    /// Iteratively performs measurements with adaptively chosen controlled-U^k operations
    /// and updates the posterior distribution. Stops when target precision is reached
    /// (if specified) or maximum measurements are exhausted. Progress callback receives
    /// current iteration count and posterior standard deviation.
    ///
    /// The adaptive strategy selects k to maximize expected information gain based on
    /// current posterior variance. This achieves near-optimal Heisenberg scaling.
    ///
    /// **Example:**
    /// ```swift
    /// let result = await estimator.run(
    ///     maxMeasurements: 200,
    ///     targetPrecision: 0.005
    /// ) { iter, sigma in
    ///     await MainActor.run { progressLabel.text = "sigma = \(sigma)" }
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - maxMeasurements: Maximum number of measurements to perform
    ///   - targetPrecision: Optional target standard deviation for early stopping
    ///   - progress: Optional callback receiving (iteration, currentStdDev)
    /// - Returns: ``Result`` with posterior statistics and estimates
    /// - Precondition: maxMeasurements must be positive
    /// - Precondition: targetPrecision must be positive (when non-nil)
    /// - Complexity: O(maxMeasurements * discretizationBins)
    @_optimize(speed)
    public func run(
        maxMeasurements: Int,
        targetPrecision: Double? = nil,
        progress: (@Sendable (Int, Double) async -> Void)? = nil,
    ) async -> Result {
        ValidationUtilities.validatePositiveInt(maxMeasurements, name: "maxMeasurements")
        if let precision = targetPrecision {
            ValidationUtilities.validatePositiveDouble(precision, name: "targetPrecision")
        }

        var measurementCount = 0
        let simulator = QuantumSimulator()
        let phaseAngle = extractPhaseFromUnitary()
        var currentStdDev = computePosteriorStdDev()

        for iteration in 0 ..< maxMeasurements {
            if let target = targetPrecision, currentStdDev <= target {
                await progress?(iteration, currentStdDev)
                break
            }

            let adaptiveK = selectAdaptiveK(stdDev: currentStdDev)

            let outcome = await performMeasurement(k: adaptiveK, simulator: simulator, phaseAngle: phaseAngle)

            updatePosterior(outcome: outcome, k: adaptiveK)

            measurementCount = iteration + 1
            currentStdDev = computePosteriorStdDev()

            await progress?(measurementCount, currentStdDev)
        }

        return buildResult(measurementCount: measurementCount)
    }

    /// Selects optimal controlled-U power based on current posterior width.
    @_optimize(speed)
    @_effects(readonly)
    @inline(__always)
    private func selectAdaptiveK(stdDev: Double) -> Int {
        guard stdDev > phaseEpsilon else { return 1 }
        let optimalK = Int(round(1.0 / (2.0 * stdDev)))
        return max(1, min(optimalK, 64))
    }

    /// Executes controlled-U^k circuit and returns measurement outcome (0 or 1).
    @_optimize(speed)
    private func performMeasurement(k: Int, simulator: QuantumSimulator, phaseAngle: Double) async -> Int {
        var circuit = QuantumCircuit(qubits: 2)

        circuit.append(.hadamard, to: 0)

        circuit.append(.controlledPhase(Double(k) * phaseAngle), to: [0, 1])

        circuit.append(.hadamard, to: 0)

        let state = await simulator.execute(circuit)

        let sample = Double.random(in: 0.0 ..< 1.0)
        let (p0, _) = state.probabilities(for: 0)

        return sample < p0 ? 0 : 1
    }

    /// Extracts eigenphase from the unitary gate's matrix representation.
    @_effects(readonly)
    private func extractPhaseFromUnitary() -> Double {
        let matrix = unitary.matrix()
        let eigenvalue1 = matrix[0][0]
        return atan2(eigenvalue1.imaginary, eigenvalue1.real)
    }

    /// Applies Bayesian update to posterior distribution given measurement outcome.
    @_optimize(speed)
    private func updatePosterior(outcome: Int, k: Int) {
        let kDouble = Double(k)
        if outcome == 0 {
            for i in 0 ..< discretizationBins {
                let cos2x = cos(kDouble * phaseGrid[i])
                posteriorBuffer[i] = (1.0 + cos2x) * 0.5 * posterior[i]
            }
        } else {
            for i in 0 ..< discretizationBins {
                let cos2x = cos(kDouble * phaseGrid[i])
                posteriorBuffer[i] = (1.0 - cos2x) * 0.5 * posterior[i]
            }
        }

        var normalization = 0.0
        vDSP_sveD(posteriorBuffer, 1, &normalization, vDSP_Length(posteriorBuffer.count))
        if normalization > normalizationEpsilon {
            var invNorm = 1.0 / normalization
            vDSP_vsmulD(posteriorBuffer, 1, &invNorm, &posteriorBuffer, 1, vDSP_Length(posteriorBuffer.count))
        }

        posterior = posteriorBuffer
    }

    /// Computes circular mean of the posterior distribution.
    @_effects(readonly)
    private func computePosteriorMean() -> Double {
        var sumSin = 0.0
        vDSP_dotprD(posterior, 1, sinGrid, 1, &sumSin, vDSP_Length(discretizationBins))
        var sumCos = 0.0
        vDSP_dotprD(posterior, 1, cosGrid, 1, &sumCos, vDSP_Length(discretizationBins))
        return atan2(sumSin, sumCos)
    }

    /// Computes circular standard deviation of the posterior distribution.
    @_effects(readonly)
    private func computePosteriorStdDev(mean: Double? = nil) -> Double {
        let resolvedMean = mean ?? computePosteriorMean()
        var variance = 0.0
        for i in 0 ..< discretizationBins {
            var diff = phaseGrid[i] - resolvedMean
            if diff > .pi { diff -= 2.0 * .pi }
            variance += posterior[i] * diff * diff
        }
        return sqrt(max(0.0, variance))
    }

    /// Finds the maximum a posteriori phase estimate.
    @_effects(readonly)
    private func findMAP() -> Double {
        var maxValue = 0.0
        var maxIndex: vDSP_Length = 0
        vDSP_maxviD(posterior, 1, &maxValue, &maxIndex, vDSP_Length(posterior.count))
        return phaseGrid[Int(maxIndex)]
    }

    /// Assembles estimation result from current posterior statistics.
    @_effects(readonly)
    private func buildResult(measurementCount: Int) -> Result {
        let mean = computePosteriorMean()
        return Result(
            mapEstimate: findMAP(),
            posteriorMean: mean,
            posteriorStdDev: computePosteriorStdDev(mean: mean),
            posterior: posterior,
            measurementCount: measurementCount,
        )
    }
}

/// Result of Ramsey interferometry measurement including phase and coherence information.
///
/// Contains accumulated phase from free evolution, fringe visibility (coherence measure),
/// optionally estimated T2* decoherence time, and excited state probability. Visibility
/// V = (P_max - P_min)/(P_max + P_min) indicates quantum coherence preservation.
///
/// **Example:**
/// ```swift
/// let result = state.ramseyResult(evolutionTime: 1e-6)
/// print("Phase: \(result.phaseAccumulation)")
/// print("Visibility: \(result.visibility)")
/// if let t2star = result.estimatedT2Star {
///     print("T2*: \(t2star)")
/// }
/// ```
///
/// - SeeAlso: ``QuantumState/ramseyResult(evolutionTime:)``
@frozen
public struct RamseyResult: Sendable, CustomStringConvertible {
    /// Accumulated phase during free evolution (radians).
    public let phaseAccumulation: Double

    /// Fringe visibility measuring coherence: V = (P_max - P_min)/(P_max + P_min).
    public let visibility: Double

    /// Estimated T2* decoherence time from visibility decay (nil if not computed).
    public let estimatedT2Star: Double?

    /// Probability of measuring the excited state |1>.
    public let excitedStateProbability: Double

    /// Multi-line formatted summary of Ramsey measurement results.
    @inlinable
    public var description: String {
        var result = """
        RamseyResult:
          Phase Accumulation: \(String(format: "%.6f", phaseAccumulation)) rad
          Visibility: \(String(format: "%.4f", visibility))
          P(|1>): \(String(format: "%.4f", excitedStateProbability))
        """
        if let t2 = estimatedT2Star {
            result += "\n  Estimated T2*: \(String(format: "%.4e", t2))"
        }
        return result
    }
}

public extension QuantumCircuit {
    /// Creates Ramsey interferometry circuit: H - Rz(phase) - H.
    ///
    /// Implements standard Ramsey sequence for phase sensing. Starting from |0>,
    /// the Hadamard creates |+>, the Rz rotation accumulates phase from free evolution,
    /// and the final Hadamard converts phase to population difference for measurement.
    ///
    /// The output probability P(|1>) = sin^2(phi/2) = (1 - cos(phi))/2 where phi is
    /// the accumulated phase. Sensitive to magnetic fields, frequency shifts, and
    /// other perturbations that cause phase accumulation.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.ramseySequence(evolutionPhase: .pi / 4)
    /// let state = circuit.execute()
    /// let p1 = state.probability(of: 1)
    /// let inferredPhase = 2.0 * asin(sqrt(p1))
    /// ```
    ///
    /// - Parameters:
    ///   - evolutionPhase: Phase accumulated during free evolution (radians)
    ///   - initialPhase: Optional initial phase offset (default: 0)
    /// - Returns: Single-qubit Ramsey circuit
    /// - Complexity: O(1) - constant 3 gates
    ///
    /// - SeeAlso: ``ramseyEcho(evolutionPhase:echoPosition:)``
    @_eagerMove
    @_effects(readonly)
    static func ramseySequence(evolutionPhase: Double, initialPhase: Double = 0) -> QuantumCircuit {
        var circuit = QuantumCircuit(qubits: 1)

        if abs(initialPhase) > phaseEpsilon {
            circuit.append(.rotationZ(initialPhase), to: 0)
        }

        circuit.append(.hadamard, to: 0)

        circuit.append(.rotationZ(evolutionPhase), to: 0)

        circuit.append(.hadamard, to: 0)

        return circuit
    }

    /// Creates Ramsey echo (Hahn echo) circuit for T2 measurement.
    ///
    /// Implements spin echo sequence that refocuses static inhomogeneous dephasing.
    /// Structure: H - Rz(phi/2) - X - Rz(phi/2) - H where the X gate (pi pulse) at
    /// echoPosition refocuses phase accumulated from static field variations.
    ///
    /// This sequence measures T2 (homogeneous dephasing) rather than T2* (inhomogeneous),
    /// as the echo cancels reversible dephasing while irreversible processes remain.
    /// The echo position (default 0.5) determines where in the evolution the refocusing
    /// pulse is applied.
    ///
    /// **Example:**
    /// ```swift
    /// let echo = QuantumCircuit.ramseyEcho(evolutionPhase: .pi / 2, echoPosition: 0.5)
    /// let simulator = QuantumSimulator()
    /// let state = await simulator.execute(echo)
    /// let p1 = state.probability(of: 1)
    /// ```
    ///
    /// - Parameters:
    ///   - evolutionPhase: Total phase accumulated during free evolution (radians)
    ///   - echoPosition: Relative position of echo pulse in [0, 1] (default: 0.5)
    /// - Returns: Single-qubit Ramsey echo circuit
    /// - Precondition: echoPosition must be in [0, 1]
    /// - Complexity: O(1) - constant 5 gates
    ///
    /// - SeeAlso: ``ramseySequence(evolutionPhase:initialPhase:)``
    @_eagerMove
    @_effects(readonly)
    static func ramseyEcho(evolutionPhase: Double, echoPosition: Double = 0.5) -> QuantumCircuit {
        ValidationUtilities.validateHalfOpenRange(echoPosition, min: 0.0, max: 1.0 + phaseEpsilon, name: "echoPosition")

        var circuit = QuantumCircuit(qubits: 1)

        circuit.append(.hadamard, to: 0)

        let preEchoPhase = evolutionPhase * echoPosition
        if abs(preEchoPhase) > phaseEpsilon {
            circuit.append(.rotationZ(preEchoPhase), to: 0)
        }

        circuit.append(.pauliX, to: 0)

        let postEchoPhase = evolutionPhase * (1.0 - echoPosition)
        if abs(postEchoPhase) > phaseEpsilon {
            circuit.append(.rotationZ(postEchoPhase), to: 0)
        }

        circuit.append(.hadamard, to: 0)

        return circuit
    }
}

public extension QuantumState {
    /// Extracts Ramsey interferometry result from quantum state measurement.
    ///
    /// Computes phase accumulation, fringe visibility, and optionally T2* from the
    /// quantum state after Ramsey sequence execution. Uses the excited state probability
    /// P(|1>) = sin^2(phi/2) to infer accumulated phase and coherence metrics.
    ///
    /// Visibility is estimated as V = |2*P(|1>) - 1| for single-shot analysis.
    /// For proper T2* estimation, multiple measurements at different evolution times
    /// should be performed and fit to exponential decay.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.ramseySequence(evolutionPhase: 1e-6 * frequency)
    /// let state = circuit.execute()
    /// let result = state.ramseyResult(evolutionTime: 1e-6)
    /// print("Phase: \(result.phaseAccumulation) rad")
    /// ```
    ///
    /// - Parameter evolutionTime: Free evolution time for T2* estimation (must be non-negative)
    /// - Returns: RamseyResult with phase, visibility, and coherence metrics
    /// - Precondition: State must be single-qubit (qubits == 1)
    /// - Precondition: evolutionTime must be non-negative
    /// - Complexity: O(1)
    ///
    /// - SeeAlso: ``RamseyResult``
    /// - SeeAlso: ``QuantumCircuit/ramseySequence(evolutionPhase:initialPhase:)``
    @_effects(readonly)
    @_eagerMove
    func ramseyResult(evolutionTime: Double) -> RamseyResult {
        ValidationUtilities.validateStateQubitCount(self, required: 1, exact: true)
        ValidationUtilities.validateNonNegativeDouble(evolutionTime, name: "evolutionTime")

        let p1 = probability(of: 1)
        let p0 = probability(of: 0)

        let phaseAccumulation = acos(max(-1.0, min(1.0, 1.0 - 2.0 * p1)))

        let visibility = abs(p0 - p1)

        var estimatedT2Star: Double?
        if evolutionTime > normalizationEpsilon, visibility > phaseEpsilon, visibility < 1.0 - phaseEpsilon {
            estimatedT2Star = -evolutionTime / log(visibility)
        }

        return RamseyResult(
            phaseAccumulation: phaseAccumulation,
            visibility: visibility,
            estimatedT2Star: estimatedT2Star,
            excitedStateProbability: p1,
        )
    }
}
