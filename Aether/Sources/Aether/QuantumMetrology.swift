// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

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
/// - SeeAlso: ``BayesianPhaseResult``
@frozen
public enum PhasePrior: Sendable {
    case uniform
    case gaussian(mean: Double, stdDev: Double)
    case custom([Double])
}

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
public struct BayesianPhaseResult: Sendable, CustomStringConvertible {
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
        BayesianPhaseResult:
          MAP Estimate: \(String(format: "%.6f", mapEstimate)) rad
          Posterior Mean: \(String(format: "%.6f", posteriorMean)) rad
          Posterior StdDev: \(String(format: "%.6f", posteriorStdDev)) rad
          Measurements: \(measurementCount)
        """
    }
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
/// - SeeAlso: ``BayesianPhaseResult``
/// - SeeAlso: ``QuantumCircuit/phaseEstimation(unitary:precisionQubits:eigenstateQubits:)``
public actor BayesianPhaseEstimation {
    private let unitary: QuantumGate
    private let discretizationBins: Int
    private var posterior: [Double]
    private let phaseGrid: [Double]
    private let deltaPhase: Double

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
        deltaPhase = 2.0 * .pi / Double(discretizationBins)

        let grid = [Double](unsafeUninitializedCapacity: discretizationBins) { buffer, count in
            for i in 0 ..< discretizationBins {
                buffer[i] = Double(i) * 2.0 * .pi / Double(discretizationBins)
            }
            count = discretizationBins
        }
        phaseGrid = grid

        switch prior {
        case .uniform:
            let uniformValue = 1.0 / Double(discretizationBins)
            posterior = [Double](repeating: uniformValue, count: discretizationBins)

        case let .gaussian(mean, stdDev):
            ValidationUtilities.validatePositiveDouble(stdDev, name: "stdDev")
            var priorValues = [Double](unsafeUninitializedCapacity: discretizationBins) { buffer, count in
                let normFactor = 1.0 / (stdDev * sqrt(2.0 * .pi))
                for i in 0 ..< discretizationBins {
                    let diff = grid[i] - mean
                    buffer[i] = normFactor * exp(-0.5 * diff * diff / (stdDev * stdDev))
                }
                count = discretizationBins
            }
            let sum = priorValues.reduce(0.0, +)
            if sum > 1e-15 {
                let invSum = 1.0 / sum
                for i in 0 ..< discretizationBins {
                    priorValues[i] *= invSum
                }
            }
            posterior = priorValues

        case let .custom(pdf):
            ValidationUtilities.validateNonEmpty(pdf, name: "custom PDF")
            var normalized = pdf
            let sum = pdf.reduce(0.0, +)
            if sum > 1e-15 {
                let invSum = 1.0 / sum
                for i in 0 ..< normalized.count {
                    normalized[i] *= invSum
                }
            }
            if normalized.count < discretizationBins {
                let padValue = 1.0 / Double(discretizationBins)
                while normalized.count < discretizationBins {
                    normalized.append(padValue)
                }
                let newSum = normalized.reduce(0.0, +)
                let invNewSum = 1.0 / newSum
                for i in 0 ..< normalized.count {
                    normalized[i] *= invNewSum
                }
            } else if normalized.count > discretizationBins {
                normalized = Array(normalized.prefix(discretizationBins))
                let newSum = normalized.reduce(0.0, +)
                let invNewSum = 1.0 / newSum
                for i in 0 ..< normalized.count {
                    normalized[i] *= invNewSum
                }
            }
            posterior = normalized
        }
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
    /// - Returns: BayesianPhaseResult with posterior statistics and estimates
    /// - Precondition: maxMeasurements must be positive
    /// - Complexity: O(maxMeasurements * discretizationBins)
    @_optimize(speed)
    public func run(
        maxMeasurements: Int,
        targetPrecision: Double? = nil,
        progress: (@Sendable (Int, Double) async -> Void)? = nil,
    ) async -> BayesianPhaseResult {
        ValidationUtilities.validatePositiveInt(maxMeasurements, name: "maxMeasurements")
        if let precision = targetPrecision {
            ValidationUtilities.validatePositiveDouble(precision, name: "targetPrecision")
        }

        var measurementCount = 0
        let simulator = QuantumSimulator()

        for iteration in 0 ..< maxMeasurements {
            let currentStdDev = computePosteriorStdDev()

            if let target = targetPrecision, currentStdDev <= target {
                await progress?(iteration, currentStdDev)
                break
            }

            let adaptiveK = selectAdaptiveK()

            let outcome = await performMeasurement(k: adaptiveK, simulator: simulator)

            updatePosterior(outcome: outcome, k: adaptiveK)

            measurementCount = iteration + 1

            await progress?(measurementCount, computePosteriorStdDev())
        }

        return buildResult(measurementCount: measurementCount)
    }

    @_optimize(speed)
    @_effects(readonly)
    private func selectAdaptiveK() -> Int {
        let stdDev = computePosteriorStdDev()
        guard stdDev > 1e-10 else { return 1 }
        let optimalK = Int(round(1.0 / (2.0 * stdDev)))
        return max(1, min(optimalK, 64))
    }

    @_optimize(speed)
    private func performMeasurement(k: Int, simulator: QuantumSimulator) async -> Int {
        var circuit = QuantumCircuit(qubits: 2)

        circuit.append(.hadamard, to: 0)

        let phaseAngle = extractPhaseFromUnitary()
        circuit.append(.controlledPhase(Double(k) * phaseAngle), to: [0, 1])

        circuit.append(.hadamard, to: 0)

        let state = await simulator.execute(circuit)

        let random = Double.random(in: 0.0 ..< 1.0)
        let (p0, _) = state.probabilities(for: 0)

        return random < p0 ? 0 : 1
    }

    @_effects(readonly)
    private func extractPhaseFromUnitary() -> Double {
        let matrix = unitary.matrix()
        let eigenvalue1 = matrix[0][0]
        let phase1 = atan2(eigenvalue1.imaginary, eigenvalue1.real)
        return phase1
    }

    @_optimize(speed)
    private func updatePosterior(outcome: Int, k: Int) {
        var newPosterior = [Double](unsafeUninitializedCapacity: discretizationBins) { buffer, count in
            for i in 0 ..< discretizationBins {
                let phi = phaseGrid[i]
                let likelihood: Double
                if outcome == 0 {
                    let cosVal = cos(Double(k) * phi / 2.0)
                    likelihood = cosVal * cosVal
                } else {
                    let sinVal = sin(Double(k) * phi / 2.0)
                    likelihood = sinVal * sinVal
                }
                buffer[i] = likelihood * posterior[i]
            }
            count = discretizationBins
        }

        let normalization = newPosterior.reduce(0.0, +)
        if normalization > 1e-15 {
            let invNorm = 1.0 / normalization
            for i in 0 ..< discretizationBins {
                newPosterior[i] *= invNorm
            }
        }

        posterior = newPosterior
    }

    @_effects(readonly)
    private func computePosteriorMean() -> Double {
        var sumSin = 0.0
        var sumCos = 0.0
        for i in 0 ..< discretizationBins {
            let phi = phaseGrid[i]
            sumSin += posterior[i] * sin(phi)
            sumCos += posterior[i] * cos(phi)
        }
        return atan2(sumSin, sumCos)
    }

    @_effects(readonly)
    private func computePosteriorStdDev() -> Double {
        let mean = computePosteriorMean()
        var variance = 0.0
        for i in 0 ..< discretizationBins {
            var diff = phaseGrid[i] - mean
            if diff > .pi { diff -= 2.0 * .pi }
            variance += posterior[i] * diff * diff
        }
        return sqrt(max(0.0, variance))
    }

    @_effects(readonly)
    private func findMAP() -> Double {
        var maxIndex = 0
        var maxValue = posterior[0]
        for i in 1 ..< discretizationBins {
            if posterior[i] > maxValue {
                maxValue = posterior[i]
                maxIndex = i
            }
        }
        return phaseGrid[maxIndex]
    }

    @_effects(readonly)
    private func buildResult(measurementCount: Int) -> BayesianPhaseResult {
        BayesianPhaseResult(
            mapEstimate: findMAP(),
            posteriorMean: computePosteriorMean(),
            posteriorStdDev: computePosteriorStdDev(),
            posterior: posterior,
            measurementCount: measurementCount,
        )
    }
}

/// Configuration parameters for Ramsey interferometry experiments.
///
/// Specifies evolution time, number of repetitions for statistical averaging,
/// and optional detuning from resonance. Used with QuantumCircuit.ramseySequence
/// and QuantumState.ramseyResult for quantum sensing applications.
///
/// **Example:**
/// ```swift
/// let config = RamseyConfig(evolutionTime: 1e-6, repetitions: 1000, detuning: 0.1)
/// ```
///
/// - SeeAlso: ``RamseyResult``
/// - SeeAlso: ``QuantumCircuit/ramseySequence(evolutionPhase:initialPhase:)``
@frozen
public struct RamseyConfig: Sendable {
    /// Free evolution time between Hadamard pulses (in seconds or natural units).
    public let evolutionTime: Double

    /// Number of experimental repetitions for statistical averaging.
    public let repetitions: Int

    /// Frequency detuning from resonance (in radians per unit time).
    public let detuning: Double

    /// Creates Ramsey interferometry configuration.
    ///
    /// **Example:**
    /// ```swift
    /// let config = RamseyConfig(evolutionTime: 1e-6, repetitions: 100)
    /// let detuned = RamseyConfig(evolutionTime: 5e-6, repetitions: 500, detuning: 0.05)
    /// ```
    ///
    /// - Parameters:
    ///   - evolutionTime: Free evolution time (must be non-negative)
    ///   - repetitions: Number of measurement repetitions (default: 100)
    ///   - detuning: Frequency detuning from resonance (default: 0)
    /// - Precondition: evolutionTime must be non-negative
    /// - Precondition: repetitions must be positive
    public init(evolutionTime: Double, repetitions: Int = 100, detuning: Double = 0) {
        ValidationUtilities.validateNonNegativeDouble(evolutionTime, name: "evolutionTime")
        ValidationUtilities.validatePositiveInt(repetitions, name: "repetitions")

        self.evolutionTime = evolutionTime
        self.repetitions = repetitions
        self.detuning = detuning
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
/// let result = state.ramseyResult(config: config)
/// print("Phase: \(result.phaseAccumulation)")
/// print("Visibility: \(result.visibility)")
/// if let t2star = result.estimatedT2Star {
///     print("T2*: \(t2star)")
/// }
/// ```
///
/// - SeeAlso: ``RamseyConfig``
/// - SeeAlso: ``QuantumState/ramseyResult(config:)``
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
    /// - SeeAlso: ``RamseyConfig``
    @_eagerMove
    static func ramseySequence(evolutionPhase: Double, initialPhase: Double = 0) -> QuantumCircuit {
        var circuit = QuantumCircuit(qubits: 1)

        if abs(initialPhase) > 1e-10 {
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
    /// let state = echo.execute()
    /// let visibility = computeVisibility(state)
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
    static func ramseyEcho(evolutionPhase: Double, echoPosition: Double = 0.5) -> QuantumCircuit {
        ValidationUtilities.validateHalfOpenRange(echoPosition, min: 0.0, max: 1.0 + 1e-10, name: "echoPosition")

        var circuit = QuantumCircuit(qubits: 1)

        circuit.append(.hadamard, to: 0)

        let preEchoPhase = evolutionPhase * echoPosition
        if abs(preEchoPhase) > 1e-10 {
            circuit.append(.rotationZ(preEchoPhase), to: 0)
        }

        circuit.append(.pauliX, to: 0)

        let postEchoPhase = evolutionPhase * (1.0 - echoPosition)
        if abs(postEchoPhase) > 1e-10 {
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
    /// let config = RamseyConfig(evolutionTime: 1e-6, repetitions: 100)
    /// let circuit = QuantumCircuit.ramseySequence(evolutionPhase: config.evolutionTime * frequency)
    /// let state = circuit.execute()
    /// let result = state.ramseyResult(config: config)
    /// print("Phase: \(result.phaseAccumulation) rad")
    /// ```
    ///
    /// - Parameter config: Ramsey configuration with timing parameters
    /// - Returns: RamseyResult with phase, visibility, and coherence metrics
    /// - Precondition: State must be single-qubit (qubits == 1)
    /// - Complexity: O(1)
    ///
    /// - SeeAlso: ``RamseyConfig``
    /// - SeeAlso: ``RamseyResult``
    /// - SeeAlso: ``QuantumCircuit/ramseySequence(evolutionPhase:initialPhase:)``
    @_effects(readonly)
    func ramseyResult(config: RamseyConfig) -> RamseyResult {
        ValidationUtilities.validateStateQubitCount(self, required: 1, exact: true)

        let p1 = probability(of: 1)
        let p0 = probability(of: 0)

        let phaseAccumulation = 2.0 * asin(sqrt(max(0.0, min(1.0, p1))))

        let visibility = abs(p0 - p1)

        var estimatedT2Star: Double? = nil
        if config.evolutionTime > 1e-15, visibility > 1e-10, visibility < 1.0 - 1e-10 {
            estimatedT2Star = -config.evolutionTime / log(visibility)
        }

        return RamseyResult(
            phaseAccumulation: phaseAccumulation,
            visibility: visibility,
            estimatedT2Star: estimatedT2Star,
            excitedStateProbability: p1,
        )
    }
}
