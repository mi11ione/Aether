// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate
import Foundation

/// Strategy selection for unified amplitude estimation.
///
/// Controls which underlying algorithm is used for estimating the amplitude of the good
/// subspace component. Each strategy has different trade-offs between ancilla qubit
/// requirements, circuit depth, and estimation precision.
///
/// **Example:**
/// ```swift
/// let standardStrategy = AmplitudeEstimationStrategy.standard
/// let mleStrategy = AmplitudeEstimationStrategy.maximumLikelihood(shots: 100)
/// let autoStrategy = AmplitudeEstimationStrategy.automatic
/// ```
///
/// - SeeAlso: ``UnifiedAmplitudeEstimation``
@frozen
public enum AmplitudeEstimationStrategy: Sendable {
    /// Standard QPE-based amplitude estimation requiring O(log(1/ε)) ancilla qubits.
    ///
    /// Uses quantum phase estimation on the Grover operator to extract the amplitude
    /// with precision determined by the number of precision qubits. Provides deterministic
    /// precision guarantees but requires the full ancilla register.
    case standard

    /// Iterative amplitude estimation using only 1 ancilla qubit.
    ///
    /// Extracts the amplitude one bit at a time via repeated single-ancilla circuits.
    /// Achieves the same O(1/ε) oracle call complexity as standard AE but with minimal
    /// ancilla overhead. Recommended when ancilla budget is limited.
    case iterative

    /// Maximum likelihood amplitude estimation with tighter confidence intervals.
    ///
    /// Runs iterative-style circuits at exponentially increasing Grover operator powers,
    /// collects measurement probabilities, and uses classical maximum likelihood estimation
    /// to find the angle θ that best explains all observations. Produces tighter confidence
    /// intervals than raw iterative extraction. The shots parameter controls how many
    /// distinct Grover powers to sample.
    case maximumLikelihood(shots: Int)

    /// Automatic strategy selection based on available qubits and target precision.
    ///
    /// Selects the optimal strategy: standard if sufficient ancilla qubits are available
    /// for the target precision, iterative if ancilla-constrained, or maximum likelihood
    /// for the best confidence intervals with limited iterations.
    case automatic
}

/// Result of unified amplitude estimation.
///
/// Contains the estimated amplitude sqrt(a), probability a, confidence interval bounds,
/// total oracle call count, and the strategy that was actually used (relevant when
/// automatic selection is enabled).
///
/// **Example:**
/// ```swift
/// let ae = UnifiedAmplitudeEstimation(oracle: oracle)
/// let result = await ae.estimate(precisionQubits: 8)
/// print(result.estimatedAmplitude)
/// print(result.confidenceInterval)
/// print(result.strategyUsed)
/// ```
///
/// - SeeAlso: ``UnifiedAmplitudeEstimation``
/// - SeeAlso: ``AmplitudeEstimationStrategy``
@frozen
public struct UnifiedAmplitudeEstimationResult: Sendable, CustomStringConvertible {
    /// Estimated amplitude sqrt(a) where a is the probability of measuring a good state.
    public let estimatedAmplitude: Double

    /// Estimated probability a = amplitude squared.
    public let estimatedProbability: Double

    /// Confidence interval for the estimated probability.
    ///
    /// Lower and upper bounds computed from the estimation precision. For MLE strategy,
    /// these bounds are typically tighter than standard or iterative approaches.
    public let confidenceInterval: (lower: Double, upper: Double)

    /// Total number of oracle (Grover operator) calls made.
    public let oracleCalls: Int

    /// Strategy that was actually used for this estimation.
    ///
    /// When automatic selection is used, this field reveals which strategy was chosen.
    public let strategyUsed: AmplitudeEstimationStrategy

    /// Creates a unified amplitude estimation result.
    ///
    /// **Example:**
    /// ```swift
    /// let result = UnifiedAmplitudeEstimationResult(
    ///     estimatedAmplitude: 0.5, estimatedProbability: 0.25,
    ///     confidenceInterval: (lower: 0.20, upper: 0.30),
    ///     oracleCalls: 128, strategyUsed: .standard
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - estimatedAmplitude: Estimated amplitude sqrt(a)
    ///   - estimatedProbability: Estimated probability a
    ///   - confidenceInterval: Lower and upper probability bounds
    ///   - oracleCalls: Number of oracle invocations
    ///   - strategyUsed: Strategy that produced this result
    /// - Complexity: O(1)
    @inlinable
    public init(
        estimatedAmplitude: Double,
        estimatedProbability: Double,
        confidenceInterval: (lower: Double, upper: Double),
        oracleCalls: Int,
        strategyUsed: AmplitudeEstimationStrategy,
    ) {
        self.estimatedAmplitude = estimatedAmplitude
        self.estimatedProbability = estimatedProbability
        self.confidenceInterval = confidenceInterval
        self.oracleCalls = oracleCalls
        self.strategyUsed = strategyUsed
    }

    /// Formatted description of the estimation result.
    ///
    /// **Example:**
    /// ```swift
    /// let result = UnifiedAmplitudeEstimationResult(
    ///     estimatedAmplitude: 0.5, estimatedProbability: 0.25,
    ///     confidenceInterval: (lower: 0.2, upper: 0.3),
    ///     oracleCalls: 64, strategyUsed: .iterative
    /// )
    /// print(result)
    /// ```
    @inlinable
    public var description: String {
        let strategyStr = switch strategyUsed {
        case .standard: "Standard QPE"
        case .iterative: "Iterative"
        case let .maximumLikelihood(shots): "MLE(\(shots) shots)"
        case .automatic: "Automatic"
        }
        return """
        UnifiedAmplitudeEstimation Result:
          Strategy: \(strategyStr)
          Amplitude: \(String(format: "%.8f", estimatedAmplitude))
          Probability: \(String(format: "%.8f", estimatedProbability))
          CI: [\(String(format: "%.8f", confidenceInterval.lower)), \(String(format: "%.8f", confidenceInterval.upper))]
          Oracle Calls: \(oracleCalls)
        """
    }
}

/// Unified amplitude estimation with automatic strategy selection.
///
/// Provides a single interface for estimating the amplitude sqrt(a) from an oracle
/// A|0⟩ = sqrt(1-a)|bad⟩ + sqrt(a)|good⟩ using the optimal strategy for the given
/// resource constraints. Supports standard QPE-based, iterative, and maximum likelihood
/// estimation strategies.
///
/// Standard QPE-based estimation uses O(1/ε) oracle calls with O(log(1/ε)) ancilla qubits.
/// Iterative estimation achieves the same oracle complexity with only 1 ancilla qubit.
/// Maximum likelihood estimation runs circuits at multiple Grover powers and performs
/// classical MLE post-processing for tighter confidence intervals.
///
/// **Example:**
/// ```swift
/// let oracle = CountingOracle(qubits: 4, markedStates: [3, 7, 11])
/// let ae = UnifiedAmplitudeEstimation(oracle: oracle)
/// let result = await ae.estimate(precisionQubits: 8)
/// print(result.estimatedAmplitude)
/// print(result.confidenceInterval)
/// ```
///
/// - Complexity: O(2^precisionQubits) oracle calls for standard, O(precisionQubits × 2^precisionQubits) for iterative
/// - SeeAlso: ``AmplitudeEstimationStrategy``
/// - SeeAlso: ``UnifiedAmplitudeEstimationResult``
/// - SeeAlso: ``AmplitudeOracle``
/// - SeeAlso: ``GeneralizedAmplification``
public actor UnifiedAmplitudeEstimation {
    private static let epsilonTolerance: Double = 1e-12
    private static let mleSearchTolerance: Double = 1e-10
    private static let mleSearchIterations = 100

    private let oracle: any AmplitudeOracle
    private let simulator: QuantumSimulator

    /// Creates a unified amplitude estimator for the given oracle.
    ///
    /// **Example:**
    /// ```swift
    /// let oracle = CountingOracle(qubits: 4, markedStates: [3, 5, 9])
    /// let ae = UnifiedAmplitudeEstimation(oracle: oracle)
    /// ```
    ///
    /// - Parameter oracle: Amplitude oracle defining state preparation and marking
    /// - Complexity: O(1)
    public init(oracle: any AmplitudeOracle) {
        self.oracle = oracle
        simulator = QuantumSimulator()
    }

    /// Estimates the amplitude using the specified strategy.
    ///
    /// Dispatches to the appropriate estimation algorithm based on the strategy parameter.
    /// When automatic selection is used, chooses the optimal strategy based on available
    /// precision qubits: standard for >= 4 precision qubits, iterative otherwise, with
    /// MLE for 6+ precision qubits for best confidence intervals.
    ///
    /// **Example:**
    /// ```swift
    /// let ae = UnifiedAmplitudeEstimation(oracle: oracle)
    /// let result = await ae.estimate(precisionQubits: 8, strategy: .standard)
    /// print(result.estimatedAmplitude)
    /// ```
    ///
    /// - Parameters:
    ///   - precisionQubits: Number of precision qubits (determines accuracy)
    ///   - strategy: Estimation strategy to use (default: .automatic)
    ///   - progress: Optional callback for status updates
    /// - Returns: Estimation result with amplitude, CI, and oracle count
    /// - Precondition: precisionQubits > 0
    /// - Precondition: precisionQubits <= 15
    /// - Complexity: O(2^precisionQubits) oracle calls
    @_optimize(speed)
    public func estimate(
        precisionQubits: Int,
        strategy: AmplitudeEstimationStrategy = .automatic,
        progress: (@Sendable (String) async -> Void)? = nil,
    ) async -> UnifiedAmplitudeEstimationResult {
        ValidationUtilities.validatePositiveInt(precisionQubits, name: "precisionQubits")
        ValidationUtilities.validateUpperBound(precisionQubits, max: 15, name: "precisionQubits")

        switch strategy {
        case .standard:
            return await runStandard(precisionQubits: precisionQubits, progress: progress)

        case .iterative:
            return await runIterative(precisionQubits: precisionQubits, progress: progress)

        case let .maximumLikelihood(shots):
            return await runMLE(
                shots: shots,
                progress: progress,
            )

        case .automatic:
            return await runAutomatic(precisionQubits: precisionQubits, progress: progress)
        }
    }

    /// Resolves automatic strategy and dispatches to best algorithm.
    @_optimize(speed)
    private func runAutomatic(
        precisionQubits: Int,
        progress: (@Sendable (String) async -> Void)?,
    ) async -> UnifiedAmplitudeEstimationResult {
        if precisionQubits >= 6 {
            await runMLE(
                shots: precisionQubits,
                progress: progress,
            )
        } else if precisionQubits >= 4 {
            await runStandard(precisionQubits: precisionQubits, progress: progress)
        } else {
            await runIterative(precisionQubits: precisionQubits, progress: progress)
        }
    }

    /// Runs standard QPE-based amplitude estimation.
    @_optimize(speed)
    @_eagerMove
    private func runStandard(
        precisionQubits: Int,
        progress: (@Sendable (String) async -> Void)?,
    ) async -> UnifiedAmplitudeEstimationResult {
        await progress?("Standard AE: building Grover operator")

        let n = precisionQubits
        let stateQubits = oracle.qubits
        let totalQubits = n + stateQubits

        var circuit = QuantumCircuit(qubits: totalQubits)

        for qubit in 0 ..< n {
            circuit.append(.hadamard, to: qubit)
        }

        var stateCircuit = QuantumCircuit(qubits: stateQubits)
        oracle.applyStatePreparation(to: &stateCircuit)
        appendShiftedOperations(from: stateCircuit, to: &circuit, offset: n)

        for controlQubit in 0 ..< n {
            let power = 1 << (n - 1 - controlQubit)
            await progress?("Standard AE: controlled-Q^(\(power)) from qubit \(controlQubit)")
            applyControlledGroverPower(
                to: &circuit,
                controlQubit: controlQubit,
                power: power,
                stateQubitOffset: n,
            )
        }

        await progress?("Standard AE: applying inverse QFT")
        applyInverseQFT(to: &circuit, qubits: n)

        let state = await simulator.execute(circuit)

        let precisionStateSize = 1 << n
        var precisionProbabilities = [Double](unsafeUninitializedCapacity: precisionStateSize) {
            buffer, count in
            buffer.initialize(repeating: 0.0)
            count = precisionStateSize
        }

        for basisIndex in 0 ..< state.stateSpaceSize {
            let precisionIndex = basisIndex % precisionStateSize
            precisionProbabilities[precisionIndex] += state.amplitudes[basisIndex].magnitudeSquared
        }

        var maxValue = 0.0
        var maxIdx: vDSP_Length = 0
        vDSP_maxviD(precisionProbabilities, 1, &maxValue, &maxIdx, vDSP_Length(precisionStateSize))
        let maxIndex = Int(maxIdx)

        let estimatedPhase = Double(maxIndex) / Double(precisionStateSize)
        let theta = estimatedPhase * .pi
        let amplitude = abs(Foundation.sin(theta))
        let probability = Foundation.sin(theta) * Foundation.sin(theta)

        let epsilon = .pi / Double(precisionStateSize)
        let thetaLower = max(0.0, theta - epsilon)
        let thetaUpper = min(.pi, theta + epsilon)
        let probLower = Foundation.sin(thetaLower) * Foundation.sin(thetaLower)
        let probUpper = Foundation.sin(thetaUpper) * Foundation.sin(thetaUpper)
        let ci = (lower: min(probLower, probUpper), upper: max(probLower, probUpper))

        let oracleCalls = 2 * ((1 << n) - 1)

        await progress?(
            "Standard AE complete: amplitude = \(String(format: "%.6f", amplitude))",
        )

        return UnifiedAmplitudeEstimationResult(
            estimatedAmplitude: amplitude,
            estimatedProbability: probability,
            confidenceInterval: ci,
            oracleCalls: oracleCalls,
            strategyUsed: .standard,
        )
    }

    /// Runs iterative amplitude estimation with 1 ancilla qubit.
    @_optimize(speed)
    @_eagerMove
    private func runIterative(
        precisionQubits: Int,
        progress: (@Sendable (String) async -> Void)?,
    ) async -> UnifiedAmplitudeEstimationResult {
        await progress?("Iterative AE: starting \(precisionQubits)-bit extraction")

        let n = precisionQubits
        let stateQubits = oracle.qubits
        let totalQubits = 1 + stateQubits

        var measuredBits: [Int] = []
        measuredBits.reserveCapacity(n)
        var totalOracleCalls = 0

        for k in 0 ..< n {
            let power = 1 << (n - 1 - k)
            await progress?("Iterative AE: bit \(k + 1)/\(n), Q^\(power)")

            var circuit = QuantumCircuit(qubits: totalQubits)
            circuit.append(.hadamard, to: 0)

            var stateCircuit = QuantumCircuit(qubits: stateQubits)
            oracle.applyStatePreparation(to: &stateCircuit)
            appendShiftedOperations(from: stateCircuit, to: &circuit, offset: 1)

            applyControlledGroverPower(
                to: &circuit,
                controlQubit: 0,
                power: power,
                stateQubitOffset: 1,
            )

            var correction = 0.0
            for j in 0 ..< measuredBits.count {
                let exponent = k - j
                correction += Double(measuredBits[j]) * .pi / Double(1 << exponent)
            }

            if abs(correction) > Self.epsilonTolerance {
                circuit.append(.rotationZ(-correction), to: 0)
            }

            circuit.append(.hadamard, to: 0)

            let state = await simulator.execute(circuit)

            var prob0 = 0.0
            for i in stride(from: 0, to: state.stateSpaceSize, by: 2) {
                prob0 += state.amplitudes[i].magnitudeSquared
            }

            let measuredBit = prob0 >= 0.5 ? 0 : 1
            measuredBits.append(measuredBit)
            totalOracleCalls += 2 * power
        }

        var phase = 0.0
        var reciprocal = 0.5
        for bit in measuredBits {
            phase += Double(bit) * reciprocal
            reciprocal *= 0.5
        }

        let theta = phase * .pi
        let amplitude = abs(Foundation.sin(theta))
        let probability = Foundation.sin(theta) * Foundation.sin(theta)

        let epsilon = .pi / Double(1 << n)
        let thetaLower = max(0.0, theta - epsilon)
        let thetaUpper = min(.pi, theta + epsilon)
        let probLower = Foundation.sin(thetaLower) * Foundation.sin(thetaLower)
        let probUpper = Foundation.sin(thetaUpper) * Foundation.sin(thetaUpper)
        let ci = (lower: min(probLower, probUpper), upper: max(probLower, probUpper))

        await progress?(
            "Iterative AE complete: amplitude = \(String(format: "%.6f", amplitude))",
        )

        return UnifiedAmplitudeEstimationResult(
            estimatedAmplitude: amplitude,
            estimatedProbability: probability,
            confidenceInterval: ci,
            oracleCalls: totalOracleCalls,
            strategyUsed: .iterative,
        )
    }

    /// Runs maximum likelihood amplitude estimation.
    @_optimize(speed)
    @_eagerMove
    private func runMLE(
        shots: Int,
        progress: (@Sendable (String) async -> Void)?,
    ) async -> UnifiedAmplitudeEstimationResult {
        await progress?("MLE AE: collecting measurement data at \(shots) Grover powers")

        let stateQubits = oracle.qubits
        let totalQubits = 1 + stateQubits

        var groverPowers: [Int] = []
        groverPowers.reserveCapacity(shots)
        var observedProbabilities: [Double] = []
        observedProbabilities.reserveCapacity(shots)
        var totalOracleCalls = 0

        for k in 0 ..< shots {
            let power = 1 << k

            var circuit = QuantumCircuit(qubits: totalQubits)
            circuit.append(.hadamard, to: 0)

            var stateCircuit = QuantumCircuit(qubits: stateQubits)
            oracle.applyStatePreparation(to: &stateCircuit)
            appendShiftedOperations(from: stateCircuit, to: &circuit, offset: 1)

            applyControlledGroverPower(
                to: &circuit,
                controlQubit: 0,
                power: power,
                stateQubitOffset: 1,
            )

            circuit.append(.hadamard, to: 0)

            let state = await simulator.execute(circuit)

            var prob0 = 0.0
            for i in stride(from: 0, to: state.stateSpaceSize, by: 2) {
                prob0 += state.amplitudes[i].magnitudeSquared
            }

            groverPowers.append(power)
            observedProbabilities.append(prob0)
            totalOracleCalls += 2 * power

            await progress?(
                "MLE AE: power \(power), P(0) = \(String(format: "%.6f", prob0))",
            )
        }

        await progress?("MLE AE: performing maximum likelihood optimization")

        let mleTheta = performMLEOptimization(
            groverPowers: groverPowers,
            observedProbabilities: observedProbabilities,
        )

        let amplitude = abs(Foundation.sin(mleTheta))
        let probability = Foundation.sin(mleTheta) * Foundation.sin(mleTheta)

        let mleCI = computeMLEConfidenceInterval(
            theta: mleTheta,
            groverPowers: groverPowers,
        )

        await progress?(
            "MLE AE complete: amplitude = \(String(format: "%.6f", amplitude))",
        )

        return UnifiedAmplitudeEstimationResult(
            estimatedAmplitude: amplitude,
            estimatedProbability: probability,
            confidenceInterval: mleCI,
            oracleCalls: totalOracleCalls,
            strategyUsed: .maximumLikelihood(shots: shots),
        )
    }

    /// Finds the MLE angle θ via golden section search.
    @_optimize(speed)
    @_effects(readonly)
    private func performMLEOptimization(
        groverPowers: [Int],
        observedProbabilities: [Double],
    ) -> Double {
        let goldenRatio = (Foundation.sqrt(5.0) - 1.0) * 0.5

        var lo = Self.mleSearchTolerance
        var hi = .pi * 0.5 - Self.mleSearchTolerance
        var x1 = hi - goldenRatio * (hi - lo)
        var x2 = lo + goldenRatio * (hi - lo)

        var f1 = negativeLogLikelihood(
            theta: x1,
            groverPowers: groverPowers,
            observedProbabilities: observedProbabilities,
        )
        var f2 = negativeLogLikelihood(
            theta: x2,
            groverPowers: groverPowers,
            observedProbabilities: observedProbabilities,
        )

        for _ in 0 ..< Self.mleSearchIterations {
            if hi - lo < Self.mleSearchTolerance { break }

            if f1 < f2 {
                hi = x2
                x2 = x1
                f2 = f1
                x1 = hi - goldenRatio * (hi - lo)
                f1 = negativeLogLikelihood(
                    theta: x1,
                    groverPowers: groverPowers,
                    observedProbabilities: observedProbabilities,
                )
            } else {
                lo = x1
                x1 = x2
                f1 = f2
                x2 = lo + goldenRatio * (hi - lo)
                f2 = negativeLogLikelihood(
                    theta: x2,
                    groverPowers: groverPowers,
                    observedProbabilities: observedProbabilities,
                )
            }
        }

        return (lo + hi) * 0.5
    }

    /// Computes negative log-likelihood for a candidate θ.
    @inline(__always)
    @_optimize(speed)
    @_effects(readonly)
    private func negativeLogLikelihood(
        theta: Double,
        groverPowers: [Int],
        observedProbabilities: [Double],
    ) -> Double {
        var nll = 0.0

        for i in 0 ..< groverPowers.count {
            let m = groverPowers[i]
            let expectedProb0 = Foundation.cos(Double(2 * m + 1) * theta)
            let expectedProb0Sq = expectedProb0 * expectedProb0
            let clampedProb = max(Self.mleSearchTolerance, min(1.0 - Self.mleSearchTolerance, expectedProb0Sq))

            let observedP0 = observedProbabilities[i]
            nll -= observedP0 * Foundation.log(clampedProb)
            nll -= (1.0 - observedP0) * Foundation.log(1.0 - clampedProb)
        }

        return nll
    }

    /// Computes confidence interval from Fisher information of MLE.
    @_optimize(speed)
    @_effects(readonly)
    private func computeMLEConfidenceInterval(
        theta: Double,
        groverPowers: [Int],
    ) -> (lower: Double, upper: Double) {
        var fisherInfo = 0.0

        for i in 0 ..< groverPowers.count {
            let m = groverPowers[i]
            let factor = Double(2 * m + 1)
            let cosVal = Foundation.cos(factor * theta)
            let sinVal = Foundation.sin(factor * theta)
            let prob = max(Self.mleSearchTolerance, cosVal * cosVal)
            let probComplement = max(Self.mleSearchTolerance, 1.0 - cosVal * cosVal)

            let dProbDTheta = -2.0 * cosVal * sinVal * factor
            fisherInfo += (dProbDTheta * dProbDTheta) / (prob * probComplement)
        }

        let safeFisherInfo = max(fisherInfo, 16.0 / (.pi * .pi))
        let standardError = 1.0 / Foundation.sqrt(safeFisherInfo)
        let z95 = 1.96

        let thetaLower = max(0.0, theta - z95 * standardError)
        let thetaUpper = min(.pi * 0.5, theta + z95 * standardError)

        let probLower = Foundation.sin(thetaLower) * Foundation.sin(thetaLower)
        let probUpper = Foundation.sin(thetaUpper) * Foundation.sin(thetaUpper)

        return (lower: min(probLower, probUpper), upper: max(probLower, probUpper))
    }

    /// Applies controlled Q^power where Q is the Grover operator.
    @_optimize(speed)
    private func applyControlledGroverPower(
        to circuit: inout QuantumCircuit,
        controlQubit: Int,
        power: Int,
        stateQubitOffset: Int,
    ) {
        for _ in 0 ..< power {
            applyControlledMarkingOracle(
                to: &circuit,
                controlQubit: controlQubit,
                stateQubitOffset: stateQubitOffset,
            )
            applyControlledDiffusion(
                to: &circuit,
                controlQubit: controlQubit,
                stateQubitOffset: stateQubitOffset,
            )
        }
    }

    /// Applies controlled marking oracle.
    @_optimize(speed)
    private func applyControlledMarkingOracle(
        to circuit: inout QuantumCircuit,
        controlQubit: Int,
        stateQubitOffset: Int,
    ) {
        var oracleCircuit = QuantumCircuit(qubits: oracle.qubits)
        oracle.applyMarkingOracle(to: &oracleCircuit)

        for op in oracleCircuit.operations {
            if case let .gate(g, qubits, _) = op {
                let shiftedQubits = [Int](unsafeUninitializedCapacity: qubits.count) {
                    buffer, count in
                    for i in 0 ..< qubits.count {
                        buffer[i] = qubits[i] + stateQubitOffset
                    }
                    count = qubits.count
                }
                let controlledGate = makeControlled(
                    gate: g, control: controlQubit, targets: shiftedQubits,
                )
                for (cg, cq) in controlledGate {
                    circuit.append(cg, to: cq)
                }
            }
        }
    }

    /// Applies controlled diffusion operator.
    @_optimize(speed)
    private func applyControlledDiffusion(
        to circuit: inout QuantumCircuit,
        controlQubit: Int,
        stateQubitOffset: Int,
    ) {
        let stateQubits = oracle.qubits

        var invCircuit = QuantumCircuit(qubits: stateQubits)
        oracle.applyStatePreparationInverse(to: &invCircuit)
        appendControlledShiftedOperations(
            from: invCircuit, to: &circuit,
            controlQubit: controlQubit, offset: stateQubitOffset,
        )

        applyControlledReflectionAboutZero(
            to: &circuit,
            controlQubit: controlQubit,
            stateQubitOffset: stateQubitOffset,
            stateQubits: stateQubits,
        )

        var prepCircuit = QuantumCircuit(qubits: stateQubits)
        oracle.applyStatePreparation(to: &prepCircuit)
        appendControlledShiftedOperations(
            from: prepCircuit, to: &circuit,
            controlQubit: controlQubit, offset: stateQubitOffset,
        )
    }

    /// Applies controlled reflection about zero state.
    @_optimize(speed)
    private func applyControlledReflectionAboutZero(
        to circuit: inout QuantumCircuit,
        controlQubit: Int,
        stateQubitOffset: Int,
        stateQubits: Int,
    ) {
        for qubit in 0 ..< stateQubits {
            circuit.append(.cnot, to: [controlQubit, stateQubitOffset + qubit])
        }

        if stateQubits == 1 {
            circuit.append(.controlledPhase(.pi), to: [controlQubit, stateQubitOffset])
        } else {
            circuit.append(.hadamard, to: stateQubitOffset + stateQubits - 1)

            let controls = [Int](unsafeUninitializedCapacity: stateQubits) { buffer, count in
                buffer[0] = controlQubit
                for q in 0 ..< stateQubits - 1 {
                    buffer[q + 1] = stateQubitOffset + q
                }
                count = stateQubits
            }
            let target = stateQubitOffset + stateQubits - 1

            let mcxGates = GroverDiffusion.buildMultiControlledXGates(
                controls: controls, target: target,
            )
            for (gate, qubits) in mcxGates {
                circuit.append(gate, to: qubits)
            }

            circuit.append(.hadamard, to: stateQubitOffset + stateQubits - 1)
        }

        for qubit in (0 ..< stateQubits).reversed() {
            circuit.append(.cnot, to: [controlQubit, stateQubitOffset + qubit])
        }
    }

    /// Creates controlled version of a gate.
    @inline(__always)
    @_optimize(speed)
    @_effects(readonly)
    private func makeControlled(
        gate: QuantumGate,
        control: Int,
        targets: [Int],
    ) -> [(gate: QuantumGate, qubits: [Int])] {
        switch gate {
        case .hadamard:
            return [(.ch, [control, targets[0]])]
        case .pauliX:
            return [(.cnot, [control, targets[0]])]
        case .pauliZ:
            return [(.controlledPhase(.value(.pi)), [control, targets[0]])]
        case let .rotationZ(.value(v)):
            let halfAngle = v * 0.5
            return [
                (.rotationZ(.value(halfAngle)), [targets[0]]),
                (.cnot, [control, targets[0]]),
                (.rotationZ(.value(-halfAngle)), [targets[0]]),
                (.cnot, [control, targets[0]]),
            ]
        case let .rotationY(.value(v)):
            let halfAngle = v * 0.5
            return [
                (.rotationY(.value(halfAngle)), [targets[0]]),
                (.cnot, [control, targets[0]]),
                (.rotationY(.value(-halfAngle)), [targets[0]]),
                (.cnot, [control, targets[0]]),
            ]
        case .cnot:
            return [(.toffoli, [control, targets[0], targets[1]])]
        case .toffoli:
            return applyControlledToffoli(control: control, targets: targets)
        case let .phase(.value(v)):
            let halfAngle = v * 0.5
            return [
                (.rotationZ(.value(halfAngle)), [targets[0]]),
                (.cnot, [control, targets[0]]),
                (.rotationZ(.value(-halfAngle)), [targets[0]]),
                (.cnot, [control, targets[0]]),
            ]
        case .sGate:
            return [
                (.rotationZ(.value(.pi / 4.0)), [targets[0]]),
                (.cnot, [control, targets[0]]),
                (.rotationZ(.value(-.pi / 4.0)), [targets[0]]),
                (.cnot, [control, targets[0]]),
            ]
        case .tGate:
            return [
                (.rotationZ(.value(.pi / 8.0)), [targets[0]]),
                (.cnot, [control, targets[0]]),
                (.rotationZ(.value(-.pi / 8.0)), [targets[0]]),
                (.cnot, [control, targets[0]]),
            ]
        case let .controlledPhase(.value(v)):
            return [
                (.rotationZ(.value(v * 0.5)), [targets[0]]),
                (.cnot, [control, targets[0]]),
                (.rotationZ(.value(-v * 0.5)), [targets[0]]),
                (.cnot, [control, targets[0]]),
                (.rotationZ(.value(v * 0.5)), [targets[1]]),
                (.cnot, [control, targets[1]]),
                (.rotationZ(.value(-v * 0.5)), [targets[1]]),
                (.cnot, [control, targets[1]]),
            ]
        default:
            return [(gate, targets)]
        }
    }

    /// Applies controlled Toffoli using ancilla decomposition.
    @inline(__always)
    @_optimize(speed)
    @_effects(readonly)
    private func applyControlledToffoli(
        control: Int,
        targets: [Int],
    ) -> [(gate: QuantumGate, qubits: [Int])] {
        let c1 = targets[0]
        let c2 = targets[1]
        let t = targets[2]
        let tDag = QuantumGate.phase(.value(-.pi / 4.0))

        return [
            (.hadamard, [t]),
            (.cnot, [c2, t]),
            (tDag, [t]),
            (.cnot, [c1, t]),
            (.tGate, [t]),
            (.cnot, [c2, t]),
            (tDag, [t]),
            (.cnot, [control, t]),
            (.tGate, [t]),
            (.cnot, [c2, t]),
            (tDag, [t]),
            (.cnot, [c1, t]),
            (.tGate, [t]),
            (.cnot, [c2, t]),
            (tDag, [t]),
            (.cnot, [control, t]),
            (.tGate, [t]),
            (.hadamard, [t]),
        ]
    }

    /// Appends circuit operations with shifted qubit indices.
    @_optimize(speed)
    private func appendShiftedOperations(
        from sourceCircuit: QuantumCircuit,
        to targetCircuit: inout QuantumCircuit,
        offset: Int,
    ) {
        for op in sourceCircuit.operations {
            if case let .gate(g, qubits, _) = op {
                let shiftedQubits = [Int](unsafeUninitializedCapacity: qubits.count) {
                    buffer, count in
                    for i in 0 ..< qubits.count {
                        buffer[i] = qubits[i] + offset
                    }
                    count = qubits.count
                }
                targetCircuit.append(g, to: shiftedQubits)
            }
        }
    }

    /// Appends controlled operations with shifted qubit indices.
    @_optimize(speed)
    private func appendControlledShiftedOperations(
        from sourceCircuit: QuantumCircuit,
        to targetCircuit: inout QuantumCircuit,
        controlQubit: Int,
        offset: Int,
    ) {
        for op in sourceCircuit.operations {
            if case let .gate(g, qubits, _) = op {
                let shiftedQubits = [Int](unsafeUninitializedCapacity: qubits.count) {
                    buffer, count in
                    for i in 0 ..< qubits.count {
                        buffer[i] = qubits[i] + offset
                    }
                    count = qubits.count
                }
                let controlledGate = makeControlled(
                    gate: g, control: controlQubit, targets: shiftedQubits,
                )
                for (cg, cq) in controlledGate {
                    targetCircuit.append(cg, to: cq)
                }
            }
        }
    }

    /// Applies inverse QFT to precision qubits.
    @_optimize(speed)
    private func applyInverseQFT(to circuit: inout QuantumCircuit, qubits n: Int) {
        for i in 0 ..< n / 2 {
            circuit.append(.swap, to: [i, n - 1 - i])
        }

        let angleTable = [Double](unsafeUninitializedCapacity: n) { buffer, count in
            for i in 0 ..< n {
                buffer[i] = -.pi / Double(1 << i)
            }
            count = n
        }

        for j in 0 ..< n {
            for k in 0 ..< j {
                circuit.append(.controlledPhase(.value(angleTable[j - k])), to: [k, j])
            }
            circuit.append(.hadamard, to: j)
        }
    }
}

/// Generalized amplitude amplification for arbitrary initial distributions.
///
/// Provides utility methods for amplitude amplification that work with any state
/// preparation oracle, not just uniform superpositions. The Grover operator
/// Q = A·S₀·A†·Sχ amplifies the good subspace component regardless of the initial
/// amplitude distribution.
///
/// **Example:**
/// ```swift
/// let oracle = CountingOracle(qubits: 4, markedStates: [3, 7])
/// let iterations = GeneralizedAmplification.optimalIterations(amplitude: 0.25)
/// let circuit = GeneralizedAmplification.amplify(oracle: oracle, iterations: iterations)
/// ```
///
/// - SeeAlso: ``AmplitudeOracle``
/// - SeeAlso: ``UnifiedAmplitudeEstimation``
public enum GeneralizedAmplification {
    /// Computes the optimal number of Grover iterations for a known initial amplitude.
    ///
    /// For initial amplitude sin(θ) = a, the optimal iteration count is
    /// k = ⌊π/(4θ) - 1/2⌋ which maximizes P(good) = sin²((2k+1)θ).
    ///
    /// **Example:**
    /// ```swift
    /// let iterations = GeneralizedAmplification.optimalIterations(amplitude: 0.25)
    /// print(iterations)  // Optimal Grover iterations for amplitude 0.25
    /// ```
    ///
    /// - Parameter amplitude: Known initial amplitude sin(θ) in range (0, 1]
    /// - Returns: Optimal number of Grover iterations
    /// - Precondition: amplitude > 0
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    public static func optimalIterations(amplitude: Double) -> Int {
        ValidationUtilities.validatePositiveDouble(amplitude, name: "amplitude")

        let theta = Foundation.asin(min(1.0, amplitude))
        guard theta > 1e-15 else { return 0 }

        let optimal = (.pi / (4.0 * theta)) - 0.5
        return max(0, Int(Foundation.floor(optimal)))
    }

    /// Builds an amplitude amplification circuit with the specified number of iterations.
    ///
    /// Constructs a circuit that applies k iterations of the Grover operator
    /// Q = A·S₀·A†·Sχ to amplify the good subspace component. After k optimal
    /// iterations, the success probability is sin²((2k+1)θ) ≈ 1.
    ///
    /// **Example:**
    /// ```swift
    /// let oracle = CountingOracle(qubits: 3, markedStates: [5])
    /// let circuit = GeneralizedAmplification.amplify(oracle: oracle, iterations: 2)
    /// let state = circuit.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - oracle: Amplitude oracle defining state preparation and marking
    ///   - iterations: Number of Grover iterations to apply
    /// - Returns: Quantum circuit implementing state preparation followed by amplification
    /// - Precondition: iterations >= 0
    /// - Complexity: O(iterations × circuit_depth)
    @_optimize(speed)
    @_eagerMove
    public static func amplify(
        oracle: any AmplitudeOracle,
        iterations: Int,
    ) -> QuantumCircuit {
        ValidationUtilities.validateNonNegativeInt(iterations, name: "iterations")

        let qubits = oracle.qubits
        var circuit = QuantumCircuit(qubits: qubits)

        oracle.applyStatePreparation(to: &circuit)

        let diffusion = GroverDiffusion(qubits: qubits)

        for _ in 0 ..< iterations {
            oracle.applyMarkingOracle(to: &circuit)

            oracle.applyStatePreparationInverse(to: &circuit)

            for (gate, gateQubits) in diffusion.gates {
                circuit.append(gate, to: gateQubits)
            }

            oracle.applyStatePreparation(to: &circuit)
        }

        return circuit
    }

    /// Computes the success probability after k iterations of amplitude amplification.
    ///
    /// For initial amplitude sin(θ) = a, the success probability after k Grover iterations
    /// is sin²((2k+1)θ), oscillating between 0 and 1.
    ///
    /// **Example:**
    /// ```swift
    /// let prob = GeneralizedAmplification.successProbability(
    ///     initialAmplitude: 0.25, iterations: 3
    /// )
    /// print(prob)
    /// ```
    ///
    /// - Parameters:
    ///   - initialAmplitude: Initial amplitude sin(θ) in range (0, 1]
    ///   - iterations: Number of Grover iterations applied
    /// - Returns: Success probability in [0, 1]
    /// - Precondition: initialAmplitude > 0
    /// - Precondition: iterations >= 0
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    public static func successProbability(initialAmplitude: Double, iterations: Int) -> Double {
        ValidationUtilities.validatePositiveDouble(initialAmplitude, name: "initialAmplitude")
        ValidationUtilities.validateNonNegativeInt(iterations, name: "iterations")

        let theta = Foundation.asin(min(1.0, initialAmplitude))
        let amplifiedAngle = Double(2 * iterations + 1) * theta
        let sinVal = Foundation.sin(amplifiedAngle)
        return sinVal * sinVal
    }
}
