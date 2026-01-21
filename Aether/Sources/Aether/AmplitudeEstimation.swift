// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under Apache 2.0

import Foundation

/// Protocol for state preparation oracle A where A|0> = sqrt(1-a)|bad> + sqrt(a)|good>.
///
/// The amplitude oracle defines a quantum state preparation that separates the Hilbert space
/// into "good" (marked) and "bad" (unmarked) subspaces. The amplitude estimation algorithm
/// estimates sqrt(a), the amplitude of the good subspace component.
///
/// Implementations must provide:
/// - State preparation circuit A
/// - Inverse state preparation A dagger
/// - Marking oracle that applies phase -1 to good states
///
/// **Example:**
/// ```swift
/// struct MyOracle: AmplitudeOracle {
///     var qubits: Int { 3 }
///     func applyStatePreparation(to circuit: inout QuantumCircuit) {
///         for q in 0..<3 { circuit.append(.hadamard, to: q) }
///     }
///     func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
///         for q in (0..<3).reversed() { circuit.append(.hadamard, to: q) }
///     }
///     func applyMarkingOracle(to circuit: inout QuantumCircuit) {
///         // Mark state |5> = |101>
///     }
/// }
/// ```
///
/// - SeeAlso: ``AmplitudeEstimation``
/// - SeeAlso: ``CountingOracle``
/// - SeeAlso: ``AmplitudeEstimationResult``
public protocol AmplitudeOracle: Sendable {
    /// Number of qubits in the state register.
    var qubits: Int { get }

    /// Applies state preparation A that creates the superposition A|0> = sqrt(1-a)|bad> + sqrt(a)|good>.
    ///
    /// - Parameter circuit: Quantum circuit to append gates to
    func applyStatePreparation(to circuit: inout QuantumCircuit)

    /// Applies inverse state preparation A dagger.
    ///
    /// - Parameter circuit: Quantum circuit to append gates to
    func applyStatePreparationInverse(to circuit: inout QuantumCircuit)

    /// Applies marking oracle S_chi that flips phase of good states: I - 2|good><good|.
    ///
    /// - Parameter circuit: Quantum circuit to append gates to
    func applyMarkingOracle(to circuit: inout QuantumCircuit)
}

/// Result of quantum amplitude estimation algorithm.
///
/// Contains the estimated amplitude sqrt(a), probability a, confidence interval,
/// oracle call count, and comparison to classical Monte Carlo sampling requirements.
/// The quadratic speedup is demonstrated by classicalEquivalentSamples requiring
/// O(1/epsilon^2) samples for the same precision achieved with O(1/epsilon) oracle calls.
///
/// **Example:**
/// ```swift
/// let oracle = CountingOracle(qubits: 4, markedStates: [3, 5, 9])
/// let config = AEConfiguration(precisionQubits: 6)
/// let ae = AmplitudeEstimation(oracle: oracle, configuration: config)
/// let result = await ae.run()
/// print(result.estimatedAmplitude)   // sqrt(a)
/// print(result.estimatedProbability) // a = amplitude^2
/// print(result.confidenceInterval)   // (lower, upper) bounds
/// ```
///
/// - SeeAlso: ``AmplitudeEstimation``
/// - SeeAlso: ``AEConfiguration``
@frozen
public struct AmplitudeEstimationResult: Sendable, CustomStringConvertible {
    /// Estimated amplitude sqrt(a) where a is the probability of measuring a good state.
    ///
    /// This is sin(theta) where theta is the estimated phase from QPE on the Grover operator.
    public let estimatedAmplitude: Double

    /// Estimated probability a = amplitude squared.
    ///
    /// This is sin^2(theta), representing the probability of measuring a marked state
    /// when sampling from the prepared superposition.
    public let estimatedProbability: Double

    /// Confidence interval for the estimated probability.
    ///
    /// Bounds are computed from the phase estimation precision: epsilon = pi/2^n
    /// gives probability bounds sin^2(theta +/- epsilon).
    public let confidenceInterval: (lower: Double, upper: Double)

    /// Number of oracle calls made during estimation.
    ///
    /// For n precision qubits, this is O(2^n) due to the controlled powers of Q.
    public let oracleCalls: Int

    /// Classical Monte Carlo samples required for equivalent precision.
    ///
    /// Classical sampling requires O(1/epsilon^2) samples to achieve precision epsilon,
    /// demonstrating the quadratic speedup of quantum amplitude estimation.
    public let classicalEquivalentSamples: Int

    /// Creates an amplitude estimation result.
    ///
    /// **Example:**
    /// ```swift
    /// let result = AmplitudeEstimationResult(
    ///     estimatedAmplitude: 0.5,
    ///     estimatedProbability: 0.25,
    ///     confidenceInterval: (lower: 0.20, upper: 0.30),
    ///     oracleCalls: 64,
    ///     classicalEquivalentSamples: 4096
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - estimatedAmplitude: The estimated amplitude sqrt(a)
    ///   - estimatedProbability: The estimated probability a
    ///   - confidenceInterval: Lower and upper bounds for probability
    ///   - oracleCalls: Number of oracle invocations
    ///   - classicalEquivalentSamples: Classical samples for equivalent precision
    public init(
        estimatedAmplitude: Double,
        estimatedProbability: Double,
        confidenceInterval: (lower: Double, upper: Double),
        oracleCalls: Int,
        classicalEquivalentSamples: Int,
    ) {
        self.estimatedAmplitude = estimatedAmplitude
        self.estimatedProbability = estimatedProbability
        self.confidenceInterval = confidenceInterval
        self.oracleCalls = oracleCalls
        self.classicalEquivalentSamples = classicalEquivalentSamples
    }

    /// Multi-line formatted summary of amplitude estimation results.
    @inlinable
    public var description: String {
        let ampStr = String(format: "%.8f", estimatedAmplitude)
        let probStr = String(format: "%.8f", estimatedProbability)
        let lowerStr = String(format: "%.8f", confidenceInterval.lower)
        let upperStr = String(format: "%.8f", confidenceInterval.upper)

        return """
        AmplitudeEstimation Result:
          Estimated Amplitude: \(ampStr)
          Estimated Probability: \(probStr)
          Confidence Interval: [\(lowerStr), \(upperStr)]
          Oracle Calls: \(oracleCalls)
          Classical Equivalent Samples: \(classicalEquivalentSamples)
          Quantum Speedup: \(classicalEquivalentSamples / max(1, oracleCalls))x
        """
    }
}

/// Configuration for amplitude estimation algorithm.
///
/// Controls the precision (number of qubits in the phase register) and whether
/// to use iterative phase estimation (IPE) instead of standard quantum phase
/// estimation (QPE). IPE uses fewer qubits but requires sequential measurements.
///
/// **Example:**
/// ```swift
/// let standardConfig = AEConfiguration(precisionQubits: 8)
/// let iterativeConfig = AEConfiguration(precisionQubits: 10, useIterative: true)
/// ```
///
/// - SeeAlso: ``AmplitudeEstimation``
/// - SeeAlso: ``IPEConfiguration``
@frozen
public struct AEConfiguration: Sendable {
    /// Number of qubits in the precision register.
    ///
    /// Determines estimation accuracy as epsilon = pi/2^n.
    /// More qubits yield higher precision but deeper circuits.
    public let precisionQubits: Int

    /// Whether to use iterative phase estimation instead of standard QPE.
    ///
    /// Iterative PE uses only 1 ancilla qubit but requires sequential measurements.
    /// Recommended for NISQ devices where qubit count is limited.
    public let useIterative: Bool

    /// Creates amplitude estimation configuration.
    ///
    /// **Example:**
    /// ```swift
    /// let config = AEConfiguration(precisionQubits: 6, useIterative: false)
    /// ```
    ///
    /// - Parameters:
    ///   - precisionQubits: Number of precision qubits (must be positive, max 15)
    ///   - useIterative: Use IPE instead of standard QPE (default: false)
    /// - Precondition: precisionQubits > 0
    /// - Precondition: precisionQubits <= 15
    /// - Complexity: O(1)
    public init(precisionQubits: Int, useIterative: Bool = false) {
        ValidationUtilities.validatePositiveInt(precisionQubits, name: "precisionQubits")
        ValidationUtilities.validateUpperBound(precisionQubits, max: 15, name: "precisionQubits")
        self.precisionQubits = precisionQubits
        self.useIterative = useIterative
    }
}

/// Built-in oracle for counting marked states using uniform superposition.
///
/// Creates state preparation A = H^n (uniform superposition over all basis states)
/// with marking oracle that phase-flips the specified marked states. Used for
/// quantum counting problems where we want to estimate the fraction of marked items.
///
/// For N = 2^n total states and M marked states, the amplitude a = M/N, so
/// amplitude estimation yields sqrt(M/N) and probability M/N.
///
/// **Example:**
/// ```swift
/// let oracle = CountingOracle(qubits: 4, markedStates: [3, 7, 11])
/// let ae = AmplitudeEstimation(oracle: oracle, configuration: AEConfiguration(precisionQubits: 6))
/// let result = await ae.run()
/// let estimatedCount = result.estimatedProbability * 16  // Approximately 3
/// ```
///
/// - SeeAlso: ``AmplitudeOracle``
/// - SeeAlso: ``AmplitudeEstimation``
@frozen
public struct CountingOracle: AmplitudeOracle, Sendable {
    /// Number of qubits in the state register.
    public let qubits: Int

    /// Basis state indices marked as "good" states.
    public let markedStates: [Int]

    /// Creates a counting oracle with specified marked states.
    ///
    /// **Example:**
    /// ```swift
    /// let oracle = CountingOracle(qubits: 3, markedStates: [1, 3, 5])
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (must be positive, max 10)
    ///   - markedStates: Array of basis state indices to mark
    /// - Precondition: qubits > 0
    /// - Precondition: qubits <= 10
    /// - Precondition: All markedStates < 2^qubits
    /// - Complexity: O(markedStates.count)
    public init(qubits: Int, markedStates: [Int]) {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateAlgorithmQubitLimit(qubits, max: 10, algorithmName: "CountingOracle")

        let stateSpaceSize = 1 << qubits
        for state in markedStates {
            ValidationUtilities.validateIndexInBounds(state, bound: stateSpaceSize, name: "Marked state")
        }

        self.qubits = qubits
        self.markedStates = markedStates
    }

    /// Applies Hadamard to all qubits creating uniform superposition.
    @_optimize(speed)
    public func applyStatePreparation(to circuit: inout QuantumCircuit) {
        for qubit in 0 ..< qubits {
            circuit.append(.hadamard, to: qubit)
        }
    }

    /// Applies Hadamard to all qubits (inverse = forward for Hadamard).
    @_optimize(speed)
    public func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
        for qubit in (0 ..< qubits).reversed() {
            circuit.append(.hadamard, to: qubit)
        }
    }

    /// Applies phase flip to marked states using Grover oracle construction.
    @_optimize(speed)
    public func applyMarkingOracle(to circuit: inout QuantumCircuit) {
        let oracleGates = QuantumCircuit.groverOracle(
            qubits: qubits,
            oracle: .multipleTargets(markedStates),
        )
        for (gate, targetQubits) in oracleGates {
            circuit.append(gate, to: targetQubits)
        }
    }
}

/// Quantum amplitude estimation combining Grover operator with phase estimation.
///
/// Estimates the amplitude a where state preparation oracle A creates
/// A|0> = sqrt(1-a)|bad> + sqrt(a)|good>. Provides quadratic speedup over
/// classical Monte Carlo sampling: O(1/epsilon) vs O(1/epsilon^2) for precision epsilon.
///
/// The algorithm constructs the Grover operator Q = A * S_0 * A^dagger * S_chi where:
/// - A is the state preparation oracle
/// - A^dagger is its inverse
/// - S_0 = I - 2|0><0| reflects about |0>
/// - S_chi = I - 2|good><good| marks good states
///
/// Q has eigenvalues e^(+/-2i*theta) where sin^2(theta) = a. Phase estimation on Q
/// extracts theta, from which we compute the amplitude sqrt(a) = sin(theta).
///
/// **Example:**
/// ```swift
/// let oracle = CountingOracle(qubits: 4, markedStates: [3, 7, 11, 15])
/// let config = AEConfiguration(precisionQubits: 8)
/// let ae = AmplitudeEstimation(oracle: oracle, configuration: config)
///
/// let result = await ae.run(progress: { message in
///     print(message)
/// })
/// print("Estimated count: \(result.estimatedProbability * 16)")
/// ```
///
/// - Complexity: O(2^n) oracle calls for n precision qubits
/// - SeeAlso: ``AmplitudeOracle``
/// - SeeAlso: ``AEConfiguration``
/// - SeeAlso: ``AmplitudeEstimationResult``
/// - SeeAlso: ``GroverDiffusion``
public actor AmplitudeEstimation {
    /// Oracle defining state preparation and marking.
    private let oracle: any AmplitudeOracle

    /// Algorithm configuration.
    private let configuration: AEConfiguration

    /// Quantum simulator for circuit execution.
    private let simulator: QuantumSimulator

    /// Creates amplitude estimation instance with specified oracle and configuration.
    ///
    /// **Example:**
    /// ```swift
    /// let oracle = CountingOracle(qubits: 4, markedStates: [5, 10])
    /// let config = AEConfiguration(precisionQubits: 6)
    /// let ae = AmplitudeEstimation(oracle: oracle, configuration: config)
    /// ```
    ///
    /// - Parameters:
    ///   - oracle: Amplitude oracle defining state preparation and marking
    ///   - configuration: Algorithm configuration
    /// - Complexity: O(1)
    public init(oracle: any AmplitudeOracle, configuration: AEConfiguration) {
        self.oracle = oracle
        self.configuration = configuration
        simulator = QuantumSimulator()
    }

    /// Runs amplitude estimation algorithm.
    ///
    /// Constructs the Grover operator from the oracle and applies phase estimation
    /// to extract the amplitude. Supports both standard QPE and iterative PE based
    /// on configuration.
    ///
    /// **Example:**
    /// ```swift
    /// let result = await ae.run(progress: { status in
    ///     print(status)
    /// })
    /// print(result)
    /// ```
    ///
    /// - Parameter progress: Optional callback receiving status messages during execution
    /// - Returns: Amplitude estimation result with amplitude, probability, and statistics
    /// - Complexity: O(2^n * circuit_depth) where n = precisionQubits
    @_optimize(speed)
    @_eagerMove
    public func run(
        progress: (@Sendable (String) async -> Void)? = nil,
    ) async -> AmplitudeEstimationResult {
        await progress?("Building Grover operator Q = A * S_0 * A^dagger * S_chi")

        let n = configuration.precisionQubits
        let stateQubits = oracle.qubits

        let estimatedPhase: Double = if configuration.useIterative {
            await runIterativePhaseEstimation(progress: progress)
        } else {
            await runStandardPhaseEstimation(progress: progress)
        }

        let theta = estimatedPhase * .pi

        let amplitude = abs(Foundation.sin(theta))
        let probability = amplitude * amplitude

        let epsilon = .pi / Double(1 << n)
        let thetaLower = max(0.0, theta - epsilon)
        let thetaUpper = min(.pi, theta + epsilon)
        let probLower = Foundation.sin(thetaLower) * Foundation.sin(thetaLower)
        let probUpper = Foundation.sin(thetaUpper) * Foundation.sin(thetaUpper)

        let confidenceInterval = (lower: min(probLower, probUpper), upper: max(probLower, probUpper))

        let oracleCalls = computeOracleCalls(precisionQubits: n, stateQubits: stateQubits)
        let classicalSamples = computeClassicalEquivalent(precisionQubits: n)

        await progress?("Amplitude estimation complete: amplitude = \(String(format: "%.6f", amplitude))")

        return AmplitudeEstimationResult(
            estimatedAmplitude: amplitude,
            estimatedProbability: probability,
            confidenceInterval: confidenceInterval,
            oracleCalls: oracleCalls,
            classicalEquivalentSamples: classicalSamples,
        )
    }

    /// Runs standard quantum phase estimation on Grover operator.
    @_optimize(speed)
    @_eagerMove
    private func runStandardPhaseEstimation(
        progress: (@Sendable (String) async -> Void)?,
    ) async -> Double {
        await progress?("Running standard quantum phase estimation")

        let n = configuration.precisionQubits
        let stateQubits = oracle.qubits
        let totalQubits = n + stateQubits

        var circuit = QuantumCircuit(qubits: totalQubits)

        for qubit in 0 ..< n {
            circuit.append(.hadamard, to: qubit)
        }

        var stateCircuit = QuantumCircuit(qubits: stateQubits)
        oracle.applyStatePreparation(to: &stateCircuit)
        for gate in stateCircuit.gates {
            let shiftedQubits = gate.qubits.map { $0 + n }
            circuit.append(gate.gate, to: shiftedQubits)
        }

        for controlQubit in 0 ..< n {
            let power = 1 << (n - 1 - controlQubit)
            await progress?("Applying controlled-Q^(\(power)) from qubit \(controlQubit)")
            applyControlledGroverPower(
                to: &circuit,
                controlQubit: controlQubit,
                power: power,
                stateQubitOffset: n,
            )
        }

        await progress?("Applying inverse QFT to precision register")
        applyInverseQFT(to: &circuit, qubits: n)

        let state = await simulator.execute(circuit)

        let precisionStateSize = 1 << n
        var precisionProbabilities = [Double](repeating: 0.0, count: precisionStateSize)

        for basisIndex in 0 ..< state.stateSpaceSize {
            let precisionIndex = basisIndex % precisionStateSize
            let probability = state.amplitudes[basisIndex].magnitudeSquared
            precisionProbabilities[precisionIndex] += probability
        }

        var maxIndex = 0
        var maxProb = precisionProbabilities[0]
        for i in 1 ..< precisionStateSize {
            if precisionProbabilities[i] > maxProb {
                maxProb = precisionProbabilities[i]
                maxIndex = i
            }
        }

        let rawPhase = Double(maxIndex) / Double(precisionStateSize)

        return rawPhase
    }

    /// Runs iterative phase estimation on Grover operator.
    @_optimize(speed)
    @_eagerMove
    private func runIterativePhaseEstimation(
        progress: (@Sendable (String) async -> Void)?,
    ) async -> Double {
        await progress?("Running iterative phase estimation")

        let n = configuration.precisionQubits
        let stateQubits = oracle.qubits
        let totalQubits = 1 + stateQubits

        var measuredBits: [Int] = []
        measuredBits.reserveCapacity(n)

        for k in 0 ..< n {
            let power = 1 << (n - 1 - k)
            await progress?("IPE iteration \(k + 1)/\(n): applying Q^\(power)")

            var circuit = QuantumCircuit(qubits: totalQubits)

            circuit.append(.hadamard, to: 0)

            var stateCircuit = QuantumCircuit(qubits: stateQubits)
            oracle.applyStatePreparation(to: &stateCircuit)
            for gate in stateCircuit.gates {
                let shiftedQubits = gate.qubits.map { $0 + 1 }
                circuit.append(gate.gate, to: shiftedQubits)
            }

            applyControlledGroverPower(
                to: &circuit,
                controlQubit: 0,
                power: power,
                stateQubitOffset: 1,
            )

            var correction = 0.0
            for (j, bit) in measuredBits.enumerated() {
                if bit == 1 {
                    let exponent = k - j
                    correction += .pi / Double(1 << exponent)
                }
            }

            if abs(correction) > 1e-12 {
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
        }

        var phase = 0.0
        for (k, bit) in measuredBits.enumerated() {
            if bit == 1 {
                phase += 1.0 / Double(1 << (k + 1))
            }
        }

        return phase
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
            applyControlledMarkingOracle(to: &circuit, controlQubit: controlQubit, stateQubitOffset: stateQubitOffset)
            applyControlledDiffusion(to: &circuit, controlQubit: controlQubit, stateQubitOffset: stateQubitOffset)
        }
    }

    /// Applies controlled marking oracle (controlled S_chi).
    @_optimize(speed)
    private func applyControlledMarkingOracle(
        to circuit: inout QuantumCircuit,
        controlQubit: Int,
        stateQubitOffset: Int,
    ) {
        var oracleCircuit = QuantumCircuit(qubits: oracle.qubits)
        oracle.applyMarkingOracle(to: &oracleCircuit)

        for gate in oracleCircuit.gates {
            let shiftedQubits = gate.qubits.map { $0 + stateQubitOffset }
            let controlledGate = makeControlled(gate: gate.gate, control: controlQubit, targets: shiftedQubits)
            for (g, q) in controlledGate {
                circuit.append(g, to: q)
            }
        }
    }

    /// Applies controlled diffusion operator (controlled A * S_0 * A^dagger).
    @_optimize(speed)
    private func applyControlledDiffusion(
        to circuit: inout QuantumCircuit,
        controlQubit: Int,
        stateQubitOffset: Int,
    ) {
        let stateQubits = oracle.qubits

        var invCircuit = QuantumCircuit(qubits: stateQubits)
        oracle.applyStatePreparationInverse(to: &invCircuit)
        for gate in invCircuit.gates {
            let shiftedQubits = gate.qubits.map { $0 + stateQubitOffset }
            let controlledGate = makeControlled(gate: gate.gate, control: controlQubit, targets: shiftedQubits)
            for (g, q) in controlledGate {
                circuit.append(g, to: q)
            }
        }

        applyControlledReflectionAboutZero(
            to: &circuit,
            controlQubit: controlQubit,
            stateQubitOffset: stateQubitOffset,
            stateQubits: stateQubits,
        )

        var prepCircuit = QuantumCircuit(qubits: stateQubits)
        oracle.applyStatePreparation(to: &prepCircuit)
        for gate in prepCircuit.gates {
            let shiftedQubits = gate.qubits.map { $0 + stateQubitOffset }
            let controlledGate = makeControlled(gate: gate.gate, control: controlQubit, targets: shiftedQubits)
            for (g, q) in controlledGate {
                circuit.append(g, to: q)
            }
        }
    }

    /// Applies controlled reflection about |0> state.
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

            var controls = [controlQubit]
            for q in 0 ..< stateQubits - 1 {
                controls.append(stateQubitOffset + q)
            }
            let target = stateQubitOffset + stateQubits - 1

            applyMultiControlledX(to: &circuit, controls: controls, target: target)

            circuit.append(.hadamard, to: stateQubitOffset + stateQubits - 1)
        }

        for qubit in (0 ..< stateQubits).reversed() {
            circuit.append(.cnot, to: [controlQubit, stateQubitOffset + qubit])
        }
    }

    /// Applies multi-controlled X gate.
    @_optimize(speed)
    private func applyMultiControlledX(
        to circuit: inout QuantumCircuit,
        controls: [Int],
        target: Int,
    ) {
        if controls.count == 2 {
            circuit.append(.toffoli, to: [controls[0], controls[1], target])
        } else {
            let mcxGates = GroverDiffusion.buildMultiControlledXGates(controls: controls, target: target)
            for (gate, qubits) in mcxGates {
                circuit.append(gate, to: qubits)
            }
        }
    }

    /// Makes a controlled version of a gate.
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

        case .pauliY:
            return [
                (.phase(.value(-.pi / 2.0)), [targets[0]]),
                (.cnot, [control, targets[0]]),
                (.sGate, [targets[0]]),
            ]

        case .pauliZ:
            return [(.controlledPhase(.value(.pi)), [control, targets[0]])]

        case let .rotationZ(angle):
            let halfAngle: Double = switch angle {
            case let .value(v): v / 2.0
            default: 0.0
            }
            return [
                (.rotationZ(.value(halfAngle)), [targets[0]]),
                (.cnot, [control, targets[0]]),
                (.rotationZ(.value(-halfAngle)), [targets[0]]),
                (.cnot, [control, targets[0]]),
            ]

        case let .rotationY(angle):
            let halfAngle: Double = switch angle {
            case let .value(v): v / 2.0
            default: 0.0
            }
            return [
                (.rotationY(.value(halfAngle)), [targets[0]]),
                (.cnot, [control, targets[0]]),
                (.rotationY(.value(-halfAngle)), [targets[0]]),
                (.cnot, [control, targets[0]]),
            ]

        case let .rotationX(angle):
            let halfAngle: Double = switch angle {
            case let .value(v): v / 2.0
            default: 0.0
            }
            return [
                (.hadamard, [targets[0]]),
                (.rotationZ(.value(halfAngle)), [targets[0]]),
                (.cnot, [control, targets[0]]),
                (.rotationZ(.value(-halfAngle)), [targets[0]]),
                (.cnot, [control, targets[0]]),
                (.hadamard, [targets[0]]),
            ]

        case .cnot:
            return [(.toffoli, [control, targets[0], targets[1]])]

        case .toffoli:
            return applyControlledToffoli(control: control, targets: targets)

        case let .phase(phi):
            let halfAngle: Double = switch phi {
            case let .value(v): v / 2.0
            default: 0.0
            }
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

        case let .controlledPhase(phi):
            let phiVal: Double = switch phi {
            case let .value(v): v
            default: 0.0
            }
            return [
                (.rotationZ(.value(phiVal / 2.0)), [targets[0]]),
                (.cnot, [control, targets[0]]),
                (.rotationZ(.value(-phiVal / 2.0)), [targets[0]]),
                (.cnot, [control, targets[0]]),
                (.rotationZ(.value(phiVal / 2.0)), [targets[1]]),
                (.cnot, [control, targets[1]]),
                (.rotationZ(.value(-phiVal / 2.0)), [targets[1]]),
                (.cnot, [control, targets[1]]),
            ]

        default:
            return [(gate, targets)]
        }
    }

    /// Applies controlled Toffoli using ancilla decomposition.
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

    /// Applies inverse QFT to the first n qubits.
    @_optimize(speed)
    private func applyInverseQFT(to circuit: inout QuantumCircuit, qubits n: Int) {
        for i in 0 ..< n / 2 {
            circuit.append(.swap, to: [i, n - 1 - i])
        }

        for j in 0 ..< n {
            for k in 0 ..< j {
                let angle = -.pi / Double(1 << (j - k))
                circuit.append(.controlledPhase(.value(angle)), to: [k, j])
            }
            circuit.append(.hadamard, to: j)
        }
    }

    /// Computes total oracle calls for amplitude estimation.
    @_effects(readonly)
    private func computeOracleCalls(precisionQubits: Int, stateQubits _: Int) -> Int {
        var total = 0
        for k in 0 ..< precisionQubits {
            let power = 1 << (precisionQubits - 1 - k)
            total += 2 * power
        }
        return total
    }

    /// Computes classical Monte Carlo samples for equivalent precision.
    ///
    /// Classical Monte Carlo requires O(1/ε²) samples for precision ε.
    /// Quantum amplitude estimation achieves precision ε = 1/2^n with O(2^n) oracle calls.
    /// For fair comparison, use ε = 1/2^n (not π/2^n), giving classical = 2^(2n).
    @_effects(readonly)
    private func computeClassicalEquivalent(precisionQubits: Int) -> Int {
        let samples = 1 << (2 * precisionQubits)
        return samples
    }
}
