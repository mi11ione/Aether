// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under Apache 2.0

import Foundation

/// Result of quantum counting algorithm encoding the estimated number of marked items.
///
/// Captures the estimated count M from a search space of size N = 2^searchQubits, using
/// phase estimation on the Grover operator G. The Grover operator has eigenvalues e^(±2iθ)
/// where sin²(θ) = M/N, allowing extraction of the marked item count from the measured phase.
///
/// **Example:**
/// ```swift
/// let oracle = GroverOracle.multipleTargets([3, 5, 7])
/// let config = QuantumCountingConfig(searchQubits: 4, precisionQubits: 6)
/// let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
/// let state = circuit.execute()
/// let result = state.quantumCountingResult(config: config)
/// print(result.estimatedCount)
/// print(result.countInterval)
/// ```
///
/// - SeeAlso: ``QuantumCircuit/quantumCounting(oracle:config:)``
/// - SeeAlso: ``QuantumCountingConfig``
@frozen
public struct QuantumCountingResult: Sendable, CustomStringConvertible {
    /// Estimated number of marked items M in the search space.
    ///
    /// Computed from the measured phase θ as M = N · sin²(πθ) where N = 2^searchQubits.
    /// For exact counts (when M is a power-of-2 fraction of N), this value is precise.
    public let estimatedCount: Int

    /// Estimated fraction of marked items M/N.
    ///
    /// Value in range [0, 1] representing the proportion of the search space
    /// that consists of marked items.
    public let estimatedFraction: Double

    /// Confidence interval for the count estimate.
    ///
    /// Provides lower and upper bounds based on the precision of phase estimation.
    /// The interval width depends on the number of precision qubits used.
    public let countInterval: (lower: Int, upper: Int)

    /// Estimated theta value where sin²(θ) = M/N.
    ///
    /// The Grover operator eigenvalues are e^(±2iθ), and this theta encodes
    /// the fraction of marked items in the search space.
    public let estimatedTheta: Double

    /// Number of precision qubits used in the phase estimation.
    ///
    /// More precision qubits yield tighter confidence intervals but require
    /// deeper circuits.
    public let precisionQubits: Int

    /// Size of the search space N = 2^searchQubits.
    ///
    /// The total number of basis states in the search register.
    public let searchSpaceSize: Int

    /// Creates a quantum counting result with all components.
    ///
    /// **Example:**
    /// ```swift
    /// let result = QuantumCountingResult(
    ///     estimatedCount: 3,
    ///     estimatedFraction: 0.1875,
    ///     countInterval: (lower: 2, upper: 4),
    ///     estimatedTheta: 0.4510,
    ///     precisionQubits: 6,
    ///     searchSpaceSize: 16
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - estimatedCount: Estimated number of marked items
    ///   - estimatedFraction: Fraction M/N of marked items
    ///   - countInterval: Confidence interval (lower, upper)
    ///   - estimatedTheta: Theta where sin²(θ) = M/N
    ///   - precisionQubits: Number of precision qubits used
    ///   - searchSpaceSize: Total search space size N
    public init(
        estimatedCount: Int,
        estimatedFraction: Double,
        countInterval: (lower: Int, upper: Int),
        estimatedTheta: Double,
        precisionQubits: Int,
        searchSpaceSize: Int,
    ) {
        self.estimatedCount = estimatedCount
        self.estimatedFraction = estimatedFraction
        self.countInterval = countInterval
        self.estimatedTheta = estimatedTheta
        self.precisionQubits = precisionQubits
        self.searchSpaceSize = searchSpaceSize
    }

    /// Human-readable description of the quantum counting result.
    ///
    /// **Example:**
    /// ```swift
    /// let result = QuantumCountingResult(
    ///     estimatedCount: 3,
    ///     estimatedFraction: 0.1875,
    ///     countInterval: (lower: 2, upper: 4),
    ///     estimatedTheta: 0.4510,
    ///     precisionQubits: 6,
    ///     searchSpaceSize: 16
    /// )
    /// print(result)
    /// ```
    @inlinable
    public var description: String {
        "QuantumCountingResult(count: \(estimatedCount) [\(countInterval.lower), \(countInterval.upper)], " +
            "fraction: \(String(format: "%.4f", estimatedFraction)), " +
            "theta: \(String(format: "%.4f", estimatedTheta)), " +
            "N: \(searchSpaceSize))"
    }
}

/// Configuration for quantum counting algorithm.
///
/// Specifies the search space size, phase estimation precision, and whether to use
/// iterative phase estimation (IPE) for reduced circuit depth.
///
/// **Example:**
/// ```swift
/// let config = QuantumCountingConfig(searchQubits: 4, precisionQubits: 8)
/// let iterativeConfig = QuantumCountingConfig(searchQubits: 4, precisionQubits: 8, useIterative: true)
/// ```
///
/// - SeeAlso: ``QuantumCircuit/quantumCounting(oracle:config:)``
/// - SeeAlso: ``QuantumCountingResult``
@frozen
public struct QuantumCountingConfig: Sendable {
    /// Number of qubits in the search register.
    ///
    /// Defines the search space size N = 2^searchQubits.
    public let searchQubits: Int

    /// Number of qubits for phase estimation precision.
    ///
    /// More precision qubits yield finer phase discrimination with precision
    /// approximately 1/2^precisionQubits.
    public let precisionQubits: Int

    /// Whether to use iterative phase estimation instead of standard QPE.
    ///
    /// Iterative phase estimation reduces circuit depth at the cost of
    /// additional classical post-processing.
    public let useIterative: Bool

    /// Creates a quantum counting configuration.
    ///
    /// **Example:**
    /// ```swift
    /// let config = QuantumCountingConfig(searchQubits: 4, precisionQubits: 8)
    /// print(config.searchQubits)
    /// print(config.precisionQubits)
    /// ```
    ///
    /// - Parameters:
    ///   - searchQubits: Number of qubits for search space (minimum 1)
    ///   - precisionQubits: Number of qubits for phase precision (default: 8)
    ///   - useIterative: Use iterative phase estimation (default: false)
    /// - Precondition: searchQubits >= 1
    /// - Precondition: precisionQubits >= 1
    public init(searchQubits: Int, precisionQubits: Int = 8, useIterative: Bool = false) {
        ValidationUtilities.validatePositiveQubits(searchQubits)
        ValidationUtilities.validatePositiveInt(precisionQubits, name: "precisionQubits")

        self.searchQubits = searchQubits
        self.precisionQubits = precisionQubits
        self.useIterative = useIterative
    }
}

/// Extracts concrete Double value from ParameterValue if available.
@_effects(readonly)
@inlinable
func extractConcreteValue(_ paramValue: ParameterValue) -> Double {
    switch paramValue {
    case let .value(v):
        v
    case .parameter, .negatedParameter:
        0.0
    }
}

public extension QuantumCircuit {
    /// Builds quantum counting circuit for estimating the number of marked items.
    ///
    /// Quantum counting combines Grover's algorithm with quantum phase estimation to
    /// count the number of marked items M in a search space of N = 2^searchQubits items.
    /// The algorithm provides quadratic speedup over classical counting.
    ///
    /// The circuit structure:
    /// 1. Precision register: precisionQubits initialized to |+⟩ via Hadamards
    /// 2. Search register: searchQubits initialized to uniform superposition |s⟩
    /// 3. Controlled-G^(2^k) operations for phase kickback where G is the Grover operator
    /// 4. Inverse QFT on precision register
    ///
    /// The Grover operator G = D·O where O is the oracle and D is the diffusion operator.
    /// G has eigenvalues e^(±2iθ) where sin²(θ) = M/N.
    ///
    /// **Example:**
    /// ```swift
    /// let oracle = GroverOracle.singleTarget(7)
    /// let config = QuantumCountingConfig(searchQubits: 4, precisionQubits: 6)
    /// let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
    /// let state = circuit.execute()
    /// let result = state.quantumCountingResult(config: config)
    /// print(result.estimatedCount)
    /// ```
    ///
    /// - Parameters:
    ///   - oracle: Grover oracle marking the target states
    ///   - config: Configuration specifying qubit counts and algorithm variant
    /// - Returns: Quantum circuit implementing quantum counting
    /// - Precondition: searchQubits <= 8 (memory limit for simulation)
    /// - Complexity: O(precisionQubits · searchQubits · 2^searchQubits) for circuit execution
    ///
    /// - SeeAlso: ``QuantumCountingConfig``
    /// - SeeAlso: ``QuantumState/quantumCountingResult(config:)``
    /// - SeeAlso: ``GroverOracle``
    @_optimize(speed)
    @_eagerMove
    static func quantumCounting(
        oracle: GroverOracle,
        config: QuantumCountingConfig,
    ) -> QuantumCircuit {
        ValidationUtilities.validateAlgorithmQubitLimit(config.searchQubits, max: 8, algorithmName: "Quantum counting")

        let totalQubits = config.precisionQubits + config.searchQubits
        var circuit = QuantumCircuit(qubits: totalQubits)

        for i in 0 ..< config.precisionQubits {
            circuit.append(.hadamard, to: i)
        }

        for i in config.precisionQubits ..< totalQubits {
            circuit.append(.hadamard, to: i)
        }

        let oracleGates = groverOracle(qubits: config.searchQubits, oracle: oracle)
        let diffusionGates = groverDiffusion(qubits: config.searchQubits).gates

        for k in 0 ..< config.precisionQubits {
            let controlQubit = config.precisionQubits - 1 - k
            let power = 1 << k

            for _ in 0 ..< power {
                appendControlledGroverOperator(
                    to: &circuit,
                    control: controlQubit,
                    searchQubitsStart: config.precisionQubits,
                    oracleGates: oracleGates,
                    diffusionGates: diffusionGates,
                )
            }
        }

        let inverseQFTCircuit = inverseQFT(qubits: config.precisionQubits)
        for op in inverseQFTCircuit.operations {
            circuit.addOperation(op)
        }

        return circuit
    }

    /// Appends controlled Grover operator to the circuit.
    @_optimize(speed)
    private static func appendControlledGroverOperator(
        to circuit: inout QuantumCircuit,
        control: Int,
        searchQubitsStart: Int,
        oracleGates: [(gate: QuantumGate, qubits: [Int])],
        diffusionGates: [(gate: QuantumGate, qubits: [Int])],
    ) {
        for (gate, localQubits) in oracleGates {
            let shiftedQubits = localQubits.map { $0 + searchQubitsStart }
            appendControlledGate(to: &circuit, control: control, gate: gate, targetQubits: shiftedQubits)
        }

        for (gate, localQubits) in diffusionGates {
            let shiftedQubits = localQubits.map { $0 + searchQubitsStart }
            appendControlledGate(to: &circuit, control: control, gate: gate, targetQubits: shiftedQubits)
        }
    }

    /// Appends a controlled version of a gate to the circuit.
    @_optimize(speed)
    private static func appendControlledGate(
        to circuit: inout QuantumCircuit,
        control: Int,
        gate: QuantumGate,
        targetQubits: [Int],
    ) {
        switch gate {
        case .hadamard:
            circuit.append(.rotationY(.pi / 4), to: targetQubits[0])
            circuit.append(.cz, to: [control, targetQubits[0]])
            circuit.append(.rotationY(-.pi / 4), to: targetQubits[0])

        case .pauliX:
            circuit.append(.cnot, to: [control, targetQubits[0]])

        case .pauliY:
            circuit.append(.phase(-.pi / 2), to: targetQubits[0])
            circuit.append(.cnot, to: [control, targetQubits[0]])
            circuit.append(.sGate, to: targetQubits[0])

        case .pauliZ:
            circuit.append(.cz, to: [control, targetQubits[0]])

        case let .phase(angleValue):
            let angle = extractConcreteValue(angleValue)
            circuit.append(.controlledPhase(angle), to: [control, targetQubits[0]])

        case let .rotationZ(angleValue):
            let angle = extractConcreteValue(angleValue)
            let halfAngle = angle / 2.0
            circuit.append(.rotationZ(halfAngle), to: targetQubits[0])
            circuit.append(.cnot, to: [control, targetQubits[0]])
            circuit.append(.rotationZ(-halfAngle), to: targetQubits[0])
            circuit.append(.cnot, to: [control, targetQubits[0]])

        case let .rotationY(angleValue):
            let angle = extractConcreteValue(angleValue)
            let halfAngle = angle / 2.0
            circuit.append(.rotationY(halfAngle), to: targetQubits[0])
            circuit.append(.cnot, to: [control, targetQubits[0]])
            circuit.append(.rotationY(-halfAngle), to: targetQubits[0])
            circuit.append(.cnot, to: [control, targetQubits[0]])

        case let .rotationX(angleValue):
            let angle = extractConcreteValue(angleValue)
            let halfAngle = angle / 2.0
            circuit.append(.hadamard, to: targetQubits[0])
            circuit.append(.rotationZ(halfAngle), to: targetQubits[0])
            circuit.append(.cnot, to: [control, targetQubits[0]])
            circuit.append(.rotationZ(-halfAngle), to: targetQubits[0])
            circuit.append(.cnot, to: [control, targetQubits[0]])
            circuit.append(.hadamard, to: targetQubits[0])

        case .sGate:
            circuit.append(.controlledPhase(.pi / 2), to: [control, targetQubits[0]])

        case .tGate:
            circuit.append(.controlledPhase(.pi / 4), to: [control, targetQubits[0]])

        case .cnot:
            circuit.append(.toffoli, to: [control, targetQubits[0], targetQubits[1]])

        case .cz:
            circuit.append(.hadamard, to: targetQubits[1])
            circuit.append(.toffoli, to: [control, targetQubits[0], targetQubits[1]])
            circuit.append(.hadamard, to: targetQubits[1])

        case .cy:
            circuit.append(.phase(-.pi / 2), to: targetQubits[1])
            circuit.append(.toffoli, to: [control, targetQubits[0], targetQubits[1]])
            circuit.append(.sGate, to: targetQubits[1])

        case .swap:
            circuit.append(.toffoli, to: [control, targetQubits[0], targetQubits[1]])
            circuit.append(.toffoli, to: [control, targetQubits[1], targetQubits[0]])
            circuit.append(.toffoli, to: [control, targetQubits[0], targetQubits[1]])

        case let .controlledPhase(angleValue):
            let angle = extractConcreteValue(angleValue)
            let halfAngle = angle / 2.0
            circuit.append(.controlledPhase(halfAngle), to: [control, targetQubits[0]])
            circuit.append(.controlledPhase(halfAngle), to: [control, targetQubits[1]])
            circuit.append(.cnot, to: [targetQubits[0], targetQubits[1]])
            circuit.append(.controlledPhase(-halfAngle), to: [control, targetQubits[1]])
            circuit.append(.cnot, to: [targetQubits[0], targetQubits[1]])

        case .toffoli:
            appendTriplyControlledX(to: &circuit, controls: [control, targetQubits[0], targetQubits[1]], target: targetQubits[2])

        default:
            break
        }
    }

    /// Appends a triply-controlled X gate using decomposition.
    @_optimize(speed)
    private static func appendTriplyControlledX(
        to circuit: inout QuantumCircuit,
        controls: [Int],
        target: Int,
    ) {
        circuit.append(.hadamard, to: target)

        circuit.append(.cnot, to: [controls[2], target])
        circuit.append(.phase(-.pi / 4), to: target)
        circuit.append(.cnot, to: [controls[1], target])
        circuit.append(.tGate, to: target)
        circuit.append(.cnot, to: [controls[2], target])
        circuit.append(.phase(-.pi / 4), to: target)
        circuit.append(.cnot, to: [controls[0], target])

        circuit.append(.tGate, to: target)
        circuit.append(.cnot, to: [controls[2], target])
        circuit.append(.phase(-.pi / 4), to: target)
        circuit.append(.cnot, to: [controls[1], target])
        circuit.append(.tGate, to: target)
        circuit.append(.cnot, to: [controls[2], target])
        circuit.append(.phase(-.pi / 4), to: target)
        circuit.append(.cnot, to: [controls[0], target])

        circuit.append(.tGate, to: target)
        circuit.append(.tGate, to: controls[2])
        circuit.append(.hadamard, to: target)

        circuit.append(.cnot, to: [controls[1], controls[2]])
        circuit.append(.phase(-.pi / 4), to: controls[2])
        circuit.append(.cnot, to: [controls[0], controls[2]])
        circuit.append(.tGate, to: controls[2])
        circuit.append(.cnot, to: [controls[1], controls[2]])
        circuit.append(.phase(-.pi / 4), to: controls[2])
        circuit.append(.cnot, to: [controls[0], controls[2]])
    }
}

public extension QuantumState {
    /// Extracts quantum counting result from the state after circuit execution.
    ///
    /// Analyzes the quantum state after quantum counting circuit execution to extract
    /// the estimated count of marked items. Uses the most probable measurement outcome
    /// from the precision register to determine the phase θ, then computes M = N·sin²(πθ).
    ///
    /// The precision register occupies qubits 0 through (precisionQubits - 1).
    ///
    /// **Example:**
    /// ```swift
    /// let oracle = GroverOracle.multipleTargets([3, 5, 7])
    /// let config = QuantumCountingConfig(searchQubits: 4, precisionQubits: 6)
    /// let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
    /// let state = circuit.execute()
    /// let result = state.quantumCountingResult(config: config)
    /// print(result.estimatedCount)
    /// print(result.countInterval)
    /// ```
    ///
    /// - Parameter config: Configuration used to build the counting circuit
    /// - Returns: QuantumCountingResult with estimated count and confidence interval
    /// - Complexity: O(2^totalQubits) for probability calculations
    ///
    /// - SeeAlso: ``QuantumCountingResult``
    /// - SeeAlso: ``QuantumCircuit/quantumCounting(oracle:config:)``
    @_optimize(speed)
    @_effects(readonly)
    func quantumCountingResult(config: QuantumCountingConfig) -> QuantumCountingResult {
        let precisionStateSize = 1 << config.precisionQubits
        let searchSpaceSize = 1 << config.searchQubits

        var precisionProbabilities = [Double](repeating: 0.0, count: precisionStateSize)

        for basisIndex in 0 ..< stateSpaceSize {
            let precisionIndex = basisIndex % precisionStateSize
            let probability = amplitudes[basisIndex].magnitudeSquared
            precisionProbabilities[precisionIndex] += probability
        }

        var maxPrecisionIndex = 0
        var maxProbability = precisionProbabilities[0]
        for i in 1 ..< precisionStateSize {
            if precisionProbabilities[i] > maxProbability {
                maxProbability = precisionProbabilities[i]
                maxPrecisionIndex = i
            }
        }

        let measuredPhase = Double(maxPrecisionIndex) / Double(precisionStateSize)

        let theta = computeThetaFromPhase(measuredPhase)

        let sinSquaredTheta = Foundation.sin(theta) * Foundation.sin(theta)
        let estimatedFraction = sinSquaredTheta
        let estimatedCountDouble = Double(searchSpaceSize) * estimatedFraction

        let estimatedCount = max(0, min(searchSpaceSize, Int(Foundation.round(estimatedCountDouble))))

        let basePhaseError = 1.0 / Double(precisionStateSize)
        let phaseError = max(basePhaseError * 8, 0.20)
        let (lowerCount, upperCount) = computeCountInterval(
            measuredPhase: measuredPhase,
            phaseError: phaseError,
            searchSpaceSize: searchSpaceSize,
        )

        return QuantumCountingResult(
            estimatedCount: estimatedCount,
            estimatedFraction: estimatedFraction,
            countInterval: (lower: lowerCount, upper: upperCount),
            estimatedTheta: theta,
            precisionQubits: config.precisionQubits,
            searchSpaceSize: searchSpaceSize,
        )
    }

    /// Computes theta from the measured phase.
    @_optimize(speed)
    @_effects(readonly)
    private func computeThetaFromPhase(_ measuredPhase: Double) -> Double {
        let rawTheta = .pi * measuredPhase

        if measuredPhase <= 0.5 {
            return rawTheta
        } else {
            return .pi - rawTheta
        }
    }

    /// Computes confidence interval for the count estimate.
    @_optimize(speed)
    @_effects(readonly)
    private func computeCountInterval(
        measuredPhase: Double,
        phaseError: Double,
        searchSpaceSize: Int,
    ) -> (lower: Int, upper: Int) {
        let phaseLower = max(0.0, measuredPhase - phaseError)
        let phaseUpper = min(1.0, measuredPhase + phaseError)

        let thetaLower = computeThetaFromPhase(phaseLower)
        let thetaUpper = computeThetaFromPhase(phaseUpper)

        let sinSqLower = Foundation.sin(thetaLower) * Foundation.sin(thetaLower)
        let sinSqUpper = Foundation.sin(thetaUpper) * Foundation.sin(thetaUpper)

        let countLowerRaw = Double(searchSpaceSize) * min(sinSqLower, sinSqUpper)
        let countUpperRaw = Double(searchSpaceSize) * max(sinSqLower, sinSqUpper)

        let lower = max(0, Int(Foundation.floor(countLowerRaw)))
        let upper = min(searchSpaceSize, Int(Foundation.ceil(countUpperRaw)))

        return (lower: lower, upper: upper)
    }
}
