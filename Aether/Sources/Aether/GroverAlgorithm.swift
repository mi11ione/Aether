// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under Apache 2.0

import Foundation

/// Grover diffusion operator implementing 2|s><s| - I reflection about uniform superposition.
///
/// The diffusion operator is the core component of Grover's amplitude amplification,
/// performing inversion about the mean amplitude. It consists of H^n * (2|0><0| - I) * H^n
/// where H^n applies Hadamard to all qubits.
///
/// **Example:**
/// ```swift
/// let diffusion = QuantumCircuit.groverDiffusion(qubits: 3)
/// print(diffusion.qubits)  // 3
/// print(diffusion.gates.count)  // Gates implementing the diffusion
/// ```
///
/// - SeeAlso: ``QuantumCircuit/groverDiffusion(qubits:)``
/// - SeeAlso: ``GroverOracle``
@frozen
public struct GroverDiffusion: Sendable {
    /// Number of qubits the diffusion operator acts on
    public let qubits: Int

    /// Gate sequence implementing the diffusion operator
    public let gates: [(gate: QuantumGate, qubits: [Int])]

    /// Creates a Grover diffusion operator for the specified number of qubits.
    ///
    /// **Example:**
    /// ```swift
    /// let diffusion = GroverDiffusion(qubits: 3)
    /// ```
    ///
    /// - Parameter qubits: Number of qubits (minimum 1)
    /// - Precondition: qubits >= 1
    @_optimize(speed)
    public init(qubits: Int) {
        ValidationUtilities.validatePositiveQubits(qubits)

        self.qubits = qubits
        var gateSequence: [(gate: QuantumGate, qubits: [Int])] = []
        gateSequence.reserveCapacity(4 * qubits + qubits)

        for qubit in 0 ..< qubits {
            gateSequence.append((.hadamard, [qubit]))
        }

        for qubit in 0 ..< qubits {
            gateSequence.append((.pauliX, [qubit]))
        }

        let mcz = Self.buildMultiControlledZGates(qubits: qubits)
        gateSequence.append(contentsOf: mcz)

        for qubit in 0 ..< qubits {
            gateSequence.append((.pauliX, [qubit]))
        }

        for qubit in 0 ..< qubits {
            gateSequence.append((.hadamard, [qubit]))
        }

        gates = gateSequence
    }

    /// Generates multi-controlled Z gate sequence.
    @_optimize(speed)
    @_effects(readonly)
    static func buildMultiControlledZGates(qubits: Int) -> [(gate: QuantumGate, qubits: [Int])] {
        if qubits == 1 {
            return [(.pauliZ, [0])]
        } else if qubits == 2 {
            return [(.controlledPhase(.pi), [0, 1])]
        } else if qubits == 3 {
            return [
                (.hadamard, [2]),
                (.toffoli, [0, 1, 2]),
                (.hadamard, [2]),
            ]
        } else {
            let target = qubits - 1
            var gates: [(gate: QuantumGate, qubits: [Int])] = []
            gates.append((.hadamard, [target]))

            let controls = Array(0 ..< qubits - 1)
            let mcxGates = buildMultiControlledXGates(controls: controls, target: target)
            gates.append(contentsOf: mcxGates)

            gates.append((.hadamard, [target]))
            return gates
        }
    }

    /// Generates multi-controlled X gate sequence using Toffoli ladder.
    @_optimize(speed)
    @_effects(readonly)
    static func buildMultiControlledXGates(controls: [Int], target: Int) -> [(gate: QuantumGate, qubits: [Int])] {
        let n = controls.count

        if n == 0 {
            return [(.pauliX, [target])]
        } else if n == 1 {
            return [(.cnot, [controls[0], target])]
        } else if n == 2 {
            return [(.toffoli, [controls[0], controls[1], target])]
        }

        // Safety: controls.count >= 3 guaranteed by prior branches
        let maxControlQubit = controls.max()!
        let maxUsedQubit = max(maxControlQubit, target)
        let firstAncilla = maxUsedQubit + 1
        let numAncilla = n - 2

        var gates: [(gate: QuantumGate, qubits: [Int])] = []

        gates.append((.toffoli, [controls[0], controls[1], firstAncilla]))

        for i in 1 ..< numAncilla {
            gates.append((.toffoli, [firstAncilla + i - 1, controls[i + 1], firstAncilla + i]))
        }

        gates.append((.toffoli, [firstAncilla + numAncilla - 1, controls[n - 1], target]))

        for i in (1 ..< numAncilla).reversed() {
            gates.append((.toffoli, [firstAncilla + i - 1, controls[i + 1], firstAncilla + i]))
        }

        gates.append((.toffoli, [controls[0], controls[1], firstAncilla]))

        return gates
    }
}

/// Specification for Grover oracle marking target states with phase flip.
///
/// The oracle implements I - 2|targets><targets| which applies a phase flip
/// to marked states. This is the "black box" function in Grover's algorithm
/// that identifies the solution states.
///
/// **Example:**
/// ```swift
/// let singleOracle = GroverOracle.singleTarget(5)
/// let multiOracle = GroverOracle.multipleTargets([3, 5, 7])
/// ```
///
/// - SeeAlso: ``QuantumCircuit/groverOracle(qubits:oracle:)``
/// - SeeAlso: ``GroverDiffusion``
@frozen
public enum GroverOracle: Sendable {
    /// Oracle marking a single target state
    case singleTarget(Int)

    /// Oracle marking multiple target states
    case multipleTargets([Int])

    /// Custom oracle with explicit gate sequence
    case custom([(gate: QuantumGate, qubits: [Int])])

    /// Number of marked states
    ///
    /// **Example:**
    /// ```swift
    /// let oracle = GroverOracle.multipleTargets([1, 3, 5])
    /// print(oracle.markedCount)  // 3
    /// ```
    @inlinable
    public var markedCount: Int {
        switch self {
        case .singleTarget:
            1
        case let .multipleTargets(targets):
            targets.count
        case .custom:
            1
        }
    }

    /// Target state indices marked by this oracle
    ///
    /// **Example:**
    /// ```swift
    /// let oracle = GroverOracle.multipleTargets([1, 3, 5])
    /// print(oracle.targetStates)  // [1, 3, 5]
    /// ```
    @inlinable
    public var targetStates: [Int] {
        switch self {
        case let .singleTarget(target):
            [target]
        case let .multipleTargets(targets):
            targets
        case .custom:
            []
        }
    }
}

/// Result from Grover search algorithm with analysis metrics.
///
/// Contains the measurement outcome, success probability estimation,
/// iteration count, and comparison to optimal iteration count.
///
/// **Example:**
/// ```swift
/// let state = QuantumCircuit.groverSearch(qubits: 4, oracle: .singleTarget(7)).execute()
/// let result = state.groverResult(oracle: .singleTarget(7), iterations: 3)
/// print(result.measuredState)  // Most probable state
/// print(result.isTarget)  // Whether it matches target
/// ```
///
/// - SeeAlso: ``QuantumState/groverResult(oracle:iterations:)``
@frozen
public struct GroverResult: Sendable, CustomStringConvertible {
    /// The measured basis state index
    public let measuredState: Int

    /// Estimated success probability from quantum state
    public let successProbability: Double

    /// Number of Grover iterations applied
    public let iterations: Int

    /// Optimal number of iterations for maximum success probability
    public let optimalIterations: Int

    /// Whether measured state is in the target set
    public let isTarget: Bool

    /// Creates a Grover result with all metrics.
    ///
    /// **Example:**
    /// ```swift
    /// let result = GroverResult(
    ///     measuredState: 5,
    ///     successProbability: 0.95,
    ///     iterations: 3,
    ///     optimalIterations: 3,
    ///     isTarget: true
    /// )
    /// ```
    public init(
        measuredState: Int,
        successProbability: Double,
        iterations: Int,
        optimalIterations: Int,
        isTarget: Bool,
    ) {
        self.measuredState = measuredState
        self.successProbability = successProbability
        self.iterations = iterations
        self.optimalIterations = optimalIterations
        self.isTarget = isTarget
    }

    /// String representation of Grover result
    @inlinable
    public var description: String {
        let status = isTarget ? "SUCCESS" : "FAILURE"
        return "GroverResult(\(status): state=\(measuredState), " +
            "prob=\(String(format: "%.4f", successProbability)), " +
            "iterations=\(iterations)/\(optimalIterations))"
    }
}

// MARK: - QuantumCircuit Extensions

public extension QuantumCircuit {
    /// Creates Grover diffusion operator (2|s><s| - I).
    ///
    /// The diffusion operator reflects the state about the uniform superposition |s> = H^n|0>.
    /// It is implemented as H^n * (2|0><0| - I) * H^n where the middle term is a phase flip
    /// on the |0...0> state.
    ///
    /// **Example:**
    /// ```swift
    /// let diffusion = QuantumCircuit.groverDiffusion(qubits: 3)
    /// var circuit = QuantumCircuit(qubits: 3)
    /// for (gate, qubits) in diffusion.gates {
    ///     circuit.append(gate, to: qubits)
    /// }
    /// ```
    ///
    /// - Parameter qubits: Number of qubits (minimum 1)
    /// - Returns: GroverDiffusion containing the gate sequence
    /// - Precondition: qubits >= 1
    /// - Complexity: O(n) gates where n = qubits
    ///
    /// - SeeAlso: ``GroverDiffusion``
    /// - SeeAlso: ``groverOracle(qubits:oracle:)``
    @_effects(readonly)
    static func groverDiffusion(qubits: Int) -> GroverDiffusion {
        GroverDiffusion(qubits: qubits)
    }

    /// Creates oracle circuit for phase flip on target states.
    ///
    /// Implements the oracle I - 2|targets><targets| which applies a pi phase
    /// to all marked states. For single targets, uses X gates to encode the
    /// complement bits, multi-controlled Z for phase inversion, then X gates
    /// to restore state.
    ///
    /// **Example:**
    /// ```swift
    /// let oracleGates = QuantumCircuit.groverOracle(
    ///     qubits: 3,
    ///     oracle: .singleTarget(5)
    /// )
    /// var circuit = QuantumCircuit(qubits: 3)
    /// for (gate, qubits) in oracleGates {
    ///     circuit.append(gate, to: qubits)
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits in the search space
    ///   - oracle: Oracle specification (single, multiple, or custom)
    /// - Returns: Gate sequence implementing the oracle
    /// - Precondition: qubits >= 1
    /// - Precondition: All target states must be < 2^qubits
    /// - Complexity: O(n * M) where n = qubits, M = marked states
    ///
    /// - SeeAlso: ``GroverOracle``
    /// - SeeAlso: ``groverDiffusion(qubits:)``
    @_optimize(speed)
    @_effects(readonly)
    static func groverOracle(qubits: Int, oracle: GroverOracle) -> [(gate: QuantumGate, qubits: [Int])] {
        ValidationUtilities.validatePositiveQubits(qubits)

        switch oracle {
        case let .singleTarget(target):
            let stateSpaceSize = 1 << qubits
            ValidationUtilities.validateIndexInBounds(target, bound: stateSpaceSize, name: "Target state")
            return buildSingleTargetOracle(qubits: qubits, target: target)

        case let .multipleTargets(targets):
            let stateSpaceSize = 1 << qubits
            for target in targets {
                ValidationUtilities.validateIndexInBounds(target, bound: stateSpaceSize, name: "Target state")
            }
            return buildMultipleTargetsOracle(qubits: qubits, targets: targets)

        case let .custom(gates):
            return gates
        }
    }

    /// Builds oracle for single target state.
    @_optimize(speed)
    @_effects(readonly)
    private static func buildSingleTargetOracle(qubits: Int, target: Int) -> [(gate: QuantumGate, qubits: [Int])] {
        var gates: [(gate: QuantumGate, qubits: [Int])] = []
        gates.reserveCapacity(4 * qubits)

        for qubit in 0 ..< qubits {
            if (target >> qubit) & 1 == 0 {
                gates.append((.pauliX, [qubit]))
            }
        }

        let mczGates = GroverDiffusion.buildMultiControlledZGates(qubits: qubits)
        gates.append(contentsOf: mczGates)

        for qubit in 0 ..< qubits {
            if (target >> qubit) & 1 == 0 {
                gates.append((.pauliX, [qubit]))
            }
        }

        return gates
    }

    /// Builds oracle for multiple target states.
    @_optimize(speed)
    @_effects(readonly)
    private static func buildMultipleTargetsOracle(qubits: Int, targets: [Int]) -> [(gate: QuantumGate, qubits: [Int])] {
        var gates: [(gate: QuantumGate, qubits: [Int])] = []

        for target in targets {
            let singleOracle = buildSingleTargetOracle(qubits: qubits, target: target)
            gates.append(contentsOf: singleOracle)
        }

        return gates
    }

    /// Computes optimal number of Grover iterations.
    ///
    /// The optimal number of iterations is floor(pi/4 * sqrt(N/M)) where
    /// N = 2^n is the search space size and M is the number of marked items.
    /// This maximizes the success probability for finding a marked state.
    ///
    /// **Example:**
    /// ```swift
    /// let optimal = QuantumCircuit.optimalGroverIterations(qubits: 4, markedItems: 1)
    /// print(optimal)  // 3 for 16-state search space with 1 target
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (search space = 2^qubits)
    ///   - markedItems: Number of target states (default: 1)
    /// - Returns: Optimal iteration count for maximum success probability
    /// - Precondition: qubits >= 1
    /// - Precondition: markedItems >= 1
    /// - Precondition: markedItems <= 2^qubits
    /// - Complexity: O(1)
    ///
    /// - SeeAlso: ``groverSearch(qubits:oracle:iterations:)``
    @_effects(readonly)
    static func optimalGroverIterations(qubits: Int, markedItems: Int = 1) -> Int {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validatePositiveInt(markedItems, name: "markedItems")

        let stateSpaceSize = 1 << qubits
        ValidationUtilities.validateUpperBound(markedItems, max: stateSpaceSize, name: "markedItems")

        let n = Double(stateSpaceSize)
        let m = Double(markedItems)
        let optimal = (Double.pi / 4.0) * sqrt(n / m)
        return max(1, Int(floor(optimal)))
    }

    /// Builds complete Grover search circuit.
    ///
    /// Creates a full Grover search circuit with uniform superposition initialization,
    /// followed by the specified number of oracle-diffusion iterations. If iterations
    /// is not specified, uses the optimal count floor(pi/4 * sqrt(N/M)).
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.groverSearch(
    ///     qubits: 4,
    ///     oracle: .singleTarget(7),
    ///     iterations: nil  // Use optimal
    /// )
    /// let state = circuit.execute()
    /// let (result, prob) = state.mostProbableState()  // Should be 7
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (search space = 2^qubits, max 10)
    ///   - oracle: Oracle specification for marking target states
    ///   - iterations: Number of iterations (nil = optimal)
    /// - Returns: Complete Grover search circuit
    /// - Precondition: qubits >= 1
    /// - Precondition: qubits <= 10
    /// - Complexity: O(k * n) gates where k = iterations, n = qubits
    ///
    /// - SeeAlso: ``optimalGroverIterations(qubits:markedItems:)``
    /// - SeeAlso: ``groverDiffusion(qubits:)``
    /// - SeeAlso: ``groverOracle(qubits:oracle:)``
    @_optimize(speed)
    @_eagerMove
    static func groverSearch(qubits: Int, oracle: GroverOracle, iterations: Int? = nil) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateAlgorithmQubitLimit(qubits, max: 10, algorithmName: "Grover search")

        let markedCount = oracle.markedCount
        let numIterations = iterations ?? optimalGroverIterations(qubits: qubits, markedItems: markedCount)

        var circuit = QuantumCircuit(qubits: qubits)

        for qubit in 0 ..< qubits {
            circuit.append(.hadamard, to: qubit)
        }

        let oracleGates = groverOracle(qubits: qubits, oracle: oracle)
        let diffusion = groverDiffusion(qubits: qubits)

        for _ in 0 ..< numIterations {
            for (gate, qubits) in oracleGates {
                circuit.append(gate, to: qubits)
            }

            for (gate, qubits) in diffusion.gates {
                circuit.append(gate, to: qubits)
            }
        }

        return circuit
    }
}

// MARK: - QuantumState Extensions

public extension QuantumState {
    /// Extracts Grover search result with analysis metrics.
    ///
    /// Analyzes the quantum state after Grover search to determine the
    /// most probable measurement outcome and compute success metrics
    /// including comparison to theoretical optimal.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.groverSearch(qubits: 4, oracle: .singleTarget(7))
    /// let state = circuit.execute()
    /// let result = state.groverResult(oracle: .singleTarget(7), iterations: 3, searchQubits: 4)
    /// print(result.isTarget)  // true if 7 was found
    /// print(result.successProbability)  // Probability of correct answer
    /// ```
    ///
    /// - Parameters:
    ///   - oracle: Oracle used in the search
    ///   - iterations: Number of iterations that were applied
    ///   - searchQubits: Number of qubits in the search space (excludes ancillas). If nil, uses state's qubit count.
    /// - Returns: GroverResult with measurement outcome and metrics
    /// - Complexity: O(2^n) for probability calculations
    ///
    /// - SeeAlso: ``GroverResult``
    /// - SeeAlso: ``QuantumCircuit/groverSearch(qubits:oracle:iterations:)``
    @_optimize(speed)
    @_effects(readonly)
    func groverResult(oracle: GroverOracle, iterations: Int, searchQubits: Int? = nil) -> GroverResult {
        let (measuredState, probability) = mostProbableState()

        let targets = oracle.targetStates
        let isTarget = targets.contains(measuredState)

        let markedCount = oracle.markedCount
        let effectiveSearchQubits = searchQubits ?? deriveSearchQubits(from: oracle)
        let optimalIterations = QuantumCircuit.optimalGroverIterations(
            qubits: effectiveSearchQubits,
            markedItems: markedCount,
        )

        var successProbability = 0.0
        for target in targets {
            if target < stateSpaceSize {
                successProbability += self.probability(of: target)
            }
        }

        if oracle.targetStates.isEmpty {
            successProbability = probability
        }

        return GroverResult(
            measuredState: measuredState,
            successProbability: successProbability,
            iterations: iterations,
            optimalIterations: optimalIterations,
            isTarget: isTarget,
        )
    }

    /// Derives the number of search qubits from the oracle by finding the minimum
    /// qubit count needed to represent all target states.
    @_optimize(speed)
    @_effects(readonly)
    private func deriveSearchQubits(from oracle: GroverOracle) -> Int {
        let targets = oracle.targetStates
        if targets.isEmpty {
            return qubits
        }

        // Safety: targets non-empty guaranteed by guard
        let maxTarget = targets.max()!

        if maxTarget == 0 {
            return 1
        }
        var bitsNeeded = 0
        var value = maxTarget
        while value > 0 {
            bitsNeeded += 1
            value >>= 1
        }
        return bitsNeeded
    }
}
