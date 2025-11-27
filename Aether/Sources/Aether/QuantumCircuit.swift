// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Single gate operation in a quantum circuit
///
/// Encapsulates a quantum gate with its target qubit indices and optional timestamp.
/// Operations are immutable value types that form the building blocks of quantum circuits.
///
/// **Qubit encoding pattern:**
/// - All gates: qubit indices are specified in the `qubits` array parameter
/// - Single-qubit gates: `qubits: [target]` (e.g., `qubits: [0]` for H gate on qubit 0)
/// - Multi-qubit gates: `qubits: [control, target]` or `qubits: [c1, c2, target]` for Toffoli
///
/// **Example:**
/// ```swift
/// let hadamard = GateOperation(gate: .hadamard, qubits: [0])
/// let cnot = GateOperation(gate: .cnot, qubits: [0, 1])  // control=0, target=1
/// let rotation = GateOperation(gate: .rotationY(theta: .pi / 4), qubits: [2], timestamp: 1.5)
/// ```
///
/// - SeeAlso: ``QuantumGate``, ``QuantumCircuit``
public struct GateOperation: Equatable, CustomStringConvertible, Sendable {
    /// Quantum gate to apply
    public let gate: QuantumGate

    /// Target qubit indices (empty for gates with encoded indices)
    public let qubits: [Int]

    /// Optional timestamp for animation or circuit scrubbing
    public let timestamp: Double?

    /// Creates a gate operation with specified gate, qubits, and optional timestamp
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply
    ///   - qubits: Target qubit indices (empty for gates with encoded indices)
    ///   - timestamp: Optional timestamp for animation or circuit scrubbing
    /// - Complexity: O(1)
    public init(gate: QuantumGate, qubits: [Int], timestamp: Double? = nil) {
        self.gate = gate
        self.qubits = qubits
        self.timestamp = timestamp
    }

    public var description: String {
        let qubitStr = qubits.isEmpty ? "" : " on qubits \(qubits)"
        if let ts = timestamp {
            return "\(gate)\(qubitStr) @ \(Self.formatTimestamp(ts))s"
        }
        return "\(gate)\(qubitStr)"
    }

    private static func formatTimestamp(_ time: Double) -> String {
        let formatted = String(format: "%.6f", time)
        let trimmed = formatted.replacingOccurrences(of: #"\.?0+$"#, with: "", options: .regularExpression)
        return trimmed.isEmpty ? "0" : trimmed
    }
}

/// Ordered sequence of quantum gates implementing quantum algorithms
///
/// Represents quantum computation as a series of gate operations transforming an initial state
/// |00...0⟩ through unitary operations to produce a final superposition state for measurement.
/// Circuits are mutable structs with value semantics, enabling safe composition and modification.
///
/// **When to use:**
/// - **QuantumCircuit**: Fixed gates with concrete parameters (Bell states, QFT, Grover)
/// - **ParameterizedQuantumCircuit**: Variational algorithms requiring symbolic parameters (VQE, QAOA)
///
/// **Key features:**
/// - Supports 1-30 qubits (30-qubit limit = 2³⁰ amplitudes ≈ 8GB memory)
/// - Auto-expands when gates reference higher indices (ancilla qubits initialized to |0⟩)
/// - Optional timestamping for animation and visualization workflows
///
/// **Performance:**
/// - Small circuits (≤10 qubits): Execute directly with CPU backend
/// - Large circuits (≥10 qubits): Use ``QuantumSimulator`` actor for GPU acceleration
/// - Step-through execution: Use ``execute(on:upToIndex:)`` for animation with external state caching
///
/// **Example:**
/// ```swift
/// var circuit = QuantumCircuit(numQubits: 2)
/// circuit.append(.hadamard, to: 0)
/// circuit.append(.cnot, to: [0, 1])
/// let bellState = circuit.execute()  // (|00⟩ + |11⟩)/√2
/// print(bellState.probability(of: 0b11))  // 0.5
/// ```
///
/// - SeeAlso: ``ParameterizedQuantumCircuit``, ``QuantumSimulator``, ``QuantumGate``, ``GateOperation``
public struct QuantumCircuit: Equatable, CustomStringConvertible, Sendable {
    public private(set) var gates: [GateOperation]
    public private(set) var numQubits: Int
    @usableFromInline var cachedMaxQubitUsed: Int

    /// Number of gates in the circuit
    ///
    /// - Complexity: O(1)
    public var gateCount: Int { gates.count }

    /// Whether the circuit contains no gates
    ///
    /// - Complexity: O(1)
    public var isEmpty: Bool { gates.isEmpty }

    // MARK: - Initialization

    /// Creates an empty quantum circuit with specified qubit count
    ///
    /// Initializes a circuit with no gate operations. Gates can be added via append or insert methods.
    /// Circuit auto-expands if gates reference qubits beyond initial size (up to 30 qubits maximum).
    ///
    /// - Parameter numQubits: Number of qubits (1-30)
    /// - Precondition: numQubits > 0 (validated by ``ValidationUtilities``)
    /// - Complexity: O(1)
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(numQubits: 3)
    /// circuit.append(.hadamard, to: 0)
    /// print(circuit.gateCount)  // 1
    /// ```
    public init(numQubits: Int) {
        ValidationUtilities.validatePositiveQubits(numQubits)
        self.numQubits = numQubits
        cachedMaxQubitUsed = numQubits - 1
        gates = []
    }

    /// Creates a circuit with predefined gate operations
    ///
    /// Useful for constructing circuits from previously saved operations or programmatic generation.
    /// Computes maximum qubit usage during initialization for ancilla detection.
    ///
    /// - Parameters:
    ///   - numQubits: Number of qubits (1-30)
    ///   - gates: Initial gate operations to include
    /// - Precondition: numQubits > 0 (validated by ``ValidationUtilities``)
    /// - Complexity: O(n) where n is number of operations
    ///
    /// **Example:**
    /// ```swift
    /// let ops = [
    ///     GateOperation(.hadamard, to: 0),
    ///     GateOperation(.cnot, to: [0, 1])
    /// ]
    /// let circuit = QuantumCircuit(numQubits: 2, gates: ops)
    /// ```
    public init(numQubits: Int, gates: [GateOperation]) {
        ValidationUtilities.validatePositiveQubits(numQubits)
        self.numQubits = numQubits
        self.gates = gates
        var maxQubit: Int = numQubits - 1
        for operation in gates {
            let gateMax: Int = operation.qubits.max() ?? -1
            if gateMax > maxQubit { maxQubit = gateMax }
        }
        cachedMaxQubitUsed = maxQubit
    }

    // MARK: - Building Methods

    /// Appends a gate to the specified qubits
    ///
    /// Adds the quantum gate to the circuit's gate sequence. Automatically expands
    /// the circuit's qubit count if the gate references indices beyond current size (up to 30 qubits).
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply
    ///   - qubits: Target qubit indices
    ///   - timestamp: Optional timestamp for animation or circuit scrubbing
    /// - Precondition: All qubit indices ≥ 0 (validated by ``ValidationUtilities``)
    /// - Precondition: Auto-expanded circuit size ≤ 30 qubits (validated by ``ValidationUtilities``)
    /// - Complexity: O(1) amortized (O(n) worst case when expanding qubit count)
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(numQubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    /// circuit.append(.rotationY(theta: .pi / 4), to: 1, timestamp: 1.5)
    /// ```
    public mutating func append(_ gate: QuantumGate, to qubits: [Int], timestamp: Double? = nil) {
        ValidationUtilities.validateNonNegativeQubits(qubits)

        let maxQubit: Int = qubits.max() ?? -1
        if maxQubit >= numQubits {
            let newNumQubits: Int = maxQubit + 1
            ValidationUtilities.validateMemoryLimit(newNumQubits)
            numQubits = newNumQubits
        }

        let operation = GateOperation(gate: gate, qubits: qubits, timestamp: timestamp)
        gates.append(operation)

        let operationMax: Int = operation.qubits.max() ?? -1
        if operationMax > cachedMaxQubitUsed { cachedMaxQubitUsed = operationMax }
    }

    /// Appends a gate to a single qubit (convenience method)
    ///
    /// Simplified interface for single-qubit gates. Equivalent to `append(_:to:[qubit])`.
    ///
    /// - Parameters:
    ///   - gate: Single-qubit gate to apply
    ///   - qubit: Target qubit index
    ///   - timestamp: Optional timestamp for animation
    /// - Precondition: qubit ≥ 0
    /// - Complexity: O(1) amortized
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(numQubits: 3)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.rotationZ(theta: .pi / 2), to: 1)
    /// ```
    public mutating func append(_ gate: QuantumGate, to qubit: Int, timestamp: Double? = nil) {
        append(gate, to: [qubit], timestamp: timestamp)
    }

    /// Inserts a gate at a specific position in the circuit
    ///
    /// Places the gate at the specified index, shifting subsequent gates forward.
    /// Useful for circuit optimization and debugging workflows.
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to insert
    ///   - qubits: Target qubit indices
    ///   - index: Position to insert at (0 = beginning, gateCount = end)
    ///   - timestamp: Optional timestamp for animation
    /// - Precondition: 0 ≤ index ≤ gateCount (validated by ``ValidationUtilities``)
    /// - Precondition: All qubits < numQubits (validated by ``ValidationUtilities``)
    /// - Complexity: O(n) where n is number of gates (array shift)
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(numQubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.hadamard, to: 1)
    /// circuit.insert(.cnot, to: [0, 1], at: 1)
    /// // Result: H(0), CNOT(0,1), H(1)
    /// ```
    public mutating func insert(_ gate: QuantumGate, to qubits: [Int], at index: Int, timestamp: Double? = nil) {
        ValidationUtilities.validateIndexInBounds(index, bound: gates.count, name: "Index")
        ValidationUtilities.validateOperationQubits(qubits, numQubits: numQubits)

        let operation = GateOperation(gate: gate, qubits: qubits, timestamp: timestamp)
        gates.insert(operation, at: index)

        let operationMax: Int = operation.qubits.max() ?? -1
        if operationMax > cachedMaxQubitUsed { cachedMaxQubitUsed = operationMax }
    }

    /// Removes the gate at the specified index
    ///
    /// Deletes the gate and shifts subsequent gates backward. Recomputes cached maximum
    /// qubit index if the removed gate was the highest-index gate.
    ///
    /// - Parameter index: Index of gate to remove (0-based)
    /// - Precondition: 0 ≤ index < gateCount (validated by ``ValidationUtilities``)
    /// - Complexity: O(n) where n is number of gates (array shift + potential recompute)
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(numQubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.pauliX, to: 1)
    /// circuit.remove(at: 0)  // Removes H gate, keeps X gate
    /// ```
    public mutating func remove(at index: Int) {
        ValidationUtilities.validateIndexInBounds(index, bound: gates.count, name: "Index")
        let removedMax: Int = gates[index].qubits.max() ?? -1
        gates.remove(at: index)

        if removedMax == cachedMaxQubitUsed { recomputeMaxQubitCache() }
    }

    /// Removes all gates from the circuit
    ///
    /// Clears the gate array while preserving the qubit count. Resets cached maximum qubit
    /// index to numQubits-1.
    ///
    /// - Complexity: O(1)
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(numQubits: 3)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.pauliX, to: 1)
    /// circuit.removeAllGates()
    /// print(circuit.isEmpty)  // true
    /// print(circuit.numQubits)  // 3
    /// ```
    public mutating func removeAllGates() {
        gates.removeAll()
        cachedMaxQubitUsed = numQubits - 1
    }

    @inline(__always)
    private mutating func recomputeMaxQubitCache() {
        var maxQubit: Int = numQubits - 1
        for operation in gates {
            let gateMax: Int = operation.qubits.max() ?? -1
            if gateMax > maxQubit { maxQubit = gateMax }
        }
        cachedMaxQubitUsed = maxQubit
    }

    // MARK: - Validation

    /// Highest qubit index referenced by any gate in circuit
    ///
    /// Used to detect ancilla qubits that may exceed logical circuit size.
    /// Value is cached and updated during append/insert/remove operations.
    ///
    /// - Returns: Highest qubit index, or numQubits-1 if no gates
    /// - Complexity: O(1) - cached value
    @_optimize(speed)
    @inlinable
    public var highestQubitIndex: Int { cachedMaxQubitUsed }

    // MARK: - Execution

    /// Expands quantum state with ancilla qubits when circuit operations exceed state size
    ///
    /// Ancilla qubits are initialized to |0⟩ and tensor-producted with the original state.
    /// Used internally by ``execute()`` methods to handle circuits that dynamically expand.
    ///
    /// - Parameters:
    ///   - state: Initial quantum state
    ///   - maxQubit: Maximum qubit index needed by circuit operations
    /// - Returns: Expanded state if maxQubit ≥ state.numQubits, otherwise original state
    /// - Complexity: O(2^(n+k)) where n = original qubits, k = ancilla qubits needed
    @_eagerMove
    static func expandStateForAncilla(_ state: QuantumState, maxQubit: Int) -> QuantumState {
        guard maxQubit >= state.numQubits else { return state }

        let numAncillaQubits: Int = maxQubit - state.numQubits + 1
        let expandedSize = 1 << (state.numQubits + numAncillaQubits)
        let originalSize: Int = state.stateSpaceSize

        let expandedAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: expandedSize) { buffer, count in
            for i in 0 ..< originalSize {
                buffer[i] = state.amplitudes[i]
            }
            for i in originalSize ..< expandedSize {
                buffer[i] = .zero
            }
            count = expandedSize
        }

        return QuantumState(numQubits: maxQubit + 1, amplitudes: expandedAmplitudes)
    }

    /// Executes the circuit on a custom initial quantum state
    ///
    /// Applies all gate operations sequentially to transform the input state through unitary evolution.
    /// Automatically expands the state with ancilla qubits (initialized to |0⟩) if circuit operations
    /// reference qubit indices beyond the initial state size.
    ///
    /// - Parameter initialState: Starting quantum state (must have numQubits matching circuit)
    /// - Returns: Final quantum state after applying all circuit operations
    /// - Precondition: initialState.numQubits == numQubits (validated by ``ValidationUtilities``)
    /// - Precondition: Circuit must be valid (validated internally)
    /// - Complexity: O(n x 2^q) where n = gate count, q = max qubit index (including ancilla)
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(numQubits: 2)
    /// circuit.append(.hadamard, to: 0)
    ///
    /// let initial = QuantumState(numQubits: 2, amplitudes: [
    ///     Complex(0.6, 0), Complex(0.8, 0), .zero, .zero
    /// ])
    /// let final = circuit.execute(on: initial)
    /// ```
    ///
    /// - SeeAlso: ``execute()`` for execution from ground state
    @_optimize(speed)
    @_eagerMove
    public func execute(on initialState: QuantumState) -> QuantumState {
        ValidationUtilities.validateStateQubitCount(initialState, required: numQubits)
        ValidationUtilities.validateCircuitOperations(gates, numQubits: numQubits)

        let maxQubit = highestQubitIndex
        var currentState = Self.expandStateForAncilla(initialState, maxQubit: maxQubit)

        for operation in gates {
            currentState = GateApplication.apply(
                operation.gate,
                to: operation.qubits,
                state: currentState
            )
        }

        return currentState
    }

    /// Executes the circuit starting from ground state |00...0⟩
    ///
    /// Primary execution method for quantum algorithms. Initializes all qubits to computational
    /// basis state |0⟩ and sequentially applies all circuit operations. For large circuits (≥10 qubits)
    /// or GPU acceleration needs, use ``QuantumSimulator`` actor instead.
    ///
    /// - Returns: Final quantum state after applying all circuit operations
    /// - Precondition: Circuit must be valid (validated internally)
    /// - Complexity: O(n x 2^q) where n = gate count, q = max qubit index
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(numQubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    /// let bellState = circuit.execute()  // (|00⟩ + |11⟩)/√2
    /// ```
    ///
    /// - SeeAlso: ``QuantumSimulator`` for GPU-accelerated execution, ``execute(on:)`` for custom initial states
    @_optimize(speed)
    @_eagerMove
    public func execute() -> QuantumState {
        let initialState = QuantumState(numQubits: numQubits)
        return execute(on: initialState)
    }

    /// Executes circuit up to a specific gate index for step-through visualization
    ///
    /// Applies only the first `upToIndex` operations, enabling incremental execution for animation
    /// workflows. External code should cache intermediate states at regular intervals for scrubbing.
    ///
    /// - Parameters:
    ///   - initialState: Starting quantum state
    ///   - upToIndex: Number of gates to execute (0 = no gates, gateCount = all gates)
    /// - Returns: Quantum state after executing operations [0..<upToIndex]
    /// - Precondition: initialState.numQubits == numQubits
    /// - Precondition: 0 ≤ upToIndex ≤ gateCount (validated by ``ValidationUtilities``)
    /// - Complexity: O(k x 2^q) where k = upToIndex, q = max qubit index
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(numQubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.pauliX, to: 1)
    /// circuit.append(.cnot, to: [0, 1])
    ///
    /// let initial = QuantumState(numQubits: 2)
    /// let afterH = circuit.execute(on: initial, upToIndex: 1)  // Just H gate
    /// let afterHX = circuit.execute(on: initial, upToIndex: 2)  // H + X gates
    /// ```
    @_optimize(speed)
    @_eagerMove
    public func execute(on initialState: QuantumState, upToIndex: Int) -> QuantumState {
        ValidationUtilities.validateStateQubitCount(initialState, required: numQubits)
        ValidationUtilities.validateUpToIndex(upToIndex, operationCount: gates.count)

        let maxQubit = highestQubitIndex
        var currentState = Self.expandStateForAncilla(initialState, maxQubit: maxQubit)

        for i in 0 ..< upToIndex {
            currentState = GateApplication.apply(
                gates[i].gate,
                to: gates[i].qubits,
                state: currentState
            )
        }

        return currentState
    }

    // MARK: - CustomStringConvertible

    /// String representation of the quantum circuit
    public var description: String {
        if gates.isEmpty { return "QuantumCircuit(\(numQubits) qubits, empty)" }

        var gateList = ""
        let limit = min(gates.count, 5)
        for i in 0 ..< limit {
            if i > 0 { gateList += ", " }
            gateList += gates[i].description
        }
        let suffix = gates.count > 5 ? ", ..." : ""

        return "QuantumCircuit(\(numQubits) qubits, \(gates.count) gates): \(gateList)\(suffix)"
    }
}

// MARK: - Pre-Built Circuits (Factory Methods)

public extension QuantumCircuit {
    /// Creates a Bell circuit: H₀ · CNOT₀₁
    ///
    /// Produces maximally entangled two-qubit state: (|00⟩ + |11⟩)/√2
    ///
    /// - Returns: 2-qubit circuit generating Bell state |Φ⁺⟩
    /// - Complexity: O(1) - 2 gate operations
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.bell()
    /// let state = circuit.execute()
    /// print(state.probability(of: 0b00))  // 0.5
    /// print(state.probability(of: 0b11))  // 0.5
    /// ```
    @_eagerMove
    static func bell() -> QuantumCircuit {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        return circuit
    }

    /// Creates a GHZ circuit: H₀ · CNOT₀₁ · CNOT₀₂ · ...
    ///
    /// Produces n-qubit maximally entangled Greenberger-Horne-Zeilinger state: (|00...0⟩ + |11...1⟩)/√2
    ///
    /// - Parameter numQubits: Number of qubits (minimum 2, default 3)
    /// - Returns: n-qubit circuit generating GHZ state
    /// - Precondition: numQubits ≥ 2 (validated by ``ValidationUtilities``)
    /// - Complexity: O(n) where n = numQubits
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.ghz(numQubits: 3)
    /// let state = circuit.execute()
    /// print(state.probability(of: 0b000))  // 0.5
    /// print(state.probability(of: 0b111))  // 0.5
    /// ```
    @_eagerMove
    static func ghz(numQubits: Int = 3) -> QuantumCircuit {
        ValidationUtilities.validateMinimumQubits(numQubits, min: 2, algorithmName: "GHZ")

        var circuit = QuantumCircuit(numQubits: numQubits)

        circuit.append(.hadamard, to: 0)

        for i in 1 ..< numQubits {
            circuit.append(.cnot, to: [0, i])
        }

        return circuit
    }

    /// Creates uniform superposition circuit: Apply H to all qubits
    ///
    /// Produces equal superposition over all basis states: (Σᵢ|i⟩)/√(2ⁿ)
    ///
    /// - Parameter numQubits: Number of qubits
    /// - Returns: Circuit that creates uniform superposition
    /// - Complexity: O(n) where n = numQubits
    @_eagerMove
    static func uniformSuperposition(numQubits: Int) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(numQubits)

        var circuit = QuantumCircuit(numQubits: numQubits)

        for i in 0 ..< numQubits {
            circuit.append(.hadamard, to: i)
        }

        return circuit
    }

    /// Creates Quantum Fourier Transform circuit for frequency domain transformation
    ///
    /// Implements the quantum analogue of the discrete Fourier transform: |j⟩ -> (1/√N)Σₖ exp(2πijk/N)|k⟩
    /// where N = 2ⁿ. QFT is a key component in Shor's algorithm and quantum phase estimation.
    ///
    /// **Algorithm structure:**
    /// - For each qubit i (from 0 to n-1):
    ///   - Apply Hadamard to qubit i
    ///   - Apply controlled-phase gates from qubits i+1 to n-1
    ///   - Phase angles: π/2, π/4, π/8, ... (decreasing powers of 2)
    /// - Reverse qubit order with SWAP gates
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.qft(numQubits: 3)
    /// let state = QuantumState(numQubits: 3)  // |000⟩
    /// let transformed = circuit.execute(on: state)
    /// // Result: uniform superposition (|000⟩ + |001⟩ + ... + |111⟩)/√8
    /// ```
    ///
    /// - Parameter numQubits: Number of qubits (typical range: 3-8, maximum: 16)
    /// - Returns: Circuit implementing QFT
    /// - Precondition: numQubits ≤ 16 (validated by ``ValidationUtilities``)
    /// - Complexity: O(n²) gates where n = numQubits
    ///
    /// - SeeAlso: ``inverseQFT(numQubits:)`` for inverse transformation
    @_optimize(speed)
    @_eagerMove
    static func qft(numQubits: Int) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(numQubits)
        ValidationUtilities.validateAlgorithmQubitLimit(numQubits, max: 16, algorithmName: "QFT")

        var circuit = QuantumCircuit(numQubits: numQubits)

        for target in 0 ..< numQubits {
            circuit.append(.hadamard, to: target)

            for control in (target + 1) ..< numQubits {
                let k = control - target + 1
                let theta = Double.pi / Double(1 << k)

                circuit.append(.controlledPhase(theta: theta), to: [control, target])
            }
        }

        let swapCount = numQubits / 2
        for i in 0 ..< swapCount {
            let j = numQubits - 1 - i
            circuit.append(.swap, to: [i, j])
        }

        return circuit
    }

    /// Creates inverse Quantum Fourier Transform circuit
    ///
    /// Reverses the QFT transformation, mapping frequency domain back to computational basis.
    /// Implements QFT† by applying gates in reverse order with negated phase angles.
    ///
    /// **Example:**
    /// ```swift
    /// let qftCircuit = QuantumCircuit.qft(numQubits: 3)
    /// let inverseCircuit = QuantumCircuit.inverseQFT(numQubits: 3)
    ///
    /// var combined = qftCircuit
    /// for gate in inverseCircuit.gates {
    ///     combined.append(gate.gate, to: gate.qubits)
    /// }
    /// // combined circuit is effectively identity (QFT · QFT† = I)
    /// ```
    ///
    /// - Parameter numQubits: Number of qubits
    /// - Returns: Circuit implementing inverse QFT
    /// - Complexity: O(n²) gates where n = numQubits
    ///
    /// - SeeAlso: ``qft(numQubits:)`` for forward transformation
    @_optimize(speed)
    @_eagerMove
    static func inverseQFT(numQubits: Int) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(numQubits)

        var circuit = QuantumCircuit(numQubits: numQubits)

        let swapCount = numQubits / 2
        for i in 0 ..< swapCount {
            let j = numQubits - 1 - i
            circuit.append(.swap, to: [i, j])
        }

        for target in (0 ..< numQubits).reversed() {
            for control in (target + 1 ..< numQubits).reversed() {
                let k = control - target + 1
                let theta = -Double.pi / Double(1 << k)

                circuit.append(.controlledPhase(theta: theta), to: [control, target])
            }

            circuit.append(.hadamard, to: target)
        }

        return circuit
    }

    /// Creates Grover search algorithm circuit
    ///
    /// Finds a marked item in an unsorted database with O(√N) quantum queries (classical: O(N)).
    /// Uses quantum amplitude amplification to quadratically speed up unstructured search.
    ///
    /// **Algorithm structure:**
    /// 1. Initialize uniform superposition (H on all qubits)
    /// 2. Repeat optimal iterations (π/4 x √(2ⁿ)):
    ///    a. Oracle: mark the target state with phase flip
    ///    b. Diffusion: amplify marked state amplitude via inversion about average
    ///
    /// - Parameters:
    ///   - numQubits: Number of qubits (search space size = 2ⁿ, maximum n=10)
    ///   - target: Target state to search for (basis state index, 0 ≤ target < 2ⁿ)
    ///   - iterations: Number of Grover iterations (defaults to optimal: ⌊π/4 x √(2ⁿ)⌋)
    /// - Returns: Circuit implementing Grover search
    /// - Precondition: numQubits ≤ 10 (validated by ``ValidationUtilities``)
    /// - Precondition: 0 ≤ target < 2ⁿ (validated by ``ValidationUtilities``)
    /// - Complexity: O(k x n x 2ⁿ) where k = iterations, n = numQubits (execution cost)
    @_optimize(speed)
    @_eagerMove
    static func grover(numQubits: Int, target: Int, iterations: Int? = nil) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(numQubits)
        ValidationUtilities.validateAlgorithmQubitLimit(numQubits, max: 10, algorithmName: "Grover")

        let stateSpaceSize = 1 << numQubits
        ValidationUtilities.validateIndexInBounds(target, bound: stateSpaceSize, name: "Target basis state")

        let optimalIterations: Int = iterations ?? Int((Double.pi / 4.0) * sqrt(Double(stateSpaceSize)))

        var circuit = QuantumCircuit(numQubits: numQubits)

        for qubit in 0 ..< numQubits {
            circuit.append(.hadamard, to: qubit)
        }

        for _ in 0 ..< optimalIterations {
            appendGroverOracle(to: &circuit, target: target, numQubits: numQubits)
            appendGroverDiffusion(to: &circuit, numQubits: numQubits)
        }

        return circuit
    }

    /// Append Grover oracle to circuit
    ///
    /// Implements the oracle function that flips the phase of the target state.
    /// Applies X gates to qubits where target bit is 0, then multi-controlled Z, then X again.
    ///
    /// - Parameters:
    ///   - circuit: Circuit to append oracle to
    ///   - target: Target state index to mark with phase flip
    ///   - numQubits: Total number of qubits in the system
    /// - Complexity: O(n) gates where n = numQubits
    private static func appendGroverOracle(to circuit: inout QuantumCircuit, target: Int, numQubits: Int) {
        for qubit in 0 ..< numQubits {
            if (target >> qubit) & 1 == 0 {
                circuit.append(.pauliX, to: qubit)
            }
        }

        appendMultiControlledZ(to: &circuit, numQubits: numQubits)

        for qubit in 0 ..< numQubits {
            if (target >> qubit) & 1 == 0 {
                circuit.append(.pauliX, to: qubit)
            }
        }
    }

    /// Append multi-controlled Z gate
    ///
    /// Flips phase when all qubits are |1⟩. Uses proper recursive decomposition (no approximations).
    ///
    /// - Parameters:
    ///   - circuit: Circuit to append to
    ///   - numQubits: Number of control qubits
    /// - Complexity: O(n) gates where n = numQubits
    private static func appendMultiControlledZ(to circuit: inout QuantumCircuit, numQubits: Int) {
        if numQubits == 1 {
            circuit.append(.pauliZ, to: 0)
        } else if numQubits == 2 {
            circuit.append(.controlledPhase(theta: .pi), to: [0, 1])
        } else if numQubits == 3 {
            circuit.append(.hadamard, to: 2)
            circuit.append(.toffoli, to: [0, 1, 2])
            circuit.append(.hadamard, to: 2)
        } else {
            let target = numQubits - 1
            circuit.append(.hadamard, to: target)

            appendMultiControlledX(to: &circuit, controls: Array(0 ..< numQubits - 1), target: target)

            circuit.append(.hadamard, to: target)
        }
    }

    /// Appends multi-controlled X (NOT) gate using ladder decomposition with ancilla qubits
    ///
    /// Implements Cⁿ(X) gate using Toffoli decomposition for arbitrary control count.
    /// Uses ancilla qubits automatically allocated beyond the maximum qubit index.
    ///
    /// **Decomposition:**
    /// - 0 controls: X gate
    /// - 1 control: CNOT gate
    /// - 2 controls: Toffoli gate
    /// - n controls (n≥3): Ladder of Toffoli gates with n-2 ancilla qubits
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(numQubits: 4)
    /// QuantumCircuit.appendMultiControlledX(to: &circuit, controls: [0, 1, 2], target: 3)
    /// // Applies X to qubit 3 only when qubits 0, 1, 2 are all |1⟩
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Circuit to append gates to
    ///   - controls: Control qubit indices (can be empty)
    ///   - target: Target qubit index
    /// - Complexity: O(n) gates where n = controls.count
    static func appendMultiControlledX(to circuit: inout QuantumCircuit, controls: [Int], target: Int) {
        let n = controls.count

        if n == 0 {
            circuit.append(.pauliX, to: target)
        } else if n == 1 {
            circuit.append(.cnot, to: [controls[0], target])
        } else if n == 2 {
            circuit.append(.toffoli, to: [controls[0], controls[1], target])
        } else {
            let maxUsedQubit: Int = max(controls.max()!, target)
            let firstAncilla: Int = maxUsedQubit + 1
            let numAncilla: Int = n - 2

            circuit.append(.toffoli, to: [controls[0], controls[1], firstAncilla])

            for i in 1 ..< numAncilla {
                circuit.append(.toffoli, to: [firstAncilla + i - 1, controls[i + 1], firstAncilla + i])
            }

            circuit.append(.toffoli, to: [firstAncilla + numAncilla - 1, controls[n - 1], target])

            for i in (1 ..< numAncilla).reversed() {
                circuit.append(.toffoli, to: [firstAncilla + i - 1, controls[i + 1], firstAncilla + i])
            }

            circuit.append(.toffoli, to: [controls[0], controls[1], firstAncilla])
        }
    }

    /// Appends multi-controlled Y gate using basis transformation
    ///
    /// Implements Cⁿ(Y) gate by decomposing Y = S†·X·S and applying multi-controlled X.
    /// More efficient than direct matrix decomposition for large control counts.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(numQubits: 3)
    /// QuantumCircuit.appendMultiControlledY(to: &circuit, controls: [0, 1], target: 2)
    /// // Applies Y to qubit 2 when both qubits 0 and 1 are |1⟩
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Circuit to append gates to
    ///   - controls: Control qubit indices
    ///   - target: Target qubit index
    /// - Complexity: O(n) gates where n = controls.count
    static func appendMultiControlledY(to circuit: inout QuantumCircuit, controls: [Int], target: Int) {
        let n = controls.count

        if n == 0 {
            circuit.append(.pauliY, to: target)
        } else if n == 1 {
            circuit.append(.cy, to: [controls[0], target])
        } else {
            circuit.append(.phase(angle: -Double.pi / 2.0), to: target)
            appendMultiControlledX(to: &circuit, controls: controls, target: target)
            circuit.append(.sGate, to: target)
        }
    }

    /// Appends multi-controlled Z gate using Hadamard sandwich
    ///
    /// Implements Cⁿ(Z) gate by decomposing Z = H·X·H and applying multi-controlled X.
    /// More efficient than Cⁿ(X) due to diagonal structure (no ancilla cleanup needed).
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(numQubits: 3)
    /// QuantumCircuit.appendMultiControlledZ(to: &circuit, controls: [0, 1], target: 2)
    /// // Applies Z to qubit 2 when both qubits 0 and 1 are |1⟩
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Circuit to append gates to
    ///   - controls: Control qubit indices
    ///   - target: Target qubit index
    /// - Complexity: O(n) gates where n = controls.count
    static func appendMultiControlledZ(to circuit: inout QuantumCircuit, controls: [Int], target: Int) {
        let n = controls.count

        if n == 0 {
            circuit.append(.pauliZ, to: target)
        } else if n == 1 {
            circuit.append(.cz, to: [controls[0], target])
        } else {
            circuit.append(.hadamard, to: target)
            appendMultiControlledX(to: &circuit, controls: controls, target: target)
            circuit.append(.hadamard, to: target)
        }
    }

    /// Appends multi-controlled arbitrary single-qubit unitary gate
    ///
    /// Applies any single-qubit gate U with n control qubits: |1⟩⊗ⁿ|ψ⟩ -> |1⟩⊗ⁿU|ψ⟩
    /// Optimizes decomposition based on gate type (Pauli gates use basis transformations).
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(numQubits: 4)
    /// let customGate = QuantumGate.rotationY(theta: .pi / 3)
    /// QuantumCircuit.appendMultiControlledU(
    ///     to: &circuit,
    ///     gate: customGate,
    ///     controls: [0, 1, 2],
    ///     target: 3
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Circuit to append gates to
    ///   - gate: Single-qubit gate to apply
    ///   - controls: Control qubit indices
    ///   - target: Target qubit index
    /// - Precondition: gate must be single-qubit (validated by ``ValidationUtilities``)
    /// - Complexity: O(n) gates where n = controls.count
    static func appendMultiControlledU(
        to circuit: inout QuantumCircuit,
        gate: QuantumGate,
        controls: [Int],
        target: Int
    ) {
        ValidationUtilities.validateControlledGateIsSingleQubit(gate.qubitsRequired)

        let n = controls.count

        if n == 0 {
            circuit.append(gate, to: target)
        } else {
            switch gate {
            case .pauliX:
                appendMultiControlledX(to: &circuit, controls: controls, target: target)
            case .pauliY:
                appendMultiControlledY(to: &circuit, controls: controls, target: target)
            case .pauliZ:
                appendMultiControlledZ(to: &circuit, controls: controls, target: target)
            case .hadamard:
                circuit.append(.rotationY(theta: .pi / 4), to: target)
                appendMultiControlledZ(to: &circuit, controls: controls, target: target)
                circuit.append(.rotationY(theta: -.pi / 4), to: target)
            default:
                circuit.append(gate, to: target)
                appendMultiControlledX(to: &circuit, controls: controls, target: target)

                let matrix = gate.matrix()
                let adjointMatrix = MatrixUtilities.hermitianConjugate(matrix)
                circuit.append(.customSingleQubit(matrix: adjointMatrix), to: target)

                appendMultiControlledX(to: &circuit, controls: controls, target: target)
                circuit.append(gate, to: target)
            }
        }
    }

    /// Append Grover diffusion operator to circuit
    ///
    /// Implements the diffusion operator 2|s⟩⟨s| - I that amplifies marked state amplitudes.
    /// Sequence: H on all -> X on all -> multi-controlled Z -> X on all -> H on all.
    ///
    /// - Parameters:
    ///   - circuit: Circuit to append diffusion operator to
    ///   - numQubits: Total number of qubits in the system
    /// - Complexity: O(n) gates where n = numQubits
    private static func appendGroverDiffusion(to circuit: inout QuantumCircuit, numQubits: Int) {
        for qubit in 0 ..< numQubits {
            circuit.append(.hadamard, to: qubit)
        }

        for qubit in 0 ..< numQubits {
            circuit.append(.pauliX, to: qubit)
        }

        appendMultiControlledZ(to: &circuit, numQubits: numQubits)

        for qubit in 0 ..< numQubits {
            circuit.append(.pauliX, to: qubit)
        }

        for qubit in 0 ..< numQubits {
            circuit.append(.hadamard, to: qubit)
        }
    }

    /// Creates quantum annealing circuit for optimization problems
    ///
    /// Implements adiabatic evolution from simple transverse-field Hamiltonian to problem Hamiltonian.
    /// Used for combinatorial optimization by encoding problems as Ising models and finding ground states.
    ///
    /// **Algorithm structure:**
    /// 1. Initialize in superposition (transverse field Hamiltonian H₀)
    /// 2. Gradually interpolate to problem Hamiltonian: H(t) = (1-t)H₀ + tHₚ
    /// 3. Evolve through discrete time steps with Trotter approximation
    /// 4. Measure to find configuration corresponding to optimal solution
    ///
    /// **Example:**
    /// ```swift
    /// // MaxCut on triangle graph
    /// let problem = QuantumCircuit.IsingProblem(fromDictionary: [
    ///     "0-1": -0.5, "1-2": -0.5, "0-2": -0.5
    /// ], numQubits: 3)
    /// let circuit = QuantumCircuit.annealing(numQubits: 3, problem: problem, annealingSteps: 20)
    /// let state = circuit.execute()
    /// // Measure to find approximate solution to MaxCut
    /// ```
    ///
    /// - Parameters:
    ///   - numQubits: Number of qubits (problem variables, maximum: 8)
    ///   - problem: Ising model defining optimization problem (local fields + couplings)
    ///   - annealingSteps: Number of time steps in annealing schedule (default: 20)
    /// - Returns: Circuit demonstrating quantum annealing process
    /// - Precondition: numQubits ≤ 8 (validated by ``ValidationUtilities``)
    /// - Complexity: O(s x n²) gates where s = annealingSteps, n = numQubits
    ///
    /// - SeeAlso: ``IsingProblem``, ``annealing(numQubits:couplings:annealingSteps:)`` for convenience constructor
    @_eagerMove
    static func annealing(numQubits: Int, problem: IsingProblem, annealingSteps: Int = 20) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(numQubits)
        ValidationUtilities.validatePositiveInt(annealingSteps, name: "annealingSteps")
        ValidationUtilities.validateAlgorithmQubitLimit(numQubits, max: 8, algorithmName: "Annealing")

        var circuit = QuantumCircuit(numQubits: numQubits)

        for qubit in 0 ..< numQubits {
            circuit.append(.hadamard, to: qubit)
        }

        for step in 0 ..< annealingSteps {
            let time = Double(step) / Double(annealingSteps - 1)

            let transverseStrength = 1.0 - time
            for qubit in 0 ..< numQubits {
                let angle = 2.0 * transverseStrength * problem.transverseField[qubit]
                circuit.append(.rotationX(theta: angle), to: qubit)
            }

            let problemStrength = time

            for qubit in 0 ..< numQubits {
                let angle = 2.0 * problemStrength * problem.localFields[qubit]
                circuit.append(.rotationZ(theta: angle), to: qubit)
            }

            for i in 0 ..< numQubits {
                for j in (i + 1) ..< numQubits {
                    let coupling = problem.couplings[i][j]
                    if abs(coupling) > 1e-10 {
                        let angle = 2.0 * problemStrength * coupling
                        appendZZCoupling(to: &circuit, qubit1: i, qubit2: j, angle: angle)
                    }
                }
            }
        }

        return circuit
    }

    /// Creates annealing circuit using simplified coupling dictionary specification
    ///
    /// Convenience constructor that accepts string-keyed dictionary for Ising model specification.
    /// More compact than manually constructing ``IsingProblem`` for simple problems.
    ///
    /// **Dictionary format:**
    /// - Single qubit keys ("0", "1") specify local fields hᵢ on individual qubits
    /// - Two qubit keys ("01", "0-1", "0,1") specify ZZ couplings Jᵢⱼ between qubit pairs
    ///
    /// **Example:**
    /// ```swift
    /// // MaxCut problem on 3-qubit triangle graph
    /// let circuit = QuantumCircuit.annealing(
    ///     numQubits: 3,
    ///     couplings: ["0-1": -0.5, "1-2": -0.5, "0-2": -0.5],
    ///     annealingSteps: 20
    /// )
    ///
    /// // Include local fields (compact notation also works)
    /// let circuit2 = QuantumCircuit.annealing(
    ///     numQubits: 3,
    ///     couplings: ["0": -0.3, "1": 0.2, "01": 0.5, "12": 0.5],
    ///     annealingSteps: 20
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - numQubits: Number of qubits (problem variables)
    ///   - couplings: Dictionary mapping qubit indices/pairs to coupling strengths
    ///   - annealingSteps: Number of time steps in annealing schedule
    /// - Returns: Annealing circuit for the specified Ising problem
    ///
    /// - SeeAlso: ``annealing(numQubits:problem:annealingSteps:)`` for explicit IsingProblem construction
    @_eagerMove
    static func annealing(numQubits: Int, couplings: [String: Double], annealingSteps: Int = 20) -> QuantumCircuit {
        let problem = IsingProblem(fromDictionary: couplings, numQubits: numQubits)
        return annealing(numQubits: numQubits, problem: problem, annealingSteps: annealingSteps)
    }

    /// Ising model problem definition for quantum annealing
    ///
    /// Encodes optimization problems as Ising Hamiltonians: H = Σᵢ hᵢZᵢ + Σ_{i<j} JᵢⱼZᵢZⱼ
    /// Used in quantum annealing to find ground states corresponding to optimal solutions.
    ///
    /// **Components:**
    /// - Local fields (hᵢ): Single-qubit energy terms
    /// - Couplings (Jᵢⱼ): Two-qubit interaction strengths
    /// - Transverse field: Enables quantum tunneling during annealing
    ///
    /// **Example:**
    /// ```swift
    /// // MaxCut on triangle graph
    /// let problem = QuantumCircuit.IsingProblem(fromDictionary: [
    ///     "0-1": -0.5, "1-2": -0.5, "0-2": -0.5
    /// ], numQubits: 3)
    /// let circuit = QuantumCircuit.annealing(numQubits: 3, problem: problem)
    /// ```
    struct IsingProblem {
        public let localFields: [Double]
        public let couplings: [[Double]]
        public let transverseField: [Double]

        /// Create Ising model problem instance
        /// - Parameters:
        ///   - localFields: Local magnetic field coefficients for each qubit
        ///   - couplings: Symmetric coupling matrix between qubits
        ///   - transverseField: Transverse field strengths (defaults to uniform field of 1.0)
        public init(localFields: [Double], couplings: [[Double]], transverseField: [Double]? = nil) {
            let n = localFields.count
            ValidationUtilities.validateSquareMatrix(couplings, name: "Couplings matrix")

            self.localFields = localFields
            self.couplings = couplings
            self.transverseField = transverseField ?? [Double](repeating: 1.0, count: n)
        }

        /// Create Ising model from dictionary specification
        ///
        /// Parses a dictionary where keys specify qubit interactions:
        /// - Single qubit keys ("0", "1", ...) specify local fields h_i
        /// - Two qubit keys ("01", "0-1", "0,1", ...) specify couplings J_ij
        ///
        /// The Ising Hamiltonian is: H = Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j
        ///
        /// - Parameters:
        ///   - dictionary: Mapping of qubit indices/pairs to coupling strengths
        ///   - numQubits: Total number of qubits in the system
        public init(fromDictionary dictionary: [String: Double], numQubits: Int) {
            ValidationUtilities.validatePositiveQubits(numQubits)

            var localFieldsBuffer = [Double](unsafeUninitializedCapacity: numQubits) { buffer, count in
                buffer.initialize(repeating: 0.0)
                count = numQubits
            }

            var couplingsBuffer = [[Double]](unsafeUninitializedCapacity: numQubits) { buffer, count in
                for i in 0 ..< numQubits {
                    buffer[i] = [Double](unsafeUninitializedCapacity: numQubits) { innerBuffer, innerCount in
                        innerBuffer.initialize(repeating: 0.0)
                        innerCount = numQubits
                    }
                }
                count = numQubits
            }

            for (key, value) in dictionary {
                let qubits = Self.parseQubitIndices(from: key)
                ValidationUtilities.validateCouplingKeyFormat(qubits.count, key: key)

                if qubits.count == 1 {
                    let qubit = qubits[0]
                    ValidationUtilities.validateQubitIndex(qubit, numQubits: numQubits)
                    localFieldsBuffer[qubit] = value
                } else {
                    let qubit1 = qubits[0]
                    let qubit2 = qubits[1]
                    ValidationUtilities.validateQubitIndex(qubit1, numQubits: numQubits)
                    ValidationUtilities.validateQubitIndex(qubit2, numQubits: numQubits)
                    ValidationUtilities.validateDistinctVertices(qubit1, qubit2)
                    couplingsBuffer[qubit1][qubit2] = value
                    couplingsBuffer[qubit2][qubit1] = value
                }
            }

            localFields = localFieldsBuffer
            couplings = couplingsBuffer
            transverseField = [Double](unsafeUninitializedCapacity: numQubits) { buffer, count in
                buffer.initialize(repeating: 1.0)
                count = numQubits
            }
        }

        /// Parse qubit indices from a coupling key string
        ///
        /// Handles multiple formats:
        /// - Single digit: "0", "1" -> [0], [1] (local fields)
        /// - Two adjacent digits: "01", "12" -> [0, 1], [1, 2] (couplings)
        /// - Separated: "0-1", "1,2", "10-12" -> [0, 1], [1, 2], [10, 12]
        private static func parseQubitIndices(from key: String) -> [Int] {
            ValidationUtilities.validateNonEmptyString(key, name: "Coupling key")

            let hasSeparator = key.contains { !$0.isNumber }

            if hasSeparator {
                return key.split { !$0.isNumber }.compactMap { Int($0) }
            } else if key.count == 2 {
                return key.compactMap(\.wholeNumberValue)
            } else {
                // Safe: validated non-empty string containing only digits at this point
                return [Int(key)!]
            }
        }

        /// Creates Ising problem for quadratic function minimization
        ///
        /// Encodes the quadratic objective function f(x) = x² - 2x as an Ising Hamiltonian,
        /// where x is represented in binary as x = Σᵢ 2ⁱzᵢ with zᵢ ∈ {0,1}.
        /// Annealing finds minimum by mapping to ground state of corresponding Ising model.
        ///
        /// **Example:**
        /// ```swift
        /// let problem = QuantumCircuit.IsingProblem.quadraticMinimum(numQubits: 4)
        /// let circuit = QuantumCircuit.annealing(numQubits: 4, problem: problem)
        /// // Annealing finds x=1 (binary 0001), minimum of x²-2x
        /// ```
        ///
        /// - Parameter numQubits: Number of qubits for binary encoding (minimum: 2, default: 4)
        /// - Returns: IsingProblem encoding quadratic minimization
        /// - Precondition: numQubits ≥ 2 (validated by ``ValidationUtilities``)
        public static func quadraticMinimum(numQubits: Int = 4) -> IsingProblem {
            ValidationUtilities.validateMinimumQubits(numQubits, min: 2, algorithmName: "Quadratic minimum")

            let localFields = [Double](unsafeUninitializedCapacity: numQubits) { buffer, count in
                for i in 0 ..< numQubits {
                    buffer[i] = -1.0 * Double(1 << i)
                }
                count = numQubits
            }

            let couplings = [[Double]](unsafeUninitializedCapacity: numQubits) { buffer, count in
                for i in 0 ..< numQubits {
                    buffer[i] = [Double](unsafeUninitializedCapacity: numQubits) { innerBuffer, innerCount in
                        innerBuffer.initialize(repeating: 0.0)
                        innerCount = numQubits
                    }
                    if i < numQubits - 1 {
                        buffer[i][i + 1] = 0.5
                    }
                }
                count = numQubits
            }

            return IsingProblem(localFields: localFields, couplings: couplings)
        }
    }

    /// Append ZZ coupling between two qubits
    ///
    /// Implements exp(-iθ Z₁Z₂) using CNOT decomposition: CNOT₁₂ · RZ₂(2θ) · CNOT₁₂
    ///
    /// - Parameters:
    ///   - circuit: Circuit to append coupling to
    ///   - qubit1: First qubit index
    ///   - qubit2: Second qubit index
    ///   - angle: Coupling strength θ
    /// - Complexity: O(1) - three gates
    private static func appendZZCoupling(to circuit: inout QuantumCircuit, qubit1: Int, qubit2: Int, angle: Double) {
        circuit.append(.cnot, to: [qubit1, qubit2])
        circuit.append(.rotationZ(theta: 2.0 * angle), to: qubit2)
        circuit.append(.cnot, to: [qubit1, qubit2])
    }
}
