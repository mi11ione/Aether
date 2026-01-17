// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Single gate operation in a quantum circuit combining gate, target qubits, and optional timestamp.
///
/// Qubit indices are specified in the `qubits` array: single-qubit gates use `[target]`,
/// two-qubit gates use `[control, target]`, and Toffoli uses `[c1, c2, target]`.
///
/// **Example:**
/// ```swift
/// let hadamard = Gate(.hadamard, to: 0)
/// let cnot = Gate(.cnot, to: [0, 1])
/// let rotation = Gate(.rotationY(.pi / 4), to: 2, timestamp: 1.5)
/// ```
///
/// - SeeAlso: ``QuantumGate``
/// - SeeAlso: ``QuantumCircuit``
public struct Gate: Equatable, Hashable, CustomStringConvertible, Sendable {
    /// Quantum gate to apply
    public let gate: QuantumGate

    /// Target qubit indices
    public let qubits: [Int]

    /// Optional timestamp for animation or circuit scrubbing
    public let timestamp: Double?

    /// Creates a gate operation with specified gate and qubit array
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply
    ///   - qubits: Target qubit indices
    ///   - timestamp: Optional timestamp for animation or circuit scrubbing
    /// - Complexity: O(1)
    ///
    /// **Example:**
    /// ```swift
    /// let cnot = Gate(.cnot, to: [0, 1])
    /// let toffoli = Gate(.toffoli, to: [0, 1, 2], timestamp: 1.5)
    /// ```
    public init(_ gate: QuantumGate, to qubits: [Int], timestamp: Double? = nil) {
        self.gate = gate
        self.qubits = qubits
        self.timestamp = timestamp
    }

    /// Creates a single-qubit gate operation
    ///
    /// Convenience initializer for single-qubit gates that matches circuit.append API pattern.
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply
    ///   - qubit: Target qubit index
    ///   - timestamp: Optional timestamp for animation or circuit scrubbing
    /// - Complexity: O(1)
    ///
    /// **Example:**
    /// ```swift
    /// let hadamard = Gate(.hadamard, to: 0)
    /// let rotation = Gate(.rotationY(.pi/4), to: 2, timestamp: 0.5)
    /// ```
    public init(_ gate: QuantumGate, to qubit: Int, timestamp: Double? = nil) {
        self.gate = gate
        qubits = [qubit]
        self.timestamp = timestamp
    }

    public var description: String {
        let qubitStr = qubits.isEmpty ? "" : " on qubits \(qubits)"
        if let ts = timestamp {
            return "\(gate)\(qubitStr) @ \(Self.formatTimestamp(ts))s"
        }
        return "\(gate)\(qubitStr)"
    }

    /// Formats timestamp with trailing zero removal for clean display
    ///
    /// Produces minimal decimal representation: 0.5 (not 0.500000), 0 (not 0.000000)
    private static func formatTimestamp(_ time: Double) -> String {
        let formatted = String(format: "%.6f", time)
        let trimmed = formatted.replacingOccurrences(of: #"\.?0+$"#, with: "", options: .regularExpression)
        return trimmed
    }
}

/// Ordered sequence of quantum gates transforming |00...0⟩ through unitary operations.
///
/// Supports 1-30 qubits with automatic expansion when gates reference higher indices (ancilla
/// qubits initialized to |0⟩). Handles both concrete values and symbolic ``Parameter`` instances
/// for variational algorithms. Small circuits execute directly on CPU; use ``QuantumSimulator``
/// for GPU acceleration on circuits with 10+ qubits.
///
/// **Example:**
/// ```swift
/// // Concrete parameters
/// var circuit = QuantumCircuit(qubits: 2)
/// circuit.append(.hadamard, to: 0)
/// circuit.append(.cnot, to: [0, 1])
/// let state = circuit.execute()
///
/// // Symbolic parameters for variational circuits
/// let theta = Parameter(name: "theta")
/// circuit.append(.rotationY(theta), to: 0)
/// let bound = circuit.binding(["theta": 0.5])
/// let boundState = bound.execute()
/// ```
///
/// - SeeAlso: ``QuantumSimulator`` for GPU-accelerated execution
/// - SeeAlso: ``QuantumGate`` for gate definitions
/// - SeeAlso: ``Parameter`` for symbolic parameters
/// - SeeAlso: ``HardwareEfficientAnsatz`` for variational circuits
public struct QuantumCircuit: Equatable, Hashable, CustomStringConvertible, Sendable {
    public private(set) var gates: [Gate]
    public private(set) var qubits: Int
    @usableFromInline var cachedMaxQubitUsed: Int

    /// Number of gates in the circuit
    ///
    /// - Complexity: O(1)
    public var count: Int { gates.count }

    /// Whether the circuit contains no gates
    ///
    /// - Complexity: O(1)
    public var isEmpty: Bool { gates.count == 0 }

    // MARK: - Initialization

    /// Creates an empty quantum circuit with specified qubit count
    ///
    /// Initializes a circuit with no gate operations. Gates can be added via append or insert methods.
    /// Circuit auto-expands if gates reference qubits beyond initial size (up to 30 qubits maximum).
    ///
    /// - Parameter qubits: Number of qubits (1-30)
    /// - Precondition: qubits > 0
    /// - Complexity: O(1)
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 3)
    /// circuit.append(.hadamard, to: 0)
    /// print(circuit.count)  // 1
    /// ```
    public init(qubits: Int) {
        ValidationUtilities.validatePositiveQubits(qubits)
        self.qubits = qubits
        cachedMaxQubitUsed = qubits - 1
        gates = []
    }

    /// Creates a circuit with predefined gate operations
    ///
    /// Useful for constructing circuits from previously saved operations or programmatic generation.
    /// Computes maximum qubit usage during initialization for ancilla detection.
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (1-30)
    ///   - gates: Initial gate operations to include
    /// - Precondition: qubits > 0
    /// - Complexity: O(n) where n is number of operations
    ///
    /// **Example:**
    /// ```swift
    /// let ops = [
    ///     Gate(.hadamard, to: 0),
    ///     Gate(.cnot, to: [0, 1])
    /// ]
    /// let circuit = QuantumCircuit(qubits: 2, gates: ops)
    /// ```
    public init(qubits: Int, gates: [Gate]) {
        ValidationUtilities.validatePositiveQubits(qubits)
        self.qubits = qubits
        self.gates = gates
        var maxQubit: Int = qubits - 1
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
    /// Supports both concrete and symbolic parameters via ``QuantumGate``'s unified design.
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply (concrete or symbolic)
    ///   - qubits: Target qubit indices
    ///   - timestamp: Optional timestamp for animation or circuit scrubbing
    /// - Precondition: All qubit indices ≥ 0
    ///   - Precondition: Auto-expanded circuit size ≤ 30 qubits
    /// - Complexity: O(1) amortized (O(n) worst case when expanding qubit count)
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.rotationY(.pi / 4), to: 1)
    ///
    /// let theta = Parameter(name: "theta")
    /// circuit.append(.rotationY(theta), to: 0)
    /// ```
    public mutating func append(_ gate: QuantumGate, to qubits: [Int], timestamp: Double? = nil) {
        ValidationUtilities.validateNonNegativeQubits(qubits)

        let maxQubit: Int = qubits.max() ?? -1
        if maxQubit >= self.qubits {
            let newqubits: Int = maxQubit + 1
            ValidationUtilities.validateMemoryLimit(newqubits)
            self.qubits = newqubits
        }

        let operation = Gate(gate, to: qubits, timestamp: timestamp)
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
    /// var circuit = QuantumCircuit(qubits: 3)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.rotationZ(.pi / 2), to: 1)
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
    ///   - index: Position to insert at (0 = beginning, count = end)
    ///   - timestamp: Optional timestamp for animation
    /// - Precondition: 0 ≤ index ≤ count
    /// - Precondition: All qubits < qubits
    /// - Complexity: O(n) where n is number of gates (array shift)
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.hadamard, to: 1)
    /// circuit.insert(.cnot, to: [0, 1], at: 1)
    /// // Result: H(0), CNOT(0,1), H(1)
    /// ```
    public mutating func insert(_ gate: QuantumGate, to qubits: [Int], at index: Int, timestamp: Double? = nil) {
        ValidationUtilities.validateIndexInBounds(index, bound: gates.count, name: "Index")
        ValidationUtilities.validateOperationQubits(qubits, numQubits: self.qubits)

        let operation = Gate(gate, to: qubits, timestamp: timestamp)
        gates.insert(operation, at: index)
    }

    /// Inserts a gate at a specific position in the circuit (single-qubit convenience)
    ///
    /// Simplified interface for inserting single-qubit gates. Equivalent to `insert(_:at:[qubit], index:)`.
    ///
    /// - Parameters:
    ///   - gate: Single-qubit gate to insert
    ///   - qubit: Target qubit index
    ///   - index: Position to insert at (0 = beginning, count = end)
    ///   - timestamp: Optional timestamp for animation
    /// - Precondition: 0 ≤ index ≤ count
    /// - Precondition: qubit < qubits
    /// - Complexity: O(n) where n is number of gates
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.hadamard, to: 1)
    /// circuit.insert(.pauliX, to: 0, at: 1)
    /// // Result: H(0), X(0), H(1)
    /// ```
    public mutating func insert(_ gate: QuantumGate, to qubit: Int, at index: Int, timestamp: Double? = nil) {
        insert(gate, to: [qubit], at: index, timestamp: timestamp)
    }

    /// Removes the gate at the specified index
    ///
    /// Deletes the gate and shifts subsequent gates backward. Recomputes cached maximum
    /// qubit index if the removed gate was the highest-index gate.
    ///
    /// - Parameter index: Index of gate to remove (0-based)
    /// - Precondition: 0 ≤ index < count
    /// - Complexity: O(n) where n is number of gates (array shift + potential recompute)
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
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
    /// index to qubits-1.
    ///
    /// - Complexity: O(1)
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 3)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.pauliX, to: 1)
    /// circuit.removeAll()
    /// print(circuit.isEmpty)  // true
    /// print(circuit.qubits)  // 3
    /// ```
    public mutating func removeAll() {
        gates.removeAll()
        cachedMaxQubitUsed = qubits - 1
    }

    @inline(__always)
    private mutating func recomputeMaxQubitCache() {
        var maxQubit: Int = qubits - 1
        for operation in gates {
            let gateMax: Int = operation.qubits.max() ?? -1
            if gateMax > maxQubit { maxQubit = gateMax }
        }
        cachedMaxQubitUsed = maxQubit
    }

    // MARK: - Introspection

    /// Highest qubit index referenced by any gate in circuit
    ///
    /// Used to detect ancilla qubits that may exceed logical circuit size.
    /// Value is cached and updated during append/insert/remove operations.
    ///
    /// - Returns: Highest qubit index, or qubits-1 if no gates
    /// - Complexity: O(1) - cached value
    @_optimize(speed)
    @inlinable
    public var highestQubitIndex: Int { cachedMaxQubitUsed }

    /// Symbolic parameters in registration order
    ///
    /// Extracts all ``Parameter`` instances from gates, maintaining order of first appearance.
    /// Empty array for circuits with only concrete parameters. Used for variational algorithms
    /// to determine parameter count and names for binding.
    ///
    /// - Returns: Ordered array of distinct parameters
    /// - Complexity: O(n x p) where n = gate count, p = parameters per gate (typically ≤3)
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// let theta = Parameter(name: "theta")
    /// let phi = Parameter(name: "phi")
    /// circuit.append(.rotationY(theta), to: 0)
    /// circuit.append(.rotationZ(phi), to: 1)
    /// circuit.append(.rotationY(theta), to: 1)
    ///
    /// print(circuit.parameters.count)  // 2 (theta, phi)
    /// print(circuit.parameters.map(\.name))  // ["theta", "phi"]
    /// ```
    ///
    /// - SeeAlso: ``parameterCount`` for count-only query
    @_optimize(speed)
    public var parameters: [Parameter] {
        var seen = Set<String>()
        var ordered: [Parameter] = []

        for operation in gates {
            for param in operation.gate.parameters() {
                if !seen.contains(param.name) {
                    ordered.append(param)
                    seen.insert(param.name)
                }
            }
        }

        return ordered
    }

    /// Number of distinct symbolic parameters in circuit
    ///
    /// Count of unique parameters requiring binding before execution.
    /// Zero for circuits with only concrete parameters.
    ///
    /// - Returns: Count of distinct parameters
    /// - Complexity: O(n x p) where n = gate count, p = parameters per gate
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// print(circuit.parameterCount)  // 0
    ///
    /// let theta = Parameter(name: "theta")
    /// circuit.append(.rotationY(theta), to: 1)
    /// print(circuit.parameterCount)  // 1
    /// ```
    @inlinable
    public var parameterCount: Int { parameters.count }

    // MARK: - Parameter Binding

    /// Binds parameters using name-value dictionary
    ///
    /// Substitutes all symbolic parameters with concrete values to produce
    /// executable quantum circuit. All parameters must have bindings. Partial
    /// binding returns circuit that may still contain symbolic parameters.
    ///
    /// **Example:**
    /// ```swift
    /// var ansatz = QuantumCircuit(qubits: 2)
    /// let theta = Parameter(name: "theta")
    /// ansatz.append(.rotationY(theta), to: 0)
    /// ansatz.append(.cnot, to: [0, 1])
    ///
    /// let bound = ansatz.binding(["theta": Double.pi / 4])
    /// let state = bound.execute()
    /// ```
    ///
    /// - Parameter values: Dictionary mapping parameter names to numerical values
    /// - Returns: Circuit with parameters substituted (may contain remaining symbolic parameters)
    /// - Precondition: All circuit parameters must have bindings
    /// - Complexity: O(n) where n = gate count
    ///
    /// - SeeAlso: ``bound(with:)`` for array-based interface
    @_optimize(speed)
    @_eagerMove
    public func binding(_ values: [String: Double]) -> QuantumCircuit {
        let params = parameters
        var paramSet = Set<String>()
        for param in params {
            paramSet.insert(param.name)
        }

        ValidationUtilities.validateCompleteParameterBindings(values, parameters: params, parameterSet: paramSet)

        var bound = QuantumCircuit(qubits: qubits)

        for operation in gates {
            let boundGate = operation.gate.bound(with: values)
            bound.append(boundGate, to: operation.qubits, timestamp: operation.timestamp)
        }

        return bound
    }

    /// Binds parameters using numerical vector
    ///
    /// Array-based interface where parameters are bound in registration order
    /// (order of first appearance in circuit). Preferred for gradient-based
    /// optimization where parameter names are less important than ordering.
    ///
    /// **Example:**
    /// ```swift
    /// var ansatz = QuantumCircuit(qubits: 3)
    /// for i in 0..<3 {
    ///     let param = Parameter(name: "theta_\(i)")
    ///     ansatz.append(.rotationY(param), to: i)
    /// }
    ///
    /// let params: [Double] = [0.1, 0.2, 0.3]
    /// let bound = ansatz.bound(with: params)
    /// ```
    ///
    /// - Parameter vector: Array of parameter values (length must match parameter count)
    /// - Returns: Circuit with all parameters bound
    /// - Precondition: Vector length must equal ``parameterCount``
    /// - Complexity: O(n) where n = gate count
    ///
    /// - SeeAlso: ``binding(_:)`` for dictionary-based interface
    @_optimize(speed)
    @_eagerMove
    public func bound(with vector: [Double]) -> QuantumCircuit {
        let params = parameters
        let paramCount: Int = params.count
        ValidationUtilities.validateParameterVectorLength(vector.count, expected: paramCount)

        let bindings = Dictionary(uniqueKeysWithValues: zip(params.lazy.map(\.name), vector))
        return binding(bindings)
    }

    // MARK: - Gradient Computation Support

    /// Generates shifted circuits for parameter shift rule gradient computation
    ///
    /// Creates two circuits with specified parameter shifted by ±shift for gradient evaluation.
    /// Implements parameter shift rule: ∂⟨H⟩/∂θᵢ = [⟨H⟩(θᵢ+π/2) - ⟨H⟩(θᵢ-π/2)]/2
    ///
    /// **Example:**
    /// ```swift
    /// var ansatz = QuantumCircuit(qubits: 2)
    /// let theta = Parameter(name: "theta")
    /// ansatz.append(.rotationY(theta), to: 0)
    ///
    /// let base = ["theta": 0.5]
    /// let (plus, minus) = ansatz.shiftedCircuits(for: "theta", base: base)
    ///
    /// let gradient = (hamiltonian.expectation(in: plus.execute())
    ///               - hamiltonian.expectation(in: minus.execute())) / 2.0
    /// ```
    ///
    /// - Parameters:
    ///   - parameter: Name of parameter to shift
    ///   - base: Base parameter values (all parameters must be present)
    ///   - shift: Shift amount (default: π/2 for standard parameter shift rule)
    /// - Returns: Tuple of (plus, minus) circuits with parameter shifted
    /// - Precondition: Parameter exists in circuit and has binding
    /// - Complexity: O(n) where n = gate count (two circuit bindings)
    ///
    /// - SeeAlso: ``shiftedCircuits(at:baseVector:shift:)`` for vector interface
    @_optimize(speed)
    @_eagerMove
    public func shiftedCircuits(
        for parameter: String,
        base: [String: Double],
        shift: Double = .pi / 2,
    ) -> (plus: QuantumCircuit, minus: QuantumCircuit) {
        let params = parameters
        var paramSet = Set<String>()
        for param in params {
            paramSet.insert(param.name)
        }

        ValidationUtilities.validateParameterExists(parameter, in: paramSet)
        ValidationUtilities.validateParameterBinding(parameter, in: base)

        // Safety: validateParameterBinding above ensures parameter exists in base
        let baseValue = base[parameter]!

        var plusBindings = base
        var minusBindings = base

        plusBindings[parameter] = baseValue + shift
        minusBindings[parameter] = baseValue - shift

        let plusCircuit = binding(plusBindings)
        let minusCircuit = binding(minusBindings)

        return (plus: plusCircuit, minus: minusCircuit)
    }

    /// Generates shifted circuits using parameter index (vector interface)
    ///
    /// Array-based interface for parameter shifting. Equivalent to dictionary interface
    /// but uses parameter registration order instead of names.
    ///
    /// **Example:**
    /// ```swift
    /// var ansatz = QuantumCircuit(qubits: 2)
    /// for i in 0..<2 {
    ///     ansatz.append(.rotationY(Parameter(name: "theta_\(i)")), to: i)
    /// }
    ///
    /// let base: [Double] = [0.5, 1.0]
    /// let (plus, minus) = ansatz.shiftedCircuits(at: 0, baseVector: base)
    /// ```
    ///
    /// - Parameters:
    ///   - index: Index of parameter in registration order
    ///   - baseVector: Base parameter values (length must match parameter count)
    ///   - shift: Shift amount (default: π/2)
    /// - Returns: Tuple of (plus, minus) circuits
    /// - Precondition: Index in bounds and vector length matches parameter count
    /// - Complexity: O(n) where n = gate count
    ///
    /// - SeeAlso: ``shiftedCircuits(for:base:shift:)`` for dictionary interface
    @_optimize(speed)
    @_eagerMove
    public func shiftedCircuits(
        at index: Int,
        baseVector: [Double],
        shift: Double = .pi / 2,
    ) -> (plus: QuantumCircuit, minus: QuantumCircuit) {
        let params = parameters
        let paramCount: Int = params.count
        ValidationUtilities.validateIndexInBounds(index, bound: paramCount, name: "index")
        ValidationUtilities.validateParameterVectorLength(baseVector.count, expected: paramCount)

        let baseBindings = Dictionary(uniqueKeysWithValues: zip(params.lazy.map(\.name), baseVector))
        let paramName = params[index].name
        return shiftedCircuits(for: paramName, base: baseBindings, shift: shift)
    }

    // MARK: - Execution

    /// Expands quantum state with ancilla qubits when circuit operations exceed state size
    ///
    /// Ancilla qubits are initialized to |0⟩ and tensor-producted with the original state.
    /// Used internally by ``execute()`` methods to handle circuits that dynamically expand.
    ///
    /// - Parameters:
    ///   - state: Initial quantum state
    ///   - maxQubit: Maximum qubit index needed by circuit operations
    /// - Returns: Expanded state if maxQubit ≥ state.qubits, otherwise original state
    /// - Complexity: O(2^(n+k)) where n = original qubits, k = ancilla qubits needed
    ///
    /// - SeeAlso: ``execute()``, ``execute(on:)``, ``execute(on:upToIndex:)``
    @_eagerMove
    static func expandStateForAncilla(_ state: QuantumState, maxQubit: Int) -> QuantumState {
        guard maxQubit >= state.qubits else { return state }

        let numAncillaQubits: Int = maxQubit - state.qubits + 1
        let expandedSize = 1 << (state.qubits + numAncillaQubits)
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

        return QuantumState(qubits: maxQubit + 1, amplitudes: expandedAmplitudes)
    }

    /// Executes the circuit on a custom initial quantum state
    ///
    /// Applies all gate operations sequentially to transform the input state through unitary evolution.
    /// Automatically expands the state with ancilla qubits (initialized to |0⟩) if circuit operations
    /// reference qubit indices beyond the initial state size.
    ///
    /// - Parameter initialState: Starting quantum state (must have qubits matching circuit)
    /// - Returns: Final quantum state after applying all circuit operations
    /// - Precondition: initialState.qubits == qubits
    /// - Precondition: Circuit must contain only concrete parameters (no symbolic)
    /// - Complexity: O(n x 2^q) where n = gate count, q = max qubit index (including ancilla)
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    ///
    /// let initial = QuantumState(qubits: 2, amplitudes: [
    ///     Complex(0.6, 0), Complex(0.8, 0), .zero, .zero
    /// ])
    /// let final = circuit.execute(on: initial)
    /// ```
    ///
    /// - SeeAlso: ``execute()`` for execution from ground state
    @_optimize(speed)
    @_eagerMove
    public func execute(on initialState: QuantumState) -> QuantumState {
        ValidationUtilities.validateConcreteCircuit(parameterCount)
        ValidationUtilities.validateStateQubitCount(initialState, required: qubits)
        ValidationUtilities.validateCircuitOperations(gates, qubits: qubits)

        let maxQubit = highestQubitIndex
        var currentState = Self.expandStateForAncilla(initialState, maxQubit: maxQubit)

        for operation in gates {
            currentState = GateApplication.apply(
                operation.gate,
                to: operation.qubits,
                state: currentState,
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
    /// - Precondition: Circuit must contain only concrete parameters (no symbolic)
    /// - Complexity: O(n x 2^q) where n = gate count, q = max qubit index
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    /// let bellState = circuit.execute()  // (|00⟩ + |11⟩)/√2
    /// ```
    ///
    /// - SeeAlso: ``QuantumSimulator`` for GPU-accelerated execution, ``execute(on:)`` for custom initial states
    @_optimize(speed)
    @_eagerMove
    public func execute() -> QuantumState {
        let initialState = QuantumState(qubits: qubits)
        return execute(on: initialState)
    }

    /// Executes circuit up to a specific gate index for step-through visualization
    ///
    /// Applies only the first `upToIndex` operations, enabling incremental execution for animation
    /// workflows. External code should cache intermediate states at regular intervals for scrubbing.
    ///
    /// - Parameters:
    ///   - initialState: Starting quantum state
    ///   - upToIndex: Number of gates to execute (0 = no gates, count = all gates)
    /// - Returns: Quantum state after executing operations [0..<upToIndex]
    /// - Precondition: initialState.qubits == qubits
    /// - Precondition: 0 ≤ upToIndex ≤ count
    /// - Precondition: Circuit must contain only concrete parameters (no symbolic)
    /// - Complexity: O(k x 2^q) where k = upToIndex, q = max qubit index
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.pauliX, to: 1)
    /// circuit.append(.cnot, to: [0, 1])
    ///
    /// let initial = QuantumState(qubits: 2)
    /// let afterH = circuit.execute(on: initial, upToIndex: 1)  // Just H gate
    /// let afterHX = circuit.execute(on: initial, upToIndex: 2)  // H + X gates
    /// ```
    @_optimize(speed)
    @_eagerMove
    public func execute(on initialState: QuantumState, upToIndex: Int) -> QuantumState {
        ValidationUtilities.validateConcreteCircuit(parameterCount)
        ValidationUtilities.validateStateQubitCount(initialState, required: qubits)
        ValidationUtilities.validateUpToIndex(upToIndex, operationCount: gates.count)

        let maxQubit = highestQubitIndex
        var currentState = Self.expandStateForAncilla(initialState, maxQubit: maxQubit)

        for i in 0 ..< upToIndex {
            currentState = GateApplication.apply(
                gates[i].gate,
                to: gates[i].qubits,
                state: currentState,
            )
        }

        return currentState
    }

    // MARK: - CustomStringConvertible

    /// String representation of the quantum circuit
    public var description: String {
        if gates.isEmpty { return "QuantumCircuit(\(qubits) qubits, empty)" }

        var gateList = ""
        let limit = min(gates.count, 5)
        for i in 0 ..< limit {
            if i > 0 { gateList += ", " }
            gateList += gates[i].description
        }
        let suffix = gates.count > 5 ? ", ..." : ""

        let paramCount = parameterCount
        if paramCount > 0 {
            return "QuantumCircuit(\(qubits) qubits, \(gates.count) gates, \(paramCount) params): \(gateList)\(suffix)"
        }

        return "QuantumCircuit(\(qubits) qubits, \(gates.count) gates): \(gateList)\(suffix)"
    }
}

// MARK: - Batch Operations

public extension QuantumCircuit {
    /// Binds multiple parameter vectors to produce batch of circuits
    ///
    /// Efficient batch binding for VQE gradient computation and grid search.
    /// Validates all parameter vectors upfront before binding any circuits.
    ///
    /// **Example:**
    /// ```swift
    /// let ansatz = HardwareEfficientAnsatz(qubits: 4, depth: 2)
    ///
    /// let parameterSets: [[Double]] = [
    ///     [0.1, 0.2, 0.3, 0.4],
    ///     [0.5, 0.6, 0.7, 0.8],
    ///     [0.9, 1.0, 1.1, 1.2]
    /// ]
    ///
    /// let circuits = ansatz.circuit.binding(batch: parameterSets)
    /// ```
    ///
    /// - Parameter vectors: Array of parameter value arrays
    /// - Returns: Array of concrete quantum circuits (one per parameter vector)
    /// - Precondition: All vectors must have length equal to ``parameterCount``
    /// - Complexity: O(B x n) where B = batch size, n = gate count
    @_optimize(speed)
    @_eagerMove
    func binding(batch vectors: [[Double]]) -> [QuantumCircuit] {
        ValidationUtilities.validateNonEmpty(vectors, name: "Parameter vectors")

        let expectedCount: Int = parameterCount

        for vector in vectors {
            ValidationUtilities.validateParameterVectorLength(vector.count, expected: expectedCount)
        }

        var circuits: [QuantumCircuit] = []
        circuits.reserveCapacity(vectors.count)

        for vector in vectors {
            let circuit: QuantumCircuit = bound(with: vector)
            circuits.append(circuit)
        }

        return circuits
    }

    /// Generates parameter vectors for gradient computation via parameter shift rule
    ///
    /// Creates 2N parameter vectors for N parameters with each parameter shifted
    /// by ±π/2. Formula: ∂⟨H⟩/∂θᵢ = [⟨H⟩(θᵢ+π/2) - ⟨H⟩(θᵢ-π/2)]/2
    ///
    /// **Example:**
    /// ```swift
    /// let ansatz = HardwareEfficientAnsatz(qubits: 4, depth: 2)
    /// let base: [Double] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    ///
    /// let (plusVectors, minusVectors) = ansatz.circuit.gradientVectors(base: base)
    /// print(plusVectors.count)  // 8 (one per parameter)
    /// print(minusVectors.count)  // 8
    /// ```
    ///
    /// - Parameters:
    ///   - base: Base parameter values
    ///   - shift: Shift amount (default: π/2 for standard parameter shift)
    /// - Returns: Tuple of (plus vectors, minus vectors)
    /// - Precondition: Base vector length must equal ``parameterCount``
    /// - Complexity: O(N²) where N = parameter count (2N vectors of length N)
    ///
    /// - SeeAlso: ``binding(batch:)`` for binding generated vectors
    @_optimize(speed)
    @_eagerMove
    func gradientVectors(
        base: [Double],
        shift: Double = .pi / 2,
    ) -> (plus: [[Double]], minus: [[Double]]) {
        let numParams: Int = parameterCount
        ValidationUtilities.validateParameterVectorLength(base.count, expected: numParams)

        var plusVectors: [[Double]] = []
        var minusVectors: [[Double]] = []

        plusVectors.reserveCapacity(numParams)
        minusVectors.reserveCapacity(numParams)

        var buffer = base

        for i in 0 ..< numParams {
            let original: Double = buffer[i]

            buffer[i] = original + shift
            plusVectors.append(buffer)

            buffer[i] = original - shift
            minusVectors.append(buffer)

            buffer[i] = original
        }

        return (plus: plusVectors, minus: minusVectors)
    }

    /// Generates parameter vectors for grid search optimization
    ///
    /// Creates Cartesian product of parameter ranges for exhaustive search.
    /// Useful for QAOA parameter optimization over low-dimensional spaces.
    ///
    /// **Example:**
    /// ```swift
    /// let cost = MaxCut.hamiltonian(edges: [(0, 1), (1, 2), (2, 0)])
    /// let mixer = MixerHamiltonian.x(qubits: 3)
    /// let ansatz = QuantumCircuit.qaoa(cost: cost, mixer: mixer, qubits: 3, depth: 1)
    ///
    /// let gammaRange = stride(from: 0.0, through: .pi, by: .pi / 10)
    /// let betaRange = stride(from: 0.0, through: .pi, by: .pi / 10)
    ///
    /// let vectors = ansatz.gridSearchVectors(ranges: [
    ///     Array(gammaRange),
    ///     Array(betaRange)
    /// ])
    /// print(vectors.count)  // 121 (11 x 11)
    /// ```
    ///
    /// - Parameter ranges: Array of value arrays (one per parameter)
    /// - Returns: Array of parameter vectors (Cartesian product)
    /// - Precondition: Ranges count must equal ``parameterCount``
    /// - Complexity: O(M^N) vectors where M = values per range, N = parameter count
    ///
    /// - SeeAlso: ``binding(batch:)`` for binding generated vectors
    @_optimize(speed)
    @_eagerMove
    func gridSearchVectors(ranges: [[Double]]) -> [[Double]] {
        let numParams: Int = parameterCount

        ValidationUtilities.validateArrayCount(ranges, expected: numParams, name: "ranges")

        var totalCombinations = 1
        for index in 0 ..< numParams {
            let rangeCount: Int = ranges[index].count
            ValidationUtilities.validatePositiveInt(rangeCount, name: "ranges[\(index)].count")
            totalCombinations *= rangeCount
        }

        var parameterVectors: [[Double]] = []
        parameterVectors.reserveCapacity(totalCombinations)

        var indices = [Int](unsafeUninitializedCapacity: numParams) { buffer, count in
            for i in 0 ..< numParams {
                buffer[i] = 0
            }
            count = numParams
        }

        var currentVector = [Double](unsafeUninitializedCapacity: numParams) { buffer, count in
            for i in 0 ..< numParams {
                buffer[i] = ranges[i][0]
            }
            count = numParams
        }

        while true {
            for paramIndex in 0 ..< numParams {
                currentVector[paramIndex] = ranges[paramIndex][indices[paramIndex]]
            }

            parameterVectors.append(currentVector)

            var incrementPosition: Int = numParams - 1
            while incrementPosition >= 0 {
                indices[incrementPosition] += 1

                if indices[incrementPosition] < ranges[incrementPosition].count {
                    break
                }

                indices[incrementPosition] = 0
                incrementPosition -= 1
            }

            if incrementPosition < 0 {
                break
            }
        }

        return parameterVectors
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
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        return circuit
    }

    /// Creates a GHZ circuit: H₀ · CNOT₀₁ · CNOT₀₂ · ...
    ///
    /// Produces n-qubit maximally entangled Greenberger-Horne-Zeilinger state: (|00...0⟩ + |11...1⟩)/√2
    ///
    /// - Parameter qubits: Number of qubits (minimum 2, default 3)
    /// - Returns: n-qubit circuit generating GHZ state
    /// - Precondition: qubits ≥ 2
    /// - Complexity: O(n) where n = qubits
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.ghz(qubits: 3)
    /// let state = circuit.execute()
    /// print(state.probability(of: 0b000))  // 0.5
    /// print(state.probability(of: 0b111))  // 0.5
    /// ```
    @_eagerMove
    static func ghz(qubits: Int = 3) -> QuantumCircuit {
        ValidationUtilities.validateMinimumQubits(qubits, min: 2, algorithmName: "GHZ")

        var circuit = QuantumCircuit(qubits: qubits)

        circuit.append(.hadamard, to: 0)

        for i in 1 ..< qubits {
            circuit.append(.cnot, to: [0, i])
        }

        return circuit
    }

    /// Creates uniform superposition circuit: Apply H to all qubits
    ///
    /// Produces equal superposition over all basis states: (Σᵢ|i⟩)/√(2ⁿ)
    ///
    /// - Parameter qubits: Number of qubits
    /// - Returns: Circuit that creates uniform superposition
    /// - Complexity: O(n) where n = qubits
    @_eagerMove
    static func uniformSuperposition(qubits: Int) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(qubits)

        var circuit = QuantumCircuit(qubits: qubits)

        for i in 0 ..< qubits {
            circuit.append(.hadamard, to: i)
        }

        return circuit
    }

    /// Creates Quantum Fourier Transform circuit for frequency domain transformation.
    ///
    /// Implements |j⟩ -> (1/√N)Σₖ exp(2πijk/N)|k⟩ where N = 2ⁿ, a key component in Shor's algorithm
    /// and quantum phase estimation. Applies Hadamard to each qubit followed by controlled-phase
    /// gates with angles π/2, π/4, π/8, ... (decreasing powers of 2), then reverses qubit order
    /// with SWAP gates.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.qft(qubits: 3)
    /// let state = QuantumState(qubits: 3)  // |000⟩
    /// let transformed = circuit.execute(on: state)
    /// // Result: uniform superposition (|000⟩ + |001⟩ + ... + |111⟩)/√8
    /// ```
    ///
    /// - Parameter qubits: Number of qubits (typical range: 3-8, maximum: 16)
    /// - Returns: Circuit implementing QFT
    /// - Precondition: qubits ≤ 16
    /// - Complexity: O(n²) gates where n = qubits
    ///
    /// - SeeAlso: ``inverseQFT(qubits:)`` for inverse transformation
    @_optimize(speed)
    @_eagerMove
    static func qft(qubits: Int) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateAlgorithmQubitLimit(qubits, max: 16, algorithmName: "QFT")

        var circuit = QuantumCircuit(qubits: qubits)

        for target in 0 ..< qubits {
            circuit.append(.hadamard, to: target)

            for control in (target + 1) ..< qubits {
                let k = control - target + 1
                let theta = Double.pi / Double(1 << k)

                circuit.append(.controlledPhase(theta), to: [control, target])
            }
        }

        let swapCount = qubits / 2
        for i in 0 ..< swapCount {
            let j = qubits - 1 - i
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
    /// let qftCircuit = QuantumCircuit.qft(qubits: 3)
    /// let inverseCircuit = QuantumCircuit.inverseQFT(qubits: 3)
    ///
    /// var combined = qftCircuit
    /// for gate in inverseCircuit.gates {
    ///     combined.append(gate.gate, to: gate.qubits)
    /// }
    /// // combined circuit is effectively identity (QFT · QFT† = I)
    /// ```
    ///
    /// - Parameter qubits: Number of qubits
    /// - Returns: Circuit implementing inverse QFT
    /// - Complexity: O(n²) gates where n = qubits
    ///
    /// - SeeAlso: ``qft(qubits:)`` for forward transformation
    @_optimize(speed)
    @_eagerMove
    static func inverseQFT(qubits: Int) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(qubits)

        var circuit = QuantumCircuit(qubits: qubits)

        let swapCount = qubits / 2
        for i in 0 ..< swapCount {
            let j = qubits - 1 - i
            circuit.append(.swap, to: [i, j])
        }

        for target in (0 ..< qubits).reversed() {
            for control in (target + 1 ..< qubits).reversed() {
                let k = control - target + 1
                let theta = -Double.pi / Double(1 << k)

                circuit.append(.controlledPhase(theta), to: [control, target])
            }

            circuit.append(.hadamard, to: target)
        }

        return circuit
    }

    /// Creates Grover search algorithm circuit for O(√N) unstructured search.
    ///
    /// Initializes uniform superposition, then repeats oracle (phase flip on target) and diffusion
    /// (inversion about average) for optimal iterations ⌊π/4 * √(2ⁿ)⌋. Achieves quadratic speedup
    /// over classical O(N) search.
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (search space size = 2ⁿ, maximum n=10)
    ///   - target: Target state to search for (basis state index, 0 ≤ target < 2ⁿ)
    ///   - iterations: Number of Grover iterations (defaults to optimal: ⌊π/4 x √(2ⁿ)⌋)
    /// - Returns: Circuit implementing Grover search
    /// - Precondition: qubits ≤ 10
    /// - Precondition: 0 ≤ target < 2ⁿ
    /// - Complexity: O(k x n x 2ⁿ) where k = iterations, n = qubits (execution cost)
    @_optimize(speed)
    @_eagerMove
    static func grover(qubits: Int, target: Int, iterations: Int? = nil) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateAlgorithmQubitLimit(qubits, max: 10, algorithmName: "Grover")

        let stateSpaceSize = 1 << qubits
        ValidationUtilities.validateIndexInBounds(target, bound: stateSpaceSize, name: "Target basis state")

        let optimalIterations: Int = iterations ?? Int((Double.pi / 4.0) * sqrt(Double(stateSpaceSize)))

        var circuit = QuantumCircuit(qubits: qubits)

        for qubit in 0 ..< qubits {
            circuit.append(.hadamard, to: qubit)
        }

        for _ in 0 ..< optimalIterations {
            appendGroverOracle(to: &circuit, target: target, qubits: qubits)
            appendGroverDiffusion(to: &circuit, qubits: qubits)
        }

        return circuit
    }

    /// Implements oracle marking for Grover search via controlled phase flip
    ///
    /// Marks target basis state with phase flip using X gates to encode complement
    /// bits, multi-controlled Z for phase inversion, then X gates to restore state.
    ///
    /// - Complexity: O(n) gates where n = qubits
    private static func appendGroverOracle(to circuit: inout QuantumCircuit, target: Int, qubits: Int) {
        for qubit in 0 ..< qubits {
            if (target >> qubit) & 1 == 0 {
                circuit.append(.pauliX, to: qubit)
            }
        }

        appendMultiControlledZ(to: &circuit, qubits: qubits)

        for qubit in 0 ..< qubits {
            if (target >> qubit) & 1 == 0 {
                circuit.append(.pauliX, to: qubit)
            }
        }
    }

    /// Appends multi-controlled Z gate for Grover diffusion operator
    ///
    /// Implements n-controlled Z using efficient decompositions based on qubit count:
    /// - 1 qubit: Single Z gate
    /// - 2 qubits: Controlled-phase gate
    /// - 3 qubits: Hadamard sandwich around Toffoli (H·CCNOT·H)
    /// - n≥4 qubits: Hadamard sandwich around multi-controlled X with ancilla
    ///
    /// - Complexity: O(n) gates where n = qubits
    private static func appendMultiControlledZ(to circuit: inout QuantumCircuit, qubits: Int) {
        if qubits == 1 {
            circuit.append(.pauliZ, to: 0)
        } else if qubits == 2 {
            circuit.append(.controlledPhase(.pi), to: [0, 1])
        } else if qubits == 3 {
            circuit.append(.hadamard, to: 2)
            circuit.append(.toffoli, to: [0, 1, 2])
            circuit.append(.hadamard, to: 2)
        } else {
            let target = qubits - 1
            circuit.append(.hadamard, to: target)

            appendMultiControlledX(to: &circuit, controls: Array(0 ..< qubits - 1), target: target)

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
    /// - Parameters:
    ///   - circuit: Circuit to append gates to
    ///   - controls: Control qubit indices (can be empty)
    ///   - target: Target qubit index
    /// - Complexity: O(n) gates where n = controls.count
    private static func appendMultiControlledX(to circuit: inout QuantumCircuit, controls: [Int], target: Int) {
        let n = controls.count

        if n == 0 {
            circuit.append(.pauliX, to: target)
        } else if n == 1 {
            circuit.append(.cnot, to: [controls[0], target])
        } else if n == 2 {
            circuit.append(.toffoli, to: [controls[0], controls[1], target])
        } else {
            // Safety: controls is non-empty (n >= 3 in this branch)
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
    /// Implements Cⁿ(Y) by converting to controlled-X via phase gates:
    /// Cⁿ(Y) = S†·Cⁿ(X)·S where S† rotates Y basis to X basis.
    ///
    /// - Complexity: O(n) gates where n = controls.count
    private static func appendMultiControlledY(to circuit: inout QuantumCircuit, controls: [Int], target: Int) {
        let n = controls.count

        if n == 0 {
            circuit.append(.pauliY, to: target)
        } else if n == 1 {
            circuit.append(.cy, to: [controls[0], target])
        } else {
            circuit.append(.phase(-Double.pi / 2.0), to: target)
            appendMultiControlledX(to: &circuit, controls: controls, target: target)
            circuit.append(.sGate, to: target)
        }
    }

    /// Appends multi-controlled Z gate using Hadamard sandwich
    ///
    /// Implements Cⁿ(Z) by converting to controlled-X via Hadamard conjugation:
    /// Cⁿ(Z) = H·Cⁿ(X)·H where H rotates Z basis to X basis.
    ///
    /// - Complexity: O(n) gates where n = controls.count
    private static func appendMultiControlledZ(to circuit: inout QuantumCircuit, controls: [Int], target: Int) {
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

    /// Appends multi-controlled arbitrary single-qubit unitary gate.
    ///
    /// Applies gate U with n controls: |1⟩⊗ⁿ|ψ⟩ -> |1⟩⊗ⁿU|ψ⟩. Pauli gates use optimized basis-specific
    /// decompositions, Hadamard uses Ry sandwich around controlled-Z, and general unitaries use
    /// Gray code decomposition U·Cⁿ(X)·U†·Cⁿ(X)·U.
    ///
    /// - Parameters:
    ///   - circuit: Circuit to append gates to
    ///   - gate: Single-qubit gate to apply
    ///   - controls: Control qubit indices
    ///   - target: Target qubit index
    /// - Precondition: gate must be single-qubit
    /// - Complexity: O(n) gates where n = controls.count
    ///
    /// - SeeAlso: ``grover(qubits:target:iterations:)``
    static func appendMultiControlledU(
        to circuit: inout QuantumCircuit,
        gate: QuantumGate,
        controls: [Int],
        target: Int,
    ) {
        ValidationUtilities.validateControlledGateIsSingleQubit(gate.qubitsRequired)

        let n = controls.count

        switch gate {
        case .pauliX:
            appendMultiControlledX(to: &circuit, controls: controls, target: target)
        case .pauliY:
            appendMultiControlledY(to: &circuit, controls: controls, target: target)
        case .pauliZ:
            appendMultiControlledZ(to: &circuit, controls: controls, target: target)
        case .hadamard:
            if n == 0 {
                circuit.append(.hadamard, to: target)
            } else {
                circuit.append(.rotationY(.pi / 4), to: target)
                appendMultiControlledZ(to: &circuit, controls: controls, target: target)
                circuit.append(.rotationY(-.pi / 4), to: target)
            }
        default:
            if n == 0 {
                circuit.append(gate, to: target)
            } else {
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

    /// Appends Grover diffusion operator (inversion about average)
    ///
    /// Implements amplitude amplification via reflection about uniform superposition:
    /// D = 2|ψ⟩⟨ψ| - I where |ψ⟩ = H⊗ⁿ|0⟩. Gate sequence: H⊗ⁿ · X⊗ⁿ · Cⁿ(Z) · X⊗ⁿ · H⊗ⁿ.
    ///
    /// - Complexity: O(n) gates where n = qubits
    private static func appendGroverDiffusion(to circuit: inout QuantumCircuit, qubits: Int) {
        for qubit in 0 ..< qubits {
            circuit.append(.hadamard, to: qubit)
        }

        for qubit in 0 ..< qubits {
            circuit.append(.pauliX, to: qubit)
        }

        appendMultiControlledZ(to: &circuit, qubits: qubits)

        for qubit in 0 ..< qubits {
            circuit.append(.pauliX, to: qubit)
        }

        for qubit in 0 ..< qubits {
            circuit.append(.hadamard, to: qubit)
        }
    }

    /// Creates quantum annealing circuit for combinatorial optimization.
    ///
    /// Implements adiabatic evolution from transverse-field Hamiltonian H₀ to problem Hamiltonian Hₚ
    /// via interpolation H(t) = (1-t)H₀ + tHₚ. Initializes in superposition, evolves through discrete
    /// time steps with Trotter approximation, then measures to find optimal configuration.
    ///
    /// **Example:**
    /// ```swift
    /// // MaxCut on triangle graph
    /// let problem = QuantumCircuit.IsingProblem(fromDictionary: [
    ///     "0-1": -0.5, "1-2": -0.5, "0-2": -0.5
    /// ], qubits: 3)
    /// let circuit = QuantumCircuit.annealing(qubits: 3, problem: problem, annealingSteps: 20)
    /// let state = circuit.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (problem variables, maximum: 8)
    ///   - problem: Ising model defining optimization problem (local fields + couplings)
    ///   - annealingSteps: Number of time steps in annealing schedule (default: 20)
    /// - Returns: Circuit demonstrating quantum annealing process
    /// - Precondition: qubits ≤ 8
    /// - Complexity: O(s x n²) gates where s = annealingSteps, n = qubits
    ///
    /// - SeeAlso: ``IsingProblem``, ``annealing(qubits:couplings:annealingSteps:)``
    @_eagerMove
    static func annealing(qubits: Int, problem: IsingProblem, annealingSteps: Int = 20) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validatePositiveInt(annealingSteps, name: "annealingSteps")
        ValidationUtilities.validateAlgorithmQubitLimit(qubits, max: 8, algorithmName: "Annealing")

        var circuit = QuantumCircuit(qubits: qubits)

        for qubit in 0 ..< qubits {
            circuit.append(.hadamard, to: qubit)
        }

        for step in 0 ..< annealingSteps {
            let time = Double(step) / Double(annealingSteps - 1)

            let transverseStrength = 1.0 - time
            for qubit in 0 ..< qubits {
                let angle = 2.0 * transverseStrength * problem.transverseField[qubit]
                circuit.append(.rotationX(angle), to: qubit)
            }

            let problemStrength = time

            for qubit in 0 ..< qubits {
                let angle = 2.0 * problemStrength * problem.localFields[qubit]
                circuit.append(.rotationZ(angle), to: qubit)
            }

            for i in 0 ..< qubits {
                for j in (i + 1) ..< qubits {
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
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.annealing(
    ///     qubits: 3,
    ///     couplings: ["0-1": -0.5, "1-2": -0.5, "0-2": -0.5],
    ///     annealingSteps: 20
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (problem variables)
    ///   - couplings: Dictionary mapping qubit indices/pairs to coupling strengths
    ///   - annealingSteps: Number of time steps in annealing schedule
    /// - Returns: Annealing circuit for the specified Ising problem
    ///
    /// - SeeAlso: ``annealing(qubits:problem:annealingSteps:)``
    @_eagerMove
    static func annealing(qubits: Int, couplings: [String: Double], annealingSteps: Int = 20) -> QuantumCircuit {
        let problem = IsingProblem(fromDictionary: couplings, qubits: qubits)
        return annealing(qubits: qubits, problem: problem, annealingSteps: annealingSteps)
    }

    /// Ising model problem definition for quantum annealing.
    ///
    /// Encodes combinatorial optimization as Ising Hamiltonian H = Σᵢ hᵢZᵢ + Σᵢⱼ JᵢⱼZᵢZⱼ where hᵢ are
    /// local fields and Jᵢⱼ are qubit-qubit couplings. Applicable to MaxCut, constraint satisfaction,
    /// QUBO, and traveling salesman problems. Transverse field enables quantum tunneling.
    ///
    /// **Example:**
    /// ```swift
    /// let problem = QuantumCircuit.IsingProblem(fromDictionary: [
    ///     "0": -1.0,        // local field on qubit 0
    ///     "0-1": -0.5,      // coupling between qubits 0 and 1
    ///     "1-2": -0.5
    /// ], qubits: 3)
    /// ```
    ///
    /// - SeeAlso: ``annealing(qubits:problem:annealingSteps:)``
    /// - SeeAlso: ``annealing(qubits:couplings:annealingSteps:)``
    struct IsingProblem {
        public let localFields: [Double]
        public let couplings: [[Double]]
        public let transverseField: [Double]

        /// Create Ising model problem instance
        public init(localFields: [Double], couplings: [[Double]], transverseField: [Double]? = nil) {
            let n = localFields.count
            ValidationUtilities.validateSquareMatrix(couplings, name: "Couplings matrix")

            self.localFields = localFields
            self.couplings = couplings
            self.transverseField = transverseField ?? [Double](repeating: 1.0, count: n)
        }

        /// Create Ising model from dictionary specification
        public init(fromDictionary dictionary: [String: Double], qubits: Int) {
            ValidationUtilities.validatePositiveQubits(qubits)

            var localFieldsBuffer = [Double](unsafeUninitializedCapacity: qubits) { buffer, count in
                buffer.initialize(repeating: 0.0)
                count = qubits
            }

            var couplingsBuffer = [[Double]](unsafeUninitializedCapacity: qubits) { buffer, count in
                for i in 0 ..< qubits {
                    buffer[i] = [Double](unsafeUninitializedCapacity: qubits) { innerBuffer, innerCount in
                        innerBuffer.initialize(repeating: 0.0)
                        innerCount = qubits
                    }
                }
                count = qubits
            }

            for (key, value) in dictionary {
                let parseQubits = Self.parseQubitIndices(from: key)
                ValidationUtilities.validateCouplingKeyFormat(parseQubits.count, key: key)

                if parseQubits.count == 1 {
                    let qubit = parseQubits[0]
                    ValidationUtilities.validateQubitIndex(qubit, qubits: qubits)
                    localFieldsBuffer[qubit] = value
                } else {
                    let qubit1 = parseQubits[0]
                    let qubit2 = parseQubits[1]
                    ValidationUtilities.validateQubitIndex(qubit1, qubits: qubits)
                    ValidationUtilities.validateQubitIndex(qubit2, qubits: qubits)
                    ValidationUtilities.validateDistinctVertices(qubit1, qubit2)
                    couplingsBuffer[qubit1][qubit2] = value
                    couplingsBuffer[qubit2][qubit1] = value
                }
            }

            localFields = localFieldsBuffer
            couplings = couplingsBuffer
            transverseField = [Double](unsafeUninitializedCapacity: qubits) { buffer, count in
                buffer.initialize(repeating: 1.0)
                count = qubits
            }
        }

        private static func parseQubitIndices(from key: String) -> [Int] {
            ValidationUtilities.validateNonEmptyString(key, name: "Coupling key")

            let hasSeparator = key.contains { !$0.isNumber }

            if hasSeparator {
                return key.split { !$0.isNumber }.compactMap { Int($0) }
            } else if key.count == 2 {
                return key.compactMap(\.wholeNumberValue)
            } else {
                // Safety: only reached when key contains only digits (no separator, length != 2)
                return [Int(key)!]
            }
        }

        /// Creates Ising problem for quadratic function minimization
        public static func quadraticMinimum(qubits: Int = 4) -> IsingProblem {
            ValidationUtilities.validateMinimumQubits(qubits, min: 2, algorithmName: "Quadratic minimum")

            let localFields = [Double](unsafeUninitializedCapacity: qubits) { buffer, count in
                for i in 0 ..< qubits {
                    buffer[i] = -1.0 * Double(1 << i)
                }
                count = qubits
            }

            let couplings = [[Double]](unsafeUninitializedCapacity: qubits) { buffer, count in
                for i in 0 ..< qubits {
                    buffer[i] = [Double](unsafeUninitializedCapacity: qubits) { innerBuffer, innerCount in
                        innerBuffer.initialize(repeating: 0.0)
                        innerCount = qubits
                    }
                    if i < qubits - 1 {
                        buffer[i][i + 1] = 0.5
                    }
                }
                count = qubits
            }

            return IsingProblem(localFields: localFields, couplings: couplings)
        }
    }

    /// Appends ZZ coupling evolution for Ising model simulation
    ///
    /// Implements exp(-iθZᵢZⱼ) decomposition using CNOT ladder:
    /// CNOT(i,j) · Rz(2θ) · CNOT(i,j) = exp(-iθZᵢZⱼ)
    ///
    /// - Complexity: O(1) - constant 3 gates
    private static func appendZZCoupling(to circuit: inout QuantumCircuit, qubit1: Int, qubit2: Int, angle: Double) {
        circuit.append(.cnot, to: [qubit1, qubit2])
        circuit.append(.rotationZ(2.0 * angle), to: qubit2)
        circuit.append(.cnot, to: [qubit1, qubit2])
    }
}
