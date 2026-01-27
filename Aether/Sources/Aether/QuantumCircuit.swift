// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Ordered sequence of quantum operations transforming |00...0⟩ through unitary and non-unitary operations.
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
/// - SeeAlso: ``CircuitOperation`` for operation types
public struct QuantumCircuit: Equatable, Hashable, CustomStringConvertible, Sendable {
    public private(set) var operations: [CircuitOperation]
    public private(set) var qubits: Int
    @usableFromInline var cachedMaxQubitUsed: Int

    /// When true, append automatically cancels adjacent identity pairs (H-H, X-X, CNOT-CNOT, etc.)
    ///
    /// Enabling this provides O(1) per-append optimization that immediately removes gates forming
    /// identity when appended. For full optimization including rotation merging and commutation
    /// reordering, call ``optimized()`` after circuit construction.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2, autoOptimize: true)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.hadamard, to: 0)  // Automatically cancelled
    /// print(circuit.count)  // 0
    /// ```
    ///
    /// - SeeAlso: ``optimized()``
    public var autoOptimize: Bool

    /// Number of operations in the circuit
    ///
    /// - Complexity: O(1)
    @inlinable
    public var count: Int { operations.count }

    /// Whether the circuit contains no operations
    ///
    /// - Complexity: O(1)
    @inlinable
    public var isEmpty: Bool { operations.count == 0 }

    // MARK: - Initialization

    /// Creates an empty quantum circuit with specified qubit count
    ///
    /// Initializes a circuit with no operations. Operations can be added via append or insert methods.
    /// Circuit auto-expands if gates reference qubits beyond initial size (up to 30 qubits maximum).
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (1-30)
    ///   - autoOptimize: When true, automatically cancel adjacent identity pairs on append (default: false)
    /// - Precondition: qubits > 0
    /// - Complexity: O(1)
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 3)
    /// circuit.append(.hadamard, to: 0)
    /// print(circuit.count)  // 1
    ///
    /// // With auto-optimization
    /// var opt = QuantumCircuit(qubits: 2, autoOptimize: true)
    /// opt.append(.hadamard, to: 0)
    /// opt.append(.hadamard, to: 0)  // Cancelled automatically
    /// print(opt.count)  // 0
    /// ```
    public init(qubits: Int, autoOptimize: Bool = false) {
        ValidationUtilities.validatePositiveQubits(qubits)
        self.qubits = qubits
        cachedMaxQubitUsed = qubits - 1
        operations = []
        self.autoOptimize = autoOptimize
    }

    /// Creates a circuit with predefined circuit operations
    ///
    /// Useful for constructing circuits from previously saved operations or programmatic generation.
    /// Computes maximum qubit usage during initialization for ancilla detection.
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (1-30)
    ///   - operations: Initial circuit operations to include
    ///   - autoOptimize: When true, automatically cancel adjacent identity pairs on append (default: false)
    /// - Precondition: qubits > 0
    /// - Complexity: O(n) where n is number of operations
    ///
    /// **Example:**
    /// ```swift
    /// let ops: [CircuitOperation] = [
    ///     .gate(.hadamard, qubits: [0]),
    ///     .gate(.cnot, qubits: [0, 1])
    /// ]
    /// let circuit = QuantumCircuit(qubits: 2, operations: ops)
    /// ```
    public init(qubits: Int, operations: [CircuitOperation], autoOptimize: Bool = false) {
        ValidationUtilities.validatePositiveQubits(qubits)
        self.qubits = qubits
        self.operations = operations
        self.autoOptimize = autoOptimize
        var maxQubit: Int = qubits - 1
        for operation in operations {
            let opMax: Int = operation.qubits.max() ?? -1
            if opMax > maxQubit { maxQubit = opMax }
        }
        cachedMaxQubitUsed = maxQubit
    }

    // MARK: - Building Methods

    /// Appends a gate to the specified qubits
    ///
    /// Adds the quantum gate to the circuit's operation sequence. Automatically expands
    /// the circuit's qubit count if the gate references indices beyond current size (up to 30 qubits).
    /// Supports both concrete and symbolic parameters via ``QuantumGate``'s unified design.
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply (concrete or symbolic)
    ///   - qubits: Target qubit indices
    ///   - timestamp: Optional timestamp for animation or circuit scrubbing
    /// - Precondition: All qubit indices >= 0
    ///   - Precondition: Auto-expanded circuit size <= 30 qubits
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

        if autoOptimize,
           let last = operations.last,
           case let .gate(lastGate, lastQubits, _) = last,
           lastQubits == qubits,
           CircuitOptimizer.gatesFormIdentity(lastGate, gate)
        {
            operations.removeLast()
            if operations.isEmpty {
                cachedMaxQubitUsed = self.qubits - 1
            }
            return
        }

        let operation = CircuitOperation.gate(gate, qubits: qubits, timestamp: timestamp)
        operations.append(operation)

        let operationMax: Int = qubits.max() ?? -1
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
    /// - Precondition: qubit >= 0
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

    /// Appends a non-unitary operation to a single qubit
    ///
    /// Adds an irreversible operation such as mid-circuit reset to the circuit's operation sequence.
    /// Non-unitary operations cannot be optimized away by identity cancellation and prevent
    /// circuit inversion.
    ///
    /// - Parameters:
    ///   - operation: Non-unitary operation to apply
    ///   - qubit: Target qubit index
    ///   - timestamp: Optional timestamp for animation or circuit scrubbing
    /// - Precondition: qubit >= 0
    /// - Complexity: O(1) amortized
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.reset, to: 0)
    /// circuit.append(.hadamard, to: 0)
    /// let state = circuit.execute()
    /// ```
    public mutating func append(_ operation: NonUnitaryOperation, to qubit: Int, timestamp: Double? = nil) {
        ValidationUtilities.validateNonNegativeQubits([qubit])

        if qubit >= qubits {
            let newqubits: Int = qubit + 1
            ValidationUtilities.validateMemoryLimit(newqubits)
            qubits = newqubits
        }

        switch operation {
        case .reset:
            let circuitOp = CircuitOperation.reset(qubit: qubit, timestamp: timestamp)
            operations.append(circuitOp)
        }

        if qubit > cachedMaxQubitUsed { cachedMaxQubitUsed = qubit }
    }

    /// Inserts a gate at a specific position in the circuit
    ///
    /// Places the gate at the specified index, shifting subsequent operations forward.
    /// Useful for circuit optimization and debugging workflows.
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to insert
    ///   - qubits: Target qubit indices
    ///   - index: Position to insert at (0 = beginning, count = end)
    ///   - timestamp: Optional timestamp for animation
    /// - Precondition: 0 <= index <= count
    /// - Precondition: All qubits < qubits
    /// - Complexity: O(n) where n is number of operations (array shift)
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
        ValidationUtilities.validateIndexInBounds(index, bound: operations.count, name: "Index")
        ValidationUtilities.validateOperationQubits(qubits, numQubits: self.qubits)

        let operation = CircuitOperation.gate(gate, qubits: qubits, timestamp: timestamp)
        operations.insert(operation, at: index)
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
    /// - Precondition: 0 <= index <= count
    /// - Precondition: qubit < qubits
    /// - Complexity: O(n) where n is number of operations
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

    /// Inserts a non-unitary operation at a specific position in the circuit
    ///
    /// Places the non-unitary operation at the specified index, shifting subsequent operations forward.
    /// Non-unitary operations such as reset are irreversible and cannot be optimized away.
    ///
    /// - Parameters:
    ///   - operation: Non-unitary operation to insert
    ///   - qubit: Target qubit index
    ///   - index: Position to insert at (0 = beginning, count = end)
    ///   - timestamp: Optional timestamp for animation
    /// - Precondition: 0 <= index <= count
    /// - Precondition: qubit < qubits
    /// - Complexity: O(n) where n is number of operations (array shift)
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    /// circuit.insert(.reset, to: 0, at: 1)
    /// // Result: H(0), reset(0), CNOT(0,1)
    /// ```
    public mutating func insert(_ operation: NonUnitaryOperation, to qubit: Int, at index: Int, timestamp: Double? = nil) {
        ValidationUtilities.validateIndexInBounds(index, bound: operations.count, name: "Index")
        ValidationUtilities.validateOperationQubits([qubit], numQubits: qubits)

        switch operation {
        case .reset:
            let circuitOp = CircuitOperation.reset(qubit: qubit, timestamp: timestamp)
            operations.insert(circuitOp, at: index)
        }
    }

    /// Adds a fully-formed circuit operation to the circuit
    ///
    /// Dispatches to the appropriate append method based on operation type, applying
    /// auto-optimization for gate operations and direct insertion for non-unitary operations.
    ///
    /// - Parameter operation: Circuit operation to add
    /// - Complexity: O(1) amortized
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.addOperation(.gate(.hadamard, qubits: [0]))
    /// circuit.addOperation(.reset(qubit: 1))
    /// circuit.addOperation(.gate(.cnot, qubits: [0, 1], timestamp: 2.0))
    /// ```
    public mutating func addOperation(_ operation: CircuitOperation) {
        switch operation {
        case let .gate(gate, qubits, timestamp):
            append(gate, to: qubits, timestamp: timestamp)
        case let .reset(qubit, timestamp):
            append(.reset, to: qubit, timestamp: timestamp)
        }
    }

    /// Removes the operation at the specified index
    ///
    /// Deletes the operation and shifts subsequent operations backward. Recomputes cached maximum
    /// qubit index if the removed operation was the highest-index operation.
    ///
    /// - Parameter index: Index of operation to remove (0-based)
    /// - Precondition: 0 <= index < count
    /// - Complexity: O(n) where n is number of operations (array shift + potential recompute)
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.pauliX, to: 1)
    /// circuit.remove(at: 0)  // Removes H gate, keeps X gate
    /// ```
    public mutating func remove(at index: Int) {
        ValidationUtilities.validateIndexInBounds(index, bound: operations.count, name: "Index")
        let removedMax: Int = operations[index].qubits.max() ?? -1
        operations.remove(at: index)

        if removedMax == cachedMaxQubitUsed { recomputeMaxQubitCache() }
    }

    /// Removes all operations from the circuit
    ///
    /// Clears the operation array while preserving the qubit count. Resets cached maximum qubit
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
        operations.removeAll()
        cachedMaxQubitUsed = qubits - 1
    }

    /// Recomputes cached maximum qubit index by scanning all operations.
    @inline(__always)
    private mutating func recomputeMaxQubitCache() {
        var maxQubit: Int = qubits - 1
        for operation in operations {
            let opMax: Int = operation.qubits.max() ?? -1
            if opMax > maxQubit { maxQubit = opMax }
        }
        cachedMaxQubitUsed = maxQubit
    }

    // MARK: - Introspection

    /// Highest qubit index referenced by any operation in circuit
    ///
    /// Used to detect ancilla qubits that may exceed logical circuit size.
    /// Value is cached and updated during append/insert/remove operations.
    ///
    /// - Returns: Highest qubit index, or qubits-1 if no operations
    /// - Complexity: O(1) - cached value
    @_optimize(speed)
    @inlinable
    public var highestQubitIndex: Int { cachedMaxQubitUsed }

    /// Symbolic parameters in registration order
    ///
    /// Extracts all ``Parameter`` instances from operations, maintaining order of first appearance.
    /// Empty array for circuits with only concrete parameters. Used for variational algorithms
    /// to determine parameter count and names for binding.
    ///
    /// - Returns: Ordered array of distinct parameters
    /// - Complexity: O(n x p) where n = operation count, p = parameters per gate (typically <=3)
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

        for operation in operations {
            for param in operation.parameters() {
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
    /// - Complexity: O(n x p) where n = operation count, p = parameters per gate
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
    /// - Complexity: O(n) where n = operation count
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

        for operation in operations {
            let boundOp = operation.bound(with: values)
            bound.addOperation(boundOp)
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
    /// - Complexity: O(n) where n = operation count
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
    /// Creates two circuits with specified parameter shifted by +/-shift for gradient evaluation.
    /// Implements parameter shift rule: d<H>/dtheta_i = [<H>(theta_i+pi/2) - <H>(theta_i-pi/2)]/2
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
    ///   - shift: Shift amount (default: pi/2 for standard parameter shift rule)
    /// - Returns: Tuple of (plus, minus) circuits with parameter shifted
    /// - Precondition: Parameter exists in circuit and has binding
    /// - Complexity: O(n) where n = operation count (two circuit bindings)
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
    ///   - shift: Shift amount (default: pi/2)
    /// - Returns: Tuple of (plus, minus) circuits
    /// - Precondition: Index in bounds and vector length matches parameter count
    /// - Complexity: O(n) where n = operation count
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
    /// Ancilla qubits are initialized to |0> and tensor-producted with the original state.
    /// Used internally by ``execute()`` methods to handle circuits that dynamically expand.
    ///
    /// - Parameters:
    ///   - state: Initial quantum state
    ///   - maxQubit: Maximum qubit index needed by circuit operations
    /// - Returns: Expanded state if maxQubit >= state.qubits, otherwise original state
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
    /// Applies all operations sequentially to transform the input state through unitary evolution
    /// and non-unitary operations. Automatically expands the state with ancilla qubits (initialized
    /// to |0>) if circuit operations reference qubit indices beyond the initial state size.
    ///
    /// - Parameter initialState: Starting quantum state (must have qubits matching circuit)
    /// - Returns: Final quantum state after applying all circuit operations
    /// - Precondition: initialState.qubits == qubits
    /// - Precondition: Circuit must contain only concrete parameters (no symbolic)
    /// - Complexity: O(n x 2^q) where n = operation count, q = max qubit index (including ancilla)
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

        let maxQubit = highestQubitIndex
        var currentState = Self.expandStateForAncilla(initialState, maxQubit: maxQubit)

        for operation in operations {
            switch operation {
            case let .gate(gate, qubits, _):
                currentState = GateApplication.apply(
                    gate,
                    to: qubits,
                    state: currentState,
                )
            case let .reset(qubit, _):
                currentState = GateApplication.applyReset(qubit: qubit, state: currentState)
            }
        }

        return currentState
    }

    /// Executes the circuit starting from ground state |00...0>
    ///
    /// Primary execution method for quantum algorithms. Initializes all qubits to computational
    /// basis state |0> and sequentially applies all circuit operations. For large circuits (>=10 qubits)
    /// or GPU acceleration needs, use ``QuantumSimulator`` actor instead.
    ///
    /// - Returns: Final quantum state after applying all circuit operations
    /// - Precondition: Circuit must contain only concrete parameters (no symbolic)
    /// - Complexity: O(n x 2^q) where n = operation count, q = max qubit index
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    /// let bellState = circuit.execute()  // (|00> + |11>)/sqrt(2)
    /// ```
    ///
    /// - SeeAlso: ``QuantumSimulator`` for GPU-accelerated execution, ``execute(on:)`` for custom initial states
    @_optimize(speed)
    @_eagerMove
    public func execute() -> QuantumState {
        let initialState = QuantumState(qubits: qubits)
        return execute(on: initialState)
    }

    /// Executes circuit up to a specific operation index for step-through visualization
    ///
    /// Applies only the first `upToIndex` operations, enabling incremental execution for animation
    /// workflows. External code should cache intermediate states at regular intervals for scrubbing.
    ///
    /// - Parameters:
    ///   - initialState: Starting quantum state
    ///   - upToIndex: Number of operations to execute (0 = no operations, count = all operations)
    /// - Returns: Quantum state after executing operations [0..<upToIndex]
    /// - Precondition: initialState.qubits == qubits
    /// - Precondition: 0 <= upToIndex <= count
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
        ValidationUtilities.validateUpToIndex(upToIndex, operationCount: operations.count)

        let maxQubit = highestQubitIndex
        var currentState = Self.expandStateForAncilla(initialState, maxQubit: maxQubit)

        for i in 0 ..< upToIndex {
            switch operations[i] {
            case let .gate(gate, qubits, _):
                currentState = GateApplication.apply(
                    gate,
                    to: qubits,
                    state: currentState,
                )
            case let .reset(qubit, _):
                currentState = GateApplication.applyReset(qubit: qubit, state: currentState)
            }
        }

        return currentState
    }

    // MARK: - CustomStringConvertible

    /// String representation of the quantum circuit
    public var description: String {
        if operations.isEmpty { return "QuantumCircuit(\(qubits) qubits, empty)" }

        var opList = ""
        let limit = min(operations.count, 5)
        for i in 0 ..< limit {
            if i > 0 { opList += ", " }
            opList += operations[i].description
        }
        let suffix = operations.count > 5 ? ", ..." : ""

        let paramCount = parameterCount
        if paramCount > 0 {
            return "QuantumCircuit(\(qubits) qubits, \(operations.count) ops, \(paramCount) params): \(opList)\(suffix)"
        }

        return "QuantumCircuit(\(qubits) qubits, \(operations.count) ops): \(opList)\(suffix)"
    }

    // MARK: - Circuit Analysis

    /// Circuit depth (minimum sequential time steps assuming unlimited parallelism)
    ///
    /// Depth is the critical path length: gates on different qubits execute in parallel,
    /// gates on same qubit must be sequential. Key metric for quantum hardware execution time
    /// since decoherence limits total circuit duration.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)    // depth 1
    /// circuit.append(.hadamard, to: 1)    // depth 1 (parallel)
    /// circuit.append(.cnot, to: [0, 1])   // depth 2
    /// print(circuit.depth)  // 2
    /// ```
    ///
    /// - Complexity: O(n) where n = operation count
    /// - SeeAlso: ``CircuitOptimizer/computeDepth(_:)``
    @_optimize(speed)
    public var depth: Int {
        CircuitOptimizer.computeDepth(self)
    }

    /// Gate count by type for resource estimation
    ///
    /// Returns dictionary mapping each gate type to its occurrence count. Useful for
    /// comparing algorithm costs, estimating hardware error rates (different gates have
    /// different fidelities), and tracking optimization effectiveness.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.grover(qubits: 3, target: 5)
    /// let counts = circuit.gateCount
    /// print("CNOTs: \(counts[.cnot] ?? 0)")
    /// print("Hadamards: \(counts[.hadamard] ?? 0)")
    /// ```
    ///
    /// - Complexity: O(n) where n = operation count
    /// - SeeAlso: ``CircuitOptimizer/gateCount(_:)``
    @_optimize(speed)
    public var gateCount: [QuantumGate: Int] {
        CircuitOptimizer.gateCount(self)
    }

    /// CNOT-equivalent two-qubit gate count
    ///
    /// CNOTs are the standard metric for two-qubit gate cost on most hardware.
    /// Other two-qubit gates convert to CNOT-equivalents: CZ = 1, SWAP = 3, Toffoli = 6.
    /// Lower CNOT count generally means higher circuit fidelity.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 3)
    /// circuit.append(.swap, to: [0, 1])  // 3 CNOTs equivalent
    /// circuit.append(.cnot, to: [1, 2])  // 1 CNOT
    /// print(circuit.cnotCount)  // 4
    /// ```
    ///
    /// - Complexity: O(n) where n = operation count
    /// - SeeAlso: ``CircuitOptimizer/cnotEquivalentCount(_:)``
    @_optimize(speed)
    public var cnotCount: Int {
        CircuitOptimizer.cnotEquivalentCount(self)
    }

    // MARK: - Circuit Transformations

    /// Inverse circuit implementing U dagger where original circuit implements U
    ///
    /// Constructs the adjoint circuit by reversing operation order and taking each gate's inverse:
    /// (G1 G2 ... Gn) dagger = Gn dagger ... G2 dagger G1 dagger. The inverse satisfies
    /// U dagger U = U U dagger = I, making it essential for uncomputation in phase estimation,
    /// error mitigation, and reversible computing.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    /// circuit.append(.rotationZ(.pi/4), to: 1)
    ///
    /// let inverse = circuit.inverse()
    /// // Contains: Rz(-pi/4) on 1, CNOT on [0,1], H on 0
    /// ```
    ///
    /// - Returns: New circuit implementing U dagger
    /// - Complexity: O(n) where n = operation count
    /// - Precondition: Circuit must contain only unitary (reversible) operations
    /// - Note: Symbolic parameters are preserved with negated angles where applicable
    /// - SeeAlso: ``QuantumGate/inverse``
    @_optimize(speed)
    @_eagerMove
    public func inverse() -> QuantumCircuit {
        ValidationUtilities.validateUnitaryCircuit(self)

        var result = QuantumCircuit(qubits: qubits)

        for operation in operations.reversed() {
            result.append(operation.gate!.inverse, to: operation.qubits, timestamp: operation.timestamp)
        }

        return result
    }

    /// Optimized circuit with reduced gate count and depth
    ///
    /// Applies the full optimization pipeline: identity cancellation removes adjacent inverse
    /// pairs (H-H, CNOT-CNOT), single-qubit fusion merges consecutive rotations, commutation
    /// reordering brings cancellable pairs adjacent. Result is semantically equivalent to
    /// original but typically has fewer gates and lower depth.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.hadamard, to: 0)  // Cancels with previous
    /// circuit.append(.rotationZ(.pi/4), to: 1)
    /// circuit.append(.rotationZ(.pi/4), to: 1)  // Merges to Rz(pi/2)
    ///
    /// let optimized = circuit.optimized()
    /// print(circuit.count)     // 4
    /// print(optimized.count)   // 1 (just Rz(pi/2))
    /// ```
    ///
    /// - Returns: Optimized circuit (may be same if no optimizations apply)
    /// - Complexity: O(n^2) where n = operation count
    /// - Precondition: Circuit must contain only concrete parameters
    /// - SeeAlso: ``CircuitOptimizer/optimize(_:)``
    @_optimize(speed)
    @_eagerMove
    public func optimized() -> QuantumCircuit {
        CircuitOptimizer.optimize(self)
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
    /// - Complexity: O(B x n) where B = batch size, n = operation count
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
    /// by +/-pi/2. Formula: d<H>/dtheta_i = [<H>(theta_i+pi/2) - <H>(theta_i-pi/2)]/2
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
    ///   - shift: Shift amount (default: pi/2 for standard parameter shift)
    /// - Returns: Tuple of (plus vectors, minus vectors)
    /// - Precondition: Base vector length must equal ``parameterCount``
    /// - Complexity: O(N^2) where N = parameter count (2N vectors of length N)
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
    /// Creates a Bell circuit: H0 * CNOT01
    ///
    /// Produces maximally entangled two-qubit state: (|00> + |11>)/sqrt(2)
    ///
    /// - Returns: 2-qubit circuit generating Bell state |Phi+>
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

    /// Creates a GHZ circuit: H0 * CNOT01 * CNOT02 * ...
    ///
    /// Produces n-qubit maximally entangled Greenberger-Horne-Zeilinger state: (|00...0> + |11...1>)/sqrt(2)
    ///
    /// - Parameter qubits: Number of qubits (minimum 2, default 3)
    /// - Returns: n-qubit circuit generating GHZ state
    /// - Precondition: qubits >= 2
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
    /// Produces equal superposition over all basis states: (sum_i|i>)/sqrt(2^n)
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.uniformSuperposition(qubits: 3)
    /// let state = circuit.execute()
    /// // All 8 basis states have probability 1/8
    /// ```
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
    /// Implements |j> -> (1/sqrt(N)) sum_k exp(2 pi i j k / N)|k> where N = 2^n, a key component
    /// in Shor's algorithm and quantum phase estimation. Applies Hadamard to each qubit followed
    /// by controlled-phase gates with angles pi/2, pi/4, pi/8, ... (decreasing powers of 2),
    /// then reverses qubit order with SWAP gates.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.qft(qubits: 3)
    /// let state = QuantumState(qubits: 3)  // |000>
    /// let transformed = circuit.execute(on: state)
    /// // Result: uniform superposition (|000> + |001> + ... + |111>)/sqrt(8)
    /// ```
    ///
    /// - Parameter qubits: Number of qubits (typical range: 3-8, maximum: 16)
    /// - Returns: Circuit implementing QFT
    /// - Precondition: qubits <= 16
    /// - Complexity: O(n^2) gates where n = qubits
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
    /// Implements QFT dagger by applying gates in reverse order with negated phase angles.
    ///
    /// **Example:**
    /// ```swift
    /// let qftCircuit = QuantumCircuit.qft(qubits: 3)
    /// let inverseCircuit = QuantumCircuit.inverseQFT(qubits: 3)
    ///
    /// var combined = qftCircuit
    /// for op in inverseCircuit.operations {
    ///     combined.addOperation(op)
    /// }
    /// // combined circuit is effectively identity (QFT * QFT dagger = I)
    /// ```
    ///
    /// - Parameter qubits: Number of qubits
    /// - Returns: Circuit implementing inverse QFT
    /// - Complexity: O(n^2) gates where n = qubits
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

    /// Creates Quantum Phase Estimation circuit for eigenvalue extraction.
    ///
    /// Implements QPE algorithm that extracts the phase phi from eigenvalue equation
    /// U|psi> = e^(2 pi i phi)|psi>. Uses precision qubits to store the binary representation
    /// of phi and controlled-U^(2^k) operations to encode phase information, followed by
    /// inverse QFT to extract the phase bits.
    ///
    /// The circuit structure:
    /// 1. Apply Hadamard to all precision qubits (superposition)
    /// 2. Apply controlled-U^(2^k) for k in 0..<precisionQubits (phase kickback)
    /// 3. Apply inverse QFT to precision register (extract phase)
    ///
    /// **Example:**
    /// ```swift
    /// let qpeCircuit = QuantumCircuit.phaseEstimation(
    ///     unitary: .pauliZ,
    ///     precisionQubits: 3,
    ///     eigenstateQubits: 1
    /// )
    /// let state = qpeCircuit.execute()
    /// let (result, _) = state.mostProbableState()
    /// ```
    ///
    /// - Parameters:
    ///   - unitary: Single-qubit unitary gate U whose eigenvalue phase to estimate
    ///   - precisionQubits: Number of qubits for phase precision (determines accuracy as 2^(-n))
    ///   - eigenstateQubits: Number of qubits for eigenstate register (default: 1)
    /// - Returns: Circuit implementing quantum phase estimation
    /// - Precondition: precisionQubits >= 1
    /// - Precondition: eigenstateQubits >= 1
    /// - Precondition: unitary must be a single-qubit gate
    /// - Complexity: O(n^2) gates where n = precisionQubits
    ///
    /// - SeeAlso: ``inverseQFT(qubits:)``
    /// - SeeAlso: ``ControlledGateDecomposer/controlledPower(of:power:control:targetQubits:)``
    @_eagerMove
    static func phaseEstimation(
        unitary: QuantumGate,
        precisionQubits: Int,
        eigenstateQubits: Int = 1,
    ) -> QuantumCircuit {
        ValidationUtilities.validatePositiveInt(precisionQubits, name: "precisionQubits")
        ValidationUtilities.validatePositiveInt(eigenstateQubits, name: "eigenstateQubits")
        ValidationUtilities.validateControlledGateIsSingleQubit(unitary.qubitsRequired)

        let totalQubits = precisionQubits + eigenstateQubits
        var circuit = QuantumCircuit(qubits: totalQubits)

        for i in 0 ..< precisionQubits {
            circuit.append(.hadamard, to: i)
        }

        let targetQubits = Array(precisionQubits ..< totalQubits)

        for k in 0 ..< precisionQubits {
            let controlQubit = precisionQubits - 1 - k
            let gates = ControlledGateDecomposer.controlledPower(
                of: unitary,
                power: k,
                control: controlQubit,
                targetQubits: targetQubits,
            )

            for (gate, qubits) in gates {
                circuit.append(gate, to: qubits)
            }
        }

        let inverseQFTGates = inverseQFTGates(qubits: precisionQubits)
        for (gate, qubits) in inverseQFTGates {
            circuit.append(gate, to: qubits)
        }

        return circuit
    }

    /// Generate inverse QFT gate sequence for specified qubit range.
    ///
    /// Produces the gate sequence for inverse Quantum Fourier Transform operating on
    /// qubits 0..<qubits. Swaps qubits first to reverse bit order, then applies inverse
    /// controlled rotations and Hadamards in reverse order of forward QFT.
    ///
    /// - Parameter qubits: Number of qubits for inverse QFT
    /// - Returns: Array of (gate, qubits) tuples implementing inverse QFT
    /// - Complexity: O(n^2) gates where n = qubits
    @_optimize(speed)
    private static func inverseQFTGates(qubits: Int) -> [(gate: QuantumGate, qubits: [Int])] {
        var gates: [(gate: QuantumGate, qubits: [Int])] = []

        let swapCount = qubits / 2
        for i in 0 ..< swapCount {
            let j = qubits - 1 - i
            gates.append((.swap, [i, j]))
        }

        for target in (0 ..< qubits).reversed() {
            for control in (target + 1 ..< qubits).reversed() {
                let k = control - target + 1
                let theta = -Double.pi / Double(1 << k)
                gates.append((.controlledPhase(theta), [control, target]))
            }

            gates.append((.hadamard, [target]))
        }

        return gates
    }

    /// Creates Grover search algorithm circuit for O(sqrt(N)) unstructured search.
    ///
    /// Initializes uniform superposition, then repeats oracle (phase flip on target) and diffusion
    /// (inversion about average) for optimal iterations floor(pi/4 * sqrt(2^n)). Achieves quadratic
    /// speedup over classical O(N) search.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.grover(qubits: 3, target: 5)
    /// let state = circuit.execute()
    /// let (mostLikely, prob) = state.mostProbableState()
    /// // mostLikely = 5 with high probability
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (search space size = 2^n, maximum n=10)
    ///   - target: Target state to search for (basis state index, 0 <= target < 2^n)
    ///   - iterations: Number of Grover iterations (defaults to optimal: floor(pi/4 x sqrt(2^n)))
    /// - Returns: Circuit implementing Grover search
    /// - Precondition: qubits <= 10
    /// - Precondition: 0 <= target < 2^n
    /// - Complexity: O(k x n x 2^n) where k = iterations, n = qubits (execution cost)
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
    /// - 3 qubits: Hadamard sandwich around Toffoli (H*CCNOT*H)
    /// - n>=4 qubits: Hadamard sandwich around multi-controlled X with ancilla
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

    /// Appends multi-controlled X (NOT) gate using ladder decomposition with ancilla qubits.
    ///
    /// Implements C^n(X) gate using Toffoli decomposition for arbitrary control count, with ancilla
    /// qubits automatically allocated beyond the maximum qubit index. For zero controls this reduces
    /// to an X gate, for one control a CNOT, and for two controls a Toffoli. With n>=3 controls,
    /// the decomposition uses a ladder of Toffoli gates requiring n-2 ancilla qubits.
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
    /// Implements C^n(Y) by converting to controlled-X via phase gates:
    /// C^n(Y) = S dagger * C^n(X) * S where S dagger rotates Y basis to X basis.
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
    /// Implements C^n(Z) by converting to controlled-X via Hadamard conjugation:
    /// C^n(Z) = H * C^n(X) * H where H rotates Z basis to X basis.
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
    /// Applies gate U with n controls: |1>^n|psi> -> |1>^n U|psi>. Pauli gates use optimized
    /// basis-specific decompositions, Hadamard uses Ry sandwich around controlled-Z, and general
    /// unitaries use Gray code decomposition U * C^n(X) * U dagger * C^n(X) * U.
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
    /// D = 2|psi><psi| - I where |psi> = H^n|0>. Gate sequence: H^n * X^n * C^n(Z) * X^n * H^n.
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
    /// Implements adiabatic evolution from transverse-field Hamiltonian H0 to problem Hamiltonian Hp
    /// via interpolation H(t) = (1-t)H0 + tHp. Initializes in superposition, evolves through discrete
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
    /// - Precondition: qubits <= 8
    /// - Complexity: O(s x n^2) gates where s = annealingSteps, n = qubits
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
    /// Encodes combinatorial optimization as Ising Hamiltonian H = sum_i h_i Z_i + sum_ij J_ij Z_i Z_j
    /// where h_i are local fields and J_ij are qubit-qubit couplings. Applicable to MaxCut, constraint
    /// satisfaction, QUBO, and traveling salesman problems. Transverse field enables quantum tunneling.
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

        /// Parses qubit indices from coupling key string (e.g., "0-1" or "01" or "0").
        private static func parseQubitIndices(from key: String) -> [Int] {
            ValidationUtilities.validateNonEmptyString(key, name: "Coupling key")

            let hasSeparator = key.contains { !$0.isNumber }

            if hasSeparator {
                return key.split { !$0.isNumber }.compactMap { Int($0) }
            } else if key.count == 2 {
                return key.compactMap(\.wholeNumberValue)
            } else {
                return [Int(key)!]
            }
        }

        /// Creates Ising problem for quadratic function minimization
        ///
        /// **Example:**
        /// ```swift
        /// let problem = IsingProblem.quadraticMinimum(qubits: 4)
        /// let circuit = QuantumCircuit.annealing(qubits: 4, problem: problem)
        /// ```
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
    /// Implements exp(-i theta Z_i Z_j) decomposition using CNOT ladder:
    /// CNOT(i,j) * Rz(2 theta) * CNOT(i,j) = exp(-i theta Z_i Z_j)
    ///
    /// - Complexity: O(1) - constant 3 gates
    private static func appendZZCoupling(to circuit: inout QuantumCircuit, qubit1: Int, qubit2: Int, angle: Double) {
        circuit.append(.cnot, to: [qubit1, qubit2])
        circuit.append(.rotationZ(2.0 * angle), to: qubit2)
        circuit.append(.cnot, to: [qubit1, qubit2])
    }
}
