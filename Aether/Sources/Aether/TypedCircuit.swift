// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Phantom type protocol encoding qubit count at the type level for compile-time circuit validation.
///
/// Swift lacks value-level generic parameters, so ``QubitSize`` encodes qubit counts as
/// distinct types. Conforming enums (``Q1`` through ``Q8``, ``Q16``) carry a static ``count``
/// that ``TypedCircuit`` reads at initialization to construct the underlying ``QuantumCircuit``
/// with the correct number of qubits. The compiler rejects mismatched circuit compositions
/// because ``TypedCircuit<Q2>`` and ``TypedCircuit<Q3>`` are unrelated types, catching
/// dimension errors before runtime.
///
/// **Example:**
/// ```swift
/// let bell = TypedCircuit<Q2> { GateStep(.hadamard, on: 0); GateStep(.cnot, on: [0, 1]) }
/// let state = bell.execute()
/// print(state.qubits)  // 2
/// ```
///
/// - SeeAlso: ``TypedCircuit``
/// - SeeAlso: ``QubitSum``
public protocol QubitSize: Sendable {
    /// Number of qubits this type represents.
    static var count: Int { get }
}

/// Phantom type representing a single-qubit register.
///
/// **Example:**
/// ```swift
/// let count = Q1.count
/// let single = TypedCircuit<Q1> { GateStep(.hadamard, on: 0) }
/// let state = single.execute()
/// ```
public enum Q1: QubitSize {
    @inlinable
    public static var count: Int { 1 }
}

/// Phantom type representing a two-qubit register.
///
/// **Example:**
/// ```swift
/// let count = Q2.count
/// let bell = TypedCircuit<Q2> { GateStep(.hadamard, on: 0); GateStep(.cnot, on: [0, 1]) }
/// let state = bell.execute()
/// ```
public enum Q2: QubitSize {
    @inlinable
    public static var count: Int { 2 }
}

/// Phantom type representing a three-qubit register.
///
/// **Example:**
/// ```swift
/// let count = Q3.count
/// let circuit = TypedCircuit<Q3> { GateStep(.hadamard, on: 0) }
/// print(circuit.circuit.qubits)
/// ```
public enum Q3: QubitSize {
    @inlinable
    public static var count: Int { 3 }
}

/// Phantom type representing a four-qubit register.
///
/// **Example:**
/// ```swift
/// let count = Q4.count
/// let typed = TypedCircuit<Q4> { GateStep(.hadamard, on: 0) }
/// print(typed.circuit.qubits)  // 4
/// ```
public enum Q4: QubitSize {
    @inlinable
    public static var count: Int { 4 }
}

/// Phantom type representing a five-qubit register.
///
/// **Example:**
/// ```swift
/// let count = Q5.count
/// let typed = TypedCircuit<Q5> { GateStep(.hadamard, on: 0) }
/// print(typed.circuit.qubits)  // 5
/// ```
public enum Q5: QubitSize {
    @inlinable
    public static var count: Int { 5 }
}

/// Phantom type representing a six-qubit register.
///
/// **Example:**
/// ```swift
/// let count = Q6.count
/// let typed = TypedCircuit<Q6> { GateStep(.hadamard, on: 0) }
/// print(typed.circuit.qubits)  // 6
/// ```
public enum Q6: QubitSize {
    @inlinable
    public static var count: Int { 6 }
}

/// Phantom type representing a seven-qubit register.
///
/// **Example:**
/// ```swift
/// let count = Q7.count
/// let typed = TypedCircuit<Q7> { GateStep(.hadamard, on: 0) }
/// print(typed.circuit.qubits)  // 7
/// ```
public enum Q7: QubitSize {
    @inlinable
    public static var count: Int { 7 }
}

/// Phantom type representing an eight-qubit register.
///
/// **Example:**
/// ```swift
/// let count = Q8.count
/// let typed = TypedCircuit<Q8> { GateStep(.hadamard, on: 0) }
/// print(typed.circuit.qubits)  // 8
/// ```
public enum Q8: QubitSize {
    @inlinable
    public static var count: Int { 8 }
}

/// Phantom type representing a sixteen-qubit register.
///
/// **Example:**
/// ```swift
/// let count = Q16.count
/// let typed = TypedCircuit<Q16> { GateStep(.hadamard, on: 0) }
/// print(typed.circuit.qubits)  // 16
/// ```
public enum Q16: QubitSize {
    @inlinable
    public static var count: Int { 16 }
}

/// Type-level addition of two ``QubitSize`` types for tensor product composition.
///
/// Enables ``TypedCircuit/composing(_:_:)`` to statically compute the output register
/// size when two circuits are composed in parallel. The resulting ``QubitSum`` conforms
/// to ``QubitSize`` with ``count`` equal to the sum of its two operand counts, so the
/// compiler enforces that the combined circuit has exactly the right number of qubits.
///
/// **Example:**
/// ```swift
/// let sum = QubitSum<Q2, Q3>.count
/// let expected = Q2.count + Q3.count
/// print(sum == expected)
/// ```
///
/// - SeeAlso: ``QubitSize``
/// - SeeAlso: ``TypedCircuit/composing(_:_:)``
public enum QubitSum<A: QubitSize, B: QubitSize>: QubitSize {
    @inlinable
    public static var count: Int { A.count + B.count }
}

/// Compile-time qubit-count-safe wrapper around ``QuantumCircuit``.
///
/// ``TypedCircuit`` parameterizes a quantum circuit by a ``QubitSize`` phantom type so the
/// Swift type checker prevents combining circuits with incompatible qubit counts. The
/// underlying ``QuantumCircuit`` is constructed with exactly ``Size/count`` qubits, and the
/// ``composing(_:_:)`` factory method uses ``QubitSum`` to produce a correctly-sized output
/// type without any runtime branching.
///
/// **Example:**
/// ```swift
/// let bell = TypedCircuit<Q2> { GateStep(.hadamard, on: 0); GateStep(.cnot, on: [0, 1]) }
/// let state = bell.execute()
/// print(state.qubits)  // 2
/// ```
///
/// - SeeAlso: ``QubitSize``
/// - SeeAlso: ``QubitSum``
/// - SeeAlso: ``QuantumCircuit``
@frozen
public struct TypedCircuit<Size: QubitSize>: Sendable {
    /// The underlying quantum circuit with ``Size/count`` qubits.
    public let circuit: QuantumCircuit

    /// Creates a typed circuit by building operations from a ``QuantumCircuitBuilder`` closure.
    ///
    /// The circuit is constructed with ``Size/count`` qubits and populated with the gate steps
    /// produced by the result builder. Each ``GateStep`` is converted to a ``CircuitOperation``
    /// and appended in declaration order.
    ///
    /// **Example:**
    /// ```swift
    /// let bell = TypedCircuit<Q2> { GateStep(.hadamard, on: 0); GateStep(.cnot, on: [0, 1]) }
    /// let state = bell.execute()
    /// print(state.probability(of: 0b00))
    /// ```
    ///
    /// - Parameter build: Result builder closure producing an array of ``GateStep`` values
    /// - Complexity: O(n) where n is the number of gate steps
    public init(@QuantumCircuitBuilder _ build: () -> [GateStep]) {
        let steps = build()
        var qc = QuantumCircuit(qubits: Size.count)
        for step in steps {
            qc.append(step.gate, to: step.qubits)
        }
        circuit = qc
    }

    /// Creates a typed circuit by wrapping an existing ``QuantumCircuit``.
    ///
    /// Validates at runtime that the circuit's qubit count matches ``Size/count``. Use this
    /// initializer when interfacing with untyped circuit APIs where the qubit count is known
    /// but not encoded in the type system.
    ///
    /// **Example:**
    /// ```swift
    /// var qc = QuantumCircuit(qubits: 2)
    /// qc.append(.hadamard, to: 0)
    /// let typed = TypedCircuit<Q2>(qc)
    /// ```
    ///
    /// - Parameter circuit: Quantum circuit whose qubit count must equal ``Size/count``
    /// - Precondition: circuit.qubits == Size.count
    /// - Complexity: O(1)
    public init(_ circuit: QuantumCircuit) {
        ValidationUtilities.validateQubitCountsEqual(
            circuit.qubits,
            Size.count,
            name1: "circuit.qubits",
            name2: "Size.count",
        )
        self.circuit = circuit
    }

    /// Executes the circuit from the ground state |00...0> and returns the resulting quantum state.
    ///
    /// Delegates to the underlying ``QuantumCircuit/execute()`` method, applying all operations
    /// sequentially to the computational basis state with all qubits initialized to |0>.
    ///
    /// **Example:**
    /// ```swift
    /// let bell = TypedCircuit<Q2> { GateStep(.hadamard, on: 0); GateStep(.cnot, on: [0, 1]) }
    /// let state = bell.execute()
    /// print(state.probability(of: 0b00))  // 0.5
    /// ```
    ///
    /// - Returns: Final quantum state after all circuit operations
    /// - Precondition: Circuit must contain only concrete parameters
    /// - Complexity: O(n x 2^q) where n = operation count, q = qubit count
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public func execute() -> QuantumState {
        circuit.execute()
    }

    /// Executes the circuit on a provided initial quantum state and returns the result.
    ///
    /// Delegates to the underlying ``QuantumCircuit/execute(on:)`` method, applying all
    /// operations sequentially to transform the given initial state.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = TypedCircuit<Q2> { GateStep(.hadamard, on: 0) }
    /// let initial = QuantumState(qubits: 2)
    /// let result = circuit.execute(on: initial)
    /// ```
    ///
    /// - Parameter initialState: Starting quantum state with qubit count matching the circuit
    /// - Returns: Final quantum state after all circuit operations
    /// - Precondition: initialState.qubits == Size.count
    /// - Precondition: Circuit must contain only concrete parameters
    /// - Complexity: O(n x 2^q) where n = operation count, q = qubit count
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public func execute(on initialState: QuantumState) -> QuantumState {
        circuit.execute(on: initialState)
    }

    /// Composes two typed circuits in parallel via tensor product, producing a circuit
    /// whose qubit count is the sum of the two inputs.
    ///
    /// Operations from the first circuit retain their original qubit indices. Operations
    /// from the second circuit have their qubit indices shifted upward by ``A/count`` so
    /// the two circuits occupy disjoint qubit registers in the combined system. The result
    /// type ``TypedCircuit<QubitSum<A, B>>`` encodes the total qubit count at compile time.
    ///
    /// **Example:**
    /// ```swift
    /// let a = TypedCircuit<Q2> { GateStep(.hadamard, on: 0); GateStep(.cnot, on: [0, 1]) }
    /// let b = TypedCircuit<Q1> { GateStep(.pauliX, on: 0) }
    /// let combined = TypedCircuit.composing(a, b)
    /// ```
    ///
    /// - Parameters:
    ///   - a: First circuit occupying qubits 0 ..< A.count
    ///   - b: Second circuit occupying qubits A.count ..< A.count + B.count
    /// - Returns: Combined circuit with qubit count A.count + B.count
    /// - Complexity: O(m + n) where m and n are the operation counts of the two circuits
    ///
    /// - SeeAlso: ``QubitSum``
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func composing<A: QubitSize, B: QubitSize>(
        _ a: TypedCircuit<A>,
        _ b: TypedCircuit<B>,
    ) -> TypedCircuit<QubitSum<A, B>> where Size == QubitSum<A, B> {
        let totalQubits = QubitSum<A, B>.count
        let shift = A.count
        var combined = QuantumCircuit(qubits: totalQubits)

        for operation in a.circuit.operations {
            combined.addOperation(operation)
        }

        for operation in b.circuit.operations {
            let shifted = shiftOperation(operation, by: shift)
            combined.addOperation(shifted)
        }

        return TypedCircuit<QubitSum<A, B>>(combined)
    }

    /// Shifts all qubit indices in a circuit operation by the given offset.
    @inline(__always)
    private static func shiftOperation(_ operation: CircuitOperation, by offset: Int) -> CircuitOperation {
        switch operation {
        case let .gate(gate, qubits, timestamp):
            let shifted = qubits.map { $0 + offset }
            return .gate(gate, qubits: shifted, timestamp: timestamp)
        case let .reset(qubit, timestamp):
            return .reset(qubit: qubit + offset, timestamp: timestamp)
        case let .measure(qubit, classicalBit, timestamp):
            return .measure(qubit: qubit + offset, classicalBit: classicalBit.map { $0 + offset }, timestamp: timestamp)
        }
    }
}
