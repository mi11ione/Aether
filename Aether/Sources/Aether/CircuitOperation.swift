// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Non-unitary quantum operation for circuit append and insert APIs.
///
/// Enumerates irreversible operations that lack unitary matrix representations.
/// Non-unitary operations break time-reversal symmetry and cannot appear in
/// unitary circuit compilation or circuit-to-matrix conversion. They are tracked
/// separately from ``QuantumGate`` to preserve type-level distinction between
/// reversible and irreversible circuit elements.
///
/// **Example:**
/// ```swift
/// let operation = NonUnitaryOperation.reset
/// print(operation == .reset)  // true
/// ```
///
/// - SeeAlso: ``CircuitOperation``
/// - SeeAlso: ``QuantumGate``
@frozen
public enum NonUnitaryOperation: Equatable, Hashable, Sendable {
    /// Mid-circuit reset projecting target qubit to |0> regardless of input state.
    ///
    /// Reset deterministically projects the target qubit to the computational basis
    /// state |0> by measuring and conditionally applying an X gate. Unlike unitary gates,
    /// reset is irreversible and has no matrix representation.
    ///
    /// **Example:**
    /// ```swift
    /// let op = NonUnitaryOperation.reset
    /// print(op)  // reset
    /// ```
    case reset

    /// Mid-circuit measurement projecting target qubit onto the computational basis.
    ///
    /// Measurement collapses the target qubit's quantum state to either |0> or |1>,
    /// storing the classical result. Unlike unitary gates, measurement is irreversible
    /// and introduces decoherence into the quantum state.
    ///
    /// **Example:**
    /// ```swift
    /// let op = NonUnitaryOperation.measure
    /// print(op == .measure)  // true
    /// ```
    case measure
}

/// Circuit-level operation wrapping unitary gates and non-unitary operations with qubit targets and optional timestamps.
///
/// Separates unitary gates from non-unitary operations at the operation level, providing
/// a uniform interface for circuit construction, parameter management, and introspection.
/// Each operation records the qubits it acts on and an optional timestamp for scheduling
/// in time-ordered circuit representations. The ``gate`` case wraps a ``QuantumGate`` with
/// its target qubits, while the ``reset`` case represents a mid-circuit qubit reset.
///
/// Parameter management delegates to the underlying ``QuantumGate`` for gate operations,
/// enabling variational circuit workflows where symbolic parameters are bound to concrete
/// values before execution. Non-unitary operations have no parameters and always return
/// empty sets.
///
/// **Example:**
/// ```swift
/// let hadamardOp = CircuitOperation.gate(.hadamard, qubits: [0])
/// let cnotOp = CircuitOperation.gate(.cnot, qubits: [0, 1], timestamp: 1.0)
/// let resetOp = CircuitOperation.reset(qubit: 2)
///
/// print(hadamardOp.isUnitary)       // true
/// print(resetOp.isUnitary)          // false
/// print(cnotOp.timestamp)           // Optional(1.0)
/// print(hadamardOp.qubits)          // [0]
/// ```
///
/// - SeeAlso: ``QuantumGate``
/// - SeeAlso: ``NonUnitaryOperation``
/// - SeeAlso: ``QuantumCircuit``
/// - SeeAlso: ``Parameter``
@frozen
public enum CircuitOperation: Equatable, Hashable, CustomStringConvertible, Sendable {
    /// Unitary gate applied to specified qubits with optional scheduling timestamp.
    ///
    /// Wraps a ``QuantumGate`` together with the qubit indices it targets and an optional
    /// timestamp for time-ordered circuit representations.
    ///
    /// **Example:**
    /// ```swift
    /// let op = CircuitOperation.gate(.hadamard, qubits: [0])
    /// let timedOp = CircuitOperation.gate(.cnot, qubits: [0, 1], timestamp: 2.5)
    /// ```
    case gate(_ gate: QuantumGate, qubits: [Int], timestamp: Double? = nil)

    /// Mid-circuit reset projecting specified qubit to |0> with optional scheduling timestamp.
    ///
    /// Represents an irreversible reset operation on a single qubit. Unlike gate operations,
    /// reset has no matrix representation and cannot be reversed.
    ///
    /// **Example:**
    /// ```swift
    /// let op = CircuitOperation.reset(qubit: 0)
    /// let timedReset = CircuitOperation.reset(qubit: 1, timestamp: 3.0)
    /// ```
    case reset(qubit: Int, timestamp: Double? = nil)

    /// Mid-circuit measurement projecting specified qubit onto the computational basis with optional classical bit and scheduling timestamp.
    ///
    /// Represents an irreversible measurement operation on a single qubit. The classical bit
    /// defaults to the same index as the qubit if not specified. Unlike gate operations,
    /// measurement has no matrix representation and collapses the quantum state.
    ///
    /// **Example:**
    /// ```swift
    /// let op = CircuitOperation.measure(qubit: 0)
    /// let timedMeasure = CircuitOperation.measure(qubit: 1, classicalBit: 1, timestamp: 3.0)
    /// ```
    case measure(qubit: Int, classicalBit: Int? = nil, timestamp: Double? = nil)

    // MARK: - Qubit Access

    /// Qubit indices targeted by this operation.
    ///
    /// For gate operations, returns the full qubit array including control and target qubits.
    /// For reset operations, returns a single-element array containing the reset qubit index.
    ///
    /// **Example:**
    /// ```swift
    /// let gateOp = CircuitOperation.gate(.cnot, qubits: [0, 1])
    /// print(gateOp.qubits)  // [0, 1]
    ///
    /// let resetOp = CircuitOperation.reset(qubit: 2)
    /// print(resetOp.qubits)  // [2]
    /// ```
    ///
    /// - Returns: Array of qubit indices this operation acts on
    /// - Complexity: O(1)
    @inlinable
    public var qubits: [Int] {
        switch self {
        case let .gate(_, qubits, _):
            qubits
        case let .reset(qubit, _):
            [qubit]
        case let .measure(qubit, _, _):
            [qubit]
        }
    }

    // MARK: - Gate Access

    /// The unitary gate for this operation, or `nil` for non-unitary operations.
    ///
    /// Provides convenient access to the gate without pattern matching.
    /// Returns the ``QuantumGate`` for `.gate` cases and `nil` for `.reset` and
    /// other non-unitary operations.
    ///
    /// - Complexity: O(1)
    ///
    /// **Example:**
    /// ```swift
    /// let op = CircuitOperation.gate(.hadamard, qubits: [0])
    /// let gate = op.gate   // Optional(QuantumGate.hadamard)
    ///
    /// let reset = CircuitOperation.reset(qubit: 0)
    /// let none = reset.gate // nil
    /// ```
    @inlinable
    public var gate: QuantumGate? {
        switch self {
        case let .gate(gate, _, _):
            gate
        case .reset:
            nil
        case .measure:
            nil
        }
    }

    // MARK: - Timestamp Access

    /// Optional scheduling timestamp for time-ordered circuit representations.
    ///
    /// Returns the timestamp associated with this operation if one was provided during
    /// construction. Used for ordering operations in circuits that support temporal scheduling.
    ///
    /// **Example:**
    /// ```swift
    /// let timed = CircuitOperation.gate(.hadamard, qubits: [0], timestamp: 1.5)
    /// print(timed.timestamp)  // Optional(1.5)
    ///
    /// let untimed = CircuitOperation.reset(qubit: 0)
    /// print(untimed.timestamp)  // nil
    /// ```
    ///
    /// - Returns: Scheduling timestamp if set, `nil` otherwise
    /// - Complexity: O(1)
    @inlinable
    public var timestamp: Double? {
        switch self {
        case let .gate(_, _, timestamp):
            timestamp
        case let .reset(_, timestamp):
            timestamp
        case let .measure(_, _, timestamp):
            timestamp
        }
    }

    // MARK: - Unitarity

    /// Whether this operation is a unitary (reversible) transformation.
    ///
    /// Gate operations are unitary and preserve quantum state normalization through
    /// the condition U dagger U equals I. Reset operations are non-unitary and irreversible.
    ///
    /// **Example:**
    /// ```swift
    /// let gateOp = CircuitOperation.gate(.pauliX, qubits: [0])
    /// print(gateOp.isUnitary)  // true
    ///
    /// let resetOp = CircuitOperation.reset(qubit: 0)
    /// print(resetOp.isUnitary)  // false
    /// ```
    ///
    /// - Returns: `true` for gate operations, `false` for reset operations
    /// - Complexity: O(1)
    @inlinable
    public var isUnitary: Bool {
        switch self {
        case .gate: true
        case .reset: false
        case .measure: false
        }
    }

    // MARK: - Parameterization

    /// Whether this operation contains symbolic parameters requiring binding before execution.
    ///
    /// Delegates to the underlying ``QuantumGate/isParameterized`` for gate operations.
    /// Reset operations are never parameterized and always return `false`.
    ///
    /// **Example:**
    /// ```swift
    /// let theta = Parameter(name: "theta")
    /// let paramOp = CircuitOperation.gate(.rotationY(theta), qubits: [0])
    /// print(paramOp.isParameterized)  // true
    ///
    /// let concreteOp = CircuitOperation.gate(.hadamard, qubits: [0])
    /// print(concreteOp.isParameterized)  // false
    ///
    /// let resetOp = CircuitOperation.reset(qubit: 0)
    /// print(resetOp.isParameterized)  // false
    /// ```
    ///
    /// - Returns: `true` if operation contains unbound symbolic parameters
    /// - Complexity: O(1)
    @inlinable
    public var isParameterized: Bool {
        switch self {
        case let .gate(gate, _, _):
            gate.isParameterized
        case .reset:
            false
        case .measure:
            false
        }
    }

    // MARK: - Parameter Extraction

    /// Extract all symbolic parameters from this operation.
    ///
    /// Delegates to the underlying ``QuantumGate/parameters()`` for gate operations,
    /// returning the set of ``Parameter`` instances that must be bound before execution.
    /// Reset operations contain no parameters and return an empty set.
    ///
    /// **Example:**
    /// ```swift
    /// let theta = Parameter(name: "theta")
    /// let phi = Parameter(name: "phi")
    /// let op = CircuitOperation.gate(
    ///     .u2(phi: .parameter(phi), lambda: .parameter(theta)),
    ///     qubits: [0]
    /// )
    /// let params = op.parameters()  // {theta, phi}
    ///
    /// let resetOp = CircuitOperation.reset(qubit: 0)
    /// let empty = resetOp.parameters()  // []
    /// ```
    ///
    /// - Returns: Set of symbolic parameters in this operation
    /// - Complexity: O(1) for gates with at most 3 parameters, O(1) for reset
    @_optimize(speed)
    @_effects(readonly)
    public func parameters() -> Set<Parameter> {
        switch self {
        case let .gate(gate, _, _):
            gate.parameters()
        case .reset:
            []
        case .measure:
            []
        }
    }

    // MARK: - Parameter Binding

    /// Bind symbolic parameters to concrete numerical values, producing a fully concrete operation.
    ///
    /// For gate operations, delegates to ``QuantumGate/bound(with:)`` to substitute all symbolic
    /// parameters with values from the bindings dictionary, preserving qubits and timestamp.
    /// Reset operations have no parameters and return themselves unchanged.
    ///
    /// **Example:**
    /// ```swift
    /// let theta = Parameter(name: "theta")
    /// let op = CircuitOperation.gate(.rotationY(theta), qubits: [0], timestamp: 1.0)
    /// let bound = op.bound(with: ["theta": .pi / 4])
    /// print(bound)  // gate(Ry(0.785), qubits: [0], t: 1.000)
    ///
    /// let resetOp = CircuitOperation.reset(qubit: 0)
    /// let same = resetOp.bound(with: ["theta": 1.0])
    /// print(same)  // reset(qubit: 0)
    /// ```
    ///
    /// - Parameter bindings: Dictionary mapping parameter names to numerical values
    /// - Returns: Operation with all symbolic parameters substituted
    /// - Complexity: O(1) for gates with at most 3 parameters, O(1) for reset
    @_optimize(speed)
    @_effects(readonly)
    public func bound(with bindings: [String: Double]) -> CircuitOperation {
        switch self {
        case let .gate(gate, qubits, timestamp):
            .gate(gate.bound(with: bindings), qubits: qubits, timestamp: timestamp)
        case .reset:
            self
        case .measure:
            self
        }
    }

    // MARK: - CustomStringConvertible

    /// Human-readable representation of this circuit operation.
    ///
    /// Gate operations display the gate description, qubit list, and optional timestamp.
    /// Reset operations display the qubit index and optional timestamp.
    ///
    /// **Example:**
    /// ```swift
    /// let gateOp = CircuitOperation.gate(.hadamard, qubits: [0])
    /// print(gateOp)  // gate(H, qubits: [0])
    ///
    /// let timedOp = CircuitOperation.gate(.cnot, qubits: [0, 1], timestamp: 2.5)
    /// print(timedOp)  // gate(CNOT, qubits: [0, 1], t: 2.500)
    ///
    /// let resetOp = CircuitOperation.reset(qubit: 3)
    /// print(resetOp)  // reset(qubit: 3)
    /// ```
    ///
    /// - Returns: Formatted string describing the operation
    public var description: String {
        switch self {
        case let .gate(gate, qubits, timestamp):
            if let t = timestamp {
                return "gate(\(gate), qubits: \(qubits), t: \(String(format: "%.3f", t)))"
            }
            return "gate(\(gate), qubits: \(qubits))"
        case let .reset(qubit, timestamp):
            if let t = timestamp {
                return "reset(qubit: \(qubit), t: \(String(format: "%.3f", t)))"
            }
            return "reset(qubit: \(qubit))"
        case let .measure(qubit, classicalBit, timestamp):
            let cbit = classicalBit ?? qubit
            if let t = timestamp {
                return "measure(qubit: \(qubit), cbit: \(cbit), t: \(String(format: "%.3f", t)))"
            }
            return "measure(qubit: \(qubit), cbit: \(cbit))"
        }
    }
}
