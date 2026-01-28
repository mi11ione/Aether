// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Lightweight value type representing a single step in a declarative quantum circuit.
///
/// Each step encodes either a unitary ``QuantumGate`` applied to one or more qubits, or a
/// ``NonUnitaryOperation`` applied to a single qubit, together with an optional scheduling
/// timestamp. Steps are collected by ``QuantumCircuitBuilder`` during result-builder
/// evaluation and converted to ``CircuitOperation`` values when the final
/// ``QuantumCircuit`` is constructed.
///
/// ``GateStep`` is intentionally minimal. It stores only the data needed to produce one
/// ``CircuitOperation`` and exposes a single conversion property ``operation`` for that
/// purpose. The struct is `@frozen` and `Sendable`, so the compiler can lay it out inline
/// and pass it across concurrency boundaries without overhead.
///
/// **Example:**
/// ```swift
/// let h = GateStep(.hadamard, on: 0)
/// let cx = GateStep(.cnot, on: 0, 1)
/// let reset = GateStep(.reset, on: 2)
/// ```
///
/// - SeeAlso: ``QuantumCircuitBuilder``
/// - SeeAlso: ``CircuitOperation``
/// - SeeAlso: ``QuantumGate``
/// - SeeAlso: ``NonUnitaryOperation``
@frozen
public struct GateStep: Sendable, Equatable {
    @usableFromInline
    enum Payload: Sendable, Equatable {
        case gate(QuantumGate, qubits: [Int])
        case nonUnitary(NonUnitaryOperation, qubit: Int)
    }

    @usableFromInline
    let payload: Payload

    /// Optional scheduling timestamp for time-ordered circuit representations.
    public let timestamp: Double?

    /// The unitary gate for this step, or `nil` for non-unitary steps.
    ///
    /// - Returns: The ``QuantumGate`` for gate payloads, or ``QuantumGate/identity`` for non-unitary payloads.
    /// - Complexity: O(1)
    @inlinable
    public var gate: QuantumGate {
        switch payload {
        case let .gate(g, _): g
        case .nonUnitary: .identity
        }
    }

    /// Qubit indices targeted by this step.
    ///
    /// - Returns: Array of qubit indices this step operates on.
    /// - Complexity: O(1)
    @inlinable
    public var qubits: [Int] {
        switch payload {
        case let .gate(_, q): q
        case let .nonUnitary(_, q): [q]
        }
    }

    // MARK: - Initializers

    /// Creates a gate step from a unitary gate and variadic qubit indices.
    ///
    /// The number of qubits provided must match the gate's ``QuantumGate/qubitsRequired``
    /// at the call site; validation is deferred to ``QuantumCircuit`` construction where
    /// centralized checks apply.
    ///
    /// - Parameters:
    ///   - gate: Unitary gate to apply
    ///   - qubits: Target qubit indices (variadic)
    ///   - timestamp: Optional scheduling timestamp
    ///
    /// - Complexity: O(1)
    ///
    /// **Example:**
    /// ```swift
    /// let step = GateStep(.hadamard, on: 0)
    /// let cx = GateStep(.cnot, on: 0, 1)
    /// let gate = cx.gate
    /// ```
    @inlinable
    public init(_ gate: QuantumGate, on qubits: Int..., timestamp: Double? = nil) {
        payload = .gate(gate, qubits: qubits)
        self.timestamp = timestamp
    }

    /// Creates a gate step from a unitary gate and an array of qubit indices.
    ///
    /// - Parameters:
    ///   - gate: Unitary gate to apply
    ///   - qubits: Target qubit indices (array)
    ///   - timestamp: Optional scheduling timestamp
    ///
    /// - Complexity: O(1)
    ///
    /// **Example:**
    /// ```swift
    /// let targets = [0, 1, 2]
    /// let step = GateStep(.toffoli, on: targets)
    /// let qubits = step.qubits
    /// ```
    @inlinable
    public init(_ gate: QuantumGate, on qubits: [Int], timestamp: Double? = nil) {
        payload = .gate(gate, qubits: qubits)
        self.timestamp = timestamp
    }

    /// Creates a gate step from a non-unitary operation on a single qubit.
    ///
    /// Non-unitary operations such as ``NonUnitaryOperation/reset`` and
    /// ``NonUnitaryOperation/measure`` act on exactly one qubit. The step
    /// converts to the corresponding ``CircuitOperation`` case during circuit
    /// construction.
    ///
    /// - Parameters:
    ///   - operation: Non-unitary operation to apply
    ///   - qubit: Target qubit index
    ///   - timestamp: Optional scheduling timestamp
    ///
    /// - Complexity: O(1)
    ///
    /// **Example:**
    /// ```swift
    /// let step = GateStep(.reset, on: 0)
    /// let meas = GateStep(.measure, on: 1)
    /// let qubit = meas.qubits[0]
    /// ```
    @inlinable
    public init(_ operation: NonUnitaryOperation, on qubit: Int, timestamp: Double? = nil) {
        payload = .nonUnitary(operation, qubit: qubit)
        self.timestamp = timestamp
    }

    // MARK: - Conversion

    /// Converts this step to the corresponding ``CircuitOperation``.
    ///
    /// Gate payloads produce ``CircuitOperation/gate(_:qubits:timestamp:)`` and
    /// non-unitary payloads produce ``CircuitOperation/reset(qubit:timestamp:)`` or
    /// ``CircuitOperation/measure(qubit:classicalBit:timestamp:)`` depending on the
    /// stored ``NonUnitaryOperation`` case.
    ///
    /// - Complexity: O(1)
    ///
    /// **Example:**
    /// ```swift
    /// let step = GateStep(.hadamard, on: 0)
    /// let op = step.operation
    /// print(op)  // gate(H, qubits: [0])
    /// ```
    ///
    /// - SeeAlso: ``CircuitOperation``
    @inlinable
    public var operation: CircuitOperation {
        switch payload {
        case let .gate(gate, qubits):
            .gate(gate, qubits: qubits, timestamp: timestamp)
        case let .nonUnitary(op, qubit):
            switch op {
            case .reset:
                .reset(qubit: qubit, timestamp: timestamp)
            case .measure:
                .measure(qubit: qubit, timestamp: timestamp)
            }
        }
    }
}

// MARK: - QuantumCircuitBuilder

/// Result builder that enables SwiftUI-style declarative quantum circuit construction.
///
/// ``QuantumCircuitBuilder`` collects ``GateStep`` values from a closure body and
/// concatenates them into a flat `[GateStep]` array. The builder supports all standard
/// result-builder features: sequential statements, `if`/`else` branching, optional
/// binding, and `for`-`in` loops. Because the builder operates on plain arrays of value
/// types, there is zero heap allocation beyond the final array itself and no runtime
/// overhead compared to imperative circuit construction.
///
/// Use the builder indirectly through the ``QuantumCircuit/init(qubits:autoOptimize:_:)``
/// initializer rather than invoking builder methods directly.
///
/// **Example:**
/// ```swift
/// let circuit = QuantumCircuit(qubits: 2) {
///     GateStep(.hadamard, on: 0)
///     GateStep(.cnot, on: 0, 1)
/// }
/// ```
///
/// - SeeAlso: ``GateStep``
/// - SeeAlso: ``QuantumCircuit``
@resultBuilder
public enum QuantumCircuitBuilder {
    /// Converts a single ``GateStep`` expression into the builder's intermediate type.
    ///
    /// - Parameter step: Gate step produced by an expression statement
    /// - Returns: Single-element array wrapping the step
    /// - Complexity: O(1)
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit(qubits: 1) {
    ///     GateStep(.hadamard, on: 0)
    /// }
    /// ```
    @inlinable
    @inline(__always)
    @_effects(readonly)
    public static func buildExpression(_ step: GateStep) -> [GateStep] {
        [step]
    }

    /// Concatenates arrays from sequential statements into a single flat array.
    ///
    /// - Parameter components: Variadic arrays produced by sequential expressions
    /// - Returns: Flat concatenation of all component arrays
    /// - Complexity: O(n) where n is the total number of steps across all components
    ///
    /// **Example:**
    /// ```swift
    /// let step1 = [GateStep(.hadamard, on: 0)]
    /// let step2 = [GateStep(.cnot, on: 0, 1)]
    /// let steps = QuantumCircuitBuilder.buildBlock(step1, step2)
    /// ```
    @inlinable
    @_optimize(speed)
    public static func buildBlock(_ components: [GateStep]...) -> [GateStep] {
        var totalCount = 0
        for component in components {
            totalCount += component.count
        }
        var result: [GateStep] = []
        result.reserveCapacity(totalCount)
        for component in components {
            result.append(contentsOf: component)
        }
        return result
    }

    /// Handles optional blocks produced by `if` statements without `else`.
    ///
    /// - Parameter steps: Array of steps if the condition was true, `nil` otherwise
    /// - Returns: The wrapped array or an empty array
    /// - Complexity: O(1)
    ///
    /// **Example:**
    /// ```swift
    /// let applyH = true
    /// let circuit = QuantumCircuit(qubits: 1) {
    ///     if applyH { GateStep(.hadamard, on: 0) }
    /// }
    /// ```
    @inlinable
    @inline(__always)
    @_effects(readonly)
    public static func buildOptional(_ steps: [GateStep]?) -> [GateStep] {
        steps ?? []
    }

    /// Handles the `true` branch of an `if`-`else` statement.
    ///
    /// - Parameter steps: Steps from the first (true) branch
    /// - Returns: The steps unchanged
    /// - Complexity: O(1)
    ///
    /// **Example:**
    /// ```swift
    /// let useH = true
    /// let circuit = QuantumCircuit(qubits: 1) {
    ///     if useH { GateStep(.hadamard, on: 0) } else { GateStep(.pauliX, on: 0) }
    /// }
    /// ```
    @inlinable
    @inline(__always)
    @_effects(readonly)
    public static func buildEither(first steps: [GateStep]) -> [GateStep] {
        steps
    }

    /// Handles the `false` branch of an `if`-`else` statement.
    ///
    /// - Parameter steps: Steps from the second (false) branch
    /// - Returns: The steps unchanged
    /// - Complexity: O(1)
    ///
    /// **Example:**
    /// ```swift
    /// let useH = false
    /// let circuit = QuantumCircuit(qubits: 1) {
    ///     if useH { GateStep(.hadamard, on: 0) } else { GateStep(.pauliX, on: 0) }
    /// }
    /// ```
    @inlinable
    @inline(__always)
    @_effects(readonly)
    public static func buildEither(second steps: [GateStep]) -> [GateStep] {
        steps
    }

    /// Handles `for`-`in` loops by concatenating iteration results.
    ///
    /// - Parameter groups: Array of step arrays, one per loop iteration
    /// - Returns: Flat concatenation of all iteration arrays
    /// - Complexity: O(n) where n is the total number of steps across all iterations
    ///
    /// **Example:**
    /// ```swift
    /// let qubits = [0, 1, 2]
    /// let circuit = QuantumCircuit(qubits: 3) {
    ///     for q in qubits { GateStep(.hadamard, on: q) }
    /// }
    /// ```
    @inlinable
    @_optimize(speed)
    public static func buildArray(_ groups: [[GateStep]]) -> [GateStep] {
        var totalCount = 0
        for group in groups {
            totalCount += group.count
        }
        var result: [GateStep] = []
        result.reserveCapacity(totalCount)
        for group in groups {
            result.append(contentsOf: group)
        }
        return result
    }
}

// MARK: - QuantumCircuit Builder Initializer

public extension QuantumCircuit {
    /// Creates a quantum circuit from a declarative builder closure.
    ///
    /// Collects ``GateStep`` values from the ``QuantumCircuitBuilder`` closure, converts
    /// each step to a ``CircuitOperation`` via ``GateStep/operation``, and delegates to
    /// ``init(qubits:operations:autoOptimize:)`` for final construction. This provides
    /// SwiftUI-style declarative syntax with zero runtime overhead compared to imperative
    /// construction through repeated `append` calls.
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (1-30)
    ///   - autoOptimize: When true, automatically cancel adjacent identity pairs on append (default: false)
    ///   - build: Builder closure producing an array of gate steps
    /// - Precondition: qubits must be positive
    /// - Complexity: O(n) where n is the number of steps in the builder closure
    ///
    /// **Example:**
    /// ```swift
    /// let bell = QuantumCircuit(qubits: 2) {
    ///     GateStep(.hadamard, on: 0)
    ///     GateStep(.cnot, on: 0, 1)
    /// }
    /// ```
    ///
    /// - SeeAlso: ``QuantumCircuitBuilder``
    /// - SeeAlso: ``GateStep``
    init(
        qubits: Int,
        autoOptimize: Bool = false,
        @QuantumCircuitBuilder _ build: () -> [GateStep],
    ) {
        let steps = build()
        var operations: [CircuitOperation] = []
        operations.reserveCapacity(steps.count)
        for step in steps {
            operations.append(step.operation)
        }
        self.init(qubits: qubits, operations: operations, autoOptimize: autoOptimize)
    }
}
