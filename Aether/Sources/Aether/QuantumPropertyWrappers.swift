// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Property wrapper providing automatic symbolic parameter tracking for variational quantum circuits.
///
/// Wraps a concrete ``Double`` value while maintaining a companion ``Parameter`` for symbolic
/// circuit construction. The wrapped value serves as the current numerical angle, and the
/// projected value (`$`-prefixed access) yields a ``ParameterValue/parameter(_:)`` suitable
/// for gate constructors that accept symbolic or concrete parameters. This enables a natural
/// declaration syntax where a single property declaration creates both the mutable angle and
/// its symbolic reference, eliminating boilerplate in variational algorithm setup.
///
/// **Example:**
/// ```swift
/// @QuantumParameter(name: "theta") var theta = 0.5
/// var circuit = QuantumCircuit(qubits: 1)
/// circuit.append(.rotationY($theta), to: 0)
/// ```
///
/// - SeeAlso: ``Parameter``
/// - SeeAlso: ``ParameterValue``
@frozen
@propertyWrapper
public struct QuantumParameter: Sendable {
    /// Current concrete angle value used when binding the parameter.
    public var wrappedValue: Double

    /// Symbolic parameter created from the provided name for use in circuit construction.
    public let parameter: Parameter

    /// Symbolic ``ParameterValue`` for use in gate constructors accepting parameterized angles.
    ///
    /// Accessing the projected value via the `$` prefix yields
    /// ``ParameterValue/parameter(_:)`` wrapping the underlying ``Parameter``, enabling
    /// seamless integration with gates such as ``QuantumGate/rotationY(_:)``.
    ///
    /// **Example:**
    /// ```swift
    /// @QuantumParameter(name: "phi") var phi = 1.57
    /// let gate: QuantumGate = .rotationZ($phi)
    /// let value = $phi.evaluate(using: ["phi": phi])
    /// ```
    ///
    /// - Complexity: O(1)
    @inlinable
    public var projectedValue: ParameterValue {
        .parameter(parameter)
    }

    /// Creates a quantum parameter with an initial value and optional symbolic name.
    ///
    /// When `name` is `nil`, a unique deterministic name is generated using a monotonically
    /// increasing counter prefixed with `_qp`, ensuring distinct identifiers across
    /// multiple unnamed parameters within the same program execution.
    ///
    /// **Example:**
    /// ```swift
    /// @QuantumParameter(name: "gamma") var gamma = 0.3
    /// @QuantumParameter var anonymous = 1.0
    /// let bindings = [gamma.parameter.name: gamma.wrappedValue]
    /// ```
    ///
    /// - Parameters:
    ///   - wrappedValue: Initial concrete angle value
    ///   - name: Symbolic parameter name; when `nil`, an auto-generated name is used
    /// - Complexity: O(1)
    public init(wrappedValue: Double, name: String? = nil) {
        self.wrappedValue = wrappedValue
        let resolvedName = name ?? Self.generateName()
        parameter = Parameter(name: resolvedName)
    }

    private static let _counter = LockedCounter()

    @usableFromInline
    static func generateName() -> String {
        let index = _counter.next()
        return "_qp\(index)"
    }
}

/// Property wrapper providing a validated non-negative qubit index reference.
///
/// Wraps an ``Int`` qubit index with a non-negativity precondition enforced at initialization
/// time via ``ValidationUtilities``. The wrapper is a zero-cost abstraction that carries no
/// runtime overhead beyond the initial validation check, and the underlying integer is accessed
/// directly through `wrappedValue` for use in gate operations and circuit construction.
///
/// **Example:**
/// ```swift
/// @Qubit var q0 = 0
/// @Qubit var q1 = 1
/// circuit.append(.controlledNot, to: q0, q1)
/// ```
///
/// - Precondition: Qubit index must be non-negative.
/// - SeeAlso: ``QuantumCircuit``
@frozen
@propertyWrapper
public struct Qubit: Sendable {
    /// Qubit index used to reference a specific qubit in a quantum register.
    public var wrappedValue: Int

    /// Creates a validated qubit index reference.
    ///
    /// **Example:**
    /// ```swift
    /// @Qubit var control = 0
    /// @Qubit var target = 1
    /// circuit.append(.controlledNot, to: control, target)
    /// ```
    ///
    /// - Parameter wrappedValue: Non-negative qubit index
    /// - Complexity: O(1)
    /// - Precondition: Qubit index must be non-negative.
    @inlinable
    public init(wrappedValue: Int) {
        ValidationUtilities.validateNonNegativeInt(wrappedValue, name: "Qubit index")
        self.wrappedValue = wrappedValue
    }
}

// MARK: - Internal Utilities

final class LockedCounter: @unchecked Sendable {
    private var _value: Int = 0
    private let _lock = NSLock()

    func next() -> Int {
        _lock.lock()
        defer { _lock.unlock() }
        let current = _value
        _value += 1
        return current
    }
}
