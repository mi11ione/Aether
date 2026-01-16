// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Symbolic parameter for variational quantum circuits
///
/// Represents a named symbolic parameter used in parameterized quantum gates.
/// Parameters are the free variables in variational algorithms (VQE, QAOA, quantum
/// machine learning) that are optimized by classical optimizers to minimize objective
/// functions.
///
/// The defer-and-bind pattern builds parameterized circuits once with symbolic parameters,
/// then binds different numerical values per optimization iteration without reconstructing
/// the circuit. This separates circuit topology from parameter values, enabling efficient
/// optimization loops. The typical workflow creates symbolic parameters, builds the circuit,
/// then iteratively binds values via ``QuantumCircuit/binding(_:)`` and executes until
/// convergence. Parameter names conventionally use Greek letters (theta, phi, gamma, beta),
/// indexed variants (theta_0, theta_1) for repeated gates, or descriptive names
/// (rotation_angle, phase_shift) for clarity.
///
/// **Example:**
/// ```swift
/// let theta = Parameter(name: "theta")
/// let phi = Parameter(name: "phi")
///
/// var circuit = QuantumCircuit(qubits: 2)
/// circuit.append(.rotationY(theta), to: 0)
/// circuit.append(.rotationZ(phi), to: 1)
///
/// let bindings = ["theta": Double.pi / 4, "phi": Double.pi / 8]
/// let bound = circuit.binding(bindings)
/// let state = bound.execute()
/// ```
///
/// - SeeAlso: ``ParameterValue``
/// - SeeAlso: ``QuantumCircuit``
/// - SeeAlso: ``QuantumGate``
@frozen
public struct Parameter: Equatable, Hashable, Sendable, CustomStringConvertible {
    /// Parameter name used for identification and binding
    public let name: String

    /// Create symbolic parameter with name
    ///
    /// **Example:**
    /// ```swift
    /// let theta = Parameter(name: "theta")
    /// let gamma_0 = Parameter(name: "gamma_0")
    /// ```
    ///
    /// - Parameter name: Parameter name (must be non-empty)
    /// - Complexity: O(1)
    /// - Precondition: Name must be non-empty
    public init(name: String) {
        ValidationUtilities.validateNonEmptyString(name, name: "Parameter name")
        self.name = name
    }

    @inlinable
    public var description: String { name }
}

/// Parameter value: symbolic parameter or concrete numerical value
///
/// Represents either a symbolic parameter (to be bound later) or a concrete
/// numerical value. This enables mixing symbolic and concrete parameters in the
/// same quantum circuit, useful for hybrid optimization where some parameters
/// are fixed while others vary.
///
/// Symbolic values require a binding dictionary at evaluation time, while concrete values
/// evaluate to themselves. This type-safe distinction prevents accidentally evaluating
/// unbound parameters. Common patterns include pure symbolic circuits where all parameters
/// are bound before execution, mixed circuits combining symbolic parameters with fixed
/// concrete angles, and partial binding where a subset of parameters remains symbolic
/// for nested optimization scenarios.
///
/// **Example:**
/// ```swift
/// let theta = Parameter(name: "theta")
/// let symbolic = ParameterValue.parameter(theta)
/// let concrete = ParameterValue.value(Double.pi / 4)
///
/// print(symbolic.isSymbolic)  // true
/// print(concrete.isSymbolic)  // false
///
/// let bindings = ["theta": 1.57]
/// let result = symbolic.evaluate(using: bindings)  // 1.57
/// let fixed = concrete.evaluate(using: [:])        // π/4
/// ```
///
/// - SeeAlso: ``Parameter``
/// - SeeAlso: ``QuantumGate``
@frozen
public enum ParameterValue: Equatable, Hashable, Sendable, CustomStringConvertible {
    /// Symbolic parameter reference requiring binding at evaluation
    case parameter(Parameter)

    /// Concrete numerical value (fixed, no binding required)
    case value(Double)

    /// Whether expression contains symbolic parameter
    ///
    /// **Example:**
    /// ```swift
    /// let symbolic = ParameterValue.parameter(Parameter(name: "theta"))
    /// let concrete = ParameterValue.value(1.57)
    ///
    /// print(symbolic.isSymbolic)  // true
    /// print(concrete.isSymbolic)  // false
    /// ```
    ///
    /// - Complexity: O(1)
    @inlinable
    public var isSymbolic: Bool {
        if case .parameter = self { return true }
        return false
    }

    /// Extract symbolic parameter if present
    ///
    /// Returns the underlying ``Parameter`` if value is symbolic, otherwise `nil`.
    /// Useful for parameter extraction and circuit introspection.
    ///
    /// **Example:**
    /// ```swift
    /// let theta = Parameter(name: "theta")
    /// let symbolic = ParameterValue.parameter(theta)
    /// let concrete = ParameterValue.value(1.57)
    ///
    /// print(symbolic.parameter?.name)  // "theta"
    /// print(concrete.parameter)        // nil
    /// ```
    ///
    /// - Returns: Parameter if symbolic, `nil` if concrete
    /// - Complexity: O(1)
    @inlinable
    public var parameter: Parameter? {
        if case let .parameter(p) = self { return p }
        return nil
    }

    /// Evaluate parameter value with bindings
    ///
    /// Substitutes symbolic parameters with concrete values from bindings dictionary.
    /// Concrete values evaluate to their stored number regardless of bindings. For
    /// symbolic values, looks up the parameter name in bindings and returns the bound
    /// value; for concrete values, returns the stored number immediately.
    ///
    /// **Example:**
    /// ```swift
    /// let theta = Parameter(name: "theta")
    /// let expr = ParameterValue.parameter(theta)
    /// let bindings = ["theta": Double.pi / 2, "phi": 0.5]
    ///
    /// let value = expr.evaluate(using: bindings)  // π/2
    ///
    /// let concrete = ParameterValue.value(1.0)
    /// let fixed = concrete.evaluate(using: [:])   // 1.0 (bindings ignored)
    /// ```
    ///
    /// - Parameter bindings: Dictionary mapping parameter names to numerical values
    /// - Returns: Evaluated numerical value
    /// - Complexity: O(1) dictionary lookup
    /// - Precondition: Symbolic parameters must have binding in dictionary
    @_optimize(speed)
    @inlinable
    public func evaluate(using bindings: [String: Double]) -> Double {
        switch self {
        case let .value(v): return v
        case let .parameter(p):
            ValidationUtilities.validateParameterBinding(p.name, in: bindings)
            // Safety: validateParameterBinding guarantees key exists
            return bindings[p.name]!
        }
    }

    @inlinable
    public var description: String {
        switch self {
        case let .parameter(p): p.name
        case let .value(v): String(format: "%.3f", v)
        }
    }
}

// MARK: - Protocol Conformances

extension ParameterValue: ExpressibleByFloatLiteral {
    /// Create concrete parameter value from float literal
    ///
    /// Enables implicit conversion from numeric literals to concrete parameter values.
    /// Allows writing `.rotationX(1.57)` instead of `.rotationX(.value(1.57))`.
    ///
    /// **Example:**
    /// ```swift
    /// let gate: QuantumGate = .rotationY(1.57)  // Implicit .value(1.57)
    /// ```
    ///
    /// - Parameter value: Float literal value
    @inlinable
    public init(floatLiteral value: Double) {
        self = .value(value)
    }
}

public extension ParameterValue {
    /// Create symbolic parameter value from parameter
    ///
    /// Enables implicit conversion from ``Parameter`` to symbolic parameter value.
    /// Allows writing `.rotationX(theta)` instead of `.rotationX(.parameter(theta))`.
    ///
    /// **Example:**
    /// ```swift
    /// let theta = Parameter(name: "theta")
    /// let gate: QuantumGate = .rotationY(theta)  // Implicit .parameter(theta)
    /// ```
    ///
    /// - Parameter parameter: Symbolic parameter to wrap
    @inlinable
    init(_ parameter: Parameter) {
        self = .parameter(parameter)
    }
}
