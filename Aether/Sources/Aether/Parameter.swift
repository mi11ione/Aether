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

    /// Optional constraint for the parameter
    public let constraint: ParameterConstraint?

    /// Create symbolic parameter with name and optional constraint
    ///
    /// **Example:**
    /// ```swift
    /// let theta = Parameter(name: "theta")
    /// let gamma_0 = Parameter(name: "gamma_0")
    /// let bounded = Parameter(name: "phase", constraint: .bounded(min: 0.0, max: 2 * .pi))
    /// ```
    ///
    /// - Parameters:
    ///   - name: Parameter name (must be non-empty)
    ///   - constraint: Optional constraint for the parameter
    /// - Complexity: O(1)
    /// - Precondition: Name must be non-empty
    public init(name: String, constraint: ParameterConstraint? = nil) {
        ValidationUtilities.validateNonEmptyString(name, name: "Parameter name")
        self.name = name
        self.constraint = constraint
    }

    @inlinable
    public var description: String { name }
}

/// Parameter value: symbolic parameter, concrete numerical value, or expression
///
/// Represents either a symbolic parameter (to be bound later), a concrete
/// numerical value, or a complex expression over parameters. This enables mixing
/// symbolic and concrete parameters in the same quantum circuit, useful for hybrid
/// optimization where some parameters are fixed while others vary.
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

    /// Negated symbolic parameter (-θ) for gate inverse computation
    case negatedParameter(Parameter)

    /// Complex expression over parameters for algebraic relationships
    ///
    /// **Example:**
    /// ```swift
    /// let theta = Parameter(name: "theta")
    /// let phi = Parameter(name: "phi")
    /// let sum = ParameterExpression(theta) + ParameterExpression(phi)
    /// let expr = ParameterValue.expression(sum)
    ///
    /// let bindings = ["theta": 1.0, "phi": 0.5]
    /// let result = expr.evaluate(using: bindings)  // 1.5
    /// ```
    case expression(ParameterExpression)

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
    /// - Complexity: O(1) for parameter/value/negatedParameter, O(n) for expression
    @inlinable
    public var isSymbolic: Bool {
        switch self {
        case .parameter, .negatedParameter: true
        case .value: false
        case let .expression(expr): expr.isSymbolic
        }
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
    /// - Returns: Parameter if symbolic, `nil` if concrete or expression
    /// - Complexity: O(1)
    @inlinable
    public var parameter: Parameter? {
        switch self {
        case let .parameter(p), let .negatedParameter(p): p
        case .value, .expression: nil
        }
    }

    /// Evaluate parameter value with bindings
    ///
    /// Substitutes symbolic parameters with concrete values from bindings dictionary.
    /// Concrete values evaluate to their stored number regardless of bindings. For
    /// symbolic values, looks up the parameter name in bindings and returns the bound
    /// value; for concrete values, returns the stored number immediately. Expression
    /// values delegate to ``ParameterExpression/evaluate(using:)``.
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
    /// - Complexity: O(1) dictionary lookup for simple cases, O(n) for expressions
    /// - Precondition: Symbolic parameters must have binding in dictionary
    @_optimize(speed)
    @inlinable
    public func evaluate(using bindings: [String: Double]) -> Double {
        switch self {
        case let .value(v):
            return v
        case let .parameter(p):
            ValidationUtilities.validateParameterBinding(p.name, in: bindings)
            return bindings[p.name]!
        case let .negatedParameter(p):
            ValidationUtilities.validateParameterBinding(p.name, in: bindings)
            return -bindings[p.name]!
        case let .expression(expr):
            return expr.evaluate(using: bindings)
        }
    }

    @inlinable
    public var description: String {
        switch self {
        case let .parameter(p): p.name
        case let .value(v): String(format: "%.3f", v)
        case let .negatedParameter(p): "-\(p.name)"
        case let .expression(expr): "expr(\(expr.node))"
        }
    }

    /// Negated parameter value for gate inverse computation
    ///
    /// Returns a new parameter value with negated sign. For concrete values, directly
    /// negates the number. For symbolic parameters, creates a negated expression that
    /// evaluates to the negative of the bound value. Negating twice returns the original.
    /// For expressions, wraps in negation.
    ///
    /// **Example:**
    /// ```swift
    /// let concrete = ParameterValue.value(1.57)
    /// print(concrete.negated)  // .value(-1.57)
    ///
    /// let theta = Parameter(name: "theta")
    /// let symbolic = ParameterValue.parameter(theta)
    /// print(symbolic.negated)  // .negatedParameter(theta)
    ///
    /// let bindings = ["theta": 1.0]
    /// print(symbolic.negated.evaluate(using: bindings))  // -1.0
    /// ```
    ///
    /// - Complexity: O(1)
    @_optimize(speed)
    @inlinable
    public var negated: ParameterValue {
        switch self {
        case let .value(v):
            .value(-v)
        case let .parameter(p):
            .negatedParameter(p)
        case let .negatedParameter(p):
            .parameter(p)
        case let .expression(expr):
            .expression(-expr)
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
