// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Symbolic parameter for variational quantum circuits
///
/// Represents a named symbolic parameter used in parameterized quantum gates.
/// Parameters are the free variables in variational algorithms (VQE, QAOA) that
/// are optimized by classical optimizers to minimize objective functions.
///
/// **Usage in variational algorithms**:
/// - VQE: Parameters represent rotation angles in hardware-efficient ansätze
/// - QAOA: Parameters γ and β control problem and mixer Hamiltonian evolution
/// - Quantum machine learning: Parameters are trainable weights in quantum circuits
///
/// **Parameter naming conventions**:
/// - Use descriptive names: "theta", "phi", "gamma", "beta"
/// - Greek letters are standard in quantum algorithms
/// - Numbered parameters: "theta_0", "theta_1" for repeated gates
///
/// Example:
/// ```swift
/// // Create symbolic parameters
/// let theta = Parameter(name: "theta")
/// let phi = Parameter(name: "phi")
///
/// // Use in parameterized gates
/// let gate = ParameterizedGate.rotationY(theta: .parameter(theta))
///
/// // Parameters are identified by name
/// let theta2 = Parameter(name: "theta")
/// print(theta == theta2)  // true (same name)
/// ```
@frozen
public struct Parameter: Equatable, Hashable, Sendable, CustomStringConvertible {
    public let name: String

    /// Create symbolic parameter with name
    /// - Parameter name: Parameter name (must be non-empty)
    public init(name: String) {
        ValidationUtilities.validateNonEmptyString(name, name: "Parameter name")
        self.name = name
    }

    /// String representation of parameter
    @inlinable
    public var description: String { name }
}

/// Parameter expression: symbolic parameter or concrete value
///
/// Represents either a symbolic parameter (to be bound later) or a concrete
/// numerical value. Enables mixing symbolic and concrete parameters in the
/// same parameterized circuit.
///
/// **Design rationale**:
/// - Evaluation requires binding dictionary for symbolic parameters
/// - Type-safe distinction between symbolic and concrete
///
/// Example:
/// ```swift
/// // Symbolic parameter
/// let symbolic = ParameterExpression.parameter(Parameter(name: "theta"))
/// print(symbolic.isSymbolic)  // true
///
/// // Concrete value
/// let concrete = ParameterExpression.value(Double.pi / 4)
/// print(concrete.isConcrete)  // true
///
/// // Evaluate with bindings
/// let bindings = ["theta": 1.57, "phi": 0.785]
/// let result = symbolic.evaluate(with: bindings)  // 1.57
///
/// // Concrete values evaluate to themselves
/// let concreteResult = concrete.evaluate(with: [:])  // π/4
/// ```
@frozen
public enum ParameterExpression: Equatable, Hashable, Sendable, CustomStringConvertible {
    /// Symbolic parameter reference
    case parameter(Parameter)

    /// Concrete numerical value
    case value(Double)

    /// Whether expression is symbolic (contains unbound parameters)
    @inlinable
    @_effects(readonly)
    public func isSymbolic() -> Bool {
        if case .parameter = self { return true }
        return false
    }

    /// Whether expression is concrete (all parameters bound)
    @inlinable
    @_effects(readonly)
    public func isConcrete() -> Bool { !isSymbolic() }

    /// Evaluate expression with parameter bindings
    ///
    /// Substitutes parameter values from bindings dictionary. Concrete values
    /// evaluate to themselves. Symbolic parameters require binding.
    ///
    /// - Parameter bindings: Dictionary mapping parameter names to values
    /// - Returns: Evaluated numerical value
    /// - Precondition: Symbolic parameters must have binding in dictionary
    ///
    /// Example:
    /// ```swift
    /// let expr = ParameterExpression.parameter(Parameter(name: "theta"))
    /// let bindings = ["theta": Double.pi / 2, "phi": 0.5]
    ///
    /// let value = expr.evaluate(with: bindings)  // π/2
    /// ```
    @_optimize(speed)
    @inlinable
    public func evaluate(with bindings: [String: Double]) -> Double {
        switch self {
        case let .value(v): return v
        case let .parameter(p):
            ValidationUtilities.validateParameterBinding(p.name, in: bindings)
            return bindings[p.name]!
        }
    }

    /// Extract symbolic parameter if present
    /// - Returns: Parameter if expression is symbolic, nil if concrete
    @inlinable
    @_effects(readonly)
    public func extractParameter() -> Parameter? {
        if case let .parameter(p) = self { return p }
        return nil
    }

    /// String representation of expression
    @inlinable
    public var description: String {
        switch self {
        case let .parameter(p): p.name
        case let .value(v): String(format: "%.3f", v)
        }
    }
}
