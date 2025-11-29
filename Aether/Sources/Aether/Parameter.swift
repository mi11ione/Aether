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
/// **Defer-and-bind pattern**: Build parameterized circuits once with symbolic parameters,
/// then bind different numerical values per optimization iteration without reconstructing
/// the circuit. This pattern separates circuit topology from parameter values, enabling
/// efficient optimization loops.
///
/// **Usage workflow**:
/// 1. Create symbolic parameters with descriptive names
/// 2. Build parameterized circuit using ``QuantumCircuit``
/// 3. Classical optimizer proposes parameter values
/// 4. Bind parameters to concrete values via `binding(_:)`
/// 5. Execute concrete circuit and measure expectation value
/// 6. Repeat steps 3-5 until convergence
///
/// **Naming conventions**:
/// - Greek letters: "theta", "phi", "gamma", "beta" (standard in quantum algorithms)
/// - Indexed parameters: "theta_0", "theta_1" for repeated gates
/// - Descriptive names: "rotation_angle", "phase_shift" for clarity
///
/// **Example**:
/// ```swift
/// let theta = Parameter(name: "theta")
/// let phi = Parameter(name: "phi")
///
/// var circuit = QuantumCircuit(numQubits: 2)
/// circuit.append(.rotationY(theta), to: 0)
/// circuit.append(.rotationZ(phi), to: 1)
///
/// let bindings = ["theta": Double.pi / 4, "phi": Double.pi / 8]
/// let bound = circuit.binding(bindings)
/// let state = bound.execute()
/// ```
///
/// - SeeAlso: ``ParameterValue``, ``QuantumCircuit``, ``QuantumGate``
public struct Parameter: Equatable, Hashable, Sendable, CustomStringConvertible {
    /// Parameter name used for identification and binding
    public let name: String

    /// Create symbolic parameter with name
    ///
    /// **Example**:
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
/// **Type-safe evaluation**: Symbolic values require binding dictionary at
/// evaluation time, while concrete values evaluate to themselves. This distinction
/// prevents accidentally evaluating unbound parameters.
///
/// **Use cases**:
/// - **Pure symbolic**: All parameters symbolic, bind before execution
/// - **Mixed circuits**: Some gates with symbolic parameters, others with concrete angles
/// - **Partial binding**: Bind subset of parameters, leaving others symbolic for nested optimization
///
/// **Example**:
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
/// - SeeAlso: ``Parameter``, ``QuantumGate``
public enum ParameterValue: Equatable, Hashable, Sendable, CustomStringConvertible {
    /// Symbolic parameter reference requiring binding at evaluation
    case parameter(Parameter)

    /// Concrete numerical value (fixed, no binding required)
    case value(Double)

    /// Whether expression contains symbolic parameter
    ///
    /// **Example**:
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
    /// **Example**:
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
    /// Concrete values evaluate to their stored number regardless of bindings.
    ///
    /// **Algorithm**:
    /// - Symbolic: Look up parameter name in bindings, return value
    /// - Concrete: Return stored value immediately
    ///
    /// **Example**:
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
