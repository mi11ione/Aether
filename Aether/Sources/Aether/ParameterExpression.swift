// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Expression node representing symbolic parameter algebra operations
///
/// Provides a directed acyclic graph (DAG) representation for symbolic parameter expressions.
/// Supports arithmetic operations, negation, and transcendental functions for building complex
/// parameter expressions in variational quantum circuits. Expression trees are evaluated lazily
/// when parameter bindings are provided.
///
/// **Example:**
/// ```swift
/// let theta = Parameter(name: "theta")
/// let expr = ExpressionNode.add(
///     .parameter(theta),
///     .constant(Double.pi / 4)
/// )
/// ```
@frozen
public indirect enum ExpressionNode: Equatable, Hashable, Sendable {
    /// Literal numeric value
    case constant(Double)
    /// Symbolic ``Parameter`` reference
    case parameter(Parameter)
    /// Sum of two sub-expressions
    case add(ExpressionNode, ExpressionNode)
    /// Difference of two sub-expressions
    case subtract(ExpressionNode, ExpressionNode)
    /// Product of two sub-expressions
    case multiply(ExpressionNode, ExpressionNode)
    /// Quotient of two sub-expressions
    case divide(ExpressionNode, ExpressionNode)
    /// Arithmetic negation of a sub-expression
    case negate(ExpressionNode)
    /// Sine function applied to a sub-expression
    case sin(ExpressionNode)
    /// Cosine function applied to a sub-expression
    case cos(ExpressionNode)
    /// Tangent function applied to a sub-expression
    case tan(ExpressionNode)
    /// Exponential function applied to a sub-expression
    case exp(ExpressionNode)
    /// Natural logarithm applied to a sub-expression
    case log(ExpressionNode)
    /// Arctangent function applied to a sub-expression
    case arctan(ExpressionNode)
}

/// Symbolic parameter expression for variational quantum circuits
///
/// Represents algebraic expressions over symbolic parameters, enabling construction of complex
/// parameter relationships like `theta + phi`, `2 * theta`, `sin(theta)`, etc. Expressions are
/// represented as a DAG of ``ExpressionNode`` values and evaluated lazily when bindings are
/// provided. Supports automatic differentiation via the ``gradient(withRespectTo:)`` method
/// for gradient-based optimization.
///
/// **Example:**
/// ```swift
/// let theta = Parameter(name: "theta")
/// let phi = Parameter(name: "phi")
///
/// let thetaExpr = ParameterExpression(theta)
/// let phiExpr = ParameterExpression(phi)
/// let sum = thetaExpr + phiExpr
/// let scaled = ParameterExpression(2.0) * thetaExpr
///
/// let bindings = ["theta": Double.pi / 4, "phi": Double.pi / 8]
/// let value = sum.evaluate(using: bindings)
///
/// let grad = sum.gradient(withRespectTo: theta)
/// ```
@frozen
public struct ParameterExpression: Equatable, Hashable, Sendable {
    /// Root node of the expression DAG
    public let node: ExpressionNode

    /// Creates expression from an expression node
    ///
    /// **Example:**
    /// ```swift
    /// let theta = Parameter(name: "theta")
    /// let node = ExpressionNode.parameter(theta)
    /// let expr = ParameterExpression(node: node)
    /// let bindings = ["theta": Double.pi / 4]
    /// let value = expr.evaluate(using: bindings)
    /// ```
    ///
    /// - Parameter node: Expression node representing the expression tree
    @inlinable
    public init(node: ExpressionNode) {
        self.node = node
    }

    /// Creates constant expression from double value
    ///
    /// **Example:**
    /// ```swift
    /// let pi = ParameterExpression(Double.pi)
    /// let half = ParameterExpression(0.5)
    /// let scaled = pi * half
    /// let result = scaled.evaluate(using: [:])
    /// ```
    ///
    /// - Parameter value: Constant numerical value
    @inlinable
    public init(_ value: Double) {
        node = .constant(value)
    }

    /// Creates symbolic expression from parameter
    ///
    /// **Example:**
    /// ```swift
    /// let theta = Parameter(name: "theta")
    /// let expr = ParameterExpression(theta)
    /// let bindings = ["theta": Double.pi / 2]
    /// let value = expr.evaluate(using: bindings)
    /// ```
    ///
    /// - Parameter parameter: Symbolic parameter
    @inlinable
    public init(_ parameter: Parameter) {
        node = .parameter(parameter)
    }

    /// Whether expression contains symbolic parameters
    ///
    /// Returns `true` if expression depends on any symbolic parameters, `false` if purely constant.
    /// Constant expressions can be evaluated without bindings.
    ///
    /// **Example:**
    /// ```swift
    /// let theta = Parameter(name: "theta")
    /// let symbolic = ParameterExpression(theta)
    /// let constant = ParameterExpression(3.14)
    ///
    /// print(symbolic.isSymbolic)  // true
    /// print(constant.isSymbolic)  // false
    /// ```
    ///
    /// - Complexity: O(n) where n is the number of nodes in the expression tree
    public var isSymbolic: Bool {
        !parameters.isEmpty
    }

    /// Set of all symbolic parameters in expression
    ///
    /// Collects all unique parameters referenced in the expression tree. Useful for determining
    /// which bindings are required for evaluation or which parameters to compute gradients for.
    ///
    /// **Example:**
    /// ```swift
    /// let theta = Parameter(name: "theta")
    /// let phi = Parameter(name: "phi")
    /// let expr = ParameterExpression(theta) + ParameterExpression(phi)
    ///
    /// print(expr.parameters)  // Set containing theta and phi
    /// ```
    ///
    /// - Complexity: O(n) where n is the number of nodes in the expression tree
    public var parameters: Set<Parameter> {
        collectParameters(from: node)
    }

    /// Evaluates expression using parameter bindings
    ///
    /// Recursively evaluates the expression tree by substituting symbolic parameters with
    /// their bound values and computing arithmetic and transcendental operations.
    ///
    /// **Example:**
    /// ```swift
    /// let theta = Parameter(name: "theta")
    /// let expr = ParameterExpression(2.0) * ParameterExpression(theta)
    /// let bindings = ["theta": Double.pi / 4]
    ///
    /// let result = expr.evaluate(using: bindings)  // π/2
    /// ```
    ///
    /// - Parameter bindings: Dictionary mapping parameter names to numerical values
    /// - Returns: Evaluated numerical result
    /// - Complexity: O(n) where n is the number of unique nodes in the expression tree
    /// - Precondition: All symbolic parameters must have bindings in dictionary
    @_effects(readonly)
    @inlinable
    public func evaluate(using bindings: [String: Double]) -> Double {
        ValidationUtilities.validateExpressionBinding(bindings, for: parameters)
        var cache: [ExpressionNode: Double] = [:]
        return evaluateNodeMemoized(node, using: bindings, cache: &cache)
    }

    /// Computes symbolic gradient with respect to parameter
    ///
    /// Returns a new ``ParameterExpression`` representing the partial derivative of this
    /// expression with respect to the specified parameter. Uses symbolic differentiation
    /// rules for all supported operations including transcendentals.
    ///
    /// **Example:**
    /// ```swift
    /// let theta = Parameter(name: "theta")
    /// let expr = ParameterExpression.sin(ParameterExpression(theta))
    /// let grad = expr.gradient(withRespectTo: theta)
    ///
    /// let bindings = ["theta": 0.0]
    /// print(grad.evaluate(using: bindings))  // 1.0 (cos(0) = 1)
    /// ```
    ///
    /// - Parameter parameter: Parameter to differentiate with respect to
    /// - Returns: Symbolic expression representing the gradient
    /// - Complexity: O(n) where n is the number of nodes in the expression tree
    @_effects(readonly)
    public func gradient(withRespectTo parameter: Parameter) -> ParameterExpression {
        ParameterExpression(node: differentiateNode(node, withRespectTo: parameter))
    }

    @_effects(readonly)
    @usableFromInline
    func evaluateNodeMemoized(_ node: ExpressionNode, using bindings: [String: Double], cache: inout [ExpressionNode: Double]) -> Double {
        if let cached = cache[node] {
            return cached
        }
        let result: Double = switch node {
        case let .constant(value):
            value
        case let .parameter(param):
            bindings[param.name]!
        case let .add(lhs, rhs):
            evaluateNodeMemoized(lhs, using: bindings, cache: &cache) + evaluateNodeMemoized(rhs, using: bindings, cache: &cache)
        case let .subtract(lhs, rhs):
            evaluateNodeMemoized(lhs, using: bindings, cache: &cache) - evaluateNodeMemoized(rhs, using: bindings, cache: &cache)
        case let .multiply(lhs, rhs):
            evaluateNodeMemoized(lhs, using: bindings, cache: &cache) * evaluateNodeMemoized(rhs, using: bindings, cache: &cache)
        case let .divide(lhs, rhs):
            evaluateNodeMemoized(lhs, using: bindings, cache: &cache) / evaluateNodeMemoized(rhs, using: bindings, cache: &cache)
        case let .negate(inner):
            -evaluateNodeMemoized(inner, using: bindings, cache: &cache)
        case let .sin(inner):
            Foundation.sin(evaluateNodeMemoized(inner, using: bindings, cache: &cache))
        case let .cos(inner):
            Foundation.cos(evaluateNodeMemoized(inner, using: bindings, cache: &cache))
        case let .tan(inner):
            Foundation.tan(evaluateNodeMemoized(inner, using: bindings, cache: &cache))
        case let .exp(inner):
            Foundation.exp(evaluateNodeMemoized(inner, using: bindings, cache: &cache))
        case let .log(inner):
            Foundation.log(evaluateNodeMemoized(inner, using: bindings, cache: &cache))
        case let .arctan(inner):
            Foundation.atan(evaluateNodeMemoized(inner, using: bindings, cache: &cache))
        }
        cache[node] = result
        return result
    }

    @_effects(readonly)
    private func collectParameters(from node: ExpressionNode) -> Set<Parameter> {
        switch node {
        case .constant:
            []
        case let .parameter(param):
            [param]
        case let .add(lhs, rhs),
             let .subtract(lhs, rhs),
             let .multiply(lhs, rhs),
             let .divide(lhs, rhs):
            collectParameters(from: lhs).union(collectParameters(from: rhs))
        case let .negate(inner),
             let .sin(inner),
             let .cos(inner),
             let .tan(inner),
             let .exp(inner),
             let .log(inner),
             let .arctan(inner):
            collectParameters(from: inner)
        }
    }

    @_effects(readonly)
    private func differentiateNode(_ node: ExpressionNode, withRespectTo param: Parameter) -> ExpressionNode {
        switch node {
        case .constant:
            .constant(0.0)
        case let .parameter(p):
            p == param ? .constant(1.0) : .constant(0.0)
        case let .add(lhs, rhs):
            .add(
                differentiateNode(lhs, withRespectTo: param),
                differentiateNode(rhs, withRespectTo: param),
            )
        case let .subtract(lhs, rhs):
            .subtract(
                differentiateNode(lhs, withRespectTo: param),
                differentiateNode(rhs, withRespectTo: param),
            )
        case let .multiply(lhs, rhs):
            .add(
                .multiply(differentiateNode(lhs, withRespectTo: param), rhs),
                .multiply(lhs, differentiateNode(rhs, withRespectTo: param)),
            )
        case let .divide(lhs, rhs):
            .divide(
                .subtract(
                    .multiply(differentiateNode(lhs, withRespectTo: param), rhs),
                    .multiply(lhs, differentiateNode(rhs, withRespectTo: param)),
                ),
                .multiply(rhs, rhs),
            )
        case let .negate(inner):
            .negate(differentiateNode(inner, withRespectTo: param))
        case let .sin(inner):
            .multiply(
                .cos(inner),
                differentiateNode(inner, withRespectTo: param),
            )
        case let .cos(inner):
            .multiply(
                .negate(.sin(inner)),
                differentiateNode(inner, withRespectTo: param),
            )
        case let .tan(inner):
            .multiply(
                .divide(.constant(1.0), .multiply(.cos(inner), .cos(inner))),
                differentiateNode(inner, withRespectTo: param),
            )
        case let .exp(inner):
            .multiply(
                .exp(inner),
                differentiateNode(inner, withRespectTo: param),
            )
        case let .log(inner):
            .multiply(
                .divide(.constant(1.0), inner),
                differentiateNode(inner, withRespectTo: param),
            )
        case let .arctan(inner):
            .multiply(
                .divide(.constant(1.0), .add(.constant(1.0), .multiply(inner, inner))),
                differentiateNode(inner, withRespectTo: param),
            )
        }
    }
}

// MARK: - Arithmetic Operators

/// Addition of parameter expressions
///
/// **Example:**
/// ```swift
/// let theta = ParameterExpression(Parameter(name: "theta"))
/// let phi = ParameterExpression(Parameter(name: "phi"))
/// let sum = theta + phi
/// ```
///
/// - Complexity: O(1)
@_effects(readonly)
@inlinable
public func + (lhs: ParameterExpression, rhs: ParameterExpression) -> ParameterExpression {
    ParameterExpression(node: .add(lhs.node, rhs.node))
}

/// Subtraction of parameter expressions
///
/// **Example:**
/// ```swift
/// let theta = ParameterExpression(Parameter(name: "theta"))
/// let phi = ParameterExpression(Parameter(name: "phi"))
/// let diff = theta - phi
/// ```
///
/// - Complexity: O(1)
@_effects(readonly)
@inlinable
public func - (lhs: ParameterExpression, rhs: ParameterExpression) -> ParameterExpression {
    ParameterExpression(node: .subtract(lhs.node, rhs.node))
}

/// Multiplication of parameter expressions
///
/// **Example:**
/// ```swift
/// let theta = ParameterExpression(Parameter(name: "theta"))
/// let two = ParameterExpression(2.0)
/// let scaled = two * theta
/// ```
///
/// - Complexity: O(1)
@_effects(readonly)
@inlinable
public func * (lhs: ParameterExpression, rhs: ParameterExpression) -> ParameterExpression {
    ParameterExpression(node: .multiply(lhs.node, rhs.node))
}

/// Division of parameter expressions
///
/// **Example:**
/// ```swift
/// let theta = ParameterExpression(Parameter(name: "theta"))
/// let two = ParameterExpression(2.0)
/// let half = theta / two
/// ```
///
/// - Complexity: O(1)
@_effects(readonly)
@inlinable
public func / (lhs: ParameterExpression, rhs: ParameterExpression) -> ParameterExpression {
    ParameterExpression(node: .divide(lhs.node, rhs.node))
}

/// Negation of parameter expression
///
/// **Example:**
/// ```swift
/// let theta = ParameterExpression(Parameter(name: "theta"))
/// let negated = -theta
/// ```
///
/// - Complexity: O(1)
@_effects(readonly)
@inlinable
public prefix func - (expr: ParameterExpression) -> ParameterExpression {
    ParameterExpression(node: .negate(expr.node))
}

// MARK: - Transcendental Functions

public extension ParameterExpression {
    /// Sine of parameter expression
    ///
    /// **Example:**
    /// ```swift
    /// let theta = ParameterExpression(Parameter(name: "theta"))
    /// let sinTheta = ParameterExpression.sin(theta)
    /// ```
    ///
    /// - Parameter expr: Input expression
    /// - Returns: Expression representing sin(expr)
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    static func sin(_ expr: ParameterExpression) -> ParameterExpression {
        ParameterExpression(node: .sin(expr.node))
    }

    /// Cosine of parameter expression
    ///
    /// **Example:**
    /// ```swift
    /// let theta = ParameterExpression(Parameter(name: "theta"))
    /// let cosTheta = ParameterExpression.cos(theta)
    /// ```
    ///
    /// - Parameter expr: Input expression
    /// - Returns: Expression representing cos(expr)
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    static func cos(_ expr: ParameterExpression) -> ParameterExpression {
        ParameterExpression(node: .cos(expr.node))
    }

    /// Tangent of parameter expression
    ///
    /// **Example:**
    /// ```swift
    /// let theta = ParameterExpression(Parameter(name: "theta"))
    /// let tanTheta = ParameterExpression.tan(theta)
    /// ```
    ///
    /// - Parameter expr: Input expression
    /// - Returns: Expression representing tan(expr)
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    static func tan(_ expr: ParameterExpression) -> ParameterExpression {
        ParameterExpression(node: .tan(expr.node))
    }

    /// Exponential of parameter expression
    ///
    /// **Example:**
    /// ```swift
    /// let theta = ParameterExpression(Parameter(name: "theta"))
    /// let expTheta = ParameterExpression.exp(theta)
    /// ```
    ///
    /// - Parameter expr: Input expression
    /// - Returns: Expression representing exp(expr)
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    static func exp(_ expr: ParameterExpression) -> ParameterExpression {
        ParameterExpression(node: .exp(expr.node))
    }

    /// Natural logarithm of parameter expression
    ///
    /// **Example:**
    /// ```swift
    /// let theta = ParameterExpression(Parameter(name: "theta"))
    /// let logTheta = ParameterExpression.log(theta)
    /// ```
    ///
    /// - Parameter expr: Input expression
    /// - Returns: Expression representing log(expr)
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    static func log(_ expr: ParameterExpression) -> ParameterExpression {
        ParameterExpression(node: .log(expr.node))
    }

    /// Arctangent of parameter expression
    ///
    /// **Example:**
    /// ```swift
    /// let theta = ParameterExpression(Parameter(name: "theta"))
    /// let atanTheta = ParameterExpression.arctan(theta)
    /// ```
    ///
    /// - Parameter expr: Input expression
    /// - Returns: Expression representing arctan(expr)
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    static func arctan(_ expr: ParameterExpression) -> ParameterExpression {
        ParameterExpression(node: .arctan(expr.node))
    }
}
