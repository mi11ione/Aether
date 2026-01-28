// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for ExpressionNode enum cases.
/// Validates constant, parameter, binary operations, unary negation,
/// and transcendental function node construction.
@Suite("ExpressionNode Cases")
struct ExpressionNodeCasesTests {
    @Test("Constant node stores value")
    func constantNode() {
        let node = ExpressionNode.constant(3.14)
        #expect(node == .constant(3.14), "Constant node should store exact value")
    }

    @Test("Parameter node stores parameter")
    func parameterNode() {
        let theta = Parameter(name: "theta")
        let node = ExpressionNode.parameter(theta)
        #expect(node == .parameter(theta), "Parameter node should store parameter")
    }

    @Test("Add node stores two operands")
    func addNode() {
        let lhs = ExpressionNode.constant(1.0)
        let rhs = ExpressionNode.constant(2.0)
        let node = ExpressionNode.add(lhs, rhs)
        #expect(node == .add(.constant(1.0), .constant(2.0)), "Add node should store both operands")
    }

    @Test("Subtract node stores two operands")
    func subtractNode() {
        let lhs = ExpressionNode.constant(5.0)
        let rhs = ExpressionNode.constant(3.0)
        let node = ExpressionNode.subtract(lhs, rhs)
        #expect(node == .subtract(.constant(5.0), .constant(3.0)), "Subtract node should store both operands")
    }

    @Test("Multiply node stores two operands")
    func multiplyNode() {
        let lhs = ExpressionNode.constant(2.0)
        let rhs = ExpressionNode.constant(3.0)
        let node = ExpressionNode.multiply(lhs, rhs)
        #expect(node == .multiply(.constant(2.0), .constant(3.0)), "Multiply node should store both operands")
    }

    @Test("Divide node stores two operands")
    func divideNode() {
        let lhs = ExpressionNode.constant(6.0)
        let rhs = ExpressionNode.constant(2.0)
        let node = ExpressionNode.divide(lhs, rhs)
        #expect(node == .divide(.constant(6.0), .constant(2.0)), "Divide node should store both operands")
    }

    @Test("Negate node stores operand")
    func negateNode() {
        let inner = ExpressionNode.constant(5.0)
        let node = ExpressionNode.negate(inner)
        #expect(node == .negate(.constant(5.0)), "Negate node should store operand")
    }

    @Test("Sin node stores operand")
    func sinNode() {
        let inner = ExpressionNode.constant(0.0)
        let node = ExpressionNode.sin(inner)
        #expect(node == .sin(.constant(0.0)), "Sin node should store operand")
    }

    @Test("Cos node stores operand")
    func cosNode() {
        let inner = ExpressionNode.constant(0.0)
        let node = ExpressionNode.cos(inner)
        #expect(node == .cos(.constant(0.0)), "Cos node should store operand")
    }

    @Test("Tan node stores operand")
    func tanNode() {
        let inner = ExpressionNode.constant(0.0)
        let node = ExpressionNode.tan(inner)
        #expect(node == .tan(.constant(0.0)), "Tan node should store operand")
    }

    @Test("Exp node stores operand")
    func expNode() {
        let inner = ExpressionNode.constant(1.0)
        let node = ExpressionNode.exp(inner)
        #expect(node == .exp(.constant(1.0)), "Exp node should store operand")
    }

    @Test("Log node stores operand")
    func logNode() {
        let inner = ExpressionNode.constant(1.0)
        let node = ExpressionNode.log(inner)
        #expect(node == .log(.constant(1.0)), "Log node should store operand")
    }

    @Test("Arctan node stores operand")
    func arctanNode() {
        let inner = ExpressionNode.constant(1.0)
        let node = ExpressionNode.arctan(inner)
        #expect(node == .arctan(.constant(1.0)), "Arctan node should store operand")
    }
}

/// Test suite for ParameterExpression initialization.
/// Validates construction from nodes, constants, and parameters
/// for building symbolic expression trees.
@Suite("ParameterExpression Initialization")
struct ParameterExpressionInitTests {
    @Test("Init from constant creates constant node")
    func initFromConstant() {
        let expr = ParameterExpression(2.5)
        #expect(expr.node == .constant(2.5), "Should create constant node")
    }

    @Test("Init from parameter creates parameter node")
    func initFromParameter() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression(theta)
        #expect(expr.node == .parameter(theta), "Should create parameter node")
    }

    @Test("Init from node stores node directly")
    func initFromNode() {
        let node = ExpressionNode.add(.constant(1.0), .constant(2.0))
        let expr = ParameterExpression(node: node)
        #expect(expr.node == node, "Should store node directly")
    }
}

/// Test suite for isSymbolic property.
/// Validates detection of symbolic vs constant expressions
/// to determine binding requirements.
@Suite("ParameterExpression isSymbolic Property")
struct ParameterExpressionIsSymbolicTests {
    @Test("Constant expression is not symbolic")
    func constantNotSymbolic() {
        let expr = ParameterExpression(3.14)
        #expect(expr.isSymbolic == false, "Constant should not be symbolic")
    }

    @Test("Parameter expression is symbolic")
    func parameterIsSymbolic() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression(theta)
        #expect(expr.isSymbolic == true, "Parameter should be symbolic")
    }

    @Test("Expression with parameter is symbolic")
    func expressionWithParameterIsSymbolic() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression(2.0) * ParameterExpression(theta)
        #expect(expr.isSymbolic == true, "Expression containing parameter should be symbolic")
    }

    @Test("Expression of only constants is not symbolic")
    func expressionOfConstantsNotSymbolic() {
        let expr = ParameterExpression(2.0) + ParameterExpression(3.0)
        #expect(expr.isSymbolic == false, "Expression of constants only should not be symbolic")
    }
}

/// Test suite for parameters property.
/// Validates extraction of all symbolic parameters
/// from expression trees for binding discovery.
@Suite("ParameterExpression Parameters Property")
struct ParameterExpressionParametersTests {
    @Test("Constant has no parameters")
    func constantNoParameters() {
        let expr = ParameterExpression(5.0)
        #expect(expr.parameters.isEmpty, "Constant should have no parameters")
    }

    @Test("Single parameter expression returns that parameter")
    func singleParameter() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression(theta)
        #expect(expr.parameters == [theta], "Should contain the parameter")
    }

    @Test("Multiple parameters collected from expression")
    func multipleParameters() {
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        let expr = ParameterExpression(theta) + ParameterExpression(phi)
        #expect(expr.parameters.count == 2, "Should have two parameters")
        #expect(expr.parameters.contains(theta), "Should contain theta")
        #expect(expr.parameters.contains(phi), "Should contain phi")
    }

    @Test("Duplicate parameter counted once")
    func duplicateParameterCountedOnce() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression(theta) * ParameterExpression(theta)
        #expect(expr.parameters.count == 1, "Duplicate parameter should be counted once")
        #expect(expr.parameters.contains(theta), "Should contain theta")
    }

    @Test("Parameters collected through transcendental")
    func parametersThroughTranscendental() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression.sin(ParameterExpression(theta))
        #expect(expr.parameters == [theta], "Should collect parameter through sin")
    }
}

/// Test suite for evaluate method.
/// Validates numerical evaluation of expressions
/// using parameter bindings for all node types.
@Suite("ParameterExpression Evaluate")
struct ParameterExpressionEvaluateTests {
    let tolerance = 1e-10

    @Test("Evaluate constant returns value")
    func evaluateConstant() {
        let expr = ParameterExpression(3.14)
        let result = expr.evaluate(using: [:])
        #expect(abs(result - 3.14) < tolerance, "Should evaluate to constant value")
    }

    @Test("Evaluate parameter uses binding")
    func evaluateParameter() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression(theta)
        let result = expr.evaluate(using: ["theta": 1.5])
        #expect(abs(result - 1.5) < tolerance, "Should use bound value")
    }

    @Test("Evaluate add computes sum")
    func evaluateAdd() {
        let expr = ParameterExpression(2.0) + ParameterExpression(3.0)
        let result = expr.evaluate(using: [:])
        #expect(abs(result - 5.0) < tolerance, "2 + 3 should equal 5")
    }

    @Test("Evaluate subtract computes difference")
    func evaluateSubtract() {
        let expr = ParameterExpression(5.0) - ParameterExpression(3.0)
        let result = expr.evaluate(using: [:])
        #expect(abs(result - 2.0) < tolerance, "5 - 3 should equal 2")
    }

    @Test("Evaluate multiply computes product")
    func evaluateMultiply() {
        let expr = ParameterExpression(2.0) * ParameterExpression(3.0)
        let result = expr.evaluate(using: [:])
        #expect(abs(result - 6.0) < tolerance, "2 * 3 should equal 6")
    }

    @Test("Evaluate divide computes quotient")
    func evaluateDivide() {
        let expr = ParameterExpression(6.0) / ParameterExpression(2.0)
        let result = expr.evaluate(using: [:])
        #expect(abs(result - 3.0) < tolerance, "6 / 2 should equal 3")
    }

    @Test("Evaluate negate computes negative")
    func evaluateNegate() {
        let expr = -ParameterExpression(5.0)
        let result = expr.evaluate(using: [:])
        #expect(abs(result - -5.0) < tolerance, "-5 should equal -5")
    }

    @Test("Evaluate sin computes sine")
    func evaluateSin() {
        let expr = ParameterExpression.sin(ParameterExpression(0.0))
        let result = expr.evaluate(using: [:])
        #expect(abs(result - 0.0) < tolerance, "sin(0) should equal 0")
    }

    @Test("Evaluate cos computes cosine")
    func evaluateCos() {
        let expr = ParameterExpression.cos(ParameterExpression(0.0))
        let result = expr.evaluate(using: [:])
        #expect(abs(result - 1.0) < tolerance, "cos(0) should equal 1")
    }

    @Test("Evaluate tan computes tangent")
    func evaluateTan() {
        let expr = ParameterExpression.tan(ParameterExpression(0.0))
        let result = expr.evaluate(using: [:])
        #expect(abs(result - 0.0) < tolerance, "tan(0) should equal 0")
    }

    @Test("Evaluate exp computes exponential")
    func evaluateExp() {
        let expr = ParameterExpression.exp(ParameterExpression(0.0))
        let result = expr.evaluate(using: [:])
        #expect(abs(result - 1.0) < tolerance, "exp(0) should equal 1")
    }

    @Test("Evaluate log computes natural logarithm")
    func evaluateLog() {
        let expr = ParameterExpression.log(ParameterExpression(1.0))
        let result = expr.evaluate(using: [:])
        #expect(abs(result - 0.0) < tolerance, "log(1) should equal 0")
    }

    @Test("Evaluate arctan computes arctangent")
    func evaluateArctan() {
        let expr = ParameterExpression.arctan(ParameterExpression(0.0))
        let result = expr.evaluate(using: [:])
        #expect(abs(result - 0.0) < tolerance, "arctan(0) should equal 0")
    }

    @Test("Evaluate complex expression with parameters")
    func evaluateComplexExpression() {
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        let expr = ParameterExpression(theta) + ParameterExpression(2.0) * ParameterExpression(phi)
        let result = expr.evaluate(using: ["theta": 1.0, "phi": 2.0])
        #expect(abs(result - 5.0) < tolerance, "1 + 2*2 should equal 5")
    }
}

/// Test suite for gradient computation.
/// Validates symbolic differentiation for all operations
/// including chain rule for transcendentals.
@Suite("ParameterExpression Gradient")
struct ParameterExpressionGradientTests {
    let tolerance = 1e-10

    @Test("Gradient of constant is zero")
    func gradientOfConstant() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression(5.0)
        let grad = expr.gradient(withRespectTo: theta)
        let result = grad.evaluate(using: ["theta": 1.0])
        #expect(abs(result - 0.0) < tolerance, "d/dtheta(5) = 0")
    }

    @Test("Gradient of parameter with respect to itself is one")
    func gradientOfSameParameter() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression(theta)
        let grad = expr.gradient(withRespectTo: theta)
        let result = grad.evaluate(using: ["theta": 1.0])
        #expect(abs(result - 1.0) < tolerance, "d/dtheta(theta) = 1")
    }

    @Test("Gradient of parameter with respect to different parameter is zero")
    func gradientOfDifferentParameter() {
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        let expr = ParameterExpression(phi)
        let grad = expr.gradient(withRespectTo: theta)
        let result = grad.evaluate(using: ["theta": 1.0, "phi": 2.0])
        #expect(abs(result - 0.0) < tolerance, "d/dtheta(phi) = 0")
    }

    @Test("Gradient of sum is sum of gradients")
    func gradientOfSum() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression(theta) + ParameterExpression(3.0)
        let grad = expr.gradient(withRespectTo: theta)
        let result = grad.evaluate(using: ["theta": 1.0])
        #expect(abs(result - 1.0) < tolerance, "d/dtheta(theta + 3) = 1")
    }

    @Test("Gradient of difference is difference of gradients")
    func gradientOfDifference() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression(theta) - ParameterExpression(2.0)
        let grad = expr.gradient(withRespectTo: theta)
        let result = grad.evaluate(using: ["theta": 1.0])
        #expect(abs(result - 1.0) < tolerance, "d/dtheta(theta - 2) = 1")
    }

    @Test("Gradient of product uses product rule")
    func gradientOfProduct() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression(2.0) * ParameterExpression(theta)
        let grad = expr.gradient(withRespectTo: theta)
        let result = grad.evaluate(using: ["theta": 3.0])
        #expect(abs(result - 2.0) < tolerance, "d/dtheta(2*theta) = 2")
    }

    @Test("Gradient of quotient uses quotient rule")
    func gradientOfQuotient() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression(theta) / ParameterExpression(2.0)
        let grad = expr.gradient(withRespectTo: theta)
        let result = grad.evaluate(using: ["theta": 4.0])
        #expect(abs(result - 0.5) < tolerance, "d/dtheta(theta/2) = 0.5")
    }

    @Test("Gradient of negation is negated gradient")
    func gradientOfNegation() {
        let theta = Parameter(name: "theta")
        let expr = -ParameterExpression(theta)
        let grad = expr.gradient(withRespectTo: theta)
        let result = grad.evaluate(using: ["theta": 1.0])
        #expect(abs(result - -1.0) < tolerance, "d/dtheta(-theta) = -1")
    }

    @Test("Gradient of sin is cos")
    func gradientOfSin() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression.sin(ParameterExpression(theta))
        let grad = expr.gradient(withRespectTo: theta)
        let result = grad.evaluate(using: ["theta": 0.0])
        #expect(abs(result - 1.0) < tolerance, "d/dtheta(sin(theta)) at 0 = cos(0) = 1")
    }

    @Test("Gradient of cos is negative sin")
    func gradientOfCos() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression.cos(ParameterExpression(theta))
        let grad = expr.gradient(withRespectTo: theta)
        let result = grad.evaluate(using: ["theta": 0.0])
        #expect(abs(result - 0.0) < tolerance, "d/dtheta(cos(theta)) at 0 = -sin(0) = 0")
    }

    @Test("Gradient of tan is sec squared")
    func gradientOfTan() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression.tan(ParameterExpression(theta))
        let grad = expr.gradient(withRespectTo: theta)
        let result = grad.evaluate(using: ["theta": 0.0])
        #expect(abs(result - 1.0) < tolerance, "d/dtheta(tan(theta)) at 0 = sec^2(0) = 1")
    }

    @Test("Gradient of exp is exp")
    func gradientOfExp() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression.exp(ParameterExpression(theta))
        let grad = expr.gradient(withRespectTo: theta)
        let result = grad.evaluate(using: ["theta": 0.0])
        #expect(abs(result - 1.0) < tolerance, "d/dtheta(exp(theta)) at 0 = exp(0) = 1")
    }

    @Test("Gradient of log is reciprocal")
    func gradientOfLog() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression.log(ParameterExpression(theta))
        let grad = expr.gradient(withRespectTo: theta)
        let result = grad.evaluate(using: ["theta": 2.0])
        #expect(abs(result - 0.5) < tolerance, "d/dtheta(log(theta)) at 2 = 1/2")
    }

    @Test("Gradient of arctan is 1/(1+x^2)")
    func gradientOfArctan() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression.arctan(ParameterExpression(theta))
        let grad = expr.gradient(withRespectTo: theta)
        let result = grad.evaluate(using: ["theta": 1.0])
        #expect(abs(result - 0.5) < tolerance, "d/dtheta(arctan(theta)) at 1 = 1/(1+1) = 0.5")
    }
}

/// Test suite for arithmetic operators.
/// Validates +, -, *, / and prefix - operators
/// create correct expression nodes.
@Suite("ParameterExpression Operators")
struct ParameterExpressionOperatorsTests {
    let tolerance = 1e-10

    @Test("Addition operator creates add node")
    func additionOperator() {
        let a = ParameterExpression(2.0)
        let b = ParameterExpression(3.0)
        let result = a + b
        #expect(result.node == .add(.constant(2.0), .constant(3.0)), "Should create add node")
    }

    @Test("Subtraction operator creates subtract node")
    func subtractionOperator() {
        let a = ParameterExpression(5.0)
        let b = ParameterExpression(3.0)
        let result = a - b
        #expect(result.node == .subtract(.constant(5.0), .constant(3.0)), "Should create subtract node")
    }

    @Test("Multiplication operator creates multiply node")
    func multiplicationOperator() {
        let a = ParameterExpression(2.0)
        let b = ParameterExpression(4.0)
        let result = a * b
        #expect(result.node == .multiply(.constant(2.0), .constant(4.0)), "Should create multiply node")
    }

    @Test("Division operator creates divide node")
    func divisionOperator() {
        let a = ParameterExpression(8.0)
        let b = ParameterExpression(2.0)
        let result = a / b
        #expect(result.node == .divide(.constant(8.0), .constant(2.0)), "Should create divide node")
    }

    @Test("Prefix negation operator creates negate node")
    func negationOperator() {
        let a = ParameterExpression(5.0)
        let result = -a
        #expect(result.node == .negate(.constant(5.0)), "Should create negate node")
    }

    @Test("Chained operators evaluate correctly")
    func chainedOperators() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression(2.0) * ParameterExpression(theta) + ParameterExpression(1.0)
        let result = expr.evaluate(using: ["theta": 3.0])
        #expect(abs(result - 7.0) < tolerance, "2*3 + 1 should equal 7")
    }
}

/// Test suite for transcendental function static methods.
/// Validates sin, cos, tan, exp, log, arctan create
/// correct expression nodes and evaluate accurately.
@Suite("ParameterExpression Transcendentals")
struct ParameterExpressionTranscendentalsTests {
    let tolerance = 1e-10

    @Test("Sin at pi/2 equals 1")
    func sinAtPiOver2() {
        let expr = ParameterExpression.sin(ParameterExpression(.pi / 2.0))
        let result = expr.evaluate(using: [:])
        #expect(abs(result - 1.0) < tolerance, "sin(pi/2) should equal 1")
    }

    @Test("Cos at pi equals -1")
    func cosAtPi() {
        let expr = ParameterExpression.cos(ParameterExpression(.pi))
        let result = expr.evaluate(using: [:])
        #expect(abs(result - -1.0) < tolerance, "cos(pi) should equal -1")
    }

    @Test("Tan at pi/4 equals 1")
    func tanAtPiOver4() {
        let expr = ParameterExpression.tan(ParameterExpression(.pi / 4.0))
        let result = expr.evaluate(using: [:])
        #expect(abs(result - 1.0) < tolerance, "tan(pi/4) should equal 1")
    }

    @Test("Exp at 1 equals e")
    func expAt1() {
        let expr = ParameterExpression.exp(ParameterExpression(1.0))
        let result = expr.evaluate(using: [:])
        #expect(abs(result - Darwin.M_E) < tolerance, "exp(1) should equal e")
    }

    @Test("Log at e equals 1")
    func logAtE() {
        let expr = ParameterExpression.log(ParameterExpression(Darwin.M_E))
        let result = expr.evaluate(using: [:])
        #expect(abs(result - 1.0) < tolerance, "log(e) should equal 1")
    }

    @Test("Arctan at 1 equals pi/4")
    func arctanAt1() {
        let expr = ParameterExpression.arctan(ParameterExpression(1.0))
        let result = expr.evaluate(using: [:])
        #expect(abs(result - .pi / 4.0) < tolerance, "arctan(1) should equal pi/4")
    }

    @Test("Transcendental with parameter evaluates correctly")
    func transcendentalWithParameter() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression.sin(ParameterExpression(theta))
        let result = expr.evaluate(using: ["theta": .pi / 6.0])
        #expect(abs(result - 0.5) < tolerance, "sin(pi/6) should equal 0.5")
    }
}

/// Test suite for edge cases and special values.
/// Validates divide by zero, log of negative, nested expressions,
/// and numerical stability at boundaries.
@Suite("ParameterExpression Edge Cases")
struct ParameterExpressionEdgeCasesTests {
    let tolerance = 1e-10

    @Test("Divide by zero returns infinity")
    func divideByZero() {
        let expr = ParameterExpression(1.0) / ParameterExpression(0.0)
        let result = expr.evaluate(using: [:])
        #expect(result.isInfinite, "Division by zero should return infinity")
    }

    @Test("Log of negative returns NaN")
    func logOfNegative() {
        let expr = ParameterExpression.log(ParameterExpression(-1.0))
        let result = expr.evaluate(using: [:])
        #expect(result.isNaN, "Log of negative should return NaN")
    }

    @Test("Deeply nested expression evaluates correctly")
    func deeplyNestedExpression() {
        let theta = Parameter(name: "theta")
        let inner = ParameterExpression(theta)
        let expr = ParameterExpression.sin(ParameterExpression.cos(inner))
        let result = expr.evaluate(using: ["theta": 0.0])
        let expected = Foundation.sin(Foundation.cos(0.0))
        #expect(abs(result - expected) < tolerance, "sin(cos(0)) should equal sin(1)")
    }

    @Test("Expression with same parameter multiple times")
    func sameParameterMultipleTimes() {
        let theta = Parameter(name: "theta")
        let t = ParameterExpression(theta)
        let expr = t * t + t
        let result = expr.evaluate(using: ["theta": 3.0])
        #expect(abs(result - 12.0) < tolerance, "3*3 + 3 should equal 12")
    }

    @Test("Zero minus expression equals negation")
    func zeroMinusExpression() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression(0.0) - ParameterExpression(theta)
        let result = expr.evaluate(using: ["theta": 5.0])
        #expect(abs(result - -5.0) < tolerance, "0 - 5 should equal -5")
    }

    @Test("Expression evaluates zero correctly")
    func expressionEvaluatesZero() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression(theta) - ParameterExpression(theta)
        let result = expr.evaluate(using: ["theta": 7.0])
        #expect(abs(result - 0.0) < tolerance, "theta - theta should equal 0")
    }
}

/// Test suite for Equatable and Hashable conformance.
/// Validates expression equality and hash consistency
/// for use in collections and caching.
@Suite("ParameterExpression Equatable Hashable")
struct ParameterExpressionEquatableHashableTests {
    @Test("Equal expressions are equal")
    func equalExpressions() {
        let expr1 = ParameterExpression(3.14)
        let expr2 = ParameterExpression(3.14)
        #expect(expr1 == expr2, "Same constant expressions should be equal")
    }

    @Test("Different expressions are not equal")
    func differentExpressions() {
        let expr1 = ParameterExpression(3.14)
        let expr2 = ParameterExpression(2.71)
        #expect(expr1 != expr2, "Different constant expressions should not be equal")
    }

    @Test("Equal expressions have same hash")
    func equalExpressionsHash() {
        let theta = Parameter(name: "theta")
        let expr1 = ParameterExpression(theta)
        let expr2 = ParameterExpression(theta)
        #expect(expr1.hashValue == expr2.hashValue, "Equal expressions should have same hash")
    }

    @Test("Complex equal expressions are equal")
    func complexEqualExpressions() {
        let theta = Parameter(name: "theta")
        let expr1 = ParameterExpression(theta) + ParameterExpression(1.0)
        let expr2 = ParameterExpression(theta) + ParameterExpression(1.0)
        #expect(expr1 == expr2, "Same structure expressions should be equal")
    }
}

/// Test suite for chain rule in gradient computation.
/// Validates correct differentiation of composed functions
/// for variational quantum optimization.
@Suite("ParameterExpression Chain Rule")
struct ParameterExpressionChainRuleTests {
    let tolerance = 1e-10

    @Test("Chain rule for sin(2*theta)")
    func chainRuleSin2Theta() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression.sin(ParameterExpression(2.0) * ParameterExpression(theta))
        let grad = expr.gradient(withRespectTo: theta)
        let result = grad.evaluate(using: ["theta": 0.0])
        #expect(abs(result - 2.0) < tolerance, "d/dtheta(sin(2*theta)) at 0 = 2*cos(0) = 2")
    }

    @Test("Chain rule for exp(theta^2)")
    func chainRuleExpThetaSquared() {
        let theta = Parameter(name: "theta")
        let t = ParameterExpression(theta)
        let expr = ParameterExpression.exp(t * t)
        let grad = expr.gradient(withRespectTo: theta)
        let result = grad.evaluate(using: ["theta": 0.0])
        #expect(abs(result - 0.0) < tolerance, "d/dtheta(exp(theta^2)) at 0 = 2*0*exp(0) = 0")
    }

    @Test("Chain rule for log(theta+1)")
    func chainRuleLogThetaPlusOne() {
        let theta = Parameter(name: "theta")
        let expr = ParameterExpression.log(ParameterExpression(theta) + ParameterExpression(1.0))
        let grad = expr.gradient(withRespectTo: theta)
        let result = grad.evaluate(using: ["theta": 1.0])
        #expect(abs(result - 0.5) < tolerance, "d/dtheta(log(theta+1)) at 1 = 1/(1+1) = 0.5")
    }
}
