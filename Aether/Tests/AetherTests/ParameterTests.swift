// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Test suite for Parameter type.
/// Validates symbolic parameter creation, equality, hashing,
/// and name validation for variational quantum circuits.
@Suite("Parameter")
struct ParameterTests {
    @Test("Create parameter with valid name")
    func createParameterValidName() {
        let param = Parameter(name: "theta")
        #expect(param.name == "theta")
    }

    @Test("Parameters with same name are equal")
    func parametersWithSameNameEqual() {
        let param1 = Parameter(name: "theta")
        let param2 = Parameter(name: "theta")
        #expect(param1 == param2)
    }

    @Test("Parameters with different names are not equal")
    func parametersWithDifferentNamesNotEqual() {
        let param1 = Parameter(name: "theta")
        let param2 = Parameter(name: "phi")
        #expect(param1 != param2)
    }

    @Test("Parameters are hashable")
    func parametersAreHashable() {
        let param1 = Parameter(name: "theta")
        let param2 = Parameter(name: "theta")
        let param3 = Parameter(name: "phi")

        var set = Set<Parameter>()
        set.insert(param1)
        set.insert(param2)
        set.insert(param3)

        #expect(set.count == 2)
        #expect(set.contains(param1))
        #expect(set.contains(param3))
    }

    @Test("Parameter description returns name")
    func parameterDescription() {
        let param = Parameter(name: "theta_0")
        #expect(param.description == "theta_0")
    }

    @Test("Greek letter parameter names")
    func greekLetterNames() {
        let theta = Parameter(name: "θ")
        let phi = Parameter(name: "φ")
        let gamma = Parameter(name: "γ")

        #expect(theta.name == "θ")
        #expect(phi.name == "φ")
        #expect(gamma.name == "γ")
    }

    @Test("Numbered parameter names")
    func numberedParameterNames() {
        let param0 = Parameter(name: "theta_0")
        let param1 = Parameter(name: "theta_1")
        let param2 = Parameter(name: "theta_2")

        #expect(param0 != param1)
        #expect(param1 != param2)
    }
}

/// Test suite for ParameterExpression type.
/// Validates symbolic and concrete parameter expressions,
/// evaluation logic, and error handling for unbound parameters.
@Suite("ParameterExpression")
struct ParameterExpressionTests {
    @Test("Create symbolic parameter expression")
    func createSymbolicExpression() {
        let param = Parameter(name: "theta")
        let expr = ParameterExpression.parameter(param)

        #expect(expr.isSymbolic())
        #expect(!expr.isConcrete())
    }

    @Test("Create concrete value expression")
    func createConcreteExpression() {
        let expr = ParameterExpression.value(Double.pi / 4.0)

        #expect(expr.isConcrete())
        #expect(!expr.isSymbolic())
    }

    @Test("Evaluate concrete expression")
    func evaluateConcreteExpression() throws {
        let expr = ParameterExpression.value(1.5)
        let result = try expr.evaluate(with: [:])

        #expect(abs(result - 1.5) < 1e-10)
    }

    @Test("Evaluate symbolic expression with binding")
    func evaluateSymbolicExpressionWithBinding() throws {
        let param = Parameter(name: "theta")
        let expr = ParameterExpression.parameter(param)
        let bindings = ["theta": 0.5]

        let result = try expr.evaluate(with: bindings)
        #expect(abs(result - 0.5) < 1e-10)
    }

    @Test("Evaluate symbolic expression throws without binding")
    func evaluateSymbolicExpressionThrowsWithoutBinding() {
        let param = Parameter(name: "theta")
        let expr = ParameterExpression.parameter(param)
        let bindings: [String: Double] = [:]

        #expect(throws: ParameterError.self) {
            try expr.evaluate(with: bindings)
        }
    }

    @Test("Evaluate symbolic expression throws with wrong binding")
    func evaluateSymbolicExpressionThrowsWrongBinding() {
        let param = Parameter(name: "theta")
        let expr = ParameterExpression.parameter(param)
        let bindings = ["phi": 0.5]

        #expect(throws: ParameterError.self) {
            try expr.evaluate(with: bindings)
        }
    }

    @Test("Extract parameter from symbolic expression")
    func extractParameterFromSymbolic() {
        let param = Parameter(name: "theta")
        let expr = ParameterExpression.parameter(param)

        let extracted = expr.extractParameter()
        #expect(extracted == param)
    }

    @Test("Extract parameter from concrete expression returns nil")
    func extractParameterFromConcreteReturnsNil() {
        let expr = ParameterExpression.value(1.5)
        let extracted = expr.extractParameter()

        #expect(extracted == nil)
    }

    @Test("Symbolic expression description shows parameter name")
    func symbolicExpressionDescription() {
        let param = Parameter(name: "theta")
        let expr = ParameterExpression.parameter(param)

        #expect(expr.description == "theta")
    }

    @Test("Concrete expression description shows formatted value")
    func concreteExpressionDescription() {
        let expr = ParameterExpression.value(1.234)
        #expect(expr.description == "1.234")
    }

    @Test("ParameterExpression equality")
    func parameterExpressionEquality() {
        let param1 = Parameter(name: "theta")
        let param2 = Parameter(name: "theta")
        let param3 = Parameter(name: "phi")

        let expr1 = ParameterExpression.parameter(param1)
        let expr2 = ParameterExpression.parameter(param2)
        let expr3 = ParameterExpression.parameter(param3)
        let expr4 = ParameterExpression.value(1.0)
        let expr5 = ParameterExpression.value(1.0)

        #expect(expr1 == expr2)
        #expect(expr1 != expr3)
        #expect(expr1 != expr4)
        #expect(expr4 == expr5)
    }

    @Test("ParameterExpression hashability")
    func parameterExpressionHashability() {
        let param = Parameter(name: "theta")
        let expr1 = ParameterExpression.parameter(param)
        let expr2 = ParameterExpression.value(1.0)

        var set = Set<ParameterExpression>()
        set.insert(expr1)
        set.insert(expr2)

        #expect(set.count == 2)
    }
}

/// Test suite for ParameterError type.
/// Validates all error cases with descriptive error messages
/// for robust error handling in variational quantum circuits.
@Suite("ParameterError")
struct ParameterErrorTests {
    @Test("Unbound parameter error")
    func unboundParameterError() {
        let error = ParameterError.unboundParameter("theta")
        let description = error.errorDescription

        #expect(description != nil)
        #expect(description!.contains("theta"))
        #expect(description!.contains("not bound"))
    }

    @Test("Extra parameters error")
    func extraParametersError() {
        let error = ParameterError.extraParameters(["gamma", "delta"])
        let description = error.errorDescription

        #expect(description != nil)
        #expect(description!.contains("Extra parameters"))
    }

    @Test("Invalid vector length error")
    func invalidVectorLengthError() {
        let error = ParameterError.invalidVectorLength(expected: 3, got: 2)
        let description = error.errorDescription

        #expect(description != nil)
        #expect(description!.contains("3"))
        #expect(description!.contains("2"))
        #expect(description!.contains("mismatch"))
    }

    @Test("Empty parameter name error")
    func emptyParameterNameError() {
        let error = ParameterError.emptyParameterName
        let description = error.errorDescription

        #expect(description != nil)
        #expect(description!.contains("empty"))
    }

    @Test("Parameter not found error")
    func parameterNotFoundError() {
        let error = ParameterError.parameterNotFound("theta")
        let description = error.errorDescription

        #expect(description != nil)
        #expect(description!.contains("theta"))
        #expect(description!.contains("not found"))
    }

    @Test("Parameter index out of bounds error")
    func parameterIndexOutOfBoundsError() {
        let error = ParameterError.parameterIndexOutOfBounds(index: 5, count: 3)
        let description = error.errorDescription

        #expect(description != nil)
        #expect(description!.contains("5"))
        #expect(description!.contains("3"))
        #expect(description!.contains("out of bounds"))
    }

    @Test("ParameterError is Equatable")
    func parameterErrorIsEquatable() {
        let error1 = ParameterError.unboundParameter("theta")
        let error2 = ParameterError.unboundParameter("theta")
        let error3 = ParameterError.unboundParameter("phi")

        #expect(error1 == error2)
        #expect(error1 != error3)
    }

    @Test("Different error types are not equal")
    func differentErrorTypesNotEqual() {
        let error1 = ParameterError.unboundParameter("theta")
        let error2 = ParameterError.emptyParameterName

        #expect(error1 != error2)
    }
}
