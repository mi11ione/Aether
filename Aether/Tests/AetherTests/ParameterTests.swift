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

/// Test suite for ParameterValue type.
/// Validates symbolic and concrete parameter expressions,
/// evaluation logic, and error handling for unbound parameters.
@Suite("ParameterValue")
struct ParameterValueTests {
    @Test("Create symbolic parameter expression")
    func createSymbolicExpression() {
        let param = Parameter(name: "theta")
        let expr = ParameterValue.parameter(param)

        #expect(expr.isSymbolic)
    }

    @Test("Create concrete value expression")
    func createConcreteExpression() {
        let expr = ParameterValue.value(Double.pi / 4.0)

        #expect(!expr.isSymbolic)
    }

    @Test("Evaluate concrete expression")
    func evaluateConcreteExpression() {
        let expr = ParameterValue.value(1.5)
        let result = expr.evaluate(using: [:])

        #expect(abs(result - 1.5) < 1e-10)
    }

    @Test("Evaluate symbolic expression with binding")
    func evaluateSymbolicExpressionWithBinding() {
        let param = Parameter(name: "theta")
        let expr = ParameterValue.parameter(param)
        let bindings = ["theta": 0.5]

        let result = expr.evaluate(using: bindings)
        #expect(abs(result - 0.5) < 1e-10)
    }

    @Test("Extract parameter from symbolic expression")
    func extractParameterFromSymbolic() {
        let param = Parameter(name: "theta")
        let expr = ParameterValue.parameter(param)

        let extracted = expr.parameter
        #expect(extracted == param)
    }

    @Test("Extract parameter from concrete expression returns nil")
    func extractParameterFromConcreteReturnsNil() {
        let expr = ParameterValue.value(1.5)
        let extracted = expr.parameter

        #expect(extracted == nil)
    }

    @Test("Symbolic expression description shows parameter name")
    func symbolicExpressionDescription() {
        let param = Parameter(name: "theta")
        let expr = ParameterValue.parameter(param)

        #expect(expr.description == "theta")
    }

    @Test("Concrete expression description shows formatted value")
    func concreteExpressionDescription() {
        let expr = ParameterValue.value(1.234)
        #expect(expr.description == "1.234")
    }

    @Test("ParameterValue equality")
    func ParameterValueEquality() {
        let param1 = Parameter(name: "theta")
        let param2 = Parameter(name: "theta")
        let param3 = Parameter(name: "phi")

        let expr1 = ParameterValue.parameter(param1)
        let expr2 = ParameterValue.parameter(param2)
        let expr3 = ParameterValue.parameter(param3)
        let expr4 = ParameterValue.value(1.0)
        let expr5 = ParameterValue.value(1.0)

        #expect(expr1 == expr2)
        #expect(expr1 != expr3)
        #expect(expr1 != expr4)
        #expect(expr4 == expr5)
    }

    @Test("ParameterValue hashability")
    func ParameterValueHashability() {
        let param = Parameter(name: "theta")
        let expr1 = ParameterValue.parameter(param)
        let expr2 = ParameterValue.value(1.0)

        var set = Set<ParameterValue>()
        set.insert(expr1)
        set.insert(expr2)

        #expect(set.count == 2)
    }
}
