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

    @Test("Convenience initializer from Parameter")
    func convenienceInitializerFromParameter() {
        let param = Parameter(name: "gamma")
        let expr = ParameterValue(param)

        #expect(expr.isSymbolic, "Should be symbolic when created from Parameter")
        #expect(expr.parameter == param, "Should contain the original parameter")
        #expect(expr == .parameter(param), "Should equal explicit .parameter case")
    }

    @Test("Negated parameter is symbolic")
    func negatedParameterIsSymbolic() {
        let param = Parameter(name: "theta")
        let expr = ParameterValue.negatedParameter(param)

        #expect(expr.isSymbolic, "Negated parameter should be symbolic")
    }

    @Test("Extract parameter from negated parameter expression")
    func extractParameterFromNegatedParameter() {
        let param = Parameter(name: "theta")
        let expr = ParameterValue.negatedParameter(param)

        let extracted = expr.parameter
        #expect(extracted == param, "Should extract original parameter from negated expression")
    }

    @Test("Evaluate negated parameter expression with binding")
    func evaluateNegatedParameterExpressionWithBinding() {
        let param = Parameter(name: "theta")
        let expr = ParameterValue.negatedParameter(param)
        let bindings = ["theta": 2.5]

        let result = expr.evaluate(using: bindings)
        #expect(abs(result - -2.5) < 1e-10, "Negated parameter should evaluate to negative of bound value")
    }

    @Test("Negated parameter expression description shows minus sign")
    func negatedParameterExpressionDescription() {
        let param = Parameter(name: "theta")
        let expr = ParameterValue.negatedParameter(param)

        #expect(expr.description == "-theta", "Description should show minus sign before parameter name")
    }

    @Test("Negating concrete value returns negated value")
    func negatingConcreteValue() {
        let expr = ParameterValue.value(1.5)
        let negated = expr.negated

        if case let .value(v) = negated {
            #expect(abs(v - -1.5) < 1e-10, "Negated value should be -1.5")
        }
    }

    @Test("Negating symbolic parameter returns negated parameter")
    func negatingSymbolicParameter() {
        let param = Parameter(name: "theta")
        let expr = ParameterValue.parameter(param)
        let negated = expr.negated

        #expect(negated == .negatedParameter(param), "Negating parameter should produce negatedParameter")
    }

    @Test("Negating negated parameter returns original parameter")
    func negatingNegatedParameterReturnsOriginal() {
        let param = Parameter(name: "theta")
        let expr = ParameterValue.negatedParameter(param)
        let doubleNegated = expr.negated

        #expect(doubleNegated == .parameter(param), "Double negation should return original parameter")
    }

    @Test("Double negation of concrete value returns original")
    func doubleNegationOfConcreteValue() {
        let expr = ParameterValue.value(3.14)
        let doubleNegated = expr.negated.negated

        if case let .value(v) = doubleNegated {
            #expect(abs(v - 3.14) < 1e-10, "Double negation should return original value")
        }
    }

    @Test("Negated parameter equality")
    func negatedParameterEquality() {
        let param1 = Parameter(name: "theta")
        let param2 = Parameter(name: "theta")
        let param3 = Parameter(name: "phi")

        let expr1 = ParameterValue.negatedParameter(param1)
        let expr2 = ParameterValue.negatedParameter(param2)
        let expr3 = ParameterValue.negatedParameter(param3)
        let expr4 = ParameterValue.parameter(param1)

        #expect(expr1 == expr2, "Same negated parameters should be equal")
        #expect(expr1 != expr3, "Different negated parameters should not be equal")
        #expect(expr1 != expr4, "Negated parameter should not equal non-negated parameter")
    }

    @Test("Negated parameter hashability")
    func negatedParameterHashability() {
        let param = Parameter(name: "theta")
        let negated = ParameterValue.negatedParameter(param)
        let regular = ParameterValue.parameter(param)

        var set = Set<ParameterValue>()
        set.insert(negated)
        set.insert(regular)

        #expect(set.count == 2, "Negated and regular parameter values should hash differently")
    }
}
