// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Test suite for ParameterConstraint type.
/// Validates bounded, nonNegative, and periodic constraints,
/// apply method, bounds property, and factory validation.
@Suite("ParameterConstraint")
struct ParameterConstraintTests {
    @Test("Bounded constraint clamps value below minimum")
    func boundedClampsBelowMin() {
        let constraint = ParameterConstraint.bounded(min: -1.0, max: 1.0)
        let result = constraint.apply(to: -5.0)
        #expect(abs(result - -1.0) < 1e-10, "Value below min should clamp to min (-1.0)")
    }

    @Test("Bounded constraint clamps value above maximum")
    func boundedClampsAboveMax() {
        let constraint = ParameterConstraint.bounded(min: -1.0, max: 1.0)
        let result = constraint.apply(to: 5.0)
        #expect(abs(result - 1.0) < 1e-10, "Value above max should clamp to max (1.0)")
    }

    @Test("Bounded constraint preserves value within range")
    func boundedPreservesWithinRange() {
        let constraint = ParameterConstraint.bounded(min: -1.0, max: 1.0)
        let result = constraint.apply(to: 0.5)
        #expect(abs(result - 0.5) < 1e-10, "Value within range should be preserved")
    }

    @Test("Bounded constraint preserves value at minimum boundary")
    func boundedPreservesAtMin() {
        let constraint = ParameterConstraint.bounded(min: 0.0, max: 2.0)
        let result = constraint.apply(to: 0.0)
        #expect(abs(result - 0.0) < 1e-10, "Value at min boundary should be preserved")
    }

    @Test("Bounded constraint preserves value at maximum boundary")
    func boundedPreservesAtMax() {
        let constraint = ParameterConstraint.bounded(min: 0.0, max: 2.0)
        let result = constraint.apply(to: 2.0)
        #expect(abs(result - 2.0) < 1e-10, "Value at max boundary should be preserved")
    }

    @Test("NonNegative constraint returns zero for negative value")
    func nonNegativeReturnsZeroForNegative() {
        let constraint = ParameterConstraint.nonNegative
        let result = constraint.apply(to: -5.0)
        #expect(abs(result - 0.0) < 1e-10, "Negative value should become zero")
    }

    @Test("NonNegative constraint preserves positive value")
    func nonNegativePreservesPositive() {
        let constraint = ParameterConstraint.nonNegative
        let result = constraint.apply(to: 3.5)
        #expect(abs(result - 3.5) < 1e-10, "Positive value should be preserved")
    }

    @Test("NonNegative constraint preserves zero")
    func nonNegativePreservesZero() {
        let constraint = ParameterConstraint.nonNegative
        let result = constraint.apply(to: 0.0)
        #expect(abs(result - 0.0) < 1e-10, "Zero should be preserved")
    }

    @Test("Periodic constraint wraps value above period")
    func periodicWrapsAbovePeriod() {
        let twoPi = 2.0 * Double.pi
        let constraint = ParameterConstraint.periodic(period: twoPi)
        let result = constraint.apply(to: 3.0 * Double.pi)
        #expect(abs(result - Double.pi) < 1e-10, "3π should wrap to π with period 2π")
    }

    @Test("Periodic constraint wraps negative value")
    func periodicWrapsNegative() {
        let constraint = ParameterConstraint.periodic(period: 2.0)
        let result = constraint.apply(to: -0.5)
        #expect(abs(result - 1.5) < 1e-10, "-0.5 should wrap to 1.5 with period 2")
    }

    @Test("Periodic constraint preserves value within period")
    func periodicPreservesWithinPeriod() {
        let twoPi = 2.0 * Double.pi
        let constraint = ParameterConstraint.periodic(period: twoPi)
        let result = constraint.apply(to: Double.pi / 2.0)
        #expect(abs(result - Double.pi / 2.0) < 1e-10, "Value within period should be preserved")
    }

    @Test("Periodic constraint preserves zero")
    func periodicPreservesZero() {
        let constraint = ParameterConstraint.periodic(period: 1.0)
        let result = constraint.apply(to: 0.0)
        #expect(abs(result - 0.0) < 1e-10, "Zero should be preserved")
    }

    @Test("Periodic constraint with large period")
    func periodicWithLargePeriod() {
        let constraint = ParameterConstraint.periodic(period: 100.0)
        let result = constraint.apply(to: 250.0)
        #expect(abs(result - 50.0) < 1e-10, "250 should wrap to 50 with period 100")
    }

    @Test("Bounded constraint returns bounds tuple")
    func boundedReturnsBounds() {
        let constraint = ParameterConstraint.bounded(min: -2.0, max: 3.0)
        let bounds = constraint.bounds
        #expect(bounds != nil, "Bounded constraint should return bounds")
        #expect(abs(bounds!.min - -2.0) < 1e-10, "Min bound should be -2.0")
        #expect(abs(bounds!.max - 3.0) < 1e-10, "Max bound should be 3.0")
    }

    @Test("NonNegative constraint returns nil bounds")
    func nonNegativeReturnsNilBounds() {
        let constraint = ParameterConstraint.nonNegative
        let bounds = constraint.bounds
        #expect(bounds == nil, "NonNegative constraint should return nil bounds")
    }

    @Test("Periodic constraint returns nil bounds")
    func periodicReturnsNilBounds() {
        let constraint = ParameterConstraint.periodic(period: 2.0)
        let bounds = constraint.bounds
        #expect(bounds == nil, "Periodic constraint should return nil bounds")
    }

    @Test("MakeBounded creates valid constraint")
    func makeBoundedCreatesValidConstraint() {
        let constraint = ParameterConstraint.makeBounded(min: 0.0, max: 1.0)
        let bounds = constraint.bounds
        #expect(bounds != nil, "makeBounded should create bounded constraint")
        #expect(abs(bounds!.min - 0.0) < 1e-10, "Min should be 0.0")
        #expect(abs(bounds!.max - 1.0) < 1e-10, "Max should be 1.0")
    }

    @Test("MakePeriodic creates valid constraint")
    func makePeriodicCreatesValidConstraint() {
        let constraint = ParameterConstraint.makePeriodic(period: 2.0 * Double.pi)
        let result = constraint.apply(to: 3.0 * Double.pi)
        #expect(abs(result - Double.pi) < 1e-10, "makePeriodic should create working periodic constraint")
    }

    @Test("ParameterConstraint equality for bounded")
    func boundedEquality() {
        let constraint1 = ParameterConstraint.bounded(min: 0.0, max: 1.0)
        let constraint2 = ParameterConstraint.bounded(min: 0.0, max: 1.0)
        let constraint3 = ParameterConstraint.bounded(min: 0.0, max: 2.0)
        #expect(constraint1 == constraint2, "Same bounded constraints should be equal")
        #expect(constraint1 != constraint3, "Different bounded constraints should not be equal")
    }

    @Test("ParameterConstraint equality for nonNegative")
    func nonNegativeEquality() {
        let constraint1 = ParameterConstraint.nonNegative
        let constraint2 = ParameterConstraint.nonNegative
        #expect(constraint1 == constraint2, "NonNegative constraints should be equal")
    }

    @Test("ParameterConstraint equality for periodic")
    func periodicEquality() {
        let constraint1 = ParameterConstraint.periodic(period: 2.0)
        let constraint2 = ParameterConstraint.periodic(period: 2.0)
        let constraint3 = ParameterConstraint.periodic(period: 3.0)
        #expect(constraint1 == constraint2, "Same periodic constraints should be equal")
        #expect(constraint1 != constraint3, "Different periodic constraints should not be equal")
    }

    @Test("ParameterConstraint hashability")
    func constraintHashability() {
        let bounded = ParameterConstraint.bounded(min: 0.0, max: 1.0)
        let nonNeg = ParameterConstraint.nonNegative
        let periodic = ParameterConstraint.periodic(period: 2.0)

        var set = Set<ParameterConstraint>()
        set.insert(bounded)
        set.insert(nonNeg)
        set.insert(periodic)

        #expect(set.count == 3, "Three different constraint types should produce three set elements")
        #expect(set.contains(bounded), "Set should contain bounded constraint")
        #expect(set.contains(nonNeg), "Set should contain nonNegative constraint")
        #expect(set.contains(periodic), "Set should contain periodic constraint")
    }

    @Test("Different constraint types are not equal")
    func differentTypesNotEqual() {
        let bounded = ParameterConstraint.bounded(min: 0.0, max: 2.0)
        let nonNeg = ParameterConstraint.nonNegative
        let periodic = ParameterConstraint.periodic(period: 2.0)
        #expect(bounded != nonNeg, "Bounded should not equal nonNegative")
        #expect(bounded != periodic, "Bounded should not equal periodic")
        #expect(nonNeg != periodic, "NonNegative should not equal periodic")
    }
}
