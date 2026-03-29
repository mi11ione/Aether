// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Test suite for SLSQPOptimizer configuration and constraint types.
/// Validates default parameter values, custom configuration, and
/// correct OptimizationConstraint enum case construction.
@Suite("SLSQPOptimizer Configuration")
struct SLSQPOptimizerConfigTests {
    @Test("Create SLSQP optimizer with defaults")
    func createWithDefaults() {
        let optimizer = SLSQPOptimizer()

        #expect(abs(optimizer.tolerance - 1e-6) < 1e-10, "Default tolerance should be 1e-6")
        #expect(abs(optimizer.parameterShift - .pi / 2) < 1e-10, "Default parameter shift should be pi/2")
        #expect(optimizer.maxLineSearchSteps == 20, "Default max line search steps should be 20")
    }

    @Test("Create SLSQP optimizer with custom values")
    func createWithCustomValues() {
        let optimizer = SLSQPOptimizer(tolerance: 1e-8, parameterShift: 0.01, maxLineSearchSteps: 30)

        #expect(abs(optimizer.tolerance - 1e-8) < 1e-10, "Custom tolerance should be 1e-8")
        #expect(abs(optimizer.parameterShift - 0.01) < 1e-10, "Custom parameter shift should be 0.01")
        #expect(optimizer.maxLineSearchSteps == 30, "Custom max line search steps should be 30")
    }

    @Test("OptimizationConstraint equality creation")
    func equalityConstraintCreation() async {
        let constraint = OptimizationConstraint.equality { params in
            params[0] + params[1] - 1.0
        }

        if case let .equality(fn) = constraint {
            let val = await fn([0.3, 0.7])
            #expect(abs(val - 0.0) < 1e-10, "Equality constraint should evaluate correctly")
        }
    }

    @Test("OptimizationConstraint inequality creation")
    func inequalityConstraintCreation() async {
        let constraint = OptimizationConstraint.inequality { params in
            params[0]
        }

        if case let .inequality(fn) = constraint {
            let val = await fn([0.5])
            #expect(abs(val - 0.5) < 1e-10, "Inequality constraint should evaluate correctly")
        }
    }
}

/// Test suite for SLSQPOptimizer constrained and unconstrained minimization.
/// Validates convergence on unconstrained quadratics, equality-constrained
/// problems, inequality constraints, and mixed constraint scenarios.
@Suite("SLSQPOptimizer Minimization")
struct SLSQPOptimizerMinimizeTests {
    @Test("Minimize unconstrained quadratic")
    func minimizeUnconstrained() async {
        let optimizer = SLSQPOptimizer(tolerance: 1e-4, parameterShift: 0.01)

        let result = await optimizer.minimize(
            { params in params[0] * params[0] + params[1] * params[1] },
            from: [1.0, 1.0],
            using: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 50),
            progress: nil,
        )

        #expect(result.value < 0.5, "SLSQP should reduce unconstrained quadratic")
    }

    @Test("Minimize with equality constraint")
    func minimizeWithEquality() async {
        let optimizer = SLSQPOptimizer(tolerance: 1e-3, parameterShift: 0.01)

        let constraints: [OptimizationConstraint] = [
            .equality { params in params[0] + params[1] - 1.0 },
        ]

        let result = await optimizer.minimize(
            { params in params[0] * params[0] + params[1] * params[1] },
            from: [0.8, 0.2],
            using: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 100),
            constraints: constraints,
            progress: nil,
        )

        let constraintViolation = abs(result.parameters[0] + result.parameters[1] - 1.0)
        #expect(constraintViolation < 0.1, "Equality constraint should be approximately satisfied")
    }

    @Test("Minimize with inequality constraint")
    func minimizeWithInequality() async {
        let optimizer = SLSQPOptimizer(tolerance: 1e-3, parameterShift: 0.01)

        let constraints: [OptimizationConstraint] = [
            .inequality { params in params[0] - 0.5 },
        ]

        let result = await optimizer.minimize(
            { params in (params[0] - 0.2) * (params[0] - 0.2) },
            from: [1.0],
            using: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 100),
            constraints: constraints,
            progress: nil,
        )

        #expect(result.parameters[0] >= 0.4, "Inequality constraint x >= 0.5 should be approximately respected")
    }

    @Test("Unconstrained through Optimizer protocol")
    func unconstrainedProtocol() async {
        let optimizer: any Optimizer = SLSQPOptimizer(tolerance: 1e-4, parameterShift: 0.01)

        let result = await optimizer.minimize(
            { params in params[0] * params[0] },
            from: [2.0],
            using: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 50),
            progress: nil,
        )

        #expect(result.value < 1.0, "Should optimize through protocol conformance")
    }

    @Test("Evaluations tracked with constraints")
    func evaluationsTrackedConstrained() async {
        let optimizer = SLSQPOptimizer(tolerance: 1e-3, parameterShift: 0.01)

        let constraints: [OptimizationConstraint] = [
            .equality { params in params[0] + params[1] - 1.0 },
        ]

        let result = await optimizer.minimize(
            { params in params[0] * params[0] + params[1] * params[1] },
            from: [0.8, 0.2],
            using: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 30),
            constraints: constraints,
            progress: nil,
        )

        #expect(result.evaluations > result.iterations, "Should track extra evaluations for gradient and constraint evaluation")
    }

    @Test("Progress callback invoked")
    func progressCallbackInvoked() async {
        actor Counter {
            var count = 0
            func increment() {
                count += 1
            }

            func get() -> Int {
                count
            }
        }

        let optimizer = SLSQPOptimizer(tolerance: 1e-3, parameterShift: 0.01)
        let counter = Counter()

        _ = await optimizer.minimize(
            { params in params[0] * params[0] },
            from: [1.0],
            using: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 10),
            progress: { _, _ in await counter.increment() },
        )

        let callCount = await counter.get()
        #expect(callCount > 0, "Progress callback should be invoked")
    }

    @Test("Max iterations termination")
    func maxIterationsTermination() async {
        let optimizer = SLSQPOptimizer(tolerance: 1e-20, parameterShift: 0.01)

        let result = await optimizer.minimize(
            { params in params[0] * params[0] + params[1] * params[1] },
            from: [10.0, 10.0],
            using: ConvergenceCriteria(energyTolerance: 1e-20, maxIterations: 1),
            progress: nil,
        )

        #expect(result.terminationReason == .maxIterationsReached, "Should hit max iterations with 1 iteration limit")
    }

    @Test("Large multiplier triggers penalty update")
    func largMultiplierPenaltyUpdate() async {
        let optimizer = SLSQPOptimizer(tolerance: 1e-10, parameterShift: 0.01)

        let constraints: [OptimizationConstraint] = [
            .equality { params in params[0] * params[0] + params[1] * params[1] - 1.0 },
        ]

        let result = await optimizer.minimize(
            { params in (params[0] - 2.0) * (params[0] - 2.0) + params[1] * params[1] },
            from: [5.0, 5.0],
            using: ConvergenceCriteria(energyTolerance: 1e-10, maxIterations: 15),
            constraints: constraints,
            progress: nil,
        )

        #expect(result.iterations >= 2, "Should run multiple iterations with nonlinear constraint")
    }

    @Test("Line search fallback on steep objective")
    func lineSearchFallback() async {
        let optimizer = SLSQPOptimizer(tolerance: 1e-3, parameterShift: 0.01, maxLineSearchSteps: 1)

        let result = await optimizer.minimize(
            { params in exp(params[0]) + exp(params[1]) },
            from: [5.0, 5.0],
            using: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 10),
            progress: nil,
        )

        #expect(result.iterations > 0, "Should complete even when line search falls back")
    }

    @Test("Steepest descent fallback at optimum")
    func steepestDescentFallbackAtOptimum() async {
        let optimizer = SLSQPOptimizer(tolerance: 1e-3, parameterShift: 0.01)

        let result = await optimizer.minimize(
            { params in params[0] * params[0] },
            from: [0.0001],
            using: ConvergenceCriteria(energyTolerance: 1e-1, maxIterations: 50),
            progress: nil,
        )

        #expect(result.value < 0.1, "Should handle near-optimal start gracefully")
    }
}
