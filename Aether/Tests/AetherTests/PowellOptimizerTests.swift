// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Test suite for PowellOptimizer configuration and parameter storage.
/// Validates default line search tolerance, iteration limits, and
/// correct propagation of custom configuration values.
@Suite("PowellOptimizer Configuration")
struct PowellOptimizerConfigTests {
    @Test("Create Powell optimizer with defaults")
    func createWithDefaults() {
        let optimizer = PowellOptimizer()

        #expect(abs(optimizer.lineSearchTolerance - 1e-8) < 1e-10, "Default line search tolerance should be 1e-8")
        #expect(optimizer.maxLineSearchIterations == 100, "Default max line search iterations should be 100")
    }

    @Test("Create Powell optimizer with custom values")
    func createWithCustomValues() {
        let optimizer = PowellOptimizer(lineSearchTolerance: 1e-6, maxLineSearchIterations: 50)

        #expect(abs(optimizer.lineSearchTolerance - 1e-6) < 1e-10, "Custom tolerance should be 1e-6")
        #expect(optimizer.maxLineSearchIterations == 50, "Custom max iterations should be 50")
    }
}

/// Test suite for PowellOptimizer conjugate direction minimization.
/// Validates derivative-free convergence on quadratic and separable
/// objectives with proper direction update and line search behavior.
@Suite("PowellOptimizer Minimization")
struct PowellOptimizerMinimizeTests {
    @Test("Minimize 2D quadratic function")
    func minimizeQuadratic() async {
        let optimizer = PowellOptimizer()

        let result = await optimizer.minimize(
            { params in params[0] * params[0] + params[1] * params[1] },
            from: [2.0, 3.0],
            using: ConvergenceCriteria(energyTolerance: 1e-6, maxIterations: 50),
            progress: nil,
        )

        #expect(result.value < 0.01, "Powell should minimize quadratic near zero")
        #expect(abs(result.parameters[0]) < 0.1, "x should be near zero at minimum")
        #expect(abs(result.parameters[1]) < 0.1, "y should be near zero at minimum")
    }

    @Test("Minimize shifted quadratic")
    func minimizeShiftedQuadratic() async {
        let optimizer = PowellOptimizer()

        let result = await optimizer.minimize(
            { params in (params[0] - 1.0) * (params[0] - 1.0) + (params[1] + 2.0) * (params[1] + 2.0) },
            from: [0.0, 0.0],
            using: ConvergenceCriteria(energyTolerance: 1e-6, maxIterations: 50),
            progress: nil,
        )

        #expect(result.value < 1e-4, "Should find minimum of shifted quadratic")
        #expect(abs(result.parameters[0] - 1.0) < 0.01, "x should be near 1.0")
        #expect(abs(result.parameters[1] + 2.0) < 0.01, "y should be near -2.0")
    }

    @Test("Minimize 1D function")
    func minimize1D() async {
        let optimizer = PowellOptimizer()

        let result = await optimizer.minimize(
            { params in (params[0] - 3.0) * (params[0] - 3.0) },
            from: [0.0],
            using: ConvergenceCriteria(energyTolerance: 1e-6, maxIterations: 50),
            progress: nil,
        )

        #expect(abs(result.parameters[0] - 3.0) < 0.01, "Should find x=3 minimum")
    }

    @Test("Convergence reported correctly")
    func convergenceReported() async {
        let optimizer = PowellOptimizer()

        let result = await optimizer.minimize(
            { params in params[0] * params[0] },
            from: [1.0],
            using: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100),
            progress: nil,
        )

        #expect(result.terminationReason == .energyConverged, "Should converge by energy tolerance")
        #expect(result.evaluations > result.iterations, "Should have more evaluations than iterations due to line search")
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

        let optimizer = PowellOptimizer()
        let counter = Counter()

        _ = await optimizer.minimize(
            { params in params[0] * params[0] },
            from: [1.0],
            using: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 20),
            progress: { _, _ in await counter.increment() },
        )

        let callCount = await counter.get()
        #expect(callCount > 0, "Progress callback should be invoked")
    }

    @Test("Max iterations termination")
    func maxIterationsTermination() async {
        let optimizer = PowellOptimizer()

        let result = await optimizer.minimize(
            { params in params[0] * params[0] + params[1] * params[1] },
            from: [10.0, 10.0],
            using: ConvergenceCriteria(energyTolerance: 1e-20, maxIterations: 1),
            progress: nil,
        )

        #expect(result.terminationReason == .maxIterationsReached, "Should hit max iterations with 1 iteration limit")
    }
}
