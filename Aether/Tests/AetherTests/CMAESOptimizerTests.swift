// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Test suite for CMAESOptimizer configuration and population sizing.
/// Validates default step size, auto-population selection, custom
/// population size acceptance, and parameter validation.
@Suite("CMAESOptimizer Configuration")
struct CMAESOptimizerConfigTests {
    @Test("Create CMA-ES optimizer with defaults")
    func createWithDefaults() {
        let optimizer = CMAESOptimizer()

        #expect(optimizer.populationSize == nil, "Default population size should be nil (auto)")
        #expect(abs(optimizer.initialStepSize - 0.5) < 1e-10, "Default initial step size should be 0.5")
    }

    @Test("Create CMA-ES optimizer with custom population")
    func createWithCustomPopulation() {
        let optimizer = CMAESOptimizer(populationSize: 20, initialStepSize: 0.3)

        #expect(optimizer.populationSize == 20, "Custom population size should be 20")
        #expect(abs(optimizer.initialStepSize - 0.3) < 1e-10, "Custom initial step size should be 0.3")
    }
}

/// Test suite for CMAESOptimizer population-based minimization.
/// Validates convergence on quadratic landscapes, population evaluation
/// counting, and covariance adaptation across generations.
@Suite("CMAESOptimizer Minimization")
struct CMAESOptimizerMinimizeTests {
    @Test("Minimize 2D quadratic function")
    func minimizeQuadratic() async {
        let optimizer = CMAESOptimizer(populationSize: 10, initialStepSize: 1.0)

        let result = await optimizer.minimize(
            { params in params[0] * params[0] + params[1] * params[1] },
            from: [3.0, 4.0],
            using: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100),
            progress: nil,
        )

        #expect(result.value < 5.0, "CMA-ES should reduce quadratic objective from initial 25")
    }

    @Test("Population evaluations tracked correctly")
    func populationEvaluationsTracked() async {
        let popSize = 8
        let optimizer = CMAESOptimizer(populationSize: popSize, initialStepSize: 0.5)

        let result = await optimizer.minimize(
            { params in params[0] * params[0] },
            from: [1.0],
            using: ConvergenceCriteria(energyTolerance: 1e-10, maxIterations: 3),
            progress: nil,
        )

        #expect(result.evaluations >= popSize, "Should evaluate at least one full population")
    }

    @Test("Auto population sizing")
    func autoPopulationSizing() async {
        let optimizer = CMAESOptimizer(initialStepSize: 1.0)

        let result = await optimizer.minimize(
            { params in params[0] * params[0] + params[1] * params[1] + params[2] * params[2] },
            from: [1.0, 1.0, 1.0],
            using: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 50),
            progress: nil,
        )

        #expect(result.evaluations > 0, "Should evaluate with auto-selected population size")
        #expect(result.iterations > 0, "Should complete at least one generation")
    }

    @Test("Minimize shifted Rosenbrock-like function")
    func minimizeShifted() async {
        let optimizer = CMAESOptimizer(populationSize: 12, initialStepSize: 0.5)

        let result = await optimizer.minimize(
            { params in (params[0] - 1.0) * (params[0] - 1.0) + 2.0 * (params[1] - 1.0) * (params[1] - 1.0) },
            from: [3.0, 3.0],
            using: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100),
            progress: nil,
        )

        #expect(result.value < 5.0, "Should reduce shifted quadratic from initial ~12")
    }

    @Test("Progress callback invoked per generation")
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

        let optimizer = CMAESOptimizer(populationSize: 6, initialStepSize: 0.5)
        let counter = Counter()

        _ = await optimizer.minimize(
            { params in params[0] * params[0] },
            from: [1.0],
            using: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 10),
            progress: { _, _ in await counter.increment() },
        )

        let callCount = await counter.get()
        #expect(callCount > 0, "Progress callback should be invoked per generation")
    }

    @Test("Max iterations termination")
    func maxIterationsTermination() async {
        let optimizer = CMAESOptimizer(populationSize: 6, initialStepSize: 1.0)

        let result = await optimizer.minimize(
            { params in params[0] * params[0] + params[1] * params[1] },
            from: [10.0, 10.0],
            using: ConvergenceCriteria(energyTolerance: 1e-30, maxIterations: 3),
            progress: nil,
        )

        #expect(result.terminationReason == .maxIterationsReached, "Should hit max iterations with unreachable energy tolerance")
    }

    @Test("Sigma convergence on near-optimal start")
    func sigmaConvergence() async {
        let optimizer = CMAESOptimizer(populationSize: 6, initialStepSize: 1e-14)

        let result = await optimizer.minimize(
            { params in params[0] * params[0] },
            from: [0.0],
            using: ConvergenceCriteria(energyTolerance: 1e-20, maxIterations: 200),
            progress: nil,
        )

        #expect(result.terminationReason == .energyConverged, "Should converge via sigma collapse")
    }
}
