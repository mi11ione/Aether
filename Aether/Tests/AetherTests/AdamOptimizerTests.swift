// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Test suite for AdamOptimizer creation and configuration.
/// Validates default parameter values, custom parameter acceptance,
/// and proper storage of all optimizer hyperparameters.
@Suite("AdamOptimizer Configuration")
struct AdamOptimizerConfigTests {
    @Test("Create Adam optimizer with defaults")
    func createWithDefaults() {
        let optimizer = AdamOptimizer()

        #expect(abs(optimizer.learningRate - 0.001) < 1e-10, "Default learning rate should be 0.001")
        #expect(abs(optimizer.beta1 - 0.9) < 1e-10, "Default beta1 should be 0.9")
        #expect(abs(optimizer.beta2 - 0.999) < 1e-10, "Default beta2 should be 0.999")
        #expect(abs(optimizer.epsilon - 1e-8) < 1e-10, "Default epsilon should be 1e-8")
    }

    @Test("Create Adam optimizer with custom values")
    func createWithCustomValues() {
        let optimizer = AdamOptimizer(
            learningRate: 0.01,
            beta1: 0.8,
            beta2: 0.99,
            epsilon: 1e-7,
            parameterShift: .pi / 4,
        )

        #expect(abs(optimizer.learningRate - 0.01) < 1e-10, "Custom learning rate should be 0.01")
        #expect(abs(optimizer.beta1 - 0.8) < 1e-10, "Custom beta1 should be 0.8")
        #expect(abs(optimizer.beta2 - 0.99) < 1e-10, "Custom beta2 should be 0.99")
        #expect(abs(optimizer.epsilon - 1e-7) < 1e-10, "Custom epsilon should be 1e-7")
        #expect(abs(optimizer.parameterShift - .pi / 4) < 1e-10, "Custom parameter shift should be pi/4")
    }

    @Test("Adam with zero beta1 reduces to RMSProp")
    func zeroBeta1Accepted() {
        let optimizer = AdamOptimizer(beta1: 0.0)
        #expect(abs(optimizer.beta1) < 1e-10, "Beta1 of 0 should be accepted for RMSProp mode")
    }
}

/// Test suite for AdamOptimizer minimization behavior.
/// Validates convergence on quadratic landscapes, adaptive learning,
/// and proper tracking of evaluations and termination conditions.
@Suite("AdamOptimizer Minimization")
struct AdamOptimizerMinimizeTests {
    @Test("Minimize quadratic function")
    func minimizeQuadratic() async {
        let optimizer = AdamOptimizer(learningRate: 0.1)

        let result = await optimizer.minimize(
            { params in params[0] * params[0] + params[1] * params[1] },
            from: [1.0, 1.0],
            using: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 200),
            progress: nil,
        )

        #expect(result.value < 0.1, "Adam should reduce quadratic objective significantly")
        #expect(result.evaluations > 0, "Evaluations should be tracked")
        #expect(result.iterations > 0, "At least one iteration should run")
    }

    @Test("Converges on 1D quadratic")
    func converges1D() async {
        let optimizer = AdamOptimizer(learningRate: 0.1)

        let result = await optimizer.minimize(
            { params in (params[0] - 2.0) * (params[0] - 2.0) },
            from: [0.0],
            using: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 300),
            progress: nil,
        )

        #expect(result.value < 0.1, "Should converge near minimum of shifted quadratic")
    }

    @Test("History records iterations")
    func historyRecorded() async {
        let optimizer = AdamOptimizer(learningRate: 0.1)

        let result = await optimizer.minimize(
            { params in params[0] * params[0] },
            from: [2.0],
            using: ConvergenceCriteria(energyTolerance: 1e-2, maxIterations: 50),
            progress: nil,
        )

        #expect(result.history.count > 1, "History should contain multiple entries")
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

        let optimizer = AdamOptimizer(learningRate: 0.1)
        let counter = Counter()

        _ = await optimizer.minimize(
            { params in params[0] * params[0] },
            from: [1.0],
            using: ConvergenceCriteria(energyTolerance: 1e-2, maxIterations: 20),
            progress: { _, _ in await counter.increment() },
        )

        let callCount = await counter.get()
        #expect(callCount > 0, "Progress callback should be invoked at least once")
    }

    @Test("Max iterations termination")
    func maxIterationsTermination() async {
        let optimizer = AdamOptimizer(learningRate: 0.0001)

        let result = await optimizer.minimize(
            { params in params[0] * params[0] + params[1] * params[1] },
            from: [10.0, 10.0],
            using: ConvergenceCriteria(energyTolerance: 1e-15, maxIterations: 5),
            progress: nil,
        )

        #expect(result.terminationReason == .maxIterationsReached, "Should hit max iterations with tiny learning rate")
        #expect(result.iterations == 5, "Should complete exactly 5 iterations")
    }

    @Test("Gradient convergence termination")
    func gradientConvergence() async {
        let optimizer = AdamOptimizer(learningRate: 0.5)

        let result = await optimizer.minimize(
            { params in params[0] * params[0] },
            from: [0.001],
            using: ConvergenceCriteria(
                energyTolerance: 1e-15,
                gradientNormTolerance: 1.0,
                maxIterations: 100,
            ),
            progress: nil,
        )

        #expect(
            result.terminationReason == .gradientConverged || result.terminationReason == .energyConverged,
            "Should converge near zero gradient",
        )
    }
}
