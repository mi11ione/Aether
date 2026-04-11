// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Test suite for RotosolveOptimizer configuration.
/// Validates default NFT mode setting and custom configuration
/// acceptance for the analytical per-parameter optimizer.
@Suite("RotosolveOptimizer Configuration")
struct RotosolveOptimizerConfigTests {
    @Test("Create Rotosolve optimizer with defaults")
    func createWithDefaults() {
        let optimizer = RotosolveOptimizer()
        #expect(!optimizer.isNFTEnabled, "NFT should be disabled by default")
    }

    @Test("Create Rotosolve optimizer with NFT enabled")
    func createWithNFTEnabled() {
        let optimizer = RotosolveOptimizer(isNFTEnabled: true)
        #expect(optimizer.isNFTEnabled, "NFT should be enabled when requested")
    }
}

/// Test suite for RotosolveOptimizer analytical minimization.
/// Validates sinusoidal parameter optimization convergence, sweep
/// behavior, and NFT two-parameter subspace extension correctness.
@Suite("RotosolveOptimizer Minimization")
struct RotosolveOptimizerMinimizeTests {
    @Test("Minimize sinusoidal objective")
    func minimizeSinusoidal() async {
        let optimizer = RotosolveOptimizer()

        let result = await optimizer.minimize(
            { params in sin(params[0]) + sin(params[1]) },
            from: [0.0, 0.0],
            using: ConvergenceCriteria(energyTolerance: 1e-6, maxIterations: 10),
            progress: nil,
        )

        #expect(result.value < -1.9, "Should find near-optimal sinusoidal minimum (-2.0)")
    }

    @Test("Minimize single parameter sinusoidal")
    func minimizeSingleParam() async {
        let optimizer = RotosolveOptimizer()

        let result = await optimizer.minimize(
            { params in sin(params[0]) },
            from: [0.0],
            using: ConvergenceCriteria(energyTolerance: 1e-6, maxIterations: 5),
            progress: nil,
        )

        #expect(result.value < -0.99, "Should find sin minimum near -1.0")
        let normalizedAngle = result.parameters[0].truncatingRemainder(dividingBy: 2.0 * .pi)
        let distToOptimum = min(
            abs(normalizedAngle + .pi / 2),
            abs(normalizedAngle + .pi / 2 - 2.0 * .pi),
            abs(normalizedAngle + .pi / 2 + 2.0 * .pi),
        )
        #expect(distToOptimum < 0.1, "Optimal angle should be near -pi/2 mod 2pi")
    }

    @Test("Minimize with NFT enabled")
    func minimizeWithNFT() async {
        let optimizer = RotosolveOptimizer(isNFTEnabled: true)

        let result = await optimizer.minimize(
            { params in sin(params[0]) + cos(params[1]) },
            from: [0.0, 0.0],
            using: ConvergenceCriteria(energyTolerance: 1e-6, maxIterations: 10),
            progress: nil,
        )

        #expect(result.value < -1.9, "NFT mode should also converge to near-minimum (-2.0)")
    }

    @Test("Convergence within few sweeps")
    func convergesQuickly() async {
        let optimizer = RotosolveOptimizer()

        let result = await optimizer.minimize(
            { params in 2.0 * sin(params[0]) + 3.0 * cos(params[1]) },
            from: [0.0, 0.0],
            using: ConvergenceCriteria(energyTolerance: 1e-8, maxIterations: 20),
            progress: nil,
        )

        #expect(result.terminationReason == .energyConverged, "Rotosolve should converge analytically in few sweeps")
        #expect(result.iterations <= 10, "Should converge in at most 10 sweeps")
    }

    @Test("Three evaluations per parameter per sweep")
    func evaluationCounting() async {
        let optimizer = RotosolveOptimizer()

        let result = await optimizer.minimize(
            { params in sin(params[0]) + sin(params[1]) + sin(params[2]) },
            from: [0.0, 0.0, 0.0],
            using: ConvergenceCriteria(energyTolerance: 1e-10, maxIterations: 2),
            progress: nil,
        )

        #expect(result.evaluations >= 25, "Should have at least 1 initial + 2 sweeps x 3 params x 4 evals")
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

        let optimizer = RotosolveOptimizer()
        let counter = Counter()

        _ = await optimizer.minimize(
            { params in sin(params[0]) },
            from: [0.0],
            using: ConvergenceCriteria(energyTolerance: 1e-6, maxIterations: 5),
            progress: { _, _ in await counter.increment() },
        )

        let callCount = await counter.get()
        #expect(callCount > 0, "Progress callback should be invoked")
    }

    @Test("Max iterations termination")
    func maxIterationsTermination() async {
        let optimizer = RotosolveOptimizer()

        let result = await optimizer.minimize(
            { params in sin(params[0]) + 0.001 * params[0] },
            from: [0.0],
            using: ConvergenceCriteria(energyTolerance: 1e-20, maxIterations: 1),
            progress: nil,
        )

        #expect(result.terminationReason == .maxIterationsReached, "Should hit max iterations with 1 sweep limit")
    }

    @Test("NFT with odd parameter count uses single-param fallback")
    func nftOddParameterCount() async {
        let optimizer = RotosolveOptimizer(isNFTEnabled: true)

        let result = await optimizer.minimize(
            { params in sin(params[0]) + sin(params[1]) + sin(params[2]) },
            from: [0.0, 0.0, 0.0],
            using: ConvergenceCriteria(energyTolerance: 1e-6, maxIterations: 10),
            progress: nil,
        )

        #expect(result.value < -2.9, "NFT with 3 params should optimize all including odd remainder")
    }
}
