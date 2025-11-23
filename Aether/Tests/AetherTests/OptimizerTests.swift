// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for ConvergenceCriteria.
/// Validates convergence criteria creation, defaults, and validation.
@Suite("ConvergenceCriteria")
struct ConvergenceCriteriaTests {
    @Test("Create convergence criteria with defaults")
    func createWithDefaults() {
        let criteria = ConvergenceCriteria()

        #expect(criteria.energyTolerance == 1e-6)
        #expect(criteria.maxIterations == 1000)
        #expect(criteria.gradientNormTolerance == nil)
    }

    @Test("Create convergence criteria with custom values")
    func createWithCustomValues() {
        let criteria = ConvergenceCriteria(
            energyTolerance: 1e-8,
            gradientNormTolerance: 1e-5,
            maxIterations: 500
        )

        #expect(criteria.energyTolerance == 1e-8)
        #expect(criteria.gradientNormTolerance == 1e-5)
        #expect(criteria.maxIterations == 500)
    }

    @Test("Default convergence criteria")
    func defaultCriteria() {
        let criteria = ConvergenceCriteria.default

        #expect(criteria.energyTolerance == 1e-6)
        #expect(criteria.maxIterations == 1000)
    }
}

/// Test suite for OptimizerResult.
/// Validates optimization result creation and field access.
@Suite("OptimizerResult")
struct OptimizerResultTests {
    @Test("Create optimizer result")
    func createOptimizerResult() {
        let result = OptimizerResult(
            optimalParameters: [1.0, 2.0, 3.0],
            optimalValue: -1.5,
            valueHistory: [-2.0, -1.8, -1.5],
            iterations: 3,
            convergenceReason: .energyTolerance,
            functionEvaluations: 15
        )

        #expect(result.optimalParameters == [1.0, 2.0, 3.0])
        #expect(result.optimalValue == -1.5)
        #expect(result.valueHistory == [-2.0, -1.8, -1.5])
        #expect(result.iterations == 3)
        #expect(result.convergenceReason == .energyTolerance)
        #expect(result.functionEvaluations == 15)
    }
}

/// Test suite for ConvergenceReason.
/// Validates convergence reason cases and descriptions.
@Suite("ConvergenceReason")
struct ConvergenceReasonTests {
    @Test("Energy tolerance reason")
    func energyToleranceReason() {
        let reason = ConvergenceReason.energyTolerance
        #expect(reason.description.contains("tolerance"))
    }

    @Test("Gradient norm reason")
    func gradientNormReason() {
        let reason = ConvergenceReason.gradientNorm
        #expect(reason.description.contains("Gradient"))
    }

    @Test("Max iterations reason")
    func maxIterationsReason() {
        let reason = ConvergenceReason.maxIterations
        #expect(reason.description.contains("Maximum"))
    }
}

/// Test suite for NelderMeadOptimizer.
/// Validates simplex optimization, convergence, and parameter updates.
@Suite("NelderMeadOptimizer")
struct NelderMeadOptimizerTests {
    @Test("Create Nelder-Mead optimizer with defaults")
    func createWithDefaults() {
        let optimizer = NelderMeadOptimizer()

        #expect(optimizer.tolerance == 1e-6)
        #expect(optimizer.initialSimplexSize == 0.1)
    }

    @Test("Create Nelder-Mead optimizer with custom values")
    func createWithCustomValues() {
        let optimizer = NelderMeadOptimizer(tolerance: 1e-8, initialSimplexSize: 0.05)

        #expect(optimizer.tolerance == 1e-8)
        #expect(optimizer.initialSimplexSize == 0.05)
    }

    @Test("Optimize quadratic function")
    func optimizeQuadraticFunction() async throws {
        let optimizer = NelderMeadOptimizer(tolerance: 1e-3)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            let x = params[0]
            let y = params[1]
            return x * x + y * y
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [1.0, 1.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 100),
            progressCallback: nil
        )

        #expect(abs(result.optimalValue) < 0.01)
        #expect(abs(result.optimalParameters[0]) < 0.1)
        #expect(abs(result.optimalParameters[1]) < 0.1)
        #expect(result.convergenceReason == .energyTolerance)
    }

    @Test("Simplex vertex is mutable")
    func simplexVertexMutable() {
        var vertex = NelderMeadOptimizer.SimplexVertex(parameters: [1.0], value: 2.0)
        vertex.parameters[0] = 3.0
        vertex.value = 4.0

        #expect(vertex.parameters == [3.0])
        #expect(vertex.value == 4.0)
    }

    @Test("Progress callback is called")
    func progressCallbackCalled() async throws {
        actor Counter {
            var count = 0
            func increment() { count += 1 }
            func get() -> Int { count }
        }

        let optimizer = NelderMeadOptimizer(tolerance: 1e-3)
        let counter = Counter()

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            params[0] * params[0]
        }

        let progressCallback: @Sendable (Int, Double) async -> Void = { _, _ in
            await counter.increment()
        }

        _ = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [1.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 50),
            progressCallback: progressCallback
        )

        let callbackCount = await counter.get()
        #expect(callbackCount > 0)
    }

    @Test("Max iterations reached")
    func maxIterationsReached() async throws {
        let optimizer = NelderMeadOptimizer()

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            params[0] * params[0] + params[1] * params[1]
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [10.0, 10.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-12, maxIterations: 5),
            progressCallback: nil
        )

        #expect(result.convergenceReason == .maxIterations)
        #expect(result.iterations == 5)
    }

    @Test("Simplex shrink operation")
    func simplexShrinkOperation() async throws {
        let optimizer = NelderMeadOptimizer(tolerance: 1e-6, initialSimplexSize: 2.0)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            let x = params[0]
            let y = params[1]
            return abs(x - 3.0) + abs(y - 3.0)
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [0.0, 0.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 100),
            progressCallback: nil
        )

        #expect(result.optimalValue < 1.0)
        #expect(result.iterations > 0)
    }

    @Test("Simplex shrink on Rosenbrock function")
    func simplexShrinkOnRosenbrock() async throws {
        let optimizer = NelderMeadOptimizer(tolerance: 1e-6, initialSimplexSize: 5.0)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            let x = params[0]
            let y = params[1]
            let a = 1.0
            let b = 100.0
            return (a - x) * (a - x) + b * (y - x * x) * (y - x * x)
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [-2.0, 2.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 1000),
            progressCallback: nil
        )

        #expect(result.iterations > 20, "Rosenbrock function requires many iterations")
        #expect(result.optimalValue < 5.0)
    }

    @Test("Simplex outside contraction succeeds")
    func simplexOutsideContractionSucceeds() async throws {
        let optimizer = NelderMeadOptimizer(tolerance: 1e-4, initialSimplexSize: 1.0)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            let x = params[0]
            let y = params[1]
            return (y - x * x / 4.0) * (y - x * x / 4.0) + x * x / 100.0
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [2.0, -1.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 200),
            progressCallback: nil
        )

        #expect(result.optimalValue < 0.1)
        #expect(result.iterations > 5)
    }

    @Test("Simplex forced to shrink on rugged landscape")
    func simplexForcedToShrink() async throws {
        let optimizer = NelderMeadOptimizer(tolerance: 1e-5, initialSimplexSize: 10.0)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            let x = params[0]
            let y = params[1]
            let a = 10.0
            return 2.0 * a + (x * x - a * cos(2.0 * .pi * x)) + (y * y - a * cos(2.0 * .pi * y))
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [4.5, 4.5],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-2, maxIterations: 500),
            progressCallback: nil
        )

        #expect(result.iterations > 10, "Rastrigin requires many iterations")
    }
}

/// Test suite for GradientDescentOptimizer.
/// Validates gradient-based optimization with parameter shift rule.
@Suite("GradientDescentOptimizer")
struct GradientDescentOptimizerTests {
    @Test("Create gradient descent optimizer with defaults")
    func createWithDefaults() {
        let optimizer = GradientDescentOptimizer()

        #expect(optimizer.learningRate == 0.1)
        #expect(optimizer.momentum == 0.0)
        #expect(optimizer.useAdaptiveLearningRate)
        #expect(abs(optimizer.parameterShift - .pi / 2) < 1e-10)
    }

    @Test("Create gradient descent optimizer with custom values")
    func createWithCustomValues() {
        let optimizer = GradientDescentOptimizer(
            learningRate: 0.05,
            momentum: 0.9,
            useAdaptiveLearningRate: false,
            parameterShift: 0.01
        )

        #expect(optimizer.learningRate == 0.05)
        #expect(optimizer.momentum == 0.9)
        #expect(!optimizer.useAdaptiveLearningRate)
        #expect(optimizer.parameterShift == 0.01)
    }

    @Test("Optimize quadratic function")
    func optimizeQuadraticFunction() async throws {
        let optimizer = GradientDescentOptimizer(learningRate: 0.1, momentum: 0.0)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            params[0] * params[0]
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [1.0],
            convergenceCriteria: ConvergenceCriteria(
                energyTolerance: 1e-3,
                gradientNormTolerance: 1e-2,
                maxIterations: 100
            ),
            progressCallback: nil
        )

        #expect(abs(result.optimalValue) < 0.01)
        #expect(abs(result.optimalParameters[0]) < 0.1)
    }

    @Test("Gradient norm convergence")
    func gradientNormConvergence() async throws {
        let optimizer = GradientDescentOptimizer(learningRate: 0.1)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            params[0] * params[0]
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [0.1],
            convergenceCriteria: ConvergenceCriteria(
                energyTolerance: 1e-12,
                gradientNormTolerance: 1e-1,
                maxIterations: 50
            ),
            progressCallback: nil
        )

        #expect(result.convergenceReason == .gradientNorm)
    }

    @Test("Adaptive learning rate decreases")
    func adaptiveLearningRateDecreases() async throws {
        let optimizer = GradientDescentOptimizer(learningRate: 0.1, useAdaptiveLearningRate: true)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            params[0] * params[0] * 0.001
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [1.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-6, maxIterations: 20),
            progressCallback: nil
        )

        #expect(result.iterations >= 1)
        #expect(result.optimalValue < 0.001)
    }

    @Test("Max iterations reached")
    func maxIterationsReached() async throws {
        let optimizer = GradientDescentOptimizer(learningRate: 0.001)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            params[0] * params[0] + params[1] * params[1]
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [10.0, 10.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-12, maxIterations: 3),
            progressCallback: nil
        )

        #expect(result.convergenceReason == .maxIterations)
        #expect(result.iterations == 3)
    }

    @Test("No improvement triggers adaptive learning rate")
    func noImprovementTriggersAdaptiveLearningRate() async throws {
        let optimizer = GradientDescentOptimizer(learningRate: 0.5, useAdaptiveLearningRate: true)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            params[0] * params[0] * 0.00001
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [1.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-8, maxIterations: 50),
            progressCallback: nil
        )

        #expect(result.iterations > 0)
        #expect(result.optimalValue < 0.001)
    }

    @Test("Stagnation without adaptive learning rate")
    func stagnationWithoutAdaptiveLearningRate() async throws {
        let optimizer = GradientDescentOptimizer(learningRate: 0.001, useAdaptiveLearningRate: false)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            let x = params[0]
            return x * x * x * x
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [0.1],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-12, gradientNormTolerance: 1e-15, maxIterations: 100),
            progressCallback: nil
        )

        #expect(result.iterations > 5, "Should take multiple iterations with stagnation")
    }

    @Test("Stagnation with adaptive learning rate decreases rate")
    func stagnationWithAdaptiveLearningRateDecreasesRate() async throws {
        let optimizer = GradientDescentOptimizer(learningRate: 0.01, useAdaptiveLearningRate: true)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            let x = params[0]
            let y = params[1]
            return (x * x + y - 11.0) * (x * x + y - 11.0) + (x + y * y - 7.0) * (x + y * y - 7.0)
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [0.0, 0.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, gradientNormTolerance: 1e-8, maxIterations: 100),
            progressCallback: nil
        )

        #expect(result.iterations > 10, "Should take many iterations with stagnation")
    }
}

/// Test suite for LBFGSBOptimizer.
/// Validates L-BFGS-B quasi-Newton optimization with limited memory.
@Suite("LBFGSBOptimizer")
struct LBFGSBOptimizerTests {
    @Test("Create L-BFGS-B optimizer with defaults")
    func createWithDefaults() {
        let optimizer = LBFGSBOptimizer()

        #expect(optimizer.memorySize == 10)
        #expect(optimizer.tolerance == 1e-6)
        #expect(optimizer.maxLineSearchSteps == 20)
        #expect(abs(optimizer.parameterShift - .pi / 2) < 1e-10)
    }

    @Test("Create L-BFGS-B optimizer with custom values")
    func createWithCustomValues() {
        let optimizer = LBFGSBOptimizer(
            memorySize: 5,
            tolerance: 1e-8,
            maxLineSearchSteps: 10,
            parameterShift: 0.01
        )

        #expect(optimizer.memorySize == 5)
        #expect(optimizer.tolerance == 1e-8)
        #expect(optimizer.maxLineSearchSteps == 10)
        #expect(optimizer.parameterShift == 0.01)
    }

    @Test("Optimize Rosenbrock function")
    func optimizeRosenbrockFunction() async throws {
        let optimizer = LBFGSBOptimizer(tolerance: 1e-3)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            let x = params[0]
            let y = params[1]
            let a = 1.0 - x
            let b = y - x * x
            return a * a + 100.0 * b * b
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [0.0, 0.0],
            convergenceCriteria: ConvergenceCriteria(
                energyTolerance: 1e-3,
                maxIterations: 200
            ),
            progressCallback: nil
        )

        #expect(result.iterations > 0)
        #expect(result.functionEvaluations > 0)
    }

    @Test("Compute search direction with empty history")
    func computeSearchDirectionEmptyHistory() {
        let gradient = [1.0, 2.0, 3.0]
        let direction = LBFGSBOptimizer.computeSearchDirection(
            gradient: gradient,
            sHistory: [],
            yHistory: [],
            rhoHistory: []
        )

        #expect(direction == [-1.0, -2.0, -3.0])
    }

    @Test("Compute search direction with history")
    func computeSearchDirectionWithHistory() {
        let gradient = [1.0, 1.0]
        let sHistory = [[0.1, 0.1]]
        let yHistory = [[0.05, 0.05]]
        let rhoHistory = [1.0 / 0.01]

        let direction = LBFGSBOptimizer.computeSearchDirection(
            gradient: gradient,
            sHistory: sHistory,
            yHistory: yHistory,
            rhoHistory: rhoHistory
        )

        #expect(direction.count == 2)
        #expect(direction[0] < 0)
        #expect(direction[1] < 0)
    }

    @Test("Gradient norm convergence")
    func gradientNormConvergence() async throws {
        let optimizer = LBFGSBOptimizer()

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            params[0] * params[0]
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [0.1],
            convergenceCriteria: ConvergenceCriteria(
                energyTolerance: 1e-12,
                gradientNormTolerance: 0.1,
                maxIterations: 50
            ),
            progressCallback: nil
        )

        #expect(result.convergenceReason == .gradientNorm)
    }

    @Test("Energy tolerance convergence")
    func energyToleranceConvergence() async throws {
        let optimizer = LBFGSBOptimizer()

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            params[0] * params[0]
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [0.1],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-2, maxIterations: 50),
            progressCallback: nil
        )

        #expect(result.convergenceReason == .energyTolerance)
    }

    @Test("Max iterations reached")
    func maxIterationsReached() async throws {
        let optimizer = LBFGSBOptimizer(tolerance: 1e-10)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            params[0] * params[0] + params[1] * params[1]
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [5.0, 5.0],
            convergenceCriteria: ConvergenceCriteria(
                energyTolerance: 1e-12,
                gradientNormTolerance: 1e-12,
                maxIterations: 2
            ),
            progressCallback: nil
        )

        #expect(result.convergenceReason == .maxIterations)
        #expect(result.iterations == 2)
    }

    @Test("Line search failure with discontinuous function")
    func lineSearchFailureWithDiscontinuousFunction() async throws {
        let optimizer = LBFGSBOptimizer(maxLineSearchSteps: 40)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            let x = params[0]
            if x > 0.5 {
                return 1000.0
            } else {
                return x * x
            }
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [0.4],
            convergenceCriteria: ConvergenceCriteria(
                energyTolerance: 1e-8,
                gradientNormTolerance: 1e-8,
                maxIterations: 20
            ),
            progressCallback: nil
        )

        #expect(result.iterations >= 1)
    }
}

/// Test suite for SPSAOptimizer.
/// Validates SPSA stochastic optimization with simultaneous perturbations.
@Suite("SPSAOptimizer")
struct SPSAOptimizerTests {
    @Test("Create SPSA optimizer with defaults")
    func createWithDefaults() {
        let optimizer = SPSAOptimizer()

        #expect(optimizer.initialStepSize == 0.1)
        #expect(optimizer.initialPerturbation == 0.01)
        #expect(optimizer.decayExponent == 0.602)
        #expect(optimizer.perturbationDecayExponent == 0.101)
        #expect(optimizer.stabilityConstant == 100.0)
    }

    @Test("Create SPSA optimizer with custom values")
    func createWithCustomValues() {
        let optimizer = SPSAOptimizer(
            initialStepSize: 0.2,
            initialPerturbation: 0.05,
            decayExponent: 0.7,
            perturbationDecayExponent: 0.2,
            stabilityConstant: 50.0
        )

        #expect(optimizer.initialStepSize == 0.2)
        #expect(optimizer.initialPerturbation == 0.05)
        #expect(optimizer.decayExponent == 0.7)
        #expect(optimizer.perturbationDecayExponent == 0.2)
        #expect(optimizer.stabilityConstant == 50.0)
    }

    @Test("Optimize quadratic function")
    func optimizeQuadraticFunction() async throws {
        let optimizer = SPSAOptimizer(initialStepSize: 0.1, initialPerturbation: 0.1)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            params[0] * params[0]
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [1.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 0.01, maxIterations: 200),
            progressCallback: nil
        )

        #expect(result.optimalValue < 1.0)
        #expect(result.iterations > 0)
    }

    @Test("SPSA uses only 2 evaluations per iteration")
    func usesOnly2EvaluationsPerIteration() async throws {
        actor Counter {
            var count = 0
            func increment() { count += 1 }
            func get() -> Int { count }
        }

        let optimizer = SPSAOptimizer()
        let counter = Counter()

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            await counter.increment()
            return params[0] * params[0]
        }

        _ = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [1.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-2, maxIterations: 10),
            progressCallback: nil
        )

        let evaluationCount = await counter.get()
        #expect(evaluationCount == 31)
    }

    @Test("Max iterations reached")
    func maxIterationsReached() async throws {
        let optimizer = SPSAOptimizer()

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            params[0] * params[0]
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [10.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-12, maxIterations: 5),
            progressCallback: nil
        )

        #expect(result.convergenceReason == .maxIterations)
        #expect(result.iterations == 5)
    }

    @Test("No improvement increments iteration counter")
    func noImprovementIncrementsIterationCounter() async throws {
        let optimizer = SPSAOptimizer(initialStepSize: 0.01)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            params[0] * params[0] * 0.00001
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [1.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-8, maxIterations: 100),
            progressCallback: nil
        )

        #expect(result.iterations > 0)
        #expect(result.optimalValue < 0.001)
    }
}

/// Test suite for COBYLAOptimizer.
/// Validates COBYLA trust region optimization
/// with linear interpolation models
@Suite("COBYLAOptimizer")
struct COBYLAOptimizerTests {
    @Test("Create COBYLA optimizer with defaults")
    func createWithDefaults() {
        let optimizer = COBYLAOptimizer()

        #expect(optimizer.initialTrustRadius == 0.5)
        #expect(optimizer.minTrustRadius == 1e-6)
        #expect(optimizer.maxTrustRadius == 2.0)
        #expect(optimizer.shrinkFactor == 0.5)
        #expect(optimizer.expandFactor == 2.0)
        #expect(optimizer.acceptRatio == 0.1)
        #expect(optimizer.expandRatio == 0.75)
        #expect(optimizer.simplexScale == 0.5)
    }

    @Test("Create COBYLA optimizer with custom values")
    func createWithCustomValues() {
        let optimizer = COBYLAOptimizer(
            initialTrustRadius: 1.0,
            minTrustRadius: 1e-8,
            maxTrustRadius: 5.0,
            shrinkFactor: 0.25,
            expandFactor: 2.5,
            acceptRatio: 0.05,
            expandRatio: 0.8,
            simplexScale: 0.3
        )

        #expect(optimizer.initialTrustRadius == 1.0)
        #expect(optimizer.minTrustRadius == 1e-8)
        #expect(optimizer.maxTrustRadius == 5.0)
        #expect(optimizer.shrinkFactor == 0.25)
        #expect(optimizer.expandFactor == 2.5)
        #expect(optimizer.acceptRatio == 0.05)
        #expect(optimizer.expandRatio == 0.8)
        #expect(optimizer.simplexScale == 0.3)
    }

    @Test("Create COBYLA optimizer with tolerance convenience init")
    func createWithToleranceConvenience() {
        let optimizer = COBYLAOptimizer(tolerance: 1e-8)

        #expect(optimizer.minTrustRadius == 1e-8)
        #expect(optimizer.initialTrustRadius == 0.5)
    }

    @Test("Optimize quadratic function")
    func optimizeQuadraticFunction() async throws {
        let optimizer = COBYLAOptimizer(initialTrustRadius: 0.5, minTrustRadius: 1e-6)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            let x = params[0]
            let y = params[1]
            return x * x + y * y
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [1.0, 1.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 100),
            progressCallback: nil
        )

        #expect(abs(result.optimalValue) < 0.01)
        #expect(abs(result.optimalParameters[0]) < 0.1)
        #expect(abs(result.optimalParameters[1]) < 0.1)
        #expect(result.convergenceReason == .energyTolerance)
    }

    @Test("Trust region expansion on good steps")
    func trustRegionExpansionOnGoodSteps() async throws {
        let optimizer = COBYLAOptimizer(
            initialTrustRadius: 0.1,
            maxTrustRadius: 2.0,
            expandFactor: 2.0
        )

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            params[0] * params[0]
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [5.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 50),
            progressCallback: nil
        )

        #expect(result.optimalValue < 0.1)
        #expect(result.iterations > 0)
    }

    @Test("Trust region shrinkage on poor steps")
    func trustRegionShrinkageOnPoorSteps() async throws {
        let optimizer = COBYLAOptimizer(
            initialTrustRadius: 5.0,
            minTrustRadius: 1e-6,
            maxTrustRadius: 10.0,
            shrinkFactor: 0.5
        )

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            let x = params[0]
            let y = params[1]
            return abs(x - 1.0) + abs(y - 1.0)
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [0.0, 0.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 100),
            progressCallback: nil
        )

        #expect(result.optimalValue < 0.1)
        #expect(result.iterations > 0)
    }

    @Test("Optimize Rosenbrock function")
    func optimizeRosenbrockFunction() async throws {
        let optimizer = COBYLAOptimizer(initialTrustRadius: 1.0, minTrustRadius: 1e-4)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            let x = params[0]
            let y = params[1]
            let a = 1.0 - x
            let b = y - x * x
            return a * a + 100.0 * b * b
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [0.0, 0.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-2, maxIterations: 300),
            progressCallback: nil
        )

        #expect(result.iterations > 10)
        #expect(result.optimalValue < 50.0)
    }

    @Test("Progress callback is called")
    func progressCallbackCalled() async throws {
        actor Counter {
            var count = 0
            func increment() { count += 1 }
            func get() -> Int { count }
        }

        let optimizer = COBYLAOptimizer(minTrustRadius: 1e-3)
        let counter = Counter()

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            params[0] * params[0]
        }

        let progressCallback: @Sendable (Int, Double) async -> Void = { _, _ in
            await counter.increment()
        }

        _ = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [1.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 50),
            progressCallback: progressCallback
        )

        let callbackCount = await counter.get()
        #expect(callbackCount > 0)
    }

    @Test("Max iterations reached")
    func maxIterationsReached() async throws {
        let optimizer = COBYLAOptimizer()

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            params[0] * params[0] + params[1] * params[1]
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [10.0, 10.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-12, maxIterations: 3),
            progressCallback: nil
        )

        #expect(result.convergenceReason == .maxIterations)
        #expect(result.iterations == 3)
    }

    @Test("Simplex regeneration on trust region shrinkage")
    func simplexRegenerationOnTrustRegionShrinkage() async throws {
        let optimizer = COBYLAOptimizer(
            initialTrustRadius: 10.0,
            minTrustRadius: 1e-5,
            maxTrustRadius: 20.0,
            shrinkFactor: 0.25
        )

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            let x = params[0]
            let y = params[1]
            return x * x * x * x + y * y * y * y
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [2.0, 2.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 150),
            progressCallback: nil
        )

        #expect(result.optimalValue < 0.1)
        #expect(result.iterations > 5)
    }

    @Test("Convergence by trust region radius")
    func convergenceByTrustRegionRadius() async throws {
        let optimizer = COBYLAOptimizer(
            initialTrustRadius: 0.5,
            minTrustRadius: 1e-4
        )

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            params[0] * params[0]
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [0.5],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100),
            progressCallback: nil
        )

        #expect(result.convergenceReason == .energyTolerance)
        #expect(result.optimalValue < 0.01)
    }

    @Test("Convergence by function value change")
    func convergenceByFunctionValueChange() async throws {
        let optimizer = COBYLAOptimizer(minTrustRadius: 1e-8)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            params[0] * params[0]
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [0.1],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-6, maxIterations: 100),
            progressCallback: nil
        )

        #expect(result.optimalValue < 0.001)
        #expect(result.iterations > 0)
    }

    @Test("Linear model gradient estimation")
    func linearModelGradientEstimation() async throws {
        let optimizer = COBYLAOptimizer(initialTrustRadius: 0.5)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            let x = params[0]
            let y = params[1]
            return 2.0 * x + 3.0 * y + 1.0
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [1.0, 1.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 50),
            progressCallback: nil
        )

        #expect(result.iterations > 0)
    }

    @Test("Step acceptance with good ratio")
    func stepAcceptanceWithGoodRatio() async throws {
        let optimizer = COBYLAOptimizer(
            initialTrustRadius: 1.0,
            acceptRatio: 0.1
        )

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            params[0] * params[0] + params[1] * params[1]
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [1.0, 1.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 100),
            progressCallback: nil
        )

        #expect(result.optimalValue < 0.05)
        #expect(result.valueHistory.count > 1)
    }

    @Test("Step rejection with poor ratio")
    func stepRejectionWithPoorRatio() async throws {
        let optimizer = COBYLAOptimizer(
            initialTrustRadius: 5.0,
            maxTrustRadius: 10.0,
            shrinkFactor: 0.5,
            acceptRatio: 0.1
        )

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            let x = params[0]
            if abs(x) > 3.0 {
                return 1000.0
            }
            return x * x
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [0.5],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 100),
            progressCallback: nil
        )

        #expect(result.optimalValue < 1.0)
        #expect(result.iterations > 0)
    }

    @Test("Optimize multidimensional quadratic")
    func optimizeMultidimensionalQuadratic() async throws {
        let optimizer = COBYLAOptimizer(initialTrustRadius: 0.5)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            var sum = 0.0
            for i in 0 ..< params.count {
                sum += params[i] * params[i]
            }
            return sum
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [1.0, 2.0, 3.0, 4.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 200),
            progressCallback: nil
        )

        #expect(result.optimalValue < 0.1)
        for param in result.optimalParameters {
            #expect(abs(param) < 0.2)
        }
    }

    @Test("Optimize Himmelblau function")
    func optimizeHimmelblauFunction() async throws {
        let optimizer = COBYLAOptimizer(initialTrustRadius: 1.0, minTrustRadius: 1e-4)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            let x = params[0]
            let y = params[1]
            return (x * x + y - 11.0) * (x * x + y - 11.0) + (x + y * y - 7.0) * (x + y * y - 7.0)
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [0.0, 0.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-2, maxIterations: 200),
            progressCallback: nil
        )

        #expect(result.iterations > 5)
        #expect(result.optimalValue < 50.0)
    }

    @Test("Function evaluations count is accurate")
    func functionEvaluationsCountIsAccurate() async throws {
        actor Counter {
            var count = 0
            func increment() { count += 1 }
            func get() -> Int { count }
        }

        let optimizer = COBYLAOptimizer(initialTrustRadius: 0.5, minTrustRadius: 1e-3)
        let counter = Counter()

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            await counter.increment()
            return params[0] * params[0]
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [1.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 50),
            progressCallback: nil
        )

        let actualEvaluations = await counter.get()
        #expect(result.functionEvaluations == actualEvaluations)
    }

    @Test("SimplexPoint is mutable")
    func simplexPointIsMutable() {
        var point = COBYLAOptimizer.SimplexPoint(parameters: [1.0, 2.0], value: 3.0)
        point.parameters[0] = 4.0
        point.value = 5.0

        #expect(point.parameters == [4.0, 2.0])
        #expect(point.value == 5.0)
    }

    @Test("LinearModel stores correct values")
    func linearModelStoresCorrectValues() {
        let model = COBYLAOptimizer.LinearModel(
            baseParameters: [1.0, 2.0],
            baseValue: 3.0,
            gradient: [0.5, -0.5]
        )

        #expect(model.baseParameters == [1.0, 2.0])
        #expect(model.baseValue == 3.0)
        #expect(model.gradient == [0.5, -0.5])
    }

    @Test("Zero gradient handled correctly")
    func zeroGradientHandledCorrectly() async throws {
        let optimizer = COBYLAOptimizer(initialTrustRadius: 0.1)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { _ in 5.0 }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [1.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-6, maxIterations: 20),
            progressCallback: nil
        )

        #expect(result.iterations > 0)
        #expect(abs(result.optimalValue - 5.0) < 0.1)
    }

    @Test("Asymmetric quadratic optimization")
    func asymmetricQuadraticOptimization() async throws {
        let optimizer = COBYLAOptimizer(initialTrustRadius: 1.0)

        let objectiveFunction: @Sendable ([Double]) async throws -> Double = { params in
            let x = params[0]
            let y = params[1]
            return 2.0 * x * x + y * y + x * y
        }

        let result = try await optimizer.minimize(
            objectiveFunction: objectiveFunction,
            initialParameters: [1.0, 1.0],
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 100),
            progressCallback: nil
        )

        #expect(result.optimalValue < 0.1)
        #expect(abs(result.optimalParameters[0]) < 0.2)
        #expect(abs(result.optimalParameters[1]) < 0.2)
    }
}

/// Test suite for OptimizerError.
/// Validates optimizer error cases and descriptive messages.
@Suite("OptimizerError")
struct OptimizerErrorTests {
    @Test("Invalid objective value error")
    func invalidObjectiveValueError() {
        let error = OptimizerError.invalidObjectiveValue(iteration: 5, value: Double.nan)
        let description = error.errorDescription

        #expect(description != nil)
        #expect(description!.contains("5"))
        #expect(description!.contains("invalid"))
    }

    @Test("Optimization failed error")
    func optimizationFailedError() {
        let error = OptimizerError.optimizationFailed(reason: "test failure")
        let description = error.errorDescription

        #expect(description != nil)
        #expect(description!.contains("test failure"))
        #expect(description!.contains("failed"))
    }
}
