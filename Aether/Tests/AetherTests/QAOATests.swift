// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Test suite for QAOA algorithm on standard graphs.
/// Validates complete optimization workflows, convergence behavior, and
/// solution quality for triangle, square, and single-edge MaxCut problems.
@Suite("QAOA Algorithm")
struct QAOAAlgorithmTests {
    @Test("Triangle graph optimization")
    func triangleGraph() async throws {
        let cost = MaxCut.hamiltonian(edges: MaxCut.Examples.triangle())

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 3,
            depth: 2,
            optimizer: COBYLAOptimizer(tolerance: 1e-4),
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 200)
        )

        let result = try await qaoa.run(initialParameters: [0.5, 0.5, 0.5, 0.5])

        #expect(result.optimalCost < -0.7)
        #expect(result.optimalCost.isFinite)
        #expect(!result.optimalCost.isNaN)
        #expect(result.optimalParameters.count == 4)
        #expect(result.iterations > 0)
        #expect(result.functionEvaluations > 0)
        #expect(!result.solutionProbabilities.isEmpty)
    }

    @Test("Square graph optimization")
    func squareGraph() async throws {
        let cost = MaxCut.hamiltonian(edges: MaxCut.Examples.square())

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 4,
            depth: 2,
            optimizer: COBYLAOptimizer(tolerance: 1e-4),
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 300)
        )

        let result = try await qaoa.run(initialParameters: [0.5, 0.5, 0.5, 0.5])

        #expect(result.optimalCost < -1.5)
        #expect(result.optimalCost.isFinite)
        #expect(!result.solutionProbabilities.isEmpty)

        let topSolutions = result.solutionProbabilities.sorted { $0.value > $1.value }.prefix(3)
        #expect(topSolutions.count > 0)

        for (_, probability) in topSolutions {
            #expect(probability > 0)
            #expect(probability <= 1.0)
        }
    }

    @Test("Single edge depth-1 optimization")
    func singleEdge() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 1,
            optimizer: COBYLAOptimizer(tolerance: 1e-4),
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100)
        )

        let result = try await qaoa.run(initialParameters: [0.5, 0.5])

        #expect(result.optimalCost < -0.3)
        #expect(result.optimalParameters.count == 2)
        #expect(!result.costHistory.isEmpty)
    }

    @Test("Pentagon graph optimization")
    func pentagonGraph() async throws {
        let cost = MaxCut.hamiltonian(edges: MaxCut.Examples.pentagon())

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 5,
            depth: 2,
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 300)
        )

        let result = try await qaoa.run(initialParameters: [0.5, 0.5, 0.5, 0.5])

        #expect(result.optimalCost < -1.5)
        #expect(result.optimalCost.isFinite)
    }

    @Test("Complete K₄ optimization")
    func completeK4() async throws {
        let cost = MaxCut.hamiltonian(edges: MaxCut.Examples.completeK4())

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 4,
            depth: 3,
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 400)
        )

        let result = try await qaoa.run(initialParameters: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        #expect(result.optimalCost < -1.5)
        #expect(result.optimalCost.isFinite)
    }

    @Test("Star graph optimization")
    func starGraph() async throws {
        let cost = MaxCut.hamiltonian(edges: MaxCut.Examples.star(numVertices: 5))

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 5,
            depth: 2,
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 300)
        )

        let result = try await qaoa.run(initialParameters: [0.5, 0.5, 0.5, 0.5])

        #expect(result.optimalCost < -1.5)
        #expect(result.optimalCost.isFinite)
    }
}

/// Test suite for QAOA parameter validation.
/// Verifies error handling for mismatched parameter counts, empty arrays,
/// and invalid parameter values.
@Suite("QAOA Parameter Validation")
struct QAOAParameterValidationTests {
    @Test("Too few parameters throws error")
    func tooFewParameters() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 2
        )

        await #expect(throws: QAOAError.self) {
            try await qaoa.run(initialParameters: [0.5, 0.5])
        }
    }

    @Test("Too many parameters throws error")
    func tooManyParameters() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 1
        )

        await #expect(throws: QAOAError.self) {
            try await qaoa.run(initialParameters: [0.5, 0.5, 0.5, 0.5])
        }
    }

    @Test("Empty parameters throws error")
    func emptyParameters() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 1
        )

        await #expect(throws: QAOAError.self) {
            try await qaoa.run(initialParameters: [])
        }
    }

    @Test("Error description is actionable")
    func errorDescriptions() {
        let paramError = QAOAError.parameterCountMismatch(expected: 4, got: 2)
        #expect(paramError.errorDescription != nil)
        #expect(paramError.errorDescription!.contains("mismatch"))
        #expect(paramError.errorDescription!.contains("4"))
        #expect(paramError.errorDescription!.contains("2"))

        let costError = QAOAError.invalidCost(value: Double.nan, parameters: [0.5, 0.5])
        #expect(costError.errorDescription != nil)
        #expect(costError.errorDescription!.contains("invalid"))
        #expect(costError.errorDescription!.contains("nan"))
    }
}

/// Test suite for QAOA progress tracking.
/// Validates callback invocation, progress state updates, and iteration
/// counting during optimization.
@Suite("QAOA Progress Tracking")
struct QAOAProgressTrackingTests {
    @Test("Progress state is queryable")
    func progressState() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 1,
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 10)
        )

        let progress = await qaoa.getProgress()
        #expect(progress.iteration >= 0)
        #expect(progress.cost.isFinite || progress.cost == 0)
    }

    @Test("Progress callback is invoked during optimization")
    func progressCallback() async throws {
        actor ProgressTracker {
            var callbackInvocations = 0
            var lastIteration = 0
            var lastCost = 0.0

            func recordCallback(iteration: Int, cost: Double) {
                callbackInvocations += 1
                lastIteration = iteration
                lastCost = cost
            }

            func getStats() -> (invocations: Int, iteration: Int, cost: Double) {
                (callbackInvocations, lastIteration, lastCost)
            }
        }

        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 1,
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 50)
        )

        let tracker = ProgressTracker()

        let result = try await qaoa.runWithProgress(initialParameters: [0.5, 0.5]) { iteration, cost in
            await tracker.recordCallback(iteration: iteration, cost: cost)
        }

        let stats = await tracker.getStats()

        #expect(stats.invocations > 0, "Progress callback should be invoked at least once during optimization")
        #expect(stats.iteration > 0, "Callback should receive non-zero iteration number")
        #expect(stats.cost.isFinite, "Callback should receive finite cost value")
        #expect(result.optimalCost.isFinite, "Optimization should complete successfully")
    }

    @Test("Progress updates are tracked internally")
    func internalProgressTracking() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 1,
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 20)
        )

        _ = try await qaoa.runWithProgress(initialParameters: [0.5, 0.5]) { iteration, cost in
            let progress = await qaoa.getProgress()
            #expect(progress.iteration == iteration, "Internal progress iteration should match callback iteration")
            #expect(abs(progress.cost - cost) < 1e-10, "Internal progress cost should match callback cost")
        }
    }
}

/// Test suite for QAOA backend selection.
/// Validates SparseHamiltonian and Observable backends produce consistent
/// results and correct backend reporting.
@Suite("QAOA Backend Selection")
struct QAOABackendSelectionTests {
    @Test("SparseHamiltonian backend is default")
    func sparseBackendDefault() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1), (1, 2)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 3,
            depth: 1,
            useSparseBackend: true
        )

        let backendInfo = await qaoa.getBackendInfo()
        #expect(backendInfo.contains("SparseHamiltonian"))
    }

    @Test("Observable backend explicit selection")
    func observableBackend() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 1,
            useSparseBackend: false
        )

        let backendInfo = await qaoa.getBackendInfo()
        #expect(backendInfo.contains("Observable"))
    }

    @Test("Backends produce consistent results")
    func backendConsistency() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1), (1, 2)])
        let initialParams = [0.5, 0.5]

        let qaoaSparse = QAOA(
            costHamiltonian: cost,
            numQubits: 3,
            depth: 1,
            optimizer: COBYLAOptimizer(tolerance: 1e-6),
            useSparseBackend: true
        )

        let qaoaObservable = QAOA(
            costHamiltonian: cost,
            numQubits: 3,
            depth: 1,
            optimizer: COBYLAOptimizer(tolerance: 1e-6),
            useSparseBackend: false
        )

        let resultSparse = try await qaoaSparse.run(initialParameters: initialParams)
        let resultObservable = try await qaoaObservable.run(initialParameters: initialParams)

        let costDifference = abs(resultSparse.optimalCost - resultObservable.optimalCost)
        #expect(costDifference < 1e-3)
    }
}

/// Test suite for QAOA with different optimizers.
/// Validates QAOA works with COBYLA, Nelder-Mead, Gradient Descent, L-BFGS-B,
/// and SPSA optimizers.
@Suite("QAOA Optimizer Integration")
struct QAOAOptimizerIntegrationTests {
    @Test("COBYLA optimizer produces finite cost")
    func cobylaOptimizer() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 1,
            optimizer: COBYLAOptimizer(tolerance: 1e-4),
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100)
        )

        let result = try await qaoa.run(initialParameters: [0.5, 0.5])
        #expect(result.optimalCost.isFinite)
    }

    @Test("Nelder-Mead optimizer produces finite cost")
    func nelderMeadOptimizer() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 1,
            optimizer: NelderMeadOptimizer(tolerance: 1e-4),
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100)
        )

        let result = try await qaoa.run(initialParameters: [0.5, 0.5])
        #expect(result.optimalCost.isFinite)
    }

    @Test("Gradient Descent optimizer produces finite cost")
    func gradientDescentOptimizer() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 1,
            optimizer: GradientDescentOptimizer(learningRate: 0.1),
            convergenceCriteria: ConvergenceCriteria(gradientNormTolerance: 1e-3, maxIterations: 100)
        )

        let result = try await qaoa.run(initialParameters: [0.5, 0.5])
        #expect(result.optimalCost.isFinite)
    }

    @Test("L-BFGS-B optimizer produces finite cost")
    func lbfgsbOptimizer() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 1,
            optimizer: LBFGSBOptimizer(memorySize: 5),
            convergenceCriteria: ConvergenceCriteria(gradientNormTolerance: 1e-3, maxIterations: 100)
        )

        let result = try await qaoa.run(initialParameters: [0.5, 0.5])
        #expect(result.optimalCost.isFinite)
    }
}

/// Test suite for QAOA convergence behavior.
/// Validates energy tolerance convergence, max iteration limits, and cost
/// history tracking throughout optimization.
@Suite("QAOA Convergence")
struct QAOAConvergenceTests {
    @Test("Max iterations limit respected")
    func maxIterations() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])
        let maxIter = 10

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 1,
            optimizer: COBYLAOptimizer(tolerance: 1e-10),
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-10, maxIterations: maxIter)
        )

        let result = try await qaoa.run(initialParameters: [0.5, 0.5])

        #expect(result.iterations <= maxIter)

        if result.iterations == maxIter {
            #expect(result.convergenceReason == .maxIterations)
        }
    }
}

/// Test suite for QAOA solution probabilities.
/// Validates probability normalization, filtering thresholds, and sorting
/// of solution bitstrings by probability.
@Suite("QAOA Solution Probabilities")
struct QAOASolutionProbabilitiesTests {
    @Test("Probabilities sum near 1.0")
    func probabilitySumNormalization() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 1,
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100)
        )

        let result = try await qaoa.run(initialParameters: [0.5, 0.5])

        let totalProbability = result.solutionProbabilities.values.reduce(0.0, +)

        #expect(totalProbability > 0.9)
        #expect(totalProbability <= 1.0)
    }

    @Test("Probabilities are above filtering threshold")
    func probabilityFiltering() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 1,
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100)
        )

        let result = try await qaoa.run(initialParameters: [0.5, 0.5])

        for (_, probability) in result.solutionProbabilities {
            #expect(probability > 1e-6)
            #expect(probability <= 1.0)
        }
    }

    @Test("Top solutions sorted by probability")
    func solutionSorting() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 2,
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 200)
        )

        let result = try await qaoa.run(initialParameters: [0.5, 0.5, 0.5, 0.5])

        let sortedSolutions = result.solutionProbabilities.sorted { $0.value > $1.value }

        #expect(sortedSolutions.count > 0)

        if sortedSolutions.count > 1 {
            for i in 1 ..< sortedSolutions.count {
                #expect(sortedSolutions[i].value <= sortedSolutions[i - 1].value)
            }
        }

        if let topSolution = sortedSolutions.first {
            #expect(topSolution.key >= 0 && topSolution.key < 4)
        }
    }
}

/// Test suite for different QAOA circuit depths.
/// Validates parameter counting and optimization behavior for depth 1, 3,
/// and 5 QAOA circuits.
@Suite("QAOA Depth Variations")
struct QAOADepthVariationsTests {
    @Test("Depth-1 has 2 parameters")
    func depth1() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1), (1, 2)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 3,
            depth: 1,
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100)
        )

        let result = try await qaoa.run(initialParameters: [0.5, 0.5])

        #expect(result.optimalParameters.count == 2)
        #expect(result.optimalCost.isFinite)
    }

    @Test("Depth-3 has 6 parameters")
    func depth3() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 3,
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 200)
        )

        let result = try await qaoa.run(initialParameters: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        #expect(result.optimalParameters.count == 6)
        #expect(result.optimalCost.isFinite)
    }

    @Test("Depth-5 has 10 parameters")
    func depth5() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 5,
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 300)
        )

        let result = try await qaoa.run(initialParameters: Array(repeating: 0.5, count: 10))

        #expect(result.optimalParameters.count == 10)
        #expect(result.optimalCost.isFinite)
    }
}

/// Test suite for QAOAResult representation.
/// Validates description formatting and content for QAOA optimization results.
@Suite("QAOA Result Representation")
struct QAOAResultRepresentationTests {
    @Test("Result description is readable")
    func resultDescription() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 1,
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 50)
        )

        let result = try await qaoa.run(initialParameters: [0.5, 0.5])

        let description = result.description

        #expect(description.contains("QAOA Result"))
        #expect(description.contains("Optimal Cost"))
        #expect(description.contains("Parameters"))
        #expect(description.contains("Iterations"))
        #expect(description.contains("Convergence"))
    }

    @Test("Result description truncates parameters beyond 4")
    func descriptionTruncation() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1), (1, 2)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 3,
            depth: 3,
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100)
        )

        let result = try await qaoa.run(initialParameters: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        let description = result.description

        #expect(description.contains(", ..."), "Description should truncate parameters beyond 4 with ', ...' suffix")
        #expect(result.optimalParameters.count == 6, "Result should contain all 6 parameters internally")
    }

    @Test("Result description shows all parameters when count is 4 or less")
    func descriptionNoTruncation() async throws {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            costHamiltonian: cost,
            numQubits: 2,
            depth: 2,
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100)
        )

        let result = try await qaoa.run(initialParameters: [0.5, 0.5, 0.5, 0.5])

        let description = result.description

        #expect(!description.contains(", ..."), "Description should not truncate when parameter count <= 4")
        #expect(result.optimalParameters.count == 4, "Result should contain all 4 parameters")
    }
}
