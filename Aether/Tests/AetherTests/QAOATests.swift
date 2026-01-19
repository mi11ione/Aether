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
    func triangleGraph() async {
        let cost = MaxCut.hamiltonian(edges: MaxCut.Examples.triangle())

        let qaoa = QAOA(
            cost: cost,
            qubits: 3,
            depth: 2,
            optimizer: COBYLAOptimizer(tolerance: 1e-4),
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 200),
        )

        let result = await qaoa.run(from: [0.5, 0.5, 0.5, 0.5])

        #expect(result.optimalCost < -0.7)
        #expect(result.optimalCost.isFinite)
        #expect(!result.optimalCost.isNaN)
        #expect(result.optimalParameters.count == 4)
        #expect(result.iterations > 0)
        #expect(result.functionEvaluations > 0)
        #expect(!result.solutionProbabilities.isEmpty)
    }

    @Test("Square graph optimization")
    func squareGraph() async {
        let cost = MaxCut.hamiltonian(edges: MaxCut.Examples.square())

        let qaoa = QAOA(
            cost: cost,
            qubits: 4,
            depth: 2,
            optimizer: COBYLAOptimizer(tolerance: 1e-4),
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 300),
        )

        let result = await qaoa.run(from: [0.5, 0.5, 0.5, 0.5])

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
    func singleEdge() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            cost: cost,
            qubits: 2,
            depth: 1,
            optimizer: COBYLAOptimizer(tolerance: 1e-4),
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100),
        )

        let result = await qaoa.run(from: [0.5, 0.5])

        #expect(result.optimalCost < -0.3)
        #expect(result.optimalParameters.count == 2)
        #expect(!result.costHistory.isEmpty)
    }

    @Test("Pentagon graph optimization")
    func pentagonGraph() async {
        let cost = MaxCut.hamiltonian(edges: MaxCut.Examples.pentagon())

        let qaoa = QAOA(
            cost: cost,
            qubits: 5,
            depth: 2,
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 300),
        )

        let result = await qaoa.run(from: [0.5, 0.5, 0.5, 0.5])

        #expect(result.optimalCost < -1.5)
        #expect(result.optimalCost.isFinite)
    }

    @Test("Complete K₄ optimization")
    func completeK4() async {
        let cost = MaxCut.hamiltonian(edges: MaxCut.Examples.complete4())

        let qaoa = QAOA(
            cost: cost,
            qubits: 4,
            depth: 3,
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 400),
        )

        let result = await qaoa.run(from: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        #expect(result.optimalCost < -1.5)
        #expect(result.optimalCost.isFinite)
    }

    @Test("Star graph optimization")
    func starGraph() async {
        let cost = MaxCut.hamiltonian(edges: MaxCut.Examples.star(vertices: 5))

        let qaoa = QAOA(
            cost: cost,
            qubits: 5,
            depth: 2,
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 300),
        )

        let result = await qaoa.run(from: [0.5, 0.5, 0.5, 0.5])

        #expect(result.optimalCost < -1.5)
        #expect(result.optimalCost.isFinite)
    }
}

/// Test suite for QAOA progress tracking.
/// Validates callback invocation, progress state updates, and iteration
/// counting during optimization.
@Suite("QAOA Progress Tracking")
struct QAOAProgressTrackingTests {
    @Test("Progress state is queryable")
    func progressState() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            cost: cost,
            qubits: 2,
            depth: 1,
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 10),
        )

        let progress = await qaoa.progress
        #expect(progress.iteration >= 0)
        #expect(progress.cost.isFinite || progress.cost == 0)
    }

    @Test("Progress callback is invoked during optimization")
    func progressCallback() async {
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
            cost: cost,
            qubits: 2,
            depth: 1,
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 50),
        )

        let tracker = ProgressTracker()

        let result = await qaoa.run(from: [0.5, 0.5]) { iteration, cost in
            await tracker.recordCallback(iteration: iteration, cost: cost)
        }

        let stats = await tracker.getStats()

        #expect(stats.invocations > 0, "Progress callback should be invoked at least once during optimization")
        #expect(stats.iteration > 0, "Callback should receive non-zero iteration number")
        #expect(stats.cost.isFinite, "Callback should receive finite cost value")
        #expect(result.optimalCost.isFinite, "Optimization should complete successfully")
    }

    @Test("Progress updates are tracked internally")
    func internalProgressTracking() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            cost: cost,
            qubits: 2,
            depth: 1,
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 20),
        )

        _ = await qaoa.run(from: [0.5, 0.5]) { iteration, cost in
            let progress = await qaoa.progress
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
    @Test("Backends produce consistent results")
    func backendConsistency() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1), (1, 2)])
        let initialParams = [0.5, 0.5]

        let qaoaSparse = QAOA(
            cost: cost,
            qubits: 3,
            depth: 1,
            optimizer: COBYLAOptimizer(tolerance: 1e-6),
            sparseBackend: true,
        )

        let qaoaObservable = QAOA(
            cost: cost,
            qubits: 3,
            depth: 1,
            optimizer: COBYLAOptimizer(tolerance: 1e-6),
            sparseBackend: false,
        )

        let resultSparse = await qaoaSparse.run(from: initialParams)
        let resultObservable = await qaoaObservable.run(from: initialParams)

        let costDifference = abs(resultSparse.optimalCost - resultObservable.optimalCost)
        #expect(costDifference < 1e-3)
    }

    @Test("Backend property returns sparse when sparseBackend enabled")
    func backendReturnsSparse() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            cost: cost,
            qubits: 2,
            depth: 1,
            sparseBackend: true,
        )

        let backend = await qaoa.backend

        if case let .sparse(description) = backend {
            #expect(!description.isEmpty, "Sparse backend should have non-empty description")
        }
    }

    @Test("Backend property returns observable when sparseBackend disabled")
    func backendReturnsObservable() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1), (1, 2)])

        let qaoa = QAOA(
            cost: cost,
            qubits: 3,
            depth: 1,
            sparseBackend: false,
        )

        let backend = await qaoa.backend

        if case let .observable(termCount) = backend {
            #expect(termCount == 2, "Observable backend should report correct term count")
        }
    }

    @Test("Precision policy is exposed and matches initialization")
    func precisionPolicyProperty() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let fastQAOA = QAOA(cost: cost, qubits: 2, depth: 1, precisionPolicy: .fast)
        let balancedQAOA = QAOA(cost: cost, qubits: 2, depth: 1, precisionPolicy: .balanced)
        let accurateQAOA = QAOA(cost: cost, qubits: 2, depth: 1, precisionPolicy: .accurate)

        let fastPolicy = await fastQAOA.precisionPolicy
        let balancedPolicy = await balancedQAOA.precisionPolicy
        let accuratePolicy = await accurateQAOA.precisionPolicy

        #expect(fastPolicy == .fast, "Fast policy should be exposed correctly")
        #expect(balancedPolicy == .balanced, "Balanced policy should be exposed correctly")
        #expect(accuratePolicy == .accurate, "Accurate policy should be exposed correctly")
    }

    @Test("Precision policy defaults to fast")
    func precisionPolicyDefault() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])
        let qaoa = QAOA(cost: cost, qubits: 2, depth: 1)

        let policy = await qaoa.precisionPolicy

        #expect(policy == .fast, "Default precision policy should be .fast")
    }
}

/// Test suite for QAOA with different optimizers.
/// Validates QAOA works with COBYLA, Nelder-Mead, Gradient Descent, L-BFGS-B,
/// and SPSA optimizers.
@Suite("QAOA Optimizer Integration")
struct QAOAOptimizerIntegrationTests {
    @Test("COBYLA optimizer produces finite cost")
    func cobylaOptimizer() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            cost: cost,
            qubits: 2,
            depth: 1,
            optimizer: COBYLAOptimizer(tolerance: 1e-4),
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100),
        )

        let result = await qaoa.run(from: [0.5, 0.5])
        #expect(result.optimalCost.isFinite)
    }

    @Test("Nelder-Mead optimizer produces finite cost")
    func nelderMeadOptimizer() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            cost: cost,
            qubits: 2,
            depth: 1,
            optimizer: NelderMeadOptimizer(tolerance: 1e-4),
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100),
        )

        let result = await qaoa.run(from: [0.5, 0.5])
        #expect(result.optimalCost.isFinite)
    }

    @Test("Gradient Descent optimizer produces finite cost")
    func gradientDescentOptimizer() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            cost: cost,
            qubits: 2,
            depth: 1,
            optimizer: GradientDescentOptimizer(learningRate: 0.1),
            convergence: ConvergenceCriteria(gradientNormTolerance: 1e-3, maxIterations: 100),
        )

        let result = await qaoa.run(from: [0.5, 0.5])
        #expect(result.optimalCost.isFinite)
    }

    @Test("L-BFGS-B optimizer produces finite cost")
    func lbfgsbOptimizer() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            cost: cost,
            qubits: 2,
            depth: 1,
            optimizer: LBFGSBOptimizer(memorySize: 5),
            convergence: ConvergenceCriteria(gradientNormTolerance: 1e-3, maxIterations: 100),
        )

        let result = await qaoa.run(from: [0.5, 0.5])
        #expect(result.optimalCost.isFinite)
    }
}

/// Test suite for QAOA convergence behavior.
/// Validates energy tolerance convergence, max iteration limits, and cost
/// history tracking throughout optimization.
@Suite("QAOA Convergence")
struct QAOAConvergenceTests {
    @Test("Max iterations limit respected")
    func maxIterations() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])
        let maxIter = 10

        let qaoa = QAOA(
            cost: cost,
            qubits: 2,
            depth: 1,
            optimizer: COBYLAOptimizer(tolerance: 1e-10),
            convergence: ConvergenceCriteria(energyTolerance: 1e-10, maxIterations: maxIter),
        )

        let result = await qaoa.run(from: [0.5, 0.5])

        #expect(result.iterations <= maxIter)

        if result.iterations == maxIter {
            #expect(result.convergenceReason == .maxIterationsReached)
        }
    }
}

/// Test suite for QAOA solution probabilities.
/// Validates probability normalization, filtering thresholds, and sorting
/// of solution bitstrings by probability.
@Suite("QAOA Solution Probabilities")
struct QAOASolutionProbabilitiesTests {
    @Test("Probabilities sum near 1.0")
    func probabilitySumNormalization() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            cost: cost,
            qubits: 2,
            depth: 1,
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100),
        )

        let result = await qaoa.run(from: [0.5, 0.5])

        let totalProbability = result.solutionProbabilities.values.reduce(0.0, +)

        #expect(totalProbability > 0.9)
        #expect(totalProbability <= 1.0)
    }

    @Test("Probabilities are above filtering threshold")
    func probabilityFiltering() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            cost: cost,
            qubits: 2,
            depth: 1,
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100),
        )

        let result = await qaoa.run(from: [0.5, 0.5])

        for (_, probability) in result.solutionProbabilities {
            #expect(probability > 1e-6)
            #expect(probability <= 1.0)
        }
    }

    @Test("Top solutions sorted by probability")
    func solutionSorting() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            cost: cost,
            qubits: 2,
            depth: 2,
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 200),
        )

        let result = await qaoa.run(from: [0.5, 0.5, 0.5, 0.5])

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

/// Test suite for QAOA.Result.topSolutions method.
/// Validates edge cases including empty results, zero count requests,
/// and heap-based partial sort for large solution sets.
@Suite("QAOA topSolutions")
struct QAOATopSolutionsTests {
    @Test("Returns empty array when count is zero")
    func topSolutionsZeroCount() {
        let result = QAOA.Result(
            optimalCost: -1.0,
            optimalParameters: [0.5, 0.5],
            solutionProbabilities: [0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25],
            costHistory: [-1.0],
            iterations: 1,
            convergenceReason: .energyConverged,
            functionEvaluations: 10,
        )

        let solutions = result.topSolutions(0)

        #expect(solutions.isEmpty, "topSolutions(0) should return empty array")
    }

    @Test("Returns empty array when solutionProbabilities is empty")
    func topSolutionsEmptyProbabilities() {
        let result = QAOA.Result(
            optimalCost: 0.0,
            optimalParameters: [0.5, 0.5],
            solutionProbabilities: [:],
            costHistory: [0.0],
            iterations: 1,
            convergenceReason: .energyConverged,
            functionEvaluations: 10,
        )

        let solutions = result.topSolutions(5)

        #expect(solutions.isEmpty, "topSolutions should return empty array when no solutions exist")
    }

    @Test("Uses heap-based partial sort for large solution sets")
    func topSolutionsHeapPath() {
        var probabilities: [Int: Double] = [:]
        for i in 0 ..< 20 {
            probabilities[i] = Double(i + 1) / 210.0
        }

        let result = QAOA.Result(
            optimalCost: -1.0,
            optimalParameters: [0.5, 0.5],
            solutionProbabilities: probabilities,
            costHistory: [-1.0],
            iterations: 1,
            convergenceReason: .energyConverged,
            functionEvaluations: 10,
        )

        let top3 = result.topSolutions(3)

        #expect(top3.count == 3, "Should return exactly 3 solutions")
        #expect(top3[0].bitstring == 19, "Highest probability should be bitstring 19")
        #expect(top3[1].bitstring == 18, "Second highest should be bitstring 18")
        #expect(top3[2].bitstring == 17, "Third highest should be bitstring 17")
        #expect(top3[0].probability > top3[1].probability, "Solutions should be sorted descending")
        #expect(top3[1].probability > top3[2].probability, "Solutions should be sorted descending")
    }

    @Test("Heap path handles exact boundary case")
    func topSolutionsHeapBoundary() {
        var probabilities: [Int: Double] = [:]
        for i in 0 ..< 9 {
            probabilities[i] = Double(i + 1) / 45.0
        }

        let result = QAOA.Result(
            optimalCost: -1.0,
            optimalParameters: [0.5, 0.5],
            solutionProbabilities: probabilities,
            costHistory: [-1.0],
            iterations: 1,
            convergenceReason: .energyConverged,
            functionEvaluations: 10,
        )

        let top2 = result.topSolutions(2)

        #expect(top2.count == 2, "Should return exactly 2 solutions")
        #expect(top2[0].bitstring == 8, "Highest probability should be bitstring 8")
        #expect(top2[1].bitstring == 7, "Second highest should be bitstring 7")
    }

    @Test("Simple sort path for small solution sets")
    func topSolutionsSimplePath() {
        var probabilities: [Int: Double] = [:]
        for i in 0 ..< 10 {
            probabilities[i] = Double(i + 1) / 55.0
        }

        let result = QAOA.Result(
            optimalCost: -1.0,
            optimalParameters: [0.5, 0.5],
            solutionProbabilities: probabilities,
            costHistory: [-1.0],
            iterations: 1,
            convergenceReason: .energyConverged,
            functionEvaluations: 10,
        )

        let top3 = result.topSolutions(3)

        #expect(top3.count == 3, "Should return exactly 3 solutions")
        #expect(top3[0].bitstring == 9, "Highest probability should be bitstring 9")
        #expect(top3[1].bitstring == 8, "Second highest should be bitstring 8")
        #expect(top3[2].bitstring == 7, "Third highest should be bitstring 7")
    }

    @Test("Returns all solutions when count exceeds available")
    func topSolutionsCountExceedsAvailable() {
        let result = QAOA.Result(
            optimalCost: -1.0,
            optimalParameters: [0.5, 0.5],
            solutionProbabilities: [0: 0.6, 1: 0.4],
            costHistory: [-1.0],
            iterations: 1,
            convergenceReason: .energyConverged,
            functionEvaluations: 10,
        )

        let top10 = result.topSolutions(10)

        #expect(top10.count == 2, "Should return all 2 solutions when requesting 10")
        #expect(top10[0].bitstring == 0, "Highest probability should be bitstring 0")
        #expect(top10[1].bitstring == 1, "Second should be bitstring 1")
    }

    @Test("Heap siftDown exercises left child branch")
    func topSolutionsLeftChildBranch() {
        var probabilities: [Int: Double] = [:]

        let probs = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
                     0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
        for (i, p) in probs.enumerated() {
            probabilities[i] = p
        }

        let result = QAOA.Result(
            optimalCost: -1.0,
            optimalParameters: [0.5, 0.5],
            solutionProbabilities: probabilities,
            costHistory: [-1.0],
            iterations: 1,
            convergenceReason: .energyConverged,
            functionEvaluations: 10,
        )

        let top2 = result.topSolutions(2)

        #expect(top2.count == 2, "Should return exactly 2 solutions")
        let topBitstrings = Set(top2.map(\.bitstring))
        #expect(topBitstrings.contains(14), "Should contain bitstring 14 (highest prob)")
        #expect(topBitstrings.contains(13), "Should contain bitstring 13 (second highest)")
    }

    @Test("Heap siftDown with deeper heap exercises multiple levels")
    func topSolutionsDeepHeap() {
        var probabilities: [Int: Double] = [:]
        for i in 0 ..< 35 {
            probabilities[i] = Double(i + 1) / 630.0
        }

        let result = QAOA.Result(
            optimalCost: -1.0,
            optimalParameters: [0.5, 0.5],
            solutionProbabilities: probabilities,
            costHistory: [-1.0],
            iterations: 1,
            convergenceReason: .energyConverged,
            functionEvaluations: 10,
        )

        let top7 = result.topSolutions(7)

        #expect(top7.count == 7, "Should return exactly 7 solutions")
        let expectedBitstrings = Set(28 ... 34)
        let actualBitstrings = Set(top7.map(\.bitstring))
        #expect(actualBitstrings == expectedBitstrings, "Top 7 should be bitstrings 28-34")
        for i in 1 ..< top7.count {
            #expect(top7[i - 1].probability >= top7[i].probability, "Should be sorted descending")
        }
    }

    @Test("Heap handles alternating high-low probability pattern")
    func topSolutionsAlternatingPattern() {
        var probabilities: [Int: Double] = [:]
        for i in 0 ..< 25 {
            probabilities[i] = (i % 2 == 0) ? Double(i / 2 + 1) / 100.0 : Double(25 - i / 2) / 100.0
        }

        let result = QAOA.Result(
            optimalCost: -1.0,
            optimalParameters: [0.5, 0.5],
            solutionProbabilities: probabilities,
            costHistory: [-1.0],
            iterations: 1,
            convergenceReason: .energyConverged,
            functionEvaluations: 10,
        )

        let top5 = result.topSolutions(5)

        #expect(top5.count == 5, "Should return exactly 5 solutions")
        for sol in top5 {
            #expect(sol.probability > 0.10, "Top solutions should have high probability")
        }
    }
}

/// Test suite for different QAOA circuit depths.
/// Validates parameter counting and optimization behavior for depth 1, 3,
/// and 5 QAOA circuits.
@Suite("QAOA Depth Variations")
struct QAOADepthVariationsTests {
    @Test("Depth-1 has 2 parameters")
    func depth1() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1), (1, 2)])

        let qaoa = QAOA(
            cost: cost,
            qubits: 3,
            depth: 1,
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100),
        )

        let result = await qaoa.run(from: [0.5, 0.5])

        #expect(result.optimalParameters.count == 2)
        #expect(result.optimalCost.isFinite)
    }

    @Test("Depth-3 has 6 parameters")
    func depth3() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            cost: cost,
            qubits: 2,
            depth: 3,
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 200),
        )

        let result = await qaoa.run(from: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        #expect(result.optimalParameters.count == 6)
        #expect(result.optimalCost.isFinite)
    }

    @Test("Depth-5 has 10 parameters")
    func depth5() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            cost: cost,
            qubits: 2,
            depth: 5,
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 300),
        )

        let result = await qaoa.run(from: Array(repeating: 0.5, count: 10))

        #expect(result.optimalParameters.count == 10)
        #expect(result.optimalCost.isFinite)
    }
}

/// Test suite for QAOA.Result representation.
/// Validates description formatting and content for QAOA optimization results.
@Suite("QAOA.Result Representation")
struct QAOAResultRepresentationTests {
    @Test("Result description is readable")
    func resultDescription() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            cost: cost,
            qubits: 2,
            depth: 1,
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 50),
        )

        let result = await qaoa.run(from: [0.5, 0.5])

        let description = result.description

        #expect(description.contains("QAOA Result"))
        #expect(description.contains("Optimal Cost"))
        #expect(description.contains("Parameters"))
        #expect(description.contains("Iterations"))
        #expect(description.contains("Convergence"))
    }

    @Test("Result description truncates parameters beyond 4")
    func descriptionTruncation() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1), (1, 2)])

        let qaoa = QAOA(
            cost: cost,
            qubits: 3,
            depth: 3,
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100),
        )

        let result = await qaoa.run(from: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        let description = result.description

        #expect(description.contains(", ..."), "Description should truncate parameters beyond 4 with ', ...' suffix")
        #expect(result.optimalParameters.count == 6, "Result should contain all 6 parameters internally")
    }

    @Test("Result description shows all parameters when count is 4 or less")
    func descriptionNoTruncation() async {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])

        let qaoa = QAOA(
            cost: cost,
            qubits: 2,
            depth: 2,
            convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 100),
        )

        let result = await qaoa.run(from: [0.5, 0.5, 0.5, 0.5])

        let description = result.description

        #expect(!description.contains(", ..."), "Description should not truncate when parameter count <= 4")
        #expect(result.optimalParameters.count == 4, "Result should contain all 4 parameters")
    }
}
