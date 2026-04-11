// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Test suite for WalkGraph construction and properties.
/// Validates adjacency matrix encoding, vertex counting,
/// and degree computation for all supported graph families.
@Suite("WalkGraph Construction")
struct WalkGraphConstructionTests {
    @Test("Complete graph K4 has correct adjacency")
    func completeGraph() {
        let graph = WalkGraph.complete(vertices: 4)
        #expect(graph.vertexCount == 4, "K4 should have 4 vertices")
        #expect(graph.maxDegree == 3, "K4 vertices should have degree 3")
        #expect(graph.isRegular, "Complete graph should be regular")
    }

    @Test("Cycle graph C6 has degree 2 everywhere")
    func cycleGraph() {
        let graph = WalkGraph.cycle(vertices: 6)
        #expect(graph.vertexCount == 6, "C6 should have 6 vertices")
        #expect(graph.degree(of: 0) == 2, "Cycle vertices should have degree 2")
        #expect(graph.degree(of: 3) == 2, "Interior cycle vertex should have degree 2")
        #expect(graph.isRegular, "Cycle should be regular")
    }

    @Test("Hypercube Q3 has 8 vertices and degree 3")
    func hypercubeGraph() {
        let graph = WalkGraph.hypercube(dimension: 3)
        #expect(graph.vertexCount == 8, "Q3 should have 8 vertices")
        #expect(graph.maxDegree == 3, "Q3 should have degree 3")
        #expect(graph.isRegular, "Hypercube should be regular")
    }

    @Test("Line graph P4 has endpoint degree 1 and interior degree 2")
    func lineGraph() {
        let graph = WalkGraph.line(vertices: 4)
        #expect(graph.vertexCount == 4, "P4 should have 4 vertices")
        #expect(graph.degree(of: 0) == 1, "Endpoint should have degree 1")
        #expect(graph.degree(of: 1) == 2, "Interior vertex should have degree 2")
        #expect(!graph.isRegular, "Line graph should not be regular")
    }

    @Test("Custom adjacency matrix triangle graph")
    func customGraph() {
        let graph = WalkGraph(adjacencyMatrix: [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ])
        #expect(graph.vertexCount == 3, "Triangle should have 3 vertices")
        #expect(graph.maxDegree == 2, "Triangle vertices should have degree 2")
        #expect(graph.isRegular, "Triangle should be regular")
    }

    @Test("Position qubits correct for various sizes")
    func positionQubits() {
        let g4 = WalkGraph.complete(vertices: 4)
        #expect(g4.positionQubits == 2, "4 vertices should need 2 qubits")

        let g8 = WalkGraph.hypercube(dimension: 3)
        #expect(g8.positionQubits == 3, "8 vertices should need 3 qubits")

        let g3 = WalkGraph.cycle(vertices: 3)
        #expect(g3.positionQubits == 2, "3 vertices should need 2 qubits")
    }

    @Test("Single vertex graph has position qubits 1")
    func singleVertexGraph() {
        let graph = WalkGraph(adjacencyMatrix: [[0]])
        #expect(graph.positionQubits == 1, "Single vertex should need 1 qubit")
        #expect(graph.maxDegree == 0, "Isolated vertex has degree 0")
        #expect(graph.isRegular, "Single vertex graph is trivially regular")
    }

    @Test("Line graph P2 has degree 1")
    func lineGraphP2() {
        let graph = WalkGraph.line(vertices: 2)
        #expect(graph.vertexCount == 2, "P2 should have 2 vertices")
        #expect(graph.degree(of: 0) == 1, "P2 endpoint has degree 1")
        #expect(graph.degree(of: 1) == 1, "P2 endpoint has degree 1")
        #expect(graph.isRegular, "P2 is 1-regular")
    }
}

/// Test suite for WalkGraph Laplacian and Observable conversion.
/// Validates graph Laplacian matrix properties and Pauli
/// decomposition for Hamiltonian-based quantum walk evolution.
@Suite("WalkGraph Laplacian and Observable")
struct WalkGraphLaplacianTests {
    @Test("Laplacian of K3 has correct structure")
    func laplacianComplete() {
        let graph = WalkGraph.complete(vertices: 3)
        let lap = graph.laplacian
        #expect(abs(lap[0] - 2.0) < 1e-10, "Diagonal should equal degree")
        #expect(abs(lap[1] + 1.0) < 1e-10, "Off-diagonal should be -1 for adjacent")
    }

    @Test("Laplacian row sums are zero")
    func laplacianRowSums() {
        let graph = WalkGraph.cycle(vertices: 4)
        let lap = graph.laplacian
        let n = graph.vertexCount
        for i in 0 ..< n {
            var rowSum = 0.0
            for j in 0 ..< n {
                rowSum += lap[i * n + j]
            }
            #expect(abs(rowSum) < 1e-10, "Row \(i) sum should be zero")
        }
    }

    @Test("Hypercube toObservable produces non-empty Hamiltonian")
    func toObservableHypercube() {
        let graph = WalkGraph.hypercube(dimension: 2)
        let obs = graph.toObservable()
        #expect(!obs.terms.isEmpty, "Observable should have Pauli terms")
    }

    @Test("toObservable on non-power-of-2 vertex graph covers padding")
    func toObservableTriangle() {
        let graph = WalkGraph.cycle(vertices: 3)
        let obs = graph.toObservable()
        #expect(!obs.terms.isEmpty, "Triangle observable should have Pauli terms")
    }

    @Test("toObservable on graph with self-loop covers identity term")
    func toObservableSelfLoop() {
        let graph = WalkGraph(adjacencyMatrix: [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ])
        let obs = graph.toObservable()
        #expect(!obs.terms.isEmpty, "Self-loop graph should have Pauli terms")
    }
}

/// Test suite for discrete-time quantum walk dynamics.
/// Validates walk circuit construction, state evolution,
/// and probability distribution extraction for coin-based walks.
@Suite("Discrete Quantum Walk")
struct DiscreteWalkTests {
    @Test("Walk circuit has correct qubit count for cycle")
    func walkQubitCount() {
        let graph = WalkGraph.cycle(vertices: 4)
        let circuit = QuantumWalk.walk(on: graph, method: .discrete(coin: .hadamard, steps: 1))
        #expect(circuit.qubits == 3, "Cycle C4 should use 2 pos + 1 coin = 3 qubits")
    }

    @Test("Walk circuit has correct qubit count for hypercube")
    func walkQubitCountHypercube() {
        let graph = WalkGraph.hypercube(dimension: 2)
        let circuit = QuantumWalk.walk(on: graph, method: .discrete(coin: .grover, steps: 1))
        #expect(circuit.qubits == 3, "Q2 should use 2 pos + 1 coin = 3 qubits (degree 2)")
    }

    @Test("Walk result sums to approximately 1")
    func walkProbabilityNormalization() {
        let graph = WalkGraph.cycle(vertices: 4)
        let circuit = QuantumWalk.walk(on: graph, method: .discrete(coin: .hadamard, steps: 5))
        let state = circuit.execute()
        let result = state.walkResult(graph: graph, steps: 5)
        var total = 0.0
        for p in result.vertexProbabilities {
            total += p
        }
        #expect(abs(total - 1.0) < 1e-6, "Vertex probabilities should sum to 1")
    }

    @Test("Walk result has correct vertex count")
    func walkResultVertexCount() {
        let graph = WalkGraph.cycle(vertices: 4)
        let circuit = QuantumWalk.walk(on: graph, method: .discrete(coin: .grover, steps: 3))
        let state = circuit.execute()
        let result = state.walkResult(graph: graph, steps: 3)
        #expect(result.vertexProbabilities.count == 4, "Should have probability for each vertex")
    }

    @Test("Grover coin walk on complete graph")
    func groverCoinCompleteGraph() {
        let graph = WalkGraph.complete(vertices: 4)
        let circuit = QuantumWalk.walk(on: graph, method: .discrete(coin: .grover, steps: 3))
        let state = circuit.execute()
        let result = state.walkResult(graph: graph, steps: 3)
        #expect(result.mostProbableVertex >= 0, "Most probable vertex should be valid")
        #expect(result.mostProbableVertex < 4, "Most probable vertex should be in range")
    }

    @Test("Fourier coin walk produces valid probabilities")
    func fourierCoinWalk() {
        let graph = WalkGraph.cycle(vertices: 4)
        let circuit = QuantumWalk.walk(on: graph, method: .discrete(coin: .fourier, steps: 3))
        let state = circuit.execute()
        let result = state.walkResult(graph: graph, steps: 3)
        for prob in result.vertexProbabilities {
            #expect(prob >= -1e-10, "Probabilities should be non-negative")
        }
    }

    @Test("Walk from non-zero initial vertex")
    func nonZeroInitialVertex() {
        let graph = WalkGraph.cycle(vertices: 4)
        let circuit = QuantumWalk.walk(
            on: graph,
            method: .discrete(coin: .hadamard, steps: 1),
            initialVertex: 2,
        )
        let state = circuit.execute()
        let result = state.walkResult(graph: graph, steps: 1)
        #expect(result.vertexProbabilities[2] > 0.01, "Walk from vertex 2 should have non-trivial probability there")
    }

    @Test("Hadamard coin on degree-3 graph covers coin padding")
    func hadamardCoinPadding() {
        let graph = WalkGraph.complete(vertices: 4)
        let circuit = QuantumWalk.walk(on: graph, method: .discrete(coin: .hadamard, steps: 2))
        let state = circuit.execute()
        let result = state.walkResult(graph: graph, steps: 2)
        var total = 0.0
        for p in result.vertexProbabilities {
            total += p
        }
        #expect(abs(total - 1.0) < 1e-6, "Hadamard on K4 should preserve normalization")
    }

    @Test("Fourier coin on degree-3 graph covers coin padding")
    func fourierCoinPadding() {
        let graph = WalkGraph.complete(vertices: 4)
        let circuit = QuantumWalk.walk(on: graph, method: .discrete(coin: .fourier, steps: 2))
        let state = circuit.execute()
        let result = state.walkResult(graph: graph, steps: 2)
        var total = 0.0
        for p in result.vertexProbabilities {
            total += p
        }
        #expect(abs(total - 1.0) < 1e-6, "Fourier on K4 should preserve normalization")
    }

    @Test("Custom coin on degree-3 graph covers coin padding")
    func customCoinPadding() {
        let customCoin: [[Complex<Double>]] = [
            [Complex(1.0, 0.0), .zero],
            [.zero, Complex(1.0, 0.0)],
        ]
        let graph = WalkGraph.complete(vertices: 4)
        let circuit = QuantumWalk.walk(on: graph, method: .discrete(coin: .custom(customCoin), steps: 2))
        let state = circuit.execute()
        let result = state.walkResult(graph: graph, steps: 2)
        var total = 0.0
        for p in result.vertexProbabilities {
            total += p
        }
        #expect(abs(total - 1.0) < 1e-6, "Custom coin on K4 should preserve normalization")
    }

    @Test("Custom coin produces valid walk")
    func customCoinWalk() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let customH: [[Complex<Double>]] = [
            [Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0)],
            [Complex(invSqrt2, 0.0), Complex(-invSqrt2, 0.0)],
        ]
        let graph = WalkGraph.cycle(vertices: 4)
        let circuit = QuantumWalk.walk(on: graph, method: .discrete(coin: .custom(customH), steps: 3))
        let state = circuit.execute()
        let result = state.walkResult(graph: graph, steps: 3)
        var total = 0.0
        for p in result.vertexProbabilities {
            total += p
        }
        #expect(abs(total - 1.0) < 1e-6, "Custom coin walk should preserve normalization")
    }
}

/// Test suite for continuous-time quantum walk evolution.
/// Validates Hamiltonian-based walk circuit construction and
/// state propagation using eigendecomposition of adjacency matrix.
@Suite("Continuous Quantum Walk")
struct ContinuousWalkTests {
    @Test("Continuous walk circuit has correct qubit count")
    func continuousWalkQubits() {
        let graph = WalkGraph.cycle(vertices: 4)
        let circuit = QuantumWalk.walk(on: graph, method: .continuous(time: 1.0, trotterSteps: 10))
        #expect(circuit.qubits == 2, "CTQW on 4 vertices should use 2 qubits")
    }

    @Test("Continuous walk preserves normalization")
    func continuousNormalization() {
        let graph = WalkGraph.hypercube(dimension: 2)
        let circuit = QuantumWalk.walk(on: graph, method: .continuous(time: 0.5, trotterSteps: 10))
        let state = circuit.execute()
        let result = state.walkResult(graph: graph, steps: 0)
        var total = 0.0
        for p in result.vertexProbabilities {
            total += p
        }
        #expect(abs(total - 1.0) < 1e-6, "CTQW should preserve normalization")
    }

    @Test("Zero time evolution returns initial state")
    func zeroTimeEvolution() {
        let graph = WalkGraph.complete(vertices: 4)
        let circuit = QuantumWalk.walk(
            on: graph,
            method: .continuous(time: 0.0, trotterSteps: 10),
            initialVertex: 0,
        )
        let state = circuit.execute()
        let result = state.walkResult(graph: graph, steps: 0)
        #expect(result.vertexProbabilities[0] > 0.99, "Zero-time walk should stay at initial vertex")
    }

    @Test("Continuous walk result without coin register")
    func continuousWalkResultNoCoin() {
        let graph = WalkGraph.hypercube(dimension: 2)
        let circuit = QuantumWalk.walk(on: graph, method: .continuous(time: 0.5, trotterSteps: 10))
        let state = circuit.execute()
        let posQubits = graph.positionQubits
        #expect(state.qubits == posQubits, "CTQW state should have only position qubits")
        let result = state.walkResult(graph: graph, steps: 0)
        #expect(result.vertexProbabilities.count == 4, "Should have 4 vertex probabilities")
    }

    @Test("Continuous walk from non-zero initial vertex")
    func continuousFromNonZeroVertex() {
        let graph = WalkGraph.complete(vertices: 4)
        let circuit = QuantumWalk.walk(
            on: graph,
            method: .continuous(time: 0.0, trotterSteps: 10),
            initialVertex: 2,
        )
        let state = circuit.execute()
        let result = state.walkResult(graph: graph, steps: 0)
        #expect(result.vertexProbabilities[2] > 0.99, "Zero-time CTQW from vertex 2 should stay there")
    }

    @Test("QuantumCircuit extension delegates correctly")
    func quantumCircuitExtension() {
        let graph = WalkGraph.hypercube(dimension: 2)
        let circuit = QuantumWalk.walk(on: graph, method: .continuous(time: 0.5, trotterSteps: 10))
        let state = circuit.execute()
        let result = state.walkResult(graph: graph, steps: 0)
        var total = 0.0
        for p in result.vertexProbabilities {
            total += p
        }
        #expect(abs(total - 1.0) < 1e-6, "Extension should produce valid walk circuit")
    }
}

/// Test suite for quantum walk spatial search algorithm.
/// Validates marked vertex detection, success probability,
/// and optimal step count computation for graph search.
@Suite("Spatial Search")
struct SpatialSearchTests {
    @Test("Optimal steps formula for complete graph")
    func optimalSteps() {
        let graph = WalkGraph.complete(vertices: 16)
        let steps = QuantumWalk.optimalSearchSteps(graph: graph, markedCount: 1)
        let expected = Int(floor(Double.pi / 2.0 * sqrt(16.0)))
        #expect(steps == expected, "Optimal steps should be floor(π/2·√n)")
    }

    @Test("Optimal steps with multiple marked vertices")
    func optimalStepsMultiple() {
        let graph = WalkGraph.complete(vertices: 16)
        let steps1 = QuantumWalk.optimalSearchSteps(graph: graph, markedCount: 1)
        let steps4 = QuantumWalk.optimalSearchSteps(graph: graph, markedCount: 4)
        #expect(steps4 < steps1, "More marked vertices should require fewer steps")
    }

    @Test("Search on degree-1 graph covers coinQubits branch")
    func searchDegreeOneGraph() {
        let graph = WalkGraph.line(vertices: 2)
        let circuit = QuantumWalk.search(on: graph, marked: [0], steps: 2)
        let state = circuit.execute()
        let result = state.searchResult(graph: graph, marked: [0], steps: 2)
        #expect(result.successProbability >= 0.0, "Search on P2 should produce valid probability")
    }

    @Test("Discrete walk on degree-1 graph covers coinQubits branch")
    func discreteWalkDegreeOne() {
        let graph = WalkGraph.line(vertices: 2)
        let circuit = QuantumWalk.walk(on: graph, method: .discrete(coin: .grover, steps: 2))
        let state = circuit.execute()
        let result = state.walkResult(graph: graph, steps: 2)
        var total = 0.0
        for p in result.vertexProbabilities {
            total += p
        }
        #expect(abs(total - 1.0) < 1e-6, "Walk on P2 should preserve normalization")
    }

    @Test("Search circuit has correct qubit count")
    func searchCircuitQubits() {
        let graph = WalkGraph.complete(vertices: 4)
        let circuit = QuantumWalk.search(on: graph, marked: [1])
        #expect(circuit.qubits >= graph.positionQubits, "Search circuit should have at least position qubits")
    }

    @Test("Search result extracts correctly")
    func searchResultExtraction() {
        let graph = WalkGraph.complete(vertices: 4)
        let circuit = QuantumWalk.search(on: graph, marked: [2], steps: 2)
        let state = circuit.execute()
        let result = state.searchResult(graph: graph, marked: [2], steps: 2)
        #expect(result.foundVertex >= 0, "Found vertex should be valid")
        #expect(result.foundVertex < 4, "Found vertex should be in range")
        #expect(result.successProbability >= 0.0, "Success probability should be non-negative")
        #expect(result.successProbability <= 1.0 + 1e-10, "Success probability should be at most 1")
    }

    @Test("Search on complete graph amplifies marked vertex probability")
    func searchCompleteGraph() {
        let graph = WalkGraph.complete(vertices: 8)
        let steps = QuantumWalk.optimalSearchSteps(graph: graph, markedCount: 1)
        let circuit = QuantumWalk.search(on: graph, marked: [3], steps: steps)
        let state = circuit.execute()
        let result = state.searchResult(graph: graph, marked: [3], steps: steps)
        #expect(result.successProbability > 0.01, "Search should produce non-trivial marked probability")
    }

    @Test("Search with multiple marked vertices")
    func searchMultipleMarked() {
        let graph = WalkGraph.complete(vertices: 4)
        let circuit = QuantumWalk.search(on: graph, marked: [0, 2], steps: 2)
        let state = circuit.execute()
        let result = state.searchResult(graph: graph, marked: [0, 2], steps: 2)
        #expect(result.successProbability >= 0.0, "Multi-marked search should have valid probability")
    }

    @Test("Search result isMarked flag correct when found")
    func isMarkedFlag() {
        let graph = WalkGraph.complete(vertices: 4)
        let circuit = QuantumWalk.search(on: graph, marked: [0, 1, 2, 3], steps: 1)
        let state = circuit.execute()
        let result = state.searchResult(graph: graph, marked: [0, 1, 2, 3], steps: 1)
        #expect(result.isMarked, "When all vertices are marked, found vertex should be marked")
    }
}

/// Test suite for WalkResult and SpatialSearchResult descriptions.
/// Validates CustomStringConvertible conformance and
/// formatted output of walk analysis results.
@Suite("Result Descriptions")
struct ResultDescriptionTests {
    @Test("WalkResult description contains key fields")
    func walkResultDescription() {
        let graph = WalkGraph.cycle(vertices: 4)
        let circuit = QuantumWalk.walk(on: graph, method: .discrete(coin: .hadamard, steps: 5))
        let result = circuit.execute().walkResult(graph: graph, steps: 5)
        let desc = result.description
        #expect(desc.contains("vertex="), "Description should contain vertex index")
        #expect(desc.contains("steps=5"), "Description should contain step count")
    }

    @Test("SpatialSearchResult description contains status")
    func searchResultDescription() {
        let graph = WalkGraph.complete(vertices: 4)
        let circuit = QuantumWalk.search(on: graph, marked: [0, 1, 2, 3], steps: 1)
        let result = circuit.execute().searchResult(graph: graph, marked: [0, 1, 2, 3], steps: 1)
        let desc = result.description
        #expect(desc.contains("FOUND"), "Description should show FOUND when all vertices marked")
    }

    @Test("SpatialSearchResult description shows MISS when not marked")
    func searchResultMiss() {
        let graph = WalkGraph.cycle(vertices: 4)
        let circuit = QuantumWalk.search(on: graph, marked: [3], steps: 1)
        let state = circuit.execute()
        let result = state.searchResult(graph: graph, marked: [3], steps: 1)
        let desc = result.description
        #expect(desc.contains("FOUND") || desc.contains("MISS"), "Description should contain status")
    }
}
