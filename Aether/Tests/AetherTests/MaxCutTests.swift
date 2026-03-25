// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Testing

/// Test suite for MaxCut problem Hamiltonian construction.
/// Validates Hamiltonian structure, edge normalization, example graphs with
/// known solutions, and error handling for invalid inputs.
@Suite("MaxCut Problem Hamiltonians")
struct MaxCutTests {
    @Test("Single edge produces correct ZZ term")
    func singleEdge() {
        let hamiltonian = MaxCut.hamiltonian(edges: [(0, 1)])

        #expect(hamiltonian.terms.count == 1, "Single edge should produce exactly one term")

        let (coefficient, pauliString) = hamiltonian.terms[0]
        #expect(abs(coefficient - -0.5) < 1e-10, "ZZ coefficient should be -0.5")
        #expect(pauliString.operators.count == 2, "ZZ term should have two operators")

        let qubits = pauliString.operators.map(\.qubit).sorted()
        #expect(qubits == [0, 1], "ZZ term should act on qubits 0 and 1")
        #expect(pauliString.operators.allSatisfy { $0.basis == .z }, "Both operators should be Z basis")
    }

    @Test("Multiple edges create multiple ZZ terms")
    func multipleEdges() {
        let hamiltonian = MaxCut.hamiltonian(edges: [(0, 1), (1, 2), (2, 3)])

        #expect(hamiltonian.terms.count == 3, "Three edges should produce three terms")

        for (coefficient, pauliString) in hamiltonian.terms {
            #expect(abs(coefficient - -0.5) < 1e-10, "Each ZZ coefficient should be -0.5")
            #expect(pauliString.operators.count == 2, "Each ZZ term should have two operators")
            #expect(pauliString.operators.allSatisfy { $0.basis == .z }, "All operators should be Z basis")
        }
    }

    @Test("Edge order normalization produces identical terms")
    func edgeOrderNormalization() {
        let hamiltonian1 = MaxCut.hamiltonian(edges: [(0, 1)])
        let hamiltonian2 = MaxCut.hamiltonian(edges: [(1, 0)])

        let qubits1 = hamiltonian1.terms[0].1.operators.map(\.qubit).sorted()
        let qubits2 = hamiltonian2.terms[0].1.operators.map(\.qubit).sorted()

        #expect(qubits1 == qubits2, "Reversed edge should produce same qubit ordering")
        #expect(qubits1 == [0, 1], "Normalized edge should act on qubits 0 and 1")
    }

    @Test("Duplicate edges create multiple terms")
    func duplicateEdges() {
        let hamiltonian = MaxCut.hamiltonian(edges: [(0, 1), (1, 0), (0, 1)])

        #expect(hamiltonian.terms.count == 3, "Duplicate edges should each produce a term")

        for (coefficient, _) in hamiltonian.terms {
            #expect(abs(coefficient - -0.5) < 1e-10, "Each duplicate term should have coefficient -0.5")
        }
    }
}

/// Test suite for example graph Hamiltonians with known MaxCut solutions.
/// Verifies structure and expected properties for triangle, square, pentagon,
/// K₄, linear chain, and star graphs.
@Suite("Example Graph MaxCut")
struct MaxCutExampleGraphsTests {
    @Test("Triangle graph has 3 edges")
    func triangleGraph() {
        let edges = MaxCut.Examples.triangle
        let hamiltonian = MaxCut.hamiltonian(edges: edges)

        #expect(edges.count == 3, "Triangle K₃ should have 3 edges")
        #expect(hamiltonian.terms.count == 3, "Triangle Hamiltonian should have 3 terms")

        for (coefficient, _) in hamiltonian.terms {
            #expect(abs(coefficient - -0.5) < 1e-10, "Triangle term coefficient should be -0.5")
        }
    }

    @Test("Square graph has 4 edges forming cycle")
    func squareGraph() {
        let edges = MaxCut.Examples.square
        let hamiltonian = MaxCut.hamiltonian(edges: edges)

        #expect(edges.count == 4, "Square C₄ should have 4 edges")
        #expect(hamiltonian.terms.count == 4, "Square Hamiltonian should have 4 terms")
    }

    @Test("Pentagon graph has 5 edges forming cycle")
    func pentagonGraph() {
        let edges = MaxCut.Examples.pentagon
        let hamiltonian = MaxCut.hamiltonian(edges: edges)

        #expect(edges.count == 5, "Pentagon C₅ should have 5 edges")
        #expect(hamiltonian.terms.count == 5, "Pentagon Hamiltonian should have 5 terms")
    }

    @Test("Complete K₄ graph has 6 edges")
    func complete4Graph() {
        let edges = MaxCut.Examples.complete4
        let hamiltonian = MaxCut.hamiltonian(edges: edges)

        #expect(edges.count == 6, "Complete K₄ should have 6 edges")
        #expect(hamiltonian.terms.count == 6, "K₄ Hamiltonian should have 6 terms")
    }

    @Test("Linear chain has n-1 edges")
    func linearChain() {
        let edges = MaxCut.Examples.linearChain(vertices: 6)
        let hamiltonian = MaxCut.hamiltonian(edges: edges)

        #expect(edges.count == 5, "6-vertex linear chain should have 5 edges")
        #expect(hamiltonian.terms.count == 5, "Linear chain Hamiltonian should have 5 terms")

        for i in 0 ..< 5 {
            #expect(edges[i] == (i, i + 1), "Edge \(i) should connect vertices \(i) and \(i + 1)")
        }
    }

    @Test("Linear chain with minimum vertices")
    func linearChainMinimum() {
        let edges = MaxCut.Examples.linearChain(vertices: 2)

        #expect(edges.count == 1, "2-vertex chain should have 1 edge")
        #expect(edges[0] == (0, 1), "Minimum chain should connect vertices 0 and 1")
    }

    @Test("Star graph has n-1 edges from center")
    func starGraph() {
        let edges = MaxCut.Examples.star(vertices: 5)
        let hamiltonian = MaxCut.hamiltonian(edges: edges)

        #expect(edges.count == 4, "5-vertex star should have 4 edges")
        #expect(hamiltonian.terms.count == 4, "Star Hamiltonian should have 4 terms")

        for i in 1 ..< 5 {
            #expect(edges[i - 1] == (0, i), "Edge should connect center vertex 0 to vertex \(i)")
        }
    }

    @Test("Star graph with minimum vertices")
    func starGraphMinimum() {
        let edges = MaxCut.Examples.star(vertices: 2)

        #expect(edges.count == 1, "2-vertex star should have 1 edge")
        #expect(edges[0] == (0, 1), "Minimum star should connect vertices 0 and 1")
    }

    @Test("Cycle graph has n edges forming ring")
    func cycleGraph() {
        let edges = MaxCut.Examples.cycle(vertices: 6)
        let hamiltonian = MaxCut.hamiltonian(edges: edges)

        #expect(edges.count == 6, "6-vertex cycle should have 6 edges")
        #expect(hamiltonian.terms.count == 6, "Cycle Hamiltonian should have 6 terms")

        for i in 0 ..< 5 {
            #expect(edges[i] == (i, i + 1), "Edge \(i) should connect consecutive vertices")
        }
        #expect(edges[5] == (5, 0), "Last edge should close the ring")
    }

    @Test("Cycle graph with minimum vertices")
    func cycleGraphMinimum() {
        let edges = MaxCut.Examples.cycle(vertices: 3)

        #expect(edges.count == 3, "3-vertex cycle should have 3 edges")
        #expect(edges[0] == (0, 1), "First edge should connect 0 to 1")
        #expect(edges[1] == (1, 2), "Second edge should connect 1 to 2")
        #expect(edges[2] == (2, 0), "Third edge should close the ring")
    }

    @Test("Cycle graph matches square for 4 vertices")
    func cycleMatchesSquare() {
        let cycleEdges = MaxCut.Examples.cycle(vertices: 4)
        let squareEdges = MaxCut.Examples.square

        #expect(cycleEdges.count == squareEdges.count, "Cycle(4) and square should have same edge count")

        for i in 0 ..< cycleEdges.count {
            #expect(cycleEdges[i] == squareEdges[i], "Edge \(i) should match between cycle(4) and square")
        }
    }

    @Test("Cycle graph matches pentagon for 5 vertices")
    func cycleMatchesPentagon() {
        let cycleEdges = MaxCut.Examples.cycle(vertices: 5)
        let pentagonEdges = MaxCut.Examples.pentagon

        #expect(cycleEdges.count == pentagonEdges.count, "Cycle(5) and pentagon should have same edge count")

        for i in 0 ..< cycleEdges.count {
            #expect(cycleEdges[i] == pentagonEdges[i], "Edge \(i) should match between cycle(5) and pentagon")
        }
    }

    @Test("Complete graph has n(n-1)/2 edges")
    func completeGraph() {
        let edges = MaxCut.Examples.complete(vertices: 5)
        let hamiltonian = MaxCut.hamiltonian(edges: edges)

        #expect(edges.count == 10, "K₅ should have 10 edges")
        #expect(hamiltonian.terms.count == 10, "K₅ Hamiltonian should have 10 terms")

        var edgeSet: Set<String> = []
        for (i, j) in edges {
            #expect(i < j, "Edges should be normalized with smaller vertex first")
            edgeSet.insert("\(i)-\(j)")
        }
        #expect(edgeSet.count == 10, "All 10 edges should be distinct")
    }

    @Test("Complete graph with minimum vertices")
    func completeGraphMinimum() {
        let edges = MaxCut.Examples.complete(vertices: 2)

        #expect(edges.count == 1, "K₂ should have 1 edge")
        #expect(edges[0] == (0, 1), "K₂ should connect vertices 0 and 1")
    }

    @Test("Complete graph matches triangle for 3 vertices")
    func completeMatchesTriangle() {
        let completeEdges = MaxCut.Examples.complete(vertices: 3)
        let triangleEdges = MaxCut.Examples.triangle

        #expect(completeEdges.count == triangleEdges.count, "K₃ and triangle should have same edge count")

        let completeSet = Set(completeEdges.map { "\(min($0.0, $0.1))-\(max($0.0, $0.1))" })
        let triangleSet = Set(triangleEdges.map { "\(min($0.0, $0.1))-\(max($0.0, $0.1))" })

        #expect(completeSet == triangleSet, "K₃ and triangle should have identical edge sets")
    }

    @Test("Complete graph matches complete4 for 4 vertices")
    func completeMatchesComplete4() {
        let completeEdges = MaxCut.Examples.complete(vertices: 4)
        let complete4Edges = MaxCut.Examples.complete4

        #expect(completeEdges.count == complete4Edges.count, "K₄ from complete() and complete4 should have same edge count")

        let completeSet = Set(completeEdges.map { "\(min($0.0, $0.1))-\(max($0.0, $0.1))" })
        let complete4Set = Set(complete4Edges.map { "\(min($0.0, $0.1))-\(max($0.0, $0.1))" })

        #expect(completeSet == complete4Set, "K₄ from complete() and complete4 should have identical edge sets")
    }
}

/// Test suite for MaxCut edge cases and large graphs.
/// Validates handling of complete graphs, disconnected components, and
/// coefficient properties across different graph structures.
@Suite("MaxCut Edge Cases")
struct MaxCutEdgeCasesTests {
    @Test("Complete K₁₀ graph has 45 edges")
    func completeK10() {
        var edges: [(Int, Int)] = []
        for i in 0 ..< 10 {
            for j in (i + 1) ..< 10 {
                edges.append((i, j))
            }
        }

        let hamiltonian = MaxCut.hamiltonian(edges: edges)

        #expect(edges.count == 45, "K₁₀ should have 45 edges")
        #expect(hamiltonian.terms.count == 45, "K₁₀ Hamiltonian should have 45 terms")

        for (coefficient, _) in hamiltonian.terms {
            #expect(abs(coefficient - -0.5) < 1e-10, "K₁₀ coefficient should be -0.5")
        }
    }

    @Test("Disconnected graph preserves all qubits")
    func disconnectedGraph() {
        let hamiltonian = MaxCut.hamiltonian(edges: [(0, 1), (2, 3)])

        #expect(hamiltonian.terms.count == 2, "Disconnected graph with 2 edges should have 2 terms")

        let allQubits = hamiltonian.terms.flatMap { $0.1.operators.map(\.qubit) }.sorted()
        #expect(allQubits == [0, 1, 2, 3], "All four qubits should appear in the Hamiltonian")
    }

    @Test("Hamiltonian coefficients are real and finite")
    func coefficientsAreFinite() {
        let hamiltonian = MaxCut.hamiltonian(edges: [(0, 1), (1, 2), (2, 0)])

        for (coefficient, _) in hamiltonian.terms {
            #expect(coefficient.isFinite, "Coefficient should be finite")
            #expect(!coefficient.isNaN, "Coefficient should not be NaN")
        }
    }

    @Test("All operators use Z basis")
    func allOperatorsZ() {
        let hamiltonian = MaxCut.hamiltonian(edges: [(0, 1), (1, 2), (2, 0)])

        for (_, pauliString) in hamiltonian.terms {
            for op in pauliString.operators {
                #expect(op.basis == .z, "MaxCut Hamiltonian should only use Z operators")
            }
        }
    }

    @Test("Normalized edges create identical Pauli strings")
    func normalizedEdges() {
        let hamiltonian = MaxCut.hamiltonian(edges: [(0, 1), (1, 0)])

        #expect(hamiltonian.terms.count == 2, "Two edges should produce two terms")

        let qubits1 = hamiltonian.terms[0].1.operators.map(\.qubit).sorted()
        let qubits2 = hamiltonian.terms[1].1.operators.map(\.qubit).sorted()

        #expect(qubits1 == [0, 1], "First term should act on qubits 0 and 1")
        #expect(qubits2 == [0, 1], "Second term should also act on qubits 0 and 1")
    }
}
