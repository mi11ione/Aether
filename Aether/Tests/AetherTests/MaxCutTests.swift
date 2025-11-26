// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Test suite for MaxCut problem Hamiltonian construction.
/// Validates Hamiltonian structure, edge normalization, example graphs with
/// known solutions, and error handling for invalid inputs.
@Suite("MaxCut Problem Hamiltonians")
struct MaxCutTests {
    @Test("Single edge produces correct ZZ term")
    func singleEdge() {
        let hamiltonian = MaxCut.hamiltonian(edges: [(0, 1)])

        #expect(hamiltonian.terms.count == 1)

        let (coefficient, pauliString) = hamiltonian.terms[0]
        #expect(abs(coefficient - -0.5) < 1e-10)
        #expect(pauliString.operators.count == 2)

        let qubits = pauliString.operators.map(\.qubit).sorted()
        #expect(qubits == [0, 1])
        #expect(pauliString.operators.allSatisfy { $0.basis == .z })
    }

    @Test("Multiple edges create multiple ZZ terms")
    func multipleEdges() {
        let hamiltonian = MaxCut.hamiltonian(edges: [(0, 1), (1, 2), (2, 3)])

        #expect(hamiltonian.terms.count == 3)

        for (coefficient, pauliString) in hamiltonian.terms {
            #expect(abs(coefficient - -0.5) < 1e-10)
            #expect(pauliString.operators.count == 2)
            #expect(pauliString.operators.allSatisfy { $0.basis == .z })
        }
    }

    @Test("Edge order normalization produces identical terms")
    func edgeOrderNormalization() {
        let hamiltonian1 = MaxCut.hamiltonian(edges: [(0, 1)])
        let hamiltonian2 = MaxCut.hamiltonian(edges: [(1, 0)])

        let qubits1 = hamiltonian1.terms[0].1.operators.map(\.qubit).sorted()
        let qubits2 = hamiltonian2.terms[0].1.operators.map(\.qubit).sorted()

        #expect(qubits1 == qubits2)
        #expect(qubits1 == [0, 1])
    }

    @Test("Duplicate edges create multiple terms")
    func duplicateEdges() {
        let hamiltonian = MaxCut.hamiltonian(edges: [(0, 1), (1, 0), (0, 1)])

        #expect(hamiltonian.terms.count == 3)

        for (coefficient, _) in hamiltonian.terms {
            #expect(abs(coefficient - -0.5) < 1e-10)
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
        let edges = MaxCut.Examples.triangle()
        let hamiltonian = MaxCut.hamiltonian(edges: edges)

        #expect(edges.count == 3)
        #expect(hamiltonian.terms.count == 3)

        for (coefficient, _) in hamiltonian.terms {
            #expect(abs(coefficient - -0.5) < 1e-10)
        }
    }

    @Test("Square graph has 4 edges forming cycle")
    func squareGraph() {
        let edges = MaxCut.Examples.square()
        let hamiltonian = MaxCut.hamiltonian(edges: edges)

        #expect(edges.count == 4)
        #expect(hamiltonian.terms.count == 4)
    }

    @Test("Pentagon graph has 5 edges forming cycle")
    func pentagonGraph() {
        let edges = MaxCut.Examples.pentagon()
        let hamiltonian = MaxCut.hamiltonian(edges: edges)

        #expect(edges.count == 5)
        #expect(hamiltonian.terms.count == 5)
    }

    @Test("Complete K₄ graph has 6 edges")
    func completeK4Graph() {
        let edges = MaxCut.Examples.completeK4()
        let hamiltonian = MaxCut.hamiltonian(edges: edges)

        #expect(edges.count == 6)
        #expect(hamiltonian.terms.count == 6)
    }

    @Test("Linear chain has n-1 edges")
    func linearChain() {
        let edges = MaxCut.Examples.linearChain(numVertices: 6)
        let hamiltonian = MaxCut.hamiltonian(edges: edges)

        #expect(edges.count == 5)
        #expect(hamiltonian.terms.count == 5)

        for i in 0 ..< 5 {
            #expect(edges[i] == (i, i + 1))
        }
    }

    @Test("Linear chain with minimum vertices")
    func linearChainMinimum() {
        let edges = MaxCut.Examples.linearChain(numVertices: 2)

        #expect(edges.count == 1)
        #expect(edges[0] == (0, 1))
    }

    @Test("Star graph has n-1 edges from center")
    func starGraph() {
        let edges = MaxCut.Examples.star(numVertices: 5)
        let hamiltonian = MaxCut.hamiltonian(edges: edges)

        #expect(edges.count == 4)
        #expect(hamiltonian.terms.count == 4)

        for i in 1 ..< 5 {
            #expect(edges[i - 1] == (0, i))
        }
    }

    @Test("Star graph with minimum vertices")
    func starGraphMinimum() {
        let edges = MaxCut.Examples.star(numVertices: 2)

        #expect(edges.count == 1)
        #expect(edges[0] == (0, 1))
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

        #expect(edges.count == 45)
        #expect(hamiltonian.terms.count == 45)

        for (coefficient, _) in hamiltonian.terms {
            #expect(abs(coefficient - -0.5) < 1e-10)
        }
    }

    @Test("Disconnected graph preserves all qubits")
    func disconnectedGraph() {
        let hamiltonian = MaxCut.hamiltonian(edges: [(0, 1), (2, 3)])

        #expect(hamiltonian.terms.count == 2)

        let allQubits = hamiltonian.terms.flatMap { $0.1.operators.map(\.qubit) }.sorted()
        #expect(allQubits == [0, 1, 2, 3])
    }

    @Test("Hamiltonian coefficients are real and finite")
    func coefficientsAreFinite() {
        let hamiltonian = MaxCut.hamiltonian(edges: [(0, 1), (1, 2), (2, 0)])

        for (coefficient, _) in hamiltonian.terms {
            #expect(coefficient.isFinite)
            #expect(!coefficient.isNaN)
        }
    }

    @Test("All operators use Z basis")
    func allOperatorsZ() {
        let hamiltonian = MaxCut.hamiltonian(edges: [(0, 1), (1, 2), (2, 0)])

        for (_, pauliString) in hamiltonian.terms {
            for op in pauliString.operators {
                #expect(op.basis == .z)
            }
        }
    }

    @Test("Normalized edges create identical Pauli strings")
    func normalizedEdges() {
        let hamiltonian = MaxCut.hamiltonian(edges: [(0, 1), (1, 0)])

        #expect(hamiltonian.terms.count == 2)

        let qubits1 = hamiltonian.terms[0].1.operators.map(\.qubit).sorted()
        let qubits2 = hamiltonian.terms[1].1.operators.map(\.qubit).sorted()

        #expect(qubits1 == [0, 1])
        #expect(qubits2 == [0, 1])
    }
}
