// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// MaxCut problem Hamiltonian constructors for QAOA
///
/// Provides cost Hamiltonian construction for the Maximum Cut (MaxCut) problem.
/// MaxCut is a classic NP-hard graph partitioning problem and the **canonical
/// benchmark** for QAOA algorithms.
///
/// **Problem Definition:**
/// Given an undirected graph G = (V, E), partition vertices into two sets S and T
/// such that the number of edges crossing the partition is maximized.
///
/// **Mathematical Formulation:**
/// - Cost function: C(z) = Σ_{(i,j)∈E} ½(1 - zᵢzⱼ) where z ∈ {±1}^n
/// - Quantum encoding: zᵢ = 2·qᵢ - 1 where qᵢ ∈ {0,1}
/// - Hamiltonian: H_p = Σ_{(i,j)∈E} ½(1 - ZᵢZⱼ) = Σ_{(i,j)∈E} ½(I - ZᵢZⱼ)
///
/// **Simplified form (drop constant):**
/// H_p = -½ Σ_{(i,j)∈E} ZᵢZⱼ  (minimization equivalent to MaxCut)
///
/// **Properties:**
/// - Minimizing H_p <-> Maximizing cut size
/// - Ground state energy: E₀ = -maxcut/2
/// - MaxCut value: maxcut = -2·E₀
///
/// **Example - 4-vertex square graph:**
/// ```swift
/// // Square graph: 0---1
/// //               |   |
/// //               3---2
/// let edges = [(0,1), (1,2), (2,3), (3,0)]
/// let hamiltonian = MaxCut.hamiltonian(edges: edges)
///
/// // Hamiltonian: H = -½(Z₀Z₁ + Z₁Z₂ + Z₂Z₃ + Z₃Z₀)
/// // Optimal cut: {0,2} vs {1,3} -> 4 edges cut
/// // Ground state energy: E₀ = -2.0
/// // Verification: maxcut = -2·E₀ = 4 ✓
///
/// // Run QAOA
/// let qaoa = await QAOA(
///     costHamiltonian: hamiltonian,
///     numQubits: 4,
///     depth: 2,
///     optimizer: COBYLAOptimizer(tolerance: 1e-6)
/// )
///
/// let result = await qaoa.run(initialParameters: [0.5, 0.5, 0.5, 0.5])
/// let maxcutValue = Int(-2.0 * result.optimalEnergy)
/// print("MaxCut value: \(maxcutValue)")  // 4
/// ```
///
/// **Example - Triangle graph (K₃):**
/// ```swift
/// // Complete graph on 3 vertices
/// let edges = [(0,1), (1,2), (0,2)]
/// let hamiltonian = MaxCut.hamiltonian(edges: edges)
///
/// // Optimal cut: any single vertex vs others -> 2 edges
/// // Ground state energy: E₀ = -1.0
/// // MaxCut value: 2
/// ```
///
/// **Applications:**
/// - Network design: Minimize communication cost between modules
/// - VLSI design: Circuit partitioning for placement
/// - Image segmentation: Partition pixels into regions
/// - Community detection: Graph clustering
/// - Benchmark for quantum optimization algorithms
@frozen
public struct MaxCut {
    /// Create MaxCut cost Hamiltonian from graph edges
    ///
    /// Constructs H_p = -½ Σ_{(i,j)∈E} ZᵢZⱼ for MaxCut optimization.
    /// Minimizing this Hamiltonian is equivalent to maximizing the cut size.
    ///
    /// **Edge encoding:**
    /// - Each edge (i,j) contributes term: -0.5·Z_i⊗Z_j
    /// - Coefficient -0.5 ensures: min H_p <-> max cut
    /// - ZᵢZⱼ = +1 if qubits in same state (not cut)
    /// - ZᵢZⱼ = -1 if qubits in different states (cut)
    ///
    /// **Hamiltonian structure:**
    /// - Number of terms: |E| (one per edge)
    /// - Sparsity: Depends on graph connectivity
    /// - Dense graphs (many edges) -> more terms -> slower QAOA
    ///
    /// **Complexity:**
    /// - Construction: O(|E|) time and space
    /// - QAOA circuit: O(|E|) gates per layer (one ZZ rotation per edge)
    ///
    /// - Parameter edges: Array of undirected edges as (vertex_i, vertex_j) pairs
    /// - Returns: Observable representing MaxCut cost Hamiltonian
    ///
    /// **Validation:**
    /// - Edges must reference valid qubit indices (≥ 0, < 30 memory limit)
    /// - Duplicate edges allowed (combined into single term automatically)
    /// - Edge order doesn't matter: (i,j) ≡ (j,i)
    ///
    /// Example:
    /// ```swift
    /// // Path graph: 0---1---2---3
    /// let edges = [(0,1), (1,2), (2,3)]
    /// let hamiltonian = MaxCut.hamiltonian(edges: edges)
    ///
    /// // Result: H = -0.5·Z₀Z₁ - 0.5·Z₁Z₂ - 0.5·Z₂Z₃
    /// print(hamiltonian.terms.count)  // 3 terms
    ///
    /// // Optimal cuts: {0,2} vs {1,3} -> 3 edges cut
    /// // Ground state energy: E₀ = -1.5
    /// // MaxCut value: 3
    ///
    /// // Use with QAOA
    /// let qaoa = await QAOA(
    ///     costHamiltonian: hamiltonian,
    ///     numQubits: 4,
    ///     depth: 3
    /// )
    /// ```
    @_optimize(speed)
    @_eagerMove
    public static func hamiltonian(edges: [(Int, Int)]) -> Observable {
        ValidationUtilities.validateNonEmpty(edges, name: "edges")

        var terms: PauliTerms = []
        terms.reserveCapacity(edges.count)

        for (i, j) in edges {
            ValidationUtilities.validateNonNegativeInt(i, name: "edge vertex i")
            ValidationUtilities.validateNonNegativeInt(j, name: "edge vertex j")
            ValidationUtilities.validateMemoryLimit(max(i, j) + 1)
            ValidationUtilities.validateDistinctVertices(i, j)

            // H_p term: -0.5 * Z_i⊗Z_j
            // Sorted order ensures (i,j) and (j,i) produce identical PauliStrings
            let vertex1 = min(i, j)
            let vertex2 = max(i, j)
            let pauliString = PauliString(operators: [
                (qubit: vertex1, basis: .z),
                (qubit: vertex2, basis: .z),
            ])

            terms.append((coefficient: -0.5, pauliString: pauliString))
        }

        return Observable(terms: terms)
    }

    /// Create example graphs for testing and demonstration
    ///
    /// Provides common graph topologies with known MaxCut solutions for
    /// algorithm validation and benchmarking.
    @frozen
    public struct Examples {
        /// Triangle graph (K₃): 3 vertices, 3 edges
        ///
        /// Complete graph on 3 vertices. Any partition has maxcut = 2.
        ///
        /// - Returns: Edges for triangle graph
        ///
        /// Example:
        /// ```swift
        /// let edges = MaxCut.Examples.triangle()
        /// let hamiltonian = MaxCut.hamiltonian(edges: edges)
        /// // Expected maxcut: 2
        /// // Expected E₀: -1.0
        /// ```
        @inlinable
        public static func triangle() -> [(Int, Int)] {
            [(0, 1), (1, 2), (0, 2)]
        }

        /// Square graph: 4 vertices in cycle
        ///
        /// Cycle graph C₄. Optimal cut partitions opposite vertices: maxcut = 4.
        ///
        /// - Returns: Edges for square graph
        ///
        /// Example:
        /// ```swift
        /// let edges = MaxCut.Examples.square()
        /// let hamiltonian = MaxCut.hamiltonian(edges: edges)
        /// // Expected maxcut: 4
        /// // Expected E₀: -2.0
        /// // Optimal partitions: {0,2} vs {1,3}
        /// ```
        @inlinable
        public static func square() -> [(Int, Int)] {
            [(0, 1), (1, 2), (2, 3), (3, 0)]
        }

        /// Pentagon graph: 5 vertices in cycle
        ///
        /// Cycle graph C₅. Optimal cut has maxcut = 4 (any partition with 2 vs 3 vertices).
        ///
        /// - Returns: Edges for pentagon graph
        ///
        /// Example:
        /// ```swift
        /// let edges = MaxCut.Examples.pentagon()
        /// let hamiltonian = MaxCut.hamiltonian(edges: edges)
        /// // Expected maxcut: 4
        /// // Expected E₀: -2.0
        /// ```
        @inlinable
        public static func pentagon() -> [(Int, Int)] {
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        }

        /// Complete graph K₄: 4 vertices, all pairs connected
        ///
        /// Every pair of vertices has an edge (6 edges total).
        /// Optimal cut partitions into equal sets: maxcut = 4.
        ///
        /// - Returns: Edges for K₄ complete graph
        ///
        /// Example:
        /// ```swift
        /// let edges = MaxCut.Examples.completeK4()
        /// let hamiltonian = MaxCut.hamiltonian(edges: edges)
        /// // Expected maxcut: 4
        /// // Expected E₀: -2.0
        /// // Optimal partitions: {0,1} vs {2,3} or similar 2-2 split
        /// ```
        @inlinable
        public static func completeK4() -> [(Int, Int)] {
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        }

        /// Linear chain: n vertices in path
        ///
        /// Path graph: 0---1---2---...---(n-1)
        /// Optimal cut alternates vertices: maxcut = n-1 (entire path cut).
        ///
        /// - Parameter numVertices: Number of vertices in path (≥ 2)
        /// - Returns: Edges for linear chain
        ///
        /// Example:
        /// ```swift
        /// let edges = MaxCut.Examples.linearChain(numVertices: 6)
        /// let hamiltonian = MaxCut.hamiltonian(edges: edges)
        /// // 6 vertices: 0---1---2---3---4---5
        /// // Expected maxcut: 5 (all edges cut)
        /// // Expected E₀: -2.5
        /// // Optimal partition: {0,2,4} vs {1,3,5}
        /// ```
        @_optimize(speed)
        @_eagerMove
        public static func linearChain(numVertices: Int) -> [(Int, Int)] {
            ValidationUtilities.validateLowerBound(numVertices, min: 2, name: "numVertices")

            let edgeCount = numVertices - 1
            return [(Int, Int)](unsafeUninitializedCapacity: edgeCount) { buffer, count in
                for i in 0 ..< edgeCount {
                    buffer[i] = (i, i + 1)
                }
                count = edgeCount
            }
        }

        /// Star graph: central vertex connected to all others
        ///
        /// One central vertex (0) connected to n-1 peripheral vertices.
        /// Optimal cut isolates center: maxcut = n-1.
        ///
        /// - Parameter numVertices: Total vertices including center (≥ 2)
        /// - Returns: Edges for star graph
        ///
        /// Example:
        /// ```swift
        /// let edges = MaxCut.Examples.star(numVertices: 5)
        /// let hamiltonian = MaxCut.hamiltonian(edges: edges)
        /// // Center (0) connected to {1,2,3,4}
        /// // Expected maxcut: 4
        /// // Expected E₀: -2.0
        /// // Optimal partition: {0} vs {1,2,3,4}
        /// ```
        @_optimize(speed)
        @_eagerMove
        public static func star(numVertices: Int) -> [(Int, Int)] {
            ValidationUtilities.validateLowerBound(numVertices, min: 2, name: "numVertices")

            let edgeCount = numVertices - 1
            return [(Int, Int)](unsafeUninitializedCapacity: edgeCount) { buffer, count in
                for i in 0 ..< edgeCount {
                    buffer[i] = (0, i + 1)
                }
                count = edgeCount
            }
        }
    }
}
