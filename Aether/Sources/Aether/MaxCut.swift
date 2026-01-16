// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// MaxCut cost Hamiltonian construction for QAOA.
///
/// Given an undirected graph G = (V, E), the MaxCut problem seeks a partition
/// of vertices into two sets that maximizes edges crossing the partition. This
/// classic NP-hard problem serves as the canonical benchmark for QAOA algorithms.
///
/// The cost Hamiltonian H = -½ Σ_{(i,j)∈E} ZᵢZⱼ encodes the objective such that
/// minimizing H is equivalent to maximizing the cut size. The ground state energy
/// E₀ relates to the optimal cut by: maxcut = -2·E₀.
///
/// **Example:**
/// ```swift
/// let hamiltonian = MaxCut.hamiltonian(edges: [(0, 1), (1, 2), (2, 0)])
/// let qaoa = await QAOA(cost: hamiltonian, mixer: MixerHamiltonian.x(qubits: 3), qubits: 3, depth: 2)
/// let result = await qaoa.run(from: [0.5, 0.5, 0.5, 0.5])
/// ```
///
/// - SeeAlso: ``QAOA``
/// - SeeAlso: ``Observable``
/// - SeeAlso: ``MixerHamiltonian``
public enum MaxCut {
    private static let zzCoefficient = -0.5

    /// Creates a MaxCut cost Hamiltonian from graph edges.
    ///
    /// Each edge (i, j) contributes a term -0.5·ZᵢZⱼ to the Hamiltonian. The ZZ
    /// operator yields +1 when qubits share the same state (edge not cut) and -1
    /// when they differ (edge cut), making minimization equivalent to maximizing
    /// the cut. Edge order is normalized internally so (i, j) and (j, i) produce
    /// identical terms.
    ///
    /// **Example:**
    /// ```swift
    /// let hamiltonian = MaxCut.hamiltonian(edges: [(0, 1), (1, 2), (2, 3)])
    /// ```
    ///
    /// - Parameter edges: Undirected edges as vertex pairs with non-negative indices
    /// - Returns: Observable with one ZZ term per edge, coefficient -0.5
    /// - Complexity: O(|E|) time and space
    /// - Precondition: Vertices must be non-negative and below the 30-qubit memory limit
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func hamiltonian(edges: [(Int, Int)]) -> Observable {
        ValidationUtilities.validateNonEmpty(edges, name: "edges")

        var terms: PauliTerms = []
        terms.reserveCapacity(edges.count)

        for (i, j) in edges {
            ValidationUtilities.validateNonNegativeInt(i, name: "edge vertex i")
            ValidationUtilities.validateNonNegativeInt(j, name: "edge vertex j")
            ValidationUtilities.validateMemoryLimit(max(i, j) + 1)
            ValidationUtilities.validateDistinctVertices(i, j)

            let vertex1 = min(i, j)
            let vertex2 = max(i, j)
            let pauliString = PauliString(.z(vertex1), .z(vertex2))

            terms.append((coefficient: zzCoefficient, pauliString: pauliString))
        }

        return Observable(terms: terms)
    }

    /// Standard graph topologies with known MaxCut solutions for testing and benchmarking.
    public enum Examples {
        /// Triangle graph K₃ with 3 vertices and 3 edges. Optimal maxcut = 2, E₀ = -1.0.
        @inlinable
        @_effects(readonly)
        public static func triangle() -> [(Int, Int)] {
            [(0, 1), (1, 2), (0, 2)]
        }

        /// Square cycle graph C₄ with 4 vertices and 4 edges. Optimal maxcut = 4, E₀ = -2.0.
        @inlinable
        @_effects(readonly)
        public static func square() -> [(Int, Int)] {
            [(0, 1), (1, 2), (2, 3), (3, 0)]
        }

        /// Pentagon cycle graph C₅ with 5 vertices and 5 edges. Optimal maxcut = 4, E₀ = -2.0.
        @inlinable
        @_effects(readonly)
        public static func pentagon() -> [(Int, Int)] {
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        }

        /// Complete graph K₄ with 4 vertices and 6 edges. Optimal maxcut = 4, E₀ = -2.0.
        @inlinable
        @_effects(readonly)
        public static func complete4() -> [(Int, Int)] {
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        }

        /// Path graph with n vertices and n-1 edges. Optimal maxcut = n-1, E₀ = -(n-1)/2.
        ///
        /// - Parameter vertices: Number of vertices in path (≥ 2)
        @_optimize(speed)
        @_eagerMove
        @_effects(readonly)
        public static func linearChain(vertices: Int) -> [(Int, Int)] {
            ValidationUtilities.validateLowerBound(vertices, min: 2, name: "vertices")

            let edgeCount = vertices - 1
            return [(Int, Int)](unsafeUninitializedCapacity: edgeCount) { buffer, count in
                for i in 0 ..< edgeCount {
                    buffer[i] = (i, i + 1)
                }
                count = edgeCount
            }
        }

        /// Star graph with central vertex 0 connected to n-1 peripheral vertices.
        /// Optimal maxcut = n-1, E₀ = -(n-1)/2.
        ///
        /// - Parameter vertices: Total vertices including center (≥ 2)
        @_optimize(speed)
        @_eagerMove
        @_effects(readonly)
        public static func star(vertices: Int) -> [(Int, Int)] {
            ValidationUtilities.validateLowerBound(vertices, min: 2, name: "vertices")

            let edgeCount = vertices - 1
            return [(Int, Int)](unsafeUninitializedCapacity: edgeCount) { buffer, count in
                for i in 0 ..< edgeCount {
                    buffer[i] = (0, i + 1)
                }
                count = edgeCount
            }
        }

        /// Cycle graph Cₙ with n vertices and n edges forming a ring.
        /// Generalizes `square()` (C₄) and `pentagon()` (C₅).
        /// For even n: maxcut = n, E₀ = -n/2. For odd n: maxcut = n-1, E₀ = -(n-1)/2.
        ///
        /// - Parameter vertices: Number of vertices in cycle (≥ 3)
        @_optimize(speed)
        @_eagerMove
        @_effects(readonly)
        public static func cycle(vertices: Int) -> [(Int, Int)] {
            ValidationUtilities.validateLowerBound(vertices, min: 3, name: "vertices")

            return [(Int, Int)](unsafeUninitializedCapacity: vertices) { buffer, count in
                for i in 0 ..< vertices - 1 {
                    buffer[i] = (i, i + 1)
                }
                buffer[vertices - 1] = (vertices - 1, 0)
                count = vertices
            }
        }

        /// Complete graph Kₙ with n vertices and n(n-1)/2 edges.
        /// Generalizes `triangle()` (K₃) and `complete4()` (K₄).
        /// Optimal maxcut = ⌊n²/4⌋, E₀ = -⌊n²/4⌋/2.
        ///
        /// - Parameter vertices: Number of vertices (≥ 2)
        @_optimize(speed)
        @_eagerMove
        @_effects(readonly)
        public static func complete(vertices: Int) -> [(Int, Int)] {
            ValidationUtilities.validateLowerBound(vertices, min: 2, name: "vertices")

            let edgeCount = vertices * (vertices - 1) / 2
            return [(Int, Int)](unsafeUninitializedCapacity: edgeCount) { buffer, count in
                var index = 0
                for i in 0 ..< vertices {
                    for j in (i + 1) ..< vertices {
                        buffer[index] = (i, j)
                        index += 1
                    }
                }
                count = edgeCount
            }
        }
    }
}
