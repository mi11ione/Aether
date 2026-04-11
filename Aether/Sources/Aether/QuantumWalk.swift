// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Graph representation for quantum walk algorithms via adjacency matrix encoding.
///
/// Stores the connectivity structure of an undirected graph as a symmetric adjacency matrix
/// in row-major flat layout. Provides factory methods for common graph families used in
/// quantum walk research (complete, cycle, hypercube, line) and conversion to
/// ``Observable`` for continuous-time quantum walk Hamiltonian evolution.
///
/// The position register encoding uses ⌈log₂(n)⌉ qubits for n vertices, with basis states
/// |0⟩ through |n-1⟩ mapping to vertices. Vertices beyond n in the 2^q-dimensional Hilbert
/// space are treated as disconnected padding.
///
/// **Example:**
/// ```swift
/// let graph = WalkGraph.cycle(vertices: 8)
/// let circuit = QuantumWalk.walk(on: graph, method: .discrete(coin: .grover, steps: 20))
/// let state = circuit.execute()
/// let result = state.walkResult(graph: graph, steps: 20)
/// ```
///
/// - SeeAlso: ``QuantumWalk``
/// - SeeAlso: ``WalkCoin``
@frozen
public struct WalkGraph: Sendable {
    private static let coefficientThreshold: Double = 1e-12

    /// Number of vertices in the graph.
    public let vertexCount: Int

    /// Flat row-major adjacency matrix with dimensions vertexCount × vertexCount.
    ///
    /// Entry at index [i * vertexCount + j] is 1.0 if vertices i and j are connected,
    /// 0.0 otherwise. The matrix is symmetric for undirected graphs.
    public let adjacencyMatrix: [Double]

    /// Creates a graph from a 2D adjacency matrix.
    ///
    /// Validates that the matrix is square and symmetric with binary entries.
    /// Self-loops (diagonal entries) are permitted.
    ///
    /// **Example:**
    /// ```swift
    /// let triangle = WalkGraph(adjacencyMatrix: [
    ///     [0, 1, 1],
    ///     [1, 0, 1],
    ///     [1, 1, 0]
    /// ])
    /// ```
    ///
    /// - Parameter adjacencyMatrix: Square symmetric matrix with 0/1 entries
    /// - Precondition: Matrix must be square and non-empty
    /// - Precondition: Matrix entries must be 0.0 or 1.0
    /// - Precondition: Matrix must be symmetric
    /// - Complexity: O(n²) where n is the number of vertices
    @_optimize(speed)
    public init(adjacencyMatrix: [[Double]]) {
        ValidationUtilities.validateSquareMatrix(adjacencyMatrix, name: "adjacencyMatrix")
        let n = adjacencyMatrix.count
        let flat = [Double](unsafeUninitializedCapacity: n * n) { buffer, count in
            for i in 0 ..< n {
                for j in 0 ..< n {
                    buffer[i * n + j] = adjacencyMatrix[i][j]
                }
            }
            count = n * n
        }
        vertexCount = n
        self.adjacencyMatrix = flat
    }

    /// Creates a graph from pre-validated flat adjacency data.
    private init(vertexCount: Int, flatAdjacency: [Double]) {
        self.vertexCount = vertexCount
        adjacencyMatrix = flatAdjacency
    }

    /// Number of qubits needed to encode vertex positions.
    ///
    /// Returns ⌈log₂(n)⌉ for n vertices, minimum 1.
    ///
    /// **Example:**
    /// ```swift
    /// let graph = WalkGraph.cycle(vertices: 8)
    /// let qubits = graph.positionQubits
    /// print(qubits)  // 3
    /// ```
    ///
    /// - Complexity: O(1)
    @inlinable
    public var positionQubits: Int {
        vertexCount <= 1 ? 1 : (Int.bitWidth - (vertexCount - 1).leadingZeroBitCount)
    }

    /// Degree of a specific vertex (number of adjacent vertices).
    ///
    /// **Example:**
    /// ```swift
    /// let graph = WalkGraph.complete(vertices: 4)
    /// let deg = graph.degree(of: 0)
    /// print(deg)  // 3
    /// ```
    ///
    /// - Parameter vertex: Vertex index in range 0..<vertexCount
    /// - Returns: Number of edges incident to vertex
    /// - Precondition: vertex must be in range 0..<vertexCount
    /// - Complexity: O(n)
    @_effects(readonly)
    public func degree(of vertex: Int) -> Int {
        ValidationUtilities.validateIndexInBounds(vertex, bound: vertexCount, name: "Vertex")
        let rowStart = vertex * vertexCount
        var deg = 0
        for j in 0 ..< vertexCount {
            if adjacencyMatrix[rowStart + j] > 0.5 {
                deg += 1
            }
        }
        return deg
    }

    /// Maximum vertex degree across all vertices.
    ///
    /// **Example:**
    /// ```swift
    /// let graph = WalkGraph.complete(vertices: 5)
    /// let maxDeg = graph.maxDegree
    /// print(maxDeg)  // 4
    /// ```
    ///
    /// - Complexity: O(n²)
    public var maxDegree: Int {
        var maxDeg = 0
        for v in 0 ..< vertexCount {
            let d = degree(of: v)
            if d > maxDeg { maxDeg = d }
        }
        return maxDeg
    }

    /// Whether all vertices have the same degree.
    ///
    /// **Example:**
    /// ```swift
    /// let cycle = WalkGraph.cycle(vertices: 6)
    /// let regular = cycle.isRegular
    /// print(regular)  // true
    /// ```
    ///
    /// - Complexity: O(n²)
    public var isRegular: Bool {
        let d0 = degree(of: 0)
        for v in 1 ..< vertexCount {
            if degree(of: v) != d0 { return false }
        }
        return true
    }

    /// Graph Laplacian matrix L = D - A in flat row-major layout.
    ///
    /// The Laplacian is positive semi-definite with smallest eigenvalue 0, corresponding to
    /// the uniform vector. Used in continuous-time quantum walk formulations where the
    /// evolution operator is exp(-iLt) instead of exp(-iAt).
    ///
    /// **Example:**
    /// ```swift
    /// let graph = WalkGraph.cycle(vertices: 4)
    /// let lap = graph.laplacian
    /// print(lap.count)  // 16
    /// ```
    ///
    /// - Complexity: O(n²)
    @_eagerMove
    public var laplacian: [Double] {
        let n = vertexCount
        return [Double](unsafeUninitializedCapacity: n * n) { buffer, count in
            for i in 0 ..< n {
                var deg = 0.0
                let rowStart = i * n
                for j in 0 ..< n {
                    deg += adjacencyMatrix[rowStart + j]
                }
                for j in 0 ..< n {
                    let idx = rowStart + j
                    buffer[idx] = (i == j) ? deg - adjacencyMatrix[idx] : -adjacencyMatrix[idx]
                }
            }
            count = n * n
        }
    }

    /// Sorted neighbor list for a vertex.
    @_effects(readonly)
    func neighbors(of vertex: Int) -> [Int] {
        let rowStart = vertex * vertexCount
        var result: [Int] = []
        for j in 0 ..< vertexCount {
            if adjacencyMatrix[rowStart + j] > 0.5 {
                result.append(j)
            }
        }
        return result
    }

    /// Converts the adjacency matrix to a Pauli Hamiltonian for continuous-time quantum walk.
    ///
    /// Decomposes the adjacency matrix A into a sum of weighted Pauli strings
    /// A = Σ_P c_P P using the trace formula c_P = Tr(PA)/2^q. The resulting ``Observable``
    /// can be passed to ``TrotterSuzuki`` or ``TimeEvolution`` for Hamiltonian simulation.
    ///
    /// For hypercube graphs, the decomposition is exact: A = X₀ + X₁ + ... + X_{d-1}.
    ///
    /// **Example:**
    /// ```swift
    /// let graph = WalkGraph.hypercube(dimension: 3)
    /// let hamiltonian = graph.toObservable()
    /// let config = TrotterConfiguration(order: .second, steps: 20)
    /// let circuit = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 3, config: config)
    /// ```
    ///
    /// - Returns: Observable representing the adjacency matrix as Pauli Hamiltonian
    /// - Complexity: O(4^q · 2^q) where q = positionQubits
    ///
    /// - SeeAlso: ``Observable``
    /// - SeeAlso: ``TrotterSuzuki``
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public func toObservable() -> Observable {
        let q = positionQubits
        let dim = 1 << q
        var terms: PauliTerms = []
        let pauliCount = 1 << (2 * q)

        for pauliIndex in 0 ..< pauliCount {
            var trace = Complex<Double>.zero
            for basisState in 0 ..< dim {
                let (phase, resultState) = QuantumWalk.applyPauliString(
                    pauliIndex: pauliIndex,
                    qubits: q,
                    basisState: basisState,
                )
                if resultState < vertexCount, basisState < vertexCount {
                    let matrixElement = adjacencyMatrix[resultState * vertexCount + basisState]
                    trace = trace + phase * Complex(matrixElement, 0.0)
                }
            }

            let coefficient = trace.real / Double(dim)
            if abs(coefficient) > Self.coefficientThreshold {
                let ops = QuantumWalk.pauliOperators(from: pauliIndex, qubits: q)
                if ops.isEmpty {
                    continue
                }
                terms.append((coefficient: coefficient, pauliString: PauliString(ops)))
            }
        }

        return Observable(terms: terms)
    }

    /// Creates a complete graph K_n where every pair of distinct vertices is connected.
    ///
    /// The complete graph on n vertices has n(n-1)/2 edges and is (n-1)-regular. Quantum
    /// walks on complete graphs exhibit optimal spatial search in O(√n) steps with the
    /// Grover coin, achieving maximum success probability.
    ///
    /// **Example:**
    /// ```swift
    /// let graph = WalkGraph.complete(vertices: 8)
    /// let deg = graph.maxDegree
    /// let circuit = QuantumWalk.search(on: graph, marked: [3])
    /// ```
    ///
    /// - Parameter vertices: Number of vertices (≥ 2)
    /// - Returns: Complete graph K_n
    /// - Precondition: vertices ≥ 2
    /// - Complexity: O(n²)
    ///
    /// - SeeAlso: ``QuantumWalk/search(on:marked:steps:)``
    @_eagerMove
    @_effects(readonly)
    public static func complete(vertices: Int) -> WalkGraph {
        ValidationUtilities.validateLowerBound(vertices, min: 2, name: "vertices")
        let n = vertices
        let flat = [Double](unsafeUninitializedCapacity: n * n) { buffer, count in
            buffer.initialize(repeating: 1.0)
            for i in 0 ..< n {
                buffer[i * n + i] = 0.0
            }
            count = n * n
        }
        return WalkGraph(vertexCount: n, flatAdjacency: flat)
    }

    /// Creates a cycle graph C_n where vertices form a ring with nearest-neighbor connections.
    ///
    /// Each vertex connects to exactly two neighbors, making the graph 2-regular. Quantum
    /// walks on cycles demonstrate ballistic spreading (σ ∝ t) compared to classical
    /// diffusive spreading (σ ∝ √t), a hallmark of quantum speedup.
    ///
    /// **Example:**
    /// ```swift
    /// let graph = WalkGraph.cycle(vertices: 8)
    /// let circuit = QuantumWalk.walk(on: graph, method: .discrete(coin: .hadamard, steps: 20))
    /// let state = circuit.execute()
    /// ```
    ///
    /// - Parameter vertices: Number of vertices (≥ 3)
    /// - Returns: Cycle graph C_n
    /// - Precondition: vertices ≥ 3
    /// - Complexity: O(n²)
    @_eagerMove
    @_effects(readonly)
    public static func cycle(vertices: Int) -> WalkGraph {
        ValidationUtilities.validateLowerBound(vertices, min: 3, name: "vertices")
        let n = vertices
        let flat = [Double](unsafeUninitializedCapacity: n * n) { buffer, count in
            buffer.initialize(repeating: 0.0)
            for i in 0 ..< n {
                let next = (i + 1) % n
                let prev = (i + n - 1) % n
                buffer[i * n + next] = 1.0
                buffer[i * n + prev] = 1.0
            }
            count = n * n
        }
        return WalkGraph(vertexCount: n, flatAdjacency: flat)
    }

    /// Creates a d-dimensional hypercube graph Q_d with 2^d vertices.
    ///
    /// The hypercube Q_d has 2^d vertices each labeled by a d-bit string. Two vertices
    /// are adjacent when their labels differ in exactly one bit, making the graph d-regular.
    /// The adjacency matrix decomposes as A = X₀ + X₁ + ... + X_{d-1} in the Pauli basis,
    /// enabling exact continuous-time quantum walk simulation via independent qubit rotations.
    ///
    /// **Example:**
    /// ```swift
    /// let cube = WalkGraph.hypercube(dimension: 3)
    /// let walk = QuantumWalk.walk(on: cube, method: .continuous(time: 1.5, trotterSteps: 20))
    /// let state = walk.execute()
    /// ```
    ///
    /// - Parameter dimension: Number of dimensions d (vertices = 2^d, 1 ≤ d ≤ 10)
    /// - Returns: Hypercube graph Q_d
    /// - Precondition: dimension in range 1...10
    /// - Complexity: O(2^d · d)
    ///
    /// - SeeAlso: ``WalkGraph/toObservable()``
    @_eagerMove
    @_effects(readonly)
    public static func hypercube(dimension: Int) -> WalkGraph {
        ValidationUtilities.validatePositiveInt(dimension, name: "dimension")
        ValidationUtilities.validateUpperBound(dimension, max: 10, name: "dimension")
        let n = 1 << dimension
        let flat = [Double](unsafeUninitializedCapacity: n * n) { buffer, count in
            buffer.initialize(repeating: 0.0)
            for v in 0 ..< n {
                for bit in 0 ..< dimension {
                    let neighbor = v ^ (1 << bit)
                    buffer[v * n + neighbor] = 1.0
                }
            }
            count = n * n
        }
        return WalkGraph(vertexCount: n, flatAdjacency: flat)
    }

    /// Creates a line (path) graph P_n with n vertices connected in sequence.
    ///
    /// The endpoints have degree 1 and interior vertices have degree 2. Line graphs are
    /// the simplest non-trivial topology for studying quantum walk dynamics, exhibiting
    /// ballistic spreading from the initial position.
    ///
    /// **Example:**
    /// ```swift
    /// let line = WalkGraph.line(vertices: 16)
    /// let circuit = QuantumWalk.walk(on: line, method: .discrete(coin: .hadamard, steps: 30))
    /// let state = circuit.execute()
    /// ```
    ///
    /// - Parameter vertices: Number of vertices (≥ 2)
    /// - Returns: Line graph P_n
    /// - Precondition: vertices ≥ 2
    /// - Complexity: O(n²)
    @_eagerMove
    @_effects(readonly)
    public static func line(vertices: Int) -> WalkGraph {
        ValidationUtilities.validateLowerBound(vertices, min: 2, name: "vertices")
        let n = vertices
        let flat = [Double](unsafeUninitializedCapacity: n * n) { buffer, count in
            buffer.initialize(repeating: 0.0)
            for i in 0 ..< n - 1 {
                buffer[i * n + (i + 1)] = 1.0
                buffer[(i + 1) * n + i] = 1.0
            }
            count = n * n
        }
        return WalkGraph(vertexCount: n, flatAdjacency: flat)
    }
}

/// Coin operator for discrete-time quantum walk step.
///
/// The coin determines the local unitary transformation applied to the internal (coin)
/// degree of freedom at each step before the conditional shift. Different coins produce
/// different walk dynamics: the Grover coin is optimal for spatial search, the Hadamard
/// coin produces asymmetric spreading on the line, and the Fourier coin gives uniform
/// phase distribution.
///
/// **Example:**
/// ```swift
/// let circuit = QuantumWalk.walk(
///     on: WalkGraph.cycle(vertices: 8),
///     method: .discrete(coin: .grover, steps: 10)
/// )
/// ```
///
/// - SeeAlso: ``QuantumWalk``
/// - SeeAlso: ``WalkMethod``
@frozen
public enum WalkCoin: Sendable {
    /// Grover diffusion coin: G = 2/d · J - I where J is the all-ones matrix.
    ///
    /// The d-dimensional Grover coin maximizes the overlap with the uniform superposition
    /// over neighbors, making it optimal for spatial search on many graph families.
    /// For degree 2, the Grover coin is equivalent to the Pauli-X gate.
    case grover

    /// Hadamard coin for degree-2 graphs (1D walks on lines and cycles).
    ///
    /// The 2×2 Hadamard matrix H = (1/√2)[[1,1],[1,-1]] produces the characteristic
    /// asymmetric probability distribution of the discrete-time quantum walk on a line.
    case hadamard

    /// Discrete Fourier transform coin: F[j,k] = exp(2πijk/d) / √d.
    ///
    /// The d-dimensional DFT coin distributes phases uniformly across directions,
    /// producing symmetric spreading on regular graphs.
    case fourier

    /// Custom unitary coin matrix for specialized walk dynamics.
    ///
    /// The matrix dimension must match the graph degree (or be padded to the next power of 2).
    case custom([[Complex<Double>]])
}

/// Walk evolution method selecting discrete-time (coin-based) or continuous-time (Hamiltonian-based) dynamics.
///
/// Discrete-time walks apply a coin operator followed by a conditional shift at each step,
/// producing the evolution W^t = (S·(I⊗C))^t. Continuous-time walks evolve under the graph
/// adjacency or Laplacian Hamiltonian via exp(-iAt), implemented through Trotter-Suzuki
/// decomposition or exact diagonalization.
///
/// **Example:**
/// ```swift
/// let graph = WalkGraph.cycle(vertices: 8)
/// let discrete = WalkMethod.discrete(coin: .grover, steps: 20)
/// let continuous = WalkMethod.continuous(time: 1.5, trotterSteps: 20)
/// ```
///
/// - SeeAlso: ``QuantumWalk``
/// - SeeAlso: ``WalkCoin``
@frozen
public enum WalkMethod: Sendable {
    /// Discrete-time quantum walk with specified coin and step count.
    ///
    /// - Parameters:
    ///   - coin: Coin operator applied at each vertex
    ///   - steps: Number of walk steps to apply
    case discrete(coin: WalkCoin, steps: Int)

    /// Continuous-time quantum walk with Hamiltonian evolution.
    ///
    /// - Parameters:
    ///   - time: Total evolution time t in exp(-iAt)
    ///   - trotterSteps: Number of Trotter-Suzuki decomposition steps
    case continuous(time: Double, trotterSteps: Int)
}

/// Result of quantum walk evolution with vertex probability distribution.
///
/// Contains the probability of finding the walker at each vertex after the walk,
/// computed by tracing out the coin register (for discrete-time walks) or directly
/// from the position register (for continuous-time walks).
///
/// **Example:**
/// ```swift
/// let graph = WalkGraph.cycle(vertices: 8)
/// let circuit = QuantumWalk.walk(on: graph, method: .discrete(coin: .grover, steps: 20))
/// let result = circuit.execute().walkResult(graph: graph, steps: 20)
/// ```
///
/// - SeeAlso: ``QuantumWalk``
/// - SeeAlso: ``QuantumState/walkResult(graph:steps:)``
@frozen
public struct WalkResult: Sendable, CustomStringConvertible {
    /// Probability of finding the walker at each vertex, indexed by vertex number.
    public let vertexProbabilities: [Double]

    /// Vertex with highest occupation probability.
    public let mostProbableVertex: Int

    /// Number of walk steps applied.
    public let steps: Int

    /// Human-readable summary of the walk result.
    ///
    /// **Example:**
    /// ```swift
    /// let result = WalkResult(vertexProbabilities: [0.5, 0.0, 0.5], mostProbableVertex: 0, steps: 10)
    /// let text = result.description
    /// print(text)
    /// ```
    @inlinable
    public var description: String {
        "WalkResult(vertex=\(mostProbableVertex), " +
            "prob=\(String(format: "%.4f", vertexProbabilities[mostProbableVertex])), " +
            "steps=\(steps))"
    }
}

/// Result of quantum walk spatial search for marked vertices.
///
/// Contains the measurement outcome, success probability, and comparison to the theoretically
/// optimal step count. The spatial search algorithm modifies the walk coin at marked vertices
/// (replacing C with -I) to amplify the probability of finding the target, analogous to
/// Grover's algorithm on structured graphs.
///
/// **Example:**
/// ```swift
/// let graph = WalkGraph.complete(vertices: 16)
/// let circuit = QuantumWalk.search(on: graph, marked: [7])
/// let result = circuit.execute().searchResult(graph: graph, marked: [7], steps: 3)
/// ```
///
/// - SeeAlso: ``QuantumWalk/search(on:marked:steps:)``
@frozen
public struct SpatialSearchResult: Sendable, CustomStringConvertible {
    /// Most probable vertex from measurement.
    public let foundVertex: Int

    /// Total probability mass on marked vertices.
    public let successProbability: Double

    /// Number of walk steps applied.
    public let steps: Int

    /// Theoretically optimal step count for this graph and marked set.
    public let optimalSteps: Int

    /// Whether the found vertex is in the marked set.
    public let isMarked: Bool

    /// Human-readable summary of the search result.
    ///
    /// **Example:**
    /// ```swift
    /// let result = SpatialSearchResult(
    ///     foundVertex: 7, successProbability: 0.95,
    ///     steps: 3, optimalSteps: 3, isMarked: true
    /// )
    /// print(result.description)
    /// ```
    @inlinable
    public var description: String {
        let status = isMarked ? "FOUND" : "MISS"
        return "SpatialSearchResult(\(status): vertex=\(foundVertex), " +
            "prob=\(String(format: "%.4f", successProbability)), " +
            "steps=\(steps)/\(optimalSteps))"
    }
}

/// Quantum walk algorithms on graphs for spatial search and state transfer.
///
/// Implements discrete-time quantum walks (coin + conditional shift framework) and
/// continuous-time quantum walks (Hamiltonian evolution under adjacency matrix) on
/// arbitrary graph topologies. Discrete-time walks achieve quadratic speedup O(√n)
/// for spatial search on many graph families including complete graphs, hypercubes,
/// and lattices of dimension d ≥ 5. Continuous-time walks on specific structures like
/// glued trees yield exponential speedups over classical random walks.
///
/// The discrete-time walk step W = S·(I⊗C) combines a coin operator C acting on the
/// internal degree of freedom with a conditional shift S that moves the walker along
/// edges based on the coin state. The continuous-time walk evolves the state under
/// |ψ(t)⟩ = exp(-iAt)|ψ(0)⟩ for adjacency matrix A.
///
/// **Example:**
/// ```swift
/// let graph = WalkGraph.complete(vertices: 16)
/// let circuit = QuantumWalk.search(on: graph, marked: [5])
/// let state = circuit.execute()
/// let result = state.searchResult(graph: graph, marked: [5], steps: 3)
/// ```
///
/// - SeeAlso: ``WalkGraph``
/// - SeeAlso: ``WalkCoin``
/// - SeeAlso: ``WalkMethod``
public enum QuantumWalk {
    private static let epsilon: Double = 1e-15

    /// Performs a quantum walk on a graph, returning the evolution circuit.
    ///
    /// For discrete walks, constructs the walk operator W = S·(I⊗C) and applies it
    /// for the specified number of steps. The circuit operates on positionQubits + coinQubits
    /// total qubits. For continuous walks, builds the Trotter-Suzuki circuit for exp(-iAt)
    /// on positionQubits qubits.
    ///
    /// **Example:**
    /// ```swift
    /// let graph = WalkGraph.cycle(vertices: 8)
    /// let circuit = QuantumWalk.walk(on: graph, method: .discrete(coin: .grover, steps: 20))
    /// let state = circuit.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - graph: Graph topology for the walk
    ///   - method: Walk type (discrete or continuous)
    ///   - initialVertex: Starting vertex (default 0)
    /// - Returns: Quantum circuit implementing the walk
    /// - Precondition: initialVertex must be in range 0..<graph.vertexCount
    /// - Precondition: Graph must have at least 2 vertices
    /// - Precondition: steps > 0 (discrete)
    /// - Precondition: time >= 0 (continuous)
    /// - Precondition: trotterSteps > 0 (continuous)
    /// - Complexity: O(steps · n²) for discrete, O(trotterSteps · terms) for continuous
    ///
    /// - SeeAlso: ``WalkGraph``
    /// - SeeAlso: ``WalkMethod``
    @_optimize(speed)
    @_eagerMove
    public static func walk(
        on graph: WalkGraph,
        method: WalkMethod,
        initialVertex: Int = 0,
    ) -> QuantumCircuit {
        ValidationUtilities.validateLowerBound(graph.vertexCount, min: 2, name: "graph.vertexCount")
        ValidationUtilities.validateIndexInBounds(initialVertex, bound: graph.vertexCount, name: "initialVertex")

        switch method {
        case let .discrete(coin, steps):
            ValidationUtilities.validatePositiveInt(steps, name: "steps")
            return buildDiscreteWalk(graph: graph, coin: coin, steps: steps, initialVertex: initialVertex)

        case let .continuous(time, trotterSteps):
            ValidationUtilities.validateNonNegativeDouble(time, name: "time")
            ValidationUtilities.validatePositiveInt(trotterSteps, name: "trotterSteps")
            return buildContinuousWalk(graph: graph, time: time, trotterSteps: trotterSteps, initialVertex: initialVertex)
        }
    }

    /// Performs spatial search for marked vertices using a modified quantum walk.
    ///
    /// Implements the quantum walk search algorithm where the coin operator at marked
    /// vertices is replaced with -I (negative identity), causing amplitude to accumulate
    /// at marked positions. After the optimal number of steps O(√(n/k)) for k marked
    /// vertices in an n-vertex graph, the marked vertices have O(1) total probability.
    ///
    /// **Example:**
    /// ```swift
    /// let graph = WalkGraph.complete(vertices: 16)
    /// let circuit = QuantumWalk.search(on: graph, marked: [7])
    /// let state = circuit.execute()
    /// let result = state.searchResult(graph: graph, marked: [7], steps: 3)
    /// ```
    ///
    /// - Parameters:
    ///   - graph: Graph topology for the search
    ///   - marked: Array of marked vertex indices to search for
    ///   - steps: Number of walk steps (nil = optimal)
    /// - Returns: Quantum circuit implementing the spatial search
    /// - Precondition: marked must be non-empty
    /// - Precondition: All marked vertices must be in range 0..<graph.vertexCount
    /// - Precondition: Graph must have at least 2 vertices
    /// - Complexity: O(√n · n²) gate construction
    ///
    /// - SeeAlso: ``SpatialSearchResult``
    /// - SeeAlso: ``optimalSearchSteps(graph:markedCount:)``
    @_optimize(speed)
    @_eagerMove
    public static func search(
        on graph: WalkGraph,
        marked: [Int],
        steps: Int? = nil,
    ) -> QuantumCircuit {
        ValidationUtilities.validateLowerBound(graph.vertexCount, min: 2, name: "graph.vertexCount")
        ValidationUtilities.validateNonEmpty(marked, name: "marked")
        for vertex in marked {
            ValidationUtilities.validateIndexInBounds(vertex, bound: graph.vertexCount, name: "Marked vertex")
        }

        let numSteps = steps ?? optimalSearchSteps(graph: graph, markedCount: marked.count)
        let markedSet = Set(marked)

        let posQubits = graph.positionQubits
        let deg = graph.maxDegree
        let coinQubits = deg <= 1 ? 1 : (Int.bitWidth - (deg - 1).leadingZeroBitCount)
        let totalQubits = posQubits + coinQubits
        let dim = 1 << totalQubits

        let walkMatrix = buildSearchWalkOperator(
            graph: graph,
            markedSet: markedSet,
            posQubits: posQubits,
            coinQubits: coinQubits,
            dim: dim,
        )

        var circuit = QuantumCircuit(qubits: totalQubits)

        let allQubits = Array(0 ..< totalQubits)
        let coinDim = 1 << coinQubits
        let statePrep = buildSearchStatePreparation(
            graph: graph,
            coinDim: coinDim,
            dim: dim,
        )
        circuit.append(.customUnitary(matrix: statePrep), to: allQubits)

        if numSteps > 1 {
            let poweredMatrix = MatrixUtilities.matrixPower(walkMatrix, exponent: numSteps)
            circuit.append(.customUnitary(matrix: poweredMatrix), to: allQubits)
        } else if numSteps == 1 {
            circuit.append(.customUnitary(matrix: walkMatrix), to: allQubits)
        }

        return circuit
    }

    /// Computes the optimal number of walk steps for spatial search.
    ///
    /// Returns floor(π/2 · √(n/k)) where n is the vertex count and k is the number
    /// of marked vertices. This maximizes the success probability for finding a marked
    /// vertex on complete graphs and provides a good estimate for other regular graph
    /// families.
    ///
    /// **Example:**
    /// ```swift
    /// let graph = WalkGraph.complete(vertices: 64)
    /// let steps = QuantumWalk.optimalSearchSteps(graph: graph, markedCount: 1)
    /// let circuit = QuantumWalk.search(on: graph, marked: [0], steps: steps)
    /// ```
    ///
    /// - Parameters:
    ///   - graph: Graph topology
    ///   - markedCount: Number of marked vertices (default 1)
    /// - Returns: Optimal step count for maximum search success probability
    /// - Precondition: markedCount ≥ 1
    /// - Precondition: markedCount ≤ graph.vertexCount
    /// - Complexity: O(1)
    @_effects(readonly)
    public static func optimalSearchSteps(graph: WalkGraph, markedCount: Int = 1) -> Int {
        ValidationUtilities.validatePositiveInt(markedCount, name: "markedCount")
        ValidationUtilities.validateUpperBound(markedCount, max: graph.vertexCount, name: "markedCount")

        let n = Double(graph.vertexCount)
        let k = Double(markedCount)
        let optimal = (Double.pi / 2.0) * sqrt(n / k)
        return max(1, Int(floor(optimal)))
    }

    // MARK: - Discrete Walk Construction

    /// Builds discrete-time quantum walk circuit.
    @_optimize(speed)
    @_eagerMove
    private static func buildDiscreteWalk(
        graph: WalkGraph,
        coin: WalkCoin,
        steps: Int,
        initialVertex: Int,
    ) -> QuantumCircuit {
        let posQubits = graph.positionQubits
        let deg = graph.maxDegree
        let coinQubits = deg <= 1 ? 1 : (Int.bitWidth - (deg - 1).leadingZeroBitCount)
        let totalQubits = posQubits + coinQubits
        let dim = 1 << totalQubits

        let walkMatrix = buildWalkOperator(
            graph: graph,
            coin: coin,
            posQubits: posQubits,
            coinQubits: coinQubits,
            dim: dim,
        )

        var circuit = QuantumCircuit(qubits: totalQubits)

        prepareInitialState(circuit: &circuit, vertex: initialVertex, posQubits: posQubits, coinQubits: coinQubits)

        let allQubits = Array(0 ..< totalQubits)

        if steps > 1 {
            let poweredMatrix = MatrixUtilities.matrixPower(walkMatrix, exponent: steps)
            circuit.append(.customUnitary(matrix: poweredMatrix), to: allQubits)
        } else {
            circuit.append(.customUnitary(matrix: walkMatrix), to: allQubits)
        }

        return circuit
    }

    /// Prepares initial state |vertex⟩|+⟩ (localized position, uniform coin superposition).
    @_optimize(speed)
    private static func prepareInitialState(
        circuit: inout QuantumCircuit,
        vertex: Int,
        posQubits: Int,
        coinQubits: Int,
    ) {
        for bit in 0 ..< posQubits {
            if (vertex >> bit) & 1 == 1 {
                circuit.append(.pauliX, to: bit)
            }
        }

        for coinBit in 0 ..< coinQubits {
            circuit.append(.hadamard, to: posQubits + coinBit)
        }
    }

    /// Builds the full walk operator W = S · (I_pos ⊗ C) as a matrix.
    @_eagerMove
    @_optimize(speed)
    @_effects(readonly)
    private static func buildWalkOperator(
        graph: WalkGraph,
        coin: WalkCoin,
        posQubits: Int,
        coinQubits: Int,
        dim: Int,
    ) -> [[Complex<Double>]] {
        let coinDim = 1 << coinQubits
        let posDim = 1 << posQubits
        let coinMatrix = buildCoinMatrix(coin: coin, degree: graph.maxDegree, coinDim: coinDim)
        let shiftMatrix = buildShiftMatrix(graph: graph, coinDim: coinDim, dim: dim)

        let coinFullMatrix = tensorProductIdentityCoin(posDim: posDim, coinMatrix: coinMatrix, dim: dim)

        return MatrixUtilities.matrixMultiply(shiftMatrix, coinFullMatrix)
    }

    /// Builds search walk operator with -I coin at marked vertices.
    @_eagerMove
    @_optimize(speed)
    @_effects(readonly)
    private static func buildSearchWalkOperator(
        graph: WalkGraph,
        markedSet: Set<Int>,
        posQubits: Int,
        coinQubits: Int,
        dim: Int,
    ) -> [[Complex<Double>]] {
        let coinDim = 1 << coinQubits
        let posDim = 1 << posQubits
        let degree = graph.maxDegree
        let groverCoin = buildCoinMatrix(coin: .grover, degree: degree, coinDim: coinDim)
        let negIdentity = buildNegativeIdentity(coinDim: coinDim)

        let coinFullMatrix = buildSearchCoinMatrix(
            posDim: posDim,
            coinDim: coinDim,
            dim: dim,
            groverCoin: groverCoin,
            negIdentity: negIdentity,
            markedSet: markedSet,
            vertexCount: graph.vertexCount,
        )

        let shiftMatrix = buildShiftMatrix(graph: graph, coinDim: coinDim, dim: dim)

        return MatrixUtilities.matrixMultiply(shiftMatrix, coinFullMatrix)
    }

    /// Builds coin unitary matrix for the specified coin type.
    @_eagerMove
    @_optimize(speed)
    @_effects(readonly)
    private static func buildCoinMatrix(
        coin: WalkCoin,
        degree: Int,
        coinDim: Int,
    ) -> [[Complex<Double>]] {
        var matrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: coinDim),
            count: coinDim,
        )

        switch coin {
        case .grover:
            let d = Double(max(degree, 1))
            let offDiag = 2.0 / d
            let diag = offDiag - 1.0
            for i in 0 ..< degree {
                for j in 0 ..< degree {
                    matrix[i][j] = Complex(i == j ? diag : offDiag, 0.0)
                }
            }
            for i in degree ..< coinDim {
                matrix[i][i] = .one
            }

        case .hadamard:
            let invSqrt2 = 1.0 / sqrt(2.0)
            matrix[0][0] = Complex(invSqrt2, 0.0)
            matrix[0][1] = Complex(invSqrt2, 0.0)
            matrix[1][0] = Complex(invSqrt2, 0.0)
            matrix[1][1] = Complex(-invSqrt2, 0.0)
            for i in 2 ..< coinDim {
                matrix[i][i] = .one
            }

        case .fourier:
            let d = max(degree, 1)
            let invSqrtD = 1.0 / sqrt(Double(d))
            let twoPiOverD = 2.0 * Double.pi / Double(d)
            for j in 0 ..< d {
                for k in 0 ..< d {
                    let angle = twoPiOverD * Double(j * k)
                    matrix[j][k] = Complex(cos(angle) * invSqrtD, sin(angle) * invSqrtD)
                }
            }
            for i in d ..< coinDim {
                matrix[i][i] = .one
            }

        case let .custom(userMatrix):
            let size = min(userMatrix.count, coinDim)
            for i in 0 ..< size {
                let rowSize = min(userMatrix[i].count, coinDim)
                for j in 0 ..< rowSize {
                    matrix[i][j] = userMatrix[i][j]
                }
            }
            for i in size ..< coinDim {
                matrix[i][i] = .one
            }
        }

        return matrix
    }

    /// Builds the negative identity matrix for marked vertex coin.
    @_eagerMove
    @_effects(readonly)
    private static func buildNegativeIdentity(coinDim: Int) -> [[Complex<Double>]] {
        var matrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: coinDim),
            count: coinDim,
        )
        for i in 0 ..< coinDim {
            matrix[i][i] = Complex(-1.0, 0.0)
        }
        return matrix
    }

    /// Builds shift matrix S where S|v,j⟩ = |neighbor_j(v), reverseIndex⟩.
    @_eagerMove
    @_optimize(speed)
    @_effects(readonly)
    private static func buildShiftMatrix(
        graph: WalkGraph,
        coinDim: Int,
        dim: Int,
    ) -> [[Complex<Double>]] {
        var shift = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: dim),
            count: dim,
        )

        for v in 0 ..< graph.vertexCount {
            let nbrs = graph.neighbors(of: v)
            for (j, w) in nbrs.enumerated() where j < coinDim {
                let wNbrs = graph.neighbors(of: w)
                var reverseIdx = 0
                for (k, u) in wNbrs.enumerated() where u == v {
                    reverseIdx = k
                    break
                }
                let fromState = v * coinDim + j
                let toState = w * coinDim + min(reverseIdx, coinDim - 1)
                if fromState < dim, toState < dim {
                    shift[toState][fromState] = .one
                }
            }
        }

        for i in 0 ..< dim {
            var hasMapping = false
            for j in 0 ..< dim {
                if shift[i][j].magnitudeSquared > 0.5 {
                    hasMapping = true
                    break
                }
            }
            if !hasMapping {
                shift[i][i] = .one
            }
        }

        return shift
    }

    /// Builds I_pos ⊗ C (tensor product of identity on position with coin).
    @_eagerMove
    @_optimize(speed)
    @_effects(readonly)
    private static func tensorProductIdentityCoin(
        posDim: Int,
        coinMatrix: [[Complex<Double>]],
        dim: Int,
    ) -> [[Complex<Double>]] {
        let coinDim = coinMatrix.count
        var result = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: dim),
            count: dim,
        )

        for v in 0 ..< posDim {
            let blockStart = v * coinDim
            for i in 0 ..< coinDim {
                for j in 0 ..< coinDim {
                    result[blockStart + i][blockStart + j] = coinMatrix[i][j]
                }
            }
        }

        return result
    }

    /// Builds position-dependent coin matrix for search (Grover at unmarked, -I at marked).
    @_eagerMove
    @_optimize(speed)
    @_effects(readonly)
    private static func buildSearchCoinMatrix(
        posDim: Int,
        coinDim: Int,
        dim: Int,
        groverCoin: [[Complex<Double>]],
        negIdentity: [[Complex<Double>]],
        markedSet: Set<Int>,
        vertexCount: Int,
    ) -> [[Complex<Double>]] {
        var result = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: dim),
            count: dim,
        )

        for v in 0 ..< posDim {
            let blockStart = v * coinDim
            let coinToUse = (v < vertexCount && markedSet.contains(v)) ? negIdentity : groverCoin
            for i in 0 ..< coinDim {
                for j in 0 ..< coinDim {
                    result[blockStart + i][blockStart + j] = coinToUse[i][j]
                }
            }
        }

        return result
    }

    /// Builds state preparation unitary mapping |0⟩ to uniform superposition over valid walk states.
    @_eagerMove
    @_optimize(speed)
    @_effects(readonly)
    private static func buildSearchStatePreparation(
        graph: WalkGraph,
        coinDim: Int,
        dim: Int,
    ) -> [[Complex<Double>]] {
        var targetAmps = [Complex<Double>](repeating: .zero, count: dim)
        var validCount = 0
        for v in 0 ..< graph.vertexCount {
            let deg = graph.degree(of: v)
            for j in 0 ..< deg {
                let idx = v * coinDim + j
                if idx < dim { validCount += 1 }
            }
        }

        let amp = Complex(1.0 / sqrt(Double(max(validCount, 1))), 0.0)
        for v in 0 ..< graph.vertexCount {
            let deg = graph.degree(of: v)
            for j in 0 ..< deg {
                let idx = v * coinDim + j
                if idx < dim { targetAmps[idx] = amp }
            }
        }

        return buildHouseholderUnitary(target: targetAmps, dim: dim)
    }

    /// Builds a Householder unitary U such that U|0⟩ = |target⟩.
    @_eagerMove
    @_optimize(speed)
    @_effects(readonly)
    private static func buildHouseholderUnitary(
        target: [Complex<Double>],
        dim: Int,
    ) -> [[Complex<Double>]] {
        let w = [Complex<Double>](unsafeUninitializedCapacity: dim) { buffer, count in
            buffer[0] = Complex(1.0, 0.0) - target[0]
            for i in 1 ..< dim {
                buffer[i] = Complex(0.0, 0.0) - target[i]
            }
            count = dim
        }

        var wNormSq = 0.0
        for i in 0 ..< dim {
            wNormSq += w[i].magnitudeSquared
        }

        var result = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: dim),
            count: dim,
        )

        let factor = -2.0 / wNormSq
        for i in 0 ..< dim {
            result[i][i] = .one
            for j in 0 ..< dim {
                result[i][j] = result[i][j] + Complex(factor, 0.0) * w[i] * w[j].conjugate
            }
        }

        return result
    }

    // MARK: - Continuous Walk Construction

    /// Builds continuous-time walk via first-order Trotter: exp(-iHt) ≈ [Π_e exp(-iH_e·dt)]^s.
    @_optimize(speed)
    @_eagerMove
    private static func buildContinuousWalk(
        graph: WalkGraph,
        time: Double,
        trotterSteps: Int,
        initialVertex: Int,
    ) -> QuantumCircuit {
        let posQubits = graph.positionQubits
        let dim = 1 << posQubits
        let n = graph.vertexCount
        let dt = time / Double(trotterSteps)

        var stepMatrix = [[Complex<Double>]](unsafeUninitializedCapacity: dim) {
            buffer, count in
            for i in 0 ..< dim {
                buffer[i] = [Complex<Double>](unsafeUninitializedCapacity: dim) { inner, innerCount in
                    for j in 0 ..< dim {
                        inner[j] = i == j ? .one : .zero
                    }
                    innerCount = dim
                }
            }
            count = dim
        }

        for u in 0 ..< n {
            let rowStart = u * n
            for v in (u + 1) ..< n {
                let weight = graph.adjacencyMatrix[rowStart + v]
                if abs(weight) < Self.epsilon { continue }
                let angle = weight * dt
                let cosA = Complex<Double>(cos(angle), 0.0)
                let mISinA = Complex<Double>(0.0, -sin(angle))
                for i in 0 ..< dim {
                    let oldU = stepMatrix[i][u]
                    let oldV = stepMatrix[i][v]
                    stepMatrix[i][u] = oldU * cosA + oldV * mISinA
                    stepMatrix[i][v] = oldU * mISinA + oldV * cosA
                }
            }
        }

        var circuit = QuantumCircuit(qubits: posQubits)

        prepareInitialState(circuit: &circuit, vertex: initialVertex, posQubits: posQubits, coinQubits: 0)

        let allQubits = Array(0 ..< posQubits)

        if trotterSteps > 1 {
            let poweredMatrix = MatrixUtilities.matrixPower(stepMatrix, exponent: trotterSteps)
            circuit.append(.customUnitary(matrix: poweredMatrix), to: allQubits)
        } else {
            circuit.append(.customUnitary(matrix: stepMatrix), to: allQubits)
        }

        return circuit
    }

    // MARK: - Pauli Decomposition Helpers

    /// Applies a Pauli string (encoded as integer index) to a basis state.
    @_optimize(speed)
    @_effects(readonly)
    static func applyPauliString(
        pauliIndex: Int,
        qubits: Int,
        basisState: Int,
    ) -> (phase: Complex<Double>, resultState: Int) {
        var state = basisState
        var phase = Complex<Double>.one

        for q in 0 ..< qubits {
            let pauliType = (pauliIndex >> (2 * q)) & 3
            let bit = (state >> q) & 1

            if pauliType == 1 {
                state ^= (1 << q)
            } else if pauliType == 2 {
                state ^= (1 << q)
                let newBit = (state >> q) & 1
                phase = phase * (newBit == 1 ? Complex(0.0, 1.0) : Complex(0.0, -1.0))
            } else if pauliType == 3 {
                phase = phase * (bit == 1 ? Complex(-1.0, 0.0) : Complex(1.0, 0.0))
            }
        }

        return (phase, state)
    }

    /// Extracts PauliOperator array from encoded Pauli index.
    @_effects(readonly)
    static func pauliOperators(
        from pauliIndex: Int,
        qubits: Int,
    ) -> [PauliOperator] {
        var ops: [PauliOperator] = []
        for q in 0 ..< qubits {
            let pauliType = (pauliIndex >> (2 * q)) & 3
            switch pauliType {
            case 1: ops.append(.x(q))
            case 2: ops.append(.y(q))
            case 3: ops.append(.z(q))
            default: break
            }
        }
        return ops
    }
}

// MARK: - QuantumState Extension

public extension QuantumState {
    /// Extracts quantum walk result with per-vertex probability distribution.
    ///
    /// For discrete-time walks, traces out the coin register to obtain vertex probabilities.
    /// For continuous-time walks, the state directly encodes vertex probabilities.
    ///
    /// **Example:**
    /// ```swift
    /// let graph = WalkGraph.cycle(vertices: 8)
    /// let state = QuantumWalk.walk(on: graph, method: .discrete(coin: .grover, steps: 20)).execute()
    /// let result = state.walkResult(graph: graph, steps: 20)
    /// ```
    ///
    /// - Parameters:
    ///   - graph: Graph used in the walk
    ///   - steps: Number of walk steps applied
    /// - Returns: ``WalkResult`` with vertex probabilities
    /// - Complexity: O(2^n) where n is total qubits
    ///
    /// - SeeAlso: ``WalkResult``
    @_optimize(speed)
    @_effects(readonly)
    func walkResult(graph: WalkGraph, steps: Int) -> WalkResult {
        let n = graph.vertexCount
        let posQubits = graph.positionQubits
        let totalQubits = qubits
        let coinQubits = totalQubits - posQubits

        let probs = [Double](unsafeUninitializedCapacity: n) { buffer, count in
            if coinQubits > 0 {
                let coinDim = 1 << coinQubits
                for v in 0 ..< n {
                    var vertexProb = 0.0
                    for c in 0 ..< coinDim {
                        let stateIdx = v * coinDim + c
                        if stateIdx < stateSpaceSize {
                            vertexProb += probability(of: stateIdx)
                        }
                    }
                    buffer[v] = vertexProb
                }
            } else {
                for v in 0 ..< n {
                    buffer[v] = probability(of: v)
                }
            }
            count = n
        }

        var maxProb = 0.0
        var maxVertex = 0
        for v in 0 ..< n {
            if probs[v] > maxProb {
                maxProb = probs[v]
                maxVertex = v
            }
        }

        return WalkResult(
            vertexProbabilities: probs,
            mostProbableVertex: maxVertex,
            steps: steps,
        )
    }

    /// Extracts spatial search result with success probability.
    ///
    /// Computes the total probability on marked vertices and identifies the most probable
    /// vertex from the measurement distribution.
    ///
    /// **Example:**
    /// ```swift
    /// let graph = WalkGraph.complete(vertices: 16)
    /// let circuit = QuantumWalk.search(on: graph, marked: [7])
    /// let state = circuit.execute()
    /// let result = state.searchResult(graph: graph, marked: [7], steps: 3)
    /// print(result.isMarked)
    /// ```
    ///
    /// - Parameters:
    ///   - graph: Graph used in the search
    ///   - marked: Marked vertex indices
    ///   - steps: Number of walk steps applied
    /// - Returns: ``SpatialSearchResult`` with search outcome
    /// - Complexity: O(2^n) where n is total qubits
    ///
    /// - SeeAlso: ``SpatialSearchResult``
    @_optimize(speed)
    @_effects(readonly)
    func searchResult(graph: WalkGraph, marked: [Int], steps: Int) -> SpatialSearchResult {
        let walkRes = walkResult(graph: graph, steps: steps)
        let markedSet = Set(marked)

        var successProb = 0.0
        for v in marked {
            if v < walkRes.vertexProbabilities.count {
                successProb += walkRes.vertexProbabilities[v]
            }
        }

        let foundVertex = walkRes.mostProbableVertex
        let isMarked = markedSet.contains(foundVertex)
        let optimalSteps = QuantumWalk.optimalSearchSteps(graph: graph, markedCount: marked.count)

        return SpatialSearchResult(
            foundVertex: foundVertex,
            successProbability: successProb,
            steps: steps,
            optimalSteps: optimalSteps,
            isMarked: isMarked,
        )
    }
}
