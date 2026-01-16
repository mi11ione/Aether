// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

// MARK: - QAOA

/// Hybrid quantum-classical algorithm for combinatorial optimization.
///
/// QAOA solves NP-hard optimization problems through variational parameter optimization of a quantum
/// ansatz composed of alternating problem and mixer Hamiltonian layers. The algorithm prepares an
/// initial superposition state, applies p alternating layers of exp(-iγₖH_p) and exp(-iβₖH_m), then
/// measures to extract solution bitstrings. Classical optimization updates the 2p parameters (γ⃗,β⃗)
/// to minimize the expectation value ⟨ψ(γ⃗,β⃗)|H_p|ψ(γ⃗,β⃗)⟩ until convergence.
///
/// The problem Hamiltonian H_p encodes the optimization objective where lower eigenvalues correspond
/// to better solutions. The mixer Hamiltonian H_m drives exploration of the solution space, typically
/// implemented as Σᵢ Xᵢ for unconstrained problems. Increasing depth p improves approximation quality
/// at the cost of circuit complexity and optimization difficulty.
///
/// Applications span graph partitioning (MaxCut), routing optimization (TSP), constraint satisfaction
/// (SAT), scheduling (graph coloring), and portfolio optimization in finance. The algorithm targets
/// near-term quantum hardware with shallow circuits suitable for NISQ devices.
///
/// **Example:**
/// ```swift
/// let qaoa = QAOA(cost: MaxCut.hamiltonian(edges: [(0,1), (1,2)]), qubits: 3, depth: 2)
/// let result = await qaoa.run(from: [0.5, 0.5, 0.5, 0.5])
/// print("Optimal cost: \(result.optimalCost)")
/// ```
///
/// - Complexity: O(iterations × (depth × 2^n + hamiltonian_sparsity))
/// - Note: Actor-based for thread safety and concurrent execution prevention
/// - SeeAlso: ``Result``
/// - SeeAlso: ``MaxCut``
/// - SeeAlso: ``MixerHamiltonian``
/// - SeeAlso: ``Observable``
/// - SeeAlso: ``SparseHamiltonian``
public actor QAOA {
    // MARK: - Configuration

    /// Cost Hamiltonian H_p encoding optimization problem
    private let costHamiltonian: Observable

    /// Mixer Hamiltonian H_m driving exploration (default: X mixer)
    private let mixerHamiltonian: Observable

    /// Number of qubits in system
    private let qubits: Int

    /// QAOA depth: number of alternating layers
    private let depth: Int

    /// Classical optimizer for parameter updates
    private let optimizer: Optimizer

    /// Convergence criteria for optimization
    private let convergence: ConvergenceCriteria

    /// High-performance sparse Hamiltonian backend (optional)
    private let sparseHamiltonian: SparseHamiltonian?

    /// Quantum simulator for circuit execution
    private let simulator: QuantumSimulator

    /// Parameterized QAOA ansatz circuit
    private let ansatz: QuantumCircuit

    /// Pre-computed parameter binder for fast repeated binding
    private let parameterBinder: QAOAParameterBinder

    // MARK: - State

    /// Current optimization iteration
    private var currentIteration: Int = 0

    /// Current best cost
    private var currentCost: Double = 0.0

    // MARK: - Initialization

    /// Creates QAOA optimizer with specified Hamiltonian and configuration.
    ///
    /// Constructs parameterized ansatz circuit with alternating problem and mixer layers. Attempts
    /// sparse Hamiltonian backend construction for hardware-accelerated expectation values when enabled,
    /// falling back to term-by-term Observable measurement on construction failure. Default X mixer applies
    /// uniform superposition transitions across computational basis states.
    ///
    /// **Example:**
    /// ```swift
    /// let cost = MaxCut.hamiltonian(edges: [(0,1), (1,2)])
    /// let qaoa = QAOA(cost: cost, qubits: 3, depth: 2)
    /// ```
    ///
    /// - Parameters:
    ///   - cost: Cost Hamiltonian H_c encoding optimization objective
    ///   - mixer: Mixer Hamiltonian H_m for solution space exploration (default: X mixer Σᵢ Xᵢ)
    ///   - qubits: System size (1-30 for statevector simulation)
    ///   - depth: Number of alternating QAOA layers (typical range 1-10)
    ///   - optimizer: Classical parameter optimization algorithm
    ///   - convergence: Termination criteria for optimization loop
    ///   - sparseBackend: Enable sparse Hamiltonian hardware acceleration
    ///   - metalAcceleration: Enable Metal GPU for quantum circuit execution
    /// - Precondition: `qubits` must be positive and ≤30 for memory constraints
    /// - SeeAlso: ``run(from:progress:)``
    /// - SeeAlso: ``MaxCut``
    /// - SeeAlso: ``MixerHamiltonian``
    public init(
        cost: Observable,
        mixer: Observable? = nil,
        qubits: Int,
        depth: Int,
        optimizer: Optimizer = COBYLAOptimizer(tolerance: 1e-6),
        convergence: ConvergenceCriteria = ConvergenceCriteria(),
        sparseBackend: Bool = true,
        metalAcceleration: Bool = true,
    ) {
        costHamiltonian = cost
        mixerHamiltonian = mixer ?? MixerHamiltonian.x(qubits: qubits)
        self.qubits = qubits
        self.depth = depth
        self.optimizer = optimizer
        self.convergence = convergence
        sparseHamiltonian = sparseBackend ? SparseHamiltonian(observable: cost) : nil
        simulator = QuantumSimulator(useMetalAcceleration: metalAcceleration)

        ansatz = QuantumCircuit.qaoa(
            cost: costHamiltonian,
            mixer: mixerHamiltonian,
            qubits: self.qubits,
            depth: depth,
        )

        parameterBinder = QAOAParameterBinder(ansatz: ansatz)
    }

    // MARK: - Execution

    /// Executes QAOA optimization from initial parameters.
    ///
    /// Runs hybrid loop alternating quantum circuit evaluation and classical parameter updates. Each
    /// iteration binds current parameters to ansatz, executes circuit on quantum simulator with optional
    /// GPU acceleration, computes expectation value via sparse or Observable backend, and updates parameters
    /// through classical optimizer. Continues until energy tolerance, gradient convergence, or maximum
    /// iterations reached. Optional progress callback receives iteration count and current cost for
    /// real-time monitoring and UI updates.
    ///
    /// **Example:**
    /// ```swift
    /// let result = await qaoa.run(from: [0.5, 0.5]) { i, c in print("[\(i)] \(c)") }
    /// print("Optimal: \(result.optimalCost), solutions: \(result.topSolutions(3))")
    /// ```
    ///
    /// - Parameters:
    ///   - parameters: Initial parameter vector (length 2×depth for γ⃗ and β⃗)
    ///   - progress: Optional callback receiving iteration count and current cost value
    /// - Returns: Optimization result with cost, parameters, solution probabilities, and convergence info
    /// - Complexity: O(iterations × (circuit_depth × 2^n + hamiltonian_operations))
    /// - Precondition: `parameters.count` must equal 2×depth
    /// - SeeAlso: ``Result``
    /// - SeeAlso: ``ConvergenceCriteria``
    @_optimize(speed)
    @_eagerMove
    public func run(
        from parameters: [Double],
        progress: ProgressCallback? = nil,
    ) async -> Result {
        let expectedParamCount = 2 * depth
        ValidationUtilities.validateParameterVectorLength(
            parameters.count,
            expected: expectedParamCount,
            name: "QAOA parameters (requires 2 x depth)",
        )

        currentIteration = 0
        currentCost = 0.0

        let costFunction: @Sendable ([Double]) async -> Double = { params in
            let concreteCircuit: QuantumCircuit = self.parameterBinder.bind(baseParameters: params)
            let state: QuantumState = await self.simulator.execute(concreteCircuit)
            let cost: Double = if let sparseH = self.sparseHamiltonian {
                await sparseH.expectationValue(of: state)
            } else {
                self.costHamiltonian.expectationValue(of: state)
            }
            return cost
        }

        let optimizerProgress: ProgressCallback? = if let callback = progress {
            { iteration, cost in
                await self.updateProgress(iteration: iteration, cost: cost)
                await callback(iteration, cost)
            }
        } else {
            { iteration, cost in
                await self.updateProgress(iteration: iteration, cost: cost)
            }
        }

        let optimizerResult: OptimizerResult = await optimizer.minimize(
            costFunction,
            from: parameters,
            using: convergence,
            progress: optimizerProgress,
        )

        let finalCircuit: QuantumCircuit = parameterBinder.bind(baseParameters: optimizerResult.parameters)
        let finalState: QuantumState = await simulator.execute(finalCircuit)
        let solutionProbabilities: [Int: Double] = extractSolutionProbabilities(state: finalState)

        return Result(
            optimalCost: optimizerResult.value,
            optimalParameters: optimizerResult.parameters,
            solutionProbabilities: solutionProbabilities,
            costHistory: optimizerResult.history,
            iterations: optimizerResult.iterations,
            convergenceReason: optimizerResult.terminationReason,
            functionEvaluations: optimizerResult.evaluations,
        )
    }

    // MARK: - Helpers

    /// Extracts significant solution probabilities from quantum state.
    ///
    /// Computes Born rule probabilities for all computational basis states and filters to values
    /// exceeding threshold for practical solution analysis. Small probabilities below 1e-6 are
    /// discarded to reduce dictionary size and focus on likely solutions.
    ///
    /// - Parameter state: Quantum state after QAOA circuit execution
    /// - Returns: Dictionary mapping bitstring index to probability (filtered > 1e-6)
    /// - Complexity: O(2^n) for probability computation and filtering
    @_optimize(speed)
    @_eagerMove
    @inline(__always)
    private func extractSolutionProbabilities(state: QuantumState) -> [Int: Double] {
        var probabilities: [Int: Double] = [:]
        probabilities.reserveCapacity(min(state.stateSpaceSize, 100))

        for i in 0 ..< state.stateSpaceSize {
            let prob: Double = state.probability(of: i)
            if prob > 1e-6 {
                probabilities[i] = prob
            }
        }

        return probabilities
    }

    // MARK: - State Queries

    /// Current optimization iteration and cost value.
    ///
    /// Returns tuple containing most recent iteration count and cost function value from optimizer.
    /// Updated after each optimization step for progress monitoring and convergence diagnostics.
    ///
    /// **Example:**
    /// ```swift
    /// let (iteration, cost) = await qaoa.progress
    /// print("Iteration \(iteration): cost = \(cost)")
    /// ```
    ///
    /// - SeeAlso: ``run(from:progress:)``
    public var progress: (iteration: Int, cost: Double) {
        (currentIteration, currentCost)
    }

    /// Backend type used for expectation value computation.
    ///
    /// Returns sparse backend with description when hardware acceleration is available, or Observable
    /// backend with term count when using term-by-term measurement. Query asynchronously as sparse
    /// backend description may require actor isolation.
    ///
    /// **Example:**
    /// ```swift
    /// let backend = await qaoa.backend
    /// if case .sparse(let desc) = backend { print("Using sparse: \(desc)") }
    /// ```
    ///
    /// - SeeAlso: ``BackendType``
    /// - SeeAlso: ``SparseHamiltonian``
    public var backend: BackendType {
        get async {
            if let sparseH = sparseHamiltonian {
                await .sparse(description: sparseH.backendDescription)
            } else {
                .observable(termCount: costHamiltonian.terms.count)
            }
        }
    }

    /// Updates current iteration and cost state.
    @inline(__always)
    private func updateProgress(iteration: Int, cost: Double) {
        currentIteration = iteration
        currentCost = cost
    }

    // MARK: - Nested Types

    /// Backend type for QAOA expectation value computation.
    ///
    /// Distinguishes between sparse Hamiltonian acceleration and term-by-term Observable measurement.
    /// Sparse backend enables hardware acceleration via Metal GPU or Accelerate framework for molecular
    /// Hamiltonians with typical 0.01-1% sparsity.
    ///
    /// **Example:**
    /// ```swift
    /// let backend = await qaoa.backend
    /// if case .sparse(let desc) = backend { print("Sparse: \(desc)") }
    /// ```
    ///
    /// - SeeAlso: ``QAOA/backend``
    /// - SeeAlso: ``SparseHamiltonian``
    @frozen
    public enum BackendType: Sendable {
        /// Sparse Hamiltonian backend with hardware acceleration
        case sparse(description: String)

        /// Observable backend with term-by-term measurement
        case observable(termCount: Int)
    }

    /// QAOA optimization result with cost, parameters, solutions, and convergence diagnostics.
    ///
    /// Encapsulates complete optimization output including optimal cost value representing minimum expectation
    /// value achieved, optimal parameter vector (γ⃗,β⃗), solution probability distribution mapping bitstring
    /// indices to measurement probabilities, and convergence metadata.
    ///
    /// **Example:**
    /// ```swift
    /// let result = await qaoa.run(from: [0.5, 0.5, 0.5, 0.5])
    /// print("Optimal cost: \(result.optimalCost)")
    /// ```
    ///
    /// - SeeAlso: ``QAOA/run(from:progress:)``
    /// - SeeAlso: ``TerminationReason``
    @frozen
    public struct Result: Sendable, CustomStringConvertible {
        /// Optimal cost found: min ⟨ψ|H_p|ψ⟩
        public let optimalCost: Double

        /// Optimal parameters (γ₀,β₀,...,γₚ₋₁,βₚ₋₁)
        public let optimalParameters: [Double]

        /// Solution probability distribution [bitstring index -> probability]
        public let solutionProbabilities: [Int: Double]

        /// Complete cost history (one per iteration)
        public let costHistory: [Double]

        /// Total optimization iterations
        public let iterations: Int

        /// Why optimization terminated
        public let convergenceReason: TerminationReason

        /// Total objective function evaluations
        public let functionEvaluations: Int

        public init(
            optimalCost: Double,
            optimalParameters: [Double],
            solutionProbabilities: [Int: Double],
            costHistory: [Double],
            iterations: Int,
            convergenceReason: TerminationReason,
            functionEvaluations: Int,
        ) {
            self.optimalCost = optimalCost
            self.optimalParameters = optimalParameters
            self.solutionProbabilities = solutionProbabilities
            self.costHistory = costHistory
            self.iterations = iterations
            self.convergenceReason = convergenceReason
            self.functionEvaluations = functionEvaluations
        }

        /// Multi-line formatted summary of optimization results.
        public var description: String {
            let paramStr = optimalParameters.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", ")
            let paramSuffix = optimalParameters.count > 4 ? ", ..." : ""

            let topSols = topSolutions(3)
            let solutionsStr = topSols.map { bitstring, prob in
                let binary = String(bitstring, radix: 2)
                return "\(binary) (\(String(format: "%.1f%%", prob * 100)))"
            }.joined(separator: ", ")

            return """
            QAOA Result:
              Optimal Cost: \(String(format: "%.8f", optimalCost))
              Parameters: [\(paramStr)\(paramSuffix)]
              Top Solutions: \(solutionsStr)
              Iterations: \(iterations)
              Function Evaluations: \(functionEvaluations)
              Convergence: \(convergenceReason)
            """
        }

        /// Returns top solutions by probability using efficient partial sort.
        ///
        /// Extracts most likely solution bitstrings from probability distribution. Uses min-heap-based
        /// partial sort achieving O(n + k log k) complexity versus O(n log n) for full sort when k << n.
        ///
        /// **Example:**
        /// ```swift
        /// for (bitstring, probability) in result.topSolutions(5) {
        ///     print("Solution \(bitstring): \(probability)")
        /// }
        /// ```
        ///
        /// - Parameter count: Number of top solutions to extract
        /// - Returns: Array of (bitstring, probability) tuples sorted descending by probability
        /// - Complexity: O(n + k log k) where n = total solutions, k = requested count
        @_optimize(speed)
        @_eagerMove
        public func topSolutions(_ count: Int) -> [(bitstring: Int, probability: Double)] {
            let numSolutions = solutionProbabilities.count
            guard numSolutions > 0, count > 0 else { return [] }

            if numSolutions <= count * 4 {
                return solutionProbabilities
                    .sorted { $0.value > $1.value }
                    .prefix(count)
                    .map { (bitstring: $0.key, probability: $0.value) }
            }

            var heap: [(bitstring: Int, probability: Double)] = []
            heap.reserveCapacity(count)

            for (bitstring, prob) in solutionProbabilities {
                if heap.count < count {
                    heap.append((bitstring: bitstring, probability: prob))
                    if heap.count == count {
                        for i in stride(from: count / 2 - 1, through: 0, by: -1) {
                            siftDown(&heap, i, count)
                        }
                    }
                } else if prob > heap[0].probability {
                    heap[0] = (bitstring: bitstring, probability: prob)
                    siftDown(&heap, 0, count)
                }
            }

            return heap.sorted { $0.probability > $1.probability }
        }

        /// Min-heap sift down for partial sort.
        @inline(__always)
        private func siftDown(_ heap: inout [(bitstring: Int, probability: Double)], _ index: Int, _ size: Int) {
            var i = index
            while true {
                let left = 2 * i + 1
                let right = 2 * i + 2
                var smallest = i

                if left < size, heap[left].probability < heap[smallest].probability {
                    smallest = left
                }
                if right < size, heap[right].probability < heap[smallest].probability {
                    smallest = right
                }

                if smallest == i { break }
                heap.swapAt(i, smallest)
                i = smallest
            }
        }
    }
}
