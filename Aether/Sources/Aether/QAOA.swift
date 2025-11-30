// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Quantum Approximate Optimization Algorithm: Hybrid quantum-classical combinatorial optimizer
///
/// Implements QAOA for solving combinatorial optimization problems like MaxCut,
/// graph coloring, TSP, and satisfiability. **Near-term algorithm** designed for
/// NISQ devices with applications in operations research, network design, and logistics.
///
/// **Algorithm Overview:**
/// 1. **Initialize**: Prepare equal superposition |+⟩^⊗n
/// 2. **Alternating layers**: Apply exp(-iγH_p) exp(-iβH_m) for p layers
/// 3. **Measure**: Sample from computational basis
/// 4. **Classical optimization**: Update (γ⃗,β⃗) to minimize ⟨H_p⟩
/// 5. **Repeat until convergence**
///
/// **Mathematical Foundation:**
/// - Problem Hamiltonian: H_p encodes optimization cost function
/// - Mixer Hamiltonian: H_m drives exploration (typically Σᵢ Xᵢ)
/// - Ansatz state: |ψ(γ⃗,β⃗)⟩ from alternating problem/mixer layers
/// - Objective: min ⟨ψ(γ⃗,β⃗)|H_p|ψ(γ⃗,β⃗)⟩
/// - Approximation ratio: ⟨H_p⟩/C_opt where C_opt is optimal cost
///
/// **Performance:**
/// - Circuit depth: O(p·|E|) where |E| = Hamiltonian terms
/// - Parameter count: 2p (γ and β per layer)
/// - Typical convergence: 50-300 optimizer iterations
/// - Uses SparseHamiltonian backend (100-1000x speedup)
///
/// **Use Cases:**
/// - **MaxCut**: Graph partitioning, network design
/// - **TSP**: Routing optimization, logistics
/// - **Graph coloring**: Scheduling, register allocation
/// - **SAT**: Boolean satisfiability, verification
/// - **Portfolio optimization**: Finance, risk management
///
/// **Example - MaxCut on square graph:**
/// ```swift
/// // Define 4-vertex square: optimal MaxCut = 4
/// let edges = [(0,1), (1,2), (2,3), (3,0)]
/// let costHamiltonian = MaxCut.hamiltonian(edges: edges)
/// let mixerHamiltonian = MixerHamiltonian.xMixer(numQubits: 4)
///
/// // Run depth-2 QAOA
/// let qaoa = await QAOA(
///     costHamiltonian: costHamiltonian,
///     mixerHamiltonian: mixerHamiltonian,
///     numQubits: 4,
///     depth: 2,
///     optimizer: COBYLAOptimizer(tolerance: 1e-6)
/// )
///
/// // Optimize starting from random parameters
/// let result = await qaoa.run(
///     initialParameters: [0.5, 0.5, 0.5, 0.5]  // (γ₀,β₀,γ₁,β₁)
/// )
///
/// // Extract solution
/// print("Best cost: \(result.optimalCost)")  // ≈ -2.0
/// print("MaxCut value: \(Int(-2.0 * result.optimalCost))")  // 4
///
/// // Top solutions (bitstrings with highest probability)
/// for (bitstring, probability) in result.solutionProbabilities.prefix(5) {
///     let binary = String(bitstring, radix: 2).padded(toLength: 4, withPad: "0", startingAt: 0)
///     print("Solution \(binary): probability = \(String(format: "%.3f", probability))")
/// }
/// // Output: 0101 (29%), 1010 (29%), ... (optimal partitions)
/// ```
///
/// **Example - Progress tracking:**
/// ```swift
/// let qaoa = await QAOA(...)
///
/// let result = await qaoa.runWithProgress(
///     initialParameters: [0.5, 0.5]
/// ) { iteration, cost in
///     print("Iteration \(iteration): cost = \(String(format: "%.6f", cost))")
/// }
/// ```
///
/// **Architecture:**
/// - Actor-based: Thread-safe, prevents data races
/// - Async optimization: Non-blocking for UI applications
/// - SparseHamiltonian: GPU/Accelerate hardware acceleration
/// - MPS batched evaluation: Grid search for small p
/// - Progress tracking: Real-time cost updates
public actor QAOA {
    // MARK: - Configuration

    /// Cost Hamiltonian H_p encoding optimization problem
    private let costHamiltonian: Observable

    /// Mixer Hamiltonian H_m driving exploration (default: X mixer)
    private let mixerHamiltonian: Observable

    /// Number of qubits in system
    private let numQubits: Int

    /// QAOA depth: number of alternating layers
    private let depth: Int

    /// Classical optimizer for parameter updates
    private let optimizer: Optimizer

    /// Convergence criteria for optimization
    private let convergenceCriteria: ConvergenceCriteria

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

    /// Create QAOA instance
    ///
    /// Configures hybrid quantum-classical optimization for combinatorial problems.
    /// Automatically constructs QAOA ansatz and SparseHamiltonian backend.
    ///
    /// **Backend Selection:**
    /// - SparseHamiltonian (default): GPU/Accelerate hardware acceleration
    /// - Observable (fallback): Term-by-term measurement
    ///
    /// **Performance:**
    /// - Sparse backend: O(nnz) where nnz = number of non-zeros
    /// - Observable backend: O(k·2^n) where k = number of Pauli terms
    ///
    /// - Parameters:
    ///   - costHamiltonian: Problem Hamiltonian H_p (e.g., MaxCut)
    ///   - mixerHamiltonian: Mixer Hamiltonian H_m (default: X mixer)
    ///   - numQubits: Number of qubits (1-30)
    ///   - depth: QAOA depth p (1-10 typical)
    ///   - optimizer: Classical optimization algorithm
    ///   - convergenceCriteria: Termination conditions (default: ε=1e-6, maxIter=1000)
    ///   - useSparseBackend: Use SparseHamiltonian acceleration (default: true)
    ///   - useMetalAcceleration: Use Metal GPU for circuit execution (default: true)
    public init(
        costHamiltonian: Observable,
        mixerHamiltonian: Observable? = nil,
        numQubits: Int,
        depth: Int,
        optimizer: Optimizer = COBYLAOptimizer(tolerance: 1e-6),
        convergenceCriteria: ConvergenceCriteria = .default,
        useSparseBackend: Bool = true,
        useMetalAcceleration: Bool = true
    ) {
        self.costHamiltonian = costHamiltonian
        self.mixerHamiltonian = mixerHamiltonian ?? MixerHamiltonian.xMixer(numQubits: numQubits)
        self.numQubits = numQubits
        self.depth = depth
        self.optimizer = optimizer
        self.convergenceCriteria = convergenceCriteria
        sparseHamiltonian = useSparseBackend ? SparseHamiltonian(observable: costHamiltonian) : nil
        simulator = QuantumSimulator(useMetalAcceleration: useMetalAcceleration)

        ansatz = QAOAAnsatz.create(
            numQubits: numQubits,
            depth: depth,
            costHamiltonian: costHamiltonian,
            mixerHamiltonian: self.mixerHamiltonian
        )

        parameterBinder = QAOAParameterBinder(ansatz: ansatz)
    }

    // MARK: - Execution

    /// Run QAOA optimization
    ///
    /// Executes hybrid quantum-classical loop until convergence or max iterations.
    /// Each iteration:
    /// 1. Bind current (γ⃗,β⃗) to ansatz -> concrete circuit
    /// 2. Execute circuit on simulator (GPU-accelerated if available)
    /// 3. Compute ⟨ψ|H_p|ψ⟩ using SparseHamiltonian (or Observable fallback)
    /// 4. Classical optimizer updates parameters
    ///
    /// **Complexity:**
    /// - Per iteration: O(d·2^n + nnz) where d = circuit depth, nnz = Hamiltonian non-zeros
    /// - Total: O(iters x (d·2^n + nnz)) where iters = optimizer iterations
    ///
    /// **Thread Safety:**
    /// - Actor isolation ensures thread-safe execution
    /// - Prevents concurrent QAOA runs
    ///
    /// - Parameter initialParameters: Starting point (γ₀,β₀,...,γₚ₋₁,βₚ₋₁)
    /// - Returns: QAOA result with optimal cost and solution bitstrings
    ///
    /// Example:
    /// ```swift
    /// let initialGuess = [0.5, 0.5, 0.5, 0.5]  // depth=2
    /// let result = await qaoa.run(initialParameters: initialGuess)
    ///
    /// print("Optimal cost: \(result.optimalCost)")
    /// print("Parameters: \(result.optimalParameters)")
    /// print("Top solution: \(result.solutionProbabilities.max(by: { $0.value < $1.value })!)")
    /// ```
    @_optimize(speed)
    @_eagerMove
    public func run(initialParameters: [Double]) async -> QAOAResult {
        await runWithProgress(initialParameters: initialParameters, progressCallback: nil)
    }

    /// Run QAOA with progress updates
    ///
    /// Same as `run()` but calls progressCallback after each iteration.
    /// Useful for UI updates, logging, and convergence visualization.
    ///
    /// **Progress Callback:**
    /// - Called after each optimizer iteration
    /// - Receives: (iteration: Int, currentCost: Double)
    /// - Async: can perform UI updates on MainActor
    ///
    /// - Parameters:
    ///   - initialParameters: Starting parameters (length = 2·depth)
    ///   - progressCallback: Optional progress updates (iteration, cost)
    /// - Returns: QAOA result
    ///
    /// Example:
    /// ```swift
    /// let result = await qaoa.runWithProgress(initialParameters: [0.5, 0.5]) { iter, cost in
    ///     print("[\(iter)] Cost = \(String(format: "%.6f", cost))")
    ///
    ///     await MainActor.run {
    ///         costLabel.text = "\(cost)"
    ///         progressBar.progress = Double(iter) / 200.0
    ///     }
    /// }
    /// ```
    @_optimize(speed)
    @_eagerMove
    public func runWithProgress(
        initialParameters: [Double],
        progressCallback: (@Sendable (Int, Double) async -> Void)?
    ) async -> QAOAResult {
        let expectedParamCount = 2 * depth
        ValidationUtilities.validateParameterVectorLength(
            initialParameters.count,
            expected: expectedParamCount,
            name: "QAOA initialParameters (requires 2 x depth)"
        )

        currentIteration = 0
        currentCost = 0.0

        // Objective function: evaluate cost for given (γ⃗,β⃗)
        let costFunction: @Sendable ([Double]) async -> Double = { parameters in
            // Bind parameters to ansatz circuit
            // Special binding handles coefficient scaling for QAOA
            let concreteCircuit: QuantumCircuit = self.parameterBinder.bind(baseParameters: parameters)

            // Execute circuit
            let state: QuantumState = await self.simulator.execute(concreteCircuit)

            // Compute cost: ⟨ψ|H_p|ψ⟩
            let cost: Double = if let sparseH = self.sparseHamiltonian {
                await sparseH.expectationValue(state: state)
            } else {
                self.costHamiltonian.expectationValue(state: state)
            }

            return cost
        }

        // Progress callback wrapper
        let optimizerProgressCallback: (@Sendable (Int, Double) async -> Void)? = if let callback = progressCallback {
            { iteration, cost in
                await self.updateProgress(iteration: iteration, cost: cost)
                await callback(iteration, cost)
            }
        } else {
            { iteration, cost in
                await self.updateProgress(iteration: iteration, cost: cost)
            }
        }

        // Run classical optimization
        let optimizerResult: OptimizerResult = await optimizer.minimize(
            costFunction,
            from: initialParameters,
            using: convergenceCriteria,
            progress: optimizerProgressCallback
        )

        // Compute final solution probabilities
        let finalCircuit: QuantumCircuit = parameterBinder.bind(baseParameters: optimizerResult.parameters)
        let finalState: QuantumState = await simulator.execute(finalCircuit)
        let solutionProbabilities: [Int: Double] = extractSolutionProbabilities(state: finalState)

        return QAOAResult(
            optimalCost: optimizerResult.value,
            optimalParameters: optimizerResult.parameters,
            solutionProbabilities: solutionProbabilities,
            costHistory: optimizerResult.history,
            iterations: optimizerResult.iterations,
            convergenceReason: optimizerResult.terminationReason,
            functionEvaluations: optimizerResult.evaluations
        )
    }

    // MARK: - Parameter Binding

    /// Extract solution probabilities from final state
    ///
    /// Returns dictionary mapping bitstring indices to probabilities.
    /// Filters to significant probabilities (> 1e-6) for practical analysis.
    ///
    /// - Parameter state: Final quantum state after QAOA circuit
    /// - Returns: Dictionary [bitstring index -> probability]
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

    /// Get current optimization progress
    /// - Returns: Tuple of (iteration, current cost)
    @_effects(readonly)
    public func getProgress() -> (iteration: Int, cost: Double) {
        (currentIteration, currentCost)
    }

    /// Get backend information (sparse or observable)
    @_effects(readonly)
    public func getBackendInfo() async -> String {
        if let sparseH = sparseHamiltonian {
            await "SparseHamiltonian: \(sparseH.backendDescription)"
        } else {
            "Observable: \(costHamiltonian.terms.count) terms"
        }
    }

    // MARK: - Private Helpers

    @inline(__always)
    private func updateProgress(iteration: Int, cost: Double) {
        currentIteration = iteration
        currentCost = cost
    }
}

// MARK: - QAOA Result

/// QAOA optimization result with solution bitstrings and convergence information
///
/// Contains optimal cost, parameters, solution probability distribution,
/// and full convergence history for analysis and visualization.
///
/// **Usage:**
/// ```swift
/// let result = await qaoa.run(initialParameters: initialGuess)
///
/// // Cost information
/// print("Optimal cost: \(result.optimalCost)")
/// print("Parameters: \(result.optimalParameters)")
///
/// // Solution analysis
/// let topSolutions = result.solutionProbabilities.sorted(by: { $0.value > $1.value })
/// for (bitstring, probability) in topSolutions.prefix(5) {
///     let binary = String(bitstring, radix: 2)
///     print("Solution \(binary): \(String(format: "%.3f", probability))")
/// }
///
/// // Convergence diagnostics
/// print("Converged: \(result.convergenceReason)")
/// print("Iterations: \(result.iterations)")
///
/// // Plot cost history
/// for (i, cost) in result.costHistory.enumerated() {
///     print("\(i),\(cost)")
/// }
/// ```
@frozen
public struct QAOAResult: Sendable, CustomStringConvertible {
    /// Optimal cost found: min ⟨ψ|H_p|ψ⟩
    public let optimalCost: Double

    /// Optimal parameters (γ₀,β₀,...,γₚ₋₁,βₚ₋₁)
    public let optimalParameters: [Double]

    /// Solution probability distribution [bitstring index -> probability]
    /// Filtered to probabilities > 1e-6
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
        functionEvaluations: Int
    ) {
        self.optimalCost = optimalCost
        self.optimalParameters = optimalParameters
        self.solutionProbabilities = solutionProbabilities
        self.costHistory = costHistory
        self.iterations = iterations
        self.convergenceReason = convergenceReason
        self.functionEvaluations = functionEvaluations
    }

    public var description: String {
        let paramStr = optimalParameters.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", ")
        let paramSuffix = optimalParameters.count > 4 ? ", ..." : ""

        let topSolutions = topKSolutions(k: 3)
        let solutionsStr = topSolutions.map { bitstring, prob in
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

    /// Get top-k solutions by probability using partial sort (O(n + k log k) vs O(n log n))
    @_optimize(speed)
    public func topKSolutions(k: Int) -> [(bitstring: Int, probability: Double)] {
        let count = solutionProbabilities.count
        guard count > 0, k > 0 else { return [] }

        if count <= k * 4 {
            return solutionProbabilities
                .sorted { $0.value > $1.value }
                .prefix(k)
                .map { (bitstring: $0.key, probability: $0.value) }
        }

        var heap: [(bitstring: Int, probability: Double)] = []
        heap.reserveCapacity(k)

        for (bitstring, prob) in solutionProbabilities {
            if heap.count < k {
                heap.append((bitstring: bitstring, probability: prob))
                if heap.count == k {
                    for i in stride(from: k / 2 - 1, through: 0, by: -1) {
                        siftDown(&heap, i, k)
                    }
                }
            } else if prob > heap[0].probability {
                heap[0] = (bitstring: bitstring, probability: prob)
                siftDown(&heap, 0, k)
            }
        }

        return heap.sorted { $0.probability > $1.probability }
    }

    /// Min-heap sift down helper
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
