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
/// - Uses SparseHamiltonian backend (100-1000× speedup)
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
/// let result = try await qaoa.run(
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
/// let result = try await qaoa.runWithProgress(
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
    private let ansatz: ParameterizedQuantumCircuit

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

        // Construct QAOA ansatz once at initialization
        ansatz = QAOAAnsatz.create(
            numQubits: numQubits,
            depth: depth,
            costHamiltonian: costHamiltonian,
            mixerHamiltonian: self.mixerHamiltonian
        )
    }

    // MARK: - Execution

    /// Run QAOA optimization
    ///
    /// Executes hybrid quantum-classical loop until convergence or max iterations.
    /// Each iteration:
    /// 1. Bind current (γ⃗,β⃗) to ansatz → concrete circuit
    /// 2. Execute circuit on simulator (GPU-accelerated if available)
    /// 3. Compute ⟨ψ|H_p|ψ⟩ using SparseHamiltonian (or Observable fallback)
    /// 4. Classical optimizer updates parameters
    ///
    /// **Complexity:**
    /// - Per iteration: O(d·2^n + nnz) where d = circuit depth, nnz = Hamiltonian non-zeros
    /// - Total: O(iters × (d·2^n + nnz)) where iters = optimizer iterations
    ///
    /// **Thread Safety:**
    /// - Actor isolation ensures thread-safe execution
    /// - Prevents concurrent QAOA runs
    ///
    /// - Parameter initialParameters: Starting point (γ₀,β₀,...,γₚ₋₁,βₚ₋₁)
    /// - Returns: QAOA result with optimal cost and solution bitstrings
    /// - Throws: QAOAError if optimization fails, CancellationError if cancelled
    ///
    /// Example:
    /// ```swift
    /// let initialGuess = [0.5, 0.5, 0.5, 0.5]  // depth=2
    /// let result = try await qaoa.run(initialParameters: initialGuess)
    ///
    /// print("Optimal cost: \(result.optimalCost)")
    /// print("Parameters: \(result.optimalParameters)")
    /// print("Top solution: \(result.solutionProbabilities.max(by: { $0.value < $1.value })!)")
    /// ```
    @_optimize(speed)
    @_eagerMove
    public func run(initialParameters: [Double]) async throws -> QAOAResult {
        try await runWithProgress(initialParameters: initialParameters, progressCallback: nil)
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
    /// - Throws: QAOAError or CancellationError
    ///
    /// Example:
    /// ```swift
    /// let result = try await qaoa.runWithProgress(initialParameters: [0.5, 0.5]) { iter, cost in
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
    ) async throws -> QAOAResult {
        let expectedParamCount = 2 * depth
        guard initialParameters.count == expectedParamCount else {
            throw QAOAError.parameterCountMismatch(
                expected: expectedParamCount,
                got: initialParameters.count
            )
        }

        currentIteration = 0
        currentCost = 0.0

        // Objective function: evaluate cost for given (γ⃗,β⃗)
        let costFunction: @Sendable ([Double]) async throws -> Double = { parameters in
            // Bind parameters to ansatz circuit
            // Special binding handles coefficient scaling for QAOA
            let concreteCircuit: QuantumCircuit = try await self.bindQAOAParameters(parameters: parameters)

            // Execute circuit
            let state: QuantumState = try await self.simulator.execute(concreteCircuit)

            // Compute cost: ⟨ψ|H_p|ψ⟩
            let cost: Double = if let sparseH = self.sparseHamiltonian {
                await sparseH.expectationValue(state: state)
            } else {
                self.costHamiltonian.expectationValue(state: state)
            }

            guard cost.isFinite else { throw QAOAError.invalidCost(value: cost, parameters: parameters) }

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
        let optimizerResult: OptimizerResult = try await optimizer.minimize(
            objectiveFunction: costFunction,
            initialParameters: initialParameters,
            convergenceCriteria: convergenceCriteria,
            progressCallback: optimizerProgressCallback
        )

        // Compute final solution probabilities
        let finalCircuit: QuantumCircuit = try bindQAOAParameters(parameters: optimizerResult.optimalParameters)
        let finalState: QuantumState = try await simulator.execute(finalCircuit)
        let solutionProbabilities: [Int: Double] = extractSolutionProbabilities(state: finalState)

        return QAOAResult(
            optimalCost: optimizerResult.optimalValue,
            optimalParameters: optimizerResult.optimalParameters,
            solutionProbabilities: solutionProbabilities,
            costHistory: optimizerResult.valueHistory,
            iterations: optimizerResult.iterations,
            convergenceReason: optimizerResult.convergenceReason,
            functionEvaluations: optimizerResult.functionEvaluations
        )
    }

    // MARK: - Parameter Binding

    /// Bind QAOA parameters to ansatz with coefficient scaling
    ///
    /// Handles special QAOA parameter binding where:
    /// - Base parameters: gamma_0, beta_0, ..., gamma_{depth-1}, beta_{depth-1}
    /// - Scaled parameters: gamma_i_c_{coeff} for each Hamiltonian term
    /// - Binding: gamma_0 = v → {gamma_0_c_1.0: v, gamma_0_c_-0.5: -0.5v, ...}
    ///
    /// - Parameter parameters: Array of 2·depth values (γ₀,β₀,...,γₚ₋₁,βₚ₋₁)
    /// - Returns: Concrete quantum circuit with bound parameters
    /// - Throws: ParameterError if binding fails
    @_optimize(speed)
    @_eagerMove
    private func bindQAOAParameters(parameters: [Double]) throws -> QuantumCircuit {
        // Build binding dictionary for scaled parameters only
        var bindings: [String: Double] = [:]
        bindings.reserveCapacity(ansatz.parameterCount())

        // Expand base parameter values to all scaled parameters
        // Ansatz contains parameters like "gamma_0_c_1.000000" which represent 2·γ·c
        // We need to bind these to actual values: parameter_value * encoded_coefficient
        for param in ansatz.parameters {
            let paramName = param.name
            if paramName.contains("_c_") {
                let components = paramName.components(separatedBy: "_c_")
                guard components.count == 2, let coefficientValue = Double(components[1]) else { continue }

                let baseName = components[0]

                let baseComponents = baseName.components(separatedBy: "_")
                guard baseComponents.count == 2, let layer = Int(baseComponents[1]) else { continue }

                let isGamma = baseComponents[0] == "gamma"
                let paramIndex = isGamma ? (2 * layer) : (2 * layer + 1)

                // Scaled value: base_value * coefficient
                // Note: coefficient already includes factor of 2 from Rz(2θ)
                bindings[paramName] = parameters[paramIndex] * coefficientValue
            }
        }

        // Bind ansatz with scaled parameters only
        return try ansatz.bind(parameters: bindings)
    }

    /// Extract solution probabilities from final state
    ///
    /// Returns dictionary mapping bitstring indices to probabilities.
    /// Filters to significant probabilities (> 1e-6) for practical analysis.
    ///
    /// - Parameter state: Final quantum state after QAOA circuit
    /// - Returns: Dictionary [bitstring index → probability]
    @_optimize(speed)
    @_eagerMove
    @inline(__always)
    private func extractSolutionProbabilities(state: QuantumState) -> [Int: Double] {
        var probabilities: [Int: Double] = [:]
        probabilities.reserveCapacity(min(state.stateSpaceSize, 100))

        for i in 0 ..< state.stateSpaceSize {
            let prob: Double = state.probability(ofState: i)
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
/// let result = try await qaoa.run(initialParameters: initialGuess)
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

    /// Solution probability distribution [bitstring index → probability]
    /// Filtered to probabilities > 1e-6
    public let solutionProbabilities: [Int: Double]

    /// Complete cost history (one per iteration)
    public let costHistory: [Double]

    /// Total optimization iterations
    public let iterations: Int

    /// Why optimization terminated
    public let convergenceReason: ConvergenceReason

    /// Total objective function evaluations
    public let functionEvaluations: Int

    public init(
        optimalCost: Double,
        optimalParameters: [Double],
        solutionProbabilities: [Int: Double],
        costHistory: [Double],
        iterations: Int,
        convergenceReason: ConvergenceReason,
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

    @inlinable
    public var description: String {
        let paramStr = optimalParameters.prefix(4).map { String(format: "%.4f", $0) }.joined(separator: ", ")
        let paramSuffix = optimalParameters.count > 4 ? ", ..." : ""

        let topSolutions = solutionProbabilities.sorted(by: { $0.value > $1.value }).prefix(3)
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
}

// MARK: - QAOA Error

@frozen
public enum QAOAError: Error, LocalizedError {
    /// Parameter count mismatch between initialParameters and QAOA depth
    case parameterCountMismatch(expected: Int, got: Int)

    /// Cost evaluation returned invalid value (NaN or Inf)
    case invalidCost(value: Double, parameters: [Double])

    public var errorDescription: String? {
        switch self {
        case let .parameterCountMismatch(expected, got):
            "Parameter count mismatch: QAOA with depth p requires 2p parameters but got \(got) values (expected \(expected)). Check that initialParameters.count == 2 * depth."

        case let .invalidCost(value, _):
            "Cost evaluation returned invalid value: \(value). Check Hamiltonian and circuit validity."
        }
    }
}
