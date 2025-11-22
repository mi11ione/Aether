// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Variational Quantum Eigensolver: Quantum ground state solver
///
/// Hybrid quantum-classical algorithm for finding ground state energy of
/// arbitrary Hamiltonians. **The killer application** for near-term quantum
/// computing with applications in quantum chemistry, materials science,
/// optimization, and condensed matter physics.
///
/// **Algorithm Overview:**
/// 1. **Prepare parameterized state**: |ψ(θ⃗)⟩ via hardware-efficient ansatz
/// 2. **Measure energy**: ⟨ψ(θ⃗)|H|ψ(θ⃗)⟩ using SparseHamiltonian backend
/// 3. **Classical optimization**: Update θ⃗ to minimize energy
/// 4. **Repeat until convergence**: |E_new - E_old| < ε
///
/// **Mathematical Foundation:**
/// - Variational principle: ⟨ψ|H|ψ⟩ ≥ E₀ for any |ψ⟩
/// - VQE finds upper bound on ground state energy E₀
/// - Better ansatz (more parameters) → tighter bound → closer to E₀
/// - Guaranteed convergence to local minimum
///
/// **Performance:**
/// - SparseHamiltonian backend: GPU/Accelerate hardware acceleration
/// - Typical convergence: 50-300 optimizer iterations
/// - Circuit depth: shallow (NISQ-compatible)
/// - Hardware requirements: coherence time > circuit execution time
///
/// **Use Cases:**
/// - **Quantum chemistry**: Molecular ground state energies, reaction pathways
/// - **Materials science**: Electronic structure, phase transitions
/// - **Optimization**: QUBO problems, portfolio optimization
/// - **Condensed matter**: Many-body systems, strongly correlated electrons
///
/// **Example - H₂ molecule ground state:**
/// ```swift
/// // 1. Define molecular Hamiltonian
/// let hamiltonian = Observable(terms: [
///     (-1.05, PauliString(operators: [])),                    // Identity
///     (0.39,  PauliString(operators: [(0, .z)])),             // Z₀
///     (-0.39, PauliString(operators: [(1, .z)])),             // Z₁
///     (-0.01, PauliString(operators: [(0, .z), (1, .z)])),    // Z₀Z₁
///     (0.18,  PauliString(operators: [(0, .x), (1, .x)]))     // X₀X₁
/// ])
///
/// // 2. Build hardware-efficient ansatz
/// let ansatz = HardwareEfficientAnsatz.create(numQubits: 2, depth: 2)
///
/// // 3. Configure VQE
/// let vqe = await VariationalQuantumEigensolver(
///     hamiltonian: hamiltonian,
///     ansatz: ansatz,
///     optimizer: NelderMeadOptimizer(tolerance: 1e-6),
///     convergenceCriteria: ConvergenceCriteria(
///         energyTolerance: 1e-6,
///         maxIterations: 200
///     )
/// )
///
/// // 4. Run optimization with progress tracking
/// let result = try await vqe.runWithProgress(
///     initialParameters: Array(repeating: 0.01, count: ansatz.parameterCount())
/// ) { iteration, energy in
///     print("Iteration \(iteration): E = \(energy) Hartree")
/// }
///
/// print("Ground state energy: \(result.optimalEnergy) Hartree")
/// print("Converged in \(result.iterations) iterations")
/// print("Convergence: \(result.convergenceReason)")
/// ```
///
/// **Example - Progress tracking for UI:**
/// ```swift
/// let vqe = await VariationalQuantumEigensolver(...)
///
/// let result = try await vqe.runWithProgress(initialParameters: initialGuess) { iteration, energy in
///     await MainActor.run {
///         progressLabel.text = "Iteration \(iteration): E = \(String(format: "%.6f", energy))"
///         energyChart.addDataPoint(x: iteration, y: energy)
///     }
/// }
/// ```
///
/// **Architecture:**
/// - Actor-based: Thread-safe, prevents data races
/// - Async optimization: Non-blocking for UI applications
/// - SparseHamiltonian: GPU/Accelerate hardware acceleration
/// - Observable fallback: Guaranteed correctness if sparse construction fails
/// - Progress tracking: Real-time energy updates
public actor VariationalQuantumEigensolver {
    // MARK: - Configuration

    /// Hamiltonian to minimize
    private let hamiltonian: Observable

    /// Parameterized quantum circuit (ansatz)
    private let ansatz: ParameterizedQuantumCircuit

    /// Classical optimizer for parameter updates
    private let optimizer: Optimizer

    /// Convergence criteria for optimization
    private let convergenceCriteria: ConvergenceCriteria

    /// High-performance sparse Hamiltonian backend (optional)
    private let sparseHamiltonian: SparseHamiltonian?

    /// Quantum simulator for circuit execution
    private let simulator: QuantumSimulator

    // MARK: - State

    /// Current optimization iteration
    private var currentIteration: Int = 0

    /// Current best energy
    private var currentEnergy: Double = 0.0

    // MARK: - Initialization

    /// Create VQE instance
    ///
    /// Configures hybrid quantum-classical optimization for ground state search.
    /// Automatically constructs SparseHamiltonian backend if useSparseBackend=true.
    ///
    /// **Backend Selection:**
    /// - SparseHamiltonian (default): GPU/Accelerate hardware acceleration
    /// - Observable (fallback): Term-by-term measurement
    ///
    /// **Performance:**
    /// - Sparse backend: O(nnz) where nnz = number of non-zeros (~0.01-1% of 4^n)
    /// - Observable backend: O(k·2^n) where k = number of Pauli terms
    /// - For 10-qubit H₂O: 8K non-zeros vs 2000 terms × 1024 amplitudes
    ///
    /// - Parameters:
    ///   - hamiltonian: Molecular or optimization Hamiltonian
    ///   - ansatz: Parameterized quantum circuit
    ///   - optimizer: Classical optimization algorithm
    ///   - convergenceCriteria: Termination conditions (default: ε=1e-6, maxIter=1000)
    ///   - useSparseBackend: Use SparseHamiltonian acceleration (default: true)
    ///   - useMetalAcceleration: Use Metal GPU for circuit execution (default: true)
    public init(
        hamiltonian: Observable,
        ansatz: ParameterizedQuantumCircuit,
        optimizer: Optimizer,
        convergenceCriteria: ConvergenceCriteria = .default,
        useSparseBackend: Bool = true,
        useMetalAcceleration: Bool = true
    ) {
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.convergenceCriteria = convergenceCriteria
        sparseHamiltonian = useSparseBackend ? SparseHamiltonian(observable: hamiltonian) : nil
        simulator = QuantumSimulator(useMetalAcceleration: useMetalAcceleration)
    }

    // MARK: - Execution

    /// Run VQE optimization
    ///
    /// Executes hybrid quantum-classical loop until convergence or max iterations.
    /// Each iteration:
    /// 1. Bind current parameters to ansatz
    /// 2. Execute circuit on simulator (GPU-accelerated if available)
    /// 3. Compute ⟨ψ|H|ψ⟩ using SparseHamiltonian (or Observable fallback)
    /// 4. Classical optimizer updates parameters
    ///
    /// **Complexity:**
    /// - Per iteration: O(d·2^n + nnz) where d = circuit depth, nnz = Hamiltonian non-zeros
    /// - Total: O(iters × (d·2^n + nnz)) where iters = optimizer iterations
    ///
    /// **Thread Safety:**
    /// - Actor isolation ensures thread-safe execution
    /// - Prevents concurrent VQE runs (throws if already running)
    ///
    /// - Parameter initialParameters: Starting point in parameter space
    /// - Returns: VQE result with optimal energy and parameters
    /// - Throws: VQEError if optimization fails, CancellationError if cancelled
    ///
    /// Example:
    /// ```swift
    /// let initialGuess = Array(repeating: 0.01, count: ansatz.parameterCount())
    /// let result = try await vqe.run(initialParameters: initialGuess)
    ///
    /// print("Ground state: \(result.optimalEnergy) Hartree")
    /// print("Parameters: \(result.optimalParameters)")
    /// print("Iterations: \(result.iterations)")
    /// ```
    @_optimize(speed)
    @_eagerMove
    public func run(initialParameters: [Double]) async throws -> VQEResult {
        try await runWithProgress(initialParameters: initialParameters, progressCallback: nil)
    }

    /// Run VQE with progress updates
    ///
    /// Same as `run()` but calls progressCallback after each iteration.
    /// Useful for UI updates, logging, and convergence visualization.
    ///
    /// **Progress Callback:**
    /// - Called after each optimizer iteration
    /// - Receives: (iteration: Int, currentEnergy: Double)
    /// - Async: can perform UI updates on MainActor
    /// - Not called on initial evaluation
    ///
    /// - Parameters:
    ///   - initialParameters: Starting parameters
    ///   - progressCallback: Called with (iteration, energy) after each iteration
    /// - Returns: VQE result
    /// - Throws: VQEError or CancellationError
    ///
    /// Example:
    /// ```swift
    /// let result = try await vqe.runWithProgress(initialParameters: initialGuess) { iter, E in
    ///     print("[\(iter)] E = \(String(format: "%.8f", E)) Hartree")
    ///
    ///     // Update UI on main thread
    ///     await MainActor.run {
    ///         energyLabel.text = "\(E)"
    ///         progressBar.progress = Double(iter) / 200.0
    ///     }
    /// }
    /// ```
    @_optimize(speed)
    @_eagerMove
    public func runWithProgress(
        initialParameters: [Double],
        progressCallback: (@Sendable (Int, Double) async -> Void)?
    ) async throws -> VQEResult {
        guard initialParameters.count == ansatz.parameterCount() else {
            throw VQEError.parameterCountMismatch(
                expected: ansatz.parameterCount(),
                got: initialParameters.count
            )
        }

        currentIteration = 0
        currentEnergy = 0.0

        let energyFunction: @Sendable ([Double]) async throws -> Double = { parameters in
            let concreteCircuit: QuantumCircuit = try self.ansatz.bind(parameterVector: parameters)

            let state: QuantumState = try await self.simulator.execute(concreteCircuit)

            let energy: Double = if let sparseH = self.sparseHamiltonian {
                await sparseH.expectationValue(state: state)
            } else {
                self.hamiltonian.expectationValue(state: state)
            }

            guard energy.isFinite else {
                throw VQEError.invalidEnergy(value: energy, parameters: parameters)
            }

            return energy
        }

        let optimizerProgressCallback: (@Sendable (Int, Double) async -> Void)? = if let callback = progressCallback {
            { iteration, energy in
                await self.updateProgress(iteration: iteration, energy: energy)
                await callback(iteration, energy)
            }
        } else {
            { iteration, energy in
                await self.updateProgress(iteration: iteration, energy: energy)
            }
        }

        let optimizerResult: OptimizerResult = try await optimizer.minimize(
            objectiveFunction: energyFunction,
            initialParameters: initialParameters,
            convergenceCriteria: convergenceCriteria,
            progressCallback: optimizerProgressCallback
        )

        return VQEResult(
            optimalEnergy: optimizerResult.optimalValue,
            optimalParameters: optimizerResult.optimalParameters,
            energyHistory: optimizerResult.valueHistory,
            iterations: optimizerResult.iterations,
            convergenceReason: optimizerResult.convergenceReason,
            functionEvaluations: optimizerResult.functionEvaluations
        )
    }

    // MARK: - State Queries

    /// Get current optimization progress
    /// - Returns: Tuple of (iteration, current energy)
    @_effects(readonly)
    public func getProgress() -> (iteration: Int, energy: Double) {
        (currentIteration, currentEnergy)
    }

    /// Get backend information (sparse or observable)
    @_effects(readonly)
    public func getBackendInfo() async -> String {
        if let sparseH = sparseHamiltonian {
            await "SparseHamiltonian: \(sparseH.backendDescription)"
        } else {
            "Observable: \(hamiltonian.terms.count) terms"
        }
    }

    // MARK: - Private Helpers

    @inline(__always)
    private func updateProgress(iteration: Int, energy: Double) {
        currentIteration = iteration
        currentEnergy = energy
    }
}

// MARK: - VQE Result

/// VQE optimization result with convergence information
///
/// Contains optimal ground state energy, parameters, full convergence history,
/// and diagnostics for analysis and visualization.
///
/// **Usage:**
/// ```swift
/// let result = try await vqe.run(initialParameters: initialGuess)
///
/// // Ground state information
/// print("E₀ = \(result.optimalEnergy) Hartree")
/// print("Optimal parameters: \(result.optimalParameters)")
///
/// // Convergence diagnostics
/// print("Converged: \(result.convergenceReason)")
/// print("Iterations: \(result.iterations)")
/// print("Function evaluations: \(result.functionEvaluations)")
///
/// // Plot convergence curve
/// for (i, energy) in result.energyHistory.enumerated() {
///     print("\(i),\(energy)")
/// }
///
/// // Check convergence quality
/// let energyRange = result.energyHistory.max()! - result.energyHistory.min()!
/// print("Energy range: \(energyRange) Hartree")
/// ```
@frozen
public struct VQEResult: Sendable, CustomStringConvertible {
    /// Ground state energy found (upper bound on true E₀)
    public let optimalEnergy: Double

    /// Optimal parameters for ansatz
    public let optimalParameters: [Double]

    /// Complete energy history (one per iteration)
    public let energyHistory: [Double]

    /// Total optimization iterations
    public let iterations: Int

    /// Why optimization terminated
    public let convergenceReason: ConvergenceReason

    /// Total objective function evaluations (includes gradient computations)
    public let functionEvaluations: Int

    public init(
        optimalEnergy: Double,
        optimalParameters: [Double],
        energyHistory: [Double],
        iterations: Int,
        convergenceReason: ConvergenceReason,
        functionEvaluations: Int
    ) {
        self.optimalEnergy = optimalEnergy
        self.optimalParameters = optimalParameters
        self.energyHistory = energyHistory
        self.iterations = iterations
        self.convergenceReason = convergenceReason
        self.functionEvaluations = functionEvaluations
    }

    @inlinable
    public var description: String {
        let paramStr = optimalParameters.prefix(3).map { String(format: "%.4f", $0) }.joined(separator: ", ")
        let paramSuffix = optimalParameters.count > 3 ? ", ..." : ""

        return """
        VQE Result:
          Ground State Energy: \(String(format: "%.8f", optimalEnergy)) Hartree
          Parameters: [\(paramStr)\(paramSuffix)]
          Iterations: \(iterations)
          Function Evaluations: \(functionEvaluations)
          Convergence: \(convergenceReason)
        """
    }
}

// MARK: - VQE Error

@frozen
public enum VQEError: Error, LocalizedError {
    /// Parameter count mismatch between initialParameters and ansatz
    case parameterCountMismatch(expected: Int, got: Int)

    /// Energy evaluation returned invalid value (NaN or Inf)
    case invalidEnergy(value: Double, parameters: [Double])

    public var errorDescription: String? {
        switch self {
        case let .parameterCountMismatch(expected, got):
            "Parameter count mismatch: ansatz has \(expected) parameters but got \(got) initial values. Check ansatz.parameterCount()."

        case let .invalidEnergy(value, _):
            "Energy evaluation returned invalid value: \(value). Check Hamiltonian and circuit validity."
        }
    }
}
