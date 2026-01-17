// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Hybrid quantum-classical algorithm for finding ground state energies.
///
/// VQE minimizes ⟨ψ(θ⃗)|H|ψ(θ⃗)⟩ through iterative quantum state preparation and classical parameter optimization.
/// The variational principle guarantees computed energies are upper bounds on the true ground state. Primary
/// application for near-term quantum hardware in molecular chemistry, materials science, optimization, and
/// condensed matter physics.
///
/// The algorithm alternates between quantum circuit execution (state preparation via parameterized ansatz) and
/// classical optimization (parameter updates) until energy convergence. Sparse Hamiltonian backend provides
/// GPU/Accelerate acceleration for molecular systems with typical 0.01-1% sparsity. Circuit depth remains
/// shallow for NISQ compatibility.
///
/// **Example:**
/// ```swift
/// let vqe = VQE(hamiltonian: h2Hamiltonian, ansatz: ansatz, optimizer: COBYLAOptimizer())
/// let result = await vqe.run(from: Array(repeating: 0.1, count: ansatz.parameterCount))
/// print("Ground state energy: \(result.optimalEnergy)")
/// ```
///
/// - Complexity: O(iterations * (circuit_depth * 2^n + hamiltonian_sparsity))
/// - Note: Actor-based for thread safety
/// - SeeAlso: ``HardwareEfficientAnsatz``
/// - SeeAlso: ``Observable``
/// - SeeAlso: ``SparseHamiltonian``
/// - SeeAlso: ``Optimizer``
/// - SeeAlso: ``ConvergenceCriteria``
public actor VQE {
    // MARK: - Configuration

    /// Hamiltonian to minimize
    private let hamiltonian: Observable

    /// Hardware-efficient Ansatz
    private let ansatz: HardwareEfficientAnsatz

    /// Classical optimizer for parameter updates
    private let optimizer: Optimizer

    /// Convergence criteria for optimization
    private let convergence: ConvergenceCriteria

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

    /// Creates VQE optimizer with specified Hamiltonian and ansatz.
    ///
    /// Constructs sparse Hamiltonian backend when enabled for hardware-accelerated expectation values on
    /// GPU/Accelerate. Falls back to Observable for term-by-term measurement if sparse construction fails.
    /// Typical molecular Hamiltonians achieve 0.01-1% sparsity enabling substantial acceleration.
    ///
    /// **Example:**
    /// ```swift
    /// let ansatz = HardwareEfficientAnsatz(qubits: 4, depth: 3)
    /// let vqe = VQE(hamiltonian: hamiltonian, ansatz: ansatz, optimizer: COBYLAOptimizer())
    /// ```
    ///
    /// - Parameters:
    ///   - hamiltonian: Target Hamiltonian for minimization
    ///   - ansatz: Parameterized quantum circuit
    ///   - optimizer: Classical optimization algorithm
    ///   - convergence: Termination criteria
    ///   - useSparseBackend: Enable sparse matrix acceleration
    ///   - useMetalAcceleration: Enable Metal GPU execution
    /// - Complexity: O(hamiltonian_construction)
    public init(
        hamiltonian: Observable,
        ansatz: HardwareEfficientAnsatz,
        optimizer: Optimizer,
        convergence: ConvergenceCriteria = .init(),
        useSparseBackend: Bool = true,
        useMetalAcceleration: Bool = true,
    ) {
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.convergence = convergence
        sparseHamiltonian = useSparseBackend ? SparseHamiltonian(observable: hamiltonian) : nil
        simulator = QuantumSimulator(useMetalAcceleration: useMetalAcceleration)
    }

    // MARK: - Execution

    /// Executes VQE optimization from initial parameters.
    ///
    /// Runs hybrid loop: binds parameters to ansatz, executes circuit, computes expectation value, updates
    /// parameters until convergence. Optional progress callback receives iteration count and current energy
    /// after each optimizer step for UI updates or logging.
    ///
    /// **Example:**
    /// ```swift
    /// let result = await vqe.run(from: initialParams) { iter, energy in print("[\(iter)] \(energy)") }
    /// print("Ground state: \(result.optimalEnergy)")
    /// ```
    ///
    /// - Parameters:
    ///   - parameters: Initial ansatz parameters
    ///   - progress: Optional callback receiving iteration and energy
    /// - Returns: Optimization result with ground state energy and parameters
    /// - Complexity: O(iterations * (circuit_depth * 2^n + hamiltonian_sparsity))
    /// - Precondition: `parameters.count == ansatz.parameterCount`
    /// - SeeAlso: ``Result``
    /// - SeeAlso: ``Progress``
    @_optimize(speed)
    @_eagerMove
    public func run(
        from parameters: [Double],
        progress: (@Sendable (Int, Double) async -> Void)? = nil,
    ) async -> Result {
        ValidationUtilities.validateArrayCount(parameters, expected: ansatz.parameterCount, name: "parameters")

        currentIteration = 0
        currentEnergy = 0.0

        let energyFunction: @Sendable ([Double]) async -> Double = { parameters in
            let concreteCircuit: QuantumCircuit = self.ansatz.circuit.bound(with: parameters)

            let state: QuantumState = await self.simulator.execute(concreteCircuit)

            let energy: Double = if let sparseH = self.sparseHamiltonian {
                await sparseH.expectationValue(of: state)
            } else {
                self.hamiltonian.expectationValue(of: state)
            }

            return energy
        }

        let optimizerProgressCallback: @Sendable (Int, Double) async -> Void = { iteration, energy in
            await self.updateProgress(iteration: iteration, energy: energy)
            await progress?(iteration, energy)
        }

        let optimizerResult: OptimizerResult = await optimizer.minimize(
            energyFunction,
            from: parameters,
            using: convergence,
            progress: optimizerProgressCallback,
        )

        return Result(
            optimalEnergy: optimizerResult.value,
            optimalParameters: optimizerResult.parameters,
            energyHistory: optimizerResult.history,
            iterations: optimizerResult.iterations,
            convergenceReason: optimizerResult.terminationReason,
            functionEvaluations: optimizerResult.evaluations,
        )
    }

    // MARK: - State Queries

    /// Current optimization progress.
    ///
    /// Updated after each optimizer iteration during ``run(from:progress:)`` execution.
    ///
    /// - Complexity: O(1)
    public var progress: Progress {
        Progress(iteration: currentIteration, energy: currentEnergy)
    }

    /// Hamiltonian backend description.
    ///
    /// Indicates whether sparse matrix or observable backend is being used for expectation value computation.
    ///
    /// - Complexity: O(1)
    public var backendInfo: String {
        get async {
            if let sparseH = sparseHamiltonian {
                await "SparseHamiltonian: \(sparseH.backendDescription)"
            } else {
                "Observable: \(hamiltonian.terms.count) terms"
            }
        }
    }

    // MARK: - Private Helpers

    /// Updates current iteration and energy state.
    @inline(__always)
    private func updateProgress(iteration: Int, energy: Double) {
        currentIteration = iteration
        currentEnergy = energy
    }

    // MARK: - Nested Types

    /// Optimization state snapshot.
    ///
    /// Captures current iteration and energy during VQE execution. Updated after each optimizer step.
    ///
    /// **Example:**
    /// ```swift
    /// let state = await vqe.progress
    /// print("Iteration \(state.iteration): E = \(state.energy)")
    /// ```
    @frozen
    public struct Progress: Sendable, Equatable {
        /// Current iteration number
        public let iteration: Int

        /// Current energy value
        public let energy: Double

        @inlinable
        public init(iteration: Int, energy: Double) {
            self.iteration = iteration
            self.energy = energy
        }
    }

    /// VQE optimization result.
    ///
    /// Contains ground state energy (upper bound on true E₀), optimal parameters, and convergence diagnostics.
    /// Energy history enables convergence analysis and visualization.
    ///
    /// **Example:**
    /// ```swift
    /// let result = await vqe.run(from: initialGuess)
    /// print("E₀ = \(result.optimalEnergy), converged: \(result.convergenceReason)")
    /// ```
    ///
    /// - SeeAlso: ``VQE``
    /// - SeeAlso: ``TerminationReason``
    @frozen
    public struct Result: Sendable, CustomStringConvertible {
        /// Ground state energy (upper bound on true E₀)
        public let optimalEnergy: Double

        /// Optimal ansatz parameters
        public let optimalParameters: [Double]

        /// Energy at each iteration
        public let energyHistory: [Double]

        /// Total iterations performed
        public let iterations: Int

        /// Termination condition reached
        public let convergenceReason: TerminationReason

        /// Total function evaluations (includes gradient computations)
        public let functionEvaluations: Int

        public init(
            optimalEnergy: Double,
            optimalParameters: [Double],
            energyHistory: [Double],
            iterations: Int,
            convergenceReason: TerminationReason,
            functionEvaluations: Int,
        ) {
            self.optimalEnergy = optimalEnergy
            self.optimalParameters = optimalParameters
            self.energyHistory = energyHistory
            self.iterations = iterations
            self.convergenceReason = convergenceReason
            self.functionEvaluations = functionEvaluations
        }

        /// Multi-line formatted summary of optimization results.
        public var description: String {
            let paramStr = optimalParameters.prefix(3)
                .map { String(format: "%.4f", $0) }
                .joined(separator: ", ")
            let paramSuffix = optimalParameters.count > 3 ? ", ..." : ""

            return """
            VQE Result:
              Ground State Energy: \(String(format: "%.8f", optimalEnergy))
              Parameters: [\(paramStr)\(paramSuffix)]
              Iterations: \(iterations)
              Function Evaluations: \(functionEvaluations)
              Convergence: \(convergenceReason)
            """
        }
    }
}
