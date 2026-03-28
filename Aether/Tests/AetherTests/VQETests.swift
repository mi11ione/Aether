// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Testing

/// Test suite for VariationalQuantumEigensolver.
/// Validates VQE initialization, execution, convergence, and error handling
/// for ground state energy computation.
@Suite("VariationalQuantumEigensolver")
struct VariationalQuantumEigensolverTests {
    @Test("Create VQE without sparse backend")
    func createVQEWithoutSparseBackend() async {
        let hamiltonian = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1)
        let optimizer = NelderMeadOptimizer()

        let vqe = VQE(
            hamiltonian: hamiltonian,
            ansatz: ansatz,
            optimizer: optimizer,
            isSparseEnabled: false,
        )

        let backendInfo = await vqe.backendInfo
        #expect(backendInfo.contains("Observable"), "Observable backend should be used when sparse disabled")
    }

    @Test("VQE finds ground state of simple Hamiltonian")
    func findGroundStateSimpleHamiltonian() async {
        let hamiltonian = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1)
        let optimizer = NelderMeadOptimizer(tolerance: 1e-3)

        let vqe = VQE(
            hamiltonian: hamiltonian,
            ansatz: ansatz,
            optimizer: optimizer,
            convergence: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 100),
        )

        let result = await vqe.run(from: [0.1])

        #expect(result.optimalEnergy < 0.0, "Ground state energy should be negative")
        #expect(result.optimalEnergy > -1.1, "Energy should be close to -1.0")
        #expect(result.iterations > 0, "Should perform at least one iteration")
    }

    @Test("VQE with progress callback")
    func vqeWithProgressCallback() async {
        actor CallbackState {
            var count = 0
            var lastEnergy: Double = 0.0
            func recordCallback(energy: Double) {
                count += 1
                lastEnergy = energy
            }

            func getCount() -> Int {
                count
            }

            func getLastEnergy() -> Double {
                lastEnergy
            }
        }

        let hamiltonian = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1)
        let optimizer = NelderMeadOptimizer(tolerance: 1e-3)

        let vqe = VQE(
            hamiltonian: hamiltonian,
            ansatz: ansatz,
            optimizer: optimizer,
            convergence: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 50),
        )

        let state = CallbackState()

        let result = await vqe.run(from: [0.1]) { _, energy in
            await state.recordCallback(energy: energy)
        }

        let callbackCount = await state.getCount()
        let lastEnergy = await state.getLastEnergy()

        #expect(callbackCount > 0, "Progress callback should be called at least once")
        #expect(abs(lastEnergy - result.optimalEnergy) < 1e-6, "Last callback energy should match final result")
    }

    @Test("VQE tracks progress")
    func vqeTracksProgress() async {
        let hamiltonian = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1)
        let optimizer = NelderMeadOptimizer(tolerance: 1e-3)

        let vqe = VQE(
            hamiltonian: hamiltonian,
            ansatz: ansatz,
            optimizer: optimizer,
            convergence: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 50),
        )

        _ = await vqe.run(from: [0.1]) { iteration, energy in
            let currentProgress = await vqe.progress
            #expect(currentProgress.iteration == iteration)
            #expect(abs(currentProgress.energy - energy) < 1e-10)
        }
    }

    @Test("VQE with two-qubit Hamiltonian")
    func twoQubitHamiltonian() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
            (1.0, PauliString(.z(1))),
        ])

        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 1)
        let optimizer = NelderMeadOptimizer(tolerance: 1e-3)

        let vqe = VQE(
            hamiltonian: hamiltonian,
            ansatz: ansatz,
            optimizer: optimizer,
            convergence: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 100),
        )

        let result = await vqe.run(from: [0.1, 0.1])

        #expect(result.optimalEnergy < -1.5, "Two-qubit ground state should be below -1.5")
        #expect(result.optimalEnergy > -2.1, "Energy should be close to -2.0")
    }

    @Test("VQE convergence via energy tolerance")
    func convergenceViaEnergyTolerance() async {
        let hamiltonian = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1)
        let optimizer = NelderMeadOptimizer(tolerance: 1e-3)

        let vqe = VQE(
            hamiltonian: hamiltonian,
            ansatz: ansatz,
            optimizer: optimizer,
            convergence: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 100),
        )

        let result = await vqe.run(from: [0.1])

        #expect(result.convergenceReason == .energyConverged, "Should converge via energy tolerance")
    }

    @Test("VQE with gradient descent optimizer")
    func vqeWithGradientDescent() async {
        let hamiltonian = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1)
        let optimizer = GradientDescentOptimizer(learningRate: 0.5)

        let vqe = VQE(
            hamiltonian: hamiltonian,
            ansatz: ansatz,
            optimizer: optimizer,
            convergence: ConvergenceCriteria(
                energyTolerance: 1e-2,
                maxIterations: 50,
            ),
        )

        let result = await vqe.run(from: [2.0])

        #expect(result.optimalEnergy < 0.0, "Gradient descent should find negative energy")
    }

    @Test("VQE with L-BFGS-B optimizer")
    func vqeWithLBFGSB() async {
        let hamiltonian = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1)
        let optimizer = LBFGSBOptimizer(tolerance: 1e-3)

        let vqe = VQE(
            hamiltonian: hamiltonian,
            ansatz: ansatz,
            optimizer: optimizer,
            convergence: ConvergenceCriteria(
                energyTolerance: 1e-3,
                maxIterations: 50,
            ),
        )

        let result = await vqe.run(from: [2.0])

        #expect(result.optimalEnergy < 0.0, "L-BFGS-B should find negative energy")
    }

    @Test("VQE with SPSA optimizer")
    func vqeWithSPSA() async {
        let hamiltonian = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1)
        let optimizer = SPSAOptimizer(initialStepSize: 0.1)

        let vqe = VQE(
            hamiltonian: hamiltonian,
            ansatz: ansatz,
            optimizer: optimizer,
            convergence: ConvergenceCriteria(energyTolerance: 0.05, maxIterations: 100),
        )

        let result = await vqe.run(from: [2.0])

        #expect(result.optimalEnergy < 0.0, "SPSA should find negative energy")
    }

    @Test("VQE backend info with sparse")
    func backendInfoWithSparse() async {
        let hamiltonian = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1)
        let optimizer = NelderMeadOptimizer()

        let vqe = VQE(
            hamiltonian: hamiltonian,
            ansatz: ansatz,
            optimizer: optimizer,
            isSparseEnabled: true,
        )

        let backendInfo = await vqe.backendInfo
        #expect(backendInfo.contains("Sparse") || backendInfo.contains("Observable"), "Backend info should mention Sparse or Observable")
    }

    @Test("VQE backend info without sparse")
    func backendInfoWithoutSparse() async {
        let hamiltonian = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1)
        let optimizer = NelderMeadOptimizer()

        let vqe = VQE(
            hamiltonian: hamiltonian,
            ansatz: ansatz,
            optimizer: optimizer,
            isSparseEnabled: false,
        )

        let backendInfo = await vqe.backendInfo
        #expect(backendInfo.contains("Observable"), "Should use Observable backend")
        #expect(backendInfo.contains("1 term"), "Should report 1 term")
    }

    @Test("VQE without Metal acceleration")
    func vqeWithoutMetalAcceleration() async {
        let hamiltonian = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1)
        let optimizer = NelderMeadOptimizer(tolerance: 1e-3)

        let vqe = VQE(
            hamiltonian: hamiltonian,
            ansatz: ansatz,
            optimizer: optimizer,
            convergence: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 100),
            precisionPolicy: .accurate,
        )

        let result = await vqe.run(from: [0.1])

        #expect(result.optimalEnergy < 0.0, "Accurate policy should find negative energy")
    }

    @Test("VQE runs with Observable backend when sparse disabled")
    func vqeRunsWithObservableBackend() async {
        let hamiltonian = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1)
        let optimizer = NelderMeadOptimizer(tolerance: 1e-3)

        let vqe = VQE(
            hamiltonian: hamiltonian,
            ansatz: ansatz,
            optimizer: optimizer,
            convergence: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 100),
            isSparseEnabled: false,
        )

        let result = await vqe.run(from: [0.1])

        #expect(result.optimalEnergy < 0.0, "Should find negative ground state energy")
        #expect(result.optimalEnergy > -1.1, "Energy should be close to -1.0")
    }

    @Test("VQE tracks function evaluations")
    func vqeTracksFunctionEvaluations() async {
        let hamiltonian = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1)
        let optimizer = NelderMeadOptimizer(tolerance: 1e-3)

        let vqe = VQE(
            hamiltonian: hamiltonian,
            ansatz: ansatz,
            optimizer: optimizer,
            convergence: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 50),
        )

        let result = await vqe.run(from: [0.1])

        #expect(result.functionEvaluations > result.iterations, "Function evaluations should exceed iterations")
    }
}

/// Test suite for VQE.Result.
/// Validates VQE result creation,
/// field access, and description.
@Suite("VQE.Result")
struct VQEResultTests {
    @Test("Create VQE result")
    func createVQEResult() {
        let result = VQE.Result(
            optimalEnergy: -1.234,
            optimalParameters: [0.5, 1.0, 1.5],
            energyHistory: [-2.0, -1.5, -1.234],
            iterations: 3,
            convergenceReason: .energyConverged,
            functionEvaluations: 15,
        )

        #expect(result.optimalEnergy == -1.234, "Optimal energy should be preserved")
        #expect(result.optimalParameters == [0.5, 1.0, 1.5], "Parameters should be preserved")
        #expect(result.energyHistory == [-2.0, -1.5, -1.234], "Energy history should be preserved")
        #expect(result.iterations == 3, "Iterations should be preserved")
        #expect(result.convergenceReason == .energyConverged, "Convergence reason should be preserved")
        #expect(result.functionEvaluations == 15, "Function evaluations should be preserved")
    }

    @Test("VQE result description")
    func vqeResultDescription() {
        let result = VQE.Result(
            optimalEnergy: -1.234,
            optimalParameters: [0.5, 1.0],
            energyHistory: [-1.234],
            iterations: 10,
            convergenceReason: .energyConverged,
            functionEvaluations: 50,
        )

        let description = result.description

        #expect(description.contains("Ground State Energy"), "Description should contain energy label")
        #expect(description.contains("-1.234"), "Description should contain energy value")
        #expect(description.contains("10"), "Description should contain iteration count")
        #expect(description.contains("50"), "Description should contain evaluation count")
        #expect(description.contains("Energy tolerance"), "Description should contain convergence reason")
    }

    @Test("VQE result description with many parameters")
    func vqeResultDescriptionManyParameters() {
        let result = VQE.Result(
            optimalEnergy: -1.0,
            optimalParameters: [1.0, 2.0, 3.0, 4.0, 5.0],
            energyHistory: [-1.0],
            iterations: 5,
            convergenceReason: .maxIterationsReached,
            functionEvaluations: 20,
        )

        let description = result.description

        #expect(description.contains("..."), "Description should truncate long parameter lists")
    }

    @Test("VQE result with gradient norm convergence")
    func vqeResultGradientNormConvergence() {
        let result = VQE.Result(
            optimalEnergy: -1.0,
            optimalParameters: [0.5],
            energyHistory: [-1.0],
            iterations: 5,
            convergenceReason: .gradientConverged,
            functionEvaluations: 25,
        )

        #expect(result.convergenceReason == .gradientConverged, "Should store gradient convergence reason")
        #expect(result.description.contains("Gradient"), "Description should mention gradient convergence")
    }
}
