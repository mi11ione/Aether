// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
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
            useSparseBackend: false
        )

        let backendInfo = await vqe.getBackendInfo()
        #expect(backendInfo.contains("Observable"))
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
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 100)
        )

        let result = await vqe.run(initialParameters: [0.1])

        #expect(result.optimalEnergy < 0.0)
        #expect(result.optimalEnergy > -1.1)
        #expect(result.iterations > 0)
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

            func getCount() -> Int { count }
            func getLastEnergy() -> Double { lastEnergy }
        }

        let hamiltonian = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1)
        let optimizer = NelderMeadOptimizer(tolerance: 1e-3)

        let vqe = VQE(
            hamiltonian: hamiltonian,
            ansatz: ansatz,
            optimizer: optimizer,
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 50)
        )

        let state = CallbackState()

        let result = await vqe.runWithProgress(initialParameters: [0.1]) { _, energy in
            await state.recordCallback(energy: energy)
        }

        let callbackCount = await state.getCount()
        let lastEnergy = await state.getLastEnergy()

        #expect(callbackCount > 0)
        #expect(abs(lastEnergy - result.optimalEnergy) < 1e-6)
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
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 50)
        )

        _ = await vqe.runWithProgress(initialParameters: [0.1]) { iteration, energy in
            let (currentIter, currentEnergy) = await vqe.getProgress()
            #expect(currentIter == iteration)
            #expect(abs(currentEnergy - energy) < 1e-10)
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
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 100)
        )

        let result = await vqe.run(initialParameters: [0.1, 0.1])

        #expect(result.optimalEnergy < -1.5)
        #expect(result.optimalEnergy > -2.1)
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
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 100)
        )

        let result = await vqe.run(initialParameters: [0.1])

        #expect(result.convergenceReason == .energyConverged)
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
            convergenceCriteria: ConvergenceCriteria(
                energyTolerance: 1e-2,
                maxIterations: 50
            )
        )

        let result = await vqe.run(initialParameters: [2.0])

        #expect(result.optimalEnergy < 0.0)
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
            convergenceCriteria: ConvergenceCriteria(
                energyTolerance: 1e-3,
                maxIterations: 50
            )
        )

        let result = await vqe.run(initialParameters: [2.0])

        #expect(result.optimalEnergy < 0.0)
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
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 0.05, maxIterations: 100)
        )

        let result = await vqe.run(initialParameters: [2.0])

        #expect(result.optimalEnergy < 0.0)
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
            useSparseBackend: true
        )

        let backendInfo = await vqe.getBackendInfo()
        #expect(backendInfo.contains("Sparse") || backendInfo.contains("Observable"))
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
            useSparseBackend: false
        )

        let backendInfo = await vqe.getBackendInfo()
        #expect(backendInfo.contains("Observable"))
        #expect(backendInfo.contains("1 term"))
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
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 100),
            useMetalAcceleration: false
        )

        let result = await vqe.run(initialParameters: [0.1])

        #expect(result.optimalEnergy < 0.0)
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
            convergenceCriteria: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 50)
        )

        let result = await vqe.run(initialParameters: [0.1])

        #expect(result.functionEvaluations > result.iterations)
    }
}

/// Test suite for VQEResult.
/// Validates VQE result creation,
/// field access, and description.
@Suite("VQEResult")
struct VQEResultTests {
    @Test("Create VQE result")
    func createVQEResult() {
        let result = VQEResult(
            optimalEnergy: -1.234,
            optimalParameters: [0.5, 1.0, 1.5],
            energyHistory: [-2.0, -1.5, -1.234],
            iterations: 3,
            convergenceReason: .energyConverged,
            functionEvaluations: 15
        )

        #expect(result.optimalEnergy == -1.234)
        #expect(result.optimalParameters == [0.5, 1.0, 1.5])
        #expect(result.energyHistory == [-2.0, -1.5, -1.234])
        #expect(result.iterations == 3)
        #expect(result.convergenceReason == .energyConverged)
        #expect(result.functionEvaluations == 15)
    }

    @Test("VQE result description")
    func vqeResultDescription() {
        let result = VQEResult(
            optimalEnergy: -1.234,
            optimalParameters: [0.5, 1.0],
            energyHistory: [-1.234],
            iterations: 10,
            convergenceReason: .energyConverged,
            functionEvaluations: 50
        )

        let description = result.description

        #expect(description.contains("Ground State Energy"))
        #expect(description.contains("-1.234"))
        #expect(description.contains("10"))
        #expect(description.contains("50"))
        #expect(description.contains("Energy tolerance"))
    }

    @Test("VQE result description with many parameters")
    func vqeResultDescriptionManyParameters() {
        let result = VQEResult(
            optimalEnergy: -1.0,
            optimalParameters: [1.0, 2.0, 3.0, 4.0, 5.0],
            energyHistory: [-1.0],
            iterations: 5,
            convergenceReason: .maxIterationsReached,
            functionEvaluations: 20
        )

        let description = result.description

        #expect(description.contains("..."))
    }

    @Test("VQE result with gradient norm convergence")
    func vqeResultGradientNormConvergence() {
        let result = VQEResult(
            optimalEnergy: -1.0,
            optimalParameters: [0.5],
            energyHistory: [-1.0],
            iterations: 5,
            convergenceReason: .gradientConverged,
            functionEvaluations: 25
        )

        #expect(result.convergenceReason == .gradientConverged)
        #expect(result.description.contains("Gradient"))
    }
}
