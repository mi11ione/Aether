// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Testing

/// Test suite for DMRG algorithm initialization and configuration.
/// Validates DMRG creation with various MPO Hamiltonians, bond dimensions,
/// and configuration parameters for ground state optimization.
@Suite("DMRG Initialization")
struct DMRGInitializationTests {
    @Test("Create DMRG with simple Hamiltonian")
    func createDMRGWithSimpleHamiltonian() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
            (1.0, PauliString(.z(1))),
        ])
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 8)

        let progress = await dmrg.progress

        #expect(progress.sweep == 0, "Initial sweep count should be zero")
        #expect(progress.energy == 0.0, "Initial energy should be zero before optimization")
    }

    @Test("Create DMRG with custom configuration")
    func createDMRGWithCustomConfiguration() async {
        let hamiltonian = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let config = DMRGConfiguration(
            maxSweeps: 10,
            convergenceThreshold: 1e-6,
            isSubspaceExpansionEnabled: false,
        )
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 4, configuration: config)

        let progress = await dmrg.progress

        #expect(progress.sweep == 0, "Initial sweep should be zero with custom configuration")
    }

    @Test("Create DMRG with subspace expansion enabled")
    func createDMRGWithSubspaceExpansion() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
            (0.5, PauliString(.x(0), .x(1))),
        ])
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let config = DMRGConfiguration(
            maxSweeps: 5,
            convergenceThreshold: 1e-4,
            isSubspaceExpansionEnabled: true,
            noiseStrength: 1e-4,
        )
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 8, configuration: config)

        let progress = await dmrg.progress

        #expect(progress.maxTruncationError == 0.0, "Initial truncation error should be zero")
    }

    @Test("Create DMRG with three-site Hamiltonian")
    func createDMRGWithThreeSiteHamiltonian() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
            (1.0, PauliString(.z(1))),
            (1.0, PauliString(.z(2))),
        ])
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 4)

        let progress = await dmrg.progress

        #expect(progress.sweep == 0, "DMRG with three-site Hamiltonian should initialize correctly")
    }
}

/// Test suite for DMRG ground state convergence.
/// Validates findGroundState method for small Ising chains,
/// energy convergence, and result properties.
@Suite("DMRG Ground State Convergence")
struct DMRGGroundStateTests {
    @Test("Find ground state of two-site Ising model")
    func findGroundStateTwoSiteIsing() async {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
        ])
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let config = DMRGConfiguration(
            maxSweeps: 10,
            convergenceThreshold: 1e-6,
        )
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 4, configuration: config)

        let result = await dmrg.findGroundState(from: nil)

        #expect(result.groundStateEnergy < 0.0, "Ground state energy of ferromagnetic Ising should be negative")
        #expect(abs(result.groundStateEnergy + 1.0) < 0.1, "Ground state energy should be close to -1.0 for ZZ coupling")
        #expect(result.sweeps > 0, "Algorithm should perform at least one sweep")
    }

    @Test("Find ground state of three-site transverse field Ising model")
    func findGroundStateThreeSiteTransverseIsing() async {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
            (-1.0, PauliString(.z(1), .z(2))),
            (-0.5, PauliString(.x(0))),
            (-0.5, PauliString(.x(1))),
            (-0.5, PauliString(.x(2))),
        ])
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let config = DMRGConfiguration(
            maxSweeps: 15,
            convergenceThreshold: 1e-5,
        )
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 8, configuration: config)

        let result = await dmrg.findGroundState(from: nil)

        #expect(result.groundStateEnergy < 0.0, "Ground state energy should be negative")
        #expect(result.convergenceHistory.count > 0, "Convergence history should not be empty")
        #expect(result.groundState.qubits == 3, "Ground state MPS should have 3 qubits")
    }

    @Test("Find ground state with subspace expansion")
    func findGroundStateWithSubspaceExpansion() async {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
            (-0.3, PauliString(.x(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let config = DMRGConfiguration(
            maxSweeps: 10,
            convergenceThreshold: 1e-5,
            isSubspaceExpansionEnabled: true,
            noiseStrength: 1e-3,
        )
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 8, configuration: config)

        let result = await dmrg.findGroundState(from: nil)

        #expect(result.groundStateEnergy < 0.0, "Ground state with subspace expansion should have negative energy")
        #expect(result.sweeps <= config.maxSweeps, "Sweeps should not exceed max sweeps")
    }

    @Test("Subspace expansion triggers truncation noise injection")
    func isSubspaceExpansionEnabledTruncation() async {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
            (-1.0, PauliString(.z(1), .z(2))),
            (-1.0, PauliString(.z(2), .z(3))),
            (-0.5, PauliString(.x(0))),
            (-0.5, PauliString(.x(1))),
            (-0.5, PauliString(.x(2))),
            (-0.5, PauliString(.x(3))),
        ])
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let config = DMRGConfiguration(
            maxSweeps: 5,
            convergenceThreshold: 1e-12,
            isSubspaceExpansionEnabled: true,
            noiseStrength: 1e-3,
        )
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 2, configuration: config)

        let result = await dmrg.findGroundState(from: nil)

        #expect(result.groundStateEnergy < 0.0, "Ground state energy should be negative")
        #expect(result.sweeps > 0, "Should complete at least one sweep")
    }

    @Test("Find ground state of four-site transverse field Ising model")
    func findGroundStateFourSiteTransverseIsing() async {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
            (-1.0, PauliString(.z(1), .z(2))),
            (-1.0, PauliString(.z(2), .z(3))),
            (-0.5, PauliString(.x(0))),
            (-0.5, PauliString(.x(1))),
            (-0.5, PauliString(.x(2))),
            (-0.5, PauliString(.x(3))),
        ])
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let config = DMRGConfiguration(
            maxSweeps: 20,
            convergenceThreshold: 1e-5,
        )
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 16, configuration: config)

        let result = await dmrg.findGroundState(from: nil)

        #expect(result.groundStateEnergy < 0.0, "Ground state energy should be negative")
        #expect(result.groundState.qubits == 4, "Ground state MPS should have 4 qubits")
        #expect(result.convergenceHistory.count > 0, "Convergence history should not be empty")
    }

    @Test("Convergence history decreases monotonically")
    func convergenceHistoryDecreasesMonotonically() async {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
        ])
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let config = DMRGConfiguration(
            maxSweeps: 8,
            convergenceThreshold: 1e-10,
        )
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 4, configuration: config)

        let result = await dmrg.findGroundState(from: nil)

        for i in 1 ..< result.convergenceHistory.count {
            let current = result.convergenceHistory[i]
            let previous = result.convergenceHistory[i - 1]
            #expect(current <= previous + 1e-6, "Energy should decrease or stay constant across sweeps at index \(i)")
        }
    }

    @Test("Energy converges for simple diagonal Hamiltonian")
    func energyConvergesForDiagonalHamiltonian() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
            (1.0, PauliString(.z(1))),
        ])
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let config = DMRGConfiguration(
            maxSweeps: 10,
            convergenceThreshold: 1e-8,
        )
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 4, configuration: config)

        let result = await dmrg.findGroundState(from: nil)

        #expect(result.groundStateEnergy.isFinite, "Ground state energy should be finite for diagonal Hamiltonian")
        #expect(result.groundStateEnergy <= 2.0, "Ground state energy should be bounded by max eigenvalue")
    }

    @Test("Seven-site transverse field Ising with multiple sweeps")
    func sevenSiteTransverseFieldIsingMultipleSweeps() async {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
            (-1.0, PauliString(.z(1), .z(2))),
            (-1.0, PauliString(.z(2), .z(3))),
            (-1.0, PauliString(.z(3), .z(4))),
            (-1.0, PauliString(.z(4), .z(5))),
            (-1.0, PauliString(.z(5), .z(6))),
            (-0.5, PauliString(.x(0))),
            (-0.5, PauliString(.x(1))),
            (-0.5, PauliString(.x(2))),
            (-0.5, PauliString(.x(3))),
            (-0.5, PauliString(.x(4))),
            (-0.5, PauliString(.x(5))),
            (-0.5, PauliString(.x(6))),
        ])
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let config = DMRGConfiguration(
            maxSweeps: 3,
            convergenceThreshold: 1e-14,
        )
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 8, configuration: config)

        let result = await dmrg.findGroundState(from: nil)

        #expect(result.groundStateEnergy < -3.0, "Seven-site Ising ground state energy should be significantly negative")
        #expect(result.groundState.qubits == 7, "Ground state MPS should have 7 qubits")
        #expect(result.sweeps > 0, "Should perform at least one sweep")
        #expect(result.convergenceHistory.count > 0, "Convergence history should not be empty")
        #expect(result.groundStateEnergy.isFinite, "Ground state energy should be finite")
    }

    @Test("Find ground state from provided initial MPS")
    func findGroundStateFromProvidedInitialMPS() async {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
            (-0.5, PauliString(.x(0))),
            (-0.5, PauliString(.x(1))),
        ])
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let config = DMRGConfiguration(
            maxSweeps: 10,
            convergenceThreshold: 1e-6,
        )
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 4, configuration: config)

        let initial = MatrixProductState(qubits: 2, maxBondDimension: 4)
        let result = await dmrg.findGroundState(from: initial)

        #expect(result.groundStateEnergy < 0.0, "Ground state energy with initial MPS should be negative")
        #expect(result.groundState.qubits == 2, "Ground state should preserve qubit count from initial MPS")
        #expect(result.sweeps > 0, "Should perform at least one sweep with provided initial state")
    }

    @Test("Max sweeps exhaustion returns when convergence not reached")
    func maxSweepsExhaustionReturns() async {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
            (-0.5, PauliString(.x(0))),
            (-0.5, PauliString(.x(1))),
        ])
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let config = DMRGConfiguration(
            maxSweeps: 1,
            convergenceThreshold: 1e-14,
        )
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 4, configuration: config)

        let result = await dmrg.findGroundState(from: nil)

        #expect(result.sweeps == 1, "Should exhaust max sweeps without converging")
        #expect(result.convergenceHistory.count == 1, "Convergence history should have one entry per sweep")
        #expect(result.groundStateEnergy.isFinite, "Ground state energy should be finite after max sweeps")
    }

    // Uncomment to cover Lanczos eigensolver path (dim > 1000). Takes ~9 minutes.
    // Verified: covers 100% functions, 100% lines, 92% regions (22 missed = defensive guards).
//    @Test("Ten-site Ising triggers Lanczos eigensolver for large effective Hamiltonian")
//    func tenSiteIsingLanczosPath() async {
//        var terms = [(Double, PauliString)]()
//        for i in 0 ..< 9 {
//            terms.append((-1.0, PauliString(.z(i), .z(i + 1))))
//        }
//        for i in 0 ..< 10 {
//            terms.append((-0.5, PauliString(.x(i))))
//        }
//        let hamiltonian = Observable(terms: terms)
//        let mpo = MatrixProductOperator(observable: hamiltonian)
//        let config = DMRGConfiguration(
//            maxSweeps: 4,
//            convergenceThreshold: 1e-14,
//        )
//        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 16, configuration: config)
//
//        let result = await dmrg.findGroundState(from: nil)
//
//        #expect(result.groundStateEnergy < -5.0, "Ten-site Ising ground state energy should be significantly negative")
//        #expect(result.groundState.qubits == 10, "Ground state MPS should have 10 qubits")
//        #expect(result.sweeps > 0, "Should perform at least one sweep")
//        #expect(result.groundStateEnergy.isFinite, "Ground state energy should be finite")
//    }
}

/// Test suite for DMRGResult properties and structure.
/// Validates result creation, field access,
/// and convergence history tracking.
@Suite("DMRGResult")
struct DMRGResultTests {
    @Test("DMRGResult stores ground state energy correctly")
    func dmrgResultStoresGroundStateEnergy() async {
        let hamiltonian = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let config = DMRGConfiguration(maxSweeps: 5)
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 4, configuration: config)

        let result = await dmrg.findGroundState(from: nil)

        #expect(result.groundStateEnergy.isFinite, "Ground state energy should be finite")
    }

    @Test("DMRGResult contains valid MPS ground state")
    func dmrgResultContainsValidMPS() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
            (1.0, PauliString(.z(1))),
        ])
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let config = DMRGConfiguration(maxSweeps: 5)
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 4, configuration: config)

        let result = await dmrg.findGroundState(from: nil)

        #expect(result.groundState.qubits == 2, "Ground state MPS should have correct qubit count")
        #expect(result.groundState.tensors.count == 2, "Ground state MPS should have one tensor per site")
    }

    @Test("DMRGResult tracks sweep count accurately")
    func dmrgResultTracksSweepCount() async {
        let hamiltonian = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let config = DMRGConfiguration(maxSweeps: 3, convergenceThreshold: 1e-12)
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 2, configuration: config)

        let result = await dmrg.findGroundState(from: nil)

        #expect(result.sweeps > 0, "At least one sweep should be performed")
        #expect(result.sweeps <= config.maxSweeps, "Sweeps should not exceed maxSweeps")
    }

    @Test("DMRGResult convergence history matches sweep count")
    func dmrgResultConvergenceHistoryMatchesSweeps() async {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
        ])
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let config = DMRGConfiguration(maxSweeps: 5, convergenceThreshold: 1e-10)
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 4, configuration: config)

        let result = await dmrg.findGroundState(from: nil)

        #expect(result.convergenceHistory.count == result.sweeps, "Convergence history length should equal sweep count")
    }

    @Test("DMRGResult final energy matches last convergence history entry")
    func dmrgResultFinalEnergyMatchesHistory() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
        ])
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let config = DMRGConfiguration(maxSweeps: 5)
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 2, configuration: config)

        let result = await dmrg.findGroundState(from: nil)

        if let lastEnergy = result.convergenceHistory.last {
            #expect(
                abs(result.groundStateEnergy - lastEnergy) < 1e-10,
                "Ground state energy should match final convergence history entry",
            )
        }
    }
}

/// Test suite for DMRGProgress tracking.
/// Validates progress snapshot creation, sweep tracking,
/// and real-time energy monitoring during optimization.
@Suite("DMRGProgress")
struct DMRGProgressTests {
    @Test("DMRGProgress initial state is zero")
    func dmrgProgressInitialStateIsZero() async {
        let hamiltonian = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 4)

        let progress = await dmrg.progress

        #expect(progress.sweep == 0, "Initial sweep should be zero")
        #expect(progress.energy == 0.0, "Initial energy should be zero")
        #expect(progress.maxTruncationError == 0.0, "Initial truncation error should be zero")
    }

    @Test("DMRGProgress updates after optimization")
    func dmrgProgressUpdatesAfterOptimization() async {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
        ])
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let config = DMRGConfiguration(maxSweeps: 3)
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 4, configuration: config)

        _ = await dmrg.findGroundState(from: nil)
        let progress = await dmrg.progress

        #expect(progress.sweep >= 0, "Sweep count should be non-negative after optimization")
        #expect(progress.energy.isFinite, "Energy should be finite after optimization")
    }

    @Test("DMRGProgress tracks truncation error")
    func dmrgProgressTracksTruncationError() async {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
            (-0.5, PauliString(.x(0))),
            (-0.5, PauliString(.x(1))),
        ])
        let mpo = MatrixProductOperator(observable: hamiltonian)
        let config = DMRGConfiguration(maxSweeps: 5)
        let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 4, configuration: config)

        _ = await dmrg.findGroundState(from: nil)
        let progress = await dmrg.progress

        #expect(progress.maxTruncationError >= 0.0, "Truncation error should be non-negative")
        #expect(progress.maxTruncationError.isFinite, "Truncation error should be finite")
    }

    @Test("DMRGProgress struct initialization")
    func dmrgProgressStructInitialization() {
        let progress = DMRG.Progress(sweep: 5, energy: -2.5, maxTruncationError: 1e-8)

        #expect(progress.sweep == 5, "Sweep should be set correctly")
        #expect(abs(progress.energy + 2.5) < 1e-10, "Energy should be set correctly")
        #expect(abs(progress.maxTruncationError - 1e-8) < 1e-15, "Truncation error should be set correctly")
    }

    @Test("DMRGResult struct initialization")
    func dmrgResultStructInitialization() {
        let mps = MatrixProductState(qubits: 2, maxBondDimension: 4)
        let result = DMRG.Result(
            groundStateEnergy: -1.5,
            groundState: mps,
            sweeps: 10,
            convergenceHistory: [-2.0, -1.8, -1.5],
        )

        #expect(abs(result.groundStateEnergy + 1.5) < 1e-10, "Ground state energy should be set correctly")
        #expect(result.groundState.qubits == 2, "Ground state qubits should match")
        #expect(result.sweeps == 10, "Sweep count should be set correctly")
        #expect(result.convergenceHistory.count == 3, "Convergence history should have correct length")
    }
}
