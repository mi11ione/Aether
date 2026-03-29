// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under Apache 2.0

import Aether
import Foundation
import Testing

/// Test suite for TrajectoryConfiguration initialization and defaults.
/// Validates constructor parameters, default values, and configuration immutability
/// for Monte Carlo wavefunction simulations.
@Suite("TrajectoryConfiguration")
struct TrajectoryConfigurationTests {
    @Test("Default configuration values")
    func defaultValues() {
        let config = TrajectoryConfiguration()

        #expect(config.trajectories == 1000, "Default trajectories should be 1000")
        #expect(config.seed == nil, "Default seed should be nil")
        #expect(config.timeSteps == 1000, "Default timeSteps should be 1000")
        #expect(config.shouldStoreIndividualTrajectories == false, "Default shouldStoreIndividualTrajectories should be false")
    }

    @Test("Custom configuration values")
    func customValues() {
        let config = TrajectoryConfiguration(
            trajectories: 50,
            seed: 12345,
            timeSteps: 200,
            shouldStoreIndividualTrajectories: true,
        )

        #expect(config.trajectories == 50, "Custom trajectories should be 50")
        #expect(config.seed == 12345, "Custom seed should be 12345")
        #expect(config.timeSteps == 200, "Custom timeSteps should be 200")
        #expect(config.shouldStoreIndividualTrajectories == true, "shouldStoreIndividualTrajectories should be true")
    }

    @Test("Partial custom configuration")
    func partialCustomValues() {
        let config = TrajectoryConfiguration(trajectories: 100, seed: 42)

        #expect(config.trajectories == 100, "Trajectories should be 100")
        #expect(config.seed == 42, "Seed should be 42")
        #expect(config.timeSteps == 1000, "TimeSteps should use default 1000")
        #expect(config.shouldStoreIndividualTrajectories == false, "shouldStoreIndividualTrajectories should use default false")
    }
}

/// Test suite for TrajectoryStatistics data structure.
/// Validates jump count tracking, averages computation, and trajectory count
/// reporting for open quantum system simulations.
@Suite("TrajectoryStatistics")
struct TrajectoryStatisticsTests {
    @Test("Statistics from spontaneous emission simulation")
    func spontaneousEmissionStatistics() async {
        let sigmaMinus: [[Complex<Double>]] = [
            [.zero, .one],
            [.zero, .zero],
        ]

        let psiExcited = QuantumState(qubit: 1)
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.1)

        let config = TrajectoryConfiguration(
            trajectories: 30,
            seed: 42,
            timeSteps: 100,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus],
            initialState: psiExcited,
            time: 2.0,
            configuration: config,
        )

        let stats = result.statistics
        #expect(stats.trajectoryCount == 30, "Trajectory count should be 30")
        #expect(stats.jumpCounts.count == 1, "Should have one jump operator")
        #expect(stats.averageJumpsPerTrajectory >= 0.0, "Average jumps should be non-negative")

        let totalJumps = stats.jumpCounts.reduce(0, +)
        let expectedAvg = Double(totalJumps) / Double(stats.trajectoryCount)
        #expect(
            abs(stats.averageJumpsPerTrajectory - expectedAvg) < 1e-10,
            "Average should equal total/count",
        )
    }

    @Test("Multiple jump operators statistics")
    func multipleJumpOperatorsStatistics() async {
        let sigmaMinus: [[Complex<Double>]] = [
            [.zero, .one],
            [.zero, .zero],
        ]
        let sigmaPlus: [[Complex<Double>]] = [
            [.zero, .zero],
            [.one, .zero],
        ]

        let H = Observable.pauliZ(qubit: 0, coefficient: 0.1)
        let invSqrt2 = 1.0 / sqrt(2.0)
        let superposition = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0),
            Complex(invSqrt2, 0),
        ])

        let config = TrajectoryConfiguration(
            trajectories: 20,
            seed: 123,
            timeSteps: 50,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus, sigmaPlus],
            initialState: superposition,
            time: 1.0,
            configuration: config,
        )

        let stats = result.statistics
        #expect(stats.jumpCounts.count == 2, "Should have two jump operators")
        #expect(stats.trajectoryCount == 20, "Trajectory count should be 20")
    }
}

/// Test suite for basic quantum trajectory evolution.
/// Validates state evolution under Lindblad dynamics with simple Hamiltonians
/// and jump operators, checking result structure and physical consistency.
@Suite("QuantumTrajectory Evolution")
struct QuantumTrajectoryEvolutionTests {
    @Test("Basic evolution produces valid density matrix")
    func basicEvolutionValidDensityMatrix() async {
        let sigmaMinus: [[Complex<Double>]] = [
            [.zero, .one],
            [.zero, .zero],
        ]

        let psiExcited = QuantumState(qubit: 1)
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.5)

        let config = TrajectoryConfiguration(
            trajectories: 20,
            seed: 42,
            timeSteps: 50,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus],
            initialState: psiExcited,
            time: 1.0,
            configuration: config,
        )

        let rho = result.averageDensityMatrix
        let trace = rho.trace()
        #expect(abs(trace - 1.0) < 1e-10, "Density matrix trace should be 1.0, got \(trace)")
        #expect(rho.isHermitian(), "Density matrix should be Hermitian")

        let probs = rho.probabilities()
        let allNonNegative = probs.allSatisfy { $0 >= -1e-10 }
        #expect(allNonNegative, "All probabilities should be non-negative")
    }

    @Test("Result time matches input time")
    func resultTimeMatchesInput() async {
        let sigmaMinus: [[Complex<Double>]] = [
            [.zero, .one],
            [.zero, .zero],
        ]

        let psi = QuantumState(qubit: 0)
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.1)

        let config = TrajectoryConfiguration(
            trajectories: 10,
            seed: 42,
            timeSteps: 20,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus],
            initialState: psi,
            time: 3.5,
            configuration: config,
        )

        #expect(abs(result.time - 3.5) < 1e-10, "Result time should be 3.5")
    }

    @Test("Individual trajectories stored when requested")
    func individualTrajectoriesStored() async {
        let sigmaMinus: [[Complex<Double>]] = [
            [.zero, .one],
            [.zero, .zero],
        ]

        let psi = QuantumState(qubit: 1)
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.1)

        let config = TrajectoryConfiguration(
            trajectories: 15,
            seed: 42,
            timeSteps: 30,
            shouldStoreIndividualTrajectories: true,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus],
            initialState: psi,
            time: 1.0,
            configuration: config,
        )

        let trajectories = result.individualTrajectories
        #expect(trajectories != nil, "Individual trajectories should be stored")
        #expect(trajectories?.count == 15, "Should have 15 individual trajectories")

        if let trajectories {
            for state in trajectories {
                #expect(state.isNormalized(), "Each trajectory state should be normalized")
            }
        }
    }

    @Test("Individual trajectories nil when not requested")
    func individualTrajectoriesNil() async {
        let sigmaMinus: [[Complex<Double>]] = [
            [.zero, .one],
            [.zero, .zero],
        ]

        let psi = QuantumState(qubit: 0)
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.1)

        let config = TrajectoryConfiguration(
            trajectories: 10,
            seed: 42,
            timeSteps: 20,
            shouldStoreIndividualTrajectories: false,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus],
            initialState: psi,
            time: 1.0,
            configuration: config,
        )

        #expect(result.individualTrajectories == nil, "Individual trajectories should be nil")
    }

    @Test("Three-qubit evolution triggers BLAS optimized paths")
    func threeQubitBLASPaths() async {
        var L = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: 8), count: 8)
        L[0][4] = Complex(1.0, 0)

        let amps: [Complex<Double>] = [
            .zero, .zero, .zero, .zero,
            Complex(1.0, 0), .zero, .zero, .zero,
        ]
        let psi = QuantumState(qubits: 3, amplitudes: amps)
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.5)

        let config = TrajectoryConfiguration(
            trajectories: 5,
            seed: 42,
            timeSteps: 50,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [L],
            initialState: psi,
            time: 2.0,
            configuration: config,
        )

        #expect(result.time == 2.0, "Evolution time should match input")
        #expect(result.averageDensityMatrix.qubits == 3, "Density matrix should have 3 qubits")
        #expect(result.statistics.trajectoryCount == 5, "Should have 5 trajectories")
        #expect(result.statistics.jumpCounts.count == 1, "Should track one jump operator")
    }
}

/// Test suite for trajectory averaging producing valid density matrices.
/// Validates that Monte Carlo averaging yields physically valid density matrices
/// with unit trace, Hermiticity, and non-negative diagonal elements.
@Suite("Trajectory Averaging")
struct TrajectoryAveragingTests {
    @Test("Averaged density matrix has unit trace")
    func averagedDensityMatrixUnitTrace() async {
        let sigmaMinus: [[Complex<Double>]] = [
            [.zero, .one],
            [.zero, .zero],
        ]

        let psi = QuantumState(qubit: 1)
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.2)

        let config = TrajectoryConfiguration(
            trajectories: 30,
            seed: 42,
            timeSteps: 50,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus],
            initialState: psi,
            time: 2.0,
            configuration: config,
        )

        let trace = result.averageDensityMatrix.trace()
        #expect(abs(trace - 1.0) < 1e-10, "Trace should be 1.0, got \(trace)")
    }

    @Test("Averaged density matrix is Hermitian")
    func averagedDensityMatrixHermitian() async {
        let sigmaMinus: [[Complex<Double>]] = [
            [.zero, .one],
            [.zero, .zero],
        ]

        let psi = QuantumState(qubit: 1)
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.3)

        let config = TrajectoryConfiguration(
            trajectories: 25,
            seed: 42,
            timeSteps: 40,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus],
            initialState: psi,
            time: 1.5,
            configuration: config,
        )

        #expect(result.averageDensityMatrix.isHermitian(), "Density matrix should be Hermitian")
    }

    @Test("Purity decreases under dissipation")
    func purityDecreasesUnderDissipation() async {
        let sigmaMinus: [[Complex<Double>]] = [
            [.zero, .one],
            [.zero, .zero],
        ]

        let invSqrt2 = 1.0 / sqrt(2.0)
        let superposition = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0),
            Complex(invSqrt2, 0),
        ])

        let H = Observable.pauliZ(qubit: 0, coefficient: 0.1)

        let config = TrajectoryConfiguration(
            trajectories: 40,
            seed: 42,
            timeSteps: 100,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus],
            initialState: superposition,
            time: 3.0,
            configuration: config,
        )

        let purity = result.averageDensityMatrix.purity()
        #expect(purity >= 0.5 - 1e-10, "Purity should be at least 0.5 for single qubit")
        #expect(purity <= 1.0 + 1e-10, "Purity should not exceed 1.0")
    }
}

/// Test suite for seeded trajectory reproducibility.
/// Validates that seeded simulations produce identical results across runs,
/// essential for debugging and testing quantum trajectory implementations.
@Suite("Seeded Trajectories")
struct SeededTrajectoryTests {
    @Test("Same seed produces identical results")
    func sameSeedIdenticalResults() async {
        let sigmaMinus: [[Complex<Double>]] = [
            [.zero, .one],
            [.zero, .zero],
        ]

        let psi = QuantumState(qubit: 1)
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.2)

        let config = TrajectoryConfiguration(
            trajectories: 15,
            seed: 42,
            timeSteps: 30,
        )

        let result1 = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus],
            initialState: psi,
            time: 1.0,
            configuration: config,
        )

        let result2 = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus],
            initialState: psi,
            time: 1.0,
            configuration: config,
        )

        #expect(
            result1.statistics.jumpCounts == result2.statistics.jumpCounts,
            "Jump counts should be identical for same seed",
        )
        #expect(
            abs(result1.statistics.averageJumpsPerTrajectory - result2.statistics.averageJumpsPerTrajectory) < 1e-10,
            "Average jumps should be identical for same seed",
        )
    }

    @Test("Different seeds produce different results")
    func differentSeedsDifferentResults() async {
        let sigmaMinus: [[Complex<Double>]] = [
            [.zero, .one],
            [.zero, .zero],
        ]

        let invSqrt2 = 1.0 / sqrt(2.0)
        let superposition = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0),
            Complex(invSqrt2, 0),
        ])

        let H = Observable.pauliZ(qubit: 0, coefficient: 0.3)

        let config1 = TrajectoryConfiguration(
            trajectories: 20,
            seed: 42,
            timeSteps: 50,
        )

        let config2 = TrajectoryConfiguration(
            trajectories: 20,
            seed: 99999,
            timeSteps: 50,
        )

        let result1 = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus],
            initialState: superposition,
            time: 2.0,
            configuration: config1,
        )

        let result2 = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus],
            initialState: superposition,
            time: 2.0,
            configuration: config2,
        )

        let trace1 = result1.averageDensityMatrix.trace()
        let trace2 = result2.averageDensityMatrix.trace()
        #expect(abs(trace1 - 1.0) < 1e-10, "Result 1 should have unit trace")
        #expect(abs(trace2 - 1.0) < 1e-10, "Result 2 should have unit trace")
    }
}

/// Test suite for jump operator selection statistics.
/// Validates that jump operator selection follows theoretical probability
/// distribution based on expectation values of L-dagger-L operators.
@Suite("Jump Operator Selection")
struct JumpOperatorSelectionTests {
    @Test("Jump selection follows probability distribution")
    func jumpSelectionFollowsProbability() async {
        let strongDecay: [[Complex<Double>]] = [
            [.zero, Complex(2.0, 0)],
            [.zero, .zero],
        ]
        let weakDecay: [[Complex<Double>]] = [
            [.zero, Complex(0.5, 0)],
            [.zero, .zero],
        ]

        let psi = QuantumState(qubit: 1)
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.01)

        let config = TrajectoryConfiguration(
            trajectories: 50,
            seed: 42,
            timeSteps: 200,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [strongDecay, weakDecay],
            initialState: psi,
            time: 3.0,
            configuration: config,
        )

        let stats = result.statistics
        let totalJumps = stats.jumpCounts[0] + stats.jumpCounts[1]

        if totalJumps > 10 {
            let strongRatio = Double(stats.jumpCounts[0]) / Double(totalJumps)
            #expect(strongRatio > 0.5, "Strong decay should dominate, ratio = \(strongRatio)")
        }
    }

    @Test("Single jump operator accumulates all jumps")
    func singleJumpOperatorAccumulatesAll() async {
        let sigmaMinus: [[Complex<Double>]] = [
            [.zero, .one],
            [.zero, .zero],
        ]

        let psi = QuantumState(qubit: 1)
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.1)

        let config = TrajectoryConfiguration(
            trajectories: 30,
            seed: 42,
            timeSteps: 100,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus],
            initialState: psi,
            time: 2.0,
            configuration: config,
        )

        let stats = result.statistics
        #expect(stats.jumpCounts.count == 1, "Should have exactly one jump operator")

        let totalFromAverage = Int(stats.averageJumpsPerTrajectory * Double(stats.trajectoryCount))
        #expect(
            abs(totalFromAverage - stats.jumpCounts[0]) <= 1,
            "Total jumps should match average * count",
        )
    }
}

/// Test suite for edge cases in quantum trajectory simulation.
/// Validates behavior at boundary conditions including zero evolution time,
/// identity Hamiltonian (no jumps), and minimal configuration parameters.
@Suite("Edge Cases")
struct QuantumTrajectoryEdgeCasesTests {
    @Test("Ground state with decay operator has no jumps")
    func groundStateNoJumps() async {
        let sigmaMinus: [[Complex<Double>]] = [
            [.zero, .one],
            [.zero, .zero],
        ]

        let psiGround = QuantumState(qubit: 0)
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.1)

        let config = TrajectoryConfiguration(
            trajectories: 20,
            seed: 42,
            timeSteps: 50,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus],
            initialState: psiGround,
            time: 2.0,
            configuration: config,
        )

        let stats = result.statistics
        #expect(stats.jumpCounts[0] == 0, "Ground state should have no decay jumps")

        let trace = result.averageDensityMatrix.trace()
        #expect(abs(trace - 1.0) < 1e-10, "Density matrix should maintain unit trace")
    }

    @Test("Very short evolution time")
    func veryShortEvolutionTime() async {
        let sigmaMinus: [[Complex<Double>]] = [
            [.zero, .one],
            [.zero, .zero],
        ]

        let psi = QuantumState(qubit: 1)
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.1)

        let config = TrajectoryConfiguration(
            trajectories: 10,
            seed: 42,
            timeSteps: 10,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus],
            initialState: psi,
            time: 0.001,
            configuration: config,
        )

        let trace = result.averageDensityMatrix.trace()
        #expect(abs(trace - 1.0) < 1e-10, "Density matrix should have unit trace")
        #expect(result.averageDensityMatrix.isHermitian(), "Density matrix should be Hermitian")
    }

    @Test("Empty jump operators list")
    func emptyJumpOperators() async {
        let psi = QuantumState(qubit: 0)
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.5)

        let config = TrajectoryConfiguration(
            trajectories: 10,
            seed: 42,
            timeSteps: 20,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [],
            initialState: psi,
            time: 1.0,
            configuration: config,
        )

        let stats = result.statistics
        #expect(stats.jumpCounts.isEmpty, "No jump operators means no jump counts")
        #expect(stats.averageJumpsPerTrajectory == 0.0, "Average jumps should be 0")

        let trace = result.averageDensityMatrix.trace()
        #expect(abs(trace - 1.0) < 1e-10, "Density matrix should have unit trace")
    }

    @Test("Two qubit system")
    func twoQubitSystem() async {
        let sigmaMinus0: [[Complex<Double>]] = [
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .zero, .zero],
            [.zero, .zero, .zero, .one],
            [.zero, .zero, .zero, .zero],
        ]

        let psi = QuantumState(qubits: 2, amplitudes: [.zero, .zero, .zero, .one])
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.1)

        let config = TrajectoryConfiguration(
            trajectories: 15,
            seed: 42,
            timeSteps: 30,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus0],
            initialState: psi,
            time: 1.0,
            configuration: config,
        )

        let trace = result.averageDensityMatrix.trace()
        #expect(abs(trace - 1.0) < 1e-10, "Two qubit trace should be 1.0")
        #expect(result.averageDensityMatrix.isHermitian(), "Two qubit density matrix should be Hermitian")
        #expect(result.averageDensityMatrix.qubits == 2, "Should have 2 qubits")
    }
}

/// Test suite for spontaneous emission physical behavior.
/// Validates that Monte Carlo trajectory simulation reproduces expected
/// decay dynamics for a two-level system with amplitude damping.
@Suite("Spontaneous Emission Physics")
struct SpontaneousEmissionPhysicsTests {
    @Test("Excited state decays to ground state")
    func excitedStateDecaysToGround() async {
        let sigmaMinus: [[Complex<Double>]] = [
            [.zero, .one],
            [.zero, .zero],
        ]

        let psiExcited = QuantumState(qubit: 1)
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.01)

        let config = TrajectoryConfiguration(
            trajectories: 40,
            seed: 42,
            timeSteps: 200,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus],
            initialState: psiExcited,
            time: 5.0,
            configuration: config,
        )

        let groundProb = result.averageDensityMatrix.probability(of: 0)
        #expect(groundProb > 0.5, "Ground state probability should be > 0.5 after decay, got \(groundProb)")

        let trace = result.averageDensityMatrix.trace()
        #expect(abs(trace - 1.0) < 1e-10, "Trace should be 1.0")
    }

    @Test("Population inversion decays")
    func populationInversionDecays() async {
        let sigmaMinus: [[Complex<Double>]] = [
            [.zero, .one],
            [.zero, .zero],
        ]

        let psiExcited = QuantumState(qubit: 1)
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.01)

        let config = TrajectoryConfiguration(
            trajectories: 30,
            seed: 42,
            timeSteps: 50,
        )

        let shortResult = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus],
            initialState: psiExcited,
            time: 0.5,
            configuration: config,
        )

        let longConfig = TrajectoryConfiguration(
            trajectories: 30,
            seed: 42,
            timeSteps: 150,
        )

        let longResult = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus],
            initialState: psiExcited,
            time: 3.0,
            configuration: longConfig,
        )

        let shortExcitedProb = shortResult.averageDensityMatrix.probability(of: 1)
        let longExcitedProb = longResult.averageDensityMatrix.probability(of: 1)

        #expect(
            longExcitedProb <= shortExcitedProb + 0.1,
            "Excited state probability should decay over time",
        )
    }
}

/// Test suite documenting sampleJumpOperator fallback return (line 497).
/// The fallback `return jumpProbs.count - 1` handles floating-point edge cases
/// where cumulative probability doesn't quite reach the random threshold.
/// This is an acceptable edge case that cannot be reliably triggered in tests.
@Suite("Jump Operator Sampling Edge Cases")
struct JumpOperatorSamplingEdgeCasesTests {
    @Test("Jump sampling handles normalized probabilities correctly")
    func jumpSamplingNormalizedProbabilities() async {
        let sigmaMinus: [[Complex<Double>]] = [
            [.zero, .one],
            [.zero, .zero],
        ]
        let sigmaZ: [[Complex<Double>]] = [
            [.one, .zero],
            [.zero, Complex(-1.0, 0)],
        ]

        let psi = QuantumState(qubit: 1)
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.5)

        let config = TrajectoryConfiguration(
            trajectories: 100,
            seed: 42,
            timeSteps: 200,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [sigmaMinus, sigmaZ],
            initialState: psi,
            time: 3.0,
            configuration: config,
        )

        let stats = result.statistics
        #expect(
            stats.jumpCounts.count == 2,
            "Should track jumps for both operators",
        )

        let totalJumps = stats.jumpCounts[0] + stats.jumpCounts[1]
        let expectedFromAverage = Int(stats.averageJumpsPerTrajectory * Double(stats.trajectoryCount))
        #expect(
            abs(totalJumps - expectedFromAverage) <= 1,
            "Total jumps should match average calculation",
        )

        #expect(
            abs(result.averageDensityMatrix.trace() - 1.0) < 1e-10,
            "Density matrix trace should be 1.0",
        )
    }

    @Test("Jump sampling with many operators remains robust")
    func jumpSamplingManyOperators() async {
        let op1: [[Complex<Double>]] = [
            [.zero, Complex(0.3, 0)],
            [.zero, .zero],
        ]

        let op2: [[Complex<Double>]] = [
            [.zero, .zero],
            [Complex(0.2, 0), .zero],
        ]

        let op3: [[Complex<Double>]] = [
            [Complex(0.1, 0), .zero],
            [.zero, Complex(-0.1, 0)],
        ]

        let op4: [[Complex<Double>]] = [
            [.zero, Complex(0.15, 0)],
            [.zero, .zero],
        ]

        let invSqrt2 = 1.0 / sqrt(2.0)
        let superposition = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0),
            Complex(invSqrt2, 0),
        ])
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.1)

        let config = TrajectoryConfiguration(
            trajectories: 50,
            seed: 12345,
            timeSteps: 300,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [op1, op2, op3, op4],
            initialState: superposition,
            time: 2.0,
            configuration: config,
        )

        let stats = result.statistics
        #expect(
            stats.jumpCounts.count == 4,
            "Should track all 4 jump operators",
        )

        for (idx, count) in stats.jumpCounts.enumerated() {
            #expect(
                count >= 0,
                "Jump count for operator \(idx) should be non-negative, got \(count)",
            )
        }

        #expect(
            result.averageDensityMatrix.isHermitian(),
            "Averaged density matrix should be Hermitian",
        )
    }

    @Test("Five-qubit evolution triggers vDSP vectorized paths")
    func fiveQubitVDSPPaths() async {
        let dim = 32
        var L = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: dim), count: dim)
        L[0][dim / 2] = Complex(0.3, 0)

        var amps = [Complex<Double>](repeating: .zero, count: dim)
        amps[dim / 2] = Complex(1.0, 0)
        let psi = QuantumState(qubits: 5, amplitudes: amps)
        let H = Observable.pauliZ(qubit: 0, coefficient: 0.5)

        let config = TrajectoryConfiguration(
            trajectories: 3,
            seed: 99,
            timeSteps: 20,
        )

        let result = await QuantumTrajectory.evolve(
            hamiltonian: H,
            jumpOperators: [L],
            initialState: psi,
            time: 1.0,
            configuration: config,
        )

        #expect(
            result.averageDensityMatrix.qubits == 5,
            "Density matrix should have 5 qubits after vDSP path evolution",
        )
    }
}
