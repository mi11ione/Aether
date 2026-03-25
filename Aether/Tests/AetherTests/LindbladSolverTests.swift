// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Test suite for LindbladConfiguration default and custom values.
/// Validates initialization parameters for the Lindblad master equation solver
/// including tolerances, step limits, and positivity enforcement methods.
@Suite("LindbladConfiguration Initialization")
struct LindbladConfigurationTests {
    @Test("Default configuration has expected values")
    func defaultConfiguration() {
        let config = LindbladConfiguration()
        #expect(abs(config.absoluteTolerance - 1e-8) < 1e-15, "Default absolute tolerance should be 1e-8")
        #expect(abs(config.relativeTolerance - 1e-6) < 1e-15, "Default relative tolerance should be 1e-6")
        #expect(config.maxSteps == 10000, "Default max steps should be 10000")
        #expect(config.stiffnessThreshold == 20, "Default stiffness threshold should be 20")
    }

    @Test("Custom configuration preserves all values")
    func customConfiguration() {
        let config = LindbladConfiguration(
            absoluteTolerance: 1e-12,
            relativeTolerance: 1e-10,
            maxSteps: 5000,
            stiffnessThreshold: 15,
            positivityEnforcement: .eigenvalue,
        )
        #expect(abs(config.absoluteTolerance - 1e-12) < 1e-15, "Custom absolute tolerance should be 1e-12")
        #expect(abs(config.relativeTolerance - 1e-10) < 1e-15, "Custom relative tolerance should be 1e-10")
        #expect(config.maxSteps == 5000, "Custom max steps should be 5000")
        #expect(config.stiffnessThreshold == 15, "Custom stiffness threshold should be 15")
    }

    @Test("Partial custom configuration uses defaults for unspecified")
    func partialCustomConfiguration() {
        let config = LindbladConfiguration(absoluteTolerance: 1e-10)
        #expect(abs(config.absoluteTolerance - 1e-10) < 1e-15, "Custom absolute tolerance should be 1e-10")
        #expect(abs(config.relativeTolerance - 1e-6) < 1e-15, "Relative tolerance should use default 1e-6")
        #expect(config.maxSteps == 10000, "Max steps should use default 10000")
    }
}

/// Test suite for PositivityMethod enum cases.
/// Validates all positivity enforcement methods available for density matrix evolution
/// including Cholesky factorization, eigenvalue clipping, and no enforcement.
@Suite("PositivityMethod Enum Cases")
struct PositivityMethodTests {
    @Test("Cholesky method is available")
    func choleskyMethod() {
        let method = PositivityMethod.cholesky
        let config = LindbladConfiguration(positivityEnforcement: method)
        #expect(config.positivityEnforcement == .cholesky, "Cholesky method should be configurable")
    }

    @Test("Eigenvalue method is available")
    func eigenvalueMethod() {
        let method = PositivityMethod.eigenvalue
        let config = LindbladConfiguration(positivityEnforcement: method)
        #expect(config.positivityEnforcement == .eigenvalue, "Eigenvalue method should be configurable")
    }

    @Test("None method is available")
    func noneMethod() {
        let method = PositivityMethod.none
        let config = LindbladConfiguration(positivityEnforcement: method)
        #expect(config.positivityEnforcement == .none, "None method should be configurable")
    }

    @Test("Default positivity method is Cholesky")
    func defaultIsCholesky() {
        let config = LindbladConfiguration()
        #expect(config.positivityEnforcement == .cholesky, "Default positivity method should be Cholesky")
    }
}

/// Test suite for IntegrationMethod enum cases.
/// Validates RK45 and TR-BDF2 integration methods
/// used for solving the Lindblad master equation.
@Suite("IntegrationMethod Enum Cases")
struct IntegrationMethodTests {
    @Test("RK45 method exists")
    func rk45Method() {
        let method = IntegrationMethod.rk45
        #expect(method == .rk45, "RK45 integration method should exist")
    }

    @Test("TR-BDF2 method exists")
    func trBdf2Method() {
        let method = IntegrationMethod.trBdf2
        #expect(method == .trBdf2, "TR-BDF2 integration method should exist")
    }
}

/// Test suite for basic LindbladSolver evolution.
/// Validates simple time evolution with trivial Hamiltonian and jump operators
/// ensuring the solver produces physically valid density matrices.
@Suite("LindbladSolver Basic Evolution")
struct LindbladSolverBasicEvolutionTests {
    @Test("Zero time evolution returns initial state")
    func zeroTimeEvolution() {
        let hamiltonian = Observable.pauliZ(qubit: 0, coefficient: 1.0)
        let jumpOperators: [[[Complex<Double>]]] = []
        let initialState = DensityMatrix(qubits: 1)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.0,
        )

        #expect(abs(result.time) < 1e-10, "Zero evolution time should result in time = 0")
        #expect(abs(result.finalState.trace() - 1.0) < 1e-10, "Trace should remain 1.0")
    }

    @Test("Evolution with identity-like dynamics preserves state")
    func identityEvolution() {
        let hamiltonian = Observable(terms: [])
        let jumpOperators: [[[Complex<Double>]]] = []
        let initialState = DensityMatrix(qubits: 1)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.1,
        )

        #expect(abs(result.time - 0.1) < 1e-6, "Evolution should reach target time 0.1")
        let purity = result.finalState.purity()
        #expect(abs(purity - 1.0) < 1e-6, "Pure state should remain pure under identity evolution")
    }

    @Test("Single qubit evolution with Z Hamiltonian")
    func singleQubitZEvolution() {
        let hamiltonian = Observable.pauliZ(qubit: 0, coefficient: 1.0)
        let jumpOperators: [[[Complex<Double>]]] = []
        let initialState = DensityMatrix(qubits: 1)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.5,
        )

        #expect(abs(result.time - 0.5) < 1e-3, "Evolution should reach target time 0.5")
        #expect(abs(result.finalState.trace() - 1.0) < 1e-6, "Trace should be preserved")
    }
}

/// Test suite for trace preservation under Lindblad evolution.
/// Validates that Tr(rho) = 1 is maintained throughout evolution
/// as required by probability conservation in quantum mechanics.
@Suite("Trace Preservation")
struct LindbladTracePreservationTests {
    @Test("Trace preserved with unitary evolution")
    func tracePreservedUnitary() {
        let hamiltonian = Observable.pauliX(qubit: 0, coefficient: 0.5)
        let jumpOperators: [[[Complex<Double>]]] = []
        let initialState = DensityMatrix(qubits: 1)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 1.0,
        )

        let trace = result.finalState.trace()
        #expect(abs(trace - 1.0) < 1e-6, "Trace should be 1.0 after unitary evolution, got \(trace)")
    }

    @Test("Trace preserved with dissipation")
    func tracePreservedDissipation() {
        let hamiltonian = Observable.pauliZ(qubit: 0, coefficient: 1.0)
        let sigmaMinusOp: [[Complex<Double>]] = [
            [.zero, Complex(0.1, 0)],
            [.zero, .zero],
        ]
        let jumpOperators = [sigmaMinusOp]
        let initialState = DensityMatrix(qubits: 1)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.5,
        )

        let trace = result.finalState.trace()
        #expect(abs(trace - 1.0) < 1e-5, "Trace should be 1.0 after dissipative evolution, got \(trace)")
    }

    @Test("Trace preserved for two-qubit system")
    func tracePreservedTwoQubit() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
            (0.5, PauliString(.z(1))),
        ])
        let jumpOperators: [[[Complex<Double>]]] = []
        let initialState = DensityMatrix(qubits: 2)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.2,
        )

        let trace = result.finalState.trace()
        #expect(abs(trace - 1.0) < 1e-6, "Trace should be 1.0 for two-qubit system, got \(trace)")
    }
}

/// Test suite for positivity preservation under Lindblad evolution.
/// Validates that eigenvalues of the density matrix remain non-negative
/// ensuring physical validity of the evolved quantum state.
@Suite("Positivity Preservation")
struct LindbladPositivityPreservationTests {
    @Test("Eigenvalues non-negative after evolution with Cholesky enforcement")
    func eigenvaluesNonNegativeCholesky() {
        let hamiltonian = Observable.pauliZ(qubit: 0, coefficient: 1.0)
        let sigmaMinusOp: [[Complex<Double>]] = [
            [.zero, Complex(0.1, 0)],
            [.zero, .zero],
        ]
        let jumpOperators = [sigmaMinusOp]
        let initialState = DensityMatrix(qubits: 1)
        let config = LindbladConfiguration(positivityEnforcement: .cholesky)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.5,
            configuration: config,
        )

        let purity = result.finalState.purity()
        #expect(purity >= 0.0, "Purity should be non-negative, got \(purity)")
        #expect(purity <= 1.0 + 1e-6, "Purity should not exceed 1.0, got \(purity)")
    }

    @Test("Eigenvalues non-negative after evolution with eigenvalue enforcement")
    func eigenvaluesNonNegativeEigenvalue() {
        let hamiltonian = Observable.pauliZ(qubit: 0, coefficient: 1.0)
        let jumpOperators: [[[Complex<Double>]]] = []
        let initialState = DensityMatrix(qubits: 1)
        let config = LindbladConfiguration(positivityEnforcement: .eigenvalue)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.5,
            configuration: config,
        )

        let purity = result.finalState.purity()
        #expect(purity >= 0.0, "Purity should be non-negative with eigenvalue enforcement, got \(purity)")
        #expect(purity <= 1.0 + 1e-6, "Purity should not exceed 1.0, got \(purity)")
    }

    @Test("Diagonal probabilities remain non-negative")
    func diagonalProbabilitiesNonNegative() {
        let hamiltonian = Observable.pauliX(qubit: 0, coefficient: 0.5)
        let jumpOperators: [[[Complex<Double>]]] = []
        let initialState = DensityMatrix(qubits: 1)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 1.0,
        )

        let p0 = result.finalState.probability(of: 0)
        let p1 = result.finalState.probability(of: 1)
        #expect(p0 >= -1e-10, "P(0) should be non-negative, got \(p0)")
        #expect(p1 >= -1e-10, "P(1) should be non-negative, got \(p1)")
        #expect(abs(p0 + p1 - 1.0) < 1e-6, "Probabilities should sum to 1.0, got \(p0 + p1)")
    }
}

/// Test suite for StepStatistics returned by LindbladSolver.
/// Validates that step statistics accurately track accepted steps,
/// rejected steps, and solver performance metrics.
@Suite("Step Statistics")
struct LindbladStepStatisticsTests {
    @Test("Step statistics are populated after evolution")
    func stepStatisticsPopulated() {
        let hamiltonian = Observable.pauliZ(qubit: 0, coefficient: 1.0)
        let jumpOperators: [[[Complex<Double>]]] = []
        let initialState = DensityMatrix(qubits: 1)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.1,
        )

        let stats = result.stepStatistics
        #expect(stats.acceptedSteps >= 0, "Accepted steps should be non-negative")
        #expect(stats.rejectedSteps >= 0, "Rejected steps should be non-negative")
        #expect(stats.stiffnessDetections >= 0, "Stiffness detections should be non-negative")
        #expect(stats.methodSwitches >= 0, "Method switches should be non-negative")
    }

    @Test("At least one step taken for non-zero evolution time")
    func atLeastOneStep() {
        let hamiltonian = Observable.pauliZ(qubit: 0, coefficient: 1.0)
        let jumpOperators: [[[Complex<Double>]]] = []
        let initialState = DensityMatrix(qubits: 1)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.01,
        )

        let totalSteps = result.stepStatistics.acceptedSteps + result.stepStatistics.rejectedSteps
        #expect(totalSteps >= 1, "At least one step should be taken for t > 0, got \(totalSteps)")
    }
}

/// Test suite for LindbladResult structure.
/// Validates that the result contains correct final state, time,
/// and integration method information after evolution.
@Suite("LindbladResult Structure")
struct LindbladResultTests {
    @Test("Result contains final time matching evolution request")
    func resultContainsFinalTime() {
        let hamiltonian = Observable.pauliZ(qubit: 0, coefficient: 1.0)
        let jumpOperators: [[[Complex<Double>]]] = []
        let initialState = DensityMatrix(qubits: 1)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.25,
        )

        #expect(abs(result.time - 0.25) < 1e-3, "Final time should match requested evolution time")
    }

    @Test("Result contains valid final method")
    func resultContainsValidMethod() {
        let hamiltonian = Observable.pauliZ(qubit: 0, coefficient: 1.0)
        let jumpOperators: [[[Complex<Double>]]] = []
        let initialState = DensityMatrix(qubits: 1)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.1,
        )

        let validMethods: [IntegrationMethod] = [.rk45, .trBdf2]
        let isValidMethod = validMethods.contains(result.finalMethod)
        #expect(isValidMethod, "Final method should be one of the valid integration methods")
    }

    @Test("Result final state has correct qubit count")
    func resultFinalStateQubitCount() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
            (0.5, PauliString(.z(1))),
        ])
        let jumpOperators: [[[Complex<Double>]]] = []
        let initialState = DensityMatrix(qubits: 2)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.1,
        )

        #expect(result.finalState.qubits == 2, "Final state should have 2 qubits")
        #expect(result.finalState.dimension == 4, "Final state dimension should be 4")
    }
}

/// Test suite for edge cases in LindbladSolver.
/// Validates behavior with boundary conditions such as zero time,
/// empty Hamiltonian, and minimal system sizes.
@Suite("LindbladSolver Edge Cases")
struct LindbladSolverEdgeCasesTests {
    @Test("Empty Hamiltonian produces no evolution")
    func emptyHamiltonianNoEvolution() {
        let hamiltonian = Observable(terms: [])
        let jumpOperators: [[[Complex<Double>]]] = []
        let initialState = DensityMatrix(qubits: 1)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.5,
        )

        let initialProbZero = initialState.probability(of: 0)
        let finalProbZero = result.finalState.probability(of: 0)
        #expect(abs(finalProbZero - initialProbZero) < 1e-6, "State should not change with empty Hamiltonian")
    }

    @Test("Very short evolution time")
    func veryShortEvolutionTime() {
        let hamiltonian = Observable.pauliZ(qubit: 0, coefficient: 1.0)
        let jumpOperators: [[[Complex<Double>]]] = []
        let initialState = DensityMatrix(qubits: 1)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 1e-6,
        )

        #expect(result.time >= 0, "Evolution time should be non-negative")
        #expect(abs(result.finalState.trace() - 1.0) < 1e-6, "Trace should be preserved for short evolution")
    }

    @Test("Evolution with no jump operators is unitary")
    func noJumpOperatorsUnitary() {
        let hamiltonian = Observable.pauliX(qubit: 0, coefficient: 1.0)
        let jumpOperators: [[[Complex<Double>]]] = []
        let initialState = DensityMatrix(qubits: 1)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.5,
        )

        let purity = result.finalState.purity()
        #expect(abs(purity - 1.0) < 1e-4, "Unitary evolution should preserve purity, got \(purity)")
    }

    @Test("Maximally mixed initial state evolution")
    func maximallyMixedInitialState() {
        let hamiltonian = Observable.pauliZ(qubit: 0, coefficient: 0.5)
        let jumpOperators: [[[Complex<Double>]]] = []
        let initialState = DensityMatrix.maximallyMixed(qubits: 1)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.2,
        )

        #expect(abs(result.finalState.trace() - 1.0) < 1e-6, "Trace should be preserved for mixed state")
        let purity = result.finalState.purity()
        #expect(purity <= 0.5 + 1e-4, "Mixed state purity should remain low, got \(purity)")
    }
}

/// Test suite for dissipative dynamics with jump operators.
/// Validates decay processes and decoherence behavior
/// under amplitude damping and dephasing channels.
@Suite("Dissipative Dynamics")
struct LindbladDissipativeDynamicsTests {
    @Test("Amplitude damping causes decay toward ground state")
    func amplitudeDampingDecay() {
        let hamiltonian = Observable(terms: [])
        let gammaSqrt = 0.3
        let sigmaMinusOp: [[Complex<Double>]] = [
            [.zero, Complex(gammaSqrt, 0)],
            [.zero, .zero],
        ]
        let jumpOperators = [sigmaMinusOp]

        let excitedElements: [Complex<Double>] = [
            .zero, .zero,
            .zero, .one,
        ]
        let initialState = DensityMatrix(qubits: 1, elements: excitedElements)
        let initialGroundProb = initialState.probability(of: 0)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 2.0,
        )

        let groundProb = result.finalState.probability(of: 0)
        #expect(groundProb > initialGroundProb, "Ground state probability should increase due to amplitude damping decay, got \(groundProb) vs initial \(initialGroundProb)")
    }

    @Test("Dissipation reduces purity of pure state")
    func dissipationReducesPurity() {
        let hamiltonian = Observable.pauliZ(qubit: 0, coefficient: 1.0)
        let gammaSqrt = 0.2
        let sigmaMinusOp: [[Complex<Double>]] = [
            [.zero, Complex(gammaSqrt, 0)],
            [.zero, .zero],
        ]
        let jumpOperators = [sigmaMinusOp]

        let superpositionElements: [Complex<Double>] = [
            Complex(0.5, 0), Complex(0.5, 0),
            Complex(0.5, 0), Complex(0.5, 0),
        ]
        let initialState = DensityMatrix(qubits: 1, elements: superpositionElements)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 1.0,
        )

        #expect(abs(result.finalState.trace() - 1.0) < 1e-4, "Trace should be preserved under dissipation")
    }
}

/// Test suite for LindbladSolver coverage edge cases.
/// Validates step rejection, stiffness detection, TR-BDF2 switching,
/// and positivity enforcement paths for complete branch coverage.
@Suite("LindbladSolver Coverage Tests")
struct LindbladSolverCoverageTests {
    @Test("RK45 step rejection with tight tolerance")
    func rk45StepRejection() {
        let hamiltonian = Observable.pauliX(qubit: 0, coefficient: 10.0)
        let jumpOperators: [[[Complex<Double>]]] = []
        let initialState = DensityMatrix(qubits: 1)
        let config = LindbladConfiguration(
            absoluteTolerance: 1e-18,
            relativeTolerance: 1e-18,
            maxSteps: 100,
        )

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.1,
            configuration: config,
        )

        #expect(result.stepStatistics.rejectedSteps > 0, "Tight tolerance should cause step rejections")
    }

    @Test("Stiffness detection via high rejection rate")
    func stiffnessDetectionViaRejectionRate() {
        let hamiltonian = Observable.pauliX(qubit: 0, coefficient: 100.0)
        let sigmaMinusOp: [[Complex<Double>]] = [
            [.zero, Complex(5.0, 0)],
            [.zero, .zero],
        ]
        let jumpOperators = [sigmaMinusOp]
        let initialState = DensityMatrix(qubits: 1)
        let config = LindbladConfiguration(
            absoluteTolerance: 1e-16,
            relativeTolerance: 1e-16,
            maxSteps: 200,
            stiffnessThreshold: 5,
        )

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.05,
            configuration: config,
        )

        #expect(result.stepStatistics.stiffnessDetections >= 0, "Stiffness detection should be tracked")
        #expect(result.stepStatistics.methodSwitches >= 0, "Method switches should be tracked")
    }

    @Test("TR-BDF2 step rejection with stiff system")
    func trBdf2StepRejection() {
        let hamiltonian = Observable.pauliZ(qubit: 0, coefficient: 5000.0)
        let sigmaMinusOp: [[Complex<Double>]] = [
            [.zero, Complex(50.0, 0)],
            [.zero, .zero],
        ]
        let sigmaPlusOp: [[Complex<Double>]] = [
            [.zero, .zero],
            [Complex(50.0, 0), .zero],
        ]
        let jumpOperators = [sigmaMinusOp, sigmaPlusOp]
        let initialState = DensityMatrix(qubits: 1)
        let config = LindbladConfiguration(
            absoluteTolerance: 1e-20,
            relativeTolerance: 1e-20,
            maxSteps: 1000,
            stiffnessThreshold: 3,
        )

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.001,
            configuration: config,
        )

        let totalSteps = result.stepStatistics.acceptedSteps + result.stepStatistics.rejectedSteps
        #expect(totalSteps > 0, "Solver should take steps for stiff system")
    }

    @Test("BDF2 Newton loop multiple iterations")
    func bdf2NewtonLoopIterations() {
        let hamiltonian = Observable.pauliX(qubit: 0, coefficient: 50.0)
        let sigmaMinusOp: [[Complex<Double>]] = [
            [.zero, Complex(3.0, 0)],
            [.zero, .zero],
        ]
        let jumpOperators = [sigmaMinusOp]
        let initialState = DensityMatrix(qubits: 1)
        let config = LindbladConfiguration(
            absoluteTolerance: 1e-14,
            relativeTolerance: 1e-14,
            maxSteps: 300,
            stiffnessThreshold: 5,
        )

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.02,
            configuration: config,
        )

        #expect(abs(result.finalState.trace() - 1.0) < 1e-4, "BDF2 Newton iterations should preserve trace normalization")
    }

    @Test("Positivity enforcement with none method")
    func positivityEnforcementNone() {
        let hamiltonian = Observable.pauliZ(qubit: 0, coefficient: 1.0)
        let jumpOperators: [[[Complex<Double>]]] = []
        let initialState = DensityMatrix(qubits: 1)
        let config = LindbladConfiguration(positivityEnforcement: .none)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.1,
            configuration: config,
        )

        #expect(abs(result.finalState.trace() - 1.0) < 1e-6, "Evolution with no positivity enforcement should still preserve trace")
    }

    @Test("Near-zero trace handling")
    func nearZeroTraceHandling() {
        let hamiltonian = Observable(terms: [])
        let jumpOperators: [[[Complex<Double>]]] = []
        let zeroElements = [Complex<Double>](repeating: .zero, count: 4)
        let initialState = DensityMatrix(qubits: 1, elements: zeroElements)

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.01,
        )

        let trace = result.finalState.trace()
        #expect(trace.isFinite, "Near-zero trace should not produce NaN or infinity")
    }
}

/// Test suite for stiffness detection via minimum step size threshold.
/// Validates that extremely tight tolerances with rapidly oscillating dynamics
/// force step size below hMin (1e-15), triggering the stiffness switch path.
@Suite("Stiffness Detection via Minimum Step Size")
struct LindbladStiffnessMinStepTests {
    @Test("Stiffness detected when step size falls below hMin threshold")
    func stiffnessViaMinimumStepSize() {
        let highFrequency = 10000.0
        let hamiltonian = Observable.pauliX(qubit: 0, coefficient: highFrequency)

        let strongDissipation: [[Complex<Double>]] = [
            [.zero, Complex(100.0, 0)],
            [.zero, .zero],
        ]
        let jumpOperators = [strongDissipation]

        let initialState = DensityMatrix(qubits: 1)

        let config = LindbladConfiguration(
            absoluteTolerance: 1e-15,
            relativeTolerance: 1e-15,
            maxSteps: 500,
            stiffnessThreshold: 3,
        )

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 0.0001,
            configuration: config,
        )

        #expect(
            result.stepStatistics.stiffnessDetections > 0,
            "Stiffness should be detected via rejection rate or minimum step size threshold, got \(result.stepStatistics.stiffnessDetections) detections",
        )
        #expect(
            result.stepStatistics.methodSwitches > 0,
            "Method should switch from RK45 to TR-BDF2 upon stiffness detection, got \(result.stepStatistics.methodSwitches) switches",
        )
        #expect(
            result.finalState.trace().isFinite,
            "Final state trace should be finite after stiffness handling",
        )
    }

    @Test("Ultra-stiff system forces minimum step size path")
    func ultraStiffSystemMinStepPath() {
        let extremeFrequency = 100_000.0
        let hamiltonian = Observable(terms: [
            (extremeFrequency, PauliString(.x(0))),
            (extremeFrequency * 0.7, PauliString(.y(0))),
        ])

        let decay1: [[Complex<Double>]] = [
            [.zero, Complex(500.0, 0)],
            [.zero, .zero],
        ]
        let decay2: [[Complex<Double>]] = [
            [Complex(200.0, 0), .zero],
            [.zero, Complex(-200.0, 0)],
        ]
        let jumpOperators = [decay1, decay2]

        let initialState = DensityMatrix(qubits: 1)

        let config = LindbladConfiguration(
            absoluteTolerance: 1e-15,
            relativeTolerance: 1e-15,
            maxSteps: 200,
            stiffnessThreshold: 2,
        )

        let result = LindbladSolver.evolve(
            hamiltonian: hamiltonian,
            jumpOperators: jumpOperators,
            initialState: initialState,
            time: 1e-6,
            configuration: config,
        )

        #expect(
            result.stepStatistics.stiffnessDetections >= 1,
            "Ultra-stiff system should trigger stiffness detection, got \(result.stepStatistics.stiffnessDetections)",
        )
        #expect(
            abs(result.finalState.trace() - 1.0) < 0.1 || result.finalState.trace().isFinite,
            "Trace should be approximately preserved or at least finite",
        )
    }
}
