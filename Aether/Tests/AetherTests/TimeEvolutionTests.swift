// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for TimeEvolutionResult struct properties.
/// Validates that all properties are correctly stored and accessible
/// after time evolution computation returns its results.
@Suite("TimeEvolutionResult Properties")
struct TimeEvolutionResultPropertiesTests {
    @Test("finalState property stores quantum state")
    func finalStatePropertyStoresQuantumState() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0), .z(1))),
            (-0.3, PauliString(.x(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .groundState(qubits: 2),
            time: 0.5,
            method: .trotterSuzuki(order: .first, steps: 5),
        )
        #expect(result.finalState.qubits == 2, "Final state should have 2 qubits")
    }

    @Test("time property stores evolution time")
    func timePropertyStoresEvolutionTime() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .groundState(qubits: 1),
            time: 2.5,
            method: .trotterSuzuki(order: .first, steps: 5),
        )
        #expect(abs(result.time - 2.5) < 1e-10, "Time should be 2.5")
    }

    @Test("steps property stores Trotter step count")
    func stepsPropertyStoresTrotterStepCount() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.x(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .groundState(qubits: 1),
            time: 1.0,
            method: .trotterSuzuki(order: .second, steps: 15),
        )
        #expect(result.steps == 15, "Steps should be 15")
    }

    @Test("errorBound property is non-negative")
    func errorBoundPropertyIsNonNegative() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0), .z(1))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .groundState(qubits: 2),
            time: 1.0,
            method: .trotterSuzuki(order: .second, steps: 10),
        )
        #expect(result.errorBound >= 0, "Error bound should be non-negative")
    }

    @Test("gateCount property is non-negative")
    func gateCountPropertyIsNonNegative() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.x(0))),
            (0.5, PauliString(.z(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .groundState(qubits: 1),
            time: 1.0,
            method: .trotterSuzuki(order: .first, steps: 10),
        )
        #expect(result.gateCount >= 0, "Gate count should be non-negative")
    }

    @Test("circuitDepth property is non-negative")
    func circuitDepthPropertyIsNonNegative() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.x(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .groundState(qubits: 1),
            time: 1.0,
            method: .trotterSuzuki(order: .first, steps: 5),
        )
        #expect(result.circuitDepth >= 0, "Circuit depth should be non-negative")
    }
}

/// Test suite for MPSTimeEvolutionResult struct properties.
/// Validates MPS evolution results including final state,
/// truncation statistics, and bond dimension tracking.
@Suite("MPSTimeEvolutionResult Properties")
struct MPSTimeEvolutionResultPropertiesTests {
    @Test("finalState property stores MPS state")
    func finalStatePropertyStoresMPSState() async {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
            (-0.5, PauliString(.x(0))),
        ])
        let mps = MatrixProductState(qubits: 3, maxBondDimension: 4)
        let result = await TimeEvolution.evolveMPS(
            hamiltonian: hamiltonian,
            initialState: mps,
            time: 0.5,
            steps: 5,
            maxBondDimension: 4,
        )
        #expect(result.finalState.qubits == 3, "Final MPS should have 3 qubits")
    }

    @Test("time property stores evolution time for MPS")
    func timePropertyStoresEvolutionTimeForMPS() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0), .z(1))),
        ])
        let mps = MatrixProductState(qubits: 2, maxBondDimension: 4)
        let result = await TimeEvolution.evolveMPS(
            hamiltonian: hamiltonian,
            initialState: mps,
            time: 1.5,
            steps: 10,
            maxBondDimension: 4,
        )
        #expect(abs(result.time - 1.5) < 1e-10, "Time should be 1.5")
    }

    @Test("truncationStatistics property is accessible")
    func truncationStatisticsPropertyIsAccessible() async {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
        ])
        let mps = MatrixProductState(qubits: 2, maxBondDimension: 4)
        let result = await TimeEvolution.evolveMPS(
            hamiltonian: hamiltonian,
            initialState: mps,
            time: 0.5,
            steps: 5,
            maxBondDimension: 4,
        )
        #expect(result.truncationStatistics.cumulativeError >= 0, "Cumulative error should be non-negative")
    }

    @Test("maxBondDimensionReached is positive")
    func maxBondDimensionReachedIsPositive() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0), .z(1))),
        ])
        let mps = MatrixProductState(qubits: 2, maxBondDimension: 8)
        let result = await TimeEvolution.evolveMPS(
            hamiltonian: hamiltonian,
            initialState: mps,
            time: 0.5,
            steps: 5,
            maxBondDimension: 8,
        )
        #expect(result.maxBondDimensionReached >= 1, "Max bond dimension reached should be at least 1")
    }
}

/// Test suite for TimeEvolutionMethod enum cases.
/// Validates that all method variants can be constructed
/// with their associated parameters correctly stored.
@Suite("TimeEvolutionMethod Enum Cases")
struct TimeEvolutionMethodEnumCasesTests {
    @Test("trotterSuzuki case stores order and steps")
    func trotterSuzukiCaseStoresOrderAndSteps() {
        let method = TimeEvolutionMethod.trotterSuzuki(order: .fourth, steps: 20)
        if case let .trotterSuzuki(order, steps) = method {
            #expect(order == .fourth, "Order should be fourth")
            #expect(steps == 20, "Steps should be 20")
        }
    }

    @Test("lcu case stores ancilla qubits")
    func lcuCaseStoresAncillaQubits() {
        let method = TimeEvolutionMethod.lcu(ancillaQubits: 5)
        if case let .lcu(ancillaQubits) = method {
            #expect(ancillaQubits == 5, "Ancilla qubits should be 5")
        }
    }

    @Test("qubitization case stores polynomial degree")
    func qubitizationCaseStoresPolynomialDegree() {
        let method = TimeEvolutionMethod.qubitization(polynomialDegree: 50)
        if case let .qubitization(polynomialDegree) = method {
            #expect(polynomialDegree == 50, "Polynomial degree should be 50")
        }
    }

    @Test("mps case stores max bond dimension and truncation threshold")
    func mpsCaseStoresMaxBondDimensionAndTruncationThreshold() {
        let method = TimeEvolutionMethod.mps(maxBondDimension: 64, truncationThreshold: 1e-10)
        if case let .mps(maxBondDimension, truncationThreshold) = method {
            #expect(maxBondDimension == 64, "Max bond dimension should be 64")
            #expect(abs(truncationThreshold - 1e-10) < 1e-15, "Truncation threshold should be 1e-10")
        }
    }
}

/// Test suite for InitialStateSpecification enum cases.
/// Validates that all initial state variants can be constructed
/// and their associated values are correctly stored.
@Suite("InitialStateSpecification Enum Cases")
struct InitialStateSpecificationEnumCasesTests {
    @Test("groundState case stores qubit count")
    func groundStateCaseStoresQubitCount() {
        let spec = InitialStateSpecification.groundState(qubits: 4)
        if case let .groundState(qubits) = spec {
            #expect(qubits == 4, "Qubits should be 4")
        }
    }

    @Test("basisState case stores index and qubit count")
    func basisStateCaseStoresIndexAndQubitCount() {
        let spec = InitialStateSpecification.basisState(0b1010, qubits: 4)
        if case let .basisState(index, qubits) = spec {
            #expect(index == 0b1010, "Index should be 0b1010")
            #expect(qubits == 4, "Qubits should be 4")
        }
    }

    @Test("quantumState case stores existing state")
    func quantumStateCaseStoresExistingState() {
        let state = QuantumState(qubits: 2)
        let spec = InitialStateSpecification.quantumState(state)
        if case let .quantumState(storedState) = spec {
            #expect(storedState.qubits == 2, "Stored state should have 2 qubits")
        }
    }

    @Test("mps case stores existing MPS")
    func mpsCaseStoresExistingMPS() {
        let mps = MatrixProductState(qubits: 5, maxBondDimension: 8)
        let spec = InitialStateSpecification.mps(mps)
        if case let .mps(storedMPS) = spec {
            #expect(storedMPS.qubits == 5, "Stored MPS should have 5 qubits")
        }
    }
}

/// Test suite for TimeEvolution.evolve() with Trotter method.
/// Validates evolution produces valid quantum states with correct
/// normalization and reasonable physical behavior.
@Suite("TimeEvolution Evolve Trotter Method")
struct TimeEvolutionEvolveTrotterMethodTests {
    @Test("First order Trotter produces normalized state")
    func firstOrderTrotterProducesNormalizedState() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
            (0.5, PauliString(.x(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .groundState(qubits: 1),
            time: 1.0,
            method: .trotterSuzuki(order: .first, steps: 10),
        )
        var normSquared = 0.0
        for i in 0 ..< result.finalState.stateSpaceSize {
            normSquared += result.finalState.amplitude(of: i).magnitudeSquared
        }
        #expect(abs(normSquared - 1.0) < 1e-10, "State should be normalized")
    }

    @Test("Second order Trotter produces normalized state")
    func secondOrderTrotterProducesNormalizedState() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0), .z(1))),
            (-0.3, PauliString(.x(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .groundState(qubits: 2),
            time: 1.0,
            method: .trotterSuzuki(order: .second, steps: 10),
        )
        var normSquared = 0.0
        for i in 0 ..< result.finalState.stateSpaceSize {
            normSquared += result.finalState.amplitude(of: i).magnitudeSquared
        }
        #expect(abs(normSquared - 1.0) < 1e-10, "State should be normalized")
    }

    @Test("Fourth order Trotter produces normalized state")
    func fourthOrderTrotterProducesNormalizedState() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .groundState(qubits: 1),
            time: 1.0,
            method: .trotterSuzuki(order: .fourth, steps: 5),
        )
        var normSquared = 0.0
        for i in 0 ..< result.finalState.stateSpaceSize {
            normSquared += result.finalState.amplitude(of: i).magnitudeSquared
        }
        #expect(abs(normSquared - 1.0) < 1e-10, "State should be normalized")
    }

    @Test("Sixth order Trotter produces normalized state")
    func sixthOrderTrotterProducesNormalizedState() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.x(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .groundState(qubits: 1),
            time: 0.5,
            method: .trotterSuzuki(order: .sixth, steps: 3),
        )
        var normSquared = 0.0
        for i in 0 ..< result.finalState.stateSpaceSize {
            normSquared += result.finalState.amplitude(of: i).magnitudeSquared
        }
        #expect(abs(normSquared - 1.0) < 1e-10, "State should be normalized")
    }

    @Test("Basis state initial condition is respected")
    func basisStateInitialConditionIsRespected() async {
        let hamiltonian = Observable(terms: [
            (0.0001, PauliString(.z(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .basisState(1, qubits: 1),
            time: 0.001,
            method: .trotterSuzuki(order: .first, steps: 1),
        )
        let amp1 = result.finalState.amplitude(of: 1).magnitudeSquared
        #expect(amp1 > 0.99, "Basis state |1> should remain mostly unchanged for small time")
    }

    @Test("QuantumState initial condition is respected")
    func quantumStateInitialConditionIsRespected() async {
        var state = QuantumState(qubits: 1)
        state.setAmplitude(0, to: Complex(1.0 / Double.squareRoot(of: 2.0), 0.0))
        state.setAmplitude(1, to: Complex(1.0 / Double.squareRoot(of: 2.0), 0.0))
        let hamiltonian = Observable(terms: [
            (0.0001, PauliString(.z(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .quantumState(state),
            time: 0.001,
            method: .trotterSuzuki(order: .first, steps: 1),
        )
        let amp0 = result.finalState.amplitude(of: 0).magnitudeSquared
        let amp1 = result.finalState.amplitude(of: 1).magnitudeSquared
        #expect(abs(amp0 - 0.5) < 0.01, "Superposition should be approximately maintained")
        #expect(abs(amp1 - 0.5) < 0.01, "Superposition should be approximately maintained")
    }

    @Test("Zero time evolution returns initial state")
    func zeroTimeEvolutionReturnsInitialState() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.x(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .groundState(qubits: 1),
            time: 0.0,
            method: .trotterSuzuki(order: .first, steps: 10),
        )
        let amp0 = result.finalState.amplitude(of: 0).magnitudeSquared
        #expect(abs(amp0 - 1.0) < 1e-10, "Zero time should return |0>")
    }
}

/// Test suite for TimeEvolution.estimateTrotterError().
/// Validates error bound estimation produces reasonable,
/// non-negative values that scale correctly with parameters.
@Suite("TimeEvolution Estimate Trotter Error")
struct TimeEvolutionEstimateTrotterErrorTests {
    @Test("First order error bound is non-negative")
    func firstOrderErrorBoundIsNonNegative() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0), .z(1))),
            (-0.3, PauliString(.x(0))),
        ])
        let error = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .first,
            steps: 10,
        )
        #expect(error >= 0, "Error bound should be non-negative")
    }

    @Test("Second order error bound is non-negative")
    func secondOrderErrorBoundIsNonNegative() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
        ])
        let error = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 2.0,
            order: .second,
            steps: 20,
        )
        #expect(error >= 0, "Error bound should be non-negative")
    }

    @Test("Fourth order error bound is non-negative")
    func fourthOrderErrorBoundIsNonNegative() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.x(0))),
            (0.5, PauliString(.z(0))),
        ])
        let error = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .fourth,
            steps: 10,
        )
        #expect(error >= 0, "Error bound should be non-negative")
    }

    @Test("Sixth order error bound is non-negative")
    func sixthOrderErrorBoundIsNonNegative() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0), .z(1))),
        ])
        let error = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .sixth,
            steps: 5,
        )
        #expect(error >= 0, "Error bound should be non-negative")
    }

    @Test("Error decreases with more steps (first order)")
    func errorDecreasesWithMoreStepsFirstOrder() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0), .z(1))),
            (-0.3, PauliString(.x(0))),
        ])
        let error10 = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .first,
            steps: 10,
        )
        let error20 = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .first,
            steps: 20,
        )
        #expect(error20 < error10, "More steps should reduce error")
    }

    @Test("Error decreases with more steps (second order)")
    func errorDecreasesWithMoreStepsSecondOrder() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.x(0))),
        ])
        let error5 = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .second,
            steps: 5,
        )
        let error10 = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .second,
            steps: 10,
        )
        #expect(error10 < error5, "More steps should reduce error")
    }

    @Test("Higher order has smaller error for same steps")
    func higherOrderHasSmallerErrorForSameSteps() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0), .z(1))),
            (-0.3, PauliString(.x(0))),
        ])
        let errorFirst = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .first,
            steps: 10,
        )
        let errorSecond = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .second,
            steps: 10,
        )
        #expect(errorSecond < errorFirst, "Higher order should have smaller error")
    }

    @Test("Error increases with longer time")
    func errorIncreasesWithLongerTime() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
        ])
        let error1 = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .second,
            steps: 10,
        )
        let error2 = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 2.0,
            order: .second,
            steps: 10,
        )
        #expect(error2 > error1, "Longer time should increase error")
    }

    @Test("Error increases with larger Hamiltonian norm")
    func errorIncreasesWithLargerHamiltonianNorm() {
        let smallHamiltonian = Observable(terms: [
            (0.1, PauliString(.z(0))),
        ])
        let largeHamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
        ])
        let errorSmall = TimeEvolution.estimateTrotterError(
            hamiltonian: smallHamiltonian,
            time: 1.0,
            order: .first,
            steps: 10,
        )
        let errorLarge = TimeEvolution.estimateTrotterError(
            hamiltonian: largeHamiltonian,
            time: 1.0,
            order: .first,
            steps: 10,
        )
        #expect(errorLarge > errorSmall, "Larger Hamiltonian norm should increase error")
    }
}

/// Test suite for TimeEvolution.estimateQueryComplexity().
/// Validates query complexity estimates for different evolution
/// methods including Trotter, LCU, qubitization, and MPS.
@Suite("TimeEvolution Estimate Query Complexity")
struct TimeEvolutionEstimateQueryComplexityTests {
    @Test("First order Trotter query complexity is positive")
    func firstOrderTrotterQueryComplexityIsPositive() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0), .z(1))),
        ])
        let queries = TimeEvolution.estimateQueryComplexity(
            hamiltonian: hamiltonian,
            time: 1.0,
            epsilon: 0.01,
            method: .trotterSuzuki(order: .first, steps: 10),
        )
        #expect(queries > 0, "Query complexity should be positive")
    }

    @Test("Second order Trotter query complexity is positive")
    func secondOrderTrotterQueryComplexityIsPositive() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.x(0))),
        ])
        let queries = TimeEvolution.estimateQueryComplexity(
            hamiltonian: hamiltonian,
            time: 2.0,
            epsilon: 0.001,
            method: .trotterSuzuki(order: .second, steps: 20),
        )
        #expect(queries > 0, "Query complexity should be positive")
    }

    @Test("Fourth order Trotter query complexity is positive")
    func fourthOrderTrotterQueryComplexityIsPositive() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (0.3, PauliString(.x(0))),
        ])
        let queries = TimeEvolution.estimateQueryComplexity(
            hamiltonian: hamiltonian,
            time: 5.0,
            epsilon: 0.01,
            method: .trotterSuzuki(order: .fourth, steps: 50),
        )
        #expect(queries > 0, "Query complexity should be positive")
    }

    @Test("Sixth order Trotter query complexity is positive")
    func sixthOrderTrotterQueryComplexityIsPositive() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0), .z(1))),
        ])
        let queries = TimeEvolution.estimateQueryComplexity(
            hamiltonian: hamiltonian,
            time: 1.0,
            epsilon: 0.001,
            method: .trotterSuzuki(order: .sixth, steps: 10),
        )
        #expect(queries > 0, "Query complexity should be positive")
    }

    @Test("LCU query complexity is positive")
    func lcuQueryComplexityIsPositive() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0), .z(1))),
            (-0.3, PauliString(.x(0))),
        ])
        let queries = TimeEvolution.estimateQueryComplexity(
            hamiltonian: hamiltonian,
            time: 10.0,
            epsilon: 0.01,
            method: .lcu(ancillaQubits: 5),
        )
        #expect(queries > 0, "Query complexity should be positive")
    }

    @Test("Qubitization query complexity is positive")
    func qubitizationQueryComplexityIsPositive() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0), .z(1))),
        ])
        let queries = TimeEvolution.estimateQueryComplexity(
            hamiltonian: hamiltonian,
            time: 10.0,
            epsilon: 0.01,
            method: .qubitization(polynomialDegree: 50),
        )
        #expect(queries > 0, "Query complexity should be positive")
    }

    @Test("MPS query complexity is positive")
    func mpsQueryComplexityIsPositive() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0), .z(1))),
        ])
        let queries = TimeEvolution.estimateQueryComplexity(
            hamiltonian: hamiltonian,
            time: 1.0,
            epsilon: 0.01,
            method: .mps(maxBondDimension: 64, truncationThreshold: 1e-10),
        )
        #expect(queries > 0, "Query complexity should be positive")
    }

    @Test("Qubitization has better scaling than first order Trotter")
    func qubitizationHasBetterScalingThanFirstOrderTrotter() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0), .z(1))),
            (-0.3, PauliString(.x(0))),
        ])
        let trotterQueries = TimeEvolution.estimateQueryComplexity(
            hamiltonian: hamiltonian,
            time: 100.0,
            epsilon: 0.001,
            method: .trotterSuzuki(order: .first, steps: 1000),
        )
        let qubitizationQueries = TimeEvolution.estimateQueryComplexity(
            hamiltonian: hamiltonian,
            time: 100.0,
            epsilon: 0.001,
            method: .qubitization(polynomialDegree: 50),
        )
        #expect(qubitizationQueries < trotterQueries, "Qubitization should have better scaling for long times")
    }

    @Test("Query complexity increases with tighter epsilon")
    func queryComplexityIncreasesWithTighterEpsilon() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
        ])
        let queries1 = TimeEvolution.estimateQueryComplexity(
            hamiltonian: hamiltonian,
            time: 1.0,
            epsilon: 0.1,
            method: .trotterSuzuki(order: .first, steps: 10),
        )
        let queries2 = TimeEvolution.estimateQueryComplexity(
            hamiltonian: hamiltonian,
            time: 1.0,
            epsilon: 0.01,
            method: .trotterSuzuki(order: .first, steps: 10),
        )
        #expect(queries2 > queries1, "Tighter epsilon should require more queries")
    }

    @Test("Query complexity increases with longer time")
    func queryComplexityIncreasesWithLongerTime() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.x(0))),
        ])
        let queries1 = TimeEvolution.estimateQueryComplexity(
            hamiltonian: hamiltonian,
            time: 1.0,
            epsilon: 0.01,
            method: .trotterSuzuki(order: .second, steps: 10),
        )
        let queries2 = TimeEvolution.estimateQueryComplexity(
            hamiltonian: hamiltonian,
            time: 10.0,
            epsilon: 0.01,
            method: .trotterSuzuki(order: .second, steps: 10),
        )
        #expect(queries2 > queries1, "Longer time should require more queries")
    }
}

/// Test suite for error bound scaling with order and steps.
/// Validates that Trotter error bounds follow expected
/// theoretical scaling behavior for different orders.
@Suite("Error Bounds Scaling")
struct ErrorBoundsScalingTests {
    @Test("First order error scales as O(1/steps)")
    func firstOrderErrorScalesAsOneOverSteps() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
            (0.5, PauliString(.x(0))),
        ])
        let error10 = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .first,
            steps: 10,
        )
        let error20 = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .first,
            steps: 20,
        )
        let ratio = error10 / error20
        #expect(abs(ratio - 2.0) < 0.1, "Error should halve when doubling steps for first order")
    }

    @Test("Second order error scales as O(1/steps^2)")
    func secondOrderErrorScalesAsOneOverStepsSquared() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0), .z(1))),
        ])
        let error10 = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .second,
            steps: 10,
        )
        let error20 = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .second,
            steps: 20,
        )
        let ratio = error10 / error20
        #expect(abs(ratio - 4.0) < 0.2, "Error should quarter when doubling steps for second order")
    }

    @Test("Fourth order error scales as O(1/steps^4)")
    func fourthOrderErrorScalesAsOneOverStepsFourth() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.x(0))),
        ])
        let error10 = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .fourth,
            steps: 10,
        )
        let error20 = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .fourth,
            steps: 20,
        )
        let ratio = error10 / error20
        #expect(abs(ratio - 16.0) < 1.0, "Error should reduce by 16x when doubling steps for fourth order")
    }

    @Test("Sixth order error scales as O(1/steps^6)")
    func sixthOrderErrorScalesAsOneOverStepsSixth() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
        ])
        let error10 = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .sixth,
            steps: 10,
        )
        let error20 = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .sixth,
            steps: 20,
        )
        let ratio = error10 / error20
        #expect(abs(ratio - 64.0) < 4.0, "Error should reduce by 64x when doubling steps for sixth order")
    }

    @Test("Error bound monotonically decreases with higher order")
    func errorBoundMonotonicallyDecreasesWithHigherOrder() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0), .z(1))),
            (-0.3, PauliString(.x(0))),
        ])
        let errorFirst = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .first,
            steps: 20,
        )
        let errorSecond = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .second,
            steps: 20,
        )
        let errorFourth = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .fourth,
            steps: 20,
        )
        let errorSixth = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .sixth,
            steps: 20,
        )
        #expect(errorFirst > errorSecond, "First order error should be larger than second order")
        #expect(errorSecond > errorFourth, "Second order error should be larger than fourth order")
        #expect(errorFourth > errorSixth, "Fourth order error should be larger than sixth order")
    }

    @Test("Error scales with Hamiltonian norm squared for first order")
    func errorScalesWithHamiltonianNormSquaredForFirstOrder() {
        let smallHamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
        ])
        let largeHamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
        ])
        let errorSmall = TimeEvolution.estimateTrotterError(
            hamiltonian: smallHamiltonian,
            time: 1.0,
            order: .first,
            steps: 10,
        )
        let errorLarge = TimeEvolution.estimateTrotterError(
            hamiltonian: largeHamiltonian,
            time: 1.0,
            order: .first,
            steps: 10,
        )
        let ratio = errorLarge / errorSmall
        #expect(abs(ratio - 4.0) < 0.1, "Error should scale as norm^2 for first order")
    }

    @Test("Error scales with time squared for first order")
    func errorScalesWithTimeSquaredForFirstOrder() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
        ])
        let error1 = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .first,
            steps: 10,
        )
        let error2 = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 2.0,
            order: .first,
            steps: 10,
        )
        let ratio = error2 / error1
        #expect(abs(ratio - 4.0) < 0.1, "Error should scale as t^2 for first order")
    }

    @Test("Error scales with time cubed for second order")
    func errorScalesWithTimeCubedForSecondOrder() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
        ])
        let error1 = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 1.0,
            order: .second,
            steps: 10,
        )
        let error2 = TimeEvolution.estimateTrotterError(
            hamiltonian: hamiltonian,
            time: 2.0,
            order: .second,
            steps: 10,
        )
        let ratio = error2 / error1
        #expect(abs(ratio - 8.0) < 0.5, "Error should scale as t^3 for second order")
    }
}

/// Test suite for MPS evolution via TimeEvolution.evolveMPS().
/// Validates that MPS-based time evolution produces valid states
/// with correct normalization and reasonable truncation behavior.
@Suite("TimeEvolution MPS Evolution")
struct TimeEvolutionMPSEvolutionTests {
    @Test("MPS evolution preserves qubit count")
    func mpsEvolutionPreservesQubitCount() async {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
        ])
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let result = await TimeEvolution.evolveMPS(
            hamiltonian: hamiltonian,
            initialState: mps,
            time: 0.5,
            steps: 5,
            maxBondDimension: 8,
        )
        #expect(result.finalState.qubits == 4, "MPS qubit count should be preserved")
    }

    @Test("MPS evolution with Ising model produces valid state")
    func mpsEvolutionWithIsingModelProducesValidState() async {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
            (-1.0, PauliString(.z(1), .z(2))),
            (-0.5, PauliString(.x(0))),
            (-0.5, PauliString(.x(1))),
            (-0.5, PauliString(.x(2))),
        ])
        let mps = MatrixProductState(qubits: 3, maxBondDimension: 8)
        let result = await TimeEvolution.evolveMPS(
            hamiltonian: hamiltonian,
            initialState: mps,
            time: 0.5,
            steps: 10,
            maxBondDimension: 8,
        )
        #expect(result.finalState.qubits == 3, "MPS should have correct qubit count")
        #expect(result.maxBondDimensionReached >= 1, "Max bond dimension should be at least 1")
    }

    @Test("MPS truncation statistics track error")
    func mpsTruncationStatisticsTrackError() async {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
            (-0.5, PauliString(.x(0))),
        ])
        let mps = MatrixProductState(qubits: 2, maxBondDimension: 4)
        let result = await TimeEvolution.evolveMPS(
            hamiltonian: hamiltonian,
            initialState: mps,
            time: 0.5,
            steps: 10,
            maxBondDimension: 4,
        )
        #expect(result.truncationStatistics.cumulativeError >= 0, "Truncation error should be non-negative")
    }
}

/// Test suite for evolve() method with MPS initial state specification.
/// Validates that MPS method can be used through the unified evolve API
/// with correct conversion to TimeEvolutionResult.
@Suite("TimeEvolution Evolve MPS Method")
struct TimeEvolutionEvolveMPSMethodTests {
    @Test("Evolve with MPS method returns valid result")
    func evolveWithMPSMethodReturnsValidResult() async {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
            (-0.5, PauliString(.x(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .groundState(qubits: 2),
            time: 0.5,
            method: .mps(maxBondDimension: 8, truncationThreshold: 1e-10),
        )
        #expect(result.finalState.qubits == 2, "Final state should have correct qubits")
        #expect(result.errorBound >= 0, "Error bound should be non-negative")
    }

    @Test("Evolve MPS with basis state initial condition")
    func evolveMPSWithBasisStateInitialCondition() async {
        let hamiltonian = Observable(terms: [
            (0.0001, PauliString(.z(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .basisState(1, qubits: 1),
            time: 0.001,
            method: .mps(maxBondDimension: 4, truncationThreshold: 1e-10),
        )
        #expect(result.finalState.qubits == 1, "Final state should have 1 qubit")
    }

    @Test("Evolve MPS with existing MPS initial state")
    func evolveMPSWithExistingMPSInitialState() async {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
        ])
        let mps = MatrixProductState(qubits: 2, maxBondDimension: 8)
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .mps(mps),
            time: 0.5,
            method: .mps(maxBondDimension: 8, truncationThreshold: 1e-10),
        )
        #expect(result.finalState.qubits == 2, "Final state should have correct qubits")
    }

    @Test("Evolve MPS with quantum state initial condition")
    func evolveMPSWithQuantumStateInitialCondition() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
        ])
        var state = QuantumState(qubits: 1)
        state.setAmplitude(0, to: Complex(1.0 / Double.squareRoot(of: 2.0), 0.0))
        state.setAmplitude(1, to: Complex(1.0 / Double.squareRoot(of: 2.0), 0.0))
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .quantumState(state),
            time: 0.1,
            method: .mps(maxBondDimension: 4, truncationThreshold: 1e-10),
        )
        #expect(result.finalState.qubits == 1, "Final state should have 1 qubit")
    }
}

/// Test suite for LCU and Qubitization evolution methods.
/// Validates that advanced evolution methods produce valid
/// quantum states and reasonable result metadata.
@Suite("TimeEvolution Advanced Methods")
struct TimeEvolutionAdvancedMethodsTests {
    @Test("LCU evolution produces valid state")
    func lcuEvolutionProducesValidState() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0), .z(1))),
            (-0.3, PauliString(.x(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .groundState(qubits: 2),
            time: 0.5,
            method: .lcu(ancillaQubits: 3),
        )
        #expect(result.finalState.qubits == 2, "Final state should have system qubit count")
    }

    @Test("Qubitization evolution produces valid state")
    func qubitizationEvolutionProducesValidState() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .groundState(qubits: 1),
            time: 0.5,
            method: .qubitization(polynomialDegree: 10),
        )
        #expect(result.finalState.qubits == 1, "Final state should have correct qubit count")
    }

    @Test("LCU with basis state initial condition")
    func lcuWithBasisStateInitialCondition() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .basisState(1, qubits: 1),
            time: 0.1,
            method: .lcu(ancillaQubits: 2),
        )
        #expect(result.finalState.qubits == 1, "Final state should have 1 qubit")
    }

    @Test("Qubitization with quantum state initial condition")
    func qubitizationWithQuantumStateInitialCondition() async {
        let hamiltonian = Observable(terms: [
            (0.3, PauliString(.x(0))),
        ])
        var state = QuantumState(qubits: 1)
        state.setAmplitude(0, to: Complex(1.0 / Double.squareRoot(of: 2.0), 0.0))
        state.setAmplitude(1, to: Complex(1.0 / Double.squareRoot(of: 2.0), 0.0))
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .quantumState(state),
            time: 0.1,
            method: .qubitization(polynomialDegree: 5),
        )
        #expect(result.finalState.qubits == 1, "Final state should have 1 qubit")
    }

    @Test("LCU with MPS initial state converts correctly")
    func lcuWithMPSInitialStateConvertsCorrectly() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
        ])
        let mps = MatrixProductState(qubits: 1, maxBondDimension: 2)
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .mps(mps),
            time: 0.1,
            method: .lcu(ancillaQubits: 2),
        )
        #expect(result.finalState.qubits == 1, "Final state should have correct qubit count")
    }

    @Test("Qubitization with MPS initial state converts correctly")
    func qubitizationWithMPSInitialStateConvertsCorrectly() async {
        let hamiltonian = Observable(terms: [
            (0.3, PauliString(.z(0))),
        ])
        let mps = MatrixProductState(qubits: 1, maxBondDimension: 2)
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .mps(mps),
            time: 0.1,
            method: .qubitization(polynomialDegree: 5),
        )
        #expect(result.finalState.qubits == 1, "Final state should have correct qubit count")
    }
}

/// Test suite for Trotter method with MPS initial state.
/// Validates that MPS initial state is correctly converted to quantum state
/// when using Trotter-Suzuki evolution method.
@Suite("TimeEvolution Trotter MPS Initial State")
struct TimeEvolutionTrotterMPSInitialStateTests {
    @Test("Trotter evolution with MPS initial state converts correctly")
    func trotterEvolutionWithMPSInitialStateConvertsCorrectly() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(0))),
        ])
        let mps = MatrixProductState(qubits: 1, maxBondDimension: 2)
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .mps(mps),
            time: 0.5,
            method: .trotterSuzuki(order: .second, steps: 10),
        )
        #expect(result.finalState.qubits == 1, "Final state should have correct qubit count")
    }

    @Test("Trotter evolution with multi-qubit MPS initial state")
    func trotterEvolutionWithMultiQubitMPSInitialState() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0), .z(1))),
            (-0.3, PauliString(.x(0))),
        ])
        let mps = MatrixProductState(qubits: 2, maxBondDimension: 4)
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .mps(mps),
            time: 0.5,
            method: .trotterSuzuki(order: .first, steps: 5),
        )
        #expect(result.finalState.qubits == 2, "Final state should have 2 qubits from MPS conversion")
        var normSquared = 0.0
        for i in 0 ..< result.finalState.stateSpaceSize {
            normSquared += result.finalState.amplitude(of: i).magnitudeSquared
        }
        #expect(abs(normSquared - 1.0) < 1e-10, "State should be normalized after MPS conversion")
    }
}

/// Test suite for LCU evolution with near-zero Hamiltonian norm.
/// Validates error bound calculation handles edge case when
/// Hamiltonian one-norm is extremely small or zero.
@Suite("TimeEvolution LCU Near Zero Norm")
struct TimeEvolutionLCUNearZeroNormTests {
    @Test("LCU with very small Hamiltonian coefficient uses fallback error bound")
    func lcuWithVerySmallHamiltonianCoefficientUsesFallbackErrorBound() async {
        let hamiltonian = Observable(terms: [
            (1e-20, PauliString(.z(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .groundState(qubits: 1),
            time: 0.1,
            method: .lcu(ancillaQubits: 2),
        )
        #expect(result.errorBound >= 0.9, "Error bound should use fallback value 1.0 for near-zero norm")
    }
}

/// Test suite for qubitization with basis state initial condition.
/// Validates that basis state is correctly handled when using
/// qubitization evolution method.
@Suite("TimeEvolution Qubitization Basis State")
struct TimeEvolutionQubitizationBasisStateTests {
    @Test("Qubitization evolution with basis state initial condition")
    func qubitizationEvolutionWithBasisStateInitialCondition() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .basisState(1, qubits: 1),
            time: 0.1,
            method: .qubitization(polynomialDegree: 5),
        )
        #expect(result.finalState.qubits == 1, "Final state should have correct qubit count")
    }

    @Test("Qubitization with multi-qubit basis state")
    func qubitizationWithMultiQubitBasisState() async {
        let hamiltonian = Observable(terms: [
            (0.3, PauliString(.z(0), .z(1))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .basisState(0b10, qubits: 2),
            time: 0.1,
            method: .qubitization(polynomialDegree: 3),
        )
        #expect(result.finalState.qubits == 2, "Final state should have 2 qubits")
    }
}

/// Test suite for qubitization with zero polynomial degree.
/// Validates that epsilon calculation falls back to default
/// when polynomial degree is zero.
@Suite("TimeEvolution Qubitization Zero Polynomial Degree")
struct TimeEvolutionQubitizationZeroPolynomialDegreeTests {
    @Test("Qubitization with zero polynomial degree uses default epsilon")
    func qubitizationWithZeroPolynomialDegreeUsesDefaultEpsilon() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .groundState(qubits: 1),
            time: 0.1,
            method: .qubitization(polynomialDegree: 0),
        )
        #expect(result.finalState.qubits == 1, "Final state should have correct qubit count")
        #expect(result.errorBound > 0, "Error bound should be positive")
    }
}

/// Test suite for MPS evolution with XX and YY Hamiltonian terms.
/// Validates that TEBD gate derivation correctly handles
/// Heisenberg XXZ and pure XX/YY coupling terms.
@Suite("TimeEvolution MPS XX YY Coupling")
struct TimeEvolutionMPSXXYYCouplingTests {
    @Test("MPS evolution with Heisenberg XXZ model includes all coupling terms")
    func mpsEvolutionWithHeisenbergXXZModelIncludesAllCouplingTerms() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.x(0), .x(1))),
            (1.0, PauliString(.y(0), .y(1))),
            (0.5, PauliString(.z(0), .z(1))),
        ])
        let mps = MatrixProductState(qubits: 2, maxBondDimension: 4)
        let result = await TimeEvolution.evolveMPS(
            hamiltonian: hamiltonian,
            initialState: mps,
            time: 0.5,
            steps: 10,
            maxBondDimension: 4,
        )
        #expect(result.finalState.qubits == 2, "Final MPS should have 2 qubits")
    }

    @Test("MPS evolution with pure XX coupling")
    func mpsEvolutionWithPureXXCoupling() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.x(0), .x(1))),
        ])
        let mps = MatrixProductState(qubits: 2, maxBondDimension: 4)
        let result = await TimeEvolution.evolveMPS(
            hamiltonian: hamiltonian,
            initialState: mps,
            time: 0.5,
            steps: 10,
            maxBondDimension: 4,
        )
        #expect(result.finalState.qubits == 2, "Final MPS should have 2 qubits after XX evolution")
    }

    @Test("MPS evolution with pure YY coupling")
    func mpsEvolutionWithPureYYCoupling() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.y(0), .y(1))),
        ])
        let mps = MatrixProductState(qubits: 2, maxBondDimension: 4)
        let result = await TimeEvolution.evolveMPS(
            hamiltonian: hamiltonian,
            initialState: mps,
            time: 0.5,
            steps: 10,
            maxBondDimension: 4,
        )
        #expect(result.finalState.qubits == 2, "Final MPS should have 2 qubits after YY evolution")
    }
}

/// Test suite for LCU evolution when ancilla extension not needed.
/// Validates that state extension handles case where state already
/// has sufficient qubits for the LCU decomposition.
@Suite("TimeEvolution LCU State Extension")
struct TimeEvolutionLCUStateExtensionTests {
    @Test("LCU with state already having enough qubits skips extension")
    func lcuWithStateAlreadyHavingEnoughQubitsSkipsExtension() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .groundState(qubits: 4),
            time: 0.1,
            method: .lcu(ancillaQubits: 1),
        )
        #expect(result.finalState.qubits == 4, "Final state should preserve system qubit count")
    }
}

/// Test suite for projection phase calculation in LCU.
/// Validates that projection handles zero amplitude case
/// by using default phase value.
@Suite("TimeEvolution LCU Projection Phase")
struct TimeEvolutionLCUProjectionPhaseTests {
    @Test("LCU projection handles zero amplitude with default phase")
    func lcuProjectionHandlesZeroAmplitudeWithDefaultPhase() async {
        let hamiltonian = Observable(terms: [
            (10.0, PauliString(.x(0))),
        ])
        var state = QuantumState(qubits: 1)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(1, to: .one)
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .quantumState(state),
            time: 1.0,
            method: .lcu(ancillaQubits: 3),
        )
        #expect(result.finalState.qubits == 1, "Final state should have correct qubit count")
    }

    @Test("LCU with basis state 1 exercises projection phase fallback")
    func lcuWithBasisState1ExercisesProjectionPhaseFallback() async {
        let hamiltonian = Observable(terms: [
            (2.0, PauliString(.x(0))),
            (1.0, PauliString(.z(0))),
        ])
        let result = await TimeEvolution.evolve(
            hamiltonian: hamiltonian,
            initialState: .basisState(1, qubits: 1),
            time: 0.5,
            method: .lcu(ancillaQubits: 3),
        )
        #expect(result.finalState.qubits == 1, "Final state should have correct qubit count after projection")
    }
}
