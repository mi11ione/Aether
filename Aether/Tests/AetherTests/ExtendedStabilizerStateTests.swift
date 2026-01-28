// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Testing

/// Test suite for ExtendedStabilizerState initialization and basic properties.
/// Validates that initial state creates single-term decomposition with rank 1
/// and correct qubit count and maxRank configuration.
@Suite("ExtendedStabilizerState Initialization")
struct ExtendedStabilizerStateInitTests {
    @Test("Init creates single-term state with rank 1")
    func initCreatesSingleTermState() {
        let state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        #expect(state.rank == 1, "Initial state should have exactly one stabilizer term")
    }

    @Test("Init stores correct qubit count")
    func initStoresCorrectQubitCount() {
        let state = ExtendedStabilizerState(qubits: 3, maxRank: 128)
        #expect(state.qubits == 3, "State should report correct number of qubits")
    }

    @Test("Init stores correct maxRank")
    func initStoresCorrectMaxRank() {
        let state = ExtendedStabilizerState(qubits: 2, maxRank: 256)
        #expect(state.maxRank == 256, "State should report correct maxRank limit")
    }

    @Test("Initial state has amplitude 1 for |0...0>")
    func initialStateAmplitudeZero() {
        let state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        let amp = state.amplitude(of: 0)
        #expect(abs(amp.real - 1.0) < 1e-10, "Amplitude of |00> should be 1.0")
        #expect(abs(amp.imaginary) < 1e-10, "Amplitude of |00> should have zero imaginary part")
    }

    @Test("Initial state has amplitude 0 for |1...1>")
    func initialStateAmplitudeOne() {
        let state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        let amp = state.amplitude(of: 3)
        #expect(abs(amp.real) < 1e-10, "Amplitude of |11> should be 0")
        #expect(abs(amp.imaginary) < 1e-10, "Amplitude of |11> should be 0")
    }
}

/// Test suite for Clifford gate application on ExtendedStabilizerState.
/// Validates that Clifford gates (H, S, X, Y, Z, CNOT, CZ) do not increase
/// the stabilizer rank, maintaining efficient simulation.
@Suite("ExtendedStabilizerState Clifford Gates")
struct ExtendedStabilizerStateCliffordTests {
    @Test("Hadamard gate does not increase rank")
    func hadamardDoesNotIncreaseRank() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state.apply(.hadamard, to: 0)
        #expect(state.rank == 1, "Hadamard should not increase stabilizer rank")
    }

    @Test("S gate does not increase rank")
    func sGateDoesNotIncreaseRank() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state.apply(.hadamard, to: 0)
        state.apply(.sGate, to: 0)
        #expect(state.rank == 1, "S gate should not increase stabilizer rank")
    }

    @Test("Pauli X gate does not increase rank")
    func pauliXDoesNotIncreaseRank() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state.apply(.pauliX, to: 0)
        #expect(state.rank == 1, "Pauli X should not increase stabilizer rank")
    }

    @Test("Pauli Y gate does not increase rank")
    func pauliYDoesNotIncreaseRank() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state.apply(.pauliY, to: 1)
        #expect(state.rank == 1, "Pauli Y should not increase stabilizer rank")
    }

    @Test("Pauli Z gate does not increase rank")
    func pauliZDoesNotIncreaseRank() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state.apply(.pauliZ, to: 0)
        #expect(state.rank == 1, "Pauli Z should not increase stabilizer rank")
    }

    @Test("CNOT gate does not increase rank")
    func cnotDoesNotIncreaseRank() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state.apply(.hadamard, to: 0)
        state.apply(.cnot, to: [0, 1])
        #expect(state.rank == 1, "CNOT should not increase stabilizer rank")
    }

    @Test("CZ gate does not increase rank")
    func czDoesNotIncreaseRank() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state.apply(.hadamard, to: 0)
        state.apply(.hadamard, to: 1)
        state.apply(.cz, to: [0, 1])
        #expect(state.rank == 1, "CZ should not increase stabilizer rank")
    }

    @Test("Multiple Clifford gates do not increase rank")
    func multipleCliffordsDoNotIncreaseRank() {
        var state = ExtendedStabilizerState(qubits: 3, maxRank: 64)
        state.apply(.hadamard, to: 0)
        state.apply(.cnot, to: [0, 1])
        state.apply(.sGate, to: 1)
        state.apply(.hadamard, to: 2)
        state.apply(.cz, to: [1, 2])
        #expect(state.rank == 1, "Sequence of Clifford gates should maintain rank 1")
    }
}

/// Test suite for T gate (non-Clifford) behavior on ExtendedStabilizerState.
/// Validates that each T gate doubles the stabilizer rank according to
/// the decomposition T|psi> = (|psi> + e^(i*pi/4) Z|psi>) / sqrt(2).
@Suite("ExtendedStabilizerState T Gate")
struct ExtendedStabilizerStateTGateTests {
    @Test("Single T gate doubles rank from 1 to 2")
    func singleTGateDoublesRank() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state.apply(.hadamard, to: 0)
        state.apply(.tGate, to: 0)
        #expect(state.rank == 2, "Single T gate should double rank from 1 to 2")
    }

    @Test("Two T gates quadruple rank from 1 to 4")
    func twoTGatesQuadrupleRank() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state.apply(.hadamard, to: 0)
        state.apply(.tGate, to: 0)
        state.apply(.tGate, to: 1)
        #expect(state.rank == 4, "Two T gates should quadruple rank to 4")
    }

    @Test("Three T gates increase rank to 8")
    func threeTGatesIncreaseRankTo8() {
        var state = ExtendedStabilizerState(qubits: 3, maxRank: 64)
        state.apply(.hadamard, to: 0)
        state.apply(.tGate, to: 0)
        state.apply(.tGate, to: 1)
        state.apply(.tGate, to: 2)
        #expect(state.rank == 8, "Three T gates should increase rank to 8")
    }

    @Test("T gate followed by Clifford maintains doubled rank")
    func tGateFollowedByCliffordMaintainsRank() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state.apply(.hadamard, to: 0)
        state.apply(.tGate, to: 0)
        state.apply(.cnot, to: [0, 1])
        state.apply(.hadamard, to: 1)
        #expect(state.rank == 2, "Clifford gates after T should not change rank")
    }

    @Test("T gate respects maxRank limit")
    func tGateRespectsMaxRankLimit() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 2)
        state.apply(.hadamard, to: 0)
        state.apply(.tGate, to: 0)
        state.apply(.tGate, to: 1)
        #expect(state.rank <= 2, "T gate should not exceed maxRank limit")
    }
}

/// Test suite for amplitude computation on ExtendedStabilizerState.
/// Validates that computed amplitudes match expected values for known
/// quantum states after applying specific gate sequences.
@Suite("ExtendedStabilizerState Amplitude")
struct ExtendedStabilizerStateAmplitudeTests {
    @Test("Hadamard creates equal superposition amplitudes")
    func hadamardCreatesEqualSuperposition() {
        var state = ExtendedStabilizerState(qubits: 1, maxRank: 64)
        state.apply(.hadamard, to: 0)
        let amp0 = state.amplitude(of: 0)
        let amp1 = state.amplitude(of: 1)
        let invSqrt2 = 1.0 / 2.0.squareRoot()
        #expect(abs(amp0.real - invSqrt2) < 1e-10, "Amplitude of |0> after H should be 1/sqrt(2)")
        #expect(abs(amp1.real - invSqrt2) < 1e-10, "Amplitude of |1> after H should be 1/sqrt(2)")
    }

    @Test("Pauli X flips |0> to |1>")
    func pauliXFlipsState() {
        var state = ExtendedStabilizerState(qubits: 1, maxRank: 64)
        state.apply(.pauliX, to: 0)
        let amp0 = state.amplitude(of: 0)
        let amp1 = state.amplitude(of: 1)
        #expect(abs(amp0.real) < 1e-10, "Amplitude of |0> after X should be 0")
        #expect(abs(amp1.real - 1.0) < 1e-10, "Amplitude of |1> after X should be 1")
    }

    @Test("Bell state has correct amplitudes")
    func bellStateAmplitudes() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state.apply(.hadamard, to: 0)
        state.apply(.cnot, to: [0, 1])
        let amp00 = state.amplitude(of: 0)
        let amp01 = state.amplitude(of: 1)
        let amp10 = state.amplitude(of: 2)
        let amp11 = state.amplitude(of: 3)
        let invSqrt2 = 1.0 / 2.0.squareRoot()
        #expect(abs(amp00.real - invSqrt2) < 1e-10, "Bell state |00> amplitude should be 1/sqrt(2)")
        #expect(abs(amp01.real) < 1e-10, "Bell state |01> amplitude should be 0")
        #expect(abs(amp10.real) < 1e-10, "Bell state |10> amplitude should be 0")
        #expect(abs(amp11.real - invSqrt2) < 1e-10, "Bell state |11> amplitude should be 1/sqrt(2)")
    }

    @Test("H-T-H state has expected amplitudes")
    func htHStateAmplitudes() {
        var state = ExtendedStabilizerState(qubits: 1, maxRank: 64)
        state.apply(.hadamard, to: 0)
        state.apply(.tGate, to: 0)
        state.apply(.hadamard, to: 0)
        let amp0 = state.amplitude(of: 0)
        let amp1 = state.amplitude(of: 1)
        let prob0 = amp0.magnitudeSquared
        let prob1 = amp1.magnitudeSquared
        #expect(abs(prob0 + prob1 - 1.0) < 1e-10, "Total probability should be 1")
    }
}

/// Test suite for probability computation on ExtendedStabilizerState.
/// Validates that probability equals |amplitude|^2 (Born rule) and
/// total probability sums to 1 for valid quantum states.
@Suite("ExtendedStabilizerState Probability")
struct ExtendedStabilizerStateProbabilityTests {
    @Test("Probability equals magnitude squared of amplitude")
    func probabilityEqualsMagnitudeSquared() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state.apply(.hadamard, to: 0)
        state.apply(.tGate, to: 0)
        let amp = state.amplitude(of: 0)
        let prob = state.probability(of: 0)
        #expect(abs(prob - amp.magnitudeSquared) < 1e-10, "Probability should equal |amplitude|^2")
    }

    @Test("Total probability sums to 1 for Clifford state")
    func totalProbabilitySumsToOneClifford() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state.apply(.hadamard, to: 0)
        state.apply(.cnot, to: [0, 1])
        var total = 0.0
        for basis in 0 ..< 4 {
            total += state.probability(of: basis)
        }
        #expect(abs(total - 1.0) < 1e-10, "Total probability should sum to 1")
    }

    @Test("Total probability sums to 1 for T-gate state")
    func totalProbabilitySumsToOneTGate() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state.apply(.hadamard, to: 0)
        state.apply(.tGate, to: 0)
        state.apply(.cnot, to: [0, 1])
        var total = 0.0
        for basis in 0 ..< 4 {
            total += state.probability(of: basis)
        }
        #expect(abs(total - 1.0) < 1e-10, "Total probability with T gate should sum to 1")
    }

    @Test("Equal superposition has 50% probability each")
    func equalSuperpositionProbabilities() {
        var state = ExtendedStabilizerState(qubits: 1, maxRank: 64)
        state.apply(.hadamard, to: 0)
        let prob0 = state.probability(of: 0)
        let prob1 = state.probability(of: 1)
        #expect(abs(prob0 - 0.5) < 1e-10, "Probability of |0> after H should be 0.5")
        #expect(abs(prob1 - 0.5) < 1e-10, "Probability of |1> after H should be 0.5")
    }

    @Test("Deterministic state has probability 1 for single outcome")
    func deterministicStateProbability() {
        let state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        let prob00 = state.probability(of: 0)
        let prob11 = state.probability(of: 3)
        #expect(abs(prob00 - 1.0) < 1e-10, "Initial state should have prob 1 for |00>")
        #expect(abs(prob11) < 1e-10, "Initial state should have prob 0 for |11>")
    }
}

/// Test suite for measurement operations on ExtendedStabilizerState.
/// Validates that measure returns valid outcomes (0 or 1) and that
/// seeded measurements produce reproducible results.
@Suite("ExtendedStabilizerState Measurement")
struct ExtendedStabilizerStateMeasurementTests {
    @Test("Measurement returns 0 or 1")
    func measurementReturnsValidOutcome() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state.apply(.hadamard, to: 0)
        let outcome = state.measure(0, seed: 42)
        #expect(outcome == 0 || outcome == 1, "Measurement outcome should be 0 or 1")
    }

    @Test("Seeded measurement is reproducible")
    func seededMeasurementIsReproducible() {
        var state1 = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state1.apply(.hadamard, to: 0)
        let outcome1 = state1.measure(0, seed: 123)

        var state2 = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state2.apply(.hadamard, to: 0)
        let outcome2 = state2.measure(0, seed: 123)

        #expect(outcome1 == outcome2, "Same seed should produce same measurement outcome")
    }

    @Test("Deterministic state always measures 0")
    func deterministicStateMeasuresZero() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        let outcome = state.measure(0, seed: nil)
        #expect(outcome == 0, "Initial |0> state should always measure 0")
    }

    @Test("Pauli X state always measures 1")
    func pauliXStateMeasuresOne() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state.apply(.pauliX, to: 0)
        let outcome = state.measure(0, seed: nil)
        #expect(outcome == 1, "State |1> should always measure 1")
    }

    @Test("Measurement with T gate produces valid outcome")
    func measurementWithTGate() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state.apply(.hadamard, to: 0)
        state.apply(.tGate, to: 0)
        state.apply(.cnot, to: [0, 1])
        let outcome = state.measure(0, seed: 456)
        #expect(outcome == 0 || outcome == 1, "T-gate state measurement should return 0 or 1")
    }

    @Test("Measurement of second qubit in Bell state")
    func measurementOfBellState() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state.apply(.hadamard, to: 0)
        state.apply(.cnot, to: [0, 1])
        let outcome = state.measure(1, seed: 789)
        #expect(outcome == 0 || outcome == 1, "Bell state measurement should return 0 or 1")
    }
}

/// Test suite for CustomStringConvertible and Equatable conformance.
/// Validates that description provides meaningful output and that
/// equality comparison works correctly for identical states.
@Suite("ExtendedStabilizerState Description and Equality")
struct ExtendedStabilizerStateDescriptionTests {
    @Test("Description contains qubit count")
    func descriptionContainsQubitCount() {
        let state = ExtendedStabilizerState(qubits: 3, maxRank: 64)
        #expect(state.description.contains("3"), "Description should contain qubit count")
    }

    @Test("Description contains rank")
    func descriptionContainsRank() {
        var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state.apply(.tGate, to: 0)
        #expect(state.description.contains("rank=2"), "Description should contain current rank")
    }

    @Test("Identical initial states are equal")
    func identicalInitialStatesAreEqual() {
        let state1 = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        let state2 = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        #expect(state1 == state2, "Two initial states with same parameters should be equal")
    }

    @Test("States with different operations are not equal")
    func statesWithDifferentOperationsNotEqual() {
        var state1 = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        let state2 = ExtendedStabilizerState(qubits: 2, maxRank: 64)
        state1.apply(.hadamard, to: 0)
        #expect(state1 != state2, "States with different operations should not be equal")
    }

    @Test("Memory usage is positive")
    func memoryUsageIsPositive() {
        let state = ExtendedStabilizerState(qubits: 3, maxRank: 64)
        #expect(state.memoryUsage > 0, "Memory usage should be positive")
    }
}
