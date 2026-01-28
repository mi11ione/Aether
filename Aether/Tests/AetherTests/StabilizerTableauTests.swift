// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for StabilizerTableau initialization.
/// Validates that qubits property is set correctly and
/// the initial state represents |00...0⟩ computational basis.
@Suite("StabilizerTableau Initialization")
struct StabilizerTableauInitTests {
    @Test("init(qubits:) sets qubits property correctly")
    func initSetsQubits() {
        let tableau = StabilizerTableau(qubits: 3)
        #expect(tableau.qubits == 3, "Tableau should have 3 qubits")
    }

    @Test("init(qubits:) creates valid stabilizer state")
    func initCreatesValidState() {
        let tableau = StabilizerTableau(qubits: 2)
        #expect(tableau.isStabilizerState, "Initial tableau should be a valid stabilizer state")
    }

    @Test("init(qubits:) gives |0⟩ state amplitude 1 for basis state 0")
    func initGivesZeroStateAmplitude() {
        let tableau = StabilizerTableau(qubits: 2)
        let amp = tableau.amplitude(of: 0)
        #expect(amp != nil, "Amplitude should be computable for small qubit count")
        #expect(abs(amp!.magnitude - 1.0) < 1e-10, "Amplitude of |00⟩ should be 1.0")
    }

    @Test("init(qubits:) gives zero amplitude for non-zero basis states")
    func initGivesZeroAmplitudeForOtherStates() {
        let tableau = StabilizerTableau(qubits: 2)
        let amp1 = tableau.amplitude(of: 1)
        let amp2 = tableau.amplitude(of: 2)
        let amp3 = tableau.amplitude(of: 3)
        #expect(amp1 != nil && abs(amp1!.magnitude) < 1e-10, "Amplitude of |01⟩ should be 0")
        #expect(amp2 != nil && abs(amp2!.magnitude) < 1e-10, "Amplitude of |10⟩ should be 0")
        #expect(amp3 != nil && abs(amp3!.magnitude) < 1e-10, "Amplitude of |11⟩ should be 0")
    }

    @Test("memoryUsage is positive for any tableau")
    func memoryUsageIsPositive() {
        let tableau = StabilizerTableau(qubits: 4)
        #expect(tableau.memoryUsage > 0, "Memory usage should be positive")
    }

    @Test("description contains qubit count")
    func descriptionContainsQubitCount() {
        let tableau = StabilizerTableau(qubits: 3)
        #expect(tableau.description.contains("3"), "Description should contain qubit count")
    }
}

/// Test suite for single-qubit Clifford gates.
/// Validates Hadamard, S, and Pauli gates transform states correctly
/// according to stabilizer formalism transformation rules.
@Suite("Single-Qubit Clifford Gates")
struct StabilizerSingleQubitGateTests {
    @Test("Hadamard creates equal superposition")
    func hadamardCreatesSuperposition() {
        var tableau = StabilizerTableau(qubits: 1)
        tableau.apply(.hadamard, to: 0)
        let (p0, p1) = tableau.probability(of: 0, measuring: .z)
        #expect(abs(p0 - 0.5) < 1e-10, "Probability of |0⟩ should be 0.5 after Hadamard")
        #expect(abs(p1 - 0.5) < 1e-10, "Probability of |1⟩ should be 0.5 after Hadamard")
    }

    @Test("Hadamard applied twice returns to |0⟩")
    func hadamardTwiceReturnsToZero() {
        var tableau = StabilizerTableau(qubits: 1)
        tableau.apply(.hadamard, to: 0)
        tableau.apply(.hadamard, to: 0)
        let (p0, p1) = tableau.probability(of: 0, measuring: .z)
        #expect(abs(p0 - 1.0) < 1e-10, "After H^2, probability of |0⟩ should be 1.0")
        #expect(abs(p1) < 1e-10, "After H^2, probability of |1⟩ should be 0.0")
    }

    @Test("Pauli X flips |0⟩ to |1⟩")
    func pauliXFlipsState() {
        var tableau = StabilizerTableau(qubits: 1)
        tableau.apply(.pauliX, to: 0)
        let (p0, p1) = tableau.probability(of: 0, measuring: .z)
        #expect(abs(p0) < 1e-10, "After X, probability of |0⟩ should be 0.0")
        #expect(abs(p1 - 1.0) < 1e-10, "After X, probability of |1⟩ should be 1.0")
    }

    @Test("Pauli Z leaves |0⟩ unchanged")
    func pauliZLeavesZeroUnchanged() {
        var tableau = StabilizerTableau(qubits: 1)
        tableau.apply(.pauliZ, to: 0)
        let (p0, p1) = tableau.probability(of: 0, measuring: .z)
        #expect(abs(p0 - 1.0) < 1e-10, "After Z on |0⟩, probability of |0⟩ should be 1.0")
        #expect(abs(p1) < 1e-10, "After Z on |0⟩, probability of |1⟩ should be 0.0")
    }

    @Test("Pauli Y flips |0⟩ to |1⟩")
    func pauliYFlipsState() {
        var tableau = StabilizerTableau(qubits: 1)
        tableau.apply(.pauliY, to: 0)
        let (p0, p1) = tableau.probability(of: 0, measuring: .z)
        #expect(abs(p0) < 1e-10, "After Y, probability of |0⟩ should be 0.0")
        #expect(abs(p1 - 1.0) < 1e-10, "After Y, probability of |1⟩ should be 1.0")
    }

    @Test("S gate applied 4 times is identity")
    func sFourTimesIsIdentity() {
        var tableau = StabilizerTableau(qubits: 1)
        tableau.apply(.hadamard, to: 0)
        let originalTableau = tableau
        for _ in 0 ..< 4 {
            tableau.apply(.sGate, to: 0)
        }
        #expect(tableau == originalTableau, "S^4 should be identity")
    }

    @Test("Identity gate leaves state unchanged")
    func identityLeavesStateUnchanged() {
        var tableau = StabilizerTableau(qubits: 2)
        tableau.apply(.hadamard, to: 0)
        let before = tableau
        tableau.apply(.identity, to: 0)
        #expect(tableau == before, "Identity gate should not change the state")
    }

    @Test("SX gate transforms |0⟩ correctly")
    func sxGateTransforms() {
        var tableau = StabilizerTableau(qubits: 1)
        tableau.apply(.sx, to: 0)
        let (p0, p1) = tableau.probability(of: 0, measuring: .z)
        #expect(abs(p0 - 0.5) < 1e-10, "SX should create superposition with p0 = 0.5")
        #expect(abs(p1 - 0.5) < 1e-10, "SX should create superposition with p1 = 0.5")
    }

    @Test("SY gate transforms |0⟩ correctly")
    func syGateTransforms() {
        var tableau = StabilizerTableau(qubits: 1)
        tableau.apply(.sy, to: 0)
        let (p0, p1) = tableau.probability(of: 0, measuring: .z)
        #expect(abs(p0 - 0.5) < 1e-10, "SY should create superposition with p0 = 0.5")
        #expect(abs(p1 - 0.5) < 1e-10, "SY should create superposition with p1 = 0.5")
    }
}

/// Test suite for two-qubit Clifford gates.
/// Validates CNOT, CZ, and SWAP gates produce correct entanglement
/// and correlations in stabilizer states.
@Suite("Two-Qubit Clifford Gates")
struct StabilizerTwoQubitGateTests {
    @Test("CNOT creates Bell state from |+0⟩")
    func cnotCreatesBellState() {
        var tableau = StabilizerTableau(qubits: 2)
        tableau.apply(.hadamard, to: 0)
        tableau.apply(.cnot, to: [0, 1])
        let amp00 = tableau.amplitude(of: 0)
        let amp11 = tableau.amplitude(of: 3)
        let amp01 = tableau.amplitude(of: 1)
        let amp10 = tableau.amplitude(of: 2)
        #expect(amp00 != nil && abs(amp00!.magnitude - 1.0 / sqrt(2.0)) < 1e-10, "Bell state should have |00⟩ amplitude 1/sqrt(2)")
        #expect(amp11 != nil && abs(amp11!.magnitude - 1.0 / sqrt(2.0)) < 1e-10, "Bell state should have |11⟩ amplitude 1/sqrt(2)")
        #expect(amp01 != nil && abs(amp01!.magnitude) < 1e-10, "Bell state should have zero |01⟩ amplitude")
        #expect(amp10 != nil && abs(amp10!.magnitude) < 1e-10, "Bell state should have zero |10⟩ amplitude")
    }

    @Test("CZ creates entanglement from |++⟩")
    func czCreatesEntanglement() {
        var tableau = StabilizerTableau(qubits: 2)
        tableau.apply(.hadamard, to: 0)
        tableau.apply(.hadamard, to: 1)
        tableau.apply(.cz, to: [0, 1])
        let zz = PauliString(.z(0), .z(1))
        let expectation = tableau.expectationValue(of: zz)
        #expect(abs(expectation) < 1e-10, "⟨ZZ⟩ for CZ|++⟩ should be 0")
    }

    @Test("SWAP exchanges qubit states")
    func swapExchangesQubits() {
        var tableau = StabilizerTableau(qubits: 2)
        tableau.apply(.pauliX, to: 0)
        tableau.apply(.swap, to: [0, 1])
        let (p0_q0, p1_q0) = tableau.probability(of: 0, measuring: .z)
        let (p0_q1, p1_q1) = tableau.probability(of: 1, measuring: .z)
        #expect(abs(p0_q0 - 1.0) < 1e-10, "After SWAP, qubit 0 should be |0⟩")
        #expect(abs(p1_q0) < 1e-10, "After SWAP, qubit 0 should be |0⟩")
        #expect(abs(p0_q1) < 1e-10, "After SWAP, qubit 1 should be |1⟩")
        #expect(abs(p1_q1 - 1.0) < 1e-10, "After SWAP, qubit 1 should be |1⟩")
    }

    @Test("CNOT applied twice is identity")
    func cnotTwiceIsIdentity() {
        var tableau = StabilizerTableau(qubits: 2)
        tableau.apply(.hadamard, to: 0)
        let before = tableau
        tableau.apply(.cnot, to: [0, 1])
        tableau.apply(.cnot, to: [0, 1])
        #expect(tableau == before, "CNOT^2 should be identity")
    }
}

/// Test suite for measurement operations.
/// Validates deterministic and probabilistic measurement outcomes
/// with proper state collapse behavior.
@Suite("Measurement Operations")
struct StabilizerMeasurementTests {
    @Test("Deterministic measurement of |0⟩ always gives 0")
    func deterministicMeasurementZero() {
        var tableau = StabilizerTableau(qubits: 1)
        let outcome = tableau.measure(0, seed: 42)
        #expect(outcome == 0, "Measuring |0⟩ should always give outcome 0")
    }

    @Test("Deterministic measurement of |1⟩ always gives 1")
    func deterministicMeasurementOne() {
        var tableau = StabilizerTableau(qubits: 1)
        tableau.apply(.pauliX, to: 0)
        let outcome = tableau.measure(0, seed: 42)
        #expect(outcome == 1, "Measuring |1⟩ should always give outcome 1")
    }

    @Test("Measurement with seed is reproducible")
    func measurementWithSeedReproducible() {
        var tableau1 = StabilizerTableau(qubits: 1)
        tableau1.apply(.hadamard, to: 0)
        let outcome1 = tableau1.measure(0, seed: 12345)

        var tableau2 = StabilizerTableau(qubits: 1)
        tableau2.apply(.hadamard, to: 0)
        let outcome2 = tableau2.measure(0, seed: 12345)

        #expect(outcome1 == outcome2, "Same seed should produce same measurement outcome")
    }

    @Test("Measurement collapses superposition to definite state")
    func measurementCollapsesSuperposition() {
        var tableau = StabilizerTableau(qubits: 1)
        tableau.apply(.hadamard, to: 0)
        let outcome = tableau.measure(0, seed: 42)
        let (p0, p1) = tableau.probability(of: 0, measuring: .z)
        if outcome == 0 {
            #expect(abs(p0 - 1.0) < 1e-10, "After measuring 0, p(0) should be 1.0")
            #expect(abs(p1) < 1e-10, "After measuring 0, p(1) should be 0.0")
        } else {
            #expect(abs(p0) < 1e-10, "After measuring 1, p(0) should be 0.0")
            #expect(abs(p1 - 1.0) < 1e-10, "After measuring 1, p(1) should be 1.0")
        }
    }

    @Test("Bell state measurement correlates qubits")
    func bellStateMeasurementCorrelation() {
        var tableau = StabilizerTableau(qubits: 2)
        tableau.apply(.hadamard, to: 0)
        tableau.apply(.cnot, to: [0, 1])
        let outcome0 = tableau.measure(0, seed: 42)
        let outcome1 = tableau.measure(1, seed: 42)
        #expect(outcome0 == outcome1, "Bell state measurement should give correlated outcomes")
    }
}

/// Test suite for probability calculations.
/// Validates probability queries in different Pauli bases
/// for various stabilizer states.
@Suite("Probability Queries")
struct StabilizerProbabilityTests {
    @Test("X-basis probability for |0⟩ is 50/50")
    func xBasisProbabilityForZero() {
        let tableau = StabilizerTableau(qubits: 1)
        let (p0, p1) = tableau.probability(of: 0, measuring: .x)
        #expect(abs(p0 - 0.5) < 1e-10, "X-basis p(+) for |0⟩ should be 0.5")
        #expect(abs(p1 - 0.5) < 1e-10, "X-basis p(-) for |0⟩ should be 0.5")
    }

    @Test("Y-basis probability for |0⟩ is 50/50")
    func yBasisProbabilityForZero() {
        let tableau = StabilizerTableau(qubits: 1)
        let (p0, p1) = tableau.probability(of: 0, measuring: .y)
        #expect(abs(p0 - 0.5) < 1e-10, "Y-basis p(+i) for |0⟩ should be 0.5")
        #expect(abs(p1 - 0.5) < 1e-10, "Y-basis p(-i) for |0⟩ should be 0.5")
    }

    @Test("Z-basis probability for |+⟩ is 50/50")
    func zBasisProbabilityForPlus() {
        var tableau = StabilizerTableau(qubits: 1)
        tableau.apply(.hadamard, to: 0)
        let (p0, p1) = tableau.probability(of: 0, measuring: .z)
        #expect(abs(p0 - 0.5) < 1e-10, "Z-basis p(0) for |+⟩ should be 0.5")
        #expect(abs(p1 - 0.5) < 1e-10, "Z-basis p(1) for |+⟩ should be 0.5")
    }

    @Test("X-basis probability for |+⟩ is deterministic")
    func xBasisProbabilityForPlus() {
        var tableau = StabilizerTableau(qubits: 1)
        tableau.apply(.hadamard, to: 0)
        let (p0, p1) = tableau.probability(of: 0, measuring: .x)
        #expect(abs(p0 - 1.0) < 1e-10, "X-basis p(+) for |+⟩ should be 1.0")
        #expect(abs(p1) < 1e-10, "X-basis p(-) for |+⟩ should be 0.0")
    }
}

/// Test suite for amplitude computation.
/// Validates amplitude and probability calculations for
/// various stabilizer states with small qubit counts.
@Suite("Amplitude Computation")
struct StabilizerAmplitudeTests {
    @Test("GHZ state has correct amplitudes")
    func ghzStateAmplitudes() {
        var tableau = StabilizerTableau(qubits: 3)
        tableau.apply(.hadamard, to: 0)
        tableau.apply(.cnot, to: [0, 1])
        tableau.apply(.cnot, to: [1, 2])
        let amp000 = tableau.amplitude(of: 0)
        let amp111 = tableau.amplitude(of: 7)
        let amp001 = tableau.amplitude(of: 1)
        #expect(amp000 != nil && abs(amp000!.magnitude - 1.0 / sqrt(2.0)) < 1e-10, "GHZ |000⟩ amplitude should be 1/sqrt(2)")
        #expect(amp111 != nil && abs(amp111!.magnitude - 1.0 / sqrt(2.0)) < 1e-10, "GHZ |111⟩ amplitude should be 1/sqrt(2)")
        #expect(amp001 != nil && abs(amp001!.magnitude) < 1e-10, "GHZ |001⟩ amplitude should be 0")
    }

    @Test("Amplitude returns nil for too many qubits")
    func amplitudeNilForManyQubits() {
        let tableau = StabilizerTableau(qubits: 25)
        let amp = tableau.amplitude(of: 0)
        #expect(amp == nil, "Amplitude should return nil for > 20 qubits")
    }

    @Test("Amplitude of invalid basis state returns nil")
    func amplitudeNilForInvalidBasisState() {
        let tableau = StabilizerTableau(qubits: 2)
        let amp = tableau.amplitude(of: 10)
        #expect(amp == nil, "Amplitude should return nil for basis state >= 2^n")
    }
}

/// Test suite for sample function.
/// Validates multi-shot sampling produces distributions
/// consistent with theoretical probabilities.
@Suite("Sampling Operations")
struct StabilizerSamplingTests {
    @Test("Sample from |0⟩ always returns 0")
    func sampleFromZeroState() {
        var tableau = StabilizerTableau(qubits: 2)
        let samples = tableau.sample(shots: 10, seed: 42)
        let allZero = samples.allSatisfy { $0 == 0 }
        #expect(allZero, "Sampling |00⟩ should always give outcome 0")
    }

    @Test("Sample from Bell state gives only 0 or 3")
    func sampleFromBellState() {
        var tableau = StabilizerTableau(qubits: 2)
        tableau.apply(.hadamard, to: 0)
        tableau.apply(.cnot, to: [0, 1])
        let samples = tableau.sample(shots: 20, seed: 42)
        let validOutcomes = samples.allSatisfy { $0 == 0 || $0 == 3 }
        #expect(validOutcomes, "Bell state samples should only be 0 (|00⟩) or 3 (|11⟩)")
    }

    @Test("Sample with seed is reproducible")
    func sampleWithSeedReproducible() {
        var tableau1 = StabilizerTableau(qubits: 2)
        tableau1.apply(.hadamard, to: 0)
        tableau1.apply(.cnot, to: [0, 1])
        let samples1 = tableau1.sample(shots: 10, seed: 99999)

        var tableau2 = StabilizerTableau(qubits: 2)
        tableau2.apply(.hadamard, to: 0)
        tableau2.apply(.cnot, to: [0, 1])
        let samples2 = tableau2.sample(shots: 10, seed: 99999)

        #expect(samples1 == samples2, "Same seed should produce identical sample sequences")
    }

    @Test("Sample returns correct number of shots")
    func sampleReturnsCorrectCount() {
        var tableau = StabilizerTableau(qubits: 2)
        tableau.apply(.hadamard, to: 0)
        let samples = tableau.sample(shots: 15, seed: 42)
        #expect(samples.count == 15, "Sample should return exactly the requested number of shots")
    }
}

/// Test suite for expectation value calculations.
/// Validates Pauli string expectation values match
/// theoretical predictions for stabilizer states.
@Suite("Expectation Values")
struct StabilizerExpectationValueTests {
    @Test("⟨Z⟩ = +1 for |0⟩")
    func expectationZForZero() {
        let tableau = StabilizerTableau(qubits: 1)
        let z = PauliString(.z(0))
        let expectation = tableau.expectationValue(of: z)
        #expect(abs(expectation - 1.0) < 1e-10, "⟨Z⟩ for |0⟩ should be +1")
    }

    @Test("⟨Z⟩ = -1 for |1⟩")
    func expectationZForOne() {
        var tableau = StabilizerTableau(qubits: 1)
        tableau.apply(.pauliX, to: 0)
        let z = PauliString(.z(0))
        let expectation = tableau.expectationValue(of: z)
        #expect(abs(expectation + 1.0) < 1e-10, "⟨Z⟩ for |1⟩ should be -1")
    }

    @Test("⟨X⟩ = 0 for |0⟩")
    func expectationXForZero() {
        let tableau = StabilizerTableau(qubits: 1)
        let x = PauliString(.x(0))
        let expectation = tableau.expectationValue(of: x)
        #expect(abs(expectation) < 1e-10, "⟨X⟩ for |0⟩ should be 0")
    }

    @Test("⟨X⟩ = +1 for |+⟩")
    func expectationXForPlus() {
        var tableau = StabilizerTableau(qubits: 1)
        tableau.apply(.hadamard, to: 0)
        let x = PauliString(.x(0))
        let expectation = tableau.expectationValue(of: x)
        #expect(abs(expectation - 1.0) < 1e-10, "⟨X⟩ for |+⟩ should be +1")
    }

    @Test("⟨ZZ⟩ = +1 for Bell state")
    func expectationZZForBell() {
        var tableau = StabilizerTableau(qubits: 2)
        tableau.apply(.hadamard, to: 0)
        tableau.apply(.cnot, to: [0, 1])
        let zz = PauliString(.z(0), .z(1))
        let expectation = tableau.expectationValue(of: zz)
        #expect(abs(expectation - 1.0) < 1e-10, "⟨ZZ⟩ for Bell state should be +1")
    }

    @Test("⟨XX⟩ = +1 for Bell state")
    func expectationXXForBell() {
        var tableau = StabilizerTableau(qubits: 2)
        tableau.apply(.hadamard, to: 0)
        tableau.apply(.cnot, to: [0, 1])
        let xx = PauliString(.x(0), .x(1))
        let expectation = tableau.expectationValue(of: xx)
        #expect(abs(expectation - 1.0) < 1e-10, "⟨XX⟩ for Bell state should be +1")
    }

    @Test("Empty Pauli string gives expectation 1")
    func emptyPauliStringExpectation() {
        let tableau = StabilizerTableau(qubits: 2)
        let identity = PauliString([])
        let expectation = tableau.expectationValue(of: identity)
        #expect(abs(expectation - 1.0) < 1e-10, "⟨I⟩ should always be +1")
    }

    @Test("⟨YY⟩ = -1 for Bell state Φ+")
    func expectationYYForBell() {
        var tableau = StabilizerTableau(qubits: 2)
        tableau.apply(.hadamard, to: 0)
        tableau.apply(.cnot, to: [0, 1])
        let yy = PauliString(.y(0), .y(1))
        let expectation = tableau.expectationValue(of: yy)
        #expect(abs(expectation + 1.0) < 1e-10, "⟨YY⟩ for Bell state Φ+ should be -1")
    }
}

/// Test suite for Equatable conformance.
/// Validates that identical tableaux are equal and
/// different tableaux are not equal.
@Suite("Equatable Conformance")
struct StabilizerEquatableTests {
    @Test("Identical tableaux are equal")
    func identicalTableauxEqual() {
        let tableau1 = StabilizerTableau(qubits: 2)
        let tableau2 = StabilizerTableau(qubits: 2)
        #expect(tableau1 == tableau2, "Freshly initialized tableaux should be equal")
    }

    @Test("Different states are not equal")
    func differentStatesNotEqual() {
        let tableau1 = StabilizerTableau(qubits: 2)
        var tableau2 = StabilizerTableau(qubits: 2)
        tableau2.apply(.hadamard, to: 0)
        #expect(tableau1 != tableau2, "States differing by Hadamard should not be equal")
    }

    @Test("Same operations yield equal tableaux")
    func sameOperationsYieldEqual() {
        var tableau1 = StabilizerTableau(qubits: 2)
        tableau1.apply(.hadamard, to: 0)
        tableau1.apply(.cnot, to: [0, 1])

        var tableau2 = StabilizerTableau(qubits: 2)
        tableau2.apply(.hadamard, to: 0)
        tableau2.apply(.cnot, to: [0, 1])

        #expect(tableau1 == tableau2, "Same gate sequence should produce equal tableaux")
    }
}
