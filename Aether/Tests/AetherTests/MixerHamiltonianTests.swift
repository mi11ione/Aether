// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Test suite for QAOA mixer Hamiltonian construction.
/// Validates X mixer structure, coefficient values, qubit coverage, and
/// mathematical properties like Hermiticity and commutation.
@Suite("Mixer Hamiltonians")
struct MixerHamiltonianTests {
    @Test("Single-qubit X mixer has one term")
    func singleQubitXMixer() {
        let mixer = MixerHamiltonian.x(qubits: 1)

        #expect(mixer.terms.count == 1, "Single-qubit mixer should have exactly one term")

        let (coefficient, pauliString) = mixer.terms[0]
        #expect(abs(coefficient - 1.0) < 1e-10, "X mixer coefficient should be +1.0")
        #expect(pauliString.operators.count == 1, "Each term should have one Pauli operator")
        #expect(pauliString.operators[0].qubit == 0, "Single-qubit mixer should target qubit 0")
        #expect(pauliString.operators[0].basis == .x, "Mixer operator should be X basis")
    }

    @Test("Multi-qubit X mixer has one X per qubit")
    func multiQubitXMixer() {
        let mixer = MixerHamiltonian.x(qubits: 5)

        #expect(mixer.terms.count == 5, "5-qubit mixer should have 5 terms")

        var observedQubits: Set<Int> = []
        for (coefficient, pauliString) in mixer.terms {
            #expect(abs(coefficient - 1.0) < 1e-10, "Each coefficient should be +1.0")
            #expect(pauliString.operators.count == 1, "Each term should have one operator")
            #expect(pauliString.operators[0].basis == .x, "Each operator should be X basis")

            let qubit = pauliString.operators[0].qubit
            #expect(!observedQubits.contains(qubit), "Each qubit should appear exactly once")
            observedQubits.insert(qubit)
        }

        #expect(observedQubits == Set(0 ..< 5), "All qubits 0..<5 should be covered")
    }

    @Test("X mixer coefficients are all +1.0")
    func coefficientsAreOne() {
        let mixer = MixerHamiltonian.x(qubits: 10)

        let coefficients = mixer.terms.map(\.0)
        #expect(coefficients.allSatisfy { abs($0 - 1.0) < 1e-10 }, "All coefficients should be +1.0")
    }

    @Test("All qubits have X operators")
    func allQubitsHaveX() {
        let mixer = MixerHamiltonian.x(qubits: 10)

        let qubits = mixer.terms.map { $0.1.operators[0].qubit }.sorted()
        #expect(qubits == Array(0 ..< 10), "Qubits should be 0..<10 in order")

        let bases = mixer.terms.map { $0.1.operators[0].basis }
        #expect(bases.allSatisfy { $0 == .x }, "All operators should be X basis")
    }

    @Test("Maximum practical qubit count")
    func maximumQubits() {
        let mixer = MixerHamiltonian.x(qubits: 30)

        #expect(mixer.terms.count == 30, "30-qubit mixer should have 30 terms")

        let qubits = Set(mixer.terms.map { $0.1.operators[0].qubit })
        #expect(qubits.count == 30, "Should have 30 unique qubits")
        #expect(qubits == Set(0 ..< 30), "All qubits 0..<30 should be covered")
    }
}

/// Test suite for mixer Hamiltonian mathematical properties.
/// Verifies Hermiticity, coefficient reality, and interaction with cost
/// Hamiltonians through non-commutation.
@Suite("Mixer Mathematical Properties")
struct MixerMathematicalPropertiesTests {
    @Test("X mixer is Hermitian")
    func xMixerHermitian() {
        let mixer = MixerHamiltonian.x(qubits: 4)

        for (coefficient, pauliString) in mixer.terms {
            #expect(coefficient.isFinite, "Coefficient should be finite")
            #expect(!coefficient.isNaN, "Coefficient should not be NaN")

            for op in pauliString.operators {
                #expect(op.basis == .x, "Hermitian X mixer should only use X operators")
            }
        }
    }

    @Test("X mixer uses X basis exclusively")
    func exclusivelyXBasis() {
        let mixer = MixerHamiltonian.x(qubits: 5)
        let cost = MaxCut.hamiltonian(edges: [(0, 1), (1, 2)])

        let mixerBases = mixer.terms.flatMap { $0.1.operators.map(\.basis) }
        let costBases = cost.terms.flatMap { $0.1.operators.map(\.basis) }

        #expect(mixerBases.allSatisfy { $0 == .x }, "Mixer should use only X basis operators")
        #expect(costBases.allSatisfy { $0 == .z }, "Cost Hamiltonian should use only Z basis operators")
    }

    @Test("Coefficient sum equals qubit count")
    func coefficientSum() {
        let mixer = MixerHamiltonian.x(qubits: 7)

        let sum = mixer.terms.reduce(0.0) { $0 + $1.0 }
        #expect(abs(sum - 7.0) < 1e-10, "Sum of 7 unit coefficients should equal 7.0")
    }

    @Test("Each qubit appears exactly once")
    func uniqueQubits() {
        let mixer = MixerHamiltonian.x(qubits: 8)

        let qubits = mixer.terms.map { $0.1.operators[0].qubit }
        #expect(qubits.count == Set(qubits).count, "Each qubit should appear exactly once")
    }

    @Test("No combined terms in Observable")
    func noCombinedTerms() {
        let mixer = MixerHamiltonian.x(qubits: 10)

        #expect(mixer.terms.count == 10, "10-qubit mixer should have 10 terms")

        for (_, pauliString) in mixer.terms {
            #expect(pauliString.operators.count == 1, "Each term should be a single-qubit operator")
        }
    }
}
