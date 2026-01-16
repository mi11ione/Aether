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

        #expect(mixer.terms.count == 1)

        let (coefficient, pauliString) = mixer.terms[0]
        #expect(abs(coefficient - 1.0) < 1e-10)
        #expect(pauliString.operators.count == 1)
        #expect(pauliString.operators[0].qubit == 0)
        #expect(pauliString.operators[0].basis == .x)
    }

    @Test("Multi-qubit X mixer has one X per qubit")
    func multiQubitXMixer() {
        let mixer = MixerHamiltonian.x(qubits: 5)

        #expect(mixer.terms.count == 5)

        var observedQubits: Set<Int> = []
        for (coefficient, pauliString) in mixer.terms {
            #expect(abs(coefficient - 1.0) < 1e-10)
            #expect(pauliString.operators.count == 1)
            #expect(pauliString.operators[0].basis == .x)

            let qubit = pauliString.operators[0].qubit
            #expect(!observedQubits.contains(qubit))
            observedQubits.insert(qubit)
        }

        #expect(observedQubits == Set(0 ..< 5))
    }

    @Test("X mixer coefficients are all +1.0")
    func coefficientsAreOne() {
        let mixer = MixerHamiltonian.x(qubits: 10)

        let coefficients = mixer.terms.map(\.0)
        #expect(coefficients.allSatisfy { abs($0 - 1.0) < 1e-10 })
    }

    @Test("All qubits have X operators")
    func allQubitsHaveX() {
        let mixer = MixerHamiltonian.x(qubits: 10)

        let qubits = mixer.terms.map { $0.1.operators[0].qubit }.sorted()
        #expect(qubits == Array(0 ..< 10))

        let bases = mixer.terms.map { $0.1.operators[0].basis }
        #expect(bases.allSatisfy { $0 == .x })
    }

    @Test("Maximum practical qubit count")
    func maximumQubits() {
        let mixer = MixerHamiltonian.x(qubits: 30)

        #expect(mixer.terms.count == 30)

        let qubits = Set(mixer.terms.map { $0.1.operators[0].qubit })
        #expect(qubits.count == 30)
        #expect(qubits == Set(0 ..< 30))
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
            #expect(coefficient.isFinite)
            #expect(!coefficient.isNaN)

            for op in pauliString.operators {
                #expect(op.basis == .x)
            }
        }
    }

    @Test("X mixer uses X basis exclusively")
    func exclusivelyXBasis() {
        let mixer = MixerHamiltonian.x(qubits: 5)
        let cost = MaxCut.hamiltonian(edges: [(0, 1), (1, 2)])

        let mixerBases = mixer.terms.flatMap { $0.1.operators.map(\.basis) }
        let costBases = cost.terms.flatMap { $0.1.operators.map(\.basis) }

        #expect(mixerBases.allSatisfy { $0 == .x })
        #expect(costBases.allSatisfy { $0 == .z })
    }

    @Test("Coefficient sum equals qubit count")
    func coefficientSum() {
        let mixer = MixerHamiltonian.x(qubits: 7)

        let sum = mixer.terms.reduce(0.0) { $0 + $1.0 }
        #expect(abs(sum - 7.0) < 1e-10)
    }

    @Test("Each qubit appears exactly once")
    func uniqueQubits() {
        let mixer = MixerHamiltonian.x(qubits: 8)

        let qubits = mixer.terms.map { $0.1.operators[0].qubit }
        #expect(qubits.count == Set(qubits).count)
    }

    @Test("No combined terms in Observable")
    func noCombinedTerms() {
        let mixer = MixerHamiltonian.x(qubits: 10)

        #expect(mixer.terms.count == 10)

        for (_, pauliString) in mixer.terms {
            #expect(pauliString.operators.count == 1)
        }
    }
}
