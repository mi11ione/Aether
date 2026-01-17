// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Tests for preparing specific quantum states and validating their properties.
/// Covers basis states via direct construction and circuit synthesis, ensuring
/// correct amplitudes, normalization, and gate application counts.
@Suite("State Preparation")
struct StatePreparationTests {
    @Test("Prepare basis state |0⟩ for single qubit")
    func basisStateSingleQubitZero() {
        let state = QuantumState.basis(qubits: 1, state: 0)

        #expect(state.qubits == 1)
        #expect(state.amplitudes.count == 2)
        #expect(state.amplitude(of: 0) == .one)
        #expect(state.amplitude(of: 1) == .zero)
        #expect(state.isNormalized())
    }

    @Test("Prepare basis state |1⟩ for single qubit")
    func basisStateSingleQubitOne() {
        let state = QuantumState.basis(qubits: 1, state: 1)

        #expect(state.qubits == 1)
        #expect(state.amplitudes.count == 2)
        #expect(state.amplitude(of: 0) == .zero)
        #expect(state.amplitude(of: 1) == .one)
        #expect(state.isNormalized())
    }

    @Test("Prepare basis state |101⟩ (index 5)")
    func basisStateThreeQubits() {
        let state = QuantumState.basis(qubits: 3, state: 5)

        #expect(state.qubits == 3)
        #expect(state.amplitudes.count == 8)

        for i in 0 ..< 8 {
            if i == 5 {
                #expect(state.amplitude(of: i) == .one)
            } else {
                #expect(state.amplitude(of: i) == .zero)
            }
        }
        #expect(state.isNormalized())
    }

    @Test("Prepare maximum basis state for n qubits")
    func basisStateMaximum() {
        let qubits = 4
        let maxIndex = (1 << qubits) - 1

        let state = QuantumState.basis(qubits: qubits, state: maxIndex)

        #expect(state.amplitude(of: maxIndex) == .one)
        #expect(state.isNormalized())

        for i in 0 ..< maxIndex {
            #expect(state.amplitude(of: i) == .zero)
        }
    }

    @Test("Basis state circuit creates correct state")
    func basisStateCircuitCorrectness() {
        let circuit = QuantumCircuit.basis(qubits: 4, state: 11)
        let finalState = circuit.execute()

        #expect(finalState.amplitude(of: 11).magnitude > 0.99)
        #expect(finalState.isNormalized())

        let prob = finalState.probability(of: 11)
        #expect(abs(prob - 1.0) < 1e-10, "State |1011⟩ should have probability 1.0")
    }
}

/// Tests for all four Bell states prepared via helper circuits.
/// Verifies normalization, expected probabilities for basis outcomes,
/// and relative phases distinguishing the ± variants.
@Suite("Bell State Variants")
struct BellStateVariantsTests {
    @Test("Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2")
    func bellPhiPlus() {
        let circuit = QuantumCircuit.bellPhiPlus()
        let state = circuit.execute()

        #expect(state.qubits == 2)
        #expect(state.isNormalized())

        let prob00 = state.probability(of: 0)
        let prob11 = state.probability(of: 3)

        #expect(abs(prob00 - 0.5) < 1e-10, "|Φ⁺⟩ should have 50% |00⟩")
        #expect(abs(prob11 - 0.5) < 1e-10, "|Φ⁺⟩ should have 50% |11⟩")

        #expect(state.probability(of: 1) < 1e-10)
        #expect(state.probability(of: 2) < 1e-10)
    }

    @Test("Bell state |Φ⁻⟩ = (|00⟩ - |11⟩)/√2")
    func bellPhiMinus() {
        let circuit = QuantumCircuit.bellPhiMinus()
        let state = circuit.execute()

        #expect(state.qubits == 2)
        #expect(state.isNormalized())

        let prob00 = state.probability(of: 0)
        let prob11 = state.probability(of: 3)

        #expect(abs(prob00 - 0.5) < 1e-10, "|Φ⁻⟩ should have 50% |00⟩")
        #expect(abs(prob11 - 0.5) < 1e-10, "|Φ⁻⟩ should have 50% |11⟩")

        let amp00 = state.amplitude(of: 0)
        let amp11 = state.amplitude(of: 3)

        #expect(amp00.real * amp11.real < 0, "Amplitudes should have opposite signs for |Φ⁻⟩")
    }

    @Test("Bell state |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2")
    func bellPsiPlus() {
        let circuit = QuantumCircuit.bellPsiPlus()
        let state = circuit.execute()

        #expect(state.qubits == 2)
        #expect(state.isNormalized())

        let prob01 = state.probability(of: 1)
        let prob10 = state.probability(of: 2)

        #expect(abs(prob01 - 0.5) < 1e-10, "|Ψ⁺⟩ should have 50% |01⟩")
        #expect(abs(prob10 - 0.5) < 1e-10, "|Ψ⁺⟩ should have 50% |10⟩")

        #expect(state.probability(of: 0) < 1e-10)
        #expect(state.probability(of: 3) < 1e-10)
    }

    @Test("Bell state |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2")
    func bellPsiMinus() {
        let circuit = QuantumCircuit.bellPsiMinus()
        let state = circuit.execute()

        #expect(state.qubits == 2)
        #expect(state.isNormalized())

        let prob01 = state.probability(of: 1)
        let prob10 = state.probability(of: 2)

        #expect(abs(prob01 - 0.5) < 1e-10, "|Ψ⁻⟩ should have 50% |01⟩")
        #expect(abs(prob10 - 0.5) < 1e-10, "|Ψ⁻⟩ should have 50% |10⟩")

        let amp01 = state.amplitude(of: 1)
        let amp10 = state.amplitude(of: 2)

        #expect(amp01.real * amp10.real < 0, "Amplitudes should have opposite signs for |Ψ⁻⟩")
    }

    @Test("All four Bell states are normalized")
    func allBellStatesNormalized() {
        let circuits = [
            QuantumCircuit.bellPhiPlus(),
            QuantumCircuit.bellPhiMinus(),
            QuantumCircuit.bellPsiPlus(),
            QuantumCircuit.bellPsiMinus(),
        ]

        for circuit in circuits {
            let state = circuit.execute()
            #expect(state.isNormalized(), "All Bell states must be normalized")
        }
    }
}

/// Tests for W-state preparation across multiple qubit counts.
/// Ensures single-excitation support, equal-amplitude distribution,
/// and total probability conservation under execution.
@Suite("W State")
struct WStateTests {
    @Test("W state for 2 qubits: |W₂⟩ = (|01⟩ + |10⟩)/√2")
    func wStateTwoQubits() {
        let state = QuantumState.w(qubits: 2)

        #expect(state.qubits == 2)
        #expect(state.isNormalized())

        let prob01 = state.probability(of: 1)
        let prob10 = state.probability(of: 2)

        #expect(abs(prob01 - 0.5) < 1e-6, "|W₂⟩ should have ~50% |01⟩")
        #expect(abs(prob10 - 0.5) < 1e-6, "|W₂⟩ should have ~50% |10⟩")

        #expect(state.probability(of: 0) < 1e-6)
        #expect(state.probability(of: 3) < 1e-6)
    }

    @Test("W state for 3 qubits: |W₃⟩ = (|100⟩ + |010⟩ + |001⟩)/√3")
    func wStateThreeQubits() {
        let state = QuantumState.w(qubits: 3)

        #expect(state.qubits == 3)
        #expect(state.isNormalized())

        let expectedProb = 1.0 / 3.0

        let prob001 = state.probability(of: 1)
        let prob010 = state.probability(of: 2)
        let prob100 = state.probability(of: 4)

        #expect(abs(prob001 - expectedProb) < 1e-4, "|001⟩ should have probability ~1/3")
        #expect(abs(prob010 - expectedProb) < 1e-4, "|010⟩ should have probability ~1/3")
        #expect(abs(prob100 - expectedProb) < 1e-4, "|100⟩ should have probability ~1/3")

        for i in [0, 3, 5, 6, 7] {
            #expect(state.probability(of: i) < 1e-4, "State |\(i)⟩ should be near zero")
        }
    }

    @Test("W state for 4 qubits has correct structure")
    func wStateFourQubits() {
        let state = QuantumState.w(qubits: 4)

        #expect(state.qubits == 4)
        #expect(state.isNormalized())

        var totalProbSingleExcitation = 0.0
        for i in 0 ..< 16 {
            if i.nonzeroBitCount == 1 {
                totalProbSingleExcitation += state.probability(of: i)
            }
        }

        #expect(abs(totalProbSingleExcitation - 1.0) < 1e-4,
                "W state probability should be concentrated in single-excitation states")
    }

    @Test("W state probabilities sum to 1")
    func wStateProbabilitiesSumToOne() {
        for n in 2 ... 6 {
            let state = QuantumState.w(qubits: n)

            let totalProb = state.probabilities().reduce(0.0, +)
            #expect(abs(totalProb - 1.0) < 1e-10, "Probabilities must sum to 1 for \(n) qubits")
        }
    }
}

/// Tests for Dicke-state preparation with varying Hamming weights.
/// Validates symmetry, equal-amplitude magnitudes across configurations,
/// edge cases (all-zeros/all-ones), and strict normalization.
@Suite("Dicke State")
struct DickeStateTests {
    @Test("Dicke state |D₀³⟩ = |000⟩")
    func dickeStateAllZeros() {
        let state = QuantumState.dicke(qubits: 3, ones: 0)

        #expect(state.isNormalized())
        #expect(state.amplitude(of: 0) == .one, "Only |000⟩ should have amplitude 1")

        for i in 1 ..< 8 {
            #expect(state.amplitude(of: i) == .zero)
        }
    }

    @Test("Dicke state |D₃³⟩ = |111⟩")
    func dickeStateAllOnes() {
        let state = QuantumState.dicke(qubits: 3, ones: 3)

        #expect(state.isNormalized())
        #expect(state.amplitude(of: 7) == .one, "Only |111⟩ should have amplitude 1")

        for i in 0 ..< 7 {
            #expect(state.amplitude(of: i) == .zero)
        }
    }

    @Test("Dicke state |D₁³⟩ = (|001⟩ + |010⟩ + |100⟩)/√3")
    func dickeStateOneOfThree() {
        let state = QuantumState.dicke(qubits: 3, ones: 1)

        #expect(state.isNormalized())

        let expectedProb = 1.0 / 3.0

        #expect(abs(state.probability(of: 1) - expectedProb) < 1e-10, "|001⟩ probability")
        #expect(abs(state.probability(of: 2) - expectedProb) < 1e-10, "|010⟩ probability")
        #expect(abs(state.probability(of: 4) - expectedProb) < 1e-10, "|100⟩ probability")

        for i in [0, 3, 5, 6, 7] {
            #expect(state.probability(of: i) < 1e-10, "State |\(i)⟩ should be zero")
        }
    }

    @Test("Dicke state |D₂⁴⟩ has 6 equal-amplitude states")
    func dickeTwoOfFour() {
        let state = QuantumState.dicke(qubits: 4, ones: 2)

        #expect(state.isNormalized())

        var count = 0
        var totalProb = 0.0

        for i in 0 ..< 16 {
            if i.nonzeroBitCount == 2 {
                count += 1
                totalProb += state.probability(of: i)
            }
        }

        #expect(count == 6, "Should have 6 states with Hamming weight 2")
        #expect(abs(totalProb - 1.0) < 1e-10, "Total probability should be 1")

        let expectedProb = 1.0 / 6.0
        for i in 0 ..< 16 {
            if i.nonzeroBitCount == 2 {
                #expect(abs(state.probability(of: i) - expectedProb) < 1e-10)
            }
        }
    }

    @Test("Dicke state amplitudes are equal")
    func dickeStateEqualAmplitudes() {
        let state = QuantumState.dicke(qubits: 5, ones: 2)

        var magnitudes: [Double] = []

        for i in 0 ..< 32 {
            if i.nonzeroBitCount == 2 {
                magnitudes.append(state.amplitude(of: i).magnitude)
            }
        }

        let firstMag = magnitudes[0]
        for mag in magnitudes {
            #expect(abs(mag - firstMag) < 1e-10, "All amplitudes should have same magnitude")
        }
    }

    @Test("Dicke state is symmetric under qubit permutation")
    func dickeStateSymmetry() {
        let state = QuantumState.dicke(qubits: 3, ones: 1)

        let prob1 = state.probability(of: 1)
        let prob2 = state.probability(of: 2)
        let prob4 = state.probability(of: 4)

        #expect(abs(prob1 - prob2) < 1e-10, "Symmetric under qubit permutation")
        #expect(abs(prob2 - prob4) < 1e-10, "Symmetric under qubit permutation")
    }
}

/// Additional validation of state preparation helpers and utilities.
/// Confirms uniform superposition correctness, auto-normalization of
/// arbitrary states, scalability, and qubit-count preservation.
@Suite("Edge Cases and Validation")
struct EdgeCasesAndValidationTests {
    @Test("Uniform superposition is correctly implemented")
    func uniformSuperpositionVerification() {
        let circuit = QuantumCircuit.uniformSuperposition(qubits: 3)
        let state = circuit.execute()

        #expect(state.qubits == 3)
        #expect(state.isNormalized())

        let expectedProb = 1.0 / 8.0
        for i in 0 ..< 8 {
            let prob = state.probability(of: i)
            #expect(abs(prob - expectedProb) < 1e-10,
                    "State |\(i)⟩ should have probability 1/8")
        }
    }

    @Test("Arbitrary state initialization auto-normalizes")
    func arbitraryStateAutoNormalization() {
        let unnormalized = [
            Complex(1.0, 0.0),
            Complex(1.0, 0.0),
            Complex(1.0, 0.0),
            Complex(1.0, 0.0),
        ]

        let state = QuantumState(qubits: 2, amplitudes: unnormalized)

        #expect(state.isNormalized(), "State should be auto-normalized")

        for i in 0 ..< 4 {
            #expect(abs(state.amplitude(of: i).magnitude - 0.5) < 1e-10)
        }
    }

    @Test("Large Dicke state is properly normalized")
    func largeDickeStateNormalization() {
        let state = QuantumState.dicke(qubits: 10, ones: 5)

        #expect(state.isNormalized(), "Large Dicke state must be normalized")

        for i in 0 ..< (1 << 10) {
            if i.nonzeroBitCount == 5 {
                #expect(state.probability(of: i) > 0, "Hamming-5 states should be non-zero")
            } else {
                #expect(state.probability(of: i) < 1e-10, "Other states should be zero")
            }
        }
    }

    @Test("State preparation methods preserve qubit count")
    func qubitCountPreservation() {
        let basisState = QuantumState.basis(qubits: 5, state: 10)
        #expect(basisState.qubits == 5)

        let dicke = QuantumState.dicke(qubits: 4, ones: 2)
        #expect(dicke.qubits == 4)

        let wState = QuantumState.w(qubits: 6)
        #expect(wState.qubits == 6)
    }
}
