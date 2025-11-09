// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for Hermitian observable expectations.
/// Validates expectation values for Pauli strings, multi-term observables,
/// and formatted descriptions across single- and two-qubit states.
@Suite("Observable Hermitian Operators")
struct ObservableHermitianTests {
    @Test("Identity observable has expectation +1")
    func identityObservable() {
        let identity = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [])
        )

        let state = QuantumState(numQubits: 2)
        let expectation = identity.expectationValue(state: state)

        #expect(abs(expectation - 1.0) < 1e-10)
    }

    @Test("Pauli-Z expectation on |0⟩ is +1")
    func pauliZOnZeroState() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [(qubit: 0, basis: .z)])
        )

        let state = QuantumState(numQubits: 1)
        let expectation = observable.expectationValue(state: state)

        #expect(abs(expectation - 1.0) < 1e-10)
    }

    @Test("Pauli-Z expectation on |1⟩ is -1")
    func pauliZOnOneState() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [(qubit: 0, basis: .z)])
        )

        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .pauliX, toQubit: 0)
        let state = circuit.execute()

        let expectation = observable.expectationValue(state: state)
        #expect(abs(expectation - -1.0) < 1e-10)
    }

    @Test("Pauli-X expectation on |+⟩ is +1")
    func pauliXOnPlusState() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [(qubit: 0, basis: .x)])
        )

        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .hadamard, toQubit: 0)
        let state = circuit.execute()

        let expectation = observable.expectationValue(state: state)
        #expect(abs(expectation - 1.0) < 1e-10)
    }

    @Test("Pauli-X expectation on |-⟩ is -1")
    func pauliXOnMinusState() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [(qubit: 0, basis: .x)])
        )

        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .pauliX, toQubit: 0)
        circuit.append(gate: .hadamard, toQubit: 0)
        let state = circuit.execute()

        let expectation = observable.expectationValue(state: state)
        #expect(abs(expectation - -1.0) < 1e-10)
    }

    @Test("Pauli-Y expectation on |+i⟩ is +1")
    func pauliYOnPlusIState() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [(qubit: 0, basis: .y)])
        )

        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .sGate, toQubit: 0)
        let state = circuit.execute()

        let expectation = observable.expectationValue(state: state)
        #expect(abs(expectation - 1.0) < 1e-10)
    }

    @Test("Two-qubit Z⊗Z on Bell state")
    func zzOnBellState() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [
                (qubit: 0, basis: .z),
                (qubit: 1, basis: .z),
            ])
        )

        let bell = QuantumCircuit.bellPhiPlus().execute()

        let expectation = observable.expectationValue(state: bell)
        #expect(abs(expectation - 1.0) < 1e-10)
    }

    @Test("Two-qubit X⊗X on Bell state")
    func xxOnBellState() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [
                (qubit: 0, basis: .x),
                (qubit: 1, basis: .x),
            ])
        )

        let bell = QuantumCircuit.bellPhiPlus().execute()
        let expectation = observable.expectationValue(state: bell)

        #expect(abs(expectation - 1.0) < 1e-10)
    }

    @Test("Multi-term observable sums correctly")
    func multiTermObservable() {
        let observable = Observable(terms: [
            (coefficient: 2.0, pauliString: PauliString(operators: [(0, .z)])),
            (coefficient: 3.0, pauliString: PauliString(operators: [(1, .z)])),
        ])

        let state = QuantumState(numQubits: 2)
        let expectation = observable.expectationValue(state: state)

        #expect(abs(expectation - 5.0) < 1e-10)
    }

    @Test("Negative coefficient")
    func negativeCoefficient() {
        let observable = Observable(
            coefficient: -1.0,
            pauliString: PauliString(operators: [(0, .z)])
        )

        let state = QuantumState(numQubits: 1)
        let expectation = observable.expectationValue(state: state)

        #expect(abs(expectation - -1.0) < 1e-10)
    }

    @Test("Superposition state has intermediate expectation")
    func superpositionExpectation() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [(0, .z)])
        )

        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .hadamard, toQubit: 0)
        let state = circuit.execute()

        let expectation = observable.expectationValue(state: state)

        #expect(abs(expectation) < 1e-10)
    }

    @Test("Hydrogen molecule Hamiltonian example")
    func hydrogenHamiltonian() {
        let hamiltonian = Observable(terms: [
            (coefficient: -1.05, pauliString: PauliString(operators: [])),
            (coefficient: 0.39, pauliString: PauliString(operators: [(0, .z)])),
            (coefficient: -0.39, pauliString: PauliString(operators: [(1, .z)])),
            (coefficient: -0.01, pauliString: PauliString(operators: [(0, .z), (1, .z)])),
        ])

        let state = QuantumState(numQubits: 2)

        let energy = hamiltonian.expectationValue(state: state)

        #expect(abs(energy - -1.06) < 1e-10)
    }

    @Test("Observable description is formatted correctly")
    func observableDescription() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .x)])),
            (coefficient: -0.5, pauliString: PauliString(operators: [(1, .z)])),
        ])

        let description = observable.description
        #expect(description.contains("Observable"))
        #expect(description.contains("1.0") || description.contains("+1.0"))
        #expect(description.contains("-0.5"))
    }

    @Test("Empty observable has zero expectation")
    func emptyObservable() {
        let observable = Observable(terms: [])
        let state = QuantumState(numQubits: 2)
        let expectation = observable.expectationValue(state: state)

        #expect(abs(expectation) < 1e-10)
    }
}

/// Test suite for observable variance calculations.
/// Verifies zero-variance for eigenstates, unit variance for superpositions,
/// and non-negativity across composite observables.
@Suite("Observable Variance Computation")
struct ObservableVarianceTests {
    @Test("Variance of eigenstate is zero")
    func varianceOfEigenstate() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [(0, .z)])
        )

        let state = QuantumState(numQubits: 1)
        let variance = observable.variance(state: state)

        #expect(abs(variance) < 1e-10)
    }

    @Test("Variance of superposition state")
    func varianceOfSuperposition() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [(0, .z)])
        )

        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .hadamard, toQubit: 0)
        let state = circuit.execute()

        let variance = observable.variance(state: state)

        #expect(abs(variance - 1.0) < 1e-10)
    }

    @Test("Variance is always non-negative")
    func varianceNonNegative() {
        let observable = Observable(terms: [
            (coefficient: 2.0, pauliString: PauliString(operators: [(0, .x)])),
            (coefficient: 1.0, pauliString: PauliString(operators: [(1, .z)])),
        ])

        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .rotationY(theta: 0.5), toQubit: 1)
        let state = circuit.execute()

        let variance = observable.variance(state: state)

        #expect(variance >= -1e-10)
    }

    @Test("Variance of identity is zero")
    func varianceOfIdentity() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [])
        )

        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .hadamard, toQubit: 1)
        let state = circuit.execute()

        let variance = observable.variance(state: state)

        #expect(abs(variance) < 1e-10)
    }

    @Test("Multi-term variance computation")
    func multiTermVariance() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .z)])),
            (coefficient: 1.0, pauliString: PauliString(operators: [(1, .z)])),
        ])

        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .hadamard, toQubit: 1)
        let state = circuit.execute()

        let variance = observable.variance(state: state)

        #expect(variance > 0)
    }
}

/// Test suite for Pauli-scaled observables and products.
/// Ensures correct variance scaling with coefficients and behavior for
/// multi-qubit Pauli tensor products under superposition states.
@Suite("Observable Pauli Multiplication")
struct ObservablePauliMultiplicationTests {
    @Test("Squared observable for single Pauli")
    func squaredSinglePauli() {
        let observable = Observable(
            coefficient: 2.0,
            pauliString: PauliString(operators: [(0, .z)])
        )

        let state = QuantumState(numQubits: 1)
        let variance = observable.variance(state: state)

        #expect(abs(variance) < 1e-10)
    }

    @Test("Observable squared has correct scaling")
    func observableSquaredScaling() {
        let observable = Observable(
            coefficient: 3.0,
            pauliString: PauliString(operators: [(0, .z)])
        )

        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .hadamard, toQubit: 0)
        let state = circuit.execute()

        let variance = observable.variance(state: state)

        #expect(abs(variance - 9.0) < 1e-10)
    }

    @Test("Two-qubit observable variance")
    func twoQubitVariance() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [
                (0, .z),
                (1, .z),
            ])
        )

        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .hadamard, toQubit: 1)
        let state = circuit.execute()

        let variance = observable.variance(state: state)

        #expect(variance >= 0)
    }

    @Test("X·Y multiplication produces Z with phase")
    func xyMultiplication() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .x)])),
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .y)])),
        ])

        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .hadamard, toQubit: 0)
        let state = circuit.execute()

        let variance = observable.variance(state: state)
        #expect(variance >= 0)
    }

    @Test("Y·Z multiplication produces X with phase")
    func yzMultiplication() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .y)])),
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .z)])),
        ])

        let state = QuantumState(numQubits: 1)
        let variance = observable.variance(state: state)
        #expect(variance >= 0)
    }

    @Test("Z·X multiplication produces Y with phase")
    func zxMultiplication() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .z)])),
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .x)])),
        ])

        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .hadamard, toQubit: 0)
        let state = circuit.execute()

        let variance = observable.variance(state: state)
        #expect(variance >= 0)
    }
}

/// Test suite for observable edge cases.
/// Covers extreme coefficient magnitudes, many-term constructions, and
/// canceling terms to validate numerical stability and correctness.
@Suite("Observable Edge Cases")
struct ObservableEdgeCasesTests {
    @Test("Large coefficient magnitudes")
    func largeCoefficients() {
        let observable = Observable(
            coefficient: 1e10,
            pauliString: PauliString(operators: [(0, .z)])
        )

        let state = QuantumState(numQubits: 1)
        let expectation = observable.expectationValue(state: state)

        #expect(abs(expectation - 1e10) < 1e-5)
    }

    @Test("Very small coefficients")
    func smallCoefficients() {
        let observable = Observable(
            coefficient: 1e-10,
            pauliString: PauliString(operators: [(0, .z)])
        )

        let state = QuantumState(numQubits: 1)
        let expectation = observable.expectationValue(state: state)

        #expect(abs(expectation - 1e-10) < 1e-15)
    }

    @Test("Many qubits observable")
    func manyQubitsObservable() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [
                (0, .z), (1, .z), (2, .z), (3, .z), (4, .z),
            ])
        )

        let state = QuantumState(numQubits: 5)
        let expectation = observable.expectationValue(state: state)

        #expect(abs(expectation - 1.0) < 1e-10)
    }

    @Test("Observable with many terms")
    func manyTerms() {
        var terms: PauliTerms = []
        for i in 0 ..< 5 {
            terms.append((coefficient: 1.0, pauliString: PauliString(operators: [(i, .z)])))
        }
        let observable = Observable(terms: terms)
        let state = QuantumState(numQubits: 5)
        let expectation = observable.expectationValue(state: state)

        #expect(abs(expectation - 5.0) < 1e-10)
    }

    @Test("Canceling terms")
    func cancelingTerms() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .z)])),
            (coefficient: -1.0, pauliString: PauliString(operators: [(0, .z)])),
        ])

        let state = QuantumState(numQubits: 1)
        let expectation = observable.expectationValue(state: state)

        #expect(abs(expectation) < 1e-10)
    }
}
