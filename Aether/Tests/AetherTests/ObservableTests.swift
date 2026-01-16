// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
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
            pauliString: PauliString(),
        )

        let state = QuantumState(qubits: 2)
        let expectation = identity.expectationValue(of: state)

        #expect(abs(expectation - 1.0) < 1e-10)
    }

    @Test("Pauli-Z expectation on |0⟩ is +1")
    func pauliZOnZeroState() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(0)),
        )

        let state = QuantumState(qubits: 1)
        let expectation = observable.expectationValue(of: state)

        #expect(abs(expectation - 1.0) < 1e-10)
    }

    @Test("Pauli-Z expectation on |1⟩ is -1")
    func pauliZOnOneState() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(0)),
        )

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        let state = circuit.execute()

        let expectation = observable.expectationValue(of: state)
        #expect(abs(expectation - -1.0) < 1e-10)
    }

    @Test("Pauli-X expectation on |+⟩ is +1")
    func pauliXOnPlusState() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.x(0)),
        )

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let state = circuit.execute()

        let expectation = observable.expectationValue(of: state)
        #expect(abs(expectation - 1.0) < 1e-10)
    }

    @Test("Pauli-X expectation on |-⟩ is -1")
    func pauliXOnMinusState() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.x(0)),
        )

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        circuit.append(.hadamard, to: 0)
        let state = circuit.execute()

        let expectation = observable.expectationValue(of: state)
        #expect(abs(expectation - -1.0) < 1e-10)
    }

    @Test("Pauli-Y expectation on |+i⟩ is +1")
    func pauliYOnPlusIState() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.y(0)),
        )

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.sGate, to: 0)
        let state = circuit.execute()

        let expectation = observable.expectationValue(of: state)
        #expect(abs(expectation - 1.0) < 1e-10)
    }

    @Test("Two-qubit Z⊗Z on Bell state")
    func zzOnBellState() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(0), .z(1)),
        )

        let bell = QuantumCircuit.bellPhiPlus().execute()

        let expectation = observable.expectationValue(of: bell)
        #expect(abs(expectation - 1.0) < 1e-10)
    }

    @Test("Two-qubit X⊗X on Bell state")
    func xxOnBellState() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.x(0), .x(1)),
        )

        let bell = QuantumCircuit.bellPhiPlus().execute()
        let expectation = observable.expectationValue(of: bell)

        #expect(abs(expectation - 1.0) < 1e-10)
    }

    @Test("Multi-term observable sums correctly")
    func multiTermObservable() {
        let observable = Observable(terms: [
            (coefficient: 2.0, pauliString: PauliString(.z(0))),
            (coefficient: 3.0, pauliString: PauliString(.z(1))),
        ])

        let state = QuantumState(qubits: 2)
        let expectation = observable.expectationValue(of: state)

        #expect(abs(expectation - 5.0) < 1e-10)
    }

    @Test("Negative coefficient")
    func negativeCoefficient() {
        let observable = Observable(
            coefficient: -1.0,
            pauliString: PauliString(.z(0)),
        )

        let state = QuantumState(qubits: 1)
        let expectation = observable.expectationValue(of: state)

        #expect(abs(expectation - -1.0) < 1e-10)
    }

    @Test("Superposition state has intermediate expectation")
    func superpositionExpectation() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(0)),
        )

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let state = circuit.execute()

        let expectation = observable.expectationValue(of: state)

        #expect(abs(expectation) < 1e-10)
    }

    @Test("Hydrogen molecule Hamiltonian example")
    func hydrogenHamiltonian() {
        let hamiltonian = Observable(terms: [
            (coefficient: -1.05, pauliString: PauliString()),
            (coefficient: 0.39, pauliString: PauliString(.z(0))),
            (coefficient: -0.39, pauliString: PauliString(.z(1))),
            (coefficient: -0.01, pauliString: PauliString(.z(0), .z(1))),
        ])

        let state = QuantumState(qubits: 2)

        let energy = hamiltonian.expectationValue(of: state)

        #expect(abs(energy - -1.06) < 1e-10)
    }

    @Test("Observable description is formatted correctly")
    func observableDescription() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.x(0))),
            (coefficient: -0.5, pauliString: PauliString(.z(1))),
        ])

        let description = observable.description
        #expect(description.contains("Observable"))
        #expect(description.contains("1.0") || description.contains("+1.0"))
        #expect(description.contains("-0.5"))
    }

    @Test("Empty observable has zero expectation")
    func emptyObservable() {
        let observable = Observable(terms: [])
        let state = QuantumState(qubits: 2)
        let expectation = observable.expectationValue(of: state)

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
            pauliString: PauliString(.z(0)),
        )

        let state = QuantumState(qubits: 1)
        let variance = observable.variance(of: state)

        #expect(abs(variance) < 1e-10)
    }

    @Test("Variance of superposition state")
    func varianceOfSuperposition() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(0)),
        )

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let state = circuit.execute()

        let variance = observable.variance(of: state)

        #expect(abs(variance - 1.0) < 1e-10)
    }

    @Test("Variance is always non-negative")
    func varianceNonNegative() {
        let observable = Observable(terms: [
            (coefficient: 2.0, pauliString: PauliString(.x(0))),
            (coefficient: 1.0, pauliString: PauliString(.z(1))),
        ])

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.rotationY(0.5), to: 1)
        let state = circuit.execute()

        let variance = observable.variance(of: state)

        #expect(variance >= -1e-10)
    }

    @Test("Variance of identity is zero")
    func varianceOfIdentity() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(),
        )

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        let state = circuit.execute()

        let variance = observable.variance(of: state)

        #expect(abs(variance) < 1e-10)
    }

    @Test("Multi-term variance computation")
    func multiTermVariance() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
            (coefficient: 1.0, pauliString: PauliString(.z(1))),
        ])

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        let state = circuit.execute()

        let variance = observable.variance(of: state)

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
            pauliString: PauliString(.z(0)),
        )

        let state = QuantumState(qubits: 1)
        let variance = observable.variance(of: state)

        #expect(abs(variance) < 1e-10)
    }

    @Test("Observable squared has correct scaling")
    func observableSquaredScaling() {
        let observable = Observable(
            coefficient: 3.0,
            pauliString: PauliString(.z(0)),
        )

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let state = circuit.execute()

        let variance = observable.variance(of: state)

        #expect(abs(variance - 9.0) < 1e-10)
    }

    @Test("Two-qubit observable variance")
    func twoQubitVariance() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(0), .z(1)),
        )

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        let state = circuit.execute()

        let variance = observable.variance(of: state)

        #expect(variance >= 0)
    }

    @Test("X·Y multiplication produces Z with phase")
    func xyMultiplication() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.x(0))),
            (coefficient: 1.0, pauliString: PauliString(.y(0))),
        ])

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let state = circuit.execute()

        let variance = observable.variance(of: state)
        #expect(variance >= 0)
    }

    @Test("Y·Z multiplication produces X with phase")
    func yzMultiplication() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.y(0))),
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
        ])

        let state = QuantumState(qubits: 1)
        let variance = observable.variance(of: state)
        #expect(variance >= 0)
    }

    @Test("Z·X multiplication produces Y with phase")
    func zxMultiplication() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
            (coefficient: 1.0, pauliString: PauliString(.x(0))),
        ])

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let state = circuit.execute()

        let variance = observable.variance(of: state)
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
            pauliString: PauliString(.z(0)),
        )

        let state = QuantumState(qubits: 1)
        let expectation = observable.expectationValue(of: state)

        #expect(abs(expectation - 1e10) < 1e-5)
    }

    @Test("Very small coefficients")
    func smallCoefficients() {
        let observable = Observable(
            coefficient: 1e-10,
            pauliString: PauliString(.z(0)),
        )

        let state = QuantumState(qubits: 1)
        let expectation = observable.expectationValue(of: state)

        #expect(abs(expectation - 1e-10) < 1e-15)
    }

    @Test("Many qubits observable")
    func manyQubitsObservable() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(0), .z(1), .z(2), .z(3), .z(4)),
        )

        let state = QuantumState(qubits: 5)
        let expectation = observable.expectationValue(of: state)

        #expect(abs(expectation - 1.0) < 1e-10)
    }

    @Test("Observable with many terms")
    func manyTerms() {
        var terms: PauliTerms = []
        for i in 0 ..< 5 {
            terms.append((coefficient: 1.0, pauliString: PauliString(.z(i))))
        }
        let observable = Observable(terms: terms)
        let state = QuantumState(qubits: 5)
        let expectation = observable.expectationValue(of: state)

        #expect(abs(expectation - 5.0) < 1e-10)
    }

    @Test("Canceling terms")
    func cancelingTerms() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
            (coefficient: -1.0, pauliString: PauliString(.z(0))),
        ])

        let state = QuantumState(qubits: 1)
        let expectation = observable.expectationValue(of: state)

        #expect(abs(expectation) < 1e-10)
    }

    @Test("Empty observable description")
    func emptyObservableDescription() {
        let observable = Observable(terms: [])
        let description = observable.description
        #expect(description == "Observable: 0")
    }

    @Test("Identity term shows as I in description")
    func identityTermDescription() {
        let observable = Observable(
            coefficient: 2.5,
            pauliString: PauliString(),
        )
        let description = observable.description
        #expect(description.contains("I"))
        #expect(description.contains("2.5"))
    }
}
