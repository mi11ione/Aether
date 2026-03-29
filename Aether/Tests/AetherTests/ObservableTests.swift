// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
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
            pauliString: PauliString(),
        )

        let state = QuantumState(qubits: 2)
        let expectation = identity.expectationValue(of: state)

        #expect(abs(expectation - 1.0) < 1e-10, "Identity expectation should be +1")
    }

    @Test("Pauli-Z expectation on |0⟩ is +1")
    func pauliZOnZeroState() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(0)),
        )

        let state = QuantumState(qubits: 1)
        let expectation = observable.expectationValue(of: state)

        #expect(abs(expectation - 1.0) < 1e-10, "Z on |0⟩ should give +1")
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
        #expect(abs(expectation - -1.0) < 1e-10, "Z on |1⟩ should give -1")
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
        #expect(abs(expectation - 1.0) < 1e-10, "X on |+⟩ should give +1")
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
        #expect(abs(expectation - -1.0) < 1e-10, "X on |-⟩ should give -1")
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
        #expect(abs(expectation - 1.0) < 1e-10, "Y on |+i⟩ should give +1")
    }

    @Test("Two-qubit Z⊗Z on Bell state")
    func zzOnBellState() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(0), .z(1)),
        )

        let bell = QuantumCircuit.bellPhiPlus().execute()

        let expectation = observable.expectationValue(of: bell)
        #expect(abs(expectation - 1.0) < 1e-10, "ZZ on Bell |Φ+⟩ should give +1")
    }

    @Test("Two-qubit X⊗X on Bell state")
    func xxOnBellState() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.x(0), .x(1)),
        )

        let bell = QuantumCircuit.bellPhiPlus().execute()
        let expectation = observable.expectationValue(of: bell)

        #expect(abs(expectation - 1.0) < 1e-10, "XX on Bell |Φ+⟩ should give +1")
    }

    @Test("Multi-term observable sums correctly")
    func multiTermObservable() {
        let observable = Observable(terms: [
            (coefficient: 2.0, pauliString: PauliString(.z(0))),
            (coefficient: 3.0, pauliString: PauliString(.z(1))),
        ])

        let state = QuantumState(qubits: 2)
        let expectation = observable.expectationValue(of: state)

        #expect(abs(expectation - 5.0) < 1e-10, "2Z₀ + 3Z₁ on |00⟩ should give 5.0")
    }

    @Test("Negative coefficient")
    func negativeCoefficient() {
        let observable = Observable(
            coefficient: -1.0,
            pauliString: PauliString(.z(0)),
        )

        let state = QuantumState(qubits: 1)
        let expectation = observable.expectationValue(of: state)

        #expect(abs(expectation - -1.0) < 1e-10, "Negative coefficient should negate expectation")
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

        #expect(abs(expectation) < 1e-10, "Z on |+⟩ should give 0")
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

        #expect(abs(energy - -1.06) < 1e-10, "H₂ ground state energy should be -1.06")
    }

    @Test("Observable description is formatted correctly")
    func observableDescription() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.x(0))),
            (coefficient: -0.5, pauliString: PauliString(.z(1))),
        ])

        let description = observable.description
        #expect(description.contains("Observable"), "Should contain Observable prefix")
        #expect(description.contains("1.0") || description.contains("+1.0"), "Should contain coefficient 1.0")
        #expect(description.contains("-0.5"), "Should contain coefficient -0.5")
    }

    @Test("Empty observable has zero expectation")
    func emptyObservable() {
        let observable = Observable(terms: [])
        let state = QuantumState(qubits: 2)
        let expectation = observable.expectationValue(of: state)

        #expect(abs(expectation) < 1e-10, "Empty observable should give zero expectation")
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

        #expect(abs(variance) < 1e-10, "Eigenstate variance should be zero")
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

        #expect(abs(variance - 1.0) < 1e-10, "Z variance on |+⟩ should be 1.0")
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

        #expect(variance >= -1e-10, "Variance must be non-negative")
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

        #expect(abs(variance) < 1e-10, "Identity variance should be zero")
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

        #expect(variance > 0, "Multi-term superposition should have positive variance")
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

        #expect(abs(variance) < 1e-10, "Eigenstate variance for squared Pauli should be zero")
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

        #expect(abs(variance - 9.0) < 1e-10, "Variance should scale as coefficient squared")
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

        #expect(variance >= 0, "Two-qubit variance must be non-negative")
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
        #expect(variance >= 0, "XY product variance must be non-negative")
    }

    @Test("Y·Z multiplication produces X with phase")
    func yzMultiplication() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.y(0))),
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
        ])

        let state = QuantumState(qubits: 1)
        let variance = observable.variance(of: state)
        #expect(variance >= 0, "YZ product variance must be non-negative")
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
        #expect(variance >= 0, "ZX product variance must be non-negative")
    }

    @Test("Squared observable filters cancelled terms from coefficient accumulation")
    func squaredFiltersCancelledTerms() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.x(0))),
            (coefficient: -1.0, pauliString: PauliString(.x(0))),
        ])

        let state = QuantumState(qubits: 1)
        let variance = observable.variance(of: state)

        #expect(abs(variance) < 1e-10, "Variance of cancelled observable should be zero")
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

        #expect(abs(expectation - 1e10) < 1e-5, "Large coefficient expectation should scale correctly")
    }

    @Test("Very small coefficients")
    func smallCoefficients() {
        let observable = Observable(
            coefficient: 1e-10,
            pauliString: PauliString(.z(0)),
        )

        let state = QuantumState(qubits: 1)
        let expectation = observable.expectationValue(of: state)

        #expect(abs(expectation - 1e-10) < 1e-10, "Small coefficient expectation should be precise")
    }

    @Test("Many qubits observable")
    func manyQubitsObservable() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(0), .z(1), .z(2)),
        )

        let state = QuantumState(qubits: 3)
        let expectation = observable.expectationValue(of: state)

        #expect(abs(expectation - 1.0) < 1e-10, "Multi-qubit Z tensor on |000⟩ should give +1")
    }

    @Test("Observable with many terms")
    func manyTerms() {
        var terms: PauliTerms = []
        for i in 0 ..< 3 {
            terms.append((coefficient: 1.0, pauliString: PauliString(.z(i))))
        }
        let observable = Observable(terms: terms)
        let state = QuantumState(qubits: 3)
        let expectation = observable.expectationValue(of: state)

        #expect(abs(expectation - 3.0) < 1e-10, "Sum of Z terms on |000⟩ should equal term count")
    }

    @Test("Canceling terms")
    func cancelingTerms() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
            (coefficient: -1.0, pauliString: PauliString(.z(0))),
        ])

        let state = QuantumState(qubits: 1)
        let expectation = observable.expectationValue(of: state)

        #expect(abs(expectation) < 1e-10, "Canceling terms should give zero expectation")
    }

    @Test("Empty observable description")
    func emptyObservableDescription() {
        let observable = Observable(terms: [])
        let description = observable.description
        #expect(description == "Observable: 0", "Empty observable should display as zero")
    }

    @Test("Identity term shows as I in description")
    func identityTermDescription() {
        let observable = Observable(
            coefficient: 2.5,
            pauliString: PauliString(),
        )
        let description = observable.description
        #expect(description.contains("I"), "Identity should show as I")
        #expect(description.contains("2.5"), "Coefficient should appear in description")
    }

    @Test("pauliX expectation value is correct")
    func pauliXExpectation() {
        let xObs = Observable.pauliX(qubit: 0)

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let plusState = circuit.execute()

        let expectation = xObs.expectationValue(of: plusState)
        #expect(abs(expectation - 1.0) < 1e-10, "⟨+|X|+⟩ should be +1")
    }

    @Test("pauliY expectation value is correct")
    func pauliYExpectation() {
        let yObs = Observable.pauliY(qubit: 0)

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.sGate, to: 0)
        let plusIState = circuit.execute()

        let expectation = yObs.expectationValue(of: plusIState)
        #expect(abs(expectation - 1.0) < 1e-10, "⟨+i|Y|+i⟩ should be +1")
    }

    @Test("pauliZ factory with custom coefficient")
    func pauliZFactoryCustomCoefficient() {
        let zObs = Observable.pauliZ(qubit: 0, coefficient: -3.14)

        #expect(abs(zObs.terms[0].coefficient - -3.14) < 1e-10, "Should use custom coefficient")
    }

    @Test("Squared observable filters near-zero coefficients from Pauli products")
    func squaredFiltersNearZeroCoefficients() {
        let observable = Observable(terms: [
            (coefficient: 1e-8, pauliString: PauliString(.x(0))),
            (coefficient: 1e-8, pauliString: PauliString(.y(0))),
        ])

        let state = QuantumState(qubits: 1)
        let variance = observable.variance(of: state)

        #expect(abs(variance) < 1e-10, "Variance with tiny coefficients should be near zero")
    }

    @Test("Squared filters zero-coefficient results from identical Pauli multiplication")
    func squaredZeroFromIdenticalPaulis() {
        let tiny = 1e-8
        let observable = Observable(terms: [
            (coefficient: tiny, pauliString: PauliString(.z(0))),
            (coefficient: tiny, pauliString: PauliString(.z(1))),
        ])

        let state = QuantumState(qubits: 2)
        let variance = observable.variance(of: state)

        #expect(abs(variance) < 1e-10, "Variance should handle near-zero squared terms")
    }

    @Test("Threshold at iteration 0 equals initial threshold")
    func thresholdAtZero() {
        let schedule = Observable.AdaptiveSchedule(
            initialThreshold: 0.5,
            finalThreshold: 0.0,
            decayRate: 0.1,
        )

        let threshold = schedule.threshold(at: 0)
        #expect(
            abs(threshold - 0.5) < 1e-10,
            "Threshold at iteration 0 should equal initialThreshold",
        )
    }

    @Test("Threshold decays exponentially")
    func thresholdDecaysExponentially() {
        let schedule = Observable.AdaptiveSchedule(
            initialThreshold: 1.0,
            finalThreshold: 0.0,
            decayRate: 0.1,
        )

        let t0 = schedule.threshold(at: 0)
        let t10 = schedule.threshold(at: 10)
        let t20 = schedule.threshold(at: 20)

        #expect(t0 > t10, "Threshold should decrease over iterations")
        #expect(t10 > t20, "Threshold should continue decreasing")

        let expected10 = exp(-1.0)
        #expect(
            abs(t10 - expected10) < 1e-10,
            "Threshold at iteration 10 should follow exponential decay",
        )
    }

    @Test("Threshold approaches final threshold at high iterations")
    func thresholdApproachesFinal() {
        let schedule = Observable.AdaptiveSchedule(
            initialThreshold: 1.0,
            finalThreshold: 0.1,
            decayRate: 0.1,
        )

        let t1000 = schedule.threshold(at: 1000)

        #expect(
            abs(t1000 - 0.1) < 1e-10,
            "Threshold should approach finalThreshold at high iterations",
        )
    }

    @Test("Threshold with non-zero final threshold")
    func thresholdNonZeroFinal() {
        let schedule = Observable.AdaptiveSchedule(
            initialThreshold: 0.5,
            finalThreshold: 0.05,
            decayRate: 0.2,
        )

        let t0 = schedule.threshold(at: 0)
        let t50 = schedule.threshold(at: 50)

        #expect(
            abs(t0 - 0.5) < 1e-10,
            "Initial threshold should be exact",
        )
        #expect(
            t50 >= 0.05,
            "Threshold should never go below finalThreshold",
        )
    }

    @Test("Applying schedule at iteration 0 uses high threshold")
    func applyingAtIterationZero() {
        let hamiltonian = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
            (coefficient: 0.3, pauliString: PauliString(.z(1))),
            (coefficient: 0.05, pauliString: PauliString(.x(0))),
        ])

        let schedule = Observable.AdaptiveSchedule(
            initialThreshold: 0.5,
            finalThreshold: 0.0,
            decayRate: 0.1,
        )

        let filtered = hamiltonian.filtering(schedule: schedule, iteration: 0)

        #expect(
            filtered.terms.count == 1,
            "Only terms with |coeff| >= 0.5 should remain at iteration 0",
        )
        #expect(
            filtered.terms[0].coefficient == 1.0,
            "Largest coefficient term should remain",
        )
    }

    @Test("Applying schedule at high iteration uses low threshold")
    func applyingAtHighIteration() {
        let hamiltonian = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
            (coefficient: 0.3, pauliString: PauliString(.z(1))),
            (coefficient: 0.05, pauliString: PauliString(.x(0))),
        ])

        let schedule = Observable.AdaptiveSchedule(
            initialThreshold: 0.5,
            finalThreshold: 0.0,
            decayRate: 0.1,
        )

        let filtered = hamiltonian.filtering(schedule: schedule, iteration: 100)

        #expect(
            filtered.terms.count == 3,
            "All terms should remain at high iteration with near-zero threshold",
        )
    }

    @Test("Applying schedule progressively includes more terms")
    func applyingProgressiveInclusion() {
        let hamiltonian = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
            (coefficient: 0.2, pauliString: PauliString(.z(1))),
            (coefficient: 0.05, pauliString: PauliString(.x(0))),
        ])

        let schedule = Observable.AdaptiveSchedule.aggressive

        let filtered0 = hamiltonian.filtering(schedule: schedule, iteration: 0)
        let filtered20 = hamiltonian.filtering(schedule: schedule, iteration: 20)
        let filtered50 = hamiltonian.filtering(schedule: schedule, iteration: 50)

        #expect(
            filtered0.terms.count <= filtered20.terms.count,
            "More terms should be included as iteration increases",
        )
        #expect(
            filtered20.terms.count <= filtered50.terms.count,
            "Term count should be non-decreasing with iteration",
        )
    }

    @Test("Applying with moderate schedule")
    func applyingModerateSchedule() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: 0.08, pauliString: PauliString(.z(1))),
            (coefficient: 0.02, pauliString: PauliString(.x(0))),
        ])

        let filtered = hamiltonian.filtering(schedule: .moderate, iteration: 0)

        #expect(
            filtered.terms.count >= 1,
            "At least largest term should survive",
        )
        #expect(
            filtered.terms[0].coefficient == 0.5,
            "Largest term should be first",
        )
    }

    @Test("Applying with conservative schedule retains most terms early")
    func applyingConservativeSchedule() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: 0.05, pauliString: PauliString(.z(1))),
            (coefficient: 0.02, pauliString: PauliString(.x(0))),
        ])

        let filtered = hamiltonian.filtering(schedule: .conservative, iteration: 0)

        #expect(
            filtered.terms.count == 3,
            "Conservative schedule should retain all significant terms",
        )
    }
}
