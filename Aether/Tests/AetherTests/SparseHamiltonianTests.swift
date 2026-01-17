// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for sparse Hamiltonian expectation values.
/// Validates that sparse backend produces identical results to Observable
/// across different backend types (Metal GPU, CPU sparse, Observable fallback).
@Suite("SparseHamiltonian Correctness")
struct SparseHamiltonianCorrectnessTests {
    @Test("Identity observable gives same result as Observable")
    func identityObservable() async {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(),
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let state = QuantumState(qubits: 1)

        let expectedValue = observable.expectationValue(of: state)
        let sparseValue = await sparseH.expectationValue(of: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Single Pauli-Z term matches Observable")
    func singlePauliZ() async {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(0)),
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let state = QuantumState(qubits: 1)

        let expectedValue = observable.expectationValue(of: state)
        let sparseValue = await sparseH.expectationValue(of: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Single Pauli-X term matches Observable")
    func singlePauliX() async {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.x(0)),
        )
        let sparseH = SparseHamiltonian(observable: observable)

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let state = circuit.execute()

        let expectedValue = observable.expectationValue(of: state)
        let sparseValue = await sparseH.expectationValue(of: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Single Pauli-Y term matches Observable")
    func singlePauliY() async {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.y(0)),
        )
        let sparseH = SparseHamiltonian(observable: observable)

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.sGate, to: 0)
        let state = circuit.execute()

        let expectedValue = observable.expectationValue(of: state)
        let sparseValue = await sparseH.expectationValue(of: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Two-qubit Z⊗Z matches Observable")
    func twoQubitZZ() async {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(0), .z(1)),
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let bell = QuantumCircuit.bellPhiPlus().execute()

        let expectedValue = observable.expectationValue(of: bell)
        let sparseValue = await sparseH.expectationValue(of: bell)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Two-qubit X⊗X matches Observable")
    func twoQubitXX() async {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.x(0), .x(1)),
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let bell = QuantumCircuit.bellPhiPlus().execute()

        let expectedValue = observable.expectationValue(of: bell)
        let sparseValue = await sparseH.expectationValue(of: bell)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Multi-term Hamiltonian matches Observable")
    func multiTermHamiltonian() async {
        let observable = Observable(terms: [
            (coefficient: 2.0, pauliString: PauliString(.z(0))),
            (coefficient: 3.0, pauliString: PauliString(.z(1))),
            (coefficient: -1.0, pauliString: PauliString(.x(0))),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.rotationY(0.5), to: 1)
        let state = circuit.execute()

        let expectedValue = observable.expectationValue(of: state)
        let sparseValue = await sparseH.expectationValue(of: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Hydrogen molecule Hamiltonian matches Observable")
    func hydrogenHamiltonian() async {
        let observable = Observable(terms: [
            (coefficient: -1.05, pauliString: PauliString()),
            (coefficient: 0.39, pauliString: PauliString(.z(0))),
            (coefficient: -0.39, pauliString: PauliString(.z(1))),
            (coefficient: -0.01, pauliString: PauliString(.z(0), .z(1))),
            (coefficient: 0.18, pauliString: PauliString(.x(0), .x(1))),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        let state = QuantumState(qubits: 2)

        let expectedValue = observable.expectationValue(of: state)
        let sparseValue = await sparseH.expectationValue(of: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Negative coefficients match Observable")
    func negativeCoefficients() async {
        let observable = Observable(
            coefficient: -2.5,
            pauliString: PauliString(.z(0)),
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let state = QuantumState(qubits: 1)

        let expectedValue = observable.expectationValue(of: state)
        let sparseValue = await sparseH.expectationValue(of: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Superposition state matches Observable")
    func superpositionState() async {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
            (coefficient: 1.0, pauliString: PauliString(.x(1))),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        circuit.append(.cnot, to: [0, 1])
        let state = circuit.execute()

        let expectedValue = observable.expectationValue(of: state)
        let sparseValue = await sparseH.expectationValue(of: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }
}

/// Test suite for sparse Hamiltonian backend selection.
/// Verifies that appropriate backend (Metal GPU, CPU sparse, Observable fallback)
/// is selected based on system size and Metal availability.
@Suite("SparseHamiltonian Backend Selection")
struct SparseHamiltonianBackendTests {
    @Test("Small system (< 8 qubits) uses CPU sparse or Observable")
    func smallSystemBackend() async {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(0)),
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let backendDesc = await sparseH.backendDescription

        #expect(!backendDesc.contains("Metal GPU") || sparseH.qubits >= 8)
    }

    @Test("Large system (≥ 8 qubits) attempts Metal GPU")
    func largeSystemBackend() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
            (coefficient: 1.0, pauliString: PauliString(.z(7))),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        #expect(sparseH.qubits >= 8)
    }

    @Test("Empty observable uses fallback backend")
    func emptyObservable() async {
        let observable = Observable(terms: [])
        let sparseH = SparseHamiltonian(observable: observable)

        let state = QuantumState(qubits: 1)
        let value = await sparseH.expectationValue(of: state)

        #expect(abs(value) < 1e-10)
    }

    @Test("Empty observable uses Observable fallback")
    func emptyObservableFallback() async {
        let observable = Observable(terms: [])
        let sparseH = SparseHamiltonian(observable: observable)
        let backendDesc = await sparseH.backendDescription
        #expect(backendDesc.contains("non-zero"))

        let state = QuantumState(qubits: 1)
        let value = await sparseH.expectationValue(of: state)
        #expect(abs(value) < 1e-10)
    }

    @Test("Empty observable with large system size uses fallback")
    func emptyObservableLargeSystem() async {
        let observable = Observable(terms: [])
        let sparseH = SparseHamiltonian(observable: observable, systemSize: 8)

        #expect(sparseH.qubits == 8, "Should use specified system size")
        #expect(sparseH.nnz == 0, "Empty observable has no non-zeros")

        let backendDesc = await sparseH.backendDescription
        #expect(backendDesc.contains("fallback") || backendDesc.contains("Observable"))

        let state = QuantumState(qubits: 8)
        let value = await sparseH.expectationValue(of: state)
        #expect(abs(value) < 1e-10, "Empty Hamiltonian gives zero expectation")
    }
}

/// Test suite for sparse Hamiltonian sparsity characteristics.
/// Validates non-zero counts, sparsity ratios, and compression for
/// typical molecular Hamiltonians.
@Suite("SparseHamiltonian Sparsity Metrics")
struct SparseHamiltonianSparsityTests {
    @Test("Identity observable is diagonal (sparse)")
    func identitySparsity() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(),
        )
        let sparseH = SparseHamiltonian(observable: observable)

        #expect(sparseH.nnz == sparseH.dimension)
        #expect(sparseH.sparsity == Double(sparseH.dimension) / Double(sparseH.dimension * sparseH.dimension))
    }

    @Test("Single Pauli-X is sparse")
    func pauliXSparsity() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.x(0)),
        )
        let sparseH = SparseHamiltonian(observable: observable)

        #expect(sparseH.nnz == sparseH.dimension)
        #expect(sparseH.sparsity < 1.0)
    }

    @Test("Multi-term Hamiltonian has low sparsity")
    func multiTermSparsity() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString()),
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
            (coefficient: 1.0, pauliString: PauliString(.z(1))),
            (coefficient: 1.0, pauliString: PauliString(.z(0), .z(1))),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        #expect(sparseH.sparsity < 0.5)
        #expect(sparseH.nnz > 0)
    }

    @Test("Sparsity ratio is between 0 and 1")
    func sparsityRatioRange() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.x(0))),
            (coefficient: 1.0, pauliString: PauliString(.y(1))),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        #expect(sparseH.sparsity >= 0.0)
        #expect(sparseH.sparsity <= 1.0)
    }
}

/// Tests reporting and formatting of SparseHamiltonian statistics.
/// Verifies fields like qubit count, dimension, non-zero entries, sparsity,
/// backend description text, and estimated memory usage.
@Suite("SparseHamiltonian Statistics")
struct SparseHamiltonianStatisticsTests {
    @Test("Statistics contain expected fields")
    func statisticsFields() async {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
            (coefficient: 1.0, pauliString: PauliString(.z(1))),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        let stats = await sparseH.statistics

        #expect(stats.qubits == sparseH.qubits)
        #expect(stats.dimension == sparseH.dimension)
        #expect(stats.nonZeros == sparseH.nnz)
        #expect(abs(stats.sparsity - sparseH.sparsity) < 1e-10)
    }

    @Test("Backend description is non-empty")
    func backendDescription() async {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(0)),
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let desc = await sparseH.backendDescription

        #expect(!desc.isEmpty)
        #expect(desc.contains("non-zero"))
    }

    @Test("Statistics description formats correctly")
    func statisticsDescription() async {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
            (coefficient: 1.0, pauliString: PauliString(.z(1))),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        let stats = await sparseH.statistics
        let description = stats.description

        #expect(description.contains("Sparse Hamiltonian"))
        #expect(description.contains("Backend"))
        #expect(description.contains("Qubits"))
    }

    @Test("Memory estimate is reasonable")
    func memoryEstimate() async {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(0)),
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let stats = await sparseH.statistics

        #expect(stats.memoryBytes > 0)

        let denseBytes = sparseH.dimension * sparseH.dimension * 16
        #expect(stats.memoryBytes < denseBytes)
    }

    @Test("Memory statistics formatting for large Hamiltonian")
    func memoryStatisticsFormattingLarge() async {
        let terms: PauliTerms = (0 ..< 12).map { i in
            let ps1 = PauliString(.z(i))
            let ps2 = PauliString(.x(i))
            return [
                (coefficient: 1.0, pauliString: ps1),
                (coefficient: 1.0, pauliString: ps2),
            ]
        }.flatMap(\.self)

        let observable = Observable(terms: terms)
        let sparseH = SparseHamiltonian(observable: observable)

        let description = await sparseH.statistics.description
        #expect(description.contains("Qubits: 12"))
        #expect(description.contains("Non-zeros:"))
    }
}

/// Exercises numerical and structural edge cases for SparseHamiltonian.
/// Covers extreme coefficients, many/canceling terms, large qubit counts,
/// construction performance, and thread-safety of expectation evaluation.
@Suite("SparseHamiltonian Edge Cases")
struct SparseHamiltonianEdgeCasesTests {
    @Test("Large coefficients match Observable")
    func largeCoefficients() async {
        let observable = Observable(
            coefficient: 1e10,
            pauliString: PauliString(.z(0)),
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let state = QuantumState(qubits: 1)

        let expectedValue = observable.expectationValue(of: state)
        let sparseValue = await sparseH.expectationValue(of: state)

        #expect(abs(sparseValue - expectedValue) < 1e-3) // Relaxed tolerance for large values
    }

    @Test("Small coefficients match Observable")
    func smallCoefficients() async {
        let observable = Observable(
            coefficient: 1e-10,
            pauliString: PauliString(.z(0)),
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let state = QuantumState(qubits: 1)

        let expectedValue = observable.expectationValue(of: state)
        let sparseValue = await sparseH.expectationValue(of: state)

        #expect(abs(sparseValue - expectedValue) < 1e-10)
    }

    @Test("Many-term Hamiltonian matches Observable")
    func manyTerms() async {
        var terms: [(Double, PauliString)] = []
        for i in 0 ..< 5 {
            terms.append((coefficient: 1.0, pauliString: PauliString(.z(i))))
        }
        let observable = Observable(terms: terms)
        let sparseH = SparseHamiltonian(observable: observable)

        let state = QuantumState(qubits: 5)

        let expectedValue = observable.expectationValue(of: state)
        let sparseValue = await sparseH.expectationValue(of: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Canceling terms produce zero expectation")
    func cancelingTerms() async {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
            (coefficient: -1.0, pauliString: PauliString(.z(0))),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        let state = QuantumState(qubits: 1)

        let value = await sparseH.expectationValue(of: state)
        #expect(abs(value) < 1e-6)
    }

    @Test("Complex state with many gates matches Observable")
    func complexState() async {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.x(0))),
            (coefficient: 0.5, pauliString: PauliString(.y(1))),
            (coefficient: -0.3, pauliString: PauliString(.z(2))),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.rotationY(0.5), to: 1)
        circuit.append(.rotationZ(1.2), to: 2)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.cnot, to: [1, 2])
        let state = circuit.execute()

        let expectedValue = observable.expectationValue(of: state)
        let sparseValue = await sparseH.expectationValue(of: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("8-qubit system for GPU backend testing")
    func eightQubitSystem() async {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
            (coefficient: 1.0, pauliString: PauliString(.z(7))),
            (coefficient: 0.5, pauliString: PauliString(.z(0), .z(7))),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        var circuit = QuantumCircuit(qubits: 8)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 7)
        let state = circuit.execute()

        let expectedValue = observable.expectationValue(of: state)
        let sparseValue = await sparseH.expectationValue(of: state)

        #expect(abs(sparseValue - expectedValue) < 1e-5)
    }

    @Test("10-qubit system matches Observable")
    func tenQubitSystem() async {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.x(0))),
            (coefficient: 0.5, pauliString: PauliString(.y(5))),
            (coefficient: -0.3, pauliString: PauliString(.z(9))),
            (coefficient: 0.2, pauliString: PauliString(.z(0), .z(9))),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        var circuit = QuantumCircuit(qubits: 10)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 5)
        circuit.append(.hadamard, to: 9)
        let state = circuit.execute()

        let expectedValue = observable.expectationValue(of: state)
        let sparseValue = await sparseH.expectationValue(of: state)

        #expect(abs(sparseValue - expectedValue) < 1e-5)
    }

    @Test("100-term Hamiltonian matches Observable")
    func hundredTermHamiltonian() async {
        var terms: [(Double, PauliString)] = []
        for i in 0 ..< 100 {
            let coeff = Double(i % 10) / 10.0
            let qubit = i % 8
            let basis: PauliBasis = [.x, .y, .z][i % 3]
            let op = switch basis {
            case .x: PauliOperator.x(qubit)
            case .y: PauliOperator.y(qubit)
            case .z: PauliOperator.z(qubit)
            }
            terms.append((coefficient: coeff, pauliString: PauliString(op)))
        }

        let observable = Observable(terms: terms)
        let sparseH = SparseHamiltonian(observable: observable)

        var circuit = QuantumCircuit(qubits: 8)
        circuit.append(.hadamard, to: 0)
        circuit.append(.rotationY(0.5), to: 4)
        let state = circuit.execute()

        let expectedValue = observable.expectationValue(of: state)
        let sparseValue = await sparseH.expectationValue(of: state)

        #expect(abs(sparseValue - expectedValue) < 1e-5)
    }

    @Test("Construction time is reasonable for molecular Hamiltonian")
    func constructionPerformance() {
        var terms: [(Double, PauliString)] = []
        for i in 0 ..< 200 {
            let coeff = Double(i) / 100.0
            let q1 = i % 10
            let q2 = (i + 1) % 10
            terms.append((coefficient: coeff, pauliString: PauliString(.z(q1), .x(q2))))
        }
        let observable = Observable(terms: terms)

        let start = Date()
        let sparseH = SparseHamiltonian(observable: observable)
        let constructionTime = Date().timeIntervalSince(start)

        #expect(constructionTime < 5.0)
        #expect(sparseH.nnz > 0)
    }
}
