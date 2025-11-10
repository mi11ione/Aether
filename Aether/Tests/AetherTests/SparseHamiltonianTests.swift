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
    func identityObservable() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [])
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let state = QuantumState(numQubits: 1)

        let expectedValue = observable.expectationValue(state: state)
        let sparseValue = sparseH.expectationValue(state: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Single Pauli-Z term matches Observable")
    func singlePauliZ() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [(0, .z)])
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let state = QuantumState(numQubits: 1)

        let expectedValue = observable.expectationValue(state: state)
        let sparseValue = sparseH.expectationValue(state: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Single Pauli-X term matches Observable")
    func singlePauliX() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [(0, .x)])
        )
        let sparseH = SparseHamiltonian(observable: observable)

        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .hadamard, toQubit: 0)
        let state = circuit.execute()

        let expectedValue = observable.expectationValue(state: state)
        let sparseValue = sparseH.expectationValue(state: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Single Pauli-Y term matches Observable")
    func singlePauliY() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [(0, .y)])
        )
        let sparseH = SparseHamiltonian(observable: observable)

        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .sGate, toQubit: 0)
        let state = circuit.execute()

        let expectedValue = observable.expectationValue(state: state)
        let sparseValue = sparseH.expectationValue(state: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Two-qubit Z⊗Z matches Observable")
    func twoQubitZZ() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [(0, .z), (1, .z)])
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let bell = QuantumCircuit.bellPhiPlus().execute()

        let expectedValue = observable.expectationValue(state: bell)
        let sparseValue = sparseH.expectationValue(state: bell)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Two-qubit X⊗X matches Observable")
    func twoQubitXX() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [(0, .x), (1, .x)])
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let bell = QuantumCircuit.bellPhiPlus().execute()

        let expectedValue = observable.expectationValue(state: bell)
        let sparseValue = sparseH.expectationValue(state: bell)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Multi-term Hamiltonian matches Observable")
    func multiTermHamiltonian() {
        let observable = Observable(terms: [
            (coefficient: 2.0, pauliString: PauliString(operators: [(0, .z)])),
            (coefficient: 3.0, pauliString: PauliString(operators: [(1, .z)])),
            (coefficient: -1.0, pauliString: PauliString(operators: [(0, .x)])),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .rotationY(theta: 0.5), toQubit: 1)
        let state = circuit.execute()

        let expectedValue = observable.expectationValue(state: state)
        let sparseValue = sparseH.expectationValue(state: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Hydrogen molecule Hamiltonian matches Observable")
    func hydrogenHamiltonian() {
        let observable = Observable(terms: [
            (coefficient: -1.05, pauliString: PauliString(operators: [])),
            (coefficient: 0.39, pauliString: PauliString(operators: [(0, .z)])),
            (coefficient: -0.39, pauliString: PauliString(operators: [(1, .z)])),
            (coefficient: -0.01, pauliString: PauliString(operators: [(0, .z), (1, .z)])),
            (coefficient: 0.18, pauliString: PauliString(operators: [(0, .x), (1, .x)])),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        let state = QuantumState(numQubits: 2)

        let expectedValue = observable.expectationValue(state: state)
        let sparseValue = sparseH.expectationValue(state: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Negative coefficients match Observable")
    func negativeCoefficients() {
        let observable = Observable(
            coefficient: -2.5,
            pauliString: PauliString(operators: [(0, .z)])
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let state = QuantumState(numQubits: 1)

        let expectedValue = observable.expectationValue(state: state)
        let sparseValue = sparseH.expectationValue(state: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Superposition state matches Observable")
    func superpositionState() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .z)])),
            (coefficient: 1.0, pauliString: PauliString(operators: [(1, .x)])),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .hadamard, toQubit: 1)
        circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
        let state = circuit.execute()

        let expectedValue = observable.expectationValue(state: state)
        let sparseValue = sparseH.expectationValue(state: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }
}

/// Test suite for sparse Hamiltonian backend selection.
/// Verifies that appropriate backend (Metal GPU, CPU sparse, Observable fallback)
/// is selected based on system size and Metal availability.
@Suite("SparseHamiltonian Backend Selection")
struct SparseHamiltonianBackendTests {
    @Test("Small system (< 8 qubits) uses CPU sparse or Observable")
    func smallSystemBackend() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [(0, .z)])
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let backendDesc = sparseH.backendDescription

        #expect(!backendDesc.contains("Metal GPU") || sparseH.numQubits >= 8)
    }

    @Test("Large system (≥ 8 qubits) attempts Metal GPU")
    func largeSystemBackend() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .z)])),
            (coefficient: 1.0, pauliString: PauliString(operators: [(7, .z)])),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        #expect(sparseH.numQubits >= 8)
    }

    @Test("Empty observable uses fallback backend")
    func emptyObservable() {
        let observable = Observable(terms: [])
        let sparseH = SparseHamiltonian(observable: observable)

        let state = QuantumState(numQubits: 1)
        let value = sparseH.expectationValue(state: state)

        #expect(abs(value) < 1e-10)
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
            pauliString: PauliString(operators: [])
        )
        let sparseH = SparseHamiltonian(observable: observable)

        #expect(sparseH.nnz == sparseH.dimension)
        #expect(sparseH.sparsity == Double(sparseH.dimension) / Double(sparseH.dimension * sparseH.dimension))
    }

    @Test("Single Pauli-X is sparse")
    func pauliXSparsity() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [(0, .x)])
        )
        let sparseH = SparseHamiltonian(observable: observable)

        #expect(sparseH.nnz == sparseH.dimension)
        #expect(sparseH.sparsity < 1.0)
    }

    @Test("Multi-term Hamiltonian has low sparsity")
    func multiTermSparsity() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(operators: [])),
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .z)])),
            (coefficient: 1.0, pauliString: PauliString(operators: [(1, .z)])),
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .z), (1, .z)])),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        #expect(sparseH.sparsity < 0.5)
        #expect(sparseH.nnz > 0)
    }

    @Test("Sparsity ratio is between 0 and 1")
    func sparsityRatioRange() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .x)])),
            (coefficient: 1.0, pauliString: PauliString(operators: [(1, .y)])),
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
    func statisticsFields() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .z)])),
            (coefficient: 1.0, pauliString: PauliString(operators: [(1, .z)])),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        let stats = sparseH.getStatistics()

        #expect(stats.numQubits == sparseH.numQubits)
        #expect(stats.dimension == sparseH.dimension)
        #expect(stats.nonZeros == sparseH.nnz)
        #expect(abs(stats.sparsity - sparseH.sparsity) < 1e-10)
    }

    @Test("Backend description is non-empty")
    func backendDescription() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [(0, .z)])
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let desc = sparseH.backendDescription

        #expect(!desc.isEmpty)
        #expect(desc.contains("non-zero"))
    }

    @Test("Statistics description formats correctly")
    func statisticsDescription() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .z)])),
            (coefficient: 1.0, pauliString: PauliString(operators: [(1, .z)])),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        let stats = sparseH.getStatistics()
        let description = stats.description

        #expect(description.contains("Sparse Hamiltonian"))
        #expect(description.contains("Backend"))
        #expect(description.contains("Qubits"))
    }

    @Test("Memory estimate is reasonable")
    func memoryEstimate() {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(operators: [(0, .z)])
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let stats = sparseH.getStatistics()

        #expect(stats.memoryBytes > 0)

        let denseBytes = sparseH.dimension * sparseH.dimension * 16
        #expect(stats.memoryBytes < denseBytes)
    }
}

/// Exercises numerical and structural edge cases for SparseHamiltonian.
/// Covers extreme coefficients, many/canceling terms, large qubit counts,
/// construction performance, and thread-safety of expectation evaluation.
@Suite("SparseHamiltonian Edge Cases")
struct SparseHamiltonianEdgeCasesTests {
    @Test("Large coefficients match Observable")
    func largeCoefficients() {
        let observable = Observable(
            coefficient: 1e10,
            pauliString: PauliString(operators: [(0, .z)])
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let state = QuantumState(numQubits: 1)

        let expectedValue = observable.expectationValue(state: state)
        let sparseValue = sparseH.expectationValue(state: state)

        #expect(abs(sparseValue - expectedValue) < 1e-3) // Relaxed tolerance for large values
    }

    @Test("Small coefficients match Observable")
    func smallCoefficients() {
        let observable = Observable(
            coefficient: 1e-10,
            pauliString: PauliString(operators: [(0, .z)])
        )
        let sparseH = SparseHamiltonian(observable: observable)

        let state = QuantumState(numQubits: 1)

        let expectedValue = observable.expectationValue(state: state)
        let sparseValue = sparseH.expectationValue(state: state)

        #expect(abs(sparseValue - expectedValue) < 1e-10)
    }

    @Test("Many-term Hamiltonian matches Observable")
    func manyTerms() {
        var terms: [(Double, PauliString)] = []
        for i in 0 ..< 5 {
            terms.append((coefficient: 1.0, pauliString: PauliString(operators: [(i, .z)])))
        }
        let observable = Observable(terms: terms)
        let sparseH = SparseHamiltonian(observable: observable)

        let state = QuantumState(numQubits: 5)

        let expectedValue = observable.expectationValue(state: state)
        let sparseValue = sparseH.expectationValue(state: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("Canceling terms produce zero expectation")
    func cancelingTerms() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .z)])),
            (coefficient: -1.0, pauliString: PauliString(operators: [(0, .z)])),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        let state = QuantumState(numQubits: 1)

        let value = sparseH.expectationValue(state: state)
        #expect(abs(value) < 1e-6)
    }

    @Test("Complex state with many gates matches Observable")
    func complexState() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .x)])),
            (coefficient: 0.5, pauliString: PauliString(operators: [(1, .y)])),
            (coefficient: -0.3, pauliString: PauliString(operators: [(2, .z)])),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        var circuit = QuantumCircuit(numQubits: 3)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .rotationY(theta: 0.5), toQubit: 1)
        circuit.append(gate: .rotationZ(theta: 1.2), toQubit: 2)
        circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
        circuit.append(gate: .cnot(control: 1, target: 2), qubits: [])
        let state = circuit.execute()

        let expectedValue = observable.expectationValue(state: state)
        let sparseValue = sparseH.expectationValue(state: state)

        #expect(abs(sparseValue - expectedValue) < 1e-6)
    }

    @Test("8-qubit system for GPU backend testing")
    func eightQubitSystem() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .z)])),
            (coefficient: 1.0, pauliString: PauliString(operators: [(7, .z)])),
            (coefficient: 0.5, pauliString: PauliString(operators: [(0, .z), (7, .z)])),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        var circuit = QuantumCircuit(numQubits: 8)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .hadamard, toQubit: 7)
        let state = circuit.execute()

        let expectedValue = observable.expectationValue(state: state)
        let sparseValue = sparseH.expectationValue(state: state)

        #expect(abs(sparseValue - expectedValue) < 1e-5)
    }

    @Test("10-qubit system matches Observable")
    func tenQubitSystem() {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .x)])),
            (coefficient: 0.5, pauliString: PauliString(operators: [(5, .y)])),
            (coefficient: -0.3, pauliString: PauliString(operators: [(9, .z)])),
            (coefficient: 0.2, pauliString: PauliString(operators: [(0, .z), (9, .z)])),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        var circuit = QuantumCircuit(numQubits: 10)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .hadamard, toQubit: 5)
        circuit.append(gate: .hadamard, toQubit: 9)
        let state = circuit.execute()

        let expectedValue = observable.expectationValue(state: state)
        let sparseValue = sparseH.expectationValue(state: state)

        #expect(abs(sparseValue - expectedValue) < 1e-5)
    }

    @Test("100-term Hamiltonian matches Observable")
    func hundredTermHamiltonian() {
        var terms: [(Double, PauliString)] = []
        for i in 0 ..< 100 {
            let coeff = Double(i % 10) / 10.0
            let qubit = i % 8
            let basis: PauliBasis = [.x, .y, .z][i % 3]
            terms.append((coefficient: coeff, pauliString: PauliString(operators: [(qubit, basis)])))
        }
        let observable = Observable(terms: terms)
        let sparseH = SparseHamiltonian(observable: observable)

        var circuit = QuantumCircuit(numQubits: 8)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .rotationY(theta: 0.5), toQubit: 4)
        let state = circuit.execute()

        let expectedValue = observable.expectationValue(state: state)
        let sparseValue = sparseH.expectationValue(state: state)

        #expect(abs(sparseValue - expectedValue) < 1e-5)
    }

    @Test("Construction time is reasonable for molecular Hamiltonian")
    func constructionPerformance() {
        var terms: [(Double, PauliString)] = []
        for i in 0 ..< 200 {
            let coeff = Double(i) / 100.0
            let q1 = i % 10
            let q2 = (i + 1) % 10
            terms.append((coefficient: coeff, pauliString: PauliString(operators: [(q1, .z), (q2, .x)])))
        }
        let observable = Observable(terms: terms)

        let start = Date()
        let sparseH = SparseHamiltonian(observable: observable)
        let constructionTime = Date().timeIntervalSince(start)

        #expect(constructionTime < 5.0)
        #expect(sparseH.nnz > 0)
    }

    @Test("Thread safety: concurrent expectation value calls")
    func threadSafety() async {
        let observable = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(operators: [(0, .z)])),
            (coefficient: 0.5, pauliString: PauliString(operators: [(1, .x)])),
        ])
        let sparseH = SparseHamiltonian(observable: observable)

        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        let state = circuit.execute()

        let expectedValue = observable.expectationValue(state: state)

        let results: [Double] = await withTaskGroup(of: (Int, Double).self) { group in
            for i in 0 ..< 10 {
                group.addTask { (i, sparseH.expectationValue(state: state)) }
            }

            var temp = Array(repeating: 0.0, count: 10)
            for await (i, value) in group {
                temp[i] = value
            }
            return temp
        }

        for result in results {
            #expect(abs(result - expectedValue) < 1e-6)
        }
    }
}
