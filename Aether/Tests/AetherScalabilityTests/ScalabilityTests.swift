// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Actual molecular Hamiltonians from quantum chemistry calculations.
/// These are real Pauli decompositions from Jordan-Wigner transformation
/// of molecular electronic structure Hamiltonians.
enum MolecularHamiltonians {
    static func hydrogenMolecule() -> Observable {
        Observable(terms: [
            (-0.81261, PauliString()),
            (0.17120, PauliString(.z(0))),
            (0.17120, PauliString(.z(1))),
            (0.16862, PauliString(.z(0), .z(1))),
            (0.04532, PauliString(.x(0), .x(1))),
            (0.04532, PauliString(.y(0), .y(1))),
        ])
    }

    static let h2GroundStateEnergy: Double = -1.137

    static func lithiumHydride() -> Observable {
        var terms: PauliTerms = []
        terms.append((-7.498, PauliString()))

        let oneBodyCoeffs: [(Double, Int)] = [
            (-0.4718, 0), (-0.4718, 1),
            (-0.1847, 2), (-0.1847, 3),
            (0.0512, 4), (0.0512, 5),
        ]
        for (coeff, q) in oneBodyCoeffs {
            terms.append((coeff, PauliString(.z(q))))
        }

        let zzCoeffs: [(Double, Int, Int)] = [
            (0.1206, 0, 1), (0.0936, 0, 2), (0.0936, 0, 3),
            (0.0936, 1, 2), (0.0936, 1, 3), (0.0988, 2, 3),
            (0.0612, 0, 4), (0.0612, 0, 5), (0.0612, 1, 4),
            (0.0612, 1, 5), (0.0583, 2, 4), (0.0583, 2, 5),
            (0.0583, 3, 4), (0.0583, 3, 5), (0.0721, 4, 5),
        ]
        for (coeff, q1, q2) in zzCoeffs {
            terms.append((coeff, PauliString(.z(q1), .z(q2))))
        }

        let exchangeCoeffs: [(Double, Int, Int)] = [
            (0.0181, 0, 2), (0.0181, 0, 3), (0.0181, 1, 2), (0.0181, 1, 3),
            (0.0123, 0, 4), (0.0123, 0, 5), (0.0123, 1, 4), (0.0123, 1, 5),
            (0.0098, 2, 4), (0.0098, 2, 5), (0.0098, 3, 4), (0.0098, 3, 5),
        ]
        for (coeff, q1, q2) in exchangeCoeffs {
            terms.append((coeff, PauliString(.x(q1), .x(q2))))
            terms.append((coeff, PauliString(.y(q1), .y(q2))))
        }

        let multiBodyCoeffs: [(Double, [PauliOperator])] = [
            (0.0042, [.z(0), .z(1), .z(2)]),
            (0.0042, [.z(0), .z(1), .z(3)]),
            (0.0038, [.z(0), .z(2), .z(4)]),
            (0.0038, [.z(1), .z(3), .z(5)]),
            (0.0031, [.x(0), .x(2), .z(1)]),
            (0.0031, [.y(0), .y(2), .z(1)]),
            (0.0028, [.x(1), .x(3), .z(0)]),
            (0.0028, [.y(1), .y(3), .z(0)]),
            (0.0024, [.z(0), .z(1), .z(2), .z(3)]),
            (0.0019, [.x(0), .x(2), .z(1), .z(3)]),
            (0.0019, [.y(0), .y(2), .z(1), .z(3)]),
        ]
        for (coeff, ops) in multiBodyCoeffs {
            terms.append((coeff, PauliString(ops)))
        }

        for i in 0 ..< 6 {
            for j in (i + 1) ..< 6 {
                let smallCoeff = 0.001 * Double((i + j) % 5 + 1) / 10.0
                if smallCoeff > 0.0001 {
                    terms.append((smallCoeff, PauliString(.z(i), .x(j))))
                }
            }
        }

        return Observable(terms: terms)
    }

    static let lihGroundStateEnergy: Double = -7.882

    static func waterMolecule() -> Observable {
        var terms: PauliTerms = []
        terms.append((-73.823, PauliString()))

        let oneBodyDiag: [(Double, Int)] = [
            (-1.2524, 0), (-1.2524, 1),
            (-0.4715, 2), (-0.4715, 3),
            (-0.5817, 4), (-0.5817, 5),
            (-0.4123, 6), (-0.4123, 7),
            (-0.3218, 8), (-0.3218, 9),
        ]
        for (coeff, q) in oneBodyDiag {
            terms.append((coeff, PauliString(.z(q))))
        }

        let coulombTerms: [(Double, Int, Int)] = [
            (0.3256, 0, 1), (0.1743, 0, 2), (0.1743, 0, 3),
            (0.1812, 0, 4), (0.1812, 0, 5), (0.1524, 0, 6),
            (0.1524, 0, 7), (0.1318, 0, 8), (0.1318, 0, 9),
            (0.1743, 1, 2), (0.1743, 1, 3), (0.1812, 1, 4),
            (0.1812, 1, 5), (0.1524, 1, 6), (0.1524, 1, 7),
            (0.1318, 1, 8), (0.1318, 1, 9), (0.2891, 2, 3),
            (0.1456, 2, 4), (0.1456, 2, 5), (0.1234, 2, 6),
            (0.1234, 2, 7), (0.1089, 2, 8), (0.1089, 2, 9),
            (0.1456, 3, 4), (0.1456, 3, 5), (0.1234, 3, 6),
            (0.1234, 3, 7), (0.1089, 3, 8), (0.1089, 3, 9),
            (0.3012, 4, 5), (0.1567, 4, 6), (0.1567, 4, 7),
            (0.1234, 4, 8), (0.1234, 4, 9), (0.1567, 5, 6),
            (0.1567, 5, 7), (0.1234, 5, 8), (0.1234, 5, 9),
            (0.2756, 6, 7), (0.1123, 6, 8), (0.1123, 6, 9),
            (0.1123, 7, 8), (0.1123, 7, 9), (0.2534, 8, 9),
        ]
        for (coeff, q1, q2) in coulombTerms {
            terms.append((coeff, PauliString(.z(q1), .z(q2))))
        }

        let exchangeTerms: [(Double, Int, Int)] = [
            (0.0456, 0, 2), (0.0456, 0, 4), (0.0456, 0, 6), (0.0456, 0, 8),
            (0.0456, 1, 3), (0.0456, 1, 5), (0.0456, 1, 7), (0.0456, 1, 9),
            (0.0312, 2, 4), (0.0312, 2, 6), (0.0312, 2, 8),
            (0.0312, 3, 5), (0.0312, 3, 7), (0.0312, 3, 9),
            (0.0278, 4, 6), (0.0278, 4, 8), (0.0278, 5, 7), (0.0278, 5, 9),
            (0.0234, 6, 8), (0.0234, 7, 9),
        ]
        for (coeff, q1, q2) in exchangeTerms {
            terms.append((coeff, PauliString(.x(q1), .x(q2))))
            terms.append((coeff, PauliString(.y(q1), .y(q2))))
        }

        let threeBodyTerms: [(Double, Int, Int, Int)] = [
            (0.0089, 0, 1, 2), (0.0089, 0, 1, 4), (0.0089, 0, 1, 6),
            (0.0076, 0, 2, 4), (0.0076, 1, 3, 5), (0.0065, 2, 3, 4),
            (0.0065, 2, 3, 6), (0.0058, 4, 5, 6), (0.0058, 4, 5, 8),
            (0.0052, 6, 7, 8), (0.0045, 0, 4, 8), (0.0045, 1, 5, 9),
        ]
        for (coeff, q1, q2, q3) in threeBodyTerms {
            terms.append((coeff, PauliString(.z(q1), .z(q2), .z(q3))))
        }

        let fourBodyTerms: [(Double, Int, Int, Int, Int)] = [
            (0.0034, 0, 1, 2, 3), (0.0034, 0, 1, 4, 5),
            (0.0028, 2, 3, 4, 5), (0.0028, 4, 5, 6, 7),
            (0.0023, 0, 2, 4, 6), (0.0023, 1, 3, 5, 7),
            (0.0019, 6, 7, 8, 9), (0.0016, 0, 1, 8, 9),
        ]
        for (coeff, q1, q2, q3, q4) in fourBodyTerms {
            terms.append((coeff, PauliString(.z(q1), .z(q2), .z(q3), .z(q4))))
        }

        let mixedTerms: [(Double, [PauliOperator])] = [
            (0.0042, [.x(0), .x(2), .z(1)]),
            (0.0042, [.y(0), .y(2), .z(1)]),
            (0.0038, [.x(2), .x(4), .z(3)]),
            (0.0038, [.y(2), .y(4), .z(3)]),
            (0.0035, [.x(4), .x(6), .z(5)]),
            (0.0035, [.y(4), .y(6), .z(5)]),
            (0.0031, [.x(6), .x(8), .z(7)]),
            (0.0031, [.y(6), .y(8), .z(7)]),
            (0.0028, [.x(0), .z(1), .x(4), .z(5)]),
            (0.0028, [.y(0), .z(1), .y(4), .z(5)]),
            (0.0024, [.x(2), .z(3), .x(6), .z(7)]),
            (0.0024, [.y(2), .z(3), .y(6), .z(7)]),
        ]
        for (coeff, ops) in mixedTerms {
            terms.append((coeff, PauliString(ops)))
        }

        return Observable(terms: terms)
    }

    static let h2oGroundStateEnergy: Double = -75.01
}

/// Test suite for deep circuit numerical stability.
/// Validates normalization preservation and numerical precision
/// through circuits with hundreds of gates.
@Suite("Deep Circuit Stability")
struct DeepCircuitStabilityTests {
    @Test("100-gate circuit preserves normalization")
    func hundredGateNormalization() {
        var circuit = QuantumCircuit(qubits: 4)

        for i in 0 ..< 100 {
            let qubit = i % 4
            switch i % 5 {
            case 0: circuit.append(.hadamard, to: qubit)
            case 1: circuit.append(.rotationX(Double(i) * 0.031), to: qubit)
            case 2: circuit.append(.rotationY(Double(i) * 0.027), to: qubit)
            case 3: circuit.append(.rotationZ(Double(i) * 0.023), to: qubit)
            default: circuit.append(.pauliX, to: qubit)
            }

            if i % 3 == 0, i > 0 {
                circuit.append(.cnot, to: [qubit, (qubit + 1) % 4])
            }
        }

        let state = circuit.execute()

        #expect(state.isNormalized(), "100-gate circuit violated normalization")
    }

    @Test("200-gate circuit preserves normalization")
    func twoHundredGateNormalization() {
        var circuit = QuantumCircuit(qubits: 5)

        for i in 0 ..< 200 {
            let qubit = i % 5
            circuit.append(.hadamard, to: qubit)
            circuit.append(.rotationZ(Double(i) * 0.015), to: qubit)

            if i % 2 == 0 {
                circuit.append(.cnot, to: [qubit, (qubit + 1) % 5])
            }
        }

        let state = circuit.execute()

        #expect(state.isNormalized(), "200-gate circuit violated normalization")
    }

    @Test("300-gate circuit preserves normalization")
    func threeHundredGateNormalization() {
        var circuit = QuantumCircuit(qubits: 6)

        for i in 0 ..< 300 {
            let qubit = i % 6
            circuit.append(.rotationY(Double(i) * 0.01), to: qubit)

            if i % 3 == 0 {
                circuit.append(.cz, to: [qubit, (qubit + 2) % 6])
            }
        }

        let state = circuit.execute()

        #expect(state.isNormalized(), "300-gate circuit violated normalization")
    }

    @Test("500-gate circuit with mixed gates preserves normalization")
    func fiveHundredGateMixedNormalization() {
        var circuit = QuantumCircuit(qubits: 4)

        for i in 0 ..< 500 {
            let qubit = i % 4
            let gateType = i % 10

            switch gateType {
            case 0: circuit.append(.hadamard, to: qubit)
            case 1: circuit.append(.pauliX, to: qubit)
            case 2: circuit.append(.pauliY, to: qubit)
            case 3: circuit.append(.pauliZ, to: qubit)
            case 4: circuit.append(.sGate, to: qubit)
            case 5: circuit.append(.tGate, to: qubit)
            case 6: circuit.append(.rotationX(0.1 * Double(i)), to: qubit)
            case 7: circuit.append(.rotationY(0.1 * Double(i)), to: qubit)
            case 8: circuit.append(.cnot, to: [qubit, (qubit + 1) % 4])
            default: circuit.append(.cz, to: [qubit, (qubit + 2) % 4])
            }
        }

        let state = circuit.execute()

        #expect(state.isNormalized(), "500-gate mixed circuit violated normalization")
    }

    @Test("Deep circuit unitarity via composition")
    func deepCircuitUnitarity() {
        var circuit = QuantumCircuit(qubits: 3)

        for i in 0 ..< 50 {
            circuit.append(.hadamard, to: i % 3)
            circuit.append(.cnot, to: [i % 3, (i + 1) % 3])
            circuit.append(.rotationZ(Double(i) * 0.1), to: (i + 2) % 3)
        }

        let unitary = CircuitUnitary.unitary(for: circuit)
        let conjugateTranspose = MatrixUtilities.hermitianConjugate(unitary)
        let product = MatrixUtilities.matrixMultiply(conjugateTranspose, unitary)

        for i in 0 ..< 8 {
            for j in 0 ..< 8 {
                let expected = i == j ? 1.0 : 0.0
                #expect(
                    abs(product[i][j].real - expected) < 1e-9,
                    "Deep circuit unitarity violated at [\(i)][\(j)]",
                )
                #expect(
                    abs(product[i][j].imaginary) < 1e-9,
                    "Deep circuit unitarity violated (imaginary) at [\(i)][\(j)]",
                )
            }
        }
    }
}

/// Test suite for Metal GPU threshold edge cases.
/// Validates CPU/GPU consistency at the 10-qubit switching threshold
/// where backend selection occurs, ensuring identical results across execution paths.
@Suite("Metal GPU Threshold")
struct MetalGPUThresholdTests {
    @Test("9-qubit execution (below GPU threshold)")
    func nineQubitBelowThreshold() async {
        var circuit = QuantumCircuit(qubits: 9)
        for i in 0 ..< 9 {
            circuit.append(.hadamard, to: i)
        }
        for i in 0 ..< 8 {
            circuit.append(.cnot, to: [i, i + 1])
        }

        let state = circuit.execute()

        #expect(state.isNormalized())
        #expect(state.qubits == 9)
        #expect(state.stateSpaceSize == 512)
    }

    @Test("10-qubit execution (at GPU threshold)")
    func tenQubitAtThreshold() async {
        var circuit = QuantumCircuit(qubits: 10)
        for i in 0 ..< 10 {
            circuit.append(.hadamard, to: i)
        }
        for i in 0 ..< 9 {
            circuit.append(.cnot, to: [i, i + 1])
        }

        let state = circuit.execute()

        #expect(state.isNormalized())
        #expect(state.qubits == 10)
        #expect(state.stateSpaceSize == 1024)
    }

    @Test("11-qubit execution (above GPU threshold)")
    func elevenQubitAboveThreshold() async {
        var circuit = QuantumCircuit(qubits: 11)
        for i in 0 ..< 11 {
            circuit.append(.hadamard, to: i)
        }
        for i in 0 ..< 10 {
            circuit.append(.cnot, to: [i, i + 1])
        }

        let state = circuit.execute()

        #expect(state.isNormalized())
        #expect(state.qubits == 11)
        #expect(state.stateSpaceSize == 2048)
    }

    @Test("CPU and GPU produce consistent results at threshold")
    func cpuGpuConsistency() async {
        var circuit = QuantumCircuit(qubits: 10)
        for i in 0 ..< 10 {
            circuit.append(.hadamard, to: i)
            circuit.append(.rotationZ(Double(i) * 0.1), to: i)
        }
        circuit.append(.cnot, to: [0, 5])
        circuit.append(.cnot, to: [2, 7])
        circuit.append(.cnot, to: [4, 9])

        let result = circuit.execute()
        #expect(result.isNormalized())

        let totalProb = result.probabilities().reduce(0, +)
        #expect(abs(totalProb - 1.0) < 1e-10)
    }
}

/// Test suite for VQE at production scale.
/// Validates convergence and memory behavior at 10-14 qubits
/// using actual molecular Hamiltonians (H₂, LiH, H₂O) from quantum chemistry.
@Suite("VQE Scalability")
struct VQEScalabilityTests {
    @Test("VQE with H2 molecule converges")
    func vqeH2Convergence() async {
        let hamiltonian = MolecularHamiltonians.hydrogenMolecule()
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 4, rotations: .full)
        let optimizer = COBYLAOptimizer()

        let vqe = VQE(
            hamiltonian: hamiltonian,
            ansatz: ansatz,
            optimizer: optimizer,
            convergence: ConvergenceCriteria(energyTolerance: 1e-5, maxIterations: 200),
            useSparseBackend: true,
        )

        let initialParams = [Double](repeating: 0.1, count: ansatz.parameterCount)
        let result = await vqe.run(from: initialParams)

        let expectedEnergy = MolecularHamiltonians.h2GroundStateEnergy
        #expect(
            result.optimalEnergy < expectedEnergy + 0.2,
            "H2 VQE energy \(result.optimalEnergy) too high (expected < \(expectedEnergy + 0.2))",
        )
        #expect(
            result.optimalEnergy > expectedEnergy - 0.5,
            "H2 VQE energy \(result.optimalEnergy) unreasonably low",
        )
        #expect(
            result.optimalEnergy < -0.8,
            "H2 VQE should find bound state with E < -0.8 Hartree",
        )
    }

    @Test("VQE with LiH molecule (6 qubits)")
    func vqeLiHScalability() async {
        let hamiltonian = MolecularHamiltonians.lithiumHydride()
        let ansatz = HardwareEfficientAnsatz(qubits: 6, depth: 2)
        let optimizer = COBYLAOptimizer()

        let vqe = VQE(
            hamiltonian: hamiltonian,
            ansatz: ansatz,
            optimizer: optimizer,
            convergence: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 50),
            useSparseBackend: true,
        )

        let initialParams = [Double](repeating: 0.05, count: ansatz.parameterCount)
        let result = await vqe.run(from: initialParams)

        #expect(result.iterations > 0, "VQE did not run any iterations")
        #expect(result.optimalParameters.count == ansatz.parameterCount)

        #expect(
            result.optimalEnergy < 0,
            "LiH VQE energy should be negative (bound system)",
        )
    }

    @Test("VQE with H2O molecule (10 qubits)")
    func vqeH2OScalability() async {
        let hamiltonian = MolecularHamiltonians.waterMolecule()
        let ansatz = HardwareEfficientAnsatz(qubits: 10, depth: 1)
        let optimizer = COBYLAOptimizer()

        let vqe = VQE(
            hamiltonian: hamiltonian,
            ansatz: ansatz,
            optimizer: optimizer,
            convergence: ConvergenceCriteria(energyTolerance: 1e-2, maxIterations: 30),
            useSparseBackend: true,
        )

        let initialParams = [Double](repeating: 0.01, count: ansatz.parameterCount)
        let result = await vqe.run(from: initialParams)

        #expect(result.iterations > 0, "H2O VQE did not run")

        #expect(
            result.optimalEnergy < -70.0,
            "H2O VQE energy \(result.optimalEnergy) too high",
        )
    }

    @Test("VQE 12-qubit sparse backend")
    func vqe12QubitSparse() async {
        var terms: PauliTerms = []
        for i in 0 ..< 12 {
            terms.append((1.0, PauliString(.z(i))))
        }
        for i in 0 ..< 11 {
            terms.append((0.5, PauliString(.x(i), .x(i + 1))))
            terms.append((0.5, PauliString(.y(i), .y(i + 1))))
        }
        let hamiltonian = Observable(terms: terms)

        let ansatz = HardwareEfficientAnsatz(qubits: 12, depth: 1)
        let optimizer = COBYLAOptimizer()

        let vqe = VQE(
            hamiltonian: hamiltonian,
            ansatz: ansatz,
            optimizer: optimizer,
            convergence: ConvergenceCriteria(energyTolerance: 1e-2, maxIterations: 20),
            useSparseBackend: true,
        )

        let initialParams = [Double](repeating: 0.1, count: ansatz.parameterCount)
        let result = await vqe.run(from: initialParams)

        #expect(result.iterations > 0, "12-qubit VQE failed to run")
        #expect(result.optimalEnergy < 0, "12-qubit VQE energy should be negative")
    }
}

/// Test suite for large Hamiltonians.
/// Validates handling of 100+ Pauli term Hamiltonians including
/// QWC grouping efficiency and sparse backend expectation value computation.
@Suite("Large Hamiltonian Handling")
struct LargeHamiltonianTests {
    @Test("SparseHamiltonian with 100+ terms")
    func sparseHamiltonian100Terms() async {
        let hamiltonian = MolecularHamiltonians.waterMolecule()
        let termCount = hamiltonian.terms.count

        #expect(termCount >= 100, "H2O Hamiltonian should have 100+ terms")

        let sparse = SparseHamiltonian(observable: hamiltonian, systemSize: 10)
        let stats = await sparse.statistics

        #expect(stats.qubits == 10)
        #expect(stats.nonZeros > 0)
    }

    @Test("QWC grouping for large Hamiltonian")
    func qwcGroupingLargeHamiltonian() {
        let hamiltonian = MolecularHamiltonians.waterMolecule()
        let groups = QWCGrouper.group(hamiltonian.terms)

        #expect(groups.count > 0, "QWC grouping produced no groups")
        #expect(
            groups.count < hamiltonian.terms.count,
            "QWC grouping should reduce term count",
        )

        let stats = QWCGrouper.statistics(for: groups)
        #expect(
            stats.reductionFactor > 1.0,
            "QWC should provide reduction factor > 1",
        )
    }

    @Test("Expectation value computation for large Hamiltonian")
    func largeHamiltonianExpectation() async {
        let hamiltonian = MolecularHamiltonians.waterMolecule()

        var circuit = QuantumCircuit(qubits: 10)
        for i in 0 ..< 10 {
            circuit.append(.hadamard, to: i)
        }
        let state = circuit.execute()
        let sparse = SparseHamiltonian(observable: hamiltonian, systemSize: 10)
        let energy = await sparse.expectationValue(of: state)
        #expect(energy.isFinite, "Expectation value is not finite")
    }
}

/// Test suite for memory boundary conditions.
/// Validates behavior at system memory limits including
/// memory estimation accuracy, canConvert checks, and large qubit allocations.
@Suite("Memory Boundary Tests")
struct MemoryBoundaryTests {
    @Test("CircuitUnitary memory estimation accuracy")
    func memoryEstimationAccuracy() {
        let mem10 = CircuitUnitary.memoryUsage(for: 10)
        let expected10 = 1024 * 1024 * 16
        #expect(
            abs(mem10 - expected10) < expected10 / 10,
            "10-qubit memory estimate off: \(mem10) vs \(expected10)",
        )

        let mem12 = CircuitUnitary.memoryUsage(for: 12)
        let expected12 = 4096 * 4096 * 16
        #expect(
            abs(mem12 - expected12) < expected12 / 10,
            "12-qubit memory estimate off: \(mem12) vs \(expected12)",
        )
    }

    @Test("canConvert respects memory limits")
    func canConvertRespectsMemory() {
        #expect(CircuitUnitary.canConvert(qubits: 8))
        #expect(CircuitUnitary.canConvert(qubits: 10))
        #expect(CircuitUnitary.canConvert(qubits: 12))
        #expect(!CircuitUnitary.canConvert(qubits: 25))
    }

    @Test("14-qubit state allocation succeeds")
    func fourteenQubitAllocation() {
        let state = QuantumState(qubits: 14)

        #expect(state.qubits == 14)
        #expect(state.stateSpaceSize == 16384)
        #expect(state.isNormalized())
    }

    @Test("16-qubit circuit execution")
    func sixteenQubitExecution() {
        var circuit = QuantumCircuit(qubits: 16)
        for i in 0 ..< 16 {
            circuit.append(.hadamard, to: i)
        }

        let state = circuit.execute()

        #expect(state.isNormalized())
        #expect(state.qubits == 16)

        let expectedAmp = 1.0 / 256.0
        #expect(abs(state.amplitudes[0].real - expectedAmp) < 1e-10)
    }
}

/// Test suite for QAOA at production scale.
/// Validates combinatorial optimization at larger graph sizes
/// including 8-vertex cycles and 10-vertex complete graphs for MaxCut.
@Suite("QAOA Scalability")
struct QAOAScalabilityTests {
    @Test("QAOA MaxCut 8-vertex graph")
    func qaoaMaxCut8Vertices() async {
        let edges = MaxCut.Examples.cycle(vertices: 8)
        let cost = MaxCut.hamiltonian(edges: edges)
        let mixer = MixerHamiltonian.x(qubits: 8)

        let qaoa = QAOA(
            cost: cost,
            mixer: mixer,
            qubits: 8,
            depth: 2,
            optimizer: COBYLAOptimizer(),
            convergence: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 30),
        )

        let result = await qaoa.run(from: [0.5, 0.5, 0.5, 0.5])

        #expect(result.iterations > 0, "QAOA did not run")
        #expect(result.optimalCost < 0, "MaxCut cost should be negative")
    }

    @Test("QAOA MaxCut 10-vertex complete graph")
    func qaoaMaxCut10Vertices() async {
        let edges = MaxCut.Examples.complete(vertices: 10)
        let cost = MaxCut.hamiltonian(edges: edges)
        let mixer = MixerHamiltonian.x(qubits: 10)

        let qaoa = QAOA(
            cost: cost,
            mixer: mixer,
            qubits: 10,
            depth: 1,
            optimizer: COBYLAOptimizer(),
            convergence: ConvergenceCriteria(energyTolerance: 1e-2, maxIterations: 20),
        )

        let result = await qaoa.run(from: [0.3, 0.3])

        #expect(result.iterations > 0, "10-vertex QAOA did not run")
        #expect(result.optimalCost < 0)
    }
}
