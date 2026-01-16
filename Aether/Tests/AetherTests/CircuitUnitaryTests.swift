// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for circuit-to-unitary matrix conversion.
/// Validates gate expansion, matrix composition, and memory feasibility checks
/// for batched GPU evaluation infrastructure.
@Suite("CircuitUnitary Conversion")
struct CircuitUnitaryTests {
    @Test("Empty circuit produces identity matrix")
    func emptyCircuitIdentity() {
        let circuit = QuantumCircuit(qubits: 2)
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 4)
        #expect(unitary[0][0] == Complex(1, 0))
        #expect(unitary[1][1] == Complex(1, 0))
        #expect(unitary[2][2] == Complex(1, 0))
        #expect(unitary[3][3] == Complex(1, 0))
        #expect(unitary[0][1].magnitude < 1e-10)
        #expect(unitary[1][0].magnitude < 1e-10)
    }

    @Test("Single Hadamard gate produces correct unitary")
    func singleHadamardGate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let unitary = CircuitUnitary.unitary(for: circuit)

        let expected = 1.0 / sqrt(2.0)
        #expect(abs(unitary[0][0].real - expected) < 1e-10)
        #expect(abs(unitary[0][1].real - expected) < 1e-10)
        #expect(abs(unitary[1][0].real - expected) < 1e-10)
        #expect(abs(unitary[1][1].real + expected) < 1e-10)
    }

    @Test("Pauli-X gate on single qubit")
    func pauliXGate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary[0][0] == Complex(0, 0))
        #expect(unitary[0][1] == Complex(1, 0))
        #expect(unitary[1][0] == Complex(1, 0))
        #expect(unitary[1][1] == Complex(0, 0))
    }

    @Test("Pauli-Y gate on single qubit")
    func pauliYGate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliY, to: 0)
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(abs(unitary[0][0].real) < 1e-10)
        #expect(abs(unitary[0][1].imaginary + 1.0) < 1e-10)
        #expect(abs(unitary[1][0].imaginary - 1.0) < 1e-10)
        #expect(abs(unitary[1][1].real) < 1e-10)
    }

    @Test("Pauli-Z gate on single qubit")
    func pauliZGate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliZ, to: 0)
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary[0][0] == Complex(1, 0))
        #expect(unitary[1][1] == Complex(-1, 0))
        #expect(unitary[0][1].magnitude < 1e-10)
        #expect(unitary[1][0].magnitude < 1e-10)
    }

    @Test("Rotation gate produces correct unitary")
    func rotationGate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationX(.pi / 2), to: 0)
        let unitary = CircuitUnitary.unitary(for: circuit)

        let expected = 1.0 / sqrt(2.0)
        #expect(abs(unitary[0][0].real - expected) < 1e-10)
        #expect(abs(unitary[0][1].imaginary + expected) < 1e-10)
        #expect(abs(unitary[1][0].imaginary + expected) < 1e-10)
        #expect(abs(unitary[1][1].real - expected) < 1e-10)
    }

    @Test("CNOT gate on two qubits")
    func cnotGate() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 4)
        #expect(unitary[0][0] == Complex(1, 0))
        #expect(unitary[1][3] == Complex(1, 0))
        #expect(unitary[2][2] == Complex(1, 0))
        #expect(unitary[3][1] == Complex(1, 0))
        #expect(unitary[0][1].magnitude < 1e-10)
        #expect(unitary[1][1].magnitude < 1e-10)
    }

    @Test("CZ gate produces correct unitary")
    func czGate() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cz, to: [0, 1])
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary[0][0] == Complex(1, 0))
        #expect(unitary[1][1] == Complex(1, 0))
        #expect(unitary[2][2] == Complex(1, 0))
        #expect(unitary[3][3] == Complex(-1, 0))
    }

    @Test("SWAP gate swaps qubit amplitudes")
    func swapGate() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.swap, to: [0, 1])
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary[0][0] == Complex(1, 0))
        #expect(unitary[1][2] == Complex(1, 0))
        #expect(unitary[2][1] == Complex(1, 0))
        #expect(unitary[3][3] == Complex(1, 0))
    }

    @Test("Controlled rotation gate")
    func controlledRotation() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlledRotationZ(.pi), to: [0, 1])
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 4)
        #expect(abs(unitary[0][0].real - 1.0) < 1e-10)
        #expect(abs(unitary[2][2].real - 1.0) < 1e-10)
        #expect(abs(unitary[1][1].imaginary + 1.0) < 1e-10)
        #expect(abs(unitary[3][3].imaginary - 1.0) < 1e-10)
    }

    @Test("Toffoli gate on three qubits")
    func toffoliGate() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.toffoli, to: [0, 1, 2])
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 8)
        #expect(unitary[0][0] == Complex(1, 0))
        #expect(unitary[3][7] == Complex(1, 0))
        #expect(unitary[7][3] == Complex(1, 0))
    }

    @Test("Multiple gates compose correctly")
    func multipleGateComposition() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 4)
        let inv_sqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(unitary[0][0].real - inv_sqrt2) < 1e-10)
        #expect(abs(unitary[3][0].real - inv_sqrt2) < 1e-10)
    }

    @Test("Hadamard on second qubit of two-qubit system")
    func hadamardOnSecondQubit() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 1)
        let unitary = CircuitUnitary.unitary(for: circuit)

        let inv_sqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(unitary[0][0].real - inv_sqrt2) < 1e-10)
        #expect(abs(unitary[0][2].real - inv_sqrt2) < 1e-10)
        #expect(abs(unitary[2][0].real - inv_sqrt2) < 1e-10)
        #expect(abs(unitary[2][2].real + inv_sqrt2) < 1e-10)
    }

    @Test("Three-qubit circuit with mixed gates")
    func threeQubitMixedGates() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.pauliZ, to: 2)
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 8)
    }

    @Test("Custom single-qubit gate")
    func customSingleQubitGate() {
        let customMatrix: [[Complex<Double>]] = [
            [Complex(0.707, 0), Complex(0.707, 0)],
            [Complex(0.707, 0), Complex(-0.707, 0)],
        ]
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.customSingleQubit(matrix: customMatrix), to: 0)
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(abs(unitary[0][0].real - 0.707) < 1e-3)
        #expect(abs(unitary[0][1].real - 0.707) < 1e-3)
    }

    @Test("Memory estimation for different qubit counts")
    func memoryEstimation() {
        let mem8 = CircuitUnitary.memoryUsage(for: 8)
        let mem10 = CircuitUnitary.memoryUsage(for: 10)
        let mem12 = CircuitUnitary.memoryUsage(for: 12)

        #expect(mem8 > 0)
        #expect(mem10 > mem8)
        #expect(mem12 > mem10)
        #expect(mem8 < 2 * 1024 * 1024)
        #expect(mem10 > 15 * 1024 * 1024)
        #expect(mem12 > 200 * 1024 * 1024)
    }

    @Test("Feasibility check for reasonable qubit counts")
    func feasibilityReasonableQubits() {
        #expect(CircuitUnitary.canConvert(qubits: 8))
        #expect(CircuitUnitary.canConvert(qubits: 10))
        #expect(CircuitUnitary.canConvert(qubits: 12))
        #expect(CircuitUnitary.canConvert(qubits: 14))
    }

    @Test("Unitarity preserved after gate composition")
    func unitarityPreserved() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.rotationZ(.pi / 4), to: 0)

        let unitary = CircuitUnitary.unitary(for: circuit)
        let conjugateTranspose = MatrixUtilities.hermitianConjugate(unitary)
        let product = MatrixUtilities.matrixMultiply(conjugateTranspose, unitary)

        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                if i == j {
                    #expect(abs(product[i][j].real - 1.0) < 1e-10)
                    #expect(abs(product[i][j].imaginary) < 1e-10)
                } else {
                    #expect(abs(product[i][j].magnitude) < 1e-10)
                }
            }
        }
    }

    @Test("S gate produces correct phase")
    func sGatePhase() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.sGate, to: 0)
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary[0][0] == Complex(1, 0))
        #expect(abs(unitary[1][1].real) < 1e-10)
        #expect(abs(unitary[1][1].imaginary - 1.0) < 1e-10)
    }

    @Test("T gate produces correct phase")
    func tGatePhase() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.tGate, to: 0)
        let unitary = CircuitUnitary.unitary(for: circuit)

        let expected = 1.0 / sqrt(2.0)
        #expect(unitary[0][0] == Complex(1, 0))
        #expect(abs(unitary[1][1].real - expected) < 1e-10)
        #expect(abs(unitary[1][1].imaginary - expected) < 1e-10)
    }

    @Test("Sequential composition matches direct multiplication")
    func sequentialComposition() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliY, to: 0)
        let unitary = CircuitUnitary.unitary(for: circuit)

        let xGate = QuantumGate.pauliX.matrix()
        let yGate = QuantumGate.pauliY.matrix()
        let manual = MatrixUtilities.matrixMultiply(yGate, xGate)

        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                #expect(abs(unitary[i][j].real - manual[i][j].real) < 1e-10)
                #expect(abs(unitary[i][j].imaginary - manual[i][j].imaginary) < 1e-10)
            }
        }
    }

    @Test("Four-qubit circuit produces correct dimension")
    func fourQubitDimension() {
        var circuit = QuantumCircuit(qubits: 4)
        circuit.append(.hadamard, to: 0)
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 16)
        #expect(unitary[0].count == 16)
    }

    @Test("CH gate on two qubits")
    func controlledHadamard() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.ch, to: [0, 1])
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary[0][0] == Complex(1, 0))
        #expect(unitary[2][2] == Complex(1, 0))
        let inv_sqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(unitary[1][1].real - inv_sqrt2) < 1e-10)
        #expect(abs(unitary[3][1].real - inv_sqrt2) < 1e-10)
    }

    @Test("CY gate on two qubits")
    func controlledY() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cy, to: [0, 1])
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary[0][0] == Complex(1, 0))
        #expect(unitary[2][2] == Complex(1, 0))
        #expect(abs(unitary[3][1].imaginary - 1.0) < 1e-10)
        #expect(abs(unitary[1][3].imaginary + 1.0) < 1e-10)
    }

    @Test("Controlled phase gate")
    func controlledPhase() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlledPhase(.pi), to: [0, 1])
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(abs(unitary[3][3].real + 1.0) < 1e-10)
    }

    @Test("sqrt-SWAP gate")
    func sqrtSwap() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.sqrtSwap, to: [0, 1])
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 4)
    }

    @Test("U1 gate produces correct phase")
    func u1Gate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u1(lambda: .pi / 2), to: 0)
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary[0][0] == Complex(1, 0))
        #expect(abs(unitary[1][1].imaginary - 1.0) < 1e-10)
    }

    @Test("U2 gate with phases")
    func u2Gate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u2(phi: 0, lambda: .pi), to: 0)
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 2)
    }

    @Test("U3 gate with all parameters")
    func u3Gate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u3(theta: .pi / 2, phi: 0, lambda: .pi), to: 0)
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 2)
    }

    @Test("SX gate square root of X")
    func sxGate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.sx, to: 0)
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(abs(unitary[0][0].real - 0.5) < 1e-10)
        #expect(abs(unitary[0][1].real - 0.5) < 1e-10)
    }

    @Test("SY gate square root of Y")
    func syGate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.sy, to: 0)
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(abs(unitary[0][0].real - 0.5) < 1e-10)
    }
}
