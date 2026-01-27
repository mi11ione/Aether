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

    @Test("Two-qubit gate with control > target (reverse qubit ordering)")
    func twoQubitGateControlGreaterThanTarget() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [1, 0])
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 4)
        #expect(unitary[0][0] == Complex(1, 0))
        #expect(unitary[1][1] == Complex(1, 0))
        #expect(unitary[2][3] == Complex(1, 0))
        #expect(unitary[3][2] == Complex(1, 0))
    }

    @Test("Controlled gate via .controlled case produces correct unitary")
    func controlledGateUnitary() {
        var circuit = QuantumCircuit(qubits: 3)
        let controlledX = QuantumGate.controlled(gate: .pauliX, controls: [0, 1])
        circuit.append(controlledX, to: [0, 1, 2])
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 8, "Controlled gate unitary should have dimension 8 for 3 qubits")

        for i in 0 ..< 6 {
            #expect(
                abs(unitary[i][i].real - 1.0) < 1e-10,
                "Diagonal element \(i) should be 1 when controls not both set",
            )
        }

        #expect(
            abs(unitary[6][7].real - 1.0) < 1e-10,
            "Element [6][7] should be 1 for controlled-X flip when both controls are 1",
        )
        #expect(
            abs(unitary[7][6].real - 1.0) < 1e-10,
            "Element [7][6] should be 1 for controlled-X flip when both controls are 1",
        )
    }

    @Test("Controlled Hadamard gate produces correct unitary structure")
    func controlledHadamardUnitary() {
        var circuit = QuantumCircuit(qubits: 2)
        let controlledH = QuantumGate.controlled(gate: .hadamard, controls: [0])
        circuit.append(controlledH, to: [0, 1])
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 4, "Controlled-H unitary should have dimension 4 for 2 qubits")

        #expect(
            abs(unitary[0][0].real - 1.0) < 1e-10,
            "Element [0][0] should be 1 when control is 0",
        )
        #expect(
            abs(unitary[1][1].real - 1.0) < 1e-10,
            "Element [1][1] should be 1 when control is 0",
        )
    }

    @Test("Controlled rotation gate produces correct unitary")
    func controlledRotationUnitary() {
        var circuit = QuantumCircuit(qubits: 2)
        let controlledRy = QuantumGate.controlled(gate: .rotationY(.pi / 2), controls: [0])
        circuit.append(controlledRy, to: [0, 1])
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 4, "Controlled-Ry unitary should have dimension 4 for 2 qubits")

        #expect(
            abs(unitary[0][0].real - 1.0) < 1e-10,
            "Element [0][0] should be 1 when control is 0",
        )

        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(
            abs(unitary[2][2].real - invSqrt2) < 1e-10,
            "Element [2][2] should be cos(pi/4) when control is 1",
        )
    }

    @Test("customUnitary gate in circuit produces correct unitary")
    func customUnitaryInCircuit() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let hadamardMatrix: [[Complex<Double>]] = [
            [Complex(invSqrt2, 0), Complex(invSqrt2, 0)],
            [Complex(invSqrt2, 0), Complex(-invSqrt2, 0)],
        ]
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.customUnitary(matrix: hadamardMatrix), to: 0)
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(abs(unitary[0][0].real - invSqrt2) < 1e-10, "customUnitary [0][0] should match Hadamard")
        #expect(abs(unitary[0][1].real - invSqrt2) < 1e-10, "customUnitary [0][1] should match Hadamard")
        #expect(abs(unitary[1][0].real - invSqrt2) < 1e-10, "customUnitary [1][0] should match Hadamard")
        #expect(abs(unitary[1][1].real + invSqrt2) < 1e-10, "customUnitary [1][1] should match Hadamard")
    }

    @Test("Two-qubit customUnitary gate in circuit produces correct unitary")
    func twoQubitCustomUnitaryInCircuit() {
        let cnotMatrix: [[Complex<Double>]] = [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .zero, .one],
            [.zero, .zero, .one, .zero],
        ]
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.customUnitary(matrix: cnotMatrix), to: [0, 1])
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 4, "Two-qubit customUnitary should produce 4x4 unitary")
        #expect(abs(unitary[0][0].real - 1.0) < 1e-10, "customUnitary CNOT [0][0] should be 1")
        #expect(abs(unitary[1][1].real - 1.0) < 1e-10, "customUnitary CNOT [1][1] should be 1")
        #expect(abs(unitary[2][3].real - 1.0) < 1e-10, "customUnitary CNOT [2][3] should be 1")
        #expect(abs(unitary[3][2].real - 1.0) < 1e-10, "customUnitary CNOT [3][2] should be 1")
    }

    @Test("Three-qubit customUnitary gate in circuit produces correct unitary")
    func threeQubitCustomUnitaryInCircuit() {
        var toffoliMatrix: [[Complex<Double>]] = Array(
            repeating: Array(repeating: Complex<Double>.zero, count: 8),
            count: 8,
        )
        for i in 0 ..< 6 {
            toffoliMatrix[i][i] = .one
        }
        toffoliMatrix[6][7] = .one
        toffoliMatrix[7][6] = .one

        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.customUnitary(matrix: toffoliMatrix), to: [0, 1, 2])
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 8, "Three-qubit customUnitary should produce 8x8 unitary")
        #expect(abs(unitary[0][0].real - 1.0) < 1e-10, "customUnitary Toffoli [0][0] should be 1")
        #expect(abs(unitary[6][7].real - 1.0) < 1e-10, "customUnitary Toffoli [6][7] should be 1")
        #expect(abs(unitary[7][6].real - 1.0) < 1e-10, "customUnitary Toffoli [7][6] should be 1")
    }

    @Test("customUnitary gate in multi-qubit system expands correctly")
    func customUnitaryExpandsInLargerSystem() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let hadamardMatrix: [[Complex<Double>]] = [
            [Complex(invSqrt2, 0), Complex(invSqrt2, 0)],
            [Complex(invSqrt2, 0), Complex(-invSqrt2, 0)],
        ]
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.customUnitary(matrix: hadamardMatrix), to: 0)
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 4, "customUnitary on qubit 0 in 2-qubit system should produce 4x4 unitary")
        #expect(abs(unitary[0][0].real - invSqrt2) < 1e-10, "customUnitary expansion [0][0] should match")
        #expect(abs(unitary[0][1].real - invSqrt2) < 1e-10, "customUnitary expansion [0][1] should match")
    }

    @Test("CCZ gate produces correct 8x8 diagonal unitary with -1 at [7][7]")
    func cczGateUnitary() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.ccz, to: [0, 1, 2])
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 8, "CCZ unitary should be 8x8 for 3 qubits")
        #expect(unitary[0].count == 8, "CCZ unitary row should have 8 columns")

        for i in 0 ..< 8 {
            for j in 0 ..< 8 {
                if i == j, i != 7 {
                    #expect(
                        abs(unitary[i][j].real - 1.0) < 1e-10,
                        "CCZ unitary diagonal element [\(i)][\(i)] should be 1.0",
                    )
                    #expect(
                        abs(unitary[i][j].imaginary) < 1e-10,
                        "CCZ unitary diagonal element [\(i)][\(i)] imaginary part should be zero",
                    )
                } else if i == 7, j == 7 {
                    #expect(
                        abs(unitary[7][7].real + 1.0) < 1e-10,
                        "CCZ element [7][7] should be -1 (phase flip on |111>)",
                    )
                    #expect(
                        abs(unitary[7][7].imaginary) < 1e-10,
                        "CCZ element [7][7] imaginary should be 0",
                    )
                } else {
                    #expect(
                        abs(unitary[i][j].magnitude) < 1e-10,
                        "CCZ unitary off-diagonal element [\(i)][\(j)] should be zero",
                    )
                }
            }
        }
    }

    @Test("CCZ gate is unitary (U†U = I)")
    func cczGateUnitarity() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.ccz, to: [0, 1, 2])
        let unitary = CircuitUnitary.unitary(for: circuit)

        let conjugateTranspose = MatrixUtilities.hermitianConjugate(unitary)
        let product = MatrixUtilities.matrixMultiply(conjugateTranspose, unitary)

        for i in 0 ..< 8 {
            for j in 0 ..< 8 {
                if i == j {
                    #expect(
                        abs(product[i][j].real - 1.0) < 1e-10,
                        "CCZ U†U diagonal [\(i)][\(j)] should be 1",
                    )
                    #expect(
                        abs(product[i][j].imaginary) < 1e-10,
                        "CCZ U†U diagonal [\(i)][\(j)] imaginary should be 0",
                    )
                } else {
                    #expect(
                        abs(product[i][j].magnitude) < 1e-10,
                        "CCZ U†U off-diagonal [\(i)][\(j)] should be 0",
                    )
                }
            }
        }
    }

    @Test("CCZ gate combined with other gates in multi-qubit circuit")
    func cczGateComposition() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        circuit.append(.hadamard, to: 2)
        circuit.append(.ccz, to: [0, 1, 2])
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 8, "CCZ composition unitary should be 8x8")

        let conjugateTranspose = MatrixUtilities.hermitianConjugate(unitary)
        let product = MatrixUtilities.matrixMultiply(conjugateTranspose, unitary)

        for i in 0 ..< 8 {
            #expect(
                abs(product[i][i].real - 1.0) < 1e-10,
                "CCZ composition U†U diagonal [\(i)][\(i)] should be 1",
            )
        }
    }

    @Test("CCZ gate in 4-qubit system expands correctly to 16x16")
    func cczGateInFourQubitSystem() {
        var circuit = QuantumCircuit(qubits: 4)
        circuit.append(.ccz, to: [0, 1, 2])
        let unitary = CircuitUnitary.unitary(for: circuit)

        #expect(unitary.count == 16, "CCZ in 4-qubit system should produce 16x16 unitary")
        #expect(unitary[0].count == 16, "CCZ in 4-qubit system row should have 16 columns")

        #expect(
            abs(unitary[7][7].real + 1.0) < 1e-10,
            "CCZ should flip phase of |0111> in 4-qubit system",
        )
        #expect(
            abs(unitary[15][15].real + 1.0) < 1e-10,
            "CCZ should flip phase of |1111> in 4-qubit system",
        )

        #expect(
            abs(unitary[0][0].real - 1.0) < 1e-10,
            "CCZ should leave |0000> unchanged in 4-qubit system",
        )
        #expect(
            abs(unitary[8][8].real - 1.0) < 1e-10,
            "CCZ should leave |1000> unchanged in 4-qubit system",
        )
    }
}
