// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for extended two-qubit and controlled gates.
/// Validates CZ, CY, and CH behavior, including symmetry, superposition
/// creation, and unitarity across the extended gate set.
@Suite("Extended Gate Set")
struct ExtendedGateTests {
    @Test("CZ gate creates Bell state variant")
    func testCZGate() {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        circuit.append(.cz, to: [0, 1])
        circuit.append(.hadamard, to: 1)

        let state = circuit.execute()

        let expected00 = 0.5
        let expected11 = 0.5

        #expect(abs(state.probability(of: 0) - expected00) < 1e-10)
        #expect(abs(state.probability(of: 3) - expected11) < 1e-10)
    }

    @Test("CZ gate is symmetric")
    func testCZSymmetry() {
        var circuit1 = QuantumCircuit(numQubits: 2)
        circuit1.append(.hadamard, to: 0)
        circuit1.append(.cz, to: [0, 1])

        var circuit2 = QuantumCircuit(numQubits: 2)
        circuit2.append(.hadamard, to: 0)
        circuit2.append(.cz, to: [1, 0])

        let state1 = circuit1.execute()
        let state2 = circuit2.execute()

        #expect(state1 == state2)
    }

    @Test("CY gate matrix is unitary")
    func testCYUnitary() {
        let gate = QuantumGate.cy
        let matrix = gate.matrix()
        #expect(QuantumGate.isUnitary(matrix))
    }

    @Test("CH gate creates controlled superposition")
    func testCHGate() {
        var state = QuantumState(numQubits: 2)
        state.setAmplitude(1, to: .one)
        state.setAmplitude(0, to: .zero)

        state = GateApplication.apply(.ch, to: [0, 1], state: state)

        let prob0 = state.probability(of: 1)
        let prob1 = state.probability(of: 3)

        #expect(abs(prob0 - 0.5) < 1e-10)
        #expect(abs(prob1 - 0.5) < 1e-10)
    }
}

/// Test suite for IBM universal single-qubit gates.
/// Validates U1/U2/U3 behavior, including phase-only U1, superposition from U2,
/// universal expressivity of U3, and unitarity of parameterized matrices.
@Suite("IBM Universal Gates")
struct IDMUniversalGateTests {
    @Test("U1 gate is pure phase")
    func testU1Gate() {
        let lambda = Double.pi / 3.0
        let state = QuantumState(qubit: 1)
        let transformed = GateApplication.apply(.u1(lambda: lambda), to: 0, state: state)

        #expect(abs(transformed.probability(of: 1) - 1.0) < 1e-10)
    }

    @Test("U2 gate creates superposition")
    func testU2Gate() {
        let phi = 0.0
        let lambda = 0.0
        let state = QuantumState(qubit: 0)
        let transformed = GateApplication.apply(.u2(phi: phi, lambda: lambda), to: 0, state: state)

        #expect(abs(transformed.probability(of: 0) - 0.5) < 1e-10)
        #expect(abs(transformed.probability(of: 1) - 0.5) < 1e-10)
    }

    @Test("U3 gate can implement any single-qubit gate")
    func testU3Gate() {
        let state = QuantumState(qubit: 0)
        let transformed = GateApplication.apply(
            .u3(theta: .pi, phi: 0.0, lambda: .pi),
            to: 0,
            state: state
        )

        #expect(abs(transformed.probability(of: 1) - 1.0) < 1e-10)
    }

    @Test("U3 matrix is unitary")
    func testU3Unitary() {
        let gate = QuantumGate.u3(theta: 1.23, phi: 2.34, lambda: 3.45)
        let matrix = gate.matrix()
        #expect(QuantumGate.isUnitary(matrix))
    }
}

/// Test suite for square-root gates.
/// Verifies SX and SY square to their respective Pauli gates and that √SWAP
/// composes to SWAP while maintaining matrix unitarity.
@Suite("Square Root Gates")
struct SquareRootGatesTests {
    @Test("SX gate squared equals X")
    func testSXSquared() {
        var state = QuantumState(qubit: 0)
        state = GateApplication.apply(.sx, to: 0, state: state)
        state = GateApplication.apply(.sx, to: 0, state: state)

        #expect(abs(state.probability(of: 1) - 1.0) < 1e-10)
    }

    @Test("SY gate squared equals Y")
    func testSYSquared() {
        var state = QuantumState(qubit: 0)
        state = GateApplication.apply(.sy, to: 0, state: state)
        state = GateApplication.apply(.sy, to: 0, state: state)

        #expect(abs(state.probability(of: 1) - 1.0) < 1e-10)
    }

    @Test("√SWAP gate squared equals SWAP")
    func testSqrtSWAPSquared() {
        var state = QuantumState(numQubits: 2)
        state.setAmplitude(1, to: .one)
        state.setAmplitude(0, to: .zero)

        state = GateApplication.apply(.sqrtSwap, to: [0, 1], state: state)
        state = GateApplication.apply(.sqrtSwap, to: [0, 1], state: state)

        #expect(abs(state.probability(of: 2) - 1.0) < 1e-10)
    }

    @Test("√SWAP matrix is unitary")
    func testSqrtSWAPUnitary() {
        let gate = QuantumGate.sqrtSwap
        let matrix = gate.matrix()
        #expect(QuantumGate.isUnitary(matrix))
    }
}

/// Test suite for controlled rotation gates.
/// Validates CRx/CRy/CRz control semantics (no-op when control=0), correct
/// target rotation when control=1, and unitary matrix properties.
@Suite("Controlled Rotation Gates")
struct ControlledRotationGatesTests {
    @Test("CRx gate with control=0 is identity")
    func testCRxControlOff() {
        let state = QuantumState(numQubits: 2)
        let theta = Double.pi / 4.0
        let transformed = GateApplication.apply(
            .controlledRotationX(theta),
            to: [0, 1],
            state: state
        )

        #expect(transformed == state)
    }

    @Test("CRx gate with control=1 rotates target")
    func testCRxControlOn() {
        var state = QuantumState(numQubits: 2)
        state.setAmplitude(1, to: .one)
        state.setAmplitude(0, to: .zero)

        let theta = Double.pi
        state = GateApplication.apply(
            .controlledRotationX(theta),
            to: [0, 1],
            state: state
        )

        #expect(abs(state.probability(of: 3) - 1.0) < 1e-10)
    }

    @Test("CRy gate matrix is unitary")
    func testCRyUnitary() {
        let gate = QuantumGate.controlledRotationY(1.23)
        let matrix = gate.matrix()
        #expect(QuantumGate.isUnitary(matrix))
    }

    @Test("CRz gate applies phase rotation")
    func testCRzGate() {
        var state = QuantumState(numQubits: 2)
        state.setAmplitude(3, to: .one)
        state.setAmplitude(0, to: .zero)

        let theta = Double.pi / 2.0
        state = GateApplication.apply(
            .controlledRotationZ(theta),
            to: [0, 1],
            state: state
        )

        #expect(abs(state.probability(of: 3) - 1.0) < 1e-10)
    }
}

/// Test suite for custom unitary gates.
/// Ensures size checks and verifies successful
/// integration of custom gates into circuits.
@Suite("Custom Unitary Gates")
struct CustomUnitaryGatesTests {
    @Test("Custom single-qubit gate validates unitarity")
    func testCustomSingleQubitValidation() {
        let validMatrix: [[Complex<Double>]] = [
            [.zero, .one],
            [.one, .zero],
        ]

        let gate = QuantumGate.custom(matrix: validMatrix)
        #expect(gate.qubitsRequired == 1)
    }

    @Test("Custom two-qubit gate validates unitarity")
    func testCustomTwoQubitValidation() {
        let validMatrix: [[Complex<Double>]] = [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .zero, .one],
            [.zero, .zero, .one, .zero],
        ]

        let gate = QuantumGate.custom(matrix: validMatrix, control: 0, target: 1)
        #expect(gate.qubitsRequired == 2)
    }

    @Test("Custom gate works in circuit")
    func testCustomGateInCircuit() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let hadamardMatrix: [[Complex<Double>]] = [
            [Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0)],
            [Complex(invSqrt2, 0.0), Complex(-invSqrt2, 0.0)],
        ]

        let customH = QuantumGate.custom(matrix: hadamardMatrix)

        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(customH, to: 0)

        let state = circuit.execute()

        #expect(abs(state.probability(of: 0) - 0.5) < 1e-10)
        #expect(abs(state.probability(of: 1) - 0.5) < 1e-10)
    }
}

/// Test suite for multi-controlled gates.
/// Validates decompositions for C^n(Y) and C^n(Z), ensuring correct behavior
/// with multiple controls and expected output state probabilities.
@Suite("Multi-Controlled Gates")
struct MultiControlledGatesTests {
    @Test("Multi-controlled U with hadamard")
    func testMultiControlledUHadamard() {
        var circuit = QuantumCircuit(numQubits: 3)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 1)

        QuantumCircuit.appendMultiControlledU(to: &circuit, gate: .hadamard, controls: [0, 1], target: 2)

        let state = circuit.execute()

        #expect(abs(state.probability(of: 3) - 0.5) < 1e-10)
        #expect(abs(state.probability(of: 7) - 0.5) < 1e-10)
    }

    @Test("Multi-controlled U with arbitrary gate")
    func testMultiControlledUArbitrary() {
        var circuit = QuantumCircuit(numQubits: 3)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 1)

        QuantumCircuit.appendMultiControlledU(to: &circuit, gate: .tGate, controls: [0, 1], target: 2)

        let state = circuit.execute()

        #expect(abs(state.probability(of: 3) - 1.0) < 1e-10)
    }

    @Test("Multi-controlled U with pauliX")
    func testMultiControlledUPauliX() {
        var circuit = QuantumCircuit(numQubits: 3)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 1)

        QuantumCircuit.appendMultiControlledU(to: &circuit, gate: .pauliX, controls: [0, 1], target: 2)

        let state = circuit.execute()

        #expect(abs(state.probability(of: 7) - 1.0) < 1e-10)
    }

    @Test("Multi-controlled U with pauliY")
    func testMultiControlledUPauliY() {
        var circuit = QuantumCircuit(numQubits: 3)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 1)

        QuantumCircuit.appendMultiControlledU(to: &circuit, gate: .pauliY, controls: [0, 1], target: 2)

        let state = circuit.execute()

        #expect(abs(state.probability(of: 7) - 1.0) < 1e-10)
    }
}

/// Test suite for extended gate properties.
/// Confirms qubit requirements, gate validation rules, matrix unitarity, and
/// human-readable descriptions for newly added gates.
@Suite("Gate Property")
struct GatePropertyTests {
    @Test("All new gates have correct qubit requirements")
    func testQubitRequirements() {
        #expect(QuantumGate.sx.qubitsRequired == 1)
        #expect(QuantumGate.sy.qubitsRequired == 1)
        #expect(QuantumGate.u1(lambda: 0.5).qubitsRequired == 1)
        #expect(QuantumGate.u2(phi: 0.5, lambda: 0.5).qubitsRequired == 1)
        #expect(QuantumGate.u3(theta: 0.5, phi: 0.5, lambda: 0.5).qubitsRequired == 1)
        #expect(QuantumGate.cz.qubitsRequired == 2)
        #expect(QuantumGate.cy.qubitsRequired == 2)
        #expect(QuantumGate.ch.qubitsRequired == 2)
        #expect(QuantumGate.sqrtSwap.qubitsRequired == 2)
        #expect(QuantumGate.controlledRotationX(0.5).qubitsRequired == 2)
    }

    @Test("All new gate matrices are unitary")
    func testAllNewGatesUnitary() {
        let gates: [QuantumGate] = [
            .sx,
            .sy,
            .u1(lambda: 1.23),
            .u2(phi: 1.23, lambda: 2.34),
            .u3(theta: 1.23, phi: 2.34, lambda: 3.45),
            .cz,
            .cy,
            .ch,
            .sqrtSwap,
            .controlledRotationX(1.23),
            .controlledRotationY(1.23),
            .controlledRotationZ(1.23),
        ]

        for gate in gates {
            let matrix = gate.matrix()
            #expect(QuantumGate.isUnitary(matrix), "Gate \(gate) is not unitary")
        }
    }

    @Test("Gate descriptions are correct")
    func testGateDescriptions() {
        #expect(QuantumGate.sx.description == "SX")
        #expect(QuantumGate.sy.description == "SY")
        #expect(QuantumGate.cz.description == "CZ")
        #expect(QuantumGate.cy.description == "CY")
        #expect(QuantumGate.ch.description == "CH")
        #expect(QuantumGate.u1(lambda: 1.234).description.contains("U1"))
        #expect(QuantumGate.u2(phi: 1.5, lambda: 2.5).description.contains("U2"))
        #expect(QuantumGate.u3(theta: 1.0, phi: 2.0, lambda: 3.0).description.contains("U3"))
        #expect(QuantumGate.sqrtSwap.description.contains("√SWAP"))
        #expect(QuantumGate.cnot.description.contains("CNOT"))
        #expect(QuantumGate.controlledRotationX(1.5).description.contains("CRx"))
        #expect(QuantumGate.controlledRotationY(1.5).description.contains("CRy"))
        #expect(QuantumGate.controlledRotationZ(1.5).description.contains("CRz"))

        let customSingleMatrix: [[Complex<Double>]] = [
            [.zero, .one],
            [.one, .zero],
        ]
        let customSingleGate = QuantumGate.custom(matrix: customSingleMatrix)
        #expect(customSingleGate.description == "CustomU(2x2)")

        let customTwoMatrix: [[Complex<Double>]] = [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .zero, .one],
            [.zero, .zero, .one, .zero],
        ]
        let customTwoGate = QuantumGate.custom(matrix: customTwoMatrix, control: 0, target: 1)
        #expect(customTwoGate.description.contains("CustomU"))
    }
}

/// Test suite for numerical precision and composition.
/// Ensures normalization preservation under extended gates and verifies IBM
/// gate composition equivalences within numerical tolerance.
@Suite("Numerical Precision")
struct NumericalPrecisionTests {
    @Test("Gates preserve normalization")
    func testNormalizationPreservation() {
        var state = QuantumState(numQubits: 2)
        state.setAmplitude(0, to: Complex(0.6, 0.0))
        state.setAmplitude(1, to: Complex(0.0, 0.8))

        let gates: [QuantumGate] = [
            .sx,
            .sy,
            .u3(theta: 1.23, phi: 2.34, lambda: 3.45),
        ]

        for gate in gates {
            let transformed = GateApplication.apply(gate, to: 0, state: state)
            #expect(transformed.isNormalized(), "Gate \(gate) did not preserve normalization")
        }
    }

    @Test("IBM gates compose correctly")
    func testIBMGateComposition() {
        var state1 = QuantumState(qubit: 0)
        state1 = GateApplication.apply(.u1(lambda: 1.23), to: 0, state: state1)

        var state2 = QuantumState(qubit: 0)
        state2 = GateApplication.apply(.u3(theta: 0.0, phi: 0.0, lambda: 1.23), to: 0, state: state2)

        #expect(abs(state1.probability(of: 0) - state2.probability(of: 0)) < 1e-10)
    }
}
