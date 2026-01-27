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
    func cZGate() {
        var circuit = QuantumCircuit(qubits: 2)
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
    func cZSymmetry() {
        var circuit1 = QuantumCircuit(qubits: 2)
        circuit1.append(.hadamard, to: 0)
        circuit1.append(.cz, to: [0, 1])

        var circuit2 = QuantumCircuit(qubits: 2)
        circuit2.append(.hadamard, to: 0)
        circuit2.append(.cz, to: [1, 0])

        let state1 = circuit1.execute()
        let state2 = circuit2.execute()

        #expect(state1 == state2)
    }

    @Test("CY gate matrix is unitary")
    func cYUnitary() {
        let gate = QuantumGate.cy
        let matrix = gate.matrix()
        #expect(QuantumGate.isUnitary(matrix))
    }

    @Test("CH gate creates controlled superposition")
    func cHGate() {
        var state = QuantumState(qubits: 2)
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
    func u1Gate() {
        let lambda = Double.pi / 3.0
        let state = QuantumState(qubit: 1)
        let transformed = GateApplication.apply(.u1(lambda: lambda), to: 0, state: state)

        #expect(abs(transformed.probability(of: 1) - 1.0) < 1e-10)
    }

    @Test("U2 gate creates superposition")
    func u2Gate() {
        let phi = 0.0
        let lambda = 0.0
        let state = QuantumState(qubit: 0)
        let transformed = GateApplication.apply(.u2(phi: phi, lambda: lambda), to: 0, state: state)

        #expect(abs(transformed.probability(of: 0) - 0.5) < 1e-10)
        #expect(abs(transformed.probability(of: 1) - 0.5) < 1e-10)
    }

    @Test("U3 gate can implement any single-qubit gate")
    func u3Gate() {
        let state = QuantumState(qubit: 0)
        let transformed = GateApplication.apply(
            .u3(theta: .pi, phi: 0.0, lambda: .pi),
            to: 0,
            state: state,
        )

        #expect(abs(transformed.probability(of: 1) - 1.0) < 1e-10)
    }

    @Test("U3 matrix is unitary")
    func u3Unitary() {
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
    func sXSquared() {
        var state = QuantumState(qubit: 0)
        state = GateApplication.apply(.sx, to: 0, state: state)
        state = GateApplication.apply(.sx, to: 0, state: state)

        #expect(abs(state.probability(of: 1) - 1.0) < 1e-10)
    }

    @Test("SY gate squared equals Y")
    func sYSquared() {
        var state = QuantumState(qubit: 0)
        state = GateApplication.apply(.sy, to: 0, state: state)
        state = GateApplication.apply(.sy, to: 0, state: state)

        #expect(abs(state.probability(of: 1) - 1.0) < 1e-10)
    }

    @Test("√SWAP gate squared equals SWAP")
    func sqrtSWAPSquared() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(1, to: .one)
        state.setAmplitude(0, to: .zero)

        state = GateApplication.apply(.sqrtSwap, to: [0, 1], state: state)
        state = GateApplication.apply(.sqrtSwap, to: [0, 1], state: state)

        #expect(abs(state.probability(of: 2) - 1.0) < 1e-10)
    }

    @Test("√SWAP matrix is unitary")
    func sqrtSWAPUnitary() {
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
    func cRxControlOff() {
        let state = QuantumState(qubits: 2)
        let theta = Double.pi / 4.0
        let transformed = GateApplication.apply(
            .controlledRotationX(theta),
            to: [0, 1],
            state: state,
        )

        #expect(transformed == state)
    }

    @Test("CRx gate with control=1 rotates target")
    func cRxControlOn() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(1, to: .one)
        state.setAmplitude(0, to: .zero)

        let theta = Double.pi
        state = GateApplication.apply(
            .controlledRotationX(theta),
            to: [0, 1],
            state: state,
        )

        #expect(abs(state.probability(of: 3) - 1.0) < 1e-10)
    }

    @Test("CRy gate matrix is unitary")
    func cRyUnitary() {
        let gate = QuantumGate.controlledRotationY(1.23)
        let matrix = gate.matrix()
        #expect(QuantumGate.isUnitary(matrix))
    }

    @Test("CRz gate applies phase rotation")
    func cRzGate() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(3, to: .one)
        state.setAmplitude(0, to: .zero)

        let theta = Double.pi / 2.0
        state = GateApplication.apply(
            .controlledRotationZ(theta),
            to: [0, 1],
            state: state,
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
    func customSingleQubitValidation() {
        let validMatrix: [[Complex<Double>]] = [
            [.zero, .one],
            [.one, .zero],
        ]

        let gate = QuantumGate.custom(matrix: validMatrix)
        #expect(gate.qubitsRequired == 1)
    }

    @Test("Custom two-qubit gate validates unitarity")
    func customTwoQubitValidation() {
        let validMatrix: [[Complex<Double>]] = [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .zero, .one],
            [.zero, .zero, .one, .zero],
        ]

        let gate = QuantumGate.custom(matrix: validMatrix)
        #expect(gate.qubitsRequired == 2)
    }

    @Test("Custom gate works in circuit")
    func customGateInCircuit() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let hadamardMatrix: [[Complex<Double>]] = [
            [Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0)],
            [Complex(invSqrt2, 0.0), Complex(-invSqrt2, 0.0)],
        ]

        let customH = QuantumGate.custom(matrix: hadamardMatrix)

        var circuit = QuantumCircuit(qubits: 1)
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
    func multiControlledUHadamard() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 1)

        QuantumCircuit.appendMultiControlledU(to: &circuit, gate: .hadamard, controls: [0, 1], target: 2)

        let state = circuit.execute()

        #expect(abs(state.probability(of: 3) - 0.5) < 1e-10)
        #expect(abs(state.probability(of: 7) - 0.5) < 1e-10)
    }

    @Test("Multi-controlled U with arbitrary gate")
    func multiControlledUArbitrary() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 1)

        QuantumCircuit.appendMultiControlledU(to: &circuit, gate: .tGate, controls: [0, 1], target: 2)

        let state = circuit.execute()

        #expect(abs(state.probability(of: 3) - 1.0) < 1e-10)
    }

    @Test("Multi-controlled U with pauliX")
    func multiControlledUPauliX() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 1)

        QuantumCircuit.appendMultiControlledU(to: &circuit, gate: .pauliX, controls: [0, 1], target: 2)

        let state = circuit.execute()

        #expect(abs(state.probability(of: 7) - 1.0) < 1e-10)
    }

    @Test("Multi-controlled U with pauliY")
    func multiControlledUPauliY() {
        var circuit = QuantumCircuit(qubits: 3)
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
    func qubitRequirements() {
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
    func allNewGatesUnitary() {
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
    func gateDescriptions() {
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
        let customTwoGate = QuantumGate.custom(matrix: customTwoMatrix)
        #expect(customTwoGate.description.contains("CustomU"))
    }
}

/// Test suite for numerical precision and composition.
/// Ensures normalization preservation under extended gates and verifies IBM
/// gate composition equivalences within numerical tolerance.
@Suite("Numerical Precision")
struct NumericalPrecisionTests {
    @Test("Gates preserve normalization")
    func normalizationPreservation() {
        var state = QuantumState(qubits: 2)
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
    func iBMGateComposition() {
        var state1 = QuantumState(qubit: 0)
        state1 = GateApplication.apply(.u1(lambda: 1.23), to: 0, state: state1)

        var state2 = QuantumState(qubit: 0)
        state2 = GateApplication.apply(.u3(theta: 0.0, phi: 0.0, lambda: 1.23), to: 0, state: state2)

        #expect(abs(state1.probability(of: 0) - state2.probability(of: 0)) < 1e-10)
    }
}

/// Test suite for CCZ gate matrix and algebraic properties.
/// Validates unitarity, Hermiticity, self-inverse behavior, diagonal structure,
/// and correct diagonal entries for the three-qubit controlled-controlled-Z gate.
@Suite("CCZ Gate Properties")
struct CCZGatePropertiesTests {
    @Test("CCZ gate requires 3 qubits")
    func cczRequiresThreeQubits() {
        #expect(QuantumGate.ccz.qubitsRequired == 3, "CCZ is a three-qubit gate and must require exactly 3 qubits")
    }

    @Test("CCZ gate matrix is unitary")
    func cczMatrixIsUnitary() {
        let matrix = QuantumGate.ccz.matrix()
        #expect(QuantumGate.isUnitary(matrix), "CCZ matrix must satisfy U†U = I to preserve quantum probability")
    }

    @Test("CCZ gate matrix is 8x8")
    func cczMatrixIs8x8() {
        let matrix = QuantumGate.ccz.matrix()
        #expect(matrix.count == 8, "CCZ matrix must have 8 rows for a 3-qubit gate (2^3 = 8)")
        #expect(matrix.allSatisfy { $0.count == 8 }, "CCZ matrix must have 8 columns in every row for a 3-qubit gate")
    }

    @Test("CCZ gate is Hermitian (self-adjoint)")
    func cczIsHermitian() {
        #expect(QuantumGate.ccz.isHermitian, "CCZ must be Hermitian since it is a real diagonal matrix with entries +/-1")
    }

    @Test("CCZ gate is self-inverse: CCZ times CCZ equals identity")
    func cczIsSelfInverse() {
        let ccz = QuantumGate.ccz.matrix()
        let product = QuantumGate.matrixMultiply(ccz, ccz)
        #expect(QuantumGate.isIdentityMatrix(product), "CCZ squared must equal identity because CCZ is its own inverse")
    }

    @Test("CCZ matrix is diagonal with all off-diagonal elements zero")
    func cczMatrixIsDiagonal() {
        let matrix = QuantumGate.ccz.matrix()
        for i in 0 ..< 8 {
            for j in 0 ..< 8 {
                if i != j {
                    #expect(abs(matrix[i][j].magnitude) < 1e-10, "CCZ off-diagonal element [\(i)][\(j)] must be zero since CCZ is a diagonal gate")
                }
            }
        }
    }

    @Test("CCZ matrix has correct diagonal: all 1s except [7][7] which is -1")
    func cczCorrectDiagonal() {
        let matrix = QuantumGate.ccz.matrix()
        for i in 0 ..< 7 {
            #expect(abs(matrix[i][i].real - 1.0) < 1e-10, "CCZ diagonal element [\(i)][\(i)] must be 1 for basis states where not all three qubits are |1>")
            #expect(abs(matrix[i][i].imaginary) < 1e-10, "CCZ diagonal element [\(i)][\(i)] imaginary part must be 0")
        }
        #expect(abs(matrix[7][7].real - -1.0) < 1e-10, "CCZ diagonal element [7][7] must be -1 for the |111> basis state")
        #expect(abs(matrix[7][7].imaginary) < 1e-10, "CCZ diagonal element [7][7] imaginary part must be 0")
    }
}

/// Test suite for CCZ gate application on quantum states.
/// Validates phase-flip behavior on all computational basis states, superposition
/// normalization, self-inverse on states, qubit symmetry, and Toffoli equivalence.
@Suite("CCZ Gate Application")
struct CCZGateApplicationTests {
    @Test("CCZ on |000> gives |000> with no phase flip")
    func cczOn000() {
        let state = QuantumState(qubits: 3)
        let result = GateApplication.apply(.ccz, to: [0, 1, 2], state: state)
        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "CCZ must leave |000> unchanged since not all qubits are |1>")
    }

    @Test("CCZ on |001> gives |001>")
    func cczOn001() {
        var state = QuantumState(qubits: 3)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(1, to: .one)
        let result = GateApplication.apply(.ccz, to: [0, 1, 2], state: state)
        #expect(abs(result.probability(of: 1) - 1.0) < 1e-10, "CCZ must leave |001> unchanged since only one qubit is |1>")
    }

    @Test("CCZ on |010> gives |010>")
    func cczOn010() {
        var state = QuantumState(qubits: 3)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(2, to: .one)
        let result = GateApplication.apply(.ccz, to: [0, 1, 2], state: state)
        #expect(abs(result.probability(of: 2) - 1.0) < 1e-10, "CCZ must leave |010> unchanged since only one qubit is |1>")
    }

    @Test("CCZ on |011> gives |011>")
    func cczOn011() {
        var state = QuantumState(qubits: 3)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(3, to: .one)
        let result = GateApplication.apply(.ccz, to: [0, 1, 2], state: state)
        #expect(abs(result.probability(of: 3) - 1.0) < 1e-10, "CCZ must leave |011> unchanged since only two qubits are |1>")
    }

    @Test("CCZ on |100> gives |100>")
    func cczOn100() {
        var state = QuantumState(qubits: 3)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(4, to: .one)
        let result = GateApplication.apply(.ccz, to: [0, 1, 2], state: state)
        #expect(abs(result.probability(of: 4) - 1.0) < 1e-10, "CCZ must leave |100> unchanged since only one qubit is |1>")
    }

    @Test("CCZ on |101> gives |101>")
    func cczOn101() {
        var state = QuantumState(qubits: 3)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(5, to: .one)
        let result = GateApplication.apply(.ccz, to: [0, 1, 2], state: state)
        #expect(abs(result.probability(of: 5) - 1.0) < 1e-10, "CCZ must leave |101> unchanged since only two qubits are |1>")
    }

    @Test("CCZ on |110> gives |110>")
    func cczOn110() {
        var state = QuantumState(qubits: 3)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(6, to: .one)
        let result = GateApplication.apply(.ccz, to: [0, 1, 2], state: state)
        #expect(abs(result.probability(of: 6) - 1.0) < 1e-10, "CCZ must leave |110> unchanged since only two qubits are |1>")
    }

    @Test("CCZ on |111> gives -|111> with phase flip")
    func cczOn111() {
        var state = QuantumState(qubits: 3)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(7, to: .one)
        let result = GateApplication.apply(.ccz, to: [0, 1, 2], state: state)
        #expect(abs(result.probability(of: 7) - 1.0) < 1e-10, "CCZ on |111> must preserve probability at index 7")
        let amp = result.amplitude(of: 7)
        #expect(abs(amp.real - -1.0) < 1e-10, "CCZ on |111> must flip the amplitude sign to -1")
        #expect(abs(amp.imaginary) < 1e-10, "CCZ on |111> must produce a purely real amplitude")
    }

    @Test("CCZ preserves normalization on superposition state")
    func cczPreservesNormalization() {
        var state = QuantumState(qubits: 3)
        state.setAmplitude(0, to: Complex(0.5, 0.0))
        state.setAmplitude(3, to: Complex(0.5, 0.0))
        state.setAmplitude(5, to: Complex(0.5, 0.0))
        state.setAmplitude(7, to: Complex(0.5, 0.0))
        let result = GateApplication.apply(.ccz, to: [0, 1, 2], state: state)
        #expect(result.isNormalized(), "CCZ must preserve normalization since it is a unitary diagonal gate")
    }

    @Test("CCZ applied twice returns to original state (self-inverse on states)")
    func cczTwiceIsIdentityOnState() {
        var state = QuantumState(qubits: 3)
        state.setAmplitude(0, to: Complex(0.5, 0.0))
        state.setAmplitude(3, to: Complex(0.0, 0.5))
        state.setAmplitude(5, to: Complex(0.5, 0.0))
        state.setAmplitude(7, to: Complex(0.0, 0.5))
        let once = GateApplication.apply(.ccz, to: [0, 1, 2], state: state)
        let twice = GateApplication.apply(.ccz, to: [0, 1, 2], state: once)
        for i in 0 ..< 8 {
            let origAmp = state.amplitude(of: i)
            let finalAmp = twice.amplitude(of: i)
            #expect(abs(origAmp.real - finalAmp.real) < 1e-10, "CCZ applied twice must restore real part of amplitude at index \(i)")
            #expect(abs(origAmp.imaginary - finalAmp.imaginary) < 1e-10, "CCZ applied twice must restore imaginary part of amplitude at index \(i)")
        }
    }

    @Test("CCZ is symmetric in all 3 qubits")
    func cczIsSymmetric() {
        var state = QuantumState(qubits: 3)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(7, to: .one)
        let result012 = GateApplication.apply(.ccz, to: [0, 1, 2], state: state)
        let result120 = GateApplication.apply(.ccz, to: [1, 2, 0], state: state)
        let result201 = GateApplication.apply(.ccz, to: [2, 0, 1], state: state)
        for i in 0 ..< 8 {
            let a012 = result012.amplitude(of: i)
            let a120 = result120.amplitude(of: i)
            let a201 = result201.amplitude(of: i)
            #expect(abs(a012.real - a120.real) < 1e-10, "CCZ must be symmetric: [0,1,2] vs [1,2,0] differ at index \(i) real part")
            #expect(abs(a012.imaginary - a120.imaginary) < 1e-10, "CCZ must be symmetric: [0,1,2] vs [1,2,0] differ at index \(i) imaginary part")
            #expect(abs(a012.real - a201.real) < 1e-10, "CCZ must be symmetric: [0,1,2] vs [2,0,1] differ at index \(i) real part")
            #expect(abs(a012.imaginary - a201.imaginary) < 1e-10, "CCZ must be symmetric: [0,1,2] vs [2,0,1] differ at index \(i) imaginary part")
        }
    }

    @Test("CCZ equivalence: H on qubit 2 then Toffoli then H on qubit 2 equals CCZ")
    func cczEquivalenceWithToffoli() {
        var state = QuantumState(qubits: 3)
        state.setAmplitude(0, to: Complex(0.5, 0.0))
        state.setAmplitude(3, to: Complex(0.0, 0.5))
        state.setAmplitude(5, to: Complex(0.5, 0.0))
        state.setAmplitude(7, to: Complex(0.0, 0.5))

        let cczResult = GateApplication.apply(.ccz, to: [0, 1, 2], state: state)

        var decomposedState = state
        decomposedState = GateApplication.apply(.hadamard, to: 2, state: decomposedState)
        decomposedState = GateApplication.apply(.toffoli, to: [0, 1, 2], state: decomposedState)
        decomposedState = GateApplication.apply(.hadamard, to: 2, state: decomposedState)

        for i in 0 ..< 8 {
            let cczAmp = cczResult.amplitude(of: i)
            let decAmp = decomposedState.amplitude(of: i)
            #expect(abs(cczAmp.real - decAmp.real) < 1e-10, "CCZ and H-Toffoli-H decomposition must agree at index \(i) real part")
            #expect(abs(cczAmp.imaginary - decAmp.imaginary) < 1e-10, "CCZ and H-Toffoli-H decomposition must agree at index \(i) imaginary part")
        }
    }
}

/// Test suite for GlobalPhase gate matrix and algebraic properties.
/// Validates unitarity, identity at zero angle, non-Hermiticity,
/// inverse composition, and scalar-times-identity matrix structure.
@Suite("GlobalPhase Gate Properties")
struct GlobalPhaseGatePropertiesTests {
    @Test("GlobalPhase gate requires 1 qubit")
    func globalPhaseRequiresOneQubit() {
        #expect(QuantumGate.globalPhase(.pi / 4).qubitsRequired == 1, "GlobalPhase is a single-qubit gate and must require exactly 1 qubit")
    }

    @Test("GlobalPhase gate matrix is unitary for multiple angles")
    func globalPhaseMatrixIsUnitary() {
        let angles = [0.0, Double.pi / 4, Double.pi / 2, Double.pi, 3 * Double.pi / 2, 1.234]
        for angle in angles {
            let matrix = QuantumGate.globalPhase(angle).matrix()
            #expect(QuantumGate.isUnitary(matrix), "GlobalPhase(\(angle)) matrix must be unitary to preserve quantum probability")
        }
    }

    @Test("GlobalPhase(0) equals identity matrix")
    func globalPhaseZeroIsIdentity() {
        let gpMatrix = QuantumGate.globalPhase(0.0).matrix()
        let idMatrix = QuantumGate.identity.matrix()
        #expect(QuantumGate.matricesEqual(gpMatrix, idMatrix), "GlobalPhase(0) must equal the identity matrix since e^(i*0) = 1")
    }

    @Test("GlobalPhase is not Hermitian for non-zero angle")
    func globalPhaseIsNotHermitian() {
        #expect(!QuantumGate.globalPhase(.pi / 4).isHermitian, "GlobalPhase(pi/4) must not be Hermitian since e^(i*pi/4) is not real")
    }

    @Test("GlobalPhase inverse: GP(phi) times GP(-phi) equals identity")
    func globalPhaseInverseComposition() {
        let phi = Double.pi / 3
        let gp = QuantumGate.globalPhase(phi).matrix()
        let gpInv = QuantumGate.globalPhase(-phi).matrix()
        let product = QuantumGate.matrixMultiply(gp, gpInv)
        #expect(QuantumGate.isIdentityMatrix(product), "GlobalPhase(phi) times GlobalPhase(-phi) must equal identity")
    }

    @Test("GlobalPhase matrix is scalar times identity: [[e^(i*phi), 0], [0, e^(i*phi)]]")
    func globalPhaseMatrixIsScalarTimesIdentity() {
        let phi = Double.pi / 5
        let matrix = QuantumGate.globalPhase(phi).matrix()
        let expectedReal = cos(phi)
        let expectedImag = sin(phi)
        #expect(abs(matrix[0][0].real - expectedReal) < 1e-10, "GlobalPhase [0][0] real part must be cos(phi)")
        #expect(abs(matrix[0][0].imaginary - expectedImag) < 1e-10, "GlobalPhase [0][0] imaginary part must be sin(phi)")
        #expect(abs(matrix[1][1].real - expectedReal) < 1e-10, "GlobalPhase [1][1] real part must be cos(phi)")
        #expect(abs(matrix[1][1].imaginary - expectedImag) < 1e-10, "GlobalPhase [1][1] imaginary part must be sin(phi)")
        #expect(abs(matrix[0][1].magnitude) < 1e-10, "GlobalPhase [0][1] must be zero for scalar-times-identity structure")
        #expect(abs(matrix[1][0].magnitude) < 1e-10, "GlobalPhase [1][0] must be zero for scalar-times-identity structure")
    }
}

/// Test suite for GlobalPhase gate application on quantum states.
/// Validates phase multiplication on basis states and superpositions,
/// probability preservation, inverse cancellation, and multi-qubit behavior.
@Suite("GlobalPhase Gate Application")
struct GlobalPhaseGateApplicationTests {
    @Test("GlobalPhase(0) on |0> gives |0>")
    func globalPhaseZeroOn0() {
        let state = QuantumState(qubits: 1)
        let result = GateApplication.apply(.globalPhase(0.0), to: 0, state: state)
        let amp = result.amplitude(of: 0)
        #expect(abs(amp.real - 1.0) < 1e-10, "GlobalPhase(0) on |0> must leave the amplitude at 1+0i")
        #expect(abs(amp.imaginary) < 1e-10, "GlobalPhase(0) on |0> must produce zero imaginary part")
    }

    @Test("GlobalPhase(pi) on |0> gives -|0>")
    func globalPhasePiOn0() {
        let state = QuantumState(qubits: 1)
        let result = GateApplication.apply(.globalPhase(.pi), to: 0, state: state)
        let amp = result.amplitude(of: 0)
        #expect(abs(amp.real - -1.0) < 1e-10, "GlobalPhase(pi) on |0> must flip the amplitude to -1")
        #expect(abs(amp.imaginary) < 1e-10, "GlobalPhase(pi) on |0> must produce zero imaginary part")
    }

    @Test("GlobalPhase(pi/2) on |0> gives i|0>")
    func globalPhaseHalfPiOn0() {
        let state = QuantumState(qubits: 1)
        let result = GateApplication.apply(.globalPhase(.pi / 2), to: 0, state: state)
        let amp = result.amplitude(of: 0)
        #expect(abs(amp.real) < 1e-10, "GlobalPhase(pi/2) on |0> must have zero real part since e^(i*pi/2) = i")
        #expect(abs(amp.imaginary - 1.0) < 1e-10, "GlobalPhase(pi/2) on |0> must have imaginary part equal to 1")
    }

    @Test("GlobalPhase preserves normalization")
    func globalPhasePreservesNormalization() {
        var state = QuantumState(qubits: 1)
        state.setAmplitude(0, to: Complex(1.0 / sqrt(2.0), 0.0))
        state.setAmplitude(1, to: Complex(0.0, 1.0 / sqrt(2.0)))
        let result = GateApplication.apply(.globalPhase(1.234), to: 0, state: state)
        #expect(result.isNormalized(), "GlobalPhase must preserve normalization since it is a unitary gate")
    }

    @Test("GlobalPhase preserves all probabilities since it only changes phases")
    func globalPhasePreservesProbabilities() {
        var state = QuantumState(qubits: 1)
        state.setAmplitude(0, to: Complex(0.6, 0.0))
        state.setAmplitude(1, to: Complex(0.0, 0.8))
        let result = GateApplication.apply(.globalPhase(2.718), to: 0, state: state)
        #expect(abs(result.probability(of: 0) - state.probability(of: 0)) < 1e-10, "GlobalPhase must preserve |0> probability since it multiplies all amplitudes by the same phase")
        #expect(abs(result.probability(of: 1) - state.probability(of: 1)) < 1e-10, "GlobalPhase must preserve |1> probability since it multiplies all amplitudes by the same phase")
    }

    @Test("GlobalPhase(phi) followed by GlobalPhase(-phi) equals identity")
    func globalPhaseAndInverseIsIdentity() {
        var state = QuantumState(qubits: 1)
        state.setAmplitude(0, to: Complex(0.6, 0.0))
        state.setAmplitude(1, to: Complex(0.0, 0.8))
        let phased = GateApplication.apply(.globalPhase(1.5), to: 0, state: state)
        let restored = GateApplication.apply(.globalPhase(-1.5), to: 0, state: phased)
        for i in 0 ..< 2 {
            let origAmp = state.amplitude(of: i)
            let resAmp = restored.amplitude(of: i)
            #expect(abs(origAmp.real - resAmp.real) < 1e-10, "GlobalPhase(phi) then GlobalPhase(-phi) must restore real part of amplitude at index \(i)")
            #expect(abs(origAmp.imaginary - resAmp.imaginary) < 1e-10, "GlobalPhase(phi) then GlobalPhase(-phi) must restore imaginary part of amplitude at index \(i)")
        }
    }

    @Test("GlobalPhase on multi-qubit state multiplies all amplitudes by e^(i*phi)")
    func globalPhaseOnMultiQubitState() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: Complex(0.5, 0.0))
        state.setAmplitude(1, to: Complex(0.0, 0.5))
        state.setAmplitude(2, to: Complex(0.5, 0.0))
        state.setAmplitude(3, to: Complex(0.0, 0.5))
        let phi = Double.pi / 6
        let result = GateApplication.apply(.globalPhase(phi), to: 0, state: state)
        let cosP = cos(phi)
        let sinP = sin(phi)
        for i in 0 ..< 4 {
            let origAmp = state.amplitude(of: i)
            let resAmp = result.amplitude(of: i)
            let expectedReal = origAmp.real * cosP - origAmp.imaginary * sinP
            let expectedImag = origAmp.real * sinP + origAmp.imaginary * cosP
            #expect(abs(resAmp.real - expectedReal) < 1e-10, "GlobalPhase on 2-qubit state must multiply amplitude at index \(i) real part by e^(i*phi)")
            #expect(abs(resAmp.imaginary - expectedImag) < 1e-10, "GlobalPhase on 2-qubit state must multiply amplitude at index \(i) imaginary part by e^(i*phi)")
        }
    }
}

/// Test suite for Reset gate enum properties.
/// Validates qubit count, Hermiticity, parameterization status,
/// parameter extraction, and description of the non-unitary Reset operation.
@Suite("Reset Gate Properties")
struct ResetGatePropertiesTests {
    @Test("Reset operation targets 1 qubit")
    func resetRequiresOneQubit() {
        let resetOp = CircuitOperation.reset(qubit: 0)
        #expect(resetOp.qubits.count == 1, "Reset is a single-qubit non-unitary operation and must target exactly 1 qubit")
    }

    @Test("Reset operation is not unitary")
    func resetIsNotUnitary() {
        let resetOp = CircuitOperation.reset(qubit: 0)
        #expect(!resetOp.isUnitary, "Reset is a non-unitary irreversible operation and cannot be unitary")
    }

    @Test("Reset operation is not parameterized")
    func resetIsNotParameterized() {
        let resetOp = CircuitOperation.reset(qubit: 0)
        #expect(!resetOp.isParameterized, "Reset has no symbolic parameters and must not be parameterized")
    }

    @Test("Reset operation has no parameters")
    func resetHasNoParameters() {
        let resetOp = CircuitOperation.reset(qubit: 0)
        let params = resetOp.parameters()
        #expect(params.isEmpty, "Reset operation parameters() must return an empty set since it takes no angle or symbolic parameters")
    }

    @Test("Reset operation description contains 'reset'")
    func resetDescription() {
        let resetOp = CircuitOperation.reset(qubit: 0)
        #expect(resetOp.description.contains("reset"), "Reset operation description must contain the string 'reset' for human-readable output")
    }
}

/// Test suite for Reset gate application on quantum states.
/// Validates deterministic projection to |0>, behavior on basis states,
/// superpositions, multi-qubit systems, entangled states, and idempotency.
@Suite("Reset Gate Application")
struct ResetGateApplicationTests {
    @Test("Reset on |0> gives |0> (already in ground state)")
    func resetOnZeroState() {
        let state = QuantumState(qubit: 0)
        let result = GateApplication.applyReset(qubit: 0, state: state)
        let amp0 = result.amplitude(of: 0)
        let amp1 = result.amplitude(of: 1)
        #expect(abs(amp0.real - 1.0) < 1e-10, "Reset on |0> must leave the |0> amplitude at 1.0 since qubit is already in ground state")
        #expect(abs(amp0.imaginary) < 1e-10, "Reset on |0> must produce zero imaginary part for |0> amplitude")
        #expect(abs(amp1.real) < 1e-10, "Reset on |0> must leave the |1> amplitude at 0.0")
        #expect(abs(amp1.imaginary) < 1e-10, "Reset on |0> must leave the |1> amplitude imaginary part at 0.0")
    }

    @Test("Reset on |1> gives |0> (flip to ground state)")
    func resetOnOneState() {
        let state = QuantumState(qubit: 1)
        let result = GateApplication.applyReset(qubit: 0, state: state)
        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "Reset on |1> must project to |0> with probability 1.0")
        #expect(abs(result.probability(of: 1)) < 1e-10, "Reset on |1> must leave |1> with probability 0.0 after projection to ground state")
    }

    @Test("Reset on (|0> + |1>)/sqrt(2) projects qubit to |0> with probability 1.0")
    func resetOnPlusSuperposition() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let state = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0),
        ])
        let result = GateApplication.applyReset(qubit: 0, state: state)
        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "Reset on |+> must project qubit to |0> with certainty regardless of initial superposition")
        #expect(abs(result.probability(of: 1)) < 1e-10, "Reset on |+> must leave |1> with zero probability after deterministic projection")
    }

    @Test("Reset on (|0> - |1>)/sqrt(2) projects qubit to |0> with probability 1.0")
    func resetOnMinusSuperposition() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let state = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0.0), Complex(-invSqrt2, 0.0),
        ])
        let result = GateApplication.applyReset(qubit: 0, state: state)
        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "Reset on |-> must project qubit to |0> with certainty since reset ignores phase information")
        #expect(abs(result.probability(of: 1)) < 1e-10, "Reset on |-> must leave |1> with zero probability after deterministic projection")
    }

    @Test("Reset preserves normalization")
    func resetPreservesNormalization() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let state = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0.0), Complex(0.0, invSqrt2),
        ])
        let result = GateApplication.applyReset(qubit: 0, state: state)
        #expect(result.isNormalized(), "Reset must preserve state normalization by renormalizing after projection to the |0> subspace")
    }

    @Test("Reset is idempotent: Reset followed by Reset gives same result as single Reset")
    func resetIsIdempotent() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let state = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0),
        ])
        let once = GateApplication.applyReset(qubit: 0, state: state)
        let twice = GateApplication.applyReset(qubit: 0, state: once)
        for i in 0 ..< 2 {
            let onceAmp = once.amplitude(of: i)
            let twiceAmp = twice.amplitude(of: i)
            #expect(abs(onceAmp.real - twiceAmp.real) < 1e-10, "Reset applied twice must match single Reset at index \(i) real part since qubit is already |0>")
            #expect(abs(onceAmp.imaginary - twiceAmp.imaginary) < 1e-10, "Reset applied twice must match single Reset at index \(i) imaginary part since qubit is already |0>")
        }
    }

    @Test("Reset on qubit 0 of 2-qubit |00> state: no change")
    func resetQubit0Of00State() {
        let state = QuantumState(qubits: 2)
        let result = GateApplication.applyReset(qubit: 0, state: state)
        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "Reset on qubit 0 of |00> must leave state unchanged since qubit 0 is already |0>")
        #expect(abs(result.probability(of: 1)) < 1e-10, "Reset on qubit 0 of |00> must keep |01> probability at zero")
        #expect(abs(result.probability(of: 2)) < 1e-10, "Reset on qubit 0 of |00> must keep |10> probability at zero")
        #expect(abs(result.probability(of: 3)) < 1e-10, "Reset on qubit 0 of |00> must keep |11> probability at zero")
    }

    @Test("Reset on qubit 0 of 2-qubit state with qubit 0 = |1>: qubit 0 becomes |0>")
    func resetQubit0WithQubit0InOneState() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(1, to: .one)
        let result = GateApplication.applyReset(qubit: 0, state: state)
        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "Reset on qubit 0 of |01> must project qubit 0 to |0>, yielding |00> with probability 1.0")
        #expect(abs(result.probability(of: 1)) < 1e-10, "Reset on qubit 0 of |01> must leave |01> with zero probability after qubit 0 is reset")
    }

    @Test("Reset on qubit 1 of 2-qubit state: only affects qubit 1")
    func resetQubit1OnlyAffectsQubit1() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(2, to: .one)
        let result = GateApplication.applyReset(qubit: 1, state: state)
        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "Reset on qubit 1 of |10> must project qubit 1 to |0>, yielding |00>")
        #expect(abs(result.probability(of: 1)) < 1e-10, "Reset on qubit 1 of |10> must not affect qubit 0 and must leave |01> with zero probability")
        #expect(abs(result.probability(of: 2)) < 1e-10, "Reset on qubit 1 of |10> must leave |10> with zero probability after qubit 1 is reset")
        #expect(abs(result.probability(of: 3)) < 1e-10, "Reset on qubit 1 of |10> must leave |11> with zero probability")
    }

    @Test("After Reset, probability of target qubit being |0> is 1.0 regardless of input")
    func resetAlwaysProjectsToZero() {
        let angles = [0.0, Double.pi / 6, Double.pi / 4, Double.pi / 3, Double.pi / 2, Double.pi]
        for angle in angles {
            let state = QuantumState(qubits: 1, amplitudes: [
                Complex(cos(angle / 2), 0.0), Complex(sin(angle / 2), 0.0),
            ])
            let result = GateApplication.applyReset(qubit: 0, state: state)
            #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "Reset must project to |0> with probability 1.0 for any input state with rotation angle \(angle)")
        }
    }

    @Test("Reset on entangled Bell state: qubit 0 is projected to |0> with certainty")
    func resetOnBellState() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bellState = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0.0), .zero, .zero, Complex(invSqrt2, 0.0),
        ])
        let result = GateApplication.applyReset(qubit: 0, state: bellState)
        let prob0x = result.probability(of: 0) + result.probability(of: 2)
        let prob1x = result.probability(of: 1) + result.probability(of: 3)
        #expect(abs(prob0x - 1.0) < 1e-10, "After resetting qubit 0 of Bell state (|00>+|11>)/sqrt(2), total probability with qubit 0 = |0> must be 1.0")
        #expect(abs(prob1x) < 1e-10, "After resetting qubit 0 of Bell state, total probability with qubit 0 = |1> must be 0.0")
        #expect(result.isNormalized(), "Reset on Bell state must produce a normalized output state")
    }
}

/// Test suite for Reset gate on density matrices.
/// Validates Kraus-operator-based reset that projects the target qubit to |0>,
/// including tests on pure |0>, pure |1>, and superposition density matrices.
@Suite("Reset Density Matrix")
struct ResetDensityMatrixTests {
    @Test("Reset on |0><0| density matrix gives |0><0|")
    func resetOnZeroDensityMatrix() {
        let dm = DensityMatrix(qubits: 1)
        let result = dm.applying(CircuitOperation.reset(qubit: 0))
        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "Reset on |0><0| density matrix must yield probability 1.0 for |0> since it is already in ground state")
        #expect(abs(result.probability(of: 1)) < 1e-10, "Reset on |0><0| density matrix must yield probability 0.0 for |1>")
        #expect(abs(result.trace() - 1.0) < 1e-10, "Reset on |0><0| must preserve trace normalization Tr(rho) = 1")
    }

    @Test("Reset on |1><1| density matrix gives |0><0|")
    func resetOnOneDensityMatrix() {
        let pureOne = QuantumState(qubit: 1)
        let dm = DensityMatrix(pureState: pureOne)
        let result = dm.applying(CircuitOperation.reset(qubit: 0))
        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "Reset on |1><1| density matrix must project to |0><0| with probability 1.0 for |0>")
        #expect(abs(result.probability(of: 1)) < 1e-10, "Reset on |1><1| density matrix must yield probability 0.0 for |1> after projection")
        #expect(abs(result.trace() - 1.0) < 1e-10, "Reset on |1><1| must preserve trace normalization Tr(rho) = 1")
    }

    @Test("Reset on (|0>+|1>)(<0|+<1|)/2 density matrix gives |0><0|")
    func resetOnPlusDensityMatrix() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let plusState = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0),
        ])
        let dm = DensityMatrix(pureState: plusState)
        let result = dm.applying(CircuitOperation.reset(qubit: 0))
        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "Reset on |+><+| density matrix must project to |0><0| with probability 1.0 for |0>")
        #expect(abs(result.probability(of: 1)) < 1e-10, "Reset on |+><+| density matrix must yield probability 0.0 for |1> after projection to ground state")
        let rho00 = result.element(row: 0, col: 0)
        #expect(abs(rho00.real - 1.0) < 1e-10, "Reset on |+><+| must produce rho[0,0] = 1.0 corresponding to pure |0><0|")
        #expect(abs(rho00.imaginary) < 1e-10, "Reset on |+><+| must produce real-valued rho[0,0]")
        let rho01 = result.element(row: 0, col: 1)
        #expect(abs(rho01.real) < 1e-10, "Reset on |+><+| must zero out off-diagonal element rho[0,1] real part")
        #expect(abs(rho01.imaginary) < 1e-10, "Reset on |+><+| must zero out off-diagonal element rho[0,1] imaginary part")
        let rho10 = result.element(row: 1, col: 0)
        #expect(abs(rho10.real) < 1e-10, "Reset on |+><+| must zero out off-diagonal element rho[1,0] real part")
        #expect(abs(rho10.imaginary) < 1e-10, "Reset on |+><+| must zero out off-diagonal element rho[1,0] imaginary part")
        let rho11 = result.element(row: 1, col: 1)
        #expect(abs(rho11.real) < 1e-10, "Reset on |+><+| must produce rho[1,1] = 0.0 corresponding to zero population in |1>")
        #expect(abs(rho11.imaginary) < 1e-10, "Reset on |+><+| must produce real-valued rho[1,1]")
        #expect(abs(result.trace() - 1.0) < 1e-10, "Reset on |+><+| must preserve trace normalization Tr(rho) = 1")
    }
}

/// Test suite for GateApplication.apply routing of CircuitOperation.reset.
/// Validates that the CircuitOperation dispatch method correctly delegates
/// reset operations to applyReset and produces the expected projected state.
@Suite("GateApplication Reset Dispatch")
struct GateApplicationResetDispatchTests {
    @Test("GateApplication.apply routes CircuitOperation.reset to applyReset on |1> state")
    func applyRoutesResetOnOneState() {
        let state = QuantumState(qubit: 1)
        let resetOp = CircuitOperation.reset(qubit: 0)

        let result = GateApplication.apply(resetOp, state: state)

        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "GateApplication.apply with CircuitOperation.reset on |1> must project qubit to |0> with probability 1.0")
        #expect(abs(result.probability(of: 1)) < 1e-10, "GateApplication.apply with CircuitOperation.reset on |1> must leave |1> with zero probability")
    }

    @Test("GateApplication.apply routes CircuitOperation.reset to applyReset on superposition")
    func applyRoutesResetOnSuperposition() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let state = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0),
        ])
        let resetOp = CircuitOperation.reset(qubit: 0)

        let result = GateApplication.apply(resetOp, state: state)

        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "GateApplication.apply with CircuitOperation.reset on |+> must project qubit to |0> with probability 1.0")
        #expect(result.isNormalized(), "GateApplication.apply with CircuitOperation.reset must produce a normalized state")
    }

    @Test("GateApplication.apply routes CircuitOperation.reset on multi-qubit state")
    func applyRoutesResetOnMultiQubitState() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(1, to: .one)
        let resetOp = CircuitOperation.reset(qubit: 0)

        let result = GateApplication.apply(resetOp, state: state)

        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "GateApplication.apply with CircuitOperation.reset on qubit 0 of |01> must yield |00> with probability 1.0")
        #expect(abs(result.probability(of: 1)) < 1e-10, "GateApplication.apply with CircuitOperation.reset on qubit 0 of |01> must leave |01> with zero probability")
    }
}

/// Test suite for MetalGateApplication.apply routing of CircuitOperation.reset.
/// Validates that the GPU dispatch method correctly delegates reset operations
/// to the CPU applyReset path and produces the expected projected state.
@Suite("MetalGateApplication Reset Dispatch")
struct MetalGateApplicationResetDispatchTests {
    @Test("MetalGateApplication.apply routes CircuitOperation.reset via circuit execution")
    func metalResetViaCircuitExecution() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        circuit.append(.reset, to: 0)

        let result = circuit.execute()

        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "Circuit with X then reset must produce |0> with probability 1.0 regardless of execution backend")
        #expect(abs(result.probability(of: 1)) < 1e-10, "Circuit with X then reset must leave |1> with zero probability")
    }

    @Test("MetalGateApplication.apply routes CircuitOperation.reset on instantiated actor")
    func metalResetOnInstantiatedActor() async {
        guard let metalApp = MetalGateApplication() else {
            return
        }

        let state = QuantumState(qubit: 1)
        let resetOp = CircuitOperation.reset(qubit: 0)

        let result = await metalApp.apply(resetOp, state: state)

        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "MetalGateApplication.apply with CircuitOperation.reset on |1> must project qubit to |0> via CPU applyReset fallback")
        #expect(result.isNormalized(), "MetalGateApplication.apply with CircuitOperation.reset must produce a normalized state")
    }

    @Test("MetalGateApplication.apply routes reset then continues with gate on actor")
    func metalResetThenGateOnActor() async {
        guard let metalApp = MetalGateApplication() else {
            return
        }

        var state = QuantumState(qubits: 1)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(1, to: .one)
        let resetOp = CircuitOperation.reset(qubit: 0)

        let afterReset = await metalApp.apply(resetOp, state: state)
        let afterHadamard = await metalApp.apply(.hadamard, to: 0, state: afterReset)

        #expect(abs(afterHadamard.probability(of: 0) - 0.5) < 1e-10, "After reset then Hadamard, probability of |0> must be 0.5")
        #expect(abs(afterHadamard.probability(of: 1) - 0.5) < 1e-10, "After reset then Hadamard, probability of |1> must be 0.5")
    }
}
