//
//  QuantumGateTests.swift
//  AetherTests
//
//  Comprehensive test suite for QuantumGate enum.
//  Validates unitary matrix representations, gate properties,
//  and mathematical correctness of quantum gate operations.
//
//  Created by mi11ion on 21/10/25.
//

@testable import Aether
import Foundation
import Testing

/// Test suite for quantum gate unitarity.
/// Validates U†U = I (unitary matrices preserve probability)
/// for all gate implementations, ensuring quantum mechanics correctness.
@Suite("Gate Matrix Unitarity")
struct GateUnitarityTests {
    @Test("Identity gate is unitary")
    func identityUnitary() {
        let gate = QuantumGate.identity
        let matrix = gate.matrix()
        #expect(QuantumGate.isUnitary(matrix))
    }

    @Test("Non-square matrix is not unitary")
    func nonSquareMatrixNotUnitary() {
        let nonSquare = [
            [Complex<Double>.one, Complex<Double>.zero, Complex<Double>.zero],
            [Complex<Double>.zero, Complex<Double>.one, Complex<Double>.zero],
        ]
        #expect(!QuantumGate.isUnitary(nonSquare))
    }

    @Test("Non-unitary matrix fails unitarity check")
    func nonUnitaryMatrixFails() {
        let nonUnitary = [
            [Complex<Double>(2.0, 0.0), Complex<Double>.zero],
            [Complex<Double>.zero, Complex<Double>(2.0, 0.0)],
        ]
        #expect(!QuantumGate.isUnitary(nonUnitary))
    }

    @Test("Hadamard gate is unitary")
    func hadamardUnitary() {
        let gate = QuantumGate.hadamard
        let matrix = gate.matrix()
        #expect(QuantumGate.isUnitary(matrix))
    }

    @Test("Pauli-X gate is unitary")
    func pauliXUnitary() {
        let gate = QuantumGate.pauliX
        let matrix = gate.matrix()
        #expect(QuantumGate.isUnitary(matrix))
    }

    @Test("Pauli-Y gate is unitary")
    func pauliYUnitary() {
        let gate = QuantumGate.pauliY
        let matrix = gate.matrix()
        #expect(QuantumGate.isUnitary(matrix))
    }

    @Test("Pauli-Z gate is unitary")
    func pauliZUnitary() {
        let gate = QuantumGate.pauliZ
        let matrix = gate.matrix()
        #expect(QuantumGate.isUnitary(matrix))
    }

    @Test("CNOT gate is unitary")
    func cnotUnitary() {
        let gate = QuantumGate.cnot(control: 0, target: 1)
        let matrix = gate.matrix()
        #expect(QuantumGate.isUnitary(matrix))
    }

    @Test("Phase gate is unitary")
    func phaseUnitary() {
        let gate = QuantumGate.phase(theta: .pi / 4.0)
        let matrix = gate.matrix()
        #expect(QuantumGate.isUnitary(matrix))
    }

    @Test("Rotation gates are unitary")
    func rotationGatesUnitary() {
        let rx = QuantumGate.rotationX(theta: .pi / 3.0)
        let ry = QuantumGate.rotationY(theta: .pi / 3.0)
        let rz = QuantumGate.rotationZ(theta: .pi / 3.0)

        #expect(QuantumGate.isUnitary(rx.matrix()))
        #expect(QuantumGate.isUnitary(ry.matrix()))
        #expect(QuantumGate.isUnitary(rz.matrix()))
    }
}

/// Test suite for specific gate matrix representations.
/// Validates exact matrix elements against quantum computing standards,
/// ensuring correct Pauli, Hadamard, and CNOT implementations.
@Suite("Specific Gate Matrices")
struct GateMatrixTests {
    @Test("Pauli-X matrix is correct")
    func pauliXMatrix() {
        let matrix = QuantumGate.pauliX.matrix()

        #expect(matrix[0][0] == .zero)
        #expect(matrix[0][1] == .one)
        #expect(matrix[1][0] == .one)
        #expect(matrix[1][1] == .zero)
    }

    @Test("Pauli-Z matrix is correct")
    func pauliZMatrix() {
        let matrix = QuantumGate.pauliZ.matrix()

        #expect(matrix[0][0] == .one)
        #expect(matrix[0][1] == .zero)
        #expect(matrix[1][0] == .zero)
        #expect(abs(matrix[1][1].real - -1.0) < 1e-10)
    }

    @Test("Hadamard matrix is correct")
    func hadamardMatrix() {
        let matrix = QuantumGate.hadamard.matrix()
        let invSqrt2 = 1.0 / sqrt(2.0)

        #expect(abs(matrix[0][0].real - invSqrt2) < 1e-10)
        #expect(abs(matrix[0][1].real - invSqrt2) < 1e-10)
        #expect(abs(matrix[1][0].real - invSqrt2) < 1e-10)
        #expect(abs(matrix[1][1].real - -invSqrt2) < 1e-10)
    }

    @Test("CNOT matrix is correct")
    func cnotMatrix() {
        let matrix = QuantumGate.cnot(control: 0, target: 1).matrix()

        #expect(matrix.count == 4)
        #expect(matrix[0].count == 4)

        #expect(matrix[0][0] == .one)
        #expect(matrix[1][1] == .one)
        #expect(matrix[2][3] == .one)
        #expect(matrix[3][2] == .one)
    }
}

/// Test suite for quantum gate properties.
/// Validates qubit requirements, parameterization flags, and gate classifications
/// essential for circuit construction and optimization algorithms.
@Suite("Gate Properties")
struct GatePropertiesTests {
    @Test("Single-qubit gates require 1 qubit")
    func singleQubitGatesRequireOne() {
        #expect(QuantumGate.hadamard.qubitsRequired == 1)
        #expect(QuantumGate.pauliX.qubitsRequired == 1)
        #expect(QuantumGate.pauliZ.qubitsRequired == 1)
        #expect(QuantumGate.phase(theta: 0).qubitsRequired == 1)
    }

    @Test("Two-qubit gates require 2 qubits")
    func twoQubitGatesRequireTwo() {
        #expect(QuantumGate.cnot(control: 0, target: 1).qubitsRequired == 2)
        #expect(QuantumGate.swap(qubit1: 0, qubit2: 1).qubitsRequired == 2)
        #expect(QuantumGate.controlledPhase(theta: 0, control: 0, target: 1).qubitsRequired == 2)
    }

    @Test("Toffoli gate requires 3 qubits")
    func toffoliRequiresThree() {
        #expect(QuantumGate.toffoli(control1: 0, control2: 1, target: 2).qubitsRequired == 3)
    }

    @Test("Parameterized gates are identified correctly")
    func parameterizedGates() {
        #expect(QuantumGate.phase(theta: 0).isParameterized)
        #expect(QuantumGate.rotationX(theta: 0).isParameterized)
        #expect(QuantumGate.rotationY(theta: 0).isParameterized)
        #expect(QuantumGate.rotationZ(theta: 0).isParameterized)
        #expect(!QuantumGate.hadamard.isParameterized)
        #expect(!QuantumGate.pauliX.isParameterized)
    }

    @Test("Hermitian gates are identified correctly")
    func hermitianGates() {
        #expect(QuantumGate.pauliX.isHermitian)
        #expect(QuantumGate.pauliY.isHermitian)
        #expect(QuantumGate.pauliZ.isHermitian)
        #expect(QuantumGate.hadamard.isHermitian)
        #expect(!QuantumGate.phase(theta: .pi / 4.0).isHermitian)
    }
}

/// Test suite for self-inverse quantum gates.
/// Validates U² = I property for gates that are their own inverse,
/// enabling circuit optimization and depth reduction.
@Suite("Self-Inverse Gates")
struct SelfInverseGateTests {
    @Test("Hadamard is self-inverse: H·H = I")
    func hadamardSelfInverse() throws {
        let h = QuantumGate.hadamard.matrix()
        let product = try TestUtilities.matrixMultiply(h, h)

        #expect(TestUtilities.isIdentityMatrix(product))
    }

    @Test("Pauli gates are self-inverse")
    func pauliGatesSelfInverse() throws {
        let x = QuantumGate.pauliX.matrix()
        let y = QuantumGate.pauliY.matrix()
        let z = QuantumGate.pauliZ.matrix()

        let xx = try TestUtilities.matrixMultiply(x, x)
        let yy = try TestUtilities.matrixMultiply(y, y)
        let zz = try TestUtilities.matrixMultiply(z, z)

        #expect(TestUtilities.isIdentityMatrix(xx))
        #expect(TestUtilities.isIdentityMatrix(yy))
        #expect(TestUtilities.isIdentityMatrix(zz))
    }

    @Test("CNOT is self-inverse")
    func cnotSelfInverse() throws {
        let cnot = QuantumGate.cnot(control: 0, target: 1).matrix()
        let product = try TestUtilities.matrixMultiply(cnot, cnot)

        #expect(TestUtilities.isIdentityMatrix(product))
    }
}

/// Test suite for parameterized quantum gates.
/// Validates angle-dependent gates (Phase, Rotation) and their special cases,
/// ensuring correct phase and rotation operations in quantum circuits.
@Suite("Parameterized Gate Tests")
struct ParameterizedGateTests {
    @Test("Phase(0) equals identity")
    func phaseZeroIsIdentity() {
        let phase = QuantumGate.phase(theta: 0).matrix()
        let identity = QuantumGate.identity.matrix()

        #expect(TestUtilities.matricesEqual(phase, identity))
    }

    @Test("Phase(π) equals Pauli-Z")
    func phasePiIsZ() {
        let phase = QuantumGate.phase(theta: .pi).matrix()
        let z = QuantumGate.pauliZ.matrix()

        #expect(TestUtilities.matricesEqual(phase, z))
    }

    @Test("S gate equals Phase(π/2)")
    func sGateEqualsPhaseHalfPi() {
        let s = QuantumGate.sGate.matrix()
        let phase = QuantumGate.phase(theta: .pi / 2.0).matrix()

        #expect(TestUtilities.matricesEqual(s, phase))
    }

    @Test("T gate equals Phase(π/4)")
    func tGateEqualsPhaseQuarterPi() {
        let t = QuantumGate.tGate.matrix()
        let phase = QuantumGate.phase(theta: .pi / 4.0).matrix()

        #expect(TestUtilities.matricesEqual(t, phase))
    }

    @Test("Rotation(0) equals identity")
    func rotationZeroIsIdentity() {
        let rx = QuantumGate.rotationX(theta: 0).matrix()
        let ry = QuantumGate.rotationY(theta: 0).matrix()
        let rz = QuantumGate.rotationZ(theta: 0).matrix()
        let identity = QuantumGate.identity.matrix()

        #expect(TestUtilities.matricesEqual(rx, identity))
        #expect(TestUtilities.matricesEqual(ry, identity))
        #expect(TestUtilities.matricesEqual(rz, identity))
    }
}

/// Test suite for quantum gate validation.
/// Ensures gate configurations are physically valid and mathematically sound,
/// preventing invalid quantum circuit constructions.
@Suite("Gate Validation")
struct GateValidationTests {
    @Test("Valid CNOT indices pass validation")
    func validCNOTIndices() {
        let gate = QuantumGate.cnot(control: 0, target: 1)
        #expect(gate.validateQubitIndices(numQubits: 2))
        #expect(gate.validateQubitIndices(numQubits: 3))
    }

    @Test("CNOT with control=target fails validation")
    func cnotSameQubitFails() {
        let gate = QuantumGate.cnot(control: 0, target: 0)
        #expect(!gate.validateQubitIndices(numQubits: 2))
    }

    @Test("CNOT with out-of-bounds indices fails validation")
    func cnotOutOfBoundsFails() {
        let gate = QuantumGate.cnot(control: 0, target: 2)
        #expect(!gate.validateQubitIndices(numQubits: 2))
    }

    @Test("Toffoli with distinct qubits passes validation")
    func validToffoliIndices() {
        let gate = QuantumGate.toffoli(control1: 0, control2: 1, target: 2)
        #expect(gate.validateQubitIndices(numQubits: 3))
    }

    @Test("Toffoli with duplicate indices fails validation")
    func toffoliDuplicateFails() {
        let gate = QuantumGate.toffoli(control1: 0, control2: 0, target: 2)
        #expect(!gate.validateQubitIndices(numQubits: 3))
    }
}

/// Test suite for quantum gate string representations.
/// Validates CustomStringConvertible implementations for debugging
/// and educational quantum circuit visualization.
@Suite("Gate String Representation")
struct GateDescriptionTests {
    @Test("Single-qubit gates have correct descriptions")
    func singleQubitDescriptions() {
        #expect(QuantumGate.hadamard.description == "H")
        #expect(QuantumGate.pauliX.description == "X")
        #expect(QuantumGate.pauliY.description == "Y")
        #expect(QuantumGate.pauliZ.description == "Z")
        #expect(QuantumGate.sGate.description == "S")
        #expect(QuantumGate.tGate.description == "T")
    }

    @Test("Two-qubit gates show indices")
    func twoQubitDescriptions() {
        let cnot = QuantumGate.cnot(control: 0, target: 1)
        #expect(cnot.description.contains("CNOT"))
        #expect(cnot.description.contains("0"))
        #expect(cnot.description.contains("1"))
    }

    @Test("All gate descriptions are non-empty")
    func allGateDescriptionsNonEmpty() {
        #expect(QuantumGate.identity.description == "I")
        #expect(QuantumGate.phase(theta: 1.234).description.contains("P"))
        #expect(QuantumGate.phase(theta: 1.234).description.contains("1.234"))
        #expect(QuantumGate.rotationX(theta: 2.345).description.contains("Rx"))
        #expect(QuantumGate.rotationY(theta: 3.456).description.contains("Ry"))
        #expect(QuantumGate.rotationZ(theta: 4.567).description.contains("Rz"))
        #expect(QuantumGate.controlledPhase(theta: 1.5, control: 0, target: 1).description.contains("CP"))
        #expect(QuantumGate.swap(qubit1: 2, qubit2: 3).description.contains("SWAP"))
        #expect(QuantumGate.toffoli(control1: 0, control2: 1, target: 2).description.contains("Toffoli"))
    }
}

/// Test suite for Toffoli (CCNOT) gate.
/// Validates three-qubit controlled gate implementation,
/// matrix generation, and unitary properties.
@Suite("Toffoli Gate Tests")
struct ToffoliGateTests {
    @Test("Toffoli gate is unitary")
    func toffoliIsUnitary() {
        let gate = QuantumGate.toffoli(control1: 0, control2: 1, target: 2)
        let matrix = gate.matrix()
        #expect(QuantumGate.isUnitary(matrix))
    }

    @Test("Toffoli matrix is 8×8")
    func toffoliMatrixSize() {
        let gate = QuantumGate.toffoli(control1: 0, control2: 1, target: 2)
        let matrix = gate.matrix()
        #expect(matrix.count == 8)
        #expect(matrix.allSatisfy { $0.count == 8 })
    }

    @Test("Toffoli matrix has identity on first 6 states")
    func toffoliMatrixStructure() {
        let gate = QuantumGate.toffoli(control1: 0, control2: 1, target: 2)
        let matrix = gate.matrix()

        for i in 0 ..< 6 {
            for j in 0 ..< 8 {
                if i == j {
                    #expect(abs(matrix[i][j].real - 1.0) < 1e-10)
                    #expect(abs(matrix[i][j].imaginary) < 1e-10)
                } else {
                    #expect(abs(matrix[i][j].magnitude) < 1e-10)
                }
            }
        }

        #expect(abs(matrix[6][7].real - 1.0) < 1e-10)
        #expect(abs(matrix[7][6].real - 1.0) < 1e-10)
        #expect(abs(matrix[6][6].magnitude) < 1e-10)
        #expect(abs(matrix[7][7].magnitude) < 1e-10)
    }

    @Test("Toffoli is self-inverse")
    func toffoliSelfInverse() throws {
        let toffoli = QuantumGate.toffoli(control1: 0, control2: 1, target: 2).matrix()
        let product = try TestUtilities.matrixMultiply(toffoli, toffoli)
        #expect(TestUtilities.isIdentityMatrix(product))
    }
}
