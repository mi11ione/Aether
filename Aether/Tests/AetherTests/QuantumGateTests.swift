// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

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
        let gate = QuantumGate.cnot
        let matrix = gate.matrix()
        #expect(QuantumGate.isUnitary(matrix))
    }

    @Test("Phase gate is unitary")
    func phaseUnitary() {
        let gate = QuantumGate.phase(angle: .pi / 4.0)
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
        let matrix = QuantumGate.cnot.matrix()

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
        #expect(QuantumGate.phase(angle: 0).qubitsRequired == 1)
    }

    @Test("Two-qubit gates require 2 qubits")
    func twoQubitGatesRequireTwo() {
        #expect(QuantumGate.cnot.qubitsRequired == 2)
        #expect(QuantumGate.swap.qubitsRequired == 2)
        #expect(QuantumGate.controlledPhase(theta: 0).qubitsRequired == 2)
    }

    @Test("Toffoli gate requires 3 qubits")
    func toffoliRequiresThree() {
        #expect(QuantumGate.toffoli.qubitsRequired == 3)
    }

    @Test("Parameterized gates are identified correctly")
    func parameterizedGates() {
        #expect(QuantumGate.phase(angle: 0).isParameterized)
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
        #expect(!QuantumGate.phase(angle: .pi / 4.0).isHermitian)
    }
}

/// Test suite for self-inverse quantum gates.
/// Validates U² = I property for gates that are their own inverse,
/// enabling circuit optimization and depth reduction.
@Suite("Self-Inverse Gates")
struct SelfInverseGateTests {
    @Test("Hadamard is self-inverse: H·H = I")
    func hadamardSelfInverse() {
        let h = QuantumGate.hadamard.matrix()
        let product = QuantumGate.matrixMultiply(h, h)

        #expect(QuantumGate.isIdentityMatrix(product))
    }

    @Test("Pauli gates are self-inverse")
    func pauliGatesSelfInverse() {
        let x = QuantumGate.pauliX.matrix()
        let y = QuantumGate.pauliY.matrix()
        let z = QuantumGate.pauliZ.matrix()

        let xx = QuantumGate.matrixMultiply(x, x)
        let yy = QuantumGate.matrixMultiply(y, y)
        let zz = QuantumGate.matrixMultiply(z, z)

        #expect(QuantumGate.isIdentityMatrix(xx))
        #expect(QuantumGate.isIdentityMatrix(yy))
        #expect(QuantumGate.isIdentityMatrix(zz))
    }

    @Test("CNOT is self-inverse")
    func cnotSelfInverse() {
        let cnot = QuantumGate.cnot.matrix()
        let product = QuantumGate.matrixMultiply(cnot, cnot)

        #expect(QuantumGate.isIdentityMatrix(product))
    }
}

/// Test suite for parameterized quantum gates.
/// Validates angle-dependent gates (Phase, Rotation) and their special cases,
/// ensuring correct phase and rotation operations in quantum circuits.
@Suite("Parameterized Gate")
struct ParameterizedGateTests {
    @Test("Phase(0) equals identity")
    func phaseZeroIsIdentity() {
        let phase = QuantumGate.phase(angle: 0).matrix()
        let identity = QuantumGate.identity.matrix()

        #expect(QuantumGate.matricesEqual(phase, identity))
    }

    @Test("Phase(π) equals Pauli-Z")
    func phasePiIsZ() {
        let phase = QuantumGate.phase(angle: .pi).matrix()
        let z = QuantumGate.pauliZ.matrix()

        #expect(QuantumGate.matricesEqual(phase, z))
    }

    @Test("S gate equals Phase(π/2)")
    func sGateEqualsPhaseHalfPi() {
        let s = QuantumGate.sGate.matrix()
        let phase = QuantumGate.phase(angle: .pi / 2.0).matrix()

        #expect(QuantumGate.matricesEqual(s, phase))
    }

    @Test("T gate equals Phase(π/4)")
    func tGateEqualsPhaseQuarterPi() {
        let t = QuantumGate.tGate.matrix()
        let phase = QuantumGate.phase(angle: .pi / 4.0).matrix()

        #expect(QuantumGate.matricesEqual(t, phase))
    }

    @Test("Rotation(0) equals identity")
    func rotationZeroIsIdentity() {
        let rx = QuantumGate.rotationX(theta: 0).matrix()
        let ry = QuantumGate.rotationY(theta: 0).matrix()
        let rz = QuantumGate.rotationZ(theta: 0).matrix()
        let identity = QuantumGate.identity.matrix()

        #expect(QuantumGate.matricesEqual(rx, identity))
        #expect(QuantumGate.matricesEqual(ry, identity))
        #expect(QuantumGate.matricesEqual(rz, identity))
    }
}

/// Test suite for quantum gate validation.
/// Ensures gate configurations are physically valid and mathematically sound,
/// preventing invalid quantum circuit constructions.
@Suite("Gate Validation")
struct GateValidationTests {
    @Test("Valid CNOT indices pass validation")
    func validCNOTIndices() {
        let gate = QuantumGate.cnot
        #expect(gate.validateQubitIndices([0, 1], maxAllowedQubit: 1))
        #expect(gate.validateQubitIndices([0, 1], maxAllowedQubit: 2))
    }

    @Test("CNOT with control=target fails validation")
    func cnotSameQubitFails() {
        let gate = QuantumGate.cnot
        #expect(!gate.validateQubitIndices([0, 0], maxAllowedQubit: 1))
    }

    @Test("CNOT with out-of-bounds indices fails validation")
    func cnotOutOfBoundsFails() {
        let gate = QuantumGate.cnot
        #expect(!gate.validateQubitIndices([0, 2], maxAllowedQubit: 1))
    }

    @Test("Toffoli with distinct qubits passes validation")
    func validToffoliIndices() {
        let gate = QuantumGate.toffoli
        #expect(gate.validateQubitIndices([0, 1, 2], maxAllowedQubit: 2))
    }

    @Test("Toffoli with duplicate indices fails validation")
    func toffoliDuplicateFails() {
        let gate = QuantumGate.toffoli
        #expect(!gate.validateQubitIndices([0, 0, 2], maxAllowedQubit: 2))
    }

    @Test("Custom two-qubit gate validates successfully with correct matrix")
    func customTwoQubitValidMatrix() {
        let swapMatrix = [
            [Complex<Double>.one, .zero, .zero, .zero],
            [.zero, .zero, .one, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .zero, .one],
        ]

        let gate = QuantumGate.custom(matrix: swapMatrix, control: 0, target: 1)
        let matrix = gate.matrix()
        #expect(matrix.count == 4)
        #expect(matrix[0].count == 4)
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

    @Test("All gate descriptions are non-empty")
    func allGateDescriptionsNonEmpty() {
        #expect(QuantumGate.identity.description == "I")
        #expect(QuantumGate.phase(angle: 1.234).description.contains("P"))
        #expect(QuantumGate.phase(angle: 1.234).description.contains("1.234"))
        #expect(QuantumGate.rotationX(theta: 2.345).description.contains("Rx"))
        #expect(QuantumGate.rotationY(theta: 3.456).description.contains("Ry"))
        #expect(QuantumGate.rotationZ(theta: 4.567).description.contains("Rz"))
        #expect(QuantumGate.controlledPhase(theta: 1.5).description.contains("CP"))
        #expect(QuantumGate.cnot.description.contains("CNOT"))
        #expect(QuantumGate.swap.description.contains("SWAP"))
        #expect(QuantumGate.toffoli.description.contains("Toffoli"))
    }
}

/// Test suite for Toffoli (CCNOT) gate.
/// Validates three-qubit controlled gate implementation,
/// matrix generation, and unitary properties.
@Suite("Toffoli Gate")
struct ToffoliGateTests {
    @Test("Toffoli gate is unitary")
    func toffoliIsUnitary() {
        let gate = QuantumGate.toffoli
        let matrix = gate.matrix()
        #expect(QuantumGate.isUnitary(matrix))
    }

    @Test("Toffoli matrix is 8x8")
    func toffoliMatrixSize() {
        let gate = QuantumGate.toffoli
        let matrix = gate.matrix()
        #expect(matrix.count == 8)
        #expect(matrix.allSatisfy { $0.count == 8 })
    }

    @Test("Toffoli matrix has identity on first 6 states")
    func toffoliMatrixStructure() {
        let gate = QuantumGate.toffoli
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
    func toffoliSelfInverse() {
        let toffoli = QuantumGate.toffoli.matrix()
        let product = QuantumGate.matrixMultiply(toffoli, toffoli)
        #expect(QuantumGate.isIdentityMatrix(product))
    }
}

/// Test suite for matrix utility functions.
/// Validates edge cases and error paths
/// for matricesEqual and isIdentityMatrix.
@Suite("Matrix Utility Functions")
struct MatrixUtilityTests {
    @Test("matricesEqual returns false for different row counts")
    func matricesEqualDifferentRows() {
        let a: [[Complex<Double>]] = [
            [Complex<Double>.one, Complex<Double>.zero],
            [Complex<Double>.zero, Complex<Double>.one],
        ]
        let b: [[Complex<Double>]] = [
            [Complex<Double>.one, Complex<Double>.zero],
        ]

        #expect(!QuantumGate.matricesEqual(a, b))
    }

    @Test("matricesEqual returns false for different column counts")
    func matricesEqualDifferentColumns() {
        let a: [[Complex<Double>]] = [
            [Complex<Double>.one, Complex<Double>.zero],
            [Complex<Double>.zero, Complex<Double>.one],
        ]
        let b: [[Complex<Double>]] = [
            [Complex<Double>.one],
            [Complex<Double>.one],
        ]

        #expect(!QuantumGate.matricesEqual(a, b))
    }

    @Test("matricesEqual returns false when real parts differ")
    func matricesEqualDifferentRealParts() {
        let a: [[Complex<Double>]] = [
            [Complex<Double>(1.0, 0.0), Complex<Double>.zero],
            [Complex<Double>.zero, Complex<Double>.one],
        ]
        let b: [[Complex<Double>]] = [
            [Complex<Double>(0.5, 0.0), Complex<Double>.zero],
            [Complex<Double>.zero, Complex<Double>.one],
        ]

        #expect(!QuantumGate.matricesEqual(a, b))
    }

    @Test("matricesEqual returns false when imaginary parts differ")
    func matricesEqualDifferentImaginaryParts() {
        let a: [[Complex<Double>]] = [
            [Complex<Double>(1.0, 0.5), Complex<Double>.zero],
            [Complex<Double>.zero, Complex<Double>.one],
        ]
        let b: [[Complex<Double>]] = [
            [Complex<Double>(1.0, 0.0), Complex<Double>.zero],
            [Complex<Double>.zero, Complex<Double>.one],
        ]

        #expect(!QuantumGate.matricesEqual(a, b))
    }

    @Test("matricesEqual returns true for identical matrices")
    func matricesEqualIdentical() {
        let a: [[Complex<Double>]] = [
            [Complex<Double>.one, Complex<Double>.zero],
            [Complex<Double>.zero, Complex<Double>.one],
        ]
        let b: [[Complex<Double>]] = [
            [Complex<Double>.one, Complex<Double>.zero],
            [Complex<Double>.zero, Complex<Double>.one],
        ]

        #expect(QuantumGate.matricesEqual(a, b))
    }

    @Test("matricesEqual respects tolerance parameter")
    func matricesEqualCustomTolerance() {
        let a: [[Complex<Double>]] = [
            [Complex<Double>(1.0, 0.0), Complex<Double>.zero],
        ]
        let b: [[Complex<Double>]] = [
            [Complex<Double>(1.0 + 1e-11, 0.0), Complex<Double>.zero],
        ]

        #expect(QuantumGate.matricesEqual(a, b, tolerance: 1e-10))
        #expect(!QuantumGate.matricesEqual(a, b, tolerance: 1e-12))
    }

    @Test("isIdentityMatrix returns false for empty matrix")
    func isIdentityMatrixEmpty() {
        let empty: [[Complex<Double>]] = []

        #expect(!QuantumGate.isIdentityMatrix(empty))
    }

    @Test("isIdentityMatrix returns false for non-square matrix")
    func isIdentityMatrixNonSquare() {
        let nonSquare: [[Complex<Double>]] = [
            [Complex<Double>.one, Complex<Double>.zero, Complex<Double>.zero],
            [Complex<Double>.zero, Complex<Double>.one, Complex<Double>.zero],
        ]

        #expect(!QuantumGate.isIdentityMatrix(nonSquare))
    }

    @Test("isIdentityMatrix returns false for matrix with wrong diagonal")
    func isIdentityMatrixWrongDiagonal() {
        let wrongDiagonal: [[Complex<Double>]] = [
            [Complex<Double>(2.0, 0.0), Complex<Double>.zero],
            [Complex<Double>.zero, Complex<Double>.one],
        ]

        #expect(!QuantumGate.isIdentityMatrix(wrongDiagonal))
    }

    @Test("isIdentityMatrix returns false for matrix with non-zero off-diagonal")
    func isIdentityMatrixNonZeroOffDiagonal() {
        let nonZeroOffDiagonal: [[Complex<Double>]] = [
            [Complex<Double>.one, Complex<Double>(0.1, 0.0)],
            [Complex<Double>.zero, Complex<Double>.one],
        ]

        #expect(!QuantumGate.isIdentityMatrix(nonZeroOffDiagonal))
    }

    @Test("isIdentityMatrix returns true for 2x2 identity")
    func isIdentityMatrix2x2() {
        let identity: [[Complex<Double>]] = [
            [Complex<Double>.one, Complex<Double>.zero],
            [Complex<Double>.zero, Complex<Double>.one],
        ]

        #expect(QuantumGate.isIdentityMatrix(identity))
    }

    @Test("isIdentityMatrix returns true for 4x4 identity")
    func isIdentityMatrix4x4() {
        let identity: [[Complex<Double>]] = [
            [Complex<Double>.one, Complex<Double>.zero, Complex<Double>.zero, Complex<Double>.zero],
            [Complex<Double>.zero, Complex<Double>.one, Complex<Double>.zero, Complex<Double>.zero],
            [Complex<Double>.zero, Complex<Double>.zero, Complex<Double>.one, Complex<Double>.zero],
            [Complex<Double>.zero, Complex<Double>.zero, Complex<Double>.zero, Complex<Double>.one],
        ]

        #expect(QuantumGate.isIdentityMatrix(identity))
    }

    @Test("isIdentityMatrix respects tolerance parameter")
    func isIdentityMatrixCustomTolerance() {
        let almostIdentity: [[Complex<Double>]] = [
            [Complex<Double>(1.0 + 1e-11, 0.0), Complex<Double>.zero],
            [Complex<Double>.zero, Complex<Double>(1.0, 0.0)],
        ]

        #expect(QuantumGate.isIdentityMatrix(almostIdentity, tolerance: 1e-10))
        #expect(!QuantumGate.isIdentityMatrix(almostIdentity, tolerance: 1e-12))
    }
}
