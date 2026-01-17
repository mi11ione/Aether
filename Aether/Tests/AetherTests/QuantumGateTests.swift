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
        let gate = QuantumGate.phase(.pi / 4.0)
        let matrix = gate.matrix()
        #expect(QuantumGate.isUnitary(matrix))
    }

    @Test("Rotation gates are unitary")
    func rotationGatesUnitary() {
        let rx = QuantumGate.rotationX(.pi / 3.0)
        let ry = QuantumGate.rotationY(.pi / 3.0)
        let rz = QuantumGate.rotationZ(.pi / 3.0)

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
        #expect(QuantumGate.phase(0).qubitsRequired == 1)
    }

    @Test("Two-qubit gates require 2 qubits")
    func twoQubitGatesRequireTwo() {
        #expect(QuantumGate.cnot.qubitsRequired == 2)
        #expect(QuantumGate.swap.qubitsRequired == 2)
        #expect(QuantumGate.controlledPhase(0).qubitsRequired == 2)
    }

    @Test("Toffoli gate requires 3 qubits")
    func toffoliRequiresThree() {
        #expect(QuantumGate.toffoli.qubitsRequired == 3)
    }

    @Test("Hermitian gates are identified correctly")
    func hermitianGates() {
        #expect(QuantumGate.pauliX.isHermitian)
        #expect(QuantumGate.pauliY.isHermitian)
        #expect(QuantumGate.pauliZ.isHermitian)
        #expect(QuantumGate.hadamard.isHermitian)
        #expect(!QuantumGate.phase(.pi / 4.0).isHermitian)
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
        let phase = QuantumGate.phase(0).matrix()
        let identity = QuantumGate.identity.matrix()

        #expect(QuantumGate.matricesEqual(phase, identity))
    }

    @Test("Phase(π) equals Pauli-Z")
    func phasePiIsZ() {
        let phase = QuantumGate.phase(.pi).matrix()
        let z = QuantumGate.pauliZ.matrix()

        #expect(QuantumGate.matricesEqual(phase, z))
    }

    @Test("S gate equals Phase(π/2)")
    func sGateEqualsPhaseHalfPi() {
        let s = QuantumGate.sGate.matrix()
        let phase = QuantumGate.phase(.pi / 2.0).matrix()

        #expect(QuantumGate.matricesEqual(s, phase))
    }

    @Test("T gate equals Phase(π/4)")
    func tGateEqualsPhaseQuarterPi() {
        let t = QuantumGate.tGate.matrix()
        let phase = QuantumGate.phase(.pi / 4.0).matrix()

        #expect(QuantumGate.matricesEqual(t, phase))
    }

    @Test("Rotation(0) equals identity")
    func rotationZeroIsIdentity() {
        let rx = QuantumGate.rotationX(0).matrix()
        let ry = QuantumGate.rotationY(0).matrix()
        let rz = QuantumGate.rotationZ(0).matrix()
        let identity = QuantumGate.identity.matrix()

        #expect(QuantumGate.matricesEqual(rx, identity))
        #expect(QuantumGate.matricesEqual(ry, identity))
        #expect(QuantumGate.matricesEqual(rz, identity))
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
        #expect(QuantumGate.phase(1.234).description.contains("P"))
        #expect(QuantumGate.phase(1.234).description.contains("1.234"))
        #expect(QuantumGate.rotationX(2.345).description.contains("Rx"))
        #expect(QuantumGate.rotationY(3.456).description.contains("Ry"))
        #expect(QuantumGate.rotationZ(4.567).description.contains("Rz"))
        #expect(QuantumGate.controlledPhase(1.5).description.contains("CP"))
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

/// Test suite for symbolic parameter extraction from gates.
/// Validates parameters() method for U1, U2, U3 gates with symbolic
/// and concrete parameter values, ensuring correct parameter discovery.
@Suite("Symbolic Parameter Extraction")
struct SymbolicParameterExtractionTests {
    @Test("isParameterized returns false for non-parameterized gates")
    func isParameterizedFalseForNonParameterized() {
        #expect(!QuantumGate.hadamard.isParameterized)
        #expect(!QuantumGate.pauliX.isParameterized)
        #expect(!QuantumGate.cnot.isParameterized)
        #expect(!QuantumGate.toffoli.isParameterized)
    }

    @Test("isParameterized returns false for gates with concrete values")
    func isParameterizedFalseForConcreteValues() {
        #expect(!QuantumGate.rotationY(.pi / 4).isParameterized)
        #expect(!QuantumGate.u1(lambda: 1.5).isParameterized)
        #expect(!QuantumGate.u2(phi: 0.5, lambda: 1.0).isParameterized)
        #expect(!QuantumGate.u3(theta: 0.1, phi: 0.2, lambda: 0.3).isParameterized)
    }

    @Test("isParameterized returns true for gates with symbolic parameters")
    func isParameterizedTrueForSymbolic() {
        let theta = Parameter(name: "theta")
        #expect(QuantumGate.rotationY(.parameter(theta)).isParameterized)
        #expect(QuantumGate.u1(lambda: .parameter(theta)).isParameterized)
    }

    @Test("U1 parameters() returns symbolic parameter")
    func u1ParametersWithSymbolic() {
        let lambda = Parameter(name: "lambda")
        let gate = QuantumGate.u1(lambda: .parameter(lambda))
        let params = gate.parameters()

        #expect(params.count == 1, "U1 with symbolic lambda should have 1 parameter")
        #expect(params.contains(lambda), "U1 should contain the lambda parameter")
    }

    @Test("U1 parameters() returns empty for concrete value")
    func u1ParametersWithConcrete() {
        let gate = QuantumGate.u1(lambda: .value(1.5))
        let params = gate.parameters()

        #expect(params.isEmpty, "U1 with concrete lambda should have no parameters")
    }

    @Test("U2 parameters() returns both symbolic parameters")
    func u2ParametersWithBothSymbolic() {
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")
        let gate = QuantumGate.u2(phi: .parameter(phi), lambda: .parameter(lambda))
        let params = gate.parameters()

        #expect(params.count == 2, "U2 with both symbolic should have 2 parameters")
        #expect(params.contains(phi), "U2 should contain phi parameter")
        #expect(params.contains(lambda), "U2 should contain lambda parameter")
    }

    @Test("U2 parameters() returns only symbolic parameter when mixed")
    func u2ParametersWithMixed() {
        let phi = Parameter(name: "phi")
        let gate = QuantumGate.u2(phi: .parameter(phi), lambda: .value(1.0))
        let params = gate.parameters()

        #expect(params.count == 1, "U2 with one symbolic should have 1 parameter")
        #expect(params.contains(phi), "U2 should contain phi parameter")
    }

    @Test("U2 parameters() returns empty for all concrete")
    func u2ParametersWithAllConcrete() {
        let gate = QuantumGate.u2(phi: .value(0.5), lambda: .value(1.0))
        let params = gate.parameters()

        #expect(params.isEmpty, "U2 with all concrete should have no parameters")
    }

    @Test("U3 parameters() returns all three symbolic parameters")
    func u3ParametersWithAllSymbolic() {
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")
        let gate = QuantumGate.u3(
            theta: .parameter(theta),
            phi: .parameter(phi),
            lambda: .parameter(lambda),
        )
        let params = gate.parameters()

        #expect(params.count == 3, "U3 with all symbolic should have 3 parameters")
        #expect(params.contains(theta), "U3 should contain theta parameter")
        #expect(params.contains(phi), "U3 should contain phi parameter")
        #expect(params.contains(lambda), "U3 should contain lambda parameter")
    }

    @Test("U3 parameters() returns only symbolic parameters when mixed")
    func u3ParametersWithMixed() {
        let theta = Parameter(name: "theta")
        let lambda = Parameter(name: "lambda")
        let gate = QuantumGate.u3(
            theta: .parameter(theta),
            phi: .value(0.5),
            lambda: .parameter(lambda),
        )
        let params = gate.parameters()

        #expect(params.count == 2, "U3 with two symbolic should have 2 parameters")
        #expect(params.contains(theta), "U3 should contain theta parameter")
        #expect(params.contains(lambda), "U3 should contain lambda parameter")
    }

    @Test("U3 parameters() returns empty for all concrete")
    func u3ParametersWithAllConcrete() {
        let gate = QuantumGate.u3(theta: .value(0.1), phi: .value(0.2), lambda: .value(0.3))
        let params = gate.parameters()

        #expect(params.isEmpty, "U3 with all concrete should have no parameters")
    }
}

/// Test suite for parameter binding in quantum gates.
/// Validates bound(with:) method for phase, U1, U2, U3, and controlled gates,
/// ensuring correct parameter substitution for variational circuits.
@Suite("Parameter Binding")
struct ParameterBindingTests {
    @Test("Phase bound() substitutes symbolic parameter")
    func phaseBoundSubstitutes() {
        let theta = Parameter(name: "theta")
        let gate = QuantumGate.phase(.parameter(theta))
        let bound = gate.bound(with: ["theta": .pi / 4])

        #expect(!bound.isParameterized, "Bound phase should not be parameterized")

        let expected = QuantumGate.phase(.pi / 4).matrix()
        let actual = bound.matrix()
        #expect(QuantumGate.matricesEqual(expected, actual), "Bound phase matrix should match")
    }

    @Test("U1 bound() substitutes symbolic parameter")
    func u1BoundSubstitutes() {
        let lambda = Parameter(name: "lambda")
        let gate = QuantumGate.u1(lambda: .parameter(lambda))
        let bound = gate.bound(with: ["lambda": .pi / 4])

        #expect(!bound.isParameterized, "Bound U1 should not be parameterized")

        let expected = QuantumGate.u1(lambda: .pi / 4).matrix()
        let actual = bound.matrix()
        #expect(QuantumGate.matricesEqual(expected, actual), "Bound U1 matrix should match")
    }

    @Test("U2 bound() substitutes both symbolic parameters")
    func u2BoundSubstitutes() {
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")
        let gate = QuantumGate.u2(phi: .parameter(phi), lambda: .parameter(lambda))
        let bound = gate.bound(with: ["phi": .pi / 2, "lambda": .pi / 4])

        #expect(!bound.isParameterized, "Bound U2 should not be parameterized")

        let expected = QuantumGate.u2(phi: .pi / 2, lambda: .pi / 4).matrix()
        let actual = bound.matrix()
        #expect(QuantumGate.matricesEqual(expected, actual), "Bound U2 matrix should match")
    }

    @Test("U3 bound() substitutes all three symbolic parameters")
    func u3BoundSubstitutes() {
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")
        let gate = QuantumGate.u3(
            theta: .parameter(theta),
            phi: .parameter(phi),
            lambda: .parameter(lambda),
        )
        let bound = gate.bound(with: ["theta": .pi / 3, "phi": .pi / 4, "lambda": .pi / 6])

        #expect(!bound.isParameterized, "Bound U3 should not be parameterized")

        let expected = QuantumGate.u3(theta: .pi / 3, phi: .pi / 4, lambda: .pi / 6).matrix()
        let actual = bound.matrix()
        #expect(QuantumGate.matricesEqual(expected, actual), "Bound U3 matrix should match")
    }

    @Test("Controlled phase bound() substitutes symbolic parameter")
    func controlledPhaseBoundSubstitutes() {
        let theta = Parameter(name: "theta")
        let gate = QuantumGate.controlledPhase(.parameter(theta))
        let bound = gate.bound(with: ["theta": .pi / 2])

        #expect(!bound.isParameterized, "Bound controlled phase should not be parameterized")

        let expected = QuantumGate.controlledPhase(.pi / 2).matrix()
        let actual = bound.matrix()
        #expect(QuantumGate.matricesEqual(expected, actual), "Bound controlled phase matrix should match")
    }

    @Test("Controlled rotation X bound() substitutes symbolic parameter")
    func controlledRotationXBoundSubstitutes() {
        let theta = Parameter(name: "theta")
        let gate = QuantumGate.controlledRotationX(.parameter(theta))
        let bound = gate.bound(with: ["theta": .pi / 4])

        #expect(!bound.isParameterized, "Bound CRx should not be parameterized")

        let expected = QuantumGate.controlledRotationX(.pi / 4).matrix()
        let actual = bound.matrix()
        #expect(QuantumGate.matricesEqual(expected, actual), "Bound CRx matrix should match")
    }

    @Test("Controlled rotation Y bound() substitutes symbolic parameter")
    func controlledRotationYBoundSubstitutes() {
        let theta = Parameter(name: "theta")
        let gate = QuantumGate.controlledRotationY(.parameter(theta))
        let bound = gate.bound(with: ["theta": .pi / 3])

        #expect(!bound.isParameterized, "Bound CRy should not be parameterized")

        let expected = QuantumGate.controlledRotationY(.pi / 3).matrix()
        let actual = bound.matrix()
        #expect(QuantumGate.matricesEqual(expected, actual), "Bound CRy matrix should match")
    }

    @Test("Controlled rotation Z bound() substitutes symbolic parameter")
    func controlledRotationZBoundSubstitutes() {
        let theta = Parameter(name: "theta")
        let gate = QuantumGate.controlledRotationZ(.parameter(theta))
        let bound = gate.bound(with: ["theta": .pi / 6])

        #expect(!bound.isParameterized, "Bound CRz should not be parameterized")

        let expected = QuantumGate.controlledRotationZ(.pi / 6).matrix()
        let actual = bound.matrix()
        #expect(QuantumGate.matricesEqual(expected, actual), "Bound CRz matrix should match")
    }
}

/// Test suite for custom gate matrix retrieval.
/// Validates that custom single-qubit and two-qubit gates
/// correctly return their provided matrices.
@Suite("Custom Gate Matrix")
struct CustomGateMatrixTests {
    @Test("Custom two-qubit gate matrix() returns provided matrix")
    func customTwoQubitMatrixReturnsProvided() {
        let customMatrix: [[Complex<Double>]] = [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .zero, .one],
            [.zero, .zero, .one, .zero],
        ]

        let gate = QuantumGate.customTwoQubit(matrix: customMatrix)
        let retrieved = gate.matrix()

        #expect(QuantumGate.matricesEqual(customMatrix, retrieved),
                "Custom two-qubit gate should return its provided matrix")
    }

    @Test("Custom two-qubit gate preserves complex values")
    func customTwoQubitPreservesComplexValues() {
        let c = Complex<Double>(0.5, 0.5)
        let d = Complex<Double>(0.5, -0.5)
        let customMatrix: [[Complex<Double>]] = [
            [.one, .zero, .zero, .zero],
            [.zero, c, d, .zero],
            [.zero, d, c, .zero],
            [.zero, .zero, .zero, .one],
        ]

        let gate = QuantumGate.customTwoQubit(matrix: customMatrix)
        let retrieved = gate.matrix()

        #expect(retrieved[1][1].real == c.real, "Complex real part should be preserved")
        #expect(retrieved[1][1].imaginary == c.imaginary, "Complex imaginary part should be preserved")
        #expect(retrieved[1][2].real == d.real, "Complex real part should be preserved")
        #expect(retrieved[1][2].imaginary == d.imaginary, "Complex imaginary part should be preserved")
    }
}
