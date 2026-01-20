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

/// Test suite for gate inverse property.
/// Validates U†U = I for all gate types including parameterized gates,
/// custom gates, and gates with symbolic parameters.
@Suite("Gate Inverse Property")
struct GateInverseTests {
    @Test("Hermitian gates are self-inverse")
    func hermitianGatesSelfInverse() {
        let hermitianGates: [QuantumGate] = [
            .identity, .pauliX, .pauliY, .pauliZ, .hadamard,
            .swap, .cnot, .cz, .toffoli,
        ]

        for gate in hermitianGates {
            #expect(gate.inverse == gate, "\(gate) should be self-inverse")
        }
    }

    @Test("Phase gate inverse negates angle")
    func phaseInverseNegatesAngle() {
        let angle = Double.pi / 4
        let gate = QuantumGate.phase(angle)
        let inverse = gate.inverse

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "P(θ)P(-θ) should equal I")
    }

    @Test("Phase gate inverse with symbolic parameter")
    func phaseInverseSymbolic() {
        let theta = Parameter(name: "theta")
        let gate = QuantumGate.phase(.parameter(theta))
        let inverse = gate.inverse

        if case let .phase(negatedAngle) = inverse {
            #expect(negatedAngle.isSymbolic, "Inverse should have symbolic parameter")
            if case let .negatedParameter(p) = negatedAngle {
                #expect(p == theta, "Should negate the same parameter")
            }
        }
    }

    @Test("S gate inverse is Phase(-π/2)")
    func sGateInverse() {
        let gate = QuantumGate.sGate
        let inverse = gate.inverse

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "S·S† should equal I")
    }

    @Test("T gate inverse is Phase(-π/4)")
    func tGateInverse() {
        let gate = QuantumGate.tGate
        let inverse = gate.inverse

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "T·T† should equal I")
    }

    @Test("RotationX inverse negates angle")
    func rotationXInverse() {
        let angle = Double.pi / 3
        let gate = QuantumGate.rotationX(angle)
        let inverse = gate.inverse

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "Rx(θ)Rx(-θ) should equal I")
    }

    @Test("RotationX inverse with symbolic parameter")
    func rotationXInverseSymbolic() {
        let theta = Parameter(name: "theta")
        let gate = QuantumGate.rotationX(.parameter(theta))
        let inverse = gate.inverse

        if case let .rotationX(negatedAngle) = inverse {
            #expect(negatedAngle.isSymbolic, "Inverse should have symbolic parameter")
        }
    }

    @Test("RotationY inverse negates angle")
    func rotationYInverse() {
        let angle = Double.pi / 5
        let gate = QuantumGate.rotationY(angle)
        let inverse = gate.inverse

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "Ry(θ)Ry(-θ) should equal I")
    }

    @Test("RotationZ inverse negates angle")
    func rotationZInverse() {
        let angle = Double.pi / 6
        let gate = QuantumGate.rotationZ(angle)
        let inverse = gate.inverse

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "Rz(θ)Rz(-θ) should equal I")
    }

    @Test("U1 inverse negates lambda")
    func u1Inverse() {
        let lambda = Double.pi / 4
        let gate = QuantumGate.u1(lambda: lambda)
        let inverse = gate.inverse

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "U1(λ)U1(-λ) should equal I")
    }

    @Test("U1 inverse with symbolic parameter")
    func u1InverseSymbolic() {
        let lambda = Parameter(name: "lambda")
        let gate = QuantumGate.u1(lambda: .parameter(lambda))
        let inverse = gate.inverse

        if case let .u1(negatedLambda) = inverse {
            #expect(negatedLambda.isSymbolic, "Inverse should have symbolic parameter")
        }
    }

    @Test("U2 inverse returns U3 with negated theta")
    func u2Inverse() {
        let phi = Double.pi / 4
        let lambda = Double.pi / 3
        let gate = QuantumGate.u2(phi: phi, lambda: lambda)
        let inverse = gate.inverse

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "U2(φ,λ)U2†(φ,λ) should equal I")
    }

    @Test("U2 inverse with symbolic parameters returns U3")
    func u2InverseSymbolic() {
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")
        let gate = QuantumGate.u2(phi: .parameter(phi), lambda: .parameter(lambda))
        let inverse = gate.inverse

        if case let .u3(newTheta, newPhi, newLambda) = inverse {
            if case let .value(thetaVal) = newTheta {
                #expect(abs(thetaVal + .pi / 2) < 1e-10, "Theta should be -π/2")
            }
            #expect(newPhi.isSymbolic, "Inverse phi should be symbolic")
            #expect(newLambda.isSymbolic, "Inverse lambda should be symbolic")
        }
    }

    @Test("U3 inverse negates and reorders parameters")
    func u3Inverse() {
        let theta = Double.pi / 4
        let phi = Double.pi / 3
        let lambda = Double.pi / 6
        let gate = QuantumGate.u3(theta: theta, phi: phi, lambda: lambda)
        let inverse = gate.inverse

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "U3(θ,φ,λ)U3†(θ,φ,λ) should equal I")
    }

    @Test("U3 inverse with symbolic parameters")
    func u3InverseSymbolic() {
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")
        let gate = QuantumGate.u3(theta: .parameter(theta), phi: .parameter(phi), lambda: .parameter(lambda))
        let inverse = gate.inverse

        if case let .u3(newTheta, newPhi, newLambda) = inverse {
            #expect(newTheta.isSymbolic, "Inverse theta should be symbolic")
            #expect(newPhi.isSymbolic, "Inverse phi should be symbolic")
            #expect(newLambda.isSymbolic, "Inverse lambda should be symbolic")
        }
    }

    @Test("SX gate inverse satisfies SX·SX† = I")
    func sxInverse() {
        let gate = QuantumGate.sx
        let inverse = gate.inverse

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "SX·SX† should equal I")
    }

    @Test("SY gate inverse satisfies SY·SY† = I")
    func syInverse() {
        let gate = QuantumGate.sy
        let inverse = gate.inverse

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "SY·SY† should equal I")
    }

    @Test("Controlled phase inverse negates angle")
    func controlledPhaseInverse() {
        let angle = Double.pi / 3
        let gate = QuantumGate.controlledPhase(angle)
        let inverse = gate.inverse

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "CP(θ)CP(-θ) should equal I")
    }

    @Test("Controlled phase inverse with symbolic parameter")
    func controlledPhaseInverseSymbolic() {
        let theta = Parameter(name: "theta")
        let gate = QuantumGate.controlledPhase(.parameter(theta))
        let inverse = gate.inverse

        if case let .controlledPhase(negatedAngle) = inverse {
            #expect(negatedAngle.isSymbolic, "Inverse should have symbolic parameter")
        }
    }

    @Test("Controlled rotation X inverse negates angle")
    func controlledRotationXInverse() {
        let angle = Double.pi / 4
        let gate = QuantumGate.controlledRotationX(angle)
        let inverse = gate.inverse

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "CRx(θ)CRx(-θ) should equal I")
    }

    @Test("Controlled rotation X inverse with symbolic parameter")
    func controlledRotationXInverseSymbolic() {
        let theta = Parameter(name: "theta")
        let gate = QuantumGate.controlledRotationX(.parameter(theta))
        let inverse = gate.inverse

        if case let .controlledRotationX(negatedAngle) = inverse {
            #expect(negatedAngle.isSymbolic, "Inverse should have symbolic parameter")
        }
    }

    @Test("Controlled rotation Y inverse negates angle")
    func controlledRotationYInverse() {
        let angle = Double.pi / 5
        let gate = QuantumGate.controlledRotationY(angle)
        let inverse = gate.inverse

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "CRy(θ)CRy(-θ) should equal I")
    }

    @Test("Controlled rotation Y inverse with symbolic parameter")
    func controlledRotationYInverseSymbolic() {
        let theta = Parameter(name: "theta")
        let gate = QuantumGate.controlledRotationY(.parameter(theta))
        let inverse = gate.inverse

        if case let .controlledRotationY(negatedAngle) = inverse {
            #expect(negatedAngle.isSymbolic, "Inverse should have symbolic parameter")
        }
    }

    @Test("Controlled rotation Z inverse negates angle")
    func controlledRotationZInverse() {
        let angle = Double.pi / 6
        let gate = QuantumGate.controlledRotationZ(angle)
        let inverse = gate.inverse

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "CRz(θ)CRz(-θ) should equal I")
    }

    @Test("Controlled rotation Z inverse with symbolic parameter")
    func controlledRotationZInverseSymbolic() {
        let theta = Parameter(name: "theta")
        let gate = QuantumGate.controlledRotationZ(.parameter(theta))
        let inverse = gate.inverse

        if case let .controlledRotationZ(negatedAngle) = inverse {
            #expect(negatedAngle.isSymbolic, "Inverse should have symbolic parameter")
        }
    }

    @Test("CY gate is self-inverse")
    func cyInverse() {
        let gate = QuantumGate.cy
        let inverse = gate.inverse

        #expect(gate == inverse, "CY should be self-inverse")

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "CY·CY should equal I")
    }

    @Test("CH gate is self-inverse")
    func chInverse() {
        let gate = QuantumGate.ch
        let inverse = gate.inverse

        #expect(gate == inverse, "CH should be self-inverse")

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "CH·CH should equal I")
    }

    @Test("sqrtSwap inverse satisfies √SWAP·√SWAP† = I")
    func sqrtSwapInverse() {
        let gate = QuantumGate.sqrtSwap
        let inverse = gate.inverse

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "√SWAP·√SWAP† should equal I")
    }

    @Test("Custom single-qubit gate inverse is hermitian conjugate")
    func customSingleQubitInverse() {
        let angle = Double.pi / 6
        let customMatrix: [[Complex<Double>]] = [
            [Complex(cos(angle), 0), Complex(-sin(angle), 0)],
            [Complex(sin(angle), 0), Complex(cos(angle), 0)],
        ]

        let gate = QuantumGate.customSingleQubit(matrix: customMatrix)
        let inverse = gate.inverse

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "Custom·Custom† should equal I")
    }

    @Test("Custom two-qubit gate inverse is hermitian conjugate")
    func customTwoQubitInverse() {
        let c = Complex<Double>(0.5, 0.5)
        let d = Complex<Double>(0.5, -0.5)
        let customMatrix: [[Complex<Double>]] = [
            [.one, .zero, .zero, .zero],
            [.zero, c, d, .zero],
            [.zero, d, c, .zero],
            [.zero, .zero, .zero, .one],
        ]

        let gate = QuantumGate.customTwoQubit(matrix: customMatrix)
        let inverse = gate.inverse

        let product = QuantumGate.matrixMultiply(gate.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product), "Custom·Custom† should equal I")
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

@Suite("Controlled Gate isNativeGate")
struct ControlledGateIsNativeGateTests {
    @Test("Controlled gate returns false for isNativeGate")
    func controlledReturnsFalse() {
        let gate = QuantumGate.controlled(gate: .pauliX, controls: [0])
        #expect(!gate.isNativeGate, "Controlled gate should not be a native gate")
    }

    @Test("PauliX returns true for isNativeGate")
    func pauliXReturnsTrue() {
        #expect(QuantumGate.pauliX.isNativeGate, "PauliX should be a native gate")
    }

    @Test("PauliY returns true for isNativeGate")
    func pauliYReturnsTrue() {
        #expect(QuantumGate.pauliY.isNativeGate, "PauliY should be a native gate")
    }

    @Test("PauliZ returns true for isNativeGate")
    func pauliZReturnsTrue() {
        #expect(QuantumGate.pauliZ.isNativeGate, "PauliZ should be a native gate")
    }

    @Test("Hadamard returns true for isNativeGate")
    func hadamardReturnsTrue() {
        #expect(QuantumGate.hadamard.isNativeGate, "Hadamard should be a native gate")
    }

    @Test("CNOT returns true for isNativeGate")
    func cnotReturnsTrue() {
        #expect(QuantumGate.cnot.isNativeGate, "CNOT should be a native gate")
    }

    @Test("Toffoli returns true for isNativeGate")
    func toffoliReturnsTrue() {
        #expect(QuantumGate.toffoli.isNativeGate, "Toffoli should be a native gate")
    }

    @Test("Phase returns true for isNativeGate")
    func phaseReturnsTrue() {
        #expect(QuantumGate.phase(.pi / 4).isNativeGate, "Phase should be a native gate")
    }

    @Test("RotationX returns true for isNativeGate")
    func rotationXReturnsTrue() {
        #expect(QuantumGate.rotationX(.pi / 3).isNativeGate, "RotationX should be a native gate")
    }

    @Test("RotationY returns true for isNativeGate")
    func rotationYReturnsTrue() {
        #expect(QuantumGate.rotationY(.pi / 3).isNativeGate, "RotationY should be a native gate")
    }

    @Test("RotationZ returns true for isNativeGate")
    func rotationZReturnsTrue() {
        #expect(QuantumGate.rotationZ(.pi / 3).isNativeGate, "RotationZ should be a native gate")
    }

    @Test("SWAP returns true for isNativeGate")
    func swapReturnsTrue() {
        #expect(QuantumGate.swap.isNativeGate, "SWAP should be a native gate")
    }

    @Test("CZ returns true for isNativeGate")
    func czReturnsTrue() {
        #expect(QuantumGate.cz.isNativeGate, "CZ should be a native gate")
    }

    @Test("Identity returns true for isNativeGate")
    func identityReturnsTrue() {
        #expect(QuantumGate.identity.isNativeGate, "Identity should be a native gate")
    }

    @Test("S gate returns true for isNativeGate")
    func sGateReturnsTrue() {
        #expect(QuantumGate.sGate.isNativeGate, "S gate should be a native gate")
    }

    @Test("T gate returns true for isNativeGate")
    func tGateReturnsTrue() {
        #expect(QuantumGate.tGate.isNativeGate, "T gate should be a native gate")
    }
}

@Suite("Controlled Gate flattenControlled")
struct ControlledGateFlattenControlledTests {
    @Test("Non-controlled gate returns self with empty controls")
    func nonControlledReturnsSelf() {
        let gate = QuantumGate.pauliX
        let (baseGate, controls) = gate.flattenControlled()

        #expect(baseGate == .pauliX, "Base gate should be pauliX")
        #expect(controls.isEmpty, "Controls should be empty for non-controlled gate")
    }

    @Test("Hadamard returns self with empty controls")
    func hadamardReturnsSelf() {
        let gate = QuantumGate.hadamard
        let (baseGate, controls) = gate.flattenControlled()

        #expect(baseGate == .hadamard, "Base gate should be hadamard")
        #expect(controls.isEmpty, "Controls should be empty for hadamard")
    }

    @Test("Single controlled returns base gate and controls")
    func singleControlledReturnsBaseAndControls() {
        let gate = QuantumGate.controlled(gate: .pauliX, controls: [0])
        let (baseGate, controls) = gate.flattenControlled()

        #expect(baseGate == .pauliX, "Base gate should be pauliX")
        #expect(controls == [0], "Controls should be [0]")
    }

    @Test("Single controlled with multiple controls")
    func singleControlledMultipleControls() {
        let gate = QuantumGate.controlled(gate: .hadamard, controls: [0, 1])
        let (baseGate, controls) = gate.flattenControlled()

        #expect(baseGate == .hadamard, "Base gate should be hadamard")
        #expect(controls == [0, 1], "Controls should be [0, 1]")
    }

    @Test("Nested controlled flattens all controls")
    func nestedControlledFlattensAll() {
        let inner = QuantumGate.controlled(gate: .pauliZ, controls: [1])
        let outer = QuantumGate.controlled(gate: inner, controls: [0])
        let (baseGate, controls) = outer.flattenControlled()

        #expect(baseGate == .pauliZ, "Base gate should be pauliZ")
        #expect(controls == [0, 1], "Controls should be [0, 1]")
    }

    @Test("Deeply nested controlled accumulates all controls")
    func deeplyNestedAccumulatesAllControls() {
        let level1 = QuantumGate.controlled(gate: .rotationX(.pi / 4), controls: [2])
        let level2 = QuantumGate.controlled(gate: level1, controls: [1])
        let level3 = QuantumGate.controlled(gate: level2, controls: [0])
        let (baseGate, controls) = level3.flattenControlled()

        if case let .rotationX(angle) = baseGate {
            #expect(abs(angle.evaluate(using: [:]) - .pi / 4) < 1e-10, "Base gate should be rotationX(pi/4)")
        }
        #expect(controls == [0, 1, 2], "Controls should be [0, 1, 2]")
    }

    @Test("Three levels of nesting preserves order")
    func threeLevelsPreservesOrder() {
        let level1 = QuantumGate.controlled(gate: .pauliY, controls: [3])
        let level2 = QuantumGate.controlled(gate: level1, controls: [2])
        let level3 = QuantumGate.controlled(gate: level2, controls: [0, 1])
        let (baseGate, controls) = level3.flattenControlled()

        #expect(baseGate == .pauliY, "Base gate should be pauliY")
        #expect(controls == [0, 1, 2, 3], "Controls should be [0, 1, 2, 3]")
    }
}

@Suite("Controlled Gate Properties")
struct ControlledGatePropertiesTests {
    @Test("qubitsRequired equals baseGate qubits plus control count")
    func qubitsRequiredEqualsBaseGatePlusControlCount() {
        let singleControl = QuantumGate.controlled(gate: .pauliX, controls: [0])
        #expect(singleControl.qubitsRequired == 2, "Single-controlled X should require 2 qubits")

        let doubleControl = QuantumGate.controlled(gate: .pauliX, controls: [0, 1])
        #expect(doubleControl.qubitsRequired == 3, "Double-controlled X should require 3 qubits")

        let tripleControl = QuantumGate.controlled(gate: .pauliX, controls: [0, 1, 2])
        #expect(tripleControl.qubitsRequired == 4, "Triple-controlled X should require 4 qubits")
    }

    @Test("qubitsRequired with two-qubit base gate")
    func qubitsRequiredWithTwoQubitBase() {
        let controlledSwap = QuantumGate.controlled(gate: .swap, controls: [0])
        #expect(controlledSwap.qubitsRequired == 3, "Controlled SWAP should require 3 qubits")

        let doubleControlledCnot = QuantumGate.controlled(gate: .cnot, controls: [0, 1])
        #expect(doubleControlledCnot.qubitsRequired == 4, "Double-controlled CNOT should require 4 qubits")
    }

    @Test("parameters returns inner gate parameters")
    func parametersReturnsInnerGateParameters() {
        let theta = Parameter(name: "theta")
        let controlledRy = QuantumGate.controlled(gate: .rotationY(.parameter(theta)), controls: [0])
        let params = controlledRy.parameters()

        #expect(params.count == 1, "Controlled Ry should have 1 parameter")
        #expect(params.contains(theta), "Parameters should contain theta")
    }

    @Test("parameters returns empty for non-parameterized inner gate")
    func parametersReturnsEmptyForNonParameterizedInner() {
        let controlledX = QuantumGate.controlled(gate: .pauliX, controls: [0])
        let params = controlledX.parameters()

        #expect(params.isEmpty, "Controlled X should have no parameters")
    }

    @Test("parameters returns multiple parameters from inner gate")
    func parametersReturnsMultipleFromInner() {
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")
        let u3 = QuantumGate.u3(
            theta: .parameter(theta),
            phi: .parameter(phi),
            lambda: .parameter(lambda),
        )
        let controlledU3 = QuantumGate.controlled(gate: u3, controls: [0])
        let params = controlledU3.parameters()

        #expect(params.count == 3, "Controlled U3 should have 3 parameters")
        #expect(params.contains(theta), "Parameters should contain theta")
        #expect(params.contains(phi), "Parameters should contain phi")
        #expect(params.contains(lambda), "Parameters should contain lambda")
    }

    @Test("bound binds inner gate and preserves controls")
    func boundBindsInnerAndPreservesControls() {
        let theta = Parameter(name: "theta")
        let controlledRy = QuantumGate.controlled(gate: .rotationY(.parameter(theta)), controls: [0, 1])
        let bound = controlledRy.bound(with: ["theta": .pi / 4])

        #expect(!bound.isParameterized, "Bound gate should not be parameterized")

        if case let .controlled(innerGate, controls) = bound {
            #expect(controls == [0, 1], "Controls should be preserved after binding")
            if case let .rotationY(angle) = innerGate {
                #expect(abs(angle.evaluate(using: [:]) - .pi / 4) < 1e-10, "Inner gate angle should be pi/4")
            }
        }
    }

    @Test("inverse inverts inner gate and preserves controls")
    func inverseInvertsInnerAndPreservesControls() {
        let controlledRz = QuantumGate.controlled(gate: .rotationZ(.pi / 3), controls: [0, 1])
        let inverse = controlledRz.inverse

        if case let .controlled(innerGate, controls) = inverse {
            #expect(controls == [0, 1], "Controls should be preserved after inversion")
            if case let .rotationZ(angle) = innerGate {
                #expect(abs(angle.evaluate(using: [:]) + .pi / 3) < 1e-10, "Inverse angle should be -pi/3")
            }
        }
    }

    @Test("inverse of controlled Hermitian gate is self")
    func inverseOfControlledHermitianIsSelf() {
        let controlledX = QuantumGate.controlled(gate: .pauliX, controls: [0])
        let inverse = controlledX.inverse

        #expect(inverse == controlledX, "Inverse of controlled-X should equal itself")
    }

    @Test("description format is C^n(gate)")
    func descriptionFormatIsCN() {
        let singleControl = QuantumGate.controlled(gate: .pauliX, controls: [0])
        #expect(singleControl.description == "C^1(X)", "Single controlled X description should be C^1(X)")

        let doubleControl = QuantumGate.controlled(gate: .hadamard, controls: [0, 1])
        #expect(doubleControl.description == "C^2(H)", "Double controlled H description should be C^2(H)")

        let tripleControl = QuantumGate.controlled(gate: .pauliZ, controls: [0, 1, 2])
        #expect(tripleControl.description == "C^3(Z)", "Triple controlled Z description should be C^3(Z)")
    }

    @Test("description with parameterized inner gate")
    func descriptionWithParameterizedInner() {
        let theta = Parameter(name: "theta")
        let controlledRy = QuantumGate.controlled(gate: .rotationY(.parameter(theta)), controls: [0])
        #expect(controlledRy.description.contains("C^1"), "Description should contain C^1")
        #expect(controlledRy.description.contains("Ry"), "Description should contain Ry")
        #expect(controlledRy.description.contains("theta"), "Description should contain theta")
    }
}

@Suite("Controlled Gate Matrix")
struct ControlledGateMatrixTests {
    @Test("Single-controlled X matches CNOT matrix")
    func singleControlledXMatchesCNOT() {
        let controlledX = QuantumGate.controlled(gate: .pauliX, controls: [0])
        let cnotGate = QuantumGate.cnot

        let controlledMatrix = controlledX.matrix()
        let cnotMatrix = cnotGate.matrix()

        #expect(QuantumGate.matricesEqual(controlledMatrix, cnotMatrix),
                "Controlled-X matrix should match CNOT matrix")
    }

    @Test("Single-controlled Z matches CZ matrix")
    func singleControlledZMatchesCZ() {
        let controlledZ = QuantumGate.controlled(gate: .pauliZ, controls: [0])
        let czGate = QuantumGate.cz

        let controlledMatrix = controlledZ.matrix()
        let czMatrix = czGate.matrix()

        #expect(QuantumGate.matricesEqual(controlledMatrix, czMatrix),
                "Controlled-Z matrix should match CZ matrix")
    }

    @Test("Double-controlled X matches Toffoli matrix")
    func doubleControlledXMatchesToffoli() {
        let controlledX = QuantumGate.controlled(gate: .pauliX, controls: [0, 1])
        let toffoliGate = QuantumGate.toffoli

        let controlledMatrix = controlledX.matrix()
        let toffoliMatrix = toffoliGate.matrix()

        #expect(QuantumGate.matricesEqual(controlledMatrix, toffoliMatrix),
                "Double-controlled-X matrix should match Toffoli matrix")
    }

    @Test("Identity on non-controlled subspace for single control")
    func identityOnNonControlledSubspaceSingleControl() {
        let controlledX = QuantumGate.controlled(gate: .pauliX, controls: [0])
        let matrix = controlledX.matrix()

        #expect(abs(matrix[0][0].real - 1.0) < 1e-10, "matrix[0][0] should be 1")
        #expect(abs(matrix[0][0].imaginary) < 1e-10, "matrix[0][0] imaginary should be 0")
        #expect(abs(matrix[1][1].real - 1.0) < 1e-10, "matrix[1][1] should be 1")
        #expect(abs(matrix[1][1].imaginary) < 1e-10, "matrix[1][1] imaginary should be 0")

        for i in 0 ..< 2 {
            for j in 0 ..< 4 {
                if i != j {
                    #expect(abs(matrix[i][j].magnitude) < 1e-10,
                            "Off-diagonal elements in non-controlled subspace should be 0")
                }
            }
        }
    }

    @Test("Gate action on controlled subspace for single control")
    func gateActionOnControlledSubspaceSingleControl() {
        let controlledX = QuantumGate.controlled(gate: .pauliX, controls: [0])
        let matrix = controlledX.matrix()

        #expect(abs(matrix[2][3].real - 1.0) < 1e-10, "matrix[2][3] should be 1 (X gate action)")
        #expect(abs(matrix[3][2].real - 1.0) < 1e-10, "matrix[3][2] should be 1 (X gate action)")
        #expect(abs(matrix[2][2].magnitude) < 1e-10, "matrix[2][2] should be 0")
        #expect(abs(matrix[3][3].magnitude) < 1e-10, "matrix[3][3] should be 0")
    }

    @Test("Identity on non-controlled subspace for double control")
    func identityOnNonControlledSubspaceDoubleControl() {
        let controlledX = QuantumGate.controlled(gate: .pauliX, controls: [0, 1])
        let matrix = controlledX.matrix()

        for i in 0 ..< 6 {
            #expect(abs(matrix[i][i].real - 1.0) < 1e-10, "Diagonal element \(i) should be 1")
            #expect(abs(matrix[i][i].imaginary) < 1e-10, "Diagonal element \(i) imaginary should be 0")
            for j in 0 ..< 8 {
                if i != j {
                    #expect(abs(matrix[i][j].magnitude) < 1e-10,
                            "Off-diagonal element [\(i)][\(j)] in non-controlled subspace should be 0")
                }
            }
        }
    }

    @Test("Gate action on controlled subspace for double control")
    func gateActionOnControlledSubspaceDoubleControl() {
        let controlledX = QuantumGate.controlled(gate: .pauliX, controls: [0, 1])
        let matrix = controlledX.matrix()

        #expect(abs(matrix[6][7].real - 1.0) < 1e-10, "matrix[6][7] should be 1 (X gate action)")
        #expect(abs(matrix[7][6].real - 1.0) < 1e-10, "matrix[7][6] should be 1 (X gate action)")
        #expect(abs(matrix[6][6].magnitude) < 1e-10, "matrix[6][6] should be 0")
        #expect(abs(matrix[7][7].magnitude) < 1e-10, "matrix[7][7] should be 0")
    }

    @Test("Controlled gate matrix is unitary")
    func controlledGateMatrixIsUnitary() {
        let controlledH = QuantumGate.controlled(gate: .hadamard, controls: [0])
        let matrix = controlledH.matrix()

        #expect(QuantumGate.isUnitary(matrix), "Controlled Hadamard matrix should be unitary")
    }

    @Test("Double-controlled gate matrix is unitary")
    func doubleControlledGateMatrixIsUnitary() {
        let controlledH = QuantumGate.controlled(gate: .hadamard, controls: [0, 1])
        let matrix = controlledH.matrix()

        #expect(QuantumGate.isUnitary(matrix), "Double-controlled Hadamard matrix should be unitary")
    }

    @Test("Controlled rotation gate matrix is unitary")
    func controlledRotationMatrixIsUnitary() {
        let controlledRy = QuantumGate.controlled(gate: .rotationY(.pi / 4), controls: [0])
        let matrix = controlledRy.matrix()

        #expect(QuantumGate.isUnitary(matrix), "Controlled Ry matrix should be unitary")
    }

    @Test("Controlled gate matrix has correct dimensions")
    func controlledGateMatrixHasCorrectDimensions() {
        let singleControl = QuantumGate.controlled(gate: .pauliX, controls: [0])
        let singleMatrix = singleControl.matrix()
        #expect(singleMatrix.count == 4, "Single-controlled gate should have 4x4 matrix")
        #expect(singleMatrix.allSatisfy { $0.count == 4 }, "All rows should have 4 columns")

        let doubleControl = QuantumGate.controlled(gate: .pauliX, controls: [0, 1])
        let doubleMatrix = doubleControl.matrix()
        #expect(doubleMatrix.count == 8, "Double-controlled gate should have 8x8 matrix")
        #expect(doubleMatrix.allSatisfy { $0.count == 8 }, "All rows should have 8 columns")

        let tripleControl = QuantumGate.controlled(gate: .pauliX, controls: [0, 1, 2])
        let tripleMatrix = tripleControl.matrix()
        #expect(tripleMatrix.count == 16, "Triple-controlled gate should have 16x16 matrix")
        #expect(tripleMatrix.allSatisfy { $0.count == 16 }, "All rows should have 16 columns")
    }

    @Test("Controlled gate with two-qubit base has correct dimensions")
    func controlledTwoQubitBaseHasCorrectDimensions() {
        let controlledSwap = QuantumGate.controlled(gate: .swap, controls: [0])
        let matrix = controlledSwap.matrix()

        #expect(matrix.count == 8, "Controlled SWAP should have 8x8 matrix")
        #expect(matrix.allSatisfy { $0.count == 8 }, "All rows should have 8 columns")
    }

    @Test("Controlled-controlled gate inverse produces identity")
    func controlledControlledInverseProducesIdentity() {
        let controlledRz = QuantumGate.controlled(gate: .rotationZ(.pi / 5), controls: [0, 1])
        let inverse = controlledRz.inverse

        let product = QuantumGate.matrixMultiply(controlledRz.matrix(), inverse.matrix())
        #expect(QuantumGate.isIdentityMatrix(product),
                "Controlled Rz times its inverse should equal identity")
    }
}
