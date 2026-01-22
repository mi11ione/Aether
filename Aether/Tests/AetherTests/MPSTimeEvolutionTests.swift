// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for TEBDGates ZZ evolution gate.
/// Validates matrix structure, diagonal form, and eigenvalue phases
/// for the exp(-i*theta*ZZ) two-site gate.
@Suite("TEBDGates ZZ Evolution")
struct TEBDGatesZZEvolutionTests {
    @Test("ZZ gate is 4x4 matrix")
    func zzGateIs4x4() {
        let gate = TEBDGates.zzEvolution(angle: 0.1)
        #expect(gate.count == 4, "ZZ gate should have 4 rows")
        #expect(gate[0].count == 4, "ZZ gate should have 4 columns")
        #expect(gate[1].count == 4, "ZZ gate row 1 should have 4 columns")
        #expect(gate[2].count == 4, "ZZ gate row 2 should have 4 columns")
        #expect(gate[3].count == 4, "ZZ gate row 3 should have 4 columns")
    }

    @Test("ZZ gate is diagonal")
    func zzGateIsDiagonal() {
        let gate = TEBDGates.zzEvolution(angle: 0.3)
        for row in 0 ..< 4 {
            for col in 0 ..< 4 {
                if row != col {
                    #expect(gate[row][col].magnitudeSquared < 1e-20, "Off-diagonal element [\(row)][\(col)] should be zero")
                }
            }
        }
    }

    @Test("ZZ gate diagonal elements have correct phases")
    func zzGateDiagonalPhases() {
        let angle = 0.5
        let gate = TEBDGates.zzEvolution(angle: angle)
        let expMinus = Complex(cos(angle), -sin(angle))
        let expPlus = Complex(cos(angle), sin(angle))

        #expect(abs(gate[0][0].real - expMinus.real) < 1e-10, "Gate[0][0] real part should be cos(angle)")
        #expect(abs(gate[0][0].imaginary - expMinus.imaginary) < 1e-10, "Gate[0][0] imaginary part should be -sin(angle)")
        #expect(abs(gate[1][1].real - expPlus.real) < 1e-10, "Gate[1][1] real part should be cos(angle)")
        #expect(abs(gate[1][1].imaginary - expPlus.imaginary) < 1e-10, "Gate[1][1] imaginary part should be sin(angle)")
        #expect(abs(gate[2][2].real - expPlus.real) < 1e-10, "Gate[2][2] real part should be cos(angle)")
        #expect(abs(gate[2][2].imaginary - expPlus.imaginary) < 1e-10, "Gate[2][2] imaginary part should be sin(angle)")
        #expect(abs(gate[3][3].real - expMinus.real) < 1e-10, "Gate[3][3] real part should be cos(angle)")
        #expect(abs(gate[3][3].imaginary - expMinus.imaginary) < 1e-10, "Gate[3][3] imaginary part should be -sin(angle)")
    }

    @Test("ZZ gate at zero angle is identity")
    func zzGateZeroAngleIsIdentity() {
        let gate = TEBDGates.zzEvolution(angle: 0.0)
        for i in 0 ..< 4 {
            #expect(abs(gate[i][i].real - 1.0) < 1e-10, "Diagonal element [\(i)][\(i)] should be 1")
            #expect(abs(gate[i][i].imaginary) < 1e-10, "Diagonal element [\(i)][\(i)] imaginary should be 0")
        }
    }
}

/// Test suite for TEBDGates XX evolution gate.
/// Validates matrix structure, block diagonal form, and off-diagonal coupling
/// for the exp(-i*theta*XX) two-site gate.
@Suite("TEBDGates XX Evolution")
struct TEBDGatesXXEvolutionTests {
    @Test("XX gate is 4x4 matrix")
    func xxGateIs4x4() {
        let gate = TEBDGates.xxEvolution(angle: 0.1)
        #expect(gate.count == 4, "XX gate should have 4 rows")
        #expect(gate[0].count == 4, "XX gate should have 4 columns")
    }

    @Test("XX gate has correct diagonal structure")
    func xxGateDiagonalStructure() {
        let angle = 0.4
        let gate = TEBDGates.xxEvolution(angle: angle)
        let c = cos(angle)

        #expect(abs(gate[0][0].real - c) < 1e-10, "Gate[0][0] should be cos(angle)")
        #expect(abs(gate[0][0].imaginary) < 1e-10, "Gate[0][0] should be real")
        #expect(abs(gate[1][1].real - c) < 1e-10, "Gate[1][1] should be cos(angle)")
        #expect(abs(gate[2][2].real - c) < 1e-10, "Gate[2][2] should be cos(angle)")
        #expect(abs(gate[3][3].real - c) < 1e-10, "Gate[3][3] should be cos(angle)")
    }

    @Test("XX gate has correct off-diagonal coupling")
    func xxGateOffDiagonalCoupling() {
        let angle = 0.4
        let gate = TEBDGates.xxEvolution(angle: angle)
        let iSin = Complex<Double>(0, -sin(angle))

        #expect(abs(gate[0][3].real - iSin.real) < 1e-10, "Gate[0][3] real should be 0")
        #expect(abs(gate[0][3].imaginary - iSin.imaginary) < 1e-10, "Gate[0][3] imaginary should be -sin(angle)")
        #expect(abs(gate[1][2].real - iSin.real) < 1e-10, "Gate[1][2] real should be 0")
        #expect(abs(gate[1][2].imaginary - iSin.imaginary) < 1e-10, "Gate[1][2] imaginary should be -sin(angle)")
        #expect(abs(gate[2][1].real - iSin.real) < 1e-10, "Gate[2][1] should match Gate[1][2]")
        #expect(abs(gate[3][0].real - iSin.real) < 1e-10, "Gate[3][0] should match Gate[0][3]")
    }

    @Test("XX gate zero elements are zero")
    func xxGateZeroElements() {
        let gate = TEBDGates.xxEvolution(angle: 0.25)

        #expect(gate[0][1].magnitudeSquared < 1e-20, "Gate[0][1] should be zero")
        #expect(gate[0][2].magnitudeSquared < 1e-20, "Gate[0][2] should be zero")
        #expect(gate[1][0].magnitudeSquared < 1e-20, "Gate[1][0] should be zero")
        #expect(gate[1][3].magnitudeSquared < 1e-20, "Gate[1][3] should be zero")
        #expect(gate[2][0].magnitudeSquared < 1e-20, "Gate[2][0] should be zero")
        #expect(gate[2][3].magnitudeSquared < 1e-20, "Gate[2][3] should be zero")
        #expect(gate[3][1].magnitudeSquared < 1e-20, "Gate[3][1] should be zero")
        #expect(gate[3][2].magnitudeSquared < 1e-20, "Gate[3][2] should be zero")
    }

    @Test("XX gate at zero angle is identity")
    func xxGateZeroAngleIsIdentity() {
        let gate = TEBDGates.xxEvolution(angle: 0.0)
        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                if i == j {
                    #expect(abs(gate[i][j].real - 1.0) < 1e-10, "Diagonal element [\(i)][\(j)] should be 1")
                } else {
                    #expect(gate[i][j].magnitudeSquared < 1e-20, "Off-diagonal element [\(i)][\(j)] should be 0")
                }
            }
        }
    }
}

/// Test suite for TEBDGates X single-site evolution gate.
/// Validates 2x2 matrix structure and rotation matrix form
/// for the exp(-i*theta*X) single-site gate.
@Suite("TEBDGates X Evolution")
struct TEBDGatesXEvolutionTests {
    @Test("X gate is 2x2 matrix")
    func xGateIs2x2() {
        let gate = TEBDGates.xEvolution(angle: 0.1)
        #expect(gate.count == 2, "X gate should have 2 rows")
        #expect(gate[0].count == 2, "X gate should have 2 columns")
        #expect(gate[1].count == 2, "X gate row 1 should have 2 columns")
    }

    @Test("X gate has correct diagonal elements")
    func xGateDiagonal() {
        let angle = 0.6
        let gate = TEBDGates.xEvolution(angle: angle)
        let c = cos(angle)

        #expect(abs(gate[0][0].real - c) < 1e-10, "Gate[0][0] should be cos(angle)")
        #expect(abs(gate[0][0].imaginary) < 1e-10, "Gate[0][0] should be real")
        #expect(abs(gate[1][1].real - c) < 1e-10, "Gate[1][1] should be cos(angle)")
        #expect(abs(gate[1][1].imaginary) < 1e-10, "Gate[1][1] should be real")
    }

    @Test("X gate has correct off-diagonal elements")
    func xGateOffDiagonal() {
        let angle = 0.6
        let gate = TEBDGates.xEvolution(angle: angle)
        let iSin = Complex<Double>(0, -sin(angle))

        #expect(abs(gate[0][1].real - iSin.real) < 1e-10, "Gate[0][1] real should be 0")
        #expect(abs(gate[0][1].imaginary - iSin.imaginary) < 1e-10, "Gate[0][1] imaginary should be -sin(angle)")
        #expect(abs(gate[1][0].real - iSin.real) < 1e-10, "Gate[1][0] real should be 0")
        #expect(abs(gate[1][0].imaginary - iSin.imaginary) < 1e-10, "Gate[1][0] imaginary should be -sin(angle)")
    }

    @Test("X gate at zero angle is identity")
    func xGateZeroAngleIsIdentity() {
        let gate = TEBDGates.xEvolution(angle: 0.0)

        #expect(abs(gate[0][0].real - 1.0) < 1e-10, "Gate[0][0] should be 1")
        #expect(abs(gate[0][0].imaginary) < 1e-10, "Gate[0][0] imaginary should be 0")
        #expect(gate[0][1].magnitudeSquared < 1e-20, "Gate[0][1] should be 0")
        #expect(gate[1][0].magnitudeSquared < 1e-20, "Gate[1][0] should be 0")
        #expect(abs(gate[1][1].real - 1.0) < 1e-10, "Gate[1][1] should be 1")
    }
}

/// Test suite for TEBDGates Z single-site evolution gate.
/// Validates 2x2 diagonal matrix structure and phase elements
/// for the exp(-i*theta*Z) single-site gate.
@Suite("TEBDGates Z Evolution")
struct TEBDGatesZEvolutionTests {
    @Test("Z gate is 2x2 diagonal matrix")
    func zGateIs2x2Diagonal() {
        let gate = TEBDGates.zEvolution(angle: 0.2)
        #expect(gate.count == 2, "Z gate should have 2 rows")
        #expect(gate[0].count == 2, "Z gate should have 2 columns")
        #expect(gate[0][1].magnitudeSquared < 1e-20, "Off-diagonal element [0][1] should be zero")
        #expect(gate[1][0].magnitudeSquared < 1e-20, "Off-diagonal element [1][0] should be zero")
    }

    @Test("Z gate has correct phase elements")
    func zGatePhases() {
        let angle = 0.7
        let gate = TEBDGates.zEvolution(angle: angle)
        let expMinus = Complex(cos(angle), -sin(angle))
        let expPlus = Complex(cos(angle), sin(angle))

        #expect(abs(gate[0][0].real - expMinus.real) < 1e-10, "Gate[0][0] real should be cos(angle)")
        #expect(abs(gate[0][0].imaginary - expMinus.imaginary) < 1e-10, "Gate[0][0] imaginary should be -sin(angle)")
        #expect(abs(gate[1][1].real - expPlus.real) < 1e-10, "Gate[1][1] real should be cos(angle)")
        #expect(abs(gate[1][1].imaginary - expPlus.imaginary) < 1e-10, "Gate[1][1] imaginary should be sin(angle)")
    }
}

/// Test suite for TEBDGates YY evolution gate.
/// Validates matrix structure and sign patterns
/// for the exp(-i*theta*YY) two-site gate.
@Suite("TEBDGates YY Evolution")
struct TEBDGatesYYEvolutionTests {
    @Test("YY gate is 4x4 matrix")
    func yyGateIs4x4() {
        let gate = TEBDGates.yyEvolution(angle: 0.1)
        #expect(gate.count == 4, "YY gate should have 4 rows")
        #expect(gate[0].count == 4, "YY gate should have 4 columns")
    }

    @Test("YY gate has correct coupling pattern")
    func yyGateCouplingPattern() {
        let angle = 0.35
        let gate = TEBDGates.yyEvolution(angle: angle)
        let c = cos(angle)
        let iSinPos = Complex<Double>(0, sin(angle))
        let iSinNeg = Complex<Double>(0, -sin(angle))

        #expect(abs(gate[0][0].real - c) < 1e-10, "Gate[0][0] should be cos(angle)")
        #expect(abs(gate[0][3].real - iSinPos.real) < 1e-10, "Gate[0][3] real should be 0")
        #expect(abs(gate[0][3].imaginary - iSinPos.imaginary) < 1e-10, "Gate[0][3] imaginary should be +sin(angle)")
        #expect(abs(gate[1][2].imaginary - iSinNeg.imaginary) < 1e-10, "Gate[1][2] imaginary should be -sin(angle)")
        #expect(abs(gate[2][1].imaginary - iSinNeg.imaginary) < 1e-10, "Gate[2][1] imaginary should be -sin(angle)")
        #expect(abs(gate[3][0].imaginary - iSinPos.imaginary) < 1e-10, "Gate[3][0] imaginary should be +sin(angle)")
    }
}

/// Test suite for TEBDGates Heisenberg XXZ evolution gate.
/// Validates the combined XX+YY+delta*ZZ interaction structure.
@Suite("TEBDGates Heisenberg XXZ Evolution")
struct TEBDGatesHeisenbergXXZTests {
    @Test("Heisenberg gate is 4x4 matrix")
    func heisenbergGateIs4x4() {
        let gate = TEBDGates.heisenbergXXZ(angle: 0.1, delta: 1.0)
        #expect(gate.count == 4, "Heisenberg gate should have 4 rows")
        #expect(gate[0].count == 4, "Heisenberg gate should have 4 columns")
    }

    @Test("Heisenberg gate corner elements are ZZ phases")
    func heisenbergGateCornerElements() {
        let angle = 0.2
        let delta = 1.5
        let gate = TEBDGates.heisenbergXXZ(angle: angle, delta: delta)
        let zzPhase = delta * angle
        let expMinusZZ = Complex(cos(zzPhase), -sin(zzPhase))

        #expect(abs(gate[0][0].real - expMinusZZ.real) < 1e-10, "Gate[0][0] real should match exp(-i*delta*angle)")
        #expect(abs(gate[0][0].imaginary - expMinusZZ.imaginary) < 1e-10, "Gate[0][0] imaginary should match exp(-i*delta*angle)")
        #expect(abs(gate[3][3].real - expMinusZZ.real) < 1e-10, "Gate[3][3] real should match exp(-i*delta*angle)")
        #expect(abs(gate[3][3].imaginary - expMinusZZ.imaginary) < 1e-10, "Gate[3][3] imaginary should match exp(-i*delta*angle)")
    }

    @Test("Heisenberg gate has block structure")
    func heisenbergGateBlockStructure() {
        let gate = TEBDGates.heisenbergXXZ(angle: 0.15, delta: 0.8)

        #expect(gate[0][1].magnitudeSquared < 1e-20, "Gate[0][1] should be zero")
        #expect(gate[0][2].magnitudeSquared < 1e-20, "Gate[0][2] should be zero")
        #expect(gate[1][0].magnitudeSquared < 1e-20, "Gate[1][0] should be zero")
        #expect(gate[1][3].magnitudeSquared < 1e-20, "Gate[1][3] should be zero")
        #expect(gate[2][0].magnitudeSquared < 1e-20, "Gate[2][0] should be zero")
        #expect(gate[2][3].magnitudeSquared < 1e-20, "Gate[2][3] should be zero")
        #expect(gate[3][1].magnitudeSquared < 1e-20, "Gate[3][1] should be zero")
        #expect(gate[3][2].magnitudeSquared < 1e-20, "Gate[3][2] should be zero")
    }

    @Test("Heisenberg gate middle block has coupling")
    func heisenbergGateMiddleBlockCoupling() {
        let gate = TEBDGates.heisenbergXXZ(angle: 0.1, delta: 1.0)

        #expect(gate[1][2].magnitudeSquared > 1e-10, "Gate[1][2] should be non-zero for non-trivial coupling")
        #expect(gate[2][1].magnitudeSquared > 1e-10, "Gate[2][1] should be non-zero for non-trivial coupling")
    }
}

/// Test suite for evolution gate unitarity verification.
/// Validates that U*U^dagger = I for all TEBD gates,
/// ensuring physical quantum evolution.
@Suite("Evolution Gates Unitarity")
struct EvolutionGatesUnitarityTests {
    @Test("ZZ gate is unitary")
    func zzGateIsUnitary() {
        let gate = TEBDGates.zzEvolution(angle: 0.47)
        let product = multiplyMatrixByConjugateTranspose(gate)

        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                if i == j {
                    #expect(abs(product[i][j].real - 1.0) < 1e-10, "Diagonal element [\(i)][\(j)] of U*U^dag should be 1")
                    #expect(abs(product[i][j].imaginary) < 1e-10, "Diagonal element [\(i)][\(j)] imaginary should be 0")
                } else {
                    #expect(product[i][j].magnitudeSquared < 1e-18, "Off-diagonal element [\(i)][\(j)] of U*U^dag should be 0")
                }
            }
        }
    }

    @Test("XX gate is unitary")
    func xxGateIsUnitary() {
        let gate = TEBDGates.xxEvolution(angle: 0.83)
        let product = multiplyMatrixByConjugateTranspose(gate)

        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                if i == j {
                    #expect(abs(product[i][j].real - 1.0) < 1e-10, "Diagonal element [\(i)][\(j)] of U*U^dag should be 1")
                } else {
                    #expect(product[i][j].magnitudeSquared < 1e-18, "Off-diagonal element [\(i)][\(j)] of U*U^dag should be 0")
                }
            }
        }
    }

    @Test("YY gate is unitary")
    func yyGateIsUnitary() {
        let gate = TEBDGates.yyEvolution(angle: 0.62)
        let product = multiplyMatrixByConjugateTranspose(gate)

        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                if i == j {
                    #expect(abs(product[i][j].real - 1.0) < 1e-10, "Diagonal element [\(i)][\(j)] of U*U^dag should be 1")
                } else {
                    #expect(product[i][j].magnitudeSquared < 1e-18, "Off-diagonal element [\(i)][\(j)] of U*U^dag should be 0")
                }
            }
        }
    }

    @Test("X gate is unitary")
    func xGateIsUnitary() {
        let gate = TEBDGates.xEvolution(angle: 0.91)
        let product = multiplyMatrixByConjugateTranspose2x2(gate)

        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                if i == j {
                    #expect(abs(product[i][j].real - 1.0) < 1e-10, "Diagonal element [\(i)][\(j)] of U*U^dag should be 1")
                } else {
                    #expect(product[i][j].magnitudeSquared < 1e-18, "Off-diagonal element [\(i)][\(j)] of U*U^dag should be 0")
                }
            }
        }
    }

    @Test("Z gate is unitary")
    func zGateIsUnitary() {
        let gate = TEBDGates.zEvolution(angle: 1.23)
        let product = multiplyMatrixByConjugateTranspose2x2(gate)

        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                if i == j {
                    #expect(abs(product[i][j].real - 1.0) < 1e-10, "Diagonal element [\(i)][\(j)] of U*U^dag should be 1")
                } else {
                    #expect(product[i][j].magnitudeSquared < 1e-18, "Off-diagonal element [\(i)][\(j)] of U*U^dag should be 0")
                }
            }
        }
    }

    @Test("Heisenberg gate is unitary")
    func heisenbergGateIsUnitary() {
        let gate = TEBDGates.heisenbergXXZ(angle: 0.31, delta: 1.2)
        let product = multiplyMatrixByConjugateTranspose(gate)

        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                if i == j {
                    #expect(abs(product[i][j].real - 1.0) < 1e-10, "Diagonal element [\(i)][\(j)] of U*U^dag should be 1")
                } else {
                    #expect(product[i][j].magnitudeSquared < 1e-18, "Off-diagonal element [\(i)][\(j)] of U*U^dag should be 0")
                }
            }
        }
    }

    private func multiplyMatrixByConjugateTranspose(_ matrix: [[Complex<Double>]]) -> [[Complex<Double>]] {
        let n = matrix.count
        var result = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: n), count: n)
        for i in 0 ..< n {
            for j in 0 ..< n {
                var sum: Complex<Double> = .zero
                for k in 0 ..< n {
                    sum = sum + matrix[i][k] * matrix[j][k].conjugate
                }
                result[i][j] = sum
            }
        }
        return result
    }

    private func multiplyMatrixByConjugateTranspose2x2(_ matrix: [[Complex<Double>]]) -> [[Complex<Double>]] {
        var result = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: 2), count: 2)
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                var sum: Complex<Double> = .zero
                for k in 0 ..< 2 {
                    sum = sum + matrix[i][k] * matrix[j][k].conjugate
                }
                result[i][j] = sum
            }
        }
        return result
    }
}

/// Test suite for TEBDResult structure properties.
/// Validates that result contains all expected fields
/// and statistics are correctly propagated.
@Suite("TEBDResult Properties")
struct TEBDResultPropertiesTests {
    @Test("TEBDResult contains final state")
    func tebdResultContainsFinalState() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveIsing(mps: mps, J: 0.5, h: 0.3, time: 0.1, steps: 2)

        #expect(result.finalState.qubits == 4, "Final state should have 4 qubits")
        #expect(result.finalState.maxBondDimension == 8, "Final state should preserve max bond dimension")
    }

    @Test("TEBDResult contains evolution time")
    func tebdResultContainsTime() async {
        let mps = MatrixProductState(qubits: 3, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.5, time: 0.75, steps: 3)

        #expect(abs(result.time - 0.75) < 1e-10, "Result time should match requested evolution time")
    }

    @Test("TEBDResult contains step count")
    func tebdResultContainsSteps() async {
        let mps = MatrixProductState(qubits: 3, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.5, time: 0.5, steps: 7)

        #expect(result.steps == 7, "Result steps should match requested step count")
    }

    @Test("TEBDResult contains truncation statistics")
    func tebdResultContainsTruncationStatistics() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.5, time: 0.2, steps: 5)

        #expect(result.truncationStatistics.cumulativeError >= 0.0, "Cumulative error should be non-negative")
        #expect(result.truncationStatistics.maxSingleError >= 0.0, "Max single error should be non-negative")
        #expect(result.truncationStatistics.truncationCount >= 0, "Truncation count should be non-negative")
    }

    @Test("TEBDResult contains max bond dimension reached")
    func tebdResultContainsMaxBondDimensionReached() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 16)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.5, time: 0.2, steps: 3)

        #expect(result.maxBondDimensionReached >= 1, "Max bond dimension reached should be at least 1")
        #expect(result.maxBondDimensionReached <= 16, "Max bond dimension reached should not exceed limit")
    }

    @Test("TEBDResult contains total gates applied")
    func tebdResultContainsTotalGatesApplied() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.5, time: 0.1, steps: 2)

        #expect(result.totalGatesApplied > 0, "Total gates applied should be positive for non-trivial evolution")
    }
}

/// Test suite for MPS Ising model time evolution.
/// Validates normalization preservation and state evolution
/// under the transverse-field Ising Hamiltonian.
@Suite("MPS Ising Evolution")
struct MPSIsingEvolutionTests {
    @Test("Ising evolution preserves normalization (small system)")
    func isingEvolutionPreservesNormalizationSmall() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 16)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.5, time: 0.5, steps: 10)

        let normSquared = result.finalState.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-6, "Norm squared should be 1.0 after Ising evolution, got \(normSquared)")
    }

    @Test("Ising evolution preserves normalization (larger system)")
    func isingEvolutionPreservesNormalizationLarger() async {
        let mps = MatrixProductState(qubits: 6, maxBondDimension: 16)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveIsing(mps: mps, J: 0.8, h: 0.3, time: 0.3, steps: 5)

        let normSquared = result.finalState.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-5, "Norm squared should be 1.0 after Ising evolution, got \(normSquared)")
    }

    @Test("Ising evolution with zero field preserves ground state")
    func isingEvolutionZeroFieldPreservesGroundState() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.0, time: 0.5, steps: 5)

        let amplitude00 = result.finalState.amplitude(of: 0)
        #expect(amplitude00.magnitudeSquared > 0.99, "Ground state |0000> should remain dominant with zero transverse field")
    }

    @Test("Ising evolution with first-order Trotter")
    func isingEvolutionFirstOrderTrotter() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.5, time: 0.2, steps: 5, order: .first)

        let normSquared = result.finalState.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-5, "Norm should be preserved with first-order Trotter")
    }

    @Test("Ising evolution with fourth-order Trotter")
    func isingEvolutionFourthOrderTrotter() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.5, time: 0.2, steps: 3, order: .fourth)

        let normSquared = result.finalState.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-5, "Norm should be preserved with fourth-order Trotter")
    }
}

/// Test suite for MPS Heisenberg model time evolution.
/// Validates normalization preservation and state evolution
/// under the Heisenberg XXZ Hamiltonian.
@Suite("MPS Heisenberg Evolution")
struct MPSHeisenbergEvolutionTests {
    @Test("Heisenberg evolution preserves normalization (isotropic)")
    func heisenbergEvolutionPreservesNormalizationIsotropic() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 16)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveHeisenberg(mps: mps, J: 1.0, delta: 1.0, time: 0.3, steps: 8)

        let normSquared = result.finalState.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-5, "Norm squared should be 1.0 after Heisenberg evolution, got \(normSquared)")
    }

    @Test("Heisenberg evolution preserves normalization (anisotropic)")
    func heisenbergEvolutionPreservesNormalizationAnisotropic() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 16)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveHeisenberg(mps: mps, J: 1.0, delta: 0.5, time: 0.3, steps: 8)

        let normSquared = result.finalState.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-5, "Norm squared should be 1.0 after anisotropic Heisenberg evolution, got \(normSquared)")
    }

    @Test("Heisenberg evolution preserves ground state for ferromagnetic coupling")
    func heisenbergEvolutionPreservesGroundState() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveHeisenberg(mps: mps, J: -1.0, delta: 1.0, time: 0.2, steps: 5)

        let amplitude00 = result.finalState.amplitude(of: 0)
        #expect(amplitude00.magnitudeSquared > 0.9, "Ferromagnetic ground state |0000> should remain dominant")
    }

    @Test("Heisenberg evolution with second-order Trotter")
    func heisenbergEvolutionSecondOrderTrotter() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveHeisenberg(mps: mps, J: 1.0, delta: 1.0, time: 0.2, steps: 5, order: .second)

        let normSquared = result.finalState.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-5, "Norm should be preserved with second-order Trotter")
    }

    @Test("Heisenberg evolution with sixth-order Trotter")
    func heisenbergEvolutionSixthOrderTrotter() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveHeisenberg(mps: mps, J: 1.0, delta: 1.0, time: 0.1, steps: 2, order: .sixth)

        let normSquared = result.finalState.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-4, "Norm should be preserved with sixth-order Trotter")
    }
}

/// Test suite for truncation statistics tracking during evolution.
/// Validates that truncation errors are properly accumulated
/// and reported in the result.
@Suite("Truncation Statistics Tracking")
struct TruncationStatisticsTrackingTests {
    @Test("Truncation count increases during evolution")
    func truncationCountIncreases() async {
        let mps = MatrixProductState(qubits: 6, maxBondDimension: 4)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.5, time: 0.5, steps: 10)

        #expect(result.truncationStatistics.truncationCount > 0, "Truncation count should increase during evolution with limited bond dimension")
    }

    @Test("Cumulative error is non-negative")
    func cumulativeErrorNonNegative() async {
        let mps = MatrixProductState(qubits: 5, maxBondDimension: 4)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveHeisenberg(mps: mps, J: 1.0, delta: 1.0, time: 0.3, steps: 5)

        #expect(result.truncationStatistics.cumulativeError >= 0.0, "Cumulative error should be non-negative")
    }

    @Test("Max single error does not exceed cumulative error")
    func maxSingleErrorBoundedByCumulative() async {
        let mps = MatrixProductState(qubits: 5, maxBondDimension: 4)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.5, time: 0.4, steps: 8)

        #expect(
            result.truncationStatistics.maxSingleError <= result.truncationStatistics.cumulativeError + 1e-15,
            "Max single error should not exceed cumulative error",
        )
    }

    @Test("Truncation statistics in final state match result")
    func truncationStatisticsMatchFinalState() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 4)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.5, time: 0.2, steps: 5)

        let finalStats = result.finalState.truncationStatistics
        let resultStats = result.truncationStatistics

        #expect(finalStats.truncationCount == resultStats.truncationCount, "Truncation count should match between final state and result")
        #expect(abs(finalStats.cumulativeError - resultStats.cumulativeError) < 1e-15, "Cumulative error should match between final state and result")
    }

    @Test("Large bond dimension yields smaller truncation error")
    func largeBondDimensionSmallerError() async {
        let mpsSmall = MatrixProductState(qubits: 5, maxBondDimension: 4)
        let mpsLarge = MatrixProductState(qubits: 5, maxBondDimension: 16)
        let evolution = MPSTimeEvolution()

        let resultSmall = await evolution.evolveIsing(mps: mpsSmall, J: 1.0, h: 0.5, time: 0.3, steps: 5)
        let resultLarge = await evolution.evolveIsing(mps: mpsLarge, J: 1.0, h: 0.5, time: 0.3, steps: 5)

        #expect(
            resultLarge.truncationStatistics.cumulativeError <= resultSmall.truncationStatistics.cumulativeError + 1e-10,
            "Larger bond dimension should yield smaller or equal truncation error",
        )
    }
}

/// Test suite for MatrixProductState extension methods for time evolution.
/// Validates the convenience methods evolvingIsing and evolvingHeisenberg.
@Suite("MatrixProductState Extension Methods")
struct MatrixProductStateExtensionMethodsTests {
    @Test("evolvingIsing returns valid result")
    func evolvingIsingReturnsValidResult() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let result = await mps.evolvingIsing(J: 1.0, h: 0.5, time: 0.2, steps: 5)

        #expect(result.finalState.qubits == 4, "Final state should have same number of qubits")
        #expect(result.steps == 5, "Result should have requested number of steps")
        #expect(abs(result.time - 0.2) < 1e-10, "Result should have requested evolution time")
    }

    @Test("evolvingIsing preserves normalization")
    func evolvingIsingPreservesNormalization() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let result = await mps.evolvingIsing(J: 1.0, h: 0.5, time: 0.3, steps: 8)

        let normSquared = result.finalState.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-5, "evolvingIsing should preserve normalization")
    }

    @Test("evolvingHeisenberg returns valid result")
    func evolvingHeisenbergReturnsValidResult() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let result = await mps.evolvingHeisenberg(J: 1.0, delta: 1.0, time: 0.2, steps: 5)

        #expect(result.finalState.qubits == 4, "Final state should have same number of qubits")
        #expect(result.steps == 5, "Result should have requested number of steps")
        #expect(abs(result.time - 0.2) < 1e-10, "Result should have requested evolution time")
    }

    @Test("evolvingHeisenberg preserves normalization")
    func evolvingHeisenbergPreservesNormalization() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let result = await mps.evolvingHeisenberg(J: 1.0, delta: 1.0, time: 0.3, steps: 8)

        let normSquared = result.finalState.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-5, "evolvingHeisenberg should preserve normalization")
    }

    @Test("Extension methods do not modify original MPS")
    func extensionMethodsDoNotModifyOriginal() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let originalAmplitude = mps.amplitude(of: 0)

        _ = await mps.evolvingIsing(J: 1.0, h: 0.5, time: 0.2, steps: 5)

        let afterAmplitude = mps.amplitude(of: 0)
        #expect(abs(originalAmplitude.real - afterAmplitude.real) < 1e-10, "Original MPS should not be modified by evolvingIsing")
        #expect(abs(originalAmplitude.imaginary - afterAmplitude.imaginary) < 1e-10, "Original MPS imaginary should not be modified")
    }
}

/// Test suite for custom gate evolution using evolveWithGate.
/// Validates the ability to apply arbitrary two-site and single-site gates.
@Suite("Custom Gate Evolution")
struct CustomGateEvolutionTests {
    @Test("evolveWithGate with identity-like gate preserves state")
    func evolveWithGateIdentityPreservesState() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let identityGate: [[Complex<Double>]] = [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .one, .zero],
            [.zero, .zero, .zero, .one],
        ]

        let result = await evolution.evolveWithGate(
            mps: mps,
            twoSiteGate: identityGate,
            singleSiteGates: [],
            time: 0.5,
            steps: 3,
        )

        let amplitude00 = result.finalState.amplitude(of: 0)
        #expect(amplitude00.magnitudeSquared > 0.99, "Identity gate should preserve ground state")
    }

    @Test("evolveWithGate applies total gates correctly")
    func evolveWithGateTotalGatesCount() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let gate = TEBDGates.zzEvolution(angle: 0.1)

        let result = await evolution.evolveWithGate(
            mps: mps,
            twoSiteGate: gate,
            singleSiteGates: [],
            time: 0.5,
            steps: 2,
        )

        #expect(result.totalGatesApplied > 0, "Total gates applied should be positive")
    }

    @Test("evolveWithGate with single-site gates")
    func evolveWithGateSingleSiteGates() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let twoSiteGate = TEBDGates.zzEvolution(angle: 0.1)
        let singleSiteGate = TEBDGates.xEvolution(angle: 0.05)
        let singleSiteGates = Array(repeating: singleSiteGate, count: 4)

        let result = await evolution.evolveWithGate(
            mps: mps,
            twoSiteGate: twoSiteGate,
            singleSiteGates: singleSiteGates,
            time: 0.2,
            steps: 3,
        )

        let normSquared = result.finalState.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-5, "Evolution with single-site gates should preserve normalization")
    }
}

/// Test suite for bond dimension behavior during evolution.
/// Validates that bond dimensions grow and are properly bounded.
@Suite("Bond Dimension Behavior")
struct BondDimensionBehaviorTests {
    @Test("Bond dimension can grow during entangling evolution")
    func bondDimensionCanGrow() async {
        let mps = MatrixProductState(qubits: 6, maxBondDimension: 16)
        let initialMaxBond = mps.currentMaxBondDimension
        let evolution = MPSTimeEvolution()

        let result = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.5, time: 0.5, steps: 10)

        #expect(result.maxBondDimensionReached >= initialMaxBond, "Bond dimension should not decrease during evolution")
    }

    @Test("Bond dimension respects maximum limit")
    func bondDimensionRespectsLimit() async {
        let maxBond = 8
        let mps = MatrixProductState(qubits: 6, maxBondDimension: maxBond)
        let evolution = MPSTimeEvolution()

        let result = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.5, time: 1.0, steps: 20)

        #expect(result.maxBondDimensionReached <= maxBond, "Bond dimension should not exceed maximum limit")
        #expect(result.finalState.currentMaxBondDimension <= maxBond, "Final state bond dimension should not exceed maximum limit")
    }

    @Test("Max bond dimension reached is tracked correctly")
    func maxBondDimensionTracked() async {
        let mps = MatrixProductState(qubits: 5, maxBondDimension: 16)
        let evolution = MPSTimeEvolution()

        let result = await evolution.evolveHeisenberg(mps: mps, J: 1.0, delta: 1.0, time: 0.3, steps: 5)

        #expect(
            result.maxBondDimensionReached >= result.finalState.currentMaxBondDimension,
            "Max bond dimension reached should be at least as large as final bond dimension",
        )
    }
}

/// Test suite for sixth-order Trotter with single-site gates.
/// Validates that applySixthOrderStep correctly applies single-site gates
/// when transverse field is present in Ising evolution.
@Suite("Sixth Order Trotter With Single Site Gates")
struct SixthOrderTrotterWithSingleSiteGatesTests {
    @Test("Ising evolution with sixth-order Trotter applies single-site gates")
    func isingSixthOrderWithTransverseField() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.5, time: 0.1, steps: 2, order: .sixth)

        let normSquared = result.finalState.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-4, "Norm should be preserved with sixth-order Trotter and transverse field")
    }

    @Test("Sixth-order Ising evolution changes state with transverse field")
    func isingSixthOrderStateChanges() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let result = await evolution.evolveIsing(mps: mps, J: 1.0, h: 1.0, time: 0.2, steps: 3, order: .sixth)

        let amplitude00 = result.finalState.amplitude(of: 0)
        #expect(amplitude00.magnitudeSquared < 0.99, "State should change with strong transverse field in sixth-order Trotter")
    }

    @Test("Sixth-order Ising applies more gates than lower orders")
    func isingSixthOrderGateCount() async {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()

        let resultSecond = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.5, time: 0.1, steps: 1, order: .second)
        let resultSixth = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.5, time: 0.1, steps: 1, order: .sixth)

        #expect(resultSixth.totalGatesApplied > resultSecond.totalGatesApplied, "Sixth-order should apply more gates than second-order")
    }
}

/// Test suite for applyTEBDStep method direct invocation.
/// Validates the public TEBD step method with separate even and odd gates
/// and optional single-site gates.
@Suite("TEBD Step Direct Invocation")
struct TEBDStepDirectInvocationTests {
    @Test("applyTEBDStep applies even and odd gates")
    func applyTEBDStepAppliesGates() async {
        var mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let evenGate = TEBDGates.zzEvolution(angle: 0.1)
        let oddGate = TEBDGates.zzEvolution(angle: 0.05)

        await evolution.applyTEBDStep(mps: &mps, evenGate: evenGate, oddGate: oddGate, singleSiteGates: nil)

        let normSquared = mps.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-10, "Norm should be preserved after TEBD step")
    }

    @Test("applyTEBDStep with single-site gates")
    func applyTEBDStepWithSingleSiteGates() async {
        var mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let evenGate = TEBDGates.zzEvolution(angle: 0.1)
        let oddGate = TEBDGates.zzEvolution(angle: 0.1)
        let singleGate = TEBDGates.xEvolution(angle: 0.2)
        let singleSiteGates = Array(repeating: singleGate, count: 4)

        await evolution.applyTEBDStep(mps: &mps, evenGate: evenGate, oddGate: oddGate, singleSiteGates: singleSiteGates)

        let normSquared = mps.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-10, "Norm should be preserved after TEBD step with single-site gates")
    }

    @Test("applyTEBDStep modifies state")
    func applyTEBDStepModifiesState() async {
        var mps = MatrixProductState(qubits: 4, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let evenGate = TEBDGates.xxEvolution(angle: 0.3)
        let oddGate = TEBDGates.xxEvolution(angle: 0.3)
        let singleGate = TEBDGates.xEvolution(angle: 0.5)
        let singleSiteGates = Array(repeating: singleGate, count: 4)

        let amplitudeBefore = mps.amplitude(of: 0)
        await evolution.applyTEBDStep(mps: &mps, evenGate: evenGate, oddGate: oddGate, singleSiteGates: singleSiteGates)
        let amplitudeAfter = mps.amplitude(of: 0)

        let changed = abs(amplitudeBefore.real - amplitudeAfter.real) > 1e-10 ||
            abs(amplitudeBefore.imaginary - amplitudeAfter.imaginary) > 1e-10
        #expect(changed, "TEBD step should modify state amplitude")
    }

    @Test("applyTEBDStep works with different even and odd gates")
    func applyTEBDStepDifferentGates() async {
        var mps = MatrixProductState(qubits: 6, maxBondDimension: 8)
        let evolution = MPSTimeEvolution()
        let evenGate = TEBDGates.zzEvolution(angle: 0.2)
        let oddGate = TEBDGates.xxEvolution(angle: 0.15)

        await evolution.applyTEBDStep(mps: &mps, evenGate: evenGate, oddGate: oddGate, singleSiteGates: nil)

        let normSquared = mps.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-10, "Norm should be preserved with different even and odd gates")
    }

    @Test("applyTEBDStep on small system with 2 qubits")
    func applyTEBDStepSmallSystem() async {
        var mps = MatrixProductState(qubits: 2, maxBondDimension: 4)
        let evolution = MPSTimeEvolution()
        let evenGate = TEBDGates.heisenbergXXZ(angle: 0.1, delta: 1.0)
        let oddGate = TEBDGates.heisenbergXXZ(angle: 0.1, delta: 1.0)

        await evolution.applyTEBDStep(mps: &mps, evenGate: evenGate, oddGate: oddGate, singleSiteGates: nil)

        let normSquared = mps.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-10, "Norm should be preserved on 2-qubit system")
    }
}

/// Test suite for GPU acceleration path in two-site gate application.
/// Validates the conditional GPU/CPU path selection based on bond dimension
/// and GPU availability.
@Suite("GPU Acceleration Path")
struct GPUAccelerationPathTests {
    @Test("Large bond dimension triggers GPU threshold check")
    func largeBondDimensionTriggersGPUCheck() async {
        let mps = MatrixProductState(qubits: 8, maxBondDimension: 64)
        let evolution = MPSTimeEvolution()

        let result = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.5, time: 0.3, steps: 5)

        let normSquared = result.finalState.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-4, "Norm should be preserved with large bond dimension")
    }

    @Test("Evolution with bond dimension at GPU threshold")
    func evolutionAtGPUThreshold() async {
        let threshold = MPSMetalAcceleration.gpuThreshold
        let mps = MatrixProductState(qubits: 10, maxBondDimension: threshold)
        let evolution = MPSTimeEvolution()

        let result = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.5, time: 0.2, steps: 3)

        let normSquared = result.finalState.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-4, "Norm should be preserved at GPU threshold bond dimension")
        #expect(result.maxBondDimensionReached <= threshold, "Bond dimension should not exceed threshold")
    }

    @Test("Evolution with bond dimension above GPU threshold")
    func evolutionAboveGPUThreshold() async {
        let threshold = MPSMetalAcceleration.gpuThreshold
        let mps = MatrixProductState(qubits: 10, maxBondDimension: threshold + 16)
        let evolution = MPSTimeEvolution()

        let result = await evolution.evolveIsing(mps: mps, J: 1.0, h: 0.5, time: 0.2, steps: 3)

        let normSquared = result.finalState.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-4, "Norm should be preserved above GPU threshold bond dimension")
    }

    @Test("applyTwoSiteGate with large bond dimension")
    func applyTwoSiteGateLargeBondDimension() async {
        var mps = MatrixProductState(qubits: 10, maxBondDimension: 64)
        let evolution = MPSTimeEvolution()
        let gate = TEBDGates.heisenbergXXZ(angle: 0.1, delta: 1.0)

        for _ in 0 ..< 5 {
            await evolution.applyTwoSiteGate(mps: &mps, gate: gate, site: 4)
        }

        let normSquared = mps.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-6, "Norm should be preserved after multiple two-site gate applications with large bond dimension")
    }

    @Test("Heisenberg evolution with large system exercises GPU path")
    func heisenbergLargeSystemGPUPath() async {
        let mps = MatrixProductState(qubits: 12, maxBondDimension: 48)
        let evolution = MPSTimeEvolution()

        let result = await evolution.evolveHeisenberg(mps: mps, J: 1.0, delta: 1.0, time: 0.15, steps: 3)

        let normSquared = result.finalState.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-3, "Norm should be preserved in large system Heisenberg evolution")
        #expect(result.totalGatesApplied > 0, "Gates should be applied during evolution")
    }
}
