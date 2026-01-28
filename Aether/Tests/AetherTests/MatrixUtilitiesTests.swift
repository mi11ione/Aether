// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Validates MatrixUtilities.matrixPower for correctness on small matrices.
/// Covers identity preservation, base cases (power 0 and 1), and known
/// algebraic identities including Hadamard and Pauli self-inverse properties.
@Suite("Matrix Power")
struct MatrixPowerTests {
    let tolerance: Double = 1e-10

    let identity2x2: [[Complex<Double>]] = MatrixUtilities.identityMatrix(dimension: 2)

    let pauliX: [[Complex<Double>]] = [
        [Complex(0.0, 0.0), Complex(1.0, 0.0)],
        [Complex(1.0, 0.0), Complex(0.0, 0.0)],
    ]

    let hadamard: [[Complex<Double>]] = {
        let s = 1.0 / sqrt(2.0)
        return [
            [Complex(s, 0.0), Complex(s, 0.0)],
            [Complex(s, 0.0), Complex(-s, 0.0)],
        ]
    }()

    let identity4x4: [[Complex<Double>]] = MatrixUtilities.identityMatrix(dimension: 4)

    @Test("Identity matrix to any power returns identity")
    func identityToAnyPower() {
        for exp in [0, 1, 2, 5, 10] {
            let result = MatrixUtilities.matrixPower(identity2x2, exponent: exp)
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    #expect(
                        abs(result[i][j].real - identity2x2[i][j].real) < tolerance,
                        "I^{\(exp)}[\(i)][\(j)] real part should match identity, got \(result[i][j].real)",
                    )
                    #expect(
                        abs(result[i][j].imaginary - identity2x2[i][j].imaginary) < tolerance,
                        "I^{\(exp)}[\(i)][\(j)] imaginary part should be zero, got \(result[i][j].imaginary)",
                    )
                }
            }
        }
    }

    @Test("4x4 identity matrix to power 7 returns 4x4 identity")
    func identity4x4ToAnyPower() {
        let result = MatrixUtilities.matrixPower(identity4x4, exponent: 7)
        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                #expect(
                    abs(result[i][j].real - identity4x4[i][j].real) < tolerance,
                    "I4^7[\(i)][\(j)] real part should match identity, got \(result[i][j].real)",
                )
                #expect(
                    abs(result[i][j].imaginary) < tolerance,
                    "I4^7[\(i)][\(j)] imaginary part should be zero, got \(result[i][j].imaginary)",
                )
            }
        }
    }

    @Test("Matrix to power 0 returns identity")
    func matrixToPowerZero() {
        let result = MatrixUtilities.matrixPower(pauliX, exponent: 0)
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                #expect(
                    abs(result[i][j].real - identity2x2[i][j].real) < tolerance,
                    "X^0[\(i)][\(j)] real part should match identity, got \(result[i][j].real)",
                )
                #expect(
                    abs(result[i][j].imaginary) < tolerance,
                    "X^0[\(i)][\(j)] imaginary part should be zero, got \(result[i][j].imaginary)",
                )
            }
        }
    }

    @Test("Matrix to power 1 returns same matrix")
    func matrixToPowerOne() {
        let result = MatrixUtilities.matrixPower(pauliX, exponent: 1)
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                #expect(
                    abs(result[i][j].real - pauliX[i][j].real) < tolerance,
                    "X^1[\(i)][\(j)] real part should match original, got \(result[i][j].real) expected \(pauliX[i][j].real)",
                )
                #expect(
                    abs(result[i][j].imaginary - pauliX[i][j].imaginary) < tolerance,
                    "X^1[\(i)][\(j)] imaginary part should match original, got \(result[i][j].imaginary)",
                )
            }
        }
    }

    @Test("Hadamard squared equals identity")
    func hadamardSquaredIsIdentity() {
        let result = MatrixUtilities.matrixPower(hadamard, exponent: 2)
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                #expect(
                    abs(result[i][j].real - identity2x2[i][j].real) < tolerance,
                    "H^2[\(i)][\(j)] real part should match identity, got \(result[i][j].real) expected \(identity2x2[i][j].real)",
                )
                #expect(
                    abs(result[i][j].imaginary) < tolerance,
                    "H^2[\(i)][\(j)] imaginary part should be zero, got \(result[i][j].imaginary)",
                )
            }
        }
    }

    @Test("Pauli X squared equals identity")
    func pauliXSquaredIsIdentity() {
        let result = MatrixUtilities.matrixPower(pauliX, exponent: 2)
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                #expect(
                    abs(result[i][j].real - identity2x2[i][j].real) < tolerance,
                    "X^2[\(i)][\(j)] real part should match identity, got \(result[i][j].real) expected \(identity2x2[i][j].real)",
                )
                #expect(
                    abs(result[i][j].imaginary) < tolerance,
                    "X^2[\(i)][\(j)] imaginary part should be zero, got \(result[i][j].imaginary)",
                )
            }
        }
    }

    @Test("Rotation matrix to known power via element comparison")
    func rotationMatrixToKnownPower() {
        let angle = Double.pi / 4.0
        let cosA = cos(angle)
        let sinA = sin(angle)
        let rotation: [[Complex<Double>]] = [
            [Complex(cosA, 0.0), Complex(-sinA, 0.0)],
            [Complex(sinA, 0.0), Complex(cosA, 0.0)],
        ]

        let result = MatrixUtilities.matrixPower(rotation, exponent: 4)

        let expectedAngle = Double.pi
        let expectedCos = cos(expectedAngle)
        let expectedSin = sin(expectedAngle)
        let expected: [[Complex<Double>]] = [
            [Complex(expectedCos, 0.0), Complex(-expectedSin, 0.0)],
            [Complex(expectedSin, 0.0), Complex(expectedCos, 0.0)],
        ]

        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                #expect(
                    abs(result[i][j].real - expected[i][j].real) < tolerance,
                    "R(pi/4)^4[\(i)][\(j)] real part should be \(expected[i][j].real), got \(result[i][j].real)",
                )
                #expect(
                    abs(result[i][j].imaginary - expected[i][j].imaginary) < tolerance,
                    "R(pi/4)^4[\(i)][\(j)] imaginary part should be \(expected[i][j].imaginary), got \(result[i][j].imaginary)",
                )
            }
        }
    }
}
