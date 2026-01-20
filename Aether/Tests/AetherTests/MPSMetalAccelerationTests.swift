// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Test suite for GPU-accelerated tensor contraction using Metal Performance Shaders.
/// Validates matrix multiplication, tensor contraction, chain contraction, CPU fallback,
/// and numerical precision for MPS operations.
@Suite("MPSMetalAcceleration")
struct MPSMetalAccelerationTests {
    @Test("Actor initializes successfully")
    func initializeAccelerator() async {
        let accelerator = MPSMetalAcceleration()
        _ = accelerator.isAvailable
    }

    @Test("isAvailable property is accessible")
    func isAvailableAccessible() async {
        let accelerator = MPSMetalAcceleration()
        let available = accelerator.isAvailable
        #expect(available == true || available == false, "isAvailable should return a boolean value")
    }

    @Test("gpuThreshold constant is 32")
    func gpuThresholdValue() {
        #expect(MPSMetalAcceleration.gpuThreshold == 32, "gpuThreshold should be 32 for optimal GPU/CPU switching")
    }

    @Test("matrixMultiply computes correct result for identity matrix")
    func matrixMultiplyIdentity() async {
        let accelerator = MPSMetalAcceleration()

        let identity: [[Complex<Double>]] = [
            [.one, .zero],
            [.zero, .one],
        ]

        let matrix: [[Complex<Double>]] = [
            [Complex(1.0, 2.0), Complex(3.0, 4.0)],
            [Complex(5.0, 6.0), Complex(7.0, 8.0)],
        ]

        let result = await accelerator.matrixMultiply(identity, matrix)

        #expect(result.count == 2, "Result should have 2 rows")
        #expect(result[0].count == 2, "Result should have 2 columns")

        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                let diff = (result[i][j] - matrix[i][j]).magnitude
                #expect(diff < 1e-10, "Identity * A should equal A at [\(i)][\(j)]")
            }
        }
    }

    @Test("matrixMultiply computes correct complex multiplication")
    func matrixMultiplyComplex() async {
        let accelerator = MPSMetalAcceleration()

        let a: [[Complex<Double>]] = [
            [Complex(1.0, 1.0), Complex(2.0, 0.0)],
            [Complex(0.0, 1.0), Complex(1.0, -1.0)],
        ]

        let b: [[Complex<Double>]] = [
            [Complex(1.0, 0.0), Complex(0.0, 1.0)],
            [Complex(1.0, 1.0), Complex(2.0, 0.0)],
        ]

        let result = await accelerator.matrixMultiply(a, b)

        let expected00 = Complex(1.0, 1.0) * Complex(1.0, 0.0) + Complex(2.0, 0.0) * Complex(1.0, 1.0)
        let expected01 = Complex(1.0, 1.0) * Complex(0.0, 1.0) + Complex(2.0, 0.0) * Complex(2.0, 0.0)
        let expected10 = Complex(0.0, 1.0) * Complex(1.0, 0.0) + Complex(1.0, -1.0) * Complex(1.0, 1.0)
        let expected11 = Complex(0.0, 1.0) * Complex(0.0, 1.0) + Complex(1.0, -1.0) * Complex(2.0, 0.0)

        #expect((result[0][0] - expected00).magnitude < 1e-10, "Result[0][0] should match expected complex product")
        #expect((result[0][1] - expected01).magnitude < 1e-10, "Result[0][1] should match expected complex product")
        #expect((result[1][0] - expected10).magnitude < 1e-10, "Result[1][0] should match expected complex product")
        #expect((result[1][1] - expected11).magnitude < 1e-10, "Result[1][1] should match expected complex product")
    }

    @Test("matrixMultiply uses CPU path for small matrices")
    func matrixMultiplySmallCPU() async {
        let accelerator = MPSMetalAcceleration()

        let size = MPSMetalAcceleration.gpuThreshold - 1
        var a = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: size), count: size)
        var b = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: size), count: size)

        for i in 0 ..< size {
            a[i][i] = .one
            b[i][i] = Complex(Double(i + 1), 0.0)
        }

        let result = await accelerator.matrixMultiply(a, b)

        #expect(result.count == size, "Result should have \(size) rows for small matrix CPU path")

        for i in 0 ..< size {
            let expected = Complex(Double(i + 1), 0.0)
            let diff = (result[i][i] - expected).magnitude
            #expect(diff < 1e-10, "Diagonal element [\(i)][\(i)] should match for CPU path")
        }
    }

    @Test("matrixMultiply uses GPU path for large matrices when available")
    func matrixMultiplyLargeGPU() async {
        let accelerator = MPSMetalAcceleration()

        let size = MPSMetalAcceleration.gpuThreshold
        var a = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: size), count: size)
        var b = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: size), count: size)

        for i in 0 ..< size {
            a[i][i] = .one
            b[i][i] = Complex(Double(i + 1), 0.0)
        }

        let result = await accelerator.matrixMultiply(a, b)

        #expect(result.count == size, "Result should have \(size) rows for large matrix path")

        let tolerance: Double = accelerator.isAvailable ? 1e-6 : 1e-10

        for i in 0 ..< size {
            let expected = Complex(Double(i + 1), 0.0)
            let diff = (result[i][i] - expected).magnitude
            #expect(diff < tolerance, "Diagonal element [\(i)][\(i)] should match within tolerance")
        }
    }

    @Test("matrixMultiply handles non-square matrices")
    func matrixMultiplyNonSquare() async {
        let accelerator = MPSMetalAcceleration()

        let a: [[Complex<Double>]] = [
            [Complex(1.0, 0.0), Complex(2.0, 0.0), Complex(3.0, 0.0)],
            [Complex(4.0, 0.0), Complex(5.0, 0.0), Complex(6.0, 0.0)],
        ]

        let b: [[Complex<Double>]] = [
            [Complex(7.0, 0.0), Complex(8.0, 0.0)],
            [Complex(9.0, 0.0), Complex(10.0, 0.0)],
            [Complex(11.0, 0.0), Complex(12.0, 0.0)],
        ]

        let result = await accelerator.matrixMultiply(a, b)

        #expect(result.count == 2, "Result should have 2 rows for 2x3 * 3x2 multiplication")
        #expect(result[0].count == 2, "Result should have 2 columns for 2x3 * 3x2 multiplication")

        let expected00 = Complex<Double>(58.0, 0.0)
        let expected01 = Complex<Double>(64.0, 0.0)
        let expected10 = Complex<Double>(139.0, 0.0)
        let expected11 = Complex<Double>(154.0, 0.0)

        #expect((result[0][0] - expected00).magnitude < 1e-10, "Result[0][0] should be 58 for non-square multiplication")
        #expect((result[0][1] - expected01).magnitude < 1e-10, "Result[0][1] should be 64 for non-square multiplication")
        #expect((result[1][0] - expected10).magnitude < 1e-10, "Result[1][0] should be 139 for non-square multiplication")
        #expect((result[1][1] - expected11).magnitude < 1e-10, "Result[1][1] should be 154 for non-square multiplication")
    }

    @Test("matrixMultiply returns empty for empty matrices")
    func matrixMultiplyEmpty() async {
        let accelerator = MPSMetalAcceleration()

        let empty: [[Complex<Double>]] = []
        let nonEmpty: [[Complex<Double>]] = [[.one]]

        let result1 = await accelerator.matrixMultiply(empty, nonEmpty)
        let result2 = await accelerator.matrixMultiply(nonEmpty, empty)
        let result3 = await accelerator.matrixMultiply(empty, empty)

        #expect(result1.isEmpty, "Empty left matrix should return empty result")
        #expect(result2.isEmpty, "Empty right matrix should return empty result")
        #expect(result3.isEmpty, "Both empty matrices should return empty result")
    }

    @Test("matrixMultiply handles 1x1 matrices")
    func matrixMultiply1x1() async {
        let accelerator = MPSMetalAcceleration()

        let a: [[Complex<Double>]] = [[Complex(3.0, 4.0)]]
        let b: [[Complex<Double>]] = [[Complex(1.0, 2.0)]]

        let result = await accelerator.matrixMultiply(a, b)

        let expected = Complex(3.0, 4.0) * Complex(1.0, 2.0)

        #expect(result.count == 1, "1x1 result should have 1 row")
        #expect(result[0].count == 1, "1x1 result should have 1 column")
        #expect((result[0][0] - expected).magnitude < 1e-10, "1x1 multiplication should compute correct product")
    }

    @Test("matrixMultiply returns empty for dimension mismatch")
    func matrixMultiplyDimensionMismatch() async {
        let accelerator = MPSMetalAcceleration()

        let a: [[Complex<Double>]] = [
            [.one, .zero],
            [.zero, .one],
        ]

        let b: [[Complex<Double>]] = [
            [.one, .zero, .zero],
            [.zero, .one, .zero],
            [.zero, .zero, .one],
        ]

        let result = await accelerator.matrixMultiply(a, b)

        #expect(result.isEmpty, "Dimension mismatch (2x2 * 3x3) should return empty result")
    }

    @Test("contractAdjacentTensors produces correct 4D result")
    func contractAdjacentTensorsBasic() async {
        let accelerator = MPSMetalAcceleration()

        let left = MPSTensor.groundState(site: 0, qubits: 4, maxBondDimension: 16)
        let right = MPSTensor.groundState(site: 1, qubits: 4, maxBondDimension: 16)

        let result = await accelerator.contractAdjacentTensors(left, right)

        #expect(result.count == left.leftBondDimension, "Result alpha dimension should match left tensor's left bond")
        #expect(result[0].count == 2, "Result should have physical dimension 2 for left physical index")
        #expect(result[0][0].count == 2, "Result should have physical dimension 2 for right physical index")
        #expect(result[0][0][0].count == right.rightBondDimension, "Result gamma dimension should match right tensor's right bond")

        #expect((result[0][0][0][0] - .one).magnitude < 1e-10, "Ground state contraction [0][0][0][0] should be 1")
        #expect(result[0][0][1][0].magnitude < 1e-10, "Ground state contraction [0][0][1][0] should be 0")
        #expect(result[0][1][0][0].magnitude < 1e-10, "Ground state contraction [0][1][0][0] should be 0")
        #expect(result[0][1][1][0].magnitude < 1e-10, "Ground state contraction [0][1][1][0] should be 0")
    }

    @Test("contractAdjacentTensors with basis state 1")
    func contractAdjacentTensorsBasisState() async {
        let accelerator = MPSMetalAcceleration()

        let left = MPSTensor.basisState(0b01, site: 0, qubits: 4, maxBondDimension: 16)
        let right = MPSTensor.basisState(0b01, site: 1, qubits: 4, maxBondDimension: 16)

        let result = await accelerator.contractAdjacentTensors(left, right)

        #expect((result[0][1][0][0] - .one).magnitude < 1e-10, "Basis |01> contraction [0][1][0][0] should be 1")
        #expect(result[0][0][0][0].magnitude < 1e-10, "Basis |01> contraction [0][0][0][0] should be 0")
        #expect(result[0][0][1][0].magnitude < 1e-10, "Basis |01> contraction [0][0][1][0] should be 0")
        #expect(result[0][1][1][0].magnitude < 1e-10, "Basis |01> contraction [0][1][1][0] should be 0")
    }

    @Test("chainContraction returns identity for empty input")
    func chainContractionEmpty() async {
        let accelerator = MPSMetalAcceleration()

        let result = await accelerator.chainContraction(matrices: [])

        #expect(result.count == 1, "Empty chain should return 1x1 identity")
        #expect(result[0].count == 1, "Empty chain should return 1x1 identity")
        #expect((result[0][0] - .one).magnitude < 1e-10, "Empty chain should return identity element [[1]]")
    }

    @Test("chainContraction returns single matrix unchanged")
    func chainContractionSingleMatrix() async {
        let accelerator = MPSMetalAcceleration()

        let matrix: [[Complex<Double>]] = [
            [Complex(1.0, 2.0), Complex(3.0, 4.0)],
            [Complex(5.0, 6.0), Complex(7.0, 8.0)],
        ]

        let result = await accelerator.chainContraction(matrices: [matrix])

        #expect(result.count == 2, "Single matrix chain should return same row count")
        #expect(result[0].count == 2, "Single matrix chain should return same column count")

        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                let diff = (result[i][j] - matrix[i][j]).magnitude
                #expect(diff < 1e-10, "Single matrix chain should be unchanged at [\(i)][\(j)]")
            }
        }
    }

    @Test("chainContraction computes correct sequential product")
    func chainContractionSequential() async {
        let accelerator = MPSMetalAcceleration()

        let m1: [[Complex<Double>]] = [
            [Complex(1.0, 0.0), Complex(2.0, 0.0)],
            [Complex(3.0, 0.0), Complex(4.0, 0.0)],
        ]

        let m2: [[Complex<Double>]] = [
            [Complex(5.0, 0.0), Complex(6.0, 0.0)],
            [Complex(7.0, 0.0), Complex(8.0, 0.0)],
        ]

        let m3: [[Complex<Double>]] = [
            [Complex(1.0, 0.0), Complex(0.0, 0.0)],
            [Complex(0.0, 0.0), Complex(1.0, 0.0)],
        ]

        let result = await accelerator.chainContraction(matrices: [m1, m2, m3])

        let m1m2Expected00 = Complex(1 * 5 + 2 * 7, 0.0)
        let m1m2Expected01 = Complex(1 * 6 + 2 * 8, 0.0)
        let m1m2Expected10 = Complex(3 * 5 + 4 * 7, 0.0)
        let m1m2Expected11 = Complex(3 * 6 + 4 * 8, 0.0)

        #expect((result[0][0] - m1m2Expected00).magnitude < 1e-10, "Chain M1*M2*I result[0][0] should be 19")
        #expect((result[0][1] - m1m2Expected01).magnitude < 1e-10, "Chain M1*M2*I result[0][1] should be 22")
        #expect((result[1][0] - m1m2Expected10).magnitude < 1e-10, "Chain M1*M2*I result[1][0] should be 43")
        #expect((result[1][1] - m1m2Expected11).magnitude < 1e-10, "Chain M1*M2*I result[1][1] should be 50")
    }

    @Test("CPU and GPU paths produce same results for medium matrices")
    func cpuGpuComparison() async {
        let accelerator = MPSMetalAcceleration()

        let smallSize = MPSMetalAcceleration.gpuThreshold - 1
        let largeSize = MPSMetalAcceleration.gpuThreshold

        var smallA = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: smallSize), count: smallSize)
        var smallB = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: smallSize), count: smallSize)
        var largeA = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: largeSize), count: largeSize)
        var largeB = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: largeSize), count: largeSize)

        for i in 0 ..< smallSize {
            for j in 0 ..< smallSize {
                smallA[i][j] = Complex(Double(i + j + 1), Double(i - j))
                smallB[i][j] = Complex(Double(i * j + 1), Double(j))
            }
        }

        for i in 0 ..< largeSize {
            for j in 0 ..< largeSize {
                largeA[i][j] = Complex(Double(i + j + 1), Double(i - j))
                largeB[i][j] = Complex(Double(i * j + 1), Double(j))
            }
        }

        let smallResult = await accelerator.matrixMultiply(smallA, smallB)
        let largeResult = await accelerator.matrixMultiply(largeA, largeB)

        #expect(smallResult.count == smallSize, "Small matrix result should have correct row count")
        #expect(largeResult.count == largeSize, "Large matrix result should have correct row count")

        for i in 0 ..< smallSize {
            for j in 0 ..< smallSize {
                var expected: Complex<Double> = .zero
                for k in 0 ..< smallSize {
                    expected = expected + smallA[i][k] * smallB[k][j]
                }
                let diff = (smallResult[i][j] - expected).magnitude
                #expect(diff < 1e-10, "Small matrix (CPU path) result[\(i)][\(j)] should match manual computation")
            }
        }

        let tolerance: Double = accelerator.isAvailable ? 1e-5 : 1e-10

        for i in 0 ..< min(5, largeSize) {
            for j in 0 ..< min(5, largeSize) {
                var expected: Complex<Double> = .zero
                for k in 0 ..< largeSize {
                    expected = expected + largeA[i][k] * largeB[k][j]
                }
                let diff = (largeResult[i][j] - expected).magnitude
                #expect(diff < tolerance, "Large matrix result[\(i)][\(j)] should match manual computation within tolerance")
            }
        }
    }

    @Test("Results are consistent between multiple calls")
    func resultConsistency() async {
        let accelerator = MPSMetalAcceleration()

        let a: [[Complex<Double>]] = [
            [Complex(1.0, 1.0), Complex(2.0, -1.0)],
            [Complex(-1.0, 2.0), Complex(3.0, 0.0)],
        ]

        let b: [[Complex<Double>]] = [
            [Complex(0.5, 0.5), Complex(1.0, 0.0)],
            [Complex(0.0, 1.0), Complex(-1.0, 1.0)],
        ]

        let result1 = await accelerator.matrixMultiply(a, b)
        let result2 = await accelerator.matrixMultiply(a, b)
        let result3 = await accelerator.matrixMultiply(a, b)

        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                let diff12 = (result1[i][j] - result2[i][j]).magnitude
                let diff13 = (result1[i][j] - result3[i][j]).magnitude
                #expect(diff12 < 1e-10, "Results should be consistent between calls at [\(i)][\(j)]")
                #expect(diff13 < 1e-10, "Results should be consistent across multiple calls at [\(i)][\(j)]")
            }
        }
    }

    @Test("matrixMultiply handles very small values")
    func matrixMultiplySmallValues() async {
        let accelerator = MPSMetalAcceleration()

        let a: [[Complex<Double>]] = [
            [Complex(1e-10, 1e-10), Complex(2e-10, 0.0)],
            [Complex(0.0, 1e-10), Complex(1e-10, -1e-10)],
        ]

        let b: [[Complex<Double>]] = [
            [Complex(1e-10, 0.0), Complex(0.0, 1e-10)],
            [Complex(1e-10, 1e-10), Complex(2e-10, 0.0)],
        ]

        let result = await accelerator.matrixMultiply(a, b)

        #expect(result.count == 2, "Small value multiplication should produce valid result")
        #expect(result[0].count == 2, "Small value multiplication should have correct dimensions")

        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                #expect(result[i][j].isFinite, "Result[\(i)][\(j)] should be finite for small input values")
            }
        }
    }

    @Test("matrixMultiply handles mixed magnitude values")
    func matrixMultiplyMixedMagnitude() async {
        let accelerator = MPSMetalAcceleration()

        let a: [[Complex<Double>]] = [
            [Complex(1e6, 0.0), Complex(1e-6, 0.0)],
            [Complex(1e-6, 0.0), Complex(1e6, 0.0)],
        ]

        let b: [[Complex<Double>]] = [
            [Complex(1e-6, 0.0), Complex(1e6, 0.0)],
            [Complex(1e6, 0.0), Complex(1e-6, 0.0)],
        ]

        let result = await accelerator.matrixMultiply(a, b)

        let expected00 = Complex(1e6 * 1e-6 + 1e-6 * 1e6, 0.0)

        #expect(result.count == 2, "Mixed magnitude multiplication should produce valid result")
        #expect(abs(result[0][0].real - expected00.real) < 1e-6, "Mixed magnitude result[0][0] should be approximately 2")
    }

    @Test("matrixMultiply handles matrices at GPU threshold boundary")
    func matrixMultiplyAtThreshold() async {
        let accelerator = MPSMetalAcceleration()

        let sizes = [
            MPSMetalAcceleration.gpuThreshold - 1,
            MPSMetalAcceleration.gpuThreshold,
            MPSMetalAcceleration.gpuThreshold + 1,
        ]

        for size in sizes {
            var a = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: size), count: size)
            var b = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: size), count: size)

            for i in 0 ..< size {
                a[i][i] = .one
                b[i][i] = Complex(2.0, 0.0)
            }

            let result = await accelerator.matrixMultiply(a, b)

            #expect(result.count == size, "Result should have \(size) rows at threshold boundary")

            let tolerance: Double = (size >= MPSMetalAcceleration.gpuThreshold && accelerator.isAvailable) ? 1e-6 : 1e-10

            for i in 0 ..< size {
                let diff = (result[i][i] - Complex(2.0, 0.0)).magnitude
                #expect(diff < tolerance, "Diagonal element [\(i)][\(i)] should be 2 for size \(size)")
            }
        }
    }

    @Test("matrixMultiply handles large matrices efficiently")
    func matrixMultiplyLarge() async {
        let accelerator = MPSMetalAcceleration()

        let size = 64
        var a = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: size), count: size)
        var b = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: size), count: size)

        for i in 0 ..< size {
            for j in 0 ..< size {
                a[i][j] = Complex(Double(i == j ? 1 : 0), 0.0)
                b[i][j] = Complex(Double((i + j) % 10), Double((i * j) % 5))
            }
        }

        let result = await accelerator.matrixMultiply(a, b)

        #expect(result.count == size, "Large 64x64 multiplication should produce correct row count")
        #expect(result[0].count == size, "Large 64x64 multiplication should produce correct column count")

        let tolerance: Double = accelerator.isAvailable ? 1e-5 : 1e-10

        for i in 0 ..< size {
            let diff = (result[i][i] - b[i][i]).magnitude
            #expect(diff < tolerance, "Identity * B diagonal element [\(i)][\(i)] should equal B[\(i)][\(i)]")
        }
    }

    @Test("matrixMultiply handles pure imaginary matrices")
    func matrixMultiplyPureImaginary() async {
        let accelerator = MPSMetalAcceleration()

        let a: [[Complex<Double>]] = [
            [Complex(0.0, 1.0), Complex(0.0, 2.0)],
            [Complex(0.0, 3.0), Complex(0.0, 4.0)],
        ]

        let b: [[Complex<Double>]] = [
            [Complex(0.0, 1.0), Complex(0.0, 0.0)],
            [Complex(0.0, 0.0), Complex(0.0, 1.0)],
        ]

        let result = await accelerator.matrixMultiply(a, b)

        let expected00 = Complex(0.0, 1.0) * Complex(0.0, 1.0) + Complex(0.0, 2.0) * Complex(0.0, 0.0)
        let expected01 = Complex(0.0, 1.0) * Complex(0.0, 0.0) + Complex(0.0, 2.0) * Complex(0.0, 1.0)
        let expected10 = Complex(0.0, 3.0) * Complex(0.0, 1.0) + Complex(0.0, 4.0) * Complex(0.0, 0.0)
        let expected11 = Complex(0.0, 3.0) * Complex(0.0, 0.0) + Complex(0.0, 4.0) * Complex(0.0, 1.0)

        #expect((result[0][0] - expected00).magnitude < 1e-10, "Pure imaginary result[0][0] should be -1")
        #expect((result[0][1] - expected01).magnitude < 1e-10, "Pure imaginary result[0][1] should be -2")
        #expect((result[1][0] - expected10).magnitude < 1e-10, "Pure imaginary result[1][0] should be -3")
        #expect((result[1][1] - expected11).magnitude < 1e-10, "Pure imaginary result[1][1] should be -4")
    }

    @Test("matrixMultiply with zero matrix produces zero result")
    func matrixMultiplyZero() async {
        let accelerator = MPSMetalAcceleration()

        let zero: [[Complex<Double>]] = [
            [.zero, .zero],
            [.zero, .zero],
        ]

        let nonZero: [[Complex<Double>]] = [
            [Complex(1.0, 2.0), Complex(3.0, 4.0)],
            [Complex(5.0, 6.0), Complex(7.0, 8.0)],
        ]

        let result1 = await accelerator.matrixMultiply(zero, nonZero)
        let result2 = await accelerator.matrixMultiply(nonZero, zero)

        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                #expect(result1[i][j].magnitude < 1e-10, "Zero * A should give zero at [\(i)][\(j)]")
                #expect(result2[i][j].magnitude < 1e-10, "A * Zero should give zero at [\(i)][\(j)]")
            }
        }
    }

    @Test("chainContraction with MPS tensor matrices")
    func chainContractionMPSTensors() async {
        let accelerator = MPSMetalAcceleration()

        let tensor0 = MPSTensor.groundState(site: 0, qubits: 3, maxBondDimension: 16)
        let tensor1 = MPSTensor.groundState(site: 1, qubits: 3, maxBondDimension: 16)
        let tensor2 = MPSTensor.groundState(site: 2, qubits: 3, maxBondDimension: 16)

        let matrices = [
            tensor0.matrixForPhysicalIndex(0),
            tensor1.matrixForPhysicalIndex(0),
            tensor2.matrixForPhysicalIndex(0),
        ]

        let result = await accelerator.chainContraction(matrices: matrices)

        #expect(result.count == 1, "Ground state chain should produce 1x1 result")
        #expect(result[0].count == 1, "Ground state chain should produce 1x1 result")
        #expect((result[0][0] - .one).magnitude < 1e-10, "Ground state |000> chain contraction should be 1")
    }

    @Test("Multiple concurrent matrixMultiply calls produce correct results")
    func concurrentMatrixMultiply() async {
        let accelerator = MPSMetalAcceleration()

        let a: [[Complex<Double>]] = [
            [Complex(1.0, 0.0), Complex(0.0, 0.0)],
            [Complex(0.0, 0.0), Complex(1.0, 0.0)],
        ]

        let b: [[Complex<Double>]] = [
            [Complex(2.0, 1.0), Complex(3.0, 2.0)],
            [Complex(4.0, 3.0), Complex(5.0, 4.0)],
        ]

        async let result1 = accelerator.matrixMultiply(a, b)
        async let result2 = accelerator.matrixMultiply(a, b)
        async let result3 = accelerator.matrixMultiply(a, b)

        let results = await [result1, result2, result3]

        for (index, result) in results.enumerated() {
            #expect(result.count == 2, "Concurrent result \(index) should have 2 rows")
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    let diff = (result[i][j] - b[i][j]).magnitude
                    #expect(diff < 1e-10, "Concurrent result \(index) at [\(i)][\(j)] should match expected")
                }
            }
        }
    }

    @Test("Full tensor contraction workflow")
    func fullTensorContractionWorkflow() async {
        let accelerator = MPSMetalAcceleration()

        let tensor0 = MPSTensor.basisState(0b101, site: 0, qubits: 3, maxBondDimension: 16)
        let tensor1 = MPSTensor.basisState(0b101, site: 1, qubits: 3, maxBondDimension: 16)
        let tensor2 = MPSTensor.basisState(0b101, site: 2, qubits: 3, maxBondDimension: 16)

        let contracted01 = await accelerator.contractAdjacentTensors(tensor0, tensor1)

        #expect(contracted01.count == 1, "First contraction should have alpha dimension 1")

        let physBits = [(0b101 >> 0) & 1, (0b101 >> 1) & 1, (0b101 >> 2) & 1]

        #expect((contracted01[0][physBits[0]][physBits[1]][0] - .one).magnitude < 1e-10, "Contracted amplitude for correct bits should be 1")

        let matrix2 = tensor2.matrixForPhysicalIndex(physBits[2])
        let finalMatrix = [[contracted01[0][physBits[0]][physBits[1]][0]]]

        let finalResult = await accelerator.chainContraction(matrices: [finalMatrix, matrix2])

        #expect((finalResult[0][0] - .one).magnitude < 1e-10, "Full contraction of |101> should give amplitude 1")
    }
}
