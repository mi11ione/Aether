// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for SVD reconstruction property.
/// Validates that M = U * S * V-dagger within numerical tolerance,
/// ensuring the decomposition preserves the original matrix information.
@Suite("SVDDecomposition - Matrix Reconstruction")
struct SVDReconstructionTests {
    @Test("Reconstruct 2x2 real matrix: M = U * S * V-dagger")
    func reconstruct2x2RealMatrix() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(2, 0)],
            [Complex(3, 0), Complex(4, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .none)

        let m = result.u.count
        let k = result.singularValues.count
        let n = result.vDagger[0].count
        var reconstructed = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: n),
            count: m,
        )
        for i in 0 ..< m {
            for j in 0 ..< n {
                var sum = Complex<Double>.zero
                for l in 0 ..< k {
                    sum = sum + result.u[i][l] * result.singularValues[l] * result.vDagger[l][j]
                }
                reconstructed[i][j] = sum
            }
        }

        for row in 0 ..< matrix.count {
            for col in 0 ..< matrix[0].count {
                let diff = (reconstructed[row][col] - matrix[row][col]).magnitude
                #expect(
                    diff < 1e-10,
                    "Reconstruction error at [\(row)][\(col)]: expected \(matrix[row][col]), got \(reconstructed[row][col]). SVD must satisfy M = U * S * V-dagger.",
                )
            }
        }
    }

    @Test("Reconstruct 2x2 complex matrix: M = U * S * V-dagger")
    func reconstruct2x2ComplexMatrix() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 2), Complex(3, -1)],
            [Complex(-2, 1), Complex(4, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .none)

        let m = result.u.count
        let k = result.singularValues.count
        let n = result.vDagger[0].count
        var reconstructed = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: n),
            count: m,
        )
        for i in 0 ..< m {
            for j in 0 ..< n {
                var sum = Complex<Double>.zero
                for l in 0 ..< k {
                    sum = sum + result.u[i][l] * result.singularValues[l] * result.vDagger[l][j]
                }
                reconstructed[i][j] = sum
            }
        }

        for row in 0 ..< matrix.count {
            for col in 0 ..< matrix[0].count {
                let diff = (reconstructed[row][col] - matrix[row][col]).magnitude
                #expect(
                    diff < 1e-10,
                    "Complex matrix reconstruction failed at [\(row)][\(col)]. Difference: \(diff). SVD decomposition must preserve complex phase information.",
                )
            }
        }
    }

    @Test("Reconstruct 3x3 Hermitian matrix")
    func reconstruct3x3HermitianMatrix() {
        let matrix: [[Complex<Double>]] = [
            [Complex(2, 0), Complex(1, 1), Complex(0, 0)],
            [Complex(1, -1), Complex(3, 0), Complex(1, 2)],
            [Complex(0, 0), Complex(1, -2), Complex(1, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .none)

        let m = result.u.count
        let k = result.singularValues.count
        let n = result.vDagger[0].count
        var reconstructed = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: n),
            count: m,
        )
        for i in 0 ..< m {
            for j in 0 ..< n {
                var sum = Complex<Double>.zero
                for l in 0 ..< k {
                    sum = sum + result.u[i][l] * result.singularValues[l] * result.vDagger[l][j]
                }
                reconstructed[i][j] = sum
            }
        }

        for row in 0 ..< matrix.count {
            for col in 0 ..< matrix[0].count {
                let diff = (reconstructed[row][col] - matrix[row][col]).magnitude
                #expect(
                    diff < 1e-10,
                    "Hermitian matrix reconstruction failed at [\(row)][\(col)]. SVD must correctly decompose Hermitian operators used in quantum mechanics.",
                )
            }
        }
    }

    @Test("Reconstruct diagonal matrix")
    func reconstructDiagonalMatrix() {
        let matrix: [[Complex<Double>]] = [
            [Complex(5, 0), Complex(0, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(3, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(0, 0), Complex(1, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .none)

        let m = result.u.count
        let k = result.singularValues.count
        let n = result.vDagger[0].count
        var reconstructed = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: n),
            count: m,
        )
        for i in 0 ..< m {
            for j in 0 ..< n {
                var sum = Complex<Double>.zero
                for l in 0 ..< k {
                    sum = sum + result.u[i][l] * result.singularValues[l] * result.vDagger[l][j]
                }
                reconstructed[i][j] = sum
            }
        }

        for row in 0 ..< matrix.count {
            for col in 0 ..< matrix[0].count {
                let diff = (reconstructed[row][col] - matrix[row][col]).magnitude
                #expect(
                    diff < 1e-10,
                    "Diagonal matrix reconstruction failed. Singular values should equal diagonal entries (in descending order).",
                )
            }
        }
    }
}

/// Test suite for singular value properties.
/// Validates that singular values are non-negative and sorted in descending order,
/// which is essential for MPS bond dimension truncation algorithms.
@Suite("SVDDecomposition - Singular Value Properties")
struct SingularValuePropertiesTests {
    @Test("Singular values are non-negative")
    func singularValuesNonNegative() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 2), Complex(-3, 1)],
            [Complex(2, -1), Complex(4, 3)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix)

        for (index, sv) in result.singularValues.enumerated() {
            #expect(
                sv >= 0,
                "Singular value at index \(index) is negative (\(sv)). By definition, singular values must be non-negative real numbers.",
            )
        }
    }

    @Test("Singular values in descending order")
    func singularValuesDescendingOrder() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(2, 0), Complex(3, 0)],
            [Complex(4, 0), Complex(5, 0), Complex(6, 0)],
            [Complex(7, 0), Complex(8, 0), Complex(9, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix)

        for i in 0 ..< result.singularValues.count - 1 {
            #expect(
                result.singularValues[i] >= result.singularValues[i + 1],
                "Singular values not in descending order: s[\(i)] = \(result.singularValues[i]) < s[\(i + 1)] = \(result.singularValues[i + 1]). LAPACK zgesdd should return sorted values.",
            )
        }
    }

    @Test("Singular values count equals min(m, n)")
    func singularValuesCountCorrect() {
        let matrix3x2: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(2, 0)],
            [Complex(3, 0), Complex(4, 0)],
            [Complex(5, 0), Complex(6, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix3x2)

        #expect(
            result.singularValues.count == 2,
            "For 3x2 matrix, expected 2 singular values (min(3,2)), got \(result.singularValues.count). Thin SVD should produce k = min(m,n) singular values.",
        )
    }

    @Test("Diagonal matrix singular values equal absolute diagonal entries")
    func diagonalMatrixSingularValues() {
        let matrix: [[Complex<Double>]] = [
            [Complex(3, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(-5, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix)

        #expect(
            abs(result.singularValues[0] - 5.0) < 1e-10,
            "Largest singular value should be 5 (|−5|), got \(result.singularValues[0]). Singular values are absolute values of eigenvalues for diagonal matrices.",
        )
        #expect(
            abs(result.singularValues[1] - 3.0) < 1e-10,
            "Second singular value should be 3, got \(result.singularValues[1]).",
        )
    }

    @Test("Rank-deficient matrix has zero singular values")
    func rankDeficientMatrixZeroSingularValues() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(2, 0)],
            [Complex(2, 0), Complex(4, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix)

        #expect(
            result.singularValues.count == 2,
            "Expected 2 singular values for 2x2 matrix.",
        )
        #expect(
            result.singularValues[1] < 1e-10,
            "Rank-1 matrix should have second singular value near zero, got \(result.singularValues[1]). The matrix rows are linearly dependent.",
        )
    }
}

/// Test suite for maxBondDimension truncation strategy.
/// Validates that truncation keeps exactly the specified number of largest singular values,
/// critical for controlling MPS tensor network bond dimensions.
@Suite("SVDDecomposition - maxBondDimension Truncation")
struct MaxBondDimensionTruncationTests {
    @Test("maxBondDimension keeps correct count")
    func maxBondDimensionKeepsCorrectCount() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(2, 0), Complex(3, 0)],
            [Complex(4, 0), Complex(5, 0), Complex(6, 0)],
            [Complex(7, 0), Complex(8, 0), Complex(9, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .maxBondDimension(2))

        #expect(
            result.singularValues.count == 2,
            "maxBondDimension(2) should keep exactly 2 singular values, got \(result.singularValues.count).",
        )
    }

    @Test("maxBondDimension truncates U columns")
    func maxBondDimensionTruncatesUColumns() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(2, 0), Complex(3, 0)],
            [Complex(4, 0), Complex(5, 0), Complex(6, 0)],
            [Complex(7, 0), Complex(8, 0), Complex(9, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .maxBondDimension(1))

        #expect(
            result.u.count == 3,
            "U should have 3 rows (m dimension preserved), got \(result.u.count).",
        )
        #expect(
            result.u[0].count == 1,
            "U should have 1 column after truncation, got \(result.u[0].count).",
        )
    }

    @Test("maxBondDimension truncates V-dagger rows")
    func maxBondDimensionTruncatesVDaggerRows() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(2, 0), Complex(3, 0)],
            [Complex(4, 0), Complex(5, 0), Complex(6, 0)],
            [Complex(7, 0), Complex(8, 0), Complex(9, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .maxBondDimension(2))

        #expect(
            result.vDagger.count == 2,
            "V-dagger should have 2 rows after truncation, got \(result.vDagger.count).",
        )
        #expect(
            result.vDagger[0].count == 3,
            "V-dagger should have 3 columns (n dimension preserved), got \(result.vDagger[0].count).",
        )
    }

    @Test("maxBondDimension larger than rank keeps all")
    func maxBondDimensionLargerThanRankKeepsAll() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(2, 0)],
            [Complex(3, 0), Complex(4, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .maxBondDimension(10))

        #expect(
            result.singularValues.count == 2,
            "maxBondDimension(10) on 2x2 matrix should keep all 2 singular values, got \(result.singularValues.count). Cannot keep more than min(m,n).",
        )
    }

    @Test("maxBondDimension(1) produces rank-1 approximation")
    func maxBondDimensionOneProducesRankOne() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(2, 0)],
            [Complex(3, 0), Complex(4, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .maxBondDimension(1))

        #expect(
            result.singularValues.count == 1,
            "maxBondDimension(1) should produce exactly 1 singular value for rank-1 approximation.",
        )
        #expect(
            result.u[0].count == 1,
            "U should have exactly 1 column for rank-1 approximation.",
        )
        #expect(
            result.vDagger.count == 1,
            "V-dagger should have exactly 1 row for rank-1 approximation.",
        )
    }
}

/// Test suite for relativeThreshold truncation strategy.
/// Validates that singular values below threshold * max(S) are discarded,
/// useful for adaptive truncation based on spectral properties.
@Suite("SVDDecomposition - relativeThreshold Truncation")
struct RelativeThresholdTruncationTests {
    @Test("relativeThreshold discards values below cutoff")
    func relativeThresholdDiscardsBelowCutoff() {
        let matrix: [[Complex<Double>]] = [
            [Complex(10, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(0.5, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .relativeThreshold(0.1))

        #expect(
            result.singularValues.count == 1,
            "relativeThreshold(0.1) should discard 0.5 (< 0.1 * 10 = 1.0), keeping only 1 singular value. Got \(result.singularValues.count).",
        )
        #expect(
            abs(result.singularValues[0] - 10.0) < 1e-10,
            "Kept singular value should be 10.0, got \(result.singularValues[0]).",
        )
    }

    @Test("relativeThreshold keeps values above cutoff")
    func relativeThresholdKeepsAboveCutoff() {
        let matrix: [[Complex<Double>]] = [
            [Complex(10, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(2, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .relativeThreshold(0.1))

        #expect(
            result.singularValues.count == 2,
            "relativeThreshold(0.1) should keep 2 (>= 0.1 * 10 = 1.0), keeping both singular values. Got \(result.singularValues.count).",
        )
    }

    @Test("relativeThreshold keeps at least one singular value")
    func relativeThresholdKeepsAtLeastOne() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(0.001, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .relativeThreshold(0.99))

        #expect(
            result.singularValues.count >= 1,
            "relativeThreshold should always keep at least 1 singular value to avoid empty decomposition. Got \(result.singularValues.count).",
        )
    }

    @Test("relativeThreshold(0) keeps all singular values")
    func relativeThresholdZeroKeepsAll() {
        let matrix: [[Complex<Double>]] = [
            [Complex(5, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(0.001, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .relativeThreshold(0))

        #expect(
            result.singularValues.count == 2,
            "relativeThreshold(0) should keep all non-zero singular values. Got \(result.singularValues.count).",
        )
    }
}

/// Test suite for cumulativeWeight truncation strategy.
/// Validates that the smallest set of singular values retaining (1-epsilon) of total weight is kept,
/// optimal for minimizing truncation error in MPS compression.
@Suite("SVDDecomposition - cumulativeWeight Truncation")
struct CumulativeWeightTruncationTests {
    @Test("cumulativeWeight retains correct weight fraction")
    func cumulativeWeightRetainsCorrectFraction() {
        let matrix: [[Complex<Double>]] = [
            [Complex(3, 0), Complex(0, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(2, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(0, 0), Complex(1, 0)],
        ]

        let fullResult = SVDDecomposition.decompose(matrix: matrix, truncation: .none)
        let totalWeight = fullResult.singularValues.reduce(0.0) { $0 + $1 * $1 }

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .cumulativeWeight(epsilon: 0.1))

        let keptWeight = result.singularValues.reduce(0.0) { $0 + $1 * $1 }
        let targetWeight = (1.0 - 0.1) * totalWeight

        #expect(
            keptWeight >= targetWeight - 1e-10,
            "cumulativeWeight(epsilon: 0.1) should retain at least 90% of total weight. Kept: \(keptWeight), Target: \(targetWeight). Total: \(totalWeight).",
        )
    }

    @Test("cumulativeWeight(epsilon: 0) keeps all singular values")
    func cumulativeWeightEpsilonZeroKeepsAll() {
        let matrix: [[Complex<Double>]] = [
            [Complex(3, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(1, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .cumulativeWeight(epsilon: 0))

        #expect(
            result.singularValues.count == 2,
            "cumulativeWeight(epsilon: 0) should keep all singular values. Got \(result.singularValues.count).",
        )
    }

    @Test("cumulativeWeight with large epsilon keeps minimum")
    func cumulativeWeightLargeEpsilonKeepsMinimum() {
        let matrix: [[Complex<Double>]] = [
            [Complex(10, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(1, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .cumulativeWeight(epsilon: 0.99))

        #expect(
            result.singularValues.count >= 1,
            "cumulativeWeight should always keep at least 1 singular value. Got \(result.singularValues.count).",
        )
    }

    @Test("cumulativeWeight minimizes kept count while meeting threshold")
    func cumulativeWeightMinimizesKeptCount() {
        let matrix: [[Complex<Double>]] = [
            [Complex(10, 0), Complex(0, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(1, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(0, 0), Complex(0.1, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .cumulativeWeight(epsilon: 0.05))

        #expect(
            result.singularValues.count <= 2,
            "With epsilon=0.05 and dominant singular value 10, should not need all 3 values. Sigma squared: 100, 1, 0.01. Total: 101.01. 95% threshold: 95.96. First value alone: 100 >= 95.96.",
        )
    }
}

/// Test suite for no truncation strategy.
/// Validates that .none preserves all singular values and produces exact decomposition.
/// The default truncation mode used when maximum accuracy is required.
@Suite("SVDDecomposition - No Truncation")
struct NoTruncationTests {
    @Test("Truncation .none keeps all singular values")
    func noTruncationKeepsAll() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(2, 0), Complex(3, 0)],
            [Complex(4, 0), Complex(5, 0), Complex(6, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .none)

        #expect(
            result.singularValues.count == 2,
            "For 2x3 matrix with .none truncation, should keep all min(2,3) = 2 singular values. Got \(result.singularValues.count).",
        )
    }

    @Test("Default truncation is .none")
    func defaultTruncationIsNone() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(2, 0)],
            [Complex(3, 0), Complex(4, 0)],
        ]

        let resultDefault = SVDDecomposition.decompose(matrix: matrix)
        let resultNone = SVDDecomposition.decompose(matrix: matrix, truncation: .none)

        #expect(
            resultDefault.singularValues.count == resultNone.singularValues.count,
            "Default truncation should be .none. Different singular value counts: default=\(resultDefault.singularValues.count), none=\(resultNone.singularValues.count).",
        )
    }

    @Test("No truncation produces zero truncation error")
    func noTruncationZeroError() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 2), Complex(3, 4)],
            [Complex(5, 6), Complex(7, 8)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .none)

        #expect(
            result.truncationError < 1e-10,
            "Truncation .none should produce zero truncation error. Got \(result.truncationError).",
        )
    }
}

/// Test suite for truncation error computation.
/// Validates that truncation error equals sum of squared discarded singular values,
/// which represents the Frobenius norm error of the low-rank approximation.
@Suite("SVDDecomposition - Truncation Error")
struct TruncationErrorTests {
    @Test("Truncation error is sum of squared discarded values")
    func truncationErrorIsSumSquaredDiscarded() {
        let matrix: [[Complex<Double>]] = [
            [Complex(5, 0), Complex(0, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(3, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(0, 0), Complex(1, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .maxBondDimension(1))

        let expectedError = 3.0 * 3.0 + 1.0 * 1.0
        #expect(
            abs(result.truncationError - expectedError) < 1e-10,
            "Truncation error should be 3^2 + 1^2 = 10.0. Got \(result.truncationError). Error equals sum of squared discarded singular values.",
        )
    }

    @Test("Truncation error is zero when keeping all")
    func truncationErrorZeroWhenKeepingAll() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(2, 0)],
            [Complex(3, 0), Complex(4, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .none)

        #expect(
            result.truncationError == 0.0,
            "Truncation error should be exactly 0.0 when keeping all singular values. Got \(result.truncationError).",
        )
    }

    @Test("Truncation error is non-negative")
    func truncationErrorNonNegative() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 2), Complex(3, -1)],
            [Complex(-2, 1), Complex(4, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .maxBondDimension(1))

        #expect(
            result.truncationError >= 0,
            "Truncation error must be non-negative (sum of squared values). Got \(result.truncationError).",
        )
    }

    @Test("Truncation error equals Frobenius norm squared difference")
    func truncationErrorEqualsFrobeniusNormDifference() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(2, 0)],
            [Complex(3, 0), Complex(4, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .maxBondDimension(1))

        let m = result.u.count
        let k = result.singularValues.count
        let n = result.vDagger[0].count
        var reconstructed = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: n),
            count: m,
        )
        for i in 0 ..< m {
            for j in 0 ..< n {
                var sum = Complex<Double>.zero
                for l in 0 ..< k {
                    sum = sum + result.u[i][l] * result.singularValues[l] * result.vDagger[l][j]
                }
                reconstructed[i][j] = sum
            }
        }

        var frobeniusNormSquared = 0.0
        for row in 0 ..< matrix.count {
            for col in 0 ..< matrix[0].count {
                let diff = matrix[row][col] - reconstructed[row][col]
                frobeniusNormSquared += diff.magnitudeSquared
            }
        }

        #expect(
            abs(result.truncationError - frobeniusNormSquared) < 1e-10,
            "Truncation error should equal ||M - M_approx||_F^2. Truncation error: \(result.truncationError), Frobenius: \(frobeniusNormSquared).",
        )
    }
}

/// Test suite for edge cases in SVD decomposition.
/// Validates correct behavior for 1x1, rectangular, and square matrices.
/// Covers zero matrices, identity matrices, and various aspect ratios.
@Suite("SVDDecomposition - Edge Cases")
struct SVDEdgeCasesTests {
    @Test("1x1 matrix decomposition")
    func oneByOneMatrix() {
        let matrix: [[Complex<Double>]] = [
            [Complex(5, 3)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix)

        #expect(
            result.singularValues.count == 1,
            "1x1 matrix should have exactly 1 singular value. Got \(result.singularValues.count).",
        )

        let expectedSV = Complex(5.0, 3.0).magnitude
        #expect(
            abs(result.singularValues[0] - expectedSV) < 1e-10,
            "1x1 matrix singular value should equal magnitude. Expected \(expectedSV), got \(result.singularValues[0]).",
        )

        #expect(
            result.u.count == 1 && result.u[0].count == 1,
            "U should be 1x1 for 1x1 input matrix.",
        )
        #expect(
            result.vDagger.count == 1 && result.vDagger[0].count == 1,
            "V-dagger should be 1x1 for 1x1 input matrix.",
        )
    }

    @Test("Tall rectangular matrix (m > n)")
    func tallRectangularMatrix() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(2, 0)],
            [Complex(3, 0), Complex(4, 0)],
            [Complex(5, 0), Complex(6, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix)

        #expect(
            result.singularValues.count == 2,
            "3x2 matrix should have 2 singular values (min(3,2)). Got \(result.singularValues.count).",
        )
        #expect(
            result.u.count == 3,
            "U should have 3 rows for 3x2 input. Got \(result.u.count).",
        )
        #expect(
            result.u[0].count == 2,
            "U should have 2 columns for 3x2 input. Got \(result.u[0].count).",
        )
        #expect(
            result.vDagger.count == 2,
            "V-dagger should have 2 rows for 3x2 input. Got \(result.vDagger.count).",
        )
        #expect(
            result.vDagger[0].count == 2,
            "V-dagger should have 2 columns for 3x2 input. Got \(result.vDagger[0].count).",
        )
    }

    @Test("Wide rectangular matrix (m < n)")
    func wideRectangularMatrix() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(2, 0), Complex(3, 0)],
            [Complex(4, 0), Complex(5, 0), Complex(6, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix)

        #expect(
            result.singularValues.count == 2,
            "2x3 matrix should have 2 singular values (min(2,3)). Got \(result.singularValues.count).",
        )
        #expect(
            result.u.count == 2,
            "U should have 2 rows for 2x3 input. Got \(result.u.count).",
        )
        #expect(
            result.u[0].count == 2,
            "U should have 2 columns for 2x3 input. Got \(result.u[0].count).",
        )
        #expect(
            result.vDagger.count == 2,
            "V-dagger should have 2 rows for 2x3 input. Got \(result.vDagger.count).",
        )
        #expect(
            result.vDagger[0].count == 3,
            "V-dagger should have 3 columns for 2x3 input. Got \(result.vDagger[0].count).",
        )
    }

    @Test("Square matrix decomposition")
    func squareMatrix() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(2, 0), Complex(3, 0)],
            [Complex(4, 0), Complex(5, 0), Complex(6, 0)],
            [Complex(7, 0), Complex(8, 0), Complex(9, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix)

        #expect(
            result.singularValues.count == 3,
            "3x3 matrix should have 3 singular values. Got \(result.singularValues.count).",
        )
        #expect(
            result.u.count == 3 && result.u[0].count == 3,
            "U should be 3x3 for 3x3 input. Got \(result.u.count)x\(result.u[0].count).",
        )
        #expect(
            result.vDagger.count == 3 && result.vDagger[0].count == 3,
            "V-dagger should be 3x3 for 3x3 input. Got \(result.vDagger.count)x\(result.vDagger[0].count).",
        )
    }

    @Test("Zero matrix has all zero singular values")
    func zeroMatrix() {
        let matrix: [[Complex<Double>]] = [
            [Complex(0, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(0, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix)

        for (index, sv) in result.singularValues.enumerated() {
            #expect(
                sv < 1e-10,
                "Zero matrix should have all zero singular values. Got s[\(index)] = \(sv).",
            )
        }
    }

    @Test("Identity matrix has all singular values equal to 1")
    func identityMatrix() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(0, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(1, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(0, 0), Complex(1, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix)

        for (index, sv) in result.singularValues.enumerated() {
            #expect(
                abs(sv - 1.0) < 1e-10,
                "Identity matrix should have all singular values = 1. Got s[\(index)] = \(sv).",
            )
        }
    }
}

/// Test suite for unitary properties of U and V matrices.
/// Validates that U and V have orthonormal columns, essential for quantum gate decompositions.
/// Ensures orthonormality is preserved even after truncation operations.
@Suite("SVDDecomposition - Unitary Properties")
struct UnitaryPropertiesTests {
    @Test("U has orthonormal columns: U-dagger * U = I")
    func uOrthonormalColumns() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 2), Complex(3, -1)],
            [Complex(-2, 1), Complex(4, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix)

        let uRows = result.u.count
        let uCols = result.u[0].count
        var uDaggerU = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: uCols),
            count: uCols,
        )
        for i in 0 ..< uCols {
            for j in 0 ..< uCols {
                var sum = Complex<Double>.zero
                for k in 0 ..< uRows {
                    sum = sum + result.u[k][i].conjugate * result.u[k][j]
                }
                uDaggerU[i][j] = sum
            }
        }

        for i in 0 ..< uDaggerU.count {
            for j in 0 ..< uDaggerU[i].count {
                let expected = (i == j) ? Complex<Double>.one : Complex<Double>.zero
                let diff = (uDaggerU[i][j] - expected).magnitude
                #expect(
                    diff < 1e-10,
                    "U-dagger * U should be identity. Element [\(i)][\(j)] = \(uDaggerU[i][j]), expected \(expected). U columns must be orthonormal for SVD.",
                )
            }
        }
    }

    @Test("V has orthonormal columns: V-dagger * V = I (or equivalently V * V-dagger = I for stored V-dagger)")
    func vOrthonormalColumns() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(2, 0)],
            [Complex(3, 0), Complex(4, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix)

        let vDaggerRows = result.vDagger.count
        let vDaggerCols = result.vDagger[0].count
        var vDaggerVDaggerDagger = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: vDaggerRows),
            count: vDaggerRows,
        )
        for i in 0 ..< vDaggerRows {
            for j in 0 ..< vDaggerRows {
                var sum = Complex<Double>.zero
                for k in 0 ..< vDaggerCols {
                    sum = sum + result.vDagger[i][k] * result.vDagger[j][k].conjugate
                }
                vDaggerVDaggerDagger[i][j] = sum
            }
        }

        for i in 0 ..< vDaggerVDaggerDagger.count {
            for j in 0 ..< vDaggerVDaggerDagger[i].count {
                let expected = (i == j) ? Complex<Double>.one : Complex<Double>.zero
                let diff = (vDaggerVDaggerDagger[i][j] - expected).magnitude
                #expect(
                    diff < 1e-10,
                    "V-dagger * V = I (rows of V-dagger are orthonormal). Element [\(i)][\(j)] = \(vDaggerVDaggerDagger[i][j]), expected \(expected).",
                )
            }
        }
    }

    @Test("U columns remain orthonormal after truncation")
    func uOrthonormalAfterTruncation() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(2, 0), Complex(3, 0)],
            [Complex(4, 0), Complex(5, 0), Complex(6, 0)],
            [Complex(7, 0), Complex(8, 0), Complex(9, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .maxBondDimension(2))

        let uRows = result.u.count
        let uCols = result.u[0].count
        var uDaggerU = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: uCols),
            count: uCols,
        )
        for i in 0 ..< uCols {
            for j in 0 ..< uCols {
                var sum = Complex<Double>.zero
                for k in 0 ..< uRows {
                    sum = sum + result.u[k][i].conjugate * result.u[k][j]
                }
                uDaggerU[i][j] = sum
            }
        }

        for i in 0 ..< uDaggerU.count {
            for j in 0 ..< uDaggerU[i].count {
                let expected = (i == j) ? Complex<Double>.one : Complex<Double>.zero
                let diff = (uDaggerU[i][j] - expected).magnitude
                #expect(
                    diff < 1e-10,
                    "Truncated U-dagger * U should still be identity. Element [\(i)][\(j)] = \(uDaggerU[i][j]). Orthonormality must be preserved after truncation.",
                )
            }
        }
    }

    @Test("V-dagger rows remain orthonormal after truncation")
    func vDaggerOrthonormalAfterTruncation() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 2), Complex(3, 4), Complex(5, 6)],
            [Complex(7, 8), Complex(9, 10), Complex(11, 12)],
            [Complex(13, 14), Complex(15, 16), Complex(17, 18)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .maxBondDimension(2))

        let vDaggerRows = result.vDagger.count
        let vDaggerCols = result.vDagger[0].count
        var vDaggerVDaggerDagger = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: vDaggerRows),
            count: vDaggerRows,
        )
        for i in 0 ..< vDaggerRows {
            for j in 0 ..< vDaggerRows {
                var sum = Complex<Double>.zero
                for k in 0 ..< vDaggerCols {
                    sum = sum + result.vDagger[i][k] * result.vDagger[j][k].conjugate
                }
                vDaggerVDaggerDagger[i][j] = sum
            }
        }

        for i in 0 ..< vDaggerVDaggerDagger.count {
            for j in 0 ..< vDaggerVDaggerDagger[i].count {
                let expected = (i == j) ? Complex<Double>.one : Complex<Double>.zero
                let diff = (vDaggerVDaggerDagger[i][j] - expected).magnitude
                #expect(
                    diff < 1e-10,
                    "Truncated V-dagger rows should remain orthonormal. Element [\(i)][\(j)] = \(vDaggerVDaggerDagger[i][j]).",
                )
            }
        }
    }
}

/// Test suite for complex-valued matrix SVD.
/// Validates correct handling of complex phases and Hermitian properties.
/// Tests unitary matrices, scaled unitaries, and pure imaginary matrices.
@Suite("SVDDecomposition - Complex Matrix Properties")
struct ComplexMatrixPropertiesTests {
    @Test("Pure imaginary matrix decomposition")
    func pureImaginaryMatrix() {
        let matrix: [[Complex<Double>]] = [
            [Complex(0, 1), Complex(0, 2)],
            [Complex(0, 3), Complex(0, 4)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix)

        let m = result.u.count
        let k = result.singularValues.count
        let n = result.vDagger[0].count
        var reconstructed = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: n),
            count: m,
        )
        for i in 0 ..< m {
            for j in 0 ..< n {
                var sum = Complex<Double>.zero
                for l in 0 ..< k {
                    sum = sum + result.u[i][l] * result.singularValues[l] * result.vDagger[l][j]
                }
                reconstructed[i][j] = sum
            }
        }

        for row in 0 ..< matrix.count {
            for col in 0 ..< matrix[0].count {
                let diff = (reconstructed[row][col] - matrix[row][col]).magnitude
                #expect(
                    diff < 1e-10,
                    "Pure imaginary matrix reconstruction failed at [\(row)][\(col)]. SVD must handle complex phases correctly.",
                )
            }
        }
    }

    @Test("Unitary matrix has all singular values equal to 1")
    func unitaryMatrixSingularValues() {
        let theta = Double.pi / 4
        let cosT = cos(theta)
        let sinT = sin(theta)
        let matrix: [[Complex<Double>]] = [
            [Complex(cosT, 0), Complex(-sinT, 0)],
            [Complex(sinT, 0), Complex(cosT, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix)

        for (index, sv) in result.singularValues.enumerated() {
            #expect(
                abs(sv - 1.0) < 1e-10,
                "Unitary matrix should have all singular values = 1 (preserves norm). Got s[\(index)] = \(sv).",
            )
        }
    }

    @Test("Scaled unitary matrix singular values equal scale factor")
    func scaledUnitaryMatrixSingularValues() {
        let scale = 3.0
        let theta = Double.pi / 6
        let cosT = cos(theta)
        let sinT = sin(theta)
        let matrix: [[Complex<Double>]] = [
            [Complex(scale * cosT, 0), Complex(-scale * sinT, 0)],
            [Complex(scale * sinT, 0), Complex(scale * cosT, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix)

        for (index, sv) in result.singularValues.enumerated() {
            #expect(
                abs(sv - scale) < 1e-10,
                "Scaled unitary matrix should have all singular values = scale (\(scale)). Got s[\(index)] = \(sv).",
            )
        }
    }
}

/// Test suite for edge case guard branches in truncation strategies.
/// Validates behavior when singular values are all zero (degenerate matrices).
/// Ensures correct handling of zero-weight matrices in truncation algorithms.
@Suite("SVDDecomposition - Zero Matrix Truncation Edge Cases")
struct ZeroMatrixTruncationEdgeCasesTests {
    @Test("Relative threshold with zero matrix returns all")
    func relativeThresholdZeroMatrix() {
        let zeroMatrix: [[Complex<Double>]] = [
            [.zero, .zero],
            [.zero, .zero],
        ]
        let result = SVDDecomposition.decompose(matrix: zeroMatrix, truncation: .relativeThreshold(0.5))
        #expect(result.singularValues.allSatisfy { $0 < 1e-10 }, "Zero matrix should have zero singular values")
    }

    @Test("Cumulative weight with zero matrix returns all")
    func cumulativeWeightZeroMatrix() {
        let zeroMatrix: [[Complex<Double>]] = [
            [.zero, .zero],
            [.zero, .zero],
        ]
        let result = SVDDecomposition.decompose(matrix: zeroMatrix, truncation: .cumulativeWeight(epsilon: 0.1))
        #expect(result.singularValues.allSatisfy { $0 < 1e-10 }, "Zero matrix should have zero singular values")
    }

    @Test("Relative threshold zero matrix keeps all singular values")
    func relativeThresholdZeroMatrixKeepsAll() {
        let zeroMatrix: [[Complex<Double>]] = [
            [.zero, .zero, .zero],
            [.zero, .zero, .zero],
        ]
        let result = SVDDecomposition.decompose(matrix: zeroMatrix, truncation: .relativeThreshold(0.9))
        #expect(
            result.singularValues.count == 2,
            "Zero 2x3 matrix with relative threshold should keep min(2,3) = 2 singular values, got \(result.singularValues.count)",
        )
    }

    @Test("Cumulative weight zero matrix keeps all singular values")
    func cumulativeWeightZeroMatrixKeepsAll() {
        let zeroMatrix: [[Complex<Double>]] = [
            [.zero, .zero, .zero],
            [.zero, .zero, .zero],
            [.zero, .zero, .zero],
        ]
        let result = SVDDecomposition.decompose(matrix: zeroMatrix, truncation: .cumulativeWeight(epsilon: 0.5))
        #expect(
            result.singularValues.count >= 1,
            "Zero matrix with cumulative weight should keep at least 1 singular value, got \(result.singularValues.count)",
        )
    }

    @Test("Near-zero matrix relative threshold behavior")
    func nearZeroMatrixRelativeThreshold() {
        let nearZero: [[Complex<Double>]] = [
            [Complex(1e-15, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(1e-16, 0)],
        ]
        let result = SVDDecomposition.decompose(matrix: nearZero, truncation: .relativeThreshold(0.5))
        #expect(
            result.singularValues.count >= 1,
            "Near-zero matrix should keep at least 1 singular value",
        )
    }

    @Test("Near-zero matrix cumulative weight behavior")
    func nearZeroMatrixCumulativeWeight() {
        let nearZero: [[Complex<Double>]] = [
            [Complex(1e-15, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(1e-16, 0)],
        ]
        let result = SVDDecomposition.decompose(matrix: nearZero, truncation: .cumulativeWeight(epsilon: 0.1))
        #expect(
            result.singularValues.count >= 1,
            "Near-zero matrix with cumulative weight should keep at least 1 singular value",
        )
    }
}

/// Test suite for SVDResult structure.
/// Validates that the result structure contains correctly shaped arrays.
/// Ensures dimensional consistency between U, singular values, and V-dagger.
@Suite("SVDDecomposition - Result Structure")
struct SVDResultStructureTests {
    @Test("SVDResult contains non-empty arrays")
    func resultContainsNonEmptyArrays() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(2, 0)],
            [Complex(3, 0), Complex(4, 0)],
        ]

        let result = SVDDecomposition.decompose(matrix: matrix)

        #expect(!result.u.isEmpty, "U matrix should not be empty.")
        #expect(!result.singularValues.isEmpty, "Singular values should not be empty.")
        #expect(!result.vDagger.isEmpty, "V-dagger matrix should not be empty.")
    }

    @Test("SVDResult has consistent dimensions")
    func resultHasConsistentDimensions() {
        let m = 4
        let n = 3
        var matrix = [[Complex<Double>]]()
        for _ in 0 ..< m {
            var row = [Complex<Double>]()
            for _ in 0 ..< n {
                row.append(Complex(Double.random(in: -1 ... 1), Double.random(in: -1 ... 1)))
            }
            matrix.append(row)
        }

        let result = SVDDecomposition.decompose(matrix: matrix)
        let k = result.singularValues.count

        #expect(
            result.u.count == m,
            "U should have m=\(m) rows. Got \(result.u.count).",
        )
        #expect(
            result.u[0].count == k,
            "U should have k=\(k) columns (number of kept singular values). Got \(result.u[0].count).",
        )
        #expect(
            result.vDagger.count == k,
            "V-dagger should have k=\(k) rows. Got \(result.vDagger.count).",
        )
        #expect(
            result.vDagger[0].count == n,
            "V-dagger should have n=\(n) columns. Got \(result.vDagger[0].count).",
        )
    }
}
