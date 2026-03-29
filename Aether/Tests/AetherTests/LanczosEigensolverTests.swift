// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Thread-safe call counter for verifying operator invocation counts.
private actor Counter {
    private(set) var value: Int = 0
    func increment() -> Int {
        value += 1
        return value
    }
}

/// Test suite for Lanczos eigensolver on diagonal matrices.
/// Validates that findLowest correctly identifies eigenvalues of diagonal matrices,
/// where eigenvalues are the diagonal entries and eigenvectors are standard basis vectors.
@Suite("LanczosEigensolver - Diagonal Matrix Tests")
struct LanczosDiagonalMatrixTests {
    @Test("findLowest on 4x4 diagonal matrix returns smallest diagonal entry")
    func findLowestDiagonal4x4() async {
        let dimension = 4
        let diagonalValues = [3.0, 1.0, 4.0, 2.0]
        let expectedLowest = 1.0

        let result = await LanczosEigensolver.findLowest(
            applying: { vector in
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    output[i] = Complex(diagonalValues[i], 0) * vector[i]
                }
                return output
            },
            dimension: dimension,
            tolerance: 1e-10,
        )

        let eigenvalueDiff = abs(result.eigenvalues[0] - expectedLowest)
        #expect(
            eigenvalueDiff < 1e-10,
            "Lowest eigenvalue should be 1.0 (smallest diagonal entry), got \(result.eigenvalues[0]). Difference: \(eigenvalueDiff).",
        )
    }

    @Test("findLowest on 10x10 diagonal matrix with sequential values")
    func findLowestDiagonal10x10() async {
        let dimension = 10
        let diagonalValues = (1 ... dimension).map { Double($0) }
        let expectedLowest = 1.0

        let result = await LanczosEigensolver.findLowest(
            applying: { vector in
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    output[i] = Complex(diagonalValues[i], 0) * vector[i]
                }
                return output
            },
            dimension: dimension,
            tolerance: 1e-10,
        )

        let eigenvalueDiff = abs(result.eigenvalues[0] - expectedLowest)
        #expect(
            eigenvalueDiff < 1e-10,
            "Lowest eigenvalue should be 1.0 for diagonal matrix with entries 1..10, got \(result.eigenvalues[0]).",
        )
    }

    @Test("findLowest on diagonal matrix with negative eigenvalues")
    func findLowestDiagonalNegative() async {
        let dimension = 5
        let diagonalValues = [2.0, -3.0, 1.0, -1.0, 0.5]
        let expectedLowest = -3.0

        let result = await LanczosEigensolver.findLowest(
            applying: { vector in
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    output[i] = Complex(diagonalValues[i], 0) * vector[i]
                }
                return output
            },
            dimension: dimension,
            tolerance: 1e-10,
        )

        let eigenvalueDiff = abs(result.eigenvalues[0] - expectedLowest)
        #expect(
            eigenvalueDiff < 1e-10,
            "Lowest eigenvalue should be -3.0 (most negative), got \(result.eigenvalues[0]).",
        )
    }

    @Test("findLowest eigenvector is normalized")
    func findLowestEigenvectorNormalized() async {
        let dimension = 6
        let diagonalValues = [5.0, 2.0, 8.0, 1.0, 3.0, 7.0]

        let result = await LanczosEigensolver.findLowest(
            applying: { vector in
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    output[i] = Complex(diagonalValues[i], 0) * vector[i]
                }
                return output
            },
            dimension: dimension,
            tolerance: 1e-10,
        )

        var normSquared = 0.0
        for element in result.eigenvectors[0] {
            normSquared += element.magnitudeSquared
        }
        let norm = sqrt(normSquared)

        let normDiff = abs(norm - 1.0)
        #expect(
            normDiff < 1e-10,
            "Eigenvector should be normalized to unit length. Got norm = \(norm).",
        )
    }
}

/// Test suite for Lanczos eigensolver on Hermitian matrices.
/// Validates convergence to correct lowest eigenvalue for general Hermitian operators,
/// comparing results against HermitianEigenDecomposition for verification.
@Suite("LanczosEigensolver - Hermitian Matrix Tests")
struct LanczosHermitianMatrixTests {
    @Test("findLowest on 3x3 Hermitian matrix matches direct decomposition")
    func findLowest3x3Hermitian() async {
        var matrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: 6),
            count: 6,
        )
        for i in 0 ..< 6 {
            matrix[i][i] = Complex(Double(i) + 1.0, 0)
            if i < 5 {
                matrix[i][i + 1] = Complex(0.4, 0.2)
                matrix[i + 1][i] = Complex(0.4, -0.2)
            }
        }

        let directResult = HermitianEigenDecomposition.decompose(matrix: matrix)
        let expectedLowest = directResult.eigenvalues[0]

        let dimension = 6
        let matrixCopy = matrix
        let result = await LanczosEigensolver.findLowest(
            applying: { vector in
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    for j in 0 ..< dimension {
                        output[i] = output[i] + matrixCopy[i][j] * vector[j]
                    }
                }
                return output
            },
            dimension: dimension,
            tolerance: 1e-10,
        )

        let eigenvalueDiff = abs(result.eigenvalues[0] - expectedLowest)
        #expect(
            eigenvalueDiff < 1e-10,
            "Lanczos eigenvalue \(result.eigenvalues[0]) should match direct decomposition \(expectedLowest). Difference: \(eigenvalueDiff).",
        )
    }

    @Test("findLowest on 5x5 Hermitian matrix matches direct decomposition")
    func findLowest5x5Hermitian() async {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(0.5, 0.2), Complex(0, 0), Complex(0.1, -0.1), Complex(0, 0)],
            [Complex(0.5, -0.2), Complex(2, 0), Complex(0.3, 0.1), Complex(0, 0), Complex(0.2, 0)],
            [Complex(0, 0), Complex(0.3, -0.1), Complex(0.5, 0), Complex(0.4, 0.3), Complex(0, 0)],
            [Complex(0.1, 0.1), Complex(0, 0), Complex(0.4, -0.3), Complex(1.5, 0), Complex(0.1, 0.2)],
            [Complex(0, 0), Complex(0.2, 0), Complex(0, 0), Complex(0.1, -0.2), Complex(3, 0)],
        ]

        let directResult = HermitianEigenDecomposition.decompose(matrix: matrix)
        let expectedLowest = directResult.eigenvalues[0]

        let dimension = 5
        let result = await LanczosEigensolver.findLowest(
            applying: { vector in
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    for j in 0 ..< dimension {
                        output[i] = output[i] + matrix[i][j] * vector[j]
                    }
                }
                return output
            },
            dimension: dimension,
            tolerance: 1e-10,
        )

        let eigenvalueDiff = abs(result.eigenvalues[0] - expectedLowest)
        #expect(
            eigenvalueDiff < 1e-10,
            "Lanczos eigenvalue \(result.eigenvalues[0]) should match direct decomposition \(expectedLowest). Difference: \(eigenvalueDiff).",
        )
    }

    @Test("findLowest on diagonal 6x6 matrix returns lowest eigenvalue")
    func findLowestDiagonal6x6() async {
        let dimension = 6
        let diagonalValues = [3.0, -1.0, 4.0, 1.0, 5.0, 2.0]

        var matrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: dimension),
            count: dimension,
        )
        for i in 0 ..< dimension {
            matrix[i][i] = Complex(diagonalValues[i], 0)
        }

        let directResult = HermitianEigenDecomposition.decompose(matrix: matrix)
        let expectedLowest = directResult.eigenvalues[0]

        let matrixCopy = matrix
        let result = await LanczosEigensolver.findLowest(
            applying: { vector in
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    for j in 0 ..< dimension {
                        output[i] = output[i] + matrixCopy[i][j] * vector[j]
                    }
                }
                return output
            },
            dimension: dimension,
            tolerance: 1e-10,
        )

        let eigenvalueDiff = abs(result.eigenvalues[0] - expectedLowest)
        #expect(
            eigenvalueDiff < 1e-10,
            "Lanczos eigenvalue \(result.eigenvalues[0]) should match direct decomposition \(expectedLowest). Difference: \(eigenvalueDiff).",
        )
    }

    @Test("findLowest on 8x8 Hermitian matrix matches direct decomposition")
    func findLowest8x8Hermitian() async {
        var matrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: 8),
            count: 8,
        )

        for i in 0 ..< 8 {
            matrix[i][i] = Complex(Double(i) + 1.0, 0)
            if i < 7 {
                matrix[i][i + 1] = Complex(0.5, 0.1)
                matrix[i + 1][i] = Complex(0.5, -0.1)
            }
        }

        let directResult = HermitianEigenDecomposition.decompose(matrix: matrix)
        let expectedLowest = directResult.eigenvalues[0]

        let dimension = 8
        let matrixCopy = matrix
        let result = await LanczosEigensolver.findLowest(
            applying: { vector in
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    for j in 0 ..< dimension {
                        output[i] = output[i] + matrixCopy[i][j] * vector[j]
                    }
                }
                return output
            },
            dimension: dimension,
            tolerance: 1e-10,
        )

        let eigenvalueDiff = abs(result.eigenvalues[0] - expectedLowest)
        #expect(
            eigenvalueDiff < 1e-10,
            "Lanczos eigenvalue \(result.eigenvalues[0]) should match direct \(expectedLowest). Difference: \(eigenvalueDiff).",
        )
    }
}

/// Test suite for Lanczos convergence behavior.
/// Validates that the algorithm converges to correct eigenvalues within tolerance,
/// and that eigenvector satisfies H|v> = lambda|v> relationship.
@Suite("LanczosEigensolver - Convergence Tests")
struct LanczosConvergenceTests {
    @Test("Convergence to lowest eigenvalue within tolerance")
    func convergenceWithinTolerance() async {
        let dimension = 12
        let tolerance = 1e-10

        var matrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: dimension),
            count: dimension,
        )
        for i in 0 ..< dimension {
            matrix[i][i] = Complex(Double(i) * 0.5 + 0.1, 0)
            if i < dimension - 1 {
                matrix[i][i + 1] = Complex(0.2, 0)
                matrix[i + 1][i] = Complex(0.2, 0)
            }
        }

        let directResult = HermitianEigenDecomposition.decompose(matrix: matrix)
        let expectedLowest = directResult.eigenvalues[0]

        let matrixCopy = matrix
        let result = await LanczosEigensolver.findLowest(
            applying: { vector in
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    for j in 0 ..< dimension {
                        output[i] = output[i] + matrixCopy[i][j] * vector[j]
                    }
                }
                return output
            },
            dimension: dimension,
            tolerance: tolerance,
        )

        let eigenvalueDiff = abs(result.eigenvalues[0] - expectedLowest)
        #expect(
            eigenvalueDiff < tolerance * 10,
            "Lanczos should converge within tolerance. Expected \(expectedLowest), got \(result.eigenvalues[0]). Difference: \(eigenvalueDiff).",
        )
    }

    @Test("Eigenvector satisfies eigenvalue equation H|v> = lambda|v>")
    func eigenvectorSatisfiesEquation() async {
        let dimension = 6
        var matrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: dimension),
            count: dimension,
        )
        for i in 0 ..< dimension {
            matrix[i][i] = Complex(Double(i + 1), 0)
            if i < dimension - 1 {
                matrix[i][i + 1] = Complex(0.3, 0.1)
                matrix[i + 1][i] = Complex(0.3, -0.1)
            }
        }

        let matrixCopy = matrix
        let result = await LanczosEigensolver.findLowest(
            applying: { vector in
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    for j in 0 ..< dimension {
                        output[i] = output[i] + matrixCopy[i][j] * vector[j]
                    }
                }
                return output
            },
            dimension: dimension,
            tolerance: 1e-6,
        )

        let eigenvalue = result.eigenvalues[0]
        let eigenvector = result.eigenvectors[0]

        var hTimesV = [Complex<Double>](repeating: .zero, count: dimension)
        for i in 0 ..< dimension {
            for j in 0 ..< dimension {
                hTimesV[i] = hTimesV[i] + matrix[i][j] * eigenvector[j]
            }
        }

        var maxResidual = 0.0
        for i in 0 ..< dimension {
            let expected = Complex(eigenvalue, 0) * eigenvector[i]
            let diff = (hTimesV[i] - expected).magnitude
            maxResidual = max(maxResidual, diff)
        }

        #expect(
            maxResidual < 1e-3,
            "Eigenvector should satisfy H|v> = lambda|v>. Max residual: \(maxResidual).",
        )
    }

    @Test("Eigenvalue is real for Hermitian matrix")
    func eigenvalueIsReal() async {
        let dimension = 4
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(0, 1), Complex(0, 0), Complex(0.5, 0)],
            [Complex(0, -1), Complex(2, 0), Complex(0, 0.5), Complex(0, 0)],
            [Complex(0, 0), Complex(0, -0.5), Complex(3, 0), Complex(0, 1)],
            [Complex(0.5, 0), Complex(0, 0), Complex(0, -1), Complex(4, 0)],
        ]

        let result = await LanczosEigensolver.findLowest(
            applying: { vector in
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    for j in 0 ..< dimension {
                        output[i] = output[i] + matrix[i][j] * vector[j]
                    }
                }
                return output
            },
            dimension: dimension,
            tolerance: 1e-10,
        )

        let eigenvalue = result.eigenvalues[0]
        #expect(
            eigenvalue.isFinite,
            "Eigenvalue should be a finite real number, got \(eigenvalue).",
        )
    }

    @Test("Multiple runs converge to same eigenvalue")
    func multipleRunsConvergeToSame() async {
        let dimension = 5
        let matrix: [[Complex<Double>]] = [
            [Complex(2, 0), Complex(0.5, 0), Complex(0, 0), Complex(0, 0), Complex(0, 0)],
            [Complex(0.5, 0), Complex(3, 0), Complex(0.5, 0), Complex(0, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(0.5, 0), Complex(1, 0), Complex(0.5, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(0, 0), Complex(0.5, 0), Complex(4, 0), Complex(0.5, 0)],
            [Complex(0, 0), Complex(0, 0), Complex(0, 0), Complex(0.5, 0), Complex(5, 0)],
        ]

        let result1 = await LanczosEigensolver.findLowest(
            applying: { vector in
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    for j in 0 ..< dimension {
                        output[i] = output[i] + matrix[i][j] * vector[j]
                    }
                }
                return output
            },
            dimension: dimension,
            tolerance: 1e-10,
        )

        let result2 = await LanczosEigensolver.findLowest(
            applying: { vector in
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    for j in 0 ..< dimension {
                        output[i] = output[i] + matrix[i][j] * vector[j]
                    }
                }
                return output
            },
            dimension: dimension,
            tolerance: 1e-10,
        )

        let eigenvalueDiff = abs(result1.eigenvalues[0] - result2.eigenvalues[0])
        #expect(
            eigenvalueDiff < 1e-10,
            "Multiple runs should converge to same eigenvalue. Run 1: \(result1.eigenvalues[0]), Run 2: \(result2.eigenvalues[0]).",
        )
    }
}

/// Test suite for Lanczos comparison with HermitianEigenDecomposition.
/// Validates that Lanczos produces equivalent results to direct diagonalization,
/// ensuring correctness across various matrix types and sizes.
@Suite("LanczosEigensolver - Direct Comparison Tests")
struct LanczosDirectComparisonTests {
    @Test("Compare with direct decomposition on tridiagonal matrix")
    func compareTridiagonal() async {
        let dimension = 10
        var matrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: dimension),
            count: dimension,
        )
        for i in 0 ..< dimension {
            matrix[i][i] = Complex(2.0, 0)
            if i > 0 {
                matrix[i][i - 1] = Complex(-1.0, 0)
            }
            if i < dimension - 1 {
                matrix[i][i + 1] = Complex(-1.0, 0)
            }
        }

        let directResult = HermitianEigenDecomposition.decompose(matrix: matrix)

        let matrixCopy = matrix
        let lanczosResult = await LanczosEigensolver.findLowest(
            applying: { vector in
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    for j in 0 ..< dimension {
                        output[i] = output[i] + matrixCopy[i][j] * vector[j]
                    }
                }
                return output
            },
            dimension: dimension,
            tolerance: 1e-12,
        )

        let eigenvalueDiff = abs(lanczosResult.eigenvalues[0] - directResult.eigenvalues[0])
        #expect(
            eigenvalueDiff < 1e-10,
            "Lanczos and direct should match on tridiagonal. Lanczos: \(lanczosResult.eigenvalues[0]), Direct: \(directResult.eigenvalues[0]).",
        )
    }

    @Test("Compare with direct decomposition on random Hermitian matrix")
    func compareRandomHermitian() async {
        let dimension = 8
        var matrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: dimension),
            count: dimension,
        )

        var seed: UInt64 = 54321
        for i in 0 ..< dimension {
            matrix[i][i] = Complex(Double(i + 1) * 0.5, 0)
            for j in (i + 1) ..< dimension {
                seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                let real = Double(Int64(bitPattern: seed)) / Double(Int64.max) * 0.3
                seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                let imag = Double(Int64(bitPattern: seed)) / Double(Int64.max) * 0.3
                matrix[i][j] = Complex(real, imag)
                matrix[j][i] = Complex(real, -imag)
            }
        }

        let directResult = HermitianEigenDecomposition.decompose(matrix: matrix)

        let matrixCopy = matrix
        let lanczosResult = await LanczosEigensolver.findLowest(
            applying: { vector in
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    for j in 0 ..< dimension {
                        output[i] = output[i] + matrixCopy[i][j] * vector[j]
                    }
                }
                return output
            },
            dimension: dimension,
            tolerance: 1e-12,
        )

        let eigenvalueDiff = abs(lanczosResult.eigenvalues[0] - directResult.eigenvalues[0])
        #expect(
            eigenvalueDiff < 1e-10,
            "Lanczos should match direct on random Hermitian. Lanczos: \(lanczosResult.eigenvalues[0]), Direct: \(directResult.eigenvalues[0]).",
        )
    }

    @Test("Compare eigenvector overlap with direct decomposition")
    func compareEigenvectorOverlap() async {
        let dimension = 6
        var matrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: dimension),
            count: dimension,
        )
        for i in 0 ..< dimension {
            matrix[i][i] = Complex(Double(i) + 1.0, 0)
            if i < dimension - 1 {
                matrix[i][i + 1] = Complex(0.25, 0)
                matrix[i + 1][i] = Complex(0.25, 0)
            }
        }

        let directResult = HermitianEigenDecomposition.decompose(matrix: matrix)
        let directEigenvector = directResult.eigenvectors[0]

        let matrixCopy = matrix
        let lanczosResult = await LanczosEigensolver.findLowest(
            applying: { vector in
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    for j in 0 ..< dimension {
                        output[i] = output[i] + matrixCopy[i][j] * vector[j]
                    }
                }
                return output
            },
            dimension: dimension,
            tolerance: 1e-12,
        )
        let lanczosEigenvector = lanczosResult.eigenvectors[0]

        var overlap = Complex<Double>.zero
        for i in 0 ..< dimension {
            overlap = overlap + directEigenvector[i].conjugate * lanczosEigenvector[i]
        }
        let overlapMagnitude = overlap.magnitude

        #expect(
            abs(overlapMagnitude - 1.0) < 1e-10,
            "Eigenvector overlap should be ~1.0 (vectors parallel). Got overlap magnitude: \(overlapMagnitude).",
        )
    }

    @Test("Compare on 15x15 banded Hermitian matrix")
    func compare15x15Banded() async {
        let dimension = 15
        var matrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: dimension),
            count: dimension,
        )
        for i in 0 ..< dimension {
            matrix[i][i] = Complex(Double(i % 5) + 1.0, 0)
            if i < dimension - 1 {
                matrix[i][i + 1] = Complex(0.4, 0.1)
                matrix[i + 1][i] = Complex(0.4, -0.1)
            }
            if i < dimension - 2 {
                matrix[i][i + 2] = Complex(0.1, 0)
                matrix[i + 2][i] = Complex(0.1, 0)
            }
        }

        let directResult = HermitianEigenDecomposition.decompose(matrix: matrix)

        let matrixCopy = matrix
        let lanczosResult = await LanczosEigensolver.findLowest(
            applying: { vector in
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    for j in 0 ..< dimension {
                        output[i] = output[i] + matrixCopy[i][j] * vector[j]
                    }
                }
                return output
            },
            dimension: dimension,
            tolerance: 1e-12,
        )

        let eigenvalueDiff = abs(lanczosResult.eigenvalues[0] - directResult.eigenvalues[0])
        #expect(
            eigenvalueDiff < 1e-10,
            "Lanczos should match direct on 15x15 banded matrix. Lanczos: \(lanczosResult.eigenvalues[0]), Direct: \(directResult.eigenvalues[0]).",
        )
    }

    @Test("Compare on 20x20 Hermitian matrix")
    func compare20x20() async {
        let dimension = 20
        var matrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: dimension),
            count: dimension,
        )
        for i in 0 ..< dimension {
            matrix[i][i] = Complex(Double(i) * 0.3 + 0.5, 0)
            if i < dimension - 1 {
                matrix[i][i + 1] = Complex(0.15, 0.05)
                matrix[i + 1][i] = Complex(0.15, -0.05)
            }
        }

        let directResult = HermitianEigenDecomposition.decompose(matrix: matrix)

        let matrixCopy = matrix
        let lanczosResult = await LanczosEigensolver.findLowest(
            applying: { vector in
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    for j in 0 ..< dimension {
                        output[i] = output[i] + matrixCopy[i][j] * vector[j]
                    }
                }
                return output
            },
            dimension: dimension,
            tolerance: 1e-12,
        )

        let eigenvalueDiff = abs(lanczosResult.eigenvalues[0] - directResult.eigenvalues[0])
        #expect(
            eigenvalueDiff < 1e-10,
            "Lanczos should match direct on 20x20 matrix. Lanczos: \(lanczosResult.eigenvalues[0]), Direct: \(directResult.eigenvalues[0]).",
        )
    }
}

/// Test suite for Lanczos edge cases and special code paths.
/// Validates invariant subspace breakdown, max restart exhaustion,
/// and uniform eigenvalue matrices.
@Suite("LanczosEigensolver - Edge Cases")
struct LanczosEdgeCasesTests {
    @Test("Uniform diagonal matrix triggers invariant subspace breakdown")
    func invariantSubspaceBreakdown() async {
        let dimension = 35
        let eigenvalue = 2.0

        let result = await LanczosEigensolver.findLowest(
            applying: { vector in
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    output[i] = Complex(eigenvalue, 0) * vector[i]
                }
                return output
            },
            dimension: dimension,
            tolerance: 1e-10,
        )

        let eigenvalueDiff = abs(result.eigenvalues[0] - eigenvalue)
        #expect(eigenvalueDiff < 1e-10, "Uniform diagonal should find eigenvalue \(eigenvalue), got \(result.eigenvalues[0])")
    }

    @Test("Max restarts exhaustion with impossibly tight tolerance on banded matrix")
    func maxRestartsExhaustion() async {
        // Exercises the fallback return path (lines 117-121) that runs when
        // the Lanczos restart loop completes without converging.
        //
        // Strategy: dimension > 30 avoids solveDirect. A tridiagonal Hermitian
        // matrix with strong coupling prevents invariant-subspace breakdown.
        // A captured call counter adds a vanishing perturbation (~1e-15) that
        // changes the operator very slightly across Krylov steps, ensuring the
        // eigenvalue estimate never stabilizes to the exact same Double and
        // |lambda_new - lambda_old| always exceeds Double.leastNonzeroMagnitude.
        let dimension = 40
        let counter = Counter()

        let result = await LanczosEigensolver.findLowest(
            applying: { vector in
                let n = await counter.increment()
                var output = [Complex<Double>](repeating: .zero, count: dimension)
                for i in 0 ..< dimension {
                    // Diagonal with a tiny perturbation that shifts each call
                    let perturbation = 1.0e-15 * Double(n) * Double(i + 1)
                    output[i] = Complex(Double(i + 1) + perturbation, 0) * vector[i]
                    // Nearest-neighbor coupling keeps beta above breakdown threshold
                    if i > 0 {
                        output[i] = output[i] + Complex(0.5, 0) * vector[i - 1]
                    }
                    if i < dimension - 1 {
                        output[i] = output[i] + Complex(0.5, 0) * vector[i + 1]
                    }
                }
                return output
            },
            dimension: dimension,
            tolerance: Double.leastNonzeroMagnitude,
        )

        #expect(result.eigenvalues[0].isFinite, "Should return finite eigenvalue after exhausting restarts")
        #expect(!result.eigenvectors[0].isEmpty, "Should return non-empty eigenvector after exhausting restarts")
        // Verify it actually went through many iterations (at least 100 restarts * 30 steps)
        let totalCalls = await counter.value
        #expect(totalCalls > 2000, "Should have made many apply calls, got \(totalCalls)")
    }
}
