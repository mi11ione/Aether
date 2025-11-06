// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation

/// Errors that can occur during matrix operations
enum MatrixError: Error {
    case incompatibleDimensions
    case emptyMatrix
    case nonSquareMatrix
}

/// Shared utilities for quantum computing tests.
/// Provides matrix operations and validation functions used across test files.
enum TestUtilities {
    /// Multiply two complex matrices with validation
    /// - Parameters:
    ///   - a: Left matrix
    ///   - b: Right matrix
    /// - Returns: Product matrix
    /// - Throws: MatrixError if dimensions are incompatible
    static func matrixMultiply(
        _ a: [[Complex<Double>]],
        _ b: [[Complex<Double>]]
    ) throws -> [[Complex<Double>]] {
        guard !a.isEmpty, !b.isEmpty else {
            throw MatrixError.emptyMatrix
        }

        let rowsA = a.count
        let colsA = a[0].count
        let rowsB = b.count
        let colsB = b[0].count

        guard a.allSatisfy({ $0.count == colsA }),
              b.allSatisfy({ $0.count == colsB })
        else {
            throw MatrixError.incompatibleDimensions
        }

        guard colsA == rowsB else {
            throw MatrixError.incompatibleDimensions
        }

        var result = Array(repeating: Array(repeating: Complex<Double>.zero, count: colsB), count: rowsA)

        for i in 0 ..< rowsA {
            for j in 0 ..< colsB {
                var sum = Complex<Double>.zero
                for k in 0 ..< colsA {
                    sum = sum + (a[i][k] * b[k][j])
                }
                result[i][j] = sum
            }
        }

        return result
    }

    /// Compare two matrices for equality within tolerance
    /// - Parameters:
    ///   - a: First matrix
    ///   - b: Second matrix
    ///   - tolerance: Maximum allowed difference (default: 1e-10)
    /// - Returns: True if matrices are equal within tolerance
    static func matricesEqual(
        _ a: [[Complex<Double>]],
        _ b: [[Complex<Double>]],
        tolerance: Double = 1e-10
    ) -> Bool {
        guard a.count == b.count else { return false }

        for i in 0 ..< a.count {
            guard a[i].count == b[i].count else { return false }
            for j in 0 ..< a[i].count {
                let diffReal = abs(a[i][j].real - b[i][j].real)
                let diffImag = abs(a[i][j].imaginary - b[i][j].imaginary)

                if diffReal > tolerance || diffImag > tolerance {
                    return false
                }
            }
        }

        return true
    }

    /// Check if matrix is the identity matrix within tolerance
    /// - Parameters:
    ///   - matrix: Matrix to check
    ///   - tolerance: Maximum allowed difference from identity (default: 1e-10)
    /// - Returns: True if matrix is identity within tolerance
    static func isIdentityMatrix(
        _ matrix: [[Complex<Double>]],
        tolerance: Double = 1e-10
    ) -> Bool {
        guard !matrix.isEmpty else { return false }

        let n = matrix.count

        guard matrix.allSatisfy({ $0.count == n }) else {
            return false
        }

        for i in 0 ..< n {
            for j in 0 ..< n {
                let expected: Complex<Double> = (i == j) ? .one : .zero
                let actual = matrix[i][j]

                let diffReal = abs(actual.real - expected.real)
                let diffImag = abs(actual.imaginary - expected.imaginary)

                if diffReal > tolerance || diffImag > tolerance {
                    return false
                }
            }
        }

        return true
    }

    /// Create identity matrix of given size
    /// - Parameter size: Matrix dimension
    /// - Returns: Identity matrix
    static func identityMatrix(size: Int) -> [[Complex<Double>]] {
        var matrix = Array(repeating: Array(repeating: Complex<Double>.zero, count: size), count: size)
        for i in 0 ..< size {
            matrix[i][i] = .one
        }
        return matrix
    }

    /// Get matrix dimensions
    /// - Parameter matrix: Matrix to analyze
    /// - Returns: Tuple of (rows, columns)
    static func matrixDimensions(_ matrix: [[Complex<Double>]]) -> (rows: Int, cols: Int)? {
        guard !matrix.isEmpty else { return nil }
        let rows = matrix.count
        let cols = matrix[0].count
        guard matrix.allSatisfy({ $0.count == cols }) else { return nil }
        return (rows, cols)
    }

    /// Check if matrix is unitary within tolerance
    /// Unitary matrices satisfy U†U = I
    /// - Parameters:
    ///   - matrix: Matrix to check
    ///   - tolerance: Tolerance for comparisons
    /// - Returns: True if matrix is unitary
    static func isUnitary(
        _ matrix: [[Complex<Double>]],
        tolerance: Double = 1e-10
    ) -> Bool {
        guard let dims = matrixDimensions(matrix), dims.rows == dims.cols else {
            return false
        }

        do {
            let conjugateTranspose = conjugateTranspose(matrix)
            let product = try matrixMultiply(conjugateTranspose, matrix)
            return isIdentityMatrix(product, tolerance: tolerance)
        } catch {
            return false
        }
    }

    /// Compute conjugate transpose of matrix (U†)
    /// - Parameter matrix: Input matrix
    /// - Returns: Conjugate transpose
    private static func conjugateTranspose(_ matrix: [[Complex<Double>]]) -> [[Complex<Double>]] {
        guard let dims = matrixDimensions(matrix) else {
            return []
        }

        var result = Array(repeating: Array(repeating: Complex<Double>.zero, count: dims.rows), count: dims.cols)

        for i in 0 ..< dims.rows {
            for j in 0 ..< dims.cols {
                result[j][i] = matrix[i][j].conjugate
            }
        }

        return result
    }
}
