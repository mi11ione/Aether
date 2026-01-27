// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Result of Hermitian matrix eigendecomposition containing eigenvalues and eigenvectors
///
/// Stores the spectral decomposition H = V * diag(lambda) * V-dagger where V contains the
/// orthonormal eigenvectors and lambda contains the real eigenvalues in ascending order.
/// For a Hermitian matrix all eigenvalues are guaranteed to be real.
///
/// **Example:**
/// ```swift
/// let matrix: [[Complex<Double>]] = [
///     [Complex(2, 0), Complex(0, -1)],
///     [Complex(0, 1), Complex(3, 0)]
/// ]
/// let result = HermitianEigenDecomposition.decompose(matrix: matrix)
/// let values = result.eigenvalues
/// let vectors = result.eigenvectors
/// ```
@frozen public struct HermitianEigenResult: Sendable {
    /// Real eigenvalues in ascending order (n elements for an n x n matrix)
    public let eigenvalues: [Double]

    /// Orthonormal eigenvectors where eigenvectors[i] is the i-th eigenvector corresponding to eigenvalues[i]
    public let eigenvectors: [[Complex<Double>]]

    /// Dimension of the decomposed matrix
    @inlinable
    public var dimension: Int { eigenvalues.count }
}

/// Hermitian matrix eigendecomposition using LAPACK zheev driver
///
/// Wraps LAPACK zheev_ (complex double-precision Hermitian eigensolver) providing efficient
/// spectral decomposition H = V * diag(lambda) * V-dagger. Uses two-stage pattern: workspace
/// query with lwork = -1 followed by computation with optimal workspace allocation. All
/// eigenvalues of a Hermitian matrix are real and returned in ascending order.
///
/// **Example:**
/// ```swift
/// let hamiltonian: [[Complex<Double>]] = [
///     [Complex(1, 0), Complex(0, -1)],
///     [Complex(0, 1), Complex(1, 0)]
/// ]
/// let result = HermitianEigenDecomposition.decompose(matrix: hamiltonian)
/// ```
@frozen public enum HermitianEigenDecomposition {
    /// Computes eigendecomposition of a Hermitian matrix
    ///
    /// Diagonalizes the input Hermitian matrix H (n x n) into H = V * diag(lambda) * V-dagger
    /// using LAPACK zheev_ routine. Eigenvalues are returned in ascending order with corresponding
    /// eigenvectors forming an orthonormal basis. Uses upper triangular part of the input matrix.
    ///
    /// - Parameter matrix: Input Hermitian complex matrix (n x n) to decompose
    /// - Returns: ``HermitianEigenResult`` containing eigenvalues and eigenvectors
    /// - Precondition: Matrix must be square and non-empty
    /// - Complexity: O(n^3) for dense Hermitian eigensolver
    ///
    /// **Example:**
    /// ```swift
    /// let pauli_z: [[Complex<Double>]] = [
    ///     [Complex(1, 0), Complex(0, 0)],
    ///     [Complex(0, 0), Complex(-1, 0)]
    /// ]
    /// let result = HermitianEigenDecomposition.decompose(matrix: pauli_z)
    /// ```
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func decompose(matrix: [[Complex<Double>]]) -> HermitianEigenResult {
        ValidationUtilities.validateSquareMatrix(matrix, name: "Hermitian input matrix")

        let n = matrix.count

        var a = [Double](unsafeUninitializedCapacity: 2 * n * n) { buffer, count in
            for col in 0 ..< n {
                let colOffset = 2 * col * n
                for row in 0 ..< n {
                    let idx = colOffset + 2 * row
                    buffer[idx] = matrix[row][col].real
                    buffer[idx + 1] = matrix[row][col].imaginary
                }
            }
            count = 2 * n * n
        }

        var w = [Double](unsafeUninitializedCapacity: n) { _, count in count = n }

        var jobz = CChar(Character("V").asciiValue!)
        var uplo = CChar(Character("U").asciiValue!)
        var nn = __LAPACK_int(n)
        var lda = __LAPACK_int(n)
        var lwork = __LAPACK_int(-1)
        var info = __LAPACK_int(0)

        let rworkSize = max(1, 3 * n - 2)
        var rwork = [Double](unsafeUninitializedCapacity: rworkSize) { _, count in count = rworkSize }

        var workQuery = [Double](unsafeUninitializedCapacity: 2) { _, count in count = 2 }

        a.withUnsafeMutableBytes { aPtr in
            workQuery.withUnsafeMutableBytes { workPtr in
                w.withUnsafeMutableBufferPointer { wPtr in
                    rwork.withUnsafeMutableBufferPointer { rworkPtr in
                        zheev_(
                            &jobz, &uplo, &nn,
                            OpaquePointer(aPtr.baseAddress),
                            &lda,
                            wPtr.baseAddress,
                            OpaquePointer(workPtr.baseAddress!),
                            &lwork,
                            rworkPtr.baseAddress,
                            &info,
                        )
                    }
                }
            }
        }

        ValidationUtilities.validateLAPACKSuccess(info, operation: "zheev_ workspace query")

        let optimalWorkSize = max(1, Int(workQuery[0]))
        lwork = __LAPACK_int(optimalWorkSize)
        var work = [Double](unsafeUninitializedCapacity: 2 * optimalWorkSize) { _, count in count = 2 * optimalWorkSize }

        a.withUnsafeMutableBytes { aPtr in
            work.withUnsafeMutableBytes { workPtr in
                w.withUnsafeMutableBufferPointer { wPtr in
                    rwork.withUnsafeMutableBufferPointer { rworkPtr in
                        zheev_(
                            &jobz, &uplo, &nn,
                            OpaquePointer(aPtr.baseAddress),
                            &lda,
                            wPtr.baseAddress,
                            OpaquePointer(workPtr.baseAddress!),
                            &lwork,
                            rworkPtr.baseAddress,
                            &info,
                        )
                    }
                }
            }
        }

        ValidationUtilities.validateLAPACKSuccess(info, operation: "zheev_ computation")

        let eigenvectors = [[Complex<Double>]](unsafeUninitializedCapacity: n) { buffer, rowCount in
            for col in 0 ..< n {
                buffer[col] = [Complex<Double>](unsafeUninitializedCapacity: n) { colBuffer, colCount in
                    for row in 0 ..< n {
                        let idx = 2 * (col * n + row)
                        colBuffer[row] = Complex(a[idx], a[idx + 1])
                    }
                    colCount = n
                }
            }
            rowCount = n
        }

        return HermitianEigenResult(
            eigenvalues: w,
            eigenvectors: eigenvectors,
        )
    }
}
