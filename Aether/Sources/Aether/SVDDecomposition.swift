// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Result of singular value decomposition containing U, S, and V-dagger matrices
///
/// Stores the decomposition A = U * diag(S) * V-dagger where U contains left singular vectors,
/// S contains singular values in descending order, and V-dagger contains right singular vectors
/// (conjugate transpose). For MPS tensor truncation, discarded singular values contribute to
/// truncation error which quantifies approximation quality.
///
/// **Example:**
/// ```swift
/// let matrix: [[Complex<Double>]] = [
///     [Complex(1, 0), Complex(2, 0)],
///     [Complex(3, 0), Complex(4, 0)]
/// ]
/// let result = SVDDecomposition.decompose(matrix: matrix, truncation: .maxBondDimension(1))
/// let u = result.u
/// let sigma = result.singularValues
/// let vDagger = result.vDagger
/// let error = result.truncationError
/// ```
@frozen public struct SVDResult: Sendable {
    /// Left singular vectors as m x k matrix (rows of U)
    public let u: [[Complex<Double>]]

    /// Singular values in descending order (k elements where k = min(m, n) or truncated)
    public let singularValues: [Double]

    /// Right singular vectors conjugate transpose as k x n matrix (rows of V-dagger)
    public let vDagger: [[Complex<Double>]]

    /// Sum of squared discarded singular values (truncation approximation error)
    public let truncationError: Double
}

/// Truncation strategies for SVD bond dimension control
///
/// Controls how singular values are truncated during MPS tensor decomposition.
/// Truncation reduces bond dimension while introducing controlled approximation error.
///
/// **Example:**
/// ```swift
/// let truncByDim = SVDTruncation.maxBondDimension(10)
/// let truncByThreshold = SVDTruncation.relativeThreshold(1e-8)
/// let truncByWeight = SVDTruncation.cumulativeWeight(epsilon: 1e-6)
/// let noTrunc = SVDTruncation.none
/// ```
@frozen public enum SVDTruncation: Sendable {
    /// Keep at most the specified number of largest singular values
    case maxBondDimension(Int)

    /// Discard singular values below threshold * max(singularValues)
    case relativeThreshold(Double)

    /// Retain smallest set where sum of kept squared values >= (1 - epsilon) * total
    case cumulativeWeight(epsilon: Double)

    /// Keep all singular values (no truncation)
    case none
}

/// Singular value decomposition using LAPACK divide-and-conquer algorithm
///
/// Wraps LAPACK zgesdd_ (complex double-precision SVD with divide-and-conquer) providing
/// efficient decomposition A = U * diag(S) * V-dagger. Supports truncation strategies for
/// MPS tensor network compression. Uses two-stage pattern: workspace query followed by
/// computation with optimal workspace allocation.
///
/// **Example:**
/// ```swift
/// let matrix: [[Complex<Double>]] = [
///     [Complex(1, 0), Complex(0, 1)],
///     [Complex(0, -1), Complex(1, 0)]
/// ]
/// let result = SVDDecomposition.decompose(matrix: matrix, truncation: .maxBondDimension(1))
/// ```
public enum SVDDecomposition {
    /// Computes SVD decomposition with optional truncation
    ///
    /// Decomposes input matrix A (m x n) into A = U * diag(S) * V-dagger using LAPACK zgesdd_
    /// divide-and-conquer algorithm. Computes thin SVD (reduced form) where U is m x k,
    /// S has k elements, and V-dagger is k x n with k = min(m, n). Applies specified truncation
    /// strategy to reduce bond dimension.
    ///
    /// - Parameters:
    ///   - matrix: Input complex matrix (m x n) to decompose
    ///   - truncation: Truncation strategy for bond dimension control (default: .none)
    /// - Returns: SVDResult containing U, singular values, V-dagger, and truncation error
    /// - Precondition: Matrix must be non-empty with consistent row lengths
    /// - Complexity: O(min(m,n) * m * n) for divide-and-conquer SVD
    ///
    /// **Example:**
    /// ```swift
    /// let matrix: [[Complex<Double>]] = [
    ///     [Complex(3, 0), Complex(0, 0)],
    ///     [Complex(0, 0), Complex(2, 0)],
    ///     [Complex(0, 0), Complex(0, 0)]
    /// ]
    /// let result = SVDDecomposition.decompose(matrix: matrix)
    /// ```
    @_optimize(speed)
    @_eagerMove
    public static func decompose(
        matrix: [[Complex<Double>]],
        truncation: SVDTruncation = .none,
    ) -> SVDResult {
        ValidationUtilities.validateNonEmpty(matrix, name: "SVD input matrix")
        let m = matrix.count
        let n = matrix[0].count
        ValidationUtilities.validatePositiveInt(n, name: "Matrix column count")

        let minDim = min(m, n)

        var a = convertToColumnMajorInterleaved(matrix, rows: m, cols: n)

        var singularValues = [Double](unsafeUninitializedCapacity: minDim) { _, count in count = minDim }
        var uMatrix = [Double](unsafeUninitializedCapacity: 2 * m * minDim) { _, count in count = 2 * m * minDim }
        var vtMatrix = [Double](unsafeUninitializedCapacity: 2 * minDim * n) { _, count in count = 2 * minDim * n }

        var jobz = CChar(Character("S").asciiValue!)
        var mm = __LAPACK_int(m)
        var nn = __LAPACK_int(n)
        var lda = __LAPACK_int(m)
        var ldu = __LAPACK_int(m)
        var ldvt = __LAPACK_int(minDim)
        var lwork = __LAPACK_int(-1)
        var info = __LAPACK_int(0)

        let rworkSize = computeRworkSize(minDim: minDim)
        var rwork = [Double](unsafeUninitializedCapacity: rworkSize) { _, count in count = rworkSize }

        let iworkSize = 8 * minDim
        var iwork = [__LAPACK_int](unsafeUninitializedCapacity: iworkSize) { _, count in count = iworkSize }

        var workQuery = [Double](unsafeUninitializedCapacity: 2) { _, count in count = 2 }

        a.withUnsafeMutableBytes { aPtr in
            singularValues.withUnsafeMutableBufferPointer { sPtr in
                uMatrix.withUnsafeMutableBytes { uPtr in
                    vtMatrix.withUnsafeMutableBytes { vtPtr in
                        workQuery.withUnsafeMutableBytes { workPtr in
                            rwork.withUnsafeMutableBufferPointer { rworkPtr in
                                iwork.withUnsafeMutableBufferPointer { iworkPtr in
                                    zgesdd_(
                                        &jobz, &mm, &nn,
                                        OpaquePointer(aPtr.baseAddress),
                                        &lda,
                                        sPtr.baseAddress,
                                        OpaquePointer(uPtr.baseAddress),
                                        &ldu,
                                        OpaquePointer(vtPtr.baseAddress),
                                        &ldvt,
                                        OpaquePointer(workPtr.baseAddress!),
                                        &lwork,
                                        rworkPtr.baseAddress,
                                        iworkPtr.baseAddress,
                                        &info,
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }

        ValidationUtilities.validateLAPACKSuccess(info, operation: "zgesdd_ workspace query")

        let optimalWorkSize = max(1, Int(workQuery[0]))
        lwork = __LAPACK_int(optimalWorkSize)
        var work = [Double](unsafeUninitializedCapacity: 2 * optimalWorkSize) { _, count in count = 2 * optimalWorkSize }

        a.withUnsafeMutableBytes { aPtr in
            singularValues.withUnsafeMutableBufferPointer { sPtr in
                uMatrix.withUnsafeMutableBytes { uPtr in
                    vtMatrix.withUnsafeMutableBytes { vtPtr in
                        work.withUnsafeMutableBytes { workPtr in
                            rwork.withUnsafeMutableBufferPointer { rworkPtr in
                                iwork.withUnsafeMutableBufferPointer { iworkPtr in
                                    zgesdd_(
                                        &jobz, &mm, &nn,
                                        OpaquePointer(aPtr.baseAddress),
                                        &lda,
                                        sPtr.baseAddress,
                                        OpaquePointer(uPtr.baseAddress),
                                        &ldu,
                                        OpaquePointer(vtPtr.baseAddress),
                                        &ldvt,
                                        OpaquePointer(workPtr.baseAddress!),
                                        &lwork,
                                        rworkPtr.baseAddress,
                                        iworkPtr.baseAddress,
                                        &info,
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }

        ValidationUtilities.validateLAPACKSuccess(info, operation: "zgesdd_ computation")

        let keepCount = computeKeepCount(singularValues: singularValues, truncation: truncation)

        let truncationError = computeTruncationError(singularValues: singularValues, keepCount: keepCount)

        let uResult = convertUFromColumnMajor(uMatrix, rows: m, cols: minDim, keepCols: keepCount)
        let vDaggerResult = convertVtFromColumnMajor(vtMatrix, rows: minDim, cols: n, keepRows: keepCount)
        let truncatedSingularValues = Array(singularValues.prefix(keepCount))

        return SVDResult(
            u: uResult,
            singularValues: truncatedSingularValues,
            vDagger: vDaggerResult,
            truncationError: truncationError,
        )
    }

    @_effects(readonly)
    private static func convertToColumnMajorInterleaved(
        _ matrix: [[Complex<Double>]],
        rows: Int,
        cols: Int,
    ) -> [Double] {
        [Double](unsafeUninitializedCapacity: 2 * rows * cols) { buffer, count in
            for col in 0 ..< cols {
                let colOffset = 2 * col * rows
                for row in 0 ..< rows {
                    let idx = colOffset + 2 * row
                    buffer[idx] = matrix[row][col].real
                    buffer[idx + 1] = matrix[row][col].imaginary
                }
            }
            count = 2 * rows * cols
        }
    }

    @_effects(readonly)
    private static func computeRworkSize(minDim: Int) -> Int {
        max(1, 5 * minDim * minDim + 5 * minDim)
    }

    @_effects(readonly)
    private static func computeKeepCount(singularValues: [Double], truncation: SVDTruncation) -> Int {
        let total = singularValues.count

        switch truncation {
        case .none:
            return total

        case let .maxBondDimension(maxDim):
            return min(maxDim, total)

        case let .relativeThreshold(threshold):
            guard let maxSV = singularValues.first, maxSV > 0 else { return total }
            let cutoff = threshold * maxSV
            var keepCount = 0
            for sv in singularValues {
                if sv >= cutoff {
                    keepCount += 1
                } else {
                    break
                }
            }
            return max(1, keepCount)

        case let .cumulativeWeight(epsilon):
            let totalWeight = singularValues.reduce(0.0) { $0 + $1 * $1 }
            guard totalWeight > 0 else { return total }
            let targetWeight = (1.0 - epsilon) * totalWeight
            var accumulated = 0.0
            var keepCount = 0
            for sv in singularValues {
                accumulated += sv * sv
                keepCount += 1
                if accumulated >= targetWeight {
                    break
                }
            }
            return max(1, keepCount)
        }
    }

    @_effects(readonly)
    private static func computeTruncationError(singularValues: [Double], keepCount: Int) -> Double {
        guard keepCount < singularValues.count else { return 0.0 }
        var error = 0.0
        for i in keepCount ..< singularValues.count {
            let sv = singularValues[i]
            error += sv * sv
        }
        return error
    }

    @_effects(readonly)
    private static func convertUFromColumnMajor(
        _ uMatrix: [Double],
        rows: Int,
        cols _: Int,
        keepCols: Int,
    ) -> [[Complex<Double>]] {
        [[Complex<Double>]](unsafeUninitializedCapacity: rows) { rowBuffer, rowCount in
            for row in 0 ..< rows {
                rowBuffer[row] = [Complex<Double>](unsafeUninitializedCapacity: keepCols) { colBuffer, colCount in
                    for col in 0 ..< keepCols {
                        let idx = 2 * (col * rows + row)
                        colBuffer[col] = Complex(uMatrix[idx], uMatrix[idx + 1])
                    }
                    colCount = keepCols
                }
            }
            rowCount = rows
        }
    }

    @_effects(readonly)
    private static func convertVtFromColumnMajor(
        _ vtMatrix: [Double],
        rows: Int,
        cols: Int,
        keepRows: Int,
    ) -> [[Complex<Double>]] {
        [[Complex<Double>]](unsafeUninitializedCapacity: keepRows) { rowBuffer, rowCount in
            for row in 0 ..< keepRows {
                rowBuffer[row] = [Complex<Double>](unsafeUninitializedCapacity: cols) { colBuffer, colCount in
                    for col in 0 ..< cols {
                        let idx = 2 * (col * rows + row)
                        colBuffer[col] = Complex(vtMatrix[idx], vtMatrix[idx + 1])
                    }
                    colCount = cols
                }
            }
            rowCount = keepRows
        }
    }
}
