// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Shared matrix utilities for quantum computing
///
/// Centralizes common matrix operations used throughout the quantum simulator for gate composition,
/// unitarity validation, and operator transformations. All operations use Apple's Accelerate BLAS
///
/// Essential for gate algebra (composing sequential gates into single unitary), verifying gate
/// correctness (U†U = I), and basis transformations (U†OU). Matrix multiplication is the fundamental
/// operation for gate fusion optimization and circuit compilation. BLAS-accelerated operations
/// provide speedup over naive loops via SIMD vectorization and cache optimization, critical for
/// unitary partitioning, circuit optimization, and multi-gate composition in variational algorithms.
///
/// **Example:**
/// ```swift
/// let h = QuantumGate.hadamard.matrix()
/// let x = QuantumGate.pauliX.matrix()
/// let composed = MatrixUtilities.matrixMultiply(h, x)  // H x X composition
/// let adjoint = MatrixUtilities.hermitianConjugate(composed)
/// let identity = MatrixUtilities.matrixMultiply(adjoint, composed)  // U† x U = I
/// ```
public enum MatrixUtilities {
    /// Compute matrix product C = A x B using BLAS
    ///
    /// Performs dense complex matrix multiplication computation.
    /// Primary use cases: gate composition (fusing sequential gates), unitarity validation (U†U),
    /// and basis transformations (U†OU for operator conjugation).
    ///
    /// **Example:**
    /// ```swift
    /// let h = QuantumGate.hadamard.matrix()
    /// let x = QuantumGate.pauliX.matrix()
    /// let hx = MatrixUtilities.matrixMultiply(h, x)  // Fuse H and X into single gate
    /// ```
    ///
    /// - Parameters:
    ///   - a: Left matrix (nxn)
    ///   - b: Right matrix (nxn)
    /// - Returns: Product matrix C = A x B (nxn)
    /// - Complexity: O(n³) with BLAS hardware acceleration (SIMD, cache optimization)
    /// - Precondition: Both matrices must be square with matching dimensions
    @_optimize(speed)
    @inlinable
    @_eagerMove
    static func matrixMultiply(_ a: [[Complex<Double>]], _ b: [[Complex<Double>]]) -> [[Complex<Double>]] {
        ValidationUtilities.validateSquareMatrix(a, name: "Matrix A")
        ValidationUtilities.validateSquareMatrix(b, name: "Matrix B")
        ValidationUtilities.validateSameDimensions(a, b, name1: "Matrix A", name2: "Matrix B")

        let n = a.count
        let nn = n * n
        let nn2 = nn * 2

        var aInterleaved = [Double](unsafeUninitializedCapacity: nn2) { buffer, count in
            for i in 0 ..< n {
                for j in 0 ..< n {
                    let idx = (i * n + j) * 2
                    buffer[idx] = a[i][j].real
                    buffer[idx + 1] = a[i][j].imaginary
                }
            }
            count = nn2
        }

        var bInterleaved = [Double](unsafeUninitializedCapacity: nn2) { buffer, count in
            for i in 0 ..< n {
                for j in 0 ..< n {
                    let idx = (i * n + j) * 2
                    buffer[idx] = b[i][j].real
                    buffer[idx + 1] = b[i][j].imaginary
                }
            }
            count = nn2
        }

        var resultInterleaved = [Double](unsafeUninitializedCapacity: nn2) { _, count in
            count = nn2
        }

        var alpha = (1.0, 0.0)
        var beta = (0.0, 0.0)

        aInterleaved.withUnsafeMutableBufferPointer { aPtr in
            bInterleaved.withUnsafeMutableBufferPointer { bPtr in
                resultInterleaved.withUnsafeMutableBufferPointer { cPtr in
                    withUnsafeMutablePointer(to: &alpha) { alphaPtr in
                        withUnsafeMutablePointer(to: &beta) { betaPtr in
                            cblas_zgemm(
                                CblasRowMajor,
                                CblasNoTrans,
                                CblasNoTrans,
                                Int32(n), Int32(n), Int32(n),
                                OpaquePointer(alphaPtr),
                                OpaquePointer(aPtr.baseAddress), Int32(n),
                                OpaquePointer(bPtr.baseAddress), Int32(n),
                                OpaquePointer(betaPtr),
                                OpaquePointer(cPtr.baseAddress), Int32(n),
                            )
                        }
                    }
                }
            }
        }

        let result = (0 ..< n).map { i in
            [Complex<Double>](unsafeUninitializedCapacity: n) { buffer, count in
                for j in 0 ..< n {
                    let idx = (i * n + j) * 2
                    buffer[j] = Complex(resultInterleaved[idx], resultInterleaved[idx + 1])
                }
                count = n
            }
        }

        return result
    }

    /// Compute Hermitian conjugate (conjugate transpose) M† = (M*)ᵀ
    ///
    /// Computes conjugate transpose by swapping row/column indices and negating imaginary parts.
    /// Essential for unitarity validation (U†U = I), operator basis transformations (U†OU), and
    /// computing adjoint evolution. For unitary gates: U† = U⁻¹ (inverse equals conjugate transpose).
    ///
    /// **Example:**
    /// ```swift
    /// let u = QuantumGate.hadamard.matrix()
    /// let uDagger = MatrixUtilities.hermitianConjugate(u)
    /// let identity = MatrixUtilities.matrixMultiply(uDagger, u)  // U† x U = I
    /// ```
    ///
    /// - Parameter matrix: Square complex matrix (nxn)
    /// - Returns: Hermitian conjugate M† (nxn)
    /// - Complexity: O(n²)
    /// - Precondition: Matrix must be square
    /// - Note: For Hermitian operators (observables): H† = H (self-adjoint property)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func hermitianConjugate(_ matrix: [[Complex<Double>]]) -> [[Complex<Double>]] {
        ValidationUtilities.validateSquareMatrix(matrix, name: "Matrix")
        let n = matrix.count

        return (0 ..< n).map { i in
            [Complex<Double>](unsafeUninitializedCapacity: n) { buffer, count in
                for j in 0 ..< n {
                    buffer[j] = matrix[j][i].conjugate
                }
                count = n
            }
        }
    }

    /// Raise square matrix to non-negative integer power via repeated squaring
    ///
    /// Computes M^n using binary exponentiation with O(log n) matrix multiplications instead of
    /// naive O(n) sequential products. Returns the identity matrix for exponent zero and the
    /// original matrix for exponent one as base cases. Used by ``QuantumCircuit/power(_:)`` to
    /// raise circuit unitaries to integer powers for repeated gate application.
    ///
    /// **Example:**
    /// ```swift
    /// let x = QuantumGate.pauliX.matrix()
    /// let x2 = MatrixUtilities.matrixPower(x, exponent: 2)  // X² = I
    /// let x3 = MatrixUtilities.matrixPower(x, exponent: 3)  // X³ = X
    /// ```
    ///
    /// - Parameters:
    ///   - matrix: Square complex matrix (nxn)
    ///   - exponent: Non-negative integer power
    /// - Returns: Matrix raised to the given power M^exponent (nxn)
    /// - Complexity: O(n³ log(exponent)) with BLAS-accelerated matrix multiplication
    /// - Precondition: Matrix must be square
    /// - Precondition: Exponent must be non-negative
    @_optimize(speed)
    @_eagerMove
    public static func matrixPower(_ matrix: [[Complex<Double>]], exponent: Int) -> [[Complex<Double>]] {
        ValidationUtilities.validateSquareMatrix(matrix, name: "Matrix")
        ValidationUtilities.validateNonNegativeInt(exponent, name: "Exponent")

        let n = matrix.count

        if exponent == 0 {
            return identityMatrix(dimension: n)
        }

        if exponent == 1 {
            return matrix
        }

        var result = identityMatrix(dimension: n)
        var base = matrix
        var exp = exponent

        while exp > 0 {
            if exp & 1 == 1 {
                result = matrixMultiply(result, base)
            }
            exp >>= 1
            if exp > 0 {
                base = matrixMultiply(base, base)
            }
        }

        return result
    }

    /// Create identity matrix with 1s on diagonal, 0s elsewhere
    ///
    /// Constructs nxn identity matrix for unitarity validation reference (U†U should equal I),
    /// initialization of unitary transformations, and matrix algebra operations. Identity satisfies
    /// I x M = M x I = M for any matrix M, and is itself unitary: I† = I.
    ///
    /// **Example:**
    /// ```swift
    /// let i4 = MatrixUtilities.identityMatrix(dimension: 4)  // 4x4 identity for 2-qubit gates
    /// ```
    ///
    /// - Parameter dimension: Size of identity matrix (nxn)
    /// - Returns: Identity matrix I_n with 1s on diagonal, 0s elsewhere
    /// - Complexity: O(n²)
    /// - Precondition: dimension > 0
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func identityMatrix(dimension: Int) -> [[Complex<Double>]] {
        ValidationUtilities.validateMatrixDimension(dimension)

        return (0 ..< dimension).map { i in
            [Complex<Double>](unsafeUninitializedCapacity: dimension) { buffer, count in
                for j in 0 ..< dimension {
                    buffer[j] = .zero
                }
                buffer[i] = .one
                count = dimension
            }
        }
    }
}
