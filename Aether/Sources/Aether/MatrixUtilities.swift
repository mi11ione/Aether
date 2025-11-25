// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Shared matrix utilities for quantum computing
///
/// Centralizes common matrix operations used throughout the quantum simulator,
/// particularly for quantum gate manipulation and unitary optimization. All
/// operations use Apple's Accelerate BLAS for optimal performance on Apple Silicon.
///
/// **Memory Layout**: Uses row-major ordering (C-style) for consistency with
/// Swift's natural 2D array indexing matrix[row][col]. BLAS operations configured
/// with CblasRowMajor to match this convention.
@frozen
public enum MatrixUtilities {
    /// Complex matrix multiplication using BLAS (A × B)
    ///
    /// Computes matrix product C = A × B for square complex matrices using Apple's
    /// Accelerate framework BLAS with hardware-accelerated vectorization and AMX
    /// coprocessor utilization.
    ///
    /// **Algorithm**:
    /// 1. Convert Swift Complex<Double> arrays to interleaved Double arrays [r₀, i₀, r₁, i₁, ...]
    /// 2. Call BLAS `cblas_zgemm` with row-major layout
    /// 3. Convert result back to Complex<Double> matrix
    ///
    /// **Memory Layout**: Row-major (CblasRowMajor)
    /// - Indexing: buffer[(row * n + col) * 2] = real, buffer[(row * n + col) * 2 + 1] = imag
    /// - Matches Swift's natural matrix[row][col] access pattern
    ///
    /// **Complexity**: O(n³) arithmetic operations with hardware acceleration via SIMD
    /// and cache optimization.
    ///
    /// **Use Cases**:
    /// - Gate composition: U = U₂ × U₁
    /// - Unitarity checking: U†U = I
    /// - Conjugate by unitary: U†PU for operator transformations
    ///
    /// - Parameters:
    ///   - a: Left matrix (n×n)
    ///   - b: Right matrix (n×n)
    /// - Returns: Product matrix C = A × B (n×n)
    @_optimize(speed)
    @inlinable
    @_eagerMove
    static func matrixMultiply(_ a: GateMatrix, _ b: GateMatrix) -> GateMatrix {
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
                                OpaquePointer(cPtr.baseAddress), Int32(n)
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

    /// Hermitian conjugate (conjugate transpose) of complex matrix
    ///
    /// Computes M† = (M*)ᵀ where * denotes complex conjugation and ᵀ denotes
    /// matrix transpose. Essential operation for unitary validation (U†U = I)
    /// and operator transformations in quantum mechanics.
    ///
    /// **Algorithm**: (M†)[i][j] = conj(M[j][i])
    /// - Swap row/column indices (transpose)
    /// - Negate imaginary part (complex conjugate)
    ///
    /// **Physical Interpretation**:
    /// - For unitary matrices: U† = U⁻¹ (inverse equals conjugate transpose)
    /// - For Hermitian operators: H† = H (observables are self-adjoint)
    /// - Time-reversal: U† reverses quantum evolution U
    ///
    /// **Complexity**: O(n²) - single pass through matrix elements
    ///
    /// **Use Cases**:
    /// - Unitarity check: isUnitary(U) = U†U ≈ I
    /// - Basis change: O' = U†OU for operator O in new basis
    /// - Adjoint expectation: ⟨ψ|O|ψ⟩ = ⟨ψ|O†|ψ⟩* for Hermitian O
    ///
    /// - Parameter matrix: Square complex matrix (n×n)
    /// - Returns: Hermitian conjugate M† (n×n)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func hermitianConjugate(_ matrix: GateMatrix) -> GateMatrix {
        ValidationUtilities.validateSquareMatrix(matrix, name: "Matrix")
        let n = matrix.count

        return (0 ..< n).map { i in
            [Complex<Double>](unsafeUninitializedCapacity: n) { buffer, count in
                for j in 0 ..< n {
                    buffer[j] = matrix[j][i].conjugate()
                }
                count = n
            }
        }
    }

    /// Create identity matrix of specified dimension
    ///
    /// Constructs n×n identity matrix with 1s on diagonal and 0s elsewhere.
    /// Used as initialization for unitary transformations and as reference
    /// for unitarity checks (U†U should equal identity).
    ///
    /// **Properties**:
    /// - I × M = M × I = M for any matrix M
    /// - I is unitary: I† = I and I†I = I
    /// - Eigenvalues: all equal to 1
    ///
    /// - Parameter dimension: Size of identity matrix (n×n)
    /// - Returns: Identity matrix I_n
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func identityMatrix(dimension: Int) -> GateMatrix {
        ValidationUtilities.validateMatrixDimension(dimension)

        return (0 ..< dimension).map { i in
            [Complex<Double>](unsafeUninitializedCapacity: dimension) { buffer, count in
                for j in 0 ..< dimension {
                    buffer[j] = (i == j) ? .one : .zero
                }
                count = dimension
            }
        }
    }
}
