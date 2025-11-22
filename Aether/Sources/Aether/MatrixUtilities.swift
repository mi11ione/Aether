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
    /// - Precondition: Both matrices must be square with same dimensions
    @_optimize(speed)
    @inlinable
    @_eagerMove
    static func matrixMultiply(_ a: GateMatrix, _ b: GateMatrix) -> GateMatrix {
        ValidationUtilities.validateSquareMatrix(a, name: "Matrix A")
        ValidationUtilities.validateSquareMatrix(b, name: "Matrix B")
        ValidationUtilities.validateSameDimensions(a, b, name1: "Matrix A", name2: "Matrix B")

        let n: Int = a.count

        // Convert to BLAS-compatible interleaved format: [real₀, imag₀, real₁, imag₁, ...]
        // Row-major layout: element (i,j) at index (i*n + j)*2
        var aInterleaved = [Double](unsafeUninitializedCapacity: n * n * 2) { buffer, count in
            for i in 0 ..< n {
                for j in 0 ..< n {
                    buffer[(i * n + j) * 2] = a[i][j].real
                    buffer[(i * n + j) * 2 + 1] = a[i][j].imaginary
                }
            }
            count = n * n * 2
        }

        var bInterleaved = [Double](unsafeUninitializedCapacity: n * n * 2) { buffer, count in
            for i in 0 ..< n {
                for j in 0 ..< n {
                    buffer[(i * n + j) * 2] = b[i][j].real
                    buffer[(i * n + j) * 2 + 1] = b[i][j].imaginary
                }
            }
            count = n * n * 2
        }

        var resultInterleaved = [Double](repeating: 0.0, count: n * n * 2)

        // BLAS matrix multiplication: C = alpha*A*B + beta*C
        // alpha = 1.0 + 0.0i, beta = 0.0 + 0.0i (pure multiplication, no accumulation)
        var alpha: [Double] = [1.0, 0.0]
        var beta: [Double] = [0.0, 0.0]

        aInterleaved.withUnsafeMutableBufferPointer { aPtr in
            bInterleaved.withUnsafeMutableBufferPointer { bPtr in
                resultInterleaved.withUnsafeMutableBufferPointer { cPtr in
                    alpha.withUnsafeMutableBufferPointer { alphaPtr in
                        beta.withUnsafeMutableBufferPointer { betaPtr in
                            cblas_zgemm(
                                CblasRowMajor,
                                CblasNoTrans,
                                CblasNoTrans,
                                Int32(n), Int32(n), Int32(n),
                                OpaquePointer(alphaPtr.baseAddress)!,
                                OpaquePointer(aPtr.baseAddress), Int32(n),
                                OpaquePointer(bPtr.baseAddress), Int32(n),
                                OpaquePointer(betaPtr.baseAddress)!,
                                OpaquePointer(cPtr.baseAddress), Int32(n)
                            )
                        }
                    }
                }
            }
        }

        // Convert back to Complex<Double> matrix
        var result: GateMatrix = Array(repeating: Array(repeating: Complex<Double>.zero, count: n), count: n)
        for i in 0 ..< n {
            for j in 0 ..< n {
                let real = resultInterleaved[(i * n + j) * 2]
                let imag = resultInterleaved[(i * n + j) * 2 + 1]
                result[i][j] = Complex(real, imag)
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
    /// - Precondition: Matrix must be square
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func hermitianConjugate(_ matrix: GateMatrix) -> GateMatrix {
        ValidationUtilities.validateSquareMatrix(matrix, name: "Matrix")
        let n: Int = matrix.count

        var result = Array(repeating: Array(repeating: Complex<Double>.zero, count: n), count: n)

        for i in 0 ..< n {
            for j in 0 ..< n {
                // Transpose indices (i ↔ j) and conjugate (negate imaginary part)
                result[i][j] = matrix[j][i].conjugate()
            }
        }

        return result
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
    /// - Precondition: Dimension must be positive
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func identityMatrix(dimension: Int) -> GateMatrix {
        ValidationUtilities.validateMatrixDimension(dimension)

        var matrix = Array(
            repeating: Array(repeating: Complex<Double>.zero, count: dimension),
            count: dimension
        )

        for i in 0 ..< dimension {
            matrix[i][i] = .one
        }

        return matrix
    }
}
