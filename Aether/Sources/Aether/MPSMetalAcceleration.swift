// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate
import Metal
import MetalPerformanceShaders

/// GPU-accelerated tensor contraction for MPS operations.
///
/// Uses Metal Performance Shaders for matrix multiplication underlying tensor contractions.
/// Falls back to BLAS when Metal unavailable or for small tensors (bond dimension < 32).
/// GPU acceleration provides substantial speedup for large bond dimensions where matrix
/// multiplication dominates computation time. Float32 GPU precision introduces ~1e-7
/// relative error, acceptable for most MPS algorithms.
///
/// **Example:**
/// ```swift
/// let accelerator = MPSMetalAcceleration()
/// let contracted = await accelerator.contractAdjacentTensors(leftTensor, rightTensor)
/// let chainResult = await accelerator.chainContraction(matrices: matrixChain)
/// ```
///
/// - SeeAlso: ``MPSTensor``
/// - SeeAlso: ``MPSBatchEvaluator``
public actor MPSMetalAcceleration {
    /// Minimum bond dimension for GPU acceleration.
    ///
    /// Below this threshold (bond dimension < 32), CPU BLAS execution is faster due to GPU
    /// overhead (buffer allocation, shader dispatch, Float64->Float32 conversion). At bond
    /// dimension 32+, GPU parallelism benefit outweighs overhead.
    ///
    /// **Example:**
    /// ```swift
    /// if tensor.leftBondDimension >= MPSMetalAcceleration.gpuThreshold {
    ///     // Use GPU path
    /// }
    /// ```
    public static let gpuThreshold = 32

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?

    /// Whether Metal GPU acceleration is available on this system.
    ///
    /// Returns `true` when Metal device initialized successfully, indicating GPU-accelerated
    /// tensor contraction is available. When `false`, all operations fall back to CPU BLAS.
    ///
    /// **Example:**
    /// ```swift
    /// let accelerator = MPSMetalAcceleration()
    /// if accelerator.isAvailable { print("GPU ready") }
    /// ```
    public nonisolated var isAvailable: Bool { device != nil }

    /// Creates GPU accelerator using system default Metal device.
    ///
    /// Initializes Metal device and command queue for GPU operations. Falls back gracefully
    /// to CPU BLAS when Metal is unavailable. Thread-safe actor isolation ensures safe
    /// concurrent access to Metal command buffers.
    ///
    /// **Example:**
    /// ```swift
    /// let accelerator = MPSMetalAcceleration()
    /// if accelerator.isAvailable {
    ///     let result = await accelerator.matrixMultiply(a, b)
    /// }
    /// ```
    public init() {
        device = MTLCreateSystemDefaultDevice()
        commandQueue = device?.makeCommandQueue()
    }

    /// Contract two adjacent MPS tensors into combined tensor.
    ///
    /// Computes C[α,i,j,γ] = Σ_β A[α,i,β] * B[β,j,γ] where α is the left bond of A,
    /// i and j are physical indices, β is the contracted bond, and γ is the right bond of B.
    /// Uses GPU for large tensors (bond dimension >= 32), CPU BLAS for small ones.
    ///
    /// **Example:**
    /// ```swift
    /// let accelerator = MPSMetalAcceleration()
    /// let leftTensor = MPSTensor.groundState(site: 0, qubits: 4, maxBondDimension: 64)
    /// let rightTensor = MPSTensor.groundState(site: 1, qubits: 4, maxBondDimension: 64)
    /// let contracted = await accelerator.contractAdjacentTensors(leftTensor, rightTensor)
    /// // contracted[alpha][i][j][gamma] contains the 4D result tensor
    /// ```
    ///
    /// - Parameters:
    ///   - left: Left MPS tensor A[α,i,β]
    ///   - right: Right MPS tensor B[β,j,γ]
    /// - Returns: 4D tensor C[α,i,j,γ] from contraction over shared bond index
    /// - Complexity: O(α * β * γ * 4) matrix multiplications, GPU-accelerated for large bonds
    /// - Precondition: left.rightBondDimension == right.leftBondDimension
    @_optimize(speed)
    @_eagerMove
    public func contractAdjacentTensors(
        _ left: MPSTensor,
        _ right: MPSTensor,
    ) -> [[[[Complex<Double>]]]] {
        let alpha = left.leftBondDimension
        let gamma = right.rightBondDimension

        ValidationUtilities.validateBondDimensionMatch(left.rightBondDimension, right.leftBondDimension)

        var result = [[[[Complex<Double>]]]](
            repeating: [[[Complex<Double>]]](
                repeating: [[Complex<Double>]](
                    repeating: [Complex<Double>](repeating: .zero, count: gamma),
                    count: 2,
                ),
                count: 2,
            ),
            count: alpha,
        )

        let leftMatrix0 = left.matrixForPhysicalIndex(0)
        let leftMatrix1 = left.matrixForPhysicalIndex(1)
        let rightMatrix0 = right.matrixForPhysicalIndex(0)
        let rightMatrix1 = right.matrixForPhysicalIndex(1)

        let c00 = matrixMultiply(leftMatrix0, rightMatrix0)
        let c01 = matrixMultiply(leftMatrix0, rightMatrix1)
        let c10 = matrixMultiply(leftMatrix1, rightMatrix0)
        let c11 = matrixMultiply(leftMatrix1, rightMatrix1)

        for a in 0 ..< alpha {
            for g in 0 ..< gamma {
                result[a][0][0][g] = c00[a][g]
                result[a][0][1][g] = c01[a][g]
                result[a][1][0][g] = c10[a][g]
                result[a][1][1][g] = c11[a][g]
            }
        }

        return result
    }

    /// Batched matrix multiplication for MPS chain contraction.
    ///
    /// Multiplies sequence of matrices: M0 * M1 * M2 * ... * Mn. Uses GPU for large matrices
    /// (dimension >= 32), CPU BLAS for small ones. Sequential multiplication from left to right
    /// maintains numerical stability for MPS contractions.
    ///
    /// **Example:**
    /// ```swift
    /// let accelerator = MPSMetalAcceleration()
    /// let matrices: [[[Complex<Double>]]] = tensors.map { $0.matrixForPhysicalIndex(0) }
    /// let chainResult = await accelerator.chainContraction(matrices: matrices)
    /// // chainResult is the product of all matrices
    /// ```
    ///
    /// - Parameter matrices: Sequence of complex matrices to multiply
    /// - Returns: Product matrix M0 * M1 * ... * Mn, or identity for empty input
    /// - Complexity: O(n * d³) where n is matrix count and d is dimension
    /// - Precondition: All matrices must be square with compatible dimensions
    @_optimize(speed)
    @_eagerMove
    public func chainContraction(
        matrices: [[[Complex<Double>]]],
    ) -> [[Complex<Double>]] {
        guard !matrices.isEmpty else {
            return [[.one]]
        }

        if matrices.count == 1 {
            return matrices[0]
        }

        var result = matrices[0]
        for i in 1 ..< matrices.count {
            result = matrixMultiply(result, matrices[i])
        }

        return result
    }

    /// Matrix multiply two complex matrices using GPU or CPU.
    ///
    /// Computes result = A * B using Metal Performance Shaders for large matrices
    /// (dimension >= 32) or CPU BLAS for small matrices. Complex multiplication is
    /// decomposed into real/imaginary parts for GPU: result_real = A_real * B_real - A_imag * B_imag,
    /// result_imag = A_real * B_imag + A_imag * B_real.
    ///
    /// **Example:**
    /// ```swift
    /// let accelerator = MPSMetalAcceleration()
    /// let a: [[Complex<Double>]] = [[.one, .zero], [.zero, .one]]
    /// let b: [[Complex<Double>]] = [[.one, .i], [.i, .one]]
    /// let product = await accelerator.matrixMultiply(a, b)
    /// ```
    ///
    /// - Parameters:
    ///   - a: Left matrix (m x k)
    ///   - b: Right matrix (k x n)
    /// - Returns: Product matrix (m x n), or empty matrix on failure
    /// - Complexity: O(m * k * n), GPU-accelerated for large matrices
    /// - Precondition: a column count must equal b row count
    @_optimize(speed)
    @_eagerMove
    public func matrixMultiply(
        _ a: [[Complex<Double>]],
        _ b: [[Complex<Double>]],
    ) -> [[Complex<Double>]] {
        guard !a.isEmpty, !b.isEmpty, !a[0].isEmpty, !b[0].isEmpty else {
            return []
        }

        let m = a.count
        let k = a[0].count
        let n = b[0].count

        guard b.count == k else {
            return []
        }

        let minDimension = min(m, k, n)
        let useGPU = minDimension >= Self.gpuThreshold && device != nil && commandQueue != nil

        if useGPU {
            return matrixMultiplyGPU(a, b, m: m, k: k, n: n)
        }

        return matrixMultiplyCPU(a, b, m: m, k: k, n: n)
    }

    @_optimize(speed)
    @_eagerMove
    private func matrixMultiplyGPU(
        _ a: [[Complex<Double>]],
        _ b: [[Complex<Double>]],
        m: Int,
        k: Int,
        n: Int,
    ) -> [[Complex<Double>]] {
        guard let device, let commandQueue else {
            return matrixMultiplyCPU(a, b, m: m, k: k, n: n)
        }

        var aReal = [Float](unsafeUninitializedCapacity: m * k) { buffer, count in
            for i in 0 ..< m {
                for j in 0 ..< k {
                    buffer[i * k + j] = Float(a[i][j].real)
                }
            }
            count = m * k
        }

        var aImag = [Float](unsafeUninitializedCapacity: m * k) { buffer, count in
            for i in 0 ..< m {
                for j in 0 ..< k {
                    buffer[i * k + j] = Float(a[i][j].imaginary)
                }
            }
            count = m * k
        }

        var bReal = [Float](unsafeUninitializedCapacity: k * n) { buffer, count in
            for i in 0 ..< k {
                for j in 0 ..< n {
                    buffer[i * n + j] = Float(b[i][j].real)
                }
            }
            count = k * n
        }

        var bImag = [Float](unsafeUninitializedCapacity: k * n) { buffer, count in
            for i in 0 ..< k {
                for j in 0 ..< n {
                    buffer[i * n + j] = Float(b[i][j].imaginary)
                }
            }
            count = k * n
        }

        let aRealRowBytes = k * MemoryLayout<Float>.stride
        let bRealRowBytes = n * MemoryLayout<Float>.stride
        let cRealRowBytes = n * MemoryLayout<Float>.stride

        guard let aRealBuffer = device.makeBuffer(bytes: &aReal, length: m * k * MemoryLayout<Float>.stride, options: .storageModeShared),
              let aImagBuffer = device.makeBuffer(bytes: &aImag, length: m * k * MemoryLayout<Float>.stride, options: .storageModeShared),
              let bRealBuffer = device.makeBuffer(bytes: &bReal, length: k * n * MemoryLayout<Float>.stride, options: .storageModeShared),
              let bImagBuffer = device.makeBuffer(bytes: &bImag, length: k * n * MemoryLayout<Float>.stride, options: .storageModeShared),
              let cRealBuffer = device.makeBuffer(length: m * n * MemoryLayout<Float>.stride, options: .storageModeShared),
              let cImagBuffer = device.makeBuffer(length: m * n * MemoryLayout<Float>.stride, options: .storageModeShared)
        else {
            return matrixMultiplyCPU(a, b, m: m, k: k, n: n)
        }

        let aRealDesc = MPSMatrixDescriptor(rows: m, columns: k, rowBytes: aRealRowBytes, dataType: .float32)
        let bRealDesc = MPSMatrixDescriptor(rows: k, columns: n, rowBytes: bRealRowBytes, dataType: .float32)
        let cRealDesc = MPSMatrixDescriptor(rows: m, columns: n, rowBytes: cRealRowBytes, dataType: .float32)

        let aRealMatrix = MPSMatrix(buffer: aRealBuffer, descriptor: aRealDesc)
        let aImagMatrix = MPSMatrix(buffer: aImagBuffer, descriptor: aRealDesc)
        let bRealMatrix = MPSMatrix(buffer: bRealBuffer, descriptor: bRealDesc)
        let bImagMatrix = MPSMatrix(buffer: bImagBuffer, descriptor: bRealDesc)
        let cRealMatrix = MPSMatrix(buffer: cRealBuffer, descriptor: cRealDesc)
        let cImagMatrix = MPSMatrix(buffer: cImagBuffer, descriptor: cRealDesc)

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return matrixMultiplyCPU(a, b, m: m, k: k, n: n)
        }

        let matMulInit = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: m, resultColumns: n, interiorColumns: k, alpha: 1.0, beta: 0.0)
        let matMulSubtract = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: m, resultColumns: n, interiorColumns: k, alpha: -1.0, beta: 1.0)
        let matMulAdd = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: m, resultColumns: n, interiorColumns: k, alpha: 1.0, beta: 1.0)

        matMulInit.encode(commandBuffer: commandBuffer, leftMatrix: aRealMatrix, rightMatrix: bRealMatrix, resultMatrix: cRealMatrix)
        matMulSubtract.encode(commandBuffer: commandBuffer, leftMatrix: aImagMatrix, rightMatrix: bImagMatrix, resultMatrix: cRealMatrix)

        matMulInit.encode(commandBuffer: commandBuffer, leftMatrix: aRealMatrix, rightMatrix: bImagMatrix, resultMatrix: cImagMatrix)
        matMulAdd.encode(commandBuffer: commandBuffer, leftMatrix: aImagMatrix, rightMatrix: bRealMatrix, resultMatrix: cImagMatrix)

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let cRealPtr = cRealBuffer.contents().bindMemory(to: Float.self, capacity: m * n)
        let cImagPtr = cImagBuffer.contents().bindMemory(to: Float.self, capacity: m * n)

        let result = (0 ..< m).map { i in
            [Complex<Double>](unsafeUninitializedCapacity: n) { buffer, count in
                for j in 0 ..< n {
                    let idx = i * n + j
                    buffer[j] = Complex(Double(cRealPtr[idx]), Double(cImagPtr[idx]))
                }
                count = n
            }
        }

        guard result.allSatisfy({ row in row.allSatisfy(\.isFinite) }) else {
            return matrixMultiplyCPU(a, b, m: m, k: k, n: n)
        }

        return result
    }

    @_optimize(speed)
    @_eagerMove
    private func matrixMultiplyCPU(
        _ a: [[Complex<Double>]],
        _ b: [[Complex<Double>]],
        m: Int,
        k: Int,
        n: Int,
    ) -> [[Complex<Double>]] {
        let mk2 = m * k * 2
        let kn2 = k * n * 2
        let mn2 = m * n * 2

        var aInterleaved = [Double](unsafeUninitializedCapacity: mk2) { buffer, count in
            for col in 0 ..< k {
                for row in 0 ..< m {
                    let idx = (col * m + row) * 2
                    buffer[idx] = a[row][col].real
                    buffer[idx + 1] = a[row][col].imaginary
                }
            }
            count = mk2
        }

        var bInterleaved = [Double](unsafeUninitializedCapacity: kn2) { buffer, count in
            for col in 0 ..< n {
                for row in 0 ..< k {
                    let idx = (col * k + row) * 2
                    buffer[idx] = b[row][col].real
                    buffer[idx + 1] = b[row][col].imaginary
                }
            }
            count = kn2
        }

        var resultInterleaved = [Double](unsafeUninitializedCapacity: mn2) { _, count in
            count = mn2
        }

        var alpha = (1.0, 0.0)
        var beta = (0.0, 0.0)

        aInterleaved.withUnsafeMutableBufferPointer { aPtr in
            bInterleaved.withUnsafeMutableBufferPointer { bPtr in
                resultInterleaved.withUnsafeMutableBufferPointer { cPtr in
                    withUnsafePointer(to: &alpha) { alphaPtr in
                        withUnsafePointer(to: &beta) { betaPtr in
                            cblas_zgemm(
                                CblasColMajor,
                                CblasNoTrans,
                                CblasNoTrans,
                                Int32(m),
                                Int32(n),
                                Int32(k),
                                OpaquePointer(alphaPtr),
                                OpaquePointer(aPtr.baseAddress),
                                Int32(m),
                                OpaquePointer(bPtr.baseAddress),
                                Int32(k),
                                OpaquePointer(betaPtr),
                                OpaquePointer(cPtr.baseAddress),
                                Int32(m),
                            )
                        }
                    }
                }
            }
        }

        let result = (0 ..< m).map { row in
            [Complex<Double>](unsafeUninitializedCapacity: n) { buffer, count in
                for col in 0 ..< n {
                    let idx = (col * m + row) * 2
                    buffer[col] = Complex(resultInterleaved[idx], resultInterleaved[idx + 1])
                }
                count = n
            }
        }

        return result
    }
}
