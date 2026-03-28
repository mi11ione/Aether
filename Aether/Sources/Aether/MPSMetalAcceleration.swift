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
/// let contracted = await accelerator.contract(leftTensor, rightTensor)
/// let chainResult = await accelerator.multiply(chain: matrixChain)
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
    /// let useGPU = tensor.leftBondDimension >= MPSMetalAcceleration.gpuThreshold
    /// ```
    public static let gpuThreshold = 32

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private var cachedM = 0
    private var cachedK = 0
    private var cachedN = 0
    private var cachedMatMulInit: MPSMatrixMultiplication?
    private var cachedMatMulSubtract: MPSMatrixMultiplication?
    private var cachedMatMulAdd: MPSMatrixMultiplication?

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
    public nonisolated var isAvailable: Bool {
        device != nil
    }

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
    ///     let product = await accelerator.multiply(a, b)
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
    /// let contracted = await accelerator.contract(leftTensor, rightTensor)
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
    public func contract(
        _ left: MPSTensor,
        _ right: MPSTensor,
    ) -> [[[[Complex<Double>]]]] {
        let alpha = left.leftBondDimension

        ValidationUtilities.validateBondDimensionMatch(left.rightBondDimension, right.leftBondDimension)

        let leftMatrix0 = left.matrix(forPhysical: 0)
        let leftMatrix1 = left.matrix(forPhysical: 1)
        let rightMatrix0 = right.matrix(forPhysical: 0)
        let rightMatrix1 = right.matrix(forPhysical: 1)

        let c00 = multiply(leftMatrix0, rightMatrix0)
        let c01 = multiply(leftMatrix0, rightMatrix1)
        let c10 = multiply(leftMatrix1, rightMatrix0)
        let c11 = multiply(leftMatrix1, rightMatrix1)

        return (0 ..< alpha).map { a in
            [
                [c00[a], c01[a]],
                [c10[a], c11[a]],
            ]
        }
    }

    /// Multiply a chain of complex matrices sequentially.
    ///
    /// Computes M0 * M1 * M2 * ... * Mn from left to right. Uses GPU for large matrices
    /// (dimension >= 32), CPU BLAS for small ones. Sequential multiplication maintains
    /// numerical stability for MPS contractions.
    ///
    /// **Example:**
    /// ```swift
    /// let accelerator = MPSMetalAcceleration()
    /// let matrices: [[[Complex<Double>]]] = tensors.map { $0.matrix(forPhysical:0) }
    /// let chainResult = await accelerator.multiply(chain: matrices)
    /// ```
    ///
    /// - Parameter matrices: Sequence of complex matrices to multiply
    /// - Returns: Product matrix M0 * M1 * ... * Mn, or identity for empty input
    /// - Complexity: O(n * d³) where n is matrix count and d is dimension
    /// - Precondition: All matrices must be square with compatible dimensions
    @_optimize(speed)
    @_eagerMove
    public func multiply(
        chain matrices: [[[Complex<Double>]]],
    ) -> [[Complex<Double>]] {
        guard !matrices.isEmpty else {
            return [[.one]]
        }

        if matrices.count == 1 {
            return matrices[0]
        }

        var result = matrices[0]
        for i in 1 ..< matrices.count {
            result = multiply(result, matrices[i])
        }

        return result
    }

    /// Multiply two complex matrices using GPU or CPU.
    ///
    /// Computes the matrix product C = A * B for complex-valued matrices. Uses Metal
    /// Performance Shaders for large matrices (dimension >= 32) or CPU BLAS for small ones.
    ///
    /// **Example:**
    /// ```swift
    /// let accelerator = MPSMetalAcceleration()
    /// let a: [[Complex<Double>]] = [[.one, .zero], [.zero, .one]]
    /// let b: [[Complex<Double>]] = [[.one, .i], [.i, .one]]
    /// let result = await accelerator.multiply(a, b)
    /// ```
    ///
    /// - Parameters:
    ///   - a: Left matrix (m x k)
    ///   - b: Right matrix (k x n)
    /// - Returns: Product matrix (m x n), or empty matrix for empty input
    /// - Complexity: O(m * k * n), GPU-accelerated for large matrices
    /// - Precondition: a column count == b row count
    @_optimize(speed)
    @_eagerMove
    public func multiply(
        _ a: [[Complex<Double>]],
        _ b: [[Complex<Double>]],
    ) -> [[Complex<Double>]] {
        guard !a.isEmpty, !b.isEmpty, !a[0].isEmpty, !b[0].isEmpty else {
            return []
        }

        let m = a.count
        let k = a[0].count
        let n = b[0].count

        ValidationUtilities.validateMatrixMultiplyDimensions(k, b.count)

        let minDimension = min(m, k, n)
        let useGPU = minDimension >= Self.gpuThreshold && device != nil && commandQueue != nil

        if useGPU {
            return multiplyGPU(a, b, m: m, k: k, n: n)
        }

        return multiplyCPU(a, b, m: m, k: k, n: n)
    }

    /// GPU path for complex matrix multiplication via Metal Performance Shaders.
    @_optimize(speed)
    @_eagerMove
    private func multiplyGPU(
        _ a: [[Complex<Double>]],
        _ b: [[Complex<Double>]],
        m: Int,
        k: Int,
        n: Int,
    ) -> [[Complex<Double>]] {
        guard let device, let commandQueue else {
            return multiplyCPU(a, b, m: m, k: k, n: n)
        }

        var aReal = [Float](unsafeUninitializedCapacity: m * k) {
            buffer, count in
            for i in 0 ..< m {
                for j in 0 ..< k {
                    buffer[i * k + j] = Float(a[i][j].real)
                }
            }
            count = m * k
        }

        var aImag = [Float](unsafeUninitializedCapacity: m * k) {
            buffer, count in
            for i in 0 ..< m {
                for j in 0 ..< k {
                    buffer[i * k + j] = Float(a[i][j].imaginary)
                }
            }
            count = m * k
        }

        var bReal = [Float](unsafeUninitializedCapacity: k * n) {
            buffer, count in
            for i in 0 ..< k {
                for j in 0 ..< n {
                    buffer[i * n + j] = Float(b[i][j].real)
                }
            }
            count = k * n
        }

        var bImag = [Float](unsafeUninitializedCapacity: k * n) {
            buffer, count in
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
            return multiplyCPU(a, b, m: m, k: k, n: n)
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
            return multiplyCPU(a, b, m: m, k: k, n: n)
        }

        if cachedM != m || cachedK != k || cachedN != n {
            cachedM = m; cachedK = k; cachedN = n
            cachedMatMulInit = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: m, resultColumns: n, interiorColumns: k, alpha: 1.0, beta: 0.0)
            cachedMatMulSubtract = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: m, resultColumns: n, interiorColumns: k, alpha: -1.0, beta: 1.0)
            cachedMatMulAdd = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: m, resultColumns: n, interiorColumns: k, alpha: 1.0, beta: 1.0)
        }

        guard let matMulInit = cachedMatMulInit,
              let matMulSubtract = cachedMatMulSubtract,
              let matMulAdd = cachedMatMulAdd
        else {
            return multiplyCPU(a, b, m: m, k: k, n: n)
        }

        matMulInit.encode(commandBuffer: commandBuffer, leftMatrix: aRealMatrix, rightMatrix: bRealMatrix, resultMatrix: cRealMatrix)
        matMulSubtract.encode(commandBuffer: commandBuffer, leftMatrix: aImagMatrix, rightMatrix: bImagMatrix, resultMatrix: cRealMatrix)

        matMulInit.encode(commandBuffer: commandBuffer, leftMatrix: aRealMatrix, rightMatrix: bImagMatrix, resultMatrix: cImagMatrix)
        matMulAdd.encode(commandBuffer: commandBuffer, leftMatrix: aImagMatrix, rightMatrix: bRealMatrix, resultMatrix: cImagMatrix)

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let cRealPtr = cRealBuffer.contents().bindMemory(to: Float.self, capacity: m * n)
        let cImagPtr = cImagBuffer.contents().bindMemory(to: Float.self, capacity: m * n)

        let mn = m * n
        let cRealD = [Double](unsafeUninitializedCapacity: mn) { buffer, count in
            // Safe: baseAddress non-nil for non-zero capacity, mn > 0 guaranteed by empty-input guard
            vDSP_vspdp(cRealPtr, 1, buffer.baseAddress!, 1, vDSP_Length(mn))
            count = mn
        }
        let cImagD = [Double](unsafeUninitializedCapacity: mn) { buffer, count in
            // Safe: baseAddress non-nil for non-zero capacity, mn > 0 guaranteed by empty-input guard
            vDSP_vspdp(cImagPtr, 1, buffer.baseAddress!, 1, vDSP_Length(mn))
            count = mn
        }

        let result = (0 ..< m).map { i in
            [Complex<Double>](unsafeUninitializedCapacity: n) { buffer, count in
                for j in 0 ..< n {
                    let idx = i * n + j
                    buffer[j] = Complex(cRealD[idx], cImagD[idx])
                }
                count = n
            }
        }

        guard result.allSatisfy({ row in row.allSatisfy(\.isFinite) }) else {
            return multiplyCPU(a, b, m: m, k: k, n: n)
        }

        return result
    }

    /// CPU path for complex matrix multiplication via BLAS zgemm.
    @_optimize(speed)
    @_eagerMove
    private func multiplyCPU(
        _ a: [[Complex<Double>]],
        _ b: [[Complex<Double>]],
        m: Int,
        k: Int,
        n: Int,
    ) -> [[Complex<Double>]] {
        let mk2 = m * k * 2
        let kn2 = k * n * 2
        let mn2 = m * n * 2

        var aInterleaved = [Double](unsafeUninitializedCapacity: mk2) {
            buffer, count in
            for col in 0 ..< k {
                for row in 0 ..< m {
                    let idx = (col * m + row) * 2
                    buffer[idx] = a[row][col].real
                    buffer[idx + 1] = a[row][col].imaginary
                }
            }
            count = mk2
        }

        var bInterleaved = [Double](unsafeUninitializedCapacity: kn2) {
            buffer, count in
            for col in 0 ..< n {
                for row in 0 ..< k {
                    let idx = (col * k + row) * 2
                    buffer[idx] = b[row][col].real
                    buffer[idx + 1] = b[row][col].imaginary
                }
            }
            count = kn2
        }

        var resultInterleaved = [Double](unsafeUninitializedCapacity: mn2) {
            _, count in
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

        return (0 ..< m).map { row in
            [Complex<Double>](unsafeUninitializedCapacity: n) { buffer, count in
                for col in 0 ..< n {
                    let idx = (col * m + row) * 2
                    buffer[col] = Complex(resultInterleaved[idx], resultInterleaved[idx + 1])
                }
                count = n
            }
        }
    }
}
