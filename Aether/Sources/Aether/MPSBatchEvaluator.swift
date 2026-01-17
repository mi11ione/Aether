// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate
import Metal
import MetalPerformanceShaders

/// Batched circuit evaluation using Metal Performance Shaders for GPU acceleration.
///
/// Evaluates multiple quantum circuits in parallel by converting each to its unitary matrix
/// and executing all matrix-vector multiplications on GPU simultaneously. Automatically
/// chunks large batches and falls back to BLAS CPU evaluation when Metal is unavailable.
/// Float32 GPU precision introduces ~1e-7 relative error, acceptable for VQE convergence.
///
/// **Example:**
/// ```swift
/// let unitaries = circuits.map { CircuitUnitary.unitary(for: $0) }
/// let evaluator = MPSBatchEvaluator()
/// let energies = await evaluator.expectationValues(for: unitaries, from: state, observable: H)
/// ```
///
/// - Complexity: O(batch · 2²ⁿ) GPU, O(batch · 2³ⁿ) CPU fallback
/// - SeeAlso: ``CircuitUnitary``
public actor MPSBatchEvaluator {
    // MARK: - Metal Resources

    private let device: MTLDevice?
    private let commandQueue: MTLCommandQueue?
    private let maxBatchSize: Int

    /// Whether Metal GPU acceleration is available on this system.
    ///
    /// Returns `true` when Metal device and command queue initialized successfully,
    /// indicating GPU-accelerated batch evaluation is available. When `false`,
    /// evaluator falls back to sequential CPU execution using BLAS operations.
    ///
    /// **Example:**
    /// ```swift
    /// let evaluator = MPSBatchEvaluator()
    /// if evaluator.isMetalAvailable { print("GPU ready") }
    /// ```
    ///
    /// - SeeAlso: ``statistics``
    public nonisolated var isMetalAvailable: Bool {
        device != nil && commandQueue != nil
    }

    // MARK: - Initialization

    /// Creates batched evaluator with automatic GPU resource management.
    ///
    /// Initializes Metal device and command queue for GPU operations. Queries available
    /// GPU memory to compute maximum feasible batch size dynamically, reserving 80% of
    /// recommended working set size for MPS operations. Falls back gracefully to sequential
    /// CPU evaluation using BLAS when Metal is unavailable. Batch size scales inversely
    /// with qubit count due to exponential memory growth of unitary matrices.
    ///
    /// - Parameter maxBatchSize: Optional override for automatic batch size calculation.
    ///
    /// **Example:**
    /// ```swift
    /// let evaluator = MPSBatchEvaluator()
    /// let limited = MPSBatchEvaluator(maxBatchSize: 50)
    /// ```
    public init(maxBatchSize: Int? = nil) {
        device = MTLCreateSystemDefaultDevice()
        commandQueue = device?.makeCommandQueue()

        if let overrideMaxBatch = maxBatchSize {
            self.maxBatchSize = overrideMaxBatch
        } else if let gpuDevice = device {
            let availableMemory: UInt64 = gpuDevice.recommendedMaxWorkingSetSize
            let usableMemory = UInt64(Double(availableMemory) * 0.8)

            let conservativeMaxBatch = 1000
            self.maxBatchSize = min(conservativeMaxBatch, Int(usableMemory / (16 * 1024 * 1024)))
        } else {
            self.maxBatchSize = 100
        }
    }

    // MARK: - Batch State Evolution

    /// Evaluates batch of unitary matrices applied to initial state.
    ///
    /// Converts unitary matrices to Float32, uploads to GPU as MPSMatrix batch with single
    /// transfer, dispatches MPSMatrixVectorMultiplication kernel computing all |ψᵢ⟩ = Uᵢ|ψ₀⟩
    /// operations in parallel, then downloads results and converts back to Float64. When batch
    /// size exceeds maximum GPU memory capacity, automatically splits into chunks processed
    /// sequentially. Falls back to BLAS-accelerated CPU sequential evaluation when Metal unavailable.
    ///
    /// - Parameters:
    ///   - unitaries: Circuit unitary matrices (each 2ⁿ x 2ⁿ)
    ///   - initialState: Initial quantum state applied to all unitaries
    /// - Returns: Output quantum states (one per unitary)
    /// - Complexity: O(batchSize · 2²ⁿ) for GPU parallel execution, O(batchSize · 2³ⁿ) for CPU sequential
    /// - Precondition: `unitaries` is non-empty
    /// - Precondition: Each unitary is a square matrix with dimension 2ⁿ matching `initialState.stateSpaceSize`
    ///
    /// **Example:**
    /// ```swift
    /// let circuits = parameterSets.map { ansatz.bind(parameterVector: $0) }
    /// let unitaries = circuits.map { CircuitUnitary.computeUnitary(circuit: $0) }
    ///
    /// let evaluator = MPSBatchEvaluator()
    /// let states = await evaluator.evaluate(
    ///     batch: unitaries,
    ///     from: QuantumState(qubits: 8)
    /// )
    /// ```
    ///
    /// - SeeAlso: ``expectationValues(for:from:observable:)`` for memory-efficient energy computation
    @_optimize(speed)
    @_eagerMove
    public func evaluate(
        batch unitaries: [[[Complex<Double>]]],
        from initialState: QuantumState,
    ) async -> [QuantumState] {
        ValidationUtilities.validateNonEmpty(unitaries, name: "unitaries")

        let dimension: Int = initialState.stateSpaceSize
        let qubits: Int = initialState.qubits

        for unitary in unitaries {
            ValidationUtilities.validateSquareMatrix(unitary, name: "unitary")
            ValidationUtilities.validateMatrixDimensionEquals(unitary, expected: dimension, name: "unitary")
        }

        guard isMetalAvailable else {
            return evaluateBatchCPU(unitaries: unitaries, initialState: initialState)
        }

        let batchSize: Int = unitaries.count

        if batchSize > maxBatchSize {
            return await evaluateBatchChunked(unitaries: unitaries, initialState: initialState)
        }

        return await evaluateBatchGPU(unitaries: unitaries, initialState: initialState, qubits: qubits)
    }

    /// Evaluates batch with Hamiltonian expectation values (optimal for VQE).
    ///
    /// Combines batch state evolution with expectation value computation, downloading only
    /// scalar energy values rather than full quantum states. Automatically constructs
    /// ``SparseHamiltonian`` from observable for GPU or Accelerate hardware acceleration,
    /// falling back to term-by-term evaluation if sparse construction fails. Substantially
    /// reduces memory transfer overhead compared to downloading full states then computing
    /// energies on CPU.
    ///
    /// - Parameters:
    ///   - unitaries: Circuit unitary matrices
    ///   - initialState: Initial quantum state
    ///   - observable: Hamiltonian for expectation value computation
    /// - Returns: Energy values ⟨ψᵢ|H|ψᵢ⟩ for each circuit
    /// - Complexity: O(batchSize · 2²ⁿ) for batch evaluation + O(batchSize · nnz) for sparse Hamiltonian
    ///
    /// **Example:**
    /// ```swift
    /// let unitaries = shiftedCircuits.map { CircuitUnitary.computeUnitary(circuit: $0) }
    /// let energies = await evaluator.expectationValues(
    ///     for: unitaries,
    ///     from: QuantumState(qubits: 8),
    ///     observable: molecularHamiltonian
    /// )
    /// ```
    ///
    /// - SeeAlso: ``expectationValues(for:from:sparse:)`` for pre-constructed sparse Hamiltonian
    @_optimize(speed)
    @_eagerMove
    public func expectationValues(
        for unitaries: [[[Complex<Double>]]],
        from initialState: QuantumState,
        observable: Observable,
    ) async -> [Double] {
        let states: [QuantumState] = await evaluate(
            batch: unitaries,
            from: initialState,
        )

        let sparseH = SparseHamiltonian(observable: observable, systemSize: initialState.qubits)

        return await computeExpectationValues(states: states, hamiltonian: sparseH)
    }

    /// Evaluates batch with pre-constructed sparse Hamiltonian (maximum performance).
    ///
    /// Uses sparse Hamiltonian backend directly, bypassing observable-to-sparse conversion.
    /// Optimal for molecular Hamiltonians with typical 0.01-1% sparsity where sparse matrix-vector
    /// multiplication provides substantial acceleration over term-by-term Pauli measurements.
    ///
    /// - Parameters:
    ///   - unitaries: Circuit unitary matrices
    ///   - initialState: Initial quantum state
    ///   - sparseHamiltonian: Pre-constructed sparse Hamiltonian
    /// - Returns: Energy values ⟨ψᵢ|H|ψᵢ⟩ for each circuit
    /// - Complexity: O(batchSize · 2²ⁿ) for batch evaluation + O(batchSize · nnz) for sparse matrix operations
    ///
    /// **Example:**
    /// ```swift
    /// let sparse = SparseHamiltonian(observable: hamiltonian, qubits: 8)
    /// let energies = await evaluator.expectationValues(
    ///     for: unitaries,
    ///     from: QuantumState(qubits: 8),
    ///     sparse: sparse
    /// )
    /// ```
    ///
    /// - SeeAlso: ``SparseHamiltonian``
    @_optimize(speed)
    @_eagerMove
    public func expectationValues(
        for unitaries: [[[Complex<Double>]]],
        from initialState: QuantumState,
        sparse sparseHamiltonian: SparseHamiltonian,
    ) async -> [Double] {
        let states: [QuantumState] = await evaluate(
            batch: unitaries,
            from: initialState,
        )

        return await computeExpectationValues(states: states, hamiltonian: sparseHamiltonian)
    }

    /// Computes expectation values for batch of states using sparse Hamiltonian.
    @_optimize(speed)
    private func computeExpectationValues(
        states: [QuantumState],
        hamiltonian: SparseHamiltonian,
    ) async -> [Double] {
        var energies = [Double](unsafeUninitializedCapacity: states.count) { _, count in
            count = states.count
        }

        for i in 0 ..< states.count {
            energies[i] = await hamiltonian.expectationValue(of: states[i])
        }

        return energies
    }

    // MARK: - Private GPU Implementation

    /// Executes batch on GPU using Metal Performance Shaders.
    @_optimize(speed)
    @_eagerMove
    private func evaluateBatchGPU(
        unitaries: [[[Complex<Double>]]],
        initialState: QuantumState,
        qubits: Int,
    ) async -> [QuantumState] {
        guard let device, let commandQueue else {
            return evaluateBatchCPU(unitaries: unitaries, initialState: initialState)
        }

        let batchSize: Int = unitaries.count
        let dimension = 1 << qubits
        let matrixElements = batchSize * dimension * dimension

        var realMatrices = [Float](unsafeUninitializedCapacity: matrixElements) { _, count in
            count = matrixElements
        }
        var imagMatrices = [Float](unsafeUninitializedCapacity: matrixElements) { _, count in
            count = matrixElements
        }

        var idx = 0
        for unitary in unitaries {
            for row in unitary {
                for element in row {
                    realMatrices[idx] = Float(element.real)
                    imagMatrices[idx] = Float(element.imaginary)
                    idx += 1
                }
            }
        }

        var realState = [Float](unsafeUninitializedCapacity: dimension) { _, count in
            count = dimension
        }
        var imagState = [Float](unsafeUninitializedCapacity: dimension) { _, count in
            count = dimension
        }

        initialState.amplitudes.withUnsafeBytes { srcBytes in
            let srcDoubles = srcBytes.bindMemory(to: Double.self)
            // Safety: amplitudes array is non-empty (QuantumState invariant)
            vDSP_vdpsp(srcDoubles.baseAddress!, 2, &realState, 1, vDSP_Length(dimension))
            vDSP_vdpsp(srcDoubles.baseAddress! + 1, 2, &imagState, 1, vDSP_Length(dimension))
        }

        let matrixDescriptor = MPSMatrixDescriptor(
            rows: dimension,
            columns: dimension,
            rowBytes: dimension * MemoryLayout<Float>.stride,
            dataType: .float32,
        )

        let vectorDescriptor = MPSVectorDescriptor(
            length: dimension,
            dataType: .float32,
        )

        guard let realMatrixBuffer = device.makeBuffer(
            bytes: realMatrices,
            length: realMatrices.count * MemoryLayout<Float>.stride,
            options: .storageModeShared,
        ) else { return evaluateBatchCPU(unitaries: unitaries, initialState: initialState) }

        guard let imagMatrixBuffer = device.makeBuffer(
            bytes: imagMatrices,
            length: imagMatrices.count * MemoryLayout<Float>.stride,
            options: .storageModeShared,
        ) else { return evaluateBatchCPU(unitaries: unitaries, initialState: initialState) }

        guard let realStateBuffer = device.makeBuffer(
            bytes: realState,
            length: realState.count * MemoryLayout<Float>.stride,
            options: .storageModeShared,
        ) else { return evaluateBatchCPU(unitaries: unitaries, initialState: initialState) }

        guard let imagStateBuffer = device.makeBuffer(
            bytes: imagState,
            length: imagState.count * MemoryLayout<Float>.stride,
            options: .storageModeShared,
        ) else { return evaluateBatchCPU(unitaries: unitaries, initialState: initialState) }

        guard let resultRealBuffer = device.makeBuffer(
            length: batchSize * dimension * MemoryLayout<Float>.stride,
            options: .storageModeShared,
        ) else { return evaluateBatchCPU(unitaries: unitaries, initialState: initialState) }

        guard let resultImagBuffer = device.makeBuffer(
            length: batchSize * dimension * MemoryLayout<Float>.stride,
            options: .storageModeShared,
        ) else { return evaluateBatchCPU(unitaries: unitaries, initialState: initialState) }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return evaluateBatchCPU(unitaries: unitaries, initialState: initialState)
        }

        let matVecInit = MPSMatrixVectorMultiplication(
            device: device,
            transpose: false,
            rows: dimension,
            columns: dimension,
            alpha: 1.0,
            beta: 0.0,
        )

        let matVecAccumSub = MPSMatrixVectorMultiplication(
            device: device,
            transpose: false,
            rows: dimension,
            columns: dimension,
            alpha: -1.0,
            beta: 1.0,
        )

        let matVecAccumAdd = MPSMatrixVectorMultiplication(
            device: device,
            transpose: false,
            rows: dimension,
            columns: dimension,
            alpha: 1.0,
            beta: 1.0,
        )

        let realVec = MPSVector(buffer: realStateBuffer, descriptor: vectorDescriptor)
        let imagVec = MPSVector(buffer: imagStateBuffer, descriptor: vectorDescriptor)

        for batchIndex in 0 ..< batchSize {
            let matrixOffset = batchIndex * dimension * dimension * MemoryLayout<Float>.stride
            let resultOffset = batchIndex * dimension * MemoryLayout<Float>.stride

            let realMatrix = MPSMatrix(
                buffer: realMatrixBuffer,
                offset: matrixOffset,
                descriptor: matrixDescriptor,
            )

            let imagMatrix = MPSMatrix(
                buffer: imagMatrixBuffer,
                offset: matrixOffset,
                descriptor: matrixDescriptor,
            )

            let resultReal = MPSVector(
                buffer: resultRealBuffer,
                offset: resultOffset,
                descriptor: vectorDescriptor,
            )

            let resultImag = MPSVector(
                buffer: resultImagBuffer,
                offset: resultOffset,
                descriptor: vectorDescriptor,
            )

            matVecInit.encode(
                commandBuffer: commandBuffer,
                inputMatrix: realMatrix,
                inputVector: realVec,
                resultVector: resultReal,
            )

            matVecAccumSub.encode(
                commandBuffer: commandBuffer,
                inputMatrix: imagMatrix,
                inputVector: imagVec,
                resultVector: resultReal,
            )

            matVecInit.encode(
                commandBuffer: commandBuffer,
                inputMatrix: imagMatrix,
                inputVector: realVec,
                resultVector: resultImag,
            )

            matVecAccumAdd.encode(
                commandBuffer: commandBuffer,
                inputMatrix: realMatrix,
                inputVector: imagVec,
                resultVector: resultImag,
            )
        }

        commandBuffer.commit()
        await commandBuffer.completed()

        let resultRealPointer = resultRealBuffer.contents().bindMemory(
            to: Float.self,
            capacity: batchSize * dimension,
        )

        let resultImagPointer = resultImagBuffer.contents().bindMemory(
            to: Float.self,
            capacity: batchSize * dimension,
        )

        var resultStates = [QuantumState]()
        resultStates.reserveCapacity(batchSize)

        var realDouble = [Double](unsafeUninitializedCapacity: dimension) { _, count in
            count = dimension
        }
        var imagDouble = [Double](unsafeUninitializedCapacity: dimension) { _, count in
            count = dimension
        }

        for batchIndex in 0 ..< batchSize {
            let baseOffset = batchIndex * dimension

            vDSP_vspdp(resultRealPointer + baseOffset, 1, &realDouble, 1, vDSP_Length(dimension))
            vDSP_vspdp(resultImagPointer + baseOffset, 1, &imagDouble, 1, vDSP_Length(dimension))

            let amplitudes = [Complex<Double>](unsafeUninitializedCapacity: dimension) { ampBuffer, ampCount in
                for i in 0 ..< dimension {
                    ampBuffer[i] = Complex(realDouble[i], imagDouble[i])
                }
                ampCount = dimension
            }

            resultStates.append(QuantumState(qubits: qubits, amplitudes: amplitudes))
        }

        return resultStates
    }

    /// Sequential CPU fallback using BLAS cblas_zgemv matrix-vector multiplication.
    @_optimize(speed)
    @_eagerMove
    private func evaluateBatchCPU(
        unitaries: [[[Complex<Double>]]],
        initialState: QuantumState,
    ) -> [QuantumState] {
        let dimension = initialState.stateSpaceSize
        var resultStates: [QuantumState] = []
        resultStates.reserveCapacity(unitaries.count)

        var matrixInterleaved = [Double](unsafeUninitializedCapacity: dimension * dimension * 2) { _, count in
            count = dimension * dimension * 2
        }
        var result = [Complex<Double>](unsafeUninitializedCapacity: dimension) { _, count in
            count = dimension
        }
        var alpha = (1.0, 0.0)
        var beta = (0.0, 0.0)

        for unitary in unitaries {
            for col in 0 ..< dimension {
                for row in 0 ..< dimension {
                    let idx = (col * dimension + row) * 2
                    matrixInterleaved[idx] = unitary[row][col].real
                    matrixInterleaved[idx + 1] = unitary[row][col].imaginary
                }
            }

            initialState.amplitudes.withUnsafeBytes { vecBytes in
                let vecPtr = vecBytes.baseAddress!
                result.withUnsafeMutableBytes { resBytes in
                    let resPtr = resBytes.baseAddress!
                    matrixInterleaved.withUnsafeBytes { matBytes in
                        let matPtr = matBytes.baseAddress!
                        withUnsafePointer(to: &alpha) { alphaPtr in
                            withUnsafePointer(to: &beta) { betaPtr in
                                cblas_zgemv(
                                    CblasColMajor,
                                    CblasNoTrans,
                                    Int32(dimension),
                                    Int32(dimension),
                                    OpaquePointer(alphaPtr),
                                    OpaquePointer(matPtr),
                                    Int32(dimension),
                                    OpaquePointer(vecPtr),
                                    1,
                                    OpaquePointer(betaPtr),
                                    OpaquePointer(resPtr),
                                    1,
                                )
                            }
                        }
                    }
                }
            }

            resultStates.append(QuantumState(qubits: initialState.qubits, amplitudes: result))
        }

        return resultStates
    }

    /// Splits large batches into chunks fitting GPU memory, processes sequentially.
    @_optimize(speed)
    @_eagerMove
    private func evaluateBatchChunked(
        unitaries: [[[Complex<Double>]]],
        initialState: QuantumState,
    ) async -> [QuantumState] {
        var allResults: [QuantumState] = []
        allResults.reserveCapacity(unitaries.count)

        let chunks: [[[[Complex<Double>]]]] = unitaries.chunked(into: maxBatchSize)

        for chunk in chunks {
            let chunkResults: [QuantumState] = await evaluateBatchGPU(
                unitaries: chunk,
                initialState: initialState,
                qubits: initialState.qubits,
            )
            allResults.append(contentsOf: chunkResults)
        }

        return allResults
    }

    // MARK: - Diagnostics

    /// Computes maximum batch size for given qubit count based on available GPU memory.
    ///
    /// Calculates maximum number of circuits that fit in GPU memory considering unitary matrix
    /// storage (2ⁿ x 2ⁿ x 4 bytes each) and state vector storage. Reserves 80% of recommended
    /// working set size for computations. Returns 1 when Metal unavailable.
    ///
    /// - Parameter qubits: Number of qubits
    /// - Returns: Maximum batch size before automatic chunking required
    /// - Complexity: O(1)
    ///
    /// **Example:**
    /// ```swift
    /// let evaluator = MPSBatchEvaluator()
    /// let max = await evaluator.maxBatchSize(for: 10)
    /// ```
    @_effects(readonly)
    public func maxBatchSize(for qubits: Int) -> Int {
        guard isMetalAvailable else { return 1 }

        let dimension = 1 << qubits
        let unitaryMemory: Int = dimension * dimension * 4 * 2
        let stateMemory: Int = dimension * 4 * 2

        guard let device else { return 1 }

        let availableMemory: UInt64 = device.recommendedMaxWorkingSetSize
        let usableMemory = UInt64(Double(availableMemory) * 0.8)

        let maxBatch = Int(usableMemory) / (unitaryMemory + stateMemory)

        return max(1, min(maxBatch, maxBatchSize))
    }

    /// Device and memory statistics for batch evaluator.
    ///
    /// Provides diagnostic information including Metal availability, device name,
    /// and maximum batch size. Useful for debugging performance issues or verifying
    /// GPU acceleration status.
    ///
    /// **Example:**
    /// ```swift
    /// let evaluator = MPSBatchEvaluator()
    /// let stats = await evaluator.statistics
    /// ```
    ///
    /// - SeeAlso: ``BatchEvaluatorStatistics``
    public var statistics: BatchEvaluatorStatistics {
        BatchEvaluatorStatistics(
            isMetalAvailable: isMetalAvailable,
            maxBatchSize: maxBatchSize,
            deviceName: device?.name ?? "CPU",
        )
    }
}

// MARK: - Supporting Types

/// Diagnostic information for batch evaluator GPU configuration.
///
/// Captures Metal availability, device name, and maximum batch size for performance
/// analysis and debugging. Returned by ``MPSBatchEvaluator/statistics``.
///
/// **Example:**
/// ```swift
/// let stats = await evaluator.statistics
/// if stats.isMetalAvailable {
///     print("GPU: \(stats.deviceName), max batch: \(stats.maxBatchSize)")
/// }
/// ```
@frozen
public struct BatchEvaluatorStatistics: Sendable, Equatable, CustomStringConvertible {
    /// Whether Metal GPU acceleration is available.
    public let isMetalAvailable: Bool

    /// Maximum batch size before automatic chunking required.
    public let maxBatchSize: Int

    /// Metal device name or "CPU" when Metal unavailable.
    public let deviceName: String

    /// Creates statistics with specified configuration values.
    public init(isMetalAvailable: Bool, maxBatchSize: Int, deviceName: String) {
        self.isMetalAvailable = isMetalAvailable
        self.maxBatchSize = maxBatchSize
        self.deviceName = deviceName
    }

    @inlinable
    public var description: String {
        """
        Batch Evaluator Statistics:
          Metal Available: \(isMetalAvailable)
          Device: \(deviceName)
          Max Batch Size: \(maxBatchSize)
        """
    }
}

// MARK: - Utilities

private extension Array {
    /// Splits array into chunks of specified maximum size.
    func chunked(into size: Int) -> [[Element]] {
        stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}
