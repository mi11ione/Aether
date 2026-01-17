// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Sparse Hamiltonian representation for accelerated expectation value computation in variational algorithms.
///
/// Converts Pauli decomposition H = Σᵢ cᵢ Pᵢ into sparse matrix format for efficient computation. Molecular
/// Hamiltonians exhibit 0.01% to 1% sparsity, storing only non-zero elements enables orders of magnitude
/// speedup over term-by-term measurement in VQE and QAOA.
///
/// Backend selection occurs automatically at initialization based on system size and hardware availability.
/// Metal GPU backend activates for larger systems when Metal is available, using custom CSR kernels with
/// Float32 precision. Accelerate Sparse backend handles smaller systems using AMX-accelerated sparse BLAS
/// on decomposed real/imaginary components. Observable backend provides guaranteed fallback with term-by-term
/// measurement.
///
/// **Example:**
///
/// ```swift
/// let hamiltonian = Observable(terms: [(1.0, PauliString([.z(0)]))])
/// let sparse = SparseHamiltonian(observable: hamiltonian)
/// let state = QuantumState(qubits: 1)
/// let energy = await sparse.expectationValue(of: state)
/// ```
///
/// - SeeAlso: ``Observable``
/// - SeeAlso: ``VQE``
/// - SeeAlso: ``QAOA``
public actor SparseHamiltonian {
    private enum Backend {
        case metalGPU(
            device: MTLDevice,
            commandQueue: MTLCommandQueue,
            pipelineState: MTLComputePipelineState,
            rowPointers: MTLBuffer,
            columnIndices: MTLBuffer,
            values: MTLBuffer,
        )

        case accelerateSparse(
            realMatrix: SparseMatrix_Double,
            imagMatrix: SparseMatrix_Double,
            dimension: Int,
        )

        case observable(Observable)

        var description: String {
            switch self {
            case .metalGPU: "Metal GPU"
            case .accelerateSparse: "Accelerate Sparse (AMX)"
            case .observable: "Observable (fallback)"
            }
        }
    }

    // MARK: - Properties

    /// Number of qubits in the quantum system.
    public nonisolated let qubits: Int

    /// Hilbert space dimension 2^qubits.
    public nonisolated let dimension: Int

    /// Count of non-zero matrix elements, typically 0.01% to 1% of dimension² for molecular Hamiltonians.
    public nonisolated let nnz: Int

    /// Sparsity ratio (nnz / dimension²).
    @inlinable
    public nonisolated var sparsity: Double {
        Double(nnz) / Double(dimension * dimension)
    }

    private let backend: Backend
    private let observable: Observable

    // MARK: - Initialization

    /// Creates sparse Hamiltonian representation with automatic backend selection.
    ///
    /// Backend selection priority: Metal GPU (larger systems) -> Accelerate Sparse (smaller systems) ->
    /// Observable fallback. The sparse representation persists across all VQE iterations, amortizing
    /// construction cost over the entire optimization.
    ///
    /// **Example:**
    ///
    /// ```swift
    /// let h = Observable(terms: [(1.0, PauliString([.z(0)]))])
    /// let sparse = SparseHamiltonian(observable: h)
    /// let stats = await sparse.statistics
    /// ```
    ///
    /// - Parameters:
    ///   - observable: Quantum observable H = Σᵢ cᵢ Pᵢ.
    ///   - systemSize: Total qubits in system. When nil, inferred from maximum qubit index in observable.
    public init(observable: Observable, systemSize: Int? = nil) {
        self.observable = observable

        var maxQubit: Int = -1
        for (_, pauliString) in observable.terms {
            for op in pauliString.operators {
                maxQubit = max(maxQubit, op.qubit)
            }
        }

        qubits = systemSize ?? max(maxQubit + 1, 1)
        dimension = 1 << qubits

        let cooMatrix: [COOElement] = Self.buildCOOMatrix(from: observable, dimension: dimension)
        nnz = cooMatrix.count

        if qubits >= 8, let metalBackend = Self.tryMetalGPUBackend(
            cooMatrix: cooMatrix,
            dimension: dimension,
        ) {
            backend = metalBackend
        } else if let accelerateBackend = Self.tryAccelerateSparseBackend(
            cooMatrix: cooMatrix,
            dimension: dimension,
        ) {
            backend = accelerateBackend
        } else {
            backend = .observable(observable)
        }
    }

    // MARK: - Sparse Matrix Construction (COO Format)

    /// Sparse matrix element in coordinate format.
    private struct COOElement {
        let row: Int
        let col: Int
        let value: Complex<Double>
    }

    /// Builds COO matrix from observable terms.
    /// - Complexity: O(terms * 2ⁿ) construction, O(nnz log nnz) sorting
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func buildCOOMatrix(
        from observable: Observable,
        dimension: Int,
    ) -> [COOElement] {
        var elements: [MatrixIndex: Complex<Double>] = [:]
        elements.reserveCapacity(observable.terms.count * dimension)

        for (coefficient, pauliString) in observable.terms {
            let pauliMatrix = pauliStringToSparseMatrix(
                pauliString,
                dimension: dimension,
            )

            for element in pauliMatrix {
                let index = MatrixIndex(row: element.row, col: element.col)
                elements[index, default: .zero] += coefficient * element.value
            }
        }

        let tolerance = 1e-12
        let nonZeros: [COOElement] = elements.compactMap { index, value -> COOElement? in
            guard abs(value.magnitude) > tolerance else { return nil }
            return COOElement(row: index.row, col: index.col, value: value)
        }

        return nonZeros.sorted { lhs, rhs in
            if lhs.row != rhs.row {
                return lhs.row < rhs.row
            }
            return lhs.col < rhs.col
        }
    }

    /// Converts Pauli string to sparse matrix elements.
    /// - Complexity: O(2ⁿ) for all Pauli strings (exactly 2ⁿ non-zeros per string)
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func pauliStringToSparseMatrix(
        _ pauliString: PauliString,
        dimension: Int,
    ) -> [COOElement] {
        var elements: [COOElement] = []
        elements.reserveCapacity(dimension)

        for row in 0 ..< dimension {
            let (col, phase) = pauliString.applyToRow(row: row)
            elements.append(COOElement(row: row, col: col, value: phase))
        }

        return elements
    }

    // MARK: - Backend Construction

    /// Attempts Metal GPU backend construction.
    private static func tryMetalGPUBackend(
        cooMatrix: [COOElement],
        dimension: Int,
    ) -> Backend? {
        guard !cooMatrix.isEmpty else { return nil }
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        guard let commandQueue = device.makeCommandQueue() else { return nil }
        guard let library = MetalUtilities.loadLibrary(device: device) else { return nil }
        guard let kernelFunction = library.makeFunction(name: "csrSparseMatVec") else { return nil }
        guard let pipelineState = try? device.makeComputePipelineState(function: kernelFunction) else { return nil }

        let (rowPointers, columnIndices, values): ([UInt32], [UInt32], [Complex<Double>]) = convertCOOtoCSR(
            cooMatrix: cooMatrix,
            numRows: dimension,
        )

        let float32Values: [(Float, Float)] = convertComplexToFloat32Pairs(values)

        let storageMode: MTLResourceOptions = .storageModeShared

        guard let rowPointerBuffer = device.makeBuffer(
            bytes: rowPointers,
            length: rowPointers.count * MemoryLayout<UInt32>.stride,
            options: storageMode,
        ) else { return nil }

        guard let columnIndexBuffer = device.makeBuffer(
            bytes: columnIndices,
            length: columnIndices.count * MemoryLayout<UInt32>.stride,
            options: storageMode,
        ) else { return nil }

        guard let valueBuffer = device.makeBuffer(
            bytes: float32Values,
            length: float32Values.count * MemoryLayout<(Float, Float)>.stride,
            options: storageMode,
        ) else { return nil }

        return .metalGPU(
            device: device,
            commandQueue: commandQueue,
            pipelineState: pipelineState,
            rowPointers: rowPointerBuffer,
            columnIndices: columnIndexBuffer,
            values: valueBuffer,
        )
    }

    /// Converts COO to CSR format.
    /// - Complexity: O(nnz) linear pass with prefix sum
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func convertCOOtoCSR(
        cooMatrix: [COOElement],
        numRows: Int,
    ) -> (rowPointers: [UInt32], columnIndices: [UInt32], values: [Complex<Double>]) {
        let nnz = cooMatrix.count

        var rowPointers = [UInt32](unsafeUninitializedCapacity: numRows + 1) { buffer, count in
            buffer.initialize(repeating: 0)
            count = numRows + 1
        }

        var columnIndices = [UInt32]()
        columnIndices.reserveCapacity(nnz)
        var values = [Complex<Double>]()
        values.reserveCapacity(nnz)

        for element in cooMatrix {
            rowPointers[element.row + 1] += 1
        }

        for i in 1 ... numRows {
            rowPointers[i] += rowPointers[i - 1]
        }

        for element in cooMatrix {
            columnIndices.append(UInt32(element.col))
            values.append(element.value)
        }

        return (rowPointers, columnIndices, values)
    }

    /// Attempts Accelerate Sparse backend construction.
    @_effects(readonly)
    private static func tryAccelerateSparseBackend(
        cooMatrix: [COOElement],
        dimension: Int,
    ) -> Backend? {
        guard !cooMatrix.isEmpty else {
            return nil
        }

        let nnz: Int = cooMatrix.count

        var realRows: [Int32] = []
        var realCols: [Int32] = []
        var realVals: [Double] = []

        var imagRows: [Int32] = []
        var imagCols: [Int32] = []
        var imagVals: [Double] = []

        realRows.reserveCapacity(nnz)
        realCols.reserveCapacity(nnz)
        realVals.reserveCapacity(nnz)
        imagRows.reserveCapacity(nnz)
        imagCols.reserveCapacity(nnz)
        imagVals.reserveCapacity(nnz)

        for element in cooMatrix {
            let row = Int32(element.row)
            let col = Int32(element.col)

            if abs(element.value.real) > 1e-15 {
                realRows.append(row)
                realCols.append(col)
                realVals.append(element.value.real)
            }

            if abs(element.value.imaginary) > 1e-15 {
                imagRows.append(row)
                imagCols.append(col)
                imagVals.append(element.value.imaginary)
            }
        }

        let realMatrix = buildAccelerateSparseMatrix(
            rows: realRows,
            cols: realCols,
            values: realVals,
            dimension: dimension,
        )

        let imagMatrix = buildAccelerateSparseMatrix(
            rows: imagRows,
            cols: imagCols,
            values: imagVals,
            dimension: dimension,
        )

        return .accelerateSparse(
            realMatrix: realMatrix,
            imagMatrix: imagMatrix,
            dimension: dimension,
        )
    }

    /// Builds Accelerate sparse matrix from COO components.
    @_eagerMove
    @_effects(readonly)
    private static func buildAccelerateSparseMatrix(
        rows: [Int32],
        cols: [Int32],
        values: [Double],
        dimension: Int,
    ) -> SparseMatrix_Double {
        let nnz: Int = rows.count

        guard nnz > 0 else {
            var emptyRows: [Int32] = [0]
            var emptyCols: [Int32] = [0]
            var emptyVals = [0.0]

            return emptyRows.withUnsafeMutableBufferPointer { rowPtr in
                emptyCols.withUnsafeMutableBufferPointer { colPtr in
                    emptyVals.withUnsafeMutableBufferPointer { valPtr in
                        // Safety: Arrays are non-empty (1 element each), baseAddress always valid
                        SparseConvertFromCoordinate(
                            Int32(dimension),
                            Int32(dimension),
                            1,
                            1,
                            SparseAttributes_t(),
                            rowPtr.baseAddress!,
                            colPtr.baseAddress!,
                            valPtr.baseAddress!,
                        )
                    }
                }
            }
        }

        var mutableRows = rows
        var mutableCols = cols
        var mutableVals = values

        return mutableRows.withUnsafeMutableBufferPointer { rowPtr in
            mutableCols.withUnsafeMutableBufferPointer { colPtr in
                mutableVals.withUnsafeMutableBufferPointer { valPtr in
                    // Safety: Arrays verified non-empty above, baseAddress always valid
                    SparseConvertFromCoordinate(
                        Int32(dimension),
                        Int32(dimension),
                        nnz,
                        1,
                        SparseAttributes_t(),
                        rowPtr.baseAddress!,
                        colPtr.baseAddress!,
                        valPtr.baseAddress!,
                    )
                }
            }
        }
    }

    // MARK: - Conversion Helpers

    /// Converts complex Double array to Float32 tuple pairs for Metal GPU.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func convertComplexToFloat32Pairs(_ complexArray: [Complex<Double>]) -> [(Float, Float)] {
        complexArray.withUnsafeBufferPointer { complexBuffer in
            guard let baseAddress = complexBuffer.baseAddress else { return [] }
            let doublePtr = UnsafeRawPointer(baseAddress)
                .assumingMemoryBound(to: Double.self)
            let doubleCount = complexArray.count * 2

            var floatBuffer = [Float](unsafeUninitializedCapacity: doubleCount) { buffer, count in
                // Safety: Buffer allocated with doubleCount capacity, baseAddress valid
                vDSP_vdpsp(doublePtr, 1, buffer.baseAddress!, 1, vDSP_Length(doubleCount))
                count = doubleCount
            }

            return floatBuffer.withUnsafeMutableBufferPointer { floatPtr in
                // Safety: floatBuffer initialized above, baseAddress valid
                let gpuPtr = UnsafeRawPointer(floatPtr.baseAddress!)
                    .assumingMemoryBound(to: (Float, Float).self)
                return Array(UnsafeBufferPointer(start: gpuPtr, count: complexArray.count))
            }
        }
    }

    /// Decomposes quantum state into separate real and imaginary Double arrays.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func decomposeStateToRealImag(_ state: QuantumState) -> (real: [Double], imag: [Double]) {
        let dimension = state.amplitudes.count
        let stateReal = [Double](unsafeUninitializedCapacity: dimension) { buffer, count in
            for i in 0 ..< dimension {
                buffer[i] = state.amplitudes[i].real
            }
            count = dimension
        }
        let stateImag = [Double](unsafeUninitializedCapacity: dimension) { buffer, count in
            for i in 0 ..< dimension {
                buffer[i] = state.amplitudes[i].imaginary
            }
            count = dimension
        }
        return (stateReal, stateImag)
    }

    // MARK: - Expectation Value Computation

    /// Computes ⟨ψ|H|ψ⟩ using sparse matrix-vector multiplication.
    ///
    /// Performs H|ψ⟩ via sparse matvec, then computes inner product ⟨ψ|H|ψ⟩. Thread-safe for concurrent
    /// calls with different states.
    ///
    /// **Example:**
    ///
    /// ```swift
    /// let state = QuantumState(qubits: 4)
    /// let energy = await sparse.expectationValue(of: state)
    /// ```
    ///
    /// - Parameter state: Normalized quantum state |ψ⟩.
    /// - Returns: Real expectation value ⟨ψ|H|ψ⟩.
    /// - Complexity: O(nnz) for sparse matvec, O(2ⁿ) for inner product
    /// - Precondition: State must have ``qubits`` qubits.
    /// - Precondition: State must be normalized.
    public func expectationValue(of state: QuantumState) -> Double {
        ValidationUtilities.validateStateQubitCount(state, required: qubits, exact: true)
        ValidationUtilities.validateNormalizedState(state)

        switch backend {
        case let .metalGPU(device, commandQueue, pipelineState, rowPointers, columnIndices, values):
            return computeMetalGPU(
                state: state,
                device: device,
                commandQueue: commandQueue,
                pipelineState: pipelineState,
                rowPointers: rowPointers,
                columnIndices: columnIndices,
                values: values,
            )

        case let .accelerateSparse(realMatrix, imagMatrix, dimension):
            return computeAccelerateSparse(
                state: state,
                realMatrix: realMatrix,
                imagMatrix: imagMatrix,
                dimension: dimension,
            )

        case let .observable(obs):
            return obs.expectationValue(of: state)
        }
    }

    /// Computes expectation value using Metal GPU backend.
    /// - Complexity: O(nnz) for sparse matvec, O(2ⁿ) for inner product
    @_optimize(speed)
    private func computeMetalGPU(
        state: QuantumState,
        device: MTLDevice,
        commandQueue: MTLCommandQueue,
        pipelineState: MTLComputePipelineState,
        rowPointers: MTLBuffer,
        columnIndices: MTLBuffer,
        values: MTLBuffer,
    ) -> Double {
        let float32State: [(Float, Float)] = Self.convertComplexToFloat32Pairs(state.amplitudes)

        guard let inputBuffer = device.makeBuffer(
            bytes: float32State,
            length: dimension * MemoryLayout<(Float, Float)>.stride,
            options: .storageModeShared,
        ) else { return observable.expectationValue(of: state) }

        guard let outputBuffer = device.makeBuffer(
            length: dimension * MemoryLayout<(Float, Float)>.stride,
            options: .storageModeShared,
        ) else { return observable.expectationValue(of: state) }

        guard let (commandBuffer, encoder) = MetalUtilities.createCommandEncoder(queue: commandQueue) else {
            return observable.expectationValue(of: state)
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(rowPointers, offset: 0, index: 0)
        encoder.setBuffer(columnIndices, offset: 0, index: 1)
        encoder.setBuffer(values, offset: 0, index: 2)
        encoder.setBuffer(inputBuffer, offset: 0, index: 3)
        encoder.setBuffer(outputBuffer, offset: 0, index: 4)
        var dim = UInt32(dimension)
        encoder.setBytes(&dim, length: MemoryLayout<UInt32>.stride, index: 5)

        let threadGroupSize: Int = min(pipelineState.maxTotalThreadsPerThreadgroup, dimension)
        let threadGroups: Int = (dimension + threadGroupSize - 1) / threadGroupSize

        encoder.dispatchThreadgroups(
            MTLSize(width: threadGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1),
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Safety: outputBuffer created successfully above, contents always valid
        let outputPointer = outputBuffer.contents().bindMemory(
            to: (Float, Float).self,
            capacity: dimension,
        )

        let gpuOutput = Array(UnsafeBufferPointer(start: outputPointer, count: dimension))

        let hPsiReal = [Double](unsafeUninitializedCapacity: dimension) { buffer, count in
            for i in 0 ..< dimension {
                buffer[i] = Double(gpuOutput[i].0)
            }
            count = dimension
        }
        let hPsiImag = [Double](unsafeUninitializedCapacity: dimension) { buffer, count in
            for i in 0 ..< dimension {
                buffer[i] = Double(gpuOutput[i].1)
            }
            count = dimension
        }

        let (stateReal, stateImag) = Self.decomposeStateToRealImag(state)

        var realDot1 = 0.0
        var realDot2 = 0.0
        vDSP_dotprD(stateReal, 1, hPsiReal, 1, &realDot1, vDSP_Length(dimension))
        vDSP_dotprD(stateImag, 1, hPsiImag, 1, &realDot2, vDSP_Length(dimension))

        return realDot1 + realDot2
    }

    /// Computes expectation value using Accelerate Sparse backend.
    /// - Complexity: O(nnz) for sparse matvec, O(2ⁿ) for inner product
    @_optimize(speed)
    private func computeAccelerateSparse(
        state: QuantumState,
        realMatrix: SparseMatrix_Double,
        imagMatrix: SparseMatrix_Double,
        dimension: Int,
    ) -> Double {
        var (stateReal, stateImag) = Self.decomposeStateToRealImag(state)

        var resultReal = [Double](unsafeUninitializedCapacity: dimension) { buffer, count in
            buffer.initialize(repeating: 0.0)
            count = dimension
        }
        var resultImag = [Double](unsafeUninitializedCapacity: dimension) { buffer, count in
            buffer.initialize(repeating: 0.0)
            count = dimension
        }

        var ax = [Double](unsafeUninitializedCapacity: dimension) { buffer, count in
            buffer.initialize(repeating: 0.0)
            count = dimension
        }
        var ay = [Double](unsafeUninitializedCapacity: dimension) { buffer, count in
            buffer.initialize(repeating: 0.0)
            count = dimension
        }
        var bx = [Double](unsafeUninitializedCapacity: dimension) { buffer, count in
            buffer.initialize(repeating: 0.0)
            count = dimension
        }
        var by = [Double](unsafeUninitializedCapacity: dimension) { buffer, count in
            buffer.initialize(repeating: 0.0)
            count = dimension
        }

        stateReal.withUnsafeMutableBufferPointer { xPtr in
            ax.withUnsafeMutableBufferPointer { axPtr in
                // Safety: Arrays allocated with dimension capacity, baseAddress valid
                let xVec = DenseVector_Double(count: Int32(dimension), data: xPtr.baseAddress!)
                let axVec = DenseVector_Double(count: Int32(dimension), data: axPtr.baseAddress!)
                SparseMultiply(realMatrix, xVec, axVec)
            }
        }

        stateImag.withUnsafeMutableBufferPointer { yPtr in
            ay.withUnsafeMutableBufferPointer { ayPtr in
                // Safety: Arrays allocated with dimension capacity, baseAddress valid
                let yVec = DenseVector_Double(count: Int32(dimension), data: yPtr.baseAddress!)
                let ayVec = DenseVector_Double(count: Int32(dimension), data: ayPtr.baseAddress!)
                SparseMultiply(realMatrix, yVec, ayVec)
            }
        }

        stateReal.withUnsafeMutableBufferPointer { xPtr in
            bx.withUnsafeMutableBufferPointer { bxPtr in
                // Safety: Arrays allocated with dimension capacity, baseAddress valid
                let xVec = DenseVector_Double(count: Int32(dimension), data: xPtr.baseAddress!)
                let bxVec = DenseVector_Double(count: Int32(dimension), data: bxPtr.baseAddress!)
                SparseMultiply(imagMatrix, xVec, bxVec)
            }
        }

        stateImag.withUnsafeMutableBufferPointer { yPtr in
            by.withUnsafeMutableBufferPointer { byPtr in
                // Safety: Arrays allocated with dimension capacity, baseAddress valid
                let yVec = DenseVector_Double(count: Int32(dimension), data: yPtr.baseAddress!)
                let byVec = DenseVector_Double(count: Int32(dimension), data: byPtr.baseAddress!)
                SparseMultiply(imagMatrix, yVec, byVec)
            }
        }

        vDSP_vsubD(by, 1, ax, 1, &resultReal, 1, vDSP_Length(dimension))
        vDSP_vaddD(ay, 1, bx, 1, &resultImag, 1, vDSP_Length(dimension))

        var realPart = 0.0
        vDSP_dotprD(stateReal, 1, resultReal, 1, &realPart, vDSP_Length(dimension))

        var imagPart = 0.0
        vDSP_dotprD(stateImag, 1, resultImag, 1, &imagPart, vDSP_Length(dimension))

        return realPart + imagPart
    }

    // MARK: - Diagnostics

    /// Backend name with sparsity statistics.
    ///
    /// **Example:**
    ///
    /// ```swift
    /// let h = Observable(terms: [(1.0, PauliString([.z(0)]))])
    /// let sparse = SparseHamiltonian(observable: h)
    /// let desc = await sparse.backendDescription
    /// ```
    public var backendDescription: String {
        "\(backend.description) (\(nnz) non-zeros, \(String(format: "%.2f%%", sparsity * 100)) sparse)"
    }

    /// Detailed sparse matrix statistics including memory usage and compression ratio.
    ///
    /// Returns diagnostic information useful for benchmarking and resource monitoring. Includes backend
    /// selection, sparsity metrics, and memory consumption compared to dense representation.
    ///
    /// **Example:**
    ///
    /// ```swift
    /// let h = Observable(terms: [(1.0, PauliString([.z(0)]))])
    /// let sparse = SparseHamiltonian(observable: h)
    /// let stats = await sparse.statistics
    /// ```
    ///
    /// - Complexity: O(1)
    @_eagerMove
    public var statistics: SparseMatrixStatistics {
        SparseMatrixStatistics(
            qubits: qubits,
            dimension: dimension,
            nonZeros: nnz,
            sparsity: sparsity,
            backend: backend.description,
            memoryBytes: estimatedMemoryUsage(),
        )
    }

    /// Estimates memory usage of sparse representation.
    @_effects(readonly)
    private func estimatedMemoryUsage() -> Int {
        let rowPointerBytes: Int = (dimension + 1) * MemoryLayout<UInt32>.stride
        let columnIndexBytes: Int = nnz * MemoryLayout<UInt32>.stride
        let valueBytes: Int = nnz * MemoryLayout<Complex<Double>>.stride

        return rowPointerBytes + columnIndexBytes + valueBytes
    }
}

// MARK: - Supporting Types

/// Matrix index for sparse construction hash table.
private struct MatrixIndex: Hashable {
    let row: Int
    let col: Int
}

/// Diagnostic statistics for sparse Hamiltonian representation.
@frozen
public struct SparseMatrixStatistics: CustomStringConvertible, Sendable {
    /// Number of qubits.
    public let qubits: Int

    /// Hilbert space dimension 2^qubits.
    public let dimension: Int

    /// Count of non-zero matrix elements.
    public let nonZeros: Int

    /// Sparsity ratio.
    public let sparsity: Double

    /// Backend name.
    public let backend: String

    /// Total memory in bytes.
    public let memoryBytes: Int

    init(qubits: Int, dimension: Int, nonZeros: Int, sparsity: Double, backend: String, memoryBytes: Int) {
        self.qubits = qubits
        self.dimension = dimension
        self.nonZeros = nonZeros
        self.sparsity = sparsity
        self.backend = backend
        self.memoryBytes = memoryBytes
    }

    @inlinable
    public var description: String {
        let sparsityPercent = String(format: "%.4f%%", sparsity * 100)
        let memoryKB = Double(memoryBytes) / 1024.0
        let memoryMB: Double = memoryKB / 1024.0

        let memoryStr: String = memoryMB >= 1.0 ?
            String(format: "%.2f MB", memoryMB) :
            String(format: "%.2f KB", memoryKB)

        let denseMemoryMB = Double(dimension * dimension * 16) / (1024.0 * 1024.0)
        let compressionRatio: Double = denseMemoryMB * 1024.0 * 1024.0 / Double(memoryBytes)

        return """
        Sparse Hamiltonian Statistics:
          Backend: \(backend)
          Qubits: \(qubits)
          Dimension: \(dimension) x \(dimension)
          Non-zeros: \(nonZeros)
          Sparsity: \(sparsityPercent)
          Memory: \(memoryStr)
          Dense equivalent: \(String(format: "%.2f MB", denseMemoryMB))
          Compression: \(String(format: "%.1fx", compressionRatio))
        """
    }
}
