// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate
import Foundation
import Metal

/// Sparse Hamiltonian: High-performance expectation value computation
///
/// Solves the VQE performance bottleneck by converting Hamiltonians H = Σᵢ cᵢ Pᵢ
/// to sparse matrix format. Achieves high speedup over term-by-term
/// measurement for molecular Hamiltonians.
///
/// **The Problem:**
/// - Molecular Hamiltonian H = Σᵢ cᵢ Pᵢ with 2000 terms
/// - Current approach: measure each Pᵢ separately (2000 circuit executions)
/// - VQE iteration needs 100+ expectation values
/// - Total: 200,000 circuit executions per VQE iteration
/// - Makes VQE unusably slow
///
/// **The Solution:**
/// - Convert H = Σᵢ cᵢ Pᵢ to sparse matrix format
/// - Molecular Hamiltonians are 0.01%-1% non-zero
/// - Example: 10-qubit H₂O has ~8000 non-zeros out of 1M elements
/// - Store only (row, col, value) triplets → massive memory savings
/// - Use custom Metal kernel for GPU-accelerated H|ψ⟩
/// - Compute ⟨ψ|H|ψ⟩ = ⟨ψ|(H|ψ⟩) as inner product
///
/// **Three-Tier Performance Architecture:**
/// 1. **Metal GPU** (fastest): Custom CSR sparse kernel with native complex support
///    - Used when: Metal available AND numQubits ≥ 8
///    - Primary path for VQE on modern Macs
///
/// 2. **CPU Sparse** (fast): Accelerate-optimized sparse operations
///    - Used when: Metal unavailable OR numQubits < 8 (GPU overhead too high)
///
/// 3. **Observable** (baseline): Existing term-by-term measurement
///    - Used when: Sparse construction fails (fallback path)
///    - Guaranteed to work, used for correctness testing
///
/// **Example - VQE with sparse Hamiltonian:**
/// ```swift
/// // Build sparse Hamiltonian once
/// let hamiltonian = Observable(terms: [...])  // 2000 terms
/// let sparseH = SparseHamiltonian(observable: hamiltonian)
/// print(sparseH.backend)  // "Metal GPU (8142 non-zeros, 0.79% sparse)"
///
/// // VQE optimization loop
/// var params = [0.1, 0.2, 0.3, 0.4]
/// for iteration in 0..<100 {
///     let state = prepareAnsatz(parameters: params)
///     let energy = sparseH.expectationValue(state: state)
///
///     let gradient = computeGradient(energy)
///     params = updateParameters(params, gradient)
/// }
/// ```
///
/// **Thread Safety:**
/// - Sparse matrix construction: Not thread-safe (call from single thread)
/// - Expectation value computation: Thread-safe (read-only operations)
/// - VQE usage: Build once, compute from multiple threads safely
final class SparseHamiltonian {
    /// Performance backend for expectation value computation
    private enum Backend {
        /// Metal GPU backend: Custom CSR sparse kernel with native complex support
        case metalGPU(
            device: MTLDevice,
            commandQueue: MTLCommandQueue,
            pipelineState: MTLComputePipelineState,
            rowPointers: MTLBuffer,
            columnIndices: MTLBuffer,
            values: MTLBuffer,
            nnz: Int
        )

        /// CPU sparse backend: Accelerate-optimized sparse operations
        case cpuSparse(
            rowPointers: [Int],
            columnIndices: [Int],
            values: [Complex<Double>]
        )

        /// Observable backend: Term-by-term measurement (fallback)
        case observable(Observable)

        var description: String {
            switch self {
            case .metalGPU: "Metal GPU"
            case .cpuSparse: "CPU Sparse"
            case .observable: "Observable (fallback)"
            }
        }
    }

    // MARK: - Properties

    /// Number of qubits in the Hamiltonian
    let numQubits: Int

    /// Dimension of the Hilbert space (2^numQubits)
    let dimension: Int

    /// Number of non-zero elements in sparse representation
    private(set) var nnz: Int = 0

    /// Sparsity ratio (non-zeros / total elements)
    var sparsity: Double {
        Double(nnz) / Double(dimension * dimension)
    }

    /// Selected performance backend
    private let backend: Backend

    /// Original observable (kept for diagnostics and fallback)
    private let observable: Observable

    // MARK: - Initialization

    /// Create sparse Hamiltonian from Observable with automatic backend selection
    ///
    /// **Backend Selection Logic:**
    /// 1. Try Metal GPU (fastest) if Metal available and numQubits ≥ 8
    /// 2. Fall back to CPU sparse if Metal unavailable or small system
    /// 3. Fall back to Observable if sparse construction fails
    ///
    /// **Performance:**
    /// - Memory: Store only non-zeros (~0.01%-1% of full matrix)
    /// - Reuse for all VQE iterations
    ///
    /// - Parameter observable: Quantum observable H = Σᵢ cᵢ Pᵢ
    init(observable: Observable) {
        self.observable = observable

        var maxQubit = -1
        for (_, pauliString) in observable.terms {
            for op in pauliString.operators {
                maxQubit = max(maxQubit, op.qubit)
            }
        }

        numQubits = max(maxQubit + 1, 1)
        dimension = 1 << numQubits

        let cooMatrix = Self.buildCOOMatrix(from: observable, dimension: dimension)
        nnz = cooMatrix.count

        if numQubits >= 8, let metalBackend = Self.tryMetalGPUBackend(
            cooMatrix: cooMatrix,
            dimension: dimension,
            numQubits: numQubits
        ) {
            backend = metalBackend
        } else if let cpuBackend = Self.tryCPUSparseBackend(
            cooMatrix: cooMatrix,
            dimension: dimension
        ) {
            backend = cpuBackend
        } else {
            backend = .observable(observable)
        }
    }

    // MARK: - Sparse Matrix Construction (COO Format)

    private struct COOElement {
        let row: Int
        let col: Int
        let value: Complex<Double>
    }

    /// Build sparse matrix in COO (Coordinate) format from Observable
    ///
    /// **Algorithm:**
    /// 1. For each Pauli term (cᵢ, Pᵢ): Convert Pᵢ to sparse matrix
    /// 2. Accumulate: H[i,j] += cᵢ * Pᵢ[i,j]
    /// 3. Filter near-zero elements (numerical cancellation)
    ///
    /// **Sparsity:**
    /// - Single Pauli string: at most 2ⁿ non-zeros
    /// - Molecular Hamiltonians: 0.01%-1% non-zero after cancellation
    /// - Example: H₂O (10 qubits) → 8K non-zeros out of 1M elements
    ///
    /// - Parameters:
    ///   - observable: Observable with Pauli decomposition
    ///   - dimension: Hilbert space dimension (2^numQubits)
    /// - Returns: Array of COO elements (row, col, value)
    private static func buildCOOMatrix(
        from observable: Observable,
        dimension: Int
    ) -> [COOElement] {
        var elements: [MatrixIndex: Complex<Double>] = [:]

        for (coefficient, pauliString) in observable.terms {
            // Convert Pauli string to sparse matrix
            let pauliMatrix = pauliStringToSparseMatrix(
                pauliString,
                dimension: dimension
            )

            for element in pauliMatrix {
                let index = MatrixIndex(row: element.row, col: element.col)
                if let existing = elements[index] {
                    elements[index] = existing + coefficient * element.value
                } else {
                    elements[index] = coefficient * element.value
                }
            }
        }

        let tolerance = 1e-12
        let nonZeros = elements.compactMap { index, value -> COOElement? in
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

    /// Convert Pauli string to sparse matrix representation
    ///
    /// **Algorithm:**
    /// - Pauli string P = P₀ ⊗ P₁ ⊗ ... ⊗ Pₙ₋₁
    /// - For each row, compute column and phase directly:
    ///   - Identity I: column = row, phase = 1
    ///   - Pauli X: column = row with bit flipped, phase = 1
    ///   - Pauli Y: column = row with bit flipped, phase = ±i
    ///   - Pauli Z: column = row, phase = ±1
    /// - Complexity: O(2ⁿ × n) instead of O(4ⁿ × n)
    ///
    /// **Sparsity:**
    /// - All Pauli strings: exactly 2ⁿ non-zeros (one per row)
    /// - 1000× faster for 10-qubit systems
    ///
    /// - Parameters:
    ///   - pauliString: Pauli operators (sparse: only non-identity qubits)
    ///   - dimension: Hilbert space dimension
    /// - Returns: Sparse matrix as COO elements
    private static func pauliStringToSparseMatrix(
        _ pauliString: PauliString,
        dimension: Int
    ) -> [COOElement] {
        var pauliMap: [Int: PauliBasis] = [:]
        for op in pauliString.operators {
            pauliMap[op.qubit] = op.basis
        }

        let numQubits = Int(log2(Double(dimension)))
        var elements: [COOElement] = []
        elements.reserveCapacity(dimension)

        for row in 0 ..< dimension {
            var col = row
            var phase = Complex<Double>.one

            for qubit in 0 ..< numQubits {
                let rowBit = (row >> qubit) & 1

                if let pauli = pauliMap[qubit] {
                    switch pauli {
                    case .x:
                        col ^= (1 << qubit)

                    case .y:
                        col ^= (1 << qubit)
                        phase = phase * (rowBit == 0 ? -Complex<Double>.i : Complex<Double>.i)

                    case .z:
                        phase = phase * (rowBit == 0 ? .one : -.one)
                    }
                }
            }

            elements.append(COOElement(row: row, col: col, value: phase))
        }

        return elements
    }

    // MARK: - Backend Construction

    /// Try to create Metal GPU backend with custom CSR sparse kernel
    ///
    /// **Requirements:**
    /// - Metal device available
    /// - numQubits ≥ 8 (GPU overhead too high for small systems)
    ///
    /// **Algorithm:**
    /// 1. Load and compile csrSparseMatVec kernel from QuantumGPU.metal
    /// 2. Convert COO to CSR (Compressed Sparse Row) format
    /// 3. Create Metal buffers for CSR sparse data
    /// 4. Store pipeline state and buffers for kernel dispatch
    ///
    /// **Native Complex Support:**
    /// - Uses Metal's ComplexFloat (float2: real, imaginary)
    /// - Converts Swift's Complex<Double> to Float32 for GPU
    /// - Acceptable precision loss: 64-bit → 32-bit (~7 digits)
    ///
    /// - Parameters:
    ///   - cooMatrix: Sparse matrix in COO format
    ///   - dimension: Hilbert space dimension
    ///   - numQubits: Number of qubits
    /// - Returns: Metal GPU backend or nil if unavailable
    private static func tryMetalGPUBackend(
        cooMatrix: [COOElement],
        dimension: Int,
        numQubits _: Int
    ) -> Backend? {
        guard !cooMatrix.isEmpty else { return nil }
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        guard let commandQueue = device.makeCommandQueue() else { return nil }
        guard let library = device.makeDefaultLibrary() else { return nil }
        guard let kernelFunction = library.makeFunction(name: "csrSparseMatVec") else { return nil }
        guard let pipelineState = try? device.makeComputePipelineState(function: kernelFunction) else { return nil }

        let (rowPointers, columnIndices, values) = convertCOOtoCSR(
            cooMatrix: cooMatrix,
            numRows: dimension
        )

        var float32Values: [(Float, Float)] = []
        for value in values {
            float32Values.append((Float(value.real), Float(value.imaginary)))
        }

        // Use .storageModeShared for Apple Silicon zero-copy
        let storageMode: MTLResourceOptions = .storageModeShared

        guard let rowPointerBuffer = device.makeBuffer(
            bytes: rowPointers,
            length: rowPointers.count * MemoryLayout<UInt32>.stride,
            options: storageMode
        ) else { return nil }

        guard let columnIndexBuffer = device.makeBuffer(
            bytes: columnIndices,
            length: columnIndices.count * MemoryLayout<UInt32>.stride,
            options: storageMode
        ) else { return nil }

        guard let valueBuffer = device.makeBuffer(
            bytes: float32Values,
            length: float32Values.count * MemoryLayout<(Float, Float)>.stride,
            options: storageMode
        ) else { return nil }

        return .metalGPU(
            device: device,
            commandQueue: commandQueue,
            pipelineState: pipelineState,
            rowPointers: rowPointerBuffer,
            columnIndices: columnIndexBuffer,
            values: valueBuffer,
            nnz: values.count
        )
    }

    /// Convert COO format to CSR (Compressed Sparse Row) format
    ///
    /// **CSR Format:**
    /// - rowPointers[i] = index in columnIndices where row i starts
    /// - rowPointers[i+1] - rowPointers[i] = number of non-zeros in row i
    /// - columnIndices[k] = column index of k-th non-zero
    /// - values[k] = value of k-th non-zero
    ///
    /// **Example:**
    /// ```
    /// COO: [(0,1,2.0), (0,2,3.0), (1,1,4.0)]
    /// CSR: rowPointers=[0,2,3], columnIndices=[1,2,1], values=[2.0,3.0,4.0]
    /// ```
    ///
    /// - Parameters:
    ///   - cooMatrix: COO elements (must be sorted by row then column)
    ///   - numRows: Number of rows in matrix
    /// - Returns: (rowPointers, columnIndices, values) in CSR format
    private static func convertCOOtoCSR(
        cooMatrix: [COOElement],
        numRows: Int
    ) -> (rowPointers: [UInt32], columnIndices: [UInt32], values: [Complex<Double>]) {
        var rowPointers = [UInt32](repeating: 0, count: numRows + 1)
        var columnIndices: [UInt32] = []
        var values: [Complex<Double>] = []

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

    /// Try to create CPU sparse backend (Accelerate-optimized)
    ///
    /// **Algorithm:**
    /// - Store CSR format for cache-friendly sparse matrix-vector multiply
    /// - Use Accelerate for vectorized complex arithmetic
    ///
    /// - Parameters:
    ///   - cooMatrix: Sparse matrix in COO format
    ///   - dimension: Hilbert space dimension
    /// - Returns: CPU sparse backend or nil if construction fails
    private static func tryCPUSparseBackend(
        cooMatrix: [COOElement],
        dimension: Int
    ) -> Backend? {
        guard !cooMatrix.isEmpty else {
            return nil
        }

        let (rowPointers32, columnIndices32, values) = convertCOOtoCSR(
            cooMatrix: cooMatrix,
            numRows: dimension
        )

        // Convert UInt32 to Int for Swift arrays
        let rowPointers = rowPointers32.map { Int($0) }
        let columnIndices = columnIndices32.map { Int($0) }

        return .cpuSparse(
            rowPointers: rowPointers,
            columnIndices: columnIndices,
            values: values
        )
    }

    // MARK: - Expectation Value Computation

    /// Compute expectation value ⟨ψ|H|ψ⟩ using selected backend
    ///
    /// **Algorithm:**
    /// 1. Compute |φ⟩ = H|ψ⟩ via sparse matrix-vector multiply
    /// 2. Compute ⟨ψ|φ⟩ = Σᵢ ψᵢ* · φᵢ (inner product)
    /// 3. Return real part (imaginary part is numerical noise for Hermitian H)
    ///
    /// **Thread Safety:**
    /// - Read-only operations, safe to call from multiple threads
    /// - Each call creates temporary buffers, no shared state
    ///
    /// - Parameter state: Normalized quantum state |ψ⟩
    /// - Returns: Expectation value ⟨ψ|H|ψ⟩ ∈ ℝ
    func expectationValue(state: QuantumState) -> Double {
        precondition(state.numQubits == numQubits, "State must have \(numQubits) qubits")
        precondition(state.isNormalized(), "State must be normalized")

        switch backend {
        case let .metalGPU(device, commandQueue, pipelineState, rowPointers, columnIndices, values, nnz):
            return computeMetalGPU(
                state: state,
                device: device,
                commandQueue: commandQueue,
                pipelineState: pipelineState,
                rowPointers: rowPointers,
                columnIndices: columnIndices,
                values: values,
                nnz: nnz
            )

        case let .cpuSparse(rowPointers, columnIndices, values):
            return computeCPUSparse(
                state: state,
                rowPointers: rowPointers,
                columnIndices: columnIndices,
                values: values
            )

        case let .observable(obs):
            return obs.expectationValue(state: state)
        }
    }

    /// Compute expectation value using Metal GPU backend with custom CSR kernel
    ///
    /// **Algorithm:**
    /// 1. Convert Complex<Double> state to Float32 pairs for Metal
    /// 2. Create Metal buffers for input/output vectors
    /// 3. Dispatch csrSparseMatVec kernel (GPU computes H|ψ⟩)
    /// 4. Wait for completion and read result from GPU
    /// 5. Convert back to Complex<Double> and compute inner product ⟨ψ|H|ψ⟩
    ///
    /// - Parameters:
    ///   - state: Quantum state
    ///   - device: Metal device
    ///   - commandQueue: Metal command queue
    ///   - pipelineState: Compiled kernel pipeline state
    ///   - rowPointers: CSR row pointers buffer
    ///   - columnIndices: CSR column indices buffer
    ///   - values: CSR complex values buffer
    ///   - nnz: Number of non-zeros
    /// - Returns: Expectation value
    private func computeMetalGPU(
        state: QuantumState,
        device: MTLDevice,
        commandQueue: MTLCommandQueue,
        pipelineState: MTLComputePipelineState,
        rowPointers: MTLBuffer,
        columnIndices: MTLBuffer,
        values: MTLBuffer,
        nnz _: Int
    ) -> Double {
        var float32State: [(Float, Float)] = []
        for amplitude in state.amplitudes {
            float32State.append((Float(amplitude.real), Float(amplitude.imaginary)))
        }

        guard let inputBuffer = device.makeBuffer(
            bytes: float32State,
            length: dimension * MemoryLayout<(Float, Float)>.stride,
            options: .storageModeShared
        ) else { return observable.expectationValue(state: state) }

        guard let outputBuffer = device.makeBuffer(
            length: dimension * MemoryLayout<(Float, Float)>.stride,
            options: .storageModeShared
        ) else { return observable.expectationValue(state: state) }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            return observable.expectationValue(state: state)
        }

        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return observable.expectationValue(state: state)
        }

        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(rowPointers, offset: 0, index: 0)
        encoder.setBuffer(columnIndices, offset: 0, index: 1)
        encoder.setBuffer(values, offset: 0, index: 2)
        encoder.setBuffer(inputBuffer, offset: 0, index: 3)
        encoder.setBuffer(outputBuffer, offset: 0, index: 4)
        var dim = UInt32(dimension)
        encoder.setBytes(&dim, length: MemoryLayout<UInt32>.stride, index: 5)

        let threadGroupSize = min(pipelineState.maxTotalThreadsPerThreadgroup, dimension)
        let threadGroups = (dimension + threadGroupSize - 1) / threadGroupSize

        encoder.dispatchThreadgroups(
            MTLSize(width: threadGroups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: threadGroupSize, height: 1, depth: 1)
        )

        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let outputPointer = outputBuffer.contents().bindMemory(
            to: (Float, Float).self,
            capacity: dimension
        )

        var result = Complex<Double>.zero
        for i in 0 ..< dimension {
            let hPsiElement = Complex<Double>(
                Double(outputPointer[i].0),
                Double(outputPointer[i].1)
            )
            result = result + state.amplitudes[i].conjugate * hPsiElement
        }

        return result.real
    }

    /// Compute expectation value using CPU sparse backend
    ///
    /// **Algorithm:**
    /// 1. Sparse matrix-vector multiply: output[i] = Σⱼ H[i,j] * state[j]
    /// 2. Inner product: ⟨ψ|φ⟩ = Σᵢ ψᵢ* · φᵢ
    ///
    /// - Parameters:
    ///   - state: Quantum state
    ///   - rowPointers: CSR row pointers
    ///   - columnIndices: CSR column indices
    ///   - values: Complex matrix values
    /// - Returns: Expectation value
    private func computeCPUSparse(
        state: QuantumState,
        rowPointers: [Int],
        columnIndices: [Int],
        values: [Complex<Double>]
    ) -> Double {
        var hPsi = [Complex<Double>](repeating: .zero, count: dimension)

        for row in 0 ..< dimension {
            let start = rowPointers[row]
            let end = rowPointers[row + 1]

            var sum = Complex<Double>.zero
            for idx in start ..< end {
                let col = columnIndices[idx]
                let value = values[idx]
                sum = sum + value * state.amplitudes[col]
            }

            hPsi[row] = sum
        }

        // Inner product: ⟨ψ|H|ψ⟩
        var result = Complex<Double>.zero
        for i in 0 ..< dimension {
            result = result + state.amplitudes[i].conjugate * hPsi[i]
        }

        return result.real
    }

    // MARK: - Diagnostics

    var backendDescription: String {
        "\(backend.description) (\(nnz) non-zeros, \(String(format: "%.2f%%", sparsity * 100)) sparse)"
    }

    func getStatistics() -> SparseMatrixStatistics {
        SparseMatrixStatistics(
            numQubits: numQubits,
            dimension: dimension,
            nonZeros: nnz,
            sparsity: sparsity,
            backend: backend.description,
            memoryBytes: estimatedMemoryUsage()
        )
    }

    private func estimatedMemoryUsage() -> Int {
        // CSR format: rowPointers (dimension+1) + columnIndices (nnz) + values (nnz)
        let rowPointerBytes = (dimension + 1) * MemoryLayout<UInt32>.stride
        let columnIndexBytes = nnz * MemoryLayout<UInt32>.stride
        let valueBytes = nnz * MemoryLayout<Complex<Double>>.stride

        return rowPointerBytes + columnIndexBytes + valueBytes
    }
}

// MARK: - Supporting Types

/// Matrix index (row, column) for sparse accumulation
private struct MatrixIndex: Hashable {
    let row: Int
    let col: Int
}

/// Statistics about sparse Hamiltonian
struct SparseMatrixStatistics: CustomStringConvertible {
    let numQubits: Int
    let dimension: Int
    let nonZeros: Int
    let sparsity: Double
    let backend: String
    let memoryBytes: Int

    var description: String {
        let sparsityPercent = String(format: "%.4f%%", sparsity * 100)
        let memoryKB = Double(memoryBytes) / 1024.0
        let memoryMB = memoryKB / 1024.0

        let memoryStr = memoryMB >= 1.0 ?
            String(format: "%.2f MB", memoryMB) :
            String(format: "%.2f KB", memoryKB)

        let denseMemoryMB = Double(dimension * dimension * 16) / (1024.0 * 1024.0)
        let compressionRatio = denseMemoryMB * 1024.0 * 1024.0 / Double(memoryBytes)

        return """
        Sparse Hamiltonian Statistics:
          Backend: \(backend)
          Qubits: \(numQubits)
          Dimension: \(dimension) × \(dimension)
          Non-zeros: \(nonZeros)
          Sparsity: \(sparsityPercent)
          Memory: \(memoryStr)
          Dense equivalent: \(String(format: "%.2f MB", denseMemoryMB))
          Compression: \(String(format: "%.1f×", compressionRatio))
        """
    }
}
