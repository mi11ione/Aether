// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate
import Metal

/// Shared Metal resources: one-time pipeline compilation for all simulator instances
///
/// Compiles compute shaders once at app launch and shares them globally.
/// Prevents expensive re-compilation when creating multiple simulators.
private enum MetalResources {
    fileprivate static let device: MTLDevice? = MTLCreateSystemDefaultDevice()
    fileprivate static let commandQueue: MTLCommandQueue? = device?.makeCommandQueue()
    fileprivate static let library: MTLLibrary? = {
        guard let device else { return nil }
        return MetalUtilities.loadLibrary(device: device)
    }()

    // Lazily compiled pipeline states (shared across all instances)
    fileprivate static let singleQubitPipeline: MTLComputePipelineState? = {
        guard let device, let library,
              let function = library.makeFunction(name: "applySingleQubitGate")
        else { return nil }
        return try? device.makeComputePipelineState(function: function)
    }()

    fileprivate static let cnotPipeline: MTLComputePipelineState? = {
        guard let device, let library,
              let function = library.makeFunction(name: "applyCNOT")
        else { return nil }
        return try? device.makeComputePipelineState(function: function)
    }()

    fileprivate static let twoQubitPipeline: MTLComputePipelineState? = {
        guard let device, let library,
              let function = library.makeFunction(name: "applyTwoQubitGate")
        else { return nil }
        return try? device.makeComputePipelineState(function: function)
    }()

    fileprivate static let toffoliPipeline: MTLComputePipelineState? = {
        guard let device, let library,
              let function = library.makeFunction(name: "applyToffoli")
        else { return nil }
        return try? device.makeComputePipelineState(function: function)
    }()
}

/// GPU-accelerated gate execution using Metal compute shaders
///
/// Parallelizes quantum gate application across GPU threads for states with ≥10 qubits. Implements the
/// same O(2^n) statevector algorithm as ``GateApplication`` but distributes amplitude pair/quartet updates
/// across thousands of concurrent GPU threads.
///
/// **How it works**:
/// - Single-qubit gates: Each GPU thread processes one amplitude pair (i, j) where indices differ only in target qubit bit
/// - Two-qubit gates: Each GPU thread processes one amplitude quartet (4 amplitudes) for control/target qubit combinations
/// - Special cases: CNOT/CZ/Toffoli use parallel conditional swaps (matches CPU optimization)
/// - Fallback: Returns ``GateApplication`` result if buffer allocation fails or computation produces NaN/Inf
///
/// **Mathematical foundation**:
/// Same unitary transformation |ψ'⟩ = U|ψ⟩ as CPU implementation. GPU parallelism exploits independence:
/// for single-qubit gate on qubit q, amplitude pairs (i, i⊕2^q) update independently. This embarrassingly
/// parallel structure maps perfectly to GPU SIMD execution.
///
/// **Precision tradeoff**:
/// GPU uses Float32 (single precision) vs ``GateApplication`` Float64 (double). For most quantum algorithms,
/// precision loss is negligible compared to gate fidelity (99.9% typical) and decoherence in real hardware.
/// vDSP handles Float64<->Float32 conversion via vectorized operations.
///
/// **Performance crossover**:
/// - n < 10 qubits: CPU faster (overhead: buffer allocation, shader dispatch, precision conversion)
/// - n ≥ 10 qubits: GPU provides speedup (parallelism saturates compute units, amortizes overhead)
/// - Shared pipelines: Compile shaders once at launch, reuse for all instances
///
/// **Example**:
/// ```swift
/// let state = QuantumState(numQubits: 12)  // 4096 amplitudes
/// let metalApp = await MetalGateApplication()!
///
/// let superposition = await metalApp.apply(.hadamard, to: 0, state: state)
///
/// let entangled = await metalApp.apply(.cnot, to: [0, 1], state: superposition)
/// // Bell state (|00⟩ + |11⟩)/√2
/// ```
///
/// - Note: Failable initializer returns nil if Metal unavailable
/// - SeeAlso: ``GateApplication``, ``MetalUtilities``, ``QuantumSimulator``
public actor MetalGateApplication {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    // Compute pipeline states (shared, not per-instance)
    private let singleQubitPipeline: MTLComputePipelineState
    private let cnotPipeline: MTLComputePipelineState
    private let twoQubitPipeline: MTLComputePipelineState
    private let toffoliPipeline: MTLComputePipelineState

    /// Minimum qubit count for GPU acceleration (states with fewer qubits use CPU)
    ///
    /// Below this threshold (n < 10), CPU execution via ``GateApplication`` is faster due to GPU
    /// overhead (buffer allocation, shader dispatch, Float64->Float32 conversion). At n=10 (1024 amplitudes),
    /// parallelism benefit begins to outweigh overhead.
    public static let minimumQubitCountForGPU = 10

    // MARK: - Conversion Helpers

    /// Convert Complex<Double> amplitudes to GPU-compatible Float pairs using vDSP
    @inline(__always)
    private static func toGPUAmplitudes(_ amplitudes: [Complex<Double>]) -> [(Float, Float)] {
        let n = amplitudes.count
        var result = [(Float, Float)](unsafeUninitializedCapacity: n) { _, count in
            count = n
        }

        amplitudes.withUnsafeBytes { srcBytes in
            let srcDoubles = srcBytes.bindMemory(to: Double.self)
            result.withUnsafeMutableBytes { dstBytes in
                let dstFloats = dstBytes.bindMemory(to: Float.self)
                vDSP_vdpsp(srcDoubles.baseAddress!, 1, dstFloats.baseAddress!, 1, vDSP_Length(n * 2))
            }
        }

        return result
    }

    /// Convert GPU Float pairs back to Complex<Double> amplitudes using vDSP
    @inline(__always)
    private static func fromGPUAmplitudes(_ pointer: UnsafePointer<(Float, Float)>, count: Int) -> [Complex<Double>] {
        var result = [Complex<Double>](unsafeUninitializedCapacity: count) { _, outCount in
            outCount = count
        }

        result.withUnsafeMutableBytes { dstBytes in
            let dstDoubles = dstBytes.bindMemory(to: Double.self)
            let srcFloats = UnsafeRawPointer(pointer).bindMemory(to: Float.self, capacity: count * 2)
            vDSP_vspdp(srcFloats, 1, dstDoubles.baseAddress!, 1, vDSP_Length(count * 2))
        }

        return result
    }

    /// Convert 2x2 gate matrix to flat GPU format
    @inline(__always)
    private static func toGPUMatrix2x2(_ matrix: [[Complex<Double>]]) -> [(Float, Float)] {
        [(Float, Float)](unsafeUninitializedCapacity: 4) { buffer, count in
            buffer[0] = (Float(matrix[0][0].real), Float(matrix[0][0].imaginary))
            buffer[1] = (Float(matrix[0][1].real), Float(matrix[0][1].imaginary))
            buffer[2] = (Float(matrix[1][0].real), Float(matrix[1][0].imaginary))
            buffer[3] = (Float(matrix[1][1].real), Float(matrix[1][1].imaginary))
            count = 4
        }
    }

    /// Convert 4x4 gate matrix to flat GPU format
    @inline(__always)
    private static func toGPUMatrix4x4(_ matrix: [[Complex<Double>]]) -> [(Float, Float)] {
        [(Float, Float)](unsafeUninitializedCapacity: 16) { buffer, count in
            var idx = 0
            for row in 0 ..< 4 {
                for col in 0 ..< 4 {
                    buffer[idx] = (Float(matrix[row][col].real), Float(matrix[row][col].imaginary))
                    idx += 1
                }
            }
            count = 16
        }
    }

    /// Create GPU gate executor using shared Metal resources
    ///
    /// Returns nil if Metal unavailable (unsupported hardware, iOS Simulator on Intel Mac) or
    /// shader compilation fails (missing QuantumGPU.metal, malformed metallib). Reuses pre-compiled
    /// compute pipelines from ``MetalResources`` singleton to avoid expensive re-compilation.
    ///
    /// **Example**:
    /// ```swift
    /// if let metalApp = await MetalGateApplication() {
    ///     let result = await metalApp.apply(.hadamard, to: 0, state: state)
    /// } else {
    ///     // Fallback to CPU
    ///     let result = GateApplication.apply(.hadamard, to: 0, state: state)
    /// }
    /// ```
    ///
    /// - Returns: GPU executor if Metal available, nil otherwise
    public init?() {
        guard let device = MetalResources.device,
              let commandQueue = MetalResources.commandQueue,
              let singleQubitPipeline = MetalResources.singleQubitPipeline,
              let cnotPipeline = MetalResources.cnotPipeline,
              let twoQubitPipeline = MetalResources.twoQubitPipeline,
              let toffoliPipeline = MetalResources.toffoliPipeline
        else { return nil }

        self.device = device
        self.commandQueue = commandQueue
        self.singleQubitPipeline = singleQubitPipeline
        self.cnotPipeline = cnotPipeline
        self.twoQubitPipeline = twoQubitPipeline
        self.toffoliPipeline = toffoliPipeline
    }

    // MARK: - Public API

    /// Apply gate to quantum state using GPU parallel execution
    ///
    /// Transforms state via unitary matrix application parallelized across GPU threads. Converts Float64
    /// CPU amplitudes to Float32 GPU buffers, dispatches Metal kernel (2^(n-1) threads for 1-qubit gates,
    /// 2^(n-2) for 2-qubit gates), then converts results back to Float64. Falls back to ``GateApplication``
    /// CPU implementation if Metal unavailable or numerical errors detected.
    ///
    /// **Example**:
    /// ```swift
    /// let state = QuantumState(numQubits: 12)
    /// let metalApp = await MetalGateApplication()!
    /// let withH = await metalApp.apply(.hadamard, to: 0, state: state)
    /// let withCNOT = await metalApp.apply(.cnot, to: [0, 1], state: withH)
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply (any ``QuantumGate`` variant)
    ///   - qubits: Target qubit indices ([control, target] for 2-qubit gates, [control1, control2, target] for Toffoli)
    ///   - state: Input quantum state (normalized statevector)
    /// - Returns: New quantum state with unitary transformation applied, maintaining normalization
    /// - Complexity: O(2^n) time, O(2^n) GPU buffer allocation
    /// - Precondition: All qubit indices must be valid for state (validated by ``ValidationUtilities``)
    @_eagerMove
    public func apply(_ gate: QuantumGate, to qubits: [Int], state: QuantumState) -> QuantumState {
        switch gate {
        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard,
             .phase, .sGate, .tGate, .rotationX, .rotationY, .rotationZ,
             .u1, .u2, .u3, .sx, .sy, .customSingleQubit:
            applySingleQubitGate(gate: gate, qubit: qubits[0], state: state)

        case .cnot:
            applyCNOT(control: qubits[0], target: qubits[1], state: state)

        case .controlledPhase, .controlledRotationX, .controlledRotationY, .controlledRotationZ, .swap, .sqrtSwap, .cz, .cy, .ch, .customTwoQubit:
            applyTwoQubitGate(gate: gate, control: qubits[0], target: qubits[1], state: state)

        case .toffoli:
            applyToffoli(control1: qubits[0], control2: qubits[1], target: qubits[2], state: state)
        }
    }

    /// Apply gate to single qubit (convenience method)
    ///
    /// Wraps qubit index in array and delegates to main apply method.
    /// Preferred syntax for single-qubit gates.
    ///
    /// **Example**:
    /// ```swift
    /// let metalApp = await MetalGateApplication()!
    /// let result = await metalApp.apply(.hadamard, to: 0, state: state)
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply (must be single-qubit variant)
    ///   - qubit: Target qubit index
    ///   - state: Input quantum state (normalized statevector)
    /// - Returns: New quantum state with gate applied
    /// - Complexity: O(2^n) time, O(2^n) GPU buffer allocation
    /// - Precondition: qubit must be valid index for state
    @_eagerMove
    public func apply(_ gate: QuantumGate, to qubit: Int, state: QuantumState) -> QuantumState {
        apply(gate, to: [qubit], state: state)
    }

    // MARK: - Private Metal Implementations

    @_optimize(speed)
    @_eagerMove
    private func applySingleQubitGate(gate: QuantumGate, qubit: Int, state: QuantumState) -> QuantumState {
        var floatAmplitudes = Self.toGPUAmplitudes(state.amplitudes)
        var floatMatrix = Self.toGPUMatrix2x2(gate.matrix())

        let stateSize = state.stateSpaceSize
        let bufferSize = stateSize * MemoryLayout<(Float, Float)>.stride

        guard let amplitudeBuffer = device.makeBuffer(
            bytes: &floatAmplitudes,
            length: bufferSize,
            options: .storageModeShared
        ) else { return GateApplication.apply(gate, to: qubit, state: state) }

        let matrixSize = 4 * MemoryLayout<(Float, Float)>.stride
        guard let matrixBuffer = device.makeBuffer(
            bytes: &floatMatrix,
            length: matrixSize,
            options: .storageModeShared
        ) else { return GateApplication.apply(gate, to: qubit, state: state) }

        var qubitValue = UInt32(qubit)
        var numQubitsValue = UInt32(state.numQubits)

        guard let (commandBuffer, encoder) = MetalUtilities.createCommandEncoder(queue: commandQueue) else {
            return GateApplication.apply(gate, to: qubit, state: state)
        }

        encoder.setComputePipelineState(singleQubitPipeline)
        encoder.setBuffer(amplitudeBuffer, offset: 0, index: 0)
        encoder.setBytes(&qubitValue, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBuffer(matrixBuffer, offset: 0, index: 2)
        encoder.setBytes(&numQubitsValue, length: MemoryLayout<UInt32>.stride, index: 3)

        let numPairs = stateSize / 2
        let threadsPerGroup = MTLSize(width: min(singleQubitPipeline.maxTotalThreadsPerThreadgroup, numPairs), height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (numPairs + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: 1,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let resultPointer = amplitudeBuffer.contents().bindMemory(to: (Float, Float).self, capacity: stateSize)
        let newAmplitudes = Self.fromGPUAmplitudes(resultPointer, count: stateSize)

        guard newAmplitudes.allSatisfy(\.isFinite) else {
            return GateApplication.apply(gate, to: qubit, state: state)
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    @_optimize(speed)
    @_eagerMove
    private func applyCNOT(control: Int, target: Int, state: QuantumState) -> QuantumState {
        var floatAmplitudes = Self.toGPUAmplitudes(state.amplitudes)
        let stateSize = state.stateSpaceSize
        let bufferSize = stateSize * MemoryLayout<(Float, Float)>.stride

        guard let inputBuffer = device.makeBuffer(bytes: &floatAmplitudes, length: bufferSize, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
        else { return GateApplication.apply(.cnot, to: [control, target], state: state) }

        guard let (commandBuffer, encoder) = MetalUtilities.createCommandEncoder(queue: commandQueue) else {
            return GateApplication.apply(.cnot, to: [control, target], state: state)
        }

        var controlValue = UInt32(control)
        var targetValue = UInt32(target)
        var numQubitsValue = UInt32(state.numQubits)

        encoder.setComputePipelineState(cnotPipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBytes(&controlValue, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBytes(&targetValue, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&numQubitsValue, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBuffer(outputBuffer, offset: 0, index: 4)

        let threadsPerGroup = MTLSize(width: min(cnotPipeline.maxTotalThreadsPerThreadgroup, stateSize), height: 1, depth: 1)
        let threadgroups = MTLSize(width: (stateSize + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let resultPointer = outputBuffer.contents().bindMemory(to: (Float, Float).self, capacity: stateSize)
        let newAmplitudes = Self.fromGPUAmplitudes(resultPointer, count: stateSize)

        guard newAmplitudes.allSatisfy(\.isFinite) else {
            return GateApplication.apply(.cnot, to: [control, target], state: state)
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    @_optimize(speed)
    @_eagerMove
    private func applyTwoQubitGate(gate: QuantumGate, control: Int, target: Int, state: QuantumState) -> QuantumState {
        var floatAmplitudes = Self.toGPUAmplitudes(state.amplitudes)
        var floatMatrix = Self.toGPUMatrix4x4(gate.matrix())

        let stateSize = state.stateSpaceSize
        let bufferSize = stateSize * MemoryLayout<(Float, Float)>.stride
        let matrixSize = 16 * MemoryLayout<(Float, Float)>.stride

        guard let inputBuffer = device.makeBuffer(bytes: &floatAmplitudes, length: bufferSize, options: .storageModeShared),
              let matrixBuffer = device.makeBuffer(bytes: &floatMatrix, length: matrixSize, options: .storageModeShared)
        else { return GateApplication.apply(gate, to: [control, target], state: state) }

        guard let (commandBuffer, encoder) = MetalUtilities.createCommandEncoder(queue: commandQueue) else {
            return GateApplication.apply(gate, to: [control, target], state: state)
        }

        var controlValue = UInt32(control)
        var targetValue = UInt32(target)
        var numQubitsValue = UInt32(state.numQubits)

        encoder.setComputePipelineState(twoQubitPipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBytes(&controlValue, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBytes(&targetValue, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBuffer(matrixBuffer, offset: 0, index: 3)
        encoder.setBytes(&numQubitsValue, length: MemoryLayout<UInt32>.stride, index: 4)

        let numQuartets = stateSize / 4
        let threadsPerGroup = MTLSize(width: min(twoQubitPipeline.maxTotalThreadsPerThreadgroup, numQuartets), height: 1, depth: 1)
        let threadgroups = MTLSize(width: (numQuartets + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let resultPointer = inputBuffer.contents().bindMemory(to: (Float, Float).self, capacity: stateSize)
        let newAmplitudes = Self.fromGPUAmplitudes(resultPointer, count: stateSize)

        guard newAmplitudes.allSatisfy(\.isFinite) else {
            return GateApplication.apply(gate, to: [control, target], state: state)
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    @_optimize(speed)
    @_eagerMove
    private func applyToffoli(control1: Int, control2: Int, target: Int, state: QuantumState) -> QuantumState {
        var floatAmplitudes = Self.toGPUAmplitudes(state.amplitudes)
        let stateSize = state.stateSpaceSize
        let bufferSize = stateSize * MemoryLayout<(Float, Float)>.stride

        guard let inputBuffer = device.makeBuffer(bytes: &floatAmplitudes, length: bufferSize, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
        else { return GateApplication.apply(.toffoli, to: [control1, control2, target], state: state) }

        guard let (commandBuffer, encoder) = MetalUtilities.createCommandEncoder(queue: commandQueue) else {
            return GateApplication.apply(.toffoli, to: [control1, control2, target], state: state)
        }

        var c1Value = UInt32(control1)
        var c2Value = UInt32(control2)
        var targetValue = UInt32(target)
        var numQubitsValue = UInt32(state.numQubits)

        encoder.setComputePipelineState(toffoliPipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBytes(&c1Value, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBytes(&c2Value, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&targetValue, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&numQubitsValue, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.setBuffer(outputBuffer, offset: 0, index: 5)

        let threadsPerGroup = MTLSize(width: min(toffoliPipeline.maxTotalThreadsPerThreadgroup, stateSize), height: 1, depth: 1)
        let threadgroups = MTLSize(width: (stateSize + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let resultPointer = outputBuffer.contents().bindMemory(to: (Float, Float).self, capacity: stateSize)
        let newAmplitudes = Self.fromGPUAmplitudes(resultPointer, count: stateSize)

        guard newAmplitudes.allSatisfy(\.isFinite) else {
            return GateApplication.apply(.toffoli, to: [control1, control2, target], state: state)
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }
}

// MARK: - Hybrid CPU/GPU Gate Application

public extension GateApplication {
    /// Apply gate with automatic CPU/GPU selection based on state size
    ///
    /// Routes to GPU (``MetalGateApplication``) for states with ≥10 qubits, otherwise uses CPU.
    /// Automatically falls back to CPU if Metal unavailable or GPU computation fails. Optimal
    /// default for most use cases: exploits GPU parallelism for large states while avoiding
    /// overhead for small states.
    ///
    /// **Example**:
    /// ```swift
    /// let small = QuantumState(numQubits: 5)
    /// let large = QuantumState(numQubits: 15)
    /// let r1 = await GateApplication.applyHybrid(.hadamard, to: [0], state: small)  // CPU
    /// let r2 = await GateApplication.applyHybrid(.hadamard, to: [0], state: large)  // GPU
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply
    ///   - qubits: Target qubit indices
    ///   - state: Input quantum state
    /// - Returns: Transformed state (via GPU or CPU)
    /// - Complexity: O(2^n) time
    @inlinable
    @_eagerMove
    static func applyHybrid(_ gate: QuantumGate, to qubits: [Int], state: QuantumState) async -> QuantumState {
        if state.numQubits >= MetalGateApplication.minimumQubitCountForGPU {
            if let metalApp = MetalGateApplication() {
                return await metalApp.apply(gate, to: qubits, state: state)
            }
        }

        return apply(gate, to: qubits, state: state)
    }

    /// Apply gate to single qubit with automatic CPU/GPU selection
    ///
    /// Wraps qubit index in array and delegates to main applyHybrid method.
    ///
    /// **Example**:
    /// ```swift
    /// let result = await GateApplication.applyHybrid(.hadamard, to: 0, state: state)
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply
    ///   - qubit: Target qubit index
    ///   - state: Input quantum state
    /// - Returns: Transformed state (via GPU or CPU)
    /// - Complexity: O(2^n) time
    @inlinable
    @_eagerMove
    static func applyHybrid(_ gate: QuantumGate, to qubit: Int, state: QuantumState) async -> QuantumState {
        await applyHybrid(gate, to: [qubit], state: state)
    }
}

/// Shared Metal utilities for GPU-accelerated quantum computing
///
/// Centralizes common Metal device initialization, library loading, and command buffer
/// creation patterns used across MetalGateApplication and SparseHamiltonian backends.
/// Implements 3-tier fallback strategy for maximum compatibility across deployment
/// environments (compiled metallib, default library, source compilation).
///
/// **Library Loading Strategy**:
/// 1. **Compiled metallib**: Fastest, pre-compiled binary shaders
/// 2. **Default library**: Built-in to app bundle during Xcode build
/// 3. **Source compilation**: Fallback for non-standard deployments (slower first load)
///
/// **Thread Safety**: All methods are stateless and thread-safe. Device/queue creation
/// can be called concurrently from multiple threads without synchronization.
public enum MetalUtilities {
    /// Load Metal shader library with 3-tier fallback strategy
    ///
    /// Attempts to load QuantumGPU shader library in order of performance preference:
    /// 1. Pre-compiled `default.metallib` file (for xcode)
    /// 2. Default library from app bundle
    /// 3. Runtime compilation from `QuantumGPU.metal` source (for CLI)
    ///
    /// **Error Handling**: Returns nil if all three loading strategies fail. Caller must
    /// handle Metal unavailability with CPU fallback.
    ///
    /// - Parameter device: Metal device to create library for
    /// - Returns: Shader library if any loading strategy succeeds, nil otherwise
    @_effects(readonly)
    public static func loadLibrary(device: MTLDevice) -> MTLLibrary? {
        if let metallibURL = Bundle.module.url(forResource: "default", withExtension: "metallib"),
           let library = try? device.makeLibrary(URL: metallibURL)
        { return library }

        if let defaultLibrary = device.makeDefaultLibrary() { return defaultLibrary }

        guard let resourceURL = Bundle.module.url(forResource: "QuantumGPU", withExtension: "metal"),
              let source = try? String(contentsOf: resourceURL, encoding: .utf8),
              let library = try? device.makeLibrary(source: source, options: nil)
        else { return nil }

        return library
    }

    /// Create Metal command buffer and compute encoder with fallback handling
    ///
    /// Convenience wrapper for the common pattern of creating command buffer + encoder
    /// with guard-let error handling. Reduces boilerplate in GPU kernel dispatch code.
    ///
    /// **Usage Pattern**:
    /// ```swift
    /// guard let (commandBuffer, encoder) = MetalUtilities.createCommandEncoder(queue: commandQueue)
    /// else {
    ///     // Fallback to CPU implementation
    ///     return cpuFallback()
    /// }
    /// // Configure encoder...
    /// encoder.endEncoding()
    /// commandBuffer.commit()
    /// ```
    ///
    /// - Parameter queue: Metal command queue to create buffer from
    /// - Returns: (command buffer, compute encoder) tuple if creation succeeds, nil otherwise
    @_effects(readonly)
    @inlinable
    public static func createCommandEncoder(
        queue: MTLCommandQueue
    ) -> (commandBuffer: MTLCommandBuffer, encoder: MTLComputeCommandEncoder)? {
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { return nil }

        return (commandBuffer, encoder)
    }
}
