// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate
import Metal

/// GPU-compatible complex number representation (real, imaginary)
public typealias GPUComplex = (Float, Float)

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

/// GPU-accelerated gate application: Metal compute shaders for quantum simulation
///
/// Provides hardware-accelerated quantum gate execution using Apple's Metal GPU framework.
/// Automatically falls back to CPU when Metal unavailable or for small states where GPU
/// overhead exceeds benefits. Enabled for states with ≥10 qubits.
///
/// **Performance characteristics**:
/// - GPU threshold: Activates for states with ≥10 qubits (2^10 = 1024 amplitudes)
/// - Overhead: Buffer allocation + data transfer (amortized over many gates)
/// - Precision: Float32 on GPU, converted to/from Float64 CPU state
///
/// **Architecture**:
/// - Shared pipelines: Compile compute shaders once, reuse across all instances
/// - Parallel dispatch: Each thread processes independent state amplitude(s)
/// - Automatic fallback: Returns to CPU if Metal unavailable or computation fails
/// - Validation: Checks for NaN/Inf after GPU computation
///
/// **Metal compute kernels**:
/// - `applySingleQubitGate`: Parallel 2×2 matrix-vector for qubit pairs
/// - `applyCNOT`: Parallel conditional amplitude swap
/// - `applyTwoQubitGate`: Parallel 4×4 matrix-vector for qubit quartets
/// - `applyToffoli`: Parallel double-controlled amplitude swap
///
/// **Memory management**:
/// - Shared storage mode: CPU-GPU shared memory (unified architecture)
/// - Float32 buffers: Reduces memory bandwidth requirements
/// - Dynamic allocation: Buffers sized to state vector length
///
/// **When GPU acceleration activates**:
/// - `QuantumSimulator(useMetalAcceleration: true)` (default)
/// - State has ≥10 qubits (threshold configurable)
/// - Metal device available (Apple Silicon, modern GPUs)
/// - Automatic per-gate: Transparent to user
///
/// Example:
/// ```swift
/// // GPU acceleration enabled by default
/// let simulator = await QuantumSimulator()
/// let largeCircuit = QuantumCircuit(numQubits: 12)  // Will use GPU
/// largeCircuit.append(gate: .hadamard, toQubit: 0)
/// largeCircuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
/// let result = await simulator.execute(largeCircuit)
/// // Gates automatically run on GPU
///
/// // Manual GPU application (rarely needed)
/// let metalApp = MetalGateApplication()
/// let state = QuantumState(numQubits: 12)
/// let newState = metalApp?.apply(gate: .hadamard, to: [0], state: state)
///
/// // Hybrid CPU/GPU (automatic threshold)
/// let anyState = QuantumState(numQubits: 5)  // CPU
/// let bigState = QuantumState(numQubits: 15)  // GPU
/// let result1 = GateApplication.applyHybrid(gate: .hadamard, to: [0], state: anyState)
/// let result2 = GateApplication.applyHybrid(gate: .hadamard, to: [0], state: bigState)
/// ```
public actor MetalGateApplication {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    // Compute pipeline states (shared, not per-instance)
    private let singleQubitPipeline: MTLComputePipelineState
    private let cnotPipeline: MTLComputePipelineState
    private let twoQubitPipeline: MTLComputePipelineState
    private let toffoliPipeline: MTLComputePipelineState

    /// Threshold: use GPU for states with >= this many qubits
    public static let gpuThreshold = 10

    // MARK: - Conversion Helpers

    /// Convert Complex<Double> amplitudes to GPU-compatible Float pairs using vDSP
    @inline(__always)
    private static func toGPUAmplitudes(_ amplitudes: AmplitudeVector) -> [GPUComplex] {
        let n = amplitudes.count
        var result = [GPUComplex](unsafeUninitializedCapacity: n) { _, count in
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
    private static func fromGPUAmplitudes(_ pointer: UnsafePointer<GPUComplex>, count: Int) -> AmplitudeVector {
        var result = AmplitudeVector(unsafeUninitializedCapacity: count) { _, outCount in
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
    private static func toGPUMatrix2x2(_ matrix: GateMatrix) -> [GPUComplex] {
        [GPUComplex](unsafeUninitializedCapacity: 4) { buffer, count in
            buffer[0] = (Float(matrix[0][0].real), Float(matrix[0][0].imaginary))
            buffer[1] = (Float(matrix[0][1].real), Float(matrix[0][1].imaginary))
            buffer[2] = (Float(matrix[1][0].real), Float(matrix[1][0].imaginary))
            buffer[3] = (Float(matrix[1][1].real), Float(matrix[1][1].imaginary))
            count = 4
        }
    }

    /// Convert 4x4 gate matrix to flat GPU format
    @inline(__always)
    private static func toGPUMatrix4x4(_ matrix: GateMatrix) -> [GPUComplex] {
        [GPUComplex](unsafeUninitializedCapacity: 16) { buffer, count in
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

    /// Apply gate using GPU acceleration
    /// - Parameters:
    ///   - gate: Quantum gate to apply
    ///   - qubits: Target qubit indices
    ///   - state: Current quantum state
    /// - Returns: Transformed state
    @_eagerMove
    public func apply(gate: QuantumGate, to qubits: [Int], state: QuantumState) -> QuantumState {
        switch gate {
        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard,
             .phase, .sGate, .tGate, .rotationX, .rotationY, .rotationZ,
             .u1, .u2, .u3, .sx, .sy, .customSingleQubit:
            applySingleQubitGate(gate: gate, qubit: qubits[0], state: state)

        case let .cnot(control, target):
            applyCNOT(control: control, target: target, state: state)

        case let .controlledPhase(_, control, target),
             let .controlledRotationX(_, control, target),
             let .controlledRotationY(_, control, target),
             let .controlledRotationZ(_, control, target),
             let .swap(control, target),
             let .sqrtSwap(control, target),
             let .cz(control, target),
             let .cy(control, target),
             let .ch(control, target),
             let .customTwoQubit(_, control, target):
            applyTwoQubitGate(gate: gate, control: control, target: target, state: state)

        case let .toffoli(c1, c2, target):
            applyToffoli(control1: c1, control2: c2, target: target, state: state)
        }
    }

    // MARK: - Private Metal Implementations

    @_optimize(speed)
    @_eagerMove
    private func applySingleQubitGate(gate: QuantumGate, qubit: Int, state: QuantumState) -> QuantumState {
        var floatAmplitudes = Self.toGPUAmplitudes(state.amplitudes)
        var floatMatrix = Self.toGPUMatrix2x2(gate.matrix())

        let stateSize = state.stateSpaceSize
        let bufferSize = stateSize * MemoryLayout<GPUComplex>.stride

        guard let amplitudeBuffer = device.makeBuffer(
            bytes: &floatAmplitudes,
            length: bufferSize,
            options: .storageModeShared
        ) else { return GateApplication.apply(gate: gate, to: [qubit], state: state) }

        let matrixSize = 4 * MemoryLayout<GPUComplex>.stride
        guard let matrixBuffer = device.makeBuffer(
            bytes: &floatMatrix,
            length: matrixSize,
            options: .storageModeShared
        ) else { return GateApplication.apply(gate: gate, to: [qubit], state: state) }

        var qubitValue = UInt32(qubit)
        var numQubitsValue = UInt32(state.numQubits)

        guard let (commandBuffer, encoder) = MetalUtilities.createCommandEncoder(queue: commandQueue) else {
            return GateApplication.apply(gate: gate, to: [qubit], state: state)
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

        let resultPointer = amplitudeBuffer.contents().bindMemory(to: GPUComplex.self, capacity: stateSize)
        let newAmplitudes = Self.fromGPUAmplitudes(resultPointer, count: stateSize)

        guard newAmplitudes.allSatisfy(\.isFinite) else {
            return GateApplication.apply(gate: gate, to: [qubit], state: state)
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    @_optimize(speed)
    @_eagerMove
    private func applyCNOT(control: Int, target: Int, state: QuantumState) -> QuantumState {
        var floatAmplitudes = Self.toGPUAmplitudes(state.amplitudes)
        let stateSize = state.stateSpaceSize
        let bufferSize = stateSize * MemoryLayout<GPUComplex>.stride

        guard let inputBuffer = device.makeBuffer(bytes: &floatAmplitudes, length: bufferSize, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
        else { return GateApplication.apply(gate: .cnot(control: control, target: target), to: [], state: state) }

        guard let (commandBuffer, encoder) = MetalUtilities.createCommandEncoder(queue: commandQueue) else {
            return GateApplication.apply(gate: .cnot(control: control, target: target), to: [], state: state)
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

        let resultPointer = outputBuffer.contents().bindMemory(to: GPUComplex.self, capacity: stateSize)
        let newAmplitudes = Self.fromGPUAmplitudes(resultPointer, count: stateSize)

        guard newAmplitudes.allSatisfy(\.isFinite) else {
            return GateApplication.apply(gate: .cnot(control: control, target: target), to: [], state: state)
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    @_optimize(speed)
    @_eagerMove
    private func applyTwoQubitGate(gate: QuantumGate, control: Int, target: Int, state: QuantumState) -> QuantumState {
        var floatAmplitudes = Self.toGPUAmplitudes(state.amplitudes)
        var floatMatrix = Self.toGPUMatrix4x4(gate.matrix())

        let stateSize = state.stateSpaceSize
        let bufferSize = stateSize * MemoryLayout<GPUComplex>.stride
        let matrixSize = 16 * MemoryLayout<GPUComplex>.stride

        guard let inputBuffer = device.makeBuffer(bytes: &floatAmplitudes, length: bufferSize, options: .storageModeShared),
              let matrixBuffer = device.makeBuffer(bytes: &floatMatrix, length: matrixSize, options: .storageModeShared)
        else { return GateApplication.apply(gate: gate, to: [control, target], state: state) }

        guard let (commandBuffer, encoder) = MetalUtilities.createCommandEncoder(queue: commandQueue) else {
            return GateApplication.apply(gate: gate, to: [control, target], state: state)
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

        let resultPointer = inputBuffer.contents().bindMemory(to: GPUComplex.self, capacity: stateSize)
        let newAmplitudes = Self.fromGPUAmplitudes(resultPointer, count: stateSize)

        guard newAmplitudes.allSatisfy(\.isFinite) else {
            return GateApplication.apply(gate: gate, to: [control, target], state: state)
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    @_optimize(speed)
    @_eagerMove
    private func applyToffoli(control1: Int, control2: Int, target: Int, state: QuantumState) -> QuantumState {
        var floatAmplitudes = Self.toGPUAmplitudes(state.amplitudes)
        let stateSize = state.stateSpaceSize
        let bufferSize = stateSize * MemoryLayout<GPUComplex>.stride

        guard let inputBuffer = device.makeBuffer(bytes: &floatAmplitudes, length: bufferSize, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
        else { return GateApplication.apply(gate: .toffoli(control1: control1, control2: control2, target: target), to: [], state: state) }

        guard let (commandBuffer, encoder) = MetalUtilities.createCommandEncoder(queue: commandQueue) else {
            return GateApplication.apply(gate: .toffoli(control1: control1, control2: control2, target: target), to: [], state: state)
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

        let resultPointer = outputBuffer.contents().bindMemory(to: GPUComplex.self, capacity: stateSize)
        let newAmplitudes = Self.fromGPUAmplitudes(resultPointer, count: stateSize)

        guard newAmplitudes.allSatisfy(\.isFinite) else {
            return GateApplication.apply(gate: .toffoli(control1: control1, control2: control2, target: target), to: [], state: state)
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }
}

// MARK: - Hybrid CPU/GPU Gate Application

public extension GateApplication {
    /// Apply gate with automatic CPU/GPU selection
    /// Uses GPU for states with >= 10 qubits
    @inlinable
    @_eagerMove
    static func applyHybrid(gate: QuantumGate, to qubits: [Int], state: QuantumState) async -> QuantumState {
        if state.numQubits >= MetalGateApplication.gpuThreshold {
            if let metalApp = MetalGateApplication() {
                return await metalApp.apply(gate: gate, to: qubits, state: state)
            }
        }

        return apply(gate: gate, to: qubits, state: state)
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
@frozen
public enum MetalUtilities {
    /// Load Metal shader library with 3-tier fallback strategy
    ///
    /// Attempts to load QuantumGPU shader library in order of performance preference:
    /// 1. Pre-compiled `default.metallib` file (fastest, production deployments)
    /// 2. Default library from app bundle (Xcode builds with Metal files)
    /// 3. Runtime compilation from `QuantumGPU.metal` source (development/non-standard)
    ///
    /// **Error Handling**: Returns nil if all three loading strategies fail. Caller must
    /// handle Metal unavailability with CPU fallback.
    ///
    /// - Parameter device: Metal device to create library for
    /// - Returns: Shader library if any loading strategy succeeds, nil otherwise
    @_effects(readonly)
    public static func loadLibrary(device: MTLDevice) -> MTLLibrary? {
        // Strategy 1: Pre-compiled metallib (xcode)
        if let metallibURL = Bundle.module.url(forResource: "default", withExtension: "metallib"),
           let library = try? device.makeLibrary(URL: metallibURL)
        { return library }

        // Strategy 2: Default library
        if let defaultLibrary = device.makeDefaultLibrary() { return defaultLibrary }

        // Strategy 3: Runtime source compilation (CLI)
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
