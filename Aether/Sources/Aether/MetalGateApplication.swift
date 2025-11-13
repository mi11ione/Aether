// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation
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
/// overhead exceeds benefits. Achieves 2-10x speedup for states with ≥10 qubits.
///
/// **Performance characteristics**:
/// - GPU threshold: Accelerates states with ≥10 qubits (2^10 = 1024 amplitudes)
/// - Speedup: 2-10x depending on hardware (M1/M2/M3 chips)
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
/// let result = try await simulator.execute(largeCircuit)
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

    public init?() {
        guard let device = MetalResources.device,
              let commandQueue = MetalResources.commandQueue,
              let singleQubitPipeline = MetalResources.singleQubitPipeline,
              let cnotPipeline = MetalResources.cnotPipeline,
              let twoQubitPipeline = MetalResources.twoQubitPipeline,
              let toffoliPipeline = MetalResources.toffoliPipeline
        else {
            print("Metal resources not available on this device")
            return nil
        }

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
        // Convert amplitudes to Float for GPU
        var floatAmplitudes: [GPUComplex] = state.amplitudes.map { amp in
            (Float(amp.real), Float(amp.imaginary))
        }

        var floatMatrix: [GPUComplex] = gate.matrix().flatMap { row in
            row.flatMap { [(Float($0.real), Float($0.imaginary))] }
        }

        let stateSize: Int = state.stateSpaceSize
        let bufferSize: Int = stateSize * MemoryLayout<GPUComplex>.stride

        guard let amplitudeBuffer = device.makeBuffer(
            bytes: &floatAmplitudes,
            length: bufferSize,
            options: .storageModeShared
        ) else {
            print("Metal amplitude buffer allocation failed - falling back to CPU")
            return GateApplication.apply(gate: gate, to: [qubit], state: state)
        }

        let matrixSize = 4 * MemoryLayout<GPUComplex>.stride
        guard let matrixBuffer = device.makeBuffer(
            bytes: &floatMatrix,
            length: matrixSize,
            options: .storageModeShared
        ) else {
            print("Metal matrix buffer allocation failed - falling back to CPU")
            return GateApplication.apply(gate: gate, to: [qubit], state: state)
        }

        var qubitValue = UInt32(qubit)
        var numQubitsValue = UInt32(state.numQubits)

        guard let (commandBuffer, encoder) = MetalUtilities.createCommandEncoder(queue: commandQueue) else {
            print("Metal command buffer/encoder creation failed - falling back to CPU")
            return GateApplication.apply(gate: gate, to: [qubit], state: state)
        }

        encoder.setComputePipelineState(singleQubitPipeline)
        encoder.setBuffer(amplitudeBuffer, offset: 0, index: 0)
        encoder.setBytes(&qubitValue, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBuffer(matrixBuffer, offset: 0, index: 2)
        encoder.setBytes(&numQubitsValue, length: MemoryLayout<UInt32>.stride, index: 3)

        let threadsPerGroup = MTLSize(width: min(singleQubitPipeline.maxTotalThreadsPerThreadgroup, stateSize), height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (stateSize + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: 1,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let resultPointer = amplitudeBuffer.contents().bindMemory(
            to: GPUComplex.self,
            capacity: stateSize
        )

        let newAmplitudes: AmplitudeVector = (0 ..< stateSize).map { i -> Complex<Double> in
            let (real, imag): GPUComplex = resultPointer[i]
            return Complex(Double(real), Double(imag))
        }

        guard newAmplitudes.allSatisfy(\.isFinite) else {
            print("GPU computation produced invalid values (NaN/Inf) - falling back to CPU")
            return GateApplication.apply(gate: gate, to: [qubit], state: state)
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    @_optimize(speed)
    @_eagerMove
    private func applyCNOT(control: Int, target: Int, state: QuantumState) -> QuantumState {
        var floatAmplitudes: [GPUComplex] = state.amplitudes.map { (Float($0.real), Float($0.imaginary)) }
        let stateSize: Int = state.stateSpaceSize
        let bufferSize: Int = stateSize * MemoryLayout<GPUComplex>.stride

        guard let inputBuffer = device.makeBuffer(bytes: &floatAmplitudes, length: bufferSize, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
        else {
            print("Metal buffer allocation failed for CNOT - falling back to CPU")
            return GateApplication.apply(gate: .cnot(control: control, target: target), to: [], state: state)
        }

        guard let (commandBuffer, encoder) = MetalUtilities.createCommandEncoder(queue: commandQueue) else {
            print("Metal command buffer/encoder creation failed for CNOT - falling back to CPU")
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
        let newAmplitudes: AmplitudeVector = (0 ..< stateSize).map { Complex(Double(resultPointer[$0].0), Double(resultPointer[$0].1)) }

        guard newAmplitudes.allSatisfy(\.isFinite) else {
            print("GPU CNOT computation produced invalid values (NaN/Inf) - falling back to CPU")
            return GateApplication.apply(gate: .cnot(control: control, target: target), to: [], state: state)
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    @_optimize(speed)
    @_eagerMove
    private func applyTwoQubitGate(gate: QuantumGate, control: Int, target: Int, state: QuantumState) -> QuantumState {
        var floatAmplitudes: [GPUComplex] = state.amplitudes.map { (Float($0.real), Float($0.imaginary)) }
        let stateSize: Int = state.stateSpaceSize
        let bufferSize: Int = stateSize * MemoryLayout<GPUComplex>.stride

        var floatMatrix: [GPUComplex] = []
        for row in gate.matrix() {
            for element in row {
                floatMatrix.append((Float(element.real), Float(element.imaginary)))
            }
        }

        let matrixSize = 16 * MemoryLayout<GPUComplex>.stride

        guard let inputBuffer = device.makeBuffer(bytes: &floatAmplitudes, length: bufferSize, options: .storageModeShared),
              let matrixBuffer = device.makeBuffer(bytes: &floatMatrix, length: matrixSize, options: .storageModeShared)
        else {
            print("Metal buffer allocation failed for two-qubit gate - falling back to CPU")
            return GateApplication.apply(gate: gate, to: [control, target], state: state)
        }

        guard let (commandBuffer, encoder) = MetalUtilities.createCommandEncoder(queue: commandQueue) else {
            print("Metal command buffer/encoder creation failed for two-qubit gate - falling back to CPU")
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

        let threadsPerGroup = MTLSize(width: min(twoQubitPipeline.maxTotalThreadsPerThreadgroup, stateSize), height: 1, depth: 1)
        let threadgroups = MTLSize(width: (stateSize + threadsPerGroup.width - 1) / threadsPerGroup.width, height: 1, depth: 1)

        encoder.dispatchThreadgroups(threadgroups, threadsPerThreadgroup: threadsPerGroup)
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        let resultPointer = inputBuffer.contents().bindMemory(to: GPUComplex.self, capacity: stateSize)
        let newAmplitudes: AmplitudeVector = (0 ..< stateSize).map { Complex(Double(resultPointer[$0].0), Double(resultPointer[$0].1)) }

        guard newAmplitudes.allSatisfy(\.isFinite) else {
            print("GPU two-qubit gate computation produced invalid values (NaN/Inf) - falling back to CPU")
            return GateApplication.apply(gate: gate, to: [control, target], state: state)
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    @_optimize(speed)
    @_eagerMove
    private func applyToffoli(control1: Int, control2: Int, target: Int, state: QuantumState) -> QuantumState {
        var floatAmplitudes: [GPUComplex] = state.amplitudes.map { (Float($0.real), Float($0.imaginary)) }
        let stateSize: Int = state.stateSpaceSize
        let bufferSize: Int = stateSize * MemoryLayout<GPUComplex>.stride

        guard let inputBuffer = device.makeBuffer(bytes: &floatAmplitudes, length: bufferSize, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared)
        else {
            print("Metal buffer allocation failed for Toffoli - falling back to CPU")
            return GateApplication.apply(gate: .toffoli(control1: control1, control2: control2, target: target), to: [], state: state)
        }

        guard let (commandBuffer, encoder) = MetalUtilities.createCommandEncoder(queue: commandQueue) else {
            print("Metal command buffer/encoder creation failed for Toffoli - falling back to CPU")
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
        let newAmplitudes: AmplitudeVector = (0 ..< stateSize).map { Complex(Double(resultPointer[$0].0), Double(resultPointer[$0].1)) }

        guard newAmplitudes.allSatisfy(\.isFinite) else {
            print("GPU Toffoli computation produced invalid values (NaN/Inf) - falling back to CPU")
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
