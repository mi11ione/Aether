//
//  MetalGateApplication.swift
//  Aether
//
//  Metal-accelerated quantum gate application for large quantum states
//  Created by mi11ion on 21/10/25.
//

import Foundation
import Metal

/// Shared Metal resources for efficient pipeline compilation
/// Pipelines are compiled once and reused across all instances
private enum MetalResources {
    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()
    static let commandQueue: MTLCommandQueue? = device?.makeCommandQueue()
    static let library: MTLLibrary? = device?.makeDefaultLibrary()

    // Lazily compiled pipeline states (shared across all instances)
    static let singleQubitPipeline: MTLComputePipelineState? = {
        guard let device, let library,
              let function = library.makeFunction(name: "applySingleQubitGate")
        else {
            return nil
        }
        return try? device.makeComputePipelineState(function: function)
    }()

    static let cnotPipeline: MTLComputePipelineState? = {
        guard let device, let library,
              let function = library.makeFunction(name: "applyCNOT")
        else {
            return nil
        }
        return try? device.makeComputePipelineState(function: function)
    }()

    static let twoQubitPipeline: MTLComputePipelineState? = {
        guard let device, let library,
              let function = library.makeFunction(name: "applyTwoQubitGate")
        else {
            return nil
        }
        return try? device.makeComputePipelineState(function: function)
    }()

    static let toffoliPipeline: MTLComputePipelineState? = {
        guard let device, let library,
              let function = library.makeFunction(name: "applyToffoli")
        else {
            return nil
        }
        return try? device.makeComputePipelineState(function: function)
    }()
}

/// GPU-accelerated gate application using Metal compute shaders
/// Automatically switches between CPU and GPU based on qubit count
/// Shares compiled pipeline states across all instances for efficiency
final class MetalGateApplication {
    // MARK: - Metal Resources (shared via MetalResources)

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue

    // Compute pipeline states (shared, not per-instance)
    private let singleQubitPipeline: MTLComputePipelineState
    private let cnotPipeline: MTLComputePipelineState
    private let twoQubitPipeline: MTLComputePipelineState
    private let toffoliPipeline: MTLComputePipelineState

    // MARK: - Configuration

    /// Threshold: use GPU for states with >= this many qubits
    static let gpuThreshold = 10

    // MARK: - Initialization

    init?() {
        // Use shared Metal resources
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
    func apply(gate: QuantumGate, to qubits: [Int], state: QuantumState) -> QuantumState {
        // Dispatch based on gate type
        switch gate {
        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard,
             .phase, .sGate, .tGate, .rotationX, .rotationY, .rotationZ:
            applySingleQubitGate(gate: gate, qubit: qubits[0], state: state)

        case let .cnot(control, target):
            applyCNOT(control: control, target: target, state: state)

        case let .controlledPhase(_, control, target),
             let .swap(control, target):
            applyTwoQubitGate(gate: gate, control: control, target: target, state: state)

        case let .toffoli(c1, c2, target):
            applyToffoli(control1: c1, control2: c2, target: target, state: state)
        }
    }

    // MARK: - Private Metal Implementations

    private func applySingleQubitGate(gate: QuantumGate, qubit: Int, state: QuantumState) -> QuantumState {
        // Convert amplitudes to Float for GPU
        var floatAmplitudes = state.amplitudes.map { amp in
            (Float(amp.real), Float(amp.imaginary))
        }

        // Get gate matrix as Float
        let gateMatrix = gate.matrix()
        var floatMatrix = gateMatrix.flatMap { row in
            row.flatMap { [(Float($0.real), Float($0.imaginary))] }
        }

        let stateSize = state.stateSpaceSize
        let bufferSize = stateSize * MemoryLayout<(Float, Float)>.stride

        // Create Metal buffers
        guard let amplitudeBuffer = device.makeBuffer(
            bytes: &floatAmplitudes,
            length: bufferSize,
            options: .storageModeShared
        ) else {
            print("Metal amplitude buffer allocation failed - falling back to CPU")
            return GateApplication.apply(gate: gate, to: [qubit], state: state)
        }

        let matrixSize = 4 * MemoryLayout<(Float, Float)>.stride
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

        // Create command buffer and encoder
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            print("Metal command buffer/encoder creation failed - falling back to CPU")
            return GateApplication.apply(gate: gate, to: [qubit], state: state)
        }

        encoder.setComputePipelineState(singleQubitPipeline)
        encoder.setBuffer(amplitudeBuffer, offset: 0, index: 0)
        encoder.setBytes(&qubitValue, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBuffer(matrixBuffer, offset: 0, index: 2)
        encoder.setBytes(&numQubitsValue, length: MemoryLayout<UInt32>.stride, index: 3)

        // Configure thread execution
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

        // Read back results
        let resultPointer = amplitudeBuffer.contents().bindMemory(
            to: (Float, Float).self,
            capacity: stateSize
        )

        let newAmplitudes = (0 ..< stateSize).map { i -> Complex<Double> in
            let (real, imag) = resultPointer[i]
            return Complex(Double(real), Double(imag))
        }

        // Validate GPU computation results
        guard newAmplitudes.allSatisfy(\.isFinite) else {
            print("GPU computation produced invalid values (NaN/Inf) - falling back to CPU")
            return GateApplication.apply(gate: gate, to: [qubit], state: state)
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    private func applyCNOT(control: Int, target: Int, state: QuantumState) -> QuantumState {
        var floatAmplitudes = state.amplitudes.map { (Float($0.real), Float($0.imaginary)) }
        let stateSize = state.stateSpaceSize
        let bufferSize = stateSize * MemoryLayout<(Float, Float)>.stride

        guard let inputBuffer = device.makeBuffer(bytes: &floatAmplitudes, length: bufferSize, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            print("Metal buffer allocation failed for CNOT - falling back to CPU")
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

        let resultPointer = outputBuffer.contents().bindMemory(to: (Float, Float).self, capacity: stateSize)
        let newAmplitudes = (0 ..< stateSize).map { Complex(Double(resultPointer[$0].0), Double(resultPointer[$0].1)) }

        // Validate GPU computation results
        guard newAmplitudes.allSatisfy(\.isFinite) else {
            print("GPU CNOT computation produced invalid values (NaN/Inf) - falling back to CPU")
            return GateApplication.apply(gate: .cnot(control: control, target: target), to: [], state: state)
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    private func applyTwoQubitGate(gate: QuantumGate, control: Int, target: Int, state: QuantumState) -> QuantumState {
        // Full GPU implementation for two-qubit gates using 4x4 matrix
        var floatAmplitudes = state.amplitudes.map { (Float($0.real), Float($0.imaginary)) }
        let stateSize = state.stateSpaceSize
        let bufferSize = stateSize * MemoryLayout<(Float, Float)>.stride

        // Get gate matrix and convert to Float
        let gateMatrix = gate.matrix()
        var floatMatrix: [(Float, Float)] = []
        for row in gateMatrix {
            for element in row {
                floatMatrix.append((Float(element.real), Float(element.imaginary)))
            }
        }

        let matrixSize = 16 * MemoryLayout<(Float, Float)>.stride

        guard let inputBuffer = device.makeBuffer(bytes: &floatAmplitudes, length: bufferSize, options: .storageModeShared),
              let matrixBuffer = device.makeBuffer(bytes: &floatMatrix, length: matrixSize, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            // If Metal fails, fall back to CPU (device limitation, not implementation)
            print("Metal buffer allocation failed for two-qubit gate - falling back to CPU")
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

        let resultPointer = inputBuffer.contents().bindMemory(to: (Float, Float).self, capacity: stateSize)
        let newAmplitudes = (0 ..< stateSize).map { Complex(Double(resultPointer[$0].0), Double(resultPointer[$0].1)) }

        // Validate GPU computation results
        guard newAmplitudes.allSatisfy(\.isFinite) else {
            print("GPU two-qubit gate computation produced invalid values (NaN/Inf) - falling back to CPU")
            return GateApplication.apply(gate: gate, to: [control, target], state: state)
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    private func applyToffoli(control1: Int, control2: Int, target: Int, state: QuantumState) -> QuantumState {
        var floatAmplitudes = state.amplitudes.map { (Float($0.real), Float($0.imaginary)) }
        let stateSize = state.stateSpaceSize
        let bufferSize = stateSize * MemoryLayout<(Float, Float)>.stride

        guard let inputBuffer = device.makeBuffer(bytes: &floatAmplitudes, length: bufferSize, options: .storageModeShared),
              let outputBuffer = device.makeBuffer(length: bufferSize, options: .storageModeShared),
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else {
            print("Metal buffer allocation failed for Toffoli - falling back to CPU")
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

        let resultPointer = outputBuffer.contents().bindMemory(to: (Float, Float).self, capacity: stateSize)
        let newAmplitudes = (0 ..< stateSize).map { Complex(Double(resultPointer[$0].0), Double(resultPointer[$0].1)) }

        // Validate GPU computation results
        guard newAmplitudes.allSatisfy(\.isFinite) else {
            print("GPU Toffoli computation produced invalid values (NaN/Inf) - falling back to CPU")
            return GateApplication.apply(gate: .toffoli(control1: control1, control2: control2, target: target), to: [], state: state)
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }
}

// MARK: - Hybrid CPU/GPU Gate Application

extension GateApplication {
    /// Apply gate with automatic CPU/GPU selection
    /// Uses GPU for states with >= 10 qubits
    static func applyHybrid(gate: QuantumGate, to qubits: [Int], state: QuantumState) -> QuantumState {
        // Use GPU for large states
        if state.numQubits >= MetalGateApplication.gpuThreshold {
            if let metalApp = MetalGateApplication() {
                return metalApp.apply(gate: gate, to: qubits, state: state)
            }
        }

        // Fall back to CPU
        return apply(gate: gate, to: qubits, state: state)
    }
}
