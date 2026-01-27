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
/// across thousands of concurrent GPU threads. Single-qubit gates dispatch one thread per amplitude pair
/// where indices differ only in the target qubit bit. Two-qubit gates dispatch one thread per amplitude
/// quartet for control/target combinations. CNOT, CZ, and Toffoli use parallel conditional swaps matching
/// the CPU optimization. Falls back to ``GateApplication`` if buffer allocation fails or computation
/// produces NaN/Inf.
///
/// The same unitary transformation |ψ'⟩ = U|ψ⟩ applies as in CPU implementation. GPU parallelism exploits
/// the independence of amplitude pair updates: for single-qubit gate on qubit q, pairs (i, i⊕2^q) update
/// independently, mapping perfectly to GPU SIMD execution.
///
/// GPU uses Float32 vs ``GateApplication`` Float64. For most quantum algorithms, precision loss is
/// negligible compared to typical gate fidelity (99.9%) and decoherence in real hardware. vDSP handles
/// Float64<->Float32 conversion via vectorized operations. Below 10 qubits CPU is faster due to buffer
/// allocation and shader dispatch overhead. At ≥10 qubits GPU parallelism saturates compute units and
/// amortizes overhead. Shared pipelines compile shaders once at launch and reuse for all instances.
///
/// **Example:**
/// ```swift
/// let state = QuantumState(qubits: 12)
/// let metalApp = MetalGateApplication()!
/// let superposition = await metalApp.apply(.hadamard, to: 0, state: state)
/// let entangled = await metalApp.apply(.cnot, to: [0, 1], state: superposition)
/// ```
///
/// - Note: Failable initializer returns nil if Metal unavailable
/// - SeeAlso: ``GateApplication``
/// - SeeAlso: ``MetalUtilities``
/// - SeeAlso: ``QuantumSimulator``
public actor MetalGateApplication {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let singleQubitPipeline: MTLComputePipelineState
    private let cnotPipeline: MTLComputePipelineState
    private let twoQubitPipeline: MTLComputePipelineState
    private let toffoliPipeline: MTLComputePipelineState

    /// Minimum qubit count for GPU acceleration under default (fast) policy.
    ///
    /// Below this threshold (n < 10), CPU execution via ``GateApplication`` is faster due to GPU
    /// overhead (buffer allocation, shader dispatch, Float64->Float32 conversion). At n=10 (1024 amplitudes),
    /// parallelism benefit begins to outweigh overhead.
    ///
    /// For policy-aware threshold selection, use ``minimumQubitCountForGPU(policy:)`` instead.
    ///
    /// - SeeAlso: ``minimumQubitCountForGPU(policy:)``
    /// - SeeAlso: ``PrecisionPolicy``
    public static let minimumQubitCountForGPU = 10

    /// Minimum qubit count for GPU acceleration under specified precision policy.
    ///
    /// Returns policy-specific threshold: `.fast` = 10, `.balanced` = 12, `.accurate` = Int.max.
    /// States with fewer qubits than the threshold use CPU execution for better precision
    /// or lower overhead.
    ///
    /// **Example:**
    /// ```swift
    /// let threshold = MetalGateApplication.minimumQubitCountForGPU(policy: .balanced)
    /// ```
    ///
    /// - Parameter policy: Precision policy governing GPU threshold
    /// - Returns: Minimum qubit count for GPU acceleration under this policy
    /// - Complexity: O(1)
    /// - SeeAlso: ``PrecisionPolicy``
    @_effects(readonly)
    @inlinable
    public static func minimumQubitCountForGPU(policy: PrecisionPolicy) -> Int {
        policy.gpuQubitThreshold
    }

    // MARK: - Conversion Helpers

    /// Convert Complex<Double> amplitudes to GPU-compatible Float pairs using vDSP
    @_effects(readonly)
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
    @_effects(readonly)
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
    @_effects(readonly)
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
    @_effects(readonly)
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
    /// Returns nil if Metal unavailable or shader compilation fails (missing QuantumGPU.metal, malformed metallib).
    /// Reuses pre-compiled compute pipelines from `MetalResources` singleton to avoid expensive re-compilation.
    ///
    /// **Example:**
    /// ```swift
    /// if let metalApp = MetalGateApplication() {
    ///     let result = await metalApp.apply(.hadamard, to: 0, state: state)
    /// } else {
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
    /// **Example:**
    /// ```swift
    /// let state = QuantumState(qubits: 12)
    /// let metalApp = MetalGateApplication()!
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
    /// - Precondition: All qubit indices must be valid for state
    @_eagerMove
    public func apply(_ gate: QuantumGate, to qubits: [Int], state: QuantumState) -> QuantumState {
        switch gate {
        case let .globalPhase(phi):
            ValidationUtilities.validateConcrete(phi, name: "global phase angle")
            return GateApplication.applyGlobalPhase(phi: phi.evaluate(using: [:]), state: state)

        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard,
             .phase, .sGate, .tGate, .rotationX, .rotationY, .rotationZ,
             .u1, .u2, .u3, .sx, .sy, .customSingleQubit:
            return applySingleQubitGate(gate: gate, qubit: qubits[0], state: state)

        case .cnot:
            return applyCNOT(control: qubits[0], target: qubits[1], state: state)

        case .controlledPhase, .controlledRotationX, .controlledRotationY, .controlledRotationZ, .swap, .sqrtSwap, .iswap, .sqrtISwap, .fswap, .givens, .xx, .yy, .zz, .cz, .cy, .ch, .customTwoQubit:
            return applyTwoQubitGate(gate: gate, control: qubits[0], target: qubits[1], state: state)

        case .toffoli, .fredkin:
            return applyToffoli(control1: qubits[0], control2: qubits[1], target: qubits[2], state: state)

        case .ccz:
            return GateApplication.applyCCZ(qubit1: qubits[0], qubit2: qubits[1], qubit3: qubits[2], state: state)

        case let .controlled(innerGate, controls):
            return applyControlledGate(gate: innerGate, controls: controls, targetQubits: qubits, state: state)

        case .customUnitary, .diagonal, .multiplexor:
            return GateApplication.applyMultiQubitGate(gate: gate, qubits: qubits, state: state)
        }
    }

    /// Apply gate to single qubit (convenience method)
    ///
    /// Wraps qubit index in array and delegates to main apply method.
    /// Preferred syntax for single-qubit gates.
    ///
    /// **Example:**
    /// ```swift
    /// let metalApp = MetalGateApplication()!
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

    /// Applies a circuit operation to a quantum state using GPU acceleration where available.
    ///
    /// Routes unitary gates through the Metal-accelerated pipeline and non-unitary operations
    /// through dedicated CPU handlers.
    ///
    /// - Parameters:
    ///   - operation: The circuit operation to apply.
    ///   - state: The quantum state to transform.
    /// - Returns: The transformed quantum state.
    /// - Complexity: O(2^n) where n is the number of qubits.
    ///
    /// **Example:**
    /// ```swift
    /// let gpu = MetalGateApplication()
    /// let state = QuantumState(qubits: 2)
    /// let op = CircuitOperation.gate(.hadamard, qubits: [0])
    /// let result = gpu.apply(op, state: state)
    /// ```
    public func apply(_ operation: CircuitOperation, state: QuantumState) -> QuantumState {
        switch operation {
        case let .gate(gate, qubits, _):
            apply(gate, to: qubits, state: state)
        case let .reset(qubit, _):
            GateApplication.applyReset(qubit: qubit, state: state)
        }
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
            options: .storageModeShared,
        ) else { return GateApplication.apply(gate, to: qubit, state: state) }

        let matrixSize = 4 * MemoryLayout<(Float, Float)>.stride
        guard let matrixBuffer = device.makeBuffer(
            bytes: &floatMatrix,
            length: matrixSize,
            options: .storageModeShared,
        ) else { return GateApplication.apply(gate, to: qubit, state: state) }

        var qubitValue = UInt32(qubit)
        var qubitsValue = UInt32(state.qubits)

        guard let (commandBuffer, encoder) = MetalUtilities.createCommandEncoder(queue: commandQueue) else {
            return GateApplication.apply(gate, to: qubit, state: state)
        }

        encoder.setComputePipelineState(singleQubitPipeline)
        encoder.setBuffer(amplitudeBuffer, offset: 0, index: 0)
        encoder.setBytes(&qubitValue, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBuffer(matrixBuffer, offset: 0, index: 2)
        encoder.setBytes(&qubitsValue, length: MemoryLayout<UInt32>.stride, index: 3)

        let numPairs = stateSize / 2
        let threadsPerGroup = MTLSize(width: min(singleQubitPipeline.maxTotalThreadsPerThreadgroup, numPairs), height: 1, depth: 1)
        let threadgroups = MTLSize(
            width: (numPairs + threadsPerGroup.width - 1) / threadsPerGroup.width,
            height: 1,
            depth: 1,
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

        return QuantumState(qubits: state.qubits, amplitudes: newAmplitudes)
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
        var qubitsValue = UInt32(state.qubits)

        encoder.setComputePipelineState(cnotPipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBytes(&controlValue, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBytes(&targetValue, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&qubitsValue, length: MemoryLayout<UInt32>.stride, index: 3)
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

        return QuantumState(qubits: state.qubits, amplitudes: newAmplitudes)
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
        var qubitsValue = UInt32(state.qubits)

        encoder.setComputePipelineState(twoQubitPipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBytes(&controlValue, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBytes(&targetValue, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBuffer(matrixBuffer, offset: 0, index: 3)
        encoder.setBytes(&qubitsValue, length: MemoryLayout<UInt32>.stride, index: 4)

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

        return QuantumState(qubits: state.qubits, amplitudes: newAmplitudes)
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
        var qubitsValue = UInt32(state.qubits)

        encoder.setComputePipelineState(toffoliPipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBytes(&c1Value, length: MemoryLayout<UInt32>.stride, index: 1)
        encoder.setBytes(&c2Value, length: MemoryLayout<UInt32>.stride, index: 2)
        encoder.setBytes(&targetValue, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&qubitsValue, length: MemoryLayout<UInt32>.stride, index: 4)
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

        return QuantumState(qubits: state.qubits, amplitudes: newAmplitudes)
    }

    @_eagerMove
    private func applyControlledGate(
        gate: QuantumGate,
        controls: [Int],
        targetQubits: [Int],
        state: QuantumState,
    ) -> QuantumState {
        GateApplication.applyControlledGate(gate: gate, controls: controls, targetQubits: targetQubits, state: state)
    }
}

// MARK: - Hybrid CPU/GPU Gate Application

/// Cached GPU executor for hybrid CPU/GPU gate application
private let sharedMetalGateApplication: MetalGateApplication? = MetalGateApplication()

public extension GateApplication {
    /// Apply gate with automatic CPU/GPU selection based on state size and precision policy.
    ///
    /// Routes to GPU (``MetalGateApplication``) when state size exceeds policy threshold,
    /// otherwise uses CPU. Automatically falls back to CPU if Metal unavailable or GPU
    /// computation fails. Policy-aware threshold selection: `.fast` = 10 qubits,
    /// `.balanced` = 12 qubits, `.accurate` = CPU only.
    ///
    /// **Example:**
    /// ```swift
    /// let state = QuantumState(qubits: 11)
    /// let r1 = await GateApplication.applyHybrid(.hadamard, to: 0, state: state)  // GPU (.fast)
    /// let r2 = await GateApplication.applyHybrid(.hadamard, to: 0, state: state, policy: .balanced)  // CPU
    /// let r3 = await GateApplication.applyHybrid(.hadamard, to: 0, state: state, policy: .accurate)  // CPU
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply
    ///   - qubits: Target qubit indices
    ///   - state: Input quantum state
    ///   - policy: Precision policy governing GPU threshold (default: `.fast`)
    /// - Returns: Transformed state (via GPU or CPU)
    /// - Complexity: O(2^n) time
    /// - SeeAlso: ``PrecisionPolicy``
    @_eagerMove
    static func applyHybrid(
        _ gate: QuantumGate,
        to qubits: [Int],
        state: QuantumState,
        policy: PrecisionPolicy = .fast,
    ) async -> QuantumState {
        if PrecisionPolicy.shouldUseGPU(qubits: state.qubits, policy: policy),
           let metalApp = sharedMetalGateApplication
        {
            return await metalApp.apply(gate, to: qubits, state: state)
        }

        return apply(gate, to: qubits, state: state)
    }

    /// Apply gate to single qubit with automatic CPU/GPU selection.
    ///
    /// Wraps qubit index in array and delegates to main applyHybrid method.
    ///
    /// **Example:**
    /// ```swift
    /// let result = await GateApplication.applyHybrid(.hadamard, to: 0, state: state, policy: .balanced)
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply
    ///   - qubit: Target qubit index
    ///   - state: Input quantum state
    ///   - policy: Precision policy governing GPU threshold (default: `.fast`)
    /// - Returns: Transformed state (via GPU or CPU)
    /// - Complexity: O(2^n) time
    /// - SeeAlso: ``PrecisionPolicy``
    @_eagerMove
    static func applyHybrid(
        _ gate: QuantumGate,
        to qubit: Int,
        state: QuantumState,
        policy: PrecisionPolicy = .fast,
    ) async -> QuantumState {
        await applyHybrid(gate, to: [qubit], state: state, policy: policy)
    }
}

/// Shared Metal utilities for GPU-accelerated quantum computing
///
/// Centralizes common Metal device initialization, library loading, and command buffer
/// creation patterns used across ``MetalGateApplication`` and ``SparseHamiltonian`` backends.
/// Implements 3-tier fallback strategy for maximum compatibility: first tries pre-compiled
/// metallib (fastest), then default library from app bundle, finally runtime source compilation
/// for non-standard deployments. All methods are stateless and thread-safe.
public enum MetalUtilities {
    /// Load Metal shader library with 3-tier fallback strategy
    ///
    /// Attempts to load QuantumGPU shader library in performance order: pre-compiled
    /// `default.metallib` file first (for Xcode builds), then default library from app bundle,
    /// finally runtime compilation from `QuantumGPU.metal` source (for CLI). Returns nil if
    /// all strategies fail; caller must handle Metal unavailability with CPU fallback.
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
    /// **Example:**
    /// ```swift
    /// guard let (commandBuffer, encoder) = MetalUtilities.createCommandEncoder(queue: queue)
    /// else { return cpuFallback() }
    /// encoder.setComputePipelineState(pipeline)
    /// encoder.endEncoding()
    /// commandBuffer.commit()
    /// ```
    ///
    /// - Parameter queue: Metal command queue to create buffer from
    /// - Returns: (command buffer, compute encoder) tuple if creation succeeds, nil otherwise
    @_effects(readonly)
    @inlinable
    public static func createCommandEncoder(
        queue: MTLCommandQueue,
    ) -> (commandBuffer: MTLCommandBuffer, encoder: MTLComputeCommandEncoder)? {
        guard let commandBuffer = queue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder()
        else { return nil }

        return (commandBuffer, encoder)
    }
}
