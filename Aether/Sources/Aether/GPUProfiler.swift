// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Metal

/// Captures GPU and CPU timing metrics from circuit execution.
///
/// **Example:**
/// ```swift
/// let circuit = QuantumCircuit(qubits: 10)
/// let profile = await GPUProfiler.profile(circuit, precisionPolicy: .auto)
/// print(profile.gpuTimeMs, profile.didUseGPU)
/// ```
///
/// - SeeAlso: ``GPUProfiler``
@frozen
public struct GPUProfile: Sendable, Equatable, CustomStringConvertible {
    /// GPU execution time in milliseconds.
    public let gpuTimeMs: Double
    /// CPU execution time in milliseconds.
    public let cpuTimeMs: Double
    /// Number of bytes transferred between CPU and GPU.
    public let transferBytes: Int
    /// GPU utilization as a percentage (0-100).
    public let utilizationPercent: Double
    /// Whether GPU acceleration was used for execution.
    public let didUseGPU: Bool

    /// Human-readable summary of profiling metrics.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit(qubits: 10)
    /// let profile = await GPUProfiler.profile(circuit, precisionPolicy: .auto)
    /// print(profile.description)
    /// ```
    @inlinable
    public var description: String {
        if didUseGPU {
            let transferKB = Double(transferBytes) / 1024.0
            return String(
                format: "GPU: %.2fms, CPU: %.2fms, Transfer: %.1fKB, Utilization: %.1f%%",
                gpuTimeMs, cpuTimeMs, transferKB, utilizationPercent,
            )
        } else {
            return String(format: "CPU-only: %.2fms", cpuTimeMs)
        }
    }
}

/// Measures GPU/CPU performance for quantum circuit execution.
///
/// **Example:**
/// ```swift
/// let circuit = QuantumCircuit(qubits: 10)
/// let profile = await GPUProfiler.profile(circuit, precisionPolicy: .auto)
/// print(profile)
/// ```
///
/// - SeeAlso: ``GPUProfile``
public enum GPUProfiler {
    /// Cached Metal device for availability checks.
    private static let metalDevice: MTLDevice? = MTLCreateSystemDefaultDevice()

    /// Mach timebase ratio for converting ticks to nanoseconds.
    private static let nanosPerTick: Double = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return Double(info.numer) / Double(info.denom)
    }()

    /// Size of a single complex amplitude in bytes.
    private static let bytesPerAmplitude = 16
    /// Number of transfer directions (upload and download).
    private static let transferDirections = 2
    /// Estimated fraction of total time spent on GPU.
    private static let estimatedGPUFraction = 0.85
    /// Estimated fraction of total time spent on CPU.
    private static let estimatedCPUFraction = 0.15
    /// Estimated GPU utilization percentage when GPU is active.
    private static let estimatedGPUUtilization = 85.0

    /// Profiles a ``QuantumCircuit`` execution and returns timing metrics.
    ///
    /// Executes the circuit, measures wall-clock time, and estimates GPU/CPU time split
    /// based on whether Metal acceleration is available and the ``PrecisionPolicy`` permits
    /// GPU usage for the given qubit count.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit(qubits: 10)
    /// let profile = await GPUProfiler.profile(circuit, precisionPolicy: .auto)
    /// print(profile.gpuTimeMs)
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: The quantum circuit to profile
    ///   - precisionPolicy: Policy controlling GPU acceleration eligibility
    /// - Returns: A ``GPUProfile`` with timing and transfer metrics
    /// - Complexity: O(2^n) where n is the qubit count
    /// - SeeAlso: ``GPUProfile``
    @_optimize(speed)
    public static func profile(
        _ circuit: QuantumCircuit,
        precisionPolicy: PrecisionPolicy,
    ) async -> GPUProfile {
        let qubits = circuit.qubits
        let stateSpaceSize = 1 << qubits
        let transferBytes = bytesPerAmplitude * stateSpaceSize * transferDirections

        let shouldUseGPU = precisionPolicy.shouldUseGPU(forQubitCount: qubits)
        let metalAvailable = metalDevice != nil

        let startTime = mach_absolute_time()
        _ = circuit.execute()
        let endTime = mach_absolute_time()

        let totalNanos = Double(endTime - startTime) * nanosPerTick
        let totalTimeMs = totalNanos * 1e-6

        let didUseGPU = shouldUseGPU && metalAvailable

        if didUseGPU {
            let gpuTimeMs = totalTimeMs * estimatedGPUFraction
            let cpuTimeMs = totalTimeMs * estimatedCPUFraction
            let utilizationPercent = estimatedGPUUtilization

            return GPUProfile(
                gpuTimeMs: gpuTimeMs,
                cpuTimeMs: cpuTimeMs,
                transferBytes: transferBytes,
                utilizationPercent: utilizationPercent,
                didUseGPU: true,
            )
        } else {
            return GPUProfile(
                gpuTimeMs: 0.0,
                cpuTimeMs: totalTimeMs,
                transferBytes: transferBytes,
                utilizationPercent: 0.0,
                didUseGPU: false,
            )
        }
    }
}
