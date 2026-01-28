// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Darwin

/// Memory profiling utilities for tracking memory usage during quantum circuit execution.
///
/// Provides tools for measuring actual memory consumption using Darwin task_info and
/// malloc_zone_statistics, as well as estimating memory requirements before execution.
/// Essential for resource planning when working with large quantum circuits.
///
/// **Example:**
/// ```swift
/// var circuit = QuantumCircuit(qubits: 10)
/// circuit.append(.hadamard, to: 0)
/// let profile = MemoryProfiler.profile(circuit)
/// print(profile.peakBytes, profile.amplitudeBytes)
/// ```
public enum MemoryProfiler {
    /// Profiles memory usage during quantum circuit execution.
    ///
    /// Executes the circuit and measures actual memory consumption using Darwin task_info
    /// for resident memory and malloc_zone_statistics for heap details. Returns peak memory
    /// observed during execution along with computed sizes for state vector and unitary matrices.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 8)
    /// circuit.append(.hadamard, to: 0)
    /// let profile = MemoryProfiler.profile(circuit)
    /// print("Peak: \(profile.peakBytes), State: \(profile.amplitudeBytes)")
    /// ```
    ///
    /// - Parameter circuit: Quantum circuit to profile
    /// - Returns: Memory profile containing peak, amplitude, and unitary byte counts
    /// - Complexity: O(n x 2^q) where n = operation count, q = qubit count
    @inlinable
    public static func profile(_ circuit: QuantumCircuit) -> MemoryProfile {
        let qubits = circuit.qubits
        let stateSpaceSize = 1 << qubits
        let amplitudeBytes = stateSpaceSize * 16
        let unitaryBytes = stateSpaceSize * stateSpaceSize * 16

        let memoryBefore = measurePeakMemory()
        let _ = circuit.execute()
        let memoryAfter = measurePeakMemory()
        let heapStats = getHeapStatistics()

        return MemoryProfile(
            peakBytes: max(memoryBefore.resident, memoryAfter.resident),
            virtualBytes: max(memoryBefore.virtual, memoryAfter.virtual),
            amplitudeBytes: amplitudeBytes,
            unitaryBytes: unitaryBytes,
            heapAllocated: heapStats.allocated,
            heapUsed: heapStats.used,
        )
    }

    /// Estimates memory requirements for a quantum system with the specified number of qubits.
    ///
    /// Computes theoretical memory needs without executing any circuit. State vector requires
    /// 16 x 2^n bytes (Complex<Double> = 16 bytes per amplitude). Unitary matrix requires
    /// 16 x 2^(2n) bytes. Provides recommendation based on available system memory.
    ///
    /// **Example:**
    /// ```swift
    /// let estimate = MemoryProfiler.estimate(qubits: 20)
    /// print("State: \(estimate.stateBytes), Recommended: \(estimate.isRecommended)")
    /// ```
    ///
    /// - Parameter qubits: Number of qubits to estimate for
    /// - Returns: Memory estimate with state vector size, unitary size, and recommendation
    /// - Precondition: qubits > 0
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    public static func estimate(qubits: Int) -> MemoryEstimate {
        ValidationUtilities.validatePositiveQubits(qubits)

        let stateSpaceSize = 1 << qubits
        let stateBytes = stateSpaceSize * 16
        let unitaryBytes = stateSpaceSize * stateSpaceSize * 16

        let availableMemory = getAvailableMemory()
        let recommended = stateBytes < availableMemory / 2

        return MemoryEstimate(
            stateBytes: stateBytes,
            unitaryBytes: unitaryBytes,
            isRecommended: recommended,
        )
    }

    @usableFromInline
    @inline(__always)
    static func measurePeakMemory() -> (resident: Int, virtual: Int) {
        var info = task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<task_basic_info>.size) / 4

        let result = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: Int32.self, capacity: Int(count)) { ptr in
                task_info(
                    mach_task_self_,
                    task_flavor_t(TASK_BASIC_INFO),
                    ptr,
                    &count,
                )
            }
        }

        if result != KERN_SUCCESS {
            return (0, 0)
        }

        return (Int(info.resident_size), Int(info.virtual_size))
    }

    @usableFromInline
    @inline(__always)
    static func getHeapStatistics() -> (allocated: Int, used: Int) {
        var stats = malloc_statistics_t()
        malloc_zone_statistics(nil, &stats)
        return (Int(stats.size_allocated), Int(stats.size_in_use))
    }

    @usableFromInline
    @inline(__always)
    static func getAvailableMemory() -> Int {
        let heapStats = getHeapStatistics()

        var info = task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<task_basic_info>.size) / 4

        let result = withUnsafeMutablePointer(to: &info) { infoPtr in
            infoPtr.withMemoryRebound(to: Int32.self, capacity: Int(count)) { ptr in
                task_info(
                    mach_task_self_,
                    task_flavor_t(TASK_BASIC_INFO),
                    ptr,
                    &count,
                )
            }
        }

        if result != KERN_SUCCESS {
            return 8 * 1024 * 1024 * 1024 - heapStats.used
        }

        var hostInfo = host_basic_info()
        var hostCount = mach_msg_type_number_t(MemoryLayout<host_basic_info>.size / MemoryLayout<Int32>.size)
        let hostResult = withUnsafeMutablePointer(to: &hostInfo) { hostPtr in
            hostPtr.withMemoryRebound(to: Int32.self, capacity: Int(hostCount)) { ptr in
                host_info(mach_host_self(), HOST_BASIC_INFO, ptr, &hostCount)
            }
        }
        let physicalMemory = hostResult == KERN_SUCCESS ? UInt64(hostInfo.max_mem) : 8 * 1024 * 1024 * 1024
        return Int(physicalMemory) - Int(info.resident_size) - heapStats.used
    }
}

/// Memory profile from quantum circuit execution.
///
/// Contains measured and computed memory statistics from profiling a quantum circuit.
/// Peak bytes reflects actual resident memory during execution, while amplitude and
/// unitary bytes are theoretical sizes based on qubit count. Includes heap statistics
/// from malloc_zone_statistics for detailed memory breakdown.
///
/// **Example:**
/// ```swift
/// let profile = MemoryProfiler.profile(circuit)
/// print(profile.peakBytes)
/// print(profile.virtualBytes)
/// print(profile.heapAllocated)
/// ```
@frozen
public struct MemoryProfile: Sendable, Equatable, CustomStringConvertible {
    /// Peak resident memory in bytes observed during circuit execution.
    public let peakBytes: Int

    /// Peak virtual memory in bytes observed during circuit execution.
    public let virtualBytes: Int

    /// Theoretical state vector size in bytes (16 x 2^n for n qubits).
    public let amplitudeBytes: Int

    /// Theoretical unitary matrix size in bytes (16 x 2^(2n) for n qubits).
    public let unitaryBytes: Int

    /// Total heap memory allocated in bytes from malloc_zone_statistics.
    public let heapAllocated: Int

    /// Heap memory currently in use in bytes from malloc_zone_statistics.
    public let heapUsed: Int

    /// Human-readable description of the memory profile.
    ///
    /// - Returns: Formatted string with all memory statistics in human-readable units
    /// - Complexity: O(1)
    @inlinable
    public var description: String {
        "MemoryProfile(peak: \(formatBytes(peakBytes)), virtual: \(formatBytes(virtualBytes)), amplitudes: \(formatBytes(amplitudeBytes)), unitary: \(formatBytes(unitaryBytes)), heapAllocated: \(formatBytes(heapAllocated)), heapUsed: \(formatBytes(heapUsed)))"
    }

    @usableFromInline
    @inline(__always)
    init(peakBytes: Int, virtualBytes: Int, amplitudeBytes: Int, unitaryBytes: Int, heapAllocated: Int, heapUsed: Int) {
        self.peakBytes = peakBytes
        self.virtualBytes = virtualBytes
        self.amplitudeBytes = amplitudeBytes
        self.unitaryBytes = unitaryBytes
        self.heapAllocated = heapAllocated
        self.heapUsed = heapUsed
    }

    /// Formats a byte count into a human-readable string with appropriate units.
    ///
    /// - Parameter bytes: Number of bytes to format
    /// - Returns: Formatted string with GB, MB, KB, or B suffix
    /// - Complexity: O(1)
    @usableFromInline
    @inline(__always)
    func formatBytes(_ bytes: Int) -> String {
        if bytes >= 1_073_741_824 {
            String(format: "%.2f GB", Double(bytes) / 1_073_741_824.0)
        } else if bytes >= 1_048_576 {
            String(format: "%.2f MB", Double(bytes) / 1_048_576.0)
        } else if bytes >= 1024 {
            String(format: "%.2f KB", Double(bytes) / 1024.0)
        } else {
            "\(bytes) B"
        }
    }
}

/// Memory estimate for a quantum system.
///
/// Provides theoretical memory requirements and recommendation for executing
/// quantum circuits with a specified number of qubits.
///
/// **Example:**
/// ```swift
/// let estimate = MemoryProfiler.estimate(qubits: 15)
/// if estimate.isRecommended {
///     print("Safe to proceed with \(estimate.stateBytes) bytes for state vector")
/// }
/// ```
@frozen
public struct MemoryEstimate: Sendable, Equatable {
    /// Theoretical state vector size in bytes (16 x 2^n for n qubits).
    public let stateBytes: Int

    /// Theoretical unitary matrix size in bytes (16 x 2^(2n) for n qubits).
    public let unitaryBytes: Int

    /// Whether executing circuits with this qubit count is recommended.
    ///
    /// Returns true if state vector fits comfortably in available memory
    /// (less than 50% of available RAM).
    public let isRecommended: Bool

    @usableFromInline
    @inline(__always)
    init(stateBytes: Int, unitaryBytes: Int, isRecommended: Bool) {
        self.stateBytes = stateBytes
        self.unitaryBytes = unitaryBytes
        self.isRecommended = isRecommended
    }
}
