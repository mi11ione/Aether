// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Darwin

/// Per-gate timing benchmarks with warmup runs and statistical aggregation.
///
/// Measures execution time of individual quantum gates with nanosecond precision using
/// `mach_absolute_time()`. Performs warmup iterations before measurement to eliminate
/// cold-start effects, then collects timing samples using online Welford algorithm for
/// numerically stable mean and variance computation without storing all samples.
///
/// **Example:**
/// ```swift
/// let benchmark = GateBenchmark(qubits: 8, iterations: 100, warmupIterations: 5)
/// let result = await benchmark.measure(.hadamard)
/// print(result.meanNs)  // Average execution time in nanoseconds
///
/// let comparison = await benchmark.compare([.hadamard, .cnot, .toffoli])
/// for r in comparison {
///     print(r)  // Formatted gate timing results
/// }
/// ```
///
/// - SeeAlso: ``GateBenchmarkResult``
/// - SeeAlso: ``GateApplication``
@frozen
public struct GateBenchmark: Sendable {
    /// Number of qubits in the benchmark state
    public let qubits: Int

    /// Number of timed iterations to perform
    public let iterations: Int

    /// Number of warmup iterations (discarded) before measurement
    public let warmupIterations: Int

    /// Creates a gate benchmark configuration.
    ///
    /// Initializes benchmark parameters for measuring quantum gate execution times.
    /// Warmup iterations prime caches and JIT compilation before timed measurements begin.
    ///
    /// **Example:**
    /// ```swift
    /// let benchmark = GateBenchmark(qubits: 10)
    /// let benchmark2 = GateBenchmark(qubits: 8, iterations: 200, warmupIterations: 10)
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits in benchmark state (must be positive)
    ///   - iterations: Number of timed iterations (default: 100)
    ///   - warmupIterations: Number of warmup iterations (default: 3)
    /// - Precondition: qubits > 0
    /// - Precondition: iterations > 0
    /// - Precondition: warmupIterations >= 0
    public init(qubits: Int, iterations: Int = 100, warmupIterations: Int = 3) {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validatePositiveInt(iterations, name: "Iterations")
        ValidationUtilities.validateNonNegativeInt(warmupIterations, name: "Warmup iterations")

        self.qubits = qubits
        self.iterations = iterations
        self.warmupIterations = warmupIterations
    }

    /// Measures execution time of a single quantum gate.
    ///
    /// Performs warmup iterations to prime caches, then measures gate application time
    /// across the specified number of iterations. Uses Welford's online algorithm for
    /// numerically stable computation of mean and variance without storing all samples.
    ///
    /// **Example:**
    /// ```swift
    /// let benchmark = GateBenchmark(qubits: 8)
    /// let result = await benchmark.measure(.hadamard)
    /// print("Mean: \(result.meanNs)ns, StdDev: \(result.stdDevNs)ns")
    /// ```
    ///
    /// - Parameter gate: The quantum gate to benchmark
    /// - Returns: Benchmark result with timing statistics
    /// - Complexity: O(iterations * 2^qubits)
    public func measure(_ gate: QuantumGate) async -> GateBenchmarkResult {
        let targetQubits = generateTargetQubits(for: gate)
        var state = QuantumState(qubits: qubits)

        for _ in 0 ..< warmupIterations {
            state = GateApplication.apply(gate, to: targetQubits, state: state)
        }

        var timebaseInfo = mach_timebase_info_data_t()
        mach_timebase_info(&timebaseInfo)
        let timebaseNumer = Double(timebaseInfo.numer)
        let timebaseDenom = Double(timebaseInfo.denom)

        var welfordCount = 0
        var welfordMean = 0.0
        var welfordM2 = 0.0
        var minNs = Double.infinity
        var maxNs = -Double.infinity

        for _ in 0 ..< iterations {
            let startTime = mach_absolute_time()
            state = GateApplication.apply(gate, to: targetQubits, state: state)
            let endTime = mach_absolute_time()

            let elapsedMach = Double(endTime - startTime)
            let elapsedNs = elapsedMach * timebaseNumer / timebaseDenom

            welfordCount += 1
            let delta = elapsedNs - welfordMean
            welfordMean += delta / Double(welfordCount)
            let delta2 = elapsedNs - welfordMean
            welfordM2 += delta * delta2

            if elapsedNs < minNs { minNs = elapsedNs }
            if elapsedNs > maxNs { maxNs = elapsedNs }
        }

        let variance = welfordCount > 1 ? welfordM2 / Double(welfordCount - 1) : 0.0
        let stdDev = variance.squareRoot()

        return GateBenchmarkResult(
            gate: gate,
            meanNs: welfordMean,
            minNs: minNs,
            maxNs: maxNs,
            stdDevNs: stdDev,
            iterations: iterations,
        )
    }

    /// Compares execution times of multiple quantum gates.
    ///
    /// Measures each gate independently and returns results in the same order as input.
    /// Useful for comparing performance characteristics of different gate types.
    ///
    /// **Example:**
    /// ```swift
    /// let benchmark = GateBenchmark(qubits: 8)
    /// let results = await benchmark.compare([.hadamard, .pauliX, .rotationZ(.pi/4)])
    /// for result in results {
    ///     print(result)
    /// }
    /// ```
    ///
    /// - Parameter gates: Array of quantum gates to benchmark
    /// - Returns: Array of benchmark results in same order as input gates
    /// - Complexity: O(gates.count * iterations * 2^qubits)
    public func compare(_ gates: [QuantumGate]) async -> [GateBenchmarkResult] {
        var results = [GateBenchmarkResult]()
        results.reserveCapacity(gates.count)

        for gate in gates {
            let result = await measure(gate)
            results.append(result)
        }

        return results
    }

    /// Generates target qubit indices based on the gate's qubit requirements.
    @inline(__always)
    private func generateTargetQubits(for gate: QuantumGate) -> [Int] {
        let required = gate.qubitsRequired
        precondition(qubits >= required, "Benchmark requires at least \(required) qubits (got \(qubits))")
        return Array(0 ..< required)
    }
}

/// Result of a gate benchmark containing timing statistics.
///
/// Contains nanosecond-precision timing measurements for quantum gate execution including
/// mean, minimum, maximum, and standard deviation computed using Welford's online algorithm.
///
/// **Example:**
/// ```swift
/// let result = await benchmark.measure(.hadamard)
/// print(result.meanNs)      // Average time in nanoseconds
/// print(result.stdDevNs)    // Standard deviation
/// print(result)             // Human-readable summary
/// ```
///
/// - SeeAlso: ``GateBenchmark``
@frozen
public struct GateBenchmarkResult: Sendable, Equatable, CustomStringConvertible {
    /// The gate that was benchmarked.
    /// - Returns: The ``QuantumGate`` instance that was measured.
    public let gate: QuantumGate

    /// Mean execution time in nanoseconds.
    /// - Returns: Average gate execution time across all iterations.
    public let meanNs: Double

    /// Minimum execution time in nanoseconds.
    /// - Returns: Fastest recorded gate execution time.
    public let minNs: Double

    /// Maximum execution time in nanoseconds.
    /// - Returns: Slowest recorded gate execution time.
    public let maxNs: Double

    /// Standard deviation of execution time in nanoseconds.
    /// - Returns: Statistical spread of timing measurements.
    public let stdDevNs: Double

    /// Number of iterations used for measurement.
    /// - Returns: Count of timed gate applications (excludes warmup).
    public let iterations: Int

    /// Human-readable description of benchmark results.
    ///
    /// Formats timing statistics with appropriate precision for display.
    ///
    /// **Example:**
    /// ```swift
    /// let result = await benchmark.measure(.hadamard)
    /// print(result.description)
    /// // "hadamard: mean=1234.56ns, min=1100.00ns, max=1500.00ns, stddev=89.12ns (100 iterations)"
    /// ```
    public var description: String {
        let gateName = gate.fullName
        let meanStr = formatDouble(meanNs)
        let minStr = formatDouble(minNs)
        let maxStr = formatDouble(maxNs)
        let stdDevStr = formatDouble(stdDevNs)
        return "\(gateName): mean=\(meanStr)ns, min=\(minStr)ns, max=\(maxStr)ns, stddev=\(stdDevStr)ns (\(iterations) iterations)"
    }

    @inline(__always)
    private func formatDouble(_ value: Double) -> String {
        let rounded = (value * 100).rounded() / 100
        let intPart = Int(rounded)
        let fracPart = Int(((rounded - Double(intPart)) * 100).rounded())
        let absFrac = abs(fracPart)
        let fracStr = absFrac < 10 ? "0\(absFrac)" : "\(absFrac)"
        return "\(intPart).\(fracStr)"
    }
}
