// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Darwin

/// Per-gate timing benchmarks with warmup runs and statistical aggregation.
///
/// Measures execution time of individual quantum gates with nanosecond precision.
/// Performs warmup iterations before measurement to eliminate cold-start effects,
/// then collects timing samples with numerically stable mean and variance computation.
///
/// **Example:**
/// ```swift
/// let benchmark = GateBenchmark(qubits: 8, iterations: 100, warmupIterations: 5)
/// let result = await benchmark.measure(.hadamard)
/// let results = await benchmark.measure([.hadamard, .cnot, .toffoli])
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
    /// across the specified number of iterations with numerically stable statistical
    /// aggregation.
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
    /// - Precondition: qubits >= gate.qubitsRequired
    /// - Complexity: O(iterations * 2^qubits)
    @_optimize(speed)
    public func measure(_ gate: QuantumGate) async -> GateBenchmarkResult {
        let targetQubits = generateTargetQubits(for: gate)
        var state = QuantumState(qubits: qubits)

        for _ in 0 ..< warmupIterations {
            state = GateApplication.apply(gate, to: targetQubits, state: state)
        }

        var timebaseInfo = mach_timebase_info_data_t()
        mach_timebase_info(&timebaseInfo)
        let timebaseRatio = Double(timebaseInfo.numer) / Double(timebaseInfo.denom)

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
            let elapsedNs = elapsedMach * timebaseRatio

            welfordCount += 1
            let delta = elapsedNs - welfordMean
            welfordMean += delta / Double(welfordCount)
            let delta2 = elapsedNs - welfordMean
            welfordM2 = welfordM2.addingProduct(delta, delta2)

            minNs = min(minNs, elapsedNs)
            maxNs = max(maxNs, elapsedNs)
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

    /// Measures execution times of multiple quantum gates.
    ///
    /// Measures each gate independently and returns results in the same order as input.
    /// Useful for comparing performance characteristics of different gate types.
    ///
    /// **Example:**
    /// ```swift
    /// let benchmark = GateBenchmark(qubits: 8)
    /// let results = await benchmark.measure([.hadamard, .pauliX, .rotationZ(.pi/4)])
    /// for result in results {
    ///     print(result)
    /// }
    /// ```
    ///
    /// - Parameter gates: Array of quantum gates to benchmark
    /// - Returns: Array of benchmark results in same order as input gates
    /// - Precondition: qubits >= gate.qubitsRequired for each gate
    /// - Complexity: O(gates.count * iterations * 2^qubits)
    public func measure(_ gates: [QuantumGate]) async -> [GateBenchmarkResult] {
        guard !gates.isEmpty else { return [] }
        var results = [GateBenchmarkResult]()
        results.reserveCapacity(gates.count)

        for gate in gates {
            let result = await measure(gate)
            results.append(result)
        }

        return results
    }

    /// Generates target qubit indices based on the gate's qubit requirements.
    @_effects(readonly)
    @inline(__always)
    private func generateTargetQubits(for gate: QuantumGate) -> [Int] {
        let required = gate.qubitsRequired
        ValidationUtilities.validateMinimumQubits(qubits, min: required, algorithmName: "Gate benchmark")
        return Array(0 ..< required)
    }
}

/// Result of a gate benchmark containing timing statistics.
///
/// Contains nanosecond-precision timing measurements for quantum gate execution including
/// mean, minimum, maximum, and standard deviation.
///
/// **Example:**
/// ```swift
/// let benchmark = GateBenchmark(qubits: 8)
/// let result = await benchmark.measure(.hadamard)
/// print(result.meanNs)
/// print(result)
/// ```
///
/// - SeeAlso: ``GateBenchmark``
@frozen
public struct GateBenchmarkResult: Sendable, Equatable, CustomStringConvertible {
    /// The gate that was benchmarked.
    public let gate: QuantumGate

    /// Mean execution time in nanoseconds.
    public let meanNs: Double

    /// Minimum execution time in nanoseconds.
    public let minNs: Double

    /// Maximum execution time in nanoseconds.
    public let maxNs: Double

    /// Standard deviation of execution time in nanoseconds.
    public let stdDevNs: Double

    /// Number of iterations used for measurement (excludes warmup).
    public let iterations: Int

    /// Create benchmark result.
    public init(gate: QuantumGate, meanNs: Double, minNs: Double, maxNs: Double, stdDevNs: Double, iterations: Int) {
        self.gate = gate
        self.meanNs = meanNs
        self.minNs = minNs
        self.maxNs = maxNs
        self.stdDevNs = stdDevNs
        self.iterations = iterations
    }

    /// Human-readable description of benchmark results.
    ///
    /// Formats timing statistics with appropriate precision for display.
    ///
    /// **Example:**
    /// ```swift
    /// let benchmark = GateBenchmark(qubits: 8)
    /// let result = await benchmark.measure(.hadamard)
    /// print(result.description)
    /// ```
    public var description: String {
        let gateName = gate.fullName
        let meanStr = formatDouble(meanNs)
        let minStr = formatDouble(minNs)
        let maxStr = formatDouble(maxNs)
        let stdDevStr = formatDouble(stdDevNs)
        return "\(gateName): mean=\(meanStr)ns, min=\(minStr)ns, max=\(maxStr)ns, stddev=\(stdDevStr)ns (\(iterations) iterations)"
    }
}

private extension GateBenchmarkResult {
    /// Formats a double to two decimal places without Foundation dependency.
    @_effects(readonly)
    @inline(__always)
    func formatDouble(_ value: Double) -> String {
        let rounded = (value * 100).rounded() / 100
        let intPart = Int(rounded)
        let fracPart = Int(((rounded - Double(intPart)) * 100).rounded())
        let absFrac = abs(fracPart)
        let fracStr = absFrac < 10 ? "0\(absFrac)" : "\(absFrac)"
        return "\(intPart).\(fracStr)"
    }
}
