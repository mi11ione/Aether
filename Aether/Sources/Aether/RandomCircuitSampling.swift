// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Darwin
import GameplayKit

/// Random circuit sampling for cross-entropy benchmarking of quantum computational advantage.
///
/// Generates random quantum circuits from a hardware-native gate set (sqrt-iSWAP two-qubit gates
/// with random SU(2) single-qubit rotations) and evaluates sampling quality via linear cross-entropy
/// benchmarking (XEB). The XEB fidelity F_XEB = 2^n mean(p(x)) - 1 quantifies how faithfully a
/// sampler reproduces the ideal output distribution, where p(x) = |bra(x)|U|0>|^2 is the ideal
/// probability of bitstring x and the average is over sampled bitstrings.
///
/// For ideal (noiseless) sampling F_XEB = 1, while uniform random sampling yields F_XEB = 0.
/// Porter-Thomas distribution validation confirms the ideal output follows the expected exponential
/// distribution P(q) = exp(-q) for rescaled probabilities q = 2^n p(x), characteristic of Haar-random
/// unitaries in the large-circuit limit.
///
/// **Example:**
/// ```swift
/// let result = RandomCircuitSampling.evaluate(qubits: 4, depth: 8, shots: 5000, seed: 42)
/// let fidelity = result.fidelity
/// let ptResult = RandomCircuitSampling.validatePorterThomas(probabilities: result.idealProbabilities)
/// ```
///
/// - SeeAlso: ``IQPSampling``
/// - SeeAlso: ``Measurement``
/// - SeeAlso: ``QuantumCircuit``
public enum RandomCircuitSampling {
    private static let maxDepth = 200
    private static let maxBenchmarkQubits = 20
    private static let antiConcentrationFactor = 3.0

    /// Cross-entropy benchmarking result containing XEB fidelity and supporting data.
    ///
    /// Encapsulates the linear XEB fidelity F_XEB = 2^n mean(p(x_i)) - 1 computed from
    /// sampled bitstrings {x_i} against the ideal output distribution. Includes the ideal
    /// probability vector for downstream analysis such as Porter-Thomas validation.
    ///
    /// **Example:**
    /// ```swift
    /// let result = RandomCircuitSampling.evaluate(qubits: 4, depth: 8, shots: 5000, seed: 42)
    /// let fidelity = result.fidelity
    /// let mean = result.meanIdealProbability
    /// ```
    ///
    /// - SeeAlso: ``RandomCircuitSampling/evaluate(qubits:depth:shots:seed:)``
    @frozen
    public struct XEBResult: Sendable {
        /// Linear XEB fidelity F_XEB = 2^n mean(p(x_i)) - 1 where p(x) is ideal probability
        public let fidelity: Double

        /// Arithmetic mean of ideal probabilities at sampled bitstrings: (1/S) sum(p(x_i))
        public let meanIdealProbability: Double

        /// Number of measurement shots used in the XEB computation
        public let sampleCount: Int

        /// Full ideal probability distribution p(x) = |bra(x)|U|0>|^2 for all 2^n bitstrings
        public let idealProbabilities: [Double]
    }

    /// Porter-Thomas distribution validation result via Kolmogorov-Smirnov test.
    ///
    /// Tests whether the ideal output probabilities follow the Porter-Thomas distribution
    /// P(q) = exp(-q) for rescaled probabilities q = 2^n p(x). This exponential distribution
    /// is the hallmark of random unitary circuits and validates that the generated circuit
    /// produces sufficiently random output.
    ///
    /// **Example:**
    /// ```swift
    /// let result = RandomCircuitSampling.evaluate(qubits: 4, depth: 8, shots: 5000, seed: 42)
    /// let ptResult = RandomCircuitSampling.validatePorterThomas(probabilities: result.idealProbabilities)
    /// let valid = ptResult.isValid
    /// ```
    ///
    /// - SeeAlso: ``RandomCircuitSampling/validatePorterThomas(probabilities:)``
    @frozen
    public struct PorterThomasResult: Sendable {
        /// Kolmogorov-Smirnov test statistic: max|F_empirical(q) - (1 - exp(-q))|
        public let ksStatistic: Double

        /// KS critical value at 95% confidence level: 1.36 / sqrt(N)
        public let criticalValue: Double

        /// Whether the distribution passes the KS test (ksStatistic < criticalValue)
        public let isValid: Bool
    }

    /// Wall-clock benchmark result for a single (qubits, depth) configuration.
    ///
    /// Records the elapsed time for full random circuit sampling pipeline including circuit
    /// generation, statevector simulation, shot sampling, and XEB fidelity computation.
    ///
    /// **Example:**
    /// ```swift
    /// let results = RandomCircuitSampling.benchmark(qubitRange: 3...6, depths: [5, 10], shots: 1000, seed: 42)
    /// let seconds = results[0].wallClockSeconds
    /// let fidelity = results[0].fidelity
    /// ```
    ///
    /// - SeeAlso: ``RandomCircuitSampling/benchmark(qubitRange:depths:shots:seed:)``
    @frozen
    public struct BenchmarkResult: Sendable {
        /// Number of qubits in the benchmark circuit
        public let qubits: Int

        /// Circuit depth (number of gate layers)
        public let depth: Int

        /// Total wall-clock time in seconds for the full sampling pipeline
        public let wallClockSeconds: Double

        /// XEB fidelity achieved for this configuration
        public let fidelity: Double
    }

    // MARK: - Circuit Generation

    /// Generates a random quantum circuit from the hardware-native gate set.
    ///
    /// Constructs a circuit with alternating layers of random single-qubit SU(2) rotations
    /// (Rz(alpha) Ry(beta) Rz(gamma) with Haar-random angles) and sqrt-iSWAP two-qubit entangling
    /// gates. Two-qubit gates alternate between even-odd and odd-even qubit pairings across layers,
    /// producing circuits representative of near-term quantum hardware topologies.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = RandomCircuitSampling.generateCircuit(qubits: 5, depth: 10, seed: 42)
    /// let state = circuit.execute()
    /// let probs = state.probabilities()
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (minimum 2 for entangling gates)
    ///   - depth: Number of circuit layers (1 to 200)
    ///   - seed: Optional RNG seed for reproducible circuit generation
    /// - Returns: Random quantum circuit ready for execution
    /// - Precondition: `qubits` >= 2
    /// - Precondition: `depth` in 1...200
    /// - Complexity: O(depth * qubits) gate count
    ///
    /// - SeeAlso: ``evaluate(qubits:depth:shots:seed:)``
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func generateCircuit(qubits: Int, depth: Int, seed: UInt64? = nil) -> QuantumCircuit {
        ValidationUtilities.validateMinimumQubits(qubits, min: 2, algorithmName: "Random circuit sampling")
        ValidationUtilities.validateMemoryLimit(qubits)
        ValidationUtilities.validatePositiveInt(depth, name: "depth")
        ValidationUtilities.validateUpperBound(depth, max: maxDepth, name: "depth")

        var rng: any RandomNumberGenerator = Measurement.createRNG(seed: seed)
        var circuit = QuantumCircuit(qubits: qubits)

        for layer in 0 ..< depth {
            appendSingleQubitLayer(to: &circuit, qubits: qubits, rng: &rng)
            appendEntanglingLayer(to: &circuit, qubits: qubits, layerParity: layer & 1)
        }

        appendSingleQubitLayer(to: &circuit, qubits: qubits, rng: &rng)

        return circuit
    }

    // MARK: - XEB Fidelity

    /// Computes linear cross-entropy benchmarking fidelity from samples and ideal probabilities.
    ///
    /// Evaluates F_XEB = 2^n mean(p(x_i)) - 1 where p(x_i) is the ideal probability of
    /// each sampled bitstring x_i. For perfect sampling F_XEB = 1; for uniform random
    /// sampling F_XEB = 0. Negative values indicate anti-correlated sampling.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = RandomCircuitSampling.generateCircuit(qubits: 4, depth: 8, seed: 42)
    /// let probs = circuit.execute().probabilities()
    /// let outcomes = Measurement.sample(circuit: circuit, shots: 5000, seed: 42)
    /// let result = RandomCircuitSampling.xebFidelity(samples: outcomes, idealProbabilities: probs)
    /// ```
    ///
    /// - Parameters:
    ///   - samples: Array of measurement outcomes (basis state indices)
    ///   - idealProbabilities: Full ideal probability distribution of length 2^n
    /// - Returns: XEB result with fidelity and supporting statistics
    /// - Precondition: `samples` is non-empty
    /// - Precondition: `idealProbabilities` is non-empty
    /// - Complexity: O(samples.count)
    ///
    /// - SeeAlso: ``evaluate(qubits:depth:shots:seed:)``
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public static func xebFidelity(samples: [Int], idealProbabilities: [Double]) -> XEBResult {
        ValidationUtilities.validateNonEmpty(samples, name: "samples")
        ValidationUtilities.validateNonEmpty(idealProbabilities, name: "idealProbabilities")

        let stateSpaceSize = idealProbabilities.count
        let sampleCount = samples.count

        var sumProbabilities = 0.0
        for outcome in samples {
            if outcome >= 0, outcome < stateSpaceSize {
                sumProbabilities += idealProbabilities[outcome]
            }
        }

        let meanProb = sumProbabilities / Double(sampleCount)
        let fidelity = Double(stateSpaceSize) * meanProb - 1.0

        return XEBResult(
            fidelity: fidelity,
            meanIdealProbability: meanProb,
            sampleCount: sampleCount,
            idealProbabilities: idealProbabilities,
        )
    }

    // MARK: - Convenience Evaluation

    /// Generates a random circuit, simulates it, samples, and computes XEB fidelity in one step.
    ///
    /// Convenience method combining circuit generation, brute-force statevector simulation for
    /// ideal probabilities, Born-rule sampling, and XEB fidelity computation. The returned result
    /// includes the ideal probability vector for downstream Porter-Thomas validation.
    ///
    /// **Example:**
    /// ```swift
    /// let result = RandomCircuitSampling.evaluate(qubits: 4, depth: 8, shots: 5000, seed: 42)
    /// let fidelity = result.fidelity
    /// let probs = result.idealProbabilities
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (minimum 2)
    ///   - depth: Number of circuit layers (1 to 200)
    ///   - shots: Number of measurement samples
    ///   - seed: Optional RNG seed for reproducible results
    /// - Returns: XEB result with fidelity and ideal probability distribution
    /// - Precondition: `qubits` >= 2
    /// - Precondition: `depth` in 1...200
    /// - Precondition: `shots` > 0
    /// - Complexity: O(depth * qubits * 2^qubits + shots)
    ///
    /// - SeeAlso: ``generateCircuit(qubits:depth:seed:)``
    /// - SeeAlso: ``xebFidelity(samples:idealProbabilities:)``
    @_optimize(speed)
    @_eagerMove
    public static func evaluate(qubits: Int, depth: Int, shots: Int, seed: UInt64? = nil) -> XEBResult {
        ValidationUtilities.validatePositiveInt(shots, name: "shots")

        let circuit = generateCircuit(qubits: qubits, depth: depth, seed: seed)
        let idealProbabilities = circuit.execute().probabilities()
        let samples = Measurement.sample(circuit: circuit, shots: shots, seed: seed)

        return xebFidelity(samples: samples, idealProbabilities: idealProbabilities)
    }

    // MARK: - Porter-Thomas Validation

    /// Validates that ideal output probabilities follow the Porter-Thomas distribution.
    ///
    /// Tests the rescaled probabilities q_i = N p(x_i) where N = 2^n against the exponential
    /// CDF F(q) = 1 - exp(-q) using the Kolmogorov-Smirnov test at 95% confidence. The
    /// Porter-Thomas distribution P(q) = exp(-q) is the universal output distribution of
    /// Haar-random unitaries and indicates the circuit generates sufficiently complex output.
    ///
    /// **Example:**
    /// ```swift
    /// let result = RandomCircuitSampling.evaluate(qubits: 4, depth: 8, shots: 5000, seed: 42)
    /// let ptResult = RandomCircuitSampling.validatePorterThomas(probabilities: result.idealProbabilities)
    /// let valid = ptResult.isValid
    /// ```
    ///
    /// - Parameter probabilities: Ideal probability distribution of length 2^n
    /// - Returns: Porter-Thomas validation result with KS statistic and pass/fail
    /// - Precondition: `probabilities` is non-empty
    /// - Complexity: O(N log N) where N = 2^n for sorting
    ///
    /// - SeeAlso: ``evaluate(qubits:depth:shots:seed:)``
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public static func validatePorterThomas(probabilities: [Double]) -> PorterThomasResult {
        ValidationUtilities.validateNonEmpty(probabilities, name: "probabilities")

        let n = probabilities.count
        let scale = Double(n)

        var rescaled = [Double](unsafeUninitializedCapacity: n) {
            buffer, count in
            for i in 0 ..< n {
                buffer[i] = scale * probabilities[i]
            }
            count = n
        }
        rescaled.sort()

        var maxDeviation = 0.0
        let invN = 1.0 / scale

        for i in 0 ..< n {
            let empiricalCDF = Double(i + 1) * invN
            let theoreticalCDF = 1.0 - Darwin.exp(-rescaled[i])

            let deviation = abs(empiricalCDF - theoreticalCDF)
            if deviation > maxDeviation {
                maxDeviation = deviation
            }

            let deviationBelow = abs(Double(i) * invN - theoreticalCDF)
            if deviationBelow > maxDeviation {
                maxDeviation = deviationBelow
            }
        }

        let criticalValue = 1.36 / scale.squareRoot()

        return PorterThomasResult(
            ksStatistic: maxDeviation,
            criticalValue: criticalValue,
            isValid: maxDeviation < criticalValue,
        )
    }

    // MARK: - Benchmark

    /// Benchmarks wall-clock time across a grid of qubit counts and circuit depths.
    ///
    /// Runs the full random circuit sampling pipeline (generation, simulation, sampling, XEB)
    /// for each combination in the qubit range x depths grid and records wall-clock timing
    /// via mach_absolute_time. Results are ordered by (qubits, depth) with qubits varying
    /// first within each depth.
    ///
    /// **Example:**
    /// ```swift
    /// let results = RandomCircuitSampling.benchmark(qubitRange: 3...6, depths: [5, 10], shots: 1000, seed: 42)
    /// for r in results {
    ///     let info = "\(r.qubits)q x \(r.depth)d: \(r.wallClockSeconds)s, F=\(r.fidelity)"
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - qubitRange: Range of qubit counts to benchmark (each >= 2)
    ///   - depths: Array of circuit depths to benchmark (each >= 1)
    ///   - shots: Number of measurement samples per configuration
    ///   - seed: Optional RNG seed for reproducible benchmarks
    /// - Returns: Array of benchmark results for each (qubits, depth) combination
    /// - Precondition: `qubitRange.lowerBound` >= 2
    /// - Precondition: All depths in 1...200
    /// - Precondition: `shots` > 0
    /// - Complexity: O(|qubitRange| * |depths| * (depth * qubits * 2^qubits + shots))
    ///
    /// - SeeAlso: ``evaluate(qubits:depth:shots:seed:)``
    @_optimize(speed)
    @_eagerMove
    public static func benchmark(
        qubitRange: ClosedRange<Int>,
        depths: [Int],
        shots: Int,
        seed: UInt64? = nil,
    ) -> [BenchmarkResult] {
        ValidationUtilities.validateMinimumQubits(
            qubitRange.lowerBound, min: 2, algorithmName: "Random circuit sampling benchmark",
        )
        ValidationUtilities.validateUpperBound(
            qubitRange.upperBound, max: maxBenchmarkQubits, name: "benchmark qubits",
        )
        ValidationUtilities.validateNonEmpty(depths, name: "depths")
        ValidationUtilities.validatePositiveInt(shots, name: "shots")

        var timebaseInfo = mach_timebase_info_data_t()
        mach_timebase_info(&timebaseInfo)
        let timebaseRatio = Double(timebaseInfo.numer) / Double(timebaseInfo.denom)
        let nsToSeconds = 1e-9

        let totalCount = qubitRange.count * depths.count
        var results = [BenchmarkResult]()
        results.reserveCapacity(totalCount)

        for depthValue in depths {
            for qubitCount in qubitRange {
                let startTime = mach_absolute_time()

                let xebResult = evaluate(
                    qubits: qubitCount,
                    depth: depthValue,
                    shots: shots,
                    seed: seed,
                )

                let endTime = mach_absolute_time()
                let elapsedNs = Double(endTime - startTime) * timebaseRatio
                let elapsedSeconds = elapsedNs * nsToSeconds

                results.append(BenchmarkResult(
                    qubits: qubitCount,
                    depth: depthValue,
                    wallClockSeconds: elapsedSeconds,
                    fidelity: xebResult.fidelity,
                ))
            }
        }

        return results
    }

    // MARK: - Private Helpers

    /// Appends random SU(2) single-qubit gates to all qubits in the circuit.
    @_optimize(speed)
    private static func appendSingleQubitLayer(
        to circuit: inout QuantumCircuit,
        qubits: Int,
        rng: inout any RandomNumberGenerator,
    ) {
        for qubit in 0 ..< qubits {
            let alpha = Double.random(in: 0 ..< 2.0 * .pi, using: &rng)
            let beta = Double.random(in: 0 ..< .pi, using: &rng)
            let gamma = Double.random(in: 0 ..< 2.0 * .pi, using: &rng)

            circuit.append(.rotationZ(alpha), to: qubit)
            circuit.append(.rotationY(beta), to: qubit)
            circuit.append(.rotationZ(gamma), to: qubit)
        }
    }

    /// Appends sqrt-iSWAP entangling gates on alternating qubit pairs.
    @_optimize(speed)
    @inline(__always)
    private static func appendEntanglingLayer(
        to circuit: inout QuantumCircuit,
        qubits: Int,
        layerParity: Int,
    ) {
        var startQubit = layerParity
        while startQubit + 1 < qubits {
            circuit.append(.sqrtISwap, to: [startQubit, startQubit + 1])
            startQubit += 2
        }
    }
}
