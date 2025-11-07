// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation
import GameplayKit

/// Quantum measurement outcome with collapsed state
///
/// Encapsulates the result of measuring a quantum state: the classical outcome
/// and the post-measurement collapsed state. Implements state collapse postulate
/// from quantum mechanics.
///
/// Example:
/// ```swift
/// var measurement = Measurement()
/// let result = measurement.measure(state: bellState)
/// // result.outcome: 0 or 3 (50% each for Bell state)
/// // result.collapsedState: |00⟩ or |11⟩ (deterministic after collapse)
/// ```
struct MeasurementResult: Equatable, CustomStringConvertible {
    /// Classical outcome (basis state index i ∈ [0, 2^n-1])
    let outcome: Int

    /// Post-measurement state |i⟩ (collapsed to measured outcome)
    let collapsedState: QuantumState

    var description: String {
        "Measurement: outcome=\(outcome), state=\(collapsedState)"
    }
}

/// Quantum measurement: Born rule implementation with state collapse
///
/// Implements projective measurement in the computational basis following quantum mechanics:
/// - **Born rule**: Probability P(i) = |cᵢ|² for measuring basis state |i⟩
/// - **State collapse**: Measurement projects state onto measured outcome
/// - **Partial measurement**: Measure subset of qubits, leaving others in superposition
/// - **Reproducibility**: Optional seeded RNG for deterministic results
///
/// **Measurement types**:
/// - Full measurement: All qubits → single basis state outcome
/// - Partial measurement: One qubit → probabilistic collapse, others remain quantum
///
/// **Statistical analysis**:
/// - Multiple runs for frequency distribution
/// - Chi-squared goodness-of-fit testing
/// - Histogram generation and comparison
///
/// **RNG modes**:
/// - System RNG (default): True randomness, non-reproducible
/// - Seeded RNG: Reproducible measurements for testing and debugging
///
/// Example:
/// ```swift
/// // Create Bell state (|00⟩ + |11⟩)/√2
/// let circuit = QuantumCircuit.bellPhiPlus()
/// let state = circuit.execute()
///
/// // Single measurement
/// var measurement = Measurement()
/// let result = measurement.measure(state: state)
/// // result.outcome: 0 (|00⟩) or 3 (|11⟩) with 50% probability each
///
/// // Multiple measurements for statistics
/// let outcomes = Measurement.runMultiple(circuit: circuit, numRuns: 1000)
/// let counts = Measurement.histogram(outcomes: outcomes, numQubits: 2)
/// // counts ≈ [500, 0, 0, 500] (roughly equal |00⟩ and |11⟩)
///
/// // Seeded measurement (reproducible)
/// var seededMeasurement = Measurement(seed: 42)
/// let result1 = seededMeasurement.measure(state: state)
/// var seededMeasurement2 = Measurement(seed: 42)
/// let result2 = seededMeasurement2.measure(state: state)
/// // result1.outcome == result2.outcome (deterministic)
///
/// // Partial measurement
/// var partialMeasurement = Measurement()
/// let (outcome, collapsed) = partialMeasurement.measureQubit(0, state: state)
/// // outcome: 0 or 1 (50% each for Bell state)
/// // collapsed: Other qubit still entangled with measured result
/// ```
struct Measurement {
    /// Random number generator for measurement sampling
    private var rng: any RandomNumberGenerator

    /// Whether it uses a seeded RNG (for reproducibility)
    let isSeeded: Bool

    /// Initialize with system random number generator (non-reproducible)
    init() {
        rng = SystemRandomNumberGenerator()
        isSeeded = false
    }

    /// Initialize with seeded random number generator (reproducible)
    init(seed: UInt64) {
        let source = GKMersenneTwisterRandomSource(seed: seed)
        rng = RandomNumberGeneratorWrapper(source: source)
        isSeeded = true
    }

    /// Wrapper for GKMersenneTwisterRandomSource to conform to RandomNumberGenerator
    private struct RandomNumberGeneratorWrapper: RandomNumberGenerator {
        let source: GKMersenneTwisterRandomSource

        mutating func next() -> UInt64 {
            let upper = UInt64(source.nextInt(upperBound: Int.max))
            let lower = UInt64(source.nextInt(upperBound: Int.max))
            return (upper << 32) | lower
        }
    }

    // MARK: - Probability Distribution

    /// Calculate probability distribution for all basis states.
    /// - Parameter state: Quantum state to measure
    /// - Returns: Array of probabilities [P(0), P(1), ..., P(2^n-1)]
    static func probabilityDistribution(state: QuantumState) -> [Double] {
        precondition(state.isNormalized(), "State must be normalized before measurement")
        return state.probabilities()
    }

    // MARK: - Weighted Random Sampling

    /// Sample outcome from probability distribution using roulette wheel algorithm.
    /// - Parameter probabilities: Probability distribution (must sum to 1.0)
    /// - Returns: Sampled outcome index
    private mutating func sampleOutcome(probabilities: [Double]) -> Int {
        precondition(!probabilities.isEmpty, "Probability array must not be empty")

        let sum = probabilities.reduce(0.0, +)
        precondition(abs(sum - 1.0) < 1e-6, "Probabilities must sum to 1.0 (got \(sum))")

        let random = Double.random(in: 0 ..< 1, using: &rng)

        // Roulette wheel: accumulate probabilities until exceed random
        var accumulated = 0.0
        for (index, probability) in probabilities.enumerated() {
            accumulated += probability
            if accumulated >= random {
                return index
            }
        }

        // Fallback for numerical precision (should rarely happen)
        // Return last non-zero probability index
        return probabilities.count - 1
    }

    // MARK: - Full Measurement

    /// Measure all qubits in computational basis (Born rule)
    ///
    /// Performs projective measurement of entire quantum state, sampling outcome
    /// according to P(i) = |cᵢ|². Collapses state to measured basis state |i⟩.
    /// This is the standard measurement operation in quantum computing.
    ///
    /// **Process**:
    /// 1. Calculate probability distribution P(i) = |cᵢ|² for all i
    /// 2. Sample outcome according to probabilities (roulette wheel)
    /// 3. Collapse state to |outcome⟩
    ///
    /// - Parameter state: Normalized quantum state to measure
    /// - Returns: Measurement result containing outcome and collapsed state
    ///
    /// Example:
    /// ```swift
    /// // Measure superposition |+⟩ = (|0⟩ + |1⟩)/√2
    /// let plus = QuantumState(numQubits: 1, amplitudes: [
    ///     Complex(1/sqrt(2), 0),
    ///     Complex(1/sqrt(2), 0)
    /// ])
    /// var measurement = Measurement()
    /// let result = measurement.measure(state: plus)
    /// // result.outcome: 0 or 1 (50% each)
    /// // result.collapsedState: |0⟩ or |1⟩
    ///
    /// // Measure Bell state (|00⟩ + |11⟩)/√2
    /// let bell = QuantumCircuit.bellPhiPlus().execute()
    /// let bellResult = measurement.measure(state: bell)
    /// // bellResult.outcome: 0 (|00⟩) or 3 (|11⟩) with equal probability
    ///
    /// // Seeded measurement for reproducibility
    /// var seeded = Measurement(seed: 123)
    /// let r1 = seeded.measure(state: plus)
    /// var seeded2 = Measurement(seed: 123)
    /// let r2 = seeded2.measure(state: plus)
    /// // r1.outcome == r2.outcome (same seed → same result)
    /// ```
    mutating func measure(state: QuantumState) -> MeasurementResult {
        precondition(state.isNormalized(), "State must be normalized before measurement")

        // Calculate probability distribution
        let probabilities = state.probabilities()

        // Sample outcome according to Born rule
        let outcome = sampleOutcome(probabilities: probabilities)

        // Collapse state to measured outcome
        let collapsedState = Self.collapseToOutcome(outcome, numQubits: state.numQubits)

        return MeasurementResult(outcome: outcome, collapsedState: collapsedState)
    }

    /// Create collapsed state for given outcome.
    /// All amplitudes are 0 except the measured outcome which is 1.0.
    ///
    /// - Parameters:
    ///   - outcome: Measured basis state index
    ///   - numQubits: Number of qubits
    /// - Returns: Collapsed quantum state
    static func collapseToOutcome(_ outcome: Int, numQubits: Int) -> QuantumState {
        let stateSpaceSize = 1 << numQubits
        precondition(outcome >= 0 && outcome < stateSpaceSize, "Outcome out of bounds")

        var amplitudes = [Complex<Double>](repeating: .zero, count: stateSpaceSize)
        amplitudes[outcome] = Complex(1.0, 0.0)

        return QuantumState(numQubits: numQubits, amplitudes: amplitudes)
    }

    // MARK: - Partial Measurement

    /// Measure single qubit, leaving others in superposition
    ///
    /// Performs projective measurement of one qubit while preserving quantum coherence
    /// in unmeasured qubits. Implements partial trace / marginalization. The collapsed
    /// state maintains entanglement structure for unmeasured qubits.
    ///
    /// **Process**:
    /// 1. Calculate marginal probabilities P(qubit=0) and P(qubit=1)
    /// 2. Sample outcome (0 or 1)
    /// 3. Collapse: zero incompatible amplitudes, renormalize rest
    ///
    /// **Applications**: Sequential measurements, mid-circuit measurement, error correction
    ///
    /// - Parameters:
    ///   - qubit: Qubit index to measure (0 to n-1)
    ///   - state: Normalized quantum state
    /// - Returns: Tuple (outcome ∈ {0,1}, partially collapsed state)
    ///
    /// Example:
    /// ```swift
    /// // Create Bell state (|00⟩ + |11⟩)/√2
    /// let bell = QuantumCircuit.bellPhiPlus().execute()
    ///
    /// // Measure qubit 0
    /// var measurement = Measurement()
    /// let (outcome, collapsed) = measurement.measureQubit(0, state: bell)
    ///
    /// // If outcome = 0: collapsed = |00⟩ (qubit 1 also collapsed to |0⟩)
    /// // If outcome = 1: collapsed = |11⟩ (qubit 1 also collapsed to |1⟩)
    /// // Each with 50% probability
    ///
    /// // Bell state exhibits perfect correlation
    /// let p00 = collapsed.probability(ofState: 0b00)  // 1.0 or 0.0
    /// let p11 = collapsed.probability(ofState: 0b11)  // 0.0 or 1.0
    ///
    /// // Product state example: |+⟩⊗|0⟩
    /// let product = QuantumState(numQubits: 2, amplitudes: [
    ///     Complex(1/sqrt(2), 0),  // |00⟩
    ///     Complex(0, 0),          // |01⟩
    ///     Complex(1/sqrt(2), 0),  // |10⟩
    ///     Complex(0, 0)           // |11⟩
    /// ])
    /// let (outcome2, collapsed2) = measurement.measureQubit(0, state: product)
    /// // outcome2: 0 or 1 (50% each)
    /// // collapsed2: |00⟩ or |10⟩ (qubit 1 unaffected, still |0⟩)
    /// ```
    mutating func measureQubit(_ qubit: Int, state: QuantumState) -> (outcome: Int, collapsedState: QuantumState) {
        precondition(qubit >= 0 && qubit < state.numQubits, "Qubit index out of bounds")
        precondition(state.isNormalized(), "State must be normalized before measurement")

        // Calculate marginal probabilities for this qubit
        // P(qubit=0) = sum of |amplitude[i]|² where bit(i, qubit) = 0
        // P(qubit=1) = sum of |amplitude[i]|² where bit(i, qubit) = 1
        let (prob0, prob1) = Self.marginalProbabilities(qubit: qubit, state: state)

        let outcome = sampleOutcome(probabilities: [prob0, prob1])

        // Collapse state: keep amplitudes compatible with outcome, zero others
        let collapsedState = Self.partialCollapse(
            qubit: qubit,
            outcome: outcome,
            state: state,
            probability: outcome == 0 ? prob0 : prob1
        )

        return (outcome: outcome, collapsedState: collapsedState)
    }

    /// Calculate marginal probabilities for measuring a single qubit.
    /// - Parameters:
    ///   - qubit: Qubit to measure
    ///   - state: Quantum state
    /// - Returns: (P(qubit=0), P(qubit=1))
    static func marginalProbabilities(qubit: Int, state: QuantumState) -> (Double, Double) {
        var prob0 = 0.0
        var prob1 = 0.0

        for i in 0 ..< state.stateSpaceSize {
            let probability = state.probability(ofState: i)

            if state.getBit(index: i, qubit: qubit) == 0 {
                prob0 += probability
            } else {
                prob1 += probability
            }
        }

        return (prob0, prob1)
    }

    /// Partially collapse state after measuring one qubit.
    /// - Parameters:
    ///   - qubit: Measured qubit index
    ///   - outcome: Measured value (0 or 1)
    ///   - state: Original quantum state
    ///   - probability: Probability of this outcome (for renormalization)
    /// - Returns: Collapsed state with unmeasured qubits still in superposition
    static func partialCollapse(
        qubit: Int,
        outcome: Int,
        state: QuantumState,
        probability: Double
    ) -> QuantumState {
        precondition(outcome == 0 || outcome == 1, "Outcome must be 0 or 1")
        precondition(probability > 0, "Probability must be positive")

        // Renormalization factor: 1/√P(outcome)
        let normalizationFactor = 1.0 / sqrt(probability)

        var newAmplitudes = [Complex<Double>](repeating: .zero, count: state.stateSpaceSize)

        for i in 0 ..< state.stateSpaceSize {
            if state.getBit(index: i, qubit: qubit) == outcome {
                newAmplitudes[i] = state.amplitudes[i] * normalizationFactor
            }
            // else: incompatible, already zero
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    // MARK: - Multiple Measurements (Statistics)

    /// Execute circuit multiple times and collect measurement statistics
    ///
    /// Runs circuit repeatedly, measuring final state each time to build empirical
    /// probability distribution. Essential for validating quantum algorithms and
    /// visualizing measurement outcomes. Use with `histogram()` for frequency analysis.
    ///
    /// **Use cases**:
    /// - Algorithm validation (compare observed vs expected distribution)
    /// - Visualization (plot measurement frequencies)
    /// - Statistical testing (chi-squared goodness-of-fit)
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit to execute
    ///   - numRuns: Number of independent executions (≥ 1000 recommended for statistics)
    /// - Returns: Array of measurement outcomes [outcome₁, outcome₂, ..., outcomeₙ]
    ///
    /// Example:
    /// ```swift
    /// // Measure Bell state 1000 times
    /// let bellCircuit = QuantumCircuit.bellPhiPlus()
    /// var measurement = Measurement()
    /// let outcomes = measurement.runMultiple(circuit: bellCircuit, numRuns: 1000)
    ///
    /// // Convert to histogram
    /// let counts = Measurement.histogram(outcomes: outcomes, numQubits: 2)
    /// // counts ≈ [~500, 0, 0, ~500] for Bell state
    /// print("Measured |00⟩: \(counts[0]) times")
    /// print("Measured |11⟩: \(counts[3]) times")
    ///
    /// // Compare to expected distribution
    /// let expected = [0.5, 0.0, 0.0, 0.5]
    /// let error = Measurement.compareDistributions(
    ///     observed: counts,
    ///     expected: expected,
    ///     totalRuns: 1000
    /// )
    /// // error < 0.1 (within 10% for reasonable sample size)
    ///
    /// // Chi-squared test
    /// let chiSq = Measurement.chiSquared(
    ///     observed: counts,
    ///     expected: expected,
    ///     totalRuns: 1000
    /// )
    /// // chiSq.chiSquared < critical value → good fit
    /// ```
    mutating func runMultiple(circuit: QuantumCircuit, numRuns: Int) -> [Int] {
        precondition(numRuns > 0, "Number of runs must be positive")

        var outcomes: [Int] = []
        outcomes.reserveCapacity(numRuns)

        for _ in 0 ..< numRuns {
            let finalState = circuit.execute()
            let result = measure(state: finalState)
            outcomes.append(result.outcome)
        }

        return outcomes
    }

    /// Convert outcomes to histogram (count per basis state).
    /// - Parameters:
    ///   - outcomes: Array of measurement outcomes
    ///   - numQubits: Number of qubits (determines state space size)
    /// - Returns: Array of counts [count(0), count(1), ..., count(2^n-1)]
    static func histogram(outcomes: [Int], numQubits: Int) -> [Int] {
        let stateSpaceSize = 1 << numQubits
        var counts = [Int](repeating: 0, count: stateSpaceSize)

        for outcome in outcomes {
            if outcome >= 0, outcome < stateSpaceSize {
                counts[outcome] += 1
            }
        }

        return counts
    }

    /// Compare observed frequencies to expected probabilities.
    /// - Parameters:
    ///   - observed: Observed outcome counts
    ///   - expected: Expected probabilities
    ///   - totalRuns: Total number of measurements
    /// - Returns: Maximum relative error across all outcomes
    static func compareDistributions(
        observed: [Int],
        expected: [Double],
        totalRuns: Int
    ) -> Double {
        precondition(observed.count == expected.count, "Array sizes must match")
        precondition(totalRuns > 0, "Total runs must be positive")

        var maxError = 0.0

        for i in 0 ..< observed.count {
            let observedFreq = Double(observed[i]) / Double(totalRuns)
            let expectedFreq = expected[i]

            // Avoid division by zero
            if expectedFreq > 0 {
                let relativeError = abs(observedFreq - expectedFreq) / expectedFreq
                maxError = max(maxError, relativeError)
            } else if observedFreq > 0 {
                // Expected 0 but observed something
                maxError = max(maxError, observedFreq)
            }
        }

        return maxError
    }

    /// Chi-squared goodness-of-fit test result
    struct ChiSquaredResult {
        /// Chi-squared statistic (lower is better fit)
        let chiSquared: Double

        /// Degrees of freedom (bins tested - 1)
        let degreesOfFreedom: Int

        /// Number of bins that were tested (expectedCount >= 5)
        let testedBins: Int

        /// Number of bins that were skipped (expectedCount < 5)
        let skippedBins: Int
    }

    /// Chi-squared goodness-of-fit test.
    /// Tests whether observed distribution matches expected.
    ///
    /// - Parameters:
    ///   - observed: Observed outcome counts
    ///   - expected: Expected probabilities
    ///   - totalRuns: Total number of measurements
    /// - Returns: Chi-squared result with statistic and degrees of freedom
    static func chiSquared(
        observed: [Int],
        expected: [Double],
        totalRuns: Int
    ) -> ChiSquaredResult {
        precondition(observed.count == expected.count, "Array sizes must match")
        precondition(totalRuns > 0, "Total runs must be positive")

        var chiSq = 0.0
        var testedBins = 0
        var skippedBins = 0

        for i in 0 ..< observed.count {
            let expectedCount = expected[i] * Double(totalRuns)

            // Skip if expected count too small (chi-squared invalid)
            if expectedCount < 5 {
                skippedBins += 1
                continue
            }

            let diff = Double(observed[i]) - expectedCount
            chiSq += (diff * diff) / expectedCount
            testedBins += 1
        }

        let degreesOfFreedom = max(testedBins - 1, 0)

        return ChiSquaredResult(
            chiSquared: chiSq,
            degreesOfFreedom: degreesOfFreedom,
            testedBins: testedBins,
            skippedBins: skippedBins
        )
    }
}

// MARK: - Static Helper Functions

extension Measurement {
    /// Measure state with a fresh RNG (convenience function).
    /// - Parameter state: Quantum state to measure
    /// - Returns: Measurement result
    static func measureOnce(state: QuantumState) -> MeasurementResult {
        var measurement = Measurement()
        return measurement.measure(state: state)
    }

    /// Run circuit multiple times and collect measurement outcomes (static convenience).
    /// - Parameters:
    ///   - circuit: Quantum circuit to execute
    ///   - numRuns: Number of measurement runs
    /// - Returns: Array of outcomes
    static func runMultiple(circuit: QuantumCircuit, numRuns: Int) -> [Int] {
        var measurement = Measurement()
        return measurement.runMultiple(circuit: circuit, numRuns: numRuns)
    }

    /// Measure single qubit with fresh RNG (convenience function).
    /// - Parameters:
    ///   - qubit: Qubit index to measure
    ///   - state: Quantum state
    /// - Returns: Outcome and collapsed state
    static func measureQubitOnce(_ qubit: Int, state: QuantumState) -> (outcome: Int, collapsedState: QuantumState) {
        var measurement = Measurement()
        return measurement.measureQubit(qubit, state: state)
    }
}
