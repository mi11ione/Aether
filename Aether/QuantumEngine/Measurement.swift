// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation
import GameplayKit

/// Result of a quantum measurement
struct MeasurementResult: Equatable, CustomStringConvertible {
    /// Classical outcome (basis state index)
    let outcome: Int

    /// Post-measurement quantum state (collapsed)
    let collapsedState: QuantumState

    /// String representation
    var description: String {
        "Measurement: outcome=\(outcome), state=\(collapsedState)"
    }
}

/// Quantum measurement system implementing Born rule.
///
/// Provides measurement operations for quantum states following the Born rule:
/// - Probability of measuring basis state |i⟩ is P(i) = |cᵢ|²
/// - Measurement collapses the state to the measured outcome
/// - Supports full measurement (all qubits) and partial measurement (subset)
///
/// Supports both random and seeded (reproducible) measurements.
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

    /// Perform full measurement of all qubits (computational basis).
    /// Implements Born rule: P(i) = |cᵢ|²
    ///
    /// - Parameter state: Quantum state to measure
    /// - Returns: Measurement result with outcome and collapsed state
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

    /// Measure a single qubit, leaving others in superposition.
    /// Implements marginalization over unmeasured qubits.
    ///
    /// - Parameters:
    ///   - qubit: Index of qubit to measure
    ///   - state: Quantum state to measure
    /// - Returns: Measurement result (0 or 1) and partially collapsed state
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

    /// Run circuit multiple times and collect measurement outcomes.
    /// Used for statistical validation and visualization.
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit to execute
    ///   - numRuns: Number of measurement runs
    /// - Returns: Array of outcomes
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
