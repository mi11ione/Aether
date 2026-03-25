// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Test suite for quantum state collapse.
/// Validates wave function collapse to computational basis states
/// following measurement, ensuring unitarity and normalization.
@Suite("State Collapse")
struct StateCollapseTests {
    @Test("Collapse to outcome 0")
    func collapseToZero() {
        let state = QuantumState(qubits: 2)
        let result = Measurement.measure(state)
        let collapsed = result.collapsedState

        #expect(result.outcome == 0, "Measuring |00> should give outcome 0")
        #expect(abs(collapsed.amplitude(of: 0).real - 1.0) < 1e-10, "Collapsed state should have amplitude 1.0 at outcome 0")
        #expect(abs(collapsed.amplitude(of: 1).magnitude) < 1e-10, "Collapsed state should have amplitude 0 at outcome 1")
        #expect(abs(collapsed.amplitude(of: 2).magnitude) < 1e-10, "Collapsed state should have amplitude 0 at outcome 2")
        #expect(abs(collapsed.amplitude(of: 3).magnitude) < 1e-10, "Collapsed state should have amplitude 0 at outcome 3")
    }

    @Test("Collapse to outcome 3")
    func collapseToThree() {
        var amplitudes = [Complex<Double>](repeating: .zero, count: 4)
        amplitudes[3] = .one
        let state = QuantumState(qubits: 2, amplitudes: amplitudes)
        let result = Measurement.measure(state)
        let collapsed = result.collapsedState

        #expect(result.outcome == 3, "Measuring |11> should give outcome 3")
        #expect(abs(collapsed.amplitude(of: 0).magnitude) < 1e-10, "Collapsed state should have amplitude 0 at outcome 0")
        #expect(abs(collapsed.amplitude(of: 1).magnitude) < 1e-10, "Collapsed state should have amplitude 0 at outcome 1")
        #expect(abs(collapsed.amplitude(of: 2).magnitude) < 1e-10, "Collapsed state should have amplitude 0 at outcome 2")
        #expect(abs(collapsed.amplitude(of: 3).real - 1.0) < 1e-10, "Collapsed state should have amplitude 1.0 at outcome 3")
    }

    @Test("Collapsed state is normalized")
    func collapsedStateNormalized() {
        var amplitudes = [Complex<Double>](repeating: .zero, count: 8)
        amplitudes[1] = .one
        let state = QuantumState(qubits: 3, amplitudes: amplitudes)
        let result = Measurement.measure(state)
        #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized")
    }
}

/// Test suite for full quantum measurement.
/// Validates complete state collapse to single basis state
/// with proper probabilistic sampling according to Born rule.
@Suite("Full Measurement")
struct FullMeasurementTests {
    @Test("Measure |0⟩ always gives 0")
    func measureZeroState() {
        let state = QuantumState(qubit: 0)

        for _ in 0 ..< 10 {
            let result = Measurement.measure(state)
            #expect(result.outcome == 0, "Measuring |0> should always give outcome 0")
            #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized")
        }
    }

    @Test("Measure |1⟩ always gives 1")
    func measureOneState() {
        let state = QuantumState(qubit: 1)

        for _ in 0 ..< 10 {
            let result = Measurement.measure(state)
            #expect(result.outcome == 1, "Measuring |1> should always give outcome 1")
        }
    }

    @Test("Collapsed state is basis state")
    func collapsedStateIsBasisState() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [
            Complex(invSqrt2, 0.0),
            Complex(invSqrt2, 0.0),
        ]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)

        let result = Measurement.measure(state)

        #expect(result.outcome == 0 || result.outcome == 1, "Measurement outcome should be 0 or 1")
        #expect(abs(result.collapsedState.amplitude(of: result.outcome).real - 1.0) < 1e-10, "Collapsed state should have amplitude 1.0 at measured outcome")
    }

    @Test("Measuring collapsed state is deterministic")
    func measureCollapsedStateIsDeterministic() {
        let state = QuantumState(qubit: 0)

        let firstResult = Measurement.measure(state)
        let secondResult = Measurement.measure(firstResult.collapsedState)

        #expect(firstResult.outcome == secondResult.outcome, "Re-measuring collapsed state should give same outcome")
    }
}

/// Test suite for measurement statistical distributions.
/// Validates that repeated measurements follow Born rule probabilities,
/// testing the fundamental link between quantum amplitudes and measurement outcomes.
@Suite("Statistical Distribution")
struct StatisticalDistributionTests {
    @Test("Equal superposition has ~50/50 distribution")
    func equalSuperpositionStatistics() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [
            Complex(invSqrt2, 0.0),
            Complex(invSqrt2, 0.0),
        ]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)

        let numRuns = 1000
        var counts = [0, 0]

        for _ in 0 ..< numRuns {
            let result = Measurement.measure(state)
            counts[result.outcome] += 1
        }

        let freq0 = Double(counts[0]) / Double(numRuns)
        let freq1 = Double(counts[1]) / Double(numRuns)

        #expect(abs(freq0 - 0.5) < 0.1, "Outcome 0 frequency should be approximately 0.5")
        #expect(abs(freq1 - 0.5) < 0.1, "Outcome 1 frequency should be approximately 0.5")
    }

    @Test("Bell state statistics")
    func bellStateStatistics() {
        let circuit = QuantumCircuit.bell()

        let numRuns = 1000
        let outcomes = Measurement.sample(circuit: circuit, shots: numRuns)
        let histogram = Measurement.histogram(outcomes: outcomes, qubits: 2)

        let freq0 = Double(histogram[0]) / Double(numRuns)
        let freq3 = Double(histogram[3]) / Double(numRuns)

        #expect(abs(freq0 - 0.5) < 0.1, "Bell state outcome 00 frequency should be approximately 0.5")
        #expect(abs(freq3 - 0.5) < 0.1, "Bell state outcome 11 frequency should be approximately 0.5")

        #expect(histogram[1] < numRuns / 20, "Bell state outcome 01 should be rare")
        #expect(histogram[2] < numRuns / 20, "Bell state outcome 10 should be rare")
    }

    @Test("Weighted state statistics")
    func weightedStateStatistics() {
        let amplitudes = [
            Complex(0.6, 0.0),
            Complex(0.8, 0.0),
        ]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)

        let numRuns = 1000
        var counts = [0, 0]

        for _ in 0 ..< numRuns {
            let result = Measurement.measure(state)
            counts[result.outcome] += 1
        }

        let freq0 = Double(counts[0]) / Double(numRuns)
        let freq1 = Double(counts[1]) / Double(numRuns)

        #expect(abs(freq0 - 0.36) < 0.1, "Outcome 0 frequency should be approximately 0.36")
        #expect(abs(freq1 - 0.64) < 0.1, "Outcome 1 frequency should be approximately 0.64")
    }
}

/// Test suite for partial quantum measurements.
/// Validates single-qubit measurement while preserving superposition in remaining qubits,
/// demonstrating conditional state updates and entanglement preservation.
@Suite("Partial Measurement")
struct PartialMeasurementTests {
    @Test("Measure single qubit of separable state")
    func measureSingleQubitSeparable() {
        var amplitudes = [Complex<Double>](repeating: .zero, count: 4)
        amplitudes[1] = .one
        let state = QuantumState(qubits: 2, amplitudes: amplitudes)

        for _ in 0 ..< 10 {
            let result = Measurement.measure(0, in: state)
            #expect(result.outcome == 1, "Measuring qubit 0 of |01> should always give 1")
            #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after partial measurement")
        }
    }

    @Test("Measure qubit in Bell state demonstrates entanglement")
    func measureBellStateEntanglement() {
        let circuit = QuantumCircuit.bell()
        let state = circuit.execute()

        let result0 = Measurement.measure(0, in: state)
        let result1 = Measurement.measure(1, in: result0.collapsedState)

        #expect(result0.outcome == result1.outcome, "Bell state qubits should be correlated")
    }

    @Test("Partial measurement statistics")
    func partialMeasurementStatistics() {
        let circuit = QuantumCircuit.bell()

        let numRuns = 1000
        var outcomes = [Int]()

        for _ in 0 ..< numRuns {
            let state = circuit.execute()
            let result = Measurement.measure(0, in: state)
            outcomes.append(result.outcome)
        }

        let count0 = outcomes.count(where: { $0 == 0 })
        let count1 = outcomes.count(where: { $0 == 1 })

        let freq0 = Double(count0) / Double(numRuns)
        let freq1 = Double(count1) / Double(numRuns)

        #expect(abs(freq0 - 0.5) < 0.1, "Partial measurement outcome 0 frequency should be approximately 0.5")
        #expect(abs(freq1 - 0.5) < 0.1, "Partial measurement outcome 1 frequency should be approximately 0.5")
    }
}

/// Test suite for statistical analysis helpers.
/// Validates histogram generation, distribution comparison, and chi-squared
/// goodness-of-fit tests for validating quantum measurement statistics.
@Suite("Statistical Helpers")
struct StatisticalHelpersTests {
    @Test("Histogram counts outcomes correctly")
    func histogramCounts() {
        let outcomes = [0, 1, 1, 0, 2, 1, 0, 3]
        let histogram = Measurement.histogram(outcomes: outcomes, qubits: 2)

        #expect(histogram[0] == 3, "Outcome 0 should appear 3 times")
        #expect(histogram[1] == 3, "Outcome 1 should appear 3 times")
        #expect(histogram[2] == 1, "Outcome 2 should appear 1 time")
        #expect(histogram[3] == 1, "Outcome 3 should appear 1 time")
    }

    @Test("Compare perfect match")
    func compareDistributionsPerfect() {
        let observed = [500, 500]
        let expected = [0.5, 0.5]

        let error = Measurement.relativeError(
            observed: observed,
            expected: expected,
            totalShots: 1000,
        )

        #expect(error < 0.01, "Relative error for perfect match should be near zero")
    }

    @Test("Compare with deviation")
    func compareDistributionsDeviation() {
        let observed = [400, 600]
        let expected = [0.5, 0.5]

        let error = Measurement.relativeError(
            observed: observed,
            expected: expected,
            totalShots: 1000,
        )

        #expect(error > 0.15, "Relative error for deviated distribution should exceed 0.15")
    }

    @Test("Chi-squared for good fit")
    func chiSquaredGoodFit() {
        let observed = [505, 495]
        let expected = [0.5, 0.5]

        let chiSq = Measurement.chiSquared(
            observed: observed,
            expected: expected,
            totalShots: 1000,
        )

        #expect(chiSq.chiSquared < 5.0, "Chi-squared value for good fit should be less than 5.0")
    }

    @Test("Run circuit multiple times")
    func runCircuitMultipleTimes() {
        let circuit = QuantumCircuit.bell()

        let outcomes = Measurement.sample(circuit: circuit, shots: 100)

        #expect(outcomes.count == 100, "Should produce exactly 100 outcomes")
        #expect(outcomes.allSatisfy { $0 >= 0 && $0 < 4 }, "All outcomes should be valid 2-qubit basis states")
    }
}

/// Test suite for measurement edge cases.
/// Validates measurement behavior across different qubit counts
/// and complex amplitude configurations.
@Suite("Edge Cases")
struct MeasurementEdgeCasesTests {
    @Test("Measure single-qubit state")
    func measureSingleQubit() {
        let state = QuantumState(qubit: 0)
        let result = Measurement.measure(state)

        #expect(result.outcome == 0, "Single-qubit |0> measurement should give outcome 0")
        #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized")
    }

    @Test("Measure 3-qubit state")
    func measureThreeQubits() {
        let circuit = QuantumCircuit.uniformSuperposition(qubits: 3)
        let state = circuit.execute()

        let result = Measurement.measure(state)

        #expect(result.outcome >= 0 && result.outcome < 8, "3-qubit measurement outcome should be in range 0..<8")
        #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized")
    }

    @Test("Measure state with complex amplitudes")
    func measureComplexAmplitudes() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [
            Complex(invSqrt2, 0.0),
            Complex(0.0, invSqrt2),
        ]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)

        var counts = [0, 0]
        for _ in 0 ..< 100 {
            let result = Measurement.measure(state)
            counts[result.outcome] += 1
        }

        #expect(counts[0] > 0, "Outcome 0 should occur at least once for equal superposition")
        #expect(counts[1] > 0, "Outcome 1 should occur at least once for equal superposition")
    }
}

/// Test suite for normalization preservation.
/// Ensures measurements maintain Σ|cᵢ|² = 1.0 throughout
/// collapse operations and partial measurements.
@Suite("Normalization Checks")
struct NormalizationChecksTests {
    @Test("All measurements preserve normalization")
    func measurementsPreserveNormalization() {
        let circuit = QuantumCircuit.bell()
        let state = circuit.execute()

        for _ in 0 ..< 10 {
            let result = Measurement.measure(state)
            #expect(result.collapsedState.isNormalized(), "Full measurement should preserve normalization")
        }
    }

    @Test("Partial measurements preserve normalization")
    func partialMeasurementsPreserveNormalization() {
        let circuit = QuantumCircuit.ghz(qubits: 3)
        let state = circuit.execute()

        let result1 = Measurement.measure(0, in: state)
        #expect(result1.collapsedState.isNormalized(), "State should be normalized after measuring qubit 0")

        let result2 = Measurement.measure(1, in: result1.collapsedState)
        #expect(result2.collapsedState.isNormalized(), "State should be normalized after measuring qubit 1")

        let result3 = Measurement.measure(2, in: result2.collapsedState)
        #expect(result3.collapsedState.isNormalized(), "State should be normalized after measuring qubit 2")
    }
}

/// Test suite for determinism after collapse.
/// Validates that measuring an already-collapsed state
/// yields the same outcome deterministically.
@Suite("Determinism After Collapse")
struct DeterminismTests {
    @Test("Re-measuring collapsed state gives same outcome")
    func remeasureCollapsedState() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [
            Complex(invSqrt2, 0.0),
            Complex(invSqrt2, 0.0),
        ]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)

        let firstResult = Measurement.measure(state)
        let secondResult = Measurement.measure(firstResult.collapsedState)
        let thirdResult = Measurement.measure(secondResult.collapsedState)

        #expect(firstResult.outcome == secondResult.outcome, "Second measurement should match first after collapse")
        #expect(secondResult.outcome == thirdResult.outcome, "Third measurement should match second after collapse")
    }
}

/// Test suite for seeded measurements and reproducibility.
/// Validates deterministic measurement sequences using seeded RNG,
/// essential for testing, debugging, and quantum circuit validation.
@Suite("Seeded Measurements")
struct SeededMeasurementTests {
    @Test("Seeded measurement is reproducible")
    func seededMeasurementReproducible() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [
            Complex(invSqrt2, 0.0),
            Complex(invSqrt2, 0.0),
        ]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)

        let result1 = Measurement.measure(state, seed: 42)
        let result2 = Measurement.measure(state, seed: 42)

        #expect(result1.outcome == result2.outcome, "Same seed should produce same measurement outcome")
    }

    @Test("Different seeds give different outcomes")
    func differentSeedsGiveDifferentOutcomes() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [
            Complex(invSqrt2, 0.0),
            Complex(invSqrt2, 0.0),
        ]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)

        var outcomes1: [Int] = []
        var outcomes2: [Int] = []

        for i in 0 ..< 20 {
            outcomes1.append(Measurement.measure(state, seed: UInt64(42 + i)).outcome)
            outcomes2.append(Measurement.measure(state, seed: UInt64(100 + i)).outcome)
        }

        #expect(outcomes1 != outcomes2, "Different seeds should produce different outcome sequences")
    }

    @Test("Unseeded measurements vary")
    func unseededMeasurementsVary() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [
            Complex(invSqrt2, 0.0),
            Complex(invSqrt2, 0.0),
        ]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)

        var outcomes: [Int] = []
        for _ in 0 ..< 100 {
            outcomes.append(Measurement.measure(state).outcome)
        }

        let unique = Set(outcomes)
        #expect(unique.count == 2, "Should observe both outcomes")
    }

    @Test("Seeded measurement produces valid outcomes")
    func seededMeasurementValidOutcomes() {
        let state = QuantumState(qubits: 2)

        for i in 0 ..< 50 {
            let result = Measurement.measure(state, seed: UInt64(123 + i))
            #expect(result.outcome >= 0 && result.outcome < 4, "Seeded measurement outcome should be valid 2-qubit basis state")
            #expect(result.collapsedState.isNormalized(), "Seeded measurement collapsed state should be normalized")
        }
    }
}

/// Test suite for MeasurementResult description and display.
/// Validates human-readable output for debugging and UI display
/// across all measurement result types.
@Suite("Measurement Result Display")
struct MeasurementResultDisplayTests {
    @Test("MeasurementResult description includes outcome")
    func descriptionIncludesOutcome() {
        let state = QuantumState(qubit: 0)
        let result = Measurement.measure(state)

        #expect(result.description.contains("outcome"), "Description should contain 'outcome'")
        #expect(result.description.contains("0"), "Description should contain the outcome value")
    }

    @Test("MeasurementResult description includes state info")
    func descriptionIncludesState() {
        let state = QuantumState(qubit: 1)
        let result = Measurement.measure(state)

        #expect(result.description.contains("state"), "Description should contain state information")
    }

    @Test("MeasurementResult description is non-empty")
    func descriptionNonEmpty() {
        let state = QuantumState(qubits: 2)
        let result = Measurement.measure(state)

        #expect(!result.description.isEmpty, "Description should not be empty")
    }
}

/// Test suite for static API ergonomics.
/// Validates clean API surface for common measurement patterns
/// including histogram, sampling, and convenience overloads.
@Suite("Static API Ergonomics")
struct StaticAPITests {
    @Test("Full state measurement")
    func fullStateMeasurement() {
        let state = QuantumState(qubit: 0)
        let result = Measurement.measure(state)

        #expect(result.outcome == 0, "Full state measurement of |0> should give outcome 0")
        #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized")
    }

    @Test("Single qubit measurement")
    func singleQubitMeasurement() {
        let state = QuantumState(qubits: 2)
        let result = Measurement.measure(0, in: state)

        #expect(result.outcome == 0 || result.outcome == 1, "Single qubit measurement outcome should be 0 or 1")
        #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after single qubit measurement")
    }

    @Test("Circuit sampling")
    func circuitSampling() {
        let circuit = QuantumCircuit.bell()
        let outcomes = Measurement.sample(circuit: circuit, shots: 50)

        #expect(outcomes.count == 50, "Should produce exactly 50 sample outcomes")
        #expect(outcomes.allSatisfy { $0 >= 0 && $0 < 4 }, "All outcomes should be valid 2-qubit basis states")
    }
}

/// Test suite for statistical analysis edge cases.
/// Validates boundary conditions and error handling in distribution comparison
/// and chi-squared test implementations.
@Suite("Statistical Analysis Edge Cases")
struct StatisticalEdgeCasesTests {
    @Test("Compare distributions when expected is zero")
    func compareDistributionsExpectedZero() {
        let observed = [100, 5, 0, 0]
        let expected = [0.95, 0.05, 0.0, 0.0]

        let error = Measurement.relativeError(
            observed: observed,
            expected: expected,
            totalShots: 105,
        )

        #expect(error >= 0.0, "Relative error should be non-negative")
    }

    @Test("Compare distributions when observed exceeds expected zero")
    func compareDistributionsObservedExceedsExpectedZero() {
        let observed = [90, 0, 10, 0]
        let expected = [0.9, 0.0, 0.0, 0.1]

        let error = Measurement.relativeError(
            observed: observed,
            expected: expected,
            totalShots: 100,
        )

        #expect(error > 0.0, "Relative error should be positive when observed deviates from expected")
    }

    @Test("Chi-squared with small expected counts")
    func chiSquaredSmallExpectedCounts() {
        let observed = [95, 3, 1, 1]
        let expected = [0.95, 0.03, 0.01, 0.01]

        let result = Measurement.chiSquared(
            observed: observed,
            expected: expected,
            totalShots: 100,
        )

        #expect(result.skippedBins > 0, "Small expected counts should cause some bins to be skipped")
        #expect(result.testedBins < 4, "Not all bins should be tested with small expected counts")
    }

    @Test("Chi-squared with all bins tested")
    func chiSquaredAllBinsTested() {
        let observed = [250, 250, 250, 250]
        let expected = [0.25, 0.25, 0.25, 0.25]

        let result = Measurement.chiSquared(
            observed: observed,
            expected: expected,
            totalShots: 1000,
        )

        #expect(result.testedBins == 4, "All 4 bins should be tested with uniform distribution")
        #expect(result.skippedBins == 0, "No bins should be skipped with uniform distribution")
        #expect(result.degreesOfFreedom == 3, "Degrees of freedom should be testedBins - 1")
    }

    @Test("Chi-squared with poor fit")
    func chiSquaredPoorFit() {
        let observed = [900, 50, 25, 25]
        let expected = [0.25, 0.25, 0.25, 0.25]

        let result = Measurement.chiSquared(
            observed: observed,
            expected: expected,
            totalShots: 1000,
        )

        #expect(result.chiSquared > 100.0, "Chi-squared should be large for poor fit")
    }

    @Test("Histogram handles out-of-bounds outcomes gracefully")
    func histogramOutOfBounds() {
        let outcomes = [0, 1, 2, 3, 10, -1, 5]
        let histogram = Measurement.histogram(outcomes: outcomes, qubits: 2)

        #expect(histogram.count == 4, "Histogram should have 4 bins for 2-qubit system")
        #expect(histogram[0] == 1, "Outcome 0 should appear once")
        #expect(histogram[1] == 1, "Outcome 1 should appear once")
        #expect(histogram[2] == 1, "Outcome 2 should appear once")
        #expect(histogram[3] == 1, "Outcome 3 should appear once")
    }
}

/// Test suite for Pauli basis measurements (X, Y, Z).
/// Validates measurement in non-computational bases with eigenvalue ±1 outcomes.
/// Essential for observable expectation values and quantum chemistry applications.
@Suite("Pauli Basis Measurements")
struct PauliBasisMeasurementTests {
    @Test("Measure |0⟩ in Z basis gives +1")
    func measureZeroInZBasis() {
        let state = QuantumState(qubits: 1)

        for _ in 0 ..< 10 {
            let result = Measurement.measure(0, basis: .z, in: state)
            #expect(result.eigenvalue == 1, "Measuring |0> in Z basis should give eigenvalue +1")
            #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after Z measurement")
        }
    }

    @Test("Measure |1⟩ in Z basis gives -1")
    func measureOneInZBasis() {
        let state = QuantumState(qubit: 1)

        for _ in 0 ..< 10 {
            let result = Measurement.measure(0, basis: .z, in: state)
            #expect(result.eigenvalue == -1, "Measuring |1> in Z basis should give eigenvalue -1")
            #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after Z measurement")
        }
    }

    @Test("Measure |+⟩ in X basis gives +1")
    func measurePlusInXBasis() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let plus = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0),
            Complex(invSqrt2, 0),
        ])

        for _ in 0 ..< 10 {
            let result = Measurement.measure(0, basis: .x, in: plus)
            #expect(result.eigenvalue == 1, "Measuring |+> in X basis should give eigenvalue +1")
            #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after X measurement")
        }
    }

    @Test("Measure |-⟩ in X basis gives -1")
    func measureMinusInXBasis() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let minus = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0),
            Complex(-invSqrt2, 0),
        ])

        for _ in 0 ..< 10 {
            let result = Measurement.measure(0, basis: .x, in: minus)
            #expect(result.eigenvalue == -1, "Measuring |-> in X basis should give eigenvalue -1")
            #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after X measurement")
        }
    }

    @Test("Measure |0⟩ in X basis gives ±1 with equal probability")
    func measureZeroInXBasis() {
        let state = QuantumState(qubits: 1)

        var outcomes = [Int]()
        for _ in 0 ..< 100 {
            let result = Measurement.measure(0, basis: .x, in: state)
            outcomes.append(result.eigenvalue)
            #expect(result.eigenvalue == 1 || result.eigenvalue == -1, "X basis eigenvalue should be +1 or -1")
        }

        let countPlus = outcomes.count(where: { $0 == 1 })
        let countMinus = outcomes.count(where: { $0 == -1 })

        #expect(abs(Double(countPlus) - 50.0) < 20.0, "X basis +1 count should be approximately 50")
        #expect(abs(Double(countMinus) - 50.0) < 20.0, "X basis -1 count should be approximately 50")
    }

    @Test("Pauli Y measurement on |0⟩")
    func measureZeroInYBasis() {
        let state = QuantumState(qubits: 1)

        var outcomes = [Int]()
        for _ in 0 ..< 100 {
            let result = Measurement.measure(0, basis: .y, in: state)
            outcomes.append(result.eigenvalue)
            #expect(result.eigenvalue == 1 || result.eigenvalue == -1, "Y basis eigenvalue should be +1 or -1")
            #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after Y measurement")
        }

        let countPlus = outcomes.count(where: { $0 == 1 })
        let countMinus = outcomes.count(where: { $0 == -1 })

        #expect(abs(Double(countPlus) - 50.0) < 20.0, "Y basis +1 count should be approximately 50")
        #expect(abs(Double(countMinus) - 50.0) < 20.0, "Y basis -1 count should be approximately 50")
    }

    @Test("Pauli measurement on Bell state")
    func pauliMeasurementBellState() {
        let bell = QuantumCircuit.bellPhiPlus().execute()

        let zResult = Measurement.measure(0, basis: .z, in: bell)
        #expect(zResult.eigenvalue == 1 || zResult.eigenvalue == -1, "Z measurement on Bell state should give +1 or -1")
        #expect(zResult.collapsedState.isNormalized(), "Collapsed state should be normalized after Z measurement on Bell state")

        let xResult = Measurement.measure(0, basis: .x, in: bell)
        #expect(xResult.eigenvalue == 1 || xResult.eigenvalue == -1, "X measurement on Bell state should give +1 or -1")
        #expect(xResult.collapsedState.isNormalized(), "Collapsed state should be normalized after X measurement on Bell state")
    }

    @Test("PauliBasis enum cases")
    func pauliBasisEnumCases() {
        let allCases = PauliBasis.allCases
        #expect(allCases.contains(.x), "PauliBasis should contain .x case")
        #expect(allCases.contains(.y), "PauliBasis should contain .y case")
        #expect(allCases.contains(.z), "PauliBasis should contain .z case")
        #expect(allCases.count == 3, "PauliBasis should have exactly 3 cases")
    }

    @Test("PauliMeasurementResult description")
    func pauliMeasurementResultDescription() {
        let state = QuantumState(qubits: 1)
        let result = Measurement.measure(0, basis: .z, in: state)

        #expect(result.description.contains("PauliMeasurement"), "Description should contain 'PauliMeasurement' prefix")
        #expect(result.description.contains("+1") || result.description.contains("-1"), "Description should contain eigenvalue +1 or -1")
    }
}

/// Test suite for arbitrary single-qubit basis measurements.
/// Validates measurement in custom unitary bases beyond standard Pauli operators,
/// including projective measurement and post-measurement state collapse.
@Suite("Custom Basis Measurements")
struct CustomBasisMeasurementTests {
    @Test("Measure in custom basis (eigenstate)")
    func measureCustomBasisEigenstate() {
        let phase = Double.pi / 4
        let customBasis = [
            Complex(1 / sqrt(2), 0),
            Complex(cos(phase) / sqrt(2), sin(phase) / sqrt(2)),
        ]

        let state = QuantumState(qubits: 1, amplitudes: customBasis)

        for _ in 0 ..< 10 {
            let result = Measurement.measure(0, basis: customBasis, in: state)
            #expect(result.outcome == 0, "Eigenstate measurement in custom basis should always give outcome 0")
            #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after custom basis measurement")
        }
    }

    @Test("Measure |0⟩ in custom basis")
    func measureZeroInCustomBasis() {
        let state = QuantumState(qubits: 1)

        let customBasis = [
            Complex(1 / sqrt(2), 0),
            Complex(1 / sqrt(2), 0),
        ]

        var outcomes = [Int]()

        for _ in 0 ..< 100 {
            let result = Measurement.measure(0, basis: customBasis, in: state)
            outcomes.append(result.outcome)
            #expect(result.outcome == 0 || result.outcome == 1, "Custom basis measurement outcome should be 0 or 1")
            #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after custom basis measurement")
        }

        let count0 = outcomes.count(where: { $0 == 0 })
        let count1 = outcomes.count(where: { $0 == 1 })
        #expect(abs(Double(count0) - 50.0) < 20.0, "Custom basis outcome 0 count should be approximately 50")
        #expect(abs(Double(count1) - 50.0) < 20.0, "Custom basis outcome 1 count should be approximately 50")
    }

    @Test("Custom basis with complex phase")
    func customBasisComplexPhase() {
        let customBasis = [
            Complex(0.6, 0.0),
            Complex(0.0, 0.8),
        ]

        let state = QuantumState(qubits: 1)

        let result = Measurement.measure(0, basis: customBasis, in: state)

        #expect(result.outcome == 0 || result.outcome == 1, "Complex phase custom basis outcome should be 0 or 1")
        #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after complex phase basis measurement")
    }

    @Test("Custom basis measurement on multi-qubit state")
    func customBasisMultiQubit() {
        let state = QuantumCircuit.bellPhiPlus().execute()

        let customBasis = [
            Complex(1 / sqrt(2), 0),
            Complex(1 / sqrt(2), 0),
        ]

        let result = Measurement.measure(0, basis: customBasis, in: state)

        #expect(result.outcome == 0 || result.outcome == 1, "Custom basis measurement on multi-qubit state should give 0 or 1")
        #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after multi-qubit custom basis measurement")
    }
}

/// Test suite for Pauli string measurements (tensor product observables).
/// Validates measurement of multi-qubit operators like X₀⊗Y₁⊗Z₂.
/// Critical for Hamiltonian expectation values in VQE and quantum chemistry.
@Suite("Multi-Qubit Pauli Measurements")
struct MultiQubitPauliMeasurementTests {
    @Test("Measure Z₀⊗Z₁ on Bell state gives +1")
    func measureZZOnBellState() {
        let bell = QuantumCircuit.bellPhiPlus().execute()
        let pauliString = PauliString(.z(0), .z(1))

        for _ in 0 ..< 10 {
            let result = Measurement.measure(pauliString, in: bell)
            #expect(result.eigenvalue == 1, "Z_0 x Z_1 on Bell Phi+ state should give eigenvalue +1")
            #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after Pauli string measurement")
            #expect(result.individualOutcomes.count == 2, "Z_0 x Z_1 should produce 2 individual outcomes")
        }
    }

    @Test("Measure X₀⊗X₁ on Bell state gives +1")
    func measureXXOnBellState() {
        let bell = QuantumCircuit.bellPhiPlus().execute()
        let pauliString = PauliString(.x(0), .x(1))

        for _ in 0 ..< 10 {
            let result = Measurement.measure(pauliString, in: bell)
            #expect(result.eigenvalue == 1, "X_0 x X_1 on Bell Phi+ state should give eigenvalue +1")
            #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after XX measurement")
        }
    }

    @Test("Measure single-qubit Pauli string")
    func measureSingleQubitPauliString() {
        let state = QuantumState(qubits: 1)
        let pauliString = PauliString(.z(0))

        let result = Measurement.measure(pauliString, in: state)

        #expect(result.eigenvalue == 1, "Single Z Pauli string on |0> should give eigenvalue +1")
        #expect(result.individualOutcomes.count == 1, "Single-qubit Pauli string should produce 1 individual outcome")
        #expect(result.individualOutcomes[0].qubit == 0, "Individual outcome should target qubit 0")
        #expect(result.individualOutcomes[0].outcome == 0, "Individual outcome for |0> in Z basis should be 0")
    }

    @Test("Measure empty Pauli string (identity)")
    func measureIdentityPauliString() {
        let state = QuantumState(qubits: 2)
        let pauliString = PauliString()

        let result = Measurement.measure(pauliString, in: state)

        #expect(result.eigenvalue == 1, "Identity Pauli string should give eigenvalue +1")
        #expect(result.individualOutcomes.isEmpty, "Identity Pauli string should produce no individual outcomes")
        #expect(result.collapsedState == state, "Identity measurement should not change the state")
    }

    @Test("Measure three-qubit Pauli string")
    func measureThreeQubitPauliString() {
        let ghz = QuantumCircuit.ghz(qubits: 3).execute()
        let pauliString = PauliString(.z(0), .z(1), .z(2))

        var outcomes = [Int]()
        for _ in 0 ..< 100 {
            let result = Measurement.measure(pauliString, in: ghz)
            outcomes.append(result.eigenvalue)
            #expect(result.eigenvalue == 1 || result.eigenvalue == -1, "ZZZ eigenvalue on GHZ state should be +1 or -1")
            #expect(result.individualOutcomes.count == 3, "Three-qubit Pauli string should produce 3 individual outcomes")
        }

        let countPlus = outcomes.count(where: { $0 == 1 })
        let countMinus = outcomes.count(where: { $0 == -1 })
        #expect(abs(Double(countPlus) - 50.0) < 20.0, "ZZZ +1 count should be approximately 50")
        #expect(abs(Double(countMinus) - 50.0) < 20.0, "ZZZ -1 count should be approximately 50")
    }

    @Test("Measure mixed Pauli string (X,Y,Z)")
    func measureMixedPauliString() {
        let state = QuantumCircuit.uniformSuperposition(qubits: 3).execute()
        let pauliString = PauliString(.x(0), .y(1), .z(2))

        let result = Measurement.measure(pauliString, in: state)

        #expect(result.eigenvalue == 1 || result.eigenvalue == -1, "Mixed XYZ Pauli string eigenvalue should be +1 or -1")
        #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after mixed Pauli string measurement")
        #expect(result.individualOutcomes.count == 3, "Mixed XYZ Pauli string should produce 3 individual outcomes")
    }

    @Test("PauliString description")
    func pauliStringDescription() {
        let pauliString = PauliString(.x(0), .z(1))

        #expect(pauliString.description.contains("X_0"), "PauliString description should contain X_0")
        #expect(pauliString.description.contains("Z_1"), "PauliString description should contain Z_1")
        #expect(pauliString.description.contains("⊗"), "PauliString description should contain tensor product symbol")
    }

    @Test("PauliString identity description")
    func pauliStringIdentityDescription() {
        let identity = PauliString()
        #expect(identity.description.contains("identity") || identity.description.contains("I"), "Empty PauliString description should indicate identity")
    }

    @Test("PauliStringMeasurementResult description")
    func pauliStringMeasurementResultDescription() {
        let state = QuantumState(qubits: 2)
        let pauliString = PauliString(.z(0))

        let result = Measurement.measure(pauliString, in: state)
        #expect(result.description.contains("PauliStringMeasurement"), "Description should contain 'PauliStringMeasurement' prefix")
        #expect(result.description.contains("+1") || result.description.contains("-1"), "Description should contain eigenvalue +1 or -1")
    }
}

/// Test suite for measuring multiple qubits simultaneously.
/// Validates joint probability distributions and partial state collapse
/// for arbitrary qubit subsets.
@Suite("Multiple Qubit Partial Measurements")
struct MultipleQubitPartialMeasurementTests {
    @Test("Measure multiple qubits on GHZ state")
    func measureMultipleQubitsGHZ() {
        let ghz = QuantumCircuit.ghz(qubits: 3).execute()

        let result = Measurement.measure([0, 1], in: ghz)

        #expect(result.outcomes.count == 2, "Measuring 2 qubits should produce 2 outcomes")
        #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after multi-qubit measurement")
        #expect((result.outcomes[0] == 0 && result.outcomes[1] == 0) || (result.outcomes[0] == 1 && result.outcomes[1] == 1), "GHZ state qubits 0 and 1 should be correlated")
    }

    @Test("Measure single qubit using measure(qubits:)")
    func measureSingleQubitUsingMeasureQubits() {
        let state = QuantumState(qubits: 2)

        let result = Measurement.measure(0, in: state)

        #expect(result.outcomes.count == 1, "Single qubit measurement should produce 1 outcome")
        #expect(result.outcomes[0] == 0, "Measuring qubit 0 of |00> should give outcome 0")
        #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized")
    }

    @Test("Measure all qubits")
    func measureAllQubits() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0),
            Complex(0, 0),
            Complex(0, 0),
            Complex(invSqrt2, 0),
        ])

        let result = Measurement.measure([0, 1], in: bell)

        #expect(result.outcomes.count == 2, "Measuring both qubits should produce 2 outcomes")
        #expect((result.outcomes == [0, 0]) || (result.outcomes == [1, 1]), "Bell state measurement should give correlated outcomes")
        #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after full measurement")
    }

    @Test("Measure non-adjacent qubits")
    func measureNonAdjacentQubits() {
        let state = QuantumCircuit.uniformSuperposition(qubits: 4).execute()

        let result = Measurement.measure([0, 2], in: state)

        #expect(result.outcomes.count == 2, "Measuring 2 non-adjacent qubits should produce 2 outcomes")
        #expect(result.outcomes[0] == 0 || result.outcomes[0] == 1, "Qubit 0 outcome should be 0 or 1")
        #expect(result.outcomes[1] == 0 || result.outcomes[1] == 1, "Qubit 2 outcome should be 0 or 1")
        #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after non-adjacent qubit measurement")
    }

    @Test("Measure qubits in arbitrary order")
    func measureQubitsArbitraryOrder() {
        let state = QuantumCircuit.ghz(qubits: 3).execute()

        let result = Measurement.measure([2, 0], in: state)

        #expect(result.outcomes.count == 2, "Measuring 2 qubits in arbitrary order should produce 2 outcomes")
        #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after arbitrary order measurement")
        #expect((result.outcomes[0] == 0 && result.outcomes[1] == 0) || (result.outcomes[0] == 1 && result.outcomes[1] == 1), "GHZ qubits 2 and 0 should be correlated")
    }

    @Test("Sequential measurements vs joint measurement")
    func sequentialVsJointMeasurement() {
        let state = QuantumCircuit.bellPhiPlus().execute()

        let jointResult = Measurement.measure([0, 1], in: state, seed: 42)
        let result0 = Measurement.measure(0, in: state, seed: 42)
        let result1 = Measurement.measure(1, in: result0.collapsedState, seed: 43)

        #expect(result0.outcome == result1.outcome, "Sequential measurements on Bell state should be correlated")
        #expect(jointResult.outcomes[0] == jointResult.outcomes[1], "Joint measurement on Bell state should give correlated outcomes")
    }

    @Test("Measure product state preserves independence")
    func measureProductStateIndependence() {
        let product = QuantumState(qubits: 3, amplitudes: [
            Complex(0, 0),
            Complex(0, 0),
            Complex(0, 0),
            Complex(0, 0),
            Complex(1 / sqrt(2), 0),
            Complex(1 / sqrt(2), 0),
            Complex(0, 0),
            Complex(0, 0),
        ])

        let result = Measurement.measure([1, 2], in: product)

        #expect(result.outcomes[0] == 0, "Qubit 1 of product state should measure to 0")
        #expect(result.outcomes[1] == 1, "Qubit 2 of product state should measure to 1")
        #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after product state measurement")

        let prob4 = result.collapsedState.probability(of: 4)
        let prob5 = result.collapsedState.probability(of: 5)
        #expect(abs(prob4 - 0.5) < 1e-10, "State 4 probability should be 0.5")
        #expect(abs(prob5 - 0.5) < 1e-10, "State 5 probability should be 0.5")
    }
}

/// Test suite for non-destructive state capture (snapshots).
/// Validates that snapshots preserve quantum coherence without measurement collapse,
/// including label handling and timestamp generation.
@Suite("Statevector Snapshots")
struct StatevectorSnapshotTests {
    @Test("Capture snapshot preserves state")
    func captureSnapshotPreservesState() {
        let bell = QuantumCircuit.bellPhiPlus().execute()
        let snapshot = Measurement.snapshot(of: bell, label: "Bell state")

        #expect(snapshot.state == bell, "Snapshot state should equal the original Bell state")
        #expect(snapshot.state.isNormalized(), "Snapshot state should be normalized")
        #expect(snapshot.label == "Bell state", "Snapshot label should be 'Bell state'")
    }

    @Test("Snapshot without label")
    func snapshotWithoutLabel() {
        let state = QuantumState(qubits: 1)
        let snapshot = Measurement.snapshot(of: state)

        #expect(snapshot.state == state, "Snapshot state should equal the original state")
        #expect(snapshot.label == nil, "Snapshot without label should have nil label")
    }

    @Test("Multiple snapshots preserve different states")
    func multipleSnapshotsPreserveDifferentStates() {
        var circuit = QuantumCircuit(qubits: 2)

        let snapshot1 = Measurement.snapshot(
            of: circuit.execute(),
            label: "Initial",
        )

        circuit.append(.hadamard, to: 0)
        let snapshot2 = Measurement.snapshot(
            of: circuit.execute(),
            label: "After H",
        )

        circuit.append(.cnot, to: [0, 1])
        let snapshot3 = Measurement.snapshot(
            of: circuit.execute(),
            label: "Bell state",
        )

        #expect(snapshot1.state != snapshot2.state, "Initial and post-H snapshots should differ")
        #expect(snapshot2.state != snapshot3.state, "Post-H and Bell state snapshots should differ")
        #expect(snapshot1.state != snapshot3.state, "Initial and Bell state snapshots should differ")
    }

    @Test("Snapshot timestamp is valid")
    func snapshotTimestampValid() {
        let state = QuantumState(qubits: 1)
        let before = Date()
        let snapshot = Measurement.snapshot(of: state)
        let after = Date()

        #expect(snapshot.timestamp >= before, "Snapshot timestamp should be at or after the before time")
        #expect(snapshot.timestamp <= after, "Snapshot timestamp should be at or before the after time")
    }

    @Test("Snapshot description with label")
    func snapshotDescriptionWithLabel() {
        let state = QuantumState(qubits: 1)
        let snapshot = Measurement.snapshot(of: state, label: "Test")

        #expect(snapshot.description.contains("Test"), "Snapshot description should contain the label 'Test'")
        #expect(snapshot.description.contains("Snapshot"), "Snapshot description should contain 'Snapshot' prefix")
    }

    @Test("Snapshot description without label")
    func snapshotDescriptionWithoutLabel() {
        let state = QuantumState(qubits: 1)
        let snapshot = Measurement.snapshot(of: state)

        #expect(snapshot.description.contains("Snapshot"), "Snapshot description should contain 'Snapshot' prefix")
    }

    @Test("Snapshot preserves complex states")
    func snapshotPreservesComplexStates() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let complexState = QuantumState(qubits: 2, amplitudes: [
            Complex(0.5, 0.5),
            Complex(0, 0),
            Complex(0, 0),
            Complex(invSqrt2, 0),
        ])

        let snapshot = Measurement.snapshot(of: complexState, label: "Complex")

        #expect(snapshot.state.amplitude(of: 0) == Complex(0.5, 0.5), "Snapshot should preserve complex amplitude at index 0")
        #expect(snapshot.state.amplitude(of: 3).real == invSqrt2, "Snapshot should preserve real amplitude at index 3")
    }

    @Test("Snapshots are independent")
    func snapshotsAreIndependent() {
        let state = QuantumCircuit.bellPhiPlus().execute()
        let snapshot1 = Measurement.snapshot(of: state, label: "First")
        let snapshot2 = Measurement.snapshot(of: state, label: "Second")

        #expect(snapshot1.state == snapshot2.state, "Snapshots of the same state should have equal states")
        #expect(snapshot1.label != snapshot2.label, "Snapshots with different labels should have different labels")
    }

    @Test("Snapshot timestamp ordering")
    func snapshotTimestampOrdering() {
        let state = QuantumState(qubits: 1)
        let snapshot1 = Measurement.snapshot(of: state, label: "First")

        Thread.sleep(forTimeInterval: 0.01)

        let snapshot2 = Measurement.snapshot(of: state, label: "Second")

        #expect(snapshot2.timestamp > snapshot1.timestamp, "Later snapshot should have a later timestamp")
    }
}

/// Test suite for PartialMeasurementResult description formatting.
/// Validates human-readable output for both single and multiple qubit measurements,
/// ensuring correct format selection based on outcome count.
@Suite("PartialMeasurementResult Description")
struct PartialMeasurementResultDescriptionTests {
    @Test("Single outcome uses singular format")
    func singleOutcomeSingularFormat() {
        let state = QuantumState(qubits: 2)
        let result = Measurement.measure(0, in: state)

        let description = result.description
        #expect(description.contains("PartialMeasurement"), "Description should contain 'PartialMeasurement' prefix")
        #expect(description.contains("outcome="), "Description should contain 'outcome=' field")
        #expect(!description.contains("outcomes="), "Single outcome should use 'outcome=' not 'outcomes='")
    }

    @Test("Multiple outcomes uses plural format")
    func multipleOutcomesPluralFormat() {
        let state = QuantumState(qubits: 3)
        let result = Measurement.measure([0, 1], in: state)

        let description = result.description
        #expect(description.contains("PartialMeasurement"), "Description should contain 'PartialMeasurement' prefix")
        #expect(description.contains("outcomes="), "Multiple outcomes should use 'outcomes='")
        #expect(description.contains("["), "Multiple outcomes should show array format")
    }

    @Test("Single outcome description includes state info")
    func singleOutcomeIncludesState() {
        let state = QuantumState(qubits: 2)
        let result = Measurement.measure(0, in: state)

        #expect(result.description.contains("state="), "Single outcome description should contain state info")
    }

    @Test("Multiple outcomes description includes state info")
    func multipleOutcomesIncludesState() {
        let ghz = QuantumCircuit.ghz(qubits: 3).execute()
        let result = Measurement.measure([0, 1, 2], in: ghz)

        #expect(result.description.contains("state="), "Multiple outcomes description should contain state info")
        #expect(result.description.contains("outcomes="), "Multiple outcomes description should contain 'outcomes=' field")
    }
}

/// Test suite for CustomBasisMeasurementResult description formatting.
/// Validates human-readable output for custom basis measurement results,
/// ensuring outcome and state information is correctly displayed.
@Suite("CustomBasisMeasurementResult Description")
struct CustomBasisMeasurementResultDescriptionTests {
    @Test("Description includes CustomBasisMeasurement prefix")
    func descriptionIncludesPrefix() {
        let state = QuantumState(qubits: 1)
        let customBasis = [
            Complex(1 / sqrt(2.0), 0),
            Complex(1 / sqrt(2.0), 0),
        ]

        let result = Measurement.measure(0, basis: customBasis, in: state)

        #expect(result.description.contains("CustomBasisMeasurement"), "Description should contain 'CustomBasisMeasurement' prefix")
    }

    @Test("Description includes outcome value")
    func descriptionIncludesOutcome() {
        let state = QuantumState(qubits: 1)
        let customBasis = [
            Complex(1 / sqrt(2.0), 0),
            Complex(1 / sqrt(2.0), 0),
        ]

        let result = Measurement.measure(0, basis: customBasis, in: state)

        #expect(result.description.contains("outcome="), "Description should contain 'outcome=' field")
        #expect(result.description.contains("0") || result.description.contains("1"), "Description should contain the outcome value")
    }

    @Test("Description includes state info")
    func descriptionIncludesState() {
        let state = QuantumState(qubits: 1)
        let customBasis = [
            Complex(1.0, 0),
            Complex(0.0, 0),
        ]

        let result = Measurement.measure(0, basis: customBasis, in: state)

        #expect(result.description.contains("state="), "Description should contain state info")
    }

    @Test("Description format is consistent")
    func descriptionFormatConsistent() {
        let state = QuantumState(qubits: 2)
        let customBasis = [
            Complex(0.6, 0),
            Complex(0.8, 0),
        ]

        let result = Measurement.measure(0, basis: customBasis, in: state)

        let description = result.description
        #expect(description.hasPrefix("CustomBasisMeasurement:"), "Description should start with 'CustomBasisMeasurement:'")
    }
}

/// Test suite for auto-sizing histogram function.
/// Validates histogram(outcomes:) that infers size from max outcome,
/// including empty input and negative value handling.
@Suite("Auto-Sizing Histogram")
struct AutoSizingHistogramTests {
    @Test("Empty outcomes returns empty histogram")
    func emptyOutcomesReturnsEmpty() {
        let outcomes: [Int] = []
        let histogram = Measurement.histogram(outcomes: outcomes)

        #expect(histogram.isEmpty, "Empty outcomes should produce an empty histogram")
    }

    @Test("Single outcome creates histogram of size max+1")
    func singleOutcomeCreatesCorrectSize() {
        let outcomes = [5]
        let histogram = Measurement.histogram(outcomes: outcomes)

        #expect(histogram.count == 6, "Histogram for outcome 5 should have 6 bins")
        #expect(histogram[5] == 1, "Outcome 5 should appear once")
        for i in 0 ..< 5 {
            #expect(histogram[i] == 0, "Outcome \(i) should have zero count")
        }
    }

    @Test("Histogram counts outcomes correctly")
    func histogramCountsCorrectly() {
        let outcomes = [0, 1, 1, 0, 2, 1, 0, 3]
        let histogram = Measurement.histogram(outcomes: outcomes)

        #expect(histogram.count == 4, "Auto-sized histogram should have 4 bins")
        #expect(histogram[0] == 3, "Outcome 0 should appear 3 times in auto-sized histogram")
        #expect(histogram[1] == 3, "Outcome 1 should appear 3 times in auto-sized histogram")
        #expect(histogram[2] == 1, "Outcome 2 should appear 1 time in auto-sized histogram")
        #expect(histogram[3] == 1, "Outcome 3 should appear 1 time in auto-sized histogram")
    }

    @Test("Negative outcomes are ignored")
    func negativeOutcomesIgnored() {
        let outcomes = [0, -1, 1, -5, 2]
        let histogram = Measurement.histogram(outcomes: outcomes)

        #expect(histogram.count == 3, "Histogram should have 3 bins ignoring negatives")
        #expect(histogram[0] == 1, "Outcome 0 should appear once with negatives ignored")
        #expect(histogram[1] == 1, "Outcome 1 should appear once with negatives ignored")
        #expect(histogram[2] == 1, "Outcome 2 should appear once with negatives ignored")
    }

    @Test("All negative outcomes returns empty")
    func allNegativeOutcomesReturnsEmpty() {
        let outcomes = [-1, -2, -3]
        let histogram = Measurement.histogram(outcomes: outcomes)

        #expect(histogram.isEmpty, "All-negative outcomes should produce an empty histogram")
    }

    @Test("Sparse outcomes create correct histogram")
    func sparseOutcomesCorrectHistogram() {
        let outcomes = [0, 10, 0, 10]
        let histogram = Measurement.histogram(outcomes: outcomes)

        #expect(histogram.count == 11, "Sparse histogram for outcomes 0 and 10 should have 11 bins")
        #expect(histogram[0] == 2, "Outcome 0 should appear twice in sparse histogram")
        #expect(histogram[10] == 2, "Outcome 10 should appear twice in sparse histogram")
        for i in 1 ..< 10 {
            #expect(histogram[i] == 0, "Outcome \(i) should have zero count in sparse histogram")
        }
    }

    @Test("Large gap in outcomes handled correctly")
    func largeGapHandled() {
        let outcomes = [0, 100]
        let histogram = Measurement.histogram(outcomes: outcomes)

        #expect(histogram.count == 101, "Histogram for outcomes 0 and 100 should have 101 bins")
        #expect(histogram[0] == 1, "Outcome 0 should appear once with large gap")
        #expect(histogram[100] == 1, "Outcome 100 should appear once with large gap")
    }

    @Test("Mixed positive and negative with zero max")
    func mixedWithZeroMax() {
        let outcomes = [-5, -1, 0, -3]
        let histogram = Measurement.histogram(outcomes: outcomes)

        #expect(histogram.count == 1, "Histogram with only non-negative value 0 should have 1 bin")
        #expect(histogram[0] == 1, "Outcome 0 should appear once in mixed-sign outcomes")
    }
}

/// Test suite for edge cases in new measurement infrastructure.
/// Validates boundary conditions and error handling
/// for Pauli strings, partial measurements, and custom bases.
@Suite("Measurement Infrastructure Edge Cases")
struct MeasurementInfrastructureEdgeCasesTests {
    @Test("Pauli measurement on multi-qubit state affects only target qubit")
    func pauliMeasurementOnlyAffectsTarget() {
        let state = QuantumCircuit.uniformSuperposition(qubits: 3).execute()

        let result = Measurement.measure(1, basis: .x, in: state)

        #expect(result.eigenvalue == 1 || result.eigenvalue == -1, "Pauli X measurement eigenvalue should be +1 or -1")
        #expect(result.collapsedState.isNormalized(), "Collapsed state should be normalized after Pauli measurement on multi-qubit state")
        #expect(result.collapsedState.qubits == 3, "Collapsed state should preserve 3-qubit system size")
    }

    @Test("Measure single qubit from list")
    func measureSingleQubitFromList() {
        let state = QuantumState(qubits: 2)

        let result = Measurement.measure(0, in: state)
        #expect(result.outcomes.count == 1, "Single qubit from list should produce 1 outcome")
    }

    @Test("Pauli measurement preserves normalization across all bases")
    func pauliMeasurementNormalizationAllBases() {
        let state = QuantumCircuit.bellPhiPlus().execute()

        for basis in PauliBasis.allCases {
            let result = Measurement.measure(0, basis: basis, in: state)
            #expect(result.collapsedState.isNormalized(), "Pauli measurement in \(basis) basis should preserve normalization")
        }
    }

    @Test("CustomBasisMeasurement description shows negative eigenvalue correctly")
    func customBasisMeasurementDescriptionNegative() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        let state = circuit.execute()

        let result = Measurement.measure(0, basis: .z, in: state)

        #expect(result.eigenvalue == -1, "Measuring |1> in Z basis should give eigenvalue -1")
        let description = result.description
        #expect(description.contains("-1"), "Description should show negative eigenvalue")
    }

    @Test("PauliStringMeasurement description shows negative eigenvalue correctly")
    func pauliStringMeasurementDescriptionNegative() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.pauliX, to: 0)
        let state = circuit.execute()

        let pauliString = PauliString(.z(0), .z(1))
        let result = Measurement.measure(pauliString, in: state)

        #expect(result.eigenvalue == -1, "Z_0 x Z_1 with qubit 0 flipped should give eigenvalue -1")
        let description = result.description
        #expect(description.contains("-1"), "PauliString description should show negative eigenvalue")
    }
}
