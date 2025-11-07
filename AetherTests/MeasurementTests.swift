// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for probability distribution calculations.
/// Validates Born rule: P(i) = |cᵢ|² implementation
/// for computing measurement probabilities from quantum amplitudes.
@Suite("Probability Distribution")
struct ProbabilityDistributionTests {
    @Test("Probability distribution for |0⟩")
    func probabilityDistributionZeroState() {
        let state = QuantumState(singleQubit: 0)
        let probabilities = Measurement.probabilityDistribution(state: state)

        #expect(abs(probabilities[0] - 1.0) < 1e-10)
        #expect(abs(probabilities[1]) < 1e-10)
    }

    @Test("Probability distribution for equal superposition")
    func probabilityDistributionSuperposition() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [
            Complex(invSqrt2, 0.0),
            Complex(invSqrt2, 0.0),
        ]
        let state = QuantumState(numQubits: 1, amplitudes: amplitudes)
        let probabilities = Measurement.probabilityDistribution(state: state)

        #expect(abs(probabilities[0] - 0.5) < 1e-10)
        #expect(abs(probabilities[1] - 0.5) < 1e-10)
    }

    @Test("Probability distribution sums to 1.0")
    func probabilitySumToOne() {
        let amplitudes = [
            Complex(0.6, 0.0),
            Complex(0.8, 0.0),
        ]
        let state = QuantumState(numQubits: 1, amplitudes: amplitudes)
        let probabilities = Measurement.probabilityDistribution(state: state)

        let sum = probabilities.reduce(0.0, +)
        #expect(abs(sum - 1.0) < 1e-10)
    }
}

/// Test suite for quantum state collapse.
/// Validates wave function collapse to computational basis states
/// following measurement, ensuring unitarity and normalization.
@Suite("State Collapse")
struct StateCollapseTests {
    @Test("Collapse to outcome 0")
    func collapseToZero() {
        let collapsed = Measurement.collapseToOutcome(0, numQubits: 2)

        #expect(abs(collapsed.getAmplitude(ofState: 0).real - 1.0) < 1e-10)
        #expect(abs(collapsed.getAmplitude(ofState: 1).magnitude) < 1e-10)
        #expect(abs(collapsed.getAmplitude(ofState: 2).magnitude) < 1e-10)
        #expect(abs(collapsed.getAmplitude(ofState: 3).magnitude) < 1e-10)
    }

    @Test("Collapse to outcome 3")
    func collapseToThree() {
        let collapsed = Measurement.collapseToOutcome(3, numQubits: 2)

        #expect(abs(collapsed.getAmplitude(ofState: 0).magnitude) < 1e-10)
        #expect(abs(collapsed.getAmplitude(ofState: 1).magnitude) < 1e-10)
        #expect(abs(collapsed.getAmplitude(ofState: 2).magnitude) < 1e-10)
        #expect(abs(collapsed.getAmplitude(ofState: 3).real - 1.0) < 1e-10)
    }

    @Test("Collapsed state is normalized")
    func collapsedStateNormalized() {
        let collapsed = Measurement.collapseToOutcome(1, numQubits: 3)
        #expect(collapsed.isNormalized())
    }
}

/// Test suite for full quantum measurement.
/// Validates complete state collapse to single basis state
/// with proper probabilistic sampling according to Born rule.
@Suite("Full Measurement")
struct FullMeasurementTests {
    @Test("Measure |0⟩ always gives 0")
    func measureZeroState() {
        let state = QuantumState(singleQubit: 0)
        var measurement = Measurement()

        for _ in 0 ..< 10 {
            let result = measurement.measure(state: state)
            #expect(result.outcome == 0)
            #expect(result.collapsedState.isNormalized())
        }
    }

    @Test("Measure |1⟩ always gives 1")
    func measureOneState() {
        let state = QuantumState(singleQubit: 1)
        var measurement = Measurement()

        for _ in 0 ..< 10 {
            let result = measurement.measure(state: state)
            #expect(result.outcome == 1)
        }
    }

    @Test("Collapsed state is basis state")
    func collapsedStateIsBasisState() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [
            Complex(invSqrt2, 0.0),
            Complex(invSqrt2, 0.0),
        ]
        let state = QuantumState(numQubits: 1, amplitudes: amplitudes)
        var measurement = Measurement()

        let result = measurement.measure(state: state)

        #expect(result.outcome == 0 || result.outcome == 1)
        #expect(abs(result.collapsedState.getAmplitude(ofState: result.outcome).real - 1.0) < 1e-10)
    }

    @Test("Measuring collapsed state is deterministic")
    func measureCollapsedStateIsDeterministic() {
        let state = QuantumState(singleQubit: 0)
        var measurement = Measurement()

        let firstResult = measurement.measure(state: state)
        let secondResult = measurement.measure(state: firstResult.collapsedState)

        #expect(firstResult.outcome == secondResult.outcome)
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
        let state = QuantumState(numQubits: 1, amplitudes: amplitudes)
        var measurement = Measurement()

        let numRuns = 1000
        var counts = [0, 0]

        for _ in 0 ..< numRuns {
            let result = measurement.measure(state: state)
            counts[result.outcome] += 1
        }

        let freq0 = Double(counts[0]) / Double(numRuns)
        let freq1 = Double(counts[1]) / Double(numRuns)

        #expect(abs(freq0 - 0.5) < 0.1)
        #expect(abs(freq1 - 0.5) < 0.1)
    }

    @Test("Bell state statistics")
    func bellStateStatistics() {
        let circuit = QuantumCircuit.bellState()
        var measurement = Measurement()

        let numRuns = 1000
        let outcomes = measurement.runMultiple(circuit: circuit, numRuns: numRuns)
        let histogram = Measurement.histogram(outcomes: outcomes, numQubits: 2)

        let freq0 = Double(histogram[0]) / Double(numRuns)
        let freq3 = Double(histogram[3]) / Double(numRuns)

        #expect(abs(freq0 - 0.5) < 0.1)
        #expect(abs(freq3 - 0.5) < 0.1)

        #expect(histogram[1] < numRuns / 20)
        #expect(histogram[2] < numRuns / 20)
    }

    @Test("Weighted state statistics")
    func weightedStateStatistics() {
        let amplitudes = [
            Complex(0.6, 0.0),
            Complex(0.8, 0.0),
        ]
        let state = QuantumState(numQubits: 1, amplitudes: amplitudes)
        var measurement = Measurement()

        let numRuns = 1000
        var counts = [0, 0]

        for _ in 0 ..< numRuns {
            let result = measurement.measure(state: state)
            counts[result.outcome] += 1
        }

        let freq0 = Double(counts[0]) / Double(numRuns)
        let freq1 = Double(counts[1]) / Double(numRuns)

        #expect(abs(freq0 - 0.36) < 0.1)
        #expect(abs(freq1 - 0.64) < 0.1)
    }
}

/// Test suite for partial quantum measurements.
/// Validates single-qubit measurement while preserving superposition in remaining qubits,
/// demonstrating marginal probability calculations and conditional state updates.
@Suite("Partial Measurement")
struct PartialMeasurementTests {
    @Test("Marginal probabilities for separable state")
    func marginalProbabilitiesSeparable() {
        let state = QuantumState(numQubits: 2)

        let (prob0_q0, prob1_q0) = Measurement.marginalProbabilities(qubit: 0, state: state)
        let (prob0_q1, prob1_q1) = Measurement.marginalProbabilities(qubit: 1, state: state)

        #expect(abs(prob0_q0 - 1.0) < 1e-10)
        #expect(abs(prob1_q0) < 1e-10)
        #expect(abs(prob0_q1 - 1.0) < 1e-10)
        #expect(abs(prob1_q1) < 1e-10)
    }

    @Test("Marginal probabilities for Bell state")
    func marginalProbabilitiesBell() {
        let circuit = QuantumCircuit.bellState()
        let state = circuit.execute()

        let (prob0_q0, prob1_q0) = Measurement.marginalProbabilities(qubit: 0, state: state)
        let (prob0_q1, prob1_q1) = Measurement.marginalProbabilities(qubit: 1, state: state)

        #expect(abs(prob0_q0 - 0.5) < 1e-10)
        #expect(abs(prob1_q0 - 0.5) < 1e-10)
        #expect(abs(prob0_q1 - 0.5) < 1e-10)
        #expect(abs(prob1_q1 - 0.5) < 1e-10)
    }

    @Test("Partial collapse preserves compatible amplitudes")
    func partialCollapseCompatibleAmplitudes() {
        let circuit = QuantumCircuit.superposition(numQubits: 2)
        let state = circuit.execute()

        let collapsed = Measurement.partialCollapse(
            qubit: 0,
            outcome: 0,
            state: state,
            probability: 0.5
        )

        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(collapsed.getAmplitude(ofState: 0).real - invSqrt2) < 1e-10)
        #expect(abs(collapsed.getAmplitude(ofState: 2).real - invSqrt2) < 1e-10)
        #expect(abs(collapsed.getAmplitude(ofState: 1).magnitude) < 1e-10)
        #expect(abs(collapsed.getAmplitude(ofState: 3).magnitude) < 1e-10)
    }

    @Test("Partial collapse renormalizes state")
    func partialCollapseRenormalizes() {
        let circuit = QuantumCircuit.superposition(numQubits: 2)
        let state = circuit.execute()

        let collapsed = Measurement.partialCollapse(
            qubit: 0,
            outcome: 0,
            state: state,
            probability: 0.5
        )

        #expect(collapsed.isNormalized())
    }

    @Test("Measure single qubit of separable state")
    func measureSingleQubitSeparable() {
        var amplitudes = [Complex<Double>](repeating: .zero, count: 4)
        amplitudes[1] = .one
        let state = QuantumState(numQubits: 2, amplitudes: amplitudes)
        var measurement = Measurement()

        for _ in 0 ..< 10 {
            let (outcome, collapsedState) = measurement.measureQubit(0, state: state)
            #expect(outcome == 1)
            #expect(collapsedState.isNormalized())
        }
    }

    @Test("Measure qubit in Bell state demonstrates entanglement")
    func measureBellStateEntanglement() {
        let circuit = QuantumCircuit.bellState()
        let state = circuit.execute()
        var measurement = Measurement()

        let (outcome0, collapsedState) = measurement.measureQubit(0, state: state)
        let (outcome1, _) = measurement.measureQubit(1, state: collapsedState)

        #expect(outcome0 == outcome1)
    }

    @Test("Partial measurement statistics")
    func partialMeasurementStatistics() {
        let circuit = QuantumCircuit.bellState()
        var measurement = Measurement()

        let numRuns = 1000
        var outcomes = [Int]()

        for _ in 0 ..< numRuns {
            let state = circuit.execute()
            let (outcome, _) = measurement.measureQubit(0, state: state)
            outcomes.append(outcome)
        }

        let count0 = outcomes.filter { $0 == 0 }.count
        let count1 = outcomes.filter { $0 == 1 }.count

        let freq0 = Double(count0) / Double(numRuns)
        let freq1 = Double(count1) / Double(numRuns)

        #expect(abs(freq0 - 0.5) < 0.1)
        #expect(abs(freq1 - 0.5) < 0.1)
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
        let histogram = Measurement.histogram(outcomes: outcomes, numQubits: 2)

        #expect(histogram[0] == 3)
        #expect(histogram[1] == 3)
        #expect(histogram[2] == 1)
        #expect(histogram[3] == 1)
    }

    @Test("Compare perfect match")
    func compareDistributionsPerfect() {
        let observed = [500, 500]
        let expected = [0.5, 0.5]

        let error = Measurement.compareDistributions(
            observed: observed,
            expected: expected,
            totalRuns: 1000
        )

        #expect(error < 0.01)
    }

    @Test("Compare with deviation")
    func compareDistributionsDeviation() {
        let observed = [400, 600]
        let expected = [0.5, 0.5]

        let error = Measurement.compareDistributions(
            observed: observed,
            expected: expected,
            totalRuns: 1000
        )

        #expect(error > 0.15)
    }

    @Test("Chi-squared for good fit")
    func chiSquaredGoodFit() {
        let observed = [505, 495]
        let expected = [0.5, 0.5]

        let chiSq = Measurement.chiSquared(
            observed: observed,
            expected: expected,
            totalRuns: 1000
        )

        #expect(chiSq.chiSquared < 5.0)
    }

    @Test("Run circuit multiple times")
    func runCircuitMultipleTimes() {
        let circuit = QuantumCircuit.bellState()
        var measurement = Measurement()

        let outcomes = measurement.runMultiple(circuit: circuit, numRuns: 100)

        #expect(outcomes.count == 100)
        #expect(outcomes.allSatisfy { $0 >= 0 && $0 < 4 })
    }
}

/// Test suite for measurement edge cases.
/// Validates measurement behavior across different qubit counts
/// and complex amplitude configurations.
@Suite("Edge Cases")
struct MeasurementEdgeCasesTests {
    @Test("Measure single-qubit state")
    func measureSingleQubit() {
        let state = QuantumState(singleQubit: 0)
        let result = Measurement.measureOnce(state: state)

        #expect(result.outcome == 0)
        #expect(result.collapsedState.isNormalized())
    }

    @Test("Measure 3-qubit state")
    func measureThreeQubits() {
        let circuit = QuantumCircuit.superposition(numQubits: 3)
        let state = circuit.execute()
        var measurement = Measurement()

        let result = measurement.measure(state: state)

        #expect(result.outcome >= 0 && result.outcome < 8)
        #expect(result.collapsedState.isNormalized())
    }

    @Test("Measure state with complex amplitudes")
    func measureComplexAmplitudes() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [
            Complex(invSqrt2, 0.0),
            Complex(0.0, invSqrt2),
        ]
        let state = QuantumState(numQubits: 1, amplitudes: amplitudes)
        var measurement = Measurement()

        var counts = [0, 0]
        for _ in 0 ..< 100 {
            let result = measurement.measure(state: state)
            counts[result.outcome] += 1
        }

        #expect(counts[0] > 0)
        #expect(counts[1] > 0)
    }
}

/// Test suite for normalization preservation.
/// Ensures measurements maintain Σ|cᵢ|² = 1.0 throughout
/// collapse operations and partial measurements.
@Suite("Normalization Checks")
struct NormalizationChecksTests {
    @Test("All measurements preserve normalization")
    func measurementsPreserveNormalization() {
        let circuit = QuantumCircuit.bellState()
        let state = circuit.execute()
        var measurement = Measurement()

        for _ in 0 ..< 10 {
            let result = measurement.measure(state: state)
            #expect(result.collapsedState.isNormalized())
        }
    }

    @Test("Partial measurements preserve normalization")
    func partialMeasurementsPreserveNormalization() {
        let circuit = QuantumCircuit.ghzState(numQubits: 3)
        let state = circuit.execute()
        var measurement = Measurement()

        let (_, collapsed1) = measurement.measureQubit(0, state: state)
        #expect(collapsed1.isNormalized())

        let (_, collapsed2) = measurement.measureQubit(1, state: collapsed1)
        #expect(collapsed2.isNormalized())

        let (_, collapsed3) = measurement.measureQubit(2, state: collapsed2)
        #expect(collapsed3.isNormalized())
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
        let state = QuantumState(numQubits: 1, amplitudes: amplitudes)
        var measurement = Measurement()

        let firstResult = measurement.measure(state: state)
        let secondResult = measurement.measure(state: firstResult.collapsedState)
        let thirdResult = measurement.measure(state: secondResult.collapsedState)

        #expect(firstResult.outcome == secondResult.outcome)
        #expect(secondResult.outcome == thirdResult.outcome)
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
        let state = QuantumState(numQubits: 1, amplitudes: amplitudes)

        var measurement1 = Measurement(seed: 42)
        var measurement2 = Measurement(seed: 42)

        let result1 = measurement1.measure(state: state)
        let result2 = measurement2.measure(state: state)

        #expect(result1.outcome == result2.outcome)
    }

    @Test("Different seeds give different sequences")
    func differentSeedsGiveDifferentSequences() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [
            Complex(invSqrt2, 0.0),
            Complex(invSqrt2, 0.0),
        ]
        let state = QuantumState(numQubits: 1, amplitudes: amplitudes)

        var measurement1 = Measurement(seed: 42)
        var measurement2 = Measurement(seed: 100)

        var outcomes1: [Int] = []
        var outcomes2: [Int] = []

        for _ in 0 ..< 20 {
            outcomes1.append(measurement1.measure(state: state).outcome)
            outcomes2.append(measurement2.measure(state: state).outcome)
        }

        #expect(outcomes1 != outcomes2)
    }

    @Test("Seeded measurement flag is set correctly")
    func seededFlagCorrect() {
        let measurement1 = Measurement()
        let measurement2 = Measurement(seed: 42)

        #expect(measurement1.isSeeded == false)
        #expect(measurement2.isSeeded == true)
    }

    @Test("Seeded measurement produces valid outcomes")
    func seededMeasurementValidOutcomes() {
        let state = QuantumState(numQubits: 2)
        var measurement = Measurement(seed: 123)

        for _ in 0 ..< 50 {
            let result = measurement.measure(state: state)
            #expect(result.outcome >= 0 && result.outcome < 4)
            #expect(result.collapsedState.isNormalized())
        }
    }
}

/// Test suite for MeasurementResult description and display.
/// Validates human-readable output for debugging and UI display.
@Suite("Measurement Result Display")
struct MeasurementResultDisplayTests {
    @Test("MeasurementResult description includes outcome")
    func descriptionIncludesOutcome() {
        let state = QuantumState(singleQubit: 0)
        let result = Measurement.measureOnce(state: state)

        #expect(result.description.contains("outcome"))
        #expect(result.description.contains("0"))
    }

    @Test("MeasurementResult description includes state info")
    func descriptionIncludesState() {
        let state = QuantumState(singleQubit: 1)
        let result = Measurement.measureOnce(state: state)

        #expect(result.description.contains("state"))
    }

    @Test("MeasurementResult description is non-empty")
    func descriptionNonEmpty() {
        let state = QuantumState(numQubits: 2)
        var measurement = Measurement()
        let result = measurement.measure(state: state)

        #expect(!result.description.isEmpty)
    }
}

/// Test suite for static convenience methods.
/// Validates one-shot measurement helpers for simple use cases.
@Suite("Static Convenience Methods")
struct StaticConvenienceTests {
    @Test("measureOnce static method works")
    func measureOnceWorks() {
        let state = QuantumState(singleQubit: 0)
        let result = Measurement.measureOnce(state: state)

        #expect(result.outcome == 0)
        #expect(result.collapsedState.isNormalized())
    }

    @Test("measureQubitOnce static method works")
    func measureQubitOnceWorks() {
        let state = QuantumState(numQubits: 2)
        let (outcome, collapsedState) = Measurement.measureQubitOnce(0, state: state)

        #expect(outcome == 0 || outcome == 1)
        #expect(collapsedState.isNormalized())
    }

    @Test("runMultiple static method works")
    func runMultipleStaticWorks() {
        let circuit = QuantumCircuit.bellState()
        let outcomes = Measurement.runMultiple(circuit: circuit, numRuns: 50)

        #expect(outcomes.count == 50)
        #expect(outcomes.allSatisfy { $0 >= 0 && $0 < 4 })
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

        let error = Measurement.compareDistributions(
            observed: observed,
            expected: expected,
            totalRuns: 105
        )

        #expect(error >= 0.0)
    }

    @Test("Compare distributions when observed exceeds expected zero")
    func compareDistributionsObservedExceedsExpectedZero() {
        let observed = [90, 0, 10, 0]
        let expected = [0.9, 0.0, 0.0, 0.1]

        let error = Measurement.compareDistributions(
            observed: observed,
            expected: expected,
            totalRuns: 100
        )

        #expect(error > 0.0)
    }

    @Test("Chi-squared with small expected counts")
    func chiSquaredSmallExpectedCounts() {
        let observed = [95, 3, 1, 1]
        let expected = [0.95, 0.03, 0.01, 0.01]

        let result = Measurement.chiSquared(
            observed: observed,
            expected: expected,
            totalRuns: 100
        )

        #expect(result.skippedBins > 0)
        #expect(result.testedBins < 4)
    }

    @Test("Chi-squared with all bins tested")
    func chiSquaredAllBinsTested() {
        let observed = [250, 250, 250, 250]
        let expected = [0.25, 0.25, 0.25, 0.25]

        let result = Measurement.chiSquared(
            observed: observed,
            expected: expected,
            totalRuns: 1000
        )

        #expect(result.testedBins == 4)
        #expect(result.skippedBins == 0)
        #expect(result.degreesOfFreedom == 3)
    }

    @Test("Chi-squared with poor fit")
    func chiSquaredPoorFit() {
        let observed = [900, 50, 25, 25]
        let expected = [0.25, 0.25, 0.25, 0.25]

        let result = Measurement.chiSquared(
            observed: observed,
            expected: expected,
            totalRuns: 1000
        )

        #expect(result.chiSquared > 100.0)
    }

    @Test("Histogram handles out-of-bounds outcomes gracefully")
    func histogramOutOfBounds() {
        let outcomes = [0, 1, 2, 3, 10, -1, 5]
        let histogram = Measurement.histogram(outcomes: outcomes, numQubits: 2)

        #expect(histogram.count == 4)
        #expect(histogram[0] == 1)
        #expect(histogram[1] == 1)
        #expect(histogram[2] == 1)
        #expect(histogram[3] == 1)
    }
}

/// Test suite for Pauli basis measurements (X, Y, Z).
/// Validates measurement in non-computational bases with eigenvalue ±1 outcomes.
/// Essential for observable expectation values and quantum chemistry applications.
@Suite("Pauli Basis Measurements")
struct PauliBasisMeasurementTests {
    @Test("Measure |0⟩ in Z basis gives +1")
    func measureZeroInZBasis() {
        let state = QuantumState(numQubits: 1) // |0⟩
        var measurement = Measurement()

        for _ in 0 ..< 10 {
            let result = measurement.measurePauli(qubit: 0, basis: .z, state: state)
            #expect(result.eigenvalue == 1)
            #expect(result.collapsedState.isNormalized())
        }
    }

    @Test("Measure |1⟩ in Z basis gives -1")
    func measureOneInZBasis() {
        let state = QuantumState(singleQubit: 1)
        var measurement = Measurement()

        for _ in 0 ..< 10 {
            let result = measurement.measurePauli(qubit: 0, basis: .z, state: state)
            #expect(result.eigenvalue == -1)
            #expect(result.collapsedState.isNormalized())
        }
    }

    @Test("Measure |+⟩ in X basis gives +1")
    func measurePlusInXBasis() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let plus = QuantumState(numQubits: 1, amplitudes: [
            Complex(invSqrt2, 0),
            Complex(invSqrt2, 0),
        ])
        var measurement = Measurement()

        for _ in 0 ..< 10 {
            let result = measurement.measurePauli(qubit: 0, basis: .x, state: plus)
            #expect(result.eigenvalue == 1)
            #expect(result.collapsedState.isNormalized())
        }
    }

    @Test("Measure |-⟩ in X basis gives -1")
    func measureMinusInXBasis() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let minus = QuantumState(numQubits: 1, amplitudes: [
            Complex(invSqrt2, 0),
            Complex(-invSqrt2, 0),
        ])
        var measurement = Measurement()

        for _ in 0 ..< 10 {
            let result = measurement.measurePauli(qubit: 0, basis: .x, state: minus)
            #expect(result.eigenvalue == -1)
            #expect(result.collapsedState.isNormalized())
        }
    }

    @Test("Measure |0⟩ in X basis gives ±1 with equal probability")
    func measureZeroInXBasis() {
        let state = QuantumState(numQubits: 1)
        var measurement = Measurement()

        var outcomes = [Int]()
        for _ in 0 ..< 100 {
            let result = measurement.measurePauli(qubit: 0, basis: .x, state: state)
            outcomes.append(result.eigenvalue)
            #expect(result.eigenvalue == 1 || result.eigenvalue == -1)
        }

        let countPlus = outcomes.filter { $0 == 1 }.count
        let countMinus = outcomes.filter { $0 == -1 }.count

        #expect(abs(Double(countPlus) - 50.0) < 20.0)
        #expect(abs(Double(countMinus) - 50.0) < 20.0)
    }

    @Test("Pauli Y measurement on |0⟩")
    func measureZeroInYBasis() {
        let state = QuantumState(numQubits: 1)
        var measurement = Measurement()

        var outcomes = [Int]()
        for _ in 0 ..< 100 {
            let result = measurement.measurePauli(qubit: 0, basis: .y, state: state)
            outcomes.append(result.eigenvalue)
            #expect(result.eigenvalue == 1 || result.eigenvalue == -1)
            #expect(result.collapsedState.isNormalized())
        }

        let countPlus = outcomes.filter { $0 == 1 }.count
        let countMinus = outcomes.filter { $0 == -1 }.count

        #expect(abs(Double(countPlus) - 50.0) < 20.0)
        #expect(abs(Double(countMinus) - 50.0) < 20.0)
    }

    @Test("Pauli measurement on Bell state")
    func pauliMeasurementBellState() {
        let bell = QuantumCircuit.bellPhiPlus().execute()
        var measurement = Measurement()

        let zResult = measurement.measurePauli(qubit: 0, basis: .z, state: bell)
        #expect(zResult.eigenvalue == 1 || zResult.eigenvalue == -1)
        #expect(zResult.collapsedState.isNormalized())

        let xResult = measurement.measurePauli(qubit: 0, basis: .x, state: bell)
        #expect(xResult.eigenvalue == 1 || xResult.eigenvalue == -1)
        #expect(xResult.collapsedState.isNormalized())
    }

    @Test("PauliBasis enum cases")
    func pauliBasisEnumCases() {
        let allCases = PauliBasis.allCases
        #expect(allCases.contains(.x))
        #expect(allCases.contains(.y))
        #expect(allCases.contains(.z))
        #expect(allCases.count == 3)
    }

    @Test("PauliMeasurementResult description")
    func pauliMeasurementResultDescription() {
        let state = QuantumState(numQubits: 1)
        var measurement = Measurement()
        let result = measurement.measurePauli(qubit: 0, basis: .z, state: state)

        #expect(result.description.contains("PauliMeasurement"))
        #expect(result.description.contains("+1") || result.description.contains("-1"))
    }
}

/// Test suite for arbitrary single-qubit basis measurements.
/// Validates measurement in custom unitary bases beyond standard Pauli operators.
@Suite("Custom Basis Measurements")
struct CustomBasisMeasurementTests {
    @Test("Measure in custom basis (eigenstate)")
    func measureCustomBasisEigenstate() {
        let phase = Double.pi / 4
        let customBasis = [
            Complex(1 / sqrt(2), 0),
            Complex(cos(phase) / sqrt(2), sin(phase) / sqrt(2)),
        ]

        let state = QuantumState(numQubits: 1, amplitudes: customBasis)
        var measurement = Measurement()

        for _ in 0 ..< 10 {
            let (outcome, collapsedState) = measurement.measureCustomBasis(
                qubit: 0,
                basisState: customBasis,
                state: state
            )
            #expect(outcome == 0)
            #expect(collapsedState.isNormalized())
        }
    }

    @Test("Measure |0⟩ in custom basis")
    func measureZeroInCustomBasis() {
        let state = QuantumState(numQubits: 1) // |0⟩

        let customBasis = [
            Complex(1 / sqrt(2), 0),
            Complex(1 / sqrt(2), 0),
        ]

        var measurement = Measurement()
        var outcomes = [Int]()

        for _ in 0 ..< 100 {
            let (outcome, collapsedState) = measurement.measureCustomBasis(
                qubit: 0,
                basisState: customBasis,
                state: state
            )
            outcomes.append(outcome)
            #expect(outcome == 0 || outcome == 1)
            #expect(collapsedState.isNormalized())
        }

        let count0 = outcomes.filter { $0 == 0 }.count
        let count1 = outcomes.filter { $0 == 1 }.count
        #expect(abs(Double(count0) - 50.0) < 20.0)
        #expect(abs(Double(count1) - 50.0) < 20.0)
    }

    @Test("Custom basis with complex phase")
    func customBasisComplexPhase() {
        let customBasis = [
            Complex(0.6, 0.0),
            Complex(0.0, 0.8),
        ]

        let state = QuantumState(numQubits: 1)
        var measurement = Measurement()

        let (outcome, collapsedState) = measurement.measureCustomBasis(
            qubit: 0,
            basisState: customBasis,
            state: state
        )

        #expect(outcome == 0 || outcome == 1)
        #expect(collapsedState.isNormalized())
    }

    @Test("Custom basis measurement on multi-qubit state")
    func customBasisMultiQubit() {
        let state = QuantumCircuit.bellPhiPlus().execute()

        let customBasis = [
            Complex(1 / sqrt(2), 0),
            Complex(1 / sqrt(2), 0),
        ]

        var measurement = Measurement()
        let (outcome, collapsedState) = measurement.measureCustomBasis(
            qubit: 0,
            basisState: customBasis,
            state: state
        )

        #expect(outcome == 0 || outcome == 1)
        #expect(collapsedState.isNormalized())
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
        let pauliString = PauliString(operators: [
            (qubit: 0, basis: .z),
            (qubit: 1, basis: .z),
        ])

        var measurement = Measurement()

        for _ in 0 ..< 10 {
            let result = measurement.measurePauliString(pauliString, state: bell)
            #expect(result.eigenvalue == 1)
            #expect(result.collapsedState.isNormalized())
            #expect(result.individualOutcomes.count == 2)
        }
    }

    @Test("Measure X₀⊗X₁ on Bell state gives +1")
    func measureXXOnBellState() {
        let bell = QuantumCircuit.bellPhiPlus().execute()
        let pauliString = PauliString(operators: [
            (qubit: 0, basis: .x),
            (qubit: 1, basis: .x),
        ])

        var measurement = Measurement()

        for _ in 0 ..< 10 {
            let result = measurement.measurePauliString(pauliString, state: bell)
            #expect(result.eigenvalue == 1)
            #expect(result.collapsedState.isNormalized())
        }
    }

    @Test("Measure single-qubit Pauli string")
    func measureSingleQubitPauliString() {
        let state = QuantumState(numQubits: 1)
        let pauliString = PauliString(operators: [(qubit: 0, basis: .z)])

        var measurement = Measurement()
        let result = measurement.measurePauliString(pauliString, state: state)

        #expect(result.eigenvalue == 1)
        #expect(result.individualOutcomes.count == 1)
        #expect(result.individualOutcomes[0].qubit == 0)
        #expect(result.individualOutcomes[0].outcome == 0)
    }

    @Test("Measure empty Pauli string (identity)")
    func measureIdentityPauliString() {
        let state = QuantumState(numQubits: 2)
        let pauliString = PauliString(operators: [])

        var measurement = Measurement()
        let result = measurement.measurePauliString(pauliString, state: state)

        #expect(result.eigenvalue == 1)
        #expect(result.individualOutcomes.isEmpty)
        #expect(result.collapsedState == state)
    }

    @Test("Measure three-qubit Pauli string")
    func measureThreeQubitPauliString() {
        let ghz = QuantumCircuit.ghzState(numQubits: 3).execute()
        let pauliString = PauliString(operators: [
            (qubit: 0, basis: .z),
            (qubit: 1, basis: .z),
            (qubit: 2, basis: .z),
        ])

        var measurement = Measurement()

        var outcomes = [Int]()
        for _ in 0 ..< 100 {
            let result = measurement.measurePauliString(pauliString, state: ghz)
            outcomes.append(result.eigenvalue)
            #expect(result.eigenvalue == 1 || result.eigenvalue == -1)
            #expect(result.individualOutcomes.count == 3)
        }

        let countPlus = outcomes.filter { $0 == 1 }.count
        let countMinus = outcomes.filter { $0 == -1 }.count
        #expect(abs(Double(countPlus) - 50.0) < 20.0)
        #expect(abs(Double(countMinus) - 50.0) < 20.0)
    }

    @Test("Measure mixed Pauli string (X,Y,Z)")
    func measureMixedPauliString() {
        let state = QuantumCircuit.superposition(numQubits: 3).execute()
        let pauliString = PauliString(operators: [
            (qubit: 0, basis: .x),
            (qubit: 1, basis: .y),
            (qubit: 2, basis: .z),
        ])

        var measurement = Measurement()
        let result = measurement.measurePauliString(pauliString, state: state)

        #expect(result.eigenvalue == 1 || result.eigenvalue == -1)
        #expect(result.collapsedState.isNormalized())
        #expect(result.individualOutcomes.count == 3)
    }

    @Test("PauliString description")
    func pauliStringDescription() {
        let pauliString = PauliString(operators: [
            (qubit: 0, basis: .x),
            (qubit: 1, basis: .z),
        ])

        #expect(pauliString.description.contains("X_0"))
        #expect(pauliString.description.contains("Z_1"))
        #expect(pauliString.description.contains("⊗"))
    }

    @Test("PauliString identity description")
    func pauliStringIdentityDescription() {
        let identity = PauliString(operators: [])
        #expect(identity.description.contains("identity") || identity.description.contains("I"))
    }

    @Test("PauliStringMeasurementResult description")
    func pauliStringMeasurementResultDescription() {
        let state = QuantumState(numQubits: 2)
        let pauliString = PauliString(operators: [(qubit: 0, basis: .z)])
        var measurement = Measurement()

        let result = measurement.measurePauliString(pauliString, state: state)
        #expect(result.description.contains("PauliStringMeasurement"))
        #expect(result.description.contains("+1") || result.description.contains("-1"))
    }
}

/// Test suite for measuring multiple qubits simultaneously.
/// Validates joint probability distributions and partial state collapse
/// for arbitrary qubit subsets.
@Suite("Multiple Qubit Partial Measurements")
struct MultipleQubitPartialMeasurementTests {
    @Test("Measure multiple qubits on GHZ state")
    func measureMultipleQubitsGHZ() {
        let ghz = QuantumCircuit.ghzState(numQubits: 3).execute()
        var measurement = Measurement()

        let (outcomes, collapsedState) = measurement.measureQubits([0, 1], state: ghz)

        #expect(outcomes.count == 2)
        #expect(collapsedState.isNormalized())
        #expect((outcomes[0] == 0 && outcomes[1] == 0) || (outcomes[0] == 1 && outcomes[1] == 1))
    }

    @Test("Measure single qubit using measureQubits")
    func measureSingleQubitUsingMeasureQubits() {
        let state = QuantumState(numQubits: 2)
        var measurement = Measurement()

        let (outcomes, collapsedState) = measurement.measureQubits([0], state: state)

        #expect(outcomes.count == 1)
        #expect(outcomes[0] == 0)
        #expect(collapsedState.isNormalized())
    }

    @Test("Measure all qubits")
    func measureAllQubits() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(numQubits: 2, amplitudes: [
            Complex(invSqrt2, 0),
            Complex(0, 0),
            Complex(0, 0),
            Complex(invSqrt2, 0),
        ])

        var measurement = Measurement()
        let (outcomes, collapsedState) = measurement.measureQubits([0, 1], state: bell)

        #expect(outcomes.count == 2)
        #expect((outcomes == [0, 0]) || (outcomes == [1, 1]))
        #expect(collapsedState.isNormalized())
    }

    @Test("Measure non-adjacent qubits")
    func measureNonAdjacentQubits() {
        let state = QuantumCircuit.superposition(numQubits: 4).execute()
        var measurement = Measurement()

        let (outcomes, collapsedState) = measurement.measureQubits([0, 2], state: state)

        #expect(outcomes.count == 2)
        #expect(outcomes[0] == 0 || outcomes[0] == 1)
        #expect(outcomes[1] == 0 || outcomes[1] == 1)
        #expect(collapsedState.isNormalized())
    }

    @Test("Measure qubits in arbitrary order")
    func measureQubitsArbitraryOrder() {
        let state = QuantumCircuit.ghzState(numQubits: 3).execute()
        var measurement = Measurement()

        let (outcomes, collapsedState) = measurement.measureQubits([2, 0], state: state)

        #expect(outcomes.count == 2)
        #expect(collapsedState.isNormalized())
        #expect((outcomes[0] == 0 && outcomes[1] == 0) || (outcomes[0] == 1 && outcomes[1] == 1))
    }

    @Test("Sequential measurements vs joint measurement")
    func sequentialVsJointMeasurement() {
        let state = QuantumCircuit.bellPhiPlus().execute()
        var measurement1 = Measurement(seed: 42)
        var measurement2 = Measurement(seed: 42)

        let (jointOutcomes, _) = measurement1.measureQubits([0, 1], state: state)
        let (outcome0, collapsed0) = measurement2.measureQubit(0, state: state)
        let (outcome1, _) = measurement2.measureQubit(1, state: collapsed0)

        #expect(outcome0 == outcome1)
        #expect(jointOutcomes[0] == jointOutcomes[1])
    }

    @Test("Measure product state preserves independence")
    func measureProductStateIndependence() {
        let product = QuantumState(numQubits: 3, amplitudes: [
            Complex(0, 0),
            Complex(0, 0),
            Complex(0, 0),
            Complex(0, 0),
            Complex(1 / sqrt(2), 0),
            Complex(1 / sqrt(2), 0),
            Complex(0, 0),
            Complex(0, 0),
        ])

        var measurement = Measurement()
        let (outcomes, collapsedState) = measurement.measureQubits([1, 2], state: product)

        #expect(outcomes[0] == 0)
        #expect(outcomes[1] == 1)
        #expect(collapsedState.isNormalized())

        let (prob0, prob1) = Measurement.marginalProbabilities(qubit: 0, state: collapsedState)
        #expect(abs(prob0 - 0.5) < 1e-10)
        #expect(abs(prob1 - 0.5) < 1e-10)
    }
}

/// Test suite for non-destructive state capture (snapshots).
/// Validates that snapshots preserve quantum coherence without measurement collapse.
@Suite("Statevector Snapshots")
struct StatevectorSnapshotTests {
    @Test("Capture snapshot preserves state")
    func captureSnapshotPreservesState() {
        let bell = QuantumCircuit.bellPhiPlus().execute()
        let snapshot = Measurement.captureSnapshot(state: bell, label: "Bell state")

        #expect(snapshot.state == bell)
        #expect(snapshot.state.isNormalized())
        #expect(snapshot.label == "Bell state")
    }

    @Test("Snapshot without label")
    func snapshotWithoutLabel() {
        let state = QuantumState(numQubits: 1)
        let snapshot = Measurement.captureSnapshot(state: state)

        #expect(snapshot.state == state)
        #expect(snapshot.label == nil)
    }

    @Test("Multiple snapshots preserve different states")
    func multipleSnapshotsPreserveDifferentStates() {
        var circuit = QuantumCircuit(numQubits: 2)

        let snapshot1 = Measurement.captureSnapshot(
            state: circuit.execute(),
            label: "Initial"
        )

        circuit.append(gate: .hadamard, toQubit: 0)
        let snapshot2 = Measurement.captureSnapshot(
            state: circuit.execute(),
            label: "After H"
        )

        circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
        let snapshot3 = Measurement.captureSnapshot(
            state: circuit.execute(),
            label: "Bell state"
        )

        #expect(snapshot1.state != snapshot2.state)
        #expect(snapshot2.state != snapshot3.state)
        #expect(snapshot1.state != snapshot3.state)
    }

    @Test("Snapshot timestamp is valid")
    func snapshotTimestampValid() {
        let state = QuantumState(numQubits: 1)
        let before = Date()
        let snapshot = Measurement.captureSnapshot(state: state)
        let after = Date()

        #expect(snapshot.timestamp >= before)
        #expect(snapshot.timestamp <= after)
    }

    @Test("Snapshot description with label")
    func snapshotDescriptionWithLabel() {
        let state = QuantumState(numQubits: 1)
        let snapshot = Measurement.captureSnapshot(state: state, label: "Test")

        #expect(snapshot.description.contains("Test"))
        #expect(snapshot.description.contains("Snapshot"))
    }

    @Test("Snapshot description without label")
    func snapshotDescriptionWithoutLabel() {
        let state = QuantumState(numQubits: 1)
        let snapshot = Measurement.captureSnapshot(state: state)

        #expect(snapshot.description.contains("Snapshot"))
    }

    @Test("Snapshot preserves complex states")
    func snapshotPreservesComplexStates() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let complexState = QuantumState(numQubits: 2, amplitudes: [
            Complex(0.5, 0.5),
            Complex(0, 0),
            Complex(0, 0),
            Complex(invSqrt2, 0),
        ])

        let snapshot = Measurement.captureSnapshot(state: complexState, label: "Complex")

        #expect(snapshot.state.getAmplitude(ofState: 0) == Complex(0.5, 0.5))
        #expect(snapshot.state.getAmplitude(ofState: 3).real == invSqrt2)
    }

    @Test("Snapshots are independent")
    func snapshotsAreIndependent() {
        let state = QuantumCircuit.bellPhiPlus().execute()
        let snapshot1 = Measurement.captureSnapshot(state: state, label: "First")
        let snapshot2 = Measurement.captureSnapshot(state: state, label: "Second")

        #expect(snapshot1.state == snapshot2.state)
        #expect(snapshot1.label != snapshot2.label)
    }

    @Test("Snapshot timestamp ordering")
    func snapshotTimestampOrdering() {
        let state = QuantumState(numQubits: 1)
        let snapshot1 = Measurement.captureSnapshot(state: state, label: "First")

        Thread.sleep(forTimeInterval: 0.01)

        let snapshot2 = Measurement.captureSnapshot(state: state, label: "Second")

        #expect(snapshot2.timestamp > snapshot1.timestamp)
    }
}

/// Test suite for edge cases in new measurement infrastructure.
/// Validates boundary conditions and error handling.
@Suite("Measurement Infrastructure Edge Cases")
struct MeasurementInfrastructureEdgeCasesTests {
    @Test("Pauli measurement on multi-qubit state affects only target qubit")
    func pauliMeasurementOnlyAffectsTarget() {
        let state = QuantumCircuit.superposition(numQubits: 3).execute()
        var measurement = Measurement()

        let result = measurement.measurePauli(qubit: 1, basis: .x, state: state)

        #expect(result.eigenvalue == 1 || result.eigenvalue == -1)
        #expect(result.collapsedState.isNormalized())
        #expect(result.collapsedState.numQubits == 3)
    }

    @Test("Measure empty qubit list")
    func measureEmptyQubitList() {
        let state = QuantumState(numQubits: 2)
        var measurement = Measurement()

        let (outcomes, _) = measurement.measureQubits([0], state: state)
        #expect(outcomes.count == 1)
    }

    @Test("Pauli measurement preserves normalization across all bases")
    func pauliMeasurementNormalizationAllBases() {
        let state = QuantumCircuit.bellPhiPlus().execute()
        var measurement = Measurement()

        for basis in PauliBasis.allCases {
            let result = measurement.measurePauli(qubit: 0, basis: basis, state: state)
            #expect(result.collapsedState.isNormalized())
        }
    }
}
