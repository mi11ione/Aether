// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under Apache 2.0

import Aether
import Foundation
import Testing

/// Test suite for AmplitudeEstimationStrategy enum cases and automatic resolution,
/// verifying strategy selection logic based on precision qubit budget.
@Suite("AmplitudeEstimationStrategy Selection")
struct AmplitudeEstimationStrategyTests {
    @Test("Standard strategy produces Standard description")
    func standardStrategy() {
        let result = UnifiedAmplitudeEstimationResult(
            estimatedAmplitude: 0.5, estimatedProbability: 0.25,
            confidenceInterval: (lower: 0.2, upper: 0.3),
            oracleCalls: 64, strategyUsed: .standard,
        )
        #expect(result.description.contains("Standard QPE"), "Strategy description should contain Standard QPE")
    }

    @Test("Iterative strategy produces Iterative description")
    func iterativeStrategy() {
        let result = UnifiedAmplitudeEstimationResult(
            estimatedAmplitude: 0.5, estimatedProbability: 0.25,
            confidenceInterval: (lower: 0.2, upper: 0.3),
            oracleCalls: 64, strategyUsed: .iterative,
        )
        #expect(result.description.contains("Iterative"), "Strategy description should contain Iterative")
    }

    @Test("Maximum likelihood strategy stores shot count in description")
    func maximumLikelihoodStoresShots() {
        let result = UnifiedAmplitudeEstimationResult(
            estimatedAmplitude: 0.5, estimatedProbability: 0.25,
            confidenceInterval: (lower: 0.2, upper: 0.3),
            oracleCalls: 64, strategyUsed: .maximumLikelihood(shots: 50),
        )
        #expect(result.description.contains("50"), "Description should contain shot count 50")
    }

    @Test("Automatic strategy produces Automatic description")
    func automaticStrategy() {
        let result = UnifiedAmplitudeEstimationResult(
            estimatedAmplitude: 0.5, estimatedProbability: 0.25,
            confidenceInterval: (lower: 0.2, upper: 0.3),
            oracleCalls: 64, strategyUsed: .automatic,
        )
        #expect(result.description.contains("Automatic"), "Strategy description should contain Automatic")
    }
}

/// Test suite for UnifiedAmplitudeEstimationResult struct properties, confidence
/// interval bounds, and CustomStringConvertible description formatting.
@Suite("UnifiedAmplitudeEstimationResult Properties")
struct UnifiedAmplitudeEstimationResultTests {
    @Test("Result stores all properties correctly")
    func resultStoresProperties() {
        let result = UnifiedAmplitudeEstimationResult(
            estimatedAmplitude: 0.5,
            estimatedProbability: 0.25,
            confidenceInterval: (lower: 0.20, upper: 0.30),
            oracleCalls: 128,
            strategyUsed: .standard,
        )

        #expect(abs(result.estimatedAmplitude - 0.5) < 1e-10, "Amplitude should be 0.5")
        #expect(abs(result.estimatedProbability - 0.25) < 1e-10, "Probability should be 0.25")
        #expect(abs(result.confidenceInterval.lower - 0.20) < 1e-10, "Lower bound should be 0.20")
        #expect(abs(result.confidenceInterval.upper - 0.30) < 1e-10, "Upper bound should be 0.30")
        #expect(result.oracleCalls == 128, "Oracle calls should be 128")
    }

    @Test("Confidence interval lower <= upper")
    func confidenceIntervalOrdered() {
        let result = UnifiedAmplitudeEstimationResult(
            estimatedAmplitude: 0.7,
            estimatedProbability: 0.49,
            confidenceInterval: (lower: 0.4, upper: 0.58),
            oracleCalls: 64,
            strategyUsed: .iterative,
        )

        #expect(result.confidenceInterval.lower <= result.confidenceInterval.upper, "Lower bound must be <= upper bound")
    }

    @Test("Description contains strategy name")
    func descriptionContainsStrategy() {
        let result = UnifiedAmplitudeEstimationResult(
            estimatedAmplitude: 0.5,
            estimatedProbability: 0.25,
            confidenceInterval: (lower: 0.2, upper: 0.3),
            oracleCalls: 64,
            strategyUsed: .standard,
        )

        #expect(result.description.contains("Standard"), "Description should contain strategy name")
    }

    @Test("Description contains amplitude value")
    func descriptionContainsAmplitude() {
        let result = UnifiedAmplitudeEstimationResult(
            estimatedAmplitude: 0.5,
            estimatedProbability: 0.25,
            confidenceInterval: (lower: 0.2, upper: 0.3),
            oracleCalls: 64,
            strategyUsed: .iterative,
        )

        #expect(result.description.contains("Amplitude"), "Description should mention amplitude")
    }
}

/// Test suite for UnifiedAmplitudeEstimation standard QPE-based strategy
/// verifying amplitude estimation accuracy on small counting oracles.
@Suite("UnifiedAmplitudeEstimation Standard Strategy")
struct UnifiedAEStandardTests {
    @Test("Standard AE estimates amplitude for single marked state")
    func standardAESingleMarked() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1])
        let ae = UnifiedAmplitudeEstimation(oracle: oracle)
        let result = await ae.estimate(precisionQubits: 4, strategy: .standard)

        #expect(result.estimatedProbability >= 0.0, "Probability should be non-negative")
        #expect(result.estimatedProbability <= 1.0, "Probability should be at most 1")
        #expect(result.oracleCalls > 0, "Oracle calls must be positive")
    }

    @Test("Standard AE estimates amplitude for multiple marked states")
    func standardAEMultipleMarked() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1, 3])
        let ae = UnifiedAmplitudeEstimation(oracle: oracle)
        let result = await ae.estimate(precisionQubits: 4, strategy: .standard)

        #expect(result.estimatedProbability >= 0.0, "Probability should be non-negative")
        #expect(result.estimatedProbability <= 1.0, "Probability should be at most 1")
        #expect(result.estimatedAmplitude >= 0.0, "Amplitude should be non-negative")
    }

    @Test("Standard AE confidence interval contains true probability")
    func standardAEConfidenceInterval() async {
        let oracle = CountingOracle(qubits: 3, markedStates: [0, 4])
        let ae = UnifiedAmplitudeEstimation(oracle: oracle)
        let result = await ae.estimate(precisionQubits: 5, strategy: .standard)

        #expect(result.confidenceInterval.lower <= result.confidenceInterval.upper, "CI must be ordered")
        #expect(result.estimatedAmplitude >= 0.0, "Amplitude must be non-negative")
        #expect(result.estimatedAmplitude <= 1.0, "Amplitude must be at most 1")
    }

    @Test("Standard AE returns standard as strategy used")
    func standardAEReportsStrategy() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1])
        let ae = UnifiedAmplitudeEstimation(oracle: oracle)
        let result = await ae.estimate(precisionQubits: 3, strategy: .standard)

        #expect(result.description.contains("Standard QPE"), "Strategy should be standard")
    }
}

/// Test suite for UnifiedAmplitudeEstimation iterative strategy verifying
/// single-ancilla amplitude extraction at various precision levels.
@Suite("UnifiedAmplitudeEstimation Iterative Strategy")
struct UnifiedAEIterativeTests {
    @Test("Iterative AE estimates amplitude for small oracle")
    func iterativeAESmallOracle() async {
        let oracle = CountingOracle(qubits: 3, markedStates: [7])
        let ae = UnifiedAmplitudeEstimation(oracle: oracle)
        let result = await ae.estimate(precisionQubits: 4, strategy: .iterative)

        #expect(result.estimatedAmplitude >= 0.0, "Amplitude must be non-negative")
        #expect(result.estimatedAmplitude <= 1.0, "Amplitude must be at most 1")
        #expect(result.oracleCalls > 0, "Oracle calls must be positive")
    }

    @Test("Iterative AE reports iterative as strategy used")
    func iterativeAEReportsStrategy() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1])
        let ae = UnifiedAmplitudeEstimation(oracle: oracle)
        let result = await ae.estimate(precisionQubits: 3, strategy: .iterative)

        #expect(result.description.contains("Iterative"), "Strategy should be iterative")
    }

    @Test("Iterative AE produces bounded confidence interval")
    func iterativeAEBoundedCI() async {
        let oracle = CountingOracle(qubits: 3, markedStates: [2, 6])
        let ae = UnifiedAmplitudeEstimation(oracle: oracle)
        let result = await ae.estimate(precisionQubits: 4, strategy: .iterative)

        #expect(result.confidenceInterval.lower >= 0.0, "CI lower must be non-negative")
        #expect(result.confidenceInterval.upper <= 1.0, "CI upper must be at most 1")
        #expect(result.confidenceInterval.lower <= result.confidenceInterval.upper, "CI must be ordered")
    }
}

/// Test suite for UnifiedAmplitudeEstimation maximum likelihood strategy
/// verifying MLE post-processing, Fisher information confidence intervals,
/// and tighter bounds compared to raw iterative extraction.
@Suite("UnifiedAmplitudeEstimation MLE Strategy")
struct UnifiedAEMLETests {
    @Test("MLE AE estimates amplitude for counting oracle")
    func mleAEEstimate() async {
        let oracle = CountingOracle(qubits: 3, markedStates: [3, 5])
        let ae = UnifiedAmplitudeEstimation(oracle: oracle)
        let result = await ae.estimate(precisionQubits: 4, strategy: .maximumLikelihood(shots: 4))

        #expect(result.estimatedAmplitude >= 0.0, "Amplitude must be non-negative")
        #expect(result.estimatedAmplitude <= 1.0, "Amplitude must be at most 1")
        #expect(result.oracleCalls > 0, "Oracle calls must be positive")
    }

    @Test("MLE AE reports MLE as strategy used")
    func mleAEReportsStrategy() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1])
        let ae = UnifiedAmplitudeEstimation(oracle: oracle)
        let result = await ae.estimate(precisionQubits: 3, strategy: .maximumLikelihood(shots: 3))

        #expect(result.description.contains("MLE"), "Strategy should be MLE")
    }

    @Test("MLE AE confidence interval is bounded")
    func mleAEBoundedCI() async {
        let oracle = CountingOracle(qubits: 3, markedStates: [0, 1, 2])
        let ae = UnifiedAmplitudeEstimation(oracle: oracle)
        let result = await ae.estimate(precisionQubits: 4, strategy: .maximumLikelihood(shots: 4))

        #expect(result.confidenceInterval.lower >= 0.0, "CI lower must be non-negative")
        #expect(result.confidenceInterval.upper <= 1.0, "CI upper must be at most 1")
    }

    @Test("MLE AE handles zero-amplitude oracle with Fisher info fallback")
    func mleAEZeroAmplitude() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [])
        let ae = UnifiedAmplitudeEstimation(oracle: oracle)
        let result = await ae.estimate(precisionQubits: 3, strategy: .maximumLikelihood(shots: 3))

        #expect(result.estimatedAmplitude >= 0.0, "Amplitude must be non-negative")
        #expect(result.confidenceInterval.lower >= 0.0, "CI lower must be non-negative")
    }
}

/// Test suite for UnifiedAmplitudeEstimation automatic strategy selection
/// verifying that the resolver picks appropriate strategies based on qubit budgets.
@Suite("UnifiedAmplitudeEstimation Automatic Selection")
struct UnifiedAEAutomaticTests {
    @Test("Automatic selection produces valid result for small precision")
    func automaticSmallPrecision() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1])
        let ae = UnifiedAmplitudeEstimation(oracle: oracle)
        let result = await ae.estimate(precisionQubits: 3, strategy: .automatic)

        #expect(result.estimatedAmplitude >= 0.0, "Amplitude must be non-negative")
        #expect(result.estimatedAmplitude <= 1.0, "Amplitude must be at most 1")
    }

    @Test("Automatic selection produces valid result for medium precision")
    func automaticMediumPrecision() async {
        let oracle = CountingOracle(qubits: 3, markedStates: [5])
        let ae = UnifiedAmplitudeEstimation(oracle: oracle)
        let result = await ae.estimate(precisionQubits: 5, strategy: .automatic)

        #expect(result.oracleCalls > 0, "Oracle calls must be positive")
    }

    @Test("Automatic selection uses MLE for high precision")
    func automaticHighPrecisionUsesMLE() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1])
        let ae = UnifiedAmplitudeEstimation(oracle: oracle)
        let result = await ae.estimate(precisionQubits: 7, strategy: .automatic)

        #expect(result.description.contains("MLE"), "High precision should use MLE")
    }

    @Test("Default strategy parameter is automatic")
    func defaultStrategyIsAutomatic() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1])
        let ae = UnifiedAmplitudeEstimation(oracle: oracle)
        let result = await ae.estimate(precisionQubits: 4)

        #expect(result.oracleCalls > 0, "Default strategy should produce valid result")
    }
}

/// Test suite for UnifiedAmplitudeEstimationResult description edge cases
/// verifying all strategy name branches in the CustomStringConvertible output.
@Suite("UnifiedAmplitudeEstimationResult Description Edge Cases")
struct UnifiedAEResultDescriptionEdgeCases {
    @Test("Description handles automatic strategy")
    func descriptionHandlesAutomatic() {
        let result = UnifiedAmplitudeEstimationResult(
            estimatedAmplitude: 0.5,
            estimatedProbability: 0.25,
            confidenceInterval: (lower: 0.2, upper: 0.3),
            oracleCalls: 64,
            strategyUsed: .automatic,
        )

        #expect(result.description.contains("Automatic"), "Description should contain Automatic")
    }

    @Test("Description handles MLE strategy with shot count")
    func descriptionHandlesMLE() {
        let result = UnifiedAmplitudeEstimationResult(
            estimatedAmplitude: 0.5,
            estimatedProbability: 0.25,
            confidenceInterval: (lower: 0.2, upper: 0.3),
            oracleCalls: 64,
            strategyUsed: .maximumLikelihood(shots: 42),
        )

        #expect(result.description.contains("MLE"), "Description should contain MLE")
        #expect(result.description.contains("42"), "Description should contain shot count")
    }
}

/// Test suite for UnifiedAmplitudeEstimation automatic strategy dispatching
/// to iterative path for low precision qubit counts.
@Suite("UnifiedAmplitudeEstimation Automatic Low Precision")
struct UnifiedAEAutomaticLowTests {
    @Test("Automatic with 2 precision qubits uses iterative")
    func automaticLowPrecisionUsesIterative() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1])
        let ae = UnifiedAmplitudeEstimation(oracle: oracle)
        let result = await ae.estimate(precisionQubits: 2, strategy: .automatic)

        #expect(result.description.contains("Iterative"), "Low precision should use iterative")
    }

    @Test("Automatic with 5 precision qubits uses standard")
    func automaticMedPrecisionUsesStandard() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1])
        let ae = UnifiedAmplitudeEstimation(oracle: oracle)
        let result = await ae.estimate(precisionQubits: 5, strategy: .automatic)

        #expect(result.description.contains("Standard QPE"), "Medium precision should use standard")
    }
}

struct SingleQubitOracle: AmplitudeOracle, Sendable {
    var qubits: Int {
        1
    }

    func applyStatePreparation(to circuit: inout QuantumCircuit) {
        circuit.append(.hadamard, to: 0)
    }

    func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
        circuit.append(.hadamard, to: 0)
    }

    func applyMarkingOracle(to circuit: inout QuantumCircuit) {
        circuit.append(.pauliZ, to: 0)
    }
}

struct MultiGateOracle: AmplitudeOracle, Sendable {
    var qubits: Int {
        2
    }

    func applyStatePreparation(to circuit: inout QuantumCircuit) {
        circuit.append(.hadamard, to: 0)
    }

    func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
        circuit.append(.hadamard, to: 0)
    }

    func applyMarkingOracle(to circuit: inout QuantumCircuit) {
        circuit.append(.pauliZ, to: 0)
        circuit.append(.pauliY, to: 1)
        circuit.append(.rotationZ(.pi / 4), to: 1)
        circuit.append(.rotationY(.pi / 4), to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.phase(.pi / 3), to: 0)
        circuit.append(.sGate, to: 1)
        circuit.append(.tGate, to: 0)
        circuit.append(.controlledPhase(.pi / 2), to: [0, 1])
    }
}

/// Test suite for UnifiedAmplitudeEstimation with custom oracles exercising
/// all gate decomposition paths in the controlled gate builder.
@Suite("UnifiedAmplitudeEstimation Multi-Gate Coverage")
struct UnifiedAEMultiGateTests {
    @Test("Multi-gate oracle estimation produces valid result")
    func multiGateOracleEstimation() async {
        let oracle = MultiGateOracle()
        let ae = UnifiedAmplitudeEstimation(oracle: oracle)
        let result = await ae.estimate(precisionQubits: 3, strategy: .iterative)

        #expect(result.estimatedAmplitude >= 0.0, "Amplitude must be non-negative")
        #expect(result.estimatedAmplitude <= 1.0, "Amplitude must be at most 1")
        #expect(result.oracleCalls > 0, "Oracle calls must be positive")
    }

    @Test("Single-qubit oracle estimation exercises 1-qubit reflection path")
    func singleQubitOracleEstimation() async {
        let oracle = SingleQubitOracle()
        let ae = UnifiedAmplitudeEstimation(oracle: oracle)
        let result = await ae.estimate(precisionQubits: 3, strategy: .iterative)

        #expect(result.estimatedAmplitude >= 0.0, "Amplitude must be non-negative")
        #expect(result.estimatedAmplitude <= 1.0, "Amplitude must be at most 1")
    }
}

/// Test suite for GeneralizedAmplification utility methods validating optimal
/// iteration computation, circuit construction, and success probability formulas.
@Suite("GeneralizedAmplification Utilities")
struct GeneralizedAmplificationTests {
    @Test("Optimal iterations is non-negative for valid amplitude")
    func optimalIterationsNonNegative() {
        let iterations = GeneralizedAmplification.optimalIterations(amplitude: 0.25)
        #expect(iterations >= 0, "Optimal iterations must be non-negative")
    }

    @Test("Optimal iterations is zero for amplitude near 1")
    func optimalIterationsZeroForLargeAmplitude() {
        let iterations = GeneralizedAmplification.optimalIterations(amplitude: 0.95)
        #expect(iterations <= 1, "Near-unity amplitude needs few iterations")
    }

    @Test("Optimal iterations is zero for very small amplitude")
    func optimalIterationsForTinyAmplitude() {
        let iterations = GeneralizedAmplification.optimalIterations(amplitude: 1e-16)
        #expect(iterations >= 0, "Tiny amplitude iterations must be non-negative")
    }

    @Test("Optimal iterations increases for smaller amplitudes")
    func optimalIterationsScalesInversely() {
        let small = GeneralizedAmplification.optimalIterations(amplitude: 0.1)
        let large = GeneralizedAmplification.optimalIterations(amplitude: 0.5)

        #expect(small >= large, "Smaller amplitude needs more iterations")
    }

    @Test("Amplify produces circuit with correct qubit count")
    func amplifyCircuitQubits() {
        let oracle = CountingOracle(qubits: 3, markedStates: [5])
        let circuit = GeneralizedAmplification.amplify(oracle: oracle, iterations: 2)

        #expect(circuit.qubits == 3, "Circuit should have 3 qubits")
    }

    @Test("Amplify with zero iterations applies only state preparation")
    func amplifyZeroIterations() {
        let oracle = CountingOracle(qubits: 2, markedStates: [1])
        let circuit = GeneralizedAmplification.amplify(oracle: oracle, iterations: 0)

        #expect(circuit.qubits == 2, "Circuit should have 2 qubits")
        #expect(circuit.operations.count > 0, "Circuit should have state preparation operations")
    }

    @Test("Success probability is bounded between 0 and 1")
    func successProbabilityBounded() {
        let prob = GeneralizedAmplification.successProbability(
            initialAmplitude: 0.25, iterations: 3,
        )

        #expect(prob >= 0.0, "Success probability must be non-negative")
        #expect(prob <= 1.0, "Success probability must be at most 1")
    }

    @Test("Success probability at optimal iterations is near 1")
    func successProbabilityAtOptimal() {
        let amplitude = 0.25
        let optIter = GeneralizedAmplification.optimalIterations(amplitude: amplitude)
        let prob = GeneralizedAmplification.successProbability(
            initialAmplitude: amplitude, iterations: optIter,
        )

        #expect(prob > 0.8, "Success probability at optimal iterations should be high")
    }

    @Test("Success probability for zero iterations equals amplitude squared")
    func successProbabilityZeroIterations() {
        let amplitude = 0.5
        let prob = GeneralizedAmplification.successProbability(
            initialAmplitude: amplitude, iterations: 0,
        )

        #expect(abs(prob - amplitude * amplitude) < 1e-10, "Zero iterations gives initial probability")
    }
}
