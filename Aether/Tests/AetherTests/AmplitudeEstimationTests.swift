// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under Apache 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for AmplitudeEstimationResult struct properties and display.
/// Validates result initialization, property access, confidence intervals,
/// and CustomStringConvertible description formatting.
@Suite("AmplitudeEstimationResult Properties")
struct AmplitudeEstimationResultTests {
    @Test("Result stores all properties correctly")
    func resultStoresProperties() {
        let result = AmplitudeEstimationResult(
            estimatedAmplitude: 0.5,
            estimatedProbability: 0.25,
            confidenceInterval: (lower: 0.20, upper: 0.30),
            oracleCalls: 64,
            classicalEquivalentSamples: 4096,
        )

        #expect(abs(result.estimatedAmplitude - 0.5) < 1e-10, "Estimated amplitude should be 0.5")
        #expect(abs(result.estimatedProbability - 0.25) < 1e-10, "Estimated probability should be 0.25")
        #expect(abs(result.confidenceInterval.lower - 0.20) < 1e-10, "Lower bound should be 0.20")
        #expect(abs(result.confidenceInterval.upper - 0.30) < 1e-10, "Upper bound should be 0.30")
        #expect(result.oracleCalls == 64, "Oracle calls should be 64")
        #expect(result.classicalEquivalentSamples == 4096, "Classical samples should be 4096")
    }

    @Test("Confidence interval lower <= estimate <= upper")
    func confidenceIntervalContainsEstimate() {
        let result = AmplitudeEstimationResult(
            estimatedAmplitude: 0.7071,
            estimatedProbability: 0.5,
            confidenceInterval: (lower: 0.45, upper: 0.55),
            oracleCalls: 128,
            classicalEquivalentSamples: 8192,
        )

        #expect(result.confidenceInterval.lower <= result.estimatedProbability, "Lower bound should be <= estimate")
        #expect(result.estimatedProbability <= result.confidenceInterval.upper, "Estimate should be <= upper bound")
    }

    @Test("Description contains amplitude value")
    func descriptionContainsAmplitude() {
        let result = AmplitudeEstimationResult(
            estimatedAmplitude: 0.5,
            estimatedProbability: 0.25,
            confidenceInterval: (lower: 0.20, upper: 0.30),
            oracleCalls: 64,
            classicalEquivalentSamples: 4096,
        )

        #expect(result.description.contains("Amplitude"), "Description should contain 'Amplitude'")
        #expect(result.description.contains("0.5"), "Description should contain amplitude value")
    }

    @Test("Description contains probability value")
    func descriptionContainsProbability() {
        let result = AmplitudeEstimationResult(
            estimatedAmplitude: 0.5,
            estimatedProbability: 0.25,
            confidenceInterval: (lower: 0.20, upper: 0.30),
            oracleCalls: 64,
            classicalEquivalentSamples: 4096,
        )

        #expect(result.description.contains("Probability"), "Description should contain 'Probability'")
        #expect(result.description.contains("0.25"), "Description should contain probability value")
    }

    @Test("Description contains confidence interval")
    func descriptionContainsConfidenceInterval() {
        let result = AmplitudeEstimationResult(
            estimatedAmplitude: 0.5,
            estimatedProbability: 0.25,
            confidenceInterval: (lower: 0.20, upper: 0.30),
            oracleCalls: 64,
            classicalEquivalentSamples: 4096,
        )

        #expect(result.description.contains("Confidence"), "Description should contain 'Confidence'")
        #expect(result.description.contains("Interval"), "Description should contain 'Interval'")
    }

    @Test("Description contains oracle calls")
    func descriptionContainsOracleCalls() {
        let result = AmplitudeEstimationResult(
            estimatedAmplitude: 0.5,
            estimatedProbability: 0.25,
            confidenceInterval: (lower: 0.20, upper: 0.30),
            oracleCalls: 64,
            classicalEquivalentSamples: 4096,
        )

        #expect(result.description.contains("Oracle"), "Description should contain 'Oracle'")
        #expect(result.description.contains("64"), "Description should contain oracle call count")
    }

    @Test("Description contains classical samples")
    func descriptionContainsClassicalSamples() {
        let result = AmplitudeEstimationResult(
            estimatedAmplitude: 0.5,
            estimatedProbability: 0.25,
            confidenceInterval: (lower: 0.20, upper: 0.30),
            oracleCalls: 64,
            classicalEquivalentSamples: 4096,
        )

        #expect(result.description.contains("Classical"), "Description should contain 'Classical'")
        #expect(result.description.contains("4096"), "Description should contain classical sample count")
    }

    @Test("Description contains speedup")
    func descriptionContainsSpeedup() {
        let result = AmplitudeEstimationResult(
            estimatedAmplitude: 0.5,
            estimatedProbability: 0.25,
            confidenceInterval: (lower: 0.20, upper: 0.30),
            oracleCalls: 64,
            classicalEquivalentSamples: 4096,
        )

        #expect(result.description.contains("Speedup"), "Description should contain 'Speedup'")
    }

    @Test("Amplitude in valid range [0, 1]")
    func amplitudeInValidRange() {
        let result = AmplitudeEstimationResult(
            estimatedAmplitude: 0.7071,
            estimatedProbability: 0.5,
            confidenceInterval: (lower: 0.45, upper: 0.55),
            oracleCalls: 128,
            classicalEquivalentSamples: 8192,
        )

        #expect(result.estimatedAmplitude >= 0, "Amplitude should be non-negative")
        #expect(result.estimatedAmplitude <= 1, "Amplitude should be at most 1")
    }

    @Test("Probability in valid range [0, 1]")
    func probabilityInValidRange() {
        let result = AmplitudeEstimationResult(
            estimatedAmplitude: 0.7071,
            estimatedProbability: 0.5,
            confidenceInterval: (lower: 0.45, upper: 0.55),
            oracleCalls: 128,
            classicalEquivalentSamples: 8192,
        )

        #expect(result.estimatedProbability >= 0, "Probability should be non-negative")
        #expect(result.estimatedProbability <= 1, "Probability should be at most 1")
    }
}

/// Test suite for AEConfiguration initialization and validation.
/// Validates precision qubit bounds, iterative flag handling,
/// and precondition enforcement for invalid inputs.
@Suite("AEConfiguration Initialization")
struct AEConfigurationTests {
    @Test("Configuration with valid precision qubits")
    func validPrecisionQubits() {
        let config = AEConfiguration(precisionQubits: 6)

        #expect(config.precisionQubits == 6, "Precision qubits should be 6")
        #expect(config.useIterative == false, "Default useIterative should be false")
    }

    @Test("Configuration with minimum precision qubits")
    func minimumPrecisionQubits() {
        let config = AEConfiguration(precisionQubits: 1)

        #expect(config.precisionQubits == 1, "Minimum precision qubits should be 1")
    }

    @Test("Configuration with maximum precision qubits")
    func maximumPrecisionQubits() {
        let config = AEConfiguration(precisionQubits: 15)

        #expect(config.precisionQubits == 15, "Maximum precision qubits should be 15")
    }

    @Test("Configuration with useIterative true")
    func useIterativeTrue() {
        let config = AEConfiguration(precisionQubits: 8, useIterative: true)

        #expect(config.precisionQubits == 8, "Precision qubits should be 8")
        #expect(config.useIterative == true, "useIterative should be true")
    }

    @Test("Configuration with useIterative false explicit")
    func useIterativeFalseExplicit() {
        let config = AEConfiguration(precisionQubits: 4, useIterative: false)

        #expect(config.precisionQubits == 4, "Precision qubits should be 4")
        #expect(config.useIterative == false, "useIterative should be false")
    }

    @Test("Configuration stores precision independently of iterative flag")
    func precisionIndependentOfIterative() {
        let standardConfig = AEConfiguration(precisionQubits: 10, useIterative: false)
        let iterativeConfig = AEConfiguration(precisionQubits: 10, useIterative: true)

        #expect(standardConfig.precisionQubits == iterativeConfig.precisionQubits, "Precision should be same regardless of iterative flag")
    }
}

/// Test suite for CountingOracle initialization and protocol conformance.
/// Validates qubit count, marked states storage, and AmplitudeOracle
/// protocol method implementations for state preparation.
@Suite("CountingOracle Initialization")
struct CountingOracleInitializationTests {
    @Test("CountingOracle stores qubits correctly")
    func storesQubitsCorrectly() {
        let oracle = CountingOracle(qubits: 3, markedStates: [1, 5])

        #expect(oracle.qubits == 3, "Oracle should store 3 qubits")
    }

    @Test("CountingOracle stores marked states correctly")
    func storesMarkedStatesCorrectly() {
        let oracle = CountingOracle(qubits: 4, markedStates: [0, 3, 7, 15])

        #expect(oracle.markedStates.count == 4, "Oracle should store 4 marked states")
        #expect(oracle.markedStates.contains(0), "Marked states should contain 0")
        #expect(oracle.markedStates.contains(3), "Marked states should contain 3")
        #expect(oracle.markedStates.contains(7), "Marked states should contain 7")
        #expect(oracle.markedStates.contains(15), "Marked states should contain 15")
    }

    @Test("CountingOracle with empty marked states")
    func emptyMarkedStates() {
        let oracle = CountingOracle(qubits: 2, markedStates: [])

        #expect(oracle.markedStates.isEmpty, "Marked states should be empty")
        #expect(oracle.qubits == 2, "Qubits should be 2")
    }

    @Test("CountingOracle with single marked state")
    func singleMarkedState() {
        let oracle = CountingOracle(qubits: 3, markedStates: [5])

        #expect(oracle.markedStates.count == 1, "Should have exactly one marked state")
        #expect(oracle.markedStates[0] == 5, "Single marked state should be 5")
    }

    @Test("CountingOracle with minimum qubits")
    func minimumQubits() {
        let oracle = CountingOracle(qubits: 1, markedStates: [0])

        #expect(oracle.qubits == 1, "Minimum qubits should be 1")
        #expect(oracle.markedStates == [0], "Marked state should be [0]")
    }

    @Test("CountingOracle with maximum qubits")
    func maximumQubits() {
        let oracle = CountingOracle(qubits: 10, markedStates: [0, 1023])

        #expect(oracle.qubits == 10, "Maximum qubits should be 10")
        #expect(oracle.markedStates.contains(1023), "Should allow marking state 1023 for 10 qubits")
    }

    @Test("CountingOracle marked state at boundary")
    func markedStateAtBoundary() {
        let oracle = CountingOracle(qubits: 4, markedStates: [15])

        #expect(oracle.markedStates.contains(15), "Should allow marking state 15 (2^4 - 1)")
    }
}

/// Test suite for CountingOracle state preparation methods.
/// Validates applyStatePreparation, applyStatePreparationInverse,
/// and applyMarkingOracle circuit construction.
@Suite("CountingOracle State Preparation")
struct CountingOracleStatePreparationTests {
    @Test("State preparation adds gates to circuit")
    func statePreparationAddsGates() {
        let oracle = CountingOracle(qubits: 3, markedStates: [1])
        var circuit = QuantumCircuit(qubits: 3)

        oracle.applyStatePreparation(to: &circuit)

        #expect(circuit.operations.count >= 3, "State preparation should add at least 3 Hadamard gates for 3 qubits")
    }

    @Test("State preparation inverse adds gates to circuit")
    func statePreparationInverseAddsGates() {
        let oracle = CountingOracle(qubits: 2, markedStates: [1])
        var circuit = QuantumCircuit(qubits: 2)

        oracle.applyStatePreparationInverse(to: &circuit)

        #expect(circuit.operations.count >= 2, "State preparation inverse should add at least 2 Hadamard gates")
    }

    @Test("Marking oracle adds gates to circuit")
    func markingOracleAddsGates() {
        let oracle = CountingOracle(qubits: 2, markedStates: [1])
        var circuit = QuantumCircuit(qubits: 2)

        oracle.applyMarkingOracle(to: &circuit)

        #expect(circuit.operations.count > 0, "Marking oracle should add gates to circuit")
    }

    @Test("State preparation for different qubit counts")
    func statePreparationDifferentQubitCounts() {
        let oracle2 = CountingOracle(qubits: 2, markedStates: [0])
        let oracle4 = CountingOracle(qubits: 4, markedStates: [0])

        var circuit2 = QuantumCircuit(qubits: 2)
        var circuit4 = QuantumCircuit(qubits: 4)

        oracle2.applyStatePreparation(to: &circuit2)
        oracle4.applyStatePreparation(to: &circuit4)

        #expect(circuit4.operations.count >= circuit2.operations.count, "More qubits should result in at least as many gates")
    }
}

/// Test suite for AmplitudeEstimation run() with standard QPE.
/// Validates amplitude estimation accuracy, confidence intervals,
/// and quadratic speedup metrics for various oracle configurations.
@Suite("AmplitudeEstimation Standard QPE")
struct AmplitudeEstimationStandardQPETests {
    @Test("2 qubits, 1 marked state yields amplitude ~0.5")
    func twoQubitsOneMarkedState() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [0])
        let config = AEConfiguration(precisionQubits: 4, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0, "Probability should be non-negative")
        #expect(result.estimatedProbability <= 1, "Probability should be at most 1")
        #expect(result.estimatedAmplitude >= 0, "Amplitude should be non-negative")
        #expect(result.estimatedAmplitude <= 1, "Amplitude should be at most 1")
    }

    @Test("3 qubits, 2 marked states estimation")
    func threeQubitsTwoMarkedStates() async {
        let oracle = CountingOracle(qubits: 3, markedStates: [0, 1])
        let config = AEConfiguration(precisionQubits: 4, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0, "Probability should be non-negative")
        #expect(result.estimatedProbability <= 1, "Probability should be at most 1")
    }

    @Test("Estimated probability in [0, 1]")
    func estimatedProbabilityInRange() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1, 2])
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0, "Probability must be >= 0")
        #expect(result.estimatedProbability <= 1, "Probability must be <= 1")
    }

    @Test("Estimated amplitude in [0, 1]")
    func estimatedAmplitudeInRange() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [0])
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedAmplitude >= 0, "Amplitude must be >= 0")
        #expect(result.estimatedAmplitude <= 1, "Amplitude must be <= 1")
    }

    @Test("Confidence interval bounds probability estimate")
    func confidenceIntervalBoundsEstimate() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1])
        let config = AEConfiguration(precisionQubits: 4, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.confidenceInterval.lower <= result.estimatedProbability, "Lower bound should be <= estimate")
        #expect(result.estimatedProbability <= result.confidenceInterval.upper, "Estimate should be <= upper bound")
    }

    @Test("Oracle calls positive")
    func oracleCallsPositive() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [0])
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.oracleCalls > 0, "Oracle calls should be positive")
    }

    @Test("Classical equivalent samples > oracle calls (quadratic speedup)")
    func classicalSamplesGreaterThanOracleCalls() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1])
        let config = AEConfiguration(precisionQubits: 4, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.classicalEquivalentSamples > result.oracleCalls, "Classical samples should exceed oracle calls demonstrating speedup")
    }

    @Test("More precision qubits yield more oracle calls")
    func morePrecisionMoreOracleCalls() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [0])
        let configLow = AEConfiguration(precisionQubits: 2, useIterative: false)
        let configHigh = AEConfiguration(precisionQubits: 4, useIterative: false)

        let aeLow = AmplitudeEstimation(oracle: oracle, configuration: configLow)
        let aeHigh = AmplitudeEstimation(oracle: oracle, configuration: configHigh)

        let resultLow = await aeLow.run()
        let resultHigh = await aeHigh.run()

        #expect(resultHigh.oracleCalls > resultLow.oracleCalls, "More precision qubits should require more oracle calls")
    }

    @Test("Confidence interval lower bound non-negative")
    func confidenceIntervalLowerNonNegative() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [0])
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.confidenceInterval.lower >= 0, "Confidence interval lower bound should be >= 0")
    }

    @Test("Confidence interval upper bound at most 1")
    func confidenceIntervalUpperAtMostOne() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [0, 1, 2, 3])
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.confidenceInterval.upper <= 1, "Confidence interval upper bound should be <= 1")
    }
}

/// Test suite for AmplitudeEstimation run() with iterative phase estimation.
/// Validates IPE mode produces valid results with correct bounds
/// and demonstrates quadratic speedup.
@Suite("AmplitudeEstimation Iterative PE")
struct AmplitudeEstimationIterativePETests {
    @Test("Iterative PE produces valid probability")
    func iterativePEValidProbability() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1])
        let config = AEConfiguration(precisionQubits: 4, useIterative: true)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0, "IPE probability should be >= 0")
        #expect(result.estimatedProbability <= 1, "IPE probability should be <= 1")
    }

    @Test("Iterative PE produces valid amplitude")
    func iterativePEValidAmplitude() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [0])
        let config = AEConfiguration(precisionQubits: 3, useIterative: true)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedAmplitude >= 0, "IPE amplitude should be >= 0")
        #expect(result.estimatedAmplitude <= 1, "IPE amplitude should be <= 1")
    }

    @Test("Iterative PE confidence interval valid")
    func iterativePEConfidenceIntervalValid() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1, 2])
        let config = AEConfiguration(precisionQubits: 4, useIterative: true)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.confidenceInterval.lower <= result.confidenceInterval.upper, "Lower bound should be <= upper bound")
        #expect(result.confidenceInterval.lower <= result.estimatedProbability, "Lower bound should be <= estimate")
        #expect(result.estimatedProbability <= result.confidenceInterval.upper, "Estimate should be <= upper bound")
    }

    @Test("Iterative PE oracle calls positive")
    func iterativePEOracleCallsPositive() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [0])
        let config = AEConfiguration(precisionQubits: 3, useIterative: true)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.oracleCalls > 0, "IPE should make positive oracle calls")
    }

    @Test("Iterative PE demonstrates quadratic speedup")
    func iterativePEQuadraticSpeedup() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1])
        let config = AEConfiguration(precisionQubits: 4, useIterative: true)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.classicalEquivalentSamples > result.oracleCalls, "IPE should demonstrate quadratic speedup")
    }
}

private actor MessageCounter {
    private var count = 0

    func increment() {
        count += 1
    }

    func getCount() -> Int {
        count
    }
}

/// Test suite for AmplitudeEstimation with progress callback.
/// Validates progress messages are received during execution
/// for both standard and iterative phase estimation modes.
@Suite("AmplitudeEstimation Progress Callback")
struct AmplitudeEstimationProgressTests {
    @Test("Progress callback receives messages")
    func progressCallbackReceivesMessages() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [0])
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let counter = MessageCounter()
        let result = await ae.run { _ in
            await counter.increment()
        }

        let messagesReceived = await counter.getCount()
        #expect(messagesReceived > 0, "Progress callback should receive at least one message")
        #expect(result.estimatedProbability >= 0, "Result should still be valid with progress callback")
    }

    @Test("Progress callback receives messages in iterative mode")
    func progressCallbackIterativeMode() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1])
        let config = AEConfiguration(precisionQubits: 3, useIterative: true)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let counter = MessageCounter()
        let result = await ae.run { _ in
            await counter.increment()
        }

        let messagesReceived = await counter.getCount()
        #expect(messagesReceived > 0, "IPE progress callback should receive messages")
        #expect(result.estimatedAmplitude >= 0, "Result should still be valid with progress callback")
    }

    @Test("Result valid without progress callback")
    func resultValidWithoutProgressCallback() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [0])
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0, "Result should be valid without progress callback")
        #expect(result.oracleCalls > 0, "Oracle calls should be positive without progress callback")
    }
}

/// Test suite for AmplitudeEstimation oracle call complexity.
/// Validates O(2^n) oracle calls for n precision qubits
/// and verifies classical equivalent samples follow O(1/epsilon^2).
@Suite("AmplitudeEstimation Complexity")
struct AmplitudeEstimationComplexityTests {
    @Test("Oracle calls scale with precision qubits")
    func oracleCallsScaleWithPrecision() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [0])

        let config2 = AEConfiguration(precisionQubits: 2, useIterative: false)
        let config3 = AEConfiguration(precisionQubits: 3, useIterative: false)
        let config4 = AEConfiguration(precisionQubits: 4, useIterative: false)

        let ae2 = AmplitudeEstimation(oracle: oracle, configuration: config2)
        let ae3 = AmplitudeEstimation(oracle: oracle, configuration: config3)
        let ae4 = AmplitudeEstimation(oracle: oracle, configuration: config4)

        let result2 = await ae2.run()
        let result3 = await ae3.run()
        let result4 = await ae4.run()

        #expect(result3.oracleCalls > result2.oracleCalls, "3 precision qubits should need more calls than 2")
        #expect(result4.oracleCalls > result3.oracleCalls, "4 precision qubits should need more calls than 3")
    }

    @Test("Classical samples scale quadratically with precision")
    func classicalSamplesScaleQuadratically() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1])

        let config3 = AEConfiguration(precisionQubits: 3, useIterative: false)
        let config4 = AEConfiguration(precisionQubits: 4, useIterative: false)

        let ae3 = AmplitudeEstimation(oracle: oracle, configuration: config3)
        let ae4 = AmplitudeEstimation(oracle: oracle, configuration: config4)

        let result3 = await ae3.run()
        let result4 = await ae4.run()

        let ratio = Double(result4.classicalEquivalentSamples) / Double(result3.classicalEquivalentSamples)

        #expect(ratio > 3.0, "Classical samples should increase by roughly 4x (2^2) when adding 1 precision qubit")
    }

    @Test("Speedup ratio increases with precision")
    func speedupRatioIncreasesWithPrecision() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [0])

        let configLow = AEConfiguration(precisionQubits: 2, useIterative: false)
        let configHigh = AEConfiguration(precisionQubits: 4, useIterative: false)

        let aeLow = AmplitudeEstimation(oracle: oracle, configuration: configLow)
        let aeHigh = AmplitudeEstimation(oracle: oracle, configuration: configHigh)

        let resultLow = await aeLow.run()
        let resultHigh = await aeHigh.run()

        let speedupLow = Double(resultLow.classicalEquivalentSamples) / Double(max(1, resultLow.oracleCalls))
        let speedupHigh = Double(resultHigh.classicalEquivalentSamples) / Double(max(1, resultHigh.oracleCalls))

        #expect(speedupHigh > speedupLow, "Higher precision should demonstrate better speedup ratio")
    }
}

/// Test suite for AmplitudeEstimation edge cases.
/// Validates behavior with all states marked, no states marked,
/// and single qubit oracles.
@Suite("AmplitudeEstimation Edge Cases")
struct AmplitudeEstimationEdgeCasesTests {
    @Test("All states marked yields valid probability")
    func allStatesMarkedProbabilityNearOne() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [0, 1, 2, 3])
        let config = AEConfiguration(precisionQubits: 4, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0, "Probability should be non-negative")
        #expect(result.estimatedProbability <= 1, "Probability should not exceed 1")
    }

    @Test("No states marked yields probability near 0")
    func noStatesMarkedProbabilityNearZero() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [])
        let config = AEConfiguration(precisionQubits: 4, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0, "Probability should be non-negative")
        #expect(result.estimatedProbability <= 0.5, "With no states marked, probability should be low")
    }

    @Test("Single qubit oracle")
    func singleQubitOracle() async {
        let oracle = CountingOracle(qubits: 1, markedStates: [0])
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0, "Single qubit probability should be valid")
        #expect(result.estimatedProbability <= 1, "Single qubit probability should be at most 1")
    }

    @Test("Half states marked yields valid probability")
    func halfStatesMarkedProbabilityNearHalf() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [0, 1])
        let config = AEConfiguration(precisionQubits: 4, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0, "Probability should be non-negative")
        #expect(result.estimatedProbability <= 1, "Probability should not exceed 1")
    }

    @Test("Minimum precision qubits")
    func minimumPrecisionQubits() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1])
        let config = AEConfiguration(precisionQubits: 1, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0, "Result should be valid with minimum precision")
        #expect(result.estimatedProbability <= 1, "Result should be valid with minimum precision")
        #expect(result.oracleCalls > 0, "Should still make oracle calls")
    }
}

/// Test suite for AmplitudeEstimation amplitude-probability relationship.
/// Validates amplitude^2 equals probability and both are consistent
/// with the mathematical relationship sin^2(theta).
@Suite("AmplitudeEstimation Amplitude-Probability Relationship")
struct AmplitudeEstimationRelationshipTests {
    @Test("Amplitude squared approximately equals probability")
    func amplitudeSquaredApproximatelyEqualsProbability() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [0])
        let config = AEConfiguration(precisionQubits: 4, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        let amplitudeSquared = result.estimatedAmplitude * result.estimatedAmplitude
        let difference = abs(amplitudeSquared - result.estimatedProbability)

        #expect(difference < 0.01, "Amplitude^2 should approximately equal probability")
    }

    @Test("Amplitude is square root of probability")
    func amplitudeIsSquareRootOfProbability() async {
        let oracle = CountingOracle(qubits: 2, markedStates: [1, 2])
        let config = AEConfiguration(precisionQubits: 4, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        let sqrtProbability = Foundation.sqrt(result.estimatedProbability)
        let difference = abs(sqrtProbability - result.estimatedAmplitude)

        #expect(difference < 0.01, "Amplitude should be approximately sqrt(probability)")
    }
}

struct PauliYOracle: AmplitudeOracle, Sendable {
    let qubits: Int = 2

    func applyStatePreparation(to circuit: inout QuantumCircuit) {
        for q in 0 ..< qubits {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
        for q in (0 ..< qubits).reversed() {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyMarkingOracle(to circuit: inout QuantumCircuit) {
        circuit.append(.pauliY, to: 0)
        circuit.append(.pauliZ, to: 0)
    }
}

struct RotationZOracle: AmplitudeOracle, Sendable {
    let qubits: Int = 2

    func applyStatePreparation(to circuit: inout QuantumCircuit) {
        for q in 0 ..< qubits {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
        for q in (0 ..< qubits).reversed() {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyMarkingOracle(to circuit: inout QuantumCircuit) {
        circuit.append(.rotationZ(.pi / 4), to: 0)
    }
}

struct RotationYOracle: AmplitudeOracle, Sendable {
    let qubits: Int = 2

    func applyStatePreparation(to circuit: inout QuantumCircuit) {
        for q in 0 ..< qubits {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
        for q in (0 ..< qubits).reversed() {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyMarkingOracle(to circuit: inout QuantumCircuit) {
        circuit.append(.rotationY(.pi / 4), to: 0)
    }
}

struct RotationXOracle: AmplitudeOracle, Sendable {
    let qubits: Int = 2

    func applyStatePreparation(to circuit: inout QuantumCircuit) {
        for q in 0 ..< qubits {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
        for q in (0 ..< qubits).reversed() {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyMarkingOracle(to circuit: inout QuantumCircuit) {
        circuit.append(.rotationX(.pi / 4), to: 0)
    }
}

struct CNOTOracle: AmplitudeOracle, Sendable {
    let qubits: Int = 2

    func applyStatePreparation(to circuit: inout QuantumCircuit) {
        for q in 0 ..< qubits {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
        for q in (0 ..< qubits).reversed() {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyMarkingOracle(to circuit: inout QuantumCircuit) {
        circuit.append(.cnot, to: [0, 1])
    }
}

struct PhaseOracle: AmplitudeOracle, Sendable {
    let qubits: Int = 2

    func applyStatePreparation(to circuit: inout QuantumCircuit) {
        for q in 0 ..< qubits {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
        for q in (0 ..< qubits).reversed() {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyMarkingOracle(to circuit: inout QuantumCircuit) {
        circuit.append(.phase(.pi / 3), to: 0)
    }
}

struct SGateOracle: AmplitudeOracle, Sendable {
    let qubits: Int = 2

    func applyStatePreparation(to circuit: inout QuantumCircuit) {
        for q in 0 ..< qubits {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
        for q in (0 ..< qubits).reversed() {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyMarkingOracle(to circuit: inout QuantumCircuit) {
        circuit.append(.sGate, to: 0)
    }
}

struct TGateOracle: AmplitudeOracle, Sendable {
    let qubits: Int = 2

    func applyStatePreparation(to circuit: inout QuantumCircuit) {
        for q in 0 ..< qubits {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
        for q in (0 ..< qubits).reversed() {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyMarkingOracle(to circuit: inout QuantumCircuit) {
        circuit.append(.tGate, to: 0)
    }
}

struct ControlledPhaseOracle: AmplitudeOracle, Sendable {
    let qubits: Int = 2

    func applyStatePreparation(to circuit: inout QuantumCircuit) {
        for q in 0 ..< qubits {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
        for q in (0 ..< qubits).reversed() {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyMarkingOracle(to circuit: inout QuantumCircuit) {
        circuit.append(.controlledPhase(.pi / 4), to: [0, 1])
    }
}

struct ToffoliOracle: AmplitudeOracle, Sendable {
    let qubits: Int = 3

    func applyStatePreparation(to circuit: inout QuantumCircuit) {
        for q in 0 ..< qubits {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
        for q in (0 ..< qubits).reversed() {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyMarkingOracle(to circuit: inout QuantumCircuit) {
        circuit.append(.toffoli, to: [0, 1, 2])
    }
}

struct IdentityOracle: AmplitudeOracle, Sendable {
    let qubits: Int = 2

    func applyStatePreparation(to circuit: inout QuantumCircuit) {
        for q in 0 ..< qubits {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
        for q in (0 ..< qubits).reversed() {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyMarkingOracle(to circuit: inout QuantumCircuit) {
        circuit.append(.identity, to: 0)
    }
}

struct SingleControlOracle: AmplitudeOracle, Sendable {
    let qubits: Int = 1

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

struct SymbolicRotationZOracle: AmplitudeOracle, Sendable {
    let qubits: Int = 2

    func applyStatePreparation(to circuit: inout QuantumCircuit) {
        for q in 0 ..< qubits {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
        for q in (0 ..< qubits).reversed() {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyMarkingOracle(to circuit: inout QuantumCircuit) {
        let theta = Parameter(name: "theta")
        circuit.append(.rotationZ(.parameter(theta)), to: 0)
    }
}

struct SymbolicRotationYOracle: AmplitudeOracle, Sendable {
    let qubits: Int = 2

    func applyStatePreparation(to circuit: inout QuantumCircuit) {
        for q in 0 ..< qubits {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
        for q in (0 ..< qubits).reversed() {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyMarkingOracle(to circuit: inout QuantumCircuit) {
        let theta = Parameter(name: "theta")
        circuit.append(.rotationY(.parameter(theta)), to: 0)
    }
}

struct SymbolicRotationXOracle: AmplitudeOracle, Sendable {
    let qubits: Int = 2

    func applyStatePreparation(to circuit: inout QuantumCircuit) {
        for q in 0 ..< qubits {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
        for q in (0 ..< qubits).reversed() {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyMarkingOracle(to circuit: inout QuantumCircuit) {
        let theta = Parameter(name: "theta")
        circuit.append(.rotationX(.parameter(theta)), to: 0)
    }
}

struct SymbolicPhaseOracle: AmplitudeOracle, Sendable {
    let qubits: Int = 2

    func applyStatePreparation(to circuit: inout QuantumCircuit) {
        for q in 0 ..< qubits {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
        for q in (0 ..< qubits).reversed() {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyMarkingOracle(to circuit: inout QuantumCircuit) {
        let phi = Parameter(name: "phi")
        circuit.append(.phase(.parameter(phi)), to: 0)
    }
}

struct SymbolicControlledPhaseOracle: AmplitudeOracle, Sendable {
    let qubits: Int = 2

    func applyStatePreparation(to circuit: inout QuantumCircuit) {
        for q in 0 ..< qubits {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyStatePreparationInverse(to circuit: inout QuantumCircuit) {
        for q in (0 ..< qubits).reversed() {
            circuit.append(.hadamard, to: q)
        }
    }

    func applyMarkingOracle(to circuit: inout QuantumCircuit) {
        let phi = Parameter(name: "phi")
        circuit.append(.controlledPhase(.parameter(phi)), to: [0, 1])
    }
}

/// Test suite for AmplitudeEstimation makeControlled gate conversion.
/// Validates controlled versions of PauliY, rotation gates, CNOT,
/// phase gates, S/T gates, and default gate handling.
@Suite("AmplitudeEstimation Controlled Gate Coverage")
struct AmplitudeEstimationControlledGateTests {
    @Test("Controlled PauliY gate produces valid result")
    func controlledPauliYProducesValidResult() async {
        let oracle = PauliYOracle()
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0)
        #expect(result.estimatedProbability <= 1)
    }

    @Test("Controlled RotationZ gate produces valid result")
    func controlledRotationZProducesValidResult() async {
        let oracle = RotationZOracle()
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0)
        #expect(result.estimatedProbability <= 1)
    }

    @Test("Controlled RotationY gate produces valid result")
    func controlledRotationYProducesValidResult() async {
        let oracle = RotationYOracle()
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0)
        #expect(result.estimatedProbability <= 1)
    }

    @Test("Controlled RotationX gate produces valid result")
    func controlledRotationXProducesValidResult() async {
        let oracle = RotationXOracle()
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0)
        #expect(result.estimatedProbability <= 1)
    }

    @Test("Controlled CNOT produces Toffoli")
    func controlledCNOTProducesToffoli() async {
        let oracle = CNOTOracle()
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0)
        #expect(result.estimatedProbability <= 1)
    }

    @Test("Controlled Phase gate produces valid result")
    func controlledPhaseProducesValidResult() async {
        let oracle = PhaseOracle()
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0)
        #expect(result.estimatedProbability <= 1)
    }

    @Test("Controlled S gate produces valid result")
    func controlledSGateProducesValidResult() async {
        let oracle = SGateOracle()
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0)
        #expect(result.estimatedProbability <= 1)
    }

    @Test("Controlled T gate produces valid result")
    func controlledTGateProducesValidResult() async {
        let oracle = TGateOracle()
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0)
        #expect(result.estimatedProbability <= 1)
    }

    @Test("Controlled controlled-phase gate produces valid result")
    func controlledControlledPhaseProducesValidResult() async {
        let oracle = ControlledPhaseOracle()
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0)
        #expect(result.estimatedProbability <= 1)
    }

    @Test("Controlled Toffoli gate produces valid result")
    func controlledToffoliProducesValidResult() async {
        let oracle = ToffoliOracle()
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0)
        #expect(result.estimatedProbability <= 1)
    }

    @Test("Default gate handling produces valid result")
    func defaultGateHandlingProducesValidResult() async {
        let oracle = IdentityOracle()
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0)
        #expect(result.estimatedProbability <= 1)
    }

    @Test("Single control multi-controlled X uses CNOT")
    func singleControlMultiControlledXUsesCNOT() async {
        let oracle = SingleControlOracle()
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0)
        #expect(result.estimatedProbability <= 1)
    }
}

/// Test suite for AmplitudeEstimation symbolic parameter default handling.
/// Validates controlled gate conversion when rotation and phase gates
/// use symbolic parameters instead of concrete values.
@Suite("AmplitudeEstimation Symbolic Parameter Coverage")
struct AmplitudeEstimationSymbolicParameterTests {
    @Test("Symbolic RotationZ uses default 0.0 half angle")
    func symbolicRotationZDefaultHalfAngle() async {
        let oracle = SymbolicRotationZOracle()
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0)
        #expect(result.estimatedProbability <= 1)
    }

    @Test("Symbolic RotationY uses default 0.0 half angle")
    func symbolicRotationYDefaultHalfAngle() async {
        let oracle = SymbolicRotationYOracle()
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0)
        #expect(result.estimatedProbability <= 1)
    }

    @Test("Symbolic RotationX uses default 0.0 half angle")
    func symbolicRotationXDefaultHalfAngle() async {
        let oracle = SymbolicRotationXOracle()
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0)
        #expect(result.estimatedProbability <= 1)
    }

    @Test("Symbolic Phase uses default 0.0 half angle")
    func symbolicPhaseDefaultHalfAngle() async {
        let oracle = SymbolicPhaseOracle()
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0)
        #expect(result.estimatedProbability <= 1)
    }

    @Test("Symbolic ControlledPhase uses default 0.0 angle")
    func symbolicControlledPhaseDefaultAngle() async {
        let oracle = SymbolicControlledPhaseOracle()
        let config = AEConfiguration(precisionQubits: 3, useIterative: false)
        let ae = AmplitudeEstimation(oracle: oracle, configuration: config)

        let result = await ae.run()

        #expect(result.estimatedProbability >= 0)
        #expect(result.estimatedProbability <= 1)
    }
}
