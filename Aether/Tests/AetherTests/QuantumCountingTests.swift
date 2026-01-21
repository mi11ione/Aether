// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under Apache 2.0

import Aether
import Foundation
import Testing

/// Test suite for QuantumCountingResult properties and formatting.
/// Validates result struct initialization, computed properties,
/// and human-readable description output for debugging.
@Suite("QuantumCountingResult Properties")
struct QuantumCountingResultPropertiesTests {
    @Test("Result stores all properties correctly")
    func resultStoresAllProperties() {
        let result = QuantumCountingResult(
            estimatedCount: 3,
            estimatedFraction: 0.1875,
            countInterval: (lower: 2, upper: 4),
            estimatedTheta: 0.4510,
            precisionQubits: 6,
            searchSpaceSize: 16,
        )

        #expect(result.estimatedCount == 3, "estimatedCount should be 3")
        #expect(abs(result.estimatedFraction - 0.1875) < 1e-10, "estimatedFraction should be 0.1875")
        #expect(result.countInterval.lower == 2, "countInterval.lower should be 2")
        #expect(result.countInterval.upper == 4, "countInterval.upper should be 4")
        #expect(abs(result.estimatedTheta - 0.4510) < 1e-10, "estimatedTheta should be 0.4510")
        #expect(result.precisionQubits == 6, "precisionQubits should be 6")
        #expect(result.searchSpaceSize == 16, "searchSpaceSize should be 16")
    }

    @Test("Result with zero count")
    func resultWithZeroCount() {
        let result = QuantumCountingResult(
            estimatedCount: 0,
            estimatedFraction: 0.0,
            countInterval: (lower: 0, upper: 0),
            estimatedTheta: 0.0,
            precisionQubits: 4,
            searchSpaceSize: 8,
        )

        #expect(result.estimatedCount == 0, "estimatedCount should be 0")
        #expect(abs(result.estimatedFraction) < 1e-10, "estimatedFraction should be 0.0")
        #expect(result.countInterval.lower == 0, "countInterval.lower should be 0")
        #expect(result.countInterval.upper == 0, "countInterval.upper should be 0")
    }

    @Test("Result with maximum count equals search space size")
    func resultWithMaximumCount() {
        let result = QuantumCountingResult(
            estimatedCount: 16,
            estimatedFraction: 1.0,
            countInterval: (lower: 15, upper: 16),
            estimatedTheta: Double.pi / 2.0,
            precisionQubits: 8,
            searchSpaceSize: 16,
        )

        #expect(result.estimatedCount == 16, "estimatedCount should equal searchSpaceSize")
        #expect(abs(result.estimatedFraction - 1.0) < 1e-10, "estimatedFraction should be 1.0")
        #expect(result.searchSpaceSize == 16, "searchSpaceSize should be 16")
    }

    @Test("Description contains count value")
    func descriptionContainsCount() {
        let result = QuantumCountingResult(
            estimatedCount: 5,
            estimatedFraction: 0.3125,
            countInterval: (lower: 4, upper: 6),
            estimatedTheta: 0.59,
            precisionQubits: 6,
            searchSpaceSize: 16,
        )

        #expect(result.description.contains("5"), "Description should contain count value")
        #expect(result.description.contains("QuantumCountingResult"), "Description should contain type name")
    }

    @Test("Description contains interval bounds")
    func descriptionContainsIntervalBounds() {
        let result = QuantumCountingResult(
            estimatedCount: 3,
            estimatedFraction: 0.1875,
            countInterval: (lower: 2, upper: 4),
            estimatedTheta: 0.45,
            precisionQubits: 6,
            searchSpaceSize: 16,
        )

        #expect(result.description.contains("2"), "Description should contain lower bound")
        #expect(result.description.contains("4"), "Description should contain upper bound")
    }

    @Test("Description contains search space size")
    func descriptionContainsSearchSpaceSize() {
        let result = QuantumCountingResult(
            estimatedCount: 2,
            estimatedFraction: 0.125,
            countInterval: (lower: 1, upper: 3),
            estimatedTheta: 0.36,
            precisionQubits: 8,
            searchSpaceSize: 16,
        )

        #expect(result.description.contains("16"), "Description should contain search space size N")
    }

    @Test("Interval lower bound is less than or equal to estimated count")
    func intervalLowerBoundConstraint() {
        let result = QuantumCountingResult(
            estimatedCount: 5,
            estimatedFraction: 0.3125,
            countInterval: (lower: 4, upper: 6),
            estimatedTheta: 0.59,
            precisionQubits: 6,
            searchSpaceSize: 16,
        )

        #expect(result.countInterval.lower <= result.estimatedCount, "Lower bound should be <= estimated count")
    }

    @Test("Interval upper bound is greater than or equal to estimated count")
    func intervalUpperBoundConstraint() {
        let result = QuantumCountingResult(
            estimatedCount: 5,
            estimatedFraction: 0.3125,
            countInterval: (lower: 4, upper: 6),
            estimatedTheta: 0.59,
            precisionQubits: 6,
            searchSpaceSize: 16,
        )

        #expect(result.countInterval.upper >= result.estimatedCount, "Upper bound should be >= estimated count")
    }
}

/// Test suite for QuantumCountingConfig initialization and validation.
/// Validates default parameter values, explicit configuration,
/// and precondition enforcement for qubit counts.
@Suite("QuantumCountingConfig Initialization")
struct QuantumCountingConfigInitializationTests {
    @Test("Default precisionQubits is 8")
    func defaultPrecisionQubits() {
        let config = QuantumCountingConfig(searchQubits: 3)

        #expect(config.precisionQubits == 8, "Default precisionQubits should be 8")
    }

    @Test("Default useIterative is false")
    func defaultUseIterative() {
        let config = QuantumCountingConfig(searchQubits: 3)

        #expect(config.useIterative == false, "Default useIterative should be false")
    }

    @Test("Explicit precisionQubits overrides default")
    func explicitPrecisionQubits() {
        let config = QuantumCountingConfig(searchQubits: 4, precisionQubits: 12)

        #expect(config.precisionQubits == 12, "Explicit precisionQubits should be 12")
    }

    @Test("Explicit useIterative overrides default")
    func explicitUseIterative() {
        let config = QuantumCountingConfig(searchQubits: 4, precisionQubits: 6, useIterative: true)

        #expect(config.useIterative == true, "Explicit useIterative should be true")
    }

    @Test("Config stores searchQubits correctly")
    func configStoresSearchQubits() {
        let config = QuantumCountingConfig(searchQubits: 5, precisionQubits: 10)

        #expect(config.searchQubits == 5, "searchQubits should be 5")
    }

    @Test("Minimum searchQubits of 1 is valid")
    func minimumSearchQubitsValid() {
        let config = QuantumCountingConfig(searchQubits: 1)

        #expect(config.searchQubits == 1, "Minimum searchQubits of 1 should be valid")
    }

    @Test("Minimum precisionQubits of 1 is valid")
    func minimumPrecisionQubitsValid() {
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 1)

        #expect(config.precisionQubits == 1, "Minimum precisionQubits of 1 should be valid")
    }

    @Test("Config with all parameters specified")
    func configWithAllParameters() {
        let config = QuantumCountingConfig(searchQubits: 4, precisionQubits: 8, useIterative: true)

        #expect(config.searchQubits == 4, "searchQubits should be 4")
        #expect(config.precisionQubits == 8, "precisionQubits should be 8")
        #expect(config.useIterative == true, "useIterative should be true")
    }
}

/// Test suite for quantum counting circuit structure.
/// Validates circuit qubit count, gate sequence composition,
/// and proper initialization of precision and search registers.
@Suite("QuantumCounting Circuit Structure")
struct QuantumCountingCircuitStructureTests {
    @Test("Circuit qubit count equals precisionQubits plus searchQubits")
    func circuitQubitCount() {
        let oracle = GroverOracle.singleTarget(1)
        let config = QuantumCountingConfig(searchQubits: 3, precisionQubits: 5)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)

        #expect(circuit.qubits == 8, "Circuit qubits should be precisionQubits + searchQubits = 5 + 3 = 8")
    }

    @Test("Circuit qubit count with default precision")
    func circuitQubitCountDefaultPrecision() {
        let oracle = GroverOracle.singleTarget(0)
        let config = QuantumCountingConfig(searchQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)

        #expect(circuit.qubits == 10, "Circuit qubits should be 8 + 2 = 10 with default precision")
    }

    @Test("Circuit qubit count with single search qubit")
    func circuitQubitCountSingleSearch() {
        let oracle = GroverOracle.singleTarget(0)
        let config = QuantumCountingConfig(searchQubits: 1, precisionQubits: 4)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)

        #expect(circuit.qubits == 5, "Circuit qubits should be 4 + 1 = 5")
    }

    @Test("Circuit contains gates for initialization")
    func circuitContainsInitializationGates() {
        let oracle = GroverOracle.singleTarget(1)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 3)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)

        #expect(circuit.gates.count > 0, "Circuit should contain gates")
    }

    @Test("Circuit with multiple target oracle")
    func circuitWithMultipleTargetOracle() {
        let oracle = GroverOracle.multipleTargets([0, 1])
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 4)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)

        #expect(circuit.qubits == 6, "Circuit qubits should be 4 + 2 = 6")
    }

    @Test("Circuit qubit count scales with precision")
    func circuitQubitCountScalesWithPrecision() {
        let oracle = GroverOracle.singleTarget(0)
        let config4 = QuantumCountingConfig(searchQubits: 2, precisionQubits: 4)
        let config6 = QuantumCountingConfig(searchQubits: 2, precisionQubits: 6)

        let circuit4 = QuantumCircuit.quantumCounting(oracle: oracle, config: config4)
        let circuit6 = QuantumCircuit.quantumCounting(oracle: oracle, config: config6)

        #expect(circuit6.qubits - circuit4.qubits == 2, "Qubit count difference should equal precision difference")
    }
}

/// Test suite for quantum counting result extraction from quantum state.
/// Validates estimated count accuracy, fraction computation,
/// and confidence interval bounds for various marked state configurations.
@Suite("QuantumCounting Result Extraction")
struct QuantumCountingResultExtractionTests {
    @Test("Two search qubits with one marked state estimates count near 1")
    func twoSearchQubitsOneMarkedState() {
        let oracle = GroverOracle.singleTarget(1)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 6)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.countInterval.lower <= 1, "Lower bound should be <= 1")
        #expect(result.countInterval.upper >= 1, "Upper bound should be >= 1")
        #expect(result.searchSpaceSize == 4, "Search space size should be 2^2 = 4")
    }

    @Test("Three search qubits with two marked states estimates count near 2")
    func threeSearchQubitsTwoMarkedStates() {
        let oracle = GroverOracle.multipleTargets([0, 3])
        let config = QuantumCountingConfig(searchQubits: 3, precisionQubits: 6)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.countInterval.lower <= 2, "Lower bound should be <= 2")
        #expect(result.countInterval.upper >= 2, "Upper bound should be >= 2")
        #expect(result.searchSpaceSize == 8, "Search space size should be 2^3 = 8")
    }

    @Test("Estimated fraction equals estimated count divided by search space size")
    func estimatedFractionComputation() {
        let oracle = GroverOracle.singleTarget(0)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 6)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        let expectedFraction = Double(result.estimatedCount) / Double(result.searchSpaceSize)
        #expect(abs(result.estimatedFraction - expectedFraction) < 0.5, "Fraction should approximately equal count/N")
    }

    @Test("Search space size equals 2 to the power of search qubits")
    func searchSpaceSizeComputation() {
        let oracle = GroverOracle.singleTarget(0)
        let config = QuantumCountingConfig(searchQubits: 3, precisionQubits: 4)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.searchSpaceSize == 8, "Search space size should be 2^3 = 8")
    }

    @Test("Count interval lower bound is less than or equal to estimated count")
    func countIntervalLowerBound() {
        let oracle = GroverOracle.singleTarget(2)
        let config = QuantumCountingConfig(searchQubits: 3, precisionQubits: 6)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.countInterval.lower <= result.estimatedCount, "Lower bound should be <= estimated count")
    }

    @Test("Count interval upper bound is greater than or equal to estimated count")
    func countIntervalUpperBound() {
        let oracle = GroverOracle.singleTarget(2)
        let config = QuantumCountingConfig(searchQubits: 3, precisionQubits: 6)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.countInterval.upper >= result.estimatedCount, "Upper bound should be >= estimated count")
    }

    @Test("Precision qubits in result matches config")
    func precisionQubitsInResult() {
        let oracle = GroverOracle.singleTarget(0)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 7)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.precisionQubits == 7, "Precision qubits in result should match config")
    }

    @Test("Estimated theta is non-negative")
    func estimatedThetaNonNegative() {
        let oracle = GroverOracle.singleTarget(1)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 6)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.estimatedTheta >= 0, "Estimated theta should be non-negative")
    }

    @Test("Estimated theta is at most pi")
    func estimatedThetaAtMostPi() {
        let oracle = GroverOracle.singleTarget(1)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 6)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.estimatedTheta <= Double.pi, "Estimated theta should be at most pi")
    }

    @Test("Estimated count is non-negative")
    func estimatedCountNonNegative() {
        let oracle = GroverOracle.singleTarget(0)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 6)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.estimatedCount >= 0, "Estimated count should be non-negative")
    }

    @Test("Estimated count does not exceed search space size")
    func estimatedCountWithinBounds() {
        let oracle = GroverOracle.singleTarget(0)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 6)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.estimatedCount <= result.searchSpaceSize, "Estimated count should not exceed N")
    }

    @Test("Estimated fraction is between 0 and 1")
    func estimatedFractionInRange() {
        let oracle = GroverOracle.singleTarget(0)
        let config = QuantumCountingConfig(searchQubits: 3, precisionQubits: 6)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.estimatedFraction >= 0.0, "Estimated fraction should be >= 0")
        #expect(result.estimatedFraction <= 1.0, "Estimated fraction should be <= 1")
    }

    @Test("Count interval lower bound is non-negative")
    func countIntervalLowerBoundNonNegative() {
        let oracle = GroverOracle.singleTarget(0)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 6)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.countInterval.lower >= 0, "Count interval lower bound should be non-negative")
    }

    @Test("Count interval upper bound does not exceed search space size")
    func countIntervalUpperBoundWithinBounds() {
        let oracle = GroverOracle.singleTarget(0)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 6)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.countInterval.upper <= result.searchSpaceSize, "Count interval upper bound should not exceed N")
    }
}

/// Test suite for quantum counting with various oracle configurations.
/// Validates algorithm behavior with single target, multiple targets,
/// and different search space sizes.
@Suite("QuantumCounting Oracle Variations")
struct QuantumCountingOracleVariationsTests {
    @Test("Single target at state 0")
    func singleTargetAtStateZero() {
        let oracle = GroverOracle.singleTarget(0)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 6)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.countInterval.lower <= 1, "Should detect approximately 1 marked state")
        #expect(result.countInterval.upper >= 1, "Should detect approximately 1 marked state")
    }

    @Test("Single target at maximum state index")
    func singleTargetAtMaxState() {
        let oracle = GroverOracle.singleTarget(3)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 6)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.countInterval.lower <= 1, "Should detect approximately 1 marked state")
        #expect(result.countInterval.upper >= 1, "Should detect approximately 1 marked state")
    }

    @Test("Multiple targets with adjacent states")
    func multipleTargetsAdjacentStates() {
        let oracle = GroverOracle.multipleTargets([0, 1])
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 6)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.countInterval.lower <= 2, "Should detect approximately 2 marked states")
        #expect(result.countInterval.upper >= 2, "Should detect approximately 2 marked states")
    }

    @Test("Multiple targets with non-adjacent states")
    func multipleTargetsNonAdjacentStates() {
        let oracle = GroverOracle.multipleTargets([0, 3])
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 6)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.countInterval.lower <= 2, "Should detect approximately 2 marked states")
        #expect(result.countInterval.upper >= 2, "Should detect approximately 2 marked states")
    }

    @Test("Larger search space with single target")
    func largerSearchSpaceSingleTarget() {
        let oracle = GroverOracle.singleTarget(5)
        let config = QuantumCountingConfig(searchQubits: 3, precisionQubits: 4)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.searchSpaceSize == 8, "Search space should be 2^3 = 8")
        #expect(result.countInterval.lower <= 1, "Should detect approximately 1 marked state")
    }

    @Test("Three targets in 8-state space")
    func threeTargetsInEightStateSpace() {
        let oracle = GroverOracle.multipleTargets([1, 3, 5])
        let config = QuantumCountingConfig(searchQubits: 3, precisionQubits: 6)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.searchSpaceSize == 8, "Search space should be 2^3 = 8")
        #expect(result.countInterval.lower <= 3, "Should detect approximately 3 marked states")
    }
}

/// Test suite for quantum counting precision scaling.
/// Validates that increasing precision qubits improves
/// count estimation accuracy and tightens confidence intervals.
@Suite("QuantumCounting Precision Scaling")
struct QuantumCountingPrecisionScalingTests {
    @Test("Higher precision yields smaller interval width on average")
    func higherPrecisionSmallerInterval() {
        let oracle = GroverOracle.singleTarget(1)

        let configLow = QuantumCountingConfig(searchQubits: 2, precisionQubits: 4)
        let circuitLow = QuantumCircuit.quantumCounting(oracle: oracle, config: configLow)
        let stateLow = circuitLow.execute()
        let resultLow = stateLow.quantumCountingResult(config: configLow)
        let intervalWidthLow = resultLow.countInterval.upper - resultLow.countInterval.lower

        let configHigh = QuantumCountingConfig(searchQubits: 2, precisionQubits: 8)
        let circuitHigh = QuantumCircuit.quantumCounting(oracle: oracle, config: configHigh)
        let stateHigh = circuitHigh.execute()
        let resultHigh = stateHigh.quantumCountingResult(config: configHigh)
        let intervalWidthHigh = resultHigh.countInterval.upper - resultHigh.countInterval.lower

        #expect(intervalWidthHigh <= intervalWidthLow, "Higher precision should yield tighter interval")
    }

    @Test("Precision qubits affect circuit depth")
    func precisionAffectsCircuitDepth() {
        let oracle = GroverOracle.singleTarget(0)

        let config4 = QuantumCountingConfig(searchQubits: 2, precisionQubits: 4)
        let config6 = QuantumCountingConfig(searchQubits: 2, precisionQubits: 6)

        let circuit4 = QuantumCircuit.quantumCounting(oracle: oracle, config: config4)
        let circuit6 = QuantumCircuit.quantumCounting(oracle: oracle, config: config6)

        #expect(circuit6.gates.count > circuit4.gates.count, "More precision qubits should yield more gates")
    }

    @Test("Minimum precision of 1 produces valid result")
    func minimumPrecisionProducesValidResult() {
        let oracle = GroverOracle.singleTarget(0)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 1)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.estimatedCount >= 0, "Count should be non-negative")
        #expect(result.estimatedCount <= result.searchSpaceSize, "Count should not exceed N")
    }
}

/// Test suite for edge cases in quantum counting.
/// Validates behavior with minimal qubit counts, boundary conditions,
/// and special configurations that stress the algorithm.
@Suite("QuantumCounting Edge Cases")
struct QuantumCountingEdgeCasesTests {
    @Test("Single search qubit with single target")
    func singleSearchQubitSingleTarget() {
        let oracle = GroverOracle.singleTarget(0)
        let config = QuantumCountingConfig(searchQubits: 1, precisionQubits: 4)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.searchSpaceSize == 2, "Search space should be 2^1 = 2")
        #expect(result.estimatedCount >= 0, "Count should be non-negative")
        #expect(result.estimatedCount <= 2, "Count should not exceed N")
    }

    @Test("All states marked in small space")
    func allStatesMarked() {
        let oracle = GroverOracle.multipleTargets([0, 1])
        let config = QuantumCountingConfig(searchQubits: 1, precisionQubits: 4)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.searchSpaceSize == 2, "Search space should be 2")
    }

    @Test("Half of states marked")
    func halfStatesMarked() {
        let oracle = GroverOracle.multipleTargets([0, 1])
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 6)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.searchSpaceSize == 4, "Search space should be 4")
        #expect(result.countInterval.lower <= 2, "Should detect approximately 2 marked states")
    }

    @Test("Circuit is executable and produces valid amplitudes")
    func circuitProducesValidAmplitudes() {
        let oracle = GroverOracle.singleTarget(0)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 4)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Total probability should be 1.0")
    }
}

/// Test suite for controlled gate decomposition in quantum counting.
/// Validates that custom oracles with various gate types are correctly
/// transformed into controlled versions during circuit construction.
@Suite("QuantumCounting Controlled Gate Coverage")
struct QuantumCountingControlledGateCoverageTests {
    @Test("Custom oracle with pauliY gate produces valid circuit")
    func customOracleWithPauliY() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.pauliY, [0]),
            (.pauliZ, [0]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 1, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with pauliY oracle should preserve normalization")
    }

    @Test("Custom oracle with phase gate produces valid circuit")
    func customOracleWithPhase() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.phase(.pi / 3), [0]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 1, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with phase oracle should preserve normalization")
    }

    @Test("Custom oracle with rotationZ gate produces valid circuit")
    func customOracleWithRotationZ() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.rotationZ(.pi / 4), [0]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 1, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with rotationZ oracle should preserve normalization")
    }

    @Test("Custom oracle with rotationY gate produces valid circuit")
    func customOracleWithRotationY() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.rotationY(.pi / 5), [0]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 1, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with rotationY oracle should preserve normalization")
    }

    @Test("Custom oracle with rotationX gate produces valid circuit")
    func customOracleWithRotationX() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.rotationX(.pi / 6), [0]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 1, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with rotationX oracle should preserve normalization")
    }

    @Test("Custom oracle with sGate produces valid circuit")
    func customOracleWithSGate() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.sGate, [0]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 1, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with sGate oracle should preserve normalization")
    }

    @Test("Custom oracle with tGate produces valid circuit")
    func customOracleWithTGate() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.tGate, [0]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 1, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with tGate oracle should preserve normalization")
    }

    @Test("Custom oracle with cnot gate produces valid circuit")
    func customOracleWithCnot() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.cnot, [0, 1]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with cnot oracle should preserve normalization")
    }

    @Test("Custom oracle with cz gate produces valid circuit")
    func customOracleWithCz() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.cz, [0, 1]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with cz oracle should preserve normalization")
    }

    @Test("Custom oracle with cy gate produces valid circuit")
    func customOracleWithCy() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.cy, [0, 1]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with cy oracle should preserve normalization")
    }

    @Test("Custom oracle with swap gate produces valid circuit")
    func customOracleWithSwap() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.swap, [0, 1]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with swap oracle should preserve normalization")
    }

    @Test("Custom oracle with controlledPhase gate produces valid circuit")
    func customOracleWithControlledPhase() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.controlledPhase(.pi / 4), [0, 1]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with controlledPhase oracle should preserve normalization")
    }

    @Test("Custom oracle with toffoli gate produces valid circuit")
    func customOracleWithToffoli() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.toffoli, [0, 1, 2]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 3, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with toffoli oracle should preserve normalization")
    }

    @Test("Custom oracle with identity gate exercises default case")
    func customOracleWithIdentity() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.identity, [0]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 1, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with identity oracle should preserve normalization")
    }

    @Test("Custom oracle with multiple mixed gates produces valid circuit")
    func customOracleWithMixedGates() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.pauliY, [0]),
            (.rotationX(.pi / 8), [0]),
            (.sGate, [0]),
            (.phase(.pi / 6), [0]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 1, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with mixed gate oracle should preserve normalization")
    }

    @Test("Custom oracle with two-qubit gates produces result with valid interval")
    func customOracleWithTwoQubitGatesProducesValidInterval() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.cnot, [0, 1]),
            (.cy, [0, 1]),
            (.swap, [0, 1]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 3)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()
        let result = state.quantumCountingResult(config: config)

        #expect(result.countInterval.lower >= 0, "Lower bound should be non-negative")
        #expect(result.countInterval.upper <= result.searchSpaceSize, "Upper bound should not exceed search space")
    }
}

/// Test suite for extractConcreteValue function parameter handling.
/// Validates that symbolic parameters are handled correctly when
/// extracting concrete values from ParameterValue instances.
@Suite("QuantumCounting Parameter Extraction")
struct QuantumCountingParameterExtractionTests {
    @Test("Circuit with symbolic parameter oracle executes correctly")
    func circuitWithSymbolicParameterOracle() {
        let param = Parameter(name: "theta")
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.rotationZ(.parameter(param)), [0]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 1, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with symbolic parameter should still execute")
    }

    @Test("Circuit with negated parameter oracle executes correctly")
    func circuitWithNegatedParameterOracle() {
        let param = Parameter(name: "phi")
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.phase(.negatedParameter(param)), [0]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 1, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with negated parameter should still execute")
    }

    @Test("Circuit with rotationY symbolic parameter executes correctly")
    func circuitWithRotationYSymbolicParameter() {
        let param = Parameter(name: "alpha")
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.rotationY(.parameter(param)), [0]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 1, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with rotationY symbolic parameter should execute")
    }

    @Test("Circuit with rotationX symbolic parameter executes correctly")
    func circuitWithRotationXSymbolicParameter() {
        let param = Parameter(name: "beta")
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.rotationX(.parameter(param)), [0]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 1, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with rotationX symbolic parameter should execute")
    }

    @Test("Circuit with controlledPhase symbolic parameter executes correctly")
    func circuitWithControlledPhaseSymbolicParameter() {
        let param = Parameter(name: "gamma")
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.controlledPhase(.parameter(param)), [0, 1]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 2, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with controlledPhase symbolic parameter should execute")
    }

    @Test("Circuit with multiple symbolic parameters executes correctly")
    func circuitWithMultipleSymbolicParameters() {
        let param1 = Parameter(name: "theta1")
        let param2 = Parameter(name: "theta2")
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.rotationZ(.parameter(param1)), [0]),
            (.phase(.negatedParameter(param2)), [0]),
        ]
        let oracle = GroverOracle.custom(customGates)
        let config = QuantumCountingConfig(searchQubits: 1, precisionQubits: 2)
        let circuit = QuantumCircuit.quantumCounting(oracle: oracle, config: config)
        let state = circuit.execute()

        var totalProbability = 0.0
        for i in 0 ..< state.stateSpaceSize {
            totalProbability += state.probability(of: i)
        }

        #expect(abs(totalProbability - 1.0) < 1e-10, "Circuit with multiple symbolic parameters should execute")
    }
}
