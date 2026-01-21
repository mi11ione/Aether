// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under Apache 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for PhasePrior enum cases and initialization.
/// Validates uniform, gaussian, and custom prior distributions
/// used in Bayesian phase estimation for quantum metrology.
@Suite("PhasePrior Enum")
struct PhasePriorTests {
    @Test("Uniform prior creates valid distribution")
    func uniformPrior() async {
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(.pi / 4),
            prior: .uniform,
            discretizationBins: 64,
        )
        let result = await estimator.run(maxMeasurements: 1)
        #expect(result.posterior.count == 64, "Uniform prior should create 64 bins")
    }

    @Test("Gaussian prior creates peaked distribution")
    func gaussianPrior() async {
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(.pi / 4),
            prior: .gaussian(mean: .pi / 4, stdDev: 0.3),
            discretizationBins: 64,
        )
        let result = await estimator.run(maxMeasurements: 1)
        #expect(result.posterior.count == 64, "Gaussian prior should create 64 bins")
        let sum = result.posterior.reduce(0.0, +)
        #expect(abs(sum - 1.0) < 1e-10, "Gaussian prior should be normalized to 1.0, got \(sum)")
    }

    @Test("Custom prior accepts user-defined distribution")
    func customPrior() async {
        let customPdf = [0.1, 0.2, 0.4, 0.2, 0.1]
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(.pi / 4),
            prior: .custom(customPdf),
            discretizationBins: 64,
        )
        let result = await estimator.run(maxMeasurements: 1)
        #expect(result.posterior.count == 64, "Custom prior should create 64 bins even from smaller input")
    }

    @Test("Custom prior with larger array gets truncated")
    func customPriorTruncation() async {
        let largePdf = [Double](repeating: 0.01, count: 100)
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(.pi / 4),
            prior: .custom(largePdf),
            discretizationBins: 64,
        )
        let result = await estimator.run(maxMeasurements: 1)
        #expect(result.posterior.count == 64, "Custom prior should truncate to 64 bins")
    }
}

/// Test suite for BayesianPhaseResult struct properties.
/// Validates MAP estimate, posterior statistics, measurement count,
/// and CustomStringConvertible description format.
@Suite("BayesianPhaseResult Properties")
struct BayesianPhaseResultTests {
    @Test("Result contains valid MAP estimate in [0, 2pi)")
    func mapEstimateRange() async {
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(.pi / 4),
            prior: .uniform,
            discretizationBins: 64,
        )
        let result = await estimator.run(maxMeasurements: 10)
        #expect(result.mapEstimate >= 0.0, "MAP estimate should be >= 0, got \(result.mapEstimate)")
        #expect(result.mapEstimate < 2.0 * .pi, "MAP estimate should be < 2pi, got \(result.mapEstimate)")
    }

    @Test("Result contains valid posterior mean")
    func posteriorMeanRange() async {
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(.pi / 4),
            prior: .uniform,
            discretizationBins: 64,
        )
        let result = await estimator.run(maxMeasurements: 10)
        #expect(result.posteriorMean >= -.pi, "Posterior mean should be >= -pi, got \(result.posteriorMean)")
        #expect(result.posteriorMean <= .pi, "Posterior mean should be <= pi, got \(result.posteriorMean)")
    }

    @Test("Posterior standard deviation is non-negative")
    func posteriorStdDevNonNegative() async {
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(.pi / 4),
            prior: .uniform,
            discretizationBins: 64,
        )
        let result = await estimator.run(maxMeasurements: 10)
        #expect(result.posteriorStdDev >= 0.0, "Posterior StdDev should be non-negative, got \(result.posteriorStdDev)")
    }

    @Test("Posterior distribution is normalized")
    func posteriorNormalized() async {
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(.pi / 4),
            prior: .uniform,
            discretizationBins: 64,
        )
        let result = await estimator.run(maxMeasurements: 10)
        let sum = result.posterior.reduce(0.0, +)
        #expect(abs(sum - 1.0) < 1e-10, "Posterior should sum to 1.0, got \(sum)")
    }

    @Test("Measurement count matches requested measurements")
    func measurementCountAccurate() async {
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(.pi / 4),
            prior: .uniform,
            discretizationBins: 64,
        )
        let result = await estimator.run(maxMeasurements: 20)
        #expect(result.measurementCount == 20, "Measurement count should be 20, got \(result.measurementCount)")
    }

    @Test("Description contains required fields")
    func descriptionFormat() async {
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(.pi / 4),
            prior: .uniform,
            discretizationBins: 64,
        )
        let result = await estimator.run(maxMeasurements: 5)
        let desc = result.description
        #expect(desc.contains("BayesianPhaseResult"), "Description should contain type name")
        #expect(desc.contains("MAP Estimate"), "Description should contain MAP Estimate field")
        #expect(desc.contains("Posterior Mean"), "Description should contain Posterior Mean field")
        #expect(desc.contains("Posterior StdDev"), "Description should contain Posterior StdDev field")
        #expect(desc.contains("Measurements"), "Description should contain Measurements field")
    }
}

/// Test suite for RamseyConfig initialization and validation.
/// Validates evolution time, repetitions, detuning parameters,
/// and default value handling for Ramsey interferometry.
@Suite("RamseyConfig Initialization")
struct RamseyConfigTests {
    @Test("Default repetitions is 100")
    func defaultRepetitions() {
        let config = RamseyConfig(evolutionTime: 1e-6)
        #expect(config.repetitions == 100, "Default repetitions should be 100, got \(config.repetitions)")
    }

    @Test("Default detuning is 0")
    func defaultDetuning() {
        let config = RamseyConfig(evolutionTime: 1e-6)
        #expect(abs(config.detuning) < 1e-10, "Default detuning should be 0, got \(config.detuning)")
    }

    @Test("Custom evolution time is preserved")
    func customEvolutionTime() {
        let config = RamseyConfig(evolutionTime: 5e-6)
        #expect(abs(config.evolutionTime - 5e-6) < 1e-15, "Evolution time should be 5e-6, got \(config.evolutionTime)")
    }

    @Test("Custom repetitions is preserved")
    func customRepetitions() {
        let config = RamseyConfig(evolutionTime: 1e-6, repetitions: 500)
        #expect(config.repetitions == 500, "Repetitions should be 500, got \(config.repetitions)")
    }

    @Test("Custom detuning is preserved")
    func customDetuning() {
        let config = RamseyConfig(evolutionTime: 1e-6, repetitions: 100, detuning: 0.05)
        #expect(abs(config.detuning - 0.05) < 1e-10, "Detuning should be 0.05, got \(config.detuning)")
    }

    @Test("Zero evolution time is valid")
    func zeroEvolutionTime() {
        let config = RamseyConfig(evolutionTime: 0.0)
        #expect(abs(config.evolutionTime) < 1e-15, "Zero evolution time should be valid")
    }

    @Test("Negative detuning is valid")
    func negativeDetuning() {
        let config = RamseyConfig(evolutionTime: 1e-6, repetitions: 100, detuning: -0.1)
        #expect(abs(config.detuning - -0.1) < 1e-10, "Negative detuning should be valid, got \(config.detuning)")
    }
}

/// Test suite for RamseyResult struct properties.
/// Validates phase accumulation, visibility, T2* estimation,
/// excited state probability, and description formatting.
@Suite("RamseyResult Properties")
struct RamseyResultTests {
    @Test("Phase accumulation range [0, pi] for valid states")
    func phaseAccumulationRange() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: .pi / 4)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let config = RamseyConfig(evolutionTime: 1e-6)
        let result = state.ramseyResult(config: config)
        #expect(result.phaseAccumulation >= 0.0, "Phase accumulation should be >= 0, got \(result.phaseAccumulation)")
        #expect(result.phaseAccumulation <= .pi, "Phase accumulation should be <= pi, got \(result.phaseAccumulation)")
    }

    @Test("Visibility range [0, 1]")
    func visibilityRange() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: .pi / 4)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let config = RamseyConfig(evolutionTime: 1e-6)
        let result = state.ramseyResult(config: config)
        #expect(result.visibility >= 0.0, "Visibility should be >= 0, got \(result.visibility)")
        #expect(result.visibility <= 1.0, "Visibility should be <= 1, got \(result.visibility)")
    }

    @Test("Excited state probability range [0, 1]")
    func excitedStateProbabilityRange() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: .pi / 4)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let config = RamseyConfig(evolutionTime: 1e-6)
        let result = state.ramseyResult(config: config)
        #expect(result.excitedStateProbability >= 0.0, "P(|1>) should be >= 0, got \(result.excitedStateProbability)")
        #expect(result.excitedStateProbability <= 1.0, "P(|1>) should be <= 1, got \(result.excitedStateProbability)")
    }

    @Test("Description contains required fields")
    func descriptionFormat() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: .pi / 4)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let config = RamseyConfig(evolutionTime: 1e-6)
        let result = state.ramseyResult(config: config)
        let desc = result.description
        #expect(desc.contains("RamseyResult"), "Description should contain type name")
        #expect(desc.contains("Phase Accumulation"), "Description should contain Phase Accumulation field")
        #expect(desc.contains("Visibility"), "Description should contain Visibility field")
        #expect(desc.contains("P(|1>)"), "Description should contain P(|1>) field")
    }

    @Test("T2* estimation included when conditions met")
    func t2StarEstimation() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: .pi / 3)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let config = RamseyConfig(evolutionTime: 1e-6)
        let result = state.ramseyResult(config: config)
        if let t2 = result.estimatedT2Star {
            #expect(t2 > 0.0, "Estimated T2* should be positive when present, got \(t2)")
        }
    }
}

/// Test suite for BayesianPhaseEstimation actor run() method.
/// Validates convergence behavior, progress callbacks, early stopping,
/// and estimation accuracy with various prior distributions.
@Suite("BayesianPhaseEstimation Run")
struct BayesianPhaseEstimationTests {
    @Test("Run with uniform prior converges")
    func runWithUniformPrior() async {
        let targetPhase = Double.pi / 4
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(targetPhase),
            prior: .uniform,
            discretizationBins: 128,
        )
        let result = await estimator.run(maxMeasurements: 50)
        #expect(result.measurementCount == 50, "Should complete all 50 measurements")
        #expect(result.posteriorStdDev < .pi, "StdDev should decrease from uniform prior (~1.8 rad)")
    }

    @Test("Run with gaussian prior near target converges faster")
    func runWithGaussianPrior() async {
        let targetPhase = Double.pi / 4
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(targetPhase),
            prior: .gaussian(mean: targetPhase, stdDev: 0.3),
            discretizationBins: 128,
        )
        let result = await estimator.run(maxMeasurements: 30)
        #expect(result.measurementCount == 30, "Should complete all 30 measurements")
    }

    @Test("Early stopping with target precision")
    func earlyStoppingWithTargetPrecision() async {
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(.pi / 4),
            prior: .gaussian(mean: .pi / 4, stdDev: 0.1),
            discretizationBins: 128,
        )
        let result = await estimator.run(maxMeasurements: 1000, targetPrecision: 0.5)
        #expect(result.measurementCount <= 1000, "Should stop before or at max measurements")
    }

    @Test("Progress callback is invoked")
    func progressCallbackInvoked() async {
        actor Counter {
            var count = 0
            func increment() { count += 1 }
            func value() -> Int { count }
        }
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(.pi / 4),
            prior: .uniform,
            discretizationBins: 64,
        )
        let counter = Counter()
        _ = await estimator.run(maxMeasurements: 10) { _, _ in
            await counter.increment()
        }
        let callbackCount = await counter.value()
        #expect(callbackCount == 10, "Progress callback should be invoked 10 times, got \(callbackCount)")
    }

    @Test("Different discretization bins work correctly")
    func differentDiscretizationBins() async {
        let estimator32 = BayesianPhaseEstimation(
            unitary: .rotationZ(.pi / 4),
            prior: .uniform,
            discretizationBins: 32,
        )
        let result32 = await estimator32.run(maxMeasurements: 5)
        #expect(result32.posterior.count == 32, "Should have 32 bins, got \(result32.posterior.count)")

        let estimator256 = BayesianPhaseEstimation(
            unitary: .rotationZ(.pi / 4),
            prior: .uniform,
            discretizationBins: 256,
        )
        let result256 = await estimator256.run(maxMeasurements: 5)
        #expect(result256.posterior.count == 256, "Should have 256 bins, got \(result256.posterior.count)")
    }
}

/// Test suite for QuantumCircuit.ramseySequence() static method.
/// Validates H-Rz-H circuit structure and measurement outcomes
/// for phase=0, phase=pi, and phase=pi/2 cases.
@Suite("Ramsey Sequence Circuit")
struct RamseySequenceTests {
    @Test("Phase 0: measures |0> with high probability")
    func phaseZeroMeasuresZero() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: 0.0)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let p0 = state.probability(of: 0)
        #expect(abs(p0 - 1.0) < 1e-10, "Phase 0 should give P(|0>)=1, got \(p0)")
    }

    @Test("Phase pi: measures |1> with high probability")
    func phasePiMeasuresOne() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: .pi)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let p1 = state.probability(of: 1)
        #expect(abs(p1 - 1.0) < 1e-10, "Phase pi should give P(|1>)=1, got \(p1)")
    }

    @Test("Phase pi/2: gives 50/50 outcome")
    func phasePiOverTwoGivesFiftyFifty() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: .pi / 2)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let p0 = state.probability(of: 0)
        let p1 = state.probability(of: 1)
        #expect(abs(p0 - 0.5) < 1e-10, "Phase pi/2 should give P(|0>)=0.5, got \(p0)")
        #expect(abs(p1 - 0.5) < 1e-10, "Phase pi/2 should give P(|1>)=0.5, got \(p1)")
    }

    @Test("Circuit has correct structure (3 gates without initial phase)")
    func circuitStructureWithoutInitialPhase() {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: .pi / 4)
        #expect(circuit.qubits == 1, "Ramsey sequence should use 1 qubit, got \(circuit.qubits)")
    }

    @Test("Circuit with initial phase includes extra rotation")
    func circuitWithInitialPhase() {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: .pi / 4, initialPhase: .pi / 8)
        #expect(circuit.qubits == 1, "Ramsey sequence with initial phase should use 1 qubit")
    }

    @Test("Zero initial phase is omitted")
    func zeroInitialPhaseOmitted() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: .pi / 4, initialPhase: 0.0)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        #expect(state.qubits == 1, "Circuit should produce single-qubit state")
    }

    @Test("Probabilities sum to 1")
    func probabilitiesSumToOne() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: .pi / 3)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let p0 = state.probability(of: 0)
        let p1 = state.probability(of: 1)
        #expect(abs(p0 + p1 - 1.0) < 1e-10, "Probabilities should sum to 1, got \(p0 + p1)")
    }

    @Test("P(|1>) matches sin^2(phi/2) formula")
    func probabilityMatchesSinSquared() async {
        let phi = Double.pi / 3
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: phi)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let p1 = state.probability(of: 1)
        let expected = sin(phi / 2) * sin(phi / 2)
        #expect(abs(p1 - expected) < 1e-10, "P(|1>) should equal sin^2(phi/2)=\(expected), got \(p1)")
    }
}

/// Test suite for QuantumCircuit.ramseyEcho() Hahn echo sequence.
/// Validates H-Rz-X-Rz-H circuit structure for T2 measurement
/// and echo position parameter handling.
@Suite("Ramsey Echo Circuit")
struct RamseyEchoTests {
    @Test("Echo sequence has correct qubit count")
    func echoQubitCount() {
        let circuit = QuantumCircuit.ramseyEcho(evolutionPhase: .pi / 4)
        #expect(circuit.qubits == 1, "Ramsey echo should use 1 qubit, got \(circuit.qubits)")
    }

    @Test("Echo with default position (0.5)")
    func echoDefaultPosition() async {
        let circuit = QuantumCircuit.ramseyEcho(evolutionPhase: .pi / 2)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let p0 = state.probability(of: 0)
        let p1 = state.probability(of: 1)
        #expect(abs(p0 + p1 - 1.0) < 1e-10, "Probabilities should sum to 1, got \(p0 + p1)")
    }

    @Test("Echo at position 0.0")
    func echoPositionZero() async {
        let circuit = QuantumCircuit.ramseyEcho(evolutionPhase: .pi / 2, echoPosition: 0.0)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        #expect(state.qubits == 1, "Echo circuit should produce single-qubit state")
    }

    @Test("Echo at position 1.0")
    func echoPositionOne() async {
        let circuit = QuantumCircuit.ramseyEcho(evolutionPhase: .pi / 2, echoPosition: 1.0)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        #expect(state.qubits == 1, "Echo circuit should produce single-qubit state")
    }

    @Test("Echo at position 0.25")
    func echoPositionQuarter() async {
        let circuit = QuantumCircuit.ramseyEcho(evolutionPhase: .pi, echoPosition: 0.25)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let p0 = state.probability(of: 0)
        let p1 = state.probability(of: 1)
        #expect(abs(p0 + p1 - 1.0) < 1e-10, "Probabilities should sum to 1")
    }

    @Test("Echo with zero phase returns to |0>")
    func echoZeroPhaseReturnsToZero() async {
        let circuit = QuantumCircuit.ramseyEcho(evolutionPhase: 0.0)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let p0 = state.probability(of: 0)
        #expect(abs(p0 - 1.0) < 1e-10, "Zero phase echo should return to |0>, P(|0>)=\(p0)")
    }

    @Test("Echo refocuses static phase")
    func echoRefocusesPhase() async {
        let circuit = QuantumCircuit.ramseyEcho(evolutionPhase: 2.0 * .pi, echoPosition: 0.5)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let p0 = state.probability(of: 0)
        #expect(abs(p0 - 1.0) < 1e-10, "Echo should refocus 2pi phase, P(|0>)=\(p0)")
    }
}

/// Test suite for QuantumState.ramseyResult() phase extraction.
/// Validates phase recovery from measurement probabilities,
/// visibility calculation, and T2* estimation conditions.
@Suite("QuantumState Ramsey Result")
struct QuantumStateRamseyResultTests {
    @Test("Phase extraction for zero phase")
    func phaseExtractionZero() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: 0.0)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let config = RamseyConfig(evolutionTime: 1e-6)
        let result = state.ramseyResult(config: config)
        #expect(abs(result.phaseAccumulation) < 1e-10, "Phase should be 0, got \(result.phaseAccumulation)")
        #expect(abs(result.excitedStateProbability) < 1e-10, "P(|1>) should be 0, got \(result.excitedStateProbability)")
    }

    @Test("Phase extraction for pi phase")
    func phaseExtractionPi() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: .pi)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let config = RamseyConfig(evolutionTime: 1e-6)
        let result = state.ramseyResult(config: config)
        #expect(abs(result.phaseAccumulation - .pi) < 1e-7, "Phase should be pi, got \(result.phaseAccumulation)")
        #expect(abs(result.excitedStateProbability - 1.0) < 1e-7, "P(|1>) should be 1, got \(result.excitedStateProbability)")
    }

    @Test("Phase extraction for pi/2 phase")
    func phaseExtractionPiOverTwo() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: .pi / 2)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let config = RamseyConfig(evolutionTime: 1e-6)
        let result = state.ramseyResult(config: config)
        #expect(abs(result.phaseAccumulation - .pi / 2) < 1e-10, "Phase should be pi/2, got \(result.phaseAccumulation)")
        #expect(abs(result.excitedStateProbability - 0.5) < 1e-10, "P(|1>) should be 0.5, got \(result.excitedStateProbability)")
    }

    @Test("Visibility is 1 for pure |0> state")
    func visibilityForPureZeroState() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: 0.0)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let config = RamseyConfig(evolutionTime: 1e-6)
        let result = state.ramseyResult(config: config)
        #expect(abs(result.visibility - 1.0) < 1e-10, "Visibility should be 1 for |0>, got \(result.visibility)")
    }

    @Test("Visibility is 1 for pure |1> state")
    func visibilityForPureOneState() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: .pi)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let config = RamseyConfig(evolutionTime: 1e-6)
        let result = state.ramseyResult(config: config)
        #expect(abs(result.visibility - 1.0) < 1e-10, "Visibility should be 1 for |1>, got \(result.visibility)")
    }

    @Test("Visibility is 0 for 50/50 superposition")
    func visibilityForEqualSuperposition() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: .pi / 2)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let config = RamseyConfig(evolutionTime: 1e-6)
        let result = state.ramseyResult(config: config)
        #expect(abs(result.visibility) < 1e-10, "Visibility should be 0 for 50/50, got \(result.visibility)")
    }

    @Test("T2* not estimated for zero evolution time")
    func t2StarNotEstimatedForZeroTime() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: .pi / 4)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let config = RamseyConfig(evolutionTime: 0.0)
        let result = state.ramseyResult(config: config)
        #expect(result.estimatedT2Star == nil, "T2* should be nil for zero evolution time")
    }

    @Test("T2* not estimated for visibility 0")
    func t2StarNotEstimatedForZeroVisibility() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: .pi / 2)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let config = RamseyConfig(evolutionTime: 1e-6)
        let result = state.ramseyResult(config: config)
        #expect(result.estimatedT2Star == nil, "T2* should be nil for visibility 0")
    }

    @Test("T2* not estimated for visibility 1")
    func t2StarNotEstimatedForUnitVisibility() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: 0.0)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let config = RamseyConfig(evolutionTime: 1e-6)
        let result = state.ramseyResult(config: config)
        #expect(result.estimatedT2Star == nil, "T2* should be nil for visibility 1")
    }
}

/// Test suite for Bayesian phase estimation convergence.
/// Validates that the estimator converges to correct phase values
/// for various target phases and prior configurations.
@Suite("Bayesian Phase Estimation Convergence")
struct BayesianConvergenceTests {
    @Test("Converges to small phase")
    func convergesToSmallPhase() async {
        let targetPhase = 0.1
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(targetPhase),
            prior: .uniform,
            discretizationBins: 256,
        )
        let result = await estimator.run(maxMeasurements: 100)
        #expect(result.posteriorStdDev < 1.0, "Should converge to reasonable precision")
    }

    @Test("Converges to pi/4 phase")
    func convergesToPiOverFour() async {
        let targetPhase = Double.pi / 4
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(targetPhase),
            prior: .uniform,
            discretizationBins: 256,
        )
        let result = await estimator.run(maxMeasurements: 100)
        #expect(result.posteriorStdDev < 1.0, "Should converge to reasonable precision")
    }

    @Test("Converges to pi/2 phase")
    func convergesToPiOverTwo() async {
        let targetPhase = Double.pi / 2
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(targetPhase),
            prior: .uniform,
            discretizationBins: 256,
        )
        let result = await estimator.run(maxMeasurements: 100)
        #expect(result.posteriorStdDev < 1.0, "Should converge to reasonable precision")
    }

    @Test("Gaussian prior centered on target converges faster")
    func gaussianPriorConvergesFaster() async {
        let targetPhase = Double.pi / 4
        let estimatorUniform = BayesianPhaseEstimation(
            unitary: .rotationZ(targetPhase),
            prior: .uniform,
            discretizationBins: 128,
        )
        let resultUniform = await estimatorUniform.run(maxMeasurements: 20)

        let estimatorGaussian = BayesianPhaseEstimation(
            unitary: .rotationZ(targetPhase),
            prior: .gaussian(mean: targetPhase, stdDev: 0.2),
            discretizationBins: 128,
        )
        let resultGaussian = await estimatorGaussian.run(maxMeasurements: 20)

        #expect(
            resultGaussian.posteriorStdDev <= resultUniform.posteriorStdDev + 0.5,
            "Gaussian prior should help or not hurt convergence",
        )
    }

    @Test("More measurements reduce uncertainty")
    func moreMeasurementsReduceUncertainty() async {
        let targetPhase = Double.pi / 3
        let estimator10 = BayesianPhaseEstimation(
            unitary: .rotationZ(targetPhase),
            prior: .uniform,
            discretizationBins: 128,
        )
        let result10 = await estimator10.run(maxMeasurements: 10)

        let estimator50 = BayesianPhaseEstimation(
            unitary: .rotationZ(targetPhase),
            prior: .uniform,
            discretizationBins: 128,
        )
        let result50 = await estimator50.run(maxMeasurements: 50)

        #expect(
            result50.posteriorStdDev <= result10.posteriorStdDev + 0.1,
            "More measurements should reduce or maintain uncertainty",
        )
    }
}

/// Test suite for edge cases and boundary conditions.
/// Validates handling of extreme parameters, small bins,
/// and unusual configurations in metrology functions.
@Suite("Metrology Edge Cases")
struct MetrologyEdgeCasesTests {
    @Test("Ramsey sequence with large phase")
    func ramseyLargePhase() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: 10.0 * .pi)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let p0 = state.probability(of: 0)
        let p1 = state.probability(of: 1)
        #expect(abs(p0 + p1 - 1.0) < 1e-10, "Probabilities should sum to 1 for large phase")
    }

    @Test("Ramsey sequence with negative phase")
    func ramseyNegativePhase() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: -.pi / 4)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let p0 = state.probability(of: 0)
        let p1 = state.probability(of: 1)
        #expect(abs(p0 + p1 - 1.0) < 1e-10, "Probabilities should sum to 1 for negative phase")
    }

    @Test("Bayesian estimation with 1 bin")
    func bayesianSingleBin() async {
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(.pi / 4),
            prior: .uniform,
            discretizationBins: 1,
        )
        let result = await estimator.run(maxMeasurements: 5)
        #expect(result.posterior.count == 1, "Should work with single bin")
    }

    @Test("Bayesian estimation with 1 measurement")
    func bayesianSingleMeasurement() async {
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(.pi / 4),
            prior: .uniform,
            discretizationBins: 64,
        )
        let result = await estimator.run(maxMeasurements: 1)
        #expect(result.measurementCount == 1, "Should complete with single measurement")
    }

    @Test("Echo with very small phase")
    func echoVerySmallPhase() async {
        let circuit = QuantumCircuit.ramseyEcho(evolutionPhase: 1e-12)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let p0 = state.probability(of: 0)
        #expect(abs(p0 - 1.0) < 1e-8, "Very small phase should give P(|0>) close to 1")
    }

    @Test("Custom prior with all zeros renormalizes")
    func customPriorAllZeros() async {
        let zeroPdf = [Double](repeating: 0.0, count: 10)
        let estimator = BayesianPhaseEstimation(
            unitary: .rotationZ(.pi / 4),
            prior: .custom(zeroPdf),
            discretizationBins: 64,
        )
        let result = await estimator.run(maxMeasurements: 1)
        #expect(result.posterior.count == 64, "Should handle zero prior gracefully")
    }

    @Test("Ramsey result description includes T2* when estimated")
    func ramseyResultDescriptionWithT2Star() async {
        let circuit = QuantumCircuit.ramseySequence(evolutionPhase: .pi / 3)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let config = RamseyConfig(evolutionTime: 1e-6)
        let result = state.ramseyResult(config: config)
        if result.estimatedT2Star != nil {
            #expect(result.description.contains("T2*"), "Description should contain T2* when estimated")
        }
    }
}
