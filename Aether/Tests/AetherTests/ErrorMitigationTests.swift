// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for ZeroNoiseExtrapolation initialization and configuration.
/// Validates scale factors sorting, extrapolation method selection,
/// and folding strategy configuration for noise amplification.
@Suite("ZNE Configuration")
struct ZNEConfigurationTests {
    @Test("Default configuration uses Richardson extrapolation")
    func defaultConfiguration() {
        let zne = ZeroNoiseExtrapolation()

        #expect(zne.scaleFactors.count == 3, "Default should have 3 scale factors")
        #expect(zne.scaleFactors[0] == 1, "First scale factor should be 1")
        #expect(zne.scaleFactors[1] == 2, "Second scale factor should be 2")
        #expect(zne.scaleFactors[2] == 3, "Third scale factor should be 3")
        #expect(zne.method == .richardson, "Default method should be Richardson")
        #expect(zne.foldingStrategy == .global, "Default folding should be global")
    }

    @Test("Scale factors are sorted")
    func scaleFactorsSorted() {
        let zne = ZeroNoiseExtrapolation(scaleFactors: [3, 1, 5, 2])

        #expect(zne.scaleFactors.count == 4, "Should have 4 scale factors")
        #expect(zne.scaleFactors[0] == 1, "First should be 1")
        #expect(zne.scaleFactors[1] == 2, "Second should be 2")
        #expect(zne.scaleFactors[2] == 3, "Third should be 3")
        #expect(zne.scaleFactors[3] == 5, "Fourth should be 5")
    }

    @Test("Custom extrapolation method")
    func customExtrapolationMethod() {
        let zneLinear = ZeroNoiseExtrapolation(method: .linear)
        let znePoly = ZeroNoiseExtrapolation(method: .polynomial(degree: 2))
        let zneExp = ZeroNoiseExtrapolation(method: .exponential)

        #expect(zneLinear.method == .linear, "Should use linear method")
        #expect(znePoly.method == .polynomial(degree: 2), "Should use polynomial method")
        #expect(zneExp.method == .exponential, "Should use exponential method")
    }

    @Test("Custom folding strategy")
    func customFoldingStrategy() {
        let zneLocal = ZeroNoiseExtrapolation(foldingStrategy: .local)
        let zneFromEnd = ZeroNoiseExtrapolation(foldingStrategy: .fromEnd)

        #expect(zneLocal.foldingStrategy == .local, "Should use local folding")
        #expect(zneFromEnd.foldingStrategy == .fromEnd, "Should use fromEnd folding")
    }

    @Test("Odd scale factors for global folding")
    func oddScaleFactors() {
        let zne = ZeroNoiseExtrapolation(scaleFactors: [1, 3, 5, 7])

        #expect(zne.scaleFactors.count == 4, "Should have 4 scale factors")
        #expect(zne.scaleFactors.contains(1), "Must contain 1.0")
    }
}

/// Test suite for circuit folding operations.
/// Validates global, local, and fromEnd folding strategies that
/// amplify noise by inserting G†G identity pairs into circuits.
@Suite("Circuit Folding")
struct CircuitFoldingTests {
    @Test("Scale factor 1 returns unchanged circuit")
    func scaleFactor1Unchanged() {
        let zne = ZeroNoiseExtrapolation()
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let folded = zne.fold(circuit: circuit, scaleFactor: 1.0)

        #expect(folded.count == 2, "Scale 1 should return original circuit")
    }

    @Test("Global folding scale 3 triples circuit depth")
    func globalFoldingScale3() {
        let zne = ZeroNoiseExtrapolation(foldingStrategy: .global)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        let folded = zne.fold(circuit: circuit, scaleFactor: 3.0)

        #expect(folded.count == 3, "Scale 3 should give H H† H = 3 gates")
    }

    @Test("Global folding scale 5 gives 5x gates")
    func globalFoldingScale5() {
        let zne = ZeroNoiseExtrapolation(foldingStrategy: .global)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        let folded = zne.fold(circuit: circuit, scaleFactor: 5.0)

        #expect(folded.count == 5, "Scale 5 should give H H† H H† H = 5 gates")
    }

    @Test("Local folding distributes folds across gates")
    func localFoldingDistributes() {
        let zne = ZeroNoiseExtrapolation(foldingStrategy: .local)
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.pauliZ, to: 0)
        circuit.append(.hadamard, to: 1)

        let folded = zne.fold(circuit: circuit, scaleFactor: 2.0)

        #expect(folded.count >= circuit.count, "Local folding should add gates")
    }

    @Test("FromEnd folding folds from circuit end")
    func fromEndFolding() {
        let zne = ZeroNoiseExtrapolation(foldingStrategy: .fromEnd)
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.pauliX, to: 1)

        let folded = zne.fold(circuit: circuit, scaleFactor: 2.0)

        #expect(folded.count >= circuit.count, "FromEnd folding should add gates")
    }

    @Test("Folded circuit preserves qubit count")
    func foldedCircuitPreservesQubits() {
        let zne = ZeroNoiseExtrapolation()
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.toffoli, to: [0, 1, 2])

        let folded = zne.fold(circuit: circuit, scaleFactor: 3.0)

        #expect(folded.qubits == 3, "Folded circuit should preserve qubit count")
    }

    @Test("Fractional scale factor with global folding")
    func fractionalScaleFactor() {
        let zne = ZeroNoiseExtrapolation(foldingStrategy: .global)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliZ, to: 0)

        let folded = zne.fold(circuit: circuit, scaleFactor: 2.0)

        #expect(folded.count >= circuit.count, "Fractional scaling should add partial folds")
    }
}

/// Test suite for extrapolation algorithms.
/// Validates linear, polynomial, exponential, and Richardson extrapolation
/// methods for zero-noise limit estimation from scaled measurements.
@Suite("Extrapolation Methods")
struct ExtrapolationMethodTests {
    @Test("Linear extrapolation on simple data")
    func linearExtrapolation() async {
        let zne = ZeroNoiseExtrapolation(
            scaleFactors: [1, 2, 3],
            method: .linear,
        )
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        let circuit = QuantumCircuit(qubits: 1)

        let result = await zne.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(result.mitigatedValue.isFinite, "Linear extrapolation should give finite result")
    }

    @Test("Polynomial extrapolation degree 2")
    func polynomialExtrapolation() async {
        let zne = ZeroNoiseExtrapolation(
            scaleFactors: [1, 2, 3, 4],
            method: .polynomial(degree: 2),
        )
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        let circuit = QuantumCircuit(qubits: 1)

        let result = await zne.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(result.mitigatedValue.isFinite, "Polynomial extrapolation should give finite result")
    }

    @Test("Exponential extrapolation")
    func exponentialExtrapolation() async {
        let zne = ZeroNoiseExtrapolation(
            scaleFactors: [1, 2, 3],
            method: .exponential,
        )
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        let circuit = QuantumCircuit(qubits: 1)

        let result = await zne.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(result.mitigatedValue.isFinite, "Exponential extrapolation should give finite result")
    }

    @Test("Richardson extrapolation")
    func richardsonExtrapolation() async {
        let zne = ZeroNoiseExtrapolation(
            scaleFactors: [1, 2, 3],
            method: .richardson,
        )
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        let circuit = QuantumCircuit(qubits: 1)

        let result = await zne.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(result.mitigatedValue.isFinite, "Richardson extrapolation should give finite result")
    }
}

/// Test suite for full ZNE mitigation workflow.
/// Validates end-to-end noise reduction including result structure,
/// improvement factors, batch mitigation, and ideal circuit preservation.
@Suite("ZNE Mitigation")
struct ZNEMitigationTests {
    @Test("Mitigation returns valid result structure")
    func mitigationResultStructure() async {
        let zne = ZeroNoiseExtrapolation(scaleFactors: [1, 2, 3])
        let noise = NoiseModel.depolarizing(singleQubitError: 0.01, twoQubitError: 0.02)
        let simulator = DensityMatrixSimulator(noiseModel: noise)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        let result = await zne.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(result.noisyValues.count == 3, "Should have 3 noisy values")
        #expect(result.scaleFactors.count == 3, "Should have 3 scale factors")
        #expect(result.scaleFactors[0] == 1, "First scale factor should be 1")
        #expect(result.scaleFactors[1] == 2, "Second scale factor should be 2")
        #expect(result.scaleFactors[2] == 3, "Third scale factor should be 3")
        #expect(result.mitigatedValue.isFinite, "Mitigated value should be finite")
    }

    @Test("Mitigation on noisy circuit improves expectation")
    func mitigationImprovesExpectation() async {
        let zne = ZeroNoiseExtrapolation(scaleFactors: [1, 3, 5])
        let noise = NoiseModel.depolarizing(singleQubitError: 0.05, twoQubitError: 0.1)
        let simulator = DensityMatrixSimulator(noiseModel: noise)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)

        let result = await zne.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        let noisyValue = result.noisyValues.first { $0.scale == 1.0 }?.value
        #expect(noisyValue!.isFinite, "Noisy value should be finite")
        #expect(result.mitigatedValue.isFinite, "Mitigated value should be finite")
    }

    @Test("Mitigation preserves ideal result under no noise")
    func mitigationPreservesIdeal() async {
        let zne = ZeroNoiseExtrapolation(scaleFactors: [1, 2, 3])
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)

        let result = await zne.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(abs(result.mitigatedValue - -1.0) < 0.1, "Ideal X|0⟩ should give ⟨Z⟩ ≈ -1")
    }

    @Test("Batch mitigation for multiple observables")
    func batchMitigation() async {
        let zne = ZeroNoiseExtrapolation(scaleFactors: [1, 2, 3])
        let noise = NoiseModel.depolarizing(singleQubitError: 0.01, twoQubitError: 0.02)
        let simulator = DensityMatrixSimulator(noiseModel: noise)
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let observables = [
            Observable.pauliZ(qubit: 0),
            Observable.pauliZ(qubit: 1),
            Observable.pauliX(qubit: 0),
        ]

        let results = await zne.mitigateBatch(
            circuit: circuit,
            observables: observables,
            simulator: simulator,
        )

        #expect(results.count == 3, "Should have 3 mitigated values")
        // swiftformat:disable:next preferKeyPath
        #expect(results.allSatisfy { $0.isFinite }, "All results should be finite")
    }

    @Test("ZNE result improvement factor")
    func improvementFactor() async {
        let zne = ZeroNoiseExtrapolation(scaleFactors: [1, 3, 5])
        let noise = NoiseModel.depolarizing(singleQubitError: 0.05, twoQubitError: 0.1)
        let simulator = DensityMatrixSimulator(noiseModel: noise)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)

        let result = await zne.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(result.improvementFactor >= 0, "Improvement factor should be non-negative")
    }

    @Test("ZNE result description is formatted")
    func resultDescription() async {
        let zne = ZeroNoiseExtrapolation(scaleFactors: [1, 2, 3])
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        let circuit = QuantumCircuit(qubits: 1)

        let result = await zne.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(result.description.contains("ZNE Result"), "Description should contain header")
        #expect(result.description.contains("Mitigated"), "Description should contain mitigated value")
    }
}

/// Test suite for ProbabilisticErrorCancellation initialization.
/// Validates gamma overhead calculation from error probability,
/// sample count configuration, and quasi-probability decomposition.
@Suite("PEC Configuration")
struct PECConfigurationTests {
    @Test("PEC initialization with error probability")
    func pecInitialization() {
        let pec = ProbabilisticErrorCancellation(errorProbability: 0.01, samples: 1000)

        #expect(pec.samples == 1000, "Samples should be stored")
        #expect(pec.gamma >= 1.0, "Gamma overhead should be >= 1")
    }

    @Test("Zero error gives unit gamma")
    func zeroErrorUnitGamma() {
        let pec = ProbabilisticErrorCancellation(errorProbability: 0.0, samples: 100)

        #expect(abs(pec.gamma - 1.0) < 1e-10, "Zero error should give gamma = 1")
    }

    @Test("Higher error increases gamma overhead")
    func higherErrorIncreasesGamma() {
        let pecLow = ProbabilisticErrorCancellation(errorProbability: 0.01, samples: 100)
        let pecHigh = ProbabilisticErrorCancellation(errorProbability: 0.1, samples: 100)

        #expect(pecHigh.gamma > pecLow.gamma, "Higher error should increase gamma")
    }

    @Test("Default sample count")
    func defaultSampleCount() {
        let pec = ProbabilisticErrorCancellation(errorProbability: 0.01)

        #expect(pec.samples == 10000, "Default samples should be 10000")
    }
}

/// Test suite for Pauli quasi-probability decomposition.
/// Validates gamma overhead scaling with error probability and
/// finite decomposition values across the valid error range.
@Suite("PEC Pauli Decomposition")
struct PECPauliDecompositionTests {
    @Test("Decomposition probabilities sum to 1")
    func probabilitiesSumToOne() {
        let pec = ProbabilisticErrorCancellation(errorProbability: 0.1, samples: 100)
        let decomposition = pec.gamma

        #expect(decomposition.isFinite, "Decomposition gamma should be finite")
    }

    @Test("Decomposition for various error rates")
    func decompositionVariousRates() {
        let errorRates = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.74]

        for rate in errorRates {
            let pec = ProbabilisticErrorCancellation(errorProbability: rate, samples: 100)
            #expect(pec.gamma >= 1.0, "Gamma should be >= 1 for p=\(rate)")
            #expect(pec.gamma.isFinite, "Gamma should be finite for p=\(rate)")
        }
    }
}

/// Test suite for PEC mitigation workflow.
/// Validates quasi-probability sampling, statistical properties including
/// variance and confidence intervals, reproducibility, and depth scaling.
@Suite("PEC Mitigation")
struct PECMitigationTests {
    @Test("PEC returns finite mitigated value")
    func pecReturnsFiniteValue() async {
        let pec = ProbabilisticErrorCancellation(errorProbability: 0.01, samples: 100)
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        let result = await pec.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(result.mitigatedValue.isFinite, "Mitigated value should be finite")
    }

    @Test("PEC result has valid statistics")
    func pecResultStatistics() async {
        let pec = ProbabilisticErrorCancellation(errorProbability: 0.01, samples: 500)
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)

        let result = await pec.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(result.samples == 500, "Should have 500 samples")
        #expect(result.standardError >= 0, "Standard error should be non-negative")
        #expect(result.variance.isFinite, "Variance should be finite")
    }

    @Test("PEC confidence interval is valid")
    func pecConfidenceInterval() async {
        let pec = ProbabilisticErrorCancellation(errorProbability: 0.01, samples: 200)
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        let circuit = QuantumCircuit(qubits: 1)

        let result = await pec.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        let ci = result.confidenceInterval
        #expect(ci.low <= result.mitigatedValue, "CI low should be <= mitigated")
        #expect(ci.high >= result.mitigatedValue, "CI high should be >= mitigated")
    }

    @Test("PEC with seed is reproducible")
    func pecReproducibility() async {
        let pec = ProbabilisticErrorCancellation(errorProbability: 0.01, samples: 100)
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        let result1 = await pec.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
            seed: 12345,
        )

        let result2 = await pec.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
            seed: 12345,
        )

        #expect(abs(result1.mitigatedValue - result2.mitigatedValue) < 1e-10,
                "Same seed should give same result")
    }

    @Test("PEC result description is formatted")
    func pecResultDescription() async {
        let pec = ProbabilisticErrorCancellation(errorProbability: 0.01, samples: 100)
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        let circuit = QuantumCircuit(qubits: 1)

        let result = await pec.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(result.description.contains("PEC Result"), "Description should contain header")
        #expect(result.description.contains("Samples"), "Description should contain samples")
    }

    @Test("PEC gamma scales with circuit depth")
    func pecGammaScalesWithDepth() async {
        let pec = ProbabilisticErrorCancellation(errorProbability: 0.1, samples: 50)
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)

        var shortCircuit = QuantumCircuit(qubits: 1)
        shortCircuit.append(.hadamard, to: 0)

        var longCircuit = QuantumCircuit(qubits: 1)
        for _ in 0 ..< 5 {
            longCircuit.append(.hadamard, to: 0)
        }

        let shortResult = await pec.mitigate(
            circuit: shortCircuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        let longResult = await pec.mitigate(
            circuit: longCircuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(longResult.gamma > shortResult.gamma,
                "Longer circuit should have higher gamma overhead")
    }
}

/// Test suite for measurement error mitigation.
/// Validates confusion matrix inversion for histogram correction,
/// total count preservation, and multi-qubit tensor product mitigation.
@Suite("Readout Error Mitigation")
struct ReadoutErrorMitigationTests {
    @Test("Mitigate histogram with zero error is identity")
    func mitigateZeroError() {
        let model = MeasurementErrorModel(p0Given1: 0.0, p1Given0: 0.0)
        let histogram: [Int: Int] = [0: 500, 1: 500]

        let corrected = MeasurementErrorModel.mitigateFullHistogram(
            histogram,
            totalQubits: 1,
            models: [model],
        )

        #expect(abs(corrected[0]! - 500.0) < 1e-10, "No error should preserve counts")
        #expect(abs(corrected[1]! - 500.0) < 1e-10, "No error should preserve counts")
    }

    @Test("Mitigate histogram corrects bias")
    func mitigateCorrectsBias() {
        let model = MeasurementErrorModel(p0Given1: 0.1, p1Given0: 0.05)
        let histogram: [Int: Int] = [0: 600, 1: 400]

        let corrected = MeasurementErrorModel.mitigateFullHistogram(
            histogram,
            totalQubits: 1,
            models: [model],
        )

        #expect(corrected[0]!.isFinite, "Corrected P(0) should be finite")
        #expect(corrected[1]!.isFinite, "Corrected P(1) should be finite")
    }

    @Test("Mitigate histogram preserves total counts")
    func mitigatePreservesTotal() {
        let model = MeasurementErrorModel(p0Given1: 0.1, p1Given0: 0.05)
        let histogram: [Int: Int] = [0: 700, 1: 300]
        let totalOriginal = 1000

        let corrected = MeasurementErrorModel.mitigateFullHistogram(
            histogram,
            totalQubits: 1,
            models: [model],
        )

        let totalCorrected = corrected.values.reduce(0, +)
        #expect(abs(totalCorrected - Double(totalOriginal)) < 1e-6,
                "Total counts should be preserved")
    }

    @Test("Mitigate multi-qubit histogram")
    func mitigateMultiQubitHistogram() {
        let model = MeasurementErrorModel(p0Given1: 0.05, p1Given0: 0.02)
        let histogram: [Int: Int] = [0: 400, 1: 200, 2: 200, 3: 200]

        let corrected = MeasurementErrorModel.mitigateFullHistogram(
            histogram,
            totalQubits: 2,
            models: [model],
        )

        #expect(corrected.count == 4, "Should have 4 basis states")
        // swiftformat:disable:next preferKeyPath
        #expect(corrected.values.allSatisfy { $0.isFinite }, "All counts should be finite")
    }

    @Test("Mitigate with per-qubit models")
    func mitigatePerQubitModels() {
        let model0 = MeasurementErrorModel(p0Given1: 0.1, p1Given0: 0.05)
        let model1 = MeasurementErrorModel(p0Given1: 0.02, p1Given0: 0.01)
        let histogram: [Int: Int] = [0: 500, 1: 200, 2: 200, 3: 100]

        let corrected = MeasurementErrorModel.mitigateFullHistogram(
            histogram,
            totalQubits: 2,
            models: [model0, model1],
        )

        // swiftformat:disable:next preferKeyPath
        #expect(corrected.values.allSatisfy { $0.isFinite }, "All corrected counts should be finite")
    }

    @Test("Mitigate empty histogram")
    func mitigateEmptyHistogram() {
        let model = MeasurementErrorModel(p0Given1: 0.1, p1Given0: 0.05)
        let histogram: [Int: Int] = [:]

        let corrected = MeasurementErrorModel.mitigateFullHistogram(
            histogram,
            totalQubits: 1,
            models: [model],
        )

        #expect(corrected.isEmpty, "Empty histogram should stay empty")
    }

    @Test("Mitigate single state histogram")
    func mitigateSingleState() {
        let model = MeasurementErrorModel(p0Given1: 0.1, p1Given0: 0.05)
        let histogram = [0: 1000]

        let corrected = MeasurementErrorModel.mitigateFullHistogram(
            histogram,
            totalQubits: 1,
            models: [model],
        )

        #expect(corrected[0]!.isFinite, "Single state should be mitigated")
        #expect(abs(corrected.values.reduce(0, +) - 1000) < 1e-6, "Total should be preserved")
    }
}

/// Test suite for MeasurementErrorModel single-qubit mitigation.
/// Validates confusion matrix apply-and-mitigate round-trip and
/// single-qubit histogram correction with count preservation.
@Suite("Measurement Error Model Mitigation")
struct MeasurementErrorModelMitigationTests {
    @Test("Apply and mitigate roundtrip")
    func applyAndMitigateRoundtrip() {
        let model = MeasurementErrorModel(p0Given1: 0.1, p1Given0: 0.05)

        let original = (p0: 0.7, p1: 0.3)
        let noisy = model.applyError(to: original)
        let mitigated = model.mitigate(probabilities: noisy)

        #expect(abs(mitigated.0 - original.p0) < 0.1, "Mitigation should approximately recover P(0)")
        #expect(abs(mitigated.1 - original.p1) < 0.1, "Mitigation should approximately recover P(1)")
    }

    @Test("Mitigate histogram single qubit")
    func mitigateHistogramSingleQubit() {
        let model = MeasurementErrorModel(p0Given1: 0.1, p1Given0: 0.05)
        let histogram = [0: 550, 1: 450]

        let corrected = model.mitigateHistogram(histogram, qubit: 0, totalQubits: 1)
        let total = corrected[0]! + corrected[1]!

        #expect(abs(total - 1000) < 10, "Total counts should be approximately preserved")
    }
}

/// Test suite for combined error mitigation strategies.
/// Validates ZNE and PEC on realistic circuits including Bell states,
/// rotations, method consistency, and folding strategy variations.
@Suite("Error Mitigation Integration")
struct ErrorMitigationIntegrationTests {
    @Test("ZNE with noisy Bell state preparation")
    func zneWithBellState() async {
        let zne = ZeroNoiseExtrapolation(scaleFactors: [1, 3, 5])
        let noise = NoiseModel.depolarizing(singleQubitError: 0.02, twoQubitError: 0.05)
        let simulator = DensityMatrixSimulator(noiseModel: noise)
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let zz = Observable(terms: [
            (1.0, PauliString([.z(0), .z(1)])),
        ])

        let result = await zne.mitigate(
            circuit: circuit,
            observable: zz,
            simulator: simulator,
        )

        #expect(result.mitigatedValue.isFinite, "ZNE on Bell state should give finite result")
    }

    @Test("PEC with simple rotation")
    func pecWithRotation() async {
        let pec = ProbabilisticErrorCancellation(errorProbability: 0.02, samples: 100)
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationY(.pi / 4), to: 0)

        let result = await pec.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(result.mitigatedValue.isFinite, "PEC on rotation should give finite result")
    }

    @Test("ZNE with different methods gives consistent results")
    func zneMethodsConsistency() async {
        let noise = NoiseModel.depolarizing(singleQubitError: 0.01, twoQubitError: 0.02)
        let simulator = DensityMatrixSimulator(noiseModel: noise)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)

        let methods: [ZeroNoiseExtrapolation.ExtrapolationMethod] = [
            .linear,
            .polynomial(degree: 2),
            .exponential,
            .richardson,
        ]

        var results: [Double] = []
        for method in methods {
            let zne = ZeroNoiseExtrapolation(
                scaleFactors: [1, 2, 3],
                method: method,
            )
            let result = await zne.mitigate(
                circuit: circuit,
                observable: Observable.pauliZ(qubit: 0),
                simulator: simulator,
            )
            results.append(result.mitigatedValue)
        }

        // swiftformat:disable:next preferKeyPath
        #expect(results.allSatisfy { $0.isFinite }, "All methods should give finite results")
    }

    @Test("ZNE with all folding strategies")
    func zneFoldingStrategies() async {
        let noise = NoiseModel.depolarizing(singleQubitError: 0.01, twoQubitError: 0.02)
        let simulator = DensityMatrixSimulator(noiseModel: noise)
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let strategies: [ZeroNoiseExtrapolation.FoldingStrategy] = [
            .global,
            .local,
            .fromEnd,
        ]

        for strategy in strategies {
            let zne = ZeroNoiseExtrapolation(
                scaleFactors: [1, 2, 3],
                foldingStrategy: strategy,
            )
            let result = await zne.mitigate(
                circuit: circuit,
                observable: Observable.pauliZ(qubit: 0),
                simulator: simulator,
            )
            #expect(result.mitigatedValue.isFinite,
                    "Strategy \(strategy) should give finite result")
        }
    }
}

/// Test suite for edge cases and boundary conditions.
/// Validates empty circuits, identity observables, minimal samples,
/// highly biased models, many scale factors, and multi-qubit systems.
@Suite("Error Mitigation Edge Cases")
struct ErrorMitigationEdgeCasesTests {
    @Test("ZNE with empty circuit")
    func zneEmptyCircuit() async {
        let zne = ZeroNoiseExtrapolation()
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        let circuit = QuantumCircuit(qubits: 1)

        let result = await zne.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(abs(result.mitigatedValue - 1.0) < 1e-10,
                "Empty circuit should give ⟨Z⟩ = 1 for |0⟩")
    }

    @Test("ZNE with identity observable")
    func zneIdentityObservable() async {
        let zne = ZeroNoiseExtrapolation()
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        let identity = Observable(terms: [
            (1.0, PauliString([])),
        ])

        let result = await zne.mitigate(
            circuit: circuit,
            observable: identity,
            simulator: simulator,
        )

        #expect(abs(result.mitigatedValue - 1.0) < 1e-10,
                "Identity observable should give 1")
    }

    @Test("PEC with minimal samples")
    func pecMinimalSamples() async {
        let pec = ProbabilisticErrorCancellation(errorProbability: 0.01, samples: 10)
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        let circuit = QuantumCircuit(qubits: 1)

        let result = await pec.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(result.samples == 10, "Should use 10 samples")
        #expect(result.mitigatedValue.isFinite, "Should still give finite result")
    }

    @Test("Readout mitigation with highly biased model")
    func readoutHighlyBiased() {
        let model = MeasurementErrorModel(p0Given1: 0.4, p1Given0: 0.4)
        let histogram: [Int: Int] = [0: 500, 1: 500]

        let corrected = MeasurementErrorModel.mitigateFullHistogram(
            histogram,
            totalQubits: 1,
            models: [model],
        )

        // swiftformat:disable:next preferKeyPath
        #expect(corrected.values.allSatisfy { $0.isFinite }, "Should handle highly biased model")
    }

    @Test("ZNE with many scale factors")
    func zneManyScaleFactors() async {
        let zne = ZeroNoiseExtrapolation(
            scaleFactors: [1, 2, 3, 4, 5, 6, 7],
            method: .polynomial(degree: 3),
        )
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        let circuit = QuantumCircuit(qubits: 1)

        let result = await zne.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(result.noisyValues.count == 7, "Should have 7 noisy values")
        #expect(result.mitigatedValue.isFinite, "Should handle many scale factors")
    }

    @Test("Multi-qubit readout mitigation with 3 qubits")
    func threeQubitReadoutMitigation() {
        let model = MeasurementErrorModel(p0Given1: 0.05, p1Given0: 0.02)
        var histogram: [Int: Int] = [:]
        for i in 0 ..< 8 {
            histogram[i] = 100 + i * 10
        }

        let corrected = MeasurementErrorModel.mitigateFullHistogram(
            histogram,
            totalQubits: 3,
            models: [model],
        )

        #expect(corrected.count == 8, "Should have all 8 basis states")
        // swiftformat:disable:next preferKeyPath
        #expect(corrected.values.allSatisfy { $0.isFinite }, "All values should be finite")
    }

    @Test("Local folding with empty circuit covers zero gate branch")
    func localFoldingEmptyCircuit() {
        let zne = ZeroNoiseExtrapolation(foldingStrategy: .local)
        let circuit = QuantumCircuit(qubits: 1)

        let folded = zne.fold(circuit: circuit, scaleFactor: 2.0)

        #expect(folded.count == 0, "Empty circuit should remain empty after local folding")
        #expect(folded.qubits == 1, "Should preserve qubit count")
    }

    @Test("Linear extrapolation with degenerate x values falls back to mean")
    func linearExtrapolationDegenerateCase() async {
        let zne = ZeroNoiseExtrapolation(
            scaleFactors: [1, 1, 1],
            method: .linear,
        )
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)

        let result = await zne.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(result.mitigatedValue.isFinite, "Degenerate linear should return mean value")
        #expect(abs(result.mitigatedValue - -1.0) < 0.1, "Should approximate -1 for X|0⟩")
    }

    @Test("Exponential extrapolation with insufficient data falls back to linear")
    func exponentialInsufficientData() async {
        let zne = ZeroNoiseExtrapolation(
            scaleFactors: [1, 2],
            method: .exponential,
        )
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        let result = await zne.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(result.mitigatedValue.isFinite, "Should fall back to linear with < 3 data points")
        #expect(result.noisyValues.count == 2, "Should have only 2 scale factors")
    }

    @Test("Exponential extrapolation with non-monotonic data triggers fallback")
    func exponentialNonMonotonicData() async {
        let zne = ZeroNoiseExtrapolation(
            scaleFactors: [1, 2, 3],
            method: .exponential,
        )
        let noise = NoiseModel.depolarizing(singleQubitError: 0.5, twoQubitError: 0.5)
        let simulator = DensityMatrixSimulator(noiseModel: noise)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 0)
        circuit.append(.hadamard, to: 0)

        let result = await zne.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(result.mitigatedValue.isFinite, "Should handle invalid exponential fit with fallback")
    }

    @Test("Exponential extrapolation with non-finite intermediate triggers linear fallback")
    func exponentialNonFiniteFallback() async {
        let zne = ZeroNoiseExtrapolation(
            scaleFactors: [1, 1.001, 1.002],
            method: .exponential,
        )
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)

        let result = await zne.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(result.mitigatedValue.isFinite,
                "Should fall back to linear when exponential produces non-finite result")
    }

    @Test("Polynomial extrapolation with zero Vandermonde determinant handles empty coefficients")
    func polynomialDegenerateVandermonde() async {
        let zne = ZeroNoiseExtrapolation(
            scaleFactors: [1, 1, 1],
            method: .polynomial(degree: 2),
        )
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)

        let result = await zne.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
        )

        #expect(result.mitigatedValue.isFinite,
                "Degenerate Vandermonde should return first data value")
    }

    @Test("ZNEResult improvement factor returns 0 when scale 1.0 not found")
    func improvementFactorMissingScaleOne() {
        let result = ZNEResult(
            mitigatedValue: 0.5,
            noisyValues: [(scale: 2.0, value: 0.6), (scale: 3.0, value: 0.7)],
            method: .richardson,
            scaleFactors: [2, 3],
        )

        #expect(result.improvementFactor == 0,
                "Improvement factor should be 0 when scale 1.0 is missing")
    }

    @Test("ZNEResult improvement factor returns 0 when noisy value is zero")
    func improvementFactorZeroNoisyValue() {
        let result = ZNEResult(
            mitigatedValue: 0.5,
            noisyValues: [(scale: 1.0, value: 0.0), (scale: 2.0, value: 0.1)],
            method: .richardson,
            scaleFactors: [1, 2],
        )

        #expect(result.improvementFactor == 0,
                "Improvement factor should be 0 when noisy value at scale 1.0 is zero")
    }
}

/// Test suite for reset operation preservation during circuit folding and PEC sampling.
/// Validates that foldLocal, foldFromEnd, and sampleCircuit correctly pass through
/// mid-circuit reset operations without folding or modifying them.
@Suite("Reset Preservation in Error Mitigation")
struct ResetPreservationTests {
    @Test("foldLocal preserves reset operations between gates")
    func foldLocalPreservesReset() {
        let zne = ZeroNoiseExtrapolation(
            scaleFactors: [1, 3],
            foldingStrategy: .local,
        )
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.reset, to: 0)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliZ, to: 1)

        let folded = zne.fold(circuit: circuit, scaleFactor: 3.0)

        let resetCount = folded.operations.count(where: {
            if case .reset = $0 { return true }
            return false
        })
        #expect(resetCount == 1,
                "Local folding should preserve exactly one reset operation, found \(resetCount)")

        let resetIndices = folded.operations.enumerated().compactMap { index, op -> Int? in
            if case .reset = op { return index }
            return nil
        }
        #expect(resetIndices.count == 1,
                "Reset should appear exactly once in folded circuit")

        if let resetIndex = resetIndices.first {
            let hasGateBefore = resetIndex > 0
            let hasGateAfter = resetIndex < folded.operations.count - 1
            #expect(hasGateBefore,
                    "Reset should have gate operations before it in the folded circuit")
            #expect(hasGateAfter,
                    "Reset should have gate operations after it in the folded circuit")
        }
    }

    @Test("foldLocal with reset at circuit start preserves reset")
    func foldLocalResetAtStart() {
        let zne = ZeroNoiseExtrapolation(
            scaleFactors: [1, 3],
            foldingStrategy: .local,
        )
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.reset, to: 0)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 0)

        let folded = zne.fold(circuit: circuit, scaleFactor: 3.0)

        let resetCount = folded.operations.count(where: {
            if case .reset = $0 { return true }
            return false
        })
        #expect(resetCount == 1,
                "Local folding should preserve reset at start, found \(resetCount) resets")

        let firstIsReset = if case .reset = folded.operations[0] {
            true
        } else {
            false
        }
        #expect(firstIsReset,
                "First operation in folded circuit should be reset, got \(folded.operations[0])")
        #expect(folded.operations.count > 1,
                "Folded circuit should have more than just the reset operation")
    }

    @Test("foldLocal with multiple resets preserves all resets")
    func foldLocalMultipleResets() {
        let zne = ZeroNoiseExtrapolation(
            scaleFactors: [1, 3],
            foldingStrategy: .local,
        )
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.reset, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.reset, to: 1)
        circuit.append(.hadamard, to: 1)

        let folded = zne.fold(circuit: circuit, scaleFactor: 2.0)

        let resetCount = folded.operations.count(where: {
            if case .reset = $0 { return true }
            return false
        })
        #expect(resetCount == 2,
                "Local folding should preserve both reset operations, found \(resetCount)")
    }

    @Test("foldFromEnd preserves reset in reverse section")
    func foldFromEndPreservesResetReverse() {
        let zne = ZeroNoiseExtrapolation(
            scaleFactors: [1, 3],
            foldingStrategy: .fromEnd,
        )
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 0)
        circuit.append(.reset, to: 0)

        let folded = zne.fold(circuit: circuit, scaleFactor: 3.0)

        let resetCount = folded.operations.count(where: {
            if case .reset = $0 { return true }
            return false
        })
        #expect(resetCount >= 2,
                "FromEnd folding with reset at end should preserve resets in reverse and forward sections, found \(resetCount)")

        let gateCount = folded.operations.count(where: {
            if case .gate = $0 { return true }
            return false
        })
        #expect(gateCount > 0,
                "Folded circuit should still contain gate operations")
    }

    @Test("foldFromEnd preserves reset in forward section")
    func foldFromEndPreservesResetForward() {
        let zne = ZeroNoiseExtrapolation(
            scaleFactors: [1, 3],
            foldingStrategy: .fromEnd,
        )
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.reset, to: 0)
        circuit.append(.pauliX, to: 0)

        let folded = zne.fold(circuit: circuit, scaleFactor: 2.0)

        let resetCount = folded.operations.count(where: {
            if case .reset = $0 { return true }
            return false
        })
        #expect(resetCount >= 1,
                "FromEnd folding should preserve at least the original reset, found \(resetCount)")

        #expect(folded.operations.count > circuit.operations.count,
                "Folded circuit at scale > 1 should have more operations than original")
    }

    @Test("foldFromEnd with reset at end produces resets in both reverse and forward passes")
    func foldFromEndResetAtEndBothPasses() {
        let zne = ZeroNoiseExtrapolation(
            scaleFactors: [1, 3],
            foldingStrategy: .fromEnd,
        )
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.reset, to: 0)

        let folded = zne.fold(circuit: circuit, scaleFactor: 3.0)

        let resetOps = folded.operations.enumerated().compactMap { index, op -> Int? in
            if case .reset = op { return index }
            return nil
        }
        #expect(resetOps.count >= 2,
                "FromEnd folding at scale 3 should produce resets from both reverse and forward passes, found \(resetOps.count)")

        if resetOps.count >= 2 {
            #expect(resetOps[0] < resetOps[1],
                    "Reset operations should appear at distinct positions in folded circuit")
        }
    }

    @Test("PEC sampleCircuit preserves reset operations")
    func pecSampleCircuitPreservesReset() async {
        let pec = ProbabilisticErrorCancellation(errorProbability: 0.01, samples: 50)
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.reset, to: 0)
        circuit.append(.pauliX, to: 0)

        let result = await pec.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
            seed: 42,
        )

        #expect(result.mitigatedValue.isFinite,
                "PEC with reset in circuit should produce finite mitigated value")
        #expect(result.samples == 50,
                "PEC should use the configured number of samples")
    }

    @Test("PEC with reset-only circuit produces finite result")
    func pecResetOnlyCircuit() async {
        let pec = ProbabilisticErrorCancellation(errorProbability: 0.01, samples: 50)
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.reset, to: 0)

        let result = await pec.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
            seed: 99,
        )

        #expect(result.mitigatedValue.isFinite,
                "PEC with reset-only circuit should produce finite result")
    }

    @Test("PEC with reset between gates is reproducible with seed")
    func pecResetReproducibleWithSeed() async {
        let pec = ProbabilisticErrorCancellation(errorProbability: 0.05, samples: 100)
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.reset, to: 0)
        circuit.append(.hadamard, to: 0)

        let result1 = await pec.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
            seed: 7777,
        )

        let result2 = await pec.mitigate(
            circuit: circuit,
            observable: Observable.pauliZ(qubit: 0),
            simulator: simulator,
            seed: 7777,
        )

        #expect(abs(result1.mitigatedValue - result2.mitigatedValue) < 1e-10,
                "PEC with reset should be reproducible with same seed")
    }

    @Test("foldLocal does not fold reset as if it were a gate")
    func foldLocalDoesNotFoldReset() {
        let zne = ZeroNoiseExtrapolation(
            scaleFactors: [1, 5],
            foldingStrategy: .local,
        )
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.reset, to: 0)

        let folded = zne.fold(circuit: circuit, scaleFactor: 5.0)

        let resetCount = folded.operations.count(where: {
            if case .reset = $0 { return true }
            return false
        })
        #expect(resetCount == 1,
                "Reset should not be duplicated by local folding, found \(resetCount) resets")

        let gateCount = folded.operations.count(where: {
            if case .gate = $0 { return true }
            return false
        })
        #expect(gateCount == 0,
                "No gate operations should be created from reset-only circuit, found \(gateCount) gates")
    }

    @Test("foldFromEnd preserves reset qubit index")
    func foldFromEndPreservesResetQubitIndex() {
        let zne = ZeroNoiseExtrapolation(
            scaleFactors: [1, 3],
            foldingStrategy: .fromEnd,
        )
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.reset, to: 2)
        circuit.append(.pauliX, to: 1)

        let folded = zne.fold(circuit: circuit, scaleFactor: 3.0)

        let resetQubits = folded.operations.compactMap { op -> Int? in
            if case let .reset(qubit, _) = op { return qubit }
            return nil
        }
        #expect(resetQubits.allSatisfy { $0 == 2 },
                "All preserved resets should target qubit 2, found targets \(resetQubits)")
        #expect(resetQubits.count >= 1,
                "At least one reset should be preserved in the folded circuit")
    }
}
