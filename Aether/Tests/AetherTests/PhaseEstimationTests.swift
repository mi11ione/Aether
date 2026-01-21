// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under Apache 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for PhaseEstimationResult initialization and stored properties.
/// Validates that all properties are correctly stored and accessible
/// after initialization with various parameter combinations.
@Suite("PhaseEstimationResult Initialization")
struct PhaseEstimationResultInitializationTests {
    @Test("All properties are correctly stored after initialization")
    func allPropertiesStoredCorrectly() {
        let result = PhaseEstimationResult(
            estimatedPhase: 0.5,
            measurementOutcome: 8,
            precisionBits: 4,
            theoreticalPrecision: 0.0625,
            successProbability: 1.0,
        )

        #expect(
            abs(result.estimatedPhase - 0.5) < 1e-10,
            "estimatedPhase should be 0.5",
        )
        #expect(
            result.measurementOutcome == 8,
            "measurementOutcome should be 8",
        )
        #expect(
            result.precisionBits == 4,
            "precisionBits should be 4",
        )
        #expect(
            abs(result.theoreticalPrecision - 0.0625) < 1e-10,
            "theoreticalPrecision should be 0.0625",
        )
        #expect(
            abs(result.successProbability - 1.0) < 1e-10,
            "successProbability should be 1.0",
        )
    }

    @Test("Zero phase initialization stores correctly")
    func zeroPhaseInitialization() {
        let result = PhaseEstimationResult(
            estimatedPhase: 0.0,
            measurementOutcome: 0,
            precisionBits: 3,
            theoreticalPrecision: 0.125,
            successProbability: 0.95,
        )

        #expect(
            abs(result.estimatedPhase) < 1e-10,
            "estimatedPhase should be 0.0",
        )
        #expect(
            result.measurementOutcome == 0,
            "measurementOutcome should be 0",
        )
    }

    @Test("High precision configuration stores correctly")
    func highPrecisionConfiguration() {
        let result = PhaseEstimationResult(
            estimatedPhase: 0.123456789,
            measurementOutcome: 126,
            precisionBits: 10,
            theoreticalPrecision: 1.0 / 1024.0,
            successProbability: 0.85,
        )

        #expect(
            abs(result.estimatedPhase - 0.123456789) < 1e-10,
            "estimatedPhase should preserve high precision value",
        )
        #expect(
            result.precisionBits == 10,
            "precisionBits should be 10",
        )
    }
}

/// Test suite for PhaseEstimationResult computed properties.
/// Validates eigenvalue and phaseRadians calculations derived
/// from the stored estimatedPhase value.
@Suite("PhaseEstimationResult Computed Properties")
struct PhaseEstimationResultComputedPropertiesTests {
    @Test("Eigenvalue for phase 0.0 equals 1")
    func eigenvaluePhaseZero() {
        let result = PhaseEstimationResult(
            estimatedPhase: 0.0,
            measurementOutcome: 0,
            precisionBits: 4,
            theoreticalPrecision: 0.0625,
            successProbability: 1.0,
        )

        let eigenvalue = result.eigenvalue
        #expect(
            abs(eigenvalue.real - 1.0) < 1e-10,
            "eigenvalue real part should be 1.0 for phase 0",
        )
        #expect(
            abs(eigenvalue.imaginary) < 1e-10,
            "eigenvalue imaginary part should be 0.0 for phase 0",
        )
    }

    @Test("Eigenvalue for phase 0.5 equals -1")
    func eigenvaluePhaseHalf() {
        let result = PhaseEstimationResult(
            estimatedPhase: 0.5,
            measurementOutcome: 8,
            precisionBits: 4,
            theoreticalPrecision: 0.0625,
            successProbability: 1.0,
        )

        let eigenvalue = result.eigenvalue
        #expect(
            abs(eigenvalue.real - -1.0) < 1e-10,
            "eigenvalue real part should be -1.0 for phase 0.5",
        )
        #expect(
            abs(eigenvalue.imaginary) < 1e-10,
            "eigenvalue imaginary part should be 0.0 for phase 0.5",
        )
    }

    @Test("Eigenvalue for phase 0.25 equals i")
    func eigenvaluePhaseQuarter() {
        let result = PhaseEstimationResult(
            estimatedPhase: 0.25,
            measurementOutcome: 4,
            precisionBits: 4,
            theoreticalPrecision: 0.0625,
            successProbability: 1.0,
        )

        let eigenvalue = result.eigenvalue
        #expect(
            abs(eigenvalue.real) < 1e-10,
            "eigenvalue real part should be 0.0 for phase 0.25",
        )
        #expect(
            abs(eigenvalue.imaginary - 1.0) < 1e-10,
            "eigenvalue imaginary part should be 1.0 for phase 0.25",
        )
    }

    @Test("Eigenvalue for phase 0.75 equals -i")
    func eigenvaluePhaseThreeQuarters() {
        let result = PhaseEstimationResult(
            estimatedPhase: 0.75,
            measurementOutcome: 12,
            precisionBits: 4,
            theoreticalPrecision: 0.0625,
            successProbability: 1.0,
        )

        let eigenvalue = result.eigenvalue
        #expect(
            abs(eigenvalue.real) < 1e-10,
            "eigenvalue real part should be 0.0 for phase 0.75",
        )
        #expect(
            abs(eigenvalue.imaginary - -1.0) < 1e-10,
            "eigenvalue imaginary part should be -1.0 for phase 0.75",
        )
    }

    @Test("Phase radians for phase 0.0 equals 0")
    func phaseRadiansZero() {
        let result = PhaseEstimationResult(
            estimatedPhase: 0.0,
            measurementOutcome: 0,
            precisionBits: 4,
            theoreticalPrecision: 0.0625,
            successProbability: 1.0,
        )

        #expect(
            abs(result.phaseRadians) < 1e-10,
            "phaseRadians should be 0 for phase 0",
        )
    }

    @Test("Phase radians for phase 0.5 equals pi")
    func phaseRadiansHalf() {
        let result = PhaseEstimationResult(
            estimatedPhase: 0.5,
            measurementOutcome: 8,
            precisionBits: 4,
            theoreticalPrecision: 0.0625,
            successProbability: 1.0,
        )

        #expect(
            abs(result.phaseRadians - Double.pi) < 1e-10,
            "phaseRadians should be pi for phase 0.5",
        )
    }

    @Test("Phase radians for phase 0.25 equals pi/2")
    func phaseRadiansQuarter() {
        let result = PhaseEstimationResult(
            estimatedPhase: 0.25,
            measurementOutcome: 4,
            precisionBits: 4,
            theoreticalPrecision: 0.0625,
            successProbability: 1.0,
        )

        #expect(
            abs(result.phaseRadians - Double.pi / 2.0) < 1e-10,
            "phaseRadians should be pi/2 for phase 0.25",
        )
    }
}

/// Test suite for PhaseEstimationResult description formatting.
/// Validates that the CustomStringConvertible implementation produces
/// correctly formatted human-readable output.
@Suite("PhaseEstimationResult Description")
struct PhaseEstimationResultDescriptionTests {
    @Test("Description contains all property values")
    func descriptionContainsAllValues() {
        let result = PhaseEstimationResult(
            estimatedPhase: 0.5,
            measurementOutcome: 8,
            precisionBits: 4,
            theoreticalPrecision: 0.0625,
            successProbability: 1.0,
        )

        let description = result.description
        #expect(
            description.contains("0.5"),
            "description should contain phase value",
        )
        #expect(
            description.contains("8"),
            "description should contain measurement outcome",
        )
        #expect(
            description.contains("4"),
            "description should contain precision bits",
        )
        #expect(
            description.contains("PhaseEstimationResult"),
            "description should contain type name",
        )
    }
}

/// Test suite for PhasePrecisionAnalysis initialization and properties.
/// Validates that precision analysis correctly stores configuration
/// and computes derived properties.
@Suite("PhasePrecisionAnalysis Initialization")
struct PhasePrecisionAnalysisInitializationTests {
    @Test("All properties are correctly stored after initialization")
    func allPropertiesStoredCorrectly() {
        let analysis = PhasePrecisionAnalysis(
            precisionBits: 10,
            minSuccessProbability: 0.9,
            maxAbsoluteError: 0.0009765625,
            requiresFloat64: false,
        )

        #expect(
            analysis.precisionBits == 10,
            "precisionBits should be 10",
        )
        #expect(
            abs(analysis.minSuccessProbability - 0.9) < 1e-10,
            "minSuccessProbability should be 0.9",
        )
        #expect(
            abs(analysis.maxAbsoluteError - 0.0009765625) < 1e-10,
            "maxAbsoluteError should be 1/1024",
        )
        #expect(
            analysis.requiresFloat64 == false,
            "requiresFloat64 should be false for 10 bits",
        )
    }

    @Test("High precision configuration requiring Float64")
    func highPrecisionRequiresFloat64() {
        let analysis = PhasePrecisionAnalysis(
            precisionBits: 20,
            minSuccessProbability: 0.95,
            maxAbsoluteError: 1.0 / Double(1 << 20),
            requiresFloat64: true,
        )

        #expect(
            analysis.precisionBits == 20,
            "precisionBits should be 20",
        )
        #expect(
            analysis.requiresFloat64 == true,
            "requiresFloat64 should be true for 20 bits",
        )
    }
}

/// Test suite for QuantumCircuit.phaseEstimationPrecision static method.
/// Validates precision analysis calculations for various target
/// precisions and success probability requirements.
@Suite("Phase Estimation Precision Analysis")
struct PhaseEstimationPrecisionAnalysisTests {
    @Test("Target precision 0.1 requires at least 4 bits")
    func targetPrecisionTenth() {
        let analysis = QuantumCircuit.phaseEstimationPrecision(
            targetPrecision: 0.1,
        )

        #expect(
            analysis.precisionBits >= 4,
            "0.1 precision requires at least 4 bits (1/16 = 0.0625)",
        )
        #expect(
            analysis.maxAbsoluteError <= 0.1,
            "max error should be at most target precision",
        )
    }

    @Test("Target precision 0.01 requires at least 7 bits")
    func targetPrecisionHundredth() {
        let analysis = QuantumCircuit.phaseEstimationPrecision(
            targetPrecision: 0.01,
        )

        #expect(
            analysis.precisionBits >= 7,
            "0.01 precision requires at least 7 bits (1/128 ≈ 0.0078)",
        )
        #expect(
            analysis.maxAbsoluteError <= 0.01,
            "max error should be at most target precision",
        )
    }

    @Test("Target precision 0.001 requires at least 10 bits")
    func targetPrecisionThousandth() {
        let analysis = QuantumCircuit.phaseEstimationPrecision(
            targetPrecision: 0.001,
        )

        #expect(
            analysis.precisionBits >= 10,
            "0.001 precision requires at least 10 bits (1/1024 ≈ 0.00098)",
        )
        #expect(
            analysis.maxAbsoluteError <= 0.001,
            "max error should be at most target precision",
        )
    }

    @Test("Higher success probability increases required bits")
    func higherSuccessProbabilityIncreasesRequiredBits() {
        let lowProbAnalysis = QuantumCircuit.phaseEstimationPrecision(
            targetPrecision: 0.01,
            minSuccessProbability: 0.5,
        )
        let highProbAnalysis = QuantumCircuit.phaseEstimationPrecision(
            targetPrecision: 0.01,
            minSuccessProbability: 0.99,
        )

        #expect(
            highProbAnalysis.precisionBits >= lowProbAnalysis.precisionBits,
            "higher success probability should require at least as many bits",
        )
    }

    @Test("RequiresFloat64 is false for 15 or fewer bits")
    func requiresFloat64FalseForLowPrecision() {
        let analysis = QuantumCircuit.phaseEstimationPrecision(
            targetPrecision: 0.01,
            minSuccessProbability: 0.5,
        )

        #expect(
            analysis.precisionBits <= 15,
            "precisionBits should be 15 or fewer for this configuration",
        )
        #expect(
            analysis.requiresFloat64 == false,
            "requiresFloat64 should be false for 15 or fewer bits",
        )
    }

    @Test("RequiresFloat64 is true for more than 15 bits")
    func requiresFloat64TrueForHighPrecision() {
        let analysis = QuantumCircuit.phaseEstimationPrecision(
            targetPrecision: 0.00001,
            minSuccessProbability: 0.99,
        )

        if analysis.precisionBits > 15 {
            #expect(
                analysis.requiresFloat64 == true,
                "requiresFloat64 should be true for more than 15 bits",
            )
        }
    }

    @Test("Max absolute error equals 1/2^n")
    func maxAbsoluteErrorFormula() {
        let analysis = QuantumCircuit.phaseEstimationPrecision(
            targetPrecision: 0.05,
        )

        let expectedError = 1.0 / Double(1 << analysis.precisionBits)
        #expect(
            abs(analysis.maxAbsoluteError - expectedError) < 1e-10,
            "max absolute error should equal 1/2^n",
        )
    }
}

/// Test suite for QuantumCircuit.extractPhase static method.
/// Validates phase extraction from measurement outcomes with
/// various precision configurations.
@Suite("Phase Extraction from Measurement")
struct PhaseExtractionTests {
    @Test("Measurement 0 extracts phase 0.0")
    func extractPhaseZero() {
        let phase = QuantumCircuit.extractPhase(from: 0, precisionBits: 4)

        #expect(
            abs(phase) < 1e-10,
            "measurement 0 should extract phase 0.0",
        )
    }

    @Test("Measurement 2^(n-1) extracts phase 0.5")
    func extractPhaseHalf() {
        let phase = QuantumCircuit.extractPhase(from: 8, precisionBits: 4)

        #expect(
            abs(phase - 0.5) < 1e-10,
            "measurement 8 with 4 bits should extract phase 0.5 (8/16)",
        )
    }

    @Test("Measurement 2^(n-2) extracts phase 0.25")
    func extractPhaseQuarter() {
        let phase = QuantumCircuit.extractPhase(from: 4, precisionBits: 4)

        #expect(
            abs(phase - 0.25) < 1e-10,
            "measurement 4 with 4 bits should extract phase 0.25 (4/16)",
        )
    }

    @Test("Measurement 3*2^(n-2) extracts phase 0.75")
    func extractPhaseThreeQuarters() {
        let phase = QuantumCircuit.extractPhase(from: 12, precisionBits: 4)

        #expect(
            abs(phase - 0.75) < 1e-10,
            "measurement 12 with 4 bits should extract phase 0.75 (12/16)",
        )
    }

    @Test("Phase extraction with different precision bits")
    func extractPhaseVariousPrecisions() {
        let phase3bit = QuantumCircuit.extractPhase(from: 4, precisionBits: 3)
        #expect(
            abs(phase3bit - 0.5) < 1e-10,
            "measurement 4 with 3 bits should extract phase 0.5 (4/8)",
        )

        let phase5bit = QuantumCircuit.extractPhase(from: 16, precisionBits: 5)
        #expect(
            abs(phase5bit - 0.5) < 1e-10,
            "measurement 16 with 5 bits should extract phase 0.5 (16/32)",
        )
    }

    @Test("Maximum measurement extracts phase near 1.0")
    func extractPhaseMaxMeasurement() {
        let phase = QuantumCircuit.extractPhase(from: 15, precisionBits: 4)

        #expect(
            abs(phase - 0.9375) < 1e-10,
            "measurement 15 with 4 bits should extract phase 0.9375 (15/16)",
        )
    }

    @Test("Single precision bit extracts binary phases")
    func extractPhaseSingleBit() {
        let phase0 = QuantumCircuit.extractPhase(from: 0, precisionBits: 1)
        let phase1 = QuantumCircuit.extractPhase(from: 1, precisionBits: 1)

        #expect(
            abs(phase0) < 1e-10,
            "measurement 0 with 1 bit should extract phase 0.0",
        )
        #expect(
            abs(phase1 - 0.5) < 1e-10,
            "measurement 1 with 1 bit should extract phase 0.5",
        )
    }
}

/// Test suite for QuantumState.phaseEstimationResult integration.
/// Validates end-to-end phase estimation workflow using actual
/// quantum circuit execution and result extraction.
@Suite("Phase Estimation Integration")
struct PhaseEstimationIntegrationTests {
    @Test("Phase estimation result has valid probability")
    func phaseEstimationResultHasValidProbability() {
        let circuit = QuantumCircuit.phaseEstimation(
            unitary: .pauliZ,
            precisionQubits: 3,
            eigenstateQubits: 1,
        )
        let state = circuit.execute()
        let result = state.phaseEstimationResult(precisionQubits: 3)

        #expect(
            result.successProbability >= 0.0,
            "success probability should be non-negative",
        )
        #expect(
            result.successProbability <= 1.0,
            "success probability should be at most 1.0",
        )
    }

    @Test("Phase estimation result has valid phase range")
    func phaseEstimationResultHasValidPhaseRange() {
        let circuit = QuantumCircuit.phaseEstimation(
            unitary: .pauliZ,
            precisionQubits: 3,
            eigenstateQubits: 1,
        )
        let state = circuit.execute()
        let result = state.phaseEstimationResult(precisionQubits: 3)

        #expect(
            result.estimatedPhase >= 0.0,
            "estimated phase should be non-negative",
        )
        #expect(
            result.estimatedPhase < 1.0,
            "estimated phase should be less than 1.0",
        )
    }

    @Test("Phase estimation result measurement outcome is valid")
    func phaseEstimationResultMeasurementOutcomeValid() {
        let precisionQubits = 4
        let circuit = QuantumCircuit.phaseEstimation(
            unitary: .pauliZ,
            precisionQubits: precisionQubits,
            eigenstateQubits: 1,
        )
        let state = circuit.execute()
        let result = state.phaseEstimationResult(precisionQubits: precisionQubits)

        let maxOutcome = 1 << precisionQubits
        #expect(
            result.measurementOutcome >= 0,
            "measurement outcome should be non-negative",
        )
        #expect(
            result.measurementOutcome < maxOutcome,
            "measurement outcome should be less than 2^precisionBits",
        )
    }

    @Test("Phase estimation result theoretical precision is correct")
    func phaseEstimationResultTheoreticalPrecisionCorrect() {
        let precisionQubits = 5
        let circuit = QuantumCircuit.phaseEstimation(
            unitary: .pauliZ,
            precisionQubits: precisionQubits,
            eigenstateQubits: 1,
        )
        let state = circuit.execute()
        let result = state.phaseEstimationResult(precisionQubits: precisionQubits)

        let expectedPrecision = 1.0 / Double(1 << precisionQubits)
        #expect(
            abs(result.theoreticalPrecision - expectedPrecision) < 1e-10,
            "theoretical precision should be 1/2^n",
        )
    }

    @Test("Phase estimation with S gate extracts phase near 0.25")
    func phaseEstimationWithSGate() {
        let circuit = QuantumCircuit.phaseEstimation(
            unitary: .sGate,
            precisionQubits: 4,
            eigenstateQubits: 1,
        )

        var stateWithEigenstate = QuantumState(qubits: 5)
        stateWithEigenstate.setAmplitude(0b10000, to: .one)
        stateWithEigenstate.setAmplitude(0, to: .zero)

        let finalState = circuit.execute(on: stateWithEigenstate)
        let result = finalState.phaseEstimationResult(precisionQubits: 4)

        #expect(
            result.precisionBits == 4,
            "precision bits should be 4",
        )
    }

    @Test("Phase estimation consistency between phase and outcome")
    func phaseEstimationConsistency() {
        let precisionQubits = 4
        let circuit = QuantumCircuit.phaseEstimation(
            unitary: .pauliZ,
            precisionQubits: precisionQubits,
            eigenstateQubits: 1,
        )
        let state = circuit.execute()
        let result = state.phaseEstimationResult(precisionQubits: precisionQubits)

        let computedPhase = Double(result.measurementOutcome) / Double(1 << precisionQubits)
        #expect(
            abs(result.estimatedPhase - computedPhase) < 1e-10,
            "estimated phase should equal measurementOutcome / 2^n",
        )
    }
}

/// Test suite for Float64 requirement threshold in precision analysis.
/// Validates that the requiresFloat64 flag correctly triggers
/// when precision exceeds single-precision floating point limits.
@Suite("Float64 Requirement Threshold")
struct Float64RequirementThresholdTests {
    @Test("Precision analysis for n=14 does not require Float64")
    func precisionBits14DoesNotRequireFloat64() {
        let analysis = PhasePrecisionAnalysis(
            precisionBits: 14,
            minSuccessProbability: 0.9,
            maxAbsoluteError: 1.0 / Double(1 << 14),
            requiresFloat64: false,
        )

        #expect(
            analysis.requiresFloat64 == false,
            "14 bits should not require Float64",
        )
    }

    @Test("Precision analysis for n=15 does not require Float64")
    func precisionBits15DoesNotRequireFloat64() {
        let analysis = PhasePrecisionAnalysis(
            precisionBits: 15,
            minSuccessProbability: 0.9,
            maxAbsoluteError: 1.0 / Double(1 << 15),
            requiresFloat64: false,
        )

        #expect(
            analysis.requiresFloat64 == false,
            "15 bits should not require Float64",
        )
    }

    @Test("Precision analysis for n=16 requires Float64")
    func precisionBits16RequiresFloat64() {
        let analysis = PhasePrecisionAnalysis(
            precisionBits: 16,
            minSuccessProbability: 0.9,
            maxAbsoluteError: 1.0 / Double(1 << 16),
            requiresFloat64: true,
        )

        #expect(
            analysis.requiresFloat64 == true,
            "16 bits should require Float64",
        )
    }

    @Test("Precision analysis for n=20 requires Float64")
    func precisionBits20RequiresFloat64() {
        let analysis = PhasePrecisionAnalysis(
            precisionBits: 20,
            minSuccessProbability: 0.95,
            maxAbsoluteError: 1.0 / Double(1 << 20),
            requiresFloat64: true,
        )

        #expect(
            analysis.requiresFloat64 == true,
            "20 bits should require Float64",
        )
    }

    @Test("phaseEstimationPrecision returns correct requiresFloat64 for high precision")
    func phaseEstimationPrecisionHighPrecisionRequirement() {
        let analysis = QuantumCircuit.phaseEstimationPrecision(
            targetPrecision: 0.000001,
            minSuccessProbability: 0.99,
        )

        if analysis.precisionBits > 15 {
            #expect(
                analysis.requiresFloat64 == true,
                "high precision analysis should require Float64 when bits > 15",
            )
        }
    }
}

/// Test suite for eigenvalue computation from various phases.
/// Validates that eigenvalue calculation correctly applies
/// Euler's formula e^(2*pi*i*phi) for all quadrant values.
@Suite("Eigenvalue Computation")
struct EigenvalueComputationTests {
    @Test("Eigenvalue magnitude is always 1")
    func eigenvalueMagnitudeIsUnity() {
        let phases = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]

        for phase in phases {
            let result = PhaseEstimationResult(
                estimatedPhase: phase,
                measurementOutcome: Int(phase * 16),
                precisionBits: 4,
                theoreticalPrecision: 0.0625,
                successProbability: 1.0,
            )

            let magnitude = result.eigenvalue.magnitude
            #expect(
                abs(magnitude - 1.0) < 1e-10,
                "eigenvalue magnitude should be 1.0 for phase \(phase)",
            )
        }
    }

    @Test("Eigenvalue for phase 0.125 has correct components")
    func eigenvaluePhaseOneEighth() {
        let result = PhaseEstimationResult(
            estimatedPhase: 0.125,
            measurementOutcome: 2,
            precisionBits: 4,
            theoreticalPrecision: 0.0625,
            successProbability: 1.0,
        )

        let eigenvalue = result.eigenvalue
        let expectedReal = cos(2.0 * Double.pi * 0.125)
        let expectedImag = sin(2.0 * Double.pi * 0.125)

        #expect(
            abs(eigenvalue.real - expectedReal) < 1e-10,
            "eigenvalue real part should match cos(pi/4)",
        )
        #expect(
            abs(eigenvalue.imaginary - expectedImag) < 1e-10,
            "eigenvalue imaginary part should match sin(pi/4)",
        )
    }

    @Test("Eigenvalue for phase 0.625 has correct components")
    func eigenvaluePhaseFiveEighths() {
        let result = PhaseEstimationResult(
            estimatedPhase: 0.625,
            measurementOutcome: 10,
            precisionBits: 4,
            theoreticalPrecision: 0.0625,
            successProbability: 1.0,
        )

        let eigenvalue = result.eigenvalue
        let expectedReal = cos(2.0 * Double.pi * 0.625)
        let expectedImag = sin(2.0 * Double.pi * 0.625)

        #expect(
            abs(eigenvalue.real - expectedReal) < 1e-10,
            "eigenvalue real part should match cos(5*pi/4)",
        )
        #expect(
            abs(eigenvalue.imaginary - expectedImag) < 1e-10,
            "eigenvalue imaginary part should match sin(5*pi/4)",
        )
    }
}

/// Test suite for zero extra bits branch in precision analysis.
/// Validates that when minSuccessProbability is at or below base probability
/// the zero extraBits code path returns base success probability.
@Suite("Phase Estimation Zero Extra Bits")
struct PhaseEstimationZeroExtraBitsTests {
    @Test("Low success probability triggers zero extra bits path")
    func lowSuccessProbabilityZeroExtraBits() {
        let analysis = QuantumCircuit.phaseEstimationPrecision(
            targetPrecision: 0.1,
            minSuccessProbability: 0.3,
        )

        let baseSuccessProbability = 4.0 / (Double.pi * Double.pi)
        #expect(
            abs(analysis.minSuccessProbability - baseSuccessProbability) < 1e-10,
            "achieved probability should equal base success probability when extraBits is 0",
        )
    }
}

/// Test suite for phase radians computation from various phases.
/// Validates correct conversion from normalized phase [0,1) to
/// radians [0, 2*pi) representation.
@Suite("Phase Radians Computation")
struct PhaseRadiansComputationTests {
    @Test("Phase radians for phase 0.125 equals pi/4")
    func phaseRadiansOneEighth() {
        let result = PhaseEstimationResult(
            estimatedPhase: 0.125,
            measurementOutcome: 2,
            precisionBits: 4,
            theoreticalPrecision: 0.0625,
            successProbability: 1.0,
        )

        #expect(
            abs(result.phaseRadians - Double.pi / 4.0) < 1e-10,
            "phaseRadians should be pi/4 for phase 0.125",
        )
    }

    @Test("Phase radians for phase 0.75 equals 3*pi/2")
    func phaseRadiansThreeQuarters() {
        let result = PhaseEstimationResult(
            estimatedPhase: 0.75,
            measurementOutcome: 12,
            precisionBits: 4,
            theoreticalPrecision: 0.0625,
            successProbability: 1.0,
        )

        #expect(
            abs(result.phaseRadians - 3.0 * Double.pi / 2.0) < 1e-10,
            "phaseRadians should be 3*pi/2 for phase 0.75",
        )
    }

    @Test("Phase radians increases linearly with phase")
    func phaseRadiansLinearRelationship() {
        for i in 0 ..< 16 {
            let phase = Double(i) / 16.0
            let result = PhaseEstimationResult(
                estimatedPhase: phase,
                measurementOutcome: i,
                precisionBits: 4,
                theoreticalPrecision: 0.0625,
                successProbability: 1.0,
            )

            let expectedRadians = 2.0 * Double.pi * phase
            #expect(
                abs(result.phaseRadians - expectedRadians) < 1e-10,
                "phaseRadians should equal 2*pi*phase for phase \(phase)",
            )
        }
    }
}
