// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under Apache 2.0

import Accelerate
import Foundation

/// Result of quantum phase estimation encoding the estimated eigenvalue phase.
///
/// Captures the estimated phase phi from eigenvalue equation U|psi> = e^(2*pi*i*phi)|psi>,
/// along with measurement statistics and precision analysis. The phase is extracted from
/// the measurement outcome using binary fraction representation.
///
/// **Example:**
/// ```swift
/// let circuit = QuantumCircuit.phaseEstimation(unitary: .pauliZ, precisionQubits: 4, eigenstateQubits: 1)
/// let state = circuit.execute()
/// let result = state.phaseEstimationResult(precisionQubits: 4)
/// print(result.estimatedPhase)  // Phase in [0, 1)
/// print(result.theoreticalPrecision)  // 1/16 for 4 bits
/// ```
///
/// - SeeAlso: ``QuantumCircuit/phaseEstimation(unitary:precisionQubits:eigenstateQubits:)``
/// - SeeAlso: ``PhasePrecisionAnalysis``
@frozen
public struct PhaseEstimationResult: Sendable, CustomStringConvertible {
    /// Estimated phase in the range [0, 1) where eigenvalue = e^(2*pi*i*phase).
    ///
    /// The phase is computed as measurementOutcome / 2^precisionBits, representing
    /// the binary fraction read from the precision register after inverse QFT.
    public let estimatedPhase: Double

    /// Raw measurement outcome from the precision register.
    ///
    /// Integer value in range [0, 2^precisionBits) representing the binary
    /// approximation of the phase.
    public let measurementOutcome: Int

    /// Number of qubits used in the precision register.
    ///
    /// Determines the resolution of phase estimation: more bits yield finer
    /// phase discrimination at the cost of deeper circuits.
    public let precisionBits: Int

    /// Theoretical precision bound: 1/2^n where n = precisionBits.
    ///
    /// The maximum absolute error in phase estimation assuming perfect
    /// quantum operations and exact eigenstate preparation.
    public let theoreticalPrecision: Double

    /// Probability of obtaining this measurement outcome.
    ///
    /// For exact phases (phi = k/2^n), success probability is 1.
    /// For inexact phases, the probability distribution peaks near the
    /// closest binary fraction with minimum success >= 4/pi^2 ≈ 0.405.
    public let successProbability: Double

    /// Creates a phase estimation result with all components.
    ///
    /// - Parameters:
    ///   - estimatedPhase: Phase value in [0, 1)
    ///   - measurementOutcome: Integer measurement from precision register
    ///   - precisionBits: Number of precision qubits used
    ///   - theoreticalPrecision: Maximum error bound 1/2^n
    ///   - successProbability: Probability of this outcome
    ///
    /// **Example:**
    /// ```swift
    /// let result = PhaseEstimationResult(
    ///     estimatedPhase: 0.5,
    ///     measurementOutcome: 8,
    ///     precisionBits: 4,
    ///     theoreticalPrecision: 0.0625,
    ///     successProbability: 1.0
    /// )
    /// ```
    public init(
        estimatedPhase: Double,
        measurementOutcome: Int,
        precisionBits: Int,
        theoreticalPrecision: Double,
        successProbability: Double,
    ) {
        self.estimatedPhase = estimatedPhase
        self.measurementOutcome = measurementOutcome
        self.precisionBits = precisionBits
        self.theoreticalPrecision = theoreticalPrecision
        self.successProbability = successProbability
    }

    /// Human-readable description of the phase estimation result.
    ///
    /// **Example:**
    /// ```swift
    /// let result = PhaseEstimationResult(
    ///     estimatedPhase: 0.5,
    ///     measurementOutcome: 8,
    ///     precisionBits: 4,
    ///     theoreticalPrecision: 0.0625,
    ///     successProbability: 1.0
    /// )
    /// print(result)
    /// // "PhaseEstimationResult(phase: 0.5, outcome: 8, bits: 4, precision: 0.0625, probability: 1.0)"
    /// ```
    @inlinable
    public var description: String {
        "PhaseEstimationResult(phase: \(estimatedPhase), outcome: \(measurementOutcome), bits: \(precisionBits), precision: \(theoreticalPrecision), probability: \(successProbability))"
    }

    /// Eigenvalue corresponding to the estimated phase: e^(2*pi*i*phi).
    ///
    /// Computes the complex eigenvalue from the phase using Euler's formula.
    ///
    /// **Example:**
    /// ```swift
    /// let result = PhaseEstimationResult(
    ///     estimatedPhase: 0.5,
    ///     measurementOutcome: 8,
    ///     precisionBits: 4,
    ///     theoreticalPrecision: 0.0625,
    ///     successProbability: 1.0
    /// )
    /// let eigenvalue = result.eigenvalue  // e^(i*pi) = -1
    /// ```
    @inlinable
    public var eigenvalue: Complex<Double> {
        Complex(phase: 2.0 * .pi * estimatedPhase)
    }

    /// Phase expressed in radians: 2*pi*phi.
    ///
    /// Converts the normalized phase [0, 1) to radians [0, 2*pi).
    ///
    /// **Example:**
    /// ```swift
    /// let result = PhaseEstimationResult(
    ///     estimatedPhase: 0.25,
    ///     measurementOutcome: 4,
    ///     precisionBits: 4,
    ///     theoreticalPrecision: 0.0625,
    ///     successProbability: 1.0
    /// )
    /// print(result.phaseRadians)  // pi/2 ≈ 1.5708
    /// ```
    @inlinable
    public var phaseRadians: Double {
        2.0 * .pi * estimatedPhase
    }
}

/// Precision analysis for quantum phase estimation algorithm configuration.
///
/// Provides guidance on required precision qubits based on target accuracy
/// and success probability constraints. Use this to determine optimal
/// circuit parameters before running QPE.
///
/// **Example:**
/// ```swift
/// let analysis = QuantumCircuit.phaseEstimationPrecision(
///     targetPrecision: 0.001,
///     minSuccessProbability: 0.95
/// )
/// print("Need \(analysis.precisionBits) qubits")
/// print("Max error: \(analysis.maxAbsoluteError)")
/// ```
///
/// - SeeAlso: ``PhaseEstimationResult``
/// - SeeAlso: ``QuantumCircuit/phaseEstimation(unitary:precisionQubits:eigenstateQubits:)``
@frozen
public struct PhasePrecisionAnalysis: Sendable {
    /// Number of precision qubits required to achieve target accuracy.
    ///
    /// Computed as ceil(log2(1/targetPrecision)) plus additional bits
    /// needed to meet success probability requirements.
    public let precisionBits: Int

    /// Minimum guaranteed success probability for this configuration.
    ///
    /// For exact phases, probability is 1. For general phases with delta
    /// bits of tolerance, P >= 1 - 1/(2*(delta-1)) when delta >= 2.
    /// The base success probability without tolerance is >= 4/pi^2 ≈ 0.405.
    public let minSuccessProbability: Double

    /// Maximum absolute error in phase estimation: 1/2^n.
    ///
    /// The estimated phase differs from the true phase by at most this value.
    public let maxAbsoluteError: Double

    /// Indicates whether Float64 precision is required for accurate computation.
    ///
    /// True when precisionBits > 15, as single-precision floating point
    /// cannot accurately represent phases with more than ~7 decimal digits.
    public let isFloat64Required: Bool

    /// Creates a precision analysis with all components.
    ///
    /// - Parameters:
    ///   - precisionBits: Number of qubits needed
    ///   - minSuccessProbability: Guaranteed minimum success probability
    ///   - maxAbsoluteError: Maximum phase estimation error
    ///   - isFloat64Required: Whether double precision is necessary
    ///
    /// **Example:**
    /// ```swift
    /// let analysis = PhasePrecisionAnalysis(
    ///     precisionBits: 10,
    ///     minSuccessProbability: 0.9,
    ///     maxAbsoluteError: 0.0009765625,
    ///     isFloat64Required: false
    /// )
    /// ```
    public init(
        precisionBits: Int,
        minSuccessProbability: Double,
        maxAbsoluteError: Double,
        isFloat64Required: Bool,
    ) {
        self.precisionBits = precisionBits
        self.minSuccessProbability = minSuccessProbability
        self.maxAbsoluteError = maxAbsoluteError
        self.isFloat64Required = isFloat64Required
    }
}

public extension QuantumCircuit {
    /// Analyzes precision requirements for quantum phase estimation.
    ///
    /// Computes the number of precision qubits needed to achieve target accuracy
    /// with specified success probability. Accounts for both the binary precision
    /// limit and probabilistic error from non-exact phases.
    ///
    /// The analysis computes base precision as `ceil(log2(1/targetPrecision))`, adds
    /// extra bits to meet the success probability bound P >= 1 - 1/(2*(delta-1)), and
    /// flags Float64 requirement when precision exceeds 15 bits.
    ///
    /// **Example:**
    /// ```swift
    /// let analysis = QuantumCircuit.phaseEstimationPrecision(
    ///     targetPrecision: 0.01,
    ///     minSuccessProbability: 0.9
    /// )
    /// print(analysis.precisionBits)  // Recommended qubit count
    /// print(analysis.maxAbsoluteError)  // Actual achievable precision
    /// ```
    ///
    /// - Parameters:
    ///   - targetPrecision: Desired maximum error in phase (must be positive)
    ///   - minSuccessProbability: Minimum acceptable success probability (default: 0.9, must be in (0, 1))
    /// - Returns: Analysis containing recommended configuration
    /// - Precondition: targetPrecision > 0
    /// - Precondition: 0 < minSuccessProbability < 1
    /// - Complexity: O(1)
    ///
    /// - SeeAlso: ``phaseEstimation(unitary:precisionQubits:eigenstateQubits:)``
    /// - SeeAlso: ``PhasePrecisionAnalysis``
    @_effects(readonly)
    @_optimize(speed)
    static func phaseEstimationPrecision(
        targetPrecision: Double,
        minSuccessProbability: Double = 0.9,
    ) -> PhasePrecisionAnalysis {
        ValidationUtilities.validatePositiveDouble(targetPrecision, name: "targetPrecision")
        ValidationUtilities.validateOpenRange(minSuccessProbability, min: 0.0, max: 1.0, name: "minSuccessProbability")

        let baseBits = Int(Foundation.ceil(Foundation.log2(1.0 / targetPrecision)))

        var extraBits = 0
        if minSuccessProbability > PhaseEstimationConstants.baseSuccessProbability {
            let requiredDelta = 1.0 / (2.0 * (1.0 - minSuccessProbability)) + 1.0
            extraBits = max(0, Int(Foundation.ceil(requiredDelta)) - 1)
        }

        let totalBits = max(1, baseBits + extraBits)
        let maxError = exp2(-Double(totalBits))

        let achievedProbability: Double = if extraBits >= 2 {
            1.0 - 1.0 / (2.0 * Double(extraBits - 1))
        } else {
            PhaseEstimationConstants.baseSuccessProbability
        }

        let needsFloat64 = totalBits > 15

        return PhasePrecisionAnalysis(
            precisionBits: totalBits,
            minSuccessProbability: achievedProbability,
            maxAbsoluteError: maxError,
            isFloat64Required: needsFloat64,
        )
    }

    /// Extracts phase value from a QPE measurement outcome.
    ///
    /// Converts the integer measurement result from the precision register
    /// into a phase value in [0, 1) using binary fraction representation:
    /// phi = measurementOutcome / 2^precisionBits.
    ///
    /// **Example:**
    /// ```swift
    /// let phase = QuantumCircuit.phase(from: 8, precisionBits: 4)
    /// print(phase)  // 0.5 (since 8/16 = 0.5)
    /// ```
    ///
    /// - Parameters:
    ///   - measurementOutcome: Integer result from precision register measurement
    ///   - precisionBits: Number of qubits in the precision register
    /// - Returns: Phase value in range [0, 1)
    /// - Precondition: measurementOutcome >= 0
    /// - Precondition: precisionBits >= 1
    /// - Precondition: measurementOutcome < 2^precisionBits
    /// - Complexity: O(1)
    ///
    /// - SeeAlso: ``PhaseEstimationResult``
    @_effects(readonly)
    @_optimize(speed)
    @inlinable
    static func phase(from measurementOutcome: Int, precisionBits: Int) -> Double {
        ValidationUtilities.validateNonNegativeInt(measurementOutcome, name: "measurementOutcome")
        ValidationUtilities.validatePositiveInt(precisionBits, name: "precisionBits")

        let stateSpaceSize = 1 << precisionBits
        ValidationUtilities.validateIndexInBounds(measurementOutcome, bound: stateSpaceSize, name: "measurementOutcome")

        return Double(measurementOutcome) / Double(stateSpaceSize)
    }
}

public extension QuantumState {
    /// Extracts phase estimation result from the final quantum state.
    ///
    /// Analyzes the quantum state after QPE circuit execution to extract
    /// the estimated phase. Uses the most probable measurement outcome
    /// from the precision register and computes associated statistics.
    ///
    /// The precision register is assumed to occupy qubits 0 through
    /// (precisionQubits - 1), following the standard QPE circuit layout.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.phaseEstimation(
    ///     unitary: .pauliZ,
    ///     precisionQubits: 4,
    ///     eigenstateQubits: 1
    /// )
    /// let state = circuit.execute()
    /// let result = state.phaseEstimationResult(precisionQubits: 4)
    /// print(result.estimatedPhase)
    /// print(result.successProbability)
    /// ```
    ///
    /// - Parameter precisionQubits: Number of qubits in the precision register
    /// - Returns: Phase estimation result with phase, outcome, and statistics
    /// - Precondition: precisionQubits >= 1
    /// - Precondition: precisionQubits <= qubits
    /// - Complexity: O(2^n) where n = total qubits
    ///
    /// - SeeAlso: ``PhaseEstimationResult``
    /// - SeeAlso: ``QuantumCircuit/phaseEstimation(unitary:precisionQubits:eigenstateQubits:)``
    @_optimize(speed)
    @_effects(readonly)
    func phaseEstimationResult(precisionQubits: Int) -> PhaseEstimationResult {
        ValidationUtilities.validatePositiveInt(precisionQubits, name: "precisionQubits")
        ValidationUtilities.validateUpperBound(precisionQubits, max: qubits, name: "precisionQubits")

        let precisionStateSize = 1 << precisionQubits

        var precisionProbabilities = [Double](unsafeUninitializedCapacity: precisionStateSize) {
            buffer, count in
            buffer.initialize(repeating: 0.0)
            count = precisionStateSize
        }

        let precisionMask = precisionStateSize - 1
        for basisIndex in 0 ..< stateSpaceSize {
            let precisionIndex = basisIndex & precisionMask
            let probability = amplitudes[basisIndex].magnitudeSquared
            precisionProbabilities[precisionIndex] += probability
        }

        var maxProbability = 0.0
        var maxIndex: vDSP_Length = 0
        vDSP_maxviD(precisionProbabilities, 1, &maxProbability, &maxIndex, vDSP_Length(precisionStateSize))
        let maxPrecisionIndex = Int(maxIndex)

        let reciprocalStateSize = 1.0 / Double(precisionStateSize)
        let estimatedPhase = Double(maxPrecisionIndex) * reciprocalStateSize
        let theoreticalPrecision = reciprocalStateSize

        return PhaseEstimationResult(
            estimatedPhase: estimatedPhase,
            measurementOutcome: maxPrecisionIndex,
            precisionBits: precisionQubits,
            theoreticalPrecision: theoreticalPrecision,
            successProbability: maxProbability,
        )
    }
}

/// Constants for phase estimation probability calculations.
private enum PhaseEstimationConstants {
    /// Minimum success probability for single-round QPE: 4/pi^2.
    @inline(__always)
    static var baseSuccessProbability: Double {
        4.0 / (.pi * .pi)
    }
}
