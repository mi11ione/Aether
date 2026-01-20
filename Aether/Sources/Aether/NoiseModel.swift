// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Measurement error model using confusion matrix.
///
/// Models readout errors where the measured outcome may differ from the actual qubit state.
/// Characterized by confusion matrix M where M[i][j] = P(measure j | prepared i).
/// Typical error rates: 1-5% on current NISQ devices.
///
/// **Example:**
/// ```swift
/// // 1% readout error in both directions
/// let model = MeasurementErrorModel(
///     p0Given1: 0.01,  // P(measure 0 | prepared 1)
///     p1Given0: 0.01   // P(measure 1 | prepared 0)
/// )
/// ```
///
/// - SeeAlso: ``NoiseModel`` for circuit-level noise configuration
@frozen
public struct MeasurementErrorModel: Sendable, Equatable {
    /// Confusion matrix M[prepared][measured].
    ///
    /// - M[0][0] = P(measure 0 | prepared 0) = 1 - p1Given0
    /// - M[0][1] = P(measure 1 | prepared 0) = p1Given0
    /// - M[1][0] = P(measure 0 | prepared 1) = p0Given1
    /// - M[1][1] = P(measure 1 | prepared 1) = 1 - p0Given1
    public let confusionMatrix: [[Double]]

    /// Inverse confusion matrix for error mitigation.
    ///
    /// Used to correct measured probability distributions: P_corrected = M⁻¹ P_measured.
    /// Precomputed at initialization for efficient mitigation.
    public let inverseMatrix: [[Double]]

    /// Create measurement error model from error probabilities.
    ///
    /// **Example:**
    /// ```swift
    /// let model = MeasurementErrorModel(p0Given1: 0.02, p1Given0: 0.01)
    /// ```
    ///
    /// - Parameters:
    ///   - p0Given1: Probability of measuring 0 when state is 1 (relaxation during readout)
    ///   - p1Given0: Probability of measuring 1 when state is 0 (excitation during readout)
    /// - Precondition: 0 ≤ p0Given1 ≤ 1
    /// - Precondition: 0 ≤ p1Given0 ≤ 1
    /// - Precondition: Resulting confusion matrix must be invertible
    public init(p0Given1: Double, p1Given0: Double) {
        ValidationUtilities.validateErrorProbability(p0Given1, name: "p0Given1")
        ValidationUtilities.validateErrorProbability(p1Given0, name: "p1Given0")

        confusionMatrix = [
            [1.0 - p1Given0, p1Given0],
            [p0Given1, 1.0 - p0Given1],
        ]

        let det = (1.0 - p1Given0) * (1.0 - p0Given1) - p1Given0 * p0Given1
        ValidationUtilities.validateNonSingularDeterminant(det, name: "Confusion matrix")

        let invDet = 1.0 / det
        inverseMatrix = [
            [(1.0 - p0Given1) * invDet, -p1Given0 * invDet],
            [-p0Given1 * invDet, (1.0 - p1Given0) * invDet],
        ]
    }

    /// Create measurement error model from explicit confusion matrix.
    ///
    /// **Example:**
    /// ```swift
    /// let matrix = [[0.98, 0.02], [0.03, 0.97]]
    /// let model = MeasurementErrorModel(confusionMatrix: matrix)
    /// ```
    ///
    /// - Parameter confusionMatrix: 2*2 stochastic matrix with rows summing to 1
    /// - Precondition: Matrix must be 2*2 with elements in [0,1] and rows summing to 1
    /// - Precondition: Matrix must be invertible (non-singular)
    public init(confusionMatrix: [[Double]]) {
        ValidationUtilities.validateConfusionMatrix(confusionMatrix)

        self.confusionMatrix = confusionMatrix

        let a = confusionMatrix[0][0]
        let b = confusionMatrix[0][1]
        let c = confusionMatrix[1][0]
        let d = confusionMatrix[1][1]

        let det = a * d - b * c
        ValidationUtilities.validateNonSingularDeterminant(det, name: "Confusion matrix")

        let invDet = 1.0 / det
        inverseMatrix = [
            [d * invDet, -b * invDet],
            [-c * invDet, a * invDet],
        ]
    }

    /// Apply measurement error to probability distribution.
    ///
    /// Transforms ideal probabilities to noisy probabilities: P_noisy = M * P_ideal.
    ///
    /// - Parameter probabilities: Two-element array [P(0), P(1)]
    /// - Returns: Noisy probability distribution
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    public func applyError(to probabilities: (p0: Double, p1: Double)) -> (p0: Double, p1: Double) {
        let noisyP0 = confusionMatrix[0][0] * probabilities.p0 + confusionMatrix[1][0] * probabilities.p1
        let noisyP1 = confusionMatrix[0][1] * probabilities.p0 + confusionMatrix[1][1] * probabilities.p1
        return (noisyP0, noisyP1)
    }

    /// Mitigate measurement error from observed probabilities.
    ///
    /// Corrects noisy probabilities using inverse matrix: P_corrected = M⁻¹ * P_noisy.
    /// May produce negative "probabilities" which should be interpreted as statistical
    /// corrections rather than true probabilities.
    ///
    /// - Parameter probabilities: Observed (noisy) probabilities [P(0), P(1)]
    /// - Returns: Corrected probability distribution
    /// - Complexity: O(1)
    /// - Note: Results may be negative or > 1 due to statistical corrections
    @_effects(readonly)
    @inlinable
    public func mitigate(probabilities: (p0: Double, p1: Double)) -> (p0: Double, p1: Double) {
        let correctedP0 = inverseMatrix[0][0] * probabilities.p0 + inverseMatrix[0][1] * probabilities.p1
        let correctedP1 = inverseMatrix[1][0] * probabilities.p0 + inverseMatrix[1][1] * probabilities.p1
        return (correctedP0, correctedP1)
    }

    /// Mitigate measurement error from histogram counts.
    ///
    /// Applies inverse confusion matrix to multi-qubit measurement histogram.
    /// Assumes independent single-qubit errors (tensor product structure).
    ///
    /// **Example:**
    /// ```swift
    /// let model = MeasurementErrorModel(p0Given1: 0.02, p1Given0: 0.01)
    /// let histogram: [Int: Int] = [0: 450, 1: 50, 2: 50, 3: 450]
    /// let corrected = model.mitigateHistogram(histogram, qubit: 0, totalQubits: 2)
    /// ```
    ///
    /// - Parameters:
    ///   - histogram: Dictionary mapping basis state indices to counts
    ///   - qubit: Qubit index to correct
    ///   - totalQubits: Total number of qubits in the system
    /// - Returns: Corrected histogram with mitigated counts
    /// - Complexity: O(2^n) where n = totalQubits
    /// - Precondition: 0 ≤ qubit < totalQubits
    @_optimize(speed)
    public func mitigateHistogram(
        _ histogram: [Int: Int],
        qubit: Int,
        totalQubits: Int,
    ) -> [Int: Double] {
        ValidationUtilities.validateQubitIndex(qubit, qubits: totalQubits)

        let mask = 1 << qubit
        var corrected: [Int: Double] = [:]

        for (state, count) in histogram {
            let bit = (state >> qubit) & 1
            let partnerState = state ^ mask

            let originalCount = Double(count)
            let partnerCount = Double(histogram[partnerState] ?? 0)

            let p0: Double
            let p1: Double

            if bit == 0 {
                p0 = originalCount
                p1 = partnerCount
            } else {
                p0 = partnerCount
                p1 = originalCount
            }

            let (corrP0, corrP1) = mitigate(probabilities: (p0, p1))

            if bit == 0 {
                corrected[state, default: 0] += corrP0
            } else {
                corrected[state, default: 0] += corrP1
            }
        }

        return corrected
    }
}

// MARK: - Noise Model

/// Complete noise model for quantum circuit simulation.
///
/// Configures noise channels for different gate types, measurement errors, and idle noise.
/// Supports per-gate-type noise configuration with typical NISQ device characteristics:
/// - Single-qubit gate error: ~0.1%
/// - Two-qubit gate error: ~1% (10x worse)
/// - Measurement error: ~1-5%
///
/// When `idleNoiseConfig` is set, qubits not involved in the current gate accumulate T₁/T₂ decay
/// during that gate's execution time, modeling realistic hardware where idle qubits decohere
/// while waiting for other operations.
///
/// **Example:**
/// ```swift
/// let model = NoiseModel(
///     singleQubitNoise: DepolarizingChannel(errorProbability: 0.001),
///     twoQubitNoise: TwoQubitDepolarizingChannel(errorProbability: 0.01),
///     measurementError: MeasurementErrorModel(p0Given1: 0.02, p1Given0: 0.01)
/// )
///
/// // With idle noise
/// let modelWithIdle = NoiseModel(
///     singleQubitNoise: DepolarizingChannel(errorProbability: 0.001),
///     twoQubitNoise: TwoQubitDepolarizingChannel(errorProbability: 0.01),
///     idleNoiseConfig: IdleNoiseConfig(t1: 100_000, t2: 80_000, timings: .ibmDefault)
/// )
/// ```
///
/// - SeeAlso: ``DensityMatrixSimulator`` for noisy circuit execution
/// - SeeAlso: ``NoiseChannel`` for available noise channels
/// - SeeAlso: ``IdleNoiseConfig`` for idle qubit decoherence configuration
@frozen
public struct NoiseModel: Sendable {
    /// Default noise channel applied after single-qubit gates.
    public let singleQubitNoise: (any NoiseChannel)?

    /// Default noise channel applied after two-qubit gates.
    public let twoQubitNoise: TwoQubitDepolarizingChannel?

    /// Measurement error model for readout confusion.
    public let measurementError: MeasurementErrorModel?

    /// Idle noise configuration for T₁/T₂ decay on non-active qubits.
    public let idleNoiseConfig: IdleNoiseConfig?

    /// Whether any noise is configured.
    @inlinable
    public var hasNoise: Bool {
        singleQubitNoise != nil || twoQubitNoise != nil || measurementError != nil || idleNoiseConfig != nil
    }

    /// Whether idle noise is configured.
    @inlinable
    public var hasIdleNoise: Bool {
        idleNoiseConfig != nil
    }

    /// Create noise model with specified noise channels.
    ///
    /// **Example:**
    /// ```swift
    /// // Minimal noise model with only depolarizing noise
    /// let model = NoiseModel(
    ///     singleQubitNoise: DepolarizingChannel(errorProbability: 0.001)
    /// )
    ///
    /// // Complete NISQ noise model with idle noise
    /// let fullModel = NoiseModel(
    ///     singleQubitNoise: DepolarizingChannel(errorProbability: 0.001),
    ///     twoQubitNoise: TwoQubitDepolarizingChannel(errorProbability: 0.01),
    ///     measurementError: MeasurementErrorModel(p0Given1: 0.02, p1Given0: 0.01),
    ///     idleNoiseConfig: IdleNoiseConfig(t1: 100_000, t2: 80_000)
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - singleQubitNoise: Noise channel for single-qubit gates (nil = no noise)
    ///   - twoQubitNoise: Noise channel for two-qubit gates (nil = no noise)
    ///   - measurementError: Measurement error model (nil = perfect readout)
    ///   - idleNoiseConfig: Idle noise configuration (nil = no idle noise)
    public init(
        singleQubitNoise: (any NoiseChannel)? = nil,
        twoQubitNoise: TwoQubitDepolarizingChannel? = nil,
        measurementError: MeasurementErrorModel? = nil,
        idleNoiseConfig: IdleNoiseConfig? = nil,
    ) {
        self.singleQubitNoise = singleQubitNoise
        self.twoQubitNoise = twoQubitNoise
        self.measurementError = measurementError
        self.idleNoiseConfig = idleNoiseConfig
    }

    // MARK: - Factory Methods

    /// Create noise-free model for comparison.
    ///
    /// **Example:**
    /// ```swift
    /// let ideal = NoiseModel.ideal
    /// ```
    public static let ideal = NoiseModel()

    /// Create depolarizing noise model with specified error rates.
    ///
    /// **Example:**
    /// ```swift
    /// let model = NoiseModel.depolarizing(
    ///     singleQubitError: 0.001,
    ///     twoQubitError: 0.01
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - singleQubitError: Error probability for single-qubit gates
    ///   - twoQubitError: Error probability for two-qubit gates
    /// - Returns: NoiseModel with depolarizing channels
    @_effects(readonly)
    public static func depolarizing(
        singleQubitError: Double,
        twoQubitError: Double,
    ) -> NoiseModel {
        NoiseModel(
            singleQubitNoise: DepolarizingChannel(errorProbability: singleQubitError),
            twoQubitNoise: TwoQubitDepolarizingChannel(errorProbability: twoQubitError),
        )
    }

    /// Create realistic NISQ device noise model.
    ///
    /// Based on typical IBM/Google superconducting qubit parameters:
    /// - Single-qubit gate error: 0.1%
    /// - Two-qubit gate error: 1%
    /// - Measurement error: 2% (asymmetric)
    ///
    /// **Example:**
    /// ```swift
    /// let model = NoiseModel.typicalNISQ
    /// ```
    public static var typicalNISQ: NoiseModel {
        NoiseModel(
            singleQubitNoise: DepolarizingChannel(errorProbability: 0.001),
            twoQubitNoise: TwoQubitDepolarizingChannel(errorProbability: 0.01),
            measurementError: MeasurementErrorModel(p0Given1: 0.02, p1Given0: 0.01),
        )
    }

    /// Create realistic NISQ device noise model with idle decoherence.
    ///
    /// Includes T₁/T₂ decay on idle qubits during gate execution.
    ///
    /// **Example:**
    /// ```swift
    /// let model = NoiseModel.typicalNISQWithIdle
    /// ```
    public static var typicalNISQWithIdle: NoiseModel {
        NoiseModel(
            singleQubitNoise: DepolarizingChannel(errorProbability: 0.001),
            twoQubitNoise: TwoQubitDepolarizingChannel(errorProbability: 0.01),
            measurementError: MeasurementErrorModel(p0Given1: 0.02, p1Given0: 0.01),
            idleNoiseConfig: IdleNoiseConfig(t1: 100_000, t2: 80000, timings: .ibmDefault),
        )
    }

    /// Create amplitude damping noise model for T₁-limited devices.
    ///
    /// **Example:**
    /// ```swift
    /// let model = NoiseModel.amplitudeDamping(t1: 100_000)
    /// ```
    ///
    /// - Parameters:
    ///   - t1: T₁ relaxation time in nanoseconds
    ///   - singleQubitGateTime: Single-qubit gate time in nanoseconds
    ///   - twoQubitGateTime: Two-qubit gate time in nanoseconds
    /// - Returns: NoiseModel with amplitude damping based on T₁
    @_effects(readonly)
    public static func amplitudeDamping(
        t1: Double,
        singleQubitGateTime: Double = 35,
        twoQubitGateTime: Double = 300,
    ) -> NoiseModel {
        let gammaSingle = 1.0 - exp(-singleQubitGateTime / t1)
        let gammaTwo = 1.0 - exp(-twoQubitGateTime / t1)

        return NoiseModel(
            singleQubitNoise: AmplitudeDampingChannel(gamma: gammaSingle),
            twoQubitNoise: TwoQubitDepolarizingChannel(errorProbability: gammaTwo),
        )
    }

    /// Create noise model from hardware profile.
    ///
    /// Uses average parameters from the hardware profile to create a uniform noise model.
    /// For per-qubit noise, use the hardware profile directly with a custom simulator.
    ///
    /// **Example:**
    /// ```swift
    /// let model = NoiseModel.from(profile: HardwareNoiseProfile.ibmManila)
    /// ```
    ///
    /// - Parameter profile: Hardware noise profile
    /// - Returns: NoiseModel based on profile averages
    @_effects(readonly)
    public static func from(profile: HardwareNoiseProfile) -> NoiseModel {
        profile.toNoiseModel()
    }

    /// Create noise model from hardware profile with idle noise.
    ///
    /// **Example:**
    /// ```swift
    /// let model = NoiseModel.fromWithIdle(profile: HardwareNoiseProfile.ibmManila)
    /// ```
    ///
    /// - Parameter profile: Hardware noise profile
    /// - Returns: NoiseModel with idle noise based on profile T₁/T₂
    @_effects(readonly)
    public static func fromWithIdle(profile: HardwareNoiseProfile) -> NoiseModel {
        let base = profile.toNoiseModel()
        let idleConfig = IdleNoiseConfig(
            t1: profile.averageT1,
            t2: profile.averageT2,
            timings: profile.gateTimings,
        )
        return NoiseModel(
            singleQubitNoise: base.singleQubitNoise,
            twoQubitNoise: base.twoQubitNoise,
            measurementError: base.measurementError,
            idleNoiseConfig: idleConfig,
        )
    }
}

// MARK: - Idle Noise Configuration

/// Configuration for idle qubit noise (T₁/T₂ decay).
///
/// Models decoherence on qubits not involved in the current gate operation.
/// During a gate on qubits (0, 1), qubits 2, 3, ... experience T₁ and T₂ decay
/// proportional to the gate's execution time.
///
/// T₁ decay uses amplitude damping with γ = 1 - exp(-t/T₁), while T₂ decay uses phase damping with
/// γ = 1 - exp(-t/T_φ) where 1/T₂ = 1/(2T₁) + 1/T_φ.
///
/// **Example:**
/// ```swift
/// let idleConfig = IdleNoiseConfig(
///     t1: 100_000,      // 100 μs T₁
///     t2: 80_000,       // 80 μs T₂
///     timings: .ibmDefault
/// )
/// ```
@frozen
public struct IdleNoiseConfig: Sendable, Equatable {
    /// T₁ relaxation time in nanoseconds.
    public let t1: Double

    /// T₂ coherence time in nanoseconds.
    public let t2: Double

    /// Gate timing model for duration lookup.
    public let timings: GateTimingModel

    /// Per-qubit T₁ times (if different across qubits).
    public let perQubitT1: [Double]?

    /// Per-qubit T₂ times (if different across qubits).
    public let perQubitT2: [Double]?

    /// Create idle noise configuration with uniform T₁/T₂.
    ///
    /// - Parameters:
    ///   - t1: T₁ relaxation time in nanoseconds
    ///   - t2: T₂ coherence time in nanoseconds (must be ≤ 2*T₁)
    ///   - timings: Gate timing model
    /// - Precondition: t1 > 0
    /// - Precondition: t2 > 0
    /// - Precondition: t2 ≤ 2*t1
    public init(t1: Double, t2: Double, timings: GateTimingModel = .ibmDefault) {
        ValidationUtilities.validatePositiveDouble(t1, name: "T₁")
        ValidationUtilities.validatePositiveDouble(t2, name: "T₂")
        ValidationUtilities.validateT2Constraint(t2, t1: t1)

        self.t1 = t1
        self.t2 = t2
        self.timings = timings
        perQubitT1 = nil
        perQubitT2 = nil
    }

    /// Create idle noise configuration with per-qubit T₁/T₂.
    ///
    /// - Parameters:
    ///   - perQubitT1: T₁ time per qubit in nanoseconds
    ///   - perQubitT2: T₂ time per qubit in nanoseconds
    ///   - timings: Gate timing model
    /// - Precondition: perQubitT1.count == perQubitT2.count
    /// - Precondition: All T₁ values > 0
    /// - Precondition: All T₂ values > 0
    /// - Precondition: T₂[i] ≤ 2*T₁[i] for all i
    public init(perQubitT1: [Double], perQubitT2: [Double], timings: GateTimingModel = .ibmDefault) {
        ValidationUtilities.validateEqualCounts(perQubitT1, perQubitT2, name1: "perQubitT1", name2: "perQubitT2")
        ValidationUtilities.validateAllPositive(perQubitT1, name: "T₁")
        ValidationUtilities.validateAllPositive(perQubitT2, name: "T₂")

        for i in 0 ..< perQubitT1.count {
            ValidationUtilities.validateT2Constraint(perQubitT2[i], t1: perQubitT1[i], index: i)
        }

        t1 = perQubitT1.reduce(0, +) / Double(perQubitT1.count)
        t2 = perQubitT2.reduce(0, +) / Double(perQubitT2.count)
        self.timings = timings
        self.perQubitT1 = perQubitT1
        self.perQubitT2 = perQubitT2
    }

    /// Get T₁ for specific qubit.
    @_effects(readonly)
    @inlinable
    public func t1ForQubit(_ qubit: Int) -> Double {
        perQubitT1?[qubit] ?? t1
    }

    /// Get T₂ for specific qubit.
    @_effects(readonly)
    @inlinable
    public func t2ForQubit(_ qubit: Int) -> Double {
        perQubitT2?[qubit] ?? t2
    }

    /// Compute amplitude damping γ for given idle time.
    @_effects(readonly)
    @inlinable
    public func amplitudeDampingGamma(idleTime: Double, qubit: Int? = nil) -> Double {
        let t1Val = qubit.map { t1ForQubit($0) } ?? t1
        return 1.0 - exp(-idleTime / t1Val)
    }

    /// Compute phase damping γ for given idle time.
    @_effects(readonly)
    @inlinable
    public func phaseDampingGamma(idleTime: Double, qubit: Int? = nil) -> Double {
        let t1Val = qubit.map { t1ForQubit($0) } ?? t1
        let t2Val = qubit.map { t2ForQubit($0) } ?? t2

        let tPhiInverse = 1.0 / t2Val - 1.0 / (2.0 * t1Val)
        if tPhiInverse <= 0 {
            return 0
        }
        let tPhi = 1.0 / tPhiInverse
        return 1.0 - exp(-idleTime / tPhi)
    }

    /// Create amplitude damping channel for idle time on specific qubit.
    @_effects(readonly)
    public func amplitudeDampingChannel(idleTime: Double, qubit: Int? = nil) -> AmplitudeDampingChannel {
        let gamma = amplitudeDampingGamma(idleTime: idleTime, qubit: qubit)
        return AmplitudeDampingChannel(gamma: min(gamma, 1.0))
    }

    /// Create phase damping channel for idle time on specific qubit.
    @_effects(readonly)
    public func phaseDampingChannel(idleTime: Double, qubit: Int? = nil) -> PhaseDampingChannel {
        let gamma = phaseDampingGamma(idleTime: idleTime, qubit: qubit)
        return PhaseDampingChannel(gamma: min(gamma, 1.0))
    }
}

// MARK: - Noise Model Extension for Gate Application

public extension NoiseModel {
    /// Apply appropriate noise channel after gate execution.
    ///
    /// Selects noise channel based on gate type and applies to density matrix.
    ///
    /// - Parameters:
    ///   - gate: Gate that was just executed
    ///   - targetQubits: Qubits the gate was applied to
    ///   - matrix: Current density matrix state
    /// - Returns: Noisy density matrix after channel application
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    func applyNoise(
        after gate: QuantumGate,
        targetQubits: [Int],
        to matrix: DensityMatrix,
    ) -> DensityMatrix {
        switch gate.qubitsRequired {
        case 1:
            guard let noise = singleQubitNoise else { return matrix }
            return noise.apply(to: matrix, qubit: targetQubits[0])

        case 2:
            guard let noise = twoQubitNoise else { return matrix }
            return noise.apply(to: matrix, qubits: targetQubits)

        default:
            return matrix
        }
    }

    /// Apply gate noise and idle noise to all affected qubits.
    ///
    /// This method:
    /// 1. Applies gate-specific noise to active qubits
    /// 2. Applies T₁/T₂ decay to idle qubits based on gate duration
    ///
    /// - Parameters:
    ///   - gate: Gate that was just executed
    ///   - targetQubits: Qubits the gate was applied to
    ///   - matrix: Current density matrix state
    ///   - totalQubits: Total number of qubits in the system
    /// - Returns: Noisy density matrix after all noise channels
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    func applyNoiseWithIdle(
        after gate: QuantumGate,
        targetQubits: [Int],
        to matrix: DensityMatrix,
        totalQubits: Int,
    ) -> DensityMatrix {
        var result = applyNoise(after: gate, targetQubits: targetQubits, to: matrix)

        guard let idleConfig = idleNoiseConfig else {
            return result
        }

        let activeSet = Set(targetQubits)
        let gateTime = idleConfig.timings.gateTime(for: gate.qubitsRequired)

        for qubit in 0 ..< totalQubits where !activeSet.contains(qubit) {
            let t1Gamma = idleConfig.amplitudeDampingGamma(idleTime: gateTime, qubit: qubit)
            let t2Gamma = idleConfig.phaseDampingGamma(idleTime: gateTime, qubit: qubit)

            if t1Gamma > 1e-12 {
                let t1Channel = AmplitudeDampingChannel(gamma: min(t1Gamma, 1.0))
                result = t1Channel.apply(to: result, qubit: qubit)
            }

            if t2Gamma > 1e-12 {
                let t2Channel = PhaseDampingChannel(gamma: min(t2Gamma, 1.0))
                result = t2Channel.apply(to: result, qubit: qubit)
            }
        }

        return result
    }
}

// MARK: - Timing-Aware Noise Model

/// Extended noise model with full timing-aware simulation capabilities.
///
/// Wraps a `NoiseModel` with a `HardwareNoiseProfile` to provide per-qubit noise
/// parameters and accurate timing-based decoherence calculation.
///
/// **Example:**
/// ```swift
/// let profile = HardwareNoiseProfile.ibmManila
/// let timingModel = TimingAwareNoiseModel(profile: profile)
/// let result = await timingModel.execute(circuit)
/// ```
@frozen
public struct TimingAwareNoiseModel: Sendable {
    /// Hardware profile with per-qubit parameters.
    public let profile: HardwareNoiseProfile

    /// Idle noise configuration with per-qubit T₁/T₂.
    public let idleConfig: IdleNoiseConfig

    /// Create timing-aware model from hardware profile.
    ///
    /// - Parameter profile: Hardware noise profile with per-qubit parameters
    public init(profile: HardwareNoiseProfile) {
        self.profile = profile
        idleConfig = IdleNoiseConfig(
            perQubitT1: profile.qubitParameters.map(\.t1),
            perQubitT2: profile.qubitParameters.map(\.t2),
            timings: profile.gateTimings,
        )
    }

    /// Apply per-qubit gate noise for single-qubit gate.
    ///
    /// - Parameters:
    ///   - qubit: Target qubit
    ///   - matrix: Density matrix
    /// - Returns: Noisy density matrix
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public func applySingleQubitNoise(qubit: Int, to matrix: DensityMatrix) -> DensityMatrix {
        let channel = profile.singleQubitChannel(for: qubit)
        return channel.apply(to: matrix, qubit: qubit)
    }

    /// Apply per-edge gate noise for two-qubit gate.
    ///
    /// - Parameters:
    ///   - qubits: Target qubit pair
    ///   - matrix: Density matrix
    /// - Returns: Noisy density matrix
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public func applyTwoQubitNoise(qubits: [Int], to matrix: DensityMatrix) -> DensityMatrix {
        let channel = profile.twoQubitChannel(for: qubits[0], qubits[1])
        return channel.apply(to: matrix, qubits: qubits)
    }

    /// Apply idle noise to all non-active qubits.
    ///
    /// - Parameters:
    ///   - activeQubits: Qubits involved in current gate
    ///   - gateTime: Duration of gate in nanoseconds
    ///   - matrix: Density matrix
    /// - Returns: Density matrix with idle decoherence applied
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public func applyIdleNoise(
        activeQubits: Set<Int>,
        gateTime: Double,
        to matrix: DensityMatrix,
    ) -> DensityMatrix {
        var result = matrix

        for qubit in 0 ..< profile.qubitCount where !activeQubits.contains(qubit) {
            let t1Gamma = idleConfig.amplitudeDampingGamma(idleTime: gateTime, qubit: qubit)
            let t2Gamma = idleConfig.phaseDampingGamma(idleTime: gateTime, qubit: qubit)

            if t1Gamma > 1e-12 {
                let t1Channel = AmplitudeDampingChannel(gamma: min(t1Gamma, 1.0))
                result = t1Channel.apply(to: result, qubit: qubit)
            }

            if t2Gamma > 1e-12 {
                let t2Channel = PhaseDampingChannel(gamma: min(t2Gamma, 1.0))
                result = t2Channel.apply(to: result, qubit: qubit)
            }
        }

        return result
    }

    /// Apply all noise for a gate operation.
    ///
    /// Combines gate-specific noise with idle decoherence on all qubits.
    ///
    /// - Parameters:
    ///   - gate: Gate that was executed
    ///   - targetQubits: Qubits the gate acted on
    ///   - matrix: Current density matrix
    /// - Returns: Fully noisy density matrix
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public func applyAllNoise(
        after gate: QuantumGate,
        targetQubits: [Int],
        to matrix: DensityMatrix,
    ) -> DensityMatrix {
        var result = matrix

        switch gate.qubitsRequired {
        case 1:
            result = applySingleQubitNoise(qubit: targetQubits[0], to: result)
        case 2:
            result = applyTwoQubitNoise(qubits: targetQubits, to: result)
        default:
            break
        }

        let activeSet = Set(targetQubits)
        let gateTime = profile.gateTimings.gateTime(for: gate.qubitsRequired)
        result = applyIdleNoise(activeQubits: activeSet, gateTime: gateTime, to: result)

        return result
    }

    /// Get per-qubit measurement error models.
    public func measurementErrorModels() -> [MeasurementErrorModel] {
        profile.measurementErrorModels()
    }
}
