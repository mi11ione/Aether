// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Protocol for quantum noise channels represented via Kraus operators.
///
/// A quantum channel E is a completely positive trace-preserving (CPTP) map represented by
/// Kraus operators {Kᵢ}: E(ρ) = Σᵢ Kᵢ ρ Kᵢ†. The completeness relation Σᵢ Kᵢ†Kᵢ = I ensures
/// trace preservation (probability conservation). Noise channels model decoherence, gate errors,
/// and environmental interactions in NISQ devices.
///
/// Any CPTP map can be written with at most d² Kraus operators for d-dimensional systems (2 for
/// single-qubit, 4 for two-qubit). Common noise models use 2-4 Kraus operators, making simulation
/// efficient.
///
/// **Example:**
/// ```swift
/// let channel = DepolarizingChannel(errorProbability: 0.01)
/// let noisyState = channel.apply(to: dm, qubit: 0)
/// ```
///
/// - SeeAlso: ``DensityMatrix`` for mixed state representation
/// - SeeAlso: ``DensityMatrixSimulator`` for noisy circuit execution
public protocol NoiseChannel: Sendable {
    /// Kraus operators defining the quantum channel.
    ///
    /// Each operator is a 2*2 matrix for single-qubit channels. The operators must satisfy
    /// the completeness relation Σᵢ Kᵢ†Kᵢ = I for valid quantum channels.
    var krausOperators: [[[Complex<Double>]]] { get }

    /// Apply noise channel to density matrix at specified qubit.
    ///
    /// Computes E(ρ) = Σᵢ Kᵢ ρ Kᵢ† where Kᵢ acts on the target qubit.
    ///
    /// - Parameters:
    ///   - matrix: Input density matrix
    ///   - qubit: Target qubit index
    /// - Returns: Noisy density matrix E(ρ)
    /// - Precondition: 0 ≤ qubit < matrix.qubits
    func apply(to matrix: DensityMatrix, qubit: Int) -> DensityMatrix
}

// MARK: - Default Implementation

public extension NoiseChannel {
    /// Default implementation applying Kraus operators to specified qubit.
    ///
    /// - Complexity: O(k * 4^n) where k = number of Kraus operators
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    func apply(to matrix: DensityMatrix, qubit: Int) -> DensityMatrix {
        ValidationUtilities.validateQubitIndex(qubit, qubits: matrix.qubits)

        let dim = matrix.dimension
        let size = dim * dim
        var resultElements = [Complex<Double>](repeating: .zero, count: size)

        for kraus in krausOperators {
            let transformed = applySingleKraus(kraus: kraus, matrix: matrix, qubit: qubit)
            for i in 0 ..< size {
                resultElements[i] = resultElements[i] + transformed[i]
            }
        }

        return DensityMatrix(qubits: matrix.qubits, elements: resultElements)
    }

    /// Apply single Kraus operator: KρK†.
    @_optimize(speed)
    private func applySingleKraus(
        kraus: [[Complex<Double>]],
        matrix: DensityMatrix,
        qubit: Int,
    ) -> [Complex<Double>] {
        let k00 = kraus[0][0], k01 = kraus[0][1]
        let k10 = kraus[1][0], k11 = kraus[1][1]

        let k00d = k00.conjugate, k01d = k01.conjugate
        let k10d = k10.conjugate, k11d = k11.conjugate

        let dim = matrix.dimension
        let size = dim * dim
        let mask = 1 << qubit

        let newElements = [Complex<Double>](unsafeUninitializedCapacity: size) { buffer, count in
            for row in 0 ..< dim {
                let row0 = row & ~mask
                let row1 = row | mask
                let rowBit = (row >> qubit) & 1

                for col in 0 ..< dim {
                    let col0 = col & ~mask
                    let col1 = col | mask
                    let colBit = (col >> qubit) & 1

                    let rho00 = matrix.element(row: row0, col: col0)
                    let rho01 = matrix.element(row: row0, col: col1)
                    let rho10 = matrix.element(row: row1, col: col0)
                    let rho11 = matrix.element(row: row1, col: col1)

                    let kRow0 = rowBit == 0 ? k00 : k10
                    let kRow1 = rowBit == 0 ? k01 : k11
                    let kCol0d = colBit == 0 ? k00d : k10d
                    let kCol1d = colBit == 0 ? k01d : k11d

                    let newVal = kRow0 * rho00 * kCol0d +
                        kRow0 * rho01 * kCol1d +
                        kRow1 * rho10 * kCol0d +
                        kRow1 * rho11 * kCol1d

                    buffer[row * dim + col] = newVal
                }
            }
            count = size
        }

        return newElements
    }
}

// MARK: - Depolarizing Channel

/// Depolarizing noise channel: ρ -> (1-p)ρ + (p/3)(XρX + YρY + ZρZ).
///
/// The depolarizing channel applies random Pauli errors with equal probability. It is the most
/// common simple noise model, representing "white noise" that uniformly randomizes the qubit state.
/// At p=0, no noise; at p=3/4, the output is maximally mixed regardless of input.
///
/// The Kraus operators are K₀ = √(1-p) I, K₁ = √(p/3) X, K₂ = √(p/3) Y, and K₃ = √(p/3) Z.
///
/// **Example:**
/// ```swift
/// let channel = DepolarizingChannel(errorProbability: 0.01)
/// let noisyState = channel.apply(to: dm, qubit: 0)
/// ```
///
/// - SeeAlso: ``BitFlipChannel`` for X-only errors
/// - SeeAlso: ``PhaseFlipChannel`` for Z-only errors
@frozen
public struct DepolarizingChannel: NoiseChannel {
    /// Error probability p ∈ [0, 1].
    public let errorProbability: Double

    /// Kraus operators for depolarizing channel.
    public let krausOperators: [[[Complex<Double>]]]

    /// Create depolarizing channel with given error probability.
    ///
    /// - Parameter errorProbability: Probability of error (0 = no noise, 0.75 = fully depolarizing)
    /// - Precondition: 0 ≤ errorProbability ≤ 1
    public init(errorProbability: Double) {
        ValidationUtilities.validateErrorProbability(errorProbability, name: "Depolarizing channel")

        self.errorProbability = errorProbability

        let sqrtOneMinusP = Complex<Double>(sqrt(1.0 - errorProbability), 0)
        let sqrtPOver3 = Complex<Double>(sqrt(errorProbability / 3.0), 0)

        let k0: [[Complex<Double>]] = [
            [sqrtOneMinusP, .zero],
            [.zero, sqrtOneMinusP],
        ]

        let k1: [[Complex<Double>]] = [
            [.zero, sqrtPOver3],
            [sqrtPOver3, .zero],
        ]

        let k2: [[Complex<Double>]] = [
            [.zero, Complex(0, -1) * sqrtPOver3],
            [Complex(0, 1) * sqrtPOver3, .zero],
        ]

        let k3: [[Complex<Double>]] = [
            [sqrtPOver3, .zero],
            [.zero, -sqrtPOver3],
        ]

        krausOperators = [k0, k1, k2, k3]
    }
}

// MARK: - Amplitude Damping Channel

/// Amplitude damping channel modeling energy relaxation (T₁ decay).
///
/// Models spontaneous emission where |1⟩ decays to |0⟩ with probability γ. This is the
/// dominant error in superconducting qubits due to energy loss to the environment.
/// The channel is asymmetric: |1⟩ can decay to |0⟩, but not vice versa.
///
/// T₁ relaxation time relates to γ via γ = 1 - exp(-t/T₁) ≈ t/T₁ for short gate times.
/// The Kraus operators are K₀ = [[1, 0], [0, √(1-γ)]] and K₁ = [[0, √γ], [0, 0]].
///
/// **Example:**
/// ```swift
/// let channel = AmplitudeDampingChannel(gamma: 0.01)
/// let noisyState = channel.apply(to: dm, qubit: 0)
/// ```
///
/// - SeeAlso: ``PhaseDampingChannel`` for T₂ dephasing
@frozen
public struct AmplitudeDampingChannel: NoiseChannel {
    /// Damping parameter γ ∈ [0, 1]. γ = 0 means no decay, γ = 1 means complete decay.
    public let gamma: Double

    /// Kraus operators for amplitude damping.
    public let krausOperators: [[[Complex<Double>]]]

    /// Create amplitude damping channel with given decay probability.
    ///
    /// - Parameter gamma: Decay probability (0 = no decay, 1 = complete decay to |0⟩)
    /// - Precondition: 0 ≤ gamma ≤ 1
    public init(gamma: Double) {
        ValidationUtilities.validateDampingParameter(gamma, name: "Amplitude damping")

        self.gamma = gamma

        let sqrtOneMinusGamma = Complex<Double>(sqrt(1.0 - gamma), 0)
        let sqrtGamma = Complex<Double>(sqrt(gamma), 0)

        let k0: [[Complex<Double>]] = [
            [.one, .zero],
            [.zero, sqrtOneMinusGamma],
        ]

        let k1: [[Complex<Double>]] = [
            [.zero, sqrtGamma],
            [.zero, .zero],
        ]

        krausOperators = [k0, k1]
    }
}

// MARK: - Phase Damping Channel

/// Phase damping channel modeling dephasing (T₂ decay) without energy loss.
///
/// Models loss of quantum coherence where superposition states lose their relative phase
/// information without population transfer. This is the "pure dephasing" component of T₂
/// decay (T₂ = T₁/2 + T_φ where T_φ is pure dephasing time).
///
/// Converts |+⟩ = (|0⟩+|1⟩)/√2 toward a mixed state while leaving |0⟩ and |1⟩ unchanged, with
/// off-diagonal elements decaying as ρ₀₁ -> √(1-γ)ρ₀₁. The Kraus operators are
/// K₀ = [[1, 0], [0, √(1-γ)]] and K₁ = [[0, 0], [0, √γ]].
///
/// **Example:**
/// ```swift
/// let channel = PhaseDampingChannel(gamma: 0.01)
/// let noisyState = channel.apply(to: dm, qubit: 0)
/// ```
///
/// - SeeAlso: ``AmplitudeDampingChannel`` for T₁ decay
@frozen
public struct PhaseDampingChannel: NoiseChannel {
    /// Dephasing parameter γ ∈ [0, 1]. γ = 0 means no dephasing, γ = 1 means complete dephasing.
    public let gamma: Double

    /// Kraus operators for phase damping.
    public let krausOperators: [[[Complex<Double>]]]

    /// Create phase damping channel with given dephasing probability.
    ///
    /// - Parameter gamma: Dephasing probability (0 = no dephasing, 1 = complete dephasing)
    /// - Precondition: 0 ≤ gamma ≤ 1
    public init(gamma: Double) {
        ValidationUtilities.validateDampingParameter(gamma, name: "Phase damping")

        self.gamma = gamma

        let sqrtOneMinusGamma = Complex<Double>(sqrt(1.0 - gamma), 0)
        let sqrtGamma = Complex<Double>(sqrt(gamma), 0)

        let k0: [[Complex<Double>]] = [
            [.one, .zero],
            [.zero, sqrtOneMinusGamma],
        ]

        let k1: [[Complex<Double>]] = [
            [.zero, .zero],
            [.zero, sqrtGamma],
        ]

        krausOperators = [k0, k1]
    }
}

// MARK: - Bit Flip Channel

/// Bit flip channel: ρ -> (1-p)ρ + pXρX.
///
/// Applies Pauli X error (bit flip) with probability p. This is a classical-like error model
/// where |0⟩ can flip to |1⟩ and vice versa. Commonly used in quantum error correction studies
/// as the simplest non-trivial noise model.
///
/// The Kraus operators are K₀ = √(1-p) I and K₁ = √p X.
///
/// **Example:**
/// ```swift
/// let channel = BitFlipChannel(errorProbability: 0.01)
/// let noisyState = channel.apply(to: dm, qubit: 0)
/// ```
///
/// - SeeAlso: ``PhaseFlipChannel`` for Z errors
/// - SeeAlso: ``DepolarizingChannel`` for combined X, Y, Z errors
@frozen
public struct BitFlipChannel: NoiseChannel {
    /// Bit flip probability p ∈ [0, 1].
    public let errorProbability: Double

    /// Kraus operators for bit flip channel.
    public let krausOperators: [[[Complex<Double>]]]

    /// Create bit flip channel with given error probability.
    ///
    /// - Parameter errorProbability: Probability of X error (0 = no error, 1 = always flip)
    /// - Precondition: 0 ≤ errorProbability ≤ 1
    public init(errorProbability: Double) {
        ValidationUtilities.validateErrorProbability(errorProbability, name: "Bit flip channel")

        self.errorProbability = errorProbability

        let sqrtOneMinusP = Complex<Double>(sqrt(1.0 - errorProbability), 0)
        let sqrtP = Complex<Double>(sqrt(errorProbability), 0)

        let k0: [[Complex<Double>]] = [
            [sqrtOneMinusP, .zero],
            [.zero, sqrtOneMinusP],
        ]

        let k1: [[Complex<Double>]] = [
            [.zero, sqrtP],
            [sqrtP, .zero],
        ]

        krausOperators = [k0, k1]
    }
}

// MARK: - Phase Flip Channel

/// Phase flip channel: ρ -> (1-p)ρ + pZρZ.
///
/// Applies Pauli Z error (phase flip) with probability p. This is a purely quantum error
/// that affects relative phases in superpositions without changing populations. Important
/// for surface codes and other topological error correction schemes.
///
/// The Kraus operators are K₀ = √(1-p) I and K₁ = √p Z.
///
/// **Example:**
/// ```swift
/// let channel = PhaseFlipChannel(errorProbability: 0.01)
/// let noisyState = channel.apply(to: dm, qubit: 0)
/// ```
///
/// - SeeAlso: ``BitFlipChannel`` for X errors
/// - SeeAlso: ``DepolarizingChannel`` for combined X, Y, Z errors
@frozen
public struct PhaseFlipChannel: NoiseChannel {
    /// Phase flip probability p ∈ [0, 1].
    public let errorProbability: Double

    /// Kraus operators for phase flip channel.
    public let krausOperators: [[[Complex<Double>]]]

    /// Create phase flip channel with given error probability.
    ///
    /// - Parameter errorProbability: Probability of Z error (0 = no error, 1 = always flip phase)
    /// - Precondition: 0 ≤ errorProbability ≤ 1
    public init(errorProbability: Double) {
        ValidationUtilities.validateErrorProbability(errorProbability, name: "Phase flip channel")

        self.errorProbability = errorProbability

        let sqrtOneMinusP = Complex<Double>(sqrt(1.0 - errorProbability), 0)
        let sqrtP = Complex<Double>(sqrt(errorProbability), 0)

        let k0: [[Complex<Double>]] = [
            [sqrtOneMinusP, .zero],
            [.zero, sqrtOneMinusP],
        ]

        let k1: [[Complex<Double>]] = [
            [sqrtP, .zero],
            [.zero, -sqrtP],
        ]

        krausOperators = [k0, k1]
    }
}

// MARK: - Bit-Phase Flip Channel

/// Bit-phase flip channel: ρ -> (1-p)ρ + pYρY.
///
/// Applies Pauli Y error (combined bit and phase flip) with probability p. Y = iXZ, so this
/// simultaneously flips the bit and the phase. Less common than X or Z errors alone but
/// important for complete depolarizing noise decomposition.
///
/// The Kraus operators are K₀ = √(1-p) I and K₁ = √p Y.
///
/// **Example:**
/// ```swift
/// let channel = BitPhaseFlipChannel(errorProbability: 0.01)
/// let noisyState = channel.apply(to: dm, qubit: 0)
/// ```
@frozen
public struct BitPhaseFlipChannel: NoiseChannel {
    /// Bit-phase flip probability p ∈ [0, 1].
    public let errorProbability: Double

    /// Kraus operators for bit-phase flip channel.
    public let krausOperators: [[[Complex<Double>]]]

    /// Create bit-phase flip channel with given error probability.
    ///
    /// - Parameter errorProbability: Probability of Y error
    /// - Precondition: 0 ≤ errorProbability ≤ 1
    public init(errorProbability: Double) {
        ValidationUtilities.validateErrorProbability(errorProbability, name: "Bit-phase flip channel")

        self.errorProbability = errorProbability

        let sqrtOneMinusP = Complex<Double>(sqrt(1.0 - errorProbability), 0)
        let sqrtP = Complex<Double>(sqrt(errorProbability), 0)

        let k0: [[Complex<Double>]] = [
            [sqrtOneMinusP, .zero],
            [.zero, sqrtOneMinusP],
        ]

        let k1: [[Complex<Double>]] = [
            [.zero, Complex(0, -1) * sqrtP],
            [Complex(0, 1) * sqrtP, .zero],
        ]

        krausOperators = [k0, k1]
    }
}

// MARK: - Generalized Amplitude Damping

/// Generalized amplitude damping for finite-temperature environments.
///
/// Unlike standard amplitude damping which only allows |1⟩->|0⟩ decay, this models thermal
/// equilibrium where both |0⟩->|1⟩ (excitation) and |1⟩->|0⟩ (relaxation) can occur.
/// The steady state is a thermal distribution determined by temperature.
///
/// The damping rate γ relates to T₁, while thermal occupation n ranges from 0 (zero temperature) to
/// 0.5 (infinite temperature). Uses four Kraus operators modeling both excitation and relaxation.
///
/// **Example:**
/// ```swift
/// let channel = GeneralizedAmplitudeDampingChannel(gamma: 0.01, thermalPopulation: 0.1)
/// let noisyState = channel.apply(to: dm, qubit: 0)
/// ```
@frozen
public struct GeneralizedAmplitudeDampingChannel: NoiseChannel {
    /// Damping parameter γ ∈ [0, 1].
    public let gamma: Double

    /// Thermal occupation n ∈ [0, 0.5]. n = 0 is zero temperature, n = 0.5 is infinite temperature.
    public let thermalPopulation: Double

    /// Kraus operators for generalized amplitude damping.
    public let krausOperators: [[[Complex<Double>]]]

    /// Create generalized amplitude damping channel.
    ///
    /// - Parameters:
    ///   - gamma: Damping probability
    ///   - thermalPopulation: Thermal occupation (0 to 0.5)
    /// - Precondition: 0 ≤ gamma ≤ 1, 0 ≤ thermalPopulation ≤ 0.5
    public init(gamma: Double, thermalPopulation: Double) {
        ValidationUtilities.validateDampingParameter(gamma, name: "Generalized amplitude damping")
        ValidationUtilities.validateThermalPopulation(thermalPopulation, name: "Generalized amplitude damping")

        self.gamma = gamma
        self.thermalPopulation = thermalPopulation

        let p = 1.0 - thermalPopulation
        let sqrtP = Complex<Double>(sqrt(p), 0)
        let sqrtOneMinusP = Complex<Double>(sqrt(1.0 - p), 0)
        let sqrtGamma = Complex<Double>(sqrt(gamma), 0)
        let sqrtOneMinusGamma = Complex<Double>(sqrt(1.0 - gamma), 0)

        let k0: [[Complex<Double>]] = [
            [sqrtP, .zero],
            [.zero, sqrtP * sqrtOneMinusGamma],
        ]

        let k1: [[Complex<Double>]] = [
            [.zero, sqrtP * sqrtGamma],
            [.zero, .zero],
        ]

        let k2: [[Complex<Double>]] = [
            [sqrtOneMinusP * sqrtOneMinusGamma, .zero],
            [.zero, sqrtOneMinusP],
        ]

        let k3: [[Complex<Double>]] = [
            [.zero, .zero],
            [sqrtOneMinusP * sqrtGamma, .zero],
        ]

        krausOperators = [k0, k1, k2, k3]
    }
}

// MARK: - Custom Kraus Channel

/// Custom noise channel defined by user-provided Kraus operators.
///
/// Allows defining arbitrary single-qubit noise channels via explicit Kraus operator matrices.
/// Validates the completeness relation Σᵢ Kᵢ†Kᵢ = I to ensure the channel is valid (CPTP).
///
/// **Example:**
/// ```swift
/// let customKraus: [[[Complex<Double>]]] = [
///     [[Complex(0.9, 0), .zero], [.zero, Complex(0.9, 0)]],
///     [[Complex(0.1, 0), .zero], [.zero, Complex(-0.1, 0)]]
/// ]
/// let channel = CustomKrausChannel(krausOperators: customKraus)
/// ```
@frozen
public struct CustomKrausChannel: NoiseChannel {
    /// User-provided Kraus operators.
    public let krausOperators: [[[Complex<Double>]]]

    /// Create custom channel from Kraus operators.
    ///
    /// - Parameter krausOperators: Array of 2*2 Kraus operator matrices
    /// - Precondition: All operators must be 2*2 and satisfy completeness relation
    public init(krausOperators: [[[Complex<Double>]]]) {
        ValidationUtilities.validateNonEmpty(krausOperators, name: "Kraus operators")

        for k in krausOperators {
            ValidationUtilities.validate2x2Matrix(k)
        }

        ValidationUtilities.validateKrausCompleteness(krausOperators)

        self.krausOperators = krausOperators
    }
}

// MARK: - Two-Qubit Depolarizing Channel

/// Two-qubit depolarizing channel for correlated noise on gate pairs.
///
/// Models noise on two-qubit gates (CNOT, CZ, etc.) where the error rate is typically
/// 10x higher than single-qubit gates. Applies correlated Pauli errors to both qubits.
///
/// Uses 16 Kraus operators corresponding to all Pauli string combinations {I,X,Y,Z}⊗{I,X,Y,Z}.
///
/// **Example:**
/// ```swift
/// let channel = TwoQubitDepolarizingChannel(errorProbability: 0.01)
/// let noisyState = channel.apply(to: dm, qubits: [0, 1])
/// ```
@frozen
public struct TwoQubitDepolarizingChannel: Sendable {
    /// Error probability p ∈ [0, 1].
    public let errorProbability: Double

    /// Kraus operators for two-qubit depolarizing channel.
    public let krausOperators: [[[Complex<Double>]]]

    /// Create two-qubit depolarizing channel.
    ///
    /// - Parameter errorProbability: Probability of correlated Pauli error
    /// - Precondition: 0 ≤ errorProbability ≤ 1
    public init(errorProbability: Double) {
        ValidationUtilities.validateErrorProbability(errorProbability, name: "Two-qubit depolarizing")

        self.errorProbability = errorProbability

        let pauliI: [[Complex<Double>]] = [[.one, .zero], [.zero, .one]]
        let pauliX: [[Complex<Double>]] = [[.zero, .one], [.one, .zero]]
        let pauliY: [[Complex<Double>]] = [[.zero, Complex(0, -1)], [Complex(0, 1), .zero]]
        let pauliZ: [[Complex<Double>]] = [[.one, .zero], [.zero, Complex(-1, 0)]]

        let paulis = [pauliI, pauliX, pauliY, pauliZ]

        var operators: [[[Complex<Double>]]] = []
        operators.reserveCapacity(16)

        let sqrtOneMinusP = sqrt(1.0 - errorProbability)
        let sqrtPOver15 = sqrt(errorProbability / 15.0)

        for (i, p1) in paulis.enumerated() {
            for (j, p2) in paulis.enumerated() {
                let coeff = (i == 0 && j == 0) ? sqrtOneMinusP : sqrtPOver15
                let tensor = Self.tensorProduct(p1, p2)
                let scaled = Self.scaleMatrix(tensor, by: coeff)
                operators.append(scaled)
            }
        }

        krausOperators = operators
    }

    /// Compute tensor product of two 2*2 matrices.
    private static func tensorProduct(
        _ a: [[Complex<Double>]],
        _ b: [[Complex<Double>]],
    ) -> [[Complex<Double>]] {
        var result = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: 4),
            count: 4,
        )

        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                for k in 0 ..< 2 {
                    for l in 0 ..< 2 {
                        result[2 * i + k][2 * j + l] = a[i][j] * b[k][l]
                    }
                }
            }
        }

        return result
    }

    /// Scale matrix by scalar coefficient.
    private static func scaleMatrix(_ m: [[Complex<Double>]], by c: Double) -> [[Complex<Double>]] {
        let coeff = Complex<Double>(c, 0)
        return m.map { row in row.map { $0 * coeff } }
    }

    /// Apply two-qubit noise channel.
    ///
    /// - Parameters:
    ///   - matrix: Input density matrix
    ///   - qubits: Two target qubit indices
    /// - Returns: Noisy density matrix
    /// - Precondition: qubits.count == 2
    /// - Precondition: All indices in [0, matrix.qubits)
    /// - Precondition: qubits[0] != qubits[1]
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public func apply(to matrix: DensityMatrix, qubits: [Int]) -> DensityMatrix {
        ValidationUtilities.validateArrayCount(qubits, expected: 2, name: "Two-qubit channel qubits")
        ValidationUtilities.validateOperationQubits(qubits, numQubits: matrix.qubits)
        ValidationUtilities.validateUniqueQubits(qubits)

        let dim = matrix.dimension
        let size = dim * dim
        var resultElements = [Complex<Double>](repeating: .zero, count: size)

        for kraus in krausOperators {
            let transformed = applyTwoQubitKraus(kraus: kraus, matrix: matrix, qubits: qubits)
            for i in 0 ..< size {
                resultElements[i] = resultElements[i] + transformed[i]
            }
        }

        return DensityMatrix(qubits: matrix.qubits, elements: resultElements)
    }

    /// Apply single two-qubit Kraus operator.
    @_optimize(speed)
    private func applyTwoQubitKraus(
        kraus: [[Complex<Double>]],
        matrix: DensityMatrix,
        qubits: [Int],
    ) -> [Complex<Double>] {
        let q0 = qubits[0]
        let q1 = qubits[1]
        let dim = matrix.dimension
        let size = dim * dim
        let mask0 = 1 << q0
        let mask1 = 1 << q1

        var newElements = [Complex<Double>](repeating: .zero, count: size)

        for row in 0 ..< dim {
            for col in 0 ..< dim {
                var sum = Complex<Double>.zero

                for a in 0 ..< 4 {
                    let aRow = (row & ~mask0 & ~mask1) |
                        ((a & 1) << q0) |
                        (((a >> 1) & 1) << q1)

                    for b in 0 ..< 4 {
                        let bCol = (col & ~mask0 & ~mask1) |
                            ((b & 1) << q0) |
                            (((b >> 1) & 1) << q1)

                        let rowIdx = ((row >> q0) & 1) | (((row >> q1) & 1) << 1)
                        let colIdx = ((col >> q0) & 1) | (((col >> q1) & 1) << 1)

                        let kElement = kraus[rowIdx][a]
                        let kDaggerElement = kraus[b][colIdx].conjugate
                        let rhoElement = matrix.element(row: aRow, col: bCol)

                        sum = sum + kElement * rhoElement * kDaggerElement
                    }
                }

                newElements[row * dim + col] = sum
            }
        }

        return newElements
    }
}
