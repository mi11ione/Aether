// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for QuantumState initialization.
/// Validates state vector creation with different qubit counts,
/// demonstrating the scalable architecture that handles 1-24+ qubits identically.
@Suite("Quantum State Initialization")
struct QuantumStateInitializationTests {
    @Test("Initialize 1-qubit state to |0⟩")
    func initializeOneQubit() {
        let state = QuantumState(numQubits: 1)
        #expect(state.numQubits == 1)
        #expect(state.stateSpaceSize == 2)
        #expect(state.amplitude(of: 0) == Complex.one)
        #expect(state.amplitude(of: 1) == Complex.zero)
    }

    @Test("Initialize 2-qubit state to |00⟩")
    func initializeTwoQubits() {
        let state = QuantumState(numQubits: 2)
        #expect(state.numQubits == 2)
        #expect(state.stateSpaceSize == 4)
        #expect(state.amplitude(of: 0) == Complex.one)
        #expect(state.amplitude(of: 1) == Complex.zero)
        #expect(state.amplitude(of: 2) == Complex.zero)
        #expect(state.amplitude(of: 3) == Complex.zero)
    }

    @Test("Initialize 8-qubit state (demo default)")
    func initializeEightQubits() {
        let state = QuantumState(numQubits: 8)
        #expect(state.numQubits == 8)
        #expect(state.stateSpaceSize == 256)
    }

    @Test("Initialize 16-qubit state (interactive mode)")
    func initializeSixteenQubits() {
        let state = QuantumState(numQubits: 16)
        #expect(state.numQubits == 16)
        #expect(state.stateSpaceSize == 65536)
    }

    @Test("Initialize 24-qubit state (execution mode ceiling)")
    func initializeTwentyFourQubits() {
        let state = QuantumState(numQubits: 24)
        #expect(state.numQubits == 24)
        #expect(state.stateSpaceSize == 16_777_216)
    }

    @Test("Single-qubit convenience initializer for |0⟩")
    func singleQubitZero() {
        let state = QuantumState(qubit: 0)
        #expect(state.numQubits == 1)
        #expect(state.amplitude(of: 0) == Complex.one)
        #expect(state.amplitude(of: 1) == Complex.zero)
    }

    @Test("Single-qubit convenience initializer for |1⟩")
    func singleQubitOne() {
        let state = QuantumState(qubit: 1)
        #expect(state.numQubits == 1)
        #expect(state.amplitude(of: 0) == Complex.zero)
        #expect(state.amplitude(of: 1) == Complex.one)
    }

    @Test("Custom amplitudes initialization")
    func customAmplitudes() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0)]
        let state = QuantumState(numQubits: 1, amplitudes: amplitudes)

        #expect(abs(state.amplitude(of: 0).real - invSqrt2) < 1e-10)
        #expect(abs(state.amplitude(of: 1).real - invSqrt2) < 1e-10)
    }
}

/// Test suite for quantum state normalization.
/// Validates that Σ|cᵢ|² = 1.0 (probability conservation)
/// and automatic renormalization for user convenience.
@Suite("Quantum State Normalization")
struct QuantumStateNormalizationTests {
    @Test("New state is normalized")
    func newStateNormalized() {
        let state = QuantumState(numQubits: 3)
        #expect(state.isNormalized())
    }

    @Test("Probability sum equals 1.0")
    func probabilitySumOne() {
        let state = QuantumState(numQubits: 2)
        let probs = state.probabilities()
        let sum = probs.reduce(0.0, +)
        #expect(abs(sum - 1.0) < 1e-10)
    }

    @Test("Normalization after modification")
    func normalizeAfterModification() {
        var state = QuantumState(numQubits: 1)
        state.setAmplitude(0, to: Complex(2.0, 0.0))
        state.setAmplitude(1, to: Complex(2.0, 0.0))

        #expect(!state.isNormalized())
        state.normalize()
        #expect(state.isNormalized())
    }
}

/// Test suite for probability calculations.
/// Validates Born rule implementation: P(i) = |cᵢ|²
/// and marginal probability calculations for measurement simulation.
@Suite("Probability Calculations")
struct QuantumStateProbabilityTests {
    @Test("Probability of |0⟩ state is 1.0")
    func probabilityZeroState() {
        let state = QuantumState(qubit: 0)
        #expect(abs(state.probability(of: 0) - 1.0) < 1e-10)
        #expect(abs(state.probability(of: 1)) < 1e-10)
    }

    @Test("Probability of |1⟩ state is 1.0")
    func probabilityOneState() {
        let state = QuantumState(qubit: 1)
        #expect(abs(state.probability(of: 0)) < 1e-10)
        #expect(abs(state.probability(of: 1) - 1.0) < 1e-10)
    }

    @Test("Equal superposition probabilities")
    func equalSuperposition() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0)]
        let state = QuantumState(numQubits: 1, amplitudes: amplitudes)

        #expect(abs(state.probability(of: 0) - 0.5) < 1e-10)
        #expect(abs(state.probability(of: 1) - 0.5) < 1e-10)
    }

    @Test("Single-qubit marginal probabilities")
    func singleQubitMarginals() {
        let state = QuantumState(qubit: 0)
        let (p0, p1) = state.probabilities(for: 0)

        #expect(abs(p0 - 1.0) < 1e-10)
        #expect(abs(p1) < 1e-10)
    }
}

/// Test suite for state validation.
/// Ensures quantum states maintain mathematical invariants
/// throughout operations and modifications.
@Suite("State Validation")
struct QuantumStateValidationTests {
    @Test("Valid state passes validation")
    func validState() {
        let state = QuantumState(numQubits: 2)
        #expect(state.validate())
    }

    @Test("State remains valid after normalization")
    func validAfterNormalization() {
        var state = QuantumState(numQubits: 1)
        state.setAmplitude(0, to: Complex(1.0, 1.0))
        state.setAmplitude(1, to: Complex(1.0, 1.0))
        state.normalize()
        #expect(state.validate())
    }
}

/// Test suite for Bell state (entangled) configurations.
/// Validates maximally entangled two-qubit states
/// fundamental to quantum algorithms and teleportation.
@Suite("Quantum Bell State")
struct QuantumStateBellStateTests {
    @Test("Bell state (|00⟩ + |11⟩)/√2 is normalized")
    func bellStateNormalized() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [
            Complex(invSqrt2, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
            Complex(invSqrt2, 0.0),
        ]
        let state = QuantumState(numQubits: 2, amplitudes: amplitudes)
        #expect(state.isNormalized())
    }

    @Test("Bell state probabilities")
    func bellStateProbabilities() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [
            Complex(invSqrt2, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
            Complex(invSqrt2, 0.0),
        ]
        let state = QuantumState(numQubits: 2, amplitudes: amplitudes)

        #expect(abs(state.probability(of: 0) - 0.5) < 1e-10)
        #expect(abs(state.probability(of: 1)) < 1e-10)
        #expect(abs(state.probability(of: 2)) < 1e-10)
        #expect(abs(state.probability(of: 3) - 0.5) < 1e-10)
    }
}

/// Test suite for scalability across qubit counts.
/// Demonstrates generic architecture handles 1-24+ qubits identically,
/// proving serious quantum simulation capability beyond toy implementations.
@Suite("Quantum State Scalability")
struct QuantumStateScalabilityTests {
    @Test("State space size scales exponentially")
    func stateSpaceSizeScaling() {
        let state1 = QuantumState(numQubits: 1)
        let state2 = QuantumState(numQubits: 2)
        let state3 = QuantumState(numQubits: 3)

        #expect(state1.stateSpaceSize == 2)
        #expect(state2.stateSpaceSize == 4)
        #expect(state3.stateSpaceSize == 8)
    }

    @Test("Architecture handles varying qubit counts identically")
    func varyingQubitCounts() {
        for n in [1, 4, 8, 12, 16, 20, 24] {
            let state = QuantumState(numQubits: n)
            #expect(state.numQubits == n)
            #expect(state.stateSpaceSize == (1 << n))
            #expect(state.isNormalized())
        }
    }
}

/// Test suite for quantum state equality.
/// Validates Equatable conformance with tolerance-based comparison
/// for numerical stability in quantum computations.
@Suite("State Equality")
struct QuantumStateEqualityTests {
    @Test("Identical states are equal")
    func identicalStatesEqual() {
        let state1 = QuantumState(numQubits: 2)
        let state2 = QuantumState(numQubits: 2)
        #expect(state1 == state2)
    }

    @Test("Different qubit counts are not equal")
    func differentQubitCountsNotEqual() {
        let state1 = QuantumState(numQubits: 1)
        let state2 = QuantumState(numQubits: 2)
        #expect(state1 != state2)
    }

    @Test("Different amplitudes are not equal")
    func differentAmplitudesNotEqual() {
        let state1 = QuantumState(qubit: 0)
        let state2 = QuantumState(qubit: 1)
        #expect(state1 != state2)
    }
}

/// Test suite for string representation.
/// Validates CustomStringConvertible implementation for debugging
/// and educational quantum state visualization.
@Suite("Quantum State String Representation")
struct QuantumStateDescriptionTests {
    @Test("Single qubit |0⟩ description")
    func singleQubitZeroDescription() {
        let state = QuantumState(qubit: 0)
        let desc = state.description
        #expect(desc.contains("1 qubit"))
        #expect(desc.contains("|0⟩"))
    }

    @Test("Superposition state description")
    func superpositionDescription() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0)]
        let state = QuantumState(numQubits: 1, amplitudes: amplitudes)
        let desc = state.description
        #expect(desc.contains("1 qubit"))
    }

    @Test("Near-zero state description")
    func nearZeroStateDescription() {
        let amplitudes = [
            Complex(1e-10, 0.0),
            Complex(1e-10, 0.0),
            Complex(1e-10, 0.0),
            Complex(1e-10, 0.0),
        ]
        let state = QuantumState(numQubits: 2, amplitudes: amplitudes)
        let desc = state.description
        #expect(desc.contains("near-zero"))
    }
}

/// Test suite for large state vectorized operations.
/// Validates Accelerate optimizations for 64+ qubit states,
/// ensuring performance scaling for production quantum simulations.
@Suite("Large State Vectorized Operations")
struct LargeStateVectorizedTests {
    @Test("Normalize large state uses vectorized path")
    func normalizeLargeState() {
        let numQubits = 7
        let amplitudes = [Complex<Double>](repeating: Complex(1.0, 0.0), count: 128)
        var state = QuantumState(numQubits: numQubits, amplitudes: amplitudes)
        #expect(state.isNormalized())

        for i in 0 ..< 128 {
            state.setAmplitude(i, to: Complex(2.0, 0.0))
        }
        #expect(!state.isNormalized())

        state.normalize()
        #expect(state.isNormalized())
    }

    @Test("Probabilities for large state uses vectorized path")
    func probabilitiesLargeState() {
        let numQubits = 7
        let invSqrtN = 1.0 / sqrt(128.0)
        let amplitudes = [Complex<Double>](repeating: Complex(invSqrtN, 0.0), count: 128)
        let state = QuantumState(numQubits: numQubits, amplitudes: amplitudes)

        let probs = state.probabilities()
        #expect(probs.count == 128)

        let sum = probs.reduce(0.0, +)
        #expect(abs(sum - 1.0) < 1e-10)
    }

    @Test("isNormalized for large state uses vectorized path")
    func isNormalizedLargeState() {
        let numQubits = 8
        let state = QuantumState(numQubits: numQubits)
        #expect(state.isNormalized())
    }

    @Test("Large state auto-normalizes on init")
    func largeStateAutoNormalizes() {
        let numQubits = 7
        let amplitudes = [Complex<Double>](repeating: Complex(1.0, 0.0), count: 128)
        let state = QuantumState(numQubits: numQubits, amplitudes: amplitudes)

        #expect(state.isNormalized())
    }
}

/// Test suite for quantum state validation.
/// Validates state invariants: amplitude count, normalization,
/// and absence of NaN/Inf values for robust simulation.
@Suite("Quantum State Validation")
struct StateValidationTests {
    @Test("Valid state passes validation")
    func validStatePassesValidation() {
        let state = QuantumState(numQubits: 2)
        #expect(state.validate())
    }

    @Test("Normalized state passes validation")
    func normalizedStatePassesValidation() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0)]
        let state = QuantumState(numQubits: 1, amplitudes: amplitudes)
        #expect(state.validate())
    }

    @Test("Unnormalized state fails validation")
    func unnormalizedStateFails() {
        var state = QuantumState(numQubits: 1)
        state.setAmplitude(0, to: Complex(2.0, 0.0))
        state.setAmplitude(1, to: Complex(2.0, 0.0))
        #expect(!state.validate())
    }

    @Test("State with NaN fails validation")
    func stateWithNaNFails() {
        var state = QuantumState(numQubits: 1)
        state.setAmplitude(0, to: Complex(Double.nan, 0.0))
        #expect(!state.validate())
    }

    @Test("State with Inf fails validation")
    func stateWithInfFails() {
        var state = QuantumState(numQubits: 1)
        state.setAmplitude(1, to: Complex(Double.infinity, 0.0))
        #expect(!state.validate())
    }
}

/// Test suite for amplitude mutation.
/// Validates setAmplitude operation and state modification behavior.
@Suite("Amplitude Mutation")
struct AmplitudeMutationTests {
    @Test("setAmplitude modifies correct amplitude")
    func setAmplitudeCorrect() {
        var state = QuantumState(numQubits: 2)
        let newAmp = Complex(0.5, 0.3)

        state.setAmplitude(2, to: newAmp)

        #expect(state.amplitude(of: 2) == newAmp)
        #expect(state.amplitude(of: 0) == Complex.one)
        #expect(state.amplitude(of: 1) == Complex.zero)
        #expect(state.amplitude(of: 3) == Complex.zero)
    }

    @Test("setAmplitude can create superposition")
    func setAmplitudeCreatesSuperposition() {
        var state = QuantumState(numQubits: 1)
        let invSqrt2 = 1.0 / sqrt(2.0)

        state.setAmplitude(0, to: Complex(invSqrt2, 0.0))
        state.setAmplitude(1, to: Complex(invSqrt2, 0.0))

        #expect(state.isNormalized())
        #expect(abs(state.probability(of: 0) - 0.5) < 1e-10)
        #expect(abs(state.probability(of: 1) - 0.5) < 1e-10)
    }
}

/// Test suite for auto-normalization on initialization.
/// Validates automatic state vector normalization when constructing
/// from custom amplitudes that aren't pre-normalized.
@Suite("Auto-Normalization")
struct AutoNormalizationTests {
    @Test("Unnormalized amplitudes get auto-normalized")
    func autoNormalizeUnnormalized() {
        let amplitudes = [Complex(2.0, 0.0), Complex(2.0, 0.0)]
        let state = QuantumState(numQubits: 1, amplitudes: amplitudes)

        #expect(state.isNormalized())
    }

    @Test("Already normalized amplitudes stay unchanged")
    func alreadyNormalizedStaysUnchanged() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0)]
        let state = QuantumState(numQubits: 1, amplitudes: amplitudes)

        #expect(abs(state.amplitude(of: 0).real - invSqrt2) < 1e-10)
        #expect(abs(state.amplitude(of: 1).real - invSqrt2) < 1e-10)
    }

    @Test("Complex amplitudes auto-normalize correctly")
    func complexAmplitudesAutoNormalize() {
        let amplitudes = [
            Complex(1.0, 1.0),
            Complex(1.0, -1.0),
        ]
        let state = QuantumState(numQubits: 1, amplitudes: amplitudes)

        #expect(state.isNormalized())
    }
}
