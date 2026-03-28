// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Test suite for QuantumState initialization.
/// Validates state vector creation with different qubit counts,
/// demonstrating the scalable architecture that handles 1-24+ qubits identically.
@Suite("Quantum State Initialization")
struct QuantumStateInitializationTests {
    @Test("Initialize 1-qubit state to |0⟩")
    func initializeOneQubit() {
        let state = QuantumState(qubits: 1)
        #expect(state.qubits == 1, "Should have 1 qubit")
        #expect(state.stateSpaceSize == 2, "1 qubit should have state space size 2")
        #expect(state.amplitude(of: 0) == Complex.one, "Amplitude of |0> should be 1")
        #expect(state.amplitude(of: 1) == Complex.zero, "Amplitude of |1> should be 0")
    }

    @Test("Initialize 2-qubit state to |00⟩")
    func initializeTwoQubits() {
        let state = QuantumState(qubits: 2)
        #expect(state.qubits == 2, "Should have 2 qubits")
        #expect(state.stateSpaceSize == 4, "2 qubits should have state space size 4")
        #expect(state.amplitude(of: 0) == Complex.one, "Amplitude of |00> should be 1")
        #expect(state.amplitude(of: 1) == Complex.zero, "Amplitude of |01> should be 0")
        #expect(state.amplitude(of: 2) == Complex.zero, "Amplitude of |10> should be 0")
        #expect(state.amplitude(of: 3) == Complex.zero, "Amplitude of |11> should be 0")
    }

    @Test("Initialize 8-qubit state (demo default)")
    func initializeEightQubits() {
        let state = QuantumState(qubits: 8)
        #expect(state.qubits == 8, "Should have 8 qubits")
        #expect(state.stateSpaceSize == 256, "8 qubits should have state space size 256")
    }

    @Test("Initialize 16-qubit state (interactive mode)")
    func initializeSixteenQubits() {
        let state = QuantumState(qubits: 16)
        #expect(state.qubits == 16, "Should have 16 qubits")
        #expect(state.stateSpaceSize == 65536, "16 qubits should have state space size 65536")
    }

    @Test("Initialize 24-qubit state (execution mode ceiling)")
    func initializeTwentyFourQubits() {
        let state = QuantumState(qubits: 24)
        #expect(state.qubits == 24, "Should have 24 qubits")
        #expect(state.stateSpaceSize == 16_777_216, "24 qubits should have state space size 2^24")
    }

    @Test("Single-qubit convenience initializer for |0⟩")
    func singleQubitZero() {
        let state = QuantumState(qubit: 0)
        #expect(state.qubits == 1, "Should have 1 qubit")
        #expect(state.amplitude(of: 0) == Complex.one, "|0> amplitude should be 1")
        #expect(state.amplitude(of: 1) == Complex.zero, "|1> amplitude should be 0")
    }

    @Test("Single-qubit convenience initializer for |1⟩")
    func singleQubitOne() {
        let state = QuantumState(qubit: 1)
        #expect(state.qubits == 1, "Should have 1 qubit")
        #expect(state.amplitude(of: 0) == Complex.zero, "|0> amplitude should be 0")
        #expect(state.amplitude(of: 1) == Complex.one, "|1> amplitude should be 1")
    }

    @Test("Custom amplitudes initialization")
    func customAmplitudes() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0)]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)

        #expect(abs(state.amplitude(of: 0).real - invSqrt2) < 1e-10, "|0> real part should be 1/sqrt(2)")
        #expect(abs(state.amplitude(of: 1).real - invSqrt2) < 1e-10, "|1> real part should be 1/sqrt(2)")
    }
}

/// Test suite for quantum state normalization.
/// Validates that Σ|cᵢ|² = 1.0 (probability conservation)
/// and automatic renormalization for user convenience.
@Suite("Quantum State Normalization")
struct QuantumStateNormalizationTests {
    @Test("New state is normalized")
    func newStateNormalized() {
        let state = QuantumState(qubits: 3)
        #expect(state.isNormalized(), "Freshly created state should be normalized")
    }

    @Test("Probability sum equals 1.0")
    func probabilitySumOne() {
        let state = QuantumState(qubits: 2)
        let probs = state.probabilities()
        let sum = probs.reduce(0.0, +)
        #expect(abs(sum - 1.0) < 1e-10, "Probability sum should be 1.0, got \(sum)")
    }

    @Test("Normalization after modification")
    func normalizeAfterModification() {
        var state = QuantumState(qubits: 1)
        state.setAmplitude(0, to: Complex(2.0, 0.0))
        state.setAmplitude(1, to: Complex(2.0, 0.0))

        #expect(!state.isNormalized(), "Modified state should not be normalized")
        state.normalize()
        #expect(state.isNormalized(), "State should be normalized after normalize()")
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
        #expect(abs(state.probability(of: 0) - 1.0) < 1e-10, "P(|0>) should be 1.0")
        #expect(abs(state.probability(of: 1)) < 1e-10, "P(|1>) should be 0.0")
    }

    @Test("Probability of |1⟩ state is 1.0")
    func probabilityOneState() {
        let state = QuantumState(qubit: 1)
        #expect(abs(state.probability(of: 0)) < 1e-10, "P(|0>) should be 0.0")
        #expect(abs(state.probability(of: 1) - 1.0) < 1e-10, "P(|1>) should be 1.0")
    }

    @Test("Equal superposition probabilities")
    func equalSuperposition() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0)]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)

        #expect(abs(state.probability(of: 0) - 0.5) < 1e-10, "P(|0>) should be 0.5 in equal superposition")
        #expect(abs(state.probability(of: 1) - 0.5) < 1e-10, "P(|1>) should be 0.5 in equal superposition")
    }

    @Test("Single-qubit marginal probabilities")
    func singleQubitMarginals() {
        let state = QuantumState(qubit: 0)
        let (p0, p1) = state.probabilities(for: 0)

        #expect(abs(p0 - 1.0) < 1e-10, "Marginal P(0) should be 1.0 for |0>")
        #expect(abs(p1) < 1e-10, "Marginal P(1) should be 0.0 for |0>")
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
        let state = QuantumState(qubits: 2, amplitudes: amplitudes)
        #expect(state.isNormalized(), "Bell state should be normalized")
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
        let state = QuantumState(qubits: 2, amplitudes: amplitudes)

        #expect(abs(state.probability(of: 0) - 0.5) < 1e-10, "P(|00>) should be 0.5")
        #expect(abs(state.probability(of: 1)) < 1e-10, "P(|01>) should be 0.0")
        #expect(abs(state.probability(of: 2)) < 1e-10, "P(|10>) should be 0.0")
        #expect(abs(state.probability(of: 3) - 0.5) < 1e-10, "P(|11>) should be 0.5")
    }
}

/// Test suite for scalability across qubit counts.
/// Demonstrates generic architecture handles 1-24+ qubits identically,
/// proving serious quantum simulation capability beyond toy implementations.
@Suite("Quantum State Scalability")
struct QuantumStateScalabilityTests {
    @Test("State space size scales exponentially")
    func stateSpaceSizeScaling() {
        let state1 = QuantumState(qubits: 1)
        let state2 = QuantumState(qubits: 2)
        let state3 = QuantumState(qubits: 3)

        #expect(state1.stateSpaceSize == 2, "1-qubit state space should be 2")
        #expect(state2.stateSpaceSize == 4, "2-qubit state space should be 4")
        #expect(state3.stateSpaceSize == 8, "3-qubit state space should be 8")
    }

    @Test("Architecture handles varying qubit counts identically")
    func varyingQubitCounts() {
        for n in [1, 4, 8, 12, 16, 20, 24] {
            let state = QuantumState(qubits: n)
            #expect(state.qubits == n, "Should have \(n) qubits")
            #expect(state.stateSpaceSize == (1 << n), "State space should be 2^\(n)")
            #expect(state.isNormalized(), "Ground state with \(n) qubits should be normalized")
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
        let state1 = QuantumState(qubits: 2)
        let state2 = QuantumState(qubits: 2)
        #expect(state1 == state2, "Two ground states with same qubits should be equal")
    }

    @Test("Different qubit counts are not equal")
    func differentQubitCountsNotEqual() {
        let state1 = QuantumState(qubits: 1)
        let state2 = QuantumState(qubits: 2)
        #expect(state1 != state2, "States with different qubit counts should not be equal")
    }

    @Test("Different amplitudes are not equal")
    func differentAmplitudesNotEqual() {
        let state1 = QuantumState(qubit: 0)
        let state2 = QuantumState(qubit: 1)
        #expect(state1 != state2, "|0> and |1> should not be equal")
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
        #expect(desc.contains("1 qubit"), "Description should mention qubit count")
        #expect(desc.contains("|0⟩"), "Description should show |0> basis state")
    }

    @Test("Superposition state description")
    func superpositionDescription() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0)]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)
        let desc = state.description
        #expect(desc.contains("1 qubit"), "Description should mention qubit count")
    }

    @Test("Near-zero state description")
    func nearZeroStateDescription() {
        let amplitudes = [
            Complex(1e-10, 0.0),
            Complex(1e-10, 0.0),
            Complex(1e-10, 0.0),
            Complex(1e-10, 0.0),
        ]
        let state = QuantumState(qubits: 2, amplitudes: amplitudes)
        let desc = state.description
        #expect(desc.contains("near-zero"), "Near-zero state should show near-zero label")
    }
}

/// Test suite for large state vectorized operations.
/// Validates Accelerate optimizations for 64+ amplitude states,
/// ensuring performance scaling for production quantum simulations.
@Suite("Large State Vectorized Operations")
struct LargeStateVectorizedTests {
    @Test("Normalize large state uses vectorized path")
    func normalizeLargeState() {
        let qubits = 7
        let amplitudes = [Complex<Double>](repeating: Complex(1.0, 0.0), count: 128)
        var state = QuantumState(qubits: qubits, amplitudes: amplitudes)
        #expect(state.isNormalized(), "Auto-normalized state should be normalized")

        for i in 0 ..< 128 {
            state.setAmplitude(i, to: Complex(2.0, 0.0))
        }
        #expect(!state.isNormalized(), "Modified state should not be normalized")

        state.normalize()
        #expect(state.isNormalized(), "State should be normalized after normalize()")
    }

    @Test("Probabilities for large state uses vectorized path")
    func probabilitiesLargeState() {
        let qubits = 7
        let invSqrtN = 1.0 / sqrt(128.0)
        let amplitudes = [Complex<Double>](repeating: Complex(invSqrtN, 0.0), count: 128)
        let state = QuantumState(qubits: qubits, amplitudes: amplitudes)

        let probs = state.probabilities()
        #expect(probs.count == 128, "Should have 128 probabilities for 7-qubit state")

        let sum = probs.reduce(0.0, +)
        #expect(abs(sum - 1.0) < 1e-10, "Probability sum should be 1.0, got \(sum)")
    }

    @Test("isNormalized for large state uses vectorized path")
    func isNormalizedLargeState() {
        let qubits = 8
        let state = QuantumState(qubits: qubits)
        #expect(state.isNormalized(), "Ground state should be normalized")
    }

    @Test("Large state auto-normalizes on init")
    func largeStateAutoNormalizes() {
        let qubits = 7
        let amplitudes = [Complex<Double>](repeating: Complex(1.0, 0.0), count: 128)
        let state = QuantumState(qubits: qubits, amplitudes: amplitudes)

        #expect(state.isNormalized(), "Auto-normalized large state should be normalized")
    }
}

/// Test suite for amplitude mutation.
/// Validates setAmplitude operation, state modification behavior,
/// and correct preservation of unmodified amplitudes.
@Suite("Amplitude Mutation")
struct AmplitudeMutationTests {
    @Test("setAmplitude modifies correct amplitude")
    func setAmplitudeCorrect() {
        var state = QuantumState(qubits: 2)
        let newAmp = Complex(0.5, 0.3)

        state.setAmplitude(2, to: newAmp)

        #expect(state.amplitude(of: 2) == newAmp, "Modified amplitude should match new value")
        #expect(state.amplitude(of: 0) == Complex.one, "Unmodified amplitude 0 should remain 1")
        #expect(state.amplitude(of: 1) == Complex.zero, "Unmodified amplitude 1 should remain 0")
        #expect(state.amplitude(of: 3) == Complex.zero, "Unmodified amplitude 3 should remain 0")
    }

    @Test("setAmplitude can create superposition")
    func setAmplitudeCreatesSuperposition() {
        var state = QuantumState(qubits: 1)
        let invSqrt2 = 1.0 / sqrt(2.0)

        state.setAmplitude(0, to: Complex(invSqrt2, 0.0))
        state.setAmplitude(1, to: Complex(invSqrt2, 0.0))

        #expect(state.isNormalized(), "Equal superposition should be normalized")
        #expect(abs(state.probability(of: 0) - 0.5) < 1e-10, "P(|0>) should be 0.5")
        #expect(abs(state.probability(of: 1) - 0.5) < 1e-10, "P(|1>) should be 0.5")
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
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)

        #expect(state.isNormalized(), "Auto-normalized state should be normalized")
    }

    @Test("Already normalized amplitudes stay unchanged")
    func alreadyNormalizedStaysUnchanged() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0)]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)

        #expect(abs(state.amplitude(of: 0).real - invSqrt2) < 1e-10, "Normalized amplitude should be unchanged")
        #expect(abs(state.amplitude(of: 1).real - invSqrt2) < 1e-10, "Normalized amplitude should be unchanged")
    }

    @Test("Complex amplitudes auto-normalize correctly")
    func complexAmplitudesAutoNormalize() {
        let amplitudes = [
            Complex(1.0, 1.0),
            Complex(1.0, -1.0),
        ]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)

        #expect(state.isNormalized(), "Complex auto-normalized state should be normalized")
    }
}
