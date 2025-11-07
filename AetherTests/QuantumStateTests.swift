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
        #expect(state.getAmplitude(ofState: 0) == Complex.one)
        #expect(state.getAmplitude(ofState: 1) == Complex.zero)
    }

    @Test("Initialize 2-qubit state to |00⟩")
    func initializeTwoQubits() {
        let state = QuantumState(numQubits: 2)
        #expect(state.numQubits == 2)
        #expect(state.stateSpaceSize == 4)
        #expect(state.getAmplitude(ofState: 0) == Complex.one)
        #expect(state.getAmplitude(ofState: 1) == Complex.zero)
        #expect(state.getAmplitude(ofState: 2) == Complex.zero)
        #expect(state.getAmplitude(ofState: 3) == Complex.zero)
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
        let state = QuantumState(singleQubit: 0)
        #expect(state.numQubits == 1)
        #expect(state.getAmplitude(ofState: 0) == Complex.one)
        #expect(state.getAmplitude(ofState: 1) == Complex.zero)
    }

    @Test("Single-qubit convenience initializer for |1⟩")
    func singleQubitOne() {
        let state = QuantumState(singleQubit: 1)
        #expect(state.numQubits == 1)
        #expect(state.getAmplitude(ofState: 0) == Complex.zero)
        #expect(state.getAmplitude(ofState: 1) == Complex.one)
    }

    @Test("Custom amplitudes initialization")
    func customAmplitudes() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0)]
        let state = QuantumState(numQubits: 1, amplitudes: amplitudes)

        #expect(abs(state.getAmplitude(ofState: 0).real - invSqrt2) < 1e-10)
        #expect(abs(state.getAmplitude(ofState: 1).real - invSqrt2) < 1e-10)
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
    func normalizeAfterModification() throws {
        var state = QuantumState(numQubits: 1)
        state.setAmplitude(ofState: 0, amplitude: Complex(2.0, 0.0))
        state.setAmplitude(ofState: 1, amplitude: Complex(2.0, 0.0))

        #expect(!state.isNormalized())
        try state.normalize()
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
        let state = QuantumState(singleQubit: 0)
        #expect(abs(state.probability(ofState: 0) - 1.0) < 1e-10)
        #expect(abs(state.probability(ofState: 1)) < 1e-10)
    }

    @Test("Probability of |1⟩ state is 1.0")
    func probabilityOneState() {
        let state = QuantumState(singleQubit: 1)
        #expect(abs(state.probability(ofState: 0)) < 1e-10)
        #expect(abs(state.probability(ofState: 1) - 1.0) < 1e-10)
    }

    @Test("Equal superposition probabilities")
    func equalSuperposition() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes = [Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0)]
        let state = QuantumState(numQubits: 1, amplitudes: amplitudes)

        #expect(abs(state.probability(ofState: 0) - 0.5) < 1e-10)
        #expect(abs(state.probability(ofState: 1) - 0.5) < 1e-10)
    }

    @Test("Single-qubit marginal probabilities")
    func singleQubitMarginals() {
        let state = QuantumState(singleQubit: 0)
        let (p0, p1) = state.singleQubitProbabilities(qubit: 0)

        #expect(abs(p0 - 1.0) < 1e-10)
        #expect(abs(p1) < 1e-10)
    }
}

/// Test suite for qubit indexing utilities.
/// Validates little-endian bit manipulation functions
/// used in gate application algorithms and state analysis.
@Suite("Qubit Indexing Utilities")
struct QuantumStateIndexingTests {
    @Test("Get bit from state index")
    func getBitFromIndex() {
        let state = QuantumState(numQubits: 3)

        #expect(state.getBit(index: 5, qubit: 0) == 1)
        #expect(state.getBit(index: 5, qubit: 1) == 0)
        #expect(state.getBit(index: 5, qubit: 2) == 1)
    }

    @Test("Set bit in state index")
    func setBitInIndex() {
        let state = QuantumState(numQubits: 3)

        let index = 0
        let newIndex = state.setBit(index: index, qubit: 1, value: 1)
        #expect(newIndex == 2)
    }

    @Test("Flip bit in state index")
    func flipBitInIndex() {
        let state = QuantumState(numQubits: 3)

        let index = 5
        let flipped = state.flipBit(index: index, qubit: 1)
        #expect(flipped == 7)
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
    func validAfterNormalization() throws {
        var state = QuantumState(numQubits: 1)
        state.setAmplitude(ofState: 0, amplitude: Complex(1.0, 1.0))
        state.setAmplitude(ofState: 1, amplitude: Complex(1.0, 1.0))
        try state.normalize()
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

        #expect(abs(state.probability(ofState: 0) - 0.5) < 1e-10)
        #expect(abs(state.probability(ofState: 1)) < 1e-10)
        #expect(abs(state.probability(ofState: 2)) < 1e-10)
        #expect(abs(state.probability(ofState: 3) - 0.5) < 1e-10)
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
        let state1 = QuantumState(singleQubit: 0)
        let state2 = QuantumState(singleQubit: 1)
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
        let state = QuantumState(singleQubit: 0)
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
    func normalizeLargeState() throws {
        let numQubits = 7
        let amplitudes = [Complex<Double>](repeating: Complex(1.0, 0.0), count: 128)
        var state = QuantumState(numQubits: numQubits, amplitudes: amplitudes)
        #expect(state.isNormalized())

        for i in 0 ..< 128 {
            state.setAmplitude(ofState: i, amplitude: Complex(2.0, 0.0))
        }
        #expect(!state.isNormalized())

        try state.normalize()
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
        state.setAmplitude(ofState: 0, amplitude: Complex(2.0, 0.0))
        state.setAmplitude(ofState: 1, amplitude: Complex(2.0, 0.0))
        #expect(!state.validate())
    }

    @Test("State with NaN fails validation")
    func stateWithNaNFails() {
        var state = QuantumState(numQubits: 1)
        state.setAmplitude(ofState: 0, amplitude: Complex(Double.nan, 0.0))
        #expect(!state.validate())
    }

    @Test("State with Inf fails validation")
    func stateWithInfFails() {
        var state = QuantumState(numQubits: 1)
        state.setAmplitude(ofState: 1, amplitude: Complex(Double.infinity, 0.0))
        #expect(!state.validate())
    }
}

/// Test suite for quantum state error handling.
/// Validates error descriptions and error throwing behavior
/// for exceptional conditions like normalizing zero states.
@Suite("Error Handling")
struct StateErrorHandlingTests {
    @Test("QuantumStateError descriptions exist")
    func errorDescriptionsExist() {
        let error1 = QuantumStateError.cannotNormalizeZeroState
        let error2 = QuantumStateError.invalidAmplitudes

        #expect(error1.errorDescription != nil)
        #expect(error2.errorDescription != nil)
        #expect(error1.errorDescription?.contains("normalize") == true)
        #expect(error2.errorDescription?.contains("invalid") == true)
    }

    @Test("Normalizing near-zero state throws error")
    func normalizeZeroStateThrows() {
        var state = QuantumState(numQubits: 2)
        state.setAmplitude(ofState: 0, amplitude: Complex(1e-20, 0.0))
        state.setAmplitude(ofState: 1, amplitude: Complex(1e-20, 0.0))
        state.setAmplitude(ofState: 2, amplitude: Complex(1e-20, 0.0))
        state.setAmplitude(ofState: 3, amplitude: Complex(1e-20, 0.0))

        #expect(throws: QuantumStateError.self) { try state.normalize() }
    }

    @Test("Error thrown is cannotNormalizeZeroState")
    func correctErrorThrown() {
        var state = QuantumState(numQubits: 1)
        state.setAmplitude(ofState: 0, amplitude: Complex.zero)
        state.setAmplitude(ofState: 1, amplitude: Complex.zero)

        do {
            try state.normalize()
            Issue.record("Expected error to be thrown")
        } catch let error as QuantumStateError {
            #expect(error == .cannotNormalizeZeroState)
        } catch {
            Issue.record("Wrong error type thrown")
        }
    }
}

/// Test suite for bit manipulation utilities.
/// Validates little-endian qubit indexing operations essential for
/// gate application and measurement algorithms.
@Suite("Bit Manipulation")
struct BitManipulationTests {
    @Test("getBit extracts correct bit")
    func getBitCorrect() {
        let state = QuantumState(numQubits: 3)
        #expect(state.getBit(index: 5, qubit: 0) == 1)
        #expect(state.getBit(index: 5, qubit: 1) == 0)
        #expect(state.getBit(index: 5, qubit: 2) == 1)
    }

    @Test("setBit sets bit to 1")
    func setBitToOne() {
        let state = QuantumState(numQubits: 3)
        let result = state.setBit(index: 0, qubit: 1, value: 1)
        #expect(result == 2)
    }

    @Test("setBit clears bit to 0")
    func setBitToZero() {
        let state = QuantumState(numQubits: 3)
        let result = state.setBit(index: 7, qubit: 1, value: 0)
        #expect(result == 5)
    }

    @Test("flipBit flips bit correctly")
    func flipBitCorrect() {
        let state = QuantumState(numQubits: 3)
        let result1 = state.flipBit(index: 5, qubit: 0)
        #expect(result1 == 4)

        let result2 = state.flipBit(index: 5, qubit: 1)
        #expect(result2 == 7)
    }

    @Test("setBit with value 1 matches flipBit on zero bit")
    func setBitMatchesFlip() {
        let state = QuantumState(numQubits: 2)
        let index = 2

        let result1 = state.setBit(index: index, qubit: 0, value: 1)
        let result2 = state.flipBit(index: index, qubit: 0)

        #expect(result1 == result2)
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

        state.setAmplitude(ofState: 2, amplitude: newAmp)

        #expect(state.getAmplitude(ofState: 2) == newAmp)
        #expect(state.getAmplitude(ofState: 0) == Complex.one)
        #expect(state.getAmplitude(ofState: 1) == Complex.zero)
        #expect(state.getAmplitude(ofState: 3) == Complex.zero)
    }

    @Test("setAmplitude can create superposition")
    func setAmplitudeCreatesSuperposition() throws {
        var state = QuantumState(numQubits: 1)
        let invSqrt2 = 1.0 / sqrt(2.0)

        state.setAmplitude(ofState: 0, amplitude: Complex(invSqrt2, 0.0))
        state.setAmplitude(ofState: 1, amplitude: Complex(invSqrt2, 0.0))

        #expect(state.isNormalized())
        #expect(abs(state.probability(ofState: 0) - 0.5) < 1e-10)
        #expect(abs(state.probability(ofState: 1) - 0.5) < 1e-10)
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

        #expect(abs(state.getAmplitude(ofState: 0).real - invSqrt2) < 1e-10)
        #expect(abs(state.getAmplitude(ofState: 1).real - invSqrt2) < 1e-10)
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

/// Test suite for test-only initialization and edge case equality checks.
/// Validates internal test helpers and equality comparison edge cases.
@Suite("Test Utilities and Edge Cases")
struct QuantumStateTestUtilitiesTests {
    @Test("Bypass validation initializer for testing")
    func bypassValidationInit() {
        let amplitudes = [Complex(1.0, 0.0)]
        let state = QuantumState(numQubits: 2, amplitudes: amplitudes, bypassValidation: true)

        #expect(state.numQubits == 2)
        #expect(state.amplitudes.count == 1)
    }

    @Test("Equality check with mismatched amplitude counts")
    func equalityMismatchedAmplitudeCounts() {
        let state1 = QuantumState(numQubits: 2)
        let state2 = QuantumState(numQubits: 2, amplitudes: [Complex.one], bypassValidation: true)

        #expect(state1 != state2, "States with different amplitude counts should not be equal")
    }
}
