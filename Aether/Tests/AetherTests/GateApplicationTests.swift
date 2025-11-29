// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for single-qubit gate applications.
/// Validates Pauli, Hadamard, and phase operations on quantum states,
/// ensuring correct state vector transformations.
@Suite("Single-Qubit Gate Application")
struct qubitGateApplicationTests {
    @Test("X gate flips |0⟩ to |1⟩")
    func xGateFlips() {
        let state = QuantumState(qubit: 0)
        let newState = state.applying(.pauliX, to: 0)

        #expect(abs(newState.amplitude(of: 0).real) < 1e-10)
        #expect(abs(newState.amplitude(of: 1).real - 1.0) < 1e-10)
    }

    @Test("X gate flips |1⟩ to |0⟩")
    func xGateFlipsOne() {
        let state = QuantumState(qubit: 1)
        let newState = state.applying(.pauliX, to: 0)

        #expect(abs(newState.amplitude(of: 0).real - 1.0) < 1e-10)
        #expect(abs(newState.amplitude(of: 1).real) < 1e-10)
    }

    @Test("H gate creates superposition from |0⟩")
    func hadamardCreatesuperposition() {
        let state = QuantumState(qubit: 0)
        let newState = state.applying(.hadamard, to: 0)

        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(newState.amplitude(of: 0).real - invSqrt2) < 1e-10)
        #expect(abs(newState.amplitude(of: 1).real - invSqrt2) < 1e-10)
    }

    @Test("Z gate adds phase to |1⟩ component")
    func zGateAddsPhase() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let state = QuantumState(numQubits: 1, amplitudes: [
            Complex(invSqrt2, 0.0),
            Complex(invSqrt2, 0.0),
        ])

        let newState = state.applying(.pauliZ, to: 0)

        #expect(abs(newState.amplitude(of: 0).real - invSqrt2) < 1e-10)
        #expect(abs(newState.amplitude(of: 1).real - -invSqrt2) < 1e-10)
    }

    @Test("Identity gate leaves state unchanged")
    func identityNoChange() {
        let state = QuantumState(qubit: 0)
        let newState = state.applying(.identity, to: 0)

        #expect(state == newState)
    }
}

/// Test suite for quantum gate reversibility.
/// Validates that applying self-inverse gates twice returns to original state,
/// ensuring unitary gate implementations are mathematically correct.
@Suite("Gate Reversibility")
struct GateReversibilityTests {
    @Test("X·X returns original state")
    func xTwiceIsIdentity() {
        let state = QuantumState(qubit: 0)
        let state1 = state.applying(.pauliX, to: 0)
        let state2 = state1.applying(.pauliX, to: 0)

        #expect(state == state2)
    }

    @Test("H·H returns original state")
    func hadamardTwiceIsIdentity() {
        let state = QuantumState(qubit: 0)
        let state1 = state.applying(.hadamard, to: 0)
        let state2 = state1.applying(.hadamard, to: 0)

        #expect(state == state2)
    }

    @Test("Y·Y returns original state")
    func yTwiceIsIdentity() {
        let state = QuantumState(qubit: 1)
        let state1 = state.applying(.pauliY, to: 0)
        let state2 = state1.applying(.pauliY, to: 0)

        #expect(state == state2)
    }
}

/// Test suite for CNOT (Controlled-NOT) gate applications.
/// Validates controlled operations and entanglement creation,
/// ensuring correct two-qubit gate implementation in little-endian convention.
@Suite("CNOT Gate Application")
struct CNOTGateTests {
    @Test("CNOT on |00⟩ gives |00⟩")
    func cnotOnZeroZero() {
        let state = QuantumState(numQubits: 2)
        let newState = GateApplication.apply(
            .cnot,
            to: [0, 1],
            state: state
        )

        #expect(abs(newState.amplitude(of: 0).real - 1.0) < 1e-10)
        #expect(abs(newState.amplitude(of: 1).real) < 1e-10)
        #expect(abs(newState.amplitude(of: 2).real) < 1e-10)
        #expect(abs(newState.amplitude(of: 3).real) < 1e-10)
    }

    @Test("CNOT on |01⟩ gives |11⟩ (little-endian: control=0 is LSB)")
    func cnotOnZeroOne() {
        let amplitudes = [
            Complex<Double>.zero,
            Complex<Double>.one,
            Complex<Double>.zero,
            Complex<Double>.zero,
        ]
        let state = QuantumState(numQubits: 2, amplitudes: amplitudes)

        let newState = GateApplication.apply(
            .cnot,
            to: [0, 1],
            state: state
        )

        #expect(abs(newState.amplitude(of: 3).real - 1.0) < 1e-10)
        #expect(abs(newState.amplitude(of: 0).real) < 1e-10)
        #expect(abs(newState.amplitude(of: 1).real) < 1e-10)
        #expect(abs(newState.amplitude(of: 2).real) < 1e-10)
    }

    @Test("CNOT twice returns original state")
    func cnotTwiceIsIdentity() {
        let state = QuantumState(numQubits: 2)
        let state1 = GateApplication.apply(
            .cnot,
            to: [0, 1],
            state: state
        )
        let state2 = GateApplication.apply(
            .cnot,
            to: [0, 1],
            state: state1
        )

        #expect(state == state2)
    }
}

/// Test suite for Bell state (maximally entangled) creation.
/// Validates H·CNOT sequence produces correct entangled states,
/// fundamental for quantum algorithms and teleportation protocols.
@Suite("Bell State Creation")
struct BellStateTests {
    @Test("H·CNOT creates Bell state (|00⟩ + |11⟩)/√2")
    func createBellState() {
        var state = QuantumState(numQubits: 2)

        state = state.applying(.hadamard, to: 0)
        state = GateApplication.apply(
            .cnot,
            to: [0, 1],
            state: state
        )

        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(state.amplitude(of: 0).real - invSqrt2) < 1e-10)
        #expect(abs(state.amplitude(of: 1).real) < 1e-10)
        #expect(abs(state.amplitude(of: 2).real) < 1e-10)
        #expect(abs(state.amplitude(of: 3).real - invSqrt2) < 1e-10)
    }

    @Test("Bell state is normalized")
    func bellStateNormalized() {
        var state = QuantumState(numQubits: 2)
        state = state.applying(.hadamard, to: 0)
        state = GateApplication.apply(
            .cnot,
            to: [0, 1],
            state: state
        )

        #expect(state.isNormalized())
    }

    @Test("Bell state has correct probabilities")
    func bellStateProbabilities() {
        var state = QuantumState(numQubits: 2)
        state = state.applying(.hadamard, to: 0)
        state = GateApplication.apply(
            .cnot,
            to: [0, 1],
            state: state
        )

        #expect(abs(state.probability(of: 0) - 0.5) < 1e-10)
        #expect(abs(state.probability(of: 3) - 0.5) < 1e-10)
        #expect(abs(state.probability(of: 1)) < 1e-10)
        #expect(abs(state.probability(of: 2)) < 1e-10)
    }
}

/// Test suite for normalization preservation.
/// Ensures unitary gates maintain Σ|cᵢ|² = 1.0 throughout computations,
/// preventing numerical drift in deep quantum circuits.
@Suite("Normalization Preservation")
struct NormalizationPreservationTests {
    @Test("Single-qubit gate preserves normalization")
    func singleQubitPreservesNorm() {
        let state = QuantumState(numQubits: 3)
        let newState = state.applying(.hadamard, to: 1)

        #expect(state.isNormalized())
        #expect(newState.isNormalized())
    }

    @Test("CNOT preserves normalization")
    func cnotPreservesNorm() {
        var state = QuantumState(numQubits: 2)
        state = state.applying(.hadamard, to: 0)

        #expect(state.isNormalized())

        state = GateApplication.apply(
            .cnot,
            to: [0, 1],
            state: state
        )

        #expect(state.isNormalized())
    }

    @Test("Deep circuit preserves normalization")
    func deepCircuitPreservesNorm() {
        var state = QuantumState(numQubits: 2)

        for _ in 0 ..< 5 {
            state = state.applying(.hadamard, to: 0)
            state = state.applying(.hadamard, to: 0)
        }

        #expect(state.isNormalized())
    }
}

/// Test suite for multi-qubit system operations.
/// Validates gate independence and correct addressing in composite systems,
/// ensuring proper tensor product structure maintenance.
@Suite("Multi-Qubit System")
struct MultiQubitSystemTests {
    @Test("Gate on qubit 0 doesn't affect qubit 1")
    func gateOnQubit0Independent() {
        let state = QuantumState(numQubits: 2)
        let newState = state.applying(.pauliX, to: 0)

        #expect(abs(newState.amplitude(of: 1).real - 1.0) < 1e-10)
    }

    @Test("Gate on qubit 1 doesn't affect qubit 0")
    func gateOnQubit1Independent() {
        let state = QuantumState(numQubits: 2)
        let newState = state.applying(.pauliX, to: 1)

        #expect(abs(newState.amplitude(of: 2).real - 1.0) < 1e-10)
    }
}

/// Test suite for gate application scalability.
/// Demonstrates generic algorithms work identically from 1 to 24+ qubits,
/// proving serious quantum simulation capability beyond toy implementations.
@Suite("Gate Application Scalability")
struct GateApplicationScalabilityTests {
    @Test("Single-qubit gate works on 8-qubit system")
    func eightQubitSystem() {
        let state = QuantumState(numQubits: 8)
        let newState = state.applying(.hadamard, to: 3)

        #expect(newState.isNormalized())
        #expect(newState.numQubits == 8)
    }

    @Test("CNOT works on 12-qubit system")
    func twelveQubitSystem() {
        let state = QuantumState(numQubits: 12)
        let newState = GateApplication.apply(
            .cnot,
            to: [5, 7],
            state: state
        )

        #expect(newState.isNormalized())
        #expect(newState.numQubits == 12)
    }

    @Test("Gate application scales to 16 qubits")
    func sixteenQubitSystem() {
        let state = QuantumState(numQubits: 16)
        let newState = state.applying(.pauliX, to: 10)

        #expect(newState.isNormalized())
        #expect(newState.numQubits == 16)
    }
}

/// Test suite for phase gate applications.
/// Validates relative phase operations and special cases (S, T gates),
/// essential for quantum algorithms requiring phase manipulation.
@Suite("Phase Gate")
struct PhaseGateTests {
    @Test("Phase(0) acts as identity")
    func phaseZeroIsIdentity() {
        let state = QuantumState(qubit: 1)
        let newState = state.applying(.phase(0), to: 0)

        #expect(state == newState)
    }

    @Test("S gate applies π/2 phase")
    func sGatePhase() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let state = QuantumState(numQubits: 1, amplitudes: [
            Complex(invSqrt2, 0.0),
            Complex(invSqrt2, 0.0),
        ])

        let newState = state.applying(.sGate, to: 0)

        #expect(abs(newState.amplitude(of: 0).real - invSqrt2) < 1e-10)
        #expect(abs(newState.amplitude(of: 1).imaginary - invSqrt2) < 1e-10)
    }
}

/// Test suite for GHZ state creation.
/// Validates multi-qubit entanglement through sequential CNOT operations,
/// testing complex gate sequences and three-qubit correlations.
@Suite("GHZ State Creation")
struct GHZStateTests {
    @Test("Create 3-qubit GHZ state")
    func createGHZState() {
        var state = QuantumState(numQubits: 3)

        state = state.applying(.hadamard, to: 0)
        state = GateApplication.apply(
            .cnot,
            to: [0, 1],
            state: state
        )

        state = GateApplication.apply(
            .cnot,
            to: [0, 2],
            state: state
        )

        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(state.amplitude(of: 0).real - invSqrt2) < 1e-10)
        #expect(abs(state.amplitude(of: 7).real - invSqrt2) < 1e-10)

        for i in 1 ..< 7 {
            #expect(abs(state.amplitude(of: i).magnitude) < 1e-10)
        }
    }
}

/// Test suite for Toffoli (CCNOT) gate application.
/// Validates three-qubit controlled gate operations and quantum logic gates,
/// ensuring correct implementation of doubly-controlled operations.
@Suite("Toffoli Gate Application")
struct ToffoliGateApplicationTests {
    @Test("Toffoli on |000⟩ gives |000⟩")
    func toffoliOnAllZeros() {
        let state = QuantumState(numQubits: 3)
        let newState = GateApplication.apply(
            .toffoli,
            to: [0, 1, 2],
            state: state
        )

        #expect(abs(newState.amplitude(of: 0).real - 1.0) < 1e-10)
        for i in 1 ..< 8 {
            #expect(abs(newState.amplitude(of: i).magnitude) < 1e-10)
        }
    }

    @Test("Toffoli twice returns original state")
    func toffoliTwiceIsIdentity() {
        var amplitudes = Array(repeating: Complex<Double>.zero, count: 8)
        amplitudes[6] = .one

        let state = QuantumState(numQubits: 3, amplitudes: amplitudes)
        let state1 = GateApplication.apply(
            .toffoli,
            to: [0, 1, 2],
            state: state
        )
        let state2 = GateApplication.apply(
            .toffoli,
            to: [0, 1, 2],
            state: state1
        )

        #expect(state == state2)
    }

    @Test("Toffoli preserves normalization")
    func toffoliPreservesNormalization() {
        var state = QuantumState(numQubits: 3)
        state = state.applying(.hadamard, to: 0)
        state = state.applying(.hadamard, to: 1)

        #expect(state.isNormalized())

        let newState = GateApplication.apply(
            .toffoli,
            to: [0, 1, 2],
            state: state
        )

        #expect(newState.isNormalized())
    }
}

/// Test suite for SWAP and Controlled-Phase gate applications.
/// Validates two-qubit gates beyond CNOT, ensuring correct matrix-based
/// gate application for qubit routing and phase operations.
@Suite("SWAP and Controlled-Phase")
struct SwapAndControlledPhaseTests {
    @Test("SWAP gate exchanges qubits")
    func swapExchangesQubits() {
        var amplitudes = Array(repeating: Complex<Double>.zero, count: 4)
        amplitudes[2] = .one

        let state = QuantumState(numQubits: 2, amplitudes: amplitudes)
        let newState = GateApplication.apply(
            .swap,
            to: [0, 1],
            state: state
        )

        #expect(abs(newState.amplitude(of: 1).real - 1.0) < 1e-10)
        #expect(abs(newState.amplitude(of: 0).magnitude) < 1e-10)
        #expect(abs(newState.amplitude(of: 2).magnitude) < 1e-10)
        #expect(abs(newState.amplitude(of: 3).magnitude) < 1e-10)
    }

    @Test("SWAP preserves normalization")
    func swapPreservesNormalization() {
        var state = QuantumState(numQubits: 2)
        state = state.applying(.hadamard, to: 0)

        let newState = GateApplication.apply(
            .swap,
            to: [0, 1],
            state: state
        )

        #expect(newState.isNormalized())
    }

    @Test("Controlled-Phase applies phase to |11⟩ state")
    func controlledPhaseAppliesPhase() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitude = Complex(invSqrt2 * invSqrt2, 0.0)
        let amplitudes = Array(repeating: amplitude, count: 4)

        let state = QuantumState(numQubits: 2, amplitudes: amplitudes)
        let theta = Double.pi / 2.0
        let newState = GateApplication.apply(
            .controlledPhase(theta),
            to: [0, 1],
            state: state
        )

        #expect(abs(newState.amplitude(of: 0).real - amplitude.real) < 1e-10)
        #expect(abs(newState.amplitude(of: 1).real - amplitude.real) < 1e-10)
        #expect(abs(newState.amplitude(of: 2).real - amplitude.real) < 1e-10)

        let expectedPhase = amplitude * Complex<Double>(phase: theta)
        let actualAmp = newState.amplitude(of: 3)
        #expect(abs(actualAmp.real - expectedPhase.real) < 1e-10)
        #expect(abs(actualAmp.imaginary - expectedPhase.imaginary) < 1e-10)
        #expect(newState.isNormalized())
    }
}

/// Test suite for convenience extension methods.
/// Validates QuantumState.applying helper methods for ergonomic gate application.
@Suite("Convenience Extension")
struct ConvenienceExtensionTests {
    @Test("applying(gate:to:) convenience method works")
    func applyingWithArrayWorks() {
        let state = QuantumState(numQubits: 2)
        let newState = state.applying(.hadamard, to: 0)

        #expect(newState.isNormalized())
        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(newState.amplitude(of: 0).real - invSqrt2) < 1e-10)
    }

    @Test("applying(gate:toQubit:) convenience method works")
    func applyingToQubitWorks() {
        let state = QuantumState(qubit: 0)
        let newState = state.applying(.pauliX, to: 0)

        #expect(abs(newState.amplitude(of: 1).real - 1.0) < 1e-10)
    }
}
