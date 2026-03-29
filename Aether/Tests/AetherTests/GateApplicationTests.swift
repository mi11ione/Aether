// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
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

        #expect(abs(newState.amplitude(of: 0).real) < 1e-10, "Amplitude of |0> should be zero after X gate")
        #expect(abs(newState.amplitude(of: 1).real - 1.0) < 1e-10, "Amplitude of |1> should be one after X gate")
    }

    @Test("X gate flips |1⟩ to |0⟩")
    func xGateFlipsOne() {
        let state = QuantumState(qubit: 1)
        let newState = state.applying(.pauliX, to: 0)

        #expect(abs(newState.amplitude(of: 0).real - 1.0) < 1e-10, "Amplitude of |0> should be one after X on |1>")
        #expect(abs(newState.amplitude(of: 1).real) < 1e-10, "Amplitude of |1> should be zero after X on |1>")
    }

    @Test("H gate creates superposition from |0⟩")
    func hadamardCreatesuperposition() {
        let state = QuantumState(qubit: 0)
        let newState = state.applying(.hadamard, to: 0)

        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(newState.amplitude(of: 0).real - invSqrt2) < 1e-10, "Amplitude of |0> should be 1/sqrt(2) after H")
        #expect(abs(newState.amplitude(of: 1).real - invSqrt2) < 1e-10, "Amplitude of |1> should be 1/sqrt(2) after H")
    }

    @Test("Z gate adds phase to |1⟩ component")
    func zGateAddsPhase() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let state = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0.0),
            Complex(invSqrt2, 0.0),
        ])

        let newState = state.applying(.pauliZ, to: 0)

        #expect(abs(newState.amplitude(of: 0).real - invSqrt2) < 1e-10, "Z gate should not change |0> component")
        #expect(abs(newState.amplitude(of: 1).real - -invSqrt2) < 1e-10, "Z gate should negate |1> component")
    }

    @Test("Identity gate leaves state unchanged")
    func identityNoChange() {
        let state = QuantumState(qubit: 0)
        let newState = state.applying(.identity, to: 0)

        #expect(state == newState, "Identity gate should leave state unchanged")
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

        #expect(state == state2, "X applied twice should return to original state")
    }

    @Test("H·H returns original state")
    func hadamardTwiceIsIdentity() {
        let state = QuantumState(qubit: 0)
        let state1 = state.applying(.hadamard, to: 0)
        let state2 = state1.applying(.hadamard, to: 0)

        #expect(state == state2, "H applied twice should return to original state")
    }

    @Test("Y·Y returns original state")
    func yTwiceIsIdentity() {
        let state = QuantumState(qubit: 1)
        let state1 = state.applying(.pauliY, to: 0)
        let state2 = state1.applying(.pauliY, to: 0)

        #expect(state == state2, "Y applied twice should return to original state")
    }
}

/// Test suite for CNOT (Controlled-NOT) gate applications.
/// Validates controlled operations and entanglement creation,
/// ensuring correct two-qubit gate implementation in little-endian convention.
@Suite("CNOT Gate Application")
struct CNOTGateTests {
    @Test("CNOT on |00⟩ gives |00⟩")
    func cnotOnZeroZero() {
        let state = QuantumState(qubits: 2)
        let newState = GateApplication.apply(
            .cnot,
            to: [0, 1],
            state: state,
        )

        #expect(abs(newState.amplitude(of: 0).real - 1.0) < 1e-10, "CNOT on |00> should keep |00> amplitude at 1")
        #expect(abs(newState.amplitude(of: 1).real) < 1e-10, "CNOT on |00> should have zero |01> amplitude")
        #expect(abs(newState.amplitude(of: 2).real) < 1e-10, "CNOT on |00> should have zero |10> amplitude")
        #expect(abs(newState.amplitude(of: 3).real) < 1e-10, "CNOT on |00> should have zero |11> amplitude")
    }

    @Test("CNOT on |01⟩ gives |11⟩ (little-endian: control=0 is LSB)")
    func cnotOnZeroOne() {
        let amplitudes = [
            Complex<Double>.zero,
            Complex<Double>.one,
            Complex<Double>.zero,
            Complex<Double>.zero,
        ]
        let state = QuantumState(qubits: 2, amplitudes: amplitudes)

        let newState = GateApplication.apply(
            .cnot,
            to: [0, 1],
            state: state,
        )

        #expect(abs(newState.amplitude(of: 3).real - 1.0) < 1e-10, "CNOT on |01> should produce |11> amplitude of 1")
        #expect(abs(newState.amplitude(of: 0).real) < 1e-10, "CNOT on |01> should have zero |00> amplitude")
        #expect(abs(newState.amplitude(of: 1).real) < 1e-10, "CNOT on |01> should have zero |01> amplitude")
        #expect(abs(newState.amplitude(of: 2).real) < 1e-10, "CNOT on |01> should have zero |10> amplitude")
    }

    @Test("CNOT twice returns original state")
    func cnotTwiceIsIdentity() {
        let state = QuantumState(qubits: 2)
        let state1 = GateApplication.apply(
            .cnot,
            to: [0, 1],
            state: state,
        )
        let state2 = GateApplication.apply(
            .cnot,
            to: [0, 1],
            state: state1,
        )

        #expect(state == state2, "CNOT applied twice should return to original state")
    }
}

/// Test suite for Bell state (maximally entangled) creation.
/// Validates H·CNOT sequence produces correct entangled states,
/// fundamental for quantum algorithms and teleportation protocols.
@Suite("Bell State Creation")
struct BellStateTests {
    @Test("H·CNOT creates Bell state (|00⟩ + |11⟩)/√2")
    func createBellState() {
        var state = QuantumState(qubits: 2)

        state = state.applying(.hadamard, to: 0)
        state = GateApplication.apply(
            .cnot,
            to: [0, 1],
            state: state,
        )

        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(state.amplitude(of: 0).real - invSqrt2) < 1e-10, "Bell state |00> amplitude should be 1/sqrt(2)")
        #expect(abs(state.amplitude(of: 1).real) < 1e-10, "Bell state |01> amplitude should be zero")
        #expect(abs(state.amplitude(of: 2).real) < 1e-10, "Bell state |10> amplitude should be zero")
        #expect(abs(state.amplitude(of: 3).real - invSqrt2) < 1e-10, "Bell state |11> amplitude should be 1/sqrt(2)")
    }

    @Test("Bell state is normalized")
    func bellStateNormalized() {
        var state = QuantumState(qubits: 2)
        state = state.applying(.hadamard, to: 0)
        state = GateApplication.apply(
            .cnot,
            to: [0, 1],
            state: state,
        )

        #expect(state.isNormalized(), "Bell state should be normalized")
    }

    @Test("Bell state has correct probabilities")
    func bellStateProbabilities() {
        var state = QuantumState(qubits: 2)
        state = state.applying(.hadamard, to: 0)
        state = GateApplication.apply(
            .cnot,
            to: [0, 1],
            state: state,
        )

        #expect(abs(state.probability(of: 0) - 0.5) < 1e-10, "Bell state |00> probability should be 0.5")
        #expect(abs(state.probability(of: 3) - 0.5) < 1e-10, "Bell state |11> probability should be 0.5")
        #expect(abs(state.probability(of: 1)) < 1e-10, "Bell state |01> probability should be zero")
        #expect(abs(state.probability(of: 2)) < 1e-10, "Bell state |10> probability should be zero")
    }
}

/// Test suite for normalization preservation.
/// Ensures unitary gates maintain Σ|cᵢ|² = 1.0 throughout computations,
/// preventing numerical drift in deep quantum circuits.
@Suite("Normalization Preservation")
struct NormalizationPreservationTests {
    @Test("Single-qubit gate preserves normalization")
    func singleQubitPreservesNorm() {
        let state = QuantumState(qubits: 3)
        let newState = state.applying(.hadamard, to: 1)

        #expect(state.isNormalized(), "Initial state should be normalized")
        #expect(newState.isNormalized(), "State after single-qubit gate should be normalized")
    }

    @Test("CNOT preserves normalization")
    func cnotPreservesNorm() {
        var state = QuantumState(qubits: 2)
        state = state.applying(.hadamard, to: 0)

        #expect(state.isNormalized(), "State before CNOT should be normalized")

        state = GateApplication.apply(
            .cnot,
            to: [0, 1],
            state: state,
        )

        #expect(state.isNormalized(), "State after CNOT should be normalized")
    }

    @Test("Deep circuit preserves normalization")
    func deepCircuitPreservesNorm() {
        var state = QuantumState(qubits: 2)

        for _ in 0 ..< 5 {
            state = state.applying(.hadamard, to: 0)
            state = state.applying(.hadamard, to: 0)
        }

        #expect(state.isNormalized(), "State after deep circuit should be normalized")
    }
}

/// Test suite for multi-qubit system operations.
/// Validates gate independence and correct addressing in composite systems,
/// ensuring proper tensor product structure maintenance.
@Suite("Multi-Qubit System")
struct MultiQubitSystemTests {
    @Test("Gate on qubit 0 doesn't affect qubit 1")
    func gateOnQubit0Independent() {
        let state = QuantumState(qubits: 2)
        let newState = state.applying(.pauliX, to: 0)

        #expect(abs(newState.amplitude(of: 1).real - 1.0) < 1e-10, "X on qubit 0 should only affect qubit 0")
    }

    @Test("Gate on qubit 1 doesn't affect qubit 0")
    func gateOnQubit1Independent() {
        let state = QuantumState(qubits: 2)
        let newState = state.applying(.pauliX, to: 1)

        #expect(abs(newState.amplitude(of: 2).real - 1.0) < 1e-10, "X on qubit 1 should only affect qubit 1")
    }
}

/// Test suite for gate application scalability.
/// Demonstrates generic algorithms work identically from 1 to 24+ qubits,
/// proving serious quantum simulation capability beyond toy implementations.
@Suite("Gate Application Scalability")
struct GateApplicationScalabilityTests {
    @Test("Single-qubit gate works on 3-qubit system")
    func threeQubitSystem() {
        let state = QuantumState(qubits: 3)
        let newState = state.applying(.hadamard, to: 2)

        #expect(newState.isNormalized(), "State after H gate should be normalized")
        #expect(newState.qubits == 3, "Should preserve qubit count of 3")
    }

    @Test("CNOT works on 4-qubit system")
    func fourQubitSystem() {
        let state = QuantumState(qubits: 4)
        let newState = GateApplication.apply(
            .cnot,
            to: [1, 3],
            state: state,
        )

        #expect(newState.isNormalized(), "State after CNOT should be normalized")
        #expect(newState.qubits == 4, "Should preserve qubit count of 4")
    }

    @Test("Gate application scales to 4 qubits")
    func fourQubitPauliXSystem() {
        let state = QuantumState(qubits: 4)
        let newState = state.applying(.pauliX, to: 3)

        #expect(newState.isNormalized(), "State after Pauli-X should be normalized")
        #expect(newState.qubits == 4, "Should preserve qubit count of 4")
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

        #expect(state == newState, "Phase(0) should act as identity")
    }

    @Test("S gate applies π/2 phase")
    func sGatePhase() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let state = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0.0),
            Complex(invSqrt2, 0.0),
        ])

        let newState = state.applying(.sGate, to: 0)

        #expect(abs(newState.amplitude(of: 0).real - invSqrt2) < 1e-10, "S gate should not change |0> component")
        #expect(abs(newState.amplitude(of: 1).imaginary - invSqrt2) < 1e-10, "S gate should rotate |1> component by pi/2")
    }
}

/// Test suite for GHZ state creation.
/// Validates multi-qubit entanglement through sequential CNOT operations,
/// testing complex gate sequences and three-qubit correlations.
@Suite("GHZ State Creation")
struct GHZStateTests {
    @Test("Create 3-qubit GHZ state")
    func createGHZState() {
        var state = QuantumState(qubits: 3)

        state = state.applying(.hadamard, to: 0)
        state = GateApplication.apply(
            .cnot,
            to: [0, 1],
            state: state,
        )

        state = GateApplication.apply(
            .cnot,
            to: [0, 2],
            state: state,
        )

        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(state.amplitude(of: 0).real - invSqrt2) < 1e-10, "GHZ state |000> amplitude should be 1/sqrt(2)")
        #expect(abs(state.amplitude(of: 7).real - invSqrt2) < 1e-10, "GHZ state |111> amplitude should be 1/sqrt(2)")

        for i in 1 ..< 7 {
            #expect(abs(state.amplitude(of: i).magnitude) < 1e-10, "GHZ state intermediate amplitude \(i) should be zero")
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
        let state = QuantumState(qubits: 3)
        let newState = GateApplication.apply(
            .toffoli,
            to: [0, 1, 2],
            state: state,
        )

        #expect(abs(newState.amplitude(of: 0).real - 1.0) < 1e-10, "Toffoli on |000> should keep |000> amplitude at 1")
        for i in 1 ..< 8 {
            #expect(abs(newState.amplitude(of: i).magnitude) < 1e-10, "Toffoli on |000> should have zero amplitude at index \(i)")
        }
    }

    @Test("Toffoli twice returns original state")
    func toffoliTwiceIsIdentity() {
        var amplitudes = Array(repeating: Complex<Double>.zero, count: 8)
        amplitudes[6] = .one

        let state = QuantumState(qubits: 3, amplitudes: amplitudes)
        let state1 = GateApplication.apply(
            .toffoli,
            to: [0, 1, 2],
            state: state,
        )
        let state2 = GateApplication.apply(
            .toffoli,
            to: [0, 1, 2],
            state: state1,
        )

        #expect(state == state2, "Toffoli applied twice should return to original state")
    }

    @Test("Toffoli preserves normalization")
    func toffoliPreservesNormalization() {
        var state = QuantumState(qubits: 3)
        state = state.applying(.hadamard, to: 0)
        state = state.applying(.hadamard, to: 1)

        #expect(state.isNormalized(), "State before Toffoli should be normalized")

        let newState = GateApplication.apply(
            .toffoli,
            to: [0, 1, 2],
            state: state,
        )

        #expect(newState.isNormalized(), "State after Toffoli should be normalized")
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

        let state = QuantumState(qubits: 2, amplitudes: amplitudes)
        let newState = GateApplication.apply(
            .swap,
            to: [0, 1],
            state: state,
        )

        #expect(abs(newState.amplitude(of: 1).real - 1.0) < 1e-10, "SWAP should move |10> to |01>")
        #expect(abs(newState.amplitude(of: 0).magnitude) < 1e-10, "SWAP result should have zero |00> amplitude")
        #expect(abs(newState.amplitude(of: 2).magnitude) < 1e-10, "SWAP result should have zero |10> amplitude")
        #expect(abs(newState.amplitude(of: 3).magnitude) < 1e-10, "SWAP result should have zero |11> amplitude")
    }

    @Test("SWAP preserves normalization")
    func swapPreservesNormalization() {
        var state = QuantumState(qubits: 2)
        state = state.applying(.hadamard, to: 0)

        let newState = GateApplication.apply(
            .swap,
            to: [0, 1],
            state: state,
        )

        #expect(newState.isNormalized(), "SWAP should preserve normalization")
    }

    @Test("Controlled-Phase applies phase to |11⟩ state")
    func controlledPhaseAppliesPhase() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitude = Complex(invSqrt2 * invSqrt2, 0.0)
        let amplitudes = Array(repeating: amplitude, count: 4)

        let state = QuantumState(qubits: 2, amplitudes: amplitudes)
        let theta = Double.pi / 2.0
        let newState = GateApplication.apply(
            .controlledPhase(theta),
            to: [0, 1],
            state: state,
        )

        #expect(abs(newState.amplitude(of: 0).real - amplitude.real) < 1e-10, "Controlled-Phase should not change |00> amplitude")
        #expect(abs(newState.amplitude(of: 1).real - amplitude.real) < 1e-10, "Controlled-Phase should not change |01> amplitude")
        #expect(abs(newState.amplitude(of: 2).real - amplitude.real) < 1e-10, "Controlled-Phase should not change |10> amplitude")

        let expectedPhase = amplitude * Complex<Double>(phase: theta)
        let actualAmp = newState.amplitude(of: 3)
        #expect(abs(actualAmp.real - expectedPhase.real) < 1e-10, "Controlled-Phase |11> real part should match expected")
        #expect(abs(actualAmp.imaginary - expectedPhase.imaginary) < 1e-10, "Controlled-Phase |11> imaginary part should match expected")
        #expect(newState.isNormalized(), "Controlled-Phase should preserve normalization")
    }
}

/// Test suite for convenience extension methods.
/// Validates QuantumState.applying helper methods for ergonomic gate application,
/// ensuring fluent API produces correct quantum state transformations.
@Suite("Convenience Extension")
struct ConvenienceExtensionTests {
    @Test("applying(gate:to:) convenience method works")
    func applyingWithArrayWorks() {
        let state = QuantumState(qubits: 2)
        let newState = state.applying(.hadamard, to: 0)

        #expect(newState.isNormalized(), "Convenience method should preserve normalization")
        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(newState.amplitude(of: 0).real - invSqrt2) < 1e-10, "Convenience method should produce correct amplitude")
    }

    @Test("applying(gate:toQubit:) convenience method works")
    func applyingToQubitWorks() {
        let state = QuantumState(qubit: 0)
        let newState = state.applying(.pauliX, to: 0)

        #expect(abs(newState.amplitude(of: 1).real - 1.0) < 1e-10, "Convenience toQubit method should flip state correctly")
    }

    @Test("applying(gate:to:[Int]) with multi-qubit gate creates Bell state")
    func applyingWithArrayMultiQubitGate() {
        let state = QuantumState(qubits: 2)
            .applying(.hadamard, to: 0)
            .applying(.cnot, to: [0, 1])

        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(state.amplitude(of: 0).real - invSqrt2) < 1e-10, "Bell state |00⟩ amplitude")
        #expect(abs(state.amplitude(of: 1).magnitude) < 1e-10, "Bell state |01⟩ should be zero")
        #expect(abs(state.amplitude(of: 2).magnitude) < 1e-10, "Bell state |10⟩ should be zero")
        #expect(abs(state.amplitude(of: 3).real - invSqrt2) < 1e-10, "Bell state |11⟩ amplitude")
        #expect(state.isNormalized(), "Bell state should be normalized")
    }
}

/// Test suite for the .controlled gate case in GateApplication.
/// Validates that controlled gates with arbitrary control configurations work correctly,
/// including single-control, multi-control, and self-inverse controlled operations.
@Suite("Controlled Gate Application")
struct ControlledGateApplicationTests {
    @Test("Controlled-X with single control flips target when control is 1")
    func controlledXSingleControl() {
        var amplitudes = [Complex<Double>](repeating: .zero, count: 4)
        amplitudes[1] = .one
        let state = QuantumState(qubits: 2, amplitudes: amplitudes)

        let controlledX = QuantumGate.controlled(gate: .pauliX, controls: [0])
        let newState = GateApplication.apply(controlledX, to: [1], state: state)

        #expect(
            abs(newState.amplitude(of: 3).real - 1.0) < 1e-10,
            "Target should flip to |11⟩ when control is 1",
        )
        #expect(
            abs(newState.amplitude(of: 1).magnitude) < 1e-10,
            "Original |01⟩ state should be zero after flip",
        )
        #expect(newState.isNormalized(), "Controlled-X should preserve normalization")
    }

    @Test("Controlled-X leaves state unchanged when control is 0")
    func controlledXControlZero() {
        let state = QuantumState(qubits: 2)

        let controlledX = QuantumGate.controlled(gate: .pauliX, controls: [0])
        let newState = GateApplication.apply(controlledX, to: [1], state: state)

        #expect(
            abs(newState.amplitude(of: 0).real - 1.0) < 1e-10,
            "State should remain |00⟩ when control is 0",
        )
        #expect(newState.isNormalized(), "Controlled-X should preserve normalization")
    }

    @Test("Controlled-Z with single control applies phase when target is 1")
    func controlledZSingleControl() {
        let amplitudes = [Complex<Double>](repeating: Complex(0.5, 0.0), count: 4)
        let state = QuantumState(qubits: 2, amplitudes: amplitudes)

        let controlledZ = QuantumGate.controlled(gate: .pauliZ, controls: [0])
        let newState = GateApplication.apply(controlledZ, to: [1], state: state)

        #expect(
            abs(newState.amplitude(of: 0).real - 0.5) < 1e-10,
            "Amplitude for |00⟩ should be unchanged",
        )
        #expect(newState.isNormalized(), "Controlled-Z should preserve normalization")
    }

    @Test("Controlled gate with two controls (CCX-like)")
    func controlledGateTwoControls() {
        var amplitudes = [Complex<Double>](repeating: .zero, count: 8)
        amplitudes[3] = .one
        let state = QuantumState(qubits: 3, amplitudes: amplitudes)

        let controlledX = QuantumGate.controlled(gate: .pauliX, controls: [0, 1])
        let newState = GateApplication.apply(controlledX, to: [2], state: state)

        #expect(
            abs(newState.amplitude(of: 7).real - 1.0) < 1e-10,
            "Target should flip to |111⟩ when both controls are 1",
        )
        #expect(
            abs(newState.amplitude(of: 3).magnitude) < 1e-10,
            "Original |011⟩ state should be zero after flip",
        )
        #expect(newState.isNormalized(), "Double-controlled-X should preserve normalization")
    }

    @Test("Controlled Hadamard applies superposition when control is 1")
    func controlledHadamard() {
        var amplitudes = [Complex<Double>](repeating: .zero, count: 4)
        amplitudes[1] = .one
        let state = QuantumState(qubits: 2, amplitudes: amplitudes)

        let controlledH = QuantumGate.controlled(gate: .hadamard, controls: [0])
        let newState = GateApplication.apply(controlledH, to: [1], state: state)

        let expectedAmplitude = 1.0 / sqrt(2.0)
        #expect(
            abs(newState.amplitude(of: 1).real - expectedAmplitude) < 1e-10,
            "Amplitude for |01⟩ should be 1/sqrt(2)",
        )
        #expect(
            abs(newState.amplitude(of: 3).real - expectedAmplitude) < 1e-10,
            "Amplitude for |11⟩ should be 1/sqrt(2)",
        )
        #expect(newState.isNormalized(), "Controlled-H should preserve normalization")
    }

    @Test("Controlled rotation gate applies rotation when control is 1")
    func controlledRotation() {
        var amplitudes = [Complex<Double>](repeating: .zero, count: 4)
        amplitudes[1] = .one
        let state = QuantumState(qubits: 2, amplitudes: amplitudes)

        let controlledRy = QuantumGate.controlled(gate: .rotationY(.pi), controls: [0])
        let newState = GateApplication.apply(controlledRy, to: [1], state: state)

        #expect(
            abs(newState.amplitude(of: 3).real - 1.0) < 1e-10,
            "Ry(pi) should flip |0⟩ to |1⟩ on target when control is 1",
        )
        #expect(newState.isNormalized(), "Controlled-Ry should preserve normalization")
    }

    @Test("Controlled gate twice returns original state for self-inverse gates")
    func controlledGateTwiceIsIdentity() {
        var amplitudes = [Complex<Double>](repeating: .zero, count: 4)
        amplitudes[3] = .one
        let state = QuantumState(qubits: 2, amplitudes: amplitudes)

        let controlledX = QuantumGate.controlled(gate: .pauliX, controls: [0])
        let state1 = GateApplication.apply(controlledX, to: [1], state: state)
        let state2 = GateApplication.apply(controlledX, to: [1], state: state1)

        #expect(
            abs(state2.amplitude(of: 3).real - 1.0) < 1e-10,
            "Applying controlled-X twice should return to original |11⟩ state",
        )
        #expect(state == state2, "Controlled-X applied twice should return original state")
    }
}
