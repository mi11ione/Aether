// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for single-qubit gate applications on MPS.
/// Validates Hadamard, Pauli, and phase gates preserve state correctness
/// when applied to Matrix Product State representations.
@Suite("MPS Single-Qubit Gates")
struct MPSSingleQubitGateTests {
    @Test("Hadamard on |0> creates equal superposition")
    func hadamardOnZeroState() {
        var mps = MatrixProductState(qubits: 3, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)

        let amp0 = mps.amplitude(of: 0b000)
        let amp1 = mps.amplitude(of: 0b001)
        let invSqrt2 = 1.0 / sqrt(2.0)

        #expect(abs(amp0.real - invSqrt2) < 1e-10, "H|0> amplitude for |0> should be 1/sqrt(2)")
        #expect(abs(amp1.real - invSqrt2) < 1e-10, "H|0> amplitude for |1> should be 1/sqrt(2)")
        #expect(abs(amp0.imaginary) < 1e-10, "Imaginary part of |0> amplitude should be zero")
        #expect(abs(amp1.imaginary) < 1e-10, "Imaginary part of |1> amplitude should be zero")
    }

    @Test("Pauli-X flips |0> to |1>")
    func pauliXFlipsState() {
        var mps = MatrixProductState(qubits: 2, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.pauliX, to: 0, mps: &mps)

        let amp0 = mps.amplitude(of: 0b00)
        let amp1 = mps.amplitude(of: 0b01)

        #expect(abs(amp0.magnitude) < 1e-10, "X|0> should have zero amplitude for |0>")
        #expect(abs(amp1.real - 1.0) < 1e-10, "X|0> should have amplitude 1 for |1>")
    }

    @Test("Pauli-Y applies correct transformation")
    func pauliYTransformation() {
        var mps = MatrixProductState(qubits: 2, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.pauliY, to: 0, mps: &mps)

        let amp0 = mps.amplitude(of: 0b00)
        let amp1 = mps.amplitude(of: 0b01)

        #expect(abs(amp0.magnitude) < 1e-10, "Y|0> should have zero amplitude for |0>")
        #expect(abs(amp1.imaginary - 1.0) < 1e-10, "Y|0> should have amplitude i for |1>")
    }

    @Test("Pauli-Z leaves |0> unchanged")
    func pauliZOnZero() {
        var mps = MatrixProductState(qubits: 2, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.pauliZ, to: 0, mps: &mps)

        let amp0 = mps.amplitude(of: 0b00)
        #expect(abs(amp0.real - 1.0) < 1e-10, "Z|0> should remain |0>")
    }

    @Test("S gate applies pi/2 phase to |1>")
    func sGatePhase() {
        var mps = MatrixProductState(qubits: 2, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applySingleQubitGate(.sGate, to: 0, mps: &mps)

        let amp1 = mps.amplitude(of: 0b01)
        let invSqrt2 = 1.0 / sqrt(2.0)

        #expect(abs(amp1.real) < 1e-10, "S gate should rotate |1> component to imaginary axis")
        #expect(abs(amp1.imaginary - invSqrt2) < 1e-10, "S|+> should have i/sqrt(2) for |1>")
    }

    @Test("T gate applies pi/4 phase to |1>")
    func tGatePhase() {
        var mps = MatrixProductState(qubits: 2, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applySingleQubitGate(.tGate, to: 0, mps: &mps)

        let amp1 = mps.amplitude(of: 0b01)
        let invSqrt2 = 1.0 / sqrt(2.0)
        let expectedPhase = Complex<Double>(phase: .pi / 4) * invSqrt2

        #expect(abs(amp1.real - expectedPhase.real) < 1e-10, "T gate should apply pi/4 phase (real)")
        #expect(abs(amp1.imaginary - expectedPhase.imaginary) < 1e-10, "T gate should apply pi/4 phase (imag)")
    }

    @Test("Gate on middle qubit preserves other qubits")
    func gateOnMiddleQubit() {
        var mps = MatrixProductState(qubits: 5, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.pauliX, to: 2, mps: &mps)

        let expectedState = 0b00100
        let amp = mps.amplitude(of: expectedState)

        #expect(abs(amp.real - 1.0) < 1e-10, "X on qubit 2 should flip only that qubit")
        #expect(abs(mps.amplitude(of: 0).magnitude) < 1e-10, "Other states should be zero")
    }

    @Test("Gate on last qubit works correctly")
    func gateOnLastQubit() {
        var mps = MatrixProductState(qubits: 4, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.pauliX, to: 3, mps: &mps)

        let amp = mps.amplitude(of: 0b1000)
        #expect(abs(amp.real - 1.0) < 1e-10, "X on last qubit should produce |1000>")
    }
}

/// Tests normalization preservation of MPS gate application.
/// Validates that unitary operations maintain sum of squared amplitudes equals 1.
/// Ensures state remains physically valid after single and multi-qubit gates.
@Suite("MPS Gate Normalization Preservation")
struct MPSGateNormalizationTests {
    @Test("Single-qubit Hadamard preserves normalization")
    func hadamardPreservesNorm() {
        var mps = MatrixProductState(qubits: 4, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)

        #expect(mps.isNormalized(), "Hadamard should preserve MPS normalization")
    }

    @Test("Multiple single-qubit gates preserve normalization")
    func multipleGatesPreserveNorm() {
        var mps = MatrixProductState(qubits: 4, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applySingleQubitGate(.pauliX, to: 1, mps: &mps)
        MPSGateApplication.applySingleQubitGate(.sGate, to: 2, mps: &mps)
        MPSGateApplication.applySingleQubitGate(.tGate, to: 3, mps: &mps)

        #expect(mps.isNormalized(), "Multiple single-qubit gates should preserve normalization")
    }

    @Test("Two-qubit CNOT preserves normalization")
    func cnotPreservesNorm() {
        var mps = MatrixProductState(qubits: 4, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 1, mps: &mps)

        #expect(mps.isNormalized(), "CNOT should preserve MPS normalization")
    }

    @Test("Two-qubit CZ preserves normalization")
    func czPreservesNorm() {
        var mps = MatrixProductState(qubits: 4, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 1, mps: &mps)
        MPSGateApplication.applyTwoQubitGate(.cz, control: 0, target: 1, mps: &mps)

        #expect(mps.isNormalized(), "CZ should preserve MPS normalization")
    }

    @Test("SWAP gate preserves normalization")
    func swapPreservesNorm() {
        var mps = MatrixProductState(qubits: 4, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applyTwoQubitGate(.swap, control: 0, target: 1, mps: &mps)

        #expect(mps.isNormalized(), "SWAP should preserve MPS normalization")
    }

    @Test("Deep circuit preserves normalization")
    func deepCircuitPreservesNorm() {
        var mps = MatrixProductState(qubits: 4, maxBondDimension: 32)

        for i in 0 ..< 4 {
            MPSGateApplication.applySingleQubitGate(.hadamard, to: i, mps: &mps)
        }
        for i in 0 ..< 3 {
            MPSGateApplication.applyTwoQubitGate(.cnot, control: i, target: i + 1, mps: &mps)
        }
        for i in 0 ..< 4 {
            MPSGateApplication.applySingleQubitGate(.tGate, to: i, mps: &mps)
        }

        #expect(mps.isNormalized(), "Deep circuit should preserve MPS normalization")
    }
}

/// Tests two-qubit gates on adjacent qubits of MPS gate application.
/// Validates CNOT, CZ, and SWAP operations on neighboring MPS sites.
/// Ensures correct state transformations for all basis state inputs.
@Suite("MPS Adjacent Two-Qubit Gates")
struct MPSAdjacentTwoQubitGateTests {
    @Test("CNOT on adjacent qubits |00> -> |00>")
    func cnotOnZeroZero() {
        var mps = MatrixProductState(qubits: 2, maxBondDimension: 16)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 1, mps: &mps)

        #expect(abs(mps.amplitude(of: 0b00).real - 1.0) < 1e-10, "CNOT|00> should be |00>")
    }

    @Test("CNOT on adjacent qubits |01> -> |11>")
    func cnotOnZeroOne() {
        var mps = MatrixProductState(qubits: 2, basisState: 0b01, maxBondDimension: 16)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 1, mps: &mps)

        #expect(abs(mps.amplitude(of: 0b11).real - 1.0) < 1e-10, "CNOT|01> should be |11>")
    }

    @Test("CNOT on adjacent qubits |10> -> |10>")
    func cnotOnOneZero() {
        var mps = MatrixProductState(qubits: 2, basisState: 0b10, maxBondDimension: 16)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 1, mps: &mps)

        #expect(abs(mps.amplitude(of: 0b10).real - 1.0) < 1e-10, "CNOT|10> should be |10>")
    }

    @Test("CNOT on adjacent qubits |11> -> |01>")
    func cnotOnOneOne() {
        var mps = MatrixProductState(qubits: 2, basisState: 0b11, maxBondDimension: 16)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 1, mps: &mps)

        #expect(abs(mps.amplitude(of: 0b01).real - 1.0) < 1e-10, "CNOT|11> should be |01>")
    }

    @Test("CZ on adjacent qubits applies phase only to |11>")
    func czAdjacentQubits() {
        var mps = MatrixProductState(qubits: 2, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 1, mps: &mps)
        MPSGateApplication.applyTwoQubitGate(.cz, control: 0, target: 1, mps: &mps)

        let amp00 = mps.amplitude(of: 0b00)
        let amp01 = mps.amplitude(of: 0b01)
        let amp10 = mps.amplitude(of: 0b10)
        let amp11 = mps.amplitude(of: 0b11)

        #expect(abs(amp00.real - 0.5) < 1e-10, "CZ should not change |00> amplitude")
        #expect(abs(amp01.real - 0.5) < 1e-10, "CZ should not change |01> amplitude")
        #expect(abs(amp10.real - 0.5) < 1e-10, "CZ should not change |10> amplitude")
        #expect(abs(amp11.real + 0.5) < 1e-10, "CZ should flip sign of |11> amplitude")
    }

    @Test("SWAP on adjacent qubits exchanges states")
    func swapAdjacentQubits() {
        var mps = MatrixProductState(qubits: 2, basisState: 0b01, maxBondDimension: 16)
        MPSGateApplication.applyTwoQubitGate(.swap, control: 0, target: 1, mps: &mps)

        #expect(abs(mps.amplitude(of: 0b10).real - 1.0) < 1e-10, "SWAP|01> should be |10>")
    }

    @Test("CNOT with control > target works correctly")
    func cnotControlGreaterThanTarget() {
        var mps = MatrixProductState(qubits: 2, basisState: 0b10, maxBondDimension: 16)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 1, target: 0, mps: &mps)

        #expect(abs(mps.amplitude(of: 0b11).real - 1.0) < 1e-10, "CNOT with control=1 on |10> should be |11>")
    }
}

/// Tests Bell state creation of MPS gate application.
/// Validates H + CNOT sequence produces correct entangled state.
/// Ensures proper amplitudes, normalization, and bond dimension.
@Suite("MPS Bell State Creation")
struct MPSBellStateTests {
    @Test("H + CNOT creates Bell state (|00> + |11>)/sqrt(2)")
    func bellStateCreation() {
        var mps = MatrixProductState(qubits: 2, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 1, mps: &mps)

        let amp00 = mps.amplitude(of: 0b00)
        let amp01 = mps.amplitude(of: 0b01)
        let amp10 = mps.amplitude(of: 0b10)
        let amp11 = mps.amplitude(of: 0b11)
        let invSqrt2 = 1.0 / sqrt(2.0)

        #expect(abs(amp00.real - invSqrt2) < 1e-10, "Bell state |00> amplitude should be 1/sqrt(2)")
        #expect(abs(amp01.magnitude) < 1e-10, "Bell state |01> amplitude should be 0")
        #expect(abs(amp10.magnitude) < 1e-10, "Bell state |10> amplitude should be 0")
        #expect(abs(amp11.real - invSqrt2) < 1e-10, "Bell state |11> amplitude should be 1/sqrt(2)")
    }

    @Test("Bell state is normalized")
    func bellStateNormalized() {
        var mps = MatrixProductState(qubits: 2, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 1, mps: &mps)

        #expect(mps.isNormalized(), "Bell state should be normalized")
    }

    @Test("Bell state has correct probabilities")
    func bellStateProbabilities() {
        var mps = MatrixProductState(qubits: 2, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 1, mps: &mps)

        #expect(abs(mps.probability(of: 0b00) - 0.5) < 1e-10, "Bell state P(|00>) should be 0.5")
        #expect(abs(mps.probability(of: 0b11) - 0.5) < 1e-10, "Bell state P(|11>) should be 0.5")
        #expect(abs(mps.probability(of: 0b01)) < 1e-10, "Bell state P(|01>) should be 0")
        #expect(abs(mps.probability(of: 0b10)) < 1e-10, "Bell state P(|10>) should be 0")
    }

    @Test("Bell state bond dimension is 2")
    func bellStateBondDimension() {
        var mps = MatrixProductState(qubits: 2, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 1, mps: &mps)

        #expect(mps.currentMaxBondDimension == 2, "Bell state should have bond dimension 2")
    }
}

/// Tests non-adjacent two-qubit gates of MPS gate application.
/// Validates SWAP network correctly brings qubits together.
/// Ensures long-range gates produce correct results via site swapping.
@Suite("MPS Non-Adjacent Two-Qubit Gates")
struct MPSNonAdjacentTwoQubitGateTests {
    @Test("CNOT on qubits 0 and 2 works correctly")
    func cnotNonAdjacent() {
        var mps = MatrixProductState(qubits: 3, basisState: 0b001, maxBondDimension: 16)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 2, mps: &mps)

        #expect(abs(mps.amplitude(of: 0b101).real - 1.0) < 1e-10, "CNOT on non-adjacent |001> should be |101>")
    }

    @Test("CNOT on qubits 0 and 3 works correctly")
    func cnotFarApart() {
        var mps = MatrixProductState(qubits: 4, basisState: 0b0001, maxBondDimension: 16)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 3, mps: &mps)

        #expect(abs(mps.amplitude(of: 0b1001).real - 1.0) < 1e-10, "CNOT on far apart |0001> should be |1001>")
    }

    @Test("CZ on non-adjacent qubits applies phase correctly")
    func czNonAdjacent() {
        var mps = MatrixProductState(qubits: 4, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 2, mps: &mps)
        MPSGateApplication.applyTwoQubitGate(.cz, control: 0, target: 2, mps: &mps)

        let amp0000 = mps.amplitude(of: 0b0000)
        let amp0001 = mps.amplitude(of: 0b0001)
        let amp0100 = mps.amplitude(of: 0b0100)
        let amp0101 = mps.amplitude(of: 0b0101)

        #expect(abs(amp0000.real - 0.5) < 1e-10, "CZ should not change |0000> amplitude")
        #expect(abs(amp0001.real - 0.5) < 1e-10, "CZ should not change |0001> amplitude")
        #expect(abs(amp0100.real - 0.5) < 1e-10, "CZ should not change |0100> amplitude")
        #expect(abs(amp0101.real + 0.5) < 1e-10, "CZ should flip sign of |0101> amplitude")
    }

    @Test("Non-adjacent gate preserves normalization")
    func nonAdjacentPreservesNorm() {
        var mps = MatrixProductState(qubits: 5, maxBondDimension: 32)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 4, mps: &mps)

        #expect(mps.isNormalized(), "Non-adjacent CNOT should preserve normalization")
    }

    @Test("Bell state via non-adjacent qubits")
    func bellStateNonAdjacent() {
        var mps = MatrixProductState(qubits: 4, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 3, mps: &mps)

        let amp0000 = mps.amplitude(of: 0b0000)
        let amp1001 = mps.amplitude(of: 0b1001)
        let invSqrt2 = 1.0 / sqrt(2.0)

        #expect(abs(amp0000.real - invSqrt2) < 1e-10, "Bell state via non-adjacent |0000> amplitude")
        #expect(abs(amp1001.real - invSqrt2) < 1e-10, "Bell state via non-adjacent |1001> amplitude")
    }
}

/// Tests Toffoli gate decomposition of MPS gate application.
/// Validates three-qubit gate truth table and correctness.
/// Ensures CCX gate works for all eight basis state inputs.
@Suite("MPS Toffoli Decomposition")
struct MPSToffoliDecompositionTests {
    @Test("Toffoli |000> -> |000>")
    func toffoliZeroZeroZero() {
        var mps = MatrixProductState(qubits: 3, basisState: 0b000, maxBondDimension: 32)
        MPSGateApplication.applyToffoli(control1: 0, control2: 1, target: 2, mps: &mps)

        #expect(abs(mps.amplitude(of: 0b000).real - 1.0) < 1e-8, "Toffoli|000> should be |000>")
    }

    @Test("Toffoli |001> -> |001>")
    func toffoliZeroZeroOne() {
        var mps = MatrixProductState(qubits: 3, basisState: 0b001, maxBondDimension: 32)
        MPSGateApplication.applyToffoli(control1: 0, control2: 1, target: 2, mps: &mps)

        #expect(abs(mps.amplitude(of: 0b001).real - 1.0) < 1e-8, "Toffoli|001> should be |001>")
    }

    @Test("Toffoli |010> -> |010>")
    func toffoliZeroOneZero() {
        var mps = MatrixProductState(qubits: 3, basisState: 0b010, maxBondDimension: 32)
        MPSGateApplication.applyToffoli(control1: 0, control2: 1, target: 2, mps: &mps)

        #expect(abs(mps.amplitude(of: 0b010).real - 1.0) < 1e-8, "Toffoli|010> should be |010>")
    }

    @Test("Toffoli |011> -> |111> (both controls active)")
    func toffoliZeroOneOne() {
        var mps = MatrixProductState(qubits: 3, basisState: 0b011, maxBondDimension: 32)
        MPSGateApplication.applyToffoli(control1: 0, control2: 1, target: 2, mps: &mps)

        #expect(abs(mps.amplitude(of: 0b111).real - 1.0) < 1e-8, "Toffoli|011> should be |111>")
    }

    @Test("Toffoli |100> -> |100>")
    func toffoliOneZeroZero() {
        var mps = MatrixProductState(qubits: 3, basisState: 0b100, maxBondDimension: 32)
        MPSGateApplication.applyToffoli(control1: 0, control2: 1, target: 2, mps: &mps)

        #expect(abs(mps.amplitude(of: 0b100).real - 1.0) < 1e-8, "Toffoli|100> should be |100>")
    }

    @Test("Toffoli |101> -> |101>")
    func toffoliOneZeroOne() {
        var mps = MatrixProductState(qubits: 3, basisState: 0b101, maxBondDimension: 32)
        MPSGateApplication.applyToffoli(control1: 0, control2: 1, target: 2, mps: &mps)

        #expect(abs(mps.amplitude(of: 0b101).real - 1.0) < 1e-8, "Toffoli|101> should be |101>")
    }

    @Test("Toffoli |110> -> |110>")
    func toffoliOneOneZero() {
        var mps = MatrixProductState(qubits: 3, basisState: 0b110, maxBondDimension: 32)
        MPSGateApplication.applyToffoli(control1: 0, control2: 1, target: 2, mps: &mps)

        #expect(abs(mps.amplitude(of: 0b110).real - 1.0) < 1e-8, "Toffoli|110> should be |110>")
    }

    @Test("Toffoli |111> -> |011> (flip target)")
    func toffoliOneOneOne() {
        var mps = MatrixProductState(qubits: 3, basisState: 0b111, maxBondDimension: 32)
        MPSGateApplication.applyToffoli(control1: 0, control2: 1, target: 2, mps: &mps)

        #expect(abs(mps.amplitude(of: 0b011).real - 1.0) < 1e-8, "Toffoli|111> should be |011>")
    }

    @Test("Toffoli preserves normalization")
    func toffoliPreservesNorm() {
        var mps = MatrixProductState(qubits: 3, maxBondDimension: 32)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 1, mps: &mps)
        MPSGateApplication.applyToffoli(control1: 0, control2: 1, target: 2, mps: &mps)

        #expect(mps.isNormalized(), "Toffoli should preserve normalization")
    }

    @Test("Toffoli with non-adjacent qubits")
    func toffoliNonAdjacent() {
        var mps = MatrixProductState(qubits: 5, basisState: 0b00011, maxBondDimension: 32)
        MPSGateApplication.applyToffoli(control1: 0, control2: 1, target: 4, mps: &mps)

        #expect(abs(mps.amplitude(of: 0b10011).real - 1.0) < 1e-8, "Toffoli with non-adjacent qubits should work")
    }
}

/// Tests MPS versus QuantumState comparison of MPS gate application.
/// Validates MPS gate application matches full statevector simulation.
/// Ensures numerical equivalence for various circuit configurations.
@Suite("MPS vs QuantumState Comparison")
struct MPSQuantumStateComparisonTests {
    @Test("Single Hadamard matches QuantumState")
    func singleHadamardComparison() {
        var mps = MatrixProductState(qubits: 3, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 1, mps: &mps)

        var state = QuantumState(qubits: 3)
        state = state.applying(.hadamard, to: 1)

        for i in 0 ..< 8 {
            let mpsAmp = mps.amplitude(of: i)
            let stateAmp = state.amplitude(of: i)
            #expect(abs(mpsAmp.real - stateAmp.real) < 1e-10, "MPS and QuantumState should match for state \(i) (real)")
            #expect(abs(mpsAmp.imaginary - stateAmp.imaginary) < 1e-10, "MPS and QuantumState should match for state \(i) (imag)")
        }
    }

    @Test("Bell state matches QuantumState")
    func bellStateComparison() {
        var mps = MatrixProductState(qubits: 2, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 1, mps: &mps)

        var state = QuantumState(qubits: 2)
        state = state.applying(.hadamard, to: 0)
        state = state.applying(.cnot, to: [0, 1])

        for i in 0 ..< 4 {
            let mpsAmp = mps.amplitude(of: i)
            let stateAmp = state.amplitude(of: i)
            #expect(abs(mpsAmp.real - stateAmp.real) < 1e-10, "Bell state MPS and QuantumState should match (real)")
            #expect(abs(mpsAmp.imaginary - stateAmp.imaginary) < 1e-10, "Bell state MPS and QuantumState should match (imag)")
        }
    }

    @Test("GHZ state matches QuantumState")
    func ghzStateComparison() {
        var mps = MatrixProductState(qubits: 3, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 1, mps: &mps)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 2, mps: &mps)

        var state = QuantumState(qubits: 3)
        state = state.applying(.hadamard, to: 0)
        state = state.applying(.cnot, to: [0, 1])
        state = state.applying(.cnot, to: [0, 2])

        for i in 0 ..< 8 {
            let mpsAmp = mps.amplitude(of: i)
            let stateAmp = state.amplitude(of: i)
            #expect(abs(mpsAmp.real - stateAmp.real) < 1e-10, "GHZ state MPS and QuantumState should match (real)")
            #expect(abs(mpsAmp.imaginary - stateAmp.imaginary) < 1e-10, "GHZ state MPS and QuantumState should match (imag)")
        }
    }

    @Test("Complex circuit matches QuantumState")
    func complexCircuitComparison() {
        var mps = MatrixProductState(qubits: 4, maxBondDimension: 32)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applySingleQubitGate(.pauliX, to: 2, mps: &mps)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 1, mps: &mps)
        MPSGateApplication.applySingleQubitGate(.sGate, to: 1, mps: &mps)
        MPSGateApplication.applyTwoQubitGate(.cz, control: 1, target: 3, mps: &mps)

        var state = QuantumState(qubits: 4)
        state = state.applying(.hadamard, to: 0)
        state = state.applying(.pauliX, to: 2)
        state = state.applying(.cnot, to: [0, 1])
        state = state.applying(.sGate, to: 1)
        state = state.applying(.cz, to: [1, 3])

        for i in 0 ..< 16 {
            let mpsAmp = mps.amplitude(of: i)
            let stateAmp = state.amplitude(of: i)
            #expect(abs(mpsAmp.real - stateAmp.real) < 1e-10, "Complex circuit MPS should match QuantumState (real, state \(i))")
            #expect(abs(mpsAmp.imaginary - stateAmp.imaginary) < 1e-10, "Complex circuit MPS should match QuantumState (imag, state \(i))")
        }
    }

    @Test("Rotation gates match QuantumState")
    func rotationGatesComparison() {
        let angle = Double.pi / 3

        var mps = MatrixProductState(qubits: 2, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.rotationY(angle), to: 0, mps: &mps)
        MPSGateApplication.applySingleQubitGate(.rotationZ(angle), to: 1, mps: &mps)

        var state = QuantumState(qubits: 2)
        state = state.applying(.rotationY(angle), to: 0)
        state = state.applying(.rotationZ(angle), to: 1)

        for i in 0 ..< 4 {
            let mpsAmp = mps.amplitude(of: i)
            let stateAmp = state.amplitude(of: i)
            #expect(abs(mpsAmp.real - stateAmp.real) < 1e-10, "Rotation gates MPS should match QuantumState (real)")
            #expect(abs(mpsAmp.imaginary - stateAmp.imaginary) < 1e-10, "Rotation gates MPS should match QuantumState (imag)")
        }
    }
}

/// Tests bond dimension truncation of MPS gate application.
/// Validates truncation occurs and error is tracked correctly.
/// Ensures bounded bond dimension with reasonable approximation quality.
@Suite("MPS Truncation")
struct MPSTruncationTests {
    @Test("Bond dimension bounded by maxBondDimension")
    func bondDimensionBounded() {
        let maxBond = 4
        var mps = MatrixProductState(qubits: 6, maxBondDimension: maxBond)

        for i in 0 ..< 6 {
            MPSGateApplication.applySingleQubitGate(.hadamard, to: i, mps: &mps)
        }
        for i in 0 ..< 5 {
            MPSGateApplication.applyTwoQubitGate(.cnot, control: i, target: i + 1, mps: &mps)
        }

        #expect(mps.currentMaxBondDimension <= maxBond, "Bond dimension should not exceed maxBondDimension")
    }

    @Test("Truncation statistics updated after two-qubit gates")
    func truncationStatisticsUpdated() {
        var mps = MatrixProductState(qubits: 4, maxBondDimension: 2)

        let initialCount = mps.truncationStatistics.truncationCount

        for i in 0 ..< 4 {
            MPSGateApplication.applySingleQubitGate(.hadamard, to: i, mps: &mps)
        }
        for i in 0 ..< 3 {
            MPSGateApplication.applyTwoQubitGate(.cnot, control: i, target: i + 1, mps: &mps)
        }

        #expect(mps.truncationStatistics.truncationCount > initialCount, "Truncation count should increase after two-qubit gates")
    }

    @Test("Low bond dimension still maintains reasonable approximation")
    func lowBondDimensionApproximation() {
        var mps = MatrixProductState(qubits: 4, maxBondDimension: 4)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 1, mps: &mps)

        #expect(mps.isNormalized(), "Even with truncation, MPS should remain normalized")

        let prob00 = mps.probability(of: 0b0000)
        let prob11 = mps.probability(of: 0b0011)

        #expect(abs(prob00 - 0.5) < 0.01, "Truncated Bell state should approximate 0.5 probability for |00>")
        #expect(abs(prob11 - 0.5) < 0.01, "Truncated Bell state should approximate 0.5 probability for |11>")
    }

    @Test("Higher bond dimension gives better accuracy")
    func higherBondDimensionBetterAccuracy() {
        var mpsLow = MatrixProductState(qubits: 5, maxBondDimension: 2)
        var mpsHigh = MatrixProductState(qubits: 5, maxBondDimension: 16)

        for i in 0 ..< 5 {
            MPSGateApplication.applySingleQubitGate(.hadamard, to: i, mps: &mpsLow)
            MPSGateApplication.applySingleQubitGate(.hadamard, to: i, mps: &mpsHigh)
        }
        for i in 0 ..< 4 {
            MPSGateApplication.applyTwoQubitGate(.cnot, control: i, target: i + 1, mps: &mpsLow)
            MPSGateApplication.applyTwoQubitGate(.cnot, control: i, target: i + 1, mps: &mpsHigh)
        }

        let lowError = mpsLow.truncationStatistics.cumulativeError
        let highError = mpsHigh.truncationStatistics.cumulativeError

        #expect(highError <= lowError, "Higher bond dimension should have less or equal truncation error")
    }
}

/// Tests fluent API of MPS gate application.
/// Validates convenience methods for gate application.
/// Ensures method chaining and functional style work correctly.
@Suite("MPS Fluent API")
struct MPSFluentAPITests {
    @Test("applySingleQubitGate mutating method works")
    func applySingleQubitGateMutating() {
        var mps = MatrixProductState(qubits: 2, maxBondDimension: 16)
        mps.applySingleQubitGate(.hadamard, to: 0)

        let amp0 = mps.amplitude(of: 0)
        let invSqrt2 = 1.0 / sqrt(2.0)

        #expect(abs(amp0.real - invSqrt2) < 1e-10, "Fluent applySingleQubitGate should work")
    }

    @Test("applyTwoQubitGate mutating method works")
    func applyTwoQubitGateMutating() {
        var mps = MatrixProductState(qubits: 2, maxBondDimension: 16)
        mps.applySingleQubitGate(.hadamard, to: 0)
        mps.applyTwoQubitGate(.cnot, control: 0, target: 1)

        let amp11 = mps.amplitude(of: 0b11)
        let invSqrt2 = 1.0 / sqrt(2.0)

        #expect(abs(amp11.real - invSqrt2) < 1e-10, "Fluent applyTwoQubitGate should work")
    }

    @Test("applying returns new MPS for single-qubit gate")
    func applyingSingleQubitGate() {
        let mps = MatrixProductState(qubits: 2, maxBondDimension: 16)
        let newMps = mps.applying(.hadamard, to: 0)

        let originalAmp = mps.amplitude(of: 0)
        let newAmp = newMps.amplitude(of: 0)

        #expect(abs(originalAmp.real - 1.0) < 1e-10, "Original MPS should be unchanged")
        #expect(abs(newAmp.real - 1.0 / sqrt(2.0)) < 1e-10, "New MPS should have gate applied")
    }

    @Test("applying returns new MPS for two-qubit gate")
    func applyingTwoQubitGate() {
        let mps = MatrixProductState(qubits: 2, maxBondDimension: 16)
            .applying(.hadamard, to: 0)
        let newMps = mps.applying(.cnot, to: [0, 1])

        let amp11 = newMps.amplitude(of: 0b11)
        let invSqrt2 = 1.0 / sqrt(2.0)

        #expect(abs(amp11.real - invSqrt2) < 1e-10, "applying with two-qubit gate should work")
    }

    @Test("applying with array for Toffoli")
    func applyingToffoli() {
        let mps = MatrixProductState(qubits: 3, basisState: 0b011, maxBondDimension: 32)
        let newMps = mps.applying(.toffoli, to: [0, 1, 2])

        #expect(abs(newMps.amplitude(of: 0b111).real - 1.0) < 1e-8, "applying Toffoli via array should work")
    }

    @Test("Method chaining works")
    func methodChaining() {
        let mps = MatrixProductState(qubits: 3, maxBondDimension: 16)
            .applying(.hadamard, to: 0)
            .applying(.cnot, to: [0, 1])
            .applying(.cnot, to: [1, 2])

        let amp000 = mps.amplitude(of: 0b000)
        let amp111 = mps.amplitude(of: 0b111)
        let invSqrt2 = 1.0 / sqrt(2.0)

        #expect(abs(amp000.real - invSqrt2) < 1e-10, "Method chaining should create GHZ state |000> component")
        #expect(abs(amp111.real - invSqrt2) < 1e-10, "Method chaining should create GHZ state |111> component")
    }

    @Test("Fluent API preserves normalization")
    func fluentAPIPreservesNormalization() {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 32)
            .applying(.hadamard, to: 0)
            .applying(.pauliX, to: 1)
            .applying(.cnot, to: [0, 2])
            .applying(.cz, to: [1, 3])

        #expect(mps.isNormalized(), "Fluent API chain should preserve normalization")
    }
}

/// Tests edge cases of MPS gate application.
/// Validates robustness for special scenarios and corner cases.
/// Ensures correct behavior for single qubits, large systems, and roundtrips.
@Suite("MPS Edge Cases")
struct MPSEdgeCasesTests {
    @Test("Single qubit MPS works")
    func singleQubitMPS() {
        var mps = MatrixProductState(qubits: 1, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)

        let amp0 = mps.amplitude(of: 0)
        let amp1 = mps.amplitude(of: 1)
        let invSqrt2 = 1.0 / sqrt(2.0)

        #expect(abs(amp0.real - invSqrt2) < 1e-10, "Single qubit MPS Hadamard should work")
        #expect(abs(amp1.real - invSqrt2) < 1e-10, "Single qubit MPS Hadamard should work")
    }

    @Test("Many qubits with product state")
    func manyQubitsProductState() {
        var mps = MatrixProductState(qubits: 20, maxBondDimension: 4)
        MPSGateApplication.applySingleQubitGate(.pauliX, to: 10, mps: &mps)

        let expectedState = 1 << 10
        #expect(abs(mps.amplitude(of: expectedState).real - 1.0) < 1e-10, "Large MPS single gate should work")
        #expect(mps.currentMaxBondDimension == 1, "Product state should have bond dimension 1")
    }

    @Test("Repeated gates on same qubit")
    func repeatedGatesSameQubit() {
        var mps = MatrixProductState(qubits: 3, maxBondDimension: 16)

        for _ in 0 ..< 10 {
            MPSGateApplication.applySingleQubitGate(.hadamard, to: 1, mps: &mps)
            MPSGateApplication.applySingleQubitGate(.hadamard, to: 1, mps: &mps)
        }

        let amp0 = mps.amplitude(of: 0)
        #expect(abs(amp0.real - 1.0) < 1e-9, "H^2 = I so repeated H pairs should return to |0>")
    }

    @Test("CNOT twice is identity")
    func cnotTwiceIsIdentity() {
        var mps = MatrixProductState(qubits: 2, basisState: 0b01, maxBondDimension: 16)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 1, mps: &mps)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 1, mps: &mps)

        #expect(abs(mps.amplitude(of: 0b01).real - 1.0) < 1e-10, "CNOT twice should return to original state")
    }

    @Test("Identity gate leaves state unchanged")
    func identityGate() {
        var mps = MatrixProductState(qubits: 3, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)

        let ampBefore = mps.amplitude(of: 0)
        MPSGateApplication.applySingleQubitGate(.identity, to: 0, mps: &mps)
        let ampAfter = mps.amplitude(of: 0)

        #expect(abs(ampBefore.real - ampAfter.real) < 1e-10, "Identity should not change state (real)")
        #expect(abs(ampBefore.imaginary - ampAfter.imaginary) < 1e-10, "Identity should not change state (imag)")
    }

    @Test("Conversion from QuantumState preserves amplitudes")
    func conversionFromQuantumState() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bellState = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])
        let mps = MatrixProductState(from: bellState, maxBondDimension: 16)

        #expect(abs(mps.amplitude(of: 0b00).real - invSqrt2) < 1e-10, "MPS from QuantumState should preserve |00> amplitude")
        #expect(abs(mps.amplitude(of: 0b11).real - invSqrt2) < 1e-10, "MPS from QuantumState should preserve |11> amplitude")
    }

    @Test("toQuantumState roundtrip preserves state")
    func toQuantumStateRoundtrip() {
        var mps = MatrixProductState(qubits: 3, maxBondDimension: 16)
        MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
        MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 1, mps: &mps)

        let state = mps.toQuantumState()

        for i in 0 ..< 8 {
            let mpsAmp = mps.amplitude(of: i)
            let stateAmp = state.amplitude(of: i)
            #expect(abs(mpsAmp.real - stateAmp.real) < 1e-10, "Roundtrip should preserve amplitude (real)")
            #expect(abs(mpsAmp.imaginary - stateAmp.imaginary) < 1e-10, "Roundtrip should preserve amplitude (imag)")
        }
    }
}
