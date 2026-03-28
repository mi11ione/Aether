// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Validates Cuccaro ripple-carry adder correctness by executing
/// addition circuits on computational basis states and verifying
/// the result register holds the expected sum.
@Suite("Ripple-Carry Adder Correctness")
struct RippleCarryAdderTests {
    @Test("1-bit: 0 + 0 = 0")
    func oneBitZeroPlusZero() {
        let result = executeRippleCarry(bits: 1, a: 0, b: 0)
        #expect(result.b == 0, "0 + 0 should equal 0")
        #expect(result.a == 0, "Register a should be preserved as 0")
    }

    @Test("1-bit: 1 + 0 = 1")
    func oneBitOnePlusZero() {
        let result = executeRippleCarry(bits: 1, a: 1, b: 0)
        #expect(result.b == 1, "1 + 0 should equal 1")
        #expect(result.a == 1, "Register a should be preserved as 1")
    }

    @Test("1-bit: 0 + 1 = 1")
    func oneBitZeroPlusOne() {
        let result = executeRippleCarry(bits: 1, a: 0, b: 1)
        #expect(result.b == 1, "0 + 1 should equal 1")
    }

    @Test("1-bit: 1 + 1 = 0 (mod 2)")
    func oneBitOnePlusOne() {
        let result = executeRippleCarry(bits: 1, a: 1, b: 1)
        #expect(result.b == 0, "1 + 1 should equal 0 mod 2")
    }

    @Test("2-bit: 2 + 1 = 3")
    func twoBitTwoPlusOne() {
        let result = executeRippleCarry(bits: 2, a: 2, b: 1)
        #expect(result.b == 3, "2 + 1 should equal 3")
        #expect(result.a == 2, "Register a should be preserved as 2")
    }

    @Test("2-bit: 3 + 2 = 1 (mod 4)")
    func twoBitThreePlusTwo() {
        let result = executeRippleCarry(bits: 2, a: 3, b: 2)
        #expect(result.b == 1, "3 + 2 should equal 1 mod 4")
    }

    @Test("3-bit: 5 + 2 = 7")
    func threeBitFivePlusTwo() {
        let result = executeRippleCarry(bits: 3, a: 5, b: 2)
        #expect(result.b == 7, "5 + 2 should equal 7")
        #expect(result.a == 5, "Register a should be preserved as 5")
    }

    @Test("3-bit: 4 + 3 = 7")
    func threeBitFourPlusThree() {
        let result = executeRippleCarry(bits: 3, a: 4, b: 3)
        #expect(result.b == 7, "4 + 3 should equal 7")
    }

    private func executeRippleCarry(bits: Int, a aVal: Int, b bVal: Int) -> (a: Int, b: Int) {
        let total = QuantumAdder.qubitCount(.rippleCarry, bits: bits)
        var circuit = QuantumCircuit(qubits: total)
        for bit in 0 ..< bits where (aVal >> bit) & 1 == 1 {
            circuit.append(.pauliX, to: bit)
        }
        for bit in 0 ..< bits where (bVal >> bit) & 1 == 1 {
            circuit.append(.pauliX, to: bits + bit)
        }
        let adder = QuantumAdder.circuit(.rippleCarry, bits: bits)
        for op in adder.operations {
            circuit.append(op)
        }
        let state = circuit.execute()
        let mask = (1 << bits) - 1
        let expectedA = aVal
        let expectedB = (aVal + bVal) & mask
        let expectedIndex = expectedA | (expectedB << bits)
        let prob = state.probability(of: expectedIndex)
        #expect(abs(prob - 1.0) < 1e-10, "Expected state \(expectedIndex) with probability 1.0, got \(prob)")
        return (a: expectedA, b: expectedB)
    }
}

/// Validates carry-lookahead out-of-place adder that preserves
/// both inputs and writes the sum to a separate output register
/// using prefix-tree carry computation with Bennett cleanup.
@Suite("Carry-Lookahead Adder Correctness")
struct CarryLookaheadAdderTests {
    @Test("1-bit: 1 + 1 = 0 (mod 2) with inputs preserved")
    func oneBitOnePlusOne() {
        let result = executeCarryLookahead(bits: 1, a: 1, b: 1)
        #expect(result.out == 0, "1 + 1 should equal 0 mod 2")
        #expect(result.a == 1, "Register a should be preserved")
        #expect(result.b == 1, "Register b should be preserved")
    }

    @Test("2-bit: 2 + 1 = 3 with inputs preserved")
    func twoBitTwoPlusOne() {
        let result = executeCarryLookahead(bits: 2, a: 2, b: 1)
        #expect(result.out == 3, "2 + 1 should equal 3")
        #expect(result.a == 2, "Register a should be preserved as 2")
        #expect(result.b == 1, "Register b should be preserved as 1")
    }

    @Test("2-bit: 3 + 3 = 2 (mod 4) with inputs preserved")
    func twoBitThreePlusThree() {
        let result = executeCarryLookahead(bits: 2, a: 3, b: 3)
        #expect(result.out == 2, "3 + 3 should equal 2 mod 4")
    }

    private func executeCarryLookahead(bits: Int, a aVal: Int, b bVal: Int) -> (a: Int, b: Int, out: Int) {
        let total = QuantumAdder.qubitCount(.carryLookahead, bits: bits)
        var circuit = QuantumCircuit(qubits: total)
        for bit in 0 ..< bits where (aVal >> bit) & 1 == 1 {
            circuit.append(.pauliX, to: bit)
        }
        for bit in 0 ..< bits where (bVal >> bit) & 1 == 1 {
            circuit.append(.pauliX, to: bits + bit)
        }
        let adder = QuantumAdder.circuit(.carryLookahead, bits: bits)
        for op in adder.operations {
            circuit.append(op)
        }
        let state = circuit.execute()
        let mask = (1 << bits) - 1
        let expectedOut = (aVal + bVal) & mask
        let expectedIndex = aVal | (bVal << bits) | (expectedOut << (2 * bits))
        let prob = state.probability(of: expectedIndex)
        #expect(abs(prob - 1.0) < 1e-10, "Expected state \(expectedIndex) with probability 1.0, got \(prob)")
        return (a: aVal, b: bVal, out: expectedOut)
    }
}

/// Tests qubit allocation, variant selection, and resource
/// counting methods for quantum adder circuits ensuring
/// consistent sizing across all variants.
@Suite("Quantum Adder Resource Queries")
struct QuantumAdderResourceTests {
    @Test("Ripple-carry qubit count is 2n+1")
    func rippleCarryQubitCount() {
        let q = QuantumAdder.qubitCount(.rippleCarry, bits: 4)
        #expect(q == 9, "Ripple-carry 4-bit should need 9 qubits")
    }

    @Test("Carry-lookahead qubit count is 5n")
    func carryLookaheadQubitCount() {
        let q = QuantumAdder.qubitCount(.carryLookahead, bits: 4)
        #expect(q == 20, "Carry-lookahead 4-bit should need 20 qubits")
    }

    @Test("Ripple-carry ancilla count is 1")
    func rippleCarryAncillaCount() {
        let a = QuantumAdder.ancillaCount(.rippleCarry, bits: 4)
        #expect(a == 1, "Ripple-carry should need 1 ancilla")
    }

    @Test("Carry-lookahead ancilla count is 2n")
    func carryLookaheadAncillaCount() {
        let a = QuantumAdder.ancillaCount(.carryLookahead, bits: 4)
        #expect(a == 8, "Carry-lookahead 4-bit should need 8 ancillas")
    }

    @Test("Optimal variant with sufficient ancillas selects carry-lookahead")
    func optimalVariantSufficientAncillas() {
        let v = QuantumAdder.optimalVariant(bits: 4, availableAncillas: 12)
        #expect(v == .carryLookahead, "Should select carry-lookahead with 12 ancillas for 4-bit")
    }

    @Test("Optimal variant with scarce ancillas selects ripple-carry")
    func optimalVariantScarceAncillas() {
        let v = QuantumAdder.optimalVariant(bits: 4, availableAncillas: 2)
        #expect(v == .rippleCarry, "Should select ripple-carry with only 2 ancillas")
    }

    @Test("Result qubits for ripple-carry are in b register")
    func resultQubitsRippleCarry() {
        let r = QuantumAdder.resultQubits(.rippleCarry, bits: 4)
        #expect(r == [4, 5, 6, 7], "Ripple-carry result should be in b register [4,5,6,7]")
    }

    @Test("Result qubits for carry-lookahead are in output register")
    func resultQubitsCarryLookahead() {
        let r = QuantumAdder.resultQubits(.carryLookahead, bits: 4)
        #expect(r == [8, 9, 10, 11], "Carry-lookahead result should be in output register [8,9,10,11]")
    }

    @Test("Convenience adder method produces valid circuit")
    func convenienceAdder() {
        let circuit = QuantumCircuit.adder(bits: 3, variant: .rippleCarry)
        #expect(circuit.count > 0, "Adder circuit should have gates")
        #expect(circuit.qubits == 7, "3-bit ripple-carry should use 7 qubits")
    }
}

/// Validates explicit qubit register assignment for adder circuits,
/// controlled adder correctness, and allocation helpers ensuring
/// no qubit collisions when control qubit is specified.
@Suite("Quantum Adder Controlled and Explicit Register")
struct QuantumAdderControlledTests {
    @Test("Explicit register ripple-carry: 2 + 1 = 3")
    func explicitRippleCarry() {
        let a = [0, 1, 2]
        let b = [3, 4, 5]
        let adder = QuantumAdder.circuit(.rippleCarry, a: a, b: b)
        var circuit = QuantumCircuit(qubits: adder.qubits)
        circuit.append(.pauliX, to: 1)
        circuit.append(.pauliX, to: 3)
        for op in adder.operations {
            circuit.append(op)
        }
        let state = circuit.execute()
        let expectedIndex = 2 | (3 << 3)
        let prob = state.probability(of: expectedIndex)
        #expect(abs(prob - 1.0) < 1e-10, "Explicit register 2+1 should equal 3")
    }

    @Test("Explicit register carry-lookahead: 1 + 1 = 2")
    func explicitCarryLookahead() {
        let a = [0, 1]
        let b = [2, 3]
        let adder = QuantumAdder.circuit(.carryLookahead, a: a, b: b)
        var circuit = QuantumCircuit(qubits: adder.qubits)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 2)
        for op in adder.operations {
            circuit.append(op)
        }
        let state = circuit.execute()
        let expectedIndex = 1 | (1 << 2) | (2 << 4)
        let prob = state.probability(of: expectedIndex)
        #expect(abs(prob - 1.0) < 1e-10, "Explicit register CLA 1+1 should equal 2")
    }

    @Test("Controlled ripple-carry: control=1 performs addition")
    func controlledRippleCarryActive() {
        let bits = 2
        let controlQubit = 8
        let adder = QuantumAdder.circuit(.rippleCarry, bits: bits, controlledBy: controlQubit)
        var circuit = QuantumCircuit(qubits: adder.qubits)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 2)
        circuit.append(.pauliX, to: controlQubit)
        for op in adder.operations {
            circuit.append(op)
        }
        let state = circuit.execute()
        let expectedIndex = 1 | (2 << 2) | (1 << controlQubit)
        let prob = state.probability(of: expectedIndex)
        #expect(abs(prob - 1.0) < 1e-10, "Controlled adder with control=1 should compute 1+1=2")
    }

    @Test("Controlled ripple-carry: control=0 preserves inputs")
    func controlledRippleCarryInactive() {
        let bits = 2
        let controlQubit = 8
        let adder = QuantumAdder.circuit(.rippleCarry, bits: bits, controlledBy: controlQubit)
        var circuit = QuantumCircuit(qubits: adder.qubits)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 2)
        for op in adder.operations {
            circuit.append(op)
        }
        let state = circuit.execute()
        let expectedIndex = 1 | (1 << 2)
        let prob = state.probability(of: expectedIndex)
        #expect(abs(prob - 1.0) < 1e-10, "Controlled adder with control=0 should preserve inputs")
    }

    @Test("Controlled carry-lookahead: control=1 computes sum")
    func controlledCarryLookaheadActive() {
        let bits = 2
        let controlQubit = 12
        let adder = QuantumAdder.circuit(.carryLookahead, bits: bits, controlledBy: controlQubit)
        var circuit = QuantumCircuit(qubits: adder.qubits)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 3)
        circuit.append(.pauliX, to: controlQubit)
        for op in adder.operations {
            circuit.append(op)
        }
        let state = circuit.execute()
        let expectedIndex = 1 | (2 << 2) | (3 << 4) | (1 << controlQubit)
        let prob = state.probability(of: expectedIndex)
        #expect(abs(prob - 1.0) < 1e-10, "Controlled CLA with control=1 should compute 1+2=3")
    }

    @Test("Controlled carry-lookahead: control=0 preserves inputs")
    func controlledCarryLookaheadInactive() {
        let bits = 2
        let controlQubit = 12
        let adder = QuantumAdder.circuit(.carryLookahead, bits: bits, controlledBy: controlQubit)
        var circuit = QuantumCircuit(qubits: adder.qubits)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 3)
        for op in adder.operations {
            circuit.append(op)
        }
        let state = circuit.execute()
        let expectedIndex = 1 | (2 << 2)
        let prob = state.probability(of: expectedIndex)
        #expect(abs(prob - 1.0) < 1e-10, "Controlled CLA with control=0 should preserve inputs")
    }

    @Test("Optimal variant rejects carry-lookahead when total exceeds 30 qubits")
    func optimalVariantExceedsLimit() {
        let v = QuantumAdder.optimalVariant(bits: 7, availableAncillas: 100)
        #expect(v == .rippleCarry, "Should fall back to ripple-carry when CLA exceeds 30 qubits")
    }

    @Test("Controlled 1-bit ripple-carry: control=1 performs addition")
    func controlledOneBitRippleCarry() {
        let controlQubit = 4
        let adder = QuantumAdder.circuit(.rippleCarry, bits: 1, controlledBy: controlQubit)
        var circuit = QuantumCircuit(qubits: adder.qubits)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.pauliX, to: controlQubit)
        for op in adder.operations {
            circuit.append(op)
        }
        let state = circuit.execute()
        let expectedIndex = 1 | (0 << 1) | (1 << controlQubit)
        let prob = state.probability(of: expectedIndex)
        #expect(abs(prob - 1.0) < 1e-10, "Controlled 1-bit 1+1=0 mod 2 with control=1")
    }

    @Test("Controlled 1-bit carry-lookahead: control=1 computes sum")
    func controlledOneBitCarryLookahead() {
        let controlQubit = 6
        let adder = QuantumAdder.circuit(.carryLookahead, bits: 1, controlledBy: controlQubit)
        var circuit = QuantumCircuit(qubits: adder.qubits)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.pauliX, to: controlQubit)
        for op in adder.operations {
            circuit.append(op)
        }
        let state = circuit.execute()
        let expectedIndex = 1 | (1 << 1) | (0 << 2) | (1 << controlQubit)
        let prob = state.probability(of: expectedIndex)
        #expect(abs(prob - 1.0) < 1e-10, "Controlled 1-bit CLA 1+1=0 mod 2 with control=1")
    }

    @Test("Controlled ripple-carry with control at ancilla position triggers skip")
    func controlledRippleCarrySkipControl() {
        let bits = 2
        let controlQubit = 2 * bits
        let adder = QuantumAdder.circuit(.rippleCarry, bits: bits, controlledBy: controlQubit)
        #expect(adder.qubits > 0, "Circuit with skipped control allocation should be valid")
        #expect(adder.count > 0, "Circuit should have gates")
    }

    @Test("Controlled carry-lookahead with control in output register range triggers allocation skip")
    func controlledCLAAllocationSkip() {
        let bits = 2
        let controlQubit = 2 * bits
        let adder = QuantumAdder.circuit(.carryLookahead, bits: bits, controlledBy: controlQubit)
        #expect(adder.qubits > 0, "CLA with allocation skip should produce valid circuit")
        #expect(adder.count > 0, "Circuit should have gates")
    }

    @Test("Controlled carry-lookahead with control at CCCX ancilla position triggers skip")
    func controlledCLACCCXCollision() {
        let bits = 1
        let controlQubit = 5
        let adder = QuantumAdder.circuit(.carryLookahead, bits: bits, controlledBy: controlQubit)
        #expect(adder.qubits > 0, "CLA with CCCX collision should produce valid circuit")
        #expect(adder.count > 0, "Circuit should have gates")
    }

    @Test("Controlled ripple-carry with control at CCCX ancilla position triggers skip")
    func controlledRippleCarryCCCXCollision() {
        let bits = 2
        let controlQubit = 2 * bits + 1
        let adder = QuantumAdder.circuit(.rippleCarry, bits: bits, controlledBy: controlQubit)
        #expect(adder.qubits > 0, "Ripple-carry with CCCX collision should produce valid circuit")
        #expect(adder.count > 0, "Circuit should have gates")
    }
}
