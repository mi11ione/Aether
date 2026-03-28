// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Validates quantum comparator circuits that compute |a⟩|b⟩|0⟩ → |a⟩|b⟩|a<b⟩.
/// Tests correctness of subtraction-based comparison across boundary
/// cases including equal values, strict ordering, and register preservation.
@Suite("Quantum Comparator Correctness")
struct QuantumComparatorCorrectnessTests {
    @Test("1-bit: 0 < 1 = true")
    func oneBitZeroLessThanOne() {
        let result = executeComparator(bits: 1, a: 0, b: 1)
        #expect(result == 1, "0 < 1 should be true (result = 1)")
    }

    @Test("1-bit: 1 < 0 = false")
    func oneBitOneLessThanZero() {
        let result = executeComparator(bits: 1, a: 1, b: 0)
        #expect(result == 0, "1 < 0 should be false (result = 0)")
    }

    @Test("1-bit: 0 < 0 = false")
    func oneBitEqualZero() {
        let result = executeComparator(bits: 1, a: 0, b: 0)
        #expect(result == 0, "0 < 0 should be false (equal values)")
    }

    @Test("1-bit: 1 < 1 = false")
    func oneBitEqualOne() {
        let result = executeComparator(bits: 1, a: 1, b: 1)
        #expect(result == 0, "1 < 1 should be false (equal values)")
    }

    @Test("2-bit: 1 < 2 = true")
    func twoBitOneLessThanTwo() {
        let result = executeComparator(bits: 2, a: 1, b: 2)
        #expect(result == 1, "1 < 2 should be true")
    }

    @Test("2-bit: 3 < 2 = false")
    func twoBitThreeLessThanTwo() {
        let result = executeComparator(bits: 2, a: 3, b: 2)
        #expect(result == 0, "3 < 2 should be false")
    }

    @Test("2-bit: 2 < 2 = false")
    func twoBitEqual() {
        let result = executeComparator(bits: 2, a: 2, b: 2)
        #expect(result == 0, "2 < 2 should be false (equal values)")
    }

    @Test("2-bit: 0 < 3 = true")
    func twoBitZeroLessThanThree() {
        let result = executeComparator(bits: 2, a: 0, b: 3)
        #expect(result == 1, "0 < 3 should be true")
    }

    @Test("3-bit: 3 < 5 = true")
    func threeBitThreeLessThanFive() {
        let result = executeComparator(bits: 3, a: 3, b: 5)
        #expect(result == 1, "3 < 5 should be true")
    }

    @Test("3-bit: 7 < 4 = false")
    func threeBitSevenLessThanFour() {
        let result = executeComparator(bits: 3, a: 7, b: 4)
        #expect(result == 0, "7 < 4 should be false")
    }

    private func executeComparator(bits: Int, a aVal: Int, b bVal: Int) -> Int {
        let total = QuantumComparator.qubitCount(bits: bits)
        let resultQubit = 2 * bits
        var circuit = QuantumCircuit(qubits: total)
        for bit in 0 ..< bits where (aVal >> bit) & 1 == 1 {
            circuit.append(.pauliX, to: bit)
        }
        for bit in 0 ..< bits where (bVal >> bit) & 1 == 1 {
            circuit.append(.pauliX, to: bits + bit)
        }
        let cmp = QuantumComparator.circuit(bits: bits)
        for op in cmp.operations {
            circuit.append(op)
        }
        let state = circuit.execute()
        let expectedLT = aVal < bVal ? 1 : 0
        let expectedIndex = aVal | (bVal << bits) | (expectedLT << resultQubit)
        let prob = state.probability(of: expectedIndex)
        #expect(abs(prob - 1.0) < 1e-10, "Expected state \(expectedIndex) with probability 1.0, got \(prob)")
        return expectedLT
    }
}

/// Tests resource counting and convenience factory methods
/// for the quantum comparator, verifying qubit allocation
/// and delegation correctness.
@Suite("Quantum Comparator Resources")
struct QuantumComparatorResourceTests {
    @Test("Qubit count is 2n+2")
    func qubitCount() {
        let q = QuantumComparator.qubitCount(bits: 4)
        #expect(q == 10, "4-bit comparator should need 10 qubits")
    }

    @Test("Convenience method produces valid circuit")
    func convenienceComparator() {
        let circuit = QuantumCircuit.comparator(bits: 3)
        #expect(circuit.count > 0, "Comparator circuit should have gates")
        #expect(circuit.qubits == 8, "3-bit comparator should use 8 qubits")
    }

    @Test("Explicit register circuit matches convenience")
    func explicitRegisters() {
        let conv = QuantumComparator.circuit(bits: 2)
        let expl = QuantumComparator.circuit(a: [0, 1], b: [2, 3], result: 4)
        #expect(conv.count == expl.count, "Explicit and convenience circuits should have equal gate counts")
    }
}
