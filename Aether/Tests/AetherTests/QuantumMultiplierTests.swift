// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Validates quantum schoolbook multiplier correctness by executing
/// multiplication circuits on computational basis states and verifying
/// the 2n-bit result register holds the expected product.
@Suite("Schoolbook Multiplier Correctness")
struct SchoolbookMultiplierTests {
    @Test("1-bit: 0 × 0 = 0")
    func oneBitZeroTimesZero() {
        let result = executeMultiplier(bits: 1, a: 0, b: 0)
        #expect(result == 0, "0 × 0 should equal 0")
    }

    @Test("1-bit: 1 × 1 = 1")
    func oneBitOneTimesOne() {
        let result = executeMultiplier(bits: 1, a: 1, b: 1)
        #expect(result == 1, "1 × 1 should equal 1")
    }

    @Test("1-bit: 1 × 0 = 0")
    func oneBitOneTimesZero() {
        let result = executeMultiplier(bits: 1, a: 1, b: 0)
        #expect(result == 0, "1 × 0 should equal 0")
    }

    @Test("2-bit: 2 × 3 = 6")
    func twoBitTwoTimesThree() {
        let result = executeMultiplier(bits: 2, a: 2, b: 3)
        #expect(result == 6, "2 × 3 should equal 6")
    }

    @Test("2-bit: 3 × 3 = 9")
    func twoBitThreeTimesThree() {
        let result = executeMultiplier(bits: 2, a: 3, b: 3)
        #expect(result == 9, "3 × 3 should equal 9")
    }

    @Test("2-bit: 1 × 2 = 2")
    func twoBitOneTimesTwo() {
        let result = executeMultiplier(bits: 2, a: 1, b: 2)
        #expect(result == 2, "1 × 2 should equal 2")
    }

    @Test("3-bit: 5 × 3 = 15")
    func threeBitFiveTimesThree() {
        let result = executeMultiplier(bits: 3, a: 5, b: 3)
        #expect(result == 15, "5 × 3 should equal 15")
    }

    @Test("3-bit: 7 × 1 = 7")
    func threeBitSevenTimesOne() {
        let result = executeMultiplier(bits: 3, a: 7, b: 1)
        #expect(result == 7, "7 × 1 should equal 7")
    }

    private func executeMultiplier(bits: Int, a aVal: Int, b bVal: Int) -> Int {
        let total = QuantumMultiplier.qubitCount(.schoolbook, bits: bits)
        var circuit = QuantumCircuit(qubits: total)
        for bit in 0 ..< bits where (aVal >> bit) & 1 == 1 {
            circuit.append(.pauliX, to: bit)
        }
        for bit in 0 ..< bits where (bVal >> bit) & 1 == 1 {
            circuit.append(.pauliX, to: bits + bit)
        }
        let mult = QuantumMultiplier.circuit(.schoolbook, bits: bits)
        for op in mult.operations {
            circuit.append(op)
        }
        let state = circuit.execute()
        let expectedProduct = aVal * bVal
        let resultMask = (1 << (2 * bits)) - 1
        let expectedIndex = aVal | (bVal << bits) | ((expectedProduct & resultMask) << (2 * bits))
        let prob = state.probability(of: expectedIndex)
        #expect(abs(prob - 1.0) < 1e-10, "Expected state \(expectedIndex) with probability 1.0, got \(prob)")
        return expectedProduct
    }
}

/// Tests resource counting, variant selection, and convenience
/// factory methods for the quantum multiplier ensuring consistent
/// qubit allocation and proper selection logic.
@Suite("Quantum Multiplier Resources")
struct QuantumMultiplierResourceTests {
    @Test("Schoolbook qubit count is 4n+1")
    func schoolbookQubitCount() {
        let q = QuantumMultiplier.qubitCount(.schoolbook, bits: 4)
        #expect(q == 18, "Schoolbook 4-bit should need 18 qubits")
    }

    @Test("Optimal variant selects schoolbook for small n")
    func optimalVariantSmall() {
        let v = QuantumMultiplier.optimalVariant(bits: 4)
        #expect(v == .schoolbook, "Should select schoolbook for 4-bit operands")
    }

    @Test("Optimal variant selects schoolbook at crossover")
    func optimalVariantCrossover() {
        let v32 = QuantumMultiplier.optimalVariant(bits: 32)
        let v33 = QuantumMultiplier.optimalVariant(bits: 33)
        #expect(v32 == .schoolbook, "Should select schoolbook at n=32")
        #expect(v33 == .karatsuba, "Should select karatsuba at n=33")
    }

    @Test("Convenience multiplier method produces valid circuit")
    func convenienceMultiplier() {
        let circuit = QuantumCircuit.multiplier(bits: 2, variant: .schoolbook)
        #expect(circuit.count > 0, "Multiplier circuit should have gates")
        #expect(circuit.qubits == 10, "2-bit schoolbook should use 10 qubits")
    }

    @Test("Explicit register circuit matches convenience")
    func explicitRegisters() {
        let conv = QuantumMultiplier.circuit(.schoolbook, bits: 2)
        let expl = QuantumMultiplier.circuit(.schoolbook, a: [0, 1], b: [2, 3], result: Array(4 ..< 8))
        #expect(conv.count == expl.count, "Explicit and convenience circuits should have equal gate counts")
    }

    @Test("Karatsuba variant delegates to schoolbook for small n")
    func karatsubaFallback() {
        let schoolbook = QuantumMultiplier.circuit(.schoolbook, bits: 2)
        let karatsuba = QuantumMultiplier.circuit(.karatsuba, bits: 2)
        #expect(schoolbook.count == karatsuba.count, "Karatsuba should delegate to schoolbook for 2-bit")
    }

    @Test("Karatsuba qubit count matches schoolbook")
    func karatsubaQubitCount() {
        let sq = QuantumMultiplier.qubitCount(.schoolbook, bits: 4)
        let kq = QuantumMultiplier.qubitCount(.karatsuba, bits: 4)
        #expect(sq == kq, "Karatsuba and schoolbook should have same qubit count for small n")
    }
}
