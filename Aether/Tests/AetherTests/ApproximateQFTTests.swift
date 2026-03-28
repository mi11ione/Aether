// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Validates truncated Quantum Fourier Transform circuits.
/// Tests error bounds, gate counts, minimum truncation selection,
/// and correctness of forward and inverse approximate QFT circuits.
@Suite("Approximate QFT Circuit Generation")
struct ApproximateQFTCircuitTests {
    @Test("Circuit with full truncation matches exact QFT gate count")
    func fullTruncationMatchesExactQFT() {
        let qubits = 4
        let exactGates = ApproximateQFT.gateCount(qubits: qubits, truncation: qubits)
        let approxGates = ApproximateQFT.gateCount(qubits: qubits, truncation: qubits + 1)
        #expect(exactGates == approxGates, "Full truncation should match exact QFT gate count")
    }

    @Test("Circuit has correct qubit count for integer parameter")
    func circuitQubitCountInteger() {
        let circuit = ApproximateQFT.circuit(qubits: 4, truncation: 3)
        #expect(circuit.qubits == 4, "Circuit should have exactly 4 qubits")
    }

    @Test("Circuit has correct qubit count for array parameter")
    func circuitQubitCountArray() {
        let circuit = ApproximateQFT.circuit(qubits: [2, 3, 4, 5], truncation: 3)
        #expect(circuit.qubits >= 6, "Circuit should span at least qubits 0-5")
    }

    @Test("Inverse circuit has same operation count as forward")
    func inverseMatchesForwardCount() {
        let forward = ApproximateQFT.circuit(qubits: 4, truncation: 3)
        let inverse = ApproximateQFT.inverseCircuit(qubits: 4, truncation: 3)
        #expect(forward.count == inverse.count, "Forward and inverse should have equal gate counts")
    }

    @Test("Gate count decreases with smaller truncation")
    func gateCountDecreasesWithTruncation() {
        let full = ApproximateQFT.gateCount(qubits: 8, truncation: 8)
        let partial = ApproximateQFT.gateCount(qubits: 8, truncation: 4)
        let minimal = ApproximateQFT.gateCount(qubits: 8, truncation: 1)
        #expect(full > partial, "Full truncation should have more gates than partial")
        #expect(partial > minimal, "Partial truncation should have more gates than minimal")
    }

    @Test("Truncation 1 produces Hadamard-only circuit plus swaps")
    func truncationOneHadamardOnly() {
        let qubits = 4
        let count = ApproximateQFT.gateCount(qubits: qubits, truncation: 1)
        let expectedHadamards = qubits
        let expectedSwaps = qubits / 2
        #expect(count == expectedHadamards + expectedSwaps, "Truncation 1 should yield only Hadamards and SWAPs")
    }

    @Test("Single qubit QFT is just a Hadamard")
    func singleQubitQFT() {
        let circuit = ApproximateQFT.circuit(qubits: 1, truncation: 1)
        #expect(circuit.count == 1, "Single qubit QFT should have exactly 1 gate")
    }
}

/// Tests error bound computation and minimum truncation selection
/// for the approximate QFT, verifying mathematical properties of
/// the truncation-fidelity tradeoff.
@Suite("Approximate QFT Error Bounds")
struct ApproximateQFTErrorTests {
    @Test("Error bound decreases with larger truncation")
    func errorDecreasesWithTruncation() {
        let e1 = ApproximateQFT.errorBound(qubits: 8, truncation: 3)
        let e2 = ApproximateQFT.errorBound(qubits: 8, truncation: 6)
        let e3 = ApproximateQFT.errorBound(qubits: 8, truncation: 9)
        #expect(e1 > e2, "Error should decrease with larger truncation")
        #expect(e2 > e3, "Error should continue decreasing")
    }

    @Test("Error bound increases with more qubits")
    func errorIncreasesWithQubits() {
        let e4 = ApproximateQFT.errorBound(qubits: 4, truncation: 5)
        let e8 = ApproximateQFT.errorBound(qubits: 8, truncation: 5)
        #expect(e8 > e4, "Error should increase with more qubits at same truncation")
    }

    @Test("Error bound is non-negative")
    func errorBoundNonNegative() {
        let error = ApproximateQFT.errorBound(qubits: 4, truncation: 3)
        #expect(error >= 0.0, "Error bound must be non-negative")
    }

    @Test("Minimum truncation achieves target fidelity")
    func minimumTruncationAchievesFidelity() {
        let qubits = 8
        let targetFidelity = 0.999
        let k = ApproximateQFT.minimumTruncation(qubits: qubits, targetFidelity: targetFidelity)
        let error = ApproximateQFT.errorBound(qubits: qubits, truncation: k)
        let achievedFidelity = 1.0 - error * error
        #expect(achievedFidelity >= targetFidelity, "Achieved fidelity \(achievedFidelity) should meet target \(targetFidelity)")
    }

    @Test("Minimum truncation increases with higher fidelity target")
    func truncationIncreasesWithFidelity() {
        let k1 = ApproximateQFT.minimumTruncation(qubits: 8, targetFidelity: 0.9)
        let k2 = ApproximateQFT.minimumTruncation(qubits: 8, targetFidelity: 0.999)
        #expect(k2 >= k1, "Higher fidelity should require equal or higher truncation")
    }

    @Test("Fidelity 1.0 returns qubit count as truncation")
    func exactFidelityReturnsFullTruncation() {
        let k = ApproximateQFT.minimumTruncation(qubits: 8, targetFidelity: 1.0)
        #expect(k == 8, "Exact fidelity should return qubits as truncation")
    }
}

/// Validates convenience factory methods on QuantumCircuit
/// for approximate QFT, ensuring delegation produces correct
/// circuits for both explicit truncation and fidelity-based selection.
@Suite("Approximate QFT QuantumCircuit Extensions")
struct ApproximateQFTExtensionTests {
    @Test("Convenience truncation method matches direct call")
    func convenienceTruncation() {
        let direct = ApproximateQFT.circuit(qubits: 4, truncation: 3)
        let convenience = QuantumCircuit.approximateQFT(qubits: 4, truncation: 3)
        #expect(direct.count == convenience.count, "Convenience method should produce same circuit")
    }

    @Test("Convenience fidelity method produces valid circuit")
    func convenienceFidelity() {
        let circuit = QuantumCircuit.approximateQFT(qubits: 4, targetFidelity: 0.99)
        #expect(circuit.count > 0, "Fidelity-based circuit should have gates")
        #expect(circuit.qubits == 4, "Circuit should have 4 qubits")
    }

    @Test("Inverse circuit with array qubits has correct qubit count")
    func inverseCircuitArrayQubits() {
        let circuit = ApproximateQFT.inverseCircuit(qubits: [2, 3, 4, 5], truncation: 2)
        #expect(circuit.qubits >= 6, "Inverse array circuit should span qubits 0-5")
        #expect(circuit.count > 0, "Inverse array circuit should have gates")
    }

    @Test("Error bound is zero when truncation equals qubit count")
    func errorBoundZeroAtFullTruncation() {
        let error = ApproximateQFT.errorBound(qubits: 4, truncation: 4)
        #expect(error == 0.0, "Error should be zero at full truncation")
    }

    @Test("Error bound is zero when truncation exceeds qubit count")
    func errorBoundZeroExceedsTruncation() {
        let error = ApproximateQFT.errorBound(qubits: 4, truncation: 10)
        #expect(error == 0.0, "Error should be zero when truncation exceeds qubits")
    }

    @Test("Minimum truncation returns 1 for low fidelity target")
    func minimumTruncationLowFidelity() {
        let k = ApproximateQFT.minimumTruncation(qubits: 2, targetFidelity: 0.01)
        #expect(k >= 1, "Minimum truncation should be at least 1")
    }
}
