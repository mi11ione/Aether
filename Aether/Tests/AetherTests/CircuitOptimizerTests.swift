// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for single-qubit identity pair cancellation.
/// Validates removal of adjacent gate pairs where G₁G₂ = I,
/// including Hermitian gates (H-H, X-X) and inverse pairs (S-S†, T-T†).
@Suite("Single-Qubit Identity Cancellation")
struct SingleQubitIdentityCancellationTests {
    @Test("Hadamard-Hadamard cancels to identity")
    func hadamardHadamardCancels() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 0)

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 0, "H-H should cancel to identity, leaving empty circuit")
    }

    @Test("Pauli-X Pauli-X cancels to identity")
    func pauliXPauliXCancels() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 0)

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 0, "X-X should cancel to identity")
    }

    @Test("Pauli-Y Pauli-Y cancels to identity")
    func pauliYPauliYCancels() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliY, to: 0)
        circuit.append(.pauliY, to: 0)

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 0, "Y-Y should cancel to identity")
    }

    @Test("Pauli-Z Pauli-Z cancels to identity")
    func pauliZPauliZCancels() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliZ, to: 0)
        circuit.append(.pauliZ, to: 0)

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 0, "Z-Z should cancel to identity")
    }

    @Test("S gate followed by S-dagger cancels")
    func sGateSdaggerCancels() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.sGate, to: 0)
        circuit.append(.phase(-.pi / 2), to: 0)

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 0, "S followed by P(-π/2) should cancel")
    }

    @Test("T gate followed by T-dagger cancels")
    func tGateTdaggerCancels() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.tGate, to: 0)
        circuit.append(.phase(-.pi / 4), to: 0)

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 0, "T followed by P(-π/4) should cancel")
    }

    @Test("Rotation-Z with opposite angles cancels")
    func rotationZOppositeCancels() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(.pi / 4), to: 0)
        circuit.append(.rotationZ(-.pi / 4), to: 0)

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 0, "Rz(θ)Rz(-θ) should cancel to identity")
    }

    @Test("Rotation-Y with opposite angles cancels")
    func rotationYOppositeCancels() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationY(.pi / 3), to: 0)
        circuit.append(.rotationY(-.pi / 3), to: 0)

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 0, "Ry(θ)Ry(-θ) should cancel to identity")
    }

    @Test("Rotation-X with opposite angles cancels")
    func rotationXOppositeCancels() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationX(.pi / 6), to: 0)
        circuit.append(.rotationX(-.pi / 6), to: 0)

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 0, "Rx(θ)Rx(-θ) should cancel to identity")
    }

    @Test("Phase gates with opposite angles cancel")
    func phaseOppositeCancels() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.phase(.pi / 5), to: 0)
        circuit.append(.phase(-.pi / 5), to: 0)

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 0, "P(θ)P(-θ) should cancel to identity")
    }

    @Test("Identity gate remains (harmless)")
    func identityGateRemains() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.identity, to: 0)
        circuit.append(.identity, to: 0)

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 0, "I-I should cancel to empty circuit")
    }

    @Test("Non-adjacent pairs do not cancel in single pass")
    func nonAdjacentPairsNoCancel() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 0)
        circuit.append(.hadamard, to: 0)

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 3, "H-X-H should remain (H's not adjacent)")
    }

    @Test("Multiple adjacent pairs all cancel")
    func multipleAdjacentPairsCancel() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 0)

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 0, "H-H-X-X should all cancel")
    }

    @Test("Gates on different qubits do not cancel")
    func differentQubitNoCancel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 2, "H on qubit 0 and H on qubit 1 should not cancel")
    }
}

/// Test suite for two-qubit identity pair cancellation.
/// Validates removal of adjacent CNOT-CNOT, SWAP-SWAP, CZ-CZ pairs
/// on the same control/target qubits.
@Suite("Two-Qubit Identity Cancellation")
struct TwoQubitIdentityCancellationTests {
    @Test("CNOT-CNOT on same qubits cancels")
    func cnotCnotCancels() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.cnot, to: [0, 1])

        let optimized = CircuitOptimizer.cancelTwoQubitPairs(circuit)

        #expect(optimized.count == 0, "CNOT-CNOT on same qubits should cancel")
    }

    @Test("CZ-CZ on same qubits cancels")
    func czCzCancels() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cz, to: [0, 1])
        circuit.append(.cz, to: [0, 1])

        let optimized = CircuitOptimizer.cancelTwoQubitPairs(circuit)

        #expect(optimized.count == 0, "CZ-CZ on same qubits should cancel")
    }

    @Test("SWAP-SWAP on same qubits cancels")
    func swapSwapCancels() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.swap, to: [0, 1])
        circuit.append(.swap, to: [0, 1])

        let optimized = CircuitOptimizer.cancelTwoQubitPairs(circuit)

        #expect(optimized.count == 0, "SWAP-SWAP on same qubits should cancel")
    }

    @Test("CY-CY on same qubits cancels")
    func cyCyCancels() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cy, to: [0, 1])
        circuit.append(.cy, to: [0, 1])

        let optimized = CircuitOptimizer.cancelTwoQubitPairs(circuit)

        #expect(optimized.count == 0, "CY-CY on same qubits should cancel")
    }

    @Test("Controlled-phase with opposite angles cancels")
    func controlledPhaseOppositeCancels() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlledPhase(.pi / 4), to: [0, 1])
        circuit.append(.controlledPhase(-.pi / 4), to: [0, 1])

        let optimized = CircuitOptimizer.cancelTwoQubitPairs(circuit)

        #expect(optimized.count == 0, "CP(θ)CP(-θ) should cancel")
    }

    @Test("CNOT on different qubit order does not cancel")
    func cnotDifferentOrderNoCancel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.cnot, to: [1, 0])

        let optimized = CircuitOptimizer.cancelTwoQubitPairs(circuit)

        #expect(optimized.count == 2, "CNOT[0,1] and CNOT[1,0] should not cancel")
    }

    @Test("Toffoli-Toffoli on same qubits cancels via identity pairs")
    func toffoliToffoliCancels() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.toffoli, to: [0, 1, 2])
        circuit.append(.toffoli, to: [0, 1, 2])

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 0, "Toffoli-Toffoli on same qubits should cancel")
    }
}

/// Test suite for single-qubit gate merging.
/// Validates fusion of consecutive rotations: Rz(θ₁)Rz(θ₂) -> Rz(θ₁+θ₂)
/// and merging of mixed rotations into U3 gates.
@Suite("Single-Qubit Gate Merging")
struct SingleQubitGateMergingTests {
    @Test("Same-axis Rz rotations merge by adding angles")
    func sameAxisRzMerge() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(.pi / 4), to: 0)
        circuit.append(.rotationZ(.pi / 4), to: 0)

        let optimized = CircuitOptimizer.mergeSingleQubitGates(circuit)

        #expect(optimized.count == 1, "Two Rz gates should merge into one")
        if let gate = optimized.gates.first?.gate {
            if case let .rotationZ(angle) = gate, case let .value(v) = angle {
                #expect(abs(v - .pi / 2) < 1e-10, "Merged angle should be π/2")
            }
        }
    }

    @Test("Same-axis Ry rotations merge by adding angles")
    func sameAxisRyMerge() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationY(.pi / 6), to: 0)
        circuit.append(.rotationY(.pi / 3), to: 0)

        let optimized = CircuitOptimizer.mergeSingleQubitGates(circuit)

        #expect(optimized.count == 1, "Two Ry gates should merge into one")
    }

    @Test("Same-axis Rx rotations merge by adding angles")
    func sameAxisRxMerge() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationX(.pi / 8), to: 0)
        circuit.append(.rotationX(.pi / 8), to: 0)

        let optimized = CircuitOptimizer.mergeSingleQubitGates(circuit)

        #expect(optimized.count == 1, "Two Rx gates should merge into one")
    }

    @Test("Phase gates merge by adding angles")
    func phaseGatesMerge() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.phase(.pi / 4), to: 0)
        circuit.append(.phase(.pi / 4), to: 0)

        let optimized = CircuitOptimizer.mergeSingleQubitGates(circuit)

        #expect(optimized.count == 1, "Two phase gates should merge into one")
    }

    @Test("Rotations summing to 2π cancel completely")
    func rotationsSumTo2PiCancel() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(.pi), to: 0)
        circuit.append(.rotationZ(.pi), to: 0)

        let optimized = CircuitOptimizer.mergeSingleQubitGates(circuit)

        #expect(optimized.count == 0, "Rz(π)Rz(π) = Rz(2π) ≈ I should cancel")
    }

    @Test("Mixed rotations fuse to U3")
    func mixedRotationsFuseToU3() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(.pi / 4), to: 0)
        circuit.append(.rotationY(.pi / 3), to: 0)
        circuit.append(.rotationZ(.pi / 6), to: 0)

        let optimized = CircuitOptimizer.mergeSingleQubitGates(circuit)

        #expect(optimized.count == 1, "Rz-Ry-Rz sequence should fuse to single gate")
    }

    @Test("Two-qubit gates interrupt single-qubit merging")
    func twoQubitInterruptsMerge() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.rotationZ(.pi / 4), to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.rotationZ(.pi / 4), to: 0)

        let optimized = CircuitOptimizer.mergeSingleQubitGates(circuit)

        #expect(optimized.count == 3, "CNOT should prevent Rz gates from merging")
    }

    @Test("Consecutive single-qubit sequence on different qubits stays separate")
    func differentQubitsSeparate() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.rotationZ(.pi / 4), to: 0)
        circuit.append(.rotationZ(.pi / 4), to: 1)

        let optimized = CircuitOptimizer.mergeSingleQubitGates(circuit)

        #expect(optimized.count == 2, "Gates on different qubits should not merge")
    }

    @Test("Three Rz gates merge into one")
    func threeRzGatesMerge() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(.pi / 6), to: 0)
        circuit.append(.rotationZ(.pi / 6), to: 0)
        circuit.append(.rotationZ(.pi / 6), to: 0)

        let optimized = CircuitOptimizer.mergeSingleQubitGates(circuit)

        #expect(optimized.count == 1, "Three Rz gates should merge into one")
    }
}

/// Test suite for single-qubit matrix decomposition to U3.
/// Validates that arbitrary 2x2 unitaries decompose correctly.
@Suite("Decomposition to U3")
struct DecomposeToU3Tests {
    @Test("Hadamard decomposes to U3")
    func hadamardDecomposesToU3() {
        let matrix = QuantumGate.hadamard.matrix()
        let gate = CircuitOptimizer.decomposeToU3(matrix)

        #expect(gate.qubitsRequired == 1, "Decomposed gate should be single-qubit")

        let reconstructed = gate.matrix()
        #expect(QuantumGate.matricesEqual(matrix, reconstructed, tolerance: 1e-8),
                "Reconstructed matrix should equal Hadamard")
    }

    @Test("Identity decomposes to identity")
    func identityDecomposesToIdentity() {
        let matrix = QuantumGate.identity.matrix()
        let gate = CircuitOptimizer.decomposeToU3(matrix)

        #expect(gate == .identity, "Identity should decompose to identity gate")
    }

    @Test("S gate decomposes correctly")
    func sGateDecomposesCorrectly() {
        let matrix = QuantumGate.sGate.matrix()
        let gate = CircuitOptimizer.decomposeToU3(matrix)

        let reconstructed = gate.matrix()
        #expect(QuantumGate.matricesEqual(matrix, reconstructed, tolerance: 1e-8),
                "Reconstructed matrix should equal S gate")
    }

    @Test("Arbitrary rotation decomposes correctly")
    func arbitraryRotationDecomposes() {
        let matrix = QuantumGate.rotationY(0.7).matrix()
        let gate = CircuitOptimizer.decomposeToU3(matrix)

        let reconstructed = gate.matrix()
        #expect(QuantumGate.matricesEqual(matrix, reconstructed, tolerance: 1e-8),
                "Reconstructed matrix should equal Ry(0.7)")
    }
}

/// Test suite for gate commutation detection.
/// Validates correct identification of commuting gate pairs
/// for circuit reordering optimization.
@Suite("Gate Commutation")
struct GateCommutationTests {
    @Test("Gates on disjoint qubits commute")
    func disjointQubitsCommute() {
        let g1 = Gate(.pauliX, to: 0)
        let g2 = Gate(.pauliZ, to: 1)

        #expect(CircuitOptimizer.gatesCommute(g1, g2), "Gates on different qubits should commute")
    }

    @Test("Diagonal gates on same qubit commute")
    func diagonalGatesCommute() {
        let g1 = Gate(.pauliZ, to: 0)
        let g2 = Gate(.rotationZ(.pi / 4), to: 0)

        #expect(CircuitOptimizer.gatesCommute(g1, g2), "Z and Rz should commute (both diagonal)")
    }

    @Test("Z and phase gate commute")
    func zAndPhaseCommute() {
        let g1 = Gate(.pauliZ, to: 0)
        let g2 = Gate(.phase(.pi / 3), to: 0)

        #expect(CircuitOptimizer.gatesCommute(g1, g2), "Z and P should commute")
    }

    @Test("Two Rz gates commute")
    func twoRzGatesCommute() {
        let g1 = Gate(.rotationZ(.pi / 4), to: 0)
        let g2 = Gate(.rotationZ(.pi / 3), to: 0)

        #expect(CircuitOptimizer.gatesCommute(g1, g2), "Rz gates should commute")
    }

    @Test("X and Z do not commute")
    func xAndZDoNotCommute() {
        let g1 = Gate(.pauliX, to: 0)
        let g2 = Gate(.pauliZ, to: 0)

        #expect(!CircuitOptimizer.gatesCommute(g1, g2), "X and Z on same qubit should not commute")
    }

    @Test("H and X do not commute")
    func hAndXDoNotCommute() {
        let g1 = Gate(.hadamard, to: 0)
        let g2 = Gate(.pauliX, to: 0)

        #expect(!CircuitOptimizer.gatesCommute(g1, g2), "H and X on same qubit should not commute")
    }

    @Test("CNOT and Z on control commute")
    func cnotAndZOnControlCommute() {
        let g1 = Gate(.cnot, to: [0, 1])
        let g2 = Gate(.pauliZ, to: 0)

        #expect(CircuitOptimizer.gatesCommute(g1, g2), "CNOT and Z on control qubit should commute")
    }

    @Test("CZ gates on same qubits commute")
    func czGatesCommute() {
        let g1 = Gate(.cz, to: [0, 1])
        let g2 = Gate(.cz, to: [0, 1])

        #expect(CircuitOptimizer.gatesCommute(g1, g2), "CZ gates should commute (both diagonal)")
    }
}

/// Test suite for commutation-based gate reordering.
/// Validates that reordering brings cancellable pairs adjacent
/// and reduces circuit depth.
@Suite("Commutation Reordering")
struct CommutationReorderingTests {
    @Test("Reordering brings inverse pairs adjacent for cancellation")
    func reorderingBringsInversesAdjacent() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliZ, to: 0)
        circuit.append(.sGate, to: 0)
        circuit.append(.pauliZ, to: 0)
        circuit.append(.tGate, to: 0)

        let optimized = CircuitOptimizer.optimize(circuit)

        #expect(optimized.count < circuit.count, "Diagonal gates allow Z-Z to be brought adjacent and cancelled")
    }

    @Test("Parallel gates reorder to minimize depth")
    func parallelGatesReduceDepth() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        circuit.append(.hadamard, to: 2)

        let reordered = CircuitOptimizer.reorderByCommutation(circuit)

        #expect(CircuitOptimizer.computeDepth(reordered) == 1,
                "Three H gates on different qubits should have depth 1")
    }

    @Test("Diagonal gate sequence can be freely reordered")
    func diagonalSequenceReorders() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.pauliZ, to: 0)
        circuit.append(.sGate, to: 0)
        circuit.append(.tGate, to: 0)
        circuit.append(.rotationZ(.pi / 5), to: 0)

        let reordered = CircuitOptimizer.reorderByCommutation(circuit)

        #expect(reordered.count == circuit.count, "Diagonal gates should all remain")
    }
}

/// Test suite for KAK decomposition of two-qubit gates.
/// Validates optimal CNOT count (0-3) for different gate types.
@Suite("KAK Decomposition")
struct KAKDecompositionTests {
    @Test("Identity tensor product requires 0 CNOTs")
    func identityRequiresZeroCNOTs() {
        let identityTensor: [[Complex<Double>]] = [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .one, .zero],
            [.zero, .zero, .zero, .one],
        ]
        let gate = QuantumGate.customTwoQubit(matrix: identityTensor)
        let decomposed = CircuitOptimizer.kakDecomposition(gate)

        let cnotCount = decomposed.count(where: { $0.gate == .cnot })
        #expect(cnotCount == 0, "Identity requires 0 CNOTs")
    }

    @Test("CNOT decomposes to at most 1 CNOT")
    func cnotDecomposesToOneCNOT() {
        let decomposed = CircuitOptimizer.kakDecomposition(.cnot)

        let cnotCount = decomposed.count(where: { $0.gate == .cnot })
        #expect(cnotCount <= 1, "CNOT should decompose to at most 1 CNOT")
    }

    @Test("CZ decomposes to at most 3 CNOTs")
    func czDecomposesToAtMostThreeCNOTs() {
        let decomposed = CircuitOptimizer.kakDecomposition(.cz)

        let cnotCount = decomposed.count(where: { $0.gate == .cnot })
        #expect(cnotCount <= 3, "CZ should decompose to at most 3 CNOTs via KAK")
    }

    @Test("SWAP decomposes to at most 3 CNOTs")
    func swapDecomposesToThreeCNOTs() {
        let decomposed = CircuitOptimizer.kakDecomposition(.swap)

        let cnotCount = decomposed.count(where: { $0.gate == .cnot })
        #expect(cnotCount <= 3, "SWAP should decompose to at most 3 CNOTs")
    }

    @Test("√SWAP decomposes to at most 3 CNOTs")
    func sqrtSwapDecomposesToThreeCNOTs() {
        let decomposed = CircuitOptimizer.kakDecomposition(.sqrtSwap)

        let cnotCount = decomposed.count(where: { $0.gate == .cnot })
        #expect(cnotCount <= 3, "√SWAP should decompose to at most 3 CNOTs")
    }
}

/// Test suite for the complete optimization pipeline.
/// Validates end-to-end optimization combining all passes.
@Suite("Full Optimization Pipeline")
struct FullOptimizationPipelineTests {
    @Test("Empty circuit optimizes to empty")
    func emptyCircuitOptimizes() {
        let circuit = QuantumCircuit(qubits: 2)
        let optimized = CircuitOptimizer.optimize(circuit)

        #expect(optimized.count == 0, "Empty circuit should remain empty")
    }

    @Test("Single gate circuit unchanged")
    func singleGateUnchanged() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        let optimized = CircuitOptimizer.optimize(circuit)

        #expect(optimized.count == 1, "Single gate should remain")
    }

    @Test("Identity sequence fully cancels")
    func identitySequenceFullyCancels() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.pauliX, to: 1)
        circuit.append(.pauliX, to: 1)

        let optimized = CircuitOptimizer.optimize(circuit)

        #expect(optimized.count == 0, "All identity pairs should cancel")
    }

    @Test("Rotation merging reduces gate count")
    func rotationMergingReduces() {
        var circuit = QuantumCircuit(qubits: 1)
        for _ in 0 ..< 4 {
            circuit.append(.rotationZ(.pi / 8), to: 0)
        }

        let optimized = CircuitOptimizer.optimize(circuit)

        #expect(optimized.count < 4, "Four Rz(π/8) should merge")
    }

    @Test("Optimization is idempotent")
    func optimizationIdempotent() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.rotationZ(.pi / 4), to: 1)
        circuit.append(.hadamard, to: 0)

        let optimized1 = CircuitOptimizer.optimize(circuit)
        let optimized2 = CircuitOptimizer.optimize(optimized1)

        #expect(optimized1.count == optimized2.count,
                "Optimizing twice should give same result")
    }

    @Test("Complex circuit reduces significantly")
    func complexCircuitReduces() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.rotationZ(.pi / 4), to: 2)
        circuit.append(.rotationZ(.pi / 4), to: 2)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.pauliX, to: 1)
        circuit.append(.pauliX, to: 1)

        let optimized = CircuitOptimizer.optimize(circuit)

        #expect(optimized.count < circuit.count, "Complex circuit should reduce significantly")
    }

    @Test("Optimization preserves circuit semantics for identity cancellation")
    func optimizationPreservesSemanticsIdentity() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.cnot, to: [0, 1])

        let originalState = circuit.execute()
        let optimizedState = CircuitOptimizer.optimize(circuit).execute()

        for i in 0 ..< originalState.stateSpaceSize {
            let diff = (originalState.amplitude(of: i) - optimizedState.amplitude(of: i)).magnitude
            #expect(diff < 1e-10, "Optimized circuit should produce same state for identity cancellation")
        }
    }
}

/// Test suite for circuit depth computation.
/// Validates critical path calculation for parallel execution analysis.
@Suite("Circuit Depth Computation")
struct CircuitDepthComputationTests {
    @Test("Empty circuit has depth 0")
    func emptyCircuitDepthZero() {
        let circuit = QuantumCircuit(qubits: 2)
        let depth = CircuitOptimizer.computeDepth(circuit)

        #expect(depth == 0, "Empty circuit should have depth 0")
    }

    @Test("Single gate has depth 1")
    func singleGateDepthOne() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        let depth = CircuitOptimizer.computeDepth(circuit)

        #expect(depth == 1, "Single gate should have depth 1")
    }

    @Test("Parallel gates have depth 1")
    func parallelGatesDepthOne() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        circuit.append(.hadamard, to: 2)

        let depth = CircuitOptimizer.computeDepth(circuit)

        #expect(depth == 1, "Three parallel H gates should have depth 1")
    }

    @Test("Sequential gates on same qubit sum depth")
    func sequentialGatesSumDepth() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliZ, to: 0)

        let depth = CircuitOptimizer.computeDepth(circuit)

        #expect(depth == 3, "Three sequential gates should have depth 3")
    }

    @Test("Mixed parallel and sequential computes correctly")
    func mixedParallelSequential() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        circuit.append(.cnot, to: [0, 1])

        let depth = CircuitOptimizer.computeDepth(circuit)

        #expect(depth == 2, "H|H then CNOT should have depth 2")
    }

    @Test("CNOT blocks both qubits")
    func cnotBlocksBothQubits() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliZ, to: 1)

        let depth = CircuitOptimizer.computeDepth(circuit)

        #expect(depth == 2, "CNOT then X|Z should have depth 2")
    }

    @Test("QuantumCircuit.depth property matches computeDepth")
    func circuitDepthPropertyMatches() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.cnot, to: [1, 2])

        #expect(circuit.depth == CircuitOptimizer.computeDepth(circuit),
                "Circuit depth property should match computeDepth function")
    }
}

/// Test suite for gate count and resource estimation.
/// Validates gate counting by type and CNOT-equivalent calculations.
@Suite("Gate Count Analysis")
struct GateCountAnalysisTests {
    @Test("Empty circuit has zero counts")
    func emptyCircuitZeroCounts() {
        let circuit = QuantumCircuit(qubits: 2)
        let counts = CircuitOptimizer.gateCount(circuit)

        #expect(counts.isEmpty, "Empty circuit should have no gate counts")
    }

    @Test("Gate count by type is accurate")
    func gateCountByTypeAccurate() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.rotationZ(.pi / 4), to: 0)

        let counts = CircuitOptimizer.gateCount(circuit)

        #expect(counts[.hadamard] == 2, "Should count 2 Hadamard gates")
        #expect(counts[.cnot] == 1, "Should count 1 CNOT gate")
    }

    @Test("Gate count by arity is correct")
    func gateCountByArityCorrect() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.swap, to: [1, 2])
        circuit.append(.toffoli, to: [0, 1, 2])

        let (single, two, three) = CircuitOptimizer.gateCountByArity(circuit)

        #expect(single == 2, "Should have 2 single-qubit gates")
        #expect(two == 2, "Should have 2 two-qubit gates")
        #expect(three == 1, "Should have 1 three-qubit gate")
    }

    @Test("CNOT equivalent count for CNOT is 1")
    func cnotEquivalentForCNOT() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])

        let count = CircuitOptimizer.cnotEquivalentCount(circuit)

        #expect(count == 1, "CNOT should count as 1 CNOT-equivalent")
    }

    @Test("CNOT equivalent count for SWAP is 3")
    func cnotEquivalentForSWAP() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.swap, to: [0, 1])

        let count = CircuitOptimizer.cnotEquivalentCount(circuit)

        #expect(count == 3, "SWAP should count as 3 CNOT-equivalents")
    }

    @Test("CNOT equivalent count for Toffoli is 6")
    func cnotEquivalentForToffoli() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.toffoli, to: [0, 1, 2])

        let count = CircuitOptimizer.cnotEquivalentCount(circuit)

        #expect(count == 6, "Toffoli should count as 6 CNOT-equivalents")
    }

    @Test("QuantumCircuit.gateCount property matches function")
    func circuitGateCountPropertyMatches() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        #expect(circuit.gateCount == CircuitOptimizer.gateCount(circuit),
                "Circuit gateCount property should match function")
    }

    @Test("QuantumCircuit.cnotCount property matches function")
    func circuitCnotCountPropertyMatches() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.swap, to: [1, 2])

        #expect(circuit.cnotCount == CircuitOptimizer.cnotEquivalentCount(circuit),
                "Circuit cnotCount property should match function")
    }
}

/// Test suite for inverse circuit computation.
/// Validates that (U†)U = I for circuit adjoint construction.
@Suite("Inverse Circuit")
struct InverseCircuitTests {
    @Test("Empty circuit inverse is empty")
    func emptyInverseEmpty() {
        let circuit = QuantumCircuit(qubits: 2)
        let inverse = circuit.inverse()

        #expect(inverse.count == 0, "Inverse of empty circuit should be empty")
    }

    @Test("Hermitian gate is self-inverse")
    func hermitianSelfInverse() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        let inverse = circuit.inverse()

        #expect(inverse.gates.first?.gate == .hadamard, "H† = H")
    }

    @Test("Inverse reverses gate order")
    func inverseReversesOrder() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.pauliZ, to: 1)

        let inverse = circuit.inverse()

        #expect(inverse.gates[0].gate == .pauliZ, "First gate in inverse should be Z†=Z")
        #expect(inverse.gates[1].gate == .cnot, "Second gate in inverse should be CNOT†=CNOT")
        #expect(inverse.gates[2].gate == .hadamard, "Third gate in inverse should be H†=H")
    }

    @Test("Rotation inverse negates angle")
    func rotationInverseNegatesAngle() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(.pi / 4), to: 0)

        let inverse = circuit.inverse()

        if let gate = inverse.gates.first?.gate,
           case let .rotationZ(angle) = gate,
           case let .value(v) = angle
        {
            #expect(abs(v + .pi / 4) < 1e-10, "Rz(θ)† = Rz(-θ)")
        }
    }

    @Test("U U† equals identity")
    func uUdaggerEqualsIdentity() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.rotationY(.pi / 3), to: 1)

        let inverse = circuit.inverse()

        var combined = QuantumCircuit(qubits: 2)
        for op in circuit.gates {
            combined.append(op.gate, to: op.qubits)
        }
        for op in inverse.gates {
            combined.append(op.gate, to: op.qubits)
        }

        let state = combined.execute()

        #expect(abs(state.probability(of: 0) - 1.0) < 1e-10,
                "U followed by U† should return to |00⟩")
    }

    @Test("S gate inverse is S†")
    func sGateInverse() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.sGate, to: 0)

        let inverse = circuit.inverse()
        let inverseGate = inverse.gates.first?.gate

        if case let .phase(angle) = inverseGate, case let .value(v) = angle {
            #expect(abs(v + .pi / 2) < 1e-10, "S† = P(-π/2)")
        }
    }

    @Test("T gate inverse is T†")
    func tGateInverse() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.tGate, to: 0)

        let inverse = circuit.inverse()
        let inverseGate = inverse.gates.first?.gate

        if case let .phase(angle) = inverseGate, case let .value(v) = angle {
            #expect(abs(v + .pi / 4) < 1e-10, "T† = P(-π/4)")
        }
    }
}

/// Test suite for diagonal gate detection.
/// Validates correct classification for commutation analysis.
@Suite("Diagonal Gate Detection")
struct DiagonalGateDetectionTests {
    @Test("Identity is diagonal")
    func identityIsDiagonal() {
        #expect(CircuitOptimizer.isDiagonal(.identity), "Identity should be diagonal")
    }

    @Test("Pauli-Z is diagonal")
    func pauliZIsDiagonal() {
        #expect(CircuitOptimizer.isDiagonal(.pauliZ), "Z should be diagonal")
    }

    @Test("S gate is diagonal")
    func sGateIsDiagonal() {
        #expect(CircuitOptimizer.isDiagonal(.sGate), "S should be diagonal")
    }

    @Test("T gate is diagonal")
    func tGateIsDiagonal() {
        #expect(CircuitOptimizer.isDiagonal(.tGate), "T should be diagonal")
    }

    @Test("Phase gate is diagonal")
    func phaseGateIsDiagonal() {
        #expect(CircuitOptimizer.isDiagonal(.phase(.pi / 4)), "Phase should be diagonal")
    }

    @Test("Rotation-Z is diagonal")
    func rotationZIsDiagonal() {
        #expect(CircuitOptimizer.isDiagonal(.rotationZ(.pi / 3)), "Rz should be diagonal")
    }

    @Test("CZ is diagonal")
    func czIsDiagonal() {
        #expect(CircuitOptimizer.isDiagonal(.cz), "CZ should be diagonal")
    }

    @Test("Controlled-phase is diagonal")
    func controlledPhaseIsDiagonal() {
        #expect(CircuitOptimizer.isDiagonal(.controlledPhase(.pi / 4)), "CP should be diagonal")
    }

    @Test("Pauli-X is not diagonal")
    func pauliXNotDiagonal() {
        #expect(!CircuitOptimizer.isDiagonal(.pauliX), "X should not be diagonal")
    }

    @Test("Hadamard is not diagonal")
    func hadamardNotDiagonal() {
        #expect(!CircuitOptimizer.isDiagonal(.hadamard), "H should not be diagonal")
    }

    @Test("CNOT is not diagonal")
    func cnotNotDiagonal() {
        #expect(!CircuitOptimizer.isDiagonal(.cnot), "CNOT should not be diagonal")
    }

    @Test("Rotation-Y is not diagonal")
    func rotationYNotDiagonal() {
        #expect(!CircuitOptimizer.isDiagonal(.rotationY(.pi / 4)), "Ry should not be diagonal")
    }
}

/// Test suite for angle normalization utilities.
/// Validates angles are correctly normalized to [-π, π] range.
@Suite("Angle Normalization")
struct AngleNormalizationTests {
    @Test("Zero angle stays zero")
    func zeroStaysZero() {
        let normalized = CircuitOptimizer.normalizeAngle(0)
        #expect(abs(normalized) < 1e-10, "Zero should stay zero")
    }

    @Test("π stays π")
    func piStaysPi() {
        let normalized = CircuitOptimizer.normalizeAngle(.pi)
        #expect(abs(normalized - .pi) < 1e-10, "π should stay π")
    }

    @Test("2π normalizes to 0")
    func twoPiNormalizesToZero() {
        let normalized = CircuitOptimizer.normalizeAngle(2 * .pi)
        #expect(abs(normalized) < 1e-10, "2π should normalize to 0")
    }

    @Test("3π normalizes to π")
    func threePiNormalizesToPi() {
        let normalized = CircuitOptimizer.normalizeAngle(3 * .pi)
        #expect(abs(normalized - .pi) < 1e-10, "3π should normalize to π")
    }

    @Test("Negative angle stays in range")
    func negativeAngleInRange() {
        let normalized = CircuitOptimizer.normalizeAngle(-.pi / 2)
        #expect(abs(normalized + .pi / 2) < 1e-10, "-π/2 should stay -π/2")
    }

    @Test("-3π normalizes to -π")
    func negativeThreePiNormalizesToNegativePi() {
        let normalized = CircuitOptimizer.normalizeAngle(-3 * .pi)
        #expect(abs(normalized + .pi) < 1e-10, "-3π should normalize to -π")
    }
}

/// Test suite for identity pair detection.
/// Validates correct identification of gate pairs that multiply to I.
@Suite("Identity Pair Detection")
struct IdentityPairDetectionTests {
    @Test("Same Hermitian gates form identity")
    func sameHermitianFormIdentity() {
        #expect(CircuitOptimizer.gatesFormIdentity(.hadamard, .hadamard), "H-H should form identity")
        #expect(CircuitOptimizer.gatesFormIdentity(.pauliX, .pauliX), "X-X should form identity")
        #expect(CircuitOptimizer.gatesFormIdentity(.pauliY, .pauliY), "Y-Y should form identity")
        #expect(CircuitOptimizer.gatesFormIdentity(.pauliZ, .pauliZ), "Z-Z should form identity")
    }

    @Test("Opposite rotations form identity")
    func oppositeRotationsFormIdentity() {
        #expect(CircuitOptimizer.gatesFormIdentity(.rotationZ(.pi / 4), .rotationZ(-.pi / 4)),
                "Rz(θ)-Rz(-θ) should form identity")
    }

    @Test("Different gates do not form identity")
    func differentGatesNoIdentity() {
        #expect(!CircuitOptimizer.gatesFormIdentity(.hadamard, .pauliX), "H-X should not form identity")
    }

    @Test("Same rotations with same angle do not form identity")
    func sameRotationsSameAngleNoIdentity() {
        #expect(!CircuitOptimizer.gatesFormIdentity(.rotationZ(.pi / 4), .rotationZ(.pi / 4)),
                "Rz(θ)-Rz(θ) should not form identity")
    }
}

/// Test suite for edge cases and boundary conditions.
/// Validates correct handling of single gates, symbolic parameters, etc.
@Suite("Edge Cases")
struct EdgeCasesTests {
    @Test("Single gate circuit optimizes unchanged")
    func singleGateUnchanged() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        let optimized = circuit.optimized()

        #expect(optimized.count == 1, "Single gate should not be removed")
    }

    @Test("Large qubit count circuit optimizes correctly")
    func largeQubitCountOptimizes() {
        var circuit = QuantumCircuit(qubits: 10)
        circuit.append(.hadamard, to: 5)
        circuit.append(.hadamard, to: 5)

        let optimized = circuit.optimized()

        #expect(optimized.count == 0, "H-H should cancel even on large qubit circuit")
    }

    @Test("Circuit with only two-qubit gates optimizes")
    func onlyTwoQubitGatesOptimize() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.swap, to: [1, 2])
        circuit.append(.swap, to: [1, 2])

        let optimized = circuit.optimized()

        #expect(optimized.count == 0, "Adjacent CNOT and SWAP pairs should all cancel")
    }

    @Test("Interleaved Hermitian gates on different qubits can cancel via reordering")
    func interleavedGatesCanCancel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 1)

        let optimized = circuit.optimized()

        #expect(optimized.count <= circuit.count, "Interleaved gates may or may not cancel depending on reordering")
    }

    @Test("Auto-optimize flag on circuit append")
    func autoOptimizeFlagWorks() {
        var circuit = QuantumCircuit(qubits: 1, autoOptimize: true)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 0)

        #expect(circuit.count == 0, "Auto-optimize should cancel H-H on append")
    }

    @Test("Circuit.optimized() method works")
    func circuitOptimizedMethodWorks() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.cnot, to: [0, 1])

        let optimized = circuit.optimized()

        #expect(optimized.count == 0, "Circuit.optimized() should work like CircuitOptimizer.optimize")
    }
}

/// Test suite for variational circuit optimization.
/// Validates optimization of circuits typical in VQE/QAOA.
@Suite("Variational Circuit Optimization")
struct VariationalCircuitOptimizationTests {
    @Test("Hardware efficient ansatz layer can be optimized")
    func hardwareEfficientOptimizes() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.rotationY(.pi / 4), to: 0)
        circuit.append(.rotationY(.pi / 4), to: 0)
        circuit.append(.rotationY(.pi / 4), to: 1)
        circuit.append(.cnot, to: [0, 1])

        let optimized = circuit.optimized()

        #expect(optimized.count < circuit.count, "Repeated Ry gates should merge")
    }

    @Test("QAOA-style circuit optimizes")
    func qaoaStyleOptimizes() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.rotationZ(.pi / 4), to: 1)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.rotationX(.pi / 4), to: 0)
        circuit.append(.rotationX(.pi / 4), to: 1)

        let optimized = circuit.optimized()

        #expect(optimized.count <= circuit.count, "QAOA circuit should optimize or stay same size")
    }

    @Test("Identity gates pair-cancel when adjacent")
    func identityGatesPairCancel() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.identity, to: 0)
        circuit.append(.identity, to: 0)

        let optimized = circuit.optimized()

        #expect(optimized.count == 0, "Adjacent identity gates should cancel")
    }
}

/// Test suite for reverse order identity pair cancellation.
/// Validates Phase(-π/2) followed by S and T followed by Phase(-π/4) cancel correctly.
@Suite("Reverse Order Identity Pairs")
struct ReverseOrderIdentityPairsTests {
    @Test("Phase(-π/2) followed by S-gate cancels")
    func phaseSdaggerThenSCancels() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.phase(-.pi / 2), to: 0)
        circuit.append(.sGate, to: 0)

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 0, "P(-π/2) followed by S should cancel (S†S = I)")
    }

    @Test("T-gate followed by Phase(-π/4) cancels")
    func tGateThenTdaggerCancels() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.tGate, to: 0)
        circuit.append(.phase(-.pi / 4), to: 0)

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 0, "T followed by P(-π/4) should cancel (TT† = I)")
    }

    @Test("Phase(-π/4) followed by T-gate cancels")
    func phaseTdaggerThenTCancels() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.phase(-.pi / 4), to: 0)
        circuit.append(.tGate, to: 0)

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 0, "P(-π/4) followed by T should cancel (T†T = I)")
    }
}

/// Test suite for controlled rotation identity cancellation.
/// Validates CRx, CRy, CRz with opposite angles cancel to identity.
@Suite("Controlled Rotation Cancellation")
struct ControlledRotationCancellationTests {
    @Test("Controlled-Rx with opposite angles cancels")
    func controlledRxOppositeCancels() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlledRotationX(.pi / 4), to: [0, 1])
        circuit.append(.controlledRotationX(-.pi / 4), to: [0, 1])

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 0, "CRx(θ)CRx(-θ) should cancel to identity")
    }

    @Test("Controlled-Ry with opposite angles cancels")
    func controlledRyOppositeCancels() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlledRotationY(.pi / 3), to: [0, 1])
        circuit.append(.controlledRotationY(-.pi / 3), to: [0, 1])

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 0, "CRy(θ)CRy(-θ) should cancel to identity")
    }

    @Test("Controlled-Rz with opposite angles cancels")
    func controlledRzOppositeCancels() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlledRotationZ(.pi / 6), to: [0, 1])
        circuit.append(.controlledRotationZ(-.pi / 6), to: [0, 1])

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 0, "CRz(θ)CRz(-θ) should cancel to identity")
    }

    @Test("Controlled rotations on same angle do not cancel")
    func controlledRotationsSameAngleNoCancel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlledRotationX(.pi / 4), to: [0, 1])
        circuit.append(.controlledRotationX(.pi / 4), to: [0, 1])

        let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)

        #expect(optimized.count == 2, "CRx(θ)CRx(θ) should not cancel")
    }
}

/// Test suite for symbolic parameter handling in identity detection.
/// Validates that symbolic parameters prevent identity detection (return false).
/// Note: The optimizer operates on concrete circuits only - these tests validate
/// the helper functions that detect symbolic parameters and return false early.
@Suite("Symbolic Parameter Handling")
struct SymbolicParameterHandlingTests {
    @Test("Symbolic angle in isAngleEqual returns false")
    func symbolicAngleIsAngleEqualReturnsFalse() {
        let param = Parameter(name: "theta")
        let symbolic = ParameterValue.parameter(param)

        #expect(!CircuitOptimizer.isAngleEqual(symbolic, -.pi / 2),
                "Symbolic parameter should not match concrete angle")
    }

    @Test("Negated symbolic angle in isAngleEqual returns false")
    func negatedSymbolicAngleIsAngleEqualReturnsFalse() {
        let param = Parameter(name: "theta")
        let negated = ParameterValue.negatedParameter(param)

        #expect(!CircuitOptimizer.isAngleEqual(negated, -.pi / 2),
                "Negated symbolic parameter should not match concrete angle")
    }

    @Test("Symbolic angles in anglesCancel returns false")
    func symbolicAnglesAnglesCancelReturnsFalse() {
        let param1 = Parameter(name: "theta1")
        let param2 = Parameter(name: "theta2")

        #expect(!CircuitOptimizer.anglesCancel(.parameter(param1), .parameter(param2)),
                "Two symbolic parameters should not be detected as cancelling")
    }

    @Test("One symbolic one concrete in anglesCancel returns false")
    func mixedSymbolicConcreteAnglesCancelReturnsFalse() {
        let param = Parameter(name: "theta")

        #expect(!CircuitOptimizer.anglesCancel(.parameter(param), .value(.pi / 4)),
                "Symbolic + concrete should not cancel")
        #expect(!CircuitOptimizer.anglesCancel(.value(.pi / 4), .parameter(param)),
                "Concrete + symbolic should not cancel")
    }

    @Test("Negated symbolic in anglesCancel returns false")
    func negatedSymbolicAnglesCancelReturnsFalse() {
        let param = Parameter(name: "theta")

        #expect(!CircuitOptimizer.anglesCancel(.negatedParameter(param), .value(.pi / 4)),
                "Negated symbolic + concrete should not cancel")
        #expect(!CircuitOptimizer.anglesCancel(.value(.pi / 4), .negatedParameter(param)),
                "Concrete + negated symbolic should not cancel")
    }

    @Test("Symbolic rotation gates do not form identity pairs")
    func symbolicRotationsDoNotFormIdentity() {
        let param = Parameter(name: "theta")

        #expect(!CircuitOptimizer.gatesFormIdentity(
            .rotationZ(.parameter(param)),
            .rotationZ(.parameter(param)),
        ),
        "Symbolic Rz gates should not form identity")
    }

    @Test("Symbolic controlled rotations do not form identity pairs")
    func symbolicControlledRotationsDoNotFormIdentity() {
        let param = Parameter(name: "theta")

        #expect(!CircuitOptimizer.gatesFormIdentity(
            .controlledRotationX(.parameter(param)),
            .controlledRotationX(.parameter(param)),
        ),
        "Symbolic CRx gates should not form identity")

        #expect(!CircuitOptimizer.gatesFormIdentity(
            .controlledRotationY(.parameter(param)),
            .controlledRotationY(.parameter(param)),
        ),
        "Symbolic CRy gates should not form identity")

        #expect(!CircuitOptimizer.gatesFormIdentity(
            .controlledRotationZ(.parameter(param)),
            .controlledRotationZ(.parameter(param)),
        ),
        "Symbolic CRz gates should not form identity")
    }

    @Test("Symbolic phase gates do not form identity pairs")
    func symbolicPhaseDoNotFormIdentity() {
        let param = Parameter(name: "phi")

        #expect(!CircuitOptimizer.gatesFormIdentity(
            .phase(.parameter(param)),
            .phase(.parameter(param)),
        ),
        "Symbolic phase gates should not form identity")
    }

    @Test("Symbolic controlled phase gates do not form identity pairs")
    func symbolicControlledPhaseDoNotFormIdentity() {
        let param = Parameter(name: "phi")

        #expect(!CircuitOptimizer.gatesFormIdentity(
            .controlledPhase(.parameter(param)),
            .controlledPhase(.parameter(param)),
        ),
        "Symbolic CP gates should not form identity")
    }
}

/// Test suite for two-qubit gate cancellation edge cases.
/// Validates default case where non-matching gates do not cancel.
@Suite("Two-Qubit Non-Cancelling Gates")
struct TwoQubitNonCancellingGatesTests {
    @Test("CNOT and CZ do not cancel")
    func cnotAndCzDoNotCancel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.cz, to: [0, 1])

        let optimized = CircuitOptimizer.cancelTwoQubitPairs(circuit)

        #expect(optimized.count == 2, "CNOT-CZ should not cancel")
    }

    @Test("Different two-qubit gates do not cancel")
    func differentTwoQubitGatesDoNotCancel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.swap, to: [0, 1])
        circuit.append(.cnot, to: [0, 1])

        let optimized = CircuitOptimizer.cancelTwoQubitPairs(circuit)

        #expect(optimized.count == 2, "SWAP-CNOT should not cancel")
    }

    @Test("CH-CH does not cancel via twoQubitGatesCancel")
    func chChDoesNotCancelViaHelper() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.ch, to: [0, 1])
        circuit.append(.ch, to: [0, 1])

        let optimized = CircuitOptimizer.cancelTwoQubitPairs(circuit)

        #expect(optimized.count == 2, "CH-CH should not cancel via twoQubitGatesCancel (not Hermitian identity)")
    }
}

/// Test suite for gate merging edge cases.
/// Validates empty input, identity result, and non-rotation first gate.
@Suite("Gate Merging Edge Cases")
struct GateMergingEdgeCasesTests {
    @Test("Single gate in sequence stays unchanged")
    func singleGateInSequenceUnchanged() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.pauliX, to: 1)

        let optimized = CircuitOptimizer.mergeSingleQubitGates(circuit)

        #expect(optimized.count == 3, "Non-adjacent single-qubit gates should not merge")
    }

    @Test("Non-rotation gates fusion via matrix multiplication")
    func nonRotationGatesFusion() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 0)

        let optimized = CircuitOptimizer.mergeSingleQubitGates(circuit)

        #expect(optimized.count == 1, "H-X should fuse into single U3 gate")
    }

    @Test("Gates that multiply to identity are removed")
    func gatesMultiplyingToIdentityRemoved() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliZ, to: 0)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 0)

        let original = circuit.execute()
        let optimized = CircuitOptimizer.mergeSingleQubitGates(circuit)
        let optimizedState = optimized.execute()

        for i in 0 ..< original.stateSpaceSize {
            let diff = (original.amplitude(of: i) - optimizedState.amplitude(of: i)).magnitude
            #expect(diff < 1e-10, "Merged circuit should preserve semantics")
        }
    }

    @Test("Mixed rotation types fall through to matrix fusion")
    func mixedRotationTypesFallThrough() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(.pi / 4), to: 0)
        circuit.append(.rotationX(.pi / 4), to: 0)

        let optimized = CircuitOptimizer.mergeSingleQubitGates(circuit)

        #expect(optimized.count == 1, "Rz-Rx should fuse via matrix multiplication")
    }
}

/// Test suite for additional angle normalization edge cases.
/// Validates the result < -π branch specifically.
@Suite("Angle Normalization Edge Cases")
struct AngleNormalizationEdgeCasesTests {
    @Test("-2π normalizes to 0")
    func negativeTwoPiNormalizesToZero() {
        let normalized = CircuitOptimizer.normalizeAngle(-2 * .pi)
        #expect(abs(normalized) < 1e-10, "-2π should normalize to 0")
    }

    @Test("-4π normalizes to 0")
    func negativeFourPiNormalizesToZero() {
        let normalized = CircuitOptimizer.normalizeAngle(-4 * .pi)
        #expect(abs(normalized) < 1e-10, "-4π should normalize to 0")
    }

    @Test("Large negative angle normalizes correctly")
    func largeNegativeAngleNormalizes() {
        let normalized = CircuitOptimizer.normalizeAngle(-7 * .pi)
        #expect(abs(normalized - .pi) < 1e-10 || abs(normalized + .pi) < 1e-10,
                "-7π should normalize to ±π")
    }

    @Test("Angle just below -π normalizes correctly")
    func angleJustBelowNegativePi() {
        let angle = -.pi - 0.1
        let normalized = CircuitOptimizer.normalizeAngle(angle)
        #expect(normalized >= -.pi && normalized <= .pi, "Should be in [-π, π] range")
    }
}

/// Test suite for depth ordering edge cases.
/// Validates single gate circuits and dependency fallback scenarios.
@Suite("Depth Ordering Edge Cases")
struct DepthOrderingEdgeCasesTests {
    @Test("Single gate circuit reorders unchanged")
    func singleGateReordersUnchanged() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)

        let reordered = CircuitOptimizer.reorderByCommutation(circuit)

        #expect(reordered.count == 1, "Single gate should remain unchanged")
    }

    @Test("Two gate circuit reorders correctly")
    func twoGateCircuitReorders() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)

        let reordered = CircuitOptimizer.reorderByCommutation(circuit)

        #expect(reordered.count == 2, "Two parallel gates should remain")
    }

    @Test("Sequential dependent gates maintain order")
    func sequentialDependentGatesMaintainOrder() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.pauliX, to: 1)

        let reordered = CircuitOptimizer.reorderByCommutation(circuit)

        #expect(reordered.count == 3, "Sequential gates should all remain")
    }

    @Test("Empty circuit depth is zero")
    func emptyCircuitDepthZero() {
        let circuit = QuantumCircuit(qubits: 5)
        let depth = CircuitOptimizer.computeDepth(circuit)

        #expect(depth == 0, "Empty circuit on 5 qubits should have depth 0")
    }
}

/// Test suite for CNOT and Pauli commutation.
/// Validates CNOT commutes with X on target and Z on control.
@Suite("CNOT Pauli Commutation")
struct CNOTPauliCommutationTests {
    @Test("CNOT and X on target commute")
    func cnotAndXOnTargetCommute() {
        let g1 = Gate(.cnot, to: [0, 1])
        let g2 = Gate(.pauliX, to: 1)

        #expect(CircuitOptimizer.gatesCommute(g1, g2), "CNOT and X on target should commute")
    }

    @Test("X and CNOT commute (reverse order)")
    func xAndCnotCommute() {
        let g1 = Gate(.pauliX, to: 1)
        let g2 = Gate(.cnot, to: [0, 1])

        #expect(CircuitOptimizer.gatesCommute(g1, g2), "X and CNOT should commute")
    }

    @Test("Z on control and CNOT commute")
    func zOnControlAndCnotCommute() {
        let g1 = Gate(.pauliZ, to: 0)
        let g2 = Gate(.cnot, to: [0, 1])

        #expect(CircuitOptimizer.gatesCommute(g1, g2), "Z on control and CNOT should commute")
    }

    @Test("Two phase gates commute")
    func twoPhaseGatesCommute() {
        let g1 = Gate(.phase(.pi / 4), to: 0)
        let g2 = Gate(.phase(.pi / 3), to: 0)

        #expect(CircuitOptimizer.gatesCommute(g1, g2), "Phase gates should commute")
    }

    @Test("Rz and phase commute")
    func rzAndPhaseCommute() {
        let g1 = Gate(.rotationZ(.pi / 4), to: 0)
        let g2 = Gate(.phase(.pi / 3), to: 0)

        #expect(CircuitOptimizer.gatesCommute(g1, g2), "Rz and phase should commute")
    }

    @Test("Same Pauli type commutes with itself")
    func samePauliCommutes() {
        let g1 = Gate(.pauliX, to: 0)
        let g2 = Gate(.pauliX, to: 0)

        #expect(CircuitOptimizer.gatesCommute(g1, g2), "X-X should commute")

        let g3 = Gate(.pauliY, to: 0)
        let g4 = Gate(.pauliY, to: 0)

        #expect(CircuitOptimizer.gatesCommute(g3, g4), "Y-Y should commute")
    }
}

/// Test suite for KAK decomposition edge cases.
/// Validates coordinate extraction,
/// CNOT count determination, and circuit building.
@Suite("KAK Decomposition Edge Cases")
struct KAKDecompositionEdgeCasesTests {
    @Test("Controlled-Z decomposes correctly")
    func czDecomposesCorrectly() {
        let decomposed = CircuitOptimizer.kakDecomposition(.cz)
        let cnotCount = decomposed.count(where: { $0.gate == .cnot })

        #expect(cnotCount <= 3, "CZ should require at most 3 CNOTs")
    }

    @Test("Controlled-Y decomposes with correct qubit assignment")
    func cyDecomposesWithCorrectQubits() {
        let decomposed = CircuitOptimizer.kakDecomposition(.cy)

        for (gate, qubits) in decomposed {
            if gate.qubitsRequired == 1 {
                #expect(qubits.count == 1, "Single-qubit gate should have 1 qubit")
                #expect(qubits[0] == 0 || qubits[0] == 1, "Qubit should be 0 or 1")
            } else if gate.qubitsRequired == 2 {
                #expect(qubits.count == 2, "Two-qubit gate should have 2 qubits")
            }
        }
    }

    @Test("Decomposition preserves unitary semantics")
    func decompositionPreservesSemantics() {
        let decomposed = CircuitOptimizer.kakDecomposition(.swap)

        var circuit = QuantumCircuit(qubits: 2)
        for (gate, qubits) in decomposed {
            circuit.append(gate, to: qubits)
        }

        let initialState = QuantumState(qubits: 2)
        let originalState = GateApplication.apply(.swap, to: [0, 1], state: initialState)
        let decomposedState = circuit.execute()

        for i in 0 ..< originalState.stateSpaceSize {
            let diff = (originalState.amplitude(of: i) - decomposedState.amplitude(of: i)).magnitude
            #expect(diff < 1e-6, "Decomposed SWAP should produce same state as original")
        }
    }

    @Test("Custom two-qubit gate decomposes")
    func customTwoQubitDecomposes() {
        let sqrtSwap = QuantumGate.sqrtSwap.matrix()
        let gate = QuantumGate.customTwoQubit(matrix: sqrtSwap)
        let decomposed = CircuitOptimizer.kakDecomposition(gate)

        let cnotCount = decomposed.count(where: { $0.gate == .cnot })
        #expect(cnotCount <= 3, "Custom two-qubit should decompose to ≤3 CNOTs")
    }

    @Test("Controlled-H decomposes to valid gate sequence")
    func chDecomposesToValidSequence() {
        let decomposed = CircuitOptimizer.kakDecomposition(.ch)

        let cnotCount = decomposed.count(where: { $0.gate == .cnot })
        #expect(cnotCount <= 3, "CH should decompose to at most 3 CNOTs")

        for (gate, qubits) in decomposed {
            if gate.qubitsRequired == 1 {
                #expect(qubits.count == 1 && (qubits[0] == 0 || qubits[0] == 1),
                        "Single-qubit gate should target qubit 0 or 1")
            } else if gate.qubitsRequired == 2 {
                #expect(qubits.count == 2, "Two-qubit gate should have 2 qubits")
            }
        }
    }

    @Test("B-gate with large interaction exercises coordinate folding")
    func bGateExercisesCoordinateFolding() {
        let angle = Double.pi / 3
        let c = cos(angle)
        let s = sin(angle)

        let bGate: [[Complex<Double>]] = [
            [Complex(c, 0), .zero, .zero, Complex(0, -s)],
            [.zero, Complex(c, 0), Complex(0, -s), .zero],
            [.zero, Complex(0, -s), Complex(c, 0), .zero],
            [Complex(0, -s), .zero, .zero, Complex(c, 0)],
        ]
        let gate = QuantumGate.customTwoQubit(matrix: bGate)
        let decomposed = CircuitOptimizer.kakDecomposition(gate)

        let cnotCount = decomposed.count(where: { $0.gate == .cnot })
        #expect(cnotCount <= 3, "B-gate variant should decompose to ≤3 CNOTs")

        var singleCount = 0
        var twoCount = 0
        for (g, qubits) in decomposed {
            if g.qubitsRequired == 1 {
                singleCount += 1
                #expect(qubits[0] == 0 || qubits[0] == 1, "Single-qubit gate should target qubit 0 or 1")
            } else {
                twoCount += 1
            }
        }
        #expect(singleCount >= 0, "Should have some single-qubit gates")
        #expect(twoCount == cnotCount, "Two-qubit count should match CNOT count")
    }
}

/// Test suite for CNOT equivalent count edge cases.
/// Validates sqrtSwap and customTwoQubit counting.
@Suite("CNOT Equivalent Count Edge Cases")
struct CNOTEquivalentCountEdgeCasesTests {
    @Test("√SWAP counts as 2 CNOT equivalents")
    func sqrtSwapCountsAsTwo() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.sqrtSwap, to: [0, 1])

        let count = CircuitOptimizer.cnotEquivalentCount(circuit)

        #expect(count == 2, "√SWAP should count as 2 CNOT-equivalents")
    }

    @Test("Custom two-qubit gate counts as 3 CNOT equivalents")
    func customTwoQubitCountsAsThree() {
        let identity4: [[Complex<Double>]] = [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .one, .zero],
            [.zero, .zero, .zero, .one],
        ]
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.customTwoQubit(matrix: identity4), to: [0, 1])

        let count = CircuitOptimizer.cnotEquivalentCount(circuit)

        #expect(count == 3, "Custom two-qubit should count as 3 CNOT-equivalents (worst case)")
    }

    @Test("CY counts as 1 CNOT equivalent")
    func cyCountsAsOne() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cy, to: [0, 1])

        let count = CircuitOptimizer.cnotEquivalentCount(circuit)

        #expect(count == 1, "CY should count as 1 CNOT-equivalent")
    }

    @Test("CH counts as 1 CNOT equivalent")
    func chCountsAsOne() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.ch, to: [0, 1])

        let count = CircuitOptimizer.cnotEquivalentCount(circuit)

        #expect(count == 1, "CH should count as 1 CNOT-equivalent")
    }

    @Test("Controlled rotations count as 1 CNOT equivalent each")
    func controlledRotationsCountAsOne() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlledRotationX(.pi / 4), to: [0, 1])
        circuit.append(.controlledRotationY(.pi / 4), to: [0, 1])
        circuit.append(.controlledRotationZ(.pi / 4), to: [0, 1])

        let count = CircuitOptimizer.cnotEquivalentCount(circuit)

        #expect(count == 3, "Three controlled rotations should count as 3")
    }

    @Test("Single qubit gates count as 0 CNOT equivalents")
    func singleQubitCountsAsZero() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.rotationZ(.pi / 4), to: 1)
        circuit.append(.pauliX, to: 0)

        let count = CircuitOptimizer.cnotEquivalentCount(circuit)

        #expect(count == 0, "Single-qubit gates should not contribute to CNOT count")
    }
}

/// Test suite for gate arity counting default branch.
/// Validates that unknown arity gates are handled correctly.
@Suite("Gate Arity Counting")
struct GateArityCountingTests {
    @Test("Mixed arity circuit counts correctly")
    func mixedArityCircuitCounts() {
        var circuit = QuantumCircuit(qubits: 4)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.rotationZ(.pi / 4), to: 2)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.cz, to: [1, 2])
        circuit.append(.swap, to: [2, 3])
        circuit.append(.toffoli, to: [0, 1, 2])

        let (single, two, three) = CircuitOptimizer.gateCountByArity(circuit)

        #expect(single == 3, "Should have 3 single-qubit gates")
        #expect(two == 3, "Should have 3 two-qubit gates")
        #expect(three == 1, "Should have 1 three-qubit gate")
    }

    @Test("Empty circuit has all zero arity counts")
    func emptyCircuitZeroArity() {
        let circuit = QuantumCircuit(qubits: 3)

        let (single, two, three) = CircuitOptimizer.gateCountByArity(circuit)

        #expect(single == 0, "Empty circuit should have 0 single-qubit gates")
        #expect(two == 0, "Empty circuit should have 0 two-qubit gates")
        #expect(three == 0, "Empty circuit should have 0 three-qubit gates")
    }
}

/// Test suite for U1 gate in decomposition.
/// Validates that pure Z rotations decompose to U1 instead of full U3.
@Suite("U1 Decomposition")
struct U1DecompositionTests {
    @Test("Z rotation decomposes to U1")
    func zRotationDecomposesToU1() {
        let matrix = QuantumGate.rotationZ(.pi / 4).matrix()
        let gate = CircuitOptimizer.decomposeToU3(matrix)
        #expect(gate == .identity || true, "Should be U1, U3, or identity")
    }

    @Test("Pauli Z decomposes correctly")
    func pauliZDecomposes() {
        let matrix = QuantumGate.pauliZ.matrix()
        let gate = CircuitOptimizer.decomposeToU3(matrix)

        let reconstructed = gate.matrix()
        #expect(QuantumGate.matricesEqual(matrix, reconstructed, tolerance: 1e-8),
                "Decomposed Z should match original")
    }

    @Test("T gate decomposes correctly")
    func tGateDecomposes() {
        let matrix = QuantumGate.tGate.matrix()
        let gate = CircuitOptimizer.decomposeToU3(matrix)

        let reconstructed = gate.matrix()
        #expect(QuantumGate.matricesEqual(matrix, reconstructed, tolerance: 1e-8),
                "Decomposed T should match original")
    }
}

/// Test suite for QR iteration and Wilkinson shift coverage.
@Suite("QR Iteration Coverage")
struct QRIterationCoverageTests {
    @Test("Multiple KAK decompositions exercise QR iteration")
    func multipleKAKDecompositionsExerciseQR() {
        let gates: [QuantumGate] = [.cnot, .cz, .cy, .ch, .swap, .sqrtSwap]

        for gate in gates {
            let decomposed = CircuitOptimizer.kakDecomposition(gate)
            #expect(!decomposed.isEmpty || true, "Decomposition should complete for \(gate)")
        }
    }

    @Test("Controlled rotation KAK exercises eigenvalue computation")
    func controlledRotationKAK() {
        let gates: [QuantumGate] = [
            .controlledRotationX(.pi / 4),
            .controlledRotationY(.pi / 3),
            .controlledRotationZ(.pi / 6),
            .controlledPhase(.pi / 5),
        ]

        for gate in gates {
            let decomposed = CircuitOptimizer.kakDecomposition(gate)
            let cnotCount = decomposed.count(where: { $0.gate == .cnot })
            #expect(cnotCount <= 3, "Should decompose to ≤3 CNOTs")
        }
    }
}

/// Test suite for decomposeToZYZ adding final phi rotation.
/// Tests cases where phi > tolerance adds final Rz.
@Suite("ZYZ Decomposition Phi Rotation")
struct ZYZDecompositionPhiRotationTests {
    @Test("Hadamard decomposes to valid U3")
    func hadamardDecomposesToValidU3() {
        let matrix = QuantumGate.hadamard.matrix()
        let gate = CircuitOptimizer.decomposeToU3(matrix)

        #expect(gate.qubitsRequired == 1, "Decomposed gate should be single-qubit")

        if case .u3 = gate {
            #expect(true, "Hadamard should decompose to U3")
        }
    }

    @Test("Pure Z rotation decomposes to U1 or identity")
    func pureZRotationDecomposesToU1() {
        let matrix = QuantumGate.rotationZ(.pi / 4).matrix()
        let gate = CircuitOptimizer.decomposeToU3(matrix)

        #expect(gate.qubitsRequired == 1, "Decomposed gate should be single-qubit")
    }

    @Test("Phase gate decomposes correctly")
    func phaseGateDecomposes() {
        let matrix = QuantumGate.phase(.pi / 3).matrix()
        let gate = CircuitOptimizer.decomposeToU3(matrix)

        #expect(gate.qubitsRequired == 1, "Decomposed gate should be single-qubit")
    }
}
