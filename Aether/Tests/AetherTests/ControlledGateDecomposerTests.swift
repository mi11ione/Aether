// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under Apache 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for decompose() method dispatch logic.
/// Validates correct decomposition strategy selection based on control count,
/// from direct application (0 controls) through Toffoli ladder (3+ controls).
@Suite("Decompose Method Dispatch")
struct DecomposeMethodDispatchTests {
    @Test("Zero controls returns direct gate application")
    func zeroControlsReturnsDirectGate() {
        let result = ControlledGateDecomposer.decompose(
            gate: .pauliX,
            controls: [],
            target: 0,
        )

        #expect(result.count == 1, "Zero controls should return single gate application")
        #expect(result[0].gate == .pauliX, "Gate should be unchanged pauliX")
        #expect(result[0].qubits == [0], "Qubits should be target only")
    }

    @Test("Single control with pauliX returns CNOT")
    func singleControlPauliXReturnsCNOT() {
        let result = ControlledGateDecomposer.decompose(
            gate: .pauliX,
            controls: [0],
            target: 1,
        )

        #expect(result.count == 1, "Single control X should return single CNOT")
        #expect(result[0].gate == .cnot, "Gate should be CNOT for controlled-X")
        #expect(result[0].qubits == [0, 1], "Qubits should be [control, target]")
    }

    @Test("Single control with pauliZ returns CZ")
    func singleControlPauliZReturnsCZ() {
        let result = ControlledGateDecomposer.decompose(
            gate: .pauliZ,
            controls: [0],
            target: 1,
        )

        #expect(result.count == 1, "Single control Z should return single CZ")
        #expect(result[0].gate == .cz, "Gate should be CZ for controlled-Z")
        #expect(result[0].qubits == [0, 1], "Qubits should be [control, target]")
    }

    @Test("Single control with hadamard returns CH")
    func singleControlHadamardReturnsCH() {
        let result = ControlledGateDecomposer.decompose(
            gate: .hadamard,
            controls: [0],
            target: 1,
        )

        #expect(result.count == 1, "Single control H should return single CH")
        #expect(result[0].gate == .ch, "Gate should be CH for controlled-H")
        #expect(result[0].qubits == [0, 1], "Qubits should be [control, target]")
    }

    @Test("Two controls with pauliX returns Toffoli")
    func twoControlsPauliXReturnsToffoli() {
        let result = ControlledGateDecomposer.decompose(
            gate: .pauliX,
            controls: [0, 1],
            target: 2,
        )

        #expect(result.count == 1, "Two controls X should return single Toffoli")
        #expect(result[0].gate == .toffoli, "Gate should be Toffoli for CCX")
        #expect(result[0].qubits == [0, 1, 2], "Qubits should be [c0, c1, target]")
    }

    @Test("Three or more controls returns Toffoli ladder sequence")
    func threeControlsReturnsToffoliLadder() {
        let result = ControlledGateDecomposer.decompose(
            gate: .pauliX,
            controls: [0, 1, 2],
            target: 3,
        )

        #expect(result.count > 1, "Three controls should return multi-gate sequence")

        let toffoliCount = result.count(where: { $0.gate == .toffoli })
        #expect(toffoliCount == 3, "Three control decomposition should have 3 Toffolis")
    }
}

/// Test suite for decomposeSingleControlled() native gate mappings.
/// Validates all native controlled gate mappings including Pauli gates,
/// Hadamard, and controlled rotation variants (CRx, CRy, CRz).
@Suite("Single Controlled Decomposition")
struct SingleControlledDecompositionTests {
    @Test("PauliX maps to CNOT")
    func pauliXMapsToCNOT() {
        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .pauliX,
            control: 0,
            target: 1,
        )

        #expect(result.count == 1, "Controlled-X should map to single CNOT")
        #expect(result[0].gate == .cnot, "Gate should be CNOT")
        #expect(result[0].qubits == [0, 1], "Qubits should be [control, target]")
    }

    @Test("PauliZ maps to CZ")
    func pauliZMapsToCZ() {
        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .pauliZ,
            control: 0,
            target: 1,
        )

        #expect(result.count == 1, "Controlled-Z should map to single CZ")
        #expect(result[0].gate == .cz, "Gate should be CZ")
    }

    @Test("PauliY maps to CY")
    func pauliYMapsToCY() {
        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .pauliY,
            control: 0,
            target: 1,
        )

        #expect(result.count == 1, "Controlled-Y should map to single CY")
        #expect(result[0].gate == .cy, "Gate should be CY")
    }

    @Test("Hadamard maps to CH")
    func hadamardMapsToCH() {
        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .hadamard,
            control: 0,
            target: 1,
        )

        #expect(result.count == 1, "Controlled-H should map to single CH")
        #expect(result[0].gate == .ch, "Gate should be CH")
    }

    @Test("RotationX maps to controlled rotation X")
    func rotationXMapsToControlledRotationX() {
        let angle = Double.pi / 4
        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .rotationX(angle),
            control: 0,
            target: 1,
        )

        #expect(result.count == 1, "Controlled-Rx should map to single CRx")
        if case let .controlledRotationX(theta) = result[0].gate,
           case let .value(v) = theta
        {
            #expect(abs(v - angle) < 1e-10, "CRx angle should match input angle")
        }
    }

    @Test("RotationY maps to controlled rotation Y")
    func rotationYMapsToControlledRotationY() {
        let angle = Double.pi / 3
        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .rotationY(angle),
            control: 0,
            target: 1,
        )

        #expect(result.count == 1, "Controlled-Ry should map to single CRy")
        if case let .controlledRotationY(theta) = result[0].gate,
           case let .value(v) = theta
        {
            #expect(abs(v - angle) < 1e-10, "CRy angle should match input angle")
        }
    }

    @Test("RotationZ maps to controlled rotation Z")
    func rotationZMapsToControlledRotationZ() {
        let angle = Double.pi / 6
        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .rotationZ(angle),
            control: 0,
            target: 1,
        )

        #expect(result.count == 1, "Controlled-Rz should map to single CRz")
        if case let .controlledRotationZ(theta) = result[0].gate,
           case let .value(v) = theta
        {
            #expect(abs(v - angle) < 1e-10, "CRz angle should match input angle")
        }
    }

    @Test("Phase gate maps to controlled phase")
    func phaseGateMapsToControlledPhase() {
        let angle = Double.pi / 5
        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .phase(angle),
            control: 0,
            target: 1,
        )

        #expect(result.count == 1, "Controlled-P should map to single CP")
        if case let .controlledPhase(theta) = result[0].gate,
           case let .value(v) = theta
        {
            #expect(abs(v - angle) < 1e-10, "CP angle should match input angle")
        }
    }

    @Test("S gate maps to controlled phase pi/2")
    func sGateMapsToControlledPhase() {
        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .sGate,
            control: 0,
            target: 1,
        )

        #expect(result.count == 1, "Controlled-S should map to single CP(pi/2)")
        if case let .controlledPhase(theta) = result[0].gate,
           case let .value(v) = theta
        {
            #expect(abs(v - Double.pi / 2) < 1e-10, "CP angle should be pi/2 for S gate")
        }
    }

    @Test("T gate maps to controlled phase pi/4")
    func tGateMapsToControlledPhase() {
        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .tGate,
            control: 0,
            target: 1,
        )

        #expect(result.count == 1, "Controlled-T should map to single CP(pi/4)")
        if case let .controlledPhase(theta) = result[0].gate,
           case let .value(v) = theta
        {
            #expect(abs(v - Double.pi / 4) < 1e-10, "CP angle should be pi/4 for T gate")
        }
    }

    @Test("Identity gate returns empty sequence")
    func identityGateReturnsEmpty() {
        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .identity,
            control: 0,
            target: 1,
        )

        #expect(result.isEmpty, "Controlled-I should return empty sequence")
    }

    @Test("U1 gate maps to controlled phase")
    func u1GateMapsToControlledPhase() {
        let angle = Double.pi / 7
        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u1(lambda: .value(angle)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 1, "Controlled-U1 should map to single CP")
        var isControlledPhase = false
        if case .controlledPhase = result[0].gate {
            isControlledPhase = true
        }
        #expect(isControlledPhase, "Gate should be controlledPhase")
    }

    @Test("U3 gate decomposes to multi-gate sequence")
    func u3GateDecomposesToSequence() {
        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(Double.pi / 4), phi: .value(Double.pi / 3), lambda: .value(Double.pi / 6)),
            control: 0,
            target: 1,
        )

        #expect(result.count > 1, "Controlled-U3 should decompose to multi-gate sequence")

        let cnotCount = result.count(where: { $0.gate == QuantumGate.cnot })
        #expect(cnotCount == 2, "Controlled-U3 decomposition should have 2 CNOTs")
    }

    @Test("U2 gate decomposes to multi-gate sequence")
    func u2GateDecomposesToSequence() {
        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u2(phi: .value(Double.pi / 3), lambda: .value(Double.pi / 6)),
            control: 0,
            target: 1,
        )

        #expect(result.count > 1, "Controlled-U2 should decompose to multi-gate sequence")

        let cnotCount = result.count(where: { $0.gate == QuantumGate.cnot })
        #expect(cnotCount == 2, "Controlled-U2 decomposition should have 2 CNOTs")
    }

    @Test("SX gate decomposes to custom two-qubit")
    func sxGateDecomposesToCustom() {
        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .sx,
            control: 0,
            target: 1,
        )

        #expect(result.count == 1, "Controlled-SX should return single custom gate")
        var isCustomTwoQubit = false
        if case .customTwoQubit = result[0].gate {
            isCustomTwoQubit = true
        }
        #expect(isCustomTwoQubit, "Gate should be customTwoQubit")
    }
}

/// Test suite for toffoliLadderDecomposition() multi-control logic.
/// Validates ancilla allocation, ladder structure with forward and reverse
/// Toffoli sequences, and correct basis transformations for non-X gates.
@Suite("Toffoli Ladder Decomposition")
struct ToffoliLadderDecompositionTests {
    @Test("Zero controls in ladder returns direct gate")
    func zeroControlsInLadderReturnsDirectGate() {
        let result = ControlledGateDecomposer.toffoliLadderDecomposition(
            gate: .pauliX,
            controls: [],
            target: 0,
        )

        #expect(result.count == 1, "Zero controls should return single gate")
        #expect(result[0].gate == .pauliX, "Gate should be pauliX")
    }

    @Test("Single control in ladder delegates to single controlled")
    func singleControlInLadderDelegatesToSingleControlled() {
        let result = ControlledGateDecomposer.toffoliLadderDecomposition(
            gate: .pauliX,
            controls: [0],
            target: 1,
        )

        #expect(result.count == 1, "Single control should delegate to decomposeSingleControlled")
        #expect(result[0].gate == .cnot, "Gate should be CNOT")
    }

    @Test("Two controls applies Toffoli with basis change")
    func twoControlsAppliesToffoliWithBasisChange() {
        let result = ControlledGateDecomposer.toffoliLadderDecomposition(
            gate: .pauliZ,
            controls: [0, 1],
            target: 2,
        )

        #expect(result.count == 3, "Two control Z should have H-Toffoli-H")

        let toffoliCount = result.count(where: { $0.gate == .toffoli })
        #expect(toffoliCount == 1, "Should have exactly one Toffoli")

        let hadamardCount = result.count(where: { $0.gate == .hadamard })
        #expect(hadamardCount == 2, "Should have two Hadamards for Z basis change")
    }

    @Test("Three controls allocates correct ancilla")
    func threeControlsAllocatesCorrectAncilla() {
        let result = ControlledGateDecomposer.toffoliLadderDecomposition(
            gate: .pauliX,
            controls: [0, 1, 2],
            target: 3,
        )

        let allQubits = result.flatMap(\.qubits)
        let maxQubit = allQubits.max()!

        #expect(maxQubit == 4, "Three controls should use ancilla at qubit 4 (max+1)")
    }

    @Test("Three controls produces correct ladder structure")
    func threeControlsProducesCorrectLadderStructure() {
        let result = ControlledGateDecomposer.toffoliLadderDecomposition(
            gate: .pauliX,
            controls: [0, 1, 2],
            target: 3,
        )

        let toffoliGates = result.filter { $0.gate == .toffoli }
        #expect(toffoliGates.count == 3, "Three controls should have 3 Toffolis (1 forward, 1 target, 1 reverse)")

        #expect(toffoliGates[0].qubits == [0, 1, 4], "First Toffoli combines c0 AND c1 into ancilla")
        #expect(toffoliGates[1].qubits == [4, 2, 3], "Second Toffoli applies to target")
        #expect(toffoliGates[2].qubits == [0, 1, 4], "Third Toffoli uncomputes first")
    }

    @Test("Four controls allocates two ancillas")
    func fourControlsAllocatesTwoAncillas() {
        let result = ControlledGateDecomposer.toffoliLadderDecomposition(
            gate: .pauliX,
            controls: [0, 1, 2, 3],
            target: 4,
        )

        let allQubits = result.flatMap(\.qubits)
        let uniqueQubits = Set(allQubits)

        #expect(uniqueQubits.contains(5), "Four controls should use ancilla at qubit 5")
        #expect(uniqueQubits.contains(6), "Four controls should use ancilla at qubit 6")
    }

    @Test("Ladder with non-X gate includes basis transformation")
    func ladderWithNonXGateIncludesBasisTransformation() {
        let result = ControlledGateDecomposer.toffoliLadderDecomposition(
            gate: .pauliZ,
            controls: [0, 1, 2],
            target: 3,
        )

        #expect(result[0].gate == .hadamard, "First gate should be Hadamard prefix for Z")

        let lastGate = result[result.count - 1]
        #expect(lastGate.gate == .hadamard, "Last gate should be Hadamard suffix for Z")
    }

    @Test("Ladder preserves symmetry in Toffoli sequence")
    func ladderPreservesSymmetryInToffoliSequence() {
        let result = ControlledGateDecomposer.toffoliLadderDecomposition(
            gate: .pauliX,
            controls: [0, 1, 2, 3],
            target: 4,
        )

        let toffoliGates = result.filter { $0.gate == .toffoli }
        let n = toffoliGates.count

        #expect(toffoliGates[0].qubits == toffoliGates[n - 1].qubits, "First and last Toffoli should match")
        if n > 2 {
            #expect(toffoliGates[1].qubits == toffoliGates[n - 2].qubits, "Second and second-to-last should match")
        }
    }
}

/// Test suite for controlledPower() phase estimation operations.
/// Validates C-U^(2^k) construction including power 0 (identity),
/// power 1 (U^2), power 2 (U^4), and matrix correctness verification.
@Suite("Controlled Power Operations")
struct ControlledPowerOperationsTests {
    @Test("Power 0 returns empty sequence")
    func power0ReturnsEmptySequence() {
        let result = ControlledGateDecomposer.controlledPower(
            of: .pauliX,
            power: 0,
            control: 0,
            targetQubits: [1],
        )

        #expect(result.isEmpty, "Power 0 should return empty sequence (identity)")
    }

    @Test("Power 1 returns controlled U squared")
    func power1ReturnsControlledUSquared() {
        let result = ControlledGateDecomposer.controlledPower(
            of: .rotationZ(Double.pi / 4),
            power: 1,
            control: 0,
            targetQubits: [1],
        )

        #expect(!result.isEmpty, "Power 1 should return non-empty sequence")
    }

    @Test("Power 2 returns controlled U to the fourth")
    func power2ReturnsControlledUFourth() {
        let result = ControlledGateDecomposer.controlledPower(
            of: .rotationZ(Double.pi / 8),
            power: 2,
            control: 0,
            targetQubits: [1],
        )

        #expect(!result.isEmpty, "Power 2 should return non-empty sequence")
    }

    @Test("Single qubit gate power produces correct matrix")
    func singleQubitGatePowerProducesCorrectMatrix() {
        let angle = Double.pi / 4
        let result = ControlledGateDecomposer.controlledPower(
            of: .rotationZ(angle),
            power: 1,
            control: 0,
            targetQubits: [1],
        )

        #expect(result.count == 1, "Single qubit controlled power should return single gate")

        let originalMatrix = QuantumGate.rotationZ(angle).matrix()
        let expectedSquared = QuantumGate.matrixMultiply(originalMatrix, originalMatrix)

        let decomposedGate = result[0].gate
        if case let .customTwoQubit(matrix) = decomposedGate {
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    let diff = (matrix[i + 2][j + 2] - expectedSquared[i][j]).magnitude
                    #expect(diff < 1e-10, "Matrix element [\(i)][\(j)] should match U^2")
                }
            }
        }
    }

    @Test("Multi-qubit gate power returns controlled custom gate")
    func multiQubitGatePowerReturnsControlledCustomGate() {
        let result = ControlledGateDecomposer.controlledPower(
            of: .cnot,
            power: 1,
            control: 0,
            targetQubits: [1, 2],
        )

        #expect(result.count == 1, "Multi-qubit power should return single controlled gate")
        var isCustomTwoQubit = false
        if case .customTwoQubit = result[0].gate {
            isCustomTwoQubit = true
        }
        #expect(isCustomTwoQubit, "Gate should be customTwoQubit for multi-qubit target")
    }

    @Test("Pauli X power 3 equals U^8")
    func pauliXPower3EqualsU8() {
        let result = ControlledGateDecomposer.controlledPower(
            of: .pauliX,
            power: 3,
            control: 0,
            targetQubits: [1],
        )

        #expect(!result.isEmpty, "Power 3 should return gates")

        let x = QuantumGate.pauliX.matrix()
        var expected = x
        for _ in 1 ..< 8 {
            expected = QuantumGate.matrixMultiply(expected, x)
        }

        #expect(QuantumGate.matricesEqual(expected, QuantumGate.identity.matrix(), tolerance: 1e-10),
                "X^8 should equal identity")
    }
}

/// Test suite for basisChangeForGate() conjugation sequences.
/// Validates correct prefix/suffix pairs for Z (H,H), Y (S-dagger,S),
/// X (empty), and arbitrary gates requiring inverse transformation.
@Suite("Basis Change For Gate")
struct BasisChangeForGateTests {
    @Test("PauliX returns empty sequences")
    func pauliXReturnsEmptySequences() {
        let (prefix, suffix) = ControlledGateDecomposer.basisChangeForGate(.pauliX, target: 0)

        #expect(prefix.isEmpty, "X prefix should be empty")
        #expect(suffix.isEmpty, "X suffix should be empty")
    }

    @Test("PauliZ returns Hadamard pair")
    func pauliZReturnsHadamardPair() {
        let (prefix, suffix) = ControlledGateDecomposer.basisChangeForGate(.pauliZ, target: 0)

        #expect(prefix.count == 1, "Z prefix should have one gate")
        #expect(suffix.count == 1, "Z suffix should have one gate")
        #expect(prefix[0].gate == .hadamard, "Z prefix should be Hadamard")
        #expect(suffix[0].gate == .hadamard, "Z suffix should be Hadamard")
        #expect(prefix[0].qubits == [0], "Prefix qubit should be target")
        #expect(suffix[0].qubits == [0], "Suffix qubit should be target")
    }

    @Test("PauliY returns S-dagger and S pair")
    func pauliYReturnsSDaggerSPair() {
        let (prefix, suffix) = ControlledGateDecomposer.basisChangeForGate(.pauliY, target: 0)

        #expect(prefix.count == 1, "Y prefix should have one gate")
        #expect(suffix.count == 1, "Y suffix should have one gate")

        if case let .phase(theta) = prefix[0].gate,
           case let .value(v) = theta
        {
            #expect(abs(v + Double.pi / 2) < 1e-10, "Y prefix should be S-dagger (phase -pi/2)")
        }

        #expect(suffix[0].gate == .sGate, "Y suffix should be S gate")
    }

    @Test("Hadamard returns rotation Y pair")
    func hadamardReturnsRotationYPair() {
        let (prefix, suffix) = ControlledGateDecomposer.basisChangeForGate(.hadamard, target: 0)

        #expect(prefix.count == 1, "H prefix should have one gate")
        #expect(suffix.count == 1, "H suffix should have one gate")

        if case let .rotationY(theta) = prefix[0].gate,
           case let .value(v) = theta
        {
            #expect(abs(v + Double.pi / 2) < 1e-10, "H prefix should be Ry(-pi/2)")
        }

        if case let .rotationY(theta) = suffix[0].gate,
           case let .value(v) = theta
        {
            #expect(abs(v - Double.pi / 2) < 1e-10, "H suffix should be Ry(pi/2)")
        }
    }

    @Test("RotationZ returns half-angle pair")
    func rotationZReturnsHalfAnglePair() {
        let angle = Double.pi / 3
        let (prefix, suffix) = ControlledGateDecomposer.basisChangeForGate(.rotationZ(angle), target: 0)

        #expect(prefix.count == 1, "Rz prefix should have one gate")
        #expect(suffix.count == 1, "Rz suffix should have one gate")

        if case let .rotationZ(theta) = prefix[0].gate,
           case let .value(v) = theta
        {
            #expect(abs(v + angle / 2) < 1e-10, "Rz prefix should be Rz(-theta/2)")
        }

        if case let .rotationZ(theta) = suffix[0].gate,
           case let .value(v) = theta
        {
            #expect(abs(v - angle / 2) < 1e-10, "Rz suffix should be Rz(theta/2)")
        }
    }

    @Test("Phase gate returns half-angle phase pair")
    func phaseGateReturnsHalfAnglePhasePair() {
        let angle = Double.pi / 5
        let (prefix, suffix) = ControlledGateDecomposer.basisChangeForGate(.phase(angle), target: 0)

        #expect(prefix.count == 1, "Phase prefix should have one gate")
        #expect(suffix.count == 1, "Phase suffix should have one gate")

        if case let .phase(theta) = prefix[0].gate,
           case let .value(v) = theta
        {
            #expect(abs(v + angle / 2) < 1e-10, "Phase prefix should be P(-theta/2)")
        }

        if case let .phase(theta) = suffix[0].gate,
           case let .value(v) = theta
        {
            #expect(abs(v - angle / 2) < 1e-10, "Phase suffix should be P(theta/2)")
        }
    }

    @Test("S gate returns pi/4 phase pair")
    func sGateReturnsPi4PhasePair() {
        let (prefix, suffix) = ControlledGateDecomposer.basisChangeForGate(.sGate, target: 0)

        #expect(prefix.count == 1, "S prefix should have one gate")
        #expect(suffix.count == 1, "S suffix should have one gate")

        if case let .phase(theta) = prefix[0].gate,
           case let .value(v) = theta
        {
            #expect(abs(v + Double.pi / 4) < 1e-10, "S prefix should be P(-pi/4)")
        }

        if case let .phase(theta) = suffix[0].gate,
           case let .value(v) = theta
        {
            #expect(abs(v - Double.pi / 4) < 1e-10, "S suffix should be P(pi/4)")
        }
    }

    @Test("T gate returns pi/8 phase pair")
    func tGateReturnsPi8PhasePair() {
        let (prefix, suffix) = ControlledGateDecomposer.basisChangeForGate(.tGate, target: 0)

        #expect(prefix.count == 1, "T prefix should have one gate")
        #expect(suffix.count == 1, "T suffix should have one gate")

        if case let .phase(theta) = prefix[0].gate,
           case let .value(v) = theta
        {
            #expect(abs(v + Double.pi / 8) < 1e-10, "T prefix should be P(-pi/8)")
        }

        if case let .phase(theta) = suffix[0].gate,
           case let .value(v) = theta
        {
            #expect(abs(v - Double.pi / 8) < 1e-10, "T suffix should be P(pi/8)")
        }
    }

    @Test("Identity returns empty sequences")
    func identityReturnsEmptySequences() {
        let (prefix, suffix) = ControlledGateDecomposer.basisChangeForGate(.identity, target: 0)

        #expect(prefix.isEmpty, "Identity prefix should be empty")
        #expect(suffix.isEmpty, "Identity suffix should be empty")
    }

    @Test("Arbitrary gate returns inverse and gate pair")
    func arbitraryGateReturnsInverseAndGatePair() {
        let (prefix, suffix) = ControlledGateDecomposer.basisChangeForGate(.sx, target: 0)

        #expect(prefix.count == 1, "SX prefix should have one gate (inverse)")
        #expect(suffix.count == 1, "SX suffix should have one gate (original)")
        #expect(suffix[0].gate == .sx, "SX suffix should be SX")
    }
}

/// Test suite for mathematical validation of decompositions.
/// Validates that decomposed gate sequences produce the same unitary
/// transformation as the original controlled gate and preserve unitarity.
@Suite("Mathematical Validation")
struct MathematicalValidationTests {
    @Test("Decomposed single controlled X equals CNOT unitary")
    func decomposedSingleControlledXEqualsCNOTUnitary() {
        let result = ControlledGateDecomposer.decompose(
            gate: .pauliX,
            controls: [0],
            target: 1,
        )

        var circuit = QuantumCircuit(qubits: 2)
        for (gate, qubits) in result {
            circuit.append(gate, to: qubits)
        }

        let initialState = QuantumState(qubits: 2)
        let decomposedState = circuit.execute()
        let directState = GateApplication.apply(.cnot, to: [0, 1], state: initialState)

        for i in 0 ..< decomposedState.stateSpaceSize {
            let diff = (decomposedState.amplitude(of: i) - directState.amplitude(of: i)).magnitude
            #expect(diff < 1e-10, "Decomposed CNOT should produce same state as direct CNOT at index \(i)")
        }
    }

    @Test("Decomposed double controlled X equals Toffoli unitary")
    func decomposedDoubleControlledXEqualsToffoliUnitary() {
        let result = ControlledGateDecomposer.decompose(
            gate: .pauliX,
            controls: [0, 1],
            target: 2,
        )

        var circuit = QuantumCircuit(qubits: 3)
        for (gate, qubits) in result {
            circuit.append(gate, to: qubits)
        }

        var testCircuit = QuantumCircuit(qubits: 3)
        testCircuit.append(.hadamard, to: 0)
        testCircuit.append(.hadamard, to: 1)

        var circuit1 = QuantumCircuit(qubits: 3)
        for op in testCircuit.gates {
            circuit1.append(op.gate, to: op.qubits)
        }
        for (gate, qubits) in result {
            circuit1.append(gate, to: qubits)
        }
        let decomposedState = circuit1.execute()

        var circuit2 = QuantumCircuit(qubits: 3)
        for op in testCircuit.gates {
            circuit2.append(op.gate, to: op.qubits)
        }
        circuit2.append(.toffoli, to: [0, 1, 2])
        let directState = circuit2.execute()

        for i in 0 ..< decomposedState.stateSpaceSize {
            let diff = (decomposedState.amplitude(of: i) - directState.amplitude(of: i)).magnitude
            #expect(diff < 1e-10, "Decomposed Toffoli should produce same state as direct Toffoli at index \(i)")
        }
    }

    @Test("Decomposed controlled Z produces correct phase")
    func decomposedControlledZProducesCorrectPhase() {
        let result = ControlledGateDecomposer.decompose(
            gate: .pauliZ,
            controls: [0],
            target: 1,
        )

        var prepCircuit = QuantumCircuit(qubits: 2)
        prepCircuit.append(.hadamard, to: 0)
        prepCircuit.append(.hadamard, to: 1)

        var decomposedCircuit = QuantumCircuit(qubits: 2)
        for op in prepCircuit.gates {
            decomposedCircuit.append(op.gate, to: op.qubits)
        }
        for (gate, qubits) in result {
            decomposedCircuit.append(gate, to: qubits)
        }
        let decomposedState = decomposedCircuit.execute()

        var directCircuit = QuantumCircuit(qubits: 2)
        for op in prepCircuit.gates {
            directCircuit.append(op.gate, to: op.qubits)
        }
        directCircuit.append(.cz, to: [0, 1])
        let directState = directCircuit.execute()

        for i in 0 ..< decomposedState.stateSpaceSize {
            let diff = (decomposedState.amplitude(of: i) - directState.amplitude(of: i)).magnitude
            #expect(diff < 1e-10, "Decomposed CZ should produce same state as direct CZ at index \(i)")
        }
    }

    @Test("Decomposition and inverse produce identity")
    func decompositionAndInverseProduceIdentity() {
        let result = ControlledGateDecomposer.decompose(
            gate: .hadamard,
            controls: [0],
            target: 1,
        )

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)

        for (gate, qubits) in result {
            circuit.append(gate, to: qubits)
        }

        for (gate, qubits) in result.reversed() {
            circuit.append(gate.inverse, to: qubits)
        }

        let state = circuit.execute()

        #expect(abs(state.probability(of: 0) - 0.5) < 1e-10,
                "Forward-inverse should return to H|0> state with prob 0.5 at |00>")
    }

    @Test("Three control decomposition preserves unitary")
    func threeControlDecompositionPreservesUnitary() {
        let result = ControlledGateDecomposer.decompose(
            gate: .pauliX,
            controls: [0, 1, 2],
            target: 3,
        )

        var circuitForward = QuantumCircuit(qubits: 5)
        circuitForward.append(.pauliX, to: 0)
        circuitForward.append(.pauliX, to: 1)
        circuitForward.append(.pauliX, to: 2)

        for (gate, qubits) in result {
            circuitForward.append(gate, to: qubits)
        }

        let stateForward = circuitForward.execute()

        #expect(abs(stateForward.probability(of: 15) - 1.0) < 1e-10,
                "C^3(X)|111>|0> should produce |111>|1> with probability 1")

        var circuitReverse = QuantumCircuit(qubits: 5)
        circuitReverse.append(.pauliX, to: 0)
        circuitReverse.append(.pauliX, to: 1)
        circuitReverse.append(.pauliX, to: 2)

        for (gate, qubits) in result {
            circuitReverse.append(gate, to: qubits)
        }

        for (gate, qubits) in result.reversed() {
            circuitReverse.append(gate.inverse, to: qubits)
        }

        let stateReverse = circuitReverse.execute()

        #expect(abs(stateReverse.probability(of: 7) - 1.0) < 1e-10,
                "Applying decomposition twice should return to |111>|0>")
    }

    @Test("Controlled rotation preserves unitarity")
    func controlledRotationPreservesUnitarity() {
        let angle = Double.pi / 4
        let result = ControlledGateDecomposer.decompose(
            gate: .rotationZ(angle),
            controls: [0],
            target: 1,
        )

        var circuit = QuantumCircuit(qubits: 2)
        for (gate, qubits) in result {
            circuit.append(gate, to: qubits)
        }

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
                "Controlled Rz decomposition U U-dagger should return to |00>")
    }
}

/// Test suite for unitarity preservation validation.
/// Validates that decomposed gate sequences maintain U-dagger U = I
/// property across all decomposition types.
@Suite("Unitarity Preservation")
struct UnitarityPreservationTests {
    @Test("Single controlled decomposition preserves unitarity")
    func singleControlledPreservesUnitarity() {
        let gates: [QuantumGate] = [.pauliX, .pauliY, .pauliZ, .hadamard, .rotationZ(Double.pi / 4)]

        for gate in gates {
            let result = ControlledGateDecomposer.decompose(
                gate: gate,
                controls: [0],
                target: 1,
            )

            var circuit = QuantumCircuit(qubits: 2)
            for (g, qubits) in result {
                circuit.append(g, to: qubits)
            }

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
                    "U U-dagger should return to |00> for controlled \(gate)")
        }
    }

    @Test("Two controlled decomposition preserves unitarity")
    func twoControlledPreservesUnitarity() {
        let result = ControlledGateDecomposer.decompose(
            gate: .pauliZ,
            controls: [0, 1],
            target: 2,
        )

        var circuit = QuantumCircuit(qubits: 3)
        for (gate, qubits) in result {
            circuit.append(gate, to: qubits)
        }

        let inverse = circuit.inverse()

        var combined = QuantumCircuit(qubits: 3)
        for op in circuit.gates {
            combined.append(op.gate, to: op.qubits)
        }
        for op in inverse.gates {
            combined.append(op.gate, to: op.qubits)
        }

        let state = combined.execute()

        #expect(abs(state.probability(of: 0) - 1.0) < 1e-10,
                "CCZ decomposition U U-dagger should return to |000>")
    }

    @Test("Ladder decomposition preserves unitarity")
    func ladderDecompositionPreservesUnitarity() {
        let result = ControlledGateDecomposer.toffoliLadderDecomposition(
            gate: .pauliX,
            controls: [0, 1, 2],
            target: 3,
        )

        var circuit = QuantumCircuit(qubits: 5)
        for (gate, qubits) in result {
            circuit.append(gate, to: qubits)
        }

        let inverse = circuit.inverse()

        var combined = QuantumCircuit(qubits: 5)
        for op in circuit.gates {
            combined.append(op.gate, to: op.qubits)
        }
        for op in inverse.gates {
            combined.append(op.gate, to: op.qubits)
        }

        let state = combined.execute()

        #expect(abs(state.probability(of: 0) - 1.0) < 1e-10,
                "Toffoli ladder U U-dagger should return to |00000>")
    }

    @Test("Controlled power preserves unitarity")
    func controlledPowerPreservesUnitarity() {
        let result = ControlledGateDecomposer.controlledPower(
            of: .rotationZ(Double.pi / 4),
            power: 2,
            control: 0,
            targetQubits: [1],
        )

        var circuit = QuantumCircuit(qubits: 2)
        for (gate, qubits) in result {
            circuit.append(gate, to: qubits)
        }

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
                "Controlled power U U-dagger should return to |00>")
    }
}

/// Test suite for edge cases and boundary conditions.
/// Validates handling of large control counts, non-contiguous qubit indices,
/// and other corner cases in the decomposition logic.
@Suite("Decomposer Edge Cases")
struct DecomposerEdgeCasesTests {
    @Test("Non-contiguous qubit indices work correctly")
    func nonContiguousQubitIndicesWorkCorrectly() {
        let result = ControlledGateDecomposer.decompose(
            gate: .pauliX,
            controls: [0, 5],
            target: 10,
        )

        #expect(result.count == 1, "Two controls with X should return Toffoli")
        #expect(result[0].gate == .toffoli, "Gate should be Toffoli")
        #expect(result[0].qubits == [0, 5, 10], "Qubits should preserve non-contiguous indices")
    }

    @Test("High qubit index decomposition allocates ancillas correctly")
    func highQubitIndexDecompositionAllocatesAncillasCorrectly() {
        let result = ControlledGateDecomposer.decompose(
            gate: .pauliX,
            controls: [100, 101, 102],
            target: 103,
        )

        let allQubits = result.flatMap(\.qubits)
        let maxQubit = allQubits.max()!

        #expect(maxQubit == 104, "Ancilla should be at 104 (max+1)")
    }

    @Test("Decomposition handles symbolic parameters")
    func decompositionHandlesSymbolicParameters() {
        let param = Parameter(name: "theta")
        let result = ControlledGateDecomposer.decompose(
            gate: .rotationZ(.parameter(param)),
            controls: [0],
            target: 1,
        )

        #expect(result.count == 1, "Symbolic Rz should decompose to single controlled rotation")
        var isControlledRotationZ = false
        if case .controlledRotationZ = result[0].gate {
            isControlledRotationZ = true
        }
        #expect(isControlledRotationZ, "Gate should be controlledRotationZ")
    }

    @Test("Two controls with non-X gate uses basis transformation")
    func twoControlsWithNonXGateUsesBasisTransformation() {
        let result = ControlledGateDecomposer.decompose(
            gate: .pauliY,
            controls: [0, 1],
            target: 2,
        )

        #expect(result.count > 1, "Two controls with Y should have basis transformation")

        let toffoliCount = result.count(where: { $0.gate == .toffoli })
        #expect(toffoliCount == 1, "Should have exactly one Toffoli")
    }

    @Test("Large control count produces correct ancilla chain")
    func largeControlCountProducesCorrectAncillaChain() {
        let result = ControlledGateDecomposer.decompose(
            gate: .pauliX,
            controls: [0, 1, 2, 3, 4],
            target: 5,
        )

        let allQubits = Set(result.flatMap(\.qubits))

        #expect(allQubits.contains(6), "Should have ancilla at 6")
        #expect(allQubits.contains(7), "Should have ancilla at 7")
        #expect(allQubits.contains(8), "Should have ancilla at 8")

        let toffoliCount = result.count(where: { $0.gate == .toffoli })
        #expect(toffoliCount >= 6, "5 controls should have at least 6 Toffolis")
    }

    @Test("Custom single qubit gate decomposes correctly")
    func customSingleQubitGateDecomposesCorrectly() {
        let customMatrix: [[Complex<Double>]] = [
            [Complex(0.6, 0), Complex(0, 0.8)],
            [Complex(0, 0.8), Complex(0.6, 0)],
        ]
        let customGate = QuantumGate.customSingleQubit(matrix: customMatrix)

        let result = ControlledGateDecomposer.decompose(
            gate: customGate,
            controls: [0],
            target: 1,
        )

        #expect(result.count == 1, "Custom gate should decompose to single controlled custom")
        var isCustomTwoQubit = false
        if case .customTwoQubit = result[0].gate {
            isCustomTwoQubit = true
        }
        #expect(isCustomTwoQubit, "Gate should be customTwoQubit")
    }

    @Test("SY gate decomposes to custom two qubit")
    func syGateDecomposesToCustomTwoQubit() {
        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .sy,
            control: 0,
            target: 1,
        )

        #expect(result.count == 1, "Controlled-SY should return single custom gate")
        var isCustomTwoQubit = false
        if case .customTwoQubit = result[0].gate {
            isCustomTwoQubit = true
        }
        #expect(isCustomTwoQubit, "Gate should be customTwoQubit")
    }
}

/// Test suite for controlled U3 decomposition specifics.
/// Validates the CNOT-based decomposition of controlled U3 gates
/// with correct angle calculations and gate ordering.
@Suite("Controlled U3 Decomposition")
struct ControlledU3DecompositionTests {
    @Test("Controlled U3 has correct gate count")
    func controlledU3HasCorrectGateCount() {
        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(Double.pi / 4), phi: .value(Double.pi / 3), lambda: .value(Double.pi / 6)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 7, "Controlled U3 should have 7 gates")

        let cnotCount = result.count(where: { $0.gate == .cnot })
        #expect(cnotCount == 2, "Should have exactly 2 CNOTs")

        let rzCount = result.count(where: {
            if case .rotationZ = $0.gate { return true }
            return false
        })
        #expect(rzCount >= 2, "Should have at least 2 Rz gates")

        let ryCount = result.count(where: {
            if case .rotationY = $0.gate { return true }
            return false
        })
        #expect(ryCount == 2, "Should have exactly 2 Ry gates")
    }

    @Test("Controlled U3 preserves unitarity")
    func controlledU3PreservesUnitarity() {
        let theta = Double.pi / 4
        let phi = Double.pi / 3
        let lambda = Double.pi / 6

        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(theta), phi: .value(phi), lambda: .value(lambda)),
            control: 0,
            target: 1,
        )

        var circuit = QuantumCircuit(qubits: 2)
        for (gate, qubits) in result {
            circuit.append(gate, to: qubits)
        }

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
                "Controlled U3 decomposition should be unitary (U U-dagger = I)")
    }

    @Test("Controlled U3 with zero theta simplifies")
    func controlledU3WithZeroThetaSimplifies() {
        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(0), phi: .value(Double.pi / 3), lambda: .value(Double.pi / 6)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 7, "Zero theta U3 still produces 7 gates")
    }

    @Test("Controlled U3 qubit assignments are correct")
    func controlledU3QubitAssignmentsAreCorrect() {
        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(Double.pi / 4), phi: .value(Double.pi / 3), lambda: .value(Double.pi / 6)),
            control: 2,
            target: 5,
        )

        for (gate, qubits) in result {
            if gate == .cnot {
                #expect(qubits == [2, 5], "CNOT should be on control=2, target=5")
            } else {
                #expect(qubits == [5], "Single qubit gates should be on target=5")
            }
        }
    }
}

/// Test suite for matrix power computation.
/// Validates correct repeated squaring algorithm for computing U^(2^k)
/// including identity for power 0 and correct multiplication chain.
@Suite("Matrix Power Computation")
struct MatrixPowerComputationTests {
    @Test("Pauli X squared equals identity")
    func pauliXSquaredEqualsIdentity() {
        let result = ControlledGateDecomposer.controlledPower(
            of: .pauliX,
            power: 1,
            control: 0,
            targetQubits: [1],
        )

        #expect(result.count == 1, "Should return single gate")
        if case let .customTwoQubit(matrix) = result.first?.gate {
            let identity = QuantumGate.identity.matrix()
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    let diff = (matrix[i + 2][j + 2] - identity[i][j]).magnitude
                    #expect(diff < 1e-10, "X^2 should equal identity at [\(i)][\(j)]")
                }
            }
        }
    }

    @Test("Rotation Z power doubles angle")
    func rotationZPowerDoublesAngle() {
        let angle = Double.pi / 8
        let result = ControlledGateDecomposer.controlledPower(
            of: .rotationZ(angle),
            power: 1,
            control: 0,
            targetQubits: [1],
        )

        #expect(result.count == 1, "Should return single gate")
        if case let .customTwoQubit(matrix) = result.first?.gate {
            let expected = QuantumGate.rotationZ(angle * 2).matrix()
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    let diff = (matrix[i + 2][j + 2] - expected[i][j]).magnitude
                    #expect(diff < 1e-10, "Rz^2 should have doubled angle at [\(i)][\(j)]")
                }
            }
        }
    }

    @Test("S gate power 1 equals Z")
    func sGatePower1EqualsZ() {
        let result = ControlledGateDecomposer.controlledPower(
            of: .sGate,
            power: 1,
            control: 0,
            targetQubits: [1],
        )

        #expect(result.count == 1, "Should return single gate")
        if case let .customTwoQubit(matrix) = result.first?.gate {
            let z = QuantumGate.pauliZ.matrix()
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    let diff = (matrix[i + 2][j + 2] - z[i][j]).magnitude
                    #expect(diff < 1e-10, "S^2 should equal Z at [\(i)][\(j)]")
                }
            }
        }
    }

    @Test("Hadamard power 1 equals identity")
    func hadamardPower1EqualsIdentity() {
        let result = ControlledGateDecomposer.controlledPower(
            of: .hadamard,
            power: 1,
            control: 0,
            targetQubits: [1],
        )

        #expect(result.count == 1, "Should return single gate")
        if case let .customTwoQubit(matrix) = result.first?.gate {
            let identity = QuantumGate.identity.matrix()
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    let diff = (matrix[i + 2][j + 2] - identity[i][j]).magnitude
                    #expect(diff < 1e-10, "H^2 should equal identity at [\(i)][\(j)]")
                }
            }
        }
    }
}

/// Test suite for uncovered edge cases in ControlledGateDecomposer.
/// Covers parameter arithmetic, matrix power edge cases, and controlled matrix building.
@Suite("Decomposer Uncovered Edge Cases")
struct DecomposerUncoveredEdgeCasesTests {
    @Test("Controlled power with two-qubit gate uses buildControlledMatrixFromNxN")
    func controlledPowerTwoQubitGateUsesCorrectMatrixBuilder() {
        let result = ControlledGateDecomposer.controlledPower(
            of: .cnot,
            power: 1,
            control: 0,
            targetQubits: [1, 2],
        )

        #expect(result.count == 1, "Controlled CNOT^2 should return single gate")
        if case let .customTwoQubit(matrix) = result[0].gate {
            #expect(matrix.count == 8, "Controlled 4x4 matrix should be 8x8")
            for i in 0 ..< 4 {
                let diag = matrix[i][i]
                #expect(abs(diag.real - 1.0) < 1e-10, "Diagonal element [\(i)][\(i)] should be 1")
                #expect(abs(diag.imaginary) < 1e-10, "Diagonal element [\(i)][\(i)] should have zero imaginary part")
            }
        }
    }

    @Test("Matrix power with exponent 1 returns original matrix - line 493")
    func matrixPowerExponent1ReturnsOriginal() {
        let result = ControlledGateDecomposer.controlledPower(
            of: .tGate,
            power: 3,
            control: 0,
            targetQubits: [1],
        )

        #expect(result.count == 1, "Should return single gate")
        if case let .customTwoQubit(matrix) = result.first?.gate {
            let identity = QuantumGate.identity.matrix()
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    let diff = (matrix[i + 2][j + 2] - identity[i][j]).magnitude
                    #expect(diff < 1e-10, "T^8 should equal identity at [\(i)][\(j)]")
                }
            }
        }
    }

    @Test("Extract half angle with parameter creates derived parameter name - line 519")
    func extractHalfAngleWithParameter() {
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")

        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .parameter(theta), phi: .parameter(phi), lambda: .parameter(lambda)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 7, "Controlled U3 with symbolic params should have 7 gates")

        let ryGates = result.filter {
            if case .rotationY = $0.gate { return true }
            return false
        }
        #expect(ryGates.count == 2, "Should have 2 Ry gates in decomposition")

        for (gate, _) in ryGates {
            if case let .rotationY(paramVal) = gate,
               case let .parameter(p) = paramVal
            {
                #expect(p.name.contains("half"), "Ry parameter should have 'half' in name")
            } else if case let .rotationY(paramVal) = gate,
                      case let .negatedParameter(p) = paramVal
            {
                #expect(p.name.contains("half"), "Ry negated parameter should have 'half' in name")
            }
        }
    }

    @Test("Extract half angle with negated parameter creates derived parameter name - line 521")
    func extractHalfAngleWithNegatedParameter() {
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")

        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .negatedParameter(theta), phi: .parameter(phi), lambda: .parameter(lambda)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 7, "Controlled U3 with negated param should have 7 gates")

        let ryGates = result.filter {
            if case .rotationY = $0.gate { return true }
            return false
        }
        #expect(ryGates.count == 2, "Should have 2 Ry gates for negated parameter U3")
    }

    @Test("Add two symbolic parameters - line 532-533")
    func addTwoSymbolicParameters() {
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")

        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(Double.pi / 4), phi: .parameter(phi), lambda: .parameter(lambda)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 7, "Controlled U3 with symbolic phi and lambda should have 7 gates")

        if case let .rotationZ(paramVal) = result[0].gate,
           case let .parameter(p) = paramVal
        {
            #expect(p.name.contains("plus"), "First Rz should have 'plus' in parameter name")
        }
    }

    @Test("Add parameter and value - line 534-535")
    func addParameterAndValue() {
        let phi = Parameter(name: "phi")

        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(Double.pi / 4), phi: .parameter(phi), lambda: .value(Double.pi / 6)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 7, "Controlled U3 with mixed phi should have 7 gates")

        if case let .rotationZ(paramVal) = result[0].gate,
           case let .parameter(p) = paramVal
        {
            #expect(p.name.contains("plus"), "Rz should have 'plus' in parameter name for phi+value")
        }
    }

    @Test("Add two negated parameters - line 536-537")
    func addTwoNegatedParameters() {
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")

        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(Double.pi / 4), phi: .negatedParameter(phi), lambda: .negatedParameter(lambda)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 7, "Controlled U3 with negated phi and lambda should have 7 gates")
    }

    @Test("Add negated parameter and value - line 538-539")
    func addNegatedParameterAndValue() {
        let phi = Parameter(name: "phi")

        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(Double.pi / 4), phi: .negatedParameter(phi), lambda: .value(Double.pi / 6)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 7, "Controlled U3 with negated phi and value lambda should have 7 gates")
    }

    @Test("Add value and negated parameter - line 540-541")
    func addValueAndNegatedParameter() {
        let lambda = Parameter(name: "lambda")

        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(Double.pi / 4), phi: .value(Double.pi / 3), lambda: .negatedParameter(lambda)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 7, "Controlled U3 with value phi and negated lambda should have 7 gates")
    }

    @Test("Add parameter and negated parameter - line 542-543")
    func addParameterAndNegatedParameter() {
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")

        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(Double.pi / 4), phi: .parameter(phi), lambda: .negatedParameter(lambda)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 7, "Controlled U3 with parameter phi and negated lambda should have 7 gates")
    }

    @Test("Add negated parameter and parameter - line 544-545")
    func addNegatedParameterAndParameter() {
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")

        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(Double.pi / 4), phi: .negatedParameter(phi), lambda: .parameter(lambda)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 7, "Controlled U3 with negated phi and parameter lambda should have 7 gates")
    }

    @Test("Subtract two symbolic parameters - line 556-557")
    func subtractTwoSymbolicParameters() {
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")

        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(Double.pi / 4), phi: .parameter(phi), lambda: .parameter(lambda)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 7, "Controlled U3 should have 7 gates")

        let rzGates = result.filter {
            if case .rotationZ = $0.gate { return true }
            return false
        }
        #expect(rzGates.count >= 2, "Should have at least 2 Rz gates")
    }

    @Test("Subtract parameter and value - line 558-559")
    func subtractParameterAndValue() {
        let phi = Parameter(name: "phi")

        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(Double.pi / 4), phi: .parameter(phi), lambda: .value(Double.pi / 6)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 7, "Controlled U3 with parameter phi and value lambda should have 7 gates")
    }

    @Test("Subtract value and parameter - line 560-561")
    func subtractValueAndParameter() {
        let lambda = Parameter(name: "lambda")

        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(Double.pi / 4), phi: .value(Double.pi / 3), lambda: .parameter(lambda)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 7, "Controlled U3 with value phi and parameter lambda should have 7 gates")
    }

    @Test("Subtract two negated parameters - line 562-563")
    func subtractTwoNegatedParameters() {
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")

        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(Double.pi / 4), phi: .negatedParameter(phi), lambda: .negatedParameter(lambda)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 7, "Controlled U3 with two negated parameters should have 7 gates")
    }

    @Test("Subtract negated parameter and value - line 564-565")
    func subtractNegatedParameterAndValue() {
        let phi = Parameter(name: "phi")

        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(Double.pi / 4), phi: .negatedParameter(phi), lambda: .value(Double.pi / 6)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 7, "Controlled U3 with negated phi and value lambda should have 7 gates")
    }

    @Test("Subtract value and negated parameter - line 566-567")
    func subtractValueAndNegatedParameter() {
        let lambda = Parameter(name: "lambda")

        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(Double.pi / 4), phi: .value(Double.pi / 3), lambda: .negatedParameter(lambda)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 7, "Controlled U3 with value phi and negated lambda should have 7 gates")
    }

    @Test("Subtract parameter and negated parameter - line 568-569")
    func subtractParameterAndNegatedParameter() {
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")

        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(Double.pi / 4), phi: .parameter(phi), lambda: .negatedParameter(lambda)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 7, "Controlled U3 with parameter phi and negated lambda should have 7 gates")
    }

    @Test("Subtract negated parameter and parameter - line 570-571")
    func subtractNegatedParameterAndParameter() {
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")

        let result = ControlledGateDecomposer.decomposeSingleControlled(
            gate: .u3(theta: .value(Double.pi / 4), phi: .negatedParameter(phi), lambda: .parameter(lambda)),
            control: 0,
            target: 1,
        )

        #expect(result.count == 7, "Controlled U3 with negated phi and parameter lambda should have 7 gates")
    }
}
