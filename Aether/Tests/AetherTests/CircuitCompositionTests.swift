// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Validates repeated() circuit composition primitive.
/// Tests count=1 identity, count=3 tripling, qubit preservation,
/// and state equivalence via execute().
@Suite("CircuitComposition: repeated(_:)")
struct RepeatedTests {
    @Test("repeated(1) returns circuit with same operation count")
    func repeatedOnceMatchesOriginalCount() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 1)
        let repeated = circuit.repeated(1)
        #expect(repeated.count == circuit.count, "repeated(1) should have same operation count as original")
    }

    @Test("repeated(1) preserves qubit count")
    func repeatedOncePreservesQubits() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        let repeated = circuit.repeated(1)
        #expect(repeated.qubits == 3, "repeated(1) should preserve qubit count of 3")
    }

    @Test("repeated(3) triples operation count")
    func repeatedThreeTripleOps() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        let repeated = circuit.repeated(3)
        #expect(repeated.count == 6, "repeated(3) of 2-op circuit should have 6 operations")
    }

    @Test("repeated(3) preserves qubit count")
    func repeatedThreePreservesQubits() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let repeated = circuit.repeated(3)
        #expect(repeated.qubits == 2, "repeated(3) should preserve qubit count of 2")
    }

    @Test("repeated(1) produces equivalent state via execute()")
    func repeatedOnceStateEquivalence() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        let originalState = circuit.execute()
        let repeatedState = circuit.repeated(1).execute()
        for i in 0 ..< originalState.stateSpaceSize {
            #expect(
                abs(originalState.amplitudes[i].real - repeatedState.amplitudes[i].real) < 1e-10,
                "Real part of amplitude \(i) should match for repeated(1)",
            )
            #expect(
                abs(originalState.amplitudes[i].imaginary - repeatedState.amplitudes[i].imaginary) < 1e-10,
                "Imaginary part of amplitude \(i) should match for repeated(1)",
            )
        }
    }

    @Test("repeated(2) of self-inverse gate returns to ground state")
    func repeatedTwoSelfInverse() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        let doubled = circuit.repeated(2)
        let state = doubled.execute()
        #expect(abs(state.probability(of: 0) - 1.0) < 1e-10, "X applied twice should return to |0> with probability 1")
    }

    @Test("repeated preserves qubit labels")
    func repeatedPreservesLabels() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.label(qubit: 0, "control")
        circuit.label(qubit: 1, "target")
        circuit.append(.hadamard, to: 0)
        let repeated = circuit.repeated(2)
        #expect(repeated.qubitLabels[0] == "control", "repeated should preserve label for qubit 0")
        #expect(repeated.qubitLabels[1] == "target", "repeated should preserve label for qubit 1")
    }
}

/// Validates power() circuit composition primitive.
/// Tests exponent=0 identity, exponent=1 same circuit,
/// exponent=2 matrix path, and state vector correctness.
@Suite("CircuitComposition: power(_:)")
struct PowerTests {
    @Test("power(0) returns empty circuit (identity)")
    func powerZeroIsIdentity() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        let powered = circuit.power(0)
        #expect(powered.count == 0, "power(0) should produce circuit with no operations")
        #expect(powered.qubits == 2, "power(0) should preserve qubit count")
    }

    @Test("power(0) produces ground state when executed")
    func powerZeroExecutesToGroundState() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let state = circuit.power(0).execute()
        #expect(abs(state.probability(of: 0) - 1.0) < 1e-10, "power(0) should leave state as |00> with probability 1")
    }

    @Test("power(1) returns same circuit")
    func powerOneReturnsSame() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        let powered = circuit.power(1)
        #expect(powered.count == circuit.count, "power(1) should have same operation count as original")
    }

    @Test("power(1) produces equivalent state")
    func powerOneStateEquivalence() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 1)
        let originalState = circuit.execute()
        let poweredState = circuit.power(1).execute()
        for i in 0 ..< originalState.stateSpaceSize {
            #expect(
                abs(originalState.amplitudes[i].real - poweredState.amplitudes[i].real) < 1e-10,
                "Real part of amplitude \(i) should match for power(1)",
            )
            #expect(
                abs(originalState.amplitudes[i].imaginary - poweredState.amplitudes[i].imaginary) < 1e-10,
                "Imaginary part of amplitude \(i) should match for power(1)",
            )
        }
    }

    @Test("power(2) of X gate produces identity (matrix path)")
    func powerTwoXGateIsIdentity() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        let powered = circuit.power(2)
        let state = powered.execute()
        #expect(abs(state.probability(of: 0) - 1.0) < 1e-10, "X^2 should be identity, leaving |0> with probability 1")
    }

    @Test("power(2) of Hadamard produces identity via matrix path")
    func powerTwoHadamardIsIdentity() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let powered = circuit.power(2)
        let state = powered.execute()
        #expect(abs(state.probability(of: 0) - 1.0) < 1e-10, "H^2 should be identity, leaving |0> with probability 1")
    }

    @Test("power(4) of S gate produces identity")
    func powerFourSGateIsIdentity() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.sGate, to: 0)
        let powered = circuit.power(4)
        let state = powered.execute()
        #expect(abs(state.probability(of: 0) - 1.0) < 1e-10, "S^4 should be identity, leaving |0> with probability 1")
    }

    @Test("power(0) preserves qubit labels")
    func powerZeroPreservesLabels() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.label(qubit: 0, "data")
        circuit.append(.hadamard, to: 0)
        let powered = circuit.power(0)
        #expect(powered.qubitLabels[0] == "data", "power(0) should preserve label for qubit 0")
    }

    @Test("power(2) two-qubit circuit via matrix path correctness")
    func powerTwoTwoQubitCircuit() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])
        let powered = circuit.power(2)
        let state = powered.execute()
        #expect(abs(state.probability(of: 0) - 1.0) < 1e-10, "CNOT^2 should be identity on |00>")
    }
}

/// Validates controlled(by:) circuit composition primitive.
/// Tests qubit count increase, control qubit addition,
/// and CNOT-equivalent behavior for controlled-X.
@Suite("CircuitComposition: controlled(by:)")
struct ControlledByTests {
    @Test("controlled(by:) increases qubit count by number of controls")
    func controlledIncreasesQubitCount() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        let controlled = circuit.controlled(by: [0])
        #expect(controlled.qubits == 2, "1-qubit circuit controlled by 1 qubit should have 2 qubits total")
    }

    @Test("controlled(by:) with two controls increases qubit count by 2")
    func controlledTwoControlsIncreasesQubitCount() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        let controlled = circuit.controlled(by: [0, 1])
        #expect(controlled.qubits == 3, "1-qubit circuit controlled by 2 qubits should have 3 qubits total")
    }

    @Test("controlled-X with control |0> leaves target unchanged")
    func controlledXControlZeroNoEffect() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        let controlled = circuit.controlled(by: [0])
        let state = controlled.execute()
        #expect(abs(state.probability(of: 0b00) - 1.0) < 1e-10, "Control qubit in |0> should leave target unchanged at |00>")
    }

    @Test("controlled-X matches CNOT behavior with control |1>")
    func controlledXMatchesCNOT() {
        var xCircuit = QuantumCircuit(qubits: 1)
        xCircuit.append(.pauliX, to: 0)
        let controlled = xCircuit.controlled(by: [0])

        var cnotCircuit = QuantumCircuit(qubits: 2)
        cnotCircuit.append(.pauliX, to: 0)
        cnotCircuit.append(.cnot, to: [0, 1])

        var setupAndControlled = QuantumCircuit(qubits: 2)
        setupAndControlled.append(.pauliX, to: 0)
        for op in controlled.operations {
            setupAndControlled.addOperation(op)
        }

        let cnotState = cnotCircuit.execute()
        let controlledState = setupAndControlled.execute()

        for i in 0 ..< cnotState.stateSpaceSize {
            #expect(
                abs(cnotState.amplitudes[i].magnitudeSquared - controlledState.amplitudes[i].magnitudeSquared) < 1e-10,
                "Controlled-X probabilities should match CNOT at index \(i)",
            )
        }
    }

    @Test("controlled preserves qubit labels with shift")
    func controlledPreservesLabelsShifted() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.label(qubit: 0, "target")
        circuit.append(.hadamard, to: 0)
        let controlled = circuit.controlled(by: [0])
        #expect(controlled.qubitLabels[1] == "target", "Original qubit label should shift to index 1")
    }

    @Test("controlled(by:) with 2 controls on multi-qubit gate produces correct 5-qubit circuit")
    func controlledTwoControlsOnMultiQubitGate() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])
        let controlled = circuit.controlled(by: [0, 1])
        #expect(controlled.qubits == 5, "2-qubit circuit controlled by 2 qubits needs 5 qubits (1 ancilla for Toffoli decomposition)")
        #expect(controlled.count > 0, "Controlled multi-qubit gate with 2 controls should produce operations")
        let state = controlled.execute()
        #expect(abs(state.probability(of: 0) - 1.0) < 1e-10, "All controls in |0> should leave state as |00000>")
    }

    @Test("controlled of Hadamard produces non-trivial circuit")
    func controlledHadamardNonTrivial() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let controlled = circuit.controlled(by: [0])
        #expect(controlled.qubits == 2, "Controlled-H should have 2 qubits")
        #expect(controlled.count > 0, "Controlled-H should have at least one operation")
    }
}

/// Validates + (series composition) operator.
/// Tests concatenation, qubit count as max, operation ordering,
/// and state equivalence for sequential execution.
@Suite("CircuitComposition: + operator")
struct PlusOperatorTests {
    @Test("a + b has operation count equal to sum of both")
    func plusPreservesOperationCount() {
        var a = QuantumCircuit(qubits: 2)
        a.append(.hadamard, to: 0)

        var b = QuantumCircuit(qubits: 2)
        b.append(.cnot, to: [0, 1])

        let combined = a + b
        #expect(combined.count == 2, "a(1 op) + b(1 op) should have 2 operations")
    }

    @Test("a + b qubit count is max of both")
    func plusQubitCountIsMax() {
        var a = QuantumCircuit(qubits: 2)
        a.append(.hadamard, to: 0)

        var b = QuantumCircuit(qubits: 3)
        b.append(.pauliX, to: 2)

        let combined = a + b
        #expect(combined.qubits == 3, "Qubit count should be max(2, 3) = 3")
    }

    @Test("a + b state equivalence with sequential execution")
    func plusStateEquivalence() {
        var a = QuantumCircuit(qubits: 2)
        a.append(.hadamard, to: 0)

        var b = QuantumCircuit(qubits: 2)
        b.append(.cnot, to: [0, 1])

        let combined = a + b
        let combinedState = combined.execute()

        let intermediateState = a.execute()
        let sequentialState = b.execute(on: intermediateState)

        for i in 0 ..< combinedState.stateSpaceSize {
            #expect(
                abs(combinedState.amplitudes[i].real - sequentialState.amplitudes[i].real) < 1e-10,
                "Real part of amplitude \(i) should match sequential execution",
            )
            #expect(
                abs(combinedState.amplitudes[i].imaginary - sequentialState.amplitudes[i].imaginary) < 1e-10,
                "Imaginary part of amplitude \(i) should match sequential execution",
            )
        }
    }

    @Test("empty + circuit equals that circuit")
    func plusEmptyLeftIdentity() {
        let empty = QuantumCircuit(qubits: 2)
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let combined = empty + circuit
        #expect(combined.count == 1, "empty + circuit should have same operation count as circuit")
    }

    @Test("circuit + empty equals that circuit")
    func plusEmptyRightIdentity() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let empty = QuantumCircuit(qubits: 2)
        let combined = circuit + empty
        #expect(combined.count == 1, "circuit + empty should have same operation count as circuit")
    }

    @Test("qubit labels merge with right-hand precedence")
    func plusLabelMerge() {
        var a = QuantumCircuit(qubits: 2)
        a.label(qubit: 0, "from_a")
        a.label(qubit: 1, "a_only")
        a.append(.hadamard, to: 0)

        var b = QuantumCircuit(qubits: 2)
        b.label(qubit: 0, "from_b")
        b.append(.pauliX, to: 0)

        let combined = a + b
        #expect(combined.qubitLabels[0] == "from_b", "Right-hand label should take precedence on collision")
        #expect(combined.qubitLabels[1] == "a_only", "Non-colliding left-hand label should be preserved")
    }
}

/// Validates tensor product operator for parallel circuit composition.
/// Tests qubit sum, operation independence, qubit shifting,
/// and state vector correctness.
@Suite("CircuitComposition: tensor product operator")
struct TensorProductTests {
    @Test("a tensor b qubit count is sum of both")
    func tensorQubitCountIsSum() {
        var a = QuantumCircuit(qubits: 2)
        a.append(.hadamard, to: 0)

        var b = QuantumCircuit(qubits: 1)
        b.append(.pauliX, to: 0)

        let tensor = a ⊗ b
        #expect(tensor.qubits == 3, "Qubit count should be 2 + 1 = 3")
    }

    @Test("a tensor b operation count is sum of both")
    func tensorOperationCountIsSum() {
        var a = QuantumCircuit(qubits: 1)
        a.append(.hadamard, to: 0)

        var b = QuantumCircuit(qubits: 1)
        b.append(.pauliX, to: 0)
        b.append(.pauliZ, to: 0)

        let tensor = a ⊗ b
        #expect(tensor.count == 3, "Operation count should be 1 + 2 = 3")
    }

    @Test("right-hand circuit qubit indices are shifted")
    func tensorShiftsRightQubits() {
        var a = QuantumCircuit(qubits: 2)
        a.append(.hadamard, to: 0)

        var b = QuantumCircuit(qubits: 1)
        b.append(.pauliX, to: 0)

        let tensor = a ⊗ b
        let lastOp = tensor.operations[tensor.count - 1]
        #expect(lastOp.qubits == [2], "Right-hand qubit 0 should shift to index 2 (lhs.qubits)")
    }

    @Test("left-hand circuit qubit indices are unchanged")
    func tensorPreservesLeftQubits() {
        var a = QuantumCircuit(qubits: 2)
        a.append(.hadamard, to: 0)
        a.append(.cnot, to: [0, 1])

        var b = QuantumCircuit(qubits: 1)
        b.append(.pauliX, to: 0)

        let tensor = a ⊗ b
        #expect(tensor.operations[0].qubits == [0], "Left-hand H qubit should remain at 0")
        #expect(tensor.operations[1].qubits == [0, 1], "Left-hand CNOT qubits should remain [0,1]")
    }

    @Test("tensor product with empty circuit preserves operations")
    func tensorWithEmptyCircuit() {
        var a = QuantumCircuit(qubits: 1)
        a.append(.hadamard, to: 0)

        let b = QuantumCircuit(qubits: 1)
        let tensor = a ⊗ b
        #expect(tensor.qubits == 2, "Qubit count should be 1 + 1 = 2")
        #expect(tensor.count == 1, "Only left-hand operations should exist")
    }

    @Test("tensor product state independence verified")
    func tensorStateIndependence() {
        var a = QuantumCircuit(qubits: 1)
        a.append(.pauliX, to: 0)

        var b = QuantumCircuit(qubits: 1)
        b.append(.hadamard, to: 0)

        let tensor = a ⊗ b
        let state = tensor.execute()

        #expect(abs(state.probability(of: 0b00)) < 1e-10, "Probability of |00> should be ~0")
        #expect(abs(state.probability(of: 0b01) - 0.5) < 1e-10, "Probability of |01> should be ~0.5 (X on q0, H on q1)")
        #expect(abs(state.probability(of: 0b10)) < 1e-10, "Probability of |10> should be ~0")
        #expect(abs(state.probability(of: 0b11) - 0.5) < 1e-10, "Probability of |11> should be ~0.5 (X on q0, H on q1)")
    }

    @Test("tensor product qubit labels are shifted for right-hand circuit")
    func tensorShiftsRightLabels() {
        var a = QuantumCircuit(qubits: 1)
        a.label(qubit: 0, "left")
        a.append(.hadamard, to: 0)

        var b = QuantumCircuit(qubits: 1)
        b.label(qubit: 0, "right")
        b.append(.pauliX, to: 0)

        let tensor = a ⊗ b
        #expect(tensor.qubitLabels[0] == "left", "Left label should remain at index 0")
        #expect(tensor.qubitLabels[1] == "right", "Right label should shift to index 1")
    }
}

/// Validates edge cases for circuit composition primitives.
/// Tests empty circuits, single-gate circuits, and combinations
/// of composition operators for boundary correctness.
@Suite("CircuitComposition: Edge Cases")
struct EdgeCaseTests {
    @Test("repeated on empty circuit stays empty")
    func repeatedEmptyCircuit() {
        let circuit = QuantumCircuit(qubits: 1)
        let repeated = circuit.repeated(3)
        #expect(repeated.count == 0, "Repeating empty circuit should remain empty")
        #expect(repeated.qubits == 1, "Qubit count should be preserved")
    }

    @Test("power(0) on empty circuit stays empty")
    func powerZeroEmptyCircuit() {
        let circuit = QuantumCircuit(qubits: 1)
        let powered = circuit.power(0)
        #expect(powered.count == 0, "power(0) on empty circuit should remain empty")
    }

    @Test("power(1) on empty circuit stays empty")
    func powerOneEmptyCircuit() {
        let circuit = QuantumCircuit(qubits: 2)
        let powered = circuit.power(1)
        #expect(powered.count == 0, "power(1) on empty circuit should remain empty")
    }

    @Test("empty + empty is empty")
    func emptyPlusEmpty() {
        let a = QuantumCircuit(qubits: 1)
        let b = QuantumCircuit(qubits: 1)
        let combined = a + b
        #expect(combined.count == 0, "empty + empty should have zero operations")
    }

    @Test("empty tensor empty has summed qubits")
    func emptyTensorEmpty() {
        let a = QuantumCircuit(qubits: 1)
        let b = QuantumCircuit(qubits: 2)
        let tensor = a ⊗ b
        #expect(tensor.qubits == 3, "Empty tensor empty should still have 1 + 2 = 3 qubits")
        #expect(tensor.count == 0, "Empty tensor empty should have zero operations")
    }

    @Test("single-gate circuit repeated(1) has one operation")
    func singleGateRepeatedOnce() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let repeated = circuit.repeated(1)
        #expect(repeated.count == 1, "Single-gate repeated(1) should have exactly 1 operation")
    }

    @Test("single-gate + single-gate has two operations")
    func singleGatePlusSingleGate() {
        var a = QuantumCircuit(qubits: 1)
        a.append(.hadamard, to: 0)
        var b = QuantumCircuit(qubits: 1)
        b.append(.pauliX, to: 0)
        let combined = a + b
        #expect(combined.count == 2, "Two single-gate circuits combined should have 2 operations")
    }

    @Test("single-gate tensor single-gate has two qubits and two operations")
    func singleGateTensorSingleGate() {
        var a = QuantumCircuit(qubits: 1)
        a.append(.hadamard, to: 0)
        var b = QuantumCircuit(qubits: 1)
        b.append(.pauliX, to: 0)
        let tensor = a ⊗ b
        #expect(tensor.qubits == 2, "Tensor of 1-qubit circuits should have 2 qubits")
        #expect(tensor.count == 2, "Tensor of single-gate circuits should have 2 operations")
    }

    @Test("power(2) followed by + produces correct result")
    func powerThenPlus() {
        var h = QuantumCircuit(qubits: 1)
        h.append(.hadamard, to: 0)

        var x = QuantumCircuit(qubits: 1)
        x.append(.pauliX, to: 0)

        let combined = h.power(2) + x
        let state = combined.execute()
        #expect(abs(state.probability(of: 1) - 1.0) < 1e-10, "H^2 is identity, so result should be X|0> = |1>")
    }
}

@Suite("CircuitComposition: uncovered branches")
struct UncoveredBranchTests {
    @Test("power(2) on 3-qubit circuit uses customUnitary matrix path")
    func powerTwoThreeQubitCircuitMatrixPath() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        let powered = circuit.power(2)
        #expect(powered.qubits == 3, "power(2) on 3-qubit circuit should preserve qubit count")
        #expect(powered.count == 1, "power(2) via matrix path should produce single customUnitary operation")
        let poweredState = powered.execute()
        let repeatedState = circuit.repeated(2).execute()
        for i in 0 ..< poweredState.stateSpaceSize {
            #expect(
                abs(poweredState.amplitudes[i].real - repeatedState.amplitudes[i].real) < 1e-10,
                "Real part of amplitude \(i) should match repeated(2) for 3-qubit power path",
            )
            #expect(
                abs(poweredState.amplitudes[i].imaginary - repeatedState.amplitudes[i].imaginary) < 1e-10,
                "Imaginary part of amplitude \(i) should match repeated(2) for 3-qubit power path",
            )
        }
    }

    @Test("power(2) on parameterized circuit falls back to repeated")
    func powerTwoParameterizedCircuitFallback() {
        let theta = Parameter(name: "theta")
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.rotationY(.parameter(theta)), to: 0)
        circuit.append(.cnot, to: [0, 1])
        let powered = circuit.power(2)
        #expect(powered.qubits == 2, "Fallback power(2) should preserve qubit count")
        #expect(powered.count == 4, "Fallback power(2) should repeat 2 operations twice yielding 4 operations")
        #expect(powered.parameterCount == 1, "Fallback power(2) should preserve the symbolic parameter")
    }

    @Test("controlled(by:) decomposes multi-qubit gate into controlled unitary")
    func controlledByDecomposesMultiQubitGate() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])
        let controlled = circuit.controlled(by: [0])
        #expect(controlled.qubits == 3, "2-qubit circuit controlled by 1 qubit should have 3 qubits")
        #expect(controlled.count > 0, "Controlled multi-qubit gate decomposition should produce operations")
        let state = controlled.execute()
        #expect(abs(state.probability(of: 0) - 1.0) < 1e-10, "Control qubit in |0> should leave state as |000>")
    }

    @Test("tensor product shifts reset operation qubit indices")
    func tensorProductShiftsResetOperation() {
        var a = QuantumCircuit(qubits: 2)
        a.append(.hadamard, to: 0)

        var b = QuantumCircuit(qubits: 1)
        b.append(.hadamard, to: 0)
        b.append(.reset, to: 0)

        let tensor = a ⊗ b
        #expect(tensor.qubits == 3, "Tensor product qubit count should be 2 + 1 = 3")
        #expect(tensor.count == 3, "Tensor product should have 1 + 2 = 3 operations")
        let lastOp = tensor.operations[tensor.count - 1]
        #expect(lastOp.qubits == [2], "Reset on rhs qubit 0 should shift to qubit index 2")
        #expect(!lastOp.isUnitary, "Shifted reset operation should remain non-unitary")
    }

    @Test("tensor product shifts measure operation qubit indices")
    func tensorProductShiftsMeasureOperation() {
        var a = QuantumCircuit(qubits: 1)
        a.append(.pauliX, to: 0)

        var b = QuantumCircuit(qubits: 2)
        b.append(.hadamard, to: 0)
        b.append(.measure, to: 1)

        let tensor = a ⊗ b
        #expect(tensor.qubits == 3, "Tensor product qubit count should be 1 + 2 = 3")
        #expect(tensor.count == 3, "Tensor product should have 1 + 2 = 3 operations")
        let lastOp = tensor.operations[tensor.count - 1]
        #expect(lastOp.qubits == [2], "Measure on rhs qubit 1 should shift to qubit index 2")
        #expect(!lastOp.isUnitary, "Shifted measure operation should remain non-unitary")
    }

    @Test("tensor product shifts measure with explicit classicalBit")
    func tensorProductShiftsMeasureWithClassicalBit() {
        var a = QuantumCircuit(qubits: 2)
        a.append(.hadamard, to: 0)

        let b = QuantumCircuit(qubits: 1, operations: [
            .measure(qubit: 0, classicalBit: 0),
        ])

        let tensor = a ⊗ b
        #expect(tensor.qubits == 3, "Tensor product qubit count should be 2 + 1 = 3")
        let lastOp = tensor.operations[tensor.count - 1]
        #expect(lastOp == .measure(qubit: 2, classicalBit: 2), "Measure classicalBit should shift from 0 to 2")
    }
}
