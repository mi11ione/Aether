// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for QubitSize phantom type static counts.
/// Validates that Q1 through Q8 and Q16 report the correct
/// qubit count used by TypedCircuit at initialization.
@Suite("QubitSize Static Counts")
struct QubitSizeTests {
    @Test("Q1.count equals 1")
    func q1Count() {
        #expect(Q1.count == 1, "Q1.count should be 1 but got \(Q1.count)")
    }

    @Test("Q2.count equals 2")
    func q2Count() {
        #expect(Q2.count == 2, "Q2.count should be 2 but got \(Q2.count)")
    }

    @Test("Q3.count equals 3")
    func q3Count() {
        #expect(Q3.count == 3, "Q3.count should be 3 but got \(Q3.count)")
    }

    @Test("Q4.count equals 4")
    func q4Count() {
        #expect(Q4.count == 4, "Q4.count should be 4 but got \(Q4.count)")
    }

    @Test("Q5.count equals 5")
    func q5Count() {
        #expect(Q5.count == 5, "Q5.count should be 5 but got \(Q5.count)")
    }

    @Test("Q6.count equals 6")
    func q6Count() {
        #expect(Q6.count == 6, "Q6.count should be 6 but got \(Q6.count)")
    }

    @Test("Q7.count equals 7")
    func q7Count() {
        #expect(Q7.count == 7, "Q7.count should be 7 but got \(Q7.count)")
    }

    @Test("Q8.count equals 8")
    func q8Count() {
        #expect(Q8.count == 8, "Q8.count should be 8 but got \(Q8.count)")
    }

    @Test("Q16.count equals 16")
    func q16Count() {
        #expect(Q16.count == 16, "Q16.count should be 16 but got \(Q16.count)")
    }
}

/// Test suite for QubitSum compile-time qubit addition.
/// Validates that QubitSum produces the correct count from
/// two QubitSize operands for tensor product composition.
@Suite("QubitSum Type-Level Addition")
struct QubitSumTests {
    @Test("QubitSum<Q2, Q3>.count equals 5")
    func sumQ2Q3() {
        let result = QubitSum<Q2, Q3>.count
        #expect(result == 5, "QubitSum<Q2, Q3>.count should be 5 but got \(result)")
    }

    @Test("QubitSum<Q1, Q1>.count equals 2")
    func sumQ1Q1() {
        let result = QubitSum<Q1, Q1>.count
        #expect(result == 2, "QubitSum<Q1, Q1>.count should be 2 but got \(result)")
    }

    @Test("QubitSum<Q4, Q4>.count equals 8")
    func sumQ4Q4() {
        let result = QubitSum<Q4, Q4>.count
        #expect(result == 8, "QubitSum<Q4, Q4>.count should be 8 but got \(result)")
    }

    @Test("QubitSum<Q1, Q7>.count equals 8")
    func sumQ1Q7() {
        let result = QubitSum<Q1, Q7>.count
        #expect(result == 8, "QubitSum<Q1, Q7>.count should be 8 but got \(result)")
    }

    @Test("Nested QubitSum<QubitSum<Q1, Q2>, Q3>.count equals 6")
    func nestedSum() {
        let result = QubitSum<QubitSum<Q1, Q2>, Q3>.count
        #expect(result == 6, "Nested QubitSum should be 6 but got \(result)")
    }
}

/// Test suite for TypedCircuit builder initializer.
/// Validates that the result-builder closure creates a circuit
/// with the correct qubit count and gate operations.
@Suite("TypedCircuit Builder Init")
struct TypedCircuitBuilderInitTests {
    @Test("Builder creates circuit with correct qubit count")
    func builderQubitCount() {
        let typed = TypedCircuit<Q2> {
            GateStep(.hadamard, on: 0)
        }
        #expect(typed.circuit.qubits == 2, "Circuit should have 2 qubits but got \(typed.circuit.qubits)")
    }

    @Test("Builder captures all gate steps in order")
    func builderCapturesSteps() {
        let typed = TypedCircuit<Q2> {
            GateStep(.hadamard, on: 0)
            GateStep(.cnot, on: 0, 1)
        }
        #expect(typed.circuit.count == 2, "Circuit should have 2 operations but got \(typed.circuit.count)")
    }

    @Test("Builder with no gates creates empty circuit")
    func builderEmptyCircuit() {
        let typed = TypedCircuit<Q3> {}
        #expect(typed.circuit.isEmpty, "Circuit should be empty but has \(typed.circuit.count) operations")
        #expect(typed.circuit.qubits == 3, "Circuit should have 3 qubits but got \(typed.circuit.qubits)")
    }

    @Test("Builder with single-qubit circuit")
    func builderSingleQubit() {
        let typed = TypedCircuit<Q1> {
            GateStep(.hadamard, on: 0)
        }
        #expect(typed.circuit.qubits == 1, "Circuit should have 1 qubit but got \(typed.circuit.qubits)")
        #expect(typed.circuit.count == 1, "Circuit should have 1 operation but got \(typed.circuit.count)")
    }
}

/// Test suite for TypedCircuit wrapping initializer.
/// Validates that wrapping an existing QuantumCircuit preserves
/// its qubit count and operations when the size matches.
@Suite("TypedCircuit Wrapping Init")
struct TypedCircuitWrappingInitTests {
    @Test("Wrapping preserves qubit count when matching")
    func wrappingMatchingQubits() {
        var qc = QuantumCircuit(qubits: 2)
        qc.append(.hadamard, to: 0)
        let typed = TypedCircuit<Q2>(qc)
        #expect(typed.circuit.qubits == 2, "Wrapped circuit should have 2 qubits but got \(typed.circuit.qubits)")
    }

    @Test("Wrapping preserves operations from source circuit")
    func wrappingPreservesOperations() {
        var qc = QuantumCircuit(qubits: 3)
        qc.append(.hadamard, to: 0)
        qc.append(.cnot, to: [0, 1])
        qc.append(.pauliX, to: 2)
        let typed = TypedCircuit<Q3>(qc)
        #expect(typed.circuit.count == 3, "Wrapped circuit should have 3 operations but got \(typed.circuit.count)")
    }

    @Test("Wrapping empty circuit preserves qubit count")
    func wrappingEmptyCircuit() {
        let qc = QuantumCircuit(qubits: 4)
        let typed = TypedCircuit<Q4>(qc)
        #expect(typed.circuit.qubits == 4, "Wrapped circuit should have 4 qubits but got \(typed.circuit.qubits)")
        #expect(typed.circuit.isEmpty, "Wrapped circuit should be empty but has \(typed.circuit.count) operations")
    }
}

/// Test suite for TypedCircuit execute from ground state.
/// Validates that building and executing a typed circuit produces
/// correct quantum state probabilities for known circuits.
@Suite("TypedCircuit Execute")
struct TypedCircuitExecuteTests {
    @Test("Bell state has equal probabilities for |00> and |11>")
    func bellStateProbabilities() {
        let bell = TypedCircuit<Q2> {
            GateStep(.hadamard, on: 0)
            GateStep(.cnot, on: 0, 1)
        }
        let state = bell.execute()
        let p00 = state.probability(of: 0b00)
        let p11 = state.probability(of: 0b11)
        #expect(abs(p00 - 0.5) < 1e-10, "P(|00>) should be 0.5 but got \(p00)")
        #expect(abs(p11 - 0.5) < 1e-10, "P(|11>) should be 0.5 but got \(p11)")
    }

    @Test("Bell state has zero probability for |01> and |10>")
    func bellStateZeroProbabilities() {
        let bell = TypedCircuit<Q2> {
            GateStep(.hadamard, on: 0)
            GateStep(.cnot, on: 0, 1)
        }
        let state = bell.execute()
        let p01 = state.probability(of: 0b01)
        let p10 = state.probability(of: 0b10)
        #expect(abs(p01) < 1e-10, "P(|01>) should be 0 but got \(p01)")
        #expect(abs(p10) < 1e-10, "P(|10>) should be 0 but got \(p10)")
    }

    @Test("Executed state has correct qubit count")
    func executeReturnsCorrectQubits() {
        let typed = TypedCircuit<Q3> {
            GateStep(.hadamard, on: 0)
        }
        let state = typed.execute()
        #expect(state.qubits == 3, "Executed state should have 3 qubits but got \(state.qubits)")
    }

    @Test("Hadamard creates equal superposition on single qubit")
    func hadamardSuperposition() {
        let typed = TypedCircuit<Q1> {
            GateStep(.hadamard, on: 0)
        }
        let state = typed.execute()
        let p0 = state.probability(of: 0)
        let p1 = state.probability(of: 1)
        #expect(abs(p0 - 0.5) < 1e-10, "P(|0>) should be 0.5 but got \(p0)")
        #expect(abs(p1 - 0.5) < 1e-10, "P(|1>) should be 0.5 but got \(p1)")
    }

    @Test("PauliX flips |0> to |1>")
    func pauliXFlip() {
        let typed = TypedCircuit<Q1> {
            GateStep(.pauliX, on: 0)
        }
        let state = typed.execute()
        let p1 = state.probability(of: 1)
        #expect(abs(p1 - 1.0) < 1e-10, "P(|1>) should be 1.0 after X gate but got \(p1)")
    }
}

/// Test suite for TypedCircuit execute on initial state.
/// Validates that executing a typed circuit on a custom initial
/// state correctly transforms the provided quantum state.
@Suite("TypedCircuit Execute On Initial State")
struct TypedCircuitExecuteOnInitialStateTests {
    @Test("Execute on ground state matches parameterless execute")
    func executeOnGroundState() {
        let typed = TypedCircuit<Q2> {
            GateStep(.hadamard, on: 0)
            GateStep(.cnot, on: 0, 1)
        }
        let groundState = QuantumState(qubits: 2)
        let fromGround = typed.execute(on: groundState)
        let fromDefault = typed.execute()

        let p00Ground = fromGround.probability(of: 0b00)
        let p00Default = fromDefault.probability(of: 0b00)
        #expect(abs(p00Ground - p00Default) < 1e-10, "Execute on ground state should match default execute")
    }

    @Test("Execute on |1> state with Hadamard produces |-> state")
    func executeOnOneState() {
        let typed = TypedCircuit<Q1> {
            GateStep(.hadamard, on: 0)
        }
        let oneState = QuantumState(qubits: 1, amplitudes: [.zero, .one])
        let result = typed.execute(on: oneState)
        let p0 = result.probability(of: 0)
        let p1 = result.probability(of: 1)
        #expect(abs(p0 - 0.5) < 1e-10, "P(|0>) should be 0.5 for |-> state but got \(p0)")
        #expect(abs(p1 - 0.5) < 1e-10, "P(|1>) should be 0.5 for |-> state but got \(p1)")
    }

    @Test("Execute on custom 2-qubit state preserves normalization")
    func executePreservesNormalization() {
        let typed = TypedCircuit<Q2> {
            GateStep(.hadamard, on: 0)
        }
        let initial = QuantumState(qubits: 2, amplitudes: [
            Complex(1.0 / sqrt(2.0), 0), Complex(0, 0),
            Complex(1.0 / sqrt(2.0), 0), Complex(0, 0),
        ])
        let result = typed.execute(on: initial)
        #expect(result.isNormalized(), "Result state should be normalized after execution")
    }
}

/// Test suite for TypedCircuit tensor product composition.
/// Validates that composing two typed circuits produces a combined
/// circuit with correct qubit count and operations from both inputs.
@Suite("TypedCircuit Composing")
struct TypedCircuitComposingTests {
    @Test("Composed circuit has sum of qubit counts")
    func composedQubitCount() {
        let a = TypedCircuit<Q2> {
            GateStep(.hadamard, on: 0)
        }
        let b = TypedCircuit<Q1> {
            GateStep(.pauliX, on: 0)
        }
        let combined: TypedCircuit<QubitSum<Q2, Q1>> = TypedCircuit.composing(a, b)
        #expect(combined.circuit.qubits == 3, "Combined circuit should have 3 qubits but got \(combined.circuit.qubits)")
    }

    @Test("Composed circuit contains operations from both circuits")
    func composedOperationCount() {
        let a = TypedCircuit<Q2> {
            GateStep(.hadamard, on: 0)
            GateStep(.cnot, on: 0, 1)
        }
        let b = TypedCircuit<Q1> {
            GateStep(.pauliX, on: 0)
        }
        let combined: TypedCircuit<QubitSum<Q2, Q1>> = TypedCircuit.composing(a, b)
        #expect(combined.circuit.count == 3, "Combined circuit should have 3 operations but got \(combined.circuit.count)")
    }

    @Test("Composed circuit shifts second circuit qubit indices")
    func composedShiftsIndices() {
        let a = TypedCircuit<Q2> {
            GateStep(.hadamard, on: 0)
        }
        let b = TypedCircuit<Q2> {
            GateStep(.pauliX, on: 0)
            GateStep(.pauliX, on: 1)
        }
        let combined: TypedCircuit<QubitSum<Q2, Q2>> = TypedCircuit.composing(a, b)
        let ops = combined.circuit.operations
        let secondOpQubits = ops[1].qubits
        let thirdOpQubits = ops[2].qubits

        #expect(secondOpQubits == [2], "Second circuit's qubit 0 should shift to 2 but got \(secondOpQubits)")
        #expect(thirdOpQubits == [3], "Second circuit's qubit 1 should shift to 3 but got \(thirdOpQubits)")
    }

    @Test("Composed circuit executes correctly")
    func composedExecution() {
        let a = TypedCircuit<Q1> {
            GateStep(.pauliX, on: 0)
        }
        let b = TypedCircuit<Q1> {
            GateStep(.hadamard, on: 0)
        }
        let combined: TypedCircuit<QubitSum<Q1, Q1>> = TypedCircuit.composing(a, b)
        let state = combined.execute()

        let p10 = state.probability(of: 0b01)
        let p11 = state.probability(of: 0b11)
        #expect(abs(p10 - 0.5) < 1e-10, "P(|10>) should be 0.5 but got \(p10)")
        #expect(abs(p11 - 0.5) < 1e-10, "P(|11>) should be 0.5 but got \(p11)")
    }

    @Test("Composing two empty circuits produces empty combined circuit")
    func composedEmptyCircuits() {
        let a = TypedCircuit<Q1> {}
        let b = TypedCircuit<Q2> {}
        let combined: TypedCircuit<QubitSum<Q1, Q2>> = TypedCircuit.composing(a, b)
        #expect(combined.circuit.qubits == 3, "Combined empty circuit should have 3 qubits but got \(combined.circuit.qubits)")
        #expect(combined.circuit.isEmpty, "Combined empty circuit should have 0 operations but got \(combined.circuit.count)")
    }

    @Test("Composing shifts reset qubit index by first circuit size")
    func composedWithReset() {
        var qc = QuantumCircuit(qubits: 1)
        qc.append(.reset, to: 0)
        let a = TypedCircuit<Q1>(qc)
        let b = TypedCircuit<Q1> {
            GateStep(.hadamard, on: 0)
        }
        let combined: TypedCircuit<QubitSum<Q1, Q1>> = TypedCircuit.composing(b, a)
        let ops = combined.circuit.operations
        #expect(ops.count == 2, "Combined circuit should have 2 operations but got \(ops.count)")
        #expect(combined.circuit.qubits == 2, "Combined circuit should have 2 qubits but got \(combined.circuit.qubits)")
        #expect(ops[1].qubits == [1], "Reset qubit should shift from 0 to 1 but got \(ops[1].qubits)")
    }

    @Test("Composing shifts measure qubit index by first circuit size")
    func composedWithMeasure() {
        var qc = QuantumCircuit(qubits: 1)
        qc.append(.measure, to: 0)
        let a = TypedCircuit<Q1>(qc)
        let b = TypedCircuit<Q1> {
            GateStep(.hadamard, on: 0)
        }
        let combined: TypedCircuit<QubitSum<Q1, Q1>> = TypedCircuit.composing(b, a)
        let ops = combined.circuit.operations
        #expect(ops.count == 2, "Combined circuit should have 2 operations but got \(ops.count)")
        #expect(combined.circuit.qubits == 2, "Combined circuit should have 2 qubits but got \(combined.circuit.qubits)")
        #expect(ops[1].qubits == [1], "Measure qubit should shift from 0 to 1 but got \(ops[1].qubits)")
    }

    @Test("Composing shifts measure with explicit classicalBit by first circuit size")
    func composedWithMeasureExplicitClassicalBit() {
        let qc = QuantumCircuit(qubits: 2, operations: [.measure(qubit: 0, classicalBit: 1)])
        let source = TypedCircuit<Q2>(qc)
        let prefix = TypedCircuit<Q1> {
            GateStep(.hadamard, on: 0)
        }
        let combined: TypedCircuit<QubitSum<Q1, Q2>> = TypedCircuit.composing(prefix, source)
        let ops = combined.circuit.operations
        #expect(ops.count == 2, "Combined circuit should have 2 operations but got \(ops.count)")
        #expect(ops[1].qubits == [1], "Measure qubit should shift from 0 to 1 but got \(ops[1].qubits)")
    }
}

/// Test suite for TypedCircuit Sendable conformance.
/// Validates that TypedCircuit can cross concurrency boundaries
/// by assigning to a nonisolated(unsafe) or Sendable-constrained context.
@Suite("TypedCircuit Sendable Conformance")
struct TypedCircuitSendableTests {
    @Test("TypedCircuit can be captured in Sendable closure")
    func sendableClosure() {
        let typed = TypedCircuit<Q2> {
            GateStep(.hadamard, on: 0)
            GateStep(.cnot, on: 0, 1)
        }
        let sendableFn: @Sendable () -> Int = {
            typed.circuit.qubits
        }
        let result = sendableFn()
        #expect(result == 2, "TypedCircuit should be usable across Sendable boundary, got qubits=\(result)")
    }

    @Test("TypedCircuit value can be stored in Sendable container")
    func sendableStorage() {
        struct Container: Sendable {
            let circuit: TypedCircuit<Q3>
        }
        let typed = TypedCircuit<Q3> {
            GateStep(.hadamard, on: 0)
        }
        let container = Container(circuit: typed)
        #expect(container.circuit.circuit.qubits == 3, "Sendable container should hold TypedCircuit with 3 qubits")
    }
}
