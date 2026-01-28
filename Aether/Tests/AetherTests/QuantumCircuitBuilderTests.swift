// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Validates GateStep creation from unitary gates with single and multiple qubits.
/// Ensures variadic and array initializers produce correct payload,
/// qubit lists, and optional timestamps.
@Suite("GateStep Single-Qubit and Multi-Qubit Creation")
struct GateStepCreationTests {
    @Test("Single-qubit gate step stores gate and qubit correctly")
    func singleQubitGateStep() {
        let step = GateStep(.hadamard, on: 0)
        #expect(step.gate == .hadamard, "Gate should be hadamard")
        #expect(step.qubits == [0], "Qubits should contain only qubit 0")
        #expect(step.timestamp == nil, "Timestamp should be nil when not provided")
    }

    @Test("Multi-qubit gate step stores gate and qubits correctly")
    func multiQubitGateStep() {
        let step = GateStep(.cnot, on: 0, 1)
        #expect(step.gate == .cnot, "Gate should be cnot")
        #expect(step.qubits == [0, 1], "Qubits should be [0, 1]")
    }

    @Test("Three-qubit gate step via array initializer")
    func threeQubitGateStep() {
        let step = GateStep(.toffoli, on: [0, 1, 2])
        #expect(step.gate == .toffoli, "Gate should be toffoli")
        #expect(step.qubits == [0, 1, 2], "Qubits should be [0, 1, 2]")
    }

    @Test("Gate step with timestamp stores timestamp value")
    func gateStepWithTimestamp() {
        let step = GateStep(.pauliX, on: 0, timestamp: 2.5)
        #expect(step.gate == .pauliX, "Gate should be pauliX")
        #expect(step.timestamp == 2.5, "Timestamp should be 2.5")
    }

    @Test("Array initializer with timestamp")
    func arrayInitWithTimestamp() {
        let step = GateStep(.cnot, on: [0, 1], timestamp: 1.0)
        #expect(step.gate == .cnot, "Gate should be cnot")
        #expect(step.qubits == [0, 1], "Qubits should be [0, 1]")
        #expect(step.timestamp == 1.0, "Timestamp should be 1.0")
    }
}

/// Validates GateStep creation from NonUnitaryOperation values.
/// Ensures reset and measure operations produce correct qubit targets,
/// identity gate fallback, and proper operation conversion.
@Suite("GateStep Non-Unitary Operation")
struct GateStepNonUnitaryTests {
    @Test("Reset operation step stores qubit and returns identity gate")
    func resetOperationStep() {
        let step = GateStep(.reset, on: 2)
        #expect(step.gate == .identity, "Non-unitary step should return identity gate")
        #expect(step.qubits == [2], "Reset should target qubit 2")
        #expect(step.timestamp == nil, "Timestamp should be nil when not provided")
    }

    @Test("Measure operation step stores qubit and returns identity gate")
    func measureOperationStep() {
        let step = GateStep(.measure, on: 1)
        #expect(step.gate == .identity, "Measure step should return identity gate")
        #expect(step.qubits == [1], "Measure should target qubit 1")
    }

    @Test("Non-unitary step with timestamp")
    func nonUnitaryWithTimestamp() {
        let step = GateStep(.reset, on: 0, timestamp: 3.0)
        #expect(step.timestamp == 3.0, "Timestamp should be 3.0")
        #expect(step.qubits == [0], "Reset should target qubit 0")
    }
}

/// Validates GateStep conversion to CircuitOperation via the operation property.
/// Ensures gate, reset, and measure steps produce correct CircuitOperation cases
/// with matching qubits and timestamps.
@Suite("GateStep Operation Conversion")
struct GateStepOperationConversionTests {
    @Test("Gate step converts to gate operation")
    func gateStepConvertsToGateOperation() {
        let step = GateStep(.hadamard, on: 0)
        let op = step.operation
        let expected = CircuitOperation.gate(.hadamard, qubits: [0])
        #expect(op == expected, "Operation should be gate(.hadamard, qubits: [0])")
    }

    @Test("Multi-qubit gate step converts with correct qubits")
    func multiQubitOperationConversion() {
        let step = GateStep(.cnot, on: 0, 1)
        let op = step.operation
        let expected = CircuitOperation.gate(.cnot, qubits: [0, 1])
        #expect(op == expected, "Operation should be gate(.cnot, qubits: [0, 1])")
    }

    @Test("Gate step with timestamp converts preserving timestamp")
    func gateStepWithTimestampConversion() {
        let step = GateStep(.pauliX, on: 0, timestamp: 1.5)
        let op = step.operation
        let expected = CircuitOperation.gate(.pauliX, qubits: [0], timestamp: 1.5)
        #expect(op == expected, "Operation should preserve timestamp 1.5")
    }

    @Test("Reset step converts to reset operation")
    func resetStepConvertsToResetOperation() {
        let step = GateStep(.reset, on: 2)
        let op = step.operation
        let expected = CircuitOperation.reset(qubit: 2)
        #expect(op == expected, "Reset step should convert to reset(qubit: 2)")
    }

    @Test("Measure step converts to measure operation")
    func measureStepConvertsToMeasureOperation() {
        let step = GateStep(.measure, on: 1, timestamp: 4.0)
        let op = step.operation
        let expected = CircuitOperation.measure(qubit: 1, timestamp: 4.0)
        #expect(op == expected, "Measure step should convert to measure(qubit: 1, timestamp: 4.0)")
    }
}

/// Validates GateStep Equatable conformance for value-type identity.
/// Ensures equal steps compare as equal and differing steps compare as not equal
/// across gate type, qubits, timestamp, and payload kind.
@Suite("GateStep Equatable")
struct GateStepEquatableTests {
    @Test("Identical gate steps are equal")
    func identicalGateStepsEqual() {
        let a = GateStep(.hadamard, on: 0)
        let b = GateStep(.hadamard, on: 0)
        #expect(a == b, "Steps with same gate and qubit should be equal")
    }

    @Test("Steps with different gates are not equal")
    func differentGatesNotEqual() {
        let a = GateStep(.hadamard, on: 0)
        let b = GateStep(.pauliX, on: 0)
        #expect(a != b, "Steps with different gates should not be equal")
    }

    @Test("Steps with different qubits are not equal")
    func differentQubitsNotEqual() {
        let a = GateStep(.hadamard, on: 0)
        let b = GateStep(.hadamard, on: 1)
        #expect(a != b, "Steps with different qubits should not be equal")
    }

    @Test("Steps with different timestamps are not equal")
    func differentTimestampsNotEqual() {
        let a = GateStep(.hadamard, on: 0, timestamp: 1.0)
        let b = GateStep(.hadamard, on: 0, timestamp: 2.0)
        #expect(a != b, "Steps with different timestamps should not be equal")
    }

    @Test("Gate step and non-unitary step are not equal")
    func gateVsNonUnitaryNotEqual() {
        let gate = GateStep(.identity, on: 0)
        let reset = GateStep(.reset, on: 0)
        #expect(gate != reset, "Gate step and non-unitary step should not be equal")
    }
}

/// Validates QuantumCircuitBuilder static methods for result-builder protocol.
/// Ensures buildExpression, buildBlock, buildOptional, buildEither, and buildArray
/// produce correct step arrays with proper concatenation and flattening.
@Suite("QuantumCircuitBuilder Static Methods")
struct QuantumCircuitBuilderMethodTests {
    @Test("buildExpression wraps single step in array")
    func buildExpressionWrapsSingleStep() {
        let step = GateStep(.hadamard, on: 0)
        let result = QuantumCircuitBuilder.buildExpression(step)
        #expect(result.count == 1, "buildExpression should produce single-element array")
        #expect(result[0] == step, "buildExpression should contain the original step")
    }

    @Test("buildBlock concatenates multiple component arrays")
    func buildBlockConcatenatesComponents() {
        let a = [GateStep(.hadamard, on: 0)]
        let b = [GateStep(.cnot, on: 0, 1)]
        let c = [GateStep(.pauliX, on: 1)]
        let result = QuantumCircuitBuilder.buildBlock(a, b, c)
        #expect(result.count == 3, "buildBlock should concatenate 3 components into 3 steps")
        #expect(result[0].gate == .hadamard, "First step should be hadamard")
        #expect(result[1].gate == .cnot, "Second step should be cnot")
        #expect(result[2].gate == .pauliX, "Third step should be pauliX")
    }

    @Test("buildBlock with empty components produces empty array")
    func buildBlockWithEmptyComponents() {
        let result = QuantumCircuitBuilder.buildBlock()
        #expect(result.isEmpty, "buildBlock with no components should produce empty array")
    }

    @Test("buildOptional returns steps when non-nil")
    func buildOptionalReturnsStepsWhenNonNil() {
        let steps: [GateStep]? = [GateStep(.hadamard, on: 0)]
        let result = QuantumCircuitBuilder.buildOptional(steps)
        #expect(result.count == 1, "buildOptional should return wrapped steps when non-nil")
    }

    @Test("buildOptional returns empty array when nil")
    func buildOptionalReturnsEmptyWhenNil() {
        let steps: [GateStep]? = nil
        let result = QuantumCircuitBuilder.buildOptional(steps)
        #expect(result.isEmpty, "buildOptional should return empty array when nil")
    }

    @Test("buildEither first returns steps unchanged")
    func buildEitherFirstReturnsSteps() {
        let steps = [GateStep(.hadamard, on: 0)]
        let result = QuantumCircuitBuilder.buildEither(first: steps)
        #expect(result == steps, "buildEither(first:) should return steps unchanged")
    }

    @Test("buildEither second returns steps unchanged")
    func buildEitherSecondReturnsSteps() {
        let steps = [GateStep(.pauliX, on: 0)]
        let result = QuantumCircuitBuilder.buildEither(second: steps)
        #expect(result == steps, "buildEither(second:) should return steps unchanged")
    }

    @Test("buildArray flattens array of arrays into single array")
    func buildArrayFlattensGroups() {
        let groups: [[GateStep]] = [
            [GateStep(.hadamard, on: 0)],
            [GateStep(.hadamard, on: 1)],
            [GateStep(.hadamard, on: 2)],
        ]
        let result = QuantumCircuitBuilder.buildArray(groups)
        #expect(result.count == 3, "buildArray should produce 3 steps from 3 single-element groups")
        #expect(result[0].qubits == [0], "First step should target qubit 0")
        #expect(result[1].qubits == [1], "Second step should target qubit 1")
        #expect(result[2].qubits == [2], "Third step should target qubit 2")
    }

    @Test("buildArray with empty groups produces empty array")
    func buildArrayWithEmptyGroups() {
        let groups: [[GateStep]] = []
        let result = QuantumCircuitBuilder.buildArray(groups)
        #expect(result.isEmpty, "buildArray with no groups should produce empty array")
    }
}

/// Validates QuantumCircuit builder initializer with declarative syntax.
/// Ensures basic circuits, conditional gates (if/else), for loops,
/// and autoOptimize flag work correctly through the builder DSL.
@Suite("QuantumCircuit Builder Init")
struct QuantumCircuitBuilderInitTests {
    @Test("Basic circuit with sequential gates")
    func basicCircuitWithSequentialGates() {
        let circuit = QuantumCircuit(qubits: 2) {
            GateStep(.hadamard, on: 0)
            GateStep(.cnot, on: 0, 1)
        }
        #expect(circuit.count == 2, "Circuit should have 2 operations")
        #expect(circuit.qubits == 2, "Circuit should have 2 qubits")
        #expect(circuit.operations[0] == .gate(.hadamard, qubits: [0]), "First op should be H on 0")
        #expect(circuit.operations[1] == .gate(.cnot, qubits: [0, 1]), "Second op should be CNOT on 0,1")
    }

    @Test("Empty builder produces empty circuit")
    func emptyBuilderProducesEmptyCircuit() {
        let circuit = QuantumCircuit(qubits: 1) {}
        #expect(circuit.isEmpty, "Empty builder should produce empty circuit")
        #expect(circuit.qubits == 1, "Qubit count should still be 1")
    }

    @Test("Conditional gate with if-true branch")
    func conditionalGateIfTrue() {
        let applyH = true
        let circuit = QuantumCircuit(qubits: 1) {
            if applyH {
                GateStep(.hadamard, on: 0)
            }
        }
        #expect(circuit.count == 1, "Circuit should have 1 gate when condition is true")
    }

    @Test("Conditional gate with if-false branch yields empty")
    func conditionalGateIfFalse() {
        let applyH = false
        let circuit = QuantumCircuit(qubits: 1) {
            if applyH {
                GateStep(.hadamard, on: 0)
            }
        }
        #expect(circuit.isEmpty, "Circuit should be empty when condition is false")
    }

    @Test("If-else selects correct branch")
    func ifElseSelectsCorrectBranch() {
        let useH = false
        let circuit = QuantumCircuit(qubits: 1) {
            if useH {
                GateStep(.hadamard, on: 0)
            } else {
                GateStep(.pauliX, on: 0)
            }
        }
        #expect(circuit.count == 1, "Circuit should have exactly 1 operation")
        #expect(circuit.operations[0] == .gate(.pauliX, qubits: [0]), "Should select else branch (pauliX)")
    }

    @Test("For loop generates gates for each iteration")
    func forLoopGeneratesGates() {
        let circuit = QuantumCircuit(qubits: 3) {
            for q in 0 ..< 3 {
                GateStep(.hadamard, on: [q])
            }
        }
        #expect(circuit.count == 3, "Circuit should have 3 hadamard gates from for loop")
        #expect(circuit.operations[0] == .gate(.hadamard, qubits: [0]), "First H should be on qubit 0")
        #expect(circuit.operations[1] == .gate(.hadamard, qubits: [1]), "Second H should be on qubit 1")
        #expect(circuit.operations[2] == .gate(.hadamard, qubits: [2]), "Third H should be on qubit 2")
    }

    @Test("Builder with autoOptimize creates valid circuit")
    func autoOptimizeCreatesValidCircuit() {
        let circuit = QuantumCircuit(qubits: 1, autoOptimize: true) {
            GateStep(.hadamard, on: 0)
            GateStep(.hadamard, on: 0)
        }
        #expect(circuit.qubits == 1, "Auto-optimized circuit should have 1 qubit")
        #expect(circuit.autoOptimize == true, "Circuit should have autoOptimize enabled")
    }

    @Test("Builder includes non-unitary operations")
    func builderIncludesNonUnitaryOps() {
        let circuit = QuantumCircuit(qubits: 2) {
            GateStep(.hadamard, on: 0)
            GateStep(.reset, on: 0)
            GateStep(.measure, on: 1)
        }
        #expect(circuit.count == 3, "Circuit should have 3 operations including non-unitary")
        #expect(circuit.operations[1] == .reset(qubit: 0), "Second op should be reset on qubit 0")
        #expect(circuit.operations[2] == .measure(qubit: 1), "Third op should be measure on qubit 1")
    }
}

/// Validates end-to-end circuit construction and execution via builder DSL.
/// Ensures Bell state creation, state probabilities, and amplitude verification
/// match expected quantum mechanical outcomes within tolerance.
@Suite("QuantumCircuitBuilder Integration")
struct QuantumCircuitBuilderIntegrationTests {
    @Test("Bell state via builder: equal superposition of |00> and |11>")
    func bellStateViaBuilder() {
        let circuit = QuantumCircuit(qubits: 2) {
            GateStep(.hadamard, on: 0)
            GateStep(.cnot, on: 0, 1)
        }
        let state = circuit.execute()
        let p00 = state.probability(of: 0b00)
        let p01 = state.probability(of: 0b01)
        let p10 = state.probability(of: 0b10)
        let p11 = state.probability(of: 0b11)
        #expect(abs(p00 - 0.5) < 1e-10, "P(|00>) should be 0.5 for Bell state")
        #expect(abs(p01) < 1e-10, "P(|01>) should be 0 for Bell state")
        #expect(abs(p10) < 1e-10, "P(|10>) should be 0 for Bell state")
        #expect(abs(p11 - 0.5) < 1e-10, "P(|11>) should be 0.5 for Bell state")
    }

    @Test("PauliX via builder flips |0> to |1>")
    func pauliXFlipsState() {
        let circuit = QuantumCircuit(qubits: 1) {
            GateStep(.pauliX, on: 0)
        }
        let state = circuit.execute()
        #expect(abs(state.probability(of: 0) - 0.0) < 1e-10, "P(|0>) should be 0 after X gate")
        #expect(abs(state.probability(of: 1) - 1.0) < 1e-10, "P(|1>) should be 1 after X gate")
    }

    @Test("Builder circuit matches imperative circuit execution")
    func builderMatchesImperative() {
        let builderCircuit = QuantumCircuit(qubits: 2) {
            GateStep(.hadamard, on: 0)
            GateStep(.pauliX, on: 1)
            GateStep(.cnot, on: 0, 1)
        }

        var imperativeCircuit = QuantumCircuit(qubits: 2)
        imperativeCircuit.append(.hadamard, to: 0)
        imperativeCircuit.append(.pauliX, to: 1)
        imperativeCircuit.append(.cnot, to: [0, 1])

        let builderState = builderCircuit.execute()
        let imperativeState = imperativeCircuit.execute()

        for i in 0 ..< 4 {
            let bProb = builderState.probability(of: i)
            let iProb = imperativeState.probability(of: i)
            #expect(abs(bProb - iProb) < 1e-10, "Builder and imperative states should match for basis state \(i)")
        }
    }

    @Test("Conditional builder produces correct state")
    func conditionalBuilderProducesCorrectState() {
        let applyX = true
        let circuit = QuantumCircuit(qubits: 1) {
            if applyX {
                GateStep(.pauliX, on: 0)
            }
        }
        let state = circuit.execute()
        #expect(abs(state.probability(of: 1) - 1.0) < 1e-10, "Conditional X should flip to |1>")
    }

    @Test("Loop-based uniform superposition via builder")
    func loopUniformSuperposition() {
        let circuit = QuantumCircuit(qubits: 3) {
            for q in 0 ..< 3 {
                GateStep(.hadamard, on: [q])
            }
        }
        let state = circuit.execute()
        let expectedProb = 1.0 / 8.0
        for i in 0 ..< 8 {
            #expect(abs(state.probability(of: i) - expectedProb) < 1e-10, "All 8 basis states should have equal probability 1/8 for state \(i)")
        }
    }
}
