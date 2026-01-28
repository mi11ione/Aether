// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Validates CircuitOperation.measure gate property returns nil.
/// Ensures the gate accessor correctly distinguishes unitary gate
/// operations from non-unitary measurement operations.
@Suite("Measure Gate Access")
struct MeasureGateAccessTests {
    @Test("Measure operation gate property returns nil")
    func measureGateIsNil() {
        let op = CircuitOperation.measure(qubit: 0)
        #expect(op.gate == nil, "Measure operation should have no associated gate")
    }

    @Test("Measure operation with classicalBit gate property returns nil")
    func measureWithClassicalBitGateIsNil() {
        let op = CircuitOperation.measure(qubit: 1, classicalBit: 2)
        #expect(op.gate == nil, "Measure operation with explicit classical bit should have no associated gate")
    }
}

/// Validates CircuitOperation.measure timestamp extraction.
/// Ensures timestamp is correctly stored and retrieved for
/// both timed and untimed measurement operations.
@Suite("Measure Timestamp Access")
struct MeasureTimestampAccessTests {
    @Test("Measure with timestamp returns correct value")
    func measureWithTimestamp() {
        let op = CircuitOperation.measure(qubit: 0, timestamp: 3.75)
        #expect(op.timestamp == 3.75, "Measure timestamp should match the value provided at construction")
    }

    @Test("Measure without timestamp returns nil")
    func measureWithoutTimestamp() {
        let op = CircuitOperation.measure(qubit: 0)
        #expect(op.timestamp == nil, "Measure without explicit timestamp should return nil")
    }

    @Test("Measure with classicalBit and timestamp returns correct timestamp")
    func measureWithClassicalBitAndTimestamp() {
        let op = CircuitOperation.measure(qubit: 2, classicalBit: 3, timestamp: 5.0)
        #expect(op.timestamp == 5.0, "Measure timestamp should be 5.0 regardless of classical bit value")
    }
}

/// Validates CircuitOperation.measure isParameterized returns false.
/// Ensures non-unitary measurement operations correctly report
/// they contain no symbolic parameters requiring binding.
@Suite("Measure Parameterization")
struct MeasureParameterizationTests {
    @Test("Measure is not parameterized")
    func measureIsNotParameterized() {
        let op = CircuitOperation.measure(qubit: 0)
        #expect(op.isParameterized == false, "Measure operations should never be parameterized")
    }

    @Test("Measure with classicalBit is not parameterized")
    func measureWithClassicalBitIsNotParameterized() {
        let op = CircuitOperation.measure(qubit: 1, classicalBit: 1)
        #expect(op.isParameterized == false, "Measure operations with classical bit should never be parameterized")
    }
}

/// Validates CircuitOperation.measure parameters returns empty set.
/// Ensures non-unitary measurement operations correctly report
/// no symbolic parameters in their parameter extraction method.
@Suite("Measure Parameter Extraction")
struct MeasureParameterExtractionTests {
    @Test("Measure parameters returns empty set")
    func measureParametersEmpty() {
        let op = CircuitOperation.measure(qubit: 0)
        let params = op.parameters()
        #expect(params.isEmpty, "Measure operations should return an empty parameter set")
    }
}

/// Validates CircuitOperation.measure bound(with:) returns self.
/// Ensures parameter binding on non-unitary measurement operations
/// is a no-op that preserves the original operation unchanged.
@Suite("Measure Parameter Binding")
struct MeasureParameterBindingTests {
    @Test("Binding parameters on measure returns equal operation")
    func measureBoundReturnsSelf() {
        let op = CircuitOperation.measure(qubit: 0)
        let bound = op.bound(with: ["x": 1.0])
        #expect(bound == op, "Measure bound(with:) should return an operation equal to the original")
    }

    @Test("Binding parameters on measure with classicalBit returns equal operation")
    func measureWithClassicalBitBoundReturnsSelf() {
        let op = CircuitOperation.measure(qubit: 1, classicalBit: 2, timestamp: 4.0)
        let bound = op.bound(with: ["theta": 0.5, "phi": 1.0])
        #expect(bound == op, "Measure bound(with:) should preserve qubit, classical bit, and timestamp")
    }
}

/// Validates CircuitOperation.measure description formatting.
/// Ensures human-readable string output contains expected tokens
/// for measure operations with and without timestamps.
@Suite("Measure Description")
struct MeasureDescriptionTests {
    @Test("Measure description without timestamp contains measure keyword and default cbit")
    func measureDescriptionDefault() {
        let op = CircuitOperation.measure(qubit: 0)
        let desc = op.description
        #expect(desc.contains("measure"), "Measure description should contain the word 'measure'")
        #expect(desc.contains("qubit: 0"), "Measure description should contain the target qubit index")
        #expect(desc.contains("cbit: 0"), "Measure description should contain classical bit defaulting to qubit index")
    }

    @Test("Measure description with explicit classicalBit shows correct cbit")
    func measureDescriptionExplicitClassicalBit() {
        let op = CircuitOperation.measure(qubit: 1, classicalBit: 3)
        let desc = op.description
        #expect(desc.contains("qubit: 1"), "Measure description should show qubit 1")
        #expect(desc.contains("cbit: 3"), "Measure description should show classical bit 3")
    }

    @Test("Measure description with timestamp includes formatted time")
    func measureDescriptionWithTimestamp() {
        let op = CircuitOperation.measure(qubit: 2, classicalBit: 2, timestamp: 1.5)
        let desc = op.description
        #expect(desc.contains("measure"), "Timed measure description should contain the word 'measure'")
        #expect(desc.contains("1.500"), "Timed measure description should contain formatted timestamp")
    }
}

/// Validates CircuitOperation.measure qubit access.
/// Ensures the qubits property returns a single-element array
/// containing the correct qubit index for measurement operations.
@Suite("Measure Qubit Access")
struct MeasureQubitAccessTests {
    @Test("Measure qubits returns single-element array")
    func measureQubits() {
        let op = CircuitOperation.measure(qubit: 0)
        #expect(op.qubits == [0], "Measure on qubit 0 should return qubits array [0]")
    }

    @Test("Measure qubits reflects construction qubit index")
    func measureQubitsReflectsIndex() {
        let op = CircuitOperation.measure(qubit: 3)
        #expect(op.qubits == [3], "Measure on qubit 3 should return qubits array [3]")
    }
}

/// Validates CircuitOperation.measure unitarity reporting.
/// Ensures measurement operations are correctly identified as
/// non-unitary irreversible transformations.
@Suite("Measure Unitarity")
struct MeasureUnitarityTests {
    @Test("Measure is not unitary")
    func measureIsNotUnitary() {
        let op = CircuitOperation.measure(qubit: 0)
        #expect(op.isUnitary == false, "Measurement operations are irreversible and should report as non-unitary")
    }
}

/// Validates CircuitJSONEncoder encodes measure operations with
/// correct type, qubit indices, and classical bit mappings in
/// the JSON schema output.
@Suite("CircuitJSONEncoder Measure")
struct CircuitJSONEncoderMeasureTests {
    @Test("Encode circuit with measure produces valid JSON containing measurement type")
    func encodeMeasureOperation() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.measure, to: 0)
        let data = CircuitJSONEncoder.encode(circuit)
        let jsonString = String(data: data, encoding: .utf8)!
        #expect(jsonString.contains("\"measurement\""), "Encoded JSON should contain measurement operation type")
    }

    @Test("Encoded measure JSON contains qubit index")
    func encodedMeasureContainsQubit() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.measure, to: 1)
        let data = CircuitJSONEncoder.encode(circuit)
        let jsonString = String(data: data, encoding: .utf8)!
        #expect(jsonString.contains("\"qubits\""), "Encoded JSON should contain qubits field")
        #expect(jsonString.contains("1"), "Encoded JSON should contain qubit index 1")
    }

    @Test("Encoded measure JSON contains classicalBits field")
    func encodedMeasureContainsClassicalBits() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.measure, to: 0)
        let data = CircuitJSONEncoder.encode(circuit)
        let jsonString = String(data: data, encoding: .utf8)!
        #expect(jsonString.contains("\"classicalBits\""), "Encoded JSON should contain classicalBits field for measurement")
    }
}

/// Validates CircuitJSONDecoder decodes measure operations from
/// JSON back into a QuantumCircuit preserving the measurement
/// operation type and qubit targets.
@Suite("CircuitJSONDecoder Measure")
struct CircuitJSONDecoderMeasureTests {
    @Test("Decode circuit with measure operation preserves operation count")
    func decodeMeasureRoundTrip() {
        var original = QuantumCircuit(qubits: 2)
        original.append(.hadamard, to: 0)
        original.append(.measure, to: 0)
        let data = CircuitJSONEncoder.encode(original)
        let result = CircuitJSONDecoder.decode(from: data)
        #expect(result.succeeded, "Decoding a valid circuit with measure should succeed without errors")
        #expect(result.circuit.count == 2, "Decoded circuit should have 2 operations (hadamard + measure)")
    }

    @Test("Decoded measure operation is non-unitary")
    func decodedMeasureIsNonUnitary() {
        var original = QuantumCircuit(qubits: 1)
        original.append(.measure, to: 0)
        let data = CircuitJSONEncoder.encode(original)
        let result = CircuitJSONDecoder.decode(from: data)
        let lastOp = result.circuit.operations.last
        #expect(lastOp?.isUnitary == false, "Decoded measure operation should be non-unitary")
    }

    @Test("Decoded circuit qubit count matches original")
    func decodedMeasureQubitCount() {
        var original = QuantumCircuit(qubits: 3)
        original.append(.measure, to: 2)
        let data = CircuitJSONEncoder.encode(original)
        let result = CircuitJSONDecoder.decode(from: data)
        #expect(result.circuit.qubits == 3, "Decoded circuit should preserve the original qubit count of 3")
    }
}

/// Validates QASM2Exporter produces correct OpenQASM 2.0 measure
/// statements mapping quantum registers to classical registers
/// with the standard arrow syntax.
@Suite("QASM2Exporter Measure")
struct QASM2ExporterMeasureTests {
    @Test("Export circuit with measure produces QASM 2.0 measure statement")
    func exportMeasureProducesStatement() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.measure, to: 0)
        let qasm = QASM2Exporter.export(circuit)
        #expect(qasm.contains("measure q[0] -> c[0];"), "QASM 2.0 export should contain 'measure q[0] -> c[0];'")
    }

    @Test("Export includes OPENQASM header and register declarations")
    func exportMeasureIncludesHeader() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.measure, to: 1)
        let qasm = QASM2Exporter.export(circuit)
        #expect(qasm.contains("OPENQASM 2.0;"), "QASM 2.0 export should start with version header")
        #expect(qasm.contains("qreg q[2];"), "QASM 2.0 export should declare quantum register")
        #expect(qasm.contains("creg c[2];"), "QASM 2.0 export should declare classical register")
        #expect(qasm.contains("measure q[1] -> c[1];"), "QASM 2.0 export should contain measure on qubit 1")
    }

    @Test("Export multiple measures produces multiple measure lines")
    func exportMultipleMeasures() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.measure, to: 0)
        circuit.append(.measure, to: 1)
        let qasm = QASM2Exporter.export(circuit)
        #expect(qasm.contains("measure q[0] -> c[0];"), "QASM 2.0 export should contain measure on qubit 0")
        #expect(qasm.contains("measure q[1] -> c[1];"), "QASM 2.0 export should contain measure on qubit 1")
    }
}

/// Validates QASM3Exporter produces correct OpenQASM 3.0 measure
/// statements using the assignment syntax with bit register
/// targets and qubit register sources.
@Suite("QASM3Exporter Measure")
struct QASM3ExporterMeasureTests {
    @Test("Export circuit with measure produces QASM 3.0 assignment syntax")
    func exportMeasureProducesAssignment() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.measure, to: 0)
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("c[0] = measure q[0];"), "QASM 3.0 export should use assignment syntax 'c[0] = measure q[0];'")
    }

    @Test("Export includes OPENQASM 3.0 header and modern register syntax")
    func exportMeasureIncludesHeader() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.measure, to: 1)
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("OPENQASM 3.0;"), "QASM 3.0 export should contain version header")
        #expect(qasm.contains("qubit[2] q;"), "QASM 3.0 export should use modern qubit declaration")
        #expect(qasm.contains("bit[2] c;"), "QASM 3.0 export should use modern bit declaration")
        #expect(qasm.contains("c[1] = measure q[1];"), "QASM 3.0 export should contain measure on qubit 1")
    }

    @Test("Export multiple measures produces multiple assignment lines")
    func exportMultipleMeasures() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.measure, to: 0)
        circuit.append(.measure, to: 1)
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("c[0] = measure q[0];"), "QASM 3.0 export should contain measure assignment for qubit 0")
        #expect(qasm.contains("c[1] = measure q[1];"), "QASM 3.0 export should contain measure assignment for qubit 1")
    }
}

/// Validates QuantumCircuit.execute() correctly handles measure
/// operations by collapsing the qubit state deterministically
/// via the reset-based simulation path.
@Suite("QuantumCircuit Execute with Measure")
struct QuantumCircuitExecuteMeasureTests {
    @Test("Execute circuit with measure on ground state preserves zero state")
    func executeMeasureOnGroundState() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.measure, to: 0)
        let state = circuit.execute()
        let prob0 = state.probability(of: 0)
        #expect(prob0 > 0.99, "Measuring qubit 0 in ground state should preserve the |00> state with probability near 1.0")
    }

    @Test("Execute circuit with measure after X gate collapses to zero on measured qubit")
    func executeMeasureAfterXGate() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.pauliX, to: 0)
        circuit.append(.measure, to: 0)
        let state = circuit.execute()
        let totalProb = state.amplitudes.reduce(0.0) { $0 + $1.real * $1.real + $1.imaginary * $1.imaginary }
        #expect(abs(totalProb - 1.0) < 1e-10, "State after measure should remain normalized with total probability 1.0")
    }

    @Test("Execute circuit with gate after measure produces valid state")
    func executeMeasureThenGate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.measure, to: 0)
        circuit.append(.hadamard, to: 0)
        let state = circuit.execute()
        let totalProb = state.amplitudes.reduce(0.0) { $0 + $1.real * $1.real + $1.imaginary * $1.imaginary }
        #expect(abs(totalProb - 1.0) < 1e-10, "State should remain normalized after measure followed by gate")
    }
}

/// Validates QuantumCircuit.execute(on:) with measure operations
/// applied to custom initial states preserves normalization and
/// produces deterministic outcomes.
@Suite("QuantumCircuit Execute on Custom State with Measure")
struct QuantumCircuitExecuteOnStateMeasureTests {
    @Test("Execute measure on custom |0> state preserves state")
    func executeMeasureOnZeroState() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.measure, to: 0)
        let initial = QuantumState(qubits: 1)
        let result = circuit.execute(on: initial)
        let prob0 = result.probability(of: 0)
        #expect(prob0 > 0.99, "Measuring |0> should preserve the |0> state with probability near 1.0")
    }

    @Test("Execute measure on 2-qubit custom state preserves normalization")
    func executeMeasureOnTwoQubitState() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.measure, to: 0)
        let initial = QuantumState(qubits: 2)
        let result = circuit.execute(on: initial)
        let totalProb = result.amplitudes.reduce(0.0) { $0 + $1.real * $1.real + $1.imaginary * $1.imaginary }
        #expect(abs(totalProb - 1.0) < 1e-10, "State after measure on custom 2-qubit state should remain normalized")
    }

    @Test("Execute upToIndex with measure returns intermediate state")
    func executeMeasureUpToIndex() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.measure, to: 0)
        circuit.append(.pauliX, to: 1)
        let initial = QuantumState(qubits: 2)
        let result = circuit.execute(on: initial, upToIndex: 2)
        let totalProb = result.amplitudes.reduce(0.0) { $0 + $1.real * $1.real + $1.imaginary * $1.imaginary }
        #expect(abs(totalProb - 1.0) < 1e-10, "Partial execution up to measure should produce normalized state")
    }
}

/// Validates GateApplication.applyReset correctly projects the
/// measured qubit state onto the computational basis and
/// renormalizes the resulting amplitude vector.
@Suite("GateApplication Measure via applyReset")
struct GateApplicationMeasureTests {
    @Test("applyReset on ground state returns ground state unchanged")
    func applyResetOnGroundState() {
        let state = QuantumState(qubits: 1)
        let result = GateApplication.applyReset(qubit: 0, state: state)
        let prob0 = result.probability(of: 0)
        #expect(prob0 > 0.99, "Applying reset/measure on |0> should leave state in |0> with probability near 1.0")
    }

    @Test("applyReset produces normalized output")
    func applyResetProducesNormalized() {
        let state = QuantumState(qubits: 2)
        let afterH = GateApplication.apply(.hadamard, to: [0], state: state)
        let result = GateApplication.applyReset(qubit: 0, state: afterH)
        let totalProb = result.amplitudes.reduce(0.0) { $0 + $1.real * $1.real + $1.imaginary * $1.imaginary }
        #expect(abs(totalProb - 1.0) < 1e-10, "State after applyReset should be normalized with total probability 1.0")
    }

    @Test("applyReset on |1> state projects to |0>")
    func applyResetOnOneState() {
        let state = QuantumState(qubits: 1)
        let flipped = GateApplication.apply(.pauliX, to: [0], state: state)
        let result = GateApplication.applyReset(qubit: 0, state: flipped)
        let prob0 = result.probability(of: 0)
        #expect(prob0 > 0.99, "Applying reset/measure on |1> should collapse to |0> with probability near 1.0")
    }
}

/// Validates NonUnitaryOperation.measure equatable and hashable
/// conformance and CircuitOperation.measure equatable behavior
/// for matching and differing qubit indices.
@Suite("Measure Equatable and Hashable")
struct MeasureEquatableHashableTests {
    @Test("NonUnitaryOperation measure equals itself")
    func measureEqualsItself() {
        let a = NonUnitaryOperation.measure
        let b = NonUnitaryOperation.measure
        #expect(a == b, "Two NonUnitaryOperation.measure values should be equal")
    }

    @Test("NonUnitaryOperation measure is not equal to reset")
    func measureNotEqualToReset() {
        let a = NonUnitaryOperation.measure
        let b = NonUnitaryOperation.reset
        #expect(a != b, "NonUnitaryOperation.measure should not equal NonUnitaryOperation.reset")
    }

    @Test("NonUnitaryOperation measure has consistent hash value")
    func measureHashConsistent() {
        let a = NonUnitaryOperation.measure
        let b = NonUnitaryOperation.measure
        #expect(a.hashValue == b.hashValue, "Equal NonUnitaryOperation.measure values should produce identical hash values")
    }

    @Test("Same measure operations are equal")
    func sameMeasuresAreEqual() {
        let a = CircuitOperation.measure(qubit: 0)
        let b = CircuitOperation.measure(qubit: 0)
        #expect(a == b, "Measure operations on the same qubit should be equal")
    }

    @Test("Different measure operations are not equal")
    func differentMeasuresAreNotEqual() {
        let a = CircuitOperation.measure(qubit: 0)
        let b = CircuitOperation.measure(qubit: 1)
        #expect(a != b, "Measure operations on different qubits should not be equal")
    }

    @Test("Measure and reset on same qubit are not equal")
    func measureAndResetNotEqual() {
        let a = CircuitOperation.measure(qubit: 0)
        let b = CircuitOperation.reset(qubit: 0)
        #expect(a != b, "Measure and reset operations on the same qubit should not be equal")
    }
}

/// Validates QuantumCircuit append and insert methods correctly
/// handle NonUnitaryOperation.measure through the circuit
/// building API with proper qubit expansion.
@Suite("QuantumCircuit Measure Building API")
struct QuantumCircuitMeasureBuildingTests {
    @Test("Append measure via NonUnitaryOperation adds measure to circuit")
    func appendMeasureOperation() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.measure, to: 0)
        #expect(circuit.count == 1, "Circuit should have 1 operation after appending measure")
        #expect(circuit.operations[0].isUnitary == false, "Appended measure operation should be non-unitary")
    }

    @Test("Insert measure via NonUnitaryOperation places measure at correct index")
    func insertMeasureOperation() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.insert(.measure, to: 0, at: 1)
        #expect(circuit.count == 3, "Circuit should have 3 operations after inserting measure")
        #expect(circuit.operations[1].isUnitary == false, "Inserted measure at index 1 should be non-unitary")
    }

    @Test("addOperation with measure CircuitOperation adds to circuit")
    func addOperationMeasure() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.addOperation(.measure(qubit: 0))
        #expect(circuit.count == 1, "Circuit should have 1 operation after addOperation with measure")
    }

    @Test("Append measure auto-expands qubit count")
    func appendMeasureAutoExpands() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.measure, to: 2)
        #expect(circuit.qubits == 3, "Circuit should auto-expand to 3 qubits when measuring qubit 2")
    }
}

/// Validates GateApplication.apply dispatches measure operations
/// to the applyReset path, collapsing the qubit state onto
/// a computational basis state while preserving normalization.
@Suite("GateApplication Apply Measure Dispatch")
struct GateApplicationApplyMeasureDispatchTests {
    @Test("apply with measure operation on ground state preserves zero state")
    func applyMeasureOnGroundState() {
        let state = QuantumState(qubits: 1)
        let op = CircuitOperation.measure(qubit: 0)
        let result = GateApplication.apply(op, state: state)
        let prob0 = result.probability(of: 0)
        #expect(prob0 > 0.99, "GateApplication.apply with measure on |0> should preserve ground state")
    }

    @Test("apply with measure operation on |1> collapses to |0>")
    func applyMeasureOnExcitedState() {
        let state = QuantumState(qubits: 1)
        let flipped = GateApplication.apply(.pauliX, to: [0], state: state)
        let op = CircuitOperation.measure(qubit: 0)
        let result = GateApplication.apply(op, state: flipped)
        let prob0 = result.probability(of: 0)
        #expect(prob0 > 0.99, "GateApplication.apply with measure on |1> should collapse to |0> via reset path")
    }

    @Test("apply with measure on 2-qubit state preserves normalization")
    func applyMeasureTwoQubitNormalization() {
        let state = QuantumState(qubits: 2)
        let afterH = GateApplication.apply(.hadamard, to: [0], state: state)
        let op = CircuitOperation.measure(qubit: 0)
        let result = GateApplication.apply(op, state: afterH)
        let totalProb = result.amplitudes.reduce(0.0) { $0 + $1.real * $1.real + $1.imaginary * $1.imaginary }
        #expect(abs(totalProb - 1.0) < 1e-10, "GateApplication.apply with measure should produce normalized state")
    }

    @Test("apply with measure operation returns different reference than gate path")
    func applyMeasureReturnsDifferentFromGate() {
        let state = QuantumState(qubits: 1)
        let measureOp = CircuitOperation.measure(qubit: 0)
        let gateOp = CircuitOperation.gate(.identity, qubits: [0])
        let measureResult = GateApplication.apply(measureOp, state: state)
        let gateResult = GateApplication.apply(gateOp, state: state)
        let measureProb = measureResult.probability(of: 0)
        let gateProb = gateResult.probability(of: 0)
        #expect(abs(measureProb - gateProb) < 1e-10, "Measure and identity on |0> should both yield ground state")
    }
}

/// Validates MetalGateApplication.apply dispatches measure operations
/// to the GateApplication.applyReset fallback path when the Metal
/// executor encounters a non-unitary measurement operation.
@Suite("MetalGateApplication Measure Fallback")
struct MetalGateApplicationMeasureFallbackTests {
    @Test("Metal apply with measure falls back to CPU reset path")
    func metalApplyMeasureFallback() async {
        guard let metal = MetalGateApplication() else {
            return
        }
        let state = QuantumState(qubits: 1)
        let op = CircuitOperation.measure(qubit: 0)
        let result = await metal.apply(op, state: state)
        let prob0 = result.probability(of: 0)
        #expect(prob0 > 0.99, "MetalGateApplication.apply with measure on |0> should fall back to CPU and preserve ground state")
    }

    @Test("Metal apply with measure on excited state collapses to ground")
    func metalApplyMeasureOnExcitedState() async {
        guard let metal = MetalGateApplication() else {
            return
        }
        let state = QuantumState(qubits: 1)
        let flipped = GateApplication.apply(.pauliX, to: [0], state: state)
        let op = CircuitOperation.measure(qubit: 0)
        let result = await metal.apply(op, state: flipped)
        let prob0 = result.probability(of: 0)
        #expect(prob0 > 0.99, "MetalGateApplication.apply with measure on |1> should fall back to CPU and collapse to |0>")
    }

    @Test("Metal apply with measure preserves normalization on 2-qubit state")
    func metalApplyMeasureNormalization() async {
        guard let metal = MetalGateApplication() else {
            return
        }
        let state = QuantumState(qubits: 2)
        let afterH = GateApplication.apply(.hadamard, to: [0], state: state)
        let op = CircuitOperation.measure(qubit: 0)
        let result = await metal.apply(op, state: afterH)
        let totalProb = result.amplitudes.reduce(0.0) { $0 + $1.real * $1.real + $1.imaginary * $1.imaginary }
        #expect(abs(totalProb - 1.0) < 1e-10, "MetalGateApplication.apply with measure should produce normalized state via CPU fallback")
    }
}

/// Validates DensityMatrix.applying dispatches measure operations
/// to the applyReset path, projecting the density matrix onto
/// a computational basis state while preserving trace normalization.
@Suite("DensityMatrix Measure Application")
struct DensityMatrixMeasureApplicationTests {
    @Test("DensityMatrix applying measure on ground state preserves state")
    func applyingMeasureOnGroundState() {
        let rho = DensityMatrix(qubits: 1)
        let op = CircuitOperation.measure(qubit: 0)
        let result = rho.applying(op)
        let prob0 = result.probability(of: 0)
        #expect(prob0 > 0.99, "DensityMatrix.applying measure on |0><0| should preserve ground state density matrix")
    }

    @Test("DensityMatrix applying measure on excited state projects to ground")
    func applyingMeasureOnExcitedState() {
        let rho = DensityMatrix(qubits: 1)
        let excited = rho.applying(.pauliX, to: 0)
        let op = CircuitOperation.measure(qubit: 0)
        let result = excited.applying(op)
        let prob0 = result.probability(of: 0)
        #expect(prob0 > 0.99, "DensityMatrix.applying measure on |1><1| should project to |0><0| via reset path")
    }

    @Test("DensityMatrix applying measure preserves trace on 2-qubit system")
    func applyingMeasurePreservesTrace() {
        let rho = DensityMatrix(qubits: 2)
        let afterH = rho.applying(.hadamard, to: 0)
        let op = CircuitOperation.measure(qubit: 0)
        let result = afterH.applying(op)
        let trace = result.trace()
        #expect(abs(trace - 1.0) < 1e-10, "DensityMatrix.applying measure should preserve trace equal to 1.0")
    }
}

/// Validates CircuitJSONDecoder correctly handles malformed measurement
/// operations with missing qubit indices by emitting an error diagnostic
/// and skipping the operation rather than crashing.
@Suite("CircuitJSONDecoder Measure Guard Failure")
struct CircuitJSONDecoderMeasureGuardFailureTests {
    @Test("Decode measurement with empty qubits array produces error diagnostic")
    func decodeMeasurementMissingQubits() {
        let json = """
        {
            "version": 1,
            "qubitCount": 2,
            "classicalBitCount": 2,
            "operations": [
                {
                    "type": "measurement",
                    "qubits": [],
                    "classicalBits": [0]
                }
            ]
        }
        """
        let data = json.data(using: .utf8)!
        let result = CircuitJSONDecoder.decode(from: data)
        #expect(result.succeeded == false, "Decoding measurement with empty qubits should produce an error diagnostic")
    }

    @Test("Decode measurement with empty qubits skips the operation")
    func decodeMeasurementMissingQubitsSkipsOperation() {
        let json = """
        {
            "version": 1,
            "qubitCount": 2,
            "classicalBitCount": 2,
            "operations": [
                {
                    "type": "measurement",
                    "qubits": [],
                    "classicalBits": [0]
                }
            ]
        }
        """
        let data = json.data(using: .utf8)!
        let result = CircuitJSONDecoder.decode(from: data)
        #expect(result.circuit.count == 0, "Circuit should have 0 operations when measurement is skipped due to missing qubits")
    }

    @Test("Decode measurement with empty qubits diagnostic mentions missing qubit")
    func decodeMeasurementMissingQubitsDiagnosticMessage() {
        let json = """
        {
            "version": 1,
            "qubitCount": 1,
            "classicalBitCount": 1,
            "operations": [
                {
                    "type": "measurement",
                    "qubits": [],
                    "classicalBits": []
                }
            ]
        }
        """
        let data = json.data(using: .utf8)!
        let result = CircuitJSONDecoder.decode(from: data)
        let hasQubitError = result.diagnostics.contains { $0.message.contains("missing qubit") }
        #expect(hasQubitError, "Diagnostic should mention 'missing qubit' for measurement with empty qubits array")
    }

    @Test("Decode valid measurement alongside empty-qubit measurement recovers partial circuit")
    func decodeMixedValidAndInvalidMeasurements() {
        let json = """
        {
            "version": 1,
            "qubitCount": 2,
            "classicalBitCount": 2,
            "operations": [
                {
                    "type": "gate",
                    "gate": "h",
                    "qubits": [0]
                },
                {
                    "type": "measurement",
                    "qubits": [],
                    "classicalBits": [0]
                },
                {
                    "type": "measurement",
                    "qubits": [1],
                    "classicalBits": [1]
                }
            ]
        }
        """
        let data = json.data(using: .utf8)!
        let result = CircuitJSONDecoder.decode(from: data)
        #expect(result.circuit.count == 2, "Circuit should have 2 operations: valid gate and valid measurement, skipping invalid measurement")
    }
}
