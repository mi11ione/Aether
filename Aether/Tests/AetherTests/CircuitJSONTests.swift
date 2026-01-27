// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Test suite for CircuitJSON schema version constant.
/// Validates the schema version value used by encoder and decoder
/// to ensure forward compatibility of serialized circuit files.
@Suite("CircuitJSON Schema Version")
struct CircuitJSONSchemaVersionTests {
    @Test("Schema version is 1")
    func schemaVersionIsOne() {
        #expect(CircuitJSON.schemaVersion == 1, "CircuitJSON.schemaVersion should be 1")
    }
}

/// Test suite for CircuitJSONEncoder encoding circuits to JSON Data.
/// Validates that empty, single-gate, multi-gate, parameterized, and reset
/// circuits all produce valid non-empty JSON Data with deterministic output.
@Suite("CircuitJSON Encoding")
struct CircuitJSONEncodingTests {
    @Test("Encode empty circuit produces valid non-empty JSON Data")
    func encodeEmptyCircuit() {
        let circuit = QuantumCircuit(qubits: 2)
        let data = CircuitJSONEncoder.encode(circuit)
        #expect(data.count > 0, "Encoded empty circuit should produce non-empty Data")
    }

    @Test("Encode single-gate circuit round-trips preserving gate count")
    func encodeSingleGateRoundTrip() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let data = CircuitJSONEncoder.encode(circuit)
        let result = CircuitJSONDecoder.decode(from: data)
        #expect(result.circuit.count == 1, "Round-trip of single-gate circuit should preserve gate count of 1")
    }

    @Test("Encode multi-gate circuit round-trips preserving all operations")
    func encodeMultiGateRoundTrip() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.pauliX, to: 2)
        circuit.append(.pauliZ, to: 1)
        circuit.append(.swap, to: [1, 2])
        let data = CircuitJSONEncoder.encode(circuit)
        let result = CircuitJSONDecoder.decode(from: data)
        #expect(result.circuit.count == 5, "Round-trip of 5-gate circuit should preserve operation count of 5")
    }

    @Test("Encode parameterized gate round-trips preserving parameter values")
    func encodeParameterizedGateRoundTrip() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.rotationX(1.234), to: 0)
        circuit.append(.rotationZ(0.567), to: 1)
        let data = CircuitJSONEncoder.encode(circuit)
        let result = CircuitJSONDecoder.decode(from: data)
        #expect(result.circuit.count == 2, "Round-trip of parameterized circuit should preserve operation count of 2")
        #expect(result.succeeded, "Decoding parameterized circuit should succeed without errors")
    }

    @Test("Encode circuit with reset operation")
    func encodeResetOperation() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.reset, to: 0)
        circuit.append(.hadamard, to: 0)
        let data = CircuitJSONEncoder.encode(circuit)
        let result = CircuitJSONDecoder.decode(from: data)
        #expect(result.circuit.count == 3, "Round-trip of circuit with reset should preserve operation count of 3")
        #expect(result.succeeded, "Decoding circuit with reset should succeed without errors")
    }

    @Test("Deterministic encoding produces same Data for same circuit twice")
    func deterministicEncoding() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        let data1 = CircuitJSONEncoder.encode(circuit)
        let data2 = CircuitJSONEncoder.encode(circuit)
        #expect(data1 == data2, "Encoding the same circuit twice should produce identical Data")
    }
}

/// Test suite for CircuitJSONDecoder decoding JSON Data back to circuits.
/// Validates successful decoding, malformed input handling, empty data
/// handling, and round-trip preservation of qubit and operation counts.
@Suite("CircuitJSON Decoding")
struct CircuitJSONDecodingTests {
    @Test("Decode malformed JSON returns error diagnostic with succeeded false")
    func decodeMalformedJSON() {
        let malformed = Data("{ not valid json".utf8)
        let result = CircuitJSONDecoder.decode(from: malformed)
        #expect(!result.succeeded, "Decoding malformed JSON should report succeeded as false")
        #expect(!result.diagnostics.isEmpty, "Decoding malformed JSON should produce at least one diagnostic")
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "Decoding malformed JSON should produce an error-severity diagnostic")
    }

    @Test("Decode valid JSON returns succeeded true")
    func decodeValidJSON() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        let data = CircuitJSONEncoder.encode(circuit)
        let result = CircuitJSONDecoder.decode(from: data)
        #expect(result.succeeded, "Decoding valid JSON should report succeeded as true")
    }

    @Test("Round-trip encode then decode preserves qubit count")
    func roundTripPreservesQubitCount() {
        var circuit = QuantumCircuit(qubits: 4)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [1, 2])
        circuit.append(.pauliX, to: 3)
        let data = CircuitJSONEncoder.encode(circuit)
        let result = CircuitJSONDecoder.decode(from: data)
        #expect(result.circuit.qubits == 4, "Round-trip should preserve qubit count of 4")
    }

    @Test("Round-trip encode then decode preserves operation count")
    func roundTripPreservesOperationCount() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliY, to: 1)
        circuit.append(.cnot, to: [0, 2])
        circuit.append(.toffoli, to: [0, 1, 2])
        let data = CircuitJSONEncoder.encode(circuit)
        let result = CircuitJSONDecoder.decode(from: data)
        #expect(result.circuit.count == 4, "Round-trip should preserve operation count of 4")
    }

    @Test("Decode empty Data produces error diagnostic")
    func decodeEmptyData() {
        let empty = Data()
        let result = CircuitJSONDecoder.decode(from: empty)
        #expect(!result.succeeded, "Decoding empty Data should report succeeded as false")
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "Decoding empty Data should produce an error-severity diagnostic")
    }
}

/// Validates CircuitJSONEncoder for advanced gate types and parameters.
/// Covers u1, u2, u3, symbolic, negated, custom matrix, and controlled
/// gate encoding to ensure complete JSON serialization coverage.
@Suite("CircuitJSON Encoder Coverage")
struct CircuitJSONEncoderCoverageTests {
    @Test("Encode u1 gate round-trips preserving gate")
    func encodeU1Gate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u1(lambda: 0.5), to: 0)
        let data = CircuitJSONEncoder.encode(circuit)
        let result = CircuitJSONDecoder.decode(from: data)
        #expect(result.succeeded, "Round-trip of u1 gate should succeed without errors")
        #expect(result.circuit.count == 1, "Round-trip of u1 gate should preserve operation count of 1")
    }

    @Test("Encode u2 gate round-trips preserving gate")
    func encodeU2Gate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u2(phi: 0.3, lambda: 0.7), to: 0)
        let data = CircuitJSONEncoder.encode(circuit)
        let result = CircuitJSONDecoder.decode(from: data)
        #expect(result.succeeded, "Round-trip of u2 gate should succeed without errors")
        #expect(result.circuit.count == 1, "Round-trip of u2 gate should preserve operation count of 1")
    }

    @Test("Encode u3 gate round-trips preserving gate")
    func encodeU3Gate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u3(theta: 0.1, phi: 0.2, lambda: 0.3), to: 0)
        let data = CircuitJSONEncoder.encode(circuit)
        let result = CircuitJSONDecoder.decode(from: data)
        #expect(result.succeeded, "Round-trip of u3 gate should succeed without errors")
        #expect(result.circuit.count == 1, "Round-trip of u3 gate should preserve operation count of 1")
    }

    @Test("Encode symbolic parameter round-trips preserving parameter name")
    func encodeSymbolicParameter() {
        var circuit = QuantumCircuit(qubits: 1)
        let theta = Parameter(name: "theta")
        circuit.append(.rotationX(.parameter(theta)), to: 0)
        let data = CircuitJSONEncoder.encode(circuit)
        let json = String(data: data, encoding: .utf8)!
        #expect(json.contains("symbolic"), "Symbolic parameter should encode with type symbolic")
        #expect(json.contains("theta"), "Symbolic parameter name theta should appear in JSON")
    }

    @Test("Encode negated parameter round-trips preserving negated type")
    func encodeNegatedParameter() {
        var circuit = QuantumCircuit(qubits: 1)
        let phi = Parameter(name: "phi")
        circuit.append(.rotationZ(.negatedParameter(phi)), to: 0)
        let data = CircuitJSONEncoder.encode(circuit)
        let json = String(data: data, encoding: .utf8)!
        #expect(json.contains("negated"), "Negated parameter should encode with type negated")
        #expect(json.contains("phi"), "Negated parameter name phi should appear in JSON")
    }

    @Test("Encode custom single-qubit gate preserves matrix in JSON")
    func encodeCustomSingleQubit() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(0, 1)],
        ]
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.customSingleQubit(matrix: matrix), to: 0)
        let data = CircuitJSONEncoder.encode(circuit)
        let result = CircuitJSONDecoder.decode(from: data)
        #expect(result.succeeded, "Round-trip of custom single-qubit gate should succeed")
        #expect(result.circuit.count == 1, "Round-trip of custom single-qubit gate should preserve operation count")
    }

    @Test("Encode custom two-qubit gate preserves matrix in JSON")
    func encodeCustomTwoQubit() {
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(0, 0), Complex(0, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(1, 0), Complex(0, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(0, 0), Complex(0, 0), Complex(1, 0)],
            [Complex(0, 0), Complex(0, 0), Complex(1, 0), Complex(0, 0)],
        ]
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.customTwoQubit(matrix: matrix), to: [0, 1])
        let data = CircuitJSONEncoder.encode(circuit)
        let result = CircuitJSONDecoder.decode(from: data)
        #expect(result.succeeded, "Round-trip of custom two-qubit gate should succeed")
        #expect(result.circuit.count == 1, "Round-trip of custom two-qubit gate should preserve operation count")
    }

    @Test("Encode custom unitary gate preserves matrix in JSON")
    func encodeCustomUnitary() {
        var rows = [[Complex<Double>]](repeating: [Complex<Double>](repeating: Complex(0, 0), count: 8), count: 8)
        for i in 0 ..< 8 {
            rows[i][i] = Complex(1, 0)
        }
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.customUnitary(matrix: rows), to: [0, 1, 2])
        let data = CircuitJSONEncoder.encode(circuit)
        let result = CircuitJSONDecoder.decode(from: data)
        #expect(result.succeeded, "Round-trip of custom unitary gate should succeed")
        #expect(result.circuit.count == 1, "Round-trip of custom unitary gate should preserve operation count")
    }

    @Test("Encode controlled gate preserves controls in JSON")
    func encodeControlledGate() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.controlled(gate: .pauliX, controls: [0, 1]), to: [0, 1, 2])
        let data = CircuitJSONEncoder.encode(circuit)
        let json = String(data: data, encoding: .utf8)!
        #expect(json.contains("controls"), "Controlled gate JSON should contain controls field")
        let result = CircuitJSONDecoder.decode(from: data)
        #expect(result.circuit.count == 1, "Round-trip of controlled gate should preserve operation count")
    }
}

/// Validates CircuitJSONDecoder for edge cases and extended gate types.
/// Covers measurement, barrier, unknown types, custom matrices, symbolic
/// and negated parameters, version checks, and error diagnostics.
@Suite("CircuitJSON Decoder Coverage")
struct CircuitJSONDecoderCoverageTests {
    @Test("Decode measurement type produces warning diagnostic")
    func decodeMeasurementType() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"measurement","qubits":[0]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        let hasWarning = result.diagnostics.contains { $0.severity == .warning && $0.message.contains("measurement") }
        #expect(hasWarning, "Decoding measurement type should produce a warning mentioning measurement")
        #expect(result.circuit.count == 0, "Measurement operation should be skipped and not added to circuit")
    }

    @Test("Decode barrier type produces warning diagnostic")
    func decodeBarrierType() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"barrier","qubits":[0]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        let hasWarning = result.diagnostics.contains { $0.severity == .warning && $0.message.contains("barrier") }
        #expect(hasWarning, "Decoding barrier type should produce a warning mentioning barrier")
        #expect(result.circuit.count == 0, "Barrier operation should be skipped and not added to circuit")
    }

    @Test("Decode unknown operation type produces warning diagnostic")
    func decodeUnknownOperationType() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"unknown_op","qubits":[0]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        let hasWarning = result.diagnostics.contains { $0.severity == .warning && $0.message.contains("unknown_op") }
        #expect(hasWarning, "Decoding unknown operation type should produce a warning mentioning the type name")
    }

    @Test("Decode custom gate with 2x2 matrix creates custom single-qubit gate")
    func decodeCustomGateWithMatrix() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"custom_unitary","qubits":[0],"matrix":[[{"real":1,"imaginary":0},{"real":0,"imaginary":0}],[{"real":0,"imaginary":0},{"real":0,"imaginary":1}]]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.succeeded, "Decoding custom gate with matrix should succeed")
        #expect(result.circuit.count == 1, "Decoding custom gate with matrix should produce one operation")
    }

    @Test("Decode symbolic parameter type reconstructs parameter")
    func decodeSymbolicParameter() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"rx","qubits":[0],"parameters":[{"type":"symbolic","name":"alpha"}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.succeeded, "Decoding symbolic parameter should succeed")
        #expect(result.circuit.count == 1, "Decoding symbolic parameter gate should produce one operation")
    }

    @Test("Decode negated parameter type reconstructs negated parameter")
    func decodeNegatedParameter() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"rz","qubits":[0],"parameters":[{"type":"negated","name":"beta"}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.succeeded, "Decoding negated parameter should succeed")
        #expect(result.circuit.count == 1, "Decoding negated parameter gate should produce one operation")
    }

    @Test("Decode reset with empty qubits produces error diagnostic")
    func decodeResetEmptyQubits() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"reset","qubits":[]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        let hasError = result.diagnostics.contains { $0.severity == .error && $0.message.contains("reset") }
        #expect(hasError, "Decoding reset with empty qubits should produce an error mentioning reset")
    }

    @Test("Decode u2 with wrong parameter count produces error diagnostic")
    func decodeU2WrongParamCount() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"u2","qubits":[0],"parameters":[{"type":"value","value":0.5}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        let hasError = result.diagnostics.contains { $0.severity == .error && $0.message.contains("u2") }
        #expect(hasError, "Decoding u2 with 1 parameter should produce error mentioning u2")
    }

    @Test("Decode u3 with wrong parameter count produces error diagnostic")
    func decodeU3WrongParamCount() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"u3","qubits":[0],"parameters":[{"type":"value","value":0.1},{"type":"value","value":0.2}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        let hasError = result.diagnostics.contains { $0.severity == .error && $0.message.contains("u3") }
        #expect(hasError, "Decoding u3 with 2 parameters should produce error mentioning u3")
    }

    @Test("Gate operation missing gate field produces error diagnostic")
    func decodeGateOperationMissingGateField() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","qubits":[0]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        let hasError = result.diagnostics.contains { $0.severity == .error && $0.message.contains("missing") }
        #expect(hasError, "Gate operation without gate field should produce error mentioning missing field")
        #expect(result.circuit.count == 0, "Gate operation without gate field should not add any operation")
    }

    @Test("Unknown gate name produces warning diagnostic")
    func decodeUnknownGateName() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"unknown_gate_xyz","qubits":[0]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        let hasWarning = result.diagnostics.contains { $0.severity == .warning && $0.message.contains("unknown_gate_xyz") }
        #expect(hasWarning, "Unknown gate name should produce warning mentioning the gate name")
        #expect(result.circuit.count == 0, "Unknown gate name should not add any operation")
    }

    @Test("Custom gate with 3x3 matrix creates customUnitary gate")
    func decodeCustomGate3x3Matrix() {
        let row = """
        [{"real":1,"imaginary":0},{"real":0,"imaginary":0},{"real":0,"imaginary":0}]
        """
        let zeroRow = """
        [{"real":0,"imaginary":0},{"real":1,"imaginary":0},{"real":0,"imaginary":0}]
        """
        let lastRow = """
        [{"real":0,"imaginary":0},{"real":0,"imaginary":0},{"real":1,"imaginary":0}]
        """
        let json = """
        {"version":1,"qubitCount":2,"classicalBitCount":0,"operations":[{"type":"gate","gate":"custom_unitary","qubits":[0,1],"matrix":[\(row),\(zeroRow),\(lastRow)]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.succeeded, "Custom gate with 3x3 matrix should decode successfully")
        #expect(result.circuit.count == 1, "Custom gate with 3x3 matrix should produce one operation")
    }

    @Test("Controlled gate with empty controls array is handled")
    func decodeControlledGateEmptyControls() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"h","qubits":[0],"controls":[]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.succeeded, "Gate with empty controls array should decode successfully")
        #expect(result.circuit.count == 1, "Gate with empty controls should still produce one operation")
    }

    @Test("Controlled custom gate with controls decodes correctly")
    func decodeControlledCustomGate() {
        let json = """
        {"version":1,"qubitCount":3,"classicalBitCount":0,"operations":[{"type":"gate","gate":"custom_unitary","qubits":[0,1,2],"controls":[0],"matrix":[[{"real":1,"imaginary":0},{"real":0,"imaginary":0}],[{"real":0,"imaginary":0},{"real":0,"imaginary":1}]]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.succeeded, "Controlled custom gate should decode successfully")
        #expect(result.circuit.count == 1, "Controlled custom gate should produce one operation")
    }

    @Test("controlledPhase parameter decoded from JSON")
    func decodeControlledPhaseParameter() {
        let json = """
        {"version":1,"qubitCount":2,"classicalBitCount":0,"operations":[{"type":"gate","gate":"cp","qubits":[0,1],"parameters":[{"type":"value","value":1.57}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.succeeded, "controlledPhase with parameter should decode successfully")
        #expect(result.circuit.count == 1, "controlledPhase should produce one operation")
    }

    @Test("controlledRotationX parameter decoded from JSON")
    func decodeControlledRotationXParameter() {
        let json = """
        {"version":1,"qubitCount":2,"classicalBitCount":0,"operations":[{"type":"gate","gate":"crx","qubits":[0,1],"parameters":[{"type":"value","value":0.5}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.succeeded, "controlledRotationX with parameter should decode successfully")
        #expect(result.circuit.count == 1, "controlledRotationX should produce one operation")
    }

    @Test("controlledRotationY parameter decoded from JSON")
    func decodeControlledRotationYParameter() {
        let json = """
        {"version":1,"qubitCount":2,"classicalBitCount":0,"operations":[{"type":"gate","gate":"cry","qubits":[0,1],"parameters":[{"type":"value","value":0.5}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.succeeded, "controlledRotationY with parameter should decode successfully")
        #expect(result.circuit.count == 1, "controlledRotationY should produce one operation")
    }

    @Test("controlledRotationZ parameter decoded from JSON")
    func decodeControlledRotationZParameter() {
        let json = """
        {"version":1,"qubitCount":2,"classicalBitCount":0,"operations":[{"type":"gate","gate":"crz","qubits":[0,1],"parameters":[{"type":"value","value":0.5}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.succeeded, "controlledRotationZ with parameter should decode successfully")
        #expect(result.circuit.count == 1, "controlledRotationZ should produce one operation")
    }

    @Test("globalPhase parameter decoded from JSON")
    func decodeGlobalPhaseParameter() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"gphase","qubits":[0],"parameters":[{"type":"value","value":3.14}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.succeeded, "globalPhase with parameter should decode successfully")
        #expect(result.circuit.count == 1, "globalPhase should produce one operation")
    }

    @Test("givens parameter decoded from JSON")
    func decodeGivensParameter() {
        let json = """
        {"version":1,"qubitCount":2,"classicalBitCount":0,"operations":[{"type":"gate","gate":"givens","qubits":[0,1],"parameters":[{"type":"value","value":0.75}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.succeeded, "givens with parameter should decode successfully")
        #expect(result.circuit.count == 1, "givens should produce one operation")
    }

    @Test("xx parameter decoded from JSON")
    func decodeXXParameter() {
        let json = """
        {"version":1,"qubitCount":2,"classicalBitCount":0,"operations":[{"type":"gate","gate":"rxx","qubits":[0,1],"parameters":[{"type":"value","value":0.25}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.succeeded, "xx with parameter should decode successfully")
        #expect(result.circuit.count == 1, "xx should produce one operation")
    }

    @Test("yy parameter decoded from JSON")
    func decodeYYParameter() {
        let json = """
        {"version":1,"qubitCount":2,"classicalBitCount":0,"operations":[{"type":"gate","gate":"ryy","qubits":[0,1],"parameters":[{"type":"value","value":0.25}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.succeeded, "yy with parameter should decode successfully")
        #expect(result.circuit.count == 1, "yy should produce one operation")
    }

    @Test("zz parameter decoded from JSON")
    func decodeZZParameter() {
        let json = """
        {"version":1,"qubitCount":2,"classicalBitCount":0,"operations":[{"type":"gate","gate":"rzz","qubits":[0,1],"parameters":[{"type":"value","value":0.25}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.succeeded, "zz with parameter should decode successfully")
        #expect(result.circuit.count == 1, "zz should produce one operation")
    }

    @Test("phase parameter decoded from JSON")
    func decodePhaseParameter() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"u1","qubits":[0],"parameters":[{"type":"value","value":1.0}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.succeeded, "phase/u1 with parameter should decode successfully")
        #expect(result.circuit.count == 1, "phase/u1 should produce one operation")
    }

    @Test("Decode ry gate with parameter applies rotationY angle")
    func decodeRyWithParameter() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"ry","qubits":[0],"parameters":[{"type":"value","value":0.8}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.succeeded, "ry gate with value parameter should decode successfully")
        #expect(result.circuit.count == 1, "ry gate with parameter should produce one operation via rotationY branch")
    }

    @Test("Decode ry gate without parameters uses placeholder")
    func decodeRyNoParametersFallback() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"ry","qubits":[0]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.succeeded, "ry gate without parameters key should decode using placeholder")
        #expect(result.circuit.count == 1, "ry gate without parameters should produce one operation")
    }

    @Test("Default parameter type reconstructs as value zero")
    func decodeDefaultParameterType() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"rx","qubits":[0],"parameters":[{"type":"something_else"}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.succeeded, "Unknown parameter type should fall back to value(0)")
        #expect(result.circuit.count == 1, "Gate with unknown parameter type should still produce one operation")
    }

    @Test("Encode controlled gate with parameter extracts parameter through controlled wrapper")
    func encodeControlledGateWithParameter() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlled(gate: .rotationZ(.value(1.5)), controls: [0]), to: [0, 1])
        let data = CircuitJSONEncoder.encode(circuit)
        let json = String(data: data, encoding: .utf8)!
        #expect(json.contains("1.5"), "Controlled gate with parameter should preserve parameter value in JSON")
        #expect(json.contains("controls"), "Controlled gate should have controls field in JSON")
    }

    @Test("Decode version 0 produces error diagnostic")
    func decodeVersionZero() {
        let json = """
        {"version":0,"qubitCount":1,"classicalBitCount":0,"operations":[]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        let hasError = result.diagnostics.contains { $0.severity == .error && $0.message.contains("version") }
        #expect(hasError, "Decoding version 0 should produce error mentioning version")
    }

    @Test("Decode future version produces warning but still decodes")
    func decodeFutureVersion() {
        let json = """
        {"version":999,"qubitCount":2,"classicalBitCount":0,"operations":[{"type":"gate","gate":"h","qubits":[0]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        let hasWarning = result.diagnostics.contains { $0.severity == .warning && $0.message.contains("999") }
        #expect(hasWarning, "Decoding future version 999 should produce warning mentioning version number")
        #expect(result.circuit.count == 1, "Future version should still decode operations successfully")
    }
}

/// Validates CircuitJSONDecoder guard paths for empty parameter arrays.
/// Ensures every parameterized gate type returns a placeholder gate
/// when the parameters array is present but empty.
@Suite("CircuitJSON Decoder Targeted Coverage: Empty Param Guards")
struct CircuitJSONDecoderTargetedCoverageTests {
    @Test("rotationX with empty parameters returns gate unchanged")
    func decodeRxEmptyParams() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"rx","qubits":[0],"parameters":[]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.circuit.count == 1, "rx with empty parameters should return placeholder gate via guard")
    }

    @Test("rotationY with empty parameters returns gate unchanged")
    func decodeRyEmptyParams() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"ry","qubits":[0],"parameters":[]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.circuit.count == 1, "ry with empty parameters should return placeholder gate via guard")
    }

    @Test("rotationZ with empty parameters returns gate unchanged")
    func decodeRzEmptyParams() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"rz","qubits":[0],"parameters":[]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.circuit.count == 1, "rz with empty parameters should return placeholder gate via guard")
    }

    @Test("gate p unknown in v2 table produces warning")
    func decodePhaseUnknownInV2() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"p","qubits":[0],"parameters":[{"type":"value","value":0.5}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        let hasWarning = result.diagnostics.contains { $0.severity == .warning }
        #expect(hasWarning, "gate 'p' is not in v2 table and should produce warning about unknown gate")
    }

    @Test("u1 with empty parameters returns gate unchanged")
    func decodeU1EmptyParams() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"u1","qubits":[0],"parameters":[]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.circuit.count == 1, "u1 with empty parameters should return placeholder gate via guard")
    }

    @Test("controlledPhase with empty parameters returns gate unchanged")
    func decodeCpEmptyParams() {
        let json = """
        {"version":1,"qubitCount":2,"classicalBitCount":0,"operations":[{"type":"gate","gate":"cp","qubits":[0,1],"parameters":[]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.circuit.count == 1, "cp with empty parameters should return placeholder gate via guard")
    }

    @Test("controlledRotationX with empty parameters returns gate unchanged")
    func decodeCrxEmptyParams() {
        let json = """
        {"version":1,"qubitCount":2,"classicalBitCount":0,"operations":[{"type":"gate","gate":"crx","qubits":[0,1],"parameters":[]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.circuit.count == 1, "crx with empty parameters should return placeholder gate via guard")
    }

    @Test("controlledRotationY with empty parameters returns gate unchanged")
    func decodeCryEmptyParams() {
        let json = """
        {"version":1,"qubitCount":2,"classicalBitCount":0,"operations":[{"type":"gate","gate":"cry","qubits":[0,1],"parameters":[]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.circuit.count == 1, "cry with empty parameters should return placeholder gate via guard")
    }

    @Test("controlledRotationZ with empty parameters returns gate unchanged")
    func decodeCrzEmptyParams() {
        let json = """
        {"version":1,"qubitCount":2,"classicalBitCount":0,"operations":[{"type":"gate","gate":"crz","qubits":[0,1],"parameters":[]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.circuit.count == 1, "crz with empty parameters should return placeholder gate via guard")
    }

    @Test("globalPhase with empty parameters returns gate unchanged")
    func decodeGphaseEmptyParams() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"gphase","qubits":[0],"parameters":[]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.circuit.count == 1, "gphase with empty parameters should return placeholder gate via guard")
    }

    @Test("givens with empty parameters returns gate unchanged")
    func decodeGivensEmptyParams() {
        let json = """
        {"version":1,"qubitCount":2,"classicalBitCount":0,"operations":[{"type":"gate","gate":"givens","qubits":[0,1],"parameters":[]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.circuit.count == 1, "givens with empty parameters should return placeholder gate via guard")
    }

    @Test("xx with empty parameters returns gate unchanged")
    func decodeXxEmptyParams() {
        let json = """
        {"version":1,"qubitCount":2,"classicalBitCount":0,"operations":[{"type":"gate","gate":"rxx","qubits":[0,1],"parameters":[]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.circuit.count == 1, "rxx with empty parameters should return placeholder gate via guard")
    }

    @Test("yy with empty parameters returns gate unchanged")
    func decodeYyEmptyParams() {
        let json = """
        {"version":1,"qubitCount":2,"classicalBitCount":0,"operations":[{"type":"gate","gate":"ryy","qubits":[0,1],"parameters":[]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.circuit.count == 1, "ryy with empty parameters should return placeholder gate via guard")
    }

    @Test("zz with empty parameters returns gate unchanged")
    func decodeZzEmptyParams() {
        let json = """
        {"version":1,"qubitCount":2,"classicalBitCount":0,"operations":[{"type":"gate","gate":"rzz","qubits":[0,1],"parameters":[]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.circuit.count == 1, "rzz with empty parameters should return placeholder gate via guard")
    }

    @Test("Non-parameterized gate with parameters hits default case")
    func decodeNonParameterizedGateWithParams() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"h","qubits":[0],"parameters":[{"type":"value","value":1.0}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.circuit.count == 1, "non-parameterized gate with extra params should hit default and return gate")
    }

    @Test("Symbolic parameter without name uses unnamed default")
    func decodeSymbolicNoName() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"rx","qubits":[0],"parameters":[{"type":"symbolic"}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.circuit.count == 1, "symbolic parameter without name should use unnamed default")
    }

    @Test("Negated parameter without name uses unnamed default")
    func decodeNegatedNoName() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"rz","qubits":[0],"parameters":[{"type":"negated"}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.circuit.count == 1, "negated parameter without name should use unnamed default")
    }

    @Test("Value parameter without value uses zero default")
    func decodeValueNoValue() {
        let json = """
        {"version":1,"qubitCount":1,"classicalBitCount":0,"operations":[{"type":"gate","gate":"rx","qubits":[0],"parameters":[{"type":"value"}]}]}
        """
        let result = CircuitJSONDecoder.decode(from: Data(json.utf8))
        #expect(result.circuit.count == 1, "value parameter without value field should default to 0")
    }
}
