// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Validates QASM2Exporter header generation and register declarations.
/// Ensures exported programs begin with correct version, include directive,
/// and properly sized quantum and classical register declarations.
@Suite("QASM2 Header and Register Format")
struct QASM2ExporterHeaderTests {
    @Test("Empty circuit exports correct header with version, include, qreg, creg")
    func emptyCircuitHeader() {
        let circuit = QuantumCircuit(qubits: 2)
        let qasm = QASM2Exporter.export(circuit)
        let lines = qasm.components(separatedBy: "\n")

        #expect(lines.count >= 4, "Header must contain at least 4 lines: version, include, qreg, creg")
        #expect(lines[0] == "OPENQASM 2.0;", "First line must be QASM version declaration")
        #expect(lines[1] == "include \"qelib1.inc\";", "Second line must include standard gate library")
        #expect(lines[2] == "qreg q[2];", "Third line must declare quantum register with correct qubit count")
        #expect(lines[3] == "creg c[2];", "Fourth line must declare classical register matching qubit count")
    }

    @Test("Output starts with OPENQASM 2.0;")
    func outputStartsWithVersion() {
        let circuit = QuantumCircuit(qubits: 1)
        let qasm = QASM2Exporter.export(circuit)

        #expect(qasm.hasPrefix("OPENQASM 2.0;"), "Exported QASM must begin with version declaration")
    }

    @Test("Output contains include qelib1.inc")
    func outputContainsInclude() {
        let circuit = QuantumCircuit(qubits: 1)
        let qasm = QASM2Exporter.export(circuit)

        #expect(qasm.contains("include \"qelib1.inc\";"), "Exported QASM must include standard gate library")
    }

    @Test("Circuit with 4 qubits exports qreg q[4] and creg c[4]")
    func multipleQubitCount() {
        let circuit = QuantumCircuit(qubits: 4)
        let qasm = QASM2Exporter.export(circuit)

        #expect(qasm.contains("qreg q[4];"), "Quantum register must match circuit qubit count of 4")
        #expect(qasm.contains("creg c[4];"), "Classical register must match circuit qubit count of 4")
    }
}

/// Validates QASM2Exporter serialization of individual gate operations.
/// Ensures standard gates like Hadamard, CNOT, and parameterized rotations
/// produce correct QASM 2.0 gate statement syntax with proper qubit references.
@Suite("QASM2 Single Gate Export")
struct QASM2ExporterSingleGateTests {
    @Test("Hadamard gate exports as h q[0];")
    func hadamardGate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let qasm = QASM2Exporter.export(circuit)

        #expect(qasm.contains("h q[0];"), "Hadamard gate must serialize as 'h q[0];'")
    }

    @Test("CNOT gate exports as cx q[0],q[1];")
    func cnotGate() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])
        let qasm = QASM2Exporter.export(circuit)

        #expect(qasm.contains("cx q[0],q[1];"), "CNOT gate must serialize as 'cx q[0],q[1];'")
    }

    @Test("RotationZ gate exports with rz prefix and parameter")
    func rotationZGate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(.value(.pi / 4)), to: 0)
        let qasm = QASM2Exporter.export(circuit)

        #expect(qasm.contains("rz(pi/4) q[0];"), "RotationZ(pi/4) must serialize as 'rz(pi/4) q[0];'")
    }
}

/// Validates QASM2Exporter output for multi-gate circuits, custom gates,
/// and controlled gate compositions. Ensures correct gate ordering,
/// custom gate declarations, and controlled gate serialization format.
@Suite("QASM2 Multi-Gate and Custom Gate Export")
struct QASM2ExporterMultiGateTests {
    @Test("Multi-gate circuit preserves gate ordering")
    func multiGateOrdering() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.pauliX, to: 1)
        let qasm = QASM2Exporter.export(circuit)

        let hIndex = qasm.range(of: "h q[0];")
        let cxIndex = qasm.range(of: "cx q[0],q[1];")
        let xIndex = qasm.range(of: "x q[1];")

        #expect(hIndex != nil, "Hadamard gate must appear in output")
        #expect(cxIndex != nil, "CNOT gate must appear in output")
        #expect(xIndex != nil, "Pauli-X gate must appear in output")
        #expect(hIndex!.lowerBound < cxIndex!.lowerBound, "Hadamard must appear before CNOT in output")
        #expect(cxIndex!.lowerBound < xIndex!.lowerBound, "CNOT must appear before Pauli-X in output")
    }

    @Test("Custom single-qubit gate emits gate declaration")
    func customGateDeclaration() {
        var circuit = QuantumCircuit(qubits: 1)
        let matrix: [[Complex<Double>]] = [
            [Complex(1.0, 0.0), Complex(0.0, 0.0)],
            [Complex(0.0, 0.0), Complex(0.0, 1.0)],
        ]
        circuit.append(.customSingleQubit(matrix: matrix), to: [0])
        let qasm = QASM2Exporter.export(circuit)

        #expect(qasm.contains("gate custom_0"), "Custom gate must emit a gate declaration with name custom_0")
        #expect(qasm.contains("custom_0 q[0];"), "Custom gate must emit application statement referencing q[0]")
    }

    @Test("Controlled gate serializes with c_ prefix")
    func controlledGateSerialization() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlled(gate: .pauliX, controls: [0]), to: [0, 1])
        let qasm = QASM2Exporter.export(circuit)

        #expect(qasm.contains("c_x q[0],q[1];"), "Controlled-X gate must serialize with 'c_x' prefix in QASM 2.0")
    }
}

/// Validates QASM2Exporter output for parameterized and special gates.
/// Ensures u1, u2, u3, reset, symbolic parameters, custom gates,
/// and angle formatting produce correct QASM 2.0 syntax.
@Suite("QASM2 Coverage: Parameterized and Special Gate Export")
struct QASM2ExporterCoverageTests {
    @Test("U1 gate exports with u1 prefix and parameter")
    func u1GateExport() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u1(lambda: .value(.pi / 4)), to: 0)
        let qasm = QASM2Exporter.export(circuit)

        #expect(qasm.contains("u1("), "U1 gate must serialize with 'u1(' prefix")
    }

    @Test("U2 gate exports with u2 prefix and parameters")
    func u2GateExport() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u2(phi: .value(0.0), lambda: .value(.pi)), to: 0)
        let qasm = QASM2Exporter.export(circuit)

        #expect(qasm.contains("u2("), "U2 gate must serialize with 'u2(' prefix")
    }

    @Test("U3 gate exports with u3 prefix and parameters")
    func u3GateExport() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u3(theta: .value(.pi / 2), phi: .value(0.0), lambda: .value(.pi)), to: 0)
        let qasm = QASM2Exporter.export(circuit)

        #expect(qasm.contains("u3("), "U3 gate must serialize with 'u3(' prefix")
    }

    @Test("Reset operation exports as reset q[n];")
    func resetExport() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.reset, to: 0)
        let qasm = QASM2Exporter.export(circuit)

        #expect(qasm.contains("reset q[0];"), "Reset operation must serialize as 'reset q[0];'")
    }

    @Test("Symbolic parameter name appears in exported output")
    func symbolicParameterExport() {
        var circuit = QuantumCircuit(qubits: 1)
        let theta = Parameter(name: "theta")
        circuit.append(.rotationY(.parameter(theta)), to: 0)
        let qasm = QASM2Exporter.export(circuit)

        #expect(qasm.contains("theta"), "Symbolic parameter name 'theta' must appear in exported QASM output")
    }

    @Test("Custom single-qubit gate emits gate declaration with custom_ prefix")
    func customGateExport() {
        var circuit = QuantumCircuit(qubits: 1)
        let matrix: [[Complex<Double>]] = [
            [Complex(0.0, 0.0), Complex(1.0, 0.0)],
            [Complex(1.0, 0.0), Complex(0.0, 0.0)],
        ]
        circuit.append(.customSingleQubit(matrix: matrix), to: [0])
        let qasm = QASM2Exporter.export(circuit)

        #expect(qasm.contains("gate custom_"), "Custom gate declaration must contain 'gate custom_' prefix")
    }

    @Test("Gate with angle -pi/4 formats as -pi/4")
    func negativePiOver4Format() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(.value(-.pi / 4)), to: 0)
        let qasm = QASM2Exporter.export(circuit)

        #expect(qasm.contains("-pi/4"), "Angle of -pi/4 must format as '-pi/4' in exported QASM output")
    }

    @Test("Negated parameter serializes with minus prefix")
    func negatedParameterExport() {
        var circuit = QuantumCircuit(qubits: 1)
        let alpha = Parameter(name: "alpha")
        circuit.append(.rotationX(.negatedParameter(alpha)), to: 0)
        let qasm = QASM2Exporter.export(circuit)

        #expect(qasm.contains("-alpha"), "Negated parameter must serialize with minus prefix as '-alpha'")
    }

    @Test("formatDouble formats pi/2 as symbolic pi/2")
    func piOver2Format() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(.value(.pi / 2)), to: 0)
        let qasm = QASM2Exporter.export(circuit)

        #expect(qasm.contains("pi/2"), "Angle pi/2 must format as symbolic 'pi/2' in QASM 2.0 output")
    }

    @Test("formatDouble formats pi/4 as symbolic pi/4")
    func piOver4Format() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(.value(.pi / 4)), to: 0)
        let qasm = QASM2Exporter.export(circuit)

        #expect(qasm.contains("pi/4"), "Angle pi/4 must format as symbolic 'pi/4' in QASM 2.0 output")
    }

    @Test("formatDouble falls through to String(value) for non-special angle")
    func nonSpecialAngleFormat() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(.value(1.234)), to: 0)
        let qasm = QASM2Exporter.export(circuit)

        #expect(qasm.contains("1.234"), "Non-special angle 1.234 must fall through to String(value) formatting")
    }
}
