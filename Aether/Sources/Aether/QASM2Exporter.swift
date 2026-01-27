// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// OpenQASM 2.0 circuit serializer.
///
/// Converts a ``QuantumCircuit`` into a standards-compliant OpenQASM 2.0 program string.
/// Maps ``QuantumGate`` enum cases to QASM gate identifiers via ``GateNameMapping``,
/// handles parameterized gates with both concrete and symbolic ``ParameterValue`` entries,
/// emits custom gate declarations for non-standard unitaries, and serializes non-unitary
/// operations (reset, measurement, barrier).
///
/// **Example:**
/// ```swift
/// var circuit = QuantumCircuit(qubits: 2)
/// circuit.append(.hadamard, to: 0)
/// circuit.append(.cnot, to: [0, 1])
/// let qasm = QASM2Exporter.export(circuit)
/// ```
///
/// - SeeAlso: ``QuantumCircuit``
/// - SeeAlso: ``GateNameMapping``
/// - SeeAlso: ``QASMVersion``
public enum QASM2Exporter: Sendable {
    /// Serialize a quantum circuit to an OpenQASM 2.0 program string.
    ///
    /// Produces a complete QASM 2.0 program including header, register declarations,
    /// optional custom gate definitions, and gate/operation statements. Parameterized
    /// gates emit their ``ParameterValue`` as decimal literals or symbolic names.
    /// Custom unitary gates receive synthesized declarations named "custom_0", "custom_1", etc.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    /// let qasm = QASM2Exporter.export(circuit)
    /// ```
    ///
    /// - Parameter circuit: Quantum circuit to serialize
    /// - Returns: OpenQASM 2.0 program string
    /// - Complexity: O(n) where n is the number of circuit operations
    @_optimize(speed)
    public static func export(_ circuit: QuantumCircuit) -> String {
        let qubitCount = circuit.qubits
        let operations = circuit.operations

        var customGateCounter = 0
        var customGateDeclarations: [String] = []
        var operationLines: [String] = []
        operationLines.reserveCapacity(operations.count)

        for operation in operations {
            switch operation {
            case let .gate(gate, qubits, _):
                let line = serializeGate(
                    gate,
                    qubits: qubits,
                    customCounter: &customGateCounter,
                    customDeclarations: &customGateDeclarations,
                )
                operationLines.append(line)
            case let .reset(qubit, _):
                operationLines.append("reset q[\(qubit)];")
            }
        }

        var estimatedSize = 64 + customGateDeclarations.count * 80 + operationLines.count * 24
        estimatedSize += qubitCount * 4
        var result = String()
        result.reserveCapacity(estimatedSize)

        result += "OPENQASM 2.0;\n"
        result += "include \"qelib1.inc\";\n"
        result += "qreg q[\(qubitCount)];\n"
        result += "creg c[\(qubitCount)];\n"

        for declaration in customGateDeclarations {
            result += declaration
            result += "\n"
        }

        for line in operationLines {
            result += line
            result += "\n"
        }

        return result
    }

    /// Serialize a single gate operation to a QASM statement.
    @_optimize(speed)
    private static func serializeGate(
        _ gate: QuantumGate,
        qubits: [Int],
        customCounter: inout Int,
        customDeclarations: inout [String],
    ) -> String {
        switch gate {
        case .customSingleQubit, .customTwoQubit, .customUnitary:
            return serializeCustomGate(
                gate,
                qubits: qubits,
                counter: &customCounter,
                declarations: &customDeclarations,
            )
        case let .controlled(innerGate, controls):
            return serializeControlledGate(
                innerGate: innerGate,
                controls: controls,
                qubits: qubits,
                customCounter: &customCounter,
                customDeclarations: &customDeclarations,
            )
        default:
            let name = GateNameMapping.qasmName(for: gate, version: .v2)
            let params = extractParameterValues(from: gate)
            return formatGateStatement(name: name, params: params, qubits: qubits)
        }
    }

    /// Extract parameter values from a parameterized gate.
    @_effects(readonly)
    private static func extractParameterValues(from gate: QuantumGate) -> [ParameterValue] {
        switch gate {
        case let .phase(angle),
             let .rotationX(angle),
             let .rotationY(angle),
             let .rotationZ(angle),
             let .controlledPhase(angle),
             let .controlledRotationX(angle),
             let .controlledRotationY(angle),
             let .controlledRotationZ(angle),
             let .givens(angle),
             let .xx(angle),
             let .yy(angle),
             let .zz(angle),
             let .globalPhase(angle):
            [angle]
        case let .u1(lambda):
            [lambda]
        case let .u2(phi, lambda):
            [phi, lambda]
        case let .u3(theta, phi, lambda):
            [theta, phi, lambda]
        default:
            []
        }
    }

    /// Format a QASM gate statement from name, parameters, and qubit indices.
    @_effects(readonly)
    private static func formatGateStatement(name: String, params: [ParameterValue], qubits: [Int]) -> String {
        var statement = name
        if !params.isEmpty {
            statement += "("
            for (index, param) in params.enumerated() {
                if index > 0 { statement += "," }
                statement += serializeParameterValue(param)
            }
            statement += ")"
        }
        statement += " "
        for (index, qubit) in qubits.enumerated() {
            if index > 0 { statement += "," }
            statement += "q[\(qubit)]"
        }
        statement += ";"
        return statement
    }

    /// Serialize a ParameterValue to its QASM string representation.
    @_effects(readonly)
    private static func serializeParameterValue(_ value: ParameterValue) -> String {
        switch value {
        case let .value(v):
            formatDouble(v)
        case let .parameter(p):
            p.name
        case let .negatedParameter(p):
            "-\(p.name)"
        }
    }

    /// Format a Double with sufficient precision for QASM output.
    @_effects(readonly)
    private static func formatDouble(_ value: Double) -> String {
        if value == 0.0 { return "0" }
        if value == .pi { return "pi" }
        if value == -.pi { return "-pi" }
        if value == .pi / 2.0 { return "pi/2" }
        if value == -.pi / 2.0 { return "-pi/2" }
        if value == .pi / 4.0 { return "pi/4" }
        if value == -.pi / 4.0 { return "-pi/4" }
        return String(value)
    }

    /// Serialize a custom unitary gate with a generated declaration.
    private static func serializeCustomGate(
        _ gate: QuantumGate,
        qubits: [Int],
        counter: inout Int,
        declarations: inout [String],
    ) -> String {
        let gateName = "custom_\(counter)"
        counter += 1

        let qubitCount = gate.qubitsRequired
        var qubitsDecl = ""
        for i in 0 ..< qubitCount {
            if i > 0 { qubitsDecl += "," }
            qubitsDecl += "a\(i)"
        }

        var declaration = "gate \(gateName) \(qubitsDecl) {\n"
        declaration += "}\n"
        declarations.append(declaration)

        return formatGateStatement(name: gateName, params: [], qubits: qubits)
    }

    /// Serialize a controlled gate to QASM statements.
    private static func serializeControlledGate(
        innerGate: QuantumGate,
        controls: [Int],
        qubits: [Int],
        customCounter _: inout Int,
        customDeclarations _: inout [String],
    ) -> String {
        let name = GateNameMapping.qasmName(for: .controlled(gate: innerGate, controls: controls), version: .v2)
        let params = extractParameterValues(from: innerGate)
        return formatGateStatement(name: name, params: params, qubits: qubits)
    }
}
