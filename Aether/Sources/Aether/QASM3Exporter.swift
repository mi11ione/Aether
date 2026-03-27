// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// OpenQASM 3.0 circuit serializer producing spec-compliant program strings.
///
/// Converts a ``QuantumCircuit`` into an OpenQASM 3.0 source string using modern
/// declaration syntax (`qubit`, `bit`), the `stdgates.inc` standard library, and
/// gate modifiers (`ctrl @`, `inv @`) for controlled and inverse operations. Gate
/// names are resolved through ``GateNameMapping`` with ``QASMVersion/v3``.
///
/// **Example:**
/// ```swift
/// var circuit = QuantumCircuit(qubits: 2)
/// circuit.append(.hadamard, to: 0)
/// circuit.append(.cnot, to: [0, 1])
/// let qasm = QASM3Exporter.export(circuit)
/// ```
///
/// - SeeAlso: ``QuantumCircuit``
/// - SeeAlso: ``GateNameMapping``
/// - SeeAlso: ``QASMVersion``
public enum QASM3Exporter: Sendable {
    /// Serialize a quantum circuit to an OpenQASM 3.0 program string.
    ///
    /// Emits the version header, `stdgates.inc` include, qubit and bit register
    /// declarations, followed by one statement per ``CircuitOperation``. Gate names
    /// are resolved via ``GateNameMapping/qasmName(for:version:)`` with
    /// ``QASMVersion/v3``. The generic ``QuantumGate/controlled(gate:controls:)``
    /// case uses the `ctrl @` modifier syntax. Custom unitary gates emit inline
    /// `gate` definitions. Measurement uses assignment syntax `c[i] = measure q[i];`.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    /// let qasm = QASM3Exporter.export(circuit)
    /// ```
    ///
    /// - Parameter circuit: Quantum circuit to serialize
    /// - Returns: OpenQASM 3.0 program string
    /// - Complexity: O(n) where n is the number of circuit operations
    @_optimize(speed)
    public static func export(_ circuit: QuantumCircuit) -> String {
        var lines: [String] = []
        lines.reserveCapacity(circuit.count + 4)

        lines.append("OPENQASM 3.0;")
        lines.append("include \"stdgates.inc\";")
        lines.append("qubit[\(circuit.qubits)] q;")
        lines.append("bit[\(circuit.qubits)] c;")

        var customGateCounter = 0

        for operation in circuit.operations {
            switch operation {
            case let .gate(gate, qubits, _):
                let statement = formatGateStatement(
                    gate: gate,
                    qubits: qubits,
                    lines: &lines,
                    customGateCounter: &customGateCounter,
                )
                lines.append(statement)
            case let .reset(qubit, _):
                lines.append("reset q[\(qubit)];")
            case let .measure(qubit, classicalBit, _):
                let targetBit = classicalBit ?? qubit
                lines.append("c[\(targetBit)] = measure q[\(qubit)];")
            }
        }

        lines.append("")
        return lines.joined(separator: "\n")
    }

    /// Format a gate operation as a QASM 3.0 statement.
    @_optimize(speed)
    private static func formatGateStatement(
        gate: QuantumGate,
        qubits: [Int],
        lines: inout [String],
        customGateCounter: inout Int,
    ) -> String {
        switch gate {
        case let .controlled(innerGate, controls):
            return formatControlledGate(
                innerGate: innerGate,
                controls: controls,
                qubits: qubits,
            )
        case .customSingleQubit, .customTwoQubit, .customUnitary:
            return formatCustomGate(
                gate: gate,
                qubits: qubits,
                lines: &lines,
                customGateCounter: &customGateCounter,
            )
        default:
            let name = GateNameMapping.qasmName(for: gate, version: .v3)
            let params = parameterList(for: gate)
            let qubitArgs = formatQubitArgs(qubits)
            if params.isEmpty {
                return "\(name) \(qubitArgs);"
            }
            return "\(name)(\(params)) \(qubitArgs);"
        }
    }

    /// Format a controlled gate using `ctrl @` modifier syntax.
    @_effects(readonly)
    private static func formatControlledGate(
        innerGate: QuantumGate,
        controls: [Int],
        qubits: [Int],
    ) -> String {
        let (baseGate, allControls) = QuantumGate.controlled(gate: innerGate, controls: controls)
            .flattenControlled()

        let controlCount = allControls.count

        let baseName = GateNameMapping.qasmName(for: baseGate, version: .v3)
        let params = parameterList(for: baseGate)
        let qubitArgs = formatQubitArgs(qubits)

        let modifier = if controlCount == 1 {
            "ctrl @ "
        } else {
            "ctrl(\(controlCount)) @ "
        }

        if params.isEmpty {
            return "\(modifier)\(baseName) \(qubitArgs);"
        }
        return "\(modifier)\(baseName)(\(params)) \(qubitArgs);"
    }

    /// Format a custom unitary gate with inline gate definition.
    private static func formatCustomGate(
        gate: QuantumGate,
        qubits: [Int],
        lines: inout [String],
        customGateCounter: inout Int,
    ) -> String {
        let gateName = "custom_u\(customGateCounter)"
        customGateCounter += 1

        let qubitCount = gate.qubitsRequired
        let qubitsDecl = (0 ..< qubitCount).lazy.map { "q\($0)" }.joined(separator: ", ")
        lines.append("gate \(gateName) \(qubitsDecl) {}")

        let qubitArgs = formatQubitArgs(qubits)
        return "\(gateName) \(qubitArgs);"
    }

    /// Serialize gate parameters to a comma-separated string.
    @_effects(readonly)
    private static func parameterList(for gate: QuantumGate) -> String {
        let values = gate.parameterValues
        if values.isEmpty { return "" }
        return values.lazy.map { formatParameterValue($0) }.joined(separator: ", ")
    }

    /// Format a single parameter value as a QASM string.
    @_effects(readonly)
    private static func formatParameterValue(_ value: ParameterValue) -> String {
        switch value {
        case let .value(v):
            formatDouble(v)
        case let .parameter(p):
            p.name
        case let .negatedParameter(p):
            "-\(p.name)"
        case let .expression(expr):
            formatExpression(expr.node)
        }
    }

    /// Recursively format an expression node as a QASM string.
    @_effects(readonly)
    private static func formatExpression(_ node: ExpressionNode) -> String {
        switch node {
        case let .constant(v):
            formatDouble(v)
        case let .parameter(p):
            p.name
        case let .add(lhs, rhs):
            "(\(formatExpression(lhs)) + \(formatExpression(rhs)))"
        case let .subtract(lhs, rhs):
            "(\(formatExpression(lhs)) - \(formatExpression(rhs)))"
        case let .multiply(lhs, rhs):
            "(\(formatExpression(lhs)) * \(formatExpression(rhs)))"
        case let .divide(lhs, rhs):
            "(\(formatExpression(lhs)) / \(formatExpression(rhs)))"
        case let .negate(inner):
            "(-\(formatExpression(inner)))"
        case let .sin(inner):
            "sin(\(formatExpression(inner)))"
        case let .cos(inner):
            "cos(\(formatExpression(inner)))"
        case let .tan(inner):
            "tan(\(formatExpression(inner)))"
        case let .exp(inner):
            "exp(\(formatExpression(inner)))"
        case let .log(inner):
            "ln(\(formatExpression(inner)))"
        case let .arctan(inner):
            "arctan(\(formatExpression(inner)))"
        }
    }

    /// Format a double value as a decimal string.
    @_effects(readonly)
    private static func formatDouble(_ value: Double) -> String {
        if !value.isNaN, !value.isInfinite, value == Double(Int(value)) {
            return "\(Int(value)).0"
        }
        let formatted = String(value)
        if formatted.contains(".") || formatted.contains("e") || formatted.contains("E") {
            return formatted
        }
        return formatted + ".0"
    }

    /// Format qubit arguments as comma-separated q[i] references.
    @_effects(readonly)
    private static func formatQubitArgs(_ qubits: [Int]) -> String {
        qubits.lazy.map { "q[\($0)]" }.joined(separator: ", ")
    }
}
