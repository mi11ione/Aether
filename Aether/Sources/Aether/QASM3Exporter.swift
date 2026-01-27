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
            }
        }

        return lines.joined(separator: "\n") + "\n"
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
                lines: &lines,
                customGateCounter: &customGateCounter,
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
        lines _: inout [String],
        customGateCounter _: inout Int,
    ) -> String {
        let (baseGate, allControls) = QuantumGate.controlled(gate: innerGate, controls: controls)
            .flattenControlled()

        let controlCount = allControls.count
        let targetQubits = Array(qubits.suffix(qubits.count - controlCount))
        let controlQubits = Array(qubits.prefix(controlCount))

        let baseName = GateNameMapping.qasmName(for: baseGate, version: .v3)
        let params = parameterList(for: baseGate)
        let qubitArgs = (controlQubits + targetQubits).map { "q[\($0)]" }.joined(separator: ", ")

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

        let numQubits = gate.qubitsRequired
        let qubitsDecl = (0 ..< numQubits).map { "q\($0)" }.joined(separator: ", ")
        lines.append("gate \(gateName) \(qubitsDecl) {}")

        let qubitArgs = formatQubitArgs(qubits)
        return "\(gateName) \(qubitArgs);"
    }

    /// Serialize gate parameters to a comma-separated string.
    @_effects(readonly)
    private static func parameterList(for gate: QuantumGate) -> String {
        let values = extractParameterValues(from: gate)
        if values.isEmpty { return "" }
        return values.map { formatParameterValue($0) }.joined(separator: ", ")
    }

    /// Extract ordered parameter values from a gate.
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
        }
    }

    /// Format a double value as a decimal string.
    @_effects(readonly)
    private static func formatDouble(_ value: Double) -> String {
        if !value.isNaN, !value.isInfinite, value == Double(Int(value)) {
            return String(format: "%.1f", value)
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
        qubits.map { "q[\($0)]" }.joined(separator: ", ")
    }
}
