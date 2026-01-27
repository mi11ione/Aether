// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Decode ``QuantumCircuit`` from JSON ``Data`` using the schema defined by ``CircuitJSON``.
///
/// Parses JSON data into an internal ``CircuitJSONSchema``, validates the schema version,
/// reconstructs gate operations with parameters, and returns a ``ParseResult`` containing the
/// best-effort circuit alongside any diagnostics. Malformed JSON produces an empty circuit with
/// an error diagnostic rather than throwing.
///
/// **Example:**
/// ```swift
/// let json = "{\"version\":1,\"qubitCount\":2,\"classicalBitCount\":0,\"operations\":[]}".data(using: .utf8)!
/// let result = CircuitJSONDecoder.decode(from: json)
/// print(result.succeeded)  // true
/// print(result.circuit.qubits)  // 2
/// ```
///
/// - SeeAlso: ``CircuitJSON``
/// - SeeAlso: ``ParseResult``
/// - SeeAlso: ``ParseDiagnostic``
public enum CircuitJSONDecoder {
    /// Decode a ``QuantumCircuit`` from JSON ``Data``.
    ///
    /// Attempts to parse the provided data as a ``CircuitJSONSchema``, validate the schema
    /// version, and convert each operation into the corresponding ``CircuitOperation``. All
    /// diagnostics are collected and returned alongside the best-effort circuit, enabling
    /// partial recovery from malformed input.
    ///
    /// **Example:**
    /// ```swift
    /// let data = try! JSONSerialization.data(withJSONObject: ["version": 1, "qubitCount": 3, "classicalBitCount": 0, "operations": []])
    /// let result = CircuitJSONDecoder.decode(from: data)
    /// print(result.circuit.qubits)  // 3
    /// ```
    ///
    /// - Parameter data: JSON-encoded circuit data conforming to the ``CircuitJSON`` schema
    /// - Returns: ``ParseResult`` containing the decoded circuit and any diagnostics
    ///
    /// - Complexity: O(n) where n is the number of operations in the schema
    @_optimize(speed)
    public static func decode(from data: Data) -> ParseResult {
        let (schema, decodeDiagnostics) = decodeSchema(from: data)

        guard let schema else {
            return ParseResult(
                circuit: QuantumCircuit(qubits: 1),
                diagnostics: decodeDiagnostics,
            )
        }

        var diagnostics = decodeDiagnostics
        validateVersion(schema.version, diagnostics: &diagnostics)

        let qubitCount = max(schema.qubitCount, 1)
        var circuit = QuantumCircuit(qubits: qubitCount)

        for (index, operation) in schema.operations.enumerated() {
            convertOperation(
                operation,
                index: index,
                circuit: &circuit,
                diagnostics: &diagnostics,
            )
        }

        return ParseResult(circuit: circuit, diagnostics: diagnostics)
    }

    /// Bridge Foundation's throwing JSONDecoder into diagnostic-based result.
    private static func decodeSchema(from data: Data) -> (CircuitJSONSchema?, [ParseDiagnostic]) {
        do {
            let decoder = JSONDecoder()
            let schema = try decoder.decode(CircuitJSONSchema.self, from: data)
            return (schema, [])
        } catch {
            let diagnostic = ParseDiagnostic(
                line: 1,
                column: 1,
                message: "Failed to decode JSON: \(error.localizedDescription)",
                severity: .error,
            )
            return (nil, [diagnostic])
        }
    }

    /// Validate schema version and emit diagnostics for incompatible versions.
    private static func validateVersion(_ version: Int, diagnostics: inout [ParseDiagnostic]) {
        if version <= 0 {
            diagnostics.append(ParseDiagnostic(
                line: 1,
                column: 1,
                message: "Invalid schema version \(version); expected positive integer",
                severity: .error,
            ))
        } else if version > CircuitJSON.schemaVersion {
            diagnostics.append(ParseDiagnostic(
                line: 1,
                column: 1,
                message: "Schema version \(version) is newer than supported version \(CircuitJSON.schemaVersion); some features may not decode correctly",
                severity: .warning,
            ))
        }
    }

    /// Convert a single schema operation into a circuit operation.
    private static func convertOperation(
        _ operation: OperationSchema,
        index: Int,
        circuit: inout QuantumCircuit,
        diagnostics: inout [ParseDiagnostic],
    ) {
        switch operation.type {
        case "gate":
            convertGateOperation(operation, index: index, circuit: &circuit, diagnostics: &diagnostics)
        case "measurement":
            diagnostics.append(ParseDiagnostic(
                line: 1,
                column: 1,
                message: "Operation \(index): measurement operations are not represented in the circuit model; skipping",
                severity: .warning,
            ))
        case "barrier":
            diagnostics.append(ParseDiagnostic(
                line: 1,
                column: 1,
                message: "Operation \(index): barrier operations are not represented in the circuit model; skipping",
                severity: .warning,
            ))
        case "reset":
            convertResetOperation(operation, index: index, circuit: &circuit, diagnostics: &diagnostics)
        default:
            diagnostics.append(ParseDiagnostic(
                line: 1,
                column: 1,
                message: "Operation \(index): unknown operation type '\(operation.type)'; skipping",
                severity: .warning,
            ))
        }
    }

    /// Convert a gate-type operation schema into a circuit gate operation.
    private static func convertGateOperation(
        _ operation: OperationSchema,
        index: Int,
        circuit: inout QuantumCircuit,
        diagnostics: inout [ParseDiagnostic],
    ) {
        guard let gateName = operation.gate else {
            diagnostics.append(ParseDiagnostic(
                line: 1,
                column: 1,
                message: "Operation \(index): gate operation missing 'gate' field",
                severity: .error,
            ))
            return
        }

        if let matrix = operation.matrix, gateName == "custom_unitary" {
            convertCustomGateOperation(
                operation,
                index: index,
                matrix: matrix,
                circuit: &circuit,
                diagnostics: &diagnostics,
            )
            return
        }

        guard var gate = GateNameMapping.gate(forQASMName: gateName, version: .v2) else {
            diagnostics.append(ParseDiagnostic(
                line: 1,
                column: 1,
                message: "Operation \(index): unknown gate name '\(gateName)'; skipping",
                severity: .warning,
            ))
            return
        }

        if let schemaParameters = operation.parameters, !schemaParameters.isEmpty {
            gate = applyParameters(to: gate, schemaParameters: schemaParameters, index: index, diagnostics: &diagnostics)
        }

        if let controls = operation.controls, !controls.isEmpty {
            gate = .controlled(gate: gate, controls: controls)
        }

        circuit.append(gate, to: operation.qubits)
    }

    /// Convert a custom unitary gate from its matrix representation.
    private static func convertCustomGateOperation(
        _ operation: OperationSchema,
        index _: Int,
        matrix: [[ComplexSchema]],
        circuit: inout QuantumCircuit,
        diagnostics _: inout [ParseDiagnostic],
    ) {
        let converted: [[Complex<Double>]] = matrix.map { row in
            row.map { Complex($0.real, $0.imaginary) }
        }

        let dimension = converted.count
        var gate: QuantumGate = if dimension == 2 {
            .customSingleQubit(matrix: converted)
        } else if dimension == 4 {
            .customTwoQubit(matrix: converted)
        } else {
            .customUnitary(matrix: converted)
        }

        if let controls = operation.controls, !controls.isEmpty {
            gate = .controlled(gate: gate, controls: controls)
        }

        circuit.append(gate, to: operation.qubits)
    }

    /// Convert a reset-type operation schema into a circuit reset operation.
    private static func convertResetOperation(
        _ operation: OperationSchema,
        index: Int,
        circuit: inout QuantumCircuit,
        diagnostics: inout [ParseDiagnostic],
    ) {
        guard let qubit = operation.qubits.first else {
            diagnostics.append(ParseDiagnostic(
                line: 1,
                column: 1,
                message: "Operation \(index): reset operation missing qubit index",
                severity: .error,
            ))
            return
        }

        circuit.append(.reset, to: qubit)
    }

    /// Apply decoded parameters to a placeholder gate from GateNameMapping.
    private static func applyParameters(
        to gate: QuantumGate,
        schemaParameters: [ParameterSchema],
        index: Int,
        diagnostics: inout [ParseDiagnostic],
    ) -> QuantumGate {
        let paramValues = schemaParameters.map { reconstructParameterValue($0) }

        switch gate {
        case .rotationX:
            guard let param = paramValues.first else { return gate }
            return .rotationX(param)
        case .rotationY:
            guard let param = paramValues.first else { return gate }
            return .rotationY(param)
        case .rotationZ:
            guard let param = paramValues.first else { return gate }
            return .rotationZ(param)
        case .u1:
            guard let param = paramValues.first else { return gate }
            return .u1(lambda: param)
        case .u2:
            guard paramValues.count >= 2 else {
                diagnostics.append(ParseDiagnostic(
                    line: 1,
                    column: 1,
                    message: "Operation \(index): u2 gate requires 2 parameters, got \(paramValues.count)",
                    severity: .error,
                ))
                return gate
            }
            return .u2(phi: paramValues[0], lambda: paramValues[1])
        case .u3:
            guard paramValues.count >= 3 else {
                diagnostics.append(ParseDiagnostic(
                    line: 1,
                    column: 1,
                    message: "Operation \(index): u3 gate requires 3 parameters, got \(paramValues.count)",
                    severity: .error,
                ))
                return gate
            }
            return .u3(theta: paramValues[0], phi: paramValues[1], lambda: paramValues[2])
        case .controlledPhase:
            guard let param = paramValues.first else { return gate }
            return .controlledPhase(param)
        case .controlledRotationX:
            guard let param = paramValues.first else { return gate }
            return .controlledRotationX(param)
        case .controlledRotationY:
            guard let param = paramValues.first else { return gate }
            return .controlledRotationY(param)
        case .controlledRotationZ:
            guard let param = paramValues.first else { return gate }
            return .controlledRotationZ(param)
        case .globalPhase:
            guard let param = paramValues.first else { return gate }
            return .globalPhase(param)
        case .givens:
            guard let param = paramValues.first else { return gate }
            return .givens(param)
        case .xx:
            guard let param = paramValues.first else { return gate }
            return .xx(param)
        case .yy:
            guard let param = paramValues.first else { return gate }
            return .yy(param)
        case .zz:
            guard let param = paramValues.first else { return gate }
            return .zz(param)
        default:
            return gate
        }
    }

    /// Reconstruct a ParameterValue from a ParameterSchema.
    private static func reconstructParameterValue(_ schema: ParameterSchema) -> ParameterValue {
        switch schema.type {
        case "value":
            .value(schema.value ?? 0)
        case "symbolic":
            .parameter(Parameter(name: schema.name ?? "unnamed"))
        case "negated":
            .negatedParameter(Parameter(name: schema.name ?? "unnamed"))
        default:
            .value(schema.value ?? 0)
        }
    }
}
