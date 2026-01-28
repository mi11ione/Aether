// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Deterministic JSON encoder for ``QuantumCircuit`` using sorted keys for reproducible output.
///
/// Converts a ``QuantumCircuit`` into internal ``CircuitJSONSchema`` types then encodes to
/// `Data` via Foundation `JSONEncoder` with sorted keys. The deterministic output is suitable
/// for hashing and content-addressable storage.
///
/// **Example:**
/// ```swift
/// var circuit = QuantumCircuit(qubits: 2)
/// circuit.append(.hadamard, to: 0)
/// circuit.append(.cnot, to: [0, 1])
/// let data = CircuitJSONEncoder.encode(circuit)
/// ```
///
/// - SeeAlso: ``CircuitJSONDecoder``
/// - SeeAlso: ``CircuitJSON``
public enum CircuitJSONEncoder {
    /// Encode a ``QuantumCircuit`` to deterministic JSON `Data`.
    ///
    /// Maps all circuit operations to ``OperationSchema`` values, wraps them in a
    /// ``CircuitJSONSchema`` envelope with the current ``CircuitJSON/schemaVersion``,
    /// and serializes with sorted keys and pretty printing for reproducibility.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// let jsonData = CircuitJSONEncoder.encode(circuit)
    /// let jsonString = String(data: jsonData, encoding: .utf8)!
    /// ```
    ///
    /// - Parameter circuit: Quantum circuit to serialize
    /// - Returns: UTF-8 encoded JSON data with sorted keys
    /// - Complexity: O(n) where n is the number of circuit operations
    @_optimize(speed)
    public static func encode(_ circuit: QuantumCircuit) -> Data {
        let schema = buildSchema(from: circuit)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
        return try! encoder.encode(schema)
    }

    /// Build CircuitJSONSchema from QuantumCircuit.
    @_effects(readonly)
    private static func buildSchema(from circuit: QuantumCircuit) -> CircuitJSONSchema {
        var operationSchemas: [OperationSchema] = []
        operationSchemas.reserveCapacity(circuit.operations.count)

        for operation in circuit.operations {
            operationSchemas.append(mapOperation(operation))
        }

        return CircuitJSONSchema(
            version: CircuitJSON.schemaVersion,
            qubitCount: circuit.qubits,
            classicalBitCount: 0,
            operations: operationSchemas,
            metadata: nil,
        )
    }

    /// Map a single CircuitOperation to OperationSchema.
    @_effects(readonly)
    private static func mapOperation(_ operation: CircuitOperation) -> OperationSchema {
        switch operation {
        case let .gate(gate, qubits, _):
            mapGateOperation(gate: gate, qubits: qubits)
        case let .reset(qubit, _):
            OperationSchema(
                type: "reset",
                gate: nil,
                qubits: [qubit],
                classicalBits: nil,
                parameters: nil,
                controls: nil,
                matrix: nil,
            )
        case let .measure(qubit, classicalBit, _):
            OperationSchema(
                type: "measurement",
                gate: nil,
                qubits: [qubit],
                classicalBits: [classicalBit ?? qubit],
                parameters: nil,
                controls: nil,
                matrix: nil,
            )
        }
    }

    /// Map a gate operation to OperationSchema with name, parameters, controls, and matrix.
    @_effects(readonly)
    private static func mapGateOperation(gate: QuantumGate, qubits: [Int]) -> OperationSchema {
        let (baseGate, controls) = gate.flattenControlled()
        let gateName = GateNameMapping.qasmName(for: baseGate, version: .v2)
        let parameters = extractParameters(from: baseGate)
        let matrix = extractMatrix(from: baseGate)
        let controlIndices: [Int]? = controls.isEmpty ? nil : controls

        return OperationSchema(
            type: "gate",
            gate: gateName,
            qubits: qubits,
            classicalBits: nil,
            parameters: parameters,
            controls: controlIndices,
            matrix: matrix,
        )
    }

    /// Extract parameter schemas from a quantum gate.
    @_effects(readonly)
    private static func extractParameters(from gate: QuantumGate) -> [ParameterSchema]? {
        let values = parameterValues(from: gate)
        guard !values.isEmpty else { return nil }
        return values.map(mapParameterValue)
    }

    /// Collect ParameterValue instances from a gate in declaration order.
    @_effects(readonly)
    private static func parameterValues(from gate: QuantumGate) -> [ParameterValue] {
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

    /// Convert a ParameterValue to ParameterSchema.
    @_effects(readonly)
    private static func mapParameterValue(_ value: ParameterValue) -> ParameterSchema {
        switch value {
        case let .value(v):
            ParameterSchema(type: "value", value: v, name: nil)
        case let .parameter(p):
            ParameterSchema(type: "symbolic", value: nil, name: p.name)
        case let .negatedParameter(p):
            ParameterSchema(type: "negated", value: nil, name: p.name)
        case let .expression(expr):
            ParameterSchema(type: "expression", value: nil, name: expr.parameters.first?.name)
        }
    }

    /// Extract custom matrix as ComplexSchema grid when applicable.
    @_effects(readonly)
    private static func extractMatrix(from gate: QuantumGate) -> [[ComplexSchema]]? {
        switch gate {
        case let .customSingleQubit(matrix):
            convertMatrix(matrix)
        case let .customTwoQubit(matrix):
            convertMatrix(matrix)
        case let .customUnitary(matrix):
            convertMatrix(matrix)
        default:
            nil
        }
    }

    /// Convert a Complex matrix to ComplexSchema grid.
    @_effects(readonly)
    private static func convertMatrix(_ matrix: [[Complex<Double>]]) -> [[ComplexSchema]] {
        matrix.map { row in
            row.map { ComplexSchema(real: $0.real, imaginary: $0.imaginary) }
        }
    }
}
