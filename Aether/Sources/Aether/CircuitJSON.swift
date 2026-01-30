// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// JSON schema version and structure definition for circuit serialization
///
/// Namespace for circuit JSON serialization constants. Provides the schema version
/// used by ``CircuitJSONEncoder`` and ``CircuitJSONDecoder`` to ensure forward
/// compatibility when reading and writing quantum circuit files.
///
/// **Example:**
/// ```swift
/// let version = CircuitJSON.schemaVersion
/// assert(version == 1)
/// assert(CircuitJSON.schemaVersion > 0)
/// ```
///
/// - SeeAlso: ``CircuitJSONEncoder``
/// - SeeAlso: ``CircuitJSONDecoder``
public enum CircuitJSON: Sendable {
    /// Current schema version for circuit JSON serialization.
    ///
    /// **Example:**
    /// ```swift
    /// let version = CircuitJSON.schemaVersion
    /// print(version) // 1
    /// ```
    ///
    /// - SeeAlso: ``CircuitJSONEncoder``
    /// - SeeAlso: ``CircuitJSONDecoder``
    public static let schemaVersion = 1
}

/// Top-level JSON schema for a serialized quantum circuit.
struct CircuitJSONSchema: Codable, Sendable {
    let version: Int
    let qubitCount: Int
    let classicalBitCount: Int
    let operations: [OperationSchema]
    let metadata: MetadataSchema?
}

/// JSON schema for a single circuit operation.
struct OperationSchema: Codable, Sendable {
    static let gateType = "gate"
    static let measurementType = "measurement"
    static let barrierType = "barrier"
    static let resetType = "reset"
    static let customUnitaryName = "custom_unitary"

    let type: String
    let gate: String?
    let qubits: [Int]
    let classicalBits: [Int]?
    let parameters: [ParameterSchema]?
    let controls: [Int]?
    let matrix: [[ComplexSchema]]?
}

/// JSON schema for a gate parameter value.
struct ParameterSchema: Codable, Sendable {
    static let valueType = "value"
    static let symbolicType = "symbolic"
    static let negatedType = "negated"
    static let expressionType = "expression"

    let type: String
    let value: Double?
    let name: String?
}

/// JSON schema for a complex number.
struct ComplexSchema: Codable, Sendable {
    let real: Double
    let imaginary: Double
}

/// JSON schema for circuit metadata.
struct MetadataSchema: Codable, Sendable {
    let name: String?
    let description: String?
}
