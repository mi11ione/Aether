// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Diagnostic from a parse operation carrying location, severity, and message.
///
/// Represents a single diagnostic emitted during circuit import or parsing. Each
/// diagnostic records the source location (line and column), a human-readable message
/// describing the issue, and a severity level indicating whether the diagnostic is a
/// recoverable warning or a fatal error. Parsers collect diagnostics into a
/// ``ParseResult`` rather than throwing errors, enabling partial recovery and
/// multi-error reporting in a single pass.
///
/// The ``description`` property formats diagnostics in the standard
/// `line:column: severity: message` format familiar from compiler output,
/// making diagnostics suitable for direct display in logs and tooling.
///
/// **Example:**
/// ```swift
/// let diag = ParseDiagnostic(line: 3, column: 15, message: "unexpected token", severity: .error)
/// print(diag)  // "3:15: error: unexpected token"
/// print(diag.severity == .error)  // true
/// ```
///
/// - SeeAlso: ``ParseResult``
/// - SeeAlso: ``QuantumCircuit``
@frozen
public struct ParseDiagnostic: Sendable, Equatable, CustomStringConvertible {
    /// Line number in source where the diagnostic originated (1-based)
    public let line: Int

    /// Column number in source where the diagnostic originated (1-based)
    public let column: Int

    /// Human-readable message describing the diagnostic
    public let message: String

    /// Severity level of this diagnostic
    public let severity: Severity

    /// Severity level for parse diagnostics.
    ///
    /// Determines whether a diagnostic represents a recoverable warning that does
    /// not prevent circuit construction or a fatal error that invalidates the parse
    /// result. ``ParseResult/succeeded`` checks for the absence of error-severity
    /// diagnostics to determine overall success.
    ///
    /// **Example:**
    /// ```swift
    /// let warning = ParseDiagnostic(line: 1, column: 1, message: "unused qubit", severity: .warning)
    /// let error = ParseDiagnostic(line: 2, column: 5, message: "unknown gate", severity: .error)
    /// print(warning.severity == .warning)  // true
    /// print(error.severity == .error)      // true
    /// ```
    @frozen
    public enum Severity: Sendable, Equatable {
        /// Recoverable issue that does not prevent circuit construction
        case warning

        /// Fatal issue that invalidates the parse result
        case error
    }

    /// Create a parse diagnostic with source location, message, and severity.
    ///
    /// - Parameter line: Line number in source (1-based)
    /// - Parameter column: Column number in source (1-based)
    /// - Parameter message: Human-readable diagnostic message
    /// - Parameter severity: Diagnostic severity level
    /// - Complexity: O(1)
    public init(line: Int, column: Int, message: String, severity: Severity) {
        self.line = line
        self.column = column
        self.message = message
        self.severity = severity
    }

    /// Formatted diagnostic string in `line:column: severity: message` format.
    ///
    /// **Example:**
    /// ```swift
    /// let diag = ParseDiagnostic(line: 10, column: 3, message: "missing semicolon", severity: .warning)
    /// print(diag.description)  // "10:3: warning: missing semicolon"
    /// ```
    ///
    /// - Complexity: O(n) where n is the message length
    @inlinable
    public var description: String {
        let severityLabel = switch severity {
        case .warning: "warning"
        case .error: "error"
        }
        return "\(line):\(column): \(severityLabel): \(message)"
    }
}

/// Result from a circuit import operation holding the parsed circuit and diagnostics.
///
/// Encapsulates the output of any circuit parser or importer using a no-throws pattern.
/// Instead of throwing errors on malformed input, parsers return a ``ParseResult``
/// containing the best-effort ``circuit`` and an array of ``diagnostics`` describing
/// any issues encountered. The ``succeeded`` property provides a quick check for
/// error-free parsing by verifying that no diagnostic carries error severity.
///
/// This design enables parsers to report multiple issues in a single pass, supports
/// partial recovery where a circuit is usable despite warnings, and avoids the
/// control-flow complexity of thrown errors propagating through import pipelines.
///
/// **Example:**
/// ```swift
/// let circuit = QuantumCircuit(qubits: 2)
/// let warning = ParseDiagnostic(line: 1, column: 1, message: "deprecated syntax", severity: .warning)
/// let result = ParseResult(circuit: circuit, diagnostics: [warning])
/// print(result.succeeded)  // true (no error-severity diagnostics)
/// ```
///
/// - SeeAlso: ``ParseDiagnostic``
/// - SeeAlso: ``QuantumCircuit``
@frozen
public struct ParseResult: Sendable {
    /// Parsed quantum circuit (best-effort result even when diagnostics are present)
    public let circuit: QuantumCircuit

    /// Diagnostics collected during parsing
    public let diagnostics: [ParseDiagnostic]

    /// Create a parse result with circuit and diagnostics.
    ///
    /// - Parameter circuit: Parsed quantum circuit
    /// - Parameter diagnostics: Diagnostics collected during parsing
    /// - Complexity: O(1)
    public init(circuit: QuantumCircuit, diagnostics: [ParseDiagnostic]) {
        self.circuit = circuit
        self.diagnostics = diagnostics
    }

    /// Whether the parse completed without error-severity diagnostics.
    ///
    /// Returns `true` when no diagnostic in ``diagnostics`` has ``ParseDiagnostic/Severity/error``
    /// severity. A result with only warnings is considered successful.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit(qubits: 1)
    /// let clean = ParseResult(circuit: circuit, diagnostics: [])
    /// print(clean.succeeded)  // true
    ///
    /// let err = ParseDiagnostic(line: 1, column: 1, message: "bad gate", severity: .error)
    /// let failed = ParseResult(circuit: circuit, diagnostics: [err])
    /// print(failed.succeeded)  // false
    /// ```
    ///
    /// - Complexity: O(n) where n is the number of diagnostics
    @inlinable
    public var succeeded: Bool {
        !diagnostics.contains { $0.severity == .error }
    }
}
