// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Token produced by QASM lexer representing all lexical elements for OpenQASM 2.0 and 3.0 parsing.
///
/// Each case captures a distinct lexical category: language keywords, user identifiers, numeric
/// literals (integer and floating-point), string literals, single-character symbols, newlines,
/// and end-of-file. The lexer produces a stream of ``QASMToken`` values consumed by the parser
/// to build an abstract syntax tree.
///
/// **Example:**
/// ```swift
/// let token = QASMToken.keyword("qreg")
/// assert(token.isKeyword)
/// assert(QASMToken.integer(42).isNumber)
/// ```
///
/// - SeeAlso: ``QASMVersion``
@frozen public enum QASMToken: Equatable, Sendable {
    case keyword(String)
    case identifier(String)
    case integer(Int)
    case real(Double)
    case string(String)
    case symbol(Character)
    case newline
    case eof
}

public extension QASMToken {
    /// Whether this token is a keyword.
    ///
    /// Returns `true` when the token is a `.keyword` case, regardless of which
    /// specific keyword string it contains.
    ///
    /// **Example:**
    /// ```swift
    /// let token = QASMToken.keyword("qreg")
    /// token.isKeyword  // true
    /// QASMToken.identifier("q").isKeyword  // false
    /// ```
    @inlinable
    var isKeyword: Bool {
        switch self {
        case .keyword: true
        default: false
        }
    }

    /// Whether this token is an identifier.
    ///
    /// Returns `true` when the token is an `.identifier` case.
    ///
    /// **Example:**
    /// ```swift
    /// let token = QASMToken.identifier("myQubit")
    /// token.isIdentifier  // true
    /// QASMToken.keyword("qreg").isIdentifier  // false
    /// ```
    @inlinable
    var isIdentifier: Bool {
        switch self {
        case .identifier: true
        default: false
        }
    }

    /// Whether this token is a numeric literal.
    ///
    /// Returns `true` when the token is an `.integer` or `.real` case,
    /// covering both integer and floating-point numeric literals.
    ///
    /// **Example:**
    /// ```swift
    /// QASMToken.integer(42).isNumber  // true
    /// QASMToken.real(3.14).isNumber   // true
    /// QASMToken.identifier("x").isNumber  // false
    /// ```
    @inlinable
    var isNumber: Bool {
        switch self {
        case .integer, .real: true
        default: false
        }
    }
}

public extension QASMToken {
    /// Determines whether a string is a reserved keyword for the given ``QASMVersion``.
    ///
    /// OpenQASM 2.0 defines a base set of keywords including declaration, control flow,
    /// and measurement primitives. OpenQASM 3.0 extends this with classical types, structured
    /// control flow, subroutine definitions, timing primitives, and modifier keywords.
    ///
    /// **Example:**
    /// ```swift
    /// QASMToken.isKeyword("qreg", version: .v2)   // true
    /// QASMToken.isKeyword("qubit", version: .v2)   // false
    /// QASMToken.isKeyword("qubit", version: .v3)   // true
    /// ```
    ///
    /// - Parameters:
    ///   - string: The string to check against the keyword set
    ///   - version: The OpenQASM version defining the keyword set
    /// - Returns: `true` if the string is a reserved keyword in the specified version
    @_effects(readonly)
    static func isKeyword(_ string: String, version: QASMVersion) -> Bool {
        switch version {
        case .v2:
            qasmV2Keywords.contains(string)
        case .v3:
            qasmV2Keywords.contains(string) || qasmV3AdditionalKeywords.contains(string)
        }
    }
}

extension QASMToken {
    @usableFromInline
    static let qasmV2Keywords: Set<String> = [
        "OPENQASM", "include", "qreg", "creg", "gate", "opaque",
        "barrier", "if", "measure", "reset",
    ]

    @usableFromInline
    static let qasmV3AdditionalKeywords: Set<String> = [
        "qubit", "bit", "int", "uint", "float", "angle", "bool",
        "const", "let", "for", "while", "else", "return",
        "def", "extern", "cal", "defcal", "delay", "stretch", "box",
        "input", "output", "ctrl", "inv", "pow", "negctrl",
    ]
}

extension QASMToken: CustomStringConvertible {
    /// Human-readable token representation for debugging.
    ///
    /// Returns a string that identifies both the token category and its payload,
    /// making lexer output easy to inspect during development.
    ///
    /// **Example:**
    /// ```swift
    /// let token = QASMToken.keyword("qreg")
    /// print(token)  // "keyword(qreg)"
    /// print(QASMToken.integer(42))  // "integer(42)"
    /// print(QASMToken.eof)  // "eof"
    /// ```
    public var description: String {
        switch self {
        case let .keyword(value): "keyword(\(value))"
        case let .identifier(value): "identifier(\(value))"
        case let .integer(value): "integer(\(value))"
        case let .real(value): "real(\(value))"
        case let .string(value): "string(\(value))"
        case let .symbol(value): "symbol(\(value))"
        case .newline: "newline"
        case .eof: "eof"
        }
    }
}
