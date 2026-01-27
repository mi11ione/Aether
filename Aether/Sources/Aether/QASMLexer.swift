// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Lexer converting QASM source text into a token stream.
///
/// Hand-written single-pass O(n) scanner that converts OpenQASM source into a flat array of
/// ``QASMToken`` values. Supports both OpenQASM 2.0 and 3.0 via the ``QASMVersion`` parameter,
/// which controls keyword classification. The scanner tracks line and column positions for
/// diagnostic reporting and handles single-line comments (`//`), block comments (`/* */`),
/// identifiers, integer and real literals (including scientific notation), string literals,
/// and single-character symbols.
///
/// **Example:**
/// ```swift
/// let source = "OPENQASM 2.0;"
/// let tokens = QASMLexer.tokenize(source, version: .v2)
/// assert(tokens.last == .eof)
/// ```
///
/// - Complexity: O(n) where n is the length of the source string
///
/// - SeeAlso: ``QASMToken``
/// - SeeAlso: ``QASMVersion``
public enum QASMLexer: Sendable {
    /// Tokenize QASM source text into an array of ``QASMToken`` values.
    ///
    /// Scans the input string left-to-right, emitting one token per lexical element.
    /// Whitespace (spaces and tabs) is consumed silently. Newlines produce ``QASMToken/newline``
    /// tokens. The returned array always ends with a single ``QASMToken/eof`` sentinel.
    ///
    /// **Example:**
    /// ```swift
    /// let source = "qreg q[2];"
    /// let tokens = QASMLexer.tokenize(source, version: .v2)
    /// // [.keyword("qreg"), .identifier("q"), .symbol("["), .integer(2), .symbol("]"), .symbol(";"), .eof]
    /// ```
    ///
    /// - Parameters:
    ///   - source: OpenQASM source text to tokenize
    ///   - version: QASM version controlling keyword classification
    /// - Returns: Array of tokens ending with ``QASMToken/eof``
    ///
    /// - Complexity: O(n) where n is the number of Unicode scalars in `source`
    @inlinable
    @_optimize(speed)
    public static func tokenize(_ source: String, version: QASMVersion) -> [QASMToken] {
        let scalars = source.unicodeScalars
        var tokens = [QASMToken]()
        tokens.reserveCapacity(scalars.count / 2)

        var index = scalars.startIndex
        var line = 1
        var column = 1

        while index < scalars.endIndex {
            let scalar = scalars[index]

            if scalar == " " || scalar == "\t" {
                index = scalars.index(after: index)
                column += 1
                continue
            }

            if scalar == "\n" {
                tokens.append(.newline)
                index = scalars.index(after: index)
                line += 1
                column = 1
                continue
            }

            if scalar == "\r" {
                tokens.append(.newline)
                index = scalars.index(after: index)
                line += 1
                column = 1
                if index < scalars.endIndex, scalars[index] == "\n" {
                    index = scalars.index(after: index)
                }
                continue
            }

            if scalar == "/" {
                let next = scalars.index(after: index)
                if next < scalars.endIndex {
                    if scalars[next] == "/" {
                        let result = skipLineComment(scalars: scalars, from: next)
                        index = result.index
                        column = result.column
                        continue
                    }
                    if scalars[next] == "*" {
                        let result = skipBlockComment(
                            scalars: scalars,
                            from: scalars.index(after: next),
                            line: line,
                            column: column + 2,
                        )
                        index = result.index
                        line = result.line
                        column = result.column
                        continue
                    }
                }
                tokens.append(.symbol("/"))
                index = scalars.index(after: index)
                column += 1
                continue
            }

            if isIdentifierStart(scalar) {
                let start = index
                var length = 0
                while index < scalars.endIndex, isIdentifierContinue(scalars[index]) {
                    index = scalars.index(after: index)
                    length += 1
                }
                let word = String(scalars[start ..< index])
                if QASMToken.isKeyword(word, version: version) {
                    tokens.append(.keyword(word))
                } else {
                    tokens.append(.identifier(word))
                }
                column += length
                continue
            }

            if isDigit(scalar) || (scalar == "." && index < scalars.endIndex && peekIsDigit(scalars: scalars, after: index)) {
                let result = scanNumber(scalars: scalars, from: index)
                tokens.append(result.token)
                index = result.index
                column += result.length
                continue
            }

            if scalar == "\"" {
                let result = scanString(scalars: scalars, from: scalars.index(after: index))
                tokens.append(result.token)
                index = result.index
                column += result.length
                continue
            }

            tokens.append(.symbol(Character(scalar)))
            index = scalars.index(after: index)
            column += 1
        }

        tokens.append(.eof)
        return tokens
    }

    /// Skip single-line comment from current position to end of line.
    @inlinable
    @inline(__always)
    static func skipLineComment(
        scalars: String.UnicodeScalarView,
        from start: String.UnicodeScalarView.Index,
    ) -> (index: String.UnicodeScalarView.Index, column: Int) {
        var index = start
        while index < scalars.endIndex, scalars[index] != "\n", scalars[index] != "\r" {
            index = scalars.index(after: index)
        }
        return (index, 1)
    }

    /// Skip block comment from after /* to past */.
    @inlinable
    static func skipBlockComment(
        scalars: String.UnicodeScalarView,
        from start: String.UnicodeScalarView.Index,
        line: Int,
        column: Int,
    ) -> (index: String.UnicodeScalarView.Index, line: Int, column: Int) {
        var index = start
        var currentLine = line
        var currentColumn = column
        while index < scalars.endIndex {
            let scalar = scalars[index]
            if scalar == "*" {
                let next = scalars.index(after: index)
                if next < scalars.endIndex, scalars[next] == "/" {
                    return (scalars.index(after: next), currentLine, currentColumn + 2)
                }
            }
            if scalar == "\n" {
                currentLine += 1
                currentColumn = 1
            } else if scalar == "\r" {
                currentLine += 1
                currentColumn = 1
                let next = scalars.index(after: index)
                if next < scalars.endIndex, scalars[next] == "\n" {
                    index = next
                }
            } else {
                currentColumn += 1
            }
            index = scalars.index(after: index)
        }
        return (index, currentLine, currentColumn)
    }

    /// Scan integer or real literal including optional scientific notation.
    @inlinable
    static func scanNumber(
        scalars: String.UnicodeScalarView,
        from start: String.UnicodeScalarView.Index,
    ) -> (token: QASMToken, index: String.UnicodeScalarView.Index, length: Int) {
        var index = start
        var length = 0
        var hasDecimalPoint = false

        if index < scalars.endIndex, scalars[index] == "." {
            hasDecimalPoint = true
            index = scalars.index(after: index)
            length += 1
        }

        while index < scalars.endIndex, isDigit(scalars[index]) {
            index = scalars.index(after: index)
            length += 1
        }

        if !hasDecimalPoint, index < scalars.endIndex, scalars[index] == "." {
            hasDecimalPoint = true
            index = scalars.index(after: index)
            length += 1
            while index < scalars.endIndex, isDigit(scalars[index]) {
                index = scalars.index(after: index)
                length += 1
            }
        }

        if index < scalars.endIndex, scalars[index] == "e" || scalars[index] == "E" {
            hasDecimalPoint = true
            index = scalars.index(after: index)
            length += 1
            if index < scalars.endIndex, scalars[index] == "+" || scalars[index] == "-" {
                index = scalars.index(after: index)
                length += 1
            }
            while index < scalars.endIndex, isDigit(scalars[index]) {
                index = scalars.index(after: index)
                length += 1
            }
        }

        let text = String(scalars[start ..< index])
        if hasDecimalPoint {
            let value = Double(text) ?? 0.0
            return (.real(value), index, length)
        }
        let value = Int(text) ?? 0
        return (.integer(value), index, length)
    }

    /// Scan string literal from after opening quote to past closing quote.
    @inlinable
    static func scanString(
        scalars: String.UnicodeScalarView,
        from start: String.UnicodeScalarView.Index,
    ) -> (token: QASMToken, index: String.UnicodeScalarView.Index, length: Int) {
        var index = start
        var length = 1

        while index < scalars.endIndex, scalars[index] != "\"" {
            index = scalars.index(after: index)
            length += 1
        }

        let content = String(scalars[start ..< index])

        if index < scalars.endIndex {
            index = scalars.index(after: index)
            length += 1
        }

        return (.string(content), index, length)
    }

    /// Whether scalar is valid identifier start character.
    @inlinable
    @inline(__always)
    static func isIdentifierStart(_ scalar: Unicode.Scalar) -> Bool {
        (scalar >= "a" && scalar <= "z")
            || (scalar >= "A" && scalar <= "Z")
            || scalar == "_"
    }

    /// Whether scalar is valid identifier continuation character.
    @inlinable
    @inline(__always)
    static func isIdentifierContinue(_ scalar: Unicode.Scalar) -> Bool {
        isIdentifierStart(scalar) || isDigit(scalar)
    }

    /// Whether scalar is an ASCII digit.
    @inlinable
    @inline(__always)
    static func isDigit(_ scalar: Unicode.Scalar) -> Bool {
        scalar >= "0" && scalar <= "9"
    }

    /// Whether the scalar after the given index is a digit.
    @inlinable
    @inline(__always)
    static func peekIsDigit(
        scalars: String.UnicodeScalarView,
        after index: String.UnicodeScalarView.Index,
    ) -> Bool {
        let next = scalars.index(after: index)
        guard next < scalars.endIndex else { return false }
        return isDigit(scalars[next])
    }
}
