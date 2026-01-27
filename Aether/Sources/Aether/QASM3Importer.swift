// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// OpenQASM 3.0 circuit parser with full error recovery.
///
/// Recursive descent parser that converts OpenQASM 3.0 source text into a ``QuantumCircuit``.
/// Extends QASM 2.0 parsing with modern syntax including `qubit`/`bit` declarations, classical
/// types (`int`, `float`, `angle`, `bool`), subroutine definitions (`def`), gate modifiers
/// (`ctrl @`, `inv @`, `pow @`), and timing directives (`delay`). All parse errors are
/// collected into ``ParseDiagnostic`` values within a ``ParseResult`` rather than thrown,
/// enabling partial recovery and multi-error reporting in a single pass.
///
/// Gate names are resolved through ``GateNameMapping/gate(forQASMName:version:)`` with
/// ``QASMVersion/v3``, mapping QASM 3.0 identifiers such as `p` (phase), `cx`, `rz` to
/// their corresponding ``QuantumGate`` cases.
///
/// **Example:**
/// ```swift
/// let source = """
///     OPENQASM 3.0;
///     include "stdgates.inc";
///     qubit[2] q;
///     h q[0];
///     cx q[0], q[1];
///     """
/// let result = QASM3Importer.parse(source)
/// let circuit = result.circuit
/// ```
///
/// - SeeAlso: ``ParseResult``
/// - SeeAlso: ``ParseDiagnostic``
/// - SeeAlso: ``QuantumCircuit``
/// - SeeAlso: ``GateNameMapping``
public enum QASM3Importer: Sendable {
    /// Parse OpenQASM 3.0 source text into a quantum circuit with diagnostics.
    ///
    /// Tokenizes the source via ``QASMLexer`` and performs recursive descent parsing of the
    /// QASM 3.0 grammar. Supports qubit/bit declarations, gate calls with modifiers (`ctrl @`,
    /// `inv @`, `pow @`), gate definitions, subroutine definitions, measurement, reset, barrier,
    /// classical type declarations, timing directives, and extern declarations. Unsupported
    /// features (classical control flow, extern bodies, timing) emit warnings rather than errors.
    ///
    /// On parse error the parser records a ``ParseDiagnostic``, skips to the next synchronization
    /// point (semicolon or closing brace), and continues. The returned ``ParseResult`` contains
    /// the best-effort circuit and all collected diagnostics.
    ///
    /// **Example:**
    /// ```swift
    /// let qasm = "OPENQASM 3.0; include \"stdgates.inc\"; qubit[2] q; h q[0]; cx q[0], q[1];"
    /// let result = QASM3Importer.parse(qasm)
    /// let bell = result.circuit
    /// ```
    ///
    /// - Parameter source: OpenQASM 3.0 program text
    /// - Returns: ``ParseResult`` containing the parsed circuit and any diagnostics
    /// - Complexity: O(n) where n is the number of tokens in the source
    @_optimize(speed)
    public static func parse(_ source: String) -> ParseResult {
        let tokens = QASMLexer.tokenize(source, version: .v3)
        var state = ParserState(tokens: tokens)
        parseProgram(&state)
        let qubitCount = max(state.totalQubits, 1)
        let circuit = QuantumCircuit(qubits: qubitCount, operations: state.operations)
        return ParseResult(circuit: circuit, diagnostics: state.diagnostics)
    }
}

/// Gate definition with parameter names and body operations.
private struct GateDefinition {
    let parameterNames: [String]
    let qubitNames: [String]
    let bodyTokens: [QASMToken]
}

/// Subroutine definition with parameter names, qubit bindings, and body tokens.
private struct SubroutineDefinition {
    let parameterNames: [String]
    let qubitNames: [String]
    let qubitSizes: [Int]
    let bodyTokens: [QASMToken]
}

/// Gate modifier type for controlled, inverse, and power operations.
private enum GateModifier {
    case ctrl(Int)
    case negctrl(Int)
    case inv
    case pow(Double)
}

/// Mutable parser state tracking position, declarations, and collected operations.
private struct ParserState {
    var tokens: [QASMToken]
    var position: Int = 0
    var diagnostics: [ParseDiagnostic] = []
    var qubitDeclarations: [String: (offset: Int, size: Int)] = [:]
    var bitDeclarations: [String: (offset: Int, size: Int)] = [:]
    var totalQubits: Int = 0
    var totalBits: Int = 0
    var operations: [CircuitOperation] = []
    var gateDefinitions: [String: GateDefinition] = [:]
    var subroutines: [String: SubroutineDefinition] = [:]
    var linePositions: [Int]

    init(tokens: [QASMToken]) {
        self.tokens = tokens
        linePositions = Self.computeLinePositions(tokens)
    }

    /// Compute a per-token array of source line numbers.
    private static func computeLinePositions(_ tokens: [QASMToken]) -> [Int] {
        var positions = [Int]()
        positions.reserveCapacity(tokens.count)
        var line = 1
        for token in tokens {
            positions.append(line)
            if token == .newline { line += 1 }
        }
        return positions
    }
}

/// Return the token at the current parser position.
private func currentToken(_ state: inout ParserState) -> QASMToken {
    guard state.position < state.tokens.count else { return .eof }
    return state.tokens[state.position]
}

/// Return the source line number at the current parser position.
private func currentLine(_ state: inout ParserState) -> Int {
    guard state.position < state.linePositions.count else { return 1 }
    return state.linePositions[state.position]
}

/// Consume the current token and advance the parser position.
@discardableResult
private func advance(_ state: inout ParserState) -> QASMToken {
    let token = currentToken(&state)
    if state.position < state.tokens.count { state.position += 1 }
    return token
}

/// Skip over any consecutive newline tokens.
private func skipNewlines(_ state: inout ParserState) {
    while currentToken(&state) == .newline {
        advance(&state)
    }
}

/// Expect and consume a specific symbol token, recording an error on mismatch.
@discardableResult
private func expect(_ state: inout ParserState, symbol: Character) -> Bool {
    skipNewlines(&state)
    if case .symbol(symbol) = currentToken(&state) {
        advance(&state)
        return true
    }
    addError(&state, "expected '\(symbol)'")
    return false
}

/// Record a parse error diagnostic at the current position.
private func addError(_ state: inout ParserState, _ message: String) {
    let line = currentLine(&state)
    state.diagnostics.append(ParseDiagnostic(line: line, column: 1, message: message, severity: .error))
}

/// Record a parse warning diagnostic at the current position.
private func addWarning(_ state: inout ParserState, _ message: String) {
    let line = currentLine(&state)
    state.diagnostics.append(ParseDiagnostic(line: line, column: 1, message: message, severity: .warning))
}

/// Skip tokens until a synchronization point (semicolon, closing brace, or EOF).
private func synchronize(_ state: inout ParserState) {
    while true {
        let token = currentToken(&state)
        switch token {
        case .eof:
            return
        case .symbol(";"):
            advance(&state)
            return
        case .symbol("}"):
            advance(&state)
            return
        default:
            advance(&state)
        }
    }
}

/// Parse the top-level QASM 3.0 program from the token stream.
private func parseProgram(_ state: inout ParserState) {
    skipNewlines(&state)
    parseVersionHeader(&state)
    while currentToken(&state) != .eof {
        skipNewlines(&state)
        if currentToken(&state) == .eof { break }
        parseStatement(&state)
    }
}

/// Parse the OPENQASM version header and validate the version number.
private func parseVersionHeader(_ state: inout ParserState) {
    skipNewlines(&state)
    guard case .keyword("OPENQASM") = currentToken(&state) else { return }
    advance(&state)
    skipNewlines(&state)

    let versionToken = currentToken(&state)
    switch versionToken {
    case let .real(v):
        if v < 3.0 {
            addWarning(&state, "expected QASM version 3.0, got \(v)")
        }
        advance(&state)
    case let .integer(v):
        if v < 3 {
            addWarning(&state, "expected QASM version 3.0, got \(v)")
        }
        advance(&state)
    default:
        addError(&state, "expected version number after OPENQASM")
    }
    expect(&state, symbol: ";")
}

/// Dispatch a single statement based on the current keyword or identifier token.
private func parseStatement(_ state: inout ParserState) {
    skipNewlines(&state)
    let token = currentToken(&state)

    switch token {
    case .keyword("include"):
        parseInclude(&state)
    case .keyword("qubit"):
        parseQubitDeclaration(&state)
    case .keyword("bit"):
        parseBitDeclaration(&state)
    case .keyword("qreg"):
        parseQregDeclaration(&state)
    case .keyword("creg"):
        parseCregDeclaration(&state)
    case .keyword("gate"):
        parseGateDefinition(&state)
    case .keyword("def"):
        parseSubroutineDefinition(&state)
    case .keyword("measure"):
        parseMeasureStatement(&state)
    case .keyword("reset"):
        parseResetStatement(&state)
    case .keyword("barrier"):
        parseBarrierStatement(&state)
    case .keyword("if"):
        parseIfStatement(&state)
    case .keyword("for"):
        parseForStatement(&state)
    case .keyword("while"):
        parseWhileStatement(&state)
    case .keyword("extern"):
        parseExternDeclaration(&state)
    case .keyword("delay"):
        parseDelayStatement(&state)
    case .keyword("int"), .keyword("uint"), .keyword("float"),
         .keyword("angle"), .keyword("bool"), .keyword("const"):
        parseClassicalDeclaration(&state)
    case .keyword("input"), .keyword("output"):
        parseIODeclaration(&state)
    case .keyword("let"):
        parseLetDeclaration(&state)
    case .keyword("return"):
        parseReturnStatement(&state)
    case .keyword("cal"), .keyword("defcal"):
        parseCalibrationBlock(&state)
    case .keyword("box"):
        parseBoxStatement(&state)
    case .keyword("stretch"):
        parseStretchDeclaration(&state)
    case .keyword("opaque"):
        parseOpaqueDeclaration(&state)
    case .keyword("ctrl"), .keyword("inv"), .keyword("pow"), .keyword("negctrl"):
        parseModifiedGateCall(&state)
    case .identifier:
        parseIdentifierStatement(&state)
    case .symbol(";"):
        advance(&state)
    default:
        addError(&state, "unexpected token: \(token)")
        synchronize(&state)
    }
}

/// Parse an include directive and consume the file path string.
private func parseInclude(_ state: inout ParserState) {
    advance(&state)
    skipNewlines(&state)
    if case .string = currentToken(&state) {
        advance(&state)
    } else {
        addError(&state, "expected file path string after include")
    }
    expect(&state, symbol: ";")
}

/// Parse a qubit declaration statement with optional size.
private func parseQubitDeclaration(_ state: inout ParserState) {
    advance(&state)
    skipNewlines(&state)

    var size = 1
    if case .symbol("[") = currentToken(&state) {
        advance(&state)
        size = parseIntegerLiteral(&state)
        expect(&state, symbol: "]")
    }

    skipNewlines(&state)
    guard case let .identifier(name) = currentToken(&state) else {
        addError(&state, "expected qubit register name")
        synchronize(&state)
        return
    }
    advance(&state)

    let offset = state.totalQubits
    state.qubitDeclarations[name] = (offset: offset, size: size)
    state.totalQubits += size
    expect(&state, symbol: ";")
}

/// Parse a bit declaration statement with optional size.
private func parseBitDeclaration(_ state: inout ParserState) {
    advance(&state)
    skipNewlines(&state)

    var size = 1
    if case .symbol("[") = currentToken(&state) {
        advance(&state)
        size = parseIntegerLiteral(&state)
        expect(&state, symbol: "]")
    }

    skipNewlines(&state)
    guard case let .identifier(name) = currentToken(&state) else {
        addError(&state, "expected bit register name")
        synchronize(&state)
        return
    }
    advance(&state)

    let offset = state.totalBits
    state.bitDeclarations[name] = (offset: offset, size: size)
    state.totalBits += size
    expect(&state, symbol: ";")
}

/// Parse a legacy qreg declaration statement.
private func parseQregDeclaration(_ state: inout ParserState) {
    advance(&state)
    skipNewlines(&state)

    guard case let .identifier(name) = currentToken(&state) else {
        addError(&state, "expected register name after qreg")
        synchronize(&state)
        return
    }
    advance(&state)

    var size = 1
    if case .symbol("[") = currentToken(&state) {
        advance(&state)
        size = parseIntegerLiteral(&state)
        expect(&state, symbol: "]")
    }

    let offset = state.totalQubits
    state.qubitDeclarations[name] = (offset: offset, size: size)
    state.totalQubits += size
    expect(&state, symbol: ";")
}

/// Parse a legacy creg declaration statement.
private func parseCregDeclaration(_ state: inout ParserState) {
    advance(&state)
    skipNewlines(&state)

    guard case let .identifier(name) = currentToken(&state) else {
        addError(&state, "expected register name after creg")
        synchronize(&state)
        return
    }
    advance(&state)

    var size = 1
    if case .symbol("[") = currentToken(&state) {
        advance(&state)
        size = parseIntegerLiteral(&state)
        expect(&state, symbol: "]")
    }

    let offset = state.totalBits
    state.bitDeclarations[name] = (offset: offset, size: size)
    state.totalBits += size
    expect(&state, symbol: ";")
}

/// Parse a gate definition with parameters, qubit names, and braced body.
private func parseGateDefinition(_ state: inout ParserState) {
    advance(&state)
    skipNewlines(&state)

    guard case let .identifier(name) = currentToken(&state) else {
        addError(&state, "expected gate name")
        synchronize(&state)
        return
    }
    advance(&state)
    skipNewlines(&state)

    var paramNames: [String] = []
    if case .symbol("(") = currentToken(&state) {
        advance(&state)
        paramNames = parseIdentifierList(&state)
        expect(&state, symbol: ")")
    }

    skipNewlines(&state)
    let qubitNames = parseIdentifierList(&state)
    skipNewlines(&state)

    let bodyTokens = parseBracedBody(&state)
    state.gateDefinitions[name] = GateDefinition(
        parameterNames: paramNames,
        qubitNames: qubitNames,
        bodyTokens: bodyTokens,
    )
}

/// Parse a subroutine definition with parameter declarations and qubit arguments.
private func parseSubroutineDefinition(_ state: inout ParserState) {
    advance(&state)
    skipNewlines(&state)

    guard case let .identifier(name) = currentToken(&state) else {
        addError(&state, "expected subroutine name")
        synchronize(&state)
        return
    }
    advance(&state)
    skipNewlines(&state)

    var paramNames: [String] = []
    if case .symbol("(") = currentToken(&state) {
        advance(&state)
        paramNames = parseParameterDeclList(&state)
        expect(&state, symbol: ")")
    }

    skipNewlines(&state)
    var qubitNames: [String] = []
    var qubitSizes: [Int] = []
    while case .keyword("qubit") = currentToken(&state) {
        advance(&state)
        var size = 1
        if case .symbol("[") = currentToken(&state) {
            advance(&state)
            size = parseIntegerLiteral(&state)
            expect(&state, symbol: "]")
        }
        skipNewlines(&state)
        if case let .identifier(qName) = currentToken(&state) {
            advance(&state)
            qubitNames.append(qName)
            qubitSizes.append(size)
        }
        skipNewlines(&state)
        if case .symbol(",") = currentToken(&state) {
            advance(&state)
            skipNewlines(&state)
        }
    }

    skipNewlines(&state)

    skipArrowIfPresent(&state)

    let bodyTokens = parseBracedBody(&state)
    state.subroutines[name] = SubroutineDefinition(
        parameterNames: paramNames,
        qubitNames: qubitNames,
        qubitSizes: qubitSizes,
        bodyTokens: bodyTokens,
    )
}

/// Skip a return-type arrow token sequence if present.
private func skipArrowIfPresent(_ state: inout ParserState) {
    skipNewlines(&state)
    if case .symbol("-") = currentToken(&state) {
        let saved = state.position
        advance(&state)
        if case .symbol(">") = currentToken(&state) {
            advance(&state)
            skipToOpenBrace(&state)
        } else {
            state.position = saved
        }
    }
}

/// Advance the parser until an open brace token is found.
private func skipToOpenBrace(_ state: inout ParserState) {
    while currentToken(&state) != .eof {
        if case .symbol("{") = currentToken(&state) { return }
        advance(&state)
    }
}

/// Parse a measure statement and consume its qubit arguments.
private func parseMeasureStatement(_ state: inout ParserState) {
    advance(&state)
    skipNewlines(&state)
    _ = parseQubitArgList(&state)
    expect(&state, symbol: ";")
}

/// Parse a reset statement and emit reset operations for each qubit.
private func parseResetStatement(_ state: inout ParserState) {
    advance(&state)
    skipNewlines(&state)

    let qubits = parseQubitArgList(&state)
    for qubit in qubits {
        state.operations.append(.reset(qubit: qubit))
    }
    expect(&state, symbol: ";")
}

/// Parse a barrier statement and skip its qubit arguments.
private func parseBarrierStatement(_ state: inout ParserState) {
    advance(&state)
    skipNewlines(&state)
    while currentToken(&state) != .eof {
        if case .symbol(";") = currentToken(&state) { break }
        if currentToken(&state) == .newline { break }
        advance(&state)
    }
    if case .symbol(";") = currentToken(&state) {
        advance(&state)
    }
}

/// Parse a classical if statement with optional else branch.
private func parseIfStatement(_ state: inout ParserState) {
    addWarning(&state, "classical if statement parsed but not fully modeled")
    advance(&state)
    skipNewlines(&state)
    if case .symbol("(") = currentToken(&state) {
        skipBalancedParens(&state)
    }
    skipNewlines(&state)
    if case .symbol("{") = currentToken(&state) {
        _ = parseBracedBody(&state)
    } else {
        parseStatement(&state)
    }
    skipNewlines(&state)
    if case .keyword("else") = currentToken(&state) {
        advance(&state)
        skipNewlines(&state)
        if case .symbol("{") = currentToken(&state) {
            _ = parseBracedBody(&state)
        } else {
            parseStatement(&state)
        }
    }
}

/// Parse a classical for loop statement.
private func parseForStatement(_ state: inout ParserState) {
    addWarning(&state, "classical for loop parsed but not fully modeled")
    advance(&state)
    skipNewlines(&state)
    while currentToken(&state) != .eof {
        if case .symbol("{") = currentToken(&state) { break }
        advance(&state)
    }
    if case .symbol("{") = currentToken(&state) {
        _ = parseBracedBody(&state)
    }
}

/// Parse a classical while loop statement.
private func parseWhileStatement(_ state: inout ParserState) {
    addWarning(&state, "classical while loop parsed but not fully modeled")
    advance(&state)
    skipNewlines(&state)
    if case .symbol("(") = currentToken(&state) {
        skipBalancedParens(&state)
    }
    skipNewlines(&state)
    if case .symbol("{") = currentToken(&state) {
        _ = parseBracedBody(&state)
    } else {
        parseStatement(&state)
    }
}

/// Parse an extern function declaration and skip its body.
private func parseExternDeclaration(_ state: inout ParserState) {
    addWarning(&state, "extern declaration parsed but not supported at runtime")
    advance(&state)
    synchronize(&state)
}

/// Parse a delay timing directive and skip its arguments.
private func parseDelayStatement(_ state: inout ParserState) {
    addWarning(&state, "delay directive parsed but not modeled in circuit")
    advance(&state)
    synchronize(&state)
}

/// Parse a classical type declaration with optional assignment.
private func parseClassicalDeclaration(_ state: inout ParserState) {
    advance(&state)
    skipNewlines(&state)
    if case .symbol("[") = currentToken(&state) {
        advance(&state)
        _ = parseIntegerLiteral(&state)
        expect(&state, symbol: "]")
    }
    skipNewlines(&state)
    if case .identifier = currentToken(&state) {
        advance(&state)
    }
    skipNewlines(&state)
    if case .symbol("=") = currentToken(&state) {
        advance(&state)
        skipToSemicolon(&state)
    }
    if case .symbol(";") = currentToken(&state) {
        advance(&state)
    }
}

/// Parse an input or output declaration and skip to the next statement.
private func parseIODeclaration(_ state: inout ParserState) {
    advance(&state)
    synchronize(&state)
}

/// Parse a let alias declaration and skip to the next statement.
private func parseLetDeclaration(_ state: inout ParserState) {
    advance(&state)
    synchronize(&state)
}

/// Parse a return statement and skip to the next statement.
private func parseReturnStatement(_ state: inout ParserState) {
    advance(&state)
    synchronize(&state)
}

/// Parse a calibration block and skip its braced body.
private func parseCalibrationBlock(_ state: inout ParserState) {
    addWarning(&state, "calibration block parsed but not supported")
    advance(&state)
    skipNewlines(&state)
    while currentToken(&state) != .eof {
        if case .symbol("{") = currentToken(&state) { break }
        if case .symbol(";") = currentToken(&state) {
            advance(&state)
            return
        }
        advance(&state)
    }
    if case .symbol("{") = currentToken(&state) {
        _ = parseBracedBody(&state)
    }
}

/// Parse a box statement with optional duration and braced body.
private func parseBoxStatement(_ state: inout ParserState) {
    addWarning(&state, "box statement parsed but not fully modeled")
    advance(&state)
    skipNewlines(&state)
    if case .symbol("[") = currentToken(&state) {
        advance(&state)
        skipToSymbol(&state, "]")
        if case .symbol("]") = currentToken(&state) {
            advance(&state)
        }
    }
    skipNewlines(&state)
    if case .symbol("{") = currentToken(&state) {
        _ = parseBracedBody(&state)
    }
}

/// Parse a stretch declaration and skip to the next statement.
private func parseStretchDeclaration(_ state: inout ParserState) {
    addWarning(&state, "stretch declaration parsed but not supported")
    advance(&state)
    synchronize(&state)
}

/// Parse an opaque gate declaration and skip to the next statement.
private func parseOpaqueDeclaration(_ state: inout ParserState) {
    advance(&state)
    synchronize(&state)
}

/// Parse an identifier-led statement as either a gate call or an assignment.
private func parseIdentifierStatement(_ state: inout ParserState) {
    guard case .identifier = currentToken(&state) else {
        addError(&state, "expected identifier")
        synchronize(&state)
        return
    }

    let savedPos = state.position
    advance(&state)
    skipNewlines(&state)

    if case .symbol("=") = currentToken(&state) {
        advance(&state)
        skipNewlines(&state)
        if case .keyword("measure") = currentToken(&state) {
            advance(&state)
            skipNewlines(&state)
            _ = parseQubitArgList(&state)
            expect(&state, symbol: ";")
            return
        }
        skipToSemicolon(&state)
        if case .symbol(";") = currentToken(&state) { advance(&state) }
        return
    }

    if case .symbol("[") = currentToken(&state) {
        advance(&state)
        skipNewlines(&state)
        _ = parseExpression(&state)
        expect(&state, symbol: "]")
        skipNewlines(&state)
        if case .symbol("=") = currentToken(&state) {
            advance(&state)
            skipNewlines(&state)
            if case .keyword("measure") = currentToken(&state) {
                advance(&state)
                skipNewlines(&state)
                _ = parseQubitArgList(&state)
                expect(&state, symbol: ";")
                return
            }
            skipToSemicolon(&state)
            if case .symbol(";") = currentToken(&state) { advance(&state) }
            return
        }
        state.position = savedPos
    } else {
        state.position = savedPos
    }

    parseGateCall(&state, modifiers: [])
}

/// Parse a gate call prefixed by one or more gate modifiers.
private func parseModifiedGateCall(_ state: inout ParserState) {
    let modifiers = parseModifiers(&state)
    parseGateCall(&state, modifiers: modifiers)
}

/// Parse a sequence of gate modifier keywords (ctrl, negctrl, inv, pow).
private func parseModifiers(_ state: inout ParserState) -> [GateModifier] {
    var modifiers: [GateModifier] = []
    while true {
        skipNewlines(&state)
        let token = currentToken(&state)
        switch token {
        case .keyword("ctrl"):
            advance(&state)
            skipNewlines(&state)
            var count = 1
            if case .symbol("(") = currentToken(&state) {
                advance(&state)
                count = parseIntegerLiteral(&state)
                expect(&state, symbol: ")")
            }
            skipNewlines(&state)
            expect(&state, symbol: "@")
            modifiers.append(.ctrl(count))
        case .keyword("negctrl"):
            advance(&state)
            skipNewlines(&state)
            var count = 1
            if case .symbol("(") = currentToken(&state) {
                advance(&state)
                count = parseIntegerLiteral(&state)
                expect(&state, symbol: ")")
            }
            skipNewlines(&state)
            expect(&state, symbol: "@")
            modifiers.append(.negctrl(count))
        case .keyword("inv"):
            advance(&state)
            skipNewlines(&state)
            expect(&state, symbol: "@")
            modifiers.append(.inv)
        case .keyword("pow"):
            advance(&state)
            skipNewlines(&state)
            expect(&state, symbol: "(")
            let exponent = parseExpression(&state)
            expect(&state, symbol: ")")
            skipNewlines(&state)
            expect(&state, symbol: "@")
            modifiers.append(.pow(exponent))
        default:
            return modifiers
        }
    }
}

/// Parse a gate call with optional parameters, qubit arguments, and applied modifiers.
private func parseGateCall(_ state: inout ParserState, modifiers: [GateModifier]) {
    skipNewlines(&state)
    guard case let .identifier(gateName) = currentToken(&state) else {
        addError(&state, "expected gate name")
        synchronize(&state)
        return
    }
    advance(&state)
    skipNewlines(&state)

    var params: [Double] = []
    if case .symbol("(") = currentToken(&state) {
        advance(&state)
        params = parseExpressionList(&state)
        expect(&state, symbol: ")")
    }

    skipNewlines(&state)
    let qubitArgs = parseQubitArgList(&state)
    expect(&state, symbol: ";")

    if let gateDef = state.gateDefinitions[gateName] {
        expandGateDefinition(&state, definition: gateDef, params: params, qubits: qubitArgs)
        return
    }

    let resolvedGate = resolveGate(gateName, params: params, state: &state)

    guard let baseGateResolved = resolvedGate else {
        addError(&state, "unknown gate '\(gateName)'")
        return
    }

    var gate = baseGateResolved

    for modifier in modifiers.reversed() {
        switch modifier {
        case .inv:
            gate = gate.inverse
        case let .ctrl(count):
            if count == 1, qubitArgs.count >= 2 {
                let controlQubits = [qubitArgs[0]]
                gate = .controlled(gate: gate, controls: controlQubits)
            } else {
                let controlQubits = Array(qubitArgs.prefix(count))
                gate = .controlled(gate: gate, controls: controlQubits)
            }
        case let .negctrl(count):
            addWarning(&state, "negctrl modifier approximated as ctrl")
            let controlQubits = Array(qubitArgs.prefix(count))
            gate = .controlled(gate: gate, controls: controlQubits)
        case let .pow(exponent):
            let intExp = Int(exponent)
            if exponent == Double(intExp), intExp > 0 {
                let baseGate = gate
                for _ in 1 ..< intExp {
                    state.operations.append(.gate(baseGate, qubits: qubitArgs))
                }
            } else {
                addWarning(&state, "non-integer pow exponent approximated")
            }
        }
    }

    let hasCtrl = modifiers.contains { if case .ctrl = $0 { return true }; if case .negctrl = $0 { return true }; return false }
    if hasCtrl {
        let (baseGate, controls) = gate.flattenControlled()
        let targetQubits = qubitArgs.filter { !controls.contains($0) }
        let allQubits = controls + targetQubits
        state.operations.append(.gate(.controlled(gate: baseGate, controls: controls), qubits: allQubits))
    } else {
        state.operations.append(.gate(gate, qubits: qubitArgs))
    }
}

/// Resolve a gate name to a parameterized QuantumGate using the gate name mapping.
private func resolveGate(_ name: String, params: [Double], state _: inout ParserState) -> QuantumGate? {
    if let templateGate = GateNameMapping.gate(forQASMName: name, version: .v3) {
        return applyParameters(to: templateGate, params: params)
    }
    if let templateGate = GateNameMapping.gate(forQASMName: name, version: .v2) {
        return applyParameters(to: templateGate, params: params)
    }
    return nil
}

/// Apply numeric parameter values to a template gate.
private func applyParameters(to gate: QuantumGate, params: [Double]) -> QuantumGate {
    guard !params.isEmpty else { return gate }

    switch gate {
    case .phase where params.count >= 1:
        return .phase(.value(params[0]))
    case .rotationX where params.count >= 1:
        return .rotationX(.value(params[0]))
    case .rotationY where params.count >= 1:
        return .rotationY(.value(params[0]))
    case .rotationZ where params.count >= 1:
        return .rotationZ(.value(params[0]))
    case .u1 where params.count >= 1:
        return .u1(lambda: .value(params[0]))
    case .u2 where params.count >= 2:
        return .u2(phi: .value(params[0]), lambda: .value(params[1]))
    case .u3 where params.count >= 3:
        return .u3(theta: .value(params[0]), phi: .value(params[1]), lambda: .value(params[2]))
    case .controlledPhase where params.count >= 1:
        return .controlledPhase(.value(params[0]))
    case .controlledRotationX where params.count >= 1:
        return .controlledRotationX(.value(params[0]))
    case .controlledRotationY where params.count >= 1:
        return .controlledRotationY(.value(params[0]))
    case .controlledRotationZ where params.count >= 1:
        return .controlledRotationZ(.value(params[0]))
    case .givens where params.count >= 1:
        return .givens(.value(params[0]))
    case .xx where params.count >= 1:
        return .xx(.value(params[0]))
    case .yy where params.count >= 1:
        return .yy(.value(params[0]))
    case .zz where params.count >= 1:
        return .zz(.value(params[0]))
    case .globalPhase where params.count >= 1:
        return .globalPhase(.value(params[0]))
    default:
        return gate
    }
}

/// Expand a user-defined gate by substituting parameters and qubit bindings.
private func expandGateDefinition(
    _ state: inout ParserState,
    definition: GateDefinition,
    params: [Double],
    qubits: [Int],
) {
    var paramBindings: [String: Double] = [:]
    for (i, pName) in definition.parameterNames.enumerated() where i < params.count {
        paramBindings[pName] = params[i]
    }

    var qubitBindings: [String: [Int]] = [:]
    for (i, qName) in definition.qubitNames.enumerated() where i < qubits.count {
        qubitBindings[qName] = [qubits[i]]
    }

    var subState = ParserState(tokens: definition.bodyTokens)
    subState.qubitDeclarations = state.qubitDeclarations
    subState.bitDeclarations = state.bitDeclarations
    subState.totalQubits = state.totalQubits
    subState.totalBits = state.totalBits
    subState.gateDefinitions = state.gateDefinitions

    while currentToken(&subState) != .eof {
        skipNewlines(&subState)
        if currentToken(&subState) == .eof { break }

        guard case let .identifier(innerGateName) = currentToken(&subState) else {
            advance(&subState)
            continue
        }
        advance(&subState)
        skipNewlines(&subState)

        var innerParams: [Double] = []
        if case .symbol("(") = currentToken(&subState) {
            advance(&subState)
            let rawParams = parseExpressionListWithBindings(&subState, bindings: paramBindings)
            innerParams = rawParams
            expect(&subState, symbol: ")")
        }

        skipNewlines(&subState)
        var innerQubits: [Int] = []
        while currentToken(&subState) != .eof {
            if case .symbol(";") = currentToken(&subState) { break }
            if currentToken(&subState) == .newline { break }

            if case let .identifier(qName) = currentToken(&subState) {
                advance(&subState)
                skipNewlines(&subState)
                if case .symbol("[") = currentToken(&subState) {
                    advance(&subState)
                    let idx = parseIntegerLiteral(&subState)
                    expect(&subState, symbol: "]")
                    if let mapped = qubitBindings[qName] {
                        if idx < mapped.count { innerQubits.append(mapped[idx]) }
                        else { innerQubits.append(mapped[0] + idx) }
                    }
                } else {
                    if let mapped = qubitBindings[qName] {
                        innerQubits.append(contentsOf: mapped)
                    }
                }
            } else {
                advance(&subState)
            }

            skipNewlines(&subState)
            if case .symbol(",") = currentToken(&subState) {
                advance(&subState)
                skipNewlines(&subState)
            }
        }
        if case .symbol(";") = currentToken(&subState) { advance(&subState) }

        if let resolved = resolveGate(innerGateName, params: innerParams, state: &state) {
            state.operations.append(.gate(resolved, qubits: innerQubits))
        }
    }
}

/// Parse a comma-separated list of qubit arguments into resolved indices.
private func parseQubitArgList(_ state: inout ParserState) -> [Int] {
    var qubits: [Int] = []
    skipNewlines(&state)

    while true {
        skipNewlines(&state)
        let token = currentToken(&state)

        switch token {
        case let .identifier(name):
            advance(&state)
            skipNewlines(&state)
            if case .symbol("[") = currentToken(&state) {
                advance(&state)
                let index = parseIntegerLiteral(&state)
                expect(&state, symbol: "]")
                if let decl = state.qubitDeclarations[name] {
                    qubits.append(decl.offset + index)
                } else {
                    qubits.append(index)
                    addError(&state, "undeclared qubit register '\(name)'")
                }
            } else {
                if let decl = state.qubitDeclarations[name] {
                    if decl.size == 1 {
                        qubits.append(decl.offset)
                    } else {
                        for i in 0 ..< decl.size {
                            qubits.append(decl.offset + i)
                        }
                    }
                } else {
                    addError(&state, "undeclared qubit register '\(name)'")
                }
            }
        default:
            return qubits
        }

        skipNewlines(&state)
        if case .symbol(",") = currentToken(&state) {
            advance(&state)
        } else {
            break
        }
    }

    return qubits
}

/// Parse a numeric expression and return its evaluated value.
private func parseExpression(_ state: inout ParserState) -> Double {
    skipNewlines(&state)
    return parseAdditive(&state)
}

/// Parse an additive expression (addition and subtraction).
private func parseAdditive(_ state: inout ParserState) -> Double {
    var left = parseMultiplicative(&state)
    while true {
        skipNewlines(&state)
        if case .symbol("+") = currentToken(&state) {
            advance(&state)
            left += parseMultiplicative(&state)
        } else if case .symbol("-") = currentToken(&state) {
            advance(&state)
            left -= parseMultiplicative(&state)
        } else {
            break
        }
    }
    return left
}

/// Parse a multiplicative expression (multiplication and division).
private func parseMultiplicative(_ state: inout ParserState) -> Double {
    var left = parseUnary(&state)
    while true {
        skipNewlines(&state)
        if case .symbol("*") = currentToken(&state) {
            advance(&state)
            left *= parseUnary(&state)
        } else if case .symbol("/") = currentToken(&state) {
            advance(&state)
            let right = parseUnary(&state)
            if right != 0 { left /= right }
        } else {
            break
        }
    }
    return left
}

/// Parse a unary expression (unary plus or minus).
private func parseUnary(_ state: inout ParserState) -> Double {
    skipNewlines(&state)
    if case .symbol("-") = currentToken(&state) {
        advance(&state)
        return -parsePrimary(&state)
    }
    if case .symbol("+") = currentToken(&state) {
        advance(&state)
        return parsePrimary(&state)
    }
    return parsePrimary(&state)
}

/// Parse a primary expression (literal, constant, function call, or parenthesized group).
private func parsePrimary(_ state: inout ParserState) -> Double {
    skipNewlines(&state)
    let token = currentToken(&state)

    switch token {
    case let .integer(v):
        advance(&state)
        return Double(v)
    case let .real(v):
        advance(&state)
        return v
    case .identifier("pi"):
        advance(&state)
        return Double.pi
    case .identifier("tau"):
        advance(&state)
        return 2.0 * Double.pi
    case .identifier("euler"):
        advance(&state)
        return M_E
    case .identifier("true"):
        advance(&state)
        return 1.0
    case .identifier("false"):
        advance(&state)
        return 0.0
    case .identifier("sin"):
        advance(&state)
        return parseFunctionCall(&state, fn: sin)
    case .identifier("cos"):
        advance(&state)
        return parseFunctionCall(&state, fn: cos)
    case .identifier("tan"):
        advance(&state)
        return parseFunctionCall(&state, fn: tan)
    case .identifier("exp"):
        advance(&state)
        return parseFunctionCall(&state, fn: exp)
    case .identifier("ln"):
        advance(&state)
        return parseFunctionCall(&state, fn: log)
    case .identifier("log"):
        advance(&state)
        return parseFunctionCall(&state, fn: log)
    case .identifier("sqrt"):
        advance(&state)
        return parseFunctionCall(&state, fn: sqrt)
    case .identifier("asin"):
        advance(&state)
        return parseFunctionCall(&state, fn: asin)
    case .identifier("acos"):
        advance(&state)
        return parseFunctionCall(&state, fn: acos)
    case .identifier("atan"):
        advance(&state)
        return parseFunctionCall(&state, fn: atan)
    case .identifier:
        advance(&state)
        return 0.0
    case .symbol("("):
        advance(&state)
        let value = parseExpression(&state)
        expect(&state, symbol: ")")
        return value
    default:
        addError(&state, "expected expression, got \(token)")
        return 0.0
    }
}

/// Parse a parenthesized function call and apply the given function.
private func parseFunctionCall(_ state: inout ParserState, fn: (Double) -> Double) -> Double {
    skipNewlines(&state)
    expect(&state, symbol: "(")
    let arg = parseExpression(&state)
    expect(&state, symbol: ")")
    return fn(arg)
}

/// Parse a comma-separated list of expressions.
private func parseExpressionList(_ state: inout ParserState) -> [Double] {
    var values: [Double] = []
    skipNewlines(&state)
    if case .symbol(")") = currentToken(&state) { return values }

    values.append(parseExpression(&state))
    while true {
        skipNewlines(&state)
        guard case .symbol(",") = currentToken(&state) else { break }
        advance(&state)
        values.append(parseExpression(&state))
    }
    return values
}

/// Parse a comma-separated list of expressions with parameter name bindings.
private func parseExpressionListWithBindings(_ state: inout ParserState, bindings: [String: Double]) -> [Double] {
    var values: [Double] = []
    skipNewlines(&state)
    if case .symbol(")") = currentToken(&state) { return values }

    values.append(parseExpressionWithBindings(&state, bindings: bindings))
    while true {
        skipNewlines(&state)
        guard case .symbol(",") = currentToken(&state) else { break }
        advance(&state)
        values.append(parseExpressionWithBindings(&state, bindings: bindings))
    }
    return values
}

/// Parse a single expression, substituting bound parameter names with their values.
private func parseExpressionWithBindings(_ state: inout ParserState, bindings: [String: Double]) -> Double {
    skipNewlines(&state)
    let token = currentToken(&state)
    if case let .identifier(name) = token, let value = bindings[name] {
        advance(&state)
        return value
    }
    return parseExpression(&state)
}

/// Parse an integer literal token and return its value.
private func parseIntegerLiteral(_ state: inout ParserState) -> Int {
    skipNewlines(&state)
    switch currentToken(&state) {
    case let .integer(v):
        advance(&state)
        return v
    case let .real(v):
        advance(&state)
        return Int(v)
    default:
        addError(&state, "expected integer literal")
        return 1
    }
}

/// Parse a comma-separated list of identifier names.
private func parseIdentifierList(_ state: inout ParserState) -> [String] {
    var names: [String] = []
    skipNewlines(&state)

    while case let .identifier(name) = currentToken(&state) {
        names.append(name)
        advance(&state)
        skipNewlines(&state)
        if case .symbol(",") = currentToken(&state) {
            advance(&state)
            skipNewlines(&state)
        } else {
            break
        }
    }
    return names
}

/// Parse a comma-separated list of typed parameter declarations, extracting names.
private func parseParameterDeclList(_ state: inout ParserState) -> [String] {
    var names: [String] = []
    skipNewlines(&state)
    if case .symbol(")") = currentToken(&state) { return names }

    while currentToken(&state) != .eof {
        skipNewlines(&state)
        if case .symbol(")") = currentToken(&state) { break }

        while currentToken(&state) != .eof {
            let t = currentToken(&state)
            if case let .identifier(name) = t {
                names.append(name)
                advance(&state)
                break
            }
            if case .symbol(")") = t { break }
            if case .symbol(",") = t { break }
            advance(&state)
        }

        skipNewlines(&state)
        if case .symbol(",") = currentToken(&state) {
            advance(&state)
        } else {
            break
        }
    }
    return names
}

/// Parse a braced block and return the contained tokens.
private func parseBracedBody(_ state: inout ParserState) -> [QASMToken] {
    skipNewlines(&state)
    guard case .symbol("{") = currentToken(&state) else {
        addError(&state, "expected '{'")
        return []
    }
    advance(&state)

    var tokens: [QASMToken] = []
    var depth = 1
    while depth > 0, currentToken(&state) != .eof {
        let token = currentToken(&state)
        if case .symbol("{") = token { depth += 1 }
        if case .symbol("}") = token { depth -= 1; if depth == 0 { advance(&state); break } }
        tokens.append(token)
        advance(&state)
    }
    tokens.append(.eof)
    return tokens
}

/// Skip over balanced parentheses including their contents.
private func skipBalancedParens(_ state: inout ParserState) {
    guard case .symbol("(") = currentToken(&state) else { return }
    advance(&state)
    var depth = 1
    while depth > 0, currentToken(&state) != .eof {
        if case .symbol("(") = currentToken(&state) { depth += 1 }
        if case .symbol(")") = currentToken(&state) { depth -= 1 }
        advance(&state)
    }
}

/// Advance the parser until a semicolon token is found.
private func skipToSemicolon(_ state: inout ParserState) {
    while currentToken(&state) != .eof {
        if case .symbol(";") = currentToken(&state) { return }
        advance(&state)
    }
}

/// Advance the parser until the specified symbol token is found.
private func skipToSymbol(_ state: inout ParserState, _ sym: Character) {
    while currentToken(&state) != .eof {
        if case .symbol(sym) = currentToken(&state) { return }
        advance(&state)
    }
}
