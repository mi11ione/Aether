// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// OpenQASM 2.0 circuit parser with full error recovery.
///
/// Recursive descent parser that converts OpenQASM 2.0 source text into a ``QuantumCircuit``
/// with comprehensive multi-error diagnostics. Uses ``QASMLexer`` for tokenization and
/// ``GateNameMapping`` for gate name resolution. On parse error the parser records a
/// ``ParseDiagnostic``, skips to the next synchronization point (semicolon or closing brace),
/// and continues parsing to collect all errors in a single pass.
///
/// **Example:**
/// ```swift
/// let source = "OPENQASM 2.0; qreg q[2]; h q[0]; cx q[0],q[1];"
/// let result = QASM2Importer.parse(source)
/// print(result.succeeded)       // true
/// print(result.circuit.qubits)  // 2
/// ```
///
/// - SeeAlso: ``QASMLexer``
/// - SeeAlso: ``GateNameMapping``
/// - SeeAlso: ``ParseResult``
/// - SeeAlso: ``ParseDiagnostic``
public enum QASM2Importer: Sendable {
    /// Parse OpenQASM 2.0 source text into a quantum circuit with diagnostics.
    ///
    /// Tokenizes the source via ``QASMLexer/tokenize(_:version:)`` then performs a full
    /// recursive descent parse of the QASM 2.0 grammar. Errors are accumulated into the
    /// returned ``ParseResult/diagnostics`` array rather than thrown. The parser recovers
    /// from each error by advancing to the next semicolon or closing brace and resuming
    /// statement-level parsing.
    ///
    /// **Example:**
    /// ```swift
    /// let qasm = "OPENQASM 2.0; qreg q[2]; h q[0]; cx q[0],q[1];"
    /// let result = QASM2Importer.parse(qasm)
    /// let state = result.circuit.execute()
    /// ```
    ///
    /// - Parameter source: OpenQASM 2.0 source text
    /// - Returns: ``ParseResult`` containing the best-effort circuit and all diagnostics
    ///
    /// - Complexity: O(n) where n is the number of tokens
    @_optimize(speed)
    public static func parse(_ source: String) -> ParseResult {
        var state = ParserState(tokens: QASMLexer.tokenize(source, version: .v2))
        state.parseProgram()
        return state.buildResult()
    }
}

/// Custom gate definition with parameter and qubit names.
private struct CustomGateDefinition {
    let name: String
    let parameterNames: [String]
    let qubitNames: [String]
    let bodyTokens: [QASMToken]
}

/// Mutable state for the recursive-descent QASM 2.0 parser.
private struct ParserState {
    var tokens: [QASMToken]
    var position: Int = 0
    var diagnostics: [ParseDiagnostic] = []
    var qubitRegisters: [(name: String, size: Int)] = []
    var classicalRegisters: [(name: String, size: Int)] = []
    var operations: [(gate: QuantumGate, qubits: [Int])] = []
    var customGates: [String: CustomGateDefinition] = [:]
    var line: Int = 1
    var column: Int = 1

    /// Initialize parser state with a token stream.
    init(tokens: [QASMToken]) {
        self.tokens = tokens
    }

    /// Return current token without advancing.
    var current: QASMToken {
        guard position < tokens.count else { return .eof }
        return tokens[position]
    }

    /// Return current token and advance position.
    mutating func advance() -> QASMToken {
        let token = current
        if position < tokens.count {
            updatePosition(for: token)
            position += 1
        }
        return token
    }

    /// Advance position and track line/column.
    private mutating func updatePosition(for token: QASMToken) {
        switch token {
        case .newline:
            line += 1
            column = 1
        default:
            column += 1
        }
    }

    /// Skip newline tokens silently.
    mutating func skipNewlines() {
        while case .newline = current {
            _ = advance()
        }
    }

    /// Check whether the current token matches a specific keyword.
    func isKeyword(_ name: String) -> Bool {
        if case let .keyword(k) = current, k == name { return true }
        return false
    }

    /// Check whether the current token matches a specific symbol.
    func isSymbol(_ char: Character) -> Bool {
        if case let .symbol(c) = current, c == char { return true }
        return false
    }

    /// Consume a specific symbol or record an error.
    mutating func expectSymbol(_ char: Character) -> Bool {
        skipNewlines()
        if isSymbol(char) {
            _ = advance()
            return true
        }
        addError("expected '\(char)'")
        return false
    }

    /// Consume an identifier and return its name, or record an error.
    mutating func expectIdentifier() -> String? {
        skipNewlines()
        if case let .identifier(name) = current {
            _ = advance()
            return name
        }
        if case let .keyword(name) = current {
            _ = advance()
            return name
        }
        addError("expected identifier")
        return nil
    }

    /// Consume an integer literal and return its value, or record an error.
    mutating func expectInteger() -> Int? {
        skipNewlines()
        if case let .integer(value) = current {
            _ = advance()
            return value
        }
        addError("expected integer")
        return nil
    }

    /// Record an error diagnostic at the current position.
    mutating func addError(_ message: String) {
        diagnostics.append(ParseDiagnostic(line: line, column: column, message: message, severity: .error))
    }

    /// Record a warning diagnostic at the current position.
    mutating func addWarning(_ message: String) {
        diagnostics.append(ParseDiagnostic(line: line, column: column, message: message, severity: .warning))
    }

    /// Skip tokens until a synchronization point (semicolon, closing brace, or EOF).
    mutating func synchronize() {
        while true {
            switch current {
            case .symbol(";"):
                _ = advance()
                return
            case .symbol("}"):
                _ = advance()
                return
            case .eof:
                return
            default:
                _ = advance()
            }
        }
    }

    /// Resolve register name and index to absolute qubit index.
    func resolveQubit(register: String, index: Int) -> Int? {
        var offset = 0
        for reg in qubitRegisters {
            if reg.name == register {
                guard index >= 0, index < reg.size else { return nil }
                return offset + index
            }
            offset += reg.size
        }
        return nil
    }

    /// Total number of qubits declared across all quantum registers.
    var totalQubits: Int {
        var total = 0
        for reg in qubitRegisters {
            total += reg.size
        }
        return total
    }

    /// Parse the entire QASM 2.0 program.
    mutating func parseProgram() {
        skipNewlines()
        parseHeader()
        while true {
            skipNewlines()
            if case .eof = current { break }
            parseStatement()
        }
    }

    /// Parse the OPENQASM 2.0 header.
    mutating func parseHeader() {
        skipNewlines()
        if isKeyword("OPENQASM") {
            _ = advance()
            skipNewlines()
            switch current {
            case let .real(v):
                _ = advance()
                if v != 2.0 {
                    addWarning("expected version 2.0, got \(v)")
                }
            case let .integer(v):
                _ = advance()
                if v != 2 {
                    addWarning("expected version 2.0, got \(v)")
                }
            default:
                addError("expected version number after OPENQASM")
            }
            skipNewlines()
            _ = expectSymbol(";")
        } else {
            addError("expected 'OPENQASM' header")
        }
    }

    /// Parse a single statement and dispatch by keyword.
    mutating func parseStatement() {
        skipNewlines()
        switch current {
        case .keyword("include"):
            parseInclude()
        case .keyword("qreg"):
            parseQregDecl()
        case .keyword("creg"):
            parseCregDecl()
        case .keyword("gate"):
            parseGateDecl()
        case .keyword("opaque"):
            parseOpaqueDecl()
        case .keyword("barrier"):
            parseBarrier()
        case .keyword("measure"):
            parseMeasure()
        case .keyword("reset"):
            parseReset()
        case .keyword("if"):
            parseIfStatement()
        case .identifier:
            parseGateOperation()
        case .eof:
            return
        default:
            addError("unexpected token: \(current)")
            synchronize()
        }
    }

    /// Parse include statement (silently consumes).
    mutating func parseInclude() {
        _ = advance()
        skipNewlines()
        if case .string = current {
            _ = advance()
        } else {
            addError("expected filename string after 'include'")
            synchronize()
            return
        }
        skipNewlines()
        _ = expectSymbol(";")
    }

    /// Parse quantum register declaration.
    mutating func parseQregDecl() {
        _ = advance()
        skipNewlines()
        guard let name = expectIdentifier() else {
            synchronize()
            return
        }
        skipNewlines()
        guard expectSymbol("[") else {
            synchronize()
            return
        }
        skipNewlines()
        guard let size = expectInteger() else {
            synchronize()
            return
        }
        skipNewlines()
        guard expectSymbol("]") else {
            synchronize()
            return
        }
        skipNewlines()
        _ = expectSymbol(";")
        qubitRegisters.append((name: name, size: size))
    }

    /// Parse classical register declaration.
    mutating func parseCregDecl() {
        _ = advance()
        skipNewlines()
        guard let name = expectIdentifier() else {
            synchronize()
            return
        }
        skipNewlines()
        guard expectSymbol("[") else {
            synchronize()
            return
        }
        skipNewlines()
        guard let size = expectInteger() else {
            synchronize()
            return
        }
        skipNewlines()
        guard expectSymbol("]") else {
            synchronize()
            return
        }
        skipNewlines()
        _ = expectSymbol(";")
        classicalRegisters.append((name: name, size: size))
    }

    /// Parse gate declaration and store as custom gate definition.
    mutating func parseGateDecl() {
        _ = advance()
        skipNewlines()
        guard let name = expectIdentifier() else {
            synchronize()
            return
        }

        var parameterNames: [String] = []
        skipNewlines()
        if isSymbol("(") {
            _ = advance()
            parameterNames = parseIdentifierList()
            skipNewlines()
            _ = expectSymbol(")")
        }

        skipNewlines()
        let qubitNames = parseIdentifierList()

        skipNewlines()
        guard expectSymbol("{") else {
            synchronize()
            return
        }

        var bodyTokens: [QASMToken] = []
        var depth = 1
        while depth > 0 {
            switch current {
            case .symbol("{"):
                depth += 1
                bodyTokens.append(advance())
            case .symbol("}"):
                depth -= 1
                if depth > 0 {
                    bodyTokens.append(advance())
                } else {
                    _ = advance()
                }
            case .eof:
                addError("unexpected end of file in gate body")
                return
            default:
                bodyTokens.append(advance())
            }
        }

        customGates[name] = CustomGateDefinition(
            name: name,
            parameterNames: parameterNames,
            qubitNames: qubitNames,
            bodyTokens: bodyTokens,
        )
    }

    /// Parse opaque gate declaration (silently skip body).
    mutating func parseOpaqueDecl() {
        _ = advance()
        synchronize()
    }

    /// Parse barrier statement (silently consumes).
    mutating func parseBarrier() {
        _ = advance()
        synchronize()
    }

    /// Parse measure statement (silently consumes).
    mutating func parseMeasure() {
        _ = advance()
        synchronize()
    }

    /// Parse reset statement and emit reset operation.
    mutating func parseReset() {
        _ = advance()
        skipNewlines()
        guard let qubitArg = parseQubitArg() else {
            synchronize()
            return
        }
        guard let absoluteIndex = resolveQubit(register: qubitArg.register, index: qubitArg.index) else {
            addError("invalid qubit reference '\(qubitArg.register)[\(qubitArg.index)]'")
            synchronize()
            return
        }
        skipNewlines()
        _ = expectSymbol(";")
        operations.append((gate: .identity, qubits: [-1 - absoluteIndex]))
    }

    /// Parse if statement with conditional gate application.
    mutating func parseIfStatement() {
        _ = advance()
        skipNewlines()
        guard expectSymbol("(") else {
            synchronize()
            return
        }
        skipNewlines()
        _ = expectIdentifier()
        skipNewlines()
        if isSymbol("=") {
            _ = advance()
            skipNewlines()
            if isSymbol("=") {
                _ = advance()
            }
        }
        skipNewlines()
        _ = expectInteger()
        skipNewlines()
        guard expectSymbol(")") else {
            synchronize()
            return
        }
        skipNewlines()
        addWarning("conditional 'if' statement ignored; gate applied unconditionally")
        parseGateOperation()
    }

    /// Parse a gate application statement.
    mutating func parseGateOperation() {
        skipNewlines()
        guard let gateName = expectIdentifier() else {
            synchronize()
            return
        }

        var params: [Double] = []
        skipNewlines()
        if isSymbol("(") {
            _ = advance()
            params = parseParameterList()
            skipNewlines()
            _ = expectSymbol(")")
        }

        skipNewlines()
        let qubitArgs = parseQubitArgList()

        skipNewlines()
        _ = expectSymbol(";")

        if let customDef = customGates[gateName] {
            applyCustomGate(customDef, params: params, qubitArgs: qubitArgs)
            return
        }

        guard let templateGate = GateNameMapping.gate(forQASMName: gateName, version: .v2) else {
            addError("unknown gate '\(gateName)'")
            return
        }

        var absoluteQubits: [Int] = []
        absoluteQubits.reserveCapacity(qubitArgs.count)
        for arg in qubitArgs {
            guard let idx = resolveQubit(register: arg.register, index: arg.index) else {
                addError("invalid qubit reference '\(arg.register)[\(arg.index)]'")
                return
            }
            absoluteQubits.append(idx)
        }

        let gate = applyParameters(to: templateGate, params: params)
        operations.append((gate: gate, qubits: absoluteQubits))
    }

    /// Apply parsed parameter values to a template gate.
    func applyParameters(to gate: QuantumGate, params: [Double]) -> QuantumGate {
        switch gate {
        case .rotationX where params.count >= 1:
            .rotationX(params[0])
        case .rotationY where params.count >= 1:
            .rotationY(params[0])
        case .rotationZ where params.count >= 1:
            .rotationZ(params[0])
        case .u1 where params.count >= 1:
            .u1(lambda: params[0])
        case .u2 where params.count >= 2:
            .u2(phi: params[0], lambda: params[1])
        case .u3 where params.count >= 3:
            .u3(theta: params[0], phi: params[1], lambda: params[2])
        case .controlledPhase where params.count >= 1:
            .controlledPhase(params[0])
        case .controlledRotationX where params.count >= 1:
            .controlledRotationX(params[0])
        case .controlledRotationY where params.count >= 1:
            .controlledRotationY(params[0])
        case .controlledRotationZ where params.count >= 1:
            .controlledRotationZ(params[0])
        case .globalPhase where params.count >= 1:
            .globalPhase(params[0])
        case .givens where params.count >= 1:
            .givens(params[0])
        case .xx where params.count >= 1:
            .xx(params[0])
        case .yy where params.count >= 1:
            .yy(params[0])
        case .zz where params.count >= 1:
            .zz(params[0])
        default:
            gate
        }
    }

    /// Apply a custom gate definition with resolved parameters and qubits.
    mutating func applyCustomGate(
        _ definition: CustomGateDefinition,
        params: [Double],
        qubitArgs: [(register: String, index: Int)],
    ) {
        var absoluteQubits: [Int] = []
        absoluteQubits.reserveCapacity(qubitArgs.count)
        for arg in qubitArgs {
            guard let idx = resolveQubit(register: arg.register, index: arg.index) else {
                addError("invalid qubit reference '\(arg.register)[\(arg.index)]'")
                return
            }
            absoluteQubits.append(idx)
        }

        var paramBindings: [String: Double] = [:]
        paramBindings.reserveCapacity(definition.parameterNames.count)
        for i in 0 ..< min(params.count, definition.parameterNames.count) {
            paramBindings[definition.parameterNames[i]] = params[i]
        }

        var qubitBindings: [String: Int] = [:]
        qubitBindings.reserveCapacity(definition.qubitNames.count)
        for i in 0 ..< min(absoluteQubits.count, definition.qubitNames.count) {
            qubitBindings[definition.qubitNames[i]] = absoluteQubits[i]
        }

        var bodyState = ParserState(tokens: definition.bodyTokens + [.eof])
        bodyState.qubitRegisters = qubitRegisters
        bodyState.classicalRegisters = classicalRegisters
        bodyState.customGates = customGates

        while true {
            bodyState.skipNewlines()
            if case .eof = bodyState.current { break }
            bodyState.parseCustomGateBody(
                paramBindings: paramBindings,
                qubitBindings: qubitBindings,
            )
        }

        operations.append(contentsOf: bodyState.operations)
        diagnostics.append(contentsOf: bodyState.diagnostics)
    }

    /// Parse gate body statement within a custom gate definition.
    mutating func parseCustomGateBody(
        paramBindings: [String: Double],
        qubitBindings: [String: Int],
    ) {
        skipNewlines()
        guard let gateName = expectIdentifier() else {
            synchronize()
            return
        }

        var params: [Double] = []
        skipNewlines()
        if isSymbol("(") {
            _ = advance()
            params = parseParameterListWithBindings(paramBindings)
            skipNewlines()
            _ = expectSymbol(")")
        }

        skipNewlines()
        var qubits: [Int] = []
        while true {
            skipNewlines()
            if case let .identifier(name) = current {
                _ = advance()
                if let idx = qubitBindings[name] {
                    qubits.append(idx)
                } else {
                    addError("unknown qubit '\(name)' in gate body")
                }
            } else if case let .keyword(name) = current {
                _ = advance()
                if let idx = qubitBindings[name] {
                    qubits.append(idx)
                } else {
                    addError("unknown qubit '\(name)' in gate body")
                }
            } else {
                break
            }
            skipNewlines()
            if isSymbol(",") {
                _ = advance()
            } else {
                break
            }
        }

        skipNewlines()
        if isSymbol(";") {
            _ = advance()
        }

        guard let templateGate = GateNameMapping.gate(forQASMName: gateName, version: .v2) else {
            addError("unknown gate '\(gateName)' in gate body")
            return
        }

        let gate = applyParameters(to: templateGate, params: params)
        operations.append((gate: gate, qubits: qubits))
    }

    /// Parse a comma-separated list of parameter expressions.
    mutating func parseParameterList() -> [Double] {
        var params: [Double] = []
        skipNewlines()
        if isSymbol(")") { return params }

        if let value = parseExpression() {
            params.append(value)
        }
        while isSymbol(",") {
            _ = advance()
            skipNewlines()
            if let value = parseExpression() {
                params.append(value)
            }
        }
        return params
    }

    /// Parse parameter list resolving identifiers against bindings.
    mutating func parseParameterListWithBindings(_ bindings: [String: Double]) -> [Double] {
        var params: [Double] = []
        skipNewlines()
        if isSymbol(")") { return params }

        if let value = parseExpressionWithBindings(bindings) {
            params.append(value)
        }
        while isSymbol(",") {
            _ = advance()
            skipNewlines()
            if let value = parseExpressionWithBindings(bindings) {
                params.append(value)
            }
        }
        return params
    }

    /// Parse an arithmetic expression with standard precedence.
    mutating func parseExpression() -> Double? {
        parseAdditive()
    }

    /// Parse expression resolving identifiers against parameter bindings.
    mutating func parseExpressionWithBindings(_ bindings: [String: Double]) -> Double? {
        parseAdditiveWithBindings(bindings)
    }

    /// Parse additive expression (+ and -).
    mutating func parseAdditive() -> Double? {
        guard var left = parseMultiplicative() else { return nil }
        skipNewlines()
        while true {
            if isSymbol("+") {
                _ = advance()
                skipNewlines()
                guard let right = parseMultiplicative() else { return nil }
                left += right
            } else if isSymbol("-") {
                _ = advance()
                skipNewlines()
                guard let right = parseMultiplicative() else { return nil }
                left -= right
            } else {
                break
            }
        }
        return left
    }

    /// Parse additive expression with bindings.
    mutating func parseAdditiveWithBindings(_ bindings: [String: Double]) -> Double? {
        guard var left = parseMultiplicativeWithBindings(bindings) else { return nil }
        skipNewlines()
        while true {
            if isSymbol("+") {
                _ = advance()
                skipNewlines()
                guard let right = parseMultiplicativeWithBindings(bindings) else { return nil }
                left += right
            } else if isSymbol("-") {
                _ = advance()
                skipNewlines()
                guard let right = parseMultiplicativeWithBindings(bindings) else { return nil }
                left -= right
            } else {
                break
            }
        }
        return left
    }

    /// Parse multiplicative expression (* and /).
    mutating func parseMultiplicative() -> Double? {
        guard var left = parseUnary() else { return nil }
        skipNewlines()
        while true {
            if isSymbol("*") {
                _ = advance()
                skipNewlines()
                guard let right = parseUnary() else { return nil }
                left *= right
            } else if isSymbol("/") {
                _ = advance()
                skipNewlines()
                guard let right = parseUnary() else { return nil }
                left /= right
            } else {
                break
            }
        }
        return left
    }

    /// Parse multiplicative expression with bindings.
    mutating func parseMultiplicativeWithBindings(_ bindings: [String: Double]) -> Double? {
        guard var left = parseUnaryWithBindings(bindings) else { return nil }
        skipNewlines()
        while true {
            if isSymbol("*") {
                _ = advance()
                skipNewlines()
                guard let right = parseUnaryWithBindings(bindings) else { return nil }
                left *= right
            } else if isSymbol("/") {
                _ = advance()
                skipNewlines()
                guard let right = parseUnaryWithBindings(bindings) else { return nil }
                left /= right
            } else {
                break
            }
        }
        return left
    }

    /// Parse unary expression (unary minus).
    mutating func parseUnary() -> Double? {
        skipNewlines()
        if isSymbol("-") {
            _ = advance()
            skipNewlines()
            guard let operand = parseUnary() else { return nil }
            return -operand
        }
        if isSymbol("+") {
            _ = advance()
            skipNewlines()
            return parseUnary()
        }
        return parsePrimary()
    }

    /// Parse unary expression with bindings.
    mutating func parseUnaryWithBindings(_ bindings: [String: Double]) -> Double? {
        skipNewlines()
        if isSymbol("-") {
            _ = advance()
            skipNewlines()
            guard let operand = parseUnaryWithBindings(bindings) else { return nil }
            return -operand
        }
        if isSymbol("+") {
            _ = advance()
            skipNewlines()
            return parseUnaryWithBindings(bindings)
        }
        return parsePrimaryWithBindings(bindings)
    }

    /// Parse primary expression (numbers, pi, functions, parenthesized expressions).
    mutating func parsePrimary() -> Double? {
        skipNewlines()
        switch current {
        case let .integer(v):
            _ = advance()
            return Double(v)
        case let .real(v):
            _ = advance()
            return v
        case .identifier("pi"):
            _ = advance()
            return Double.pi
        case .identifier("sin"):
            _ = advance()
            return parseFunctionCall(sin)
        case .identifier("cos"):
            _ = advance()
            return parseFunctionCall(cos)
        case .identifier("tan"):
            _ = advance()
            return parseFunctionCall(tan)
        case .identifier("exp"):
            _ = advance()
            return parseFunctionCall(exp)
        case .identifier("ln"):
            _ = advance()
            return parseFunctionCall(log)
        case .identifier("sqrt"):
            _ = advance()
            return parseFunctionCall(sqrt)
        case .symbol("("):
            _ = advance()
            let value = parseExpression()
            skipNewlines()
            _ = expectSymbol(")")
            return value
        default:
            addError("unexpected token in expression: \(current)")
            return nil
        }
    }

    /// Parse primary expression with bindings.
    mutating func parsePrimaryWithBindings(_ bindings: [String: Double]) -> Double? {
        skipNewlines()
        switch current {
        case let .integer(v):
            _ = advance()
            return Double(v)
        case let .real(v):
            _ = advance()
            return v
        case .identifier("pi"):
            _ = advance()
            return Double.pi
        case .identifier("sin"):
            _ = advance()
            return parseFunctionCallWithBindings(sin, bindings)
        case .identifier("cos"):
            _ = advance()
            return parseFunctionCallWithBindings(cos, bindings)
        case .identifier("tan"):
            _ = advance()
            return parseFunctionCallWithBindings(tan, bindings)
        case .identifier("exp"):
            _ = advance()
            return parseFunctionCallWithBindings(exp, bindings)
        case .identifier("ln"):
            _ = advance()
            return parseFunctionCallWithBindings(log, bindings)
        case .identifier("sqrt"):
            _ = advance()
            return parseFunctionCallWithBindings(sqrt, bindings)
        case let .identifier(name):
            _ = advance()
            if let value = bindings[name] {
                return value
            }
            addError("unbound parameter '\(name)'")
            return 0.0
        case .symbol("("):
            _ = advance()
            let value = parseExpressionWithBindings(bindings)
            skipNewlines()
            _ = expectSymbol(")")
            return value
        default:
            addError("unexpected token in expression: \(current)")
            return nil
        }
    }

    /// Parse a function call like sin(expr).
    mutating func parseFunctionCall(_ fn: (Double) -> Double) -> Double? {
        skipNewlines()
        guard expectSymbol("(") else { return nil }
        guard let arg = parseExpression() else { return nil }
        skipNewlines()
        _ = expectSymbol(")")
        return fn(arg)
    }

    /// Parse a function call with parameter bindings.
    mutating func parseFunctionCallWithBindings(
        _ fn: (Double) -> Double,
        _ bindings: [String: Double],
    ) -> Double? {
        skipNewlines()
        guard expectSymbol("(") else { return nil }
        guard let arg = parseExpressionWithBindings(bindings) else { return nil }
        skipNewlines()
        _ = expectSymbol(")")
        return fn(arg)
    }

    /// Parse a single qubit argument (register[index]).
    mutating func parseQubitArg() -> (register: String, index: Int)? {
        skipNewlines()
        guard let name = expectIdentifier() else { return nil }
        skipNewlines()
        guard expectSymbol("[") else { return nil }
        skipNewlines()
        guard let index = expectInteger() else { return nil }
        skipNewlines()
        _ = expectSymbol("]")
        return (register: name, index: index)
    }

    /// Parse comma-separated list of qubit arguments.
    mutating func parseQubitArgList() -> [(register: String, index: Int)] {
        var args: [(register: String, index: Int)] = []
        skipNewlines()
        guard let first = parseQubitArg() else { return args }
        args.append(first)
        while isSymbol(",") {
            _ = advance()
            skipNewlines()
            guard let arg = parseQubitArg() else { break }
            args.append(arg)
        }
        return args
    }

    /// Parse comma-separated list of identifiers.
    mutating func parseIdentifierList() -> [String] {
        var names: [String] = []
        skipNewlines()
        if case let .identifier(name) = current {
            _ = advance()
            names.append(name)
        } else {
            return names
        }
        while isSymbol(",") {
            _ = advance()
            skipNewlines()
            if case let .identifier(name) = current {
                _ = advance()
                names.append(name)
            } else {
                break
            }
        }
        return names
    }

    /// Build the final ParseResult from accumulated parser state.
    func buildResult() -> ParseResult {
        let qubitCount = max(totalQubits, 1)
        var circuit = QuantumCircuit(qubits: qubitCount)

        var circuitOps: [CircuitOperation] = []
        circuitOps.reserveCapacity(operations.count)

        for op in operations {
            if op.qubits.count == 1, op.qubits[0] < 0 {
                let resetQubit = -1 - op.qubits[0]
                circuitOps.append(.reset(qubit: resetQubit))
            } else {
                circuitOps.append(.gate(op.gate, qubits: op.qubits))
            }
        }

        for circuitOp in circuitOps {
            circuit.addOperation(circuitOp)
        }

        return ParseResult(circuit: circuit, diagnostics: diagnostics)
    }
}
