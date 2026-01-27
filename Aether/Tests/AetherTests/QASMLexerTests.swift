// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Validates QASMLexer tokenization of OpenQASM source text including whitespace handling,
/// comment skipping, numeric literal parsing with scientific notation, string and symbol
/// recognition, version-specific keyword classification, and QASMToken computed properties.
@Suite
struct QASMLexerTests {
    @Test func crlfLineEndingsProduceNewlineTokens() {
        let tokens = QASMLexer.tokenize("a\r\nb", version: .v2)
        #expect(tokens[0] == .identifier("a"), "first token should be identifier 'a'")
        #expect(tokens[1] == .newline, "CRLF should produce a single newline token")
        #expect(tokens[2] == .identifier("b"), "token after CRLF should be identifier 'b'")
        #expect(tokens[3] == .eof, "stream should end with eof")
    }

    @Test func crlfCountsAsSingleNewline() {
        let tokens = QASMLexer.tokenize("\r\n\r\n", version: .v2)
        let newlines = tokens.filter { $0 == .newline }
        #expect(newlines.count == 2, "two CRLF sequences should produce exactly two newline tokens")
    }

    @Test func lineCommentWithContentIsSkipped() {
        let tokens = QASMLexer.tokenize("// comment text\ngate", version: .v2)
        #expect(tokens[0] == .newline, "line comment content should be skipped, leaving newline")
        #expect(tokens[1] == .keyword("gate"), "token after line comment should be the next keyword")
        #expect(tokens[2] == .eof, "stream should end with eof")
    }

    @Test func lineCommentAtEndOfInputIsSkipped() {
        let tokens = QASMLexer.tokenize("x // trailing", version: .v2)
        #expect(tokens[0] == .identifier("x"), "identifier before comment should be present")
        #expect(tokens[1] == .eof, "line comment at end of input should be consumed, leaving eof")
    }

    @Test func blockCommentIsSkipped() {
        let tokens = QASMLexer.tokenize("/* comment */gate", version: .v2)
        #expect(tokens[0] == .keyword("gate"), "block comment should be skipped, next token is keyword")
        #expect(tokens[1] == .eof, "stream should end with eof after block comment and keyword")
    }

    @Test func blockCommentBetweenTokens() {
        let tokens = QASMLexer.tokenize("a/* mid */b", version: .v2)
        #expect(tokens[0] == .identifier("a"), "identifier before block comment should be present")
        #expect(tokens[1] == .identifier("b"), "identifier after block comment should be present")
        #expect(tokens[2] == .eof, "stream should end with eof")
    }

    @Test func crlfInsideBlockComment() {
        let tokens = QASMLexer.tokenize("/*\r\ntext*/gate", version: .v2)
        #expect(tokens[0] == .keyword("gate"), "CRLF inside block comment should be handled, next token is keyword")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func nonNewlineCharsInBlockComment() {
        let tokens = QASMLexer.tokenize("/*abc*/x", version: .v2)
        #expect(tokens[0] == .identifier("x"), "non-newline characters in block comment should be consumed")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func unterminatedBlockComment() {
        let tokens = QASMLexer.tokenize("/* no end", version: .v2)
        #expect(tokens.count == 1, "unterminated block comment should consume remaining input leaving only eof")
        #expect(tokens[0] == .eof, "unterminated block comment should still produce eof")
    }

    @Test func scientificNotationWithPlusSign() {
        let tokens = QASMLexer.tokenize("1e+5", version: .v2)
        #expect(tokens[0] == .real(1e+5), "1e+5 should be tokenized as a real literal with value 100000")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func scientificNotationWithMinusSign() {
        let tokens = QASMLexer.tokenize("2e-3", version: .v2)
        #expect(tokens[0] == .real(2e-3), "2e-3 should be tokenized as a real literal with value 0.002")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func scientificNotationWithoutSign() {
        let tokens = QASMLexer.tokenize("1e5", version: .v2)
        #expect(tokens[0] == .real(1e5), "1e5 should be tokenized as a real literal with value 100000")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func scientificNotationWithDecimalPoint() {
        let tokens = QASMLexer.tokenize("3.14e2", version: .v2)
        #expect(tokens[0] == .real(3.14e2), "3.14e2 should be tokenized as real with value 314.0")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func leadingDecimalPointProducesReal() {
        let tokens = QASMLexer.tokenize(".5", version: .v2)
        #expect(tokens[0] == .real(0.5), "leading decimal point .5 should produce real token with value 0.5")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func integerLiteral() {
        let tokens = QASMLexer.tokenize("42", version: .v2)
        #expect(tokens[0] == .integer(42), "42 should be tokenized as integer literal")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func realLiteralWithDecimalPoint() {
        let tokens = QASMLexer.tokenize("3.14", version: .v2)
        #expect(tokens[0] == .real(3.14), "3.14 should be tokenized as real literal")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func identifierTokenization() {
        let tokens = QASMLexer.tokenize("myVar", version: .v2)
        #expect(tokens[0] == .identifier("myVar"), "myVar should be tokenized as identifier")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func identifierWithUnderscoreAndDigits() {
        let tokens = QASMLexer.tokenize("_q0", version: .v2)
        #expect(tokens[0] == .identifier("_q0"), "underscore-prefixed identifier with digit should tokenize")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func stringLiteral() {
        let tokens = QASMLexer.tokenize("\"hello\"", version: .v2)
        #expect(tokens[0] == .string("hello"), "quoted text should produce string token with content 'hello'")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func symbolTokenization() {
        let tokens = QASMLexer.tokenize(";", version: .v2)
        #expect(tokens[0] == .symbol(";"), "semicolon should be tokenized as symbol")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func multipleSymbols() {
        let tokens = QASMLexer.tokenize("[]()", version: .v2)
        #expect(tokens[0] == .symbol("["), "open bracket should be tokenized as symbol")
        #expect(tokens[1] == .symbol("]"), "close bracket should be tokenized as symbol")
        #expect(tokens[2] == .symbol("("), "open paren should be tokenized as symbol")
        #expect(tokens[3] == .symbol(")"), "close paren should be tokenized as symbol")
        #expect(tokens[4] == .eof, "stream should end with eof")
    }

    @Test func slashNotFollowedByCommentIsSymbol() {
        let tokens = QASMLexer.tokenize("/", version: .v2)
        #expect(tokens[0] == .symbol("/"), "lone slash should be tokenized as symbol")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func keywordTokenization() {
        let tokens = QASMLexer.tokenize("OPENQASM", version: .v2)
        #expect(tokens[0] == .keyword("OPENQASM"), "OPENQASM should be recognized as keyword in v2")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func v2Keywords() {
        let v2Words = ["OPENQASM", "include", "qreg", "creg", "gate", "opaque", "barrier", "if", "measure", "reset"]
        for word in v2Words {
            let tokens = QASMLexer.tokenize(word, version: .v2)
            #expect(tokens[0] == .keyword(word), "\(word) should be a keyword in v2")
        }
    }

    @Test func v3AdditionalKeywords() {
        let v3Only = ["qubit", "bit", "int", "uint", "float", "angle", "bool",
                      "const", "let", "for", "while", "else", "return",
                      "def", "extern", "cal", "defcal", "delay", "stretch", "box",
                      "input", "output", "ctrl", "inv", "pow", "negctrl"]
        for word in v3Only {
            let tokensV3 = QASMLexer.tokenize(word, version: .v3)
            #expect(tokensV3[0] == .keyword(word), "\(word) should be a keyword in v3")
            let tokensV2 = QASMLexer.tokenize(word, version: .v2)
            #expect(tokensV2[0] == .identifier(word), "\(word) should be an identifier in v2, not a keyword")
        }
    }

    @Test func v2KeywordsAlsoRecognizedInV3() {
        let shared = ["OPENQASM", "qreg", "gate", "measure"]
        for word in shared {
            let tokens = QASMLexer.tokenize(word, version: .v3)
            #expect(tokens[0] == .keyword(word), "\(word) should be a keyword in v3 as well as v2")
        }
    }

    @Test func fullStatement() {
        let tokens = QASMLexer.tokenize("qreg q[2];", version: .v2)
        #expect(tokens[0] == .keyword("qreg"), "first token should be keyword qreg")
        #expect(tokens[1] == .identifier("q"), "second token should be identifier q")
        #expect(tokens[2] == .symbol("["), "third token should be open bracket symbol")
        #expect(tokens[3] == .integer(2), "fourth token should be integer 2")
        #expect(tokens[4] == .symbol("]"), "fifth token should be close bracket symbol")
        #expect(tokens[5] == .symbol(";"), "sixth token should be semicolon symbol")
        #expect(tokens[6] == .eof, "stream should end with eof")
    }

    @Test func emptyInputProducesOnlyEof() {
        let tokens = QASMLexer.tokenize("", version: .v2)
        #expect(tokens.count == 1, "empty input should produce exactly one token")
        #expect(tokens[0] == .eof, "empty input should produce only eof")
    }

    @Test func whitespaceOnlyInput() {
        let tokens = QASMLexer.tokenize("   \t\t  ", version: .v2)
        #expect(tokens.count == 1, "whitespace-only input should produce exactly one token")
        #expect(tokens[0] == .eof, "whitespace-only input should produce only eof")
    }

    @Test func newlineOnlyInput() {
        let tokens = QASMLexer.tokenize("\n", version: .v2)
        #expect(tokens[0] == .newline, "newline-only input should produce a newline token")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func tokenIsKeywordProperty() {
        let kw = QASMToken.keyword("OPENQASM")
        #expect(kw.isKeyword, "keyword token should return true for isKeyword")
        let id = QASMToken.identifier("foo")
        #expect(!id.isKeyword, "identifier token should return false for isKeyword")
        let num = QASMToken.integer(1)
        #expect(!num.isKeyword, "integer token should return false for isKeyword")
    }

    @Test func tokenIsIdentifierProperty() {
        let id = QASMToken.identifier("foo")
        #expect(id.isIdentifier, "identifier token should return true for isIdentifier")
        let kw = QASMToken.keyword("gate")
        #expect(!kw.isIdentifier, "keyword token should return false for isIdentifier")
        let sym = QASMToken.symbol(";")
        #expect(!sym.isIdentifier, "symbol token should return false for isIdentifier")
    }

    @Test func tokenIsNumberProperty() {
        let intToken = QASMToken.integer(42)
        #expect(intToken.isNumber, "integer token should return true for isNumber")
        let realToken = QASMToken.real(3.14)
        #expect(realToken.isNumber, "real token should return true for isNumber")
        let idToken = QASMToken.identifier("x")
        #expect(!idToken.isNumber, "identifier token should return false for isNumber")
        let strToken = QASMToken.string("hello")
        #expect(!strToken.isNumber, "string token should return false for isNumber")
        let eofToken = QASMToken.eof
        #expect(!eofToken.isNumber, "eof token should return false for isNumber")
    }

    @Test func parseDiagnosticErrorDescription() {
        let diag = ParseDiagnostic(line: 3, column: 15, message: "unexpected token", severity: .error)
        #expect(diag.description == "3:15: error: unexpected token",
                "error diagnostic description should follow 'line:column: error: message' format")
    }

    @Test func parseDiagnosticWarningDescription() {
        let diag = ParseDiagnostic(line: 10, column: 1, message: "deprecated syntax", severity: .warning)
        #expect(diag.description == "10:1: warning: deprecated syntax",
                "warning diagnostic description should follow 'line:column: warning: message' format")
    }

    @Test func multipleNewlinesProduceMultipleTokens() {
        let tokens = QASMLexer.tokenize("a\n\nb", version: .v2)
        #expect(tokens[0] == .identifier("a"), "first token should be identifier a")
        #expect(tokens[1] == .newline, "second token should be newline")
        #expect(tokens[2] == .newline, "third token should be second newline")
        #expect(tokens[3] == .identifier("b"), "fourth token should be identifier b")
        #expect(tokens[4] == .eof, "stream should end with eof")
    }

    @Test func blockCommentWithNewlineInside() {
        let tokens = QASMLexer.tokenize("/*\nline2\n*/x", version: .v2)
        #expect(tokens[0] == .identifier("x"), "block comment with newlines inside should be consumed")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func carriageReturnAloneProducesNewline() {
        let tokens = QASMLexer.tokenize("a\rb", version: .v2)
        #expect(tokens[0] == .identifier("a"), "first token should be identifier a")
        #expect(tokens[1] == .newline, "bare carriage return should produce newline token")
        #expect(tokens[2] == .identifier("b"), "token after bare CR should be identifier b")
        #expect(tokens[3] == .eof, "stream should end with eof")
    }

    @Test func isKeywordStaticMethod() {
        #expect(QASMToken.isKeyword("qreg", version: .v2), "qreg should be recognized as v2 keyword")
        #expect(!QASMToken.isKeyword("qubit", version: .v2), "qubit should not be a v2 keyword")
        #expect(QASMToken.isKeyword("qubit", version: .v3), "qubit should be recognized as v3 keyword")
        #expect(QASMToken.isKeyword("qreg", version: .v3), "qreg should also be recognized as v3 keyword")
        #expect(!QASMToken.isKeyword("foo", version: .v3), "arbitrary word should not be a keyword")
    }

    @Test func tabsAreSkipped() {
        let tokens = QASMLexer.tokenize("\tq", version: .v2)
        #expect(tokens[0] == .identifier("q"), "tab should be skipped, producing identifier q")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func uppercaseScientificNotation() {
        let tokens = QASMLexer.tokenize("5E2", version: .v2)
        #expect(tokens[0] == .real(5e2), "5E2 with uppercase E should be tokenized as real literal")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func spaceOnlyBlockCommentIsSkipped() {
        let tokens = QASMLexer.tokenize("/* */x", version: .v2)
        #expect(tokens[0] == .identifier("x"), "space-only block comment should be consumed leaving identifier")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func emptyUnterminatedBlockComment() {
        let tokens = QASMLexer.tokenize("/*", version: .v2)
        #expect(tokens.count == 1, "empty unterminated block comment should consume all input leaving only eof")
        #expect(tokens[0] == .eof, "empty unterminated block comment should produce eof")
    }

    @Test func unterminatedBlockCommentEndingWithAsterisk() {
        let tokens = QASMLexer.tokenize("/*abc*", version: .v2)
        #expect(tokens.count == 1, "unterminated block comment ending with * should consume all input leaving only eof")
        #expect(tokens[0] == .eof, "unterminated block comment ending with * should produce only eof")
    }

    @Test func trailingDotProducesRealToken() {
        let tokens = QASMLexer.tokenize("0.", version: .v2)
        #expect(tokens[0] == .real(0.0), "trailing dot after 0 should produce real token with value 0.0")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func leadingZerosIntegerParsesCorrectly() {
        let tokens = QASMLexer.tokenize("00123", version: .v2)
        #expect(tokens[0] == .integer(123), "leading zeros in integer should parse to value 123")
        #expect(tokens[1] == .eof, "stream should end with eof")
    }

    @Test func inputEndingWithDotAfterDigit() {
        let tokens = QASMLexer.tokenize("5.", version: .v2)
        #expect(tokens[0] == .real(5.0), "digit followed by dot at end of input should produce real token 5.0")
        #expect(tokens[1] == .eof, "stream should end with eof after trailing dot number")
    }

    @Test func dotAloneAtEndOfInput() {
        let tokens = QASMLexer.tokenize("a.", version: .v2)
        #expect(tokens[0] == .identifier("a"), "identifier before dot at end should be present")
        #expect(tokens[1] == .symbol("."), "dot at end of input with no following digit should be symbol")
        #expect(tokens[2] == .eof, "stream should end with eof")
    }
}
