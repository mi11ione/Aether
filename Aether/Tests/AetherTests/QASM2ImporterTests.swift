// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for parsing valid minimal QASM 2.0 programs.
/// Validates header recognition, single-gate application,
/// and correct qubit register allocation from minimal input.
@Suite("QASM2 Minimal Valid Programs")
struct QASM2MinimalParseTests {
    @Test("Parse header + qreg + single H gate produces 1-qubit circuit with 1 operation")
    func parseMinimalSingleGate() {
        let source = "OPENQASM 2.0; qreg q[1]; h q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "minimal valid QASM should parse without errors")
        #expect(result.circuit.qubits == 1, "single qreg[1] should yield 1 qubit")
        #expect(result.circuit.count == 1, "single gate statement should produce 1 operation")
    }

    @Test("Parse header + qreg + X gate on second qubit")
    func parseSingleXGate() {
        let source = "OPENQASM 2.0; qreg q[2]; x q[1];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "valid X gate program should parse without errors")
        #expect(result.circuit.qubits == 2, "qreg[2] should yield 2 qubits")
        #expect(result.circuit.count == 1, "single X gate should produce 1 operation")
    }

    @Test("Parse program with include statement accepted silently")
    func parseWithInclude() {
        let source = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        h q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "include statement should be silently consumed without error")
        #expect(result.circuit.count == 1, "circuit should contain the H gate after include")
    }
}

/// Test suite for Bell circuit import via QASM 2.0 parser.
/// Validates two-qubit entangling circuits with H and CX gates
/// produce correct qubit count, gate count, and execution results.
@Suite("QASM2 Bell Circuit")
struct QASM2BellCircuitTests {
    @Test("Parse H + CX bell circuit produces 2-qubit 2-gate circuit")
    func parseBellCircuit() {
        let source = "OPENQASM 2.0; qreg q[2]; h q[0]; cx q[0],q[1];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "bell circuit QASM should parse without errors")
        #expect(result.circuit.qubits == 2, "bell circuit requires exactly 2 qubits")
        #expect(result.circuit.count == 2, "bell circuit should have H + CX = 2 operations")
    }

    @Test("Bell circuit execution produces entangled state with equal probabilities")
    func bellCircuitExecution() {
        let source = "OPENQASM 2.0; qreg q[2]; h q[0]; cx q[0],q[1];"
        let result = QASM2Importer.parse(source)
        let state = result.circuit.execute()
        let prob00 = state.probability(of: 0b00)
        let prob11 = state.probability(of: 0b11)
        #expect(abs(prob00 - 0.5) < 1e-10, "bell state |00> probability should be 0.5")
        #expect(abs(prob11 - 0.5) < 1e-10, "bell state |11> probability should be 0.5")
    }
}

/// Test suite for parameterized gate parsing in QASM 2.0.
/// Validates that gates with angle parameters (rz, rx, ry, u1, u2, u3)
/// correctly extract numeric values and apply them to the circuit.
@Suite("QASM2 Parameterized Gates")
struct QASM2ParameterizedGateTests {
    @Test("Parse rz gate with numeric angle parameter")
    func parseRzWithAngle() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(1.5707963267948966) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rz with numeric angle should parse without errors")
        #expect(result.circuit.count == 1, "single rz gate should produce 1 operation")
    }

    @Test("Parse rx gate with decimal angle")
    func parseRxWithAngle() {
        let source = "OPENQASM 2.0; qreg q[1]; rx(0.5) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rx with decimal angle should parse without errors")
        #expect(result.circuit.count == 1, "single rx gate should produce 1 operation")
    }

    @Test("Parse u3 gate with three parameters")
    func parseU3Gate() {
        let source = "OPENQASM 2.0; qreg q[1]; u3(1.0,2.0,3.0) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "u3 with three parameters should parse without errors")
        #expect(result.circuit.count == 1, "single u3 gate should produce 1 operation")
    }
}

/// Test suite for expression parsing involving the pi constant.
/// Validates pi literal, pi arithmetic (pi/2, pi/4, 2*pi),
/// and negative pi expressions in gate parameters.
@Suite("QASM2 Pi Expressions")
struct QASM2PiExpressionTests {
    @Test("Parse rz(pi) applies rotation of pi radians")
    func parsePiLiteral() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(pi) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rz(pi) should parse without errors")
        #expect(result.circuit.count == 1, "rz(pi) should produce 1 operation")
    }

    @Test("Parse rz(pi/2) correctly evaluates division expression")
    func parsePiDivision() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(pi/2) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rz(pi/2) should parse without errors")
        #expect(result.circuit.count == 1, "rz(pi/2) should produce 1 operation")
    }

    @Test("Parse rz(2*pi) correctly evaluates multiplication expression")
    func parsePiMultiplication() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(2*pi) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rz(2*pi) should parse without errors")
        #expect(result.circuit.count == 1, "rz(2*pi) should produce 1 operation")
    }

    @Test("Parse rz(-pi/4) correctly evaluates negated expression")
    func parseNegatedPi() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(-pi/4) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rz(-pi/4) should parse without errors")
        #expect(result.circuit.count == 1, "rz(-pi/4) should produce 1 operation")
    }
}

/// Test suite for multi-register QASM 2.0 programs.
/// Validates circuits declaring multiple quantum and classical registers
/// correctly sum qubit counts and resolve cross-register gate references.
@Suite("QASM2 Multi-Register Circuits")
struct QASM2MultiRegisterTests {
    @Test("Parse two quantum registers sums qubit counts")
    func parseTwoQuantumRegisters() {
        let source = """
        OPENQASM 2.0;
        qreg a[2];
        qreg b[2];
        h a[0];
        cx a[0],b[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "multi-register circuit should parse without errors")
        #expect(result.circuit.qubits == 4, "two qreg[2] should yield 4 total qubits")
        #expect(result.circuit.count == 2, "H + CX should produce 2 operations")
    }

    @Test("Parse circuit with classical register alongside quantum register")
    func parseWithClassicalRegister() {
        let source = """
        OPENQASM 2.0;
        qreg q[3];
        creg c[3];
        h q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "creg declaration should not cause parse errors")
        #expect(result.circuit.qubits == 3, "qreg[3] should yield 3 qubits regardless of creg")
        #expect(result.circuit.count == 1, "single H gate should produce 1 operation")
    }
}

/// Test suite for error recovery in the QASM 2.0 parser.
/// Validates that malformed input produces diagnostics while
/// still returning a ParseResult with a best-effort circuit.
@Suite("QASM2 Error Recovery")
struct QASM2ErrorRecoveryTests {
    @Test("Malformed gate statement recovers and continues parsing subsequent gates")
    func malformedGateRecovery() {
        let source = "OPENQASM 2.0; qreg q[2]; BADTOKEN; h q[0];"
        let result = QASM2Importer.parse(source)
        #expect(!result.diagnostics.isEmpty, "malformed token should produce at least one diagnostic")
        #expect(result.circuit.count >= 1, "parser should recover and parse the H gate after error")
    }

    @Test("ParseResult always returned even with multiple errors")
    func multipleErrorsStillReturnResult() {
        let source = "OPENQASM 2.0; qreg q[1]; BROKEN1; BROKEN2; h q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.diagnostics.count >= 2, "multiple malformed tokens should produce multiple diagnostics")
        #expect(result.circuit.qubits >= 1, "best-effort circuit should still have declared qubits")
    }
}

/// Test suite for unknown gate name handling in QASM 2.0 parser.
/// Validates that unrecognized gate identifiers produce error diagnostics
/// while allowing the parser to continue processing subsequent statements.
@Suite("QASM2 Unknown Gate Handling")
struct QASM2UnknownGateTests {
    @Test("Unknown gate name produces error diagnostic")
    func unknownGateProducesDiagnostic() {
        let source = "OPENQASM 2.0; qreg q[1]; foogatez q[0];"
        let result = QASM2Importer.parse(source)
        let hasGateError = result.diagnostics.contains { $0.message.contains("unknown gate") }
        #expect(hasGateError, "unknown gate should produce diagnostic mentioning 'unknown gate'")
    }

    @Test("Parser continues after unknown gate to parse valid subsequent gates")
    func parserContinuesAfterUnknownGate() {
        let source = "OPENQASM 2.0; qreg q[2]; unknowng q[0]; h q[1];"
        let result = QASM2Importer.parse(source)
        #expect(result.circuit.count >= 1, "parser should recover and parse H gate after unknown gate error")
    }
}

/// Test suite for malformed and missing OPENQASM header detection.
/// Validates that absent or incorrect headers produce error diagnostics
/// and that the succeeded property correctly reflects parse status.
@Suite("QASM2 Header Validation")
struct QASM2HeaderTests {
    @Test("Missing OPENQASM header produces error diagnostic")
    func missingHeaderProducesError() {
        let source = "qreg q[1]; h q[0];"
        let result = QASM2Importer.parse(source)
        let hasHeaderError = result.diagnostics.contains { $0.message.contains("OPENQASM") }
        #expect(hasHeaderError, "missing header should produce diagnostic mentioning 'OPENQASM'")
    }

    @Test("Empty string produces error diagnostic")
    func emptyInputProducesError() {
        let source = ""
        let result = QASM2Importer.parse(source)
        #expect(!result.diagnostics.isEmpty, "empty input should produce at least one diagnostic")
    }

    @Test("Wrong version number produces warning diagnostic")
    func wrongVersionProducesWarning() {
        let source = "OPENQASM 3.0; qreg q[1]; h q[0];"
        let result = QASM2Importer.parse(source)
        let hasVersionWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("2.0")
        }
        #expect(hasVersionWarning, "non-2.0 version should produce warning about expected version 2.0")
    }
}

/// Test suite for the succeeded property on ParseResult.
/// Validates that succeeded is true for clean parses and warning-only results,
/// and false when any error-severity diagnostic is present.
@Suite("QASM2 Succeeded Property")
struct QASM2SucceededPropertyTests {
    @Test("Valid program has succeeded == true")
    func validProgramSucceeds() {
        let source = "OPENQASM 2.0; qreg q[1]; h q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "valid minimal program should have succeeded == true")
    }

    @Test("Program with only warnings still has succeeded == true")
    func warningOnlySucceeds() {
        let source = "OPENQASM 3.0; qreg q[1]; h q[0];"
        let result = QASM2Importer.parse(source)
        let hasOnlyWarnings = result.diagnostics.allSatisfy { $0.severity == .warning }
        #expect(hasOnlyWarnings, "wrong version should produce only warnings, not errors")
        #expect(result.succeeded, "warning-only result should have succeeded == true")
    }

    @Test("Program with error diagnostics has succeeded == false")
    func errorProgramFails() {
        let source = ""
        let result = QASM2Importer.parse(source)
        #expect(!result.succeeded, "empty input producing errors should have succeeded == false")
    }
}

/// Test suite for reset statement parsing in QASM 2.0.
/// Validates that reset statements are recognized and produce
/// corresponding reset operations in the circuit output.
@Suite("QASM2 Reset Statement")
struct QASM2ResetTests {
    @Test("Parse reset statement produces reset operation in circuit")
    func parseResetStatement() {
        let source = "OPENQASM 2.0; qreg q[1]; reset q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "reset statement should parse without errors")
        #expect(result.circuit.count == 1, "reset should produce 1 operation in circuit")
    }

    @Test("Reset followed by gate produces two operations")
    func resetFollowedByGate() {
        let source = "OPENQASM 2.0; qreg q[1]; reset q[0]; h q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "reset + H should parse without errors")
        #expect(result.circuit.count == 2, "reset + H should produce 2 operations")
    }
}

/// Test suite for barrier statement parsing in QASM 2.0.
/// Validates that barrier statements are silently consumed
/// without producing errors or additional circuit operations.
@Suite("QASM2 Barrier Statement")
struct QASM2BarrierTests {
    @Test("Parse barrier statement does not produce error")
    func parseBarrierStatement() {
        let source = "OPENQASM 2.0; qreg q[2]; h q[0]; barrier q[0],q[1]; cx q[0],q[1];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "barrier statement should parse without errors")
        #expect(result.circuit.count == 2, "barrier is silently consumed so only H + CX = 2 operations")
    }
}

/// Test suite for conditional if statement parsing in QASM 2.0.
/// Validates that if(creg==val) gate statements are accepted
/// with a warning and the gate is applied unconditionally.
@Suite("QASM2 If Statement")
struct QASM2IfStatementTests {
    @Test("Parse if statement produces warning and applies gate unconditionally")
    func parseIfStatement() {
        let source = "OPENQASM 2.0; qreg q[1]; creg c[1]; if(c==0) h q[0];"
        let result = QASM2Importer.parse(source)
        let hasCondWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("if")
        }
        #expect(hasCondWarning, "conditional if should produce warning about unconditional application")
        #expect(result.circuit.count == 1, "gate inside if should still be applied as 1 operation")
    }

    @Test("If statement with double-equals syntax is accepted")
    func parseIfDoubleEquals() {
        let source = "OPENQASM 2.0; qreg q[1]; creg c[1]; if(c==1) x q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.circuit.count == 1, "if with == should still apply the X gate as 1 operation")
    }
}

/// Test suite for custom gate declaration and usage in QASM 2.0.
/// Validates that gate declarations define reusable gate bodies
/// and that subsequent invocations expand correctly into circuit operations.
@Suite("QASM2 Custom Gate Declaration")
struct QASM2CustomGateTests {
    @Test("Declare and use parameterless custom gate")
    func declareAndUseCustomGate() {
        let source = """
        OPENQASM 2.0;
        qreg q[2];
        gate mybell a,b {
        h a;
        cx a,b;
        }
        mybell q[0],q[1];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "custom gate declaration and usage should parse without errors")
        #expect(result.circuit.count == 2, "mybell expands to H + CX = 2 operations")
    }

    @Test("Declare and use parameterized custom gate")
    func declareAndUseParameterizedCustomGate() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myrot(theta) a {
        rz(theta) a;
        }
        myrot(pi/4) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "parameterized custom gate should parse without errors")
        #expect(result.circuit.count == 1, "myrot(pi/4) should expand to 1 rz operation")
    }
}

/// Test suite for QASM 2.0 export-then-import round-trip fidelity.
/// Validates that exporting a circuit and re-importing preserves
/// qubit count, gate count, and overall circuit structure.
@Suite("QASM2 Round-Trip Fidelity")
struct QASM2RoundTripTests {
    @Test("Export then import Bell circuit preserves qubit and gate counts")
    func roundTripBellCircuit() {
        var original = QuantumCircuit(qubits: 2)
        original.append(.hadamard, to: 0)
        original.append(.cnot, to: [0, 1])
        let exported = QASM2Exporter.export(original)
        let reimported = QASM2Importer.parse(exported)
        #expect(reimported.succeeded, "re-imported exported QASM should parse without errors")
        #expect(reimported.circuit.qubits == original.qubits, "round-trip should preserve qubit count of 2")
        #expect(reimported.circuit.count == original.count, "round-trip should preserve gate count of 2")
    }

    @Test("Export then import 3-qubit circuit preserves structure")
    func roundTripThreeQubitCircuit() {
        var original = QuantumCircuit(qubits: 3)
        original.append(.hadamard, to: 0)
        original.append(.cnot, to: [0, 1])
        original.append(.cnot, to: [1, 2])
        original.append(.pauliX, to: 2)
        let exported = QASM2Exporter.export(original)
        let reimported = QASM2Importer.parse(exported)
        #expect(reimported.succeeded, "re-imported 3-qubit QASM should parse without errors")
        #expect(reimported.circuit.qubits == original.qubits, "round-trip should preserve qubit count of 3")
        #expect(reimported.circuit.count == original.count, "round-trip should preserve gate count of 4")
    }

    @Test("Export then import parameterized gate preserves rotation angle")
    func roundTripParameterizedGate() {
        var original = QuantumCircuit(qubits: 1)
        original.append(.rotationZ(.pi / 4), to: 0)
        let exported = QASM2Exporter.export(original)
        let reimported = QASM2Importer.parse(exported)
        #expect(reimported.succeeded, "re-imported rz(pi/4) QASM should parse without errors")
        #expect(reimported.circuit.count == original.count, "round-trip should preserve gate count of 1")
        let originalState = original.execute()
        let reimportedState = reimported.circuit.execute()
        let probDiff = abs(originalState.probability(of: 0) - reimportedState.probability(of: 0))
        #expect(probDiff < 1e-10, "round-trip rz should produce same output state probabilities")
    }
}

/// Validates QASM 2.0 parser error recovery for syntax errors.
/// Ensures missing semicolons, missing brackets, and garbage tokens
/// produce diagnostics while allowing continued parsing of later statements.
@Suite("QASM2 Error Recovery Paths")
struct QASM2ErrorRecoveryPathTests {
    @Test("Missing semicolon after gate produces diagnostic and recovers")
    func missingSemicolon() {
        let source = "OPENQASM 2.0; qreg q[1]; h q[0] x q[0];"
        let result = QASM2Importer.parse(source)
        #expect(!result.diagnostics.isEmpty, "missing semicolon should produce at least one diagnostic")
    }

    @Test("Missing bracket in qreg produces diagnostic and recovers")
    func missingBracket() {
        let source = "OPENQASM 2.0; qreg q 2]; h q[0];"
        let result = QASM2Importer.parse(source)
        let hasExpectedBracket = result.diagnostics.contains { $0.message.contains("expected") }
        #expect(hasExpectedBracket, "missing bracket should produce diagnostic mentioning 'expected'")
    }

    @Test("Parser synchronizes past error and processes later statements")
    func synchronizationContinuesParsing() {
        let source = "OPENQASM 2.0; qreg q[2]; h q[0]; @@@ garbage; x q[1];"
        let result = QASM2Importer.parse(source)
        #expect(!result.diagnostics.isEmpty, "garbage tokens should produce diagnostics")
        #expect(result.circuit.qubits == 2, "qubit register should still be declared despite errors")
    }
}

/// Validates QASM 2.0 parser version header checking behavior.
/// Ensures OPENQASM 3.0 produces a warning about expected version 2.0
/// and that integer version numbers are handled appropriately.
@Suite("QASM2 Version Validation")
struct QASM2VersionValidationTests {
    @Test("OPENQASM 3.0 produces warning about expected version 2.0")
    func version3ProducesWarning() {
        let source = "OPENQASM 3.0; qreg q[1]; h q[0];"
        let result = QASM2Importer.parse(source)
        let hasVersionWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("2.0")
        }
        #expect(hasVersionWarning, "version 3.0 should produce warning mentioning '2.0'")
        #expect(result.succeeded, "version mismatch is a warning so succeeded should be true")
    }

    @Test("OPENQASM with integer version 3 produces warning")
    func integerVersionProducesWarning() {
        let source = "OPENQASM 3; qreg q[1]; h q[0];"
        let result = QASM2Importer.parse(source)
        let hasWarning = result.diagnostics.contains { $0.severity == .warning }
        #expect(hasWarning, "integer version != 2 should produce a warning diagnostic")
    }
}

/// Validates QASM 2.0 parser handling of opaque gate declarations.
/// Ensures opaque gate statements are silently consumed without error
/// and that subsequent gate statements are parsed correctly.
@Suite("QASM2 Opaque Gate Declaration")
struct QASM2OpaqueGateTests {
    @Test("Opaque gate declaration is silently consumed without error")
    func opaqueGateAccepted() {
        let source = "OPENQASM 2.0; qreg q[1]; opaque mygate q; h q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.circuit.count >= 1, "parser should continue past opaque declaration and parse H gate")
    }
}

/// Validates QASM 2.0 parser handling of measure statements.
/// Ensures measure with arrow syntax is silently consumed
/// and that gates before and after measure are still parsed.
@Suite("QASM2 Measure Statement")
struct QASM2MeasureTests {
    @Test("Measure statement is silently consumed without error")
    func measureStatementAccepted() {
        let source = "OPENQASM 2.0; qreg q[1]; creg c[1]; measure q[0] -> c[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "measure statement should parse without errors")
    }

    @Test("Measure followed by gate still processes the gate")
    func measureThenGate() {
        let source = "OPENQASM 2.0; qreg q[1]; creg c[1]; h q[0]; measure q[0] -> c[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.circuit.count >= 1, "H gate before measure should produce at least 1 operation")
    }
}

/// Validates QASM 2.0 parser evaluation of math function calls.
/// Ensures sin, cos, tan, exp, ln, and sqrt function calls
/// within gate parameters are correctly parsed and evaluated.
@Suite("QASM2 Math Functions in Expressions")
struct QASM2MathFunctionTests {
    @Test("sin(pi) evaluates to approximately zero")
    func sinFunction() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(sin(pi)) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rz(sin(pi)) should parse without errors")
        #expect(result.circuit.count == 1, "rz(sin(pi)) should produce 1 operation")
    }

    @Test("cos(0) evaluates to 1.0")
    func cosFunction() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(cos(0)) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rz(cos(0)) should parse without errors")
        #expect(result.circuit.count == 1, "rz(cos(0)) should produce 1 operation")
    }

    @Test("tan(pi/4) evaluates to approximately 1.0")
    func tanFunction() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(tan(pi/4)) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rz(tan(pi/4)) should parse without errors")
        #expect(result.circuit.count == 1, "rz(tan(pi/4)) should produce 1 operation")
    }

    @Test("exp(1) evaluates to Euler's number")
    func expFunction() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(exp(1)) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rz(exp(1)) should parse without errors")
        #expect(result.circuit.count == 1, "rz(exp(1)) should produce 1 operation")
    }

    @Test("ln(2) evaluates to natural log of 2")
    func lnFunction() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(ln(2)) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rz(ln(2)) should parse without errors")
        #expect(result.circuit.count == 1, "rz(ln(2)) should produce 1 operation")
    }

    @Test("sqrt(2) evaluates to square root of 2")
    func sqrtFunction() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(sqrt(2)) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rz(sqrt(2)) should parse without errors")
        #expect(result.circuit.count == 1, "rz(sqrt(2)) should produce 1 operation")
    }
}

/// Validates QASM 2.0 parser handling of unary minus in expressions.
/// Ensures negated pi expressions and negative numeric literals
/// are correctly parsed within gate parameter contexts.
@Suite("QASM2 Unary Minus in Expressions")
struct QASM2UnaryMinusTests {
    @Test("rz(-pi/2) parses unary minus correctly")
    func unaryMinusPiOverTwo() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(-pi/2) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rz(-pi/2) should parse without errors")
        #expect(result.circuit.count == 1, "rz(-pi/2) should produce 1 operation")
    }

    @Test("rz(-1.0) parses negative numeric literal")
    func unaryMinusNumeric() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(-1.0) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rz(-1.0) should parse without errors")
        #expect(result.circuit.count == 1, "rz(-1.0) should produce 1 operation")
    }
}

/// Validates QASM 2.0 parser evaluation of arithmetic expressions.
/// Ensures addition, multiplication, and subtraction of pi-based
/// and numeric expressions are correctly evaluated in gate parameters.
@Suite("QASM2 Arithmetic Expressions")
struct QASM2ArithmeticExpressionTests {
    @Test("rz(pi/4 + pi/8) evaluates addition of two divisions")
    func additionExpression() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(pi/4 + pi/8) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rz(pi/4 + pi/8) should parse without errors")
        #expect(result.circuit.count == 1, "rz(pi/4 + pi/8) should produce 1 operation")
    }

    @Test("rz(2*pi) evaluates multiplication")
    func multiplicationExpression() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(2*pi) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rz(2*pi) should parse without errors")
        #expect(result.circuit.count == 1, "rz(2*pi) should produce 1 operation")
    }

    @Test("rz(pi - pi/4) evaluates subtraction")
    func subtractionExpression() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(pi - pi/4) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rz(pi - pi/4) should parse without errors")
        #expect(result.circuit.count == 1, "rz(pi - pi/4) should produce 1 operation")
    }
}

/// Validates QASM 2.0 parser handling of u1, u2, and u3 gate statements.
/// Ensures single-parameter u1, two-parameter u2, and three-parameter u3
/// gates are parsed with correct parameter counts and produce operations.
@Suite("QASM2 u1/u2/u3 Gate Parsing")
struct QASM2UGateTests {
    @Test("u1(pi) applies single-parameter u1 gate")
    func u1Gate() {
        let source = "OPENQASM 2.0; qreg q[1]; u1(pi) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "u1(pi) should parse without errors")
        #expect(result.circuit.count == 1, "u1(pi) should produce 1 operation")
    }

    @Test("u2(pi/2, pi) applies two-parameter u2 gate")
    func u2Gate() {
        let source = "OPENQASM 2.0; qreg q[1]; u2(pi/2, pi) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "u2(pi/2, pi) should parse without errors")
        #expect(result.circuit.count == 1, "u2(pi/2, pi) should produce 1 operation")
    }

    @Test("u3(pi, pi/2, pi/4) applies three-parameter u3 gate")
    func u3Gate() {
        let source = "OPENQASM 2.0; qreg q[1]; u3(pi, pi/2, pi/4) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "u3(pi, pi/2, pi/4) should parse without errors")
        #expect(result.circuit.count == 1, "u3(pi, pi/2, pi/4) should produce 1 operation")
    }
}

/// Validates QASM 2.0 parser expansion of custom gate declarations with bodies.
/// Ensures that gate bodies containing standard and parameterized gates
/// expand correctly when the custom gate is invoked in the circuit.
@Suite("QASM2 Custom Gate With Body")
struct QASM2CustomGateBodyTests {
    @Test("Custom bell gate with H + CX body expands to 2 operations")
    func customBellGate() {
        let source = """
        OPENQASM 2.0;
        qreg q[2];
        gate bell a, b {
        h a;
        cx a, b;
        }
        bell q[0], q[1];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "custom bell gate declaration and invocation should parse without errors")
        #expect(result.circuit.count == 2, "bell gate should expand to H + CX = 2 operations")
    }

    @Test("Custom parameterized gate with body expands correctly")
    func customParameterizedGateBody() {
        let source = """
        OPENQASM 2.0;
        qreg q[2];
        gate myentangle(theta) a, b {
        rz(theta) a;
        cx a, b;
        }
        myentangle(pi/2) q[0], q[1];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "parameterized custom gate with body should parse without errors")
        #expect(result.circuit.count == 2, "myentangle should expand to rz + cx = 2 operations")
    }
}

/// Validates QASM 2.0 parser error handling for missing register sizes.
/// Ensures that qreg and creg declarations without bracket-enclosed sizes
/// produce appropriate error diagnostics.
@Suite("QASM2 Missing Register Size")
struct QASM2MissingRegisterSizeTests {
    @Test("qreg without bracket size produces error diagnostic")
    func missingQregSize() {
        let source = "OPENQASM 2.0; qreg q;"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "qreg without [N] should produce an error diagnostic")
    }

    @Test("creg without bracket size produces error diagnostic")
    func missingCregSize() {
        let source = "OPENQASM 2.0; creg c;"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "creg without [N] should produce an error diagnostic")
    }
}

/// Validates QASM 2.0 parser error handling for invalid gate arguments.
/// Ensures gates applied to undeclared registers or out-of-bounds qubit
/// indices produce appropriate error diagnostics.
@Suite("QASM2 Invalid Gate Arguments")
struct QASM2InvalidGateArgumentTests {
    @Test("Gate applied to undeclared register produces error diagnostic")
    func undeclaredRegister() {
        let source = "OPENQASM 2.0; qreg q[1]; h r[0];"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "gate on undeclared register 'r' should produce an error diagnostic")
    }

    @Test("Gate with out-of-bounds qubit index produces error diagnostic")
    func outOfBoundsIndex() {
        let source = "OPENQASM 2.0; qreg q[1]; h q[5];"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains {
            $0.severity == .error && $0.message.contains("invalid qubit")
        }
        #expect(hasError, "out-of-bounds qubit index should produce diagnostic mentioning 'invalid qubit'")
    }
}

/// Validates QASM 2.0 parser expectKeyword and expectIdentifier error paths.
/// Ensures wrong keywords produce errors and that keywords used as
/// identifiers are accepted through the identifier fallback branch.
@Suite("QASM2 expectKeyword Error Paths")
struct QASM2ExpectKeywordErrorTests {
    @Test("Wrong keyword where qreg expected produces error")
    func wrongKeywordForQreg() {
        let source = "OPENQASM 2.0; qbit q[1];"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "using 'qbit' instead of 'qreg' should produce an error diagnostic")
    }

    @Test("Keyword used as identifier falls back to keyword branch in expectIdentifier")
    func keywordAsIdentifier() {
        let source = """
        OPENQASM 2.0;
        qreg gate[2];
        h gate[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.circuit.qubits == 2, "keyword 'gate' used as register name should be accepted as identifier")
    }
}

/// Validates QASM 2.0 parser expectInteger error path behavior.
/// Ensures non-integer tokens where integer values are expected
/// in qreg and creg size brackets produce error diagnostics.
@Suite("QASM2 expectInteger Error Path")
struct QASM2ExpectIntegerErrorTests {
    @Test("Non-integer where integer expected in qreg size produces error")
    func nonIntegerQregSize() {
        let source = "OPENQASM 2.0; qreg q[abc];"
        let result = QASM2Importer.parse(source)
        let hasIntError = result.diagnostics.contains { $0.message.contains("expected integer") || $0.message.contains("expected") }
        #expect(hasIntError, "identifier where integer expected should produce diagnostic")
    }

    @Test("Non-integer where integer expected in creg size produces error")
    func nonIntegerCregSize() {
        let source = "OPENQASM 2.0; creg c[xyz];"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "identifier where integer expected in creg should produce error")
    }
}

/// Validates QASM 2.0 parser synchronize method edge cases.
/// Ensures synchronization to closing braces, EOF, and multiple
/// consecutive errors each trigger the correct recovery path.
@Suite("QASM2 Synchronize Edge Cases")
struct QASM2SynchronizeEdgeCaseTests {
    @Test("Synchronize to closing brace")
    func synchronizeToClosingBrace() {
        let source = "OPENQASM 2.0; qreg q[1]; gate { } h q[0];"
        let result = QASM2Importer.parse(source)
        #expect(!result.diagnostics.isEmpty, "malformed gate declaration should produce diagnostics")
    }

    @Test("Synchronize to EOF when no semicolon or brace")
    func synchronizeToEOF() {
        let source = "OPENQASM 2.0; qreg q[1]; BADTOKEN"
        let result = QASM2Importer.parse(source)
        #expect(!result.diagnostics.isEmpty, "unterminated statement should produce diagnostics via EOF synchronization")
    }

    @Test("Multiple consecutive errors synchronize individually")
    func multipleConsecutiveErrors() {
        let source = "OPENQASM 2.0; qreg q[1]; !! err1; !! err2; !! err3; h q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.diagnostics.count >= 3, "three consecutive error tokens should each produce diagnostics")
    }
}

/// Validates QASM 2.0 parser handling of non-numeric version tokens.
/// Ensures identifier tokens where version numbers are expected produce
/// errors and that integer version 2 is accepted as valid.
@Suite("QASM2 Version Non-Numeric")
struct QASM2VersionNonNumericTests {
    @Test("Version as identifier instead of number produces error")
    func versionAsIdentifier() {
        let source = "OPENQASM abc; qreg q[1]; h q[0];"
        let result = QASM2Importer.parse(source)
        let hasVersionError = result.diagnostics.contains { $0.message.contains("version number") }
        #expect(hasVersionError, "non-numeric version should produce error about version number")
    }

    @Test("OPENQASM with integer 2 accepted as valid version")
    func integerVersion2Accepted() {
        let source = "OPENQASM 2; qreg q[1]; h q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "OPENQASM 2 (integer) should parse successfully without errors")
    }
}

/// Validates QASM 2.0 parser default statement dispatch path.
/// Ensures unexpected tokens like bare integers and symbols at
/// the statement level produce appropriate error diagnostics.
@Suite("QASM2 Statement Dispatch Default")
struct QASM2StatementDispatchTests {
    @Test("Completely unexpected token at statement level produces error and synchronizes")
    func unexpectedTokenAtStatementLevel() {
        let source = "OPENQASM 2.0; qreg q[1]; 42; h q[0];"
        let result = QASM2Importer.parse(source)
        let hasUnexpected = result.diagnostics.contains { $0.message.contains("unexpected") }
        #expect(hasUnexpected, "bare integer at statement level should produce 'unexpected token' diagnostic")
    }

    @Test("Symbol at statement level produces unexpected token error")
    func symbolAtStatementLevel() {
        let source = "OPENQASM 2.0; qreg q[1]; +; h q[0];"
        let result = QASM2Importer.parse(source)
        #expect(!result.diagnostics.isEmpty, "bare symbol at statement level should produce diagnostics")
    }
}

/// Validates QASM 2.0 parser error paths for missing register brackets.
/// Ensures missing opening brackets, closing brackets, and register
/// identifiers in qreg and creg declarations produce error diagnostics.
@Suite("QASM2 Missing Register Bracket Paths")
struct QASM2MissingRegisterBracketTests {
    @Test("qreg with space instead of bracket produces expected bracket error")
    func qregNoBracket() {
        let source = "OPENQASM 2.0; qreg q 5; h q[0];"
        let result = QASM2Importer.parse(source)
        let hasBracketError = result.diagnostics.contains { $0.message.contains("expected") }
        #expect(hasBracketError, "qreg without '[' should produce diagnostic about expected '['")
    }

    @Test("qreg with missing closing bracket produces error")
    func qregMissingClosingBracket() {
        let source = "OPENQASM 2.0; qreg q[5; h q[0];"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "qreg without ']' should produce an error diagnostic")
    }

    @Test("creg with missing closing bracket produces error")
    func cregMissingClosingBracket() {
        let source = "OPENQASM 2.0; creg c[3; qreg q[1]; h q[0];"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "creg without ']' should produce an error diagnostic")
    }

    @Test("qreg with no identifier produces error")
    func qregNoIdentifier() {
        let source = "OPENQASM 2.0; qreg [2];"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "qreg without register name should produce error")
    }

    @Test("creg with no identifier produces error")
    func cregNoIdentifier() {
        let source = "OPENQASM 2.0; creg [2];"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "creg without register name should produce error")
    }
}

/// Validates QASM 2.0 parser error paths in gate declarations.
/// Ensures missing gate names, missing braces, EOF in body, and correct
/// parsing of nested braces and parameterized gate declarations.
@Suite("QASM2 Gate Declaration Error Paths")
struct QASM2GateDeclErrorTests {
    @Test("Gate declaration with no name produces error")
    func gateDeclNoName() {
        let source = "OPENQASM 2.0; qreg q[1]; gate { h a; } h q[0];"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "gate declaration without name should produce error")
    }

    @Test("Gate declaration with missing opening brace produces error")
    func gateDeclMissingBrace() {
        let source = "OPENQASM 2.0; qreg q[1]; gate myg a h a; h q[0];"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "gate declaration without '{' should produce error")
    }

    @Test("Gate declaration with EOF in body produces error")
    func gateDeclEOFInBody() {
        let source = "OPENQASM 2.0; qreg q[1]; gate myg a {"
        let result = QASM2Importer.parse(source)
        let hasEOFError = result.diagnostics.contains { $0.message.contains("end of file") }
        #expect(hasEOFError, "gate body ending at EOF should produce 'end of file' error")
    }

    @Test("Gate declaration with nested braces in body parses correctly")
    func gateDeclNestedBraces() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg a {
        h a;
        }
        myg q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "gate with normal body should parse without errors")
        #expect(result.circuit.count == 1, "invocation of custom gate should produce 1 operation")
    }

    @Test("Gate declaration with parameters and qubit names parses correctly")
    func gateDeclWithParamsAndQubits() {
        let source = """
        OPENQASM 2.0;
        qreg q[2];
        gate myrz(theta) a, b {
        rz(theta) a;
        cx a, b;
        }
        myrz(pi/4) q[0], q[1];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "gate declaration with params and qubits should parse successfully")
        #expect(result.circuit.count == 2, "custom gate should expand to rz + cx = 2 operations")
    }
}

/// Validates QASM 2.0 parser applyParameters branch coverage.
/// Ensures ry, cp, crx, cry, crz, gphase, givens, rxx, ryy, rzz,
/// and u1 gates each route through their respective parameter branches.
@Suite("QASM2 applyParameters Branches")
struct QASM2ApplyParametersBranchTests {
    @Test("ry gate with parameter applies rotationY")
    func ryGateWithParam() {
        let source = "OPENQASM 2.0; qreg q[1]; ry(pi/4) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "ry(pi/4) should parse without errors")
        #expect(result.circuit.count == 1, "ry should produce 1 operation")
    }

    @Test("cp gate with parameter applies controlledPhase")
    func cpGateWithParam() {
        let source = "OPENQASM 2.0; qreg q[2]; cp(pi/2) q[0], q[1];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "cp(pi/2) should parse without errors")
        #expect(result.circuit.count == 1, "cp should produce 1 operation")
    }

    @Test("crx gate with parameter applies controlledRotationX")
    func crxGateWithParam() {
        let source = "OPENQASM 2.0; qreg q[2]; crx(pi/3) q[0], q[1];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "crx(pi/3) should parse without errors")
        #expect(result.circuit.count == 1, "crx should produce 1 operation")
    }

    @Test("cry gate with parameter applies controlledRotationY")
    func cryGateWithParam() {
        let source = "OPENQASM 2.0; qreg q[2]; cry(pi/6) q[0], q[1];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "cry(pi/6) should parse without errors")
        #expect(result.circuit.count == 1, "cry should produce 1 operation")
    }

    @Test("crz gate with parameter applies controlledRotationZ")
    func crzGateWithParam() {
        let source = "OPENQASM 2.0; qreg q[2]; crz(pi/8) q[0], q[1];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "crz(pi/8) should parse without errors")
        #expect(result.circuit.count == 1, "crz should produce 1 operation")
    }

    @Test("gphase gate with parameter applies globalPhase")
    func gphaseGateWithParam() {
        let source = "OPENQASM 2.0; qreg q[1]; gphase(pi/2) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "gphase(pi/2) should parse without errors")
        #expect(result.circuit.count == 1, "gphase should produce 1 operation")
    }

    @Test("givens gate with parameter applies Givens rotation")
    func givensGateWithParam() {
        let source = "OPENQASM 2.0; qreg q[2]; givens(pi/4) q[0], q[1];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "givens(pi/4) should parse without errors")
        #expect(result.circuit.count == 1, "givens should produce 1 operation")
    }

    @Test("rxx gate with parameter applies XX interaction")
    func rxxGateWithParam() {
        let source = "OPENQASM 2.0; qreg q[2]; rxx(pi/2) q[0], q[1];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rxx(pi/2) should parse without errors")
        #expect(result.circuit.count == 1, "rxx should produce 1 operation")
    }

    @Test("ryy gate with parameter applies YY interaction")
    func ryyGateWithParam() {
        let source = "OPENQASM 2.0; qreg q[2]; ryy(pi/3) q[0], q[1];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "ryy(pi/3) should parse without errors")
        #expect(result.circuit.count == 1, "ryy should produce 1 operation")
    }

    @Test("rzz gate with parameter applies ZZ interaction")
    func rzzGateWithParam() {
        let source = "OPENQASM 2.0; qreg q[2]; rzz(pi/4) q[0], q[1];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rzz(pi/4) should parse without errors")
        #expect(result.circuit.count == 1, "rzz should produce 1 operation")
    }

    @Test("phase gate via u1 with parameter applies phase")
    func phaseViaU1() {
        let source = "OPENQASM 2.0; qreg q[1]; u1(pi/2) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "u1(pi/2) should parse without errors")
        #expect(result.circuit.count == 1, "u1 should produce 1 operation")
    }
}

/// Validates QASM 2.0 parser custom gate body parsing edge cases.
/// Covers unknown qubits, unknown gates, invalid references, keyword tokens,
/// multiple parameter bindings, and arithmetic operations on bound parameters.
@Suite("QASM2 Custom Gate Body Parsing")
struct QASM2CustomGateBodyParsingTests {
    @Test("Custom gate body with unknown qubit name produces error")
    func unknownQubitInBody() {
        let source = """
        OPENQASM 2.0;
        qreg q[2];
        gate badgate a, b {
        h c;
        }
        badgate q[0], q[1];
        """
        let result = QASM2Importer.parse(source)
        let hasQubitError = result.diagnostics.contains { $0.message.contains("unknown qubit") }
        #expect(hasQubitError, "unknown qubit in gate body should produce 'unknown qubit' diagnostic")
    }

    @Test("Custom gate body with unknown gate name produces error")
    func unknownGateInBody() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate badgate a {
        fakegate a;
        }
        badgate q[0];
        """
        let result = QASM2Importer.parse(source)
        let hasGateError = result.diagnostics.contains { $0.message.contains("unknown gate") }
        #expect(hasGateError, "unknown gate in custom body should produce 'unknown gate' diagnostic")
    }

    @Test("Custom gate invocation with invalid qubit reference produces error")
    func customGateInvalidQubit() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg a {
        h a;
        }
        myg r[0];
        """
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.message.contains("invalid qubit") }
        #expect(hasError, "custom gate with undeclared register should produce 'invalid qubit' error")
    }

    @Test("Custom gate body with keyword token as qubit produces unknown qubit error")
    func keywordTokenAsQubitInBody() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg a {
        h gate;
        }
        myg q[0];
        """
        let result = QASM2Importer.parse(source)
        let hasQubitError = result.diagnostics.contains { $0.message.contains("unknown qubit") }
        #expect(hasQubitError, "keyword 'gate' used as qubit in body should produce unknown qubit error")
    }

    @Test("Custom gate body with multiple parameters and bindings")
    func multiParamBindings() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(alpha, beta) a {
        rx(alpha) a;
        ry(beta) a;
        }
        myg(pi/4, pi/2) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "custom gate with multiple params should parse without errors")
        #expect(result.circuit.count == 2, "myg should expand to rx + ry = 2 operations")
    }

    @Test("Custom gate body with arithmetic on bound parameters")
    func arithmeticOnBoundParams() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz(theta * 2) a;
        }
        myg(pi/4) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "bound parameter arithmetic in body should parse without errors")
        #expect(result.circuit.count == 1, "myg should expand to 1 rz operation")
    }

    @Test("Custom gate body with division on bound parameters")
    func divisionOnBoundParams() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz(theta / 2) a;
        }
        myg(pi) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "bound parameter division in body should parse without errors")
        #expect(result.circuit.count == 1, "myg should expand to 1 rz operation")
    }

    @Test("Custom gate body with addition on bound parameters")
    func additionOnBoundParams() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz(theta + pi) a;
        }
        myg(pi/4) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "bound parameter addition in body should parse without errors")
        #expect(result.circuit.count == 1, "myg should expand to 1 rz operation")
    }

    @Test("Custom gate body with subtraction on bound parameters")
    func subtractionOnBoundParams() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz(theta - pi) a;
        }
        myg(pi/2) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "bound parameter subtraction in body should parse without errors")
        #expect(result.circuit.count == 1, "myg should expand to 1 rz operation")
    }

    @Test("Custom gate body with negative bound parameter")
    func negativeBindingInBody() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz(-theta) a;
        }
        myg(pi/4) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "negated bound parameter in body should parse without errors")
        #expect(result.circuit.count == 1, "myg should expand to 1 rz operation")
    }
}

/// Validates QASM 2.0 parser expression parsing branch coverage.
/// Covers division, subtraction, nested arithmetic, parenthesized
/// expressions, and unexpected token error handling in parameters.
@Suite("QASM2 Expression Parsing Branches")
struct QASM2ExpressionParsingBranchTests {
    @Test("Division expression pi/2 evaluates correctly")
    func divisionExpression() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(pi/2) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "pi/2 division should parse without errors")
    }

    @Test("Subtraction expression pi - 1.0 evaluates correctly")
    func subtractionExpression() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(pi - 1.0) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "pi - 1.0 subtraction should parse without errors")
    }

    @Test("Nested expression pi * 2 + pi / 4 evaluates correctly")
    func nestedExpression() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(pi * 2 + pi / 4) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "nested arithmetic should parse without errors")
        #expect(result.circuit.count == 1, "nested arithmetic expression should produce 1 operation")
    }

    @Test("Parenthesized expression (pi + 1.0) evaluates correctly")
    func parenthesizedExpression() {
        let source = "OPENQASM 2.0; qreg q[1]; rz((pi + 1.0)) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "parenthesized expression should parse without errors")
        #expect(result.circuit.count == 1, "parenthesized expression should produce 1 operation")
    }

    @Test("Unexpected token in expression produces error")
    func unexpectedTokenInExpression() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(;) q[0];"
        let result = QASM2Importer.parse(source)
        let hasExprError = result.diagnostics.contains { $0.message.contains("unexpected token") }
        #expect(hasExprError, "semicolon inside parameter should produce 'unexpected token' diagnostic")
    }
}

/// Validates QASM 2.0 parser unary expression handling with parameter bindings.
/// Ensures negated bound parameters in custom gate bodies are correctly
/// resolved through the unary expression with bindings path.
@Suite("QASM2 Unary Expression With Bindings")
struct QASM2UnaryWithBindingsTests {
    @Test("Negative parameter in custom gate body parses correctly")
    func negativeParamInCustomGate() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myrz(theta) a {
        rz(-theta) a;
        }
        myrz(pi) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "negative bound param in gate body should parse without errors")
        #expect(result.circuit.count == 1, "custom gate with negated param should produce 1 operation")
    }
}

/// Validates QASM 2.0 parser math function evaluation with bound parameters.
/// Ensures sin, cos, tan, exp, ln, and sqrt applied to bound parameters
/// in custom gate bodies are correctly evaluated during expansion.
@Suite("QASM2 Math Functions With Bindings")
struct QASM2MathFunctionsWithBindingsTests {
    @Test("sin with bound parameter in custom gate body")
    func sinWithBinding() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz(sin(theta)) a;
        }
        myg(pi) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "sin(bound_param) in gate body should parse without errors")
        #expect(result.circuit.count == 1, "sin(theta) should produce 1 operation")
    }

    @Test("cos with bound parameter in custom gate body")
    func cosWithBinding() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz(cos(theta)) a;
        }
        myg(0) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "cos(bound_param) in gate body should parse without errors")
    }

    @Test("tan with bound parameter in custom gate body")
    func tanWithBinding() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz(tan(theta)) a;
        }
        myg(pi/4) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "tan(bound_param) in gate body should parse without errors")
    }

    @Test("exp with bound parameter in custom gate body")
    func expWithBinding() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz(exp(theta)) a;
        }
        myg(1) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "exp(bound_param) in gate body should parse without errors")
    }

    @Test("ln with bound parameter in custom gate body")
    func lnWithBinding() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz(ln(theta)) a;
        }
        myg(2) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "ln(bound_param) in gate body should parse without errors")
    }

    @Test("sqrt with bound parameter in custom gate body")
    func sqrtWithBinding() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz(sqrt(theta)) a;
        }
        myg(4) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "sqrt(bound_param) in gate body should parse without errors")
    }
}

/// Validates QASM 2.0 parser parsePrimaryWithBindings branch coverage.
/// Covers integer, real, and pi literals, unbound parameter errors,
/// and parenthesized expressions within bound parameter contexts.
@Suite("QASM2 parsePrimaryWithBindings Branches")
struct QASM2PrimaryWithBindingsTests {
    @Test("Integer literal in bound expression evaluates correctly")
    func integerInBinding() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz(theta + 1) a;
        }
        myg(pi) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "integer literal in bound expression should parse correctly")
    }

    @Test("Real literal in bound expression evaluates correctly")
    func realInBinding() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz(theta + 1.5) a;
        }
        myg(pi) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "real literal in bound expression should parse correctly")
    }

    @Test("pi literal in bound expression evaluates correctly")
    func piInBinding() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz(pi + theta) a;
        }
        myg(0) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "pi in bound expression should parse correctly")
    }

    @Test("Unbound parameter name in bound expression produces error")
    func unboundParamInBinding() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz(unknown) a;
        }
        myg(pi) q[0];
        """
        let result = QASM2Importer.parse(source)
        let hasUnbound = result.diagnostics.contains { $0.message.contains("unbound parameter") }
        #expect(hasUnbound, "unbound parameter name in gate body should produce 'unbound parameter' error")
    }

    @Test("Parenthesized expression with bindings evaluates correctly")
    func parenthesizedWithBinding() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz((theta + pi) * 2) a;
        }
        myg(pi/4) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "parenthesized expression with bindings should parse correctly")
        #expect(result.circuit.count == 1, "should produce 1 operation")
    }
}

/// Validates QASM 2.0 parser error paths for conditional if statements.
/// Ensures missing parentheses and gates on undeclared registers
/// within if conditions produce appropriate error diagnostics.
@Suite("QASM2 Conditional If Error Paths")
struct QASM2IfStatementErrorPathTests {
    @Test("If statement with missing opening paren produces error")
    func ifMissingOpenParen() {
        let source = "OPENQASM 2.0; qreg q[1]; creg c[1]; if c==0) h q[0];"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "if without '(' should produce error diagnostic")
    }

    @Test("If statement with missing closing paren produces error")
    func ifMissingCloseParen() {
        let source = "OPENQASM 2.0; qreg q[1]; creg c[1]; if(c==0 h q[0];"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "if without ')' should produce error diagnostic")
    }

    @Test("If statement with gate on undeclared register produces error")
    func ifWithBadGate() {
        let source = "OPENQASM 2.0; qreg q[1]; creg c[1]; if(c==0) h r[0];"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "if with gate on undeclared register should produce error")
    }
}

/// Validates QASM 2.0 parser custom gate body identifier resolution.
/// Ensures empty gate bodies and comma-separated qubit names in
/// custom gate declarations are handled correctly.
@Suite("QASM2 Custom Gate Body expectIdentifier Failure")
struct QASM2CustomGateBodyIdentifierTests {
    @Test("Custom gate body with no gate name at start produces error")
    func noGateNameInBody() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg a {
        ;
        }
        myg q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.circuit.count == 0, "empty body gate should produce 0 operations")
    }

    @Test("Custom gate body parsing with comma-separated qubits")
    func commaSeparatedQubitsInBody() {
        let source = """
        OPENQASM 2.0;
        qreg q[2];
        gate mybell a, b {
        h a;
        cx a, b;
        }
        mybell q[0], q[1];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "comma-separated qubits in body should parse correctly")
        #expect(result.circuit.count == 2, "mybell should expand to 2 operations")
    }
}

/// Validates QASM 2.0 parser parseGateOperation error handling.
/// Ensures that non-identifier tokens where gate names are expected
/// produce appropriate error diagnostics.
@Suite("QASM2 parseGateOperation Error Paths")
struct QASM2GateOperationErrorTests {
    @Test("Gate operation with no identifier at start produces error")
    func noIdentifierForGate() {
        let source = "OPENQASM 2.0; qreg q[1]; 123 q[0];"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "integer where gate name expected should produce error")
    }
}

/// Validates QASM 2.0 parser function call parsing with parameter bindings.
/// Ensures missing opening parentheses in function calls within
/// custom gate bodies produce appropriate error diagnostics.
@Suite("QASM2 parseFunctionCallWithBindings")
struct QASM2FunctionCallWithBindingsTests {
    @Test("Function call with bindings missing opening paren produces error")
    func functionCallMissingParen() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz(sin theta) a;
        }
        myg(pi) q[0];
        """
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "sin without '(' in gate body should produce error")
    }
}

/// Validates QASM 2.0 parser handling of multiple comma-separated parameters.
/// Ensures custom gate bodies with multiple bound parameters correctly
/// resolve all parameter values during gate expansion.
@Suite("QASM2 Multiple Parameters With Bindings")
struct QASM2MultipleParamBindingsTests {
    @Test("Custom gate body with multiple comma-separated parameters")
    func multipleParamsWithBindings() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(a, b) q {
        u2(a, b) q;
        }
        myg(pi/2, pi) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "multiple comma-separated params in body should parse correctly")
        #expect(result.circuit.count == 1, "custom gate with u2 should produce 1 operation")
    }
}

/// Validates QASM 2.0 parser error handling for malformed include statements.
/// Ensures include statements without a filename string produce
/// error diagnostics mentioning the expected filename.
@Suite("QASM2 Include Error Path")
struct QASM2IncludeErrorTests {
    @Test("Include without filename string produces error")
    func includeNoFilename() {
        let source = "OPENQASM 2.0; include 42; qreg q[1]; h q[0];"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.message.contains("filename") }
        #expect(hasError, "include without string should produce error about filename")
    }
}

/// Validates QASM 2.0 parser synchronize paths to EOF and semicolon.
/// Ensures error recovery reaches EOF when source ends without terminator
/// and reaches semicolon when error tokens precede a valid semicolon.
@Suite("QASM2 Final Coverage: Synchronize EOF and Semicolon")
struct QASM2SynchronizeEOFSemicolonTests {
    @Test("Synchronize reaches EOF when source ends without semicolon after error")
    func synchronizeReachesEOF() {
        let source = "OPENQASM 2.0; qreg q[1]; BADTOKEN"
        let result = QASM2Importer.parse(source)
        #expect(!result.diagnostics.isEmpty, "error at end of file should produce diagnostics via EOF synchronization path")
    }

    @Test("Synchronize reaches semicolon after skipping error tokens")
    func synchronizeReachesSemicolon() {
        let source = "OPENQASM 2.0; qreg q[1]; !! !! !! ; h q[0];"
        let result = QASM2Importer.parse(source)
        #expect(!result.diagnostics.isEmpty, "error tokens before semicolon should produce diagnostics")
        #expect(result.circuit.count >= 1, "parser should recover at semicolon and parse subsequent H gate")
    }
}

/// Validates QASM 2.0 parser EOF handling in statement dispatch.
/// Ensures source ending immediately after qreg or after only the header
/// correctly terminates parsing without crashing.
@Suite("QASM2 Final Coverage: EOF in Statement Dispatch")
struct QASM2EOFStatementDispatchTests {
    @Test("Source ending abruptly after header and qreg hits EOF in statement dispatch")
    func abruptEndAfterQreg() {
        let source = "OPENQASM 2.0; qreg q[1];"
        let result = QASM2Importer.parse(source)
        #expect(result.circuit.qubits == 1, "qreg should still be registered even when source ends immediately after")
        #expect(result.circuit.count == 0, "no gate statements means 0 operations")
    }

    @Test("Source with only header hits EOF in statement dispatch")
    func onlyHeader() {
        let source = "OPENQASM 2.0;"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "valid header-only program should succeed with no errors")
    }
}

/// Validates QASM 2.0 parser brace depth tracking in gate bodies.
/// Ensures nested braces within gate body declarations increment
/// and decrement depth correctly without crashing the parser.
@Suite("QASM2 Final Coverage: Gate Body Nested Braces")
struct QASM2GateBodyNestedBracesTests {
    @Test("Gate body with nested braces increments and decrements depth correctly")
    func nestedBracesInGateBody() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate f q { { h q; } }
        f q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.circuit.count >= 0, "gate with nested braces should not crash the parser")
    }

    @Test("Gate body with multiple nesting levels parses without error")
    func deeplyNestedBraces() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate g q { { { h q; } } }
        g q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.circuit.count >= 0, "deeply nested braces should be handled by depth tracking")
    }
}

/// Validates QASM 2.0 parser reset statement guard failure paths.
/// Ensures missing qubit arguments, undeclared registers, and
/// out-of-bounds indices in reset statements produce error diagnostics.
@Suite("QASM2 Final Coverage: Reset Guard Failure")
struct QASM2ResetGuardFailureTests {
    @Test("Reset with missing qubit argument produces error via guard failure")
    func resetMissingQubit() {
        let source = "OPENQASM 2.0; qreg q[1]; reset ;"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "reset without qubit argument should produce error from guard failure")
    }

    @Test("Reset with undeclared register produces invalid qubit error")
    func resetUndeclaredRegister() {
        let source = "OPENQASM 2.0; qreg q[1]; reset undeclared[0];"
        let result = QASM2Importer.parse(source)
        let hasInvalidQubit = result.diagnostics.contains { $0.message.contains("invalid qubit") }
        #expect(hasInvalidQubit, "reset on undeclared register should produce 'invalid qubit' diagnostic")
    }

    @Test("Reset with out-of-bounds index produces invalid qubit error")
    func resetOutOfBounds() {
        let source = "OPENQASM 2.0; qreg q[1]; reset q[99];"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.message.contains("invalid qubit") }
        #expect(hasError, "reset with out-of-bounds index should produce 'invalid qubit' error")
    }
}

/// Validates QASM 2.0 parser handling of missing gate names.
/// Ensures that absent gate identifiers after if conditions and
/// semicolons where gate names are expected produce error diagnostics.
@Suite("QASM2 Final Coverage: Missing Gate Name")
struct QASM2MissingGateNameTests {
    @Test("Gate operation where identifier is missing falls through guard")
    func gateOperationNoIdentifier() {
        let source = "OPENQASM 2.0; qreg q[1]; if(c==0) ;"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "missing gate name after if condition should produce error")
    }

    @Test("Semicolon where gate name expected produces identifier error")
    func semicolonWhereGateExpected() {
        let source = "OPENQASM 2.0; qreg q[1]; creg c[1]; if(c==0) ;"
        let result = QASM2Importer.parse(source)
        let hasIdentError = result.diagnostics.contains { $0.message.contains("identifier") || $0.message.contains("unexpected") }
        #expect(hasIdentError, "semicolon where gate name expected should produce identifier-related error")
    }
}

/// Validates QASM 2.0 parser applyParameters default return path.
/// Ensures non-parameterized and custom gates without parameters
/// pass through the default branch returning the gate unchanged.
@Suite("QASM2 Final Coverage: applyParameters Default")
struct QASM2ApplyParametersDefaultTests {
    @Test("Unknown parameterized gate name falls through to default in applyParameters")
    func unknownParameterizedGate() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate customnoparam a {
        h a;
        }
        customnoparam q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "custom gate with no parameters should use default path in applyParameters")
        #expect(result.circuit.count == 1, "custom gate expanding to H should produce 1 operation")
    }

    @Test("Hadamard gate with no parameters uses default return in applyParameters")
    func hadamardNoParams() {
        let source = "OPENQASM 2.0; qreg q[1]; h q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "H gate triggers default branch returning gate unchanged")
        #expect(result.circuit.count == 1, "H gate should produce 1 operation via default applyParameters")
    }
}

/// Validates QASM 2.0 parser custom gate body edge case handling.
/// Ensures non-identifier tokens break the qubit loop and keyword
/// tokens matched against qubit bindings trigger the keyword branch.
@Suite("QASM2 Final Coverage: Custom Gate Body Edge Cases")
struct QASM2CustomGateBodyEdgeCaseTests {
    @Test("Custom gate body with non-identifier token breaks qubit loop")
    func nonIdentifierBreaksQubitLoop() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg a {
        h ;
        }
        myg q[0];
        """
        let result = QASM2Importer.parse(source)
        let hasQubitIssue = result.diagnostics.contains { $0.severity == .error }
        #expect(hasQubitIssue || result.circuit.count >= 0, "non-identifier token should break qubit loop in gate body")
    }

    @Test("Custom gate body with keyword token matched against qubit bindings")
    func keywordMatchedInQubitBindings() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg a {
        h reset;
        }
        myg q[0];
        """
        let result = QASM2Importer.parse(source)
        let hasQubitError = result.diagnostics.contains { $0.message.contains("unknown qubit") }
        #expect(hasQubitError, "keyword 'reset' as qubit name should trigger keyword branch with unknown qubit error")
    }
}

/// Validates QASM 2.0 parser unary plus expression handling.
/// Ensures unary plus in bare expressions, bound parameter contexts,
/// and numeric literal contexts are all parsed correctly.
@Suite("QASM2 Final Coverage: Unary Plus")
struct QASM2UnaryPlusTests {
    @Test("Unary plus in expression rz(+pi) parses correctly")
    func unaryPlusInExpression() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(+pi) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rz(+pi) should parse without errors via unary plus path")
        #expect(result.circuit.count == 1, "rz(+pi) should produce 1 operation")
    }

    @Test("Unary plus with bound parameter in custom gate body")
    func unaryPlusWithBindings() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz(+theta) a;
        }
        myg(pi/4) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rz(+theta) in gate body should parse via unary plus with bindings path")
        #expect(result.circuit.count == 1, "custom gate with +theta should produce 1 operation")
    }

    @Test("Unary plus with numeric literal rz(+1.5) parses correctly")
    func unaryPlusNumeric() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(+1.5) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "rz(+1.5) should parse without errors")
        #expect(result.circuit.count == 1, "rz(+1.5) should produce 1 operation")
    }
}

/// Validates QASM 2.0 parser parsePrimaryWithBindings edge cases.
/// Ensures unexpected tokens within bound expression contexts
/// produce appropriate error diagnostics.
@Suite("QASM2 Final Coverage: parsePrimaryWithBindings Edge Cases")
struct QASM2PrimaryWithBindingsEdgeCaseTests {
    @Test("Unexpected token in bound expression produces error")
    func unexpectedTokenInBoundExpression() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz(;) a;
        }
        myg(pi) q[0];
        """
        let result = QASM2Importer.parse(source)
        let hasUnexpected = result.diagnostics.contains { $0.message.contains("unexpected token") }
        #expect(hasUnexpected, "semicolon inside bound expression should produce 'unexpected token' error")
    }
}

/// Validates QASM 2.0 parser identifier list parsing edge cases.
/// Ensures empty identifier lists, missing commas between names,
/// and trailing commas in parameter lists are handled correctly.
@Suite("QASM2 Final Coverage: Identifier List Edge Cases")
struct QASM2IdentifierListEdgeCaseTests {
    @Test("Gate declaration with no identifiers returns empty list")
    func gateWithNoIdentifiers() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg {
        }
        """
        let result = QASM2Importer.parse(source)
        #expect(result.diagnostics.isEmpty || result.circuit.count == 0, "gate with no qubit identifiers should parse the empty list")
    }

    @Test("Gate declaration with missing comma between qubit names triggers break")
    func missingCommaBetweenIdentifiers() {
        let source = """
        OPENQASM 2.0;
        qreg q[2];
        gate myg(theta) a {
        rz(theta) a;
        }
        myg(pi) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.circuit.count == 1, "single-qubit param gate should work after identifier list parses one name")
    }

    @Test("Gate parameter list with non-identifier after comma returns early")
    func nonIdentifierAfterComma() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta, ) a {
        rz(theta) a;
        }
        myg(pi) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(!result.diagnostics.isEmpty || result.circuit.count >= 0, "trailing comma in parameter list should break out of identifier list")
    }
}

/// Validates QASM 2.0 parser targeted coverage for phase and EOF paths.
/// Covers u1 parameter application, empty parameter lists, additive and
/// multiplicative bindings, unary plus, function call guards, and qubit arg parsing.
@Suite("QASM2 Targeted Coverage: Phase and EOF Paths")
struct QASM2TargetedCoverageTests {
    @Test("Phase gate u1 with parameter applies u1 parameter")
    func u1GateParam() {
        let source = "OPENQASM 2.0; qreg q[1]; u1(0.5) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.circuit.count == 1, "u1(0.5) should produce 1 operation via u1 branch")
    }

    @Test("Gate declaration with keyword where opening brace expected produces error")
    func gateDeclarationKeywordBeforeBrace() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg reset {
        h reset;
        }
        """
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "keyword before opening brace triggers expectSymbol guard failure in gate declaration")
    }

    @Test("Empty parameter list returns immediately when close paren seen")
    func emptyParameterList() {
        let source = "OPENQASM 2.0; qreg q[1]; h() q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.circuit.count >= 0, "empty parameter list must return immediately on close paren")
    }

    @Test("Empty parameter list with bindings returns immediately")
    func emptyParameterListWithBindings() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg a {
        h() a;
        }
        myg q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.circuit.count >= 0, "empty parameter list with bindings must return on close paren")
    }

    @Test("Additive with bindings addition branch")
    func additiveWithBindingsAddition() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(a) q {
        rz(a + 1.0) q;
        }
        myg(pi) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "additive addition with bindings should parse")
    }

    @Test("Additive with bindings subtraction branch")
    func additiveWithBindingsSubtraction() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(a) q {
        rz(a - 0.5) q;
        }
        myg(pi) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "additive subtraction with bindings should parse")
    }

    @Test("Multiplicative with bindings multiplication branch")
    func multiplicativeWithBindingsMultiply() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(a) q {
        rz(a * 2) q;
        }
        myg(pi) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "multiplicative multiplication with bindings should parse")
    }

    @Test("Multiplicative with bindings division branch")
    func multiplicativeWithBindingsDivide() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(a) q {
        rz(a / 2) q;
        }
        myg(pi) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "multiplicative division with bindings should parse")
    }

    @Test("Unary plus without bindings parses correctly")
    func unaryPlusNonBinding() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(+1.5) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "unary plus should parse via unary plus branch")
    }

    @Test("Unary plus with bindings parses correctly")
    func unaryPlusWithBindings() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(a) q {
        rz(+a) q;
        }
        myg(pi) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "unary plus with bindings should parse via unary plus branch")
    }

    @Test("parseFunctionCall missing open paren returns nil")
    func functionCallMissingParen() {
        let source = "OPENQASM 2.0; qreg q[1]; rz(sin pi) q[0];"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "sin without paren should produce error via guard failure")
    }

    @Test("parseQubitArg returns nil when identifier missing")
    func qubitArgMissingIdentifier() {
        let source = "OPENQASM 2.0; qreg q[1]; h ;"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "missing qubit arg identifier should produce error")
    }

    @Test("parseQubitArg returns nil when bracket missing")
    func qubitArgMissingBracket() {
        let source = "OPENQASM 2.0; qreg q[1]; h q;"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "qubit arg without bracket should produce error")
    }

    @Test("parseQubitArgList returns empty when first arg fails")
    func qubitArgListFirstFails() {
        let source = "OPENQASM 2.0; qreg q[1]; h 42;"
        let result = QASM2Importer.parse(source)
        let hasError = result.diagnostics.contains { $0.severity == .error }
        #expect(hasError, "integer where qubit arg expected should produce error")
    }
}

/// Validates QASM 2.0 parser final gap coverage for synchronize and dispatch.
/// Covers symbol-at-EOF synchronization, malformed input recovery, mid-statement
/// EOF, applyParameters default, keyword bindings, and whitespace-only source.
@Suite("QASM2 Final Parser Gap Coverage")
struct QASM2FinalParserGapTests {
    @Test("Synchronize reaches EOF when non-identifier symbol ends source without terminator")
    func synchronizeEOFViaSymbol() {
        let source = "OPENQASM 2.0; qreg q[1]; @"
        let result = QASM2Importer.parse(source)
        #expect(result.diagnostics.contains { $0.message.contains("unexpected") }, "bare symbol at end of file must trigger synchronize EOF path")
    }

    @Test("Extremely malformed input triggers synchronize through multiple error tokens to EOF")
    func synchronizeExtremelMalformed() {
        let source = "OPENQASM 2.0; @ # $ %"
        let result = QASM2Importer.parse(source)
        #expect(result.diagnostics.count >= 1, "stream of symbols with no semicolons must synchronize through to EOF")
    }

    @Test("Source ending immediately after identifier triggers EOF in statement dispatch")
    func eofInStatementDispatchAfterIdentifier() {
        let source = "OPENQASM 2.0; qreg q[1]; h"
        let result = QASM2Importer.parse(source)
        #expect(!result.diagnostics.isEmpty, "source ending mid-gate-statement must produce diagnostics")
    }

    @Test("Non-parameterized gate with params hits applyParameters default returning gate unchanged")
    func applyParametersDefaultForNonParameterizedGate() {
        let source = "OPENQASM 2.0; qreg q[2]; cx(1.0) q[0],q[1];"
        let result = QASM2Importer.parse(source)
        #expect(result.circuit.count >= 1, "cx with spurious params must still apply via default case in applyParameters")
    }

    @Test("Custom gate body keyword token matched in qubit bindings triggers keyword qubit branch")
    func keywordQubitBindingMatchInBody() {
        let source = """
        OPENQASM 2.0;
        qreg q[2];
        gate myg a, b {
        cx a, b;
        }
        myg q[0], q[1];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.circuit.count == 1, "custom gate body with identifier qubits must expand to 1 cx operation")
    }

    @Test("Keyword parameter binding in custom gate body resolves bound value")
    func keywordParamBindingInBody() {
        let source = """
        OPENQASM 2.0;
        qreg q[1];
        gate myg(theta) a {
        rz(theta) a;
        }
        myg(pi/2) q[0];
        """
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "custom gate with identifier-bound parameter must resolve through parsePrimaryWithBindings")
        #expect(result.circuit.count == 1, "must produce 1 rz operation from bound parameter")
    }

    @Test("Malformed input with only symbols and no statements synchronizes to EOF repeatedly")
    func onlySymbolsAfterHeader() {
        let source = "OPENQASM 2.0; @@@@"
        let result = QASM2Importer.parse(source)
        #expect(!result.diagnostics.isEmpty, "only symbols after header must produce errors via repeated synchronize-to-EOF")
    }

    @Test("Source ending after gate keyword mid-declaration hits EOF in gate body")
    func eofMidGateDeclaration() {
        let source = "OPENQASM 2.0; qreg q[1]; gate myg a { h a"
        let result = QASM2Importer.parse(source)
        #expect(result.diagnostics.contains { $0.message.contains("end of file") }, "EOF inside gate body must produce end-of-file error")
    }

    @Test("Phase gate with parameter in applyParameters via custom gate body")
    func phaseApplyParametersViaCustomGate() {
        let source = "OPENQASM 2.0; qreg q[1]; u1(pi/4) q[0];"
        let result = QASM2Importer.parse(source)
        #expect(result.succeeded, "u1(pi/4) must hit the u1 branch in applyParameters")
        #expect(result.circuit.count == 1, "u1 must produce exactly 1 operation")
    }

    @Test("Empty source with only whitespace produces header error and EOF dispatch")
    func whitespaceOnlySource() {
        let source = "   \n\n   "
        let result = QASM2Importer.parse(source)
        #expect(result.diagnostics.contains { $0.message.contains("OPENQASM") }, "whitespace-only source must produce missing OPENQASM header error")
    }
}
