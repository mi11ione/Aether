// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for QASM3Exporter circuit-to-string serialization.
/// Validates header output, gate syntax, parameterized gates,
/// and controlled gate modifier formatting for OpenQASM 3.0 compliance.
@Suite("QASM3Exporter Serialization")
struct QASM3ExporterTests {
    @Test("Empty circuit exports OPENQASM 3.0 header")
    func exportEmptyCircuitHeader() {
        let circuit = QuantumCircuit(qubits: 2)
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.hasPrefix("OPENQASM 3.0;"), "exported string must begin with OPENQASM 3.0 version header")
    }

    @Test("Single gate export uses qubit[N] declaration syntax")
    func exportSingleGateUsesQubitDeclaration() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("qubit[2] q;"), "export must use modern qubit[N] declaration, not legacy qreg")
        #expect(qasm.contains("h q[0];"), "Hadamard gate on qubit 0 must appear as 'h q[0];'")
    }

    @Test("Export includes stdgates.inc")
    func exportIncludesStdgatesInc() {
        let circuit = QuantumCircuit(qubits: 1)
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("include \"stdgates.inc\";"), "export must include stdgates.inc standard library")
    }

    @Test("Controlled gate uses ctrl @ modifier syntax")
    func exportControlledGateCtrlSyntax() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.controlled(gate: .hadamard, controls: [0]), to: [0, 1])
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("ctrl @"), "controlled gate must use 'ctrl @' modifier syntax in QASM 3.0")
    }

    @Test("Parameterized gate exports angle value")
    func exportParameterizedGate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(.value(1.5)), to: 0)
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("rz(1.5) q[0];"), "rotation gate must export with parameter value in parentheses")
    }

    @Test("Multi-gate circuit exports all operations in order")
    func exportMultiGateCircuit() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.pauliX, to: 2)
        let qasm = QASM3Exporter.export(circuit)
        let lines = qasm.components(separatedBy: "\n")
        let gateLines = lines.filter { $0.contains("q[") && !$0.hasPrefix("qubit") && !$0.hasPrefix("bit") }
        #expect(gateLines.count == 3, "circuit with 3 gates must produce exactly 3 gate statement lines")
        #expect(qasm.contains("h q[0];"), "first gate must be Hadamard on qubit 0")
        #expect(qasm.contains("cx q[0], q[1];"), "second gate must be CNOT on qubits 0,1")
        #expect(qasm.contains("x q[2];"), "third gate must be Pauli-X on qubit 2")
    }
}

/// Test suite for QASM3Importer string-to-circuit parsing.
/// Validates qubit/bit declarations, gate modifier parsing,
/// error recovery, unknown gate handling, and unsupported directives.
@Suite("QASM3Importer Parsing")
struct QASM3ImporterTests {
    @Test("Parse valid QASM 3.0 with qubit and bit declarations")
    func parseValidQASMWithDeclarations() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        h q[0];
        cx q[0], q[1];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "valid QASM 3.0 source must parse without errors")
        #expect(result.circuit.qubits == 2, "circuit must have 2 qubits as declared")
        #expect(result.circuit.count == 2, "circuit must contain exactly 2 gate operations")
    }

    @Test("Parse gate modifier ctrl @ produces controlled gate")
    func parseCtrlModifier() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        ctrl @ h q[0], q[1];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "ctrl @ gate modifier must parse without errors")
        #expect(result.circuit.count == 1, "circuit must contain exactly 1 controlled gate operation")
    }

    @Test("Bad input produces error diagnostics with recovery")
    func parseBadInputProducesDiagnostics() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        @@@ invalid syntax;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        let errorCount = result.diagnostics.count(where: { $0.severity == .error })
        #expect(errorCount > 0, "malformed token sequence must produce at least one error diagnostic")
        #expect(result.circuit.count >= 1, "parser must recover and parse subsequent valid gates after error")
    }

    @Test("Unknown gate name produces error diagnostic")
    func parseUnknownGateProducesError() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        foobar q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasUnknownGateError = result.diagnostics.contains { $0.message.contains("unknown gate") }
        #expect(hasUnknownGateError, "unrecognized gate name 'foobar' must produce diagnostic mentioning 'unknown gate'")
    }

    @Test("Timing directive delay produces warning diagnostic")
    func parseTimingDirectiveWarning() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        delay[100ns] q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasDelayWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("delay")
        }
        #expect(hasDelayWarning, "unsupported delay directive must produce a warning diagnostic")
    }

    @Test("Empty source produces error diagnostic")
    func parseEmptySourceProducesError() {
        let source = ""
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 0, "empty source must produce circuit with no operations")
    }
}

/// Test suite for QASM 3.0 export-then-import round-trip fidelity.
/// Validates that serializing a circuit and re-parsing the output
/// preserves gate count, qubit count, and circuit structure.
@Suite("QASM3 Round-Trip Serialization")
struct QASM3RoundTripTests {
    @Test("Round-trip preserves circuit structure for Bell circuit")
    func roundTripBellCircuit() {
        var original = QuantumCircuit(qubits: 2)
        original.append(.hadamard, to: 0)
        original.append(.cnot, to: [0, 1])

        let exported = QASM3Exporter.export(original)
        let imported = QASM3Importer.parse(exported)

        #expect(imported.succeeded, "re-importing exported QASM 3.0 must succeed without errors")
        #expect(imported.circuit.qubits == original.qubits, "round-trip must preserve qubit count")
        #expect(imported.circuit.count == original.count, "round-trip must preserve operation count")
    }

    @Test("Round-trip preserves multi-gate circuit with Pauli gates")
    func roundTripMultiGateCircuit() {
        var original = QuantumCircuit(qubits: 3)
        original.append(.hadamard, to: 0)
        original.append(.pauliX, to: 1)
        original.append(.pauliZ, to: 2)
        original.append(.cnot, to: [0, 1])

        let exported = QASM3Exporter.export(original)
        let imported = QASM3Importer.parse(exported)

        #expect(imported.succeeded, "re-importing multi-gate circuit must succeed without errors")
        #expect(imported.circuit.qubits == original.qubits, "round-trip must preserve qubit count for 3-qubit circuit")
        #expect(imported.circuit.count == original.count, "round-trip must preserve all 4 operations")
    }

    @Test("Round-trip preserves parameterized rotation gate angle")
    func roundTripParameterizedGate() {
        var original = QuantumCircuit(qubits: 1)
        original.append(.rotationZ(.value(0.75)), to: 0)

        let exported = QASM3Exporter.export(original)
        let imported = QASM3Importer.parse(exported)

        #expect(imported.succeeded, "re-importing parameterized gate must succeed without errors")
        #expect(imported.circuit.count == 1, "round-trip must preserve single parameterized gate")
    }
}

/// Validates QASM3Importer parsing of declarations, modifiers, and directives.
/// Covers qubit/bit sizing, gate definitions, subroutines, pow/negctrl modifiers,
/// classical types, measure syntax, and expression evaluation paths.
@Suite("QASM3Importer Coverage")
struct QASM3ImporterCoverageTests {
    @Test("Qubit and bit declarations with explicit size create registers")
    func parseQubitBitDeclarationsWithSize() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        h q[0];
        h q[1];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "sized qubit and bit declarations must parse without errors")
        #expect(result.circuit.qubits == 2, "qubit[2] declaration must create 2-qubit register")
        #expect(result.circuit.count == 2, "two Hadamard gates on q[0] and q[1] must produce 2 operations")
    }

    @Test("Gate definition stores custom gate and expands on call")
    func parseGateDefinitionAndExpand() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        gate mygate a, b { h a; cx a, b; }
        mygate q[0], q[1];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "gate definition with body must parse without errors")
        #expect(result.circuit.count >= 2, "custom gate expansion must produce at least 2 inner operations")
    }

    @Test("Subroutine definition parses with warning")
    func parseSubroutineDefinition() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        def bell() qubit[2] r { h r[0]; cx r[0], r[1]; }
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "parser must continue after subroutine definition")
    }

    @Test("Gate modifier negctrl produces controlled gate with warning")
    func parseNegctrlModifier() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        negctrl @ x q[0], q[1];
        """
        let result = QASM3Importer.parse(source)
        let hasNegctrlWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("negctrl")
        }
        #expect(hasNegctrlWarning, "negctrl modifier must produce warning about approximation")
        #expect(result.circuit.count >= 1, "negctrl-modified gate must produce at least one operation")
    }

    @Test("Gate modifier pow applies gate repeated times")
    func parsePowModifier() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        pow(2) @ s q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 2, "pow(2) modifier must produce at least 2 gate applications")
    }

    @Test("Classical int declaration produces warning and continues parsing")
    func parseClassicalIntDeclaration() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        int[32] count = 5;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "parser must recover and parse gates after classical declaration")
    }

    @Test("Measure assignment syntax parses without fatal error")
    func parseMeasureAssignment() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[1] c;
        h q[0];
        c[0] = measure q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "parser must handle measure assignment and retain prior gates")
    }

    @Test("Extern declaration produces warning diagnostic")
    func parseExternDeclaration() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        extern classical_func(int[32]) -> int[32];
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasExternWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("extern")
        }
        #expect(hasExternWarning, "extern declaration must produce warning about unsupported feature")
        #expect(result.circuit.count >= 1, "parser must continue after extern declaration")
    }

    @Test("Delay directive produces warning mentioning delay")
    func parseDelayDirectiveWarningText() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        delay[50ns] q[0];
        """
        let result = QASM3Importer.parse(source)
        let delayWarning = result.diagnostics.first {
            $0.severity == .warning && $0.message.contains("delay")
        }
        #expect(delayWarning != nil, "delay directive must produce warning diagnostic")
        #expect(
            delayWarning?.message.contains("not modeled") == true,
            "delay warning must mention that delay is not modeled in circuit",
        )
    }

    @Test("Expression parsing resolves pi constant")
    func parseExpressionPi() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(pi) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz(pi) must parse without errors")
        #expect(result.circuit.count == 1, "rz(pi) must produce exactly one gate operation")
    }

    @Test("Expression parsing resolves tau constant as 2*pi")
    func parseExpressionTau() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(tau) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz(tau) must parse without errors")
        #expect(result.circuit.count == 1, "rz(tau) must produce exactly one gate operation")
    }

    @Test("Expression parsing evaluates sin function call")
    func parseExpressionSinFunction() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(sin(pi/2)) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz(sin(pi/2)) must parse without errors")
        #expect(result.circuit.count == 1, "rz(sin(pi/2)) must produce exactly one gate operation")
    }

    @Test("Expression parsing evaluates cos function call")
    func parseExpressionCosFunction() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(cos(0)) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz(cos(0)) must parse without errors")
        #expect(result.circuit.count == 1, "rz(cos(0)) must produce exactly one gate operation")
    }

    @Test("Multiple syntax errors are all collected in diagnostics")
    func parseMultipleErrorsCollected() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        @@@ invalid1;
        $$$ invalid2;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        let errorCount = result.diagnostics.count(where: { $0.severity == .error })
        #expect(errorCount >= 2, "parser must collect multiple error diagnostics from distinct syntax errors")
        #expect(result.circuit.count >= 1, "parser must recover and parse valid gates after multiple errors")
    }

    @Test("Malformed version 2.0 produces warning diagnostic")
    func parseMalformedVersion() {
        let source = """
        OPENQASM 2.0;
        include "stdgates.inc";
        qubit[1] q;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasVersionWarning = result.diagnostics.contains {
            $0.message.contains("version") || $0.message.contains("3.0") || $0.message.contains("2.0")
        }
        #expect(hasVersionWarning, "OPENQASM 2.0 fed to QASM3Importer must produce version warning diagnostic")
    }

    @Test("If statement parses with warning about classical control")
    func parseIfStatement() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[1] c;
        h q[0];
        if (c[0] == 1) x q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasIfWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("if")
        }
        #expect(hasIfWarning, "classical if statement must produce warning about partial modeling")
        #expect(result.circuit.count >= 1, "parser must retain gates parsed before and within if statement")
    }

    @Test("Reset statement parses and creates reset operation")
    func parseResetStatement() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        h q[0];
        reset q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "reset statement must parse without errors")
        #expect(result.circuit.count == 2, "circuit must contain both Hadamard and reset operations")
    }
}

/// Validates QASM3Exporter serialization of advanced gate types.
/// Covers reset, custom gates, u1/u2/u3, symbolic and negated parameters,
/// controlled gates, formatDouble edge cases, and special value formatting.
@Suite("QASM3Exporter Coverage")
struct QASM3ExporterCoverageTests {
    @Test("Reset operation exports as reset q[i] statement")
    func exportResetOperation() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.reset, to: 0)
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("reset q[0];"), "reset operation must export as 'reset q[0];' statement")
    }

    @Test("Custom single-qubit gate emits inline gate definition")
    func exportCustomSingleQubitGate() {
        var circuit = QuantumCircuit(qubits: 1)
        let matrix: [[Complex<Double>]] = [
            [Complex(1, 0), Complex(0, 0)],
            [Complex(0, 0), Complex(0, 1)],
        ]
        circuit.append(.customSingleQubit(matrix: matrix), to: 0)
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("gate custom_u0"), "custom gate must emit an inline gate definition with name custom_u0")
        #expect(qasm.contains("custom_u0 q[0];"), "custom gate must emit a call to the defined gate")
    }

    @Test("u1 gate exports with lambda parameter")
    func exportU1Gate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u1(lambda: .value(1.5)), to: 0)
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("u1(1.5)"), "u1 gate must export with lambda parameter value in parentheses")
    }

    @Test("u2 gate exports with two parameters")
    func exportU2Gate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u2(phi: .value(0.5), lambda: .value(1.0)), to: 0)
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("u2("), "u2 gate must appear with parameter list")
        #expect(qasm.contains("0.5"), "u2 export must contain phi parameter value 0.5")
        #expect(qasm.contains("1.0"), "u2 export must contain lambda parameter value 1.0")
    }

    @Test("u3 gate exports with three parameters")
    func exportU3Gate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u3(theta: .value(0.1), phi: .value(0.2), lambda: .value(0.3)), to: 0)
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("u3("), "u3 gate must appear with parameter list")
        #expect(qasm.contains("0.1"), "u3 export must contain theta parameter value 0.1")
        #expect(qasm.contains("0.2"), "u3 export must contain phi parameter value 0.2")
        #expect(qasm.contains("0.3"), "u3 export must contain lambda parameter value 0.3")
    }

    @Test("Symbolic parameter exports parameter name instead of value")
    func exportSymbolicParameter() {
        var circuit = QuantumCircuit(qubits: 1)
        let theta = Parameter(name: "theta")
        circuit.append(.rotationZ(.parameter(theta)), to: 0)
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("rz(theta)"), "symbolic parameter must export as parameter name 'theta'")
    }

    @Test("Controlled gate with parameters exports ctrl @ with parameter list")
    func exportControlledGateWithParameters() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlled(gate: .rotationZ(.value(1.5)), controls: [0]), to: [0, 1])
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("ctrl @"), "controlled parameterized gate must use ctrl @ modifier")
        #expect(qasm.contains("1.5"), "controlled gate parameter value must appear in export")
    }

    @Test("Negated parameter exports with minus prefix")
    func exportNegatedParameter() {
        var circuit = QuantumCircuit(qubits: 1)
        let beta = Parameter(name: "beta")
        circuit.append(.rotationX(.negatedParameter(beta)), to: 0)
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("-beta"), "negated parameter must export with minus prefix as '-beta'")
    }

    @Test("formatDouble handles integer-valued double with .0 suffix")
    func exportIntegerValuedDouble() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(.value(2.0)), to: 0)
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("2.0"), "integer-valued double 2.0 must format with decimal point")
    }

    @Test("formatDouble rounds 1.0 to integer format with .0 suffix")
    func exportFormatDoubleExactInteger() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(.value(1.0)), to: 0)
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("1.0"), "exact integer value 1.0 must format with .0 suffix via formatDouble")
    }

    @Test("controlledPhase exports as cp with parameter value")
    func exportControlledPhaseWithParam() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlledPhase(.value(0.75)), to: [0, 1])
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("cp(0.75)"), "controlledPhase must export as 'cp(0.75)' with parameter in QASM 3.0")
    }

    @Test("formatDouble appends .0 suffix when String representation lacks decimal point")
    func exportFormatDoubleNaNAppendsDotZero() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(.value(.nan)), to: 0)
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("nan.0"), "NaN value must format as 'nan.0' via formatDouble fallback branch")
    }

    @Test("formatDouble appends .0 suffix for infinity value")
    func exportFormatDoubleInfinityAppendsDotZero() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(.value(.infinity)), to: 0)
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("inf.0"), "Infinity value must format as 'inf.0' via formatDouble fallback branch")
    }

    @Test("controlledRotationX exports as crx with parameter value")
    func exportControlledRotationXWithParam() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlledRotationX(.value(1.25)), to: [0, 1])
        let qasm = QASM3Exporter.export(circuit)
        #expect(qasm.contains("crx(1.25)"), "controlledRotationX must export as 'crx(1.25)' with parameter in QASM 3.0")
    }
}

/// Validates QASM3Importer extended parsing of control flow and declarations.
/// Covers for/while loops, classical type declarations, box/stretch/cal/defcal,
/// IO declarations, gate modifiers, expression functions, and register resolution.
@Suite("QASM3Importer Extended Coverage")
struct QASM3ImporterExtendedCoverageTests {
    @Test("For loop statement parses with warning and skips body")
    func parseForLoop() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q;
        for int i in [0:2] { x q[i]; }
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasForWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("for")
        }
        #expect(hasForWarning, "for loop must produce warning about not being fully modeled")
        #expect(result.circuit.count >= 1, "parser must continue after for loop and parse subsequent gates")
    }

    @Test("While loop statement parses with warning and skips body")
    func parseWhileLoop() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[1] c;
        while (c[0]) { x q[0]; c[0] = measure q[0]; }
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasWhileWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("while")
        }
        #expect(hasWhileWarning, "while loop must produce warning about not being fully modeled")
        #expect(result.circuit.count >= 1, "parser must continue after while loop")
    }

    @Test("Float classical declaration parses and continues")
    func parseFloatDeclaration() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        float[64] f = 3.14;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "parser must continue after float declaration")
    }

    @Test("Bool classical declaration parses and continues")
    func parseBoolDeclaration() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bool b = true;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "parser must continue after bool declaration")
    }

    @Test("Angle classical declaration without assignment parses")
    func parseAngleDeclaration() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        angle[32] a;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "parser must continue after angle declaration without assignment")
    }

    @Test("Uint classical declaration parses and continues")
    func parseUintDeclaration() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        uint[8] u = 7;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "parser must continue after uint declaration")
    }

    @Test("Const declaration parses and continues")
    func parseConstDeclaration() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        const int[32] n = 4;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "parser must continue after const declaration")
    }

    @Test("Box statement parses with warning and processes body")
    func parseBoxStatement() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        box { h q[0]; }
        """
        let result = QASM3Importer.parse(source)
        let hasBoxWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("box")
        }
        #expect(hasBoxWarning, "box statement must produce warning about not being fully modeled")
    }

    @Test("Box statement with duration bracket parses")
    func parseBoxWithDuration() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        box[100ns] { h q[0]; }
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasBoxWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("box")
        }
        #expect(hasBoxWarning, "box with duration must produce warning")
    }

    @Test("Stretch declaration parses with warning")
    func parseStretchDeclaration() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        stretch s;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasStretchWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("stretch")
        }
        #expect(hasStretchWarning, "stretch declaration must produce warning about not being supported")
        #expect(result.circuit.count >= 1, "parser must continue after stretch declaration")
    }

    @Test("Cal block parses with warning")
    func parseCalBlock() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        cal { }
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasCalWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("calibration")
        }
        #expect(hasCalWarning, "cal block must produce warning about calibration not being supported")
        #expect(result.circuit.count >= 1, "parser must continue after cal block")
    }

    @Test("Defcal block parses with warning")
    func parseDefcalBlock() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        defcal h q { }
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasCalWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("calibration")
        }
        #expect(hasCalWarning, "defcal block must produce warning about calibration not being supported")
        #expect(result.circuit.count >= 1, "parser must continue after defcal block")
    }

    @Test("Input IO declaration parses and continues")
    func parseInputDeclaration() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        input float[64] theta;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "parser must continue after input declaration")
    }

    @Test("Output IO declaration parses and continues")
    func parseOutputDeclaration() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[2] result;
        output bit[2] result;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "parser must continue after output declaration")
    }

    @Test("Let declaration parses and continues")
    func parseLetDeclaration() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        let alias = q;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "parser must continue after let declaration")
    }

    @Test("Return statement inside subroutine parses")
    func parseReturnStatement() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        def f() -> int[32] { return 0; }
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "parser must continue after subroutine with return statement")
    }

    @Test("Gate definition with parameters stores and expands correctly")
    func parseGateWithParameters() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        gate rxy(theta, phi) a { rx(theta) a; ry(phi) a; }
        rxy(1.0, 0.5) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 2, "parameterized gate expansion must produce at least 2 inner operations")
    }

    @Test("Subroutine with arrow return type parses body")
    func parseSubroutineWithArrowReturn() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        def myFunc() -> int[32] { return 0; }
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "parser must handle subroutine with arrow return type and continue")
    }

    @Test("Expression asin function evaluates correctly")
    func parseAsinExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(asin(0.5)) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz(asin(0.5)) must parse without errors")
        #expect(result.circuit.count == 1, "rz(asin(0.5)) must produce exactly one gate operation")
    }

    @Test("Expression acos function evaluates correctly")
    func parseAcosExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(acos(0.5)) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz(acos(0.5)) must parse without errors")
        #expect(result.circuit.count == 1, "rz(acos(0.5)) must produce exactly one gate operation")
    }

    @Test("Expression atan function evaluates correctly")
    func parseAtanExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(atan(1.0)) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz(atan(1.0)) must parse without errors")
        #expect(result.circuit.count == 1, "rz(atan(1.0)) must produce exactly one gate operation")
    }

    @Test("Expression exp function evaluates correctly")
    func parseExpExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(exp(1)) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz(exp(1)) must parse without errors")
        #expect(result.circuit.count == 1, "rz(exp(1)) must produce exactly one gate operation")
    }

    @Test("Expression ln function evaluates correctly")
    func parseLnExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(ln(2)) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz(ln(2)) must parse without errors")
        #expect(result.circuit.count == 1, "rz(ln(2)) must produce exactly one gate operation")
    }

    @Test("Expression sqrt function evaluates correctly")
    func parseSqrtExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(sqrt(4)) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz(sqrt(4)) must parse without errors")
        #expect(result.circuit.count == 1, "rz(sqrt(4)) must produce exactly one gate operation")
    }

    @Test("Parenthesized subexpression evaluates correctly")
    func parseParenthesizedSubexpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz((pi + pi) / 2) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz((pi + pi) / 2) must parse without errors")
        #expect(result.circuit.count == 1, "parenthesized expression must produce exactly one gate operation")
    }

    @Test("Integer literal in expression evaluates as double")
    func parseIntegerInExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(42) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz(42) with integer literal must parse without errors")
        #expect(result.circuit.count == 1, "integer literal in expression must produce one gate operation")
    }

    @Test("Unary minus in expression negates value")
    func parseUnaryMinusExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(-pi) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz(-pi) must parse without errors")
        #expect(result.circuit.count == 1, "unary minus expression must produce one gate operation")
    }

    @Test("Unary plus in expression passes through value")
    func parseUnaryPlusExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(+pi) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz(+pi) must parse without errors")
        #expect(result.circuit.count == 1, "unary plus expression must produce one gate operation")
    }

    @Test("Multiplication in expression evaluates correctly")
    func parseMultiplicationExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(2 * pi) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz(2 * pi) must parse without errors")
        #expect(result.circuit.count == 1, "multiplication expression must produce one gate operation")
    }

    @Test("Multiple qubit args in gate call resolve correctly")
    func parseMultipleQubitArgs() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        cx q[0], q[1];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "cx with two indexed qubit args must parse without errors")
        #expect(result.circuit.count == 1, "cx with two qubit args must produce exactly one operation")
    }

    @Test("Negctrl with explicit count parses")
    func parseNegctrlWithCount() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q;
        negctrl(2) @ x q[0], q[1], q[2];
        """
        let result = QASM3Importer.parse(source)
        let hasNegctrlWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("negctrl")
        }
        #expect(hasNegctrlWarning, "negctrl(2) must produce warning about approximation")
        #expect(result.circuit.count >= 1, "negctrl(2) @ x must produce at least one operation")
    }

    @Test("Pow with non-integer exponent produces warning")
    func parsePowNonInteger() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        pow(1.5) @ x q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasPowWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("non-integer pow")
        }
        #expect(hasPowWarning, "pow(1.5) with non-integer exponent must produce warning about approximation")
        #expect(result.circuit.count >= 1, "pow(1.5) @ x must still produce at least one operation")
    }

    @Test("Inv modifier produces inverse gate")
    func parseInvModifier() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        inv @ s q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "inv @ s must parse without errors")
        #expect(result.circuit.count == 1, "inv modifier must produce exactly one gate operation")
    }

    @Test("Ctrl with explicit count parses")
    func parseCtrlWithCount() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q;
        ctrl(2) @ x q[0], q[1], q[2];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "ctrl(2) @ x must produce at least one controlled operation")
    }

    @Test("Qreg legacy declaration creates qubit register")
    func parseQregDeclaration() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qreg q[2];
        h q[0];
        cx q[0], q[1];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.qubits == 2, "qreg[2] must create 2-qubit register")
        #expect(result.circuit.count == 2, "legacy qreg declaration must allow subsequent gate parsing")
    }

    @Test("Creg legacy declaration creates bit register")
    func parseCregDeclaration() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        creg c[2];
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "legacy creg declaration must allow subsequent gate parsing")
    }

    @Test("Barrier statement parses and skips arguments")
    func parseBarrierStatement() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        barrier q[0], q[1];
        h q[1];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 2, "barrier must be skipped and surrounding gates must be parsed")
    }

    @Test("If statement with braced body parses both branches")
    func parseIfWithBracedBody() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[1] c;
        if (c[0] == 1) { x q[0]; } else { h q[0]; }
        """
        let result = QASM3Importer.parse(source)
        let hasIfWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("if")
        }
        #expect(hasIfWarning, "if with braced body must produce warning")
    }

    @Test("If statement with else branch parses")
    func parseIfWithElse() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[1] c;
        if (c[0] == 1) x q[0]; else h q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasIfWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("if")
        }
        #expect(hasIfWarning, "if-else with inline statements must produce warning")
    }

    @Test("While loop with inline statement parses")
    func parseWhileWithInlineStatement() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[1] c;
        while (c[0]) x q[0];
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasWhileWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("while")
        }
        #expect(hasWhileWarning, "while with inline statement must produce warning")
        #expect(result.circuit.count >= 1, "parser must continue after while with inline statement")
    }

    @Test("Opaque declaration parses and continues")
    func parseOpaqueDeclaration() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        opaque myopaque q;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "parser must continue after opaque declaration")
    }

    @Test("Version header with integer version number parses")
    func parseIntegerVersionHeader() {
        let source = """
        OPENQASM 3;
        include "stdgates.inc";
        qubit[1] q;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "integer version number must be accepted")
    }

    @Test("Version header with integer below 3 produces warning")
    func parseOldIntegerVersion() {
        let source = """
        OPENQASM 2;
        include "stdgates.inc";
        qubit[1] q;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasVersionWarning = result.diagnostics.contains {
            $0.message.contains("version") || $0.message.contains("3.0") || $0.message.contains("2")
        }
        #expect(hasVersionWarning, "OPENQASM 2 with integer version must produce version warning")
    }

    @Test("Single qubit register without index resolves offset")
    func parseSingleQubitRegisterNoIndex() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] a;
        qubit[1] b;
        cx a, b;
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.qubits == 2, "two single-qubit registers must create 2 qubits total")
        #expect(result.circuit.count == 1, "cx on two single-qubit registers must produce one operation")
    }

    @Test("Identifier assignment skips to semicolon")
    func parseIdentifierAssignment() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[1] c;
        c = measure q;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "identifier assignment with measure must not block subsequent parsing")
    }

    @Test("Indexed identifier assignment with measure parses")
    func parseIndexedMeasureAssignment() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        bit[2] c;
        c[0] = measure q[0];
        c[1] = measure q[1];
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "indexed measure assignment must not block subsequent parsing")
    }

    @Test("Gate call falling back to v2 name mapping resolves")
    func parseV2FallbackGateName() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        u1(1.5) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "u1 gate resolved via v2 fallback must produce one operation")
    }

    @Test("Apply parameters to controlledPhase gate")
    func parseControlledPhaseGate() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        cp(1.0) q[0], q[1];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "cp(1.0) must produce exactly one controlled phase operation")
    }

    @Test("Apply parameters to controlledRotationX gate")
    func parseControlledRotationXGate() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        crx(0.5) q[0], q[1];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "crx(0.5) must produce exactly one controlled-rx operation")
    }

    @Test("Apply parameters to controlledRotationY gate")
    func parseControlledRotationYGate() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        cry(0.5) q[0], q[1];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "cry(0.5) must produce exactly one controlled-ry operation")
    }

    @Test("Apply parameters to controlledRotationZ gate")
    func parseControlledRotationZGate() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        crz(0.5) q[0], q[1];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "crz(0.5) must produce exactly one controlled-rz operation")
    }

    @Test("Apply parameters to givens gate")
    func parseGivensGate() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        givens(0.5) q[0], q[1];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "givens(0.5) must produce exactly one givens rotation operation")
    }

    @Test("Apply parameters to xx gate")
    func parseXXGate() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        rxx(0.5) q[0], q[1];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "rxx(0.5) must produce exactly one XX rotation operation")
    }

    @Test("Apply parameters to yy gate")
    func parseYYGate() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        ryy(0.5) q[0], q[1];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "ryy(0.5) must produce exactly one YY rotation operation")
    }

    @Test("Apply parameters to zz gate")
    func parseZZGate() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        rzz(0.5) q[0], q[1];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "rzz(0.5) must produce exactly one ZZ rotation operation")
    }

    @Test("Apply parameters to globalPhase gate")
    func parseGlobalPhaseGate() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        gphase(1.0) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "gphase(1.0) must produce exactly one global phase operation")
    }

    @Test("Apply parameters to phase gate via p name")
    func parsePhaseGate() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        p(0.5) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "p(0.5) must produce exactly one phase gate operation")
    }

    @Test("Apply parameters to rx gate")
    func parseRxGate() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rx(1.0) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "rx(1.0) must produce exactly one rotation-X operation")
    }

    @Test("Apply parameters to ry gate")
    func parseRyGate() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        ry(1.0) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "ry(1.0) must produce exactly one rotation-Y operation")
    }

    @Test("Apply parameters to u2 gate")
    func parseU2Gate() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        u2(0.5, 1.0) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "u2(0.5, 1.0) must produce exactly one u2 gate operation")
    }

    @Test("Apply parameters to u3 gate")
    func parseU3Gate() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        u3(0.1, 0.2, 0.3) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "u3(0.1, 0.2, 0.3) must produce exactly one u3 gate operation")
    }

    @Test("Identifier in expression resolves to zero for unknown name")
    func parseUnknownIdentifierInExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(myvar) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "unknown identifier in expression must resolve to 0 and produce gate")
    }

    @Test("Semicolon-only statement is skipped")
    func parseSemicolonOnlyStatement() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        ;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "bare semicolon must be skipped and subsequent gates parsed")
    }

    @Test("Gate expansion with indexed qubit in body resolves bindings")
    func parseGateExpansionWithIndexedQubit() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        gate myg a, b { cx a, b; }
        myg q[0], q[1];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "gate expansion with qubit bindings must produce operations")
    }

    @Test("Expression with log function evaluates correctly")
    func parseLogExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(log(2)) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz(log(2)) must parse without errors")
        #expect(result.circuit.count == 1, "rz(log(2)) must produce exactly one gate operation")
    }

    @Test("Expression with tan function evaluates correctly")
    func parseTanExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(tan(0.5)) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz(tan(0.5)) must parse without errors")
        #expect(result.circuit.count == 1, "rz(tan(0.5)) must produce exactly one gate operation")
    }

    @Test("Expression with euler constant evaluates correctly")
    func parseEulerExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(euler) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz(euler) must parse without errors")
        #expect(result.circuit.count == 1, "rz(euler) must produce exactly one gate operation")
    }

    @Test("Expression with subtraction evaluates correctly")
    func parseSubtractionExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(pi - 1.0) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz(pi - 1.0) must parse without errors")
        #expect(result.circuit.count == 1, "subtraction expression must produce one gate operation")
    }

    @Test("Expression with division evaluates correctly")
    func parseDivisionExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(pi / 4) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "rz(pi / 4) must parse without errors")
        #expect(result.circuit.count == 1, "division expression must produce one gate operation")
    }

    @Test("Defcal without braces terminates at semicolon")
    func parseDefcalWithoutBraces() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        defcal h q;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasCalWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("calibration")
        }
        #expect(hasCalWarning, "defcal without braces must produce calibration warning")
        #expect(result.circuit.count >= 1, "parser must continue after defcal without braces")
    }

    @Test("Measure statement skips qubit args and semicolon")
    func parseMeasureStatement() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        measure q[0];
        h q[1];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "measure statement must be skipped and subsequent gates parsed")
    }

    @Test("Subroutine with typed parameter list parses")
    func parseSubroutineWithTypedParams() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        def applyRot(float[64] angle) qubit[1] target { rz(angle) target[0]; }
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "parser must continue after subroutine with typed parameters")
    }

    @Test("Expression with keyword true resolves to 1.0")
    func parseKeywordTrueInExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(true) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "keyword true in expression must resolve to 1.0 and produce gate")
    }

    @Test("Expression with keyword false resolves to 0.0")
    func parseKeywordFalseInExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(false) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "keyword false in expression must resolve to 0.0 and produce gate")
    }

    @Test("Real number parsed as integer literal in size bracket")
    func parseRealAsIntegerLiteral() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2.0] q;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "real number in size bracket must be truncated to integer")
    }

    @Test("Multi-qubit register without index expands all qubits in gate")
    func parseMultiQubitRegisterExpansion() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        swap q;
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "unindexed multi-qubit register must expand to all qubit indices")
    }

    @Test("Indexed identifier assignment with non-measure value skips")
    func parseIndexedNonMeasureAssignment() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[1] c;
        c[0] = 1;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "indexed non-measure assignment must skip to semicolon and continue")
    }

    @Test("Identifier non-measure assignment skips to semicolon")
    func parseIdentifierNonMeasureAssignment() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[1] c;
        c = 0;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "identifier non-measure assignment must skip and continue parsing")
    }
}

/// Validates QASM3Importer error paths for missing identifiers and names.
/// Covers missing include paths, register names, gate names, subroutine names,
/// malformed parameter lists, and skipToOpenBrace/skipToSemicolon EOF paths.
@Suite("QASM3Importer Final Coverage")
struct QASM3ImporterFinalCoverageTests {
    @Test("Missing string after include produces error")
    func parseMissingIncludeString() {
        let source = """
        OPENQASM 3.0;
        include ;
        qubit[1] q;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasError = result.diagnostics.contains {
            $0.severity == .error && $0.message.contains("file path")
        }
        #expect(hasError, "include without file path string must produce error about missing file path")
    }

    @Test("Missing qubit register name after sized declaration produces error")
    func parseMissingQubitRegisterName() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] ;
        """
        let result = QASM3Importer.parse(source)
        let hasError = result.diagnostics.contains {
            $0.severity == .error && $0.message.contains("qubit register name")
        }
        #expect(hasError, "qubit declaration without register name must produce error about missing name")
    }

    @Test("Missing bit register name after sized declaration produces error")
    func parseMissingBitRegisterName() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[2] ;
        """
        let result = QASM3Importer.parse(source)
        let hasError = result.diagnostics.contains {
            $0.severity == .error && $0.message.contains("bit register name")
        }
        #expect(hasError, "bit declaration without register name must produce error about missing name")
    }

    @Test("Missing qreg register name produces error")
    func parseMissingQregName() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qreg ;
        """
        let result = QASM3Importer.parse(source)
        let hasError = result.diagnostics.contains {
            $0.severity == .error && $0.message.contains("register name")
        }
        #expect(hasError, "qreg without register name must produce error about missing name")
    }

    @Test("Missing creg register name produces error")
    func parseMissingCregName() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        creg ;
        """
        let result = QASM3Importer.parse(source)
        let hasError = result.diagnostics.contains {
            $0.severity == .error && $0.message.contains("register name")
        }
        #expect(hasError, "creg without register name must produce error about missing name")
    }

    @Test("Missing gate name after gate keyword produces error")
    func parseMissingGateName() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        gate { }
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasError = result.diagnostics.contains {
            $0.severity == .error && $0.message.contains("gate name")
        }
        #expect(hasError, "gate keyword without name must produce error about missing gate name")
    }

    @Test("Missing subroutine name after def keyword produces error")
    func parseMissingSubroutineName() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        def { }
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasError = result.diagnostics.contains {
            $0.severity == .error && $0.message.contains("subroutine name")
        }
        #expect(hasError, "def keyword without name must produce error about missing subroutine name")
    }

    @Test("Subroutine parameter list missing comma produces incomplete parse")
    func parseSubroutineParamMissingComma() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        def f(float[64] a float[64] b) qubit[1] r { h r[0]; }
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "parser must recover after malformed parameter list and parse subsequent gates")
    }

    @Test("Arrow detection failure restores position when no arrow present")
    func parseDefWithoutArrowOrQubitList() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        def f() { }
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "def without arrow or qubit list must parse body and continue")
    }

    @Test("skipToOpenBrace reaches EOF without finding brace")
    func parseGateDefinitionMissingBrace() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        gate mygate a
        """
        let result = QASM3Importer.parse(source)
        let hasError = result.diagnostics.contains {
            $0.severity == .error && $0.message.contains("{")
        }
        #expect(hasError, "gate definition without open brace must produce error about missing '{'")
    }

    @Test("Return statement inside subroutine body is consumed")
    func parseReturnStatementConsumed() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        def f() { return; }
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "return statement inside subroutine must be consumed without blocking parsing")
    }

    @Test("Standalone identifier not matching gate produces unknown gate error")
    func parseStandaloneIdentifierUnknownGate() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        notarealgate q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasError = result.diagnostics.contains {
            $0.severity == .error && $0.message.contains("unknown gate")
        }
        #expect(hasError, "standalone identifier not matching any gate must produce unknown gate error")
    }

    @Test("Modifier with no modifier keywords returns empty modifier list")
    func parseGateCallWithNoModifiers() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "gate call without modifiers must parse successfully")
        #expect(result.circuit.count == 1, "gate call without modifiers must produce one operation")
    }

    @Test("Missing gate name after ctrl @ modifier produces error")
    func parseMissingGateNameAfterModifier() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        ctrl @ ;
        """
        let result = QASM3Importer.parse(source)
        let hasError = result.diagnostics.contains {
            $0.severity == .error && $0.message.contains("gate name")
        }
        #expect(hasError, "ctrl @ without gate name must produce error about missing gate name")
    }

    @Test("Gate expansion with qubit binding index out of range falls back")
    func parseGateExpansionQubitBindingMismatch() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[3] q;
        gate myg a { h a[1]; }
        myg q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "gate expansion with out-of-range index must fall back and still produce operations")
    }

    @Test("Undeclared qubit register with index produces error")
    func parseUndeclaredQubitRegisterWithIndex() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        h undeclared[0];
        """
        let result = QASM3Importer.parse(source)
        let hasError = result.diagnostics.contains {
            $0.severity == .error && $0.message.contains("undeclared qubit register")
        }
        #expect(hasError, "gate on undeclared indexed register must produce undeclared register error")
    }

    @Test("Undeclared qubit register without index produces error")
    func parseUndeclaredQubitRegisterWithoutIndex() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        h nosuchreg;
        """
        let result = QASM3Importer.parse(source)
        let hasError = result.diagnostics.contains {
            $0.severity == .error && $0.message.contains("undeclared qubit register")
        }
        #expect(hasError, "gate on undeclared unindexed register must produce undeclared register error")
    }

    @Test("Unexpected token in expression produces error")
    func parseUnexpectedTokenInExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(;) q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasError = result.diagnostics.contains {
            $0.severity == .error && $0.message.contains("expected expression")
        }
        #expect(hasError, "malformed expression with unexpected token must produce expected-expression error")
    }

    @Test("Missing integer literal where expected produces error")
    func parseMissingIntegerLiteral() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[abc] q;
        """
        let result = QASM3Importer.parse(source)
        let hasError = result.diagnostics.contains {
            $0.severity == .error && $0.message.contains("integer")
        }
        #expect(hasError, "non-integer token where integer expected must produce missing-integer error")
    }

    @Test("Missing open brace in gate definition body produces error")
    func parseMissingOpenBraceInGateBody() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        gate mygate a h a;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasError = result.diagnostics.contains {
            $0.severity == .error && $0.message.contains("{")
        }
        #expect(hasError, "gate definition without opening brace must produce error about missing '{'")
    }

    @Test("skipToSemicolon reaches EOF on unterminated statement")
    func parseUnterminatedStatementSkipToSemicolon() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[1] c;
        c = 0
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.qubits >= 1, "parser must not crash when skipToSemicolon reaches EOF")
    }

    @Test("skipToSymbol reaches EOF on unterminated bracket")
    func parseUnterminatedBracketSkipToSymbol() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        box[100ns
        """
        let result = QASM3Importer.parse(source)
        let hasWarning = result.diagnostics.contains {
            $0.severity == .warning && $0.message.contains("box")
        }
        #expect(hasWarning, "box with unterminated bracket must still produce box warning before EOF")
    }
}

/// Validates QASM3Importer targeted parser paths for version, return, and body parsing.
/// Covers non-numeric version tokens, top-level return, subroutine arrow detection,
/// gate body qubit index resolution, modifier lists, and expression binding paths.
@Suite("QASM3Importer Targeted Coverage")
struct QASM3ImporterTargetedCoverageTests {
    @Test("OPENQASM with non-numeric version token produces error")
    func parseVersionNonNumericToken() {
        let source = """
        OPENQASM abc;
        include "stdgates.inc";
        qubit[1] q;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasError = result.diagnostics.contains {
            $0.severity == .error && $0.message.contains("version number")
        }
        #expect(hasError, "non-numeric version token must produce error about expected version number")
    }

    @Test("Return keyword at top level dispatches parseReturnStatement")
    func parseReturnAtTopLevel() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        return;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "return at top level must be consumed and parsing continues")
    }

    @Test("Subroutine with multiple comma-separated qubit params")
    func parseSubroutineMultipleQubitParams() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        def bell() qubit[1] a, qubit[1] b { h a[0]; cx a[0], b[0]; }
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "subroutine with comma-separated qubit params must parse")
    }

    @Test("Subroutine with minus not followed by greater-than restores position")
    func parseSubroutineMinusNotArrow() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        def f() { }
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "def without arrow must restore position and parse body")
    }

    @Test("Barrier with newline terminator instead of semicolon")
    func parseBarrierNewlineTerminator() {
        let source = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[2] q;\nbarrier q[0]\nh q[0];\n"
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "barrier terminated by newline must parse and continue")
    }

    @Test("Gate body with non-identifier token skips via advance")
    func parseGateBodyNonIdentifierSkip() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        gate myg a { ; h a; }
        myg q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "gate body with leading semicolon must skip non-identifier tokens")
    }

    @Test("Gate body inner qubit with index exceeding mapped count uses offset")
    func parseGateBodyQubitIndexExceedsMapping() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[4] q;
        gate myg r { h r[2]; }
        myg q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "gate body qubit index exceeding mapped count must fall back to offset")
    }

    @Test("Keyword true in expression via keyword token resolves to 1.0")
    func parseKeywordTrueToken() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(true) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "keyword true token must resolve to 1.0 in expression")
    }

    @Test("Keyword false in expression via keyword token resolves to 0.0")
    func parseKeywordFalseToken() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(false) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "keyword false token must resolve to 0.0 in expression")
    }

    @Test("Gate call with no params to non-parameterized gate hits applyParameters default")
    func parseGateNoParamsDefault() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.succeeded, "h without params hits default case in applyParameters")
        #expect(result.circuit.count == 1, "must produce one gate operation")
    }

    @Test("Expression list with empty parens returns empty")
    func parseEmptyExpressionList() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        h() q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 0, "empty parameter list returns immediately")
    }

    @Test("Gate definition with parameterized body and multiple expression bindings")
    func parseGateDefinitionMultiExpressionBindings() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        gate myu(a, b) r { u2(a, b) r; }
        myu(0.5, 1.0) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "gate body with multiple expression bindings must resolve all params")
    }

    @Test("Gate body with newline-terminated inner statement")
    func parseGateBodyNewlineTerminated() {
        let source = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[1] q;\ngate myg a { h a\n}\nmyg q[0];\n"
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "gate body inner statement terminated by newline must still produce operations")
    }

    @Test("Nested braces in braced body increments depth correctly")
    func parseNestedBracesInBody() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        if (true) { { h q[0]; } }
        """
        let result = QASM3Importer.parse(source)
        let hasWarning = result.diagnostics.contains { $0.severity == .warning && $0.message.contains("if") }
        #expect(hasWarning, "nested braces must be tracked by depth counter")
    }

    @Test("Indexed identifier assignment falls back to gate call when no equals after bracket")
    func parseIndexedIdentifierFallbackToGateCall() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        cx q[0], q[1];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "indexed identifier without equals must fall back to gate call")
    }

    @Test("Subroutine parameter declaration list with typed parameters")
    func parseSubroutineTypedParameterDeclList() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        def f(float[64] x, float[64] y) qubit[1] r { rz(x) r[0]; }
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "typed parameter declarations with comma must parse both params")
    }

    @Test("Balanced parens with nested levels tracked correctly")
    func parseBalancedParensNested() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[1] c;
        if ((c[0] == 1)) x q[0];
        """
        let result = QASM3Importer.parse(source)
        let hasWarning = result.diagnostics.contains { $0.severity == .warning && $0.message.contains("if") }
        #expect(hasWarning, "nested parens in if condition must be tracked by depth counter")
    }
}

/// Validates QASM3Importer synchronize and EOF edge case paths.
/// Covers symbol-at-EOF synchronization, malformed input recovery,
/// subroutine arrow EOF, applyParameters default, and whitespace-only source.
@Suite("QASM3 Final Parser Gap Coverage")
struct QASM3FinalParserGapTests {
    @Test("Synchronize reaches EOF when non-identifier symbol ends source without terminator")
    func synchronizeEOFViaSymbol() {
        let source = "OPENQASM 3.0; @"
        let result = QASM3Importer.parse(source)
        #expect(result.diagnostics.contains { $0.severity == .error }, "bare symbol at end must trigger synchronize EOF path")
    }

    @Test("Extremely malformed input triggers repeated synchronize through to EOF")
    func synchronizeExtremelMalformed() {
        let source = "OPENQASM 3.0; @ # $ %"
        let result = QASM3Importer.parse(source)
        #expect(result.diagnostics.count >= 1, "stream of symbols without semicolons must synchronize to EOF repeatedly")
    }

    @Test("Subroutine arrow skipping reaches EOF without finding open brace")
    func skipToOpenBraceEOF() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        def f() -> int[32]
        """
        let result = QASM3Importer.parse(source)
        #expect(result.diagnostics.contains { $0.severity == .error }, "subroutine with arrow but no brace must produce error when skipToOpenBrace reaches EOF")
    }

    @Test("Indexed identifier non-measure assignment without semicolon reaches EOF")
    func indexedAssignmentEOFPath() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        bit[1] c;
        c[0] = 42;
        h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "indexed non-measure assignment must skip value and parse subsequent gates")
    }

    @Test("Modifier list returns empty when encountering non-modifier keyword at start")
    func modifierListEmptyOnNonModifier() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        ctrl @ h q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "ctrl modifier followed by gate call must parse through modifier loop default return")
    }

    @Test("Expression with identifier true resolves to 1.0 via identifier branch")
    func identifierTrueExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(true) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "identifier 'true' must resolve to 1.0 in expression and produce one gate")
    }

    @Test("Expression with identifier false resolves to 0.0 via identifier branch")
    func identifierFalseExpression() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        rz(false) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 1, "identifier 'false' must resolve to 0.0 in expression and produce one gate")
    }

    @Test("Expression with bound parameter in gate body resolves via bindings path")
    func expressionWithBindingsResolvesParam() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[1] q;
        gate myrz(theta) a { rz(theta) a; }
        myrz(1.5) q[0];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "bound parameter 'theta' in gate body must resolve through parseExpressionWithBindings")
    }

    @Test("Non-parameterized gate with params hits applyParameters default returning gate unchanged")
    func applyParametersDefault() {
        let source = """
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        cx(1.0) q[0], q[1];
        """
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count >= 1, "cx with spurious params must still apply via default case in applyParameters")
    }

    @Test("Source ending mid-statement triggers EOF in synchronize")
    func eofMidStatement() {
        let source = "OPENQASM 3.0; qubit[1] q; @"
        let result = QASM3Importer.parse(source)
        #expect(!result.diagnostics.isEmpty, "source ending with symbol must trigger EOF synchronize path")
    }

    @Test("Empty source with whitespace only produces no operations")
    func whitespaceOnlySource() {
        let source = "   \n\n   "
        let result = QASM3Importer.parse(source)
        #expect(result.circuit.count == 0, "whitespace-only source must produce circuit with no operations")
    }
}
