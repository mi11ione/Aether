// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Test suite for empty circuit rendering.
/// Validates that circuits with no operations produce the expected
/// placeholder string with correct qubit count grammar.
@Suite("Empty Circuit Rendering")
struct EmptyCircuitRenderingTests {
    @Test("Empty single-qubit circuit produces singular label")
    func emptySingleQubit() {
        let circuit = QuantumCircuit(qubits: 1)
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram == "Empty circuit (1 qubit)", "Single qubit should use singular 'qubit'")
    }

    @Test("Empty multi-qubit circuit produces plural label")
    func emptyMultiQubit() {
        let circuit = QuantumCircuit(qubits: 3)
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram == "Empty circuit (3 qubits)", "Multiple qubits should use plural 'qubits'")
    }
}

/// Test suite for single-qubit gate box rendering.
/// Validates that gates like Hadamard render with correct
/// box-drawing delimiters and label content.
@Suite("Single Gate Box Rendering")
struct SingleGateBoxRenderingTests {
    @Test("Hadamard gate renders with box delimiters")
    func hadamardGateBox() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("┤H├"), "Hadamard should render as ┤H├ in box-drawing format")
    }

    @Test("Pauli-X gate renders with box delimiters")
    func pauliXGateBox() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("┤X├"), "Pauli-X should render as ┤X├ in box-drawing format")
    }

    @Test("Wire label starts with q0")
    func wireLabelPresent() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("q0"), "Wire should be labeled with qubit index q0")
    }

    @Test("Wire ends with trailing dash")
    func wireTrailingDash() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.hasSuffix("─"), "Wire line should end with trailing dash character")
    }
}

/// Test suite for CNOT gate rendering.
/// Validates that CNOT displays control dot and target oplus
/// symbols on the correct qubit wires.
@Suite("CNOT Gate Rendering")
struct CNOTGateRenderingTests {
    @Test("CNOT shows control dot on control qubit")
    func cnotControlDot() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        let controlLine = lines[0]
        #expect(controlLine.contains("●"), "Control qubit wire should contain filled circle ●")
    }

    @Test("CNOT shows target symbol on target qubit")
    func cnotTargetSymbol() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        let targetLine = lines[2]
        #expect(targetLine.contains("⊕"), "Target qubit wire should contain circled plus ⊕")
    }

    @Test("CNOT renders vertical connector between qubits")
    func cnotVerticalConnector() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        let spacerLine = lines[1]
        #expect(spacerLine.contains("│"), "Spacer line between qubits should contain vertical connector │")
    }
}

/// Test suite for SWAP gate rendering.
/// Validates that SWAP gates display cross symbols on both
/// qubit wires and vertical connectors between them.
@Suite("SWAP Gate Rendering")
struct SWAPGateRenderingTests {
    @Test("SWAP shows cross on both qubits")
    func swapCrossSymbols() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.swap, to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        let firstWire = lines[0]
        let secondWire = lines[2]
        #expect(firstWire.contains("×"), "First qubit wire should contain cross × for SWAP")
        #expect(secondWire.contains("×"), "Second qubit wire should contain cross × for SWAP")
    }

    @Test("SWAP has vertical connector between qubits")
    func swapVerticalConnector() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.swap, to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        let spacerLine = lines[1]
        #expect(spacerLine.contains("│"), "Spacer between SWAP qubits should contain vertical connector │")
    }
}

/// Test suite for multi-qubit gate double-line border rendering.
/// Validates that multi-qubit custom gates use double-line box
/// drawing characters for top and bottom segments.
@Suite("Multi-Qubit Gate Double-Line Borders")
struct MultiQubitGateRenderingTests {
    @Test("iSWAP renders double-line top border")
    func multiGateTopBorder() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.iswap, to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        let topWire = lines[0]
        #expect(topWire.contains("╔"), "Top segment of multi-qubit gate should contain ╔")
        #expect(topWire.contains("╗"), "Top segment of multi-qubit gate should contain ╗")
    }

    @Test("iSWAP renders double-line bottom border")
    func multiGateBottomBorder() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.iswap, to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        let bottomWire = lines[2]
        #expect(bottomWire.contains("╚"), "Bottom segment of multi-qubit gate should contain ╚")
        #expect(bottomWire.contains("╝"), "Bottom segment of multi-qubit gate should contain ╝")
    }

    @Test("iSWAP renders gate label in top segment")
    func multiGateLabel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.iswap, to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("iSW"), "Multi-qubit gate should display label iSW for iSWAP")
    }
}

/// Test suite for measurement gate and classical wire rendering.
/// Validates that measurement renders as boxed M and subsequent
/// idle segments use double-line classical wire characters.
@Suite("Measurement and Classical Wire Rendering")
struct MeasurementRenderingTests {
    @Test("Measurement renders with M in box")
    func measurementBox() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.measure, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("┤M├"), "Measurement should render as ┤M├")
    }

    @Test("Classical wire uses double-line after measurement")
    func classicalWireAfterMeasurement() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.measure, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("═"), "Wire after measurement should use double-line ═ for classical wire")
    }

    @Test("Wire before measurement uses single-line dash")
    func quantumWireBeforeMeasurement() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.measure, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        let unmeasuredLine = lines[2]
        #expect(unmeasuredLine.contains("─"), "Unmeasured qubit wire should use single-line ─")
    }
}

/// Test suite for controlled gates with non-adjacent qubits.
/// Validates that vertical connectors pass through intermediate
/// qubit spacer lines when control and target are separated.
@Suite("Non-Adjacent Controlled Gate Rendering")
struct NonAdjacentControlledGateTests {
    @Test("Toffoli with 3 qubits shows vertical connectors through intermediates")
    func toffoliVerticalConnectors() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.toffoli, to: [0, 1, 2])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        let spacer01 = lines[1]
        let spacer12 = lines[3]
        #expect(spacer01.contains("│"), "Spacer between q0-q1 should have vertical connector for Toffoli")
        #expect(spacer12.contains("│"), "Spacer between q1-q2 should have vertical connector for Toffoli")
    }

    @Test("CNOT across non-adjacent qubits shows vertical through middle qubit spacer")
    func cnotNonAdjacent() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.cnot, to: [0, 2])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        let spacer01 = lines[1]
        let spacer12 = lines[3]
        #expect(spacer01.contains("│"), "Spacer between q0-q1 should have vertical connector for non-adjacent CNOT")
        #expect(spacer12.contains("│"), "Spacer between q1-q2 should have vertical connector for non-adjacent CNOT")
    }

    @Test("Middle qubit wire shows idle dash when not involved in gate")
    func middleQubitIdle() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.cnot, to: [0, 2])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        let middleWire = lines[2]
        #expect(middleWire.contains("─"), "Middle qubit not in gate should show idle wire dash ─")
    }
}

/// Test suite for color mode enabled output.
/// Validates that ANSI escape codes are present in rendered
/// output when colorEnabled parameter is set to true.
@Suite("Color Mode Enabled")
struct ColorModeEnabledTests {
    @Test("Color mode adds ANSI escape codes for gate")
    func colorModeHasEscapeCodes() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit, colorEnabled: true)
        #expect(diagram.contains("\u{1b}["), "Color-enabled output should contain ANSI escape sequences")
    }

    @Test("Color mode adds cyan for gate label")
    func colorModeCyanGate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit, colorEnabled: true)
        #expect(diagram.contains("\u{1b}[36m"), "Gate label should use cyan ANSI code \\e[36m")
    }

    @Test("Color mode adds yellow for control dot")
    func colorModeYellowControl() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit, colorEnabled: true)
        #expect(diagram.contains("\u{1b}[33m"), "Control dot should use yellow ANSI code \\e[33m")
    }

    @Test("Color mode adds reset code after colored segment")
    func colorModeResetCode() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit, colorEnabled: true)
        #expect(diagram.contains("\u{1b}[0m"), "Color output should contain ANSI reset code \\e[0m")
    }

    @Test("Color mode adds magenta for measurement")
    func colorModeMagentaMeasure() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.measure, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit, colorEnabled: true)
        #expect(diagram.contains("\u{1b}[35m"), "Measurement should use magenta ANSI code \\e[35m")
    }

    @Test("Color mode adds green for reset gate")
    func colorModeGreenReset() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.reset, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit, colorEnabled: true)
        #expect(diagram.contains("\u{1b}[32m"), "Reset gate should use green ANSI code \\e[32m")
    }
}

/// Test suite for color mode disabled output.
/// Validates that no ANSI escape codes appear in rendered
/// output when colorEnabled is false (the default).
@Suite("Color Mode Disabled")
struct ColorModeDisabledTests {
    @Test("Default mode produces no ANSI escape codes")
    func noEscapeCodesDefault() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(!diagram.contains("\u{1b}["), "Default rendering should not contain ANSI escape sequences")
    }

    @Test("Explicit colorEnabled false produces no ANSI escape codes")
    func noEscapeCodesExplicit() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit, colorEnabled: false)
        #expect(!diagram.contains("\u{1b}["), "Explicit colorEnabled=false should not contain ANSI escape sequences")
    }
}

/// Test suite for rotation gate angle display in box.
/// Validates that parameterized gates render with angle values
/// formatted inside box-drawing delimiters.
@Suite("Rotation Gate Angle Rendering")
struct RotationGateAngleRenderingTests {
    @Test("Ry gate renders with angle in box")
    func ryGateAngle() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationY(.pi / 4), to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("Ry(π/4)"), "Ry(pi/4) should render with formatted angle Ry(π/4)")
    }

    @Test("Rz gate renders with pi label")
    func rzGatePi() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(.pi), to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("Rz(π)"), "Rz(pi) should render with π symbol")
    }

    @Test("Phase gate renders with angle in box")
    func phaseGateAngle() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.phase(.pi / 2), to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("P(π/2)"), "Phase(pi/2) should render as P(π/2)")
    }

    @Test("Controlled rotation renders angle on target wire")
    func controlledRotationAngle() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlledRotationY(.pi / 4), to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("Ry(π/4)"), "Controlled Ry should display angle Ry(π/4) on target wire")
        #expect(diagram.contains("●"), "Controlled Ry should display control dot ● on control wire")
    }

    @Test("Negative angle renders correctly")
    func negativeAngle() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(-.pi / 2), to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("Rz(-π/2)"), "Negative pi/2 angle should render as Rz(-π/2)")
    }
}

/// Test suite for reset gate rendering.
/// Validates that reset operations render with zero label
/// inside box-drawing delimiters.
@Suite("Reset Gate Rendering")
struct ResetGateRenderingTests {
    @Test("Reset gate renders as boxed zero")
    func resetGateBox() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.reset, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("┤0├"), "Reset gate should render as ┤0├")
    }
}

/// Test suite for multi-wire diagram structure.
/// Validates that diagrams with multiple qubits produce the correct
/// number of lines and proper qubit label formatting.
@Suite("Multi-Wire Diagram Structure")
struct MultiWireDiagramStructureTests {
    @Test("Two-qubit diagram has 3 lines (2 wires + 1 spacer)")
    func twoQubitLineCount() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n")
        #expect(lines.count == 3, "2-qubit diagram should have 3 lines: 2 wire lines + 1 spacer, got \(lines.count)")
    }

    @Test("Three-qubit diagram has 5 lines")
    func threeQubitLineCount() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n")
        #expect(lines.count == 5, "3-qubit diagram should have 5 lines: 3 wire lines + 2 spacers, got \(lines.count)")
    }

    @Test("Qubit labels are sequential q0, q1, q2")
    func qubitLabelsSequential() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("q0"), "Diagram should contain label q0")
        #expect(diagram.contains("q1"), "Diagram should contain label q1")
        #expect(diagram.contains("q2"), "Diagram should contain label q2")
    }

    @Test("Wire labels include colon separator")
    func wireLabelColonSeparator() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains(": "), "Wire label should be followed by colon-space separator")
    }
}

@Suite("Qubit Label Padding")
struct QubitLabelPaddingTests {
    @Test("Labels pad to width of largest qubit index")
    func labelPaddingForDoubleDigitQubits() {
        var circuit = QuantumCircuit(qubits: 4)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 3)
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        let firstWire = lines[0]
        let lastWire = lines[6]
        #expect(firstWire.hasPrefix("q0"), "First wire should start with q0 label")
        #expect(lastWire.hasPrefix("q3"), "Last wire should start with q3 label")
        let prefixQ0 = firstWire.prefix(while: { $0 != ":" })
        let prefixQ3 = lastWire.prefix(while: { $0 != ":" })
        #expect(prefixQ0.count == prefixQ3.count, "Qubit labels q0 and q3 should be padded to equal width")
    }
}

@Suite("Classical Idle Rendering")
struct ClassicalIdleRenderingTests {
    @Test("Measured qubit shows classical idle wire when another qubit has a gate")
    func classicalIdleAfterMeasure() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.measure, to: 0)
        circuit.append(.hadamard, to: 1)
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        let measuredWire = lines[0]
        let measuredSegments = measuredWire.split(separator: "├").last ?? ""
        #expect(measuredSegments.contains("═"), "Measured qubit wire should use double-line ═ for classical idle in subsequent layers")
    }
}

@Suite("Controlled Gate Label Rendering")
struct ControlledGateLabelRenderingTests {
    @Test("Controlled Hadamard renders control dot and gate label")
    func controlledHadamardLabel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlled(gate: .hadamard, controls: [0]), to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        #expect(lines[0].contains("●"), "Control qubit should show filled circle ● for controlled Hadamard")
        #expect(lines[2].contains("H"), "Target qubit should show H label for controlled Hadamard")
    }
}

@Suite("Negative Pi/4 Parameter Formatting")
struct NegativePiFourthParameterTests {
    @Test("Gate with -pi/4 angle renders as -π/4")
    func negativePiFourthAngle() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(-.pi / 4), to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("Rz(-π/4)"), "Rotation gate with -.pi/4 should render as Rz(-π/4)")
    }
}

@Suite("Named Parameter Rendering")
struct NamedParameterRenderingTests {
    @Test("Symbolic parameter renders with parameter name")
    func namedParameterDisplay() {
        let theta = Parameter(name: "theta")
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationY(.parameter(theta)), to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("Ry(theta)"), "Symbolic parameter should render with its name as Ry(theta)")
    }
}

@Suite("Negated Parameter Rendering")
struct NegatedParameterRenderingTests {
    @Test("Negated symbolic parameter renders with minus prefix")
    func negatedParameterDisplay() {
        let alpha = Parameter(name: "alpha")
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationX(.negatedParameter(alpha)), to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("Rx(-alpha)"), "Negated parameter should render as Rx(-alpha)")
    }
}

@Suite("Fredkin Gate Rendering")
struct FredkinGateRenderingTests {
    @Test("Fredkin gate renders control dot and swap crosses")
    func fredkinControlAndSwaps() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.fredkin, to: [0, 1, 2])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        #expect(lines[0].contains("●"), "Fredkin control qubit should show ●")
        #expect(lines[2].contains("×"), "Fredkin first target should show × swap cross")
        #expect(lines[4].contains("×"), "Fredkin second target should show × swap cross")
    }
}

@Suite("CCZ Gate Rendering")
struct CCZGateRenderingTests {
    @Test("CCZ gate renders two control dots and Z label")
    func cczControlsAndTarget() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.ccz, to: [0, 1, 2])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        #expect(lines[0].contains("●"), "CCZ first control should show ●")
        #expect(lines[2].contains("●"), "CCZ second control should show ●")
        #expect(lines[4].contains("Z"), "CCZ target should show Z label")
    }
}

@Suite("Controlled Gate Role Classification")
struct ControlledGateRoleClassificationTests {
    @Test("Controlled PauliX with single control renders as CNOT-style target")
    func controlledPauliXAsCnotTarget() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlled(gate: .pauliX, controls: [0]), to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        #expect(lines[0].contains("●"), "Controlled pauliX control qubit should show ●")
        #expect(lines[2].contains("⊕"), "Controlled pauliX target should show ⊕ like CNOT")
    }

    @Test("Controlled PauliZ renders as single gate label on target")
    func controlledPauliZLabel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlled(gate: .pauliZ, controls: [0]), to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        #expect(lines[0].contains("●"), "Controlled pauliZ control qubit should show ●")
        #expect(lines[2].contains("Z"), "Controlled pauliZ target should show Z label")
    }

    @Test("Multi-control controlled gate renders multiple control dots")
    func multiControlGateRendering() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.controlled(gate: .hadamard, controls: [0, 1]), to: [0, 1, 2])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        #expect(lines[0].contains("●"), "First control qubit should show ●")
        #expect(lines[2].contains("●"), "Second control qubit should show ●")
        #expect(lines[4].contains("H"), "Target qubit should show H gate label")
    }
}

@Suite("Diagonal Multiplexor CustomUnitary Multi-Qubit Rendering")
struct DiagonalMultiplexorCustomUnitaryTests {
    @Test("Diagonal gate on 2 qubits renders as multi-gate segment")
    func diagonalTwoQubitRendering() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.diagonal(phases: [0.0, 0.1, 0.2, 0.3]), to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("Diag"), "Diagonal gate should render with Diag label")
        #expect(diagram.contains("╔"), "Multi-qubit diagonal should have double-line top border ╔")
        #expect(diagram.contains("╝"), "Multi-qubit diagonal should have double-line bottom border ╝")
    }

    @Test("Diagonal gate on 1 qubit renders as single gate")
    func diagonalSingleQubitRendering() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.diagonal(phases: [0.0, 0.1]), to: [0])
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("┤Diag├"), "Single-qubit diagonal should render as boxed single gate ┤Diag├")
    }

    @Test("CustomUnitary on 2 qubits renders as multi-gate segment")
    func customUnitaryTwoQubitRendering() {
        let matrix: [[Complex<Double>]] = [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .zero, .one],
            [.zero, .zero, .one, .zero],
        ]
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.customUnitary(matrix: matrix), to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("╔"), "Multi-qubit customUnitary should render with double-line top ╔")
        #expect(diagram.contains("╝"), "Multi-qubit customUnitary should render with double-line bottom ╝")
    }
}

@Suite("Multi-Gate Segment Middle Qubit Rendering")
struct MultiGateSegmentMiddleQubitTests {
    @Test("3-qubit gate renders middle qubit with double-line side borders")
    func threeQubitGateMiddleSegment() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.customUnitary(matrix: Array(
            repeating: Array(repeating: Complex<Double>.zero, count: 8),
            count: 8,
        )), to: [0, 1, 2])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        let topWire = lines[0]
        let middleWire = lines[2]
        let bottomWire = lines[4]
        #expect(topWire.contains("╔"), "Top qubit of 3-qubit gate should contain ╔")
        #expect(middleWire.contains("╠"), "Middle qubit of 3-qubit gate should contain ╠")
        #expect(middleWire.contains("╣"), "Middle qubit of 3-qubit gate should contain ╣")
        #expect(bottomWire.contains("╚"), "Bottom qubit of 3-qubit gate should contain ╚")
    }

    @Test("Controlled gate with multi-qubit target renders middle segment")
    func controlledMultiQubitTargetMiddle() {
        var circuit = QuantumCircuit(qubits: 4)
        circuit.append(.controlled(gate: .iswap, controls: [0]), to: [0, 1, 2, 3])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        #expect(lines[0].contains("●"), "Control qubit should show ● for controlled multi-qubit gate")
        #expect(lines[2].contains("╔"), "First target qubit should contain ╔ top border")
        #expect(lines[4].contains("╠") || lines[4].contains("═"), "Middle target qubit should contain ╠ or ═ double-line segment")
        #expect(lines[6].contains("╚"), "Last target qubit should contain ╚ bottom border")
    }
}

@Suite("Qubit Label Padding Multiple Qubits")
struct QubitLabelPaddingMultipleTests {
    @Test("Qubit labels are padded correctly for small circuits")
    func qubitLabelPadding() {
        var circuit = QuantumCircuit(qubits: 4)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 3)
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        let firstWire = lines[0]
        let lastWire = lines[6]
        #expect(firstWire.hasPrefix("q0"), "First wire should start with q0 label")
        #expect(lastWire.hasPrefix("q3"), "Last wire should start with q3 label")
        let prefixQ0 = firstWire.prefix(while: { $0 != ":" })
        let prefixQ3 = lastWire.prefix(while: { $0 != ":" })
        #expect(prefixQ0.count == prefixQ3.count, "q0 and q3 labels should be padded to equal width")
    }
}

@Suite("Classical Idle Wire in Multi-Step Circuit")
struct ClassicalIdleMultiStepTests {
    @Test("Measured qubit renders classical idle in layer where another qubit has a gate")
    func classicalIdleMultiStep() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.measure, to: 0)
        circuit.append(.pauliX, to: 1)
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        let measuredWire = lines[0]
        let afterMeasure = measuredWire.split(separator: "├").last.map(String.init) ?? ""
        #expect(afterMeasure.contains("═"), "Measured qubit q0 should show classical idle ═ in layer after measurement")
    }

    @Test("Measured qubit trailing wire uses double-line")
    func classicalTrailingWire() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.measure, to: 0)
        circuit.append(.hadamard, to: 1)
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("═"), "Diagram should contain classical double-line wire ═ after measurement")
        let lines = diagram.split(separator: "\n").map(String.init)
        let measuredLine = lines[0]
        #expect(measuredLine.hasSuffix("═"), "Measured qubit wire should end with classical double-line ═")
    }
}

@Suite("Gate Label Branch Coverage")
struct GateLabelBranchCoverageTests {
    @Test("CY gate renders Y label on target wire")
    func cyGateLabel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cy, to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        #expect(lines[0].contains("●"), "CY control should show ●")
        #expect(lines[2].contains("Y"), "CY target should show Y label")
    }

    @Test("CZ gate renders Z label on target wire")
    func czGateLabel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cz, to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        let lines = diagram.split(separator: "\n").map(String.init)
        #expect(lines[0].contains("●"), "CZ control should show ●")
        #expect(lines[2].contains("Z"), "CZ target should show Z label")
    }

    @Test("Controlled phase gate renders P label with angle")
    func controlledPhaseLabel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlledPhase(.pi / 2), to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("P(π/2)"), "Controlled phase should render P(π/2) on target wire")
        #expect(diagram.contains("●"), "Controlled phase should show control dot ●")
    }

    @Test("Controlled rotation X renders Rx label with angle")
    func controlledRotationXLabel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlledRotationX(.pi / 4), to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("Rx(π/4)"), "CRx should render Rx(π/4) on target wire")
        #expect(diagram.contains("●"), "CRx should show control dot ●")
    }

    @Test("Controlled rotation Z renders Rz label with angle")
    func controlledRotationZLabel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlledRotationZ(.pi), to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("Rz(π)"), "CRz should render Rz(π) on target wire")
        #expect(diagram.contains("●"), "CRz should show control dot ●")
    }

    @Test("sqrtSwap gate renders with √SW label")
    func sqrtSwapLabel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.sqrtSwap, to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("√SW"), "sqrtSwap should render with √SW label")
    }

    @Test("sqrtISwap gate renders with √iSW label")
    func sqrtISwapLabel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.sqrtISwap, to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("√iSW"), "sqrtISwap should render with √iSW label")
    }

    @Test("fswap gate renders with fSW label")
    func fswapLabel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.fswap, to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("fSW"), "fswap should render with fSW label")
    }

    @Test("Givens gate renders with Giv label and angle")
    func givensLabel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.givens(.pi / 4), to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("Giv(π/4)"), "Givens should render with Giv(π/4) label")
    }

    @Test("XX gate renders with XX label and angle")
    func xxLabel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.xx(.pi / 2), to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("XX(π/2)"), "XX gate should render with XX(π/2) label")
    }

    @Test("YY gate renders with YY label and angle")
    func yyLabel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.yy(.pi / 4), to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("YY(π/4)"), "YY gate should render with YY(π/4) label")
    }

    @Test("ZZ gate renders with ZZ label and angle")
    func zzLabel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.zz(.pi), to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("ZZ(π)"), "ZZ gate should render with ZZ(π) label")
    }

    @Test("Multiplexor gate renders with Mux label")
    func multiplexorLabel() {
        let identity: [[Complex<Double>]] = [[.one, .zero], [.zero, .one]]
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.multiplexor(unitaries: [identity, identity]), to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("Mux"), "Multiplexor should render with Mux label")
    }

    @Test("CustomUnitary identity 2x2 renders U label as single gate")
    func customUnitarySingleQubitLabel() {
        let identity: [[Complex<Double>]] = [[.one, .zero], [.zero, .one]]
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.customUnitary(matrix: identity), to: [0])
        let diagram = CircuitDiagramUnicode.render(circuit)
        #expect(diagram.contains("┤U├"), "Single-qubit customUnitary should render as boxed ┤U├")
    }
}

@Suite("ANSI Color Branch Coverage")
struct ANSIColorBranchCoverageTests {
    @Test("Measurement gate with color uses magenta escape code")
    func measureColorMagenta() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.measure, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit, colorEnabled: true)
        #expect(diagram.contains("\u{1b}[35m"), "Measurement with color should contain magenta ANSI code \\e[35m")
        #expect(diagram.contains("\u{1b}[0m"), "Measurement with color should contain reset ANSI code \\e[0m")
    }

    @Test("Control dot with color uses yellow escape code")
    func controlDotColorYellow() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit, colorEnabled: true)
        #expect(diagram.contains("\u{1b}[33m"), "Control dot with color should contain yellow ANSI code \\e[33m")
    }

    @Test("CNOT target with color uses cyan escape code")
    func cnotTargetColorCyan() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit, colorEnabled: true)
        let lines = diagram.split(separator: "\n").map(String.init)
        let targetLine = lines[2]
        #expect(targetLine.contains("\u{1b}[36m"), "CNOT target with color should contain cyan ANSI code \\e[36m on target wire")
    }

    @Test("SWAP crosses with color use cyan escape code")
    func swapCrossColorCyan() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.swap, to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit, colorEnabled: true)
        #expect(diagram.contains("\u{1b}[36m"), "SWAP crosses with color should contain cyan ANSI code \\e[36m")
    }

    @Test("Multi-qubit gate top segment with color uses cyan for label")
    func multiQubitTopColorCyan() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.iswap, to: [0, 1])
        let diagram = CircuitDiagramUnicode.render(circuit, colorEnabled: true)
        let lines = diagram.split(separator: "\n").map(String.init)
        let topLine = lines[0]
        #expect(topLine.contains("\u{1b}[36m"), "Multi-qubit gate top should contain cyan ANSI code \\e[36m for label")
        #expect(topLine.contains("\u{1b}[0m"), "Multi-qubit gate top should contain reset ANSI code \\e[0m")
    }

    @Test("Reset gate with color uses green escape code")
    func resetColorGreen() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.reset, to: 0)
        let diagram = CircuitDiagramUnicode.render(circuit, colorEnabled: true)
        #expect(diagram.contains("\u{1b}[32m"), "Reset gate with color should contain green ANSI code \\e[32m")
        #expect(diagram.contains("\u{1b}[0m"), "Reset gate with color should contain reset ANSI code \\e[0m")
    }

    @Test("Classical idle wire does not inject ANSI codes")
    func classicalIdleNoColor() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.measure, to: 0)
        circuit.append(.hadamard, to: 1)
        let diagram = CircuitDiagramUnicode.render(circuit, colorEnabled: true)
        let lines = diagram.split(separator: "\n").map(String.init)
        let measuredLine = lines[0]
        #expect(measuredLine.contains("═"), "Classical idle segment should contain ═ even with color enabled")
    }
}

@Suite("Multi-Qubit Custom Gate Color Rendering")
struct MultiQubitCustomGateColorTests {
    @Test("3-qubit custom gate with color renders cyan label on top segment")
    func threeQubitCustomGateColor() {
        let size = 8
        let matrix = Array(
            repeating: Array(repeating: Complex<Double>.zero, count: size),
            count: size,
        )
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.customUnitary(matrix: matrix), to: [0, 1, 2])
        let diagram = CircuitDiagramUnicode.render(circuit, colorEnabled: true)
        let lines = diagram.split(separator: "\n").map(String.init)
        let topLine = lines[0]
        #expect(topLine.contains("\u{1b}[36m"), "3-qubit custom gate top should contain cyan ANSI code \\e[36m")
        #expect(topLine.contains("\u{1b}[0m"), "3-qubit custom gate top should contain reset ANSI code \\e[0m")
        #expect(topLine.contains("╔"), "3-qubit custom gate top should contain ╔ border")
    }

    @Test("3-qubit custom gate with color renders middle and bottom segments")
    func threeQubitCustomGateColorMiddleBottom() {
        let size = 8
        let matrix = Array(
            repeating: Array(repeating: Complex<Double>.zero, count: size),
            count: size,
        )
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.customUnitary(matrix: matrix), to: [0, 1, 2])
        let diagram = CircuitDiagramUnicode.render(circuit, colorEnabled: true)
        let lines = diagram.split(separator: "\n").map(String.init)
        let middleLine = lines[2]
        let bottomLine = lines[4]
        #expect(middleLine.contains("╠"), "3-qubit custom gate middle should contain ╠ border")
        #expect(middleLine.contains("╣"), "3-qubit custom gate middle should contain ╣ border")
        #expect(bottomLine.contains("╚"), "3-qubit custom gate bottom should contain ╚ border")
        #expect(bottomLine.contains("╝"), "3-qubit custom gate bottom should contain ╝ border")
    }

    @Test("Controlled multi-qubit gate with color renders control dot and gate borders")
    func controlledMultiQubitGateColor() {
        var circuit = QuantumCircuit(qubits: 4)
        circuit.append(.controlled(gate: .iswap, controls: [0]), to: [0, 1, 2, 3])
        let diagram = CircuitDiagramUnicode.render(circuit, colorEnabled: true)
        let lines = diagram.split(separator: "\n").map(String.init)
        #expect(lines[0].contains("\u{1b}[33m"), "Control dot with color should contain yellow ANSI code \\e[33m")
        #expect(lines[2].contains("\u{1b}[36m"), "Multi-qubit gate top with color should contain cyan ANSI code \\e[36m")
        #expect(lines[2].contains("╔"), "First target qubit should contain ╔ top border")
    }
}
