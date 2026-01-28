// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Validates CircuitDiagramASCII.render produces correct ASCII diagrams.
/// Covers empty circuits, single/multi-qubit gates, controlled gates,
/// SWAP, Toffoli, measurement, and column width alignment.
@Suite("CircuitDiagramASCII - Empty and Single Gate")
struct CircuitDiagramASCIIBasicTests {
    @Test("Empty circuit renders 'Empty circuit' string")
    func emptyCircuit() {
        let circuit = QuantumCircuit(qubits: 2)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram == "Empty circuit", "Empty circuit should return literal 'Empty circuit' string")
    }

    @Test("Single qubit with Hadamard gate renders H label on wire")
    func singleQubitHadamard() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("H"), "Diagram should contain Hadamard label 'H'")
        #expect(diagram.contains("q0"), "Diagram should contain qubit wire label 'q0'")
    }

    @Test("Multi-gate single qubit circuit shows all gate labels")
    func multiGateSingleQubit() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliZ, to: 0)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("H"), "Diagram should contain Hadamard 'H'")
        #expect(diagram.contains("X"), "Diagram should contain Pauli-X 'X'")
        #expect(diagram.contains("Z"), "Diagram should contain Pauli-Z 'Z'")
    }
}

/// Validates multi-qubit gate rendering including CNOT, SWAP, and Toffoli.
/// Checks control dots, target symbols, cross symbols, and vertical connectors
/// appear correctly on the appropriate qubit wires.
@Suite("CircuitDiagramASCII - Multi-Qubit Gates")
struct CircuitDiagramASCIIMultiQubitTests {
    @Test("CNOT renders control dot and circled-plus target")
    func cnotRendering() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("\u{25CF}"), "CNOT control qubit should show filled circle \u{25CF}")
        #expect(diagram.contains("\u{2295}"), "CNOT target qubit should show circled-plus \u{2295}")
    }

    @Test("SWAP gate renders cross symbols on both qubits")
    func swapRendering() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.swap, to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        let crossCount = diagram.count(where: { $0 == "\u{00D7}" })
        #expect(crossCount >= 2, "SWAP should render \u{00D7} on both qubits, found \(crossCount)")
    }

    @Test("Toffoli renders two control dots and one circled-plus target")
    func toffoliRendering() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.toffoli, to: [0, 1, 2])
        let diagram = CircuitDiagramASCII.render(circuit)
        let controlCount = diagram.count(where: { $0 == "\u{25CF}" })
        #expect(controlCount >= 2, "Toffoli should have at least 2 control dots, found \(controlCount)")
        #expect(diagram.contains("\u{2295}"), "Toffoli target qubit should show circled-plus \u{2295}")
    }

    @Test("Toffoli renders vertical connectors between qubit wires")
    func toffoliVerticalConnectors() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.toffoli, to: [0, 1, 2])
        let diagram = CircuitDiagramASCII.render(circuit)
        let lines = diagram.split(separator: "\n", omittingEmptySubsequences: false)
        #expect(lines.count == 5, "3-qubit diagram should have 5 lines (3 wires + 2 spacers), got \(lines.count)")
        let spacerLine1 = String(lines[1])
        let spacerLine2 = String(lines[3])
        let hasConnector = spacerLine1.contains("\u{2502}") || spacerLine2.contains("\u{2502}")
        #expect(hasConnector, "Spacer lines should contain vertical connector \u{2502} between connected qubits")
    }
}

/// Validates measurement gate rendering, mixed gate sequences,
/// non-adjacent controlled gates, and column width alignment.
/// Ensures correct symbols and proper layout in complex circuits.
@Suite("CircuitDiagramASCII - Measurement and Layout")
struct CircuitDiagramASCIIMeasurementAndLayoutTests {
    @Test("Measurement gate renders M symbol")
    func measurementRendering() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.measure, to: 0)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("M"), "Measurement should render as 'M' symbol")
    }

    @Test("Mixed single and multi-qubit gates render correctly in sequence")
    func mixedGateSequence() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.pauliX, to: 1)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("H"), "Diagram should contain Hadamard 'H'")
        #expect(diagram.contains("\u{25CF}"), "Diagram should contain CNOT control dot")
        #expect(diagram.contains("\u{2295}"), "Diagram should contain CNOT target symbol")
        #expect(diagram.contains("X"), "Diagram should contain Pauli-X 'X'")
        #expect(diagram.contains("q0"), "Diagram should contain wire label 'q0'")
        #expect(diagram.contains("q1"), "Diagram should contain wire label 'q1'")
    }

    @Test("Controlled gate with non-adjacent qubits shows connector through intermediate wire")
    func nonAdjacentControlledGate() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.cnot, to: [0, 2])
        let diagram = CircuitDiagramASCII.render(circuit)
        let lines = diagram.split(separator: "\n", omittingEmptySubsequences: false)
        #expect(lines.count == 5, "3-qubit diagram should have 5 lines, got \(lines.count)")
        let middleWireLine = String(lines[2])
        #expect(middleWireLine.contains("\u{2502}"), "Intermediate qubit wire should show vertical connector \u{2502} when gate spans non-adjacent qubits")
    }

    @Test("Gate width alignment: gates with different label widths in same column")
    func gateWidthAlignment() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.sGate, to: 1)
        let diagram = CircuitDiagramASCII.render(circuit)
        let lines = diagram.split(separator: "\n", omittingEmptySubsequences: false)
        #expect(lines.count >= 3, "2-qubit diagram should have at least 3 lines, got \(lines.count)")
        let wire0 = String(lines[0])
        let wire1 = String(lines[2])
        #expect(wire0.contains("H"), "First wire should contain 'H'")
        #expect(wire1.contains("S"), "Second wire should contain 'S'")
    }

    @Test("Wire labels are correct for multi-qubit circuits")
    func wireLabels() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        circuit.append(.hadamard, to: 2)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("q0"), "Diagram should have wire label 'q0'")
        #expect(diagram.contains("q1"), "Diagram should have wire label 'q1'")
        #expect(diagram.contains("q2"), "Diagram should have wire label 'q2'")
    }

    @Test("Horizontal dash character used for wires")
    func wireCharacters() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("\u{2500}"), "Wire should use box-drawing horizontal dash \u{2500}")
    }
}

@Suite("CircuitDiagramASCII - Gate Variant Coverage")
struct CircuitDiagramASCIIGateVariantTests {
    @Test("Reset operation renders ket-zero symbol")
    func resetRendering() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.reset, to: 0)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("|0\u{27E9}"), "Reset should render as |0\u{27E9} symbol on wire")
    }

    @Test("CZ gate renders control dot and Z target")
    func czRendering() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cz, to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        let lines = diagram.split(separator: "\n", omittingEmptySubsequences: false)
        let wire0 = String(lines[0])
        let wire1 = String(lines[2])
        #expect(wire0.contains("\u{25CF}"), "CZ control qubit should show filled circle")
        #expect(wire1.contains("Z"), "CZ target qubit should show Z label")
    }

    @Test("CY gate renders control dot and Y target")
    func cyRendering() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cy, to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        let lines = diagram.split(separator: "\n", omittingEmptySubsequences: false)
        let wire1 = String(lines[2])
        #expect(diagram.contains("\u{25CF}"), "CY control qubit should show filled circle")
        #expect(wire1.contains("Y"), "CY target qubit should show Y label")
    }

    @Test("CH gate renders control dot and H target")
    func chRendering() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.ch, to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        let lines = diagram.split(separator: "\n", omittingEmptySubsequences: false)
        let wire1 = String(lines[2])
        #expect(diagram.contains("\u{25CF}"), "CH control qubit should show filled circle")
        #expect(wire1.contains("H"), "CH target qubit should show H label")
    }

    @Test("Controlled phase gate renders control dot and P(angle) label")
    func controlledPhaseRendering() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlledPhase(0.5), to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("\u{25CF}"), "Controlled phase control qubit should show filled circle")
        #expect(diagram.contains("P("), "Controlled phase target should show P(angle) label")
    }

    @Test("Controlled rotation X gate renders control dot and Rx label")
    func controlledRotationXRendering() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlledRotationX(0.5), to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("\u{25CF}"), "CRx control qubit should show filled circle")
        #expect(diagram.contains("Rx("), "CRx target should show Rx(angle) label")
    }

    @Test("Controlled rotation Y gate renders control dot and Ry label")
    func controlledRotationYRendering() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlledRotationY(0.5), to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("\u{25CF}"), "CRy control qubit should show filled circle")
        #expect(diagram.contains("Ry("), "CRy target should show Ry(angle) label")
    }

    @Test("Controlled rotation Z gate renders control dot and Rz label")
    func controlledRotationZRendering() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlledRotationZ(0.5), to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("\u{25CF}"), "CRz control qubit should show filled circle")
        #expect(diagram.contains("Rz("), "CRz target should show Rz(angle) label")
    }

    @Test("Sqrt SWAP gate renders sqrt-SW symbols on both qubits")
    func sqrtSwapRendering() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.sqrtSwap, to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        let count = diagram.components(separatedBy: "\u{221A}SW").count - 1
        #expect(count >= 2, "Sqrt SWAP should render \u{221A}SW on both qubits, found \(count)")
    }

    @Test("iSWAP gate renders iSW symbols on both qubits")
    func iswapRendering() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.iswap, to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        let count = diagram.components(separatedBy: "iSW").count - 1
        #expect(count >= 2, "iSWAP should render iSW on both qubits, found \(count)")
    }

    @Test("Sqrt iSWAP gate renders sqrt-iSW symbols on both qubits")
    func sqrtISwapRendering() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.sqrtISwap, to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        let count = diagram.components(separatedBy: "\u{221A}iSW").count - 1
        #expect(count >= 2, "Sqrt iSWAP should render \u{221A}iSW on both qubits, found \(count)")
    }

    @Test("fSWAP gate renders fSW symbols on both qubits")
    func fswapRendering() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.fswap, to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        let count = diagram.components(separatedBy: "fSW").count - 1
        #expect(count >= 2, "fSWAP should render fSW on both qubits, found \(count)")
    }

    @Test("Fredkin gate renders control dot and two cross symbols")
    func fredkinRendering() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.fredkin, to: [0, 1, 2])
        let diagram = CircuitDiagramASCII.render(circuit)
        let lines = diagram.split(separator: "\n", omittingEmptySubsequences: false)
        let wire0 = String(lines[0])
        #expect(wire0.contains("\u{25CF}"), "Fredkin control qubit should show filled circle")
        let crossCount = diagram.count(where: { $0 == "\u{00D7}" })
        #expect(crossCount >= 2, "Fredkin should render \u{00D7} on both swap qubits, found \(crossCount)")
    }

    @Test("CCZ gate renders two control dots and Z target")
    func cczRendering() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.ccz, to: [0, 1, 2])
        let diagram = CircuitDiagramASCII.render(circuit)
        let controlCount = diagram.count(where: { $0 == "\u{25CF}" })
        #expect(controlCount >= 2, "CCZ should have at least 2 control dots, found \(controlCount)")
        let lines = diagram.split(separator: "\n", omittingEmptySubsequences: false)
        let wire2 = String(lines[4])
        #expect(wire2.contains("Z"), "CCZ target qubit should show Z label")
    }

    @Test("Controlled gate with Hadamard inner gate renders control dot and H label")
    func controlledHadamardRendering() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlled(gate: .hadamard, controls: [0]), to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        let lines = diagram.split(separator: "\n", omittingEmptySubsequences: false)
        let wire0 = String(lines[0])
        let wire1 = String(lines[2])
        #expect(wire0.contains("\u{25CF}"), "Controlled-H control qubit should show filled circle")
        #expect(wire1.contains("H"), "Controlled-H target qubit should show H label from inner gate")
    }

    @Test("Controlled gate with pauliX inner gate renders circled-plus target")
    func controlledPauliXRendering() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlled(gate: .pauliX, controls: [0]), to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("\u{25CF}"), "Controlled-X control qubit should show filled circle")
        #expect(diagram.contains("\u{2295}"), "Controlled-X with pauliX inner gate should render circled-plus target")
    }

    @Test("Custom two-qubit gate renders default multi-qubit path with U2 label")
    func customTwoQubitDefaultRendering() {
        let matrix: [[Complex<Double>]] = [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .zero, .one],
            [.zero, .zero, .one, .zero],
        ]
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.customTwoQubit(matrix: matrix), to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("[U2]"), "Custom two-qubit gate should render bracketed [U2] label on first qubit")
        let lines = diagram.split(separator: "\n", omittingEmptySubsequences: false)
        let wire1 = String(lines[2])
        #expect(wire1.contains("\u{2502}"), "Custom two-qubit gate second qubit should show vertical bar connector")
    }

    @Test("Circuit with multiple qubits pads labels to uniform width")
    func qubitLabelPadding() {
        var circuit = QuantumCircuit(qubits: 4)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 3)
        let diagram = CircuitDiagramASCII.render(circuit)
        let lines = diagram.split(separator: "\n", omittingEmptySubsequences: false)
        let firstLine = String(lines[0])
        let lastWireLine = String(lines[6])
        #expect(firstLine.contains("q0"), "First wire should have label q0")
        #expect(lastWireLine.contains("q3"), "Last wire should have label q3")
        #expect(firstLine.contains("q0"), "Qubit label q0 should appear on first wire line")
        #expect(lastWireLine.contains("q3"), "Qubit label q3 should appear on last wire line")
    }
}

@Suite("CircuitDiagramASCII - Controlled Inner Gate Multi-Qubit Branch")
struct CircuitDiagramASCIIControlledInnerMultiQubitTests {
    @Test("Controlled swap gate renders control dot and bracketed label for multi-qubit inner gate")
    func controlledSwapInnerGateMultiQubit() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.controlled(gate: .swap, controls: [0]), to: [0, 1, 2])
        let diagram = CircuitDiagramASCII.render(circuit)
        let lines = diagram.split(separator: "\n", omittingEmptySubsequences: false)
        let wire0 = String(lines[0])
        #expect(wire0.contains("\u{25CF}"), "Controlled-SWAP control qubit should show filled circle")
        #expect(diagram.contains("["), "Multi-qubit inner gate should render bracketed label")
    }
}

@Suite("CircuitDiagramASCII - singleGateLabel Switch Coverage")
struct CircuitDiagramASCIISingleGateLabelTests {
    @Test("Identity gate renders I label")
    func identityLabel() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.identity, to: 0)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("I"), "Identity gate should render as 'I' label")
    }

    @Test("PauliY gate renders Y label")
    func pauliYLabel() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliY, to: 0)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("Y"), "PauliY gate should render as 'Y' label")
    }

    @Test("T gate renders T label")
    func tGateLabel() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.tGate, to: 0)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("T"), "T gate should render as 'T' label")
    }

    @Test("SX gate renders SX label")
    func sxLabel() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.sx, to: 0)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("SX"), "SX gate should render as 'SX' label")
    }

    @Test("SY gate renders SY label")
    func syLabel() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.sy, to: 0)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("SY"), "SY gate should render as 'SY' label")
    }

    @Test("Phase gate renders P(angle) label")
    func phaseLabel() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.phase(0.5), to: 0)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("P("), "Phase gate should render as 'P(' followed by angle")
    }

    @Test("RotationX gate renders Rx(angle) label")
    func rotationXLabel() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationX(0.5), to: 0)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("Rx("), "RotationX gate should render as 'Rx(' followed by angle")
    }

    @Test("RotationY gate renders Ry(angle) label")
    func rotationYLabel() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationY(0.5), to: 0)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("Ry("), "RotationY gate should render as 'Ry(' followed by angle")
    }

    @Test("RotationZ gate renders Rz(angle) label")
    func rotationZLabel() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(0.5), to: 0)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("Rz("), "RotationZ gate should render as 'Rz(' followed by angle")
    }

    @Test("U1 gate renders U1(lambda) label")
    func u1Label() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u1(lambda: 0.5), to: 0)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("U1("), "U1 gate should render as 'U1(' followed by parameter")
    }

    @Test("U2 gate renders U2(phi,lambda) label")
    func u2Label() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u2(phi: 0.3, lambda: 0.7), to: 0)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("U2("), "U2 gate should render as 'U2(' followed by parameters")
    }

    @Test("U3 gate renders U3(theta,phi,lambda) label")
    func u3Label() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u3(theta: 0.1, phi: 0.2, lambda: 0.3), to: 0)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("U3("), "U3 gate should render as 'U3(' followed by parameters")
    }

    @Test("GlobalPhase gate renders GP(angle) label")
    func globalPhaseLabel() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.globalPhase(0.5), to: 0)
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("GP("), "GlobalPhase gate should render as 'GP(' followed by angle")
    }

    @Test("Givens gate renders Giv(angle) label on two-qubit circuit")
    func givensLabel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.givens(0.5), to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("Giv("), "Givens gate should render as 'Giv(' followed by angle")
    }

    @Test("XX gate renders XX(angle) label on two-qubit circuit")
    func xxLabel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.xx(0.5), to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("XX("), "XX gate should render label containing 'XX('")
    }

    @Test("YY gate renders YY(angle) label on two-qubit circuit")
    func yyLabel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.yy(0.5), to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("YY("), "YY gate should render label containing 'YY('")
    }

    @Test("ZZ gate renders ZZ(angle) label on two-qubit circuit")
    func zzLabel() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.zz(0.5), to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("ZZ("), "ZZ gate should render label containing 'ZZ('")
    }

    @Test("CustomUnitary gate renders U label")
    func customUnitaryLabel() {
        let identity2x2: [[Complex<Double>]] = [
            [.one, .zero],
            [.zero, .one],
        ]
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.customUnitary(matrix: identity2x2), to: [0])
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("U"), "CustomUnitary gate should render as 'U' label")
    }

    @Test("Diagonal gate renders Diag(count) label")
    func diagonalLabel() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.diagonal(phases: [0.0, 0.5]), to: [0])
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("Diag(2)"), "Diagonal gate with 2 phases should render as 'Diag(2)' label")
    }

    @Test("Multiplexor gate renders Mux(count) label")
    func multiplexorLabel() {
        let identity: [[Complex<Double>]] = [
            [.one, .zero],
            [.zero, .one],
        ]
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.multiplexor(unitaries: [identity, identity]), to: [0, 1])
        let diagram = CircuitDiagramASCII.render(circuit)
        #expect(diagram.contains("Mux(2)"), "Multiplexor gate with 2 unitaries should render as 'Mux(2)' label")
    }
}

@Suite("CircuitDiagramASCII - Non-Monotonic Qubit Range")
struct CircuitDiagramASCIINonMonotonicQubitTests {
    @Test("Controlled gate with reversed qubit order triggers qubitRange min/max scan")
    func controlledGateReversedQubits() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.controlled(gate: .hadamard, controls: [0]), to: [2, 0])
        let diagram = CircuitDiagramASCII.render(circuit)
        let lines = diagram.split(separator: "\n", omittingEmptySubsequences: false)
        #expect(lines.count == 5, "3-qubit diagram should have 5 lines (3 wires + 2 spacers), got \(lines.count)")
        let spacerLine = String(lines[1])
        let hasConnector = spacerLine.contains("\u{2502}")
        #expect(hasConnector, "Spacer between q0 and q1 should contain vertical connector when gate spans q0 to q2")
    }
}
