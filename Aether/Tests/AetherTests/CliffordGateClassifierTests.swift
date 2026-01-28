// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Testing

/// Validates CliffordGateClassifier gate classification behavior.
/// Tests classify() for Clifford vs non-Clifford gates, isClifford() Bool returns,
/// and analyze() circuit-level Clifford detection with T-count accumulation.
@Suite("Clifford Gate Classifier - classify()")
struct CliffordGateClassifierClassifyTests {
    @Test("Hadamard gate classifies as Clifford")
    func hadamardIsClifford() {
        let result = CliffordGateClassifier.classify(.hadamard)
        #expect(result == .clifford, "Hadamard gate should be classified as Clifford")
    }

    @Test("S gate classifies as Clifford")
    func sGateIsClifford() {
        let result = CliffordGateClassifier.classify(.sGate)
        #expect(result == .clifford, "S gate should be classified as Clifford")
    }

    @Test("CNOT gate classifies as Clifford")
    func cnotIsClifford() {
        let result = CliffordGateClassifier.classify(.cnot)
        #expect(result == .clifford, "CNOT gate should be classified as Clifford")
    }

    @Test("CZ gate classifies as Clifford")
    func czIsClifford() {
        let result = CliffordGateClassifier.classify(.cz)
        #expect(result == .clifford, "CZ gate should be classified as Clifford")
    }

    @Test("Pauli-X gate classifies as Clifford")
    func pauliXIsClifford() {
        let result = CliffordGateClassifier.classify(.pauliX)
        #expect(result == .clifford, "Pauli-X gate should be classified as Clifford")
    }

    @Test("Pauli-Y gate classifies as Clifford")
    func pauliYIsClifford() {
        let result = CliffordGateClassifier.classify(.pauliY)
        #expect(result == .clifford, "Pauli-Y gate should be classified as Clifford")
    }

    @Test("Pauli-Z gate classifies as Clifford")
    func pauliZIsClifford() {
        let result = CliffordGateClassifier.classify(.pauliZ)
        #expect(result == .clifford, "Pauli-Z gate should be classified as Clifford")
    }

    @Test("Identity gate classifies as Clifford")
    func identityIsClifford() {
        let result = CliffordGateClassifier.classify(.identity)
        #expect(result == .clifford, "Identity gate should be classified as Clifford")
    }

    @Test("SWAP gate classifies as Clifford")
    func swapIsClifford() {
        let result = CliffordGateClassifier.classify(.swap)
        #expect(result == .clifford, "SWAP gate should be classified as Clifford")
    }

    @Test("iSWAP gate classifies as Clifford")
    func iswapIsClifford() {
        let result = CliffordGateClassifier.classify(.iswap)
        #expect(result == .clifford, "iSWAP gate should be classified as Clifford")
    }

    @Test("T gate classifies as non-Clifford with tCount 1")
    func tGateIsNonClifford() {
        let result = CliffordGateClassifier.classify(.tGate)
        #expect(result == .nonClifford(tCount: 1), "T gate should be classified as non-Clifford with tCount 1")
    }

    @Test("Toffoli gate classifies as non-Clifford with tCount 7")
    func toffoliIsNonClifford() {
        let result = CliffordGateClassifier.classify(.toffoli)
        #expect(result == .nonClifford(tCount: 7), "Toffoli gate should be classified as non-Clifford with tCount 7")
    }
}

/// Validates isClifford() convenience method for Boolean Clifford checks.
/// Ensures correct true/false returns matching classify() results
/// for common Clifford and non-Clifford gate types.
@Suite("Clifford Gate Classifier - isClifford()")
struct CliffordGateClassifierIsCliffordTests {
    @Test("isClifford returns true for Hadamard")
    func isCliffordHadamard() {
        let result = CliffordGateClassifier.isClifford(.hadamard)
        #expect(result == true, "isClifford should return true for Hadamard gate")
    }

    @Test("isClifford returns true for CNOT")
    func isCliffordCnot() {
        let result = CliffordGateClassifier.isClifford(.cnot)
        #expect(result == true, "isClifford should return true for CNOT gate")
    }

    @Test("isClifford returns true for Pauli gates")
    func isCliffordPauliGates() {
        #expect(CliffordGateClassifier.isClifford(.pauliX) == true, "isClifford should return true for Pauli-X")
        #expect(CliffordGateClassifier.isClifford(.pauliY) == true, "isClifford should return true for Pauli-Y")
        #expect(CliffordGateClassifier.isClifford(.pauliZ) == true, "isClifford should return true for Pauli-Z")
    }

    @Test("isClifford returns false for T gate")
    func isCliffordTGate() {
        let result = CliffordGateClassifier.isClifford(.tGate)
        #expect(result == false, "isClifford should return false for T gate")
    }

    @Test("isClifford returns false for Toffoli")
    func isCliffordToffoli() {
        let result = CliffordGateClassifier.isClifford(.toffoli)
        #expect(result == false, "isClifford should return false for Toffoli gate")
    }

    @Test("isClifford returns true for S gate")
    func isCliffordSGate() {
        let result = CliffordGateClassifier.isClifford(.sGate)
        #expect(result == true, "isClifford should return true for S gate")
    }

    @Test("isClifford returns true for CZ gate")
    func isCliffordCzGate() {
        let result = CliffordGateClassifier.isClifford(.cz)
        #expect(result == true, "isClifford should return true for CZ gate")
    }
}

/// Validates analyze() circuit-level Clifford analysis.
/// Tests isClifford flag and tCount accumulation for pure Clifford circuits,
/// mixed circuits, and circuits with multiple non-Clifford gates.
@Suite("Clifford Gate Classifier - analyze()")
struct CliffordGateClassifierAnalyzeTests {
    @Test("Pure Clifford circuit returns isClifford true and tCount 0")
    func pureCliffordCircuit() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.pauliX, to: 1)
        circuit.append(.sGate, to: 0)

        let result = CliffordGateClassifier.analyze(circuit)
        #expect(result.isClifford == true, "Pure Clifford circuit should have isClifford true")
        #expect(result.tCount == 0, "Pure Clifford circuit should have tCount 0")
    }

    @Test("Circuit with single T gate returns isClifford false and tCount 1")
    func singleTGateCircuit() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.tGate, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let result = CliffordGateClassifier.analyze(circuit)
        #expect(result.isClifford == false, "Circuit with T gate should have isClifford false")
        #expect(result.tCount == 1, "Circuit with single T gate should have tCount 1")
    }

    @Test("Circuit with multiple T gates accumulates tCount")
    func multipleTGatesCircuit() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.tGate, to: 0)
        circuit.append(.hadamard, to: 1)
        circuit.append(.tGate, to: 1)
        circuit.append(.tGate, to: 0)

        let result = CliffordGateClassifier.analyze(circuit)
        #expect(result.isClifford == false, "Circuit with T gates should have isClifford false")
        #expect(result.tCount == 3, "Circuit with three T gates should have tCount 3")
    }

    @Test("Empty circuit returns isClifford true and tCount 0")
    func emptyCircuit() {
        let circuit = QuantumCircuit(qubits: 2)

        let result = CliffordGateClassifier.analyze(circuit)
        #expect(result.isClifford == true, "Empty circuit should have isClifford true")
        #expect(result.tCount == 0, "Empty circuit should have tCount 0")
    }

    @Test("Circuit with Toffoli returns isClifford false and tCount 7")
    func toffoliCircuit() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.toffoli, to: [0, 1, 2])

        let result = CliffordGateClassifier.analyze(circuit)
        #expect(result.isClifford == false, "Circuit with Toffoli should have isClifford false")
        #expect(result.tCount == 7, "Toffoli gate contributes tCount of 7")
    }

    @Test("Mixed circuit accumulates T-count from multiple sources")
    func mixedCircuit() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.tGate, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.tGate, to: 1)

        let result = CliffordGateClassifier.analyze(circuit)
        #expect(result.isClifford == false, "Mixed circuit should have isClifford false")
        #expect(result.tCount == 2, "Mixed circuit with two T gates should have tCount 2")
    }
}

@Suite("Clifford Gate Classifier - Extended Gate Coverage")
struct CliffordGateClassifierExtendedTests {
    @Test("Fredkin gate classifies as non-Clifford with tCount 7")
    func classifyFredkinGate() async {
        let classification = CliffordGateClassifier.classify(.fredkin)
        #expect(classification == .nonClifford(tCount: 7), "Fredkin should be non-Clifford with tCount 7")
    }

    @Test("CCZ gate classifies as non-Clifford with tCount 7")
    func classifyCczGate() async {
        let classification = CliffordGateClassifier.classify(.ccz)
        #expect(classification == .nonClifford(tCount: 7), "CCZ should be non-Clifford with tCount 7")
    }

    @Test("Phase with non-Clifford angle classifies as non-Clifford")
    func classifyPhaseNonCliffordAngle() async {
        let classification = CliffordGateClassifier.classify(.phase(.value(.pi / 5)))
        #expect(classification == .nonClifford(tCount: 3), "Phase(pi/5) should be non-Clifford")
    }

    @Test("RotationX with non-Clifford angle classifies as non-Clifford")
    func classifyRotationXNonCliffordAngle() async {
        let classification = CliffordGateClassifier.classify(.rotationX(.value(.pi / 5)))
        #expect(classification == .nonClifford(tCount: 3), "RotationX(pi/5) should be non-Clifford")
    }

    @Test("RotationY with non-Clifford angle classifies as non-Clifford")
    func classifyRotationYNonCliffordAngle() async {
        let classification = CliffordGateClassifier.classify(.rotationY(.value(.pi / 5)))
        #expect(classification == .nonClifford(tCount: 3), "RotationY(pi/5) should be non-Clifford")
    }

    @Test("RotationZ with non-Clifford angle classifies as non-Clifford")
    func classifyRotationZNonCliffordAngle() async {
        let classification = CliffordGateClassifier.classify(.rotationZ(.value(.pi / 5)))
        #expect(classification == .nonClifford(tCount: 3), "RotationZ(pi/5) should be non-Clifford")
    }

    @Test("U1 gate with Clifford angle classifies as Clifford")
    func classifyU1CliffordAngle() async {
        let classification = CliffordGateClassifier.classify(.u1(lambda: .value(.pi / 2)))
        #expect(classification == .clifford, "U1(pi/2) should be Clifford")
    }

    @Test("U1 gate with non-Clifford angle classifies as non-Clifford")
    func classifyU1NonCliffordAngle() async {
        let classification = CliffordGateClassifier.classify(.u1(lambda: .value(.pi / 5)))
        #expect(classification == .nonClifford(tCount: 3), "U1(pi/5) should be non-Clifford")
    }

    @Test("U2 gate with Clifford angles classifies as Clifford")
    func classifyU2CliffordAngles() async {
        let classification = CliffordGateClassifier.classify(.u2(phi: .value(0), lambda: .value(.pi)))
        #expect(classification == .clifford, "U2(0, pi) should be Clifford")
    }

    @Test("U2 gate with non-Clifford angles classifies as non-Clifford")
    func classifyU2NonCliffordAngles() async {
        let classification = CliffordGateClassifier.classify(.u2(phi: .value(.pi / 5), lambda: .value(.pi / 3)))
        #expect(classification == .nonClifford(tCount: 6), "U2 with non-Clifford angles should be non-Clifford")
    }

    @Test("U3 gate with Clifford angles classifies as Clifford")
    func classifyU3CliffordAngles() async {
        let classification = CliffordGateClassifier.classify(.u3(theta: .value(.pi), phi: .value(0), lambda: .value(.pi / 2)))
        #expect(classification == .clifford, "U3(pi, 0, pi/2) should be Clifford")
    }

    @Test("U3 gate with non-Clifford angles classifies as non-Clifford")
    func classifyU3NonCliffordAngles() async {
        let classification = CliffordGateClassifier.classify(.u3(theta: .value(.pi / 5), phi: .value(.pi / 3), lambda: .value(.pi / 7)))
        #expect(classification == .nonClifford(tCount: 9), "U3 with non-Clifford angles should be non-Clifford")
    }

    @Test("ControlledPhase with Clifford angle classifies as Clifford")
    func classifyControlledPhaseClifford() async {
        let classification = CliffordGateClassifier.classify(.controlledPhase(.value(.pi)))
        #expect(classification == .clifford, "ControlledPhase(pi) should be Clifford")
    }

    @Test("ControlledPhase with non-Clifford angle classifies as non-Clifford")
    func classifyControlledPhaseNonClifford() async {
        let classification = CliffordGateClassifier.classify(.controlledPhase(.value(.pi / 5)))
        #expect(classification == .nonClifford(tCount: 3), "ControlledPhase(pi/5) should be non-Clifford")
    }

    @Test("ControlledRotationX with non-Clifford angle classifies as non-Clifford")
    func classifyControlledRotationXNonClifford() async {
        let classification = CliffordGateClassifier.classify(.controlledRotationX(.value(.pi / 5)))
        #expect(classification == .nonClifford(tCount: 3), "ControlledRotationX(pi/5) should be non-Clifford")
    }

    @Test("ControlledRotationY with non-Clifford angle classifies as non-Clifford")
    func classifyControlledRotationYNonClifford() async {
        let classification = CliffordGateClassifier.classify(.controlledRotationY(.value(.pi / 5)))
        #expect(classification == .nonClifford(tCount: 3), "ControlledRotationY(pi/5) should be non-Clifford")
    }

    @Test("ControlledRotationZ with non-Clifford angle classifies as non-Clifford")
    func classifyControlledRotationZNonClifford() async {
        let classification = CliffordGateClassifier.classify(.controlledRotationZ(.value(.pi / 5)))
        #expect(classification == .nonClifford(tCount: 3), "ControlledRotationZ(pi/5) should be non-Clifford")
    }

    @Test("SqrtSwap gate classifies as non-Clifford with tCount 1")
    func classifySqrtSwapGate() async {
        let classification = CliffordGateClassifier.classify(.sqrtSwap)
        #expect(classification == .nonClifford(tCount: 1), "SqrtSwap should be non-Clifford with tCount 1")
    }

    @Test("SqrtISwap gate classifies as non-Clifford with tCount 1")
    func classifySqrtISwapGate() async {
        let classification = CliffordGateClassifier.classify(.sqrtISwap)
        #expect(classification == .nonClifford(tCount: 1), "SqrtISwap should be non-Clifford with tCount 1")
    }

    @Test("FSwap gate classifies as non-Clifford with tCount 1")
    func classifyFswapGate() async {
        let classification = CliffordGateClassifier.classify(.fswap)
        #expect(classification == .nonClifford(tCount: 1), "FSwap should be non-Clifford with tCount 1")
    }

    @Test("Givens gate with Clifford angle classifies as Clifford")
    func classifyGivensClifford() async {
        let classification = CliffordGateClassifier.classify(.givens(.value(.pi / 2)))
        #expect(classification == .clifford, "Givens(pi/2) should be Clifford")
    }

    @Test("Givens gate with non-Clifford angle classifies as non-Clifford")
    func classifyGivensNonClifford() async {
        let classification = CliffordGateClassifier.classify(.givens(.value(.pi / 5)))
        #expect(classification == .nonClifford(tCount: 3), "Givens(pi/5) should be non-Clifford")
    }

    @Test("XX gate with Clifford angle classifies as Clifford")
    func classifyXXClifford() async {
        let classification = CliffordGateClassifier.classify(.xx(.value(.pi)))
        #expect(classification == .clifford, "XX(pi) should be Clifford")
    }

    @Test("XX gate with non-Clifford angle classifies as non-Clifford")
    func classifyXXNonClifford() async {
        let classification = CliffordGateClassifier.classify(.xx(.value(.pi / 5)))
        #expect(classification == .nonClifford(tCount: 3), "XX(pi/5) should be non-Clifford")
    }

    @Test("YY gate with Clifford angle classifies as Clifford")
    func classifyYYClifford() async {
        let classification = CliffordGateClassifier.classify(.yy(.value(.pi / 2)))
        #expect(classification == .clifford, "YY(pi/2) should be Clifford")
    }

    @Test("YY gate with non-Clifford angle classifies as non-Clifford")
    func classifyYYNonClifford() async {
        let classification = CliffordGateClassifier.classify(.yy(.value(.pi / 5)))
        #expect(classification == .nonClifford(tCount: 3), "YY(pi/5) should be non-Clifford")
    }

    @Test("ZZ gate with Clifford angle classifies as Clifford")
    func classifyZZClifford() async {
        let classification = CliffordGateClassifier.classify(.zz(.value(.pi)))
        #expect(classification == .clifford, "ZZ(pi) should be Clifford")
    }

    @Test("ZZ gate with non-Clifford angle classifies as non-Clifford")
    func classifyZZNonClifford() async {
        let classification = CliffordGateClassifier.classify(.zz(.value(.pi / 5)))
        #expect(classification == .nonClifford(tCount: 3), "ZZ(pi/5) should be non-Clifford")
    }

    @Test("Diagonal gate with Clifford phases classifies as Clifford")
    func classifyDiagonalClifford() async {
        let classification = CliffordGateClassifier.classify(.diagonal(phases: [0, .pi / 2, .pi, 3 * .pi / 2]))
        #expect(classification == .clifford, "Diagonal with Clifford phases should be Clifford")
    }

    @Test("Diagonal gate with non-Clifford phases classifies as non-Clifford")
    func classifyDiagonalNonClifford() async {
        let classification = CliffordGateClassifier.classify(.diagonal(phases: [0, .pi / 5, .pi / 3, .pi / 7]))
        #expect(classification == .nonClifford(tCount: 9), "Diagonal with non-Clifford phases should be non-Clifford")
    }

    @Test("Multiplexor with Clifford unitaries classifies as Clifford")
    func classifyMultiplexorClifford() async {
        let hadamard: [[Complex<Double>]] = [
            [Complex(1.0 / 2.0.squareRoot(), 0), Complex(1.0 / 2.0.squareRoot(), 0)],
            [Complex(1.0 / 2.0.squareRoot(), 0), Complex(-1.0 / 2.0.squareRoot(), 0)],
        ]
        let pauliX: [[Complex<Double>]] = [
            [.zero, .one],
            [.one, .zero],
        ]
        let classification = CliffordGateClassifier.classify(.multiplexor(unitaries: [hadamard, pauliX]))
        #expect(classification == .clifford, "Multiplexor with Clifford unitaries should be Clifford")
    }

    @Test("Multiplexor with non-Clifford unitaries classifies as non-Clifford")
    func classifyMultiplexorNonClifford() async {
        let nonClifford: [[Complex<Double>]] = [
            [Complex(0.9, 0), Complex(0.1, 0)],
            [Complex(0.1, 0), Complex(0.9, 0)],
        ]
        let classification = CliffordGateClassifier.classify(.multiplexor(unitaries: [nonClifford, nonClifford]))
        #expect(classification != .clifford, "Multiplexor with non-Clifford unitaries should be non-Clifford")
    }

    @Test("Controlled Clifford gate with Pauli-X classifies as Clifford")
    func classifyControlledPauliX() async {
        let classification = CliffordGateClassifier.classify(.controlled(gate: .pauliX, controls: [0]))
        #expect(classification == .clifford, "Controlled Pauli-X should be Clifford")
    }

    @Test("Controlled Clifford gate with Hadamard classifies as Clifford")
    func classifyControlledHadamard() async {
        let classification = CliffordGateClassifier.classify(.controlled(gate: .hadamard, controls: [0]))
        #expect(classification == .clifford, "Controlled Hadamard should be Clifford")
    }

    @Test("Controlled non-Clifford gate classifies as non-Clifford")
    func classifyControlledNonClifford() async {
        let classification = CliffordGateClassifier.classify(.controlled(gate: .tGate, controls: [0]))
        #expect(classification == .nonClifford(tCount: 1), "Controlled T gate should be non-Clifford")
    }

    @Test("CustomSingleQubit Clifford matrix classifies as Clifford")
    func classifyCustomSingleQubitClifford() async {
        let pauliZ: [[Complex<Double>]] = [
            [.one, .zero],
            [.zero, Complex(-1.0, 0.0)],
        ]
        let classification = CliffordGateClassifier.classify(.customSingleQubit(matrix: pauliZ))
        #expect(classification == .clifford, "Custom Pauli-Z matrix should be Clifford")
    }

    @Test("CustomSingleQubit non-Clifford matrix classifies as non-Clifford")
    func classifyCustomSingleQubitNonClifford() async {
        let nonClifford: [[Complex<Double>]] = [
            [Complex(0.9, 0), Complex(0.1, 0.2)],
            [Complex(0.1, -0.2), Complex(0.9, 0)],
        ]
        let classification = CliffordGateClassifier.classify(.customSingleQubit(matrix: nonClifford))
        #expect(classification == .nonClifford(tCount: 1), "Custom non-Clifford matrix should be non-Clifford")
    }

    @Test("CustomTwoQubit matrix classifies as non-Clifford")
    func classifyCustomTwoQubitNonClifford() async {
        let nonClifford: [[Complex<Double>]] = [
            [Complex(0.9, 0), .zero, .zero, Complex(0.1, 0)],
            [.zero, Complex(0.8, 0), Complex(0.2, 0), .zero],
            [.zero, Complex(0.2, 0), Complex(0.8, 0), .zero],
            [Complex(0.1, 0), .zero, .zero, Complex(0.9, 0)],
        ]
        let classification = CliffordGateClassifier.classify(.customTwoQubit(matrix: nonClifford))
        #expect(classification == .nonClifford(tCount: 4), "Custom 4x4 non-Clifford matrix should be non-Clifford")
    }
}
