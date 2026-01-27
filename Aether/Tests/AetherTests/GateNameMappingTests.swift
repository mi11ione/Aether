// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Validates QASMVersion enum cases exist and are distinct.
/// Ensures both v2 and v3 variants are available for gate
/// name resolution across QASM language versions.
@Suite("QASMVersion Enum")
struct QASMVersionTests {
    @Test("QASMVersion has distinct v2 and v3 cases")
    func versionCasesAreDistinct() {
        let v2: QASMVersion = .v2
        let v3: QASMVersion = .v3
        #expect(v2 != v3, "QASMVersion.v2 and .v3 must be distinct enum cases")
    }
}

/// Validates forward mapping from QuantumGate to QASM name strings.
/// Covers single-qubit, two-qubit, multi-qubit, and parameterized gates
/// for both QASM v2 and v3 language versions.
@Suite("Forward Mapping: qasmName(for:version:)")
struct QASMNameForwardMappingTests {
    @Test("Identity gate maps to 'id'")
    func identityName() {
        let name = GateNameMapping.qasmName(for: .identity, version: .v2)
        #expect(name == "id", "identity gate should map to 'id', got '\(name)'")
    }

    @Test("Pauli X gate maps to 'x'")
    func pauliXName() {
        let name = GateNameMapping.qasmName(for: .pauliX, version: .v3)
        #expect(name == "x", "pauliX gate should map to 'x', got '\(name)'")
    }

    @Test("Pauli Y gate maps to 'y'")
    func pauliYName() {
        let name = GateNameMapping.qasmName(for: .pauliY, version: .v2)
        #expect(name == "y", "pauliY gate should map to 'y', got '\(name)'")
    }

    @Test("Pauli Z gate maps to 'z'")
    func pauliZName() {
        let name = GateNameMapping.qasmName(for: .pauliZ, version: .v3)
        #expect(name == "z", "pauliZ gate should map to 'z', got '\(name)'")
    }

    @Test("Hadamard gate maps to 'h'")
    func hadamardName() {
        let name = GateNameMapping.qasmName(for: .hadamard, version: .v2)
        #expect(name == "h", "hadamard gate should map to 'h', got '\(name)'")
    }

    @Test("S gate maps to 's'")
    func sGateName() {
        let name = GateNameMapping.qasmName(for: .sGate, version: .v3)
        #expect(name == "s", "sGate should map to 's', got '\(name)'")
    }

    @Test("T gate maps to 't'")
    func tGateName() {
        let name = GateNameMapping.qasmName(for: .tGate, version: .v2)
        #expect(name == "t", "tGate should map to 't', got '\(name)'")
    }

    @Test("SX gate maps to 'sx'")
    func sxGateName() {
        let name = GateNameMapping.qasmName(for: .sx, version: .v3)
        #expect(name == "sx", "sx gate should map to 'sx', got '\(name)'")
    }

    @Test("SY gate maps to 'sy'")
    func syGateName() {
        let name = GateNameMapping.qasmName(for: .sy, version: .v2)
        #expect(name == "sy", "sy gate should map to 'sy', got '\(name)'")
    }

    @Test("CNOT gate maps to 'cx'")
    func cnotName() {
        let name = GateNameMapping.qasmName(for: .cnot, version: .v2)
        #expect(name == "cx", "cnot gate should map to 'cx', got '\(name)'")
    }

    @Test("CZ gate maps to 'cz'")
    func czName() {
        let name = GateNameMapping.qasmName(for: .cz, version: .v3)
        #expect(name == "cz", "cz gate should map to 'cz', got '\(name)'")
    }

    @Test("CY gate maps to 'cy'")
    func cyName() {
        let name = GateNameMapping.qasmName(for: .cy, version: .v2)
        #expect(name == "cy", "cy gate should map to 'cy', got '\(name)'")
    }

    @Test("CH gate maps to 'ch'")
    func chName() {
        let name = GateNameMapping.qasmName(for: .ch, version: .v3)
        #expect(name == "ch", "ch gate should map to 'ch', got '\(name)'")
    }

    @Test("SWAP gate maps to 'swap'")
    func swapName() {
        let name = GateNameMapping.qasmName(for: .swap, version: .v2)
        #expect(name == "swap", "swap gate should map to 'swap', got '\(name)'")
    }

    @Test("sqrtSwap gate maps to 'sqsw'")
    func sqrtSwapName() {
        let name = GateNameMapping.qasmName(for: .sqrtSwap, version: .v3)
        #expect(name == "sqsw", "sqrtSwap gate should map to 'sqsw', got '\(name)'")
    }

    @Test("iSWAP gate maps to 'iswap'")
    func iswapName() {
        let name = GateNameMapping.qasmName(for: .iswap, version: .v2)
        #expect(name == "iswap", "iswap gate should map to 'iswap', got '\(name)'")
    }

    @Test("sqrtISwap gate maps to 'siswap'")
    func sqrtISwapName() {
        let name = GateNameMapping.qasmName(for: .sqrtISwap, version: .v3)
        #expect(name == "siswap", "sqrtISwap gate should map to 'siswap', got '\(name)'")
    }

    @Test("fSWAP gate maps to 'fswap'")
    func fswapName() {
        let name = GateNameMapping.qasmName(for: .fswap, version: .v2)
        #expect(name == "fswap", "fswap gate should map to 'fswap', got '\(name)'")
    }

    @Test("Toffoli gate maps to 'ccx'")
    func toffoliName() {
        let name = GateNameMapping.qasmName(for: .toffoli, version: .v2)
        #expect(name == "ccx", "toffoli gate should map to 'ccx', got '\(name)'")
    }

    @Test("Fredkin gate maps to 'cswap'")
    func fredkinName() {
        let name = GateNameMapping.qasmName(for: .fredkin, version: .v3)
        #expect(name == "cswap", "fredkin gate should map to 'cswap', got '\(name)'")
    }

    @Test("CCZ gate maps to 'ccz'")
    func cczName() {
        let name = GateNameMapping.qasmName(for: .ccz, version: .v2)
        #expect(name == "ccz", "ccz gate should map to 'ccz', got '\(name)'")
    }

    @Test("Rotation X gate maps to 'rx'")
    func rotationXName() {
        let name = GateNameMapping.qasmName(for: .rotationX(.value(0.5)), version: .v2)
        #expect(name == "rx", "rotationX gate should map to 'rx', got '\(name)'")
    }

    @Test("Rotation Y gate maps to 'ry'")
    func rotationYName() {
        let name = GateNameMapping.qasmName(for: .rotationY(.value(1.0)), version: .v3)
        #expect(name == "ry", "rotationY gate should map to 'ry', got '\(name)'")
    }

    @Test("Rotation Z gate maps to 'rz'")
    func rotationZName() {
        let name = GateNameMapping.qasmName(for: .rotationZ(.value(0.0)), version: .v2)
        #expect(name == "rz", "rotationZ gate should map to 'rz', got '\(name)'")
    }

    @Test("U1 gate maps to 'u1'")
    func u1Name() {
        let name = GateNameMapping.qasmName(for: .u1(lambda: .value(0.0)), version: .v2)
        #expect(name == "u1", "u1 gate should map to 'u1', got '\(name)'")
    }

    @Test("U2 gate maps to 'u2'")
    func u2Name() {
        let name = GateNameMapping.qasmName(for: .u2(phi: .value(0.0), lambda: .value(0.0)), version: .v3)
        #expect(name == "u2", "u2 gate should map to 'u2', got '\(name)'")
    }

    @Test("U3 gate maps to 'u3'")
    func u3Name() {
        let name = GateNameMapping.qasmName(for: .u3(theta: .value(0.0), phi: .value(0.0), lambda: .value(0.0)), version: .v2)
        #expect(name == "u3", "u3 gate should map to 'u3', got '\(name)'")
    }

    @Test("Global phase gate maps to 'gphase'")
    func globalPhaseName() {
        let name = GateNameMapping.qasmName(for: .globalPhase(.value(1.0)), version: .v3)
        #expect(name == "gphase", "globalPhase gate should map to 'gphase', got '\(name)'")
    }

    @Test("Controlled phase gate maps to 'cp'")
    func controlledPhaseName() {
        let name = GateNameMapping.qasmName(for: .controlledPhase(.value(0.5)), version: .v2)
        #expect(name == "cp", "controlledPhase gate should map to 'cp', got '\(name)'")
    }

    @Test("Controlled rotation gates map to crx/cry/crz")
    func controlledRotationNames() {
        let crx = GateNameMapping.qasmName(for: .controlledRotationX(.value(0.0)), version: .v3)
        let cry = GateNameMapping.qasmName(for: .controlledRotationY(.value(0.0)), version: .v3)
        let crz = GateNameMapping.qasmName(for: .controlledRotationZ(.value(0.0)), version: .v3)
        #expect(crx == "crx", "controlledRotationX should map to 'crx', got '\(crx)'")
        #expect(cry == "cry", "controlledRotationY should map to 'cry', got '\(cry)'")
        #expect(crz == "crz", "controlledRotationZ should map to 'crz', got '\(crz)'")
    }

    @Test("Givens gate maps to 'givens'")
    func givensName() {
        let name = GateNameMapping.qasmName(for: .givens(.value(0.0)), version: .v2)
        #expect(name == "givens", "givens gate should map to 'givens', got '\(name)'")
    }

    @Test("XX gate maps to 'rxx'")
    func xxName() {
        let name = GateNameMapping.qasmName(for: .xx(.value(0.0)), version: .v3)
        #expect(name == "rxx", "xx gate should map to 'rxx', got '\(name)'")
    }

    @Test("YY gate maps to 'ryy'")
    func yyName() {
        let name = GateNameMapping.qasmName(for: .yy(.value(0.0)), version: .v2)
        #expect(name == "ryy", "yy gate should map to 'ryy', got '\(name)'")
    }

    @Test("ZZ gate maps to 'rzz'")
    func zzName() {
        let name = GateNameMapping.qasmName(for: .zz(.value(0.0)), version: .v3)
        #expect(name == "rzz", "zz gate should map to 'rzz', got '\(name)'")
    }
}

/// Validates version-dependent gate name differences between QASM v2 and v3.
/// Ensures phase gate produces 'u1' for v2 and 'p' for v3,
/// and controlled gates use version-appropriate prefix syntax.
@Suite("Version-Dependent Name Mapping")
struct QASMVersionDependentTests {
    @Test("Phase gate maps to 'u1' in v2")
    func phaseV2() {
        let name = GateNameMapping.qasmName(for: .phase(.value(0.5)), version: .v2)
        #expect(name == "u1", "phase gate in v2 should map to 'u1', got '\(name)'")
    }

    @Test("Phase gate maps to 'p' in v3")
    func phaseV3() {
        let name = GateNameMapping.qasmName(for: .phase(.value(0.5)), version: .v3)
        #expect(name == "p", "phase gate in v3 should map to 'p', got '\(name)'")
    }

    @Test("Controlled gate uses 'c_' prefix in v2")
    func controlledPrefixV2() {
        let gate = QuantumGate.controlled(gate: .hadamard, controls: [0])
        let name = GateNameMapping.qasmName(for: gate, version: .v2)
        #expect(name == "c_h", "controlled hadamard in v2 should be 'c_h', got '\(name)'")
    }

    @Test("Controlled gate uses 'ctrl_' prefix in v3")
    func controlledPrefixV3() {
        let gate = QuantumGate.controlled(gate: .hadamard, controls: [0])
        let name = GateNameMapping.qasmName(for: gate, version: .v3)
        #expect(name == "ctrl_h", "controlled hadamard in v3 should be 'ctrl_h', got '\(name)'")
    }

    @Test("Controlled phase gate uses version-appropriate inner name")
    func controlledPhaseInnerGateVersioning() {
        let gate = QuantumGate.controlled(gate: .phase(.value(0.0)), controls: [0])
        let nameV2 = GateNameMapping.qasmName(for: gate, version: .v2)
        let nameV3 = GateNameMapping.qasmName(for: gate, version: .v3)
        #expect(nameV2 == "c_u1", "controlled phase in v2 should be 'c_u1', got '\(nameV2)'")
        #expect(nameV3 == "ctrl_p", "controlled phase in v3 should be 'ctrl_p', got '\(nameV3)'")
    }
}

/// Validates reverse mapping from QASM name strings to QuantumGate.
/// Covers standard gate lookups, version-specific entries,
/// and nil return for unknown gate names.
@Suite("Reverse Mapping: gate(forQASMName:version:)")
struct QASMNameReverseMappingTests {
    @Test("Standard non-parameterized gates reverse map correctly (v2)")
    func standardGatesReverseV2() {
        let idGate = GateNameMapping.gate(forQASMName: "id", version: .v2)
        #expect(idGate == .identity, "id should map to .identity, got \(String(describing: idGate))")

        let xGate = GateNameMapping.gate(forQASMName: "x", version: .v2)
        #expect(xGate == .pauliX, "x should map to .pauliX, got \(String(describing: xGate))")

        let hGate = GateNameMapping.gate(forQASMName: "h", version: .v2)
        #expect(hGate == .hadamard, "h should map to .hadamard, got \(String(describing: hGate))")

        let sGate = GateNameMapping.gate(forQASMName: "s", version: .v2)
        #expect(sGate == .sGate, "s should map to .sGate, got \(String(describing: sGate))")

        let tGate = GateNameMapping.gate(forQASMName: "t", version: .v2)
        #expect(tGate == .tGate, "t should map to .tGate, got \(String(describing: tGate))")
    }

    @Test("Two-qubit gates reverse map correctly")
    func twoQubitGatesReverse() {
        let cx = GateNameMapping.gate(forQASMName: "cx", version: .v2)
        #expect(cx == .cnot, "cx should map to .cnot, got \(String(describing: cx))")

        let cz = GateNameMapping.gate(forQASMName: "cz", version: .v3)
        #expect(cz == .cz, "cz should map to .cz, got \(String(describing: cz))")

        let sw = GateNameMapping.gate(forQASMName: "swap", version: .v2)
        #expect(sw == .swap, "swap should map to .swap, got \(String(describing: sw))")

        let sqsw = GateNameMapping.gate(forQASMName: "sqsw", version: .v3)
        #expect(sqsw == .sqrtSwap, "sqsw should map to .sqrtSwap, got \(String(describing: sqsw))")
    }

    @Test("Multi-qubit gates reverse map correctly")
    func multiQubitGatesReverse() {
        let ccx = GateNameMapping.gate(forQASMName: "ccx", version: .v2)
        #expect(ccx == .toffoli, "ccx should map to .toffoli, got \(String(describing: ccx))")

        let cswap = GateNameMapping.gate(forQASMName: "cswap", version: .v3)
        #expect(cswap == .fredkin, "cswap should map to .fredkin, got \(String(describing: cswap))")

        let ccz = GateNameMapping.gate(forQASMName: "ccz", version: .v2)
        #expect(ccz == .ccz, "ccz should map to .ccz, got \(String(describing: ccz))")
    }

    @Test("Parameterized gates reverse map with zero placeholder")
    func parameterizedGatesReversePlaceholder() {
        let placeholder: ParameterValue = .value(0)

        let rx = GateNameMapping.gate(forQASMName: "rx", version: .v2)
        #expect(rx == .rotationX(placeholder), "rx should map to .rotationX(.value(0)), got \(String(describing: rx))")

        let ry = GateNameMapping.gate(forQASMName: "ry", version: .v3)
        #expect(ry == .rotationY(placeholder), "ry should map to .rotationY(.value(0)), got \(String(describing: ry))")

        let cp = GateNameMapping.gate(forQASMName: "cp", version: .v2)
        #expect(cp == .controlledPhase(placeholder), "cp should map to .controlledPhase(.value(0)), got \(String(describing: cp))")
    }

    @Test("v3-specific 'p' maps to phase gate")
    func v3PhaseReverse() {
        let placeholder: ParameterValue = .value(0)
        let pGate = GateNameMapping.gate(forQASMName: "p", version: .v3)
        #expect(pGate == .phase(placeholder), "p in v3 should map to .phase(.value(0)), got \(String(describing: pGate))")
    }

    @Test("v2 'u1' maps to u1 gate")
    func v2U1Reverse() {
        let placeholder: ParameterValue = .value(0)
        let u1Gate = GateNameMapping.gate(forQASMName: "u1", version: .v2)
        #expect(u1Gate == .u1(lambda: placeholder), "u1 in v2 should map to .u1(lambda: .value(0)), got \(String(describing: u1Gate))")
    }

    @Test("Unknown gate name returns nil")
    func unknownNameReturnsNil() {
        let result = GateNameMapping.gate(forQASMName: "nonexistent_gate", version: .v2)
        #expect(result == nil, "unknown gate name should return nil, got \(String(describing: result))")

        let resultV3 = GateNameMapping.gate(forQASMName: "foo_bar_baz", version: .v3)
        #expect(resultV3 == nil, "unknown gate name should return nil in v3, got \(String(describing: resultV3))")
    }

    @Test("Empty string returns nil")
    func emptyStringReturnsNil() {
        let result = GateNameMapping.gate(forQASMName: "", version: .v2)
        #expect(result == nil, "empty string should return nil, got \(String(describing: result))")
    }

    @Test("Case-sensitive lookup: 'H' does not match 'h'")
    func caseSensitiveLookup() {
        let result = GateNameMapping.gate(forQASMName: "H", version: .v2)
        #expect(result == nil, "uppercase 'H' should not match lowercase 'h', got \(String(describing: result))")
    }
}

/// Validates round-trip consistency: qasmName then gate(forQASMName)
/// recovers an equivalent gate for all non-parameterized built-in gates,
/// ensuring forward and reverse mappings are inverses of each other.
@Suite("Round-Trip Mapping Consistency")
struct QASMNameRoundTripTests {
    @Test("Non-parameterized gates round-trip through qasmName and gate(forQASMName)")
    func nonParameterizedRoundTrip() {
        let gates: [QuantumGate] = [
            .identity, .pauliX, .pauliY, .pauliZ, .hadamard,
            .sGate, .tGate, .sx, .sy,
            .cnot, .cz, .cy, .ch,
            .swap, .sqrtSwap, .iswap, .sqrtISwap, .fswap,
            .toffoli, .fredkin, .ccz,
        ]

        for gate in gates {
            let nameV2 = GateNameMapping.qasmName(for: gate, version: .v2)
            let recoveredV2 = GateNameMapping.gate(forQASMName: nameV2, version: .v2)
            #expect(recoveredV2 == gate, "round-trip failed for \(gate) in v2: name='\(nameV2)', recovered=\(String(describing: recoveredV2))")

            let nameV3 = GateNameMapping.qasmName(for: gate, version: .v3)
            let recoveredV3 = GateNameMapping.gate(forQASMName: nameV3, version: .v3)
            #expect(recoveredV3 == gate, "round-trip failed for \(gate) in v3: name='\(nameV3)', recovered=\(String(describing: recoveredV3))")
        }
    }

    @Test("Parameterized gates round-trip maps to placeholder-valued gate")
    func parameterizedRoundTripPlaceholder() {
        let placeholder: ParameterValue = .value(0)
        let gate = QuantumGate.rotationX(.value(1.5))
        let name = GateNameMapping.qasmName(for: gate, version: .v2)
        let recovered = GateNameMapping.gate(forQASMName: name, version: .v2)
        #expect(recovered == .rotationX(placeholder), "parameterized round-trip should yield placeholder gate, got \(String(describing: recovered))")
    }
}

/// Validates fallback naming for custom and non-standard gates.
/// Ensures customSingleQubit, customTwoQubit, and customUnitary
/// all produce the 'custom_unitary' fallback name.
@Suite("Custom Gate Fallback Names")
struct QASMNameCustomGateTests {
    @Test("customSingleQubit maps to 'custom_unitary'")
    func customSingleQubitFallback() {
        let matrix: [[Complex<Double>]] = [
            [.one, .zero],
            [.zero, .one],
        ]
        let gate = QuantumGate.customSingleQubit(matrix: matrix)
        let name = GateNameMapping.qasmName(for: gate, version: .v2)
        #expect(name == "custom_unitary", "customSingleQubit should map to 'custom_unitary', got '\(name)'")
    }

    @Test("customTwoQubit maps to 'custom_unitary'")
    func customTwoQubitFallback() {
        let matrix: [[Complex<Double>]] = [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .one, .zero],
            [.zero, .zero, .zero, .one],
        ]
        let gate = QuantumGate.customTwoQubit(matrix: matrix)
        let name = GateNameMapping.qasmName(for: gate, version: .v3)
        #expect(name == "custom_unitary", "customTwoQubit should map to 'custom_unitary', got '\(name)'")
    }

    @Test("customUnitary maps to 'custom_unitary'")
    func customUnitaryFallback() {
        let matrix: [[Complex<Double>]] = [
            [.one, .zero],
            [.zero, .one],
        ]
        let gate = QuantumGate.customUnitary(matrix: matrix)
        let name = GateNameMapping.qasmName(for: gate, version: .v2)
        #expect(name == "custom_unitary", "customUnitary should map to 'custom_unitary', got '\(name)'")
    }

    @Test("Diagonal gate maps to 'diagonal'")
    func diagonalGateName() {
        let name = GateNameMapping.qasmName(for: .diagonal(phases: [0.0]), version: .v2)
        #expect(name == "diagonal", "diagonal gate should map to 'diagonal', got '\(name)'")
    }

    @Test("Multiplexor gate maps to 'multiplexor'")
    func multiplexorGateName() {
        let name = GateNameMapping.qasmName(for: .multiplexor(unitaries: []), version: .v2)
        #expect(name == "multiplexor", "multiplexor gate should map to 'multiplexor', got '\(name)'")
    }
}
