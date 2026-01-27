// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// OpenQASM language version selector for gate name resolution.
///
/// Distinguishes between QASM 2.0 (qelib1.inc conventions) and QASM 3.0 (stdgates.inc
/// conventions) where certain gate identifiers differ. The primary divergence is the phase
/// gate, exported as "u1" in v2 and "p" in v3. Passed to ``GateNameMapping`` methods to
/// select the appropriate identifier set.
///
/// **Example:**
/// ```swift
/// let version: QASMVersion = .v3
/// let name = GateNameMapping.qasmName(for: .phase(.value(0)), version: version)
/// assert(name == "p")
/// ```
///
/// - SeeAlso: ``GateNameMapping``
@frozen public enum QASMVersion: Sendable {
    case v2
    case v3
}

/// Bidirectional mapping between ``QuantumGate`` enum cases and OpenQASM gate identifier strings.
///
/// Provides the canonical translation layer used by all QASM exporters and importers. Forward
/// mapping (gate to string) is total for all exportable gates and returns a deterministic
/// identifier. Reverse mapping (string to gate) returns nil for unrecognized identifiers and
/// constructs parameterized gates with placeholder ``ParameterValue/value(_:)`` of zero so
/// the importer can substitute actual parameter values after parsing.
///
/// **Example:**
/// ```swift
/// let name = GateNameMapping.qasmName(for: .hadamard, version: .v2)
/// let roundTripped = GateNameMapping.gate(forQASMName: name, version: .v2)
/// assert(roundTripped == .hadamard)
/// ```
///
/// - SeeAlso: ``QuantumGate``
/// - SeeAlso: ``QASMVersion``
public enum GateNameMapping: Sendable {
    @usableFromInline static let placeholder: ParameterValue = .value(0)

    /// Resolve the OpenQASM gate identifier for a given ``QuantumGate``.
    ///
    /// Returns the canonical QASM string used in gate statements. Version-sensitive gates
    /// such as the phase gate return "u1" for QASM 2.0 and "p" for QASM 3.0. Custom matrix
    /// gates that lack a standard QASM decomposition return "custom_unitary". For the
    /// generic ``QuantumGate/controlled(gate:controls:)`` case the result is the inner
    /// gate name prefixed with version-appropriate control syntax.
    ///
    /// **Example:**
    /// ```swift
    /// let name = GateNameMapping.qasmName(for: .cnot, version: .v2)
    /// assert(name == "cx")
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to translate
    ///   - version: Target QASM language version
    /// - Returns: QASM gate identifier string
    /// - Complexity: O(1) for non-controlled gates, O(d) for d-level nested controlled gates
    /// - SeeAlso: ``gate(forQASMName:version:)``
    @inlinable
    @_effects(readonly)
    public static func qasmName(for gate: QuantumGate, version: QASMVersion) -> String {
        switch gate {
        case .identity:
            "id"
        case .pauliX:
            "x"
        case .pauliY:
            "y"
        case .pauliZ:
            "z"
        case .hadamard:
            "h"
        case .phase:
            switch version {
            case .v2: "u1"
            case .v3: "p"
            }
        case .sGate:
            "s"
        case .tGate:
            "t"
        case .rotationX:
            "rx"
        case .rotationY:
            "ry"
        case .rotationZ:
            "rz"
        case .u1:
            "u1"
        case .u2:
            "u2"
        case .u3:
            "u3"
        case .sx:
            "sx"
        case .sy:
            "sy"
        case .globalPhase:
            "gphase"
        case .cnot:
            "cx"
        case .cz:
            "cz"
        case .cy:
            "cy"
        case .ch:
            "ch"
        case .controlledPhase:
            "cp"
        case .controlledRotationX:
            "crx"
        case .controlledRotationY:
            "cry"
        case .controlledRotationZ:
            "crz"
        case .swap:
            "swap"
        case .sqrtSwap:
            "sqsw"
        case .iswap:
            "iswap"
        case .sqrtISwap:
            "siswap"
        case .fswap:
            "fswap"
        case .givens:
            "givens"
        case .xx:
            "rxx"
        case .yy:
            "ryy"
        case .zz:
            "rzz"
        case .toffoli:
            "ccx"
        case .fredkin:
            "cswap"
        case .ccz:
            "ccz"
        case .customSingleQubit, .customTwoQubit, .customUnitary:
            "custom_unitary"
        case .diagonal:
            "diagonal"
        case .multiplexor:
            "multiplexor"
        case let .controlled(innerGate, _):
            controlledQASMName(for: innerGate, version: version)
        }
    }

    /// Resolve a ``QuantumGate`` from an OpenQASM gate identifier string.
    ///
    /// Performs case-sensitive lookup of the QASM gate name and returns the corresponding
    /// ``QuantumGate`` case. Parameterized gates are returned with a zero-valued placeholder
    /// parameter that the importer replaces with parsed argument values. Returns nil for
    /// unrecognized gate names.
    ///
    /// **Example:**
    /// ```swift
    /// let gate = GateNameMapping.gate(forQASMName: "rx", version: .v2)
    /// assert(gate == .rotationX(.value(0)))
    /// ```
    ///
    /// - Parameters:
    ///   - name: QASM gate identifier string
    ///   - version: Source QASM language version
    /// - Returns: Corresponding quantum gate or nil if unrecognized
    /// - Complexity: O(1) average via dictionary lookup
    /// - SeeAlso: ``qasmName(for:version:)``
    @_optimize(speed)
    @_effects(readonly)
    public static func gate(forQASMName name: String, version: QASMVersion) -> QuantumGate? {
        switch version {
        case .v2: v2NameToGate[name]
        case .v3: v3NameToGate[name]
        }
    }

    @usableFromInline static let v2NameToGate: [String: QuantumGate] = buildV2Table()

    @usableFromInline static let v3NameToGate: [String: QuantumGate] = buildV3Table()

    /// Build the shared gate-name-to-gate lookup table common to both QASM versions.
    private static func buildCommonTable() -> [String: QuantumGate] {
        [
            "id": .identity,
            "x": .pauliX,
            "y": .pauliY,
            "z": .pauliZ,
            "h": .hadamard,
            "s": .sGate,
            "t": .tGate,
            "rx": .rotationX(placeholder),
            "ry": .rotationY(placeholder),
            "rz": .rotationZ(placeholder),
            "u1": .u1(lambda: placeholder),
            "u2": .u2(phi: placeholder, lambda: placeholder),
            "u3": .u3(theta: placeholder, phi: placeholder, lambda: placeholder),
            "sx": .sx,
            "sy": .sy,
            "gphase": .globalPhase(placeholder),
            "cx": .cnot,
            "cz": .cz,
            "cy": .cy,
            "ch": .ch,
            "cp": .controlledPhase(placeholder),
            "crx": .controlledRotationX(placeholder),
            "cry": .controlledRotationY(placeholder),
            "crz": .controlledRotationZ(placeholder),
            "swap": .swap,
            "sqsw": .sqrtSwap,
            "iswap": .iswap,
            "siswap": .sqrtISwap,
            "fswap": .fswap,
            "givens": .givens(placeholder),
            "rxx": .xx(placeholder),
            "ryy": .yy(placeholder),
            "rzz": .zz(placeholder),
            "ccx": .toffoli,
            "cswap": .fredkin,
            "ccz": .ccz,
        ]
    }

    /// Build the QASM 2.0 gate-name lookup table with version-specific overrides.
    private static func buildV2Table() -> [String: QuantumGate] {
        var table = buildCommonTable()
        table["u1"] = .u1(lambda: placeholder)
        return table
    }

    /// Build the QASM 3.0 gate-name lookup table with version-specific overrides.
    private static func buildV3Table() -> [String: QuantumGate] {
        var table = buildCommonTable()
        table["p"] = .phase(placeholder)
        return table
    }

    @usableFromInline static func controlledQASMName(for innerGate: QuantumGate, version: QASMVersion) -> String {
        let baseName = qasmName(for: innerGate, version: version)
        switch version {
        case .v2:
            return "c_\(baseName)"
        case .v3:
            return "ctrl_\(baseName)"
        }
    }
}
