// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Unicode box-drawing circuit diagram renderer for ``QuantumCircuit``.
///
/// Converts a ``QuantumCircuit`` into a human-readable Unicode string using box-drawing
/// characters for gate boundaries, filled circles for control qubits, and double-line
/// boxes for multi-qubit custom gates. The renderer organizes operations into time layers
/// and renders each qubit wire with box-drawing characters, producing compact diagrams
/// with precise gate alignment across qubit wires. Optionally emits ANSI color escape
/// sequences for terminal rendering when ``isColorEnabled`` is set.
///
/// **Example:**
/// ```swift
/// var circuit = QuantumCircuit(qubits: 2)
/// circuit.append(.hadamard, to: 0)
/// circuit.append(.cnot, to: [0, 1])
/// let diagram = CircuitDiagramUnicode.render(circuit)
/// ```
///
/// - SeeAlso: ``QuantumCircuit``
/// - SeeAlso: ``CircuitOperation``
/// - SeeAlso: ``QuantumGate``
public enum CircuitDiagramUnicode: Sendable {
    /// Render a quantum circuit as a Unicode box-drawing diagram string.
    ///
    /// Produces a multi-line string where each qubit is drawn as a labeled horizontal wire
    /// with gates rendered inline using box-drawing delimiters. Single-qubit gates appear as
    /// `┤Label├`, controlled gates show `●` on control wires connected by `│` to target
    /// symbols, SWAP gates display `×` on both wires, and reset operations render as `┤0├`.
    /// When ``isColorEnabled`` is true, gate names are cyan, control dots yellow, and reset
    /// operations green, with ANSI reset codes after each colored segment.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit(qubits: 2)
    /// circuit.addGate(.hadamard, qubit: 0)
    /// circuit.addGate(.cnot, control: 0, target: 1)
    /// let diagram = CircuitDiagramUnicode.render(circuit)
    /// print(diagram)
    /// ```
    ///
    /// - Parameter circuit: Quantum circuit to render as a diagram
    /// - Parameter isColorEnabled: When true, emit ANSI escape codes for colored output
    /// - Returns: Multi-line Unicode string representing the circuit diagram
    /// - Complexity: O(ops x qubits) where ops is the number of circuit operations
    /// - SeeAlso: ``QuantumCircuit``
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    public static func render(_ circuit: QuantumCircuit, isColorEnabled: Bool = false) -> String {
        let qubitCount = circuit.qubits
        let operations = circuit.operations

        if operations.isEmpty {
            return "Empty circuit (\(qubitCount) qubit\(qubitCount == 1 ? "" : "s"))"
        }

        let layers = assignLayers(operations: operations, qubitCount: qubitCount)
        let layerCount = layers.count

        var columnWidths = [Int](unsafeUninitializedCapacity: layerCount) { buffer, count in
            buffer.initialize(repeating: 1)
            count = layerCount
        }

        for layerIndex in 0 ..< layerCount {
            for operation in layers[layerIndex] {
                let label = gateLabel(for: operation)
                let labelWidth = label.count + 2
                columnWidths[layerIndex] = max(columnWidths[layerIndex], labelWidth)
            }
        }

        let labelWidth = CircuitDiagramUtilities.qubitLabelWidth(qubitCount)
        let totalQubitLines = 2 * qubitCount - 1

        var estimatedSize = totalQubitLines * (labelWidth + layerCount * 8 + 4)
        estimatedSize += qubitCount * 16
        var result = String()
        result.reserveCapacity(estimatedSize)

        for qubitLine in 0 ..< totalQubitLines {
            let isWireLine = qubitLine % 2 == 0
            let qubitIndex = qubitLine / 2

            if isWireLine {
                let label = "q\(qubitIndex)"
                result += label
                let padding = labelWidth - label.count
                result += String(repeating: " ", count: padding)
                result += ": "

                var measured = false
                for layerIndex in 0 ..< layerCount {
                    let colWidth = columnWidths[layerIndex]
                    let rawRole = qubitRole(qubit: qubitIndex, layer: layers[layerIndex])
                    let role: QubitRole = if case .idle = rawRole, measured {
                        .classicalIdle
                    } else {
                        rawRole
                    }
                    renderWireSegment(
                        role: role,
                        colWidth: colWidth,
                        isColorEnabled: isColorEnabled,
                        into: &result,
                    )
                    if case .measureGate = role {
                        measured = true
                    }
                }

                if measured {
                    result += "═"
                } else {
                    result += "─"
                }
            } else {
                result += String(repeating: " ", count: labelWidth + 2)

                let upperQubit = qubitIndex
                let lowerQubit = qubitIndex + 1

                for layerIndex in 0 ..< layerCount {
                    let colWidth = columnWidths[layerIndex]
                    let needsVertical = layerNeedsVerticalBetween(
                        upper: upperQubit,
                        lower: lowerQubit,
                        layer: layers[layerIndex],
                    )
                    renderSpacerSegment(
                        colWidth: colWidth,
                        needsVertical: needsVertical,
                        into: &result,
                    )
                }

                result += " "
            }

            if qubitLine < totalQubitLines - 1 {
                result += "\n"
            }
        }

        return result
    }

    /// Boundary position flags for a qubit within a multi-qubit gate span.
    @usableFromInline
    typealias BoundaryFlags = (isTop: Bool, isBottom: Bool, isMiddle: Bool)

    /// Rendering role of a qubit wire within a single time layer.
    @frozen
    @usableFromInline
    enum QubitRole {
        case idle
        case classicalIdle
        case singleGate(label: String)
        case controlDot
        case cnotTarget
        case swapCross
        case resetGate
        case measureGate
        case multiGateSegment(label: String, isTop: Bool, isBottom: Bool, isMiddle: Bool)
    }

    /// Assign operations to time layers using greedy scheduling.
    @_effects(readonly)
    @_optimize(speed)
    @usableFromInline
    static func assignLayers(operations: [CircuitOperation], qubitCount: Int) -> [[CircuitOperation]] {
        let assignments = CircuitDiagramUtilities.assignLayers(operations: operations, qubitCount: qubitCount)
        let layerCount = (assignments.max() ?? -1) + 1
        var layers = [[CircuitOperation]](repeating: [], count: layerCount)
        var layerCounts = [Int](unsafeUninitializedCapacity: layerCount) { buffer, count in
            buffer.initialize(repeating: 0)
            count = layerCount
        }
        for i in operations.indices {
            layerCounts[assignments[i]] += 1
        }
        for i in 0 ..< layerCount {
            layers[i].reserveCapacity(layerCounts[i])
        }
        for i in operations.indices {
            layers[assignments[i]].append(operations[i])
        }
        return layers
    }

    /// Produce the display label for a gate operation.
    @_effects(readonly)
    @usableFromInline
    static func gateLabel(for operation: CircuitOperation) -> String {
        switch operation {
        case let .gate(gate, qubits, _):
            gateLabelForGate(gate, qubits: qubits)
        case .reset:
            "0"
        case .measure:
            "M"
        }
    }

    /// Produce the Unicode-specific display label for a quantum gate.
    @_effects(readonly)
    @usableFromInline
    static func gateLabelForGate(_ gate: QuantumGate, qubits: [Int]) -> String {
        if let shared = CircuitDiagramUtilities.gateLabel(gate) {
            return shared
        }
        return switch gate {
        case let .phase(angle): "P(\(formatAngle(angle)))"
        case let .rotationX(theta): "Rx(\(formatAngle(theta)))"
        case let .rotationY(theta): "Ry(\(formatAngle(theta)))"
        case let .rotationZ(theta): "Rz(\(formatAngle(theta)))"
        case let .u1(lambda): "U1(\(formatAngle(lambda)))"
        case let .u2(phi, lambda): "U2(\(formatAngle(phi)),\(formatAngle(lambda)))"
        case let .u3(theta, phi, lambda): "U3(\(formatAngle(theta)),\(formatAngle(phi)),\(formatAngle(lambda)))"
        case let .globalPhase(phi): "GP(\(formatAngle(phi)))"
        case .cnot: "X"
        case .cz: "Z"
        case .cy: "Y"
        case .ch: "H"
        case let .controlledPhase(theta): "P(\(formatAngle(theta)))"
        case let .controlledRotationX(theta): "Rx(\(formatAngle(theta)))"
        case let .controlledRotationY(theta): "Ry(\(formatAngle(theta)))"
        case let .controlledRotationZ(theta): "Rz(\(formatAngle(theta)))"
        case .swap: "\u{00D7}"
        case .sqrtSwap: "\u{221A}SW"
        case .iswap: "iSW"
        case .sqrtISwap: "\u{221A}iSW"
        case .fswap: "fSW"
        case let .givens(theta): "Giv(\(formatAngle(theta)))"
        case let .xx(theta): "XX(\(formatAngle(theta)))"
        case let .yy(theta): "YY(\(formatAngle(theta)))"
        case let .zz(theta): "ZZ(\(formatAngle(theta)))"
        case .customTwoQubit: "U"
        case .toffoli: "X"
        case .fredkin: "\u{00D7}"
        case .ccz: "Z"
        case let .controlled(innerGate, _):
            gateLabelForGate(innerGate, qubits: qubits)
        case .diagonal: "Diag"
        case .multiplexor: "Mux"
        default: gate.description
        }
    }

    /// Epsilon for floating-point angle comparisons in display formatting.
    @usableFromInline
    static let angleEpsilon: Double = 1e-12

    /// Format a parameter angle value for display.
    @_effects(readonly)
    @inline(__always)
    @usableFromInline
    static func formatAngle(_ value: ParameterValue) -> String {
        switch value {
        case let .value(v):
            if abs(v - .pi) < angleEpsilon { return "π" }
            if abs(v + .pi) < angleEpsilon { return "-π" }
            if abs(v - .pi / 2.0) < angleEpsilon { return "π/2" }
            if abs(v + .pi / 2.0) < angleEpsilon { return "-π/2" }
            if abs(v - .pi / 4.0) < angleEpsilon { return "π/4" }
            if abs(v + .pi / 4.0) < angleEpsilon { return "-π/4" }
            let rounded = (v * 100.0).rounded() / 100.0
            return String(rounded)
        case let .parameter(p):
            return p.name
        case let .negatedParameter(p):
            return "-\(p.name)"
        case .expression:
            return "expr"
        }
    }

    /// Determine the role of a qubit within a specific layer.
    @_effects(readonly)
    @usableFromInline
    static func qubitRole(qubit: Int, layer: [CircuitOperation]) -> QubitRole {
        for operation in layer {
            let qubits = operation.qubits
            if !qubits.contains(qubit) { continue }

            switch operation {
            case let .gate(gate, opQubits, _):
                return classifyGateRole(gate: gate, qubit: qubit, qubits: opQubits)
            case .reset:
                return .resetGate
            case .measure:
                return .measureGate
            }
        }
        return .idle
    }

    /// Compute qubit boundary position flags from a qubit array.
    @inlinable
    @_effects(readonly)
    static func qubitBoundaryFlags(qubit: Int, in qubits: [Int]) -> BoundaryFlags {
        let minQ = qubits.min()! // Safe: qubits guaranteed non-empty for multi-qubit gates
        let maxQ = qubits.max()! // Safe: qubits guaranteed non-empty for multi-qubit gates
        let isTop = qubit == minQ
        let isBottom = qubit == maxQ
        let isMiddle = !isTop && !isBottom
        return (isTop, isBottom, isMiddle)
    }

    /// Classify the role of a qubit within a gate operation.
    @_effects(readonly)
    @usableFromInline
    static func classifyGateRole(gate: QuantumGate, qubit: Int, qubits: [Int]) -> QubitRole {
        switch gate {
        case .cnot:
            if qubit == qubits[0] { return .controlDot }
            return .cnotTarget

        case .cz, .cy, .ch,
             .controlledPhase, .controlledRotationX, .controlledRotationY, .controlledRotationZ:
            if qubit == qubits[0] { return .controlDot }
            let label = gateLabelForGate(gate, qubits: qubits)
            return .singleGate(label: label)

        case .swap:
            return .swapCross

        case .toffoli:
            if qubit == qubits[0] || qubit == qubits[1] { return .controlDot }
            return .cnotTarget

        case .fredkin:
            if qubit == qubits[0] { return .controlDot }
            return .swapCross

        case .ccz:
            if qubit == qubits[0] || qubit == qubits[1] { return .controlDot }
            let label = gateLabelForGate(gate, qubits: qubits)
            return .singleGate(label: label)

        case let .controlled(innerGate, controls):
            let controlQubits = Array(qubits.prefix(controls.count))
            let targetQubits = Array(qubits.suffix(from: controls.count))
            if controlQubits.contains(qubit) { return .controlDot }
            if innerGate == .pauliX && targetQubits.count == 1 && targetQubits[0] == qubit {
                return .cnotTarget
            }
            if innerGate == .swap || innerGate == .fredkin {
                return .swapCross
            }
            let label = gateLabelForGate(innerGate, qubits: targetQubits)
            if targetQubits.count == 1 {
                return .singleGate(label: label)
            }
            let flags = qubitBoundaryFlags(qubit: qubit, in: targetQubits)
            return .multiGateSegment(label: label, isTop: flags.isTop, isBottom: flags.isBottom, isMiddle: flags.isMiddle)

        case .sqrtSwap, .iswap, .sqrtISwap, .fswap,
             .givens, .xx, .yy, .zz, .customTwoQubit:
            let label = gateLabelForGate(gate, qubits: qubits)
            let flags = qubitBoundaryFlags(qubit: qubit, in: qubits)
            return .multiGateSegment(label: label, isTop: flags.isTop, isBottom: flags.isBottom, isMiddle: false)

        case .diagonal, .multiplexor, .customUnitary:
            let label = gateLabelForGate(gate, qubits: qubits)
            if qubits.count == 1 { return .singleGate(label: label) }
            let flags = qubitBoundaryFlags(qubit: qubit, in: qubits)
            return .multiGateSegment(label: label, isTop: flags.isTop, isBottom: flags.isBottom, isMiddle: flags.isMiddle)

        default:
            let label = gateLabelForGate(gate, qubits: qubits)
            return .singleGate(label: label)
        }
    }

    /// Determine if a vertical connector is needed between two adjacent qubit lines.
    @_effects(readonly)
    @usableFromInline
    static func layerNeedsVerticalBetween(upper: Int, lower: Int, layer: [CircuitOperation]) -> Bool {
        for operation in layer {
            let qubits = operation.qubits
            if qubits.count < 2 { continue }
            let minQ = qubits.min()! // Safe: guard above ensures qubits.count >= 2
            let maxQ = qubits.max()! // Safe: guard above ensures qubits.count >= 2
            if upper >= minQ, lower <= maxQ { return true }
        }
        return false
    }

    /// ANSI escape code for cyan text.
    private static let ansiCyan = "\u{1b}[36m"
    /// ANSI escape code for yellow text.
    private static let ansiYellow = "\u{1b}[33m"
    /// ANSI escape code for green text.
    private static let ansiGreen = "\u{1b}[32m"
    /// ANSI escape code for magenta text.
    private static let ansiMagenta = "\u{1b}[35m"
    /// ANSI escape code to reset text formatting.
    private static let ansiReset = "\u{1b}[0m"

    /// Render a wire-line segment for a given qubit role and column width.
    @_optimize(speed)
    @usableFromInline
    static func renderWireSegment(
        role: QubitRole,
        colWidth: Int,
        isColorEnabled: Bool,
        into result: inout String,
    ) {
        switch role {
        case .idle:
            result += String(repeating: "─", count: colWidth + 2)

        case .classicalIdle:
            result += String(repeating: "═", count: colWidth + 2)

        case let .singleGate(label):
            result += "┤"
            if isColorEnabled { result += Self.ansiCyan }
            result += label
            if isColorEnabled { result += Self.ansiReset }
            result += "├"
            let remaining = colWidth + 2 - label.count - 2
            if remaining > 0 { result += String(repeating: "─", count: remaining) }

        case .controlDot:
            let mid = colWidth / 2
            result += String(repeating: "─", count: mid)
            if isColorEnabled { result += Self.ansiYellow }
            result += "●"
            if isColorEnabled { result += Self.ansiReset }
            result += String(repeating: "─", count: colWidth + 2 - mid - 1)

        case .cnotTarget:
            let mid = colWidth / 2
            result += String(repeating: "─", count: mid)
            if isColorEnabled { result += Self.ansiCyan }
            result += "⊕"
            if isColorEnabled { result += Self.ansiReset }
            result += String(repeating: "─", count: colWidth + 2 - mid - 1)

        case .swapCross:
            let mid = colWidth / 2
            result += String(repeating: "─", count: mid)
            if isColorEnabled { result += Self.ansiCyan }
            result += "×"
            if isColorEnabled { result += Self.ansiReset }
            result += String(repeating: "─", count: colWidth + 2 - mid - 1)

        case .resetGate:
            result += "┤"
            if isColorEnabled { result += ansiGreen }
            result += "0"
            if isColorEnabled { result += ansiReset }
            result += "├"
            let remaining = colWidth + 2 - 3
            if remaining > 0 { result += String(repeating: "─", count: remaining) }

        case .measureGate:
            result += "┤"
            if isColorEnabled { result += ansiMagenta }
            result += "M"
            if isColorEnabled { result += ansiReset }
            result += "├"
            let remaining = colWidth + 2 - 3
            if remaining > 0 { result += String(repeating: "─", count: remaining) }

        case let .multiGateSegment(label, isTop, isBottom, isMiddle):
            if isTop {
                result += "╔"
                if isColorEnabled { result += Self.ansiCyan }
                result += label
                if isColorEnabled { result += Self.ansiReset }
                let remaining = colWidth + 1 - label.count - 1
                if remaining > 0 { result += String(repeating: "═", count: remaining) }
                result += "╗"
            } else if isMiddle {
                result += "╠"
                result += String(repeating: "═", count: colWidth)
                result += "╣"
            } else if isBottom {
                result += "╚"
                result += String(repeating: "═", count: colWidth)
                result += "╝"
            }
        }
    }

    /// Render a spacer line segment between two qubit wire lines.
    @_optimize(speed)
    @usableFromInline
    static func renderSpacerSegment(
        colWidth: Int,
        needsVertical: Bool,
        into result: inout String,
    ) {
        if needsVertical {
            let mid = colWidth / 2
            result += String(repeating: " ", count: mid)
            result += "│"
            result += String(repeating: " ", count: colWidth + 2 - mid - 1)
        } else {
            result += String(repeating: " ", count: colWidth + 2)
        }
    }
}
