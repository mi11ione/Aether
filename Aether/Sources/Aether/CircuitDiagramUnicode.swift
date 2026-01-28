// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Unicode box-drawing circuit diagram renderer for ``QuantumCircuit``.
///
/// Converts a ``QuantumCircuit`` into a human-readable Unicode string using box-drawing
/// characters for gate boundaries, filled circles for control qubits, and double-line
/// boxes for multi-qubit custom gates. The renderer uses a greedy column-major layout
/// algorithm that assigns each ``CircuitOperation`` to the earliest available time layer,
/// producing compact diagrams with precise gate alignment across qubit wires. Optionally
/// emits ANSI color escape sequences for terminal rendering when ``colorEnabled`` is set.
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
    /// ``┤Label├``, controlled gates show ``●`` on control wires connected by ``│`` to target
    /// symbols, SWAP gates display ``×`` on both wires, and reset operations render as ``┤0├``.
    /// When ``colorEnabled`` is true, gate names are cyan, control dots yellow, and reset
    /// operations green, with ANSI reset codes after each colored segment.
    ///
    /// - Parameter circuit: Quantum circuit to render as a diagram
    /// - Parameter colorEnabled: When true, emit ANSI escape codes for colored output
    /// - Returns: Multi-line Unicode string representing the circuit diagram
    /// - Complexity: O(ops x qubits) where ops is the number of circuit operations
    /// - SeeAlso: ``QuantumCircuit``
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    public static func render(_ circuit: QuantumCircuit, colorEnabled: Bool = false) -> String {
        let qubitCount = circuit.qubits
        let operations = circuit.operations

        if operations.isEmpty {
            return "Empty circuit (\(qubitCount) qubit\(qubitCount == 1 ? "" : "s"))"
        }

        let layers = assignLayers(operations: operations, qubitCount: qubitCount)
        let layerCount = layers.count

        var columnWidths = [Int](unsafeUninitializedCapacity: layerCount) { buffer, count in
            for i in 0 ..< layerCount {
                buffer[i] = 1
            }
            count = layerCount
        }

        for layerIndex in 0 ..< layerCount {
            for operation in layers[layerIndex] {
                let label = gateLabel(for: operation)
                let labelWidth = label.count + 2
                if labelWidth > columnWidths[layerIndex] {
                    columnWidths[layerIndex] = labelWidth
                }
            }
        }

        let labelWidth = qubitLabelWidth(qubitCount: qubitCount)
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
                for _ in 0 ..< padding {
                    result += " "
                }
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
                        colorEnabled: colorEnabled,
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
                for _ in 0 ..< labelWidth + 2 {
                    result += " "
                }

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
    @_optimize(speed)
    @usableFromInline
    static func assignLayers(operations: [CircuitOperation], qubitCount: Int) -> [[CircuitOperation]] {
        var nextAvailable = [Int](unsafeUninitializedCapacity: qubitCount) { buffer, count in
            for i in 0 ..< qubitCount {
                buffer[i] = 0
            }
            count = qubitCount
        }

        let upperBound = operations.count
        var layers = [[CircuitOperation]]()
        layers.reserveCapacity(upperBound)

        for operation in operations {
            let involvedQubits = operation.qubits
            var layer = 0
            for q in involvedQubits {
                if q < qubitCount, nextAvailable[q] > layer {
                    layer = nextAvailable[q]
                }
            }

            while layers.count <= layer {
                layers.append([])
            }
            layers[layer].append(operation)

            for q in involvedQubits {
                if q < qubitCount {
                    nextAvailable[q] = layer + 1
                }
            }
        }

        return layers
    }

    /// Produce the display label for a gate operation.
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

    /// Produce the display label for a quantum gate.
    @usableFromInline
    static func gateLabelForGate(_ gate: QuantumGate, qubits: [Int]) -> String {
        switch gate {
        case .identity: "I"
        case .pauliX: "X"
        case .pauliY: "Y"
        case .pauliZ: "Z"
        case .hadamard: "H"
        case let .phase(angle): "P(\(formatAngle(angle)))"
        case .sGate: "S"
        case .tGate: "T"
        case let .rotationX(theta): "Rx(\(formatAngle(theta)))"
        case let .rotationY(theta): "Ry(\(formatAngle(theta)))"
        case let .rotationZ(theta): "Rz(\(formatAngle(theta)))"
        case let .u1(lambda): "U1(\(formatAngle(lambda)))"
        case let .u2(phi, lambda): "U2(\(formatAngle(phi)),\(formatAngle(lambda)))"
        case let .u3(theta, phi, lambda): "U3(\(formatAngle(theta)),\(formatAngle(phi)),\(formatAngle(lambda)))"
        case .sx: "SX"
        case .sy: "SY"
        case .customSingleQubit: "U"
        case let .globalPhase(phi): "GP(\(formatAngle(phi)))"
        case .cnot: "X"
        case .cz: "Z"
        case .cy: "Y"
        case .ch: "H"
        case let .controlledPhase(theta): "P(\(formatAngle(theta)))"
        case let .controlledRotationX(theta): "Rx(\(formatAngle(theta)))"
        case let .controlledRotationY(theta): "Ry(\(formatAngle(theta)))"
        case let .controlledRotationZ(theta): "Rz(\(formatAngle(theta)))"
        case .swap: "×"
        case .sqrtSwap: "√SW"
        case .iswap: "iSW"
        case .sqrtISwap: "√iSW"
        case .fswap: "fSW"
        case let .givens(theta): "Giv(\(formatAngle(theta)))"
        case let .xx(theta): "XX(\(formatAngle(theta)))"
        case let .yy(theta): "YY(\(formatAngle(theta)))"
        case let .zz(theta): "ZZ(\(formatAngle(theta)))"
        case .customTwoQubit: "U"
        case .toffoli: "X"
        case .fredkin: "×"
        case .ccz: "Z"
        case let .controlled(innerGate, _):
            gateLabelForGate(innerGate, qubits: qubits)
        case .diagonal: "Diag"
        case .multiplexor: "Mux"
        case .customUnitary: "U"
        }
    }

    /// Format a parameter angle value for display.
    @usableFromInline
    static func formatAngle(_ value: ParameterValue) -> String {
        switch value {
        case let .value(v):
            if v == .pi { return "π" }
            if v == -.pi { return "-π" }
            if v == .pi / 2.0 { return "π/2" }
            if v == -.pi / 2.0 { return "-π/2" }
            if v == .pi / 4.0 { return "π/4" }
            if v == -.pi / 4.0 { return "-π/4" }
            let rounded = (v * 100.0).rounded() / 100.0
            return String(rounded)
        case let .parameter(p):
            return p.name
        case let .negatedParameter(p):
            return "-\(p.name)"
        }
    }

    /// Compute the width needed for qubit labels.
    @usableFromInline
    static func qubitLabelWidth(qubitCount: Int) -> Int {
        var maxWidth = 2
        for i in 0 ..< qubitCount {
            var width = 1
            var n = i
            if n == 0 {
                width = 1
            } else {
                width = 0
                while n > 0 {
                    width += 1
                    n /= 10
                }
            }
            let totalWidth = width + 1
            if totalWidth > maxWidth { maxWidth = totalWidth }
        }
        return maxWidth
    }

    /// Determine the role of a qubit within a specific layer.
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
    static func qubitBoundaryFlags(qubit: Int, in qubits: [Int]) -> (isTop: Bool, isBottom: Bool, isMiddle: Bool) {
        let minQ = qubits.min()!
        let maxQ = qubits.max()!
        let isTop = qubit == minQ
        let isBottom = qubit == maxQ
        let isMiddle = !isTop && !isBottom
        return (isTop, isBottom, isMiddle)
    }

    /// Classify the role of a qubit within a gate operation.
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
    @usableFromInline
    static func layerNeedsVerticalBetween(upper: Int, lower: Int, layer: [CircuitOperation]) -> Bool {
        for operation in layer {
            let qubits = operation.qubits
            if qubits.count < 2 { continue }
            let minQ = qubits.min()!
            let maxQ = qubits.max()!
            if upper >= minQ, lower <= maxQ { return true }
        }
        return false
    }

    /// Render a wire-line segment for a given qubit role and column width.
    @usableFromInline
    static func renderWireSegment(
        role: QubitRole,
        colWidth: Int,
        colorEnabled: Bool,
        into result: inout String,
    ) {
        let ansiCyan = "\u{1b}[36m"
        let ansiYellow = "\u{1b}[33m"
        let ansiGreen = "\u{1b}[32m"
        let ansiReset = "\u{1b}[0m"

        switch role {
        case .idle:
            for _ in 0 ..< colWidth + 2 {
                result += "─"
            }

        case .classicalIdle:
            for _ in 0 ..< colWidth + 2 {
                result += "═"
            }

        case let .singleGate(label):
            result += "┤"
            if colorEnabled { result += ansiCyan }
            result += label
            if colorEnabled { result += ansiReset }
            result += "├"
            let used = label.count + 2
            for _ in used ..< colWidth + 2 {
                result += "─"
            }

        case .controlDot:
            let mid = colWidth / 2
            for _ in 0 ..< mid {
                result += "─"
            }
            if colorEnabled { result += ansiYellow }
            result += "●"
            if colorEnabled { result += ansiReset }
            for _ in (mid + 1) ..< colWidth + 2 {
                result += "─"
            }

        case .cnotTarget:
            let mid = colWidth / 2
            for _ in 0 ..< mid {
                result += "─"
            }
            if colorEnabled { result += ansiCyan }
            result += "⊕"
            if colorEnabled { result += ansiReset }
            for _ in (mid + 1) ..< colWidth + 2 {
                result += "─"
            }

        case .swapCross:
            let mid = colWidth / 2
            for _ in 0 ..< mid {
                result += "─"
            }
            if colorEnabled { result += ansiCyan }
            result += "×"
            if colorEnabled { result += ansiReset }
            for _ in (mid + 1) ..< colWidth + 2 {
                result += "─"
            }

        case .resetGate:
            result += "┤"
            if colorEnabled { result += ansiGreen }
            result += "0"
            if colorEnabled { result += ansiReset }
            result += "├"
            let used = 3
            for _ in used ..< colWidth + 2 {
                result += "─"
            }

        case .measureGate:
            let ansiMagenta = "\u{1b}[35m"
            result += "┤"
            if colorEnabled { result += ansiMagenta }
            result += "M"
            if colorEnabled { result += ansiReset }
            result += "├"
            let mUsed = 3
            for _ in mUsed ..< colWidth + 2 {
                result += "─"
            }

        case let .multiGateSegment(label, isTop, isBottom, isMiddle):
            if isTop {
                result += "╔"
                if colorEnabled { result += ansiCyan }
                result += label
                if colorEnabled { result += ansiReset }
                let used = label.count + 1
                for _ in used ..< colWidth + 1 {
                    result += "═"
                }
                result += "╗"
            } else if isMiddle {
                result += "╠"
                for _ in 1 ..< colWidth + 1 {
                    result += "═"
                }
                result += "╣"
            } else if isBottom {
                result += "╚"
                for _ in 1 ..< colWidth + 1 {
                    result += "═"
                }
                result += "╝"
            }
        }
    }

    /// Render a spacer line segment between two qubit wire lines.
    @usableFromInline
    static func renderSpacerSegment(
        colWidth: Int,
        needsVertical: Bool,
        into result: inout String,
    ) {
        if needsVertical {
            let mid = colWidth / 2
            for _ in 0 ..< mid {
                result += " "
            }
            result += "│"
            for _ in (mid + 1) ..< colWidth + 2 {
                result += " "
            }
        } else {
            for _ in 0 ..< colWidth + 2 {
                result += " "
            }
        }
    }
}
