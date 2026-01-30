// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// ASCII text renderer for quantum circuit diagrams with wire-based layout.
///
/// Converts a ``QuantumCircuit`` into a human-readable ASCII string where each qubit is
/// rendered as a horizontal wire with dashes, gate labels appear inline, control qubits
/// display as filled circles, targets as circled-plus symbols, and vertical bars connect
/// multi-qubit gate endpoints through intermediate wires. Uses greedy column-major layout
/// to assign each operation to the earliest available time layer.
///
/// **Example:**
/// ```swift
/// var circuit = QuantumCircuit(qubits: 2)
/// circuit.append(.hadamard, to: 0)
/// circuit.append(.cnot, to: [0, 1])
/// let diagram = CircuitDiagramASCII.render(circuit)
/// print(diagram)
/// ```
///
/// - SeeAlso: ``QuantumCircuit``
/// - SeeAlso: ``QuantumGate``
/// - SeeAlso: ``CircuitOperation``
public enum CircuitDiagramASCII: Sendable {
    /// Render a quantum circuit as an ASCII text diagram with wire-based layout.
    ///
    /// Assigns each ``CircuitOperation`` to a time layer column using greedy scheduling,
    /// computes column widths from the widest gate label per layer, then draws qubit wires
    /// with dashes and gate symbols inline. Multi-qubit gates connect control and target
    /// qubits with vertical bar connectors through intermediate wires. CNOT renders as
    /// control dot plus circled-plus target, SWAP renders as crosses on both qubits,
    /// Toffoli renders as two control dots plus circled-plus target, and general controlled
    /// gates render as control dots plus the inner gate label.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 3)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    /// circuit.append(.hadamard, to: 2)
    /// let diagram = CircuitDiagramASCII.render(circuit)
    /// print(diagram)
    /// ```
    ///
    /// - Parameter circuit: Quantum circuit to render as ASCII diagram
    /// - Returns: Multi-line ASCII string representing the circuit diagram
    /// - Complexity: O(n * q) where n is the number of operations and q is the number of qubits
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    public static func render(_ circuit: QuantumCircuit) -> String {
        let qubitCount = circuit.qubits
        let operations = circuit.operations

        if operations.isEmpty {
            return "Empty circuit"
        }

        let layerAssignments = CircuitDiagramUtilities.assignLayers(operations: operations, qubitCount: qubitCount)
        let layerCount = computeLayerCount(layerAssignments)
        let symbolGrid = buildSymbolGrid(
            operations: operations,
            layerAssignments: layerAssignments,
            layerCount: layerCount,
            qubitCount: qubitCount,
        )
        let columnWidths = computeColumnWidths(symbolGrid: symbolGrid, layerCount: layerCount, qubitCount: qubitCount)
        let connectorGrid = buildConnectorGrid(
            operations: operations,
            layerAssignments: layerAssignments,
            layerCount: layerCount,
            qubitCount: qubitCount,
        )

        return renderOutput(
            qubitCount: qubitCount,
            layerCount: layerCount,
            symbolGrid: symbolGrid,
            columnWidths: columnWidths,
            connectorGrid: connectorGrid,
        )
    }

    /// Compute total number of layers from assignments.
    @inline(__always)
    @_effects(readonly)
    @usableFromInline
    static func computeLayerCount(_ layerAssignments: [Int]) -> Int {
        guard let m = layerAssignments.max() else { return 0 }
        return m + 1
    }

    /// Build 2D grid of gate symbol strings indexed by [qubit][layer].
    @_effects(readonly)
    @_optimize(speed)
    @usableFromInline
    static func buildSymbolGrid(
        operations: [CircuitOperation],
        layerAssignments: [Int],
        layerCount: Int,
        qubitCount: Int,
    ) -> [[String]] {
        var grid = (0 ..< qubitCount).map { _ in [String](repeating: "", count: layerCount) }

        for opIndex in operations.indices {
            let operation = operations[opIndex]
            let layer = layerAssignments[opIndex]
            let qubits = operation.qubits

            switch operation {
            case let .gate(gate, opQubits, _):
                let symbols = gateSymbols(gate: gate, qubits: opQubits)
                for symbolIndex in qubits.indices {
                    let q = qubits[symbolIndex]
                    if q < qubitCount, symbolIndex < symbols.count {
                        grid[q][layer] = symbols[symbolIndex]
                    }
                }
                if qubits.count > 1 {
                    let range = qubitRange(qubits)
                    let minQ = range.min
                    let maxQ = range.max
                    if minQ + 1 < maxQ {
                        for q in (minQ + 1) ..< maxQ {
                            if q < qubitCount, grid[q][layer].isEmpty {
                                grid[q][layer] = "\u{2502}"
                            }
                        }
                    }
                }
            case let .reset(qubit, _):
                if qubit < qubitCount {
                    grid[qubit][layer] = "|0\u{27E9}"
                }
            case let .measure(qubit, _, _):
                if qubit < qubitCount {
                    grid[qubit][layer] = "M"
                }
            }
        }

        return grid
    }

    /// Determine display symbols for each qubit involved in a gate.
    @_effects(readonly)
    @usableFromInline
    static func gateSymbols(gate: QuantumGate, qubits: [Int]) -> [String] {
        switch gate {
        case .cnot:
            return ["\u{25CF}", "\u{2295}"]
        case .cz:
            return ["\u{25CF}", "Z"]
        case .cy:
            return ["\u{25CF}", "Y"]
        case .ch:
            return ["\u{25CF}", "H"]
        case let .controlledPhase(theta):
            return ["\u{25CF}", "P(\(theta))"]
        case let .controlledRotationX(theta):
            return ["\u{25CF}", "Rx(\(theta))"]
        case let .controlledRotationY(theta):
            return ["\u{25CF}", "Ry(\(theta))"]
        case let .controlledRotationZ(theta):
            return ["\u{25CF}", "Rz(\(theta))"]
        case .swap:
            return ["\u{00D7}", "\u{00D7}"]
        case .sqrtSwap:
            return ["\u{221A}SW", "\u{221A}SW"]
        case .iswap:
            return ["iSW", "iSW"]
        case .sqrtISwap:
            return ["\u{221A}iSW", "\u{221A}iSW"]
        case .fswap:
            return ["fSW", "fSW"]
        case .toffoli:
            return ["\u{25CF}", "\u{25CF}", "\u{2295}"]
        case .fredkin:
            return ["\u{25CF}", "\u{00D7}", "\u{00D7}"]
        case .ccz:
            return ["\u{25CF}", "\u{25CF}", "Z"]
        case let .controlled(innerGate, controls):
            var symbols = [String](repeating: "\u{25CF}", count: controls.count)
            let innerSymbols = innerGateSymbols(innerGate)
            symbols.append(contentsOf: innerSymbols)
            return symbols
        default:
            let label = singleGateLabel(gate)
            if qubits.count == 1 {
                return [label]
            }
            var symbols = [String]()
            symbols.reserveCapacity(qubits.count)
            symbols.append("[\(label)]")
            for _ in 1 ..< qubits.count {
                symbols.append("\u{2502}")
            }
            return symbols
        }
    }

    /// Produce symbols for the inner gate of a controlled gate.
    @_effects(readonly)
    @usableFromInline
    static func innerGateSymbols(_ gate: QuantumGate) -> [String] {
        switch gate {
        case .pauliX:
            return ["\u{2295}"]
        default:
            let label = singleGateLabel(gate)
            let count = gate.qubitsRequired
            if count == 1 {
                return [label]
            }
            var symbols = [String]()
            symbols.reserveCapacity(count)
            symbols.append("[\(label)]")
            for _ in 1 ..< count {
                symbols.append("\u{2502}")
            }
            return symbols
        }
    }

    /// Map a gate to its ASCII-specific display label string.
    @_effects(readonly)
    @usableFromInline
    static func singleGateLabel(_ gate: QuantumGate) -> String {
        if let shared = CircuitDiagramUtilities.gateLabel(gate) {
            return shared
        }
        return switch gate {
        case let .phase(angle): "P(\(angle))"
        case let .rotationX(theta): "Rx(\(theta))"
        case let .rotationY(theta): "Ry(\(theta))"
        case let .rotationZ(theta): "Rz(\(theta))"
        case let .u1(lambda): "U1(\(lambda))"
        case let .u2(phi, lambda): "U2(\(phi),\(lambda))"
        case let .u3(theta, phi, lambda): "U3(\(theta),\(phi),\(lambda))"
        case let .globalPhase(phi): "GP(\(phi))"
        case let .givens(theta): "Giv(\(theta))"
        case let .xx(theta): "XX(\(theta))"
        case let .yy(theta): "YY(\(theta))"
        case let .zz(theta): "ZZ(\(theta))"
        case .customTwoQubit: "U2"
        case let .diagonal(phases): "Diag(\(phases.count))"
        case let .multiplexor(unitaries): "Mux(\(unitaries.count))"
        default: gate.description
        }
    }

    /// Compute the rendering width for each layer column.
    @_effects(readonly)
    @usableFromInline
    static func computeColumnWidths(symbolGrid: [[String]], layerCount: Int, qubitCount: Int) -> [Int] {
        var widths = [Int](unsafeUninitializedCapacity: layerCount) { buffer, count in
            buffer.initialize(repeating: 3)
            count = layerCount
        }
        for layer in 0 ..< layerCount {
            for qubit in 0 ..< qubitCount {
                let symbol = symbolGrid[qubit][layer]
                if !symbol.isEmpty {
                    let len = symbol.count + 2
                    widths[layer] = max(widths[layer], len)
                }
            }
        }
        return widths
    }

    /// Build 2D grid tracking vertical connector requirements indexed by [qubit][layer].
    @_effects(readonly)
    @_optimize(speed)
    @usableFromInline
    static func buildConnectorGrid(
        operations: [CircuitOperation],
        layerAssignments: [Int],
        layerCount: Int,
        qubitCount: Int,
    ) -> [[Bool]] {
        var grid = (0 ..< qubitCount).map { _ in [Bool](repeating: false, count: layerCount) }

        for opIndex in operations.indices {
            let operation = operations[opIndex]
            let layer = layerAssignments[opIndex]
            let qubits = operation.qubits
            if qubits.count > 1 {
                let range = qubitRange(qubits)
                for q in range.min ... range.max {
                    if q < qubitCount {
                        grid[q][layer] = true
                    }
                }
            }
        }

        return grid
    }

    /// Render the final ASCII output string from the computed grids.
    @_effects(readonly)
    @_optimize(speed)
    @usableFromInline
    static func renderOutput(
        qubitCount: Int,
        layerCount: Int,
        symbolGrid: [[String]],
        columnWidths: [Int],
        connectorGrid: [[Bool]],
    ) -> String {
        let labelWidth = CircuitDiagramUtilities.qubitLabelWidth(qubitCount)
        let totalWidth = labelWidth + 2 + columnWidths.reduce(0, +) + 4

        let totalLines = qubitCount * 2 - 1
        var result = String()
        result.reserveCapacity(totalLines * totalWidth)

        for line in 0 ..< totalLines {
            if line > 0 {
                result += "\n"
            }

            let isWireLine = line % 2 == 0
            let qubit = line / 2

            if isWireLine {
                let label = "q\(qubit)"
                result += label
                let padding = labelWidth - label.count
                result += String(repeating: " ", count: padding)
                result += ": "

                for layer in 0 ..< layerCount {
                    let symbol = symbolGrid[qubit][layer]
                    let width = columnWidths[layer]

                    if !symbol.isEmpty {
                        let symLen = symbol.count
                        let leftDashes = (width - symLen) / 2
                        let rightDashes = width - symLen - leftDashes
                        result += String(repeating: "\u{2500}", count: leftDashes)
                        result += symbol
                        result += String(repeating: "\u{2500}", count: rightDashes)
                    } else {
                        result += String(repeating: "\u{2500}", count: width)
                    }
                }

                result += "\u{2500}"
            } else {
                result += String(repeating: " ", count: labelWidth)
                result += "  "

                for layer in 0 ..< layerCount {
                    let width = columnWidths[layer]
                    let aboveQubit = line / 2
                    let belowQubit = line / 2 + 1

                    let hasConnector = aboveQubit < qubitCount
                        && belowQubit < qubitCount
                        && connectorGrid[aboveQubit][layer]
                        && connectorGrid[belowQubit][layer]

                    if hasConnector {
                        let mid = width / 2
                        result += String(repeating: " ", count: mid)
                        result += "\u{2502}"
                        result += String(repeating: " ", count: width - mid - 1)
                    } else {
                        result += String(repeating: " ", count: width)
                    }
                }

                result += " "
            }
        }

        return result
    }

    /// Compute the minimum and maximum qubit indices from a qubit array.
    @_effects(readonly)
    @usableFromInline
    static func qubitRange(_ qubits: [Int]) -> (min: Int, max: Int) {
        guard !qubits.isEmpty else { return (0, 0) }
        var minQ = qubits[0]
        var maxQ = qubits[0]
        for q in qubits {
            minQ = min(minQ, q)
            maxQ = max(maxQ, q)
        }
        return (minQ, maxQ)
    }
}
