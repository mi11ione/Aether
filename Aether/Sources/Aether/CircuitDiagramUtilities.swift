// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Shared helpers used by both ASCII and Unicode circuit diagram renderers.
///
/// Provides the common scheduling algorithm, qubit label width computation,
/// and base gate-to-label mapping that ``CircuitDiagramASCII`` and
/// ``CircuitDiagramUnicode`` share. Each renderer may extend or override the
/// base gate label for renderer-specific formatting (e.g. angle display).
@usableFromInline
enum CircuitDiagramUtilities {
    /// Assign each operation to the earliest available time layer using greedy scheduling.
    @inlinable
    @_effects(readonly)
    @_optimize(speed)
    static func assignLayers(operations: [CircuitOperation], qubitCount: Int) -> [Int] {
        var nextAvailable = [Int](unsafeUninitializedCapacity: qubitCount) {
            buffer, count in
            buffer.initialize(repeating: 0)
            count = qubitCount
        }
        return [Int](unsafeUninitializedCapacity: operations.count) { buffer, count in
            for i in 0 ..< operations.count {
                let qubits = operations[i].qubits
                var layer = 0
                for q in qubits {
                    if q < qubitCount {
                        layer = max(layer, nextAvailable[q])
                    }
                }
                buffer[i] = layer
                for q in qubits {
                    if q < qubitCount {
                        nextAvailable[q] = layer + 1
                    }
                }
            }
            count = operations.count
        }
    }

    /// Compute the character width needed for qubit labels like "q0", "q12".
    @inline(__always)
    @inlinable
    @_effects(readonly)
    static func qubitLabelWidth(_ qubitCount: Int) -> Int {
        if qubitCount <= 1 { return 2 }
        var maxIndex = qubitCount - 1
        var digits = 1
        while maxIndex >= 10 {
            digits += 1
            maxIndex /= 10
        }
        return digits + 1
    }

    /// Return the display label for gates shared identically by both renderers, or nil if renderer-specific.
    @inline(__always)
    @inlinable
    @_effects(readonly)
    static func label(for gate: QuantumGate) -> String? {
        switch gate {
        case .identity: "I"
        case .pauliX: "X"
        case .pauliY: "Y"
        case .pauliZ: "Z"
        case .hadamard: "H"
        case .sGate: "S"
        case .tGate: "T"
        case .sx: "SX"
        case .sy: "SY"
        case .customSingleQubit: "U"
        case .customUnitary: "U"
        default: nil
        }
    }
}
