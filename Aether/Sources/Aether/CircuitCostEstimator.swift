// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Pre-execution resource analysis for quantum circuits providing gate counts, circuit depth,
/// CNOT-equivalent cost, and T-count for fault-tolerance budgeting.
///
/// Analyzes circuit resource requirements without executing the circuit, enabling cost comparison
/// between algorithms, fault-tolerance overhead estimation, and hardware resource planning.
/// CNOT-equivalent costs follow standard decomposition: single-qubit gates cost 0, CNOT/CZ cost 1,
/// Toffoli costs 6, and generic controlled unitaries cost 2.
///
/// **Example:**
/// ```swift
/// var circuit = QuantumCircuit(qubits: 3)
/// circuit.append(.hadamard, to: 0)
/// circuit.append(.cnot, to: [0, 1])
/// let cost = CircuitCostEstimator.estimate(circuit)
/// print(cost.totalGates, cost.depth, cost.cnotEquivalent)
/// ```
///
/// - SeeAlso: ``QuantumCircuit``
/// - SeeAlso: ``CircuitOptimizer``
public enum CircuitCostEstimator {
    /// Estimate resource costs for a quantum circuit.
    ///
    /// Performs single-pass analysis computing gate counts by type, circuit depth (critical path),
    /// CNOT-equivalent two-qubit gate cost, and T-gate count for fault-tolerance budgeting.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.grover(qubits: 3, target: 5)
    /// let cost = CircuitCostEstimator.estimate(circuit)
    /// print("Total gates: \(cost.totalGates)")
    /// print("Circuit depth: \(cost.depth)")
    /// print("CNOT-equivalent: \(cost.cnotEquivalent)")
    /// print("T-count: \(cost.tCount)")
    /// ```
    ///
    /// - Parameter circuit: Quantum circuit to analyze
    /// - Returns: Complete cost analysis including gate counts, depth, CNOT-equivalent, and T-count
    /// - Complexity: O(n) where n = operation count
    @_optimize(speed)
    @_effects(readonly)
    public static func estimate(_ circuit: QuantumCircuit) -> CircuitCost {
        var gateCount: [QuantumGate: Int] = [:]
        var totalGates = 0
        var cnotEquivalent = 0
        var tCount = 0
        var qubitDepth = [Int](repeating: 0, count: max(circuit.qubits, 1))

        for operation in circuit.operations {
            guard let gate = operation.gate else { continue }

            gateCount[gate, default: 0] += 1
            totalGates += 1

            cnotEquivalent += cnotEquivalentCost(gate)

            if case .tGate = gate {
                tCount += 1
            }

            var maxDepth = 0
            for qubit in operation.qubits {
                if qubit < qubitDepth.count {
                    maxDepth = max(maxDepth, qubitDepth[qubit])
                } else {
                    let newSize = qubit + 1
                    qubitDepth.append(contentsOf: [Int](repeating: 0, count: newSize - qubitDepth.count))
                }
            }

            let newDepth = maxDepth + 1
            for qubit in operation.qubits {
                if qubit < qubitDepth.count {
                    qubitDepth[qubit] = newDepth
                }
            }
        }

        let depth = qubitDepth.max() ?? 0

        return CircuitCost(
            gateCount: gateCount,
            depth: depth,
            cnotEquivalent: cnotEquivalent,
            tCount: tCount,
            totalGates: totalGates,
        )
    }

    /// Returns the CNOT-equivalent cost for a given gate type.
    @_optimize(speed)
    @_effects(readonly)
    private static func cnotEquivalentCost(_ gate: QuantumGate) -> Int {
        switch gate {
        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard, .phase, .sGate, .tGate,
             .rotationX, .rotationY, .rotationZ, .u1, .u2, .u3, .sx, .sy,
             .customSingleQubit, .globalPhase:
            0
        case .cnot, .cz:
            1
        case .cy, .ch, .controlledPhase, .controlledRotationX, .controlledRotationY, .controlledRotationZ,
             .sqrtSwap, .iswap, .sqrtISwap, .fswap, .givens, .xx, .yy, .zz, .customTwoQubit:
            2
        case .swap:
            3
        case .toffoli, .fredkin, .ccz:
            6
        case .diagonal, .multiplexor, .customUnitary:
            2
        case .controlled:
            2
        }
    }
}

/// Complete cost analysis of a quantum circuit for resource estimation and fault-tolerance budgeting.
///
/// Provides gate counts by type, circuit depth (critical path length), CNOT-equivalent two-qubit
/// gate cost, and T-gate count. T-count is critical for fault-tolerant quantum computing where
/// T-gates dominate resource overhead due to magic state distillation requirements.
///
/// **Example:**
/// ```swift
/// let cost = CircuitCostEstimator.estimate(circuit)
/// print(cost)  // "CircuitCost(gates: 10, depth: 5, CNOT-eq: 4, T-count: 2)"
/// ```
///
/// - SeeAlso: ``CircuitCostEstimator``
@frozen public struct CircuitCost: Sendable, Equatable, CustomStringConvertible {
    /// Gate count by type for detailed resource breakdown.
    public let gateCount: [QuantumGate: Int]

    /// Circuit depth (minimum sequential time steps assuming unlimited parallelism).
    public let depth: Int

    /// Total CNOT-equivalent two-qubit gate cost.
    public let cnotEquivalent: Int

    /// T-gate count for fault-tolerance resource estimation.
    public let tCount: Int

    /// Total number of gates in the circuit.
    public let totalGates: Int

    /// Human-readable summary of circuit cost metrics.
    @inlinable
    public var description: String {
        "CircuitCost(gates: \(totalGates), depth: \(depth), CNOT-eq: \(cnotEquivalent), T-count: \(tCount))"
    }
}
