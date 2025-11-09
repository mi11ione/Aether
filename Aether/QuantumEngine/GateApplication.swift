// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Gate application: CPU-based quantum gate execution engine
///
/// Implements efficient quantum gate application through optimized matrix-vector multiplication
/// without computing exponentially large tensor products. This is the core classical simulation
/// algorithm that transforms quantum states according to gate matrices.
///
/// **Mathematical foundation**:
/// - State transformation: |ψ'⟩ = U|ψ⟩ where U is unitary gate matrix
/// - Single-qubit gate: Affects 2^(n-1) amplitude pairs in statevector
/// - Two-qubit gate: Affects 2^(n-2) amplitude quartets
/// - Computational complexity: O(2^n) per gate (fundamental quantum simulation limit)
///
/// **Architecture**:
/// - Zero tensor products: Never computes full 2^n × 2^n matrices
/// - Direct amplitude updates: Applies 2×2 or 4×4 matrices to state vector subsets
/// - Generic qubit count: Works identically for 1-30+ qubits
/// - Optimized special cases: CNOT, CZ, Toffoli faster than general matrix multiplication
///
/// **Algorithm efficiency**:
/// - Single-qubit: Only processes (qubit=0) indices, updates pairs via XOR
/// - Two-qubit: Masks select 4-tuples differing only in control/target bits
/// - CNOT: Simple amplitude swap, no matrix multiplication
/// - Toffoli: Conditional swap based on control qubit states
///
/// **Little-endian qubit ordering**:
/// Qubit 0 is LSB in state index. For |01⟩ (state index 1):
/// - Qubit 0 = |1⟩ (LSB)
/// - Qubit 1 = |0⟩ (MSB)
///
/// Example:
/// ```swift
/// // Apply Hadamard to qubit 0: creates superposition
/// let initial = QuantumState(numQubits: 2)  // |00⟩
/// let after = GateApplication.apply(gate: .hadamard, to: [0], state: initial)
/// // Result: (|00⟩ + |01⟩)/√2
///
/// // Apply CNOT(0→1): creates Bell state
/// let cnot = QuantumGate.cnot(control: 0, target: 1)
/// let bell = GateApplication.apply(gate: cnot, to: [], state: after)
/// // Result: (|00⟩ + |11⟩)/√2
///
/// // Custom gate application
/// let rotationY = QuantumGate.rotationY(angle: .pi / 4)
/// let rotated = GateApplication.apply(gate: rotationY, to: [1], state: bell)
///
/// // Convenience method
/// let result = initial.applying(gate: .hadamard, toQubit: 0)
/// ```
public enum GateApplication {
    // MARK: - Main Application Function

    /// Apply quantum gate to state at specified qubit(s)
    /// - Parameters:
    ///   - gate: Gate to apply
    ///   - qubits: Target qubit indices (for single-qubit gates)
    ///   - state: Current quantum state
    /// - Returns: New transformed quantum state
    public static func apply(gate: QuantumGate, to qubits: [Int], state: QuantumState) -> QuantumState {
        precondition(qubits.allSatisfy { $0 >= 0 && $0 < state.numQubits },
                     "Qubit indices out of bounds")

        switch gate {
        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard,
             .phase, .sGate, .tGate, .rotationX, .rotationY, .rotationZ,
             .u1, .u2, .u3, .sx, .sy, .customSingleQubit:
            precondition(qubits.count == 1, "Single-qubit gate requires exactly 1 qubit")
            return applySingleQubitGate(gate: gate, qubit: qubits[0], state: state)

        case let .cnot(control, target):
            return applyCNOT(control: control, target: target, state: state)

        case let .cz(control, target):
            return applyCZ(control: control, target: target, state: state)

        case let .cy(control, target),
             let .ch(control, target),
             let .controlledPhase(_, control, target),
             let .controlledRotationX(_, control, target),
             let .controlledRotationY(_, control, target),
             let .controlledRotationZ(_, control, target),
             let .customTwoQubit(_, control, target):
            return applyTwoQubitGate(gate: gate, control: control, target: target, state: state)

        case let .swap(q1, q2), let .sqrtSwap(q1, q2):
            return applyTwoQubitGate(gate: gate, control: q1, target: q2, state: state)

        case let .toffoli(c1, c2, target):
            return applyToffoli(control1: c1, control2: c2, target: target, state: state)
        }
    }

    // MARK: - Single-Qubit Gate Application

    /// Apply single-qubit gate to specified qubit
    /// - Parameters:
    ///   - gate: Single-qubit gate
    ///   - qubit: Target qubit index
    ///   - state: Current quantum state
    /// - Returns: Transformed state
    private static func applySingleQubitGate(
        gate: QuantumGate,
        qubit: Int,
        state: QuantumState
    ) -> QuantumState {
        let gateMatrix = gate.matrix()
        let g00 = gateMatrix[0][0]
        let g01 = gateMatrix[0][1]
        let g10 = gateMatrix[1][0]
        let g11 = gateMatrix[1][1]

        // Create new amplitude array (allocate directly instead of copying)
        var newAmplitudes = [Complex<Double>](repeating: .zero, count: state.stateSpaceSize)
        let bitMask = 1 << qubit

        for i in 0 ..< state.stateSpaceSize {
            // Only process indices where target qubit is 0
            // (to avoid processing each pair twice)
            if (i & bitMask) == 0 {
                let j = i | bitMask

                let ci = state.amplitudes[i]
                let cj = state.amplitudes[j]

                newAmplitudes[i] = g00 * ci + g01 * cj
                newAmplitudes[j] = g10 * ci + g11 * cj
            }
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    // MARK: - Two-Qubit Gate Application

    /// Apply two-qubit gate (general 4×4 matrix)
    /// - Parameters:
    ///   - gate: Two-qubit gate
    ///   - control: Control qubit index
    ///   - target: Target qubit index
    ///   - state: Current quantum state
    /// - Returns: Transformed state
    private static func applyTwoQubitGate(
        gate: QuantumGate,
        control: Int,
        target: Int,
        state: QuantumState
    ) -> QuantumState {
        let gateMatrix = gate.matrix()
        var newAmplitudes = state.amplitudes

        // States differing only in control and target qubits
        let controlMask = 1 << control
        let targetMask = 1 << target
        let bothMask = controlMask | targetMask

        let stateSize = state.stateSpaceSize

        for i in 0 ..< stateSize {
            if (i & bothMask) == 0 {
                let i00 = i // control=0, target=0
                let i01 = i | targetMask // control=0, target=1
                let i10 = i | controlMask // control=1, target=0
                let i11 = i | bothMask // control=1, target=1

                let c00 = state.amplitudes[i00]
                let c01 = state.amplitudes[i01]
                let c10 = state.amplitudes[i10]
                let c11 = state.amplitudes[i11]

                newAmplitudes[i00] = gateMatrix[0][0] * c00 + gateMatrix[0][1] * c01 +
                    gateMatrix[0][2] * c10 + gateMatrix[0][3] * c11
                newAmplitudes[i01] = gateMatrix[1][0] * c00 + gateMatrix[1][1] * c01 +
                    gateMatrix[1][2] * c10 + gateMatrix[1][3] * c11
                newAmplitudes[i10] = gateMatrix[2][0] * c00 + gateMatrix[2][1] * c01 +
                    gateMatrix[2][2] * c10 + gateMatrix[2][3] * c11
                newAmplitudes[i11] = gateMatrix[3][0] * c00 + gateMatrix[3][1] * c01 +
                    gateMatrix[3][2] * c10 + gateMatrix[3][3] * c11
            }
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    // MARK: - CNOT

    /// CNOT gate application
    /// Much faster than general 4×4 matrix multiplication
    /// - Parameters:
    ///   - control: Control qubit index
    ///   - target: Target qubit index
    ///   - state: Current quantum state
    /// - Returns: Transformed state
    private static func applyCNOT(
        control: Int,
        target: Int,
        state: QuantumState
    ) -> QuantumState {
        var newAmplitudes = Array(repeating: Complex<Double>.zero, count: state.stateSpaceSize)

        let controlMask = 1 << control
        let targetMask = 1 << target

        for i in 0 ..< state.stateSpaceSize {
            if (i & controlMask) != 0 {
                newAmplitudes[i ^ targetMask] = state.amplitudes[i]
            } else {
                newAmplitudes[i] = state.amplitudes[i]
            }
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    /// CZ gate application (Controlled-Z)
    /// Diagonal gate: only applies phase when both qubits are |1⟩
    /// Much faster than general 4×4 matrix multiplication
    /// - Parameters:
    ///   - control: Control qubit index
    ///   - target: Target qubit index
    ///   - state: Current quantum state
    /// - Returns: Transformed state
    private static func applyCZ(
        control: Int,
        target: Int,
        state: QuantumState
    ) -> QuantumState {
        var newAmplitudes = state.amplitudes

        let controlMask = 1 << control
        let targetMask = 1 << target

        for i in 0 ..< state.stateSpaceSize {
            if (i & controlMask) != 0, (i & targetMask) != 0 {
                newAmplitudes[i] = -newAmplitudes[i]
            }
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    // MARK: - Toffoli Gate Application

    /// Apply Toffoli (CCNOT) gate
    /// - Parameters:
    ///   - control1: First control qubit
    ///   - control2: Second control qubit
    ///   - target: Target qubit
    ///   - state: Current quantum state
    /// - Returns: Transformed state
    private static func applyToffoli(
        control1: Int,
        control2: Int,
        target: Int,
        state: QuantumState
    ) -> QuantumState {
        var newAmplitudes = Array(repeating: Complex<Double>.zero, count: state.stateSpaceSize)

        let c1Mask = 1 << control1
        let c2Mask = 1 << control2
        let targetMask = 1 << target

        for i in 0 ..< state.stateSpaceSize {
            if (i & c1Mask) != 0, (i & c2Mask) != 0 {
                newAmplitudes[i ^ targetMask] = state.amplitudes[i]
            } else {
                newAmplitudes[i] = state.amplitudes[i]
            }
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }
}

// MARK: - Convenience Extension

public extension QuantumState {
    /// Apply gate to this state (convenience method)
    /// - Parameters:
    ///   - gate: Gate to apply
    ///   - qubits: Target qubit indices
    /// - Returns: New transformed state
    func applying(gate: QuantumGate, to qubits: [Int]) -> QuantumState {
        GateApplication.apply(gate: gate, to: qubits, state: self)
    }

    /// Apply gate to single qubit (convenience)
    /// - Parameters:
    ///   - gate: Single-qubit gate
    ///   - qubit: Target qubit
    /// - Returns: New transformed state
    func applying(gate: QuantumGate, toQubit qubit: Int) -> QuantumState {
        GateApplication.apply(gate: gate, to: [qubit], state: self)
    }
}
