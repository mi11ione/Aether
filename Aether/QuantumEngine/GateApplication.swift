//
//  GateApplication.swift
//  Aether
//
//  Created by mi11ion on 21/10/25.
//

import Foundation

/// Gate application engine - core quantum simulation logic.
///
/// Applies quantum gates to quantum states through matrix-vector multiplication.
/// Implements efficient algorithms that avoid computing full tensor products.
///
/// Mathematical basis:
/// - Full state transformation: |ψ'⟩ = U|ψ⟩
/// - Single-qubit gate affects 2^(n-1) amplitude pairs
/// - Multi-qubit gate affects larger amplitude groups
/// - Complexity: O(2^n) operations per gate (cannot be faster)
///
/// Architecture: Generic over n qubits - no hardcoded limits.
/// Works identically for 1-24+ qubits, only difference is array size.
enum GateApplication {
    // MARK: - Main Application Function

    /// Apply quantum gate to state at specified qubit(s)
    /// - Parameters:
    ///   - gate: Gate to apply
    ///   - qubits: Target qubit indices (for single-qubit gates)
    ///   - state: Current quantum state
    /// - Returns: New transformed quantum state
    static func apply(gate: QuantumGate, to qubits: [Int], state: QuantumState) -> QuantumState {
        // Validate qubit indices
        precondition(qubits.allSatisfy { $0 >= 0 && $0 < state.numQubits },
                     "Qubit indices out of bounds")

        // Dispatch based on gate type
        switch gate {
        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard,
             .phase, .sGate, .tGate, .rotationX, .rotationY, .rotationZ:
            // Single-qubit gates
            precondition(qubits.count == 1, "Single-qubit gate requires exactly 1 qubit")
            return applySingleQubitGate(gate: gate, qubit: qubits[0], state: state)

        case let .cnot(control, target):
            return applyCNOT(control: control, target: target, state: state)

        case let .controlledPhase(_, control, target):
            return applyTwoQubitGate(gate: gate, control: control, target: target, state: state)

        case let .swap(q1, q2):
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
        // Get 2×2 gate matrix
        let gateMatrix = gate.matrix()
        let g00 = gateMatrix[0][0]
        let g01 = gateMatrix[0][1]
        let g10 = gateMatrix[1][0]
        let g11 = gateMatrix[1][1]

        // Create new amplitude array (allocate directly instead of copying)
        var newAmplitudes = [Complex<Double>](repeating: .zero, count: state.stateSpaceSize)

        // Process amplitude pairs
        // For each pair (i, j) differing only at target qubit bit:
        // - i has qubit=0, j has qubit=1
        // - Apply 2×2 transformation
        let stateSize = state.stateSpaceSize
        let bitMask = 1 << qubit

        for i in 0 ..< stateSize {
            // Only process indices where target qubit is 0
            // (to avoid processing each pair twice)
            if (i & bitMask) == 0 {
                let j = i | bitMask // Index with qubit=1

                let ci = state.amplitudes[i] // Amplitude for qubit=0
                let cj = state.amplitudes[j] // Amplitude for qubit=1

                // Matrix multiplication: [g00 g01] [ci]
                //                        [g10 g11] [cj]
                newAmplitudes[i] = g00 * ci + g01 * cj
                newAmplitudes[j] = g10 * ci + g11 * cj
            } else {
                // This amplitude was already set in the pair processing above
                // Skip to avoid double processing
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
        // Get 4×4 gate matrix
        let gateMatrix = gate.matrix()

        // Create new amplitude array
        var newAmplitudes = state.amplitudes

        // Process 4-amplitude groups
        // States differing only in control and target qubits
        let controlMask = 1 << control
        let targetMask = 1 << target
        let bothMask = controlMask | targetMask

        let stateSize = state.stateSpaceSize

        for i in 0 ..< stateSize {
            // Only process base index (both control and target = 0)
            if (i & bothMask) == 0 {
                // Four states in this group
                let i00 = i // control=0, target=0
                let i01 = i | targetMask // control=0, target=1
                let i10 = i | controlMask // control=1, target=0
                let i11 = i | bothMask // control=1, target=1

                // Extract 4 amplitudes
                let c00 = state.amplitudes[i00]
                let c01 = state.amplitudes[i01]
                let c10 = state.amplitudes[i10]
                let c11 = state.amplitudes[i11]

                // Apply 4×4 matrix
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

        // For each state: check control bit
        for i in 0 ..< state.stateSpaceSize {
            if (i & controlMask) != 0 {
                // Control is 1: flip target bit
                let j = i ^ targetMask
                newAmplitudes[j] = state.amplitudes[i]
            } else {
                // Control is 0: identity (no flip)
                newAmplitudes[i] = state.amplitudes[i]
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

        // For each state: check both control bits
        for i in 0 ..< state.stateSpaceSize {
            if (i & c1Mask) != 0, (i & c2Mask) != 0 {
                // Both controls are 1: flip target
                let j = i ^ targetMask
                newAmplitudes[j] = state.amplitudes[i]
            } else {
                // Otherwise: identity (no flip)
                newAmplitudes[i] = state.amplitudes[i]
            }
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }
}

// MARK: - Convenience Extension

extension QuantumState {
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
