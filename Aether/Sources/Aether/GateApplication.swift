// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Amplitude vector for quantum states
public typealias AmplitudeVector = [Complex<Double>]

/// Gate matrix representation
public typealias GateMatrix = [AmplitudeVector]

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
/// // Apply CNOT(0->1): creates Bell state
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
@frozen
public enum GateApplication {
    // MARK: - Main Application Function

    /// Apply quantum gate to state at specified qubit(s)
    /// - Parameters:
    ///   - gate: Gate to apply
    ///   - qubits: Target qubit indices (for single-qubit gates)
    ///   - state: Current quantum state
    /// - Returns: New transformed quantum state
    @_effects(readonly)
    @inlinable
    @_eagerMove
    public static func apply(gate: QuantumGate, to qubits: [Int], state: QuantumState) -> QuantumState {
        ValidationUtilities.validateOperationQubits(qubits, numQubits: state.numQubits)

        switch gate {
        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard,
             .phase, .sGate, .tGate, .rotationX, .rotationY, .rotationZ,
             .u1, .u2, .u3, .sx, .sy, .customSingleQubit:
            ValidationUtilities.validateSingleQubitGate(qubits)
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
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func applySingleQubitGate(
        gate: QuantumGate,
        qubit: Int,
        state: QuantumState
    ) -> QuantumState {
        let gateMatrix: GateMatrix = gate.matrix()
        let g00: Complex<Double> = gateMatrix[0][0]
        let g01: Complex<Double> = gateMatrix[0][1]
        let g10: Complex<Double> = gateMatrix[1][0]
        let g11: Complex<Double> = gateMatrix[1][1]

        let stateSize = state.stateSpaceSize
        let bitMask = BitUtilities.bitMask(qubit: qubit)

        let newAmplitudes = AmplitudeVector(unsafeUninitializedCapacity: stateSize) { buffer, count in
            for i in 0 ..< stateSize where (i & bitMask) == 0 {
                let j = i | bitMask

                let ci = state.amplitudes[i]
                let cj = state.amplitudes[j]

                buffer[i] = g00 * ci + g01 * cj
                buffer[j] = g10 * ci + g11 * cj
            }
            count = stateSize
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
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func applyTwoQubitGate(
        gate: QuantumGate,
        control: Int,
        target: Int,
        state: QuantumState
    ) -> QuantumState {
        let gateMatrix: GateMatrix = gate.matrix()
        let stateSize = state.stateSpaceSize

        let controlMask = BitUtilities.bitMask(qubit: control)
        let targetMask = BitUtilities.bitMask(qubit: target)
        let bothMask = controlMask | targetMask

        let g00 = gateMatrix[0][0], g01 = gateMatrix[0][1], g02 = gateMatrix[0][2], g03 = gateMatrix[0][3]
        let g10 = gateMatrix[1][0], g11 = gateMatrix[1][1], g12 = gateMatrix[1][2], g13 = gateMatrix[1][3]
        let g20 = gateMatrix[2][0], g21 = gateMatrix[2][1], g22 = gateMatrix[2][2], g23 = gateMatrix[2][3]
        let g30 = gateMatrix[3][0], g31 = gateMatrix[3][1], g32 = gateMatrix[3][2], g33 = gateMatrix[3][3]

        let newAmplitudes = AmplitudeVector(unsafeUninitializedCapacity: stateSize) { buffer, count in
            for i in 0 ..< stateSize where (i & bothMask) == 0 {
                let i00 = i
                let i01 = i | targetMask
                let i10 = i | controlMask
                let i11 = i | bothMask

                let c00 = state.amplitudes[i00]
                let c01 = state.amplitudes[i01]
                let c10 = state.amplitudes[i10]
                let c11 = state.amplitudes[i11]

                buffer[i00] = g00 * c00 + g01 * c01 + g02 * c10 + g03 * c11
                buffer[i01] = g10 * c00 + g11 * c01 + g12 * c10 + g13 * c11
                buffer[i10] = g20 * c00 + g21 * c01 + g22 * c10 + g23 * c11
                buffer[i11] = g30 * c00 + g31 * c01 + g32 * c10 + g33 * c11
            }
            count = stateSize
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
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func applyCNOT(
        control: Int,
        target: Int,
        state: QuantumState
    ) -> QuantumState {
        let stateSize = state.stateSpaceSize
        let controlMask = BitUtilities.bitMask(qubit: control)

        let newAmplitudes = AmplitudeVector(unsafeUninitializedCapacity: stateSize) { buffer, count in
            for i in 0 ..< stateSize {
                if (i & controlMask) != 0 {
                    buffer[BitUtilities.flipBit(i, qubit: target)] = state.amplitudes[i]
                } else {
                    buffer[i] = state.amplitudes[i]
                }
            }
            count = stateSize
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
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func applyCZ(
        control: Int,
        target: Int,
        state: QuantumState
    ) -> QuantumState {
        let stateSize = state.stateSpaceSize
        let bothMask = BitUtilities.bitMask(qubit: control) | BitUtilities.bitMask(qubit: target)

        let newAmplitudes = AmplitudeVector(unsafeUninitializedCapacity: stateSize) { buffer, count in
            for i in 0 ..< stateSize {
                if (i & bothMask) == bothMask {
                    buffer[i] = -state.amplitudes[i]
                } else {
                    buffer[i] = state.amplitudes[i]
                }
            }
            count = stateSize
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
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func applyToffoli(
        control1: Int,
        control2: Int,
        target: Int,
        state: QuantumState
    ) -> QuantumState {
        let stateSize = state.stateSpaceSize
        let c1Mask = BitUtilities.bitMask(qubit: control1)
        let c2Mask = BitUtilities.bitMask(qubit: control2)
        let bothControlMask = c1Mask | c2Mask
        let targetMask = BitUtilities.bitMask(qubit: target)

        let newAmplitudes = AmplitudeVector(unsafeUninitializedCapacity: stateSize) { buffer, count in
            for i in 0 ..< stateSize {
                if (i & bothControlMask) == bothControlMask {
                    buffer[i ^ targetMask] = state.amplitudes[i]
                } else {
                    buffer[i] = state.amplitudes[i]
                }
            }
            count = stateSize
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
    @_effects(readonly)
    @inlinable
    @_eagerMove
    func applying(gate: QuantumGate, to qubits: [Int]) -> QuantumState {
        GateApplication.apply(gate: gate, to: qubits, state: self)
    }

    /// Apply gate to single qubit (convenience)
    /// - Parameters:
    ///   - gate: Single-qubit gate
    ///   - qubit: Target qubit
    /// - Returns: New transformed state
    @_effects(readonly)
    @inlinable
    @_eagerMove
    func applying(gate: QuantumGate, toQubit qubit: Int) -> QuantumState {
        GateApplication.apply(gate: gate, to: [qubit], state: self)
    }
}
