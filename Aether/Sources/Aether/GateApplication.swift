// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// CPU-based gate execution using optimized matrix-vector multiplication
///
/// Transforms quantum states by applying gate matrices without computing full 2^n x 2^n tensor products.
/// The core classical simulation algorithm that makes n-qubit systems tractable: a Hadamard gate on qubit 0
/// of a 20-qubit state processes 2^19 amplitude pairs in milliseconds rather than constructing a million-by-million
/// matrix.
///
/// **How it works**:
/// - Single-qubit gates: Apply 2x2 matrix to 2^(n-1) amplitude pairs where target qubit differs
/// - Two-qubit gates: Apply 4x4 matrix to 2^(n-2) amplitude quartets where control/target bits vary
/// - Special cases: CNOT/CZ/Toffoli use conditional swaps instead of matrix multiplication (2-3x faster)
///
/// **Mathematical foundation**:
/// State transformation |ψ'⟩ = U|ψ⟩ where U is unitary. For single-qubit gate on qubit q, only amplitudes
/// at indices differing in bit q are coupled. This reduces full O(4^n) tensor product to O(2^n) targeted updates.
///
/// **Qubit indexing (little-endian)**:
/// Qubit 0 is LSB in basis state index. State |01⟩ has index 1 in binary:
/// - Bit 0 (qubit 0) = 1 -> qubit is |1⟩
/// - Bit 1 (qubit 1) = 0 -> qubit is |0⟩
///
/// **Performance characteristics**:
/// - Complexity: O(2^n) per gate (optimal for classical simulation)
/// - Best for: n < 10 qubits (pure CPU, no GPU overhead)
/// - See ``MetalGateApplication`` for n ≥ 10 qubits (GPU provides 2-10x speedup)
///
/// **Example**:
/// ```swift
/// let state = QuantumState(numQubits: 2)  // |00⟩
///
/// // Single-qubit gate: Hadamard on qubit 0
/// let superposition = GateApplication.apply(.hadamard, to: 0, state: state)
/// // (|00⟩ + |01⟩)/√2
///
/// // Two-qubit gate: CNOT with control=0, target=1
/// let entangled = GateApplication.apply(.cnot, to: [0, 1], state: superposition)
/// // (|00⟩ + |11⟩)/√2 - Bell state
///
/// // Convenience method on QuantumState
/// let rotated = state.applying(.rotationY(angle: .pi / 4), to: 1)
/// ```
///
/// - SeeAlso: ``MetalGateApplication``, ``QuantumSimulator``, ``QuantumCircuit``
public enum GateApplication {
    // MARK: - Main Application Function

    /// Apply gate to quantum state at specified qubits
    ///
    /// Transforms state by applying gate's unitary matrix to amplitudes at target qubit indices.
    /// For single-qubit gates, pass one index. For two-qubit gates (CNOT, CZ, SWAP), pass two.
    /// For Toffoli, pass three [control1, control2, target].
    ///
    /// **Example**:
    /// ```swift
    /// let state = QuantumState(numQubits: 3)
    /// let withH = GateApplication.apply(.hadamard, to: 0, state: state)
    /// let withCNOT = GateApplication.apply(.cnot, to: [0, 1], state: withH)
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply
    ///   - qubits: Target qubit indices (single element for single-qubit gates, multiple for multi-qubit)
    ///   - state: Input quantum state
    /// - Returns: Transformed state with gate applied
    /// - Complexity: O(2^n) time, O(2^n) space
    /// - Precondition: All qubits must be valid indices for state
    @_effects(readonly)
    @inlinable
    @_eagerMove
    public static func apply(_ gate: QuantumGate, to qubits: [Int], state: QuantumState) -> QuantumState {
        ValidationUtilities.validateOperationQubits(qubits, numQubits: state.numQubits)

        switch gate {
        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard,
             .phase, .sGate, .tGate, .rotationX, .rotationY, .rotationZ,
             .u1, .u2, .u3, .sx, .sy, .customSingleQubit:
            ValidationUtilities.validateSingleQubitGate(qubits)
            return applySingleQubitGate(gate: gate, qubit: qubits[0], state: state)

        case .cnot:
            return applyCNOT(control: qubits[0], target: qubits[1], state: state)

        case .cz:
            return applyCZ(control: qubits[0], target: qubits[1], state: state)

        case .cy, .ch, .controlledPhase, .controlledRotationX, .controlledRotationY, .controlledRotationZ, .customTwoQubit:
            return applyTwoQubitGate(gate: gate, control: qubits[0], target: qubits[1], state: state)

        case .swap, .sqrtSwap:
            return applyTwoQubitGate(gate: gate, control: qubits[0], target: qubits[1], state: state)

        case .toffoli:
            return applyToffoli(control1: qubits[0], control2: qubits[1], target: qubits[2], state: state)
        }
    }

    /// Apply gate to single qubit (convenience method)
    ///
    /// Wraps qubit index in array and delegates to main apply method.
    /// Cleaner syntax for single-qubit gates.
    ///
    /// **Example**:
    /// ```swift
    /// let state = QuantumState(numQubits: 2)
    /// let result = GateApplication.apply(.hadamard, to: 0, state: state)
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply
    ///   - qubit: Target qubit index
    ///   - state: Input quantum state
    /// - Returns: Transformed state with gate applied
    /// - Complexity: O(2^n) time, O(2^n) space
    /// - Precondition: qubit must be valid index for state
    @_effects(readonly)
    @inlinable
    @_eagerMove
    public static func apply(_ gate: QuantumGate, to qubit: Int, state: QuantumState) -> QuantumState {
        apply(gate, to: [qubit], state: state)
    }

    // MARK: - Single-Qubit Gate Application

    /// Apply 2x2 gate matrix to single qubit
    ///
    /// Processes 2^(n-1) amplitude pairs (i, j) where i and j differ only in target qubit bit.
    /// Uses bitmask to identify pairs: j = i | (1 << qubit). Computes new amplitudes via
    /// matrix-vector multiplication: [c'_i, c'_j] = G x [c_i, c_j] where G is 2x2 gate matrix.
    ///
    /// - Complexity: O(2^n) - processes half the state space, each pair once
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func applySingleQubitGate(
        gate: QuantumGate,
        qubit: Int,
        state: QuantumState
    ) -> QuantumState {
        let gateMatrix = gate.matrix()
        let g00: Complex<Double> = gateMatrix[0][0]
        let g01: Complex<Double> = gateMatrix[0][1]
        let g10: Complex<Double> = gateMatrix[1][0]
        let g11: Complex<Double> = gateMatrix[1][1]

        let stateSize = state.stateSpaceSize
        let bitMask = BitUtilities.bitMask(qubit: qubit)

        let newAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
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

    /// Apply 4x4 gate matrix to two qubits
    ///
    /// Processes 2^(n-2) amplitude quartets where control and target qubits vary independently.
    /// Each quartet (c00, c01, c10, c11) maps to indices differing only in the two target bits.
    /// Applies full 4x4 matrix multiplication to transform all four amplitudes simultaneously.
    ///
    /// - Complexity: O(2^n) - processes quarter of state space, 16 operations per quartet
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
        let gateMatrix = gate.matrix()
        let stateSize = state.stateSpaceSize

        let controlMask = BitUtilities.bitMask(qubit: control)
        let targetMask = BitUtilities.bitMask(qubit: target)
        let bothMask = controlMask | targetMask

        let g00 = gateMatrix[0][0], g01 = gateMatrix[0][1], g02 = gateMatrix[0][2], g03 = gateMatrix[0][3]
        let g10 = gateMatrix[1][0], g11 = gateMatrix[1][1], g12 = gateMatrix[1][2], g13 = gateMatrix[1][3]
        let g20 = gateMatrix[2][0], g21 = gateMatrix[2][1], g22 = gateMatrix[2][2], g23 = gateMatrix[2][3]
        let g30 = gateMatrix[3][0], g31 = gateMatrix[3][1], g32 = gateMatrix[3][2], g33 = gateMatrix[3][3]

        let newAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
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

    /// Apply CNOT via conditional amplitude swap
    ///
    /// When control qubit is |1⟩, flips target qubit (X gate). When control is |0⟩, does nothing.
    /// Implemented as conditional swap: if bit(i, control)==1, swap amplitudes[i] <-> amplitudes[i⊕target].
    /// No matrix multiplication needed - just conditional index calculation and assignment.
    ///
    /// **Performance**: 2-3x faster than general 4x4 matrix due to avoiding 16 complex multiplications per quartet.
    ///
    /// - Complexity: O(2^n) - one pass through state, branch per amplitude
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

        let newAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
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

    /// Apply CZ (Controlled-Z) via conditional phase flip
    ///
    /// Diagonal gate that only modifies phase: negates amplitude when both qubits are |1⟩, leaves others unchanged.
    /// Implemented as single-pass conditional negation: if both bits set, negate amplitude. No matrix multiplication.
    ///
    /// **Performance**: 2-3x faster than general 4x4 matrix. Diagonal structure eliminates all off-diagonal terms.
    ///
    /// - Complexity: O(2^n) - one pass through state, simple branch per amplitude
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

        let newAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
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

    /// Apply Toffoli (CCNOT) via XOR-based conditional swap
    ///
    /// Flips target qubit when both controls are |1⟩. Implemented as: if both control bits set,
    /// swap amplitudes[i] <-> amplitudes[i⊕target] using XOR to compute paired index.
    /// Single pass with simple conditional logic, no 8x8 matrix multiplication.
    ///
    /// **Performance**: Similar speedup to CNOT vs general multi-qubit gates.
    ///
    /// - Complexity: O(2^n) - one pass, branch per amplitude
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

        let newAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
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
    ///
    /// Delegates to ``GateApplication`` with self as state parameter.
    /// Cleaner syntax for chaining transformations.
    ///
    /// **Example**:
    /// ```swift
    /// let state = QuantumState(numQubits: 2)
    ///     .applying(.hadamard, to: 0)
    ///     .applying(.cnot, to: [0, 1])
    /// // Bell state (|00⟩ + |11⟩)/√2
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply
    ///   - qubits: Target qubit indices
    /// - Returns: New transformed state
    /// - Complexity: O(2^n)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    func applying(_ gate: QuantumGate, to qubits: [Int]) -> QuantumState {
        GateApplication.apply(gate, to: qubits, state: self)
    }

    /// Apply gate to single qubit (convenience method)
    ///
    /// Wraps qubit index in array and delegates to ``GateApplication``.
    /// Preferred for single-qubit operations in fluent chains.
    ///
    /// **Example**:
    /// ```swift
    /// let state = QuantumState(numQubits: 2)
    ///     .applying(.hadamard, to: 0)
    ///     .applying(.rotationZ(theta: .pi / 4), to: 1)
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply
    ///   - qubit: Target qubit index
    /// - Returns: New transformed state
    /// - Complexity: O(2^n)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    func applying(_ gate: QuantumGate, to qubit: Int) -> QuantumState {
        GateApplication.apply(gate, to: qubit, state: self)
    }
}
