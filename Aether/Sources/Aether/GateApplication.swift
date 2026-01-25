// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// CPU-based gate execution using optimized matrix-vector multiplication.
///
/// Transforms quantum states by applying gate matrices without computing full 2^n * 2^n tensor products.
/// Single-qubit gates apply 2*2 matrices to 2^(n-1) amplitude pairs, two-qubit gates apply 4*4 matrices
/// to 2^(n-2) quartets. CNOT, CZ, and Toffoli use conditional swaps instead of matrix multiplication.
///
/// Uses little-endian qubit indexing where qubit 0 is LSB in basis state index. Complexity is O(2^n)
/// per gate, optimal for classical simulation. Best for circuits under 10 qubits; use
/// ``MetalGateApplication`` for larger circuits.
///
/// **Example:**
/// ```swift
/// let state = QuantumState(qubits: 2)  // |00⟩
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
/// let rotated = state.applying(.rotationY(.pi / 4), to: 1)
/// ```
///
/// - SeeAlso: ``MetalGateApplication``
/// - SeeAlso: ``QuantumSimulator``
/// - SeeAlso: ``QuantumCircuit``
public enum GateApplication {
    // MARK: - Main Application Function

    /// Apply gate to quantum state at specified qubits
    ///
    /// Transforms state by applying gate's unitary matrix to amplitudes at target qubit indices.
    /// For single-qubit gates, pass one index. For two-qubit gates (CNOT, CZ, SWAP), pass two.
    /// For Toffoli, pass three [control1, control2, target].
    ///
    /// **Example:**
    /// ```swift
    /// let state = QuantumState(qubits: 3)
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
        ValidationUtilities.validateOperationQubits(qubits, numQubits: state.qubits)

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

        case .swap, .sqrtSwap, .sqrtISwap, .givens, .xx:
            return applyTwoQubitGate(gate: gate, control: qubits[0], target: qubits[1], state: state)

        case .iswap:
            return applyISwap(qubit1: qubits[0], qubit2: qubits[1], state: state)

        case .fswap:
            return applyFSwap(qubit1: qubits[0], qubit2: qubits[1], state: state)

        case .toffoli:
            return applyToffoli(control1: qubits[0], control2: qubits[1], target: qubits[2], state: state)

        case .fredkin:
            return applyFredkin(control: qubits[0], target1: qubits[1], target2: qubits[2], state: state)

        case let .controlled(innerGate, controls):
            return applyControlledGate(gate: innerGate, controls: controls, targetQubits: qubits, state: state)

        case .customUnitary, .multiplexor:
            return applyMultiQubitGate(gate: gate, qubits: qubits, state: state)

        case let .diagonal(phases):
            return applyDiagonal(phases: phases, qubits: qubits, state: state)
        }
    }

    /// Apply gate to single qubit (convenience method)
    ///
    /// Wraps qubit index in array and delegates to main apply method.
    /// Cleaner syntax for single-qubit gates.
    ///
    /// **Example:**
    /// ```swift
    /// let state = QuantumState(qubits: 2)
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
        state: QuantumState,
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

        return QuantumState(qubits: state.qubits, amplitudes: newAmplitudes)
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
        state: QuantumState,
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

        return QuantumState(qubits: state.qubits, amplitudes: newAmplitudes)
    }

    // MARK: - CNOT

    /// Apply CNOT via conditional amplitude swap.
    ///
    /// Flips target qubit when control is |1⟩, does nothing when |0⟩. Implemented as conditional swap
    /// without matrix multiplication, avoiding 16 complex multiplications per quartet.
    ///
    /// - Complexity: O(2^n) - one pass through state
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func applyCNOT(
        control: Int,
        target: Int,
        state: QuantumState,
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

        return QuantumState(qubits: state.qubits, amplitudes: newAmplitudes)
    }

    /// Apply CZ (Controlled-Z) via conditional phase flip.
    ///
    /// Negates amplitude when both qubits are |1⟩, leaves others unchanged. Diagonal structure
    /// eliminates matrix multiplication - just single-pass conditional negation.
    ///
    /// - Complexity: O(2^n) - one pass through state
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func applyCZ(
        control: Int,
        target: Int,
        state: QuantumState,
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

        return QuantumState(qubits: state.qubits, amplitudes: newAmplitudes)
    }

    // MARK: - Toffoli Gate Application

    /// Apply Toffoli (CCNOT) via XOR-based conditional swap
    ///
    /// Flips target qubit when both controls are |1⟩. Implemented as: if both control bits set,
    /// swap amplitudes[i] <-> amplitudes[i⊕target] using XOR to compute paired index.
    /// Single pass with simple conditional logic, no 8x8 matrix multiplication. Similar speedup
    /// to CNOT vs general multi-qubit gates.
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
        state: QuantumState,
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

        return QuantumState(qubits: state.qubits, amplitudes: newAmplitudes)
    }

    // MARK: - Fredkin Gate Application

    /// Apply Fredkin (CSWAP) via conditional bit swap
    ///
    /// Swaps target1 and target2 qubits when control is |1⟩, does nothing when |0⟩. Implemented as
    /// conditional swap without matrix multiplication, avoiding 8x8 matrix operations. For each
    /// amplitude where control bit is set and target bits differ, swaps to the index with target
    /// bits exchanged.
    ///
    /// - Complexity: O(2^n) - one pass through state
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func applyFredkin(
        control: Int,
        target1: Int,
        target2: Int,
        state: QuantumState,
    ) -> QuantumState {
        let stateSize = state.stateSpaceSize
        let controlMask = BitUtilities.bitMask(qubit: control)
        let target1Mask = BitUtilities.bitMask(qubit: target1)
        let target2Mask = BitUtilities.bitMask(qubit: target2)

        let newAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            for i in 0 ..< stateSize {
                if (i & controlMask) != 0 {
                    let bit1 = (i & target1Mask) != 0
                    let bit2 = (i & target2Mask) != 0
                    if bit1 != bit2 {
                        let swapped = i ^ target1Mask ^ target2Mask
                        buffer[swapped] = state.amplitudes[i]
                    } else {
                        buffer[i] = state.amplitudes[i]
                    }
                } else {
                    buffer[i] = state.amplitudes[i]
                }
            }
            count = stateSize
        }

        return QuantumState(qubits: state.qubits, amplitudes: newAmplitudes)
    }

    // MARK: - iSWAP Gate Application

    /// Apply iSWAP via conditional swap with i phase
    ///
    /// Swaps |01⟩ and |10⟩ states with multiplication by imaginary unit i. States |00⟩ and |11⟩
    /// remain unchanged. Implemented as conditional swap with phase, avoiding 4x4 matrix multiplication.
    /// For pairs where exactly one qubit is set, swaps amplitudes and multiplies by i.
    ///
    /// - Complexity: O(2^n) - one pass through state
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func applyISwap(
        qubit1: Int,
        qubit2: Int,
        state: QuantumState,
    ) -> QuantumState {
        let stateSize = state.stateSpaceSize
        let mask1 = BitUtilities.bitMask(qubit: qubit1)
        let mask2 = BitUtilities.bitMask(qubit: qubit2)
        let bothMask = mask1 | mask2
        let iPhase = Complex<Double>(0, 1)

        let newAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            for i in 0 ..< stateSize {
                let bits = i & bothMask
                if bits == mask1 {
                    let swapped = (i ^ mask1) | mask2
                    buffer[swapped] = state.amplitudes[i] * iPhase
                } else if bits == mask2 {
                    let swapped = (i ^ mask2) | mask1
                    buffer[swapped] = state.amplitudes[i] * iPhase
                } else {
                    buffer[i] = state.amplitudes[i]
                }
            }
            count = stateSize
        }

        return QuantumState(qubits: state.qubits, amplitudes: newAmplitudes)
    }

    // MARK: - FSWAP Gate Application

    /// Apply FSWAP via swap with conditional negation
    ///
    /// Swaps |01⟩ and |10⟩ states like regular SWAP, but negates |11⟩ state amplitude. States |00⟩
    /// remain unchanged. Implements fermionic swap operation without 4x4 matrix multiplication.
    /// Particularly useful for fermionic simulations where antisymmetry must be preserved.
    ///
    /// - Complexity: O(2^n) - one pass through state
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func applyFSwap(
        qubit1: Int,
        qubit2: Int,
        state: QuantumState,
    ) -> QuantumState {
        let stateSize = state.stateSpaceSize
        let mask1 = BitUtilities.bitMask(qubit: qubit1)
        let mask2 = BitUtilities.bitMask(qubit: qubit2)
        let bothMask = mask1 | mask2

        let newAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            for i in 0 ..< stateSize {
                let bits = i & bothMask
                if bits == mask1 {
                    let swapped = (i ^ mask1) | mask2
                    buffer[swapped] = state.amplitudes[i]
                } else if bits == mask2 {
                    let swapped = (i ^ mask2) | mask1
                    buffer[swapped] = state.amplitudes[i]
                } else if bits == bothMask {
                    buffer[i] = -state.amplitudes[i]
                } else {
                    buffer[i] = state.amplitudes[i]
                }
            }
            count = stateSize
        }

        return QuantumState(qubits: state.qubits, amplitudes: newAmplitudes)
    }

    // MARK: - Diagonal Gate Application

    /// Apply diagonal gate via phase multiplication
    ///
    /// Multiplies each amplitude by e^{i*phase[index]} where index is computed from the specified
    /// qubit bits. Diagonal gates preserve computational basis states, only modifying phases.
    /// Avoids full matrix multiplication since diagonal matrices have trivial action on basis states.
    ///
    /// - Complexity: O(2^n) - one pass through state
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func applyDiagonal(
        phases: [Double],
        qubits: [Int],
        state: QuantumState,
    ) -> QuantumState {
        let stateSize = state.stateSpaceSize

        let newAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            for i in 0 ..< stateSize {
                let phaseIndex = BitUtilities.getBits(i, qubits: qubits)
                let phase = phases[phaseIndex]
                let phaseFactor = Complex<Double>(Foundation.cos(phase), Foundation.sin(phase))
                buffer[i] = state.amplitudes[i] * phaseFactor
            }
            count = stateSize
        }

        return QuantumState(qubits: state.qubits, amplitudes: newAmplitudes)
    }

    // MARK: - Controlled Gate Application

    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func applyControlledGate(
        gate: QuantumGate,
        controls: [Int],
        targetQubits: [Int],
        state: QuantumState,
    ) -> QuantumState {
        let stateSize = state.stateSpaceSize
        var controlMask = 0
        for control in controls {
            controlMask |= BitUtilities.bitMask(qubit: control)
        }

        let gateMatrix = gate.matrix()
        let gateSize = gateMatrix.count
        let targetCount = gate.qubitsRequired

        let newAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            for i in 0 ..< stateSize {
                buffer[i] = .zero
            }

            for i in 0 ..< stateSize {
                if (i & controlMask) == controlMask {
                    var targetBits = 0
                    for (idx, qubit) in targetQubits.enumerated() where idx < targetCount {
                        if (i & BitUtilities.bitMask(qubit: qubit)) != 0 {
                            targetBits |= (1 << idx)
                        }
                    }

                    for col in 0 ..< gateSize {
                        var newIndex = i
                        for (idx, qubit) in targetQubits.enumerated() where idx < targetCount {
                            let colBit = (col >> idx) & 1
                            let mask = BitUtilities.bitMask(qubit: qubit)
                            if colBit == 1 {
                                newIndex |= mask
                            } else {
                                newIndex &= ~mask
                            }
                        }
                        buffer[i] = buffer[i] + gateMatrix[targetBits][col] * state.amplitudes[newIndex]
                    }
                } else {
                    buffer[i] = buffer[i] + state.amplitudes[i]
                }
            }
            count = stateSize
        }

        return QuantumState(qubits: state.qubits, amplitudes: newAmplitudes)
    }

    // MARK: - Multi-Qubit Gate Application

    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func applyMultiQubitGate(
        gate: QuantumGate,
        qubits: [Int],
        state: QuantumState,
    ) -> QuantumState {
        let gateMatrix = gate.matrix()
        let gateSize = gateMatrix.count
        let stateSize = state.stateSpaceSize

        let newAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            for i in 0 ..< stateSize {
                buffer[i] = .zero
            }

            for i in 0 ..< stateSize {
                var rowBits = 0
                for (idx, qubit) in qubits.enumerated() {
                    if (i & BitUtilities.bitMask(qubit: qubit)) != 0 {
                        rowBits |= (1 << idx)
                    }
                }

                for col in 0 ..< gateSize {
                    let matrixElement = gateMatrix[rowBits][col]
                    if matrixElement.real == 0, matrixElement.imaginary == 0 {
                        continue
                    }

                    var sourceIndex = i
                    for (idx, qubit) in qubits.enumerated() {
                        let colBit = (col >> idx) & 1
                        let mask = BitUtilities.bitMask(qubit: qubit)
                        if colBit == 1 {
                            sourceIndex |= mask
                        } else {
                            sourceIndex &= ~mask
                        }
                    }
                    buffer[i] = buffer[i] + matrixElement * state.amplitudes[sourceIndex]
                }
            }
            count = stateSize
        }

        return QuantumState(qubits: state.qubits, amplitudes: newAmplitudes)
    }
}

// MARK: - Convenience Extension

public extension QuantumState {
    /// Apply gate to this state (convenience method)
    ///
    /// Delegates to ``GateApplication`` with self as state parameter.
    /// Cleaner syntax for chaining transformations.
    ///
    /// **Example:**
    /// ```swift
    /// let state = QuantumState(qubits: 2)
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
    /// **Example:**
    /// ```swift
    /// let state = QuantumState(qubits: 2)
    ///     .applying(.hadamard, to: 0)
    ///     .applying(.rotationZ(.pi / 4), to: 1)
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
