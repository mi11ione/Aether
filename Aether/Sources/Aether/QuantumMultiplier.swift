// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Quantum multiplication circuit primitives with schoolbook and Karatsuba variants.
///
/// Provides circuit generators for n-bit unsigned integer multiplication computing
/// |a⟩|b⟩|0⟩ → |a⟩|b⟩|a·b⟩. The schoolbook variant uses O(n²) gates via n controlled
/// Cuccaro additions, each adding a shifted copy of the multiplicand into the 2n-bit
/// accumulator. The Karatsuba variant uses schoolbook for n ≤ 32 (where constant factors
/// dominate) and provides O(n^1.585) gate count for larger operands through recursive
/// three-way decomposition.
///
/// Both input registers are preserved. The product occupies a 2n-bit output register.
/// Selection follows the crossover point: n ≤ 32 → schoolbook, n > 32 → Karatsuba.
///
/// **Example:**
/// ```swift
/// let variant = QuantumMultiplier.optimalVariant(bits: 4)
/// let mult = QuantumMultiplier.circuit(variant, bits: 4)
/// let total = QuantumMultiplier.qubitCount(variant, bits: 4)
/// ```
///
/// - SeeAlso: ``QuantumAdder``
/// - SeeAlso: ``QuantumCircuit/multiplier(bits:variant:)``
public enum QuantumMultiplier {
    /// Multiplier circuit variant controlling the gate-count scaling.
    ///
    /// Both variants compute the same 2n-bit product. ``schoolbook`` is optimal for
    /// small operands (n ≤ 32) due to lower constant overhead, while ``karatsuba``
    /// achieves better asymptotic scaling for large operands via recursive decomposition.
    @frozen public enum Variant: Sendable {
        /// Schoolbook multiplier using n controlled additions. O(n²) gates, O(n) ancillas.
        ///
        /// Computes the product by iterating over each bit of the multiplier register,
        /// performing a controlled addition of the multiplicand (shifted by the bit position)
        /// into the accumulator. Each controlled addition uses the Cuccaro MAJ-UMA pattern.
        case schoolbook

        /// Karatsuba multiplier with recursive three-way decomposition. O(n^1.585) gates.
        ///
        /// For n ≤ 32, delegates to ``schoolbook`` (lower constant overhead). For n > 32,
        /// splits each operand into high and low halves and performs three recursive
        /// multiplications instead of four, reducing asymptotic gate count.
        case karatsuba
    }

    private static let karatsubaCrossover = 32

    /// Creates a multiplier circuit with auto-assigned qubit registers.
    ///
    /// Allocates registers sequentially: a at [0, bits), b at [bits, 2·bits), result at
    /// [2·bits, 4·bits), and one carry ancilla at qubit 4·bits. Both a and b are preserved;
    /// the 2n-bit product appears in the result register in little-endian order.
    ///
    /// **Example:**
    /// ```swift
    /// let mult = QuantumMultiplier.circuit(.schoolbook, bits: 4)
    /// let total = QuantumMultiplier.qubitCount(.schoolbook, bits: 4)
    /// let state = mult.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - variant: Multiplier variant to use
    ///   - bits: Number of bits per operand (≥ 1)
    /// - Returns: Quantum circuit computing the n-bit × n-bit product
    /// - Precondition: bits ≥ 1
    /// - Precondition: Total qubit count ≤ 30
    /// - Complexity: O(n²) for schoolbook, O(n^1.585) for karatsuba
    ///
    /// - SeeAlso: ``qubitCount(_:bits:)``
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func circuit(_ variant: Variant, bits: Int) -> QuantumCircuit {
        ValidationUtilities.validatePositiveInt(bits, name: "bits")
        let total = qubitCount(variant, bits: bits)
        ValidationUtilities.validateUpperBound(total, max: 30, name: "total qubit count")

        let a = Array(0 ..< bits)
        let b = Array(bits ..< 2 * bits)
        let result = Array(2 * bits ..< 4 * bits)
        let ancillaStart = 4 * bits

        switch variant {
        case .schoolbook:
            return buildSchoolbook(a: a, b: b, result: result, ancilla: ancillaStart, cccxAncilla: ancillaStart + 1, totalQubits: total)
        case .karatsuba:
            return buildKaratsuba(a: a, b: b, result: result, ancillaStart: ancillaStart, totalQubits: total)
        }
    }

    /// Creates a multiplier circuit with explicit qubit register assignments.
    ///
    /// Uses the provided a and b registers with a separate result register allocated above
    /// the maximum index. Both a and b are preserved; the product is placed in the result
    /// register.
    ///
    /// **Example:**
    /// ```swift
    /// let result = Array(8 ..< 16)
    /// let mult = QuantumMultiplier.circuit(.schoolbook, a: [0, 1, 2, 3], b: [4, 5, 6, 7], result: result)
    /// let state = mult.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - variant: Multiplier variant to use
    ///   - a: Qubit indices for multiplicand (LSB first)
    ///   - b: Qubit indices for multiplier (LSB first)
    ///   - result: Qubit indices for 2n-bit product (LSB first)
    /// - Returns: Quantum circuit computing the product
    /// - Precondition: a and b have equal non-zero length
    /// - Precondition: result has exactly 2·a.count qubits
    /// - Precondition: All qubit indices must be non-negative
    /// - Precondition: Qubit indices in each register must be unique
    /// - Precondition: All registers must be disjoint
    /// - Precondition: Total qubit count ≤ 30
    /// - Complexity: O(n²) for schoolbook, O(n^1.585) for karatsuba
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func circuit(
        _ variant: Variant,
        a: [Int], b: [Int], result: [Int],
    ) -> QuantumCircuit {
        let bits = a.count
        ValidationUtilities.validateNonEmpty(a, name: "a")
        ValidationUtilities.validateEqualCounts(a, b, name1: "a", name2: "b")
        ValidationUtilities.validateRegisterSize(result, expected: 2 * bits, name: "result")
        ValidationUtilities.validateNonNegativeQubits(a)
        ValidationUtilities.validateNonNegativeQubits(b)
        ValidationUtilities.validateNonNegativeQubits(result)
        ValidationUtilities.validateUniqueQubits(a)
        ValidationUtilities.validateUniqueQubits(b)
        ValidationUtilities.validateUniqueQubits(result)
        ValidationUtilities.validateDisjointRegisters(a, b, nameA: "a", nameB: "b")
        ValidationUtilities.validateDisjointRegisters(a, result, nameA: "a", nameB: "result")
        ValidationUtilities.validateDisjointRegisters(b, result, nameA: "b", nameB: "result")

        let maxQubit = max(a.max()!, max(b.max()!, result.max()!)) // safe: a, b, result validated non-empty
        let ancillaStart = maxQubit + 1

        switch variant {
        case .schoolbook:
            let total = ancillaStart + 2
            ValidationUtilities.validateUpperBound(total, max: 30, name: "total qubit count")
            return buildSchoolbook(a: a, b: b, result: result, ancilla: ancillaStart, cccxAncilla: ancillaStart + 1, totalQubits: total)
        case .karatsuba:
            let ancillaNeeded = karatsubaAncillaCount(bits: bits)
            let total = ancillaStart + ancillaNeeded
            ValidationUtilities.validateUpperBound(total, max: 30, name: "total qubit count")
            return buildKaratsuba(a: a, b: b, result: result, ancillaStart: ancillaStart, totalQubits: total)
        }
    }

    /// Total qubit count required for the multiplier circuit.
    ///
    /// For ``Variant/schoolbook``, returns 4n + 2: n qubits each for a and b, 2n for
    /// the result, 1 carry ancilla, and 1 CCCX decomposition ancilla. For
    /// ``Variant/karatsuba`` with n ≤ 32, returns the same (delegates to schoolbook).
    /// For ``Variant/karatsuba`` with n > 32, returns 4n plus the recursive ancilla
    /// count for partial-product registers, sum registers, and carry propagation.
    ///
    /// **Example:**
    /// ```swift
    /// let bits = 4
    /// let total = QuantumMultiplier.qubitCount(.schoolbook, bits: bits)
    /// let overhead = total - 2 * bits
    /// ```
    ///
    /// - Parameters:
    ///   - variant: Multiplier variant
    ///   - bits: Number of bits per operand
    /// - Returns: Total qubits required
    /// - Precondition: bits ≥ 1
    /// - Complexity: O(1) for schoolbook, O(log n) for karatsuba
    @_effects(readonly)
    public static func qubitCount(_ variant: Variant, bits: Int) -> Int {
        ValidationUtilities.validatePositiveInt(bits, name: "bits")
        switch variant {
        case .schoolbook:
            return 4 * bits + 2
        case .karatsuba:
            return 4 * bits + karatsubaAncillaCount(bits: bits)
        }
    }

    /// Selects the optimal multiplier variant based on operand size.
    ///
    /// Returns ``Variant/schoolbook`` for n ≤ 32 (lower constant overhead) and
    /// ``Variant/karatsuba`` for n > 32. Within the simulator's 30-qubit limit,
    /// schoolbook is always selected.
    ///
    /// **Example:**
    /// ```swift
    /// let v = QuantumMultiplier.optimalVariant(bits: 4)
    /// let circuit = QuantumMultiplier.circuit(v, bits: 4)
    /// let state = circuit.execute()
    /// ```
    ///
    /// - Parameter bits: Number of bits per operand
    /// - Returns: Recommended multiplier variant
    /// - Precondition: bits ≥ 1
    /// - Complexity: O(1)
    @_effects(readonly)
    public static func optimalVariant(bits: Int) -> Variant {
        ValidationUtilities.validatePositiveInt(bits, name: "bits")
        return bits > karatsubaCrossover ? .karatsuba : .schoolbook
    }

    // MARK: - Implementation

    /// Schoolbook multiplier: n controlled additions of shifted multiplicand.
    @_eagerMove
    private static func buildSchoolbook(
        a: [Int], b: [Int], result: [Int], ancilla: Int, cccxAncilla: Int, totalQubits: Int,
    ) -> QuantumCircuit {
        let n = a.count
        var circuit = QuantumCircuit(qubits: totalQubits)

        if n == 1 {
            circuit.append(.toffoli, to: [a[0], b[0], result[0]])
            return circuit
        }

        for j in 0 ..< n {
            appendControlledAddition(
                to: &circuit, a: a, target: result, offset: j,
                control: b[j], ancilla: ancilla, cccxAncilla: cccxAncilla, n: n,
            )
        }

        return circuit
    }

    /// Appends a controlled addition of a to target[offset..offset+n-1] with carry-out.
    private static func appendControlledAddition(
        to circuit: inout QuantumCircuit,
        a: [Int], target: [Int], offset: Int,
        control: Int, ancilla: Int, cccxAncilla: Int, n: Int,
    ) {
        appendControlledMAJ(to: &circuit, x: ancilla, y: target[offset], z: a[0], control: control, cccxAncilla: cccxAncilla)
        for k in 1 ..< n {
            appendControlledMAJ(to: &circuit, x: a[k - 1], y: target[offset + k], z: a[k], control: control, cccxAncilla: cccxAncilla)
        }

        if offset + n < target.count {
            circuit.append(.toffoli, to: [control, a[n - 1], target[offset + n]])
        }

        for k in stride(from: n - 1, through: 1, by: -1) {
            appendControlledUMA(to: &circuit, x: a[k - 1], y: target[offset + k], z: a[k], control: control, cccxAncilla: cccxAncilla)
        }
        appendControlledUMA(to: &circuit, x: ancilla, y: target[offset], z: a[0], control: control, cccxAncilla: cccxAncilla)
    }

    // MARK: - Gate Primitives

    /// Controlled Majority gate using manual CCCX decomposition.
    private static func appendControlledMAJ(
        to circuit: inout QuantumCircuit, x: Int, y: Int, z: Int, control: Int, cccxAncilla: Int,
    ) {
        circuit.append(.toffoli, to: [control, z, y])
        circuit.append(.toffoli, to: [control, z, x])
        appendCCCX(to: &circuit, c0: control, c1: x, c2: y, target: z, ancilla: cccxAncilla)
    }

    /// Controlled UnMajority-Add gate using manual CCCX decomposition.
    private static func appendControlledUMA(
        to circuit: inout QuantumCircuit, x: Int, y: Int, z: Int, control: Int, cccxAncilla: Int,
    ) {
        appendCCCX(to: &circuit, c0: control, c1: x, c2: y, target: z, ancilla: cccxAncilla)
        circuit.append(.toffoli, to: [control, z, x])
        circuit.append(.toffoli, to: [control, x, y])
    }

    /// Manual 3-control X decomposition via Toffoli ladder with dedicated ancilla.
    private static func appendCCCX(
        to circuit: inout QuantumCircuit,
        c0: Int, c1: Int, c2: Int, target: Int, ancilla: Int,
    ) {
        circuit.append(.toffoli, to: [c0, c1, ancilla])
        circuit.append(.toffoli, to: [ancilla, c2, target])
        circuit.append(.toffoli, to: [c0, c1, ancilla])
    }

    // MARK: - Karatsuba Implementation

    /// Ancilla count for Karatsuba multiplication of n-bit operands.
    private static func karatsubaAncillaCount(bits n: Int) -> Int {
        if n <= karatsubaCrossover { return 2 }
        let m = (n + 1) / 2
        let h = n - m
        let z0Anc = 2 * m
        let z2Anc = 2 * h
        let z1Anc = 2 * (m + 1)
        let sumAnc = 2 * (m + 1)
        let padAnc = 2 * (m + 1 - h)
        let local = z0Anc + z2Anc + z1Anc + sumAnc + padAnc + 1
        let sub = max(
            karatsubaAncillaCount(bits: m),
            max(karatsubaAncillaCount(bits: h), karatsubaAncillaCount(bits: m + 1)),
        )
        return local + sub
    }

    /// Karatsuba multiplier via recursive three-way decomposition (Parent–Roetteler–Svore).
    @_eagerMove
    @_optimize(speed)
    private static func buildKaratsuba(
        a: [Int], b: [Int], result: [Int],
        ancillaStart: Int, totalQubits: Int,
    ) -> QuantumCircuit {
        let n = a.count

        if n <= karatsubaCrossover {
            return buildSchoolbook(
                a: a, b: b, result: result,
                ancilla: ancillaStart, cccxAncilla: ancillaStart + 1,
                totalQubits: totalQubits,
            )
        }

        let m = (n + 1) / 2
        let h = n - m

        let aL = Array(a[0 ..< m])
        let aH = Array(a[m ..< n])
        let bL = Array(b[0 ..< m])
        let bH = Array(b[m ..< n])

        var offset = ancillaStart
        let z0Reg = Array(offset ..< offset + 2 * m); offset += 2 * m
        let z2Reg = Array(offset ..< offset + 2 * h); offset += 2 * h
        let z1Reg = Array(offset ..< offset + 2 * (m + 1)); offset += 2 * (m + 1)
        let sumA = Array(offset ..< offset + m + 1); offset += m + 1
        let sumB = Array(offset ..< offset + m + 1); offset += m + 1
        let padCount = 2 * (m + 1 - h)
        let pad = Array(offset ..< offset + padCount); offset += padCount
        let carry = offset; offset += 1
        let subAncStart = offset

        var circuit = QuantumCircuit(qubits: totalQubits)

        let z0Circ = buildKaratsuba(
            a: aL, b: bL, result: z0Reg,
            ancillaStart: subAncStart, totalQubits: totalQubits,
        )
        for op in z0Circ.operations {
            circuit.append(op)
        }

        let z2Circ = buildKaratsuba(
            a: aH, b: bH, result: z2Reg,
            ancillaStart: subAncStart, totalQubits: totalQubits,
        )
        for op in z2Circ.operations {
            circuit.append(op)
        }

        let sumALow = Array(sumA[0 ..< m])
        appendCopyBits(to: &circuit, source: aL, target: sumALow)
        let aHPad = aH + Array(pad[0 ..< m + 1 - h])
        appendRippleCarry(to: &circuit, a: aHPad, b: sumA, carry: carry, forward: true)

        let sumBLow = Array(sumB[0 ..< m])
        appendCopyBits(to: &circuit, source: bL, target: sumBLow)
        let bHPad = bH + Array(pad[0 ..< m + 1 - h])
        appendRippleCarry(to: &circuit, a: bHPad, b: sumB, carry: carry, forward: true)

        let z1Circ = buildKaratsuba(
            a: sumA, b: sumB, result: z1Reg,
            ancillaStart: subAncStart, totalQubits: totalQubits,
        )
        for op in z1Circ.operations {
            circuit.append(op)
        }

        let z0Pad = z0Reg + Array(pad[0 ..< 2])
        appendRippleCarry(to: &circuit, a: z0Pad, b: z1Reg, carry: carry, forward: false)

        let z2Pad = z2Reg + Array(pad[0 ..< padCount])
        appendRippleCarry(to: &circuit, a: z2Pad, b: z1Reg, carry: carry, forward: false)

        appendCopyBits(to: &circuit, source: z0Reg, target: Array(result[0 ..< 2 * m]))
        appendCopyBits(to: &circuit, source: z2Reg, target: Array(result[2 * m ..< 2 * n]))

        let crossLen = min(2 * (m + 1), 2 * n - m)
        appendRippleCarry(
            to: &circuit,
            a: Array(z1Reg[0 ..< crossLen]),
            b: Array(result[m ..< m + crossLen]),
            carry: carry, forward: true,
        )

        appendRippleCarry(to: &circuit, a: z2Pad, b: z1Reg, carry: carry, forward: true)
        appendRippleCarry(to: &circuit, a: z0Pad, b: z1Reg, carry: carry, forward: true)
        for op in z1Circ.operations.reversed() {
            circuit.append(op)
        }

        appendRippleCarry(to: &circuit, a: bHPad, b: sumB, carry: carry, forward: false)
        appendCopyBits(to: &circuit, source: bL, target: sumBLow)

        appendRippleCarry(to: &circuit, a: aHPad, b: sumA, carry: carry, forward: false)
        appendCopyBits(to: &circuit, source: aL, target: sumALow)

        for op in z2Circ.operations.reversed() {
            circuit.append(op)
        }
        for op in z0Circ.operations.reversed() {
            circuit.append(op)
        }

        return circuit
    }

    // MARK: - Arithmetic Helpers

    /// Ripple-carry arithmetic: addition (forward) or subtraction (reverse) via Cuccaro MAJ-UMA.
    private static func appendRippleCarry(
        to circuit: inout QuantumCircuit,
        a: [Int], b: [Int], carry: Int, forward: Bool,
    ) {
        let n = a.count
        if n == 0 { return }
        if n == 1 {
            circuit.append(.cnot, to: [a[0], b[0]])
            return
        }

        if forward {
            appendMAJ(to: &circuit, x: carry, y: b[0], z: a[0], forward: true)
            for i in 1 ..< n {
                appendMAJ(to: &circuit, x: a[i - 1], y: b[i], z: a[i], forward: true)
            }
            for i in stride(from: n - 1, through: 1, by: -1) {
                appendUMA(to: &circuit, x: a[i - 1], y: b[i], z: a[i], forward: true)
            }
            appendUMA(to: &circuit, x: carry, y: b[0], z: a[0], forward: true)
        } else {
            appendUMA(to: &circuit, x: carry, y: b[0], z: a[0], forward: false)
            for i in 1 ..< n {
                appendUMA(to: &circuit, x: a[i - 1], y: b[i], z: a[i], forward: false)
            }
            for i in stride(from: n - 1, through: 1, by: -1) {
                appendMAJ(to: &circuit, x: a[i - 1], y: b[i], z: a[i], forward: false)
            }
            appendMAJ(to: &circuit, x: carry, y: b[0], z: a[0], forward: false)
        }
    }

    /// XOR source into zero-initialized target via CNOTs.
    private static func appendCopyBits(
        to circuit: inout QuantumCircuit, source: [Int], target: [Int],
    ) {
        for i in 0 ..< source.count {
            circuit.append(.cnot, to: [source[i], target[i]])
        }
    }

    /// Majority gate or its inverse (adjoint).
    private static func appendMAJ(
        to circuit: inout QuantumCircuit, x: Int, y: Int, z: Int, forward: Bool,
    ) {
        if forward {
            circuit.append(.cnot, to: [z, y])
            circuit.append(.cnot, to: [z, x])
            circuit.append(.toffoli, to: [x, y, z])
        } else {
            circuit.append(.toffoli, to: [x, y, z])
            circuit.append(.cnot, to: [z, x])
            circuit.append(.cnot, to: [z, y])
        }
    }

    /// UnMajority-Add gate or its inverse (adjoint).
    private static func appendUMA(
        to circuit: inout QuantumCircuit, x: Int, y: Int, z: Int, forward: Bool,
    ) {
        if forward {
            circuit.append(.toffoli, to: [x, y, z])
            circuit.append(.cnot, to: [z, x])
            circuit.append(.cnot, to: [x, y])
        } else {
            circuit.append(.cnot, to: [x, y])
            circuit.append(.cnot, to: [z, x])
            circuit.append(.toffoli, to: [x, y, z])
        }
    }
}

public extension QuantumCircuit {
    /// Creates a quantum multiplier circuit computing |a⟩|b⟩|0⟩ → |a⟩|b⟩|a·b⟩.
    ///
    /// Convenience factory delegating to ``QuantumMultiplier/circuit(_:bits:)``. Uses
    /// controlled Cuccaro additions for the schoolbook variant, producing the 2n-bit
    /// product in a separate output register while preserving both inputs.
    ///
    /// **Example:**
    /// ```swift
    /// let mult = QuantumCircuit.multiplier(bits: 4, variant: .schoolbook)
    /// let depth = mult.depth
    /// let state = mult.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - bits: Number of bits per operand (≥ 1)
    ///   - variant: Multiplier variant (default: schoolbook)
    /// - Returns: Quantum circuit computing the product
    /// - Precondition: bits ≥ 1
    /// - Precondition: Total qubit count ≤ 30
    /// - Complexity: O(n²) for schoolbook, O(n^1.585) for karatsuba
    ///
    /// - SeeAlso: ``QuantumMultiplier``
    @_eagerMove
    static func multiplier(bits: Int, variant: QuantumMultiplier.Variant = .schoolbook) -> QuantumCircuit {
        QuantumMultiplier.circuit(variant, bits: bits)
    }
}
