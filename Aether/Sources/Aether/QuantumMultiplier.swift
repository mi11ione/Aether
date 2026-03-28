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
    /// - Complexity: O(n²) gates for schoolbook
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
        let ancilla = 4 * bits
        let cccxAncilla = 4 * bits + 1
        return buildSchoolbook(a: a, b: b, result: result, ancilla: ancilla, cccxAncilla: cccxAncilla, totalQubits: total)
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
    /// - Precondition: All registers must be disjoint
    /// - Complexity: O(n²) gates for schoolbook
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func circuit(
        _ variant: Variant,
        a: [Int], b: [Int], result: [Int],
    ) -> QuantumCircuit {
        _ = variant
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
        let ancilla = maxQubit + 1
        let cccxAncilla = maxQubit + 2
        let total = cccxAncilla + 1
        ValidationUtilities.validateUpperBound(total, max: 30, name: "total qubit count")
        return buildSchoolbook(a: a, b: b, result: result, ancilla: ancilla, cccxAncilla: cccxAncilla, totalQubits: total)
    }

    /// Total qubit count required for the multiplier circuit.
    ///
    /// Returns 4n + 2: n qubits each for a and b, 2n for the result, 1 carry ancilla,
    /// and 1 ancilla for controlled-Toffoli decomposition.
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
    /// - Complexity: O(1)
    @_effects(readonly)
    public static func qubitCount(_ variant: Variant, bits: Int) -> Int {
        _ = variant
        ValidationUtilities.validatePositiveInt(bits, name: "bits")
        return 4 * bits + 2
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
    /// - Complexity: O(n²) gates for schoolbook
    ///
    /// - SeeAlso: ``QuantumMultiplier``
    @_eagerMove
    static func multiplier(bits: Int, variant: QuantumMultiplier.Variant = .schoolbook) -> QuantumCircuit {
        QuantumMultiplier.circuit(variant, bits: bits)
    }
}
