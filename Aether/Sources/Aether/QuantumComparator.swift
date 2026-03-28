// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Constant-ancilla quantum comparator computing |a⟩|b⟩|0⟩ → |a⟩|b⟩|a<b⟩.
///
/// Implements comparison by computing the borrow bit of the subtraction a − b via
/// a ripple-borrow subtractor built from the Cuccaro MAJ gate pattern. The carry-out
/// of a + ~b + 1 (two's complement subtraction) is 0 when a < b and 1 when a ≥ b;
/// the result qubit receives the negation of this carry-out. Both input registers
/// are fully preserved through forward MAJ cascade, sign-bit extraction, and
/// reversed MAJ uncomputation.
///
/// **Example:**
/// ```swift
/// let cmp = QuantumComparator.circuit(bits: 4)
/// let total = QuantumComparator.qubitCount(bits: 4)
/// let state = cmp.execute()
/// ```
///
/// - SeeAlso: ``QuantumAdder``
/// - SeeAlso: ``QuantumCircuit/comparator(bits:)``
public enum QuantumComparator {
    /// Creates a comparator circuit with auto-assigned qubit registers.
    ///
    /// Allocates registers sequentially: a at [0, bits), b at [bits, 2·bits), result at
    /// qubit 2·bits, and carry-in ancilla at qubit 2·bits+1. After execution, the result
    /// qubit holds 1 if a < b (interpreting registers as unsigned integers in little-endian)
    /// and 0 otherwise. Both a and b are preserved.
    ///
    /// **Example:**
    /// ```swift
    /// let cmp = QuantumComparator.circuit(bits: 4)
    /// let total = QuantumComparator.qubitCount(bits: 4)
    /// let state = cmp.execute()
    /// ```
    ///
    /// - Parameter bits: Number of bits per operand (≥ 1)
    /// - Returns: Circuit computing |a⟩|b⟩|0⟩ → |a⟩|b⟩|a<b⟩
    /// - Precondition: bits ≥ 1
    /// - Precondition: 2·bits + 2 ≤ 30
    /// - Complexity: O(n) depth and O(n) gates
    ///
    /// - SeeAlso: ``qubitCount(bits:)``
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func circuit(bits: Int) -> QuantumCircuit {
        ValidationUtilities.validatePositiveInt(bits, name: "bits")
        let total = qubitCount(bits: bits)
        ValidationUtilities.validateUpperBound(total, max: 30, name: "total qubit count")

        let a = Array(0 ..< bits)
        let b = Array(bits ..< 2 * bits)
        let result = 2 * bits
        let ancilla = 2 * bits + 1
        return buildComparator(a: a, b: b, result: result, ancilla: ancilla, totalQubits: total)
    }

    /// Creates a comparator circuit with explicit qubit register assignments.
    ///
    /// Applies the comparison circuit to specified qubit indices, with the ancilla
    /// auto-allocated above the maximum used index. Both a and b are preserved;
    /// the result qubit receives 1 if the unsigned value of a is strictly less than b.
    ///
    /// **Example:**
    /// ```swift
    /// let a = [0, 1, 2, 3]
    /// let cmp = QuantumComparator.circuit(a: a, b: [4, 5, 6, 7], result: 8)
    /// let state = cmp.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - a: Qubit indices for first operand (LSB first)
    ///   - b: Qubit indices for second operand (LSB first)
    ///   - result: Qubit index for the comparison result
    /// - Returns: Circuit computing |a⟩|b⟩|0⟩ → |a⟩|b⟩|a<b⟩
    /// - Precondition: a and b have equal non-zero length
    /// - Precondition: Registers must not overlap and result must not be in any register
    /// - Precondition: All qubit indices ≥ 0
    /// - Complexity: O(n) depth and O(n) gates
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func circuit(a: [Int], b: [Int], result: Int) -> QuantumCircuit {
        ValidationUtilities.validateNonEmpty(a, name: "a")
        ValidationUtilities.validateEqualCounts(a, b, name1: "a", name2: "b")
        ValidationUtilities.validateNonNegativeQubits(a)
        ValidationUtilities.validateNonNegativeQubits(b)
        ValidationUtilities.validateUniqueQubits(a)
        ValidationUtilities.validateUniqueQubits(b)
        ValidationUtilities.validateDisjointRegisters(a, b, nameA: "a", nameB: "b")
        ValidationUtilities.validateNonNegativeInt(result, name: "result")
        ValidationUtilities.validateQubitNotInRegisters(result, registers: [a, b], qubitName: "result")

        let maxQubit = max(a.max()!, max(b.max()!, result)) // safe: a, b validated non-empty
        let ancilla = maxQubit + 1
        let total = ancilla + 1
        ValidationUtilities.validateUpperBound(total, max: 30, name: "total qubit count")
        return buildComparator(a: a, b: b, result: result, ancilla: ancilla, totalQubits: total)
    }

    /// Total qubit count required for the comparator circuit.
    ///
    /// Returns 2n + 2: n qubits each for a and b registers, 1 result qubit, and 1 carry-in
    /// ancilla qubit.
    ///
    /// **Example:**
    /// ```swift
    /// let bits = 4
    /// let total = QuantumComparator.qubitCount(bits: bits)
    /// let ancillas = total - 2 * bits
    /// ```
    ///
    /// - Parameter bits: Number of bits per operand
    /// - Returns: Total qubits required
    /// - Precondition: bits ≥ 1
    /// - Complexity: O(1)
    @_effects(readonly)
    public static func qubitCount(bits: Int) -> Int {
        ValidationUtilities.validatePositiveInt(bits, name: "bits")
        return 2 * bits + 2
    }

    // MARK: - Implementation

    /// Builds comparator: |a⟩|b⟩|0⟩ → |a⟩|b⟩|a<b⟩ via subtraction sign bit.
    private static func buildComparator(
        a: [Int], b: [Int], result: Int, ancilla: Int, totalQubits: Int,
    ) -> QuantumCircuit {
        let n = a.count
        var circuit = QuantumCircuit(qubits: totalQubits)

        if n == 1 {
            circuit.append(.pauliX, to: a[0])
            circuit.append(.toffoli, to: [a[0], b[0], result])
            circuit.append(.pauliX, to: a[0])
            return circuit
        }

        for i in 0 ..< n {
            circuit.append(.pauliX, to: b[i])
        }
        circuit.append(.pauliX, to: ancilla)

        appendMAJ(to: &circuit, x: ancilla, y: b[0], z: a[0])
        for i in 1 ..< n {
            appendMAJ(to: &circuit, x: a[i - 1], y: b[i], z: a[i])
        }

        circuit.append(.cnot, to: [a[n - 1], result])

        for i in stride(from: n - 1, through: 1, by: -1) {
            appendReversedMAJ(to: &circuit, x: a[i - 1], y: b[i], z: a[i])
        }
        appendReversedMAJ(to: &circuit, x: ancilla, y: b[0], z: a[0])

        circuit.append(.pauliX, to: ancilla)
        for i in 0 ..< n {
            circuit.append(.pauliX, to: b[i])
        }

        circuit.append(.pauliX, to: result)

        return circuit
    }

    // MARK: - Gate Primitives

    /// Majority gate: MAJ(x, y, z) = CNOT(z,y); CNOT(z,x); Toffoli(x,y,z).
    private static func appendMAJ(to circuit: inout QuantumCircuit, x: Int, y: Int, z: Int) {
        circuit.append(.cnot, to: [z, y])
        circuit.append(.cnot, to: [z, x])
        circuit.append(.toffoli, to: [x, y, z])
    }

    /// Reversed Majority gate for uncomputation (no sum).
    private static func appendReversedMAJ(to circuit: inout QuantumCircuit, x: Int, y: Int, z: Int) {
        circuit.append(.toffoli, to: [x, y, z])
        circuit.append(.cnot, to: [z, x])
        circuit.append(.cnot, to: [z, y])
    }
}

public extension QuantumCircuit {
    /// Creates a quantum comparator circuit computing |a⟩|b⟩|0⟩ → |a⟩|b⟩|a<b⟩.
    ///
    /// Convenience factory delegating to ``QuantumComparator/circuit(bits:)``. Uses
    /// a ripple-borrow subtractor to extract the sign bit of a − b with O(1) ancillas
    /// while preserving both input registers.
    ///
    /// **Example:**
    /// ```swift
    /// let cmp = QuantumCircuit.comparator(bits: 4)
    /// let depth = cmp.depth
    /// let state = cmp.execute()
    /// ```
    ///
    /// - Parameter bits: Number of bits per operand (≥ 1)
    /// - Returns: Circuit computing the comparison
    /// - Precondition: bits ≥ 1
    /// - Precondition: 2·bits + 2 ≤ 30
    /// - Complexity: O(n) depth and O(n) gates
    ///
    /// - SeeAlso: ``QuantumComparator``
    @_eagerMove
    static func comparator(bits: Int) -> QuantumCircuit {
        QuantumComparator.circuit(bits: bits)
    }
}
