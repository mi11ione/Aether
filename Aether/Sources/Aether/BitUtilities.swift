// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Shared bit manipulation utilities for quantum computing
///
/// Centralizes common bit operations used throughout the quantum simulator for
/// qubit indexing, basis state manipulation, and gate application. All operations
/// work on raw integers representing basis state indices in little-endian qubit
/// ordering (qubit 0 is LSB).
///
/// **Qubit Ordering Convention**:
/// - Little-endian: qubit 0 is least significant bit
/// - Basis state |q₂q₁q₀⟩ maps to integer q₂×4 + q₁×2 + q₀×1
/// - Example: |101⟩ = 5 (binary 0b101)
///
/// **Performance**: All operations are O(1) bitwise operations compiled to
/// single CPU instructions (shift, AND, OR, XOR).
@frozen
public enum BitUtilities {
    /// Extract bit value at specific qubit position
    ///
    /// Retrieves the value (0 or 1) of a single qubit from a basis state index.
    /// Uses right shift and mask to isolate the target bit.
    ///
    /// **Algorithm**: `(index >> qubit) & 1`
    /// - Right shift moves target bit to LSB position
    /// - AND with 1 isolates the bit value
    ///
    /// **Example**:
    /// ```swift
    /// let state = 5  // |101⟩ in binary
    /// BitUtilities.getBit(state, qubit: 0)  // → 1 (LSB)
    /// BitUtilities.getBit(state, qubit: 1)  // → 0
    /// BitUtilities.getBit(state, qubit: 2)  // → 1 (MSB)
    /// ```
    ///
    /// **Use Cases**:
    /// - Determining qubit measurement outcomes from basis states
    /// - Computing marginal probabilities (sum over bit=0 vs bit=1)
    /// - Gate application logic (conditional operations based on control qubits)
    ///
    /// - Parameters:
    ///   - index: Basis state index (0 to 2^n-1)
    ///   - qubit: Qubit position (0 = LSB, n-1 = MSB)
    /// - Returns: Bit value (0 or 1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func getBit(_ index: Int, qubit: Int) -> Int {
        (index >> qubit) & 1
    }

    /// Set bit at specific qubit position to given value
    ///
    /// Modifies a basis state index by setting the specified qubit to 0 or 1.
    /// Uses bitwise operations to efficiently update single bits.
    ///
    /// **Algorithms**:
    /// - Set to 0 (clear): `index & ~(1 << qubit)`
    ///   * Creates mask with 0 at target position, 1s elsewhere
    ///   * AND operation clears the bit
    /// - Set to 1: `index | (1 << qubit)`
    ///   * Creates mask with 1 at target position
    ///   * OR operation sets the bit
    ///
    /// **Example**:
    /// ```swift
    /// let state = 5  // |101⟩
    /// BitUtilities.setBit(state, qubit: 1, value: 1)  // → 7 (|111⟩)
    /// BitUtilities.setBit(state, qubit: 0, value: 0)  // → 4 (|100⟩)
    /// ```
    ///
    /// **Use Cases**:
    /// - Constructing specific basis states
    /// - Measurement collapse (setting qubits to measured values)
    /// - Controlled gate operations (setting target based on control)
    ///
    /// - Parameters:
    ///   - index: Original basis state index
    ///   - qubit: Qubit position to modify
    ///   - value: New bit value (0 or 1)
    /// - Returns: Modified basis state index
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func setBit(_ index: Int, qubit: Int, value: Int) -> Int {
        if value == 0 {
            index & ~(1 << qubit) // Clear bit
        } else {
            index | (1 << qubit) // Set bit
        }
    }

    /// Flip bit at specific qubit position
    ///
    /// Toggles the specified qubit: 0→1 or 1→0. Essential for implementing
    /// X gates and CNOT operations which flip target qubits.
    ///
    /// **Algorithm**: `index ^ (1 << qubit)`
    /// - Creates mask with 1 at target position
    /// - XOR flips the bit (0⊕1=1, 1⊕1=0)
    ///
    /// **Example**:
    /// ```swift
    /// let state = 5  // |101⟩
    /// BitUtilities.flipBit(state, qubit: 1)  // → 7 (|111⟩)
    /// BitUtilities.flipBit(state, qubit: 0)  // → 4 (|100⟩)
    /// BitUtilities.flipBit(state, qubit: 2)  // → 1 (|001⟩)
    /// ```
    ///
    /// **Use Cases**:
    /// - X gate: flips target qubit unconditionally
    /// - CNOT gate: flips target if control=1
    /// - Toffoli gate: flips target if both controls=1
    ///
    /// - Parameters:
    ///   - index: Original basis state index
    ///   - qubit: Qubit position to flip
    /// - Returns: Modified basis state index with flipped bit
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func flipBit(_ index: Int, qubit: Int) -> Int {
        index ^ (1 << qubit)
    }

    /// Clear bit at specific qubit position (set to 0)
    ///
    /// Convenience method equivalent to `setBit(index, qubit, value: 0)`.
    /// Provided for clarity when intent is specifically to clear a bit.
    ///
    /// - Parameters:
    ///   - index: Original basis state index
    ///   - qubit: Qubit position to clear
    /// - Returns: Modified basis state index with bit cleared
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func clearBit(_ index: Int, qubit: Int) -> Int {
        index & ~(1 << qubit)
    }

    /// Create bit mask for specific qubit position
    ///
    /// Generates integer with single bit set at qubit position, all others zero.
    /// Useful for efficient batch operations and gate decompositions.
    ///
    /// **Algorithm**: `1 << qubit`
    ///
    /// **Example**:
    /// ```swift
    /// BitUtilities.bitMask(qubit: 0)  // → 1 (0b001)
    /// BitUtilities.bitMask(qubit: 1)  // → 2 (0b010)
    /// BitUtilities.bitMask(qubit: 2)  // → 4 (0b100)
    /// ```
    ///
    /// **Use Cases**:
    /// - Gate application: precompute masks for control/target qubits
    /// - Batch operations: apply same mask to multiple indices
    /// - Bit field manipulation: combine multiple masks with OR
    ///
    /// - Parameter qubit: Qubit position (0 = LSB)
    /// - Returns: Integer with single bit set (2^qubit)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func bitMask(qubit: Int) -> Int {
        1 << qubit
    }

    /// Extract multiple bits as integer value
    ///
    /// Retrieves values of multiple qubits and combines them into single integer.
    /// Useful for multi-qubit gate conditions and measurement outcomes.
    ///
    /// **Algorithm**: For each qubit, extract bit and combine with appropriate weight
    ///
    /// **Example**:
    /// ```swift
    /// let state = 7  // |111⟩
    /// BitUtilities.getBits(state, qubits: [0, 1])  // → 3 (both bits are 1)
    /// BitUtilities.getBits(state, qubits: [1, 2])  // → 3
    /// ```
    ///
    /// **Use Cases**:
    /// - Toffoli gate: check if both control qubits are 1
    /// - Multi-qubit measurement: extract outcomes for subset of qubits
    /// - Conditional operations: execute based on multiple qubit values
    ///
    /// - Parameters:
    ///   - index: Basis state index
    ///   - qubits: Array of qubit positions to extract
    /// - Returns: Integer formed from extracted bits (qubits[0] = LSB)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func getBits(_ index: Int, qubits: [Int]) -> Int {
        var result = 0
        for (position, qubit) in qubits.enumerated() {
            let bit = getBit(index, qubit: qubit)
            result |= (bit << position)
        }
        return result
    }
}
