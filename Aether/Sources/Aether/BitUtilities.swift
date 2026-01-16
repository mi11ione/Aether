// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Bit manipulation utilities for quantum basis state indexing
///
/// Centralizes common bit operations used throughout the quantum simulator for qubit indexing,
/// basis state manipulation, and gate application. All operations work on raw integers representing
/// basis state indices in little-endian qubit ordering (qubit 0 is LSB). Essential for gate
/// application logic, measurement outcome extraction, and control qubit evaluation.
///
/// **Qubit Ordering Convention**: Little-endian where qubit 0 is least significant bit.
/// Basis state |q₂q₁q₀⟩ maps to integer q₂x4 + q₁x2 + q₀x1. Example: |101⟩ = 5 (binary 0b101).
///
/// **Performance**: All operations are O(1) bitwise operations compiled to single CPU instructions
/// (shift, AND, OR, XOR) for zero-cost abstractions.
///
/// **Example:**
/// ```swift
/// let state = 5  // |101⟩ in binary
/// let bit0 = BitUtilities.getBit(state, qubit: 0)  // 1
/// let flipped = BitUtilities.flipBit(state, qubit: 1)  // 7 (|111⟩)
/// let mask = BitUtilities.bitMask(qubit: 2)  // 4 (0b100)
/// ```
public enum BitUtilities {
    /// Extract bit value at specific qubit position
    ///
    /// Retrieves the value (0 or 1) of a single qubit from a basis state index using right shift
    /// and mask operations. Algorithm: `(index >> qubit) & 1` moves target bit to LSB position
    /// and isolates it. Used for measurement outcome extraction, marginal probability computation,
    /// and control qubit evaluation in gate application.
    ///
    /// **Example:**
    /// ```swift
    /// let state = 5  // |101⟩ in binary
    /// BitUtilities.getBit(state, qubit: 0)  // 1 (LSB)
    /// BitUtilities.getBit(state, qubit: 1)  // 0
    /// BitUtilities.getBit(state, qubit: 2)  // 1 (MSB)
    /// ```
    ///
    /// - Parameters:
    ///   - index: Basis state index (0 to 2^n-1)
    ///   - qubit: Qubit position (0 = LSB, n-1 = MSB)
    /// - Returns: Bit value (0 or 1)
    /// - Complexity: O(1) - single CPU instruction (shift + AND)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func getBit(_ index: Int, qubit: Int) -> Int {
        (index >> qubit) & 1
    }

    /// Set bit at specific qubit position to given value
    ///
    /// Modifies a basis state index by setting the specified qubit to 0 or 1 using bitwise operations.
    /// Algorithm: `(index & ~(1 << qubit)) | (value << qubit)` clears target bit then sets it to value.
    /// Used for basis state construction, measurement collapse, and controlled gate operations.
    ///
    /// **Example:**
    /// ```swift
    /// let state = 5  // |101⟩
    /// BitUtilities.setBit(state, qubit: 1, value: 1)  // 7 (|111⟩)
    /// BitUtilities.setBit(state, qubit: 0, value: 0)  // 4 (|100⟩)
    /// ```
    ///
    /// - Parameters:
    ///   - index: Original basis state index
    ///   - qubit: Qubit position to modify
    ///   - value: New bit value (0 or 1)
    /// - Returns: Modified basis state index
    /// - Complexity: O(1) - constant bitwise operations (AND, OR, shift)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func setBit(_ index: Int, qubit: Int, value: Int) -> Int {
        (index & ~(1 << qubit)) | (value << qubit)
    }

    /// Flip bit at specific qubit position
    ///
    /// Toggles the specified qubit: 0->1 or 1->0 using XOR operation. Essential for implementing
    /// X gates and CNOT operations which flip target qubits. Algorithm: `index ^ (1 << qubit)`
    /// creates mask with 1 at target position and XOR flips the bit (0⊕1=1, 1⊕1=0).
    ///
    /// **Example:**
    /// ```swift
    /// let state = 5  // |101⟩
    /// BitUtilities.flipBit(state, qubit: 1)  // 7 (|111⟩)
    /// BitUtilities.flipBit(state, qubit: 0)  // 4 (|100⟩)
    /// BitUtilities.flipBit(state, qubit: 2)  // 1 (|001⟩)
    /// ```
    ///
    /// - Parameters:
    ///   - index: Original basis state index
    ///   - qubit: Qubit position to flip
    /// - Returns: Modified basis state index with flipped bit
    /// - Complexity: O(1) - single CPU instruction (XOR)
    /// - Note: Core operation for X, CNOT, and Toffoli gate implementations
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func flipBit(_ index: Int, qubit: Int) -> Int {
        index ^ (1 << qubit)
    }

    /// Clear bit at specific qubit position (set to 0)
    ///
    /// Convenience method equivalent to `setBit(_:qubit:value:)` with value 0, provided for clarity when
    /// intent is specifically to clear a bit. Algorithm: `index & ~(1 << qubit)` creates mask with
    /// 0 at target position and 1s elsewhere, then AND operation clears the bit.
    ///
    /// **Example:**
    /// ```swift
    /// let state = 7  // |111⟩
    /// BitUtilities.clearBit(state, qubit: 1)  // 5 (|101⟩)
    /// ```
    ///
    /// - Parameters:
    ///   - index: Original basis state index
    ///   - qubit: Qubit position to clear
    /// - Returns: Modified basis state index with bit cleared
    /// - Complexity: O(1) - constant bitwise operations
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func clearBit(_ index: Int, qubit: Int) -> Int {
        index & ~(1 << qubit)
    }

    /// Create bit mask for specific qubit position
    ///
    /// Generates integer with single bit set at qubit position, all others zero. Algorithm: `1 << qubit`
    /// shifts 1 left by qubit positions to create mask value 2^qubit. Useful for precomputing masks in
    /// gate application, batch operations, and bit field manipulation.
    ///
    /// **Example:**
    /// ```swift
    /// BitUtilities.bitMask(qubit: 0)  // 1 (0b001)
    /// BitUtilities.bitMask(qubit: 1)  // 2 (0b010)
    /// BitUtilities.bitMask(qubit: 2)  // 4 (0b100)
    /// ```
    ///
    /// - Parameter qubit: Qubit position (0 = LSB)
    /// - Returns: Integer with single bit set (2^qubit)
    /// - Complexity: O(1) - single shift operation
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func bitMask(qubit: Int) -> Int {
        1 << qubit
    }

    /// Extract multiple bits as integer value
    ///
    /// Retrieves values of multiple qubits and combines them into single integer by iterating through
    /// qubit positions, extracting each bit, and combining with appropriate weight. Useful for multi-qubit
    /// gate conditions (Toffoli control checks) and multi-qubit measurement outcomes.
    ///
    /// **Example:**
    /// ```swift
    /// let state = 7  // |111⟩
    /// BitUtilities.getBits(state, qubits: [0, 1])  // 3 (both bits are 1)
    /// BitUtilities.getBits(state, qubits: [1, 2])  // 3
    /// ```
    ///
    /// - Parameters:
    ///   - index: Basis state index
    ///   - qubits: Array of qubit positions to extract
    /// - Returns: Integer formed from extracted bits (qubits[0] = LSB)
    /// - Complexity: O(k) where k = qubits.count
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func getBits(_ index: Int, qubits: [Int]) -> Int {
        var result = 0
        for position in 0 ..< qubits.count {
            let bit = (index >> qubits[position]) & 1
            result |= (bit << position)
        }
        return result
    }

    /// Extract two bits as integer value (optimized for Toffoli gate)
    ///
    /// Specialized fast path for the common 2-qubit case avoiding loop overhead entirely with direct
    /// bit extraction. Algorithm: `((index >> qubit0) & 1) | (((index >> qubit1) & 1) << 1)` extracts
    /// both bits in parallel. Primarily used for Toffoli gate control qubit checks.
    ///
    /// **Example:**
    /// ```swift
    /// let state = 7  // |111⟩
    /// BitUtilities.getTwoBits(state, qubit0: 0, qubit1: 1)  // 3
    /// BitUtilities.getTwoBits(state, qubit0: 1, qubit1: 2)  // 3
    /// ```
    ///
    /// - Parameters:
    ///   - index: Basis state index
    ///   - qubit0: First qubit position (becomes bit 0 of result)
    ///   - qubit1: Second qubit position (becomes bit 1 of result)
    /// - Returns: Integer 0-3 formed from the two extracted bits
    /// - Complexity: O(1) - direct bit extraction without loop
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func getTwoBits(_ index: Int, qubit0: Int, qubit1: Int) -> Int {
        ((index >> qubit0) & 1) | (((index >> qubit1) & 1) << 1)
    }

    /// Insert zero bit at position, shifting higher bits up
    ///
    /// Transforms iteration index for efficient masked traversal by inserting a zero at the specified
    /// bit position. Bits below position remain unchanged, bits at and above shift up by one. Enables
    /// iterating dimension/2 values instead of dimension values with guard-continue for single-qubit
    /// gate expansion where target qubit bit must be zero in the base index.
    ///
    /// **Example:**
    /// ```swift
    /// BitUtilities.insertZeroBit(0b11, at: 1)  // 0b101 (5)
    /// BitUtilities.insertZeroBit(0b11, at: 0)  // 0b110 (6)
    /// ```
    ///
    /// - Parameters:
    ///   - value: Iteration index to transform
    ///   - position: Bit position to insert zero
    /// - Returns: Transformed index with zero at position
    /// - Complexity: O(1) - constant bitwise operations
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func insertZeroBit(_ value: Int, at position: Int) -> Int {
        let lowMask = (1 << position) - 1
        let low = value & lowMask
        let high = (value & ~lowMask) << 1
        return high | low
    }

    /// Insert two zero bits at positions, shifting higher bits up
    ///
    /// Transforms iteration index for efficient dual-masked traversal by inserting zeros at two bit
    /// positions. Enables iterating dimension/4 values instead of dimension values with guard-continue
    /// for two-qubit gate expansion where both control and target qubit bits must be zero in base index.
    ///
    /// **Example:**
    /// ```swift
    /// BitUtilities.insertTwoZeroBits(0b1, low: 0, high: 2)  // 0b0100 (4)
    /// ```
    ///
    /// - Parameters:
    ///   - value: Iteration index to transform
    ///   - low: Lower bit position (must be < high)
    ///   - high: Higher bit position (must be > low)
    /// - Returns: Transformed index with zeros at both positions
    /// - Complexity: O(1) - constant bitwise operations
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func insertTwoZeroBits(_ value: Int, low: Int, high: Int) -> Int {
        let afterLow = insertZeroBit(value, at: low)
        return insertZeroBit(afterLow, at: high)
    }
}
