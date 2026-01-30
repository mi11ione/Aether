// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Bit manipulation utilities for quantum basis state indexing
///
/// Centralizes common bit operations used throughout the quantum simulator for qubit indexing,
/// basis state manipulation, and gate application. All operations work on raw integers representing
/// basis state indices in little-endian qubit ordering (qubit 0 is LSB). Essential for gate
/// application logic, measurement outcome extraction, and control qubit evaluation.
///
/// Basis state |q2q1q0> maps to integer q2x4 + q1x2 + q0x1. Example: |101> = 5 (binary 0b101).
/// All operations are O(1) bitwise operations compiled to single CPU instructions
/// (shift, AND, OR, XOR) for zero-cost abstractions.
///
/// **Example:**
/// ```swift
/// let state = 5  // |101> in binary
/// let bit0 = BitUtilities.bit(state, qubit: 0)  // 1
/// let flipped = BitUtilities.flipBit(state, qubit: 1)  // 7 (|111>)
/// let mask = BitUtilities.bitMask(qubit: 2)  // 4 (0b100)
/// ```
///
/// - SeeAlso: ``GateApplication``
/// - SeeAlso: ``Measurement``
public enum BitUtilities {
    /// Extract bit value at specific qubit position
    ///
    /// Retrieves the value (0 or 1) of a single qubit from a basis state index using right shift
    /// and mask operations. Used for measurement outcome extraction, marginal probability computation,
    /// and control qubit evaluation in gate application.
    ///
    /// **Example:**
    /// ```swift
    /// let state = 5  // |101> in binary
    /// BitUtilities.bit(state, qubit: 0)  // 1 (LSB)
    /// BitUtilities.bit(state, qubit: 1)  // 0
    /// BitUtilities.bit(state, qubit: 2)  // 1 (MSB)
    /// ```
    ///
    /// - Parameters:
    ///   - index: Basis state index (0 to 2^n-1)
    ///   - qubit: Qubit position (0 = LSB, n-1 = MSB)
    /// - Returns: Bit value (0 or 1)
    /// - Complexity: O(1) - single CPU instruction (shift + AND)
    /// - Precondition: qubit >= 0
    /// - Precondition: index >= 0
    /// - SeeAlso: ``bits(_:qubits:)``
    /// - SeeAlso: ``bitMask(qubit:)``
    @_effects(readonly)
    @inlinable
    @inline(__always)
    @_optimize(speed)
    public static func bit(_ index: Int, qubit: Int) -> Int {
        (index >> qubit) & 1
    }

    /// Flip bit at specific qubit position
    ///
    /// Toggles the specified qubit: 0->1 or 1->0 using XOR operation. Essential for implementing
    /// X gates and CNOT operations which flip target qubits. Uses XOR to toggle the target bit.
    /// Core operation for X, CNOT, and Toffoli gate implementations.
    ///
    /// **Example:**
    /// ```swift
    /// let state = 5  // |101>
    /// BitUtilities.flipBit(state, qubit: 1)  // 7 (|111>)
    /// BitUtilities.flipBit(state, qubit: 0)  // 4 (|100>)
    /// BitUtilities.flipBit(state, qubit: 2)  // 1 (|001>)
    /// ```
    ///
    /// - Parameters:
    ///   - index: Original basis state index
    ///   - qubit: Qubit position to flip
    /// - Returns: Modified basis state index with flipped bit
    /// - Complexity: O(1) - single CPU instruction (XOR)
    /// - Precondition: qubit >= 0
    /// - Precondition: index >= 0
    /// - SeeAlso: ``clearBit(_:qubit:)``
    /// - SeeAlso: ``bit(_:qubit:)``
    @_effects(readonly)
    @inlinable
    @inline(__always)
    @_optimize(speed)
    public static func flipBit(_ index: Int, qubit: Int) -> Int {
        index ^ (1 << qubit)
    }

    /// Clear bit at specific qubit position (set to 0)
    ///
    /// Convenience method provided for clarity when
    /// intent is specifically to clear a bit. Uses AND with an inverted mask to force
    /// the target bit to zero while preserving all other bits.
    ///
    /// **Example:**
    /// ```swift
    /// let state = 7  // |111>
    /// let cleared = BitUtilities.clearBit(state, qubit: 1)  // 5 (|101>)
    /// assert(cleared == 5)
    /// let state2 = 6  // |110>
    /// BitUtilities.clearBit(state2, qubit: 2)  // 2 (|010>)
    /// ```
    ///
    /// - Parameters:
    ///   - index: Original basis state index
    ///   - qubit: Qubit position to clear
    /// - Returns: Modified basis state index with bit cleared
    /// - Complexity: O(1) - constant bitwise operations
    /// - Precondition: qubit >= 0
    /// - Precondition: index >= 0
    /// - SeeAlso: ``flipBit(_:qubit:)``
    @_effects(readonly)
    @inlinable
    @inline(__always)
    @_optimize(speed)
    public static func clearBit(_ index: Int, qubit: Int) -> Int {
        index & ~(1 << qubit)
    }

    /// Create bit mask for specific qubit position
    ///
    /// Generates integer with single bit set at qubit position, all others zero. Useful for
    /// precomputing masks in gate application, batch operations, and bit field manipulation.
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
    /// - Precondition: qubit >= 0
    /// - SeeAlso: ``bit(_:qubit:)``
    @_effects(readonly)
    @inlinable
    @inline(__always)
    @_optimize(speed)
    public static func bitMask(qubit: Int) -> Int {
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
    /// let state = 7  // |111>
    /// BitUtilities.bits(state, qubits: [0, 1])  // 3 (both bits are 1)
    /// BitUtilities.bits(state, qubits: [1, 2])  // 3
    /// ```
    ///
    /// - Parameters:
    ///   - index: Basis state index
    ///   - qubits: Array of qubit positions to extract
    /// - Returns: Integer formed from extracted bits (qubits[0] = LSB)
    /// - Complexity: O(k) where k = qubits.count
    /// - Precondition: All qubit indices >= 0
    /// - Precondition: index >= 0
    /// - SeeAlso: ``bit(_:qubit:)``
    @_effects(readonly)
    @inlinable
    @inline(__always)
    @_optimize(speed)
    public static func bits(_ index: Int, qubits: [Int]) -> Int {
        var result = 0
        for position in 0 ..< qubits.count {
            let bit = (index >> qubits[position]) & 1
            result |= (bit << position)
        }
        return result
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
    /// let value = 0b11  // 3
    /// let result = BitUtilities.insertZeroBit(value, at: 1)  // 0b101 (5)
    /// assert(result == 5)
    /// BitUtilities.insertZeroBit(0b11, at: 0)  // 0b110 (6)
    /// ```
    ///
    /// - Parameters:
    ///   - value: Iteration index to transform
    ///   - position: Bit position to insert zero
    /// - Returns: Transformed index with zero at position
    /// - Complexity: O(1) - constant bitwise operations
    /// - Precondition: position >= 0
    /// - SeeAlso: ``insertTwoZeroBits(_:low:high:)``
    @_effects(readonly)
    @inlinable
    @inline(__always)
    @_optimize(speed)
    public static func insertZeroBit(_ value: Int, at position: Int) -> Int {
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
    /// let value = 0b1  // 1
    /// let result = BitUtilities.insertTwoZeroBits(value, low: 0, high: 2)  // 0b0100 (4)
    /// assert(result == 4)
    /// let value2 = 0b11  // 3
    /// BitUtilities.insertTwoZeroBits(value2, low: 1, high: 3)  // 0b10100 (20)
    /// ```
    ///
    /// - Parameters:
    ///   - value: Iteration index to transform
    ///   - low: Lower bit position (must be < high)
    ///   - high: Higher bit position (must be > low)
    /// - Returns: Transformed index with zeros at both positions
    /// - Complexity: O(1) - constant bitwise operations
    /// - Precondition: low < high
    /// - Precondition: low >= 0
    /// - SeeAlso: ``insertZeroBit(_:at:)``
    @_effects(readonly)
    @inlinable
    @inline(__always)
    @_optimize(speed)
    public static func insertTwoZeroBits(_ value: Int, low: Int, high: Int) -> Int {
        let afterLow = insertZeroBit(value, at: low)
        return insertZeroBit(afterLow, at: high)
    }
}
