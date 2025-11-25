// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Maps qubit index to Pauli basis for measurement.
///
/// Used throughout measurement optimization to specify which basis (X, Y, or Z)
/// should be measured for each qubit in a group of commuting Pauli operators.
public typealias MeasurementBasis = [Int: PauliBasis]

/// Utilities for checking commutation of Pauli operators and strings.
///
/// Two Pauli operators commute if their product is symmetric (commutes with itself).
/// For multi-qubit Pauli strings, they commute if the number of non-identity,
/// non-matching operator pairs is even.
public enum PauliCommutation {
    // MARK: - Single-Qubit Commutation

    /// Check if two single-qubit Pauli operators commute.
    ///
    /// - Parameters:
    ///   - p1: First Pauli basis operator (nil = identity)
    ///   - p2: Second Pauli basis operator (nil = identity)
    /// - Returns: True if operators commute, false otherwise
    ///
    /// Commutation rules:
    /// - I commutes with everything
    /// - X commutes with X and I
    /// - Y commutes with Y and I
    /// - Z commutes with Z and I
    /// - X and Y don't commute
    /// - X and Z don't commute
    /// - Y and Z don't commute
    @_optimize(speed)
    @inlinable
    @_effects(readonly)
    public static func commute(_ p1: PauliBasis?, _ p2: PauliBasis?) -> Bool {
        // Identity commutes with everything
        if p1 == nil || p2 == nil { return true }

        // Same operators commute
        if p1 == p2 { return true }

        // Different non-identity Pauli operators anticommute
        return false
    }

    // MARK: - Multi-Qubit Commutation

    /// Check if two Pauli strings commute.
    ///
    /// - Parameters:
    ///   - ps1: First Pauli string
    ///   - ps2: Second Pauli string
    /// - Returns: True if Pauli strings commute, false otherwise
    ///
    /// Two Pauli strings commute if the number of positions where they have
    /// different, non-identity operators is even.
    ///
    /// Example:
    /// - X₀Y₁ and Y₀X₁ commute (2 differences: positions 0 and 1)
    /// - X₀Y₁ and X₀Z₁ don't commute (1 difference: position 1)
    /// - X₀I₁ and I₀Y₁ commute (0 differences)
    @_optimize(speed)
    @inlinable
    @_effects(readonly)
    public static func commute(_ ps1: PauliString, _ ps2: PauliString) -> Bool {
        var anticommutingCount = 0

        for op1 in ps1.operators {
            for op2 in ps2.operators {
                if op1.qubit == op2.qubit, op1.basis != op2.basis {
                    anticommutingCount += 1
                    break
                }
            }
        }

        return anticommutingCount & 1 == 0
    }

    // MARK: - Qubit-wise Commutation

    /// Check if two Pauli strings are qubit-wise commuting (QWC).
    ///
    /// - Parameters:
    ///   - ps1: First Pauli string
    ///   - ps2: Second Pauli string
    /// - Returns: True if strings are qubit-wise commuting
    ///
    /// Two Pauli strings are QWC if for every qubit position, the operators
    /// either match or one is identity. This is stricter than general commutation.
    ///
    /// QWC strings can be measured simultaneously with a single basis rotation.
    ///
    /// Example:
    /// - X₀X₁ and X₀Y₁ are QWC (qubit 0: both X, qubit 1: X and Y)
    /// - X₀Y₁ and Y₀X₁ are NOT QWC (qubit 0: X vs Y, qubit 1: Y vs X)
    /// - X₀I₁ and I₀Y₁ are QWC (no conflicts)
    @_optimize(speed)
    @inlinable
    @_effects(readonly)
    public static func qubitWiseCommute(_ ps1: PauliString, _ ps2: PauliString) -> Bool {
        for op1 in ps1.operators {
            for op2 in ps2.operators {
                if op1.qubit == op2.qubit, op1.basis != op2.basis {
                    return false
                }
            }
        }
        return true
    }

    // MARK: - Measurement Basis Determination

    /// Determine the measurement basis for a group of qubit-wise commuting Pauli strings.
    ///
    /// - Parameter strings: Array of QWC Pauli strings
    /// - Returns: Dictionary mapping qubit index to measurement basis, or nil if strings don't QWC
    ///
    /// For each qubit, the measurement basis is the first non-identity operator encountered
    /// across all strings in the group.
    @_optimize(speed)
    @_eagerMove
    public static func measurementBasis(for strings: [PauliString]) -> MeasurementBasis? {
        guard !strings.isEmpty else { return [:] }

        var basis: MeasurementBasis = [:]

        for string in strings {
            for op in string.operators {
                if let existing = basis[op.qubit] {
                    if existing != op.basis { return nil }
                } else {
                    basis[op.qubit] = op.basis
                }
            }
        }

        return basis
    }
}
