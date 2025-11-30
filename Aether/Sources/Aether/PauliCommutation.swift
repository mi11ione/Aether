// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Utilities for checking commutation relationships between Pauli operators.
///
/// Commutation analysis is fundamental to measurement optimization in quantum algorithms.
/// Commuting observables can be measured simultaneously, enabling significant reduction
/// in circuit executions required for expectation value estimation.
///
/// This namespace provides three levels of commutation checking: single-qubit Pauli basis
/// operators, general multi-qubit Pauli strings, and the stricter qubit-wise commutation
/// (QWC) property required for simultaneous measurement. Use ``areQWC(_:_:)`` to validate
/// grouping compatibility, then `measurementBasis(of:)` to extract the shared basis.
///
/// **Example:**
/// ```swift
/// let ps1 = PauliString(operators: [.init(basis: .x, qubit: 0)])
/// let ps2 = PauliString(operators: [.init(basis: .x, qubit: 0), .init(basis: .y, qubit: 1)])
///
/// // Check QWC property for measurement grouping
/// if PauliCommutation.areQWC(ps1, ps2) {
///     let basis = PauliCommutation.measurementBasis(of: [ps1, ps2])
///     // basis = [0: .x, 1: .y] - measure qubit 0 in X basis, qubit 1 in Y basis
/// }
/// ```
///
/// - SeeAlso: ``QWCGrouper``, ``Observable``
public enum PauliCommutation {
    // MARK: - Single-Qubit Commutation

    /// Checks whether two single-qubit Pauli basis operators commute.
    ///
    /// Single-qubit Pauli operators follow standard commutation rules: identity commutes
    /// with all operators, identical operators commute, and distinct non-identity operators
    /// anticommute. This forms the foundation for multi-qubit string commutation analysis.
    ///
    /// **Example:**
    /// ```swift
    /// PauliCommutation.commute(.x, .x)     // true - same operator
    /// PauliCommutation.commute(.x, .y)     // false - different Paulis anticommute
    /// PauliCommutation.commute(.x, nil)    // true - identity commutes with everything
    /// ```
    ///
    /// - Parameters:
    ///   - p1: First Pauli basis operator, or `nil` for identity
    ///   - p2: Second Pauli basis operator, or `nil` for identity
    /// - Returns: `true` if operators commute, `false` if they anticommute
    /// - Complexity: O(1)
    @_optimize(speed)
    @inlinable
    @_effects(readonly)
    public static func commute(_ p1: PauliBasis?, _ p2: PauliBasis?) -> Bool {
        if p1 == nil || p2 == nil { return true }
        if p1 == p2 { return true }
        return false
    }

    // MARK: - Multi-Qubit Commutation

    /// Checks whether two multi-qubit Pauli strings commute.
    ///
    /// Multi-qubit commutation follows the parity rule: strings commute when they have an
    /// even number of qubit positions with different non-identity operators. This is less
    /// restrictive than qubit-wise commutation and allows measurement of some non-QWC terms
    /// through basis transformation techniques.
    ///
    /// **Example:**
    /// ```swift
    /// let xz = PauliString(operators: [.init(basis: .x, qubit: 0), .init(basis: .z, qubit: 1)])
    /// let zx = PauliString(operators: [.init(basis: .z, qubit: 0), .init(basis: .x, qubit: 1)])
    /// let xy = PauliString(operators: [.init(basis: .x, qubit: 0), .init(basis: .y, qubit: 1)])
    ///
    /// PauliCommutation.commute(xz, zx)  // true - 2 differences (even)
    /// PauliCommutation.commute(xz, xy)  // false - 1 difference (odd)
    /// ```
    ///
    /// - Parameters:
    ///   - ps1: First Pauli string
    ///   - ps2: Second Pauli string
    /// - Returns: `true` if strings commute, `false` if they anticommute
    /// - Complexity: O(n·m) where n, m are the operator counts in each string
    /// - SeeAlso: ``areQWC(_:_:)`` for stricter qubit-wise commutation
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

    /// Checks whether two Pauli strings satisfy qubit-wise commutation (QWC).
    ///
    /// QWC is a stricter property than general commutation, requiring that operators on
    /// each qubit either match or one is identity. QWC strings share a common measurement
    /// basis and can be measured simultaneously without additional circuit overhead, making
    /// this the key property for measurement grouping optimization.
    ///
    /// **Example:**
    /// ```swift
    /// let xx = PauliString(operators: [.init(basis: .x, qubit: 0), .init(basis: .x, qubit: 1)])
    /// let xy = PauliString(operators: [.init(basis: .x, qubit: 0), .init(basis: .y, qubit: 1)])
    /// let yx = PauliString(operators: [.init(basis: .y, qubit: 0), .init(basis: .x, qubit: 1)])
    ///
    /// PauliCommutation.areQWC(xx, xy)  // true - qubit 0 matches (X), qubit 1 compatible
    /// PauliCommutation.areQWC(xy, yx)  // false - conflict on both qubits
    /// ```
    ///
    /// - Parameters:
    ///   - ps1: First Pauli string
    ///   - ps2: Second Pauli string
    /// - Returns: `true` if strings are qubit-wise commuting
    /// - Complexity: O(n·m) where n, m are the operator counts in each string
    @_optimize(speed)
    @inlinable
    @_effects(readonly)
    public static func areQWC(_ ps1: PauliString, _ ps2: PauliString) -> Bool {
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

    /// Extracts the shared measurement basis from a group of qubit-wise commuting Pauli strings.
    ///
    /// QWC strings can be measured simultaneously using a single basis rotation circuit. This
    /// method determines which basis (X, Y, or Z) should be applied to each qubit. Returns `nil`
    /// if the input strings are not QWC, indicating they cannot be measured together.
    ///
    /// **Example:**
    /// ```swift
    /// let xz = PauliString(operators: [.init(basis: .x, qubit: 0), .init(basis: .z, qubit: 1)])
    /// let xi = PauliString(operators: [.init(basis: .x, qubit: 0)])
    /// let iz = PauliString(operators: [.init(basis: .z, qubit: 1)])
    ///
    /// let basis = PauliCommutation.measurementBasis(of: [xz, xi, iz])
    /// // basis = [0: .x, 1: .z] - measure qubit 0 in X, qubit 1 in Z
    /// ```
    ///
    /// - Parameter strings: Array of Pauli strings expected to be QWC
    /// - Returns: Dictionary mapping qubit indices to measurement bases, or `nil` if strings conflict
    /// - Complexity: O(total operators across all strings)
    /// - Note: Empty array returns empty dictionary (vacuously true QWC property)
    /// - SeeAlso: ``areQWC(_:_:)``, ``QWCGrouper``
    @_optimize(speed)
    @_eagerMove
    public static func measurementBasis(of strings: [PauliString]) -> [Int: PauliBasis]? {
        guard !strings.isEmpty else { return [:] }

        var basis: [Int: PauliBasis] = [:]

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

    /// Extracts the measurement basis from a single Pauli string.
    ///
    /// Convenience overload for the common single-string case. Unlike the array version, this
    /// always succeeds since a single string vacuously satisfies QWC with itself.
    ///
    /// **Example:**
    /// ```swift
    /// let pauliString = PauliString(operators: [
    ///     .init(basis: .x, qubit: 0),
    ///     .init(basis: .y, qubit: 2),
    ///     .init(basis: .z, qubit: 3)
    /// ])
    ///
    /// let basis = PauliCommutation.measurementBasis(of: pauliString)
    /// // basis = [0: .x, 2: .y, 3: .z]
    /// ```
    ///
    /// - Parameter string: A Pauli string
    /// - Returns: Dictionary mapping qubit indices to measurement bases
    /// - Complexity: O(n) where n is the number of operators in the string
    @_optimize(speed)
    @_eagerMove
    public static func measurementBasis(of string: PauliString) -> [Int: PauliBasis] {
        var basis: [Int: PauliBasis] = [:]

        for op in string.operators {
            basis[op.qubit] = op.basis
        }

        return basis
    }
}
