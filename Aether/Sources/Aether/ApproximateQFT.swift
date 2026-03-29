// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Truncated Quantum Fourier Transform with proven error bounds and automatic threshold selection.
///
/// Standard QFT applies O(n²) controlled-phase rotations; approximate QFT truncates rotations
/// with index k > t where t is the truncation parameter, reducing gate count from O(n²) to
/// O(n·t). The operator-norm approximation error is bounded at O(n/2^t). For practical quantum
/// phase estimation where full QFT precision is unnecessary, setting t = O(log n) achieves
/// O(n log n) gates with polynomially bounded error.
///
/// Each qubit retains only its t nearest controlled-phase interactions rather than all n-1.
/// Given a target fidelity F, ``minimumTruncation(qubits:targetFidelity:)`` computes the
/// smallest t satisfying ||U_approx - U_exact|| ≤ √(1 - F).
///
/// **Example:**
/// ```swift
/// let k = ApproximateQFT.minimumTruncation(qubits: 8, targetFidelity: 0.999)
/// let circuit = ApproximateQFT.circuit(qubits: 8, truncation: k)
/// let error = ApproximateQFT.errorBound(qubits: 8, truncation: k)
/// ```
///
/// - SeeAlso: ``QuantumCircuit/qft(qubits:)``
/// - SeeAlso: ``QuantumCircuit/approximateQFT(qubits:truncation:)``
public enum ApproximateQFT {
    /// Creates approximate QFT circuit with truncated controlled rotations.
    ///
    /// Builds a QFT circuit on qubits indexed 0 through n-1, retaining only controlled-phase
    /// rotations R_k with k ≤ truncation. When truncation ≥ n the result is identical to the
    /// exact QFT. The circuit contains Hadamard gates, truncated controlled-phase gates, and
    /// ⌊n/2⌋ SWAP gates for qubit reversal.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = ApproximateQFT.circuit(qubits: 8, truncation: 5)
    /// let gates = ApproximateQFT.gateCount(qubits: 8, truncation: 5)
    /// let state = circuit.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (1-16)
    ///   - truncation: Maximum rotation index k to retain (≥ 1)
    /// - Returns: Circuit implementing the approximate QFT
    /// - Precondition: qubits in 1...16
    /// - Precondition: truncation ≥ 1
    /// - Complexity: O(n · min(t, n)) gates where t = truncation
    ///
    /// - SeeAlso: ``minimumTruncation(qubits:targetFidelity:)``
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func circuit(qubits: Int, truncation: Int) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateAlgorithmQubitLimit(qubits, max: 16, algorithmName: "ApproximateQFT")
        ValidationUtilities.validatePositiveInt(truncation, name: "truncation")

        return buildCircuit(
            qubits: Array(0 ..< qubits),
            totalQubits: qubits,
            truncation: truncation,
            inverse: false,
        )
    }

    /// Creates approximate QFT on explicit qubit indices for circuit composition.
    ///
    /// Applies the truncated QFT to specified qubit indices, enabling embedding within larger
    /// circuits where qubit allocation is managed externally. Qubit order in the array determines
    /// the logical bit ordering of the transform.
    ///
    /// **Example:**
    /// ```swift
    /// let qubits = [4, 5, 6, 7]
    /// let circuit = ApproximateQFT.circuit(qubits: qubits, truncation: 3)
    /// let state = circuit.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Array of qubit indices (non-empty, unique, non-negative)
    ///   - truncation: Maximum rotation index k to retain (≥ 1)
    /// - Returns: Circuit implementing the approximate QFT on specified qubits
    /// - Precondition: qubits is non-empty with unique non-negative indices
    /// - Precondition: truncation ≥ 1
    /// - Complexity: O(n · min(t, n)) gates where n = qubits.count
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func circuit(qubits: [Int], truncation: Int) -> QuantumCircuit {
        ValidationUtilities.validateNonEmpty(qubits, name: "qubits")
        ValidationUtilities.validateNonNegativeQubits(qubits)
        ValidationUtilities.validateUniqueQubits(qubits)
        ValidationUtilities.validatePositiveInt(truncation, name: "truncation")

        let totalQubits = qubits.max()! + 1 // safe: qubits validated non-empty
        return buildCircuit(
            qubits: qubits,
            totalQubits: totalQubits,
            truncation: truncation,
            inverse: false,
        )
    }

    /// Creates inverse approximate QFT circuit with truncated controlled rotations.
    ///
    /// Builds the adjoint of the approximate QFT by reversing gate order and negating
    /// phase angles. Composing circuit with this inverse using the
    /// same truncation approximates the identity within the error bound.
    ///
    /// **Example:**
    /// ```swift
    /// let forward = ApproximateQFT.circuit(qubits: 4, truncation: 3)
    /// let inverse = ApproximateQFT.inverseCircuit(qubits: 4, truncation: 3)
    /// let roundTrip = forward.count + inverse.count
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (1-16)
    ///   - truncation: Maximum rotation index k to retain (≥ 1)
    /// - Returns: Circuit implementing the inverse approximate QFT
    /// - Precondition: qubits in 1...16
    /// - Precondition: truncation ≥ 1
    /// - Complexity: O(n · min(t, n)) gates where t = truncation
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func inverseCircuit(qubits: Int, truncation: Int) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateAlgorithmQubitLimit(qubits, max: 16, algorithmName: "ApproximateQFT")
        ValidationUtilities.validatePositiveInt(truncation, name: "truncation")

        return buildCircuit(
            qubits: Array(0 ..< qubits),
            totalQubits: qubits,
            truncation: truncation,
            inverse: true,
        )
    }

    /// Creates inverse approximate QFT on explicit qubit indices.
    ///
    /// Applies the inverse truncated QFT to specified qubit indices for composition within
    /// larger circuits.
    ///
    /// **Example:**
    /// ```swift
    /// let qubits = [4, 5, 6, 7]
    /// let inverse = ApproximateQFT.inverseCircuit(qubits: qubits, truncation: 3)
    /// let state = inverse.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Array of qubit indices (non-empty, unique, non-negative)
    ///   - truncation: Maximum rotation index k to retain (≥ 1)
    /// - Returns: Circuit implementing the inverse approximate QFT on specified qubits
    /// - Precondition: qubits is non-empty with unique non-negative indices
    /// - Precondition: truncation ≥ 1
    /// - Complexity: O(n · min(t, n)) gates where n = qubits.count
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func inverseCircuit(qubits: [Int], truncation: Int) -> QuantumCircuit {
        ValidationUtilities.validateNonEmpty(qubits, name: "qubits")
        ValidationUtilities.validateNonNegativeQubits(qubits)
        ValidationUtilities.validateUniqueQubits(qubits)
        ValidationUtilities.validatePositiveInt(truncation, name: "truncation")

        let totalQubits = qubits.max()! + 1 // safe: qubits validated non-empty
        return buildCircuit(
            qubits: qubits,
            totalQubits: totalQubits,
            truncation: truncation,
            inverse: true,
        )
    }

    /// Computes the minimum truncation parameter achieving a target fidelity.
    ///
    /// Given n qubits and target fidelity F, returns the smallest integer t such that
    /// the operator-norm error satisfies ||U_approx - U_exact|| ≤ √(1 - F). Uses the
    /// bound nπ/2^t for the approximation error. When F = 1.0, returns n (exact QFT).
    ///
    /// **Example:**
    /// ```swift
    /// let k = ApproximateQFT.minimumTruncation(qubits: 8, targetFidelity: 0.999)
    /// let circuit = ApproximateQFT.circuit(qubits: 8, truncation: k)
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits
    ///   - targetFidelity: Desired fidelity in (0, 1]
    /// - Returns: Minimum truncation parameter t
    /// - Precondition: qubits > 0
    /// - Precondition: 0 < targetFidelity ≤ 1
    /// - Complexity: O(1)
    ///
    /// - SeeAlso: ``errorBound(qubits:truncation:)``
    @_effects(readonly)
    public static func minimumTruncation(qubits: Int, targetFidelity: Double) -> Int {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateOpenMinRange(targetFidelity, min: 0.0, max: 1.0, name: "targetFidelity")

        if targetFidelity >= 1.0 {
            return qubits
        }

        let maxError = (1.0 - targetFidelity).squareRoot()
        let n = Double(qubits)
        let ratio = n * Double.pi / maxError
        let t = Int(Foundation.log2(ratio).rounded(.up))
        return min(max(t, 1), qubits)
    }

    /// Computes the operator-norm error bound for the approximate QFT.
    ///
    /// Returns the upper bound ||U_approx - U_exact|| ≤ nπ/2^t where n is the qubit count
    /// and t is the truncation parameter. This bound comes from summing the contribution of
    /// each dropped controlled-phase rotation across all n target qubits, where each dropped
    /// rotation at distance k > t contributes at most π/2^k.
    ///
    /// **Example:**
    /// ```swift
    /// let error = ApproximateQFT.errorBound(qubits: 8, truncation: 5)
    /// let fidelity = 1.0 - error * error
    /// let isAcceptable = fidelity > 0.99
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits
    ///   - truncation: Truncation parameter
    /// - Returns: Upper bound on operator-norm error
    /// - Precondition: qubits > 0
    /// - Precondition: truncation ≥ 1
    /// - Complexity: O(1)
    ///
    /// - SeeAlso: ``minimumTruncation(qubits:targetFidelity:)``
    @_effects(readonly)
    public static func errorBound(qubits: Int, truncation: Int) -> Double {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validatePositiveInt(truncation, name: "truncation")

        if truncation >= qubits { return 0.0 }
        let n = Double(qubits)
        let denominator = pow(2.0, Double(truncation))
        return n * Double.pi / denominator
    }

    /// Computes the total gate count for the approximate QFT circuit.
    ///
    /// Returns the number of gates: n Hadamards + truncated controlled-phase gates + ⌊n/2⌋
    /// SWAPs. For truncation t and n qubits, the controlled-phase count is
    /// Σᵢ min(t-1, n-1-i) which is O(n·t) for t ≪ n.
    ///
    /// **Example:**
    /// ```swift
    /// let exact = ApproximateQFT.gateCount(qubits: 8, truncation: 8)
    /// let approx = ApproximateQFT.gateCount(qubits: 8, truncation: 3)
    /// let reduction = Double(exact - approx) / Double(exact)
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits
    ///   - truncation: Truncation parameter
    /// - Returns: Total number of gates in the circuit
    /// - Precondition: qubits > 0
    /// - Precondition: truncation ≥ 1
    /// - Complexity: O(n)
    @_effects(readonly)
    public static func gateCount(qubits: Int, truncation: Int) -> Int {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validatePositiveInt(truncation, name: "truncation")

        var count = qubits
        for i in 0 ..< qubits {
            count += max(min(truncation - 1, qubits - 1 - i), 0)
        }
        count += qubits / 2
        return count
    }

    /// Builds the approximate QFT or inverse circuit on given qubit indices.
    private static func buildCircuit(
        qubits: [Int],
        totalQubits: Int,
        truncation: Int,
        inverse: Bool,
    ) -> QuantumCircuit {
        let n = qubits.count
        var circuit = QuantumCircuit(qubits: totalQubits)

        if inverse {
            let swapCount = n / 2
            for i in 0 ..< swapCount {
                circuit.append(.swap, to: [qubits[i], qubits[n - 1 - i]])
            }

            for target in stride(from: n - 1, through: 0, by: -1) {
                let maxControl = min(target + truncation, n)
                for control in stride(from: maxControl - 1, through: target + 1, by: -1) {
                    let k = control - target + 1
                    let theta = -Double.pi / Double(1 << k)
                    circuit.append(.controlledPhase(theta), to: [qubits[control], qubits[target]])
                }
                circuit.append(.hadamard, to: qubits[target])
            }
        } else {
            for target in 0 ..< n {
                circuit.append(.hadamard, to: qubits[target])

                let maxControl = min(target + truncation, n)
                for control in (target + 1) ..< maxControl {
                    let k = control - target + 1
                    let theta = Double.pi / Double(1 << k)
                    circuit.append(.controlledPhase(theta), to: [qubits[control], qubits[target]])
                }
            }

            let swapCount = n / 2
            for i in 0 ..< swapCount {
                circuit.append(.swap, to: [qubits[i], qubits[n - 1 - i]])
            }
        }

        return circuit
    }
}

public extension QuantumCircuit {
    /// Creates approximate QFT circuit with explicit truncation parameter.
    ///
    /// Convenience factory delegating to ``ApproximateQFT``.
    /// Truncates controlled-phase rotations R_k with k > truncation, reducing gate count
    /// from O(n²) to O(n·t).
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.approximateQFT(qubits: 8, truncation: 5)
    /// let depth = circuit.depth
    /// let state = circuit.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (1-16)
    ///   - truncation: Maximum rotation index k to retain (≥ 1)
    /// - Returns: Circuit implementing the approximate QFT
    /// - Precondition: qubits in 1...16
    /// - Precondition: truncation ≥ 1
    /// - Complexity: O(n · min(t, n)) gates
    ///
    /// - SeeAlso: ``ApproximateQFT``
    /// - SeeAlso: ``qft(qubits:)``
    @_eagerMove
    static func approximateQFT(qubits: Int, truncation: Int) -> QuantumCircuit {
        ApproximateQFT.circuit(qubits: qubits, truncation: truncation)
    }

    /// Creates approximate QFT circuit with automatic truncation from target fidelity.
    ///
    /// Computes the minimum truncation parameter t such that the operator-norm error
    /// ||U_approx - U_exact|| ≤ √(1 - F), then builds the truncated circuit. Convenience
    /// for callers who reason about fidelity rather than truncation depth.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.approximateQFT(qubits: 8, targetFidelity: 0.999)
    /// let gates = circuit.count
    /// let state = circuit.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (1-16)
    ///   - targetFidelity: Desired fidelity in (0, 1]
    /// - Returns: Circuit implementing the approximate QFT achieving target fidelity
    /// - Precondition: qubits in 1...16
    /// - Precondition: 0 < targetFidelity ≤ 1
    /// - Complexity: O(n · min(t, n)) gates where t is auto-selected
    ///
    /// - SeeAlso: ``ApproximateQFT/minimumTruncation(qubits:targetFidelity:)``
    @_eagerMove
    static func approximateQFT(qubits: Int, targetFidelity: Double) -> QuantumCircuit {
        let truncation = ApproximateQFT.minimumTruncation(qubits: qubits, targetFidelity: targetFidelity)
        return ApproximateQFT.circuit(qubits: qubits, truncation: truncation)
    }
}
