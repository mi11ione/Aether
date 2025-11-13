// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Shared validation utilities for quantum computing parameters
///
/// Centralizes common precondition checks used throughout the quantum simulator.
/// Ensures consistent error messages and validation logic across all modules.
/// All validations use `precondition` which terminates on failure in debug builds
/// and may be optimized away in release builds for performance.
///
/// **Design Philosophy**:
/// - Consistent error messages across codebase
/// - Single source of truth for validation rules
/// - Zero runtime cost in optimized builds (precondition inlining)
/// - Clear, actionable error messages for developers
@frozen
public enum ValidationUtilities {
    /// Validate that number of qubits is positive (at least 1)
    ///
    /// Quantum circuits require at least one qubit to be meaningful.
    /// Maximum practical limit is typically 30 qubits due to memory constraints
    /// (2^30 = 1GB of Complex<Double> amplitudes).
    ///
    /// - Parameter numQubits: Number of qubits to validate
    /// - Precondition: numQubits must be > 0
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validatePositiveQubits(_ numQubits: Int) {
        precondition(numQubits > 0, "Number of qubits must be positive (got \(numQubits))")
    }

    /// Validate that number of qubits is within memory limits
    ///
    /// States with >30 qubits require >8GB memory for amplitude storage.
    /// Enforces practical upper bound to prevent memory exhaustion.
    ///
    /// - Parameter numQubits: Number of qubits to validate
    /// - Precondition: numQubits must be <= 30
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateMemoryLimit(_ numQubits: Int) {
        precondition(
            numQubits <= 30,
            "Number of qubits must not exceed 30 (would require \(1 << numQubits) amplitudes, got \(numQubits) qubits)"
        )
    }

    /// Validate that quantum state satisfies normalization constraint
    ///
    /// All valid quantum states must have Σᵢ |cᵢ|² = 1 for Born rule probability
    /// interpretation. Non-normalized states indicate numerical errors or invalid
    /// state construction.
    ///
    /// - Parameter state: Quantum state to validate
    /// - Precondition: state must be normalized (within 1e-10 tolerance)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateNormalizedState(_ state: QuantumState) {
        precondition(
            state.isNormalized(),
            "State must be normalized (Σ|cᵢ|² = 1) before measurement or expectation value computation"
        )
    }

    /// Validate that qubit index is within bounds
    ///
    /// Qubit indices range from 0 to numQubits-1 (inclusive).
    /// Out-of-bounds access would corrupt state vector or cause undefined behavior.
    ///
    /// - Parameters:
    ///   - qubit: Qubit index to validate
    ///   - numQubits: Total number of qubits in system
    /// - Precondition: 0 <= qubit < numQubits
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateQubitIndex(_ qubit: Int, numQubits: Int) {
        precondition(
            qubit >= 0 && qubit < numQubits,
            "Qubit index \(qubit) out of bounds (valid range: 0..<\(numQubits))"
        )
    }

    /// Validate that all qubits in operation are within bounds
    ///
    /// Multi-qubit gates (CNOT, Toffoli, etc.) must operate on valid qubit indices.
    /// Checks that all indices are non-negative and less than total qubit count.
    ///
    /// - Parameters:
    ///   - qubits: Array of qubit indices to validate
    ///   - numQubits: Total number of qubits in system
    /// - Precondition: All qubits must satisfy 0 <= qubit < numQubits
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateOperationQubits(_ qubits: [Int], numQubits: Int) {
        precondition(
            qubits.allSatisfy { $0 >= 0 && $0 < numQubits },
            "All qubit indices must be in range 0..<\(numQubits) (got \(qubits))"
        )
    }

    /// Validate that array index is within bounds
    ///
    /// Generic bounds check for array access in circuit operations, parameter lists, etc.
    ///
    /// - Parameters:
    ///   - index: Index to validate
    ///   - count: Array length
    /// - Precondition: 0 <= index < count
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateArrayIndex(_ index: Int, count: Int) {
        precondition(
            index >= 0 && index < count,
            "Index \(index) out of bounds (valid range: 0..<\(count))"
        )
    }

    /// Validate that basis state index is within Hilbert space bounds
    ///
    /// Basis state indices range from 0 to 2^n-1 for n-qubit systems.
    ///
    /// - Parameters:
    ///   - stateIndex: Basis state index to validate
    ///   - stateSpaceSize: Hilbert space dimension (2^numQubits)
    /// - Precondition: 0 <= stateIndex < stateSpaceSize
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateBasisStateIndex(_ stateIndex: Int, stateSpaceSize: Int) {
        precondition(
            stateIndex >= 0 && stateIndex < stateSpaceSize,
            "Basis state index \(stateIndex) out of bounds (valid range: 0..<\(stateSpaceSize))"
        )
    }

    /// Validate that sample count is positive
    ///
    /// Measurement runs, shot allocation, and statistical tests require positive sample counts.
    ///
    /// - Parameter count: Number of samples/runs
    /// - Precondition: count > 0
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validatePositiveCount(_ count: Int, name: String = "count") {
        precondition(count > 0, "\(name.capitalized) must be positive (got \(count))")
    }
}
