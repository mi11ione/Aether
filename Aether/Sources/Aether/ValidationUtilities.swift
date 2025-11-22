// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

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

    /// Validate that index is within bounds [0, bound)
    ///
    /// Generic bounds check for indices, qubit indices, array access, state space indices, etc.
    /// Validates that index is non-negative and strictly less than the upper bound.
    ///
    /// - Parameters:
    ///   - index: Index to validate
    ///   - bound: Upper bound (exclusive)
    ///   - name: Descriptive name for error message
    /// - Precondition: 0 <= index < bound
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateIndexInBounds(_ index: Int, bound: Int, name: String) {
        precondition(
            index >= 0 && index < bound,
            "\(name) \(index) out of bounds (valid range: 0..<\(bound))"
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

    // MARK: - Numeric Validations

    /// Validate that integer value is positive
    ///
    /// - Parameters:
    ///   - value: Value to validate
    ///   - name: Parameter name for error message
    /// - Precondition: value > 0
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validatePositiveInt(_ value: Int, name: String) {
        precondition(value > 0, "\(name) must be positive (got \(value))")
    }

    /// Validate that double value is positive
    ///
    /// - Parameters:
    ///   - value: Value to validate
    ///   - name: Parameter name for error message
    /// - Precondition: value > 0
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validatePositiveDouble(_ value: Double, name: String) {
        precondition(value > 0, "\(name) must be positive (got \(value))")
    }

    /// Validate that double value is non-negative
    ///
    /// - Parameters:
    ///   - value: Value to validate
    ///   - name: Parameter name for error message
    /// - Precondition: value >= 0
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateNonNegativeDouble(_ value: Double, name: String) {
        precondition(value >= 0, "\(name) must be non-negative (got \(value))")
    }

    /// Validate that integer value is within inclusive upper bound
    ///
    /// - Parameters:
    ///   - value: Value to validate
    ///   - max: Maximum allowed value (inclusive)
    ///   - name: Parameter name for error message
    /// - Precondition: value <= max
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateUpperBound(_ value: Int, max: Int, name: String) {
        precondition(value <= max, "\(name) must be ≤ \(max) (got \(value))")
    }

    /// Validate that double value is within half-open range [min, max)
    ///
    /// - Parameters:
    ///   - value: Value to validate
    ///   - min: Minimum allowed value (inclusive)
    ///   - max: Maximum allowed value (exclusive)
    ///   - name: Parameter name for error message
    /// - Precondition: min <= value < max
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateHalfOpenRange(_ value: Double, min: Double, max: Double, name: String) {
        precondition(value >= min && value < max, "\(name) must be in [\(min), \(max)) (got \(value))")
    }

    /// Validate that double value is within open range (min, max]
    ///
    /// - Parameters:
    ///   - value: Value to validate
    ///   - min: Minimum allowed value (exclusive)
    ///   - max: Maximum allowed value (inclusive)
    ///   - name: Parameter name for error message
    /// - Precondition: min < value <= max
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateOpenMinRange(_ value: Double, min: Double, max: Double, name: String) {
        precondition(value > min && value <= max, "\(name) must be in (\(min), \(max)] (got \(value))")
    }

    // MARK: - Array Validations

    /// Validate that array is not empty
    ///
    /// - Parameters:
    ///   - array: Array to validate
    ///   - name: Array name for error message
    /// - Precondition: !array.isEmpty
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateNonEmpty(_ array: [some Any], name: String) {
        precondition(!array.isEmpty, "\(name) must not be empty")
    }

    /// Validate that all integers in array are binary (0 or 1)
    ///
    /// - Parameters:
    ///   - array: Array to validate
    ///   - name: Array name for error message
    /// - Precondition: All elements must be 0 or 1
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateBinaryArray(_ array: [Int], name: String) {
        precondition(array.allSatisfy { $0 == 0 || $0 == 1 }, "\(name) must contain only 0 or 1 (got \(array))")
    }

    /// Validate that binary array contains at least one 1
    ///
    /// - Parameters:
    ///   - array: Binary array to validate
    ///   - name: Array name for error message
    /// - Precondition: array.contains(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateNonZeroBinary(_ array: [Int], name: String) {
        precondition(array.contains(1), "\(name) must contain at least one 1 (cannot be all zeros)")
    }

    /// Validate that all qubit indices are non-negative
    ///
    /// - Parameters:
    ///   - qubits: Qubit indices to validate
    /// - Precondition: All qubits >= 0
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateNonNegativeQubits(_ qubits: [Int]) {
        precondition(qubits.allSatisfy { $0 >= 0 }, "Qubit indices must be non-negative (got \(qubits))")
    }

    /// Validate that qubit indices are unique
    ///
    /// - Parameters:
    ///   - qubits: Qubit indices to validate
    /// - Precondition: No duplicate qubit indices
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateUniqueQubits(_ qubits: [Int]) {
        let uniqueQubits = Set(qubits)
        precondition(uniqueQubits.count == qubits.count, "Qubit indices must be unique (got \(qubits))")
    }

    /// Validate that two arrays have equal count
    ///
    /// - Parameters:
    ///   - array1: First array
    ///   - array2: Second array
    ///   - name1: First array name for error message
    ///   - name2: Second array name for error message
    /// - Precondition: array1.count == array2.count
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateEqualCounts(_ array1: [some Any], _ array2: [some Any], name1: String, name2: String) {
        precondition(
            array1.count == array2.count,
            "\(name1) and \(name2) must have equal counts (got \(array1.count) and \(array2.count))"
        )
    }

    // MARK: - Matrix Validations

    /// Validate that matrix is square
    ///
    /// - Parameters:
    ///   - matrix: Matrix to validate
    ///   - name: Matrix name for error message
    /// - Precondition: All rows have same length as number of rows
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateSquareMatrix(_ matrix: [[some Any]], name: String) {
        precondition(!matrix.isEmpty, "\(name) must not be empty")
        let n = matrix.count
        precondition(matrix.allSatisfy { $0.count == n }, "\(name) must be square (got \(matrix.count)×\(matrix[0].count))")
    }

    /// Validate that two matrices have same dimensions
    ///
    /// - Parameters:
    ///   - matrix1: First matrix
    ///   - matrix2: Second matrix
    ///   - name1: First matrix name
    ///   - name2: Second matrix name
    /// - Precondition: Matrices must have same dimensions
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateSameDimensions(_ matrix1: [[some Any]], _ matrix2: [[some Any]], name1: String, name2: String) {
        precondition(!matrix1.isEmpty && !matrix2.isEmpty, "Matrices must not be empty")
        precondition(
            matrix1.count == matrix2.count,
            "\(name1) and \(name2) must have same dimensions (got \(matrix1.count)×\(matrix1[0].count) and \(matrix2.count)×\(matrix2[0].count))"
        )
    }

    /// Validate that matrix dimension is positive
    ///
    /// - Parameters:
    ///   - dimension: Matrix dimension
    /// - Precondition: dimension > 0
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateMatrixDimension(_ dimension: Int) {
        precondition(dimension > 0, "Matrix dimension must be positive (got \(dimension))")
    }

    // MARK: - Probability Validations

    /// Validate that probability array sums to 1.0 (within tolerance)
    ///
    /// - Parameters:
    ///   - probabilities: Probability distribution
    ///   - tolerance: Numerical tolerance (default 1e-10)
    /// - Precondition: abs(sum - 1.0) < tolerance
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateProbabilityDistribution(_ probabilities: [Double], tolerance: Double = 1e-10) {
        precondition(!probabilities.isEmpty, "Probability array must not be empty")
        let sum = probabilities.reduce(0.0, +)
        precondition(abs(sum - 1.0) < tolerance, "Probabilities must sum to 1.0 (got \(sum))")
    }

    // MARK: - Quantum-Specific Validations

    /// Validate that value is binary (0 or 1)
    ///
    /// Used for measurement outcomes, single-qubit states, and other binary values.
    ///
    /// - Parameters:
    ///   - value: Value to validate
    ///   - name: Descriptive name for error message
    /// - Precondition: value == 0 || value == 1
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateBinaryValue(_ value: Int, name: String) {
        precondition(value == 0 || value == 1, "\(name) must be 0 or 1 (got \(value))")
    }

    /// Validate that basis state has exactly 2 components
    ///
    /// - Parameter basisState: Basis state vector
    /// - Precondition: basisState.count == 2
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateTwoComponentBasis(_ basisState: [Complex<Double>]) {
        precondition(basisState.count == 2, "Basis state must have 2 components (got \(basisState.count))")
    }

    /// Validate that basis state is normalized
    ///
    /// - Parameters:
    ///   - basisState: Basis state vector
    ///   - tolerance: Numerical tolerance (default 1e-10)
    /// - Precondition: abs(norm - 1.0) < tolerance
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateNormalizedBasis(_ basisState: [Complex<Double>], tolerance: Double = 1e-10) {
        let norm = basisState.reduce(0.0) { $0 + $1.magnitudeSquared }
        precondition(abs(norm - 1.0) < tolerance, "Basis state must be normalized (got norm² = \(norm))")
    }

    /// Validate that gate requires exactly one qubit
    ///
    /// - Parameter qubits: Qubit array
    /// - Precondition: qubits.count == 1
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateSingleQubitGate(_ qubits: [Int]) {
        precondition(qubits.count == 1, "Single-qubit gate requires exactly 1 qubit (got \(qubits.count))")
    }

    /// Validate that amplitude count matches state space size
    ///
    /// - Parameters:
    ///   - amplitudes: Amplitude array
    ///   - numQubits: Number of qubits
    /// - Precondition: amplitudes.count == (1 << numQubits)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateAmplitudeCount(_ amplitudes: [Complex<Double>], numQubits: Int) {
        let expectedCount = 1 << numQubits
        precondition(
            amplitudes.count == expectedCount,
            "Amplitude count must be 2^\(numQubits) = \(expectedCount) (got \(amplitudes.count))"
        )
    }

    /// Validate that quantum state has correct number of qubits
    ///
    /// - Parameters:
    ///   - state: Quantum state to validate
    ///   - required: Required number of qubits
    ///   - exact: If true, require exact match; if false, require minimum (default: false)
    /// - Precondition: exact ? state.numQubits == required : state.numQubits >= required
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateStateQubitCount(_ state: QuantumState, required: Int, exact: Bool = false) {
        if exact {
            precondition(
                state.numQubits == required,
                "State must have exactly \(required) qubits (got \(state.numQubits))"
            )
        } else {
            precondition(
                state.numQubits >= required,
                "State must have at least \(required) qubits (got \(state.numQubits))"
            )
        }
    }

    /// Validate that number of ones is within valid range for Dicke state
    ///
    /// - Parameters:
    ///   - numOnes: Number of qubits in |1⟩ state
    ///   - numQubits: Total number of qubits
    /// - Precondition: 0 <= numOnes <= numQubits
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateDickeParameters(_ numOnes: Int, numQubits: Int) {
        precondition(
            numOnes >= 0 && numOnes <= numQubits,
            "Number of ones must be in range 0...\(numQubits) (got \(numOnes))"
        )
    }

    // MARK: - String Validations

    /// Validate that string is not empty
    ///
    /// - Parameters:
    ///   - string: String to validate
    ///   - name: String name for error message
    /// - Precondition: !string.isEmpty
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateNonEmptyString(_ string: String, name: String) {
        precondition(!string.isEmpty, "\(name) cannot be empty")
    }

    // MARK: - Circuit Validations

    /// Validate that circuit passes validation
    ///
    /// - Parameter isValid: Result of circuit.validate()
    /// - Precondition: isValid == true
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateCircuit(_ isValid: Bool) {
        precondition(isValid, "Circuit validation failed")
    }

    /// Validate that up-to index is within operation count bounds
    ///
    /// - Parameters:
    ///   - upToIndex: Index to validate
    ///   - operationCount: Total number of operations
    /// - Precondition: 0 <= upToIndex <= operationCount
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateUpToIndex(_ upToIndex: Int, operationCount: Int) {
        precondition(
            upToIndex >= 0 && upToIndex <= operationCount,
            "upToIndex must be in range 0...\(operationCount) (got \(upToIndex))"
        )
    }

    /// Validate that controlled gate is single-qubit
    ///
    /// - Parameter qubitsRequired: Number of qubits required by gate
    /// - Precondition: qubitsRequired == 1
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateControlledGateIsSingleQubit(_ qubitsRequired: Int) {
        precondition(qubitsRequired == 1, "Multi-controlled U requires single-qubit gate (got \(qubitsRequired)-qubit gate)")
    }

    // MARK: - Educational Algorithm Validations

    /// Validate qubit count for educational algorithms
    ///
    /// Many educational algorithms have practical qubit limits for simulation.
    ///
    /// - Parameters:
    ///   - numQubits: Number of qubits
    ///   - max: Maximum allowed qubits
    ///   - algorithmName: Algorithm name for error message
    /// - Precondition: numQubits <= max
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateAlgorithmQubitLimit(_ numQubits: Int, max: Int, algorithmName: String) {
        precondition(numQubits <= max, "\(algorithmName) with >\(max) qubits is computationally expensive (got \(numQubits))")
    }

    /// Validate that algorithm has minimum required qubits
    ///
    /// - Parameters:
    ///   - numQubits: Number of qubits
    ///   - min: Minimum required qubits
    ///   - algorithmName: Algorithm name for error message
    /// - Precondition: numQubits >= min
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateMinimumQubits(_ numQubits: Int, min: Int, algorithmName: String) {
        precondition(numQubits >= min, "\(algorithmName) requires at least \(min) qubit\(min > 1 ? "s" : "") (got \(numQubits))")
    }

    // MARK: - Optimizer Validations

    /// Validate trust region radius relationships
    ///
    /// Trust region optimizers require: min < initial <= max
    ///
    /// - Parameters:
    ///   - minRadius: Minimum trust region radius
    ///   - initialRadius: Initial trust region radius
    ///   - maxRadius: Maximum trust region radius
    /// - Precondition: minRadius < initialRadius <= maxRadius
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateTrustRadiusRelationships(min minRadius: Double, initial initialRadius: Double, max maxRadius: Double) {
        precondition(
            minRadius < initialRadius,
            "minTrustRadius must be less than initialTrustRadius (got \(minRadius) >= \(initialRadius))"
        )
        precondition(
            initialRadius <= maxRadius,
            "initialTrustRadius must be less than or equal to maxTrustRadius (got \(initialRadius) > \(maxRadius))"
        )
    }

    /// Validate that accept ratio is less than expand ratio
    ///
    /// Trust region optimizers require acceptRatio < expandRatio for proper step acceptance logic.
    ///
    /// - Parameters:
    ///   - acceptRatio: Threshold for accepting steps
    ///   - expandRatio: Threshold for expanding trust region
    /// - Precondition: acceptRatio < expandRatio
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateAcceptExpandRatios(accept acceptRatio: Double, expand expandRatio: Double) {
        precondition(
            acceptRatio < expandRatio,
            "acceptRatio must be less than expandRatio (got \(acceptRatio) >= \(expandRatio))"
        )
    }

    // MARK: - Special Validations

    /// Validate that allocation dictionary contains required index
    ///
    /// - Parameters:
    ///   - allocation: Allocation dictionary
    ///   - index: Required index
    /// - Precondition: allocation[index] != nil
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateAllocationContainsIndex(_ allocation: [Int: Int], index: Int) {
        precondition(allocation[index] != nil, "Allocation must contain entry for term index \(index)")
    }

    /// Validate that bypass flag is enabled (for testing-only code paths)
    ///
    /// - Parameter bypass: Bypass validation flag
    /// - Precondition: bypass == true
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateBypassEnabled(_ bypass: Bool) {
        precondition(bypass, "This initializer is for testing only")
    }
}
