// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Centralized validation utilities for quantum computing parameters
///
/// Provides consistent precondition checks used throughout the quantum simulator ensuring uniform
/// error messages and validation logic across all modules. All validations use `precondition` which
/// terminates on failure in debug builds and may be optimized away in release builds for zero runtime
/// cost. Single source of truth for validation rules prevents inconsistent error handling.
///
/// **Example**:
/// ```swift
/// ValidationUtilities.validatePositiveQubits(numQubits)
/// ValidationUtilities.validateQubitIndex(qubit, numQubits: state.numQubits)
/// ValidationUtilities.validateNormalizedState(state)
/// ValidationUtilities.validateUnitary(gateMatrix)
/// ```
public enum ValidationUtilities {
    /// Validate that number of qubits is positive (at least 1)
    ///
    /// - Parameter numQubits: Number of qubits to validate
    /// - Precondition: numQubits must be > 0
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validatePositiveQubits(_ numQubits: Int) {
        precondition(numQubits > 0, "Number of qubits must be positive (got \(numQubits))")
    }

    /// Validate that number of qubits is within memory limits
    ///
    /// - Parameter numQubits: Number of qubits to validate
    /// - Precondition: numQubits must be <= 30
    /// - Complexity: O(1)
    /// - Note: 30-qubit limit = 2^30 amplitudes = ~8GB for Complex<Double>
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
    /// - Parameter state: Quantum state to validate
    /// - Precondition: state must be normalized (within 1e-10 tolerance)
    /// - Complexity: O(1) - delegates to state.isNormalized()
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
    /// - Parameters:
    ///   - index: Index to validate
    ///   - bound: Upper bound (exclusive)
    ///   - name: Descriptive name for error message
    /// - Precondition: 0 <= index < bound
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateIndexInBounds(_ index: Int, bound: Int, name: String) {
        precondition(
            index >= 0 && index < bound,
            "\(name) \(index) out of bounds (valid range: 0..<\(bound))"
        )
    }

    /// Validate that qubit index is within bounds
    ///
    /// - Parameters:
    ///   - qubit: Qubit index to validate
    ///   - numQubits: Total number of qubits in system
    /// - Precondition: 0 <= qubit < numQubits
    /// - Complexity: O(1)
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
    /// - Parameters:
    ///   - qubits: Array of qubit indices to validate
    ///   - numQubits: Total number of qubits in system
    /// - Precondition: All qubits must satisfy 0 <= qubit < numQubits
    /// - Complexity: O(k) where k = qubits.count
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

    /// Validate that complex number denominator is non-zero for division
    ///
    /// - Parameters:
    ///   - magnitudeSquared: Magnitude squared of denominator
    ///   - threshold: Division threshold below which denominator is considered zero
    /// - Precondition: magnitudeSquared > threshold
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateComplexDivisionByZero<T: BinaryFloatingPoint>(_ magnitudeSquared: T, threshold: T) {
        precondition(magnitudeSquared > threshold, "Complex division by zero")
    }

    /// Validate that integer value is positive
    ///
    /// - Parameters:
    ///   - value: Value to validate
    ///   - name: Parameter name for error message
    /// - Precondition: value > 0
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validatePositiveInt(_ value: Int, name: String) {
        precondition(value > 0, "\(name) must be positive (got \(value))")
    }

    /// Validate that integer value is non-negative
    ///
    /// - Parameters:
    ///   - value: Value to validate
    ///   - name: Parameter name for error message
    /// - Precondition: value >= 0
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateNonNegativeInt(_ value: Int, name: String) {
        precondition(value >= 0, "\(name) must be non-negative (got \(value))")
    }

    /// Validate that double value is positive
    ///
    /// - Parameters:
    ///   - value: Value to validate
    ///   - name: Parameter name for error message
    /// - Precondition: value > 0
    /// - Complexity: O(1)
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
    /// - Complexity: O(1)
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
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateUpperBound(_ value: Int, max: Int, name: String) {
        precondition(value <= max, "\(name) must be ≤ \(max) (got \(value))")
    }

    /// Validate that integer value is within inclusive lower bound
    ///
    /// - Parameters:
    ///   - value: Value to validate
    ///   - min: Minimum allowed value (inclusive)
    ///   - name: Parameter name for error message
    /// - Precondition: value >= min
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateLowerBound(_ value: Int, min: Int, name: String) {
        precondition(value >= min, "\(name) must be ≥ \(min) (got \(value))")
    }

    /// Validate that double value is within half-open range [min, max)
    ///
    /// - Parameters:
    ///   - value: Value to validate
    ///   - min: Minimum allowed value (inclusive)
    ///   - max: Maximum allowed value (exclusive)
    ///   - name: Parameter name for error message
    /// - Precondition: min <= value < max
    /// - Complexity: O(1)
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
    /// - Complexity: O(1)
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
    /// - Complexity: O(1)
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
    /// - Complexity: O(n)
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
    /// - Complexity: O(n)
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
    /// - Complexity: O(n)
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
    /// - Complexity: O(1) for 2-3 qubits, O(n) for larger arrays
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateUniqueQubits(_ qubits: [Int]) {
        switch qubits.count {
        case 0, 1: return
        case 2: precondition(qubits[0] != qubits[1], "Qubit indices must be unique (got \(qubits))")
        case 3:
            precondition(
                qubits[0] != qubits[1] && qubits[0] != qubits[2] && qubits[1] != qubits[2],
                "Qubit indices must be unique (got \(qubits))"
            )
        default:
            let uniqueQubits = Set(qubits)
            precondition(uniqueQubits.count == qubits.count, "Qubit indices must be unique (got \(qubits))")
        }
    }

    /// Validate that two arrays have equal count
    ///
    /// - Parameters:
    ///   - array1: First array
    ///   - array2: Second array
    ///   - name1: First array name for error message
    ///   - name2: Second array name for error message
    /// - Precondition: array1.count == array2.count
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateEqualCounts(_ array1: [some Any], _ array2: [some Any], name1: String, name2: String) {
        precondition(
            array1.count == array2.count,
            "\(name1) and \(name2) must have equal counts (got \(array1.count) and \(array2.count))"
        )
    }

    /// Validate array count matches expected value
    ///
    /// - Parameters:
    ///   - array: Array to validate
    ///   - expected: Expected count
    ///   - name: Array name for error message
    /// - Precondition: array.count == expected
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateArrayCount(_ array: [some Any], expected: Int, name: String) {
        precondition(
            array.count == expected,
            "\(name) count must be \(expected) but got \(array.count)"
        )
    }

    // MARK: - Matrix Validations

    /// Validate that matrix is square
    ///
    /// - Parameters:
    ///   - matrix: Matrix to validate
    ///   - name: Matrix name for error message
    /// - Precondition: All rows have same length as number of rows
    /// - Complexity: O(n) to check all rows
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateSquareMatrix(_ matrix: [[some Any]], name: String) {
        precondition(!matrix.isEmpty, "\(name) must not be empty")
        let n = matrix.count
        precondition(matrix.allSatisfy { $0.count == n }, "\(name) must be square (got \(matrix.count)x\(matrix[0].count))")
    }

    /// Validate that two matrices have same dimensions
    ///
    /// - Parameters:
    ///   - matrix1: First matrix
    ///   - matrix2: Second matrix
    ///   - name1: First matrix name
    ///   - name2: Second matrix name
    /// - Precondition: Matrices must have same dimensions
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateSameDimensions(_ matrix1: [[some Any]], _ matrix2: [[some Any]], name1: String, name2: String) {
        precondition(!matrix1.isEmpty && !matrix2.isEmpty, "Matrices must not be empty")
        precondition(
            matrix1.count == matrix2.count,
            "\(name1) and \(name2) must have same dimensions (got \(matrix1.count)x\(matrix1[0].count) and \(matrix2.count)x\(matrix2[0].count))"
        )
    }

    /// Validate that matrix dimension is positive
    ///
    /// - Parameters:
    ///   - dimension: Matrix dimension
    /// - Precondition: dimension > 0
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateMatrixDimension(_ dimension: Int) {
        precondition(dimension > 0, "Matrix dimension must be positive (got \(dimension))")
    }

    /// Validate that square matrix has expected dimension
    ///
    /// - Parameters:
    ///   - matrix: Square matrix to validate
    ///   - expected: Expected dimension (rows and columns)
    ///   - name: Matrix name for error message
    /// - Precondition: matrix.count == expected
    /// - Complexity: O(1)
    /// - Note: Used for batch unitary validation where all matrices must match state space size
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateMatrixDimensionEquals(_ matrix: [[some Any]], expected: Int, name: String) {
        precondition(
            matrix.count == expected,
            "\(name) dimension must be \(expected)x\(expected) (got \(matrix.count)x\(matrix.count))"
        )
    }

    // MARK: - Probability Validations

    /// Validate that probability array sums to 1.0 (within tolerance)
    ///
    /// - Parameters:
    ///   - probabilities: Probability distribution
    ///   - tolerance: Numerical tolerance (default 1e-10)
    /// - Precondition: abs(sum - 1.0) < tolerance
    /// - Complexity: O(n)
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
    /// - Complexity: O(1)
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
    /// - Complexity: O(1)
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
    /// - Complexity: O(n) where n = basisState.count
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
    /// - Complexity: O(1)
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
    /// - Complexity: O(1)
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
    /// - Complexity: O(1)
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
    /// - Complexity: O(1)
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
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateNonEmptyString(_ string: String, name: String) {
        precondition(!string.isEmpty, "\(name) cannot be empty")
    }

    // MARK: - Circuit Validations

    /// Validate circuit operations and qubit indices
    ///
    /// Checks that all operations have valid qubit indices within circuit bounds and that
    /// gates are properly formed. Central validation for quantum circuits before execution.
    ///
    /// - Parameters:
    ///   - operations: Array of gate operations to validate
    ///   - numQubits: Number of qubits in the circuit
    /// - Precondition: All operation qubits must be in range [0, numQubits)
    /// - Precondition: All gates must validate against maxAllowedQubit (29 for 30-qubit limit)
    /// - Complexity: O(n x m) where n = operations count, m = qubits per operation
    @_effects(readonly)
    @inlinable
    static func validateCircuitOperations(_ operations: [GateOperation], numQubits: Int) {
        let maxAllowedQubit = 29

        for operation in operations {
            precondition(
                operation.qubits.allSatisfy { $0 >= 0 && $0 < numQubits },
                "Circuit operation has qubit index out of bounds [0, \(numQubits))"
            )

            precondition(
                operation.gate.validateQubitIndices(operation.qubits, maxAllowedQubit: maxAllowedQubit),
                "Gate \(operation.gate) has invalid qubit configuration"
            )
        }
    }

    /// Validate that up-to index is within operation count bounds
    ///
    /// - Parameters:
    ///   - upToIndex: Index to validate
    ///   - operationCount: Total number of operations
    /// - Precondition: 0 <= upToIndex <= operationCount
    /// - Complexity: O(1)
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
    /// - Complexity: O(1)
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
    /// - Complexity: O(1)
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
    /// - Complexity: O(1)
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
    /// - Complexity: O(1)
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
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateAcceptExpandRatios(accept acceptRatio: Double, expand expandRatio: Double) {
        precondition(
            acceptRatio < expandRatio,
            "acceptRatio must be less than expandRatio (got \(acceptRatio) >= \(expandRatio))"
        )
    }

    // MARK: - Graph Validations

    /// Validate that graph edge connects two distinct vertices
    ///
    /// Self-loops (edges from a vertex to itself) are invalid in graph problems
    /// like MaxCut where edges must connect distinct vertices.
    ///
    /// - Parameters:
    ///   - vertex1: First vertex index
    ///   - vertex2: Second vertex index
    /// - Precondition: vertex1 != vertex2
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateDistinctVertices(_ vertex1: Int, _ vertex2: Int) {
        precondition(
            vertex1 != vertex2,
            "Self-loop edge (\(vertex1), \(vertex1)) is invalid. Edges must connect distinct vertices."
        )
    }

    // MARK: - Special Validations

    /// Validate that allocation dictionary contains required index
    ///
    /// - Parameters:
    ///   - allocation: Allocation dictionary
    ///   - index: Required index
    /// - Precondition: allocation[index] != nil
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateAllocationContainsIndex(_ allocation: ShotAllocation, index: Int) {
        precondition(allocation[index] != nil, "Allocation must contain entry for term index \(index)")
    }

    /// Validate that bypass flag is enabled (for testing-only code paths)
    ///
    /// - Parameter bypass: Bypass validation flag
    /// - Precondition: bypass == true
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateBypassEnabled(_ bypass: Bool) {
        precondition(bypass, "This initializer is for testing only")
    }

    /// Validates that a coupling dictionary key specifies exactly 1 or 2 qubits
    ///
    /// - Parameters:
    ///   - count: Number of qubit indices parsed from the key
    ///   - key: Original key string for error message
    /// - Precondition: count == 1 || count == 2
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateCouplingKeyFormat(_ count: Int, key: String) {
        precondition(
            count == 1 || count == 2,
            "Invalid coupling key '\(key)': must specify 1-2 qubits (e.g., '0' for local field, '01' or '0-1' for coupling)"
        )
    }

    // MARK: - Parameter Vector Validations

    /// Validate parameter vector length matches expected count
    ///
    /// Used for VQE/QAOA parameter binding where vector length must match
    /// the number of parameters in the parameterized circuit.
    ///
    /// - Parameters:
    ///   - actual: Actual vector length
    ///   - expected: Expected vector length
    ///   - name: Parameter name for error message (default: "Parameter vector")
    /// - Precondition: actual == expected
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateParameterVectorLength(_ actual: Int, expected: Int, name: String = "Parameter vector") {
        precondition(
            actual == expected,
            "\(name) length must be \(expected) (got \(actual))"
        )
    }

    // MARK: - Matrix Size Validations

    /// Validate that parameter exists in circuit's parameter set
    ///
    /// - Parameters:
    ///   - parameterName: Name of parameter to check
    ///   - parameterSet: Set of valid parameter names in circuit
    /// - Precondition: parameterSet.contains(parameterName)
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateParameterExists(_ parameterName: String, in parameterSet: Set<String>) {
        precondition(parameterSet.contains(parameterName), "Parameter '\(parameterName)' not found in circuit")
    }

    /// Validate that matrix is 2x2 for single-qubit gates
    ///
    /// - Parameter matrix: Matrix to validate
    /// - Precondition: matrix is 2x2
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validate2x2Matrix(_ matrix: [[Complex<Double>]]) {
        precondition(matrix.count == 2, "Single-qubit gate requires 2x2 matrix (got \(matrix.count) rows)")
        precondition(matrix[0].count == 2, "Single-qubit gate requires 2x2 matrix (row 0 has \(matrix[0].count) columns)")
        precondition(matrix[1].count == 2, "Single-qubit gate requires 2x2 matrix (row 1 has \(matrix[1].count) columns)")
    }

    /// Validate that matrix is 4x4 for two-qubit gates
    ///
    /// - Parameter matrix: Matrix to validate
    /// - Precondition: matrix is 4x4
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validate4x4Matrix(_ matrix: [[Complex<Double>]]) {
        precondition(matrix.count == 4, "Two-qubit gate requires 4x4 matrix (got \(matrix.count) rows)")
        precondition(matrix[0].count == 4, "Two-qubit gate requires 4x4 matrix (row 0 has \(matrix[0].count) columns)")
        precondition(matrix[1].count == 4, "Two-qubit gate requires 4x4 matrix (row 1 has \(matrix[1].count) columns)")
        precondition(matrix[2].count == 4, "Two-qubit gate requires 4x4 matrix (row 2 has \(matrix[2].count) columns)")
        precondition(matrix[3].count == 4, "Two-qubit gate requires 4x4 matrix (row 3 has \(matrix[3].count) columns)")
    }

    /// Validate that matrix is unitary (U†U = I)
    ///
    /// Custom quantum gates must be unitary to preserve quantum state normalization.
    ///
    /// - Parameter matrix: Matrix to validate
    /// - Precondition: matrix must be unitary within tolerance (1e-10)
    /// - Complexity: O(n³) where n = matrix dimension (2 for single-qubit, 4 for two-qubit gates)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateUnitary(_ matrix: [[Complex<Double>]]) {
        precondition(QuantumGate.isUnitary(matrix), "Matrix is not unitary (U†U ≠ I)")
    }

    /// Validate that parameter bindings contain a specific parameter
    ///
    /// Used for gradient computation where base bindings must include
    /// the parameter being shifted.
    ///
    /// - Parameters:
    ///   - parameterName: Name of parameter to check
    ///   - bindings: Dictionary of parameter bindings
    /// - Precondition: bindings[parameterName] != nil
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateParameterBinding(_ parameterName: String, in bindings: [String: Double]) {
        precondition(
            bindings[parameterName] != nil,
            "Parameter '\(parameterName)' must have a binding in baseBindings"
        )
    }

    /// Validate that parameter bindings exactly match circuit parameters
    ///
    /// Ensures all circuit parameters have bindings and no extra bindings exist.
    /// This is a programmer error check - you should know which parameters your circuit has.
    ///
    /// - Parameters:
    ///   - bindings: Dictionary of parameter bindings provided
    ///   - parameters: Array of parameters in the circuit
    ///   - parameterSet: Set of parameter names for O(1) lookup
    /// - Precondition: bindings keys exactly match parameter names
    /// - Complexity: O(n + m) where n = parameters.count, m = bindings.count
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateCompleteParameterBindings(
        _ bindings: [String: Double],
        parameters: [Parameter],
        parameterSet: Set<String>
    ) {
        for param in parameters {
            precondition(
                bindings[param.name] != nil,
                "Missing binding for parameter '\(param.name)'"
            )
        }
        for key in bindings.keys {
            precondition(
                parameterSet.contains(key),
                "Extra parameter '\(key)' not in circuit"
            )
        }
    }
}
