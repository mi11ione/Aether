// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Centralized validation utilities for quantum computing parameters
///
/// Provides consistent precondition checks used throughout the quantum simulator ensuring uniform
/// error messages and validation logic across all modules. All validations use `precondition` which
/// terminates on failure in debug builds and may be optimized away in release builds for zero runtime
/// cost. Single source of truth for validation rules prevents inconsistent error handling.
///
/// **Example:**
/// ```swift
/// ValidationUtilities.validatePositiveQubits(qubits)
/// ValidationUtilities.validateQubitIndex(qubit, qubits: state.qubits)
/// ValidationUtilities.validateNormalizedState(state)
/// ValidationUtilities.validateUnitary(gateMatrix)
/// ```
public enum ValidationUtilities {
    /// Validate that number of qubits is positive (at least 1)
    ///
    /// - Parameter qubits: Number of qubits to validate
    /// - Precondition: qubits must be > 0
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validatePositiveQubits(_ qubits: Int) {
        precondition(qubits > 0, "Number of qubits must be positive (got \(qubits))")
    }

    /// Validate that number of qubits is within memory limits
    ///
    /// - Parameter qubits: Number of qubits to validate
    /// - Precondition: qubits must be <= 30
    /// - Complexity: O(1)
    /// - Note: 30-qubit limit = 2^30 amplitudes = ~8GB for Complex<Double>
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateMemoryLimit(_ qubits: Int) {
        precondition(
            qubits <= 30,
            "Number of qubits must not exceed 30 (would require \(1 << qubits) amplitudes, got \(qubits) qubits)",
        )
    }

    /// Validate that quantum state satisfies normalization constraint
    ///
    /// - Parameter state: Quantum state to validate
    /// - Precondition: state must be normalized (within 1e-10 tolerance)
    /// - Complexity: O(2^n) where n = qubits (iterates all amplitudes)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateNormalizedState(_ state: QuantumState) {
        precondition(
            state.isNormalized(),
            "State must be normalized (Σ|cᵢ|² = 1) before measurement or expectation value computation",
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
            "\(name) \(index) out of bounds (valid range: 0..<\(bound))",
        )
    }

    /// Validate that qubit index is within bounds
    ///
    /// - Parameters:
    ///   - qubit: Qubit index to validate
    ///   - qubits: Total number of qubits in system
    /// - Precondition: 0 <= qubit < qubits
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateQubitIndex(_ qubit: Int, qubits: Int) {
        precondition(
            qubit >= 0 && qubit < qubits,
            "Qubit index \(qubit) out of bounds (valid range: 0..<\(qubits))",
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
            "All qubit indices must be in range 0..<\(numQubits) (got \(qubits))",
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
                "Qubit indices must be unique (got \(qubits))",
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
            "\(name1) and \(name2) must have equal counts (got \(array1.count) and \(array2.count))",
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
            "\(name) count must be \(expected) but got \(array.count)",
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
            "\(name1) and \(name2) must have same dimensions (got \(matrix1.count)x\(matrix1[0].count) and \(matrix2.count)x\(matrix2[0].count))",
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
            "\(name) dimension must be \(expected)x\(expected) (got \(matrix.count)x\(matrix.count))",
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
    ///   - qubits: Number of qubits
    /// - Precondition: amplitudes.count == (1 << qubits)
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateAmplitudeCount(_ amplitudes: [Complex<Double>], qubits: Int) {
        let expectedCount = 1 << qubits
        precondition(
            amplitudes.count == expectedCount,
            "Amplitude count must be 2^\(qubits) = \(expectedCount) (got \(amplitudes.count))",
        )
    }

    /// Validate that quantum state has correct number of qubits
    ///
    /// - Parameters:
    ///   - state: Quantum state to validate
    ///   - required: Required number of qubits
    ///   - exact: If true, require exact match; if false, require minimum (default: false)
    /// - Precondition: exact ? state.qubits == required : state.qubits >= required
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateStateQubitCount(_ state: QuantumState, required: Int, exact: Bool = false) {
        if exact {
            precondition(
                state.qubits == required,
                "State must have exactly \(required) qubits (got \(state.qubits))",
            )
        } else {
            precondition(
                state.qubits >= required,
                "State must have at least \(required) qubits (got \(state.qubits))",
            )
        }
    }

    /// Validate that number of ones is within valid range for Dicke state
    ///
    /// - Parameters:
    ///   - numOnes: Number of qubits in |1⟩ state
    ///   - qubits: Total number of qubits
    /// - Precondition: 0 <= numOnes <= qubits
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateDickeParameters(_ numOnes: Int, qubits: Int) {
        precondition(
            numOnes >= 0 && numOnes <= qubits,
            "Number of ones must be in range 0...\(qubits) (got \(numOnes))",
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

    /// Validates that a circuit contains only unitary operations.
    ///
    /// - Precondition: Circuit must not contain non-unitary operations such as reset.
    ///
    /// **Example:**
    /// ```swift
    /// ValidationUtilities.validateUnitaryCircuit(circuit)
    /// ```
    @inline(__always)
    public static func validateUnitaryCircuit(_ circuit: QuantumCircuit) {
        for operation in circuit.operations {
            precondition(operation.isUnitary, "Circuit contains non-unitary operation (\(operation)). Cannot compute unitary matrix or inverse for circuits with reset, measurement, or other non-unitary operations.")
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
            "upToIndex must be in range 0...\(operationCount) (got \(upToIndex))",
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

    /// Validate that gate is a two-qubit gate for KAK decomposition.
    ///
    /// - Parameter qubitsRequired: Number of qubits required by the gate
    /// - Precondition: qubitsRequired == 2
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateTwoQubitGate(_ qubitsRequired: Int) {
        precondition(qubitsRequired == 2, "KAK decomposition requires two-qubit gate (got \(qubitsRequired)-qubit gate)")
    }

    // MARK: - Educational Algorithm Validations

    /// Validate qubit count for educational algorithms
    ///
    /// Many educational algorithms have practical qubit limits for simulation.
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits
    ///   - max: Maximum allowed qubits
    ///   - algorithmName: Algorithm name for error message
    /// - Precondition: qubits <= max
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateAlgorithmQubitLimit(_ qubits: Int, max: Int, algorithmName: String) {
        precondition(qubits <= max, "\(algorithmName) with >\(max) qubits is computationally expensive (got \(qubits))")
    }

    /// Validate that algorithm has minimum required qubits
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits
    ///   - min: Minimum required qubits
    ///   - algorithmName: Algorithm name for error message
    /// - Precondition: qubits >= min
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateMinimumQubits(_ qubits: Int, min: Int, algorithmName: String) {
        precondition(qubits >= min, "\(algorithmName) requires at least \(min) qubit\(min > 1 ? "s" : "") (got \(qubits))")
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
            "minTrustRadius must be less than initialTrustRadius (got \(minRadius) >= \(initialRadius))",
        )
        precondition(
            initialRadius <= maxRadius,
            "initialTrustRadius must be less than or equal to maxTrustRadius (got \(initialRadius) > \(maxRadius))",
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
            "acceptRatio must be less than expandRatio (got \(acceptRatio) >= \(expandRatio))",
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
            "Self-loop edge (\(vertex1), \(vertex1)) is invalid. Edges must connect distinct vertices.",
        )
    }

    // MARK: - Density Matrix Validations

    /// Validate that number of qubits is within density matrix memory limits
    ///
    /// Density matrices require 2^(2n) complex numbers (4^n scaling), severely limiting
    /// practical qubit counts. At 14 qubits: 4^14 * 16 bytes ≈ 4.3 GB.
    ///
    /// - Parameter qubits: Number of qubits to validate
    /// - Precondition: qubits must be <= 14
    /// - Complexity: O(1)
    /// - Note: 14-qubit limit = 4^14 elements ≈ 4.3GB for Complex<Double>
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateDensityMatrixMemoryLimit(_ qubits: Int) {
        precondition(
            qubits <= 14,
            "Density matrix simulation limited to 14 qubits (4^n scaling, got \(qubits) qubits requiring \(1 << (2 * qubits)) elements)",
        )
    }

    /// Validate that density matrix element count matches dimension squared
    ///
    /// - Parameters:
    ///   - elements: Flattened density matrix elements (row-major)
    ///   - dimension: Expected matrix dimension (2^qubits)
    /// - Precondition: elements.count == dimension * dimension
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateDensityMatrixElementCount(_ elements: [Complex<Double>], dimension: Int) {
        let expectedCount = dimension * dimension
        precondition(
            elements.count == expectedCount,
            "Density matrix element count must be \(dimension)*\(dimension) = \(expectedCount) (got \(elements.count))",
        )
    }

    /// Validate that noise channel error probability is in valid range [0, 1]
    ///
    /// - Parameters:
    ///   - probability: Error probability to validate
    ///   - name: Channel name for error message
    /// - Precondition: 0 <= probability <= 1
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateErrorProbability(_ probability: Double, name: String) {
        precondition(
            probability >= 0.0 && probability <= 1.0,
            "\(name) error probability must be in [0, 1] (got \(probability))",
        )
    }

    /// Validate that damping parameter gamma is in valid range [0, 1]
    ///
    /// - Parameters:
    ///   - gamma: Damping parameter to validate
    ///   - name: Channel name for error message
    /// - Precondition: 0 <= gamma <= 1
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateDampingParameter(_ gamma: Double, name: String) {
        precondition(
            gamma >= 0.0 && gamma <= 1.0,
            "\(name) damping parameter gamma must be in [0, 1] (got \(gamma))",
        )
    }

    /// Validate that thermal population is in valid range [0, 0.5]
    ///
    /// - Parameters:
    ///   - population: Thermal population to validate
    ///   - name: Channel name for error message
    /// - Precondition: 0 <= population <= 0.5
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateThermalPopulation(_ population: Double, name: String) {
        precondition(
            population >= 0.0 && population <= 0.5,
            "\(name) thermal population must be in [0, 0.5] (got \(population))",
        )
    }

    /// Validate that matrix determinant is non-singular
    ///
    /// - Parameters:
    ///   - determinant: Computed determinant value
    ///   - tolerance: Minimum absolute value for non-singularity
    ///   - name: Matrix name for error message
    /// - Precondition: |determinant| > tolerance
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateNonSingularDeterminant(_ determinant: Double, tolerance: Double = 1e-15, name: String) {
        precondition(
            abs(determinant) > tolerance,
            "\(name) is singular (determinant ≈ 0, got \(determinant))",
        )
    }

    /// Validate T₂ coherence time constraint: T₂ ≤ 2*T₁
    ///
    /// Physical constraint from decoherence theory: T₂ cannot exceed 2*T₁.
    ///
    /// - Parameters:
    ///   - t2: T₂ coherence time
    ///   - t1: T₁ relaxation time
    ///   - index: Optional qubit index for per-qubit validation
    /// - Precondition: T₂ ≤ 2*T₁
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateT2Constraint(_ t2: Double, t1: Double, index: Int? = nil) {
        let indexStr = index.map { "[\($0)]" } ?? ""
        precondition(
            t2 <= 2 * t1 + 1e-10,
            "T₂\(indexStr) must be ≤ 2*T₁\(indexStr) (got T₂=\(t2), T₁=\(t1))",
        )
    }

    /// Validate that all elements in array are positive
    ///
    /// - Parameters:
    ///   - values: Array of values to validate
    ///   - name: Array name for error message
    /// - Precondition: All values > 0
    /// - Complexity: O(n)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateAllPositive(_ values: [Double], name: String) {
        precondition(
            values.allSatisfy { $0 > 0 },
            "All \(name) values must be positive",
        )
    }

    /// Validate that confusion matrix is valid for measurement error model
    ///
    /// Confusion matrix M must be 2*2 with rows summing to 1.0 (stochastic matrix).
    /// M[i][j] = P(measure j | prepared i).
    ///
    /// - Parameter matrix: 2*2 confusion matrix to validate
    /// - Precondition: Matrix is 2*2, all elements in [0,1], rows sum to 1.0
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateConfusionMatrix(_ matrix: [[Double]]) {
        precondition(matrix.count == 2, "Confusion matrix must be 2*2 (got \(matrix.count) rows)")
        precondition(matrix[0].count == 2, "Confusion matrix must be 2*2 (row 0 has \(matrix[0].count) columns)")
        precondition(matrix[1].count == 2, "Confusion matrix must be 2*2 (row 1 has \(matrix[1].count) columns)")

        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                precondition(
                    matrix[i][j] >= 0.0 && matrix[i][j] <= 1.0,
                    "Confusion matrix elements must be in [0, 1] (got \(matrix[i][j]) at [\(i)][\(j)])",
                )
            }
            let rowSum = matrix[i][0] + matrix[i][1]
            precondition(
                abs(rowSum - 1.0) < 1e-10,
                "Confusion matrix rows must sum to 1.0 (row \(i) sums to \(rowSum))",
            )
        }
    }

    /// Validate that Kraus operators satisfy completeness relation Σᵢ Kᵢ†Kᵢ = I
    ///
    /// This is the fundamental constraint for quantum channels to preserve trace.
    ///
    /// - Parameters:
    ///   - krausOperators: Array of Kraus operator matrices
    ///   - tolerance: Numerical tolerance for identity check (default 1e-10)
    /// - Precondition: Σᵢ Kᵢ†Kᵢ = I within tolerance
    /// - Complexity: O(k * n³) where k = number of Kraus operators, n = matrix dimension
    @_effects(readonly)
    static func validateKrausCompleteness(_ krausOperators: [[[Complex<Double>]]], tolerance: Double = 1e-10) {
        precondition(!krausOperators.isEmpty, "Kraus operators array must not be empty")

        let n = krausOperators[0].count
        var sum = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: n), count: n)

        for kraus in krausOperators {
            let kDagger = MatrixUtilities.hermitianConjugate(kraus)
            let product = MatrixUtilities.matrixMultiply(kDagger, kraus)
            for i in 0 ..< n {
                for j in 0 ..< n {
                    sum[i][j] = sum[i][j] + product[i][j]
                }
            }
        }

        for i in 0 ..< n {
            for j in 0 ..< n {
                let expected: Complex<Double> = (i == j) ? .one : .zero
                let diff = sum[i][j] - expected
                precondition(
                    diff.magnitudeSquared < tolerance * tolerance,
                    "Kraus operators violate completeness relation at [\(i)][\(j)]: expected \(expected), got \(sum[i][j])",
                )
            }
        }
    }

    // MARK: - Qubit Count Validations

    /// Validate that two qubit counts are equal.
    ///
    /// - Parameters:
    ///   - count1: First qubit count
    ///   - count2: Second qubit count
    ///   - name1: Description of first count
    ///   - name2: Description of second count
    /// - Precondition: count1 == count2
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateQubitCountsEqual(_ count1: Int, _ count2: Int, name1: String, name2: String) {
        precondition(
            count1 == count2,
            "\(name1) (\(count1)) must equal \(name2) (\(count2))",
        )
    }

    /// Validate that qubit count is within a limit.
    ///
    /// - Parameters:
    ///   - count: Qubit count to validate
    ///   - limit: Maximum allowed count
    ///   - name: Description of the count
    ///   - limitName: Description of the limit
    /// - Precondition: count <= limit
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateQubitCountWithinLimit(_ count: Int, limit: Int, name: String, limitName: String) {
        precondition(
            count <= limit,
            "\(name) (\(count)) must not exceed \(limitName) (\(limit))",
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
    static func validateAllocationContainsIndex(_ allocation: [Int: Int], index: Int) {
        precondition(allocation[index] != nil, "Allocation must contain entry for term index \(index)")
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
            "Invalid coupling key '\(key)': must specify 1-2 qubits (e.g., '0' for local field, '01' or '0-1' for coupling)",
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
            "\(name) length must be \(expected) (got \(actual))",
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

    /// Validate custom gate matrix: size (2x2 or 4x4) and unitarity
    ///
    /// Validates that a custom gate matrix is either 2x2 (single-qubit) or 4x4 (two-qubit)
    /// and satisfies the unitary condition U†U = I. Combines size and unitarity checks
    /// for ``QuantumGate/custom(matrix:)`` factory method.
    ///
    /// - Parameter matrix: Matrix to validate
    /// - Precondition: Matrix must be 2x2 or 4x4, square, and unitary
    /// - Complexity: O(n³) for unitarity check where n ∈ {2, 4}
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateCustomGateMatrix(_ matrix: [[Complex<Double>]]) {
        let size = matrix.count
        precondition(
            size == 2 || size == 4,
            "Custom gate matrix must be 2x2 (single-qubit) or 4x4 (two-qubit), got \(size)x\(size)",
        )

        if size == 2 {
            validate2x2Matrix(matrix)
        } else {
            validate4x4Matrix(matrix)
        }

        validateUnitary(matrix)
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
            "Parameter '\(parameterName)' must have a binding in baseBindings",
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
        parameterSet: Set<String>,
    ) {
        for param in parameters {
            precondition(
                bindings[param.name] != nil,
                "Missing binding for parameter '\(param.name)'",
            )
        }
        for key in bindings.keys {
            precondition(
                parameterSet.contains(key),
                "Extra parameter '\(key)' not in circuit",
            )
        }
    }

    /// Validate that parameter value is concrete (not symbolic)
    ///
    /// Ensures ``ParameterValue`` is .value case before matrix generation or gate application.
    /// Symbolic parameters must be bound via ``QuantumCircuit/binding(_:)`` before execution.
    ///
    /// **Example:**
    /// ```swift
    /// let theta = Parameter(name: "theta")
    /// let symbolic = ParameterValue.parameter(theta)
    /// let concrete = ParameterValue.value(.pi / 4)
    ///
    /// ValidationUtilities.validateConcrete(concrete, name: "rotation angle")
    /// ValidationUtilities.validateConcrete(symbolic, name: "rotation angle")
    /// ```
    ///
    /// - Parameters:
    ///   - value: Parameter value to validate
    ///   - name: Descriptive name for error message
    /// - Precondition: value must be .value case (not .parameter)
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateConcrete(_ value: ParameterValue, name: String) {
        if case let .parameter(param) = value {
            preconditionFailure(
                "\(name) must be concrete before matrix generation (symbolic parameter '\(param.name)' requires binding)",
            )
        }
    }

    /// Validate that circuit contains only concrete parameters (no symbolic)
    ///
    /// Ensures circuit can be executed directly without requiring parameter binding.
    /// Circuits with symbolic ``Parameter`` instances must use ``QuantumCircuit/binding(_:)``
    /// or ``QuantumCircuit/bound(with:)`` before execution.
    ///
    /// - Parameter parameterCount: Number of symbolic parameters in circuit
    /// - Precondition: parameterCount must be 0
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateConcreteCircuit(_ parameterCount: Int) {
        precondition(
            parameterCount == 0,
            "Cannot execute circuit with symbolic parameters. Use binding(_:) or bound(with:) first.",
        )
    }

    // MARK: - Error Mitigation Validations

    /// Validate ZNE scale factors array.
    ///
    /// - Parameter scaleFactors: Array of noise scale factors
    /// - Precondition: Array has at least 2 elements, contains 1.0, and all values are positive
    /// - Complexity: O(n)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateZNEScaleFactors(_ scaleFactors: [Double]) {
        precondition(scaleFactors.count >= 2, "ZNE requires at least 2 scale factors")
        precondition(scaleFactors.contains(1.0), "Scale factors must include 1.0 (unscaled)")
        precondition(scaleFactors.allSatisfy { $0 > 0 }, "All scale factors must be positive")
    }

    /// Validates PEC error probability is in valid range.
    ///
    /// - Parameter probability: Error probability to validate
    /// - Precondition: 0 ≤ probability < 0.75 (depolarizing channel invertible limit)
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validatePECErrorProbability(_ probability: Double) {
        precondition(probability >= 0 && probability < 0.75,
                     "PEC error probability must be in [0, 0.75)")
    }

    // MARK: - MPS Tensor Validations

    /// Validate that MPS tensor element count matches bond dimensions.
    ///
    /// For a rank-3 tensor A[alpha,i,beta], the total element count must equal
    /// leftBondDimension * 2 * rightBondDimension (physical dimension is always 2 for qubits).
    ///
    /// - Parameters:
    ///   - elements: Flattened tensor elements
    ///   - leftBond: Left bond dimension
    ///   - rightBond: Right bond dimension
    /// - Precondition: elements.count == leftBond * 2 * rightBond
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateMPSTensorElementCount(_ elements: [Complex<Double>], leftBond: Int, rightBond: Int) {
        let expectedCount = leftBond * 2 * rightBond
        precondition(
            elements.count == expectedCount,
            "MPS tensor element count must be \(leftBond) * 2 * \(rightBond) = \(expectedCount) (got \(elements.count))",
        )
    }

    /// Validate that qubit count is within MPS limits.
    ///
    /// MPS supports up to 1000 qubits efficiently due to O(n * chi^2) memory scaling.
    ///
    /// - Parameter qubits: Number of qubits to validate
    /// - Precondition: 1 <= qubits <= 1000
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateMPSQubitCount(_ qubits: Int) {
        precondition(
            qubits >= 1 && qubits <= 1000,
            "MPS qubit count must be in range 1...1000 (got \(qubits))",
        )
    }

    /// Validate that site index is within MPS bounds.
    ///
    /// - Parameters:
    ///   - site: Site index to validate
    ///   - qubits: Total number of qubits in the MPS
    /// - Precondition: 0 <= site < qubits
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateMPSSiteIndex(_ site: Int, qubits: Int) {
        precondition(
            site >= 0 && site < qubits,
            "MPS site index \(site) out of bounds (valid range: 0..<\(qubits))",
        )
    }

    /// Validate that qubit count is within limit for MPS to statevector conversion.
    ///
    /// Converting MPS to full statevector requires O(2^n) memory, limiting practical
    /// conversion to small systems.
    ///
    /// - Parameter qubits: Number of qubits to validate
    /// - Precondition: qubits <= 20
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateMPSToStatevectorLimit(_ qubits: Int) {
        precondition(
            qubits <= 20,
            "MPS to statevector conversion limited to 20 qubits (got \(qubits))",
        )
    }

    /// Validate LAPACK operation succeeded (info == 0).
    ///
    /// LAPACK routines return info=0 on success, negative values for invalid arguments,
    /// and positive values for algorithm-specific failures (e.g., convergence issues).
    ///
    /// - Parameters:
    ///   - info: LAPACK return code to validate
    ///   - operation: Description of the LAPACK operation for error message
    /// - Precondition: info == 0
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateLAPACKSuccess(_ info: Int32, operation: String) {
        precondition(info == 0, "LAPACK \(operation) failed with info=\(info)")
    }

    /// Validate adjacent MPS tensor bond dimensions match.
    ///
    /// For MPS tensor contraction, the right bond dimension of the left tensor must
    /// equal the left bond dimension of the right tensor to ensure valid contraction.
    ///
    /// - Parameters:
    ///   - leftRight: Right bond dimension of left tensor
    ///   - rightLeft: Left bond dimension of right tensor
    /// - Precondition: leftRight == rightLeft
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateBondDimensionMatch(_ leftRight: Int, _ rightLeft: Int) {
        precondition(
            leftRight == rightLeft,
            "Adjacent tensor bond dimensions must match (got \(leftRight) and \(rightLeft))",
        )
    }

    /// Validate that multiplexor unitaries array is not empty.
    ///
    /// Multiplexor gates require at least one unitary matrix in the unitaries array
    /// to determine dimensions and construct the block-diagonal matrix.
    ///
    /// - Parameter unitaries: Array of unitary matrices for multiplexor
    /// - Precondition: !unitaries.isEmpty
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateMultiplexorNotEmpty(_ unitaries: [[[Complex<Double>]]]) {
        precondition(!unitaries.isEmpty, "Multiplexor unitaries array must not be empty")
    }

    // MARK: - Subsystem Validations

    /// Validate that two subsystem qubit arrays have no common elements
    ///
    /// Ensures subsystem A and subsystem B are disjoint, which is required for
    /// operations on independent subsystems such as partial trace or bipartite entanglement.
    ///
    /// - Parameters:
    ///   - subsystemA: First subsystem qubit indices
    ///   - subsystemB: Second subsystem qubit indices
    /// - Precondition: subsystemA and subsystemB must have no common elements
    /// - Complexity: O(n + m) where n = subsystemA.count, m = subsystemB.count
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateDisjointSubsystems(_ subsystemA: [Int], _ subsystemB: [Int]) {
        let setA = Set(subsystemA)
        let overlap = subsystemB.filter { setA.contains($0) }
        precondition(
            overlap.isEmpty,
            "Subsystems must be disjoint (overlapping qubit indices: \(overlap))",
        )
    }

    /// Validate that subsystem qubit indices form a proper subsystem
    ///
    /// Ensures all qubit indices are valid, the subsystem is non-empty, and at least one
    /// qubit remains in the complement (subsystem does not cover all qubits).
    ///
    /// - Parameters:
    ///   - subsystemQubits: Qubit indices defining the subsystem
    ///   - totalQubits: Total number of qubits in the full system
    /// - Precondition: subsystemQubits is non-empty
    /// - Precondition: All indices must be in [0, totalQubits)
    /// - Precondition: subsystemQubits.count < totalQubits
    /// - Complexity: O(n) where n = subsystemQubits.count
    @_effects(readonly)
    @inlinable
    @inline(__always)
    static func validateProperSubsystem(_ subsystemQubits: [Int], totalQubits: Int) {
        precondition(
            !subsystemQubits.isEmpty,
            "Subsystem must be non-empty",
        )
        precondition(
            subsystemQubits.allSatisfy { $0 >= 0 && $0 < totalQubits },
            "All subsystem qubit indices must be in range 0..<\(totalQubits) (got \(subsystemQubits))",
        )
        precondition(
            subsystemQubits.count < totalQubits,
            "Subsystem must not cover all qubits (got \(subsystemQubits.count) of \(totalQubits) qubits, at least one must remain in complement)",
        )
    }
}
