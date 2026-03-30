// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Rank-4 tensor W[alpha,s,s',beta] for Matrix Product Operator representation.
///
/// Each tensor in an MPO chain represents a local operator with four indices: left bond index (alpha),
/// physical input index (s = 0 or 1 for qubit basis states), physical output index (s' = 0 or 1),
/// and right bond index (beta). Elements are stored in flattened row-major order for efficient operations.
/// The flattened index is: `alpha * (d * d * rightBondDimension) + physIn * (d * rightBondDimension) + physOut * rightBondDimension + beta`
/// where d = physicalDimension (default 2 for qubits).
///
/// MPO tensors represent quantum operators (Hamiltonians, observables, channels) in tensor network form.
/// For local operators like single-site Pauli matrices, bond dimensions are 1. For long-range interactions,
/// bond dimensions grow to capture operator structure. Boundary conditions: first tensor has leftBondDimension=1,
/// last tensor has rightBondDimension=1.
///
/// **Example:**
/// ```swift
/// let elements: [Complex<Double>] = [.one, .zero, .zero, .one]
/// let identity = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements)
/// let element = identity[0, 0, 0, 0]
/// ```
///
/// - SeeAlso: ``MPSTensor`` for rank-3 state tensors
/// - SeeAlso: ``MatrixProductState`` for MPS representation of quantum states
@frozen
public struct MPOTensor: Sendable, Equatable {
    /// Left bond dimension (alpha index range: 0..<leftBondDimension)
    ///
    /// For the first tensor in the chain (site 0), leftBondDimension is always 1.
    /// For interior tensors, it equals the right bond dimension of the previous tensor.
    ///
    /// **Example:**
    /// ```swift
    /// let elements: [Complex<Double>] = [.one, .zero, .zero, .one]
    /// let tensor = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements)
    /// print(tensor.leftBondDimension)
    /// ```
    public let leftBondDimension: Int

    /// Right bond dimension (beta index range: 0..<rightBondDimension)
    ///
    /// For the last tensor in the chain (site n-1), rightBondDimension is always 1.
    /// For interior tensors, it captures operator structure across subsequent sites.
    ///
    /// **Example:**
    /// ```swift
    /// let elements: [Complex<Double>] = [.one, .zero, .zero, .one]
    /// let tensor = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements)
    /// print(tensor.rightBondDimension)
    /// ```
    public let rightBondDimension: Int

    /// Physical dimension for qubit basis (default 2: |0> and |1>)
    ///
    /// For qubit systems, this is always 2 corresponding to computational basis states.
    /// Both input and output physical indices range over [0, physicalDimension).
    ///
    /// **Example:**
    /// ```swift
    /// let elements: [Complex<Double>] = [.one, .zero, .zero, .one]
    /// let tensor = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements)
    /// print(tensor.physicalDimension)
    /// ```
    public let physicalDimension: Int

    /// Flattened tensor elements in row-major order
    ///
    /// Index mapping: `elements[alpha * (d * d * rightBond) + physIn * (d * rightBond) + physOut * rightBond + beta]`
    /// where alpha in [0, leftBondDimension), physIn in [0, d), physOut in [0, d), beta in [0, rightBondDimension),
    /// and d = physicalDimension.
    ///
    /// **Example:**
    /// ```swift
    /// let elements: [Complex<Double>] = [.one, .zero, .zero, .one]
    /// let tensor = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements)
    /// print(tensor.elements.count)
    /// ```
    public let elements: [Complex<Double>]

    /// Creates an MPO tensor with specified bond dimensions and elements.
    ///
    /// Validates that element count matches the product of all dimensions:
    /// leftBondDimension * physicalDimension * physicalDimension * rightBondDimension.
    /// Physical dimension defaults to 2 for qubit systems.
    ///
    /// **Example:**
    /// ```swift
    /// let elements: [Complex<Double>] = [.one, .zero, .zero, .one]
    /// let tensor = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements)
    /// ```
    ///
    /// - Parameters:
    ///   - leftBondDimension: Size of left bond index (must be positive)
    ///   - rightBondDimension: Size of right bond index (must be positive)
    ///   - elements: Flattened tensor elements (count must equal leftBond * d * d * rightBond where d=2)
    /// - Complexity: O(1)
    /// - Precondition: leftBondDimension > 0
    /// - Precondition: rightBondDimension > 0
    /// - Precondition: elements.count == leftBondDimension * 4 * rightBondDimension
    public init(leftBondDimension: Int, rightBondDimension: Int, elements: [Complex<Double>]) {
        ValidationUtilities.validatePositiveInt(leftBondDimension, name: "Left bond dimension")
        ValidationUtilities.validatePositiveInt(rightBondDimension, name: "Right bond dimension")

        ValidationUtilities.validateMPOTensorElementCount(elements, leftBond: leftBondDimension, rightBond: rightBondDimension)

        self.leftBondDimension = leftBondDimension
        self.rightBondDimension = rightBondDimension
        physicalDimension = 2
        self.elements = elements
    }

    /// Accesses tensor element W[left, physIn, physOut, right].
    ///
    /// Provides direct access to tensor elements using natural four-index notation.
    /// Maps to flattened storage via: left * (d * d * rightBond) + physIn * (d * rightBond) + physOut * rightBond + right.
    ///
    /// **Example:**
    /// ```swift
    /// let elements: [Complex<Double>] = [.one, .zero, .zero, .one]
    /// let tensor = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements)
    /// let amplitude = tensor[0, 0, 0, 0]
    /// ```
    ///
    /// - Parameters:
    ///   - left: Left bond index (0 to leftBondDimension-1)
    ///   - physIn: Physical input index (0 to physicalDimension-1)
    ///   - physOut: Physical output index (0 to physicalDimension-1)
    ///   - right: Right bond index (0 to rightBondDimension-1)
    /// - Returns: Complex amplitude W[left, physIn, physOut, right]
    /// - Complexity: O(1)
    /// - Precondition: 0 <= left < leftBondDimension
    /// - Precondition: 0 <= physIn < physicalDimension
    /// - Precondition: 0 <= physOut < physicalDimension
    /// - Precondition: 0 <= right < rightBondDimension
    @inlinable
    public subscript(left: Int, physIn: Int, physOut: Int, right: Int) -> Complex<Double> {
        ValidationUtilities.validateIndexInBounds(left, bound: leftBondDimension, name: "MPO left bond index")
        ValidationUtilities.validateIndexInBounds(physIn, bound: physicalDimension, name: "MPO physical input index")
        ValidationUtilities.validateIndexInBounds(physOut, bound: physicalDimension, name: "MPO physical output index")
        ValidationUtilities.validateIndexInBounds(right, bound: rightBondDimension, name: "MPO right bond index")

        let d = physicalDimension
        let flatIndex = left * (d * d * rightBondDimension) + physIn * (d * rightBondDimension) + physOut * rightBondDimension + right
        return elements[flatIndex]
    }
}
