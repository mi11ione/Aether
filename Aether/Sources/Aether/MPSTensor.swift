// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Rank-3 tensor A[alpha,i,beta] for Matrix Product State representation of quantum states.
///
/// Each tensor in an MPS chain represents one qubit with three indices: left bond index (alpha),
/// physical index (i = 0 or 1 for qubit basis states), and right bond index (beta). Elements are
/// stored in flattened row-major order for efficient BLAS operations. The flattened index is:
/// `alpha * (2 * rightBondDimension) + physical * rightBondDimension + beta`.
///
/// For product states like |00...0⟩, bond dimensions are 1 and only one tensor element is nonzero.
/// For entangled states, bond dimensions grow (up to maxBondDimension) to capture quantum correlations.
/// Boundary conditions: first tensor has leftBondDimension=1, last tensor has rightBondDimension=1.
///
/// **Example:**
/// ```swift
/// let tensor = MPSTensor.groundState(site: 0, qubits: 4, maxBondDimension: 16)
/// let element = tensor[0, 0, 0]
/// let leftContracted = tensor.contractLeft(with: [Complex<Double>.one])
/// ```
///
/// - SeeAlso: ``QuantumState`` for full state vector representation
@frozen
public struct MPSTensor: Sendable, Equatable {
    /// Left bond dimension (alpha index range: 0..<leftBondDimension)
    ///
    /// For the first tensor in the chain (site 0), leftBondDimension is always 1.
    /// For interior tensors, it equals the right bond dimension of the previous tensor.
    ///
    /// **Example:**
    /// ```swift
    /// let tensor = MPSTensor.groundState(site: 0, qubits: 4, maxBondDimension: 16)
    /// print(tensor.leftBondDimension)  // 1 (boundary tensor)
    /// ```
    public let leftBondDimension: Int

    /// Physical dimension for qubit basis (always 2: |0⟩ and |1⟩)
    ///
    /// For qubit systems, this is always 2 corresponding to computational basis states.
    ///
    /// **Example:**
    /// ```swift
    /// let tensor = MPSTensor.groundState(site: 0, qubits: 4, maxBondDimension: 16)
    /// print(tensor.physicalDimension)  // 2
    /// ```
    public let physicalDimension: Int

    /// Right bond dimension (beta index range: 0..<rightBondDimension)
    ///
    /// For the last tensor in the chain (site n-1), rightBondDimension is always 1.
    /// For interior tensors, it captures entanglement with subsequent qubits.
    ///
    /// **Example:**
    /// ```swift
    /// let tensor = MPSTensor.groundState(site: 3, qubits: 4, maxBondDimension: 16)
    /// print(tensor.rightBondDimension)  // 1 (boundary tensor)
    /// ```
    public let rightBondDimension: Int

    /// Flattened tensor elements in row-major order
    ///
    /// Index mapping: `elements[alpha * (2 * rightBondDimension) + physical * rightBondDimension + beta]`
    /// where alpha in [0, leftBondDimension), physical in [0, 2), beta in [0, rightBondDimension).
    ///
    /// **Example:**
    /// ```swift
    /// let tensor = MPSTensor.groundState(site: 0, qubits: 4, maxBondDimension: 16)
    /// print(tensor.elements.count)  // leftBondDimension * 2 * rightBondDimension
    /// ```
    public let elements: [Complex<Double>]

    /// Site index within the MPS chain (0 to qubits-1)
    ///
    /// Identifies which qubit this tensor represents in the chain.
    ///
    /// **Example:**
    /// ```swift
    /// let tensor = MPSTensor.groundState(site: 2, qubits: 4, maxBondDimension: 16)
    /// print(tensor.site)  // 2
    /// ```
    public let site: Int

    /// Creates an MPS tensor with specified bond dimensions and elements.
    ///
    /// Validates that element count matches the product of all dimensions.
    ///
    /// **Example:**
    /// ```swift
    /// let elements: [Complex<Double>] = [.one, .zero]
    /// let tensor = MPSTensor(leftBondDimension: 1, rightBondDimension: 1, site: 0, elements: elements)
    /// ```
    ///
    /// - Parameters:
    ///   - leftBondDimension: Size of left bond index (must be positive)
    ///   - rightBondDimension: Size of right bond index (must be positive)
    ///   - site: Site index in MPS chain (must be non-negative)
    ///   - elements: Flattened tensor elements (count must equal leftBondDimension * 2 * rightBondDimension)
    /// - Complexity: O(1)
    /// - Precondition: Bond dimensions must be positive, elements count must match dimensions
    /// - SeeAlso: ``groundState(site:qubits:maxBondDimension:)``
    /// - SeeAlso: ``basisState(_:site:qubits:maxBondDimension:)``
    public init(leftBondDimension: Int, rightBondDimension: Int, site: Int, elements: [Complex<Double>]) {
        ValidationUtilities.validatePositiveInt(leftBondDimension, name: "Left bond dimension")
        ValidationUtilities.validatePositiveInt(rightBondDimension, name: "Right bond dimension")
        ValidationUtilities.validateNonNegativeInt(site, name: "Site index")
        ValidationUtilities.validateMPSTensorElementCount(elements, leftBond: leftBondDimension, rightBond: rightBondDimension)

        self.leftBondDimension = leftBondDimension
        physicalDimension = 2
        self.rightBondDimension = rightBondDimension
        self.elements = elements
        self.site = site
    }

    /// Creates an MPS tensor for the ground state |00...0⟩ at the given site.
    ///
    /// For ground state, only physical index 0 is populated: A[0,0,0] = 1, A[0,1,0] = 0.
    /// Product states have bond dimension 1, so maxBondDimension does not affect the result.
    ///
    /// **Example:**
    /// ```swift
    /// let tensor = MPSTensor.groundState(site: 0, qubits: 4, maxBondDimension: 16)
    /// print(tensor[0, 0, 0])  // (1.0, 0.0)
    /// print(tensor[0, 1, 0])  // (0.0, 0.0)
    /// ```
    ///
    /// - Parameters:
    ///   - site: Site index (0 to qubits-1)
    ///   - qubits: Total number of qubits in the system
    ///   - maxBondDimension: Maximum allowed bond dimension (unused for product states)
    /// - Returns: MPS tensor representing |0⟩ at this site
    /// - Complexity: O(1)
    /// - Precondition: 0 <= site < qubits, qubits > 0
    /// - SeeAlso: ``basisState(_:site:qubits:maxBondDimension:)``
    @_eagerMove
    public static func groundState(site: Int, qubits: Int, maxBondDimension: Int) -> MPSTensor {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateIndexInBounds(site, bound: qubits, name: "Site index")
        ValidationUtilities.validatePositiveInt(maxBondDimension, name: "Max bond dimension")

        let elements: [Complex<Double>] = [.one, .zero]
        return MPSTensor(leftBondDimension: 1, rightBondDimension: 1, site: site, elements: elements)
    }

    /// Creates an MPS tensor for computational basis state |basisState⟩ at the given site.
    ///
    /// Extracts the bit at the specified site from the basis state integer and creates
    /// a tensor with A[0,bit,0] = 1. Uses little-endian bit ordering where site 0 is the
    /// least significant bit.
    ///
    /// **Example:**
    /// ```swift
    /// let tensor = MPSTensor.basisState(0b1010, site: 1, qubits: 4, maxBondDimension: 16)
    /// print(tensor[0, 1, 0])  // (1.0, 0.0) - bit 1 of 0b1010 is 1
    /// print(tensor[0, 0, 0])  // (0.0, 0.0)
    /// ```
    ///
    /// - Parameters:
    ///   - basisState: Integer encoding the computational basis state (little-endian)
    ///   - site: Site index (0 to qubits-1)
    ///   - qubits: Total number of qubits in the system
    ///   - maxBondDimension: Maximum allowed bond dimension (unused for product states)
    /// - Returns: MPS tensor with the appropriate physical index populated
    /// - Complexity: O(1)
    /// - Precondition: 0 <= site < qubits, qubits > 0, 0 <= basisState < 2^qubits
    /// - SeeAlso: ``groundState(site:qubits:maxBondDimension:)``
    @_eagerMove
    public static func basisState(_ basisState: Int, site: Int, qubits: Int, maxBondDimension: Int) -> MPSTensor {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateIndexInBounds(site, bound: qubits, name: "Site index")
        ValidationUtilities.validatePositiveInt(maxBondDimension, name: "Max bond dimension")
        ValidationUtilities.validateIndexInBounds(basisState, bound: 1 << qubits, name: "Basis state")

        let bit = (basisState >> site) & 1
        let elements: [Complex<Double>] = bit == 0 ? [.one, .zero] : [.zero, .one]
        return MPSTensor(leftBondDimension: 1, rightBondDimension: 1, site: site, elements: elements)
    }

    /// Accesses tensor element A[left, physical, right].
    ///
    /// Provides direct access to tensor elements using natural three-index notation.
    /// Maps to flattened storage via: left * (2 * rightBondDimension) + physical * rightBondDimension + right.
    ///
    /// **Example:**
    /// ```swift
    /// let tensor = MPSTensor.groundState(site: 0, qubits: 4, maxBondDimension: 16)
    /// let amplitude = tensor[0, 0, 0]  // (1.0, 0.0)
    /// ```
    ///
    /// - Parameters:
    ///   - left: Left bond index (0 to leftBondDimension-1)
    ///   - physical: Physical index (0 or 1)
    ///   - right: Right bond index (0 to rightBondDimension-1)
    /// - Returns: Complex amplitude A[left, physical, right]
    /// - Complexity: O(1)
    /// - Precondition: All indices must be within valid ranges
    /// - SeeAlso: ``elements``
    /// - SeeAlso: ``matrix(forPhysical:)``
    @inlinable
    public subscript(left: Int, physical: Int, right: Int) -> Complex<Double> {
        ValidationUtilities.validateIndexInBounds(left, bound: leftBondDimension, name: "Left bond index")
        ValidationUtilities.validateIndexInBounds(physical, bound: physicalDimension, name: "Physical index")
        ValidationUtilities.validateIndexInBounds(right, bound: rightBondDimension, name: "Right bond index")

        let flatIndex = left * (physicalDimension * rightBondDimension) + physical * rightBondDimension + right
        return elements[flatIndex]
    }

    /// Contracts the tensor with a left vector: result[i,beta] = Sum_alpha v[alpha] * A[alpha,i,beta].
    ///
    /// Performs left contraction to propagate MPS computation from left to right.
    /// Vectorized for large tensors.
    ///
    /// **Example:**
    /// ```swift
    /// let tensor = MPSTensor.groundState(site: 1, qubits: 4, maxBondDimension: 16)
    /// let leftVec: [Complex<Double>] = [.one]
    /// let result = tensor.contractLeft(with: leftVec)
    /// ```
    ///
    /// - Parameter leftVector: Vector of size leftBondDimension
    /// - Returns: 2D array [physical][rightBond] of contracted values
    /// - Complexity: O(leftBond * physicalDim * rightBond)
    /// - Precondition: leftVector.count == leftBondDimension
    /// - SeeAlso: ``contractRight(with:)``
    @_effects(readonly)
    @_optimize(speed)
    @_eagerMove
    public func contractLeft(with leftVector: [Complex<Double>]) -> [[Complex<Double>]] {
        ValidationUtilities.validateArrayCount(leftVector, expected: leftBondDimension, name: "Left vector")

        let totalElements = leftBondDimension * physicalDimension * rightBondDimension

        if totalElements >= 64 {
            return contractLeftBLAS(with: leftVector)
        }

        return contractLeftScalar(with: leftVector)
    }

    /// Scalar-loop left contraction for small tensors.
    @_effects(readonly)
    @_optimize(speed)
    @_eagerMove
    private func contractLeftScalar(with leftVector: [Complex<Double>]) -> [[Complex<Double>]] {
        (0 ..< physicalDimension).map { physical in
            [Complex<Double>](unsafeUninitializedCapacity: rightBondDimension) { buffer, count in
                for beta in 0 ..< rightBondDimension {
                    var sum: Complex<Double> = .zero
                    for alpha in 0 ..< leftBondDimension {
                        let flatIndex = alpha * (physicalDimension * rightBondDimension) + physical * rightBondDimension + beta
                        sum = sum + leftVector[alpha] * elements[flatIndex]
                    }
                    buffer[beta] = sum
                }
                count = rightBondDimension
            }
        }
    }

    /// BLAS-accelerated left contraction for large tensors.
    @_effects(readonly)
    @_optimize(speed)
    @_eagerMove
    private func contractLeftBLAS(with leftVector: [Complex<Double>]) -> [[Complex<Double>]] {
        let m = physicalDimension * rightBondDimension
        let n = leftBondDimension

        var matrixInterleaved = [Double](unsafeUninitializedCapacity: m * n * 2) { buffer, count in
            for alpha in 0 ..< n {
                for physical in 0 ..< physicalDimension {
                    for beta in 0 ..< rightBondDimension {
                        let flatIndex = alpha * (physicalDimension * rightBondDimension) + physical * rightBondDimension + beta
                        let colIndex = physical * rightBondDimension + beta
                        let idx = (colIndex * n + alpha) * 2
                        buffer[idx] = elements[flatIndex].real
                        buffer[idx + 1] = elements[flatIndex].imaginary
                    }
                }
            }
            count = m * n * 2
        }

        var vecInterleaved = [Double](unsafeUninitializedCapacity: n * 2) { buffer, count in
            for i in 0 ..< n {
                buffer[i * 2] = leftVector[i].real
                buffer[i * 2 + 1] = leftVector[i].imaginary
            }
            count = n * 2
        }

        var resultInterleaved = [Double](unsafeUninitializedCapacity: m * 2) { _, count in
            count = m * 2
        }

        var alpha = (1.0, 0.0)
        var beta = (0.0, 0.0)

        matrixInterleaved.withUnsafeMutableBufferPointer { matPtr in
            vecInterleaved.withUnsafeMutableBufferPointer { vecPtr in
                resultInterleaved.withUnsafeMutableBufferPointer { resPtr in
                    withUnsafePointer(to: &alpha) { alphaPtr in
                        withUnsafePointer(to: &beta) { betaPtr in
                            cblas_zgemv(
                                CblasColMajor,
                                CblasNoTrans,
                                Int32(m),
                                Int32(n),
                                OpaquePointer(alphaPtr),
                                OpaquePointer(matPtr.baseAddress),
                                Int32(m),
                                OpaquePointer(vecPtr.baseAddress),
                                1,
                                OpaquePointer(betaPtr),
                                OpaquePointer(resPtr.baseAddress),
                                1,
                            )
                        }
                    }
                }
            }
        }

        let result = (0 ..< physicalDimension).map { physical in
            [Complex<Double>](unsafeUninitializedCapacity: rightBondDimension) { buffer, count in
                for betaIdx in 0 ..< rightBondDimension {
                    let idx = (physical * rightBondDimension + betaIdx) * 2
                    buffer[betaIdx] = Complex(resultInterleaved[idx], resultInterleaved[idx + 1])
                }
                count = rightBondDimension
            }
        }

        return result
    }

    /// Contracts the tensor with a right vector: result[alpha,i] = Sum_beta A[alpha,i,beta] * v[beta].
    ///
    /// Performs right contraction to propagate MPS computation from right to left.
    /// Vectorized for large tensors.
    ///
    /// **Example:**
    /// ```swift
    /// let tensor = MPSTensor.groundState(site: 2, qubits: 4, maxBondDimension: 16)
    /// let rightVec: [Complex<Double>] = [.one]
    /// let result = tensor.contractRight(with: rightVec)
    /// ```
    ///
    /// - Parameter rightVector: Vector of size rightBondDimension
    /// - Returns: 2D array [leftBond][physical] of contracted values
    /// - Complexity: O(leftBond * physicalDim * rightBond)
    /// - Precondition: rightVector.count == rightBondDimension
    /// - SeeAlso: ``contractLeft(with:)``
    @_effects(readonly)
    @_optimize(speed)
    @_eagerMove
    public func contractRight(with rightVector: [Complex<Double>]) -> [[Complex<Double>]] {
        ValidationUtilities.validateArrayCount(rightVector, expected: rightBondDimension, name: "Right vector")

        let totalElements = leftBondDimension * physicalDimension * rightBondDimension

        if totalElements >= 64 {
            return contractRightBLAS(with: rightVector)
        }

        return contractRightScalar(with: rightVector)
    }

    /// Scalar-loop right contraction for small tensors.
    @_effects(readonly)
    @_optimize(speed)
    @_eagerMove
    private func contractRightScalar(with rightVector: [Complex<Double>]) -> [[Complex<Double>]] {
        (0 ..< leftBondDimension).map { alpha in
            [Complex<Double>](unsafeUninitializedCapacity: physicalDimension) { buffer, count in
                for physical in 0 ..< physicalDimension {
                    var sum: Complex<Double> = .zero
                    for beta in 0 ..< rightBondDimension {
                        let flatIndex = alpha * (physicalDimension * rightBondDimension) + physical * rightBondDimension + beta
                        sum = sum + elements[flatIndex] * rightVector[beta]
                    }
                    buffer[physical] = sum
                }
                count = physicalDimension
            }
        }
    }

    /// BLAS-accelerated right contraction for large tensors.
    @_effects(readonly)
    @_optimize(speed)
    @_eagerMove
    private func contractRightBLAS(with rightVector: [Complex<Double>]) -> [[Complex<Double>]] {
        let m = leftBondDimension * physicalDimension
        let n = rightBondDimension

        var matrixInterleaved = [Double](unsafeUninitializedCapacity: m * n * 2) { buffer, count in
            for alpha in 0 ..< leftBondDimension {
                for physical in 0 ..< physicalDimension {
                    let rowIndex = alpha * physicalDimension + physical
                    for beta in 0 ..< n {
                        let flatIndex = alpha * (physicalDimension * rightBondDimension) + physical * rightBondDimension + beta
                        let idx = (beta * m + rowIndex) * 2
                        buffer[idx] = elements[flatIndex].real
                        buffer[idx + 1] = elements[flatIndex].imaginary
                    }
                }
            }
            count = m * n * 2
        }

        var vecInterleaved = [Double](unsafeUninitializedCapacity: n * 2) { buffer, count in
            for i in 0 ..< n {
                buffer[i * 2] = rightVector[i].real
                buffer[i * 2 + 1] = rightVector[i].imaginary
            }
            count = n * 2
        }

        var resultInterleaved = [Double](unsafeUninitializedCapacity: m * 2) { _, count in
            count = m * 2
        }

        var alpha = (1.0, 0.0)
        var beta = (0.0, 0.0)

        matrixInterleaved.withUnsafeMutableBufferPointer { matPtr in
            vecInterleaved.withUnsafeMutableBufferPointer { vecPtr in
                resultInterleaved.withUnsafeMutableBufferPointer { resPtr in
                    withUnsafePointer(to: &alpha) { alphaPtr in
                        withUnsafePointer(to: &beta) { betaPtr in
                            cblas_zgemv(
                                CblasColMajor,
                                CblasNoTrans,
                                Int32(m),
                                Int32(n),
                                OpaquePointer(alphaPtr),
                                OpaquePointer(matPtr.baseAddress),
                                Int32(m),
                                OpaquePointer(vecPtr.baseAddress),
                                1,
                                OpaquePointer(betaPtr),
                                OpaquePointer(resPtr.baseAddress),
                                1,
                            )
                        }
                    }
                }
            }
        }

        let result = (0 ..< leftBondDimension).map { alphaIdx in
            [Complex<Double>](unsafeUninitializedCapacity: physicalDimension) { buffer, count in
                for physical in 0 ..< physicalDimension {
                    let idx = (alphaIdx * physicalDimension + physical) * 2
                    buffer[physical] = Complex(resultInterleaved[idx], resultInterleaved[idx + 1])
                }
                count = physicalDimension
            }
        }

        return result
    }

    /// Extracts the matrix A[alpha,beta] for a fixed physical index.
    ///
    /// Returns the leftBondDimension x rightBondDimension slice of the tensor
    /// corresponding to the specified physical basis state.
    ///
    /// **Example:**
    /// ```swift
    /// let tensor = MPSTensor.groundState(site: 0, qubits: 4, maxBondDimension: 16)
    /// let matrixZero = tensor.matrix(forPhysical: 0)
    /// let matrixOne = tensor.matrix(forPhysical: 1)
    /// ```
    ///
    /// - Parameter physicalIndex: Physical index (0 or 1)
    /// - Returns: 2D matrix [leftBond][rightBond] for the given physical index
    /// - Complexity: O(leftBond * rightBond)
    /// - Precondition: physicalIndex must be 0 or 1
    /// - SeeAlso: ``subscript(left:physical:right:)``
    /// - SeeAlso: ``reshapeForSVD(mergingLeft:)``
    @_effects(readonly)
    @_optimize(speed)
    @_eagerMove
    public func matrix(forPhysical physicalIndex: Int) -> [[Complex<Double>]] {
        ValidationUtilities.validateIndexInBounds(physicalIndex, bound: physicalDimension, name: "Physical index")

        return (0 ..< leftBondDimension).map { alpha in
            [Complex<Double>](unsafeUninitializedCapacity: rightBondDimension) { buffer, count in
                for beta in 0 ..< rightBondDimension {
                    let flatIndex = alpha * (physicalDimension * rightBondDimension) + physicalIndex * rightBondDimension + beta
                    buffer[beta] = elements[flatIndex]
                }
                count = rightBondDimension
            }
        }
    }

    /// Reshapes the tensor into a matrix for SVD decomposition.
    ///
    /// For mergingLeft=true: combines left bond and physical indices into rows,
    /// resulting in (leftBond * 2) x rightBond matrix.
    /// For mergingLeft=false: combines physical and right bond indices into columns,
    /// resulting in leftBond x (2 * rightBond) matrix.
    ///
    /// **Example:**
    /// ```swift
    /// let tensor = MPSTensor(leftBondDimension: 2, rightBondDimension: 3, site: 1, elements: elements)
    /// let leftMerged = tensor.reshapeForSVD(mergingLeft: true)   // 4x3 matrix
    /// let rightMerged = tensor.reshapeForSVD(mergingLeft: false) // 2x6 matrix
    /// ```
    ///
    /// - Parameter mergingLeft: If true, merge (alpha,i) into rows; if false, merge (i,beta) into columns
    /// - Returns: 2D matrix suitable for SVD decomposition
    /// - Complexity: O(leftBond * 2 * rightBond)
    /// - SeeAlso: ``matrix(forPhysical:)``
    @_effects(readonly)
    @_optimize(speed)
    @_eagerMove
    public func reshapeForSVD(mergingLeft: Bool) -> [[Complex<Double>]] {
        if mergingLeft {
            let rows = leftBondDimension * physicalDimension
            let cols = rightBondDimension

            return (0 ..< rows).map { rowIndex in
                let alpha = rowIndex / physicalDimension
                let physical = rowIndex % physicalDimension
                return [Complex<Double>](unsafeUninitializedCapacity: cols) { buffer, count in
                    for beta in 0 ..< cols {
                        let flatIndex = alpha * (physicalDimension * rightBondDimension) + physical * rightBondDimension + beta
                        buffer[beta] = elements[flatIndex]
                    }
                    count = cols
                }
            }
        } else {
            let rows = leftBondDimension
            let cols = physicalDimension * rightBondDimension

            return (0 ..< rows).map { alpha in
                [Complex<Double>](unsafeUninitializedCapacity: cols) { buffer, count in
                    for physical in 0 ..< physicalDimension {
                        for beta in 0 ..< rightBondDimension {
                            let colIndex = physical * rightBondDimension + beta
                            let flatIndex = alpha * (physicalDimension * rightBondDimension) + physical * rightBondDimension + beta
                            buffer[colIndex] = elements[flatIndex]
                        }
                    }
                    count = cols
                }
            }
        }
    }
}
