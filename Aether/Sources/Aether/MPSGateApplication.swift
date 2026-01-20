// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Gate application operations for Matrix Product States using tensor network methods.
///
/// Applies quantum gates to MPS representation with bond dimension management via SVD truncation.
/// Single-qubit gates preserve bond dimension by contracting gate matrix with physical index.
/// Two-qubit gates may increase bond dimension requiring SVD truncation to maintain efficiency.
/// Non-adjacent two-qubit gates use SWAP network to bring qubits adjacent before applying gate.
///
/// **Example:**
/// ```swift
/// var mps = MatrixProductState(qubits: 10, maxBondDimension: 32)
/// MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
/// MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 1, mps: &mps)
/// MPSGateApplication.applyToffoli(control1: 0, control2: 1, target: 2, mps: &mps)
/// ```
///
/// - SeeAlso: ``MatrixProductState``
/// - SeeAlso: ``MPSTensor``
/// - SeeAlso: ``SVDDecomposition``
public enum MPSGateApplication {
    /// Apply single-qubit gate to MPS site.
    ///
    /// Contracts gate 2x2 matrix with physical index of tensor at specified site.
    /// Bond dimensions remain unchanged since single-qubit gates do not increase entanglement.
    /// Computes A'[alpha,i',beta] = Sum_i U[i',i] * A[alpha,i,beta] for all bond indices.
    ///
    /// **Example:**
    /// ```swift
    /// var mps = MatrixProductState(qubits: 5, maxBondDimension: 16)
    /// MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
    /// MPSGateApplication.applySingleQubitGate(.rotationY(.pi / 4), to: 2, mps: &mps)
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Single-qubit quantum gate to apply
    ///   - site: Site index in MPS (0 to qubits-1)
    ///   - mps: Matrix Product State to modify
    /// - Complexity: O(chi^2) where chi is the bond dimension at the site
    /// - Precondition: gate must be single-qubit, site must be valid index
    @_optimize(speed)
    public static func applySingleQubitGate(
        _ gate: QuantumGate,
        to site: Int,
        mps: inout MatrixProductState,
    ) {
        ValidationUtilities.validateMPSSiteIndex(site, qubits: mps.qubits)

        let gateMatrix = gate.matrix()
        let u00 = gateMatrix[0][0]
        let u01 = gateMatrix[0][1]
        let u10 = gateMatrix[1][0]
        let u11 = gateMatrix[1][1]

        let tensor = mps.tensors[site]
        let leftDim = tensor.leftBondDimension
        let rightDim = tensor.rightBondDimension

        let newElements = [Complex<Double>](unsafeUninitializedCapacity: leftDim * 2 * rightDim) { buffer, count in
            for alpha in 0 ..< leftDim {
                for beta in 0 ..< rightDim {
                    let a0 = tensor[alpha, 0, beta]
                    let a1 = tensor[alpha, 1, beta]

                    let newA0 = u00 * a0 + u01 * a1
                    let newA1 = u10 * a0 + u11 * a1

                    let idx0 = alpha * (2 * rightDim) + 0 * rightDim + beta
                    let idx1 = alpha * (2 * rightDim) + 1 * rightDim + beta

                    buffer[idx0] = newA0
                    buffer[idx1] = newA1
                }
            }
            count = leftDim * 2 * rightDim
        }

        mps.updateTensor(at: site, with: MPSTensor(
            leftBondDimension: leftDim,
            rightBondDimension: rightDim,
            site: site,
            elements: newElements,
        ))
    }

    /// Apply two-qubit gate to MPS sites.
    ///
    /// For adjacent sites: contracts tensors, applies 4x4 gate matrix, performs SVD decomposition
    /// with truncation to maxBondDimension, and splits back into two tensors.
    /// For non-adjacent sites: uses SWAP network to bring qubits adjacent, applies gate, then
    /// SWAPs back to restore qubit ordering.
    ///
    /// **Example:**
    /// ```swift
    /// var mps = MatrixProductState(qubits: 5, maxBondDimension: 32)
    /// MPSGateApplication.applySingleQubitGate(.hadamard, to: 0, mps: &mps)
    /// MPSGateApplication.applyTwoQubitGate(.cnot, control: 0, target: 1, mps: &mps)
    /// MPSGateApplication.applyTwoQubitGate(.cz, control: 0, target: 3, mps: &mps)
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Two-qubit quantum gate to apply
    ///   - control: Control qubit index (first qubit for symmetric gates)
    ///   - target: Target qubit index (second qubit for symmetric gates)
    ///   - mps: Matrix Product State to modify
    /// - Complexity: O(chi^3) for adjacent sites, O(d * chi^3) for distance d apart
    /// - Precondition: gate must be two-qubit, control and target must be distinct valid indices
    @_optimize(speed)
    public static func applyTwoQubitGate(
        _ gate: QuantumGate,
        control: Int,
        target: Int,
        mps: inout MatrixProductState,
    ) {
        ValidationUtilities.validateMPSSiteIndex(control, qubits: mps.qubits)
        ValidationUtilities.validateMPSSiteIndex(target, qubits: mps.qubits)
        ValidationUtilities.validateUniqueQubits([control, target])

        let left = min(control, target)
        let right = max(control, target)
        let distance = right - left

        if distance == 1 {
            let controlIsLeft = control < target
            applyAdjacentTwoQubitGate(gate, leftSite: left, controlIsLeft: controlIsLeft, mps: &mps)
        } else {
            applyNonAdjacentTwoQubitGate(gate, control: control, target: target, mps: &mps)
        }
    }

    /// Apply Toffoli gate via decomposition into two-qubit gates.
    ///
    /// Decomposes Toffoli (CCNOT) into a sequence of CNOT, T, T-dagger, and Hadamard gates
    /// following the standard decomposition. This allows application to MPS without requiring
    /// three-site tensor contraction.
    ///
    /// **Example:**
    /// ```swift
    /// var mps = MatrixProductState(qubits: 5, maxBondDimension: 32)
    /// MPSGateApplication.applySingleQubitGate(.pauliX, to: 0, mps: &mps)
    /// MPSGateApplication.applySingleQubitGate(.pauliX, to: 1, mps: &mps)
    /// MPSGateApplication.applyToffoli(control1: 0, control2: 1, target: 2, mps: &mps)
    /// ```
    ///
    /// - Parameters:
    ///   - control1: First control qubit index
    ///   - control2: Second control qubit index
    ///   - target: Target qubit index
    ///   - mps: Matrix Product State to modify
    /// - Complexity: O(d * chi^3) where d is max distance between qubits
    /// - Precondition: All qubit indices must be distinct and valid
    @_optimize(speed)
    public static func applyToffoli(
        control1: Int,
        control2: Int,
        target: Int,
        mps: inout MatrixProductState,
    ) {
        ValidationUtilities.validateMPSSiteIndex(control1, qubits: mps.qubits)
        ValidationUtilities.validateMPSSiteIndex(control2, qubits: mps.qubits)
        ValidationUtilities.validateMPSSiteIndex(target, qubits: mps.qubits)
        ValidationUtilities.validateUniqueQubits([control1, control2, target])

        applySingleQubitGate(.hadamard, to: target, mps: &mps)
        applyTwoQubitGate(.cnot, control: control2, target: target, mps: &mps)
        applySingleQubitGate(.tGate.inverse, to: target, mps: &mps)
        applyTwoQubitGate(.cnot, control: control1, target: target, mps: &mps)
        applySingleQubitGate(.tGate, to: target, mps: &mps)
        applyTwoQubitGate(.cnot, control: control2, target: target, mps: &mps)
        applySingleQubitGate(.tGate.inverse, to: target, mps: &mps)
        applyTwoQubitGate(.cnot, control: control1, target: target, mps: &mps)
        applySingleQubitGate(.tGate, to: control2, mps: &mps)
        applySingleQubitGate(.tGate, to: target, mps: &mps)
        applyTwoQubitGate(.cnot, control: control1, target: control2, mps: &mps)
        applySingleQubitGate(.hadamard, to: target, mps: &mps)
        applySingleQubitGate(.tGate, to: control1, mps: &mps)
        applySingleQubitGate(.tGate.inverse, to: control2, mps: &mps)
        applyTwoQubitGate(.cnot, control: control1, target: control2, mps: &mps)
    }

    @_optimize(speed)
    private static func applyAdjacentTwoQubitGate(
        _ gate: QuantumGate,
        leftSite: Int,
        controlIsLeft: Bool,
        mps: inout MatrixProductState,
    ) {
        let rightSite = leftSite + 1
        let tensorA = mps.tensors[leftSite]
        let tensorB = mps.tensors[rightSite]

        let chiL = tensorA.leftBondDimension
        let chiM = tensorA.rightBondDimension
        let chiR = tensorB.rightBondDimension

        var combined = [Complex<Double>](repeating: .zero, count: chiL * 2 * 2 * chiR)

        for alpha in 0 ..< chiL {
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    for gamma in 0 ..< chiR {
                        var sum: Complex<Double> = .zero
                        for beta in 0 ..< chiM {
                            sum = sum + tensorA[alpha, i, beta] * tensorB[beta, j, gamma]
                        }
                        let idx = alpha * (4 * chiR) + i * (2 * chiR) + j * chiR + gamma
                        combined[idx] = sum
                    }
                }
            }
        }

        let gateMatrix = gate.matrix()
        var transformed = [Complex<Double>](repeating: .zero, count: chiL * 2 * 2 * chiR)

        for alpha in 0 ..< chiL {
            for gamma in 0 ..< chiR {
                for iPrime in 0 ..< 2 {
                    for jPrime in 0 ..< 2 {
                        var sum: Complex<Double> = .zero
                        for i in 0 ..< 2 {
                            for j in 0 ..< 2 {
                                let (gateRow, gateCol) = controlIsLeft
                                    ? (iPrime * 2 + jPrime, i * 2 + j)
                                    : (jPrime * 2 + iPrime, j * 2 + i)
                                let combIdx = alpha * (4 * chiR) + i * (2 * chiR) + j * chiR + gamma
                                sum = sum + gateMatrix[gateRow][gateCol] * combined[combIdx]
                            }
                        }
                        let outIdx = alpha * (4 * chiR) + iPrime * (2 * chiR) + jPrime * chiR + gamma
                        transformed[outIdx] = sum
                    }
                }
            }
        }

        let rows = chiL * 2
        let cols = 2 * chiR
        var matrix = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: cols), count: rows)
        for alpha in 0 ..< chiL {
            for i in 0 ..< 2 {
                let row = alpha * 2 + i
                for j in 0 ..< 2 {
                    for gamma in 0 ..< chiR {
                        let col = j * chiR + gamma
                        let idx = alpha * (4 * chiR) + i * (2 * chiR) + j * chiR + gamma
                        matrix[row][col] = transformed[idx]
                    }
                }
            }
        }

        let svdResult = SVDDecomposition.decompose(matrix: matrix, truncation: .maxBondDimension(mps.maxBondDimension))
        mps.addTruncationError(svdResult.truncationError)

        let newChiM = svdResult.singularValues.count

        let sqrtS = [Double](unsafeUninitializedCapacity: newChiM) { buffer, count in
            for k in 0 ..< newChiM {
                buffer[k] = svdResult.singularValues[k].squareRoot()
            }
            count = newChiM
        }

        let newAElements = [Complex<Double>](unsafeUninitializedCapacity: chiL * 2 * newChiM) { buffer, count in
            for alpha in 0 ..< chiL {
                for i in 0 ..< 2 {
                    let row = alpha * 2 + i
                    for beta in 0 ..< newChiM {
                        let idx = alpha * (2 * newChiM) + i * newChiM + beta
                        buffer[idx] = svdResult.u[row][beta] * sqrtS[beta]
                    }
                }
            }
            count = chiL * 2 * newChiM
        }

        let newBElements = [Complex<Double>](unsafeUninitializedCapacity: newChiM * 2 * chiR) { buffer, count in
            for beta in 0 ..< newChiM {
                for j in 0 ..< 2 {
                    for gamma in 0 ..< chiR {
                        let col = j * chiR + gamma
                        let idx = beta * (2 * chiR) + j * chiR + gamma
                        buffer[idx] = sqrtS[beta] * svdResult.vDagger[beta][col]
                    }
                }
            }
            count = newChiM * 2 * chiR
        }

        mps.updateTensor(at: leftSite, with: MPSTensor(
            leftBondDimension: chiL,
            rightBondDimension: newChiM,
            site: leftSite,
            elements: newAElements,
        ))

        mps.updateTensor(at: rightSite, with: MPSTensor(
            leftBondDimension: newChiM,
            rightBondDimension: chiR,
            site: rightSite,
            elements: newBElements,
        ))
    }

    @_optimize(speed)
    private static func applyNonAdjacentTwoQubitGate(
        _ gate: QuantumGate,
        control: Int,
        target: Int,
        mps: inout MatrixProductState,
    ) {
        let left = min(control, target)
        let right = max(control, target)

        for site in left ..< right - 1 {
            applyAdjacentSwap(site: site, mps: &mps)
        }

        let controlIsLeft = control < target
        applyAdjacentTwoQubitGate(gate, leftSite: right - 1, controlIsLeft: controlIsLeft, mps: &mps)

        for site in stride(from: right - 2, through: left, by: -1) {
            applyAdjacentSwap(site: site, mps: &mps)
        }
    }

    @_optimize(speed)
    private static func applyAdjacentSwap(site: Int, mps: inout MatrixProductState) {
        applyAdjacentTwoQubitGate(.swap, leftSite: site, controlIsLeft: true, mps: &mps)
    }
}

/// Convenience extension for fluent API on MatrixProductState.
///
/// Provides mutation and transformation methods for applying quantum gates directly
/// on MPS instances without explicit calls to ``MPSGateApplication``.
///
/// **Example:**
/// ```swift
/// var mps = MatrixProductState(qubits: 10, maxBondDimension: 32)
/// mps.applySingleQubitGate(.hadamard, to: 0)
/// mps.applyTwoQubitGate(.cnot, control: 0, target: 1)
///
/// let result = mps
///     .applying(.hadamard, to: 2)
///     .applying(.cnot, to: [2, 3])
/// ```
public extension MatrixProductState {
    /// Apply single-qubit gate to this MPS in place.
    ///
    /// **Example:**
    /// ```swift
    /// var mps = MatrixProductState(qubits: 5, maxBondDimension: 16)
    /// mps.applySingleQubitGate(.hadamard, to: 0)
    /// mps.applySingleQubitGate(.rotationZ(.pi / 4), to: 1)
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Single-qubit quantum gate to apply
    ///   - site: Site index (0 to qubits-1)
    /// - Complexity: O(chi^2)
    mutating func applySingleQubitGate(_ gate: QuantumGate, to site: Int) {
        MPSGateApplication.applySingleQubitGate(gate, to: site, mps: &self)
    }

    /// Apply two-qubit gate to this MPS in place.
    ///
    /// **Example:**
    /// ```swift
    /// var mps = MatrixProductState(qubits: 5, maxBondDimension: 32)
    /// mps.applySingleQubitGate(.hadamard, to: 0)
    /// mps.applyTwoQubitGate(.cnot, control: 0, target: 1)
    /// mps.applyTwoQubitGate(.cz, control: 0, target: 3)
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Two-qubit quantum gate to apply
    ///   - control: Control qubit index
    ///   - target: Target qubit index
    /// - Complexity: O(chi^3) for adjacent, O(d * chi^3) for distance d
    mutating func applyTwoQubitGate(_ gate: QuantumGate, control: Int, target: Int) {
        MPSGateApplication.applyTwoQubitGate(gate, control: control, target: target, mps: &self)
    }

    /// Return new MPS with gate applied to specified qubits.
    ///
    /// Creates a copy and applies the gate, supporting method chaining for
    /// functional-style circuit construction.
    ///
    /// **Example:**
    /// ```swift
    /// let mps = MatrixProductState(qubits: 5, maxBondDimension: 32)
    /// let result = mps
    ///     .applying(.hadamard, to: [0])
    ///     .applying(.cnot, to: [0, 1])
    ///     .applying(.cz, to: [1, 2])
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply
    ///   - qubits: Target qubit indices (1 for single-qubit, 2 for two-qubit gates)
    /// - Returns: New MPS with gate applied
    /// - Complexity: O(chi^2) for single-qubit, O(chi^3) for two-qubit gates
    @_eagerMove
    func applying(_ gate: QuantumGate, to qubits: [Int]) -> MatrixProductState {
        var copy = self
        switch qubits.count {
        case 1:
            MPSGateApplication.applySingleQubitGate(gate, to: qubits[0], mps: &copy)
        case 2:
            MPSGateApplication.applyTwoQubitGate(gate, control: qubits[0], target: qubits[1], mps: &copy)
        case 3:
            MPSGateApplication.applyToffoli(control1: qubits[0], control2: qubits[1], target: qubits[2], mps: &copy)
        default:
            ValidationUtilities.validateArrayCount(qubits, expected: gate.qubitsRequired, name: "Qubit array")
        }
        return copy
    }

    /// Return new MPS with single-qubit gate applied.
    ///
    /// Convenience method for single-qubit gates with cleaner syntax.
    ///
    /// **Example:**
    /// ```swift
    /// let mps = MatrixProductState(qubits: 5, maxBondDimension: 32)
    /// let result = mps
    ///     .applying(.hadamard, to: 0)
    ///     .applying(.rotationY(.pi / 4), to: 1)
    ///     .applying(.pauliX, to: 2)
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Single-qubit quantum gate to apply
    ///   - qubit: Target qubit index
    /// - Returns: New MPS with gate applied
    /// - Complexity: O(chi^2)
    @_eagerMove
    func applying(_ gate: QuantumGate, to qubit: Int) -> MatrixProductState {
        applying(gate, to: [qubit])
    }
}
