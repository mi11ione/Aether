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
/// MPSGateApplication.apply(.hadamard, to: [0], mps: &mps)
/// MPSGateApplication.apply(.cnot, to: [0, 1], mps: &mps)
/// MPSGateApplication.apply(.toffoli, to: [0, 1, 2], mps: &mps)
/// ```
///
/// - SeeAlso: ``MatrixProductState``
/// - SeeAlso: ``MPSTensor``
/// - SeeAlso: ``SVDDecomposition``
public enum MPSGateApplication {
    /// Apply quantum gate to MPS at specified qubit indices.
    ///
    /// Routes to single-qubit, two-qubit, or Toffoli implementation based on qubit count.
    /// Single-qubit gates preserve bond dimension via O(chi^2) contraction. Two-qubit gates
    /// perform SVD truncation at O(chi^3). Non-adjacent two-qubit gates use SWAP networks.
    ///
    /// **Example:**
    /// ```swift
    /// var mps = MatrixProductState(qubits: 10, maxBondDimension: 32)
    /// MPSGateApplication.apply(.hadamard, to: [0], mps: &mps)
    /// MPSGateApplication.apply(.cnot, to: [0, 1], mps: &mps)
    /// MPSGateApplication.apply(.toffoli, to: [0, 1, 2], mps: &mps)
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply
    ///   - qubits: Target qubit indices (1 for single-qubit, 2 for two-qubit, 3 for Toffoli)
    ///   - mps: Matrix Product State to modify
    /// - Complexity: O(chi^2) single-qubit, O(chi^3) two-qubit, O(d * chi^3) non-adjacent
    /// - Precondition: 0 <= qubit < mps.qubits for all qubit indices
    /// - Precondition: All qubit indices must be distinct
    /// - Precondition: qubits.count must match gate qubit requirement (1, 2, or 3)
    @_optimize(speed)
    public static func apply(
        _ gate: QuantumGate,
        to qubits: [Int],
        mps: inout MatrixProductState,
    ) {
        switch qubits.count {
        case 1:
            applySingleQubitGate(gate, to: qubits[0], mps: &mps)
        case 2:
            applyTwoQubitGate(gate, control: qubits[0], target: qubits[1], mps: &mps)
        case 3:
            applyToffoli(control1: qubits[0], control2: qubits[1], target: qubits[2], mps: &mps)
        default:
            ValidationUtilities.validateArrayCount(qubits, expected: gate.qubitsRequired, name: "Qubit array")
        }
    }

    /// Contract gate 2x2 matrix with physical index of tensor at specified site.
    @_optimize(speed)
    private static func applySingleQubitGate(
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

    /// Route two-qubit gate to adjacent or non-adjacent implementation.
    @_optimize(speed)
    private static func applyTwoQubitGate(
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

    /// Decompose Toffoli into CNOT, T, T-dagger, and Hadamard sequence.
    @_optimize(speed)
    private static func applyToffoli(
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

    /// Contract two adjacent MPS tensors, apply 4x4 gate, and split via SVD truncation.
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

        let combinedCount = chiL * 2 * 2 * chiR
        let combined = [Complex<Double>](unsafeUninitializedCapacity: combinedCount) {
            buffer, count in
            for alpha in 0 ..< chiL {
                for i in 0 ..< 2 {
                    for j in 0 ..< 2 {
                        for gamma in 0 ..< chiR {
                            var sum: Complex<Double> = .zero
                            for beta in 0 ..< chiM {
                                sum = sum + tensorA[alpha, i, beta] * tensorB[beta, j, gamma]
                            }
                            let idx = alpha * (4 * chiR) + i * (2 * chiR) + j * chiR + gamma
                            buffer[idx] = sum
                        }
                    }
                }
            }
            count = combinedCount
        }

        let gateMatrix = gate.matrix()
        let transformed = [Complex<Double>](unsafeUninitializedCapacity: combinedCount) {
            buffer, count in
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
                            buffer[outIdx] = sum
                        }
                    }
                }
            }
            count = combinedCount
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

    /// SWAP network to bring qubits adjacent, apply gate, then SWAP back.
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

    /// Apply SWAP gate between adjacent sites for qubit routing.
    @_optimize(speed)
    private static func applyAdjacentSwap(site: Int, mps: inout MatrixProductState) {
        applyAdjacentTwoQubitGate(.swap, leftSite: site, controlIsLeft: true, mps: &mps)
    }

    /// Performs projective measurement on the specified qubit using the Born rule.
    ///
    /// Computes marginal probability P(0) via transfer matrix contraction,
    /// probabilistically selects outcome, projects the local tensor onto the
    /// measured subspace, and renormalizes the MPS.
    ///
    /// **Example:**
    /// ```swift
    /// var mps = MatrixProductState(qubits: 3, maxBondDimension: 16)
    /// MPSGateApplication.apply(.hadamard, to: [0], mps: &mps)
    /// MPSGateApplication.apply(.cnot, to: [0, 1], mps: &mps)
    /// let outcome = MPSGateApplication.measure(0, mps: &mps)
    /// ```
    ///
    /// - Parameters:
    ///   - qubit: Qubit index to measure (0 to qubits-1)
    ///   - mps: Matrix Product State to modify
    /// - Returns: Measurement outcome (0 or 1)
    /// - Precondition: qubit must be in range 0..<mps.qubits
    /// - Complexity: O(n * chi^3)
    /// - SeeAlso: ``reset(_:mps:)``
    @_optimize(speed)
    @discardableResult
    public static func measure(_ qubit: Int, mps: inout MatrixProductState) -> Int {
        ValidationUtilities.validateMPSSiteIndex(qubit, qubits: mps.qubits)

        let prob0 = marginalProbability(qubit: qubit, outcome: 0, mps: mps)

        var rng = SystemRandomNumberGenerator()
        let random = Double.random(in: 0 ..< 1, using: &rng)
        let outcome = random < prob0 ? 0 : 1

        projectOntoOutcome(qubit: qubit, outcome: outcome, mps: &mps)
        mps.normalize()

        return outcome
    }

    /// Resets the specified qubit to |0⟩ via measure-and-conditionally-flip.
    ///
    /// Performs a projective measurement followed by a conditional Pauli-X
    /// correction if the measurement outcome is |1⟩. The resulting state
    /// has the target qubit in |0⟩ with correlations properly collapsed.
    ///
    /// **Example:**
    /// ```swift
    /// var mps = MatrixProductState(qubits: 3, maxBondDimension: 16)
    /// MPSGateApplication.apply(.hadamard, to: [0], mps: &mps)
    /// MPSGateApplication.apply(.cnot, to: [0, 1], mps: &mps)
    /// MPSGateApplication.reset(0, mps: &mps)
    /// ```
    ///
    /// - Parameters:
    ///   - qubit: Qubit index to reset (0 to qubits-1)
    ///   - mps: Matrix Product State to modify
    /// - Precondition: qubit must be in range 0..<mps.qubits
    /// - Complexity: O(n * chi^3)
    /// - SeeAlso: ``measure(_:mps:)``
    @_optimize(speed)
    public static func reset(_ qubit: Int, mps: inout MatrixProductState) {
        let outcome = measure(qubit, mps: &mps)
        if outcome == 1 {
            applySingleQubitGate(.pauliX, to: qubit, mps: &mps)
        }
    }

    /// Computes marginal probability P(outcome) for a qubit via transfer matrix contraction.
    @_optimize(speed)
    @_effects(readonly)
    private static func marginalProbability(qubit: Int, outcome: Int, mps: MatrixProductState) -> Double {
        var transfer: [Complex<Double>] = [.one]

        for site in 0 ..< mps.qubits {
            let tensor = mps.tensors[site]
            let leftDim = tensor.leftBondDimension
            let rightDim = tensor.rightBondDimension
            let elems = tensor.elements
            let physStride = rightDim

            if site == qubit {
                let j = outcome
                let tmp = [Complex<Double>](unsafeUninitializedCapacity: leftDim * rightDim) { buffer, count in
                    for betaPrime in 0 ..< leftDim {
                        for alpha in 0 ..< rightDim {
                            var sum: Complex<Double> = .zero
                            for beta in 0 ..< leftDim {
                                sum = sum + transfer[beta * leftDim + betaPrime] * elems[beta * 2 * physStride + j * physStride + alpha].conjugate
                            }
                            buffer[betaPrime * rightDim + alpha] = sum
                        }
                    }
                    count = leftDim * rightDim
                }

                transfer = [Complex<Double>](unsafeUninitializedCapacity: rightDim * rightDim) { buffer, count in
                    for alpha in 0 ..< rightDim {
                        for alphaPrime in 0 ..< rightDim {
                            var sum: Complex<Double> = .zero
                            for betaPrime in 0 ..< leftDim {
                                sum = sum + tmp[betaPrime * rightDim + alpha] * elems[betaPrime * 2 * physStride + j * physStride + alphaPrime]
                            }
                            buffer[alpha * rightDim + alphaPrime] = sum
                        }
                    }
                    count = rightDim * rightDim
                }
            } else {
                let tmp = [Complex<Double>](unsafeUninitializedCapacity: leftDim * 2 * rightDim) { buffer, count in
                    for betaPrime in 0 ..< leftDim {
                        for physical in 0 ..< 2 {
                            for alpha in 0 ..< rightDim {
                                var sum: Complex<Double> = .zero
                                for beta in 0 ..< leftDim {
                                    sum = sum + transfer[beta * leftDim + betaPrime] * elems[beta * 2 * physStride + physical * physStride + alpha].conjugate
                                }
                                buffer[betaPrime * 2 * rightDim + physical * rightDim + alpha] = sum
                            }
                        }
                    }
                    count = leftDim * 2 * rightDim
                }

                transfer = [Complex<Double>](unsafeUninitializedCapacity: rightDim * rightDim) { buffer, count in
                    for alpha in 0 ..< rightDim {
                        for alphaPrime in 0 ..< rightDim {
                            var sum: Complex<Double> = .zero
                            for betaPrime in 0 ..< leftDim {
                                for physical in 0 ..< 2 {
                                    sum = sum + tmp[betaPrime * 2 * rightDim + physical * rightDim + alpha] * elems[betaPrime * 2 * physStride + physical * physStride + alphaPrime]
                                }
                            }
                            buffer[alpha * rightDim + alphaPrime] = sum
                        }
                    }
                    count = rightDim * rightDim
                }
            }
        }

        return max(0.0, transfer[0].real)
    }

    /// Projects qubit onto specified measurement outcome by zeroing non-selected physical index.
    @_optimize(speed)
    private static func projectOntoOutcome(qubit: Int, outcome: Int, mps: inout MatrixProductState) {
        let tensor = mps.tensors[qubit]
        let leftDim = tensor.leftBondDimension
        let rightDim = tensor.rightBondDimension
        let elemCount = leftDim * 2 * rightDim
        let nonOutcome = 1 - outcome

        let newElements = [Complex<Double>](unsafeUninitializedCapacity: elemCount) { buffer, count in
            for alpha in 0 ..< leftDim {
                for beta in 0 ..< rightDim {
                    buffer[alpha * 2 * rightDim + outcome * rightDim + beta] = tensor.elements[alpha * 2 * rightDim + outcome * rightDim + beta]
                    buffer[alpha * 2 * rightDim + nonOutcome * rightDim + beta] = .zero
                }
            }
            count = elemCount
        }

        mps.updateTensor(
            at: qubit,
            with: MPSTensor(leftBondDimension: leftDim, rightBondDimension: rightDim, site: qubit, elements: newElements),
        )
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
/// mps.apply(.hadamard, to: 0)
/// mps.apply(.cnot, to: [0, 1])
///
/// let result = mps
///     .applying(.hadamard, to: 2)
///     .applying(.cnot, to: [2, 3])
/// ```
public extension MatrixProductState {
    /// Apply quantum gate to this MPS in place.
    ///
    /// **Example:**
    /// ```swift
    /// var mps = MatrixProductState(qubits: 5, maxBondDimension: 32)
    /// mps.apply(.hadamard, to: [0])
    /// mps.apply(.cnot, to: [0, 1])
    /// mps.apply(.toffoli, to: [0, 1, 2])
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply
    ///   - qubits: Target qubit indices
    /// - Complexity: O(chi^2) single-qubit, O(chi^3) two-qubit
    mutating func apply(_ gate: QuantumGate, to qubits: [Int]) {
        MPSGateApplication.apply(gate, to: qubits, mps: &self)
    }

    /// Apply single-qubit gate to this MPS in place.
    ///
    /// **Example:**
    /// ```swift
    /// var mps = MatrixProductState(qubits: 5, maxBondDimension: 16)
    /// mps.apply(.hadamard, to: 0)
    /// mps.apply(.rotationZ(.pi / 4), to: 1)
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Single-qubit quantum gate to apply
    ///   - qubit: Target qubit index
    /// - Complexity: O(chi^2)
    mutating func apply(_ gate: QuantumGate, to qubit: Int) {
        MPSGateApplication.apply(gate, to: [qubit], mps: &self)
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
        MPSGateApplication.apply(gate, to: qubits, mps: &copy)
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
