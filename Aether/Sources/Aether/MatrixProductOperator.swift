// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Matrix Product Operator representation of quantum operators as tensor networks.
///
/// Represents operators as tensor networks: O = Sum A^1[s1,s1'] A^2[s2,s2'] ... A^n[sn,sn'] where
/// each A^k is a rank-4 tensor with left bond index, physical input index, physical output index,
/// and right bond index. This representation enables efficient storage and manipulation of
/// local Hamiltonians with O(1) bond dimension, and supports time evolution via MPO-MPS multiplication.
///
/// Local Hamiltonians (those with nearest-neighbor or finite-range interactions) can be exactly
/// represented with constant bond dimension, making MPO ideal for quantum many-body simulations.
/// The MPO-MPS product yields a new MPS with bond dimension chi_MPO * chi_MPS, which can be
/// truncated via SVD to control computational cost.
///
/// **Example:**
/// ```swift
/// let zz = Observable(terms: [(1.0, PauliString(.z(0), .z(1)))])
/// let mpo = MatrixProductOperator(observable: zz)
/// let mps = MatrixProductState(qubits: 2)
/// let result = mpo.applying(to: mps, truncation: .maxBondDimension(16))
/// let expectation = mpo.expectationValue(bra: mps, ket: mps)
/// ```
///
/// - SeeAlso: ``MatrixProductState``
/// - SeeAlso: ``MPOTensor``
/// - SeeAlso: ``Observable``
@frozen public struct MatrixProductOperator: Sendable {
    /// Number of sites (qubits) in the operator chain.
    ///
    /// **Example:**
    /// ```swift
    /// let mpo = MatrixProductOperator(pauliString: PauliString(.z(0)), sites: 4)
    /// print(mpo.sites)  // 4
    /// ```
    public let sites: Int

    /// Array of MPO tensors forming the operator chain.
    ///
    /// Contains exactly ``sites`` tensors, one per site. First tensor has left bond dimension 1,
    /// last tensor has right bond dimension 1 (boundary conditions).
    ///
    /// **Example:**
    /// ```swift
    /// let mpo = MatrixProductOperator(pauliString: PauliString(.z(0)), sites: 2)
    /// print(mpo.tensors.count)  // 2
    /// ```
    public let tensors: [MPOTensor]

    /// Creates MPO from a quantum observable (weighted sum of Pauli strings).
    ///
    /// Converts an observable O = Sum_i c_i P_i into MPO form by summing individual
    /// Pauli string MPOs with appropriate coefficients. The number of sites is determined
    /// by the maximum qubit index appearing in any Pauli string plus one.
    ///
    /// **Example:**
    /// ```swift
    /// let h = Observable(terms: [
    ///     (0.5, PauliString(.z(0))),
    ///     (0.5, PauliString(.z(1))),
    ///     (1.0, PauliString(.x(0), .x(1)))
    /// ])
    /// let mpo = MatrixProductOperator(observable: h)
    /// ```
    ///
    /// - Parameter observable: Observable as weighted sum of Pauli strings
    /// - Complexity: O(k * n * d^4) where k is number of terms, n is sites, d is physical dimension
    /// - Precondition: Observable must have at least one term with at least one Pauli operator
    public init(observable: Observable) {
        let maxQubit = Self.findMaxQubit(observable: observable)
        ValidationUtilities.validatePositiveInt(maxQubit + 1, name: "Sites count")

        let numSites = maxQubit + 1
        sites = numSites

        var accumulatedTensors: [MPOTensor]?

        for term in observable.terms {
            let pauliMPO = Self.buildPauliStringMPO(
                pauliString: term.pauliString,
                sites: numSites,
                coefficient: term.coefficient,
            )

            if accumulatedTensors == nil {
                accumulatedTensors = pauliMPO
            } else {
                accumulatedTensors = Self.addMPOTensors(accumulatedTensors!, pauliMPO)
            }
        }

        tensors = accumulatedTensors!
    }

    /// Creates MPO from a single Pauli string operator.
    ///
    /// Constructs an MPO representing the tensor product of Pauli operators specified
    /// in the Pauli string. Sites without explicit Pauli operators have identity tensors.
    ///
    /// **Example:**
    /// ```swift
    /// let xz = PauliString(.x(0), .z(2))
    /// let mpo = MatrixProductOperator(pauliString: xz, sites: 4)
    /// // Represents X_0 tensor I_1 tensor Z_2 tensor I_3
    /// ```
    ///
    /// - Parameters:
    ///   - pauliString: Pauli string operator defining which Paulis act on which sites
    ///   - sites: Total number of sites in the MPO chain
    /// - Complexity: O(n) where n is sites
    /// - Precondition: sites > 0, all Pauli operator qubit indices < sites
    public init(pauliString: PauliString, sites: Int) {
        ValidationUtilities.validatePositiveInt(sites, name: "Sites count")

        for op in pauliString.operators {
            ValidationUtilities.validateIndexInBounds(op.qubit, bound: sites, name: "Pauli operator qubit")
        }

        self.sites = sites
        tensors = Self.buildPauliStringMPO(pauliString: pauliString, sites: sites, coefficient: 1.0)
    }

    /// Applies MPO to MPS: |psi'> = O|psi> with bond dimension truncation.
    ///
    /// Computes the MPO-MPS product by contracting MPO tensors with MPS tensors at each site,
    /// yielding a new MPS with bond dimension chi_MPO * chi_MPS. SVD truncation is applied
    /// to control bond dimension growth. This operation is fundamental for time evolution
    /// (O = exp(-iHt)) and operator expectation value computation.
    ///
    /// **Example:**
    /// ```swift
    /// let mpo = MatrixProductOperator(pauliString: PauliString(.x(0)), sites: 2)
    /// let mps = MatrixProductState(qubits: 2)
    /// let result = mpo.applying(to: mps, truncation: .maxBondDimension(32))
    /// ```
    ///
    /// - Parameters:
    ///   - mps: Input matrix product state |psi>
    ///   - truncation: SVD truncation strategy for bond dimension control
    /// - Returns: New MPS representing O|psi> with truncated bond dimensions
    /// - Complexity: O(chi_MPO * chi_MPS^2 * d^2 * n) where d is physical dimension
    /// - Precondition: mps.qubits == sites
    @_optimize(speed)
    public func applying(to mps: MatrixProductState, truncation: SVDTruncation) -> MatrixProductState {
        ValidationUtilities.validateQubitCountsEqual(mps.qubits, sites, name1: "MPS qubits", name2: "MPO sites")

        var resultTensors = [MPSTensor]()
        resultTensors.reserveCapacity(sites)

        for site in 0 ..< sites {
            let mpoTensor = tensors[site]
            let mpsTensor = mps.tensors[site]
            let contracted = Self.contractMPOMPS(mpo: mpoTensor, mps: mpsTensor, site: site)
            resultTensors.append(contracted)
        }

        var result = Self.buildMPSFromTensors(tensors: resultTensors, maxBondDimension: mps.maxBondDimension)
        result = Self.truncateMPS(mps: result, truncation: truncation)
        result = Self.canonicalizeMPS(mps: result, truncation: truncation)

        return result
    }

    /// Computes expectation value <bra|O|ket> using transfer matrix contraction.
    ///
    /// Evaluates the matrix element <phi|O|psi> by contracting the bra MPS, MPO, and ket MPS
    /// from left to right using transfer matrices. For bra == ket, this gives the standard
    /// expectation value <psi|O|psi>.
    ///
    /// **Example:**
    /// ```swift
    /// let h = Observable(terms: [(1.0, PauliString(.z(0), .z(1)))])
    /// let mpo = MatrixProductOperator(observable: h)
    /// let mps = MatrixProductState(qubits: 2)
    /// let energy = mpo.expectationValue(bra: mps, ket: mps)
    /// ```
    ///
    /// - Parameters:
    ///   - bra: Left MPS <phi| (will be conjugated)
    ///   - ket: Right MPS |psi>
    /// - Returns: Real part of matrix element <phi|O|psi>
    /// - Complexity: O(chi_bra * chi_ket * chi_MPO * d^2 * n)
    /// - Precondition: bra.qubits == ket.qubits == sites
    @_optimize(speed)
    public func expectationValue(bra: MatrixProductState, ket: MatrixProductState) -> Double {
        ValidationUtilities.validateQubitCountsEqual(bra.qubits, sites, name1: "Bra qubits", name2: "MPO sites")
        ValidationUtilities.validateQubitCountsEqual(ket.qubits, sites, name1: "Ket qubits", name2: "MPO sites")

        var transfer = Self.initializeTransferMatrix(
            braLeftDim: 1,
            ketLeftDim: 1,
            mpoLeftDim: 1,
        )

        let cachedBraTensors = bra.tensors
        let cachedKetTensors = ket.tensors
        let cachedMpoTensors = tensors

        for site in 0 ..< sites {
            transfer = Self.propagateTransferMatrixOptimized(
                transfer: transfer,
                braTensor: cachedBraTensors[site],
                mpoTensor: cachedMpoTensors[site],
                ketTensor: cachedKetTensors[site],
            )
        }

        return transfer[0][0][0].real
    }

    /// Finds maximum qubit index in observable to determine system size.
    @_effects(readonly)
    private static func findMaxQubit(observable: Observable) -> Int {
        var maxQubit = -1
        for term in observable.terms {
            for op in term.pauliString.operators {
                maxQubit = max(maxQubit, op.qubit)
            }
        }
        return max(maxQubit, 0)
    }

    /// Constructs MPO tensor chain for a single Pauli string operator.
    @_effects(readonly)
    private static func buildPauliStringMPO(
        pauliString: PauliString,
        sites: Int,
        coefficient: Double,
    ) -> [MPOTensor] {
        var pauliMap = [Int: PauliBasis]()
        for op in pauliString.operators {
            pauliMap[op.qubit] = op.basis
        }

        var mpoTensors = [MPOTensor]()
        mpoTensors.reserveCapacity(sites)

        for site in 0 ..< sites {
            let pauliAtSite = pauliMap[site]
            let matrix = pauliMatrix(pauliAtSite)

            var elements = [Complex<Double>]()
            elements.reserveCapacity(4)

            for physIn in 0 ..< 2 {
                for physOut in 0 ..< 2 {
                    var element = matrix[physIn][physOut]
                    if site == 0 {
                        element = element * coefficient
                    }
                    elements.append(element)
                }
            }

            mpoTensors.append(MPOTensor(
                leftBondDimension: 1,
                rightBondDimension: 1,
                elements: elements,
            ))
        }

        return mpoTensors
    }

    /// Adds two MPOs using direct sum of bond dimensions with optional truncation.
    @_optimize(speed)
    private static func addMPOTensors(_ mpo1: [MPOTensor], _ mpo2: [MPOTensor]) -> [MPOTensor] {
        var result = [MPOTensor]()
        result.reserveCapacity(mpo1.count)

        let maxMPOBondDimension = 64

        for site in 0 ..< mpo1.count {
            let t1 = mpo1[site]
            let t2 = mpo2[site]

            let isFirst = site == 0
            let isLast = site == mpo1.count - 1

            if isFirst, isLast {
                var elements = [Complex<Double>]()
                elements.reserveCapacity(4)
                for physIn in 0 ..< 2 {
                    for physOut in 0 ..< 2 {
                        elements.append(t1[0, physIn, physOut, 0] + t2[0, physIn, physOut, 0])
                    }
                }
                result.append(MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements))
            } else if isFirst {
                let rawRightDim = t1.rightBondDimension + t2.rightBondDimension
                let newRightDim = min(rawRightDim, maxMPOBondDimension)

                if rawRightDim <= maxMPOBondDimension {
                    var elements = [Complex<Double>]()
                    elements.reserveCapacity(4 * newRightDim)

                    for physIn in 0 ..< 2 {
                        for physOut in 0 ..< 2 {
                            for beta in 0 ..< t1.rightBondDimension {
                                elements.append(t1[0, physIn, physOut, beta])
                            }
                            for beta in 0 ..< t2.rightBondDimension {
                                elements.append(t2[0, physIn, physOut, beta])
                            }
                        }
                    }

                    result.append(MPOTensor(leftBondDimension: 1, rightBondDimension: newRightDim, elements: elements))
                } else {
                    let truncated = truncateMPOTensorRight(
                        t1: t1,
                        t2: t2,
                        targetDim: maxMPOBondDimension,
                        isFirst: true,
                    )
                    result.append(truncated)
                }
            } else if isLast {
                let rawLeftDim = t1.leftBondDimension + t2.leftBondDimension
                let newLeftDim = min(rawLeftDim, maxMPOBondDimension)

                if rawLeftDim <= maxMPOBondDimension {
                    var elements = [Complex<Double>]()
                    elements.reserveCapacity(newLeftDim * 4)

                    for alpha in 0 ..< t1.leftBondDimension {
                        for physIn in 0 ..< 2 {
                            for physOut in 0 ..< 2 {
                                elements.append(t1[alpha, physIn, physOut, 0])
                            }
                        }
                    }
                    for alpha in 0 ..< t2.leftBondDimension {
                        for physIn in 0 ..< 2 {
                            for physOut in 0 ..< 2 {
                                elements.append(t2[alpha, physIn, physOut, 0])
                            }
                        }
                    }

                    result.append(MPOTensor(leftBondDimension: newLeftDim, rightBondDimension: 1, elements: elements))
                } else {
                    let truncated = truncateMPOTensorLeft(
                        t1: t1,
                        t2: t2,
                        targetDim: maxMPOBondDimension,
                        isLast: true,
                    )
                    result.append(truncated)
                }
            } else {
                let rawLeftDim = t1.leftBondDimension + t2.leftBondDimension
                let rawRightDim = t1.rightBondDimension + t2.rightBondDimension
                let newLeftDim = min(rawLeftDim, maxMPOBondDimension)
                let newRightDim = min(rawRightDim, maxMPOBondDimension)

                if rawLeftDim <= maxMPOBondDimension, rawRightDim <= maxMPOBondDimension {
                    var elements = [Complex<Double>]()
                    elements.reserveCapacity(newLeftDim * 4 * newRightDim)

                    for alpha in 0 ..< newLeftDim {
                        for physIn in 0 ..< 2 {
                            for physOut in 0 ..< 2 {
                                for beta in 0 ..< newRightDim {
                                    if alpha < t1.leftBondDimension, beta < t1.rightBondDimension {
                                        elements.append(t1[alpha, physIn, physOut, beta])
                                    } else if alpha >= t1.leftBondDimension, beta >= t1.rightBondDimension {
                                        let a2 = alpha - t1.leftBondDimension
                                        let b2 = beta - t1.rightBondDimension
                                        elements.append(t2[a2, physIn, physOut, b2])
                                    } else {
                                        elements.append(.zero)
                                    }
                                }
                            }
                        }
                    }

                    result.append(MPOTensor(leftBondDimension: newLeftDim, rightBondDimension: newRightDim, elements: elements))
                } else {
                    let truncated = truncateMPOTensorBoth(
                        t1: t1,
                        t2: t2,
                        targetLeftDim: maxMPOBondDimension,
                        targetRightDim: maxMPOBondDimension,
                    )
                    result.append(truncated)
                }
            }
        }

        return result
    }

    /// Truncates combined MPO tensor right bond dimension via SVD.
    @_optimize(speed)
    private static func truncateMPOTensorRight(
        t1: MPOTensor,
        t2: MPOTensor,
        targetDim: Int,
        isFirst: Bool,
    ) -> MPOTensor {
        let leftDim = isFirst ? 1 : t1.leftBondDimension + t2.leftBondDimension
        let combinedRightDim = t1.rightBondDimension + t2.rightBondDimension

        var matrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: combinedRightDim),
            count: leftDim * 4,
        )

        for alpha in 0 ..< (isFirst ? 1 : t1.leftBondDimension) {
            for physIn in 0 ..< 2 {
                for physOut in 0 ..< 2 {
                    let row = alpha * 4 + physIn * 2 + physOut
                    for beta in 0 ..< t1.rightBondDimension {
                        matrix[row][beta] = t1[isFirst ? 0 : alpha, physIn, physOut, beta]
                    }
                }
            }
        }

        if !isFirst {
            for alpha in 0 ..< t2.leftBondDimension {
                for physIn in 0 ..< 2 {
                    for physOut in 0 ..< 2 {
                        let row = (t1.leftBondDimension + alpha) * 4 + physIn * 2 + physOut
                        for beta in 0 ..< t2.rightBondDimension {
                            matrix[row][t1.rightBondDimension + beta] = t2[alpha, physIn, physOut, beta]
                        }
                    }
                }
            }
        } else {
            for physIn in 0 ..< 2 {
                for physOut in 0 ..< 2 {
                    let row = physIn * 2 + physOut
                    for beta in 0 ..< t2.rightBondDimension {
                        matrix[row][t1.rightBondDimension + beta] = t2[0, physIn, physOut, beta]
                    }
                }
            }
        }

        let svdResult = SVDDecomposition.decompose(matrix: matrix, truncation: .maxBondDimension(targetDim))
        let newRightDim = svdResult.singularValues.count

        var elements = [Complex<Double>]()
        let actualLeftDim = isFirst ? 1 : leftDim
        elements.reserveCapacity(actualLeftDim * 4 * newRightDim)

        for alpha in 0 ..< actualLeftDim {
            for physIn in 0 ..< 2 {
                for physOut in 0 ..< 2 {
                    for beta in 0 ..< newRightDim {
                        let row = alpha * 4 + physIn * 2 + physOut
                        elements.append(svdResult.u[row][beta] * svdResult.singularValues[beta])
                    }
                }
            }
        }

        return MPOTensor(leftBondDimension: actualLeftDim, rightBondDimension: newRightDim, elements: elements)
    }

    /// Truncates combined MPO tensor left bond dimension via SVD.
    @_optimize(speed)
    private static func truncateMPOTensorLeft(
        t1: MPOTensor,
        t2: MPOTensor,
        targetDim: Int,
        isLast: Bool,
    ) -> MPOTensor {
        let combinedLeftDim = t1.leftBondDimension + t2.leftBondDimension
        let rightDim = isLast ? 1 : t1.rightBondDimension + t2.rightBondDimension

        var matrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: rightDim * 4),
            count: combinedLeftDim,
        )

        for alpha in 0 ..< t1.leftBondDimension {
            for physIn in 0 ..< 2 {
                for physOut in 0 ..< 2 {
                    for beta in 0 ..< (isLast ? 1 : t1.rightBondDimension) {
                        let col = beta * 4 + physIn * 2 + physOut
                        matrix[alpha][col] = t1[alpha, physIn, physOut, isLast ? 0 : beta]
                    }
                }
            }
        }

        for alpha in 0 ..< t2.leftBondDimension {
            for physIn in 0 ..< 2 {
                for physOut in 0 ..< 2 {
                    for beta in 0 ..< (isLast ? 1 : t2.rightBondDimension) {
                        let col = (isLast ? 0 : t1.rightBondDimension + beta) * 4 + physIn * 2 + physOut
                        matrix[t1.leftBondDimension + alpha][col] = t2[alpha, physIn, physOut, isLast ? 0 : beta]
                    }
                }
            }
        }

        let svdResult = SVDDecomposition.decompose(matrix: matrix, truncation: .maxBondDimension(targetDim))
        let newLeftDim = svdResult.singularValues.count

        var elements = [Complex<Double>]()
        let actualRightDim = isLast ? 1 : rightDim
        elements.reserveCapacity(newLeftDim * 4 * actualRightDim)

        for alpha in 0 ..< newLeftDim {
            for physIn in 0 ..< 2 {
                for physOut in 0 ..< 2 {
                    for beta in 0 ..< actualRightDim {
                        let col = beta * 4 + physIn * 2 + physOut
                        var sum: Complex<Double> = .zero
                        for k in 0 ..< combinedLeftDim {
                            sum = sum + svdResult.u[k][alpha].conjugate * svdResult.singularValues[alpha] * matrix[k][col]
                        }
                        elements.append(sum)
                    }
                }
            }
        }

        return MPOTensor(leftBondDimension: newLeftDim, rightBondDimension: actualRightDim, elements: elements)
    }

    /// Truncates combined MPO tensor on both left and right bond dimensions.
    @_optimize(speed)
    private static func truncateMPOTensorBoth(
        t1: MPOTensor,
        t2: MPOTensor,
        targetLeftDim: Int,
        targetRightDim: Int,
    ) -> MPOTensor {
        let combinedLeftDim = t1.leftBondDimension + t2.leftBondDimension
        let combinedRightDim = t1.rightBondDimension + t2.rightBondDimension

        var fullTensor = [Complex<Double>](repeating: .zero, count: combinedLeftDim * 4 * combinedRightDim)

        for alpha in 0 ..< t1.leftBondDimension {
            for physIn in 0 ..< 2 {
                for physOut in 0 ..< 2 {
                    for beta in 0 ..< t1.rightBondDimension {
                        let idx = alpha * 4 * combinedRightDim + physIn * 2 * combinedRightDim + physOut * combinedRightDim + beta
                        fullTensor[idx] = t1[alpha, physIn, physOut, beta]
                    }
                }
            }
        }

        for alpha in 0 ..< t2.leftBondDimension {
            for physIn in 0 ..< 2 {
                for physOut in 0 ..< 2 {
                    for beta in 0 ..< t2.rightBondDimension {
                        let newAlpha = t1.leftBondDimension + alpha
                        let newBeta = t1.rightBondDimension + beta
                        let idx = newAlpha * 4 * combinedRightDim + physIn * 2 * combinedRightDim + physOut * combinedRightDim + newBeta
                        fullTensor[idx] = t2[alpha, physIn, physOut, beta]
                    }
                }
            }
        }

        var matrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: combinedRightDim),
            count: combinedLeftDim * 4,
        )

        for alpha in 0 ..< combinedLeftDim {
            for physIn in 0 ..< 2 {
                for physOut in 0 ..< 2 {
                    let row = alpha * 4 + physIn * 2 + physOut
                    for beta in 0 ..< combinedRightDim {
                        let idx = alpha * 4 * combinedRightDim + physIn * 2 * combinedRightDim + physOut * combinedRightDim + beta
                        matrix[row][beta] = fullTensor[idx]
                    }
                }
            }
        }

        let svdResult = SVDDecomposition.decompose(matrix: matrix, truncation: .maxBondDimension(targetRightDim))
        let newRightDim = svdResult.singularValues.count

        var intermediate = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: newRightDim),
            count: combinedLeftDim * 4,
        )

        for row in 0 ..< combinedLeftDim * 4 {
            for beta in 0 ..< newRightDim {
                intermediate[row][beta] = svdResult.u[row][beta] * svdResult.singularValues[beta]
            }
        }

        var leftMatrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: 4 * newRightDim),
            count: combinedLeftDim,
        )

        for alpha in 0 ..< combinedLeftDim {
            for physIn in 0 ..< 2 {
                for physOut in 0 ..< 2 {
                    let row = alpha * 4 + physIn * 2 + physOut
                    for beta in 0 ..< newRightDim {
                        let col = physIn * 2 * newRightDim + physOut * newRightDim + beta
                        leftMatrix[alpha][col] = intermediate[row][beta]
                    }
                }
            }
        }

        let svdResult2 = SVDDecomposition.decompose(matrix: leftMatrix, truncation: .maxBondDimension(targetLeftDim))
        let newLeftDim = svdResult2.singularValues.count

        var elements = [Complex<Double>]()
        elements.reserveCapacity(newLeftDim * 4 * newRightDim)

        for alpha in 0 ..< newLeftDim {
            for physIn in 0 ..< 2 {
                for physOut in 0 ..< 2 {
                    for beta in 0 ..< newRightDim {
                        let col = physIn * 2 * newRightDim + physOut * newRightDim + beta
                        var sum: Complex<Double> = .zero
                        for k in 0 ..< combinedLeftDim {
                            sum = sum + svdResult2.u[k][alpha].conjugate * svdResult2.singularValues[alpha] * leftMatrix[k][col]
                        }
                        elements.append(sum)
                    }
                }
            }
        }

        return MPOTensor(leftBondDimension: newLeftDim, rightBondDimension: newRightDim, elements: elements)
    }

    /// Returns 2x2 matrix representation of Pauli operator or identity.
    @_effects(readonly)
    private static func pauliMatrix(_ basis: PauliBasis?) -> [[Complex<Double>]] {
        guard let basis else {
            return [[.one, .zero], [.zero, .one]]
        }

        switch basis {
        case .x:
            return [[.zero, .one], [.one, .zero]]
        case .y:
            return [[.zero, Complex(0, -1)], [Complex(0, 1), .zero]]
        case .z:
            return [[.one, .zero], [.zero, Complex(-1, 0)]]
        }
    }

    /// Contracts single MPO tensor with MPS tensor at one site.
    @_optimize(speed)
    private static func contractMPOMPS(mpo: MPOTensor, mps: MPSTensor, site: Int) -> MPSTensor {
        let mpoLeftDim = mpo.leftBondDimension
        let mpoRightDim = mpo.rightBondDimension
        let mpsLeftDim = mps.leftBondDimension
        let mpsRightDim = mps.rightBondDimension

        let newLeftDim = mpoLeftDim * mpsLeftDim
        let newRightDim = mpoRightDim * mpsRightDim

        var elements = [Complex<Double>]()
        elements.reserveCapacity(newLeftDim * 2 * newRightDim)

        for alphaO in 0 ..< mpoLeftDim {
            for alphaS in 0 ..< mpsLeftDim {
                for physOut in 0 ..< 2 {
                    for betaO in 0 ..< mpoRightDim {
                        for betaS in 0 ..< mpsRightDim {
                            var sum: Complex<Double> = .zero
                            for physIn in 0 ..< 2 {
                                let mpoElement = mpo[alphaO, physIn, physOut, betaO]
                                let mpsElement = mps[alphaS, physIn, betaS]
                                sum = sum + mpoElement * mpsElement
                            }
                            elements.append(sum)
                        }
                    }
                }
            }
        }

        return MPSTensor(
            leftBondDimension: newLeftDim,
            rightBondDimension: newRightDim,
            site: site,
            elements: elements,
        )
    }

    /// Creates MPS from array of tensors with specified bond dimension limit.
    @_optimize(speed)
    private static func buildMPSFromTensors(tensors: [MPSTensor], maxBondDimension: Int) -> MatrixProductState {
        var mps = MatrixProductState(qubits: tensors.count, maxBondDimension: maxBondDimension)

        for site in 0 ..< tensors.count {
            mps.updateTensor(at: site, with: tensors[site])
        }

        return mps
    }

    /// Truncates MPS bond dimensions via left-to-right SVD sweep.
    @_optimize(speed)
    private static func truncateMPS(mps: MatrixProductState, truncation: SVDTruncation) -> MatrixProductState {
        var result = mps

        for site in 0 ..< result.qubits - 1 {
            let tensor = result.tensors[site]
            let matrix = tensor.reshapeForSVD(mergeLeft: true)

            let svdResult = SVDDecomposition.decompose(matrix: matrix, truncation: truncation)

            let newRightDim = svdResult.singularValues.count
            let leftDim = tensor.leftBondDimension

            var leftElements = [Complex<Double>]()
            leftElements.reserveCapacity(leftDim * 2 * newRightDim)

            for alpha in 0 ..< leftDim {
                for physical in 0 ..< 2 {
                    for beta in 0 ..< newRightDim {
                        let rowIdx = alpha * 2 + physical
                        leftElements.append(svdResult.u[rowIdx][beta])
                    }
                }
            }

            let leftTensor = MPSTensor(
                leftBondDimension: leftDim,
                rightBondDimension: newRightDim,
                site: site,
                elements: leftElements,
            )

            result.updateTensor(at: site, with: leftTensor)
            result.addTruncationError(svdResult.truncationError)

            let nextTensor = result.tensors[site + 1]
            let nextRightDim = nextTensor.rightBondDimension

            var nextElements = [Complex<Double>]()
            nextElements.reserveCapacity(newRightDim * 2 * nextRightDim)

            for alpha in 0 ..< newRightDim {
                for physical in 0 ..< 2 {
                    for beta in 0 ..< nextRightDim {
                        var sum: Complex<Double> = .zero
                        let oldLeftDim = nextTensor.leftBondDimension
                        for gamma in 0 ..< min(oldLeftDim, svdResult.vDagger[0].count) {
                            let sigma = svdResult.singularValues[alpha]
                            let vElement = svdResult.vDagger[alpha][gamma]
                            let sVt = vElement * sigma
                            if gamma < oldLeftDim {
                                sum = sum + sVt * nextTensor[gamma, physical, beta]
                            }
                        }
                        nextElements.append(sum)
                    }
                }
            }

            let nextNewTensor = MPSTensor(
                leftBondDimension: newRightDim,
                rightBondDimension: nextRightDim,
                site: site + 1,
                elements: nextElements,
            )

            result.updateTensor(at: site + 1, with: nextNewTensor)
        }

        return result
    }

    /// Creates initial transfer matrix with identity boundary condition.
    @_effects(readonly)
    private static func initializeTransferMatrix(
        braLeftDim: Int,
        ketLeftDim: Int,
        mpoLeftDim: Int,
    ) -> [[[Complex<Double>]]] {
        var transfer = [[[Complex<Double>]]](
            repeating: [[Complex<Double>]](
                repeating: [Complex<Double>](repeating: .zero, count: ketLeftDim),
                count: mpoLeftDim,
            ),
            count: braLeftDim,
        )
        transfer[0][0][0] = .one
        return transfer
    }

    /// Propagates transfer matrix through one site by tensor contraction.
    @_optimize(speed)
    private static func propagateTransferMatrix(
        transfer: [[[Complex<Double>]]],
        braTensor: MPSTensor,
        mpoTensor: MPOTensor,
        ketTensor: MPSTensor,
    ) -> [[[Complex<Double>]]] {
        let braRightDim = braTensor.rightBondDimension
        let mpoRightDim = mpoTensor.rightBondDimension
        let ketRightDim = ketTensor.rightBondDimension

        var newTransfer = [[[Complex<Double>]]](
            repeating: [[Complex<Double>]](
                repeating: [Complex<Double>](repeating: .zero, count: ketRightDim),
                count: mpoRightDim,
            ),
            count: braRightDim,
        )

        let braLeftDim = braTensor.leftBondDimension
        let mpoLeftDim = mpoTensor.leftBondDimension
        let ketLeftDim = ketTensor.leftBondDimension

        for alphaBra in 0 ..< braLeftDim {
            for alphaO in 0 ..< mpoLeftDim {
                for alphaKet in 0 ..< ketLeftDim {
                    let transferElement = transfer[alphaBra][alphaO][alphaKet]
                    if transferElement.magnitudeSquared < 1e-30 {
                        continue
                    }

                    for physBra in 0 ..< 2 {
                        for physKet in 0 ..< 2 {
                            for betaBra in 0 ..< braRightDim {
                                for betaO in 0 ..< mpoRightDim {
                                    for betaKet in 0 ..< ketRightDim {
                                        let braConj = braTensor[alphaBra, physBra, betaBra].conjugate
                                        let mpoElement = mpoTensor[alphaO, physKet, physBra, betaO]
                                        let ketElement = ketTensor[alphaKet, physKet, betaKet]

                                        let contribution = transferElement * braConj * mpoElement * ketElement
                                        newTransfer[betaBra][betaO][betaKet] = newTransfer[betaBra][betaO][betaKet] + contribution
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return newTransfer
    }

    /// Propagates transfer matrix with intermediate caching for better performance.
    @_optimize(speed)
    private static func propagateTransferMatrixOptimized(
        transfer: [[[Complex<Double>]]],
        braTensor: MPSTensor,
        mpoTensor: MPOTensor,
        ketTensor: MPSTensor,
    ) -> [[[Complex<Double>]]] {
        let braLeftDim = braTensor.leftBondDimension
        let braRightDim = braTensor.rightBondDimension
        let mpoLeftDim = mpoTensor.leftBondDimension
        let mpoRightDim = mpoTensor.rightBondDimension
        let ketLeftDim = ketTensor.leftBondDimension
        let ketRightDim = ketTensor.rightBondDimension

        let braElements = braTensor.elements
        let mpoElements = mpoTensor.elements
        let ketElements = ketTensor.elements

        var braMpoIntermediate = [[[[Complex<Double>]]]](
            repeating: [[[Complex<Double>]]](
                repeating: [[Complex<Double>]](
                    repeating: [Complex<Double>](repeating: .zero, count: mpoRightDim),
                    count: braRightDim,
                ),
                count: 2,
            ),
            count: mpoLeftDim,
        )

        for alphaO in 0 ..< mpoLeftDim {
            for alphaBra in 0 ..< braLeftDim {
                for physBra in 0 ..< 2 {
                    for betaBra in 0 ..< braRightDim {
                        let braIdx = alphaBra * (2 * braRightDim) + physBra * braRightDim + betaBra
                        let braConj = Complex(braElements[braIdx].real, -braElements[braIdx].imaginary)

                        for alphaKet in 0 ..< ketLeftDim {
                            let transferElement = transfer[alphaBra][alphaO][alphaKet]
                            if transferElement.magnitudeSquared < 1e-30 {
                                continue
                            }

                            let scaledBra = transferElement * braConj

                            for physKet in 0 ..< 2 {
                                let mpoIdx = alphaO * (4 * mpoRightDim) + physKet * (2 * mpoRightDim) + physBra * mpoRightDim

                                for betaO in 0 ..< mpoRightDim {
                                    let mpoElement = mpoElements[mpoIdx + betaO]
                                    braMpoIntermediate[alphaO][physKet][betaBra][betaO] = braMpoIntermediate[alphaO][physKet][betaBra][betaO] + scaledBra * mpoElement
                                }
                            }
                        }
                    }
                }
            }
        }

        var newTransfer = [[[Complex<Double>]]](
            repeating: [[Complex<Double>]](
                repeating: [Complex<Double>](repeating: .zero, count: ketRightDim),
                count: mpoRightDim,
            ),
            count: braRightDim,
        )

        for alphaKet in 0 ..< ketLeftDim {
            for physKet in 0 ..< 2 {
                for betaKet in 0 ..< ketRightDim {
                    let ketIdx = alphaKet * (2 * ketRightDim) + physKet * ketRightDim + betaKet
                    let ketElement = ketElements[ketIdx]

                    for betaBra in 0 ..< braRightDim {
                        for betaO in 0 ..< mpoRightDim {
                            for alphaO in 0 ..< mpoLeftDim {
                                let intermediate = braMpoIntermediate[alphaO][physKet][betaBra][betaO]
                                newTransfer[betaBra][betaO][betaKet] = newTransfer[betaBra][betaO][betaKet] + intermediate * ketElement
                            }
                        }
                    }
                }
            }
        }

        return newTransfer
    }

    /// Brings MPS to right-canonical form with bond dimension truncation.
    @_optimize(speed)
    private static func canonicalizeMPS(mps: MatrixProductState, truncation: SVDTruncation) -> MatrixProductState {
        var result = mps

        let targetDim: Int
        switch truncation {
        case let .maxBondDimension(dim):
            targetDim = dim
        case .relativeThreshold, .cumulativeWeight, .none:
            return result
        }

        for site in (1 ..< result.qubits).reversed() {
            let tensor = result.tensors[site]
            if tensor.leftBondDimension <= targetDim {
                continue
            }

            let matrix = tensor.reshapeForSVD(mergeLeft: false)
            let svdResult = SVDDecomposition.decompose(matrix: matrix, truncation: .maxBondDimension(targetDim))

            let newLeftDim = svdResult.singularValues.count
            let rightDim = tensor.rightBondDimension

            var rightElements = [Complex<Double>]()
            rightElements.reserveCapacity(newLeftDim * 2 * rightDim)

            for alpha in 0 ..< newLeftDim {
                for physical in 0 ..< 2 {
                    for beta in 0 ..< rightDim {
                        let colIdx = physical * rightDim + beta
                        rightElements.append(svdResult.vDagger[alpha][colIdx])
                    }
                }
            }

            let rightTensor = MPSTensor(
                leftBondDimension: newLeftDim,
                rightBondDimension: rightDim,
                site: site,
                elements: rightElements,
            )

            result.updateTensor(at: site, with: rightTensor)
            result.addTruncationError(svdResult.truncationError)

            let prevTensor = result.tensors[site - 1]
            let prevLeftDim = prevTensor.leftBondDimension

            var prevElements = [Complex<Double>]()
            prevElements.reserveCapacity(prevLeftDim * 2 * newLeftDim)

            for alpha in 0 ..< prevLeftDim {
                for physical in 0 ..< 2 {
                    for beta in 0 ..< newLeftDim {
                        var sum: Complex<Double> = .zero
                        let oldRightDim = prevTensor.rightBondDimension
                        for gamma in 0 ..< min(oldRightDim, svdResult.u.count) {
                            let sigma = svdResult.singularValues[beta]
                            let uElement = svdResult.u[gamma][beta]
                            let uSigma = uElement * sigma
                            if gamma < oldRightDim {
                                sum = sum + prevTensor[alpha, physical, gamma] * uSigma
                            }
                        }
                        prevElements.append(sum)
                    }
                }
            }

            let prevNewTensor = MPSTensor(
                leftBondDimension: prevLeftDim,
                rightBondDimension: newLeftDim,
                site: site - 1,
                elements: prevElements,
            )

            result.updateTensor(at: site - 1, with: prevNewTensor)
        }

        return result
    }
}
