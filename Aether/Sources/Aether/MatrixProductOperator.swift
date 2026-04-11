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
    ///
    /// - SeeAlso: ``tensors``
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
    ///
    /// - SeeAlso: ``MPOTensor``
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
    /// - SeeAlso: ``Observable``
    /// - SeeAlso: ``init(pauliString:sites:)``
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
                // Safety: accumulatedTensors is non-nil in this branch
                accumulatedTensors = Self.addMPOTensors(accumulatedTensors!, pauliMPO)
            }
        }

        // Safety: precondition ensures at least one term, so loop executes at least once
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
    /// - SeeAlso: ``PauliString``
    /// - SeeAlso: ``init(observable:)``
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
    /// - SeeAlso: ``MatrixProductState``
    /// - SeeAlso: ``SVDTruncation``
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
    /// - SeeAlso: ``MatrixProductState``
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
        let pauliMap = Dictionary(uniqueKeysWithValues: pauliString.operators.map { ($0.qubit, $0.basis) })

        var mpoTensors = [MPOTensor]()
        mpoTensors.reserveCapacity(sites)

        for site in 0 ..< sites {
            let pauliAtSite = pauliMap[site]
            let matrix = pauliMatrix(pauliAtSite)

            let coeff = site == 0 ? coefficient : 1.0
            let elements = [Complex<Double>](unsafeUninitializedCapacity: 4) { buffer, count in
                for physIn in 0 ..< 2 {
                    for physOut in 0 ..< 2 {
                        buffer[physIn * 2 + physOut] = matrix[physIn][physOut] * coeff
                    }
                }
                count = 4
            }

            mpoTensors.append(MPOTensor(
                leftBondDimension: 1,
                rightBondDimension: 1,
                elements: elements,
            ))
        }

        return mpoTensors
    }

    /// Magnitude-squared threshold below which transfer matrix elements are skipped.
    private static let transferMatrixSkipThreshold: Double = 1e-30

    /// 3D array indexed [bra][mpo][ket] for transfer matrix contraction.
    private typealias TransferMatrix = [[[Complex<Double>]]]
    /// 4D cache indexed [mpoLeft][physKet][braRight][mpoRight] for intermediate contraction.
    private typealias BraMPOCache = [[[[Complex<Double>]]]]

    /// Adds two MPOs using direct sum of bond dimensions.
    @_effects(readonly)
    @_optimize(speed)
    private static func addMPOTensors(_ mpo1: [MPOTensor], _ mpo2: [MPOTensor]) -> [MPOTensor] {
        var result = [MPOTensor]()
        result.reserveCapacity(mpo1.count)

        for site in 0 ..< mpo1.count {
            let t1 = mpo1[site]
            let t2 = mpo2[site]

            let isFirst = site == 0
            let isLast = site == mpo1.count - 1

            if isFirst, isLast {
                let elements = [Complex<Double>](unsafeUninitializedCapacity: 4) { buffer, count in
                    for physIn in 0 ..< 2 {
                        for physOut in 0 ..< 2 {
                            buffer[physIn * 2 + physOut] = t1[0, physIn, physOut, 0] + t2[0, physIn, physOut, 0]
                        }
                    }
                    count = 4
                }
                result.append(MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements))
            } else if isFirst {
                let newRightDim = t1.rightBondDimension + t2.rightBondDimension
                let elements = [Complex<Double>](unsafeUninitializedCapacity: 4 * newRightDim) { buffer, count in
                    var idx = 0
                    for physIn in 0 ..< 2 {
                        for physOut in 0 ..< 2 {
                            for beta in 0 ..< t1.rightBondDimension {
                                buffer[idx] = t1[0, physIn, physOut, beta]
                                idx += 1
                            }
                            for beta in 0 ..< t2.rightBondDimension {
                                buffer[idx] = t2[0, physIn, physOut, beta]
                                idx += 1
                            }
                        }
                    }
                    count = 4 * newRightDim
                }
                result.append(MPOTensor(leftBondDimension: 1, rightBondDimension: newRightDim, elements: elements))
            } else if isLast {
                let newLeftDim = t1.leftBondDimension + t2.leftBondDimension
                let elements = [Complex<Double>](unsafeUninitializedCapacity: newLeftDim * 4) { buffer, count in
                    var idx = 0
                    for alpha in 0 ..< t1.leftBondDimension {
                        for physIn in 0 ..< 2 {
                            for physOut in 0 ..< 2 {
                                buffer[idx] = t1[alpha, physIn, physOut, 0]
                                idx += 1
                            }
                        }
                    }
                    for alpha in 0 ..< t2.leftBondDimension {
                        for physIn in 0 ..< 2 {
                            for physOut in 0 ..< 2 {
                                buffer[idx] = t2[alpha, physIn, physOut, 0]
                                idx += 1
                            }
                        }
                    }
                    count = newLeftDim * 4
                }
                result.append(MPOTensor(leftBondDimension: newLeftDim, rightBondDimension: 1, elements: elements))
            } else {
                let newLeftDim = t1.leftBondDimension + t2.leftBondDimension
                let newRightDim = t1.rightBondDimension + t2.rightBondDimension
                let totalCount = newLeftDim * 4 * newRightDim
                let elements = [Complex<Double>](unsafeUninitializedCapacity: totalCount) { buffer, count in
                    buffer.initialize(repeating: .zero)
                    for alpha in 0 ..< t1.leftBondDimension {
                        for physIn in 0 ..< 2 {
                            for physOut in 0 ..< 2 {
                                for beta in 0 ..< t1.rightBondDimension {
                                    buffer[alpha * 4 * newRightDim + physIn * 2 * newRightDim + physOut * newRightDim + beta] = t1[alpha, physIn, physOut, beta]
                                }
                            }
                        }
                    }
                    for alpha in 0 ..< t2.leftBondDimension {
                        for physIn in 0 ..< 2 {
                            for physOut in 0 ..< 2 {
                                for beta in 0 ..< t2.rightBondDimension {
                                    let idx = (t1.leftBondDimension + alpha) * 4 * newRightDim + physIn * 2 * newRightDim + physOut * newRightDim + (t1.rightBondDimension + beta)
                                    buffer[idx] = t2[alpha, physIn, physOut, beta]
                                }
                            }
                        }
                    }
                    count = totalCount
                }
                result.append(MPOTensor(leftBondDimension: newLeftDim, rightBondDimension: newRightDim, elements: elements))
            }
        }

        return result
    }

    /// 2x2 identity matrix.
    private static let pauliI: [[Complex<Double>]] = [[.one, .zero], [.zero, .one]]
    /// 2x2 Pauli-X matrix.
    private static let pauliX: [[Complex<Double>]] = [[.zero, .one], [.one, .zero]]
    /// 2x2 Pauli-Y matrix.
    private static let pauliY: [[Complex<Double>]] = [[.zero, Complex(0, -1)], [Complex(0, 1), .zero]]
    /// 2x2 Pauli-Z matrix.
    private static let pauliZ: [[Complex<Double>]] = [[.one, .zero], [.zero, Complex(-1, 0)]]

    /// Returns 2x2 matrix representation of Pauli operator or identity.
    @inline(__always)
    @_effects(readonly)
    private static func pauliMatrix(_ basis: PauliBasis?) -> [[Complex<Double>]] {
        guard let basis else { return pauliI }
        switch basis {
        case .x: return pauliX
        case .y: return pauliY
        case .z: return pauliZ
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

        let elements = [Complex<Double>](unsafeUninitializedCapacity: newLeftDim * 2 * newRightDim) { buffer, count in
            var idx = 0
            for alphaO in 0 ..< mpoLeftDim {
                for alphaS in 0 ..< mpsLeftDim {
                    for physOut in 0 ..< 2 {
                        for betaO in 0 ..< mpoRightDim {
                            for betaS in 0 ..< mpsRightDim {
                                var sum: Complex<Double> = .zero
                                for physIn in 0 ..< 2 {
                                    sum = sum + mpo[alphaO, physIn, physOut, betaO] * mps[alphaS, physIn, betaS]
                                }
                                buffer[idx] = sum
                                idx += 1
                            }
                        }
                    }
                }
            }
            count = newLeftDim * 2 * newRightDim
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
            let matrix = tensor.reshapeForSVD(mergingLeft: true)

            let svdResult = SVDDecomposition.decompose(matrix: matrix, truncation: truncation)

            let newRightDim = svdResult.singularValues.count
            let leftDim = tensor.leftBondDimension

            let leftElements = [Complex<Double>](unsafeUninitializedCapacity: leftDim * 2 * newRightDim) { buffer, count in
                var idx = 0
                for alpha in 0 ..< leftDim {
                    for physical in 0 ..< 2 {
                        let rowIdx = alpha * 2 + physical
                        for beta in 0 ..< newRightDim {
                            buffer[idx] = svdResult.u[rowIdx][beta]
                            idx += 1
                        }
                    }
                }
                count = leftDim * 2 * newRightDim
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

            let oldLeftDim = nextTensor.leftBondDimension
            let nextElements = [Complex<Double>](unsafeUninitializedCapacity: newRightDim * 2 * nextRightDim) { buffer, count in
                var idx = 0
                for alpha in 0 ..< newRightDim {
                    let sigma = svdResult.singularValues[alpha]
                    for physical in 0 ..< 2 {
                        for beta in 0 ..< nextRightDim {
                            var sum: Complex<Double> = .zero
                            for gamma in 0 ..< min(oldLeftDim, svdResult.vDagger[0].count) {
                                sum = sum + svdResult.vDagger[alpha][gamma] * nextTensor[gamma, physical, beta]
                            }
                            buffer[idx] = sum * sigma
                            idx += 1
                        }
                    }
                }
                count = newRightDim * 2 * nextRightDim
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
    ) -> TransferMatrix {
        var transfer: TransferMatrix = [[[Complex<Double>]]](
            repeating: [[Complex<Double>]](
                repeating: [Complex<Double>](repeating: .zero, count: ketLeftDim),
                count: mpoLeftDim,
            ),
            count: braLeftDim,
        )
        transfer[0][0][0] = .one
        return transfer
    }

    /// Propagates transfer matrix using flat contiguous arrays for cache-friendly access.
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    private static func propagateTransferMatrixOptimized(
        transfer: TransferMatrix,
        braTensor: MPSTensor,
        mpoTensor: MPOTensor,
        ketTensor: MPSTensor,
    ) -> TransferMatrix {
        let braLeftDim = braTensor.leftBondDimension
        let braRightDim = braTensor.rightBondDimension
        let mpoLeftDim = mpoTensor.leftBondDimension
        let mpoRightDim = mpoTensor.rightBondDimension
        let ketLeftDim = ketTensor.leftBondDimension
        let ketRightDim = ketTensor.rightBondDimension

        let braElements = braTensor.elements
        let mpoElements = mpoTensor.elements
        let ketElements = ketTensor.elements

        let tBraStride = mpoLeftDim * ketLeftDim
        let tMpoStride = ketLeftDim
        let transferFlat = [Complex<Double>](unsafeUninitializedCapacity: braLeftDim * tBraStride) {
            buffer, count in
            for i in 0 ..< braLeftDim {
                let s0 = transfer[i]
                for j in 0 ..< mpoLeftDim {
                    let s1 = s0[j]
                    let base = i * tBraStride + j * tMpoStride
                    for k in 0 ..< ketLeftDim {
                        buffer[base + k] = s1[k]
                    }
                }
            }
            count = braLeftDim * tBraStride
        }

        let iStride0 = 2 * braRightDim * mpoRightDim
        let iStride1 = braRightDim * mpoRightDim
        let iStride2 = mpoRightDim
        var intermediate = [Complex<Double>](unsafeUninitializedCapacity: mpoLeftDim * iStride0) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = mpoLeftDim * iStride0
        }

        for alphaO in 0 ..< mpoLeftDim {
            for alphaBra in 0 ..< braLeftDim {
                for physBra in 0 ..< 2 {
                    for betaBra in 0 ..< braRightDim {
                        let braIdx = alphaBra * (2 * braRightDim) + physBra * braRightDim + betaBra
                        let braConj = Complex(braElements[braIdx].real, -braElements[braIdx].imaginary)

                        for alphaKet in 0 ..< ketLeftDim {
                            let tIdx = alphaBra * tBraStride + alphaO * tMpoStride + alphaKet
                            let transferElement = transferFlat[tIdx]
                            if transferElement.magnitudeSquared < transferMatrixSkipThreshold {
                                continue
                            }

                            let scaledBra = transferElement * braConj

                            for physKet in 0 ..< 2 {
                                let mpoIdx = alphaO * (4 * mpoRightDim) + physKet * (2 * mpoRightDim) + physBra * mpoRightDim
                                let iBase = alphaO * iStride0 + physKet * iStride1 + betaBra * iStride2

                                for betaO in 0 ..< mpoRightDim {
                                    intermediate[iBase + betaO] = intermediate[iBase + betaO] + scaledBra * mpoElements[mpoIdx + betaO]
                                }
                            }
                        }
                    }
                }
            }
        }

        let nStride0 = mpoRightDim * ketRightDim
        let nStride1 = ketRightDim
        var newTransferFlat = [Complex<Double>](unsafeUninitializedCapacity: braRightDim * nStride0) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = braRightDim * nStride0
        }

        for alphaKet in 0 ..< ketLeftDim {
            for physKet in 0 ..< 2 {
                for betaKet in 0 ..< ketRightDim {
                    let ketIdx = alphaKet * (2 * ketRightDim) + physKet * ketRightDim + betaKet
                    let ketElement = ketElements[ketIdx]

                    for betaBra in 0 ..< braRightDim {
                        let nBase = betaBra * nStride0
                        for betaO in 0 ..< mpoRightDim {
                            let nIdx = nBase + betaO * nStride1 + betaKet
                            for alphaO in 0 ..< mpoLeftDim {
                                let iIdx = alphaO * iStride0 + physKet * iStride1 + betaBra * iStride2 + betaO
                                newTransferFlat[nIdx] = newTransferFlat[nIdx] + intermediate[iIdx] * ketElement
                            }
                        }
                    }
                }
            }
        }

        var result: TransferMatrix = []
        result.reserveCapacity(braRightDim)
        for i in 0 ..< braRightDim {
            var mpoSlice: [[Complex<Double>]] = []
            mpoSlice.reserveCapacity(mpoRightDim)
            for j in 0 ..< mpoRightDim {
                let base = i * nStride0 + j * nStride1
                let ketSlice = [Complex<Double>](unsafeUninitializedCapacity: ketRightDim) {
                    buffer, count in
                    for k in 0 ..< ketRightDim {
                        buffer[k] = newTransferFlat[base + k]
                    }
                    count = ketRightDim
                }
                mpoSlice.append(ketSlice)
            }
            result.append(mpoSlice)
        }

        return result
    }
}
