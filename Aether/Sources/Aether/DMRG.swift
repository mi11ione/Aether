// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate
import Foundation

/// Density Matrix Renormalization Group algorithm for finding ground states of 1D Hamiltonians.
///
/// Implements two-site DMRG with variational optimization over Matrix Product States by sweeping
/// left-to-right and right-to-left, optimizing pairs of adjacent tensors at each step. The algorithm
/// builds and caches left and right environment tensors for efficient effective Hamiltonian construction.
///
/// DMRG achieves polynomial scaling O(chi^3 * d^2 * n) per sweep where chi is bond dimension, d is
/// physical dimension (2 for qubits), and n is system size. This enables accurate ground state
/// calculations for systems with hundreds of sites when entanglement follows area law.
///
/// Automatically selects between direct and iterative eigensolvers based on effective Hamiltonian
/// size. Subspace expansion adds perturbative corrections controlled by
/// ``DMRGConfiguration/noiseStrength`` to escape local minima.
///
/// **Example:**
/// ```swift
/// let hamiltonian = MatrixProductOperator(observable: isingObservable)
/// let dmrg = DMRG(hamiltonian: hamiltonian, maxBondDimension: 64)
/// let result = await dmrg.findGroundState(from: nil)
/// print("Ground state energy: \(result.groundStateEnergy)")
/// ```
///
/// - Complexity: O(chi^3 * d^2 * n) per sweep where chi = bond dimension, d = 2, n = sites
/// - SeeAlso: ``DMRGConfiguration``
/// - SeeAlso: ``Result``
/// - SeeAlso: ``Progress``
/// - SeeAlso: ``MatrixProductOperator``
/// - SeeAlso: ``MatrixProductState``
public actor DMRG {
    private let hamiltonian: MatrixProductOperator
    private let maxBondDimension: Int
    private let configuration: DMRGConfiguration

    private var currentSweep: Int = 0
    private var currentEnergy: Double = 0.0
    private var currentMaxTruncationError: Double = 0.0

    /// Creates DMRG optimizer for finding ground state of Hamiltonian represented as MPO.
    ///
    /// Initializes the DMRG algorithm with specified Hamiltonian and bond dimension limit.
    /// Configuration controls sweep count, convergence threshold, and subspace expansion.
    ///
    /// **Example:**
    /// ```swift
    /// let h = MatrixProductOperator(observable: heisenbergObservable)
    /// let dmrg = DMRG(hamiltonian: h, maxBondDimension: 128, configuration: .init(maxSweeps: 50))
    /// ```
    ///
    /// - Parameters:
    ///   - hamiltonian: Target Hamiltonian as Matrix Product Operator
    ///   - maxBondDimension: Maximum bond dimension for MPS truncation
    ///   - configuration: DMRG algorithm parameters (default: ``DMRGConfiguration/init()``)
    /// - Complexity: O(1)
    /// - Precondition: maxBondDimension must be positive
    public init(
        hamiltonian: MatrixProductOperator,
        maxBondDimension: Int,
        configuration: DMRGConfiguration = .init(),
    ) {
        ValidationUtilities.validatePositiveInt(maxBondDimension, name: "maxBondDimension")

        self.hamiltonian = hamiltonian
        self.maxBondDimension = maxBondDimension
        self.configuration = configuration
    }

    /// Finds the ground state of the Hamiltonian using two-site DMRG.
    ///
    /// Performs variational optimization by sweeping through the MPS chain, optimizing pairs of
    /// adjacent tensors at each step. Left-to-right sweep followed by right-to-left sweep constitutes
    /// one full sweep. Converges when energy change between sweeps falls below threshold.
    ///
    /// **Example:**
    /// ```swift
    /// let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 64)
    /// let result = await dmrg.findGroundState(from: nil)
    /// print("E0 = \(result.groundStateEnergy), sweeps = \(result.sweeps)")
    /// ```
    ///
    /// - Parameter initial: Initial MPS guess (nil for random initialization)
    /// - Returns: ``Result`` containing ground state energy, MPS, and convergence history
    /// - Complexity: O(sweeps * chi^3 * d^2 * n) where chi = bond dimension, d = 2, n = sites
    /// - SeeAlso: ``Result``
    @_optimize(speed)
    public func findGroundState(from initial: MatrixProductState?) async -> Result {
        let sites = hamiltonian.sites
        var mps = initial ?? initializeRandomMPS(sites: sites)

        mps = canonicalizeMPS(mps, direction: .rightToLeft)

        var leftEnvironments = buildInitialLeftEnvironments(mps: mps)
        var rightEnvironments = buildInitialRightEnvironments(mps: mps)

        var convergenceHistory = [Double]()
        convergenceHistory.reserveCapacity(configuration.maxSweeps)
        var previousEnergy = Double.infinity
        currentSweep = 0

        for sweep in 0 ..< configuration.maxSweeps {
            currentSweep = sweep

            let noiseStrength = configuration.isSubspaceExpansionEnabled ? computeNoise(sweep: sweep) : 0.0

            (mps, leftEnvironments, currentEnergy) = await performLeftToRightSweep(
                mps: mps,
                leftEnvironments: leftEnvironments,
                rightEnvironments: rightEnvironments,
                noiseStrength: noiseStrength,
            )

            rightEnvironments = buildInitialRightEnvironments(mps: mps)

            (mps, rightEnvironments, currentEnergy) = await performRightToLeftSweep(
                mps: mps,
                leftEnvironments: leftEnvironments,
                rightEnvironments: rightEnvironments,
                noiseStrength: noiseStrength,
            )

            leftEnvironments = buildInitialLeftEnvironments(mps: mps)

            convergenceHistory.append(currentEnergy)
            currentMaxTruncationError = mps.truncationStatistics.maxSingleError

            let energyChange = abs(currentEnergy - previousEnergy)
            if energyChange < configuration.convergenceThreshold {
                return Result(
                    groundStateEnergy: currentEnergy,
                    groundState: mps,
                    sweeps: sweep + 1,
                    convergenceHistory: convergenceHistory,
                )
            }

            previousEnergy = currentEnergy
        }

        return Result(
            groundStateEnergy: currentEnergy,
            groundState: mps,
            sweeps: configuration.maxSweeps,
            convergenceHistory: convergenceHistory,
        )
    }

    /// Current optimization progress snapshot.
    ///
    /// Provides read-only access to current sweep number, energy, and maximum truncation error.
    /// Updated after each half-sweep during ``findGroundState(from:)`` execution.
    ///
    /// **Example:**
    /// ```swift
    /// let state = await dmrg.progress
    /// print("Sweep \(state.sweep): E = \(state.energy)")
    /// ```
    ///
    /// - Complexity: O(1)
    /// - SeeAlso: ``Progress``
    public var progress: Progress {
        Progress(
            sweep: currentSweep,
            energy: currentEnergy,
            maxTruncationError: currentMaxTruncationError,
        )
    }

    /// Performs left-to-right sweep optimizing adjacent tensor pairs.
    @_optimize(speed)
    private func performLeftToRightSweep(
        mps: MatrixProductState,
        leftEnvironments: [[[[Complex<Double>]]]],
        rightEnvironments: [[[[Complex<Double>]]]],
        noiseStrength: Double,
    ) async -> (MatrixProductState, [[[[Complex<Double>]]]], Double) {
        var currentMPS = mps
        var currentLeftEnvs = leftEnvironments
        var energy = 0.0

        for site in 0 ..< hamiltonian.sites - 1 {
            let rightEnvIndex = min(site + 2, rightEnvironments.count - 1)
            let rightEnv = rightEnvironments[rightEnvIndex]
            let leftEnv = currentLeftEnvs[site]

            let effectiveH = buildEffectiveHamiltonian(
                leftEnv: leftEnv,
                rightEnv: rightEnv,
                mpoLeft: hamiltonian.tensors[site],
                mpoRight: hamiltonian.tensors[site + 1],
            )

            let (eigenvalue, eigenvector) = await solveEigenproblem(effectiveH: effectiveH)
            energy = eigenvalue

            let (leftTensor, rightTensor, truncationError) = splitTwoSiteTensor(
                eigenvector: eigenvector,
                site: site,
                leftDim: currentMPS.tensors[site].leftBondDimension,
                rightDim: currentMPS.tensors[site + 1].rightBondDimension,
                direction: .leftToRight,
                noiseStrength: noiseStrength,
            )

            currentMPS.updateTensor(at: site, with: leftTensor)
            currentMPS.updateTensor(at: site + 1, with: rightTensor)
            currentMPS.addTruncationError(truncationError)

            if site < hamiltonian.sites - 2, site + 1 < currentLeftEnvs.count {
                currentLeftEnvs[site + 1] = updateLeftEnvironment(
                    leftEnv: currentLeftEnvs[site],
                    mpsTensor: currentMPS.tensors[site],
                    mpoTensor: hamiltonian.tensors[site],
                )
            }
        }

        return (currentMPS, currentLeftEnvs, energy)
    }

    /// Performs right-to-left sweep optimizing adjacent tensor pairs.
    @_optimize(speed)
    private func performRightToLeftSweep(
        mps: MatrixProductState,
        leftEnvironments: [[[[Complex<Double>]]]],
        rightEnvironments: [[[[Complex<Double>]]]],
        noiseStrength: Double,
    ) async -> (MatrixProductState, [[[[Complex<Double>]]]], Double) {
        var currentMPS = mps
        var currentRightEnvs = rightEnvironments
        var energy = 0.0

        for site in stride(from: hamiltonian.sites - 2, through: 0, by: -1) {
            let rightEnvIndex = min(site + 2, currentRightEnvs.count - 1)
            let rightEnv = currentRightEnvs[rightEnvIndex]
            let leftEnv = leftEnvironments[site]

            let effectiveH = buildEffectiveHamiltonian(
                leftEnv: leftEnv,
                rightEnv: rightEnv,
                mpoLeft: hamiltonian.tensors[site],
                mpoRight: hamiltonian.tensors[site + 1],
            )

            let (eigenvalue, eigenvector) = await solveEigenproblem(effectiveH: effectiveH)
            energy = eigenvalue

            let (leftTensor, rightTensor, truncationError) = splitTwoSiteTensor(
                eigenvector: eigenvector,
                site: site,
                leftDim: currentMPS.tensors[site].leftBondDimension,
                rightDim: currentMPS.tensors[site + 1].rightBondDimension,
                direction: .rightToLeft,
                noiseStrength: noiseStrength,
            )

            currentMPS.updateTensor(at: site, with: leftTensor)
            currentMPS.updateTensor(at: site + 1, with: rightTensor)
            currentMPS.addTruncationError(truncationError)

            if site > 0, site + 1 < currentRightEnvs.count {
                let sourceRightEnvIndex = min(site + 2, currentRightEnvs.count - 1)
                currentRightEnvs[site + 1] = updateRightEnvironment(
                    rightEnv: currentRightEnvs[sourceRightEnvIndex],
                    mpsTensor: currentMPS.tensors[site + 1],
                    mpoTensor: hamiltonian.tensors[site + 1],
                )
            }
        }

        return (currentMPS, currentRightEnvs, energy)
    }

    /// Constructs effective Hamiltonian for two-site optimization.
    @_optimize(speed)
    private func buildEffectiveHamiltonian(
        leftEnv: [[[Complex<Double>]]],
        rightEnv: [[[Complex<Double>]]],
        mpoLeft: MPOTensor,
        mpoRight: MPOTensor,
    ) -> [[Complex<Double>]] {
        let leftBraDim = leftEnv.count
        let leftMPODim = leftBraDim > 0 ? leftEnv[0].count : 1
        let leftKetDim = (leftBraDim > 0 && leftMPODim > 0) ? leftEnv[0][0].count : 1

        let rightBraDim = rightEnv.count
        let rightMPODim = rightBraDim > 0 ? rightEnv[0].count : 1
        let rightKetDim = (rightBraDim > 0 && rightMPODim > 0) ? rightEnv[0][0].count : 1

        let physicalDimension = 2
        let effectiveDim = max(1, leftKetDim) * physicalDimension * physicalDimension * max(1, rightKetDim)

        var flatReal = [Double](repeating: 0.0, count: effectiveDim * effectiveDim)
        var flatImag = [Double](repeating: 0.0, count: effectiveDim * effectiveDim)

        let constrainedLeftMPODim = min(leftMPODim, mpoLeft.leftBondDimension)
        let constrainedRightMPODim = min(rightMPODim, mpoRight.rightBondDimension)

        let gammaDim = mpoLeft.rightBondDimension
        let d2 = physicalDimension * physicalDimension
        let safeLeftMPO = max(1, constrainedLeftMPODim)
        let safeRightMPO = max(1, constrainedRightMPODim)
        let mpoCacheStride0 = safeRightMPO * d2 * d2
        let mpoCacheStride1 = d2 * d2
        let mpoCacheStride2 = d2
        var mpoCache = [Complex<Double>](repeating: .zero, count: safeLeftMPO * mpoCacheStride0)

        for alphaO in 0 ..< constrainedLeftMPODim {
            for betaO in 0 ..< constrainedRightMPODim {
                for s1Bra in 0 ..< physicalDimension {
                    for s2Bra in 0 ..< physicalDimension {
                        for s1Ket in 0 ..< physicalDimension {
                            for s2Ket in 0 ..< physicalDimension {
                                var mpoSum: Complex<Double> = .zero
                                for gammaO in 0 ..< gammaDim {
                                    let wLeft = mpoLeft[alphaO, s1Ket, s1Bra, gammaO]
                                    let wRight = mpoRight[gammaO, s2Ket, s2Bra, betaO]
                                    mpoSum = mpoSum + wLeft * wRight
                                }
                                let braPhys = s1Bra * physicalDimension + s2Bra
                                let ketPhys = s1Ket * physicalDimension + s2Ket
                                mpoCache[alphaO * mpoCacheStride0 + betaO * mpoCacheStride1 + braPhys * mpoCacheStride2 + ketPhys] = mpoSum
                            }
                        }
                    }
                }
            }
        }

        let safeRightKetDim = max(1, rightKetDim)
        let safeRightBraDim = max(1, rightBraDim)
        let braStride0 = physicalDimension * physicalDimension * safeRightBraDim
        let braStride1 = physicalDimension * safeRightBraDim
        let ketStride0 = physicalDimension * physicalDimension * safeRightKetDim
        let ketStride1 = physicalDimension * safeRightKetDim

        for alphaBra in 0 ..< leftBraDim {
            guard alphaBra < leftEnv.count else { continue }
            let leftEnvAlpha = leftEnv[alphaBra]

            for alphaO in 0 ..< constrainedLeftMPODim {
                guard alphaO < leftEnvAlpha.count else { continue }
                let leftEnvAlphaO = leftEnvAlpha[alphaO]

                for alphaKet in 0 ..< leftKetDim {
                    guard alphaKet < leftEnvAlphaO.count else { continue }
                    let leftVal = leftEnvAlphaO[alphaKet]
                    if leftVal.magnitudeSquared < 1e-30 { continue }

                    for betaBra in 0 ..< rightBraDim {
                        guard betaBra < rightEnv.count else { continue }
                        let rightEnvBeta = rightEnv[betaBra]

                        for betaO in 0 ..< constrainedRightMPODim {
                            guard betaO < rightEnvBeta.count else { continue }
                            let rightEnvBetaO = rightEnvBeta[betaO]

                            for betaKet in 0 ..< rightKetDim {
                                guard betaKet < rightEnvBetaO.count else { continue }
                                let rightVal = rightEnvBetaO[betaKet]
                                if rightVal.magnitudeSquared < 1e-30 { continue }

                                let envProduct = leftVal * rightVal

                                for braPhys in 0 ..< (physicalDimension * physicalDimension) {
                                    let s1Bra = braPhys / physicalDimension
                                    let s2Bra = braPhys % physicalDimension
                                    let braIdx = alphaBra * braStride0 + s1Bra * braStride1 + s2Bra * safeRightBraDim + betaBra

                                    guard braIdx < effectiveDim else { continue }
                                    let braOffset = braIdx * effectiveDim

                                    for ketPhys in 0 ..< (physicalDimension * physicalDimension) {
                                        let s1Ket = ketPhys / physicalDimension
                                        let s2Ket = ketPhys % physicalDimension
                                        let ketIdx = alphaKet * ketStride0 + s1Ket * ketStride1 + s2Ket * safeRightKetDim + betaKet

                                        guard ketIdx < effectiveDim else { continue }
                                        let mpoVal = mpoCache[alphaO * mpoCacheStride0 + betaO * mpoCacheStride1 + braPhys * mpoCacheStride2 + ketPhys]
                                        let contribution = envProduct * mpoVal
                                        let flatIdx = braOffset + ketIdx
                                        guard flatIdx < flatReal.count else { continue }
                                        flatReal[flatIdx] += contribution.real
                                        flatImag[flatIdx] += contribution.imaginary
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        var effectiveH = [[Complex<Double>]]()
        effectiveH.reserveCapacity(effectiveDim)
        for i in 0 ..< effectiveDim {
            let rowOffset = i * effectiveDim
            let row = [Complex<Double>](unsafeUninitializedCapacity: effectiveDim) { buffer, count in
                for j in 0 ..< effectiveDim {
                    buffer.initializeElement(at: j, to: Complex(flatReal[rowOffset + j], flatImag[rowOffset + j]))
                }
                count = effectiveDim
            }
            effectiveH.append(row)
        }

        return effectiveH
    }

    /// Solves eigenproblem using direct or Lanczos method based on dimension.
    private func solveEigenproblem(effectiveH: [[Complex<Double>]]) async -> (Double, [Complex<Double>]) {
        let dimension = effectiveH.count

        if dimension <= 1000 {
            let result = HermitianEigenDecomposition.decompose(matrix: effectiveH)
            return (result.eigenvalues[0], result.eigenvectors[0])
        }

        let flatH = Self.flattenHamiltonian(effectiveH)
        let applyH: @Sendable ([Complex<Double>]) async -> [Complex<Double>] = { vector in
            Self.applyFlatHamiltonian(flatH, dimension: dimension, to: vector)
        }

        let result = await LanczosEigensolver.findLowest(
            applying: applyH,
            dimension: dimension,
            tolerance: configuration.convergenceThreshold * 0.1,
        )

        return (result.eigenvalues[0], result.eigenvectors[0])
    }

    /// Flattens 2D complex matrix into interleaved real/imaginary for BLAS.
    private static func flattenHamiltonian(_ H: [[Complex<Double>]]) -> [Double] {
        let n = H.count
        return [Double](unsafeUninitializedCapacity: n * n * 2) { buffer, count in
            var idx = 0
            for i in 0 ..< n {
                let row = H[i]
                for j in 0 ..< n {
                    buffer[idx] = row[j].real
                    buffer[idx + 1] = row[j].imaginary
                    idx += 2
                }
            }
            count = n * n * 2
        }
    }

    /// Applies flattened Hamiltonian to vector using cblas_zgemv.
    @_optimize(speed)
    private static func applyFlatHamiltonian(
        _ flatH: [Double],
        dimension n: Int,
        to vector: [Complex<Double>],
    ) -> [Complex<Double>] {
        var flatVec = [Double](unsafeUninitializedCapacity: n * 2) { buffer, count in
            for j in 0 ..< n {
                buffer[j * 2] = vector[j].real
                buffer[j * 2 + 1] = vector[j].imaginary
            }
            count = n * 2
        }
        var flatResult = [Double](unsafeUninitializedCapacity: n * 2) { _, count in count = n * 2 }
        var alpha: (Double, Double) = (1.0, 0.0)
        var beta: (Double, Double) = (0.0, 0.0)
        // Safety: baseAddress is non-nil for all buffers since n > 1000 guarantees non-empty arrays
        flatH.withUnsafeBufferPointer { hPtr in
            flatVec.withUnsafeMutableBufferPointer { vecPtr in
                flatResult.withUnsafeMutableBufferPointer { resPtr in
                    withUnsafePointer(to: &alpha) { alphaPtr in
                        withUnsafePointer(to: &beta) { betaPtr in
                            cblas_zgemv(
                                CblasRowMajor, CblasNoTrans,
                                Int32(n), Int32(n),
                                OpaquePointer(alphaPtr),
                                OpaquePointer(hPtr.baseAddress!), Int32(n),
                                OpaquePointer(vecPtr.baseAddress!), 1,
                                OpaquePointer(betaPtr),
                                OpaquePointer(resPtr.baseAddress!), 1,
                            )
                        }
                    }
                }
            }
        }
        return [Complex<Double>](unsafeUninitializedCapacity: n) { buffer, count in
            for i in 0 ..< n {
                buffer.initializeElement(at: i, to: Complex(flatResult[i * 2], flatResult[i * 2 + 1]))
            }
            count = n
        }
    }

    /// Splits optimized two-site tensor via SVD into left and right MPS tensors.
    @_optimize(speed)
    private func splitTwoSiteTensor(
        eigenvector: [Complex<Double>],
        site: Int,
        leftDim: Int,
        rightDim: Int,
        direction: DMRGSweepDirection,
        noiseStrength: Double,
    ) -> (MPSTensor, MPSTensor, Double) {
        let physicalDimension = 2

        var matrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: physicalDimension * rightDim),
            count: leftDim * physicalDimension,
        )

        let vecStride0 = physicalDimension * physicalDimension * rightDim
        let vecStride1 = physicalDimension * rightDim
        for alpha in 0 ..< leftDim {
            let alphaOffset = alpha * vecStride0
            for s1 in 0 ..< physicalDimension {
                let s1Offset = alphaOffset + s1 * vecStride1
                let row = alpha * physicalDimension + s1
                for s2 in 0 ..< physicalDimension {
                    let s2Offset = s1Offset + s2 * rightDim
                    for beta in 0 ..< rightDim {
                        let vecIdx = s2Offset + beta
                        let col = s2 * rightDim + beta
                        if vecIdx < eigenvector.count {
                            matrix[row][col] = eigenvector[vecIdx]
                        }
                    }
                }
            }
        }

        if noiseStrength > 0 {
            matrix = addSubspaceExpansion(matrix: matrix, strength: noiseStrength)
        }

        let svdResult = SVDDecomposition.decompose(
            matrix: matrix,
            truncation: .maxBondDimension(maxBondDimension),
        )

        let newBondDim = svdResult.singularValues.count

        var leftElements = [Complex<Double>]()
        leftElements.reserveCapacity(leftDim * physicalDimension * newBondDim)

        var rightElements = [Complex<Double>]()
        rightElements.reserveCapacity(newBondDim * physicalDimension * rightDim)

        if direction == .leftToRight {
            for alpha in 0 ..< leftDim {
                for s in 0 ..< physicalDimension {
                    for gamma in 0 ..< newBondDim {
                        let row = alpha * physicalDimension + s
                        leftElements.append(svdResult.u[row][gamma])
                    }
                }
            }

            for gamma in 0 ..< newBondDim {
                let sv = Complex(svdResult.singularValues[gamma], 0)
                for s in 0 ..< physicalDimension {
                    for beta in 0 ..< rightDim {
                        let col = s * rightDim + beta
                        rightElements.append(sv * svdResult.vDagger[gamma][col])
                    }
                }
            }
        } else {
            for alpha in 0 ..< leftDim {
                for s in 0 ..< physicalDimension {
                    for gamma in 0 ..< newBondDim {
                        let row = alpha * physicalDimension + s
                        let sv = Complex(svdResult.singularValues[gamma], 0)
                        leftElements.append(svdResult.u[row][gamma] * sv)
                    }
                }
            }

            for gamma in 0 ..< newBondDim {
                for s in 0 ..< physicalDimension {
                    for beta in 0 ..< rightDim {
                        let col = s * rightDim + beta
                        rightElements.append(svdResult.vDagger[gamma][col])
                    }
                }
            }
        }

        let leftTensor = MPSTensor(
            leftBondDimension: leftDim,
            rightBondDimension: newBondDim,
            site: site,
            elements: leftElements,
        )

        let rightTensor = MPSTensor(
            leftBondDimension: newBondDim,
            rightBondDimension: rightDim,
            site: site + 1,
            elements: rightElements,
        )

        return (leftTensor, rightTensor, svdResult.truncationError)
    }

    /// Builds initial left environment tensors from MPS.
    private func buildInitialLeftEnvironments(mps: MatrixProductState) -> [[[[Complex<Double>]]]] {
        let sites = hamiltonian.sites
        var envs = [[[[Complex<Double>]]]](repeating: [], count: sites + 1)

        envs[0] = [[[.one]]]

        for site in 0 ..< sites {
            envs[site + 1] = updateLeftEnvironment(
                leftEnv: envs[site],
                mpsTensor: mps.tensors[site],
                mpoTensor: hamiltonian.tensors[site],
            )
        }

        return envs
    }

    /// Builds initial right environment tensors from MPS.
    private func buildInitialRightEnvironments(mps: MatrixProductState) -> [[[[Complex<Double>]]]] {
        let sites = hamiltonian.sites

        var envs = [[[[Complex<Double>]]]](repeating: [[[.one]]], count: sites + 2)
        envs[sites + 1] = [[[.one]]]
        envs[sites] = [[[.one]]]

        for site in stride(from: sites - 1, through: 0, by: -1) {
            let rightEnvIndex = min(site + 2, sites + 1)
            envs[site + 1] = updateRightEnvironment(
                rightEnv: envs[rightEnvIndex],
                mpsTensor: mps.tensors[site],
                mpoTensor: hamiltonian.tensors[site],
            )
        }

        return envs
    }

    /// Updates left environment by contracting with MPS and MPO tensors.
    @_optimize(speed)
    private func updateLeftEnvironment(
        leftEnv: [[[Complex<Double>]]],
        mpsTensor: MPSTensor,
        mpoTensor: MPOTensor,
    ) -> [[[Complex<Double>]]] {
        let newBraDim = mpsTensor.rightBondDimension
        let newMPODim = mpoTensor.rightBondDimension
        let newKetDim = mpsTensor.rightBondDimension

        var newEnv = [[[Complex<Double>]]](
            repeating: [[Complex<Double>]](
                repeating: [Complex<Double>](repeating: .zero, count: newKetDim),
                count: newMPODim,
            ),
            count: newBraDim,
        )

        let oldBraDim = min(leftEnv.count, mpsTensor.leftBondDimension)
        let oldMPODim = min(leftEnv.isEmpty ? 1 : leftEnv[0].count, mpoTensor.leftBondDimension)
        let oldKetDim = min(leftEnv.isEmpty ? 1 : (leftEnv[0].isEmpty ? 1 : leftEnv[0][0].count), mpsTensor.leftBondDimension)

        for alphaBra in 0 ..< oldBraDim {
            for alphaO in 0 ..< oldMPODim {
                for alphaKet in 0 ..< oldKetDim {
                    let leftVal = leftEnv[alphaBra][alphaO][alphaKet]
                    if leftVal.magnitudeSquared < 1e-30 { continue }

                    for sBra in 0 ..< 2 {
                        for sKet in 0 ..< 2 {
                            for betaBra in 0 ..< newBraDim {
                                let mpsConj = mpsTensor[alphaBra, sBra, betaBra].conjugate
                                let leftTimesConj = leftVal * mpsConj
                                for betaO in 0 ..< newMPODim {
                                    let mpo = mpoTensor[alphaO, sKet, sBra, betaO]
                                    let partial = leftTimesConj * mpo
                                    for betaKet in 0 ..< newKetDim {
                                        let mpsKet = mpsTensor[alphaKet, sKet, betaKet]
                                        newEnv[betaBra][betaO][betaKet] = newEnv[betaBra][betaO][betaKet] + partial * mpsKet
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return newEnv
    }

    /// Updates right environment by contracting with MPS and MPO tensors.
    @_optimize(speed)
    private func updateRightEnvironment(
        rightEnv: [[[Complex<Double>]]],
        mpsTensor: MPSTensor,
        mpoTensor: MPOTensor,
    ) -> [[[Complex<Double>]]] {
        let newBraDim = mpsTensor.leftBondDimension
        let newMPODim = mpoTensor.leftBondDimension
        let newKetDim = mpsTensor.leftBondDimension

        var newEnv = [[[Complex<Double>]]](
            repeating: [[Complex<Double>]](
                repeating: [Complex<Double>](repeating: .zero, count: newKetDim),
                count: newMPODim,
            ),
            count: newBraDim,
        )

        let oldBraDim = min(rightEnv.count, mpsTensor.rightBondDimension)
        let oldMPODim = min(rightEnv.isEmpty ? 1 : rightEnv[0].count, mpoTensor.rightBondDimension)
        let oldKetDim = min(rightEnv.isEmpty ? 1 : (rightEnv[0].isEmpty ? 1 : rightEnv[0][0].count), mpsTensor.rightBondDimension)

        for betaBra in 0 ..< oldBraDim {
            for betaO in 0 ..< oldMPODim {
                for betaKet in 0 ..< oldKetDim {
                    let rightVal = rightEnv[betaBra][betaO][betaKet]
                    if rightVal.magnitudeSquared < 1e-30 { continue }

                    for sBra in 0 ..< 2 {
                        for sKet in 0 ..< 2 {
                            for alphaBra in 0 ..< newBraDim {
                                let mpsConj = mpsTensor[alphaBra, sBra, betaBra].conjugate
                                let rightTimesConj = rightVal * mpsConj
                                for alphaO in 0 ..< newMPODim {
                                    let mpo = mpoTensor[alphaO, sKet, sBra, betaO]
                                    let partial = rightTimesConj * mpo
                                    for alphaKet in 0 ..< newKetDim {
                                        let mpsKet = mpsTensor[alphaKet, sKet, betaKet]
                                        newEnv[alphaBra][alphaO][alphaKet] = newEnv[alphaBra][alphaO][alphaKet] + partial * mpsKet
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return newEnv
    }

    /// Creates random normalized MPS as initial guess for optimization.
    private func initializeRandomMPS(sites: Int) -> MatrixProductState {
        var mps = MatrixProductState(qubits: sites, maxBondDimension: maxBondDimension)

        var seed: UInt64 = 42

        for site in 0 ..< sites {
            let leftDim = mps.tensors[site].leftBondDimension
            let rightDim = mps.tensors[site].rightBondDimension

            let elementCount = leftDim * 2 * rightDim
            var elements = [Complex<Double>](unsafeUninitializedCapacity: elementCount) { buffer, count in
                for idx in 0 ..< elementCount {
                    seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                    let real = Double(Int64(bitPattern: seed)) / Double(Int64.max)
                    seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                    let imag = Double(Int64(bitPattern: seed)) / Double(Int64.max)
                    buffer.initializeElement(at: idx, to: Complex(real, imag))
                }
                count = elementCount
            }

            let tensor = MPSTensor(
                leftBondDimension: leftDim,
                rightBondDimension: rightDim,
                site: site,
                elements: elements,
            )
            mps.updateTensor(at: site, with: tensor)
        }

        mps.normalize()
        return mps
    }

    /// Brings MPS into canonical form for specified sweep direction.
    private func canonicalizeMPS(_ mps: MatrixProductState, direction: DMRGSweepDirection) -> MatrixProductState {
        var result = mps

        if direction == .rightToLeft {
            for site in stride(from: mps.qubits - 1, through: 1, by: -1) {
                let tensor = result.tensors[site]
                let matrix = tensor.reshapeForSVD(mergeLeft: false)

                let svdResult = SVDDecomposition.decompose(
                    matrix: matrix,
                    truncation: .maxBondDimension(maxBondDimension),
                )

                let newLeftDim = svdResult.singularValues.count
                let rightDim = tensor.rightBondDimension

                var rightElements = [Complex<Double>]()
                rightElements.reserveCapacity(newLeftDim * 2 * rightDim)

                for gamma in 0 ..< newLeftDim {
                    for s in 0 ..< 2 {
                        for beta in 0 ..< rightDim {
                            let col = s * rightDim + beta
                            rightElements.append(svdResult.vDagger[gamma][col])
                        }
                    }
                }

                let newRightTensor = MPSTensor(
                    leftBondDimension: newLeftDim,
                    rightBondDimension: rightDim,
                    site: site,
                    elements: rightElements,
                )
                result.updateTensor(at: site, with: newRightTensor)

                let leftTensor = result.tensors[site - 1]
                let oldLeftDim = leftTensor.leftBondDimension

                var leftElements = [Complex<Double>]()
                leftElements.reserveCapacity(oldLeftDim * 2 * newLeftDim)

                for alpha in 0 ..< oldLeftDim {
                    for s in 0 ..< 2 {
                        for gamma in 0 ..< newLeftDim {
                            let sv = Complex(svdResult.singularValues[gamma], 0)
                            var sum: Complex<Double> = .zero
                            for oldBeta in 0 ..< leftTensor.rightBondDimension {
                                let leftVal = leftTensor[alpha, s, oldBeta]
                                if oldBeta < svdResult.u.count, gamma < svdResult.u[oldBeta].count {
                                    sum = sum + leftVal * svdResult.u[oldBeta][gamma]
                                }
                            }
                            leftElements.append(sum * sv)
                        }
                    }
                }

                let newLeftTensor = MPSTensor(
                    leftBondDimension: oldLeftDim,
                    rightBondDimension: newLeftDim,
                    site: site - 1,
                    elements: leftElements,
                )
                result.updateTensor(at: site - 1, with: newLeftTensor)
                result.addTruncationError(svdResult.truncationError)
            }
        }

        result.normalize()
        return result
    }

    /// Adds perturbative noise to tensor for subspace expansion.
    @_optimize(speed)
    private func addSubspaceExpansion(matrix: [[Complex<Double>]], strength: Double) -> [[Complex<Double>]] {
        let rows = matrix.count
        let cols = matrix.isEmpty ? 0 : matrix[0].count

        guard rows > 0, cols > 0 else { return matrix }

        let fullSVD = SVDDecomposition.decompose(matrix: matrix, truncation: .none)
        let fullSingularValues = fullSVD.singularValues
        let keptCount = min(maxBondDimension, fullSingularValues.count)

        guard keptCount < fullSingularValues.count else { return matrix }

        var result = matrix

        for k in keptCount ..< fullSingularValues.count {
            let discardedWeight = fullSingularValues[k] * strength
            guard k < fullSVD.u.count, k < fullSVD.vDagger.count else { continue }
            let scaledWeight = Complex(discardedWeight, 0)
            let vRow = fullSVD.vDagger[k]
            for i in 0 ..< rows {
                let scaled = fullSVD.u[i][k] * scaledWeight
                for j in 0 ..< cols {
                    result[i][j] = result[i][j] + scaled * vRow[j]
                }
            }
        }

        return result
    }

    /// Computes decaying noise strength for current sweep.
    private func computeNoise(sweep: Int) -> Double {
        let baseNoise = configuration.noiseStrength
        let decayFactor = pow(0.5, Double(sweep))
        return baseNoise * decayFactor
    }

    /// Result of DMRG ground state optimization.
    ///
    /// Contains the computed ground state energy (variational upper bound), the optimized MPS
    /// representing the ground state, total sweeps performed, and energy convergence history.
    ///
    /// **Example:**
    /// ```swift
    /// let result = await dmrg.findGroundState(from: nil)
    /// print("E0 = \(result.groundStateEnergy)")
    /// print("Converged in \(result.sweeps) sweeps")
    /// ```
    ///
    /// - SeeAlso: ``DMRG``
    /// - SeeAlso: ``MatrixProductState``
    @frozen
    public struct Result: Sendable {
        /// Ground state energy (variational upper bound on true E0).
        public let groundStateEnergy: Double

        /// Optimized Matrix Product State representing the ground state.
        public let groundState: MatrixProductState

        /// Total number of DMRG sweeps performed.
        public let sweeps: Int

        /// Energy values at end of each sweep for convergence analysis.
        public let convergenceHistory: [Double]

        /// Creates a DMRG result with specified values.
        ///
        /// - Parameters:
        ///   - groundStateEnergy: Computed ground state energy
        ///   - groundState: Optimized MPS
        ///   - sweeps: Number of sweeps performed
        ///   - convergenceHistory: Energy history per sweep
        init(
            groundStateEnergy: Double,
            groundState: MatrixProductState,
            sweeps: Int,
            convergenceHistory: [Double],
        ) {
            self.groundStateEnergy = groundStateEnergy
            self.groundState = groundState
            self.sweeps = sweeps
            self.convergenceHistory = convergenceHistory
        }
    }

    /// Snapshot of DMRG optimization progress.
    ///
    /// Captures current sweep number, energy estimate, and maximum truncation error.
    /// Updated after each half-sweep during optimization.
    ///
    /// **Example:**
    /// ```swift
    /// let progress = await dmrg.progress
    /// print("Sweep \(progress.sweep): E = \(progress.energy), truncation = \(progress.maxTruncationError)")
    /// ```
    ///
    /// - SeeAlso: ``DMRG``
    @frozen
    public struct Progress: Sendable {
        /// Current sweep number (0-indexed).
        public let sweep: Int

        /// Current energy estimate.
        public let energy: Double

        /// Maximum truncation error encountered in current sweep.
        public let maxTruncationError: Double

        /// Creates a progress snapshot.
        ///
        /// - Parameters:
        ///   - sweep: Current sweep number
        ///   - energy: Current energy value
        ///   - maxTruncationError: Maximum truncation error
        init(sweep: Int, energy: Double, maxTruncationError: Double) {
            self.sweep = sweep
            self.energy = energy
            self.maxTruncationError = maxTruncationError
        }
    }
}
