// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Thick-restart Lanczos eigensolver for finding lowest eigenvalue of large Hermitian matrices.
///
/// Implements thick-restart Lanczos algorithm for finding the lowest eigenvalue E₀ and corresponding
/// eigenvector |ψ₀⟩ satisfying H|ψ₀⟩ = E₀|ψ₀⟩ for large Hermitian matrices where direct
/// diagonalization is prohibitively expensive. Used by DMRG when effective Hamiltonian dimension
/// exceeds 1000. Uses modified Gram-Schmidt for reorthogonalization and
/// ``HermitianEigenDecomposition`` for tridiagonal eigenproblem.
///
/// **Example:**
/// ```swift
/// let dimension = 2000
/// let result = await LanczosEigensolver.findLowest(
///     applying: { vector in
///         var output = [Complex<Double>](repeating: .zero, count: dimension)
///         for i in 0..<dimension {
///             output[i] = Complex(Double(i) + 1.0, 0) * vector[i]
///         }
///         return output
///     },
///     dimension: dimension,
///     tolerance: 1e-10
/// )
/// let groundStateEnergy = result.eigenvalues[0]
/// let groundStateVector = result.eigenvectors[0]
/// ```
///
/// - SeeAlso: ``HermitianEigenDecomposition``
/// - SeeAlso: ``HermitianEigenResult``
/// - SeeAlso: ``DMRG``
public enum LanczosEigensolver {
    /// Maximum number of thick-restart iterations before returning best result.
    private static let maxRestarts = 100
    /// Maximum Krylov subspace dimension.
    private static let maxKrylovDimension = 30
    /// Threshold below which Lanczos beta signals invariant subspace breakdown.
    private static let invariantBreakdownThreshold: Double = 1e-14
    /// Threshold for overlap magnitude squared in reorthogonalization.
    private static let overlapMagnitudeSquaredThreshold: Double = 1e-28

    /// Finds lowest eigenvalue and eigenvector using thick-restart Lanczos algorithm.
    ///
    /// Iteratively builds Krylov subspace K = span{v, Hv, H²v, ...} and extracts the lowest
    /// eigenvalue E₀ satisfying H|ψ₀⟩ = E₀|ψ₀⟩ via Ritz approximation. Uses thick-restart
    /// strategy to maintain convergence when Krylov dimension reaches limit. Convergence
    /// achieved when |λ_new - λ_old| < tolerance.
    ///
    /// - Parameters:
    ///   - applying: Async closure that applies Hermitian matrix H to input vector, returning H|v⟩
    ///   - dimension: Dimension of the Hilbert space (size of input/output vectors)
    ///   - tolerance: Convergence threshold for eigenvalue change between iterations
    /// - Returns: ``HermitianEigenResult`` containing lowest eigenvalue and corresponding eigenvector
    /// - Precondition: `dimension` > 0
    /// - Precondition: `tolerance` > 0
    /// - Complexity: O(k * dimension * iterations) where k = Krylov dimension
    ///
    /// - SeeAlso: ``HermitianEigenResult``
    /// - SeeAlso: ``HermitianEigenDecomposition``
    ///
    /// **Example:**
    /// ```swift
    /// let result = await LanczosEigensolver.findLowest(
    ///     applying: hamiltonian.apply,
    ///     dimension: hamiltonian.dimension,
    ///     tolerance: 1e-12
    /// )
    /// ```
    @_optimize(speed)
    public static func findLowest(
        applying: @Sendable ([Complex<Double>]) async -> [Complex<Double>],
        dimension: Int,
        tolerance: Double,
    ) async -> HermitianEigenResult {
        ValidationUtilities.validatePositiveInt(dimension, name: "Lanczos dimension")
        ValidationUtilities.validatePositiveDouble(tolerance, name: "Lanczos tolerance")

        let krylovDimension = computeKrylovDimension(dimension)

        if dimension <= krylovDimension {
            return await solveDirect(applying: applying, dimension: dimension)
        }

        var lanczosVectors = [[Complex<Double>]](repeating: [], count: krylovDimension + 1)
        var alphas = [Double](repeating: 0.0, count: krylovDimension)
        var betas = [Double](repeating: 0.0, count: krylovDimension)

        var initialVector = createRandomNormalizedVector(dimension: dimension)
        var previousEigenvalue = Double.infinity

        for _ in 0 ..< maxRestarts {
            let (converged, eigenvalue, eigenvector) = await lanczosIteration(
                applying: applying,
                krylovDimension: krylovDimension,
                initialVector: initialVector,
                lanczosVectors: &lanczosVectors,
                alphas: &alphas,
                betas: &betas,
                previousEigenvalue: previousEigenvalue,
                tolerance: tolerance,
            )

            if converged {
                return HermitianEigenResult(
                    eigenvalues: [eigenvalue],
                    eigenvectors: [eigenvector],
                )
            }

            previousEigenvalue = eigenvalue
            initialVector = eigenvector
        }

        let finalEigenvector = initialVector
        return HermitianEigenResult(
            eigenvalues: [previousEigenvalue],
            eigenvectors: [finalEigenvector],
        )
    }

    /// Computes optimal Krylov subspace dimension based on problem size.
    @_effects(readonly)
    private static func computeKrylovDimension(_ dimension: Int) -> Int {
        min(maxKrylovDimension, dimension)
    }

    /// Performs single Lanczos iteration building Krylov subspace.
    @_optimize(speed)
    private static func lanczosIteration(
        applying: @Sendable ([Complex<Double>]) async -> [Complex<Double>],
        krylovDimension: Int,
        initialVector: [Complex<Double>],
        lanczosVectors: inout [[Complex<Double>]],
        alphas: inout [Double],
        betas: inout [Double],
        previousEigenvalue: Double,
        tolerance: Double,
    ) async -> (converged: Bool, eigenvalue: Double, eigenvector: [Complex<Double>]) {
        lanczosVectors[0] = initialVector

        for j in 0 ..< krylovDimension {
            var w = await applying(lanczosVectors[j])

            let alpha = realInnerProduct(lanczosVectors[j], w)
            alphas[j] = alpha

            subtractScaled(&w, alpha, lanczosVectors[j])

            if j > 0 {
                subtractScaled(&w, betas[j - 1], lanczosVectors[j - 1])
            }

            reorthogonalize(&w, against: lanczosVectors, count: j + 1)

            let beta = computeNorm(w)
            betas[j] = beta

            if beta < invariantBreakdownThreshold {
                let (eigenvalue, eigenvector) = solveTridiagonal(
                    alphas: alphas,
                    betas: betas,
                    size: j + 1,
                    lanczosVectors: lanczosVectors,
                )
                return (true, eigenvalue, eigenvector)
            }

            if j < krylovDimension - 1 {
                lanczosVectors[j + 1] = normalizeVector(w, norm: beta)
            }
        }

        let (eigenvalue, eigenvector) = solveTridiagonal(
            alphas: alphas,
            betas: betas,
            size: krylovDimension,
            lanczosVectors: lanczosVectors,
        )

        let converged = abs(eigenvalue - previousEigenvalue) < tolerance

        return (converged, eigenvalue, eigenvector)
    }

    /// Solves small eigenproblems by explicit matrix construction and diagonalization.
    @_optimize(speed)
    private static func solveDirect(
        applying: @Sendable ([Complex<Double>]) async -> [Complex<Double>],
        dimension: Int,
    ) async -> HermitianEigenResult {
        var matrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: dimension),
            count: dimension,
        )

        var basisVector = [Complex<Double>](repeating: .zero, count: dimension)
        for j in 0 ..< dimension {
            basisVector[j] = .one
            let column = await applying(basisVector)
            basisVector[j] = .zero
            for i in 0 ..< dimension {
                matrix[i][j] = column[i]
            }
        }

        let fullResult = HermitianEigenDecomposition.decompose(matrix: matrix)

        return HermitianEigenResult(
            eigenvalues: [fullResult.eigenvalues[0]],
            eigenvectors: [fullResult.eigenvectors[0]],
        )
    }

    /// Diagonalizes tridiagonal matrix and reconstructs eigenvector in original basis.
    @_effects(readonly)
    @_optimize(speed)
    private static func solveTridiagonal(
        alphas: [Double],
        betas: [Double],
        size: Int,
        lanczosVectors: [[Complex<Double>]],
    ) -> (eigenvalue: Double, eigenvector: [Complex<Double>]) {
        var tridiagonal = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: size),
            count: size,
        )

        for i in 0 ..< size {
            tridiagonal[i][i] = Complex(alphas[i], 0)
            if i > 0 {
                tridiagonal[i][i - 1] = Complex(betas[i - 1], 0)
                tridiagonal[i - 1][i] = Complex(betas[i - 1], 0)
            }
        }

        let result = HermitianEigenDecomposition.decompose(matrix: tridiagonal)

        let lowestEigenvalue = result.eigenvalues[0]
        let ritzVector = result.eigenvectors[0]

        let dimension = lanczosVectors[0].count
        var eigenvector = [Complex<Double>](repeating: .zero, count: dimension)

        for j in 0 ..< size {
            let coefficient = ritzVector[j]
            addScaled(&eigenvector, coefficient, lanczosVectors[j])
        }

        let norm = computeNorm(eigenvector)
        if norm > invariantBreakdownThreshold {
            eigenvector = normalizeVector(eigenvector, norm: norm)
        }

        return (lowestEigenvalue, eigenvector)
    }

    /// Real-valued inner product Re(v1† · v2) between two complex vectors.
    @_effects(readonly)
    @_optimize(speed)
    private static func realInnerProduct(_ v1: [Complex<Double>], _ v2: [Complex<Double>]) -> Double {
        var result = 0.0
        let count = v1.count
        for i in 0 ..< count {
            result = Double.fusedMultiplyAdd(v1[i].real, v2[i].real,
                     Double.fusedMultiplyAdd(v1[i].imaginary, v2[i].imaginary, result))
        }
        return result
    }

    /// Computes Euclidean norm of a complex vector.
    @_effects(readonly)
    @_optimize(speed)
    private static func computeNorm(_ v: [Complex<Double>]) -> Double {
        var sumSquared = 0.0
        for element in v {
            sumSquared = Double.fusedMultiplyAdd(element.real, element.real,
                         Double.fusedMultiplyAdd(element.imaginary, element.imaginary, sumSquared))
        }
        return Double.squareRoot(of: sumSquared)
    }

    /// Subtracts scaled source vector from target in place.
    @_optimize(speed)
    private static func subtractScaled(_ target: inout [Complex<Double>], _ scalar: Double, _ source: [Complex<Double>]) {
        let count = target.count
        let negScalar = -scalar
        for i in 0 ..< count {
            target[i] = Complex(
                Double.fusedMultiplyAdd(negScalar, source[i].real, target[i].real),
                Double.fusedMultiplyAdd(negScalar, source[i].imaginary, target[i].imaginary)
            )
        }
    }

    /// Adds scaled source vector to target in place.
    @_optimize(speed)
    private static func addScaled(_ target: inout [Complex<Double>], _ scalar: Complex<Double>, _ source: [Complex<Double>]) {
        let count = target.count
        let sr = scalar.real, si = scalar.imaginary
        for i in 0 ..< count {
            let srcR = source[i].real, srcI = source[i].imaginary
            target[i] = Complex(
                Double.fusedMultiplyAdd(sr, srcR, Double.fusedMultiplyAdd(-si, srcI, target[i].real)),
                Double.fusedMultiplyAdd(sr, srcI, Double.fusedMultiplyAdd(si, srcR, target[i].imaginary))
            )
        }
    }

    /// Returns normalized copy of vector given precomputed norm.
    @_effects(readonly)
    @_optimize(speed)
    private static func normalizeVector(_ v: [Complex<Double>], norm: Double) -> [Complex<Double>] {
        let invNorm = 1.0 / norm
        let result = [Complex<Double>](unsafeUninitializedCapacity: v.count) { buffer, count in
            for i in 0 ..< v.count {
                buffer[i] = v[i] * invNorm
            }
            count = v.count
        }
        return result
    }

    /// Reorthogonalizes vector against Lanczos basis using modified Gram-Schmidt.
    @_optimize(speed)
    private static func reorthogonalize(_ w: inout [Complex<Double>], against vectors: [[Complex<Double>]], count: Int) {
        for _ in 0 ..< 2 {
            for j in 0 ..< count {
                let overlap = complexInnerProduct(vectors[j], w)
                if overlap.magnitudeSquared > overlapMagnitudeSquaredThreshold {
                    subtractScaled(&w, overlap, vectors[j])
                }
            }
        }
    }

    /// Complex-valued inner product v1† · v2 between two vectors.
    @_effects(readonly)
    @_optimize(speed)
    private static func complexInnerProduct(_ v1: [Complex<Double>], _ v2: [Complex<Double>]) -> Complex<Double> {
        var rr = 0.0, ri = 0.0
        let count = v1.count
        for i in 0 ..< count {
            let ar = v1[i].real, ai = v1[i].imaginary
            let br = v2[i].real, bi = v2[i].imaginary
            rr = Double.fusedMultiplyAdd(ar, br, Double.fusedMultiplyAdd(ai, bi, rr))
            ri = Double.fusedMultiplyAdd(ar, bi, Double.fusedMultiplyAdd(-ai, br, ri))
        }
        return Complex(rr, ri)
    }

    /// Subtracts complex scalar times source vector from target in place.
    @_optimize(speed)
    private static func subtractScaled(_ target: inout [Complex<Double>], _ scalar: Complex<Double>, _ source: [Complex<Double>]) {
        let count = target.count
        let sr = scalar.real, si = scalar.imaginary
        for i in 0 ..< count {
            let srcR = source[i].real, srcI = source[i].imaginary
            target[i] = Complex(
                Double.fusedMultiplyAdd(-sr, srcR, Double.fusedMultiplyAdd(si, srcI, target[i].real)),
                Double.fusedMultiplyAdd(-sr, srcI, Double.fusedMultiplyAdd(-si, srcR, target[i].imaginary))
            )
        }
    }

    /// Creates deterministic pseudo-random normalized vector for initial guess.
    @_effects(readonly)
    @_optimize(speed)
    private static func createRandomNormalizedVector(dimension: Int) -> [Complex<Double>] {
        let vector = [Complex<Double>](unsafeUninitializedCapacity: dimension) { buffer, count in
            var seed: UInt64 = 12345
            for i in 0 ..< dimension {
                seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                let real = Double(Int64(bitPattern: seed)) / Double(Int64.max)
                seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
                let imag = Double(Int64(bitPattern: seed)) / Double(Int64.max)
                buffer[i] = Complex(real, imag)
            }
            count = dimension
        }

        let norm = computeNorm(vector)
        return normalizeVector(vector, norm: norm)
    }
}
