// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Thick-restart Lanczos eigensolver for finding lowest eigenvalue of large Hermitian matrices
///
/// Implements thick-restart Lanczos algorithm for finding the lowest eigenvalue and corresponding
/// eigenvector of large Hermitian matrices where direct diagonalization is prohibitively expensive.
/// Used by DMRG when effective Hamiltonian dimension exceeds 1000. Uses modified Gram-Schmidt for
/// reorthogonalization and HermitianEigenDecomposition for tridiagonal eigenproblem.
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
public enum LanczosEigensolver {
    /// Finds lowest eigenvalue and eigenvector using thick-restart Lanczos algorithm
    ///
    /// Iteratively builds Krylov subspace and extracts lowest eigenvalue via Ritz approximation.
    /// Uses thick-restart strategy to maintain convergence when Krylov dimension reaches limit.
    /// Convergence achieved when |lambda_new - lambda_old| < tolerance.
    ///
    /// - Parameters:
    ///   - applying: Async closure that applies Hermitian matrix H to input vector, returning H|v>
    ///   - dimension: Dimension of the Hilbert space (size of input/output vectors)
    ///   - tolerance: Convergence threshold for eigenvalue change between iterations
    /// - Returns: ``HermitianEigenResult`` containing lowest eigenvalue and corresponding eigenvector
    /// - Precondition: dimension must be positive
    /// - Precondition: tolerance must be positive
    /// - Complexity: O(k * dimension * iterations) where k = Krylov dimension
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
        let maxRestarts = 100

        for _ in 0 ..< maxRestarts {
            let (converged, eigenvalue, eigenvector) = await lanczosIteration(
                applying: applying,
                dimension: dimension,
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
    @inlinable
    static func computeKrylovDimension(_ dimension: Int) -> Int {
        min(30, dimension / 2)
    }

    /// Performs single Lanczos iteration building Krylov subspace.
    @_optimize(speed)
    private static func lanczosIteration(
        applying: @Sendable ([Complex<Double>]) async -> [Complex<Double>],
        dimension _: Int,
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

            let alpha = computeRealInnerProduct(lanczosVectors[j], w)
            alphas[j] = alpha

            subtractScaled(&w, alpha, lanczosVectors[j])

            if j > 0 {
                subtractScaled(&w, betas[j - 1], lanczosVectors[j - 1])
            }

            reorthogonalize(&w, against: lanczosVectors, count: j + 1)

            let beta = computeNorm(w)
            betas[j] = beta

            if beta < 1e-14 {
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

        for j in 0 ..< dimension {
            var basisVector = [Complex<Double>](repeating: .zero, count: dimension)
            basisVector[j] = .one
            let column = await applying(basisVector)
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
        if norm > 1e-14 {
            eigenvector = normalizeVector(eigenvector, norm: norm)
        }

        return (lowestEigenvalue, eigenvector)
    }

    /// Computes real part of inner product between two complex vectors.
    @_effects(readonly)
    @_optimize(speed)
    @inlinable
    static func computeRealInnerProduct(_ v1: [Complex<Double>], _ v2: [Complex<Double>]) -> Double {
        var result = 0.0
        let count = v1.count
        for i in 0 ..< count {
            let conj = v1[i].conjugate
            let prod = conj * v2[i]
            result += prod.real
        }
        return result
    }

    /// Computes Euclidean norm of a complex vector.
    @_effects(readonly)
    @_optimize(speed)
    @inlinable
    static func computeNorm(_ v: [Complex<Double>]) -> Double {
        var sumSquared = 0.0
        for element in v {
            sumSquared += element.magnitudeSquared
        }
        return Double.squareRoot(of: sumSquared)
    }

    /// Subtracts scaled source vector from target in place.
    @_optimize(speed)
    @inlinable
    static func subtractScaled(_ target: inout [Complex<Double>], _ scalar: Double, _ source: [Complex<Double>]) {
        let count = target.count
        for i in 0 ..< count {
            target[i] = target[i] - Complex(scalar, 0) * source[i]
        }
    }

    /// Adds scaled source vector to target in place.
    @_optimize(speed)
    @inlinable
    static func addScaled(_ target: inout [Complex<Double>], _ scalar: Complex<Double>, _ source: [Complex<Double>]) {
        let count = target.count
        for i in 0 ..< count {
            target[i] = target[i] + scalar * source[i]
        }
    }

    /// Returns normalized copy of vector given precomputed norm.
    @_effects(readonly)
    @_optimize(speed)
    @inlinable
    static func normalizeVector(_ v: [Complex<Double>], norm: Double) -> [Complex<Double>] {
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
    static func reorthogonalize(_ w: inout [Complex<Double>], against vectors: [[Complex<Double>]], count: Int) {
        for _ in 0 ..< 2 {
            for j in 0 ..< count {
                let overlap = computeComplexInnerProduct(vectors[j], w)
                if overlap.magnitudeSquared > 1e-28 {
                    subtractComplex(&w, overlap, vectors[j])
                }
            }
        }
    }

    /// Computes complex inner product between two vectors.
    @_effects(readonly)
    @_optimize(speed)
    @inlinable
    static func computeComplexInnerProduct(_ v1: [Complex<Double>], _ v2: [Complex<Double>]) -> Complex<Double> {
        var result = Complex<Double>.zero
        let count = v1.count
        for i in 0 ..< count {
            result = result + v1[i].conjugate * v2[i]
        }
        return result
    }

    /// Subtracts complex-scaled source vector from target in place.
    @_optimize(speed)
    @inlinable
    static func subtractComplex(_ target: inout [Complex<Double>], _ scalar: Complex<Double>, _ source: [Complex<Double>]) {
        let count = target.count
        for i in 0 ..< count {
            target[i] = target[i] - scalar * source[i]
        }
    }

    /// Creates deterministic pseudo-random normalized vector for initial guess.
    @_effects(readonly)
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
