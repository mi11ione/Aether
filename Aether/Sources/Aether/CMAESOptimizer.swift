// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate
import Foundation

/// CMA-ES optimizer: Covariance Matrix Adaptation Evolution Strategy
///
/// Population-based evolutionary optimization that learns the covariance structure of the
/// objective landscape. Each generation samples λ candidate solutions from a multivariate
/// Gaussian N(m, σ²C), evaluates all candidates, selects the μ best, updates the mean m
/// toward the selected candidates, adapts the step size σ via cumulative step-size adaptation
/// (CSA), and updates the covariance matrix C via rank-1 and rank-μ updates to capture
/// the local curvature.
///
/// The covariance matrix C is maintained in factored form C = BD²Bᵀ via periodic
/// eigendecomposition (LAPACK dsyev), where B contains eigenvectors and D the square root
/// of eigenvalues. Sampling uses x = m + σBDz where z ~ N(0,I). Population size defaults
/// to λ = 4 + ⌊3·ln(n)⌋ (Hansen's recommendation), selection count μ = ⌊λ/2⌋ with
/// log-linear recombination weights.
///
/// No gradients required — handles multimodal, non-convex, noisy landscapes. Population
/// evaluations are independent and trivially parallelizable. O(n²) memory for the covariance
/// matrix, O(n³) per generation for eigendecomposition. Best suited for moderate dimensions
/// (< 100 parameters) where gradient-free robustness is needed and evaluation budget allows
/// population-based search.
///
/// For high dimensions, prefer ``SPSAOptimizer`` (O(1) evaluations per iteration). For smooth
/// unimodal landscapes, ``LBFGSBOptimizer`` converges faster. For very noisy objectives with
/// few parameters, ``COBYLAOptimizer`` may be more sample-efficient.
///
/// **Example:**
/// ```swift
/// let optimizer = CMAESOptimizer(initialStepSize: 0.5)
/// let result = await optimizer.minimize(objective, from: initial, using: .init(), progress: nil)
/// print("Optimal energy: \(result.value)")
/// ```
///
/// - Complexity: O(maxIterations x λ) evaluations, O(n²) memory, O(n³) per eigendecomposition
/// - SeeAlso: ``SPSAOptimizer`` for stochastic gradient-free optimization
/// - SeeAlso: ``NelderMeadOptimizer`` for simplex-based derivative-free optimization
/// - SeeAlso: ``Optimizer`` for protocol definition
@frozen
public struct CMAESOptimizer: Optimizer {
    /// Population size per generation (λ)
    ///
    /// Number of candidate solutions sampled per generation. nil uses Hansen's default
    /// λ = 4 + ⌊3·ln(n)⌋ which scales logarithmically with dimension. Larger populations
    /// explore more broadly but require more evaluations per generation.
    ///
    /// Typical values: nil (auto), 10 (small), 50 (large for multimodal)
    public let populationSize: Int?

    /// Initial step size (σ₀) controlling search radius
    ///
    /// Sets the initial standard deviation of the search distribution. Should be
    /// approximately 1/3 to 1/2 of the expected distance to the optimum. Too large
    /// wastes evaluations exploring irrelevant regions, too small risks premature
    /// convergence.
    ///
    /// Typical values: 0.5 (default), 0.1 (conservative), 1.0 (broad search)
    public let initialStepSize: Double

    /// Create CMA-ES optimizer with population size and initial step size
    ///
    /// - Parameters:
    ///   - populationSize: Candidates per generation, nil for auto 4+⌊3·ln(n)⌋ (default: nil)
    ///   - initialStepSize: Initial search radius σ₀ (default: 0.5)
    /// - Precondition: populationSize > 0 if provided
    /// - Precondition: initialStepSize > 0
    public init(
        populationSize: Int? = nil,
        initialStepSize: Double = 0.5,
    ) {
        if let pop = populationSize {
            ValidationUtilities.validatePositiveInt(pop, name: "populationSize")
        }
        ValidationUtilities.validatePositiveDouble(initialStepSize, name: "initialStepSize")

        self.populationSize = populationSize
        self.initialStepSize = initialStepSize
    }

    /// Minimize objective function using CMA-ES evolution strategy
    ///
    /// Initializes search distribution N(m₀, σ₀²I), iteratively samples population, selects
    /// best candidates, updates mean, step size (CSA), and covariance matrix (rank-1 + rank-μ).
    /// Eigendecomposition refreshes sampling basis periodically.
    ///
    /// **Example:**
    /// ```swift
    /// let optimizer = CMAESOptimizer(initialStepSize: 0.3)
    /// let result = await optimizer.minimize(
    ///     { params in params.map { $0 * $0 }.reduce(0, +) },
    ///     from: [1.0, 1.0],
    ///     using: ConvergenceCriteria(),
    ///     progress: nil
    /// )
    /// ```
    ///
    /// - Precondition: initialParameters is non-empty
    /// - Complexity: O(maxIterations x λ) evaluations, O(n³) per eigendecomposition
    @_optimize(speed)
    @_eagerMove
    public func minimize(
        _ objectiveFunction: @Sendable ([Double]) async -> Double,
        from initialParameters: [Double],
        using convergenceCriteria: ConvergenceCriteria,
        progress: ProgressCallback?,
    ) async -> OptimizerResult {
        let n = initialParameters.count
        ValidationUtilities.validateNonEmpty(initialParameters, name: "initialParameters")

        let lambda = populationSize ?? (4 + Int(3.0 * log(Double(n))))
        let mu = lambda / 2

        var weights = [Double](unsafeUninitializedCapacity: mu) {
            buffer, count in
            for i in 0 ..< mu {
                buffer[i] = log(Double(mu) + 0.5) - log(Double(i + 1))
            }
            count = mu
        }
        var weightSum = 0.0
        vDSP_sveD(weights, 1, &weightSum, vDSP_Length(mu))
        var invWeightSum = 1.0 / weightSum
        vDSP_vsmulD(weights, 1, &invWeightSum, &weights, 1, vDSP_Length(mu))

        var muEffNumerator = 0.0
        vDSP_sveD(weights, 1, &muEffNumerator, vDSP_Length(mu))
        var weightsSq = [Double](unsafeUninitializedCapacity: mu) {
            _, count in count = mu
        }
        vDSP_vsqD(weights, 1, &weightsSq, 1, vDSP_Length(mu))
        var weightsSqSum = 0.0
        vDSP_sveD(weightsSq, 1, &weightsSqSum, vDSP_Length(mu))
        let muEff = (muEffNumerator * muEffNumerator) / weightsSqSum

        let cc = (4.0 + muEff / Double(n)) / (Double(n) + 4.0 + 2.0 * muEff / Double(n))
        let cs = (muEff + 2.0) / (Double(n) + muEff + 5.0)
        let c1 = 2.0 / ((Double(n) + 1.3) * (Double(n) + 1.3) + muEff)
        let cmu = min(
            1.0 - c1,
            2.0 * (muEff - 2.0 + 1.0 / muEff) / ((Double(n) + 2.0) * (Double(n) + 2.0) + muEff),
        )
        let damps = 1.0 + 2.0 * max(0.0, sqrt((muEff - 1.0) / (Double(n) + 1.0)) - 1.0) + cs
        let chiN = sqrt(Double(n)) * (1.0 - 1.0 / (4.0 * Double(n)) + 1.0 / (21.0 * Double(n) * Double(n)))

        var mean = initialParameters
        var sigma = initialStepSize

        var covarianceMatrix = [Double](unsafeUninitializedCapacity: n * n) {
            buffer, count in
            buffer.initialize(repeating: 0.0)
            for i in 0 ..< n {
                buffer[i * n + i] = 1.0
            }
            count = n * n
        }

        var eigenvectors = covarianceMatrix
        var eigenvalues = [Double](unsafeUninitializedCapacity: n) {
            buffer, count in
            buffer.initialize(repeating: 1.0)
            count = n
        }
        var sqrtEigenvalues = eigenvalues

        var pathSigma = [Double](unsafeUninitializedCapacity: n) {
            buffer, count in
            buffer.initialize(repeating: 0.0)
            count = n
        }
        var pathC = [Double](unsafeUninitializedCapacity: n) {
            buffer, count in
            buffer.initialize(repeating: 0.0)
            count = n
        }

        var bestValue = await objectiveFunction(mean)
        var bestParams = mean
        var valueHistory: [Double] = [bestValue]
        var functionEvaluations = 1

        let eigenUpdateInterval = max(1, n / 10)

        for generation in 0 ..< convergenceCriteria.maxIterations {
            if let callback = progress {
                await callback(generation, bestValue)
            }

            var population = [[Double]]()
            population.reserveCapacity(lambda)
            var populationValues = [Double](unsafeUninitializedCapacity: lambda) {
                _, count in
                count = lambda
            }

            for i in 0 ..< lambda {
                let z = CMAESOptimizer.sampleStandardNormal(dimension: n)

                var scaledZ = [Double](unsafeUninitializedCapacity: n) {
                    _, count in count = n
                }
                for j in 0 ..< n {
                    scaledZ[j] = z[j] * sqrtEigenvalues[j]
                }

                var rotated = [Double](unsafeUninitializedCapacity: n) {
                    buffer, count in
                    buffer.initialize(repeating: 0.0)
                    count = n
                }
                for row in 0 ..< n {
                    var dotResult = 0.0
                    vDSP_dotprD(
                        [Double](eigenvectors[(row * n) ..< (row * n + n)]), 1,
                        scaledZ, 1,
                        &dotResult, vDSP_Length(n),
                    )
                    rotated[row] = dotResult
                }

                var candidate = [Double](unsafeUninitializedCapacity: n) {
                    _, count in count = n
                }
                var sigmaVal = sigma
                vDSP_vsmaD(rotated, 1, &sigmaVal, mean, 1, &candidate, 1, vDSP_Length(n))

                population.append(candidate)
                populationValues[i] = await objectiveFunction(candidate)
            }
            functionEvaluations += lambda

            var indices = Array(0 ..< lambda)
            indices.sort { populationValues[$0] < populationValues[$1] }

            if populationValues[indices[0]] < bestValue {
                bestValue = populationValues[indices[0]]
                bestParams = population[indices[0]]
            }

            let previousMean = mean
            mean = [Double](unsafeUninitializedCapacity: n) { buffer, count in
                buffer.initialize(repeating: 0.0)
                count = n
            }
            for i in 0 ..< mu {
                var w = weights[i]
                vDSP_vsmaD(population[indices[i]], 1, &w, mean, 1, &mean, 1, vDSP_Length(n))
            }

            var meanDiff = [Double](unsafeUninitializedCapacity: n) {
                _, count in count = n
            }
            vDSP_vsubD(previousMean, 1, mean, 1, &meanDiff, 1, vDSP_Length(n))

            let invSqrtEigenvalues = [Double](unsafeUninitializedCapacity: n) {
                buffer, count in
                for j in 0 ..< n {
                    buffer[j] = 1.0 / sqrtEigenvalues[j]
                }
                count = n
            }

            var invCHalfMeanDiff = [Double](unsafeUninitializedCapacity: n) {
                buffer, count in
                buffer.initialize(repeating: 0.0)
                count = n
            }
            var projectedDiff = [Double](unsafeUninitializedCapacity: n) {
                _, count in count = n
            }
            for row in 0 ..< n {
                var dotResult = 0.0
                for col in 0 ..< n {
                    dotResult += eigenvectors[col * n + row] * meanDiff[col]
                }
                projectedDiff[row] = dotResult * invSqrtEigenvalues[row]
            }
            for row in 0 ..< n {
                var dotResult = 0.0
                for col in 0 ..< n {
                    dotResult += eigenvectors[row * n + col] * projectedDiff[col]
                }
                invCHalfMeanDiff[row] = dotResult
            }

            let csScale = sqrt(cs * (2.0 - cs) * muEff) / sigma
            var csComplementVal = 1.0 - cs
            vDSP_vsmulD(pathSigma, 1, &csComplementVal, &pathSigma, 1, vDSP_Length(n))
            var csScaleVal = csScale
            vDSP_vsmaD(invCHalfMeanDiff, 1, &csScaleVal, pathSigma, 1, &pathSigma, 1, vDSP_Length(n))

            var pathSigmaNormSq = 0.0
            vDSP_svesqD(pathSigma, 1, &pathSigmaNormSq, vDSP_Length(n))
            let pathSigmaNorm = sqrt(pathSigmaNormSq)
            sigma *= exp((cs / damps) * (pathSigmaNorm / chiN - 1.0))

            let hsig: Double = pathSigmaNorm / sqrt(1.0 - pow(1.0 - cs, 2.0 * Double(generation + 1)))
                < (1.4 + 2.0 / (Double(n) + 1.0)) * chiN ? 1.0 : 0.0

            let ccComplement = 1.0 - cc
            let ccScale = hsig * sqrt(cc * (2.0 - cc) * muEff) / sigma
            var ccComplementVal = ccComplement
            vDSP_vsmulD(pathC, 1, &ccComplementVal, &pathC, 1, vDSP_Length(n))
            var ccScaleVal = ccScale
            vDSP_vsmaD(meanDiff, 1, &ccScaleVal, pathC, 1, &pathC, 1, vDSP_Length(n))

            let oldCovWeight = 1.0 - c1 - cmu + (1.0 - hsig) * c1 * cc * (2.0 - cc)
            var oldWeight = oldCovWeight
            vDSP_vsmulD(covarianceMatrix, 1, &oldWeight, &covarianceMatrix, 1, vDSP_Length(n * n))

            for i in 0 ..< n {
                for j in 0 ..< n {
                    covarianceMatrix[i * n + j] += c1 * pathC[i] * pathC[j]
                }
            }

            for k in 0 ..< mu {
                var diff = [Double](unsafeUninitializedCapacity: n) {
                    _, count in count = n
                }
                vDSP_vsubD(previousMean, 1, population[indices[k]], 1, &diff, 1, vDSP_Length(n))
                let invSigma = 1.0 / sigma
                var scaleFactor = invSigma
                vDSP_vsmulD(diff, 1, &scaleFactor, &diff, 1, vDSP_Length(n))

                let rankWeight = cmu * weights[k]
                for i in 0 ..< n {
                    for j in 0 ..< n {
                        covarianceMatrix[i * n + j] += rankWeight * diff[i] * diff[j]
                    }
                }
            }

            if (generation + 1) % eigenUpdateInterval == 0 {
                CMAESOptimizer.eigendecompose(
                    matrix: &covarianceMatrix,
                    eigenvectors: &eigenvectors,
                    eigenvalues: &eigenvalues,
                    sqrtEigenvalues: &sqrtEigenvalues,
                    dimension: n,
                )
            }

            let generationBestValue = populationValues[indices[0]]
            valueHistory.append(generationBestValue)

            if valueHistory.count >= 2 {
                let prevValue = valueHistory[valueHistory.count - 2]
                if abs(prevValue - generationBestValue) < convergenceCriteria.energyTolerance {
                    return OptimizerResult(
                        parameters: bestParams,
                        value: bestValue,
                        history: valueHistory,
                        iterations: generation + 1,
                        terminationReason: .energyConverged,
                        evaluations: functionEvaluations,
                    )
                }
            }
        }

        return OptimizerResult(
            parameters: bestParams,
            value: bestValue,
            history: valueHistory,
            iterations: convergenceCriteria.maxIterations,
            terminationReason: .maxIterationsReached,
            evaluations: functionEvaluations,
        )
    }

    /// Sample from standard multivariate normal N(0, I) using Box-Muller transform
    @_optimize(speed)
    @_eagerMove
    private static func sampleStandardNormal(dimension: Int) -> [Double] {
        let pairs = (dimension + 1) / 2
        let result = [Double](unsafeUninitializedCapacity: dimension) {
            buffer, count in
            let byteCount = pairs * 2
            let uniformBytes = [UInt64](unsafeUninitializedCapacity: byteCount) {
                buf, cnt in
                arc4random_buf(buf.baseAddress!, byteCount * MemoryLayout<UInt64>.size)
                cnt = byteCount
            }

            for i in 0 ..< pairs {
                let u1 = (Double(uniformBytes[2 * i] & 0x1F_FFFF_FFFF_FFFF) + 1.0) / Double(1 << 53)
                let u2 = Double(uniformBytes[2 * i + 1] & 0x1F_FFFF_FFFF_FFFF) / Double(1 << 53)

                let r = sqrt(-2.0 * log(u1))
                let theta = 2.0 * Double.pi * u2

                buffer[2 * i] = r * cos(theta)
                if 2 * i + 1 < dimension {
                    buffer[2 * i + 1] = r * sin(theta)
                }
            }
            count = dimension
        }
        return result
    }

    /// Eigendecompose symmetric covariance matrix via LAPACK dsyev
    @_optimize(speed)
    private static func eigendecompose(
        matrix: inout [Double],
        eigenvectors: inout [Double],
        eigenvalues: inout [Double],
        sqrtEigenvalues: inout [Double],
        dimension: Int,
    ) {
        eigenvectors = matrix

        var n = __LAPACK_int(dimension)
        var jobz = CChar(Character("V").asciiValue!)
        var uplo = CChar(Character("U").asciiValue!)
        var info = __LAPACK_int(0)
        var lwork = __LAPACK_int(-1)
        var workQuery = 0.0

        dsyev_(&jobz, &uplo, &n, &eigenvectors, &n, &eigenvalues, &workQuery, &lwork, &info)

        lwork = __LAPACK_int(workQuery)
        let workSize = Int(lwork)
        var work = [Double](unsafeUninitializedCapacity: workSize) {
            buf, cnt in
            buf.initialize(repeating: 0.0)
            cnt = workSize
        }

        dsyev_(&jobz, &uplo, &n, &eigenvectors, &n, &eigenvalues, &work, &lwork, &info)

        for i in 0 ..< dimension {
            eigenvalues[i] = max(eigenvalues[i], 1e-20)
            sqrtEigenvalues[i] = sqrt(eigenvalues[i])
        }
    }
}
