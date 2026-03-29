// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Rotosolve optimizer with analytically optimal per-parameter updates
///
/// Exploits the sinusoidal structure of parameterized quantum circuits to find the globally
/// optimal angle for each parameter analytically. For single-qubit rotation gates (Rx, Ry, Rz),
/// the energy landscape E(θ_k) with all other parameters fixed is exactly sinusoidal:
/// E(θ_k) = A sin(θ_k + B) + C. Three function evaluations determine A, B, C uniquely,
/// yielding the closed-form optimum θ* = θ₀ - π/2 - atan2(2E₀ - E₊ - E₋, √3(E₊ - E₋))
/// where E₀, E₊, E₋ are evaluations at θ₀, θ₀ + 2π/3, θ₀ - 2π/3.
///
/// Each sweep through all p parameters requires exactly 3p function evaluations — no gradient
/// circuits needed. Per-parameter optimization is sequential (coordinate descent), with each
/// parameter set to its exact optimum before moving to the next. This guarantees monotonic
/// energy decrease per sweep. Convergence occurs when the total energy change across a full
/// sweep falls below the energy tolerance.
///
/// The Nakanishi-Fujii-Todo (NFT) extension generalizes to two-parameter subspaces where
/// E(θ_i, θ_j) is a bisinusoidal surface, finding the joint optimum from a structured set
/// of evaluations. This captures parameter correlations missed by sequential single-parameter
/// optimization.
///
/// Best suited for variational circuits with single-qubit rotation gates where the sinusoidal
/// assumption holds exactly. Not applicable to multi-qubit parameterized gates or non-rotation
/// parameterizations. For general optimization, prefer ``AdamOptimizer`` or ``LBFGSBOptimizer``.
///
/// **Example:**
/// ```swift
/// let optimizer = RotosolveOptimizer()
/// let result = await optimizer.minimize(objective, from: initial, using: .init(), progress: nil)
/// print("Optimal energy: \(result.value)")
/// ```
///
/// - Complexity: O(maxIterations x 3p) evaluations where p = parameter count
/// - SeeAlso: ``GradientDescentOptimizer`` for gradient-based alternative
/// - SeeAlso: ``AdamOptimizer`` for adaptive gradient method
/// - SeeAlso: ``Optimizer`` for protocol definition
@frozen
public struct RotosolveOptimizer: Optimizer {
    /// Enable Nakanishi-Fujii-Todo extension for two-parameter subspace optimization
    ///
    /// When enabled, pairs adjacent parameters and optimizes each pair via iterative
    /// refinement: optimize first parameter, then second, then re-optimize first to capture
    /// correlation. Falls back to single-parameter Rotosolve for the last parameter if
    /// the count is odd.
    ///
    /// Recommended: false (default) for most circuits, true for strongly correlated parameters
    public let isNFTEnabled: Bool

    private let twoPiOverThree = 2.0 * Double.pi / 3.0
    private let sqrtThree = 1.7320508075688772

    /// Create Rotosolve optimizer with optional NFT extension
    ///
    /// - Parameter isNFTEnabled: Enable two-parameter NFT subspace optimization (default: false)
    public init(isNFTEnabled: Bool = false) {
        self.isNFTEnabled = isNFTEnabled
    }

    /// Minimize objective function using analytical per-parameter Rotosolve
    ///
    /// Sweeps through each parameter, evaluates the objective at three equidistant angles
    /// separated by 2π/3, fits the sinusoidal curve, and sets each parameter to its analytical
    /// optimum. Repeats sweeps until convergence.
    ///
    /// **Example:**
    /// ```swift
    /// let optimizer = RotosolveOptimizer()
    /// let result = await optimizer.minimize(
    ///     { params in params.map { $0 * $0 }.reduce(0, +) },
    ///     from: [1.0, 1.0],
    ///     using: ConvergenceCriteria(),
    ///     progress: nil
    /// )
    /// ```
    ///
    /// - Precondition: initialParameters is non-empty
    /// - Complexity: O(maxIterations x 3p) evaluations for standard, O(maxIterations x 5p/2) for NFT
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

        var params = initialParameters
        var currentValue = await objectiveFunction(params)
        var valueHistory: [Double] = [currentValue]
        var functionEvaluations = 1

        for iteration in 0 ..< convergenceCriteria.maxIterations {
            if let callback = progress {
                await callback(iteration, currentValue)
            }

            let sweepStartValue = currentValue

            if isNFTEnabled {
                var i = 0
                while i + 1 < n {
                    let nftResult = await optimizePairNFT(
                        objectiveFunction: objectiveFunction,
                        params: &params,
                        index1: i,
                        index2: i + 1,
                    )
                    currentValue = nftResult.value
                    functionEvaluations += nftResult.evaluations
                    i += 2
                }
                if i < n {
                    let singleResult = await optimizeSingleParameter(
                        objectiveFunction: objectiveFunction,
                        params: &params,
                        index: i,
                    )
                    currentValue = singleResult.value
                    functionEvaluations += singleResult.evaluations
                }
            } else {
                for i in 0 ..< n {
                    let result = await optimizeSingleParameter(
                        objectiveFunction: objectiveFunction,
                        params: &params,
                        index: i,
                    )
                    currentValue = result.value
                    functionEvaluations += result.evaluations
                }
            }

            valueHistory.append(currentValue)

            if abs(sweepStartValue - currentValue) < convergenceCriteria.energyTolerance {
                return OptimizerResult(
                    parameters: params,
                    value: currentValue,
                    history: valueHistory,
                    iterations: iteration + 1,
                    terminationReason: .energyConverged,
                    evaluations: functionEvaluations,
                )
            }
        }

        return OptimizerResult(
            parameters: params,
            value: currentValue,
            history: valueHistory,
            iterations: convergenceCriteria.maxIterations,
            terminationReason: .maxIterationsReached,
            evaluations: functionEvaluations,
        )
    }

    private struct ParameterOptResult {
        let value: Double
        let evaluations: Int
    }

    /// Optimize a single parameter analytically via three-point sinusoidal fit
    @_optimize(speed)
    private func optimizeSingleParameter(
        objectiveFunction: @Sendable ([Double]) async -> Double,
        params: inout [Double],
        index: Int,
    ) async -> ParameterOptResult {
        let theta0 = params[index]

        let e0 = await objectiveFunction(params)

        params[index] = theta0 + twoPiOverThree
        let ePlus = await objectiveFunction(params)

        params[index] = theta0 - twoPiOverThree
        let eMinus = await objectiveFunction(params)

        let sinComponent = 2.0 * e0 - ePlus - eMinus
        let cosComponent = sqrtThree * (ePlus - eMinus)

        let thetaStar = theta0 - Double.pi / 2.0 - atan2(sinComponent, cosComponent)
        params[index] = thetaStar

        let optimalValue = await objectiveFunction(params)
        return ParameterOptResult(value: optimalValue, evaluations: 4)
    }

    /// Optimize a pair of parameters via iterative refinement
    @_optimize(speed)
    private func optimizePairNFT(
        objectiveFunction: @Sendable ([Double]) async -> Double,
        params: inout [Double],
        index1: Int,
        index2: Int,
    ) async -> ParameterOptResult {
        let result1 = await optimizeSingleParameter(
            objectiveFunction: objectiveFunction,
            params: &params,
            index: index1,
        )
        let result2 = await optimizeSingleParameter(
            objectiveFunction: objectiveFunction,
            params: &params,
            index: index2,
        )

        let refinement = await optimizeSingleParameter(
            objectiveFunction: objectiveFunction,
            params: &params,
            index: index1,
        )

        return ParameterOptResult(
            value: refinement.value,
            evaluations: result1.evaluations + result2.evaluations + refinement.evaluations,
        )
    }
}
