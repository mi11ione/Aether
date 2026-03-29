// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Powell's conjugate direction optimizer (derivative-free)
///
/// Derivative-free optimization via sequential one-dimensional line searches along a set
/// of conjugate directions. Starts with coordinate axis directions, performs golden section
/// line search along each direction to find the per-direction minimum, then updates the
/// direction set by replacing the direction of maximum function decrease with the net
/// displacement vector. This direction update builds approximate conjugacy without gradient
/// information, converging quadratically on quadratic objectives.
///
/// Each iteration performs p line searches (one per direction) with O(maxLineSearchIterations)
/// function evaluations each, giving O(p x lineSearchEvals) evaluations per iteration.
/// Direction update via maximum decrease criterion prevents linear dependence of the direction
/// set, maintaining convergence guarantees. No gradient or Hessian required — only function
/// evaluations — making Powell robust for noisy or non-differentiable objectives.
///
/// Best suited for smooth, moderately-dimensional landscapes (< 30 parameters) where
/// gradients are unavailable or expensive. For high dimensions, prefer ``SPSAOptimizer``
/// (O(1) evaluations per iteration) or ``CMAESOptimizer`` (population parallelism). For
/// smooth landscapes with available gradients, ``LBFGSBOptimizer`` converges faster using
/// curvature information.
///
/// **Example:**
/// ```swift
/// let optimizer = PowellOptimizer(lineSearchTolerance: 1e-8)
/// let result = await optimizer.minimize(objective, from: initial, using: .init(), progress: nil)
/// print("Optimal energy: \(result.value)")
/// ```
///
/// - Complexity: O(maxIterations x p x lineSearchEvals) evaluations where p = parameter count
/// - SeeAlso: ``NelderMeadOptimizer`` for simplex-based derivative-free optimization
/// - SeeAlso: ``COBYLAOptimizer`` for trust-region derivative-free optimization
/// - SeeAlso: ``Optimizer`` for protocol definition
@frozen
public struct PowellOptimizer: Optimizer {
    /// Tolerance for golden section line search convergence
    ///
    /// Line search terminates when the bracket width falls below this threshold.
    /// Tighter tolerance gives more precise per-direction minimization at the cost
    /// of additional function evaluations.
    ///
    /// Typical values: 1e-8 (default), 1e-6 (loose), 1e-10 (tight)
    public let lineSearchTolerance: Double

    /// Maximum iterations per golden section line search
    ///
    /// Limits function evaluations within each one-dimensional search. Golden section
    /// converges geometrically (bracket shrinks by factor 0.618 per step), so 100
    /// iterations achieves ~10⁻²¹ relative precision — well beyond double precision.
    ///
    /// Typical values: 100 (default), 50 (fast), 200 (thorough)
    public let maxLineSearchIterations: Int

    private let goldenRatio: Double = 0.6180339887498949
    private let initialBracketStep: Double = 0.1

    /// Create Powell optimizer with line search configuration
    ///
    /// - Parameters:
    ///   - lineSearchTolerance: Convergence tolerance for line search bracket width (default: 1e-8)
    ///   - maxLineSearchIterations: Maximum iterations per line search (default: 100)
    /// - Precondition: lineSearchTolerance > 0
    /// - Precondition: maxLineSearchIterations > 0
    public init(
        lineSearchTolerance: Double = 1e-8,
        maxLineSearchIterations: Int = 100,
    ) {
        ValidationUtilities.validatePositiveDouble(lineSearchTolerance, name: "lineSearchTolerance")
        ValidationUtilities.validatePositiveInt(maxLineSearchIterations, name: "maxLineSearchIterations")

        self.lineSearchTolerance = lineSearchTolerance
        self.maxLineSearchIterations = maxLineSearchIterations
    }

    /// Minimize objective function using Powell's conjugate direction method
    ///
    /// Initializes with coordinate axis directions, iteratively performs line search along
    /// each direction, identifies direction of maximum decrease, replaces it with net
    /// displacement to build conjugacy. Converges when function change falls below tolerance.
    ///
    /// **Example:**
    /// ```swift
    /// let optimizer = PowellOptimizer()
    /// let result = await optimizer.minimize(
    ///     { params in params.map { $0 * $0 }.reduce(0, +) },
    ///     from: [1.0, 1.0],
    ///     using: ConvergenceCriteria(),
    ///     progress: nil
    /// )
    /// ```
    ///
    /// - Precondition: initialParameters is non-empty
    /// - Complexity: O(maxIterations x p x lineSearchIterations) evaluations
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

        var directions = [[Double]](unsafeUninitializedCapacity: n) {
            buffer, count in
            for i in 0 ..< n {
                buffer[i] = [Double](unsafeUninitializedCapacity: n) { inner, innerCount in
                    inner.initialize(repeating: 0.0)
                    inner[i] = 1.0
                    innerCount = n
                }
            }
            count = n
        }

        var params = initialParameters
        var currentValue = await objectiveFunction(params)
        var valueHistory: [Double] = [currentValue]
        var functionEvaluations = 1

        for iteration in 0 ..< convergenceCriteria.maxIterations {
            if let callback = progress {
                await callback(iteration, currentValue)
            }

            let iterationStartValue = currentValue
            let iterationStartParams = params
            var maxDecrease = 0.0
            var maxDecreaseIndex = 0

            for i in 0 ..< n {
                let valueBeforeSearch = currentValue
                let lineResult = await goldenSectionLineSearch(
                    objectiveFunction: objectiveFunction,
                    origin: params,
                    direction: directions[i],
                )
                functionEvaluations += lineResult.evaluations

                var stepSize = lineResult.stepSize
                vDSP_vsmaD(
                    directions[i], 1, &stepSize,
                    params, 1, &params, 1, vDSP_Length(n),
                )
                currentValue = lineResult.value

                let decrease = valueBeforeSearch - currentValue
                if decrease > maxDecrease {
                    maxDecrease = decrease
                    maxDecreaseIndex = i
                }
            }

            let totalDecrease = iterationStartValue - currentValue

            if totalDecrease < convergenceCriteria.energyTolerance {
                return OptimizerResult(
                    parameters: params,
                    value: currentValue,
                    history: valueHistory,
                    iterations: iteration + 1,
                    terminationReason: .energyConverged,
                    evaluations: functionEvaluations,
                )
            }

            valueHistory.append(currentValue)

            var netDirection = [Double](unsafeUninitializedCapacity: n) {
                _, count in
                count = n
            }
            vDSP_vsubD(iterationStartParams, 1, params, 1, &netDirection, 1, vDSP_Length(n))

            var netNormSq = 0.0
            vDSP_svesqD(netDirection, 1, &netNormSq, vDSP_Length(n))

            if netNormSq > 1e-30 {
                let invNorm = 1.0 / sqrt(netNormSq)
                var scale = invNorm
                vDSP_vsmulD(netDirection, 1, &scale, &netDirection, 1, vDSP_Length(n))

                var extrapolatedParams = [Double](unsafeUninitializedCapacity: n) {
                    _, count in
                    count = n
                }
                var two = 2.0
                var negOne = -1.0
                vDSP_vsmulD(params, 1, &two, &extrapolatedParams, 1, vDSP_Length(n))
                vDSP_vsmaD(iterationStartParams, 1, &negOne, extrapolatedParams, 1, &extrapolatedParams, 1, vDSP_Length(n))

                let extrapolatedValue = await objectiveFunction(extrapolatedParams)
                functionEvaluations += 1

                let shouldReplace = extrapolatedValue < iterationStartValue
                    || 2.0 * (iterationStartValue - 2.0 * currentValue + extrapolatedValue)
                    * (iterationStartValue - currentValue - maxDecrease)
                    * (iterationStartValue - currentValue - maxDecrease)
                    < maxDecrease * (iterationStartValue - extrapolatedValue)
                    * (iterationStartValue - extrapolatedValue)

                if shouldReplace {
                    directions[maxDecreaseIndex] = netDirection

                    let lineResult = await goldenSectionLineSearch(
                        objectiveFunction: objectiveFunction,
                        origin: params,
                        direction: netDirection,
                    )
                    functionEvaluations += lineResult.evaluations

                    var step = lineResult.stepSize
                    vDSP_vsmaD(
                        netDirection, 1, &step,
                        params, 1, &params, 1, vDSP_Length(n),
                    )
                    currentValue = lineResult.value
                }
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

    /// Result of a one-dimensional golden section line search
    private struct LineSearchResult {
        let stepSize: Double
        let value: Double
        let evaluations: Int
    }

    /// Perform golden section line search along a direction from an origin point
    @_optimize(speed)
    private func goldenSectionLineSearch(
        objectiveFunction: @Sendable ([Double]) async -> Double,
        origin: [Double],
        direction: [Double],
    ) async -> LineSearchResult {
        let n = origin.count
        var evaluations = 0

        var stepA = 0.0
        var stepB = initialBracketStep
        var tempParams = [Double](unsafeUninitializedCapacity: n) {
            _, count in
            count = n
        }

        let valueA = await objectiveFunction(origin)
        evaluations += 1

        var bStep = stepB
        vDSP_vsmaD(direction, 1, &bStep, origin, 1, &tempParams, 1, vDSP_Length(n))
        var valueB = await objectiveFunction(tempParams)
        evaluations += 1

        if valueB > valueA {
            stepB = -initialBracketStep
            bStep = stepB
            vDSP_vsmaD(direction, 1, &bStep, origin, 1, &tempParams, 1, vDSP_Length(n))
            valueB = await objectiveFunction(tempParams)
            evaluations += 1
        }

        var stepC = stepB + goldenRatio * stepB
        var cStep = stepC
        vDSP_vsmaD(direction, 1, &cStep, origin, 1, &tempParams, 1, vDSP_Length(n))
        var valueC = await objectiveFunction(tempParams)
        evaluations += 1

        var bracketIterations = 0
        while valueC < valueB, bracketIterations < maxLineSearchIterations {
            stepA = stepB
            stepB = stepC
            valueB = valueC
            stepC = stepB + goldenRatio * (stepB - stepA)

            cStep = stepC
            vDSP_vsmaD(direction, 1, &cStep, origin, 1, &tempParams, 1, vDSP_Length(n))
            valueC = await objectiveFunction(tempParams)
            evaluations += 1
            bracketIterations += 1
        }

        if stepA > stepC { swap(&stepA, &stepC) }

        var lower = stepA
        var upper = stepC

        for _ in 0 ..< maxLineSearchIterations {
            if abs(upper - lower) < lineSearchTolerance { break }

            let probe1 = upper - goldenRatio * (upper - lower)
            let probe2 = lower + goldenRatio * (upper - lower)

            var p1Step = probe1
            vDSP_vsmaD(direction, 1, &p1Step, origin, 1, &tempParams, 1, vDSP_Length(n))
            let value1 = await objectiveFunction(tempParams)

            var p2Step = probe2
            vDSP_vsmaD(direction, 1, &p2Step, origin, 1, &tempParams, 1, vDSP_Length(n))
            let value2 = await objectiveFunction(tempParams)
            evaluations += 2

            if value1 < value2 {
                upper = probe2
            } else {
                lower = probe1
            }
        }

        let optimalStep = (lower + upper) / 2.0
        var finalStep = optimalStep
        vDSP_vsmaD(direction, 1, &finalStep, origin, 1, &tempParams, 1, vDSP_Length(n))
        let finalValue = await objectiveFunction(tempParams)
        evaluations += 1

        return LineSearchResult(stepSize: optimalStep, value: finalValue, evaluations: evaluations)
    }
}
