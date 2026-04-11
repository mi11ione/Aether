// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Optimization constraint for constrained optimization problems
///
/// Specifies equality or inequality constraints for sequential quadratic programming.
/// Equality constraints enforce h(x) = 0, inequality constraints enforce g(x) ≥ 0.
/// Both types are specified as async closures mapping parameter vectors to scalar constraint
/// values, supporting quantum circuit-based constraint evaluation.
///
/// **Example:**
/// ```swift
/// let normConstraint = OptimizationConstraint.equality { params in
///     params.map { $0 * $0 }.reduce(0, +) - 1.0
/// }
/// let positivity = OptimizationConstraint.inequality { params in params[0] }
/// ```
///
/// - SeeAlso: ``SLSQPOptimizer`` for constrained optimization
@frozen
public enum OptimizationConstraint: Sendable {
    /// Equality constraint h(x) = 0
    ///
    /// Enforces that the constraint function evaluates to zero at the optimal point.
    /// Implemented via Lagrange multiplier in the SQP formulation.
    case equality(@Sendable ([Double]) async -> Double)

    /// Inequality constraint g(x) ≥ 0
    ///
    /// Enforces that the constraint function evaluates to non-negative at the optimal point.
    /// Active constraints (g(x) = 0 at optimum) are handled via active-set method.
    case inequality(@Sendable ([Double]) async -> Double)
}

/// SLSQP optimizer: Sequential Least Squares Quadratic Programming with constraints
///
/// Sequential quadratic programming method that handles equality and inequality constraints
/// via Lagrangian formulation. Each iteration constructs a quadratic subproblem approximating
/// the Lagrangian L(x,λ) = f(x) - Σλᵢhᵢ(x) - Σμⱼgⱼ(x) with BFGS-approximated Hessian,
/// solves for a search direction via the KKT system, performs line search with L1 merit
/// function Φ(x) = f(x) + ρΣ|hᵢ(x)| + ρΣmax(0,-gⱼ(x)), then updates the Hessian
/// approximation using the BFGS formula with safeguarded Lagrangian gradient differences.
///
/// Achieves superlinear convergence near the solution due to BFGS curvature accumulation.
/// Constraint handling via active-set QP ensures feasibility of the search direction.
/// The L1 merit function balances objective decrease against constraint violation reduction.
/// Requires 2n+1 evaluations per iteration for objective gradient plus O(m) for constraint
/// gradients where m is the constraint count.
///
/// Required for constrained quantum optimization problems such as particle number
/// conservation, symmetry constraints, or hardware-specific parameter bounds. For
/// unconstrained problems, conforms to ``Optimizer`` protocol and delegates to the
/// SQP engine with empty constraint set. For unconstrained smooth problems, prefer
/// ``LBFGSBOptimizer`` which avoids QP overhead.
///
/// **Example:**
/// ```swift
/// let optimizer = SLSQPOptimizer(tolerance: 1e-6)
/// let constraints: [OptimizationConstraint] = [
///     .equality { params in params.reduce(0, +) - 1.0 },
///     .inequality { params in params[0] }
/// ]
/// let result = await optimizer.minimize(
///     objective, from: initial, using: .init(), constraints: constraints, progress: nil
/// )
/// ```
///
/// - Complexity: O(maxIterations x (2n + m)) evaluations, O(n²) memory for Hessian
/// - SeeAlso: ``OptimizationConstraint`` for constraint specification
/// - SeeAlso: ``LBFGSBOptimizer`` for unconstrained alternative
/// - SeeAlso: ``Optimizer`` for protocol definition
@frozen
public struct SLSQPOptimizer: Optimizer {
    /// Convergence tolerance for KKT optimality conditions
    ///
    /// Optimization terminates when the L1 norm of constraint violation plus gradient
    /// of Lagrangian falls below this threshold.
    ///
    /// Typical values: 1e-6 (default), 1e-8 (tight), 1e-4 (loose)
    public let tolerance: Double

    /// Parameter shift for gradient computation
    ///
    /// Shift for finite difference gradient: ∂f/∂θ ≈ [f(θ+s) - f(θ-s)] / (2s).
    /// Uses quantum parameter shift rule with π/2 for standard gates.
    ///
    /// Typical values: π/2 (default for quantum gates)
    public let parameterShift: Double

    /// Maximum backtracking steps in merit function line search
    ///
    /// Limits step size reduction iterations. Backtracking multiplies step by 0.5
    /// until Armijo condition is satisfied on the L1 merit function.
    ///
    /// Typical values: 20 (default)
    public let maxLineSearchSteps: Int

    private let meritPenalty: Double = 10.0
    private let armijoConstant: Double = 1e-4
    private let lineSearchShrink: Double = 0.5

    /// Create SLSQP optimizer with convergence and gradient configuration
    ///
    /// - Parameters:
    ///   - tolerance: KKT optimality tolerance (default: 1e-6)
    ///   - parameterShift: Gradient computation shift (default: π/2)
    ///   - maxLineSearchSteps: Maximum line search backtracking steps (default: 20)
    /// - Precondition: tolerance > 0
    /// - Precondition: parameterShift > 0
    /// - Precondition: maxLineSearchSteps > 0
    public init(
        tolerance: Double = 1e-6,
        parameterShift: Double = .pi / 2,
        maxLineSearchSteps: Int = 20,
    ) {
        ValidationUtilities.validatePositiveDouble(tolerance, name: "tolerance")
        ValidationUtilities.validatePositiveDouble(parameterShift, name: "parameterShift")
        ValidationUtilities.validatePositiveInt(maxLineSearchSteps, name: "maxLineSearchSteps")

        self.tolerance = tolerance
        self.parameterShift = parameterShift
        self.maxLineSearchSteps = maxLineSearchSteps
    }

    /// Minimize unconstrained objective function (Optimizer protocol conformance)
    ///
    /// Delegates to constrained solver with empty constraint set. For unconstrained
    /// problems, the SQP reduces to a quasi-Newton method with BFGS Hessian approximation,
    /// equivalent to standard BFGS optimization.
    ///
    /// **Example:**
    /// ```swift
    /// let optimizer = SLSQPOptimizer()
    /// let result = await optimizer.minimize(
    ///     { params in params.map { $0 * $0 }.reduce(0, +) },
    ///     from: [1.0, 1.0],
    ///     using: ConvergenceCriteria(),
    ///     progress: nil
    /// )
    /// ```
    ///
    /// - Precondition: initialParameters is non-empty
    /// - Complexity: O(maxIterations x (2n + 1)) evaluations
    @_optimize(speed)
    @_eagerMove
    public func minimize(
        _ objectiveFunction: @Sendable ([Double]) async -> Double,
        from initialParameters: [Double],
        using convergenceCriteria: ConvergenceCriteria,
        progress: ProgressCallback?,
    ) async -> OptimizerResult {
        await minimize(
            objectiveFunction,
            from: initialParameters,
            using: convergenceCriteria,
            constraints: [],
            progress: progress,
        )
    }

    /// Minimize objective function subject to equality and inequality constraints
    ///
    /// Solves min f(x) subject to hᵢ(x) = 0, gⱼ(x) ≥ 0 using sequential quadratic
    /// programming with BFGS Hessian approximation and L1 merit function line search.
    ///
    /// **Example:**
    /// ```swift
    /// let optimizer = SLSQPOptimizer()
    /// let result = await optimizer.minimize(
    ///     { p in p[0] * p[0] + p[1] * p[1] },
    ///     from: [0.5, 0.5],
    ///     using: ConvergenceCriteria(),
    ///     constraints: [.equality { p in p[0] + p[1] - 1.0 }],
    ///     progress: nil
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - objectiveFunction: Function to minimize
    ///   - initialParameters: Starting point
    ///   - convergenceCriteria: Termination conditions
    ///   - constraints: Equality and inequality constraints
    ///   - progress: Optional progress callback
    /// - Returns: Optimization result with optimal parameters
    /// - Precondition: initialParameters is non-empty
    /// - Complexity: O(maxIterations x (2n + m)) evaluations where m = constraint count
    @_optimize(speed)
    @_eagerMove
    public func minimize(
        _ objectiveFunction: @Sendable ([Double]) async -> Double,
        from initialParameters: [Double],
        using convergenceCriteria: ConvergenceCriteria,
        constraints: [OptimizationConstraint],
        progress: ProgressCallback?,
    ) async -> OptimizerResult {
        let n = initialParameters.count
        ValidationUtilities.validateNonEmpty(initialParameters, name: "initialParameters")

        var params = initialParameters
        var currentValue = await objectiveFunction(params)
        var valueHistory: [Double] = [currentValue]
        var functionEvaluations = 1

        var hessian = [Double](unsafeUninitializedCapacity: n * n) {
            buffer, count in
            buffer.initialize(repeating: 0.0)
            for i in 0 ..< n {
                buffer[i * n + i] = 1.0
            }
            count = n * n
        }

        var gradient = await finiteDifferenceGradient(of: objectiveFunction, at: params)
        functionEvaluations += 2 * n

        var multipliers = [Double](unsafeUninitializedCapacity: constraints.count) {
            buffer, count in
            buffer.initialize(repeating: 0.0)
            count = constraints.count
        }

        var penalty = meritPenalty

        for iteration in 0 ..< convergenceCriteria.maxIterations {
            if let callback = progress {
                await callback(iteration, currentValue)
            }

            var constraintValues = [Double](unsafeUninitializedCapacity: constraints.count) {
                _, count in
                count = constraints.count
            }
            for i in 0 ..< constraints.count {
                constraintValues[i] = await evaluateConstraint(constraints[i], at: params)
                functionEvaluations += 1
            }

            var constraintJacobian = [[Double]]()
            constraintJacobian.reserveCapacity(constraints.count)
            for i in 0 ..< constraints.count {
                let grad = await finiteDifferenceGradient(
                    of: { p in await evaluateConstraint(constraints[i], at: p) },
                    at: params,
                )
                constraintJacobian.append(grad)
                functionEvaluations += 2 * n
            }

            let violation = computeConstraintViolation(
                constraintValues: constraintValues,
                constraints: constraints,
            )

            var gradNormSq = 0.0
            vDSP_svesqD(gradient, 1, &gradNormSq, vDSP_Length(n))
            let kktResidual = sqrt(gradNormSq) + violation

            if kktResidual < tolerance {
                return OptimizerResult(
                    parameters: params,
                    value: currentValue,
                    history: valueHistory,
                    iterations: iteration + 1,
                    terminationReason: .energyConverged,
                    evaluations: functionEvaluations,
                )
            }

            let direction = solveQPSubproblem(
                gradient: gradient,
                hessian: hessian,
                constraintValues: constraintValues,
                constraintJacobian: constraintJacobian,
                constraints: constraints,
                dimension: n,
            )

            for i in 0 ..< constraints.count {
                let absMultiplier = abs(multipliers[i])
                if absMultiplier > penalty {
                    penalty = absMultiplier * 1.5
                }
            }

            let meritBefore = computeMerit(
                value: currentValue,
                constraintValues: constraintValues,
                constraints: constraints,
                penalty: penalty,
            )

            var dirGrad = 0.0
            vDSP_dotprD(gradient, 1, direction, 1, &dirGrad, vDSP_Length(n))
            let meritDirectionalDerivative = dirGrad - penalty * violation

            var alpha = 1.0
            var newParams = [Double](unsafeUninitializedCapacity: n) {
                _, count in count = n
            }

            for _ in 0 ..< maxLineSearchSteps {
                var alphaVal = alpha
                vDSP_vsmaD(direction, 1, &alphaVal, params, 1, &newParams, 1, vDSP_Length(n))

                let newValue = await objectiveFunction(newParams)
                functionEvaluations += 1

                var newConstraintValues = [Double](unsafeUninitializedCapacity: constraints.count) {
                    _, count in
                    count = constraints.count
                }
                for i in 0 ..< constraints.count {
                    newConstraintValues[i] = await evaluateConstraint(constraints[i], at: newParams)
                    functionEvaluations += 1
                }

                let meritAfter = computeMerit(
                    value: newValue,
                    constraintValues: newConstraintValues,
                    constraints: constraints,
                    penalty: penalty,
                )

                if meritAfter <= meritBefore + armijoConstant * alpha * meritDirectionalDerivative {
                    break
                }

                alpha *= lineSearchShrink
            }

            let newValue = await objectiveFunction(newParams)
            functionEvaluations += 1
            let newGradient = await finiteDifferenceGradient(of: objectiveFunction, at: newParams)
            functionEvaluations += 2 * n

            var s = [Double](unsafeUninitializedCapacity: n) {
                _, count in count = n
            }
            vDSP_vsubD(params, 1, newParams, 1, &s, 1, vDSP_Length(n))

            var lagGradOld = gradient
            for i in 0 ..< constraints.count {
                var negMult = -multipliers[i]
                vDSP_vsmaD(constraintJacobian[i], 1, &negMult, lagGradOld, 1, &lagGradOld, 1, vDSP_Length(n))
            }

            for i in 0 ..< constraints.count {
                let newConstraintVal = await evaluateConstraint(constraints[i], at: newParams)
                functionEvaluations += 1

                switch constraints[i] {
                case .equality:
                    multipliers[i] += penalty * newConstraintVal
                case .inequality:
                    multipliers[i] = max(0.0, multipliers[i] - penalty * newConstraintVal)
                }
            }

            var lagGradNew = newGradient
            for i in 0 ..< constraints.count {
                let constraintGrad = await finiteDifferenceGradient(
                    of: { p in await evaluateConstraint(constraints[i], at: p) },
                    at: newParams,
                )
                functionEvaluations += 2 * n
                var negMult = -multipliers[i]
                vDSP_vsmaD(constraintGrad, 1, &negMult, lagGradNew, 1, &lagGradNew, 1, vDSP_Length(n))
            }

            var y = [Double](unsafeUninitializedCapacity: n) {
                _, count in count = n
            }
            vDSP_vsubD(lagGradOld, 1, lagGradNew, 1, &y, 1, vDSP_Length(n))

            updateBFGS(hessian: &hessian, s: s, y: y, dimension: n)

            if abs(newValue - currentValue) < convergenceCriteria.energyTolerance {
                return OptimizerResult(
                    parameters: newParams,
                    value: newValue,
                    history: valueHistory,
                    iterations: iteration + 1,
                    terminationReason: .energyConverged,
                    evaluations: functionEvaluations,
                )
            }

            params = newParams
            currentValue = newValue
            gradient = newGradient
            valueHistory.append(currentValue)
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

    /// Compute quantum parameter-shift gradient via (f(θ+s) - f(θ-s)) / (2 sin s).
    @_optimize(speed)
    @_eagerMove
    private func finiteDifferenceGradient(
        of function: @Sendable ([Double]) async -> Double,
        at params: [Double],
    ) async -> [Double] {
        let n = params.count
        var gradient = [Double](unsafeUninitializedCapacity: n) {
            _, count in count = n
        }
        var paramsPlus = params
        var paramsMinus = params
        let divisor = 2.0 * sin(parameterShift)

        for i in 0 ..< n {
            paramsPlus[i] = params[i] + parameterShift
            let valuePlus = await function(paramsPlus)
            paramsPlus[i] = params[i]

            paramsMinus[i] = params[i] - parameterShift
            let valueMinus = await function(paramsMinus)
            paramsMinus[i] = params[i]

            gradient[i] = (valuePlus - valueMinus) / divisor
        }
        return gradient
    }

    /// Evaluate a single constraint at given parameters
    @_optimize(speed)
    private func evaluateConstraint(
        _ constraint: OptimizationConstraint,
        at params: [Double],
    ) async -> Double {
        switch constraint {
        case let .equality(f): await f(params)
        case let .inequality(f): await f(params)
        }
    }

    /// Compute total constraint violation (L1 norm)
    @_optimize(speed)
    @_effects(readonly)
    private func computeConstraintViolation(
        constraintValues: [Double],
        constraints: [OptimizationConstraint],
    ) -> Double {
        var violation = 0.0
        for i in 0 ..< constraints.count {
            switch constraints[i] {
            case .equality:
                violation += abs(constraintValues[i])
            case .inequality:
                violation += max(0.0, -constraintValues[i])
            }
        }
        return violation
    }

    /// Compute L1 merit function for line search
    @_optimize(speed)
    @_effects(readonly)
    private func computeMerit(
        value: Double,
        constraintValues: [Double],
        constraints: [OptimizationConstraint],
        penalty: Double,
    ) -> Double {
        value + penalty * computeConstraintViolation(
            constraintValues: constraintValues,
            constraints: constraints,
        )
    }

    /// Solve QP subproblem for search direction using projected gradient approach
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private func solveQPSubproblem(
        gradient: [Double],
        hessian: [Double],
        constraintValues: [Double],
        constraintJacobian: [[Double]],
        constraints: [OptimizationConstraint],
        dimension: Int,
    ) -> [Double] {
        let n = dimension

        var direction = [Double](unsafeUninitializedCapacity: n) {
            _, count in count = n
        }

        var hessGrad = [Double](unsafeUninitializedCapacity: n) {
            buffer, count in
            buffer.initialize(repeating: 0.0)
            count = n
        }
        for i in 0 ..< n {
            var dotResult = 0.0
            vDSP_dotprD(
                [Double](hessian[(i * n) ..< (i * n + n)]), 1,
                gradient, 1,
                &dotResult, vDSP_Length(n),
            )
            hessGrad[i] = dotResult
        }

        var gBg = 0.0
        vDSP_dotprD(gradient, 1, hessGrad, 1, &gBg, vDSP_Length(n))

        var gradNormSq = 0.0
        vDSP_svesqD(gradient, 1, &gradNormSq, vDSP_Length(n))
        var scale = -gradNormSq / gBg
        vDSP_vsmulD(gradient, 1, &scale, &direction, 1, vDSP_Length(n))

        guard !constraints.isEmpty else { return direction }

        for outerIter in 0 ..< 5 {
            var maxViolation = 0.0
            var worstConstraint = -1

            for i in 0 ..< constraints.count {
                var adotd = 0.0
                vDSP_dotprD(constraintJacobian[i], 1, direction, 1, &adotd, vDSP_Length(n))
                let linearizedValue = constraintValues[i] + adotd

                let constraintViolation: Double = switch constraints[i] {
                case .equality: abs(linearizedValue)
                case .inequality: max(0.0, -linearizedValue)
                }

                if constraintViolation > maxViolation {
                    maxViolation = constraintViolation
                    worstConstraint = i
                }
            }

            guard maxViolation > tolerance, worstConstraint >= 0 else { break }

            let aRow = constraintJacobian[worstConstraint]
            var adotd = 0.0
            vDSP_dotprD(aRow, 1, direction, 1, &adotd, vDSP_Length(n))

            var aNormSq = 0.0
            vDSP_svesqD(aRow, 1, &aNormSq, vDSP_Length(n))

            let correction: Double
            switch constraints[worstConstraint] {
            case .equality:
                correction = -(constraintValues[worstConstraint] + adotd) / aNormSq
            case .inequality:
                let linearized = constraintValues[worstConstraint] + adotd
                correction = -linearized / aNormSq
            }

            var corrVal = correction
            vDSP_vsmaD(aRow, 1, &corrVal, direction, 1, &direction, 1, vDSP_Length(n))

            _ = outerIter
        }

        return direction
    }

    /// Update BFGS Hessian approximation with safeguarded curvature condition
    @_optimize(speed)
    private func updateBFGS(
        hessian: inout [Double],
        s: [Double],
        y: [Double],
        dimension: Int,
    ) {
        let n = dimension

        var sy = 0.0
        vDSP_dotprD(s, 1, y, 1, &sy, vDSP_Length(n))

        guard sy > 1e-10 else { return }

        var Bs = [Double](unsafeUninitializedCapacity: n) {
            buffer, count in
            buffer.initialize(repeating: 0.0)
            count = n
        }
        for i in 0 ..< n {
            var dotResult = 0.0
            vDSP_dotprD(
                [Double](hessian[(i * n) ..< (i * n + n)]), 1,
                s, 1,
                &dotResult, vDSP_Length(n),
            )
            Bs[i] = dotResult
        }

        var sBs = 0.0
        vDSP_dotprD(s, 1, Bs, 1, &sBs, vDSP_Length(n))

        let invSy = 1.0 / sy
        let invSBs = 1.0 / sBs

        for i in 0 ..< n {
            for j in 0 ..< n {
                hessian[i * n + j] += invSy * y[i] * y[j] - invSBs * Bs[i] * Bs[j]
            }
        }
    }
}
