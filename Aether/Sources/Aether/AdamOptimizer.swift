// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Adam optimizer with adaptive per-parameter learning rates via bias-corrected moment estimates
///
/// Combines momentum (first moment) with RMSProp (second moment) to adapt learning rates
/// per parameter based on gradient history. First moment m tracks gradient mean for momentum,
/// second moment v tracks gradient variance for per-parameter scaling. Bias correction
/// compensates for zero-initialization of both moments, critical in early iterations where
/// uncorrected estimates severely underestimate true moments.
///
/// Update rule per iteration t:
/// m_t = β₁·m_{t-1} + (1-β₁)·g_t,   v_t = β₂·v_{t-1} + (1-β₂)·g_t²
/// m̂_t = m_t/(1-β₁ᵗ),               v̂_t = v_t/(1-β₂ᵗ)
/// θ_t = θ_{t-1} - η·m̂_t/(√v̂_t + ε)
///
/// Subsumes RMSProp as special case (β₁=0 disables momentum). Per-parameter learning rate
/// adaptation makes Adam robust to heterogeneous gradient scales across parameters, common
/// in variational quantum circuits where rotation and entanglement parameters have different
/// sensitivity. Requires 2n+1 evaluations per iteration for parameter shift gradient.
/// O(p) memory for two moment vectors.
///
/// Preferred over ``GradientDescentOptimizer`` for most VQE applications due to adaptive
/// learning rates and momentum. For smooth landscapes with many parameters, ``LBFGSBOptimizer``
/// may converge faster using curvature information. For noisy objectives, gradient-free
/// methods like ``COBYLAOptimizer`` or ``SPSAOptimizer`` avoid gradient noise amplification.
///
/// **Example:**
/// ```swift
/// let optimizer = AdamOptimizer(learningRate: 0.001, beta1: 0.9, beta2: 0.999)
/// let result = await optimizer.minimize(objective, from: initial, using: .init(), progress: nil)
/// print("Optimal energy: \(result.value)")
/// ```
///
/// - Complexity: O(maxIterations x (2n + 1)) evaluations, O(n) memory for moment vectors
/// - SeeAlso: ``GradientDescentOptimizer`` for simpler first-order method
/// - SeeAlso: ``LBFGSBOptimizer`` for curvature-aware gradient optimization
/// - SeeAlso: ``Optimizer`` for protocol definition
@frozen
public struct AdamOptimizer: Optimizer {
    /// Learning rate (step size) for parameter updates
    ///
    /// Global learning rate scaled by per-parameter adaptive factors. Adam's adaptive
    /// mechanism means the effective per-parameter rate is η/√v̂, so the global rate
    /// can be set higher than for vanilla gradient descent.
    ///
    /// Typical values: 0.001 (default, Kingma & Ba recommended), 0.01 (aggressive), 0.0001 (conservative)
    public let learningRate: Double

    /// Exponential decay rate for first moment (momentum) estimates
    ///
    /// Controls how quickly the running mean of gradients decays. Higher values give
    /// more momentum (smoother trajectory), lower values track recent gradients more closely.
    /// Setting β₁=0 recovers RMSProp (no momentum).
    ///
    /// Typical values: 0.9 (default), 0.0 (RMSProp mode), 0.95 (heavy momentum)
    public let beta1: Double

    /// Exponential decay rate for second moment (variance) estimates
    ///
    /// Controls how quickly the running variance of gradients decays. Higher values give
    /// more stable per-parameter scaling but slower adaptation. Should be close to 1.0
    /// for stable optimization.
    ///
    /// Typical values: 0.999 (default), 0.99 (faster adaptation), 0.9999 (very stable)
    public let beta2: Double

    /// Numerical stability constant for division safety
    ///
    /// Prevents division by zero in the update rule when second moment estimates are
    /// near zero. Should be small enough not to affect optimization but large enough
    /// to prevent numerical overflow.
    ///
    /// Typical values: 1e-8 (default), 1e-7 (more conservative)
    public let epsilon: Double

    /// Parameter shift for gradient computation via quantum parameter shift rule
    ///
    /// Shift value for gradient: ∂E/∂θ = [E(θ+s) - E(θ-s)] / 2.
    /// Standard quantum parameter shift uses π/2 for gates with eigenvalues ±1.
    ///
    /// Typical values: π/2 (default for standard quantum gates)
    public let parameterShift: Double

    /// Create Adam optimizer with full configuration
    ///
    /// - Parameters:
    ///   - learningRate: Global learning rate (default: 0.001)
    ///   - beta1: First moment decay rate (default: 0.9)
    ///   - beta2: Second moment decay rate (default: 0.999)
    ///   - epsilon: Numerical stability constant (default: 1e-8)
    ///   - parameterShift: Shift for quantum parameter shift rule (default: π/2)
    /// - Precondition: learningRate > 0
    /// - Precondition: beta1 ∈ [0, 1)
    /// - Precondition: beta2 ∈ [0, 1)
    /// - Precondition: epsilon > 0
    /// - Precondition: parameterShift > 0
    public init(
        learningRate: Double = 0.001,
        beta1: Double = 0.9,
        beta2: Double = 0.999,
        epsilon: Double = 1e-8,
        parameterShift: Double = .pi / 2,
    ) {
        ValidationUtilities.validatePositiveDouble(learningRate, name: "learningRate")
        ValidationUtilities.validateHalfOpenRange(beta1, min: 0, max: 1, name: "beta1")
        ValidationUtilities.validateHalfOpenRange(beta2, min: 0, max: 1, name: "beta2")
        ValidationUtilities.validatePositiveDouble(epsilon, name: "epsilon")
        ValidationUtilities.validatePositiveDouble(parameterShift, name: "parameterShift")

        self.learningRate = learningRate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.parameterShift = parameterShift
    }

    /// Minimize objective function using Adam with bias-corrected adaptive moments
    ///
    /// Computes gradients via quantum parameter shift rule, updates exponential moving
    /// averages of gradient (first moment) and squared gradient (second moment), applies
    /// bias correction, then performs per-parameter update with adaptive learning rate.
    ///
    /// **Example:**
    /// ```swift
    /// let optimizer = AdamOptimizer()
    /// let result = await optimizer.minimize(
    ///     { params in params.map { $0 * $0 }.reduce(0, +) },
    ///     from: [1.0, 1.0],
    ///     using: ConvergenceCriteria(),
    ///     progress: nil
    /// )
    /// ```
    ///
    /// - Precondition: initialParameters is non-empty
    /// - Complexity: O(maxIterations x (2n + 1)) evaluations where n = parameter count
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

        var firstMoment = [Double](unsafeUninitializedCapacity: n) {
            buffer, count in
            buffer.initialize(repeating: 0.0)
            count = n
        }
        var secondMoment = [Double](unsafeUninitializedCapacity: n) {
            buffer, count in
            buffer.initialize(repeating: 0.0)
            count = n
        }
        var gradient = [Double](unsafeUninitializedCapacity: n) {
            _, count in
            count = n
        }

        var paramsPlus = params
        var paramsMinus = params
        var gradientSquared = [Double](unsafeUninitializedCapacity: n) {
            _, count in
            count = n
        }

        for iteration in 0 ..< convergenceCriteria.maxIterations {
            if let callback = progress {
                await callback(iteration, currentValue)
            }

            for i in 0 ..< n {
                paramsPlus[i] = params[i] + parameterShift
                let valuePlus = await objectiveFunction(paramsPlus)
                paramsPlus[i] = params[i]

                paramsMinus[i] = params[i] - parameterShift
                let valueMinus = await objectiveFunction(paramsMinus)
                paramsMinus[i] = params[i]

                gradient[i] = (valuePlus - valueMinus) / 2.0
            }
            functionEvaluations += 2 * n

            let t = Double(iteration + 1)
            var oneMinusBeta1 = 1.0 - beta1
            var oneMinusBeta2 = 1.0 - beta2
            var beta1Val = beta1
            var beta2Val = beta2

            vDSP_vsmulD(firstMoment, 1, &beta1Val, &firstMoment, 1, vDSP_Length(n))
            vDSP_vsmaD(gradient, 1, &oneMinusBeta1, firstMoment, 1, &firstMoment, 1, vDSP_Length(n))

            vDSP_vsqD(gradient, 1, &gradientSquared, 1, vDSP_Length(n))
            vDSP_vsmulD(secondMoment, 1, &beta2Val, &secondMoment, 1, vDSP_Length(n))
            vDSP_vsmaD(gradientSquared, 1, &oneMinusBeta2, secondMoment, 1, &secondMoment, 1, vDSP_Length(n))

            let biasCorrection1 = 1.0 / (1.0 - pow(beta1, t))
            let biasCorrection2 = 1.0 / (1.0 - pow(beta2, t))

            for i in 0 ..< n {
                let mHat = firstMoment[i] * biasCorrection1
                let vHat = secondMoment[i] * biasCorrection2
                params[i] -= learningRate * mHat / (sqrt(vHat) + epsilon)
            }

            let newValue = await objectiveFunction(params)
            functionEvaluations += 1

            var gradientNormSq = 0.0
            vDSP_svesqD(gradient, 1, &gradientNormSq, vDSP_Length(n))

            if let gnt = convergenceCriteria.gradientNormTolerance,
               gradientNormSq < gnt * gnt
            {
                return OptimizerResult(
                    parameters: params,
                    value: newValue,
                    history: valueHistory,
                    iterations: iteration + 1,
                    terminationReason: .gradientConverged,
                    evaluations: functionEvaluations,
                )
            }

            if abs(newValue - currentValue) < convergenceCriteria.energyTolerance {
                return OptimizerResult(
                    parameters: params,
                    value: newValue,
                    history: valueHistory,
                    iterations: iteration + 1,
                    terminationReason: .energyConverged,
                    evaluations: functionEvaluations,
                )
            }

            currentValue = newValue
            valueHistory.append(newValue)
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
}
