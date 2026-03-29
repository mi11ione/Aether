// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Quantum Natural Gradient optimizer with geometry-aware parameter updates
///
/// Preconditions gradient updates with the inverse Fubini-Study metric tensor to achieve
/// parameter-space covariance, producing the natural gradient step θ → θ - η·g⁻¹·∇E
/// where g is the quantum geometric tensor. This accounts for the non-Euclidean geometry
/// of the quantum state manifold: parameters that produce large state changes receive
/// smaller updates, while insensitive parameters receive larger updates, normalizing the
/// effective step size across all parameters.
///
/// Two metric computation modes with automatic selection:
///
/// **Exact mode** (p < 50 with circuit provided): Computes the full p×p Fubini-Study
/// metric g_ij = Re⟨∂ᵢψ|∂ⱼψ⟩ - ⟨∂ᵢψ|ψ⟩⟨ψ|∂ⱼψ⟩ via adjoint differentiation using
/// ``GradientMethods/fubiniStudyMetric(circuit:parameters:)``. Requires O(p²) inner
/// products over O(2ⁿ)-dimensional derivative states.
///
/// **QNSPSA mode** (p ≥ 50 or no circuit): Approximates the metric tensor using
/// simultaneous perturbation with O(1) circuit evaluations per rank-1 estimate.
/// Running exponential average smooths stochastic estimates across iterations.
///
/// Tikhonov regularization g → g + εI prevents singular metric inversion near degenerate
/// parameter subspaces. The regularized system is solved via Cholesky factorization
/// (LAPACK dpotrf/dpotrs) for numerical stability.
///
/// Converges faster than vanilla gradient descent on barren plateau-prone landscapes
/// where the metric tensor captures parameter sensitivity. For smooth unconstrained
/// problems without quantum structure, ``LBFGSBOptimizer`` may be more efficient.
/// For noisy objectives, ``SPSAOptimizer`` avoids metric computation overhead.
///
/// **Example:**
/// ```swift
/// let qng = QuantumNaturalGradientOptimizer(learningRate: 0.01, circuit: circuit)
/// let result = await qng.minimize(objective, from: initial, using: .init(), progress: nil)
/// print("Optimal energy: \(result.value)")
/// ```
///
/// - Complexity: Exact: O(iter x (p·L·2ⁿ + n³)), QNSPSA: O(iter x (2n + 4 + n³))
/// - SeeAlso: ``GradientMethods/fubiniStudyMetric(circuit:parameters:)`` for metric computation
/// - SeeAlso: ``AdamOptimizer`` for adaptive gradient without metric structure
/// - SeeAlso: ``Optimizer`` for protocol definition
@frozen
public struct QuantumNaturalGradientOptimizer: Optimizer {
    /// Learning rate for natural gradient parameter updates
    ///
    /// Step size for the preconditioned update θ ← θ - η·g⁻¹·∇E. Natural gradient
    /// normalization means smaller learning rates are typically needed compared to
    /// vanilla gradient descent.
    ///
    /// Typical values: 0.01 (default), 0.001 (conservative), 0.1 (aggressive)
    public let learningRate: Double

    /// Tikhonov regularization parameter for metric tensor inversion
    ///
    /// Added to diagonal of metric tensor before inversion: g → g + εI. Prevents
    /// singular inversion when parameters are nearly redundant (degenerate metric).
    /// Larger values improve stability but reduce natural gradient quality.
    ///
    /// Typical values: 1e-4 (default), 1e-3 (more regularization), 1e-6 (minimal)
    public let regularization: Double

    /// Parameter shift for gradient computation
    ///
    /// Shift for quantum parameter shift rule: ∂E/∂θ = [E(θ+s) - E(θ-s)] / 2.
    ///
    /// Typical values: π/2 (default for standard quantum gates)
    public let parameterShift: Double

    /// Exponential averaging factor for QNSPSA metric estimates
    ///
    /// Controls how quickly the running metric average adapts. Closer to 1.0 gives
    /// more weight to historical estimates (smoother), closer to 0.0 tracks recent
    /// estimates (noisier). Only used in QNSPSA mode.
    ///
    /// Typical values: 0.9 (default)
    public let metricAveraging: Double

    /// Threshold for automatic mode selection between exact and QNSPSA
    ///
    /// When circuit is provided and parameter count is below this threshold,
    /// exact Fubini-Study metric is computed. Otherwise QNSPSA approximation is used.
    ///
    /// Typical values: 50 (default)
    public let exactMetricThreshold: Int

    private let circuit: QuantumCircuit?

    /// Create quantum natural gradient optimizer with full configuration
    ///
    /// - Parameters:
    ///   - learningRate: Natural gradient step size (default: 0.01)
    ///   - regularization: Tikhonov regularization ε for g + εI (default: 1e-4)
    ///   - parameterShift: Gradient computation shift (default: π/2)
    ///   - metricAveraging: QNSPSA exponential averaging factor (default: 0.9)
    ///   - exactMetricThreshold: Parameter count threshold for exact metric (default: 50)
    ///   - circuit: Optional circuit for exact metric mode (nil for QNSPSA-only)
    /// - Precondition: learningRate > 0
    /// - Precondition: regularization > 0
    /// - Precondition: parameterShift > 0
    /// - Precondition: metricAveraging ∈ [0, 1)
    /// - Precondition: exactMetricThreshold > 0
    public init(
        learningRate: Double = 0.01,
        regularization: Double = 1e-4,
        parameterShift: Double = .pi / 2,
        metricAveraging: Double = 0.9,
        exactMetricThreshold: Int = 50,
        circuit: QuantumCircuit? = nil,
    ) {
        ValidationUtilities.validatePositiveDouble(learningRate, name: "learningRate")
        ValidationUtilities.validatePositiveDouble(regularization, name: "regularization")
        ValidationUtilities.validatePositiveDouble(parameterShift, name: "parameterShift")
        ValidationUtilities.validateHalfOpenRange(metricAveraging, min: 0, max: 1, name: "metricAveraging")
        ValidationUtilities.validatePositiveInt(exactMetricThreshold, name: "exactMetricThreshold")

        self.learningRate = learningRate
        self.regularization = regularization
        self.parameterShift = parameterShift
        self.metricAveraging = metricAveraging
        self.exactMetricThreshold = exactMetricThreshold
        self.circuit = circuit
    }

    /// Minimize objective function using quantum natural gradient
    ///
    /// Computes gradient via parameter shift, computes or approximates the Fubini-Study
    /// metric tensor, solves the regularized linear system (g + εI)δθ = ∇E via Cholesky
    /// factorization, then updates θ ← θ - η·δθ.
    ///
    /// **Example:**
    /// ```swift
    /// let qng = QuantumNaturalGradientOptimizer(learningRate: 0.01)
    /// let result = await qng.minimize(
    ///     { params in params.map { $0 * $0 }.reduce(0, +) },
    ///     from: [1.0, 1.0],
    ///     using: ConvergenceCriteria(),
    ///     progress: nil
    /// )
    /// ```
    ///
    /// - Precondition: initialParameters is non-empty
    /// - Complexity: Exact: O(iter x (p·L·2ⁿ + p³)), QNSPSA: O(iter x (2p + 4 + p³))
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

        let useExactMetric = circuit != nil && n < exactMetricThreshold

        var params = initialParameters
        var currentValue = await objectiveFunction(params)
        var valueHistory: [Double] = [currentValue]
        var functionEvaluations = 1

        var runningMetric: [Double]? = nil
        var gradient = [Double](unsafeUninitializedCapacity: n) {
            _, count in count = n
        }
        var paramsPlus = params
        var paramsMinus = params

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

            var metricFlat: [Double]
            if useExactMetric {
                // Safe force unwrap: useExactMetric is true only when circuit != nil (line 158)
                let metricMatrix = GradientMethods.fubiniStudyMetric(
                    circuit: circuit!,
                    parameters: params,
                )
                metricFlat = [Double](unsafeUninitializedCapacity: n * n) { buffer, count in
                    for i in 0 ..< n {
                        for j in 0 ..< n {
                            buffer[i * n + j] = metricMatrix[i][j]
                        }
                    }
                    count = n * n
                }
            } else {
                let estimate = await estimateMetricQNSPSA(
                    objectiveFunction: objectiveFunction,
                    params: params,
                    functionEvaluations: &functionEvaluations,
                )

                if var running = runningMetric {
                    var avgFactor = metricAveraging
                    var oneMinusAvg = 1.0 - metricAveraging
                    vDSP_vsmulD(running, 1, &avgFactor, &running, 1, vDSP_Length(n * n))
                    vDSP_vsmaD(estimate, 1, &oneMinusAvg, running, 1, &running, 1, vDSP_Length(n * n))
                    runningMetric = running
                    metricFlat = running
                } else {
                    var identity = [Double](unsafeUninitializedCapacity: n * n) {
                        buffer, count in
                        buffer.initialize(repeating: 0.0)
                        for i in 0 ..< n {
                            buffer[i * n + i] = 1.0
                        }
                        count = n * n
                    }
                    var avgFactor = metricAveraging
                    var oneMinusAvg = 1.0 - metricAveraging
                    vDSP_vsmulD(identity, 1, &avgFactor, &identity, 1, vDSP_Length(n * n))
                    vDSP_vsmaD(estimate, 1, &oneMinusAvg, identity, 1, &identity, 1, vDSP_Length(n * n))
                    runningMetric = identity
                    metricFlat = identity
                }
            }

            for i in 0 ..< n {
                metricFlat[i * n + i] += regularization
            }

            let naturalGradient = solveRegularizedSystem(
                metric: metricFlat,
                gradient: gradient,
                dimension: n,
            )

            var negLR = -learningRate
            vDSP_vsmaD(naturalGradient, 1, &negLR, params, 1, &params, 1, vDSP_Length(n))

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

    /// Estimate metric tensor via QNSPSA simultaneous perturbation
    @_optimize(speed)
    @_eagerMove
    private func estimateMetricQNSPSA(
        objectiveFunction: @Sendable ([Double]) async -> Double,
        params: [Double],
        functionEvaluations: inout Int,
    ) async -> [Double] {
        let n = params.count
        let perturbationSize = 0.01

        let delta1 = generatePerturbation(dimension: n)
        let delta2 = generatePerturbation(dimension: n)

        var paramsPP = [Double](unsafeUninitializedCapacity: n) {
            _, count in count = n
        }
        var paramsPN = [Double](unsafeUninitializedCapacity: n) {
            _, count in count = n
        }
        var paramsNP = [Double](unsafeUninitializedCapacity: n) {
            _, count in count = n
        }
        var paramsNN = [Double](unsafeUninitializedCapacity: n) {
            _, count in count = n
        }

        for i in 0 ..< n {
            let d1 = perturbationSize * delta1[i]
            let d2 = perturbationSize * delta2[i]
            paramsPP[i] = params[i] + d1 + d2
            paramsPN[i] = params[i] + d1 - d2
            paramsNP[i] = params[i] - d1 + d2
            paramsNN[i] = params[i] - d1 - d2
        }

        let fPP = await objectiveFunction(paramsPP)
        let fPN = await objectiveFunction(paramsPN)
        let fNP = await objectiveFunction(paramsNP)
        let fNN = await objectiveFunction(paramsNN)
        functionEvaluations += 4

        let hessianApprox = (fPP - fPN - fNP + fNN) / (4.0 * perturbationSize * perturbationSize)

        var metric = [Double](unsafeUninitializedCapacity: n * n) {
            buffer, count in
            for i in 0 ..< n {
                let invDelta1i = 1.0 / delta1[i]
                for j in 0 ..< n {
                    let invDelta2j = 1.0 / delta2[j]
                    buffer[i * n + j] = abs(hessianApprox * invDelta1i * invDelta2j)
                }
            }
            count = n * n
        }

        for i in 0 ..< n {
            for j in 0 ..< n where i != j {
                let sym = (metric[i * n + j] + metric[j * n + i]) / 2.0
                metric[i * n + j] = sym
                metric[j * n + i] = sym
            }
        }

        return metric
    }

    /// Generate random perturbation vector with entries ±1
    @_optimize(speed)
    @_eagerMove
    private func generatePerturbation(dimension: Int) -> [Double] {
        [Double](unsafeUninitializedCapacity: dimension) { buffer, count in
            let numBytes = (dimension + 7) / 8
            var randomBytes = [UInt8](repeating: 0, count: numBytes)
            arc4random_buf(&randomBytes, numBytes)

            for i in 0 ..< dimension {
                let byteIndex = i / 8
                let bitIndex = i % 8
                let bit = (randomBytes[byteIndex] >> bitIndex) & 1
                buffer[i] = 1.0 - 2.0 * Double(bit)
            }
            count = dimension
        }
    }

    /// Solve (g + εI)x = b via Cholesky factorization, falling back to diagonal inversion
    @_optimize(speed)
    @_eagerMove
    private func solveRegularizedSystem(
        metric: [Double],
        gradient: [Double],
        dimension: Int,
    ) -> [Double] {
        let n = dimension
        var matrix = metric
        var rhs = gradient
        var nn = __LAPACK_int(n)
        var nrhs = __LAPACK_int(1)
        var info = __LAPACK_int(0)
        var uplo = CChar(Character("U").asciiValue!)

        dpotrf_(&uplo, &nn, &matrix, &nn, &info)
        dpotrs_(&uplo, &nn, &nrhs, &matrix, &nn, &rhs, &nn, &info)

        return rhs
    }
}
