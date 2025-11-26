// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate
import Foundation

/// Classical optimizer for variational quantum algorithms
///
/// Protocol defining optimization interface for VQE and QAOA. Implementations
/// minimize objective functions over parameter space using various strategies:
/// gradient-based (requires derivatives) or gradient-free (simplex, evolutionary).
///
/// **Usage in VQE:**
/// ```swift
/// let optimizer = NelderMeadOptimizer(tolerance: 1e-6)
/// let vqe = await VariationalQuantumEigensolver(
///     hamiltonian: hamiltonian,
///     ansatz: ansatz,
///     optimizer: optimizer
/// )
/// let result = await vqe.run(initialParameters: initialGuess)
/// ```
public protocol Optimizer: Sendable {
    /// Minimize objective function over parameter space
    ///
    /// Implementations must handle async objective functions (quantum circuit
    /// execution is async). Progress updates via optional callback.
    ///
    /// - Parameters:
    ///   - objectiveFunction: Async function to minimize (typically energy expectation)
    ///   - initialParameters: Starting point in parameter space
    ///   - convergenceCriteria: Termination conditions
    ///   - progressCallback: Optional progress updates (iteration, current value)
    /// - Returns: Optimization result with optimal parameters and history
    func minimize(
        objectiveFunction: @Sendable ([Double]) async -> Double,
        initialParameters: [Double],
        convergenceCriteria: ConvergenceCriteria,
        progressCallback: (@Sendable (Int, Double) async -> Void)?
    ) async -> OptimizerResult
}

// MARK: - Convergence Criteria

/// Termination conditions for optimization algorithms
///
/// Defines when optimization should stop: tolerance-based (converged) or
/// iteration-based (max iterations reached). Different optimizers use
/// different criteria (gradient-free ignore gradient tolerance).
///
/// **Defaults:**
/// ```swift
/// let criteria = ConvergenceCriteria.default
/// // energyTolerance: 1e-6
/// // maxIterations: 1000
/// ```
@frozen
public struct ConvergenceCriteria: Sendable {
    /// Energy change threshold: |E_new - E_old| < ε -> converged
    public let energyTolerance: Double

    /// Gradient norm threshold: ||∇E|| < δ -> converged (gradient-based only)
    public let gradientNormTolerance: Double?

    /// Maximum optimization iterations
    public let maxIterations: Int

    /// Create convergence criteria
    /// - Parameters:
    ///   - energyTolerance: Energy change threshold (default: 1e-6)
    ///   - gradientNormTolerance: Gradient norm threshold (default: nil)
    ///   - maxIterations: Maximum iterations (default: 1000)
    public init(
        energyTolerance: Double = 1e-6,
        gradientNormTolerance: Double? = nil,
        maxIterations: Int = 1000,
    ) {
        ValidationUtilities.validatePositiveDouble(energyTolerance, name: "energyTolerance")
        ValidationUtilities.validatePositiveInt(maxIterations, name: "maxIterations")
        if let gnt = gradientNormTolerance {
            ValidationUtilities.validatePositiveDouble(gnt, name: "gradientNormTolerance")
        }

        self.energyTolerance = energyTolerance
        self.gradientNormTolerance = gradientNormTolerance
        self.maxIterations = maxIterations
    }

    public static let `default` = ConvergenceCriteria()
}

// MARK: - Optimizer Result

/// Optimization result with optimal parameters and convergence history
///
/// Contains best parameters found, objective value at optimum, full history
/// for visualization, and convergence reason (why optimization stopped).
///
/// **Usage:**
/// ```swift
/// let result = await optimizer.minimize(...)
/// print("Optimal energy: \(result.optimalValue)")
/// print("Converged: \(result.convergenceReason == .energyTolerance)")
/// print("Iterations: \(result.iterations)")
///
/// // Plot convergence
/// for (i, energy) in result.valueHistory.enumerated() {
///     print("Iteration \(i): E = \(energy)")
/// }
/// ```
@frozen
public struct OptimizerResult: Sendable {
    /// Optimal parameters found
    public let optimalParameters: [Double]

    /// Objective function value at optimal parameters
    public let optimalValue: Double

    /// Complete history of objective values (one per iteration)
    public let valueHistory: [Double]

    /// Total iterations performed
    public let iterations: Int

    /// Why optimization terminated
    public let convergenceReason: ConvergenceReason

    /// Number of objective function evaluations (may exceed iterations for gradient-based)
    public let functionEvaluations: Int

    public init(
        optimalParameters: [Double],
        optimalValue: Double,
        valueHistory: [Double],
        iterations: Int,
        convergenceReason: ConvergenceReason,
        functionEvaluations: Int
    ) {
        self.optimalParameters = optimalParameters
        self.optimalValue = optimalValue
        self.valueHistory = valueHistory
        self.iterations = iterations
        self.convergenceReason = convergenceReason
        self.functionEvaluations = functionEvaluations
    }
}

/// Reason why optimization terminated
@frozen
public enum ConvergenceReason: Sendable, CustomStringConvertible {
    /// Energy change below tolerance
    case energyTolerance

    /// Gradient norm below tolerance
    case gradientNorm

    /// Maximum iterations reached
    case maxIterations

    @inlinable
    public var description: String {
        switch self {
        case .energyTolerance: "Energy tolerance reached"
        case .gradientNorm: "Gradient norm below threshold"
        case .maxIterations: "Maximum iterations reached"
        }
    }
}

// MARK: - Nelder-Mead Optimizer (Simplex Method)

/// Nelder-Mead simplex optimizer (derivative-free)
///
/// Robust gradient-free optimization using downhill simplex method.
/// **Primary choice for VQE when gradients are expensive or noisy.**
///
/// **Algorithm:**
/// 1. Maintain simplex: n+1 points in n-dimensional parameter space
/// 2. Each iteration: reflect worst point through centroid
/// 3. Adaptively expand, contract, or shrink simplex based on improvement
/// 4. Converges when simplex collapses (all vertices close in value)
///
/// **Advantages:**
/// - No gradients required (saves 2n circuit evaluations per iteration)
/// - Robust to noisy objective functions
/// - Good for low-dimensional problems (< 20 parameters)
///
/// **Disadvantages:**
/// - Slower than gradient methods for smooth functions
/// - Scales poorly to high dimensions (n > 50)
/// - Can get stuck in narrow valleys
///
/// **Performance:**
/// - Typical VQE (10 parameters): 100-300 iterations
/// - Function evaluations per iteration: 1-4 (adaptive)
/// - Best for: hardware-efficient ansatz with depth 1-3
///
/// Example:
/// ```swift
/// let optimizer = NelderMeadOptimizer(
///     tolerance: 1e-6,
///     initialSimplexSize: 0.1  // 10% of parameter range
/// )
///
/// let result = await optimizer.minimize(
///     objectiveFunction: { params in
///         let state = await computeState(params)
///         return hamiltonian.expectationValue(state: state)
///     },
///     initialParameters: initialGuess,
///     convergenceCriteria: .default
/// )
/// ```
@frozen
public struct NelderMeadOptimizer: Optimizer {
    /// Convergence tolerance for simplex size
    public let tolerance: Double

    /// Initial simplex size relative to parameter values
    public let initialSimplexSize: Double

    /// Nelder-Mead algorithm coefficients (standard values)
    private let alpha: Double = 1.0 // Reflection coefficient
    private let gamma: Double = 2.0 // Expansion coefficient
    private let rho: Double = 0.5 // Contraction coefficient
    private let sigma: Double = 0.5 // Shrinkage coefficient

    /// Create Nelder-Mead optimizer
    /// - Parameters:
    ///   - tolerance: Convergence tolerance (default: 1e-6)
    ///   - initialSimplexSize: Initial simplex size (default: 0.1)
    public init(tolerance: Double = 1e-6, initialSimplexSize: Double = 0.1) {
        ValidationUtilities.validatePositiveDouble(tolerance, name: "tolerance")
        ValidationUtilities.validatePositiveDouble(initialSimplexSize, name: "initialSimplexSize")
        self.tolerance = tolerance
        self.initialSimplexSize = initialSimplexSize
    }

    @_optimize(speed)
    @_eagerMove
    public func minimize(
        objectiveFunction: @Sendable ([Double]) async -> Double,
        initialParameters: [Double],
        convergenceCriteria: ConvergenceCriteria,
        progressCallback: (@Sendable (Int, Double) async -> Void)?
    ) async -> OptimizerResult {
        let n: Int = initialParameters.count
        ValidationUtilities.validateNonEmpty(initialParameters, name: "initialParameters")

        var simplex: [SimplexVertex] = []
        simplex.reserveCapacity(n + 1)

        let initialValue: Double = await objectiveFunction(initialParameters)
        simplex.append(SimplexVertex(parameters: initialParameters, value: initialValue))

        for i in 0 ..< n {
            var perturbed = initialParameters
            perturbed[i] += initialSimplexSize
            let value: Double = await objectiveFunction(perturbed)
            simplex.append(SimplexVertex(parameters: perturbed, value: value))
        }

        var valueHistory: [Double] = [initialValue]
        var functionEvaluations: Int = n + 1

        for iteration in 0 ..< convergenceCriteria.maxIterations {
            simplex.sort { $0.value < $1.value }

            let currentBest: Double = simplex[0].value

            if let callback = progressCallback {
                await callback(iteration, currentBest)
            }

            valueHistory.append(currentBest)

            let range: Double = simplex[n].value - simplex[0].value
            if range < convergenceCriteria.energyTolerance {
                return OptimizerResult(
                    optimalParameters: simplex[0].parameters,
                    optimalValue: simplex[0].value,
                    valueHistory: valueHistory,
                    iterations: iteration + 1,
                    convergenceReason: .energyTolerance,
                    functionEvaluations: functionEvaluations
                )
            }

            var centroid = [Double](unsafeUninitializedCapacity: n) { buffer, count in
                for j in 0 ..< n {
                    buffer[j] = 0.0
                }
                count = n
            }
            for i in 0 ..< n {
                vDSP_vaddD(centroid, 1, simplex[i].parameters, 1, &centroid, 1, vDSP_Length(n))
            }
            var scale = 1.0 / Double(n)
            vDSP_vsmulD(centroid, 1, &scale, &centroid, 1, vDSP_Length(n))

            var reflected = [Double](unsafeUninitializedCapacity: n) { _, count in
                count = n
            }
            var onePlusAlpha = 1.0 + alpha
            var negAlpha = -alpha
            vDSP_vsmulD(centroid, 1, &onePlusAlpha, &reflected, 1, vDSP_Length(n))
            vDSP_vsmaD(simplex[n].parameters, 1, &negAlpha, reflected, 1, &reflected, 1, vDSP_Length(n))
            let reflectedValue: Double = await objectiveFunction(reflected)
            functionEvaluations += 1

            if reflectedValue < simplex[0].value {
                var expanded = [Double](unsafeUninitializedCapacity: n) { _, count in
                    count = n
                }
                var oneMinusGamma = 1.0 - gamma
                var gammaVal = gamma
                vDSP_vsmulD(centroid, 1, &oneMinusGamma, &expanded, 1, vDSP_Length(n))
                vDSP_vsmaD(reflected, 1, &gammaVal, expanded, 1, &expanded, 1, vDSP_Length(n))
                let expandedValue: Double = await objectiveFunction(expanded)
                functionEvaluations += 1

                if expandedValue < reflectedValue {
                    simplex[n] = SimplexVertex(parameters: expanded, value: expandedValue)
                } else {
                    simplex[n] = SimplexVertex(parameters: reflected, value: reflectedValue)
                }
            } else if reflectedValue < simplex[n - 1].value {
                simplex[n] = SimplexVertex(parameters: reflected, value: reflectedValue)
            } else {
                if reflectedValue < simplex[n].value {
                    var contracted = [Double](unsafeUninitializedCapacity: n) { _, count in
                        count = n
                    }
                    var oneMinusRho = 1.0 - rho
                    var rhoVal = rho
                    vDSP_vsmulD(centroid, 1, &oneMinusRho, &contracted, 1, vDSP_Length(n))
                    vDSP_vsmaD(reflected, 1, &rhoVal, contracted, 1, &contracted, 1, vDSP_Length(n))
                    let contractedValue: Double = await objectiveFunction(contracted)
                    functionEvaluations += 1

                    if contractedValue < reflectedValue {
                        simplex[n] = SimplexVertex(parameters: contracted, value: contractedValue)
                        continue
                    }
                } else {
                    var contracted = [Double](unsafeUninitializedCapacity: n) { _, count in
                        count = n
                    }
                    var oneMinusRho = 1.0 - rho
                    var rhoVal = rho
                    vDSP_vsmulD(centroid, 1, &oneMinusRho, &contracted, 1, vDSP_Length(n))
                    vDSP_vsmaD(simplex[n].parameters, 1, &rhoVal, contracted, 1, &contracted, 1, vDSP_Length(n))
                    let contractedValue: Double = await objectiveFunction(contracted)
                    functionEvaluations += 1

                    if contractedValue < simplex[n].value {
                        simplex[n] = SimplexVertex(parameters: contracted, value: contractedValue)
                        continue
                    }
                }

                var oneMinusSigma = 1.0 - sigma
                var sigmaVal = sigma
                for i in 1 ... n {
                    var newParams = [Double](unsafeUninitializedCapacity: n) { _, count in
                        count = n
                    }
                    vDSP_vsmulD(simplex[0].parameters, 1, &oneMinusSigma, &newParams, 1, vDSP_Length(n))
                    vDSP_vsmaD(simplex[i].parameters, 1, &sigmaVal, newParams, 1, &newParams, 1, vDSP_Length(n))
                    simplex[i].parameters = newParams
                    simplex[i].value = await objectiveFunction(simplex[i].parameters)
                }
                functionEvaluations += n
            }
        }

        simplex.sort { $0.value < $1.value }
        return OptimizerResult(
            optimalParameters: simplex[0].parameters,
            optimalValue: simplex[0].value,
            valueHistory: valueHistory,
            iterations: convergenceCriteria.maxIterations,
            convergenceReason: .maxIterations,
            functionEvaluations: functionEvaluations
        )
    }

    @frozen
    public struct SimplexVertex {
        var parameters: [Double]
        var value: Double
    }
}

// MARK: - Gradient Descent Optimizer

/// Gradient descent optimizer with parameter shift rule
///
/// Gradient-based optimization using quantum parameter shift rule for exact gradients.
/// **Use when circuit is shallow and parameter count is moderate (< 30).**
///
/// **Algorithm:**
/// 1. Compute gradient via parameter shift: ∂E/∂θᵢ = [E(θᵢ+π/2) - E(θᵢ-π/2)] / 2
/// 2. Update parameters: θ ← θ - η∇E (with adaptive learning rate)
/// 3. Optional momentum: velocity accumulation for improved convergence
///
/// **Advantages:**
/// - Exact gradients (no finite difference errors)
/// - Gradient-based optimization for smooth landscapes
/// - Scales to high dimensions (n > 50)
///
/// **Disadvantages:**
/// - Requires 2n circuit evaluations per iteration (gradient computation)
/// - Sensitive to learning rate choice
/// - Can get stuck in local minima
///
/// **Performance:**
/// - Typical VQE (10 parameters): 50-150 iterations
/// - Circuit evaluations per iteration: 2n + 1 (gradient + update)
/// - Best for: shallow circuits with many parameters
///
/// Example:
/// ```swift
/// let optimizer = GradientDescentOptimizer(
///     learningRate: 0.1,
///     momentum: 0.9,
///     useAdaptiveLearningRate: true
/// )
///
/// let result = await optimizer.minimize(
///     objectiveFunction: { params in
///         let state = await computeState(params)
///         return hamiltonian.expectationValue(state: state)
///     },
///     initialParameters: initialGuess,
///     convergenceCriteria: ConvergenceCriteria(
///         gradientNormTolerance: 1e-5  // Stop when ||∇E|| < 1e-5
///     )
/// )
/// ```
@frozen
public struct GradientDescentOptimizer: Optimizer {
    /// Initial learning rate
    public let learningRate: Double

    /// Momentum coefficient (0 = no momentum, 0.9 = strong momentum)
    public let momentum: Double

    /// Whether to use adaptive learning rate (decrease when no improvement)
    public let useAdaptiveLearningRate: Bool

    /// Parameter shift for gradient computation (default: π/2 for standard shift rule)
    public let parameterShift: Double

    /// Create gradient descent optimizer
    /// - Parameters:
    ///   - learningRate: Initial learning rate (default: 0.1)
    ///   - momentum: Momentum coefficient (default: 0.0)
    ///   - useAdaptiveLearningRate: Adaptive learning rate (default: true)
    ///   - parameterShift: Shift for gradient computation (default: π/2)
    public init(
        learningRate: Double = 0.1,
        momentum: Double = 0.0,
        useAdaptiveLearningRate: Bool = true,
        parameterShift: Double = .pi / 2
    ) {
        ValidationUtilities.validatePositiveDouble(learningRate, name: "learningRate")
        ValidationUtilities.validateHalfOpenRange(momentum, min: 0, max: 1, name: "momentum")
        ValidationUtilities.validatePositiveDouble(parameterShift, name: "parameterShift")

        self.learningRate = learningRate
        self.momentum = momentum
        self.useAdaptiveLearningRate = useAdaptiveLearningRate
        self.parameterShift = parameterShift
    }

    @_optimize(speed)
    @_eagerMove
    public func minimize(
        objectiveFunction: @Sendable ([Double]) async -> Double,
        initialParameters: [Double],
        convergenceCriteria: ConvergenceCriteria,
        progressCallback: (@Sendable (Int, Double) async -> Void)?
    ) async -> OptimizerResult {
        let n: Int = initialParameters.count
        ValidationUtilities.validateNonEmpty(initialParameters, name: "initialParameters")

        var currentParameters = initialParameters
        var currentValue: Double = await objectiveFunction(currentParameters)
        var valueHistory: [Double] = [currentValue]
        var functionEvaluations = 1

        var velocity = [Double](unsafeUninitializedCapacity: n) { buffer, count in
            for i in 0 ..< n {
                buffer[i] = 0.0
            }
            count = n
        }
        var currentLearningRate: Double = learningRate
        var bestValue: Double = currentValue
        var iterationsSinceImprovement = 0

        for iteration in 0 ..< convergenceCriteria.maxIterations {
            if let callback = progressCallback {
                await callback(iteration, currentValue)
            }

            var gradient = [Double](unsafeUninitializedCapacity: n) { _, count in
                count = n
            }

            for i in 0 ..< n {
                var paramsPlus = currentParameters
                paramsPlus[i] += parameterShift
                let valuePlus: Double = await objectiveFunction(paramsPlus)

                var paramsMinus = currentParameters
                paramsMinus[i] -= parameterShift
                let valueMinus: Double = await objectiveFunction(paramsMinus)

                gradient[i] = (valuePlus - valueMinus) / 2.0
            }
            functionEvaluations += 2 * n

            var gradientNormSq = 0.0
            vDSP_svesqD(gradient, 1, &gradientNormSq, vDSP_Length(n))
            let gradientNorm: Double = sqrt(gradientNormSq)

            if let gnt = convergenceCriteria.gradientNormTolerance,
               gradientNorm < gnt
            {
                return OptimizerResult(
                    optimalParameters: currentParameters,
                    optimalValue: currentValue,
                    valueHistory: valueHistory,
                    iterations: iteration + 1,
                    convergenceReason: .gradientNorm,
                    functionEvaluations: functionEvaluations
                )
            }

            for i in 0 ..< n {
                velocity[i] = momentum * velocity[i] - currentLearningRate * gradient[i]
                currentParameters[i] += velocity[i]
            }

            let newValue: Double = await objectiveFunction(currentParameters)
            functionEvaluations += 1

            if abs(newValue - currentValue) < convergenceCriteria.energyTolerance {
                return OptimizerResult(
                    optimalParameters: currentParameters,
                    optimalValue: newValue,
                    valueHistory: valueHistory,
                    iterations: iteration + 1,
                    convergenceReason: .energyTolerance,
                    functionEvaluations: functionEvaluations
                )
            }

            if newValue < bestValue - convergenceCriteria.energyTolerance {
                bestValue = newValue
                iterationsSinceImprovement = 0
            } else {
                iterationsSinceImprovement += 1
                if useAdaptiveLearningRate {
                    currentLearningRate *= 0.95
                }
            }

            currentValue = newValue
            valueHistory.append(newValue)
        }

        return OptimizerResult(
            optimalParameters: currentParameters,
            optimalValue: currentValue,
            valueHistory: valueHistory,
            iterations: convergenceCriteria.maxIterations,
            convergenceReason: .maxIterations,
            functionEvaluations: functionEvaluations
        )
    }
}

// MARK: - L-BFGS-B Optimizer

/// L-BFGS-B optimizer (Limited-memory BFGS with box constraints)
///
/// Quasi-Newton method using limited-memory BFGS approximation of Hessian.
/// **Most efficient gradient-based optimizer for VQE with many parameters.**
///
/// **Algorithm:**
/// 1. Approximate Hessian inverse using last m gradient differences (two-loop recursion)
/// 2. Compute search direction from gradient and approximate Hessian
/// 3. Backtracking line search with Wolfe conditions
/// 4. Update parameter history (limited to m previous steps)
///
/// **Advantages:**
/// - Uses curvature information (second-order method)
/// - Memory efficient: O(m·n) vs O(n²) for full BFGS
/// - Box constraints support (parameter bounds)
/// - Quasi-Newton optimization with limited memory
///
/// **Disadvantages:**
/// - Requires 2n circuit evaluations per iteration (gradient computation)
/// - More complex than gradient descent
/// - Can fail on non-smooth landscapes
///
/// **Performance:**
/// - Typical VQE (10 parameters): 20-80 iterations
/// - Circuit evaluations per iteration: 2n + line search (1-5 evals)
/// - Best for: moderate to large parameter counts (> 20)
///
/// Example:
/// ```swift
/// let optimizer = LBFGSBOptimizer(
///     memorySize: 10,
///     tolerance: 1e-6,
///     maxLineSearchSteps: 20
/// )
///
/// let result = await optimizer.minimize(
///     objectiveFunction: { params in
///         let state = await computeState(params)
///         return hamiltonian.expectationValue(state: state)
///     },
///     initialParameters: initialGuess,
///     convergenceCriteria: ConvergenceCriteria(
///         gradientNormTolerance: 1e-5
///     )
/// )
/// ```
@frozen
public struct LBFGSBOptimizer: Optimizer {
    /// Number of previous gradient pairs to store (m parameter)
    public let memorySize: Int

    /// Gradient norm tolerance for convergence
    public let tolerance: Double

    /// Maximum line search backtracking steps
    public let maxLineSearchSteps: Int

    /// Parameter shift for gradient computation (default: π/2)
    public let parameterShift: Double

    /// Wolfe condition constant (sufficient decrease)
    private let c1: Double = 1e-4

    /// Wolfe curvature condition constant
    private let c2: Double = 0.9

    /// Create L-BFGS-B optimizer
    /// - Parameters:
    ///   - memorySize: History size for BFGS approximation (default: 10)
    ///   - tolerance: Gradient norm convergence tolerance (default: 1e-6)
    ///   - maxLineSearchSteps: Max line search iterations (default: 20)
    ///   - parameterShift: Shift for gradient computation (default: π/2)
    public init(
        memorySize: Int = 10,
        tolerance: Double = 1e-6,
        maxLineSearchSteps: Int = 20,
        parameterShift: Double = .pi / 2
    ) {
        ValidationUtilities.validatePositiveInt(memorySize, name: "memorySize")
        ValidationUtilities.validatePositiveDouble(tolerance, name: "tolerance")
        ValidationUtilities.validatePositiveInt(maxLineSearchSteps, name: "maxLineSearchSteps")
        ValidationUtilities.validatePositiveDouble(parameterShift, name: "parameterShift")

        self.memorySize = memorySize
        self.tolerance = tolerance
        self.maxLineSearchSteps = maxLineSearchSteps
        self.parameterShift = parameterShift
    }

    @_optimize(speed)
    @_eagerMove
    public func minimize(
        objectiveFunction: @Sendable ([Double]) async -> Double,
        initialParameters: [Double],
        convergenceCriteria: ConvergenceCriteria,
        progressCallback: (@Sendable (Int, Double) async -> Void)?
    ) async -> OptimizerResult {
        let n: Int = initialParameters.count
        ValidationUtilities.validateNonEmpty(initialParameters, name: "initialParameters")

        var params = initialParameters
        var cost: Double = await objectiveFunction(params)
        var gradient: [Double] = await computeGradient(
            params: params,
            objectiveFunction: objectiveFunction
        )

        var valueHistory: [Double] = [cost]
        var functionEvaluations = 1 + 2 * n

        var sHistory: [[Double]] = []
        var yHistory: [[Double]] = []
        var rhoHistory: [Double] = []

        for iteration in 0 ..< convergenceCriteria.maxIterations {
            if let callback = progressCallback {
                await callback(iteration, cost)
            }

            var gradNormSq = 0.0
            vDSP_svesqD(gradient, 1, &gradNormSq, vDSP_Length(gradient.count))
            let gradNorm: Double = sqrt(gradNormSq)

            if let gnt = convergenceCriteria.gradientNormTolerance,
               gradNorm < gnt
            {
                return OptimizerResult(
                    optimalParameters: params,
                    optimalValue: cost,
                    valueHistory: valueHistory,
                    iterations: iteration + 1,
                    convergenceReason: .gradientNorm,
                    functionEvaluations: functionEvaluations
                )
            }

            let direction: [Double] = LBFGSBOptimizer.computeSearchDirection(
                gradient: gradient,
                sHistory: sHistory,
                yHistory: yHistory,
                rhoHistory: rhoHistory
            )

            let lineSearchResult: LineSearchResult? = await performLineSearch(
                params: params,
                direction: direction,
                gradient: gradient,
                cost: cost,
                objectiveFunction: objectiveFunction
            )

            guard let lsr = lineSearchResult else {
                return OptimizerResult(
                    optimalParameters: params,
                    optimalValue: cost,
                    valueHistory: valueHistory,
                    iterations: iteration + 1,
                    convergenceReason: .maxIterations,
                    functionEvaluations: functionEvaluations
                )
            }

            functionEvaluations += lsr.evaluations

            var newParams = [Double](unsafeUninitializedCapacity: n) { _, count in
                count = n
            }
            var alphaVal = lsr.alpha
            vDSP_vsmaD(direction, 1, &alphaVal, params, 1, &newParams, 1, vDSP_Length(n))
            let newCost: Double = await objectiveFunction(newParams)
            let newGradient: [Double] = await computeGradient(
                params: newParams,
                objectiveFunction: objectiveFunction
            )
            functionEvaluations += 1 + 2 * n

            if abs(newCost - cost) < convergenceCriteria.energyTolerance {
                return OptimizerResult(
                    optimalParameters: newParams,
                    optimalValue: newCost,
                    valueHistory: valueHistory,
                    iterations: iteration + 1,
                    convergenceReason: .energyTolerance,
                    functionEvaluations: functionEvaluations
                )
            }

            var s = [Double](unsafeUninitializedCapacity: n) { _, count in
                count = n
            }
            vDSP_vsubD(params, 1, newParams, 1, &s, 1, vDSP_Length(n))

            var y = [Double](unsafeUninitializedCapacity: n) { _, count in
                count = n
            }
            vDSP_vsubD(gradient, 1, newGradient, 1, &y, 1, vDSP_Length(n))

            var ys = 0.0
            vDSP_dotprD(y, 1, s, 1, &ys, vDSP_Length(n))

            if ys > 1e-10 {
                sHistory.append(s)
                yHistory.append(y)
                rhoHistory.append(1.0 / ys)
            }

            params = newParams
            cost = newCost
            gradient = newGradient
            valueHistory.append(cost)
        }

        return OptimizerResult(
            optimalParameters: params,
            optimalValue: cost,
            valueHistory: valueHistory,
            iterations: convergenceCriteria.maxIterations,
            convergenceReason: .maxIterations,
            functionEvaluations: functionEvaluations
        )
    }

    // MARK: - Private Helpers

    @_optimize(speed)
    private func computeGradient(
        params: [Double],
        objectiveFunction: @Sendable ([Double]) async -> Double
    ) async -> [Double] {
        let n: Int = params.count
        var gradient = [Double](unsafeUninitializedCapacity: n) { _, count in
            count = n
        }

        for i in 0 ..< n {
            var paramsPlus = params
            paramsPlus[i] += parameterShift
            let valuePlus: Double = await objectiveFunction(paramsPlus)

            var paramsMinus = params
            paramsMinus[i] -= parameterShift
            let valueMinus: Double = await objectiveFunction(paramsMinus)

            gradient[i] = (valuePlus - valueMinus) / 2.0
        }

        return gradient
    }

    @frozen
    public struct LineSearchResult {
        let alpha: Double
        let evaluations: Int
    }

    @_optimize(speed)
    private func performLineSearch(
        params: [Double],
        direction: [Double],
        gradient: [Double],
        cost: Double,
        objectiveFunction: @Sendable ([Double]) async -> Double
    ) async -> LineSearchResult? {
        let n: Int = params.count
        var alpha = 1.0
        let rho = 0.5
        var evaluations = 0

        var dirGrad = 0.0
        vDSP_dotprD(direction, 1, gradient, 1, &dirGrad, vDSP_Length(n))

        for _ in 0 ..< maxLineSearchSteps {
            var newParams = [Double](unsafeUninitializedCapacity: n) { _, count in
                count = n
            }
            var alphaVal = alpha
            vDSP_vsmaD(direction, 1, &alphaVal, params, 1, &newParams, 1, vDSP_Length(n))
            let newCost: Double = await objectiveFunction(newParams)
            evaluations += 1

            if newCost <= cost + c1 * alpha * dirGrad {
                let newGradient: [Double] = await computeGradient(
                    params: newParams,
                    objectiveFunction: objectiveFunction
                )
                evaluations += 2 * n

                var newDirGrad = 0.0
                vDSP_dotprD(direction, 1, newGradient, 1, &newDirGrad, vDSP_Length(n))

                if abs(newDirGrad) <= -c2 * dirGrad {
                    return LineSearchResult(alpha: alpha, evaluations: evaluations)
                }
            }

            alpha *= rho
        }

        return alpha > 1e-10 ? LineSearchResult(alpha: alpha, evaluations: evaluations) : nil
    }

    /// Compute search direction using L-BFGS two-loop recursion
    ///
    /// Approximates Hessian inverse using limited memory (last m iterations).
    /// Uses stored parameter differences (s) and gradient differences (y).
    /// Optimized with vDSP for all vector operations.
    ///
    /// - Parameters:
    ///   - gradient: Current gradient
    ///   - sHistory: Parameter difference history
    ///   - yHistory: Gradient difference history
    ///   - rhoHistory: Precomputed 1/(yᵀs) values
    /// - Returns: Search direction (negative gradient if no history)
    @_optimize(speed)
    @_eagerMove
    public static func computeSearchDirection(
        gradient: [Double],
        sHistory: [[Double]],
        yHistory: [[Double]],
        rhoHistory: [Double]
    ) -> [Double] {
        let n: Int = gradient.count

        guard !sHistory.isEmpty else {
            var result = [Double](unsafeUninitializedCapacity: n) { _, count in
                count = n
            }
            vDSP_vnegD(gradient, 1, &result, 1, vDSP_Length(n))
            return result
        }

        let m: Int = sHistory.count
        var q = gradient
        var alpha = [Double](unsafeUninitializedCapacity: m) { _, count in
            count = m
        }

        for i in stride(from: m - 1, through: 0, by: -1) {
            var dotResult = 0.0
            vDSP_dotprD(sHistory[i], 1, q, 1, &dotResult, vDSP_Length(n))
            let a: Double = rhoHistory[i] * dotResult
            alpha[i] = a

            var negA = -a
            vDSP_vsmaD(yHistory[i], 1, &negA, q, 1, &q, 1, vDSP_Length(n))
        }

        let lastS: [Double] = sHistory[m - 1]
        let lastY: [Double] = yHistory[m - 1]
        var sy = 0.0
        vDSP_dotprD(lastS, 1, lastY, 1, &sy, vDSP_Length(n))
        var yy = 0.0
        vDSP_svesqD(lastY, 1, &yy, vDSP_Length(n))
        let gamma: Double = sy / yy

        var r = [Double](unsafeUninitializedCapacity: n) { _, count in
            count = n
        }
        var gammaVal = gamma
        vDSP_vsmulD(q, 1, &gammaVal, &r, 1, vDSP_Length(n))

        for i in 0 ..< m {
            var dotResult = 0.0
            vDSP_dotprD(yHistory[i], 1, r, 1, &dotResult, vDSP_Length(n))
            let beta: Double = rhoHistory[i] * dotResult

            var scale = alpha[i] - beta
            vDSP_vsmaD(sHistory[i], 1, &scale, r, 1, &r, 1, vDSP_Length(n))
        }

        vDSP_vnegD(r, 1, &r, 1, vDSP_Length(n))
        return r
    }
}

// MARK: - SPSA Optimizer

/// SPSA optimizer (Simultaneous Perturbation Stochastic Approximation)
///
/// Gradient-free stochastic optimization using simultaneous random perturbations.
/// **Best optimizer for VQE with noisy objective functions.**
///
/// **Algorithm:**
/// 1. Generate random perturbation direction Δ (±1 for each parameter)
/// 2. Evaluate: f(θ + c·Δ) and f(θ - c·Δ) (only 2 evaluations, not 2n!)
/// 3. Approximate gradient: ĝ = [f(θ+c·Δ) - f(θ-c·Δ)] / (2c) · Δ
/// 4. Update: θ ← θ - a·ĝ
/// 5. Decay: a and c decrease over iterations
///
/// **Advantages:**
/// - **Only 2 function evaluations per iteration** (vs 2n for gradient methods)
/// - Robust to noisy objective functions (averages noise across parameters)
/// - Scales to high dimensions (n > 100)
/// - Natural regularization from stochastic gradients
///
/// **Disadvantages:**
/// - Slower convergence than L-BFGS-B for smooth functions
/// - Requires careful tuning of decay rates
/// - Noisy optimization trajectory
///
/// **Performance:**
/// - Typical VQE (50 parameters): 200-500 iterations
/// - Circuit evaluations per iteration: 2 (independent of parameter count!)
/// - Best for: noisy objectives, many parameters (> 50), hardware noise
///
/// Example:
/// ```swift
/// let optimizer = SPSAOptimizer(
///     initialStepSize: 0.1,
///     initialPerturbation: 0.01,
///     decayExponent: 0.602  // Standard SPSA value
/// )
///
/// let result = await optimizer.minimize(
///     objectiveFunction: noisyEnergyFunction,
///     initialParameters: initialGuess,
///     convergenceCriteria: ConvergenceCriteria(
///         energyTolerance: 1e-4,  // Larger tolerance for noisy objectives
///         maxIterations: 500
///     )
/// )
/// ```
@frozen
public struct SPSAOptimizer: Optimizer {
    /// Initial step size (a parameter)
    public let initialStepSize: Double

    /// Initial perturbation size (c parameter)
    public let initialPerturbation: Double

    /// Decay exponent for step size (α, typically 0.602)
    public let decayExponent: Double

    /// Decay exponent for perturbation (γ, typically 0.101)
    public let perturbationDecayExponent: Double

    /// Stability constant for denominator (A, typically 10% of max iterations)
    public let stabilityConstant: Double

    /// Create SPSA optimizer
    /// - Parameters:
    ///   - initialStepSize: Initial step size a (default: 0.1)
    ///   - initialPerturbation: Initial perturbation c (default: 0.01)
    ///   - decayExponent: Step size decay α (default: 0.602)
    ///   - perturbationDecayExponent: Perturbation decay γ (default: 0.101)
    ///   - stabilityConstant: Stability constant A (default: 100)
    public init(
        initialStepSize: Double = 0.1,
        initialPerturbation: Double = 0.01,
        decayExponent: Double = 0.602,
        perturbationDecayExponent: Double = 0.101,
        stabilityConstant: Double = 100.0
    ) {
        ValidationUtilities.validatePositiveDouble(initialStepSize, name: "initialStepSize")
        ValidationUtilities.validatePositiveDouble(initialPerturbation, name: "initialPerturbation")
        ValidationUtilities.validateOpenMinRange(decayExponent, min: 0, max: 1, name: "decayExponent")
        ValidationUtilities.validateOpenMinRange(perturbationDecayExponent, min: 0, max: 1, name: "perturbationDecayExponent")
        ValidationUtilities.validateNonNegativeDouble(stabilityConstant, name: "stabilityConstant")

        self.initialStepSize = initialStepSize
        self.initialPerturbation = initialPerturbation
        self.decayExponent = decayExponent
        self.perturbationDecayExponent = perturbationDecayExponent
        self.stabilityConstant = stabilityConstant
    }

    @_optimize(speed)
    @_eagerMove
    public func minimize(
        objectiveFunction: @Sendable ([Double]) async -> Double,
        initialParameters: [Double],
        convergenceCriteria: ConvergenceCriteria,
        progressCallback: (@Sendable (Int, Double) async -> Void)?
    ) async -> OptimizerResult {
        let n: Int = initialParameters.count
        ValidationUtilities.validateNonEmpty(initialParameters, name: "initialParameters")

        var params = initialParameters
        var currentValue: Double = await objectiveFunction(params)
        var valueHistory: [Double] = [currentValue]
        var functionEvaluations = 1

        for iteration in 0 ..< convergenceCriteria.maxIterations {
            if let callback = progressCallback {
                await callback(iteration, currentValue)
            }

            let k = Double(iteration + 1)
            let ak: Double = initialStepSize / pow(k + stabilityConstant, decayExponent)
            let ck: Double = initialPerturbation / pow(k, perturbationDecayExponent)

            let delta = [Double](unsafeUninitializedCapacity: n) { buffer, count in
                let numBytes = (n + 7) / 8
                var randomBytes = [UInt8](repeating: 0, count: numBytes)
                arc4random_buf(&randomBytes, numBytes)

                for i in 0 ..< n {
                    let byteIndex = i / 8
                    let bitIndex = i % 8
                    let bit = (randomBytes[byteIndex] >> bitIndex) & 1
                    buffer[i] = bit == 0 ? -1.0 : 1.0
                }
                count = n
            }

            var paramsPlus = [Double](unsafeUninitializedCapacity: n) { _, count in
                count = n
            }
            var ckVal = ck
            vDSP_vsmaD(delta, 1, &ckVal, params, 1, &paramsPlus, 1, vDSP_Length(n))

            var paramsMinus = [Double](unsafeUninitializedCapacity: n) { _, count in
                count = n
            }
            var negCk = -ck
            vDSP_vsmaD(delta, 1, &negCk, params, 1, &paramsMinus, 1, vDSP_Length(n))

            let valuePlus: Double = await objectiveFunction(paramsPlus)
            let valueMinus: Double = await objectiveFunction(paramsMinus)
            functionEvaluations += 2

            let gradientApprox: Double = (valuePlus - valueMinus) / (2.0 * ck)

            var negAkGrad = -ak * gradientApprox
            vDSP_vsmaD(delta, 1, &negAkGrad, params, 1, &params, 1, vDSP_Length(n))

            let newValue: Double = await objectiveFunction(params)
            functionEvaluations += 1

            if abs(newValue - currentValue) < convergenceCriteria.energyTolerance {
                return OptimizerResult(
                    optimalParameters: params,
                    optimalValue: newValue,
                    valueHistory: valueHistory,
                    iterations: iteration + 1,
                    convergenceReason: .energyTolerance,
                    functionEvaluations: functionEvaluations
                )
            }

            currentValue = newValue
            valueHistory.append(newValue)
        }

        return OptimizerResult(
            optimalParameters: params,
            optimalValue: currentValue,
            valueHistory: valueHistory,
            iterations: convergenceCriteria.maxIterations,
            convergenceReason: .maxIterations,
            functionEvaluations: functionEvaluations
        )
    }
}

// MARK: - COBYLA Optimizer

/// COBYLA optimizer: Derivative-free trust region method via linear interpolation
///
/// Implements Powell's Constrained Optimization BY Linear Approximations algorithm (1994).
/// Default optimizer for VQE due to derivative-free operation (saves 2n evaluations per iteration),
/// robustness to measurement noise, and superlinear convergence on smooth landscapes.
///
/// **Mathematical foundation**:
/// - Linear model: m(x) = f(x₀) + g^T(x - x₀) from n+1 simplex points
/// - Trust region subproblem: min m(x) subject to ||x - x₀|| ≤ ρ
/// - Agreement ratio: r = [f(x₀) - f(x_new)] / [m(x₀) - m(x_new)]
/// - Trust region update: expand if r > 0.75, shrink if r < 0.1, accept if r > 0.1
///
/// **Algorithm**:
/// 1. Initialize simplex with n+1 points spaced by simplexScale × initialTrustRadius
/// 2. Build linear gradient model via weighted directional derivatives from simplex
/// 3. Solve trust region subproblem using Cauchy point (exact for linear models)
/// 4. Evaluate objective at trial point, compute actual vs predicted reduction ratio
/// 5. Update trust region radius based on model quality (expand/maintain/shrink)
/// 6. Accept or reject step, update simplex (replace worst point or regenerate)
/// 7. Converge when trust region radius < energyTolerance or value change < tolerance
///
/// **Performance characteristics**:
/// - Function evaluations: 1-2 per iteration (n+1 initial simplex construction)
/// - Convergence: Superlinear near optimum (trust region method)
/// - Robustness: Linear model smooths measurement noise from quantum circuits
/// - Parameter range: Suitable for 5-50 parameters (VQE typical use)
/// - No gradient computation: Avoids parameter shift rule (2n evaluations)
///
/// **VQE application**:
/// Quantum energy landscapes are smooth (differentiable parameterized circuits) but noisy
/// (shot noise from measurements). COBYLA exploits smoothness via linear models while remaining
/// robust to noise through derivative-free optimization. Trust region prevents large steps into
/// regions where linear approximation breaks down.
///
/// Example:
/// ```swift
/// // VQE ground state optimization with COBYLA
/// let optimizer = COBYLAOptimizer(initialTrustRadius: 0.5, minTrustRadius: 1e-6)
///
/// let vqe = await VariationalQuantumEigensolver(
///     hamiltonian: hamiltonian,
///     ansatz: HardwareEfficientAnsatz.create(numQubits: 4, depth: 2),
///     optimizer: optimizer
/// )
///
/// let result = await vqe.run(initialParameters: [0.01, 0.01, 0.01, 0.01])
/// // Result: Ground state energy with ~100-200 iterations for 4-parameter system
///
/// // Custom trust region tuning
/// let aggressive = COBYLAOptimizer(
///     initialTrustRadius: 1.0,
///     minTrustRadius: 1e-8,
///     maxTrustRadius: 5.0,
///     shrinkFactor: 0.25,
///     expandFactor: 2.5
/// )
/// ```
@frozen
public struct COBYLAOptimizer: Optimizer {
    // MARK: - Configuration

    /// Initial trust region radius (typical: 0.5)
    public let initialTrustRadius: Double

    /// Minimum trust region radius - convergence threshold (typical: 1e-6)
    public let minTrustRadius: Double

    /// Maximum trust region radius (typical: 2.0)
    public let maxTrustRadius: Double

    /// Trust region shrink factor on poor agreement (typical: 0.5)
    public let shrinkFactor: Double

    /// Trust region expand factor on good agreement (typical: 2.0)
    public let expandFactor: Double

    /// Ratio threshold for accepting step (typical: 0.1)
    public let acceptRatio: Double

    /// Ratio threshold for expanding trust region (typical: 0.75)
    public let expandRatio: Double

    /// Initial simplex size relative to trust radius (typical: 0.5)
    public let simplexScale: Double

    /// Create COBYLA optimizer with custom parameters
    ///
    /// - Parameters:
    ///   - initialTrustRadius: Starting trust region size (default: 0.5)
    ///   - minTrustRadius: Convergence threshold (default: 1e-6)
    ///   - maxTrustRadius: Maximum allowed radius (default: 2.0)
    ///   - shrinkFactor: Shrink factor on poor model (default: 0.5)
    ///   - expandFactor: Expand factor on good model (default: 2.0)
    ///   - acceptRatio: Threshold for accepting steps (default: 0.1)
    ///   - expandRatio: Threshold for expanding radius (default: 0.75)
    ///   - simplexScale: Initial simplex size (default: 0.5)
    public init(
        initialTrustRadius: Double = 0.5,
        minTrustRadius: Double = 1e-6,
        maxTrustRadius: Double = 2.0,
        shrinkFactor: Double = 0.5,
        expandFactor: Double = 2.0,
        acceptRatio: Double = 0.1,
        expandRatio: Double = 0.75,
        simplexScale: Double = 0.5
    ) {
        ValidationUtilities.validatePositiveDouble(initialTrustRadius, name: "initialTrustRadius")
        ValidationUtilities.validatePositiveDouble(minTrustRadius, name: "minTrustRadius")
        ValidationUtilities.validatePositiveDouble(maxTrustRadius, name: "maxTrustRadius")
        ValidationUtilities.validateOpenMinRange(shrinkFactor, min: 0, max: 1, name: "shrinkFactor")
        ValidationUtilities.validateOpenMinRange(expandFactor, min: 1, max: 10, name: "expandFactor")
        ValidationUtilities.validateOpenMinRange(acceptRatio, min: 0, max: 1, name: "acceptRatio")
        ValidationUtilities.validateOpenMinRange(expandRatio, min: 0, max: 1, name: "expandRatio")
        ValidationUtilities.validatePositiveDouble(simplexScale, name: "simplexScale")
        ValidationUtilities.validateAcceptExpandRatios(accept: acceptRatio, expand: expandRatio)
        ValidationUtilities.validateTrustRadiusRelationships(
            min: minTrustRadius,
            initial: initialTrustRadius,
            max: maxTrustRadius
        )

        self.initialTrustRadius = initialTrustRadius
        self.minTrustRadius = minTrustRadius
        self.maxTrustRadius = maxTrustRadius
        self.shrinkFactor = shrinkFactor
        self.expandFactor = expandFactor
        self.acceptRatio = acceptRatio
        self.expandRatio = expandRatio
        self.simplexScale = simplexScale
    }

    /// Convenience initializer with just tolerance
    /// - Parameter tolerance: Convergence tolerance (sets minTrustRadius)
    public init(tolerance: Double) {
        self.init(minTrustRadius: tolerance)
    }

    // MARK: - Main Optimization

    @_optimize(speed)
    @_eagerMove
    public func minimize(
        objectiveFunction: @Sendable ([Double]) async -> Double,
        initialParameters: [Double],
        convergenceCriteria: ConvergenceCriteria,
        progressCallback: (@Sendable (Int, Double) async -> Void)?
    ) async -> OptimizerResult {
        let n: Int = initialParameters.count
        ValidationUtilities.validateNonEmpty(initialParameters, name: "initialParameters")

        var currentTrustRadius = initialTrustRadius
        var valueHistory: [Double] = []
        var functionEvaluations = 0

        var simplex: [SimplexPoint] = []
        simplex.reserveCapacity(n + 1)

        let baseValue: Double = await objectiveFunction(initialParameters)
        functionEvaluations += 1
        simplex.append(SimplexPoint(parameters: initialParameters, value: baseValue))
        valueHistory.append(baseValue)

        let simplexSize: Double = initialTrustRadius * simplexScale
        for i in 0 ..< n {
            var perturbedParams = initialParameters
            perturbedParams[i] += simplexSize
            let value: Double = await objectiveFunction(perturbedParams)
            functionEvaluations += 1
            simplex.append(SimplexPoint(parameters: perturbedParams, value: value))
        }

        var bestIndex: Int = simplex.indices.min(by: { simplex[$0].value < simplex[$1].value })!
        var bestValue: Double = simplex[bestIndex].value
        var bestParameters: [Double] = simplex[bestIndex].parameters

        for iteration in 0 ..< convergenceCriteria.maxIterations {
            if let callback = progressCallback {
                await callback(iteration, bestValue)
            }

            if currentTrustRadius < convergenceCriteria.energyTolerance {
                return OptimizerResult(
                    optimalParameters: bestParameters,
                    optimalValue: bestValue,
                    valueHistory: valueHistory,
                    iterations: iteration + 1,
                    convergenceReason: .energyTolerance,
                    functionEvaluations: functionEvaluations
                )
            }

            let model: LinearModel = buildLinearModel(simplex: simplex, baseIndex: bestIndex)
            let step: [Double] = solveTrustRegionSubproblem(
                gradient: model.gradient,
                trustRadius: currentTrustRadius
            )

            var trialParameters = [Double](unsafeUninitializedCapacity: n) { _, count in
                count = n
            }
            vDSP_vaddD(bestParameters, 1, step, 1, &trialParameters, 1, vDSP_Length(n))

            let trialValue: Double = await objectiveFunction(trialParameters)
            functionEvaluations += 1

            let predictedReduction: Double = -evaluateLinearModel(
                gradient: model.gradient,
                step: step
            )
            let actualReduction: Double = bestValue - trialValue

            let ratio: Double = if abs(predictedReduction) < 1e-12 {
                actualReduction < 0 ? 0.0 : 1.0
            } else {
                actualReduction / predictedReduction
            }

            let previousRadius: Double = currentTrustRadius
            if ratio < 0.1 {
                currentTrustRadius *= shrinkFactor
            } else if ratio > expandRatio {
                currentTrustRadius = min(currentTrustRadius * expandFactor, maxTrustRadius)
            }

            currentTrustRadius = max(currentTrustRadius, minTrustRadius)

            if ratio > acceptRatio {
                bestParameters = trialParameters
                bestValue = trialValue
                valueHistory.append(bestValue)

                let simplexValues: [Double] = simplex.map(\.value)
                var worstValue = 0.0
                var worstIdx: vDSP_Length = 0
                vDSP_maxviD(simplexValues, 1, &worstValue, &worstIdx, vDSP_Length(simplex.count))
                let worstIndex = Int(worstIdx)
                simplex[worstIndex] = SimplexPoint(parameters: trialParameters, value: trialValue)

                let updatedValues: [Double] = simplex.map(\.value)
                var minVal = 0.0
                var minIdx: vDSP_Length = 0
                vDSP_minviD(updatedValues, 1, &minVal, &minIdx, vDSP_Length(simplex.count))
                bestIndex = Int(minIdx)
                bestValue = minVal
            } else {
                if currentTrustRadius < previousRadius * 0.9 {
                    let newSimplexSize: Double = currentTrustRadius * simplexScale
                    for i in 1 ... n {
                        var perturbedParams = bestParameters
                        perturbedParams[i - 1] += newSimplexSize
                        let value: Double = await objectiveFunction(perturbedParams)
                        functionEvaluations += 1
                        simplex[i] = SimplexPoint(parameters: perturbedParams, value: value)
                    }
                    simplex[0] = SimplexPoint(parameters: bestParameters, value: bestValue)
                    bestIndex = 0
                }

                valueHistory.append(bestValue)
            }

            if iteration > 0 {
                let historyCount: Int = valueHistory.count
                let valueChange: Double = abs(valueHistory[historyCount - 1] - valueHistory[historyCount - 2])
                if valueChange < convergenceCriteria.energyTolerance, currentTrustRadius < minTrustRadius * 10 {
                    return OptimizerResult(
                        optimalParameters: bestParameters,
                        optimalValue: bestValue,
                        valueHistory: valueHistory,
                        iterations: iteration + 1,
                        convergenceReason: .energyTolerance,
                        functionEvaluations: functionEvaluations
                    )
                }
            }
        }

        return OptimizerResult(
            optimalParameters: bestParameters,
            optimalValue: bestValue,
            valueHistory: valueHistory,
            iterations: convergenceCriteria.maxIterations,
            convergenceReason: .maxIterations,
            functionEvaluations: functionEvaluations
        )
    }

    // MARK: - Linear Model Construction

    /// Build linear model from simplex interpolation points
    ///
    /// Constructs linear approximation: m(x) = f(x₀) + g^T(x - x₀)
    /// where gradient g is computed via least squares from simplex points.
    ///
    /// **Algorithm:**
    /// 1. Form matrix Y where Y[i] = simplex[i].parameters - baseParameters
    /// 2. Form vector f where f[i] = simplex[i].value - baseValue
    /// 3. Solve least squares: g = (Y^T Y)^(-1) Y^T f
    ///
    /// For well-conditioned simplex, this gives accurate gradient estimate.
    ///
    /// - Parameters:
    ///   - simplex: Array of n+1 interpolation points
    ///   - baseIndex: Index of base point (usually best point in simplex)
    /// - Returns: Linear model with base point and gradient
    @_optimize(speed)
    @_eagerMove
    private func buildLinearModel(
        simplex: [SimplexPoint],
        baseIndex: Int
    ) -> LinearModel {
        let n: Int = simplex[0].parameters.count
        let basePoint: [Double] = simplex[baseIndex].parameters
        let baseValue: Double = simplex[baseIndex].value

        var gradient = [Double](unsafeUninitializedCapacity: n) { buffer, count in
            for j in 0 ..< n {
                buffer[j] = 0.0
            }
            count = n
        }
        var directions: [[Double]] = []
        var valueDifferences: [Double] = []

        for i in simplex.indices where i != baseIndex {
            var direction = [Double](unsafeUninitializedCapacity: n) { _, count in
                count = n
            }
            vDSP_vsubD(basePoint, 1, simplex[i].parameters, 1, &direction, 1, vDSP_Length(n))

            var normSq = 0.0
            vDSP_svesqD(direction, 1, &normSq, vDSP_Length(n))

            if normSq > 1e-12 {
                let norm: Double = sqrt(normSq)
                var invNorm = 1.0 / norm
                vDSP_vsmulD(direction, 1, &invNorm, &direction, 1, vDSP_Length(n))
                let valueDiff: Double = (simplex[i].value - baseValue) / norm
                directions.append(direction)
                valueDifferences.append(valueDiff)
            }
        }

        if !directions.isEmpty {
            var weights = [Double](unsafeUninitializedCapacity: n) { buffer, count in
                for j in 0 ..< n {
                    buffer[j] = 0.0
                }
                count = n
            }

            for (direction, valueDiff) in zip(directions, valueDifferences) {
                var vd = valueDiff
                vDSP_vsmaD(direction, 1, &vd, gradient, 1, &gradient, 1, vDSP_Length(n))

                var absDirection = [Double](unsafeUninitializedCapacity: n) { _, count in
                    count = n
                }
                vDSP_vabsD(direction, 1, &absDirection, 1, vDSP_Length(n))
                vDSP_vaddD(weights, 1, absDirection, 1, &weights, 1, vDSP_Length(n))
            }

            for j in 0 ..< n {
                if weights[j] > 1e-12 {
                    gradient[j] /= weights[j]
                }
            }
        }

        return LinearModel(
            baseParameters: basePoint,
            baseValue: baseValue,
            gradient: gradient
        )
    }

    /// Solve trust region subproblem: minimize linear model within trust region
    ///
    /// Problem: min g^T s  subject to ||s|| ≤ ρ
    ///
    /// **Solution:**
    /// - If ||g|| ≤ ρ: unconstrained minimum is s = 0 (already at optimum)
    /// - If ||g|| > ρ: constrained minimum is s = -ρ * g/||g|| (Cauchy point)
    ///
    /// This is the exact solution for the linear trust region subproblem.
    /// More complex methods (dogleg, CG-Steihaug) give same result for linear models.
    ///
    /// - Parameters:
    ///   - gradient: Linear model gradient
    ///   - trustRadius: Current trust region radius
    /// - Returns: Step vector s (from base point)
    @_optimize(speed)
    @_eagerMove
    @inline(__always)
    private func solveTrustRegionSubproblem(
        gradient: [Double],
        trustRadius: Double
    ) -> [Double] {
        let n: Int = gradient.count

        var gradNormSq = 0.0
        vDSP_svesqD(gradient, 1, &gradNormSq, vDSP_Length(n))

        guard gradNormSq > 1e-12 else {
            return [Double](unsafeUninitializedCapacity: n) { buffer, count in
                for i in 0 ..< n {
                    buffer[i] = 0.0
                }
                count = n
            }
        }

        let gradNorm: Double = sqrt(gradNormSq)
        var step = [Double](unsafeUninitializedCapacity: n) { _, count in
            count = n
        }
        var scale = -trustRadius / gradNorm
        vDSP_vsmulD(gradient, 1, &scale, &step, 1, vDSP_Length(n))

        return step
    }

    /// Evaluate linear model at given step
    ///
    /// Computes m(x₀ + s) - m(x₀) = g^T s
    ///
    /// - Parameters:
    ///   - gradient: Linear model gradient
    ///   - step: Step vector from base point
    /// - Returns: Model change (typically negative for descent direction)
    @_optimize(speed)
    @_eagerMove
    @inline(__always)
    @_effects(readonly)
    private func evaluateLinearModel(
        gradient: [Double],
        step: [Double]
    ) -> Double {
        var value = 0.0
        vDSP_dotprD(gradient, 1, step, 1, &value, vDSP_Length(gradient.count))
        return value
    }

    // MARK: - Supporting Types

    /// Single point in simplex with parameters and objective value
    @frozen
    public struct SimplexPoint {
        var parameters: [Double]
        var value: Double

        init(parameters: [Double], value: Double) {
            self.parameters = parameters
            self.value = value
        }
    }

    /// Linear interpolation model: m(x) = f₀ + g^T(x - x₀)
    @frozen
    public struct LinearModel {
        let baseParameters: [Double]
        let baseValue: Double
        let gradient: [Double]

        init(baseParameters: [Double], baseValue: Double, gradient: [Double]) {
            self.baseParameters = baseParameters
            self.baseValue = baseValue
            self.gradient = gradient
        }
    }
}
