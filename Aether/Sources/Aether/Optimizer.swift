// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate
import Foundation

/// Classical optimizer for variational quantum algorithms
///
/// Protocol defining optimization interface for VQE and QAOA. Implementations
/// minimize objective functions over parameter space using various strategies:
/// gradient-based (requires derivatives) or gradient-free (simplex, stochastic).
///
/// For most VQE applications, ``COBYLAOptimizer`` is the recommended default due to
/// derivative-free operation and noise robustness. ``NelderMeadOptimizer`` offers a
/// simpler alternative, while ``LBFGSBOptimizer`` excels on smooth landscapes with
/// many parameters. ``GradientDescentOptimizer`` provides basic gradient optimization,
/// and ``SPSAOptimizer`` handles noisy objectives or very high dimensions efficiently.
/// Gradient-free methods require 1-4 function evaluations per iteration, while
/// gradient-based methods require 2n+1 evaluations where n is the parameter count.
///
/// **Example:**
/// ```swift
/// let optimizer = COBYLAOptimizer(tolerance: 1e-6)
/// let result = await optimizer.minimize(
///     { params in
///         let state = circuit.bound(with: params).execute()
///         return hamiltonian.expectationValue(state: state)
///     },
///     from: [0.1, 0.1, 0.1],
///     using: ConvergenceCriteria(),
///     progress: nil
/// )
/// ```
///
/// - Complexity: O(iterations x evaluations_per_iteration), where evaluations depends on algorithm
/// - SeeAlso:
///   - ``COBYLAOptimizer`` for recommended default
///   - ``ConvergenceCriteria`` for termination conditions
///   - ``OptimizerResult`` for output structure
public protocol Optimizer: Sendable {
    /// Minimize objective function over parameter space
    ///
    /// Iteratively evaluates objective function to find parameters that minimize it.
    /// All implementations support async objectives (quantum circuit evaluation),
    /// optional progress callbacks, and configurable convergence criteria.
    ///
    /// - Parameters:
    ///   - objectiveFunction: Async function to minimize (typically energy expectation)
    ///   - initialParameters: Starting point in parameter space
    ///   - convergenceCriteria: Termination conditions (energy tolerance, max iterations)
    ///   - progress: Optional callback receiving iteration number and current objective value
    /// - Returns: Optimization result with optimal parameters and convergence history
    /// - Complexity: Implementation-dependent, see specific optimizer documentation
    func minimize(
        _ objectiveFunction: @Sendable ([Double]) async -> Double,
        from initialParameters: [Double],
        using convergenceCriteria: ConvergenceCriteria,
        progress: ProgressCallback?,
    ) async -> OptimizerResult
}

/// Progress callback for optimization iterations
///
/// Receives iteration number and current objective value at each optimization step.
/// Use for UI updates, logging, or early termination logic.
///
/// **Example:**
/// ```swift
/// let progress: ProgressCallback = { iteration, value in
///     print("Iteration \(iteration): E = \(value)")
/// }
/// ```
///
/// - SeeAlso: ``Optimizer/minimize(_:from:using:progress:)`` for usage context
public typealias ProgressCallback = @Sendable (_ iteration: Int, _ currentValue: Double) async -> Void

// MARK: - Convergence Criteria

/// Termination conditions for optimization algorithms
///
/// Defines when optimization should stop based on energy convergence, gradient norm,
/// or maximum iterations. Different optimizers use different criteria: gradient-free
/// methods (Nelder-Mead, COBYLA, SPSA) ignore ``gradientNormTolerance``, while
/// gradient-based methods (L-BFGS-B, Gradient Descent) check both energy and gradient.
///
/// **Example:**
/// ```swift
/// // Default criteria (energy tolerance only)
/// let criteria = ConvergenceCriteria()
///
/// // Gradient-based optimization with strict convergence
/// let strict = ConvergenceCriteria(
///     energyTolerance: 1e-8,
///     gradientNormTolerance: 1e-6,
///     maxIterations: 2000
/// )
/// ```
///
/// - SeeAlso: ``Optimizer`` for usage context
@frozen
public struct ConvergenceCriteria: Sendable {
    /// Energy change threshold for convergence
    ///
    /// Optimization terminates when |E_new - E_old| < energyTolerance.
    /// All optimizers check this criterion.
    ///
    /// Typical values: 1e-6 (default), 1e-8 (tight), 1e-4 (loose for noisy objectives)
    public let energyTolerance: Double

    /// Gradient norm threshold for convergence (gradient-based optimizers only)
    ///
    /// Optimization terminates when ||∇E|| < gradientNormTolerance.
    /// Ignored by gradient-free optimizers (Nelder-Mead, COBYLA, SPSA).
    ///
    /// Typical values: nil (disabled), 1e-5, 1e-6
    public let gradientNormTolerance: Double?

    /// Maximum optimization iterations
    ///
    /// Hard limit on iteration count regardless of convergence status.
    /// Prevents infinite loops when objective is difficult to minimize.
    ///
    /// Typical values: 1000 (default), 500 (fast), 2000 (thorough)
    public let maxIterations: Int

    /// Create convergence criteria with specified thresholds
    ///
    /// - Parameters:
    ///   - energyTolerance: Energy change threshold (default: 1e-6)
    ///   - gradientNormTolerance: Gradient norm threshold, nil to disable (default: nil)
    ///   - maxIterations: Maximum iterations (default: 1000)
    /// - Precondition: energyTolerance > 0, maxIterations > 0, gradientNormTolerance > 0 if provided
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
}

// MARK: - Optimizer Result

/// Optimization result with optimal parameters and convergence history
///
/// Contains optimal solution, objective value, full iteration history for analysis,
/// termination reason, and function evaluation count for performance tracking.
///
/// **Example:**
/// ```swift
/// let result = await optimizer.minimize(objective, from: initial, using: .init(), progress: nil)
/// print("Optimal energy: \(result.value)")
/// print("Converged: \(result.terminationReason == .energyConverged)")
/// print("Iterations: \(result.iterations), Evaluations: \(result.evaluations)")
///
/// // Analyze convergence trajectory
/// for (i, energy) in result.history.enumerated() {
///     print("Iteration \(i): E = \(energy)")
/// }
/// ```
///
/// - SeeAlso:
///   - ``TerminationReason`` for convergence status
///   - ``Optimizer`` for optimization methods
@frozen
public struct OptimizerResult: Sendable {
    /// Optimal parameters found by minimization
    ///
    /// These parameters minimize the objective function within convergence tolerances.
    /// Array length matches input parameter count.
    public let parameters: [Double]

    /// Objective function value at optimal parameters
    ///
    /// Final minimized value. For VQE, this is the ground state energy upper bound.
    public let value: Double

    /// Complete optimization history (objective values per iteration)
    ///
    /// Tracks objective value at each iteration for convergence analysis.
    /// Length equals ``iterations``. Use for plotting convergence trajectories.
    public let history: [Double]

    /// Total optimization iterations performed
    ///
    /// Number of optimization steps executed before termination.
    /// Does not count individual function evaluations (see ``evaluations``).
    public let iterations: Int

    /// Reason why optimization terminated
    ///
    /// Indicates whether optimization converged or hit iteration limit.
    public let terminationReason: TerminationReason

    /// Total objective function evaluations
    ///
    /// Counts all function calls during optimization. Typically exceeds ``iterations``
    /// for gradient-based methods (2n evaluations per iteration for parameter shift rule).
    public let evaluations: Int

    /// Create optimization result
    ///
    /// - Parameters:
    ///   - parameters: Optimal parameter values
    ///   - value: Objective value at optimum
    ///   - history: Iteration history
    ///   - iterations: Total iterations
    ///   - terminationReason: Why optimization stopped
    ///   - evaluations: Total function evaluations
    public init(
        parameters: [Double],
        value: Double,
        history: [Double],
        iterations: Int,
        terminationReason: TerminationReason,
        evaluations: Int,
    ) {
        self.parameters = parameters
        self.value = value
        self.history = history
        self.iterations = iterations
        self.terminationReason = terminationReason
        self.evaluations = evaluations
    }
}

/// Reason optimization terminated
///
/// Indicates whether optimization converged (energy or gradient criteria met)
/// or hit iteration limit without convergence.
///
/// - SeeAlso: ``ConvergenceCriteria`` for threshold configuration
@frozen
public enum TerminationReason: Sendable, CustomStringConvertible {
    /// Energy change converged below tolerance threshold
    case energyConverged

    /// Gradient norm converged below tolerance threshold (gradient-based only)
    case gradientConverged

    /// Maximum iterations reached without satisfying convergence criteria
    case maxIterationsReached

    @inlinable
    public var description: String {
        switch self {
        case .energyConverged: "Energy tolerance reached"
        case .gradientConverged: "Gradient norm below threshold"
        case .maxIterationsReached: "Maximum iterations reached"
        }
    }
}

// MARK: - Nelder-Mead Optimizer (Simplex Method)

/// Nelder-Mead simplex optimizer (derivative-free)
///
/// Gradient-free optimization using downhill simplex method with adaptive reflection,
/// expansion, contraction, and shrinkage operations. Best suited for low-dimensional
/// problems (< 20 parameters) with noisy objectives where gradients are unavailable or
/// expensive to compute. For smooth landscapes with many parameters, prefer
/// ``LBFGSBOptimizer`` which leverages curvature information. Scales poorly beyond
/// 50 parameters - consider ``SPSAOptimizer`` for high dimensions or ``COBYLAOptimizer``
/// for trust region robustness.
///
/// The algorithm maintains a simplex of n+1 points in n-dimensional parameter space.
/// Each iteration reflects the worst point through the centroid of the remaining points,
/// then adaptively expands, contracts, or shrinks based on the quality of the reflection.
/// Convergence occurs when the simplex collapses (all vertices reach similar objective values).
/// Adaptive simplex operations require 1-4 evaluations per iteration.
///
/// **Example:**
/// ```swift
/// let optimizer = NelderMeadOptimizer(tolerance: 1e-6)
/// let result = await optimizer.minimize(objective, from: initial, using: .init(), progress: nil)
/// print("Optimal energy: \(result.value)")
/// ```
///
/// - Complexity: O(iterations x evals_per_iteration), where evals_per_iteration ∈ [1,4]
/// - SeeAlso:
///   - ``COBYLAOptimizer`` for recommended derivative-free optimizer
///   - ``SPSAOptimizer`` for high-dimensional noisy objectives
///   - ``Optimizer`` for protocol definition
@frozen
public struct NelderMeadOptimizer: Optimizer {
    /// Convergence tolerance for simplex size
    ///
    /// Optimization terminates when range of simplex vertex values is below tolerance.
    /// Smaller values require tighter convergence (more iterations).
    ///
    /// Typical values: 1e-6 (default), 1e-8 (tight), 1e-4 (loose)
    public let tolerance: Double

    /// Initial simplex size relative to parameter values
    ///
    /// Controls spread of initial n+1 simplex points around starting parameters.
    /// Each vertex is perturbed by this amount along one coordinate axis.
    ///
    /// Typical values: 0.1 (default, 10% perturbation), 0.05 (small), 0.2 (large)
    public let initialSimplexSize: Double

    private let alpha: Double = 1.0
    private let gamma: Double = 2.0
    private let rho: Double = 0.5
    private let sigma: Double = 0.5

    /// Create Nelder-Mead optimizer with specified tolerance and simplex size
    ///
    /// - Parameters:
    ///   - tolerance: Convergence tolerance for simplex value range (default: 1e-6)
    ///   - initialSimplexSize: Initial simplex vertex perturbation size (default: 0.1)
    /// - Precondition: tolerance > 0, initialSimplexSize > 0
    public init(tolerance: Double = 1e-6, initialSimplexSize: Double = 0.1) {
        ValidationUtilities.validatePositiveDouble(tolerance, name: "tolerance")
        ValidationUtilities.validatePositiveDouble(initialSimplexSize, name: "initialSimplexSize")
        self.tolerance = tolerance
        self.initialSimplexSize = initialSimplexSize
    }

    /// Create Nelder-Mead optimizer with default simplex size
    ///
    /// Convenience initializer using default `initialSimplexSize = 0.1`.
    ///
    /// - Parameter tolerance: Convergence tolerance for simplex value range
    public init(tolerance: Double) {
        self.init(tolerance: tolerance, initialSimplexSize: 0.1)
    }

    /// Minimize objective function using Nelder-Mead simplex method
    ///
    /// Initializes n+1 simplex vertices, then iteratively transforms simplex via
    /// reflection, expansion, contraction, or shrinkage until convergence.
    ///
    /// - Precondition: initialParameters is non-empty
    /// - Complexity: O(maxIterations x evals_per_iteration) where evals_per_iteration ∈ [1,4]
    @_optimize(speed)
    @_eagerMove
    public func minimize(
        _ objectiveFunction: @Sendable ([Double]) async -> Double,
        from initialParameters: [Double],
        using convergenceCriteria: ConvergenceCriteria,
        progress: ProgressCallback?,
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

            if let callback = progress {
                await callback(iteration, currentBest)
            }

            valueHistory.append(currentBest)

            let range: Double = simplex[n].value - simplex[0].value
            if range < convergenceCriteria.energyTolerance {
                return OptimizerResult(
                    parameters: simplex[0].parameters,
                    value: simplex[0].value,
                    history: valueHistory,
                    iterations: iteration + 1,
                    terminationReason: .energyConverged,
                    evaluations: functionEvaluations,
                )
            }

            var centroid = [Double](unsafeUninitializedCapacity: n) { buffer, count in
                buffer.initialize(repeating: 0.0)
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
            parameters: simplex[0].parameters,
            value: simplex[0].value,
            history: valueHistory,
            iterations: convergenceCriteria.maxIterations,
            terminationReason: .maxIterationsReached,
            evaluations: functionEvaluations,
        )
    }

    struct SimplexVertex {
        var parameters: [Double]
        var value: Double
    }
}

// MARK: - Gradient Descent Optimizer

/// Gradient descent optimizer with parameter shift rule
///
/// First-order gradient-based optimization using quantum parameter shift rule for exact
/// gradient computation. Computes gradients via ∂E/∂θᵢ = [E(θᵢ+π/2) - E(θᵢ-π/2)] / 2,
/// then updates parameters with momentum v ← βv - η∇E. Adaptive learning rate multiplies
/// by 0.95 when objective value doesn't improve. Simpler than ``LBFGSBOptimizer`` but
/// less efficient for smooth landscapes - prefer L-BFGS-B for most VQE problems unless
/// Hessian approximation fails. For noisy objectives, use gradient-free methods like
/// ``COBYLAOptimizer`` or ``SPSAOptimizer``. Requires 2n evaluations per iteration for
/// parameter shift gradient computation.
///
/// Best suited for educational purposes (simplest gradient-based method), shallow circuits
/// where gradient computation is inexpensive, or debugging gradient-based optimization approaches.
///
/// **Example:**
/// ```swift
/// let optimizer = GradientDescentOptimizer(learningRate: 0.1, momentum: 0.9)
/// let result = await optimizer.minimize(objective, from: initial, using: criteria, progress: nil)
/// print("Converged: \(result.terminationReason)")
/// ```
///
/// - Complexity: O(maxIterations x (2n + 1)) where n = parameter count
/// - SeeAlso:
///   - ``LBFGSBOptimizer`` for superior gradient-based optimization
///   - ``COBYLAOptimizer`` for gradient-free alternative
///   - ``Optimizer`` for protocol definition
@frozen
public struct GradientDescentOptimizer: Optimizer {
    /// Initial learning rate (step size)
    ///
    /// Controls magnitude of parameter updates. Too large causes instability,
    /// too small causes slow convergence. Adaptive mode automatically reduces
    /// when optimization stagnates.
    ///
    /// Typical values: 0.1 (default), 0.01 (conservative), 0.5 (aggressive)
    public let learningRate: Double

    /// Momentum coefficient for velocity accumulation
    ///
    /// Accumulates gradient history to smooth optimization trajectory and accelerate
    /// convergence. 0 = no momentum (standard gradient descent), 0.9 = strong momentum.
    ///
    /// Typical values: 0.0 (default, no momentum), 0.9 (standard momentum)
    public let momentum: Double

    /// Enable adaptive learning rate reduction
    ///
    /// Multiplies learning rate by 0.95 when objective value doesn't improve.
    /// Helps stabilize convergence near local minima.
    ///
    /// Recommended: true (default) for most applications
    public let adaptiveLearningRate: Bool

    /// Parameter shift for gradient computation
    ///
    /// Shift value for parameter shift rule: ∂E/∂θ = [E(θ+s) - E(θ-s)] / 2.
    /// Standard quantum parameter shift uses π/2 for gates with eigenvalues ±1.
    ///
    /// Typical values: π/2 (default for standard gates), π/4 (for some advanced gates)
    public let parameterShift: Double

    /// Create gradient descent optimizer with full configuration
    ///
    /// - Parameters:
    ///   - learningRate: Step size for parameter updates (default: 0.1)
    ///   - momentum: Velocity accumulation coefficient (default: 0.0)
    ///   - adaptiveLearningRate: Enable learning rate decay on stagnation (default: true)
    ///   - parameterShift: Shift for parameter shift rule (default: π/2)
    /// - Precondition: learningRate > 0, momentum ∈ [0, 1), parameterShift > 0
    public init(
        learningRate: Double = 0.1,
        momentum: Double = 0.0,
        adaptiveLearningRate: Bool = true,
        parameterShift: Double = .pi / 2,
    ) {
        ValidationUtilities.validatePositiveDouble(learningRate, name: "learningRate")
        ValidationUtilities.validateHalfOpenRange(momentum, min: 0, max: 1, name: "momentum")
        ValidationUtilities.validatePositiveDouble(parameterShift, name: "parameterShift")

        self.learningRate = learningRate
        self.momentum = momentum
        self.adaptiveLearningRate = adaptiveLearningRate
        self.parameterShift = parameterShift
    }

    /// Create gradient descent optimizer with default momentum and adaptive learning rate
    ///
    /// Convenience initializer for simple usage with just learning rate configuration.
    ///
    /// - Parameter learningRate: Step size for parameter updates
    public init(learningRate: Double) {
        self.init(learningRate: learningRate, momentum: 0.0, adaptiveLearningRate: true)
    }

    /// Minimize objective function using gradient descent with parameter shift rule
    ///
    /// Computes gradients via quantum parameter shift, then updates parameters with
    /// momentum and adaptive learning rate until convergence.
    ///
    /// - Precondition: initialParameters is non-empty
    /// - Complexity: O(maxIterations x (2n + 1)) evaluations, where n = parameter count
    @_optimize(speed)
    @_eagerMove
    public func minimize(
        _ objectiveFunction: @Sendable ([Double]) async -> Double,
        from initialParameters: [Double],
        using convergenceCriteria: ConvergenceCriteria,
        progress: ProgressCallback?,
    ) async -> OptimizerResult {
        let n: Int = initialParameters.count
        ValidationUtilities.validateNonEmpty(initialParameters, name: "initialParameters")

        var currentParameters = initialParameters
        var currentValue: Double = await objectiveFunction(currentParameters)
        var valueHistory: [Double] = [currentValue]
        var functionEvaluations = 1

        var velocity = [Double](unsafeUninitializedCapacity: n) { buffer, count in
            buffer.initialize(repeating: 0.0)
            count = n
        }
        var currentLearningRate: Double = learningRate
        var bestValue: Double = currentValue
        var iterationsSinceImprovement = 0

        for iteration in 0 ..< convergenceCriteria.maxIterations {
            if let callback = progress {
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
                    parameters: currentParameters,
                    value: currentValue,
                    history: valueHistory,
                    iterations: iteration + 1,
                    terminationReason: .gradientConverged,
                    evaluations: functionEvaluations,
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
                    parameters: currentParameters,
                    value: newValue,
                    history: valueHistory,
                    iterations: iteration + 1,
                    terminationReason: .energyConverged,
                    evaluations: functionEvaluations,
                )
            }

            if newValue < bestValue - convergenceCriteria.energyTolerance {
                bestValue = newValue
                iterationsSinceImprovement = 0
            } else {
                iterationsSinceImprovement += 1
                if adaptiveLearningRate {
                    currentLearningRate *= 0.95
                }
            }

            currentValue = newValue
            valueHistory.append(newValue)
        }

        return OptimizerResult(
            parameters: currentParameters,
            value: currentValue,
            history: valueHistory,
            iterations: convergenceCriteria.maxIterations,
            terminationReason: .maxIterationsReached,
            evaluations: functionEvaluations,
        )
    }
}

// MARK: - L-BFGS-B Optimizer

/// L-BFGS-B optimizer (Limited-memory BFGS with box constraints)
///
/// Quasi-Newton method using limited-memory BFGS approximation of Hessian inverse.
/// Best gradient-based optimizer for smooth VQE landscapes with moderate to many parameters.
/// Approximates Hessian inverse H⁻¹ using last m gradient/parameter differences via two-loop
/// recursion, computes search direction p = -H⁻¹∇E, performs Wolfe backtracking line search
/// for step size α, then updates parameters θ ← θ + αp. Uses curvature information for faster
/// convergence than first-order methods.
///
/// Preferred over ``GradientDescentOptimizer`` for most VQE applications with smooth energy
/// landscapes (hardware-efficient ansatz, shallow depth) and moderate to many parameters
/// (> 20 where gradient descent struggles). For noisy objectives, use gradient-free methods
/// like ``COBYLAOptimizer`` or ``SPSAOptimizer``. For very few parameters (< 10),
/// ``NelderMeadOptimizer`` or ``COBYLAOptimizer`` often converge faster. Hessian approximation
/// may fail on ill-conditioned problems. Requires 2n evaluations per iteration for gradient
/// computation plus 1-5 for line search.
///
/// **Example:**
/// ```swift
/// let optimizer = LBFGSBOptimizer(memorySize: 10, tolerance: 1e-6)
/// let result = await optimizer.minimize(objective, from: initial, using: criteria, progress: nil)
/// print("Iterations: \(result.iterations), Evaluations: \(result.evaluations)")
/// ```
///
/// - Complexity: O(maxIterations x (2n + lineSearch)) evaluations, O(m·n) memory
/// - SeeAlso:
///   - ``GradientDescentOptimizer`` for simpler first-order method
///   - ``COBYLAOptimizer`` for gradient-free alternative
///   - ``Optimizer`` for protocol definition
@frozen
public struct LBFGSBOptimizer: Optimizer {
    /// Number of previous (s,y) gradient pairs to store
    ///
    /// Controls Hessian approximation quality vs memory usage. Larger m improves
    /// curvature approximation but increases memory by O(m·n). Typical value: 10.
    ///
    /// Trade-off: m=5 (low memory), m=10 (default), m=20 (high accuracy)
    public let memorySize: Int

    /// Gradient norm tolerance for convergence
    ///
    /// Optimization terminates when ||∇E|| < tolerance. Stricter than energy tolerance
    /// since small gradient indicates stationary point.
    ///
    /// Typical values: 1e-6 (default), 1e-8 (tight), 1e-4 (loose)
    public let tolerance: Double

    /// Maximum backtracking steps in Wolfe line search
    ///
    /// Limits line search iterations to prevent infinite loops. Line search finds
    /// step size satisfying Wolfe conditions (sufficient decrease + curvature).
    ///
    /// Typical values: 20 (default), 10 (fast), 50 (thorough)
    public let maxLineSearchSteps: Int

    /// Parameter shift for gradient computation
    ///
    /// Shift for quantum parameter shift rule: ∂E/∂θ = [E(θ+s) - E(θ-s)] / 2.
    /// Uses π/2 for standard quantum gates with ±1 eigenvalues.
    ///
    /// Typical values: π/2 (default for most gates)
    public let parameterShift: Double

    private let c1: Double = 1e-4
    private let c2: Double = 0.9

    /// Create L-BFGS-B optimizer with full configuration
    ///
    /// - Parameters:
    ///   - memorySize: Number of (s,y) pairs to store for Hessian approximation (default: 10)
    ///   - tolerance: Gradient norm convergence threshold (default: 1e-6)
    ///   - maxLineSearchSteps: Maximum Wolfe line search backtracking steps (default: 20)
    ///   - parameterShift: Shift for quantum parameter shift rule (default: π/2)
    /// - Precondition: memorySize > 0, tolerance > 0, maxLineSearchSteps > 0, parameterShift > 0
    public init(
        memorySize: Int = 10,
        tolerance: Double = 1e-6,
        maxLineSearchSteps: Int = 20,
        parameterShift: Double = .pi / 2,
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

    /// Create L-BFGS-B optimizer with default memory size and line search
    ///
    /// Convenience initializer using `memorySize = 10` and `maxLineSearchSteps = 20`.
    ///
    /// - Parameter tolerance: Gradient norm convergence threshold
    public init(tolerance: Double) {
        self.init(memorySize: 10, tolerance: tolerance, maxLineSearchSteps: 20)
    }

    /// Minimize objective function using L-BFGS-B quasi-Newton method
    ///
    /// Approximates Hessian inverse using limited memory (last m iterations), computes
    /// search direction via two-loop recursion, performs Wolfe line search, then updates
    /// parameter and gradient history.
    ///
    /// - Precondition: initialParameters is non-empty
    /// - Complexity: O(maxIterations x (2n + lineSearchSteps)) evaluations, O(m·n) memory
    @_optimize(speed)
    @_eagerMove
    public func minimize(
        _ objectiveFunction: @Sendable ([Double]) async -> Double,
        from initialParameters: [Double],
        using convergenceCriteria: ConvergenceCriteria,
        progress: ProgressCallback?,
    ) async -> OptimizerResult {
        let n: Int = initialParameters.count
        ValidationUtilities.validateNonEmpty(initialParameters, name: "initialParameters")

        var params = initialParameters
        var cost: Double = await objectiveFunction(params)
        var gradient: [Double] = await computeGradient(
            params: params,
            objectiveFunction: objectiveFunction,
        )

        var valueHistory: [Double] = [cost]
        var functionEvaluations = 1 + 2 * n

        var sHistory: [[Double]] = []
        var yHistory: [[Double]] = []
        var rhoHistory: [Double] = []

        for iteration in 0 ..< convergenceCriteria.maxIterations {
            if let callback = progress {
                await callback(iteration, cost)
            }

            var gradNormSq = 0.0
            vDSP_svesqD(gradient, 1, &gradNormSq, vDSP_Length(gradient.count))
            let gradNorm: Double = sqrt(gradNormSq)

            if let gnt = convergenceCriteria.gradientNormTolerance,
               gradNorm < gnt
            {
                return OptimizerResult(
                    parameters: params,
                    value: cost,
                    history: valueHistory,
                    iterations: iteration + 1,
                    terminationReason: .gradientConverged,
                    evaluations: functionEvaluations,
                )
            }

            let direction: [Double] = LBFGSBOptimizer.computeSearchDirection(
                gradient: gradient,
                sHistory: sHistory,
                yHistory: yHistory,
                rhoHistory: rhoHistory,
            )

            let lineSearchResult: LineSearchResult? = await performLineSearch(
                params: params,
                direction: direction,
                gradient: gradient,
                cost: cost,
                objectiveFunction: objectiveFunction,
            )

            guard let lsr = lineSearchResult else {
                return OptimizerResult(
                    parameters: params,
                    value: cost,
                    history: valueHistory,
                    iterations: iteration + 1,
                    terminationReason: .maxIterationsReached,
                    evaluations: functionEvaluations,
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
                objectiveFunction: objectiveFunction,
            )
            functionEvaluations += 1 + 2 * n

            if abs(newCost - cost) < convergenceCriteria.energyTolerance {
                return OptimizerResult(
                    parameters: newParams,
                    value: newCost,
                    history: valueHistory,
                    iterations: iteration + 1,
                    terminationReason: .energyConverged,
                    evaluations: functionEvaluations,
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
            parameters: params,
            value: cost,
            history: valueHistory,
            iterations: convergenceCriteria.maxIterations,
            terminationReason: .maxIterationsReached,
            evaluations: functionEvaluations,
        )
    }

    // MARK: - Private Helpers

    @_optimize(speed)
    private func computeGradient(
        params: [Double],
        objectiveFunction: @Sendable ([Double]) async -> Double,
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

    struct LineSearchResult {
        let alpha: Double
        let evaluations: Int
    }

    @_optimize(speed)
    private func performLineSearch(
        params: [Double],
        direction: [Double],
        gradient: [Double],
        cost: Double,
        objectiveFunction: @Sendable ([Double]) async -> Double,
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
                    objectiveFunction: objectiveFunction,
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
    static func computeSearchDirection(
        gradient: [Double],
        sHistory: [[Double]],
        yHistory: [[Double]],
        rhoHistory: [Double],
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
/// Stochastic gradient-free optimization using random simultaneous perturbations.
/// Unique advantage: only 2 evaluations per iteration regardless of parameter count.
/// Generates random perturbation Δ ∈ {-1,+1}ⁿ, evaluates f(θ ± cₖΔ), approximates
/// gradient ĝₖ = [f(θ+cₖΔ) - f(θ-cₖΔ)] / (2cₖ) · Δ, then updates θ ← θ - aₖĝₖ.
/// Uses Spall's theoretically optimal decay schedules: aₖ = a/(A+k)^α with α=0.602
/// for step size and cₖ = c/k^γ with γ=0.101 for perturbation magnitude.
///
/// Best for noisy objectives (hardware measurement noise, shot noise in VQE), very high
/// dimensions (n > 100 where 2n gradient evaluations impractical), or limited evaluation
/// budget. For smooth landscapes, use ``LBFGSBOptimizer`` which converges faster. For
/// low-moderate dimensions (< 50), gradient methods or ``COBYLAOptimizer`` are more
/// efficient. SPSA requires many iterations due to stochastic noise - expect slower
/// convergence than deterministic methods.
///
/// **Example:**
/// ```swift
/// let optimizer = SPSAOptimizer(initialStepSize: 0.1, initialPerturbation: 0.01)
/// let result = await optimizer.minimize(noisyObjective, from: initial, using: criteria, progress: nil)
/// print("Final value: \(result.value)")
/// ```
///
/// - Complexity: O(maxIterations x 2) evaluations (constant in n!), O(n) memory
/// - SeeAlso:
///   - ``COBYLAOptimizer`` for deterministic gradient-free alternative
///   - ``NelderMeadOptimizer`` for simpler gradient-free method
///   - ``Optimizer`` for protocol definition
@frozen
public struct SPSAOptimizer: Optimizer {
    /// Initial step size for parameter updates (a₀ in Spall's notation)
    ///
    /// Controls magnitude of first iteration update. Decays over iterations as aₖ = a₀/(A+k)^α.
    /// Too large causes instability, too small slows convergence.
    ///
    /// Typical values: 0.1 (default), 0.01 (conservative), 0.5 (aggressive)
    public let initialStepSize: Double

    /// Initial perturbation magnitude for gradient approximation (c₀ in Spall's notation)
    ///
    /// Controls size of simultaneous perturbation. Decays as cₖ = c₀/k^γ. Smaller values
    /// give more accurate gradient approximation but amplify measurement noise.
    ///
    /// Typical values: 0.01 (default), 0.001 (fine-grained), 0.1 (coarse)
    public let initialPerturbation: Double

    /// Step size decay exponent (α in Spall's notation)
    ///
    /// Controls rate of step size decay. Spall proved α=0.602 is optimal for asymptotic
    /// convergence. Do not change unless you have specific theoretical justification.
    ///
    /// Standard value: 0.602 (default, theoretically optimal)
    public let decayExponent: Double

    /// Perturbation decay exponent (γ in Spall's notation)
    ///
    /// Controls rate of perturbation decay. Spall proved γ=0.101 is optimal paired with α=0.602.
    /// Slower decay than step size to maintain gradient approximation quality.
    ///
    /// Standard value: 0.101 (default, theoretically optimal)
    public let perturbationDecayExponent: Double

    /// Stability constant for step size decay denominator (A in Spall's notation)
    ///
    /// Prevents division by zero in first iterations: aₖ = a₀/(A+k)^α. Typical choice:
    /// A ≈ 10% of expected maximum iterations.
    ///
    /// Typical values: 100 (default for ~1000 iterations), 50 (for ~500 iterations)
    public let stabilityConstant: Double

    /// Create SPSA optimizer with full configuration
    ///
    /// - Parameters:
    ///   - initialStepSize: Initial parameter update step size (default: 0.1)
    ///   - initialPerturbation: Initial perturbation magnitude (default: 0.01)
    ///   - decayExponent: Step size decay exponent α (default: 0.602, Spall's optimal)
    ///   - perturbationDecayExponent: Perturbation decay exponent γ (default: 0.101, Spall's optimal)
    ///   - stabilityConstant: Denominator stability constant A (default: 100)
    /// - Precondition: initialStepSize > 0, initialPerturbation > 0, decayExponent ∈ (0, 1), perturbationDecayExponent ∈ (0, 1), stabilityConstant ≥ 0
    public init(
        initialStepSize: Double = 0.1,
        initialPerturbation: Double = 0.01,
        decayExponent: Double = 0.602,
        perturbationDecayExponent: Double = 0.101,
        stabilityConstant: Double = 100.0,
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

    /// Create SPSA optimizer with standard decay parameters
    ///
    /// Uses Spall's theoretically optimal decay exponents (α=0.602, γ=0.101) and
    /// standard stability constant (A=100).
    ///
    /// - Parameters:
    ///   - initialStepSize: Initial parameter update step size (default: 0.1)
    ///   - initialPerturbation: Initial perturbation magnitude (default: 0.01)
    public init(initialStepSize: Double = 0.1, initialPerturbation: Double = 0.01) {
        self.init(
            initialStepSize: initialStepSize,
            initialPerturbation: initialPerturbation,
            decayExponent: 0.602,
            perturbationDecayExponent: 0.101,
            stabilityConstant: 100.0,
        )
    }

    /// Minimize objective function using SPSA stochastic approximation
    ///
    /// Generates random perturbations, approximates gradient via two evaluations (independent
    /// of parameter count), updates parameters with decaying step size.
    ///
    /// - Precondition: initialParameters is non-empty
    /// - Complexity: O(maxIterations x 2) evaluations (constant in n!), O(n) memory
    @_optimize(speed)
    @_eagerMove
    public func minimize(
        _ objectiveFunction: @Sendable ([Double]) async -> Double,
        from initialParameters: [Double],
        using convergenceCriteria: ConvergenceCriteria,
        progress: ProgressCallback?,
    ) async -> OptimizerResult {
        let n: Int = initialParameters.count
        ValidationUtilities.validateNonEmpty(initialParameters, name: "initialParameters")

        var params = initialParameters
        var currentValue: Double = await objectiveFunction(params)
        var valueHistory: [Double] = [currentValue]
        var functionEvaluations = 1

        for iteration in 0 ..< convergenceCriteria.maxIterations {
            if let callback = progress {
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
                    buffer[i] = 1.0 - 2.0 * Double(bit) // branchless: 0->+1, 1->-1
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

// MARK: - COBYLA Optimizer

/// COBYLA optimizer: Derivative-free trust region method via linear interpolation
///
/// Powell's Constrained Optimization BY Linear Approximations (1994). Default optimizer
/// for VQE due to derivative-free operation (saves 2n evaluations per iteration), robustness
/// to measurement noise, and superlinear convergence on smooth landscapes.
///
/// Builds linear model m(x) = f(x₀) + g^T(x - x₀) from n+1 simplex points, solves trust
/// region subproblem min m(x) subject to ||x - x₀|| ≤ ρ using Cauchy point (exact for
/// linear models), evaluates trial point, computes agreement ratio r = [f(x₀) - f(x_new)] /
/// [m(x₀) - m(x_new)], then updates trust region radius (expand if r > 0.75, shrink if
/// r < 0.1, accept if r > 0.1). Converges when trust region radius falls below tolerance.
///
/// Quantum energy landscapes are smooth (differentiable parameterized circuits) but noisy
/// (shot noise from measurements). COBYLA exploits smoothness via linear models while remaining
/// robust to noise through derivative-free optimization. Trust region prevents large steps into
/// regions where linear approximation breaks down. Requires 1-2 function evaluations per iteration
/// (n+1 initial simplex construction), achieves superlinear convergence near optimum. Suitable
/// for 5-50 parameters typical in VQE applications.
///
/// **Example:**
/// ```swift
/// let optimizer = COBYLAOptimizer(initialTrustRadius: 0.5, minTrustRadius: 1e-6)
/// let result = await optimizer.minimize(objective, from: initial, using: .init(), progress: nil)
/// print("Ground state energy: \(result.value)")
/// ```
///
/// - SeeAlso:
///   - ``NelderMeadOptimizer`` for simpler derivative-free method
///   - ``LBFGSBOptimizer`` for gradient-based alternative
///   - ``Optimizer`` for protocol definition
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
    /// - Precondition: All radii > 0, minTrustRadius ≤ initialTrustRadius ≤ maxTrustRadius, shrinkFactor ∈ (0, 1), expandFactor ∈ (1, 10), acceptRatio < expandRatio, simplexScale > 0
    public init(
        initialTrustRadius: Double = 0.5,
        minTrustRadius: Double = 1e-6,
        maxTrustRadius: Double = 2.0,
        shrinkFactor: Double = 0.5,
        expandFactor: Double = 2.0,
        acceptRatio: Double = 0.1,
        expandRatio: Double = 0.75,
        simplexScale: Double = 0.5,
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
            max: maxTrustRadius,
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

    /// Minimize objective function using COBYLA trust region method
    ///
    /// Builds linear model from simplex interpolation, solves trust region subproblem
    /// via Cauchy point, updates trust region radius based on actual/predicted reduction.
    ///
    /// - Precondition: initialParameters is non-empty
    /// - Complexity: O(maxIterations x 1-2) evaluations per iteration, O(n²) for simplex rebuild
    @_optimize(speed)
    @_eagerMove
    public func minimize(
        _ objectiveFunction: @Sendable ([Double]) async -> Double,
        from initialParameters: [Double],
        using convergenceCriteria: ConvergenceCriteria,
        progress: ProgressCallback?,
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

        // Safe force unwrap: simplex contains n+1 elements (base + n perturbations) appended above
        var bestIndex: Int = simplex.indices.min(by: { simplex[$0].value < simplex[$1].value })!
        var bestValue: Double = simplex[bestIndex].value
        var bestParameters: [Double] = simplex[bestIndex].parameters

        for iteration in 0 ..< convergenceCriteria.maxIterations {
            if let callback = progress {
                await callback(iteration, bestValue)
            }

            if currentTrustRadius < convergenceCriteria.energyTolerance {
                return OptimizerResult(
                    parameters: bestParameters,
                    value: bestValue,
                    history: valueHistory,
                    iterations: iteration + 1,
                    terminationReason: .energyConverged,
                    evaluations: functionEvaluations,
                )
            }

            let model: LinearModel = buildLinearModel(simplex: simplex, baseIndex: bestIndex)
            let step: [Double] = solveTrustRegionSubproblem(
                gradient: model.gradient,
                trustRadius: currentTrustRadius,
            )

            var trialParameters = [Double](unsafeUninitializedCapacity: n) { _, count in
                count = n
            }
            vDSP_vaddD(bestParameters, 1, step, 1, &trialParameters, 1, vDSP_Length(n))

            let trialValue: Double = await objectiveFunction(trialParameters)
            functionEvaluations += 1

            let predictedReduction: Double = -evaluateLinearModel(
                gradient: model.gradient,
                step: step,
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
                        parameters: bestParameters,
                        value: bestValue,
                        history: valueHistory,
                        iterations: iteration + 1,
                        terminationReason: .energyConverged,
                        evaluations: functionEvaluations,
                    )
                }
            }
        }

        return OptimizerResult(
            parameters: bestParameters,
            value: bestValue,
            history: valueHistory,
            iterations: convergenceCriteria.maxIterations,
            terminationReason: .maxIterationsReached,
            evaluations: functionEvaluations,
        )
    }

    // MARK: - Linear Model Construction

    /// Build linear model from simplex interpolation points.
    ///
    /// Constructs linear approximation m(x) = f(x₀) + gᵀ(x - x₀) where the gradient g is computed
    /// via least squares from simplex points. The algorithm forms matrix Y with rows Y[i] = simplex[i].parameters - baseParameters
    /// and vector f with f[i] = simplex[i].value - baseValue, then solves the least squares problem g = (YᵀY)⁻¹Yᵀf.
    /// For a well-conditioned simplex, this yields an accurate gradient estimate.
    ///
    /// - Parameters:
    ///   - simplex: Array of n+1 interpolation points
    ///   - baseIndex: Index of base point (usually best point in simplex)
    /// - Returns: Linear model with base point and gradient
    @_optimize(speed)
    @_eagerMove
    private func buildLinearModel(
        simplex: [SimplexPoint],
        baseIndex: Int,
    ) -> LinearModel {
        let n: Int = simplex[0].parameters.count
        let basePoint: [Double] = simplex[baseIndex].parameters
        let baseValue: Double = simplex[baseIndex].value

        var gradient = [Double](unsafeUninitializedCapacity: n) { buffer, count in
            buffer.initialize(repeating: 0.0)
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
                buffer.initialize(repeating: 0.0)
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
            gradient: gradient,
        )
    }

    /// Solve trust region subproblem: minimize linear model within trust region
    ///
    /// Minimizes m(s) = g^T·s subject to ||s|| ≤ ρ, where g is the gradient and s is
    /// the step vector. For linear models, the exact solution is the Cauchy point:
    /// s = -ρ·g/||g|| when ||g|| > ρ, otherwise s = 0 (already at optimum). More complex
    /// methods like dogleg or CG-Steihaug produce identical results for linear models.
    ///
    /// - Parameters:
    ///   - gradient: Linear model gradient g
    ///   - trustRadius: Current trust region radius ρ
    /// - Returns: Step vector s from base point
    @_optimize(speed)
    @_eagerMove
    @inline(__always)
    private func solveTrustRegionSubproblem(
        gradient: [Double],
        trustRadius: Double,
    ) -> [Double] {
        let n: Int = gradient.count

        var gradNormSq = 0.0
        vDSP_svesqD(gradient, 1, &gradNormSq, vDSP_Length(n))

        guard gradNormSq > 1e-12 else {
            return [Double](unsafeUninitializedCapacity: n) { buffer, count in
                buffer.initialize(repeating: 0.0)
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
        step: [Double],
    ) -> Double {
        var value = 0.0
        vDSP_dotprD(gradient, 1, step, 1, &value, vDSP_Length(gradient.count))
        return value
    }

    // MARK: - Supporting Types

    /// Single point in simplex with parameters and objective value
    struct SimplexPoint {
        var parameters: [Double]
        var value: Double

        init(parameters: [Double], value: Double) {
            self.parameters = parameters
            self.value = value
        }
    }

    /// Linear interpolation model: m(x) = f₀ + g^T(x - x₀)
    struct LinearModel {
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
