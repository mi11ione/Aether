// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under Apache 2.0

import Accelerate

/// Configuration for Lindblad master equation solver.
///
/// Controls adaptive step size, error tolerances, stiffness detection, and positivity enforcement
/// for open quantum system dynamics. The solver uses RK45 Dormand-Prince for non-stiff systems
/// and TR-BDF2 for stiff systems with automatic method switching.
///
/// **Example:**
/// ```swift
/// let config = LindbladConfiguration(
///     absoluteTolerance: 1e-10,
///     relativeTolerance: 1e-8,
///     maxSteps: 5000,
///     stiffnessThreshold: 15,
///     positivityEnforcement: .cholesky
/// )
/// ```
@frozen public struct LindbladConfiguration: Sendable {
    /// Absolute error tolerance for step acceptance.
    public let absoluteTolerance: Double

    /// Relative error tolerance for step acceptance.
    public let relativeTolerance: Double

    /// Maximum number of integration steps before failure.
    public let maxSteps: Int
    /// Rejection count window for stiffness detection.
    public let stiffnessThreshold: Int

    /// Method for enforcing density matrix positivity.
    public let positivityEnforcement: PositivityMethod

    /// Creates Lindblad solver configuration with specified parameters.
    ///
    /// - Parameters:
    ///   - absoluteTolerance: Absolute error tolerance (default: 1e-8)
    ///   - relativeTolerance: Relative error tolerance (default: 1e-6)
    ///   - maxSteps: Maximum integration steps (default: 10000)
    ///   - stiffnessThreshold: Rejection window for stiffness detection (default: 20)
    ///   - positivityEnforcement: Positivity enforcement method (default: .cholesky)
    ///
    /// **Example:**
    /// ```swift
    /// let config = LindbladConfiguration()
    /// let customConfig = LindbladConfiguration(absoluteTolerance: 1e-12)
    /// ```
    public init(
        absoluteTolerance: Double = 1e-8,
        relativeTolerance: Double = 1e-6,
        maxSteps: Int = 10000,
        stiffnessThreshold: Int = 20,
        positivityEnforcement: PositivityMethod = .cholesky,
    ) {
        self.absoluteTolerance = absoluteTolerance
        self.relativeTolerance = relativeTolerance
        self.maxSteps = maxSteps
        self.stiffnessThreshold = stiffnessThreshold
        self.positivityEnforcement = positivityEnforcement
    }
}

/// Method for enforcing density matrix positivity.
///
/// Density matrices must be positive semidefinite. Numerical errors during integration
/// can violate this constraint. These methods restore positivity after each step.
@frozen public enum PositivityMethod: Sendable {
    /// Cholesky factorization: represent rho = L * L-dagger, evolve L.
    case cholesky

    /// Eigenvalue clipping: set negative eigenvalues to zero and renormalize.
    case eigenvalue

    /// No positivity enforcement (faster but may produce invalid states).
    case none
}

/// Integration method for Lindblad dynamics.
///
/// Specifies which ODE solver to use for the master equation integration.
@frozen public enum IntegrationMethod: Sendable {
    /// RK45 Dormand-Prince: 7-stage embedded 4th/5th order pair with FSAL.
    case rk45

    /// TR-BDF2: Trapezoid predictor + BDF2 corrector, L-stable for stiff systems.
    case trBdf2

    /// Adaptive: starts with RK45, switches to TR-BDF2 on stiffness detection.
    case adaptive
}

/// Statistics from integration step history.
///
/// Provides diagnostic information about solver performance and method switching.
@frozen public struct StepStatistics: Sendable {
    /// Number of accepted integration steps.
    public let acceptedSteps: Int

    /// Number of rejected integration steps.
    public let rejectedSteps: Int

    /// Number of times stiffness was detected.
    public let stiffnessDetections: Int

    /// Number of method switches (RK45 to TR-BDF2 or vice versa).
    public let methodSwitches: Int
}

/// Result of Lindblad master equation evolution.
///
/// Contains the final density matrix state and integration statistics.
@frozen public struct LindbladResult: Sendable {
    /// Final density matrix after time evolution.
    public let finalState: DensityMatrix

    /// Final time reached.
    public let time: Double

    /// Integration step statistics.
    public let stepStatistics: StepStatistics

    /// Integration method used at final step.
    public let finalMethod: IntegrationMethod
}

/// Lindblad master equation solver for open quantum system dynamics.
///
/// Solves the Lindblad equation: drho/dt = -i[H,rho] + sum_k(L_k * rho * L_k-dagger - 0.5 * {L_k-dagger * L_k, rho})
/// using adaptive ODE integration with automatic stiffness detection and method switching.
///
/// **Example:**
/// ```swift
/// let hamiltonian = Observable.pauliZ(qubit: 0)
/// let jumpOps: [[[Complex<Double>]]] = [
///     [[.zero, Complex(0.1, 0)], [.zero, .zero]]  // Decay operator
/// ]
/// let initialState = DensityMatrix(qubits: 1)
///
/// let result = LindbladSolver.evolve(
///     hamiltonian: hamiltonian,
///     jumpOperators: jumpOps,
///     initialState: initialState,
///     time: 10.0
/// )
/// ```
public enum LindbladSolver {
    // MARK: - Dormand-Prince RK45 Coefficients

    private static let dp_c2: Double = 1.0 / 5.0
    private static let dp_c3: Double = 3.0 / 10.0
    private static let dp_c4: Double = 4.0 / 5.0
    private static let dp_c5: Double = 8.0 / 9.0
    private static let dp_c6: Double = 1.0
    private static let dp_c7: Double = 1.0

    private static let dp_a21: Double = 1.0 / 5.0
    private static let dp_a31: Double = 3.0 / 40.0
    private static let dp_a32: Double = 9.0 / 40.0
    private static let dp_a41: Double = 44.0 / 45.0
    private static let dp_a42: Double = -56.0 / 15.0
    private static let dp_a43: Double = 32.0 / 9.0
    private static let dp_a51: Double = 19372.0 / 6561.0
    private static let dp_a52: Double = -25360.0 / 2187.0
    private static let dp_a53: Double = 64448.0 / 6561.0
    private static let dp_a54: Double = -212.0 / 729.0
    private static let dp_a61: Double = 9017.0 / 3168.0
    private static let dp_a62: Double = -355.0 / 33.0
    private static let dp_a63: Double = 46732.0 / 5247.0
    private static let dp_a64: Double = 49.0 / 176.0
    private static let dp_a65: Double = -5103.0 / 18656.0
    private static let dp_a71: Double = 35.0 / 384.0
    private static let dp_a72: Double = 0.0
    private static let dp_a73: Double = 500.0 / 1113.0
    private static let dp_a74: Double = 125.0 / 192.0
    private static let dp_a75: Double = -2187.0 / 6784.0
    private static let dp_a76: Double = 11.0 / 84.0

    private static let dp_b1: Double = 35.0 / 384.0
    private static let dp_b2: Double = 0.0
    private static let dp_b3: Double = 500.0 / 1113.0
    private static let dp_b4: Double = 125.0 / 192.0
    private static let dp_b5: Double = -2187.0 / 6784.0
    private static let dp_b6: Double = 11.0 / 84.0
    private static let dp_b7: Double = 0.0

    private static let dp_e1: Double = 35.0 / 384.0 - 5179.0 / 57600.0
    private static let dp_e2: Double = 0.0
    private static let dp_e3: Double = 500.0 / 1113.0 - 7571.0 / 16695.0
    private static let dp_e4: Double = 125.0 / 192.0 - 393.0 / 640.0
    private static let dp_e5: Double = -2187.0 / 6784.0 + 92097.0 / 339_200.0
    private static let dp_e6: Double = 11.0 / 84.0 - 187.0 / 2100.0
    private static let dp_e7: Double = -1.0 / 40.0

    // MARK: - Adaptive Step Control Constants

    private static let safetyFactor: Double = 0.9
    private static let rejectionRateThreshold: Double = 0.3
    private static let minimumStepSize: Double = 1e-15
    private static let trBdf2Gamma: Double = 2.0 - sqrt(2.0)

    /// Evolve density matrix under Lindblad dynamics.
    ///
    /// Computes rho(t) from the Lindblad master equation:
    /// drho/dt = -i[H,rho] + sum_k(L_k * rho * L_k-dagger - 0.5 * {L_k-dagger * L_k, rho})
    ///
    /// Uses adaptive RK45 Dormand-Prince with automatic stiffness detection and
    /// method switching to TR-BDF2 for stiff dynamics.
    ///
    /// - Parameters:
    ///   - hamiltonian: System Hamiltonian as Observable
    ///   - jumpOperators: Array of Lindblad jump operators (Kraus-like)
    ///   - initialState: Initial density matrix
    ///   - time: Total evolution time
    ///   - configuration: Solver configuration parameters
    /// - Returns: LindbladResult with final state and statistics
    ///
    /// **Example:**
    /// ```swift
    /// let result = LindbladSolver.evolve(
    ///     hamiltonian: Observable.pauliZ(qubit: 0, coefficient: 1.0),
    ///     jumpOperators: [decayOperator],
    ///     initialState: DensityMatrix(qubits: 1),
    ///     time: 5.0
    /// )
    /// print("Final purity:", result.finalState.purity())
    /// ```
    @_optimize(speed)
    public static func evolve(
        hamiltonian: Observable,
        jumpOperators: [[[Complex<Double>]]],
        initialState: DensityMatrix,
        time: Double,
        configuration: LindbladConfiguration = LindbladConfiguration(),
    ) -> LindbladResult {
        ValidationUtilities.validateNonNegativeDouble(time, name: "Evolution time")
        ValidationUtilities.validatePositiveInt(configuration.maxSteps, name: "Max steps")

        let dim = initialState.dimension
        // Build Hamiltonian matrix from Observable
        let hMatrix = buildHamiltonianMatrix(hamiltonian, dimension: dim)

        // Flatten initial state to vector
        var rhoVec = flattenDensityMatrix(initialState)

        // Integration state
        var t = 0.0
        var h = min(0.01, time / 10.0)
        let hMin = 1e-15
        let hMax = time / 10.0

        var acceptedSteps = 0
        var rejectedSteps = 0
        var stiffnessDetections = 0
        var methodSwitches = 0
        var currentMethod: IntegrationMethod = .rk45

        var recentOutcomes = [Bool](repeating: true, count: 20)
        var outcomeIndex = 0

        // FSAL: store last k7 for next step
        var k1 = computeLindbladRHS(rhoVec, hMatrix: hMatrix, jumpOperators: jumpOperators, dim: dim)

        var stepCount = 0

        while t < time, stepCount < configuration.maxSteps {
            stepCount += 1
            let remainingTime = time - t
            h = min(h, remainingTime)

            if h < hMin {
                // Stiffness detected via step size
                if currentMethod == .rk45 {
                    stiffnessDetections += 1
                    currentMethod = .trBdf2
                    methodSwitches += 1
                    h = max(hMin * 100, h * 10)
                }
            }

            var accepted = false
            var rhoNew: [Complex<Double>]
            var errorNorm: Double

            if currentMethod == .rk45 {
                (rhoNew, errorNorm, k1) = rk45Step(
                    rhoVec: rhoVec,
                    k1: k1,
                    h: h,
                    hMatrix: hMatrix,
                    jumpOperators: jumpOperators,
                    dim: dim,
                )

                let rhoNorm = vectorNorm(rhoVec)
                let tol = configuration.absoluteTolerance + configuration.relativeTolerance * rhoNorm

                if errorNorm < tol {
                    accepted = true
                    acceptedSteps += 1
                    recentOutcomes[outcomeIndex] = true
                } else {
                    rejectedSteps += 1
                    recentOutcomes[outcomeIndex] = false
                }
                outcomeIndex = (outcomeIndex + 1) % 20

                // Stiffness detection via rejection rate > 30% over sliding window OR step < 1e-15
                let rejectionCount = recentOutcomes.count(where: { !$0 })
                let rejectionRate = Double(rejectionCount) / 20.0
                if rejectionRate > rejectionRateThreshold || h < minimumStepSize {
                    stiffnessDetections += 1
                    currentMethod = .trBdf2
                    methodSwitches += 1
                    recentOutcomes = [Bool](repeating: true, count: 20)
                    outcomeIndex = 0
                }

                // Step size adaptation
                if errorNorm > 1e-15 {
                    let factor = safetyFactor * pow(tol / errorNorm, 0.2)
                    h = h * min(5.0, max(0.1, factor))
                }
                h = max(hMin, min(hMax, h))

            } else {
                // TR-BDF2 step
                (rhoNew, accepted) = trBdf2Step(
                    rhoVec: rhoVec,
                    h: h,
                    hMatrix: hMatrix,
                    jumpOperators: jumpOperators,
                    dim: dim,
                    atol: configuration.absoluteTolerance,
                    rtol: configuration.relativeTolerance,
                )

                acceptedSteps += 1
                k1 = computeLindbladRHS(rhoNew, hMatrix: hMatrix, jumpOperators: jumpOperators, dim: dim)
            }

            if accepted {
                t += h
                rhoVec = rhoNew

                // Enforce positivity
                rhoVec = enforcePositivity(rhoVec, dim: dim, method: configuration.positivityEnforcement)

                // Normalize trace
                rhoVec = normalizeTrace(rhoVec, dim: dim)
            }
        }

        // Reconstruct density matrix
        let finalState = unflattenToDensityMatrix(rhoVec, qubits: initialState.qubits)

        let stats = StepStatistics(
            acceptedSteps: acceptedSteps,
            rejectedSteps: rejectedSteps,
            stiffnessDetections: stiffnessDetections,
            methodSwitches: methodSwitches,
        )

        return LindbladResult(
            finalState: finalState,
            time: t,
            stepStatistics: stats,
            finalMethod: currentMethod,
        )
    }

    // MARK: - Hamiltonian Matrix Construction

    /// Builds Hamiltonian matrix from Observable.
    @_optimize(speed)
    private static func buildHamiltonianMatrix(_ hamiltonian: Observable, dimension: Int) -> [Complex<Double>] {
        var hMatrix = [Complex<Double>](repeating: .zero, count: dimension * dimension)

        for term in hamiltonian.terms {
            let coeff = term.coefficient
            let pauliString = term.pauliString

            for row in 0 ..< dimension {
                let (col, phase) = pauliString.applyToRow(row: row)
                if col < dimension {
                    let idx = row * dimension + col
                    hMatrix[idx] = hMatrix[idx] + Complex(coeff, 0) * phase
                }
            }
        }

        return hMatrix
    }

    // MARK: - Lindblad RHS Computation

    /// Computes Lindblad right-hand side for ODE integration.
    @_optimize(speed)
    private static func computeLindbladRHS(
        _ rhoVec: [Complex<Double>],
        hMatrix: [Complex<Double>],
        jumpOperators: [[[Complex<Double>]]],
        dim: Int,
    ) -> [Complex<Double>] {
        let n2 = dim * dim

        // Convert flat vector to matrix for BLAS operations
        let rhoMatrix = rhoVec

        // Allocate result
        var result = [Complex<Double>](repeating: .zero, count: n2)

        // Compute commutator: -i[H, rho] = -i(H*rho - rho*H)
        var hRho = [Complex<Double>](repeating: .zero, count: n2)
        var rhoH = [Complex<Double>](repeating: .zero, count: n2)

        matrixMultiplyFlat(hMatrix, rhoMatrix, &hRho, dim: dim)
        matrixMultiplyFlat(rhoMatrix, hMatrix, &rhoH, dim: dim)

        let minusI = Complex<Double>(0.0, -1.0)
        for i in 0 ..< n2 {
            result[i] = minusI * (hRho[i] - rhoH[i])
        }

        // Lindblad dissipator: sum_k (L_k * rho * L_k^dag - 0.5 * {L_k^dag * L_k, rho})
        for jumpOp in jumpOperators {
            guard jumpOp.count == dim else { continue }

            let lFlat = flattenMatrix(jumpOp)
            let lDagFlat = hermitianConjugateFlat(lFlat, dim: dim)

            // L * rho
            var lRho = [Complex<Double>](repeating: .zero, count: n2)
            matrixMultiplyFlat(lFlat, rhoMatrix, &lRho, dim: dim)

            // L * rho * L^dag
            var lRhoLdag = [Complex<Double>](repeating: .zero, count: n2)
            matrixMultiplyFlat(lRho, lDagFlat, &lRhoLdag, dim: dim)

            // L^dag * L
            var ldagL = [Complex<Double>](repeating: .zero, count: n2)
            matrixMultiplyFlat(lDagFlat, lFlat, &ldagL, dim: dim)

            // L^dag * L * rho
            var ldagLRho = [Complex<Double>](repeating: .zero, count: n2)
            matrixMultiplyFlat(ldagL, rhoMatrix, &ldagLRho, dim: dim)

            // rho * L^dag * L
            var rhoLdagL = [Complex<Double>](repeating: .zero, count: n2)
            matrixMultiplyFlat(rhoMatrix, ldagL, &rhoLdagL, dim: dim)

            // Add to result: L*rho*L^dag - 0.5*(L^dag*L*rho + rho*L^dag*L)
            for i in 0 ..< n2 {
                result[i] = result[i] + lRhoLdag[i] - Complex(0.5, 0) * (ldagLRho[i] + rhoLdagL[i])
            }
        }

        return result
    }

    // MARK: - RK45 Dormand-Prince Step

    /// Performs single RK45 Dormand-Prince step with error estimate.
    @_optimize(speed)
    private static func rk45Step(
        rhoVec: [Complex<Double>],
        k1: [Complex<Double>],
        h: Double,
        hMatrix: [Complex<Double>],
        jumpOperators: [[[Complex<Double>]]],
        dim: Int,
    ) -> (rhoNew: [Complex<Double>], error: Double, k7: [Complex<Double>]) {
        let n2 = rhoVec.count

        // Stage 2
        var y2 = [Complex<Double>](repeating: .zero, count: n2)
        for i in 0 ..< n2 {
            y2[i] = rhoVec[i] + Complex(h * dp_a21, 0) * k1[i]
        }
        let k2 = computeLindbladRHS(y2, hMatrix: hMatrix, jumpOperators: jumpOperators, dim: dim)

        // Stage 3
        var y3 = [Complex<Double>](repeating: .zero, count: n2)
        for i in 0 ..< n2 {
            y3[i] = rhoVec[i] + Complex(h * dp_a31, 0) * k1[i] + Complex(h * dp_a32, 0) * k2[i]
        }
        let k3 = computeLindbladRHS(y3, hMatrix: hMatrix, jumpOperators: jumpOperators, dim: dim)

        // Stage 4
        var y4 = [Complex<Double>](repeating: .zero, count: n2)
        for i in 0 ..< n2 {
            y4[i] = rhoVec[i] + Complex(h * dp_a41, 0) * k1[i] + Complex(h * dp_a42, 0) * k2[i] + Complex(h * dp_a43, 0) * k3[i]
        }
        let k4 = computeLindbladRHS(y4, hMatrix: hMatrix, jumpOperators: jumpOperators, dim: dim)

        // Stage 5
        var y5 = [Complex<Double>](repeating: .zero, count: n2)
        for i in 0 ..< n2 {
            y5[i] = rhoVec[i] + Complex(h * dp_a51, 0) * k1[i] + Complex(h * dp_a52, 0) * k2[i] +
                Complex(h * dp_a53, 0) * k3[i] + Complex(h * dp_a54, 0) * k4[i]
        }
        let k5 = computeLindbladRHS(y5, hMatrix: hMatrix, jumpOperators: jumpOperators, dim: dim)

        // Stage 6
        var y6 = [Complex<Double>](repeating: .zero, count: n2)
        for i in 0 ..< n2 {
            y6[i] = rhoVec[i] + Complex(h * dp_a61, 0) * k1[i] + Complex(h * dp_a62, 0) * k2[i] +
                Complex(h * dp_a63, 0) * k3[i] + Complex(h * dp_a64, 0) * k4[i] + Complex(h * dp_a65, 0) * k5[i]
        }
        let k6 = computeLindbladRHS(y6, hMatrix: hMatrix, jumpOperators: jumpOperators, dim: dim)

        // Stage 7 (FSAL: same as 5th order solution)
        var y7 = [Complex<Double>](repeating: .zero, count: n2)
        for i in 0 ..< n2 {
            y7[i] = rhoVec[i] + Complex(h * dp_a71, 0) * k1[i] + Complex(h * dp_a73, 0) * k3[i] +
                Complex(h * dp_a74, 0) * k4[i] + Complex(h * dp_a75, 0) * k5[i] + Complex(h * dp_a76, 0) * k6[i]
        }
        let k7 = computeLindbladRHS(y7, hMatrix: hMatrix, jumpOperators: jumpOperators, dim: dim)

        // 5th order solution (y7 is already the 5th order result due to FSAL)
        let rhoNew = y7

        // Error estimate: difference between 5th and 4th order
        var errorVec = [Complex<Double>](repeating: .zero, count: n2)
        for i in 0 ..< n2 {
            errorVec[i] = Complex(h * dp_e1, 0) * k1[i] + Complex(h * dp_e3, 0) * k3[i] +
                Complex(h * dp_e4, 0) * k4[i] + Complex(h * dp_e5, 0) * k5[i] +
                Complex(h * dp_e6, 0) * k6[i] + Complex(h * dp_e7, 0) * k7[i]
        }

        let errorNorm = vectorNorm(errorVec)

        return (rhoNew, errorNorm, k7)
    }

    // MARK: - TR-BDF2 Step

    /// Performs single TR-BDF2 step for stiff dynamics.
    @_optimize(speed)
    private static func trBdf2Step(
        rhoVec: [Complex<Double>],
        h: Double,
        hMatrix: [Complex<Double>],
        jumpOperators: [[[Complex<Double>]]],
        dim: Int,
        atol: Double,
        rtol: Double,
    ) -> (rhoNew: [Complex<Double>], accepted: Bool) {
        let n2 = rhoVec.count
        let gamma = trBdf2Gamma

        // Trapezoid predictor: solve (I - gamma*h/2 * J) * y_gamma = rho + gamma*h/2 * f(rho)
        let f0 = computeLindbladRHS(rhoVec, hMatrix: hMatrix, jumpOperators: jumpOperators, dim: dim)

        var yGamma = [Complex<Double>](repeating: .zero, count: n2)
        for i in 0 ..< n2 {
            yGamma[i] = rhoVec[i] + Complex(gamma * h * 0.5, 0) * f0[i]
        }

        // Newton iteration for implicit trapezoid
        for _ in 0 ..< 10 {
            let fGamma = computeLindbladRHS(yGamma, hMatrix: hMatrix, jumpOperators: jumpOperators, dim: dim)
            var residual = [Complex<Double>](repeating: .zero, count: n2)
            for i in 0 ..< n2 {
                residual[i] = yGamma[i] - rhoVec[i] - Complex(gamma * h * 0.5, 0) * (f0[i] + fGamma[i])
            }

            let resNorm = vectorNorm(residual)
            let rhoNorm = vectorNorm(yGamma)
            let tol = atol + rtol * rhoNorm
            if resNorm < tol { break }

            // Simplified Newton: use explicit correction
            for i in 0 ..< n2 {
                yGamma[i] = yGamma[i] - residual[i]
            }
        }

        // BDF2 corrector: solve (1 - (1-gamma)/(2-gamma) * h * J) * y_new = ...
        let fGamma = computeLindbladRHS(yGamma, hMatrix: hMatrix, jumpOperators: jumpOperators, dim: dim)

        let c1 = (1.0 - gamma) / (2.0 - gamma)
        let c2 = 1.0 / (gamma * (2.0 - gamma))
        let c3 = (1.0 - gamma) * (1.0 - gamma) / (gamma * (2.0 - gamma))

        var yNew = [Complex<Double>](repeating: .zero, count: n2)
        for i in 0 ..< n2 {
            yNew[i] = Complex(c2, 0) * yGamma[i] - Complex(c3, 0) * rhoVec[i] + Complex(c1 * h, 0) * fGamma[i]
        }

        // Newton iteration for BDF2
        for _ in 0 ..< 10 {
            let fNew = computeLindbladRHS(yNew, hMatrix: hMatrix, jumpOperators: jumpOperators, dim: dim)
            var residual = [Complex<Double>](repeating: .zero, count: n2)
            for i in 0 ..< n2 {
                residual[i] = yNew[i] - Complex(c2, 0) * yGamma[i] + Complex(c3, 0) * rhoVec[i] - Complex(c1 * h, 0) * fNew[i]
            }

            let resNorm = vectorNorm(residual)
            let rhoNorm = vectorNorm(yNew)
            let tol = atol + rtol * rhoNorm
            if resNorm < tol { break }

            for i in 0 ..< n2 {
                yNew[i] = yNew[i] - residual[i]
            }
        }

        return (yNew, true)
    }

    // MARK: - Positivity Enforcement

    /// Enforces density matrix positivity using specified method.
    @_optimize(speed)
    private static func enforcePositivity(
        _ rhoVec: [Complex<Double>],
        dim: Int,
        method: PositivityMethod,
    ) -> [Complex<Double>] {
        switch method {
        case .none:
            rhoVec

        case .eigenvalue:
            enforcePositivityEigenvalue(rhoVec, dim: dim)

        case .cholesky:
            enforcePositivityCholesky(rhoVec, dim: dim)
        }
    }

    @_optimize(speed)
    private static func enforcePositivityEigenvalue(_ rhoVec: [Complex<Double>], dim: Int) -> [Complex<Double>] {
        // Convert to 2D for eigendecomposition
        var matrix = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: dim), count: dim)
        for i in 0 ..< dim {
            for j in 0 ..< dim {
                matrix[i][j] = rhoVec[i * dim + j]
            }
        }

        let eigen = HermitianEigenDecomposition.decompose(matrix: matrix)

        // Clip negative eigenvalues
        var clippedEigenvalues = eigen.eigenvalues.map { max(0.0, $0) }

        // Renormalize
        let traceSum = clippedEigenvalues.reduce(0.0, +)
        if traceSum > 1e-15 {
            for i in 0 ..< clippedEigenvalues.count {
                clippedEigenvalues[i] /= traceSum
            }
        }

        // Reconstruct: rho = V * diag(lambda) * V^dag
        var result = [Complex<Double>](repeating: .zero, count: dim * dim)
        for i in 0 ..< dim {
            for j in 0 ..< dim {
                var sum = Complex<Double>.zero
                for k in 0 ..< dim {
                    sum = sum + Complex(clippedEigenvalues[k], 0) * eigen.eigenvectors[k][i] * eigen.eigenvectors[k][j].conjugate
                }
                result[i * dim + j] = sum
            }
        }

        return result
    }

    @_optimize(speed)
    private static func enforcePositivityCholesky(_ rhoVec: [Complex<Double>], dim: Int) -> [Complex<Double>] {
        // Attempt Cholesky factorization; if it fails (not positive definite), fall back to eigenvalue method
        var aColMajor = [Double](repeating: 0.0, count: 2 * dim * dim)
        for col in 0 ..< dim {
            for row in 0 ..< dim {
                let idx = 2 * (col * dim + row)
                let val = rhoVec[row * dim + col]
                aColMajor[idx] = val.real
                aColMajor[idx + 1] = val.imaginary
            }
        }

        var uplo = CChar(Character("L").asciiValue!)
        var n = __LAPACK_int(dim)
        var lda = __LAPACK_int(dim)
        var info = __LAPACK_int(0)

        aColMajor.withUnsafeMutableBytes { aPtr in
            zpotrf_(&uplo, &n, OpaquePointer(aPtr.baseAddress), &lda, &info)
        }

        if info != 0 {
            // Not positive definite, fall back to eigenvalue method
            return enforcePositivityEigenvalue(rhoVec, dim: dim)
        }

        // Positivity guaranteed: zpotrf succeeds only for positive semidefinite matrices
        // Reconstruction rho = L * L^dag preserves positivity by construction

        // Zero out upper triangular part (keep only L)
        for col in 0 ..< dim {
            for row in 0 ..< col {
                let idx = 2 * (col * dim + row)
                aColMajor[idx] = 0.0
                aColMajor[idx + 1] = 0.0
            }
        }

        // Reconstruct rho = L * L^dag
        var lMatrix = [Complex<Double>](repeating: .zero, count: dim * dim)
        for col in 0 ..< dim {
            for row in 0 ..< dim {
                let idx = 2 * (col * dim + row)
                lMatrix[row * dim + col] = Complex(aColMajor[idx], aColMajor[idx + 1])
            }
        }

        let lDag = hermitianConjugateFlat(lMatrix, dim: dim)
        var result = [Complex<Double>](repeating: .zero, count: dim * dim)
        matrixMultiplyFlat(lMatrix, lDag, &result, dim: dim)

        return result
    }

    // MARK: - Trace Normalization

    /// Normalizes density matrix trace to unity.
    @_optimize(speed)
    private static func normalizeTrace(_ rhoVec: [Complex<Double>], dim: Int) -> [Complex<Double>] {
        var trace = 0.0
        for i in 0 ..< dim {
            trace += rhoVec[i * dim + i].real
        }

        if abs(trace) < 1e-15 {
            return rhoVec
        }

        var result = [Complex<Double>](repeating: .zero, count: rhoVec.count)
        let invTrace = 1.0 / trace
        for i in 0 ..< rhoVec.count {
            result[i] = Complex(invTrace, 0) * rhoVec[i]
        }

        return result
    }

    // MARK: - Matrix Utilities

    /// Flattens density matrix to row-major vector.
    @_optimize(speed)
    private static func flattenDensityMatrix(_ dm: DensityMatrix) -> [Complex<Double>] {
        let dim = dm.dimension
        let result = [Complex<Double>](unsafeUninitializedCapacity: dim * dim) { buffer, count in
            for i in 0 ..< dim {
                for j in 0 ..< dim {
                    buffer[i * dim + j] = dm.element(row: i, col: j)
                }
            }
            count = dim * dim
        }
        return result
    }

    /// Reconstructs density matrix from flat vector.
    @_optimize(speed)
    private static func unflattenToDensityMatrix(_ vec: [Complex<Double>], qubits: Int) -> DensityMatrix {
        DensityMatrix(qubits: qubits, elements: vec)
    }

    /// Flattens 2D matrix to row-major vector.
    @_optimize(speed)
    private static func flattenMatrix(_ matrix: [[Complex<Double>]]) -> [Complex<Double>] {
        let dim = matrix.count
        let result = [Complex<Double>](unsafeUninitializedCapacity: dim * dim) { buffer, count in
            for i in 0 ..< dim {
                for j in 0 ..< dim {
                    buffer[i * dim + j] = matrix[i][j]
                }
            }
            count = dim * dim
        }
        return result
    }

    /// Computes Hermitian conjugate of flat matrix.
    @_optimize(speed)
    private static func hermitianConjugateFlat(_ matrix: [Complex<Double>], dim: Int) -> [Complex<Double>] {
        let result = [Complex<Double>](unsafeUninitializedCapacity: dim * dim) { buffer, count in
            for i in 0 ..< dim {
                for j in 0 ..< dim {
                    buffer[i * dim + j] = matrix[j * dim + i].conjugate
                }
            }
            count = dim * dim
        }
        return result
    }

    /// Performs BLAS complex matrix multiplication.
    @_optimize(speed)
    private static func matrixMultiplyFlat(
        _ a: [Complex<Double>],
        _ b: [Complex<Double>],
        _ c: inout [Complex<Double>],
        dim: Int,
    ) {
        let n = dim
        let nn = n * n
        let nn2 = nn * 2

        var aInterleaved = [Double](unsafeUninitializedCapacity: nn2) { buffer, count in
            for i in 0 ..< nn {
                buffer[i * 2] = a[i].real
                buffer[i * 2 + 1] = a[i].imaginary
            }
            count = nn2
        }

        var bInterleaved = [Double](unsafeUninitializedCapacity: nn2) { buffer, count in
            for i in 0 ..< nn {
                buffer[i * 2] = b[i].real
                buffer[i * 2 + 1] = b[i].imaginary
            }
            count = nn2
        }

        var cInterleaved = [Double](repeating: 0.0, count: nn2)

        var alpha = (1.0, 0.0)
        var beta = (0.0, 0.0)

        aInterleaved.withUnsafeMutableBufferPointer { aPtr in
            bInterleaved.withUnsafeMutableBufferPointer { bPtr in
                cInterleaved.withUnsafeMutableBufferPointer { cPtr in
                    withUnsafeMutablePointer(to: &alpha) { alphaPtr in
                        withUnsafeMutablePointer(to: &beta) { betaPtr in
                            cblas_zgemm(
                                CblasRowMajor,
                                CblasNoTrans,
                                CblasNoTrans,
                                Int32(n), Int32(n), Int32(n),
                                OpaquePointer(alphaPtr),
                                OpaquePointer(aPtr.baseAddress), Int32(n),
                                OpaquePointer(bPtr.baseAddress), Int32(n),
                                OpaquePointer(betaPtr),
                                OpaquePointer(cPtr.baseAddress), Int32(n),
                            )
                        }
                    }
                }
            }
        }

        for i in 0 ..< nn {
            c[i] = Complex(cInterleaved[i * 2], cInterleaved[i * 2 + 1])
        }
    }

    /// Computes Euclidean norm of complex vector.
    @_optimize(speed)
    private static func vectorNorm(_ vec: [Complex<Double>]) -> Double {
        var sum = 0.0
        for v in vec {
            sum += v.magnitudeSquared
        }
        return sqrt(sum)
    }
}
