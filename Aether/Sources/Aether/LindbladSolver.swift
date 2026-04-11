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
///
/// - SeeAlso: ``LindbladSolver``
/// - SeeAlso: ``PositivityMethod``
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
///
/// **Example:**
/// ```swift
/// let method = PositivityMethod.cholesky
/// let config = LindbladConfiguration(positivityEnforcement: method)
/// ```
///
/// - SeeAlso: ``LindbladConfiguration``
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
///
/// **Example:**
/// ```swift
/// let result = LindbladSolver.evolve(
///     hamiltonian: Observable.pauliZ(qubit: 0),
///     jumpOperators: [],
///     initialState: DensityMatrix(qubits: 1),
///     time: 1.0
/// )
/// let method: IntegrationMethod = result.finalMethod
/// ```
///
/// - SeeAlso: ``LindbladResult``
/// - SeeAlso: ``LindbladSolver``
@frozen public enum IntegrationMethod: Sendable {
    /// RK45 Dormand-Prince: 7-stage embedded 4th/5th order pair with FSAL.
    case rk45

    /// TR-BDF2: Trapezoid predictor + BDF2 corrector, L-stable for stiff systems.
    case trBdf2
}

/// Statistics from integration step history.
///
/// Provides diagnostic information about solver performance and method switching.
///
/// **Example:**
/// ```swift
/// let result = LindbladSolver.evolve(
///     hamiltonian: Observable.pauliZ(qubit: 0),
///     jumpOperators: [],
///     initialState: DensityMatrix(qubits: 1),
///     time: 1.0
/// )
/// let stats = result.stepStatistics
/// let total = stats.acceptedSteps + stats.rejectedSteps
/// ```
///
/// - SeeAlso: ``LindbladResult``
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
///
/// **Example:**
/// ```swift
/// let result = LindbladSolver.evolve(
///     hamiltonian: Observable.pauliZ(qubit: 0),
///     jumpOperators: [],
///     initialState: DensityMatrix(qubits: 1),
///     time: 1.0
/// )
/// let finalRho = result.finalState
/// let elapsed = result.time
/// ```
///
/// - SeeAlso: ``LindbladSolver``
/// - SeeAlso: ``StepStatistics``
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
///
/// - SeeAlso: ``LindbladConfiguration``
/// - SeeAlso: ``LindbladResult``
public enum LindbladSolver {
    // MARK: - Private Types

    /// Precomputed flat jump operator matrices for efficient RHS evaluation.
    private struct JumpOperatorData {
        let flatOps: [[Complex<Double>]]
        let flatDags: [[Complex<Double>]]
        let ldagL: [[Complex<Double>]]
    }

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
    private static let defaultInitialStepSize: Double = 0.01
    private static let maxStepFraction: Double = 0.1
    private static let maxNewtonIterations: Int = 10
    private static let epsilon: Double = 1e-15
    private static let stepGrowthExponent: Double = 0.2
    private static let maxStepGrowthFactor: Double = 5.0
    private static let minStepShrinkFactor: Double = 0.1
    private static let stiffnessRecoveryMultiplier: Double = 100.0
    private static let stepBoostFactor: Double = 10.0

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
    /// - Returns: ``LindbladResult`` with final state and statistics
    /// - Precondition: time >= 0
    /// - Precondition: configuration.maxSteps > 0
    /// - Complexity: O(steps * dim^3) where dim = 2^qubits and steps depends on adaptive tolerance
    /// - SeeAlso: ``LindbladResult``
    /// - SeeAlso: ``LindbladConfiguration``
    ///
    /// **Example:**
    /// ```swift
    /// let decay: [[Complex<Double>]] = [[.zero, Complex(0.1, 0)], [.zero, .zero]]
    /// let result = LindbladSolver.evolve(
    ///     hamiltonian: Observable.pauliZ(qubit: 0),
    ///     jumpOperators: [decay],
    ///     initialState: DensityMatrix(qubits: 1),
    ///     time: 5.0
    /// )
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
        let hMatrix = buildHamiltonianMatrix(hamiltonian, dimension: dim)
        var rhoVec = flattenDensityMatrix(initialState)

        let validJumpOps = jumpOperators.filter { $0.count == dim }
        let flatOps = validJumpOps.map { flattenMatrix($0) }
        let flatDags = flatOps.map { hermitianConjugateFlat($0, dim: dim) }
        var ldagLProducts = [[Complex<Double>]]()
        ldagLProducts.reserveCapacity(flatOps.count)
        for i in 0 ..< flatOps.count {
            var product = [Complex<Double>](unsafeUninitializedCapacity: dim * dim) {
                buffer, count in
                buffer.initialize(repeating: .zero)
                count = dim * dim
            }
            matrixMultiplyFlat(flatDags[i], flatOps[i], &product, dim: dim)
            ldagLProducts.append(product)
        }
        let jumpData = JumpOperatorData(flatOps: flatOps, flatDags: flatDags, ldagL: ldagLProducts)

        var t = 0.0
        var h = min(defaultInitialStepSize, time * maxStepFraction)
        let hMax = time * maxStepFraction

        var acceptedSteps = 0
        var rejectedSteps = 0
        var stiffnessDetections = 0
        var methodSwitches = 0
        var currentMethod: IntegrationMethod = .rk45

        let windowSize = configuration.stiffnessThreshold
        var recentOutcomes = [Bool](unsafeUninitializedCapacity: windowSize) {
            buffer, count in
            buffer.initialize(repeating: true)
            count = windowSize
        }
        var outcomeIndex = 0
        var runningRejectionCount = 0

        var k1 = computeLindbladRHS(rhoVec, hMatrix: hMatrix, jumpData: jumpData, dim: dim)
        var cachedRhoNorm = vectorNorm(rhoVec)

        var stepCount = 0

        while t < time, stepCount < configuration.maxSteps {
            stepCount += 1
            let remainingTime = time - t
            h = min(h, remainingTime)

            if h < minimumStepSize {
                if currentMethod == .rk45 {
                    stiffnessDetections += 1
                    currentMethod = .trBdf2
                    methodSwitches += 1
                    h = max(minimumStepSize * stiffnessRecoveryMultiplier, h * stepBoostFactor)
                }
            }

            var accepted = false
            var rhoNew: [Complex<Double>]

            if currentMethod == .rk45 {
                let (stepRho, errorNorm, _) = rk45Step(
                    rhoVec: rhoVec,
                    k1: k1,
                    h: h,
                    hMatrix: hMatrix,
                    jumpData: jumpData,
                    dim: dim,
                )
                rhoNew = stepRho

                let tol = configuration.absoluteTolerance + configuration.relativeTolerance * cachedRhoNorm

                let previousWasRejection = !recentOutcomes[outcomeIndex]
                if previousWasRejection { runningRejectionCount -= 1 }

                if errorNorm < tol {
                    accepted = true
                    acceptedSteps += 1
                    recentOutcomes[outcomeIndex] = true
                } else {
                    rejectedSteps += 1
                    recentOutcomes[outcomeIndex] = false
                    runningRejectionCount += 1
                }
                outcomeIndex = (outcomeIndex + 1) % windowSize

                let rejectionRate = Double(runningRejectionCount) / Double(windowSize)
                if rejectionRate > rejectionRateThreshold || h < minimumStepSize {
                    stiffnessDetections += 1
                    currentMethod = .trBdf2
                    methodSwitches += 1
                    for i in 0 ..< windowSize {
                        recentOutcomes[i] = true
                    }
                    outcomeIndex = 0
                    runningRejectionCount = 0
                }

                if errorNorm > epsilon {
                    let factor = safetyFactor * pow(tol / errorNorm, stepGrowthExponent)
                    h = h * min(maxStepGrowthFactor, max(minStepShrinkFactor, factor))
                }
                h = max(minimumStepSize, min(hMax, h))

            } else {
                (rhoNew, accepted) = trBdf2Step(
                    rhoVec: rhoVec,
                    f0: k1,
                    h: h,
                    hMatrix: hMatrix,
                    jumpData: jumpData,
                    dim: dim,
                )

                if accepted {
                    acceptedSteps += 1
                } else {
                    rejectedSteps += 1
                    h = h * minStepShrinkFactor
                    h = max(minimumStepSize, h)
                }
            }

            if accepted {
                t += h
                rhoVec = rhoNew
                rhoVec = enforcePositivity(rhoVec, dim: dim, method: configuration.positivityEnforcement)
                rhoVec = normalizeTrace(rhoVec, dim: dim)
                cachedRhoNorm = vectorNorm(rhoVec)
                k1 = computeLindbladRHS(rhoVec, hMatrix: hMatrix, jumpData: jumpData, dim: dim)
            }
        }

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
    @_effects(readonly)
    @_optimize(speed)
    private static func buildHamiltonianMatrix(_ hamiltonian: Observable, dimension: Int) -> [Complex<Double>] {
        var hMatrix = [Complex<Double>](unsafeUninitializedCapacity: dimension * dimension) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = dimension * dimension
        }

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
    @_effects(readonly)
    @_optimize(speed)
    private static func computeLindbladRHS(
        _ rhoVec: [Complex<Double>],
        hMatrix: [Complex<Double>],
        jumpData: JumpOperatorData,
        dim: Int,
    ) -> [Complex<Double>] {
        let n2 = dim * dim

        var hRho = [Complex<Double>](unsafeUninitializedCapacity: n2) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = n2
        }
        var rhoH = [Complex<Double>](unsafeUninitializedCapacity: n2) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = n2
        }

        matrixMultiplyFlat(hMatrix, rhoVec, &hRho, dim: dim)
        matrixMultiplyFlat(rhoVec, hMatrix, &rhoH, dim: dim)

        var result = [Complex<Double>](unsafeUninitializedCapacity: n2) {
            buffer, count in
            for i in 0 ..< n2 {
                let diff = hRho[i] - rhoH[i]
                buffer[i] = Complex(diff.imaginary, -diff.real)
            }
            count = n2
        }
        var lRho = [Complex<Double>](unsafeUninitializedCapacity: n2) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = n2
        }
        var lRhoLdag = [Complex<Double>](unsafeUninitializedCapacity: n2) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = n2
        }
        var ldagLRho = [Complex<Double>](unsafeUninitializedCapacity: n2) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = n2
        }
        var rhoLdagL = [Complex<Double>](unsafeUninitializedCapacity: n2) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = n2
        }

        for opIndex in 0 ..< jumpData.flatOps.count {
            matrixMultiplyFlat(jumpData.flatOps[opIndex], rhoVec, &lRho, dim: dim)
            matrixMultiplyFlat(lRho, jumpData.flatDags[opIndex], &lRhoLdag, dim: dim)
            matrixMultiplyFlat(jumpData.ldagL[opIndex], rhoVec, &ldagLRho, dim: dim)
            matrixMultiplyFlat(rhoVec, jumpData.ldagL[opIndex], &rhoLdagL, dim: dim)

            for i in 0 ..< n2 {
                let anticomm = ldagLRho[i] + rhoLdagL[i]
                result[i] = result[i] + lRhoLdag[i] - Complex(anticomm.real * 0.5, anticomm.imaginary * 0.5)
            }
        }

        return result
    }

    // MARK: - RK45 Dormand-Prince Step

    /// Performs single RK45 Dormand-Prince step with error estimate.
    @_effects(readonly)
    @_optimize(speed)
    private static func rk45Step(
        rhoVec: [Complex<Double>],
        k1: [Complex<Double>],
        h: Double,
        hMatrix: [Complex<Double>],
        jumpData: JumpOperatorData,
        dim: Int,
    ) -> (rhoNew: [Complex<Double>], error: Double, k7: [Complex<Double>]) {
        let n2 = rhoVec.count

        let h_a21 = Complex<Double>(h * dp_a21, 0)
        let y2 = [Complex<Double>](unsafeUninitializedCapacity: n2) { buffer, count in
            for i in 0 ..< n2 {
                buffer[i] = rhoVec[i] + h_a21 * k1[i]
            }
            count = n2
        }
        let k2 = computeLindbladRHS(y2, hMatrix: hMatrix, jumpData: jumpData, dim: dim)

        let h_a31 = Complex<Double>(h * dp_a31, 0)
        let h_a32 = Complex<Double>(h * dp_a32, 0)
        let y3 = [Complex<Double>](unsafeUninitializedCapacity: n2) { buffer, count in
            for i in 0 ..< n2 {
                buffer[i] = rhoVec[i] + h_a31 * k1[i] + h_a32 * k2[i]
            }
            count = n2
        }
        let k3 = computeLindbladRHS(y3, hMatrix: hMatrix, jumpData: jumpData, dim: dim)

        let h_a41 = Complex<Double>(h * dp_a41, 0)
        let h_a42 = Complex<Double>(h * dp_a42, 0)
        let h_a43 = Complex<Double>(h * dp_a43, 0)
        let y4 = [Complex<Double>](unsafeUninitializedCapacity: n2) { buffer, count in
            for i in 0 ..< n2 {
                buffer[i] = rhoVec[i] + h_a41 * k1[i] + h_a42 * k2[i] + h_a43 * k3[i]
            }
            count = n2
        }
        let k4 = computeLindbladRHS(y4, hMatrix: hMatrix, jumpData: jumpData, dim: dim)

        let h_a51 = Complex<Double>(h * dp_a51, 0)
        let h_a52 = Complex<Double>(h * dp_a52, 0)
        let h_a53 = Complex<Double>(h * dp_a53, 0)
        let h_a54 = Complex<Double>(h * dp_a54, 0)
        let y5 = [Complex<Double>](unsafeUninitializedCapacity: n2) { buffer, count in
            for i in 0 ..< n2 {
                buffer[i] = rhoVec[i] + h_a51 * k1[i] + h_a52 * k2[i] +
                    h_a53 * k3[i] + h_a54 * k4[i]
            }
            count = n2
        }
        let k5 = computeLindbladRHS(y5, hMatrix: hMatrix, jumpData: jumpData, dim: dim)

        let h_a61 = Complex<Double>(h * dp_a61, 0)
        let h_a62 = Complex<Double>(h * dp_a62, 0)
        let h_a63 = Complex<Double>(h * dp_a63, 0)
        let h_a64 = Complex<Double>(h * dp_a64, 0)
        let h_a65 = Complex<Double>(h * dp_a65, 0)
        let y6 = [Complex<Double>](unsafeUninitializedCapacity: n2) { buffer, count in
            for i in 0 ..< n2 {
                buffer[i] = rhoVec[i] + h_a61 * k1[i] + h_a62 * k2[i] +
                    h_a63 * k3[i] + h_a64 * k4[i] + h_a65 * k5[i]
            }
            count = n2
        }
        let k6 = computeLindbladRHS(y6, hMatrix: hMatrix, jumpData: jumpData, dim: dim)

        let h_a71 = Complex<Double>(h * dp_a71, 0)
        let h_a73 = Complex<Double>(h * dp_a73, 0)
        let h_a74 = Complex<Double>(h * dp_a74, 0)
        let h_a75 = Complex<Double>(h * dp_a75, 0)
        let h_a76 = Complex<Double>(h * dp_a76, 0)
        let rhoNew = [Complex<Double>](unsafeUninitializedCapacity: n2) { buffer, count in
            for i in 0 ..< n2 {
                buffer[i] = rhoVec[i] + h_a71 * k1[i] + h_a73 * k3[i] +
                    h_a74 * k4[i] + h_a75 * k5[i] + h_a76 * k6[i]
            }
            count = n2
        }
        let k7 = computeLindbladRHS(rhoNew, hMatrix: hMatrix, jumpData: jumpData, dim: dim)

        let h_e1 = Complex<Double>(h * dp_e1, 0)
        let h_e3 = Complex<Double>(h * dp_e3, 0)
        let h_e4 = Complex<Double>(h * dp_e4, 0)
        let h_e5 = Complex<Double>(h * dp_e5, 0)
        let h_e6 = Complex<Double>(h * dp_e6, 0)
        let h_e7 = Complex<Double>(h * dp_e7, 0)
        let errorVec = [Complex<Double>](unsafeUninitializedCapacity: n2) { buffer, count in
            for i in 0 ..< n2 {
                buffer[i] = h_e1 * k1[i] + h_e3 * k3[i] +
                    h_e4 * k4[i] + h_e5 * k5[i] +
                    h_e6 * k6[i] + h_e7 * k7[i]
            }
            count = n2
        }

        let errorNorm = vectorNorm(errorVec)

        return (rhoNew, errorNorm, k7)
    }

    // MARK: - TR-BDF2 Step

    /// Performs single TR-BDF2 step for stiff dynamics via direct linear solve.
    @_optimize(speed)
    private static func trBdf2Step(
        rhoVec: [Complex<Double>],
        f0: [Complex<Double>],
        h: Double,
        hMatrix: [Complex<Double>],
        jumpData: JumpOperatorData,
        dim: Int,
    ) -> (rhoNew: [Complex<Double>], accepted: Bool) {
        let n2 = rhoVec.count
        let gamma = trBdf2Gamma

        let superop = constructSuperoperator(hMatrix: hMatrix, jumpData: jumpData, dim: dim)

        let alpha1 = gamma * h * 0.5
        let alpha1C = Complex<Double>(alpha1, 0)
        let rhs1 = [Complex<Double>](unsafeUninitializedCapacity: n2) {
            buffer, count in
            for i in 0 ..< n2 {
                buffer[i] = rhoVec[i] + alpha1C * f0[i]
            }
            count = n2
        }

        guard let yGamma = solveJacobianSystem(superop: superop, alpha: alpha1, rhs: rhs1, n2: n2) else {
            return (rhoVec, false)
        }

        let oneMinusGamma = 1.0 - gamma
        let twoMinusGamma = 2.0 - gamma
        let recipGammaTwoMinusGamma = 1.0 / (gamma * twoMinusGamma)
        let c1 = oneMinusGamma / twoMinusGamma
        let c2 = recipGammaTwoMinusGamma
        let c3 = oneMinusGamma * oneMinusGamma * recipGammaTwoMinusGamma

        let alpha2 = c1 * h
        let cc2 = Complex<Double>(c2, 0)
        let cc3 = Complex<Double>(c3, 0)

        let rhs2 = [Complex<Double>](unsafeUninitializedCapacity: n2) {
            buffer, count in
            for i in 0 ..< n2 {
                buffer[i] = cc2 * yGamma[i] - cc3 * rhoVec[i]
            }
            count = n2
        }

        guard let yNew = solveJacobianSystem(superop: superop, alpha: alpha2, rhs: rhs2, n2: n2) else {
            return (rhoVec, false)
        }

        return (yNew, true)
    }

    /// Construct Lindblad superoperator L in column-major layout by probing with basis vectors.
    @_optimize(speed)
    @_eagerMove
    private static func constructSuperoperator(
        hMatrix: [Complex<Double>],
        jumpData: JumpOperatorData,
        dim: Int,
    ) -> [Complex<Double>] {
        let n2 = dim * dim

        var basis = [Complex<Double>](unsafeUninitializedCapacity: n2) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = n2
        }

        return [Complex<Double>](unsafeUninitializedCapacity: n2 * n2) {
            buffer, count in
            for k in 0 ..< n2 {
                basis[k] = .one
                let col = computeLindbladRHS(basis, hMatrix: hMatrix, jumpData: jumpData, dim: dim)
                for i in 0 ..< n2 {
                    buffer[k * n2 + i] = col[i]
                }
                basis[k] = .zero
            }
            count = n2 * n2
        }
    }

    /// Solve (I - α·L)·x = rhs using LAPACK zgesv_ LU factorization.
    @_optimize(speed)
    @_eagerMove
    private static func solveJacobianSystem(
        superop: [Complex<Double>],
        alpha: Double,
        rhs: [Complex<Double>],
        n2: Int,
    ) -> [Complex<Double>]? {
        let alphaC = Complex<Double>(alpha, 0)

        var jacobian = [Double](unsafeUninitializedCapacity: 2 * n2 * n2) {
            buffer, count in
            for col in 0 ..< n2 {
                for row in 0 ..< n2 {
                    let superIdx = col * n2 + row
                    let val = (row == col ? Complex<Double>.one : .zero) - alphaC * superop[superIdx]
                    let idx = 2 * (col * n2 + row)
                    buffer[idx] = val.real
                    buffer[idx + 1] = val.imaginary
                }
            }
            count = 2 * n2 * n2
        }

        var b = [Double](unsafeUninitializedCapacity: 2 * n2) {
            buffer, count in
            for i in 0 ..< n2 {
                buffer[2 * i] = rhs[i].real
                buffer[2 * i + 1] = rhs[i].imaginary
            }
            count = 2 * n2
        }

        var nn = __LAPACK_int(n2)
        var nrhs = __LAPACK_int(1)
        var lda = __LAPACK_int(n2)
        var ldb = __LAPACK_int(n2)
        var pivots = [__LAPACK_int](unsafeUninitializedCapacity: n2) {
            _, count in count = n2
        }
        var info = __LAPACK_int(0)

        jacobian.withUnsafeMutableBytes { aPtr in
            b.withUnsafeMutableBytes { bPtr in
                pivots.withUnsafeMutableBufferPointer { pPtr in
                    zgesv_(
                        &nn, &nrhs,
                        OpaquePointer(aPtr.baseAddress),
                        &lda,
                        pPtr.baseAddress,
                        OpaquePointer(bPtr.baseAddress),
                        &ldb,
                        &info,
                    )
                }
            }
        }

        guard info == 0 else { return nil }

        return [Complex<Double>](unsafeUninitializedCapacity: n2) {
            buffer, count in
            for i in 0 ..< n2 {
                buffer[i] = Complex(b[2 * i], b[2 * i + 1])
            }
            count = n2
        }
    }

    // MARK: - Positivity Enforcement

    /// Enforces density matrix positivity using specified method.
    @_effects(readonly)
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

    /// Enforces positivity via eigenvalue clipping and renormalization.
    @_effects(readonly)
    @_optimize(speed)
    private static func enforcePositivityEigenvalue(_ rhoVec: [Complex<Double>], dim: Int) -> [Complex<Double>] {
        let matrix = (0 ..< dim).map { i in
            [Complex<Double>](unsafeUninitializedCapacity: dim) { buffer, count in
                for j in 0 ..< dim {
                    buffer.initializeElement(at: j, to: rhoVec[i * dim + j])
                }
                count = dim
            }
        }

        let eigen = HermitianEigenDecomposition.decompose(matrix: matrix)

        var clippedEigenvalues = eigen.eigenvalues.map { max(0.0, $0) }

        let traceSum = clippedEigenvalues.reduce(0.0, +)
        if traceSum > epsilon {
            for i in 0 ..< clippedEigenvalues.count {
                clippedEigenvalues[i] /= traceSum
            }
        }

        let sqrtEigenvalues = clippedEigenvalues.map { sqrt($0) }
        let wFlat = [Complex<Double>](unsafeUninitializedCapacity: dim * dim) { buffer, count in
            for i in 0 ..< dim {
                for k in 0 ..< dim {
                    buffer[i * dim + k] = Complex(sqrtEigenvalues[k], 0) * eigen.eigenvectors[k][i]
                }
            }
            count = dim * dim
        }

        let wDag = hermitianConjugateFlat(wFlat, dim: dim)
        var result = [Complex<Double>](unsafeUninitializedCapacity: dim * dim) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = dim * dim
        }
        matrixMultiplyFlat(wFlat, wDag, &result, dim: dim)

        return result
    }

    /// Enforces positivity via Cholesky factorization with eigenvalue fallback.
    @_effects(readonly)
    @_optimize(speed)
    private static func enforcePositivityCholesky(_ rhoVec: [Complex<Double>], dim: Int) -> [Complex<Double>] {
        var aColMajor = [Double](unsafeUninitializedCapacity: 2 * dim * dim) {
            buffer, count in
            for col in 0 ..< dim {
                for row in 0 ..< dim {
                    let idx = 2 * (col * dim + row)
                    let val = rhoVec[row * dim + col]
                    buffer[idx] = val.real
                    buffer[idx + 1] = val.imaginary
                }
            }
            count = 2 * dim * dim
        }

        // Safety: ASCII 'L' always has a value
        var uplo = CChar(Character("L").asciiValue!)
        var n = __LAPACK_int(dim)
        var lda = __LAPACK_int(dim)
        var info = __LAPACK_int(0)

        aColMajor.withUnsafeMutableBytes { aPtr in
            zpotrf_(&uplo, &n, OpaquePointer(aPtr.baseAddress), &lda, &info)
        }

        if info != 0 {
            return enforcePositivityEigenvalue(rhoVec, dim: dim)
        }

        for col in 0 ..< dim {
            for row in 0 ..< col {
                let idx = 2 * (col * dim + row)
                aColMajor[idx] = 0.0
                aColMajor[idx + 1] = 0.0
            }
        }

        let lMatrix = [Complex<Double>](unsafeUninitializedCapacity: dim * dim) { buffer, count in
            for col in 0 ..< dim {
                for row in 0 ..< dim {
                    let idx = 2 * (col * dim + row)
                    buffer[row * dim + col] = Complex(aColMajor[idx], aColMajor[idx + 1])
                }
            }
            count = dim * dim
        }

        let lDag = hermitianConjugateFlat(lMatrix, dim: dim)
        var result = [Complex<Double>](unsafeUninitializedCapacity: dim * dim) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = dim * dim
        }
        matrixMultiplyFlat(lMatrix, lDag, &result, dim: dim)

        return result
    }

    // MARK: - Trace Normalization

    /// Normalizes density matrix trace to unity.
    @_effects(readonly)
    @_optimize(speed)
    private static func normalizeTrace(_ rhoVec: [Complex<Double>], dim: Int) -> [Complex<Double>] {
        var trace = 0.0
        for i in 0 ..< dim {
            trace += rhoVec[i * dim + i].real
        }

        if abs(trace) < epsilon {
            return rhoVec
        }

        let invTraceC = Complex<Double>(1.0 / trace, 0)
        let result = [Complex<Double>](unsafeUninitializedCapacity: rhoVec.count) { buffer, count in
            for i in 0 ..< rhoVec.count {
                buffer[i] = invTraceC * rhoVec[i]
            }
            count = rhoVec.count
        }

        return result
    }

    // MARK: - Matrix Utilities

    /// Flattens density matrix to row-major vector.
    @_effects(readonly)
    @_optimize(speed)
    private static func flattenDensityMatrix(_ dm: DensityMatrix) -> [Complex<Double>] {
        let dim = dm.dimension
        let result = [Complex<Double>](unsafeUninitializedCapacity: dim * dim) { buffer, count in
            for i in 0 ..< dim {
                for j in 0 ..< dim {
                    buffer[i * dim + j] = dm[row: i, col: j]
                }
            }
            count = dim * dim
        }
        return result
    }

    /// Reconstructs density matrix from flat vector.
    @_effects(readonly)
    @_optimize(speed)
    private static func unflattenToDensityMatrix(_ vec: [Complex<Double>], qubits: Int) -> DensityMatrix {
        DensityMatrix(qubits: qubits, elements: vec)
    }

    /// Flattens 2D matrix to row-major vector.
    @_effects(readonly)
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
    @_effects(readonly)
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
        var alpha = (1.0, 0.0)
        var beta = (0.0, 0.0)
        let n = Int32(dim)

        a.withUnsafeBufferPointer { aPtr in
            b.withUnsafeBufferPointer { bPtr in
                c.withUnsafeMutableBufferPointer { cPtr in
                    withUnsafeMutablePointer(to: &alpha) { alphaPtr in
                        withUnsafeMutablePointer(to: &beta) { betaPtr in
                            cblas_zgemm(
                                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                n, n, n,
                                OpaquePointer(alphaPtr),
                                OpaquePointer(aPtr.baseAddress), n,
                                OpaquePointer(bPtr.baseAddress), n,
                                OpaquePointer(betaPtr),
                                OpaquePointer(cPtr.baseAddress), n,
                            )
                        }
                    }
                }
            }
        }
    }

    /// Computes Euclidean norm of complex vector.
    @inline(__always)
    @_effects(readonly)
    @_optimize(speed)
    private static func vectorNorm(_ vec: [Complex<Double>]) -> Double {
        var sum = 0.0
        for v in vec {
            sum += v.magnitudeSquared
        }
        return sqrt(sum)
    }
}
