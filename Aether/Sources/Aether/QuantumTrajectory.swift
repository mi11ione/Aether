// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under Apache 2.0

import Accelerate
import GameplayKit

/// Configuration for Monte Carlo wavefunction trajectory simulation.
///
/// Controls the number of trajectories, random seeding, time discretization, and
/// whether to retain individual trajectory states for post-processing analysis.
///
/// **Example:**
/// ```swift
/// let config = TrajectoryConfiguration(
///     trajectories: 500,
///     seed: 12345,
///     timeSteps: 2000,
///     storeIndividualTrajectories: true
/// )
/// ```
///
/// - SeeAlso: ``QuantumTrajectory``
/// - SeeAlso: ``TrajectoryResult``
@frozen
public struct TrajectoryConfiguration: Sendable {
    /// Number of independent trajectory realizations to simulate.
    ///
    /// More trajectories improve statistical accuracy of the averaged density matrix.
    /// Standard error scales as 1/sqrt(trajectories). Typical values: 100-10000.
    public let trajectories: Int

    /// Optional seed for reproducible random number generation.
    ///
    /// When nil, uses system entropy. Set to a fixed value for deterministic results
    /// across runs, useful for debugging and testing.
    public let seed: UInt64?

    /// Number of discrete time steps for evolution from t=0 to final time.
    ///
    /// Determines step size dt = time/timeSteps. Smaller dt improves accuracy but
    /// increases computation. Should satisfy dt << 1/||H|| for stability.
    public let timeSteps: Int

    /// Whether to store final states from all individual trajectories.
    ///
    /// When true, ``TrajectoryResult/individualTrajectories`` contains all M final
    /// states. Useful for computing higher-order statistics beyond the mean. When
    /// false, only the averaged density matrix is retained, reducing memory usage.
    public let storeIndividualTrajectories: Bool

    /// Creates a trajectory simulation configuration.
    ///
    /// - Parameters:
    ///   - trajectories: Number of Monte Carlo samples (default: 1000)
    ///   - seed: Random seed for reproducibility (default: nil for system random)
    ///   - timeSteps: Time discretization steps (default: 1000)
    ///   - storeIndividualTrajectories: Retain all trajectory states (default: false)
    ///
    /// **Example:**
    /// ```swift
    /// let config = TrajectoryConfiguration(trajectories: 2000, seed: 42)
    /// ```
    public init(
        trajectories: Int = 1000,
        seed: UInt64? = nil,
        timeSteps: Int = 1000,
        storeIndividualTrajectories: Bool = false,
    ) {
        self.trajectories = trajectories
        self.seed = seed
        self.timeSteps = timeSteps
        self.storeIndividualTrajectories = storeIndividualTrajectories
    }
}

/// Statistics collected during quantum trajectory simulation.
///
/// Tracks jump events across all trajectories for each Lindblad operator, providing
/// insight into decoherence rates and dominant dissipation channels.
///
/// **Example:**
/// ```swift
/// let stats = result.statistics
/// print("Total jumps for operator 0: \(stats.jumpCounts[0])")
/// print("Average jumps per trajectory: \(stats.averageJumpsPerTrajectory)")
/// ```
///
/// - SeeAlso: ``QuantumTrajectory``
/// - SeeAlso: ``TrajectoryResult``
@frozen
public struct TrajectoryStatistics: Sendable {
    /// Total jump counts per Lindblad operator index across all trajectories.
    ///
    /// jumpCounts[k] is the total number of times operator Lₖ caused a jump.
    public let jumpCounts: [Int]

    /// Average number of quantum jumps per trajectory.
    ///
    /// Higher values indicate stronger dissipation. For weak damping,
    /// expect averageJumpsPerTrajectory << timeSteps.
    public let averageJumpsPerTrajectory: Double

    /// Total number of trajectories simulated.
    public let trajectoryCount: Int
}

/// Result of quantum trajectory Monte Carlo simulation.
///
/// Contains the averaged density matrix, final evolution time, jump statistics,
/// and optionally all individual trajectory final states for post-processing.
///
/// **Example:**
/// ```swift
/// let result = await QuantumTrajectory.evolve(
///     hamiltonian: H,
///     jumpOperators: [L1, L2],
///     initialState: psi0,
///     time: 10.0
/// )
/// print("Final purity: \(result.averageDensityMatrix.purity())")
/// ```
///
/// - SeeAlso: ``QuantumTrajectory``
/// - SeeAlso: ``TrajectoryConfiguration``
@frozen
public struct TrajectoryResult: Sendable {
    /// Averaged density matrix ρ = (1/M) Σⱼ |ψⱼ⟩⟨ψⱼ| over all trajectories.
    ///
    /// Reproduces the Lindblad master equation solution in the limit M → ∞.
    public let averageDensityMatrix: DensityMatrix

    /// Final evolution time.
    public let time: Double

    /// Jump statistics collected during simulation.
    public let statistics: TrajectoryStatistics

    /// Individual trajectory final states (nil unless storeIndividualTrajectories was true).
    ///
    /// When populated, contains M quantum states for computing higher-order correlations
    /// or trajectory-resolved observables beyond the mean.
    public let individualTrajectories: [QuantumState]?
}

/// Monte Carlo wavefunction method for open quantum system dynamics.
///
/// Implements stochastic simulation via quantum trajectories: pure state evolution
/// with random jumps that, when averaged, recovers the Lindblad master equation
/// dynamics. Each trajectory evolves under an effective non-Hermitian Hamiltonian
/// with probabilistic collapses (quantum jumps) when the norm drops below a
/// random threshold.
///
/// The method solves: dρ/dt = -i[H,ρ] + Σₖ (LₖρLₖ† - ½{Lₖ†Lₖ, ρ})
///
/// Algorithm per trajectory:
/// 1. Construct H_eff = H - (i/2)Σₖ Lₖ†Lₖ
/// 2. Evolve |ψ(t+dt)⟩ = (I - iH_eff×dt)|ψ⟩
/// 3. Compute jump probability dp = 1 - ⟨ψ|ψ⟩
/// 4. If random < dp: apply jump Lₖ with probability ∝ ⟨ψ|Lₖ†Lₖ|ψ⟩
/// 5. Renormalize and repeat
///
/// **Example:**
/// ```swift
/// // Spontaneous emission: Lindblad operator L = σ₋ = |0⟩⟨1|
/// let sigma_minus: [[Complex<Double>]] = [
///     [.zero, .one],
///     [.zero, .zero]
/// ]
/// let H = Observable.pauliZ(qubit: 0, coefficient: 1.0)
/// let psi0 = QuantumState(qubits: 1, amplitudes: [.zero, .one])  // |1⟩
///
/// let result = await QuantumTrajectory.evolve(
///     hamiltonian: H,
///     jumpOperators: [sigma_minus],
///     initialState: psi0,
///     time: 5.0,
///     configuration: TrajectoryConfiguration(trajectories: 1000)
/// )
/// ```
///
/// - SeeAlso: ``TrajectoryConfiguration``
/// - SeeAlso: ``TrajectoryResult``
/// - SeeAlso: ``DensityMatrix``
@frozen
public enum QuantumTrajectory {
    /// Evolve quantum state under Lindblad dynamics using Monte Carlo wavefunction method.
    ///
    /// Simulates open quantum system evolution by averaging over stochastic pure state
    /// trajectories with quantum jumps. The result converges to the Lindblad master
    /// equation solution as the number of trajectories increases.
    ///
    /// - Parameters:
    ///   - hamiltonian: System Hamiltonian as Pauli string observable
    ///   - jumpOperators: Lindblad jump operators Lₖ as complex matrices
    ///   - initialState: Normalized initial pure state |ψ₀⟩
    ///   - time: Total evolution time (must be positive)
    ///   - configuration: Simulation parameters (trajectories, seed, time steps)
    /// - Returns: Trajectory result with averaged density matrix and statistics
    ///
    /// **Example:**
    /// ```swift
    /// let result = await QuantumTrajectory.evolve(
    ///     hamiltonian: Observable.pauliZ(qubit: 0),
    ///     jumpOperators: [sigmaMinus],
    ///     initialState: QuantumState(qubit: 1),
    ///     time: 10.0
    /// )
    /// ```
    ///
    /// - Complexity: O(trajectories × timeSteps × dim²) where dim = 2^n
    /// - Precondition: initialState must be normalized
    /// - Precondition: time > 0
    /// - Precondition: trajectories > 0
    /// - Precondition: timeSteps > 0
    /// - Precondition: Jump operator dimensions must match state dimension
    @_optimize(speed)
    public static func evolve(
        hamiltonian: Observable,
        jumpOperators: [[[Complex<Double>]]],
        initialState: QuantumState,
        time: Double,
        configuration: TrajectoryConfiguration = TrajectoryConfiguration(),
    ) async -> TrajectoryResult {
        ValidationUtilities.validateNormalizedState(initialState)
        ValidationUtilities.validatePositiveDouble(time, name: "Evolution time")
        ValidationUtilities.validatePositiveInt(configuration.trajectories, name: "Trajectory count")
        ValidationUtilities.validatePositiveInt(configuration.timeSteps, name: "Time steps")

        let dim = initialState.stateSpaceSize
        for (idx, L) in jumpOperators.enumerated() {
            ValidationUtilities.validateSquareMatrix(L, name: "Jump operator \(idx)")
            ValidationUtilities.validateMatrixDimensionEquals(L, expected: dim, name: "Jump operator \(idx)")
        }

        let dt = time / Double(configuration.timeSteps)
        let hMatrix = buildHamiltonianMatrix(hamiltonian: hamiltonian, dimension: dim)
        let hEff = buildEffectiveHamiltonian(hMatrix: hMatrix, jumpOperators: jumpOperators, dimension: dim)
        let lDaggerL = buildLDaggerL(jumpOperators: jumpOperators, dimension: dim)

        let numOperators = jumpOperators.count
        let numTrajectories = configuration.trajectories
        let timeSteps = configuration.timeSteps

        let trajectoryResults = await withTaskGroup(
            of: (state: QuantumState, jumpCounts: [Int]).self,
            returning: [(state: QuantumState, jumpCounts: [Int])].self,
        ) { group in
            for trajIdx in 0 ..< numTrajectories {
                group.addTask {
                    let seed: UInt64? = configuration.seed.map { $0 &+ UInt64(trajIdx) }
                    return runSingleTrajectory(
                        initialState: initialState,
                        hEff: hEff,
                        jumpOperators: jumpOperators,
                        lDaggerL: lDaggerL,
                        dt: dt,
                        timeSteps: timeSteps,
                        numOperators: numOperators,
                        dimension: dim,
                        seed: seed,
                    )
                }
            }

            var results: [(state: QuantumState, jumpCounts: [Int])] = []
            results.reserveCapacity(numTrajectories)
            for await result in group {
                results.append(result)
            }
            return results
        }

        var totalJumpCounts = [Int](repeating: 0, count: numOperators)
        var totalJumps = 0
        var individualStates: [QuantumState]? = configuration.storeIndividualTrajectories ? [] : nil

        for result in trajectoryResults {
            for k in 0 ..< numOperators {
                totalJumpCounts[k] += result.jumpCounts[k]
                totalJumps += result.jumpCounts[k]
            }
            if configuration.storeIndividualTrajectories {
                individualStates?.append(result.state)
            }
        }

        let averageDensityMatrix = computeAverageDensityMatrix(
            states: trajectoryResults.map(\.state),
            qubits: initialState.qubits,
        )

        let statistics = TrajectoryStatistics(
            jumpCounts: totalJumpCounts,
            averageJumpsPerTrajectory: Double(totalJumps) / Double(numTrajectories),
            trajectoryCount: numTrajectories,
        )

        return TrajectoryResult(
            averageDensityMatrix: averageDensityMatrix,
            time: time,
            statistics: statistics,
            individualTrajectories: individualStates,
        )
    }

    // MARK: - Private Implementation

    @_optimize(speed)
    @inlinable
    static func runSingleTrajectory(
        initialState: QuantumState,
        hEff: [[Complex<Double>]],
        jumpOperators: [[[Complex<Double>]]],
        lDaggerL: [[[Complex<Double>]]],
        dt: Double,
        timeSteps: Int,
        numOperators: Int,
        dimension: Int,
        seed: UInt64?,
    ) -> (state: QuantumState, jumpCounts: [Int]) {
        var rng = Measurement.createRNG(seed: seed)
        var psi = initialState.amplitudes
        var jumpCounts = [Int](repeating: 0, count: numOperators)

        for _ in 0 ..< timeSteps {
            let psiEvolved = applyNonHermitianStep(psi: psi, hEff: hEff, dt: dt, dimension: dimension)

            let normSquared = computeNormSquared(psiEvolved, dimension: dimension)
            let dp = 1.0 - normSquared

            let r = Double.random(in: 0.0 ..< 1.0, using: &rng)

            if r < dp, dp > 1e-15 {
                let jumpProbs = computeJumpProbabilities(
                    psi: psi,
                    lDaggerL: lDaggerL,
                    numOperators: numOperators,
                    dimension: dimension,
                )

                let kSelected = sampleJumpOperator(jumpProbs: jumpProbs, rng: &rng)
                jumpCounts[kSelected] += 1

                psi = applyJump(psi: psi, L: jumpOperators[kSelected], dimension: dimension)
            } else {
                psi = normalizeVector(psiEvolved, dimension: dimension)
            }
        }

        let finalState = QuantumState(qubits: initialState.qubits, amplitudes: psi)
        return (finalState, jumpCounts)
    }

    @_optimize(speed)
    @inlinable
    static func buildHamiltonianMatrix(
        hamiltonian: Observable,
        dimension: Int,
    ) -> [[Complex<Double>]] {
        var hMatrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: dimension),
            count: dimension,
        )

        for term in hamiltonian.terms {
            let coeff = term.coefficient
            let pauliString = term.pauliString

            for row in 0 ..< dimension {
                let (col, phase) = pauliString.applyToRow(row: row)
                hMatrix[row][col] = hMatrix[row][col] + Complex(coeff, 0) * phase
            }
        }

        return hMatrix
    }

    @_optimize(speed)
    @inlinable
    static func buildEffectiveHamiltonian(
        hMatrix: [[Complex<Double>]],
        jumpOperators: [[[Complex<Double>]]],
        dimension: Int,
    ) -> [[Complex<Double>]] {
        var hEff = hMatrix

        let minusHalfI = Complex<Double>(0.0, -0.5)

        for L in jumpOperators {
            let lDagger = MatrixUtilities.hermitianConjugate(L)
            let lDaggerL = MatrixUtilities.matrixMultiply(lDagger, L)

            for i in 0 ..< dimension {
                for j in 0 ..< dimension {
                    hEff[i][j] = hEff[i][j] + minusHalfI * lDaggerL[i][j]
                }
            }
        }

        return hEff
    }

    @_optimize(speed)
    @inlinable
    static func buildLDaggerL(
        jumpOperators: [[[Complex<Double>]]],
        dimension _: Int,
    ) -> [[[Complex<Double>]]] {
        jumpOperators.map { L in
            let lDagger = MatrixUtilities.hermitianConjugate(L)
            return MatrixUtilities.matrixMultiply(lDagger, L)
        }
    }

    @_optimize(speed)
    @inlinable
    static func applyNonHermitianStep(
        psi: [Complex<Double>],
        hEff: [[Complex<Double>]],
        dt: Double,
        dimension: Int,
    ) -> [Complex<Double>] {
        let minusIDt = Complex<Double>(0.0, -dt)

        let result = [Complex<Double>](unsafeUninitializedCapacity: dimension) { buffer, count in
            for i in 0 ..< dimension {
                var sum = psi[i]
                for j in 0 ..< dimension {
                    sum = sum + minusIDt * hEff[i][j] * psi[j]
                }
                buffer[i] = sum
            }
            count = dimension
        }

        return result
    }

    @_optimize(speed)
    @inlinable
    static func computeNormSquared(_ psi: [Complex<Double>], dimension: Int) -> Double {
        var sum = 0.0
        for i in 0 ..< dimension {
            sum += psi[i].magnitudeSquared
        }
        return sum
    }

    @_optimize(speed)
    @inlinable
    static func computeJumpProbabilities(
        psi: [Complex<Double>],
        lDaggerL: [[[Complex<Double>]]],
        numOperators: Int,
        dimension: Int,
    ) -> [Double] {
        var probs = [Double](unsafeUninitializedCapacity: numOperators) { buffer, count in
            for k in 0 ..< numOperators {
                var expectation = 0.0
                for i in 0 ..< dimension {
                    var lDaggerLPsi_i = Complex<Double>.zero
                    for j in 0 ..< dimension {
                        lDaggerLPsi_i = lDaggerLPsi_i + lDaggerL[k][i][j] * psi[j]
                    }
                    expectation += (psi[i].conjugate * lDaggerLPsi_i).real
                }
                buffer[k] = max(0.0, expectation)
            }
            count = numOperators
        }

        let total = probs.reduce(0.0, +)
        if total > 1e-15 {
            let invTotal = 1.0 / total
            for k in 0 ..< numOperators {
                probs[k] *= invTotal
            }
        }

        return probs
    }

    @_optimize(speed)
    @inlinable
    static func sampleJumpOperator(
        jumpProbs: [Double],
        rng: inout any RandomNumberGenerator,
    ) -> Int {
        let r = Double.random(in: 0.0 ..< 1.0, using: &rng)
        var cumulative = 0.0
        for k in 0 ..< jumpProbs.count {
            cumulative += jumpProbs[k]
            if r < cumulative {
                return k
            }
        }
        return jumpProbs.count - 1
    }

    @_optimize(speed)
    @inlinable
    static func applyJump(
        psi: [Complex<Double>],
        L: [[Complex<Double>]],
        dimension: Int,
    ) -> [Complex<Double>] {
        let lPsi = [Complex<Double>](unsafeUninitializedCapacity: dimension) { buffer, count in
            for i in 0 ..< dimension {
                var sum = Complex<Double>.zero
                for j in 0 ..< dimension {
                    sum = sum + L[i][j] * psi[j]
                }
                buffer[i] = sum
            }
            count = dimension
        }

        return normalizeVector(lPsi, dimension: dimension)
    }

    @_optimize(speed)
    @inlinable
    static func normalizeVector(_ psi: [Complex<Double>], dimension: Int) -> [Complex<Double>] {
        let normSq = computeNormSquared(psi, dimension: dimension)
        let invNorm = 1.0 / sqrt(normSq)

        return [Complex<Double>](unsafeUninitializedCapacity: dimension) { buffer, count in
            for i in 0 ..< dimension {
                buffer[i] = psi[i] * invNorm
            }
            count = dimension
        }
    }

    @_optimize(speed)
    @inlinable
    static func computeAverageDensityMatrix(
        states: [QuantumState],
        qubits: Int,
    ) -> DensityMatrix {
        let dim = 1 << qubits
        let size = dim * dim
        let numStates = states.count
        let invCount = 1.0 / Double(numStates)

        var elements = [Complex<Double>](unsafeUninitializedCapacity: size) { buffer, count in
            buffer.initialize(repeating: .zero)
            count = size
        }

        for state in states {
            let amps = state.amplitudes
            for i in 0 ..< dim {
                let ci = amps[i]
                for j in 0 ..< dim {
                    let cj = amps[j]
                    elements[i * dim + j] = elements[i * dim + j] + ci * cj.conjugate
                }
            }
        }

        for idx in 0 ..< size {
            elements[idx] = elements[idx] * invCount
        }

        return DensityMatrix(qubits: qubits, elements: elements)
    }
}
