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
///     shouldStoreIndividualTrajectories: true
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
    public let shouldStoreIndividualTrajectories: Bool

    /// Creates a trajectory simulation configuration.
    ///
    /// - Parameters:
    ///   - trajectories: Number of Monte Carlo samples (default: 1000)
    ///   - seed: Random seed for reproducibility (default: nil for system random)
    ///   - timeSteps: Time discretization steps (default: 1000)
    ///   - shouldStoreIndividualTrajectories: Retain all trajectory states (default: false)
    ///
    /// **Example:**
    /// ```swift
    /// let config = TrajectoryConfiguration(
    ///     trajectories: 2000,
    ///     seed: 42,
    ///     timeSteps: 500
    /// )
    /// ```
    public init(
        trajectories: Int = 1000,
        seed: UInt64? = nil,
        timeSteps: Int = 1000,
        shouldStoreIndividualTrajectories: Bool = false,
    ) {
        self.trajectories = trajectories
        self.seed = seed
        self.timeSteps = timeSteps
        self.shouldStoreIndividualTrajectories = shouldStoreIndividualTrajectories
    }
}

/// Statistics collected during quantum trajectory simulation.
///
/// Tracks jump events across all trajectories for each Lindblad operator, providing
/// insight into decoherence rates and dominant dissipation channels.
///
/// **Example:**
/// ```swift
/// let stats = TrajectoryStatistics(jumpCounts: [15, 7], averageJumpsPerTrajectory: 2.2, trajectoryCount: 10)
/// let totalJumps = stats.jumpCounts.reduce(0, +)
/// let avgJumps = stats.averageJumpsPerTrajectory
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
/// let L: [[Complex<Double>]] = [[.zero, .one], [.zero, .zero]]
/// let psi = QuantumState(qubits: 1, amplitudes: [.zero, .one])
/// let result = await QuantumTrajectory.evolve(
///     hamiltonian: Observable.pauliZ(qubit: 0, coefficient: 1.0),
///     jumpOperators: [L], initialState: psi, time: 5.0)
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

    /// Individual trajectory final states (nil unless shouldStoreIndividualTrajectories was true).
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
/// Each trajectory constructs an effective Hamiltonian H_eff = H - (i/2)Σₖ Lₖ†Lₖ,
/// then iterates: evolve |ψ(t+dt)⟩ = (I - iH_eff×dt)|ψ⟩, compute jump probability
/// dp = 1 - ⟨ψ|ψ⟩, apply a jump Lₖ with probability proportional to ⟨ψ|Lₖ†Lₖ|ψ⟩
/// if a random threshold is exceeded, and renormalize.
///
/// **Example:**
/// ```swift
/// let L: [[Complex<Double>]] = [[.zero, .one], [.zero, .zero]]
/// let psi = QuantumState(qubits: 1, amplitudes: [.zero, .one])
/// let H = Observable.pauliZ(qubit: 0, coefficient: 1.0)
/// let result = await QuantumTrajectory.evolve(
///     hamiltonian: H, jumpOperators: [L], initialState: psi, time: 5.0)
/// ```
///
/// - SeeAlso: ``TrajectoryConfiguration``
/// - SeeAlso: ``TrajectoryResult``
/// - SeeAlso: ``DensityMatrix``
public enum QuantumTrajectory {
    private static let jumpProbabilityEpsilon: Double = 1e-15
    private static let normalizationEpsilon: Double = 1e-30

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
    /// let L: [[Complex<Double>]] = [[.zero, .one], [.zero, .zero]]
    /// let psi = QuantumState(qubits: 1, amplitudes: [.zero, .one])
    /// let result = await QuantumTrajectory.evolve(
    ///     hamiltonian: Observable.pauliZ(qubit: 0, coefficient: 1.0),
    ///     jumpOperators: [L], initialState: psi, time: 10.0)
    /// ```
    ///
    /// - Complexity: O(trajectories × timeSteps × dim²) where dim = 2^n
    /// - Precondition: initialState must be normalized
    /// - Precondition: time > 0
    /// - Precondition: trajectories > 0
    /// - Precondition: timeSteps > 0
    /// - Precondition: Each jump operator must be a square matrix
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
        for idx in 0 ..< jumpOperators.count {
            ValidationUtilities.validateSquareMatrix(jumpOperators[idx], name: "Jump operator \(idx)")
            ValidationUtilities.validateMatrixDimensionEquals(jumpOperators[idx], expected: dim, name: "Jump operator \(idx)")
        }

        let dt = time / Double(configuration.timeSteps)
        let lDaggerL = buildLDaggerL(jumpOperators: jumpOperators, dimension: dim)
        let hMatrix = buildHamiltonianMatrix(hamiltonian: hamiltonian, dimension: dim)
        let hEff = buildEffectiveHamiltonian(hMatrix: hMatrix, lDaggerL: lDaggerL, dimension: dim)
        let flatJumpOps: [[Complex<Double>]] = jumpOperators.map { L in
            let d = L.count
            return [Complex<Double>](unsafeUninitializedCapacity: d * d) { buffer, count in
                for i in 0 ..< d {
                    for j in 0 ..< d {
                        buffer[i * d + j] = L[i][j]
                    }
                }
                count = d * d
            }
        }

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
                        flatJumpOperators: flatJumpOps,
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
        var individualStates: [QuantumState]?
        if configuration.shouldStoreIndividualTrajectories {
            var states = [QuantumState]()
            states.reserveCapacity(numTrajectories)
            individualStates = states
        }

        for result in trajectoryResults {
            for k in 0 ..< numOperators {
                totalJumpCounts[k] += result.jumpCounts[k]
                totalJumps += result.jumpCounts[k]
            }
            if configuration.shouldStoreIndividualTrajectories {
                individualStates?.append(result.state)
            }
        }

        let averageDensityMatrix = computeAverageDensityMatrix(
            trajectoryResults: trajectoryResults,
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

    /// Constructs flat Hamiltonian matrix from Pauli observable.
    @_optimize(speed)
    @_effects(readonly)
    private static func buildHamiltonianMatrix(
        hamiltonian: Observable,
        dimension: Int,
    ) -> [Complex<Double>] {
        let size = dimension * dimension
        var hMatrix = [Complex<Double>](unsafeUninitializedCapacity: size) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = size
        }

        for term in hamiltonian.terms {
            let coeffComplex = Complex(term.coefficient, 0)
            let pauliString = term.pauliString
            for row in 0 ..< dimension {
                let (col, phase) = pauliString.applyToRow(row: row)
                hMatrix[row * dimension + col] = hMatrix[row * dimension + col] + coeffComplex * phase
            }
        }

        return hMatrix
    }

    /// Computes L†L products for each jump operator in flat format.
    @_optimize(speed)
    @_effects(readonly)
    private static func buildLDaggerL(
        jumpOperators: [[[Complex<Double>]]],
        dimension: Int,
    ) -> [[Complex<Double>]] {
        jumpOperators.map { L in
            let lDagger = MatrixUtilities.hermitianConjugate(L)
            let product = MatrixUtilities.matrixMultiply(lDagger, L)
            return [Complex<Double>](unsafeUninitializedCapacity: dimension * dimension) { buffer, count in
                for i in 0 ..< dimension {
                    for j in 0 ..< dimension {
                        buffer[i * dimension + j] = product[i][j]
                    }
                }
                count = dimension * dimension
            }
        }
    }

    /// Builds effective non-Hermitian Hamiltonian H_eff = H - (i/2)Σ L†L.
    @_optimize(speed)
    @_effects(readonly)
    private static func buildEffectiveHamiltonian(
        hMatrix: [Complex<Double>],
        lDaggerL: [[Complex<Double>]],
        dimension: Int,
    ) -> [Complex<Double>] {
        var hEff = hMatrix
        let size = dimension * dimension
        if size >= 64 {
            var scale = (0.0, -0.5)
            withUnsafePointer(to: &scale) { scalePtr in
                for ldl in lDaggerL {
                    ldl.withUnsafeBufferPointer { ldlBuf in
                        hEff.withUnsafeMutableBufferPointer { hEffBuf in
                            cblas_zaxpy(
                                Int32(size),
                                OpaquePointer(scalePtr),
                                OpaquePointer(ldlBuf.baseAddress), 1, // Safety: ldl has size elements where size >= 64
                                OpaquePointer(hEffBuf.baseAddress), 1, // Safety: hEff has size elements where size >= 64
                            )
                        }
                    }
                }
            }
        } else {
            let minusHalfI = Complex<Double>(0.0, -0.5)
            for ldl in lDaggerL {
                for idx in 0 ..< size {
                    hEff[idx] = hEff[idx] + minusHalfI * ldl[idx]
                }
            }
        }
        return hEff
    }

    /// Simulates a single stochastic quantum trajectory with jumps.
    @_optimize(speed)
    private static func runSingleTrajectory(
        initialState: QuantumState,
        hEff: [Complex<Double>],
        flatJumpOperators: [[Complex<Double>]],
        lDaggerL: [[Complex<Double>]],
        dt: Double,
        timeSteps: Int,
        numOperators: Int,
        dimension: Int,
        seed: UInt64?,
    ) -> (state: QuantumState, jumpCounts: [Int]) {
        var rng = Measurement.createRNG(seed: seed)
        var psi = initialState.amplitudes
        var jumpCounts = [Int](repeating: 0, count: numOperators)
        var scratch = [Complex<Double>](repeating: .zero, count: dimension)
        var jumpProbs = [Double](repeating: 0.0, count: numOperators)

        for _ in 0 ..< timeSteps {
            applyNonHermitianStep(psi: psi, hEff: hEff, dt: dt, dimension: dimension, result: &scratch)

            let normSquared = computeNormSquared(scratch, dimension: dimension)
            let dp = 1.0 - normSquared

            let r = Double.random(in: 0.0 ..< 1.0, using: &rng)

            if r < dp, dp > jumpProbabilityEpsilon {
                computeJumpProbabilities(
                    psi: psi,
                    lDaggerL: lDaggerL,
                    numOperators: numOperators,
                    dimension: dimension,
                    probs: &jumpProbs,
                    tmpVec: &scratch,
                )

                let kSelected = sampleJumpOperator(jumpProbs: jumpProbs, rng: &rng)
                jumpCounts[kSelected] += 1

                matrixVectorMultiply(matrix: flatJumpOperators[kSelected], vector: psi, result: &scratch, dimension: dimension)
            }

            normalizeVectorInPlace(&scratch, dimension: dimension)
            swap(&psi, &scratch)
        }

        let finalState = QuantumState(qubits: initialState.qubits, amplitudes: psi)
        return (finalState, jumpCounts)
    }

    /// Multiplies flat complex matrix by vector with BLAS threshold dispatch.
    @_optimize(speed)
    private static func matrixVectorMultiply(
        matrix: [Complex<Double>],
        vector: [Complex<Double>],
        result: inout [Complex<Double>],
        dimension: Int,
    ) {
        if dimension <= 4 {
            for i in 0 ..< dimension {
                var sum = Complex<Double>.zero
                for j in 0 ..< dimension {
                    sum = sum + matrix[i * dimension + j] * vector[j]
                }
                result[i] = sum
            }
        } else {
            var alpha = (1.0, 0.0)
            var beta = (0.0, 0.0)
            withUnsafePointer(to: &alpha) { alphaPtr in
                withUnsafePointer(to: &beta) { betaPtr in
                    matrix.withUnsafeBufferPointer { matPtr in
                        vector.withUnsafeBufferPointer { vecPtr in
                            result.withUnsafeMutableBufferPointer { resPtr in
                                cblas_zgemv(
                                    CblasRowMajor, CblasNoTrans,
                                    Int32(dimension), Int32(dimension),
                                    OpaquePointer(alphaPtr),
                                    OpaquePointer(matPtr.baseAddress), Int32(dimension), // Safety: matrix has dimension² elements where dimension >= 5
                                    OpaquePointer(vecPtr.baseAddress), 1, // Safety: vector has dimension elements where dimension >= 5
                                    OpaquePointer(betaPtr),
                                    OpaquePointer(resPtr.baseAddress), 1, // Safety: result has dimension elements where dimension >= 5
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    /// Applies one Euler step under the effective non-Hermitian Hamiltonian.
    @_optimize(speed)
    private static func applyNonHermitianStep(
        psi: [Complex<Double>],
        hEff: [Complex<Double>],
        dt: Double,
        dimension: Int,
        result: inout [Complex<Double>],
    ) {
        if dimension <= 4 {
            let minusIDt = Complex<Double>(0.0, -dt)
            for i in 0 ..< dimension {
                var sum = psi[i]
                for j in 0 ..< dimension {
                    sum = sum + minusIDt * hEff[i * dimension + j] * psi[j]
                }
                result[i] = sum
            }
        } else {
            var alpha = (0.0, -dt)
            var beta = (1.0, 0.0)
            withUnsafePointer(to: &alpha) { alphaPtr in
                withUnsafePointer(to: &beta) { betaPtr in
                    hEff.withUnsafeBufferPointer { hPtr in
                        psi.withUnsafeBufferPointer { psiPtr in
                            result.withUnsafeMutableBufferPointer { resPtr in
                                resPtr.baseAddress!.update(from: psiPtr.baseAddress!, count: dimension) // Safety: both buffers have dimension elements where dimension >= 5
                                cblas_zgemv(
                                    CblasRowMajor, CblasNoTrans,
                                    Int32(dimension), Int32(dimension),
                                    OpaquePointer(alphaPtr),
                                    OpaquePointer(hPtr.baseAddress), Int32(dimension), // Safety: hEff has dimension² elements
                                    OpaquePointer(psiPtr.baseAddress), 1,
                                    OpaquePointer(betaPtr),
                                    OpaquePointer(resPtr.baseAddress), 1,
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    /// Computes squared norm of a complex vector.
    @_optimize(speed)
    @_effects(readonly)
    private static func computeNormSquared(_ psi: [Complex<Double>], dimension: Int) -> Double {
        if dimension < 32 {
            var sum = 0.0
            for i in 0 ..< dimension {
                sum += psi[i].magnitudeSquared
            }
            return sum
        }
        return psi.withUnsafeBufferPointer { buf in
            let base = UnsafeRawPointer(buf.baseAddress!).assumingMemoryBound(to: Double.self) // Safety: psi has dimension complex elements where dimension >= 32
            var result = 0.0
            vDSP_svesqD(base, 1, &result, vDSP_Length(2 * dimension))
            return result
        }
    }

    /// Computes normalized jump probabilities for each Lindblad operator.
    @_optimize(speed)
    private static func computeJumpProbabilities(
        psi: [Complex<Double>],
        lDaggerL: [[Complex<Double>]],
        numOperators: Int,
        dimension: Int,
        probs: inout [Double],
        tmpVec: inout [Complex<Double>],
    ) {
        if dimension <= 4 {
            for k in 0 ..< numOperators {
                var expectation = 0.0
                for i in 0 ..< dimension {
                    var ldlPsi_i = Complex<Double>.zero
                    for j in 0 ..< dimension {
                        ldlPsi_i = ldlPsi_i + lDaggerL[k][i * dimension + j] * psi[j]
                    }
                    expectation += (psi[i].conjugate * ldlPsi_i).real
                }
                probs[k] = max(0.0, expectation)
            }
        } else {
            var alpha = (1.0, 0.0)
            var beta = (0.0, 0.0)

            withUnsafePointer(to: &alpha) { alphaPtr in
                withUnsafePointer(to: &beta) { betaPtr in
                    for k in 0 ..< numOperators {
                        lDaggerL[k].withUnsafeBufferPointer { ldlPtr in
                            psi.withUnsafeBufferPointer { psiPtr in
                                tmpVec.withUnsafeMutableBufferPointer { tmpPtr in
                                    cblas_zgemv(
                                        CblasRowMajor, CblasNoTrans,
                                        Int32(dimension), Int32(dimension),
                                        OpaquePointer(alphaPtr),
                                        OpaquePointer(ldlPtr.baseAddress), Int32(dimension), // Safety: lDaggerL[k] has dimension² elements where dimension >= 5
                                        OpaquePointer(psiPtr.baseAddress), 1,
                                        OpaquePointer(betaPtr),
                                        OpaquePointer(tmpPtr.baseAddress), 1,
                                    )
                                }
                            }
                        }
                        psi.withUnsafeBufferPointer { psiBuf in
                            tmpVec.withUnsafeBufferPointer { tmpBuf in
                                let psiBase = UnsafeRawPointer(psiBuf.baseAddress!).assumingMemoryBound(to: Double.self) // Safety: psi has dimension elements where dimension >= 5
                                let tmpBase = UnsafeRawPointer(tmpBuf.baseAddress!).assumingMemoryBound(to: Double.self) // Safety: tmpVec has dimension elements where dimension >= 5
                                var dotResult = 0.0
                                vDSP_dotprD(psiBase, 1, tmpBase, 1, &dotResult, vDSP_Length(2 * dimension))
                                probs[k] = max(0.0, dotResult)
                            }
                        }
                    }
                }
            }
        }

        let total = probs.reduce(0.0, +)
        if total > jumpProbabilityEpsilon {
            let invTotal = 1.0 / total
            for k in 0 ..< numOperators {
                probs[k] *= invTotal
            }
        }
    }

    /// Selects a jump operator index by sampling from the probability distribution.
    @_optimize(speed)
    private static func sampleJumpOperator(
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

    /// Normalizes a complex vector to unit norm in place.
    @_optimize(speed)
    private static func normalizeVectorInPlace(_ psi: inout [Complex<Double>], dimension: Int) {
        let normSq = computeNormSquared(psi, dimension: dimension)
        let invNorm = 1.0 / sqrt(normSq)

        if dimension < 32 {
            for i in 0 ..< dimension {
                psi[i] = psi[i] * invNorm
            }
        } else {
            psi.withUnsafeMutableBufferPointer { buf in
                let base = UnsafeMutableRawPointer(buf.baseAddress!).assumingMemoryBound(to: Double.self) // Safety: psi has dimension elements where dimension >= 32
                var scale = invNorm
                vDSP_vsmulD(base, 1, &scale, base, 1, vDSP_Length(2 * dimension))
            }
        }
    }

    /// Averages outer products of trajectory states into a density matrix.
    @_optimize(speed)
    @_effects(readonly)
    private static func computeAverageDensityMatrix(
        trajectoryResults: [(state: QuantumState, jumpCounts: [Int])],
        qubits: Int,
    ) -> DensityMatrix {
        let dim = 1 << qubits
        let size = dim * dim
        let numStates = trajectoryResults.count
        let invCount = 1.0 / Double(numStates)

        var elements = [Complex<Double>](unsafeUninitializedCapacity: size) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = size
        }

        var alpha = (1.0, 0.0)
        withUnsafePointer(to: &alpha) { alphaPtr in
            for result in trajectoryResults {
                let amps = result.state.amplitudes
                amps.withUnsafeBufferPointer { ampsBuf in
                    elements.withUnsafeMutableBufferPointer { elemBuf in
                        let ampsPtr = UnsafeRawPointer(ampsBuf.baseAddress!).assumingMemoryBound(to: Double.self) // Safety: amps has dim elements where dim >= 2
                        let elemPtr = UnsafeMutableRawPointer(elemBuf.baseAddress!).assumingMemoryBound(to: Double.self) // Safety: elements has dim² elements where dim >= 2
                        cblas_zgerc(
                            CblasRowMajor, Int32(dim), Int32(dim),
                            OpaquePointer(alphaPtr),
                            OpaquePointer(ampsPtr), 1,
                            OpaquePointer(ampsPtr), 1,
                            OpaquePointer(elemPtr), Int32(dim),
                        )
                    }
                }
            }
        }

        elements.withUnsafeMutableBufferPointer { buf in
            let base = UnsafeMutableRawPointer(buf.baseAddress!).assumingMemoryBound(to: Double.self) // Safety: elements has size complex elements where size >= 4
            var scale = invCount
            vDSP_vsmulD(base, 1, &scale, base, 1, vDSP_Length(2 * size))
        }

        return DensityMatrix(qubits: qubits, elements: elements)
    }
}
