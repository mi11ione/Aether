// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// TEBD evolution gate generators for common Hamiltonians.
///
/// Provides efficient gate matrix computation for time evolution of nearest-neighbor
/// Hamiltonians using Time-Evolving Block Decimation (TEBD). Each gate represents
/// the exponential exp(-i*theta*O) of a local operator O for a given angle theta.
/// Gates are returned as 4x4 (two-site) or 2x2 (single-site) unitary matrices.
///
/// **Example:**
/// ```swift
/// let zzGate = TEBDGates.zzEvolution(angle: 0.1)
/// let xxGate = TEBDGates.xxEvolution(angle: 0.1)
/// let heisenbergGate = TEBDGates.heisenbergXXZ(angle: 0.1, delta: 1.0)
/// ```
///
/// - SeeAlso: ``MPSTimeEvolution``
/// - SeeAlso: ``TEBDResult``
public enum TEBDGates {
    /// Compute ZZ evolution gate exp(-i*theta*ZZ) as 4x4 matrix.
    ///
    /// The ZZ operator has eigenvalues +1 for |00>, |11> and -1 for |01>, |10>.
    /// The exponential is diagonal: diag(e^(-i*theta), e^(i*theta), e^(i*theta), e^(-i*theta)).
    ///
    /// **Example:**
    /// ```swift
    /// let gate = TEBDGates.zzEvolution(angle: 0.5)
    /// let diagonal00 = gate[0][0]  // exp(-0.5i)
    /// let diagonal11 = gate[3][3]  // exp(-0.5i)
    /// ```
    ///
    /// - Parameter angle: Rotation angle theta in exp(-i*theta*ZZ)
    /// - Returns: 4x4 unitary matrix for the ZZ evolution gate
    /// - Complexity: O(1)
    @_optimize(speed)
    @_effects(readonly)
    public static func zzEvolution(angle: Double) -> [[Complex<Double>]] {
        let expMinus = Complex(cos(angle), -sin(angle))
        let expPlus = Complex(cos(angle), sin(angle))
        return [
            [expMinus, .zero, .zero, .zero],
            [.zero, expPlus, .zero, .zero],
            [.zero, .zero, expPlus, .zero],
            [.zero, .zero, .zero, expMinus],
        ]
    }

    /// Compute XX evolution gate exp(-i*theta*XX) as 4x4 matrix.
    ///
    /// The XX operator couples |00> with |11> and |01> with |10>. The exponential
    /// has block structure with cos(theta) on diagonal and -i*sin(theta) on off-diagonal
    /// within each 2x2 block.
    ///
    /// **Example:**
    /// ```swift
    /// let gate = TEBDGates.xxEvolution(angle: 0.5)
    /// let diagonal = gate[0][0]     // cos(0.5)
    /// let offDiag = gate[0][3]      // -i*sin(0.5)
    /// ```
    ///
    /// - Parameter angle: Rotation angle theta in exp(-i*theta*XX)
    /// - Returns: 4x4 unitary matrix for the XX evolution gate
    /// - Complexity: O(1)
    @_optimize(speed)
    @_effects(readonly)
    public static func xxEvolution(angle: Double) -> [[Complex<Double>]] {
        let c = Complex<Double>(cos(angle), 0)
        let iSin = Complex<Double>(0, -sin(angle))
        return [
            [c, .zero, .zero, iSin],
            [.zero, c, iSin, .zero],
            [.zero, iSin, c, .zero],
            [iSin, .zero, .zero, c],
        ]
    }

    /// Compute YY evolution gate exp(-i*theta*YY) as 4x4 matrix.
    ///
    /// The YY operator couples |00> with |11> (with sign flip) and |01> with |10>.
    /// The exponential has similar structure to XX but with different phase factors.
    ///
    /// **Example:**
    /// ```swift
    /// let gate = TEBDGates.yyEvolution(angle: 0.5)
    /// let diagonal = gate[0][0]     // cos(0.5)
    /// let offDiag = gate[0][3]      // i*sin(0.5)
    /// ```
    ///
    /// - Parameter angle: Rotation angle theta in exp(-i*theta*YY)
    /// - Returns: 4x4 unitary matrix for the YY evolution gate
    /// - Complexity: O(1)
    @_optimize(speed)
    @_effects(readonly)
    public static func yyEvolution(angle: Double) -> [[Complex<Double>]] {
        let c = Complex<Double>(cos(angle), 0)
        let iSinPos = Complex<Double>(0, sin(angle))
        let iSinNeg = Complex<Double>(0, -sin(angle))
        return [
            [c, .zero, .zero, iSinPos],
            [.zero, c, iSinNeg, .zero],
            [.zero, iSinNeg, c, .zero],
            [iSinPos, .zero, .zero, c],
        ]
    }

    /// Compute Heisenberg XXZ evolution exp(-i*theta*(XX + YY + delta*ZZ)).
    ///
    /// Combines XX, YY, and ZZ interactions for the anisotropic Heisenberg model.
    /// At delta=1 (isotropic case), this is the standard Heisenberg XXX model.
    /// Uses direct matrix exponentiation in the computational basis.
    ///
    /// **Example:**
    /// ```swift
    /// let isoGate = TEBDGates.heisenbergXXZ(angle: 0.1, delta: 1.0)
    /// let anisoGate = TEBDGates.heisenbergXXZ(angle: 0.1, delta: 0.5)
    /// ```
    ///
    /// - Parameters:
    ///   - angle: Rotation angle theta in exp(-i*theta*H)
    ///   - delta: Anisotropy parameter for ZZ term (delta=1 is isotropic)
    /// - Returns: 4x4 unitary matrix for the Heisenberg XXZ evolution gate
    /// - Complexity: O(1)
    @_optimize(speed)
    @_effects(readonly)
    public static func heisenbergXXZ(angle: Double, delta: Double) -> [[Complex<Double>]] {
        let zzPhase = delta * angle
        let expMinusZZ = Complex(cos(zzPhase), -sin(zzPhase))
        let expPlusZZ = Complex(cos(zzPhase), sin(zzPhase))

        let xyAngle = 2.0 * angle
        let cosXY = cos(xyAngle)
        let sinXY = sin(xyAngle)

        let diagMiddle = Complex<Double>(cosXY, 0) * expPlusZZ
        let offDiagMiddle = Complex<Double>(0, -sinXY) * expPlusZZ

        return [
            [expMinusZZ, .zero, .zero, .zero],
            [.zero, diagMiddle, offDiagMiddle, .zero],
            [.zero, offDiagMiddle, diagMiddle, .zero],
            [.zero, .zero, .zero, expMinusZZ],
        ]
    }

    /// Compute single-site X evolution exp(-i*theta*X) as 2x2 matrix.
    ///
    /// Implements rotation about the X axis on the Bloch sphere. The matrix has
    /// cos(theta) on the diagonal and -i*sin(theta) on the off-diagonal.
    ///
    /// **Example:**
    /// ```swift
    /// let gate = TEBDGates.xEvolution(angle: 0.5)
    /// let diagonal = gate[0][0]     // cos(0.5)
    /// let offDiag = gate[0][1]      // -i*sin(0.5)
    /// ```
    ///
    /// - Parameter angle: Rotation angle theta in exp(-i*theta*X)
    /// - Returns: 2x2 unitary matrix for the X evolution gate
    /// - Complexity: O(1)
    @_optimize(speed)
    @_effects(readonly)
    public static func xEvolution(angle: Double) -> [[Complex<Double>]] {
        let c = Complex<Double>(cos(angle), 0)
        let iSin = Complex<Double>(0, -sin(angle))
        return [
            [c, iSin],
            [iSin, c],
        ]
    }

    /// Compute single-site Z evolution exp(-i*theta*Z) as 2x2 matrix.
    ///
    /// Implements rotation about the Z axis on the Bloch sphere. The matrix is
    /// diagonal with exp(-i*theta) for |0> and exp(i*theta) for |1>.
    ///
    /// **Example:**
    /// ```swift
    /// let gate = TEBDGates.zEvolution(angle: 0.5)
    /// let phase0 = gate[0][0]  // exp(-0.5i)
    /// let phase1 = gate[1][1]  // exp(0.5i)
    /// ```
    ///
    /// - Parameter angle: Rotation angle theta in exp(-i*theta*Z)
    /// - Returns: 2x2 unitary matrix for the Z evolution gate
    /// - Complexity: O(1)
    @_optimize(speed)
    @_effects(readonly)
    public static func zEvolution(angle: Double) -> [[Complex<Double>]] {
        let expMinus = Complex(cos(angle), -sin(angle))
        let expPlus = Complex(cos(angle), sin(angle))
        return [
            [expMinus, .zero],
            [.zero, expPlus],
        ]
    }
}

/// Result of MPS time evolution via TEBD.
///
/// Contains the final evolved state along with statistics about the evolution process
/// including truncation errors accumulated during SVD operations, the maximum bond
/// dimension reached, and the total number of gates applied.
///
/// **Example:**
/// ```swift
/// let evolution = MPSTimeEvolution()
/// let result = await evolution.evolveIsing(mps: initialState, J: 1.0, h: 0.5, time: 1.0, steps: 10)
/// print(result.time)
/// print(result.truncationStatistics.cumulativeError)
/// print(result.maxBondDimensionReached)
/// ```
///
/// - SeeAlso: ``MPSTimeEvolution``
/// - SeeAlso: ``MPSTruncationStatistics``
@frozen
public struct TEBDResult: Sendable {
    /// The final MPS state after time evolution
    public let finalState: MatrixProductState

    /// Total evolution time
    public let time: Double

    /// Number of Trotter steps performed
    public let steps: Int

    /// Statistics on truncation errors accumulated during evolution
    public let truncationStatistics: MPSTruncationStatistics

    /// Maximum bond dimension reached during evolution
    public let maxBondDimensionReached: Int

    /// Total number of two-site gates applied
    public let totalGatesApplied: Int
}

/// MPS-optimized time evolution using TEBD.
///
/// Implements Time-Evolving Block Decimation (TEBD) for efficient simulation of
/// one-dimensional quantum systems with nearest-neighbor interactions. Uses
/// Trotter-Suzuki decomposition to approximate the time evolution operator as
/// a sequence of two-site gates applied in even-odd sweeps.
///
/// TEBD is particularly efficient for MPS because two-site gates only affect
/// adjacent tensors, allowing local SVD truncation without full chain contraction.
/// GPU acceleration via Metal Performance Shaders is automatically enabled for
/// bond dimensions >= 32.
///
/// **Example:**
/// ```swift
/// let evolution = MPSTimeEvolution()
/// var mps = MatrixProductState(qubits: 20, maxBondDimension: 64)
///
/// let isingResult = await evolution.evolveIsing(
///     mps: mps,
///     J: 1.0,
///     h: 0.5,
///     time: 1.0,
///     steps: 100,
///     order: .second
/// )
///
/// let heisenbergResult = await evolution.evolveHeisenberg(
///     mps: mps,
///     J: 1.0,
///     delta: 1.0,
///     time: 1.0,
///     steps: 100
/// )
/// ```
///
/// - SeeAlso: ``TEBDGates``
/// - SeeAlso: ``TEBDResult``
/// - SeeAlso: ``TrotterOrder``
public actor MPSTimeEvolution {
    private let accelerator: MPSMetalAcceleration

    /// Creates an MPS time evolution engine with GPU acceleration.
    ///
    /// Initializes the Metal accelerator for GPU-accelerated tensor operations.
    /// GPU acceleration is automatically used when bond dimension >= 32.
    ///
    /// **Example:**
    /// ```swift
    /// let evolution = MPSTimeEvolution()
    /// ```
    public init() {
        accelerator = MPSMetalAcceleration()
    }

    /// Evolve MPS under Ising Hamiltonian using TEBD.
    ///
    /// Simulates time evolution under the transverse-field Ising model:
    /// H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i
    ///
    /// The ZZ interaction is applied as two-site gates on alternating bonds,
    /// while the transverse field X terms are applied as single-site gates.
    ///
    /// **Example:**
    /// ```swift
    /// let evolution = MPSTimeEvolution()
    /// var mps = MatrixProductState(qubits: 10, maxBondDimension: 32)
    /// let result = await evolution.evolveIsing(
    ///     mps: mps,
    ///     J: 1.0,
    ///     h: 0.5,
    ///     time: 1.0,
    ///     steps: 50,
    ///     order: .second
    /// )
    /// let finalEnergy = result.finalState.expectationValue(of: isingHamiltonian)
    /// ```
    ///
    /// - Parameters:
    ///   - mps: Initial MPS state to evolve
    ///   - J: ZZ coupling strength
    ///   - h: Transverse field strength
    ///   - time: Total evolution time
    ///   - steps: Number of Trotter steps
    ///   - order: Trotter decomposition order (default: .second)
    /// - Returns: TEBDResult containing evolved state and statistics
    /// - Complexity: O(steps * qubits * chi^3) where chi is bond dimension
    @_optimize(speed)
    public func evolveIsing(
        mps: MatrixProductState,
        J: Double,
        h: Double,
        time: Double,
        steps: Int,
        order: TrotterOrder = .second,
    ) async -> TEBDResult {
        ValidationUtilities.validatePositiveInt(steps, name: "TEBD steps")
        ValidationUtilities.validatePositiveDouble(time, name: "Evolution time")

        let dt = time / Double(steps)

        var singleSiteGates: [[[Complex<Double>]]]? = nil
        if abs(h) > 1e-15 {
            let xGate = TEBDGates.xEvolution(angle: h * dt)
            singleSiteGates = Array(repeating: xGate, count: mps.qubits)
        }

        let zzAngle = J * dt
        let zzGate = TEBDGates.zzEvolution(angle: zzAngle)

        return await evolveWithGate(
            mps: mps,
            twoSiteGate: zzGate,
            singleSiteGates: singleSiteGates ?? [],
            time: time,
            steps: steps,
            order: order,
        )
    }

    /// Evolve MPS under Heisenberg XXZ Hamiltonian using TEBD.
    ///
    /// Simulates time evolution under the anisotropic Heisenberg model:
    /// H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + delta * Z_i Z_{i+1})
    ///
    /// At delta=1, this is the isotropic Heisenberg XXX model. The entire
    /// two-site interaction is combined into a single gate for efficiency.
    ///
    /// **Example:**
    /// ```swift
    /// let evolution = MPSTimeEvolution()
    /// var mps = MatrixProductState(qubits: 10, maxBondDimension: 32)
    /// let result = await evolution.evolveHeisenberg(
    ///     mps: mps,
    ///     J: 1.0,
    ///     delta: 1.0,
    ///     time: 1.0,
    ///     steps: 50,
    ///     order: .second
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - mps: Initial MPS state to evolve
    ///   - J: Exchange coupling strength
    ///   - delta: Anisotropy parameter (delta=1 is isotropic)
    ///   - time: Total evolution time
    ///   - steps: Number of Trotter steps
    ///   - order: Trotter decomposition order (default: .second)
    /// - Returns: TEBDResult containing evolved state and statistics
    /// - Complexity: O(steps * qubits * chi^3) where chi is bond dimension
    @_optimize(speed)
    public func evolveHeisenberg(
        mps: MatrixProductState,
        J: Double,
        delta: Double,
        time: Double,
        steps: Int,
        order: TrotterOrder = .second,
    ) async -> TEBDResult {
        ValidationUtilities.validatePositiveInt(steps, name: "TEBD steps")
        ValidationUtilities.validatePositiveDouble(time, name: "Evolution time")

        let dt = time / Double(steps)
        let heisenbergGate = TEBDGates.heisenbergXXZ(angle: J * dt, delta: delta)

        return await evolveWithGate(
            mps: mps,
            twoSiteGate: heisenbergGate,
            singleSiteGates: [],
            time: time,
            steps: steps,
            order: order,
        )
    }

    /// Evolve MPS under general nearest-neighbor Hamiltonian.
    ///
    /// Applies a custom two-site gate to all adjacent pairs using TEBD with
    /// the specified Trotter order. Optional single-site gates are applied
    /// after each sweep of two-site gates.
    ///
    /// **Example:**
    /// ```swift
    /// let evolution = MPSTimeEvolution()
    /// let customGate = TEBDGates.zzEvolution(angle: 0.1)
    /// let xGates = (0..<10).map { _ in TEBDGates.xEvolution(angle: 0.05) }
    /// let result = await evolution.evolveWithGate(
    ///     mps: mps,
    ///     twoSiteGate: customGate,
    ///     singleSiteGates: xGates,
    ///     time: 1.0,
    ///     steps: 100,
    ///     order: .second
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - mps: Initial MPS state to evolve
    ///   - twoSiteGate: 4x4 unitary matrix for nearest-neighbor interaction
    ///   - singleSiteGates: Optional array of 2x2 single-site gates (one per site)
    ///   - time: Total evolution time
    ///   - steps: Number of Trotter steps
    ///   - order: Trotter decomposition order (default: .second)
    /// - Returns: TEBDResult containing evolved state and statistics
    /// - Complexity: O(steps * qubits * chi^3) where chi is bond dimension
    @_optimize(speed)
    public func evolveWithGate(
        mps: MatrixProductState,
        twoSiteGate: [[Complex<Double>]],
        singleSiteGates: [[[Complex<Double>]]],
        time: Double,
        steps: Int,
        order: TrotterOrder = .second,
    ) async -> TEBDResult {
        ValidationUtilities.validatePositiveInt(steps, name: "TEBD steps")

        var state = mps
        var totalGates = 0
        var maxBondDim = state.currentMaxBondDimension

        let singleGates: [[[Complex<Double>]]]? = singleSiteGates.isEmpty ? nil : singleSiteGates

        for _ in 0 ..< steps {
            let gatesApplied = await applyTrotterStep(
                mps: &state,
                twoSiteGate: twoSiteGate,
                singleSiteGates: singleGates,
                order: order,
            )
            totalGates += gatesApplied
            maxBondDim = max(maxBondDim, state.currentMaxBondDimension)
        }

        return TEBDResult(
            finalState: state,
            time: time,
            steps: steps,
            truncationStatistics: state.truncationStatistics,
            maxBondDimensionReached: maxBondDim,
            totalGatesApplied: totalGates,
        )
    }

    @_optimize(speed)
    private func applyTrotterStep(
        mps: inout MatrixProductState,
        twoSiteGate: [[Complex<Double>]],
        singleSiteGates: [[[Complex<Double>]]]?,
        order: TrotterOrder,
    ) async -> Int {
        switch order {
        case .first:
            await applyFirstOrderStep(
                mps: &mps,
                gate: twoSiteGate,
                singleSiteGates: singleSiteGates,
            )
        case .second:
            await applySecondOrderStep(
                mps: &mps,
                gate: twoSiteGate,
                singleSiteGates: singleSiteGates,
            )
        case .fourth:
            await applyFourthOrderStep(
                mps: &mps,
                gate: twoSiteGate,
                singleSiteGates: singleSiteGates,
            )
        case .sixth:
            await applySixthOrderStep(
                mps: &mps,
                gate: twoSiteGate,
                singleSiteGates: singleSiteGates,
            )
        }
    }

    @_optimize(speed)
    private func applyFirstOrderStep(
        mps: inout MatrixProductState,
        gate: [[Complex<Double>]],
        singleSiteGates: [[[Complex<Double>]]]?,
    ) async -> Int {
        var gatesApplied = 0

        gatesApplied += await applyEvenOddSweep(mps: &mps, gate: gate, factor: 1.0)

        if let singleGates = singleSiteGates {
            applySingleSiteGates(mps: &mps, gates: singleGates)
        }

        return gatesApplied
    }

    @_optimize(speed)
    private func applySecondOrderStep(
        mps: inout MatrixProductState,
        gate: [[Complex<Double>]],
        singleSiteGates: [[[Complex<Double>]]]?,
    ) async -> Int {
        var gatesApplied = 0

        gatesApplied += await applyEvenBonds(mps: &mps, gate: gate, factor: 0.5)
        gatesApplied += await applyOddBonds(mps: &mps, gate: gate, factor: 1.0)
        gatesApplied += await applyEvenBonds(mps: &mps, gate: gate, factor: 0.5)

        if let singleGates = singleSiteGates {
            applySingleSiteGates(mps: &mps, gates: singleGates)
        }

        return gatesApplied
    }

    @_optimize(speed)
    private func applyFourthOrderStep(
        mps: inout MatrixProductState,
        gate: [[Complex<Double>]],
        singleSiteGates: [[[Complex<Double>]]]?,
    ) async -> Int {
        let s = 1.0 / (4.0 - pow(4.0, 1.0 / 3.0))
        let centralFactor = 1.0 - 4.0 * s

        var gatesApplied = 0

        gatesApplied += await applySecondOrderSweep(mps: &mps, gate: gate, factor: s)
        gatesApplied += await applySecondOrderSweep(mps: &mps, gate: gate, factor: s)
        gatesApplied += await applySecondOrderSweep(mps: &mps, gate: gate, factor: centralFactor)
        gatesApplied += await applySecondOrderSweep(mps: &mps, gate: gate, factor: s)
        gatesApplied += await applySecondOrderSweep(mps: &mps, gate: gate, factor: s)

        if let singleGates = singleSiteGates {
            applySingleSiteGates(mps: &mps, gates: singleGates)
        }

        return gatesApplied
    }

    @_optimize(speed)
    private func applySixthOrderStep(
        mps: inout MatrixProductState,
        gate: [[Complex<Double>]],
        singleSiteGates: [[[Complex<Double>]]]?,
    ) async -> Int {
        let s = 1.0 / (4.0 - pow(4.0, 1.0 / 5.0))
        let centralFactor = 1.0 - 4.0 * s

        var gatesApplied = 0

        gatesApplied += await applyFourthOrderSweep(mps: &mps, gate: gate, factor: s)
        gatesApplied += await applyFourthOrderSweep(mps: &mps, gate: gate, factor: s)
        gatesApplied += await applyFourthOrderSweep(mps: &mps, gate: gate, factor: centralFactor)
        gatesApplied += await applyFourthOrderSweep(mps: &mps, gate: gate, factor: s)
        gatesApplied += await applyFourthOrderSweep(mps: &mps, gate: gate, factor: s)

        if let singleGates = singleSiteGates {
            applySingleSiteGates(mps: &mps, gates: singleGates)
        }

        return gatesApplied
    }

    @_optimize(speed)
    private func applySecondOrderSweep(
        mps: inout MatrixProductState,
        gate: [[Complex<Double>]],
        factor: Double,
    ) async -> Int {
        var gatesApplied = 0
        gatesApplied += await applyEvenBonds(mps: &mps, gate: gate, factor: factor * 0.5)
        gatesApplied += await applyOddBonds(mps: &mps, gate: gate, factor: factor)
        gatesApplied += await applyEvenBonds(mps: &mps, gate: gate, factor: factor * 0.5)
        return gatesApplied
    }

    @_optimize(speed)
    private func applyFourthOrderSweep(
        mps: inout MatrixProductState,
        gate: [[Complex<Double>]],
        factor: Double,
    ) async -> Int {
        let s = 1.0 / (4.0 - pow(4.0, 1.0 / 3.0))
        let centralFactor = 1.0 - 4.0 * s

        var gatesApplied = 0
        gatesApplied += await applySecondOrderSweep(mps: &mps, gate: gate, factor: factor * s)
        gatesApplied += await applySecondOrderSweep(mps: &mps, gate: gate, factor: factor * s)
        gatesApplied += await applySecondOrderSweep(mps: &mps, gate: gate, factor: factor * centralFactor)
        gatesApplied += await applySecondOrderSweep(mps: &mps, gate: gate, factor: factor * s)
        gatesApplied += await applySecondOrderSweep(mps: &mps, gate: gate, factor: factor * s)
        return gatesApplied
    }

    @_optimize(speed)
    private func applyEvenOddSweep(
        mps: inout MatrixProductState,
        gate: [[Complex<Double>]],
        factor: Double,
    ) async -> Int {
        var gatesApplied = 0
        gatesApplied += await applyEvenBonds(mps: &mps, gate: gate, factor: factor)
        gatesApplied += await applyOddBonds(mps: &mps, gate: gate, factor: factor)
        return gatesApplied
    }

    @_optimize(speed)
    private func applyEvenBonds(
        mps: inout MatrixProductState,
        gate: [[Complex<Double>]],
        factor: Double,
    ) async -> Int {
        let scaledGate = scaleGate(gate, factor: factor)
        var gatesApplied = 0

        var site = 0
        while site + 1 < mps.qubits {
            await applyTwoSiteGate(mps: &mps, gate: scaledGate, site: site)
            gatesApplied += 1
            site += 2
        }

        return gatesApplied
    }

    @_optimize(speed)
    private func applyOddBonds(
        mps: inout MatrixProductState,
        gate: [[Complex<Double>]],
        factor: Double,
    ) async -> Int {
        let scaledGate = scaleGate(gate, factor: factor)
        var gatesApplied = 0

        var site = 1
        while site + 1 < mps.qubits {
            await applyTwoSiteGate(mps: &mps, gate: scaledGate, site: site)
            gatesApplied += 1
            site += 2
        }

        return gatesApplied
    }

    @_optimize(speed)
    @_effects(readonly)
    private func scaleGate(_ gate: [[Complex<Double>]], factor: Double) -> [[Complex<Double>]] {
        guard abs(factor - 1.0) > 1e-15 else { return gate }

        return gate.map { row in
            row.map { element in
                let scaledPhase = element.phase * factor
                let mag = element.magnitude
                return Complex(mag * cos(scaledPhase), mag * sin(scaledPhase))
            }
        }
    }

    @_optimize(speed)
    private func applySingleSiteGates(
        mps: inout MatrixProductState,
        gates: [[[Complex<Double>]]],
    ) {
        let numGates = min(gates.count, mps.qubits)

        for site in 0 ..< numGates {
            let gate = gates[site]
            applySingleSiteGate(mps: &mps, gate: gate, site: site)
        }
    }

    @_optimize(speed)
    private func applySingleSiteGate(
        mps: inout MatrixProductState,
        gate: [[Complex<Double>]],
        site: Int,
    ) {
        let tensor = mps.tensors[site]
        let leftDim = tensor.leftBondDimension
        let rightDim = tensor.rightBondDimension

        let u00 = gate[0][0]
        let u01 = gate[0][1]
        let u10 = gate[1][0]
        let u11 = gate[1][1]

        let newElements = [Complex<Double>](unsafeUninitializedCapacity: leftDim * 2 * rightDim) { buffer, count in
            for alpha in 0 ..< leftDim {
                for beta in 0 ..< rightDim {
                    let a0 = tensor[alpha, 0, beta]
                    let a1 = tensor[alpha, 1, beta]

                    let newA0 = u00 * a0 + u01 * a1
                    let newA1 = u10 * a0 + u11 * a1

                    let idx0 = alpha * (2 * rightDim) + 0 * rightDim + beta
                    let idx1 = alpha * (2 * rightDim) + 1 * rightDim + beta

                    buffer[idx0] = newA0
                    buffer[idx1] = newA1
                }
            }
            count = leftDim * 2 * rightDim
        }

        mps.updateTensor(at: site, with: MPSTensor(
            leftBondDimension: leftDim,
            rightBondDimension: rightDim,
            site: site,
            elements: newElements,
        ))
    }

    /// Apply single TEBD step (even-odd sweep).
    ///
    /// Performs one complete TEBD step by applying two-site gates to all even bonds,
    /// then all odd bonds. Optionally applies single-site gates after the sweep.
    ///
    /// **Example:**
    /// ```swift
    /// let evolution = MPSTimeEvolution()
    /// var mps = MatrixProductState(qubits: 10, maxBondDimension: 32)
    /// let evenGate = TEBDGates.zzEvolution(angle: 0.05)
    /// let oddGate = TEBDGates.zzEvolution(angle: 0.1)
    /// await evolution.applyTEBDStep(
    ///     mps: &mps,
    ///     evenGate: evenGate,
    ///     oddGate: oddGate,
    ///     singleSiteGates: nil
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - mps: MPS state to modify in place
    ///   - evenGate: 4x4 gate for even bonds (0-1, 2-3, ...)
    ///   - oddGate: 4x4 gate for odd bonds (1-2, 3-4, ...)
    ///   - singleSiteGates: Optional array of 2x2 single-site gates
    /// - Complexity: O(qubits * chi^3) where chi is bond dimension
    @_optimize(speed)
    func applyTEBDStep(
        mps: inout MatrixProductState,
        evenGate: [[Complex<Double>]],
        oddGate: [[Complex<Double>]],
        singleSiteGates: [[[Complex<Double>]]]?,
    ) async {
        var site = 0
        while site + 1 < mps.qubits {
            await applyTwoSiteGate(mps: &mps, gate: evenGate, site: site)
            site += 2
        }

        site = 1
        while site + 1 < mps.qubits {
            await applyTwoSiteGate(mps: &mps, gate: oddGate, site: site)
            site += 2
        }

        if let singleGates = singleSiteGates {
            applySingleSiteGates(mps: &mps, gates: singleGates)
        }
    }

    /// Apply two-site gate to adjacent sites with SVD truncation.
    ///
    /// Contracts the tensors at site and site+1, applies the 4x4 gate matrix,
    /// then performs SVD to split back into two tensors with truncation to
    /// maxBondDimension. GPU acceleration is used when bond dimension >= 32.
    ///
    /// **Example:**
    /// ```swift
    /// let evolution = MPSTimeEvolution()
    /// var mps = MatrixProductState(qubits: 10, maxBondDimension: 32)
    /// let gate = TEBDGates.zzEvolution(angle: 0.1)
    /// await evolution.applyTwoSiteGate(mps: &mps, gate: gate, site: 3)
    /// ```
    ///
    /// - Parameters:
    ///   - mps: MPS state to modify in place
    ///   - gate: 4x4 unitary gate matrix
    ///   - site: Left site index (gate acts on site and site+1)
    /// - Complexity: O(chi^3) for SVD decomposition
    @_optimize(speed)
    func applyTwoSiteGate(
        mps: inout MatrixProductState,
        gate: [[Complex<Double>]],
        site: Int,
    ) async {
        let rightSite = site + 1
        let tensorA = mps.tensors[site]
        let tensorB = mps.tensors[rightSite]

        let chiL = tensorA.leftBondDimension
        let chiM = tensorA.rightBondDimension
        let chiR = tensorB.rightBondDimension

        let maxDim = max(chiL, chiM, chiR)
        let useGPU = maxDim >= MPSMetalAcceleration.gpuThreshold && accelerator.isAvailable

        let combined: [Complex<Double>]
        if useGPU {
            let contracted = await accelerator.contractAdjacentTensors(tensorA, tensorB)
            combined = flattenContracted(contracted, chiL: chiL, chiR: chiR)
        } else {
            combined = contractTensorsCPU(tensorA, tensorB, chiL: chiL, chiM: chiM, chiR: chiR)
        }

        var transformed = [Complex<Double>](repeating: .zero, count: chiL * 4 * chiR)

        for alpha in 0 ..< chiL {
            for gamma in 0 ..< chiR {
                for iPrime in 0 ..< 2 {
                    for jPrime in 0 ..< 2 {
                        var sum: Complex<Double> = .zero
                        for i in 0 ..< 2 {
                            for j in 0 ..< 2 {
                                let gateRow = iPrime * 2 + jPrime
                                let gateCol = i * 2 + j
                                let combIdx = alpha * (4 * chiR) + i * (2 * chiR) + j * chiR + gamma
                                sum = sum + gate[gateRow][gateCol] * combined[combIdx]
                            }
                        }
                        let outIdx = alpha * (4 * chiR) + iPrime * (2 * chiR) + jPrime * chiR + gamma
                        transformed[outIdx] = sum
                    }
                }
            }
        }

        let rows = chiL * 2
        let cols = 2 * chiR
        var matrix = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: cols), count: rows)
        for alpha in 0 ..< chiL {
            for i in 0 ..< 2 {
                let row = alpha * 2 + i
                for j in 0 ..< 2 {
                    for gamma in 0 ..< chiR {
                        let col = j * chiR + gamma
                        let idx = alpha * (4 * chiR) + i * (2 * chiR) + j * chiR + gamma
                        matrix[row][col] = transformed[idx]
                    }
                }
            }
        }

        let svdResult = SVDDecomposition.decompose(matrix: matrix, truncation: .maxBondDimension(mps.maxBondDimension))
        mps.addTruncationError(svdResult.truncationError)

        let newChiM = svdResult.singularValues.count

        let sqrtS = [Double](unsafeUninitializedCapacity: newChiM) { buffer, count in
            for k in 0 ..< newChiM {
                buffer[k] = svdResult.singularValues[k].squareRoot()
            }
            count = newChiM
        }

        let newAElements = [Complex<Double>](unsafeUninitializedCapacity: chiL * 2 * newChiM) { buffer, count in
            for alpha in 0 ..< chiL {
                for i in 0 ..< 2 {
                    let row = alpha * 2 + i
                    for beta in 0 ..< newChiM {
                        let idx = alpha * (2 * newChiM) + i * newChiM + beta
                        buffer[idx] = svdResult.u[row][beta] * sqrtS[beta]
                    }
                }
            }
            count = chiL * 2 * newChiM
        }

        let newBElements = [Complex<Double>](unsafeUninitializedCapacity: newChiM * 2 * chiR) { buffer, count in
            for beta in 0 ..< newChiM {
                for j in 0 ..< 2 {
                    for gamma in 0 ..< chiR {
                        let col = j * chiR + gamma
                        let idx = beta * (2 * chiR) + j * chiR + gamma
                        buffer[idx] = sqrtS[beta] * svdResult.vDagger[beta][col]
                    }
                }
            }
            count = newChiM * 2 * chiR
        }

        mps.updateTensor(at: site, with: MPSTensor(
            leftBondDimension: chiL,
            rightBondDimension: newChiM,
            site: site,
            elements: newAElements,
        ))

        mps.updateTensor(at: rightSite, with: MPSTensor(
            leftBondDimension: newChiM,
            rightBondDimension: chiR,
            site: rightSite,
            elements: newBElements,
        ))
    }

    @_optimize(speed)
    @_effects(readonly)
    private func flattenContracted(
        _ contracted: [[[[Complex<Double>]]]],
        chiL: Int,
        chiR: Int,
    ) -> [Complex<Double>] {
        [Complex<Double>](unsafeUninitializedCapacity: chiL * 4 * chiR) { buffer, count in
            for alpha in 0 ..< chiL {
                for i in 0 ..< 2 {
                    for j in 0 ..< 2 {
                        for gamma in 0 ..< chiR {
                            let idx = alpha * (4 * chiR) + i * (2 * chiR) + j * chiR + gamma
                            buffer[idx] = contracted[alpha][i][j][gamma]
                        }
                    }
                }
            }
            count = chiL * 4 * chiR
        }
    }

    @_optimize(speed)
    @_effects(readonly)
    private func contractTensorsCPU(
        _ tensorA: MPSTensor,
        _ tensorB: MPSTensor,
        chiL: Int,
        chiM: Int,
        chiR: Int,
    ) -> [Complex<Double>] {
        [Complex<Double>](unsafeUninitializedCapacity: chiL * 4 * chiR) { buffer, count in
            for alpha in 0 ..< chiL {
                for i in 0 ..< 2 {
                    for j in 0 ..< 2 {
                        for gamma in 0 ..< chiR {
                            var sum: Complex<Double> = .zero
                            for beta in 0 ..< chiM {
                                sum = sum + tensorA[alpha, i, beta] * tensorB[beta, j, gamma]
                            }
                            let idx = alpha * (4 * chiR) + i * (2 * chiR) + j * chiR + gamma
                            buffer[idx] = sum
                        }
                    }
                }
            }
            count = chiL * 4 * chiR
        }
    }
}

/// Extension for convenient MPS time evolution.
///
/// Provides direct time evolution methods on MatrixProductState for common
/// Hamiltonians without explicitly creating an MPSTimeEvolution instance.
///
/// **Example:**
/// ```swift
/// var mps = MatrixProductState(qubits: 10, maxBondDimension: 32)
/// let isingResult = await mps.evolvingIsing(J: 1.0, h: 0.5, time: 1.0, steps: 100)
/// let heisenbergResult = await mps.evolvingHeisenberg(J: 1.0, delta: 1.0, time: 1.0, steps: 100)
/// ```
public extension MatrixProductState {
    /// Evolve under Ising Hamiltonian.
    ///
    /// Simulates time evolution under H = -J * sum_i Z_i Z_{i+1} - h * sum_i X_i
    /// using second-order Trotter-Suzuki decomposition.
    ///
    /// **Example:**
    /// ```swift
    /// var mps = MatrixProductState(qubits: 10, maxBondDimension: 32)
    /// let result = await mps.evolvingIsing(J: 1.0, h: 0.5, time: 1.0, steps: 100)
    /// print(result.finalState.currentMaxBondDimension)
    /// ```
    ///
    /// - Parameters:
    ///   - J: ZZ coupling strength
    ///   - h: Transverse field strength
    ///   - time: Total evolution time
    ///   - steps: Number of Trotter steps
    /// - Returns: TEBDResult containing evolved state and statistics
    /// - Complexity: O(steps * qubits * chi^3)
    func evolvingIsing(
        J: Double,
        h: Double,
        time: Double,
        steps: Int,
    ) async -> TEBDResult {
        let evolution = MPSTimeEvolution()
        return await evolution.evolveIsing(
            mps: self,
            J: J,
            h: h,
            time: time,
            steps: steps,
            order: .second,
        )
    }

    /// Evolve under Heisenberg XXZ Hamiltonian.
    ///
    /// Simulates time evolution under H = J * sum_i (X_i X_{i+1} + Y_i Y_{i+1} + delta * Z_i Z_{i+1})
    /// using second-order Trotter-Suzuki decomposition.
    ///
    /// **Example:**
    /// ```swift
    /// var mps = MatrixProductState(qubits: 10, maxBondDimension: 32)
    /// let result = await mps.evolvingHeisenberg(J: 1.0, delta: 1.0, time: 1.0, steps: 100)
    /// print(result.truncationStatistics.cumulativeError)
    /// ```
    ///
    /// - Parameters:
    ///   - J: Exchange coupling strength
    ///   - delta: Anisotropy parameter (delta=1 is isotropic)
    ///   - time: Total evolution time
    ///   - steps: Number of Trotter steps
    /// - Returns: TEBDResult containing evolved state and statistics
    /// - Complexity: O(steps * qubits * chi^3)
    func evolvingHeisenberg(
        J: Double,
        delta: Double,
        time: Double,
        steps: Int,
    ) async -> TEBDResult {
        let evolution = MPSTimeEvolution()
        return await evolution.evolveHeisenberg(
            mps: self,
            J: J,
            delta: delta,
            time: time,
            steps: steps,
            order: .second,
        )
    }
}
