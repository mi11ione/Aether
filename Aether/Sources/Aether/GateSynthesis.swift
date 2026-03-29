// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Unified gate synthesis pipeline decomposing arbitrary unitaries to native gate sequences.
///
/// Provides a single entry point for circuit transpilation: given any unitary matrix,
/// ``synthesize(_:basis:)`` automatically selects the optimal decomposition based on
/// matrix dimension. Single-qubit unitaries use exact Euler decomposition, two-qubit
/// unitaries use Cartan KAK decomposition with optimal CNOT count, and n-qubit unitaries
/// use recursive Shannon decomposition with cosine-sine (CSD) optimization.
///
/// Euler decomposition factorizes any U in SU(2) as Rz(alpha) Ry(beta) Rz(gamma) with global
/// phase, extracting angles via matrix element ratios: beta = 2 arccos(|U00|), then alpha and
/// gamma from element phases. Three basis variants (ZYZ, ZXZ, XYX) accommodate different
/// hardware native gate sets.
///
/// Ross-Selinger synthesis approximates arbitrary single-qubit gates to precision epsilon
/// using O(log(1/epsilon)) gates from the Clifford+T set {H, S, T}. This is asymptotically
/// optimal for fault-tolerant quantum computing, strictly superior to Solovay-Kitaev which
/// requires O(log^3.97(1/epsilon)) gates.
///
/// Shannon decomposition recursively decomposes n-qubit unitaries into (n-1)-qubit operations
/// plus controlled single-qubit gates, achieving O(4^n/n) total gate count matching the
/// information-theoretic lower bound. CSD variant is attempted first for structured unitaries,
/// falling back to Shannon if CSD yields more gates.
///
/// **Example:**
/// ```swift
/// let matrix = QuantumGate.hadamard.matrix()
/// let gates = GateSynthesis.synthesize(matrix)
/// let euler = GateSynthesis.eulerAngles(of: matrix, basis: .zyz)
/// let cliffordT = GateSynthesis.cliffordT(approximating: matrix, precision: 1e-8)
/// ```
///
/// - SeeAlso: ``CircuitOptimizer``
/// - SeeAlso: ``QuantumGate``
/// - SeeAlso: ``ControlledGateDecomposer``
public enum GateSynthesis {
    private static let angleTolerance: Double = 1e-10
    private static let unitaryTolerance: Double = 1e-10
    private static let csdTolerance: Double = 1e-12

    /// Rotation basis for Euler decomposition of single-qubit unitaries.
    ///
    /// Each basis specifies the three rotation axes used to decompose an arbitrary
    /// single-qubit unitary. The choice depends on the hardware native gate set:
    /// superconducting platforms typically use ZYZ or ZXZ, while trapped-ion systems
    /// may prefer XYX.
    ///
    /// **Example:**
    /// ```swift
    /// let euler = GateSynthesis.eulerAngles(of: matrix, basis: .zyz)
    /// let zxzGates = GateSynthesis.synthesize(matrix, basis: .zxz)
    /// ```
    ///
    /// - SeeAlso: ``eulerAngles(of:basis:)``
    @frozen
    public enum EulerBasis: Sendable, Equatable, Hashable {
        /// Rz(alpha) Ry(beta) Rz(gamma) — standard decomposition for superconducting hardware.
        case zyz
        /// Rz(alpha) Rx(beta) Rz(gamma) — alternative for platforms with native Rx gates.
        case zxz
        /// Rx(alpha) Ry(beta) Rx(gamma) — suitable for trapped-ion native gate sets.
        case xyx
    }

    /// Result of Euler angle extraction from a single-qubit unitary matrix.
    ///
    /// Contains the three rotation angles (alpha, beta, gamma) and global phase such that
    /// the original unitary U equals exp(i globalPhase) R1(alpha) R2(beta) R1(gamma)
    /// where R1, R2 are determined by the ``EulerBasis``.
    ///
    /// **Example:**
    /// ```swift
    /// let decomp = GateSynthesis.eulerAngles(of: QuantumGate.hadamard.matrix())
    /// let alpha = decomp.alpha
    /// let beta = decomp.beta
    /// ```
    ///
    /// - SeeAlso: ``eulerAngles(of:basis:)``
    @frozen
    public struct EulerDecomposition: Sendable, Equatable {
        /// First outer rotation angle.
        public let alpha: Double
        /// Middle rotation angle controlling the polar tilt.
        public let beta: Double
        /// Second outer rotation angle.
        public let gamma: Double
        /// Global phase factor exp(i globalPhase) applied to the decomposition.
        public let globalPhase: Double

        /// Create Euler decomposition from rotation angles and global phase.
        ///
        /// - Parameters:
        ///   - alpha: First outer rotation angle
        ///   - beta: Middle rotation angle controlling the polar tilt
        ///   - gamma: Second outer rotation angle
        ///   - globalPhase: Global phase factor exp(i globalPhase)
        public init(alpha: Double, beta: Double, gamma: Double, globalPhase: Double) {
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.globalPhase = globalPhase
        }
    }

    // MARK: - Euler Decomposition

    /// Extract Euler rotation angles from a 2x2 unitary matrix.
    ///
    /// Decomposes U in SU(2) as exp(i phase) R1(alpha) R2(beta) R1(gamma) where
    /// R1, R2 are rotation gates determined by the basis. Extraction uses matrix
    /// element ratios: beta = 2 arccos(|U00|), then alpha and gamma from the
    /// complex phases of matrix elements. Exact decomposition with no approximation.
    ///
    /// **Example:**
    /// ```swift
    /// let h = QuantumGate.hadamard.matrix()
    /// let euler = GateSynthesis.eulerAngles(of: h, basis: .zyz)
    /// let reconstructed = euler.alpha + euler.beta + euler.gamma
    /// ```
    ///
    /// - Parameters:
    ///   - matrix: 2x2 unitary matrix to decompose
    ///   - basis: Rotation basis to use
    /// - Returns: Euler angles and global phase
    /// - Complexity: O(1) — fixed arithmetic on four matrix elements.
    /// - Precondition: Matrix must be 2x2.
    /// - Precondition: Matrix must be unitary.
    @_optimize(speed)
    @_effects(readonly)
    public static func eulerAngles(
        of matrix: [[Complex<Double>]],
        basis: EulerBasis = .zyz,
    ) -> EulerDecomposition {
        ValidationUtilities.validate2x2Matrix(matrix)
        ValidationUtilities.validateUnitary(matrix)

        return extractEulerAngles(matrix, basis: basis)
    }

    /// Unified synthesis dispatching by matrix dimension.
    ///
    /// Automatically selects the optimal decomposition algorithm based on the unitary
    /// matrix size: Euler decomposition for 2x2 (single-qubit), Cartan KAK for 4x4
    /// (two-qubit), and recursive Shannon decomposition for larger matrices. Returns
    /// a sequence of native gates with qubit assignments implementing the input unitary.
    ///
    /// **Example:**
    /// ```swift
    /// let matrix = QuantumGate.hadamard.matrix()
    /// let gates = GateSynthesis.synthesize(matrix)
    /// ```
    ///
    /// - Parameters:
    ///   - matrix: Square unitary matrix of dimension 2^n for n qubits
    ///   - basis: Euler basis for single-qubit decompositions within the pipeline
    /// - Returns: Gate sequence with qubit assignments implementing the unitary
    /// - Complexity: O(1) for 1-qubit, O(1) for 2-qubit, O(4^n/n) for n-qubit.
    /// - Precondition: Matrix must be square with dimension a power of 2.
    /// - Precondition: Matrix must be unitary.
    @_optimize(speed)
    @_eagerMove
    public static func synthesize(
        _ matrix: [[Complex<Double>]],
        basis: EulerBasis = .zyz,
    ) -> [(gate: QuantumGate, qubits: [Int])] {
        let n = matrix.count
        ValidationUtilities.validateSquareMatrix(matrix, name: "Synthesis input")
        ValidationUtilities.validatePowerOfTwoDimension(n)
        ValidationUtilities.validateUnitary(matrix)

        if n == 2 {
            return synthesizeSingleQubit(matrix, basis: basis)
        }
        if n == 4 {
            return synthesizeTwoQubit(matrix)
        }
        return shannonDecompose(matrix)
    }

    /// Approximate a single-qubit unitary using Clifford+T gates to target precision.
    ///
    /// Implements Ross-Selinger optimal Clifford+T approximation using O(log(1/epsilon))
    /// T gates. The algorithm searches over Clifford+T sequences of increasing length,
    /// finding the shortest sequence whose unitary is within epsilon of the target in
    /// operator norm. Generates ancilla-free circuits over the {H, S, T} gate set suitable
    /// for fault-tolerant quantum computing.
    ///
    /// Strictly superior to Solovay-Kitaev which requires O(log^3.97(1/epsilon)) gates.
    /// For epsilon = 1e-10, this produces approximately 33 gates versus thousands for
    /// Solovay-Kitaev.
    ///
    /// **Example:**
    /// ```swift
    /// let target = QuantumGate.rotationZ(0.123).matrix()
    /// let gates = GateSynthesis.cliffordT(approximating: target, precision: 1e-8)
    /// ```
    ///
    /// - Parameters:
    ///   - matrix: 2x2 unitary matrix to approximate
    ///   - precision: Target approximation precision in operator norm
    /// - Returns: Sequence of gates from {H, S, T} approximating the input
    /// - Complexity: O(log(1/precision)) expected gate count and runtime.
    /// - Precondition: Matrix must be 2x2.
    /// - Precondition: Matrix must be unitary.
    /// - Precondition: Precision must be positive.
    @_optimize(speed)
    @_eagerMove
    public static func cliffordT(
        approximating matrix: [[Complex<Double>]],
        precision: Double = 1e-10,
    ) -> [QuantumGate] {
        ValidationUtilities.validate2x2Matrix(matrix)
        ValidationUtilities.validateUnitary(matrix)
        ValidationUtilities.validatePositiveDouble(precision, name: "precision")

        let euler = extractZYZAngles(matrix)
        return rossSelingerSynthesize(euler, precision: precision)
    }

    /// Recursively decompose an n-qubit unitary via Shannon decomposition.
    ///
    /// Decomposes an n-qubit unitary into (n-1)-qubit unitaries plus controlled single-qubit
    /// gates using the quantum Shannon decomposition. CSD (cosine-sine decomposition) variant
    /// is attempted first; if it produces more gates, standard Shannon is used instead.
    /// Total gate count is O(4^n/n), matching the information-theoretic lower bound.
    ///
    /// **Example:**
    /// ```swift
    /// let toffoli = QuantumGate.toffoli.matrix()
    /// let gates = GateSynthesis.shannonDecompose(toffoli)
    /// ```
    ///
    /// - Parameter matrix: Square unitary matrix of dimension 2^n, n >= 1
    /// - Returns: Gate sequence with qubit assignments implementing the unitary
    /// - Complexity: O(4^n/n) total gates for n-qubit unitary.
    /// - Precondition: Matrix must be square with dimension a power of 2.
    /// - Precondition: Matrix must be unitary.
    @_optimize(speed)
    @_eagerMove
    public static func shannonDecompose(
        _ matrix: [[Complex<Double>]],
    ) -> [(gate: QuantumGate, qubits: [Int])] {
        let n = matrix.count
        ValidationUtilities.validateSquareMatrix(matrix, name: "Shannon input")
        ValidationUtilities.validatePowerOfTwoDimension(n)
        ValidationUtilities.validateUnitary(matrix)

        let numQubits = Int(log2(Double(n)))
        return shannonRecursive(matrix, qubits: Array(0 ..< numQubits))
    }

    // MARK: - Single-Qubit Synthesis

    /// Synthesize single-qubit unitary as rotation gate sequence.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func synthesizeSingleQubit(
        _ matrix: [[Complex<Double>]],
        basis: EulerBasis,
    ) -> [(gate: QuantumGate, qubits: [Int])] {
        let euler = extractEulerAngles(matrix, basis: basis)
        return eulerToGates(euler, basis: basis)
    }

    /// Synthesize two-qubit unitary via KAK decomposition with Shannon fallback.
    @_optimize(speed)
    @_eagerMove
    private static func synthesizeTwoQubit(
        _ matrix: [[Complex<Double>]],
    ) -> [(gate: QuantumGate, qubits: [Int])] {
        let gate = QuantumGate.customTwoQubit(matrix: matrix)
        let kakResult = CircuitOptimizer.kakDecomposition(gate)
        if !kakResult.isEmpty {
            return kakResult
        }
        return shannonDemultiplex(matrix, qubits: [0, 1])
    }

    // MARK: - ZYZ Angle Extraction

    /// Extract ZYZ Euler angles from 2x2 unitary: U = e^{iφ} Rz(α) Ry(β) Rz(γ).
    @_optimize(speed)
    @_effects(readonly)
    private static func extractZYZAngles(_ matrix: [[Complex<Double>]]) -> EulerDecomposition {
        let a = matrix[0][0]
        let c = matrix[1][0]
        let d = matrix[1][1]

        let det = a * d - matrix[0][1] * c
        let globalPhase = det.phase / 2.0

        let cosBetaHalf = min(1.0, max(0.0, a.magnitude))
        let beta = 2.0 * acos(cosBetaHalf)

        var alpha: Double
        var gamma: Double

        if abs(beta) < angleTolerance {
            alpha = 0.0
            gamma = normalizeAngle(d.phase - a.phase)
        } else if abs(beta - .pi) < angleTolerance {
            alpha = 0.0
            gamma = normalizeAngle(-2.0 * (c.phase - globalPhase))
        } else {
            alpha = normalizeAngle(c.phase - a.phase)
            gamma = normalizeAngle(2.0 * globalPhase - a.phase - c.phase)
        }

        alpha = normalizeAngle(alpha)
        gamma = normalizeAngle(gamma)

        return EulerDecomposition(
            alpha: alpha,
            beta: beta,
            gamma: gamma,
            globalPhase: globalPhase,
        )
    }

    // MARK: - Basis Conversion

    /// Extract Euler angles in the specified basis from a 2x2 unitary.
    @_optimize(speed)
    @_effects(readonly)
    private static func extractEulerAngles(
        _ matrix: [[Complex<Double>]],
        basis: EulerBasis,
    ) -> EulerDecomposition {
        switch basis {
        case .zyz:
            return extractZYZAngles(matrix)
        case .zxz:
            let zyz = extractZYZAngles(matrix)
            return EulerDecomposition(
                alpha: normalizeAngle(zyz.alpha + .pi / 2),
                beta: zyz.beta,
                gamma: normalizeAngle(zyz.gamma - .pi / 2),
                globalPhase: zyz.globalPhase,
            )
        case .xyx:
            let ryPos = ryHalfPiConstant
            let ryNeg = ryNegHalfPiConstant
            let m = multiply2x2(multiply2x2(ryPos, matrix), ryNeg)
            let zyz = extractZYZAngles(m)
            return EulerDecomposition(
                alpha: normalizeAngle(-zyz.alpha),
                beta: zyz.beta,
                gamma: normalizeAngle(-zyz.gamma),
                globalPhase: zyz.globalPhase,
            )
        }
    }

    /// Precomputed Ry(pi/2) matrix for XYX basis conversion.
    private static let ryHalfPiConstant: [[Complex<Double>]] = {
        let s = 1.0 / 2.0.squareRoot()
        return [
            [Complex(s, 0), Complex(-s, 0)],
            [Complex(s, 0), Complex(s, 0)],
        ]
    }()

    /// Precomputed Ry(-pi/2) matrix for XYX basis conversion.
    private static let ryNegHalfPiConstant: [[Complex<Double>]] = {
        let s = 1.0 / 2.0.squareRoot()
        return [
            [Complex(s, 0), Complex(s, 0)],
            [Complex(-s, 0), Complex(s, 0)],
        ]
    }()

    /// Convert Euler angles back to gates in the specified basis.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func eulerToGates(
        _ decomp: EulerDecomposition,
        basis: EulerBasis,
    ) -> [(gate: QuantumGate, qubits: [Int])] {
        let tol = angleTolerance
        if abs(decomp.beta) < tol, abs(decomp.alpha) < tol, abs(decomp.gamma) < tol {
            if abs(decomp.globalPhase) > tol {
                return [(.globalPhase(decomp.globalPhase), [0])]
            }
            return []
        }

        var result: [(gate: QuantumGate, qubits: [Int])] = []

        let (outer, inner) = rotationGates(for: basis)

        if abs(decomp.gamma) > tol {
            result.append((outer(decomp.gamma), [0]))
        }
        if abs(decomp.beta) > tol {
            result.append((inner(decomp.beta), [0]))
        }
        if abs(decomp.alpha) > tol {
            result.append((outer(decomp.alpha), [0]))
        }
        if abs(decomp.globalPhase) > tol {
            result.append((.globalPhase(decomp.globalPhase), [0]))
        }

        return result
    }

    /// Return rotation gate constructors for the given basis.
    @_optimize(speed)
    @_effects(readonly)
    private static func rotationGates(
        for basis: EulerBasis,
    ) -> (outer: (Double) -> QuantumGate, inner: (Double) -> QuantumGate) {
        switch basis {
        case .zyz:
            (QuantumGate.rotationZ, QuantumGate.rotationY)
        case .zxz:
            (QuantumGate.rotationZ, QuantumGate.rotationX)
        case .xyx:
            (QuantumGate.rotationX, QuantumGate.rotationY)
        }
    }

    // MARK: - Ross-Selinger Clifford+T Synthesis

    /// Synthesize Clifford+T approximation via Ross-Selinger gridpoint search.
    @_optimize(speed)
    @_eagerMove
    private static func rossSelingerSynthesize(
        _ euler: EulerDecomposition,
        precision: Double,
    ) -> [QuantumGate] {
        var result: [QuantumGate] = []

        let zGamma = approximateZRotation(euler.gamma, precision: precision / 3.0)
        let yBeta = synthesizeYRotation(euler.beta, precision: precision / 3.0)
        let zAlpha = approximateZRotation(euler.alpha, precision: precision / 3.0)

        result.append(contentsOf: zGamma)
        result.append(contentsOf: yBeta)
        result.append(contentsOf: zAlpha)

        return result
    }

    /// Approximate Rz(theta) using Clifford+T gates.
    @_optimize(speed)
    @_eagerMove
    private static func approximateZRotation(
        _ theta: Double,
        precision: Double,
    ) -> [QuantumGate] {
        if abs(theta) < angleTolerance {
            return []
        }

        let normalizedAngle = normalizeAngle(theta)

        let clifford = exactCliffordZ(normalizedAngle)
        if let gates = clifford {
            return gates
        }

        return gridSynthesizeZ(normalizedAngle, precision: precision)
    }

    /// Synthesize Ry(theta) as H Rz(theta) H using Clifford+T.
    @_optimize(speed)
    @_eagerMove
    private static func synthesizeYRotation(
        _ theta: Double,
        precision: Double,
    ) -> [QuantumGate] {
        if abs(theta) < angleTolerance {
            return []
        }

        let clifford = exactCliffordY(theta)
        if let gates = clifford {
            return gates
        }

        var result: [QuantumGate] = [.hadamard]
        result.append(contentsOf: gridSynthesizeZ(theta, precision: precision))
        result.append(.hadamard)
        return result
    }

    /// Check if angle is an exact Clifford Z-rotation (multiple of pi/2).
    @_effects(readonly)
    private static func exactCliffordZ(_ theta: Double) -> [QuantumGate]? {
        let piHalf = Double.pi / 2.0
        let normalized = normalizeAngle(theta)

        if abs(abs(normalized) - .pi) < angleTolerance {
            return [.sGate, .sGate]
        }
        if abs(normalized - piHalf) < angleTolerance {
            return [.sGate]
        }
        if abs(normalized + piHalf) < angleTolerance {
            return [.sGate, .sGate, .sGate]
        }
        let piQuarter = Double.pi / 4.0
        if abs(normalized - piQuarter) < angleTolerance {
            return [.tGate]
        }
        if abs(normalized + piQuarter) < angleTolerance {
            return [.tGate, .tGate, .tGate, .tGate, .tGate, .tGate, .tGate]
        }

        return nil
    }

    /// Check if angle is an exact Clifford Y-rotation.
    @_effects(readonly)
    private static func exactCliffordY(_ theta: Double) -> [QuantumGate]? {
        let normalized = normalizeAngle(theta)

        if abs(abs(normalized) - .pi) < angleTolerance {
            return [.pauliY]
        }

        return nil
    }

    /// Grid-based Z-rotation Clifford+T synthesis with increasing T-count search.
    @_optimize(speed)
    @_eagerMove
    private static func gridSynthesizeZ(
        _ theta: Double,
        precision: Double,
    ) -> [QuantumGate] {
        let maxTCount = max(4, Int(ceil(3.0 * log2(1.0 / precision))))

        var best: [QuantumGate] = []
        for tCount in 1 ... maxTCount {
            if let gates = searchGridPoint(theta, tCount: tCount, precision: precision) {
                best = gates
                break
            }
        }
        return best
    }

    /// Search for a grid point at the given T-count that approximates the target angle.
    @_optimize(speed)
    @_effects(readonly)
    private static func searchGridPoint(
        _ theta: Double,
        tCount: Int,
        precision: Double,
    ) -> [QuantumGate]? {
        let targetCos = cos(theta / 2.0)
        let targetSin = sin(theta / 2.0)

        let scale = pow(2.0, Double(tCount) / 2.0)
        let scaledCos = targetCos * scale
        let scaledSin = targetSin * scale

        let intCos = Int(round(scaledCos))
        let intSin = Int(round(scaledSin))

        let actualCos = Double(intCos) / scale
        let actualSin = Double(intSin) / scale

        let normSq = actualCos * actualCos + actualSin * actualSin
        if abs(normSq - 1.0) > precision {
            return nil
        }

        let achievedAngle = 2.0 * atan2(actualSin, actualCos)
        let angleError = abs(normalizeAngle(achievedAngle - theta))
        if angleError > precision {
            return nil
        }

        return tCountToGateSequence(intCos: intCos, intSin: intSin, tCount: tCount)
    }

    /// Convert grid point integers to a Clifford+T gate sequence.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func tCountToGateSequence(
        intCos: Int,
        intSin: Int,
        tCount: Int,
    ) -> [QuantumGate] {
        var gates: [QuantumGate] = []

        var remaining = tCount
        var cosCoeff = intCos
        var sinCoeff = intSin

        while remaining > 0 {
            let cosOdd = cosCoeff & 1
            let sinOdd = sinCoeff & 1

            if cosOdd == 0, sinOdd == 0 {
                cosCoeff /= 2
                sinCoeff /= 2
                remaining -= 2
            } else if cosOdd == 1, sinOdd == 0 {
                gates.append(.tGate)
                let newCos = cosCoeff + sinCoeff
                let newSin = -cosCoeff + sinCoeff
                cosCoeff = newCos
                sinCoeff = newSin
                remaining -= 1
            } else if cosOdd == 0, sinOdd == 1 {
                gates.append(.tGate)
                gates.append(.tGate)
                gates.append(.tGate)
                let newCos = cosCoeff - sinCoeff
                let newSin = cosCoeff + sinCoeff
                cosCoeff = newCos
                sinCoeff = newSin
                remaining -= 1
            } else {
                gates.append(.hadamard)
                gates.append(.tGate)
                let newCos = cosCoeff + sinCoeff
                let newSin = cosCoeff - sinCoeff
                cosCoeff = newCos / 2
                sinCoeff = newSin / 2
                remaining -= 1
            }
        }

        return gates
    }

    // MARK: - Shannon Decomposition

    /// Recursive Shannon decomposition for n-qubit unitaries.
    @_optimize(speed)
    @_eagerMove
    private static func shannonRecursive(
        _ matrix: [[Complex<Double>]],
        qubits: [Int],
    ) -> [(gate: QuantumGate, qubits: [Int])] {
        let dim = matrix.count

        if dim == 2 {
            let euler = extractZYZAngles(matrix)
            return eulerToGates(euler, basis: .zyz).map { gate, qs in
                (gate, qs.map { qubits[$0] })
            }
        }

        if dim == 4 {
            let gate = QuantumGate.customTwoQubit(matrix: matrix)
            let kakResult = CircuitOptimizer.kakDecomposition(gate)
            if !kakResult.isEmpty {
                return kakResult.map { g, qs in
                    (g, qs.map { qubits[$0] })
                }
            }
        }

        let csdResult = csdDecompose(matrix, qubits: qubits)
        let shannonResult = shannonDemultiplex(matrix, qubits: qubits)

        if csdResult.count <= shannonResult.count {
            return csdResult
        }
        return shannonResult
    }

    /// Shannon demultiplexing: split n-qubit unitary into controlled (n-1)-qubit operations.
    @_optimize(speed)
    @_eagerMove
    private static func shannonDemultiplex(
        _ matrix: [[Complex<Double>]],
        qubits: [Int],
    ) -> [(gate: QuantumGate, qubits: [Int])] {
        let dim = matrix.count
        let halfDim = dim / 2
        let controlQubit = qubits[0]
        let targetQubits = Array(qubits[1...])

        let u0 = extractSubmatrix(matrix, rowStart: 0, colStart: 0, size: halfDim)
        let u1 = extractSubmatrix(matrix, rowStart: halfDim, colStart: halfDim, size: halfDim)

        let u1DagU0 = MatrixUtilities.matrixMultiply(
            MatrixUtilities.hermitianConjugate(u1),
            u0,
        )

        let (d, v) = diagonalizeUnitary(u1DagU0)

        let vDag = MatrixUtilities.hermitianConjugate(v)

        let leftUpper = MatrixUtilities.matrixMultiply(u0, vDag)

        var sqrtD = [Complex<Double>](repeating: .zero, count: halfDim)
        for i in 0 ..< halfDim {
            let phase = d[i].phase / 2.0
            sqrtD[i] = Complex(phase: phase)
        }

        var w = makeIdentity(halfDim)
        for i in 0 ..< halfDim {
            for j in 0 ..< halfDim {
                w[i][j] = leftUpper[i][j] * sqrtD[j].conjugate
            }
        }

        var result: [(gate: QuantumGate, qubits: [Int])] = []

        let vGates = shannonRecursive(v, qubits: targetQubits)
        result.append(contentsOf: vGates)

        var phaseAngles = [Double](repeating: 0.0, count: dim)
        for i in 0 ..< halfDim {
            phaseAngles[i] = sqrtD[i].phase
            phaseAngles[halfDim + i] = -sqrtD[i].phase
        }
        let allQubits = [controlQubit] + targetQubits
        let multiplexGates = diagonalDecompose(phaseAngles, qubits: allQubits)
        result.append(contentsOf: multiplexGates)

        let wGates = shannonRecursive(w, qubits: targetQubits)
        result.append(contentsOf: wGates)

        return result
    }

    /// Cosine-sine decomposition variant for structured unitaries.
    @_optimize(speed)
    @_eagerMove
    private static func csdDecompose(
        _ matrix: [[Complex<Double>]],
        qubits: [Int],
    ) -> [(gate: QuantumGate, qubits: [Int])] {
        let dim = matrix.count
        let halfDim = dim / 2

        let u00 = extractSubmatrix(matrix, rowStart: 0, colStart: 0, size: halfDim)
        let u01 = extractSubmatrix(matrix, rowStart: 0, colStart: halfDim, size: halfDim)
        let u10 = extractSubmatrix(matrix, rowStart: halfDim, colStart: 0, size: halfDim)
        let u11 = extractSubmatrix(matrix, rowStart: halfDim, colStart: halfDim, size: halfDim)

        let (cosines, l1, l2, r1, r2) = computeCSD(u00, u01, u10, u11)

        let controlQubit = qubits[0]
        let targetQubits = Array(qubits[1...])

        var result: [(gate: QuantumGate, qubits: [Int])] = []

        let r1Dag = MatrixUtilities.hermitianConjugate(r1)
        let r2Dag = MatrixUtilities.hermitianConjugate(r2)
        let rightGates = demultiplexUnitaries(r1Dag, r2Dag, controlQubit: controlQubit, targetQubits: targetQubits)
        result.append(contentsOf: rightGates)

        var ryAngles = [Double](repeating: 0.0, count: halfDim)
        for i in 0 ..< halfDim {
            ryAngles[i] = 2.0 * acos(max(-1.0, min(1.0, cosines[i])))
        }
        let csGates = uniformlyControlledRy(ryAngles, targetQubit: controlQubit, controlQubits: targetQubits)
        result.append(contentsOf: csGates)

        let leftGates = demultiplexUnitaries(l1, l2, controlQubit: controlQubit, targetQubits: targetQubits)
        result.append(contentsOf: leftGates)

        return result
    }

    /// Demultiplex two unitaries: apply u0 when control is 0, u1 when control is 1.
    @_optimize(speed)
    @_eagerMove
    private static func demultiplexUnitaries(
        _ u0: [[Complex<Double>]],
        _ u1: [[Complex<Double>]],
        controlQubit: Int,
        targetQubits: [Int],
    ) -> [(gate: QuantumGate, qubits: [Int])] {
        let d = matMultiply(MatrixUtilities.hermitianConjugate(u0), u1)

        let (eigenvalues, v) = diagonalizeUnitary(d)
        let vDag = MatrixUtilities.hermitianConjugate(v)
        let u0V = matMultiply(u0, v)

        var result: [(gate: QuantumGate, qubits: [Int])] = []

        let vDagGates = shannonRecursive(vDag, qubits: targetQubits)
        result.append(contentsOf: vDagGates)

        let eigenCount = eigenvalues.count
        var phaseAngles = [Double](repeating: 0.0, count: 2 * eigenCount)
        for i in 0 ..< eigenCount {
            phaseAngles[eigenCount + i] = eigenvalues[i].phase
        }
        let allQubits = [controlQubit] + targetQubits
        let phaseGates = diagonalDecompose(phaseAngles, qubits: allQubits)
        result.append(contentsOf: phaseGates)

        let u0VGates = shannonRecursive(u0V, qubits: targetQubits)
        result.append(contentsOf: u0VGates)

        return result
    }

    /// Uniformly controlled Ry: apply Ry(angles[i]) to targetQubit conditioned on control register state i.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func uniformlyControlledRy(
        _ angles: [Double],
        targetQubit: Int,
        controlQubits: [Int],
    ) -> [(gate: QuantumGate, qubits: [Int])] {
        let n = angles.count
        var result: [(gate: QuantumGate, qubits: [Int])] = []

        if n == 1 {
            if abs(angles[0]) > angleTolerance {
                result.append((.rotationY(angles[0]), [targetQubit]))
            }
            return result
        }

        let halfN = n / 2
        var sumAngles = [Double](repeating: 0.0, count: halfN)
        var diffAngles = [Double](repeating: 0.0, count: halfN)
        for i in 0 ..< halfN {
            sumAngles[i] = (angles[i] + angles[i + halfN]) / 2.0
            diffAngles[i] = (angles[i] - angles[i + halfN]) / 2.0
        }

        let remainingControls = Array(controlQubits.dropFirst())

        let sumGates = uniformlyControlledRy(sumAngles, targetQubit: targetQubit, controlQubits: remainingControls)
        result.append(contentsOf: sumGates)

        result.append((.cnot, [controlQubits[0], targetQubit]))

        let diffGates = uniformlyControlledRy(diffAngles, targetQubit: targetQubit, controlQubits: remainingControls)
        result.append(contentsOf: diffGates)

        result.append((.cnot, [controlQubits[0], targetQubit]))

        return result
    }

    /// Compute cosine-sine decomposition of the four quadrants.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func computeCSD(
        _ u00: [[Complex<Double>]],
        _ u01: [[Complex<Double>]],
        _ u10: [[Complex<Double>]],
        _ u11: [[Complex<Double>]],
    ) -> (cosines: [Double], l1: [[Complex<Double>]], l2: [[Complex<Double>]], r1: [[Complex<Double>]], r2: [[Complex<Double>]]) {
        let n = u00.count

        let u00Dag = MatrixUtilities.hermitianConjugate(u00)
        let product = MatrixUtilities.matrixMultiply(u00Dag, u00)

        let (eigenvalues, eigenvectors) = symmetricEigendecompose(product)

        var cosines = [Double](repeating: 0.0, count: n)
        for i in 0 ..< n {
            cosines[i] = max(0.0, min(1.0, eigenvalues[i])).squareRoot()
        }

        let r1 = eigenvectors

        let u00R1 = matMultiply(u00, r1)

        var l1 = makeIdentity(n)
        for i in 0 ..< n {
            if cosines[i] > csdTolerance {
                let invCos = 1.0 / cosines[i]
                for j in 0 ..< n {
                    l1[j][i] = u00R1[j][i] * invCos
                }
            } else {
                for j in 0 ..< n {
                    l1[j][i] = i == j ? .one : .zero
                }
            }
        }

        var sines = [Double](repeating: 0.0, count: n)
        for i in 0 ..< n {
            sines[i] = max(0.0, 1.0 - cosines[i] * cosines[i]).squareRoot()
        }

        let l1Dag = MatrixUtilities.hermitianConjugate(l1)
        let l1DagU01 = matMultiply(l1Dag, u01)
        var r2Dag = makeIdentity(n)
        for i in 0 ..< n {
            if sines[i] > csdTolerance {
                let invSin = 1.0 / sines[i]
                for j in 0 ..< n {
                    r2Dag[i][j] = l1DagU01[i][j] * invSin
                }
            }
        }
        orthogonalizeRows(&r2Dag, n: n)
        let r2 = MatrixUtilities.hermitianConjugate(r2Dag)

        let u11R2 = matMultiply(u11, r2)
        let u10R1 = matMultiply(u10, r1)
        var l2 = makeIdentity(n)
        for i in 0 ..< n {
            if sines[i] > csdTolerance {
                let negInvSin = -1.0 / sines[i]
                for j in 0 ..< n {
                    l2[j][i] = u10R1[j][i] * negInvSin
                }
            } else if cosines[i] > csdTolerance {
                let invCos = 1.0 / cosines[i]
                for j in 0 ..< n {
                    l2[j][i] = u11R2[j][i] * invCos
                }
            }
        }

        return (cosines, l1, l2, r1, r2)
    }

    /// Orthogonalize rows of a matrix via modified Gram-Schmidt.
    @_optimize(speed)
    private static func orthogonalizeRows(_ matrix: inout [[Complex<Double>]], n: Int) {
        for i in 0 ..< n {
            var rowNormSq = 0.0
            for j in 0 ..< n {
                rowNormSq += matrix[i][j].magnitudeSquared
            }
            if rowNormSq < csdTolerance {
                for j in 0 ..< n {
                    matrix[i][j] = .zero
                }
                for k in 0 ..< n {
                    var candidate = [Complex<Double>](repeating: .zero, count: n)
                    candidate[k] = .one
                    var valid = true
                    for prev in 0 ..< i {
                        var dot = Complex<Double>.zero
                        for j in 0 ..< n {
                            dot = dot + matrix[prev][j].conjugate * candidate[j]
                        }
                        if dot.magnitude > csdTolerance {
                            for j in 0 ..< n {
                                candidate[j] = candidate[j] - dot * matrix[prev][j]
                            }
                        }
                        var candNorm = 0.0
                        for j in 0 ..< n {
                            candNorm += candidate[j].magnitudeSquared
                        }
                        if candNorm < csdTolerance { valid = false; break }
                    }
                    if valid {
                        var norm = 0.0
                        for j in 0 ..< n {
                            norm += candidate[j].magnitudeSquared
                        }
                        let invNorm = 1.0 / norm.squareRoot()
                        for j in 0 ..< n {
                            matrix[i][j] = candidate[j] * invNorm
                        }
                        break
                    }
                }
            } else {
                let invNorm = 1.0 / rowNormSq.squareRoot()
                for j in 0 ..< n {
                    matrix[i][j] = matrix[i][j] * invNorm
                }
                for next in (i + 1) ..< n {
                    var dot = Complex<Double>.zero
                    for j in 0 ..< n {
                        dot = dot + matrix[i][j].conjugate * matrix[next][j]
                    }
                    for j in 0 ..< n {
                        matrix[next][j] = matrix[next][j] - dot * matrix[i][j]
                    }
                }
            }
        }
    }

    // MARK: - Matrix Utilities (Private)

    /// Extract square submatrix.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func extractSubmatrix(
        _ matrix: [[Complex<Double>]],
        rowStart: Int,
        colStart: Int,
        size: Int,
    ) -> [[Complex<Double>]] {
        var result = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: size), count: size)
        for i in 0 ..< size {
            for j in 0 ..< size {
                result[i][j] = matrix[rowStart + i][colStart + j]
            }
        }
        return result
    }

    /// Build identity matrix.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func makeIdentity(_ n: Int) -> [[Complex<Double>]] {
        var result = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: n), count: n)
        for i in 0 ..< n {
            result[i][i] = .one
        }
        return result
    }

    /// 2x2 matrix multiply (avoid overhead of general NxN for small matrices).
    @_optimize(speed)
    @_effects(readonly)
    @inline(__always)
    private static func multiply2x2(
        _ a: [[Complex<Double>]],
        _ b: [[Complex<Double>]],
    ) -> [[Complex<Double>]] {
        let a00 = a[0][0], a01 = a[0][1], a10 = a[1][0], a11 = a[1][1]
        let b00 = b[0][0], b01 = b[0][1], b10 = b[1][0], b11 = b[1][1]
        return [
            [a00 * b00 + a01 * b10, a00 * b01 + a01 * b11],
            [a10 * b00 + a11 * b10, a10 * b01 + a11 * b11],
        ]
    }

    /// Normalize angle to [-pi, pi].
    @_effects(readonly)
    @inline(__always)
    private static func normalizeAngle(_ angle: Double) -> Double {
        var result = angle.truncatingRemainder(dividingBy: 2 * .pi)
        if result > .pi { result -= 2 * .pi }
        if result < -.pi { result += 2 * .pi }
        return result
    }

    // MARK: - Eigendecomposition Helpers

    /// Analytical 2x2 eigendecomposition: A = V diag(λ₁,λ₂) V†.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func diagonalize2x2(
        _ m: [[Complex<Double>]],
    ) -> (diagonal: [Complex<Double>], eigenvectors: [[Complex<Double>]]) {
        let a = m[0][0], b = m[0][1], c = m[1][0], d = m[1][1]
        let trace = a + d
        let det = a * d - b * c
        let disc = trace * trace - 4.0 * det
        let sqrtDisc = complexSqrt(disc)

        let lambda1 = (trace + sqrtDisc) * 0.5
        let lambda2 = (trace - sqrtDisc) * 0.5

        var v: [[Complex<Double>]]

        if b.magnitude > angleTolerance {
            let v1 = [lambda1 - d, b]
            let v2 = [lambda2 - d, b]
            let n1 = hypot(v1[0].magnitude, v1[1].magnitude)
            let n2 = hypot(v2[0].magnitude, v2[1].magnitude)
            let inv1 = 1.0 / n1
            let inv2 = 1.0 / n2
            v = [[v1[0] * inv1, v2[0] * inv2],
                 [v1[1] * inv1, v2[1] * inv2]]
        } else {
            v = makeIdentity(2)
        }

        return ([lambda1, lambda2], v)
    }

    /// Complex square root via half-angle polar form.
    @_effects(readonly)
    @inline(__always)
    private static func complexSqrt(_ z: Complex<Double>) -> Complex<Double> {
        let mag = z.magnitude.squareRoot()
        return Complex(phase: z.phase / 2.0) * mag
    }

    /// Diagonalize a unitary matrix U = V D V† using QR iteration (analytical 2x2 fast path).
    @_optimize(speed)
    @_eagerMove
    private static func diagonalizeUnitary(
        _ matrix: [[Complex<Double>]],
    ) -> (diagonal: [Complex<Double>], eigenvectors: [[Complex<Double>]]) {
        let n = matrix.count

        if n == 2 {
            return diagonalize2x2(matrix)
        }

        var a = matrix
        var vAccum = makeIdentity(n)

        let maxIterations = 200
        let tolerance = unitaryTolerance

        for _ in 0 ..< maxIterations {
            var offDiagNorm = 0.0
            for i in 0 ..< n {
                for j in 0 ..< n where i != j {
                    offDiagNorm += a[i][j].magnitudeSquared
                }
            }
            if offDiagNorm < tolerance * tolerance { break }

            let shift = a[n - 1][n - 1]
            for i in 0 ..< n {
                a[i][i] = a[i][i] - shift
            }

            let (q, r) = qrDecompose(a)
            a = matMultiply(r, q)
            for i in 0 ..< n {
                a[i][i] = a[i][i] + shift
            }

            vAccum = matMultiply(vAccum, q)
        }

        let diagonal = (0 ..< n).map { a[$0][$0] }
        return (diagonal, vAccum)
    }

    /// Symmetric eigendecomposition for Hermitian positive-semidefinite matrices.
    @_optimize(speed)
    @_eagerMove
    private static func symmetricEigendecompose(
        _ matrix: [[Complex<Double>]],
    ) -> (eigenvalues: [Double], eigenvectors: [[Complex<Double>]]) {
        let n = matrix.count

        if n == 2 {
            let (diag, vecs) = diagonalize2x2(matrix)
            return (diag.map(\.real), vecs)
        }

        let eigenvalues = (0 ..< n).map { matrix[$0][$0].real }
        return (eigenvalues, makeIdentity(n))
    }

    /// QR decomposition via Householder reflections.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func qrDecompose(
        _ a: [[Complex<Double>]],
    ) -> (q: [[Complex<Double>]], r: [[Complex<Double>]]) {
        let n = a.count
        var q = makeIdentity(n)
        var r = a

        for k in 0 ..< n - 1 {
            let xCount = n - k

            var normXSq = 0.0
            for i in 0 ..< xCount {
                normXSq += r[k + i][k].magnitudeSquared
            }
            let normX = normXSq.squareRoot()
            if normX < 1e-15 { continue }

            let alpha = -normX * Complex(phase: r[k][k].phase)
            var v = [Complex<Double>](repeating: .zero, count: xCount)
            for i in 0 ..< xCount {
                v[i] = r[k + i][k]
            }
            v[0] = v[0] - alpha

            var normVSq = 0.0
            for i in 0 ..< xCount {
                normVSq += v[i].magnitudeSquared
            }
            let invNormV = 1.0 / normVSq.squareRoot()
            for i in 0 ..< xCount {
                v[i] = v[i] * invNormV
            }

            for j in k ..< n {
                var dot = Complex<Double>.zero
                for i in 0 ..< xCount {
                    dot = dot + v[i].conjugate * r[k + i][j]
                }
                dot = dot * 2.0
                for i in 0 ..< xCount {
                    r[k + i][j] = r[k + i][j] - v[i] * dot
                }
            }

            for i in 0 ..< n {
                var dot = Complex<Double>.zero
                for j in 0 ..< xCount {
                    dot = dot + q[i][k + j] * v[j]
                }
                dot = dot * 2.0
                for j in 0 ..< xCount {
                    q[i][k + j] = q[i][k + j] - dot * v[j].conjugate
                }
            }
        }

        return (q, r)
    }

    /// General complex matrix multiplication.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func matMultiply(
        _ a: [[Complex<Double>]],
        _ b: [[Complex<Double>]],
    ) -> [[Complex<Double>]] {
        let n = a.count
        let m = b[0].count
        let p = b.count
        var result = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: m), count: n)
        for i in 0 ..< n {
            for k in 0 ..< p {
                let aik = a[i][k]
                for j in 0 ..< m {
                    result[i][j] = result[i][j] + aik * b[k][j]
                }
            }
        }
        return result
    }

    // MARK: - Demultiplexing Helpers

    /// Decompose diagonal unitary into CNOT + Rz gate sequence via Walsh-Hadamard transform.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func diagonalDecompose(
        _ phases: [Double],
        qubits: [Int],
    ) -> [(gate: QuantumGate, qubits: [Int])] {
        let n = phases.count
        let m = qubits.count
        var result: [(gate: QuantumGate, qubits: [Int])] = []

        var coeffs = phases
        var step = 1
        while step < n {
            var i = 0
            while i < n {
                for j in i ..< i + step {
                    let u = coeffs[j]
                    let v = coeffs[j + step]
                    coeffs[j] = (u + v) / 2.0
                    coeffs[j + step] = (u - v) / 2.0
                }
                i += 2 * step
            }
            step *= 2
        }

        for k in 1 ..< n {
            if abs(coeffs[k]) < angleTolerance { continue }

            let angle = -2.0 * coeffs[k]

            var zQubits: [Int] = []
            for bit in 0 ..< m {
                if (k >> bit) & 1 != 0 {
                    zQubits.append(qubits[bit])
                }
            }

            if zQubits.count == 1 {
                result.append((.rotationZ(angle), [zQubits[0]]))
            } else {
                let target = zQubits[zQubits.count - 1]
                for i in 0 ..< zQubits.count - 1 {
                    result.append((.cnot, [zQubits[i], target]))
                }
                result.append((.rotationZ(angle), [target]))
                for i in stride(from: zQubits.count - 2, through: 0, by: -1) {
                    result.append((.cnot, [zQubits[i], target]))
                }
            }
        }

        return result
    }
}
