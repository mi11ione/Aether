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
        /// **Example:**
        /// ```swift
        /// let decomp = EulerDecomposition(alpha: 0.5, beta: 1.2, gamma: -0.3, globalPhase: 0.1)
        /// ```
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
        let detPhaseHalf = det.phase / 2.0

        let cosBetaHalf = min(1.0, max(0.0, a.magnitude))
        let beta = 2.0 * acos(cosBetaHalf)

        var alpha: Double
        var gamma: Double

        if abs(beta) < angleTolerance {
            alpha = 0.0
            gamma = normalizeAngle(d.phase - a.phase)
        } else if abs(beta - .pi) < angleTolerance {
            alpha = 0.0
            gamma = normalizeAngle(-2.0 * (c.phase - detPhaseHalf))
        } else {
            alpha = normalizeAngle(c.phase - a.phase)
            gamma = normalizeAngle(2.0 * detPhaseHalf - a.phase - c.phase)
        }

        alpha = normalizeAngle(alpha)
        gamma = normalizeAngle(gamma)

        let globalPhase = cosBetaHalf > angleTolerance
            ? normalizeAngle(a.phase + (alpha + gamma) / 2.0)
            : normalizeAngle(c.phase - (alpha - gamma) / 2.0)

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

    /// Synthesize Clifford+T approximation via Z[ω] gridpoint search.
    @_optimize(speed)
    @_eagerMove
    private static func rossSelingerSynthesize(
        _ euler: EulerDecomposition,
        precision: Double,
    ) -> [QuantumGate] {
        var result: [QuantumGate] = []

        var nonZeroCount = 0
        if abs(euler.gamma) >= angleTolerance { nonZeroCount += 1 }
        if abs(euler.beta) >= angleTolerance { nonZeroCount += 1 }
        if abs(euler.alpha) >= angleTolerance { nonZeroCount += 1 }
        let perAnglePrecision = nonZeroCount > 0 ? precision / Double(nonZeroCount) : precision

        let zGamma = approximateZRotation(euler.gamma, precision: perAnglePrecision)
        let yBeta = synthesizeYRotation(euler.beta, precision: perAnglePrecision)
        let zAlpha = approximateZRotation(euler.alpha, precision: perAnglePrecision)

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

        let gridResult = gridSynthesizeZ(normalizedAngle, precision: precision)
        if !gridResult.isEmpty { return gridResult }

        let piQuarter = Double.pi / 4.0
        let nearestJ = Int((normalizedAngle / piQuarter).rounded())
        let nearestAngle = Double(nearestJ) * piQuarter
        if abs(normalizeAngle(nearestAngle - normalizedAngle)) <= precision {
            return tPowerGates(nearestJ)
        }

        return []
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

    /// Check if angle is an exact T-power rotation (multiple of pi/4).
    @_eagerMove
    @_effects(readonly)
    private static func exactCliffordZ(_ theta: Double) -> [QuantumGate]? {
        let normalized = normalizeAngle(theta)
        let piQuarter = Double.pi / 4.0
        let nearestJ = Int((normalized / piQuarter).rounded())
        let nearestAngle = Double(nearestJ) * piQuarter

        if abs(normalizeAngle(nearestAngle - normalized)) < angleTolerance {
            return tPowerGates(nearestJ)
        }

        return nil
    }

    /// Check if angle is an exact Clifford Y-rotation.
    @_eagerMove
    @_effects(readonly)
    private static func exactCliffordY(_ theta: Double) -> [QuantumGate]? {
        let normalized = normalizeAngle(theta)

        if abs(abs(normalized) - .pi) < angleTolerance {
            return [.pauliY]
        }

        return nil
    }

    // MARK: - Z[ω] Ring for Clifford+T Synthesis

    private struct ZOmega: Sendable, Equatable {
        let a: Int
        let b: Int
        let c: Int
        let d: Int

        static let zero = ZOmega(a: 0, b: 0, c: 0, d: 0)
        static let one = ZOmega(a: 1, b: 0, c: 0, d: 0)

        @_optimize(speed)
        @inline(__always)
        static func + (l: ZOmega, r: ZOmega) -> ZOmega {
            ZOmega(a: l.a + r.a, b: l.b + r.b, c: l.c + r.c, d: l.d + r.d)
        }

        @_optimize(speed)
        @inline(__always)
        static func - (l: ZOmega, r: ZOmega) -> ZOmega {
            ZOmega(a: l.a - r.a, b: l.b - r.b, c: l.c - r.c, d: l.d - r.d)
        }

        @_optimize(speed)
        @inline(__always)
        static prefix func - (x: ZOmega) -> ZOmega {
            ZOmega(a: -x.a, b: -x.b, c: -x.c, d: -x.d)
        }

        @_optimize(speed)
        static func * (l: ZOmega, r: ZOmega) -> ZOmega {
            ZOmega(
                a: l.a * r.a - l.b * r.d - l.c * r.c - l.d * r.b,
                b: l.a * r.b + l.b * r.a - l.c * r.d - l.d * r.c,
                c: l.a * r.c + l.b * r.b + l.c * r.a - l.d * r.d,
                d: l.a * r.d + l.b * r.c + l.c * r.b + l.d * r.a,
            )
        }

        @_optimize(speed)
        func timesOmegaPower(_ n: Int) -> ZOmega {
            switch ((n % 8) + 8) % 8 {
            case 0: self
            case 1: ZOmega(a: -d, b: a, c: b, d: c)
            case 2: ZOmega(a: -c, b: -d, c: a, d: b)
            case 3: ZOmega(a: -b, b: -c, c: -d, d: a)
            case 4: ZOmega(a: -a, b: -b, c: -c, d: -d)
            case 5: ZOmega(a: d, b: -a, c: -b, d: -c)
            case 6: ZOmega(a: c, b: d, c: -a, d: -b)
            case 7: ZOmega(a: b, b: c, c: d, d: -a)
            default: self
            }
        }

        var conjugate: ZOmega {
            ZOmega(a: a, b: -d, c: -c, d: -b)
        }

        @_optimize(speed)
        var toComplex: Complex<Double> {
            let s = 1.0 / 2.0.squareRoot()
            return Complex(Double(a) + Double(b - d) * s, Double(c) + Double(b + d) * s)
        }

        var squaredMagnitude: (int: Int, sqrt2: Int) {
            let p = self * conjugate
            return (p.a, p.b)
        }

        var isDivisibleByTwo: Bool {
            (a & 1) == 0 && (b & 1) == 0 && (c & 1) == 0 && (d & 1) == 0
        }

        var dividedByTwo: ZOmega {
            ZOmega(a: a >> 1, b: b >> 1, c: c >> 1, d: d >> 1)
        }

        var isDivisibleByEta: Bool {
            (a & 1) == (c & 1) && (b & 1) == (d & 1)
        }

        var dividedByEta: ZOmega {
            ZOmega(a: (b - d) >> 1, b: (a + c) >> 1, c: (b + d) >> 1, d: (c - a) >> 1)
        }
    }

    private struct ZOmMat: Sendable {
        var m00: ZOmega
        var m01: ZOmega
        var m10: ZOmega
        var m11: ZOmega
        var denomExp: Int

        var isAllDivisibleByTwo: Bool {
            m00.isDivisibleByTwo && m01.isDivisibleByTwo
                && m10.isDivisibleByTwo && m11.isDivisibleByTwo
        }

        var dividedByTwo: ZOmMat {
            ZOmMat(
                m00: m00.dividedByTwo, m01: m01.dividedByTwo,
                m10: m10.dividedByTwo, m11: m11.dividedByTwo,
                denomExp: denomExp - 2,
            )
        }

        func rightMultiplyTj(_ j: Int) -> ZOmMat {
            ZOmMat(
                m00: m00, m01: m01.timesOmegaPower(j),
                m10: m10, m11: m11.timesOmegaPower(j),
                denomExp: denomExp,
            )
        }

        func rightMultiplyH() -> ZOmMat {
            ZOmMat(
                m00: m00 + m01, m01: m00 - m01,
                m10: m10 + m11, m11: m10 - m11,
                denomExp: denomExp + 1,
            )
        }

        func rightMultiplyX() -> ZOmMat {
            ZOmMat(
                m00: m01, m01: m00, m10: m11, m11: m10,
                denomExp: denomExp,
            )
        }

        var isAllDivisibleByEta: Bool {
            m00.isDivisibleByEta && m01.isDivisibleByEta
                && m10.isDivisibleByEta && m11.isDivisibleByEta
        }

        var dividedByEta: ZOmMat {
            ZOmMat(
                m00: m00.dividedByEta, m01: m01.dividedByEta,
                m10: m10.dividedByEta, m11: m11.dividedByEta,
                denomExp: denomExp - 1,
            )
        }
    }

    // MARK: - Clifford+T Grid Synthesis

    /// Z[ω]-based Clifford+T synthesis via lattice enumeration and exact decomposition.
    @_optimize(speed)
    @_eagerMove
    private static func gridSynthesizeZ(
        _ theta: Double,
        precision: Double,
    ) -> [QuantumGate] {
        guard let mat = zomegaGridSearch(theta: theta, precision: precision) else {
            return []
        }
        guard let gates = exactSynthesizeCliffordT(mat) else {
            return []
        }
        if verifyCliffordTSequence(gates, target: theta, precision: precision * 2.0) {
            return gates
        }
        return []
    }

    /// Search Z[ω] lattice for a unitary approximating Rz(theta) within precision.
    @_optimize(speed)
    private static func zomegaGridSearch(
        theta: Double,
        precision: Double,
    ) -> ZOmMat? {
        let target = Complex<Double>(phase: -theta / 2.0)
        let clampedPrecision = max(precision, 1e-9)
        let minK = max(4, Int(ceil(2.0 * log2(1.0 / clampedPrecision))))
        let maxK = min(minK + 30, 62)
        let invSqrt2 = 1.0 / 2.0.squareRoot()
        let precisionSq = precision * precision

        for k in minK ... maxK {
            let scale = pow(2.0, Double(k) / 2.0)
            let zx = target.real * scale
            let zy = target.imaginary * scale
            let powerOf2 = 1 << k

            var bestDistSq = Double.infinity
            var bestU = ZOmega.zero
            var bestV = ZOmega.zero

            for a1 in -4 ... 4 {
                for a3 in -4 ... 4 {
                    let shiftX = Double(a1 - a3) * invSqrt2
                    let shiftY = Double(a1 + a3) * invSqrt2
                    let a0 = Int((zx - shiftX).rounded())
                    let a2 = Int((zy - shiftY).rounded())

                    let u = ZOmega(a: a0, b: a1, c: a2, d: a3)
                    let uc = u.toComplex
                    let dx = uc.real - zx
                    let dy = uc.imaginary - zy
                    let distSq = dx * dx + dy * dy

                    if distSq >= bestDistSq { continue }

                    let mag = u.squaredMagnitude
                    let remainInt = powerOf2 - mag.int
                    let remainSqrt2 = -mag.sqrt2
                    let remainReal = Double(remainInt) + Double(remainSqrt2) * 2.0.squareRoot()
                    let maxRemain = max(precisionSq * Double(powerOf2), 4.0)

                    if remainReal < -0.5 || remainReal > maxRemain { continue }

                    if let v = solveNormEquation(intPart: remainInt, sqrt2Part: remainSqrt2) {
                        let approxErrSq = distSq / (scale * scale)
                        if approxErrSq < precisionSq {
                            bestDistSq = distSq
                            bestU = u
                            bestV = v
                        }
                    }
                }
            }

            if bestDistSq < Double.infinity {
                return ZOmMat(
                    m00: bestU, m01: -bestV.conjugate,
                    m10: bestV, m11: bestU.conjugate,
                    denomExp: k,
                )
            }
        }
        return nil
    }

    /// Find Z[ω] element with given squared magnitude via four-square decomposition.
    @_optimize(speed)
    @_effects(readonly)
    private static func solveNormEquation(intPart n0: Int, sqrt2Part n1: Int) -> ZOmega? {
        if n0 == 0, n1 == 0 { return .zero }
        if n0 < 0 { return nil }
        let realValue = Double(n0) + Double(n1) * 2.0.squareRoot()
        if realValue < -0.5 { return nil }

        let bound = min(Int(Double(max(n0, 1)).squareRoot()) + 1, 14)
        for v0 in -bound ... bound {
            let r0 = v0 * v0
            if r0 > n0 { continue }
            for v1 in -bound ... bound {
                let r01 = r0 + v1 * v1
                if r01 > n0 { continue }
                for v2 in -bound ... bound {
                    let r012 = r01 + v2 * v2
                    if r012 > n0 { continue }
                    let v3sq = n0 - r012
                    let v3abs = Int(Double(v3sq).squareRoot().rounded())
                    if v3abs * v3abs != v3sq { continue }
                    for v3 in v3abs == 0 ? [0] : [-v3abs, v3abs] {
                        let sqrt2Coeff = v0 * (v1 - v3) + v2 * (v1 + v3)
                        if sqrt2Coeff == n1 {
                            return ZOmega(a: v0, b: v1, c: v2, d: v3)
                        }
                    }
                }
            }
        }
        return nil
    }

    /// Recursively decompose a Z[ω] unitary matrix into Clifford+T gates.
    @_optimize(speed)
    @_eagerMove
    private static func exactSynthesizeCliffordT(
        _ matrix: ZOmMat,
        depth: Int = 0,
    ) -> [QuantumGate]? {
        var m = matrix
        if depth > 200 { return nil }

        while m.isAllDivisibleByEta, m.denomExp > 0 {
            m = m.dividedByEta
        }

        if m.denomExp <= 0 {
            return identifyDenomZero(m)
        }

        for j in 0 ..< 8 {
            let afterTH = m.rightMultiplyTj(8 - j).rightMultiplyH()
            if afterTH.isAllDivisibleByTwo {
                let reduced = afterTH.dividedByTwo
                guard let rest = exactSynthesizeCliffordT(reduced, depth: depth + 1) else { continue }
                var gates = rest
                gates.append(.hadamard)
                gates.append(contentsOf: tPowerGates(j))
                return gates
            }
        }

        let swapped = m.rightMultiplyX()
        for j in 0 ..< 8 {
            let afterTH = swapped.rightMultiplyTj(8 - j).rightMultiplyH()
            if afterTH.isAllDivisibleByTwo {
                let reduced = afterTH.dividedByTwo
                guard let rest = exactSynthesizeCliffordT(reduced, depth: depth + 1) else { continue }
                var gates = rest
                gates.append(.hadamard)
                gates.append(contentsOf: tPowerGates(j))
                gates.append(.hadamard)
                gates.append(.sGate)
                gates.append(.sGate)
                gates.append(.hadamard)
                return gates
            }
        }

        return nil
    }

    /// Identify a denominator-zero Z[ω] matrix as a Clifford gate sequence.
    @_eagerMove
    @_effects(readonly)
    private static func identifyDenomZero(_ mat: ZOmMat) -> [QuantumGate]? {
        if mat.m01 == .zero, mat.m10 == .zero {
            let j = ((omegaPower(mat.m11) - omegaPower(mat.m00)) % 8 + 8) % 8
            return tPowerGates(j)
        }
        if mat.m00 == .zero, mat.m11 == .zero {
            let j = ((omegaPower(mat.m01) - omegaPower(mat.m10)) % 8 + 8) % 8
            var gates: [QuantumGate] = [.hadamard, .sGate, .sGate, .hadamard]
            gates.append(contentsOf: tPowerGates(j))
            return gates
        }
        return nil
    }

    /// Determine which power of omega (0-7) equals the given Z[ω] unit.
    @_effects(readonly)
    private static func omegaPower(_ u: ZOmega) -> Int {
        for n in 0 ..< 8 {
            if ZOmega.one.timesOmegaPower(n) == u { return n }
        }
        return 0
    }

    /// Map T-gate power index (mod 8) to the corresponding gate sequence.
    @_eagerMove
    @_effects(readonly)
    private static func tPowerGates(_ j: Int) -> [QuantumGate] {
        switch ((j % 8) + 8) % 8 {
        case 0: []
        case 1: [.tGate]
        case 2: [.sGate]
        case 3: [.sGate, .tGate]
        case 4: [.sGate, .sGate]
        case 5: [.sGate, .sGate, .tGate]
        case 6: [.sGate, .sGate, .sGate]
        case 7: [.sGate, .sGate, .sGate, .tGate]
        default: []
        }
    }

    /// Verify a Clifford+T gate sequence approximates Rz(theta) within precision.
    @_optimize(speed)
    @_effects(readonly)
    private static func verifyCliffordTSequence(
        _ gates: [QuantumGate],
        target theta: Double,
        precision: Double,
    ) -> Bool {
        if gates.isEmpty { return false }
        var product = makeIdentity(2)
        for gate in gates {
            product = multiply2x2(product, gate.matrix())
        }
        let rz = QuantumGate.rotationZ(theta).matrix()
        let adj: [[Complex<Double>]] = [
            [product[0][0].conjugate, product[1][0].conjugate],
            [product[0][1].conjugate, product[1][1].conjugate],
        ]
        let check = multiply2x2(adj, rz)
        let trace = check[0][0] + check[1][1]
        return trace.magnitude > 2.0 - 2.0 * precision
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

        let sqrtD = [Complex<Double>](unsafeUninitializedCapacity: halfDim) { buffer, count in
            for i in 0 ..< halfDim {
                buffer.initializeElement(at: i, to: Complex(phase: d[i].phase / 2.0))
            }
            count = halfDim
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

        let phaseAngles = [Double](unsafeUninitializedCapacity: dim) { buffer, count in
            for i in 0 ..< halfDim {
                buffer.initializeElement(at: i, to: sqrtD[i].phase)
                buffer.initializeElement(at: halfDim + i, to: -sqrtD[i].phase)
            }
            count = dim
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

        let ryAngles = [Double](unsafeUninitializedCapacity: halfDim) { buffer, count in
            for i in 0 ..< halfDim {
                buffer.initializeElement(at: i, to: 2.0 * acos(max(-1.0, min(1.0, cosines[i]))))
            }
            count = halfDim
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
        let phaseAngles = [Double](unsafeUninitializedCapacity: 2 * eigenCount) { buffer, count in
            buffer.initialize(repeating: 0.0)
            for i in 0 ..< eigenCount {
                buffer[eigenCount + i] = eigenvalues[i].phase
            }
            count = 2 * eigenCount
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
        let sumAngles = [Double](unsafeUninitializedCapacity: halfN) { buffer, count in
            for i in 0 ..< halfN {
                buffer.initializeElement(at: i, to: (angles[i] + angles[i + halfN]) / 2.0)
            }
            count = halfN
        }
        let diffAngles = [Double](unsafeUninitializedCapacity: halfN) { buffer, count in
            for i in 0 ..< halfN {
                buffer.initializeElement(at: i, to: (angles[i] - angles[i + halfN]) / 2.0)
            }
            count = halfN
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

        let cosines = [Double](unsafeUninitializedCapacity: n) { buffer, count in
            for i in 0 ..< n {
                buffer.initializeElement(at: i, to: max(0.0, min(1.0, eigenvalues[i])).squareRoot())
            }
            count = n
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

        let sines = [Double](unsafeUninitializedCapacity: n) { buffer, count in
            for i in 0 ..< n {
                buffer.initializeElement(at: i, to: max(0.0, 1.0 - cosines[i] * cosines[i]).squareRoot())
            }
            count = n
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
                    var candidate = [Complex<Double>](unsafeUninitializedCapacity: n) {
                        buffer, count in
                        buffer.initialize(repeating: .zero)
                        count = n
                    }
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
        (0 ..< size).map { i in
            [Complex<Double>](unsafeUninitializedCapacity: size) { buffer, count in
                for j in 0 ..< size {
                    buffer.initializeElement(at: j, to: matrix[rowStart + i][colStart + j])
                }
                count = size
            }
        }
    }

    /// Build identity matrix.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func makeIdentity(_ n: Int) -> [[Complex<Double>]] {
        (0 ..< n).map { i in
            [Complex<Double>](unsafeUninitializedCapacity: n) { buffer, count in
                buffer.initialize(repeating: .zero)
                buffer[i] = .one
                count = n
            }
        }
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

    /// Diagonalize a unitary matrix U = V D V† via shifted QR with Wilkinson shift.
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

            let trSub = a[n - 2][n - 2] + a[n - 1][n - 1]
            let detSub = a[n - 2][n - 2] * a[n - 1][n - 1] - a[n - 2][n - 1] * a[n - 1][n - 2]
            let disc = trSub * trSub - 4.0 * detSub
            let sqrtDisc = complexSqrt(disc)
            let lam1 = (trSub + sqrtDisc) * 0.5
            let lam2 = (trSub - sqrtDisc) * 0.5
            let corner = a[n - 1][n - 1]
            let shift = (lam1 - corner).magnitude < (lam2 - corner).magnitude ? lam1 : lam2

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

    /// Hermitian eigendecomposition via LAPACK zheev (analytical 2x2 fast path).
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

        let result = HermitianEigenDecomposition.decompose(matrix: matrix)

        let eigenvectorMatrix = [[Complex<Double>]](unsafeUninitializedCapacity: n) {
            buffer, count in
            for row in 0 ..< n {
                buffer[row] = [Complex<Double>](unsafeUninitializedCapacity: n) {
                    rowBuffer, rowCount in
                    for col in 0 ..< n {
                        rowBuffer[col] = result.eigenvectors[col][row]
                    }
                    rowCount = n
                }
            }
            count = n
        }

        return (result.eigenvalues, eigenvectorMatrix)
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
            var v = [Complex<Double>](unsafeUninitializedCapacity: xCount) {
                buffer, count in
                for i in 0 ..< xCount {
                    buffer.initializeElement(at: i, to: r[k + i][k])
                }
                count = xCount
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
        var result = (0 ..< n).map { _ in
            [Complex<Double>](unsafeUninitializedCapacity: m) { buffer, count in
                buffer.initialize(repeating: .zero)
                count = m
            }
        }
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
