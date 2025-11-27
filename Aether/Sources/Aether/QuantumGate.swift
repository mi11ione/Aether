// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Precomputed 1/√2 for Hadamard and related gates
private let invSqrt2: Double = 1.0 / 2.0.squareRoot()

/// Quantum gates: unitary transformations for quantum circuits
///
/// Defines all supported quantum gates with their matrix representations and metadata.
/// Gates are the fundamental building blocks of quantum computation, implementing
/// reversible unitary transformations that preserve quantum state normalization.
///
/// **Mathematical foundation**:
/// - Unitarity: U†U = I (probability conservation, reversibility)
/// - Single-qubit: 2x2 complex matrices operating on C²
/// - Two-qubit: 4x4 complex matrices operating on C⁴
/// - Multi-qubit: 2^n x 2^n matrices operating on C^(2^n)
///
/// **Gate categories**:
/// - **Pauli gates**: X (bit flip), Y (bit+phase flip), Z (phase flip)
/// - **Hadamard**: Creates superposition (basis rotation)
/// - **Phase gates**: P(θ), S (π/2), T (π/4), U1(λ)
/// - **Rotation gates**: Rx(θ), Ry(θ), Rz(θ) - parameterized rotations
/// - **IBM gates**: U1, U2, U3 (universal single-qubit gates)
/// - **Controlled gates**: CNOT, CZ, CY, CH, controlled rotations
/// - **Multi-qubit**: SWAP, √SWAP, Toffoli (CCNOT)
/// - **Custom gates**: User-defined unitaries with validation
///
/// **Usage patterns**:
/// - Single-qubit: Apply to specific qubit in circuit
/// - Controlled gates: Specify control and target qubits
/// - Matrix access: Use ``matrix()`` to get unitary representation
/// - Validation: Gates validate qubit indices and matrix unitarity
///
/// Example:
/// ```swift
/// var circuit = QuantumCircuit(numQubits: 2)
/// circuit.append(gate: .hadamard, qubit: 0)
/// circuit.append(gate: .cnot, qubits: [0, 1])
/// let bellState = circuit.execute()
/// ```
///
/// - SeeAlso: ``QuantumCircuit``, ``GateApplication``, ``ParameterizedGate``
public enum QuantumGate: Equatable, Hashable, CustomStringConvertible, Sendable {
    // MARK: - Single-Qubit Gates

    case identity
    case pauliX
    case pauliY
    case pauliZ
    case hadamard
    case phase(angle: Double)
    case sGate
    case tGate
    case rotationX(theta: Double)
    case rotationY(theta: Double)
    case rotationZ(theta: Double)
    case u1(lambda: Double)
    case u2(phi: Double, lambda: Double)
    case u3(theta: Double, phi: Double, lambda: Double)
    case sx
    case sy
    case customSingleQubit(matrix: [[Complex<Double>]])

    // MARK: - Two-Qubit Gates

    case cnot
    case cz
    case cy
    case ch
    case controlledPhase(theta: Double)
    case controlledRotationX(theta: Double)
    case controlledRotationY(theta: Double)
    case controlledRotationZ(theta: Double)
    case swap
    case sqrtSwap
    case customTwoQubit(matrix: [[Complex<Double>]])

    // MARK: - Multi-Qubit Gates

    case toffoli

    // MARK: - Gate Properties

    /// Number of qubits this gate operates on
    ///
    /// Single-qubit gates return 1, two-qubit gates return 2, Toffoli returns 3.
    /// Used for circuit validation and gate application.
    ///
    /// Example:
    /// ```swift
    /// QuantumGate.hadamard.qubitsRequired  // 1
    /// QuantumGate.cnot.qubitsRequired  // 2
    /// QuantumGate.toffoli.qubitsRequired  // 3
    /// ```
    @inlinable
    public var qubitsRequired: Int {
        switch self {
        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard,
             .phase, .sGate, .tGate, .rotationX, .rotationY, .rotationZ,
             .u1, .u2, .u3, .sx, .sy, .customSingleQubit: 1
        case .cnot, .cz, .cy, .ch, .controlledPhase,
             .controlledRotationX, .controlledRotationY, .controlledRotationZ,
             .swap, .sqrtSwap, .customTwoQubit: 2
        case .toffoli: 3
        }
    }

    /// Whether gate has parameter(s) that affect its matrix
    ///
    /// Parameterized gates include rotation gates (Rx, Ry, Rz), phase gates,
    /// IBM universal gates (U1, U2, U3), and controlled rotations.
    /// Used for gradient computation and circuit optimization.
    ///
    /// Example:
    /// ```swift
    /// QuantumGate.hadamard.isParameterized  // false
    /// QuantumGate.rotationY(theta: .pi/4).isParameterized  // true
    /// QuantumGate.controlledPhase(theta: .pi).isParameterized  // true
    /// ```
    @inlinable
    public var isParameterized: Bool {
        switch self {
        case .phase, .rotationX, .rotationY, .rotationZ, .u1, .u2, .u3,
             .controlledPhase, .controlledRotationX, .controlledRotationY, .controlledRotationZ: true
        default: false
        }
    }

    /// Whether gate is Hermitian (self-adjoint): U† = U
    ///
    /// Hermitian gates are their own inverse and represent observables in quantum mechanics.
    /// Pauli gates (X, Y, Z), Hadamard, and SWAP are Hermitian.
    ///
    /// Example:
    /// ```swift
    /// QuantumGate.pauliX.isHermitian  // true (X† = X)
    /// QuantumGate.hadamard.isHermitian  // true (H† = H)
    /// QuantumGate.tGate.isHermitian  // false (T† ≠ T)
    /// ```
    @inlinable
    public var isHermitian: Bool {
        switch self {
        case .pauliX, .pauliY, .pauliZ, .hadamard, .swap: true
        default: false
        }
    }

    // MARK: - Matrix Generation

    /// Generate unitary matrix representation of gate
    ///
    /// Returns the complex matrix that represents this gate's action on quantum states.
    /// Matrix size depends on gate type: 2x2 for single-qubit, 4x4 for two-qubit,
    /// 8x8 for three-qubit gates. All matrices satisfy U†U = I (unitarity).
    ///
    /// - Returns: 2D array of complex numbers (2x2, 4x4, or 8x8)
    /// - Complexity: O(1) for fixed-size matrices (2x2, 4x4, 8x8)
    ///
    /// Example:
    /// ```swift
    /// let h = QuantumGate.hadamard
    /// let matrix = h.matrix()
    /// // [[0.707+0i, 0.707+0i], [0.707+0i, -0.707+0i]]
    /// ```
    ///
    /// - SeeAlso: ``MatrixUtilities``, ``isUnitary(_:)``
    @_optimize(speed)
    @_eagerMove
    public func matrix() -> [[Complex<Double>]] {
        switch self {
        case .identity: identityMatrix()
        case .pauliX: pauliXMatrix()
        case .pauliY: pauliYMatrix()
        case .pauliZ: pauliZMatrix()
        case .hadamard: hadamardMatrix()
        case .sGate: phaseMatrix(theta: .pi / 2.0)
        case .tGate: phaseMatrix(theta: .pi / 4.0)
        case .sx: sxMatrix()
        case .sy: syMatrix()
        case .cnot: cnotMatrix()
        case .cz: czMatrix()
        case .cy: cyMatrix()
        case .ch: chMatrix()
        case .swap: swapMatrix()
        case .sqrtSwap: sqrtSwapMatrix()
        case .toffoli: toffoliMatrix()
        case let .phase(angle): phaseMatrix(theta: angle)
        case let .rotationX(theta): rotationXMatrix(theta: theta)
        case let .rotationY(theta): rotationYMatrix(theta: theta)
        case let .rotationZ(theta): rotationZMatrix(theta: theta)
        case let .u1(lambda): u1Matrix(lambda: lambda)
        case let .u2(phi, lambda): u2Matrix(phi: phi, lambda: lambda)
        case let .u3(theta, phi, lambda): u3Matrix(theta: theta, phi: phi, lambda: lambda)
        case let .controlledPhase(theta): controlledPhaseMatrix(theta: theta)
        case let .controlledRotationX(theta): controlledRotationXMatrix(theta: theta)
        case let .controlledRotationY(theta): controlledRotationYMatrix(theta: theta)
        case let .controlledRotationZ(theta): controlledRotationZMatrix(theta: theta)
        case let .customSingleQubit(matrix): matrix
        case let .customTwoQubit(matrix): matrix
        }
    }

    // MARK: - Single-Qubit Matrix Implementations

    private func identityMatrix() -> [[Complex<Double>]] {
        [
            [.one, .zero],
            [.zero, .one],
        ]
    }

    private func pauliXMatrix() -> [[Complex<Double>]] {
        [
            [.zero, .one],
            [.one, .zero],
        ]
    }

    private func pauliYMatrix() -> [[Complex<Double>]] {
        [
            [.zero, -Complex.i],
            [Complex.i, .zero],
        ]
    }

    private func pauliZMatrix() -> [[Complex<Double>]] {
        [
            [.one, .zero],
            [.zero, Complex(-1.0, 0.0)],
        ]
    }

    private func hadamardMatrix() -> [[Complex<Double>]] {
        let c: Complex<Double> = Complex(invSqrt2, 0.0)
        return [
            [c, c],
            [c, -c],
        ]
    }

    private func phaseMatrix(theta: Double) -> [[Complex<Double>]] {
        let phaseFactor = Complex<Double>(phase: theta)
        return [
            [.one, .zero],
            [.zero, phaseFactor],
        ]
    }

    private func rotationXMatrix(theta: Double) -> [[Complex<Double>]] {
        let halfTheta: Double = theta / 2.0
        let c: Complex<Double> = Complex(cos(halfTheta), 0.0)
        let s: Complex<Double> = Complex(0.0, -sin(halfTheta))
        return [
            [c, s],
            [s, c],
        ]
    }

    private func rotationYMatrix(theta: Double) -> [[Complex<Double>]] {
        let halfTheta: Double = theta / 2.0
        let c: Complex<Double> = Complex(cos(halfTheta), 0.0)
        let s: Complex<Double> = Complex(sin(halfTheta), 0.0)
        return [
            [c, -s],
            [s, c],
        ]
    }

    private func rotationZMatrix(theta: Double) -> [[Complex<Double>]] {
        let halfTheta: Double = theta / 2.0
        let negPhase = Complex<Double>(phase: -halfTheta)
        let posPhase = Complex<Double>(phase: halfTheta)
        return [
            [negPhase, .zero],
            [.zero, posPhase],
        ]
    }

    private func u1Matrix(lambda: Double) -> [[Complex<Double>]] {
        let phaseFactor = Complex<Double>(phase: lambda)
        return [
            [.one, .zero],
            [.zero, phaseFactor],
        ]
    }

    private func u2Matrix(phi: Double, lambda: Double) -> [[Complex<Double>]] {
        let expPhi = Complex<Double>(phase: phi)
        let expLambda = Complex<Double>(phase: lambda)
        let expPhiLambda: Complex<Double> = expPhi * expLambda

        return [
            [Complex(invSqrt2, 0.0), -expLambda * invSqrt2],
            [expPhi * invSqrt2, expPhiLambda * invSqrt2],
        ]
    }

    private func u3Matrix(theta: Double, phi: Double, lambda: Double) -> [[Complex<Double>]] {
        let halfTheta: Double = theta / 2.0
        let cosHalfTheta: Double = cos(halfTheta)
        let sinHalfTheta: Double = sin(halfTheta)

        let expPhi = Complex<Double>(phase: phi)
        let expLambda = Complex<Double>(phase: lambda)
        let expPhiLambda: Complex<Double> = expPhi * expLambda

        return [
            [Complex(cosHalfTheta, 0.0), -expLambda * sinHalfTheta],
            [expPhi * sinHalfTheta, expPhiLambda * cosHalfTheta],
        ]
    }

    private func sxMatrix() -> [[Complex<Double>]] {
        let a: Complex<Double> = Complex(0.5, 0.5)
        let b: Complex<Double> = Complex(0.5, -0.5)
        return [
            [a, b],
            [b, a],
        ]
    }

    private func syMatrix() -> [[Complex<Double>]] {
        let a: Complex<Double> = Complex(0.5, 0.5)
        let b: Complex<Double> = Complex(-0.5, -0.5)
        return [
            [a, b],
            [a, a],
        ]
    }

    // MARK: - Two-Qubit Matrix Implementations

    private func cnotMatrix() -> [[Complex<Double>]] {
        [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .zero, .one],
            [.zero, .zero, .one, .zero],
        ]
    }

    private func controlledPhaseMatrix(theta: Double) -> [[Complex<Double>]] {
        let phaseFactor = Complex<Double>(phase: theta)
        return [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .one, .zero],
            [.zero, .zero, .zero, phaseFactor],
        ]
    }

    private func swapMatrix() -> [[Complex<Double>]] {
        [
            [.one, .zero, .zero, .zero],
            [.zero, .zero, .one, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .zero, .one],
        ]
    }

    private func czMatrix() -> [[Complex<Double>]] {
        [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .one, .zero],
            [.zero, .zero, .zero, Complex(-1.0, 0.0)],
        ]
    }

    private func cyMatrix() -> [[Complex<Double>]] {
        [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .zero, -Complex.i],
            [.zero, .zero, Complex.i, .zero],
        ]
    }

    private func chMatrix() -> [[Complex<Double>]] {
        let c: Complex<Double> = Complex(invSqrt2, 0.0)
        return [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, c, c],
            [.zero, .zero, c, -c],
        ]
    }

    private func sqrtSwapMatrix() -> [[Complex<Double>]] {
        let a: Complex<Double> = Complex(0.5, 0.5)
        let b: Complex<Double> = Complex(0.5, -0.5)
        return [
            [.one, .zero, .zero, .zero],
            [.zero, a, b, .zero],
            [.zero, b, a, .zero],
            [.zero, .zero, .zero, .one],
        ]
    }

    private func controlledRotationXMatrix(theta: Double) -> [[Complex<Double>]] {
        let halfTheta: Double = theta / 2.0
        let c: Complex<Double> = Complex(cos(halfTheta), 0.0)
        let s: Complex<Double> = Complex(0.0, -sin(halfTheta))
        return [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, c, s],
            [.zero, .zero, s, c],
        ]
    }

    private func controlledRotationYMatrix(theta: Double) -> [[Complex<Double>]] {
        let halfTheta: Double = theta / 2.0
        let c: Complex<Double> = Complex(cos(halfTheta), 0.0)
        let s: Complex<Double> = Complex(sin(halfTheta), 0.0)
        return [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, c, -s],
            [.zero, .zero, s, c],
        ]
    }

    private func controlledRotationZMatrix(theta: Double) -> [[Complex<Double>]] {
        let halfTheta: Double = theta / 2.0
        let negPhase = Complex<Double>(phase: -halfTheta)
        let posPhase = Complex<Double>(phase: halfTheta)
        return [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, negPhase, .zero],
            [.zero, .zero, .zero, posPhase],
        ]
    }

    // MARK: - Multi-Qubit Matrix Implementations

    private func toffoliMatrix() -> [[Complex<Double>]] {
        [
            [.one, .zero, .zero, .zero, .zero, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero, .zero, .zero, .zero, .zero],
            [.zero, .zero, .one, .zero, .zero, .zero, .zero, .zero],
            [.zero, .zero, .zero, .one, .zero, .zero, .zero, .zero],
            [.zero, .zero, .zero, .zero, .one, .zero, .zero, .zero],
            [.zero, .zero, .zero, .zero, .zero, .one, .zero, .zero],
            [.zero, .zero, .zero, .zero, .zero, .zero, .zero, .one],
            [.zero, .zero, .zero, .zero, .zero, .zero, .one, .zero],
        ]
    }

    // MARK: - Validation

    /// Validate qubit indices for gate application
    ///
    /// Checks that qubit array has correct length and all indices are within bounds and distinct.
    /// Required before applying gate to ensure circuit correctness.
    ///
    /// - Parameters:
    ///   - qubits: Array of qubit indices to validate
    ///   - maxAllowedQubit: Maximum allowed qubit index (inclusive)
    /// - Returns: True if qubit array is valid for this gate
    /// - Complexity: O(n) where n is number of qubits
    ///
    /// Example:
    /// ```swift
    /// QuantumGate.cnot.validateQubitIndices([0, 1], maxAllowedQubit: 2)  // true
    /// QuantumGate.cnot.validateQubitIndices([0, 0], maxAllowedQubit: 2)  // false (duplicate)
    /// QuantumGate.toffoli.validateQubitIndices([0, 1, 2], maxAllowedQubit: 3)  // true
    /// ```
    ///
    /// - SeeAlso: ``ValidationUtilities``
    @_optimize(speed)
    @_effects(readonly)
    public func validateQubitIndices(_ qubits: [Int], maxAllowedQubit: Int) -> Bool {
        guard qubits.count == qubitsRequired else { return false }
        guard qubits.allSatisfy({ $0 >= 0 && $0 <= maxAllowedQubit }) else { return false }
        guard Set(qubits).count == qubits.count else { return false }

        return true
    }

    // MARK: - CustomStringConvertible

    /// String representation of the gate
    public var description: String {
        switch self {
        case .identity: "I"
        case .pauliX: "X"
        case .pauliY: "Y"
        case .pauliZ: "Z"
        case .hadamard: "H"
        case let .phase(angle): "P(\(Self.formatAngle(angle)))"
        case .sGate: "S"
        case .tGate: "T"
        case let .rotationX(theta): "Rx(\(Self.formatAngle(theta)))"
        case let .rotationY(theta): "Ry(\(Self.formatAngle(theta)))"
        case let .rotationZ(theta): "Rz(\(Self.formatAngle(theta)))"
        case let .u1(lambda): "U1(\(Self.formatAngle(lambda)))"
        case let .u2(phi, lambda): "U2(\(Self.formatAngle(phi)), \(Self.formatAngle(lambda)))"
        case let .u3(theta, phi, lambda): "U3(\(Self.formatAngle(theta)), \(Self.formatAngle(phi)), \(Self.formatAngle(lambda)))"
        case .sx: "SX"
        case .sy: "SY"
        case .customSingleQubit: "CustomU(2x2)"
        case .cnot: "CNOT"
        case .cz: "CZ"
        case .cy: "CY"
        case .ch: "CH"
        case .swap: "SWAP"
        case .sqrtSwap: "√SWAP"
        case .toffoli: "Toffoli"
        case let .controlledPhase(theta):
            "CP(\(Self.formatAngle(theta)))"
        case let .controlledRotationX(theta):
            "CRx(\(Self.formatAngle(theta)))"
        case let .controlledRotationY(theta):
            "CRy(\(Self.formatAngle(theta)))"
        case let .controlledRotationZ(theta):
            "CRz(\(Self.formatAngle(theta)))"
        case .customTwoQubit:
            "CustomU(4x4)"
        }
    }

    private static func formatAngle(_ angle: Double) -> String {
        let formatted = String(format: "%.6f", angle)
        let trimmed = formatted.replacingOccurrences(of: #"\.?0+$"#, with: "", options: .regularExpression)
        return trimmed.isEmpty ? "0" : trimmed
    }
}

// MARK: - Matrix Utilities

public extension QuantumGate {
    /// Verify matrix unitarity: U†U = I
    ///
    /// Checks if matrix preserves quantum state normalization through unitary condition.
    /// All valid quantum gates must be unitary for probability conservation and
    /// reversibility. Uses numerical tolerance (1e-10) for floating-point comparison.
    ///
    /// **Unitarity condition**: U†U = I where U† is conjugate transpose
    ///
    /// - Parameter matrix: Square complex matrix to check
    /// - Returns: True if unitary within tolerance (1e-10)
    /// - Complexity: O(n³) where n is matrix dimension
    ///
    /// Example:
    /// ```swift
    /// let hMatrix = QuantumGate.hadamard.matrix()
    /// QuantumGate.isUnitary(hMatrix)  // true
    ///
    /// let invalid = [[Complex(1, 0), Complex(1, 0)], [Complex(0, 0), Complex(1, 0)]]
    /// QuantumGate.isUnitary(invalid)  // false
    /// ```
    @_optimize(speed)
    @_effects(readonly)
    static func isUnitary(_ matrix: [[Complex<Double>]]) -> Bool {
        let n: Int = matrix.count
        guard matrix.allSatisfy({ $0.count == n }) else { return false }

        let product: [[Complex<Double>]] = matrixMultiply(MatrixUtilities.hermitianConjugate(matrix), matrix)

        for i in 0 ..< n {
            for j in 0 ..< n {
                let expected: Complex<Double> = (i == j) ? .one : .zero
                if abs(product[i][j].real - expected.real) > 1e-10 ||
                    abs(product[i][j].imaginary - expected.imaginary) > 1e-10
                {
                    return false
                }
            }
        }

        return true
    }

    /// Multiply two square matrices using Accelerate BLAS
    ///
    /// - Parameters:
    ///   - a: Left matrix
    ///   - b: Right matrix
    /// - Returns: Matrix product A x B
    /// - Complexity: O(n³) for nxn matrices (optimized for n≤4, BLAS for n>4)
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    static func matrixMultiply(_ a: [[Complex<Double>]], _ b: [[Complex<Double>]]) -> [[Complex<Double>]] {
        let n: Int = a.count

        if n == 2 {
            return [
                [
                    a[0][0] * b[0][0] + a[0][1] * b[1][0],
                    a[0][0] * b[0][1] + a[0][1] * b[1][1],
                ],
                [
                    a[1][0] * b[0][0] + a[1][1] * b[1][0],
                    a[1][0] * b[0][1] + a[1][1] * b[1][1],
                ],
            ]
        }

        guard n > 4 else {
            var result: [[Complex<Double>]] = Array(repeating: Array(repeating: Complex<Double>.zero, count: n), count: n)
            for i in 0 ..< n {
                for j in 0 ..< n {
                    var sum = Complex<Double>.zero
                    for k in 0 ..< n {
                        sum = sum + (a[i][k] * b[k][j])
                    }
                    result[i][j] = sum
                }
            }
            return result
        }

        return MatrixUtilities.matrixMultiply(a, b)
    }

    /// Compare two matrices for equality within numerical tolerance
    ///
    /// Validates that two complex matrices have identical dimensions and that
    /// corresponding elements are equal within tolerance. Essential for testing
    /// gate equivalence and verifying circuit transformations.
    ///
    /// - Parameters:
    ///   - a: First matrix
    ///   - b: Second matrix
    ///   - tolerance: Maximum allowed difference (default: 1e-10)
    /// - Returns: True if matrices are equal within tolerance
    /// - Complexity: O(n²) where n is matrix dimension
    ///
    /// Example:
    /// ```swift
    /// let phase = QuantumGate.phase(angle: .pi).matrix()
    /// let z = QuantumGate.pauliZ.matrix()
    /// QuantumGate.matricesEqual(phase, z)  // true
    /// ```
    @_optimize(speed)
    @_effects(readonly)
    static func matricesEqual(
        _ a: [[Complex<Double>]],
        _ b: [[Complex<Double>]],
        tolerance: Double = 1e-10
    ) -> Bool {
        guard a.count == b.count else { return false }

        for i in 0 ..< a.count {
            guard a[i].count == b[i].count else { return false }
            for j in 0 ..< a[i].count {
                let diffReal = abs(a[i][j].real - b[i][j].real)
                let diffImag = abs(a[i][j].imaginary - b[i][j].imaginary)

                if diffReal > tolerance || diffImag > tolerance { return false }
            }
        }

        return true
    }

    /// Check if matrix is the identity matrix within tolerance
    ///
    /// Validates that matrix is diagonal with 1s on diagonal and 0s elsewhere.
    /// Used for testing self-inverse gates (H² = I, X² = I) and verifying
    /// gate decompositions.
    ///
    /// - Parameters:
    ///   - matrix: Matrix to check
    ///   - tolerance: Maximum allowed difference from identity (default: 1e-10)
    /// - Returns: True if matrix is identity within tolerance
    /// - Complexity: O(n²) where n is matrix dimension
    ///
    /// Example:
    /// ```swift
    /// let h = QuantumGate.hadamard.matrix()
    /// let hh = QuantumGate.matrixMultiply(h, h)
    /// QuantumGate.isIdentityMatrix(hh)  // true
    /// ```
    @_optimize(speed)
    @_effects(readonly)
    static func isIdentityMatrix(
        _ matrix: [[Complex<Double>]],
        tolerance: Double = 1e-10
    ) -> Bool {
        guard !matrix.isEmpty else { return false }

        let n: Int = matrix.count

        guard matrix.allSatisfy({ $0.count == n }) else { return false }

        for i in 0 ..< n {
            for j in 0 ..< n {
                let expected: Complex<Double> = (i == j) ? .one : .zero
                let actual: Complex<Double> = matrix[i][j]

                let diffReal = abs(actual.real - expected.real)
                let diffImag = abs(actual.imaginary - expected.imaginary)

                if diffReal > tolerance || diffImag > tolerance { return false }
            }
        }

        return true
    }

    // MARK: - Custom Gate Factory Methods

    /// Create validated custom single-qubit gate
    ///
    /// Builds custom quantum gate from user-provided 2x2 complex matrix.
    /// Validates matrix size and unitarity before creating gate. Useful for
    /// research, custom algorithms, and gate decomposition.
    ///
    /// - Parameter matrix: 2x2 complex matrix (must be unitary)
    /// - Returns: Custom single-qubit gate
    ///
    /// Example:
    /// ```swift
    /// let angle = Double.pi / 6
    /// let customMatrix = [
    ///     [Complex(cos(angle), 0), Complex(-sin(angle), 0)],
    ///     [Complex(sin(angle), 0), Complex(cos(angle), 0)]
    /// ]
    /// let gate = QuantumGate.custom(matrix: customMatrix)
    /// ```
    ///
    /// - SeeAlso: ``isUnitary(_:)``, ``ValidationUtilities``
    @_eagerMove
    static func custom(matrix: [[Complex<Double>]]) -> QuantumGate {
        ValidationUtilities.validate2x2Matrix(matrix)
        ValidationUtilities.validateUnitary(matrix)

        return .customSingleQubit(matrix: matrix)
    }

    /// Create validated custom two-qubit gate
    ///
    /// Builds custom two-qubit gate from user-provided 4x4 complex matrix.
    /// Validates matrix size and unitarity. Useful for custom entangling operations,
    /// variational quantum circuits, and quantum algorithm research.
    ///
    /// - Parameters:
    ///   - matrix: 4x4 complex matrix (must be unitary)
    ///   - control: Control qubit index
    ///   - target: Target qubit index
    /// - Returns: Custom two-qubit gate
    ///
    /// Example:
    /// ```swift
    /// let theta = Double.pi / 4
    /// let c = cos(theta / 2)
    /// let s = sin(theta / 2)
    /// let customMatrix = [
    ///     [Complex(1, 0), Complex(0, 0), Complex(0, 0), Complex(0, 0)],
    ///     [Complex(0, 0), Complex(1, 0), Complex(0, 0), Complex(0, 0)],
    ///     [Complex(0, 0), Complex(0, 0), Complex(c, 0), Complex(-s, 0)],
    ///     [Complex(0, 0), Complex(0, 0), Complex(s, 0), Complex(c, 0)]
    /// ]
    /// let gate = QuantumGate.custom(matrix: customMatrix, control: 0, target: 1)
    /// ```
    ///
    /// - SeeAlso: ``isUnitary(_:)``, ``ValidationUtilities``
    @_eagerMove
    static func custom(
        matrix: [[Complex<Double>]],
        control _: Int,
        target _: Int
    ) -> QuantumGate {
        ValidationUtilities.validate4x4Matrix(matrix)
        ValidationUtilities.validateUnitary(matrix)

        return .customTwoQubit(matrix: matrix)
    }
}
