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
/// - **Phase gates**: P(θ), S (π/2), T (π/4), U1(λ) - support symbolic parameters
/// - **Rotation gates**: Rx(θ), Ry(θ), Rz(θ) - parameterized rotations with symbolic/concrete values
/// - **IBM gates**: U1, U2, U3 (universal single-qubit gates with symbolic parameters)
/// - **Controlled gates**: CNOT, CZ, CY, CH, controlled rotations (symbolic parameters)
/// - **Multi-qubit**: SWAP, √SWAP, Toffoli (CCNOT)
/// - **Custom gates**: User-defined unitaries with validation
///
/// **Parameter support**:
/// Gates with angles support both symbolic parameters and concrete values via ``ParameterValue``.
/// This enables building variational circuits once, then binding different parameter values
/// per optimization iteration without circuit reconstruction.
///
/// **Usage patterns**:
/// - **Static circuits**: Use concrete values directly
/// - **Variational circuits**: Use symbolic ``Parameter`` instances, bind before execution
/// - **Mixed circuits**: Combine symbolic and concrete gates in same circuit
///
/// Example:
/// ```swift
/// // Static circuit with concrete values
/// var circuit = QuantumCircuit(numQubits: 2)
/// circuit.append(.hadamard, to: 0)
/// circuit.append(.rotationY(.pi/4), to: 1)
/// circuit.append(.cnot, to: [0, 1])
///
/// // Variational circuit with symbolic parameters
/// let theta = Parameter(name: "theta")
/// circuit.append(.rotationY(theta), to: 0)
/// let bound = circuit.binding(["theta": 1.57])
/// let state = bound.execute()
/// ```
///
/// - SeeAlso: ``QuantumCircuit``, ``GateApplication``, ``Parameter``, ``ParameterValue``
public enum QuantumGate: Equatable, Hashable, CustomStringConvertible, Sendable {
    // MARK: - Single-Qubit Gates

    case identity
    case pauliX
    case pauliY
    case pauliZ
    case hadamard
    case phase(ParameterValue)
    case sGate
    case tGate
    case rotationX(ParameterValue)
    case rotationY(ParameterValue)
    case rotationZ(ParameterValue)
    case u1(lambda: ParameterValue)
    case u2(phi: ParameterValue, lambda: ParameterValue)
    case u3(theta: ParameterValue, phi: ParameterValue, lambda: ParameterValue)
    case sx
    case sy
    case customSingleQubit(matrix: [[Complex<Double>]])

    // MARK: - Two-Qubit Gates

    case cnot
    case cz
    case cy
    case ch
    case controlledPhase(ParameterValue)
    case controlledRotationX(ParameterValue)
    case controlledRotationY(ParameterValue)
    case controlledRotationZ(ParameterValue)
    case swap
    case sqrtSwap
    case customTwoQubit(matrix: [[Complex<Double>]])

    // MARK: - Multi-Qubit Gates

    case toffoli

    // MARK: - Convenience Constructors

    /// Create rotation-X gate with concrete angle
    @inlinable
    public static func rotationX(_ theta: Double) -> QuantumGate {
        .rotationX(.value(theta))
    }

    /// Create rotation-X gate with symbolic parameter
    @inlinable
    public static func rotationX(_ param: Parameter) -> QuantumGate {
        .rotationX(.parameter(param))
    }

    /// Create rotation-Y gate with concrete angle
    @inlinable
    public static func rotationY(_ theta: Double) -> QuantumGate {
        .rotationY(.value(theta))
    }

    /// Create rotation-Y gate with symbolic parameter
    @inlinable
    public static func rotationY(_ param: Parameter) -> QuantumGate {
        .rotationY(.parameter(param))
    }

    /// Create rotation-Z gate with concrete angle
    @inlinable
    public static func rotationZ(_ theta: Double) -> QuantumGate {
        .rotationZ(.value(theta))
    }

    /// Create rotation-Z gate with symbolic parameter
    @inlinable
    public static func rotationZ(_ param: Parameter) -> QuantumGate {
        .rotationZ(.parameter(param))
    }

    /// Create phase gate with concrete angle
    @inlinable
    public static func phase(_ angle: Double) -> QuantumGate {
        .phase(.value(angle))
    }

    /// Create phase gate with symbolic parameter
    @inlinable
    public static func phase(_ param: Parameter) -> QuantumGate {
        .phase(.parameter(param))
    }

    /// Create controlled-phase gate with concrete angle
    @inlinable
    public static func controlledPhase(_ theta: Double) -> QuantumGate {
        .controlledPhase(.value(theta))
    }

    /// Create controlled-phase gate with symbolic parameter
    @inlinable
    public static func controlledPhase(_ param: Parameter) -> QuantumGate {
        .controlledPhase(.parameter(param))
    }

    /// Create controlled-Rx gate with concrete angle
    @inlinable
    public static func controlledRotationX(_ theta: Double) -> QuantumGate {
        .controlledRotationX(.value(theta))
    }

    /// Create controlled-Rx gate with symbolic parameter
    @inlinable
    public static func controlledRotationX(_ param: Parameter) -> QuantumGate {
        .controlledRotationX(.parameter(param))
    }

    /// Create controlled-Ry gate with concrete angle
    @inlinable
    public static func controlledRotationY(_ theta: Double) -> QuantumGate {
        .controlledRotationY(.value(theta))
    }

    /// Create controlled-Ry gate with symbolic parameter
    @inlinable
    public static func controlledRotationY(_ param: Parameter) -> QuantumGate {
        .controlledRotationY(.parameter(param))
    }

    /// Create controlled-Rz gate with concrete angle
    @inlinable
    public static func controlledRotationZ(_ theta: Double) -> QuantumGate {
        .controlledRotationZ(.value(theta))
    }

    /// Create controlled-Rz gate with symbolic parameter
    @inlinable
    public static func controlledRotationZ(_ param: Parameter) -> QuantumGate {
        .controlledRotationZ(.parameter(param))
    }

    /// Create U1 gate with concrete lambda
    @inlinable
    public static func u1(lambda: Double) -> QuantumGate {
        .u1(lambda: .value(lambda))
    }

    /// Create U1 gate with symbolic parameter
    @inlinable
    public static func u1(lambda param: Parameter) -> QuantumGate {
        .u1(lambda: .parameter(param))
    }

    /// Create U2 gate with concrete angles
    @inlinable
    public static func u2(phi: Double, lambda: Double) -> QuantumGate {
        .u2(phi: .value(phi), lambda: .value(lambda))
    }

    /// Create U2 gate with symbolic parameters
    @inlinable
    public static func u2(phi phiParam: Parameter, lambda lambdaParam: Parameter) -> QuantumGate {
        .u2(phi: .parameter(phiParam), lambda: .parameter(lambdaParam))
    }

    /// Create U3 gate with concrete angles
    @inlinable
    public static func u3(theta: Double, phi: Double, lambda: Double) -> QuantumGate {
        .u3(theta: .value(theta), phi: .value(phi), lambda: .value(lambda))
    }

    /// Create U3 gate with symbolic parameters
    @inlinable
    public static func u3(theta thetaParam: Parameter, phi phiParam: Parameter, lambda lambdaParam: Parameter) -> QuantumGate {
        .u3(theta: .parameter(thetaParam), phi: .parameter(phiParam), lambda: .parameter(lambdaParam))
    }

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

    /// Whether gate contains symbolic parameters requiring binding
    ///
    /// Returns `true` if any ``ParameterValue`` in gate is symbolic (.parameter case).
    /// Used for circuit validation and parameter tracking.
    ///
    /// Example:
    /// ```swift
    /// QuantumGate.hadamard.hasSymbolicParameters  // false
    /// QuantumGate.rotationY(.pi/4).hasSymbolicParameters  // false
    /// let theta = Parameter(name: "theta")
    /// QuantumGate.rotationY(theta).hasSymbolicParameters  // true
    /// ```
    @inlinable
    public var hasSymbolicParameters: Bool {
        !parameters().isEmpty
    }

    /// Extract symbolic parameters from gate
    ///
    /// Returns all ``Parameter`` instances used in gate. Empty set for non-parameterized
    /// gates and gates with only concrete values. Used for circuit-level parameter discovery.
    ///
    /// **Example:**
    /// ```swift
    /// let theta = Parameter(name: "theta")
    /// let phi = Parameter(name: "phi")
    /// let gate = QuantumGate.u2(phi: .parameter(phi), lambda: .parameter(theta))
    ///
    /// gate.parameters()  // {theta, phi}
    /// ```
    ///
    /// - Returns: Set of symbolic parameters
    /// - Complexity: O(1) - gates have at most 3 parameters
    @_optimize(speed)
    @inlinable
    @_effects(readonly)
    public func parameters() -> Set<Parameter> {
        switch self {
        case let .phase(angle),
             let .rotationX(angle),
             let .rotationY(angle),
             let .rotationZ(angle),
             let .controlledPhase(angle),
             let .controlledRotationX(angle),
             let .controlledRotationY(angle),
             let .controlledRotationZ(angle):
            if let p = angle.parameter { return [p] }
            return []

        case let .u1(lambda):
            if let p = lambda.parameter { return [p] }
            return []

        case let .u2(phi, lambda):
            var params: [Parameter] = []
            params.reserveCapacity(2)
            if let p = phi.parameter { params.append(p) }
            if let p = lambda.parameter { params.append(p) }
            return Set(params)

        case let .u3(theta, phi, lambda):
            var params: [Parameter] = []
            params.reserveCapacity(3)
            if let p = theta.parameter { params.append(p) }
            if let p = phi.parameter { params.append(p) }
            if let p = lambda.parameter { params.append(p) }
            return Set(params)

        default: return []
        }
    }

    /// Bind symbolic parameters to concrete values
    ///
    /// Evaluates all ``ParameterValue`` instances using provided bindings, producing
    /// gate with only concrete values. Gates without symbolic parameters return unchanged.
    ///
    /// **Example:**
    /// ```swift
    /// let theta = Parameter(name: "theta")
    /// let gate = QuantumGate.rotationY(theta)
    ///
    /// let bound = gate.binding(["theta": .pi / 4])
    /// // Returns QuantumGate.rotationY(.pi/4)
    /// ```
    ///
    /// - Parameter bindings: Dictionary mapping parameter names to numerical values
    /// - Returns: Gate with all parameters substituted
    /// - Precondition: All symbolic parameters must exist in bindings
    @_optimize(speed)
    @inlinable
    @_eagerMove
    public func binding(_ bindings: [String: Double]) -> QuantumGate {
        switch self {
        case let .phase(angle):
            .phase(.value(angle.evaluate(using: bindings)))
        case let .rotationX(theta):
            .rotationX(.value(theta.evaluate(using: bindings)))
        case let .rotationY(theta):
            .rotationY(.value(theta.evaluate(using: bindings)))
        case let .rotationZ(theta):
            .rotationZ(.value(theta.evaluate(using: bindings)))
        case let .u1(lambda):
            .u1(lambda: .value(lambda.evaluate(using: bindings)))
        case let .u2(phi, lambda):
            .u2(phi: .value(phi.evaluate(using: bindings)),
                lambda: .value(lambda.evaluate(using: bindings)))
        case let .u3(theta, phi, lambda):
            .u3(theta: .value(theta.evaluate(using: bindings)),
                phi: .value(phi.evaluate(using: bindings)),
                lambda: .value(lambda.evaluate(using: bindings)))
        case let .controlledPhase(theta):
            .controlledPhase(.value(theta.evaluate(using: bindings)))
        case let .controlledRotationX(theta):
            .controlledRotationX(.value(theta.evaluate(using: bindings)))
        case let .controlledRotationY(theta):
            .controlledRotationY(.value(theta.evaluate(using: bindings)))
        case let .controlledRotationZ(theta):
            .controlledRotationZ(.value(theta.evaluate(using: bindings)))
        default: self // Non-parameterized gates return unchanged
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
    /// For gates with symbolic parameters, all values must be concrete before calling.
    /// Use ``binding(_:)`` to substitute symbolic parameters first.
    ///
    /// - Returns: 2D array of complex numbers (2x2, 4x4, or 8x8)
    /// - Complexity: O(1) for fixed-size matrices
    /// - Precondition: All parameters must be concrete (.value case)
    ///
    /// Example:
    /// ```swift
    /// let h = QuantumGate.hadamard
    /// let matrix = h.matrix()
    /// // [[0.707+0i, 0.707+0i], [0.707+0i, -0.707+0i]]
    ///
    /// let theta = Parameter(name: "theta")
    /// let symbolic = QuantumGate.rotationY(theta)
    /// let bound = symbolic.binding(["theta": .pi/4])
    /// let concreteMatrix = bound.matrix()  // OK
    /// // let fail = symbolic.matrix()  // Precondition failure!
    /// ```
    ///
    /// - SeeAlso: ``MatrixUtilities``, ``binding(_:)``
    @_optimize(speed)
    @_eagerMove
    public func matrix() -> [[Complex<Double>]] {
        switch self {
        case .identity: return identityMatrix()
        case .pauliX: return pauliXMatrix()
        case .pauliY: return pauliYMatrix()
        case .pauliZ: return pauliZMatrix()
        case .hadamard: return hadamardMatrix()
        case .sGate: return phaseMatrix(theta: .pi / 2.0)
        case .tGate: return phaseMatrix(theta: .pi / 4.0)
        case .sx: return sxMatrix()
        case .sy: return syMatrix()
        case .cnot: return cnotMatrix()
        case .cz: return czMatrix()
        case .cy: return cyMatrix()
        case .ch: return chMatrix()
        case .swap: return swapMatrix()
        case .sqrtSwap: return sqrtSwapMatrix()
        case .toffoli: return toffoliMatrix()
        case let .phase(angle):
            ValidationUtilities.validateConcrete(angle, name: "phase angle")
            return phaseMatrix(theta: angle.evaluate(using: [:]))
        case let .rotationX(theta):
            ValidationUtilities.validateConcrete(theta, name: "rotation X angle")
            return rotationXMatrix(theta: theta.evaluate(using: [:]))
        case let .rotationY(theta):
            ValidationUtilities.validateConcrete(theta, name: "rotation Y angle")
            return rotationYMatrix(theta: theta.evaluate(using: [:]))
        case let .rotationZ(theta):
            ValidationUtilities.validateConcrete(theta, name: "rotation Z angle")
            return rotationZMatrix(theta: theta.evaluate(using: [:]))
        case let .u1(lambda):
            ValidationUtilities.validateConcrete(lambda, name: "U1 lambda")
            return u1Matrix(lambda: lambda.evaluate(using: [:]))
        case let .u2(phi, lambda):
            ValidationUtilities.validateConcrete(phi, name: "U2 phi")
            ValidationUtilities.validateConcrete(lambda, name: "U2 lambda")
            return u2Matrix(phi: phi.evaluate(using: [:]), lambda: lambda.evaluate(using: [:]))
        case let .u3(theta, phi, lambda):
            ValidationUtilities.validateConcrete(theta, name: "U3 theta")
            ValidationUtilities.validateConcrete(phi, name: "U3 phi")
            ValidationUtilities.validateConcrete(lambda, name: "U3 lambda")
            return u3Matrix(theta: theta.evaluate(using: [:]), phi: phi.evaluate(using: [:]), lambda: lambda.evaluate(using: [:]))
        case let .controlledPhase(theta):
            ValidationUtilities.validateConcrete(theta, name: "controlled phase angle")
            return controlledPhaseMatrix(theta: theta.evaluate(using: [:]))
        case let .controlledRotationX(theta):
            ValidationUtilities.validateConcrete(theta, name: "controlled rotation X angle")
            return controlledRotationXMatrix(theta: theta.evaluate(using: [:]))
        case let .controlledRotationY(theta):
            ValidationUtilities.validateConcrete(theta, name: "controlled rotation Y angle")
            return controlledRotationYMatrix(theta: theta.evaluate(using: [:]))
        case let .controlledRotationZ(theta):
            ValidationUtilities.validateConcrete(theta, name: "controlled rotation Z angle")
            return controlledRotationZMatrix(theta: theta.evaluate(using: [:]))
        case let .customSingleQubit(matrix): return matrix
        case let .customTwoQubit(matrix): return matrix
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

    // MARK: - CustomStringConvertible

    /// String representation of the gate
    public var description: String {
        switch self {
        case .identity: "I"
        case .pauliX: "X"
        case .pauliY: "Y"
        case .pauliZ: "Z"
        case .hadamard: "H"
        case let .phase(angle): "P(\(angle))"
        case .sGate: "S"
        case .tGate: "T"
        case let .rotationX(theta): "Rx(\(theta))"
        case let .rotationY(theta): "Ry(\(theta))"
        case let .rotationZ(theta): "Rz(\(theta))"
        case let .u1(lambda): "U1(\(lambda))"
        case let .u2(phi, lambda): "U2(\(phi), \(lambda))"
        case let .u3(theta, phi, lambda): "U3(\(theta), \(phi), \(lambda))"
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
        case let .controlledPhase(theta): "CP(\(theta))"
        case let .controlledRotationX(theta): "CRx(\(theta))"
        case let .controlledRotationY(theta): "CRy(\(theta))"
        case let .controlledRotationZ(theta): "CRz(\(theta))"
        case .customTwoQubit: "CustomU(4x4)"
        }
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
