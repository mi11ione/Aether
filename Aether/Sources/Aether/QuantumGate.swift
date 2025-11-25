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
/// - Single-qubit: 2×2 complex matrices operating on C²
/// - Two-qubit: 4×4 complex matrices operating on C⁴
/// - Multi-qubit: 2^n × 2^n matrices operating on C^(2^n)
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
/// - Matrix access: Use `matrix()` to get unitary representation
/// - Validation: Gates validate qubit indices and matrix unitarity
///
/// Example:
/// ```swift
/// // Single-qubit gates
/// let x = QuantumGate.pauliX
/// let h = QuantumGate.hadamard
/// let rz = QuantumGate.rotationZ(theta: .pi/4)
///
/// // Two-qubit gates
/// let cnot = QuantumGate.cnot(control: 0, target: 1)
/// let cphase = QuantumGate.controlledPhase(theta: .pi/2, control: 1, target: 2)
///
/// // Multi-qubit gates
/// let toffoli = QuantumGate.toffoli(control1: 0, control2: 1, target: 2)
///
/// // Create Bell state circuit
/// var circuit = QuantumCircuit(numQubits: 2)
/// circuit.append(gate: .hadamard, toQubit: 0)
/// circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
/// let bellState = circuit.execute()
///
/// // Custom gate with validation
/// let customMatrix = [
///     [Complex(0.5, 0.5), Complex(0.5, -0.5)],
///     [Complex(0.5, -0.5), Complex(0.5, 0.5)]
/// ]
/// let customGate = try QuantumGate.createCustomSingleQubit(matrix: customMatrix)
///
/// // Gate properties
/// print(cnot.qubitsRequired)  // 2
/// print(rz.isParameterized)   // true
/// print(h.isHermitian)        // true
/// ```
@frozen
public enum QuantumGate: Equatable, Hashable, CustomStringConvertible, Sendable {
    // MARK: - Single-Qubit Gates

    /// Identity gate - no operation
    case identity

    /// Pauli-X gate (NOT gate, bit flip)
    /// Matrix: [[0, 1], [1, 0]]
    case pauliX

    /// Pauli-Y gate (bit flip + phase flip)
    /// Matrix: [[0, -i], [i, 0]]
    case pauliY

    /// Pauli-Z gate (phase flip)
    /// Matrix: [[1, 0], [0, -1]]
    case pauliZ

    /// Hadamard gate (superposition creator)
    /// Matrix: (1/√2)[[1, 1], [1, -1]]
    /// Creates equal superposition from basis states
    case hadamard

    /// Phase gate with rotation angle θ
    /// Matrix: [[1, 0], [0, e^(iθ)]]
    /// Applies relative phase rotation to |1⟩ component
    case phase(theta: Double)

    /// S gate (Phase π/2, quarter-turn phase)
    /// Equivalent to phase(π/2)
    case sGate

    /// T gate (Phase π/4, eighth-turn phase)
    /// Equivalent to phase(π/4)
    case tGate

    /// Rotation around X-axis by angle θ
    /// Matrix uses cos(θ/2) and sin(θ/2)
    case rotationX(theta: Double)

    /// Rotation around Y-axis by angle θ
    /// Matrix uses cos(θ/2) and sin(θ/2)
    case rotationY(theta: Double)

    /// Rotation around Z-axis by angle θ
    /// Matrix uses e^(-iθ/2) and e^(iθ/2)
    case rotationZ(theta: Double)

    /// IBM U1 gate - single-parameter phase gate
    /// U1(λ) = diag(1, e^(iλ))
    /// Pure phase gate, equivalent to Rz(λ) up to global phase
    case u1(lambda: Double)

    /// IBM U2 gate - two-parameter gate
    /// U2(φ,λ) creates superposition with phases
    /// Equivalent to Rz(φ)·Ry(π/2)·Rz(λ)
    case u2(phi: Double, lambda: Double)

    /// IBM U3 gate - three-parameter universal single-qubit gate
    /// U3(θ,φ,λ) is most general single-qubit rotation
    /// Can decompose any single-qubit gate into U3
    case u3(theta: Double, phi: Double, lambda: Double)

    /// Square root of X gate (SX)
    /// SX · SX = X
    /// Native gate on IBM hardware
    /// Used in efficient gate decompositions
    case sx

    /// Square root of Y gate (SY)
    /// SY · SY = Y
    /// Completes square root Pauli gates
    case sy

    /// Arbitrary single-qubit unitary
    /// User-provided 2×2 complex matrix
    /// Validates unitarity: U†U = I
    /// Used for custom gates and research
    case customSingleQubit(matrix: GateMatrix)

    // MARK: - Two-Qubit Gates

    /// CNOT (Controlled-NOT) gate
    /// Control qubit: if |1⟩, flip target qubit
    /// Most important two-qubit gate for entanglement
    case cnot(control: Int, target: Int)

    /// Controlled-Z gate
    /// Symmetric: CZ(a,b) = CZ(b,a)
    /// Native to many superconducting architectures
    /// Matrix: diag(1,1,1,-1) on computational basis
    case cz(control: Int, target: Int)

    /// Controlled-Y gate
    /// Completes controlled Pauli set (CX, CY, CZ)
    /// Less common but necessary for full gate library
    case cy(control: Int, target: Int)

    /// Controlled-Hadamard gate
    /// Creates specific entangled states
    /// Used in quantum communication protocols
    case ch(control: Int, target: Int)

    /// Controlled-Phase gate with angle θ
    /// Applies phase e^(iθ) only when both qubits are |1⟩
    /// Critical for QFT algorithm
    case controlledPhase(theta: Double, control: Int, target: Int)

    /// Controlled rotation around X-axis
    /// Parameterized controlled gate
    /// Essential for variational quantum circuits
    case controlledRotationX(theta: Double, control: Int, target: Int)

    /// Controlled rotation around Y-axis
    /// Parameterized controlled gate
    /// Essential for variational quantum circuits
    case controlledRotationY(theta: Double, control: Int, target: Int)

    /// Controlled rotation around Z-axis
    /// Parameterized controlled gate
    /// Essential for variational quantum circuits
    case controlledRotationZ(theta: Double, control: Int, target: Int)

    /// SWAP gate - exchanges two qubits
    /// Useful for qubit routing
    case swap(qubit1: Int, qubit2: Int)

    /// Square root of SWAP gate
    /// (√SWAP)² = SWAP
    /// Creates entanglement more gradually than CNOT
    /// Used in quantum communication protocols
    case sqrtSwap(qubit1: Int, qubit2: Int)

    /// Arbitrary two-qubit unitary
    /// User-provided 4×4 complex matrix
    /// Validates unitarity: U†U = I
    /// Used for custom gates and research
    case customTwoQubit(matrix: GateMatrix, control: Int, target: Int)

    // MARK: - Multi-Qubit Gates

    /// Toffoli gate (CCNOT - controlled-controlled-NOT)
    /// Two control qubits, one target
    /// Flips target only if both controls are |1⟩
    case toffoli(control1: Int, control2: Int, target: Int)

    // MARK: - Gate Properties

    /// Number of qubits this gate operates on
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

    /// Whether gate has parameter(s)
    @inlinable
    public var isParameterized: Bool {
        switch self {
        case .phase, .rotationX, .rotationY, .rotationZ, .u1, .u2, .u3,
             .controlledPhase, .controlledRotationX, .controlledRotationY, .controlledRotationZ: true
        default: false
        }
    }

    /// Whether gate is Hermitian (self-adjoint)
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
    /// Matrix size depends on gate type: 2×2 for single-qubit, 4×4 for two-qubit,
    /// 8×8 for three-qubit gates. All matrices satisfy U†U = I (unitarity).
    ///
    /// - Returns: 2D array of complex numbers (2×2, 4×4, or 8×8)
    ///
    /// Example:
    /// ```swift
    /// // Hadamard matrix: (1/√2)[[1,1],[1,-1]]
    /// let h = QuantumGate.hadamard
    /// let hMatrix = h.matrix()
    /// // [[0.707+0i, 0.707+0i],
    /// //  [0.707+0i, -0.707+0i]]
    ///
    /// // Pauli-X matrix: [[0,1],[1,0]]
    /// let x = QuantumGate.pauliX
    /// let xMatrix = x.matrix()
    /// // [[0+0i, 1+0i],
    /// //  [1+0i, 0+0i]]
    ///
    /// // CNOT matrix: 4×4 identity with last two rows swapped
    /// let cnot = QuantumGate.cnot(control: 0, target: 1)
    /// let cnotMatrix = cnot.matrix()
    /// ```
    @_optimize(speed)
    @_eagerMove
    public func matrix() -> GateMatrix {
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
        case let .phase(theta): phaseMatrix(theta: theta)
        case let .rotationX(theta): rotationXMatrix(theta: theta)
        case let .rotationY(theta): rotationYMatrix(theta: theta)
        case let .rotationZ(theta): rotationZMatrix(theta: theta)
        case let .u1(lambda): u1Matrix(lambda: lambda)
        case let .u2(phi, lambda): u2Matrix(phi: phi, lambda: lambda)
        case let .u3(theta, phi, lambda): u3Matrix(theta: theta, phi: phi, lambda: lambda)
        case let .controlledPhase(theta, _, _): controlledPhaseMatrix(theta: theta)
        case let .controlledRotationX(theta, _, _): controlledRotationXMatrix(theta: theta)
        case let .controlledRotationY(theta, _, _): controlledRotationYMatrix(theta: theta)
        case let .controlledRotationZ(theta, _, _): controlledRotationZMatrix(theta: theta)
        case let .customSingleQubit(matrix): matrix
        case let .customTwoQubit(matrix, _, _): matrix
        }
    }

    // MARK: - Single-Qubit Matrix Implementations

    private func identityMatrix() -> GateMatrix {
        [
            [.one, .zero],
            [.zero, .one],
        ]
    }

    private func pauliXMatrix() -> GateMatrix {
        [
            [.zero, .one],
            [.one, .zero],
        ]
    }

    private func pauliYMatrix() -> GateMatrix {
        [
            [.zero, -Complex.i],
            [Complex.i, .zero],
        ]
    }

    private func pauliZMatrix() -> GateMatrix {
        [
            [.one, .zero],
            [.zero, Complex(-1.0, 0.0)],
        ]
    }

    private func hadamardMatrix() -> GateMatrix {
        let c: Complex<Double> = Complex(invSqrt2, 0.0)
        return [
            [c, c],
            [c, -c],
        ]
    }

    private func phaseMatrix(theta: Double) -> GateMatrix {
        let phaseFactor = Complex<Double>.exp(theta)
        return [
            [.one, .zero],
            [.zero, phaseFactor],
        ]
    }

    private func rotationXMatrix(theta: Double) -> GateMatrix {
        let halfTheta: Double = theta / 2.0
        let c: Complex<Double> = Complex(cos(halfTheta), 0.0)
        let s: Complex<Double> = Complex(0.0, -sin(halfTheta))
        return [
            [c, s],
            [s, c],
        ]
    }

    private func rotationYMatrix(theta: Double) -> GateMatrix {
        let halfTheta: Double = theta / 2.0
        let c: Complex<Double> = Complex(cos(halfTheta), 0.0)
        let s: Complex<Double> = Complex(sin(halfTheta), 0.0)
        return [
            [c, -s],
            [s, c],
        ]
    }

    private func rotationZMatrix(theta: Double) -> GateMatrix {
        let halfTheta: Double = theta / 2.0
        let negPhase = Complex<Double>.exp(-halfTheta)
        let posPhase = Complex<Double>.exp(halfTheta)
        return [
            [negPhase, .zero],
            [.zero, posPhase],
        ]
    }

    private func u1Matrix(lambda: Double) -> GateMatrix {
        // U1(λ) = diag(1, e^(iλ))
        let phaseFactor = Complex<Double>.exp(lambda)
        return [
            [.one, .zero],
            [.zero, phaseFactor],
        ]
    }

    private func u2Matrix(phi: Double, lambda: Double) -> GateMatrix {
        // U2(φ,λ) = (1/√2) * [[1, -e^(iλ)], [e^(iφ), e^(i(φ+λ))]]
        let expPhi = Complex<Double>.exp(phi)
        let expLambda = Complex<Double>.exp(lambda)
        let expPhiLambda: Complex<Double> = expPhi * expLambda

        return [
            [Complex(invSqrt2, 0.0), -expLambda * invSqrt2],
            [expPhi * invSqrt2, expPhiLambda * invSqrt2],
        ]
    }

    private func u3Matrix(theta: Double, phi: Double, lambda: Double) -> GateMatrix {
        // U3(θ,φ,λ) = [[cos(θ/2), -e^(iλ)sin(θ/2)], [e^(iφ)sin(θ/2), e^(i(φ+λ))cos(θ/2)]]
        let halfTheta: Double = theta / 2.0
        let cosHalfTheta: Double = cos(halfTheta)
        let sinHalfTheta: Double = sin(halfTheta)

        let expPhi = Complex<Double>.exp(phi)
        let expLambda = Complex<Double>.exp(lambda)
        let expPhiLambda: Complex<Double> = expPhi * expLambda

        return [
            [Complex(cosHalfTheta, 0.0), -expLambda * sinHalfTheta],
            [expPhi * sinHalfTheta, expPhiLambda * cosHalfTheta],
        ]
    }

    private func sxMatrix() -> GateMatrix {
        // SX = √X = (1/2) * [[1+i, 1-i], [1-i, 1+i]]
        let a: Complex<Double> = Complex(0.5, 0.5)
        let b: Complex<Double> = Complex(0.5, -0.5)
        return [
            [a, b],
            [b, a],
        ]
    }

    private func syMatrix() -> GateMatrix {
        // SY = √Y = (1/2) * [[1+i, -1-i], [1+i, 1+i]]
        let a: Complex<Double> = Complex(0.5, 0.5)
        let b: Complex<Double> = Complex(-0.5, -0.5)
        return [
            [a, b],
            [a, a],
        ]
    }

    // MARK: - Two-Qubit Matrix Implementations

    private func cnotMatrix() -> GateMatrix {
        // Standard basis order: |00⟩, |01⟩, |10⟩, |11⟩
        // Identity on |00⟩ and |01⟩ (control=0)
        // Swap |10⟩↔|11⟩ (control=1, flip target)
        [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .zero, .one],
            [.zero, .zero, .one, .zero],
        ]
    }

    private func controlledPhaseMatrix(theta: Double) -> GateMatrix {
        // Applies phase only when both qubits are |1⟩
        let phaseFactor = Complex<Double>.exp(theta)
        return [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .one, .zero],
            [.zero, .zero, .zero, phaseFactor],
        ]
    }

    private func swapMatrix() -> GateMatrix {
        // Exchanges two qubits
        [
            [.one, .zero, .zero, .zero],
            [.zero, .zero, .one, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .zero, .one],
        ]
    }

    private func czMatrix() -> GateMatrix {
        // Controlled-Z: diag(1,1,1,-1)
        // Symmetric: CZ(a,b) = CZ(b,a)
        [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .one, .zero],
            [.zero, .zero, .zero, Complex(-1.0, 0.0)],
        ]
    }

    private func cyMatrix() -> GateMatrix {
        // Controlled-Y: Identity on control=0, Y on control=1
        // Y = [[0, -i], [i, 0]]
        [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .zero, -Complex.i],
            [.zero, .zero, Complex.i, .zero],
        ]
    }

    private func chMatrix() -> GateMatrix {
        // Controlled-Hadamard
        let c: Complex<Double> = Complex(invSqrt2, 0.0)
        return [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, c, c],
            [.zero, .zero, c, -c],
        ]
    }

    private func sqrtSwapMatrix() -> GateMatrix {
        // √SWAP matrix: (√SWAP)² = SWAP
        // Matrix: [[1, 0, 0, 0],
        //          [0, (1+i)/2, (1-i)/2, 0],
        //          [0, (1-i)/2, (1+i)/2, 0],
        //          [0, 0, 0, 1]]
        let a: Complex<Double> = Complex(0.5, 0.5)
        let b: Complex<Double> = Complex(0.5, -0.5)
        return [
            [.one, .zero, .zero, .zero],
            [.zero, a, b, .zero],
            [.zero, b, a, .zero],
            [.zero, .zero, .zero, .one],
        ]
    }

    private func controlledRotationXMatrix(theta: Double) -> GateMatrix {
        // Controlled Rx: Identity on control=0, Rx(θ) on control=1
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

    private func controlledRotationYMatrix(theta: Double) -> GateMatrix {
        // Controlled Ry: Identity on control=0, Ry(θ) on control=1
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

    private func controlledRotationZMatrix(theta: Double) -> GateMatrix {
        // Controlled Rz: Identity on control=0, Rz(θ) on control=1
        let halfTheta: Double = theta / 2.0
        let negPhase = Complex<Double>.exp(-halfTheta)
        let posPhase = Complex<Double>.exp(halfTheta)
        return [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, negPhase, .zero],
            [.zero, .zero, .zero, posPhase],
        ]
    }

    // MARK: - Multi-Qubit Matrix Implementations

    private func toffoliMatrix() -> GateMatrix {
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
    /// Checks that all qubit indices are within bounds and distinct (no qubit
    /// appears multiple times). Required before applying gate to ensure circuit
    /// correctness.
    ///
    /// - Parameter maxAllowedQubit: Maximum allowed qubit index (inclusive)
    /// - Returns: True if all qubit indices are valid and distinct
    ///
    /// Example:
    /// ```swift
    /// let cnot = QuantumGate.cnot(control: 0, target: 1)
    /// cnot.validateQubitIndices(maxAllowedQubit: 2)  // true
    /// cnot.validateQubitIndices(maxAllowedQubit: 0)  // false (target=1 > max)
    ///
    /// let invalid = QuantumGate.cnot(control: 0, target: 0)
    /// invalid.validateQubitIndices(maxAllowedQubit: 5)  // false (control == target)
    ///
    /// let toffoli = QuantumGate.toffoli(control1: 0, control2: 1, target: 2)
    /// toffoli.validateQubitIndices(maxAllowedQubit: 3)  // true
    /// ```
    @_optimize(speed)
    @_effects(readonly)
    public func validateQubitIndices(maxAllowedQubit: Int) -> Bool {
        switch self {
        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard,
             .phase, .sGate, .tGate, .rotationX, .rotationY, .rotationZ,
             .u1, .u2, .u3, .sx, .sy, .customSingleQubit: true

        case let .cnot(control, target),
             let .cz(control, target),
             let .cy(control, target),
             let .ch(control, target):
            control != target &&
                control >= 0 && control <= maxAllowedQubit &&
                target >= 0 && target <= maxAllowedQubit

        case let .controlledPhase(_, control, target),
             let .controlledRotationX(_, control, target),
             let .controlledRotationY(_, control, target),
             let .controlledRotationZ(_, control, target),
             let .customTwoQubit(_, control, target):
            control != target &&
                control >= 0 && control <= maxAllowedQubit &&
                target >= 0 && target <= maxAllowedQubit

        case let .swap(q1, q2), let .sqrtSwap(q1, q2):
            q1 != q2 &&
                q1 >= 0 && q1 <= maxAllowedQubit &&
                q2 >= 0 && q2 <= maxAllowedQubit

        case let .toffoli(c1, c2, target):
            c1 != c2 && c1 != target && c2 != target &&
                c1 >= 0 && c1 <= maxAllowedQubit &&
                c2 >= 0 && c2 <= maxAllowedQubit &&
                target >= 0 && target <= maxAllowedQubit
        }
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
        case let .phase(theta): "P(\(String(format: "%.3f", theta)))"
        case .sGate: "S"
        case .tGate: "T"
        case let .rotationX(theta): "Rx(\(String(format: "%.3f", theta)))"
        case let .rotationY(theta): "Ry(\(String(format: "%.3f", theta)))"
        case let .rotationZ(theta): "Rz(\(String(format: "%.3f", theta)))"
        case let .u1(lambda): "U1(\(String(format: "%.3f", lambda)))"
        case let .u2(phi, lambda): "U2(\(String(format: "%.3f", phi)), \(String(format: "%.3f", lambda)))"
        case let .u3(theta, phi, lambda): "U3(\(String(format: "%.3f", theta)), \(String(format: "%.3f", phi)), \(String(format: "%.3f", lambda)))"
        case .sx: "SX"
        case .sy: "SY"
        case .customSingleQubit: "CustomU(2×2)"
        case let .cnot(control, target): "CNOT(c:\(control), t:\(target))"
        case let .cz(control, target): "CZ(c:\(control), t:\(target))"
        case let .cy(control, target): "CY(c:\(control), t:\(target))"
        case let .ch(control, target): "CH(c:\(control), t:\(target))"
        case let .swap(q1, q2): "SWAP(\(q1), \(q2))"
        case let .sqrtSwap(q1, q2): "√SWAP(\(q1), \(q2))"
        case let .toffoli(c1, c2, target): "Toffoli(c1:\(c1), c2:\(c2), t:\(target))"
        case let .controlledPhase(theta, control, target):
            "CP(\(String(format: "%.3f", theta)), c:\(control), t:\(target))"
        case let .controlledRotationX(theta, control, target):
            "CRx(\(String(format: "%.3f", theta)), c:\(control), t:\(target))"
        case let .controlledRotationY(theta, control, target):
            "CRy(\(String(format: "%.3f", theta)), c:\(control), t:\(target))"
        case let .controlledRotationZ(theta, control, target):
            "CRz(\(String(format: "%.3f", theta)), c:\(control), t:\(target))"
        case let .customTwoQubit(_, control, target):
            "CustomU(4×4, c:\(control), t:\(target))"
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
    ///
    /// Example:
    /// ```swift
    /// // Hadamard is unitary
    /// let hMatrix = QuantumGate.hadamard.matrix()
    /// QuantumGate.isUnitary(hMatrix)  // true
    ///
    /// // Invalid matrix: not unitary
    /// let invalid = [
    ///     [Complex(1, 0), Complex(1, 0)],
    ///     [Complex(0, 0), Complex(1, 0)]
    /// ]
    /// QuantumGate.isUnitary(invalid)  // false
    ///
    /// // Identity is trivially unitary
    /// let identity = QuantumGate.identity.matrix()
    /// QuantumGate.isUnitary(identity)  // true
    /// ```
    @_optimize(speed)
    @_effects(readonly)
    static func isUnitary(_ matrix: GateMatrix) -> Bool {
        let n: Int = matrix.count
        guard matrix.allSatisfy({ $0.count == n }) else { return false }

        // Compute U†U
        let product: GateMatrix = matrixMultiply(MatrixUtilities.hermitianConjugate(matrix), matrix)

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
    /// - Parameters:
    ///   - a: Left matrix
    ///   - b: Right matrix
    /// - Returns: Matrix product A × B
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    static func matrixMultiply(_ a: GateMatrix, _ b: GateMatrix) -> GateMatrix {
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
            var result: GateMatrix = Array(repeating: Array(repeating: Complex<Double>.zero, count: n), count: n)
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

        // Delegate to shared MatrixUtilities for larger matrices
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
    ///
    /// Example:
    /// ```swift
    /// let phase = QuantumGate.phase(theta: .pi).matrix()
    /// let z = QuantumGate.pauliZ.matrix()
    /// QuantumGate.matricesEqual(phase, z)  // true
    ///
    /// let s = QuantumGate.sGate.matrix()
    /// let phaseHalfPi = QuantumGate.phase(theta: .pi / 2).matrix()
    /// QuantumGate.matricesEqual(s, phaseHalfPi)  // true
    /// ```
    @_optimize(speed)
    @_effects(readonly)
    static func matricesEqual(
        _ a: GateMatrix,
        _ b: GateMatrix,
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
    ///
    /// Example:
    /// ```swift
    /// // H · H = I
    /// let h = QuantumGate.hadamard.matrix()
    /// let hh = QuantumGate.matrixMultiply(h, h)
    /// QuantumGate.isIdentityMatrix(hh)  // true
    ///
    /// // X · X = I
    /// let x = QuantumGate.pauliX.matrix()
    /// let xx = QuantumGate.matrixMultiply(x, x)
    /// QuantumGate.isIdentityMatrix(xx)  // true
    ///
    /// // CNOT · CNOT = I
    /// let cnot = QuantumGate.cnot(control: 0, target: 1).matrix()
    /// let cnotSquared = QuantumGate.matrixMultiply(cnot, cnot)
    /// QuantumGate.isIdentityMatrix(cnotSquared)  // true
    /// ```
    @_optimize(speed)
    @_effects(readonly)
    static func isIdentityMatrix(
        _ matrix: GateMatrix,
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
    /// Builds custom quantum gate from user-provided 2×2 complex matrix.
    /// Validates matrix size and unitarity before creating gate. Useful for
    /// research, custom algorithms, and gate decomposition.
    ///
    /// - Parameter matrix: 2×2 complex matrix (must be unitary)
    /// - Returns: Custom single-qubit gate
    /// - Throws: QuantumGateError if matrix invalid or non-unitary
    ///
    /// Example:
    /// ```swift
    /// // Custom rotation gate
    /// let angle = Double.pi / 6
    /// let customMatrix = [
    ///     [Complex(cos(angle), 0), Complex(-sin(angle), 0)],
    ///     [Complex(sin(angle), 0), Complex(cos(angle), 0)]
    /// ]
    /// let gate = try QuantumGate.createCustomSingleQubit(matrix: customMatrix)
    ///
    /// // Use in circuit
    /// var circuit = QuantumCircuit(numQubits: 1)
    /// circuit.append(gate: gate, toQubit: 0)
    ///
    /// // Invalid: wrong size
    /// let wrongSize = [[Complex(1, 0)]]
    /// do {
    ///     let _ = try QuantumGate.createCustomSingleQubit(matrix: wrongSize)
    /// } catch QuantumGateError.invalidMatrixSize(let msg) {
    ///     print(msg)  // "Custom single-qubit gate requires 2×2 matrix"
    /// }
    ///
    /// // Invalid: not unitary
    /// let notUnitary = [
    ///     [Complex(2, 0), Complex(0, 0)],
    ///     [Complex(0, 0), Complex(1, 0)]
    /// ]
    /// do {
    ///     let _ = try QuantumGate.createCustomSingleQubit(matrix: notUnitary)
    /// } catch QuantumGateError.notUnitary(let msg) {
    ///     print(msg)  // "Matrix is not unitary (U†U ≠ I)"
    /// }
    /// ```
    @_eagerMove
    static func createCustomSingleQubit(matrix: GateMatrix) throws -> QuantumGate {
        guard matrix.count == 2, matrix.allSatisfy({ $0.count == 2 }) else {
            throw QuantumGateError.invalidMatrixSize("Custom single-qubit gate requires 2×2 matrix")
        }

        guard isUnitary(matrix) else {
            throw QuantumGateError.notUnitary("Matrix is not unitary (U†U ≠ I)")
        }

        return .customSingleQubit(matrix: matrix)
    }

    /// Create validated custom two-qubit gate
    ///
    /// Builds custom two-qubit gate from user-provided 4×4 complex matrix.
    /// Validates matrix size and unitarity. Useful for custom entangling operations,
    /// variational quantum circuits, and quantum algorithm research.
    ///
    /// - Parameters:
    ///   - matrix: 4×4 complex matrix (must be unitary)
    ///   - control: Control qubit index
    ///   - target: Target qubit index
    /// - Returns: Custom two-qubit gate
    /// - Throws: QuantumGateError if matrix invalid or non-unitary
    ///
    /// Example:
    /// ```swift
    /// // Custom controlled rotation
    /// let theta = Double.pi / 4
    /// let c = cos(theta / 2)
    /// let s = sin(theta / 2)
    /// let customMatrix = [
    ///     [Complex(1, 0), Complex(0, 0), Complex(0, 0), Complex(0, 0)],
    ///     [Complex(0, 0), Complex(1, 0), Complex(0, 0), Complex(0, 0)],
    ///     [Complex(0, 0), Complex(0, 0), Complex(c, 0), Complex(-s, 0)],
    ///     [Complex(0, 0), Complex(0, 0), Complex(s, 0), Complex(c, 0)]
    /// ]
    /// let gate = try QuantumGate.createCustomTwoQubit(
    ///     matrix: customMatrix,
    ///     control: 0,
    ///     target: 1
    /// )
    ///
    /// // Use in circuit
    /// var circuit = QuantumCircuit(numQubits: 2)
    /// circuit.append(gate: gate, qubits: [])
    ///
    /// // Invalid: wrong size
    /// let wrongSize = [[Complex(1, 0)]]
    /// do {
    ///     let _ = try QuantumGate.createCustomTwoQubit(
    ///         matrix: wrongSize,
    ///         control: 0,
    ///         target: 1
    ///     )
    /// } catch QuantumGateError.invalidMatrixSize(let msg) {
    ///     print(msg)  // "Custom two-qubit gate requires 4×4 matrix"
    /// }
    /// ```
    @_eagerMove
    static func createCustomTwoQubit(
        matrix: GateMatrix,
        control: Int,
        target: Int
    ) throws -> QuantumGate {
        guard matrix.count == 4, matrix.allSatisfy({ $0.count == 4 }) else {
            throw QuantumGateError.invalidMatrixSize("Custom two-qubit gate requires 4×4 matrix")
        }

        guard isUnitary(matrix) else {
            throw QuantumGateError.notUnitary("Matrix is not unitary (U†U ≠ I)")
        }

        return .customTwoQubit(matrix: matrix, control: control, target: target)
    }
}

/// Errors that can occur when creating or validating quantum gates
@frozen
public enum QuantumGateError: Error, LocalizedError {
    case invalidMatrixSize(String)
    case notUnitary(String)

    public var errorDescription: String? {
        switch self {
        case let .invalidMatrixSize(message): message
        case let .notUnitary(message): message
        }
    }
}
