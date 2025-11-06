// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Enumeration of all quantum gates with their unitary matrix representations.
///
/// Gates are the building blocks of quantum circuits. Each gate applies a specific
/// unitary transformation to quantum states. All gates preserve normalization
/// (probability conservation) through unitarity: U†U = I.
///
/// Gate types:
/// - Single-qubit gates: 2×2 complex matrices (Pauli, Hadamard, Phase, Rotation)
/// - Two-qubit gates: 4×4 complex matrices (CNOT, Controlled-Phase, SWAP)
/// - Multi-qubit gates: 8×8+ matrices (Toffoli, etc.)
enum QuantumGate: Equatable, Hashable, CustomStringConvertible {
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

    // MARK: - Two-Qubit Gates

    /// CNOT (Controlled-NOT) gate
    /// Control qubit: if |1⟩, flip target qubit
    /// Most important two-qubit gate for entanglement
    case cnot(control: Int, target: Int)

    /// Controlled-Phase gate with angle θ
    /// Applies phase e^(iθ) only when both qubits are |1⟩
    /// Critical for QFT algorithm
    case controlledPhase(theta: Double, control: Int, target: Int)

    /// SWAP gate - exchanges two qubits
    /// Useful for qubit routing
    case swap(qubit1: Int, qubit2: Int)

    // MARK: - Multi-Qubit Gates

    /// Toffoli gate (CCNOT - controlled-controlled-NOT)
    /// Two control qubits, one target
    /// Flips target only if both controls are |1⟩
    case toffoli(control1: Int, control2: Int, target: Int)

    // MARK: - Gate Properties

    /// Number of qubits this gate operates on
    var qubitsRequired: Int {
        switch self {
        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard,
             .phase, .sGate, .tGate, .rotationX, .rotationY, .rotationZ: 1
        case .cnot, .controlledPhase, .swap: 2
        case .toffoli: 3
        }
    }

    /// Whether gate has parameter(s)
    var isParameterized: Bool {
        switch self {
        case .phase, .rotationX, .rotationY, .rotationZ, .controlledPhase: true
        default: false
        }
    }

    /// Whether gate is Hermitian (self-adjoint)
    var isHermitian: Bool {
        switch self {
        case .pauliX, .pauliY, .pauliZ, .hadamard: true
        default: false
        }
    }

    // MARK: - Matrix Generation

    /// Generate unitary matrix for this gate
    /// - Returns: 2D array of complex numbers representing the gate
    func matrix() -> [[Complex<Double>]] {
        switch self {
        case .identity: identityMatrix()
        case .pauliX: pauliXMatrix()
        case .pauliY: pauliYMatrix()
        case .pauliZ: pauliZMatrix()
        case .hadamard: hadamardMatrix()
        case .sGate: phaseMatrix(theta: .pi / 2.0)
        case .tGate: phaseMatrix(theta: .pi / 4.0)
        case .cnot: cnotMatrix()
        case .swap: swapMatrix()
        case .toffoli: toffoliMatrix()
        case let .phase(theta): phaseMatrix(theta: theta)
        case let .rotationX(theta): rotationXMatrix(theta: theta)
        case let .rotationY(theta): rotationYMatrix(theta: theta)
        case let .rotationZ(theta): rotationZMatrix(theta: theta)
        case let .controlledPhase(theta, _, _): controlledPhaseMatrix(theta: theta)
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
        let invSqrt2 = 1.0 / sqrt(2.0)
        let c = Complex(invSqrt2, 0.0)
        return [
            [c, c],
            [c, -c],
        ]
    }

    private func phaseMatrix(theta: Double) -> [[Complex<Double>]] {
        let phaseFactor = Complex<Double>.exp(theta)
        return [
            [.one, .zero],
            [.zero, phaseFactor],
        ]
    }

    private func rotationXMatrix(theta: Double) -> [[Complex<Double>]] {
        let halfTheta = theta / 2.0
        let c = Complex(cos(halfTheta), 0.0)
        let s = Complex(0.0, -sin(halfTheta))
        return [
            [c, s],
            [s, c],
        ]
    }

    private func rotationYMatrix(theta: Double) -> [[Complex<Double>]] {
        let halfTheta = theta / 2.0
        let c = Complex(cos(halfTheta), 0.0)
        let s = Complex(sin(halfTheta), 0.0)
        return [
            [c, -s],
            [s, c],
        ]
    }

    private func rotationZMatrix(theta: Double) -> [[Complex<Double>]] {
        let halfTheta = theta / 2.0
        let negPhase = Complex<Double>.exp(-halfTheta)
        let posPhase = Complex<Double>.exp(halfTheta)
        return [
            [negPhase, .zero],
            [.zero, posPhase],
        ]
    }

    // MARK: - Two-Qubit Matrix Implementations

    private func cnotMatrix() -> [[Complex<Double>]] {
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

    private func controlledPhaseMatrix(theta: Double) -> [[Complex<Double>]] {
        // Applies phase only when both qubits are |1⟩
        let phaseFactor = Complex<Double>.exp(theta)
        return [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .one, .zero],
            [.zero, .zero, .zero, phaseFactor],
        ]
    }

    private func swapMatrix() -> [[Complex<Double>]] {
        // Exchanges two qubits
        [
            [.one, .zero, .zero, .zero],
            [.zero, .zero, .one, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .zero, .one],
        ]
    }

    // MARK: - Multi-Qubit Matrix Implementations

    private func toffoliMatrix() -> [[Complex<Double>]] {
        // 8×8 matrix for 3 qubits
        // Flips target only if both controls are |1⟩
        // Identity on all states except |110⟩↔|111⟩
        var matrix = Array(repeating: Array(repeating: Complex<Double>.zero, count: 8), count: 8)

        for i in 0 ..< 6 {
            matrix[i][i] = .one
        }

        matrix[6][7] = .one
        matrix[7][6] = .one

        return matrix
    }

    // MARK: - Validation

    /// Validate qubit indices for this gate
    /// - Parameter maxAllowedQubit: Maximum allowed qubit index (inclusive)
    /// - Returns: True if all qubit indices are valid and distinct
    func validateQubitIndices(maxAllowedQubit: Int) -> Bool {
        switch self {
        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard,
             .phase, .sGate, .tGate, .rotationX, .rotationY, .rotationZ:
            return true

        case let .cnot(control, target):
            return control != target &&
                control >= 0 && control <= maxAllowedQubit &&
                target >= 0 && target <= maxAllowedQubit

        case let .controlledPhase(_, control, target):
            return control != target &&
                control >= 0 && control <= maxAllowedQubit &&
                target >= 0 && target <= maxAllowedQubit

        case let .swap(q1, q2):
            return q1 != q2 &&
                q1 >= 0 && q1 <= maxAllowedQubit &&
                q2 >= 0 && q2 <= maxAllowedQubit

        case let .toffoli(c1, c2, target):
            let indices = Set([c1, c2, target])
            return indices.count == 3 && // All distinct
                c1 >= 0 && c1 <= maxAllowedQubit &&
                c2 >= 0 && c2 <= maxAllowedQubit &&
                target >= 0 && target <= maxAllowedQubit
        }
    }

    // MARK: - CustomStringConvertible

    /// String representation of the gate
    var description: String {
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
        case let .cnot(control, target): "CNOT(c:\(control), t:\(target))"
        case let .swap(q1, q2): "SWAP(\(q1), \(q2))"
        case let .toffoli(c1, c2, target): "Toffoli(c1:\(c1), c2:\(c2), t:\(target))"
        case let .controlledPhase(theta, control, target):
            "CP(\(String(format: "%.3f", theta)), c:\(control), t:\(target))"
        }
    }
}

// MARK: - Matrix Utilities

extension QuantumGate {
    /// Check if matrix is unitary: U†U = I
    /// - Parameter matrix: Matrix to check
    /// - Returns: True if unitary within tolerance
    static func isUnitary(_ matrix: [[Complex<Double>]]) -> Bool {
        let n = matrix.count
        guard matrix.allSatisfy({ $0.count == n }) else { return false }

        // Compute U†U
        let product = matrixMultiply(conjugateTranspose(matrix), matrix)

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

    /// Compute conjugate transpose of matrix (U†)
    /// - Parameter matrix: Input matrix
    /// - Returns: Conjugate transpose of the input matrix
    private static func conjugateTranspose(_ matrix: [[Complex<Double>]]) -> [[Complex<Double>]] {
        let n = matrix.count
        var result = Array(repeating: Array(repeating: Complex<Double>.zero, count: n), count: n)

        for i in 0 ..< n {
            for j in 0 ..< n {
                result[j][i] = matrix[i][j].conjugate
            }
        }

        return result
    }

    /// Multiply two square matrices
    /// - Parameters:
    ///   - a: Left matrix
    ///   - b: Right matrix
    /// - Returns: Matrix product A × B
    private static func matrixMultiply(_ a: [[Complex<Double>]], _ b: [[Complex<Double>]]) -> [[Complex<Double>]] {
        let n = a.count
        var result = Array(repeating: Array(repeating: Complex<Double>.zero, count: n), count: n)

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
}
