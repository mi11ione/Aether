// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Multi-controlled gate decomposition to basis gates for quantum circuit compilation.
///
/// Transforms arbitrary controlled unitaries C^n(U) into sequences of native gates (single-qubit
/// gates, CNOT, Toffoli) suitable for hardware execution. Essential for Phase Estimation which
/// requires controlled-U^(2^k) operations where control qubit j applies C-U^(2^j) to the target.
///
/// The decomposition strategy depends on control count:
/// - 0 controls: Direct gate application
/// - 1 control: Native controlled gates (CNOT, CZ, CY, CH, controlled rotations)
/// - 2 controls + X: Toffoli gate
/// - n >= 3 controls: Toffoli ladder decomposition with (n-2) ancilla qubits
///
/// For non-X target gates with multiple controls, basis transformation converts C^n(U) to
/// C^n(X) via conjugation: C^n(Z) = H * C^n(X) * H, C^n(Y) = S^dagger * C^n(X) * S.
///
/// **Example:**
/// ```swift
/// let gates = ControlledGateDecomposer.decompose(
///     gate: .pauliZ,
///     controls: [0, 1, 2],
///     target: 3
/// )
/// for (gate, qubits) in gates {
///     circuit.append(gate, to: qubits)
/// }
/// ```
///
/// - Complexity: O(n) gates for n controls using Toffoli ladder decomposition
///
/// - SeeAlso: ``QuantumGate``
/// - SeeAlso: ``CircuitOptimizer``
public enum ControlledGateDecomposer {
    private static let halfPi = Double.pi / 2.0
    private static let negHalfPi = -Double.pi / 2.0

    /// Decompose controlled gate into basis gate sequence.
    ///
    /// Main entry point for multi-controlled gate decomposition. Handles arbitrary control
    /// counts by selecting optimal decomposition strategy: direct application for 0 controls,
    /// native controlled gates for 1 control, Toffoli for 2 controls with X gate, and
    /// Toffoli ladder for 3+ controls.
    ///
    /// **Example:**
    /// ```swift
    /// let decomposition = ControlledGateDecomposer.decompose(
    ///     gate: .pauliX,
    ///     controls: [0, 1],
    ///     target: 2
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Single-qubit gate to apply conditionally
    ///   - controls: Control qubit indices (can be empty)
    ///   - target: Target qubit index
    /// - Returns: Array of (gate, qubits) tuples implementing the controlled operation
    ///
    /// - Complexity: O(n) where n = controls.count
    ///
    /// - Precondition: Gate must be single-qubit, target must not be in controls, all indices non-negative
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func decompose(
        gate: QuantumGate,
        controls: [Int],
        target: Int,
    ) -> [(gate: QuantumGate, qubits: [Int])] {
        ValidationUtilities.validateControlledGateIsSingleQubit(gate.qubitsRequired)
        ValidationUtilities.validateNonNegativeQubits(controls)
        ValidationUtilities.validateNonNegativeInt(target, name: "target")

        let n = controls.count

        if n == 0 {
            return [(gate, [target])]
        }

        if n == 1 {
            return decomposeSingleControlled(gate: gate, control: controls[0], target: target)
        }

        if n == 2 {
            if case .pauliX = gate {
                return [(.toffoli, [controls[0], controls[1], target])]
            }
        }

        return toffoliLadderDecomposition(gate: gate, controls: controls, target: target)
    }

    /// Decompose single-controlled gate to native gates.
    ///
    /// Maps single-controlled gates to their native implementations when available.
    /// Pauli gates map to CNOT/CZ/CY, Hadamard to CH, and rotations to controlled
    /// rotation variants. Custom gates are converted to 4x4 controlled matrices.
    ///
    /// **Example:**
    /// ```swift
    /// let gates = ControlledGateDecomposer.decomposeSingleControlled(
    ///     gate: .pauliZ,
    ///     control: 0,
    ///     target: 1
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Single-qubit gate to control
    ///   - control: Control qubit index
    ///   - target: Target qubit index
    /// - Returns: Array of (gate, qubits) tuples implementing the controlled operation
    ///
    /// - Complexity: O(1) for native gates, O(n^2) for custom gates requiring matrix construction
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func decomposeSingleControlled(
        gate: QuantumGate,
        control: Int,
        target: Int,
    ) -> [(gate: QuantumGate, qubits: [Int])] {
        switch gate {
        case .pauliX:
            return [(.cnot, [control, target])]

        case .pauliZ:
            return [(.cz, [control, target])]

        case .pauliY:
            return [(.cy, [control, target])]

        case .hadamard:
            return [(.ch, [control, target])]

        case .identity:
            return []

        case let .rotationX(theta):
            return [(.controlledRotationX(theta), [control, target])]

        case let .rotationY(theta):
            return [(.controlledRotationY(theta), [control, target])]

        case let .rotationZ(theta):
            return [(.controlledRotationZ(theta), [control, target])]

        case let .phase(theta):
            return [(.controlledPhase(theta), [control, target])]

        case .sGate:
            return [(.controlledPhase(.value(halfPi)), [control, target])]

        case .tGate:
            return [(.controlledPhase(.value(Double.pi / 4.0)), [control, target])]

        case let .u1(lambda):
            return [(.controlledPhase(lambda), [control, target])]

        case let .u3(theta, phi, lambda):
            return decomposeControlledU3(theta: theta, phi: phi, lambda: lambda, control: control, target: target)

        case let .u2(phi, lambda):
            let theta = ParameterValue.value(halfPi)
            return decomposeControlledU3(theta: theta, phi: phi, lambda: lambda, control: control, target: target)

        default:
            let matrix = gate.matrix()
            let controlledMatrix = buildControlledMatrix(from: matrix)
            return [(.customTwoQubit(matrix: controlledMatrix), [control, target])]
        }
    }

    /// Decompose multi-controlled gate using Toffoli ladder.
    ///
    /// Implements C^n(U) for n >= 3 controls using a ladder of Toffoli gates with
    /// (n-2) ancilla qubits. The ancillas are allocated starting at max(controls, target) + 1.
    /// For non-X gates, applies basis transformation before and after the ladder.
    ///
    /// The decomposition computes controls progressively: first Toffoli combines c[0] AND c[1]
    /// into ancilla[0], subsequent Toffolis combine ancilla[i-1] AND c[i+1] into ancilla[i].
    /// Final Toffoli applies X to target when all controls are satisfied. Ladder is then
    /// uncomputed in reverse order.
    ///
    /// **Example:**
    /// ```swift
    /// let gates = ControlledGateDecomposer.toffoliLadderDecomposition(
    ///     gate: .pauliX,
    ///     controls: [0, 1, 2, 3],
    ///     target: 4
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Single-qubit gate to apply conditionally
    ///   - controls: Control qubit indices (n >= 3)
    ///   - target: Target qubit index
    /// - Returns: Array of (gate, qubits) tuples implementing the multi-controlled operation
    ///
    /// - Complexity: O(n) Toffoli gates where n = controls.count
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func toffoliLadderDecomposition(
        gate: QuantumGate,
        controls: [Int],
        target: Int,
    ) -> [(gate: QuantumGate, qubits: [Int])] {
        let n = controls.count

        if n == 0 {
            return [(gate, [target])]
        }

        if n == 1 {
            return decomposeSingleControlled(gate: gate, control: controls[0], target: target)
        }

        if n == 2 {
            let (prefix, suffix) = basisChangeForGate(gate, target: target)
            var result = prefix
            result.append((.toffoli, [controls[0], controls[1], target]))
            result.append(contentsOf: suffix)
            return result
        }

        let (prefix, suffix) = basisChangeForGate(gate, target: target)

        // Safety: Controls always non-empty when n≥3 path reached
        let maxControlQubit = controls.max()!
        let maxUsedQubit = max(maxControlQubit, target)
        let firstAncilla = maxUsedQubit + 1
        let numAncilla = n - 2

        var result: [(gate: QuantumGate, qubits: [Int])] = []
        result.reserveCapacity(prefix.count + suffix.count + 2 * (numAncilla + 1))

        result.append(contentsOf: prefix)

        result.append((.toffoli, [controls[0], controls[1], firstAncilla]))

        for i in 1 ..< numAncilla {
            result.append((.toffoli, [firstAncilla + i - 1, controls[i + 1], firstAncilla + i]))
        }

        result.append((.toffoli, [firstAncilla + numAncilla - 1, controls[n - 1], target]))

        for i in (1 ..< numAncilla).reversed() {
            result.append((.toffoli, [firstAncilla + i - 1, controls[i + 1], firstAncilla + i]))
        }

        result.append((.toffoli, [controls[0], controls[1], firstAncilla]))

        result.append(contentsOf: suffix)

        return result
    }

    /// Compute controlled power of unitary for Phase Estimation.
    ///
    /// Constructs C-U^(2^k) by repeated matrix squaring, essential for QPE where control
    /// qubit j applies C-U^(2^j) to extract eigenvalue phase bits. The powered unitary
    /// is computed as U^(2^power) and wrapped in a controlled gate before decomposition.
    ///
    /// **Example:**
    /// ```swift
    /// let gates = ControlledGateDecomposer.controlledPower(
    ///     of: .rotationZ(.pi / 4),
    ///     power: 3,
    ///     control: 0,
    ///     targetQubits: [4]
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Base unitary gate U
    ///   - power: Exponent k where final gate is C-U^(2^k)
    ///   - control: Control qubit index
    ///   - targetQubits: Target qubit indices for the unitary
    /// - Returns: Array of (gate, qubits) tuples implementing C-U^(2^power)
    ///
    /// - Complexity: O(k) matrix multiplications for power k, plus decomposition cost
    ///
    /// - Precondition: Power must be non-negative, targetQubits must be non-empty
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func controlledPower(
        of gate: QuantumGate,
        power: Int,
        control: Int,
        targetQubits: [Int],
    ) -> [(gate: QuantumGate, qubits: [Int])] {
        ValidationUtilities.validateNonNegativeInt(power, name: "power")
        ValidationUtilities.validateNonEmpty(targetQubits, name: "targetQubits")

        let baseMatrix = gate.matrix()
        let poweredMatrix = matrixPower(baseMatrix, exponent: 1 << power)

        if gate.qubitsRequired == 1 {
            let poweredGate = QuantumGate.customSingleQubit(matrix: poweredMatrix)
            return decompose(gate: poweredGate, controls: [control], target: targetQubits[0])
        }

        let controlledMatrix = buildControlledMatrixFromNxN(poweredMatrix)
        let controlledGate = QuantumGate.customTwoQubit(matrix: controlledMatrix)

        var allQubits = [control]
        allQubits.append(contentsOf: targetQubits)

        return [(controlledGate, allQubits)]
    }

    /// Generate basis change gates for converting C^n(X) to C^n(U).
    ///
    /// Returns prefix and suffix gate sequences that transform C^n(X) to C^n(U) via
    /// conjugation: C^n(U) = prefix * C^n(X) * suffix. Standard basis changes:
    /// - Z: (H, H) since HXH = Z
    /// - Y: (S^dagger, S) since S^dagger X S = Y
    /// - X: empty (no change needed)
    /// - Arbitrary U: (U^dagger, U) decomposed to basis gates
    ///
    /// **Example:**
    /// ```swift
    /// let (prefix, suffix) = ControlledGateDecomposer.basisChangeForGate(
    ///     .pauliZ,
    ///     target: 2
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Target gate to transform from X
    ///   - target: Target qubit index
    /// - Returns: Tuple of (prefix, suffix) gate sequences
    ///
    /// - Complexity: O(1) for standard gates, O(1) for custom gates
    @_optimize(speed)
    @_effects(readonly)
    public static func basisChangeForGate(
        _ gate: QuantumGate,
        target: Int,
    ) -> (prefix: [(gate: QuantumGate, qubits: [Int])], suffix: [(gate: QuantumGate, qubits: [Int])]) {
        switch gate {
        case .pauliX:
            return ([], [])

        case .pauliZ:
            return (
                [(.hadamard, [target])],
                [(.hadamard, [target])],
            )

        case .pauliY:
            return (
                [(.phase(.value(negHalfPi)), [target])],
                [(.sGate, [target])],
            )

        case .hadamard:
            return (
                [(.rotationY(negHalfPi), [target])],
                [(.rotationY(halfPi), [target])],
            )

        case let .rotationZ(theta):
            let halfTheta = extractHalfAngle(theta)
            return (
                [(.rotationZ(halfTheta.negated), [target])],
                [(.rotationZ(halfTheta), [target])],
            )

        case let .phase(theta):
            let halfTheta = extractHalfAngle(theta)
            return (
                [(.phase(halfTheta.negated), [target])],
                [(.phase(halfTheta), [target])],
            )

        case .sGate:
            return (
                [(.phase(.value(-Double.pi / 4.0)), [target])],
                [(.phase(.value(Double.pi / 4.0)), [target])],
            )

        case .tGate:
            return (
                [(.phase(.value(-Double.pi / 8.0)), [target])],
                [(.phase(.value(Double.pi / 8.0)), [target])],
            )

        case .identity:
            return ([], [])

        default:
            let inverseGate = gate.inverse
            return (
                [(inverseGate, [target])],
                [(gate, [target])],
            )
        }
    }

    /// Build 4x4 controlled matrix from 2x2 single-qubit matrix.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func buildControlledMatrix(from matrix: [[Complex<Double>]]) -> [[Complex<Double>]] {
        [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, matrix[0][0], matrix[0][1]],
            [.zero, .zero, matrix[1][0], matrix[1][1]],
        ]
    }

    /// Build controlled matrix from arbitrary n x n unitary (for two-qubit targets).
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func buildControlledMatrixFromNxN(_ matrix: [[Complex<Double>]]) -> [[Complex<Double>]] {
        let n = matrix.count

        let dimension = 2 * n
        var result = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: dimension), count: dimension)

        for i in 0 ..< n {
            result[i][i] = .one
        }

        for i in 0 ..< n {
            for j in 0 ..< n {
                result[n + i][n + j] = matrix[i][j]
            }
        }

        return result
    }

    /// Decompose controlled U3 gate into native gates.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func decomposeControlledU3(
        theta: ParameterValue,
        phi: ParameterValue,
        lambda: ParameterValue,
        control: Int,
        target: Int,
    ) -> [(gate: QuantumGate, qubits: [Int])] {
        let halfTheta = extractHalfAngle(theta)
        let sumPhiLambda = addParameterValues(phi, lambda)
        let diffPhiLambda = subtractParameterValues(phi, lambda)
        let halfSumPhiLambda = extractHalfAngle(sumPhiLambda)
        let halfDiffPhiLambda = extractHalfAngle(diffPhiLambda)

        var result: [(gate: QuantumGate, qubits: [Int])] = []
        result.reserveCapacity(8)

        result.append((.rotationZ(halfSumPhiLambda), [target]))
        result.append((.cnot, [control, target]))
        result.append((.rotationY(halfTheta.negated), [target]))
        result.append((.rotationZ(halfDiffPhiLambda.negated), [target]))
        result.append((.cnot, [control, target]))
        result.append((.rotationZ(halfDiffPhiLambda), [target]))
        result.append((.rotationY(halfTheta), [target]))

        return result
    }

    /// Compute matrix power using repeated squaring.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func matrixPower(_ matrix: [[Complex<Double>]], exponent: Int) -> [[Complex<Double>]] {
        let n = matrix.count

        var result = MatrixUtilities.identityMatrix(dimension: n)
        var base = matrix
        var exp = exponent

        while exp > 0 {
            if exp & 1 == 1 {
                result = QuantumGate.matrixMultiply(result, base)
            }
            base = QuantumGate.matrixMultiply(base, base)
            exp >>= 1
        }

        return result
    }

    /// Extract half of a parameter value.
    @_optimize(speed)
    @_effects(readonly)
    private static func extractHalfAngle(_ value: ParameterValue) -> ParameterValue {
        switch value {
        case let .value(v):
            .value(v / 2.0)
        case let .parameter(p):
            .parameter(Parameter(name: "\(p.name)_half"))
        case let .negatedParameter(p):
            .negatedParameter(Parameter(name: "\(p.name)_half"))
        }
    }

    /// Add two parameter values.
    @_optimize(speed)
    @_effects(readonly)
    private static func addParameterValues(_ a: ParameterValue, _ b: ParameterValue) -> ParameterValue {
        switch (a, b) {
        case let (.value(v1), .value(v2)):
            .value(v1 + v2)
        case let (.parameter(p1), .parameter(p2)):
            .parameter(Parameter(name: "\(p1.name)_plus_\(p2.name)"))
        case let (.value(v), .parameter(p)), let (.parameter(p), .value(v)):
            .parameter(Parameter(name: "\(p.name)_plus_\(v)"))
        case let (.negatedParameter(p1), .negatedParameter(p2)):
            .parameter(Parameter(name: "-\(p1.name)_plus_-\(p2.name)"))
        case let (.negatedParameter(p), .value(v)):
            .parameter(Parameter(name: "-\(p.name)_plus_\(v)"))
        case let (.value(v), .negatedParameter(p)):
            .parameter(Parameter(name: "\(v)_plus_-\(p.name)"))
        case let (.parameter(p1), .negatedParameter(p2)):
            .parameter(Parameter(name: "\(p1.name)_plus_-\(p2.name)"))
        case let (.negatedParameter(p1), .parameter(p2)):
            .parameter(Parameter(name: "-\(p1.name)_plus_\(p2.name)"))
        }
    }

    /// Subtract two parameter values.
    @_optimize(speed)
    @_effects(readonly)
    private static func subtractParameterValues(_ a: ParameterValue, _ b: ParameterValue) -> ParameterValue {
        switch (a, b) {
        case let (.value(v1), .value(v2)):
            .value(v1 - v2)
        case let (.parameter(p1), .parameter(p2)):
            .parameter(Parameter(name: "\(p1.name)_minus_\(p2.name)"))
        case let (.parameter(p), .value(v)):
            .parameter(Parameter(name: "\(p.name)_minus_\(v)"))
        case let (.value(v), .parameter(p)):
            .parameter(Parameter(name: "\(v)_minus_\(p.name)"))
        case let (.negatedParameter(p1), .negatedParameter(p2)):
            .parameter(Parameter(name: "-\(p1.name)_minus_-\(p2.name)"))
        case let (.negatedParameter(p), .value(v)):
            .parameter(Parameter(name: "-\(p.name)_minus_\(v)"))
        case let (.value(v), .negatedParameter(p)):
            .parameter(Parameter(name: "\(v)_minus_-\(p.name)"))
        case let (.parameter(p1), .negatedParameter(p2)):
            .parameter(Parameter(name: "\(p1.name)_minus_-\(p2.name)"))
        case let (.negatedParameter(p1), .parameter(p2)):
            .parameter(Parameter(name: "-\(p1.name)_minus_\(p2.name)"))
        }
    }
}
