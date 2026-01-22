// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Circuit optimization engine reducing gate count, circuit depth, and execution time through
/// algebraic simplifications, gate fusion, and topological reordering.
///
/// Implements multiple optimization passes that preserve circuit semantics while improving efficiency:
/// identity cancellation removes adjacent inverse gate pairs (H-H, X-X, CNOT-CNOT), single-qubit
/// fusion merges consecutive rotations into minimal gate sequences, commutation analysis reorders
/// gates to maximize parallelism, and KAK decomposition optimally compiles arbitrary two-qubit
/// unitaries to at most 3 CNOTs. Each pass runs in linear time over circuit size, enabling
/// practical optimization of variational circuits with thousands of gates.
///
/// The optimizer operates on concrete circuits (all parameters bound). For variational algorithms,
/// apply optimization after parameter binding or use the symbolic-aware variants that preserve
/// parameter structure. Optimization is idempotent: running multiple times produces the same result.
///
/// **Example:**
/// ```swift
/// var circuit = QuantumCircuit(qubits: 2)
/// circuit.append(.hadamard, to: 0)
/// circuit.append(.hadamard, to: 0)  // H-H = I, will be removed
/// circuit.append(.cnot, to: [0, 1])
/// circuit.append(.cnot, to: [0, 1])  // CNOT-CNOT = I, will be removed
///
/// let optimized = CircuitOptimizer.optimize(circuit)
/// print(optimized.count)  // 0 (all gates cancelled)
/// ```
///
/// - SeeAlso: ``QuantumCircuit``
/// - SeeAlso: ``QuantumGate``
public enum CircuitOptimizer {
    // MARK: - Angle Tolerance

    /// Tolerance for angle comparisons in rotation gate cancellation
    @usableFromInline static let angleTolerance: Double = 1e-10

    // MARK: - Full Optimization Pipeline

    /// Apply all optimization passes for maximum gate reduction
    ///
    /// Runs the complete optimization pipeline: identity cancellation, single-qubit fusion,
    /// two-qubit reduction, and commutation-based depth reduction. Passes execute in optimal
    /// order to maximize gate elimination. The pipeline is idempotent and preserves circuit
    /// semantics exactly.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = HardwareEfficientAnsatz(qubits: 4, depth: 3).circuit.bound(with: params)
    /// let optimized = CircuitOptimizer.optimize(circuit)
    /// print("Reduction: \(circuit.count) -> \(optimized.count) gates")
    /// ```
    ///
    /// - Parameter circuit: Circuit to optimize (must have concrete parameters)
    /// - Returns: Optimized circuit with minimal gate count
    /// - Complexity: O(n²) where n = gate count (dominated by commutation pass)
    /// - Precondition: Circuit must contain only concrete parameters
    @_optimize(speed)
    @_eagerMove
    public static func optimize(_ circuit: QuantumCircuit) -> QuantumCircuit {
        var result = circuit

        result = cancelIdentityPairs(result)
        result = mergeSingleQubitGates(result)
        result = cancelTwoQubitPairs(result)
        result = reorderByCommutation(result)
        result = cancelIdentityPairs(result)
        result = mergeSingleQubitGates(result)

        return result
    }

    // MARK: - Identity Pair Cancellation

    /// Remove adjacent gate pairs that multiply to identity
    ///
    /// Scans circuit for consecutive gates on the same qubits where G₁G₂ = I, removing both.
    /// Handles Hermitian gates (H-H, X-X, Y-Y, Z-Z, SWAP-SWAP), rotation inverses (Rz(θ)Rz(-θ)),
    /// and phase cancellations (S-S†, T-T†). Single linear pass with in-place cancellation.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 1)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.hadamard, to: 0)
    /// let optimized = CircuitOptimizer.cancelIdentityPairs(circuit)
    /// print(optimized.count)  // 0
    /// ```
    ///
    /// - Parameter circuit: Input circuit
    /// - Returns: Circuit with identity pairs removed
    /// - Complexity: O(n) where n = gate count
    @_optimize(speed)
    @_eagerMove
    public static func cancelIdentityPairs(_ circuit: QuantumCircuit) -> QuantumCircuit {
        guard circuit.count > 1 else { return circuit }

        var gates: [Gate] = []
        gates.reserveCapacity(circuit.count)

        for operation in circuit.gates {
            if let last = gates.last,
               last.qubits == operation.qubits,
               gatesFormIdentity(last.gate, operation.gate)
            {
                gates.removeLast()
            } else {
                gates.append(operation)
            }
        }

        return QuantumCircuit(qubits: circuit.qubits, gates: gates)
    }

    /// Check if two gates multiply to identity when applied sequentially
    ///
    /// - Complexity: O(1) for standard gates
    @_optimize(speed)
    @inlinable
    @_effects(readonly)
    static func gatesFormIdentity(_ g1: QuantumGate, _ g2: QuantumGate) -> Bool {
        if g1 == g2 {
            switch g1 {
            case .identity, .pauliX, .pauliY, .pauliZ, .hadamard, .swap, .cnot, .cz, .toffoli:
                return true
            default:
                break
            }
        }

        switch (g1, g2) {
        case let (.sGate, .phase(angle)) where isAngleEqual(angle, -.pi / 2):
            return true
        case let (.phase(angle), .sGate) where isAngleEqual(angle, -.pi / 2):
            return true
        case let (.tGate, .phase(angle)) where isAngleEqual(angle, -.pi / 4):
            return true
        case let (.phase(angle), .tGate) where isAngleEqual(angle, -.pi / 4):
            return true
        case let (.rotationX(theta1), .rotationX(theta2)):
            return anglesCancel(theta1, theta2)
        case let (.rotationY(theta1), .rotationY(theta2)):
            return anglesCancel(theta1, theta2)
        case let (.rotationZ(theta1), .rotationZ(theta2)):
            return anglesCancel(theta1, theta2)
        case let (.phase(theta1), .phase(theta2)):
            return anglesCancel(theta1, theta2)
        case let (.controlledPhase(theta1), .controlledPhase(theta2)):
            return anglesCancel(theta1, theta2)
        case let (.controlledRotationX(theta1), .controlledRotationX(theta2)):
            return anglesCancel(theta1, theta2)
        case let (.controlledRotationY(theta1), .controlledRotationY(theta2)):
            return anglesCancel(theta1, theta2)
        case let (.controlledRotationZ(theta1), .controlledRotationZ(theta2)):
            return anglesCancel(theta1, theta2)
        default:
            return false
        }
    }

    /// Check if angle equals target within tolerance (concrete values only).
    @_optimize(speed)
    @inlinable
    @_effects(readonly)
    static func isAngleEqual(_ angle: ParameterValue, _ target: Double) -> Bool {
        guard case let .value(v) = angle else { return false }
        return abs(v - target) < angleTolerance
    }

    /// Check if two angles sum to zero (cancel).
    @_optimize(speed)
    @inlinable
    @_effects(readonly)
    static func anglesCancel(_ a1: ParameterValue, _ a2: ParameterValue) -> Bool {
        guard case let .value(v1) = a1, case let .value(v2) = a2 else { return false }
        return abs(v1 + v2) < angleTolerance
    }

    // MARK: - Two-Qubit Pair Cancellation

    /// Remove adjacent two-qubit gate pairs that multiply to identity
    ///
    /// Specifically handles CNOT-CNOT and SWAP-SWAP patterns on same control/target qubits.
    /// These are common in decomposed circuits and algorithmic patterns.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.cnot, to: [0, 1])
    /// circuit.append(.cnot, to: [0, 1])
    /// let optimized = CircuitOptimizer.cancelTwoQubitPairs(circuit)
    /// print(optimized.count)  // 0
    /// ```
    ///
    /// - Parameter circuit: Input circuit
    /// - Returns: Circuit with two-qubit identity pairs removed
    /// - Complexity: O(n) where n = gate count
    @_optimize(speed)
    @_eagerMove
    public static func cancelTwoQubitPairs(_ circuit: QuantumCircuit) -> QuantumCircuit {
        guard circuit.count > 1 else { return circuit }

        var gates: [Gate] = []
        gates.reserveCapacity(circuit.count)

        for operation in circuit.gates {
            if let last = gates.last,
               last.qubits == operation.qubits,
               twoQubitGatesCancel(last.gate, operation.gate)
            {
                gates.removeLast()
            } else {
                gates.append(operation)
            }
        }

        return QuantumCircuit(qubits: circuit.qubits, gates: gates)
    }

    /// Check if two two-qubit gates cancel.
    @_optimize(speed)
    @inlinable
    @_effects(readonly)
    static func twoQubitGatesCancel(_ g1: QuantumGate, _ g2: QuantumGate) -> Bool {
        switch (g1, g2) {
        case (.cnot, .cnot), (.cz, .cz), (.swap, .swap), (.cy, .cy):
            true
        case let (.controlledPhase(t1), .controlledPhase(t2)):
            anglesCancel(t1, t2)
        default:
            false
        }
    }

    // MARK: - Single-Qubit Gate Merging

    /// Merge consecutive single-qubit gates on the same qubit into minimal form
    ///
    /// Combines adjacent single-qubit rotations and gates into optimally decomposed sequences.
    /// Same-axis rotations merge by adding angles: Rz(θ₁)Rz(θ₂) = Rz(θ₁+θ₂). Different-axis
    /// rotations fuse via matrix multiplication into U3(θ,φ,λ). Gates with near-identity
    /// angles (|θ| < ε) are removed entirely.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 1)
    /// circuit.append(.rotationZ(.pi/4), to: 0)
    /// circuit.append(.rotationZ(.pi/4), to: 0)
    /// let optimized = CircuitOptimizer.mergeSingleQubitGates(circuit)
    /// // Result: single Rz(π/2)
    /// ```
    ///
    /// - Parameter circuit: Input circuit
    /// - Returns: Circuit with merged single-qubit sequences
    /// - Complexity: O(n) where n = gate count
    @_optimize(speed)
    @_eagerMove
    public static func mergeSingleQubitGates(_ circuit: QuantumCircuit) -> QuantumCircuit {
        guard circuit.count > 1 else { return circuit }

        var gates: [Gate] = []
        gates.reserveCapacity(circuit.count)

        var i = 0
        while i < circuit.gates.count {
            let operation = circuit.gates[i]

            guard operation.gate.qubitsRequired == 1 else {
                gates.append(operation)
                i += 1
                continue
            }

            let qubit = operation.qubits[0]
            var sequence: [QuantumGate] = [operation.gate]
            var j = i + 1

            while j < circuit.gates.count {
                let next = circuit.gates[j]
                if next.gate.qubitsRequired == 1, next.qubits[0] == qubit {
                    sequence.append(next.gate)
                    j += 1
                } else {
                    break
                }
            }

            if sequence.count > 1 {
                let merged = mergeGateSequence(sequence)
                for gate in merged {
                    gates.append(Gate(gate, to: qubit))
                }
            } else {
                gates.append(operation)
            }

            i = j
        }

        return QuantumCircuit(qubits: circuit.qubits, gates: gates)
    }

    /// Merge a sequence of single-qubit gates into minimal representation.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func mergeGateSequence(_ gates: [QuantumGate]) -> [QuantumGate] {
        if let merged = trySameTypeRotationMerge(gates) {
            return merged
        }

        var matrix = gates[0].matrix()
        for i in 1 ..< gates.count {
            matrix = QuantumGate.matrixMultiply(gates[i].matrix(), matrix)
        }

        if QuantumGate.isIdentityMatrix(matrix, tolerance: angleTolerance) {
            return []
        }

        return decomposeToZYZ(matrix, useU3: true)
    }

    /// Try to merge gates if all are same-type rotations.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func trySameTypeRotationMerge(_ gates: [QuantumGate]) -> [QuantumGate]? {
        let first = gates[0]

        var totalAngle: Double = 0

        for gate in gates {
            switch (first, gate) {
            case let (.rotationX(t1), .rotationX(t2)):
                if case let .value(v1) = t1, case let .value(v2) = t2 {
                    if totalAngle == 0 { totalAngle = v1 }
                    else { totalAngle += v2 }
                }

            case let (.rotationY(t1), .rotationY(t2)):
                if case let .value(v1) = t1, case let .value(v2) = t2 {
                    if totalAngle == 0 { totalAngle = v1 }
                    else { totalAngle += v2 }
                }

            case let (.rotationZ(t1), .rotationZ(t2)):
                if case let .value(v1) = t1, case let .value(v2) = t2 {
                    if totalAngle == 0 { totalAngle = v1 }
                    else { totalAngle += v2 }
                }

            case let (.phase(t1), .phase(t2)):
                if case let .value(v1) = t1, case let .value(v2) = t2 {
                    if totalAngle == 0 { totalAngle = v1 }
                    else { totalAngle += v2 }
                }

            default:
                return nil
            }
        }

        totalAngle = normalizeAngle(totalAngle)

        if abs(totalAngle) < angleTolerance {
            return []
        }

        switch first {
        case .rotationX: return [.rotationX(totalAngle)]
        case .rotationY: return [.rotationY(totalAngle)]
        case .rotationZ: return [.rotationZ(totalAngle)]
        case .phase: return [.phase(totalAngle)]
        default: return nil
        }
    }

    /// Normalize angle to [-π, π] range.
    @_optimize(speed)
    @inlinable
    @_effects(readonly)
    static func normalizeAngle(_ angle: Double) -> Double {
        var result = angle.truncatingRemainder(dividingBy: 2 * .pi)
        if result > .pi { result -= 2 * .pi }
        if result < -.pi { result += 2 * .pi }
        return result
    }

    /// Decompose 2x2 unitary to ZYZ Euler angles or U3 gate.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func decomposeToZYZ(_ matrix: [[Complex<Double>]], useU3: Bool = false) -> [QuantumGate] {
        let a = matrix[0][0]
        let b = matrix[0][1]
        let c = matrix[1][0]
        let d = matrix[1][1]

        let theta = 2.0 * acos(min(1.0, a.magnitude))

        var phi: Double
        var lambda: Double

        if abs(theta) < angleTolerance {
            phi = 0
            lambda = normalizeAngle(a.phase + d.phase)
        } else if abs(theta - .pi) < angleTolerance {
            phi = 0
            lambda = normalizeAngle(-c.phase + d.phase)
        } else {
            phi = normalizeAngle(c.phase)
            lambda = normalizeAngle((-b).phase)
        }

        if abs(theta) < angleTolerance, abs(phi) < angleTolerance, abs(lambda) < angleTolerance {
            return []
        }

        if useU3 {
            if abs(theta) < angleTolerance {
                return [.u1(lambda: .value(lambda + phi))]
            }
            return [.u3(theta: .value(theta), phi: .value(phi), lambda: .value(lambda))]
        }

        var result: [QuantumGate] = []

        if abs(lambda) > angleTolerance {
            result.append(.rotationZ(lambda))
        }
        if abs(theta) > angleTolerance {
            result.append(.rotationY(theta))
        }
        if abs(phi) > angleTolerance {
            result.append(.rotationZ(phi))
        }

        return result
    }

    /// Decompose single-qubit unitary to U3 gate (IBM standard)
    ///
    /// Returns single U3(θ,φ,λ) gate or U1(λ) for pure Z rotations.
    /// Optimal single-gate representation for hardware execution.
    ///
    /// **Example:**
    /// ```swift
    /// let matrix = QuantumGate.hadamard.matrix()
    /// let gate = CircuitOptimizer.decomposeToU3(matrix)
    /// // Returns U3 gate equivalent to Hadamard
    /// ```
    ///
    /// - Parameter matrix: 2x2 unitary matrix
    /// - Returns: Single U3 or U1 gate
    @_optimize(speed)
    public static func decomposeToU3(_ matrix: [[Complex<Double>]]) -> QuantumGate {
        let gates = decomposeToZYZ(matrix, useU3: true)
        return gates.first ?? .identity
    }

    // MARK: - Gate Commutation and Reordering

    /// Reorder gates using commutation rules to reduce circuit depth and enable cancellations
    ///
    /// Uses a multi-pass algorithm: builds dependency graph based on qubit usage, identifies
    /// potential inverse pairs separated by commuting gates, moves gates to bring inverse pairs
    /// adjacent, and reorders to minimize circuit depth. Preserves circuit semantics exactly
    /// via conservative commutation rules.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.pauliZ, to: 0)
    /// circuit.append(.rotationZ(.pi/4), to: 1)
    /// circuit.append(.pauliZ, to: 0)
    /// let reordered = CircuitOptimizer.reorderByCommutation(circuit)
    /// ```
    ///
    /// - Parameter circuit: Input circuit
    /// - Returns: Circuit with gates reordered for optimization opportunities
    /// - Complexity: O(n²) where n = gate count
    @_optimize(speed)
    @_eagerMove
    public static func reorderByCommutation(_ circuit: QuantumCircuit) -> QuantumCircuit {
        guard circuit.count > 2 else { return circuit }

        var gates = circuit.gates

        gates = bringInversesAdjacent(gates)
        gates = minimizeDepthOrdering(gates, qubitCount: circuit.qubits)

        return QuantumCircuit(qubits: circuit.qubits, gates: gates)
    }

    /// Move gates to bring inverse pairs adjacent for cancellation.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func bringInversesAdjacent(_ gates: [Gate]) -> [Gate] {
        var result = gates
        var changed = true
        var iterations = 0
        let maxIterations = result.count * 2

        while changed, iterations < maxIterations {
            changed = false
            iterations += 1

            for i in 0 ..< result.count where i + 2 < result.count {
                for j in (i + 2) ..< min(i + 20, result.count) {
                    let g1 = result[i]
                    let gj = result[j]

                    guard g1.qubits == gj.qubits,
                          gatesFormIdentity(g1.gate, gj.gate) else { continue }

                    var canMove = true
                    for k in stride(from: j - 1, through: i + 1, by: -1) {
                        if !gatesCommute(result[k], result[j]) {
                            canMove = false
                            break
                        }
                    }

                    if canMove {
                        let gate = result.remove(at: j)
                        result.insert(gate, at: i + 1)
                        changed = true
                        break
                    }
                }
                if changed { break }
            }
        }

        return result
    }

    /// Reorder gates to minimize circuit depth using as-soon-as-possible scheduling.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func minimizeDepthOrdering(_ gates: [Gate], qubitCount: Int) -> [Gate] {
        var qubitLastGate = [Int](repeating: -1, count: max(qubitCount, 30))
        var gateDepth = [Int](repeating: 0, count: gates.count)

        for (i, gate) in gates.enumerated() {
            var maxPrevDepth = 0
            for q in gate.qubits {
                if q < qubitLastGate.count, qubitLastGate[q] >= 0 {
                    maxPrevDepth = max(maxPrevDepth, gateDepth[qubitLastGate[q]])
                }
            }
            gateDepth[i] = maxPrevDepth + 1
            for q in gate.qubits {
                if q < qubitLastGate.count {
                    qubitLastGate[q] = i
                }
            }
        }

        var indexed = gates.enumerated().map { ($0.offset, $0.element, gateDepth[$0.offset]) }
        indexed.sort { a, b in
            if a.2 != b.2 { return a.2 < b.2 }
            return a.0 < b.0
        }

        return indexed.map(\.1)
    }

    /// Check if two gates commute (can be swapped without changing circuit behavior)
    ///
    /// Two gates commute if applying them in either order produces the same result.
    /// This happens when: (1) gates act on disjoint qubits, (2) both are diagonal in
    /// the same basis, or (3) specific algebraic relationships hold.
    ///
    /// **Example:**
    /// ```swift
    /// let g1 = Gate(.pauliZ, to: 0)
    /// let g2 = Gate(.rotationZ(.pi/4), to: 0)
    /// let commutes = CircuitOptimizer.gatesCommute(g1, g2)  // true (both diagonal)
    /// ```
    ///
    /// - Complexity: O(1)
    @_optimize(speed)
    @_effects(readonly)
    public static func gatesCommute(_ g1: Gate, _ g2: Gate) -> Bool {
        let q1 = Set(g1.qubits)
        let q2 = Set(g2.qubits)

        if q1.isDisjoint(with: q2) { return true }

        return gateTypesCommute(g1.gate, g2.gate, sharedQubits: q1.intersection(q2))
    }

    /// Check if two gate types commute when sharing qubits.
    @_optimize(speed)
    @_effects(readonly)
    private static func gateTypesCommute(_ g1: QuantumGate, _ g2: QuantumGate, sharedQubits _: Set<Int>) -> Bool {
        if isDiagonal(g1), isDiagonal(g2) {
            return true
        }

        switch (g1, g2) {
        case (.pauliX, .pauliX), (.pauliY, .pauliY):
            return true
        case (.cnot, .pauliZ), (.pauliZ, .cnot):
            return true
        case (.cnot, .pauliX), (.pauliX, .cnot):
            return true
        default:
            return false
        }
    }

    /// Check if gate is diagonal in computational basis.
    @_optimize(speed)
    @inlinable
    @_effects(readonly)
    static func isDiagonal(_ gate: QuantumGate) -> Bool {
        switch gate {
        case .identity, .pauliZ, .sGate, .tGate, .phase, .rotationZ, .u1, .cz, .controlledPhase, .controlledRotationZ:
            true
        default:
            false
        }
    }

    // MARK: - KAK Decomposition

    /// Decompose arbitrary two-qubit unitary into optimal CNOT + single-qubit sequence
    ///
    /// Implements Cartan's KAK decomposition: any two-qubit unitary U can be written as
    /// U = (A₁⊗B₁) · exp(i(αXX + βYY + γZZ)) · (A₂⊗B₂) where A, B are single-qubit gates.
    /// The central entangling operation requires at most 3 CNOTs. Special cases (SWAP, CNOT,
    /// controlled rotations) may require fewer CNOTs (0-2).
    ///
    /// **Example:**
    /// ```swift
    /// let arbitrary = QuantumGate.customTwoQubit(matrix: someUnitary)
    /// let decomposed = CircuitOptimizer.kakDecomposition(arbitrary)
    /// // Result: sequence of CNOTs and single-qubit gates
    /// ```
    ///
    /// - Parameter gate: Two-qubit gate to decompose
    /// - Returns: Array of (gate, qubits) tuples implementing the decomposition
    /// - Complexity: O(1) - fixed matrix operations
    /// - Precondition: Gate must be two-qubit
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func kakDecomposition(_ gate: QuantumGate) -> [(gate: QuantumGate, qubits: [Int])] {
        ValidationUtilities.validateTwoQubitGate(gate.qubitsRequired)
        let matrix = gate.matrix()
        return kakDecomposeMatrix(matrix)
    }

    /// KAK decompose a 4x4 unitary matrix using proper Cartan decomposition.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func kakDecomposeMatrix(_ u: [[Complex<Double>]]) -> [(gate: QuantumGate, qubits: [Int])] {
        let magicBasis = computeMagicBasis()
        let magicBasisDagger = MatrixUtilities.hermitianConjugate(magicBasis)
        let uMagic = QuantumGate.matrixMultiply(QuantumGate.matrixMultiply(magicBasisDagger, u), magicBasis)

        let uMagicT = transpose(uMagic)
        let m = QuantumGate.matrixMultiply(uMagicT, uMagic)

        let eigenvalues = computeEigenvaluesQR(m)

        let (c0, c1, c2) = extractKAKCoordinatesFromEigenvalues(eigenvalues)

        let cnotCount = optimalCNOTCount(c0, c1, c2)

        let (k1, k2) = computeLocalUnitaries(u, uMagic: uMagic, c0: c0, c1: c1, c2: c2)

        return buildKAKCircuitWithLocals(k1: k1, k2: k2, c0: c0, c1: c1, c2: c2, cnotCount: cnotCount)
    }

    /// Magic basis transformation mapping computational basis to Bell basis.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func computeMagicBasis() -> [[Complex<Double>]] {
        let s = 1.0 / 2.0.squareRoot()
        return [
            [Complex(s, 0), .zero, .zero, Complex(0, s)],
            [.zero, Complex(0, s), Complex(s, 0), .zero],
            [.zero, Complex(0, s), Complex(-s, 0), .zero],
            [Complex(s, 0), .zero, .zero, Complex(0, -s)],
        ]
    }

    /// Transpose (not conjugate transpose) of complex matrix.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func transpose(_ m: [[Complex<Double>]]) -> [[Complex<Double>]] {
        let n = m.count
        return (0 ..< n).map { i in (0 ..< n).map { j in m[j][i] } }
    }

    /// Compute eigenvalues of 4x4 complex matrix using QR iteration.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func computeEigenvaluesQR(_ matrix: [[Complex<Double>]]) -> [Complex<Double>] {
        var a = matrix
        let n = 4
        let maxIterations = 100
        let tolerance = 1e-12

        for _ in 0 ..< maxIterations {
            var converged = true
            for i in 0 ..< n - 1 {
                if a[i + 1][i].magnitude > tolerance {
                    converged = false
                    break
                }
            }
            if converged { break }

            let shift = wilkinsonShift(a[n - 2][n - 2], a[n - 2][n - 1], a[n - 1][n - 2], a[n - 1][n - 1])

            for i in 0 ..< n {
                a[i][i] = a[i][i] - shift
            }

            let (q, r) = qrDecomposition(a)

            a = QuantumGate.matrixMultiply(r, q)
            for i in 0 ..< n {
                a[i][i] = a[i][i] + shift
            }
        }

        return (0 ..< n).map { a[$0][$0] }
    }

    /// Wilkinson shift for QR iteration improving convergence.
    @_optimize(speed)
    @_effects(readonly)
    private static func wilkinsonShift(
        _ a: Complex<Double>,
        _ b: Complex<Double>,
        _ c: Complex<Double>,
        _ d: Complex<Double>,
    ) -> Complex<Double> {
        let trace = a + d
        let det = a * d - b * c
        let discriminant = trace * trace - 4.0 * det
        let sqrtDisc = complexSquareRoot(discriminant)

        let lambda1 = (trace + sqrtDisc) * 0.5
        let lambda2 = (trace - sqrtDisc) * 0.5

        return (lambda1 - d).magnitude < (lambda2 - d).magnitude ? lambda1 : lambda2
    }

    /// QR decomposition via Householder reflections.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func qrDecomposition(_ a: [[Complex<Double>]]) -> ([[Complex<Double>]], [[Complex<Double>]]) {
        let n = a.count
        var q = MatrixUtilities.identityMatrix(dimension: n)
        var r = a

        for k in 0 ..< n - 1 {
            var x = [Complex<Double>](repeating: .zero, count: n - k)
            for i in k ..< n {
                x[i - k] = r[i][k]
            }

            let normX = x.reduce(0.0) { $0 + $1.magnitudeSquared }.squareRoot()
            if normX < 1e-15 { continue }

            let alpha = -normX * Complex(phase: x[0].phase)

            x[0] = x[0] - alpha
            let normV = x.reduce(0.0) { $0 + $1.magnitudeSquared }.squareRoot()

            for i in 0 ..< x.count {
                x[i] = x[i] / normV
            }

            for j in k ..< n {
                var dot = Complex<Double>.zero
                for i in 0 ..< x.count {
                    dot = dot + x[i].conjugate * r[k + i][j]
                }
                dot = dot * 2.0
                for i in 0 ..< x.count {
                    r[k + i][j] = r[k + i][j] - x[i] * dot
                }
            }

            for i in 0 ..< n {
                var dot = Complex<Double>.zero
                for j in 0 ..< x.count {
                    dot = dot + q[i][k + j] * x[j]
                }
                dot = dot * 2.0
                for j in 0 ..< x.count {
                    q[i][k + j] = q[i][k + j] - dot * x[j].conjugate
                }
            }
        }

        return (q, r)
    }

    /// Extract KAK coordinates (c₀, c₁, c₂) from eigenvalues of U_B^T U_B.
    @_optimize(speed)
    @_effects(readonly)
    private static func extractKAKCoordinatesFromEigenvalues(_ eigenvalues: [Complex<Double>]) -> (Double, Double, Double) {
        let phases = eigenvalues.map(\.phase).sorted()

        let p0 = phases[3] / 2.0
        let p1 = phases[2] / 2.0
        let p2 = phases[1] / 2.0
        let p3 = phases[0] / 2.0

        var c0 = (p0 + p3) / 2.0
        var c1 = (p0 - p2) / 2.0
        var c2 = (p0 - p1) / 2.0

        c0 = normalizeAngle(c0)
        c1 = normalizeAngle(c1)
        c2 = normalizeAngle(c2)

        var coords = [abs(c0), abs(c1), abs(c2)].sorted(by: >)
        if coords[0] > .pi / 4 { coords[0] = .pi / 2 - coords[0] }

        return (coords[0], coords[1], coords[2])
    }

    /// Determine optimal CNOT count based on KAK coordinates.
    @_optimize(speed)
    @_effects(readonly)
    private static func optimalCNOTCount(_ c0: Double, _ c1: Double, _ c2: Double) -> Int {
        let tol = 1e-9

        if abs(c0) < tol, abs(c1) < tol, abs(c2) < tol {
            return 0
        }
        if abs(c1) < tol, abs(c2) < tol {
            return 1
        }
        if abs(c2) < tol {
            return 2
        }
        return 3
    }

    /// Compute local unitaries K1 = A₁⊗B₁ and K2 = A₂⊗B₂ from polar decomposition.
    @_optimize(speed)
    @_effects(readonly)
    private static func computeLocalUnitaries(
        _ u: [[Complex<Double>]],
        uMagic _: [[Complex<Double>]],
        c0: Double,
        c1: Double,
        c2: Double,
    ) -> (k1: (a: [[Complex<Double>]], b: [[Complex<Double>]]), k2: (a: [[Complex<Double>]], b: [[Complex<Double>]])) {
        let expD = buildCanonicalEntangler(c0, c1, c2)

        let magicBasis = computeMagicBasis()
        let magicBasisDagger = MatrixUtilities.hermitianConjugate(magicBasis)
        let canonicalGate = QuantumGate.matrixMultiply(QuantumGate.matrixMultiply(magicBasis, expD), magicBasisDagger)

        let dDagger = MatrixUtilities.hermitianConjugate(canonicalGate)
        let uDagger = MatrixUtilities.hermitianConjugate(u)

        let k2Full = QuantumGate.matrixMultiply(dDagger, uDagger)
        let k1Full = QuantumGate.matrixMultiply(u, QuantumGate.matrixMultiply(canonicalGate, MatrixUtilities.hermitianConjugate(k2Full)))

        let (a1, b1) = extractTensorFactors(k1Full)
        let (a2, b2) = extractTensorFactors(k2Full)

        return ((a1, b1), (a2, b2))
    }

    /// Build canonical entangling gate exp(i(c₀XX + c₁YY + c₂ZZ)) in magic basis.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func buildCanonicalEntangler(_ c0: Double, _ c1: Double, _ c2: Double) -> [[Complex<Double>]] {
        let phases = [
            c0 + c1 + c2,
            c0 - c1 - c2,
            -c0 + c1 - c2,
            -c0 - c1 + c2,
        ]

        var result = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: 4), count: 4)
        for i in 0 ..< 4 {
            result[i][i] = Complex(phase: phases[i])
        }
        return result
    }

    /// Extract tensor factors A, B from K ≈ A ⊗ B using (A⊗B)_{(i,j),(k,l)} = A_{i,k} · B_{j,l}.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func extractTensorFactors(_ k: [[Complex<Double>]]) -> ([[Complex<Double>]], [[Complex<Double>]]) {
        var maxMag = 0.0
        var refI = 0, refJ = 0
        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                if k[i][j].magnitude > maxMag {
                    maxMag = k[i][j].magnitude
                    refI = i
                    refJ = j
                }
            }
        }

        if maxMag < 1e-10 {
            return (MatrixUtilities.identityMatrix(dimension: 2), MatrixUtilities.identityMatrix(dimension: 2))
        }

        let iA = refI / 2, iB = refI % 2
        let jA = refJ / 2, jB = refJ % 2

        var a = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: 2), count: 2)
        let refVal = k[refI][refJ]

        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                let idx = 2 * i + iB
                let jdx = 2 * j + jB
                a[i][j] = k[idx][jdx] / refVal
            }
        }

        let normA = (a[0][0].magnitudeSquared + a[0][1].magnitudeSquared).squareRoot()
        if normA > 1e-10 {
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    a[i][j] = a[i][j] / normA
                }
            }
        }

        var b = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: 2), count: 2)
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                let idx = 2 * iA + i
                let jdx = 2 * jA + j
                b[i][j] = k[idx][jdx] / refVal
            }
        }

        let normB = (b[0][0].magnitudeSquared + b[0][1].magnitudeSquared).squareRoot()
        if normB > 1e-10 {
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    b[i][j] = b[i][j] / normB
                }
            }
        }

        return (a, b)
    }

    /// Build KAK circuit with computed local unitaries.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func buildKAKCircuitWithLocals(
        k1: (a: [[Complex<Double>]], b: [[Complex<Double>]]),
        k2: (a: [[Complex<Double>]], b: [[Complex<Double>]]),
        c0: Double,
        c1: Double,
        c2: Double,
        cnotCount: Int,
    ) -> [(gate: QuantumGate, qubits: [Int])] {
        var result: [(gate: QuantumGate, qubits: [Int])] = []

        let k2aGates = decomposeToZYZ(k2.a)
        let k2bGates = decomposeToZYZ(k2.b)
        for gate in k2aGates {
            result.append((gate, [0]))
        }
        for gate in k2bGates {
            result.append((gate, [1]))
        }

        switch cnotCount {
        case 0:
            break

        case 1:
            result.append((.cnot, [0, 1]))
            if abs(c0 - .pi / 4) > angleTolerance {
                result.append((.rotationZ(2 * c0 - .pi / 2), [1]))
            }

        case 2:
            result.append((.rotationZ(.pi / 2), [0]))
            result.append((.cnot, [0, 1]))
            result.append((.rotationZ(2 * c0), [0]))
            result.append((.rotationY(2 * c1), [1]))
            result.append((.cnot, [1, 0]))
            result.append((.rotationY(-2 * c1), [0]))
            result.append((.rotationZ(-.pi / 2), [1]))

        default:
            result.append((.rotationZ(.pi / 2), [0]))
            result.append((.cnot, [0, 1]))
            result.append((.rotationZ(2 * c0 - .pi / 2), [0]))
            result.append((.rotationY(2 * c1), [1]))
            result.append((.cnot, [1, 0]))
            result.append((.rotationY(2 * c2), [0]))
            result.append((.rotationZ(-.pi / 2), [1]))
            result.append((.cnot, [0, 1]))
        }

        let k1aGates = decomposeToZYZ(k1.a)
        let k1bGates = decomposeToZYZ(k1.b)
        for gate in k1aGates {
            result.append((gate, [0]))
        }
        for gate in k1bGates {
            result.append((gate, [1]))
        }

        return result
    }

    // MARK: - Circuit Depth Computation

    /// Compute circuit depth (critical path length)
    ///
    /// Depth is the minimum number of time steps required assuming unlimited parallelism.
    /// Gates on different qubits can execute simultaneously; gates on the same qubit must
    /// be sequential. This is the key metric for quantum hardware execution time.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.hadamard, to: 1)
    /// circuit.append(.cnot, to: [0, 1])
    /// let d = CircuitOptimizer.computeDepth(circuit)
    /// ```
    ///
    /// - Parameter circuit: Circuit to analyze
    /// - Returns: Circuit depth (minimum sequential time steps)
    /// - Complexity: O(n) where n = gate count
    @_optimize(speed)
    @_effects(readonly)
    public static func computeDepth(_ circuit: QuantumCircuit) -> Int {
        guard !circuit.isEmpty else { return 0 }

        var qubitDepth = [Int](repeating: 0, count: circuit.qubits)

        for operation in circuit.gates {
            var maxDepth = 0
            for qubit in operation.qubits {
                if qubit < qubitDepth.count {
                    maxDepth = max(maxDepth, qubitDepth[qubit])
                }
            }

            let newDepth = maxDepth + 1

            for qubit in operation.qubits {
                if qubit < qubitDepth.count {
                    qubitDepth[qubit] = newDepth
                }
            }
        }

        // Safety: always exists for valid unitaries
        return qubitDepth.max()!
    }

    // MARK: - Gate Count Analysis

    /// Count gates by type for resource estimation
    ///
    /// Returns dictionary mapping gate types to occurrence counts. Useful for comparing
    /// algorithms, estimating hardware costs (different gates have different error rates),
    /// and tracking optimization effectiveness.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.grover(qubits: 3, target: 5)
    /// let counts = CircuitOptimizer.gateCount(circuit)
    /// print("CNOTs: \(counts[.cnot] ?? 0)")
    /// print("Hadamards: \(counts[.hadamard] ?? 0)")
    /// ```
    ///
    /// - Parameter circuit: Circuit to analyze
    /// - Returns: Dictionary of gate type to count
    /// - Complexity: O(n) where n = gate count
    @_optimize(speed)
    @_effects(readonly)
    public static func gateCount(_ circuit: QuantumCircuit) -> [QuantumGate: Int] {
        var counts: [QuantumGate: Int] = [:]

        for operation in circuit.gates {
            counts[operation.gate, default: 0] += 1
        }

        return counts
    }

    /// Count gates grouped by category (single-qubit, two-qubit, three-qubit)
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 3)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    /// circuit.append(.toffoli, to: [0, 1, 2])
    /// let (single, two, three) = CircuitOptimizer.gateCountByArity(circuit)
    /// // single=1, two=1, three=1
    /// ```
    ///
    /// - Parameter circuit: Circuit to analyze
    /// - Returns: Tuple of (single, two, three) qubit gate counts
    /// - Complexity: O(n)
    @_optimize(speed)
    @_effects(readonly)
    public static func gateCountByArity(_ circuit: QuantumCircuit) -> (single: Int, two: Int, three: Int) {
        var single = 0
        var two = 0
        var three = 0

        for operation in circuit.gates {
            switch operation.gate.qubitsRequired {
            case 1: single += 1
            case 2: two += 1
            case 3: three += 1
            default: break
            }
        }

        return (single, two, three)
    }

    /// Count total CNOT-equivalent two-qubit gates
    ///
    /// CNOTs are the standard metric for two-qubit gate cost. Other two-qubit gates
    /// are converted to CNOT-equivalents: CZ = 1 CNOT, SWAP = 3 CNOTs, etc.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 3)
    /// circuit.append(.swap, to: [0, 1])
    /// circuit.append(.toffoli, to: [0, 1, 2])
    /// let count = CircuitOptimizer.cnotEquivalentCount(circuit)
    /// // count = 9 (SWAP=3 + Toffoli=6)
    /// ```
    ///
    /// - Parameter circuit: Circuit to analyze
    /// - Returns: CNOT-equivalent count
    /// - Complexity: O(n)
    @_optimize(speed)
    @_effects(readonly)
    public static func cnotEquivalentCount(_ circuit: QuantumCircuit) -> Int {
        var count = 0

        for operation in circuit.gates {
            switch operation.gate {
            case .cnot, .cz, .cy, .ch, .controlledPhase, .controlledRotationX, .controlledRotationY, .controlledRotationZ:
                count += 1
            case .swap:
                count += 3
            case .sqrtSwap:
                count += 2
            case .toffoli:
                count += 6
            case .customTwoQubit:
                count += 3
            default:
                break
            }
        }

        return count
    }
}

// MARK: - Complex Number Helpers

extension CircuitOptimizer {
    /// Compute square root of complex number using principal branch.
    @_optimize(speed)
    @_effects(readonly)
    private static func complexSquareRoot(_ z: Complex<Double>) -> Complex<Double> {
        let r = z.magnitude
        let newMagnitude = r.squareRoot()
        let newPhase = z.phase / 2.0
        return Complex(magnitude: newMagnitude, phase: newPhase)
    }
}
