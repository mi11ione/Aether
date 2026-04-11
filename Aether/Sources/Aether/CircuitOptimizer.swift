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
/// - SeeAlso: ``CircuitOperation``
public enum CircuitOptimizer {
    // MARK: - Angle Tolerance

    /// Tolerance for angle comparisons in rotation gate cancellation
    @usableFromInline static let angleTolerance: Double = 1e-10

    /// Tolerance for eigenvalue convergence in QR iteration
    @usableFromInline static let eigenvalueTolerance: Double = 1e-12

    /// Tolerance for near-zero norm in QR decomposition Householder reflections
    @usableFromInline static let qrTolerance: Double = 1e-15

    /// Tolerance for KAK coordinate comparison in optimal CNOT count determination
    @usableFromInline static let cnotCountTolerance: Double = 1e-9

    /// Tolerance for norm comparison in tensor factor normalization
    @usableFromInline static let normTolerance: Double = 1e-10

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
    /// - Complexity: O(n^2) where n = gate count (dominated by commutation pass)
    /// - Precondition: Circuit must contain only concrete parameters
    @_optimize(speed)
    @_eagerMove
    public static func optimize(_ circuit: QuantumCircuit) -> QuantumCircuit {
        var result = circuit

        result = cancelIdentityPairs(result)
        result = mergeSingleQubitGates(result)
        result = reorderByCommutation(result)
        result = cancelIdentityPairs(result)
        result = mergeSingleQubitGates(result)

        return result
    }

    // MARK: - Identity Pair Cancellation

    /// Remove adjacent gate pairs that multiply to identity
    ///
    /// Scans circuit for consecutive gates on the same qubits where G1G2 = I, removing both.
    /// Handles Hermitian gates (H-H, X-X, Y-Y, Z-Z, SWAP-SWAP), rotation inverses (Rz(theta)Rz(-theta)),
    /// and phase cancellations (S-S dagger, T-T dagger). Single linear pass with in-place cancellation.
    /// Non-gate operations (such as reset) are passed through unchanged.
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
    /// - Complexity: O(n) where n = operation count
    @_optimize(speed)
    @_eagerMove
    public static func cancelIdentityPairs(_ circuit: QuantumCircuit) -> QuantumCircuit {
        guard circuit.count > 1 else { return circuit }

        var ops: [CircuitOperation] = []
        ops.reserveCapacity(circuit.count)

        for operation in circuit.operations {
            guard let gate = operation.gate else {
                ops.append(operation)
                continue
            }

            if let last = ops.last,
               let lastGate = last.gate,
               last.qubits == operation.qubits,
               gatesFormIdentity(lastGate, gate)
            {
                ops.removeLast()
            } else {
                ops.append(operation)
            }
        }

        return QuantumCircuit(qubits: circuit.qubits, operations: ops)
    }

    /// Check if two gates multiply to identity when applied sequentially.
    ///
    /// Recognizes self-inverse gates (Pauli, Hadamard, SWAP, CNOT, CZ, CY, Toffoli, CCZ),
    /// rotation angle cancellation, and controlled-phase inverse pairs.
    ///
    /// **Example:**
    /// ```swift
    /// let cancel = CircuitOptimizer.gatesFormIdentity(.hadamard, .hadamard)
    /// // cancel == true (Hadamard is self-inverse)
    /// ```
    ///
    /// - Parameter g1: First gate in sequence.
    /// - Parameter g2: Second gate in sequence.
    /// - Returns: `true` if applying g1 then g2 produces the identity operation.
    /// - Complexity: O(1)
    @_optimize(speed)
    @inline(__always)
    @inlinable
    @_effects(readonly)
    public static func gatesFormIdentity(_ g1: QuantumGate, _ g2: QuantumGate) -> Bool {
        if g1 == g2 {
            switch g1 {
            case .identity, .pauliX, .pauliY, .pauliZ, .hadamard, .swap, .cnot, .cz, .cy, .toffoli, .ccz:
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
        case let (.xx(theta1), .xx(theta2)):
            return anglesCancel(theta1, theta2)
        case let (.yy(theta1), .yy(theta2)):
            return anglesCancel(theta1, theta2)
        case let (.zz(theta1), .zz(theta2)):
            return anglesCancel(theta1, theta2)
        case let (.globalPhase(p1), .globalPhase(p2)):
            return anglesCancel(p1, p2)
        default:
            return false
        }
    }

    /// Check if a parameter value equals a target angle within tolerance.
    ///
    /// **Example:**
    /// ```swift
    /// let equal = CircuitOptimizer.isAngleEqual(.value(3.14159), .pi)
    /// // equal == true (within angleTolerance)
    /// ```
    ///
    /// - Parameter angle: Parameter value to check.
    /// - Parameter target: Target angle in radians.
    /// - Returns: `true` if the angle equals the target within `angleTolerance`.
    /// - Complexity: O(1)
    @_optimize(speed)
    @inline(__always)
    @inlinable
    @_effects(readonly)
    public static func isAngleEqual(_ angle: ParameterValue, _ target: Double) -> Bool {
        guard case let .value(v) = angle else { return false }
        return abs(v - target) < angleTolerance
    }

    /// Check if two parameter angles sum to zero (or a multiple of 2π) within tolerance.
    ///
    /// **Example:**
    /// ```swift
    /// let cancel = CircuitOptimizer.anglesCancel(.value(.pi / 2), .value(-.pi / 2))
    /// // cancel == true
    /// ```
    ///
    /// - Parameter a1: First angle.
    /// - Parameter a2: Second angle.
    /// - Returns: `true` if the angles cancel each other.
    /// - Complexity: O(1)
    @_optimize(speed)
    @inline(__always)
    @inlinable
    @_effects(readonly)
    public static func anglesCancel(_ a1: ParameterValue, _ a2: ParameterValue) -> Bool {
        guard case let .value(v1) = a1, case let .value(v2) = a2 else { return false }
        return abs(v1 + v2) < angleTolerance
    }

    // MARK: - Single-Qubit Gate Merging

    /// Merge consecutive single-qubit gates on the same qubit into minimal form
    ///
    /// Combines adjacent single-qubit rotations and gates into optimally decomposed sequences.
    /// Same-axis rotations merge by adding angles: Rz(theta1)Rz(theta2) = Rz(theta1+theta2). Different-axis
    /// rotations fuse via matrix multiplication into U3(theta,phi,lambda). Gates with near-identity
    /// angles (|theta| < epsilon) are removed entirely.
    /// Non-gate operations (such as reset) are passed through unchanged and break merge sequences.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 1)
    /// circuit.append(.rotationZ(.pi/4), to: 0)
    /// circuit.append(.rotationZ(.pi/4), to: 0)
    /// let optimized = CircuitOptimizer.mergeSingleQubitGates(circuit)
    /// // Result: single Rz(pi/2)
    /// ```
    ///
    /// - Parameter circuit: Input circuit
    /// - Returns: Circuit with merged single-qubit sequences
    /// - Complexity: O(n) where n = operation count
    @_optimize(speed)
    @_eagerMove
    public static func mergeSingleQubitGates(_ circuit: QuantumCircuit) -> QuantumCircuit {
        guard circuit.count > 1 else { return circuit }

        var ops: [CircuitOperation] = []
        ops.reserveCapacity(circuit.count)

        var i = 0
        while i < circuit.operations.count {
            let operation = circuit.operations[i]

            guard let gate = operation.gate, gate.qubitsRequired == 1 else {
                ops.append(operation)
                i += 1
                continue
            }

            let qubit = operation.qubits[0]
            var sequence: [QuantumGate] = [gate]
            var j = i + 1

            while j < circuit.operations.count {
                let next = circuit.operations[j]
                guard let nextGate = next.gate,
                      nextGate.qubitsRequired == 1,
                      next.qubits[0] == qubit
                else {
                    break
                }
                sequence.append(nextGate)
                j += 1
            }

            if sequence.count > 1 {
                let merged = mergeGateSequence(sequence)
                for mergedGate in merged {
                    ops.append(.gate(mergedGate, qubits: [qubit]))
                }
            } else {
                ops.append(operation)
            }

            i = j
        }

        return QuantumCircuit(qubits: circuit.qubits, operations: ops)
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

    /// Extract the concrete rotation angle from a rotation gate.
    @_optimize(speed)
    @inline(__always)
    @_effects(readonly)
    private static func rotationAngle(_ gate: QuantumGate) -> Double? {
        let param: ParameterValue
        switch gate {
        case let .rotationX(t): param = t
        case let .rotationY(t): param = t
        case let .rotationZ(t): param = t
        case let .phase(t): param = t
        case let .globalPhase(t): param = t
        default: return nil
        }
        if case let .value(v) = param { return v }
        return nil
    }

    /// Check if two gates are the same rotation type.
    @_optimize(speed)
    @inline(__always)
    @_effects(readonly)
    private static func sameRotationType(_ a: QuantumGate, _ b: QuantumGate) -> Bool {
        switch (a, b) {
        case (.rotationX, .rotationX): true
        case (.rotationY, .rotationY): true
        case (.rotationZ, .rotationZ): true
        case (.phase, .phase): true
        case (.globalPhase, .globalPhase): true
        default: false
        }
    }

    /// Try to merge gates if all are same-type rotations.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func trySameTypeRotationMerge(_ gates: [QuantumGate]) -> [QuantumGate]? {
        let first = gates[0]
        var totalAngle: Double = 0

        for gate in gates {
            guard let angle = rotationAngle(gate), sameRotationType(first, gate) else { return nil }
            totalAngle += angle
        }

        totalAngle = normalizeAngle(totalAngle)
        if abs(totalAngle) < angleTolerance { return [] }

        return switch first {
        case .rotationX: [.rotationX(totalAngle)]
        case .rotationY: [.rotationY(totalAngle)]
        case .rotationZ: [.rotationZ(totalAngle)]
        case .phase: [.phase(totalAngle)]
        case .globalPhase: [.globalPhase(totalAngle)]
        default: nil
        }
    }

    /// Normalize an angle to the range [-π, π].
    ///
    /// **Example:**
    /// ```swift
    /// let normalized = CircuitOptimizer.normalizeAngle(3 * .pi)
    /// // normalized ≈ .pi
    /// ```
    ///
    /// - Parameter angle: Angle in radians.
    /// - Returns: Equivalent angle in the range [-π, π].
    /// - Complexity: O(1)
    @_optimize(speed)
    @inline(__always)
    @inlinable
    @_effects(readonly)
    public static func normalizeAngle(_ angle: Double) -> Double {
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

        let theta = 2.0 * acos(max(0.0, min(1.0, a.magnitude)))

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
    /// Returns single U3(theta,phi,lambda) gate or U1(lambda) for pure Z rotations.
    /// Optimal single-gate representation for hardware execution.
    ///
    /// **Example:**
    /// ```swift
    /// let matrix = QuantumGate.hadamard.matrix()
    /// let gate = CircuitOptimizer.decompose(matrix)
    /// // Returns U3 gate equivalent to Hadamard
    /// ```
    ///
    /// - Parameter matrix: 2x2 unitary matrix
    /// - Returns: Single U3 or U1 gate
    /// - Complexity: O(1) — fixed-size matrix operations on 2×2 input.
    /// - Precondition: Matrix must be 2×2 unitary.
    @_optimize(speed)
    public static func decompose(_ matrix: [[Complex<Double>]]) -> QuantumGate {
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
    /// Non-gate operations (such as reset) are treated as non-commuting for safety.
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
    /// - Returns: Circuit with operations reordered for optimization opportunities
    /// - Complexity: O(n^2) where n = operation count
    @_optimize(speed)
    @_eagerMove
    public static func reorderByCommutation(_ circuit: QuantumCircuit) -> QuantumCircuit {
        guard circuit.count > 2 else { return circuit }

        var ops = circuit.operations

        ops = bringInversesAdjacent(ops)
        ops = minimizeDepthOrdering(ops, qubitCount: circuit.qubits)

        return QuantumCircuit(qubits: circuit.qubits, operations: ops)
    }

    /// Move operations to bring inverse pairs adjacent for cancellation.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func bringInversesAdjacent(_ ops: [CircuitOperation]) -> [CircuitOperation] {
        var result = ops
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

                    guard let gate1 = g1.gate,
                          let gatej = gj.gate,
                          g1.qubits == gj.qubits,
                          gatesFormIdentity(gate1, gatej) else { continue }

                    var canMove = true
                    for k in stride(from: j - 1, through: i + 1, by: -1) {
                        if !operationsCommute(result[k], result[j]) {
                            canMove = false
                            break
                        }
                    }

                    if canMove {
                        let op = result[j]
                        for k in stride(from: j, to: i + 1, by: -1) {
                            result[k] = result[k - 1]
                        }
                        result[i + 1] = op
                        changed = true
                        break
                    }
                }
                if changed { break }
            }
        }

        return result
    }

    /// Reorder operations to minimize circuit depth using as-soon-as-possible scheduling.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func minimizeDepthOrdering(_ ops: [CircuitOperation], qubitCount: Int) -> [CircuitOperation] {
        var qubitLastOp = [Int](repeating: -1, count: qubitCount)
        var opDepth = [Int](repeating: 0, count: ops.count)

        for (i, operation) in ops.enumerated() {
            var maxPrevDepth = 0
            for q in operation.qubits {
                if q < qubitLastOp.count, qubitLastOp[q] >= 0 {
                    maxPrevDepth = max(maxPrevDepth, opDepth[qubitLastOp[q]])
                }
            }
            opDepth[i] = maxPrevDepth + 1
            for q in operation.qubits {
                if q < qubitLastOp.count {
                    qubitLastOp[q] = i
                }
            }
        }

        var indices = Array(0 ..< ops.count)
        indices.sort { a, b in
            if opDepth[a] != opDepth[b] { return opDepth[a] < opDepth[b] }
            return a < b
        }

        return indices.map { ops[$0] }
    }

    /// Check if two circuit operations commute (can be swapped without changing circuit behavior)
    ///
    /// Two operations commute if applying them in either order produces the same result.
    /// This happens when: (1) operations act on disjoint qubits, (2) both are diagonal in
    /// the same basis, or (3) specific algebraic relationships hold.
    /// Non-gate operations are treated as non-commuting (conservative).
    ///
    /// **Example:**
    /// ```swift
    /// let g1 = CircuitOperation.gate(.pauliZ, qubits: [0])
    /// let g2 = CircuitOperation.gate(.rotationZ(.pi/4), qubits: [0])
    /// let commutes = CircuitOptimizer.operationsCommute(g1, g2)  // true (both diagonal)
    /// ```
    ///
    /// - Parameter op1: First circuit operation.
    /// - Parameter op2: Second circuit operation.
    /// - Returns: `true` if the operations commute and can be safely reordered.
    /// - Complexity: O(1)
    @_optimize(speed)
    @_effects(readonly)
    public static func operationsCommute(_ op1: CircuitOperation, _ op2: CircuitOperation) -> Bool {
        guard let gate1 = op1.gate,
              let gate2 = op2.gate
        else {
            return false
        }

        let qubits1 = op1.qubits
        let qubits2 = op2.qubits

        var disjoint = true
        outer: for q in qubits1 {
            for r in qubits2 {
                if q == r { disjoint = false; break outer }
            }
        }

        if disjoint { return true }

        return gateTypesCommute(gate1, gate2)
    }

    /// Check if two gate types commute when sharing qubits.
    @_optimize(speed)
    @_effects(readonly)
    private static func gateTypesCommute(_ g1: QuantumGate, _ g2: QuantumGate) -> Bool {
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

    /// Check if a gate is diagonal in the computational basis.
    ///
    /// Diagonal gates (phase, rotationZ, controlledPhase, etc.) commute with each other
    /// and with measurement operations.
    ///
    /// **Example:**
    /// ```swift
    /// let diag = CircuitOptimizer.isDiagonal(.phase(.value(.pi / 4)))
    /// // diag == true
    /// ```
    ///
    /// - Parameter gate: Gate to check.
    /// - Returns: `true` if the gate is diagonal in the computational basis.
    /// - Complexity: O(1)
    @_optimize(speed)
    @inline(__always)
    @inlinable
    @_effects(readonly)
    public static func isDiagonal(_ gate: QuantumGate) -> Bool {
        switch gate {
        case .identity, .pauliZ, .sGate, .tGate, .phase, .rotationZ, .u1, .cz, .controlledPhase, .controlledRotationZ, .zz, .ccz, .globalPhase:
            true
        default:
            false
        }
    }

    // MARK: - KAK Decomposition

    /// Decompose arbitrary two-qubit unitary into optimal CNOT + single-qubit sequence
    ///
    /// Implements Cartan's KAK decomposition: any two-qubit unitary U can be written as
    /// U = (A1 tensor B1) exp(i(alpha XX + beta YY + gamma ZZ)) (A2 tensor B2) where A, B are single-qubit gates.
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
        let uMagic = QuantumGate.matrixMultiply(QuantumGate.matrixMultiply(Self.magicBasisDagger, u), Self.magicBasis)

        let uMagicT = transpose(uMagic)
        let m = QuantumGate.matrixMultiply(uMagicT, uMagic)

        let (eigenvalues, eigenvectors) = computeEigendecompositionQR(m)

        let (c0, c1, c2) = extractKAKCoordinatesFromEigenvalues(eigenvalues)

        let cnotCount = optimalCNOTCount(c0, c1, c2)

        let (k1, k2) = computeLocalUnitariesFromEigenvectors(uMagic, eigenvectors: eigenvectors, c0: c0, c1: c1, c2: c2)

        return buildKAKCircuitWithLocals(k1: k1, k2: k2, c0: c0, c1: c1, c2: c2, cnotCount: cnotCount)
    }

    /// Magic basis transformation mapping computational basis to Bell basis (computed once).
    private static let magicBasis: [[Complex<Double>]] = {
        let s = 1.0 / 2.0.squareRoot()
        return [
            [Complex(s, 0), .zero, .zero, Complex(0, s)],
            [.zero, Complex(0, s), Complex(s, 0), .zero],
            [.zero, Complex(0, s), Complex(-s, 0), .zero],
            [Complex(s, 0), .zero, .zero, Complex(0, -s)],
        ]
    }()

    /// Hermitian conjugate of the magic basis (computed once).
    private static let magicBasisDagger: [[Complex<Double>]] = MatrixUtilities.hermitianConjugate(magicBasis)

    /// Transpose (not conjugate transpose) of complex matrix.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func transpose(_ m: [[Complex<Double>]]) -> [[Complex<Double>]] {
        let n = m.count
        return (0 ..< n).map { i in
            [Complex<Double>](unsafeUninitializedCapacity: n) { buffer, count in
                for j in 0 ..< n {
                    buffer.initializeElement(at: j, to: m[j][i])
                }
                count = n
            }
        }
    }

    /// Compute eigenvalues and eigenvectors of 4x4 complex matrix using QR iteration.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func computeEigendecompositionQR(
        _ matrix: [[Complex<Double>]],
    ) -> (eigenvalues: [Complex<Double>], eigenvectors: [[Complex<Double>]]) {
        var a = matrix
        let n = 4
        let maxIterations = 100
        let tolerance = eigenvalueTolerance

        var v = MatrixUtilities.identityMatrix(dimension: n)

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

            a[0][0] = a[0][0] - shift
            a[1][1] = a[1][1] - shift
            a[2][2] = a[2][2] - shift
            a[3][3] = a[3][3] - shift

            let (q, r) = qrDecomposition(a)

            a = QuantumGate.matrixMultiply(r, q)
            a[0][0] = a[0][0] + shift
            a[1][1] = a[1][1] + shift
            a[2][2] = a[2][2] + shift
            a[3][3] = a[3][3] + shift

            v = QuantumGate.matrixMultiply(v, q)
        }

        let eigenvalues = (0 ..< n).map { a[$0][$0] }
        return (eigenvalues, v)
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

        return (lambda1 - d).magnitudeSquared < (lambda2 - d).magnitudeSquared ? lambda1 : lambda2
    }

    /// QR decomposition via Householder reflections.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func qrDecomposition(_ a: [[Complex<Double>]]) -> ([[Complex<Double>]], [[Complex<Double>]]) {
        let n = a.count
        var q = MatrixUtilities.identityMatrix(dimension: n)
        var r = a

        var x = [Complex<Double>](repeating: .zero, count: n)

        for k in 0 ..< n - 1 {
            let xCount = n - k
            for i in 0 ..< xCount {
                x[i] = r[k + i][k]
            }

            var normXSq = 0.0
            for i in 0 ..< xCount {
                normXSq += x[i].magnitudeSquared
            }
            let normX = normXSq.squareRoot()
            if normX < qrTolerance { continue }

            let alpha = -normX * Complex(phase: x[0].phase)

            x[0] = x[0] - alpha
            var normVSq = 0.0
            for i in 0 ..< xCount {
                normVSq += x[i].magnitudeSquared
            }
            let normV = normVSq.squareRoot()

            let invNormV = 1.0 / normV
            for i in 0 ..< xCount {
                x[i] = x[i] * invNormV
            }

            for j in k ..< n {
                var dot = Complex<Double>.zero
                for i in 0 ..< xCount {
                    dot = dot + x[i].conjugate * r[k + i][j]
                }
                dot = dot * 2.0
                for i in 0 ..< xCount {
                    r[k + i][j] = r[k + i][j] - x[i] * dot
                }
            }

            for i in 0 ..< n {
                var dot = Complex<Double>.zero
                for j in 0 ..< xCount {
                    dot = dot + q[i][k + j] * x[j]
                }
                dot = dot * 2.0
                for j in 0 ..< xCount {
                    q[i][k + j] = q[i][k + j] - dot * x[j].conjugate
                }
            }
        }

        return (q, r)
    }

    /// Extract KAK coordinates (c0, c1, c2) from eigenvalues of U_B^T U_B.
    @_optimize(speed)
    @_effects(readonly)
    private static func extractKAKCoordinatesFromEigenvalues(_ eigenvalues: [Complex<Double>]) -> (Double, Double, Double) {
        var ph0 = eigenvalues[0].phase
        var ph1 = eigenvalues[1].phase
        var ph2 = eigenvalues[2].phase
        var ph3 = eigenvalues[3].phase
        if ph0 > ph1 { swap(&ph0, &ph1) }
        if ph2 > ph3 { swap(&ph2, &ph3) }
        if ph0 > ph2 { swap(&ph0, &ph2) }
        if ph1 > ph3 { swap(&ph1, &ph3) }
        if ph1 > ph2 { swap(&ph1, &ph2) }

        let p0 = ph3 / 2.0
        let p1 = ph2 / 2.0
        let p2 = ph1 / 2.0
        let p3 = ph0 / 2.0

        var c0 = (p0 + p1) / 2.0
        var c1 = (p0 + p2) / 2.0
        var c2 = (p0 + p3) / 2.0

        c0 = normalizeAngle(c0)
        c1 = normalizeAngle(c1)
        c2 = normalizeAngle(c2)

        var s0 = abs(c0), s1 = abs(c1), s2 = abs(c2)

        if s0 < s1 { swap(&s0, &s1) }
        if s1 < s2 { swap(&s1, &s2) }
        if s0 < s1 { swap(&s0, &s1) }

        if s0 > .pi / 4 { s0 = .pi / 2 - s0 }
        if s1 > .pi / 4 { s1 = .pi / 2 - s1 }
        if s2 > .pi / 4 { s2 = .pi / 2 - s2 }

        if s0 < s1 { swap(&s0, &s1) }
        if s1 < s2 { swap(&s1, &s2) }
        if s0 < s1 { swap(&s0, &s1) }

        return (s0, s1, s2)
    }

    /// Determine optimal CNOT count based on KAK coordinates.
    @_optimize(speed)
    @_effects(readonly)
    private static func optimalCNOTCount(_ c0: Double, _ c1: Double, _ c2: Double) -> Int {
        let tol = cnotCountTolerance

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

    /// Compute local unitaries from eigenvectors of U_B^T U_B.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func computeLocalUnitariesFromEigenvectors(
        _ uMagic: [[Complex<Double>]],
        eigenvectors: [[Complex<Double>]],
        c0: Double,
        c1: Double,
        c2: Double,
    ) -> (k1: (a: [[Complex<Double>]], b: [[Complex<Double>]]), k2: (a: [[Complex<Double>]], b: [[Complex<Double>]])) {
        let expD = buildCanonicalEntangler(c0, c1, c2)
        let expDDagger = MatrixUtilities.hermitianConjugate(expD)

        let k2Magic = eigenvectors
        let k2MagicInv = MatrixUtilities.hermitianConjugate(k2Magic)

        let k1Magic = QuantumGate.matrixMultiply(
            QuantumGate.matrixMultiply(uMagic, k2MagicInv),
            expDDagger,
        )

        let k2Full = QuantumGate.matrixMultiply(
            QuantumGate.matrixMultiply(Self.magicBasis, k2Magic),
            Self.magicBasisDagger,
        )
        let k1Full = QuantumGate.matrixMultiply(
            QuantumGate.matrixMultiply(Self.magicBasis, k1Magic),
            Self.magicBasisDagger,
        )

        let (a1, b1) = extractTensorFactors(k1Full)
        let (a2, b2) = extractTensorFactors(k2Full)

        return ((a1, b1), (a2, b2))
    }

    /// Build canonical entangling gate exp(i(c0 XX + c1 YY + c2 ZZ)) in magic basis.
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

        return [
            [Complex(phase: phases[0]), .zero, .zero, .zero],
            [.zero, Complex(phase: phases[1]), .zero, .zero],
            [.zero, .zero, Complex(phase: phases[2]), .zero],
            [.zero, .zero, .zero, Complex(phase: phases[3])],
        ]
    }

    /// Extract tensor factors A, B from K approximately equal to A tensor B using (A tensor B)_{(i,j),(k,l)} = A_{i,k} B_{j,l}.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func extractTensorFactors(_ k: [[Complex<Double>]]) -> ([[Complex<Double>]], [[Complex<Double>]]) {
        var maxMag = 0.0
        var refI = 0, refJ = 0
        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                if k[i][j].magnitudeSquared > maxMag {
                    maxMag = k[i][j].magnitudeSquared
                    refI = i
                    refJ = j
                }
            }
        }

        let iA = refI / 2, iB = refI % 2
        let jA = refJ / 2, jB = refJ % 2

        let refVal = k[refI][refJ]

        var a: [[Complex<Double>]] = [
            [k[iB][jB] / refVal, k[iB][2 + jB] / refVal],
            [k[2 + iB][jB] / refVal, k[2 + iB][2 + jB] / refVal],
        ]

        let normA = hypot(a[0][0].magnitude, a[0][1].magnitude)
        if normA > normTolerance {
            let invNormA = 1.0 / normA
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    a[i][j] = a[i][j] * invNormA
                }
            }
        }

        var b: [[Complex<Double>]] = [
            [k[2 * iA][2 * jA] / refVal, k[2 * iA][2 * jA + 1] / refVal],
            [k[2 * iA + 1][2 * jA] / refVal, k[2 * iA + 1][2 * jA + 1] / refVal],
        ]

        let normB = hypot(b[0][0].magnitude, b[0][1].magnitude)
        if normB > normTolerance {
            let invNormB = 1.0 / normB
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    b[i][j] = b[i][j] * invNormB
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
    /// Operations on different qubits can execute simultaneously; operations on the same qubit must
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
    /// - Complexity: O(n) where n = operation count
    @_optimize(speed)
    @_effects(readonly)
    public static func computeDepth(_ circuit: QuantumCircuit) -> Int {
        guard !circuit.isEmpty else { return 0 }

        var qubitDepth = [Int](repeating: 0, count: circuit.qubits)
        var globalMaxDepth = 0

        for operation in circuit.operations {
            var maxDepth = 0
            for qubit in operation.qubits {
                if qubit < qubitDepth.count {
                    maxDepth = max(maxDepth, qubitDepth[qubit])
                }
            }

            let newDepth = maxDepth + 1
            globalMaxDepth = max(globalMaxDepth, newDepth)

            for qubit in operation.qubits {
                if qubit < qubitDepth.count {
                    qubitDepth[qubit] = newDepth
                }
            }
        }

        return globalMaxDepth
    }

    // MARK: - Gate Count Analysis

    /// Count gates by type for resource estimation
    ///
    /// Returns dictionary mapping gate types to occurrence counts. Useful for comparing
    /// algorithms, estimating hardware costs (different gates have different error rates),
    /// and tracking optimization effectiveness. Only counts gate operations; non-gate
    /// operations (such as reset) are excluded.
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
    /// - Complexity: O(n) where n = operation count
    @_optimize(speed)
    @_effects(readonly)
    public static func gateCount(_ circuit: QuantumCircuit) -> [QuantumGate: Int] {
        var counts: [QuantumGate: Int] = [:]
        counts.reserveCapacity(min(circuit.count, 20))

        for operation in circuit.operations {
            guard let gate = operation.gate else { continue }
            counts[gate, default: 0] += 1
        }

        return counts
    }

    /// Count gates grouped by category (single-qubit, two-qubit, three-qubit)
    ///
    /// Only counts gate operations; non-gate operations (such as reset) are excluded.
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

        for operation in circuit.operations {
            guard let gate = operation.gate else { continue }
            switch gate.qubitsRequired {
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
    /// Only counts gate operations; non-gate operations (such as reset) are excluded.
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

        for operation in circuit.operations {
            guard let gate = operation.gate else { continue }
            switch gate {
            case .cnot, .cz, .cy, .ch, .controlledPhase, .controlledRotationX, .controlledRotationY, .controlledRotationZ:
                count += 1
            case .swap:
                count += 3
            case .sqrtSwap:
                count += 2
            case .toffoli, .ccz:
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
