// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Circuit-to-unitary matrix converter for batched GPU evaluation.
///
/// Converts quantum circuits to dense unitary matrices via sequential gate composition. A quantum circuit
/// C = G₁G₂...Gₙ is represented as unitary U = Gₙ...G₂G₁ through right-to-left composition, where each
/// gate Gᵢ is a 2ⁿ x 2ⁿ matrix with sparse structure (identity on non-target qubits). Composition uses
/// BLAS-accelerated matrix multiplication for efficiency.
///
/// Unitary conversion enables batched circuit execution on GPU through ``MPSBatchEvaluator``. The conversion
/// cost O(depth · 2³ⁿ) is paid once per circuit structure and amortized over many parameter evaluations,
/// making it efficient for VQE gradient computation, QAOA grid search, and population-based optimizers where
/// the same circuit structure is evaluated with different parameters.
///
/// Memory usage scales as 2ⁿ⁺¹ · 16 bytes per unitary matrix. Practical limits: 8 qubits = 1 MB, 10 qubits = 16 MB,
/// 12 qubits = 256 MB, 14 qubits = 4 GB. Use ``canConvert(qubits:)`` to check feasibility before conversion.
///
/// Optimal for batch sizes ≥ 10 with 5-12 qubits and moderate circuit depth (<50 gates). For single evaluations
/// or very deep circuits, gate-by-gate execution through ``QuantumSimulator`` is more efficient.
///
/// ```swift
/// let ansatz = HardwareEfficientAnsatz.create(numQubits: 8, depth: 3)
/// let circuits = (0..<100).map { i in
///     ansatz.bind(parameterVector: generateParameters(seed: i))
/// }
/// let unitaries = circuits.map { CircuitUnitary.unitary(for: $0) }
/// let evaluator = await MPSBatchEvaluator()
/// let energies = await evaluator.evaluateExpectationValues(
///     unitaries: unitaries,
///     initialState: QuantumState(numQubits: 8),
///     hamiltonian: hamiltonian
/// )
/// ```
///
/// - Note: Unitary matrices are computed using ``MatrixUtilities`` with Accelerate BLAS for optimal performance.
/// - SeeAlso: ``MPSBatchEvaluator``, ``MatrixUtilities``, ``QuantumCircuit``
public enum CircuitUnitary {
    // MARK: - Public API

    /// Compute full circuit unitary matrix via gate composition.
    ///
    /// Converts quantum circuit to dense 2ⁿ x 2ⁿ unitary matrix through sequential matrix multiplication.
    /// Starting from identity matrix I, each gate G in the circuit is expanded to full Hilbert space via
    /// tensor product (single-qubit: I ⊗ ... ⊗ G ⊗ ... ⊗ I, two-qubit and Toffoli similar) and composed
    /// via right-multiplication: U ← G · U. Gate expansion uses direct index computation rather than full
    /// tensor product construction for efficiency.
    ///
    /// ```swift
    /// var circuit = QuantumCircuit(numQubits: 2)
    /// circuit.append(gate: .hadamard, toQubit: 0)
    /// circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
    /// let unitary = CircuitUnitary.unitary(for: circuit)
    /// ```
    ///
    /// - Parameter circuit: Quantum circuit to convert.
    /// - Returns: Dense unitary matrix as 2ⁿ x 2ⁿ complex matrix.
    /// - Complexity: O(depth · 2³ⁿ) where depth is number of gates. Per-gate cost includes O(2²ⁿ) expansion and O(2³ⁿ) matrix multiply.
    /// - Precondition: Circuit must have at least 1 qubit.
    /// - Note: Does not validate unitarity of result. Assumes circuit gates are valid unitary operations.
    /// - SeeAlso: ``MatrixUtilities`` for BLAS-accelerated composition, ``MPSBatchEvaluator`` for batched evaluation.
    @_optimize(speed)
    @_eagerMove
    public static func unitary(for circuit: QuantumCircuit) -> [[Complex<Double>]] {
        let numQubits: Int = circuit.numQubits
        ValidationUtilities.validatePositiveQubits(numQubits)
        ValidationUtilities.validateMemoryLimit(numQubits)

        let dimension = 1 << numQubits
        var unitary: [[Complex<Double>]] = MatrixUtilities.identityMatrix(dimension: dimension)

        for operation in circuit.gates {
            let gateMatrix: [[Complex<Double>]] = expandGateToFullSpace(
                gate: operation.gate,
                qubits: operation.qubits,
                numQubits: numQubits
            )

            unitary = MatrixUtilities.matrixMultiply(gateMatrix, unitary)
        }

        return unitary
    }

    // MARK: - Gate Expansion

    /// Expand gate to full Hilbert space.
    ///
    /// Embeds small gate matrix (2x2, 4x4, or 8x8) into full 2ⁿ x 2ⁿ space via tensor product with identity
    /// on non-target qubits. Uses direct index computation rather than explicit tensor product construction:
    /// single-qubit gates check target bit match, two-qubit gates check control and target bits.
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to expand.
    ///   - qubits: Qubit indices for gate application.
    ///   - numQubits: Total number of qubits in circuit.
    /// - Returns: Full 2ⁿ x 2ⁿ matrix.
    /// - Complexity: O(2²ⁿ) to fill full matrix.
    @_optimize(speed)
    @_eagerMove
    private static func expandGateToFullSpace(
        gate: QuantumGate,
        qubits: [Int],
        numQubits: Int
    ) -> [[Complex<Double>]] {
        let dimension = 1 << numQubits

        switch gate {
        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard,
             .phase, .sGate, .tGate, .rotationX, .rotationY, .rotationZ,
             .u1, .u2, .u3, .sx, .sy, .customSingleQubit:
            let targetQubit: Int = qubits[0]
            return expandSingleQubitGate(gate: gate, targetQubit: targetQubit, dimension: dimension)

        case .cnot, .cz, .cy, .ch, .controlledPhase, .controlledRotationX, .controlledRotationY, .controlledRotationZ, .swap, .sqrtSwap, .customTwoQubit:
            return expandTwoQubitGate(gate: gate, control: qubits[0], target: qubits[1], dimension: dimension)

        case .toffoli:
            return expandToffoliGate(control1: qubits[0], control2: qubits[1], target: qubits[2], dimension: dimension)
        }
    }

    /// Expand single-qubit gate to full space.
    ///
    /// Directly computes valid indices rather than iterating all dimension² pairs. For single-qubit gates,
    /// only pairs where non-target bits match have non-zero entries. For each "other bits" pattern
    /// (dimension/2 patterns), the 2x2 gate contributes 4 entries to the full matrix.
    ///
    /// Example: Hadamard on qubit 0 of 2-qubit system produces 4x4 matrix where row 0 (basis state |00⟩)
    /// maps |0⟩ -> (|0⟩+|1⟩)/√2 giving [1/√2, 1/√2, 0, 0], and similarly for other basis states.
    ///
    /// - Complexity: O(2ⁿ⁺²) vs O(2²ⁿ) for naive approach.
    @_optimize(speed)
    @_eagerMove
    private static func expandSingleQubitGate(
        gate: QuantumGate,
        targetQubit: Int,
        dimension: Int
    ) -> [[Complex<Double>]] {
        let smallMatrix: [[Complex<Double>]] = gate.matrix()
        var fullMatrix: [[Complex<Double>]] = Array(
            repeating: Array(repeating: Complex<Double>.zero, count: dimension),
            count: dimension
        )

        let targetMask = 1 << targetQubit

        for otherBits in 0 ..< dimension {
            guard (otherBits & targetMask) == 0 else { continue }

            let base0 = otherBits
            let base1 = otherBits | targetMask

            fullMatrix[base0][base0] = smallMatrix[0][0]
            fullMatrix[base0][base1] = smallMatrix[0][1]
            fullMatrix[base1][base0] = smallMatrix[1][0]
            fullMatrix[base1][base1] = smallMatrix[1][1]
        }

        return fullMatrix
    }

    /// Expand two-qubit gate to full space.
    ///
    /// Directly computes valid indices for two-qubit gates. Only pairs where non-control and non-target bits
    /// match have non-zero entries. For each "other bits" pattern (dimension/4 patterns), the 4x4 gate matrix
    /// contributes 16 entries. Gate matrix indices 0-3 map to control-target bit pairs (0,0), (0,1), (1,0), (1,1).
    ///
    /// - Complexity: O(2ⁿ⁺²) vs O(2²ⁿ) for naive approach.
    @_optimize(speed)
    @_eagerMove
    private static func expandTwoQubitGate(
        gate: QuantumGate,
        control: Int,
        target: Int,
        dimension: Int
    ) -> [[Complex<Double>]] {
        let smallMatrix: [[Complex<Double>]] = gate.matrix()
        var fullMatrix: [[Complex<Double>]] = Array(
            repeating: Array(repeating: Complex<Double>.zero, count: dimension),
            count: dimension
        )

        let controlMask = 1 << control
        let targetMask = 1 << target
        let bothMask: Int = controlMask | targetMask

        for otherBits in 0 ..< dimension {
            guard (otherBits & bothMask) == 0 else { continue }

            let base00 = otherBits
            let base01 = otherBits | targetMask
            let base10 = otherBits | controlMask
            let base11 = otherBits | bothMask

            let bases = [base00, base01, base10, base11]

            for rowIdx in 0 ..< 4 {
                let row = bases[rowIdx]
                for colIdx in 0 ..< 4 {
                    let col = bases[colIdx]
                    fullMatrix[row][col] = smallMatrix[rowIdx][colIdx]
                }
            }
        }

        return fullMatrix
    }

    /// Expand Toffoli gate to full space.
    ///
    /// Constructs Toffoli matrix directly: if both control qubits are 1, flip target (|c1,c2,t⟩ -> |c1,c2,t⊕1⟩),
    /// otherwise identity. Most entries are diagonal (1), with off-diagonal entries only in rows where both
    /// controls equal 1.
    ///
    /// - Complexity: O(2ⁿ) to construct matrix.
    @_optimize(speed)
    @_eagerMove
    private static func expandToffoliGate(
        control1: Int,
        control2: Int,
        target: Int,
        dimension: Int
    ) -> [[Complex<Double>]] {
        var fullMatrix: [[Complex<Double>]] = Array(
            repeating: Array(repeating: Complex<Double>.zero, count: dimension),
            count: dimension
        )

        let c1Mask = 1 << control1
        let c2Mask = 1 << control2
        let bothControlsMask = c1Mask | c2Mask
        let targetMask = 1 << target

        for row in 0 ..< dimension {
            if (row & bothControlsMask) == bothControlsMask {
                let flippedRow = row ^ targetMask
                fullMatrix[row][flippedRow] = Complex<Double>(1, 0)
            } else {
                fullMatrix[row][row] = Complex<Double>(1, 0)
            }
        }

        return fullMatrix
    }

    // MARK: - Validation Helpers

    /// Estimate memory usage for unitary matrix.
    ///
    /// Computes memory required for 2ⁿ x 2ⁿ complex matrix.
    ///
    /// ```swift
    /// let bytes = CircuitUnitary.memoryUsage(for: 10)
    /// print("\(bytes / 1_048_576) MB")
    /// ```
    ///
    /// - Parameter numQubits: Number of qubits.
    /// - Returns: Memory in bytes.
    /// - Complexity: O(1).
    @_effects(readonly)
    @inlinable
    public static func memoryUsage(for numQubits: Int) -> Int {
        let dimension = 1 << numQubits
        let complexSize: Int = MemoryLayout<Complex<Double>>.stride
        return dimension * dimension * complexSize
    }

    /// Check if unitary computation is feasible for given number of qubits.
    ///
    /// Validates qubit count is positive and within memory limit (≤30), then checks if required memory
    /// is less than 80% of available physical memory.
    ///
    /// ```swift
    /// if CircuitUnitary.canConvert(qubits: 14) {
    ///     let unitary = CircuitUnitary.unitary(for: largeCircuit)
    /// }
    /// ```
    ///
    /// - Parameter qubits: Number of qubits.
    /// - Returns: True if computation is feasible within memory constraints.
    /// - Complexity: O(1).
    /// - SeeAlso: ``memoryUsage(for:)`` for estimated memory usage.
    @_effects(readonly)
    @inlinable
    public static func canConvert(qubits: Int) -> Bool {
        guard qubits > 0, qubits <= 30 else { return false }

        let memoryBytes: Int = memoryUsage(for: qubits)
        let availableMemory: UInt64 = ProcessInfo.processInfo.physicalMemory
        let threshold: UInt64 = (availableMemory * 80) / 100

        return UInt64(memoryBytes) < threshold
    }
}
