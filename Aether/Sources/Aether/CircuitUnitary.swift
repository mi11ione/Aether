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
/// let ansatz = HardwareEfficientAnsatz(qubits: 8, depth: 3)
/// let circuits = (0..<100).map { i in
///     ansatz.circuit.bound(with: generateParameters(seed: i))
/// }
/// let unitaries = circuits.map { CircuitUnitary.unitary(for: $0) }
/// let evaluator = MPSBatchEvaluator()
/// let states = await evaluator.evaluate(batch: circuits, from: QuantumState(qubits: 8))
/// ```
///
/// - Note: Unitary matrices are computed using ``MatrixUtilities`` with Accelerate BLAS for optimal performance.
/// - SeeAlso: ``MPSBatchEvaluator``
/// - SeeAlso: ``MatrixUtilities``
/// - SeeAlso: ``QuantumCircuit``
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
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    /// let unitary = CircuitUnitary.unitary(for: circuit)
    /// ```
    ///
    /// - Parameter circuit: Quantum circuit to convert.
    /// - Returns: Dense unitary matrix as 2ⁿ x 2ⁿ complex matrix.
    /// - Complexity: O(depth · 2³ⁿ) where depth is number of gates. Per-gate cost includes O(2²ⁿ) expansion and O(2³ⁿ) matrix multiply.
    /// - Precondition: `circuit.qubits > 0` and `circuit.qubits <= 30`.
    /// - Note: Does not validate unitarity of result. Assumes circuit gates are valid unitary operations.
    /// - SeeAlso: ``MatrixUtilities``
    /// - SeeAlso: ``MPSBatchEvaluator``
    @_optimize(speed)
    @_eagerMove
    public static func unitary(for circuit: QuantumCircuit) -> [[Complex<Double>]] {
        let qubits: Int = circuit.qubits
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateMemoryLimit(qubits)

        let dimension = 1 << qubits
        var unitary: [[Complex<Double>]] = MatrixUtilities.identityMatrix(dimension: dimension)

        for operation in circuit.gates {
            let gateMatrix: [[Complex<Double>]] = expandGateToFullSpace(
                gate: operation.gate,
                qubits: operation.qubits,
                numQubits: qubits,
            )

            unitary = MatrixUtilities.matrixMultiply(gateMatrix, unitary)
        }

        return unitary
    }

    // MARK: - Gate Expansion

    /// Embed gate matrix into full 2ⁿ x 2ⁿ Hilbert space via tensor product with identity.
    @_optimize(speed)
    @_eagerMove
    private static func expandGateToFullSpace(
        gate: QuantumGate,
        qubits: [Int],
        numQubits: Int,
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

        case .controlled:
            return expandControlledGate(gate: gate, qubits: qubits, dimension: dimension)

        case .customUnitary:
            return expandMultiQubitGate(gate: gate, qubits: qubits, dimension: dimension)
        }
    }

    /// Expand single-qubit gate to full space via zero-bit insertion for O(2ⁿ⁻¹) iteration.
    @_optimize(speed)
    @_eagerMove
    private static func expandSingleQubitGate(
        gate: QuantumGate,
        targetQubit: Int,
        dimension: Int,
    ) -> [[Complex<Double>]] {
        let smallMatrix: [[Complex<Double>]] = gate.matrix()
        var fullMatrix: [[Complex<Double>]] = Array(
            repeating: Array(repeating: Complex<Double>.zero, count: dimension),
            count: dimension,
        )

        let halfDimension = dimension >> 1
        let targetMask = 1 << targetQubit

        for i in 0 ..< halfDimension {
            let base0 = BitUtilities.insertZeroBit(i, at: targetQubit)
            let base1 = base0 | targetMask

            fullMatrix[base0][base0] = smallMatrix[0][0]
            fullMatrix[base0][base1] = smallMatrix[0][1]
            fullMatrix[base1][base0] = smallMatrix[1][0]
            fullMatrix[base1][base1] = smallMatrix[1][1]
        }

        return fullMatrix
    }

    /// Expand two-qubit gate to full space via two-bit insertion for O(2ⁿ⁻²) iteration.
    @_optimize(speed)
    @_eagerMove
    private static func expandTwoQubitGate(
        gate: QuantumGate,
        control: Int,
        target: Int,
        dimension: Int,
    ) -> [[Complex<Double>]] {
        let smallMatrix: [[Complex<Double>]] = gate.matrix()
        var fullMatrix: [[Complex<Double>]] = Array(
            repeating: Array(repeating: Complex<Double>.zero, count: dimension),
            count: dimension,
        )

        let controlMask = 1 << control
        let targetMask = 1 << target
        let quarterDimension = dimension >> 2
        let (lowPos, highPos) = control < target ? (control, target) : (target, control)

        for i in 0 ..< quarterDimension {
            let base00 = BitUtilities.insertTwoZeroBits(i, low: lowPos, high: highPos)
            let base01 = base00 | targetMask
            let base10 = base00 | controlMask
            let base11 = base00 | controlMask | targetMask

            fullMatrix[base00][base00] = smallMatrix[0][0]
            fullMatrix[base00][base01] = smallMatrix[0][1]
            fullMatrix[base00][base10] = smallMatrix[0][2]
            fullMatrix[base00][base11] = smallMatrix[0][3]

            fullMatrix[base01][base00] = smallMatrix[1][0]
            fullMatrix[base01][base01] = smallMatrix[1][1]
            fullMatrix[base01][base10] = smallMatrix[1][2]
            fullMatrix[base01][base11] = smallMatrix[1][3]

            fullMatrix[base10][base00] = smallMatrix[2][0]
            fullMatrix[base10][base01] = smallMatrix[2][1]
            fullMatrix[base10][base10] = smallMatrix[2][2]
            fullMatrix[base10][base11] = smallMatrix[2][3]

            fullMatrix[base11][base00] = smallMatrix[3][0]
            fullMatrix[base11][base01] = smallMatrix[3][1]
            fullMatrix[base11][base10] = smallMatrix[3][2]
            fullMatrix[base11][base11] = smallMatrix[3][3]
        }

        return fullMatrix
    }

    /// Expand Toffoli gate to full space via conditional target flip when both controls are 1.
    @_optimize(speed)
    @_eagerMove
    private static func expandToffoliGate(
        control1: Int,
        control2: Int,
        target: Int,
        dimension: Int,
    ) -> [[Complex<Double>]] {
        var fullMatrix: [[Complex<Double>]] = Array(
            repeating: Array(repeating: Complex<Double>.zero, count: dimension),
            count: dimension,
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

    @_optimize(speed)
    @_eagerMove
    private static func expandControlledGate(
        gate: QuantumGate,
        qubits _: [Int],
        dimension: Int,
    ) -> [[Complex<Double>]] {
        let gateMatrix = gate.matrix()
        var fullMatrix: [[Complex<Double>]] = Array(
            repeating: Array(repeating: Complex<Double>.zero, count: dimension),
            count: dimension,
        )

        for row in 0 ..< dimension {
            for col in 0 ..< dimension {
                fullMatrix[row][col] = gateMatrix[row][col]
            }
        }

        return fullMatrix
    }

    @_optimize(speed)
    @_eagerMove
    private static func expandMultiQubitGate(
        gate: QuantumGate,
        qubits: [Int],
        dimension: Int,
    ) -> [[Complex<Double>]] {
        let gateMatrix = gate.matrix()
        let gateSize = gateMatrix.count
        var fullMatrix: [[Complex<Double>]] = Array(
            repeating: Array(repeating: Complex<Double>.zero, count: dimension),
            count: dimension,
        )

        for row in 0 ..< dimension {
            var gateRow = 0
            for (idx, qubit) in qubits.enumerated() {
                if (row & (1 << qubit)) != 0 {
                    gateRow |= (1 << idx)
                }
            }

            for gateCol in 0 ..< gateSize {
                let matrixElement = gateMatrix[gateRow][gateCol]
                if matrixElement.real == 0, matrixElement.imaginary == 0 {
                    continue
                }

                var col = row
                for (idx, qubit) in qubits.enumerated() {
                    let colBit = (gateCol >> idx) & 1
                    let mask = 1 << qubit
                    if colBit == 1 {
                        col |= mask
                    } else {
                        col &= ~mask
                    }
                }
                fullMatrix[row][col] = matrixElement
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
    /// - Parameter qubits: Number of qubits.
    /// - Returns: Memory in bytes.
    /// - Complexity: O(1).
    @_effects(readonly)
    @inlinable
    public static func memoryUsage(for qubits: Int) -> Int {
        let dimension = 1 << qubits
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
    /// - Precondition: `qubits > 0`.
    /// - Precondition: `qubits <= 30`.
    /// - SeeAlso: ``memoryUsage(for:)`` for estimated memory usage.
    @_effects(readonly)
    @inlinable
    public static func canConvert(qubits: Int) -> Bool {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateMemoryLimit(qubits)

        let memoryBytes: Int = memoryUsage(for: qubits)
        let availableMemory: UInt64 = ProcessInfo.processInfo.physicalMemory
        let threshold: UInt64 = (availableMemory * 80) / 100

        return UInt64(memoryBytes) < threshold
    }
}
