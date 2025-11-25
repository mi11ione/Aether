// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Circuit-to-unitary matrix converter for batched GPU evaluation
///
/// Converts quantum circuits to dense unitary matrices via sequential gate composition.
/// Enables batched circuit execution on GPU through Metal Performance Shaders matrix-vector
/// operations, achieving 50-100× speedup for parameter space exploration in VQE and QAOA.
///
/// **Mathematical Foundation:**
/// - Quantum circuit C = G₁G₂...Gₙ represented as unitary U = Gₙ...G₂G₁ (right-to-left composition)
/// - Each gate Gᵢ is 2ⁿ × 2ⁿ matrix with sparse structure (identity on non-target qubits)
/// - Composition via matrix multiplication: U = Gₙ · ... · G₂ · G₁
/// - Complexity: O(depth · 2³ⁿ) for n-qubit circuit with d gates
///
/// **Memory Scaling:**
/// - Unitary matrix: 2ⁿ × 2ⁿ complex numbers = 2ⁿ⁺¹ · 16 bytes (Double)
/// - 8 qubits: 256×256 × 16 = 1 MB
/// - 10 qubits: 1024×1024 × 16 = 16 MB
/// - 12 qubits: 4096×4096 × 16 = 256 MB
/// - 14 qubits: 16384×16384 × 16 = 4 GB (approaching practical limit)
///
/// **Performance Characteristics:**
/// - Gate expansion: O(2²ⁿ) to embed small gate in full Hilbert space
/// - Matrix multiply: O(2³ⁿ) per gate using BLAS acceleration
/// - Total: O(depth · 2³ⁿ) - expensive but one-time per circuit structure
/// - Amortized over batch: Cost paid once, reused for 100+ parameter evaluations
///
/// **Use Cases:**
/// 1. **VQE gradient computation**: Compute θᵢ±π/2 circuits once, batch evaluate all
/// 2. **Grid search**: Convert ansatz(θ) for each grid point, batch evaluate all
/// 3. **Population optimizers**: Genetic algorithms, particle swarm with parallel fitness
/// 4. **Hyperparameter tuning**: Test multiple ansätze simultaneously
///
/// **Trade-offs:**
/// - **When to use**: Batch size ≥ 10, same circuit structure with different parameters
/// - **When not to use**: Single evaluation, very deep circuits (compute gate-by-gate faster)
/// - **Sweet spot**: 5-12 qubits, 10-100 circuits in batch, moderate depth (<50 gates)
///
/// Example - VQE gradient computation:
/// ```swift
/// // Build hardware-efficient ansatz
/// let ansatz = HardwareEfficientAnsatz.create(numQubits: 8, depth: 3)
///
/// // Bind parameter values
/// let baseParams: [Double] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
/// let baseCircuit = try ansatz.bind(parameterVector: baseParams)
///
/// // Generate shifted circuits for gradient (2 × 8 = 16 circuits)
/// var shiftedCircuits: [QuantumCircuit] = []
/// for i in 0..<baseParams.count {
///     let (plus, minus) = try ansatz.generateShiftedCircuits(
///         parameterIndex: i,
///         baseVector: baseParams
///     )
///     shiftedCircuits.append(plus)
///     shiftedCircuits.append(minus)
/// }
///
/// // Convert all to unitaries once (expensive but one-time)
/// let unitaries = try shiftedCircuits.map { try CircuitUnitary.computeUnitary(circuit: $0) }
///
/// // Batch evaluate on GPU (50-100× faster than sequential)
/// let batchEvaluator = await MPSBatchEvaluator()
/// let energies = try await batchEvaluator.evaluateExpectationValues(
///     unitaries: unitaries,
///     initialState: QuantumState(numQubits: 8),
///     hamiltonian: molecularHamiltonian
/// )
///
/// // Extract gradients from energies
/// for i in 0..<baseParams.count {
///     let energyPlus = energies[2 * i]
///     let energyMinus = energies[2 * i + 1]
///     let gradient = (energyPlus - energyMinus) / 2.0
///     print("∂E/∂θ[\(i)] = \(gradient)")
/// }
/// ```
///
/// Example - Grid search optimization:
/// ```swift
/// // Define parameter grid
/// let gammaRange = stride(from: 0.0, through: .pi, by: .pi / 10)
/// let betaRange = stride(from: 0.0, through: .pi, by: .pi / 10)
///
/// // Build all circuits
/// var circuits: [QuantumCircuit] = []
/// for gamma in gammaRange {
///     for beta in betaRange {
///         let circuit = try qaoaAnsatz.bind(parameterVector: [gamma, beta])
///         circuits.append(circuit)
///     }
/// }
/// // 11 × 11 = 121 circuits
///
/// // Convert to unitaries (one-time cost)
/// let unitaries = try circuits.map { try CircuitUnitary.computeUnitary(circuit: $0) }
///
/// // Batch evaluate all grid points in parallel
/// let energies = try await batchEvaluator.evaluateExpectationValues(
///     unitaries: unitaries,
///     initialState: initialState,
///     hamiltonian: maxCutHamiltonian
/// )
///
/// // Find optimal parameters
/// let minIndex = energies.enumerated().min(by: { $0.element < $1.element })!.offset
/// let optimalGamma = Array(gammaRange)[minIndex / 11]
/// let optimalBeta = Array(betaRange)[minIndex % 11]
/// ```
@frozen
public enum CircuitUnitary {
    // MARK: - Public API

    /// Compute full circuit unitary matrix via gate composition
    ///
    /// Converts quantum circuit to dense 2ⁿ × 2ⁿ unitary matrix through sequential
    /// matrix multiplication of gate matrices. Uses BLAS-accelerated matrix multiply
    /// from MatrixUtilities for 10-100× speedup over naive loops.
    ///
    /// **Algorithm:**
    /// 1. Start with identity matrix I (dimension 2ⁿ × 2ⁿ)
    /// 2. For each gate G in circuit (left to right):
    ///    - Expand G to full Hilbert space (embed small gate in 2ⁿ × 2ⁿ matrix)
    ///    - Compose: U ← G · U (right-multiply, gates apply right-to-left)
    /// 3. Return final U
    ///
    /// **Gate Expansion:**
    /// - Single-qubit gate on qubit k: I ⊗ ... ⊗ G ⊗ ... ⊗ I (G at position k)
    /// - Two-qubit gate on qubits (c,t): Similar tensor product structure
    /// - Three-qubit Toffoli: Full 8×8 matrix embedded in 2ⁿ space
    ///
    /// **Complexity:**
    /// - Per gate: O(2³ⁿ) for matrix multiply + O(2²ⁿ) for expansion
    /// - Total: O(depth · 2³ⁿ) where depth = number of gates
    ///
    /// **Memory:**
    /// - Working matrices: 3 × 2ⁿ × 2ⁿ × 16 bytes (current U, gate matrix, result)
    /// - Peak usage: ~750 MB for 12 qubits
    ///
    /// **Validation:**
    /// - Checks circuit has at least 1 qubit
    /// - Validates memory limit (numQubits ≤ 30)
    /// - Does NOT validate unitarity (assume circuit is valid)
    ///
    /// - Parameter circuit: Quantum circuit to convert
    /// - Returns: Dense unitary matrix (2ⁿ × 2ⁿ complex matrix)
    ///
    /// Example:
    /// ```swift
    /// // Simple Bell state circuit
    /// var circuit = QuantumCircuit(numQubits: 2)
    /// circuit.append(gate: .hadamard, toQubit: 0)
    /// circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
    ///
    /// // Convert to unitary
    /// let unitary = try CircuitUnitary.computeUnitary(circuit: circuit)
    /// // unitary is 4×4 matrix representing H₀ · CNOT₀₁
    ///
    /// // Verify: applying unitary to |00⟩ gives (|00⟩ + |11⟩)/√2
    /// let initialState = [Complex<Double>(1,0), .zero, .zero, .zero]
    /// let finalState = matrixVectorMultiply(unitary, initialState)
    /// // finalState ≈ [1/√2, 0, 0, 1/√2]
    /// ```
    @_optimize(speed)
    @_eagerMove
    public static func computeUnitary(circuit: QuantumCircuit) -> GateMatrix {
        let numQubits: Int = circuit.numQubits
        ValidationUtilities.validatePositiveQubits(numQubits)
        ValidationUtilities.validateMemoryLimit(numQubits)

        let dimension = 1 << numQubits
        var unitary: GateMatrix = MatrixUtilities.identityMatrix(dimension: dimension)

        for operation in circuit.operations {
            let gateMatrix: GateMatrix = expandGateToFullSpace(
                gate: operation.gate,
                qubits: operation.qubits,
                numQubits: numQubits
            )

            unitary = MatrixUtilities.matrixMultiply(gateMatrix, unitary)
        }

        return unitary
    }

    // MARK: - Gate Expansion

    /// Expand gate to full Hilbert space (2ⁿ × 2ⁿ matrix)
    ///
    /// Embeds small gate matrix (2×2 or 4×4 or 8×8) into full 2ⁿ × 2ⁿ space
    /// via tensor product with identity on non-target qubits.
    ///
    /// **Algorithm:**
    /// - Single-qubit: Tensor product I ⊗ ... ⊗ G ⊗ ... ⊗ I
    /// - Two-qubit: Similar but with 4×4 gate matrix
    /// - Three-qubit: Direct construction from Toffoli structure
    ///
    /// **Complexity:** O(2²ⁿ) to fill full matrix
    ///
    /// **Optimization:** Direct index computation instead of full tensor product
    /// - Single-qubit: For each (row,col), check if target bit matches
    /// - Two-qubit: Check control and target bits
    /// - Avoids expensive tensor product construction
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to expand
    ///   - qubits: Qubit indices (for gates that store qubits in gate enum)
    ///   - numQubits: Total number of qubits in circuit
    /// - Returns: Full 2ⁿ × 2ⁿ matrix
    @_optimize(speed)
    @_eagerMove
    private static func expandGateToFullSpace(
        gate: QuantumGate,
        qubits: [Int],
        numQubits: Int
    ) -> GateMatrix {
        let dimension = 1 << numQubits

        switch gate {
        // Single-qubit gates
        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard,
             .phase, .sGate, .tGate, .rotationX, .rotationY, .rotationZ,
             .u1, .u2, .u3, .sx, .sy, .customSingleQubit:
            let targetQubit: Int = qubits[0]
            return expandSingleQubitGate(gate: gate, targetQubit: targetQubit, dimension: dimension)

        // Two-qubit gates
        case let .cnot(control, target),
             let .cz(control, target),
             let .cy(control, target),
             let .ch(control, target),
             let .controlledPhase(_, control, target),
             let .controlledRotationX(_, control, target),
             let .controlledRotationY(_, control, target),
             let .controlledRotationZ(_, control, target),
             let .swap(control, target),
             let .sqrtSwap(control, target),
             let .customTwoQubit(_, control, target):
            return expandTwoQubitGate(gate: gate, control: control, target: target, dimension: dimension)

        // Three-qubit gates
        case let .toffoli(control1, control2, target):
            return expandToffoliGate(control1: control1, control2: control2, target: target, dimension: dimension)
        }
    }

    /// Expand single-qubit gate to full space
    ///
    /// **Algorithm:**
    /// Instead of iterating all dimension² pairs, directly compute valid indices.
    /// For single-qubit gate, only pairs where other bits match are non-zero.
    /// For each "other bits" pattern (dimension/2 patterns), we have 4 entries (2×2 gate).
    ///
    /// **Complexity:** O(2ⁿ × 4) = O(2ⁿ⁺²) instead of O(2²ⁿ)
    ///
    /// **Example:** H on qubit 0 of 2-qubit system
    /// ```
    /// Basis: |00⟩, |01⟩, |10⟩, |11⟩
    /// Row 0 (00): H maps |0⟩ -> (|0⟩+|1⟩)/√2, so row = [1/√2, 1/√2, 0, 0]
    /// Row 1 (01): H maps |1⟩ -> (|0⟩-|1⟩)/√2, so row = [1/√2, -1/√2, 0, 0]
    /// Row 2 (10): H maps |0⟩ -> (|0⟩+|1⟩)/√2, so row = [0, 0, 1/√2, 1/√2]
    /// Row 3 (11): H maps |1⟩ -> (|0⟩-|1⟩)/√2, so row = [0, 0, 1/√2, -1/√2]
    /// ```
    @_optimize(speed)
    @_eagerMove
    private static func expandSingleQubitGate(
        gate: QuantumGate,
        targetQubit: Int,
        dimension: Int
    ) -> GateMatrix {
        let smallMatrix: GateMatrix = gate.matrix()
        var fullMatrix: GateMatrix = Array(
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

    /// Expand two-qubit gate to full space
    ///
    /// **Algorithm:**
    /// Instead of iterating all dimension² pairs, directly compute valid indices.
    /// For two-qubit gate, only pairs where other bits match are non-zero.
    /// For each "other bits" pattern (dimension/4 patterns), we have 16 entries (4×4 gate).
    ///
    /// **Complexity:** O(2ⁿ × 16 / 4) = O(2ⁿ⁺²) instead of O(2²ⁿ)
    ///
    /// **Index Mapping:**
    /// - Gate matrix is 4×4: indices 0-3 map to (c,t) = (0,0), (0,1), (1,0), (1,1)
    /// - Row index: 2*controlBit + targetBit
    @_optimize(speed)
    @_eagerMove
    private static func expandTwoQubitGate(
        gate: QuantumGate,
        control: Int,
        target: Int,
        dimension: Int
    ) -> GateMatrix {
        let smallMatrix: GateMatrix = gate.matrix()
        var fullMatrix: GateMatrix = Array(
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

    /// Expand Toffoli gate to full space
    ///
    /// **Toffoli Structure:**
    /// - If both controls are 1, flip target
    /// - Otherwise, identity
    ///
    /// **Matrix Form:**
    /// ```
    /// For basis state |c1,c2,t⟩:
    /// - If c1=1 AND c2=1: |c1,c2,t⟩ -> |c1,c2,t⊕1⟩
    /// - Otherwise: |c1,c2,t⟩ -> |c1,c2,t⟩
    /// ```
    ///
    /// **Implementation:**
    /// Direct construction without creating identity first.
    /// - Most entries are identity (diagonal = 1)
    /// - Only rows where both controls = 1 have off-diagonal entries
    ///
    /// **Complexity:** O(2ⁿ) to set diagonal + O(2ⁿ/4) to modify control rows
    @_optimize(speed)
    @_eagerMove
    private static func expandToffoliGate(
        control1: Int,
        control2: Int,
        target: Int,
        dimension: Int
    ) -> GateMatrix {
        var fullMatrix: GateMatrix = Array(
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

    /// Estimate memory usage for unitary matrix
    /// - Parameter numQubits: Number of qubits
    /// - Returns: Memory in bytes
    @_effects(readonly)
    @inlinable
    public static func estimateMemoryUsage(numQubits: Int) -> Int {
        let dimension = 1 << numQubits
        let complexSize: Int = MemoryLayout<Complex<Double>>.stride
        return dimension * dimension * complexSize
    }

    /// Check if unitary computation is feasible
    /// - Parameter numQubits: Number of qubits
    /// - Returns: True if computation is feasible
    @_effects(readonly)
    @inlinable
    public static func isFeasible(numQubits: Int) -> Bool {
        ValidationUtilities.validateMemoryLimit(numQubits)
        ValidationUtilities.validatePositiveQubits(numQubits)

        let memoryBytes: Int = estimateMemoryUsage(numQubits: numQubits)
        let availableMemory: UInt64 = ProcessInfo.processInfo.physicalMemory
        let threshold: UInt64 = (availableMemory * 80) / 100

        return UInt64(memoryBytes) < threshold
    }
}
