// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Density matrix representation ρ for mixed quantum states in 2^n * 2^n Hilbert space.
///
/// Unlike pure states (statevectors |ψ⟩), density matrices can represent statistical mixtures
/// and entangled subsystems. A density matrix ρ is a positive semidefinite Hermitian operator
/// with Tr(ρ) = 1. Pure states satisfy ρ = |ψ⟩⟨ψ| with Tr(ρ²) = 1, while mixed states have
/// Tr(ρ²) < 1 indicating classical uncertainty beyond quantum superposition.
///
/// Density matrices require 4^n complex numbers (2^n * 2^n matrix) compared to 2^n for statevectors,
/// limiting practical simulation to approximately 14 qubits (4.3 GB). For noise-free simulation
/// beyond 14 qubits, use ``QuantumState`` with ``QuantumSimulator``. Uses row-major flattened array
/// for BLAS compatibility where element ρ[i,j] is stored at index i * dimension + j. Little-endian
/// qubit ordering matches ``QuantumState``.
///
/// **Example:**
/// ```swift
/// // Ground state density matrix |0⟩⟨0|
/// let ground = DensityMatrix(qubits: 2)
/// ground.purity()  // 1.0 (pure state)
///
/// // Create from pure state
/// let bell = QuantumState(qubits: 2, amplitudes: [
///     Complex(1/sqrt(2), 0), .zero, .zero, Complex(1/sqrt(2), 0)
/// ])
/// let bellDensity = DensityMatrix(pureState: bell)
/// bellDensity.probability(of: 0b00)  // 0.5
///
/// // Maximally mixed state
/// let mixed = DensityMatrix.maximallyMixed(qubits: 2)
/// mixed.purity()  // 0.25 (fully mixed)
/// ```
///
/// - SeeAlso: ``QuantumState`` for pure state simulation
/// - SeeAlso: ``NoiseChannel`` for quantum noise operations
/// - SeeAlso: ``DensityMatrixSimulator`` for noisy circuit execution
@frozen
public struct DensityMatrix: Equatable, CustomStringConvertible, Sendable {
    // MARK: - Properties

    /// Flattened density matrix elements in row-major order.
    ///
    /// Element ρ[row, col] is stored at index row * dimension + col.
    /// Total size is dimension² = 4^qubits complex numbers.
    private var elements: [Complex<Double>]

    /// Number of qubits in this quantum system.
    public let qubits: Int

    /// Hilbert space dimension (2^qubits).
    ///
    /// **Example:**
    /// ```swift
    /// let dm = DensityMatrix(qubits: 3)
    /// print(dm.dimension)  // 8
    /// ```
    ///
    /// - Complexity: O(1)
    @inlinable
    public var dimension: Int { 1 << qubits }

    // MARK: - Initialization

    /// Initialize ground state density matrix: |0...0⟩⟨0...0|.
    ///
    /// Creates computational basis projector onto the all-zeros state. This is a pure state
    /// with ρ[0,0] = 1 and all other elements zero.
    ///
    /// **Example:**
    /// ```swift
    /// let ground = DensityMatrix(qubits: 2)  // |00⟩⟨00|
    /// ground.probability(of: 0)  // 1.0
    /// ground.purity()  // 1.0
    /// ```
    ///
    /// - Parameter qubits: Number of qubits (1-14)
    /// - Complexity: O(4^n)
    /// - Precondition: 1 ≤ qubits ≤ 14
    public init(qubits: Int) {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateDensityMatrixMemoryLimit(qubits)

        self.qubits = qubits
        let dim = 1 << qubits
        let size = dim * dim

        elements = [Complex<Double>](unsafeUninitializedCapacity: size) { buffer, count in
            buffer.initialize(repeating: .zero)
            buffer[0] = .one
            count = size
        }
    }

    /// Initialize density matrix from pure state: ρ = |ψ⟩⟨ψ|.
    ///
    /// Constructs rank-1 projector from statevector via outer product. The resulting
    /// density matrix has Tr(ρ²) = 1 (pure state) and represents the same quantum state
    /// as the input but in density matrix formalism.
    ///
    /// **Example:**
    /// ```swift
    /// let plus = QuantumState(qubits: 1, amplitudes: [
    ///     Complex(1/sqrt(2), 0), Complex(1/sqrt(2), 0)
    /// ])
    /// let dm = DensityMatrix(pureState: plus)
    /// dm.element(row: 0, col: 1)  // 0.5 (off-diagonal coherence)
    /// ```
    ///
    /// - Parameter pureState: Normalized quantum state |ψ⟩
    /// - Complexity: O(4^n)
    /// - Precondition: State must be normalized and have ≤ 14 qubits
    public init(pureState: QuantumState) {
        ValidationUtilities.validateDensityMatrixMemoryLimit(pureState.qubits)

        qubits = pureState.qubits
        let dim = pureState.stateSpaceSize
        let size = dim * dim

        elements = [Complex<Double>](unsafeUninitializedCapacity: size) { buffer, count in
            for i in 0 ..< dim {
                let ci = pureState.amplitudes[i]
                for j in 0 ..< dim {
                    let cj = pureState.amplitudes[j]
                    buffer[i * dim + j] = ci * cj.conjugate
                }
            }
            count = size
        }
    }

    /// Initialize density matrix from explicit elements.
    ///
    /// Creates density matrix from flattened row-major element array. Validates that
    /// element count matches dimension squared. Does not validate positive semidefiniteness
    /// or trace normalization for performance; use ``isHermitian()`` and ``isTraceNormalized()`` to check.
    ///
    /// **Example:**
    /// ```swift
    /// // Maximally mixed single-qubit state
    /// let elements: [Complex<Double>] = [
    ///     Complex(0.5, 0), .zero,
    ///     .zero, Complex(0.5, 0)
    /// ]
    /// let mixed = DensityMatrix(qubits: 1, elements: elements)
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits
    ///   - elements: Flattened row-major elements (dimension² count)
    /// - Complexity: O(4^n)
    /// - Precondition: elements.count == 4^qubits
    public init(qubits: Int, elements: [Complex<Double>]) {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateDensityMatrixMemoryLimit(qubits)

        let dim = 1 << qubits
        ValidationUtilities.validateDensityMatrixElementCount(elements, dimension: dim)

        self.qubits = qubits
        self.elements = elements
    }

    // MARK: - Factory Methods

    /// Create maximally mixed state ρ = I/2^n.
    ///
    /// The maximally mixed state has equal probability for all basis states and no
    /// coherence (off-diagonal elements are zero). Represents complete lack of information
    /// about the quantum state. Purity = 1/2^n (minimum possible).
    ///
    /// **Example:**
    /// ```swift
    /// let mixed = DensityMatrix.maximallyMixed(qubits: 2)
    /// mixed.probability(of: 0)  // 0.25
    /// mixed.purity()  // 0.25
    /// ```
    ///
    /// - Parameter qubits: Number of qubits (1-14)
    /// - Returns: Maximally mixed density matrix I/2^n
    /// - Complexity: O(4^n)
    /// - Precondition: 1 ≤ qubits ≤ 14
    @_effects(readonly)
    @_eagerMove
    public static func maximallyMixed(qubits: Int) -> DensityMatrix {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateDensityMatrixMemoryLimit(qubits)

        let dim = 1 << qubits
        let size = dim * dim
        let diagValue = Complex<Double>(1.0 / Double(dim), 0)

        let elements = [Complex<Double>](unsafeUninitializedCapacity: size) { buffer, count in
            buffer.initialize(repeating: .zero)
            for i in 0 ..< dim {
                buffer[i * dim + i] = diagValue
            }
            count = size
        }

        return DensityMatrix(qubits: qubits, elements: elements)
    }

    /// Create computational basis state |i⟩⟨i|.
    ///
    /// Creates pure state projector onto basis state |i⟩ where i is the basis index
    /// in little-endian qubit ordering.
    ///
    /// **Example:**
    /// ```swift
    /// let dm = DensityMatrix.basis(qubits: 2, state: 0b11)  // |11⟩⟨11|
    /// dm.probability(of: 0b11)  // 1.0
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (1-14)
    ///   - state: Basis state index (0 to 2^n - 1)
    /// - Returns: Projector density matrix |i⟩⟨i|
    /// - Complexity: O(4^n)
    /// - Precondition: 1 ≤ qubits ≤ 14
    /// - Precondition: 0 ≤ state < 2^qubits
    @_effects(readonly)
    @_eagerMove
    public static func basis(qubits: Int, state: Int) -> DensityMatrix {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateDensityMatrixMemoryLimit(qubits)
        ValidationUtilities.validateIndexInBounds(state, bound: 1 << qubits, name: "Basis state")

        let dim = 1 << qubits
        let size = dim * dim

        let elements = [Complex<Double>](unsafeUninitializedCapacity: size) { buffer, count in
            buffer.initialize(repeating: .zero)
            buffer[state * dim + state] = .one
            count = size
        }

        return DensityMatrix(qubits: qubits, elements: elements)
    }

    // MARK: - Element Access

    /// Get density matrix element ρ[row, col].
    ///
    /// **Example:**
    /// ```swift
    /// let dm = DensityMatrix(qubits: 1)
    /// dm.element(row: 0, col: 0)  // 1.0
    /// ```
    ///
    /// - Parameters:
    ///   - row: Row index (0 to dimension-1)
    ///   - col: Column index (0 to dimension-1)
    /// - Returns: Complex element ρ[row, col]
    /// - Complexity: O(1)
    /// - Precondition: 0 ≤ row < dimension
    /// - Precondition: 0 ≤ col < dimension
    @_effects(readonly)
    public func element(row: Int, col: Int) -> Complex<Double> {
        let dim = dimension
        ValidationUtilities.validateIndexInBounds(row, bound: dim, name: "Row index")
        ValidationUtilities.validateIndexInBounds(col, bound: dim, name: "Column index")
        return elements[row * dim + col]
    }

    /// Set density matrix element ρ[row, col].
    ///
    /// - Parameters:
    ///   - row: Row index (0 to dimension-1)
    ///   - col: Column index (0 to dimension-1)
    ///   - value: New complex value
    /// - Complexity: O(1)
    /// - Precondition: 0 ≤ row < dimension
    /// - Precondition: 0 ≤ col < dimension
    /// - Note: May invalidate trace normalization or positive semidefiniteness
    public mutating func setElement(row: Int, col: Int, to value: Complex<Double>) {
        let dim = dimension
        ValidationUtilities.validateIndexInBounds(row, bound: dim, name: "Row index")
        ValidationUtilities.validateIndexInBounds(col, bound: dim, name: "Column index")
        elements[row * dim + col] = value
    }

    // MARK: - Properties

    /// Compute trace Tr(ρ) = Σᵢ ρ[i,i].
    ///
    /// For valid density matrices, trace equals 1.0 (probability conservation).
    /// Non-unit trace indicates unnormalized or invalid state.
    ///
    /// **Example:**
    /// ```swift
    /// let dm = DensityMatrix(qubits: 2)
    /// dm.trace()  // 1.0
    /// ```
    ///
    /// - Returns: Sum of diagonal elements
    /// - Complexity: O(2^n)
    @_optimize(speed)
    @_effects(readonly)
    public func trace() -> Double {
        let dim = dimension
        var sum = 0.0
        for i in 0 ..< dim {
            sum += elements[i * dim + i].real
        }
        return sum
    }

    /// Compute purity Tr(ρ²).
    ///
    /// Purity measures how mixed a quantum state is:
    /// - Tr(ρ²) = 1 for pure states
    /// - Tr(ρ²) = 1/d for maximally mixed states (d = dimension)
    /// - 1/d ≤ Tr(ρ²) ≤ 1 for all valid states
    ///
    /// **Example:**
    /// ```swift
    /// let pure = DensityMatrix(qubits: 1)
    /// pure.purity()  // 1.0
    ///
    /// let mixed = DensityMatrix.maximallyMixed(qubits: 1)
    /// mixed.purity()  // 0.5
    /// ```
    ///
    /// - Returns: Purity value in [1/d, 1]
    /// - Complexity: O(4^n) for matrix multiplication
    @_optimize(speed)
    @_effects(readonly)
    public func purity() -> Double {
        let dim = dimension
        var sum = 0.0

        for i in 0 ..< dim {
            for k in 0 ..< dim {
                let rhoIK = elements[i * dim + k]
                let rhoKI = elements[k * dim + i]
                sum += (rhoIK * rhoKI).real
            }
        }

        return sum
    }

    /// Check if state is pure (Tr(ρ²) ≈ 1).
    ///
    /// **Example:**
    /// ```swift
    /// let dm = DensityMatrix(qubits: 1)
    /// dm.isPure()  // true
    /// ```
    ///
    /// - Parameter tolerance: Numerical tolerance (default 1e-10)
    /// - Returns: True if purity is within tolerance of 1.0
    /// - Complexity: O(4^n)
    @_effects(readonly)
    public func isPure(tolerance: Double = 1e-10) -> Bool {
        abs(purity() - 1.0) < tolerance
    }

    /// Check if trace equals 1.0 (probability conservation).
    ///
    /// - Parameter tolerance: Numerical tolerance (default 1e-10)
    /// - Returns: True if trace is within tolerance of 1.0
    /// - Complexity: O(2^n)
    @_effects(readonly)
    public func isTraceNormalized(tolerance: Double = 1e-10) -> Bool {
        abs(trace() - 1.0) < tolerance
    }

    /// Check if density matrix is Hermitian (ρ† = ρ).
    ///
    /// All valid density matrices must be Hermitian, guaranteeing real eigenvalues
    /// (probabilities) and self-adjoint measurement operators.
    ///
    /// - Parameter tolerance: Numerical tolerance (default 1e-10)
    /// - Returns: True if ρ[i,j] = ρ[j,i]* within tolerance
    /// - Complexity: O(4^n)
    @_effects(readonly)
    public func isHermitian(tolerance: Double = 1e-10) -> Bool {
        let dim = dimension
        for i in 0 ..< dim {
            for j in i ..< dim {
                let rhoIJ = elements[i * dim + j]
                let rhoJI = elements[j * dim + i]
                let diff = rhoIJ - rhoJI.conjugate
                if diff.magnitudeSquared > tolerance * tolerance {
                    return false
                }
            }
        }
        return true
    }

    // MARK: - Probabilities

    /// Get probability of measuring basis state i: P(i) = ρ[i,i].
    ///
    /// Diagonal elements of density matrix give Born rule probabilities.
    ///
    /// **Example:**
    /// ```swift
    /// let dm = DensityMatrix.maximallyMixed(qubits: 2)
    /// dm.probability(of: 0)  // 0.25
    /// ```
    ///
    /// - Parameter stateIndex: Basis state index (0 to 2^n - 1)
    /// - Returns: Probability P(i) = ρ[i,i] ∈ [0, 1]
    /// - Complexity: O(1)
    /// - Precondition: 0 ≤ stateIndex < dimension
    @_effects(readonly)
    public func probability(of stateIndex: Int) -> Double {
        let dim = dimension
        ValidationUtilities.validateIndexInBounds(stateIndex, bound: dim, name: "Basis state index")
        return elements[stateIndex * dim + stateIndex].real
    }

    /// Get full probability distribution over basis states.
    ///
    /// **Example:**
    /// ```swift
    /// let dm = DensityMatrix.maximallyMixed(qubits: 2)
    /// dm.probabilities()  // [0.25, 0.25, 0.25, 0.25]
    /// ```
    ///
    /// - Returns: Array of probabilities [P(0), P(1), ..., P(2^n-1)]
    /// - Complexity: O(2^n)
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public func probabilities() -> [Double] {
        let dim = dimension
        return [Double](unsafeUninitializedCapacity: dim) { buffer, count in
            for i in 0 ..< dim {
                buffer[i] = elements[i * dim + i].real
            }
            count = dim
        }
    }

    /// Find basis state with highest probability.
    ///
    /// - Returns: Tuple of (basisStateIndex, probability)
    /// - Complexity: O(2^n)
    @_optimize(speed)
    @_effects(readonly)
    public func mostProbableState() -> (index: Int, probability: Double) {
        let dim = dimension
        var maxIndex = 0
        var maxProb = elements[0].real

        for i in 1 ..< dim {
            let prob = elements[i * dim + i].real
            if prob > maxProb {
                maxProb = prob
                maxIndex = i
            }
        }

        return (maxIndex, maxProb)
    }

    // MARK: - Expectation Values

    /// Compute expectation value ⟨O⟩ = Tr(ρO) for observable.
    ///
    /// Uses linearity: Tr(ρO) = Σᵢ cᵢ Tr(ρPᵢ) where O = Σᵢ cᵢPᵢ.
    /// Each Pauli string expectation is computed efficiently using the
    /// sparse structure of Pauli operators.
    ///
    /// **Example:**
    /// ```swift
    /// let dm = DensityMatrix(qubits: 1)  // |0⟩⟨0|
    /// let z = Observable.pauliZ(qubit: 0)
    /// dm.expectationValue(of: z)  // 1.0
    /// ```
    ///
    /// - Parameter observable: Hermitian observable O = Σᵢ cᵢPᵢ
    /// - Returns: Real expectation value Tr(ρO)
    /// - Complexity: O(k * 4^n) where k = number of Pauli terms
    @_optimize(speed)
    @_effects(readonly)
    public func expectationValue(of observable: Observable) -> Double {
        var total = 0.0

        for term in observable.terms {
            let pauliExpectation = computePauliStringExpectation(term.pauliString)
            total += term.coefficient * pauliExpectation
        }

        return total
    }

    /// Compute Tr(ρP) using Pauli permutation structure.
    @_optimize(speed)
    @_effects(readonly)
    private func computePauliStringExpectation(_ pauliString: PauliString) -> Double {
        if pauliString.operators.isEmpty {
            return trace()
        }

        let dim = dimension
        var sum = Complex<Double>.zero

        for i in 0 ..< dim {
            let (col, phase) = pauliString.applyToRow(row: i)
            sum = sum + phase * elements[col * dim + i]
        }

        return sum.real
    }

    // MARK: - Partial Trace

    /// Compute partial trace over specified qubits: ρ_A = Tr_B(ρ_AB).
    ///
    /// Traces out (removes) the specified qubits, returning reduced density matrix
    /// for the remaining subsystem. Essential for analyzing entanglement and computing
    /// subsystem properties.
    ///
    /// **Example:**
    /// ```swift
    /// let bell = QuantumState(qubits: 2, amplitudes: [
    ///     Complex(1/sqrt(2), 0), .zero, .zero, Complex(1/sqrt(2), 0)
    /// ])
    /// let dm = DensityMatrix(pureState: bell)
    /// let reduced = dm.partialTrace(over: [1])  // Trace out qubit 1
    /// reduced.purity()  // 0.5 (maximally mixed - entangled state)
    /// ```
    ///
    /// - Parameter qubitsToTrace: Qubit indices to trace out
    /// - Returns: Reduced density matrix on remaining qubits
    /// - Complexity: O(4^n)
    /// - Precondition: qubitsToTrace is non-empty
    /// - Precondition: All indices in [0, qubits)
    /// - Precondition: No duplicate indices
    /// - Precondition: qubitsToTrace.count < qubits (at least one qubit must remain)
    @_optimize(speed)
    @_eagerMove
    public func partialTrace(over qubitsToTrace: [Int]) -> DensityMatrix {
        ValidationUtilities.validateNonEmpty(qubitsToTrace, name: "Qubits to trace")
        ValidationUtilities.validateOperationQubits(qubitsToTrace, numQubits: qubits)
        ValidationUtilities.validateUniqueQubits(qubitsToTrace)

        let remainingQubits = qubits - qubitsToTrace.count
        ValidationUtilities.validatePositiveInt(remainingQubits, name: "Remaining qubits after partial trace")

        let tracedSet = Set(qubitsToTrace)
        let keptQubits = (0 ..< qubits).filter { !tracedSet.contains($0) }

        let newDim = 1 << remainingQubits
        let oldDim = dimension
        let newSize = newDim * newDim

        var newElements = [Complex<Double>](repeating: .zero, count: newSize)

        for oldRow in 0 ..< oldDim {
            for oldCol in 0 ..< oldDim {
                var tracedBitsMatch = true
                for q in qubitsToTrace {
                    if BitUtilities.getBit(oldRow, qubit: q) != BitUtilities.getBit(oldCol, qubit: q) {
                        tracedBitsMatch = false
                        break
                    }
                }

                if tracedBitsMatch {
                    var newRow = 0
                    var newCol = 0
                    for (newIdx, oldIdx) in keptQubits.enumerated() {
                        if BitUtilities.getBit(oldRow, qubit: oldIdx) == 1 {
                            newRow |= (1 << newIdx)
                        }
                        if BitUtilities.getBit(oldCol, qubit: oldIdx) == 1 {
                            newCol |= (1 << newIdx)
                        }
                    }
                    newElements[newRow * newDim + newCol] =
                        newElements[newRow * newDim + newCol] + elements[oldRow * oldDim + oldCol]
                }
            }
        }

        return DensityMatrix(qubits: remainingQubits, elements: newElements)
    }

    // MARK: - Gate Application

    /// Apply unitary gate: ρ -> UρU†.
    ///
    /// Transforms density matrix by conjugation with gate unitary. This is the
    /// density matrix equivalent of |ψ⟩ -> U|ψ⟩ for statevectors.
    ///
    /// **Example:**
    /// ```swift
    /// var dm = DensityMatrix(qubits: 1)
    /// dm = dm.applying(.hadamard, to: 0)
    /// dm.probability(of: 0)  // 0.5
    /// dm.probability(of: 1)  // 0.5
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply
    ///   - targetQubits: Target qubit indices
    /// - Returns: Transformed density matrix UρU†
    /// - Complexity: O(4^n) for general case
    /// - Precondition: All indices in [0, qubits)
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public func applying(_ gate: QuantumGate, to targetQubits: [Int]) -> DensityMatrix {
        ValidationUtilities.validateOperationQubits(targetQubits, numQubits: qubits)

        switch gate.qubitsRequired {
        case 1:
            ValidationUtilities.validateSingleQubitGate(targetQubits)
            return applySingleQubitGate(gate: gate, qubit: targetQubits[0])
        case 2:
            return applyTwoQubitGate(gate: gate, qubit0: targetQubits[0], qubit1: targetQubits[1])
        default:
            return applyGeneralGate(gate: gate, targetQubits: targetQubits)
        }
    }

    /// Apply unitary gate to single qubit: ρ -> UρU†.
    ///
    /// **Example:**
    /// ```swift
    /// var dm = DensityMatrix(qubits: 1)
    /// dm = dm.applying(.pauliX, to: 0)
    /// dm.probability(of: 1)  // 1.0
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Single-qubit quantum gate to apply
    ///   - qubit: Target qubit index
    /// - Returns: Transformed density matrix UρU†
    /// - Complexity: O(4^n)
    /// - Precondition: 0 ≤ qubit < qubits
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public func applying(_ gate: QuantumGate, to qubit: Int) -> DensityMatrix {
        applying(gate, to: [qubit])
    }

    /// Apply single-qubit gate ρ -> UρU† via 2*2 block processing.
    @_optimize(speed)
    @_eagerMove
    private func applySingleQubitGate(gate: QuantumGate, qubit: Int) -> DensityMatrix {
        let gateMatrix = gate.matrix()
        let u00 = gateMatrix[0][0], u01 = gateMatrix[0][1]
        let u10 = gateMatrix[1][0], u11 = gateMatrix[1][1]

        let u00d = u00.conjugate, u01d = u01.conjugate
        let u10d = u10.conjugate, u11d = u11.conjugate

        let dim = dimension
        let size = dim * dim
        let mask = 1 << qubit

        var newElements = [Complex<Double>](unsafeUninitializedCapacity: size) { buffer, count in
            buffer.initialize(repeating: .zero)
            count = size
        }

        for row in 0 ..< dim {
            let row0 = row & ~mask
            let row1 = row | mask
            let rowBit = (row >> qubit) & 1

            for col in 0 ..< dim {
                let col0 = col & ~mask
                let col1 = col | mask
                let colBit = (col >> qubit) & 1

                let rho00 = elements[row0 * dim + col0]
                let rho01 = elements[row0 * dim + col1]
                let rho10 = elements[row1 * dim + col0]
                let rho11 = elements[row1 * dim + col1]

                let uRow0 = rowBit == 0 ? u00 : u10
                let uRow1 = rowBit == 0 ? u01 : u11
                let uCol0d = colBit == 0 ? u00d : u10d
                let uCol1d = colBit == 0 ? u01d : u11d

                let newVal = uRow0 * rho00 * uCol0d +
                    uRow0 * rho01 * uCol1d +
                    uRow1 * rho10 * uCol0d +
                    uRow1 * rho11 * uCol1d

                newElements[row * dim + col] = newVal
            }
        }

        return DensityMatrix(qubits: qubits, elements: newElements)
    }

    /// Optimized two-qubit gate application.
    @_optimize(speed)
    @_eagerMove
    private func applyTwoQubitGate(gate: QuantumGate, qubit0: Int, qubit1: Int) -> DensityMatrix {
        let gateMatrix = gate.matrix()
        let dim = dimension
        let size = dim * dim
        let mask0 = 1 << qubit0
        let mask1 = 1 << qubit1

        var newElements = [Complex<Double>](repeating: .zero, count: size)

        for row in 0 ..< dim {
            for col in 0 ..< dim {
                var sum = Complex<Double>.zero

                for a in 0 ..< 4 {
                    let aRow = (row & ~mask0 & ~mask1) |
                        (((a >> 1) & 1) << qubit0) |
                        ((a & 1) << qubit1)

                    for b in 0 ..< 4 {
                        let bCol = (col & ~mask0 & ~mask1) |
                            (((b >> 1) & 1) << qubit0) |
                            ((b & 1) << qubit1)

                        let rowIdx = (((row >> qubit0) & 1) << 1) | ((row >> qubit1) & 1)
                        let colIdx = (((col >> qubit0) & 1) << 1) | ((col >> qubit1) & 1)

                        let uElement = gateMatrix[rowIdx][a]
                        let uDaggerElement = gateMatrix[b][colIdx].conjugate
                        let rhoElement = elements[aRow * dim + bCol]

                        sum = sum + uElement * rhoElement * uDaggerElement
                    }
                }

                newElements[row * dim + col] = sum
            }
        }

        return DensityMatrix(qubits: qubits, elements: newElements)
    }

    /// General gate application using full matrix expansion.
    @_optimize(speed)
    @_eagerMove
    private func applyGeneralGate(gate: QuantumGate, targetQubits: [Int]) -> DensityMatrix {
        let fullU = expandGateToFullSpace(gate: gate, targetQubits: targetQubits)
        let fullUDagger = MatrixUtilities.hermitianConjugate(fullU)

        let dim = dimension

        var rho2D = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: dim), count: dim)
        for i in 0 ..< dim {
            for j in 0 ..< dim {
                rho2D[i][j] = elements[i * dim + j]
            }
        }

        let uRho = MatrixUtilities.matrixMultiply(fullU, rho2D)
        let result2D = MatrixUtilities.matrixMultiply(uRho, fullUDagger)

        let size = dim * dim
        let newElements = [Complex<Double>](unsafeUninitializedCapacity: size) { buffer, count in
            for i in 0 ..< dim {
                for j in 0 ..< dim {
                    buffer[i * dim + j] = result2D[i][j]
                }
            }
            count = size
        }

        return DensityMatrix(qubits: qubits, elements: newElements)
    }

    /// Expand gate matrix to full Hilbert space.
    private func expandGateToFullSpace(gate: QuantumGate, targetQubits: [Int]) -> [[Complex<Double>]] {
        let gateMatrix = gate.matrix()
        let gateDim = gateMatrix.count
        let fullDim = dimension

        var fullMatrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: fullDim),
            count: fullDim,
        )

        for i in 0 ..< fullDim {
            for j in 0 ..< fullDim {
                var nonTargetMatch = true
                for q in 0 ..< qubits where !targetQubits.contains(q) {
                    if BitUtilities.getBit(i, qubit: q) != BitUtilities.getBit(j, qubit: q) {
                        nonTargetMatch = false
                        break
                    }
                }

                if nonTargetMatch {
                    var gateRow = 0
                    var gateCol = 0
                    for (idx, q) in targetQubits.enumerated() {
                        if BitUtilities.getBit(i, qubit: q) == 1 {
                            gateRow |= (1 << idx)
                        }
                        if BitUtilities.getBit(j, qubit: q) == 1 {
                            gateCol |= (1 << idx)
                        }
                    }
                    if gateRow < gateDim, gateCol < gateDim {
                        fullMatrix[i][j] = gateMatrix[gateRow][gateCol]
                    }
                }
            }
        }

        return fullMatrix
    }

    // MARK: - CustomStringConvertible

    /// String representation showing significant diagonal probabilities.
    public var description: String {
        let dim = dimension
        let threshold = 1e-6
        var terms: [String] = []
        terms.reserveCapacity(min(dim, 16))

        for i in 0 ..< dim {
            let prob = elements[i * dim + i].real
            if prob > threshold {
                let probStr = String(format: "%.4f", prob)
                let binaryStr = String(i, radix: 2)
                let paddedBinary = String(repeating: "0", count: max(0, qubits - binaryStr.count)) + binaryStr
                terms.append("\(probStr)|" + paddedBinary + "⟩⟨" + paddedBinary + "|")
            }
        }

        let purityStr = String(format: "%.4f", purity())
        let header = "DensityMatrix(\(qubits) qubits, purity=\(purityStr))"

        return terms.isEmpty
            ? "\(header): near-zero"
            : "\(header): " + terms.joined(separator: " + ")
    }
}
