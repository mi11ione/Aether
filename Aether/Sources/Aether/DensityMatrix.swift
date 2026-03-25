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
    @usableFromInline var elements: [Complex<Double>]

    /// Number of qubits in this quantum system.
    ///
    /// **Example:**
    /// ```swift
    /// let dm = DensityMatrix(qubits: 2)
    /// dm.qubits  // 2
    /// ```
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

    /// Return type for ``mostProbableState()``, containing the basis state index and its probability.
    public typealias MostProbableResult = (index: Int, probability: Double)

    /// Probability threshold below which diagonal entries are omitted from ``description``.
    private static let descriptionProbabilityThreshold: Double = 1e-6

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
    /// dm[row: 0, col: 1]  // 0.5 (off-diagonal coherence)
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
            buffer.initialize(repeating: .zero)
            pureState.amplitudes.withUnsafeBufferPointer { ampsBuf in
                let ampsPtr = UnsafeRawPointer(ampsBuf.baseAddress!).assumingMemoryBound(to: Double.self) // Safety: amplitudes is non-empty (>= 2 elements)
                let bufPtr = UnsafeMutableRawPointer(buffer.baseAddress!).assumingMemoryBound(to: Double.self) // Safety: buffer has capacity dim*dim where dim >= 2
                var alpha = (1.0, 0.0)
                withUnsafePointer(to: &alpha) { alphaPtr in
                    cblas_zgerc(CblasRowMajor, Int32(dim), Int32(dim),
                                OpaquePointer(alphaPtr),
                                OpaquePointer(ampsPtr), Int32(1),
                                OpaquePointer(ampsPtr), Int32(1),
                                OpaquePointer(bufPtr), Int32(dim))
                }
            }
            count = size
        }
    }

    /// Initialize density matrix from explicit elements.
    ///
    /// Creates density matrix from flattened row-major element array. Validates that
    /// element count matches dimension squared. Does not validate positive semidefiniteness
    /// or trace normalization for performance; use `isHermitian()` and `isTraceNormalized()` to check.
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
    /// - Precondition: `qubits` must be positive.
    /// - Precondition: `qubits` must be at most 14.
    /// - Precondition: `elements.count` must equal `4^qubits`.
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

    /// Access density matrix element ρ[row, col].
    ///
    /// **Example:**
    /// ```swift
    /// let dm = DensityMatrix(qubits: 1)
    /// dm[row: 0, col: 0]  // 1.0
    /// ```
    ///
    /// - Parameters:
    ///   - row: Row index (0 to dimension-1)
    ///   - col: Column index (0 to dimension-1)
    /// - Returns: Complex element ρ[row, col]
    /// - Complexity: O(1)
    /// - Precondition: 0 ≤ row < dimension
    /// - Precondition: 0 ≤ col < dimension
    /// - Note: Setting may invalidate trace normalization or positive semidefiniteness
    public subscript(row row: Int, col col: Int) -> Complex<Double> {
        get {
            let dim = dimension
            ValidationUtilities.validateIndexInBounds(row, bound: dim, name: "Row index")
            ValidationUtilities.validateIndexInBounds(col, bound: dim, name: "Column index")
            return elements[row * dim + col]
        }
        set {
            let dim = dimension
            ValidationUtilities.validateIndexInBounds(row, bound: dim, name: "Row index")
            ValidationUtilities.validateIndexInBounds(col, bound: dim, name: "Column index")
            elements[row * dim + col] = newValue
        }
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
        return elements.withUnsafeBufferPointer { buf in
            let base = UnsafeRawPointer(buf.baseAddress!).assumingMemoryBound(to: Double.self) // Safety: elements is non-empty
            let stride = vDSP_Stride((dim + 1) * 2)
            var sum = 0.0
            vDSP_sveD(base, stride, &sum, vDSP_Length(dim))
            return sum
        }
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
        return elements.withUnsafeBufferPointer { buf in
            let base = UnsafeRawPointer(buf.baseAddress!).assumingMemoryBound(to: Double.self) // Safety: elements is non-empty
            let totalDoubles = dim * dim * 2
            var sum = 0.0
            vDSP_svesqD(base, 1, &sum, vDSP_Length(totalDoubles))
            return sum
        }
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
    @inlinable
    @_effects(readonly)
    public func isPure(tolerance: Double = 1e-10) -> Bool {
        abs(purity() - 1.0) < tolerance
    }

    /// Check if trace equals 1.0 (probability conservation).
    ///
    /// **Example:**
    /// ```swift
    /// let dm = DensityMatrix(qubits: 1)
    /// dm.isTraceNormalized()  // true
    /// ```
    ///
    /// - Parameter tolerance: Numerical tolerance (default 1e-10)
    /// - Returns: True if trace is within tolerance of 1.0
    /// - Complexity: O(2^n)
    @inlinable
    @_effects(readonly)
    public func isTraceNormalized(tolerance: Double = 1e-10) -> Bool {
        abs(trace() - 1.0) < tolerance
    }

    /// Check if density matrix is Hermitian (ρ† = ρ).
    ///
    /// All valid density matrices must be Hermitian, guaranteeing real eigenvalues
    /// (probabilities) and self-adjoint measurement operators.
    ///
    /// **Example:**
    /// ```swift
    /// let dm = DensityMatrix(qubits: 1)
    /// dm.isHermitian()  // true
    /// ```
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
    @inlinable
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
            elements.withUnsafeBufferPointer { elBuf in
                let base = UnsafeRawPointer(elBuf.baseAddress!).assumingMemoryBound(to: Double.self) // Safety: elements is non-empty
                let srcStride = Int32((dim + 1) * 2)
                cblas_dcopy(Int32(dim), base, srcStride, buffer.baseAddress!, 1) // Safety: buffer has capacity dim >= 2
            }
            count = dim
        }
    }

    /// Find basis state with highest probability.
    ///
    /// **Example:**
    /// ```swift
    /// let dm = DensityMatrix(qubits: 1)
    /// let result = dm.mostProbableState()
    /// result.index  // 0
    /// ```
    ///
    /// - Returns: Tuple of (basisStateIndex, probability)
    /// - Complexity: O(2^n)
    @_optimize(speed)
    @_effects(readonly)
    public func mostProbableState() -> MostProbableResult {
        let dim = dimension
        return elements.withUnsafeBufferPointer { buf in
            let base = UnsafeRawPointer(buf.baseAddress!).assumingMemoryBound(to: Double.self) // Safety: elements is non-empty
            let stride = vDSP_Stride((dim + 1) * 2)
            var maxVal = 0.0
            var maxIdx: vDSP_Length = 0
            vDSP_maxviD(base, stride, &maxVal, &maxIdx, vDSP_Length(dim))
            return (Int(maxIdx) / ((dim + 1) * 2), maxVal)
        }
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
        var sumReal = 0.0
        var sumImag = 0.0

        for i in 0 ..< dim {
            let (col, phase) = pauliString.applyToRow(row: i)
            let elem = elements[col * dim + i]
            let product = phase * elem
            sumReal += product.real
            sumImag += product.imaginary
        }

        return sumReal
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
    @_effects(readonly)
    @_eagerMove
    public func partialTrace(over qubitsToTrace: [Int]) -> DensityMatrix {
        ValidationUtilities.validateNonEmpty(qubitsToTrace, name: "Qubits to trace")
        ValidationUtilities.validateOperationQubits(qubitsToTrace, numQubits: qubits)
        ValidationUtilities.validateUniqueQubits(qubitsToTrace)

        let remainingQubits = qubits - qubitsToTrace.count
        ValidationUtilities.validatePositiveInt(remainingQubits, name: "Remaining qubits after partial trace")

        var tracedMask = 0
        for q in qubitsToTrace {
            tracedMask |= (1 << q)
        }
        let keptQubits = (0 ..< qubits).filter { (tracedMask >> $0) & 1 == 0 }

        let newDim = 1 << remainingQubits
        let oldDim = dimension
        let newSize = newDim * newDim

        let keptMasks = keptQubits.map { 1 << $0 }

        var newElements = [Complex<Double>](unsafeUninitializedCapacity: newSize) { buffer, count in
            buffer.initialize(repeating: .zero)
            count = newSize
        }

        for oldRow in 0 ..< oldDim {
            for oldCol in 0 ..< oldDim {
                let tracedBitsMatch = (oldRow ^ oldCol) & tracedMask == 0

                if tracedBitsMatch {
                    var newRow = 0
                    var newCol = 0
                    for newIdx in 0 ..< keptQubits.count {
                        let oldMask = keptMasks[newIdx]
                        let newBit = 1 << newIdx
                        if oldRow & oldMask != 0 { newRow |= newBit }
                        if oldCol & oldMask != 0 { newCol |= newBit }
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

    /// Applies a circuit operation to this density matrix.
    ///
    /// Routes unitary gates through the gate application pipeline and non-unitary operations
    /// through dedicated quantum channel handlers.
    ///
    /// - Parameter operation: The circuit operation to apply.
    /// - Returns: The transformed density matrix.
    /// - Complexity: O(4^n) where n is the number of qubits.
    ///
    /// **Example:**
    /// ```swift
    /// var rho = DensityMatrix(pureState: QuantumState(qubits: 1))
    /// let op = CircuitOperation.gate(.hadamard, qubits: [0])
    /// rho = rho.applying(op)
    /// ```
    @_optimize(speed)
    @_effects(readonly)
    public func applying(_ operation: CircuitOperation) -> DensityMatrix {
        switch operation {
        case let .gate(gate, qubits, _):
            applying(gate, to: qubits)
        case let .reset(qubit, _):
            applyReset(qubit: qubit)
        case let .measure(qubit, _, _):
            applyReset(qubit: qubit)
        }
    }

    /// Apply single-qubit gate ρ -> UρU† via 2*2 block processing.
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    private func applySingleQubitGate(gate: QuantumGate, qubit: Int) -> DensityMatrix {
        let gateMatrix = gate.matrix()
        let u00 = gateMatrix[0][0], u01 = gateMatrix[0][1]
        let u10 = gateMatrix[1][0], u11 = gateMatrix[1][1]

        let u00d = u00.conjugate, u01d = u01.conjugate
        let u10d = u10.conjugate, u11d = u11.conjugate

        let uRows = [(u00, u01), (u10, u11)]
        let uColsD = [(u00d, u01d), (u10d, u11d)]

        let dim = dimension
        let size = dim * dim
        let mask = 1 << qubit

        var newElements = [Complex<Double>](unsafeUninitializedCapacity: size) { _, count in
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

                let (uRow0, uRow1) = uRows[rowBit]
                let (uCol0d, uCol1d) = uColsD[colBit]

                let newVal = uRow0 * rho00 * uCol0d +
                    uRow0 * rho01 * uCol1d +
                    uRow1 * rho10 * uCol0d +
                    uRow1 * rho11 * uCol1d

                newElements[row * dim + col] = newVal
            }
        }

        return DensityMatrix(qubits: qubits, elements: newElements)
    }

    /// Apply reset channel on the specified qubit by tracing out and replacing with |0⟩.
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    private func applyReset(qubit: Int) -> DensityMatrix {
        let dim = dimension
        let size = dim * dim
        let mask = 1 << qubit

        var newElements = [Complex<Double>](unsafeUninitializedCapacity: size) { buffer, count in
            buffer.initialize(repeating: .zero)
            count = size
        }

        for row in 0 ..< dim where row & mask == 0 {
            let row0 = row & ~mask

            for col in 0 ..< dim where col & mask == 0 {
                let col0 = col & ~mask

                let rho00 = elements[row0 * dim + col0]
                let rho11 = elements[(row0 | mask) * dim + (col0 | mask)]

                newElements[row0 * dim + col0] = rho00 + rho11
            }
        }

        return DensityMatrix(qubits: qubits, elements: newElements)
    }

    /// Optimized two-qubit gate application.
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    private func applyTwoQubitGate(gate: QuantumGate, qubit0: Int, qubit1: Int) -> DensityMatrix {
        let gateMatrix = gate.matrix()
        let dim = dimension
        let size = dim * dim
        let mask0 = 1 << qubit0
        let mask1 = 1 << qubit1

        var newElements = [Complex<Double>](unsafeUninitializedCapacity: size) { buffer, count in
            buffer.initialize(repeating: .zero)
            count = size
        }

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

    /// Apply general multi-qubit gate via full matrix expansion with flat BLAS multiply.
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    private func applyGeneralGate(gate: QuantumGate, targetQubits: [Int]) -> DensityMatrix {
        let dim = dimension
        let nn = dim * dim

        var fullU = expandGateToFullSpaceFlat(gate: gate, targetQubits: targetQubits)

        var fullUDagger = [Complex<Double>](unsafeUninitializedCapacity: nn) { buffer, count in
            for i in 0 ..< dim {
                for j in 0 ..< dim {
                    buffer[i * dim + j] = fullU[j * dim + i].conjugate
                }
            }
            count = nn
        }

        var alpha = (1.0, 0.0)
        var beta = (0.0, 0.0)

        var temp = [Complex<Double>](unsafeUninitializedCapacity: nn) { _, count in
            count = nn
        }

        fullU.withUnsafeMutableBufferPointer { uPtr in
            elements.withUnsafeBufferPointer { rhoPtr in
                temp.withUnsafeMutableBufferPointer { tPtr in
                    withUnsafeMutablePointer(to: &alpha) { alphaPtr in
                        withUnsafeMutablePointer(to: &beta) { betaPtr in
                            cblas_zgemm(
                                CblasRowMajor,
                                CblasNoTrans,
                                CblasNoTrans,
                                Int32(dim), Int32(dim), Int32(dim),
                                OpaquePointer(alphaPtr),
                                OpaquePointer(uPtr.baseAddress), Int32(dim),
                                OpaquePointer(rhoPtr.baseAddress), Int32(dim),
                                OpaquePointer(betaPtr),
                                OpaquePointer(tPtr.baseAddress), Int32(dim),
                            )
                        }
                    }
                }
            }
        }

        var newElements = [Complex<Double>](unsafeUninitializedCapacity: nn) { _, count in
            count = nn
        }

        temp.withUnsafeMutableBufferPointer { tPtr in
            fullUDagger.withUnsafeMutableBufferPointer { udPtr in
                newElements.withUnsafeMutableBufferPointer { rPtr in
                    withUnsafeMutablePointer(to: &alpha) { alphaPtr in
                        withUnsafeMutablePointer(to: &beta) { betaPtr in
                            cblas_zgemm(
                                CblasRowMajor,
                                CblasNoTrans,
                                CblasNoTrans,
                                Int32(dim), Int32(dim), Int32(dim),
                                OpaquePointer(alphaPtr),
                                OpaquePointer(tPtr.baseAddress), Int32(dim),
                                OpaquePointer(udPtr.baseAddress), Int32(dim),
                                OpaquePointer(betaPtr),
                                OpaquePointer(rPtr.baseAddress), Int32(dim),
                            )
                        }
                    }
                }
            }
        }

        return DensityMatrix(qubits: qubits, elements: newElements)
    }

    /// Expand gate matrix to full Hilbert space as flat row-major array.
    @_optimize(speed)
    @_effects(readonly)
    private func expandGateToFullSpaceFlat(gate: QuantumGate, targetQubits: [Int]) -> [Complex<Double>] {
        let gateMatrix = gate.matrix()
        let gateDim = gateMatrix.count
        let fullDim = dimension

        var targetMask = 0
        for q in targetQubits {
            targetMask |= (1 << q)
        }
        let nonTargetMask = ((1 << qubits) - 1) & ~targetMask

        var flatMatrix = [Complex<Double>](unsafeUninitializedCapacity: fullDim * fullDim) { buffer, count in
            buffer.initialize(repeating: .zero)
            count = fullDim * fullDim
        }

        for i in 0 ..< fullDim {
            for j in 0 ..< fullDim {
                let nonTargetMatch = (i ^ j) & nonTargetMask == 0

                if nonTargetMatch {
                    var gateRow = 0
                    var gateCol = 0
                    for idx in 0 ..< targetQubits.count {
                        let q = targetQubits[idx]
                        if BitUtilities.bit(i, qubit: q) == 1 {
                            gateRow |= (1 << idx)
                        }
                        if BitUtilities.bit(j, qubit: q) == 1 {
                            gateCol |= (1 << idx)
                        }
                    }
                    if gateRow < gateDim, gateCol < gateDim {
                        flatMatrix[i * fullDim + j] = gateMatrix[gateRow][gateCol]
                    }
                }
            }
        }

        return flatMatrix
    }

    // MARK: - State Conversion

    /// Extract pure state from density matrix via eigendecomposition.
    ///
    /// Returns the eigenvector corresponding to eigenvalue 1 (the pure state).
    /// For mixed states, returns the dominant eigenvector.
    ///
    /// **Example:**
    /// ```swift
    /// let dm = DensityMatrix(pureState: QuantumState(qubits: 1))
    /// let state = dm.toQuantumState()
    /// ```
    ///
    /// - Returns: QuantumState representing the pure state
    /// - Complexity: O(d^3) where d = 2^n for eigendecomposition
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public func toQuantumState() -> QuantumState {
        let dim = dimension
        let matrix = extractMatrix()
        let eigen = HermitianEigenDecomposition.decompose(matrix: matrix)
        var maxVal: Double = 0
        var maxIdx: vDSP_Length = 0
        eigen.eigenvalues.withUnsafeBufferPointer { buf in
            vDSP_maxviD(buf.baseAddress!, 1, &maxVal, &maxIdx, vDSP_Length(buf.count)) // Safety: eigenvalues is non-empty
        }
        let dominantIdx = Int(maxIdx)
        let amplitudes = eigen.eigenvectors.map { $0[dominantIdx] }
        return QuantumState(qubits: qubits, amplitudes: amplitudes)
    }

    // MARK: - CustomStringConvertible

    /// String representation showing significant diagonal probabilities.
    ///
    /// **Example:**
    /// ```swift
    /// let dm = DensityMatrix(qubits: 1)
    /// print(dm.description)
    /// ```
    ///
    /// - Complexity: O(2^n) where n is the number of qubits.
    public var description: String {
        let dim = dimension
        let threshold = Self.descriptionProbabilityThreshold
        var terms: [String] = []
        terms.reserveCapacity(min(dim, 16))

        for i in 0 ..< dim {
            guard terms.count < 32 else {
                terms.append("...")
                break
            }
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
