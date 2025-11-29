// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import GameplayKit

/// Computational basis measurement outcome with collapsed state
///
/// Encapsulates the result of measuring all qubits in the computational basis (Z-basis).
/// Contains the classical measurement outcome (basis state index) and the post-measurement
/// collapsed state. Implements the measurement postulate from quantum mechanics.
///
/// **Born rule**: Measurement yields outcome i with probability P(i) = |cᵢ|²
///
/// **State collapse**: Post-measurement state is deterministic basis state |i⟩
///
/// **Example**:
/// ```swift
/// let bell = QuantumCircuit.bellPhiPlus().execute()
/// let result = Measurement.measure(bell)
/// // result.outcome: 0 or 3 (50% each)
/// // result.collapsedState: |00⟩ or |11⟩
/// ```
///
/// - SeeAlso: ``Measurement/measure(_:seed:)``
public struct MeasurementResult: Equatable, CustomStringConvertible {
    /// Classical outcome: basis state index i ∈ [0, 2^n-1]
    public let outcome: Int

    /// Post-measurement state |i⟩ collapsed to measured outcome
    public let collapsedState: QuantumState

    public var description: String {
        "Measurement: outcome=\(outcome), state=\(collapsedState)"
    }
}

/// Partial measurement outcome for qubit subset
///
/// Result of measuring a subset of qubits while preserving quantum coherence
/// in unmeasured qubits. Supports both single-qubit and multi-qubit partial
/// measurements with a unified API.
///
/// **Single-qubit convenience**: Use `.outcome` property for single-qubit measurements
///
/// **Multi-qubit access**: Use `.outcomes` array for multiple qubits
///
/// **Example**:
/// ```swift
/// let bell = QuantumCircuit.bellPhiPlus().execute()
/// let result = Measurement.measure(0, in: bell)
/// print(result.outcome)  // 0 or 1
/// print(result.outcomes) // [0] or [1]
/// ```
///
/// - SeeAlso: ``Measurement``
public struct PartialMeasurementResult: Equatable, CustomStringConvertible {
    /// Measurement outcomes: length 1 for single qubit, N for multiple qubits
    public let outcomes: [Int]

    /// Post-measurement partially collapsed state with unmeasured qubits in superposition
    public let collapsedState: QuantumState

    /// Convenience accessor for single-qubit measurements
    ///
    /// **Example**:
    /// ```swift
    /// let result = Measurement.measure(0, in: state)
    /// print(result.outcome)  // Direct access (equivalent to outcomes[0])
    /// ```
    ///
    /// - Precondition: `outcomes.count == 1`
    public var outcome: Int {
        precondition(outcomes.count == 1, "outcome property only valid for single-qubit measurement")
        return outcomes[0]
    }

    public var description: String {
        let outcomeStr = outcomes.count == 1
            ? "outcome=\(outcomes[0])"
            : "outcomes=\(outcomes)"
        return "PartialMeasurement: \(outcomeStr), state=\(collapsedState)"
    }
}

/// Custom basis measurement outcome
///
/// Result of measuring a qubit in an arbitrary single-qubit basis defined by
/// a provided basis state |ψ⟩. Returns outcome (0 or 1) corresponding to
/// projection onto {|ψ⟩, |ψ⊥⟩} basis.
///
/// **Basis definition**: Provide normalized 2D complex vector |ψ⟩ = c₀|0⟩ + c₁|1⟩
///
/// **Orthogonal complement**: |ψ⊥⟩ constructed automatically via unitarity
///
/// **Example**:
/// ```swift
/// let customBasis = [Complex(1/sqrt(2), 0), Complex(1/sqrt(2), 0)]  // |+⟩
/// let zero = QuantumState(numQubits: 1)
/// let result = Measurement.measure(0, basis: customBasis, in: zero)
/// // result.outcome: 0 or 1 (50% each for |0⟩ measured in |+⟩ basis)
/// ```
///
/// - SeeAlso: ``Measurement``
public struct CustomBasisMeasurementResult: Equatable, CustomStringConvertible {
    /// Measurement outcome: 0 (projected onto |ψ⟩) or 1 (projected onto |ψ⊥⟩)
    public let outcome: Int

    /// Post-measurement collapsed state in original computational basis
    public let collapsedState: QuantumState

    public var description: String {
        "CustomBasisMeasurement: outcome=\(outcome), state=\(collapsedState)"
    }
}

/// Pauli measurement basis selector
///
/// Defines the measurement basis for single-qubit Pauli measurements.
/// Each basis corresponds to measuring a Pauli observable (X, Y, or Z)
/// with eigenvalues ±1.
///
/// **Basis eigenstates**:
/// - **X**: |+⟩ = (|0⟩ + |1⟩)/√2 (λ=+1), |-⟩ = (|0⟩ - |1⟩)/√2 (λ=-1)
/// - **Y**: |+i⟩ = (|0⟩ + i|1⟩)/√2 (λ=+1), |-i⟩ = (|0⟩ - i|1⟩)/√2 (λ=-1)
/// - **Z**: |0⟩ (λ=+1), |1⟩ (λ=-1)
///
/// **Example**:
/// ```swift
/// let plus = QuantumState(numQubits: 1, amplitudes: [
///     Complex(1/sqrt(2), 0),
///     Complex(1/sqrt(2), 0)
/// ])
/// let xResult = Measurement.measure(0, basis: .x, in: plus)
/// // xResult.eigenvalue: +1 (deterministic, |+⟩ is X eigenstate)
///
/// let zResult = Measurement.measure(0, basis: .z, in: plus)
/// // zResult.eigenvalue: ±1 (50% each, |+⟩ is superposition in Z basis)
/// ```
///
/// - SeeAlso: ``Measurement``
public enum PauliBasis: String, CaseIterable, Sendable {
    /// X (bit-flip) basis with eigenstates |+⟩ and |-⟩
    case x

    /// Y (bit-flip + phase) basis with eigenstates |+i⟩ and |-i⟩
    case y

    /// Z (computational) basis with eigenstates |0⟩ and |1⟩
    case z
}

/// Pauli basis measurement outcome with eigenvalue
///
/// Result of measuring a qubit in X, Y, or Z basis. Unlike computational basis
/// measurements (which return 0/1), Pauli measurements return eigenvalues ±1
/// corresponding to positive/negative eigenstates.
///
/// **Eigenvalue interpretation**:
/// - +1: Qubit measured in positive eigenstate (|+⟩, |+i⟩, or |0⟩)
/// - -1: Qubit measured in negative eigenstate (|-⟩, |-i⟩, or |1⟩)
///
/// **Observable expectation**: ⟨σ⟩ = average of eigenvalues over many measurements
///
/// **Example**:
/// ```swift
/// let zero = QuantumState(numQubits: 1)
///
/// let xResult = Measurement.measure(0, basis: .x, in: zero)
/// // xResult.eigenvalue: ±1 (50% each, |0⟩ is superposition in X basis)
///
/// let zResult = Measurement.measure(0, basis: .z, in: zero)
/// // zResult.eigenvalue: +1 (deterministic, |0⟩ is Z eigenstate with λ=+1)
/// ```
///
/// - SeeAlso: ``Measurement``, ``PauliBasis``
public struct PauliMeasurementResult: Equatable, CustomStringConvertible {
    /// Pauli eigenvalue: +1 or -1
    public let eigenvalue: Int

    /// Post-measurement collapsed state in original computational basis
    public let collapsedState: QuantumState

    public var description: String {
        "PauliMeasurement: eigenvalue=\(eigenvalue > 0 ? "+1" : "-1"), state=\(collapsedState)"
    }
}

/// Single Pauli operator acting on specific qubit
///
/// Represents one Pauli operator (X, Y, or Z) applied to a specific qubit index.
/// Used as building block for ``PauliString`` tensor product observables.
///
/// **Factory methods**: Static methods `.x(_:)`, `.y(_:)`, `.z(_:)` for concise construction
///
/// **Example**:
/// ```swift
/// let x0 = PauliOperator(qubit: 0, basis: .x)  // X₀
/// let z2 = PauliOperator(qubit: 2, basis: .z)  // Z₂
///
/// let x0Concise = PauliOperator.x(0)  // Equivalent, more readable
/// let z2Concise = PauliOperator.z(2)
/// ```
///
/// - SeeAlso: ``PauliString``, ``PauliBasis``
public struct PauliOperator: Equatable, Hashable, Sendable {
    /// Target qubit index (0 to n-1)
    public let qubit: Int

    /// Pauli basis selector (X, Y, or Z)
    public let basis: PauliBasis

    @inlinable
    public init(qubit: Int, basis: PauliBasis) {
        self.qubit = qubit
        self.basis = basis
    }

    /// Create X Pauli operator on specified qubit
    ///
    /// **Example**:
    /// ```swift
    /// let x0 = PauliOperator.x(0)  // X₀
    /// ```
    ///
    /// - Parameter qubit: Target qubit index
    /// - Returns: X operator on qubit
    @inlinable
    public static func x(_ qubit: Int) -> PauliOperator {
        PauliOperator(qubit: qubit, basis: .x)
    }

    /// Create Y Pauli operator on specified qubit
    ///
    /// **Example**:
    /// ```swift
    /// let y1 = PauliOperator.y(1)  // Y₁
    /// ```
    ///
    /// - Parameter qubit: Target qubit index
    /// - Returns: Y operator on qubit
    @inlinable
    public static func y(_ qubit: Int) -> PauliOperator {
        PauliOperator(qubit: qubit, basis: .y)
    }

    /// Create Z Pauli operator on specified qubit
    ///
    /// **Example**:
    /// ```swift
    /// let z2 = PauliOperator.z(2)  // Z₂
    /// ```
    ///
    /// - Parameter qubit: Target qubit index
    /// - Returns: Z operator on qubit
    @inlinable
    public static func z(_ qubit: Int) -> PauliOperator {
        PauliOperator(qubit: qubit, basis: .z)
    }
}

/// Pauli string: tensor product of Pauli operators
///
/// Represents a multi-qubit observable as tensor product of Pauli operators,
/// e.g., X₀⊗Y₁⊗Z₂. Identity operators are implicit on unspecified qubits.
/// This is the fundamental observable representation for VQE and quantum chemistry.
///
/// **Mathematical form**: O = ⊗ᵢ Pᵢ where Pᵢ ∈ {I, X, Y, Z}
///
/// **Eigenvalues**: Product of individual Pauli eigenvalues (±1)
///
/// **Construction**: Use variadic initializer or array for programmatic construction
///
/// **Example**:
/// ```swift
/// let z0 = PauliString(.z(0))                    // Z₀
/// let xz = PauliString(.x(0), .z(1))             // X₀⊗Z₁
///
/// let ops: [PauliOperator] = [.x(0), .y(1), .z(2)]
/// let xyz = PauliString(ops)                      // X₀⊗Y₁⊗Z₂
///
/// let bell = QuantumCircuit.bellPhiPlus().execute()
/// let result = Measurement.measure(xz, in: bell)
/// // result.eigenvalue: ±1 (product of X₀ and Z₁ eigenvalues)
/// ```
///
/// - SeeAlso: ``PauliOperator``, ``Measurement``
public struct PauliString: Equatable, Hashable, CustomStringConvertible, Sendable {
    /// Array of Pauli operators (identity implicit on unspecified qubits)
    public let operators: [PauliOperator]

    /// Create Pauli string from operator array
    ///
    /// **Example**:
    /// ```swift
    /// let ops: [PauliOperator] = [.x(0), .z(1)]
    /// let ps = PauliString(ops)  // X₀⊗Z₁
    /// ```
    ///
    /// - Parameter operators: Array of Pauli operators
    public init(_ operators: [PauliOperator]) {
        self.operators = operators
    }

    /// Create Pauli string from variadic operator list
    ///
    /// **Example**:
    /// ```swift
    /// let single = PauliString(.z(0))           // Z₀
    /// let multi = PauliString(.x(0), .z(1))     // X₀⊗Z₁
    /// ```
    ///
    /// - Parameter operators: Variadic list of Pauli operators
    @inlinable
    public init(_ operators: PauliOperator...) {
        self.operators = operators
    }

    public var description: String {
        if operators.isEmpty {
            return "I (identity)"
        }

        let terms = operators.map { "\($0.basis.rawValue.uppercased())_\($0.qubit)" }
        return terms.joined(separator: "⊗")
    }
}

/// Individual qubit measurement outcome
///
/// Records the computational basis measurement result (0 or 1) for a specific qubit.
/// Used in multi-qubit Pauli measurement results to track individual qubit outcomes
/// before computing the product eigenvalue.
///
/// **Example**:
/// ```swift
/// let outcome = MeasurementOutcome(qubit: 0, outcome: 1)  // Qubit 0 measured as |1⟩
/// ```
///
/// - SeeAlso: ``PauliStringMeasurementResult``
public struct MeasurementOutcome: Equatable {
    /// Qubit index that was measured
    public let qubit: Int

    /// Measurement result: 0 or 1 (computational basis)
    public let outcome: Int
}

/// Pauli string measurement outcome with product eigenvalue
///
/// Result of measuring a multi-qubit Pauli string observable (tensor product of Paulis).
/// Contains the overall product eigenvalue (±1) and individual measurement outcomes
/// for each qubit in the Pauli string.
///
/// **Eigenvalue computation**: λ = ∏ᵢ λᵢ where λᵢ = (outcomeᵢ == 0) ? +1 : -1
///
/// **Use case**: Hamiltonian expectation values in VQE (⟨H⟩ = Σᵢ cᵢ⟨Pᵢ⟩)
///
/// **Example**:
/// ```swift
/// let zz = PauliString(.z(0), .z(1))
/// let bell = QuantumCircuit.bellPhiPlus().execute()
/// let result = Measurement.measure(zz, in: bell)
///
/// // Bell state: (|00⟩ + |11⟩)/√2
/// // 50% |00⟩: λ₀=+1, λ₁=+1 → product=+1
/// // 50% |11⟩: λ₀=-1, λ₁=-1 → product=+1
/// // result.eigenvalue: +1 (deterministic!)
/// ```
///
/// - SeeAlso: ``PauliString``, ``Measurement``
public struct PauliStringMeasurementResult: Equatable, CustomStringConvertible {
    /// Overall eigenvalue: product of individual Pauli eigenvalues (±1)
    public let eigenvalue: Int

    /// Post-measurement collapsed state in original computational basis
    public let collapsedState: QuantumState

    /// Individual measurement outcomes for each qubit in Pauli string
    public let individualOutcomes: [MeasurementOutcome]

    public var description: String {
        let outcomeStr = individualOutcomes.map { "q\($0.qubit)=\($0.outcome)" }.joined(separator: ", ")
        return "PauliStringMeasurement: eigenvalue=\(eigenvalue > 0 ? "+1" : "-1"), outcomes=[\(outcomeStr)]"
    }
}

/// Non-destructive state snapshot for debugging and analysis
///
/// Captures the full quantum state at a specific point without measurement collapse.
/// Preserves quantum coherence and superposition. Useful for debugging, algorithm
/// verification, and state tomography.
///
/// **Physical realizability**: Not physically realizable on quantum hardware (violates
/// no-cloning theorem). This is a simulation-only tool for analysis.
///
/// **Use cases**:
/// - Algorithm verification: Compare intermediate states to theory
/// - Debugging: Inspect wavefunction at specific circuit points
/// - State tomography: Reconstruct density matrix from multiple snapshots
/// - Visualization: Display wavefunction evolution over time
///
/// **Example**:
/// ```swift
/// var circuit = QuantumCircuit(numQubits: 2)
/// circuit.append(gate: .hadamard, to: 0)
/// let snapshot1 = Measurement.snapshot(of: circuit.execute(), label: "After H(0)")
/// // snapshot1.state: |+⟩⊗|0⟩ (no collapse, superposition preserved)
///
/// circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
/// let snapshot2 = Measurement.snapshot(of: circuit.execute(), label: "After CNOT")
/// // snapshot2.state: (|00⟩ + |11⟩)/√2 (Bell state)
/// ```
///
/// - SeeAlso: ``Measurement/snapshot(of:label:)``
public struct StateSnapshot: Equatable, CustomStringConvertible, Sendable {
    /// Captured quantum state (full statevector copy)
    public let state: QuantumState

    /// Optional descriptive label for snapshot identification
    public let label: String?

    /// Timestamp when snapshot was captured
    public let timestamp: Date

    public var description: String {
        if let label {
            return "Snapshot[\(label)]: \(state)"
        }
        return "Snapshot: \(state)"
    }
}

/// Chi-squared goodness-of-fit test result
///
/// Statistical test result for comparing observed measurement frequencies to
/// expected theoretical probabilities. Lower chi-squared values indicate better fit.
///
/// **Interpretation**:
/// - χ² ≈ 0: Perfect fit (observed matches expected exactly)
/// - χ² < degrees of freedom: Good fit (within statistical fluctuations)
/// - χ² >> degrees of freedom: Poor fit (systematic deviation from theory)
///
/// **Example**:
/// ```swift
/// let observed = [48, 52]      // Measured |0⟩ 48 times, |1⟩ 52 times
/// let expected = [0.5, 0.5]    // Theoretical 50/50 split
/// let result = Measurement.chiSquared(
///     observed: observed,
///     expected: expected,
///     totalShots: 100
/// )
/// // result.chiSquared: ~0.16 (good fit, χ² < 1 degree of freedom)
/// ```
///
/// - SeeAlso: ``Measurement/chiSquared(observed:expected:totalShots:)``
public struct ChiSquaredResult {
    /// Chi-squared statistic: Σ((observed - expected)² / expected)
    public let chiSquared: Double

    /// Degrees of freedom: number of tested bins - 1
    public let degreesOfFreedom: Int

    /// Number of bins with expectedCount ≥ 5 (included in test)
    public let testedBins: Int

    /// Number of bins with expectedCount < 5 (excluded from test)
    public let skippedBins: Int
}

/// Quantum measurement operations: Born rule implementation with state collapse
///
/// Static namespace providing projective measurement operations in computational,
/// Pauli, and custom bases. Implements the measurement postulate from quantum
/// mechanics with optional reproducibility via seeding.
///
/// **Measurement types**:
/// - **Computational basis**: Standard Z-basis measurement (|0⟩, |1⟩) returning 0/1
/// - **Pauli basis**: X, Y, or Z measurements returning eigenvalues ±1
/// - **Custom basis**: Arbitrary single-qubit unitary basis
/// - **Pauli strings**: Multi-qubit tensor product observables (X₀⊗Y₁⊗Z₂)
/// - **Partial measurement**: Measure qubit subset, preserve coherence in rest
/// - **Snapshots**: Non-destructive state capture (no collapse)
///
/// **Born rule**: Measurement yields outcome i with probability P(i) = |cᵢ|²
///
/// **State collapse**: Post-measurement state projects onto measured outcome
///
/// **Reproducibility**: Optional `seed` parameter enables deterministic results
///
/// **Performance**: Measurement is O(2^n) for full state, O(2^n) for partial measurement
///
/// **Example**:
/// ```swift
/// let bell = QuantumCircuit.bellPhiPlus().execute()
///
/// // Computational basis measurement
/// let result = Measurement.measure(bell)
/// // result.outcome: 0 (|00⟩) or 3 (|11⟩) with 50% probability each
///
/// // Pauli X measurement
/// let xResult = Measurement.measure(0, basis: .x, in: bell)
/// // xResult.eigenvalue: ±1
///
/// // Pauli string measurement
/// let zz = PauliString(.z(0), .z(1))
/// let zzResult = Measurement.measure(zz, in: bell)
/// // zzResult.eigenvalue: +1 (deterministic for Bell state)
///
/// // Seeded measurement for reproducibility
/// let r1 = Measurement.measure(bell, seed: 123)
/// let r2 = Measurement.measure(bell, seed: 123)
/// // r1.outcome == r2.outcome (same seed → same result)
///
/// // Non-destructive snapshot
/// let snapshot = Measurement.snapshot(of: bell, label: "Bell state")
/// // snapshot.state preserves superposition (no collapse)
/// ```
///
/// - SeeAlso: ``QuantumState``, ``QuantumCircuit``, ``PauliString``
public enum Measurement {
    // MARK: - Computational Basis Measurement

    /// Measure all qubits in computational basis (Born rule)
    ///
    /// Performs projective measurement of entire quantum state, sampling outcome
    /// according to Born rule probabilities P(i) = |cᵢ|². Collapses state to
    /// deterministic basis state |i⟩. This is the standard measurement operation
    /// in quantum computing.
    ///
    /// **Algorithm**:
    /// 1. Compute probability distribution P(i) = |cᵢ|² for all i ∈ [0, 2^n-1]
    /// 2. Sample outcome via roulette wheel selection
    /// 3. Collapse state to |outcome⟩
    ///
    /// **Example**:
    /// ```swift
    /// let plus = QuantumState(numQubits: 1, amplitudes: [
    ///     Complex(1/sqrt(2), 0),
    ///     Complex(1/sqrt(2), 0)
    /// ])
    /// let result = Measurement.measure(plus)
    /// // result.outcome: 0 or 1 (50% each)
    /// // result.collapsedState: |0⟩ or |1⟩
    ///
    /// let bell = QuantumCircuit.bellPhiPlus().execute()
    /// let bellResult = Measurement.measure(bell)
    /// // bellResult.outcome: 0 (|00⟩) or 3 (|11⟩) with equal probability
    ///
    /// let r1 = Measurement.measure(plus, seed: 123)
    /// let r2 = Measurement.measure(plus, seed: 123)
    /// // r1.outcome == r2.outcome (reproducible with same seed)
    /// ```
    ///
    /// - Parameters:
    ///   - state: Normalized quantum state to measure
    ///   - seed: Optional seed for reproducible results (default: system random)
    /// - Returns: Measurement result with outcome and collapsed state
    /// - Complexity: O(2^n) time for probability computation, O(2^n) space for state
    /// - Precondition: State must be normalized
    @_eagerMove
    public static func measure(_ state: QuantumState, seed: UInt64? = nil) -> MeasurementResult {
        ValidationUtilities.validateNormalizedState(state)

        let probabilities = state.probabilities()
        let outcome = sampleOutcome(probabilities: probabilities, seed: seed)
        let collapsedState = collapseToOutcome(outcome, numQubits: state.numQubits)

        return MeasurementResult(outcome: outcome, collapsedState: collapsedState)
    }

    // MARK: - Partial Measurement

    /// Measure single qubit, leaving others in superposition
    ///
    /// Performs projective measurement of one qubit while preserving quantum coherence
    /// in unmeasured qubits. Implements partial trace (marginalization). The collapsed
    /// state maintains entanglement structure for unmeasured qubits.
    ///
    /// **Algorithm**:
    /// 1. Calculate marginal probabilities P(qubit=0) and P(qubit=1)
    /// 2. Sample outcome (0 or 1)
    /// 3. Zero incompatible amplitudes, renormalize compatible ones
    ///
    /// **Example**:
    /// ```swift
    /// let bell = QuantumCircuit.bellPhiPlus().execute()  // (|00⟩ + |11⟩)/√2
    /// let result = Measurement.measure(0, in: bell)
    /// // result.outcome: 0 or 1 (50% each)
    /// // If outcome = 0: collapsedState = |00⟩ (qubit 1 also collapsed to |0⟩)
    /// // If outcome = 1: collapsedState = |11⟩ (qubit 1 also collapsed to |1⟩)
    /// ```
    ///
    /// - Parameters:
    ///   - qubit: Qubit index to measure (0 to n-1)
    ///   - state: Normalized quantum state
    ///   - seed: Optional seed for reproducible results
    /// - Returns: Partial measurement result with single outcome
    /// - Complexity: O(2^n) time, O(2^n) space
    /// - Precondition: State normalized, qubit index in bounds
    /// - SeeAlso: ``measure(_:in:seed:)`` for multi-qubit partial measurement
    @_eagerMove
    public static func measure(_ qubit: Int, in state: QuantumState, seed: UInt64? = nil) -> PartialMeasurementResult {
        measure([qubit], in: state, seed: seed)
    }

    /// Measure multiple qubits, leaving others in superposition
    ///
    /// Performs projective measurement of multiple qubits while preserving quantum
    /// coherence in unmeasured qubits. Generalizes single-qubit partial measurement
    /// to arbitrary qubit subsets.
    ///
    /// **Algorithm**:
    /// 1. Calculate joint probability distribution for specified qubits
    /// 2. Sample multi-qubit outcome (bitstring)
    /// 3. Zero incompatible amplitudes, renormalize compatible ones
    ///
    /// **Example**:
    /// ```swift
    /// let ghz = QuantumCircuit.ghzState(numQubits: 3).execute()  // (|000⟩ + |111⟩)/√2
    /// let result = Measurement.measure([0, 1], in: ghz)
    /// // result.outcomes: [0, 0] or [1, 1] (50% each)
    /// // result.collapsedState: |000⟩ or |111⟩ (qubit 2 collapses due to entanglement)
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Array of qubit indices to measure
    ///   - state: Normalized quantum state
    ///   - seed: Optional seed for reproducible results
    /// - Returns: Partial measurement result with multiple outcomes
    /// - Complexity: O(2^n) time, O(2^n) space
    /// - Precondition: Qubits non-empty, unique, in bounds, state normalized
    /// - SeeAlso: ``measure(_:in:seed:)-swift.type.method`` for single-qubit convenience
    @_eagerMove
    public static func measure(_ qubits: [Int], in state: QuantumState, seed: UInt64? = nil) -> PartialMeasurementResult {
        ValidationUtilities.validateNonEmpty(qubits, name: "qubits")
        ValidationUtilities.validateNormalizedState(state)
        ValidationUtilities.validateUniqueQubits(qubits)

        for qubit in qubits {
            ValidationUtilities.validateIndexInBounds(qubit, bound: state.numQubits, name: "Qubit index")
        }

        let numOutcomes = 1 << qubits.count
        var probabilities = [Double](repeating: 0.0, count: numOutcomes)

        for i in 0 ..< state.stateSpaceSize {
            let outcomeIndex = BitUtilities.getBits(i, qubits: qubits)
            probabilities[outcomeIndex] += state.probability(of: i)
        }

        let jointOutcome = sampleOutcome(probabilities: probabilities, seed: seed)

        let outcomes = [Int](unsafeUninitializedCapacity: qubits.count) { buffer, count in
            for bitPosition in 0 ..< qubits.count {
                buffer[bitPosition] = (jointOutcome >> bitPosition) & 1
            }
            count = qubits.count
        }

        let collapsedState = multiQubitCollapse(
            qubits: qubits,
            outcomes: outcomes,
            state: state,
            probability: probabilities[jointOutcome]
        )

        return PartialMeasurementResult(outcomes: outcomes, collapsedState: collapsedState)
    }

    // MARK: - Pauli Basis Measurements

    /// Measure qubit in Pauli basis (X, Y, or Z)
    ///
    /// Performs measurement in specified Pauli eigenbasis, returning eigenvalue ±1
    /// instead of computational basis outcome 0/1. This is essential for observable
    /// expectation values and quantum chemistry applications.
    ///
    /// **Implementation**:
    /// - X basis: Apply H, measure Z, map 0→+1, 1→-1, apply H†
    /// - Y basis: Apply S†H, measure Z, map 0→+1, 1→-1, apply HS
    /// - Z basis: Measure Z directly (computational basis)
    ///
    /// **Eigenvalue mapping**: outcome 0 → +1 (positive eigenstate), 1 → -1 (negative eigenstate)
    ///
    /// **Example**:
    /// ```swift
    /// let plus = QuantumState(numQubits: 1, amplitudes: [
    ///     Complex(1/sqrt(2), 0),
    ///     Complex(1/sqrt(2), 0)
    /// ])
    /// let xResult = Measurement.measure(0, basis: .x, in: plus)
    /// // xResult.eigenvalue: +1 (deterministic, |+⟩ is +1 eigenstate of X)
    ///
    /// let zero = QuantumState(numQubits: 1)
    /// let zResult = Measurement.measure(0, basis: .z, in: zero)
    /// // zResult.eigenvalue: +1 (deterministic, |0⟩ is +1 eigenstate of Z)
    /// ```
    ///
    /// - Parameters:
    ///   - qubit: Qubit index to measure (0 to n-1)
    ///   - basis: Pauli basis (.x, .y, or .z)
    ///   - state: Normalized quantum state
    ///   - seed: Optional seed for reproducible results
    /// - Returns: Pauli measurement result with eigenvalue ±1
    /// - Complexity: O(2^n) time, O(2^n) space
    /// - Precondition: Qubit in bounds, state normalized
    /// - SeeAlso: ``PauliBasis``
    @_eagerMove
    public static func measure(_ qubit: Int, basis: PauliBasis, in state: QuantumState, seed: UInt64? = nil) -> PauliMeasurementResult {
        ValidationUtilities.validateIndexInBounds(qubit, bound: state.numQubits, name: "Qubit index")
        ValidationUtilities.validateNormalizedState(state)

        let rotatedState = rotateToPauliBasis(qubit: qubit, basis: basis, state: state)
        let partialResult = measure(qubit, in: rotatedState, seed: seed)
        let eigenvalue = (partialResult.outcome == 0) ? 1 : -1
        let finalState = rotateFromPauliBasis(qubit: qubit, basis: basis, state: partialResult.collapsedState)

        return PauliMeasurementResult(eigenvalue: eigenvalue, collapsedState: finalState)
    }

    /// Measure multi-qubit Pauli string observable
    ///
    /// Measures tensor product of Pauli operators (e.g., X₀⊗Y₁⊗Z₂). Returns joint
    /// eigenvalue (±1) computed as product of individual Pauli eigenvalues. This is
    /// the fundamental operation for Hamiltonian expectation values in VQE and
    /// quantum chemistry.
    ///
    /// **Algorithm**:
    /// 1. Apply basis rotations to diagonalize each Pauli operator
    /// 2. Measure all qubits in computational basis
    /// 3. Map each outcome to eigenvalue: 0→+1, 1→-1
    /// 4. Compute product eigenvalue: λ = ∏ᵢ λᵢ
    /// 5. Rotate collapsed state back to original basis
    ///
    /// **Example**:
    /// ```swift
    /// let zz = PauliString(.z(0), .z(1))
    /// let bell = QuantumCircuit.bellPhiPlus().execute()  // (|00⟩ + |11⟩)/√2
    /// let result = Measurement.measure(zz, in: bell)
    /// // 50% |00⟩: λ₀=+1, λ₁=+1 → product=+1
    /// // 50% |11⟩: λ₀=-1, λ₁=-1 → product=+1
    /// // result.eigenvalue: +1 (deterministic!)
    /// ```
    ///
    /// - Parameters:
    ///   - pauliString: Pauli string operator (tensor product of Paulis)
    ///   - state: Normalized quantum state
    ///   - seed: Optional seed for reproducible results
    /// - Returns: Measurement result with overall eigenvalue and individual outcomes
    /// - Complexity: O(k·2^n) time where k is number of operators, O(2^n) space
    /// - Precondition: State normalized, qubits in bounds, qubits unique
    /// - SeeAlso: ``PauliString``
    @_eagerMove
    public static func measure(_ pauliString: PauliString, in state: QuantumState, seed: UInt64? = nil) -> PauliStringMeasurementResult {
        ValidationUtilities.validateNormalizedState(state)

        for op in pauliString.operators {
            ValidationUtilities.validateIndexInBounds(op.qubit, bound: state.numQubits, name: "Qubit index")
        }

        let qubits = pauliString.operators.map(\.qubit)
        ValidationUtilities.validateUniqueQubits(qubits)

        if pauliString.operators.isEmpty {
            return PauliStringMeasurementResult(
                eigenvalue: 1,
                collapsedState: state,
                individualOutcomes: []
            )
        }

        var rotatedState = state
        for op in pauliString.operators {
            rotatedState = rotateToPauliBasis(qubit: op.qubit, basis: op.basis, state: rotatedState)
        }

        let partialResult = measure(qubits, in: rotatedState, seed: seed)

        var productEigenvalue = 1
        var individualOutcomes: [MeasurementOutcome] = []

        for (index, op) in pauliString.operators.enumerated() {
            let outcome = partialResult.outcomes[index]
            let eigenvalue = (outcome == 0) ? 1 : -1
            productEigenvalue *= eigenvalue
            individualOutcomes.append(MeasurementOutcome(qubit: op.qubit, outcome: outcome))
        }

        var finalState = partialResult.collapsedState
        for op in pauliString.operators.reversed() {
            finalState = rotateFromPauliBasis(qubit: op.qubit, basis: op.basis, state: finalState)
        }

        return PauliStringMeasurementResult(
            eigenvalue: productEigenvalue,
            collapsedState: finalState,
            individualOutcomes: individualOutcomes
        )
    }

    // MARK: - Custom Basis Measurement

    /// Measure qubit in arbitrary single-qubit basis
    ///
    /// Performs measurement in custom basis defined by provided basis state |ψ⟩.
    /// The measurement projects onto {|ψ⟩, |ψ⊥⟩} where |ψ⊥⟩ is the orthogonal complement
    /// constructed via unitarity.
    ///
    /// **Algorithm**:
    /// 1. Construct unitary U rotating |ψ⟩ → |0⟩
    /// 2. Apply U to state
    /// 3. Measure in computational basis
    /// 4. Apply U† to collapsed state
    ///
    /// **Basis state format**: 2-element complex array [c₀, c₁] representing |ψ⟩ = c₀|0⟩ + c₁|1⟩
    ///
    /// **Example**:
    /// ```swift
    /// let customBasis = [
    ///     Complex(1/sqrt(2), 0),
    ///     Complex(cos(Double.pi/4)/sqrt(2), sin(Double.pi/4)/sqrt(2))
    /// ]
    /// let zero = QuantumState(numQubits: 1)
    /// let result = Measurement.measure(0, basis: customBasis, in: zero)
    /// // result.outcome: 0 or 1 (probabilities depend on |⟨ψ|0⟩|²)
    /// ```
    ///
    /// - Parameters:
    ///   - qubit: Qubit index to measure
    ///   - basisState: Normalized 2D complex vector defining measurement basis
    ///   - state: Quantum state to measure
    ///   - seed: Optional seed for reproducible results
    /// - Returns: Custom basis measurement result with outcome and collapsed state
    /// - Complexity: O(2^n) time, O(2^n) space
    /// - Precondition: Qubit in bounds, basis 2-element and normalized, state normalized
    /// - SeeAlso: ``PauliBasis`` for standard Pauli bases
    @_eagerMove
    public static func measure(
        _ qubit: Int,
        basis basisState: [Complex<Double>],
        in state: QuantumState,
        seed: UInt64? = nil
    ) -> CustomBasisMeasurementResult {
        ValidationUtilities.validateIndexInBounds(qubit, bound: state.numQubits, name: "Qubit index")
        ValidationUtilities.validateTwoComponentBasis(basisState)
        ValidationUtilities.validateNormalizedState(state)
        ValidationUtilities.validateNormalizedBasis(basisState)

        let c0 = basisState[0]
        let c1 = basisState[1]

        let rotationMatrix = [
            [c0.conjugate, c1.conjugate],
            [-c1, c0],
        ]

        let rotationGate = QuantumGate.custom(matrix: rotationMatrix)
        let rotatedState = GateApplication.apply(rotationGate, to: qubit, state: state)
        let partialResult = measure(qubit, in: rotatedState, seed: seed)

        let adjointMatrix = MatrixUtilities.hermitianConjugate(rotationMatrix)
        let inverseGate = QuantumGate.custom(matrix: adjointMatrix)
        let finalState = GateApplication.apply(inverseGate, to: qubit, state: partialResult.collapsedState)

        return CustomBasisMeasurementResult(outcome: partialResult.outcome, collapsedState: finalState)
    }

    // MARK: - Snapshots

    /// Capture non-destructive statevector snapshot
    ///
    /// Creates a snapshot of the quantum state without measurement collapse.
    /// Preserves full quantum coherence and superposition. Useful for debugging,
    /// algorithm verification, and state tomography.
    ///
    /// **Physical realizability**: Not physically realizable on quantum hardware
    /// (violates no-cloning theorem). This is a simulation-only analysis tool.
    ///
    /// **Example**:
    /// ```swift
    /// var circuit = QuantumCircuit(numQubits: 2)
    /// circuit.append(gate: .hadamard, to: 0)
    /// let snapshot = Measurement.snapshot(of: circuit.execute(), label: "After H(0)")
    /// // snapshot.state: |+⟩⊗|0⟩ (no collapse, full superposition preserved)
    /// ```
    ///
    /// - Parameters:
    ///   - state: Quantum state to capture
    ///   - label: Optional descriptive label for snapshot identification
    /// - Returns: Snapshot containing state copy, label, and timestamp
    /// - Complexity: O(2^n) space for state copy, O(1) time
    /// - SeeAlso: ``StateSnapshot``
    @_effects(readonly)
    @_eagerMove
    public static func snapshot(of state: QuantumState, label: String? = nil) -> StateSnapshot {
        StateSnapshot(
            state: state,
            label: label,
            timestamp: Date()
        )
    }

    // MARK: - Multiple Measurements (Statistics)

    /// Execute circuit multiple times and collect measurement statistics
    ///
    /// Runs circuit repeatedly, measuring final state each time to build empirical
    /// probability distribution. Essential for validating quantum algorithms and
    /// visualizing measurement outcomes.
    ///
    /// **Example**:
    /// ```swift
    /// let bellCircuit = QuantumCircuit.bellPhiPlus()
    /// let outcomes = Measurement.sample(circuit: bellCircuit, shots: 1000)
    ///
    /// let counts = Measurement.histogram(outcomes: outcomes, numQubits: 2)
    /// // counts ≈ [~500, 0, 0, ~500] for Bell state
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit to execute
    ///   - shots: Number of independent executions (≥1000 recommended for statistics)
    ///   - seed: Optional seed for reproducible results
    /// - Returns: Array of measurement outcomes (basis state indices)
    /// - Complexity: O(shots · circuit_depth · 2^n) time, O(shots) space
    /// - Precondition: shots > 0
    /// - SeeAlso: ``histogram(outcomes:numQubits:)`` for converting to counts
    @_eagerMove
    public static func sample(circuit: QuantumCircuit, shots: Int, seed: UInt64? = nil) -> [Int] {
        ValidationUtilities.validatePositiveInt(shots, name: "shots")

        var outcomes: [Int] = []
        outcomes.reserveCapacity(shots)

        var rng = createRNG(seed: seed)

        for _ in 0 ..< shots {
            let finalState = circuit.execute()
            let probabilities = finalState.probabilities()
            let outcome = sampleOutcome(probabilities: probabilities, rng: &rng)
            outcomes.append(outcome)
        }

        return outcomes
    }

    /// Convert outcomes to histogram (count per basis state)
    ///
    /// Transforms outcome array to count distribution for visualization and analysis.
    ///
    /// **Example**:
    /// ```swift
    /// let outcomes = [0, 0, 3, 0, 3, 3]
    /// let counts = Measurement.histogram(outcomes: outcomes, numQubits: 2)
    /// // counts = [3, 0, 0, 3]
    /// ```
    ///
    /// - Parameters:
    ///   - outcomes: Array of measurement outcomes
    ///   - numQubits: Number of qubits (determines state space size 2^n)
    /// - Returns: Array of counts [count(0), count(1), ..., count(2^n-1)]
    /// - Complexity: O(outcomes.count) time, O(2^n) space
    /// - SeeAlso: ``histogram(outcomes:)`` for auto-sizing variant
    @_effects(readonly)
    @inlinable
    @_eagerMove
    public static func histogram(outcomes: [Int], numQubits: Int) -> [Int] {
        let stateSpaceSize = 1 << numQubits

        var counts = [Int](unsafeUninitializedCapacity: stateSpaceSize) { buffer, count in
            buffer.initialize(repeating: 0)
            count = stateSpaceSize
        }

        for outcome in outcomes {
            if outcome >= 0, outcome < stateSpaceSize {
                counts[outcome] &+= 1
            }
        }

        return counts
    }

    /// Convert outcomes to sparse histogram (auto-sized to max outcome)
    ///
    /// Infers histogram size from maximum outcome value. Useful when state space
    /// size is unknown or when most outcomes are clustered in low indices.
    ///
    /// **Example**:
    /// ```swift
    /// let outcomes = [0, 0, 5, 0, 5]
    /// let counts = Measurement.histogram(outcomes: outcomes)
    /// // counts = [3, 0, 0, 0, 0, 2] (length 6, auto-sized to max+1)
    /// ```
    ///
    /// - Parameter outcomes: Array of measurement outcomes
    /// - Returns: Array of counts with length = max(outcomes) + 1, or empty if no outcomes
    /// - Complexity: O(outcomes.count) time, O(max(outcomes)) space
    /// - SeeAlso: ``histogram(outcomes:numQubits:)`` for explicit sizing
    @_effects(readonly)
    @inlinable
    @_eagerMove
    public static func histogram(outcomes: [Int]) -> [Int] {
        guard let maxOutcome = outcomes.max() else { return [] }

        let stateSpaceSize = maxOutcome + 1

        var counts = [Int](unsafeUninitializedCapacity: stateSpaceSize) { buffer, count in
            buffer.initialize(repeating: 0)
            count = stateSpaceSize
        }

        for outcome in outcomes {
            if outcome >= 0 {
                counts[outcome] &+= 1
            }
        }

        return counts
    }

    /// Compare observed frequencies to expected probabilities
    ///
    /// Computes maximum relative error across all outcomes for goodness-of-fit testing.
    /// Relative error = |observed - expected| / expected for each outcome.
    ///
    /// **Example**:
    /// ```swift
    /// let observed = [48, 52]
    /// let expected = [0.5, 0.5]
    /// let error = Measurement.relativeError(
    ///     observed: observed,
    ///     expected: expected,
    ///     totalShots: 100
    /// )
    /// // error ≈ 0.04 (4% max deviation)
    /// ```
    ///
    /// - Parameters:
    ///   - observed: Observed outcome counts
    ///   - expected: Expected probabilities (must sum to 1)
    ///   - totalShots: Total number of measurements
    /// - Returns: Maximum relative error across all outcomes
    /// - Complexity: O(n) time where n = observed.count, O(1) space
    /// - Precondition: observed.count == expected.count, totalShots > 0
    /// - SeeAlso: ``chiSquared(observed:expected:totalShots:)`` for statistical hypothesis testing
    public static func relativeError(
        observed: [Int],
        expected: [Double],
        totalShots: Int
    ) -> Double {
        ValidationUtilities.validateEqualCounts(observed, expected, name1: "observed", name2: "expected")
        ValidationUtilities.validatePositiveInt(totalShots, name: "totalShots")

        var maxError = 0.0

        for i in 0 ..< observed.count {
            let observedFreq = Double(observed[i]) / Double(totalShots)
            let expectedFreq = expected[i]

            if expectedFreq > 0 {
                let relativeError = abs(observedFreq - expectedFreq) / expectedFreq
                maxError = max(maxError, relativeError)
            } else if observedFreq > 0 {
                maxError = max(maxError, observedFreq)
            }
        }

        return maxError
    }

    /// Chi-squared goodness-of-fit test
    ///
    /// Tests whether observed distribution matches expected theoretical distribution.
    /// Computes χ² = Σ((observed - expected)² / expected) over bins with expected ≥ 5.
    ///
    /// **Interpretation**:
    /// - χ² ≈ 0: Perfect fit
    /// - χ² < degrees of freedom: Good fit (within statistical fluctuations)
    /// - χ² >> degrees of freedom: Poor fit (systematic deviation)
    ///
    /// **Example**:
    /// ```swift
    /// let observed = [48, 52]
    /// let expected = [0.5, 0.5]
    /// let result = Measurement.chiSquared(
    ///     observed: observed,
    ///     expected: expected,
    ///     totalShots: 100
    /// )
    /// // result.chiSquared ≈ 0.16
    /// // result.degreesOfFreedom: 1
    /// ```
    ///
    /// - Parameters:
    ///   - observed: Observed outcome counts
    ///   - expected: Expected probabilities (must sum to 1)
    ///   - totalShots: Total number of measurements
    /// - Returns: Chi-squared result with statistic, degrees of freedom, and bin counts
    /// - Complexity: O(n) time where n = observed.count, O(1) space
    /// - Precondition: observed.count == expected.count, totalShots > 0
    /// - Note: Bins with expected count < 5 are skipped (standard chi-squared practice)
    /// - SeeAlso: ``ChiSquaredResult``, ``relativeError(observed:expected:totalShots:)``
    public static func chiSquared(
        observed: [Int],
        expected: [Double],
        totalShots: Int
    ) -> ChiSquaredResult {
        ValidationUtilities.validateEqualCounts(observed, expected, name1: "observed", name2: "expected")
        ValidationUtilities.validatePositiveInt(totalShots, name: "totalShots")

        var chiSq = 0.0
        var testedBins = 0
        var skippedBins = 0

        for i in 0 ..< observed.count {
            let expectedCount = expected[i] * Double(totalShots)

            if expectedCount < 5 {
                skippedBins += 1
                continue
            }

            let diff = Double(observed[i]) - expectedCount
            chiSq += (diff * diff) / expectedCount
            testedBins += 1
        }

        let degreesOfFreedom = max(testedBins - 1, 0)

        return ChiSquaredResult(
            chiSquared: chiSq,
            degreesOfFreedom: degreesOfFreedom,
            testedBins: testedBins,
            skippedBins: skippedBins
        )
    }

    // MARK: - Internal Helpers

    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func collapseToOutcome(_ outcome: Int, numQubits: Int) -> QuantumState {
        let stateSpaceSize = 1 << numQubits
        ValidationUtilities.validateIndexInBounds(outcome, bound: stateSpaceSize, name: "Outcome")

        let amplitudes = [Complex<Double>](unsafeUninitializedCapacity: stateSpaceSize) { buffer, count in
            for i in 0 ..< stateSpaceSize {
                buffer[i] = i == outcome ? .one : .zero
            }
            count = stateSpaceSize
        }

        return QuantumState(numQubits: numQubits, amplitudes: amplitudes)
    }

    @_effects(readonly)
    @_eagerMove
    static func multiQubitCollapse(
        qubits: [Int],
        outcomes: [Int],
        state: QuantumState,
        probability: Double
    ) -> QuantumState {
        ValidationUtilities.validateEqualCounts(qubits, outcomes, name1: "qubits", name2: "outcomes")
        ValidationUtilities.validatePositiveDouble(probability, name: "probability")

        let normalizationFactor = 1.0 / sqrt(probability)

        var mask = 0
        var expected = 0
        for (qubit, outcome) in zip(qubits, outcomes) {
            mask |= (1 << qubit)
            expected |= (outcome << qubit)
        }

        let newAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: state.stateSpaceSize) { buffer, count in
            for i in 0 ..< state.stateSpaceSize {
                buffer[i] = (i & mask) == expected ? state.amplitudes[i] * normalizationFactor : .zero
            }
            count = state.stateSpaceSize
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func rotateToPauliBasis(qubit: Int, basis: PauliBasis, state: QuantumState) -> QuantumState {
        switch basis {
        case .x:
            return GateApplication.apply(.hadamard, to: qubit, state: state)
        case .y:
            var rotated = GateApplication.apply(.phase(-Double.pi / 2.0), to: qubit, state: state)
            rotated = GateApplication.apply(.hadamard, to: qubit, state: rotated)
            return rotated
        case .z:
            return state
        }
    }

    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func rotateFromPauliBasis(qubit: Int, basis: PauliBasis, state: QuantumState) -> QuantumState {
        switch basis {
        case .x:
            return GateApplication.apply(.hadamard, to: qubit, state: state)
        case .y:
            var rotated = GateApplication.apply(.hadamard, to: qubit, state: state)
            rotated = GateApplication.apply(.sGate, to: qubit, state: rotated)
            return rotated
        case .z:
            return state
        }
    }

    @usableFromInline
    static func sampleOutcome(probabilities: [Double], seed: UInt64?) -> Int {
        var rng = createRNG(seed: seed)
        return sampleOutcome(probabilities: probabilities, rng: &rng)
    }

    @usableFromInline
    static func sampleOutcome(probabilities: [Double], rng: inout any RandomNumberGenerator) -> Int {
        ValidationUtilities.validateProbabilityDistribution(probabilities)

        let random = Double.random(in: 0 ..< 1, using: &rng)

        var accumulated = 0.0
        for (index, probability) in probabilities.enumerated() {
            accumulated += probability
            if accumulated >= random {
                return index
            }
        }

        return probabilities.count - 1
    }

    @usableFromInline
    static func createRNG(seed: UInt64?) -> any RandomNumberGenerator {
        if let seed {
            let source = GKMersenneTwisterRandomSource(seed: seed)
            return RandomNumberGeneratorWrapper(source: source)
        } else {
            return SystemRandomNumberGenerator()
        }
    }

    @usableFromInline
    struct RandomNumberGeneratorWrapper: RandomNumberGenerator {
        let source: GKMersenneTwisterRandomSource

        @usableFromInline mutating func next() -> UInt64 {
            let upper = UInt64(source.nextInt(upperBound: Int.max))
            let lower = UInt64(source.nextInt(upperBound: Int.max))
            return (upper << 32) | lower
        }
    }
}
