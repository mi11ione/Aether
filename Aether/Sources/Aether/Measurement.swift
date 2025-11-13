// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation
import GameplayKit

/// Quantum measurement outcome with collapsed state
///
/// Encapsulates the result of measuring a quantum state: the classical outcome
/// and the post-measurement collapsed state. Implements state collapse postulate
/// from quantum mechanics.
///
/// Example:
/// ```swift
/// var measurement = Measurement()
/// let result = measurement.measure(state: bellState)
/// // result.outcome: 0 or 3 (50% each for Bell state)
/// // result.collapsedState: |00⟩ or |11⟩ (deterministic after collapse)
/// ```
@frozen
public struct MeasurementResult: Equatable, CustomStringConvertible {
    /// Classical outcome (basis state index i ∈ [0, 2^n-1])
    public let outcome: Int

    /// Post-measurement state |i⟩ (collapsed to measured outcome)
    public let collapsedState: QuantumState

    public var description: String {
        "Measurement: outcome=\(outcome), state=\(collapsedState)"
    }
}

/// Pauli measurement basis selection
///
/// Defines the measurement basis for Pauli measurements. Each basis corresponds
/// to measuring an observable (X, Y, or Z) with eigenvalues ±1.
///
/// **Basis definitions**:
/// - **X basis**: Eigenstates |+⟩ = (|0⟩ + |1⟩)/√2 (eigenvalue +1)
///                          |-⟩ = (|0⟩ - |1⟩)/√2 (eigenvalue -1)
/// - **Y basis**: Eigenstates |+i⟩ = (|0⟩ + i|1⟩)/√2 (eigenvalue +1)
///                          |-i⟩ = (|0⟩ - i|1⟩)/√2 (eigenvalue -1)
/// - **Z basis**: Eigenstates |0⟩ (eigenvalue +1), |1⟩ (eigenvalue -1)
///
/// Example:
/// ```swift
/// var measurement = Measurement()
/// let plus = QuantumState(numQubits: 1, amplitudes: [
///     Complex(1/sqrt(2), 0),
///     Complex(1/sqrt(2), 0)
/// ])  // |+⟩ state
///
/// let result = measurement.measurePauli(qubit: 0, basis: .x, state: plus)
/// // result.eigenvalue: +1 (deterministic, |+⟩ is X eigenstate)
///
/// let result2 = measurement.measurePauli(qubit: 0, basis: .z, state: plus)
/// // result2.eigenvalue: ±1 (50% each, superposition in Z basis)
/// ```
@frozen
public enum PauliBasis: String, CaseIterable, Sendable {
    /// X (bit-flip) basis: |+⟩, |-⟩
    case x

    /// Y (bit-flip + phase) basis: |+i⟩, |-i⟩
    case y

    /// Z (computational) basis: |0⟩, |1⟩
    case z
}

/// Pauli measurement result with eigenvalue ±1
///
/// Result of measuring a qubit in X, Y, or Z basis. Unlike computational basis
/// measurements (which return 0/1), Pauli measurements return eigenvalues ±1.
///
/// **Eigenvalue mapping**:
/// - +1: Measured qubit in positive eigenstate (|+⟩, |+i⟩, or |0⟩)
/// - -1: Measured qubit in negative eigenstate (|-⟩, |-i⟩, or |1⟩)
///
/// Example:
/// ```swift
/// var measurement = Measurement()
/// let zero = QuantumState(numQubits: 1)  // |0⟩
///
/// let xResult = measurement.measurePauli(qubit: 0, basis: .x, state: zero)
/// // xResult.eigenvalue: ±1 (50% each, |0⟩ is superposition in X basis)
///
/// let zResult = measurement.measurePauli(qubit: 0, basis: .z, state: zero)
/// // zResult.eigenvalue: +1 (deterministic, |0⟩ is Z eigenstate with λ=+1)
/// ```
@frozen
public struct PauliMeasurementResult: Equatable, CustomStringConvertible {
    /// Eigenvalue: +1 or -1
    public let eigenvalue: Int

    /// Post-measurement collapsed state
    public let collapsedState: QuantumState

    public var description: String {
        "PauliMeasurement: eigenvalue=\(eigenvalue > 0 ? "+1" : "-1"), state=\(collapsedState)"
    }
}

/// Single Pauli operator on a specific qubit
///
/// Represents one Pauli operator (X, Y, or Z) applied to a specific qubit.
/// Used as building blocks for PauliString (tensor product of operators).
///
/// Example:
/// ```swift
/// let xOn0 = PauliOperator(qubit: 0, basis: .x)  // X₀
/// let zOn2 = PauliOperator(qubit: 2, basis: .z)  // Z₂
/// ```
@frozen
public struct PauliOperator: Equatable, Hashable, Sendable {
    /// Target qubit index (0 to n-1)
    public let qubit: Int

    /// Pauli basis (X, Y, or Z)
    public let basis: PauliBasis
}

/// Pauli string operator for multi-qubit measurements
///
/// Represents a tensor product of Pauli operators, e.g., X₀⊗Y₁⊗Z₂.
/// Used for measuring multi-qubit observables in quantum chemistry and VQE.
///
/// **Mathematical representation**: O = ⊗ᵢ Pᵢ where Pᵢ ∈ {I, X, Y, Z}
/// **Eigenvalue**: Product of individual Pauli eigenvalues (±1)
///
/// **Construction**:
/// - Specify (qubit, basis) pairs for non-identity operators
/// - Identity operators on unspecified qubits (implicit)
/// - Order doesn't matter (tensor product commutes with disjoint qubits)
///
/// Example:
/// ```swift
/// // Measure X₀⊗Z₁ (X on qubit 0, Z on qubit 1)
/// let pauliString = PauliString(operators: [
///     (qubit: 0, basis: .x),
///     (qubit: 1, basis: .z)
/// ])
///
/// var measurement = Measurement()
/// let bell = QuantumCircuit.bellPhiPlus().execute()  // (|00⟩ + |11⟩)/√2
/// let result = measurement.measurePauliString(pauliString, state: bell)
/// // result.eigenvalue: ±1 (eigenvalue of X⊗Z observable)
///
/// // Single-qubit case: just Z₂
/// let singlePauli = PauliString(operators: [(qubit: 2, basis: .z)])
/// ```
@frozen
public struct PauliString: Equatable, Hashable, CustomStringConvertible, Sendable {
    /// Array of Pauli operators
    /// Identity operators on unspecified qubits
    public let operators: [PauliOperator]

    /// Create Pauli string from operator list
    /// - Parameter operators: List of (qubit, basis) pairs
    public init(operators: [(qubit: Int, basis: PauliBasis)]) {
        self.operators = operators.map { PauliOperator(qubit: $0.qubit, basis: $0.basis) }
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
/// Records the measurement result (0 or 1) for a specific qubit.
/// Used in multi-qubit measurement results to track individual qubit outcomes.
///
/// Example:
/// ```swift
/// let outcome = MeasurementOutcome(qubit: 0, outcome: 1)  // Qubit 0 measured as |1⟩
/// ```
@frozen
public struct MeasurementOutcome: Equatable {
    /// Qubit index that was measured
    public let qubit: Int

    /// Measurement result: 0 or 1 (computational basis)
    public let outcome: Int
}

/// Multi-qubit Pauli measurement result
///
/// Result of measuring a Pauli string observable (tensor product of Paulis).
/// Contains overall eigenvalue (±1) and individual measurement outcomes.
///
/// **Eigenvalue computation**: λ = ∏ᵢ λᵢ where λᵢ ∈ {±1} is individual outcome
///
/// Example:
/// ```swift
/// // Measure Z₀⊗Z₁ on Bell state
/// let pauliString = PauliString(operators: [
///     (qubit: 0, basis: .z),
///     (qubit: 1, basis: .z)
/// ])
///
/// var measurement = Measurement()
/// let bell = QuantumCircuit.bellPhiPlus().execute()
/// let result = measurement.measurePauliString(pauliString, state: bell)
///
/// // Bell state: 50% |00⟩ (λ₀=+1, λ₁=+1 → product=+1)
/// //             50% |11⟩ (λ₀=-1, λ₁=-1 → product=+1)
/// // result.eigenvalue: +1 (deterministic!)
/// // result.individualOutcomes: [(0, 0), (1, 0)] or [(0, 1), (1, 1)]
/// ```
@frozen
public struct PauliStringMeasurementResult: Equatable, CustomStringConvertible {
    /// Overall eigenvalue: product of individual eigenvalues (±1)
    public let eigenvalue: Int

    /// Post-measurement collapsed state
    public let collapsedState: QuantumState

    /// Individual measurement outcomes for each qubit
    public let individualOutcomes: [MeasurementOutcome]

    public var description: String {
        let outcomeStr = individualOutcomes.map { "q\($0.qubit)=\($0.outcome)" }.joined(separator: ", ")
        return "PauliStringMeasurement: eigenvalue=\(eigenvalue > 0 ? "+1" : "-1"), outcomes=[\(outcomeStr)]"
    }
}

/// Statevector snapshot for non-destructive state capture
///
/// Captures the full quantum state at a specific point without measurement collapse.
/// Useful for debugging, algorithm verification, and state tomography. Unlike measurements,
/// snapshots preserve quantum coherence and superposition.
///
/// **Use cases**:
/// - **Debugging**: Verify circuit produces expected state
/// - **Algorithm verification**: Compare intermediate states to theory
/// - **State tomography**: Reconstruct density matrix from snapshots
/// - **Visualization**: Display wavefunction evolution
///
/// Example:
/// ```swift
/// var circuit = QuantumCircuit(numQubits: 2)
/// circuit.append(gate: .hadamard, toQubit: 0)
///
/// let snapshot1 = Measurement.captureSnapshot(
///     state: circuit.execute(),
///     label: "After H(0)"
/// )
/// // snapshot1.state: |+⟩⊗|0⟩ (no collapse)
///
/// circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
///
/// let snapshot2 = Measurement.captureSnapshot(
///     state: circuit.execute(),
///     label: "After CNOT"
/// )
/// // snapshot2.state: Bell state (|00⟩ + |11⟩)/√2
///
/// // Verify expectation values without destroying state
/// let zExpectation = snapshot2.state.probability(ofState: 0)
///     + snapshot2.state.probability(ofState: 3)
/// // zExpectation ≈ 1.0 for Bell state
/// ```
@frozen
public struct StateSnapshot: Equatable, CustomStringConvertible, Sendable {
    /// Captured quantum state (full statevector)
    public let state: QuantumState

    /// Optional descriptive label
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

/// Quantum measurement: Born rule implementation with state collapse
///
/// Implements projective measurement in computational and arbitrary bases following
/// quantum mechanics. Supports full measurements, partial measurements, Pauli observables,
/// and non-destructive state snapshots.
///
/// **Measurement types**:
/// - **Computational basis**: Standard Z-basis measurement (|0⟩, |1⟩)
/// - **Pauli basis**: X, Y, or Z measurements with ±1 eigenvalues
/// - **Custom basis**: Arbitrary single-qubit unitary basis
/// - **Multi-qubit Pauli**: Tensor product observables (X₀⊗Y₁⊗Z₂)
/// - **Partial measurement**: Measure subset of qubits, preserve rest
/// - **Snapshots**: Non-destructive state capture (no collapse)
///
/// **Born rule**: Probability P(i) = |cᵢ|² for measuring basis state |i⟩
/// **State collapse**: Measurement projects state onto measured outcome
/// **Reproducibility**: Optional seeded RNG for deterministic results
///
/// Example:
/// ```swift
/// // Create Bell state (|00⟩ + |11⟩)/√2
/// let circuit = QuantumCircuit.bellPhiPlus()
/// let state = circuit.execute()
///
/// // Computational basis measurement
/// var measurement = Measurement()
/// let result = measurement.measure(state: state)
/// // result.outcome: 0 (|00⟩) or 3 (|11⟩) with 50% probability each
///
/// // Pauli X measurement
/// let xResult = measurement.measurePauli(qubit: 0, basis: .x, state: state)
/// // xResult.eigenvalue: ±1
///
/// // Multi-qubit Pauli measurement (Z₀⊗Z₁)
/// let pauliString = PauliString(operators: [
///     (qubit: 0, basis: .z),
///     (qubit: 1, basis: .z)
/// ])
/// let zzResult = measurement.measurePauliString(pauliString, state: state)
/// // zzResult.eigenvalue: +1 (deterministic for Bell state)
///
/// // Non-destructive snapshot
/// let snapshot = Measurement.captureSnapshot(state: state, label: "Bell state")
/// // snapshot.state preserves superposition (no collapse)
/// ```
public struct Measurement {
    /// Random number generator for measurement sampling
    private var rng: any RandomNumberGenerator

    /// Whether it uses a seeded RNG (for reproducibility)
    public let isSeeded: Bool

    /// Initialize with system random number generator (non-reproducible)
    public init() {
        rng = SystemRandomNumberGenerator()
        isSeeded = false
    }

    /// Initialize with seeded random number generator (reproducible)
    public init(seed: UInt64) {
        let source = GKMersenneTwisterRandomSource(seed: seed)
        rng = RandomNumberGeneratorWrapper(source: source)
        isSeeded = true
    }

    /// Wrapper for GKMersenneTwisterRandomSource to conform to RandomNumberGenerator
    private struct RandomNumberGeneratorWrapper: RandomNumberGenerator {
        let source: GKMersenneTwisterRandomSource

        mutating func next() -> UInt64 {
            let upper = UInt64(source.nextInt(upperBound: Int.max))
            let lower = UInt64(source.nextInt(upperBound: Int.max))
            return (upper << 32) | lower
        }
    }

    // MARK: - Probability Distribution

    /// Calculate probability distribution for all basis states.
    /// - Parameter state: Quantum state to measure
    /// - Returns: Array of probabilities [P(0), P(1), ..., P(2^n-1)]
    @_effects(readonly)
    @inlinable
    @_eagerMove
    public static func probabilityDistribution(state: QuantumState) -> [Double] {
        ValidationUtilities.validateNormalizedState(state)
        return state.probabilities()
    }

    // MARK: - Weighted Random Sampling

    /// Sample outcome from probability distribution using roulette wheel algorithm.
    /// - Parameter probabilities: Probability distribution (must sum to 1.0)
    /// - Returns: Sampled outcome index
    private mutating func sampleOutcome(probabilities: [Double]) -> Int {
        precondition(!probabilities.isEmpty, "Probability array must not be empty")

        let sum = probabilities.reduce(0.0, +)
        precondition(abs(sum - 1.0) < 1e-10, "Probabilities must sum to 1.0 (got \(sum))")

        let random = Double.random(in: 0 ..< 1, using: &rng)

        // Roulette wheel: accumulate probabilities until exceed random
        var accumulated = 0.0
        for (index, probability) in probabilities.enumerated() {
            accumulated += probability
            if accumulated >= random {
                return index
            }
        }

        // Fallback for numerical precision (should rarely happen)
        // Return last non-zero probability index
        return probabilities.count - 1
    }

    // MARK: - Computational Basis Measurement

    /// Measure all qubits in computational basis (Born rule)
    ///
    /// Performs projective measurement of entire quantum state, sampling outcome
    /// according to P(i) = |cᵢ|². Collapses state to measured basis state |i⟩.
    /// This is the standard measurement operation in quantum computing.
    ///
    /// **Process**:
    /// 1. Calculate probability distribution P(i) = |cᵢ|² for all i
    /// 2. Sample outcome according to probabilities (roulette wheel)
    /// 3. Collapse state to |outcome⟩
    ///
    /// - Parameter state: Normalized quantum state to measure
    /// - Returns: Measurement result containing outcome and collapsed state
    ///
    /// Example:
    /// ```swift
    /// // Measure superposition |+⟩ = (|0⟩ + |1⟩)/√2
    /// let plus = QuantumState(numQubits: 1, amplitudes: [
    ///     Complex(1/sqrt(2), 0),
    ///     Complex(1/sqrt(2), 0)
    /// ])
    /// var measurement = Measurement()
    /// let result = measurement.measure(state: plus)
    /// // result.outcome: 0 or 1 (50% each)
    /// // result.collapsedState: |0⟩ or |1⟩
    ///
    /// // Measure Bell state (|00⟩ + |11⟩)/√2
    /// let bell = QuantumCircuit.bellPhiPlus().execute()
    /// let bellResult = measurement.measure(state: bell)
    /// // bellResult.outcome: 0 (|00⟩) or 3 (|11⟩) with equal probability
    ///
    /// // Seeded measurement for reproducibility
    /// var seeded = Measurement(seed: 123)
    /// let r1 = seeded.measure(state: plus)
    /// var seeded2 = Measurement(seed: 123)
    /// let r2 = seeded2.measure(state: plus)
    /// // r1.outcome == r2.outcome (same seed → same result)
    /// ```
    @_eagerMove
    public mutating func measure(state: QuantumState) -> MeasurementResult {
        ValidationUtilities.validateNormalizedState(state)

        // Calculate probability distribution
        let probabilities = state.probabilities()

        // Sample outcome according to Born rule
        let outcome = sampleOutcome(probabilities: probabilities)

        // Collapse state to measured outcome
        let collapsedState = Self.collapseToOutcome(outcome, numQubits: state.numQubits)

        return MeasurementResult(outcome: outcome, collapsedState: collapsedState)
    }

    /// Create collapsed state for given outcome.
    /// All amplitudes are 0 except the measured outcome which is 1.0.
    ///
    /// - Parameters:
    ///   - outcome: Measured basis state index
    ///   - numQubits: Number of qubits
    /// - Returns: Collapsed quantum state
    @_effects(readonly)
    @inlinable
    @_eagerMove
    public static func collapseToOutcome(_ outcome: Int, numQubits: Int) -> QuantumState {
        let stateSpaceSize = 1 << numQubits
        precondition(outcome >= 0 && outcome < stateSpaceSize, "Outcome out of bounds")

        var amplitudes = AmplitudeVector(repeating: .zero, count: stateSpaceSize)
        amplitudes[outcome] = Complex(1.0, 0.0)

        return QuantumState(numQubits: numQubits, amplitudes: amplitudes)
    }

    // MARK: - Pauli Basis Measurements

    /// Measure qubit in Pauli basis (X, Y, or Z)
    ///
    /// Performs measurement in specified Pauli eigenbasis, returning eigenvalue ±1
    /// instead of computational basis outcome 0/1. This is essential for observable
    /// expectation values and quantum chemistry applications.
    ///
    /// **Implementation**:
    /// - **X basis**: Apply H gate, measure Z, map 0→+1, 1→-1
    /// - **Y basis**: Apply S†H gates, measure Z, map 0→+1, 1→-1
    /// - **Z basis**: Measure Z directly (computational basis)
    ///
    /// **Eigenvalue mapping**:
    /// - Outcome 0 → eigenvalue +1 (positive eigenstate)
    /// - Outcome 1 → eigenvalue -1 (negative eigenstate)
    ///
    /// - Parameters:
    ///   - qubit: Qubit index to measure (0 to n-1)
    ///   - basis: Pauli basis (.x, .y, or .z)
    ///   - state: Normalized quantum state
    /// - Returns: Pauli measurement result with eigenvalue ±1
    ///
    /// Example:
    /// ```swift
    /// var measurement = Measurement()
    ///
    /// // Measure |+⟩ in X basis (eigenstate)
    /// let plus = QuantumState(numQubits: 1, amplitudes: [
    ///     Complex(1/sqrt(2), 0),
    ///     Complex(1/sqrt(2), 0)
    /// ])
    /// let xResult = measurement.measurePauli(qubit: 0, basis: .x, state: plus)
    /// // xResult.eigenvalue: +1 (deterministic, |+⟩ is +1 eigenstate of X)
    ///
    /// // Measure |0⟩ in X basis (superposition)
    /// let zero = QuantumState(numQubits: 1)
    /// let xResult2 = measurement.measurePauli(qubit: 0, basis: .x, state: zero)
    /// // xResult2.eigenvalue: ±1 (50% each)
    ///
    /// // Measure |0⟩ in Z basis (eigenstate)
    /// let zResult = measurement.measurePauli(qubit: 0, basis: .z, state: zero)
    /// // zResult.eigenvalue: +1 (deterministic, |0⟩ is +1 eigenstate of Z)
    /// ```
    @_eagerMove
    public mutating func measurePauli(qubit: Int, basis: PauliBasis, state: QuantumState) -> PauliMeasurementResult {
        ValidationUtilities.validateQubitIndex(qubit, numQubits: state.numQubits)
        ValidationUtilities.validateNormalizedState(state)

        // Apply basis rotation to diagonalize the Pauli operator
        let rotatedState = Self.rotateToPauliBasis(qubit: qubit, basis: basis, state: state)

        // Measure in computational basis
        let (outcome, collapsedState) = measureQubit(qubit, state: rotatedState)

        // Map outcome to eigenvalue: 0 → +1, 1 → -1
        let eigenvalue = (outcome == 0) ? 1 : -1

        // Rotate collapsed state back to original basis
        let finalState = Self.rotateFromPauliBasis(qubit: qubit, basis: basis, state: collapsedState)

        return PauliMeasurementResult(eigenvalue: eigenvalue, collapsedState: finalState)
    }

    /// Rotate state to diagonalize Pauli operator
    ///
    /// Applies unitary transformation that rotates Pauli eigenstates to computational basis:
    /// - X basis: H gate (|+⟩ → |0⟩, |-⟩ → |1⟩)
    /// - Y basis: HS† gates (|+i⟩ → |0⟩, |-i⟩ → |1⟩)
    /// - Z basis: Identity (already diagonal)
    ///
    /// - Parameters:
    ///   - qubit: Target qubit index
    ///   - basis: Pauli basis to rotate to
    ///   - state: Input quantum state
    /// - Returns: Rotated state where Pauli operator is diagonal
    @_effects(readonly)
    @inlinable
    @_eagerMove
    public static func rotateToPauliBasis(qubit: Int, basis: PauliBasis, state: QuantumState) -> QuantumState {
        switch basis {
        case .x:
            // X basis: Apply H to rotate |+⟩→|0⟩, |-⟩→|1⟩
            return GateApplication.apply(gate: .hadamard, to: [qubit], state: state)

        case .y:
            // Y basis: Apply S†H to rotate |+i⟩→|0⟩, |-i⟩→|1⟩
            // S† = phase(-π/2)
            var rotated = GateApplication.apply(gate: .phase(theta: -Double.pi / 2.0), to: [qubit], state: state)
            rotated = GateApplication.apply(gate: .hadamard, to: [qubit], state: rotated)
            return rotated

        case .z:
            // Z basis: Already diagonal (computational basis)
            return state
        }
    }

    /// Rotate state back from Pauli basis to original basis
    ///
    /// Applies inverse of rotation used to diagonalize Pauli operator:
    /// - X basis: H† = H (Hadamard is self-inverse)
    /// - Y basis: H†S = HS (inverse of S†H)
    /// - Z basis: Identity
    ///
    /// - Parameters:
    ///   - qubit: Target qubit index
    ///   - basis: Pauli basis to rotate from
    ///   - state: Collapsed state in rotated basis
    /// - Returns: State rotated back to original basis
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func rotateFromPauliBasis(qubit: Int, basis: PauliBasis, state: QuantumState) -> QuantumState {
        switch basis {
        case .x:
            // Inverse of H is H (self-inverse)
            return GateApplication.apply(gate: .hadamard, to: [qubit], state: state)

        case .y:
            // Inverse of S†H is HS
            var rotated = GateApplication.apply(gate: .hadamard, to: [qubit], state: state)
            rotated = GateApplication.apply(gate: .sGate, to: [qubit], state: rotated)
            return rotated

        case .z:
            // Z basis: No rotation needed
            return state
        }
    }

    // MARK: - Custom Basis Measurement

    /// Measure qubit in arbitrary single-qubit basis
    ///
    /// Performs measurement in custom basis defined by provided basis state |ψ⟩.
    /// The measurement projects onto {|ψ⟩, |ψ⊥⟩} where |ψ⊥⟩ is orthogonal complement.
    /// Generalizes Pauli measurements to any single-qubit unitary basis.
    ///
    /// **Algorithm**:
    /// 1. Construct unitary U that rotates |ψ⟩ → |0⟩
    /// 2. Apply U to state
    /// 3. Measure in computational basis
    /// 4. Apply U† to collapsed state
    ///
    /// **Basis state format**: 2-element complex array [c₀, c₁] representing |ψ⟩ = c₀|0⟩ + c₁|1⟩
    ///
    /// - Parameters:
    ///   - qubit: Qubit index to measure
    ///   - basisState: Normalized 2D complex vector defining measurement basis
    ///   - state: Quantum state to measure
    /// - Returns: Tuple (outcome ∈ {0,1}, collapsed state)
    ///
    /// Example:
    /// ```swift
    /// var measurement = Measurement()
    ///
    /// // Measure in |ψ⟩ = (|0⟩ + e^(iπ/4)|1⟩)/√2 basis
    /// let customBasis = [
    ///     Complex(1/sqrt(2), 0),
    ///     Complex(cos(Double.pi/4)/sqrt(2), sin(Double.pi/4)/sqrt(2))
    /// ]
    ///
    /// let zero = QuantumState(numQubits: 1)
    /// let result = measurement.measureCustomBasis(
    ///     qubit: 0,
    ///     basisState: customBasis,
    ///     state: zero
    /// )
    /// // result.outcome: 0 or 1 (probabilities depend on overlap)
    ///
    /// // Verify: measuring |+⟩ in custom basis
    /// let plus = QuantumState(numQubits: 1, amplitudes: [
    ///     Complex(1/sqrt(2), 0),
    ///     Complex(1/sqrt(2), 0)
    /// ])
    /// let result2 = measurement.measureCustomBasis(
    ///     qubit: 0,
    ///     basisState: customBasis,
    ///     state: plus
    /// )
    /// ```
    @_eagerMove
    public mutating func measureCustomBasis(
        qubit: Int,
        basisState: AmplitudeVector,
        state: QuantumState
    ) -> (outcome: Int, collapsedState: QuantumState) {
        ValidationUtilities.validateQubitIndex(qubit, numQubits: state.numQubits)
        precondition(basisState.count == 2, "Basis state must have 2 components")
        ValidationUtilities.validateNormalizedState(state)

        // Validate basis state normalization
        // Use 1e-10 tolerance to account for accumulated floating-point errors
        // in user-provided basis states (e.g., cos(θ)/sqrt(2) chains)
        let norm = sqrt(basisState[0].magnitudeSquared + basisState[1].magnitudeSquared)
        precondition(abs(norm - 1.0) < 1e-10, "Basis state must be normalized")

        // Construct unitary that rotates basisState → |0⟩
        // U = [[c₀*, c₁*], [-c₁, c₀]] where basisState = [c₀, c₁]
        let c0 = basisState[0]
        let c1 = basisState[1]

        let rotationMatrix = [
            [c0.conjugate(), c1.conjugate()],
            [-c1, c0],
        ]

        let rotationGate = try! QuantumGate.createCustomSingleQubit(matrix: rotationMatrix)

        let rotatedState = GateApplication.apply(gate: rotationGate, to: [qubit], state: state)

        let (outcome, collapsedRotated) = measureQubit(qubit, state: rotatedState)

        let adjointMatrix = MatrixUtilities.hermitianConjugate(rotationMatrix)

        let inverseGate = try! QuantumGate.createCustomSingleQubit(matrix: adjointMatrix)

        let finalState = GateApplication.apply(gate: inverseGate, to: [qubit], state: collapsedRotated)

        return (outcome: outcome, collapsedState: finalState)
    }

    // MARK: - Multi-Qubit Pauli Measurement

    /// Measure multi-qubit Pauli string observable
    ///
    /// Measures tensor product of Pauli operators, e.g., X₀⊗Y₁⊗Z₂. Returns joint
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
    /// **Mathematical representation**: ⟨ψ|P₀⊗P₁⊗...⊗Pₙ|ψ⟩ where Pᵢ ∈ {X, Y, Z}
    ///
    /// - Parameters:
    ///   - pauliString: Pauli string operator (tensor product)
    ///   - state: Normalized quantum state
    /// - Returns: Measurement result with overall eigenvalue and individual outcomes
    ///
    /// Example:
    /// ```swift
    /// var measurement = Measurement()
    ///
    /// // Measure Z₀⊗Z₁ on Bell state (|00⟩ + |11⟩)/√2
    /// let pauliString = PauliString(operators: [
    ///     (qubit: 0, basis: .z),
    ///     (qubit: 1, basis: .z)
    /// ])
    ///
    /// let bell = QuantumCircuit.bellPhiPlus().execute()
    /// let result = measurement.measurePauliString(pauliString, state: bell)
    /// // Outcomes: 50% |00⟩ (λ₀=+1, λ₁=+1 → product=+1)
    /// //           50% |11⟩ (λ₀=-1, λ₁=-1 → product=+1)
    /// // result.eigenvalue: +1 (deterministic!)
    ///
    /// // Measure X₀⊗X₁ on Bell state
    /// let xxString = PauliString(operators: [
    ///     (qubit: 0, basis: .x),
    ///     (qubit: 1, basis: .x)
    /// ])
    /// let xxResult = measurement.measurePauliString(xxString, state: bell)
    /// // xxResult.eigenvalue: +1 (Bell state is +1 eigenstate of X⊗X)
    ///
    /// // Single-qubit case: just Z₀
    /// let singlePauli = PauliString(operators: [(qubit: 0, basis: .z)])
    /// let zero = QuantumState(numQubits: 1)
    /// let singleResult = measurement.measurePauliString(singlePauli, state: zero)
    /// // singleResult.eigenvalue: +1 (|0⟩ is +1 eigenstate of Z)
    /// ```
    @_eagerMove
    public mutating func measurePauliString(
        _ pauliString: PauliString,
        state: QuantumState
    ) -> PauliStringMeasurementResult {
        ValidationUtilities.validateNormalizedState(state)

        // Validate all qubit indices
        for op in pauliString.operators {
            ValidationUtilities.validateQubitIndex(op.qubit, numQubits: state.numQubits)
        }

        // Check for duplicate qubits (measuring same qubit in multiple bases is undefined)
        let qubits = pauliString.operators.map(\.qubit)
        let uniqueQubits = Set(qubits)
        precondition(
            uniqueQubits.count == qubits.count,
            "Pauli string contains duplicate qubit indices: cannot measure same qubit in multiple bases simultaneously"
        )

        // Handle identity operator (empty Pauli string)
        if pauliString.operators.isEmpty {
            return PauliStringMeasurementResult(
                eigenvalue: 1,
                collapsedState: state,
                individualOutcomes: []
            )
        }

        // Apply basis rotations to diagonalize all Pauli operators
        var rotatedState = state
        for op in pauliString.operators {
            rotatedState = Self.rotateToPauliBasis(qubit: op.qubit, basis: op.basis, state: rotatedState)
        }

        // Measure all qubits involved in the Pauli string
        let (outcomes, collapsedRotated) = measureQubits(qubits, state: rotatedState)

        // Compute individual eigenvalues and product
        var productEigenvalue = 1
        var individualOutcomes: [MeasurementOutcome] = []

        for (index, op) in pauliString.operators.enumerated() {
            let outcome = outcomes[index]
            let eigenvalue = (outcome == 0) ? 1 : -1
            productEigenvalue *= eigenvalue
            individualOutcomes.append(MeasurementOutcome(qubit: op.qubit, outcome: outcome))
        }

        // Rotate collapsed state back to original basis (reverse order)
        var finalState = collapsedRotated
        for op in pauliString.operators.reversed() {
            finalState = Self.rotateFromPauliBasis(qubit: op.qubit, basis: op.basis, state: finalState)
        }

        return PauliStringMeasurementResult(
            eigenvalue: productEigenvalue,
            collapsedState: finalState,
            individualOutcomes: individualOutcomes
        )
    }

    // MARK: - Partial Measurement (Multiple Qubits)

    /// Measure subset of qubits, leaving others in superposition
    ///
    /// Performs projective measurement of multiple qubits while preserving quantum
    /// coherence in unmeasured qubits. Generalizes single-qubit partial measurement
    /// to arbitrary qubit subsets. Essential for mid-circuit measurement, error
    /// correction, and conditional quantum operations.
    ///
    /// **Process**:
    /// 1. Calculate joint probability distribution for specified qubits
    /// 2. Sample outcome (multi-qubit bitstring)
    /// 3. Collapse: zero incompatible amplitudes, renormalize rest
    ///
    /// **Applications**:
    /// - Mid-circuit measurement with classical feedback
    /// - Quantum error correction syndrome extraction
    /// - Conditional teleportation and dense coding
    ///
    /// - Parameters:
    ///   - qubits: Array of qubit indices to measure
    ///   - state: Normalized quantum state
    /// - Returns: Tuple (array of outcomes, partially collapsed state)
    ///
    /// Example:
    /// ```swift
    /// var measurement = Measurement()
    ///
    /// // Create GHZ state (|000⟩ + |111⟩)/√2
    /// let ghz = QuantumCircuit.ghzState(numQubits: 3).execute()
    ///
    /// // Measure qubits 0 and 1, leave qubit 2 unmeasured
    /// let (outcomes, collapsed) = measurement.measureQubits([0, 1], state: ghz)
    /// // outcomes: [0, 0] or [1, 1] (50% each)
    /// // collapsed: |000⟩ or |111⟩ (qubit 2 collapses due to entanglement)
    ///
    /// // Product state example: |+⟩⊗|0⟩⊗|1⟩
    /// let product = QuantumState(numQubits: 3, amplitudes: [
    ///     Complex(0, 0),           // |000⟩
    ///     Complex(0, 0),           // |001⟩
    ///     Complex(0, 0),           // |010⟩
    ///     Complex(0, 0),           // |011⟩
    ///     Complex(1/sqrt(2), 0),   // |100⟩
    ///     Complex(1/sqrt(2), 0),   // |101⟩
    ///     Complex(0, 0),           // |110⟩
    ///     Complex(0, 0)            // |111⟩
    /// ])
    ///
    /// // Measure qubits 1 and 2
    /// let (outcomes2, collapsed2) = measurement.measureQubits([1, 2], state: product)
    /// // outcomes2: [0, 1] (deterministic)
    /// // collapsed2: |100⟩ or |101⟩ (qubit 0 still in superposition)
    ///
    /// // Single-qubit measurement (equivalent to measureQubit)
    /// let (outcomes3, collapsed3) = measurement.measureQubits([0], state: ghz)
    /// // outcomes3: [0] or [1]
    /// ```
    @_eagerMove
    public mutating func measureQubits(
        _ qubits: [Int],
        state: QuantumState
    ) -> (outcomes: [Int], collapsedState: QuantumState) {
        precondition(!qubits.isEmpty, "Must specify at least one qubit to measure")
        ValidationUtilities.validateNormalizedState(state)

        // Validate qubit indices are unique and in bounds
        let uniqueQubits = Set(qubits)
        precondition(uniqueQubits.count == qubits.count, "Qubit indices must be unique")
        for qubit in qubits {
            ValidationUtilities.validateQubitIndex(qubit, numQubits: state.numQubits)
        }

        // Calculate joint probability distribution for specified qubits
        let numOutcomes = 1 << qubits.count
        var probabilities = [Double](repeating: 0.0, count: numOutcomes)

        for i in 0 ..< state.stateSpaceSize {
            // Extract measurement outcome for these qubits
            let outcomeIndex = BitUtilities.getBits(i, qubits: qubits)

            probabilities[outcomeIndex] += state.probability(ofState: i)
        }

        // Sample joint outcome
        let jointOutcome = sampleOutcome(probabilities: probabilities)

        // Convert joint outcome to individual outcomes
        var outcomes: [Int] = []
        for bitPosition in 0 ..< qubits.count {
            let bit = (jointOutcome >> bitPosition) & 1
            outcomes.append(bit)
        }

        // Collapse state: keep amplitudes compatible with all measured outcomes
        let collapsedState = Self.multiQubitCollapse(
            qubits: qubits,
            outcomes: outcomes,
            state: state,
            probability: probabilities[jointOutcome]
        )

        return (outcomes: outcomes, collapsedState: collapsedState)
    }

    /// Collapse state after measuring multiple qubits
    ///
    /// Projects state onto subspace consistent with all measured outcomes.
    /// Zeroes incompatible amplitudes and renormalizes remaining ones.
    ///
    /// - Parameters:
    ///   - qubits: Measured qubit indices
    ///   - outcomes: Measured values for each qubit
    ///   - state: Original quantum state
    ///   - probability: Joint probability of this outcome (for renormalization)
    /// - Returns: Collapsed state with unmeasured qubits still in superposition
    @_effects(readonly)
    @_eagerMove
    private static func multiQubitCollapse(
        qubits: [Int],
        outcomes: [Int],
        state: QuantumState,
        probability: Double
    ) -> QuantumState {
        precondition(qubits.count == outcomes.count, "Number of qubits and outcomes must match")
        precondition(probability > 0, "Probability must be positive")

        // Renormalization factor: 1/√P(outcome)
        let normalizationFactor = 1.0 / sqrt(probability)

        let newAmplitudes = AmplitudeVector(unsafeUninitializedCapacity: state.stateSpaceSize) { buffer, count in
            for i in 0 ..< state.stateSpaceSize {
                var compatible = true
                for (qubit, outcome) in zip(qubits, outcomes) {
                    if state.getBit(index: i, qubit: qubit) != outcome {
                        compatible = false
                        break
                    }
                }

                buffer[i] = compatible ? state.amplitudes[i] * normalizationFactor : .zero
            }
            count = state.stateSpaceSize
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    /// Measure single qubit, leaving others in superposition
    ///
    /// Performs projective measurement of one qubit while preserving quantum coherence
    /// in unmeasured qubits. Implements partial trace / marginalization. The collapsed
    /// state maintains entanglement structure for unmeasured qubits.
    ///
    /// **Process**:
    /// 1. Calculate marginal probabilities P(qubit=0) and P(qubit=1)
    /// 2. Sample outcome (0 or 1)
    /// 3. Collapse: zero incompatible amplitudes, renormalize rest
    ///
    /// **Applications**: Sequential measurements, mid-circuit measurement, error correction
    ///
    /// - Parameters:
    ///   - qubit: Qubit index to measure (0 to n-1)
    ///   - state: Normalized quantum state
    /// - Returns: Tuple (outcome ∈ {0,1}, partially collapsed state)
    ///
    /// Example:
    /// ```swift
    /// // Create Bell state (|00⟩ + |11⟩)/√2
    /// let bell = QuantumCircuit.bellPhiPlus().execute()
    ///
    /// // Measure qubit 0
    /// var measurement = Measurement()
    /// let (outcome, collapsed) = measurement.measureQubit(0, state: bell)
    ///
    /// // If outcome = 0: collapsed = |00⟩ (qubit 1 also collapsed to |0⟩)
    /// // If outcome = 1: collapsed = |11⟩ (qubit 1 also collapsed to |1⟩)
    /// // Each with 50% probability
    ///
    /// // Bell state exhibits perfect correlation
    /// let p00 = collapsed.probability(ofState: 0b00)  // 1.0 or 0.0
    /// let p11 = collapsed.probability(ofState: 0b11)  // 0.0 or 1.0
    ///
    /// // Product state example: |+⟩⊗|0⟩
    /// let product = QuantumState(numQubits: 2, amplitudes: [
    ///     Complex(1/sqrt(2), 0),  // |00⟩
    ///     Complex(0, 0),          // |01⟩
    ///     Complex(1/sqrt(2), 0),  // |10⟩
    ///     Complex(0, 0)           // |11⟩
    /// ])
    /// let (outcome2, collapsed2) = measurement.measureQubit(0, state: product)
    /// // outcome2: 0 or 1 (50% each)
    /// // collapsed2: |00⟩ or |10⟩ (qubit 1 unaffected, still |0⟩)
    /// ```
    @_eagerMove
    public mutating func measureQubit(_ qubit: Int, state: QuantumState) -> (outcome: Int, collapsedState: QuantumState) {
        ValidationUtilities.validateQubitIndex(qubit, numQubits: state.numQubits)
        ValidationUtilities.validateNormalizedState(state)

        // Calculate marginal probabilities for this qubit
        // P(qubit=0) = sum of |amplitude[i]|² where bit(i, qubit) = 0
        // P(qubit=1) = sum of |amplitude[i]|² where bit(i, qubit) = 1
        let (prob0, prob1) = Self.marginalProbabilities(qubit: qubit, state: state)

        let outcome = sampleOutcome(probabilities: [prob0, prob1])

        // Collapse state: keep amplitudes compatible with outcome, zero others
        let collapsedState = Self.partialCollapse(
            qubit: qubit,
            outcome: outcome,
            state: state,
            probability: outcome == 0 ? prob0 : prob1
        )

        return (outcome: outcome, collapsedState: collapsedState)
    }

    /// Calculate marginal probabilities for measuring a single qubit.
    /// - Parameters:
    ///   - qubit: Qubit to measure
    ///   - state: Quantum state
    /// - Returns: (P(qubit=0), P(qubit=1))
    @_effects(readonly)
    @inlinable
    public static func marginalProbabilities(qubit: Int, state: QuantumState) -> (Double, Double) {
        var prob0 = 0.0
        var prob1 = 0.0

        for i in 0 ..< state.stateSpaceSize {
            let probability = state.probability(ofState: i)

            if state.getBit(index: i, qubit: qubit) == 0 {
                prob0 += probability
            } else {
                prob1 += probability
            }
        }

        return (prob0, prob1)
    }

    /// Partially collapse state after measuring one qubit.
    /// - Parameters:
    ///   - qubit: Measured qubit index
    ///   - outcome: Measured value (0 or 1)
    ///   - state: Original quantum state
    ///   - probability: Probability of this outcome (for renormalization)
    /// - Returns: Collapsed state with unmeasured qubits still in superposition
    @_effects(readonly)
    @inlinable
    @_eagerMove
    public static func partialCollapse(
        qubit: Int,
        outcome: Int,
        state: QuantumState,
        probability: Double
    ) -> QuantumState {
        precondition(outcome == 0 || outcome == 1, "Outcome must be 0 or 1")
        precondition(probability > 0, "Probability must be positive")

        // Renormalization factor: 1/√P(outcome)
        let normalizationFactor = 1.0 / sqrt(probability)

        let newAmplitudes = AmplitudeVector(unsafeUninitializedCapacity: state.stateSpaceSize) { buffer, count in
            for i in 0 ..< state.stateSpaceSize {
                buffer[i] = state.getBit(index: i, qubit: qubit) == outcome
                    ? state.amplitudes[i] * normalizationFactor
                    : .zero
            }
            count = state.stateSpaceSize
        }

        return QuantumState(numQubits: state.numQubits, amplitudes: newAmplitudes)
    }

    // MARK: - Statevector Snapshots

    /// Capture non-destructive statevector snapshot
    ///
    /// Creates a snapshot of the quantum state without measurement collapse.
    /// Preserves full quantum coherence and superposition. Unlike measurements,
    /// snapshots don't affect the state and can be taken at any point in computation.
    ///
    /// **Use cases**:
    /// - **Algorithm debugging**: Verify intermediate states match theory
    /// - **State tomography**: Reconstruct density matrix from full statevector
    /// - **Visualization**: Display wavefunction evolution over time
    /// - **Testing**: Compare expected vs actual states during development
    ///
    /// **Note**: This operation is not physically realizable on quantum hardware
    /// (would violate no-cloning theorem). It's a simulation-only tool for analysis.
    ///
    /// - Parameters:
    ///   - state: Quantum state to capture
    ///   - label: Optional descriptive label for snapshot
    /// - Returns: Snapshot containing state copy, label, and timestamp
    ///
    /// Example:
    /// ```swift
    /// // Capture snapshots at different circuit points
    /// var circuit = QuantumCircuit(numQubits: 2)
    /// let initial = Measurement.captureSnapshot(
    ///     state: circuit.execute(),
    ///     label: "Initial |00⟩"
    /// )
    ///
    /// circuit.append(gate: .hadamard, toQubit: 0)
    /// let afterH = Measurement.captureSnapshot(
    ///     state: circuit.execute(),
    ///     label: "After H(0)"
    /// )
    /// // afterH.state: |+⟩⊗|0⟩ (no collapse, full superposition preserved)
    ///
    /// circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
    /// let final = Measurement.captureSnapshot(
    ///     state: circuit.execute(),
    ///     label: "Bell state"
    /// )
    /// // final.state: (|00⟩ + |11⟩)/√2
    ///
    /// // Verify expectation values without measurement
    /// let p00 = final.state.probability(ofState: 0b00)  // 0.5
    /// let p11 = final.state.probability(ofState: 0b11)  // 0.5
    ///
    /// // Compare snapshots
    /// print("Evolution from \(initial.label ?? "unknown") to \(final.label ?? "unknown")")
    /// print("Time elapsed: \(final.timestamp.timeIntervalSince(initial.timestamp))s")
    /// ```
    @_effects(readonly)
    @_eagerMove
    public static func captureSnapshot(state: QuantumState, label: String? = nil) -> StateSnapshot {
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
    /// visualizing measurement outcomes. Use with `histogram()` for frequency analysis.
    ///
    /// **Use cases**:
    /// - Algorithm validation (compare observed vs expected distribution)
    /// - Visualization (plot measurement frequencies)
    /// - Statistical testing (chi-squared goodness-of-fit)
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit to execute
    ///   - numRuns: Number of independent executions (≥ 1000 recommended for statistics)
    /// - Returns: Array of measurement outcomes [outcome₁, outcome₂, ..., outcomeₙ]
    ///
    /// Example:
    /// ```swift
    /// // Measure Bell state 1000 times
    /// let bellCircuit = QuantumCircuit.bellPhiPlus()
    /// var measurement = Measurement()
    /// let outcomes = measurement.runMultiple(circuit: bellCircuit, numRuns: 1000)
    ///
    /// // Convert to histogram
    /// let counts = Measurement.histogram(outcomes: outcomes, numQubits: 2)
    /// // counts ≈ [~500, 0, 0, ~500] for Bell state
    /// print("Measured |00⟩: \(counts[0]) times")
    /// print("Measured |11⟩: \(counts[3]) times")
    ///
    /// // Compare to expected distribution
    /// let expected = [0.5, 0.0, 0.0, 0.5]
    /// let error = Measurement.compareDistributions(
    ///     observed: counts,
    ///     expected: expected,
    ///     totalRuns: 1000
    /// )
    /// // error < 0.1 (within 10% for reasonable sample size)
    ///
    /// // Chi-squared test
    /// let chiSq = Measurement.chiSquared(
    ///     observed: counts,
    ///     expected: expected,
    ///     totalRuns: 1000
    /// )
    /// // chiSq.chiSquared < critical value → good fit
    /// ```
    @_eagerMove
    public mutating func runMultiple(circuit: QuantumCircuit, numRuns: Int) -> [Int] {
        precondition(numRuns > 0, "Number of runs must be positive")

        var outcomes: [Int] = []
        outcomes.reserveCapacity(numRuns)

        for _ in 0 ..< numRuns {
            let finalState = circuit.execute()
            let result = measure(state: finalState)
            outcomes.append(result.outcome)
        }

        return outcomes
    }

    /// Convert outcomes to histogram (count per basis state).
    /// - Parameters:
    ///   - outcomes: Array of measurement outcomes
    ///   - numQubits: Number of qubits (determines state space size)
    /// - Returns: Array of counts [count(0), count(1), ..., count(2^n-1)]
    @_effects(readonly)
    @inlinable
    @_eagerMove
    public static func histogram(outcomes: [Int], numQubits: Int) -> [Int] {
        let stateSpaceSize = 1 << numQubits
        var counts = [Int](repeating: 0, count: stateSpaceSize)

        for outcome in outcomes {
            if outcome >= 0, outcome < stateSpaceSize {
                counts[outcome] += 1
            }
        }

        return counts
    }

    /// Compare observed frequencies to expected probabilities.
    /// - Parameters:
    ///   - observed: Observed outcome counts
    ///   - expected: Expected probabilities
    ///   - totalRuns: Total number of measurements
    /// - Returns: Maximum relative error across all outcomes
    public static func compareDistributions(
        observed: [Int],
        expected: [Double],
        totalRuns: Int
    ) -> Double {
        precondition(observed.count == expected.count, "Array sizes must match")
        precondition(totalRuns > 0, "Total runs must be positive")

        var maxError = 0.0

        for i in 0 ..< observed.count {
            let observedFreq = Double(observed[i]) / Double(totalRuns)
            let expectedFreq = expected[i]

            // Avoid division by zero
            if expectedFreq > 0 {
                let relativeError = abs(observedFreq - expectedFreq) / expectedFreq
                maxError = max(maxError, relativeError)
            } else if observedFreq > 0 {
                // Expected 0 but observed something
                maxError = max(maxError, observedFreq)
            }
        }

        return maxError
    }

    /// Chi-squared goodness-of-fit test result
    @frozen
    public struct ChiSquaredResult {
        /// Chi-squared statistic (lower is better fit)
        public let chiSquared: Double

        /// Degrees of freedom (bins tested - 1)
        public let degreesOfFreedom: Int

        /// Number of bins that were tested (expectedCount >= 5)
        public let testedBins: Int

        /// Number of bins that were skipped (expectedCount < 5)
        public let skippedBins: Int
    }

    /// Chi-squared goodness-of-fit test.
    /// Tests whether observed distribution matches expected.
    ///
    /// - Parameters:
    ///   - observed: Observed outcome counts
    ///   - expected: Expected probabilities
    ///   - totalRuns: Total number of measurements
    /// - Returns: Chi-squared result with statistic and degrees of freedom
    public static func chiSquared(
        observed: [Int],
        expected: [Double],
        totalRuns: Int
    ) -> ChiSquaredResult {
        precondition(observed.count == expected.count, "Array sizes must match")
        precondition(totalRuns > 0, "Total runs must be positive")

        var chiSq = 0.0
        var testedBins = 0
        var skippedBins = 0

        for i in 0 ..< observed.count {
            let expectedCount = expected[i] * Double(totalRuns)

            // Skip if expected count too small (chi-squared invalid)
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
}

// MARK: - Static Helper Functions

public extension Measurement {
    /// Measure state with a fresh RNG (convenience function).
    /// - Parameter state: Quantum state to measure
    /// - Returns: Measurement result
    static func measureOnce(state: QuantumState) -> MeasurementResult {
        var measurement = Measurement()
        return measurement.measure(state: state)
    }

    /// Run circuit multiple times and collect measurement outcomes (static convenience).
    /// - Parameters:
    ///   - circuit: Quantum circuit to execute
    ///   - numRuns: Number of measurement runs
    /// - Returns: Array of outcomes
    static func runMultiple(circuit: QuantumCircuit, numRuns: Int) -> [Int] {
        var measurement = Measurement()
        return measurement.runMultiple(circuit: circuit, numRuns: numRuns)
    }

    /// Measure single qubit with fresh RNG (convenience function).
    /// - Parameters:
    ///   - qubit: Qubit index to measure
    ///   - state: Quantum state
    /// - Returns: Outcome and collapsed state
    static func measureQubitOnce(_ qubit: Int, state: QuantumState) -> (outcome: Int, collapsedState: QuantumState) {
        var measurement = Measurement()
        return measurement.measureQubit(qubit, state: state)
    }
}
