// Copyright (c) 2025-2026 Roman Zhuzhgov, Apache License 2.0

import GameplayKit

/// Extended stabilizer state for non-Clifford circuit simulation via stabilizer rank decomposition.
///
/// Represents a quantum state as a weighted sum of stabilizer states: |psi> = sum_i c_i |s_i>.
/// This enables exact simulation of circuits containing non-Clifford gates (particularly T-gates)
/// at a cost that grows exponentially with the number of non-Clifford gates rather than the
/// number of qubits. Practical for circuits with up to ~50 T-gates where rank reaches 2^50.
///
/// The T-gate decomposes as: T|psi> = (|psi> + e^{i*pi/4} Z|psi>) / sqrt(2), doubling the
/// stabilizer rank with each application. Clifford gates apply to all terms without rank growth.
///
/// **Example:**
/// ```swift
/// var state = ExtendedStabilizerState(qubits: 3, maxRank: 1024)
/// state.apply(.hadamard, to: 0)
/// state.apply(.tGate, to: 0)
/// state.apply(.cnot, to: [0, 1])
/// let prob = state.probability(of: 0)
/// ```
///
/// - SeeAlso: ``StabilizerTableau``
/// - SeeAlso: ``QuantumGate``
@frozen public struct ExtendedStabilizerState: Sendable, Equatable, CustomStringConvertible {
    @usableFromInline static let epsilon: Double = 1e-15
    @usableFromInline static let tPhase = Complex<Double>(phase: .pi / 4.0)
    @usableFromInline static let tCoeff1 = (Complex<Double>.one + tPhase) * 0.5
    @usableFromInline static let tCoeff2 = (Complex<Double>.one - tPhase) * 0.5

    @usableFromInline var terms: ContiguousArray<(coefficient: Complex<Double>, tableau: StabilizerTableau)>

    /// Number of qubits in this extended stabilizer state.
    ///
    /// **Example:**
    /// ```swift
    /// let state = ExtendedStabilizerState(qubits: 5, maxRank: 256)
    /// let n = state.qubits
    /// print(n)  // 5
    /// ```
    public let qubits: Int

    /// Maximum allowed stabilizer rank before truncation.
    ///
    /// Controls memory growth during non-Clifford gate application. Each T-gate doubles rank,
    /// so maxRank limits the number of T-gates to approximately log2(maxRank).
    ///
    /// **Example:**
    /// ```swift
    /// let state = ExtendedStabilizerState(qubits: 10, maxRank: 1 << 20)
    /// let limit = state.maxRank
    /// print(limit)  // 1048576
    /// ```
    public let maxRank: Int

    /// Current stabilizer rank (number of stabilizer state terms in the decomposition).
    ///
    /// Starts at 1 for pure stabilizer states and doubles with each T-gate application.
    /// Equal to the number of terms in the sum |psi> = sum_i c_i |s_i>.
    ///
    /// **Example:**
    /// ```swift
    /// var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
    /// state.apply(.tGate, to: 0)
    /// print(state.rank)  // 2 (T-gate doubles rank)
    /// ```
    @inlinable
    public var rank: Int {
        terms.count
    }

    /// Approximate memory usage in bytes for the extended stabilizer state.
    ///
    /// Memory scales as O(rank * n^2) where n is the number of qubits, as each term
    /// contains a full stabilizer tableau requiring O(n^2) bits of storage.
    ///
    /// **Example:**
    /// ```swift
    /// let state = ExtendedStabilizerState(qubits: 10, maxRank: 256)
    /// let bytes = state.memoryUsage
    /// print(bytes)  // Bytes used by all tableaus and coefficients
    /// ```
    ///
    /// - Complexity: O(rank)
    @inlinable
    public var memoryUsage: Int {
        let coeffSize = MemoryLayout<Complex<Double>>.size
        var total = 0
        for term in terms {
            total += term.tableau.memoryUsage + coeffSize
        }
        return total
    }

    /// Creates an extended stabilizer state initialized to |0...0>.
    ///
    /// The initial state is a single stabilizer state term with coefficient 1.0,
    /// representing the computational basis state where all qubits are in |0>.
    ///
    /// **Example:**
    /// ```swift
    /// let state = ExtendedStabilizerState(qubits: 4, maxRank: 1024)
    /// print(state.rank)       // 1
    /// print(state.maxRank)    // 1024
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (must be positive)
    ///   - maxRank: Maximum stabilizer rank limit (must be positive)
    /// - Precondition: qubits > 0
    /// - Precondition: maxRank > 0
    /// - Complexity: O(n^2/w) where n = qubits, w = 64
    public init(qubits: Int, maxRank: Int) {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validatePositiveInt(maxRank, name: "maxRank")

        self.qubits = qubits
        self.maxRank = maxRank

        let initialTableau = StabilizerTableau(qubits: qubits)
        terms = [(coefficient: .one, tableau: initialTableau)]
    }

    /// Applies a single-qubit gate to the specified qubit.
    ///
    /// Clifford gates (H, S, X, Y, Z) are applied to all terms without increasing rank.
    /// The T-gate doubles the rank via the decomposition T|psi> = (|psi> + e^{i*pi/4} Z|psi>) / sqrt(2).
    ///
    /// **Example:**
    /// ```swift
    /// var state = ExtendedStabilizerState(qubits: 3, maxRank: 256)
    /// state.apply(.hadamard, to: 0)
    /// state.apply(.tGate, to: 0)
    /// state.apply(.sGate, to: 1)
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Single-qubit gate to apply
    ///   - qubit: Target qubit index (0 to qubits-1)
    /// - Precondition: qubit >= 0 && qubit < qubits
    /// - Complexity: O(rank * n/w) for Clifford gates, O(rank * n^2/w) for T-gate due to rank doubling
    @inlinable
    @_optimize(speed)
    public mutating func apply(_ gate: QuantumGate, to qubit: Int) {
        ValidationUtilities.validateQubitIndex(qubit, qubits: qubits)

        switch gate {
        case .tGate:
            applyTGate(to: qubit)
        case let .rotationX(angle):
            if case let .value(theta) = angle {
                applyRotation(pauliAxis: .pauliX, angle: theta, to: qubit)
            }
        case let .rotationY(angle):
            if case let .value(theta) = angle {
                applyRotation(pauliAxis: .pauliY, angle: theta, to: qubit)
            }
        case let .rotationZ(angle):
            if case let .value(theta) = angle {
                applyRotation(pauliAxis: .pauliZ, angle: theta, to: qubit)
            }
        case let .phase(angle):
            if case let .value(theta) = angle {
                applyPhaseGate(angle: theta, to: qubit)
            }
        default:
            applyCliffordSingleQubit(gate, to: qubit)
        }
    }

    /// Applies a multi-qubit gate to the specified qubits.
    ///
    /// Supported Clifford gates: CNOT, CZ, SWAP. These are applied to all terms
    /// without increasing the stabilizer rank.
    ///
    /// **Example:**
    /// ```swift
    /// var state = ExtendedStabilizerState(qubits: 3, maxRank: 256)
    /// state.apply(.hadamard, to: 0)
    /// state.apply(.cnot, to: [0, 1])
    /// state.apply(.cz, to: [1, 2])
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Multi-qubit Clifford gate to apply
    ///   - qubits: Target qubit indices (order depends on gate type)
    /// - Precondition: All qubit indices must be valid and unique
    /// - Complexity: O(rank * n/w)
    @inlinable
    @_optimize(speed)
    public mutating func apply(_ gate: QuantumGate, to qubits: [Int]) {
        ValidationUtilities.validateOperationQubits(qubits, numQubits: self.qubits)

        for i in 0 ..< terms.count {
            var tableau = terms[i].tableau
            tableau.apply(gate, to: qubits)
            terms[i] = (coefficient: terms[i].coefficient, tableau: tableau)
        }
    }

    /// Computes the amplitude of a specific computational basis state.
    ///
    /// The amplitude is computed as: amplitude(k) = sum_i c_i * tableau_i.amplitude(k)
    /// where the sum runs over all stabilizer state terms in the decomposition.
    ///
    /// **Example:**
    /// ```swift
    /// var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
    /// state.apply(.hadamard, to: 0)
    /// state.apply(.tGate, to: 0)
    /// let amp = state.amplitude(of: 0)
    /// print(amp)  // Amplitude of |00>
    /// ```
    ///
    /// - Parameter basisState: Computational basis state index (0 to 2^n-1)
    /// - Returns: Complex amplitude of the basis state
    /// - Precondition: basisState >= 0
    /// - Complexity: O(rank * n^3) worst case
    @_effects(readonly)
    @inlinable
    public func amplitude(of basisState: Int) -> Complex<Double> {
        ValidationUtilities.validateNonNegativeInt(basisState, name: "Basis state")

        var result = Complex<Double>.zero

        for term in terms {
            guard let tableauAmplitude = term.tableau.amplitude(of: basisState) else {
                continue
            }
            result = result + term.coefficient * tableauAmplitude
        }

        return result
    }

    /// Computes the probability of measuring a specific computational basis state.
    ///
    /// The probability is |amplitude(basisState)|^2.
    ///
    /// **Example:**
    /// ```swift
    /// var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
    /// state.apply(.hadamard, to: 0)
    /// state.apply(.tGate, to: 0)
    /// let prob = state.probability(of: 0)
    /// print(prob)  // Probability of measuring |00>
    /// ```
    ///
    /// - Parameter basisState: Computational basis state index (0 to 2^n-1)
    /// - Returns: Probability of measuring the basis state
    /// - Complexity: O(rank * n^3)
    @_effects(readonly)
    @inlinable
    public func probability(of basisState: Int) -> Double {
        let amp = amplitude(of: basisState)
        return amp.magnitudeSquared
    }

    /// Performs a projective measurement on the specified qubit.
    ///
    /// Measures the qubit in the computational (Z) basis, collapsing the state
    /// and returning the measurement outcome (0 or 1). The state is updated
    /// to reflect the post-measurement collapsed state.
    ///
    /// **Example:**
    /// ```swift
    /// var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
    /// state.apply(.hadamard, to: 0)
    /// state.apply(.tGate, to: 0)
    /// state.apply(.cnot, to: [0, 1])
    /// let outcome = state.measure(0, seed: 42)
    /// print(outcome)  // 0 or 1
    /// ```
    ///
    /// - Parameters:
    ///   - qubit: Qubit index to measure (0 to qubits-1)
    ///   - seed: Optional seed for reproducible random outcomes
    /// - Returns: Measurement outcome (0 or 1)
    /// - Precondition: qubit >= 0 && qubit < qubits
    /// - Complexity: O(rank * n^2/w)
    @inlinable
    @_optimize(speed)
    public mutating func measure(_ qubit: Int, seed: UInt64?) -> Int {
        ValidationUtilities.validateQubitIndex(qubit, qubits: qubits)

        let prob0 = computeMeasurementProbability(qubit: qubit)

        let random: Double
        if let seed {
            let source = GKMersenneTwisterRandomSource(seed: seed)
            random = Double(source.nextUniform())
        } else {
            random = Double.random(in: 0.0 ..< 1.0)
        }

        let outcome = random < prob0 ? 0 : 1
        let probability = outcome == 0 ? prob0 : 1.0 - prob0

        collapseOnMeasurement(qubit: qubit, outcome: outcome, probability: probability)

        return outcome
    }

    /// Converts the extended stabilizer state to a full statevector representation.
    ///
    /// Computes all 2^n complex amplitudes by summing weighted contributions
    /// from each stabilizer term. The result is a normalized ``QuantumState``
    /// suitable for statevector operations and ``BackendDispatch``
    /// interoperability.
    ///
    /// **Example:**
    /// ```swift
    /// var state = ExtendedStabilizerState(qubits: 2, maxRank: 64)
    /// state.apply(.hadamard, to: 0)
    /// state.apply(.tGate, to: 0)
    /// let quantumState = state.toQuantumState()
    /// ```
    ///
    /// - Returns: Full statevector quantum state
    /// - Precondition: Number of qubits must be 20 or fewer
    /// - Complexity: O(rank * n^3 * 2^n)
    /// - SeeAlso: ``amplitude(of:)``
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public func toQuantumState() -> QuantumState {
        ValidationUtilities.validateStabilizerToStatevectorLimit(qubits)

        let stateSize = 1 << qubits
        let amplitudes = [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            for i in 0 ..< stateSize {
                buffer[i] = self.amplitude(of: i)
            }
            count = stateSize
        }
        return QuantumState(qubits: qubits, rawAmplitudes: amplitudes)
    }

    /// Human-readable description of the extended stabilizer state.
    ///
    /// **Example:**
    /// ```swift
    /// let state = ExtendedStabilizerState(qubits: 5, maxRank: 256)
    /// print(state)
    /// // ExtendedStabilizerState(5 qubits, rank=1, maxRank=256)
    /// ```
    public var description: String {
        "ExtendedStabilizerState(\(qubits) qubits, rank=\(rank), maxRank=\(maxRank))"
    }

    @inlinable
    public static func == (lhs: ExtendedStabilizerState, rhs: ExtendedStabilizerState) -> Bool {
        guard lhs.qubits == rhs.qubits else { return false }
        guard lhs.maxRank == rhs.maxRank else { return false }
        guard lhs.terms.count == rhs.terms.count else { return false }

        for i in 0 ..< lhs.terms.count {
            guard lhs.terms[i].coefficient == rhs.terms[i].coefficient else { return false }
            guard lhs.terms[i].tableau == rhs.terms[i].tableau else { return false }
        }

        return true
    }

    /// Applies a Clifford gate to all stabilizer terms without rank growth.
    @inlinable
    @_optimize(speed)
    mutating func applyCliffordSingleQubit(_ gate: QuantumGate, to qubit: Int) {
        for i in 0 ..< terms.count {
            var tableau = terms[i].tableau
            tableau.apply(gate, to: qubit)
            terms[i] = (coefficient: terms[i].coefficient, tableau: tableau)
        }
    }

    /// Doubles stabilizer rank via T-gate decomposition: T|ψ⟩ = (|ψ⟩ + e^{iπ/4} Z|ψ⟩)/√2.
    @inlinable
    @_optimize(speed)
    mutating func applyTGate(to qubit: Int) {
        let currentCount = terms.count
        let newCount = currentCount * 2

        guard newCount <= maxRank else {
            for i in 0 ..< terms.count {
                var tableau = terms[i].tableau
                tableau.apply(.sGate, to: qubit)
                terms[i] = (coefficient: terms[i].coefficient, tableau: tableau)
            }
            return
        }

        var newTerms = ContiguousArray<(coefficient: Complex<Double>, tableau: StabilizerTableau)>()
        newTerms.reserveCapacity(newCount)

        for term in terms {
            newTerms.append((coefficient: term.coefficient * Self.tCoeff1, tableau: term.tableau))

            var zTableau = term.tableau
            zTableau.apply(.pauliZ, to: qubit)
            newTerms.append((coefficient: term.coefficient * Self.tCoeff2, tableau: zTableau))
        }

        terms = newTerms
    }

    /// Rotation R_P(θ) = cos(θ/2)·I − i·sin(θ/2)·P via stabilizer rank decomposition.
    @inlinable
    @_optimize(speed)
    mutating func applyRotation(pauliAxis: QuantumGate, angle theta: Double, to qubit: Int) {
        let cosHalf = cos(theta / 2.0)
        let sinHalf = sin(theta / 2.0)

        if abs(sinHalf) < Self.epsilon {
            return
        }

        if abs(cosHalf) < Self.epsilon {
            let globalPhase = Complex<Double>(0.0, sinHalf > 0 ? -1.0 : 1.0)
            for i in 0 ..< terms.count {
                var tableau = terms[i].tableau
                tableau.apply(pauliAxis, to: qubit)
                terms[i] = (coefficient: terms[i].coefficient * globalPhase, tableau: tableau)
            }
            return
        }

        let currentCount = terms.count
        let newCount = currentCount * 2

        guard newCount <= maxRank else { return }

        let c1 = Complex<Double>(cosHalf, 0.0)
        let c2 = Complex<Double>(0.0, -sinHalf)

        var newTerms = ContiguousArray<(coefficient: Complex<Double>, tableau: StabilizerTableau)>()
        newTerms.reserveCapacity(newCount)

        for term in terms {
            newTerms.append((coefficient: term.coefficient * c1, tableau: term.tableau))

            var pauliTableau = term.tableau
            pauliTableau.apply(pauliAxis, to: qubit)
            newTerms.append((coefficient: term.coefficient * c2, tableau: pauliTableau))
        }

        terms = newTerms
    }

    /// Phase gate P(θ) = diag(1, e^{iθ}) via stabilizer rank decomposition.
    @inlinable
    @_optimize(speed)
    mutating func applyPhaseGate(angle theta: Double, to qubit: Int) {
        let expPhase = Complex<Double>(phase: theta)
        let c1 = (Complex<Double>.one + expPhase) * 0.5
        let c2 = (Complex<Double>.one - expPhase) * 0.5

        if c2.magnitudeSquared < Self.epsilon * Self.epsilon {
            for i in 0 ..< terms.count {
                terms[i] = (coefficient: terms[i].coefficient * c1, tableau: terms[i].tableau)
            }
            return
        }

        if c1.magnitudeSquared < Self.epsilon * Self.epsilon {
            for i in 0 ..< terms.count {
                var tableau = terms[i].tableau
                tableau.apply(.pauliZ, to: qubit)
                terms[i] = (coefficient: terms[i].coefficient * c2, tableau: tableau)
            }
            return
        }

        let currentCount = terms.count
        let newCount = currentCount * 2

        guard newCount <= maxRank else { return }

        var newTerms = ContiguousArray<(coefficient: Complex<Double>, tableau: StabilizerTableau)>()
        newTerms.reserveCapacity(newCount)

        for term in terms {
            newTerms.append((coefficient: term.coefficient * c1, tableau: term.tableau))

            var zTableau = term.tableau
            zTableau.apply(.pauliZ, to: qubit)
            newTerms.append((coefficient: term.coefficient * c2, tableau: zTableau))
        }

        terms = newTerms
    }

    /// Computes probability of a specific measurement outcome on one qubit.
    @inlinable
    @_effects(readonly)
    func computeMeasurementProbability(qubit: Int) -> Double {
        let halfDimension = 1 << (qubits - 1)
        var totalProb = 0.0

        for i in 0 ..< halfDimension {
            let basisState = BitUtilities.insertZeroBit(i, at: qubit)
            let amp = amplitude(of: basisState)
            totalProb += amp.magnitudeSquared
        }

        return totalProb
    }

    /// Collapses state by projecting terms onto measurement outcome and renormalizing.
    @inlinable
    @_optimize(speed)
    mutating func collapseOnMeasurement(qubit: Int, outcome: Int, probability: Double) {
        var newTerms = ContiguousArray<(coefficient: Complex<Double>, tableau: StabilizerTableau)>()
        newTerms.reserveCapacity(terms.count)

        for term in terms {
            var tableau = term.tableau
            guard let scaleFactor = tableau.projectMeasurement(qubit, outcome: outcome) else {
                continue
            }
            let scaledCoeff = term.coefficient * scaleFactor
            newTerms.append((coefficient: scaledCoeff, tableau: tableau))
        }

        if newTerms.isEmpty {
            let tableau = StabilizerTableau(qubits: qubits)
            newTerms.append((coefficient: .one, tableau: tableau))
        }

        if probability > Self.epsilon {
            let normFactor = 1.0 / probability.squareRoot()
            for i in 0 ..< newTerms.count {
                let scaledCoeff = newTerms[i].coefficient * normFactor
                newTerms[i] = (coefficient: scaledCoeff, tableau: newTerms[i].tableau)
            }
        }

        terms = newTerms
    }
}
