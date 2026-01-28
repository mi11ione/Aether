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
    @usableFromInline var terms: ContiguousArray<(coefficient: Complex<Double>, tableau: StabilizerTableau)>

    /// Number of qubits in this extended stabilizer state.
    ///
    /// **Example:**
    /// ```swift
    /// let state = ExtendedStabilizerState(qubits: 5, maxRank: 256)
    /// print(state.qubits)  // 5
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
    /// print(state.maxRank)  // 1048576
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
    /// print(state.memoryUsage)  // Bytes used by all tableaus and coefficients
    /// ```
    @inlinable
    public var memoryUsage: Int {
        var total = 0
        for term in terms {
            total += term.tableau.memoryUsage + MemoryLayout<Complex<Double>>.size
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
    /// print(state.rank)  // 1
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (must be positive)
    ///   - maxRank: Maximum stabilizer rank limit (must be positive)
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
    /// - Complexity: O(rank * n/w) for Clifford gates, O(rank * n^2/w) for T-gate due to rank doubling
    @inlinable
    @_optimize(speed)
    public mutating func apply(_ gate: QuantumGate, to qubit: Int) {
        ValidationUtilities.validateQubitIndex(qubit, qubits: qubits)

        switch gate {
        case .tGate:
            applyTGate(to: qubit)
        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard, .sGate, .sx, .sy:
            applyCliffordSingleQubit(gate, to: qubit)
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
    /// - Complexity: O(rank * n/w)
    @inlinable
    @_optimize(speed)
    public mutating func apply(_ gate: QuantumGate, to qubits: [Int]) {
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
    /// - Complexity: O(rank * n^2/w)
    @inlinable
    @_optimize(speed)
    public mutating func measure(_ qubit: Int, seed: UInt64?) -> Int {
        ValidationUtilities.validateQubitIndex(qubit, qubits: qubits)

        let prob0 = computeMeasurementProbability(qubit: qubit, outcome: 0)

        let random: Double
        if let seed {
            let source = GKMersenneTwisterRandomSource(seed: seed)
            random = Double(source.nextUniform())
        } else {
            random = Double.random(in: 0.0 ..< 1.0)
        }

        let outcome = random < prob0 ? 0 : 1

        collapseOnMeasurement(qubit: qubit, outcome: outcome)

        return outcome
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

    @inlinable
    @_optimize(speed)
    mutating func applyCliffordSingleQubit(_ gate: QuantumGate, to qubit: Int) {
        for i in 0 ..< terms.count {
            var tableau = terms[i].tableau
            tableau.apply(gate, to: qubit)
            terms[i] = (coefficient: terms[i].coefficient, tableau: tableau)
        }
    }

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

        let tPhase = Complex<Double>(phase: .pi / 4.0)
        let coeff1 = (Complex<Double>.one + tPhase) * 0.5
        let coeff2 = (Complex<Double>.one - tPhase) * 0.5

        var newTerms = ContiguousArray<(coefficient: Complex<Double>, tableau: StabilizerTableau)>()
        newTerms.reserveCapacity(newCount)

        for term in terms {
            newTerms.append((coefficient: term.coefficient * coeff1, tableau: term.tableau))

            var zTableau = term.tableau
            zTableau.apply(.pauliZ, to: qubit)
            newTerms.append((coefficient: term.coefficient * coeff2, tableau: zTableau))
        }

        terms = newTerms
    }

    @inlinable
    @_effects(readonly)
    func computeMeasurementProbability(qubit: Int, outcome: Int) -> Double {
        let dimension = 1 << qubits
        var totalProb = 0.0

        for basisState in 0 ..< dimension {
            let bit = (basisState >> qubit) & 1
            if bit == outcome {
                let amp = amplitude(of: basisState)
                totalProb += amp.magnitudeSquared
            }
        }

        return totalProb
    }

    @inlinable
    @_optimize(speed)
    mutating func collapseOnMeasurement(qubit: Int, outcome: Int) {
        var newTerms = ContiguousArray<(coefficient: Complex<Double>, tableau: StabilizerTableau)>()
        newTerms.reserveCapacity(terms.count)

        for term in terms {
            var tableau = term.tableau
            let measureOutcome = tableau.measure(qubit, seed: nil)

            if measureOutcome == outcome {
                newTerms.append((coefficient: term.coefficient, tableau: tableau))
            }
        }

        if newTerms.isEmpty {
            let tableau = StabilizerTableau(qubits: qubits)
            newTerms.append((coefficient: .one, tableau: tableau))
        }

        var normSq = 0.0
        let dimension = 1 << qubits

        terms = newTerms

        for basisState in 0 ..< dimension {
            let amp = amplitude(of: basisState)
            normSq += amp.magnitudeSquared
        }

        if normSq > 1e-15 {
            let normFactor = 1.0 / normSq.squareRoot()
            for i in 0 ..< terms.count {
                let scaledCoeff = terms[i].coefficient * normFactor
                terms[i] = (coefficient: scaledCoeff, tableau: terms[i].tableau)
            }
        }
    }
}
