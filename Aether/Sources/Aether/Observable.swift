// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Array of Pauli terms with coefficients.
///
/// Each term is a (coefficient, PauliString) pair where the coefficient is real.
/// Used to represent weighted Pauli strings in Hamiltonians and observables.
public typealias PauliTerms = [(coefficient: Double, pauliString: PauliString)]

/// Quantum observable represented as a weighted sum of Pauli strings.
///
/// An observable is a Hermitian operator O = Σᵢ cᵢ Pᵢ where each Pᵢ is a tensor product
/// of Pauli operators {I,X,Y,Z} and cᵢ are real coefficients. This representation is fundamental
/// to quantum chemistry, VQE optimization, and measurement theory. Real coefficients ensure
/// Hermiticity (O† = O), guaranteeing real eigenvalues as required for physical observables.
///
/// The Pauli decomposition spans all Hermitian operators on n qubits, making this representation
/// complete for any measurable quantum property. Common applications include molecular
/// Hamiltonians in quantum chemistry, where electronic structure problems are mapped to
/// Pauli string sums via fermion-to-qubit transformations.
///
/// - SeeAlso: ``PauliString``
/// - SeeAlso: ``VQE``
/// - SeeAlso: ``SparseHamiltonian``
///
/// **Example:**
/// ```swift
/// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
/// let state = QuantumState(qubits: 2)
/// let energy = H.expectationValue(of: state)
/// ```
@frozen
public struct Observable: CustomStringConvertible, Sendable {
    /// Weighted Pauli string terms (cᵢ, Pᵢ) comprising this observable.
    ///
    /// Coefficients must be real to ensure Hermiticity.
    public let terms: PauliTerms

    /// Creates observable from weighted Pauli string terms.
    ///
    /// - Parameter terms: Array of (coefficient, Pauli string) pairs
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
    /// ```
    public init(terms: PauliTerms) {
        self.terms = terms
    }

    /// Creates single-term observable.
    ///
    /// - Parameters:
    ///   - coefficient: Real coefficient
    ///   - pauliString: Pauli string operator
    ///
    /// **Example:**
    /// ```swift
    /// let zObs = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
    /// ```
    public init(coefficient: Double, pauliString: PauliString) {
        terms = [(coefficient, pauliString)]
    }

    // MARK: - Expectation Value

    /// Compute exact expectation value ⟨ψ|O|ψ⟩ from statevector.
    ///
    /// Uses linearity of expectation to compute ⟨O⟩ = Σᵢ cᵢ⟨Pᵢ⟩ where each Pauli string
    /// expectation is evaluated by rotating to the measurement basis where Pᵢ is diagonal,
    /// then computing the weighted sum of eigenvalues ±1 according to the Born rule.
    ///
    /// - Parameter state: Normalized quantum state |ψ⟩
    /// - Returns: Real expectation value ⟨ψ|O|ψ⟩
    /// - Complexity: O(k·2ⁿ) where k is the number of terms and n is the number of qubits
    /// - Precondition: State must be normalized (Σ|cᵢ|² = 1)
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0)))])
    /// let energy = H.expectationValue(of: QuantumState(qubits: 1))
    /// ```
    @_optimize(speed)
    public func expectationValue(of state: QuantumState) -> Double {
        ValidationUtilities.validateNormalizedState(state)

        var totalExpectation = 0.0

        for i in 0 ..< terms.count {
            let pauliExpectation = Self.computePauliExpectation(
                pauliString: terms[i].pauliString,
                for: state,
            )
            totalExpectation += terms[i].coefficient * pauliExpectation
        }

        return totalExpectation
    }

    /// Compute expectation value of single Pauli string ⟨ψ|P|ψ⟩.
    ///
    /// Rotates state to the basis where the Pauli operator is diagonal, then computes
    /// expectation from eigenvalues ±1 and probability distribution via the Born rule.
    ///
    /// - Parameters:
    ///   - pauliString: Pauli string operator
    ///   - state: Normalized quantum state
    /// - Returns: Expectation value in range [-1, +1]
    @_optimize(speed)
    static func computePauliExpectation(
        pauliString: PauliString,
        for state: QuantumState,
    ) -> Double {
        if pauliString.operators.isEmpty { return 1.0 }

        var rotatedState = state
        var qubitMask = 0

        for i in 0 ..< pauliString.operators.count {
            let op = pauliString.operators[i]
            rotatedState = Measurement.rotateToPauliBasis(
                qubit: op.qubit,
                basis: op.basis,
                state: rotatedState,
            )
            qubitMask |= (1 << op.qubit)
        }

        var expectation = 0.0

        for i in 0 ..< rotatedState.stateSpaceSize {
            let parity = (i & qubitMask).nonzeroBitCount & 1
            let eigenvalue = 1 - 2 * parity
            expectation += Double(eigenvalue) * rotatedState.probability(of: i)
        }

        return expectation
    }

    // MARK: - Variance

    /// Compute variance Var(O) = ⟨O²⟩ - ⟨O⟩².
    ///
    /// Variance quantifies measurement uncertainty and determines the number of shots
    /// required for target accuracy. Used in VQE convergence criteria, error analysis,
    /// and shot allocation strategies where variance-weighted sampling improves efficiency.
    ///
    /// Computed by expanding O² = (Σᵢ cᵢ Pᵢ)² = Σᵢⱼ cᵢcⱼ PᵢPⱼ via Pauli multiplication rules,
    /// then evaluating ⟨O²⟩ - ⟨O⟩² using standard expectation value computation.
    ///
    /// - Parameter state: Normalized quantum state
    /// - Returns: Non-negative variance Var(O) ≥ 0
    /// - Complexity: O(k²·2ⁿ) where k is the number of terms
    /// - Precondition: State must be normalized
    /// - SeeAlso: ``ShotAllocator``
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0)))])
    /// let variance = H.variance(of: QuantumState(qubits: 1))
    /// ```
    @_optimize(speed)
    public func variance(of state: QuantumState) -> Double {
        let mean = expectationValue(of: state)
        let meanSquared = squared().expectationValue(of: state)
        return meanSquared - mean * mean
    }

    /// Computes O² by expanding all Pauli string products and combining like terms.
    @_optimize(speed)
    @_eagerMove
    private func squared() -> Observable {
        var coefficientMap: [PauliString: Double] = [:]
        coefficientMap.reserveCapacity(terms.count * terms.count)

        for i in 0 ..< terms.count {
            let ci = terms[i].coefficient
            let Pi = terms[i].pauliString

            for j in 0 ..< terms.count {
                let cj = terms[j].coefficient
                let Pj = terms[j].pauliString

                let (phase, product) = multiplyPauliStrings(Pi, Pj)
                let coefficient = ci * cj * phase.real

                if abs(coefficient) > 1e-15 {
                    coefficientMap[product, default: 0.0] += coefficient
                }
            }
        }

        let squaredTerms: PauliTerms = coefficientMap.compactMap { key, value in
            abs(value) > 1e-15 ? (coefficient: value, pauliString: key) : nil
        }

        return Observable(terms: squaredTerms)
    }

    /// Multiplies two Pauli strings using standard Pauli algebra with phase tracking.
    @_optimize(speed)
    @_eagerMove
    private func multiplyPauliStrings(
        _ lhs: PauliString,
        _ rhs: PauliString,
    ) -> (phase: Complex<Double>, result: PauliString) {
        var pauliMap: [Int: (left: PauliBasis?, right: PauliBasis?)] = [:]
        pauliMap.reserveCapacity(lhs.operators.count + rhs.operators.count)

        for op in lhs.operators {
            pauliMap[op.qubit] = (left: op.basis, right: nil)
        }
        for op in rhs.operators {
            if let existing = pauliMap[op.qubit] {
                pauliMap[op.qubit] = (left: existing.left, right: op.basis)
            } else {
                pauliMap[op.qubit] = (left: nil, right: op.basis)
            }
        }

        var phase = Complex<Double>(1.0, 0.0)
        var resultOperators: [PauliOperator] = []
        resultOperators.reserveCapacity(pauliMap.count)

        for qubit in pauliMap.keys.sorted() {
            // Safety: qubit comes from pauliMap.keys, guaranteed to exist
            let (left, right) = pauliMap[qubit]!

            let (localPhase, resultPauli) = multiplySingleQubitPaulis(
                left: left,
                right: right,
            )

            phase = phase * localPhase

            if let pauli = resultPauli {
                switch pauli {
                case .x: resultOperators.append(.x(qubit))
                case .y: resultOperators.append(.y(qubit))
                case .z: resultOperators.append(.z(qubit))
                }
            }
        }

        return (phase, PauliString(resultOperators))
    }

    /// Multiplies single-qubit Pauli operators returning (phase, result) tuple.
    @_optimize(speed)
    private func multiplySingleQubitPaulis(
        left: PauliBasis?,
        right: PauliBasis?,
    ) -> (phase: Complex<Double>, result: PauliBasis?) {
        guard let l = left else { return (Complex(1.0, 0.0), right) }
        guard let r = right else { return (Complex(1.0, 0.0), left) }

        switch (l, r) {
        case (.x, .x), (.y, .y), (.z, .z):
            return (Complex(1.0, 0.0), nil)
        case (.x, .y):
            return (Complex(0.0, 1.0), .z)
        case (.y, .x):
            return (Complex(0.0, -1.0), .z)
        case (.y, .z):
            return (Complex(0.0, 1.0), .x)
        case (.z, .y):
            return (Complex(0.0, -1.0), .x)
        case (.z, .x):
            return (Complex(0.0, 1.0), .y)
        case (.x, .z):
            return (Complex(0.0, -1.0), .y)
        }
    }

    /// Combines terms with identical Pauli strings by summing coefficients.
    @_optimize(speed)
    @_eagerMove
    private func combineLikeTerms(
        _ inputTerms: PauliTerms,
    ) -> PauliTerms {
        var coefficientMap: [PauliString: Double] = [:]
        coefficientMap.reserveCapacity(inputTerms.count)

        for i in 0 ..< inputTerms.count {
            coefficientMap[inputTerms[i].pauliString, default: 0.0] += inputTerms[i].coefficient
        }

        return coefficientMap.compactMap { key, value in
            abs(value) > 1e-15 ? (coefficient: value, pauliString: key) : nil
        }
    }

    // MARK: - Convenience Factories

    /// Create Pauli-X observable on a single qubit.
    ///
    /// - Parameters:
    ///   - qubit: Target qubit index
    ///   - coefficient: Observable coefficient (default: 1.0)
    /// - Returns: Observable representing coefficient · X
    ///
    /// **Example:**
    /// ```swift
    /// let xObs = Observable.pauliX(qubit: 0)
    /// let value = xObs.expectationValue(of: QuantumState(qubits: 1))
    /// ```
    @_effects(readonly)
    public static func pauliX(qubit: Int, coefficient: Double = 1.0) -> Observable {
        Observable(
            coefficient: coefficient,
            pauliString: PauliString(.x(qubit)),
        )
    }

    /// Create Pauli-Y observable on a single qubit.
    ///
    /// - Parameters:
    ///   - qubit: Target qubit index
    ///   - coefficient: Observable coefficient (default: 1.0)
    /// - Returns: Observable representing coefficient · Y
    ///
    /// **Example:**
    /// ```swift
    /// let yObs = Observable.pauliY(qubit: 1)
    /// ```
    @_effects(readonly)
    public static func pauliY(qubit: Int, coefficient: Double = 1.0) -> Observable {
        Observable(
            coefficient: coefficient,
            pauliString: PauliString(.y(qubit)),
        )
    }

    /// Create Pauli-Z observable on a single qubit.
    ///
    /// - Parameters:
    ///   - qubit: Target qubit index
    ///   - coefficient: Observable coefficient (default: 1.0)
    /// - Returns: Observable representing coefficient · Z
    ///
    /// **Example:**
    /// ```swift
    /// let zObs = Observable.pauliZ(qubit: 0)
    /// let value = zObs.expectationValue(of: QuantumState(qubits: 1))
    /// ```
    @_effects(readonly)
    public static func pauliZ(qubit: Int, coefficient: Double = 1.0) -> Observable {
        Observable(
            coefficient: coefficient,
            pauliString: PauliString(.z(qubit)),
        )
    }

    // MARK: - CustomStringConvertible

    @inlinable
    public var description: String {
        if terms.isEmpty { return "Observable: 0" }

        var result = "Observable: "
        for i in 0 ..< terms.count {
            let coeff = terms[i].coefficient
            let pauli = terms[i].pauliString
            let sign = coeff >= 0 ? "+" : ""
            let pauliDesc = pauli.operators.isEmpty ? "I" : pauli.description
            if i > 0 { result += " " }
            result += "\(sign)\(coeff)·\(pauliDesc)"
        }
        return result
    }
}
