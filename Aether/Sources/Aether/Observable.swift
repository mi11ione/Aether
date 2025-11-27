// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Array of Pauli terms with coefficients.
///
/// Represents weighted Pauli strings used in Hamiltonians and observables.
/// Each term is a (coefficient, PauliString) pair where the coefficient is real.
public typealias PauliTerms = [(coefficient: Double, pauliString: PauliString)]

/// Quantum observable (Hermitian operator) for expectation value measurements
///
/// Represents a Hermitian operator as a weighted sum of Pauli strings:
/// O = Σᵢ cᵢ Pᵢ where Pᵢ ∈ {I,X,Y,Z}^⊗n and cᵢ ∈ ℝ
///
/// **Mathematical Properties:**
/// - Hermitian: O† = O (real eigenvalues, observable in quantum mechanics)
/// - Real coefficients ensure Hermiticity when Pauli strings are Hermitian
/// - Linear combination of Pauli operators spans all Hermitian operators
///
/// **Use Cases:**
/// - **VQE (Variational Quantum Eigensolver)**: Minimize ⟨ψ(θ)|H|ψ(θ)⟩
/// - **Quantum Chemistry**: Molecular Hamiltonians as Pauli decompositions
/// - **Observable measurements**: Energy, magnetization, correlation functions
/// - **Algorithm verification**: Compare expected vs measured observables
///
/// **Example - Hydrogen molecule Hamiltonian:**
/// ```swift
/// // H = -1.05·I + 0.39·Z₀ - 0.39·Z₁ - 0.01·Z₀⊗Z₁ + 0.18·X₀⊗X₁
/// let hamiltonian = Observable(terms: [
///     (-1.05, PauliString(operators: [])),                    // Identity
///     (0.39,  PauliString(operators: [(0, .z)])),             // Z₀
///     (-0.39, PauliString(operators: [(1, .z)])),             // Z₁
///     (-0.01, PauliString(operators: [(0, .z), (1, .z)])),    // Z₀⊗Z₁
///     (0.18,  PauliString(operators: [(0, .x), (1, .x)]))     // X₀⊗X₁
/// ])
///
/// // Compute ground state energy
/// let groundState = // ... prepare trial state
/// let energy = hamiltonian.expectationValue(state: groundState)
/// ```
///
/// **Example - Simple observable:**
/// ```swift
/// // Measure Z₀ (Pauli-Z on qubit 0)
/// let observable = Observable(
///     coefficient: 1.0,
///     pauliString: PauliString(operators: [(qubit: 0, basis: .z)])
/// )
///
/// let state = QuantumState(numQubits: 1)  // |0⟩
/// let expectation = observable.expectationValue(state: state)
/// // expectation = +1.0 (|0⟩ is +1 eigenstate of Z)
/// ```
@frozen
public struct Observable: CustomStringConvertible, Sendable {
    /// Terms in the observable: weighted Pauli strings (cᵢ, Pᵢ)
    /// Coefficients are real (required for Hermiticity)
    public let terms: PauliTerms

    /// Create observable from weighted Pauli string terms
    /// - Parameter terms: Array of (coefficient, Pauli string) pairs
    public init(terms: PauliTerms) {
        self.terms = terms
    }

    /// Convenience initializer for single-term observable
    /// - Parameters:
    ///   - coefficient: Real coefficient
    ///   - pauliString: Pauli string operator
    public init(coefficient: Double, pauliString: PauliString) {
        terms = [(coefficient, pauliString)]
    }

    // MARK: - Expectation Value

    /// Compute exact expectation value ⟨ψ|O|ψ⟩ from statevector
    ///
    /// Uses linearity of expectation: ⟨O⟩ = Σᵢ cᵢ⟨Pᵢ⟩
    /// Each Pauli string expectation computed via basis rotation.
    ///
    /// **Algorithm:**
    /// 1. For each term (cᵢ, Pᵢ):
    ///    - Rotate state to basis where Pᵢ is diagonal
    ///    - Compute ⟨Pᵢ⟩ = Σₓ λₓ P(x) where λₓ ∈ {±1}
    /// 2. Sum weighted contributions: ⟨O⟩ = Σᵢ cᵢ⟨Pᵢ⟩
    ///
    /// **Complexity:** O(k·2ⁿ) where k = number of terms, n = qubits
    ///
    /// - Parameter state: Normalized quantum state |ψ⟩
    /// - Returns: Expectation value ⟨ψ|O|ψ⟩ ∈ ℝ
    ///
    /// Example:
    /// ```swift
    /// // Compute energy of trial state
    /// let hamiltonian = Observable(terms: [...])
    /// let trialState = prepareAnsatz(parameters: theta)
    /// let energy = hamiltonian.expectationValue(state: trialState)
    /// ```
    @_optimize(speed)
    public func expectationValue(state: QuantumState) -> Double {
        ValidationUtilities.validateNormalizedState(state)

        var totalExpectation = 0.0

        for i in 0 ..< terms.count {
            let pauliExpectation = Self.computePauliExpectation(
                pauliString: terms[i].pauliString,
                state: state
            )
            totalExpectation += terms[i].coefficient * pauliExpectation
        }

        return totalExpectation
    }

    /// Compute expectation value of single Pauli string: ⟨ψ|P|ψ⟩
    ///
    /// Applies basis rotations to diagonalize Pauli operator, then computes
    /// expectation from probability distribution and eigenvalues.
    ///
    /// - Parameters:
    ///   - pauliString: Pauli string operator
    ///   - state: Normalized quantum state
    /// - Returns: Expectation value ⟨P⟩ ∈ [-1, +1]
    @_optimize(speed)
    static func computePauliExpectation(
        pauliString: PauliString,
        state: QuantumState
    ) -> Double {
        if pauliString.operators.isEmpty { return 1.0 }

        var rotatedState = state
        var qubitMask = 0

        for i in 0 ..< pauliString.operators.count {
            let op = pauliString.operators[i]
            rotatedState = Measurement.rotateToPauliBasis(
                qubit: op.qubit,
                basis: op.basis,
                state: rotatedState
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

    /// Compute variance Var(O) = ⟨O²⟩ - ⟨O⟩²
    ///
    /// Variance measures uncertainty in observable measurements.
    /// Used for:
    /// - Error analysis in VQE
    /// - Determining required measurement shots
    /// - Convergence criteria for variational algorithms
    ///
    /// **Algorithm:**
    /// 1. Compute ⟨O⟩ = expectation value
    /// 2. Construct O² by expanding (Σᵢ cᵢ Pᵢ)² = Σᵢⱼ cᵢcⱼ PᵢPⱼ
    /// 3. Compute ⟨O²⟩ = expectation of squared observable
    /// 4. Var(O) = ⟨O²⟩ - ⟨O⟩²
    ///
    /// **Complexity:** O(k²·2ⁿ) where k = number of terms
    ///
    /// - Parameter state: Normalized quantum state
    /// - Returns: Variance Var(O) ≥ 0
    ///
    /// Example:
    /// ```swift
    /// let variance = hamiltonian.variance(state: state)
    /// let uncertainty = sqrt(variance)
    /// // Number of shots needed: O(variance / ε²) for accuracy ε
    /// ```
    @_optimize(speed)
    public func variance(state: QuantumState) -> Double {
        let mean = expectationValue(state: state)
        let meanSquared = squared().expectationValue(state: state)
        return meanSquared - mean * mean
    }

    /// Compute O² by expanding (Σᵢ cᵢ Pᵢ)² = Σᵢⱼ cᵢcⱼ PᵢPⱼ
    ///
    /// Multiplies all pairs of Pauli strings and combines like terms inline.
    /// Result is guaranteed Hermitian (real coefficients) if input is Hermitian.
    ///
    /// - Returns: Observable representing O²
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

    /// Multiply two Pauli strings with proper phase tracking
    ///
    /// Pauli multiplication rules (per qubit):
    /// - X·X = Y·Y = Z·Z = I
    /// - X·Y = iZ,  Y·X = -iZ
    /// - Y·Z = iX,  Z·Y = -iX
    /// - Z·X = iY,  X·Z = -iY
    ///
    /// For tensor products, multiply qubit-by-qubit and track global phase.
    ///
    /// - Parameters:
    ///   - lhs: Left Pauli string
    ///   - rhs: Right Pauli string
    /// - Returns: (phase, resulting Pauli string)
    @_optimize(speed)
    @_eagerMove
    private func multiplyPauliStrings(
        _ lhs: PauliString,
        _ rhs: PauliString
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
        var resultOperators: [(qubit: Int, basis: PauliBasis)] = []
        resultOperators.reserveCapacity(pauliMap.count)

        for qubit in pauliMap.keys.sorted() {
            let (left, right) = pauliMap[qubit]!

            let (localPhase, resultPauli) = multiplySingleQubitPaulis(
                left: left,
                right: right
            )

            phase = phase * localPhase

            if let pauli = resultPauli {
                resultOperators.append((qubit: qubit, basis: pauli))
            }
        }

        return (phase, PauliString(operators: resultOperators))
    }

    /// Multiply single-qubit Pauli operators with phase
    ///
    /// - Parameters:
    ///   - left: Left Pauli (nil = identity)
    ///   - right: Right Pauli (nil = identity)
    /// - Returns: (phase, result Pauli or nil for identity)
    @_optimize(speed)
    @inlinable
    func multiplySingleQubitPaulis(
        left: PauliBasis?,
        right: PauliBasis?
    ) -> (phase: Complex<Double>, result: PauliBasis?) {
        // Handle identity cases
        guard let l = left else { return (Complex(1.0, 0.0), right) }
        guard let r = right else { return (Complex(1.0, 0.0), left) }

        switch (l, r) {
        case (.x, .x), (.y, .y), (.z, .z):
            return (Complex(1.0, 0.0), nil) // P·P = I
        case (.x, .y):
            return (Complex(0.0, 1.0), .z) // X·Y = iZ
        case (.y, .x):
            return (Complex(0.0, -1.0), .z) // Y·X = -iZ
        case (.y, .z):
            return (Complex(0.0, 1.0), .x) // Y·Z = iX
        case (.z, .y):
            return (Complex(0.0, -1.0), .x) // Z·Y = -iX
        case (.z, .x):
            return (Complex(0.0, 1.0), .y) // Z·X = iY
        case (.x, .z):
            return (Complex(0.0, -1.0), .y) // X·Z = -iY
        }
    }

    /// Combine terms with identical Pauli strings
    ///
    /// Groups terms by Pauli string and sums their coefficients.
    /// Removes terms with near-zero coefficients.
    ///
    /// - Parameter terms: Unsimplified terms
    /// - Returns: Simplified terms with combined coefficients
    @_optimize(speed)
    @_eagerMove
    private func combineLikeTerms(
        _ inputTerms: PauliTerms
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
