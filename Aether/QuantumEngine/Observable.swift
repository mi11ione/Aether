// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

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
struct Observable: CustomStringConvertible {
    /// Terms in the observable: weighted Pauli strings (cᵢ, Pᵢ)
    /// Coefficients are real (required for Hermiticity)
    let terms: [(coefficient: Double, pauliString: PauliString)]

    /// Create observable from weighted Pauli string terms
    /// - Parameter terms: Array of (coefficient, Pauli string) pairs
    init(terms: [(coefficient: Double, pauliString: PauliString)]) {
        self.terms = terms
    }

    /// Convenience initializer for single-term observable
    /// - Parameters:
    ///   - coefficient: Real coefficient
    ///   - pauliString: Pauli string operator
    init(coefficient: Double, pauliString: PauliString) {
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
    func expectationValue(state: QuantumState) -> Double {
        precondition(state.isNormalized(), "State must be normalized")

        var totalExpectation = 0.0

        for (coefficient, pauliString) in terms {
            let pauliExpectation = computePauliExpectation(
                pauliString: pauliString,
                state: state
            )
            totalExpectation += coefficient * pauliExpectation
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
    private func computePauliExpectation(
        pauliString: PauliString,
        state: QuantumState
    ) -> Double {
        if pauliString.operators.isEmpty {
            return 1.0 // ⟨I⟩ = 1
        }

        var rotatedState = state
        for op in pauliString.operators {
            rotatedState = Measurement.rotateToPauliBasis(
                qubit: op.qubit,
                basis: op.basis,
                state: rotatedState
            )
        }

        // Compute expectation value: ⟨P⟩ = Σᵢ λᵢ P(i)
        // λᵢ is product of individual Pauli eigenvalues (±1 for each qubit)
        var expectation = 0.0

        for i in 0 ..< rotatedState.stateSpaceSize {
            var eigenvalue = 1
            for op in pauliString.operators {
                let bit = rotatedState.getBit(index: i, qubit: op.qubit)
                eigenvalue *= (bit == 0) ? 1 : -1
            }

            expectation += Double(eigenvalue) * rotatedState.probability(ofState: i)
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
    func variance(state: QuantumState) -> Double {
        let mean = expectationValue(state: state)
        let meanSquared = squared().expectationValue(state: state)
        return meanSquared - mean * mean
    }

    /// Compute O² by expanding (Σᵢ cᵢ Pᵢ)² = Σᵢⱼ cᵢcⱼ PᵢPⱼ
    ///
    /// Multiplies all pairs of Pauli strings and combines like terms.
    /// Result is guaranteed Hermitian (real coefficients) if input is Hermitian.
    ///
    /// - Returns: Observable representing O²
    private func squared() -> Observable {
        var squaredTerms: [(coefficient: Double, pauliString: PauliString)] = []

        // Expand (Σᵢ cᵢ Pᵢ)²
        for (ci, Pi) in terms {
            for (cj, Pj) in terms {
                let (phase, product) = multiplyPauliStrings(Pi, Pj)

                let coefficient = ci * cj * phase.real
                if abs(coefficient) > 1e-15 {
                    squaredTerms.append((coefficient, product))
                }
            }
        }

        return Observable(terms: combineLikeTerms(squaredTerms))
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
    private func multiplyPauliStrings(
        _ lhs: PauliString,
        _ rhs: PauliString
    ) -> (phase: Complex<Double>, result: PauliString) {
        var lhsMap: [Int: PauliBasis] = [:]
        for op in lhs.operators {
            lhsMap[op.qubit] = op.basis
        }

        var rhsMap: [Int: PauliBasis] = [:]
        for op in rhs.operators {
            rhsMap[op.qubit] = op.basis
        }

        let allQubits = Set(lhsMap.keys).union(rhsMap.keys).sorted()

        var phase = Complex<Double>(1.0, 0.0)
        var resultOperators: [(qubit: Int, basis: PauliBasis)] = []

        for qubit in allQubits {
            let leftPauli = lhsMap[qubit]
            let rightPauli = rhsMap[qubit]

            let (localPhase, resultPauli) = multiplySingleQubitPaulis(
                left: leftPauli,
                right: rightPauli
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
    private func multiplySingleQubitPaulis(
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
    private func combineLikeTerms(
        _ terms: [(coefficient: Double, pauliString: PauliString)]
    ) -> [(coefficient: Double, pauliString: PauliString)] {
        var combinedTerms: [(coefficient: Double, pauliString: PauliString)] = []

        for (coefficient, pauliString) in terms {
            if let index = combinedTerms.firstIndex(where: { $0.pauliString == pauliString }) {
                combinedTerms[index].coefficient += coefficient
            } else {
                combinedTerms.append((coefficient, pauliString))
            }
        }

        return combinedTerms.filter { abs($0.coefficient) > 1e-15 }
    }

    // MARK: - CustomStringConvertible

    var description: String {
        if terms.isEmpty { return "Observable: 0" }

        let termStrings = terms.map { coeff, pauli in
            let sign = coeff >= 0 ? "+" : ""
            let pauliDesc = pauli.operators.isEmpty ? "I" : pauli.description
            return "\(sign)\(coeff)·\(pauliDesc)"
        }

        return "Observable: " + termStrings.joined(separator: " ")
    }
}
