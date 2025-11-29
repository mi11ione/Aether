// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Mixer Hamiltonian constructors for QAOA
///
/// Provides standard mixer Hamiltonians used in QAOA algorithms. The mixer
/// Hamiltonian H_m drives transitions between computational basis states,
/// enabling exploration of the solution space.
///
/// **Standard X Mixer:**
/// H_m = Σᵢ Xᵢ (sum of X operators on all qubits)
///
/// This is the **default mixer** for unconstrained combinatorial optimization.
/// Generates transitions between all 2^n basis states with equal amplitudes.
///
/// **Mathematical properties:**
/// - Hermitian: H_m† = H_m (real eigenvalues)
/// - Non-commuting with Z-based cost Hamiltonians (enables optimization)
/// - exp(-iβH_m) implemented as product of Rx(2β) gates
///
/// **Usage in QAOA:**
/// The mixer layer applies exp(-iβH_m) which creates superpositions:
/// - Starting from |+⟩^⊗n (equal superposition)
/// - β parameter controls mixing strength
/// - Alternates with problem Hamiltonian exp(-iγH_p)
///
/// **Example - Standard QAOA with X mixer:**
/// ```swift
/// let numQubits = 4
/// let costHamiltonian = MaxCut.hamiltonian(edges: [(0,1), (1,2), (2,3)])
/// let mixerHamiltonian = MixerHamiltonian.xMixer(numQubits: numQubits)
///
/// let qaoa = await QAOA(
///     costHamiltonian: costHamiltonian,
///     mixerHamiltonian: mixerHamiltonian,
///     numQubits: numQubits,
///     depth: 2
/// )
/// ```
///
/// **Future extensions:**
/// - XY mixer: Preserves Hamming weight (constrained optimization)
/// - Custom mixers: Problem-specific transition operators
/// - Warm-start mixers: Bias toward known good solutions
@frozen
public struct MixerHamiltonian {
    /// Create standard X mixer Hamiltonian: H_m = Σᵢ Xᵢ
    ///
    /// Generates mixer that creates uniform superpositions across all basis states.
    /// This is the **default choice** for unconstrained QAOA problems.
    ///
    /// **Implementation:**
    /// - One X (Pauli-X) term per qubit
    /// - Coefficient +1.0 for each term (standard convention)
    /// - Commutes with itself: [H_m, H_m] = 0
    /// - Non-commuting with typical cost Hamiltonians (Z-based)
    ///
    /// **Physical interpretation:**
    /// - Each X term flips the corresponding qubit
    /// - exp(-iβ·X) = Rx(2β) rotation around X axis
    /// - Creates superposition when applied to computational basis
    ///
    /// **Complexity:** O(n) terms, O(n) gates in circuit
    ///
    /// - Parameter numQubits: Number of qubits in system (1-30)
    /// - Returns: Observable representing Σᵢ Xᵢ
    ///
    /// Example:
    /// ```swift
    /// // 3-qubit X mixer: H_m = X₀ + X₁ + X₂
    /// let mixer = MixerHamiltonian.xMixer(numQubits: 3)
    ///
    /// // Verify structure
    /// print(mixer.terms.count)  // 3 terms
    /// // Terms: [(1.0, X₀), (1.0, X₁), (1.0, X₂)]
    ///
    /// // Use in QAOA
    /// let qaoa = await QAOA(
    ///     costHamiltonian: costHamiltonian,
    ///     mixerHamiltonian: mixer,
    ///     numQubits: 3,
    ///     depth: 2
    /// )
    /// ```
    @_optimize(speed)
    @_eagerMove
    public static func xMixer(numQubits: Int) -> Observable {
        ValidationUtilities.validatePositiveQubits(numQubits)
        ValidationUtilities.validateMemoryLimit(numQubits)

        let terms = PauliTerms(unsafeUninitializedCapacity: numQubits) { buffer, count in
            for qubit in 0 ..< numQubits {
                buffer[qubit] = (coefficient: 1.0, pauliString: PauliString(.x(qubit)))
            }
            count = numQubits
        }

        return Observable(terms: terms)
    }
}
