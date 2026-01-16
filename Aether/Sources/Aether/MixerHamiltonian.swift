// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Mixer Hamiltonian constructors for QAOA.
///
/// The mixer Hamiltonian H_m drives transitions between computational basis states,
/// enabling exploration of the solution space. The standard X mixer H_m = Σᵢ Xᵢ
/// is the default choice for unconstrained combinatorial optimization, generating
/// transitions between all 2ⁿ basis states. The mixer layer exp(-iβH_m) alternates
/// with the cost layer exp(-iγH_p) in QAOA circuits.
///
/// **Example:**
/// ```swift
/// let mixer = MixerHamiltonian.x(qubits: 4)
/// let cost = MaxCut.hamiltonian(edges: [(0, 1), (1, 2), (2, 0)])
/// let qaoa = await QAOA(cost: cost, mixer: mixer, qubits: 3, depth: 2)
/// ```
///
/// - SeeAlso: ``QAOA``
/// - SeeAlso: ``MaxCut``
/// - SeeAlso: ``Observable``
public enum MixerHamiltonian {
    private static let xCoefficient = 1.0

    /// Creates the standard X mixer Hamiltonian H_m = Σᵢ Xᵢ.
    ///
    /// Each qubit contributes one X term with coefficient +1.0. This mixer does not
    /// commute with Z-based cost Hamiltonians, enabling QAOA optimization. The mixer
    /// layer exp(-iβH_m) is implemented as a product of Rx(2β) rotations.
    ///
    /// **Example:**
    /// ```swift
    /// let mixer = MixerHamiltonian.x(qubits: 4)
    /// ```
    ///
    /// - Parameter qubits: Number of qubits in system (1-30)
    /// - Returns: Observable with one X term per qubit
    /// - Complexity: O(n) terms and gates
    /// - Precondition: `qubits` in range 1...30
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func x(qubits: Int) -> Observable {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateMemoryLimit(qubits)

        let terms = PauliTerms(unsafeUninitializedCapacity: qubits) { buffer, count in
            for qubit in 0 ..< qubits {
                buffer[qubit] = (coefficient: xCoefficient, pauliString: PauliString(.x(qubit)))
            }
            count = qubits
        }

        return Observable(terms: terms)
    }
}
