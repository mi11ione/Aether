// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Transverse-field Ising model Hamiltonian constructors.
///
/// The Ising model is fundamental in condensed matter physics, describing spin-spin interactions
/// and external transverse fields. The Hamiltonian H = -Σᵢⱼ JᵢⱼZᵢZⱼ - Σᵢ hᵢXᵢ exhibits quantum
/// phase transitions and is widely used for quantum simulation benchmarks. In one dimension,
/// the model undergoes a quantum phase transition at h/J = 1 between ferromagnetic (h < J) and
/// paramagnetic (h > J) phases.
///
/// The ZZ coupling terms represent spin-spin interactions favoring aligned (J > 0, ferromagnetic)
/// or anti-aligned (J < 0, antiferromagnetic) spins. The X field terms represent a transverse
/// magnetic field that induces quantum fluctuations between spin states.
///
/// **Example:**
/// ```swift
/// let chain = IsingHamiltonian.chain(qubits: 4, J: 1.0, h: 0.5)
/// let state = QuantumState(qubits: 4)
/// let energy = chain.expectationValue(of: state)
/// ```
///
/// - SeeAlso: ``Observable``
/// - SeeAlso: ``QAOA``
/// - SeeAlso: ``VQE``
public enum IsingHamiltonian {
    /// Create 1D Ising chain with nearest-neighbor coupling.
    ///
    /// Constructs H = -J Σᵢ ZᵢZᵢ₊₁ - h Σᵢ Xᵢ for a linear chain of qubits. Each site i
    /// couples to site i+1 with strength J, and each site experiences a transverse field h.
    /// With periodic boundary conditions, site N-1 also couples to site 0.
    ///
    /// **Example:**
    /// ```swift
    /// let openChain = IsingHamiltonian.chain(qubits: 4, J: 1.0, h: 0.5, periodic: false)
    /// let periodicChain = IsingHamiltonian.chain(qubits: 4, J: 1.0, h: 0.5, periodic: true)
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of sites in the chain (1-30)
    ///   - J: Coupling strength between neighboring spins
    ///   - h: Transverse field strength applied to each site
    ///   - periodic: If true, adds coupling between last and first sites (default: false)
    /// - Returns: Observable representing the Ising chain Hamiltonian
    /// - Complexity: O(n) terms where n is the number of qubits
    /// - Precondition: `qubits` in range 1...30
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func chain(
        qubits: Int,
        J: Double,
        h: Double,
        periodic: Bool = false,
    ) -> Observable {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateMemoryLimit(qubits)

        let zzCount = periodic ? qubits : qubits - 1
        let termCount = zzCount + qubits

        let terms = PauliTerms(unsafeUninitializedCapacity: termCount) { buffer, count in
            var index = 0

            for i in 0 ..< qubits - 1 {
                buffer[index] = (coefficient: -J, pauliString: PauliString(.z(i), .z(i + 1)))
                index += 1
            }

            if periodic, qubits > 2 {
                buffer[index] = (coefficient: -J, pauliString: PauliString(.z(qubits - 1), .z(0)))
                index += 1
            }

            for i in 0 ..< qubits {
                buffer[index] = (coefficient: -h, pauliString: PauliString(.x(i)))
                index += 1
            }

            count = index
        }

        return Observable(terms: terms)
    }

    /// Create 2D Ising lattice on rectangular grid.
    ///
    /// Constructs H = -J Σ⟨ij⟩ ZᵢZⱼ - h Σᵢ Xᵢ for a rows * cols grid with nearest-neighbor
    /// interactions. Each site (r, c) couples to its right neighbor (r, c+1) and down neighbor
    /// (r+1, c) with strength J. Qubit indices are assigned as index = row * cols + col.
    /// With periodic boundary conditions, the lattice becomes a torus.
    ///
    /// **Example:**
    /// ```swift
    /// let lattice = IsingHamiltonian.lattice(rows: 2, cols: 3, J: 1.0, h: 0.5)
    /// let torusLattice = IsingHamiltonian.lattice(rows: 3, cols: 3, J: 1.0, h: 0.5, periodic: true)
    /// ```
    ///
    /// - Parameters:
    ///   - rows: Number of rows in the grid (≥1)
    ///   - cols: Number of columns in the grid (≥1)
    ///   - J: Coupling strength between neighboring spins
    ///   - h: Transverse field strength applied to each site
    ///   - periodic: If true, uses periodic boundary conditions (torus topology, default: false)
    /// - Returns: Observable representing the 2D Ising lattice Hamiltonian
    /// - Complexity: O(rows * cols) terms
    /// - Precondition: rows * cols ≤ 30
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func lattice(
        rows: Int,
        cols: Int,
        J: Double,
        h: Double,
        periodic: Bool = false,
    ) -> Observable {
        ValidationUtilities.validatePositiveInt(rows, name: "rows")
        ValidationUtilities.validatePositiveInt(cols, name: "cols")

        let qubits = rows * cols
        ValidationUtilities.validateMemoryLimit(qubits)

        var terms: PauliTerms = []

        let horizontalEdges = periodic ? rows * cols : rows * (cols - 1)
        let verticalEdges = periodic ? rows * cols : (rows - 1) * cols
        let estimatedTerms = horizontalEdges + verticalEdges + qubits
        terms.reserveCapacity(estimatedTerms)

        for row in 0 ..< rows {
            for col in 0 ..< cols {
                let site = row * cols + col

                if col < cols - 1 {
                    let rightNeighbor = row * cols + (col + 1)
                    terms.append((coefficient: -J, pauliString: PauliString(.z(site), .z(rightNeighbor))))
                } else if periodic, cols > 1 {
                    let rightNeighbor = row * cols
                    terms.append((coefficient: -J, pauliString: PauliString(.z(site), .z(rightNeighbor))))
                }

                if row < rows - 1 {
                    let downNeighbor = (row + 1) * cols + col
                    terms.append((coefficient: -J, pauliString: PauliString(.z(site), .z(downNeighbor))))
                } else if periodic, rows > 1 {
                    let downNeighbor = col
                    terms.append((coefficient: -J, pauliString: PauliString(.z(site), .z(downNeighbor))))
                }
            }
        }

        for i in 0 ..< qubits {
            terms.append((coefficient: -h, pauliString: PauliString(.x(i))))
        }

        return Observable(terms: terms)
    }

    /// Create Ising Hamiltonian from explicit coupling specification.
    ///
    /// Constructs H = -Σ(i,j,Jij) Jij·ZᵢZⱼ - Σ(i,hi) hi·Xᵢ from user-specified couplings
    /// and field strengths. This allows arbitrary graph topologies and non-uniform coupling
    /// strengths for studying disordered systems or specific problem instances.
    ///
    /// **Example:**
    /// ```swift
    /// let couplings = [(0, 1, 1.0), (1, 2, 0.5), (0, 2, 0.8)]
    /// let fields = [0: 0.3, 1: 0.5, 2: 0.3]
    /// let hamiltonian = IsingHamiltonian.fromCouplings(
    ///     zzCouplings: couplings,
    ///     xFields: fields,
    ///     qubits: 3
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - zzCouplings: Array of (qubit_i, qubit_j, J_ij) tuples specifying ZZ interactions
    ///   - xFields: Dictionary mapping qubit index to transverse field strength h_i
    ///   - qubits: Total number of qubits in the system
    /// - Returns: Observable representing the custom Ising Hamiltonian
    /// - Complexity: O(|zzCouplings| + |xFields|) terms
    /// - Precondition: All qubit indices in range [0, qubits), qubits ≤ 30
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func fromCouplings(
        zzCouplings: [(Int, Int, Double)],
        xFields: [Int: Double],
        qubits: Int,
    ) -> Observable {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateMemoryLimit(qubits)

        let termCount = zzCouplings.count + xFields.count
        var terms: PauliTerms = []
        terms.reserveCapacity(termCount)

        for (i, j, Jij) in zzCouplings {
            ValidationUtilities.validateQubitIndex(i, qubits: qubits)
            ValidationUtilities.validateQubitIndex(j, qubits: qubits)
            ValidationUtilities.validateDistinctVertices(i, j)

            let site1 = min(i, j)
            let site2 = max(i, j)
            terms.append((coefficient: -Jij, pauliString: PauliString(.z(site1), .z(site2))))
        }

        for (qubit, hi) in xFields {
            ValidationUtilities.validateQubitIndex(qubit, qubits: qubits)
            terms.append((coefficient: -hi, pauliString: PauliString(.x(qubit))))
        }

        return Observable(terms: terms)
    }

    /// Create long-range Ising model with power-law decay J(r) = J₀/r^α.
    ///
    /// Constructs H = -J₀ Σᵢ<ⱼ ZᵢZⱼ/|i-j|^α - h Σᵢ Xᵢ with all-to-all couplings that decay
    /// as a power law with distance. This model is relevant to trapped-ion quantum simulators
    /// where α ≈ 0-3 depending on the implementation. The limit α -> ∞ recovers nearest-neighbor
    /// interactions, while α = 0 gives uniform all-to-all coupling.
    ///
    /// **Example:**
    /// ```swift
    /// let longRange = IsingHamiltonian.longRange(qubits: 6, J0: 1.0, alpha: 1.5, h: 0.5)
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of sites in the chain (1-30)
    ///   - J0: Base coupling strength (coupling at distance 1)
    ///   - alpha: Power-law exponent for distance decay (α ≥ 0)
    ///   - h: Transverse field strength applied to each site
    /// - Returns: Observable representing the long-range Ising Hamiltonian
    /// - Complexity: O(n²) terms for all-to-all coupling
    /// - Precondition: `qubits` in range 1...30, alpha ≥ 0
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func longRange(
        qubits: Int,
        J0: Double,
        alpha: Double,
        h: Double,
    ) -> Observable {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateMemoryLimit(qubits)
        ValidationUtilities.validateNonNegativeDouble(alpha, name: "alpha")

        let zzCount = qubits * (qubits - 1) / 2
        let termCount = zzCount + qubits

        var terms: PauliTerms = []
        terms.reserveCapacity(termCount)

        for i in 0 ..< qubits {
            for j in (i + 1) ..< qubits {
                let distance = Double(j - i)
                let coupling = J0 / pow(distance, alpha)
                terms.append((coefficient: -coupling, pauliString: PauliString(.z(i), .z(j))))
            }
        }

        for i in 0 ..< qubits {
            terms.append((coefficient: -h, pauliString: PauliString(.x(i))))
        }

        return Observable(terms: terms)
    }

    /// Create random-field Ising model with disorder.
    ///
    /// Constructs H = -J Σᵢ ZᵢZᵢ₊₁ - Σᵢ hᵢXᵢ where hᵢ are site-dependent field strengths
    /// provided by the caller. This model is used to study localization and disorder effects
    /// in quantum many-body systems. The hValues array specifies the transverse field at each
    /// site, allowing for random or structured disorder patterns.
    ///
    /// **Example:**
    /// ```swift
    /// let disorderedFields = [0.3, 0.8, 0.2, 0.6, 0.5]
    /// let randomField = IsingHamiltonian.randomField(
    ///     qubits: 5,
    ///     J: 1.0,
    ///     hValues: disorderedFields
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of sites in the chain (1-30)
    ///   - J: Coupling strength between neighboring spins
    ///   - hValues: Array of per-site field strengths (length must equal qubits)
    ///   - periodic: If true, adds coupling between last and first sites (default: false)
    /// - Returns: Observable representing the random-field Ising Hamiltonian
    /// - Complexity: O(n) terms where n is the number of qubits
    /// - Precondition: `qubits` in range 1...30, hValues.count == qubits
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func randomField(
        qubits: Int,
        J: Double,
        hValues: [Double],
        periodic: Bool = false,
    ) -> Observable {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateMemoryLimit(qubits)
        ValidationUtilities.validateArrayCount(hValues, expected: qubits, name: "hValues")

        let zzCount = periodic ? qubits : qubits - 1
        let termCount = zzCount + qubits

        let terms = PauliTerms(unsafeUninitializedCapacity: termCount) { buffer, count in
            var index = 0

            for i in 0 ..< qubits - 1 {
                buffer[index] = (coefficient: -J, pauliString: PauliString(.z(i), .z(i + 1)))
                index += 1
            }

            if periodic, qubits > 1 {
                buffer[index] = (coefficient: -J, pauliString: PauliString(.z(qubits - 1), .z(0)))
                index += 1
            }

            for i in 0 ..< qubits {
                buffer[index] = (coefficient: -hValues[i], pauliString: PauliString(.x(i)))
                index += 1
            }

            count = index
        }

        return Observable(terms: terms)
    }
}
