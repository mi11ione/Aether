// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Fermi-Hubbard model Hamiltonian constructors with Jordan-Wigner transformation.
///
/// The Fermi-Hubbard model describes strongly correlated electrons on a lattice, capturing
/// the competition between kinetic energy (hopping) and potential energy (on-site interaction).
/// The Hamiltonian H = -t Σ⟨ij⟩σ (c†ᵢσcⱼσ + h.c.) + U Σᵢ nᵢ↑nᵢ↓ consists of nearest-neighbor
/// hopping with amplitude t and on-site Coulomb repulsion U. This model exhibits rich physics
/// including Mott insulator transitions, antiferromagnetic ordering, and is believed to capture
/// essential features of high-temperature superconductivity.
///
/// The Jordan-Wigner transformation maps fermionic creation/annihilation operators to Pauli
/// strings, enabling simulation on qubit-based quantum computers. For spin-1/2 fermions on N
/// sites, 2N qubits are required (N for spin-up, N for spin-down). The transformation preserves
/// fermionic anticommutation relations through a string of Z operators.
///
/// **Example:**
/// ```swift
/// let hubbard = HubbardHamiltonian.chain(sites: 4, t: 1.0, U: 4.0)
/// let state = QuantumState(qubits: hubbard.qubits)
/// let energy = hubbard.observable.expectationValue(of: state)
/// ```
///
/// - SeeAlso: ``Observable``
/// - SeeAlso: ``IsingHamiltonian``
/// - SeeAlso: ``VQE``
public enum HubbardHamiltonian {
    /// Result containing Observable and qubit mapping information for Hubbard model.
    ///
    /// Encapsulates the Pauli representation of the Fermi-Hubbard Hamiltonian along with
    /// physical parameters and qubit layout. The qubit mapping places spin-up electrons
    /// at indices 0 to sites-1 and spin-down electrons at indices sites to 2*sites-1.
    ///
    /// **Example:**
    /// ```swift
    /// let model = HubbardHamiltonian.chain(sites: 2, t: 1.0, U: 4.0)
    /// print(model.qubits)     // 4 (2 sites * 2 spins)
    /// print(model.sites)      // 2
    /// print(model.t)          // 1.0 (hopping)
    /// print(model.U)          // 4.0 (interaction)
    /// ```
    @frozen
    public struct HubbardModel: Sendable {
        /// Pauli string representation of the Hamiltonian.
        public let observable: Observable

        /// Total number of qubits (2 * sites for spin-up and spin-down).
        public let qubits: Int

        /// Number of physical lattice sites.
        public let sites: Int

        /// Hopping parameter t (kinetic energy scale).
        public let t: Double

        /// On-site interaction parameter U (Coulomb repulsion).
        public let U: Double
    }

    /// Create 1D Hubbard chain with nearest-neighbor hopping.
    ///
    /// Constructs H = -t Σᵢσ (c†ᵢσcᵢ₊₁σ + h.c.) + U Σᵢ nᵢ↑nᵢ↓ for a linear chain of sites.
    /// Each site i couples to site i+1 with hopping amplitude t for both spin species.
    /// With periodic boundary conditions, site N-1 also couples to site 0.
    ///
    /// The qubit layout uses 2N qubits: qubits 0 to N-1 for spin-up electrons and
    /// qubits N to 2N-1 for spin-down electrons.
    ///
    /// **Example:**
    /// ```swift
    /// let openChain = HubbardHamiltonian.chain(sites: 4, t: 1.0, U: 4.0)
    /// let periodicChain = HubbardHamiltonian.chain(sites: 4, t: 1.0, U: 4.0, periodic: true)
    /// let state = QuantumState(qubits: openChain.qubits)
    /// let energy = openChain.observable.expectationValue(of: state)
    /// ```
    ///
    /// - Parameters:
    ///   - sites: Number of lattice sites (1-15)
    ///   - t: Hopping amplitude between neighboring sites
    ///   - U: On-site Coulomb interaction strength
    ///   - periodic: If true, adds hopping between last and first sites (default: false)
    /// - Returns: HubbardModel containing the Hamiltonian Observable
    /// - Complexity: O(sites) terms for hopping, O(sites) terms for interaction
    /// - Precondition: `sites` in range 1...15 (2*sites ≤ 30 qubits)
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func chain(
        sites: Int,
        t: Double,
        U: Double,
        periodic: Bool = false,
    ) -> HubbardModel {
        ValidationUtilities.validatePositiveInt(sites, name: "sites")
        ValidationUtilities.validateMemoryLimit(2 * sites)

        var hoppings: [(Int, Int, Double)] = []
        let hoppingCount = periodic ? sites : sites - 1
        hoppings.reserveCapacity(hoppingCount)

        for i in 0 ..< sites - 1 {
            hoppings.append((i, i + 1, t))
        }

        if periodic, sites > 1 {
            hoppings.append((sites - 1, 0, t))
        }

        return fromHoppings(hoppings: hoppings, U: U, sites: sites)
    }

    /// Create 2D Hubbard lattice on rectangular grid.
    ///
    /// Constructs H = -t Σ⟨ij⟩σ (c†ᵢσcⱼσ + h.c.) + U Σᵢ nᵢ↑nᵢ↓ for a rows * cols grid
    /// with nearest-neighbor interactions. Each site (r, c) couples to its right neighbor
    /// (r, c+1) and down neighbor (r+1, c) with hopping amplitude t for both spin species.
    /// Site indices are assigned as index = row * cols + col.
    ///
    /// With periodic boundary conditions, the lattice becomes a torus with additional
    /// couplings across the boundaries.
    ///
    /// **Example:**
    /// ```swift
    /// let lattice = HubbardHamiltonian.lattice(rows: 2, cols: 2, t: 1.0, U: 4.0)
    /// let torus = HubbardHamiltonian.lattice(rows: 2, cols: 2, t: 1.0, U: 4.0, periodic: true)
    /// let state = QuantumState(qubits: lattice.qubits)
    /// let energy = lattice.observable.expectationValue(of: state)
    /// ```
    ///
    /// - Parameters:
    ///   - rows: Number of rows in the grid (≥1)
    ///   - cols: Number of columns in the grid (≥1)
    ///   - t: Hopping amplitude between neighboring sites
    ///   - U: On-site Coulomb interaction strength
    ///   - periodic: If true, uses periodic boundary conditions (torus topology, default: false)
    /// - Returns: HubbardModel containing the Hamiltonian Observable
    /// - Complexity: O(rows * cols) terms for hopping and interaction
    /// - Precondition: 2 * rows * cols ≤ 30
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func lattice(
        rows: Int,
        cols: Int,
        t: Double,
        U: Double,
        periodic: Bool = false,
    ) -> HubbardModel {
        ValidationUtilities.validatePositiveInt(rows, name: "rows")
        ValidationUtilities.validatePositiveInt(cols, name: "cols")

        let sites = rows * cols
        ValidationUtilities.validateMemoryLimit(2 * sites)

        var hoppings: [(Int, Int, Double)] = []
        let horizontalEdges = periodic ? rows * cols : rows * (cols - 1)
        let verticalEdges = periodic ? rows * cols : (rows - 1) * cols
        hoppings.reserveCapacity(horizontalEdges + verticalEdges)

        for row in 0 ..< rows {
            for col in 0 ..< cols {
                let site = row * cols + col

                if col < cols - 1 {
                    let rightNeighbor = row * cols + (col + 1)
                    hoppings.append((site, rightNeighbor, t))
                } else if periodic, cols > 1 {
                    let rightNeighbor = row * cols
                    hoppings.append((site, rightNeighbor, t))
                }

                if row < rows - 1 {
                    let downNeighbor = (row + 1) * cols + col
                    hoppings.append((site, downNeighbor, t))
                } else if periodic, rows > 1 {
                    let downNeighbor = col
                    hoppings.append((site, downNeighbor, t))
                }
            }
        }

        return fromHoppings(hoppings: hoppings, U: U, sites: sites)
    }

    /// Create Hubbard model from explicit hopping specification.
    ///
    /// Constructs H = -Σ(i,j,tij) tij·(c†ᵢσcⱼσ + h.c.) + U Σᵢ nᵢ↑nᵢ↓ from user-specified
    /// hopping terms. This allows arbitrary graph topologies and non-uniform hopping
    /// strengths for studying disordered systems, molecular structures, or custom lattices.
    ///
    /// The Jordan-Wigner transformation is applied to convert fermionic operators to
    /// Pauli strings, with the standard qubit ordering: spin-up at qubits 0 to sites-1,
    /// spin-down at qubits sites to 2*sites-1.
    ///
    /// **Example:**
    /// ```swift
    /// let triangleHoppings = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
    /// let model = HubbardHamiltonian.fromHoppings(
    ///     hoppings: triangleHoppings,
    ///     U: 4.0,
    ///     sites: 3
    /// )
    /// let state = QuantumState(qubits: model.qubits)
    /// let energy = model.observable.expectationValue(of: state)
    /// ```
    ///
    /// - Parameters:
    ///   - hoppings: Array of (site_i, site_j, t_ij) tuples specifying hopping terms
    ///   - U: On-site Coulomb interaction strength
    ///   - sites: Total number of lattice sites
    /// - Returns: HubbardModel containing the Hamiltonian Observable
    /// - Complexity: O(|hoppings| + sites) Pauli terms
    /// - Precondition: All site indices in range [0, sites), 2*sites ≤ 30
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func fromHoppings(
        hoppings: [(Int, Int, Double)],
        U: Double,
        sites: Int,
    ) -> HubbardModel {
        ValidationUtilities.validatePositiveInt(sites, name: "sites")
        ValidationUtilities.validateMemoryLimit(2 * sites)

        let qubits = 2 * sites
        var terms: PauliTerms = []

        let estimatedTerms = 4 * hoppings.count * 2 + 4 * sites
        terms.reserveCapacity(estimatedTerms)

        for (siteI, siteJ, tij) in hoppings {
            ValidationUtilities.validateQubitIndex(siteI, qubits: sites)
            ValidationUtilities.validateQubitIndex(siteJ, qubits: sites)
            ValidationUtilities.validateDistinctVertices(siteI, siteJ)

            let minSite = min(siteI, siteJ)
            let maxSite = max(siteI, siteJ)

            let upFrom = minSite
            let upTo = maxSite
            let upTerms = jordanWignerHopping(from: upFrom, to: upTo, coefficient: -tij)
            terms.append(contentsOf: upTerms)

            let downFrom = sites + minSite
            let downTo = sites + maxSite
            let downTerms = jordanWignerHopping(from: downFrom, to: downTo, coefficient: -tij)
            terms.append(contentsOf: downTerms)
        }

        if abs(U) > 1e-15 {
            for site in 0 ..< sites {
                let upQubit = site
                let downQubit = sites + site
                let interactionTerms = jordanWignerInteraction(upQubit: upQubit, downQubit: downQubit, U: U)
                terms.append(contentsOf: interactionTerms)
            }
        }

        let nonZeroTerms = terms.filter { abs($0.0) > 1e-15 }
        let observable = Observable(terms: nonZeroTerms)
        let avgT = hoppings.isEmpty ? 0.0 : hoppings.reduce(0.0) { $0 + $1.2 } / Double(hoppings.count)

        return HubbardModel(
            observable: observable,
            qubits: qubits,
            sites: sites,
            t: avgT,
            U: U,
        )
    }

    /// Convert fermionic hopping term to Pauli terms via Jordan-Wigner transformation.
    ///
    /// Transforms c†ᵢcⱼ + c†ⱼcᵢ (for i < j) to Pauli strings using the Jordan-Wigner mapping:
    /// c†ᵢcⱼ + h.c. = (1/2)(Xᵢ · Zᵢ₊₁ · ... · Zⱼ₋₁ · Xⱼ + Yᵢ · Zᵢ₊₁ · ... · Zⱼ₋₁ · Yⱼ)
    ///
    /// The string of Z operators between sites i and j (exclusive) ensures proper
    /// fermionic anticommutation relations are preserved in the qubit representation.
    ///
    /// **Example:**
    /// ```swift
    /// let terms = HubbardHamiltonian.jordanWignerHopping(from: 0, to: 2, coefficient: -1.0)
    /// // Returns terms for -1.0 * (c†₀c₂ + c†₂c₀)
    /// // = -0.5 * X₀Z₁X₂ - 0.5 * Y₀Z₁Y₂
    /// ```
    ///
    /// - Parameters:
    ///   - from: Source qubit index (must be < to)
    ///   - to: Target qubit index (must be > from)
    ///   - coefficient: Hopping coefficient (typically -t)
    /// - Returns: Array of (coefficient, PauliString) pairs representing the hopping term
    /// - Complexity: O(to - from) operators per Pauli string
    @_optimize(speed)
    @_effects(readonly)
    static func jordanWignerHopping(
        from: Int,
        to: Int,
        coefficient: Double,
    ) -> [(Double, PauliString)] {
        let halfCoeff = coefficient / 2.0

        var xxOperators: [PauliOperator] = []
        var yyOperators: [PauliOperator] = []
        xxOperators.reserveCapacity(to - from + 1)
        yyOperators.reserveCapacity(to - from + 1)

        xxOperators.append(.x(from))
        yyOperators.append(.y(from))

        for k in (from + 1) ..< to {
            xxOperators.append(.z(k))
            yyOperators.append(.z(k))
        }

        xxOperators.append(.x(to))
        yyOperators.append(.y(to))

        return [
            (halfCoeff, PauliString(xxOperators)),
            (halfCoeff, PauliString(yyOperators)),
        ]
    }

    /// Convert density-density interaction nᵢ↑nᵢ↓ to Pauli terms.
    ///
    /// Transforms the on-site interaction nᵢ↑nᵢ↓ using the number operator mapping:
    /// nᵢ = (I - Zᵢ)/2, which gives:
    /// nᵢ↑nᵢ↓ = (I - Zᵢ↑)(I - Zᵢ↓)/4 = (I - Zᵢ↑ - Zᵢ↓ + Zᵢ↑Zᵢ↓)/4
    ///
    /// This represents the Coulomb repulsion energy when both spin-up and spin-down
    /// electrons occupy the same lattice site.
    ///
    /// **Example:**
    /// ```swift
    /// let terms = HubbardHamiltonian.jordanWignerInteraction(upQubit: 0, downQubit: 2, U: 4.0)
    /// // Returns terms for 4.0 * n₀↑n₂↓
    /// // = 1.0*I - 1.0*Z₀ - 1.0*Z₂ + 1.0*Z₀Z₂
    /// ```
    ///
    /// - Parameters:
    ///   - upQubit: Qubit index for spin-up electron at this site
    ///   - downQubit: Qubit index for spin-down electron at this site
    ///   - U: On-site interaction strength
    /// - Returns: Array of (coefficient, PauliString) pairs representing the interaction
    /// - Complexity: O(1) - always produces 4 terms
    @_optimize(speed)
    @_effects(readonly)
    static func jordanWignerInteraction(
        upQubit: Int,
        downQubit: Int,
        U: Double,
    ) -> [(Double, PauliString)] {
        let quarterU = U / 4.0

        return [
            (quarterU, PauliString([])),
            (-quarterU, PauliString(.z(upQubit))),
            (-quarterU, PauliString(.z(downQubit))),
            (quarterU, PauliString(.z(upQubit), .z(downQubit))),
        ]
    }
}
