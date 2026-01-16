// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Measurement optimization for efficient Hamiltonian expectation values.
///
/// Provides advanced grouping techniques that reduce the number of quantum circuits required
/// to measure Hamiltonian expectation values. Integrates qubit-wise commuting (QWC) grouping
/// and unitary partitioning with actor-based caching and automatic strategy selection.
///
/// QWC grouping reduces typical molecular Hamiltonians from thousands of terms to tens or
/// hundreds of measurement groups by identifying Pauli strings that can be measured
/// simultaneously. Unitary partitioning extends this further by finding optimal unitary
/// transformations that diagonalize multiple non-commuting terms together.
///
/// Automatic strategy selection chooses between term-by-term measurement,
/// QWC grouping, or unitary partitioning based on problem size.
///
/// - SeeAlso: ``QWCGrouper``
/// - SeeAlso: ``UnitaryPartitioner``
/// - SeeAlso: ``ShotAllocator``
///
/// **Example:**
/// ```swift
/// let hamiltonian = Observable(terms: molecularTerms)
/// let groups = await hamiltonian.qwcGroups()
/// print("Reduced \(hamiltonian.terms.count) terms to \(groups.count) groups")
/// ```
public extension Observable {
    /// Thread-safe cache for expensive grouping computations.
    ///
    /// Uses Swift actor isolation to provide automatic serialization of cache access
    /// without manual lock management. All methods are isolated to the actor, ensuring
    /// thread-safe reads and writes.
    private actor GroupingCache {
        private struct CacheEntry<T> {
            let terms: PauliTerms
            let value: T
        }

        private var qwcGroups: [Int: CacheEntry<[QWCGroup]>] = [:]
        private var unitaryPartitions: [Int: CacheEntry<[UnitaryPartition]>] = [:]

        /// Retrieve cached QWC groups if hash matches and terms are equal.
        func getQWCGroups(
            hash: Int,
            terms: PauliTerms,
        ) -> [QWCGroup]? {
            guard let entry = qwcGroups[hash] else { return nil }
            guard termsEqual(entry.terms, terms) else { return nil }
            return entry.value
        }

        /// Store QWC groups in cache keyed by hash.
        func setQWCGroups(
            hash: Int,
            terms: PauliTerms,
            groups: [QWCGroup],
        ) {
            qwcGroups[hash] = CacheEntry(terms: terms, value: groups)
        }

        /// Retrieve cached unitary partitions if hash matches and terms are equal.
        func getUnitaryPartitions(
            hash: Int,
            terms: PauliTerms,
        ) -> [UnitaryPartition]? {
            guard let entry = unitaryPartitions[hash] else { return nil }
            guard termsEqual(entry.terms, terms) else { return nil }
            return entry.value
        }

        /// Store unitary partitions in cache keyed by hash.
        func setUnitaryPartitions(
            hash: Int,
            terms: PauliTerms,
            partitions: [UnitaryPartition],
        ) {
            unitaryPartitions[hash] = CacheEntry(terms: terms, value: partitions)
        }

        /// Remove all cached entries.
        func clear() {
            qwcGroups.removeAll()
            unitaryPartitions.removeAll()
        }

        /// Check if two PauliTerms arrays are element-wise equal.
        @_effects(readonly)
        private func termsEqual(
            _ lhs: PauliTerms,
            _ rhs: PauliTerms,
        ) -> Bool {
            guard lhs.count == rhs.count else { return false }
            for i in 0 ..< lhs.count {
                if lhs[i].coefficient != rhs[i].coefficient { return false }
                if lhs[i].pauliString != rhs[i].pauliString { return false }
            }
            return true
        }
    }

    private static let cache = GroupingCache()

    // MARK: - Cache Key Generation

    /// Compute stable hash combining coefficient bit patterns and Pauli operator hashes.
    @_effects(readonly)
    private func termsHash() -> Int {
        var hasher = 0
        for i in 0 ..< terms.count {
            let bits = terms[i].coefficient.bitPattern
            hasher ^= Int(truncatingIfNeeded: bits) &* 31
            hasher ^= terms[i].pauliString.hashValue &* 17
        }
        return hasher
    }

    // MARK: - Cached Groupings

    /// Compute or retrieve QWC groups from cache.
    ///
    /// Groups are cached based on terms hash and recomputed only when the hash changes
    /// or a collision is detected by comparing actual terms. Actor-based caching ensures
    /// thread-safe access without manual lock management.
    ///
    /// - Returns: Array of QWC groups where terms within each group commute qubit-wise
    /// - Complexity: O(k²) where k is the number of terms (graph coloring), or O(1) if cached
    ///
    /// **Example:**
    /// ```swift
    /// let groups = await hamiltonian.qwcGroups()
    /// for group in groups {
    ///     let basis = group.measurementBasis
    /// }
    /// ```
    @_eagerMove
    func qwcGroups() async -> [QWCGroup] {
        let hash: Int = termsHash()

        if let cached = await Self.cache.getQWCGroups(hash: hash, terms: terms) {
            return cached
        }

        let groups: [QWCGroup] = QWCGrouper.group(terms)
        await Self.cache.setQWCGroups(hash: hash, terms: terms, groups: groups)
        return groups
    }

    /// Compute or retrieve unitary partitions from cache.
    ///
    /// Uses optimal unitary transformations to group terms more efficiently than QWC grouping.
    /// This is computationally expensive, so caching is essential for practical use. Hash
    /// collisions are detected by comparing actual terms.
    ///
    /// - Parameter config: Partitioner configuration controlling depth and convergence
    /// - Returns: Array of unitary partitions where each partition defines a measurement circuit
    /// - Complexity: O(g² · iter · depth · 2^(2n)) where g is number of groups, or O(1) if cached
    ///
    /// **Example:**
    /// ```swift
    /// let partitions = await hamiltonian.unitaryPartitions()
    /// print("Reduced to \(partitions.count) measurement circuits")
    /// ```
    @_eagerMove
    func unitaryPartitions(config: UnitaryPartitioner.Config = .init()) async -> [UnitaryPartition] {
        let hash = termsHash()

        if let cached = await Self.cache.getUnitaryPartitions(hash: hash, terms: terms) {
            return cached
        }

        let partitioner = UnitaryPartitioner(config: config)
        let partitions: [UnitaryPartition] = partitioner.partition(terms: terms)
        await Self.cache.setUnitaryPartitions(hash: hash, terms: terms, partitions: partitions)
        return partitions
    }

    /// Clear all cached groupings.
    ///
    /// Useful for testing or memory management when working with many different Hamiltonians.
    ///
    /// **Example:**
    /// ```swift
    /// await Observable.clearGroupingCaches()
    /// ```
    static func clearGroupingCaches() async { await cache.clear() }

    // MARK: - Measurement Strategies

    /// Strategy for measuring Hamiltonian expectation values.
    @frozen
    enum MeasurementStrategy {
        /// Measure each term independently without optimization.
        case termByTerm

        /// Group qubit-wise commuting terms for simultaneous measurement.
        case qwcGrouping

        /// Use variational unitary optimization for maximal grouping.
        case unitaryPartitioning(config: UnitaryPartitioner.Config)

        /// Automatically select best strategy based on problem size.
        case automatic
    }

    /// Compute effective number of measurement circuits for a strategy.
    ///
    /// - Parameter strategy: Measurement strategy to evaluate
    /// - Returns: Number of quantum circuits required
    /// - Complexity: O(1) for termByTerm, O(k²) for qwcGrouping, O(g² · iter · depth · 2^(2n)) for unitaryPartitioning
    ///
    /// **Example:**
    /// ```swift
    /// let circuitCount = await hamiltonian.measureCircuitCount(for: .qwcGrouping)
    /// print("Need \(circuitCount) circuits")
    /// ```
    func measureCircuitCount(for strategy: MeasurementStrategy) async -> Int {
        switch strategy {
        case .termByTerm: terms.count
        case .qwcGrouping: await qwcGroups().count
        case let .unitaryPartitioning(config): await unitaryPartitions(config: config).count
        case .automatic: await measureCircuitCount(for: selectOptimalStrategy())
        }
    }

    /// Select optimal measurement strategy based on Hamiltonian size.
    @_effects(readonly)
    private func selectOptimalStrategy() -> MeasurementStrategy {
        let numTerms: Int = terms.count

        if numTerms < 20 { return .termByTerm }
        if numTerms < 500 { return .qwcGrouping }

        return .unitaryPartitioning(config: .init())
    }

    // MARK: - Shot Allocation

    /// Allocate measurement shots optimally across terms.
    ///
    /// Distributes total shots using variance-weighted allocation where high-variance terms
    /// receive proportionally more shots to minimize total estimation error. Optionally uses
    /// a test state to estimate variances accurately.
    ///
    /// - Parameters:
    ///   - totalShots: Total measurement shots available
    ///   - minShotsPerTerm: Min shot configuration
    ///   - state: Quantum state for variance estimation (optional)
    /// - Returns: Shot allocation mapping term indices to shot counts
    /// - Complexity: O(k) where k is the number of terms
    /// - SeeAlso: ``ShotAllocator``
    ///
    /// **Example:**
    /// ```swift
    /// let allocation = hamiltonian.allocateShots(
    ///     totalShots: 10000,
    ///     state: currentState
    /// )
    /// ```
    @_eagerMove
    func allocateShots(
        totalShots: Int,
        minShotsPerTerm: Int = 10,
        state: QuantumState? = nil,
    ) -> [Int: Int] {
        let allocator = ShotAllocator(minShotsPerTerm: minShotsPerTerm)
        return allocator.allocate(for: terms, totalShots: totalShots, state: state)
    }

    /// Allocate measurement shots optimally across QWC groups.
    ///
    /// Combines QWC grouping with variance-weighted shot allocation to minimize total shots
    /// required for target accuracy. Groups are treated as atomic measurement units with
    /// variance determined from constituent terms.
    ///
    /// - Parameters:
    ///   - totalShots: Total measurement shots available
    ///   - minShotsPerTerm: Minimum shots per term (avoid zero allocation)
    ///   - state: Quantum state for variance estimation (optional)
    /// - Returns: Shot allocation mapping group indices to shot counts
    /// - Complexity: O(k² + g) where k is terms and g is groups
    ///
    /// **Example:**
    /// ```swift
    /// let allocation = await hamiltonian.allocateShotsForGroups(
    ///     totalShots: 10000,
    ///     state: currentState
    /// )
    /// ```
    @_eagerMove
    func allocateShotsForGroups(
        totalShots: Int,
        minShotsPerTerm: Int = 10,
        state: QuantumState? = nil,
    ) async -> [Int: Int] {
        let allocator = ShotAllocator(minShotsPerTerm: minShotsPerTerm)
        let groups: [QWCGroup] = await qwcGroups()
        return allocator.allocate(forGroups: groups, totalShots: totalShots, state: state)
    }

    // MARK: - Comprehensive Statistics

    /// Statistics describing measurement optimization effectiveness.
    @frozen
    struct MeasurementOptimizationStats: CustomStringConvertible {
        /// Number of Hamiltonian terms.
        public let numTerms: Int

        /// Number of QWC groups.
        public let numQWCGroups: Int

        /// Number of unitary partitions (if computed).
        public let numUnitaryPartitions: Int?

        /// Reduction factor from QWC grouping.
        public let qwcReduction: Double

        /// Reduction factor from unitary partitioning (if computed).
        public let unitaryReduction: Double?

        /// Estimated speedup from QWC grouping.
        public let estimatedSpeedupQWC: Double

        /// Estimated speedup from unitary partitioning (if computed).
        public let estimatedSpeedupUnitary: Double?

        public var description: String {
            var text = """
            Measurement Optimization Statistics:
            - Hamiltonian terms: \(numTerms)
            - QWC groups: \(numQWCGroups)
            - QWC reduction: \(String(format: "%.1f", qwcReduction))x
            - Estimated speedup (QWC): \(String(format: "%.1f", estimatedSpeedupQWC))x
            """

            if let numPartitions = numUnitaryPartitions,
               let unitaryRed = unitaryReduction,
               let unitarySpeedup = estimatedSpeedupUnitary
            {
                text += """

                - Unitary partitions: \(numPartitions)
                - Unitary reduction: \(String(format: "%.1f", unitaryRed))x
                - Estimated speedup (Unitary): \(String(format: "%.1f", unitarySpeedup))x
                """
            }

            return text
        }
    }

    /// Compute comprehensive measurement optimization statistics.
    ///
    /// - Parameter includeUnitary: Whether to compute expensive unitary partitioning statistics
    /// - Returns: Statistics structure with reduction factors and speedup estimates
    /// - Complexity: O(k²) for QWC only, O(k² + g² · iter · depth · 2^(2n)) if includeUnitary
    ///
    /// **Example:**
    /// ```swift
    /// let stats = await hamiltonian.optimizationStatistics()
    /// print("QWC reduces circuits by \(stats.qwcReduction)x")
    /// ```
    @_eagerMove
    func optimizationStatistics(includeUnitary: Bool = false) async -> MeasurementOptimizationStats {
        let numTerms: Int = terms.count
        let groups: [QWCGroup] = await qwcGroups()
        let numGroups: Int = groups.count

        let qwcReduction: Double = numTerms > 0 ? Double(numTerms) / Double(max(numGroups, 1)) : 1.0
        let qwcSpeedup: Double = qwcReduction

        var numPartitions: Int?
        var unitaryReduction: Double?
        var unitarySpeedup: Double?

        if includeUnitary {
            let partitions = await unitaryPartitions()
            numPartitions = partitions.count
            unitaryReduction = numTerms > 0 ? Double(numTerms) / Double(max(partitions.count, 1)) : 1.0
            unitarySpeedup = unitaryReduction
        }

        return MeasurementOptimizationStats(
            numTerms: numTerms,
            numQWCGroups: numGroups,
            numUnitaryPartitions: numPartitions,
            qwcReduction: qwcReduction,
            unitaryReduction: unitaryReduction,
            estimatedSpeedupQWC: qwcSpeedup,
            estimatedSpeedupUnitary: unitarySpeedup,
        )
    }
}
