// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Maps term or group index to number of measurement shots allocated.
///
/// Used in shot allocation to specify how many measurement shots should be
/// dedicated to each Hamiltonian term or measurement group.
public typealias ShotAllocation = [Int: Int]

/// Optimal allocation of measurement shots for Hamiltonian expectation values.
///
/// For shot-based measurements (real hardware or shot-based simulation), the variance
/// of the Hamiltonian expectation value is:
///
/// Var(⟨H⟩) = Σᵢ cᵢ² Var(⟨Pᵢ⟩) / nᵢ
///
/// where nᵢ is the number of shots allocated to term i.
///
/// To minimize total shots N for target variance σ², the optimal allocation is:
///
/// nᵢ = N × (|cᵢ| √Var(Pᵢ)) / (Σⱼ |cⱼ| √Var(Pⱼ))
///
/// This typically reduces shot requirements by 2-10× compared to uniform allocation.
public struct ShotAllocator {
    // MARK: - Configuration

    public struct Config: Sendable {
        /// Minimum shots per term (avoid zero allocation)
        public let minShotsPerTerm: Int

        /// Whether to round to nearest integer or use fractional shots
        public let roundToInteger: Bool

        public static let `default` = Config(
            minShotsPerTerm: 10,
            roundToInteger: true
        )

        public init(minShotsPerTerm: Int, roundToInteger: Bool) {
            self.minShotsPerTerm = minShotsPerTerm
            self.roundToInteger = roundToInteger
        }
    }

    public let config: Config

    public init(config: Config = .default) { self.config = config }

    // MARK: - Shot Allocation

    /// Allocate shots optimally across Hamiltonian terms.
    ///
    /// - Parameters:
    ///   - terms: Array of (coefficient, PauliString) pairs
    ///   - totalShots: Total number of shots available
    ///   - state: Quantum state for variance estimation (optional)
    /// - Returns: Dictionary mapping term index to number of shots
    ///
    /// Example:
    /// ```swift
    /// let allocator = ShotAllocator()
    /// let allocation = allocator.allocate(
    ///     terms: hamiltonian.terms,
    ///     totalShots: 10000,
    ///     state: currentState
    /// )
    /// // allocation[0] = 250  (high-weight term gets more shots)
    /// // allocation[1] = 100  (low-weight term gets fewer shots)
    /// ```
    public func allocate(
        terms: PauliTerms,
        totalShots: Int,
        state: QuantumState? = nil
    ) -> ShotAllocation {
        guard !terms.isEmpty else { return [:] }
        guard totalShots > 0 else { return [:] }

        let variances: [Double] = if let state {
            batchEstimateVariances(terms: terms, state: state)
        } else {
            Array(repeating: 1.0, count: terms.count)
        }

        let weights: [Double] = zip(terms, variances).map { term, variance in
            abs(term.coefficient) * sqrt(variance)
        }

        let totalWeight: Double = weights.reduce(0.0, +)

        guard totalWeight > 0 else {
            return uniformAllocation(numTerms: terms.count, totalShots: totalShots)
        }

        var allocation: ShotAllocation = [:]
        var shotsRemaining: Int = totalShots

        let maxPossibleMin: Int = totalShots / max(terms.count, 1)
        let effectiveMinShots: Int = min(config.minShotsPerTerm, maxPossibleMin)

        for (index, weight) in weights.enumerated() {
            var shots = Int(round(Double(totalShots) * weight / totalWeight))

            shots = max(shots, effectiveMinShots)

            allocation[index] = shots
            shotsRemaining -= shots
        }

        if shotsRemaining > 0 {
            allocation = distributeRemainingShots(
                allocation: allocation,
                remaining: shotsRemaining,
                weights: weights
            )
        } else if shotsRemaining < 0 {
            allocation = reduceShots(
                allocation: allocation,
                excess: -shotsRemaining,
                weights: weights,
                minShots: effectiveMinShots
            )
        }

        return allocation
    }

    /// Allocate shots optimally for QWC groups.
    ///
    /// - Parameters:
    ///   - groups: Array of QWC groups
    ///   - totalShots: Total shots available
    ///   - state: Quantum state for variance estimation (optional)
    /// - Returns: Dictionary mapping group index to shots
    ///
    /// Example:
    /// ```swift
    /// let groups = QWCGrouper.group(terms: hamiltonian.terms)
    /// let allocator = ShotAllocator()
    /// let allocation = allocator.allocateForGroups(
    ///     groups: groups,
    ///     totalShots: 10000,
    ///     state: currentState
    /// )
    /// // Combines grouping (41× reduction) + optimal allocation (2-10× fewer shots)
    /// ```
    public func allocateForGroups(
        groups: [QWCGroup],
        totalShots: Int,
        state: QuantumState? = nil
    ) -> ShotAllocation {
        let groupWeights: [Double] = groups.map { group -> Double in
            group.terms.map { term in
                let variance = estimateVariance(
                    pauliString: term.pauliString,
                    coefficient: term.coefficient,
                    state: state
                )
                return abs(term.coefficient) * sqrt(variance)
            }.reduce(0.0, +)
        }

        let totalWeight: Double = groupWeights.reduce(0.0, +)

        guard totalWeight > 0 else {
            return uniformAllocation(numTerms: groups.count, totalShots: totalShots)
        }

        var allocation: ShotAllocation = [:]
        var shotsRemaining: Int = totalShots

        let maxPossibleMin: Int = totalShots / max(groups.count, 1)
        let effectiveMinShots: Int = min(config.minShotsPerTerm, maxPossibleMin)

        for (index, weight) in groupWeights.enumerated() {
            var shots = Int(round(Double(totalShots) * weight / totalWeight))
            shots = max(shots, effectiveMinShots)

            allocation[index] = shots
            shotsRemaining -= shots
        }

        if shotsRemaining > 0 {
            allocation = distributeRemainingShots(
                allocation: allocation,
                remaining: shotsRemaining,
                weights: groupWeights
            )
        } else if shotsRemaining < 0 {
            allocation = reduceShots(
                allocation: allocation,
                excess: -shotsRemaining,
                weights: groupWeights,
                minShots: effectiveMinShots
            )
        }

        return allocation
    }

    // MARK: - Variance Estimation

    /// Batch estimate variances for multiple Pauli strings (efficient).
    ///
    /// Creates single Observable for all terms to avoid repeated construction.
    ///
    /// - Parameters:
    ///   - terms: Array of (coefficient, PauliString) pairs
    ///   - state: Quantum state
    /// - Returns: Array of variances for each term
    private func batchEstimateVariances(
        terms: PauliTerms,
        state: QuantumState
    ) -> [Double] {
        terms.map { term in
            let expectation = computeExpectation(
                pauliString: term.pauliString,
                state: state
            )
            return 1.0 - expectation * expectation
        }
    }

    /// Estimate variance of Pauli string measurement.
    ///
    /// For a Pauli operator P: Var(P) = 1 - ⟨P⟩²
    ///
    /// - Parameters:
    ///   - pauliString: Pauli string to measure
    ///   - coefficient: Coefficient of the term
    ///   - state: Quantum state (if available for exact calculation)
    /// - Returns: Estimated variance
    private func estimateVariance(
        pauliString: PauliString,
        coefficient _: Double,
        state: QuantumState?
    ) -> Double {
        if let state {
            let expectation = computeExpectation(pauliString: pauliString, state: state)
            return 1.0 - expectation * expectation
        } else {
            return 1.0
        }
    }

    /// Compute expectation value of Pauli string.
    ///
    /// - Parameters:
    ///   - pauliString: Pauli string operator
    ///   - state: Quantum state
    /// - Returns: Expectation value ⟨ψ|P|ψ⟩
    private func computeExpectation(pauliString: PauliString, state: QuantumState) -> Double {
        Observable.computePauliExpectation(pauliString: pauliString, state: state)
    }

    // MARK: - Allocation Helpers

    private func uniformAllocation(numTerms: Int, totalShots: Int) -> ShotAllocation {
        let maxPossibleMin: Int = totalShots / max(numTerms, 1)
        let effectiveMinShots: Int = min(config.minShotsPerTerm, maxPossibleMin)
        let shotsPerTerm: Int = max(totalShots / numTerms, effectiveMinShots)
        var allocation: ShotAllocation = [:]

        for i in 0 ..< numTerms {
            allocation[i] = shotsPerTerm
        }

        return allocation
    }

    private func distributeRemainingShots(
        allocation: ShotAllocation,
        remaining: Int,
        weights: [Double]
    ) -> ShotAllocation {
        var newAllocation = allocation
        var shotsToDistribute: Int = remaining

        let sortedIndices: [Int] = weights.enumerated()
            .sorted { $0.element > $1.element }
            .map(\.offset)

        for index in sortedIndices {
            if shotsToDistribute <= 0 { break }
            newAllocation[index]! += 1
            shotsToDistribute -= 1
        }

        return newAllocation
    }

    private func reduceShots(
        allocation: ShotAllocation,
        excess: Int,
        weights: [Double],
        minShots: Int
    ) -> ShotAllocation {
        var newAllocation = allocation
        var shotsToRemove: Int = excess

        let sortedIndices: [Int] = weights.enumerated()
            .sorted { $0.element < $1.element }
            .map(\.offset)

        for index in sortedIndices {
            if shotsToRemove <= 0 { break }

            let currentShots = newAllocation[index]!
            let removable: Int = max(0, currentShots - minShots)
            let toRemove: Int = min(removable, shotsToRemove)

            newAllocation[index] = currentShots - toRemove
            shotsToRemove -= toRemove
        }

        return newAllocation
    }

    // MARK: - Statistics

    public struct AllocationStats {
        public let totalShots: Int
        public let numTerms: Int
        public let minShots: Int
        public let maxShots: Int
        public let averageShots: Double
        public let distribution: ShotAllocation

        public init(totalShots: Int, numTerms: Int, minShots: Int, maxShots: Int, averageShots: Double, distribution: ShotAllocation) {
            self.totalShots = totalShots
            self.numTerms = numTerms
            self.minShots = minShots
            self.maxShots = maxShots
            self.averageShots = averageShots
            self.distribution = distribution
        }

        public var description: String {
            """
            Shot Allocation Statistics:
            - Total shots: \(totalShots)
            - Terms: \(numTerms)
            - Min shots: \(minShots)
            - Max shots: \(maxShots)
            - Average: \(String(format: "%.1f", averageShots)) shots/term
            """
        }
    }

    public func statistics(for allocation: ShotAllocation) -> AllocationStats {
        let shots: Dictionary<Int, Int>.Values = allocation.values
        let totalShots: Int = shots.reduce(0, +)
        let numTerms: Int = allocation.count
        let minShots: Int = shots.min() ?? 0
        let maxShots: Int = shots.max() ?? 0
        let averageShots: Double = numTerms > 0 ? Double(totalShots) / Double(numTerms) : 0.0

        return AllocationStats(
            totalShots: totalShots,
            numTerms: numTerms,
            minShots: minShots,
            maxShots: maxShots,
            averageShots: averageShots,
            distribution: allocation
        )
    }

    // MARK: - Variance Reduction Estimation

    /// Estimate variance reduction compared to uniform allocation.
    ///
    /// - Parameters:
    ///   - terms: Hamiltonian terms
    ///   - allocation: Optimal shot allocation
    ///   - uniformShots: Shots per term with uniform allocation
    ///   - state: Quantum state for variance estimation (optional)
    /// - Returns: Variance reduction factor
    public func estimateVarianceReduction(
        terms: PauliTerms,
        allocation: ShotAllocation,
        uniformShots: Int,
        state: QuantumState? = nil
    ) -> Double {
        let optimalVariance: Double = terms.enumerated().map { index, term in
            precondition(allocation[index] != nil, "Allocation must contain entry for term index \(index)")
            let shots = Double(allocation[index]!)
            let variance = estimateVariance(pauliString: term.pauliString, coefficient: term.coefficient, state: state)
            return term.coefficient * term.coefficient * variance / shots
        }.reduce(0.0, +)

        let uniformVariance: Double = terms.enumerated().map { _, term in
            let variance = estimateVariance(pauliString: term.pauliString, coefficient: term.coefficient, state: state)
            return term.coefficient * term.coefficient * variance / Double(uniformShots)
        }.reduce(0.0, +)

        return uniformVariance / optimalVariance
    }
}
