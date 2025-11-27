// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate
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
/// nᵢ = N x (|cᵢ| √Var(Pᵢ)) / (Σⱼ |cⱼ| √Var(Pⱼ))
///
/// This reduces shot requirements compared to uniform allocation by concentrating
/// shots on high-variance, high-weight terms.
@frozen
public struct ShotAllocator {
    // MARK: - Configuration

    @frozen
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
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public func allocate(
        terms: PauliTerms,
        totalShots: Int,
        state: QuantumState? = nil
    ) -> ShotAllocation {
        let termCount: Int = terms.count
        ValidationUtilities.validateNonEmpty(terms, name: "terms")
        ValidationUtilities.validatePositiveInt(totalShots, name: "totalShots")

        let variances: [Double] = if let state {
            batchEstimateVariances(terms: terms, state: state)
        } else {
            Array(repeating: 1.0, count: termCount)
        }

        let weights = [Double](unsafeUninitializedCapacity: termCount) { buffer, count in
            for i in 0 ..< termCount {
                buffer[i] = abs(terms[i].coefficient) * sqrt(variances[i])
            }
            count = termCount
        }

        return allocateWithWeights(weights: weights, totalShots: totalShots)
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
    /// // Combines grouping reduction + optimal allocation
    /// ```
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public func allocateForGroups(
        groups: [QWCGroup],
        totalShots: Int,
        state: QuantumState? = nil
    ) -> ShotAllocation {
        let groupCount: Int = groups.count
        guard groupCount > 0, totalShots > 0 else { return [:] }

        let groupWeights: [Double] = if let state {
            computeGroupWeightsBatched(groups: groups, state: state)
        } else {
            [Double](unsafeUninitializedCapacity: groupCount) { buffer, count in
                for groupIndex in 0 ..< groupCount {
                    var groupWeight = 0.0
                    for term in groups[groupIndex].terms {
                        groupWeight += abs(term.coefficient)
                    }
                    buffer[groupIndex] = groupWeight
                }
                count = groupCount
            }
        }

        return allocateWithWeights(weights: groupWeights, totalShots: totalShots)
    }

    /// Compute group weights with batched variance estimation.
    @_optimize(speed)
    @_effects(readonly)
    private func computeGroupWeightsBatched(
        groups: [QWCGroup],
        state: QuantumState
    ) -> [Double] {
        let groupCount: Int = groups.count

        var totalTerms = 0
        let groupBoundaries = [Int](unsafeUninitializedCapacity: groupCount + 1) { buffer, count in
            buffer[0] = 0
            for i in 0 ..< groupCount {
                totalTerms += groups[i].terms.count
                buffer[i + 1] = totalTerms
            }
            count = groupCount + 1
        }

        var allTerms = PauliTerms()
        allTerms.reserveCapacity(totalTerms)
        for group in groups {
            allTerms.append(contentsOf: group.terms)
        }

        let allVariances = batchEstimateVariances(terms: allTerms, state: state)

        return [Double](unsafeUninitializedCapacity: groupCount) { buffer, count in
            for groupIndex in 0 ..< groupCount {
                let start = groupBoundaries[groupIndex]
                let end = groupBoundaries[groupIndex + 1]
                var groupWeight = 0.0
                for termIndex in start ..< end {
                    groupWeight += abs(allTerms[termIndex].coefficient) * sqrt(allVariances[termIndex])
                }
                buffer[groupIndex] = groupWeight
            }
            count = groupCount
        }
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
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    func batchEstimateVariances(
        terms: PauliTerms,
        state: QuantumState
    ) -> [Double] {
        let termCount: Int = terms.count
        return [Double](unsafeUninitializedCapacity: termCount) { buffer, count in
            for i in 0 ..< termCount {
                let expectation = Observable.computePauliExpectation(
                    pauliString: terms[i].pauliString,
                    state: state
                )
                buffer[i] = 1.0 - expectation * expectation
            }
            count = termCount
        }
    }

    // MARK: - Allocation Helpers

    /// Shared allocation logic for both term-based and group-based allocation.
    @_optimize(speed)
    @_effects(readonly)
    private func allocateWithWeights(
        weights: [Double],
        totalShots: Int
    ) -> ShotAllocation {
        let weightCount: Int = weights.count

        var totalWeight = 0.0
        vDSP_sveD(weights, 1, &totalWeight, vDSP_Length(weightCount))

        guard totalWeight > 0 else {
            return uniformAllocation(numTerms: weightCount, totalShots: totalShots)
        }

        var allocation = ShotAllocation(minimumCapacity: weightCount)
        var shotsRemaining: Int = totalShots

        let maxPossibleMin: Int = totalShots / max(weightCount, 1)
        let effectiveMinShots: Int = min(config.minShotsPerTerm, maxPossibleMin)
        let shotMultiplier = Double(totalShots) / totalWeight

        for index in 0 ..< weightCount {
            var shots = Int(round(shotMultiplier * weights[index]))
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

    @_effects(readonly)
    private func uniformAllocation(numTerms: Int, totalShots: Int) -> ShotAllocation {
        let maxPossibleMin: Int = totalShots / max(numTerms, 1)
        let effectiveMinShots: Int = min(config.minShotsPerTerm, maxPossibleMin)
        let shotsPerTerm: Int = max(totalShots / numTerms, effectiveMinShots)

        var allocation = ShotAllocation(minimumCapacity: numTerms)
        for i in 0 ..< numTerms {
            allocation[i] = shotsPerTerm
        }

        return allocation
    }

    @_optimize(speed)
    @_effects(readonly)
    private func distributeRemainingShots(
        allocation: ShotAllocation,
        remaining: Int,
        weights: [Double]
    ) -> ShotAllocation {
        var newAllocation = allocation
        var shotsToDistribute: Int = remaining

        let weightCount: Int = weights.count
        var sortedIndices = [Int](unsafeUninitializedCapacity: weightCount) { buffer, count in
            for i in 0 ..< weightCount {
                buffer[i] = i
            }
            count = weightCount
        }
        sortedIndices.sort { weights[$0] > weights[$1] }

        for index in sortedIndices {
            if shotsToDistribute <= 0 { break }
            // Safe: allocation was built with all indices 0..<weightCount
            newAllocation[index]! += 1
            shotsToDistribute -= 1
        }

        return newAllocation
    }

    @_optimize(speed)
    @_effects(readonly)
    private func reduceShots(
        allocation: ShotAllocation,
        excess: Int,
        weights: [Double],
        minShots: Int
    ) -> ShotAllocation {
        var newAllocation = allocation
        var shotsToRemove: Int = excess

        let weightCount: Int = weights.count
        var sortedIndices = [Int](unsafeUninitializedCapacity: weightCount) { buffer, count in
            for i in 0 ..< weightCount {
                buffer[i] = i
            }
            count = weightCount
        }
        sortedIndices.sort { weights[$0] < weights[$1] }

        for index in sortedIndices {
            if shotsToRemove <= 0 { break }

            // Safe: allocation was built with all indices 0..<weightCount
            let currentShots = newAllocation[index]!
            let removable: Int = max(0, currentShots - minShots)
            let toRemove: Int = min(removable, shotsToRemove)

            newAllocation[index] = currentShots - toRemove
            shotsToRemove -= toRemove
        }

        return newAllocation
    }

    // MARK: - Statistics

    @frozen
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

    @_effects(readonly)
    @inlinable
    public func statistics(for allocation: ShotAllocation) -> AllocationStats {
        let numTerms: Int = allocation.count

        guard numTerms > 0 else {
            return AllocationStats(
                totalShots: 0,
                numTerms: 0,
                minShots: 0,
                maxShots: 0,
                averageShots: 0.0,
                distribution: allocation
            )
        }

        var totalShots = 0
        var minShots = Int.max
        var maxShots = Int.min
        for shots in allocation.values {
            totalShots += shots
            if shots < minShots { minShots = shots }
            if shots > maxShots { maxShots = shots }
        }

        let averageShots = Double(totalShots) / Double(numTerms)

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
    @_optimize(speed)
    @_effects(readonly)
    public func estimateVarianceReduction(
        terms: PauliTerms,
        allocation: ShotAllocation,
        uniformShots: Int,
        state: QuantumState? = nil
    ) -> Double {
        let termCount: Int = terms.count
        let uniformShotsDouble = Double(uniformShots)

        let variances: [Double] = if let state {
            batchEstimateVariances(terms: terms, state: state)
        } else {
            Array(repeating: 1.0, count: termCount)
        }

        var optimalVariance = 0.0
        var uniformVariance = 0.0

        for index in 0 ..< termCount {
            ValidationUtilities.validateAllocationContainsIndex(allocation, index: index)
            // Safe: validated above
            let shots = Double(allocation[index]!)
            let coeffSquared = terms[index].coefficient * terms[index].coefficient
            let varianceContrib = coeffSquared * variances[index]

            optimalVariance += varianceContrib / shots
            uniformVariance += varianceContrib / uniformShotsDouble
        }

        return uniformVariance / optimalVariance
    }
}
