// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Variance-weighted shot allocation for Hamiltonian expectation value measurements.
///
/// Optimally distributes measurement shots across Hamiltonian terms or measurement groups
/// to minimize total variance for a fixed shot budget. The variance of a Hamiltonian
/// expectation value ⟨H⟩ = Σᵢ cᵢ⟨Pᵢ⟩ depends on shot allocation nᵢ:
///
/// ```
/// Var(⟨H⟩) = Σᵢ cᵢ² Var(⟨Pᵢ⟩) / nᵢ
/// ```
///
/// Optimal allocation minimizes this variance by concentrating shots on high-variance,
/// high-weight terms:
///
/// ```
/// nᵢ ∝ |cᵢ| √Var(Pᵢ)
/// ```
///
/// Typical reduction: 2-10x fewer shots needed compared to uniform allocation for molecular
/// Hamiltonians with diverse term weights.
///
/// **Example:**
/// ```swift
/// let allocator = ShotAllocator(minShotsPerTerm: 10)
/// let allocation = allocator.allocate(
///     for: hamiltonian.terms,
///     totalShots: 10000,
///     state: currentState
/// )
/// for (index, shots) in allocation {
///     print("Term \(index): \(shots) shots")
/// }
/// ```
///
/// - SeeAlso: ``Observable``
/// - SeeAlso: ``QWCGroup``
@frozen
public struct ShotAllocator {
    /// Minimum shots allocated to each term, preventing zero allocation.
    ///
    /// Small-weight terms receive at least this many shots to avoid complete neglect.
    /// Typical values: 5-20 shots. Higher values reduce risk of missing contributions
    /// from low-weight terms but decrease overall optimization effectiveness.
    public let minShotsPerTerm: Int

    /// Creates a shot allocator with specified minimum shot constraint.
    ///
    /// - Parameter minShotsPerTerm: Minimum shots per term. Default 10 provides good
    ///   balance between optimization and coverage.
    /// - Precondition: `minShotsPerTerm` must be positive
    public init(minShotsPerTerm: Int = 10) {
        ValidationUtilities.validatePositiveInt(minShotsPerTerm, name: "minShotsPerTerm")
        self.minShotsPerTerm = minShotsPerTerm
    }

    // MARK: - Shot Allocation

    /// Allocates measurement shots optimally across Hamiltonian terms.
    ///
    /// Distributes shots using variance-weighted allocation: terms with large coefficients
    /// and high measurement variance receive proportionally more shots. When state is provided,
    /// computes exact variances Var(Pᵢ) = 1 - ⟨Pᵢ⟩² for each term. Without state, assumes
    /// unit variance (conservative worst-case).
    ///
    /// **Example:**
    /// ```swift
    /// let allocator = ShotAllocator()
    /// let allocation = allocator.allocate(
    ///     for: hamiltonian.terms,
    ///     totalShots: 10000,
    ///     state: currentState
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - terms: Hamiltonian terms as (coefficient, PauliString) pairs
    ///   - totalShots: Total measurement shots to distribute
    ///   - state: Quantum state for variance estimation. If nil, assumes unit variance for all terms.
    /// - Returns: Dictionary mapping term index to allocated shot count
    /// - Complexity: O(n) where n is number of terms
    /// - Precondition: `terms` must be non-empty
    /// - Precondition: `totalShots` must be positive
    /// - SeeAlso: ``allocate(forGroups:totalShots:state:)``
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public func allocate(
        for terms: PauliTerms,
        totalShots: Int,
        state: QuantumState? = nil,
    ) -> [Int: Int] {
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

    /// Allocates measurement shots optimally across QWC groups.
    ///
    /// Treats each qubit-wise commuting group as an atomic measurement unit and distributes
    /// shots based on aggregate group weight and variance. Combines measurement reduction
    /// from QWC grouping with variance-weighted allocation for maximum efficiency.
    ///
    /// Group weight is sum of constituent term weights: w_group = Σⱼ |cⱼ| √Var(Pⱼ) for
    /// terms j in group.
    ///
    /// **Example:**
    /// ```swift
    /// let groups = QWCGrouper.group(terms: hamiltonian.terms)
    /// let allocator = ShotAllocator()
    /// let allocation = allocator.allocate(
    ///     forGroups: groups,
    ///     totalShots: 10000,
    ///     state: currentState
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - groups: QWC groups from grouping algorithm
    ///   - totalShots: Total measurement shots to distribute
    ///   - state: Quantum state for variance estimation. If nil, uses coefficient magnitudes only.
    /// - Returns: Dictionary mapping group index to allocated shot count
    /// - Complexity: O(n) where n is total number of terms across all groups
    /// - Precondition: `groups` must be non-empty
    /// - Precondition: `totalShots` must be positive
    /// - SeeAlso: ``allocate(for:totalShots:state:)``
    /// - SeeAlso: ``QWCGroup``
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public func allocate(
        forGroups groups: [QWCGroup],
        totalShots: Int,
        state: QuantumState? = nil,
    ) -> [Int: Int] {
        ValidationUtilities.validateNonEmpty(groups, name: "groups")
        ValidationUtilities.validatePositiveInt(totalShots, name: "totalShots")
        let groupCount: Int = groups.count

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

    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    private func computeGroupWeightsBatched(
        groups: [QWCGroup],
        state: QuantumState,
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

    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    private func batchEstimateVariances(
        terms: PauliTerms,
        state: QuantumState,
    ) -> [Double] {
        let termCount: Int = terms.count
        return [Double](unsafeUninitializedCapacity: termCount) { buffer, count in
            for i in 0 ..< termCount {
                let expectation = Observable.computePauliExpectation(
                    pauliString: terms[i].pauliString,
                    for: state,
                )
                buffer[i] = 1.0 - expectation * expectation
            }
            count = termCount
        }
    }

    // MARK: - Allocation Helpers

    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    private func allocateWithWeights(
        weights: [Double],
        totalShots: Int,
    ) -> [Int: Int] {
        let weightCount: Int = weights.count

        var totalWeight = 0.0
        vDSP_sveD(weights, 1, &totalWeight, vDSP_Length(weightCount))

        guard totalWeight > 0 else {
            return uniformAllocation(numTerms: weightCount, totalShots: totalShots)
        }

        var allocation = [Int: Int](minimumCapacity: weightCount)
        var shotsRemaining: Int = totalShots

        let maxPossibleMin: Int = totalShots / max(weightCount, 1)
        let effectiveMinShots: Int = min(minShotsPerTerm, maxPossibleMin)
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
                weights: weights,
            )
        } else if shotsRemaining < 0 {
            allocation = reduceShots(
                allocation: allocation,
                excess: -shotsRemaining,
                weights: weights,
                minShots: effectiveMinShots,
            )
        }

        return allocation
    }

    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    private func uniformAllocation(numTerms: Int, totalShots: Int) -> [Int: Int] {
        let maxPossibleMin: Int = totalShots / max(numTerms, 1)
        let effectiveMinShots: Int = min(minShotsPerTerm, maxPossibleMin)
        let shotsPerTerm: Int = max(totalShots / numTerms, effectiveMinShots)

        var allocation = [Int: Int](minimumCapacity: numTerms)
        for i in 0 ..< numTerms {
            allocation[i] = shotsPerTerm
        }

        return allocation
    }

    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    private func distributeRemainingShots(
        allocation: [Int: Int],
        remaining: Int,
        weights: [Double],
    ) -> [Int: Int] {
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
            // Safety: index comes from sortedIndices which contains only 0..<weightCount,
            // and newAllocation was populated with all indices in allocateWithWeights
            newAllocation[index]! += 1
            shotsToDistribute -= 1
        }

        return newAllocation
    }

    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    private func reduceShots(
        allocation: [Int: Int],
        excess: Int,
        weights: [Double],
        minShots: Int,
    ) -> [Int: Int] {
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

            // Safety: index comes from sortedIndices which contains only 0..<weightCount,
            // and newAllocation was populated with all indices in allocateWithWeights
            let currentShots = newAllocation[index]!
            let removable: Int = max(0, currentShots - minShots)
            let toRemove: Int = min(removable, shotsToRemove)

            newAllocation[index] = currentShots - toRemove
            shotsToRemove -= toRemove
        }

        return newAllocation
    }

    // MARK: - Variance Reduction Estimation

    /// Computes variance reduction factor from optimal allocation compared to uniform distribution.
    ///
    /// Calculates ratio of variance from uniform allocation (equal shots per term) to variance
    /// from optimal weighted allocation. Return value > 1 indicates improvement: 2.0 means optimal
    /// allocation achieves same accuracy with half the shots, or twice the accuracy with same shots.
    ///
    /// **Example:**
    /// ```swift
    /// let allocator = ShotAllocator()
    /// let allocation = allocator.allocate(for: terms, totalShots: 1000)
    /// let uniformShots = 1000 / terms.count
    /// let reduction = allocator.varianceReduction(
    ///     for: terms,
    ///     using: allocation,
    ///     comparedTo: uniformShots
    /// )
    /// print("Optimal allocation is \(reduction)x better")
    /// ```
    ///
    /// - Parameters:
    ///   - terms: Hamiltonian terms
    ///   - allocation: Optimal shot allocation from ``allocate(for:totalShots:state:)``
    ///   - uniformShots: Shots per term under uniform allocation
    ///   - state: Quantum state for variance estimation. If nil, assumes unit variance.
    /// - Returns: Variance reduction factor (uniformVariance / optimalVariance). Values > 1 indicate improvement.
    /// - Complexity: O(n) where n is number of terms
    /// - Precondition: `terms` must be non-empty
    /// - Precondition: `uniformShots` must be positive
    /// - Precondition: `allocation` must contain entry for each term index
    /// - SeeAlso: ``allocate(for:totalShots:state:)``
    @_optimize(speed)
    @_effects(readonly)
    public func varianceReduction(
        for terms: PauliTerms,
        using allocation: [Int: Int],
        comparedTo uniformShots: Int,
        state: QuantumState? = nil,
    ) -> Double {
        ValidationUtilities.validateNonEmpty(terms, name: "terms")
        ValidationUtilities.validatePositiveInt(uniformShots, name: "uniformShots")
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
            // Safety: validateAllocationContainsIndex ensures key exists
            let shots = Double(allocation[index]!)
            let coeffSquared = terms[index].coefficient * terms[index].coefficient
            let varianceContrib = coeffSquared * variances[index]

            optimalVariance += varianceContrib / shots
            uniformVariance += varianceContrib / uniformShotsDouble
        }

        return uniformVariance / optimalVariance
    }
}
