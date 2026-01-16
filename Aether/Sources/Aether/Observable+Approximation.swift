// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Low-rank Hamiltonian approximation for adaptive VQE optimization.
///
/// Provides coefficient-based filtering techniques that reduce the computational cost of
/// Hamiltonian expectation values by removing or truncating small-magnitude terms. Enables
/// adaptive VQE strategies where early optimization iterations use cheap approximations
/// and later iterations refine with the full Hamiltonian for accurate convergence.
///
/// Approximation methods include coefficient truncation (removing terms below a threshold),
/// top-k selection (keeping only the largest magnitude terms), and adaptive schedules
/// (dynamic thresholding based on iteration count). Typical use involves aggressive
/// approximation in the initial exploration phase followed by progressive refinement as
/// optimization converges.
///
/// - SeeAlso: ``Observable``
/// - SeeAlso: ``VQE``
///
/// **Example:**
/// ```swift
/// let hamiltonian = Observable(terms: molecularTerms)
/// for iteration in 0..<100 {
///     let threshold = iteration < 20 ? 0.1 : 0.0
///     let approxH = hamiltonian.filtering(coefficientThreshold: threshold)
/// }
/// ```
public extension Observable {
    /// Filter observable by removing small-coefficient terms.
    ///
    /// Retains only terms with absolute coefficient above the specified threshold. If all terms
    /// fall below the threshold, returns the single largest-magnitude term to avoid producing
    /// an empty observable. Useful for adaptive VQE where early iterations tolerate lower
    /// accuracy for computational savings.
    ///
    /// - Parameter threshold: Minimum absolute coefficient to retain
    /// - Returns: Filtered observable with fewer terms
    /// - Complexity: O(k) where k is the number of terms
    ///
    /// **Example:**
    /// ```swift
    /// let approxH = hamiltonian.filtering(coefficientThreshold: 0.1)
    /// let energy = approxH.expectationValue(state: state)
    /// ```
    @_eagerMove
    @_effects(readonly)
    func filtering(coefficientThreshold threshold: Double) -> Observable {
        var significantTerms: PauliTerms = []
        significantTerms.reserveCapacity(terms.count)

        var maxCoeff = -Double.infinity
        var maxIndex = -1

        for i in 0 ..< terms.count {
            let coeff = abs(terms[i].coefficient)
            if coeff >= threshold {
                significantTerms.append(terms[i])
            }
            if coeff > maxCoeff {
                maxCoeff = coeff
                maxIndex = i
            }
        }

        if significantTerms.isEmpty, maxIndex >= 0 {
            return Observable(terms: [terms[maxIndex]])
        }

        return Observable(terms: significantTerms)
    }

    /// Retain only the largest-coefficient terms.
    ///
    /// Sorts terms by absolute coefficient magnitude and retains the top k entries. If count
    /// exceeds the number of available terms, returns all terms. Guarantees at least one term
    /// in the result by returning the largest-magnitude term when count is zero or negative.
    ///
    /// - Parameter count: Number of largest terms to keep
    /// - Returns: Observable with at most count terms
    /// - Complexity: O(k log k) where k is the number of terms (dominated by sorting)
    ///
    /// **Example:**
    /// ```swift
    /// let approxH = hamiltonian.keepingLargest(100)
    /// print("Reduced from \(hamiltonian.terms.count) to \(approxH.terms.count)")
    /// ```
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    func keepingLargest(_ count: Int) -> Observable {
        guard count > 0 else {
            if let largest = terms.max(by: { abs($0.coefficient) < abs($1.coefficient) }) {
                return Observable(terms: [largest])
            }
            return Observable(terms: [])
        }

        let sortedTerms = terms.sorted { abs($0.coefficient) > abs($1.coefficient) }

        let actualCount = min(count, sortedTerms.count)
        let topTerms = PauliTerms(unsafeUninitializedCapacity: actualCount) { buffer, outCount in
            for i in 0 ..< actualCount {
                buffer[i] = sortedTerms[i]
            }
            outCount = actualCount
        }

        return Observable(terms: topTerms)
    }

    /// Compute absolute approximation error on a quantum state.
    ///
    /// - Parameters:
    ///   - approximate: Approximated observable
    ///   - state: Quantum state for evaluation
    /// - Returns: Absolute error |⟨H⟩ - ⟨H'⟩|
    /// - Complexity: O(k·2ⁿ) where k is the number of terms and n is the number of qubits
    /// - Precondition: State must be normalized
    ///
    /// **Example:**
    /// ```swift
    /// let approx = hamiltonian.filtering(coefficientThreshold: 0.1)
    /// let err = hamiltonian.error(of: approx, state: state)
    /// ```
    @_effects(readonly)
    func error(of approximate: Observable, state: QuantumState) -> Double {
        let exactValue: Double = expectationValue(of: state)
        let approxValue: Double = approximate.expectationValue(of: state)
        return abs(exactValue - approxValue)
    }

    /// Compute relative approximation error on a quantum state.
    ///
    /// Returns zero if the exact expectation value is near zero to avoid division issues.
    ///
    /// - Parameters:
    ///   - approximate: Approximated observable
    ///   - state: Quantum state for evaluation
    /// - Returns: Relative error |⟨H⟩ - ⟨H'⟩| / |⟨H⟩|
    /// - Complexity: O(k·2ⁿ) where k is the number of terms and n is the number of qubits
    /// - Precondition: State must be normalized
    ///
    /// **Example:**
    /// ```swift
    /// let approx = hamiltonian.filtering(coefficientThreshold: 0.1)
    /// let relErr = hamiltonian.relativeError(of: approx, state: state)
    /// ```
    @_effects(readonly)
    func relativeError(of approximate: Observable, state: QuantumState) -> Double {
        let exactValue: Double = expectationValue(of: state)
        guard abs(exactValue) > 1e-10 else { return 0.0 }

        let approxValue: Double = approximate.expectationValue(of: state)
        return abs(exactValue - approxValue) / abs(exactValue)
    }

    // MARK: - Adaptive Approximation Strategies

    /// Adaptive threshold schedule for progressive refinement during optimization.
    ///
    /// Defines an exponential decay schedule from an initial threshold (aggressive truncation)
    /// to a final threshold (accurate convergence). Typical usage involves starting with high
    /// threshold values to accelerate early exploration, then gradually reducing the threshold
    /// as optimization progresses toward convergence.
    @frozen
    struct AdaptiveSchedule: Sendable {
        /// Initial threshold for early iterations (largest truncation).
        public let initialThreshold: Double

        /// Final threshold for late iterations (minimal truncation).
        public let finalThreshold: Double

        /// Exponential decay rate controlling convergence speed.
        public let decayRate: Double

        /// Compute threshold for a given optimization iteration.
        ///
        /// - Parameter iteration: Current iteration number
        /// - Returns: Threshold value following exponential decay
        @_effects(readonly)
        @inlinable
        public func threshold(at iteration: Int) -> Double {
            let t = Double(iteration)
            return finalThreshold + (initialThreshold - finalThreshold) * exp(-decayRate * t)
        }

        /// Aggressive schedule: fast decay from 0.5 to 0.0.
        public static let aggressive = AdaptiveSchedule(
            initialThreshold: 0.5,
            finalThreshold: 0.0,
            decayRate: 0.1,
        )

        /// Moderate schedule: balanced decay from 0.1 to 0.0.
        public static let moderate = AdaptiveSchedule(
            initialThreshold: 0.1,
            finalThreshold: 0.0,
            decayRate: 0.05,
        )

        /// Conservative schedule: slow decay from 0.01 to 0.0.
        public static let conservative = AdaptiveSchedule(
            initialThreshold: 0.01,
            finalThreshold: 0.0,
            decayRate: 0.02,
        )
    }

    /// Apply adaptive filtering based on iteration-dependent schedule.
    ///
    /// - Parameters:
    ///   - schedule: Adaptive threshold schedule
    ///   - iteration: Current optimization iteration
    /// - Returns: Filtered observable for this iteration
    /// - Complexity: O(k) where k is the number of terms
    ///
    /// **Example:**
    /// ```swift
    /// let approxH = hamiltonian.applying(schedule: .moderate, iteration: iteration)
    /// ```
    @_eagerMove
    @_effects(readonly)
    func applying(schedule: AdaptiveSchedule, iteration: Int) -> Observable {
        let threshold: Double = schedule.threshold(at: iteration)
        return filtering(coefficientThreshold: threshold)
    }

    // MARK: - Approximation Statistics

    /// Statistics describing the quality of a Hamiltonian approximation.
    @frozen
    struct ApproximationStats: Sendable, CustomStringConvertible {
        /// Number of terms in original observable.
        public let originalTerms: Int

        /// Number of terms in approximation.
        public let approximateTerms: Int

        /// Ratio of original to approximate term count.
        public let reductionFactor: Double

        /// Sum of absolute coefficients in original observable.
        public let coefficientSumOriginal: Double

        /// Sum of absolute coefficients in approximation.
        public let coefficientSumApproximate: Double

        /// Fraction of coefficient magnitude retained by approximation.
        public let coefficientRetention: Double

        public var description: String {
            """
            Approximation Statistics:
            - Original terms: \(originalTerms)
            - Approximate terms: \(approximateTerms)
            - Reduction: \(String(format: "%.1f", reductionFactor))x
            - Coefficient retention: \(String(format: "%.1f%%", coefficientRetention * 100))
            """
        }
    }

    /// Compute statistics comparing approximation quality to original observable.
    ///
    /// - Parameter approximate: Approximated observable
    /// - Returns: Statistics structure with reduction metrics
    /// - Complexity: O(k) where k is the total number of terms in both observables
    ///
    /// **Example:**
    /// ```swift
    /// let approx = hamiltonian.filtering(coefficientThreshold: 0.1)
    /// let stats = hamiltonian.approximationStatistics(of: approx)
    /// print(stats.reductionFactor)
    /// ```
    @_effects(readonly)
    func approximationStatistics(of approximate: Observable) -> ApproximationStats {
        let originalTerms = terms.count
        let approximateTerms = approximate.terms.count

        var originalSum = 0.0
        var approximateSum = 0.0

        for i in 0 ..< terms.count {
            originalSum += abs(terms[i].coefficient)
        }
        for i in 0 ..< approximate.terms.count {
            approximateSum += abs(approximate.terms[i].coefficient)
        }

        let reductionFactor = originalTerms > 0 ? Double(originalTerms) / Double(approximateTerms) : 1.0
        let retention = originalSum > 0 ? approximateSum / originalSum : 0.0

        return ApproximationStats(
            originalTerms: originalTerms,
            approximateTerms: approximateTerms,
            reductionFactor: reductionFactor,
            coefficientSumOriginal: originalSum,
            coefficientSumApproximate: approximateSum,
            coefficientRetention: retention,
        )
    }

    // MARK: - Validation

    /// Check whether approximation satisfies error tolerance on a test state.
    ///
    /// - Parameters:
    ///   - approximate: Approximated observable
    ///   - state: Test state for error evaluation
    ///   - tolerance: Maximum acceptable absolute error
    /// - Returns: True if error is within tolerance
    /// - Complexity: O(k·2ⁿ) where k is the number of terms and n is the number of qubits
    /// - Precondition: State must be normalized
    ///
    /// **Example:**
    /// ```swift
    /// let approx = hamiltonian.filtering(coefficientThreshold: 0.1)
    /// if hamiltonian.meetsAccuracy(approx, state: state, tolerance: 0.01) {
    ///     print("Approximation is accurate enough")
    /// }
    /// ```
    @_effects(readonly)
    func meetsAccuracy(
        _ approximate: Observable,
        state: QuantumState,
        tolerance: Double,
    ) -> Bool {
        let errorValue: Double = error(of: approximate, state: state)
        return errorValue <= tolerance
    }

    /// Find minimum threshold satisfying error tolerance via binary search.
    ///
    /// Performs binary search over the coefficient range to identify the smallest threshold
    /// that produces an approximation meeting the specified error tolerance. Useful for
    /// determining the optimal trade-off between accuracy and computational cost.
    ///
    /// - Parameters:
    ///   - state: Test state for error evaluation
    ///   - maxError: Maximum acceptable error
    ///   - searchSteps: Number of binary search iterations (default: 20)
    /// - Returns: Minimum threshold meeting error tolerance
    /// - Complexity: O(searchSteps · k · 2ⁿ) where k is the number of terms
    /// - Precondition: State must be normalized
    ///
    /// **Example:**
    /// ```swift
    /// let threshold = hamiltonian.findOptimalThreshold(state: state, maxError: 0.01)
    /// let approx = hamiltonian.filtering(coefficientThreshold: threshold)
    /// ```
    @_optimize(speed)
    @_effects(readonly)
    func findOptimalThreshold(
        state: QuantumState,
        maxError: Double,
        searchSteps: Int = 20,
    ) -> Double {
        var high = 0.0
        for i in 0 ..< terms.count {
            let coeff = abs(terms[i].coefficient)
            if coeff > high { high = coeff }
        }
        if high == 0.0 { high = 1.0 }

        let exactValue = expectationValue(of: state)

        var low = 0.0

        for _ in 0 ..< searchSteps {
            let mid = (low + high) / 2.0
            let approx = filtering(coefficientThreshold: mid)
            let approxValue = approx.expectationValue(of: state)
            let errorValue = abs(exactValue - approxValue)

            if errorValue <= maxError { low = mid } else { high = mid }
        }

        return low
    }
}
