// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Observable approximation methods for adaptive VQE optimization
///
/// Provides low-rank Hamiltonian approximation techniques that enable faster VQE
/// convergence by using cheaper approximations in early iterations and refining
/// with the full Hamiltonian in later iterations.
///
/// **Key techniques**:
/// - **Coefficient truncation**: Remove terms with |cᵢ| < threshold
/// - **Top-k selection**: Keep only k largest-magnitude terms
/// - **Adaptive schedules**: Dynamic thresholding based on iteration count
///
/// **Use cases**:
/// - Adaptive VQE with progressive refinement
/// - Fast initial optimization with approximate Hamiltonians
/// - Memory/computation tradeoffs for large molecular systems
///
/// **Typical workflow**:
/// 1. Early iterations: Use aggressive approximation (threshold = 0.1)
/// 2. Mid iterations: Moderate approximation (threshold = 0.01)
/// 3. Final iterations: Full Hamiltonian for accurate convergence
///
/// Example:
/// ```swift
/// let hamiltonian = Observable(terms: molecularTerms)  // 2000 terms
///
/// // Adaptive VQE loop
/// for iteration in 0..<100 {
///     let threshold = iteration < 20 ? 0.1 : iteration < 50 ? 0.01 : 0.0
///     let approxH = hamiltonian.truncate(threshold: threshold)
///     // Use approxH for faster energy evaluation
/// }
/// ```

// MARK: - Low-Rank Approximation

public extension Observable {
    /// Approximate observable by truncating small-coefficient terms.
    ///
    /// - Parameter threshold: Minimum absolute coefficient to keep
    /// - Returns: Approximated observable with fewer terms
    ///
    /// This is useful for adaptive VQE where early iterations can use
    /// a cheaper approximation and later iterations refine with full Hamiltonian.
    ///
    /// Example:
    /// ```swift
    /// // Early VQE iterations
    /// let approxH = hamiltonian.truncate(threshold: 0.1)  // Fast
    ///
    /// // Final iterations
    /// let energy = hamiltonian.expectationValue(state: state)  // Accurate
    /// ```
    @_eagerMove
    @_effects(readonly)
    func truncate(threshold: Double) -> Observable {
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

    /// Approximate observable by keeping only top k terms.
    ///
    /// - Parameter k: Number of terms to keep
    /// - Returns: Observable with k largest-coefficient terms
    ///
    /// Example:
    /// ```swift
    /// let approxH = hamiltonian.topK(k: 100)  // Keep 100 largest terms
    /// print("Using \(approxH.terms.count) out of \(hamiltonian.terms.count) terms")
    /// ```
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    func topK(k: Int) -> Observable {
        guard k > 0 else {
            if let largest = terms.max(by: { abs($0.coefficient) < abs($1.coefficient) }) {
                return Observable(terms: [largest])
            }
            return Observable(terms: [])
        }

        let sortedTerms = terms.sorted { abs($0.coefficient) > abs($1.coefficient) }

        let count = min(k, sortedTerms.count)
        let topTerms = PauliTerms(unsafeUninitializedCapacity: count) { buffer, outCount in
            for i in 0 ..< count {
                buffer[i] = sortedTerms[i]
            }
            outCount = count
        }

        return Observable(terms: topTerms)
    }

    /// Compute approximation error relative to exact observable.
    ///
    /// - Parameters:
    ///   - approximate: Approximated observable
    ///   - state: Quantum state to evaluate on
    /// - Returns: Absolute error |⟨H⟩ - ⟨H'⟩|
    @_effects(readonly)
    func approximationError(approximate: Observable, state: QuantumState) -> Double {
        let exactValue: Double = expectationValue(state: state)
        let approxValue: Double = approximate.expectationValue(state: state)
        return abs(exactValue - approxValue)
    }

    /// Compute relative approximation error.
    ///
    /// - Parameters:
    ///   - approximate: Approximated observable
    ///   - state: Quantum state to evaluate on
    /// - Returns: Relative error |⟨H⟩ - ⟨H'⟩| / |⟨H⟩|
    @_effects(readonly)
    func relativeApproximationError(approximate: Observable, state: QuantumState) -> Double {
        let exactValue: Double = expectationValue(state: state)
        guard abs(exactValue) > 1e-10 else { return 0.0 }

        let approxValue: Double = approximate.expectationValue(state: state)
        return abs(exactValue - approxValue) / abs(exactValue)
    }

    // MARK: - Adaptive Approximation Strategies

    /// Adaptive threshold schedule for VQE optimization.
    @frozen
    struct AdaptiveSchedule: Sendable {
        /// Initial threshold (largest, most aggressive truncation)
        public let initialThreshold: Double

        /// Final threshold (smallest, most accurate)
        public let finalThreshold: Double

        /// Decay rate (exponential decay)
        public let decayRate: Double

        /// Current iteration
        public var iteration: Int = 0

        /// Get threshold for current iteration.
        @inlinable
        @_effects(readonly)
        public func threshold() -> Double {
            let t = Double(iteration)
            return finalThreshold + (initialThreshold - finalThreshold) * exp(-decayRate * t)
        }

        public mutating func advance() {
            iteration += 1
        }

        public static let aggressive = AdaptiveSchedule(
            initialThreshold: 0.5,
            finalThreshold: 0.0,
            decayRate: 0.1
        )

        public static let moderate = AdaptiveSchedule(
            initialThreshold: 0.1,
            finalThreshold: 0.0,
            decayRate: 0.05
        )

        public static let conservative = AdaptiveSchedule(
            initialThreshold: 0.01,
            finalThreshold: 0.0,
            decayRate: 0.02
        )
    }

    /// Apply adaptive truncation based on schedule.
    ///
    /// - Parameter schedule: Adaptive threshold schedule
    /// - Returns: Truncated observable for current iteration
    @_eagerMove
    @_effects(readonly)
    func adaptiveTruncate(schedule: AdaptiveSchedule) -> Observable {
        let threshold: Double = schedule.threshold()
        return truncate(threshold: threshold)
    }

    // MARK: - Approximation Statistics

    @frozen
    struct ApproximationStats {
        public let originalTerms: Int
        public let approximateTerms: Int
        public let reductionFactor: Double
        public let coefficientSumOriginal: Double
        public let coefficientSumApproximate: Double
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

    /// Compute approximation statistics.
    @_eagerMove
    @_effects(readonly)
    func approximationStatistics(approximate: Observable) -> ApproximationStats {
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
            coefficientRetention: retention
        )
    }

    // MARK: - Validation

    /// Validate that approximation meets error tolerance.
    ///
    /// - Parameters:
    ///   - approximate: Approximated observable
    ///   - state: Test state
    ///   - tolerance: Maximum acceptable absolute error
    /// - Returns: True if error is within tolerance
    @_effects(readonly)
    func validateApproximation(
        approximate: Observable,
        state: QuantumState,
        tolerance: Double
    ) -> Bool {
        let error: Double = approximationError(approximate: approximate, state: state)
        return error <= tolerance
    }

    /// Find minimum threshold that meets error tolerance.
    ///
    /// - Parameters:
    ///   - state: Test state
    ///   - maxError: Maximum acceptable error
    ///   - searchSteps: Number of binary search steps
    /// - Returns: Minimum threshold meeting error tolerance
    @_optimize(speed)
    func findOptimalThreshold(
        state: QuantumState,
        maxError: Double,
        searchSteps: Int = 20
    ) -> Double {
        var high = 0.0
        for i in 0 ..< terms.count {
            let coeff = abs(terms[i].coefficient)
            if coeff > high { high = coeff }
        }
        if high == 0.0 { high = 1.0 }

        let exactValue = expectationValue(state: state)

        var low = 0.0

        for _ in 0 ..< searchSteps {
            let mid = (low + high) / 2.0
            let approx = truncate(threshold: mid)
            let approxValue = approx.expectationValue(state: state)
            let error = abs(exactValue - approxValue)

            if error <= maxError { low = mid } else { high = mid }
        }

        return low
    }
}
