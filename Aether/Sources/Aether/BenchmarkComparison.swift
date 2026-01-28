// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Compares benchmark results between baseline and optimized circuits for regression detection
/// and performance analysis.
///
/// Provides speedup, gate reduction, and depth reduction metrics to quantify optimization
/// effectiveness. Speedup greater than 1.0 indicates the optimized circuit is faster.
/// Reduction metrics are expressed as fractions (0-1) representing percentage improvement.
///
/// **Example:**
/// ```swift
/// let baseline = CircuitCostEstimator.estimate(originalCircuit)
/// let optimized = CircuitCostEstimator.estimate(optimizedCircuit)
/// let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)
/// print(comparison.speedup)        // 1.5 (50% faster)
/// print(comparison.gateReduction)  // 0.33 (33% fewer gates)
/// print(comparison.isImprovement)  // true
/// ```
///
/// - SeeAlso: ``CircuitCost``
/// - SeeAlso: ``CircuitCostEstimator``
@frozen public struct BenchmarkComparison: Sendable, Equatable, CustomStringConvertible {
    /// Baseline circuit cost before optimization.
    public let baseline: CircuitCost

    /// Optimized circuit cost after optimization.
    public let optimized: CircuitCost

    /// Creates a benchmark comparison between baseline and optimized circuit costs.
    ///
    /// **Example:**
    /// ```swift
    /// let comparison = BenchmarkComparison(baseline: baselineCost, optimized: optimizedCost)
    /// ```
    ///
    /// - Parameters:
    ///   - baseline: Circuit cost before optimization
    ///   - optimized: Circuit cost after optimization
    @inlinable
    public init(baseline: CircuitCost, optimized: CircuitCost) {
        self.baseline = baseline
        self.optimized = optimized
    }

    /// Speedup factor comparing baseline to optimized gate counts.
    ///
    /// Computed as baseline.totalGates / optimized.totalGates. Values greater than 1.0
    /// indicate the optimized circuit has fewer gates (is faster). Returns 0.0 if
    /// optimized.totalGates is zero.
    ///
    /// **Example:**
    /// ```swift
    /// let comparison = BenchmarkComparison(baseline: cost100, optimized: cost50)
    /// print(comparison.speedup)  // 2.0 (2x speedup)
    /// ```
    ///
    /// - Returns: Speedup factor (>1.0 means faster), or 0.0 if optimized gate count is zero.
    @inlinable
    public var speedup: Double {
        guard optimized.totalGates > 0 else { return 0.0 }
        return Double(baseline.totalGates) / Double(optimized.totalGates)
    }

    /// Gate count reduction as a fraction (0-1).
    ///
    /// Computed as (baseline.totalGates - optimized.totalGates) / baseline.totalGates.
    /// Values greater than 0 indicate fewer gates in the optimized circuit.
    /// Returns 0.0 if baseline.totalGates is zero.
    ///
    /// **Example:**
    /// ```swift
    /// let comparison = BenchmarkComparison(baseline: cost100, optimized: cost75)
    /// print(comparison.gateReduction)  // 0.25 (25% reduction)
    /// ```
    ///
    /// - Returns: Reduction fraction (0-1), or 0.0 if baseline gate count is zero.
    @inlinable
    public var gateReduction: Double {
        guard baseline.totalGates > 0 else { return 0.0 }
        return Double(baseline.totalGates - optimized.totalGates) / Double(baseline.totalGates)
    }

    /// Circuit depth reduction as a fraction (0-1).
    ///
    /// Computed as (baseline.depth - optimized.depth) / baseline.depth.
    /// Values greater than 0 indicate shallower optimized circuit.
    /// Returns 0.0 if baseline.depth is zero.
    ///
    /// **Example:**
    /// ```swift
    /// let comparison = BenchmarkComparison(baseline: costDepth10, optimized: costDepth8)
    /// print(comparison.depthReduction)  // 0.2 (20% reduction)
    /// ```
    ///
    /// - Returns: Reduction fraction (0-1), or 0.0 if baseline depth is zero.
    @inlinable
    public var depthReduction: Double {
        guard baseline.depth > 0 else { return 0.0 }
        return Double(baseline.depth - optimized.depth) / Double(baseline.depth)
    }

    /// Whether the optimization resulted in any improvement.
    ///
    /// Returns true if speedup > 1.0, gateReduction > 0, or depthReduction > 0.
    ///
    /// **Example:**
    /// ```swift
    /// if comparison.isImprovement {
    ///     print("Optimization successful")
    /// }
    /// ```
    ///
    /// - Returns: `true` if any metric shows improvement, `false` otherwise.
    @inlinable
    public var isImprovement: Bool {
        speedup > 1.0 || gateReduction > 0 || depthReduction > 0
    }

    /// Human-readable summary of the benchmark comparison.
    @inlinable
    public var description: String {
        "BenchmarkComparison(speedup: \(String(format: "%.2f", speedup))x, gateReduction: \(String(format: "%.1f", gateReduction * 100))%, depthReduction: \(String(format: "%.1f", depthReduction * 100))%)"
    }
}
