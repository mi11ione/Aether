// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Tests for BenchmarkComparison functionality.
/// Validates speedup, gate reduction, depth reduction metrics,
/// and isImprovement flag across baseline/optimized combinations.
@Suite("BenchmarkComparison")
struct BenchmarkComparisonTests {
    let tolerance: Double = 1e-10

    @Test("Initialization stores baseline and optimized costs")
    func initializationStoresCosts() {
        let baseline = CircuitCost(gateCount: [:], depth: 10, cnotEquivalent: 5, tCount: 2, totalGates: 100)
        let optimized = CircuitCost(gateCount: [:], depth: 8, cnotEquivalent: 3, tCount: 1, totalGates: 50)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        #expect(comparison.baseline == baseline, "Baseline cost should be stored correctly")
        #expect(comparison.optimized == optimized, "Optimized cost should be stored correctly")
    }

    @Test("Speedup computes ratio of baseline to optimized gates")
    func speedupComputesCorrectRatio() {
        let baseline = CircuitCost(gateCount: [:], depth: 10, cnotEquivalent: 5, tCount: 2, totalGates: 100)
        let optimized = CircuitCost(gateCount: [:], depth: 8, cnotEquivalent: 3, tCount: 1, totalGates: 50)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        #expect(abs(comparison.speedup - 2.0) < tolerance, "Speedup should be 2.0 when baseline has twice the gates")
    }

    @Test("Speedup returns zero when optimized has zero gates")
    func speedupReturnsZeroForZeroOptimizedGates() {
        let baseline = CircuitCost(gateCount: [:], depth: 10, cnotEquivalent: 5, tCount: 2, totalGates: 100)
        let optimized = CircuitCost(gateCount: [:], depth: 0, cnotEquivalent: 0, tCount: 0, totalGates: 0)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        #expect(abs(comparison.speedup - 0.0) < tolerance, "Speedup should be 0.0 when optimized gates is zero")
    }

    @Test("Gate reduction computes fractional reduction correctly")
    func gateReductionComputesFraction() {
        let baseline = CircuitCost(gateCount: [:], depth: 10, cnotEquivalent: 5, tCount: 2, totalGates: 100)
        let optimized = CircuitCost(gateCount: [:], depth: 8, cnotEquivalent: 3, tCount: 1, totalGates: 75)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        #expect(abs(comparison.gateReduction - 0.25) < tolerance, "Gate reduction should be 0.25 for 25% reduction")
    }

    @Test("Gate reduction returns zero when baseline has zero gates")
    func gateReductionReturnsZeroForZeroBaseline() {
        let baseline = CircuitCost(gateCount: [:], depth: 0, cnotEquivalent: 0, tCount: 0, totalGates: 0)
        let optimized = CircuitCost(gateCount: [:], depth: 5, cnotEquivalent: 2, tCount: 1, totalGates: 50)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        #expect(abs(comparison.gateReduction - 0.0) < tolerance, "Gate reduction should be 0.0 when baseline gates is zero")
    }

    @Test("Gate reduction handles negative value when optimized has more gates")
    func gateReductionHandlesNegative() {
        let baseline = CircuitCost(gateCount: [:], depth: 5, cnotEquivalent: 2, tCount: 1, totalGates: 50)
        let optimized = CircuitCost(gateCount: [:], depth: 10, cnotEquivalent: 5, tCount: 2, totalGates: 100)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        #expect(comparison.gateReduction < 0, "Gate reduction should be negative when optimized has more gates")
        #expect(abs(comparison.gateReduction - -1.0) < tolerance, "Gate reduction should be -1.0 when optimized has twice the gates")
    }

    @Test("Depth reduction computes fractional reduction correctly")
    func depthReductionComputesFraction() {
        let baseline = CircuitCost(gateCount: [:], depth: 10, cnotEquivalent: 5, tCount: 2, totalGates: 100)
        let optimized = CircuitCost(gateCount: [:], depth: 8, cnotEquivalent: 3, tCount: 1, totalGates: 50)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        #expect(abs(comparison.depthReduction - 0.2) < tolerance, "Depth reduction should be 0.2 for 20% reduction")
    }

    @Test("Depth reduction returns zero when baseline depth is zero")
    func depthReductionReturnsZeroForZeroBaseline() {
        let baseline = CircuitCost(gateCount: [:], depth: 0, cnotEquivalent: 5, tCount: 2, totalGates: 100)
        let optimized = CircuitCost(gateCount: [:], depth: 5, cnotEquivalent: 3, tCount: 1, totalGates: 50)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        #expect(abs(comparison.depthReduction - 0.0) < tolerance, "Depth reduction should be 0.0 when baseline depth is zero")
    }

    @Test("isImprovement returns true when speedup exceeds one")
    func isImprovementTrueForSpeedupGreaterThanOne() {
        let baseline = CircuitCost(gateCount: [:], depth: 10, cnotEquivalent: 5, tCount: 2, totalGates: 100)
        let optimized = CircuitCost(gateCount: [:], depth: 10, cnotEquivalent: 5, tCount: 2, totalGates: 50)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        #expect(comparison.isImprovement, "isImprovement should be true when speedup > 1.0")
    }

    @Test("isImprovement returns true when gate reduction is positive")
    func isImprovementTrueForPositiveGateReduction() {
        let baseline = CircuitCost(gateCount: [:], depth: 10, cnotEquivalent: 5, tCount: 2, totalGates: 100)
        let optimized = CircuitCost(gateCount: [:], depth: 10, cnotEquivalent: 5, tCount: 2, totalGates: 90)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        #expect(comparison.isImprovement, "isImprovement should be true when gateReduction > 0")
    }

    @Test("isImprovement returns true when depth reduction is positive")
    func isImprovementTrueForPositiveDepthReduction() {
        let baseline = CircuitCost(gateCount: [:], depth: 10, cnotEquivalent: 5, tCount: 2, totalGates: 100)
        let optimized = CircuitCost(gateCount: [:], depth: 8, cnotEquivalent: 5, tCount: 2, totalGates: 100)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        #expect(comparison.isImprovement, "isImprovement should be true when depthReduction > 0")
    }

    @Test("isImprovement returns false when no improvement exists")
    func isImprovementFalseForNoImprovement() {
        let baseline = CircuitCost(gateCount: [:], depth: 10, cnotEquivalent: 5, tCount: 2, totalGates: 100)
        let optimized = CircuitCost(gateCount: [:], depth: 10, cnotEquivalent: 5, tCount: 2, totalGates: 100)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        #expect(!comparison.isImprovement, "isImprovement should be false when baseline equals optimized")
    }

    @Test("isImprovement returns false when optimized is worse")
    func isImprovementFalseForRegression() {
        let baseline = CircuitCost(gateCount: [:], depth: 10, cnotEquivalent: 5, tCount: 2, totalGates: 100)
        let optimized = CircuitCost(gateCount: [:], depth: 15, cnotEquivalent: 8, tCount: 3, totalGates: 150)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        #expect(!comparison.isImprovement, "isImprovement should be false when optimized is worse than baseline")
    }

    @Test("Description contains speedup gate reduction and depth reduction")
    func descriptionContainsAllMetrics() {
        let baseline = CircuitCost(gateCount: [:], depth: 10, cnotEquivalent: 5, tCount: 2, totalGates: 100)
        let optimized = CircuitCost(gateCount: [:], depth: 8, cnotEquivalent: 3, tCount: 1, totalGates: 50)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        let desc = comparison.description
        #expect(desc.contains("speedup"), "Description should contain speedup metric")
        #expect(desc.contains("gateReduction"), "Description should contain gateReduction metric")
        #expect(desc.contains("depthReduction"), "Description should contain depthReduction metric")
    }

    @Test("Equatable conformance compares baseline and optimized")
    func equatableComparesCorrectly() {
        let cost1 = CircuitCost(gateCount: [:], depth: 10, cnotEquivalent: 5, tCount: 2, totalGates: 100)
        let cost2 = CircuitCost(gateCount: [:], depth: 8, cnotEquivalent: 3, tCount: 1, totalGates: 50)
        let comparison1 = BenchmarkComparison(baseline: cost1, optimized: cost2)
        let comparison2 = BenchmarkComparison(baseline: cost1, optimized: cost2)
        let comparison3 = BenchmarkComparison(baseline: cost2, optimized: cost1)

        #expect(comparison1 == comparison2, "Comparisons with same baseline and optimized should be equal")
        #expect(comparison1 != comparison3, "Comparisons with different values should not be equal")
    }

    @Test("Speedup handles equal gate counts")
    func speedupHandlesEqualGates() {
        let baseline = CircuitCost(gateCount: [:], depth: 10, cnotEquivalent: 5, tCount: 2, totalGates: 100)
        let optimized = CircuitCost(gateCount: [:], depth: 10, cnotEquivalent: 5, tCount: 2, totalGates: 100)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        #expect(abs(comparison.speedup - 1.0) < tolerance, "Speedup should be 1.0 when gate counts are equal")
    }

    @Test("Speedup less than one indicates regression")
    func speedupLessThanOneIndicatesRegression() {
        let baseline = CircuitCost(gateCount: [:], depth: 10, cnotEquivalent: 5, tCount: 2, totalGates: 50)
        let optimized = CircuitCost(gateCount: [:], depth: 15, cnotEquivalent: 8, tCount: 3, totalGates: 100)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        #expect(comparison.speedup < 1.0, "Speedup should be less than 1.0 when optimized has more gates")
        #expect(abs(comparison.speedup - 0.5) < tolerance, "Speedup should be 0.5 when optimized has twice the gates")
    }

    @Test("Speedup calculation with CircuitCostEstimator")
    func speedupWithEstimator() {
        var baselineCircuit = QuantumCircuit(qubits: 2)
        for _ in 0 ..< 4 {
            baselineCircuit.append(.hadamard, to: 0)
        }
        var optimizedCircuit = QuantumCircuit(qubits: 2)
        for _ in 0 ..< 2 {
            optimizedCircuit.append(.hadamard, to: 0)
        }
        let baseline = CircuitCostEstimator.estimate(baselineCircuit)
        let optimized = CircuitCostEstimator.estimate(optimizedCircuit)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        #expect(abs(comparison.speedup - 2.0) < tolerance, "Speedup should be 2.0 when baseline has twice the gates of optimized")
    }

    @Test("Gate reduction calculation with CircuitCostEstimator")
    func gateReductionWithEstimator() {
        var baselineCircuit = QuantumCircuit(qubits: 2)
        for _ in 0 ..< 4 {
            baselineCircuit.append(.pauliX, to: 0)
        }
        var optimizedCircuit = QuantumCircuit(qubits: 2)
        for _ in 0 ..< 3 {
            optimizedCircuit.append(.pauliX, to: 0)
        }
        let baseline = CircuitCostEstimator.estimate(baselineCircuit)
        let optimized = CircuitCostEstimator.estimate(optimizedCircuit)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        #expect(abs(comparison.gateReduction - 0.25) < tolerance, "Gate reduction should be 0.25 when reducing from 4 to 3 gates")
    }

    @Test("Depth reduction calculation with CircuitCostEstimator")
    func depthReductionWithEstimator() {
        var baselineCircuit = QuantumCircuit(qubits: 1)
        for _ in 0 ..< 5 {
            baselineCircuit.append(.hadamard, to: 0)
        }
        var optimizedCircuit = QuantumCircuit(qubits: 1)
        for _ in 0 ..< 4 {
            optimizedCircuit.append(.hadamard, to: 0)
        }
        let baseline = CircuitCostEstimator.estimate(baselineCircuit)
        let optimized = CircuitCostEstimator.estimate(optimizedCircuit)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        #expect(abs(comparison.depthReduction - 0.2) < tolerance, "Depth reduction should be 0.2 when reducing depth from 5 to 4")
    }

    @Test("isImprovement true when any metric improves with CircuitCostEstimator")
    func isImprovementWithEstimator() {
        var baselineCircuit = QuantumCircuit(qubits: 2)
        baselineCircuit.append(.hadamard, to: 0)
        baselineCircuit.append(.hadamard, to: 1)
        baselineCircuit.append(.cnot, to: [0, 1])
        var optimizedCircuit = QuantumCircuit(qubits: 2)
        optimizedCircuit.append(.hadamard, to: 0)
        optimizedCircuit.append(.cnot, to: [0, 1])
        let baseline = CircuitCostEstimator.estimate(baselineCircuit)
        let optimized = CircuitCostEstimator.estimate(optimizedCircuit)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        #expect(comparison.isImprovement, "isImprovement should be true when optimized circuit has fewer gates")
    }

    @Test("Description contains formatted values with CircuitCostEstimator")
    func descriptionWithEstimator() {
        var baselineCircuit = QuantumCircuit(qubits: 2)
        baselineCircuit.append(.hadamard, to: 0)
        baselineCircuit.append(.cnot, to: [0, 1])
        var optimizedCircuit = QuantumCircuit(qubits: 2)
        optimizedCircuit.append(.hadamard, to: 0)
        let baseline = CircuitCostEstimator.estimate(baselineCircuit)
        let optimized = CircuitCostEstimator.estimate(optimizedCircuit)
        let comparison = BenchmarkComparison(baseline: baseline, optimized: optimized)

        let desc = comparison.description
        #expect(desc.contains("x"), "Description should contain speedup value with 'x' suffix")
        #expect(desc.contains("%"), "Description should contain percentage values")
    }
}
