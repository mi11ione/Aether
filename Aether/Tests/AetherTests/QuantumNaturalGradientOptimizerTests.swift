// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Test suite for QuantumNaturalGradientOptimizer configuration.
/// Validates default learning rate, regularization, metric averaging,
/// exact/QNSPSA threshold, and optional circuit parameter handling.
@Suite("QuantumNaturalGradientOptimizer Configuration")
struct QNGOptimizerConfigTests {
    @Test("Create QNG optimizer with defaults")
    func createWithDefaults() {
        let optimizer = QuantumNaturalGradientOptimizer()

        #expect(abs(optimizer.learningRate - 0.01) < 1e-10, "Default learning rate should be 0.01")
        #expect(abs(optimizer.regularization - 1e-4) < 1e-10, "Default regularization should be 1e-4")
        #expect(abs(optimizer.metricAveraging - 0.9) < 1e-10, "Default metric averaging should be 0.9")
        #expect(optimizer.exactMetricThreshold == 50, "Default exact metric threshold should be 50")
    }

    @Test("Create QNG optimizer with custom values")
    func createWithCustomValues() {
        let optimizer = QuantumNaturalGradientOptimizer(
            learningRate: 0.05,
            regularization: 1e-3,
            parameterShift: .pi / 4,
            metricAveraging: 0.95,
            exactMetricThreshold: 30,
        )

        #expect(abs(optimizer.learningRate - 0.05) < 1e-10, "Custom learning rate should be 0.05")
        #expect(abs(optimizer.regularization - 1e-3) < 1e-10, "Custom regularization should be 1e-3")
        #expect(abs(optimizer.parameterShift - .pi / 4) < 1e-10, "Custom parameter shift should be pi/4")
        #expect(abs(optimizer.metricAveraging - 0.95) < 1e-10, "Custom metric averaging should be 0.95")
        #expect(optimizer.exactMetricThreshold == 30, "Custom threshold should be 30")
    }
}

/// Test suite for QuantumNaturalGradientOptimizer QNSPSA minimization.
/// Validates convergence with approximate metric tensor estimation,
/// gradient preconditioning, and natural gradient step quality.
@Suite("QuantumNaturalGradientOptimizer Minimization")
struct QNGOptimizerMinimizeTests {
    @Test("QNSPSA mode minimizes quadratic")
    func qnspsaMinimizesQuadratic() async {
        let optimizer = QuantumNaturalGradientOptimizer(learningRate: 0.1)

        let result = await optimizer.minimize(
            { params in params[0] * params[0] + params[1] * params[1] },
            from: [1.0, 1.0],
            using: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 100),
            progress: nil,
        )

        #expect(result.value < 1.0, "QNG should reduce quadratic objective")
    }

    @Test("QNSPSA mode evaluations include metric estimation")
    func qnspsaEvaluationCount() async {
        let optimizer = QuantumNaturalGradientOptimizer(learningRate: 0.1)

        let result = await optimizer.minimize(
            { params in params[0] * params[0] },
            from: [1.0],
            using: ConvergenceCriteria(energyTolerance: 1e-10, maxIterations: 3),
            progress: nil,
        )

        #expect(result.evaluations >= 3 * (2 + 4 + 1), "Should count gradient (2n) + metric (4) + new value (1) per iteration")
    }

    @Test("Regularization prevents singular metric")
    func regularizationPreventsSignularMetric() async {
        let optimizer = QuantumNaturalGradientOptimizer(
            learningRate: 0.1,
            regularization: 1.0,
        )

        let result = await optimizer.minimize(
            { params in params[0] * params[0] },
            from: [2.0],
            using: ConvergenceCriteria(energyTolerance: 1e-2, maxIterations: 50),
            progress: nil,
        )

        #expect(result.value < 4.0, "Heavy regularization should still allow optimization progress")
    }

    @Test("Progress callback invoked")
    func progressCallbackInvoked() async {
        actor Counter {
            var count = 0
            func increment() {
                count += 1
            }

            func get() -> Int {
                count
            }
        }

        let optimizer = QuantumNaturalGradientOptimizer(learningRate: 0.1)
        let counter = Counter()

        _ = await optimizer.minimize(
            { params in params[0] * params[0] },
            from: [1.0],
            using: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 10),
            progress: { _, _ in await counter.increment() },
        )

        let callCount = await counter.get()
        #expect(callCount > 0, "Progress callback should be invoked")
    }

    @Test("Max iterations termination")
    func maxIterationsTermination() async {
        let optimizer = QuantumNaturalGradientOptimizer(learningRate: 0.0001)

        let result = await optimizer.minimize(
            { params in params[0] * params[0] + params[1] * params[1] },
            from: [10.0, 10.0],
            using: ConvergenceCriteria(energyTolerance: 1e-15, maxIterations: 3),
            progress: nil,
        )

        #expect(result.terminationReason == .maxIterationsReached, "Should hit max iterations with tiny learning rate")
    }

    @Test("Exact metric mode with circuit provided")
    func exactMetricWithCircuit() async {
        var circuitBuilder = QuantumCircuit(qubits: 1)
        circuitBuilder.append(.rotationY(.parameter(Parameter(name: "theta"))), to: 0)
        let circuit = circuitBuilder

        let optimizer = QuantumNaturalGradientOptimizer(
            learningRate: 0.1,
            regularization: 1e-4,
            circuit: circuit,
        )

        let observable = Observable.pauliZ(qubit: 0)
        let result = await optimizer.minimize(
            { params in
                let state = circuit.bound(with: params).execute()
                return observable.expectationValue(of: state)
            },
            from: [0.5],
            using: ConvergenceCriteria(energyTolerance: 1e-3, maxIterations: 50),
            progress: nil,
        )

        #expect(result.iterations > 0, "Exact QNG with circuit should complete at least one iteration")
    }

    @Test("Gradient convergence check")
    func gradientConvergenceCheck() async {
        let optimizer = QuantumNaturalGradientOptimizer(learningRate: 0.5)

        let result = await optimizer.minimize(
            { params in params[0] * params[0] },
            from: [0.001],
            using: ConvergenceCriteria(
                energyTolerance: 1e-15,
                gradientNormTolerance: 1.0,
                maxIterations: 50,
            ),
            progress: nil,
        )

        #expect(
            result.terminationReason == .gradientConverged || result.terminationReason == .energyConverged,
            "Should detect convergence near zero gradient",
        )
    }
}

/// Test suite for GradientMethods.fubiniStudyMetric computation.
/// Validates metric tensor symmetry, positive semi-definiteness,
/// and correct inner product structure from derivative states.
@Suite("Fubini-Study Metric")
struct FubiniStudyMetricTests {
    @Test("Metric for single-parameter circuit")
    func singleParameterMetric() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationY(.parameter(Parameter(name: "theta"))), to: 0)

        let metric = GradientMethods.fubiniStudyMetric(circuit: circuit, parameters: [0.5])

        #expect(metric.count == 1, "Should produce 1x1 metric for single parameter")
        #expect(metric[0].count == 1, "Inner dimension should be 1")
        #expect(metric[0][0] >= -1e-10, "Metric diagonal should be non-negative")
    }

    @Test("Metric symmetry for two-parameter circuit")
    func twoParameterSymmetry() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.rotationY(.parameter(Parameter(name: "b"))), to: 1)

        let metric = GradientMethods.fubiniStudyMetric(circuit: circuit, parameters: [0.5, 0.3])

        #expect(metric.count == 2, "Should produce 2x2 metric")
        #expect(abs(metric[0][1] - metric[1][0]) < 1e-10, "Metric should be symmetric: g_01 == g_10")
    }

    @Test("Metric positive semi-definiteness")
    func positiveSemiDefiniteness() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationY(.parameter(Parameter(name: "theta"))), to: 0)

        let metric = GradientMethods.fubiniStudyMetric(circuit: circuit, parameters: [1.0])

        #expect(metric[0][0] >= -1e-10, "1x1 metric must be non-negative (PSD)")
    }

    @Test("Metric with non-gate operation in circuit")
    func metricWithNonGateOperation() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        circuit.append(.measure, to: 1)
        circuit.append(.rotationZ(.parameter(Parameter(name: "b"))), to: 0)

        let metric = GradientMethods.fubiniStudyMetric(circuit: circuit, parameters: [0.5, 0.3])

        #expect(metric.count == 2, "Should produce 2x2 metric despite non-gate operation in circuit")
        #expect(abs(metric[0][1] - metric[1][0]) < 1e-10, "Metric should still be symmetric")
    }

    @Test("Empty circuit returns empty metric")
    func emptyCircuitMetric() {
        let circuit = QuantumCircuit(qubits: 1)
        let metric = GradientMethods.fubiniStudyMetric(circuit: circuit, parameters: [])

        #expect(metric.isEmpty, "No-parameter circuit should return empty metric")
    }
}
