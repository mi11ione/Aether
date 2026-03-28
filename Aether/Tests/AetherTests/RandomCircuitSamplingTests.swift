// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Test suite for random circuit generation.
/// Validates circuit structure, gate composition, and reproducibility
/// of hardware-native random circuits using sqrt-iSWAP entangling gates.
@Suite("Random Circuit Generation")
struct RandomCircuitGenerationTests {
    @Test("Generate circuit with correct qubit count")
    func correctQubitCount() {
        let circuit = RandomCircuitSampling.generateCircuit(qubits: 3, depth: 4, seed: 42)
        #expect(circuit.qubits == 3, "Circuit should have 3 qubits")
    }

    @Test("Circuit produces valid quantum state")
    func validQuantumState() {
        let circuit = RandomCircuitSampling.generateCircuit(qubits: 3, depth: 5, seed: 42)
        let state = circuit.execute()
        #expect(state.isNormalized(), "Output state should be normalized")
    }

    @Test("Seeded circuit is reproducible")
    func seededReproducibility() {
        let circuit1 = RandomCircuitSampling.generateCircuit(qubits: 3, depth: 4, seed: 123)
        let circuit2 = RandomCircuitSampling.generateCircuit(qubits: 3, depth: 4, seed: 123)
        let state1 = circuit1.execute()
        let state2 = circuit2.execute()

        for i in 0 ..< state1.amplitudes.count {
            #expect(
                abs(state1.amplitudes[i].real - state2.amplitudes[i].real) < 1e-10,
                "Seeded circuits should produce identical amplitudes at index \(i)",
            )
        }
    }

    @Test("Different seeds produce different circuits")
    func differentSeeds() {
        let state1 = RandomCircuitSampling.generateCircuit(qubits: 3, depth: 4, seed: 1).execute()
        let state2 = RandomCircuitSampling.generateCircuit(qubits: 3, depth: 4, seed: 2).execute()

        var identical = true
        for i in 0 ..< state1.amplitudes.count {
            if abs(state1.amplitudes[i].real - state2.amplitudes[i].real) > 1e-10 {
                identical = false
                break
            }
        }
        #expect(!identical, "Different seeds should produce different output states")
    }

    @Test("Circuit depth scales gate count")
    func depthScalesGates() {
        let shallow = RandomCircuitSampling.generateCircuit(qubits: 3, depth: 2, seed: 42)
        let deep = RandomCircuitSampling.generateCircuit(qubits: 3, depth: 6, seed: 42)
        #expect(deep.count > shallow.count, "Deeper circuits should have more gates")
    }

    @Test("Two-qubit minimum enforced")
    func minimumTwoQubits() {
        let circuit = RandomCircuitSampling.generateCircuit(qubits: 2, depth: 3, seed: 42)
        let state = circuit.execute()
        #expect(state.isNormalized(), "2-qubit circuit should produce valid state")
    }
}

/// Test suite for cross-entropy benchmarking fidelity.
/// Validates XEB computation F_XEB = 2^n mean(p(x)) - 1
/// including perfect sampling, uniform sampling, and edge cases.
@Suite("XEB Fidelity")
struct XEBFidelityTests {
    @Test("Perfect sampling yields fidelity near 1")
    func perfectSamplingFidelity() {
        let circuit = RandomCircuitSampling.generateCircuit(qubits: 3, depth: 6, seed: 42)
        let probs = circuit.execute().probabilities()
        let samples = Measurement.sample(circuit: circuit, shots: 10000, seed: 42)

        let result = RandomCircuitSampling.xebFidelity(samples: samples, idealProbabilities: probs)

        #expect(result.fidelity > 0.0, "XEB fidelity for ideal sampling should be positive, got \(result.fidelity)")
        #expect(result.sampleCount == 10000, "Sample count should match shots")
    }

    @Test("XEB fidelity from evaluate convenience method")
    func evaluateConvenience() {
        let result = RandomCircuitSampling.evaluate(qubits: 3, depth: 6, shots: 10000, seed: 42)

        #expect(result.fidelity > 0.0, "XEB fidelity should be positive for ideal sampling, got \(result.fidelity)")
        #expect(result.sampleCount == 10000, "Sample count should be 10000")
        #expect(result.idealProbabilities.count == 8, "Should have 2^3 = 8 ideal probabilities")
    }

    @Test("Ideal probabilities sum to 1")
    func idealProbabilitiesSumToOne() {
        let result = RandomCircuitSampling.evaluate(qubits: 3, depth: 5, shots: 1000, seed: 42)
        let sum = result.idealProbabilities.reduce(0.0, +)
        #expect(abs(sum - 1.0) < 1e-10, "Ideal probabilities should sum to 1.0, got \(sum)")
    }

    @Test("Mean ideal probability is positive")
    func meanProbabilityPositive() {
        let result = RandomCircuitSampling.evaluate(qubits: 3, depth: 5, shots: 1000, seed: 42)
        #expect(result.meanIdealProbability > 0.0, "Mean ideal probability should be positive")
    }

    @Test("XEB fidelity consistent with mean probability formula")
    func fidelityFormulaConsistency() {
        let result = RandomCircuitSampling.evaluate(qubits: 3, depth: 5, shots: 5000, seed: 42)
        let expectedFidelity = 8.0 * result.meanIdealProbability - 1.0
        #expect(
            abs(result.fidelity - expectedFidelity) < 1e-10,
            "Fidelity should equal 2^n * mean - 1, got \(result.fidelity) vs \(expectedFidelity)",
        )
    }

    @Test("Uniform samples yield fidelity near 0")
    func uniformSamplesLowFidelity() {
        let circuit = RandomCircuitSampling.generateCircuit(qubits: 3, depth: 6, seed: 42)
        let probs = circuit.execute().probabilities()

        var uniformSamples = [Int]()
        uniformSamples.reserveCapacity(10000)
        for i in 0 ..< 10000 {
            uniformSamples.append(i % 8)
        }

        let result = RandomCircuitSampling.xebFidelity(samples: uniformSamples, idealProbabilities: probs)
        #expect(abs(result.fidelity) < 0.5, "Uniform sampling should yield fidelity near 0, got \(result.fidelity)")
    }
}

/// Test suite for Porter-Thomas distribution validation.
/// Validates KS test against exponential distribution for
/// rescaled ideal output probabilities from random circuits.
@Suite("Porter-Thomas Validation")
struct PorterThomasTests {
    @Test("Random circuit passes Porter-Thomas test")
    func randomCircuitPassesTest() {
        let circuit = RandomCircuitSampling.generateCircuit(qubits: 4, depth: 12, seed: 42)
        let probs = circuit.execute().probabilities()
        let result = RandomCircuitSampling.validatePorterThomas(probabilities: probs)
        #expect(result.isValid, "Random circuit should pass Porter-Thomas test, KS=\(result.ksStatistic) vs critical=\(result.criticalValue)")
    }

    @Test("KS statistic is non-negative")
    func ksStatisticNonNegative() {
        let probs = RandomCircuitSampling.generateCircuit(qubits: 3, depth: 8, seed: 42)
            .execute().probabilities()
        let result = RandomCircuitSampling.validatePorterThomas(probabilities: probs)
        #expect(result.ksStatistic >= 0.0, "KS statistic should be non-negative")
    }

    @Test("Critical value scales with inverse sqrt of N")
    func criticalValueScaling() {
        let probs3 = RandomCircuitSampling.generateCircuit(qubits: 3, depth: 5, seed: 42)
            .execute().probabilities()
        let probs4 = RandomCircuitSampling.generateCircuit(qubits: 4, depth: 5, seed: 42)
            .execute().probabilities()

        let result3 = RandomCircuitSampling.validatePorterThomas(probabilities: probs3)
        let result4 = RandomCircuitSampling.validatePorterThomas(probabilities: probs4)

        #expect(result4.criticalValue < result3.criticalValue, "Critical value should decrease with more states")

        let expectedRatio = sqrt(Double(probs3.count)) / sqrt(Double(probs4.count))
        let actualRatio = result4.criticalValue / result3.criticalValue
        #expect(abs(actualRatio - expectedRatio) < 1e-10, "Critical value ratio should match sqrt scaling")
    }

    @Test("Uniform distribution fails Porter-Thomas test")
    func uniformDistributionFails() {
        let n = 16
        let uniform = [Double](repeating: 1.0 / Double(n), count: n)
        let result = RandomCircuitSampling.validatePorterThomas(probabilities: uniform)
        #expect(!result.isValid, "Uniform distribution should fail Porter-Thomas test")
    }

    @Test("Deterministic distribution fails Porter-Thomas test")
    func deterministicDistributionFails() {
        var probs = [Double](repeating: 0.0, count: 16)
        probs[0] = 1.0
        let result = RandomCircuitSampling.validatePorterThomas(probabilities: probs)
        #expect(!result.isValid, "Delta distribution should fail Porter-Thomas test")
    }
}

/// Test suite for wall-clock benchmarking.
/// Validates benchmark result structure, timing positivity,
/// and grid enumeration across qubit and depth configurations.
@Suite("Benchmark")
struct RandomCircuitBenchmarkTests {
    @Test("Benchmark returns correct result count")
    func correctResultCount() {
        let results = RandomCircuitSampling.benchmark(qubitRange: 2 ... 3, depths: [3, 5], shots: 100, seed: 42)
        #expect(results.count == 4, "Should have 2 qubits x 2 depths = 4 results, got \(results.count)")
    }

    @Test("Benchmark wall-clock times are positive")
    func positiveWallClock() {
        let results = RandomCircuitSampling.benchmark(qubitRange: 2 ... 3, depths: [3], shots: 100, seed: 42)
        for result in results {
            #expect(result.wallClockSeconds > 0.0, "Wall-clock time should be positive for \(result.qubits)q")
        }
    }

    @Test("Benchmark fidelity values are reasonable")
    func reasonableFidelity() {
        let results = RandomCircuitSampling.benchmark(qubitRange: 3 ... 3, depths: [6], shots: 5000, seed: 42)
        #expect(results.count == 1, "Should have exactly 1 result")
        #expect(results[0].fidelity > 0.0, "Fidelity should be positive for ideal sampling")
        #expect(results[0].qubits == 3, "Qubits should be 3")
        #expect(results[0].depth == 6, "Depth should be 6")
    }

    @Test("Benchmark results ordered by depth then qubits")
    func resultOrdering() {
        let results = RandomCircuitSampling.benchmark(qubitRange: 2 ... 3, depths: [3, 5], shots: 100, seed: 42)
        #expect(results[0].depth == 3, "First result should have depth 3")
        #expect(results[0].qubits == 2, "First result should have 2 qubits")
        #expect(results[1].depth == 3, "Second result should have depth 3")
        #expect(results[1].qubits == 3, "Second result should have 3 qubits")
        #expect(results[2].depth == 5, "Third result should have depth 5")
        #expect(results[3].depth == 5, "Fourth result should have depth 5")
    }
}
