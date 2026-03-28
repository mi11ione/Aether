// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Test suite for IQP circuit generation.
/// Validates circuit structure including H-D-H form,
/// diagonal unitary decomposition, and reproducibility.
@Suite("IQP Circuit Generation")
struct IQPCircuitGenerationTests {
    @Test("Random IQP circuit produces valid state")
    func randomCircuitValidState() {
        let circuit = IQPSampling.generateCircuit(qubits: 3, seed: 42)
        let state = circuit.execute()
        #expect(state.isNormalized(), "IQP output state should be normalized")
    }

    @Test("Explicit angles circuit produces valid state")
    func explicitAnglesValidState() {
        let singles = [0.5, 1.2, 0.8]
        let pairs: [(Int, Int, Double)] = [(0, 1, 0.7), (1, 2, 1.1)]
        let circuit = IQPSampling.generateCircuit(qubits: 3, singleAngles: singles, pairAngles: pairs)
        let state = circuit.execute()
        #expect(state.isNormalized(), "Explicit-angle IQP state should be normalized")
    }

    @Test("Seeded random IQP is reproducible")
    func seededReproducibility() {
        let state1 = IQPSampling.generateCircuit(qubits: 3, seed: 99).execute()
        let state2 = IQPSampling.generateCircuit(qubits: 3, seed: 99).execute()

        for i in 0 ..< state1.amplitudes.count {
            #expect(
                abs(state1.amplitudes[i].real - state2.amplitudes[i].real) < 1e-10,
                "Seeded IQP circuits should produce identical amplitudes at index \(i)",
            )
            #expect(
                abs(state1.amplitudes[i].imaginary - state2.amplitudes[i].imaginary) < 1e-10,
                "Seeded IQP circuits should produce identical imaginary parts at index \(i)",
            )
        }
    }

    @Test("Different seeds produce different states")
    func differentSeeds() {
        let state1 = IQPSampling.generateCircuit(qubits: 3, seed: 1).execute()
        let state2 = IQPSampling.generateCircuit(qubits: 3, seed: 2).execute()

        var identical = true
        for i in 0 ..< state1.amplitudes.count {
            if abs(state1.amplitudes[i].real - state2.amplitudes[i].real) > 1e-10 {
                identical = false
                break
            }
        }
        #expect(!identical, "Different seeds should produce different IQP output states")
    }

    @Test("IQP circuit has correct qubit count")
    func correctQubitCount() {
        let circuit = IQPSampling.generateCircuit(qubits: 4, seed: 42)
        #expect(circuit.qubits == 4, "IQP circuit should have 4 qubits")
    }

    @Test("Zero angles produce ground state")
    func zeroAnglesGroundState() {
        let singles = [0.0, 0.0, 0.0]
        let pairs: [(Int, Int, Double)] = [(0, 1, 0.0), (1, 2, 0.0)]
        let circuit = IQPSampling.generateCircuit(qubits: 3, singleAngles: singles, pairAngles: pairs)
        let state = circuit.execute()
        let probs = state.probabilities()

        #expect(abs(probs[0] - 1.0) < 1e-10, "Zero-angle IQP should return to |000>, got p(0) = \(probs[0])")
        for i in 1 ..< probs.count {
            #expect(
                abs(probs[i]) < 1e-10,
                "Zero-angle IQP should have zero probability for state \(i), got \(probs[i])",
            )
        }
    }

    @Test("Explicit angles with all pairs")
    func allPairsExplicit() {
        let singles = [0.3, 0.7]
        let pairs = [(0, 1, 1.5)]
        let circuit = IQPSampling.generateCircuit(qubits: 2, singleAngles: singles, pairAngles: pairs)
        let state = circuit.execute()
        #expect(state.isNormalized(), "All-pairs 2-qubit IQP should produce normalized state")

        let probSum = state.probabilities().reduce(0.0, +)
        #expect(abs(probSum - 1.0) < 1e-10, "Probabilities should sum to 1.0")
    }
}

/// Test suite for collision probability computation.
/// Validates empirical collision probability calculation,
/// uniform threshold, and anti-concentration classification.
@Suite("Collision Probability")
struct CollisionProbabilityTests {
    @Test("Uniform outcomes yield collision near 1/2^n")
    func uniformCollision() {
        var outcomes = [Int]()
        outcomes.reserveCapacity(8000)
        for i in 0 ..< 8000 {
            outcomes.append(i % 8)
        }

        let result = IQPSampling.collisionProbability(outcomes: outcomes, qubits: 3)
        #expect(
            abs(result.collisionProbability - result.uniformThreshold) < 0.01,
            "Perfectly uniform outcomes should have collision near 1/8, got \(result.collisionProbability)",
        )
        #expect(result.isAntiConcentrated, "Uniform distribution should be anti-concentrated")
    }

    @Test("Peaked outcomes yield high collision probability")
    func peakedCollision() {
        let outcomes = [Int](repeating: 0, count: 1000)
        let result = IQPSampling.collisionProbability(outcomes: outcomes, qubits: 3)
        #expect(
            abs(result.collisionProbability - 1.0) < 1e-10,
            "All-same outcomes should have collision probability 1.0, got \(result.collisionProbability)",
        )
        #expect(!result.isAntiConcentrated, "Peaked distribution should not be anti-concentrated")
    }

    @Test("Uniform threshold is 1/2^n")
    func uniformThresholdValue() {
        let outcomes = [0, 1, 2, 3]
        let result = IQPSampling.collisionProbability(outcomes: outcomes, qubits: 3)
        #expect(
            abs(result.uniformThreshold - 0.125) < 1e-10,
            "Uniform threshold for 3 qubits should be 1/8 = 0.125, got \(result.uniformThreshold)",
        )
    }

    @Test("Collision result includes anti-concentration flag")
    func collisionIncludesAntiConcentration() {
        let circuit = IQPSampling.generateCircuit(qubits: 3, seed: 42)
        let outcomes = Measurement.sample(circuit: circuit, shots: 5000, seed: 42)

        let result = IQPSampling.collisionProbability(outcomes: outcomes, qubits: 3)

        let expectedAntiConcentrated = result.collisionProbability <= 3.0 * result.uniformThreshold
        #expect(
            result.isAntiConcentrated == expectedAntiConcentrated,
            "isAntiConcentrated should match bound check: collision=\(result.collisionProbability), threshold=\(result.uniformThreshold)",
        )
    }

    @Test("Two-outcome collision probability")
    func twoOutcomeCollision() {
        var outcomes = [Int]()
        outcomes.reserveCapacity(1000)
        for i in 0 ..< 1000 {
            outcomes.append(i % 2 == 0 ? 0 : 1)
        }

        let result = IQPSampling.collisionProbability(outcomes: outcomes, qubits: 2)
        let expectedCollision = 2.0 * (0.5 * 0.5)
        #expect(
            abs(result.collisionProbability - expectedCollision) < 1e-10,
            "50/50 on 2 states should have collision 0.5, got \(result.collisionProbability)",
        )
    }
}

/// Test suite for IQP convenience evaluation.
/// Validates end-to-end IQP pipeline from circuit generation
/// through sampling to anti-concentration verification.
@Suite("IQP Evaluation")
struct IQPEvaluationTests {
    @Test("Evaluate returns valid collision result")
    func evaluateReturnsValid() {
        let result = IQPSampling.evaluate(qubits: 3, shots: 5000, seed: 42)
        #expect(result.collisionProbability > 0.0, "Collision probability should be positive")
        #expect(result.uniformThreshold > 0.0, "Uniform threshold should be positive")
    }

    @Test("Evaluate collision bounded by 1")
    func evaluateCollisionBound() {
        let result = IQPSampling.evaluate(qubits: 3, shots: 5000, seed: 42)
        #expect(result.collisionProbability <= 1.0, "Collision probability should be at most 1.0")
    }

    @Test("Evaluate with large shots tends toward anti-concentration")
    func evaluateLargeShots() {
        let result = IQPSampling.evaluate(qubits: 3, shots: 10000, seed: 42)
        #expect(
            result.collisionProbability < 1.0,
            "IQP with random angles should not be fully concentrated, got \(result.collisionProbability)",
        )
    }

    @Test("Evaluate uniform threshold matches qubit count")
    func evaluateThresholdMatchesQubits() {
        let result3 = IQPSampling.evaluate(qubits: 3, shots: 1000, seed: 42)
        let result4 = IQPSampling.evaluate(qubits: 4, shots: 1000, seed: 42)

        #expect(abs(result3.uniformThreshold - 1.0 / 8.0) < 1e-10, "3-qubit threshold should be 1/8")
        #expect(abs(result4.uniformThreshold - 1.0 / 16.0) < 1e-10, "4-qubit threshold should be 1/16")
    }

    @Test("Seeded evaluate is reproducible")
    func seededEvaluateReproducible() {
        let result1 = IQPSampling.evaluate(qubits: 3, shots: 1000, seed: 77)
        let result2 = IQPSampling.evaluate(qubits: 3, shots: 1000, seed: 77)

        #expect(
            abs(result1.collisionProbability - result2.collisionProbability) < 1e-10,
            "Seeded evaluate should produce identical collision probabilities",
        )
    }
}
