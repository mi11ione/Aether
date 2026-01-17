// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for quantum circuit building operations.
/// Validates gate appending, insertion, removal, and circuit structure
/// maintenance for arbitrary qubit counts.
@Suite("Circuit Building")
struct CircuitBuildingTests {
    @Test("Create empty circuit")
    func createEmptyCircuit() {
        let circuit = QuantumCircuit(qubits: 2)
        #expect(circuit.qubits == 2)
        #expect(circuit.count == 0)
        #expect(circuit.isEmpty)
    }

    @Test("Append single gate")
    func appendSingleGate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        #expect(circuit.count == 1)
        #expect(!circuit.isEmpty)
    }

    @Test("Append multiple gates")
    func appendMultipleGates() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        #expect(circuit.count == 2)
    }

    @Test("Insert gate at position")
    func insertGateAtPosition() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.insert(.pauliZ, to: 0, at: 1)

        #expect(circuit.count == 3)
        #expect(circuit.gates[1].gate == .pauliZ)
    }

    @Test("Remove gate")
    func removeGate() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 0)

        circuit.remove(at: 0)

        #expect(circuit.count == 1)
        #expect(circuit.gates[0].gate == .pauliX)
    }

    @Test("Clear circuit")
    func clearCircuit() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)

        circuit.removeAll()

        #expect(circuit.isEmpty)
        #expect(circuit.count == 0)
    }
}

/// Test suite for quantum circuit execution.
/// Validates sequential gate application, state transformation correctness,
/// and normalization preservation through complex circuit sequences.
@Suite("Circuit Execution")
struct CircuitExecutionTests {
    @Test("Empty circuit returns unchanged state")
    func emptyCircuitUnchanged() {
        let circuit = QuantumCircuit(qubits: 1)
        let initialState = QuantumState(qubit: 0)
        let finalState = circuit.execute(on: initialState)

        #expect(initialState == finalState)
    }

    @Test("Single-gate circuit")
    func singleGateCircuit() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)

        let initialState = QuantumState(qubit: 0)
        let finalState = circuit.execute(on: initialState)

        #expect(abs(finalState.amplitude(of: 1).real - 1.0) < 1e-10)
    }

    @Test("Multi-gate circuit")
    func multiGateCircuit() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliZ, to: 0)
        circuit.append(.hadamard, to: 0)

        let initialState = QuantumState(qubit: 0)
        let finalState = circuit.execute(on: initialState)

        #expect(abs(finalState.amplitude(of: 1).real - 1.0) < 1e-10)
    }

    @Test("Execute from default |0⟩ state")
    func executeFromDefaultState() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        let finalState = circuit.execute()

        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(finalState.amplitude(of: 0).real - invSqrt2) < 1e-10)
        #expect(abs(finalState.amplitude(of: 1).real - invSqrt2) < 1e-10)
    }
}

/// Test suite for step-through circuit execution.
/// Validates partial execution for animation, debugging, and state inspection,
/// enabling quantum algorithm visualization and educational interfaces.
@Suite("Step-Through Execution")
struct StepThroughTests {
    @Test("Execute up to index 0 (no gates)")
    func executeUpToZero() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        let initialState = QuantumState(qubit: 0)
        let finalState = circuit.execute(on: initialState, upToIndex: 0)

        #expect(initialState == finalState)
    }

    @Test("Execute up to middle of circuit")
    func executeUpToMiddle() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliZ, to: 0)
        circuit.append(.hadamard, to: 0)

        let initialState = QuantumState(qubit: 0)
        let state1 = circuit.execute(on: initialState, upToIndex: 1)
        let invSqrt2 = 1.0 / sqrt(2.0)

        #expect(abs(state1.amplitude(of: 0).real - invSqrt2) < 1e-10)
    }

    @Test("Step through entire circuit")
    func stepThroughEntireCircuit() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 0)

        let initialState = QuantumState(qubit: 0)

        let state0 = circuit.execute(on: initialState, upToIndex: 0)
        let state1 = circuit.execute(on: initialState, upToIndex: 1)
        let state2 = circuit.execute(on: initialState, upToIndex: 2)

        #expect(state0 == initialState)
        #expect(abs(state1.amplitude(of: 1).real - 1.0) < 1e-10)
        #expect(abs(state2.amplitude(of: 0).real - 1.0) < 1e-10)
    }
}

/// Test suite for Bell state circuit factory.
/// Validates maximally entangled two-qubit state creation,
/// fundamental for quantum teleportation and superdense coding.
@Suite("Bell State Circuit")
struct BellStateCircuitTests {
    @Test("Bell state circuit creates entanglement")
    func bellStateCreatesEntanglement() {
        let circuit = QuantumCircuit.bell()
        let finalState = circuit.execute()
        let invSqrt2 = 1.0 / sqrt(2.0)

        #expect(abs(finalState.amplitude(of: 0).real - invSqrt2) < 1e-10)
        #expect(abs(finalState.amplitude(of: 3).real - invSqrt2) < 1e-10)
        #expect(abs(finalState.amplitude(of: 1).magnitude) < 1e-10)
        #expect(abs(finalState.amplitude(of: 2).magnitude) < 1e-10)
    }

    @Test("Bell state is normalized")
    func bellStateNormalized() {
        let circuit = QuantumCircuit.bell()
        let finalState = circuit.execute()

        #expect(finalState.isNormalized())
    }

    @Test("Bell state has correct probabilities")
    func bellStateProbabilities() {
        let circuit = QuantumCircuit.bell()
        let finalState = circuit.execute()

        #expect(abs(finalState.probability(of: 0) - 0.5) < 1e-10)
        #expect(abs(finalState.probability(of: 3) - 0.5) < 1e-10)
    }
}

/// Test suite for GHZ state circuit factory.
/// Validates multi-qubit Greenberger-Horne-Zeilinger state creation,
/// demonstrating genuine multipartite entanglement beyond Bell states.
@Suite("GHZ State Circuit")
struct GHZStateCircuitTests {
    @Test("3-qubit GHZ state")
    func threeQubitGHZ() {
        let circuit = QuantumCircuit.ghz(qubits: 3)
        let finalState = circuit.execute()

        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(finalState.amplitude(of: 0).real - invSqrt2) < 1e-10)
        #expect(abs(finalState.amplitude(of: 7).real - invSqrt2) < 1e-10)

        for i in 1 ..< 7 {
            #expect(abs(finalState.amplitude(of: i).magnitude) < 1e-10)
        }
    }

    @Test("4-qubit GHZ state")
    func fourQubitGHZ() {
        let circuit = QuantumCircuit.ghz(qubits: 4)
        let finalState = circuit.execute()

        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(finalState.amplitude(of: 0).real - invSqrt2) < 1e-10)
        #expect(abs(finalState.amplitude(of: 15).real - invSqrt2) < 1e-10)
    }
}

/// Test suite for superposition circuit factory.
/// Validates equal superposition state creation across all computational basis states,
/// essential for quantum search and amplitude estimation algorithms.
@Suite("Superposition Circuit")
struct SuperpositionCircuitTests {
    @Test("Single qubit superposition")
    func singleQubitSuperposition() {
        let circuit = QuantumCircuit.uniformSuperposition(qubits: 1)
        let finalState = circuit.execute()

        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(finalState.amplitude(of: 0).real - invSqrt2) < 1e-10)
        #expect(abs(finalState.amplitude(of: 1).real - invSqrt2) < 1e-10)
    }

    @Test("2-qubit equal superposition")
    func twoQubitSuperposition() {
        let circuit = QuantumCircuit.uniformSuperposition(qubits: 2)
        let finalState = circuit.execute()

        let expectedProb = 0.25
        for i in 0 ..< 4 {
            #expect(abs(finalState.probability(of: i) - expectedProb) < 1e-10)
        }
    }

    @Test("3-qubit equal superposition")
    func threeQubitSuperposition() {
        let circuit = QuantumCircuit.uniformSuperposition(qubits: 3)
        let finalState = circuit.execute()

        let expectedProb = 1.0 / 8.0
        for i in 0 ..< 8 {
            #expect(abs(finalState.probability(of: i) - expectedProb) < 1e-10)
        }
    }
}

/// Test suite for quantum circuit properties.
/// Validates circuit metadata consistency including qubit count maintenance,
/// gate count tracking, and structural invariants during circuit modification.
@Suite("Circuit Properties")
struct CircuitPropertiesTests {
    @Test("Circuit maintains qubit count")
    func maintainsQubitCount() {
        let circuit = QuantumCircuit(qubits: 5)
        #expect(circuit.qubits == 5)
    }

    @Test("Gate count updates correctly")
    func countUpdates() {
        var circuit = QuantumCircuit(qubits: 2)
        #expect(circuit.count == 0)

        circuit.append(.hadamard, to: 0)
        #expect(circuit.count == 1)

        circuit.append(.hadamard, to: 1)
        #expect(circuit.count == 2)

        circuit.remove(at: 0)
        #expect(circuit.count == 1)
    }
}

/// Test suite for quantum circuit equality comparison.
/// Validates Equatable conformance for circuit comparison operations,
/// ensuring identical gate sequences are recognized as equal while different configurations are distinguished.
@Suite("Circuit Equality")
struct CircuitEqualityTests {
    @Test("Identical circuits are equal")
    func identicalCircuitsEqual() {
        var circuit1 = QuantumCircuit(qubits: 2)
        circuit1.append(.hadamard, to: 0)

        var circuit2 = QuantumCircuit(qubits: 2)
        circuit2.append(.hadamard, to: 0)

        #expect(circuit1 == circuit2)
    }

    @Test("Different gate sequences are not equal")
    func differentSequencesNotEqual() {
        var circuit1 = QuantumCircuit(qubits: 2)
        circuit1.append(.hadamard, to: 0)

        var circuit2 = QuantumCircuit(qubits: 2)
        circuit2.append(.pauliX, to: 0)

        #expect(circuit1 != circuit2)
    }
}

/// Test suite for quantum circuit string representation.
/// Validates CustomStringConvertible implementation for circuit debugging
/// and educational quantum circuit visualization with gate count and structure summaries.
@Suite("Circuit String Representation")
struct CircuitDescriptionTests {
    @Test("Empty circuit description")
    func emptyCircuitDescription() {
        let circuit = QuantumCircuit(qubits: 2)
        let desc = circuit.description

        #expect(desc.contains("2 qubits"))
        #expect(desc.contains("empty"))
    }

    @Test("Non-empty circuit description")
    func nonEmptyCircuitDescription() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)

        let desc = circuit.description
        #expect(desc.contains("2 qubits"))
        #expect(desc.contains("1 gates"))
    }
}

/// Test suite for quantum circuit scalability.
/// Validates circuit performance and correctness across varying qubit counts,
/// demonstrating robust quantum simulation capabilities from small to large system sizes.
@Suite("Circuit Scalability")
struct CircuitScalabilityTests {
    @Test("Circuit works with 8 qubits")
    func eightQubitCircuit() {
        let circuit = QuantumCircuit.uniformSuperposition(qubits: 8)
        let finalState = circuit.execute()

        #expect(finalState.qubits == 8)
        #expect(finalState.isNormalized())
    }

    @Test("Circuit works with 12 qubits")
    func twelveQubitCircuit() {
        var circuit = QuantumCircuit(qubits: 12)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 5])

        let finalState = circuit.execute()
        #expect(finalState.qubits == 12)
        #expect(finalState.isNormalized())
    }

    @Test("Large circuit with many gates")
    func largeCircuit() {
        var circuit = QuantumCircuit(qubits: 4)

        for _ in 0 ..< 5 {
            for q in 0 ..< 4 {
                circuit.append(.hadamard, to: q)
            }
        }

        #expect(circuit.count == 20)

        let finalState = circuit.execute()
        #expect(finalState.isNormalized())
    }
}

/// Test suite for Quantum Fourier Transform circuit factory.
/// Validates QFT algorithm implementation, the cornerstone of Shor's algorithm
/// and quantum phase estimation, transforming between computational and frequency domains.
@Suite("QFT Circuit")
struct QFTCircuitTests {
    @Test("QFT circuit creates valid structure")
    func qftCircuitStructure() {
        let circuit = QuantumCircuit.qft(qubits: 3)

        #expect(circuit.qubits == 3)
        #expect(circuit.count > 0)
    }

    @Test("QFT preserves normalization")
    func qftPreservesNormalization() {
        let circuit = QuantumCircuit.qft(qubits: 3)
        let finalState = circuit.execute()

        #expect(finalState.isNormalized())
    }

    @Test("QFT on |0⟩ state")
    func qftOnZeroState() {
        let circuit = QuantumCircuit.qft(qubits: 3)
        let finalState = circuit.execute()

        let expectedAmp = 1.0 / sqrt(8.0)

        for i in 0 ..< 8 {
            let amp = finalState.amplitude(of: i)
            #expect(abs(amp.magnitude - expectedAmp) < 1e-10)
        }
    }

    @Test("Inverse QFT reverses QFT")
    func inverseQFTReversesQFT() {
        var amplitudes = [Complex<Double>](repeating: .zero, count: 8)
        amplitudes[5] = .one
        let initialState = QuantumState(qubits: 3, amplitudes: amplitudes)

        let qft = QuantumCircuit.qft(qubits: 3)
        let afterQFT = qft.execute(on: initialState)

        let inverseQFT = QuantumCircuit.inverseQFT(qubits: 3)
        let finalState = inverseQFT.execute(on: afterQFT)

        #expect(abs(finalState.amplitude(of: 5).real - 1.0) < 1e-8)

        for i in 0 ..< 8 {
            if i != 5 {
                #expect(abs(finalState.amplitude(of: i).magnitude) < 1e-8)
            }
        }
    }

    @Test("QFT gate count")
    func qftcount() {
        let n = 4
        let circuit = QuantumCircuit.qft(qubits: n)

        let expectedCount = n + (n * (n - 1)) / 2 + (n / 2)
        #expect(circuit.count == expectedCount)
    }

    @Test("QFT works with different qubit counts")
    func qftDifferentSizes() {
        for qubits in 2 ... 6 {
            let circuit = QuantumCircuit.qft(qubits: qubits)
            let finalState = circuit.execute()

            #expect(finalState.qubits == qubits)
            #expect(finalState.isNormalized())
        }
    }

    @Test("QFT transforms basis states")
    func qftTransformsBasisStates() {
        let n = 3

        for k in 0 ..< (1 << n) {
            var amplitudes = [Complex<Double>](repeating: .zero, count: 1 << n)
            amplitudes[k] = .one
            let initialState = QuantumState(qubits: n, amplitudes: amplitudes)

            let qft = QuantumCircuit.qft(qubits: n)
            let finalState = qft.execute(on: initialState)

            #expect(finalState.isNormalized())

            let expectedMag = 1.0 / sqrt(Double(1 << n))
            for i in 0 ..< (1 << n) {
                let amp = finalState.amplitude(of: i)
                #expect(abs(amp.magnitude - expectedMag) < 1e-9)
            }
        }
    }
}

/// Test suite for state caching in step-through execution.
/// Validates intermediate state caching for efficient scrubbing through quantum circuits,
/// enabling smooth animation and debugging without recomputing from scratch.
@Suite("State Caching")
struct StateCachingTests {
    @Test("Cached step execution matches full execution")
    func cachedStepMatchesFullExecution() {
        var circuit = QuantumCircuit(qubits: 3)
        for i in 0 ..< 12 {
            let qubit = i % 3
            circuit.append(.hadamard, to: qubit)
        }

        let initialState = QuantumState(qubits: 3)

        let fullState = circuit.execute(on: initialState)
        let stepState = circuit.execute(on: initialState, upToIndex: circuit.count)

        #expect(fullState == stepState)
    }

    @Test("Multiple step executions benefit from cache")
    func multipleStepExecutions() {
        var circuit = QuantumCircuit(qubits: 2)

        for _ in 0 ..< 5 {
            circuit.append(.hadamard, to: 0)
            circuit.append(.hadamard, to: 1)
            circuit.append(.cnot, to: [0, 1])
            circuit.append(.hadamard, to: 0)
        }

        let initialState = QuantumState(qubits: 2)

        let state5 = circuit.execute(on: initialState, upToIndex: 5)
        let state10 = circuit.execute(on: initialState, upToIndex: 10)
        let state15 = circuit.execute(on: initialState, upToIndex: 15)
        let state20 = circuit.execute(on: initialState, upToIndex: 20)

        #expect(state5.isNormalized())
        #expect(state10.isNormalized())
        #expect(state15.isNormalized())
        #expect(state20.isNormalized())
    }

    @Test("Cache invalidates on circuit modification")
    func cacheInvalidatesOnModification() {
        var circuit = QuantumCircuit(qubits: 2)

        for _ in 0 ..< 10 {
            circuit.append(.hadamard, to: 0)
        }

        let initialState = QuantumState(qubits: 2)

        _ = circuit.execute(on: initialState, upToIndex: 10)
        circuit.append(.pauliX, to: 1)

        let finalState = circuit.execute(on: initialState, upToIndex: 11)
        #expect(finalState.isNormalized())
    }

    @Test("Backward scrubbing works correctly")
    func backwardScrubbingWorks() {
        var circuit = QuantumCircuit(qubits: 2)

        for _ in 0 ..< 15 {
            circuit.append(.hadamard, to: 0)
        }

        let initialState = QuantumState(qubits: 2)

        _ = circuit.execute(on: initialState, upToIndex: 15)
        let state10 = circuit.execute(on: initialState, upToIndex: 10)
        let state5 = circuit.execute(on: initialState, upToIndex: 5)

        #expect(state10.isNormalized())
        #expect(state5.isNormalized())
    }

    @Test("Cache works with empty upToIndex")
    func cacheWorksWithZeroIndex() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let initialState = QuantumState(qubits: 2)
        let state = circuit.execute(on: initialState, upToIndex: 0)

        #expect(state == initialState)
    }

    @Test("Cache with QFT circuit")
    func cacheWithQFTCircuit() {
        let circuit = QuantumCircuit.qft(qubits: 4)
        let initialState = QuantumState(qubits: 4)

        let midState = circuit.execute(on: initialState, upToIndex: circuit.count / 2)
        let finalState = circuit.execute(on: initialState, upToIndex: circuit.count)

        #expect(midState.isNormalized())
        #expect(finalState.isNormalized())
    }
}

/// Test suite for Grover search algorithm circuit factory.
/// Validates quantum search implementation providing O(√N) speedup over classical O(N),
/// testing oracle construction, diffusion operators, and amplitude amplification.
@Suite("Grover Circuit")
struct GroverCircuitTests {
    @Test("Grover circuit structure is valid")
    func groverCircuitStructure() {
        let circuit = QuantumCircuit.grover(qubits: 3, target: 5)

        #expect(circuit.qubits == 3)
        #expect(circuit.count > 0)

        let firstOps = circuit.gates.prefix(3)
        #expect(firstOps.allSatisfy { op in
            if case .hadamard = op.gate { return true }
            return false
        })
    }

    @Test("Grover preserves normalization")
    func groverPreservesNormalization() {
        let circuit = QuantumCircuit.grover(qubits: 2, target: 3)
        let initialState = QuantumState(qubits: circuit.qubits)
        let finalState = circuit.execute(on: initialState)

        #expect(finalState.isNormalized())
    }

    @Test("Grover finds target with high probability (2-qubit)")
    func groverFindsTarget2Qubit() {
        let target = 2
        let circuit = QuantumCircuit.grover(qubits: 2, target: target)

        let initialState = QuantumState(qubits: circuit.qubits)
        let finalState = circuit.execute(on: initialState)

        let targetProb = finalState.probability(of: target)

        #expect(targetProb > 0.8, "Target probability: \(targetProb)")
    }

    @Test("Grover finds target with high probability (3-qubit)")
    func groverFindsTarget3Qubit() {
        let target = 5
        let circuit = QuantumCircuit.grover(qubits: 3, target: target)

        let initialState = QuantumState(qubits: circuit.qubits)
        let finalState = circuit.execute(on: initialState)

        let targetProb = finalState.probability(of: target)

        #expect(targetProb > 0.8, "Target probability: \(targetProb)")
    }

    @Test("Grover optimal iteration count")
    func groverOptimalIterations() {
        let circuit2 = QuantumCircuit.grover(qubits: 2, target: 0)
        let circuit3 = QuantumCircuit.grover(qubits: 3, target: 0)

        #expect(circuit2.count > 0)
        #expect(circuit3.count > circuit2.count)
    }

    @Test("Grover with custom iterations")
    func groverCustomIterations() {
        let circuit = QuantumCircuit.grover(qubits: 2, target: 1, iterations: 2)
        let initialState = QuantumState(qubits: circuit.qubits)
        let finalState = circuit.execute(on: initialState)

        #expect(finalState.isNormalized())
    }

    @Test("Grover searches different targets correctly")
    func groverDifferentTargets() {
        let qubits = 3

        for target in 0 ..< (1 << qubits) {
            let circuit = QuantumCircuit.grover(qubits: qubits, target: target)
            let initialState = QuantumState(qubits: circuit.qubits)
            let finalState = circuit.execute(on: initialState)

            let targetProb = finalState.probability(of: target)

            #expect(
                targetProb > 0.7,
                "Target \(target) probability: \(targetProb)",
            )
        }
    }

    @Test("Grover with 1 qubit")
    func grover1Qubit() {
        let circuit = QuantumCircuit.grover(qubits: 1, target: 1)
        let initialState = QuantumState(qubits: circuit.qubits)
        let finalState = circuit.execute(on: initialState)

        #expect(finalState.isNormalized())

        let prob = finalState.probability(of: 1)
        #expect(prob > 0.49)
    }

    @Test("Grover statistical validation")
    func groverStatisticalValidation() {
        let target = 3
        let circuit = QuantumCircuit.grover(qubits: 2, target: target)

        let results = Measurement.sample(circuit: circuit, shots: 1000)

        let targetCount = results.count(where: { $0 == target })
        let targetFrequency = Double(targetCount) / 1000.0

        #expect(targetFrequency > 0.7, "Target frequency: \(targetFrequency)")
    }

    @Test("Quantum annealing circuit creation")
    func createAnnealingCircuit() {
        let problem = QuantumCircuit.IsingProblem.quadraticMinimum(qubits: 3)
        let circuit = QuantumCircuit.annealing(qubits: 3, problem: problem)

        #expect(circuit.qubits == 3)
        #expect(circuit.count > 0, "Annealing circuit should have gates")
    }

    @Test("Annealing circuit execution")
    func executeAnnealingCircuit() {
        let problem = QuantumCircuit.IsingProblem.quadraticMinimum(qubits: 2)
        let circuit = QuantumCircuit.annealing(qubits: 2, problem: problem, annealingSteps: 5)

        let finalState = circuit.execute()
        #expect(finalState.isNormalized(), "Final state should be normalized")

        var outcomes: [Int] = []
        for _ in 0 ..< 100 {
            let result = Measurement.measure(finalState)
            outcomes.append(result.outcome)
        }

        let uniqueOutcomes = Set(outcomes)
        #expect(uniqueOutcomes.count > 1, "Should explore multiple states with partial annealing")
    }

    @Test("Annealing finds optimal solution")
    func annealingFindsOptimalSolution() {
        let problem = QuantumCircuit.IsingProblem.quadraticMinimum(qubits: 2)
        let circuit = QuantumCircuit.annealing(qubits: 2, problem: problem, annealingSteps: 20)

        var outcomes: [Int] = []
        for _ in 0 ..< 1000 {
            let finalState = circuit.execute()
            let result = Measurement.measure(finalState)
            outcomes.append(result.outcome)
        }

        var counts = [Int: Int]()
        for outcome in outcomes {
            counts[outcome, default: 0] += 1
        }

        let mostFrequent = counts.max(by: { $0.value < $1.value })?.key
        let frequency = Double(counts[mostFrequent!]!) / Double(outcomes.count)

        #expect(mostFrequent == 3, "Optimal solution should be |11⟩ (state 3), got \(mostFrequent)")
        #expect(frequency > 0.5, "Optimal solution frequency: \(frequency) (should be >0.5)")
    }

    @Test("Annealing convenience method with couplings parameter")
    func annealingConvenienceMethod() {
        let couplings = ["0-1": 0.5, "1-2": -0.3]
        let circuit = QuantumCircuit.annealing(qubits: 3, couplings: couplings, annealingSteps: 10)

        #expect(circuit.qubits == 3)
        #expect(circuit.count > 0)

        let finalState = circuit.execute()
        #expect(finalState.isNormalized())
    }

    @Test("Ising problem from dictionary with local fields")
    func isingProblemFromDictionaryWithLocalFields() {
        let dictionary: [String: Double] = [
            "0": 0.5,
            "1": -0.3,
            "2": 0.8,
            "0-1": 0.2,
            "12": 0.4,
        ]

        let problem = QuantumCircuit.IsingProblem(fromDictionary: dictionary, qubits: 3)

        #expect(problem.localFields[0] == 0.5)
        #expect(problem.localFields[1] == -0.3)
        #expect(problem.localFields[2] == 0.8)

        #expect(problem.couplings[0][1] == 0.2)
        #expect(problem.couplings[1][0] == 0.2)
        #expect(problem.couplings[1][2] == 0.4)
        #expect(problem.couplings[2][1] == 0.4)

        #expect(problem.transverseField.allSatisfy { $0 == 1.0 })
    }

    @Test("Ising problem dictionary supports multiple key formats")
    func isingProblemDictionaryKeyFormats() {
        let dictionary: [String: Double] = [
            "01": 0.1,
            "1-2": 0.2,
            "0,2": 0.3,
        ]

        let problem = QuantumCircuit.IsingProblem(fromDictionary: dictionary, qubits: 3)

        #expect(problem.couplings[0][1] == 0.1)
        #expect(problem.couplings[1][2] == 0.2)
        #expect(problem.couplings[0][2] == 0.3)
    }

    @Test("Ising problem creation")
    func createIsingProblem() {
        let localFields = [1.0, -0.5, 0.0]
        let couplings = [
            [0.0, 0.5, -0.2],
            [0.5, 0.0, 0.1],
            [-0.2, 0.1, 0.0],
        ]

        let problem = QuantumCircuit.IsingProblem(localFields: localFields, couplings: couplings)

        #expect(problem.localFields == localFields)
        #expect(problem.couplings == couplings)
        #expect(problem.transverseField.count == 3)
        #expect(problem.transverseField.allSatisfy { $0 == 1.0 })
    }

    @Test("Quadratic minimum problem")
    func createQuadraticProblem() {
        let problem = QuantumCircuit.IsingProblem.quadraticMinimum(qubits: 3)

        #expect(problem.localFields.count == 3)
        #expect(problem.couplings.count == 3)
        #expect(problem.couplings[0][1] == 0.5)
        #expect(problem.couplings[1][2] == 0.5)
        #expect(problem.couplings[0][2] == 0.0)
        #expect(problem.localFields[0] == -1.0)
        #expect(problem.localFields[1] == -2.0)
        #expect(problem.localFields[2] == -4.0)
    }
}

/// Test suite for uncovered QuantumCircuit initialization and operations.
/// Validates circuit construction with predefined operations, timestamp annotations,
/// validation edge cases, and multi-controlled gate decomposition strategies.
@Suite("Circuit Coverage")
struct CircuitCoverageTests {
    @Test("Initialize circuit with predefined operations")
    func initWithOperations() {
        let gates = [
            Gate(.hadamard, to: 0),
            Gate(.pauliX, to: 1),
            Gate(.cnot, to: [0, 1]),
        ]

        let circuit = QuantumCircuit(qubits: 2, gates: gates)

        #expect(circuit.qubits == 2)
        #expect(circuit.count == 3)
        #expect(circuit.gates[0].gate == .hadamard)
    }

    @Test("Append gate with timestamp")
    func appendGateWithTimestamp() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0, timestamp: 1.5)
        circuit.append(.pauliX, to: 1, timestamp: 3.0)

        #expect(circuit.count == 2)
        #expect(circuit.gates[0].timestamp == 1.5)
        #expect(circuit.gates[1].timestamp == 3.0)
    }

    @Test("Gate description with timestamp")
    func GateDescriptionWithTimestamp() {
        let op1 = Gate(.hadamard, to: 0, timestamp: 2.5)
        let op2 = Gate(.cnot, to: [0, 1], timestamp: 1.25)

        #expect(op1.description.contains("2.5s"))
        #expect(op1.description.contains("s"))
        #expect(op2.description.contains("1.25"))
    }

    @Test("Gate description without timestamp")
    func GateDescriptionNoTimestamp() {
        let op = Gate(.hadamard, to: 0)
        #expect(!op.description.contains("@"))
        #expect(op.description.contains("H"))
    }

    @Test("Multi-controlled Z on 4 qubits")
    func multiControlledZ4Qubits() {
        let circuit = QuantumCircuit.qft(qubits: 4)

        #expect(circuit.qubits == 4)
        #expect(circuit.count > 0)
    }

    @Test("Multi-controlled X with one control")
    func multiControlledXOneControl() {
        let circuit = QuantumCircuit.grover(qubits: 2, target: 0)

        #expect(circuit.count > 0)
    }

    @Test("Multi-controlled X with three+ controls")
    func multiControlledXMultipleControls() {
        let circuit = QuantumCircuit.grover(qubits: 4, target: 5)

        #expect(circuit.qubits >= 4)
    }

    @Test("maxQubitUsed with empty operations returns qubits minus 1")
    func maxQubitUsedEmptyCircuit() {
        let circuit = QuantumCircuit(qubits: 5)
        #expect(circuit.highestQubitIndex == 4)
    }

    @Test("maxQubitUsed detects single-qubit gate qubits")
    func maxQubitUsedSingleQubitGates() {
        var circuit = QuantumCircuit(qubits: 3)

        circuit.append(.hadamard, to: 2)
        #expect(circuit.highestQubitIndex == 2)

        circuit.append(.pauliX, to: 1)
        #expect(circuit.highestQubitIndex == 2)

        circuit.append(.phase(.pi), to: 0)
        #expect(circuit.highestQubitIndex == 2)
    }

    @Test("Grover finds target with high probability (5-qubit)")
    func groverFindsTarget5Qubit() {
        let target = 13
        let circuit = QuantumCircuit.grover(qubits: 5, target: target)
        let initialState = QuantumState(qubits: circuit.qubits)
        let finalState = circuit.execute(on: initialState)
        let targetProb = finalState.probability(of: target)

        #expect(targetProb > 0.7, "Target probability: \(targetProb)")
    }

    @Test("Circuit description with more than 5 gates")
    func circuitDescriptionLongCircuit() {
        var circuit = QuantumCircuit(qubits: 2)

        for _ in 0 ..< 7 {
            circuit.append(.hadamard, to: 0)
        }

        let desc = circuit.description
        #expect(desc.contains("..."), "Description should contain '...' for circuits with >5 gates")
        #expect(desc.contains("7 gates"))
    }

    @Test("maxQubitUsed with single-qubit gate having empty qubits array")
    func maxQubitUsedEmptyQubitArray() {
        var circuit = QuantumCircuit(qubits: 3)
        let ops = [Gate(.hadamard, to: [])]
        circuit = QuantumCircuit(qubits: 3, gates: ops)

        #expect(circuit.highestQubitIndex == 2)
    }

    @Test("Gate description with empty qubits array omits qubit info")
    func gateDescriptionEmptyQubits() {
        let gate = Gate(.hadamard, to: [])
        let desc = gate.description

        #expect(!desc.contains("on qubits"), "Empty qubits should not show 'on qubits' text")
        #expect(desc.contains("H"), "Description should still contain gate name")
    }

    @Test("Init with gates exceeding initial qubit count updates maxQubitUsed")
    func initWithHighQubitGates() {
        let gates = [
            Gate(.hadamard, to: 0),
            Gate(.pauliX, to: 5),
            Gate(.pauliZ, to: 3),
        ]

        let circuit = QuantumCircuit(qubits: 2, gates: gates)

        #expect(circuit.highestQubitIndex == 5, "Max qubit should be 5 from gate on qubit 5")
    }

    @Test("Append with empty qubits array handles gracefully")
    func appendEmptyQubitsArray() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: [])

        #expect(circuit.count == 1, "Gate should be appended")
        #expect(circuit.gates[0].qubits.isEmpty, "Gate should have empty qubits")
        #expect(circuit.highestQubitIndex == 2, "Max qubit should remain qubits-1")
    }

    @Test("Insert with empty qubits array handles gracefully")
    func insertEmptyQubitsArray() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.pauliX, to: 1)
        circuit.insert(.hadamard, to: [], at: 0)

        #expect(circuit.count == 2, "Both gates should be present")
        #expect(circuit.gates[0].qubits.isEmpty, "Inserted gate should have empty qubits")
        #expect(circuit.highestQubitIndex == 2, "Max qubit should be qubits-1=2 (baseline)")
    }

    @Test("Remove gate with empty qubits triggers recompute correctly")
    func removeEmptyQubitsGateThenRecompute() {
        let gates = [
            Gate(.hadamard, to: []),
            Gate(.pauliX, to: 1),
        ]
        var circuit = QuantumCircuit(qubits: 3, gates: gates)
        #expect(circuit.highestQubitIndex == 2, "Max should be qubits-1=2 (baseline)")

        circuit.remove(at: 0)
        #expect(circuit.highestQubitIndex == 2, "Max should remain qubits-1=2")
    }

    @Test("Remove highest qubit gate triggers recompute with empty qubits gate remaining")
    func removeHighestQubitRecomputeWithEmptyQubits() {
        let gates = [
            Gate(.hadamard, to: []),
            Gate(.pauliX, to: 2),
        ]
        var circuit = QuantumCircuit(qubits: 2, gates: gates)

        #expect(circuit.highestQubitIndex == 2, "Max should be 2 from X gate")

        circuit.remove(at: 1)
        #expect(circuit.highestQubitIndex == 1, "After removing qubit 2 gate, max should be qubits-1=1")
        #expect(circuit.count == 1, "Only empty-qubits gate should remain")
    }

    @Test("Recompute finds remaining gate at high qubit index")
    func recomputeFindsRemainingHighQubitGate() {
        let gates = [
            Gate(.hadamard, to: 5),
            Gate(.pauliX, to: 5),
        ]
        var circuit = QuantumCircuit(qubits: 2, gates: gates)

        #expect(circuit.highestQubitIndex == 5, "Max should be 5")

        circuit.remove(at: 0)
        #expect(circuit.highestQubitIndex == 5, "Max should still be 5 from remaining X gate")
        #expect(circuit.count == 1, "One gate should remain")
    }

    @Test("Description includes parameter count for parameterized circuits")
    func parameterizedCircuitDescription() {
        var circuit = QuantumCircuit(qubits: 2)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        circuit.append(.rotationY(.parameter(theta)), to: 0)
        circuit.append(.rotationZ(.parameter(phi)), to: 1)

        let description = circuit.description

        #expect(description.contains("2 params"), "Description should show parameter count")
        #expect(description.contains("2 qubits"), "Description should show qubit count")
        #expect(description.contains("2 gates"), "Description should show gate count")
    }

    @Test("Multi-controlled U with zero controls applies gate directly")
    func multiControlledUZeroControls() {
        var circuit = QuantumCircuit(qubits: 2)

        QuantumCircuit.appendMultiControlledU(to: &circuit, gate: .pauliX, controls: [], target: 0)
        QuantumCircuit.appendMultiControlledU(to: &circuit, gate: .pauliY, controls: [], target: 1)
        QuantumCircuit.appendMultiControlledU(to: &circuit, gate: .pauliZ, controls: [], target: 0)
        QuantumCircuit.appendMultiControlledU(to: &circuit, gate: .hadamard, controls: [], target: 1)
        QuantumCircuit.appendMultiControlledU(to: &circuit, gate: .tGate, controls: [], target: 0)
        QuantumCircuit.appendMultiControlledU(to: &circuit, gate: .sGate, controls: [], target: 1)

        #expect(circuit.count == 6, "Should have 6 gates appended directly")

        let state = circuit.execute()
        #expect(state.isNormalized(), "Result should be normalized")
    }

    @Test("Multi-controlled X with one control produces CNOT")
    func multiControlledXOneControlDirect() {
        var circuit = QuantumCircuit(qubits: 2)
        QuantumCircuit.appendMultiControlledU(to: &circuit, gate: .pauliX, controls: [0], target: 1)

        #expect(circuit.count == 1, "Should produce single CNOT gate")

        let state = QuantumState(qubits: 2, amplitudes: [.zero, .one, .zero, .zero])
        let result = circuit.execute(on: state)

        #expect(result.probability(of: 3) > 0.99, "CNOT on |01⟩ should give |11⟩")
    }

    @Test("Multi-controlled Y with one control produces CY")
    func multiControlledYOneControl() {
        var circuit = QuantumCircuit(qubits: 2)
        QuantumCircuit.appendMultiControlledU(to: &circuit, gate: .pauliY, controls: [0], target: 1)

        #expect(circuit.count == 1, "Should produce single CY gate")

        let state = QuantumState(qubits: 2, amplitudes: [.zero, .one, .zero, .zero])
        let result = circuit.execute(on: state)

        #expect(result.probability(of: 3) > 0.99, "CY on |01⟩ should give i|11⟩ with prob 1")
    }

    @Test("Multi-controlled Z with one control produces CZ")
    func multiControlledZOneControl() {
        var circuit = QuantumCircuit(qubits: 2)
        QuantumCircuit.appendMultiControlledU(to: &circuit, gate: .pauliZ, controls: [0], target: 1)

        #expect(circuit.count == 1, "Should produce single CZ gate")

        let plusPlus = QuantumState(qubits: 2, amplitudes: [0.5, 0.5, 0.5, 0.5])
        let result = circuit.execute(on: plusPlus)

        #expect(result.amplitude(of: 3).real < -0.4, "CZ flips phase of |11⟩ component")
    }

    @Test("Multi-controlled Z with multiple controls")
    func multiControlledZMultipleControls() {
        var circuit = QuantumCircuit(qubits: 3)
        QuantumCircuit.appendMultiControlledU(to: &circuit, gate: .pauliZ, controls: [0, 1], target: 2)

        #expect(circuit.count > 1, "Should decompose into multiple gates")

        let state = circuit.execute()
        #expect(state.isNormalized(), "Result should be normalized")
    }

    @Test("Ancilla expansion when circuit references higher qubit indices")
    func ancillaExpansionOnExecute() {
        let gates = [
            Gate(.hadamard, to: 0),
            Gate(.pauliX, to: 3),
        ]
        let circuit = QuantumCircuit(qubits: 2, gates: gates)

        #expect(circuit.qubits == 2, "Circuit qubits should be 2")
        #expect(circuit.highestQubitIndex == 3, "Highest qubit should be 3")

        let initialState = QuantumState(qubits: 2)
        let finalState = circuit.execute(on: initialState, upToIndex: circuit.count)

        #expect(finalState.qubits == 4, "State should expand to 4 qubits (indices 0-3)")
        #expect(finalState.isNormalized(), "Expanded state should be normalized")

        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(
            abs(finalState.amplitude(of: 0b1000).real - invSqrt2) < 1e-10,
            "X on qubit 3 flips |0000⟩ component to |1000⟩",
        )
        #expect(
            abs(finalState.amplitude(of: 0b1001).real - invSqrt2) < 1e-10,
            "X on qubit 3 flips |0001⟩ component to |1001⟩",
        )
    }
}

/// Test suite for parameter shift gradient computation.
/// Validates circuit shifting for parameter shift rule gradient estimation,
/// enabling VQE and QAOA gradient-based optimization.
@Suite("Shifted Circuits")
struct ShiftedCircuitsTests {
    @Test("Single parameter shifted circuits have correct values")
    func singleParameterShiftedCircuits() {
        var circuit = QuantumCircuit(qubits: 2)
        let theta = Parameter(name: "theta")
        circuit.append(.rotationY(.parameter(theta)), to: 0)
        circuit.append(.cnot, to: [0, 1])

        let baseValue = 0.5
        let base = ["theta": baseValue]
        let shift = Double.pi / 2

        let (plus, minus) = circuit.shiftedCircuits(for: "theta", base: base, shift: shift)

        let plusState = plus.execute()
        let minusState = minus.execute()

        #expect(plusState.isNormalized(), "Plus circuit should produce normalized state")
        #expect(minusState.isNormalized(), "Minus circuit should produce normalized state")

        let directPlus = circuit.binding(["theta": baseValue + shift]).execute()
        let directMinus = circuit.binding(["theta": baseValue - shift]).execute()

        #expect(plusState == directPlus, "Plus circuit should match direct binding")
        #expect(minusState == directMinus, "Minus circuit should match direct binding")
    }

    @Test("Multi-parameter circuit shifts only target parameter")
    func multiParameterShiftOnlyTarget() {
        var circuit = QuantumCircuit(qubits: 2)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        circuit.append(.rotationY(.parameter(theta)), to: 0)
        circuit.append(.rotationZ(.parameter(phi)), to: 1)

        let base = ["theta": 0.3, "phi": 0.7]
        let shift = Double.pi / 2

        let (plus, minus) = circuit.shiftedCircuits(for: "theta", base: base, shift: shift)

        let directPlus = circuit.binding(["theta": 0.3 + shift, "phi": 0.7]).execute()
        let directMinus = circuit.binding(["theta": 0.3 - shift, "phi": 0.7]).execute()

        #expect(plus.execute() == directPlus, "Only theta should be shifted in plus circuit")
        #expect(minus.execute() == directMinus, "Only theta should be shifted in minus circuit")
    }

    @Test("Custom shift value works correctly")
    func customShiftValue() {
        var circuit = QuantumCircuit(qubits: 1)
        let theta = Parameter(name: "theta")
        circuit.append(.rotationY(.parameter(theta)), to: 0)

        let base = ["theta": 1.0]
        let customShift = 0.1

        let (plus, minus) = circuit.shiftedCircuits(for: "theta", base: base, shift: customShift)

        let directPlus = circuit.binding(["theta": 1.1]).execute()
        let directMinus = circuit.binding(["theta": 0.9]).execute()

        #expect(plus.execute() == directPlus, "Custom shift should be applied to plus circuit")
        #expect(minus.execute() == directMinus, "Custom shift should be applied to minus circuit")
    }

    @Test("Default shift is pi/2")
    func defaultShiftIsPiOver2() {
        var circuit = QuantumCircuit(qubits: 1)
        let theta = Parameter(name: "theta")
        circuit.append(.rotationY(.parameter(theta)), to: 0)

        let base = ["theta": 0.0]

        let (plus, minus) = circuit.shiftedCircuits(for: "theta", base: base)

        let directPlus = circuit.binding(["theta": Double.pi / 2]).execute()
        let directMinus = circuit.binding(["theta": -Double.pi / 2]).execute()

        #expect(plus.execute() == directPlus, "Default shift should be π/2")
        #expect(minus.execute() == directMinus, "Default shift should be -π/2")
    }

    @Test("Vector-based shifted circuits basic functionality")
    func vectorBasedShiftedCircuits() {
        var circuit = QuantumCircuit(qubits: 2)
        let theta0 = Parameter(name: "theta_0")
        let theta1 = Parameter(name: "theta_1")
        circuit.append(.rotationY(.parameter(theta0)), to: 0)
        circuit.append(.rotationY(.parameter(theta1)), to: 1)

        let baseVector = [0.5, 1.0]
        let shift = Double.pi / 2

        let (plus, minus) = circuit.shiftedCircuits(at: 0, baseVector: baseVector, shift: shift)

        let directPlus = circuit.bound(with: [0.5 + shift, 1.0]).execute()
        let directMinus = circuit.bound(with: [0.5 - shift, 1.0]).execute()

        #expect(plus.execute() == directPlus, "Vector interface plus should match direct binding")
        #expect(minus.execute() == directMinus, "Vector interface minus should match direct binding")
    }

    @Test("Vector-based shifted circuits at different indices")
    func vectorBasedShiftAtDifferentIndices() {
        var circuit = QuantumCircuit(qubits: 3)
        for i in 0 ..< 3 {
            let param = Parameter(name: "theta_\(i)")
            circuit.append(.rotationY(.parameter(param)), to: i)
        }

        let baseVector = [0.1, 0.2, 0.3]
        let shift = Double.pi / 4

        for index in 0 ..< 3 {
            let (plus, minus) = circuit.shiftedCircuits(at: index, baseVector: baseVector, shift: shift)

            var expectedPlus = baseVector
            var expectedMinus = baseVector
            expectedPlus[index] += shift
            expectedMinus[index] -= shift

            let directPlus = circuit.bound(with: expectedPlus).execute()
            let directMinus = circuit.bound(with: expectedMinus).execute()

            #expect(plus.execute() == directPlus, "Shift at index \(index) plus should match")
            #expect(minus.execute() == directMinus, "Shift at index \(index) minus should match")
        }
    }

    @Test("Gradient computation via parameter shift rule")
    func gradientComputationViaParameterShift() {
        var circuit = QuantumCircuit(qubits: 1)
        let theta = Parameter(name: "theta")
        circuit.append(.rotationY(.parameter(theta)), to: 0)

        let observable = Observable.pauliZ(qubit: 0)

        let baseValue = 0.5
        let base = ["theta": baseValue]

        let (plus, minus) = circuit.shiftedCircuits(for: "theta", base: base)

        let plusEnergy = observable.expectationValue(of: plus.execute())
        let minusEnergy = observable.expectationValue(of: minus.execute())
        let gradient = (plusEnergy - minusEnergy) / 2.0

        let analyticalGradient = -sin(baseValue)

        #expect(
            abs(gradient - analyticalGradient) < 1e-10,
            "Parameter shift gradient should match analytical: got \(gradient), expected \(analyticalGradient)",
        )
    }
}

/// Validates Cartesian product generation for parameter
/// grid search optimization, enabling systematic
/// hyperparameter exploration in variational algorithms.
@Suite("Grid Search Vectors")
struct GridSearchVectorsTests {
    @Test("Single parameter single value")
    func singleParameterSingleValue() {
        var circuit = QuantumCircuit(qubits: 1)
        let theta = Parameter(name: "theta")
        circuit.append(.rotationY(.parameter(theta)), to: 0)

        let ranges = [[0.5]]
        let vectors = circuit.gridSearchVectors(ranges: ranges)

        #expect(vectors.count == 1, "Should produce 1 vector")
        #expect(vectors[0] == [0.5], "Vector should be [0.5]")
    }

    @Test("Single parameter multiple values")
    func singleParameterMultipleValues() {
        var circuit = QuantumCircuit(qubits: 1)
        let theta = Parameter(name: "theta")
        circuit.append(.rotationY(.parameter(theta)), to: 0)

        let ranges: [[Double]] = [[0.0, 0.5, 1.0]]
        let vectors = circuit.gridSearchVectors(ranges: ranges)

        #expect(vectors.count == 3, "Should produce 3 vectors")
        #expect(vectors[0] == [0.0], "First vector")
        #expect(vectors[1] == [0.5], "Second vector")
        #expect(vectors[2] == [1.0], "Third vector")
    }

    @Test("Two parameters Cartesian product")
    func twoParametersCartesianProduct() {
        var circuit = QuantumCircuit(qubits: 2)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        circuit.append(.rotationY(.parameter(theta)), to: 0)
        circuit.append(.rotationZ(.parameter(phi)), to: 1)

        let ranges: [[Double]] = [[0.0, 1.0], [2.0, 3.0]]
        let vectors = circuit.gridSearchVectors(ranges: ranges)

        #expect(vectors.count == 4, "Should produce 2*2=4 vectors")
        #expect(vectors.contains([0.0, 2.0]), "Should contain [0.0, 2.0]")
        #expect(vectors.contains([0.0, 3.0]), "Should contain [0.0, 3.0]")
        #expect(vectors.contains([1.0, 2.0]), "Should contain [1.0, 2.0]")
        #expect(vectors.contains([1.0, 3.0]), "Should contain [1.0, 3.0]")
    }

    @Test("Three parameters asymmetric ranges")
    func threeParametersAsymmetricRanges() {
        var circuit = QuantumCircuit(qubits: 3)
        for i in 0 ..< 3 {
            let param = Parameter(name: "p\(i)")
            circuit.append(.rotationY(.parameter(param)), to: i)
        }

        let ranges: [[Double]] = [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]]
        let vectors = circuit.gridSearchVectors(ranges: ranges)

        #expect(vectors.count == 6, "Should produce 2*1*3=6 vectors")
        #expect(vectors[0][1] == 3.0, "Middle parameter always 3.0")
        #expect(vectors.contains([1.0, 3.0, 4.0]), "Should contain first combination")
        #expect(vectors.contains([2.0, 3.0, 6.0]), "Should contain last combination")
    }

    @Test("Grid search vectors produce valid circuits")
    func gridSearchVectorsProduceValidCircuits() {
        var circuit = QuantumCircuit(qubits: 2)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        circuit.append(.rotationY(.parameter(theta)), to: 0)
        circuit.append(.rotationY(.parameter(phi)), to: 1)

        let ranges: [[Double]] = [[0.0, Double.pi / 2], [0.0, Double.pi]]
        let vectors = circuit.gridSearchVectors(ranges: ranges)

        for vector in vectors {
            let bound = circuit.bound(with: vector)
            let state = bound.execute()
            #expect(state.isNormalized(), "Bound circuit should produce normalized state")
        }
    }
}

/// Validates batch parameter binding for efficient
/// parallel circuit generation, enabling vectorized
/// VQE optimization and gradient computation.
@Suite("Batch Parameter Binding")
struct BatchParameterBindingTests {
    @Test("Batch binding produces correct number of circuits")
    func batchBindingProducesCorrectCount() {
        var circuit = QuantumCircuit(qubits: 1)
        let theta = Parameter(name: "theta")
        circuit.append(.rotationY(.parameter(theta)), to: 0)

        let vectors: [[Double]] = [[0.0], [0.5], [1.0], [1.5]]
        let circuits = circuit.binding(batch: vectors)

        #expect(circuits.count == 4, "Should produce 4 circuits")
    }

    @Test("Batch binding matches individual binding")
    func batchBindingMatchesIndividualBinding() {
        var circuit = QuantumCircuit(qubits: 2)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        circuit.append(.rotationY(.parameter(theta)), to: 0)
        circuit.append(.rotationZ(.parameter(phi)), to: 1)

        let vectors: [[Double]] = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        let batchCircuits = circuit.binding(batch: vectors)

        for (index, vector) in vectors.enumerated() {
            let individualCircuit = circuit.bound(with: vector)
            let batchState = batchCircuits[index].execute()
            let individualState = individualCircuit.execute()

            #expect(batchState == individualState, "Batch circuit \(index) should match individual binding")
        }
    }

    @Test("Batch binding with single vector")
    func batchBindingWithSingleVector() {
        var circuit = QuantumCircuit(qubits: 1)
        let theta = Parameter(name: "theta")
        circuit.append(.rotationY(.parameter(theta)), to: 0)

        let vectors: [[Double]] = [[Double.pi / 4]]
        let circuits = circuit.binding(batch: vectors)

        #expect(circuits.count == 1, "Should produce 1 circuit")

        let state = circuits[0].execute()
        let expectedProb1 = pow(sin(Double.pi / 8), 2)
        #expect(abs(state.probability(of: 1) - expectedProb1) < 1e-10, "State should match Ry(π/4)")
    }

    @Test("Batch binding with many parameters")
    func batchBindingWithManyParameters() {
        var circuit = QuantumCircuit(qubits: 4)
        for i in 0 ..< 4 {
            let param = Parameter(name: "theta_\(i)")
            circuit.append(.rotationY(.parameter(param)), to: i)
        }

        let vectors: [[Double]] = [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ]
        let circuits = circuit.binding(batch: vectors)

        #expect(circuits.count == 2, "Should produce 2 circuits")

        for bound in circuits {
            #expect(bound.parameterCount == 0, "Bound circuits should have no free parameters")
            let state = bound.execute()
            #expect(state.isNormalized(), "Each circuit should produce normalized state")
        }
    }

    @Test("Batch binding integrates with grid search")
    func batchBindingIntegratesWithGridSearch() {
        var circuit = QuantumCircuit(qubits: 2)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        circuit.append(.rotationY(.parameter(theta)), to: 0)
        circuit.append(.rotationY(.parameter(phi)), to: 1)

        let ranges: [[Double]] = [[0.0, Double.pi / 2], [0.0, Double.pi / 2]]
        let vectors = circuit.gridSearchVectors(ranges: ranges)
        let circuits = circuit.binding(batch: vectors)

        #expect(circuits.count == 4, "Grid search should produce 4 vectors, batch should produce 4 circuits")

        for bound in circuits {
            let state = bound.execute()
            #expect(state.isNormalized(), "All grid search circuits should produce valid states")
        }
    }
}
