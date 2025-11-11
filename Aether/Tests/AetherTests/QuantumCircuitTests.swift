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
        let circuit = QuantumCircuit(numQubits: 2)
        #expect(circuit.numQubits == 2)
        #expect(circuit.gateCount == 0)
        #expect(circuit.isEmpty)
    }

    @Test("Append single gate")
    func appendSingleGate() {
        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .hadamard, toQubit: 0)

        #expect(circuit.gateCount == 1)
        #expect(!circuit.isEmpty)
    }

    @Test("Append multiple gates")
    func appendMultipleGates() {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])

        #expect(circuit.gateCount == 2)
    }

    @Test("Insert gate at position")
    func insertGateAtPosition() {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .pauliX, toQubit: 1)
        circuit.insert(gate: .pauliZ, qubits: [0], at: 1)

        #expect(circuit.gateCount == 3)
        #expect(circuit.operation(at: 1).gate == .pauliZ)
    }

    @Test("Remove gate")
    func removeGate() {
        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .pauliX, toQubit: 0)

        circuit.remove(at: 0)

        #expect(circuit.gateCount == 1)
        #expect(circuit.operation(at: 0).gate == .pauliX)
    }

    @Test("Clear circuit")
    func clearCircuit() {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .hadamard, toQubit: 1)

        circuit.clear()

        #expect(circuit.isEmpty)
        #expect(circuit.gateCount == 0)
    }
}

/// Test suite for quantum circuit execution.
/// Validates sequential gate application, state transformation correctness,
/// and normalization preservation through complex circuit sequences.
@Suite("Circuit Execution")
struct CircuitExecutionTests {
    @Test("Empty circuit returns unchanged state")
    func emptyCircuitUnchanged() {
        let circuit = QuantumCircuit(numQubits: 1)
        let initialState = QuantumState(singleQubit: 0)
        let finalState = circuit.execute(on: initialState)

        #expect(initialState == finalState)
    }

    @Test("Single-gate circuit")
    func singleGateCircuit() {
        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .pauliX, toQubit: 0)

        let initialState = QuantumState(singleQubit: 0)
        let finalState = circuit.execute(on: initialState)

        #expect(abs(finalState.getAmplitude(ofState: 1).real - 1.0) < 1e-10)
    }

    @Test("Multi-gate circuit")
    func multiGateCircuit() {
        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .pauliZ, toQubit: 0)
        circuit.append(gate: .hadamard, toQubit: 0)

        let initialState = QuantumState(singleQubit: 0)
        let finalState = circuit.execute(on: initialState)

        #expect(abs(finalState.getAmplitude(ofState: 1).real - 1.0) < 1e-10)
    }

    @Test("Execute from default |0⟩ state")
    func executeFromDefaultState() {
        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .hadamard, toQubit: 0)

        let finalState = circuit.execute()

        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(finalState.getAmplitude(ofState: 0).real - invSqrt2) < 1e-10)
        #expect(abs(finalState.getAmplitude(ofState: 1).real - invSqrt2) < 1e-10)
    }
}

/// Test suite for step-through circuit execution.
/// Validates partial execution for animation, debugging, and state inspection,
/// enabling quantum algorithm visualization and educational interfaces.
@Suite("Step-Through Execution")
struct StepThroughTests {
    @Test("Execute up to index 0 (no gates)")
    func executeUpToZero() {
        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .hadamard, toQubit: 0)

        let initialState = QuantumState(singleQubit: 0)
        let finalState = circuit.execute(on: initialState, upToIndex: 0)

        #expect(initialState == finalState)
    }

    @Test("Execute up to middle of circuit")
    func executeUpToMiddle() {
        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .pauliZ, toQubit: 0)
        circuit.append(gate: .hadamard, toQubit: 0)

        let initialState = QuantumState(singleQubit: 0)
        let state1 = circuit.execute(on: initialState, upToIndex: 1)
        let invSqrt2 = 1.0 / sqrt(2.0)

        #expect(abs(state1.getAmplitude(ofState: 0).real - invSqrt2) < 1e-10)
    }

    @Test("Step through entire circuit")
    func stepThroughEntireCircuit() {
        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .pauliX, toQubit: 0)
        circuit.append(gate: .pauliX, toQubit: 0)

        let initialState = QuantumState(singleQubit: 0)

        let state0 = circuit.execute(on: initialState, upToIndex: 0)
        let state1 = circuit.execute(on: initialState, upToIndex: 1)
        let state2 = circuit.execute(on: initialState, upToIndex: 2)

        #expect(state0 == initialState)
        #expect(abs(state1.getAmplitude(ofState: 1).real - 1.0) < 1e-10)
        #expect(abs(state2.getAmplitude(ofState: 0).real - 1.0) < 1e-10)
    }
}

/// Test suite for Bell state circuit factory.
/// Validates maximally entangled two-qubit state creation,
/// fundamental for quantum teleportation and superdense coding.
@Suite("Bell State Circuit")
struct BellStateCircuitTests {
    @Test("Bell state circuit creates entanglement")
    func bellStateCreatesEntanglement() {
        let circuit = QuantumCircuit.bellState()
        let finalState = circuit.execute()
        let invSqrt2 = 1.0 / sqrt(2.0)

        #expect(abs(finalState.getAmplitude(ofState: 0).real - invSqrt2) < 1e-10)
        #expect(abs(finalState.getAmplitude(ofState: 3).real - invSqrt2) < 1e-10)
        #expect(abs(finalState.getAmplitude(ofState: 1).magnitude) < 1e-10)
        #expect(abs(finalState.getAmplitude(ofState: 2).magnitude) < 1e-10)
    }

    @Test("Bell state is normalized")
    func bellStateNormalized() {
        let circuit = QuantumCircuit.bellState()
        let finalState = circuit.execute()

        #expect(finalState.isNormalized())
    }

    @Test("Bell state has correct probabilities")
    func bellStateProbabilities() {
        let circuit = QuantumCircuit.bellState()
        let finalState = circuit.execute()

        #expect(abs(finalState.probability(ofState: 0) - 0.5) < 1e-10)
        #expect(abs(finalState.probability(ofState: 3) - 0.5) < 1e-10)
    }
}

/// Test suite for GHZ state circuit factory.
/// Validates multi-qubit Greenberger-Horne-Zeilinger state creation,
/// demonstrating genuine multipartite entanglement beyond Bell states.
@Suite("GHZ State Circuit")
struct GHZStateCircuitTests {
    @Test("3-qubit GHZ state")
    func threeQubitGHZ() {
        let circuit = QuantumCircuit.ghzState(numQubits: 3)
        let finalState = circuit.execute()

        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(finalState.getAmplitude(ofState: 0).real - invSqrt2) < 1e-10)
        #expect(abs(finalState.getAmplitude(ofState: 7).real - invSqrt2) < 1e-10)

        for i in 1 ..< 7 {
            #expect(abs(finalState.getAmplitude(ofState: i).magnitude) < 1e-10)
        }
    }

    @Test("4-qubit GHZ state")
    func fourQubitGHZ() {
        let circuit = QuantumCircuit.ghzState(numQubits: 4)
        let finalState = circuit.execute()

        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(finalState.getAmplitude(ofState: 0).real - invSqrt2) < 1e-10)
        #expect(abs(finalState.getAmplitude(ofState: 15).real - invSqrt2) < 1e-10)
    }
}

/// Test suite for superposition circuit factory.
/// Validates equal superposition state creation across all computational basis states,
/// essential for quantum search and amplitude estimation algorithms.
@Suite("Superposition Circuit")
struct SuperpositionCircuitTests {
    @Test("Single qubit superposition")
    func singleQubitSuperposition() {
        let circuit = QuantumCircuit.superposition(numQubits: 1)
        let finalState = circuit.execute()

        let invSqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(finalState.getAmplitude(ofState: 0).real - invSqrt2) < 1e-10)
        #expect(abs(finalState.getAmplitude(ofState: 1).real - invSqrt2) < 1e-10)
    }

    @Test("2-qubit equal superposition")
    func twoQubitSuperposition() {
        let circuit = QuantumCircuit.superposition(numQubits: 2)
        let finalState = circuit.execute()

        let expectedProb = 0.25
        for i in 0 ..< 4 {
            #expect(abs(finalState.probability(ofState: i) - expectedProb) < 1e-10)
        }
    }

    @Test("3-qubit equal superposition")
    func threeQubitSuperposition() {
        let circuit = QuantumCircuit.superposition(numQubits: 3)
        let finalState = circuit.execute()

        let expectedProb = 1.0 / 8.0
        for i in 0 ..< 8 {
            #expect(abs(finalState.probability(ofState: i) - expectedProb) < 1e-10)
        }
    }
}

/// Test suite for quantum circuit validation.
/// Validates circuit structure integrity, gate-qubit compatibility,
/// and error detection for invalid quantum circuit configurations.
@Suite("Circuit Validation")
struct CircuitValidationTests {
    @Test("Valid circuit passes validation")
    func validCircuit() {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])

        #expect(circuit.validate())
    }

    @Test("Empty circuit is valid")
    func emptyCircuitValid() {
        let circuit = QuantumCircuit(numQubits: 2)
        #expect(circuit.validate())
    }
}

/// Test suite for quantum circuit properties.
/// Validates circuit metadata consistency including qubit count maintenance,
/// gate count tracking, and structural invariants during circuit modification.
@Suite("Circuit Properties")
struct CircuitPropertiesTests {
    @Test("Circuit maintains qubit count")
    func maintainsQubitCount() {
        let circuit = QuantumCircuit(numQubits: 5)
        #expect(circuit.numQubits == 5)
    }

    @Test("Gate count updates correctly")
    func gateCountUpdates() {
        var circuit = QuantumCircuit(numQubits: 2)
        #expect(circuit.gateCount == 0)

        circuit.append(gate: .hadamard, toQubit: 0)
        #expect(circuit.gateCount == 1)

        circuit.append(gate: .hadamard, toQubit: 1)
        #expect(circuit.gateCount == 2)

        circuit.remove(at: 0)
        #expect(circuit.gateCount == 1)
    }
}

/// Test suite for quantum circuit equality comparison.
/// Validates Equatable conformance for circuit comparison operations,
/// ensuring identical gate sequences are recognized as equal while different configurations are distinguished.
@Suite("Circuit Equality")
struct CircuitEqualityTests {
    @Test("Identical circuits are equal")
    func identicalCircuitsEqual() {
        var circuit1 = QuantumCircuit(numQubits: 2)
        circuit1.append(gate: .hadamard, toQubit: 0)

        var circuit2 = QuantumCircuit(numQubits: 2)
        circuit2.append(gate: .hadamard, toQubit: 0)

        #expect(circuit1 == circuit2)
    }

    @Test("Different gate sequences are not equal")
    func differentSequencesNotEqual() {
        var circuit1 = QuantumCircuit(numQubits: 2)
        circuit1.append(gate: .hadamard, toQubit: 0)

        var circuit2 = QuantumCircuit(numQubits: 2)
        circuit2.append(gate: .pauliX, toQubit: 0)

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
        let circuit = QuantumCircuit(numQubits: 2)
        let desc = circuit.description

        #expect(desc.contains("2 qubits"))
        #expect(desc.contains("empty"))
    }

    @Test("Non-empty circuit description")
    func nonEmptyCircuitDescription() {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)

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
        let circuit = QuantumCircuit.superposition(numQubits: 8)
        let finalState = circuit.execute()

        #expect(finalState.numQubits == 8)
        #expect(finalState.isNormalized())
    }

    @Test("Circuit works with 12 qubits")
    func twelveQubitCircuit() {
        var circuit = QuantumCircuit(numQubits: 12)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .cnot(control: 0, target: 5), qubits: [])

        let finalState = circuit.execute()
        #expect(finalState.numQubits == 12)
        #expect(finalState.isNormalized())
    }

    @Test("Large circuit with many gates")
    func largeCircuit() {
        var circuit = QuantumCircuit(numQubits: 4)

        for _ in 0 ..< 5 {
            for q in 0 ..< 4 {
                circuit.append(gate: .hadamard, toQubit: q)
            }
        }

        #expect(circuit.gateCount == 20)

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
        let circuit = QuantumCircuit.qft(numQubits: 3)

        #expect(circuit.numQubits == 3)
        #expect(circuit.gateCount > 0)
        #expect(circuit.validate())
    }

    @Test("QFT preserves normalization")
    func qftPreservesNormalization() {
        let circuit = QuantumCircuit.qft(numQubits: 3)
        let finalState = circuit.execute()

        #expect(finalState.isNormalized())
    }

    @Test("QFT on |0⟩ state")
    func qftOnZeroState() {
        let circuit = QuantumCircuit.qft(numQubits: 3)
        let finalState = circuit.execute()

        let expectedAmp = 1.0 / sqrt(8.0)

        for i in 0 ..< 8 {
            let amp = finalState.getAmplitude(ofState: i)
            #expect(abs(amp.magnitude - expectedAmp) < 1e-10)
        }
    }

    @Test("Inverse QFT reverses QFT")
    func inverseQFTReversesQFT() {
        var amplitudes = AmplitudeVector(repeating: .zero, count: 8)
        amplitudes[5] = .one
        let initialState = QuantumState(numQubits: 3, amplitudes: amplitudes)

        let qft = QuantumCircuit.qft(numQubits: 3)
        let afterQFT = qft.execute(on: initialState)

        let inverseQFT = QuantumCircuit.inverseQFT(numQubits: 3)
        let finalState = inverseQFT.execute(on: afterQFT)

        #expect(abs(finalState.getAmplitude(ofState: 5).real - 1.0) < 1e-8)

        for i in 0 ..< 8 {
            if i != 5 {
                #expect(abs(finalState.getAmplitude(ofState: i).magnitude) < 1e-8)
            }
        }
    }

    @Test("QFT gate count")
    func qftGateCount() {
        let n = 4
        let circuit = QuantumCircuit.qft(numQubits: n)

        let expectedCount = n + (n * (n - 1)) / 2 + (n / 2)
        #expect(circuit.gateCount == expectedCount)
    }

    @Test("QFT works with different qubit counts")
    func qftDifferentSizes() {
        for numQubits in 2 ... 6 {
            let circuit = QuantumCircuit.qft(numQubits: numQubits)
            let finalState = circuit.execute()

            #expect(finalState.numQubits == numQubits)
            #expect(finalState.isNormalized())
        }
    }

    @Test("QFT transforms basis states")
    func qftTransformsBasisStates() {
        let n = 3

        for k in 0 ..< (1 << n) {
            var amplitudes = AmplitudeVector(repeating: .zero, count: 1 << n)
            amplitudes[k] = .one
            let initialState = QuantumState(numQubits: n, amplitudes: amplitudes)

            let qft = QuantumCircuit.qft(numQubits: n)
            let finalState = qft.execute(on: initialState)

            #expect(finalState.isNormalized())

            let expectedMag = 1.0 / sqrt(Double(1 << n))
            for i in 0 ..< (1 << n) {
                let amp = finalState.getAmplitude(ofState: i)
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
        var circuit = QuantumCircuit(numQubits: 3)
        for i in 0 ..< 12 {
            let qubit = i % 3
            circuit.append(gate: .hadamard, toQubit: qubit)
        }

        let initialState = QuantumState(numQubits: 3)

        let fullState = circuit.execute(on: initialState)
        let stepState = circuit.execute(on: initialState, upToIndex: circuit.gateCount)

        #expect(fullState == stepState)
    }

    @Test("Multiple step executions benefit from cache")
    func multipleStepExecutions() {
        var circuit = QuantumCircuit(numQubits: 2)

        for _ in 0 ..< 5 {
            circuit.append(gate: .hadamard, toQubit: 0)
            circuit.append(gate: .hadamard, toQubit: 1)
            circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
            circuit.append(gate: .hadamard, toQubit: 0)
        }

        let initialState = QuantumState(numQubits: 2)

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
        var circuit = QuantumCircuit(numQubits: 2)

        for _ in 0 ..< 10 {
            circuit.append(gate: .hadamard, toQubit: 0)
        }

        let initialState = QuantumState(numQubits: 2)

        _ = circuit.execute(on: initialState, upToIndex: 10)
        circuit.append(gate: .pauliX, toQubit: 1)

        let finalState = circuit.execute(on: initialState, upToIndex: 11)
        #expect(finalState.isNormalized())
    }

    @Test("Backward scrubbing works correctly")
    func backwardScrubbingWorks() {
        var circuit = QuantumCircuit(numQubits: 2)

        for _ in 0 ..< 15 {
            circuit.append(gate: .hadamard, toQubit: 0)
        }

        let initialState = QuantumState(numQubits: 2)

        _ = circuit.execute(on: initialState, upToIndex: 15)
        let state10 = circuit.execute(on: initialState, upToIndex: 10)
        let state5 = circuit.execute(on: initialState, upToIndex: 5)

        #expect(state10.isNormalized())
        #expect(state5.isNormalized())
    }

    @Test("Cache works with empty upToIndex")
    func cacheWorksWithZeroIndex() {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        let initialState = QuantumState(numQubits: 2)
        let state = circuit.execute(on: initialState, upToIndex: 0)

        #expect(state == initialState)
    }

    @Test("Cache with QFT circuit")
    func cacheWithQFTCircuit() {
        let circuit = QuantumCircuit.qft(numQubits: 4)
        let initialState = QuantumState(numQubits: 4)

        let midState = circuit.execute(on: initialState, upToIndex: circuit.gateCount / 2)
        let finalState = circuit.execute(on: initialState, upToIndex: circuit.gateCount)

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
        let circuit = QuantumCircuit.grover(numQubits: 3, target: 5)

        #expect(circuit.numQubits == 3)
        #expect(circuit.gateCount > 0)

        let firstOps = circuit.operations.prefix(3)
        #expect(firstOps.allSatisfy { op in
            if case .hadamard = op.gate { return true }
            return false
        })
    }

    @Test("Grover preserves normalization")
    func groverPreservesNormalization() {
        let circuit = QuantumCircuit.grover(numQubits: 2, target: 3)
        let initialState = QuantumState(numQubits: circuit.numQubits)
        let finalState = circuit.execute(on: initialState)

        #expect(finalState.isNormalized())
    }

    @Test("Grover finds target with high probability (2-qubit)")
    func groverFindsTarget2Qubit() {
        let target = 2
        let circuit = QuantumCircuit.grover(numQubits: 2, target: target)

        let initialState = QuantumState(numQubits: circuit.numQubits)
        let finalState = circuit.execute(on: initialState)

        let targetProb = finalState.probability(ofState: target)

        #expect(targetProb > 0.8, "Target probability: \(targetProb)")
    }

    @Test("Grover finds target with high probability (3-qubit)")
    func groverFindsTarget3Qubit() {
        let target = 5
        let circuit = QuantumCircuit.grover(numQubits: 3, target: target)

        let initialState = QuantumState(numQubits: circuit.numQubits)
        let finalState = circuit.execute(on: initialState)

        let targetProb = finalState.probability(ofState: target)

        #expect(targetProb > 0.8, "Target probability: \(targetProb)")
    }

    @Test("Grover optimal iteration count")
    func groverOptimalIterations() {
        let circuit2 = QuantumCircuit.grover(numQubits: 2, target: 0)
        let circuit3 = QuantumCircuit.grover(numQubits: 3, target: 0)

        #expect(circuit2.gateCount > 0)
        #expect(circuit3.gateCount > circuit2.gateCount)
    }

    @Test("Grover with custom iterations")
    func groverCustomIterations() {
        let circuit = QuantumCircuit.grover(numQubits: 2, target: 1, iterations: 2)
        let initialState = QuantumState(numQubits: circuit.numQubits)
        let finalState = circuit.execute(on: initialState)

        #expect(finalState.isNormalized())
    }

    @Test("Grover searches different targets correctly")
    func groverDifferentTargets() {
        let numQubits = 3

        for target in 0 ..< (1 << numQubits) {
            let circuit = QuantumCircuit.grover(numQubits: numQubits, target: target)
            let initialState = QuantumState(numQubits: circuit.numQubits)
            let finalState = circuit.execute(on: initialState)

            let targetProb = finalState.probability(ofState: target)

            #expect(
                targetProb > 0.7,
                "Target \(target) probability: \(targetProb)"
            )
        }
    }

    @Test("Grover with 1 qubit")
    func grover1Qubit() {
        let circuit = QuantumCircuit.grover(numQubits: 1, target: 1)
        let initialState = QuantumState(numQubits: circuit.numQubits)
        let finalState = circuit.execute(on: initialState)

        #expect(finalState.isNormalized())

        let prob = finalState.probability(ofState: 1)
        #expect(prob > 0.49)
    }

    @Test("Grover statistical validation")
    func groverStatisticalValidation() {
        let target = 3
        let circuit = QuantumCircuit.grover(numQubits: 2, target: target)

        let results = Measurement.runMultiple(circuit: circuit, numRuns: 1000)

        let targetCount = results.filter { $0 == target }.count
        let targetFrequency = Double(targetCount) / 1000.0

        #expect(targetFrequency > 0.7, "Target frequency: \(targetFrequency)")
    }

    @Test("Quantum annealing circuit creation")
    func createAnnealingCircuit() {
        let problem = QuantumCircuit.IsingProblem.quadraticMinimum(numQubits: 3)
        let circuit = QuantumCircuit.annealing(numQubits: 3, problem: problem)

        #expect(circuit.numQubits == 3)
        #expect(circuit.gateCount > 0, "Annealing circuit should have gates")
        #expect(circuit.validate(), "Circuit should be valid")
    }

    @Test("Annealing circuit execution")
    func executeAnnealingCircuit() {
        let problem = QuantumCircuit.IsingProblem.quadraticMinimum(numQubits: 2)
        let circuit = QuantumCircuit.annealing(numQubits: 2, problem: problem, annealingSteps: 5)

        let finalState = circuit.execute()
        #expect(finalState.isNormalized(), "Final state should be normalized")

        var outcomes: [Int] = []
        for _ in 0 ..< 100 {
            let result = Measurement.measureOnce(state: finalState)
            outcomes.append(result.outcome)
        }

        let uniqueOutcomes = Set(outcomes)
        #expect(uniqueOutcomes.count > 1, "Should explore multiple states with partial annealing")
    }

    @Test("Annealing finds optimal solution")
    func annealingFindsOptimalSolution() {
        let problem = QuantumCircuit.IsingProblem.quadraticMinimum(numQubits: 2)
        let circuit = QuantumCircuit.annealing(numQubits: 2, problem: problem, annealingSteps: 20)

        var outcomes: [Int] = []
        for _ in 0 ..< 1000 {
            let finalState = circuit.execute()
            let result = Measurement.measureOnce(state: finalState)
            outcomes.append(result.outcome)
        }

        var counts = ShotAllocation()
        for outcome in outcomes {
            counts[outcome, default: 0] += 1
        }

        let mostFrequent = counts.max(by: { $0.value < $1.value })?.key ?? 0
        let frequency = Double(counts[mostFrequent] ?? 0) / Double(outcomes.count)

        #expect(mostFrequent == 3, "Optimal solution should be |11⟩ (state 3), got \(mostFrequent)")
        #expect(frequency > 0.5, "Optimal solution frequency: \(frequency) (should be >0.5)")
    }

    @Test("Annealing convenience method with couplings parameter")
    func annealingConvenienceMethod() {
        let couplings = ["0-1": 0.5, "1-2": -0.3]
        let circuit = QuantumCircuit.annealing(numQubits: 3, couplings: couplings, annealingSteps: 10)

        #expect(circuit.numQubits == 3)
        #expect(circuit.gateCount > 0)
        #expect(circuit.validate())

        let finalState = circuit.execute()
        #expect(finalState.isNormalized())
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

    @Test("Random Ising problem")
    func createRandomIsingProblem() {
        let problem = QuantumCircuit.IsingProblem.random(numQubits: 3, maxCoupling: 2.0)

        #expect(problem.localFields.count == 3)
        #expect(problem.couplings.count == 3)
        #expect(problem.couplings.allSatisfy { $0.count == 3 })

        for i in 0 ..< 3 {
            for j in 0 ..< 3 {
                #expect(problem.couplings[i][j] == problem.couplings[j][i])
            }
        }
    }

    @Test("Quadratic minimum problem")
    func createQuadraticProblem() {
        let problem = QuantumCircuit.IsingProblem.quadraticMinimum(numQubits: 3)

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
        let ops = [
            GateOperation(gate: .hadamard, qubits: [0]),
            GateOperation(gate: .pauliX, qubits: [1]),
            GateOperation(gate: .cnot(control: 0, target: 1), qubits: []),
        ]

        let circuit = QuantumCircuit(numQubits: 2, operations: ops)

        #expect(circuit.numQubits == 2)
        #expect(circuit.gateCount == 3)
        #expect(circuit.operations[0].gate == .hadamard)
    }

    @Test("Append gate with timestamp")
    func appendGateWithTimestamp() {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, qubits: [0], timestamp: 1.5)
        circuit.append(gate: .pauliX, qubits: [1], timestamp: 3.0)

        #expect(circuit.gateCount == 2)
        #expect(circuit.operations[0].timestamp == 1.5)
        #expect(circuit.operations[1].timestamp == 3.0)
    }

    @Test("GateOperation description with timestamp")
    func gateOperationDescriptionWithTimestamp() {
        let op1 = GateOperation(gate: .hadamard, qubits: [0], timestamp: 2.5)
        let op2 = GateOperation(gate: .cnot(control: 0, target: 1), qubits: [], timestamp: 1.25)

        #expect(op1.description.contains("2.50"))
        #expect(op1.description.contains("s"))
        #expect(op2.description.contains("1.25"))
    }

    @Test("GateOperation description without timestamp")
    func gateOperationDescriptionNoTimestamp() {
        let op = GateOperation(gate: .hadamard, qubits: [0])
        #expect(!op.description.contains("@"))
        #expect(op.description.contains("H"))
    }

    @Test("Circuit validation fails with invalid gate indices")
    func validationFailsInvalidGate() {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .cnot(control: 0, target: 0), qubits: [])

        #expect(!circuit.validate())
    }

    @Test("Circuit validation passes for valid circuit")
    func validationPassesForValid() {
        var circuit = QuantumCircuit(numQubits: 3)
        circuit.append(gate: .hadamard, qubits: [0])
        circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
        circuit.append(gate: .toffoli(control1: 0, control2: 1, target: 2), qubits: [])

        #expect(circuit.validate())
    }

    @Test("Multi-controlled Z on 4 qubits")
    func multiControlledZ4Qubits() {
        let circuit = QuantumCircuit.qft(numQubits: 4)

        #expect(circuit.numQubits == 4)
        #expect(circuit.gateCount > 0)
        #expect(circuit.validate())
    }

    @Test("Multi-controlled X with no controls")
    func multiControlledXNoControls() {
        let circuit = QuantumCircuit.grover(numQubits: 1, target: 0)

        #expect(circuit.validate())
    }

    @Test("Multi-controlled X with one control")
    func multiControlledXOneControl() {
        let circuit = QuantumCircuit.grover(numQubits: 2, target: 0)

        #expect(circuit.validate())
        #expect(circuit.gateCount > 0)
    }

    @Test("Multi-controlled X with three+ controls")
    func multiControlledXMultipleControls() {
        let circuit = QuantumCircuit.grover(numQubits: 4, target: 5)

        #expect(circuit.numQubits >= 4)
        #expect(circuit.validate())
    }

    @Test("Multi-controlled X: n=0 controls (just X gate)")
    func multiControlledXZeroControls() {
        var circuit = QuantumCircuit(numQubits: 1)
        let controls: [Int] = []
        let target = 0

        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: controls, target: target)

        #expect(circuit.validate())
        #expect(circuit.gateCount == 1, "Should add single X gate")

        let finalState = circuit.execute()
        #expect(abs(finalState.getAmplitude(ofState: 1).real - 1.0) < 1e-10)
    }

    @Test("Multi-controlled X: n=1 control (CNOT)")
    func multiControlledXOneControlDirect() {
        var circuit = QuantumCircuit(numQubits: 2)
        let controls = [0]
        let target = 1

        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: controls, target: target)

        #expect(circuit.validate())
        #expect(circuit.gateCount > 0)

        let finalState = circuit.execute()
        #expect(finalState.isNormalized())
    }

    @Test("Multi-controlled X: n=2 controls branch")
    func multiControlledXTwoControls() {
        var circuit = QuantumCircuit(numQubits: 3)

        let controls = [0, 1]
        let target = 2

        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: controls, target: target)

        #expect(circuit.validate())
        #expect(circuit.gateCount > 0, "Should have decomposed into gates")
    }

    @Test("Multi-controlled X with 3 controls flips when all are |1⟩")
    func multiControlledX3ControlsAllOnes() {
        var circuit = QuantumCircuit(numQubits: 4)
        let controls = [0, 1, 2]
        let target = 3

        for control in controls {
            circuit.append(gate: .pauliX, toQubit: control)
        }

        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: controls, target: target)

        let finalState = circuit.execute()
        let allOnesState = 0b1111

        #expect(abs(finalState.probability(ofState: allOnesState) - 1.0) < 1e-6)
    }

    @Test("Multi-controlled X with 3 controls does not flip with one |0⟩")
    func multiControlledX3ControlsOneZero() {
        var circuit = QuantumCircuit(numQubits: 4)
        let controls = [0, 1, 2]
        let target = 3

        circuit.append(gate: .pauliX, toQubit: controls[1])
        circuit.append(gate: .pauliX, toQubit: controls[2])

        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: controls, target: target)

        let finalState = circuit.execute()
        let expectedState = 0b0110

        #expect(abs(finalState.probability(ofState: expectedState) - 1.0) < 1e-6)
    }

    @Test("Multi-controlled X with 4 controls flips when all are |1⟩")
    func multiControlledX4ControlsAllOnes() {
        var circuit = QuantumCircuit(numQubits: 5)
        let controls = [0, 1, 2, 3]
        let target = 4

        for control in controls {
            circuit.append(gate: .pauliX, toQubit: control)
        }

        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: controls, target: target)

        let finalState = circuit.execute()
        let allOnesState = 0b11111

        #expect(abs(finalState.probability(ofState: allOnesState) - 1.0) < 1e-6)
    }

    @Test("Multi-controlled X with 4 controls does not flip with one |0⟩")
    func multiControlledX4ControlsOneZero() {
        var circuit = QuantumCircuit(numQubits: 5)
        let controls = [0, 1, 2, 3]
        let target = 4

        for i in 1 ..< controls.count {
            circuit.append(gate: .pauliX, toQubit: controls[i])
        }

        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: controls, target: target)

        let finalState = circuit.execute()
        let expectedState = 0b01110

        #expect(abs(finalState.probability(ofState: expectedState) - 1.0) < 1e-6)
    }

    @Test("Multi-controlled X with 5 controls flips when all are |1⟩")
    func multiControlledX5ControlsAllOnes() {
        var circuit = QuantumCircuit(numQubits: 6)
        let controls = [0, 1, 2, 3, 4]
        let target = 5

        for control in controls {
            circuit.append(gate: .pauliX, toQubit: control)
        }

        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: controls, target: target)

        let finalState = circuit.execute()
        let allOnesState = 0b111111

        #expect(abs(finalState.probability(ofState: allOnesState) - 1.0) < 1e-6)
    }

    @Test("Multi-controlled X with 5 controls does not flip with one |0⟩")
    func multiControlledX5ControlsOneZero() {
        var circuit = QuantumCircuit(numQubits: 6)
        let controls = [0, 1, 2, 3, 4]
        let target = 5

        for i in 1 ..< controls.count {
            circuit.append(gate: .pauliX, toQubit: controls[i])
        }

        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: controls, target: target)

        let finalState = circuit.execute()
        let expectedState = 0b011110

        #expect(abs(finalState.probability(ofState: expectedState) - 1.0) < 1e-6)
    }

    @Test("Multi-controlled X with 6 controls validates correctness")
    func multiControlledX6Controls() {
        var circuit = QuantumCircuit(numQubits: 7)
        let controls = [0, 1, 2, 3, 4, 5]
        let target = 6

        for control in controls {
            circuit.append(gate: .pauliX, toQubit: control)
        }

        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: controls, target: target)

        let finalState = circuit.execute()
        let allOnesState = 0b1111111

        #expect(finalState.isNormalized())
        #expect(abs(finalState.probability(ofState: allOnesState) - 1.0) < 1e-6)
    }

    @Test("State expansion with ancilla qubits preserves original amplitudes")
    func stateExpansionPreservesAmplitudes() {
        var setupCircuit = QuantumCircuit(numQubits: 4)
        setupCircuit.append(gate: .pauliX, toQubit: 0)
        setupCircuit.append(gate: .pauliX, toQubit: 1)
        let initialState = setupCircuit.execute()

        var circuit = QuantumCircuit(numQubits: 4)
        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: [0, 1, 2], target: 3)

        let finalState = circuit.execute(on: initialState)

        #expect(finalState.isNormalized())
        #expect(finalState.probability(ofState: 0b0011) > 0.99)
    }

    @Test("State expansion allocates ancilla at zero state")
    func stateExpansionAncillaAtZero() {
        var circuit = QuantumCircuit(numQubits: 4)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .hadamard, toQubit: 1)
        circuit.append(gate: .hadamard, toQubit: 2)

        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: [0, 1, 2], target: 3)

        let finalState = circuit.execute()
        #expect(finalState.isNormalized())

        var probabilityWithAncillaZero = 0.0
        for state in 0 ..< 16 {
            probabilityWithAncillaZero += finalState.probability(ofState: state)
        }
        #expect(abs(probabilityWithAncillaZero - 1.0) < 1e-10, "Ancilla should be uncomputed to |0⟩")
    }

    @Test("maxQubitUsed detects ancilla qubits")
    func maxQubitUsedDetectsAncilla() {
        var circuit = QuantumCircuit(numQubits: 3)
        circuit.append(gate: .hadamard, toQubit: 0)
        #expect(circuit.maxQubitUsed() == 2)

        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: [0, 1], target: 2)
        #expect(circuit.maxQubitUsed() == 2)

        circuit = QuantumCircuit(numQubits: 4)
        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: [0, 1, 2], target: 3)
        #expect(circuit.maxQubitUsed() == 4, "Should detect ancilla qubit 4")
    }

    @Test("State expansion handles multiple ancilla qubits")
    func stateExpansionMultipleAncilla() {
        var circuit = QuantumCircuit(numQubits: 6)

        let controls = [0, 1, 2, 3, 4]
        let target = 5

        for control in controls {
            circuit.append(gate: .pauliX, toQubit: control)
        }

        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: controls, target: target)

        #expect(circuit.maxQubitUsed() == 8, "Should use ancilla qubits 6, 7, 8")

        let finalState = circuit.execute()

        #expect(finalState.isNormalized())
        #expect(finalState.probability(ofState: 0b111111) > 0.99, "Target should flip")
    }

    @Test("State expansion in execute upToIndex with ancilla qubits")
    func stateExpansionExecuteUpToIndex() {
        var setupCircuit = QuantumCircuit(numQubits: 4)
        setupCircuit.append(gate: .pauliX, toQubit: 0)
        setupCircuit.append(gate: .pauliX, toQubit: 1)
        setupCircuit.append(gate: .pauliX, toQubit: 2)
        let initialState = setupCircuit.execute()

        var circuit = QuantumCircuit(numQubits: 4)
        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: [0, 1, 2], target: 3)
        circuit.append(gate: .pauliX, toQubit: 3)

        let partialState = circuit.execute(on: initialState, upToIndex: circuit.gateCount - 1)
        #expect(partialState.isNormalized())
        #expect(partialState.probability(ofState: 0b1111) > 0.99)

        let fullState = circuit.execute(on: initialState)
        #expect(fullState.probability(ofState: 0b0111) > 0.99)
    }

    @Test("maxQubitUsed with empty operations returns numQubits minus 1")
    func maxQubitUsedEmptyCircuit() {
        let circuit = QuantumCircuit(numQubits: 5)
        #expect(circuit.maxQubitUsed() == 4)
    }

    @Test("maxQubitUsed detects single-qubit gate qubits")
    func maxQubitUsedSingleQubitGates() {
        var circuit = QuantumCircuit(numQubits: 3)

        circuit.append(gate: .hadamard, toQubit: 2)
        #expect(circuit.maxQubitUsed() == 2)

        circuit.append(gate: .pauliX, toQubit: 1)
        #expect(circuit.maxQubitUsed() == 2)

        circuit.append(gate: .phase(theta: .pi), toQubit: 0)
        #expect(circuit.maxQubitUsed() == 2)
    }

    @Test("Grover finds target with high probability (5-qubit)")
    func groverFindsTarget5Qubit() {
        let target = 13
        let circuit = QuantumCircuit.grover(numQubits: 5, target: target)
        let initialState = QuantumState(numQubits: circuit.numQubits)
        let finalState = circuit.execute(on: initialState)
        let targetProb = finalState.probability(ofState: target)

        #expect(targetProb > 0.7, "Target probability: \(targetProb)")
    }

    @Test("Append gate with dynamic circuit growth")
    func appendGateDynamicGrowth() {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, qubits: [5])

        #expect(circuit.numQubits == 6, "Circuit should grow to accommodate qubit 5")
        #expect(circuit.gateCount == 1)
        #expect(circuit.validate())
    }

    @Test("Circuit validation with out-of-bounds qubit in operations array")
    func validationOutOfBoundsQubitInArray() {
        var circuit = QuantumCircuit(numQubits: 2)

        circuit.append(gate: .hadamard, qubits: [0])

        let ops = [
            GateOperation(gate: .hadamard, qubits: [5]),
        ]
        let invalidCircuit = QuantumCircuit(numQubits: 2, operations: ops)

        #expect(!invalidCircuit.validate(), "Should detect out-of-bounds qubit")
    }

    @Test("Circuit validation with negative qubit in operations")
    func validationNegativeQubitInOps() {
        let ops = [
            GateOperation(gate: .hadamard, qubits: [-1]),
        ]
        let invalidCircuit = QuantumCircuit(numQubits: 2, operations: ops)

        #expect(!invalidCircuit.validate(), "Should detect negative qubit index")
    }

    @Test("Circuit validation passes empty circuit")
    func validationPassesEmptyCircuit() {
        let circuit = QuantumCircuit(numQubits: 3)
        #expect(circuit.validate(), "Empty circuit should be valid")
    }

    @Test("Circuit description with more than 5 gates")
    func circuitDescriptionLongCircuit() {
        var circuit = QuantumCircuit(numQubits: 2)

        for _ in 0 ..< 7 {
            circuit.append(gate: .hadamard, toQubit: 0)
        }

        let desc = circuit.description
        #expect(desc.contains("..."), "Description should contain '...' for circuits with >5 gates")
        #expect(desc.contains("7 gates"))
    }

    @Test("maxQubitUsed with single-qubit gate having empty qubits array")
    func maxQubitUsedEmptyQubitArray() {
        var circuit = QuantumCircuit(numQubits: 3)
        let ops = [GateOperation(gate: .hadamard, qubits: [])]
        circuit = QuantumCircuit(numQubits: 3, operations: ops)

        #expect(circuit.maxQubitUsed() == 2)
    }

    @Test("Multi-controlled X with empty controls array")
    func multiControlledXEmptyControlsArray() {
        var circuit = QuantumCircuit(numQubits: 2)
        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: [], target: 0)

        #expect(circuit.gateCount == 1)
        let state = circuit.execute()
        #expect(state.probability(ofState: 0b01) > 0.99, "Target qubit should be flipped")
    }

    @Test("appendMultiControlledX with empty controls array")
    func multiControlledXEmptyControls() {
        var circuit = QuantumCircuit(numQubits: 2)
        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: [], target: 1)

        let finalState = circuit.execute()
        #expect(finalState.probability(ofState: 0b10) > 0.99)
    }
}
