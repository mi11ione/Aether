// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Test suite for QuantumSimulator actor.
/// Validates Swift concurrency features: async execution
/// progress reporting and thread-safe quantum circuit simulation.
@Suite("Quantum Simulator")
struct QuantumSimulatorTests {
    @Test("Simulator executes simple circuit asynchronously")
    func simulatorExecutesSimpleCircuit() async {
        let simulator = QuantumSimulator()

        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let finalState = await simulator.execute(circuit)

        #expect(finalState.numQubits == 2)
        #expect(finalState.isNormalized())

        let p0 = finalState.probability(of: 0)
        let p3 = finalState.probability(of: 3)
        #expect(abs(p0 - 0.5) < 1e-10)
        #expect(abs(p3 - 0.5) < 1e-10)
    }

    @Test("Simulator executes from custom initial state")
    func simulatorExecutesFromCustomState() async {
        let simulator = QuantumSimulator()

        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(.pauliX, to: 0)

        let initialState = QuantumState(numQubits: 1)
        let finalState = await simulator.execute(circuit, from: initialState)

        #expect(abs(finalState.probability(of: 1) - 1.0) < 1e-10)
    }

    @Test("Simulator reports progress during execution")
    func simulatorReportsProgress() async {
        let simulator = QuantumSimulator()

        var circuit = QuantumCircuit(numQubits: 2)
        for _ in 0 ..< 20 {
            circuit.append(.hadamard, to: 0)
        }

        actor ProgressAccumulator {
            private(set) var values: [Double] = []
            func append(_ value: Double) { values.append(value) }
        }

        let accumulator = ProgressAccumulator()

        _ = await simulator.execute(circuit, progressHandler: { progress in
            await accumulator.append(progress)
        })

        let progressUpdates = await accumulator.values

        #expect(progressUpdates.count > 0)

        for i in 1 ..< progressUpdates.count {
            #expect(progressUpdates[i] >= progressUpdates[i - 1])
        }

        if let lastProgress = progressUpdates.last {
            #expect(lastProgress > 0.9)
        }
    }

    @Test("Simulator handles empty circuit")
    func simulatorHandlesEmptyCircuit() async {
        let simulator = QuantumSimulator()
        let circuit = QuantumCircuit(numQubits: 2)

        let finalState = await simulator.execute(circuit)

        #expect(finalState.probability(of: 0) == 1.0)
    }

    @Test("Simulator handles pre-built Bell state circuit")
    func simulatorHandlesBellStateCircuit() async {
        let simulator = QuantumSimulator()
        let circuit = QuantumCircuit.bell()
        let finalState = await simulator.execute(circuit)

        #expect(finalState.isNormalized())
        #expect(abs(finalState.probability(of: 0) - 0.5) < 1e-10)
        #expect(abs(finalState.probability(of: 3) - 0.5) < 1e-10)
    }

    @Test("Simulator handles QFT circuit")
    func simulatorHandlesQFTCircuit() async {
        let simulator = QuantumSimulator()
        let circuit = QuantumCircuit.qft(numQubits: 3)
        let finalState = await simulator.execute(circuit)

        #expect(finalState.isNormalized())
        #expect(finalState.numQubits == 3)
    }

    @Test("Simulator handles Grover circuit")
    func simulatorHandlesGroverCircuit() async {
        let simulator = QuantumSimulator()
        let circuit = QuantumCircuit.grover(numQubits: 2, target: 3)
        let finalState = await simulator.execute(circuit)

        #expect(finalState.isNormalized())

        let targetProb = finalState.probability(of: 3)
        #expect(targetProb > 0.8)
    }

    @Test("Simulator with Metal disabled works correctly")
    func simulatorWithoutMetalWorks() async {
        let simulator = QuantumSimulator(useMetalAcceleration: false)

        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let finalState = await simulator.execute(circuit)

        #expect(finalState.numQubits == 2)
        #expect(finalState.isNormalized())
    }

    @Test("Simulator uses Metal acceleration for large circuits")
    func simulatorUsesMetalAcceleration() async {
        guard MetalGateApplication() != nil else { return }

        let simulator = QuantumSimulator(useMetalAcceleration: true)

        var circuit = QuantumCircuit(numQubits: 12)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.pauliZ, to: 2)

        let finalState = await simulator.execute(circuit)

        #expect(finalState.numQubits == 12)
        #expect(finalState.isNormalized())
    }

    @Test("Simulator uses Metal acceleration with progress reporting")
    func simulatorUsesMetalWithProgress() async {
        guard MetalGateApplication() != nil else { return }

        let simulator = QuantumSimulator(useMetalAcceleration: true)

        var circuit = QuantumCircuit(numQubits: 12)
        for _ in 0 ..< 8 {
            circuit.append(.hadamard, to: 0)
        }

        actor ProgressTracker {
            private(set) var progressUpdates: [Double] = []
            func append(_ value: Double) { progressUpdates.append(value) }
        }

        let tracker = ProgressTracker()

        let finalState = await simulator.execute(circuit, progressHandler: { progress in
            await tracker.append(progress)
        })

        #expect(finalState.numQubits == 12)
        #expect(finalState.isNormalized())

        let updates = await tracker.progressUpdates
        #expect(updates.count > 0)

        if let lastProgress = updates.last {
            #expect(lastProgress >= 0.99)
        }
    }

    @Test("Simulator executes with progress from custom initial state")
    func simulatorExecutesWithProgressFromCustomState() async {
        let simulator = QuantumSimulator()

        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        var initialState = QuantumState(numQubits: 2)
        initialState.setAmplitude(1, to: .one)
        initialState.setAmplitude(0, to: .zero)

        actor ProgressTracker {
            private(set) var progressCalled = false
            func markCalled() { progressCalled = true }
        }

        let tracker = ProgressTracker()

        let finalState = await simulator.execute(circuit, from: initialState, progressHandler: { _ in
            await tracker.markCalled()
        })

        #expect(finalState.numQubits == 2)
        #expect(finalState.isNormalized())
        #expect(await tracker.progressCalled)
    }
}
