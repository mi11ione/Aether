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
    func simulatorExecutesSimpleCircuit() async throws {
        let simulator = QuantumSimulator()

        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        await circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])

        let finalState = try await simulator.execute(circuit)

        #expect(finalState.numQubits == 2)
        #expect(finalState.isNormalized())

        let p0 = finalState.probability(ofState: 0)
        let p3 = finalState.probability(ofState: 3)
        #expect(abs(p0 - 0.5) < 1e-10)
        #expect(abs(p3 - 0.5) < 1e-10)
    }

    @Test("Simulator executes from custom initial state")
    func simulatorExecutesFromCustomState() async throws {
        let simulator = QuantumSimulator()

        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .pauliX, toQubit: 0)

        let initialState = QuantumState(numQubits: 1)
        let finalState = try await simulator.execute(circuit, from: initialState)

        #expect(abs(finalState.probability(ofState: 1) - 1.0) < 1e-10)
    }

    @Test("Simulator reports progress during execution")
    func simulatorReportsProgress() async throws {
        let simulator = QuantumSimulator()

        var circuit = QuantumCircuit(numQubits: 2)
        for _ in 0 ..< 20 {
            circuit.append(gate: .hadamard, toQubit: 0)
        }

        actor ProgressAccumulator {
            private(set) var values: [Double] = []
            func append(_ value: Double) { values.append(value) }
        }

        let accumulator = ProgressAccumulator()

        _ = try await simulator.execute(circuit, progressHandler: { progress in
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

    @Test("Simulator tracks execution progress")
    func simulatorTracksProgress() async throws {
        let simulator = QuantumSimulator()

        var circuit = QuantumCircuit(numQubits: 1)
        for _ in 0 ..< 10 {
            circuit.append(gate: .hadamard, toQubit: 0)
        }

        actor ProgressTracker {
            private(set) var capturedProgress: (executed: Int, total: Int, percentage: Double)?
            func capture(_ progress: (Int, Int, Double)) { capturedProgress = progress }
        }

        let tracker = ProgressTracker()

        _ = try await simulator.execute(circuit, progressHandler: { _ in
            let progress = await simulator.getProgress()
            await tracker.capture(progress)
        })

        let captured = await tracker.capturedProgress
        #expect(captured != nil)
        if let progress = captured {
            #expect(progress.total == 10)
            #expect(progress.executed > 0)
            #expect(progress.percentage > 0.0 && progress.percentage <= 1.0)
        }
    }

    @Test("Simulator provides current state during execution")
    func simulatorProvidesCurrentState() async throws {
        let simulator = QuantumSimulator()

        var circuit = QuantumCircuit(numQubits: 2)
        for _ in 0 ..< 50 {
            circuit.append(gate: .hadamard, toQubit: 0)
        }

        actor StateTracker {
            private(set) var capturedState: QuantumState?
            func capture(_ state: QuantumState?) { capturedState = state }
        }

        let tracker = StateTracker()

        _ = try await simulator.execute(circuit, progressHandler: { _ in
            let state = await simulator.getCurrentState()
            await tracker.capture(state)
        })

        let state = await tracker.capturedState
        #expect(state != nil)
        if let capturedState = state {
            #expect(capturedState.numQubits == 2)
            #expect(capturedState.isNormalized())
        }
    }

    @Test("Circuit can be executed asynchronously via convenience method")
    func circuitExecutesAsyncConvenience() async throws {
        var circuit = QuantumCircuit(numQubits: 1)
        circuit.append(gate: .hadamard, toQubit: 0)

        let finalState = try await circuit.executeAsync()

        #expect(finalState.isNormalized())
        #expect(abs(finalState.probability(ofState: 0) - 0.5) < 1e-10)
    }

    @Test("Circuit async execution with progress handler")
    func circuitExecutesAsyncWithProgress() async throws {
        var circuit = QuantumCircuit(numQubits: 1)
        for _ in 0 ..< 15 {
            circuit.append(gate: .hadamard, toQubit: 0)
        }

        actor ProgressFlag {
            private(set) var called = false
            func mark() { called = true }
        }

        let flag = ProgressFlag()

        _ = try await circuit.executeAsync { _ in
            await flag.mark()
        }

        #expect(await flag.called)
    }

    @Test("Simulator handles empty circuit")
    func simulatorHandlesEmptyCircuit() async throws {
        let simulator = QuantumSimulator()
        let circuit = QuantumCircuit(numQubits: 2)

        let progress = await simulator.getProgress()
        #expect(progress.total == 0)
        #expect(progress.percentage == 0.0)

        let finalState = try await simulator.execute(circuit)

        #expect(finalState.probability(ofState: 0) == 1.0)
    }

    @Test("Simulator handles pre-built Bell state circuit")
    func simulatorHandlesBellStateCircuit() async throws {
        let simulator = QuantumSimulator()
        let circuit = QuantumCircuit.bellState()
        let finalState = try await simulator.execute(circuit)

        #expect(finalState.isNormalized())
        #expect(abs(finalState.probability(ofState: 0) - 0.5) < 1e-10)
        #expect(abs(finalState.probability(ofState: 3) - 0.5) < 1e-10)
    }

    @Test("Simulator handles QFT circuit")
    func simulatorHandlesQFTCircuit() async throws {
        let simulator = QuantumSimulator()
        let circuit = QuantumCircuit.qft(numQubits: 3)
        let finalState = try await simulator.execute(circuit)

        #expect(finalState.isNormalized())
        #expect(finalState.numQubits == 3)
    }

    @Test("Simulator handles Grover circuit")
    func simulatorHandlesGroverCircuit() async throws {
        let simulator = QuantumSimulator()
        let circuit = QuantumCircuit.grover(numQubits: 2, target: 3)
        let finalState = try await simulator.execute(circuit)

        #expect(finalState.isNormalized())

        let targetProb = finalState.probability(ofState: 3)
        #expect(targetProb > 0.8)
    }

    @Test("SimulatorError has correct descriptions")
    func simulatorErrorDescriptions() {
        let error1 = SimulatorError.invalidCircuit
        let error2 = SimulatorError.metalNotAvailable

        #expect(error1.errorDescription != nil)
        #expect(error2.errorDescription != nil)
        #expect(error1.errorDescription?.contains("invalid") == true)
        #expect(error2.errorDescription?.contains("Metal") == true)
    }

    @Test("Simulator with Metal disabled works correctly")
    func simulatorWithoutMetalWorks() async throws {
        let simulator = QuantumSimulator(useMetalAcceleration: false)

        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        await circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])

        let finalState = try await simulator.execute(circuit)

        #expect(finalState.numQubits == 2)
        #expect(finalState.isNormalized())
    }

    @Test("Simulator uses Metal acceleration for large circuits")
    func simulatorUsesMetalAcceleration() async throws {
        _ = try #require(MetalGateApplication(), "Metal not available on this device")

        let simulator = QuantumSimulator(useMetalAcceleration: true)

        var circuit = QuantumCircuit(numQubits: 12)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .pauliX, toQubit: 1)
        circuit.append(gate: .pauliZ, toQubit: 2)

        let finalState = try await simulator.execute(circuit)

        #expect(finalState.numQubits == 12)
        #expect(finalState.isNormalized())
    }

    @Test("Simulator uses Metal acceleration with progress reporting")
    func simulatorUsesMetalWithProgress() async throws {
        _ = try #require(MetalGateApplication(), "Metal not available on this device")

        let simulator = QuantumSimulator(useMetalAcceleration: true)

        var circuit = QuantumCircuit(numQubits: 12)
        for _ in 0 ..< 8 {
            circuit.append(gate: .hadamard, toQubit: 0)
        }

        actor ProgressTracker {
            private(set) var progressUpdates: [Double] = []
            func append(_ value: Double) { progressUpdates.append(value) }
        }

        let tracker = ProgressTracker()

        let finalState = try await simulator.execute(circuit, progressHandler: { progress in
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
    func simulatorExecutesWithProgressFromCustomState() async throws {
        let simulator = QuantumSimulator()

        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])

        var initialState = QuantumState(numQubits: 2)
        initialState.setAmplitude(ofState: 1, amplitude: .one)
        initialState.setAmplitude(ofState: 0, amplitude: .zero)

        actor ProgressTracker {
            private(set) var progressCalled = false
            func markCalled() { progressCalled = true }
        }

        let tracker = ProgressTracker()

        let finalState = try await simulator.execute(circuit, from: initialState, progressHandler: { _ in
            await tracker.markCalled()
        })

        #expect(finalState.numQubits == 2)
        #expect(finalState.isNormalized())
        #expect(await tracker.progressCalled)
    }

    @Test("Simulator handles state expansion with ancilla qubits")
    func simulatorHandlesStateExpansionWithAncilla() async throws {
        let simulator = QuantumSimulator()
        var circuit = QuantumCircuit(numQubits: 4)

        circuit.append(gate: .pauliX, toQubit: 0)
        circuit.append(gate: .pauliX, toQubit: 1)
        circuit.append(gate: .pauliX, toQubit: 2)
        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: [0, 1, 2], target: 3)

        let finalState = try await simulator.execute(circuit)

        #expect(finalState.isNormalized())
        #expect(finalState.probability(ofState: 0b1111) > 0.99, "Target should flip")
    }

    @Test("Simulator state expansion preserves amplitudes")
    func simulatorStateExpansionPreservesAmplitudes() async throws {
        let simulator = QuantumSimulator()

        var initialCircuit = QuantumCircuit(numQubits: 4)
        initialCircuit.append(gate: .pauliX, toQubit: 0)
        initialCircuit.append(gate: .hadamard, toQubit: 1)
        let initialState = initialCircuit.execute()

        var circuit = QuantumCircuit(numQubits: 4)
        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: [0, 1, 2], target: 3)

        let finalState = try await simulator.execute(circuit, from: initialState)
        #expect(finalState.isNormalized())

        let p0001 = finalState.probability(ofState: 0b0001)
        let p0011 = finalState.probability(ofState: 0b0011)
        #expect(abs(p0001 - 0.5) < 1e-10, "State |0001⟩ probability")
        #expect(abs(p0011 - 0.5) < 1e-10, "State |0011⟩ probability")
    }

    @Test("Simulator executeWithProgress handles ancilla expansion")
    func simulatorExecuteWithProgressHandlesAncilla() async throws {
        let simulator = QuantumSimulator()
        var circuit = QuantumCircuit(numQubits: 5)

        circuit.append(gate: .pauliX, toQubit: 0)
        circuit.append(gate: .pauliX, toQubit: 1)
        circuit.append(gate: .pauliX, toQubit: 2)
        circuit.append(gate: .pauliX, toQubit: 3)

        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: [0, 1, 2, 3], target: 4)

        actor ProgressTracker {
            private(set) var progressCalled = false
            func markCalled() { progressCalled = true }
        }

        let tracker = ProgressTracker()

        let finalState = try await simulator.execute(circuit, progressHandler: { _ in
            await tracker.markCalled()
        })

        #expect(finalState.isNormalized())
        #expect(finalState.probability(ofState: 0b11111) > 0.99)
        #expect(await tracker.progressCalled)
    }

    @Test("Simulator handles multiple ancilla qubits correctly")
    func simulatorHandlesMultipleAncilla() async throws {
        let simulator = QuantumSimulator()

        var circuit = QuantumCircuit(numQubits: 6)

        for qubit in 0 ..< 5 {
            circuit.append(gate: .pauliX, toQubit: qubit)
        }

        QuantumCircuit.appendMultiControlledX(to: &circuit, controls: [0, 1, 2, 3, 4], target: 5)

        let finalState = try await simulator.execute(circuit)

        #expect(finalState.isNormalized())
        #expect(finalState.probability(ofState: 0b111111) > 0.99, "All qubits should be |1⟩")
    }
}
