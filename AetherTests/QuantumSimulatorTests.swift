//
//  QuantumSimulatorTests.swift
//  AetherTests
//
//  Test suite for QuantumSimulator actor.
//  Validates async/await execution, progress reporting, cancellation,
//  batch processing, and thread-safe concurrent circuit evaluation.
//
//  Created by mi11ion on 23/10/25.
//

@testable import Aether
import Testing

/// Test suite for QuantumSimulator actor.
/// Validates Swift concurrency features: async execution, progress reporting,
/// task cancellation, and thread-safe quantum circuit simulation.
@Suite("Quantum Simulator Tests")
struct QuantumSimulatorTests {
    @Test("Simulator executes simple circuit asynchronously")
    func simulatorExecutesSimpleCircuit() async throws {
        let simulator = await QuantumSimulator()

        var circuit = await QuantumCircuit(numQubits: 2)
        await circuit.append(gate: .hadamard, toQubit: 0)
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
        let simulator = await QuantumSimulator()

        var circuit = await QuantumCircuit(numQubits: 1)
        await circuit.append(gate: .pauliX, toQubit: 0)

        let initialState = await QuantumState(numQubits: 1) // |0⟩
        let finalState = try await simulator.execute(circuit, from: initialState)

        #expect(abs(finalState.probability(ofState: 1) - 1.0) < 1e-10)
    }

    @Test("Simulator reports progress during execution")
    func simulatorReportsProgress() async throws {
        let simulator = await QuantumSimulator()

        var circuit = await QuantumCircuit(numQubits: 2)
        for _ in 0 ..< 20 {
            await circuit.append(gate: .hadamard, toQubit: 0)
        }

        actor ProgressAccumulator {
            private(set) var values: [Double] = []
            func append(_ value: Double) { values.append(value) }
        }

        let accumulator = ProgressAccumulator()

        _ = try await simulator.executeWithProgress(circuit) { progress in
            await accumulator.append(progress)
        }

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
        let simulator = await QuantumSimulator()

        var circuit = await QuantumCircuit(numQubits: 1)
        for _ in 0 ..< 10 {
            await circuit.append(gate: .hadamard, toQubit: 0)
        }

        Task {
            let _ = try await simulator.execute(circuit)
        }

        try await Task.sleep(for: .milliseconds(10))

        let (executed, total, percentage) = await simulator.getProgress()

        #expect(total == 10)
        #expect(executed >= 0)
        #expect(percentage >= 0.0 && percentage <= 1.0)
    }

    @Test("Simulator provides current state during execution")
    func simulatorProvidesCurrentState() async throws {
        let simulator = await QuantumSimulator()

        var circuit = await QuantumCircuit(numQubits: 2)
        for _ in 0 ..< 50 {
            await circuit.append(gate: .hadamard, toQubit: 0)
        }

        Task {
            let _ = try await simulator.execute(circuit)
        }

        try await Task.sleep(for: .milliseconds(10))

        let currentState = await simulator.getCurrentState()

        if let state = currentState {
            #expect(state.numQubits == 2)
            #expect(state.isNormalized())
        }
    }

    @Test("Simulator tracks execution status")
    func simulatorTracksExecutionStatus() async throws {
        let simulator = await QuantumSimulator()

        var circuit = await QuantumCircuit(numQubits: 1)
        for _ in 0 ..< 100 {
            await circuit.append(gate: .hadamard, toQubit: 0)
        }

        let beforeExecuting = await simulator.isCurrentlyExecuting()
        #expect(beforeExecuting == false)

        Task {
            let _ = try await simulator.execute(circuit)
        }

        try await Task.sleep(for: .milliseconds(100))
        let afterExecuting = await simulator.isCurrentlyExecuting()
        #expect(afterExecuting == false)
    }

    @Test("Simulator executes batch of circuits in parallel")
    func simulatorExecutesBatchCircuits() async throws {
        let simulator = await QuantumSimulator()

        var circuit1 = await QuantumCircuit(numQubits: 2)
        await circuit1.append(gate: .hadamard, toQubit: 0)

        var circuit2 = await QuantumCircuit(numQubits: 2)
        await circuit2.append(gate: .pauliX, toQubit: 1)

        var circuit3 = await QuantumCircuit(numQubits: 2)
        await circuit3.append(gate: .hadamard, toQubit: 0)
        await circuit3.append(gate: .cnot(control: 0, target: 1), qubits: [])

        let circuits = [circuit1, circuit2, circuit3]

        let results = try await simulator.executeBatch(circuits)

        #expect(results.count == 3)

        for state in results {
            #expect(state.isNormalized())
        }

        #expect(results[0].numQubits == 2)
        #expect(results[1].numQubits == 2)
        #expect(results[2].numQubits == 2)
    }

    @Test("Circuit can be executed asynchronously via convenience method")
    func circuitExecutesAsyncConvenience() async throws {
        var circuit = await QuantumCircuit(numQubits: 1)
        await circuit.append(gate: .hadamard, toQubit: 0)

        let finalState = try await circuit.executeAsync()

        #expect(finalState.isNormalized())
        #expect(abs(finalState.probability(ofState: 0) - 0.5) < 1e-10)
    }

    @Test("Circuit async execution with progress handler")
    func circuitExecutesAsyncWithProgress() async throws {
        var circuit = await QuantumCircuit(numQubits: 1)
        for _ in 0 ..< 15 {
            await circuit.append(gate: .hadamard, toQubit: 0)
        }

        var progressCalled = false

        _ = try await circuit.executeAsync { _ in
            progressCalled = true
        }

        #expect(progressCalled)
    }

    @Test("Simulator handles empty circuit")
    func simulatorHandlesEmptyCircuit() async throws {
        let simulator = await QuantumSimulator()
        let circuit = await QuantumCircuit(numQubits: 2)
        let finalState = try await simulator.execute(circuit)

        #expect(finalState.probability(ofState: 0) == 1.0)
    }

    @Test("Simulator supports task cancellation")
    func simulatorSupportsCancellation() async throws {
        let simulator = await QuantumSimulator()

        var circuit = await QuantumCircuit(numQubits: 1)
        for _ in 0 ..< 1000 {
            await circuit.append(gate: .hadamard, toQubit: 0)
        }

        let task = Task {
            try await simulator.execute(circuit)
        }

        task.cancel()

        do {
            _ = try await task.value
        } catch {
            #expect(error is CancellationError)
        }
    }

    @Test("Simulator handles pre-built Bell state circuit")
    func simulatorHandlesBellStateCircuit() async throws {
        let simulator = await QuantumSimulator()
        let circuit = await QuantumCircuit.bellState()
        let finalState = try await simulator.execute(circuit)

        #expect(finalState.isNormalized())
        #expect(abs(finalState.probability(ofState: 0) - 0.5) < 1e-10)
        #expect(abs(finalState.probability(ofState: 3) - 0.5) < 1e-10)
    }

    @Test("Simulator handles QFT circuit")
    func simulatorHandlesQFTCircuit() async throws {
        let simulator = await QuantumSimulator()
        let circuit = await QuantumCircuit.qft(numQubits: 3)
        let finalState = try await simulator.execute(circuit)

        #expect(finalState.isNormalized())
        #expect(finalState.numQubits == 3)
    }

    @Test("Simulator handles Grover circuit")
    func simulatorHandlesGroverCircuit() async throws {
        let simulator = await QuantumSimulator()
        let circuit = await QuantumCircuit.grover(numQubits: 2, target: 3)
        let finalState = try await simulator.execute(circuit)

        #expect(finalState.isNormalized())

        let targetProb = finalState.probability(ofState: 3)
        #expect(targetProb > 0.8)
    }

    @Test("SimulatorError has correct descriptions")
    func simulatorErrorDescriptions() {
        let error1 = SimulatorError.alreadyExecuting
        let error2 = SimulatorError.invalidCircuit
        let error3 = SimulatorError.metalNotAvailable

        #expect(error1.errorDescription != nil)
        #expect(error2.errorDescription != nil)
        #expect(error3.errorDescription != nil)
        #expect(error1.errorDescription!.contains("executing"))
        #expect(error2.errorDescription!.contains("invalid"))
        #expect(error3.errorDescription!.contains("Metal"))
    }

    @Test("Simulator throws alreadyExecuting when concurrent execution attempted")
    func simulatorThrowsAlreadyExecuting() async throws {
        let simulator = await QuantumSimulator()

        var circuit = await QuantumCircuit(numQubits: 1)
        for _ in 0 ..< 1000 {
            await circuit.append(gate: .hadamard, toQubit: 0)
        }

        let task1 = Task {
            try await simulator.execute(circuit)
        }

        try await Task.sleep(for: .milliseconds(10))

        do {
            _ = try await simulator.execute(circuit)
            Issue.record("Expected alreadyExecuting error")
        } catch let error as SimulatorError {
            #expect(error == .alreadyExecuting)
        }

        task1.cancel()
        _ = try? await task1.value
    }

    @Test("Simulator throws alreadyExecuting for executeWithProgress during concurrent execution")
    func simulatorThrowsAlreadyExecutingWithProgress() async throws {
        let simulator = await QuantumSimulator()

        var circuit = await QuantumCircuit(numQubits: 1)
        for _ in 0 ..< 1000 {
            await circuit.append(gate: .hadamard, toQubit: 0)
        }

        let task1 = Task {
            try await simulator.executeWithProgress(circuit) { _ in }
        }

        try await Task.sleep(for: .milliseconds(10))

        do {
            _ = try await simulator.executeWithProgress(circuit) { _ in }
            Issue.record("Expected alreadyExecuting error")
        } catch let error as SimulatorError {
            #expect(error == .alreadyExecuting)
        }

        task1.cancel()
        _ = try? await task1.value
    }

    @Test("Simulator with Metal disabled works correctly")
    func simulatorWithoutMetalWorks() async throws {
        let simulator = await QuantumSimulator(useMetalAcceleration: false)

        var circuit = await QuantumCircuit(numQubits: 2)
        await circuit.append(gate: .hadamard, toQubit: 0)
        await circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])

        let finalState = try await simulator.execute(circuit)

        #expect(finalState.numQubits == 2)
        #expect(finalState.isNormalized())
    }

    @Test("Progress handler receives final progress value")
    func progressHandlerReceivesFinalProgress() async throws {
        let simulator = await QuantumSimulator()

        var circuit = await QuantumCircuit(numQubits: 1)
        for _ in 0 ..< 10 {
            await circuit.append(gate: .hadamard, toQubit: 0)
        }

        actor FinalProgressTracker {
            private(set) var finalProgress: Double = 0.0
            func update(_ value: Double) { finalProgress = value }
        }

        let tracker = FinalProgressTracker()

        _ = try await simulator.executeWithProgress(circuit) { progress in
            await tracker.update(progress)
        }

        let final = await tracker.finalProgress
        #expect(final >= 0.99)
    }

    @Test("Simulator uses Metal acceleration for large circuits")
    func simulatorUsesMetalAcceleration() async throws {
        let simulator = await QuantumSimulator(useMetalAcceleration: true)

        var circuit = await QuantumCircuit(numQubits: 12)
        await circuit.append(gate: .hadamard, toQubit: 0)
        await circuit.append(gate: .pauliX, toQubit: 1)
        await circuit.append(gate: .pauliZ, toQubit: 2)

        let finalState = try await simulator.execute(circuit)

        #expect(finalState.numQubits == 12)
        #expect(finalState.isNormalized())
    }

    @Test("Simulator uses Metal acceleration with progress reporting")
    func simulatorUsesMetalWithProgress() async throws {
        let simulator = await QuantumSimulator(useMetalAcceleration: true)

        var circuit = await QuantumCircuit(numQubits: 12)
        for _ in 0 ..< 8 {
            await circuit.append(gate: .hadamard, toQubit: 0)
        }

        actor ProgressTracker {
            private(set) var progressUpdates: [Double] = []
            func append(_ value: Double) { progressUpdates.append(value) }
        }

        let tracker = ProgressTracker()

        let finalState = try await simulator.executeWithProgress(circuit) { progress in
            await tracker.append(progress)
        }

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
        let simulator = await QuantumSimulator()

        var circuit = await QuantumCircuit(numQubits: 2)
        await circuit.append(gate: .hadamard, toQubit: 0)
        await circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])

        var initialState = await QuantumState(numQubits: 2)
        await initialState.setAmplitude(ofState: 1, amplitude: .one)
        await initialState.setAmplitude(ofState: 0, amplitude: .zero)

        actor ProgressTracker {
            private(set) var progressCalled = false
            func markCalled() { progressCalled = true }
        }

        let tracker = ProgressTracker()

        let finalState = try await simulator.executeWithProgress(circuit, from: initialState) { _ in
            await tracker.markCalled()
        }

        #expect(finalState.numQubits == 2)
        #expect(finalState.isNormalized())
        #expect(await tracker.progressCalled)
    }
}
