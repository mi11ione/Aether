// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for DebugSnapshot value type.
/// Validates snapshot properties including operation index, state capture,
/// and timing information for step-through debugging.
@Suite("DebugSnapshot")
struct DebugSnapshotTests {
    @Test("Initial snapshot has index -1 and nil operation")
    func initialSnapshotProperties() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let debug = DebugExecution(circuit: circuit)

        let initialSnapshot = debug.trace[0]
        #expect(initialSnapshot.index == -1, "Initial snapshot should have index -1")
        #expect(initialSnapshot.operation == nil, "Initial snapshot should have nil operation")
        #expect(initialSnapshot.elapsedNs == 0, "Initial snapshot should have zero elapsed time")
    }

    @Test("Snapshot after step has correct index and operation")
    func snapshotAfterStep() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        var debug = DebugExecution(circuit: circuit)

        let snapshot = debug.step()
        #expect(snapshot.index == 0, "First step snapshot should have index 0")
        #expect(snapshot.operation != nil, "Step snapshot should have non-nil operation")
        #expect(snapshot.operation?.gate == .hadamard, "Operation gate should be Hadamard")
    }

    @Test("Snapshot state reflects applied operation")
    func snapshotStateCorrectness() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        var debug = DebugExecution(circuit: circuit)

        let snapshot = debug.step()
        let prob1 = snapshot.state.probability(of: 1)
        #expect(abs(prob1 - 1.0) < 1e-10, "After X gate, |1⟩ probability should be 1.0")
    }

    @Test("DebugSnapshot conforms to Equatable")
    func snapshotEquatable() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let debug1 = DebugExecution(circuit: circuit)
        let debug2 = DebugExecution(circuit: circuit)

        let snap1 = debug1.trace[0]
        let snap2 = debug2.trace[0]
        #expect(snap1 == snap2, "Initial snapshots from identical circuits should be equal")
    }
}

/// Test suite for QubitAmplitudeBreakdown computation.
/// Validates probability extraction and Bloch vector calculation
/// from reduced single-qubit density matrices.
@Suite("QubitAmplitudeBreakdown")
struct QubitAmplitudeBreakdownTests {
    @Test("Ground state gives p0=1, p1=0, Bloch z=1")
    func groundStateBreakdown() {
        let circuit = QuantumCircuit(qubits: 1)
        let debug = DebugExecution(circuit: circuit)

        let breakdown = debug.amplitudes(qubit: 0)
        #expect(breakdown.qubit == 0, "Breakdown should report correct qubit index")
        #expect(abs(breakdown.p0 - 1.0) < 1e-10, "Ground state p0 should be 1.0")
        #expect(abs(breakdown.p1) < 1e-10, "Ground state p1 should be 0.0")
        #expect(abs(breakdown.blochVector.z - 1.0) < 1e-10, "Ground state Bloch z should be 1.0")
        #expect(abs(breakdown.blochVector.x) < 1e-10, "Ground state Bloch x should be 0.0")
        #expect(abs(breakdown.blochVector.y) < 1e-10, "Ground state Bloch y should be 0.0")
    }

    @Test("Excited state gives p0=0, p1=1, Bloch z=-1")
    func excitedStateBreakdown() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        var debug = DebugExecution(circuit: circuit)
        _ = debug.step()

        let breakdown = debug.amplitudes(qubit: 0)
        #expect(abs(breakdown.p0) < 1e-10, "Excited state p0 should be 0.0")
        #expect(abs(breakdown.p1 - 1.0) < 1e-10, "Excited state p1 should be 1.0")
        #expect(abs(breakdown.blochVector.z + 1.0) < 1e-10, "Excited state Bloch z should be -1.0")
    }

    @Test("Hadamard state gives equal probabilities and Bloch x=1")
    func hadamardStateBreakdown() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        var debug = DebugExecution(circuit: circuit)
        _ = debug.step()

        let breakdown = debug.amplitudes(qubit: 0)
        #expect(abs(breakdown.p0 - 0.5) < 1e-10, "Hadamard state p0 should be 0.5")
        #expect(abs(breakdown.p1 - 0.5) < 1e-10, "Hadamard state p1 should be 0.5")
        #expect(abs(breakdown.blochVector.x - 1.0) < 1e-10, "Hadamard state Bloch x should be 1.0")
        #expect(abs(breakdown.blochVector.z) < 1e-10, "Hadamard state Bloch z should be 0.0")
    }

    @Test("Y-basis state gives Bloch y component")
    func yBasisStateBreakdown() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.sGate, to: 0)
        var debug = DebugExecution(circuit: circuit)
        _ = debug.step(count: 2)

        let breakdown = debug.amplitudes(qubit: 0)
        #expect(abs(breakdown.p0 - 0.5) < 1e-10, "|+i⟩ state p0 should be 0.5")
        #expect(abs(breakdown.p1 - 0.5) < 1e-10, "|+i⟩ state p1 should be 0.5")
        #expect(abs(breakdown.blochVector.y - 1.0) < 1e-10, "|+i⟩ state Bloch y should be 1.0")
    }

    @Test("Multi-qubit breakdown traces out other qubits")
    func multiQubitBreakdown() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        var debug = DebugExecution(circuit: circuit)
        _ = debug.step(count: 2)

        let breakdown0 = debug.amplitudes(qubit: 0)
        let breakdown1 = debug.amplitudes(qubit: 1)
        #expect(abs(breakdown0.p0 - 0.5) < 1e-10, "Entangled qubit 0 p0 should be 0.5")
        #expect(abs(breakdown0.p1 - 0.5) < 1e-10, "Entangled qubit 0 p1 should be 0.5")
        #expect(abs(breakdown1.p0 - 0.5) < 1e-10, "Entangled qubit 1 p0 should be 0.5")
        #expect(abs(breakdown1.p1 - 0.5) < 1e-10, "Entangled qubit 1 p1 should be 0.5")
    }

    @Test("Amplitudes returns valid breakdown with correct qubit index")
    func amplitudesReturnsValidBreakdown() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.pauliX, to: 1)
        var debug = DebugExecution(circuit: circuit)
        _ = debug.step()

        let breakdown = debug.amplitudes(qubit: 1)
        #expect(breakdown.qubit == 1, "Breakdown should report qubit index 1")
        #expect(abs(breakdown.p0 + breakdown.p1 - 1.0) < 1e-10, "Breakdown probabilities should sum to 1.0")
        let blochMagnitude = breakdown.blochVector.x * breakdown.blochVector.x + breakdown.blochVector.y * breakdown.blochVector.y + breakdown.blochVector.z * breakdown.blochVector.z
        #expect(blochMagnitude <= 1.0 + 1e-10, "Bloch vector magnitude should be at most 1.0 for valid state")
    }
}

/// Test suite for DebugExecution initialization.
/// Validates debugger creation with default and custom initial states,
/// trace initialization, and circuit binding.
@Suite("DebugExecution Initialization")
struct DebugExecutionInitializationTests {
    @Test("Initialize with default ground state")
    func initializeDefaultState() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let debug = DebugExecution(circuit: circuit)

        #expect(debug.circuit.qubits == 2, "Debug should bind to circuit with 2 qubits")
        #expect(debug.currentIndex == 0, "Initial currentIndex should be 0")
        #expect(debug.trace.count == 1, "Initial trace should contain one snapshot")
        let prob0 = debug.currentState.probability(of: 0)
        #expect(abs(prob0 - 1.0) < 1e-10, "Default initial state should be |00⟩")
    }

    @Test("Initialize with custom initial state")
    func initializeCustomState() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliZ, to: 0)
        let customState = QuantumState(qubit: 1)
        let debug = DebugExecution(circuit: circuit, initialState: customState)

        let prob1 = debug.currentState.probability(of: 1)
        #expect(abs(prob1 - 1.0) < 1e-10, "Custom initial state |1⟩ should persist")
    }

    @Test("Custom initial state uses provided QuantumState")
    func customInitialStateUsesProvidedQuantumState() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])
        var prepCircuit = QuantumCircuit(qubits: 2)
        prepCircuit.append(.hadamard, to: 0)
        let prepState = prepCircuit.execute()
        let debug = DebugExecution(circuit: circuit, initialState: prepState)

        let prob0 = debug.currentState.probability(of: 0)
        #expect(abs(prob0 - 0.5) < 1e-10, "Custom state should start in superposition with p(|00⟩)=0.5")
    }

    @Test("Nil initial state defaults to ground state")
    func nilInitialStateDefaultsToGround() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let debug = DebugExecution(circuit: circuit, initialState: nil)

        let prob00 = debug.currentState.probability(of: 0)
        #expect(abs(prob00 - 1.0) < 1e-10, "Nil initialState should default to |00⟩ ground state")
    }

    @Test("Empty circuit creates valid debugger")
    func initializeEmptyCircuit() {
        let circuit = QuantumCircuit(qubits: 1)
        let debug = DebugExecution(circuit: circuit)

        #expect(debug.isComplete, "Empty circuit should be immediately complete")
        #expect(debug.trace.count == 1, "Empty circuit should have initial snapshot only")
    }
}

/// Test suite for DebugExecution step operations.
/// Validates single step, multi-step, and stepping past completion
/// with correct state progression and trace recording.
@Suite("DebugExecution Step Operations")
struct DebugExecutionStepTests {
    @Test("Single step advances index and records snapshot")
    func singleStep() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 0)
        var debug = DebugExecution(circuit: circuit)

        _ = debug.step()
        #expect(debug.currentIndex == 1, "currentIndex should advance to 1 after step")
        #expect(debug.trace.count == 2, "Trace should have 2 snapshots after step")
    }

    @Test("Multi-step advances by specified count")
    func multiStep() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliZ, to: 0)
        var debug = DebugExecution(circuit: circuit)

        _ = debug.step(count: 2)
        #expect(debug.currentIndex == 2, "currentIndex should be 2 after step(count: 2)")
        #expect(debug.trace.count == 3, "Trace should have 3 snapshots after 2 steps")
    }

    @Test("Step when complete returns last snapshot")
    func stepWhenComplete() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        var debug = DebugExecution(circuit: circuit)

        let first = debug.step()
        let second = debug.step()
        #expect(debug.isComplete, "Debugger should be complete after all steps")
        #expect(first == second, "Repeated step on complete should return same snapshot")
        #expect(debug.trace.count == 2, "Trace should not grow after completion")
    }

    @Test("Multi-step stops at circuit end")
    func multiStepStopsAtEnd() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        var debug = DebugExecution(circuit: circuit)

        _ = debug.step(count: 10)
        #expect(debug.currentIndex == 1, "currentIndex should stop at circuit count")
        #expect(debug.isComplete, "Debugger should be complete")
    }

    @Test("Step returns snapshot with timing information")
    func stepRecordsTiming() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        var debug = DebugExecution(circuit: circuit)

        let snapshot = debug.step()
        #expect(snapshot.elapsedNs >= 0, "Elapsed time should be non-negative")
    }

    @Test("Step returns valid DebugSnapshot with correct properties")
    func stepReturnsValidDebugSnapshot() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        var debug = DebugExecution(circuit: circuit)

        let snapshot = debug.step()
        #expect(snapshot.index == 0, "Returned snapshot should have index 0")
        #expect(snapshot.operation?.gate == .hadamard, "Returned snapshot should contain Hadamard gate")
        let prob = snapshot.state.probability(of: 0) + snapshot.state.probability(of: 1)
        #expect(abs(prob - 1.0) < 1e-10, "Returned snapshot state should have valid probabilities summing to ~1")
    }

    @Test("Step count advances correct number of operations")
    func stepCountAdvancesCorrectNumber() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.cnot, to: [0, 1])
        var debug = DebugExecution(circuit: circuit)

        let snapshot = debug.step(count: 2)
        #expect(snapshot.index == 1, "step(count:2) should return snapshot at index 1")
        #expect(debug.currentIndex == 2, "currentIndex should be 2 after step(count:2)")
        #expect(snapshot.operation?.gate == .pauliX, "Returned snapshot should be from second operation")
    }

    @Test("State progression matches direct execution")
    func stateProgressionCorrectness() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        var debug = DebugExecution(circuit: circuit)

        _ = debug.step(count: 2)
        let prob00 = debug.currentState.probability(of: 0)
        let prob11 = debug.currentState.probability(of: 3)
        #expect(abs(prob00 - 0.5) < 1e-10, "Bell state |00⟩ probability should be 0.5")
        #expect(abs(prob11 - 0.5) < 1e-10, "Bell state |11⟩ probability should be 0.5")
    }
}

/// Test suite for DebugExecution reset functionality.
/// Validates state restoration, trace clearing, and index reset
/// for time-travel debugging support.
@Suite("DebugExecution Reset")
struct DebugExecutionResetTests {
    @Test("Reset restores initial state")
    func resetRestoresState() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        var debug = DebugExecution(circuit: circuit)

        _ = debug.step()
        let probBefore = debug.currentState.probability(of: 1)
        #expect(abs(probBefore - 1.0) < 1e-10, "After X, prob(|1⟩) should be 1.0")

        debug.reset()
        let probAfter = debug.currentState.probability(of: 0)
        #expect(abs(probAfter - 1.0) < 1e-10, "After reset, prob(|0⟩) should be 1.0")
    }

    @Test("Reset clears trace except initial snapshot")
    func resetClearsTrace() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 0)
        var debug = DebugExecution(circuit: circuit)

        _ = debug.step(count: 2)
        #expect(debug.trace.count == 3, "Trace should have 3 snapshots before reset")

        debug.reset()
        #expect(debug.trace.count == 1, "Trace should have 1 snapshot after reset")
        #expect(debug.trace[0].index == -1, "Only initial snapshot should remain")
    }

    @Test("Reset sets currentIndex to zero")
    func resetSetsIndexToZero() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        var debug = DebugExecution(circuit: circuit)

        _ = debug.step()
        #expect(debug.currentIndex == 1, "Index should be 1 after step")

        debug.reset()
        #expect(debug.currentIndex == 0, "Index should be 0 after reset")
        #expect(!debug.isComplete, "Debugger should not be complete after reset")
    }

    @Test("Can step again after reset")
    func stepAfterReset() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        var debug = DebugExecution(circuit: circuit)

        let first = debug.step()
        debug.reset()
        let second = debug.step()

        #expect(first.index == second.index, "Re-stepped snapshot should have same index")
        #expect(first.operation == second.operation, "Re-stepped snapshot should have same operation")
    }

    @Test("Reset clears trace to initial snapshot only")
    func resetClearsTraceToInitial() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        var debug = DebugExecution(circuit: circuit)

        _ = debug.step(count: 2)
        let initialSnapshot = debug.trace[0]
        debug.reset()
        #expect(debug.trace.count == 1, "Reset should leave only initial snapshot in trace")
        #expect(debug.trace[0] == initialSnapshot, "Reset trace should contain original initial snapshot")
        #expect(debug.trace[0].index == -1, "Reset trace first element should have index -1")
    }
}

/// Test suite for DebugExecution isComplete property.
/// Validates completion detection for circuits of various sizes
/// including empty circuits and multi-operation circuits.
@Suite("DebugExecution Completion")
struct DebugExecutionCompletionTests {
    @Test("isComplete false before all steps")
    func notCompleteBeforeAllSteps() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 0)
        var debug = DebugExecution(circuit: circuit)

        #expect(!debug.isComplete, "Should not be complete at start")
        _ = debug.step()
        #expect(!debug.isComplete, "Should not be complete after first step")
    }

    @Test("isComplete true after all steps")
    func completeAfterAllSteps() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        var debug = DebugExecution(circuit: circuit)

        _ = debug.step()
        #expect(debug.isComplete, "Should be complete after stepping through all operations")
    }

    @Test("Empty circuit is immediately complete")
    func emptyCircuitComplete() {
        let circuit = QuantumCircuit(qubits: 1)
        let debug = DebugExecution(circuit: circuit)

        #expect(debug.isComplete, "Empty circuit should be immediately complete")
    }

    @Test("isComplete true after executing all operations")
    func isCompleteTrueAfterAllOperations() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        var debug = DebugExecution(circuit: circuit)

        _ = debug.step(count: 2)
        #expect(debug.isComplete, "isComplete should be true after all operations executed")
        #expect(debug.currentIndex == circuit.operations.count, "currentIndex should equal operation count when complete")
    }

    @Test("currentState matches expected state after operations")
    func currentStateMatchesExpected() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 1)
        var debug = DebugExecution(circuit: circuit)

        _ = debug.step(count: 2)
        let prob11 = debug.currentState.probability(of: 3)
        #expect(abs(prob11 - 1.0) < 1e-10, "currentState should be |11⟩ after X on both qubits")
        let prob00 = debug.currentState.probability(of: 0)
        #expect(abs(prob00) < 1e-10, "currentState probability of |00⟩ should be 0")
    }
}

/// Test suite for DebugExecution trace recording.
/// Validates that execution trace captures all steps with correct
/// indices, operations, and state snapshots.
@Suite("DebugExecution Trace")
struct DebugExecutionTraceTests {
    @Test("Trace records all steps in order")
    func traceRecordsAllSteps() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliZ, to: 0)
        var debug = DebugExecution(circuit: circuit)

        _ = debug.step(count: 3)
        #expect(debug.trace.count == 4, "Trace should have initial + 3 step snapshots")
        #expect(debug.trace[0].index == -1, "First trace entry should be initial snapshot")
        #expect(debug.trace[1].index == 0, "Second trace entry should have index 0")
        #expect(debug.trace[2].index == 1, "Third trace entry should have index 1")
        #expect(debug.trace[3].index == 2, "Fourth trace entry should have index 2")
    }

    @Test("Trace snapshots have correct operations")
    func traceSnapshotsHaveCorrectOperations() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 0)
        var debug = DebugExecution(circuit: circuit)

        _ = debug.step(count: 2)
        #expect(debug.trace[1].operation?.gate == .hadamard, "First step should record Hadamard")
        #expect(debug.trace[2].operation?.gate == .pauliX, "Second step should record Pauli-X")
    }

    @Test("Trace states show progression")
    func traceStatesShowProgression() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        var debug = DebugExecution(circuit: circuit)

        _ = debug.step()
        let initialProb0 = debug.trace[0].state.probability(of: 0)
        let afterXProb1 = debug.trace[1].state.probability(of: 1)
        #expect(abs(initialProb0 - 1.0) < 1e-10, "Initial state should be |0⟩")
        #expect(abs(afterXProb1 - 1.0) < 1e-10, "After X gate, state should be |1⟩")
    }
}
