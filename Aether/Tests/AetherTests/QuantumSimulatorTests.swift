// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Testing

/// Test suite for QuantumSimulator actor.
/// Validates Swift concurrency features: async execution
/// progress reporting and thread-safe quantum circuit simulation.
@Suite("Quantum Simulator")
struct QuantumSimulatorTests {
    @Test("Simulator executes simple circuit asynchronously")
    func simulatorExecutesSimpleCircuit() async {
        let simulator = QuantumSimulator()

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let finalState = await simulator.execute(circuit)

        #expect(finalState.qubits == 2, "Bell circuit should produce 2-qubit state")
        #expect(finalState.isNormalized(), "Final state should be normalized")

        let p0 = finalState.probability(of: 0)
        let p3 = finalState.probability(of: 3)
        #expect(abs(p0 - 0.5) < 1e-10, "P(|00>) should be 0.5 in Bell state")
        #expect(abs(p3 - 0.5) < 1e-10, "P(|11>) should be 0.5 in Bell state")
    }

    @Test("Simulator executes from custom initial state")
    func simulatorExecutesFromCustomState() async {
        let simulator = QuantumSimulator()

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)

        let initialState = QuantumState(qubits: 1)
        let finalState = await simulator.execute(circuit, from: initialState)

        #expect(abs(finalState.probability(of: 1) - 1.0) < 1e-10, "X gate on |0> should give |1>")
    }

    @Test("Simulator reports progress during execution")
    func simulatorReportsProgress() async {
        let simulator = QuantumSimulator()

        var circuit = QuantumCircuit(qubits: 2)
        for _ in 0 ..< 20 {
            circuit.append(.hadamard, to: 0)
        }

        actor ProgressAccumulator {
            private(set) var values: [Double] = []
            func append(_ value: Double) {
                values.append(value)
            }
        }

        let accumulator = ProgressAccumulator()

        _ = await simulator.execute(circuit, progressHandler: { progress in
            await accumulator.append(progress)
        })

        let progressUpdates = await accumulator.values

        #expect(progressUpdates.count > 0, "Should receive at least one progress update")

        for i in 1 ..< progressUpdates.count {
            #expect(progressUpdates[i] >= progressUpdates[i - 1], "Progress updates should be monotonically increasing")
        }

        if let lastProgress = progressUpdates.last {
            #expect(lastProgress > 0.9, "Final progress should be near completion")
        }
    }

    @Test("Simulator handles empty circuit")
    func simulatorHandlesEmptyCircuit() async {
        let simulator = QuantumSimulator()
        let circuit = QuantumCircuit(qubits: 2)

        let finalState = await simulator.execute(circuit)

        #expect(abs(finalState.probability(of: 0) - 1.0) < 1e-10, "Empty circuit should leave ground state unchanged")
    }

    @Test("Simulator handles pre-built Bell state circuit")
    func simulatorHandlesBellStateCircuit() async {
        let simulator = QuantumSimulator()
        let circuit = QuantumCircuit.bell()
        let finalState = await simulator.execute(circuit)

        #expect(finalState.isNormalized(), "Bell state should be normalized")
        #expect(abs(finalState.probability(of: 0) - 0.5) < 1e-10, "P(|00>) should be 0.5 in Bell state")
        #expect(abs(finalState.probability(of: 3) - 0.5) < 1e-10, "P(|11>) should be 0.5 in Bell state")
    }

    @Test("Simulator handles QFT circuit")
    func simulatorHandlesQFTCircuit() async {
        let simulator = QuantumSimulator()
        let circuit = QuantumCircuit.qft(qubits: 3)
        let finalState = await simulator.execute(circuit)

        #expect(finalState.isNormalized(), "QFT state should be normalized")
        #expect(finalState.qubits == 3, "QFT circuit should produce 3-qubit state")
    }

    @Test("Simulator handles Grover circuit")
    func simulatorHandlesGroverCircuit() async {
        let simulator = QuantumSimulator()
        let circuit = QuantumCircuit.grover(qubits: 2, target: 3)
        let finalState = await simulator.execute(circuit)

        #expect(finalState.isNormalized(), "Grover state should be normalized")

        let targetProb = finalState.probability(of: 3)
        #expect(targetProb > 0.8, "Grover should amplify target state probability above 0.8")
    }

    @Test("Simulator with Metal disabled works correctly")
    func simulatorWithoutMetalWorks() async {
        let simulator = QuantumSimulator(precisionPolicy: .accurate)

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let finalState = await simulator.execute(circuit)

        #expect(finalState.qubits == 2, "Accurate mode should produce 2-qubit state")
        #expect(finalState.isNormalized(), "State should be normalized in accurate mode")
    }

    @Test("Simulator uses Metal acceleration for large circuits")
    func simulatorUsesMetalAcceleration() async {
        guard MetalGateApplication() != nil else { return }

        let simulator = QuantumSimulator(precisionPolicy: .fast)

        var circuit = QuantumCircuit(qubits: 12)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.pauliZ, to: 2)

        let finalState = await simulator.execute(circuit)

        #expect(finalState.qubits == 12, "Metal path should produce 12-qubit state")
        #expect(finalState.isNormalized(), "State should be normalized after GPU execution")
    }

    @Test("Simulator uses Metal acceleration with progress reporting")
    func simulatorUsesMetalWithProgress() async {
        guard MetalGateApplication() != nil else { return }

        let simulator = QuantumSimulator(precisionPolicy: .fast)

        var circuit = QuantumCircuit(qubits: 12)
        for _ in 0 ..< 8 {
            circuit.append(.hadamard, to: 0)
        }

        actor ProgressTracker {
            private(set) var progressUpdates: [Double] = []
            func append(_ value: Double) {
                progressUpdates.append(value)
            }
        }

        let tracker = ProgressTracker()

        let finalState = await simulator.execute(circuit, progressHandler: { progress in
            await tracker.append(progress)
        })

        #expect(finalState.qubits == 12, "Metal+progress should produce 12-qubit state")
        #expect(finalState.isNormalized(), "State should be normalized after GPU execution with progress")

        let updates = await tracker.progressUpdates
        #expect(updates.count > 0, "Should receive progress updates during Metal execution")

        if let lastProgress = updates.last {
            #expect(lastProgress >= 0.99, "Final progress should indicate completion")
        }
    }

    @Test("Simulator executes with progress from custom initial state")
    func simulatorExecutesWithProgressFromCustomState() async {
        let simulator = QuantumSimulator()

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        var initialState = QuantumState(qubits: 2)
        initialState.setAmplitude(1, to: .one)
        initialState.setAmplitude(0, to: .zero)

        actor ProgressTracker {
            private(set) var progressCalled = false
            func markCalled() {
                progressCalled = true
            }
        }

        let tracker = ProgressTracker()

        let finalState = await simulator.execute(circuit, from: initialState, progressHandler: { _ in
            await tracker.markCalled()
        })

        #expect(finalState.qubits == 2, "Custom state execution should produce 2-qubit state")
        #expect(finalState.isNormalized(), "State should be normalized after custom state execution")
        #expect(await tracker.progressCalled, "Progress callback should be invoked")
    }

    @Test("Progress percentage computes correctly for non-zero total")
    func progressPercentageNonZeroTotal() {
        let progress = QuantumSimulator.Progress(executed: 5, total: 10)

        #expect(
            abs(progress.fractionCompleted - 0.5) < 1e-10,
            "Progress should be 50% when 5 of 10 gates executed",
        )
    }

    @Test("Progress percentage returns zero for zero total")
    func progressPercentageZeroTotal() {
        let progress = QuantumSimulator.Progress(executed: 0, total: 0)

        #expect(
            progress.fractionCompleted == 0.0,
            "Progress should be 0.0 when total is zero to avoid division by zero",
        )
    }

    @Test("Progress percentage computes full completion")
    func progressPercentageFullCompletion() {
        let progress = QuantumSimulator.Progress(executed: 20, total: 20)

        #expect(
            abs(progress.fractionCompleted - 1.0) < 1e-10,
            "Progress should be 100% when all gates executed",
        )
    }
}

/// Test suite for PrecisionPolicy enum.
/// Validates tolerance values, description formatting,
/// GPU threshold logic, and backend selection.
@Suite("Precision Policy")
struct PrecisionPolicyTests {
    @Test("Tolerance returns correct values for each policy")
    func toleranceValues() {
        #expect(
            abs(PrecisionPolicy.fast.tolerance - 1e-5) < 1e-15,
            "Fast policy should have 1e-5 tolerance",
        )
        #expect(
            abs(PrecisionPolicy.balanced.tolerance - 1e-7) < 1e-15,
            "Balanced policy should have 1e-7 tolerance",
        )
        #expect(
            abs(PrecisionPolicy.accurate.tolerance - 1e-10) < 1e-15,
            "Accurate policy should have 1e-10 tolerance",
        )
    }

    @Test("Description formats correctly for accurate policy")
    func descriptionAccurate() {
        let description = PrecisionPolicy.accurate.description

        #expect(description.contains("Accurate"), "Should contain policy name")
        #expect(description.contains("CPU-only"), "Should indicate CPU-only execution")
        #expect(description.contains("1e-10"), "Should show tolerance value")
    }

    @Test("Description formats correctly for all policies")
    func descriptionAllPolicies() {
        let fastDesc = PrecisionPolicy.fast.description
        let balancedDesc = PrecisionPolicy.balanced.description
        let accurateDesc = PrecisionPolicy.accurate.description

        #expect(fastDesc.contains("Fast"), "Fast description should contain policy name")
        #expect(fastDesc.contains("10"), "Fast description should show GPU threshold")

        #expect(balancedDesc.contains("Balanced"), "Balanced description should contain policy name")
        #expect(balancedDesc.contains("12"), "Balanced description should show GPU threshold")

        #expect(accurateDesc.contains("Accurate"), "Accurate description should contain policy name")
    }

    @Test("GPU qubit threshold values are correct")
    func gpuQubitThreshold() {
        #expect(PrecisionPolicy.fast.gpuQubitThreshold == 10, "Fast should use 10 qubits")
        #expect(PrecisionPolicy.balanced.gpuQubitThreshold == 12, "Balanced should use 12 qubits")
        #expect(PrecisionPolicy.accurate.gpuQubitThreshold == Int.max, "Accurate should disable GPU")
    }

    @Test("isGPUEnabled returns correct values")
    func isGPUEnabled() {
        #expect(PrecisionPolicy.fast.isGPUEnabled, "Fast should enable GPU")
        #expect(PrecisionPolicy.balanced.isGPUEnabled, "Balanced should enable GPU")
        #expect(!PrecisionPolicy.accurate.isGPUEnabled, "Accurate should disable GPU")
    }

    @Test("shouldUseGPU respects policy and qubit count")
    func shouldUseGPU() {
        #expect(!PrecisionPolicy.fast.shouldUseGPU(forQubitCount: 9), "9 qubits below fast threshold")
        #expect(PrecisionPolicy.fast.shouldUseGPU(forQubitCount: 10), "10 qubits meets fast threshold")
        #expect(PrecisionPolicy.fast.shouldUseGPU(forQubitCount: 15), "15 qubits above fast threshold")

        #expect(!PrecisionPolicy.balanced.shouldUseGPU(forQubitCount: 11), "11 qubits below balanced threshold")
        #expect(PrecisionPolicy.balanced.shouldUseGPU(forQubitCount: 12), "12 qubits meets balanced threshold")

        #expect(!PrecisionPolicy.accurate.shouldUseGPU(forQubitCount: 20), "Accurate never uses GPU")
        #expect(!PrecisionPolicy.accurate.shouldUseGPU(forQubitCount: 100), "Accurate never uses GPU regardless of size")
    }
}
