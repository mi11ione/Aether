// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Test suite for ExtendedStabilizerSimulator actor.
/// Validates async circuit execution with T gates, amplitude computation,
/// and sampling from extended stabilizer state decompositions.
@Suite("Extended Stabilizer Simulator")
struct ExtendedStabilizerSimulatorTests {
    let tolerance: Double = 1e-10

    @Test("execute processes circuit with single T gate")
    func executeProcessesSingleTGate() async {
        let simulator = ExtendedStabilizerSimulator(maxRank: 64)

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.tGate, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let state = await simulator.execute(circuit)

        #expect(
            state.qubits == 2,
            "State should have 2 qubits after executing circuit",
        )
        #expect(
            state.rank == 2,
            "Single T gate should double rank from 1 to 2",
        )
    }

    @Test("execute processes circuit with multiple T gates")
    func executeProcessesMultipleTGates() async {
        let simulator = ExtendedStabilizerSimulator(maxRank: 64)

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.tGate, to: 0)
        circuit.append(.tGate, to: 1)

        let state = await simulator.execute(circuit)

        #expect(
            state.rank == 4,
            "Two T gates should result in rank 4 (2^2)",
        )
    }

    @Test("execute handles Clifford-only circuit without rank increase")
    func executeCliffordOnlyCircuit() async {
        let simulator = ExtendedStabilizerSimulator(maxRank: 64)

        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.sGate, to: 2)
        circuit.append(.pauliZ, to: 0)

        let state = await simulator.execute(circuit)

        #expect(
            state.rank == 1,
            "Clifford-only circuit should maintain rank of 1",
        )
    }

    @Test("execute from custom initial state")
    func executeFromCustomInitialState() async {
        let simulator = ExtendedStabilizerSimulator(maxRank: 64)
        let initial = ExtendedStabilizerState(qubits: 2, maxRank: 64)

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.tGate, to: 0)

        let state = await simulator.execute(circuit, from: initial)

        #expect(
            state.qubits == 2,
            "State should preserve qubit count from initial state",
        )
        #expect(
            state.rank == 2,
            "T gate should double rank from initial state",
        )
    }

    @Test("amplitude returns correct value for ground state")
    func amplitudeReturnsGroundStateValue() async {
        let simulator = ExtendedStabilizerSimulator(maxRank: 64)

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.identity, to: 0)

        let amp = await simulator.amplitude(circuit, of: 0b00)

        #expect(
            abs(amp.real - 1.0) < tolerance,
            "Ground state amplitude real part should be 1.0 for identity circuit",
        )
        #expect(
            abs(amp.imaginary) < tolerance,
            "Ground state amplitude imaginary part should be 0.0 for identity circuit",
        )
    }

    @Test("amplitude returns correct value after Hadamard")
    func amplitudeAfterHadamard() async {
        let simulator = ExtendedStabilizerSimulator(maxRank: 64)

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        let amp0 = await simulator.amplitude(circuit, of: 0)
        let amp1 = await simulator.amplitude(circuit, of: 1)

        let expected = 1.0 / 2.0.squareRoot()

        #expect(
            abs(amp0.real - expected) < tolerance,
            "Hadamard should produce equal superposition amplitude for |0>",
        )
        #expect(
            abs(amp1.real - expected) < tolerance,
            "Hadamard should produce equal superposition amplitude for |1>",
        )
    }

    @Test("amplitude returns correct value after T gate")
    func amplitudeAfterTGate() async {
        let simulator = ExtendedStabilizerSimulator(maxRank: 64)

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.tGate, to: 0)

        let amp0 = await simulator.amplitude(circuit, of: 0)
        let amp1 = await simulator.amplitude(circuit, of: 1)

        let prob0 = amp0.magnitudeSquared
        let prob1 = amp1.magnitudeSquared

        #expect(
            abs(prob0 - 0.5) < tolerance,
            "Probability of |0> should be 0.5 after H and T gates",
        )
        #expect(
            abs(prob1 - 0.5) < tolerance,
            "Probability of |1> should be 0.5 after H and T gates",
        )
        #expect(
            abs(prob0 + prob1 - 1.0) < tolerance,
            "Total probability should sum to 1.0",
        )
    }

    @Test("sample returns valid outcomes within basis state range")
    func sampleReturnsValidOutcomes() async {
        let simulator = ExtendedStabilizerSimulator(maxRank: 64)

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.tGate, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let samples = await simulator.sample(circuit, shots: 100, seed: 42)

        #expect(
            samples.count == 100,
            "Sample should return exactly 100 outcomes",
        )

        for outcome in samples {
            #expect(
                outcome >= 0 && outcome < 4,
                "Each sample outcome should be in range [0, 3] for 2 qubits",
            )
        }
    }

    @Test("sample produces reproducible results with seed")
    func sampleReproducibleWithSeed() async {
        let simulator = ExtendedStabilizerSimulator(maxRank: 64)

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.tGate, to: 0)

        let samples1 = await simulator.sample(circuit, shots: 50, seed: 12345)
        let samples2 = await simulator.sample(circuit, shots: 50, seed: 12345)

        #expect(
            samples1 == samples2,
            "Same seed should produce identical sample sequences",
        )
    }

    @Test("sample distribution approximates expected probabilities")
    func sampleDistributionMatchesProbabilities() async {
        let simulator = ExtendedStabilizerSimulator(maxRank: 64)

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.tGate, to: 0)

        let shots = 1000
        let samples = await simulator.sample(circuit, shots: shots, seed: 98765)

        let count0 = samples.count(where: { $0 == 0 })
        let count1 = samples.count(where: { $0 == 1 })

        let freq0 = Double(count0) / Double(shots)
        let freq1 = Double(count1) / Double(shots)

        #expect(
            abs(freq0 - 0.5) < 0.1,
            "Frequency of |0> should be approximately 0.5 within statistical tolerance",
        )
        #expect(
            abs(freq1 - 0.5) < 0.1,
            "Frequency of |1> should be approximately 0.5 within statistical tolerance",
        )
    }

    @Test("simulator respects maxRank parameter")
    func simulatorRespectsMaxRank() async {
        let simulator = ExtendedStabilizerSimulator(maxRank: 16)

        #expect(
            simulator.maxRank == 16,
            "Simulator maxRank should match initialization parameter",
        )
    }

    @Test("execute handles Bell state with T gate")
    func executeBellStateWithTGate() async {
        let simulator = ExtendedStabilizerSimulator(maxRank: 64)

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.tGate, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let amp00 = await simulator.amplitude(circuit, of: 0b00)
        let amp11 = await simulator.amplitude(circuit, of: 0b11)

        let prob00 = amp00.magnitudeSquared
        let prob11 = amp11.magnitudeSquared

        #expect(
            abs(prob00 + prob11 - 1.0) < tolerance,
            "Bell-like state with T gate should have probabilities summing to 1",
        )
        #expect(
            prob00 > 0.1,
            "Probability of |00> should be non-negligible",
        )
        #expect(
            prob11 > 0.1,
            "Probability of |11> should be non-negligible",
        )
    }

    @Test("execute empty circuit returns ground state")
    func executeEmptyCircuit() async {
        let simulator = ExtendedStabilizerSimulator(maxRank: 64)
        let circuit = QuantumCircuit(qubits: 2)

        let state = await simulator.execute(circuit)

        #expect(
            state.rank == 1,
            "Empty circuit should maintain rank 1",
        )

        let prob00 = state.probability(of: 0b00)
        #expect(
            abs(prob00 - 1.0) < tolerance,
            "Empty circuit should leave system in ground state |00>",
        )
    }

    @Test("execute skips reset and measure operations")
    func executeSkipsResetAndMeasure() async {
        let simulator = ExtendedStabilizerSimulator(maxRank: 64)

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.reset, to: 1)
        circuit.append(.measure, to: 0)

        let state = await simulator.execute(circuit)

        #expect(
            state.rank == 1,
            "Circuit with reset/measure should still execute Clifford gates",
        )
    }

    @Test("execute applies Toffoli gate decomposition")
    func executeAppliesToffoliDecomposition() async {
        let simulator = ExtendedStabilizerSimulator(maxRank: 1024)

        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.toffoli, to: [0, 1, 2])

        let state = await simulator.execute(circuit)

        #expect(
            state.rank > 1,
            "Toffoli decomposition should increase stabilizer rank",
        )
    }

    @Test("execute applies CCZ gate decomposition")
    func executeAppliesCCZDecomposition() async {
        let simulator = ExtendedStabilizerSimulator(maxRank: 1024)

        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.pauliX, to: 2)
        circuit.append(.ccz, to: [0, 1, 2])

        let state = await simulator.execute(circuit)

        #expect(
            state.rank > 1,
            "CCZ decomposition should increase stabilizer rank",
        )
    }

    @Test("sample without seed produces valid outcomes")
    func sampleWithoutSeedProducesValidOutcomes() async {
        let simulator = ExtendedStabilizerSimulator(maxRank: 64)

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)

        let samples = await simulator.sample(circuit, shots: 50, seed: nil)

        #expect(
            samples.count == 50,
            "Sample without seed should return requested number of outcomes",
        )

        let allValid = samples.allSatisfy { $0 >= 0 && $0 < 4 }
        #expect(
            allValid,
            "All samples should be valid basis states for 2 qubits",
        )
    }
}
