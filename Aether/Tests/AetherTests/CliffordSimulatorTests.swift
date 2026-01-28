// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for CliffordSimulator actor.
/// Validates Clifford circuit execution, tableau-based simulation
/// and measurement sampling with correct probability distributions.
@Suite("Clifford Simulator")
struct CliffordSimulatorTests {
    @Test("execute applies Hadamard gate correctly")
    func executeAppliesHadamard() async {
        let simulator = CliffordSimulator()

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)

        let tableau = await simulator.execute(circuit)
        let (p0, p1) = tableau.probability(of: 0, measuring: .z)

        #expect(
            abs(p0 - 0.5) < 1e-10,
            "Hadamard on |0> should give 50% probability of measuring 0",
        )
        #expect(
            abs(p1 - 0.5) < 1e-10,
            "Hadamard on |0> should give 50% probability of measuring 1",
        )
    }

    @Test("execute applies CNOT gate correctly creating Bell state")
    func executeAppliesCNOT() async {
        let simulator = CliffordSimulator()

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let tableau = await simulator.execute(circuit)

        let zz = PauliString(.z(0), .z(1))
        let xx = PauliString(.x(0), .x(1))
        let zzExpectation = tableau.expectationValue(of: zz)
        let xxExpectation = tableau.expectationValue(of: xx)

        #expect(
            abs(zzExpectation - 1.0) < 1e-10,
            "Bell state should have ZZ expectation value of +1",
        )
        #expect(
            abs(xxExpectation - 1.0) < 1e-10,
            "Bell state should have XX expectation value of +1",
        )
    }

    @Test("execute applies GHZ circuit correctly")
    func executeAppliesGHZCircuit() async {
        let simulator = CliffordSimulator()

        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.cnot, to: [1, 2])

        let tableau = await simulator.execute(circuit)

        let xxx = PauliString(.x(0), .x(1), .x(2))
        let expectation = tableau.expectationValue(of: xxx)

        #expect(
            abs(expectation - 1.0) < 1e-10,
            "GHZ state should have XXX expectation value of +1",
        )
    }

    @Test("execute from custom initial tableau")
    func executeFromCustomTableau() async {
        let simulator = CliffordSimulator()

        var initial = StabilizerTableau(qubits: 2)
        initial.apply(.hadamard, to: 0)

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])

        let final = await simulator.execute(circuit, from: initial)

        let zz = PauliString(.z(0), .z(1))
        let expectation = final.expectationValue(of: zz)

        #expect(
            abs(expectation - 1.0) < 1e-10,
            "Executing CNOT from H|0> state should create Bell state with ZZ = +1",
        )
    }

    @Test("execute from initial preserves prior state transformations")
    func executeFromInitialPreservesState() async {
        let simulator = CliffordSimulator()

        var initial = StabilizerTableau(qubits: 3)
        initial.apply(.pauliX, to: 0)

        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.cnot, to: [0, 1])

        let final = await simulator.execute(circuit, from: initial)

        let z0z1 = PauliString(.z(0), .z(1))
        let z0z1Expectation = final.expectationValue(of: z0z1)

        #expect(
            abs(z0z1Expectation - 1.0) < 1e-10,
            "After X(0) then CNOT(0,1), state |11> should have Z0Z1 expectation +1",
        )

        let (p0_q0, p1_q0) = final.probability(of: 0, measuring: .z)
        #expect(
            abs(p1_q0 - 1.0) < 1e-10,
            "Qubit 0 should remain in |1> state after CNOT",
        )
        #expect(
            abs(p0_q0) < 1e-10,
            "Qubit 0 should have zero probability of measuring 0",
        )
    }

    @Test("sample returns valid measurement outcomes")
    func sampleReturnsValidOutcomes() async {
        let simulator = CliffordSimulator()

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let samples = await simulator.sample(circuit, shots: 500, seed: 12345)

        #expect(
            samples.count == 500,
            "Sample should return exactly 500 measurement outcomes",
        )

        let validOutcomes = samples.allSatisfy { $0 == 0 || $0 == 3 }
        #expect(
            validOutcomes,
            "Bell state samples should only be 0 (|00>) or 3 (|11>)",
        )
    }

    @Test("sample with seed produces reproducible results")
    func sampleWithSeedReproducible() async {
        let simulator = CliffordSimulator()

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)

        let samples1 = await simulator.sample(circuit, shots: 100, seed: 42)
        let samples2 = await simulator.sample(circuit, shots: 100, seed: 42)

        #expect(
            samples1 == samples2,
            "Same seed should produce identical sample sequences",
        )
    }

    @Test("sample distribution matches expected probabilities for Bell state")
    func sampleDistributionBellState() async {
        let simulator = CliffordSimulator()

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let shots = 1000
        let samples = await simulator.sample(circuit, shots: shots, seed: 54321)

        let count00 = samples.count(where: { $0 == 0 })
        let count11 = samples.count(where: { $0 == 3 })

        let expected = Double(shots) / 2.0
        let observed00 = Double(count00)
        let observed11 = Double(count11)

        let chiSquared = pow(observed00 - expected, 2) / expected + pow(observed11 - expected, 2) / expected

        #expect(
            chiSquared < 10.83,
            "Chi-squared test failed: distribution deviates significantly from 50/50 (chi^2 = \(chiSquared), threshold = 10.83 for p=0.001)",
        )
    }

    @Test("sample distribution matches expected probabilities for deterministic state")
    func sampleDistributionDeterministic() async {
        let simulator = CliffordSimulator()

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 1)

        let samples = await simulator.sample(circuit, shots: 100, seed: 999)

        let allThrees = samples.allSatisfy { $0 == 3 }

        #expect(
            allThrees,
            "X on both qubits should produce deterministic |11> = 3 outcome",
        )
    }

    @Test("sample distribution for single qubit superposition")
    func sampleDistributionSingleQubit() async {
        let simulator = CliffordSimulator()

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        let shots = 1000
        let samples = await simulator.sample(circuit, shots: shots, seed: 77777)

        let count0 = samples.count(where: { $0 == 0 })
        let count1 = samples.count(where: { $0 == 1 })

        let freq0 = Double(count0) / Double(shots)
        let freq1 = Double(count1) / Double(shots)

        #expect(
            abs(freq0 - 0.5) < 0.1,
            "Hadamard state should have ~50% probability of 0 (observed: \(freq0 * 100)%)",
        )
        #expect(
            abs(freq1 - 0.5) < 0.1,
            "Hadamard state should have ~50% probability of 1 (observed: \(freq1 * 100)%)",
        )
    }

    @Test("execute handles empty circuit")
    func executeHandlesEmptyCircuit() async {
        let simulator = CliffordSimulator()
        let circuit = QuantumCircuit(qubits: 2)

        let tableau = await simulator.execute(circuit)

        let (p0_q0, _) = tableau.probability(of: 0, measuring: .z)
        let (p0_q1, _) = tableau.probability(of: 1, measuring: .z)

        #expect(
            abs(p0_q0 - 1.0) < 1e-10,
            "Empty circuit should leave qubit 0 in |0> state",
        )
        #expect(
            abs(p0_q1 - 1.0) < 1e-10,
            "Empty circuit should leave qubit 1 in |0> state",
        )
    }

    @Test("execute applies S gate correctly")
    func executeAppliesSGate() async {
        let simulator = CliffordSimulator()

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.sGate, to: 0)

        let tableau = await simulator.execute(circuit)
        let (p0, p1) = tableau.probability(of: 0, measuring: .y)

        #expect(
            abs(p0 - 1.0) < 1e-10,
            "H followed by S should create |+i> state with Y measurement giving 0 (the +1 eigenvalue)",
        )
        #expect(
            abs(p1) < 1e-10,
            "H followed by S should have zero probability of Y measurement giving 1",
        )
    }

    @Test("execute applies CZ gate correctly")
    func executeAppliesCZGate() async {
        let simulator = CliffordSimulator()

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        circuit.append(.cz, to: [0, 1])

        let tableau = await simulator.execute(circuit)

        let xz = PauliString(.x(0), .z(1))
        let zx = PauliString(.z(0), .x(1))
        let xzExpectation = tableau.expectationValue(of: xz)
        let zxExpectation = tableau.expectationValue(of: zx)

        #expect(
            abs(xzExpectation - 1.0) < 1e-10,
            "CZ on |++> should have XZ expectation of +1",
        )
        #expect(
            abs(zxExpectation - 1.0) < 1e-10,
            "CZ on |++> should have ZX expectation of +1",
        )
    }

    @Test("execute applies reset operation")
    func executeAppliesReset() async {
        let simulator = CliffordSimulator()

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.pauliX, to: 0)
        circuit.append(.reset, to: 0)

        let tableau = await simulator.execute(circuit)
        let (p0, _) = tableau.probability(of: 0, measuring: .z)

        #expect(
            abs(p0 - 1.0) < 1e-10 || abs(p0) < 1e-10,
            "Reset should project qubit to a definite state",
        )
    }

    @Test("execute applies measure operation")
    func executeAppliesMeasure() async {
        let simulator = CliffordSimulator()

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.measure, to: 0)

        let tableau = await simulator.execute(circuit)
        let (p0, p1) = tableau.probability(of: 0, measuring: .z)

        #expect(
            abs(p0 - 1.0) < 1e-10 || abs(p1 - 1.0) < 1e-10,
            "Measurement should collapse qubit to a definite state",
        )
    }

    @Test("execute applies CY gate correctly")
    func executeAppliesCYGate() async {
        let simulator = CliffordSimulator()

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cy, to: [0, 1])

        let tableau = await simulator.execute(circuit)
        let (p0, _) = tableau.probability(of: 1, measuring: .z)

        #expect(
            abs(p0 - 1.0) < 1e-10,
            "CY with control in |0> should leave target unchanged",
        )
    }

    @Test("execute applies CH gate correctly")
    func executeAppliesCHGate() async {
        let simulator = CliffordSimulator()

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.pauliX, to: 0)
        circuit.append(.ch, to: [0, 1])

        let tableau = await simulator.execute(circuit)
        let (p0, p1) = tableau.probability(of: 1, measuring: .z)

        #expect(
            abs(p0 - 0.5) < 1e-10 && abs(p1 - 0.5) < 1e-10,
            "CH with control in |1> should create superposition on target",
        )
    }

    @Test("execute applies ISWAP gate correctly")
    func executeAppliesISWAPGate() async {
        let simulator = CliffordSimulator()

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.pauliX, to: 0)
        circuit.append(.iswap, to: [0, 1])

        let tableau = await simulator.execute(circuit)
        let (p0_q0, _) = tableau.probability(of: 0, measuring: .z)
        let (_, p1_q1) = tableau.probability(of: 1, measuring: .z)

        #expect(
            abs(p0_q0 - 1.0) < 1e-10,
            "ISWAP should swap qubit 0 back to |0>",
        )
        #expect(
            abs(p1_q1 - 1.0) < 1e-10,
            "ISWAP should swap qubit 1 to |1>",
        )
    }

    @Test("execute applies phase gate at pi/2")
    func executeAppliesPhaseGatePiOver2() async {
        let simulator = CliffordSimulator()

        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.phase(.value(.pi / 2.0)), to: 0)

        let tableau = await simulator.execute(circuit)
        let (p0, _) = tableau.probability(of: 0, measuring: .y)

        #expect(
            abs(p0 - 1.0) < 1e-10,
            "Phase(pi/2) after H should create |+i> state",
        )
    }

    @Test("execute applies controlled S gate")
    func executeAppliesControlledSGate() async {
        let simulator = CliffordSimulator()

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.pauliX, to: 0)
        circuit.append(.hadamard, to: 1)
        circuit.append(.controlled(gate: .sGate, controls: [0]), to: [0, 1])

        let tableau = await simulator.execute(circuit)
        let (p0, _) = tableau.probability(of: 1, measuring: .y)

        #expect(
            abs(p0 - 1.0) < 1e-10,
            "Controlled-S with control in |1> should apply S to target",
        )
    }
}
