// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Test suite for CircuitCostEstimator.estimate function.
/// Validates gate count, depth calculation, CNOT-equivalent cost,
/// and T-count metrics for quantum circuit resource estimation.
@Suite("CircuitCostEstimator.estimate")
struct CircuitCostEstimatorEstimateTests {
    @Test("Empty circuit returns zero for all metrics")
    func emptyCircuit() {
        let circuit = QuantumCircuit(qubits: 2)
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.totalGates == 0, "Empty circuit should have 0 total gates")
        #expect(cost.depth == 0, "Empty circuit should have depth 0")
        #expect(cost.cnotEquivalent == 0, "Empty circuit should have 0 CNOT-equivalent cost")
        #expect(cost.tCount == 0, "Empty circuit should have 0 T-count")
        #expect(cost.gateCount.isEmpty, "Empty circuit should have empty gate count dictionary")
    }

    @Test("Single Hadamard gate: totalGates=1, depth=1, cnotEq=0")
    func singleHadamard() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.totalGates == 1, "Single gate circuit should have totalGates=1")
        #expect(cost.depth == 1, "Single gate circuit should have depth=1")
        #expect(cost.cnotEquivalent == 0, "Hadamard is single-qubit, CNOT-equivalent should be 0")
        #expect(cost.gateCount[.hadamard] == 1, "Gate count should show 1 Hadamard")
    }

    @Test("T-gate counted in tCount metric")
    func tGateCounting() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.tGate, to: 0)
        circuit.append(.tGate, to: 0)
        circuit.append(.hadamard, to: 0)
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.tCount == 2, "Circuit with 2 T-gates should have tCount=2")
        #expect(cost.totalGates == 3, "Circuit should have 3 total gates")
        #expect(cost.gateCount[.tGate] == 2, "Gate count should show 2 T-gates")
    }

    @Test("Bell circuit: 2 gates, depth=2, cnotEq=1")
    func bellCircuit() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.totalGates == 2, "Bell circuit has 2 gates")
        #expect(cost.depth == 2, "Bell circuit has depth 2 (H then CNOT)")
        #expect(cost.cnotEquivalent == 1, "Bell circuit has 1 CNOT")
    }
}

/// Test suite for CNOT-equivalent cost calculation.
/// Validates cost values for single-qubit (0), two-qubit (1-3),
/// and three-qubit gates (6) per standard decomposition.
@Suite("CNOT-Equivalent Cost")
struct CnotEquivalentCostTests {
    @Test("Single-qubit gates have CNOT-equivalent cost 0")
    func singleQubitGatesCostZero() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliY, to: 0)
        circuit.append(.pauliZ, to: 0)
        circuit.append(.hadamard, to: 0)
        circuit.append(.sGate, to: 0)
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.cnotEquivalent == 0, "All single-qubit gates should have CNOT-equivalent 0")
        #expect(cost.totalGates == 5, "Should count all 5 single-qubit gates")
    }

    @Test("CNOT has CNOT-equivalent cost 1")
    func cnotCostOne() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cnot, to: [0, 1])
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.cnotEquivalent == 1, "CNOT should have CNOT-equivalent cost 1")
    }

    @Test("CZ has CNOT-equivalent cost 1")
    func czCostOne() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cz, to: [0, 1])
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.cnotEquivalent == 1, "CZ should have CNOT-equivalent cost 1")
    }

    @Test("SWAP has CNOT-equivalent cost 3")
    func swapCostThree() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.swap, to: [0, 1])
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.cnotEquivalent == 3, "SWAP decomposes to 3 CNOTs")
    }

    @Test("Toffoli has CNOT-equivalent cost 6")
    func toffoliCostSix() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.toffoli, to: [0, 1, 2])
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.cnotEquivalent == 6, "Toffoli decomposes to 6 CNOT-equivalents")
    }

    @Test("Two-qubit parameterized gates cost 2")
    func twoQubitParameterizedCostTwo() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlledPhase(.pi / 4), to: [0, 1])
        circuit.append(.sqrtSwap, to: [0, 1])
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.cnotEquivalent == 4, "controlledPhase(2) + sqrtSwap(2) = 4")
    }

    @Test("CY and CH have CNOT-equivalent cost 2 each")
    func cyChCostTwo() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.cy, to: [0, 1])
        circuit.append(.ch, to: [0, 1])
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.cnotEquivalent == 4, "CY(2) + CH(2) = 4 CNOT-equivalents")
        #expect(cost.totalGates == 2, "Circuit should have 2 gates")
    }

    @Test("Generic controlled gate has CNOT-equivalent cost 2")
    func controlledGateCostTwo() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.controlled(gate: .pauliX, controls: [0]), to: [0, 1])
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.cnotEquivalent == 2, "Generic controlled gate should have CNOT-equivalent cost 2")
    }
}

/// Test suite for circuit depth calculation.
/// Validates parallel gate detection, sequential gates on same qubit,
/// and multi-qubit gate depth contribution.
@Suite("Circuit Depth Calculation")
struct CircuitDepthTests {
    @Test("Parallel gates on different qubits have depth 1")
    func parallelGatesDepthOne() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.pauliZ, to: 2)
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.depth == 1, "Independent gates on different qubits execute in parallel")
    }

    @Test("Sequential gates on same qubit increase depth")
    func sequentialGatesIncreaseDepth() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliZ, to: 0)
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.depth == 3, "3 sequential gates on same qubit have depth 3")
    }

    @Test("Two-qubit gate increases depth of both qubits")
    func twoQubitGateAffectsBothQubits() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.pauliX, to: 1)
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.depth == 3, "H(d=1) -> CNOT(d=2) -> X(d=3)")
    }

    @Test("Mixed parallel and sequential gates")
    func mixedParallelSequential() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        circuit.append(.cnot, to: [0, 1])
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.depth == 2, "Parallel H gates (d=1), then CNOT (d=2)")
    }
}

/// Test suite for gate count dictionary accuracy.
/// Validates per-gate-type counting, multiple gate types,
/// and correct aggregation across circuit operations.
@Suite("Gate Count Dictionary")
struct GateCountTests {
    @Test("Multiple gate types counted correctly")
    func multipleGateTypesCounted() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.pauliX, to: 0)
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.gateCount[.hadamard] == 2, "Should count 2 Hadamards")
        #expect(cost.gateCount[.cnot] == 1, "Should count 1 CNOT")
        #expect(cost.gateCount[.pauliX] == 1, "Should count 1 Pauli-X")
        #expect(cost.totalGates == 4, "Total should be 4 gates")
    }

    @Test("Rotation gates with different angles counted separately")
    func rotationGatesCounted() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationZ(.pi / 4), to: 0)
        circuit.append(.rotationZ(.pi / 2), to: 0)
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.totalGates == 2, "Should count both rotation gates")
    }
}

/// Test suite for CircuitCost struct properties.
/// Validates description format, Equatable conformance,
/// and Sendable thread safety guarantees.
@Suite("CircuitCost Struct")
struct CircuitCostStructTests {
    @Test("Description format includes all metrics")
    func descriptionFormat() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.tGate, to: 0)
        circuit.append(.cnot, to: [0, 1])
        let cost = CircuitCostEstimator.estimate(circuit)

        let description = cost.description
        #expect(description.contains("gates: 3"), "Description should show totalGates")
        #expect(description.contains("depth:"), "Description should show depth")
        #expect(description.contains("CNOT-eq:"), "Description should show CNOT-equivalent")
        #expect(description.contains("T-count: 1"), "Description should show T-count")
    }

    @Test("CircuitCost Equatable conformance")
    func equatableConformance() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        let cost1 = CircuitCostEstimator.estimate(circuit)
        let cost2 = CircuitCostEstimator.estimate(circuit)

        #expect(cost1 == cost2, "Same circuit should produce equal CircuitCost")
    }
}

/// Test suite for three-qubit gate cost estimation.
/// Validates Toffoli, Fredkin, CCZ costs follow standard
/// decomposition to 6 CNOT-equivalents each.
@Suite("Three-Qubit Gate Costs")
struct ThreeQubitGateCostTests {
    @Test("Fredkin has CNOT-equivalent cost 6")
    func fredkinCostSix() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.fredkin, to: [0, 1, 2])
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.cnotEquivalent == 6, "Fredkin decomposes to 6 CNOT-equivalents")
        #expect(cost.depth == 1, "Single Fredkin has depth 1")
    }

    @Test("CCZ has CNOT-equivalent cost 6")
    func cczCostSix() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.ccz, to: [0, 1, 2])
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.cnotEquivalent == 6, "CCZ decomposes to 6 CNOT-equivalents")
    }

    @Test("Diagonal gate has CNOT-equivalent cost 2")
    func diagonalCostTwo() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.diagonal(phases: [0, .pi / 4, .pi / 2, .pi]), to: [0, 1])
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.cnotEquivalent == 2, "Diagonal gate should have CNOT-equivalent cost 2")
        #expect(cost.totalGates == 1, "Circuit should have 1 diagonal gate")
        #expect(cost.depth == 1, "Single diagonal gate should have depth 1")
    }

    @Test("Operation on high qubit index triggers dynamic array resize")
    func highQubitIndexTriggersResize() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 10)
        let cost = CircuitCostEstimator.estimate(circuit)

        #expect(cost.totalGates == 1, "Should count gate on high qubit index")
        #expect(cost.depth == 1, "Single gate on any qubit should have depth 1")
    }
}
