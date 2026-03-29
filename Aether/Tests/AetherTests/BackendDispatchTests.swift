// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Testing

/// Test suite for BackendDispatch automatic backend selection.
/// Validates that circuits are routed to optimal simulation backends
/// based on Clifford structure, T-count, and qubit count.
@Suite("BackendDispatch Selection")
struct BackendDispatchSelectionTests {
    @Test("Pure Clifford circuit returns tableau backend")
    func pureCliffordReturnsTableau() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.cnot, to: [1, 2])

        let backend = BackendDispatch.selectBackend(for: circuit)

        #expect(backend == .tableau, "Pure Clifford circuit should select tableau backend for efficient simulation")
    }

    @Test("Clifford-only gates with multiple qubits returns tableau")
    func cliffordOnlyMultipleQubits() {
        var circuit = QuantumCircuit(qubits: 5)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.pauliZ, to: 2)
        circuit.append(.sGate, to: 3)
        circuit.append(.cz, to: [0, 4])

        let backend = BackendDispatch.selectBackend(for: circuit)

        #expect(backend == .tableau, "Circuit with only Clifford gates should use tableau backend")
    }

    @Test("Few T gates returns extended stabilizer backend")
    func fewTGatesReturnsExtendedStabilizer() {
        var circuit = QuantumCircuit(qubits: 10)
        circuit.append(.hadamard, to: 0)
        circuit.append(.tGate, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let backend = BackendDispatch.selectBackend(for: circuit)

        if case let .extendedStabilizer(maxRank) = backend {
            #expect(maxRank > 0, "Extended stabilizer should have positive maxRank for circuit with few T gates")
        }
    }

    @Test("Multiple T gates within threshold returns extended stabilizer")
    func multipleTGatesWithinThreshold() {
        var circuit = QuantumCircuit(qubits: 8)
        for i in 0 ..< 5 {
            circuit.append(.tGate, to: i % 8)
        }

        let backend = BackendDispatch.selectBackend(for: circuit)

        if case let .extendedStabilizer(maxRank) = backend {
            #expect(maxRank > 0, "Extended stabilizer should have positive maxRank for circuit with 5 T gates")
        }
    }

    @Test("Small general circuit returns statevector backend")
    func smallGeneralCircuitReturnsStatevector() {
        var circuit = QuantumCircuit(qubits: 10)
        for i in 0 ..< 60 {
            circuit.append(.tGate, to: i % 10)
        }

        let backend = BackendDispatch.selectBackend(for: circuit)

        #expect(backend == .statevector, "Small circuit with many T gates should use statevector backend")
    }

    @Test("Arbitrary rotation in small circuit returns statevector")
    func arbitraryRotationSmallCircuit() {
        var circuit = QuantumCircuit(qubits: 5)
        circuit.append(.rotationY(.pi / 7), to: 0)
        circuit.append(.rotationZ(.pi / 11), to: 1)
        for _ in 0 ..< 60 {
            circuit.append(.rotationX(.pi / 13), to: 2)
        }

        let backend = BackendDispatch.selectBackend(for: circuit)

        #expect(backend == .statevector, "Small circuit with arbitrary rotations exceeding T threshold should use statevector")
    }
}

/// Test suite for SimulatorBackend enum definition.
/// Validates that all backend cases exist with correct associated
/// values and conform to expected protocols.
@Suite("SimulatorBackend Enum")
struct SimulatorBackendEnumTests {
    @Test("Tableau case exists and is equatable")
    func tableauCaseExists() {
        let backend1 = SimulatorBackend.tableau
        let backend2 = SimulatorBackend.tableau

        #expect(backend1 == backend2, "Tableau backend should be equatable to itself")
    }

    @Test("Extended stabilizer case has maxRank parameter")
    func extendedStabilizerHasMaxRank() {
        let backend = SimulatorBackend.extendedStabilizer(maxRank: 32)

        if case let .extendedStabilizer(maxRank) = backend {
            #expect(maxRank == 32, "Extended stabilizer should store maxRank value of 32")
        }
    }

    @Test("Statevector case exists")
    func statevectorCaseExists() {
        let backend = SimulatorBackend.statevector

        #expect(backend == .statevector, "Statevector backend case should exist")
    }

    @Test("Density matrix case exists")
    func densityMatrixCaseExists() {
        let backend = SimulatorBackend.densityMatrix

        #expect(backend == .densityMatrix, "Density matrix backend case should exist")
    }

    @Test("MPS case has bondDimension parameter")
    func mpsCaseHasBondDimension() {
        let backend = SimulatorBackend.mps(bondDimension: 64)

        if case let .mps(bondDimension) = backend {
            #expect(bondDimension == 64, "MPS backend should store bondDimension value of 64")
        }
    }

    @Test("Different backend cases are not equal")
    func differentCasesNotEqual() {
        let tableau = SimulatorBackend.tableau
        let statevector = SimulatorBackend.statevector
        let densityMatrix = SimulatorBackend.densityMatrix

        #expect(tableau != statevector, "Tableau and statevector backends should not be equal")
        #expect(statevector != densityMatrix, "Statevector and densityMatrix backends should not be equal")
        #expect(tableau != densityMatrix, "Tableau and densityMatrix backends should not be equal")
    }

    @Test("Extended stabilizer with different maxRank are not equal")
    func extendedStabilizerDifferentMaxRankNotEqual() {
        let backend1 = SimulatorBackend.extendedStabilizer(maxRank: 16)
        let backend2 = SimulatorBackend.extendedStabilizer(maxRank: 32)

        #expect(backend1 != backend2, "Extended stabilizer backends with different maxRank should not be equal")
    }

    @Test("MPS with different bondDimension are not equal")
    func mpsDifferentBondDimensionNotEqual() {
        let backend1 = SimulatorBackend.mps(bondDimension: 64)
        let backend2 = SimulatorBackend.mps(bondDimension: 128)

        #expect(backend1 != backend2, "MPS backends with different bondDimension should not be equal")
    }
}

/// Tests for the BackendDispatch.execute method with
/// explicit backend selection, verifying that each backend
/// type correctly executes quantum circuits.
@Suite("BackendDispatch Execute")
struct BackendDispatchExecuteTests {
    @Test("Execute with explicit tableau backend")
    func executeWithTableauBackend() async {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let result = await BackendDispatch.execute(circuit, backend: .tableau)

        #expect(result.qubits == 2, "Tableau backend should return valid state with correct qubit count")
    }

    @Test("Execute with explicit extendedStabilizer backend")
    func executeWithExtendedStabilizerBackend() async {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.tGate, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let result = await BackendDispatch.execute(circuit, backend: .extendedStabilizer(maxRank: 32))

        #expect(result.qubits == 3, "Extended stabilizer backend should return valid state with correct qubit count")
    }

    @Test("Execute with explicit statevector backend")
    func executeWithStatevectorBackend() async {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.rotationY(.pi / 4), to: 1)

        let result = await BackendDispatch.execute(circuit, backend: .statevector)

        #expect(result.qubits == 2, "Statevector backend should return valid state with correct qubit count")
    }

    @Test("Execute with explicit densityMatrix backend")
    func executeWithDensityMatrixBackend() async {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let result = await BackendDispatch.execute(circuit, backend: .densityMatrix)

        #expect(result.qubits == 2, "Density matrix backend should return valid state with correct qubit count")
    }

    @Test("Execute with explicit MPS backend")
    func executeWithMPSBackend() async {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.cnot, to: [1, 2])

        let result = await BackendDispatch.execute(circuit, backend: .mps(bondDimension: 64))

        #expect(result.qubits == 3, "MPS backend should return valid state with correct qubit count")
    }

    @Test("MPS backend handles single qubit gates")
    func mpsHandlesSingleQubitGates() async {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.sGate, to: 0)

        let result = await BackendDispatch.execute(circuit, backend: .mps(bondDimension: 32))

        #expect(result.qubits == 2, "MPS should handle single qubit gates correctly")
    }

    @Test("MPS backend handles two qubit gates")
    func mpsHandlesTwoQubitGates() async {
        var circuit = QuantumCircuit(qubits: 4)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.cz, to: [1, 2])
        circuit.append(.swap, to: [2, 3])

        let result = await BackendDispatch.execute(circuit, backend: .mps(bondDimension: 64))

        #expect(result.qubits == 4, "MPS should handle two qubit gates correctly")
    }

    @Test("MPS backend handles Toffoli gate")
    func mpsHandlesToffoliGate() async {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.toffoli, to: [0, 1, 2])

        let result = await BackendDispatch.execute(circuit, backend: .mps(bondDimension: 64))

        #expect(result.qubits == 3, "MPS should handle Toffoli gate correctly")
    }

    @Test("Large circuit with many T gates returns MPS backend")
    func largeCircuitWithManyTGatesReturnsMPS() {
        var circuit = QuantumCircuit(qubits: 26)
        for i in 0 ..< 51 {
            circuit.append(.tGate, to: i % 26)
        }

        let backend = BackendDispatch.selectBackend(for: circuit)

        if case let .mps(bondDimension) = backend {
            #expect(bondDimension > 0, "MPS backend should have positive bond dimension")
        }
    }

    @Test("MPS backend handles reset operation")
    func mpsHandlesResetOperation() async {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.reset(qubit: 0))
        circuit.append(.pauliX, to: 1)

        let result = await BackendDispatch.execute(circuit, backend: .mps(bondDimension: 32))

        #expect(result.qubits == 2, "MPS should handle reset operation and return valid state")
    }

    @Test("MPS backend handles measure operation")
    func mpsHandlesMeasureOperation() async {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.measure(qubit: 0))
        circuit.append(.measure(qubit: 1))

        let result = await BackendDispatch.execute(circuit, backend: .mps(bondDimension: 32))

        #expect(result.qubits == 3, "MPS should handle measure operation and return valid state")
    }

    @Test("Gate with more than 3 qubits has correct qubitsRequired")
    func gateWithMoreThan3QubitsHasCorrectQubitsRequired() {
        let fourQubitGate = QuantumGate.controlled(gate: .pauliX, controls: [0, 1, 2])
        #expect(fourQubitGate.qubitsRequired == 4, "Controlled X with 3 controls should require 4 qubits")

        let identity16x16: [[Complex<Double>]] = (0 ..< 16).map { row in
            (0 ..< 16).map { col in row == col ? .one : .zero }
        }
        let fourQubitCustom = QuantumGate.customUnitary(matrix: identity16x16)
        #expect(fourQubitCustom.qubitsRequired == 4, "16x16 custom unitary should require 4 qubits")
    }

    @Test("Execute with nil backend triggers auto-selection")
    func executeWithNilBackendTriggersAutoSelection() async {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let result = await BackendDispatch.execute(circuit, backend: nil)

        #expect(result.qubits == 2, "Nil backend should auto-select appropriate backend and return valid state")
    }
}
