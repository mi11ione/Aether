// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Test suite for HardwareEfficientAnsatz.
/// Validates ansatz construction, parameter counting, rotation gate choices,
/// and entangling patterns for variational quantum algorithms.
@Suite("HardwareEfficientAnsatz")
struct HardwareEfficientAnsatzTests {
    @Test("Create basic hardware-efficient ansatz")
    func createBasicAnsatz() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 1)

        #expect(ansatz.circuit.numQubits == 2)
        #expect(ansatz.parameterCount == 2)
    }

    @Test("Parameter count scales with depth")
    func parameterCountScalesWithDepth() {
        let depth1 = HardwareEfficientAnsatz(qubits: 3, depth: 1)
        let depth2 = HardwareEfficientAnsatz(qubits: 3, depth: 2)
        let depth3 = HardwareEfficientAnsatz(qubits: 3, depth: 3)

        #expect(depth1.parameterCount == 3)
        #expect(depth2.parameterCount == 6)
        #expect(depth3.parameterCount == 9)
    }

    @Test("Parameter count scales with qubits")
    func parameterCountScalesWithQubits() {
        let q2 = HardwareEfficientAnsatz(qubits: 2, depth: 2)
        let q4 = HardwareEfficientAnsatz(qubits: 4, depth: 2)
        let q6 = HardwareEfficientAnsatz(qubits: 6, depth: 2)

        #expect(q2.parameterCount == 4)
        #expect(q4.parameterCount == 8)
        #expect(q6.parameterCount == 12)
    }

    @Test("Ry rotation gates (default)")
    func ryRotationGates() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 1, rotations: .ry)

        #expect(ansatz.parameterCount == 2)
        #expect(ansatz.circuit.parameters.count == 2)
    }

    @Test("Rx rotation gates")
    func rxRotationGates() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 1, rotations: .rx)

        #expect(ansatz.parameterCount == 2)
    }

    @Test("Rz rotation gates")
    func rzRotationGates() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 1, rotations: .rz)

        #expect(ansatz.parameterCount == 2)
    }

    @Test("Full rotation gates (3x parameters)")
    func fullRotationGates() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 1, rotations: .full)

        #expect(ansatz.parameterCount == 6)
    }

    @Test("Full rotation scales correctly")
    func fullRotationScalesCorrectly() {
        let ansatz = HardwareEfficientAnsatz(qubits: 3, depth: 2, rotations: .full)

        #expect(ansatz.parameterCount == 18)
    }

    @Test("Linear entangling pattern (default)")
    func linearEntanglingPattern() {
        let ansatz = HardwareEfficientAnsatz(qubits: 4, depth: 1, entanglement: .linear)

        #expect(ansatz.circuit.count == 7)
    }

    @Test("Circular entangling pattern")
    func circularEntanglingPattern() {
        let ansatz = HardwareEfficientAnsatz(qubits: 4, depth: 1, entanglement: .circular)

        #expect(ansatz.circuit.count == 8)
    }

    @Test("All-to-all entangling pattern")
    func allToAllEntanglingPattern() {
        let ansatz = HardwareEfficientAnsatz(qubits: 3, depth: 1, entanglement: .allToAll)

        #expect(ansatz.circuit.count == 6)
    }

    @Test("Parameter naming convention")
    func parameterNamingConvention() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 2)
        let params = ansatz.circuit.parameters

        #expect(params.contains(Parameter(name: "theta_0_0")))
        #expect(params.contains(Parameter(name: "theta_0_1")))
        #expect(params.contains(Parameter(name: "theta_1_0")))
        #expect(params.contains(Parameter(name: "theta_1_1")))
    }

    @Test("Full rotation parameter naming")
    func fullRotationParameterNaming() {
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1, rotations: .full)
        let params = ansatz.circuit.parameters

        #expect(params.contains(Parameter(name: "theta_0_0_z1")))
        #expect(params.contains(Parameter(name: "theta_0_0_y")))
        #expect(params.contains(Parameter(name: "theta_0_0_z2")))
    }

    @Test("Can bind and execute ansatz")
    func canBindAndExecuteAnsatz() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 1)

        let params: [Double] = [0.1, 0.2]
        let boundCircuit = ansatz.circuit.bound(with: params)

        #expect(boundCircuit.numQubits == 2)
        #expect(boundCircuit.count > 0)
    }

    @Test("Single qubit ansatz")
    func singleQubitAnsatz() {
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1)

        #expect(ansatz.circuit.numQubits == 1)
        #expect(ansatz.parameterCount == 1)
    }

    @Test("Maximum qubits (30)")
    func maximumQubits() {
        let ansatz = HardwareEfficientAnsatz(qubits: 30, depth: 1)

        #expect(ansatz.circuit.numQubits == 30)
        #expect(ansatz.parameterCount == 30)
    }

    @Test("Maximum depth (100)")
    func maximumDepth() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 100)

        #expect(ansatz.parameterCount == 200)
    }

    @Test("Minimal ansatz (1 qubit, depth 1)")
    func minimalAnsatz() {
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1)

        #expect(ansatz.circuit.count == 1)
        #expect(ansatz.parameterCount == 1)
    }

    @Test("Two-qubit ansatz has CNOT")
    func twoQubitAnsatzHasCNOT() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 1)

        #expect(ansatz.circuit.count == 3)
    }

    @Test("Circular pattern on 2 qubits")
    func circularPatternTwoQubits() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 1, entanglement: .circular)

        #expect(ansatz.circuit.count == 4)
    }

    @Test("RotationGateSet parameters per qubit")
    func rotationGateSetParametersPerQubit() {
        #expect(HardwareEfficientAnsatz.Rotations.ry.parametersPerQubit == 1)
        #expect(HardwareEfficientAnsatz.Rotations.rx.parametersPerQubit == 1)
        #expect(HardwareEfficientAnsatz.Rotations.rz.parametersPerQubit == 1)
        #expect(HardwareEfficientAnsatz.Rotations.full.parametersPerQubit == 3)
    }
}
