// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Testing

/// Test suite for HardwareEfficientAnsatz.
/// Validates ansatz construction, parameter counting, rotation gate choices,
/// and entangling patterns for variational quantum algorithms.
@Suite("HardwareEfficientAnsatz")
struct HardwareEfficientAnsatzTests {
    @Test("Create basic hardware-efficient ansatz")
    func createBasicAnsatz() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 1)

        #expect(ansatz.qubits == 2, "Ansatz should have 2 qubits")
        #expect(ansatz.parameterCount == 2, "2 qubits x 1 layer x 1 param = 2 parameters")
    }

    @Test("Parameter count scales with depth")
    func parameterCountScalesWithDepth() {
        let depth1 = HardwareEfficientAnsatz(qubits: 3, depth: 1)
        let depth2 = HardwareEfficientAnsatz(qubits: 3, depth: 2)
        let depth3 = HardwareEfficientAnsatz(qubits: 3, depth: 3)

        #expect(depth1.parameterCount == 3, "3 qubits x 1 layer = 3 parameters")
        #expect(depth2.parameterCount == 6, "3 qubits x 2 layers = 6 parameters")
        #expect(depth3.parameterCount == 9, "3 qubits x 3 layers = 9 parameters")
    }

    @Test("Parameter count scales with qubits")
    func parameterCountScalesWithQubits() {
        let q2 = HardwareEfficientAnsatz(qubits: 2, depth: 2)
        let q4 = HardwareEfficientAnsatz(qubits: 4, depth: 2)
        let q6 = HardwareEfficientAnsatz(qubits: 6, depth: 2)

        #expect(q2.parameterCount == 4, "2 qubits x 2 layers = 4 parameters")
        #expect(q4.parameterCount == 8, "4 qubits x 2 layers = 8 parameters")
        #expect(q6.parameterCount == 12, "6 qubits x 2 layers = 12 parameters")
    }

    @Test("Ry rotation gates (default)")
    func ryRotationGates() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 1, rotations: .ry)

        #expect(ansatz.parameterCount == 2, "Ry uses 1 param per qubit per layer")
        #expect(ansatz.circuit.parameters.count == 2, "Circuit should contain 2 unique parameters")
    }

    @Test("Rx rotation gates")
    func rxRotationGates() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 1, rotations: .rx)

        #expect(ansatz.parameterCount == 2, "Rx uses 1 param per qubit per layer")
    }

    @Test("Rz rotation gates")
    func rzRotationGates() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 1, rotations: .rz)

        #expect(ansatz.parameterCount == 2, "Rz uses 1 param per qubit per layer")
    }

    @Test("Full rotation gates (3x parameters)")
    func fullRotationGates() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 1, rotations: .full)

        #expect(ansatz.parameterCount == 6, "Full rotations use 3 params per qubit per layer")
    }

    @Test("Full rotation scales correctly")
    func fullRotationScalesCorrectly() {
        let ansatz = HardwareEfficientAnsatz(qubits: 3, depth: 2, rotations: .full)

        #expect(ansatz.parameterCount == 18, "3 qubits x 2 layers x 3 params = 18")
    }

    @Test("Linear entangling pattern (default)")
    func linearEntanglingPattern() {
        let ansatz = HardwareEfficientAnsatz(qubits: 4, depth: 1, entanglement: .linear)

        #expect(ansatz.circuit.count == 7, "4 rotations + 3 linear CNOTs = 7 gates")
    }

    @Test("Circular entangling pattern")
    func circularEntanglingPattern() {
        let ansatz = HardwareEfficientAnsatz(qubits: 4, depth: 1, entanglement: .circular)

        #expect(ansatz.circuit.count == 8, "4 rotations + 3 linear CNOTs + 1 wrap CNOT = 8 gates")
    }

    @Test("All-to-all entangling pattern")
    func allToAllEntanglingPattern() {
        let ansatz = HardwareEfficientAnsatz(qubits: 3, depth: 1, entanglement: .allToAll)

        #expect(ansatz.circuit.count == 6, "3 rotations + 3 all-to-all CNOTs = 6 gates")
    }

    @Test("Parameter naming convention")
    func parameterNamingConvention() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 2)
        let params = ansatz.circuit.parameters

        #expect(params.contains(Parameter(name: "theta_0_0")), "Layer 0 qubit 0 parameter should exist")
        #expect(params.contains(Parameter(name: "theta_0_1")), "Layer 0 qubit 1 parameter should exist")
        #expect(params.contains(Parameter(name: "theta_1_0")), "Layer 1 qubit 0 parameter should exist")
        #expect(params.contains(Parameter(name: "theta_1_1")), "Layer 1 qubit 1 parameter should exist")
    }

    @Test("Full rotation parameter naming")
    func fullRotationParameterNaming() {
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1, rotations: .full)
        let params = ansatz.circuit.parameters

        #expect(params.contains(Parameter(name: "theta_0_0_z1")), "First Rz parameter should exist")
        #expect(params.contains(Parameter(name: "theta_0_0_y")), "Ry parameter should exist")
        #expect(params.contains(Parameter(name: "theta_0_0_z2")), "Second Rz parameter should exist")
    }

    @Test("Can bind and execute ansatz")
    func canBindAndExecuteAnsatz() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 1)

        let params: [Double] = [0.1, 0.2]
        let boundCircuit = ansatz.circuit.bound(with: params)

        #expect(boundCircuit.qubits == 2, "Bound circuit should preserve qubit count")
        #expect(boundCircuit.count > 0, "Bound circuit should contain gates")
    }

    @Test("Single qubit ansatz")
    func singleQubitAnsatz() {
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1)

        #expect(ansatz.qubits == 1, "Ansatz should have 1 qubit")
        #expect(ansatz.parameterCount == 1, "1 qubit x 1 layer = 1 parameter")
    }

    @Test("Maximum qubits (30)")
    func maximumQubits() {
        let ansatz = HardwareEfficientAnsatz(qubits: 30, depth: 1)

        #expect(ansatz.qubits == 30, "Ansatz should accept maximum 30 qubits")
        #expect(ansatz.parameterCount == 30, "30 qubits x 1 layer = 30 parameters")
    }

    @Test("Maximum depth (100)")
    func maximumDepth() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 100)

        #expect(ansatz.parameterCount == 200, "2 qubits x 100 layers = 200 parameters")
    }

    @Test("Minimal ansatz (1 qubit, depth 1)")
    func minimalAnsatz() {
        let ansatz = HardwareEfficientAnsatz(qubits: 1, depth: 1)

        #expect(ansatz.circuit.count == 1, "Minimal ansatz should have 1 rotation gate only")
        #expect(ansatz.parameterCount == 1, "Minimal ansatz should have 1 parameter")
    }

    @Test("Two-qubit ansatz has CNOT")
    func twoQubitAnsatzHasCNOT() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 1)

        #expect(ansatz.circuit.count == 3, "2 rotations + 1 CNOT = 3 gates")
    }

    @Test("Circular pattern on 2 qubits")
    func circularPatternTwoQubits() {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 1, entanglement: .circular)

        #expect(ansatz.circuit.count == 4, "2 rotations + 1 linear CNOT + 1 wrap CNOT = 4 gates")
    }

    @Test("RotationGateSet parameters per qubit")
    func rotationGateSetParametersPerQubit() {
        #expect(HardwareEfficientAnsatz.Rotations.ry.parametersPerQubit == 1, "Ry should use 1 parameter")
        #expect(HardwareEfficientAnsatz.Rotations.rx.parametersPerQubit == 1, "Rx should use 1 parameter")
        #expect(HardwareEfficientAnsatz.Rotations.rz.parametersPerQubit == 1, "Rz should use 1 parameter")
        #expect(HardwareEfficientAnsatz.Rotations.full.parametersPerQubit == 3, "Full should use 3 parameters")
    }
}
