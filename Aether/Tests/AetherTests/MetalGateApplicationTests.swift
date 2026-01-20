// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for Metal GPU-accelerated gate application.
/// Validates that Metal compute shaders produce identical results to CPU implementation,
/// ensuring mathematical correctness while providing performance acceleration for large states.
@Suite("Metal Gate Application")
struct MetalGateApplicationTests {
    @Test("Metal applies Hadamard gate correctly")
    func metalAppliesHadamard() async {
        guard let metal = MetalGateApplication() else { return }

        let state = QuantumState(qubits: 4)
        let newState = await metal.apply(.hadamard, to: 0, state: state)

        #expect(newState.isNormalized())
        #expect(abs(newState.amplitude(of: 0).magnitude - 1.0 / sqrt(2.0)) < 1e-5)
        #expect(abs(newState.amplitude(of: 1).magnitude - 1.0 / sqrt(2.0)) < 1e-5)
    }

    @Test("Metal applies Pauli-X gate correctly")
    func metalAppliesPauliX() async {
        guard let metal = MetalGateApplication() else { return }

        let state = QuantumState(qubits: 4)
        let newState = await metal.apply(.pauliX, to: 0, state: state)

        #expect(newState.isNormalized())
        #expect(abs(newState.amplitude(of: 1).real - 1.0) < 1e-5)
    }

    @Test("Metal applies Phase gate correctly")
    func metalAppliesPhase() async {
        guard let metal = MetalGateApplication() else { return }

        var circuit = QuantumCircuit(qubits: 4)
        circuit.append(.hadamard, to: 0)
        let superposition = circuit.execute()

        let newState = await metal.apply(.phase(.pi / 2), to: 0, state: superposition)

        #expect(newState.isNormalized())
    }

    @Test("Metal applies CNOT gate correctly")
    func metalAppliesCNOT() async {
        guard let metal = MetalGateApplication() else { return }

        var amplitudes = [Complex<Double>](repeating: .zero, count: 16)
        amplitudes[2] = .one
        let state = QuantumState(qubits: 4, amplitudes: amplitudes)

        let newState = await metal.apply(.cnot, to: [0, 1], state: state)

        #expect(newState.isNormalized())
        #expect(abs(newState.amplitude(of: 2).real - 1.0) < 1e-5)
    }

    @Test("Metal creates Bell state")
    func metalCreatesBellState() async {
        guard let metal = MetalGateApplication() else { return }

        var state = QuantumState(qubits: 4)

        state = await metal.apply(.hadamard, to: 0, state: state)
        state = await metal.apply(.cnot, to: [0, 1], state: state)

        #expect(state.isNormalized())

        let p0 = state.probability(of: 0)
        let p3 = state.probability(of: 3)

        #expect(abs(p0 - 0.5) < 1e-4)
        #expect(abs(p3 - 0.5) < 1e-4)
    }

    @Test("Metal applies Toffoli gate")
    func metalAppliesToffoli() async {
        guard let metal = MetalGateApplication() else { return }

        var amplitudes = [Complex<Double>](repeating: .zero, count: 16)
        amplitudes[3] = .one
        let state = QuantumState(qubits: 4, amplitudes: amplitudes)

        let newState = await metal.apply(.toffoli, to: [0, 1, 2], state: state)

        #expect(newState.isNormalized())
        #expect(abs(newState.amplitude(of: 7).real - 1.0) < 1e-5)
    }

    @Test("Metal results match CPU for single-qubit gate")
    func metalMatchesCPUForSingleQubit() async {
        guard let metal = MetalGateApplication() else { return }

        let state = QuantumState(qubits: 4)
        let cpuState = GateApplication.apply(.hadamard, to: 0, state: state)
        let gpuState = await metal.apply(.hadamard, to: 0, state: state)

        for i in 0 ..< cpuState.stateSpaceSize {
            let cpuAmp = cpuState.amplitude(of: i)
            let gpuAmp = gpuState.amplitude(of: i)

            #expect(abs(cpuAmp.real - gpuAmp.real) < 1e-5)
            #expect(abs(cpuAmp.imaginary - gpuAmp.imaginary) < 1e-5)
        }
    }

    @Test("Metal results match CPU for CNOT")
    func metalMatchesCPUForCNOT() async {
        guard let metal = MetalGateApplication() else { return }

        var circuit = QuantumCircuit(qubits: 4)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        let state = circuit.execute()

        let cpuState = GateApplication.apply(.cnot, to: [0, 1], state: state)
        let gpuState = await metal.apply(.cnot, to: [0, 1], state: state)

        for i in 0 ..< cpuState.stateSpaceSize {
            let cpuAmp = cpuState.amplitude(of: i)
            let gpuAmp = gpuState.amplitude(of: i)

            #expect(abs(cpuAmp.real - gpuAmp.real) < 1e-5)
            #expect(abs(cpuAmp.imaginary - gpuAmp.imaginary) < 1e-5)
        }
    }

    @Test("Metal preserves normalization")
    func metalPreservesNormalization() async {
        guard let metal = MetalGateApplication() else { return }

        let gates: [QuantumGate] = [
            .hadamard,
            .pauliX,
            .pauliY,
            .pauliZ,
            .phase(.pi / 4),
            .sGate,
            .tGate,
        ]

        for gate in gates {
            let state = QuantumState(qubits: 4)
            let newState = await metal.apply(gate, to: 0, state: state)

            #expect(newState.isNormalized(), "Gate \(gate) should preserve normalization")
        }
    }

    @Test("Metal handles multiple sequential gates")
    func metalHandlesSequentialGates() async {
        guard let metal = MetalGateApplication() else { return }

        var state = QuantumState(qubits: 4)

        for _ in 0 ..< 10 {
            state = await metal.apply(.hadamard, to: 0, state: state)
        }

        #expect(state.isNormalized())
        #expect(abs(state.amplitude(of: 0).real - 1.0) < 1e-4)
    }

    @Test("applyHybrid uses CPU for small states")
    func hybridUsesCPUForSmallStates() async {
        let state = QuantumState(qubits: 5)

        let newState = await GateApplication.applyHybrid(.hadamard, to: 0, state: state)

        #expect(newState.isNormalized())
        #expect(abs(newState.amplitude(of: 0).magnitude - 1.0 / sqrt(2.0)) < 1e-10)
    }

    @Test("applyHybrid attempts GPU for large states")
    func hybridAttemptsGPUForLargeStates() async {
        let state = QuantumState(qubits: 10)
        let newState = await GateApplication.applyHybrid(.hadamard, to: 0, state: state)

        #expect(newState.isNormalized())
    }

    @Test("Metal handles identity gate")
    func metalHandlesIdentity() async {
        guard let metal = MetalGateApplication() else { return }

        let state = QuantumState(qubits: 4)
        let newState = await metal.apply(.identity, to: 0, state: state)

        #expect(abs(newState.amplitude(of: 0).real - 1.0) < 1e-10)
    }

    @Test("Metal handles rotation gates")
    func metalHandlesRotations() async {
        guard let metal = MetalGateApplication() else { return }

        let state = QuantumState(qubits: 4)

        let rx = await metal.apply(.rotationX(.pi / 4), to: 0, state: state)
        let ry = await metal.apply(.rotationY(.pi / 4), to: 0, state: state)
        let rz = await metal.apply(.rotationZ(.pi / 4), to: 0, state: state)

        #expect(rx.isNormalized())
        #expect(ry.isNormalized())
        #expect(rz.isNormalized())
    }

    @Test("Metal applies controlled-phase gate")
    func metalAppliesControlledPhase() async {
        guard let metal = MetalGateApplication() else { return }

        var circuit = QuantumCircuit(qubits: 4)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        let state = circuit.execute()

        let newState = await metal.apply(.controlledPhase(.pi / 2), to: [0, 1], state: state)

        #expect(newState.isNormalized(), "Controlled-phase gate should preserve normalization")
    }

    @Test("Metal applies SWAP gate")
    func metalAppliesSwap() async {
        guard let metal = MetalGateApplication() else { return }

        var amplitudes = [Complex<Double>](repeating: .zero, count: 16)
        amplitudes[1] = .one
        let state = QuantumState(qubits: 4, amplitudes: amplitudes)
        let newState = await metal.apply(.swap, to: [0, 1], state: state)

        #expect(newState.isNormalized(), "SWAP gate should preserve normalization")
        #expect(abs(newState.amplitude(of: 2).real - 1.0) < 1e-5)
    }

    @Test("Metal two-qubit gates match CPU results")
    func metalTwoQubitMatchesCPU() async {
        guard let metal = MetalGateApplication() else { return }

        var circuit = QuantumCircuit(qubits: 4)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        let state = circuit.execute()

        let cpuState = GateApplication.apply(.controlledPhase(.pi / 4), to: [0, 1], state: state)
        let gpuState = await metal.apply(.controlledPhase(.pi / 4), to: [0, 1], state: state)

        for i in 0 ..< cpuState.stateSpaceSize {
            let cpuAmp = cpuState.amplitude(of: i)
            let gpuAmp = gpuState.amplitude(of: i)

            #expect(abs(cpuAmp.real - gpuAmp.real) < 1e-5)
            #expect(abs(cpuAmp.imaginary - gpuAmp.imaginary) < 1e-5)
        }
    }

    @Test("minimumQubitCountForGPU returns correct threshold for each policy")
    func minimumQubitCountForGPUPolicy() {
        let fastThreshold = MetalGateApplication.minimumQubitCountForGPU(policy: .fast)
        let balancedThreshold = MetalGateApplication.minimumQubitCountForGPU(policy: .balanced)
        let accurateThreshold = MetalGateApplication.minimumQubitCountForGPU(policy: .accurate)

        #expect(fastThreshold == 10, "Fast policy should use 10 qubit threshold")
        #expect(balancedThreshold == 12, "Balanced policy should use 12 qubit threshold")
        #expect(accurateThreshold == Int.max, "Accurate policy should disable GPU entirely")
    }

    @Test("minimumQubitCountForGPU matches static constant for fast policy")
    func minimumQubitCountForGPUConsistency() {
        let staticThreshold = MetalGateApplication.minimumQubitCountForGPU
        let policyThreshold = MetalGateApplication.minimumQubitCountForGPU(policy: .fast)

        #expect(
            staticThreshold == policyThreshold,
            "Static constant should match fast policy threshold",
        )
    }

    @Test("applyHybrid respects precision policy GPU threshold")
    func hybridRespectsPolicy() async {
        let state = QuantumState(qubits: 11)

        let fastResult = await GateApplication.applyHybrid(.hadamard, to: 0, state: state, policy: .fast)
        let balancedResult = await GateApplication.applyHybrid(.hadamard, to: 0, state: state, policy: .balanced)

        #expect(fastResult.isNormalized(), "Fast policy result should be normalized")
        #expect(balancedResult.isNormalized(), "Balanced policy result should be normalized")
    }

    @Test("Metal applies controlled gate with single control")
    func metalAppliesControlledGateSingleControl() async {
        guard let metal = MetalGateApplication() else { return }

        var amplitudes = [Complex<Double>](repeating: .zero, count: 16)
        amplitudes[1] = .one
        let state = QuantumState(qubits: 4, amplitudes: amplitudes)

        let controlledX = QuantumGate.controlled(gate: .pauliX, controls: [0])
        let newState = await metal.apply(controlledX, to: [1], state: state)

        #expect(newState.isNormalized(), "Controlled-X via Metal should preserve normalization")

        #expect(
            abs(newState.amplitude(of: 3).real - 1.0) < 1e-5,
            "Target should flip to |0011⟩ when control is 1",
        )
        #expect(
            abs(newState.amplitude(of: 1).magnitude) < 1e-5,
            "Original |0001⟩ state should be zero after flip",
        )
    }

    @Test("Metal applies controlled gate with two controls")
    func metalAppliesControlledGateTwoControls() async {
        guard let metal = MetalGateApplication() else { return }

        var amplitudes = [Complex<Double>](repeating: .zero, count: 16)
        amplitudes[3] = .one
        let state = QuantumState(qubits: 4, amplitudes: amplitudes)

        let controlledX = QuantumGate.controlled(gate: .pauliX, controls: [0, 1])
        let newState = await metal.apply(controlledX, to: [2], state: state)

        #expect(newState.isNormalized(), "Double-controlled-X via Metal should preserve normalization")

        #expect(
            abs(newState.amplitude(of: 7).real - 1.0) < 1e-5,
            "Target should flip to |0111⟩ when both controls are 1",
        )
    }

    @Test("Metal controlled gate matches CPU result")
    func metalControlledMatchesCPU() async {
        guard let metal = MetalGateApplication() else { return }

        var amplitudes = [Complex<Double>](repeating: .zero, count: 16)
        amplitudes[1] = .one
        let state = QuantumState(qubits: 4, amplitudes: amplitudes)

        let controlledH = QuantumGate.controlled(gate: .hadamard, controls: [0])
        let cpuState = GateApplication.apply(controlledH, to: [1], state: state)
        let gpuState = await metal.apply(controlledH, to: [1], state: state)

        for i in 0 ..< cpuState.stateSpaceSize {
            let cpuAmp = cpuState.amplitude(of: i)
            let gpuAmp = gpuState.amplitude(of: i)

            #expect(
                abs(cpuAmp.real - gpuAmp.real) < 1e-5,
                "CPU and GPU real parts should match for amplitude \(i)",
            )
            #expect(
                abs(cpuAmp.imaginary - gpuAmp.imaginary) < 1e-5,
                "CPU and GPU imaginary parts should match for amplitude \(i)",
            )
        }
    }

    @Test("Metal controlled rotation gate preserves normalization")
    func metalControlledRotationPreservesNormalization() async {
        guard let metal = MetalGateApplication() else { return }

        var amplitudes = [Complex<Double>](repeating: .zero, count: 16)
        amplitudes[3] = .one
        let state = QuantumState(qubits: 4, amplitudes: amplitudes)

        let controlledRy = QuantumGate.controlled(gate: .rotationY(.pi / 4), controls: [0, 1])
        let newState = await metal.apply(controlledRy, to: [2], state: state)

        #expect(newState.isNormalized(), "Controlled-Ry via Metal should preserve normalization")
    }
}
