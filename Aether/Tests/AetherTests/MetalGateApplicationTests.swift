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

        let state = QuantumState(numQubits: 4)
        let newState = await metal.apply(.hadamard, to: 0, state: state)

        #expect(newState.isNormalized())
        #expect(abs(newState.amplitude(of: 0).magnitude - 1.0 / sqrt(2.0)) < 1e-5)
        #expect(abs(newState.amplitude(of: 1).magnitude - 1.0 / sqrt(2.0)) < 1e-5)
    }

    @Test("Metal applies Pauli-X gate correctly")
    func metalAppliesPauliX() async {
        guard let metal = MetalGateApplication() else { return }

        let state = QuantumState(numQubits: 4)
        let newState = await metal.apply(.pauliX, to: 0, state: state)

        #expect(newState.isNormalized())
        #expect(abs(newState.amplitude(of: 1).real - 1.0) < 1e-5)
    }

    @Test("Metal applies Phase gate correctly")
    func metalAppliesPhase() async {
        guard let metal = MetalGateApplication() else { return }

        var circuit = QuantumCircuit(numQubits: 4)
        circuit.append(.hadamard, to: 0)
        let superposition = circuit.execute()

        let newState = await metal.apply(.phase(angle: .pi / 2), to: 0, state: superposition)

        #expect(newState.isNormalized())
    }

    @Test("Metal applies CNOT gate correctly")
    func metalAppliesCNOT() async {
        guard let metal = MetalGateApplication() else { return }

        var amplitudes = [Complex<Double>](repeating: .zero, count: 16)
        amplitudes[2] = .one
        let state = QuantumState(numQubits: 4, amplitudes: amplitudes)

        let newState = await metal.apply(.cnot, to: [0, 1], state: state)

        #expect(newState.isNormalized())
        #expect(abs(newState.amplitude(of: 2).real - 1.0) < 1e-5)
    }

    @Test("Metal creates Bell state")
    func metalCreatesBellState() async {
        guard let metal = MetalGateApplication() else { return }

        var state = QuantumState(numQubits: 4)

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
        let state = QuantumState(numQubits: 4, amplitudes: amplitudes)

        let newState = await metal.apply(.toffoli, to: [0, 1, 2], state: state)

        #expect(newState.isNormalized())
        #expect(abs(newState.amplitude(of: 7).real - 1.0) < 1e-5)
    }

    @Test("Metal results match CPU for single-qubit gate")
    func metalMatchesCPUForSingleQubit() async {
        guard let metal = MetalGateApplication() else { return }

        let state = QuantumState(numQubits: 4)
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

        var circuit = QuantumCircuit(numQubits: 4)
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
            .phase(angle: .pi / 4),
            .sGate,
            .tGate,
        ]

        for gate in gates {
            let state = QuantumState(numQubits: 4)
            let newState = await metal.apply(gate, to: 0, state: state)

            #expect(newState.isNormalized(), "Gate \(gate) should preserve normalization")
        }
    }

    @Test("Metal handles multiple sequential gates")
    func metalHandlesSequentialGates() async {
        guard let metal = MetalGateApplication() else { return }

        var state = QuantumState(numQubits: 4)

        for _ in 0 ..< 10 {
            state = await metal.apply(.hadamard, to: 0, state: state)
        }

        #expect(state.isNormalized())
        #expect(abs(state.amplitude(of: 0).real - 1.0) < 1e-4)
    }

    @Test("applyHybrid uses CPU for small states")
    func hybridUsesCPUForSmallStates() async {
        let state = QuantumState(numQubits: 5)

        let newState = await GateApplication.applyHybrid(.hadamard, to: 0, state: state)

        #expect(newState.isNormalized())
        #expect(abs(newState.amplitude(of: 0).magnitude - 1.0 / sqrt(2.0)) < 1e-10)
    }

    @Test("applyHybrid attempts GPU for large states")
    func hybridAttemptsGPUForLargeStates() async {
        let state = QuantumState(numQubits: 10)
        let newState = await GateApplication.applyHybrid(.hadamard, to: 0, state: state)

        #expect(newState.isNormalized())
    }

    @Test("Metal handles identity gate")
    func metalHandlesIdentity() async {
        guard let metal = MetalGateApplication() else { return }

        let state = QuantumState(numQubits: 4)
        let newState = await metal.apply(.identity, to: 0, state: state)

        #expect(abs(newState.amplitude(of: 0).real - 1.0) < 1e-10)
    }

    @Test("Metal handles rotation gates")
    func metalHandlesRotations() async {
        guard let metal = MetalGateApplication() else { return }

        let state = QuantumState(numQubits: 4)

        let rx = await metal.apply(.rotationX(theta: .pi / 4), to: 0, state: state)
        let ry = await metal.apply(.rotationY(theta: .pi / 4), to: 0, state: state)
        let rz = await metal.apply(.rotationZ(theta: .pi / 4), to: 0, state: state)

        #expect(rx.isNormalized())
        #expect(ry.isNormalized())
        #expect(rz.isNormalized())
    }

    @Test("Metal applies controlled-phase gate")
    func metalAppliesControlledPhase() async {
        guard let metal = MetalGateApplication() else { return }

        var circuit = QuantumCircuit(numQubits: 4)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        let state = circuit.execute()

        let newState = await metal.apply(.controlledPhase(theta: .pi / 2), to: [0, 1], state: state)

        #expect(newState.isNormalized(), "Controlled-phase gate should preserve normalization")
    }

    @Test("Metal applies SWAP gate")
    func metalAppliesSwap() async {
        guard let metal = MetalGateApplication() else { return }

        var amplitudes = [Complex<Double>](repeating: .zero, count: 16)
        amplitudes[1] = .one
        let state = QuantumState(numQubits: 4, amplitudes: amplitudes)
        let newState = await metal.apply(.swap, to: [0, 1], state: state)

        #expect(newState.isNormalized(), "SWAP gate should preserve normalization")
        #expect(abs(newState.amplitude(of: 2).real - 1.0) < 1e-5)
    }

    @Test("Metal two-qubit gates match CPU results")
    func metalTwoQubitMatchesCPU() async {
        guard let metal = MetalGateApplication() else { return }

        var circuit = QuantumCircuit(numQubits: 4)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        let state = circuit.execute()

        let cpuState = GateApplication.apply(.controlledPhase(theta: .pi / 4), to: [0, 1], state: state)
        let gpuState = await metal.apply(.controlledPhase(theta: .pi / 4), to: [0, 1], state: state)

        for i in 0 ..< cpuState.stateSpaceSize {
            let cpuAmp = cpuState.amplitude(of: i)
            let gpuAmp = gpuState.amplitude(of: i)

            #expect(abs(cpuAmp.real - gpuAmp.real) < 1e-5)
            #expect(abs(cpuAmp.imaginary - gpuAmp.imaginary) < 1e-5)
        }
    }
}
