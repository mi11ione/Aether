//
//  MetalGateApplicationTests.swift
//  AetherTests
//
//  Test suite for Metal GPU-accelerated quantum gate application.
//  Validates compute shader implementations, CPU/GPU result consistency,
//  and hybrid execution strategy for optimal performance.
//
//  Created by mi11ion on 23/10/25.
//

@testable import Aether
import Foundation
import Testing

/// Test suite for Metal GPU-accelerated gate application.
/// Validates that Metal compute shaders produce identical results to CPU implementation,
/// ensuring mathematical correctness while providing performance acceleration for large states.
@Suite("Metal Gate Application Tests")
struct MetalGateApplicationTests {
    @Test("Metal applies Hadamard gate correctly")
    func metalAppliesHadamard() {
        guard let metal = MetalGateApplication() else { return }

        let state = QuantumState(numQubits: 4)
        let newState = metal.apply(gate: .hadamard, to: [0], state: state)

        #expect(newState.isNormalized())
        #expect(abs(newState.getAmplitude(ofState: 0).magnitude - 1.0 / sqrt(2.0)) < 1e-5)
        #expect(abs(newState.getAmplitude(ofState: 1).magnitude - 1.0 / sqrt(2.0)) < 1e-5)
    }

    @Test("Metal applies Pauli-X gate correctly")
    func metalAppliesPauliX() {
        guard let metal = MetalGateApplication() else { return }

        let state = QuantumState(numQubits: 4)
        let newState = metal.apply(gate: .pauliX, to: [0], state: state)

        #expect(newState.isNormalized())
        #expect(abs(newState.getAmplitude(ofState: 1).real - 1.0) < 1e-5)
    }

    @Test("Metal applies Phase gate correctly")
    func metalAppliesPhase() {
        guard let metal = MetalGateApplication() else { return }

        var circuit = QuantumCircuit(numQubits: 4)
        circuit.append(gate: .hadamard, toQubit: 0)
        let superposition = circuit.execute()

        let newState = metal.apply(gate: .phase(theta: .pi / 2), to: [0], state: superposition)

        #expect(newState.isNormalized())
    }

    @Test("Metal applies CNOT gate correctly")
    func metalAppliesCNOT() {
        guard let metal = MetalGateApplication() else { return }

        var amplitudes = [Complex<Double>](repeating: .zero, count: 16)
        amplitudes[2] = .one
        let state = QuantumState(numQubits: 4, amplitudes: amplitudes)

        let newState = metal.apply(gate: .cnot(control: 0, target: 1), to: [], state: state)

        #expect(newState.isNormalized())
        #expect(abs(newState.getAmplitude(ofState: 2).real - 1.0) < 1e-5)
    }

    @Test("Metal creates Bell state")
    func metalCreatesBellState() {
        guard let metal = MetalGateApplication() else { return }

        var state = QuantumState(numQubits: 4)

        state = metal.apply(gate: .hadamard, to: [0], state: state)
        state = metal.apply(gate: .cnot(control: 0, target: 1), to: [], state: state)

        #expect(state.isNormalized())

        let p0 = state.probability(ofState: 0)
        let p3 = state.probability(ofState: 3)

        #expect(abs(p0 - 0.5) < 1e-4)
        #expect(abs(p3 - 0.5) < 1e-4)
    }

    @Test("Metal applies Toffoli gate")
    func metalAppliesToffoli() {
        guard let metal = MetalGateApplication() else { return }

        var amplitudes = [Complex<Double>](repeating: .zero, count: 16)
        amplitudes[3] = .one
        let state = QuantumState(numQubits: 4, amplitudes: amplitudes)

        let newState = metal.apply(gate: .toffoli(control1: 0, control2: 1, target: 2), to: [], state: state)

        #expect(newState.isNormalized())
        #expect(abs(newState.getAmplitude(ofState: 7).real - 1.0) < 1e-5)
    }

    @Test("Metal results match CPU for single-qubit gate")
    func metalMatchesCPUForSingleQubit() {
        guard let metal = MetalGateApplication() else { return }

        let state = QuantumState(numQubits: 4)
        let cpuState = GateApplication.apply(gate: .hadamard, to: [0], state: state)
        let gpuState = metal.apply(gate: .hadamard, to: [0], state: state)

        for i in 0 ..< cpuState.stateSpaceSize {
            let cpuAmp = cpuState.getAmplitude(ofState: i)
            let gpuAmp = gpuState.getAmplitude(ofState: i)

            #expect(abs(cpuAmp.real - gpuAmp.real) < 1e-5)
            #expect(abs(cpuAmp.imaginary - gpuAmp.imaginary) < 1e-5)
        }
    }

    @Test("Metal results match CPU for CNOT")
    func metalMatchesCPUForCNOT() {
        guard let metal = MetalGateApplication() else { return }

        var circuit = QuantumCircuit(numQubits: 4)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .hadamard, toQubit: 1)
        let state = circuit.execute()

        let cpuState = GateApplication.apply(gate: .cnot(control: 0, target: 1), to: [], state: state)
        let gpuState = metal.apply(gate: .cnot(control: 0, target: 1), to: [], state: state)

        for i in 0 ..< cpuState.stateSpaceSize {
            let cpuAmp = cpuState.getAmplitude(ofState: i)
            let gpuAmp = gpuState.getAmplitude(ofState: i)

            #expect(abs(cpuAmp.real - gpuAmp.real) < 1e-5)
            #expect(abs(cpuAmp.imaginary - gpuAmp.imaginary) < 1e-5)
        }
    }

    @Test("Metal preserves normalization")
    func metalPreservesNormalization() {
        guard let metal = MetalGateApplication() else { return }

        let gates: [QuantumGate] = [
            .hadamard,
            .pauliX,
            .pauliY,
            .pauliZ,
            .phase(theta: .pi / 4),
            .sGate,
            .tGate,
        ]

        for gate in gates {
            let state = QuantumState(numQubits: 4)
            let newState = metal.apply(gate: gate, to: [0], state: state)

            #expect(newState.isNormalized(), "Gate \(gate) should preserve normalization")
        }
    }

    @Test("Metal handles multiple sequential gates")
    func metalHandlesSequentialGates() {
        guard let metal = MetalGateApplication() else { return }

        var state = QuantumState(numQubits: 4)

        for _ in 0 ..< 10 {
            state = metal.apply(gate: .hadamard, to: [0], state: state)
        }

        #expect(state.isNormalized())
        #expect(abs(state.getAmplitude(ofState: 0).real - 1.0) < 1e-4)
    }

    @Test("applyHybrid uses CPU for small states")
    func hybridUsesCPUForSmallStates() {
        let state = QuantumState(numQubits: 5)

        let newState = GateApplication.applyHybrid(gate: .hadamard, to: [0], state: state)

        #expect(newState.isNormalized())
        #expect(abs(newState.getAmplitude(ofState: 0).magnitude - 1.0 / sqrt(2.0)) < 1e-10)
    }

    @Test("applyHybrid attempts GPU for large states")
    func hybridAttemptsGPUForLargeStates() {
        let state = QuantumState(numQubits: 10)
        let newState = GateApplication.applyHybrid(gate: .hadamard, to: [0], state: state)

        #expect(newState.isNormalized())
    }

    @Test("Metal handles identity gate")
    func metalHandlesIdentity() {
        guard let metal = MetalGateApplication() else { return }

        let state = QuantumState(numQubits: 4)
        let newState = metal.apply(gate: .identity, to: [0], state: state)

        #expect(abs(newState.getAmplitude(ofState: 0).real - 1.0) < 1e-10)
    }

    @Test("Metal handles rotation gates")
    func metalHandlesRotations() {
        guard let metal = MetalGateApplication() else { return }

        let state = QuantumState(numQubits: 4)

        let rx = metal.apply(gate: .rotationX(theta: .pi / 4), to: [0], state: state)
        let ry = metal.apply(gate: .rotationY(theta: .pi / 4), to: [0], state: state)
        let rz = metal.apply(gate: .rotationZ(theta: .pi / 4), to: [0], state: state)

        #expect(rx.isNormalized())
        #expect(ry.isNormalized())
        #expect(rz.isNormalized())
    }

    @Test("Metal applies controlled-phase gate")
    func metalAppliesControlledPhase() {
        guard let metal = MetalGateApplication() else { return }

        var circuit = QuantumCircuit(numQubits: 4)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .hadamard, toQubit: 1)
        let state = circuit.execute()

        let newState = metal.apply(gate: .controlledPhase(theta: .pi / 2, control: 0, target: 1), to: [0, 1], state: state)

        #expect(newState.isNormalized(), "Controlled-phase gate should preserve normalization")
    }

    @Test("Metal applies SWAP gate")
    func metalAppliesSwap() {
        guard let metal = MetalGateApplication() else { return }

        var amplitudes = [Complex<Double>](repeating: .zero, count: 16)
        amplitudes[1] = .one
        let state = QuantumState(numQubits: 4, amplitudes: amplitudes)
        let newState = metal.apply(gate: .swap(qubit1: 0, qubit2: 1), to: [0, 1], state: state)

        #expect(newState.isNormalized(), "SWAP gate should preserve normalization")
        #expect(abs(newState.getAmplitude(ofState: 2).real - 1.0) < 1e-5)
    }

    @Test("Metal two-qubit gates match CPU results")
    func metalTwoQubitMatchesCPU() {
        guard let metal = MetalGateApplication() else { return }

        var circuit = QuantumCircuit(numQubits: 4)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .hadamard, toQubit: 1)
        let state = circuit.execute()

        let cpuState = GateApplication.apply(gate: .controlledPhase(theta: .pi / 4, control: 0, target: 1), to: [0, 1], state: state)
        let gpuState = metal.apply(gate: .controlledPhase(theta: .pi / 4, control: 0, target: 1), to: [0, 1], state: state)

        for i in 0 ..< cpuState.stateSpaceSize {
            let cpuAmp = cpuState.getAmplitude(ofState: i)
            let gpuAmp = gpuState.getAmplitude(ofState: i)

            #expect(abs(cpuAmp.real - gpuAmp.real) < 1e-5)
            #expect(abs(cpuAmp.imaginary - gpuAmp.imaginary) < 1e-5)
        }
    }
}
