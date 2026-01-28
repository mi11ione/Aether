// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Testing

/// Tests for GPUProfiler functionality.
/// Validates timing metrics, GPU/CPU mode selection,
/// and profile output for circuit characteristics.
@Suite("GPUProfiler")
struct GPUProfilerTests {
    @Test("GPUProfile description contains expected labels for GPU mode")
    func gpuProfileDescriptionWithGPU() async {
        // Use a large circuit to trigger GPU mode (if Metal is available)
        var circuit = QuantumCircuit(qubits: 12)
        circuit.append(.hadamard, to: 0)

        let profile = await GPUProfiler.profile(circuit, precisionPolicy: .fast)

        // Verify description format based on whether GPU was used
        if profile.didUseGPU {
            #expect(profile.description.contains("GPU:"), "Description should contain GPU label for GPU mode")
            #expect(profile.description.contains("CPU:"), "Description should contain CPU label for GPU mode")
            #expect(profile.description.contains("Transfer:"), "Description should contain transfer info for GPU mode")
            #expect(profile.description.contains("Utilization:"), "Description should contain utilization for GPU mode")
        } else {
            #expect(profile.description.contains("CPU-only:"), "Description should indicate CPU-only mode")
        }
    }

    @Test("GPUProfile with CPU-only mode has correct description format")
    func gpuProfileDescriptionCPUOnly() async {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)

        // Accurate policy forces CPU-only execution
        let profile = await GPUProfiler.profile(circuit, precisionPolicy: .accurate)

        #expect(profile.description.contains("CPU-only:"), "Description should indicate CPU-only mode when GPU not used")
        #expect(!profile.description.contains("GPU:"), "Description should not contain GPU label in CPU-only mode")
    }

    @Test("GPUProfile stores consistent property values")
    func gpuProfilePropertyStorage() async {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let profile = await GPUProfiler.profile(circuit, precisionPolicy: .fast)

        // Verify properties are consistent with each other
        #expect(profile.cpuTimeMs >= 0.0, "cpuTimeMs should be non-negative")
        #expect(profile.gpuTimeMs >= 0.0, "gpuTimeMs should be non-negative")
        #expect(profile.transferBytes > 0, "transferBytes should be positive")
        #expect(profile.utilizationPercent >= 0.0, "utilizationPercent should be non-negative")

        // Verify GPU/CPU time consistency with didUseGPU flag
        if profile.didUseGPU {
            #expect(profile.gpuTimeMs > 0.0, "GPU time should be positive when GPU is used")
        } else {
            #expect(profile.gpuTimeMs == 0.0, "GPU time should be zero when CPU-only")
            #expect(profile.utilizationPercent == 0.0, "Utilization should be zero when CPU-only")
        }
    }

    @Test("GPUProfile Equatable conformance")
    func gpuProfileEquatable() async {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)

        // Get two profiles from different executions
        let profile1 = await GPUProfiler.profile(circuit, precisionPolicy: .accurate)
        let profile2 = await GPUProfiler.profile(circuit, precisionPolicy: .accurate)

        // Both should be CPU-only with same transfer bytes (timing may differ)
        #expect(profile1.didUseGPU == profile2.didUseGPU, "Same circuit/policy should yield same GPU usage")
        #expect(profile1.transferBytes == profile2.transferBytes, "Same circuit should yield same transfer bytes")
    }

    @Test("GPUProfiler.profile returns non-negative timing values")
    func profilerReturnsNonNegativeTimings() async {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let profile = await GPUProfiler.profile(circuit, precisionPolicy: .fast)

        #expect(profile.cpuTimeMs >= 0.0, "CPU time should be non-negative")
        #expect(profile.gpuTimeMs >= 0.0, "GPU time should be non-negative")
        #expect(profile.transferBytes >= 0, "Transfer bytes should be non-negative")
        #expect(profile.utilizationPercent >= 0.0, "Utilization percent should be non-negative")
    }

    @Test("GPUProfiler.profile with accurate policy uses CPU only")
    func profilerAccuratePolicyUsesCPU() async {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)

        let profile = await GPUProfiler.profile(circuit, precisionPolicy: .accurate)

        #expect(profile.didUseGPU == false, "Accurate policy should force CPU-only execution")
        #expect(profile.gpuTimeMs == 0.0, "GPU time should be zero when CPU-only")
        #expect(profile.utilizationPercent == 0.0, "GPU utilization should be zero when CPU-only")
    }

    @Test("GPUProfiler.profile calculates transfer bytes from state space")
    func profilerCalculatesTransferBytes() async {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)

        let profile = await GPUProfiler.profile(circuit, precisionPolicy: .fast)

        let expectedStateSpaceSize = 1 << 3
        let expectedTransferBytes = 16 * expectedStateSpaceSize * 2

        #expect(profile.transferBytes == expectedTransferBytes, "Transfer bytes should be 16 * 2^qubits * 2")
    }

    @Test("GPUProfiler.profile with small qubit count uses CPU")
    func profilerSmallCircuitUsesCPU() async {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)

        let profile = await GPUProfiler.profile(circuit, precisionPolicy: .fast)

        #expect(profile.didUseGPU == false, "Small circuits below GPU threshold should use CPU")
    }
}
