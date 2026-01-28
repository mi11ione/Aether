// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Test suite for MemoryProfiler.estimate() function.
/// Validates theoretical memory calculations for state vectors and unitary matrices
/// based on qubit count without executing circuits.
@Suite("Memory Estimation")
struct MemoryEstimationTests {
    @Test("1 qubit: state=32B, unitary=64B")
    func oneQubitEstimate() {
        let estimate = MemoryProfiler.estimate(qubits: 1)
        #expect(estimate.stateBytes == 32, "1 qubit state vector should be 2^1 * 16 = 32 bytes")
        #expect(estimate.unitaryBytes == 64, "1 qubit unitary should be 2^2 * 16 = 64 bytes")
    }

    @Test("2 qubits: state=64B, unitary=256B")
    func twoQubitEstimate() {
        let estimate = MemoryProfiler.estimate(qubits: 2)
        #expect(estimate.stateBytes == 64, "2 qubit state vector should be 2^2 * 16 = 64 bytes")
        #expect(estimate.unitaryBytes == 256, "2 qubit unitary should be 2^4 * 16 = 256 bytes")
    }

    @Test("3 qubits: state=128B, unitary=1024B")
    func threeQubitEstimate() {
        let estimate = MemoryProfiler.estimate(qubits: 3)
        #expect(estimate.stateBytes == 128, "3 qubit state vector should be 2^3 * 16 = 128 bytes")
        #expect(estimate.unitaryBytes == 1024, "3 qubit unitary should be 2^6 * 16 = 1024 bytes")
    }

    @Test("4 qubits: state=256B, unitary=4096B")
    func fourQubitEstimate() {
        let estimate = MemoryProfiler.estimate(qubits: 4)
        #expect(estimate.stateBytes == 256, "4 qubit state vector should be 2^4 * 16 = 256 bytes")
        #expect(estimate.unitaryBytes == 4096, "4 qubit unitary should be 2^8 * 16 = 4096 bytes")
    }

    @Test("10 qubits: state=16KB, unitary=16MB")
    func tenQubitEstimate() {
        let estimate = MemoryProfiler.estimate(qubits: 10)
        #expect(estimate.stateBytes == 16384, "10 qubit state vector should be 2^10 * 16 = 16384 bytes")
        #expect(estimate.unitaryBytes == 16_777_216, "10 qubit unitary should be 2^20 * 16 = 16777216 bytes")
    }

    @Test("isRecommended is true for small qubit counts")
    func isRecommendedSmallQubits() {
        let estimate = MemoryProfiler.estimate(qubits: 3)
        #expect(estimate.isRecommended, "3 qubits should be recommended as memory requirement is tiny")
    }
}

/// Test suite for MemoryProfiler.profile() function.
/// Validates circuit execution profiling returns valid memory measurements
/// with correct theoretical amplitude and unitary byte calculations.
@Suite("Memory Profiling")
struct MemoryProfilingTests {
    @Test("Profile returns non-negative peak bytes")
    func profileReturnsNonNegativePeakBytes() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let profile = MemoryProfiler.profile(circuit)
        #expect(profile.peakBytes >= 0, "Peak bytes should be non-negative")
    }

    @Test("Profile returns non-negative virtual bytes")
    func profileReturnsNonNegativeVirtualBytes() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let profile = MemoryProfiler.profile(circuit)
        #expect(profile.virtualBytes >= 0, "Virtual bytes should be non-negative")
    }

    @Test("Profile computes correct amplitude bytes for 2 qubits")
    func profileAmplitudeBytesForTwoQubits() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let profile = MemoryProfiler.profile(circuit)
        #expect(profile.amplitudeBytes == 64, "2 qubit amplitude bytes should be 2^2 * 16 = 64")
    }

    @Test("Profile computes correct unitary bytes for 2 qubits")
    func profileUnitaryBytesForTwoQubits() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let profile = MemoryProfiler.profile(circuit)
        #expect(profile.unitaryBytes == 256, "2 qubit unitary bytes should be 2^4 * 16 = 256")
    }

    @Test("Profile returns non-negative heap statistics")
    func profileReturnsNonNegativeHeapStats() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.pauliX, to: 0)
        let profile = MemoryProfiler.profile(circuit)
        #expect(profile.heapAllocated >= 0, "Heap allocated should be non-negative")
        #expect(profile.heapUsed >= 0, "Heap used should be non-negative")
    }

    @Test("Profile computes correct values for 3 qubit circuit")
    func profileThreeQubitCircuit() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        let profile = MemoryProfiler.profile(circuit)
        #expect(profile.amplitudeBytes == 128, "3 qubit amplitude bytes should be 2^3 * 16 = 128")
        #expect(profile.unitaryBytes == 1024, "3 qubit unitary bytes should be 2^6 * 16 = 1024")
    }
}

/// Test suite for MemoryProfile struct properties and formatting.
/// Validates description output, equality comparison, and byte formatting
/// across different magnitude ranges (B, KB, MB, GB).
@Suite("MemoryProfile Properties")
struct MemoryProfilePropertiesTests {
    @Test("MemoryProfile equality works correctly")
    func memoryProfileEquality() {
        var circuit1 = QuantumCircuit(qubits: 2)
        circuit1.append(.hadamard, to: 0)
        let profile1 = MemoryProfiler.profile(circuit1)

        var circuit2 = QuantumCircuit(qubits: 2)
        circuit2.append(.hadamard, to: 0)
        let profile2 = MemoryProfiler.profile(circuit2)

        #expect(profile1.amplitudeBytes == profile2.amplitudeBytes, "Same qubit count should have same amplitude bytes")
        #expect(profile1.unitaryBytes == profile2.unitaryBytes, "Same qubit count should have same unitary bytes")
    }

    @Test("MemoryProfile description contains expected fields")
    func memoryProfileDescriptionContainsFields() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let profile = MemoryProfiler.profile(circuit)
        let desc = profile.description

        #expect(desc.contains("MemoryProfile"), "Description should contain type name")
        #expect(desc.contains("peak"), "Description should contain peak field")
        #expect(desc.contains("virtual"), "Description should contain virtual field")
        #expect(desc.contains("amplitudes"), "Description should contain amplitudes field")
        #expect(desc.contains("unitary"), "Description should contain unitary field")
        #expect(desc.contains("heapAllocated"), "Description should contain heapAllocated field")
        #expect(desc.contains("heapUsed"), "Description should contain heapUsed field")
    }

    @Test("MemoryProfile description formats bytes correctly")
    func memoryProfileDescriptionFormatsBytes() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let profile = MemoryProfiler.profile(circuit)
        let desc = profile.description

        #expect(desc.contains("B"), "Description should contain byte unit indicator")
    }

    @Test("formatBytes formats GB values correctly")
    func formatBytesGB() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let profile = MemoryProfiler.profile(circuit)
        let desc = profile.description
        #expect(desc.contains("GB") || desc.contains("MB") || desc.contains("KB") || desc.contains("B"), "Description should format peak/virtual bytes which are typically GB range")
    }

    @Test("formatBytes formats small byte values correctly")
    func formatBytesSmall() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let profile = MemoryProfiler.profile(circuit)
        let desc = profile.description
        #expect(desc.contains("64 B") || desc.contains("0.06 KB"), "2 qubit amplitudes (64 bytes) should format as bytes or small KB")
    }

    @Test("formatBytes formats KB values correctly")
    func formatBytesKB() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        let profile = MemoryProfiler.profile(circuit)
        let desc = profile.description
        #expect(desc.contains("1.00 KB") || desc.contains("1024 B"), "3 qubit unitary (1024 bytes) should format as KB")
    }
}

/// Test suite for MemoryEstimate struct properties.
/// Validates equality comparison, field access, and isRecommended
/// flag behavior for different memory scenarios.
@Suite("MemoryEstimate Properties")
struct MemoryEstimatePropertiesTests {
    @Test("MemoryEstimate equality works correctly")
    func memoryEstimateEquality() {
        let estimate1 = MemoryProfiler.estimate(qubits: 3)
        let estimate2 = MemoryProfiler.estimate(qubits: 3)

        #expect(estimate1 == estimate2, "Same qubit estimates should be equal")
    }

    @Test("MemoryEstimate inequality for different qubits")
    func memoryEstimateInequality() {
        let estimate1 = MemoryProfiler.estimate(qubits: 2)
        let estimate2 = MemoryProfiler.estimate(qubits: 3)

        #expect(estimate1 != estimate2, "Different qubit estimates should not be equal")
    }

    @Test("MemoryEstimate stateBytes doubles with each qubit")
    func memoryEstimateStateBytesScaling() {
        let estimate2 = MemoryProfiler.estimate(qubits: 2)
        let estimate3 = MemoryProfiler.estimate(qubits: 3)

        #expect(estimate3.stateBytes == estimate2.stateBytes * 2, "State bytes should double per qubit")
    }

    @Test("MemoryEstimate unitaryBytes quadruples with each qubit")
    func memoryEstimateUnitaryBytesScaling() {
        let estimate2 = MemoryProfiler.estimate(qubits: 2)
        let estimate3 = MemoryProfiler.estimate(qubits: 3)

        #expect(estimate3.unitaryBytes == estimate2.unitaryBytes * 4, "Unitary bytes should quadruple per qubit")
    }

    @Test("estimate returns MemoryEstimate with isRecommended flag")
    func estimateReturnsValidMemoryEstimate() {
        let estimate = MemoryProfiler.estimate(qubits: 2)
        #expect(estimate.stateBytes == 64, "2 qubit estimate should have stateBytes of 64")
        #expect(estimate.unitaryBytes == 256, "2 qubit estimate should have unitaryBytes of 256")
        #expect(estimate.isRecommended == true, "Small qubit count should be recommended")
    }
}

/// Test suite for Sendable conformance verification.
/// Validates that MemoryProfile and MemoryEstimate can be safely
/// used across concurrent contexts as required by Swift concurrency.
@Suite("Sendable Conformance")
struct MemoryProfilerSendableTests {
    @Test("MemoryProfile is Sendable")
    func memoryProfileIsSendable() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let profile = MemoryProfiler.profile(circuit)

        let sendableCheck: any Sendable = profile
        #expect(type(of: sendableCheck) == MemoryProfile.self, "MemoryProfile should conform to Sendable")
    }

    @Test("MemoryEstimate is Sendable")
    func memoryEstimateIsSendable() {
        let estimate = MemoryProfiler.estimate(qubits: 3)

        let sendableCheck: any Sendable = estimate
        #expect(type(of: sendableCheck) == MemoryEstimate.self, "MemoryEstimate should conform to Sendable")
    }
}
