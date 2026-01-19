// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for batched circuit evaluation using Metal Performance Shaders.
/// Validates batch state evolution, expectation value computation, CPU fallback,
/// and memory-aware chunking for VQE gradient acceleration.
@Suite("MPSBatchEvaluator")
struct MPSBatchEvaluatorTests {
    @Test("Evaluator initializes successfully")
    func initializeEvaluator() async {
        let evaluator = MPSBatchEvaluator()
        let stats = await evaluator.statistics

        #expect(stats.maxBatchSize > 0)
    }

    @Test("Single circuit batch evaluation")
    func singleCircuitBatch() async {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)

        let unitary = CircuitUnitary.unitary(for: circuit)
        let initialState = QuantumState(qubits: 2)

        let evaluator = MPSBatchEvaluator()
        let results = await evaluator.evaluate(
            batch: [unitary],
            from: initialState,
        )

        #expect(results.count == 1)
        #expect(results[0].qubits == 2)

        let expected = circuit.execute()
        for i in 0 ..< 4 {
            #expect(abs(results[0].amplitudes[i].real - expected.amplitudes[i].real) < 1e-5)
            #expect(abs(results[0].amplitudes[i].imaginary - expected.amplitudes[i].imaginary) < 1e-5)
        }
    }

    @Test("Multiple circuits batch evaluation")
    func multipleCircuitsBatch() async {
        var circuit1 = QuantumCircuit(qubits: 2)
        circuit1.append(.hadamard, to: 0)

        var circuit2 = QuantumCircuit(qubits: 2)
        circuit2.append(.pauliX, to: 0)

        var circuit3 = QuantumCircuit(qubits: 2)
        circuit3.append(.pauliZ, to: 1)

        let unitaries = [
            CircuitUnitary.unitary(for: circuit1),
            CircuitUnitary.unitary(for: circuit2),
            CircuitUnitary.unitary(for: circuit3),
        ]

        let initialState = QuantumState(qubits: 2)
        let evaluator = MPSBatchEvaluator()
        let results = await evaluator.evaluate(
            batch: unitaries,
            from: initialState,
        )

        #expect(results.count == 3)

        let expected1 = circuit1.execute()
        let expected2 = circuit2.execute()
        let expected3 = circuit3.execute()

        for i in 0 ..< 4 {
            #expect(abs(results[0].amplitudes[i].real - expected1.amplitudes[i].real) < 1e-5)
            #expect(abs(results[1].amplitudes[i].real - expected2.amplitudes[i].real) < 1e-5)
            #expect(abs(results[2].amplitudes[i].real - expected3.amplitudes[i].real) < 1e-5)
        }
    }

    @Test("Batch evaluation with expectation values using Observable")
    func batchExpectationValuesObservable() async {
        let hamiltonian = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(0)),
        )

        let circuit1 = QuantumCircuit(qubits: 2)
        var circuit2 = QuantumCircuit(qubits: 2)
        circuit2.append(.pauliX, to: 0)

        let unitaries = [
            CircuitUnitary.unitary(for: circuit1),
            CircuitUnitary.unitary(for: circuit2),
        ]

        let initialState = QuantumState(qubits: 2)
        let evaluator = MPSBatchEvaluator()
        let energies = await evaluator.expectationValues(
            for: unitaries,
            from: initialState,
            observable: hamiltonian,
        )

        #expect(energies.count == 2)

        let expected1 = hamiltonian.expectationValue(of: circuit1.execute())
        let expected2 = hamiltonian.expectationValue(of: circuit2.execute())

        #expect(abs(energies[0] - expected1) < 1e-5)
        #expect(abs(energies[1] - expected2) < 1e-5)
    }

    @Test("Batch evaluation with SparseHamiltonian")
    func batchExpectationValuesSparse() async {
        let observable = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(0)),
        )
        let sparseH = SparseHamiltonian(observable: observable, systemSize: 2)

        let circuit1 = QuantumCircuit(qubits: 2)
        var circuit2 = QuantumCircuit(qubits: 2)
        circuit2.append(.hadamard, to: 0)

        let unitaries = [
            CircuitUnitary.unitary(for: circuit1),
            CircuitUnitary.unitary(for: circuit2),
        ]

        let initialState = QuantumState(qubits: 2)
        let evaluator = MPSBatchEvaluator()
        let energies = await evaluator.expectationValues(
            for: unitaries,
            from: initialState,
            sparse: sparseH,
        )

        #expect(energies.count == 2)

        let expected1 = await sparseH.expectationValue(of: circuit1.execute())
        let expected2 = await sparseH.expectationValue(of: circuit2.execute())

        #expect(abs(energies[0] - expected1) < 1e-5)
        #expect(abs(energies[1] - expected2) < 1e-5)
    }

    @Test("Three-qubit circuit batch evaluation")
    func threeQubitBatch() async {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let unitary = CircuitUnitary.unitary(for: circuit)
        let initialState = QuantumState(qubits: 3)

        let evaluator = MPSBatchEvaluator()
        let results = await evaluator.evaluate(
            batch: [unitary],
            from: initialState,
        )

        #expect(results.count == 1)
        #expect(results[0].qubits == 3)

        let expected = circuit.execute()
        for i in 0 ..< 8 {
            #expect(abs(results[0].amplitudes[i].real - expected.amplitudes[i].real) < 1e-5)
        }
    }

    @Test("Batch size query returns reasonable value")
    func batchSizeQuery() async {
        let evaluator = MPSBatchEvaluator()
        let maxBatch8 = await evaluator.maxBatchSize(for: 8)
        let maxBatch10 = await evaluator.maxBatchSize(for: 10)

        #expect(maxBatch8 > 0)
        #expect(maxBatch10 > 0)
        #expect(maxBatch8 >= maxBatch10)
    }

    @Test("Statistics provide device information")
    func statistics() async {
        let evaluator = MPSBatchEvaluator()
        let stats = await evaluator.statistics

        #expect(!stats.deviceName.isEmpty)
        #expect(stats.maxBatchSize > 0)
    }

    @Test("VQE gradient parameter set batch evaluation")
    func vqeGradientBatch() async {
        let ansatz = HardwareEfficientAnsatz(qubits: 2, depth: 1)
        let baseParams: [Double] = [0.1, 0.2]

        let (plusVectors, minusVectors) = ansatz.circuit.gradientVectors(base: baseParams)

        var allCircuits: [QuantumCircuit] = []
        for params in plusVectors {
            allCircuits.append(ansatz.circuit.bound(with: params))
        }
        for params in minusVectors {
            allCircuits.append(ansatz.circuit.bound(with: params))
        }

        let unitaries = allCircuits.map { CircuitUnitary.unitary(for: $0) }

        let hamiltonian = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(0)),
        )

        let evaluator = MPSBatchEvaluator()
        let energies = await evaluator.expectationValues(
            for: unitaries,
            from: QuantumState(qubits: 2),
            observable: hamiltonian,
        )

        #expect(energies.count == 4)
    }

    @Test("Bell state preparation batch")
    func bellStateBatch() async {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let unitary = CircuitUnitary.unitary(for: circuit)
        let initialState = QuantumState(qubits: 2)

        let evaluator = MPSBatchEvaluator()
        let results = await evaluator.evaluate(
            batch: [unitary],
            from: initialState,
        )

        let inv_sqrt2 = 1.0 / sqrt(2.0)
        #expect(abs(results[0].amplitudes[0].real - inv_sqrt2) < 1e-5)
        #expect(abs(results[0].amplitudes[3].real - inv_sqrt2) < 1e-5)
        #expect(abs(results[0].amplitudes[1].magnitude) < 1e-5)
        #expect(abs(results[0].amplitudes[2].magnitude) < 1e-5)
    }

    @Test("Rotation sweep batch evaluation")
    func rotationSweep() async {
        let angles = stride(from: 0.0, through: .pi, by: .pi / 4)
        var circuits: [QuantumCircuit] = []

        for angle in angles {
            var circuit = QuantumCircuit(qubits: 2)
            circuit.append(.rotationY(angle), to: 0)
            circuits.append(circuit)
        }

        let unitaries = circuits.map { CircuitUnitary.unitary(for: $0) }
        let evaluator = MPSBatchEvaluator()
        let results = await evaluator.evaluate(
            batch: unitaries,
            from: QuantumState(qubits: 2),
        )

        #expect(results.count == 5)

        for (index, circuit) in circuits.enumerated() {
            let expected = circuit.execute()
            for i in 0 ..< 4 {
                #expect(abs(results[index].amplitudes[i].real - expected.amplitudes[i].real) < 1e-5)
            }
        }
    }

    @Test("Four-qubit batch evaluation")
    func fourQubitBatch() async {
        var circuit = QuantumCircuit(qubits: 4)
        circuit.append(.hadamard, to: 0)

        let unitary = CircuitUnitary.unitary(for: circuit)
        let initialState = QuantumState(qubits: 4)

        let evaluator = MPSBatchEvaluator()
        let results = await evaluator.evaluate(
            batch: [unitary],
            from: initialState,
        )

        #expect(results.count == 1)
        #expect(results[0].qubits == 4)
    }

    @Test("Batch with identity circuits")
    func identityCircuits() async {
        let circuit1 = QuantumCircuit(qubits: 2)
        let circuit2 = QuantumCircuit(qubits: 2)

        let unitaries = [
            CircuitUnitary.unitary(for: circuit1),
            CircuitUnitary.unitary(for: circuit2),
        ]

        let initialState = QuantumState(qubits: 2)
        let evaluator = MPSBatchEvaluator()
        let results = await evaluator.evaluate(
            batch: unitaries,
            from: initialState,
        )

        #expect(results.count == 2)

        for result in results {
            #expect(abs(result.amplitudes[0].real - 1.0) < 1e-5)
            #expect(abs(result.amplitudes[1].magnitude) < 1e-5)
        }
    }

    @Test("Batch evaluation preserves normalization")
    func preservesNormalization() async {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.rotationY(.pi / 4), to: 1)

        let unitary = CircuitUnitary.unitary(for: circuit)
        let initialState = QuantumState(qubits: 2)

        let evaluator = MPSBatchEvaluator()
        let results = await evaluator.evaluate(
            batch: [unitary],
            from: initialState,
        )

        var norm = 0.0
        for amplitude in results[0].amplitudes {
            norm += amplitude.magnitude * amplitude.magnitude
        }

        #expect(abs(norm - 1.0) < 1e-5)
    }

    @Test("Multiple Hamiltonian measurements in batch")
    func multipleHamiltonianMeasurements() async {
        let hamiltonian1 = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(0)),
        )

        let hamiltonian2 = Observable(
            coefficient: 1.0,
            pauliString: PauliString(.z(1)),
        )

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)

        let unitary = CircuitUnitary.unitary(for: circuit)
        let initialState = QuantumState(qubits: 2)

        let evaluator = MPSBatchEvaluator()

        let energies1 = await evaluator.expectationValues(
            for: [unitary],
            from: initialState,
            observable: hamiltonian1,
        )

        let energies2 = await evaluator.expectationValues(
            for: [unitary],
            from: initialState,
            observable: hamiltonian2,
        )

        #expect(energies1.count == 1)
        #expect(energies2.count == 1)

        let state = circuit.execute()
        let expected1 = hamiltonian1.expectationValue(of: state)
        let expected2 = hamiltonian2.expectationValue(of: state)

        #expect(abs(energies1[0] - expected1) < 1e-5)
        #expect(abs(energies2[0] - expected2) < 1e-5)
    }

    @Test("Batch with Toffoli gate")
    func batchWithToffoli() async {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.toffoli, to: [0, 1, 2])

        let unitary = CircuitUnitary.unitary(for: circuit)
        let initialState = QuantumState(qubits: 3)

        let evaluator = MPSBatchEvaluator()
        let results = await evaluator.evaluate(
            batch: [unitary],
            from: initialState,
        )

        let expected = circuit.execute()
        for i in 0 ..< 8 {
            #expect(abs(results[0].amplitudes[i].real - expected.amplitudes[i].real) < 1e-5)
        }
    }

    @Test("Batch evaluation with custom initial state")
    func customInitialState() async {
        var prepCircuit = QuantumCircuit(qubits: 2)
        prepCircuit.append(.pauliX, to: 0)
        let initialState = prepCircuit.execute()

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 1)

        let unitary = CircuitUnitary.unitary(for: circuit)

        let evaluator = MPSBatchEvaluator()
        let results = await evaluator.evaluate(
            batch: [unitary],
            from: initialState,
        )

        prepCircuit.append(.hadamard, to: 1)
        let expected = prepCircuit.execute()

        for i in 0 ..< 4 {
            #expect(abs(results[0].amplitudes[i].real - expected.amplitudes[i].real) < 1e-5)
        }
    }

    @Test("Large angle rotation batch")
    func largeAngleRotation() async {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.rotationX(2 * .pi), to: 0)

        let unitary = CircuitUnitary.unitary(for: circuit)
        let initialState = QuantumState(qubits: 2)

        let evaluator = MPSBatchEvaluator()
        let results = await evaluator.evaluate(
            batch: [unitary],
            from: initialState,
        )

        #expect(abs(results[0].amplitudes[0].real + 1.0) < 1e-5)
    }

    @Test("Batch evaluation correctness vs sequential")
    func batchVsSequential() async {
        var circuits: [QuantumCircuit] = []
        for i in 0 ..< 5 {
            var circuit = QuantumCircuit(qubits: 2)
            circuit.append(.rotationY(Double(i) * .pi / 4), to: 0)
            circuits.append(circuit)
        }

        let unitaries = circuits.map { CircuitUnitary.unitary(for: $0) }
        let initialState = QuantumState(qubits: 2)

        let evaluator = MPSBatchEvaluator()
        let batchResults = await evaluator.evaluate(
            batch: unitaries,
            from: initialState,
        )

        #expect(batchResults.count == 5)

        for (index, circuit) in circuits.enumerated() {
            let sequential = circuit.execute()
            for i in 0 ..< 4 {
                #expect(abs(batchResults[index].amplitudes[i].real - sequential.amplitudes[i].real) < 1e-5)
                #expect(abs(batchResults[index].amplitudes[i].imaginary - sequential.amplitudes[i].imaginary) < 1e-5)
            }
        }
    }

    @Test("Batch exceeding 1000 circuits triggers chunking")
    func largeAutomaticChunking() async {
        let evaluator = MPSBatchEvaluator()
        let batchSize = 1050

        var circuits: [QuantumCircuit] = []
        circuits.reserveCapacity(batchSize)

        for i in 0 ..< batchSize {
            var circuit = QuantumCircuit(qubits: 2)
            circuit.append(.rotationY(Double(i) * 0.01), to: 0)
            circuits.append(circuit)
        }

        let unitaries = circuits.map { CircuitUnitary.unitary(for: $0) }
        let initialState = QuantumState(qubits: 2)

        let results = await evaluator.evaluate(
            batch: unitaries,
            from: initialState,
        )

        #expect(results.count == batchSize)

        let testIndices = [0, 200, 500, 800, 1049]
        for index in testIndices {
            let expected = circuits[index].execute()
            for i in 0 ..< 4 {
                #expect(abs(results[index].amplitudes[i].real - expected.amplitudes[i].real) < 1e-5)
                #expect(abs(results[index].amplitudes[i].imaginary - expected.amplitudes[i].imaginary) < 1e-5)
            }
        }
    }

    @Test("BatchEvaluatorStatistics description formats correctly")
    func statisticsDescription() async {
        let evaluator = MPSBatchEvaluator()
        let stats = await evaluator.statistics
        let description = stats.description

        #expect(description.contains("Metal Available"), "Should show Metal availability")
        #expect(description.contains("Device"), "Should show device name")
        #expect(description.contains("Max Batch Size"), "Should show max batch size")
        #expect(description.contains(stats.deviceName), "Should include actual device name")
        #expect(description.contains("\(stats.maxBatchSize)"), "Should include actual batch size")
    }

    @Test("BatchEvaluatorStatistics shows CPU when Metal unavailable")
    func statisticsCPUDeviceName() {
        let stats = BatchEvaluatorStatistics(
            isMetalAvailable: false,
            maxBatchSize: 100,
            deviceName: "CPU",
            precisionPolicy: .fast,
        )

        #expect(stats.deviceName == "CPU", "Should show CPU when Metal unavailable")
        #expect(!stats.isMetalAvailable, "Metal should be marked unavailable")
    }

    @Test("Batch evaluation with many small circuits tests chunking path")
    func manySmallCircuitsChunking() async {
        let evaluator = MPSBatchEvaluator()

        var circuits: [QuantumCircuit] = []
        for i in 0 ..< 25 {
            var circuit = QuantumCircuit(qubits: 2)
            circuit.append(.rotationX(Double(i) * 0.1), to: 0)
            circuit.append(.hadamard, to: 1)
            circuits.append(circuit)
        }

        let unitaries = circuits.map { CircuitUnitary.unitary(for: $0) }
        let initialState = QuantumState(qubits: 2)

        let results = await evaluator.evaluate(
            batch: unitaries,
            from: initialState,
        )

        #expect(results.count == 25, "All circuits should be evaluated")

        for (index, result) in results.enumerated() {
            var sumProbabilities = 0.0
            for amplitude in result.amplitudes {
                sumProbabilities += amplitude.magnitude * amplitude.magnitude
            }
            #expect(
                abs(sumProbabilities - 1.0) < 1e-5,
                "Circuit \(index) should maintain normalization",
            )
        }
    }

    @Test("getMaxBatchSize returns valid values for different qubit counts")
    func maxBatchSizeDifferentQubits() async {
        let evaluator = MPSBatchEvaluator()

        let batch2 = await evaluator.maxBatchSize(for: 2)
        let batch4 = await evaluator.maxBatchSize(for: 4)
        let batch6 = await evaluator.maxBatchSize(for: 6)
        let batch8 = await evaluator.maxBatchSize(for: 8)

        #expect(batch2 > 0, "2-qubit batch size should be positive")
        #expect(batch4 > 0, "4-qubit batch size should be positive")
        #expect(batch6 > 0, "6-qubit batch size should be positive")
        #expect(batch8 > 0, "8-qubit batch size should be positive")

        #expect(batch2 >= batch4, "Smaller circuits should allow larger batches")
        #expect(batch4 >= batch6, "Batch size should decrease with qubit count")
        #expect(batch6 >= batch8, "Batch size monotonically decreases")
    }

    @Test("Evaluator respects maxBatchSize override")
    func maxBatchSizeOverride() async {
        let customLimit = 42
        let evaluator = MPSBatchEvaluator(maxBatchSize: customLimit)
        let stats = await evaluator.statistics

        #expect(
            stats.maxBatchSize == customLimit,
            "Should use provided maxBatchSize override instead of auto-calculated value",
        )
    }

    @Test("Custom maxBatchSize triggers chunking at lower threshold")
    func customMaxBatchSizeChunking() async {
        let evaluator = MPSBatchEvaluator(maxBatchSize: 5)

        var circuits: [QuantumCircuit] = []
        for i in 0 ..< 10 {
            var circuit = QuantumCircuit(qubits: 2)
            circuit.append(.rotationY(Double(i) * 0.1), to: 0)
            circuits.append(circuit)
        }

        let unitaries = circuits.map { CircuitUnitary.unitary(for: $0) }
        let results = await evaluator.evaluate(
            batch: unitaries,
            from: QuantumState(qubits: 2),
        )

        #expect(results.count == 10, "All circuits should be evaluated despite chunking")

        for (index, circuit) in circuits.enumerated() {
            let expected = circuit.execute()
            #expect(
                abs(results[index].amplitudes[0].real - expected.amplitudes[0].real) < 1e-5,
                "Circuit \(index) result should match sequential execution",
            )
        }
    }

    @Test("Precision policy is exposed and matches initialization")
    func precisionPolicyProperty() {
        let fastEvaluator = MPSBatchEvaluator(precisionPolicy: .fast)
        let balancedEvaluator = MPSBatchEvaluator(precisionPolicy: .balanced)
        let accurateEvaluator = MPSBatchEvaluator(precisionPolicy: .accurate)

        #expect(fastEvaluator.precisionPolicy == .fast, "Fast policy should be exposed correctly")
        #expect(balancedEvaluator.precisionPolicy == .balanced, "Balanced policy should be exposed correctly")
        #expect(accurateEvaluator.precisionPolicy == .accurate, "Accurate policy should be exposed correctly")
    }

    @Test("Accurate precision policy forces CPU execution")
    func accuratePolicyForcesCPU() async {
        let evaluator = MPSBatchEvaluator(precisionPolicy: .accurate)

        #expect(!evaluator.isMetalAvailable, "Accurate policy should disable Metal")

        let stats = await evaluator.statistics
        #expect(stats.deviceName == "CPU", "Accurate policy should report CPU device")
        #expect(!stats.isMetalAvailable, "Statistics should reflect Metal unavailable")
    }

    @Test("BatchEvaluatorStatistics includes precision policy")
    func statisticsIncludesPrecisionPolicy() async {
        let evaluator = MPSBatchEvaluator(precisionPolicy: .balanced)
        let stats = await evaluator.statistics

        #expect(stats.precisionPolicy == .balanced, "Statistics should include precision policy")
        #expect(stats.description.contains("Precision Policy"), "Description should show precision policy")
    }
}
