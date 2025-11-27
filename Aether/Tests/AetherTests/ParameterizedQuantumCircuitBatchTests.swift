// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Test suite for batched parameter binding extensions.
/// Validates batch circuit binding, gradient parameter generation, and grid search
/// utilities for efficient VQE and QAOA workflows.
@Suite("ParameterizedQuantumCircuit Batch Extensions")
struct ParameterizedQuantumCircuitBatchTests {
    @Test("Bind empty batch returns empty array")
    func bindEmptyBatch() {
        let circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let results = circuit.bindBatch(parameterVectors: [])

        #expect(results.isEmpty)
    }

    @Test("Bind single parameter set")
    func bindSingleParameterSet() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")
        circuit.append(gate: .rotationY(theta: .parameter(theta)), qubit: 0)

        let parameterSets = [[0.5]]
        let results = circuit.bindBatch(parameterVectors: parameterSets)

        #expect(results.count == 1)
        #expect(results[0].numQubits == 1)
    }

    @Test("Bind multiple parameter sets")
    func bindMultipleParameterSets() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), qubit: 0)
        circuit.append(gate: .rotationZ(theta: .parameter(phi)), qubit: 1)

        let parameterSets: [[Double]] = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]

        let results = circuit.bindBatch(parameterVectors: parameterSets)

        #expect(results.count == 3)
        #expect(results[0].numQubits == 2)
        #expect(results[1].numQubits == 2)
        #expect(results[2].numQubits == 2)
    }

    @Test("Generate gradient parameter vectors for single parameter")
    func generateGradientSingleParameter() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")
        circuit.append(gate: .rotationY(theta: .parameter(theta)), qubit: 0)

        let baseParams = [0.5]
        let (plus, minus) = circuit.generateGradientParameterVectors(baseParameters: baseParams)

        #expect(plus.count == 1)
        #expect(minus.count == 1)
        #expect(abs(plus[0][0] - (0.5 + .pi / 2)) < 1e-10)
        #expect(abs(minus[0][0] - (0.5 - .pi / 2)) < 1e-10)
    }

    @Test("Generate gradient parameter vectors for multiple parameters")
    func generateGradientMultipleParameters() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), qubit: 0)
        circuit.append(gate: .rotationZ(theta: .parameter(phi)), qubit: 1)

        let baseParams: [Double] = [0.1, 0.2]
        let (plus, minus) = circuit.generateGradientParameterVectors(baseParameters: baseParams)

        #expect(plus.count == 2)
        #expect(minus.count == 2)

        #expect(abs(plus[0][0] - (0.1 + .pi / 2)) < 1e-10)
        #expect(abs(plus[0][1] - 0.2) < 1e-10)

        #expect(abs(plus[1][0] - 0.1) < 1e-10)
        #expect(abs(plus[1][1] - (0.2 + .pi / 2)) < 1e-10)

        #expect(abs(minus[0][0] - (0.1 - .pi / 2)) < 1e-10)
        #expect(abs(minus[0][1] - 0.2) < 1e-10)

        #expect(abs(minus[1][0] - 0.1) < 1e-10)
        #expect(abs(minus[1][1] - (0.2 - .pi / 2)) < 1e-10)
    }

    @Test("Generate gradient with custom shift")
    func generateGradientCustomShift() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")
        circuit.append(gate: .rotationY(theta: .parameter(theta)), qubit: 0)

        let baseParams = [1.0]
        let customShift = Double.pi / 4
        let (plus, minus) = circuit.generateGradientParameterVectors(
            baseParameters: baseParams,
            shift: customShift
        )

        #expect(abs(plus[0][0] - (1.0 + customShift)) < 1e-10)
        #expect(abs(minus[0][0] - (1.0 - customShift)) < 1e-10)
    }

    @Test("Generate grid search vectors for single parameter")
    func generateGridSearchSingleParameter() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")
        circuit.append(gate: .rotationY(theta: .parameter(theta)), qubit: 0)

        let range = [0.0, 0.5, 1.0]
        let vectors = circuit.generateGridSearchVectors(ranges: [range])

        #expect(vectors.count == 3)
        #expect(vectors[0][0] == 0.0)
        #expect(vectors[1][0] == 0.5)
        #expect(vectors[2][0] == 1.0)
    }

    @Test("Generate grid search vectors for two parameters")
    func generateGridSearchTwoParameters() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), qubit: 0)
        circuit.append(gate: .rotationZ(theta: .parameter(phi)), qubit: 1)

        let range1 = [0.0, 1.0]
        let range2 = [0.0, 0.5, 1.0]
        let vectors = circuit.generateGridSearchVectors(ranges: [range1, range2])

        #expect(vectors.count == 6)

        #expect(vectors[0] == [0.0, 0.0])
        #expect(vectors[1] == [0.0, 0.5])
        #expect(vectors[2] == [0.0, 1.0])
        #expect(vectors[3] == [1.0, 0.0])
        #expect(vectors[4] == [1.0, 0.5])
        #expect(vectors[5] == [1.0, 1.0])
    }

    @Test("Generate grid search with three parameters")
    func generateGridSearchThreeParameters() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 3)
        for i in 0 ..< 3 {
            let param = Parameter(name: "theta_\(i)")
            circuit.append(gate: .rotationY(theta: .parameter(param)), qubit: i)
        }

        let range1 = [0.0, 1.0]
        let range2 = [0.0, 1.0]
        let range3 = [0.0, 1.0]
        let vectors = circuit.generateGridSearchVectors(ranges: [range1, range2, range3])

        #expect(vectors.count == 8)

        #expect(vectors[0] == [0.0, 0.0, 0.0])
        #expect(vectors[7] == [1.0, 1.0, 1.0])
    }

    @Test("Batch bind with gradient vectors")
    func batchBindWithGradient() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), qubit: 0)
        circuit.append(gate: .rotationZ(theta: .parameter(phi)), qubit: 1)

        let baseParams: [Double] = [0.1, 0.2]
        let (plus, minus) = circuit.generateGradientParameterVectors(baseParameters: baseParams)

        let plusCircuits = circuit.bindBatch(parameterVectors: plus)
        let minusCircuits = circuit.bindBatch(parameterVectors: minus)

        #expect(plusCircuits.count == 2)
        #expect(minusCircuits.count == 2)
    }

    @Test("Batch bind with grid search vectors")
    func batchBindWithGridSearch() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), qubit: 0)
        circuit.append(gate: .rotationZ(theta: .parameter(phi)), qubit: 1)

        let range1 = stride(from: 0.0, through: .pi, by: .pi / 2)
        let range2 = stride(from: 0.0, through: .pi, by: .pi / 2)

        let vectors = circuit.generateGridSearchVectors(
            ranges: [Array(range1), Array(range2)]
        )

        let circuits = circuit.bindBatch(parameterVectors: vectors)

        #expect(circuits.count == 9)
    }

    @Test("Complete VQE gradient workflow")
    func completeVQEGradientWorkflow() {
        let ansatz = HardwareEfficientAnsatz.create(numQubits: 2, depth: 1)
        let baseParams: [Double] = [0.1, 0.2]

        let (plus, minus) = ansatz.generateGradientParameterVectors(baseParameters: baseParams)

        let plusCircuits = ansatz.bindBatch(parameterVectors: plus)
        let minusCircuits = ansatz.bindBatch(parameterVectors: minus)

        #expect(plusCircuits.count == 2)
        #expect(minusCircuits.count == 2)

        for circuit in plusCircuits {
            let state = circuit.execute()
            #expect(state.numQubits == 2)
        }
    }

    @Test("Complete QAOA grid search workflow")
    func completeQAOAGridSearchWorkflow() {
        var qaoa = ParameterizedQuantumCircuit(numQubits: 3)
        let gamma = Parameter(name: "gamma")
        let beta = Parameter(name: "beta")

        for i in 0 ..< 3 {
            qaoa.append(gate: .concrete(.hadamard), qubit: i)
        }

        qaoa.append(gate: .rotationZ(theta: .parameter(gamma)), qubit: 0)
        qaoa.append(gate: .rotationX(theta: .parameter(beta)), qubit: 0)

        let gammaRange = stride(from: 0.0, to: .pi, by: .pi / 4)
        let betaRange = stride(from: 0.0, to: .pi, by: .pi / 4)

        let vectors = qaoa.generateGridSearchVectors(
            ranges: [Array(gammaRange), Array(betaRange)]
        )

        let circuits = qaoa.bindBatch(parameterVectors: vectors)

        #expect(vectors.count == 16)
        #expect(circuits.count == 16)
    }

    @Test("Gradient vectors have correct length")
    func gradientVectorsCorrectLength() {
        let ansatz = HardwareEfficientAnsatz.create(numQubits: 3, depth: 2)
        let paramCount = ansatz.parameterCount()
        let baseParams = Array(repeating: 0.1, count: paramCount)

        let (plus, minus) = ansatz.generateGradientParameterVectors(baseParameters: baseParams)

        #expect(plus.count == paramCount)
        #expect(minus.count == paramCount)

        for vector in plus {
            #expect(vector.count == paramCount)
        }
        for vector in minus {
            #expect(vector.count == paramCount)
        }
    }

    @Test("Grid search cartesian product correctness")
    func gridSearchCartesianProduct() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), qubit: 0)
        circuit.append(gate: .rotationZ(theta: .parameter(phi)), qubit: 1)

        let range1 = [1.0, 2.0, 3.0]
        let range2 = [4.0, 5.0]

        let vectors = circuit.generateGridSearchVectors(ranges: [range1, range2])

        #expect(vectors.count == 6)

        let expected = [
            [1.0, 4.0],
            [1.0, 5.0],
            [2.0, 4.0],
            [2.0, 5.0],
            [3.0, 4.0],
            [3.0, 5.0],
        ]

        for (index, vector) in vectors.enumerated() {
            #expect(vector == expected[index])
        }
    }

    @Test("Large grid search produces correct count")
    func largeGridSearchCount() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 3)
        for i in 0 ..< 3 {
            let param = Parameter(name: "theta_\(i)")
            circuit.append(gate: .rotationY(theta: .parameter(param)), qubit: i)
        }

        let range = stride(from: 0.0, through: .pi, by: .pi / 5)
        let rangeArray = Array(range)

        let vectors = circuit.generateGridSearchVectors(
            ranges: [rangeArray, rangeArray, rangeArray]
        )

        let expectedCount = rangeArray.count * rangeArray.count * rangeArray.count
        #expect(vectors.count == expectedCount)
    }

    @Test("Gradient generation preserves other parameters")
    func gradientPreservesOtherParams() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 4)
        for i in 0 ..< 4 {
            let param = Parameter(name: "theta_\(i)")
            circuit.append(gate: .rotationY(theta: .parameter(param)), qubit: i)
        }

        let baseParams: [Double] = [0.1, 0.2, 0.3, 0.4]
        let (plus, minus) = circuit.generateGradientParameterVectors(baseParameters: baseParams)

        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                if j == i {
                    #expect(abs(plus[i][j] - (baseParams[j] + .pi / 2)) < 1e-10)
                    #expect(abs(minus[i][j] - (baseParams[j] - .pi / 2)) < 1e-10)
                } else {
                    #expect(plus[i][j] == baseParams[j])
                    #expect(minus[i][j] == baseParams[j])
                }
            }
        }
    }

    @Test("Batch bind with hardware efficient ansatz")
    func batchBindHardwareEfficientAnsatz() {
        let ansatz = HardwareEfficientAnsatz.create(numQubits: 4, depth: 2)
        let paramCount = ansatz.parameterCount()

        var parameterSets: [[Double]] = []
        for i in 0 ..< 5 {
            let params = Array(repeating: Double(i) * 0.1, count: paramCount)
            parameterSets.append(params)
        }

        let circuits = ansatz.bindBatch(parameterVectors: parameterSets)

        #expect(circuits.count == 5)

        for circuit in circuits {
            #expect(circuit.numQubits == 4)
        }
    }

    @Test("Empty parameter circuit batch binding")
    func emptyParameterCircuit() {
        let circuit = ParameterizedQuantumCircuit(numQubits: 2)

        let vectors: [[Double]] = [[], [], []]
        let results = circuit.bindBatch(parameterVectors: vectors)

        #expect(results.count == 3)
    }

    @Test("Single range grid search")
    func singleRangeGridSearch() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")
        circuit.append(gate: .rotationY(theta: .parameter(theta)), qubit: 0)

        let singleRange = [0.0]
        let vectors = circuit.generateGridSearchVectors(ranges: [singleRange])

        #expect(vectors.count == 1)
        #expect(vectors[0] == [0.0])
    }
}
