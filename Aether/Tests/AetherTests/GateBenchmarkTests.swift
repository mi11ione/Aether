// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Validates GateBenchmark initialization, measurement, and comparison functionality.
/// Ensures timing statistics are computed correctly with valid ranges.
/// Verifies GateBenchmarkResult properties and description formatting.
@Suite("GateBenchmark")
struct GateBenchmarkTests {
    @Test
    func initializationStoresParameters() {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 2)

        #expect(benchmark.qubits == 2, "qubits should be stored as 2")
        #expect(benchmark.iterations == 5, "iterations should be stored as 5")
        #expect(benchmark.warmupIterations == 2, "warmupIterations should be stored as 2")
    }

    @Test
    func initializationWithDefaults() {
        let benchmark = GateBenchmark(qubits: 3)

        #expect(benchmark.qubits == 3, "qubits should be stored as 3")
        #expect(benchmark.iterations == 100, "iterations should default to 100")
        #expect(benchmark.warmupIterations == 3, "warmupIterations should default to 3")
    }

    @Test
    func initializationAllowsZeroWarmup() {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 0)

        #expect(benchmark.warmupIterations == 0, "warmupIterations of 0 should be allowed")
    }

    @Test
    func measureSingleQubitGateReturnsResult() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 1)
        let result = await benchmark.measure(.hadamard)

        #expect(result.gate == .hadamard, "result gate should match measured gate")
        #expect(result.iterations == 5, "result iterations should match benchmark iterations")
    }

    @Test
    func measureTwoQubitGateReturnsResult() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 1)
        let result = await benchmark.measure(.cnot)

        #expect(result.gate == .cnot, "result gate should match measured gate")
        #expect(result.iterations == 5, "result iterations should match benchmark iterations")
    }

    @Test
    func measureThreeQubitGateReturnsResult() async {
        let benchmark = GateBenchmark(qubits: 3, iterations: 5, warmupIterations: 1)
        let result = await benchmark.measure(.toffoli)

        #expect(result.gate == .toffoli, "result gate should match measured gate")
        #expect(result.iterations == 5, "result iterations should match benchmark iterations")
    }

    @Test
    func measureReturnsNonNegativeTimings() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 1)
        let result = await benchmark.measure(.pauliX)

        #expect(result.meanNs >= 0, "mean timing should be non-negative")
        #expect(result.minNs >= 0, "min timing should be non-negative")
        #expect(result.maxNs >= 0, "max timing should be non-negative")
        #expect(result.stdDevNs >= 0, "standard deviation should be non-negative")
    }

    @Test
    func measureMinNotGreaterThanMean() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 1)
        let result = await benchmark.measure(.pauliY)

        #expect(result.minNs <= result.meanNs, "min should not exceed mean")
    }

    @Test
    func measureMaxNotLessThanMean() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 1)
        let result = await benchmark.measure(.pauliZ)

        #expect(result.maxNs >= result.meanNs, "max should not be less than mean")
    }

    @Test
    func measureMinNotGreaterThanMax() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 1)
        let result = await benchmark.measure(.sGate)

        #expect(result.minNs <= result.maxNs, "min should not exceed max")
    }

    @Test
    func compareEmptyArrayReturnsEmptyResults() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 1)
        let results = await benchmark.compare([])

        #expect(results.isEmpty, "comparing empty array should return empty results")
    }

    @Test
    func compareSingleGateReturnsSingleResult() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 1)
        let results = await benchmark.compare([.tGate])

        #expect(results.count == 1, "comparing single gate should return one result")
        #expect(results[0].gate == .tGate, "result gate should match input gate")
    }

    @Test
    func compareMultipleGatesReturnsResultsInOrder() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 1)
        let gates: [QuantumGate] = [.hadamard, .pauliX, .pauliZ]
        let results = await benchmark.compare(gates)

        #expect(results.count == 3, "comparing three gates should return three results")
        #expect(results[0].gate == .hadamard, "first result should be hadamard")
        #expect(results[1].gate == .pauliX, "second result should be pauliX")
        #expect(results[2].gate == .pauliZ, "third result should be pauliZ")
    }

    @Test
    func compareWithMixedQubitGates() async {
        let benchmark = GateBenchmark(qubits: 3, iterations: 5, warmupIterations: 1)
        let gates: [QuantumGate] = [.hadamard, .cnot, .toffoli]
        let results = await benchmark.compare(gates)

        #expect(results.count == 3, "comparing mixed qubit gates should return three results")
        #expect(results[0].gate == .hadamard, "first result should be single-qubit hadamard")
        #expect(results[1].gate == .cnot, "second result should be two-qubit cnot")
        #expect(results[2].gate == .toffoli, "third result should be three-qubit toffoli")
    }

    @Test
    func resultDescriptionContainsGateName() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 1)
        let result = await benchmark.measure(.hadamard)

        #expect(result.description.contains("hadamard"), "description should contain gate name")
    }

    @Test
    func resultDescriptionContainsMean() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 1)
        let result = await benchmark.measure(.identity)

        #expect(result.description.contains("mean="), "description should contain mean label")
    }

    @Test
    func resultDescriptionContainsMin() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 1)
        let result = await benchmark.measure(.identity)

        #expect(result.description.contains("min="), "description should contain min label")
    }

    @Test
    func resultDescriptionContainsMax() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 1)
        let result = await benchmark.measure(.identity)

        #expect(result.description.contains("max="), "description should contain max label")
    }

    @Test
    func resultDescriptionContainsStddev() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 1)
        let result = await benchmark.measure(.identity)

        #expect(result.description.contains("stddev="), "description should contain stddev label")
    }

    @Test
    func resultDescriptionContainsIterationCount() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 1)
        let result = await benchmark.measure(.identity)

        #expect(result.description.contains("5 iterations"), "description should contain iteration count")
    }

    @Test
    func resultDescriptionContainsNsUnits() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 1)
        let result = await benchmark.measure(.identity)

        #expect(result.description.contains("ns"), "description should contain nanosecond units")
    }

    @Test
    func resultEquatableWithSameValues() {
        let result1 = GateBenchmarkResult(
            gate: .hadamard,
            meanNs: 100.0,
            minNs: 50.0,
            maxNs: 150.0,
            stdDevNs: 25.0,
            iterations: 10,
        )
        let result2 = GateBenchmarkResult(
            gate: .hadamard,
            meanNs: 100.0,
            minNs: 50.0,
            maxNs: 150.0,
            stdDevNs: 25.0,
            iterations: 10,
        )

        #expect(result1 == result2, "results with same values should be equal")
    }

    @Test
    func resultNotEquatableWithDifferentGate() {
        let result1 = GateBenchmarkResult(
            gate: .hadamard,
            meanNs: 100.0,
            minNs: 50.0,
            maxNs: 150.0,
            stdDevNs: 25.0,
            iterations: 10,
        )
        let result2 = GateBenchmarkResult(
            gate: .pauliX,
            meanNs: 100.0,
            minNs: 50.0,
            maxNs: 150.0,
            stdDevNs: 25.0,
            iterations: 10,
        )

        #expect(result1 != result2, "results with different gates should not be equal")
    }

    @Test
    func resultNotEquatableWithDifferentMean() {
        let result1 = GateBenchmarkResult(
            gate: .hadamard,
            meanNs: 100.0,
            minNs: 50.0,
            maxNs: 150.0,
            stdDevNs: 25.0,
            iterations: 10,
        )
        let result2 = GateBenchmarkResult(
            gate: .hadamard,
            meanNs: 200.0,
            minNs: 50.0,
            maxNs: 150.0,
            stdDevNs: 25.0,
            iterations: 10,
        )

        #expect(result1 != result2, "results with different meanNs should not be equal")
    }

    @Test
    func resultNotEquatableWithDifferentIterations() {
        let result1 = GateBenchmarkResult(
            gate: .hadamard,
            meanNs: 100.0,
            minNs: 50.0,
            maxNs: 150.0,
            stdDevNs: 25.0,
            iterations: 10,
        )
        let result2 = GateBenchmarkResult(
            gate: .hadamard,
            meanNs: 100.0,
            minNs: 50.0,
            maxNs: 150.0,
            stdDevNs: 25.0,
            iterations: 20,
        )

        #expect(result1 != result2, "results with different iterations should not be equal")
    }

    @Test
    func measureWithParameterizedGate() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 1)
        let result = await benchmark.measure(.rotationZ(.pi / 4))

        #expect(result.gate == .rotationZ(.pi / 4), "result gate should match parameterized gate")
        #expect(result.iterations == 5, "result iterations should match benchmark iterations")
    }

    @Test
    func measureWithSingleIteration() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 1, warmupIterations: 0)
        let result = await benchmark.measure(.hadamard)

        #expect(result.iterations == 1, "single iteration benchmark should work")
        #expect(result.stdDevNs == 0, "single iteration should have zero standard deviation")
    }

    @Test
    func measureMinEqualsMaxWithSingleIteration() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 1, warmupIterations: 0)
        let result = await benchmark.measure(.pauliX)

        #expect(result.minNs == result.maxNs, "single iteration min and max should be equal")
        #expect(result.minNs == result.meanNs, "single iteration min should equal mean")
    }

    @Test
    func measureReturnsValidMinMeanMaxOrdering() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 1)
        let result = await benchmark.measure(.hadamard)

        #expect(result.minNs <= result.meanNs, "min should be less than or equal to mean")
        #expect(result.meanNs <= result.maxNs, "mean should be less than or equal to max")
    }

    @Test
    func compareReturnsArrayOfResultsWithValidTimings() async {
        let benchmark = GateBenchmark(qubits: 2, iterations: 5, warmupIterations: 1)
        let gates: [QuantumGate] = [.hadamard, .pauliX]
        let results = await benchmark.compare(gates)

        #expect(results.count == gates.count, "compare should return same number of results as input gates")
        for (index, result) in results.enumerated() {
            #expect(result.gate == gates[index], "result at index \(index) should match input gate")
            #expect(result.meanNs >= 0, "result at index \(index) should have non-negative mean")
        }
    }

    @Test
    func resultDescriptionFormatsCorrectly() {
        let result = GateBenchmarkResult(
            gate: .hadamard,
            meanNs: 1234.56,
            minNs: 1100.00,
            maxNs: 1500.00,
            stdDevNs: 89.12,
            iterations: 10,
        )

        let desc = result.description
        #expect(desc.contains("hadamard"), "description should contain gate name")
        #expect(desc.contains("mean=1234.56ns"), "description should contain formatted mean")
        #expect(desc.contains("min=1100.0"), "description should contain formatted min")
        #expect(desc.contains("max=1500.0"), "description should contain formatted max")
        #expect(desc.contains("stddev=89.12ns"), "description should contain formatted stddev")
        #expect(desc.contains("10 iterations"), "description should contain iteration count")
    }

    @Test
    func resultDescriptionFormatDoubleWithSmallFraction() {
        let result = GateBenchmarkResult(
            gate: .pauliX,
            meanNs: 100.05,
            minNs: 50.01,
            maxNs: 150.09,
            stdDevNs: 25.03,
            iterations: 5,
        )

        let desc = result.description
        #expect(desc.contains("mean=100.05ns"), "description should format small fraction with leading zero")
        #expect(desc.contains("min=50.01ns"), "description should format small fraction with leading zero")
        #expect(desc.contains("stddev=25.03ns"), "description should format small fraction with leading zero")
    }

    @Test
    func resultDescriptionFormatDoubleWithLargeFraction() {
        let result = GateBenchmarkResult(
            gate: .pauliZ,
            meanNs: 100.99,
            minNs: 50.55,
            maxNs: 150.10,
            stdDevNs: 25.75,
            iterations: 5,
        )

        let desc = result.description
        #expect(desc.contains("mean=100.99ns"), "description should format large fraction without leading zero")
        #expect(desc.contains("min=50.55ns"), "description should format large fraction without leading zero")
        #expect(desc.contains("stddev=25.75ns"), "description should format large fraction without leading zero")
    }
}
