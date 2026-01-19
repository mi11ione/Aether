// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation

enum PerformanceBaselines {
    static let circuitExecution8q: Double = 45.0 // Measured: 36.8ms
    static let circuitExecution10q: Double = 165.0 // Measured: 135.4ms
    static let circuitExecution12q: Double = 530.0 // Measured: 437.2ms

    static let sparseConstruction8q50t: Double = 15.0 // Measured: 12.1ms
    static let sparseConstruction10q50t: Double = 65.0 // Measured: 52.9ms
    static let sparseConstruction10q100t: Double = 100.0 // Measured: 83.3ms
    static let sparseExpectation10q50t: Double = 1.0 // Measured: 0.79ms

    static let mpsBatch6q100c: Double = 36.0 // Measured: 30.0ms
    static let mpsBatch8q50c: Double = 250.0 // Measured: 205.4ms
    static let mpsBatch10q20c: Double = 1500.0 // Measured: 1244.0ms
    static let mpsExpect8q50c: Double = 270.0 // Measured: 223.8ms

    static let qwcGrouping500: Double = 85.0 // Measured: 70.3ms
    static let qwcGrouping1000: Double = 345.0 // Measured: 283.7ms
    static let qwcGrouping2000: Double = 1260.0 // Measured: 1042.6ms

    static let vqe2qObservable: Double = 1.5 // Measured: 1.19ms
    static let vqe4qSparse: Double = 5.5 // Measured: 4.36ms
    static let vqe6qSparse: Double = 14.0 // Measured: 11.31ms

    static let qaoa4qSquare: Double = 21.0 // Measured: 17.15ms
    static let qaoa6qK6: Double = 185.0 // Measured: 152.9ms
    static let qaoa8qCycle: Double = 720.0 // Measured: 594.2ms

    static let regressionThreshold: Double = 0.20
}

struct BenchmarkSpec {
    let name: String
    let baselineMs: Double

    func isRegression(_ measuredMs: Double) -> Bool {
        let threshold = baselineMs * (1.0 + PerformanceBaselines.regressionThreshold)
        return measuredMs > threshold
    }

    func regressionPercent(_ measuredMs: Double) -> Double {
        ((measuredMs - baselineMs) / baselineMs) * 100.0
    }
}

enum BenchmarkRegistry {
    static let circuitExecution8q = BenchmarkSpec(
        name: "Circuit Execution (8 qubits)",
        baselineMs: PerformanceBaselines.circuitExecution8q,
    )
    static let circuitExecution10q = BenchmarkSpec(
        name: "Circuit Execution (10 qubits)",
        baselineMs: PerformanceBaselines.circuitExecution10q,
    )
    static let circuitExecution12q = BenchmarkSpec(
        name: "Circuit Execution (12 qubits)",
        baselineMs: PerformanceBaselines.circuitExecution12q,
    )
    static let sparseConstruction8q50t = BenchmarkSpec(
        name: "SparseHamiltonian Construction (8q, 50 terms)",
        baselineMs: PerformanceBaselines.sparseConstruction8q50t,
    )
    static let sparseConstruction10q50t = BenchmarkSpec(
        name: "SparseHamiltonian Construction (10q, 50 terms)",
        baselineMs: PerformanceBaselines.sparseConstruction10q50t,
    )
    static let sparseConstruction10q100t = BenchmarkSpec(
        name: "SparseHamiltonian Construction (10q, 100 terms)",
        baselineMs: PerformanceBaselines.sparseConstruction10q100t,
    )
    static let sparseExpectation10q50t = BenchmarkSpec(
        name: "Sparse Expectation Value (10q, 50 terms)",
        baselineMs: PerformanceBaselines.sparseExpectation10q50t,
    )
    static let mpsBatch6q100c = BenchmarkSpec(
        name: "MPSBatchEvaluator (6 qubits, 100 circuits)",
        baselineMs: PerformanceBaselines.mpsBatch6q100c,
    )
    static let mpsBatch8q50c = BenchmarkSpec(
        name: "MPSBatchEvaluator (8 qubits, 50 circuits)",
        baselineMs: PerformanceBaselines.mpsBatch8q50c,
    )
    static let mpsExpect8q50c = BenchmarkSpec(
        name: "MPSBatchEvaluator Expectation (8 qubits, 50 circuits)",
        baselineMs: PerformanceBaselines.mpsExpect8q50c,
    )
    static let qwcGrouping500 = BenchmarkSpec(
        name: "QWC Grouping (500 terms)",
        baselineMs: PerformanceBaselines.qwcGrouping500,
    )
    static let qwcGrouping1000 = BenchmarkSpec(
        name: "QWC Grouping (1000 terms)",
        baselineMs: PerformanceBaselines.qwcGrouping1000,
    )
    static let qwcGrouping2000 = BenchmarkSpec(
        name: "QWC Grouping (2000 terms)",
        baselineMs: PerformanceBaselines.qwcGrouping2000,
    )
    static let vqe2qObservable = BenchmarkSpec(
        name: "VQE 10-iter (2q, Observable backend)",
        baselineMs: PerformanceBaselines.vqe2qObservable,
    )
    static let vqe4qSparse = BenchmarkSpec(
        name: "VQE 10-iter (4q, Sparse backend)",
        baselineMs: PerformanceBaselines.vqe4qSparse,
    )
    static let vqe6qSparse = BenchmarkSpec(
        name: "VQE 10-iter (6q, Sparse backend)",
        baselineMs: PerformanceBaselines.vqe6qSparse,
    )
    static let qaoa4qSquare = BenchmarkSpec(
        name: "QAOA MaxCut Square (4q, p=2, 20 iter)",
        baselineMs: PerformanceBaselines.qaoa4qSquare,
    )
    static let qaoa6qK6 = BenchmarkSpec(
        name: "QAOA MaxCut K6 (6q, p=2, 20 iter)",
        baselineMs: PerformanceBaselines.qaoa6qK6,
    )
    static let qaoa8qCycle = BenchmarkSpec(
        name: "QAOA MaxCut Cycle (8q, p=2, 20 iter)",
        baselineMs: PerformanceBaselines.qaoa8qCycle,
    )
}

struct CIBenchmarkResult {
    let spec: BenchmarkSpec
    let result: BenchmarkResult
    let isRegression: Bool
    let regressionPercent: Double

    var status: String {
        if isRegression {
            "FAIL (+\(String(format: "%.1f", regressionPercent))%)"
        } else if regressionPercent > 0 {
            "PASS (+\(String(format: "%.1f", regressionPercent))%)"
        } else {
            "PASS (\(String(format: "%.1f", regressionPercent))%)"
        }
    }
}

struct BenchmarkTimer {
    private var startTime: UInt64 = 0
    private var endTime: UInt64 = 0

    mutating func start() {
        startTime = mach_absolute_time()
    }

    mutating func stop() -> UInt64 {
        endTime = mach_absolute_time()
        return endTime - startTime
    }

    static func machTimeToMilliseconds(_ machTime: UInt64) -> Double {
        var timebaseInfo = mach_timebase_info_data_t()
        mach_timebase_info(&timebaseInfo)
        let nanoseconds = Double(machTime) * Double(timebaseInfo.numer) / Double(timebaseInfo.denom)
        return nanoseconds / 1_000_000.0
    }

    static func machTimeToMicroseconds(_ machTime: UInt64) -> Double {
        var timebaseInfo = mach_timebase_info_data_t()
        mach_timebase_info(&timebaseInfo)
        let nanoseconds = Double(machTime) * Double(timebaseInfo.numer) / Double(timebaseInfo.denom)
        return nanoseconds / 1000.0
    }
}

struct BenchmarkResult {
    let name: String
    let iterations: Int
    let totalTimeMs: Double
    let meanTimeMs: Double
    let minTimeMs: Double
    let maxTimeMs: Double
    let stdDevMs: Double
    let throughput: Double?
    let throughputUnit: String?

    var summary: String {
        var result = """
        \(name)
          Iterations: \(iterations)
          Total:      \(String(format: "%.3f", totalTimeMs)) ms
          Mean:       \(String(format: "%.3f", meanTimeMs)) ms
          Min:        \(String(format: "%.3f", minTimeMs)) ms
          Max:        \(String(format: "%.3f", maxTimeMs)) ms
          Std Dev:    \(String(format: "%.3f", stdDevMs)) ms
        """
        if let tp = throughput, let unit = throughputUnit {
            result += "\n  Throughput: \(String(format: "%.2f", tp)) \(unit)"
        }
        return result
    }
}

func runBenchmark(
    name: String,
    warmupIterations: Int = 3,
    iterations: Int,
    throughputUnit: String? = nil,
    throughputDivisor: Double = 1.0,
    operation: () async throws -> Void,
) async -> BenchmarkResult {
    for _ in 0 ..< warmupIterations {
        try? await operation()
    }

    var times = [Double]()
    times.reserveCapacity(iterations)

    var timer = BenchmarkTimer()
    for _ in 0 ..< iterations {
        timer.start()
        try? await operation()
        let elapsed = timer.stop()
        times.append(BenchmarkTimer.machTimeToMilliseconds(elapsed))
    }

    let totalTime = times.reduce(0, +)
    let meanTime = totalTime / Double(iterations)
    let minTime = times.min() ?? 0
    let maxTime = times.max() ?? 0

    let variance = times.map { ($0 - meanTime) * ($0 - meanTime) }.reduce(0, +) / Double(iterations)
    let stdDev = sqrt(variance)

    let throughput: Double? = throughputUnit != nil ? throughputDivisor / meanTime * 1000.0 : nil

    return BenchmarkResult(
        name: name,
        iterations: iterations,
        totalTimeMs: totalTime,
        meanTimeMs: meanTime,
        minTimeMs: minTime,
        maxTimeMs: maxTime,
        stdDevMs: stdDev,
        throughput: throughput,
        throughputUnit: throughputUnit,
    )
}

func benchmarkSparseHamiltonian() async -> [BenchmarkResult] {
    print("\n" + String(repeating: "=", count: 60))
    print("SPARSE HAMILTONIAN BENCHMARKS")
    print(String(repeating: "=", count: 60))

    var results = [BenchmarkResult]()

    func generateHamiltonian(qubits: Int, terms: Int) -> Observable {
        var pauliTerms: PauliTerms = []
        pauliTerms.reserveCapacity(terms)

        for i in 0 ..< terms {
            var operators = [PauliOperator]()
            let numOps = 1 + (i % 3)
            for j in 0 ..< numOps {
                let qubit = (i + j) % qubits
                let basis: PauliBasis = switch (i + j) % 3 {
                case 0: .x
                case 1: .y
                default: .z
                }
                operators.append(PauliOperator(qubit: qubit, basis: basis))
            }
            let coefficient = 0.1 + Double(i % 10) * 0.05
            pauliTerms.append((coefficient, PauliString(operators)))
        }

        return Observable(terms: pauliTerms)
    }

    print("  Preparing test Hamiltonians...")

    let obs8q50t = generateHamiltonian(qubits: 8, terms: 50)
    let obs10q50t = generateHamiltonian(qubits: 10, terms: 50)
    let obs10q100t = generateHamiltonian(qubits: 10, terms: 100)

    let result1 = await runBenchmark(
        name: "SparseHamiltonian Construction (8q, 50 terms)",
        iterations: 20,
    ) {
        _ = SparseHamiltonian(observable: obs8q50t, systemSize: 8)
    }
    print(result1.summary)
    results.append(result1)

    let result2 = await runBenchmark(
        name: "SparseHamiltonian Construction (10q, 50 terms)",
        iterations: 10,
    ) {
        _ = SparseHamiltonian(observable: obs10q50t, systemSize: 10)
    }
    print(result2.summary)
    results.append(result2)

    let result3 = await runBenchmark(
        name: "SparseHamiltonian Construction (10q, 100 terms)",
        iterations: 10,
    ) {
        _ = SparseHamiltonian(observable: obs10q100t, systemSize: 10)
    }
    print(result3.summary)
    results.append(result3)

    print("  Preparing test state...")
    let sparseH = SparseHamiltonian(observable: obs10q50t, systemSize: 10)
    var circuit = QuantumCircuit(qubits: 10)
    for i in 0 ..< 10 {
        circuit.append(.hadamard, to: i)
    }
    let state = circuit.execute()

    let result4 = await runBenchmark(
        name: "Sparse Expectation Value (10q, 50 terms)",
        iterations: 100,
        throughputUnit: "evals/sec",
        throughputDivisor: 1.0,
    ) {
        _ = await sparseH.expectationValue(of: state)
    }
    print(result4.summary)
    results.append(result4)

    return results
}

func benchmarkMPSBatchEvaluator() async -> [BenchmarkResult] {
    print("\n" + String(repeating: "=", count: 60))
    print("MPS BATCH EVALUATOR BENCHMARKS")
    print(String(repeating: "=", count: 60))

    var results = [BenchmarkResult]()

    let evaluator = MPSBatchEvaluator()
    let stats = await evaluator.statistics
    print("Backend: \(stats.deviceName)")
    print("Max Batch Size: \(stats.maxBatchSize)")

    func generateCircuitBatch(qubits: Int, count: Int, depth: Int, label: String) -> [[[Complex<Double>]]] {
        print("  Generating \(label)...", terminator: "")
        fflush(stdout)
        var unitaries = [[[Complex<Double>]]]()
        unitaries.reserveCapacity(count)

        for i in 0 ..< count {
            var circuit = QuantumCircuit(qubits: qubits)
            for j in 0 ..< depth {
                circuit.append(.hadamard, to: j % qubits)
                if qubits >= 2 {
                    circuit.append(.cnot, to: [j % qubits, (j + 1) % qubits])
                }
                circuit.append(.rotationZ(Double(i + j) * 0.1), to: j % qubits)
            }
            unitaries.append(CircuitUnitary.unitary(for: circuit))
        }
        print(" done")
        return unitaries
    }

    let batch6q = generateCircuitBatch(qubits: 6, count: 100, depth: 8, label: "6-qubit batch (100 circuits)")
    let initialState6 = QuantumState(qubits: 6)

    let result1 = await runBenchmark(
        name: "MPSBatchEvaluator (6 qubits, 100 circuits)",
        iterations: 20,
        throughputUnit: "circuits/sec",
        throughputDivisor: 100.0,
    ) {
        _ = await evaluator.evaluate(batch: batch6q, from: initialState6)
    }
    print(result1.summary)
    results.append(result1)

    let batch8q = generateCircuitBatch(qubits: 8, count: 50, depth: 8, label: "8-qubit batch (50 circuits)")
    let initialState8 = QuantumState(qubits: 8)

    let result2 = await runBenchmark(
        name: "MPSBatchEvaluator (8 qubits, 50 circuits)",
        iterations: 10,
        throughputUnit: "circuits/sec",
        throughputDivisor: 50.0,
    ) {
        _ = await evaluator.evaluate(batch: batch8q, from: initialState8)
    }
    print(result2.summary)
    results.append(result2)

    let hamiltonian = Observable(terms: [
        (1.0, PauliString(.z(0))),
        (0.5, PauliString(.z(1))),
        (0.3, PauliString(.x(0), .x(1))),
    ])

    let result4 = await runBenchmark(
        name: "MPSBatchEvaluator Expectation (8 qubits, 50 circuits)",
        iterations: 10,
        throughputUnit: "circuits/sec",
        throughputDivisor: 50.0,
    ) {
        _ = await evaluator.expectationValues(
            for: batch8q,
            from: initialState8,
            observable: hamiltonian,
        )
    }
    print(result4.summary)
    results.append(result4)

    return results
}

func benchmarkVQE() async -> [BenchmarkResult] {
    print("\n" + String(repeating: "=", count: 60))
    print("VQE BENCHMARKS")
    print(String(repeating: "=", count: 60))

    var results = [BenchmarkResult]()
    print("  Preparing VQE instances...")

    let optimizer = COBYLAOptimizer()

    let hamiltonian2q = Observable(terms: [
        (1.0, PauliString(.z(0))),
        (1.0, PauliString(.z(1))),
        (0.5, PauliString(.x(0), .x(1))),
    ])
    let ansatz2q = HardwareEfficientAnsatz(qubits: 2, depth: 2)
    let vqe2q = VQE(
        hamiltonian: hamiltonian2q,
        ansatz: ansatz2q,
        optimizer: optimizer,
        convergence: ConvergenceCriteria(energyTolerance: 1e-6, maxIterations: 10),
        useSparseBackend: false,
    )

    let result1 = await runBenchmark(
        name: "VQE 10-iter (2q, Observable backend)",
        iterations: 20,
    ) {
        _ = await vqe2q.run(from: [0.1, 0.2, 0.3, 0.4])
    }
    print(result1.summary)
    results.append(result1)

    let hamiltonian4q = Observable(terms: [
        (1.0, PauliString(.z(0))),
        (1.0, PauliString(.z(1))),
        (1.0, PauliString(.z(2))),
        (1.0, PauliString(.z(3))),
        (0.5, PauliString(.x(0), .x(1))),
        (0.5, PauliString(.x(2), .x(3))),
        (0.3, PauliString(.z(0), .z(2))),
    ])
    let ansatz4q = HardwareEfficientAnsatz(qubits: 4, depth: 2)
    let vqe4q = VQE(
        hamiltonian: hamiltonian4q,
        ansatz: ansatz4q,
        optimizer: optimizer,
        convergence: ConvergenceCriteria(energyTolerance: 1e-6, maxIterations: 10),
        useSparseBackend: true,
    )
    let initialParams4q = [Double](repeating: 0.1, count: ansatz4q.parameterCount)

    let result2 = await runBenchmark(
        name: "VQE 10-iter (4q, Sparse backend)",
        iterations: 10,
    ) {
        _ = await vqe4q.run(from: initialParams4q)
    }
    print(result2.summary)
    results.append(result2)

    let hamiltonian6q = Observable(terms: [
        (1.0, PauliString(.z(0))), (1.0, PauliString(.z(1))),
        (1.0, PauliString(.z(2))), (1.0, PauliString(.z(3))),
        (1.0, PauliString(.z(4))), (1.0, PauliString(.z(5))),
        (0.5, PauliString(.x(0), .x(1))), (0.5, PauliString(.x(2), .x(3))),
        (0.5, PauliString(.x(4), .x(5))), (0.3, PauliString(.z(0), .z(3))),
    ])
    let ansatz6q = HardwareEfficientAnsatz(qubits: 6, depth: 2)
    let vqe6q = VQE(
        hamiltonian: hamiltonian6q,
        ansatz: ansatz6q,
        optimizer: optimizer,
        convergence: ConvergenceCriteria(energyTolerance: 1e-6, maxIterations: 10),
        useSparseBackend: true,
    )
    let initialParams6q = [Double](repeating: 0.1, count: ansatz6q.parameterCount)

    let result3 = await runBenchmark(
        name: "VQE 10-iter (6q, Sparse backend)",
        iterations: 5,
    ) {
        _ = await vqe6q.run(from: initialParams6q)
    }
    print(result3.summary)
    results.append(result3)

    return results
}

func benchmarkQWCGrouping() async -> [BenchmarkResult] {
    print("\n" + String(repeating: "=", count: 60))
    print("QWC GROUPING BENCHMARKS")
    print(String(repeating: "=", count: 60))

    var results = [BenchmarkResult]()

    func generateLargeHamiltonian(qubits: Int, terms: Int) -> Observable {
        var pauliTerms: PauliTerms = []
        pauliTerms.reserveCapacity(terms)

        for i in 0 ..< terms {
            var operators = [PauliOperator]()
            let numOps = 1 + (i % 4)
            for j in 0 ..< numOps {
                let qubit = (i * 7 + j * 3) % qubits
                let basis: PauliBasis = switch (i * 11 + j) % 3 {
                case 0: .x
                case 1: .y
                default: .z
                }
                if !operators.contains(where: { $0.qubit == qubit }) {
                    operators.append(PauliOperator(qubit: qubit, basis: basis))
                }
            }
            if !operators.isEmpty {
                let coefficient = 0.01 + Double(i % 100) * 0.001
                pauliTerms.append((coefficient, PauliString(operators)))
            }
        }

        return Observable(terms: pauliTerms)
    }

    print("  Generating test Hamiltonians...")
    let obs500 = generateLargeHamiltonian(qubits: 10, terms: 500)
    let obs1000 = generateLargeHamiltonian(qubits: 12, terms: 1000)
    let obs2000 = generateLargeHamiltonian(qubits: 14, terms: 2000)

    let result1 = await runBenchmark(
        name: "QWC Grouping (500 terms)",
        iterations: 20,
    ) {
        _ = QWCGrouper.group(obs500.terms)
    }
    print(result1.summary)
    results.append(result1)

    let result2 = await runBenchmark(
        name: "QWC Grouping (1000 terms)",
        iterations: 10,
    ) {
        _ = QWCGrouper.group(obs1000.terms)
    }
    print(result2.summary)
    results.append(result2)

    let result3 = await runBenchmark(
        name: "QWC Grouping (2000 terms)",
        iterations: 5,
    ) {
        _ = QWCGrouper.group(obs2000.terms)
    }
    print(result3.summary)
    results.append(result3)

    let groups = QWCGrouper.group(obs2000.terms)
    let stats = QWCGrouper.statistics(for: groups)
    print("  Reduction: \(stats.numTerms) terms -> \(stats.numGroups) groups (\(String(format: "%.0f", stats.reductionFactor))x)")

    return results
}

func benchmarkCircuitExecution() async -> [BenchmarkResult] {
    print("\n" + String(repeating: "=", count: 60))
    print("CIRCUIT EXECUTION BENCHMARKS")
    print(String(repeating: "=", count: 60))

    var results = [BenchmarkResult]()

    func createBenchmarkCircuit(qubits: Int, depth: Int) -> QuantumCircuit {
        var circuit = QuantumCircuit(qubits: qubits)
        for layer in 0 ..< depth {
            for q in 0 ..< qubits {
                circuit.append(.hadamard, to: q)
                circuit.append(.rotationZ(Double(layer) * 0.1), to: q)
            }
            for q in 0 ..< (qubits - 1) {
                circuit.append(.cnot, to: [q, q + 1])
            }
        }
        return circuit
    }

    print("  Preparing circuits...")

    let circuit8q = createBenchmarkCircuit(qubits: 8, depth: 20)
    let gateCount8 = circuit8q.count
    let result1 = await runBenchmark(
        name: "Circuit Execution (8 qubits, \(gateCount8) gates)",
        iterations: 30,
        throughputUnit: "gates/sec",
        throughputDivisor: Double(gateCount8),
    ) {
        _ = circuit8q.execute()
    }
    print(result1.summary)
    results.append(result1)

    let circuit10q = createBenchmarkCircuit(qubits: 10, depth: 15)
    let gateCount10 = circuit10q.count
    let result2 = await runBenchmark(
        name: "Circuit Execution (10 qubits, \(gateCount10) gates)",
        iterations: 20,
        throughputUnit: "gates/sec",
        throughputDivisor: Double(gateCount10),
    ) {
        _ = circuit10q.execute()
    }
    print(result2.summary)
    results.append(result2)

    let circuit12q = createBenchmarkCircuit(qubits: 12, depth: 10)
    let gateCount12 = circuit12q.count
    let result3 = await runBenchmark(
        name: "Circuit Execution (12 qubits, \(gateCount12) gates)",
        iterations: 10,
        throughputUnit: "gates/sec",
        throughputDivisor: Double(gateCount12),
    ) {
        _ = circuit12q.execute()
    }
    print(result3.summary)
    results.append(result3)

    return results
}

func benchmarkQAOA() async -> [BenchmarkResult] {
    print("\n" + String(repeating: "=", count: 60))
    print("QAOA BENCHMARKS")
    print(String(repeating: "=", count: 60))

    var results = [BenchmarkResult]()
    print("  Preparing QAOA instances...")

    let optimizer = COBYLAOptimizer()

    let edges4 = MaxCut.Examples.square()
    let cost4 = MaxCut.hamiltonian(edges: edges4)
    let mixer4 = MixerHamiltonian.x(qubits: 4)
    let qaoa4 = QAOA(
        cost: cost4,
        mixer: mixer4,
        qubits: 4,
        depth: 2,
        optimizer: optimizer,
        convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 20),
    )

    let result1 = await runBenchmark(
        name: "QAOA MaxCut Square (4q, p=2, 20 iter)",
        iterations: 10,
    ) {
        _ = await qaoa4.run(from: [0.1, 0.2, 0.3, 0.4])
    }
    print(result1.summary)
    results.append(result1)

    let edges6 = MaxCut.Examples.complete(vertices: 6)
    let cost6 = MaxCut.hamiltonian(edges: edges6)
    let mixer6 = MixerHamiltonian.x(qubits: 6)
    let qaoa6 = QAOA(
        cost: cost6,
        mixer: mixer6,
        qubits: 6,
        depth: 2,
        optimizer: optimizer,
        convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 20),
    )

    let result2 = await runBenchmark(
        name: "QAOA MaxCut K6 (6q, p=2, 20 iter)",
        iterations: 5,
    ) {
        _ = await qaoa6.run(from: [0.1, 0.2, 0.3, 0.4])
    }
    print(result2.summary)
    results.append(result2)

    let edges8 = MaxCut.Examples.cycle(vertices: 8)
    let cost8 = MaxCut.hamiltonian(edges: edges8)
    let mixer8 = MixerHamiltonian.x(qubits: 8)
    let qaoa8 = QAOA(
        cost: cost8,
        mixer: mixer8,
        qubits: 8,
        depth: 2,
        optimizer: optimizer,
        convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 20),
    )

    let result3 = await runBenchmark(
        name: "QAOA MaxCut Cycle (8q, p=2, 20 iter)",
        iterations: 3,
    ) {
        _ = await qaoa8.run(from: [0.1, 0.2, 0.3, 0.4])
    }
    print(result3.summary)
    results.append(result3)

    return results
}

func runCIMode() async -> Int32 {
    print("""
    ╔═════════════════════════════════════════════════════════════════════════════════╗
    ║                   AETHER PERFORMANCE BENCHMARKS - CI MODE                       ║
    ╚═════════════════════════════════════════════════════════════════════════════════╝
    """)

    let processInfo = ProcessInfo.processInfo
    print("System: \(processInfo.activeProcessorCount) cores, \(processInfo.physicalMemory / (1024 * 1024 * 1024)) GB RAM")
    print("Regression threshold: \(Int(PerformanceBaselines.regressionThreshold * 100))%")
    print("")

    var ciResults = [CIBenchmarkResult]()

    let circuitResults = await benchmarkCircuitExecutionCI()
    let sparseResults = await benchmarkSparseHamiltonianCI()
    let mpsResults = await benchmarkMPSBatchEvaluatorCI()
    let qwcResults = await benchmarkQWCGroupingCI()
    let vqeResults = await benchmarkVQECI()
    let qaoaResults = await benchmarkQAOACI()

    ciResults.append(contentsOf: circuitResults)
    ciResults.append(contentsOf: sparseResults)
    ciResults.append(contentsOf: mpsResults)
    ciResults.append(contentsOf: qwcResults)
    ciResults.append(contentsOf: vqeResults)
    ciResults.append(contentsOf: qaoaResults)

    print("\n" + String(repeating: "=", count: 85))
    print("CI REGRESSION REPORT")
    print(String(repeating: "=", count: 85))

    let header = "Benchmark".padding(toLength: 50, withPad: " ", startingAt: 0)
        + "Baseline".padding(toLength: 12, withPad: " ", startingAt: 0)
        + "Measured".padding(toLength: 12, withPad: " ", startingAt: 0)
        + "Status"
    print(header)
    print(String(repeating: "-", count: 85))

    var regressionCount = 0
    for ci in ciResults {
        let status = ci.isRegression ? "FAIL" : "PASS"
        let percentStr = ci.regressionPercent >= 0
            ? "+\(String(format: "%.1f", ci.regressionPercent))%"
            : "\(String(format: "%.1f", ci.regressionPercent))%"

        let nameCol = String(ci.spec.name.prefix(48)).padding(toLength: 50, withPad: " ", startingAt: 0)
        let baselineCol = String(format: "%.2fms", ci.spec.baselineMs).padding(toLength: 12, withPad: " ", startingAt: 0)
        let measuredCol = String(format: "%.2fms", ci.result.meanTimeMs).padding(toLength: 12, withPad: " ", startingAt: 0)
        let statusCol = "\(status) \(percentStr)"

        print(nameCol + baselineCol + measuredCol + statusCol)

        if ci.isRegression {
            regressionCount += 1
        }
    }

    print(String(repeating: "=", count: 85))

    if regressionCount > 0 {
        print("\n!! REGRESSION DETECTED: \(regressionCount) benchmark(s) exceeded \(Int(PerformanceBaselines.regressionThreshold * 100))% threshold")
        print("Review the FAIL entries above for details.")
        return 1
    } else {
        print("\n++ ALL BENCHMARKS PASSED")
        print("No performance regressions detected.")
        return 0
    }
}

func benchmarkCircuitExecutionCI() async -> [CIBenchmarkResult] {
    print("Running: Circuit Execution...")
    var results = [CIBenchmarkResult]()

    func createBenchmarkCircuit(qubits: Int, depth: Int) -> QuantumCircuit {
        var circuit = QuantumCircuit(qubits: qubits)
        for layer in 0 ..< depth {
            for q in 0 ..< qubits {
                circuit.append(.hadamard, to: q)
                circuit.append(.rotationZ(Double(layer) * 0.1), to: q)
            }
            for q in 0 ..< (qubits - 1) {
                circuit.append(.cnot, to: [q, q + 1])
            }
        }
        return circuit
    }

    let circuit8q = createBenchmarkCircuit(qubits: 8, depth: 20)
    let result8 = await runBenchmark(name: "8q", iterations: 30) { _ = circuit8q.execute() }
    let spec8 = BenchmarkRegistry.circuitExecution8q
    results.append(CIBenchmarkResult(
        spec: spec8,
        result: result8,
        isRegression: spec8.isRegression(result8.meanTimeMs),
        regressionPercent: spec8.regressionPercent(result8.meanTimeMs),
    ))

    let circuit10q = createBenchmarkCircuit(qubits: 10, depth: 15)
    let result10 = await runBenchmark(name: "10q", iterations: 20) { _ = circuit10q.execute() }
    let spec10 = BenchmarkRegistry.circuitExecution10q
    results.append(CIBenchmarkResult(
        spec: spec10,
        result: result10,
        isRegression: spec10.isRegression(result10.meanTimeMs),
        regressionPercent: spec10.regressionPercent(result10.meanTimeMs),
    ))

    let circuit12q = createBenchmarkCircuit(qubits: 12, depth: 10)
    let result12 = await runBenchmark(name: "12q", iterations: 10) { _ = circuit12q.execute() }
    let spec12 = BenchmarkRegistry.circuitExecution12q
    results.append(CIBenchmarkResult(
        spec: spec12,
        result: result12,
        isRegression: spec12.isRegression(result12.meanTimeMs),
        regressionPercent: spec12.regressionPercent(result12.meanTimeMs),
    ))

    return results
}

func benchmarkSparseHamiltonianCI() async -> [CIBenchmarkResult] {
    print("Running: SparseHamiltonian...")
    var results = [CIBenchmarkResult]()

    func generateHamiltonian(qubits: Int, terms: Int) -> Observable {
        var pauliTerms: PauliTerms = []
        pauliTerms.reserveCapacity(terms)
        for i in 0 ..< terms {
            var operators = [PauliOperator]()
            let numOps = 1 + (i % 3)
            for j in 0 ..< numOps {
                let qubit = (i + j) % qubits
                let basis: PauliBasis = switch (i + j) % 3 {
                case 0: .x
                case 1: .y
                default: .z
                }
                operators.append(PauliOperator(qubit: qubit, basis: basis))
            }
            let coefficient = 0.1 + Double(i % 10) * 0.05
            pauliTerms.append((coefficient, PauliString(operators)))
        }
        return Observable(terms: pauliTerms)
    }

    let obs8q50t = generateHamiltonian(qubits: 8, terms: 50)
    let result1 = await runBenchmark(name: "8q50t", iterations: 20) {
        _ = SparseHamiltonian(observable: obs8q50t, systemSize: 8)
    }
    let spec1 = BenchmarkRegistry.sparseConstruction8q50t
    results.append(CIBenchmarkResult(
        spec: spec1, result: result1,
        isRegression: spec1.isRegression(result1.meanTimeMs),
        regressionPercent: spec1.regressionPercent(result1.meanTimeMs),
    ))

    let obs10q50t = generateHamiltonian(qubits: 10, terms: 50)
    let result2 = await runBenchmark(name: "10q50t", iterations: 10) {
        _ = SparseHamiltonian(observable: obs10q50t, systemSize: 10)
    }
    let spec2 = BenchmarkRegistry.sparseConstruction10q50t
    results.append(CIBenchmarkResult(
        spec: spec2, result: result2,
        isRegression: spec2.isRegression(result2.meanTimeMs),
        regressionPercent: spec2.regressionPercent(result2.meanTimeMs),
    ))

    let obs10q100t = generateHamiltonian(qubits: 10, terms: 100)
    let result3 = await runBenchmark(name: "10q100t", iterations: 10) {
        _ = SparseHamiltonian(observable: obs10q100t, systemSize: 10)
    }
    let spec3 = BenchmarkRegistry.sparseConstruction10q100t
    results.append(CIBenchmarkResult(
        spec: spec3, result: result3,
        isRegression: spec3.isRegression(result3.meanTimeMs),
        regressionPercent: spec3.regressionPercent(result3.meanTimeMs),
    ))

    let sparseH = SparseHamiltonian(observable: obs10q50t, systemSize: 10)
    var circuit = QuantumCircuit(qubits: 10)
    for i in 0 ..< 10 {
        circuit.append(.hadamard, to: i)
    }
    let state = circuit.execute()

    let result4 = await runBenchmark(name: "expect", iterations: 100) {
        _ = await sparseH.expectationValue(of: state)
    }
    let spec4 = BenchmarkRegistry.sparseExpectation10q50t
    results.append(CIBenchmarkResult(
        spec: spec4, result: result4,
        isRegression: spec4.isRegression(result4.meanTimeMs),
        regressionPercent: spec4.regressionPercent(result4.meanTimeMs),
    ))

    return results
}

func benchmarkMPSBatchEvaluatorCI() async -> [CIBenchmarkResult] {
    print("Running: MPSBatchEvaluator...")
    var results = [CIBenchmarkResult]()

    let evaluator = MPSBatchEvaluator()

    func generateCircuitBatch(qubits: Int, count: Int, depth: Int) -> [[[Complex<Double>]]] {
        var unitaries = [[[Complex<Double>]]]()
        unitaries.reserveCapacity(count)
        for i in 0 ..< count {
            var circuit = QuantumCircuit(qubits: qubits)
            for j in 0 ..< depth {
                circuit.append(.hadamard, to: j % qubits)
                if qubits >= 2 {
                    circuit.append(.cnot, to: [j % qubits, (j + 1) % qubits])
                }
                circuit.append(.rotationZ(Double(i + j) * 0.1), to: j % qubits)
            }
            unitaries.append(CircuitUnitary.unitary(for: circuit))
        }
        return unitaries
    }

    let batch6q = generateCircuitBatch(qubits: 6, count: 100, depth: 8)
    let initialState6 = QuantumState(qubits: 6)
    let result1 = await runBenchmark(name: "6q100c", iterations: 20) {
        _ = await evaluator.evaluate(batch: batch6q, from: initialState6)
    }
    let spec1 = BenchmarkRegistry.mpsBatch6q100c
    results.append(CIBenchmarkResult(
        spec: spec1, result: result1,
        isRegression: spec1.isRegression(result1.meanTimeMs),
        regressionPercent: spec1.regressionPercent(result1.meanTimeMs),
    ))

    let batch8q = generateCircuitBatch(qubits: 8, count: 50, depth: 8)
    let initialState8 = QuantumState(qubits: 8)
    let result2 = await runBenchmark(name: "8q50c", iterations: 10) {
        _ = await evaluator.evaluate(batch: batch8q, from: initialState8)
    }
    let spec2 = BenchmarkRegistry.mpsBatch8q50c
    results.append(CIBenchmarkResult(
        spec: spec2, result: result2,
        isRegression: spec2.isRegression(result2.meanTimeMs),
        regressionPercent: spec2.regressionPercent(result2.meanTimeMs),
    ))

    let hamiltonian = Observable(terms: [
        (1.0, PauliString(.z(0))),
        (0.5, PauliString(.z(1))),
        (0.3, PauliString(.x(0), .x(1))),
    ])
    let result4 = await runBenchmark(name: "expect8q", iterations: 10) {
        _ = await evaluator.expectationValues(for: batch8q, from: initialState8, observable: hamiltonian)
    }
    let spec4 = BenchmarkRegistry.mpsExpect8q50c
    results.append(CIBenchmarkResult(
        spec: spec4, result: result4,
        isRegression: spec4.isRegression(result4.meanTimeMs),
        regressionPercent: spec4.regressionPercent(result4.meanTimeMs),
    ))

    return results
}

func benchmarkQWCGroupingCI() async -> [CIBenchmarkResult] {
    print("Running: QWC Grouping...")
    var results = [CIBenchmarkResult]()

    func generateLargeHamiltonian(qubits: Int, terms: Int) -> Observable {
        var pauliTerms: PauliTerms = []
        pauliTerms.reserveCapacity(terms)
        for i in 0 ..< terms {
            var operators = [PauliOperator]()
            let numOps = 1 + (i % 4)
            for j in 0 ..< numOps {
                let qubit = (i * 7 + j * 3) % qubits
                let basis: PauliBasis = switch (i * 11 + j) % 3 {
                case 0: .x
                case 1: .y
                default: .z
                }
                if !operators.contains(where: { $0.qubit == qubit }) {
                    operators.append(PauliOperator(qubit: qubit, basis: basis))
                }
            }
            if !operators.isEmpty {
                let coefficient = 0.01 + Double(i % 100) * 0.001
                pauliTerms.append((coefficient, PauliString(operators)))
            }
        }
        return Observable(terms: pauliTerms)
    }

    let obs500 = generateLargeHamiltonian(qubits: 10, terms: 500)
    let result1 = await runBenchmark(name: "500t", iterations: 20) {
        _ = QWCGrouper.group(obs500.terms)
    }
    let spec1 = BenchmarkRegistry.qwcGrouping500
    results.append(CIBenchmarkResult(
        spec: spec1, result: result1,
        isRegression: spec1.isRegression(result1.meanTimeMs),
        regressionPercent: spec1.regressionPercent(result1.meanTimeMs),
    ))

    let obs1000 = generateLargeHamiltonian(qubits: 12, terms: 1000)
    let result2 = await runBenchmark(name: "1000t", iterations: 10) {
        _ = QWCGrouper.group(obs1000.terms)
    }
    let spec2 = BenchmarkRegistry.qwcGrouping1000
    results.append(CIBenchmarkResult(
        spec: spec2, result: result2,
        isRegression: spec2.isRegression(result2.meanTimeMs),
        regressionPercent: spec2.regressionPercent(result2.meanTimeMs),
    ))

    let obs2000 = generateLargeHamiltonian(qubits: 14, terms: 2000)
    let result3 = await runBenchmark(name: "2000t", iterations: 5) {
        _ = QWCGrouper.group(obs2000.terms)
    }
    let spec3 = BenchmarkRegistry.qwcGrouping2000
    results.append(CIBenchmarkResult(
        spec: spec3, result: result3,
        isRegression: spec3.isRegression(result3.meanTimeMs),
        regressionPercent: spec3.regressionPercent(result3.meanTimeMs),
    ))

    return results
}

func benchmarkVQECI() async -> [CIBenchmarkResult] {
    print("Running: VQE...")
    var results = [CIBenchmarkResult]()

    let optimizer = COBYLAOptimizer()

    let hamiltonian2q = Observable(terms: [
        (1.0, PauliString(.z(0))),
        (1.0, PauliString(.z(1))),
        (0.5, PauliString(.x(0), .x(1))),
    ])
    let ansatz2q = HardwareEfficientAnsatz(qubits: 2, depth: 2)
    let vqe2q = VQE(
        hamiltonian: hamiltonian2q, ansatz: ansatz2q, optimizer: optimizer,
        convergence: ConvergenceCriteria(energyTolerance: 1e-6, maxIterations: 10),
        useSparseBackend: false,
    )
    let result1 = await runBenchmark(name: "2q", iterations: 20) {
        _ = await vqe2q.run(from: [0.1, 0.2, 0.3, 0.4])
    }
    let spec1 = BenchmarkRegistry.vqe2qObservable
    results.append(CIBenchmarkResult(
        spec: spec1, result: result1,
        isRegression: spec1.isRegression(result1.meanTimeMs),
        regressionPercent: spec1.regressionPercent(result1.meanTimeMs),
    ))

    let hamiltonian4q = Observable(terms: [
        (1.0, PauliString(.z(0))), (1.0, PauliString(.z(1))),
        (1.0, PauliString(.z(2))), (1.0, PauliString(.z(3))),
        (0.5, PauliString(.x(0), .x(1))), (0.5, PauliString(.x(2), .x(3))),
        (0.3, PauliString(.z(0), .z(2))),
    ])
    let ansatz4q = HardwareEfficientAnsatz(qubits: 4, depth: 2)
    let vqe4q = VQE(
        hamiltonian: hamiltonian4q, ansatz: ansatz4q, optimizer: optimizer,
        convergence: ConvergenceCriteria(energyTolerance: 1e-6, maxIterations: 10),
        useSparseBackend: true,
    )
    let initialParams4q = [Double](repeating: 0.1, count: ansatz4q.parameterCount)
    let result2 = await runBenchmark(name: "4q", iterations: 10) {
        _ = await vqe4q.run(from: initialParams4q)
    }
    let spec2 = BenchmarkRegistry.vqe4qSparse
    results.append(CIBenchmarkResult(
        spec: spec2, result: result2,
        isRegression: spec2.isRegression(result2.meanTimeMs),
        regressionPercent: spec2.regressionPercent(result2.meanTimeMs),
    ))

    let hamiltonian6q = Observable(terms: [
        (1.0, PauliString(.z(0))), (1.0, PauliString(.z(1))),
        (1.0, PauliString(.z(2))), (1.0, PauliString(.z(3))),
        (1.0, PauliString(.z(4))), (1.0, PauliString(.z(5))),
        (0.5, PauliString(.x(0), .x(1))), (0.5, PauliString(.x(2), .x(3))),
        (0.5, PauliString(.x(4), .x(5))), (0.3, PauliString(.z(0), .z(3))),
    ])
    let ansatz6q = HardwareEfficientAnsatz(qubits: 6, depth: 2)
    let vqe6q = VQE(
        hamiltonian: hamiltonian6q, ansatz: ansatz6q, optimizer: optimizer,
        convergence: ConvergenceCriteria(energyTolerance: 1e-6, maxIterations: 10),
        useSparseBackend: true,
    )
    let initialParams6q = [Double](repeating: 0.1, count: ansatz6q.parameterCount)
    let result3 = await runBenchmark(name: "6q", iterations: 5) {
        _ = await vqe6q.run(from: initialParams6q)
    }
    let spec3 = BenchmarkRegistry.vqe6qSparse
    results.append(CIBenchmarkResult(
        spec: spec3, result: result3,
        isRegression: spec3.isRegression(result3.meanTimeMs),
        regressionPercent: spec3.regressionPercent(result3.meanTimeMs),
    ))

    return results
}

func benchmarkQAOACI() async -> [CIBenchmarkResult] {
    print("Running: QAOA...")
    var results = [CIBenchmarkResult]()

    let optimizer = COBYLAOptimizer()

    let edges4 = MaxCut.Examples.square()
    let cost4 = MaxCut.hamiltonian(edges: edges4)
    let mixer4 = MixerHamiltonian.x(qubits: 4)
    let qaoa4 = QAOA(
        cost: cost4, mixer: mixer4, qubits: 4, depth: 2, optimizer: optimizer,
        convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 20),
    )
    let result1 = await runBenchmark(name: "4q", iterations: 10) {
        _ = await qaoa4.run(from: [0.1, 0.2, 0.3, 0.4])
    }
    let spec1 = BenchmarkRegistry.qaoa4qSquare
    results.append(CIBenchmarkResult(
        spec: spec1, result: result1,
        isRegression: spec1.isRegression(result1.meanTimeMs),
        regressionPercent: spec1.regressionPercent(result1.meanTimeMs),
    ))

    let edges6 = MaxCut.Examples.complete(vertices: 6)
    let cost6 = MaxCut.hamiltonian(edges: edges6)
    let mixer6 = MixerHamiltonian.x(qubits: 6)
    let qaoa6 = QAOA(
        cost: cost6, mixer: mixer6, qubits: 6, depth: 2, optimizer: optimizer,
        convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 20),
    )
    let result2 = await runBenchmark(name: "6q", iterations: 5) {
        _ = await qaoa6.run(from: [0.1, 0.2, 0.3, 0.4])
    }
    let spec2 = BenchmarkRegistry.qaoa6qK6
    results.append(CIBenchmarkResult(
        spec: spec2, result: result2,
        isRegression: spec2.isRegression(result2.meanTimeMs),
        regressionPercent: spec2.regressionPercent(result2.meanTimeMs),
    ))

    let edges8 = MaxCut.Examples.cycle(vertices: 8)
    let cost8 = MaxCut.hamiltonian(edges: edges8)
    let mixer8 = MixerHamiltonian.x(qubits: 8)
    let qaoa8 = QAOA(
        cost: cost8, mixer: mixer8, qubits: 8, depth: 2, optimizer: optimizer,
        convergence: ConvergenceCriteria(energyTolerance: 1e-4, maxIterations: 20),
    )
    let result3 = await runBenchmark(name: "8q", iterations: 3) {
        _ = await qaoa8.run(from: [0.1, 0.2, 0.3, 0.4])
    }
    let spec3 = BenchmarkRegistry.qaoa8qCycle
    results.append(CIBenchmarkResult(
        spec: spec3, result: result3,
        isRegression: spec3.isRegression(result3.meanTimeMs),
        regressionPercent: spec3.regressionPercent(result3.meanTimeMs),
    ))

    return results
}

@main
struct AetherBenchmarks {
    static func main() async {
        let args = CommandLine.arguments

        if args.contains("--ci") {
            let exitCode = await runCIMode()
            exit(exitCode)
        }

        if args.contains("--help") || args.contains("-h") {
            print("""
            Aether Performance Benchmarks

            Usage:
              AetherBenchmarks           Run full benchmarks with detailed output
              AetherBenchmarks --ci      Run CI mode with regression detection (exit 1 on failure)
              AetherBenchmarks --help    Show this help message

            CI Mode:
              - Compares results against established baselines
              - Fails (exit code 1) if any benchmark exceeds 20% regression
              - Outputs compact table suitable for CI logs

            Baselines:
              Based on M4, 24GB RAM, macOS 26.2
              Thresholds set at 120% of measured baseline values
            """)
            return
        }

        print("""

        ╔══════════════════════════════════════════════════════════════════╗
        ║                  AETHER PERFORMANCE BENCHMARKS                   ║
        ║                                                                  ║
        ║        Quantum Computing Simulator - Performance Metrics         ║
        ╚══════════════════════════════════════════════════════════════════╝

        System Information:
        """)

        let processInfo = ProcessInfo.processInfo
        print("  Physical Memory: \(processInfo.physicalMemory / (1024 * 1024 * 1024)) GB")
        print("  Active Processors: \(processInfo.activeProcessorCount)")
        print("  OS Version: \(processInfo.operatingSystemVersionString)")
        print("  Date: \(Date())")

        var allResults = [BenchmarkResult]()

        await allResults.append(contentsOf: benchmarkCircuitExecution())
        await allResults.append(contentsOf: benchmarkSparseHamiltonian())
        await allResults.append(contentsOf: benchmarkMPSBatchEvaluator())
        await allResults.append(contentsOf: benchmarkQWCGrouping())
        await allResults.append(contentsOf: benchmarkVQE())
        await allResults.append(contentsOf: benchmarkQAOA())

        print("\n" + String(repeating: "=", count: 68))
        print("BENCHMARK SUMMARY")
        print(String(repeating: "=", count: 68))
        print("Benchmark".padding(toLength: 55, withPad: " ", startingAt: 0) + "Mean (ms)")
        print(String(repeating: "-", count: 68))

        for result in allResults {
            let name = result.name.padding(toLength: 55, withPad: " ", startingAt: 0)
            let time = String(format: "%12.3f", result.meanTimeMs)
            print(name + time)
        }

        print(String(repeating: "=", count: 68))
        print("\nBenchmarks completed successfully.")
        print("Use these numbers as baseline for performance regression detection.")
        print("\nTo run in CI mode: AetherBenchmarks --ci")
    }
}
