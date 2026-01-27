// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import GameplayKit
import Testing

/// Deterministic random number generator for reproducible property-based tests.
/// Wraps GKMersenneTwisterRandomSource for high-quality random sequences
/// with seedable state for test reproducibility.
struct SeededRandomGenerator: RandomNumberGenerator {
    private let source: GKMersenneTwisterRandomSource

    init(seed: UInt64) {
        source = GKMersenneTwisterRandomSource(seed: seed)
    }

    mutating func next() -> UInt64 {
        let high = UInt64(bitPattern: Int64(source.nextInt()))
        let low = UInt64(bitPattern: Int64(source.nextInt()))
        return (high << 32) | (low & 0xFFFF_FFFF)
    }

    mutating func nextDouble() -> Double {
        Double(next() & 0x001F_FFFF_FFFF_FFFF) / Double(1 << 53)
    }

    mutating func nextDouble(in range: Range<Double>) -> Double {
        range.lowerBound + nextDouble() * (range.upperBound - range.lowerBound)
    }

    mutating func nextInt(in range: Range<Int>) -> Int {
        let span = UInt64(range.upperBound - range.lowerBound)
        return range.lowerBound + Int(next() % span)
    }

    mutating func nextInt(in range: ClosedRange<Int>) -> Int {
        nextInt(in: range.lowerBound ..< (range.upperBound + 1))
    }
}

/// Generates random quantum circuits for property-based testing.
/// Produces structurally valid circuits with configurable qubit count and depth.
/// Gate selection weighted toward common gates with proper qubit targeting.
enum RandomCircuitGenerator {
    private static let singleQubitGates: [QuantumGate] = [
        .identity, .pauliX, .pauliY, .pauliZ, .hadamard, .sGate, .tGate, .sx, .sy,
    ]

    private static let twoQubitGates: [QuantumGate] = [
        .cnot, .cz, .cy, .ch, .swap, .sqrtSwap,
    ]

    static func generate(
        qubits: Int,
        depth: Int,
        rng: inout SeededRandomGenerator,
        includeParameterized: Bool = true,
        includeToffoli: Bool = true,
    ) -> QuantumCircuit {
        var circuit = QuantumCircuit(qubits: qubits)

        for _ in 0 ..< depth {
            let gateType = rng.nextInt(in: 0 ..< 100)

            if gateType < 50 {
                appendSingleQubitGate(
                    to: &circuit,
                    qubits: qubits,
                    rng: &rng,
                    includeParameterized: includeParameterized,
                )
            } else if gateType < 85 {
                if qubits >= 2 {
                    appendTwoQubitGate(
                        to: &circuit,
                        qubits: qubits,
                        rng: &rng,
                        includeParameterized: includeParameterized,
                    )
                } else {
                    appendSingleQubitGate(
                        to: &circuit,
                        qubits: qubits,
                        rng: &rng,
                        includeParameterized: includeParameterized,
                    )
                }
            } else {
                if includeToffoli, qubits >= 3 {
                    appendToffoliGate(to: &circuit, qubits: qubits, rng: &rng)
                } else if qubits >= 2 {
                    appendTwoQubitGate(
                        to: &circuit,
                        qubits: qubits,
                        rng: &rng,
                        includeParameterized: includeParameterized,
                    )
                } else {
                    appendSingleQubitGate(
                        to: &circuit,
                        qubits: qubits,
                        rng: &rng,
                        includeParameterized: includeParameterized,
                    )
                }
            }
        }

        return circuit
    }

    private static func appendSingleQubitGate(
        to circuit: inout QuantumCircuit,
        qubits: Int,
        rng: inout SeededRandomGenerator,
        includeParameterized: Bool,
    ) {
        let target = rng.nextInt(in: 0 ..< qubits)

        if includeParameterized, rng.nextInt(in: 0 ..< 100) < 40 {
            let angle = rng.nextDouble(in: -Double.pi ..< Double.pi)
            let rotationType = rng.nextInt(in: 0 ..< 4)
            let gate: QuantumGate = switch rotationType {
            case 0: .rotationX(angle)
            case 1: .rotationY(angle)
            case 2: .rotationZ(angle)
            default: .phase(angle)
            }
            circuit.append(gate, to: target)
        } else {
            let gate = singleQubitGates[rng.nextInt(in: 0 ..< singleQubitGates.count)]
            circuit.append(gate, to: target)
        }
    }

    private static func appendTwoQubitGate(
        to circuit: inout QuantumCircuit,
        qubits: Int,
        rng: inout SeededRandomGenerator,
        includeParameterized: Bool,
    ) {
        let q1 = rng.nextInt(in: 0 ..< qubits)
        var q2 = rng.nextInt(in: 0 ..< qubits)
        while q2 == q1 {
            q2 = rng.nextInt(in: 0 ..< qubits)
        }

        if includeParameterized, rng.nextInt(in: 0 ..< 100) < 30 {
            let angle = rng.nextDouble(in: -Double.pi ..< Double.pi)
            let rotationType = rng.nextInt(in: 0 ..< 4)
            let gate: QuantumGate = switch rotationType {
            case 0: .controlledRotationX(angle)
            case 1: .controlledRotationY(angle)
            case 2: .controlledRotationZ(angle)
            default: .controlledPhase(angle)
            }
            circuit.append(gate, to: [q1, q2])
        } else {
            let gate = twoQubitGates[rng.nextInt(in: 0 ..< twoQubitGates.count)]
            circuit.append(gate, to: [q1, q2])
        }
    }

    private static func appendToffoliGate(
        to circuit: inout QuantumCircuit,
        qubits: Int,
        rng: inout SeededRandomGenerator,
    ) {
        var indices = [Int]()
        while indices.count < 3 {
            let q = rng.nextInt(in: 0 ..< qubits)
            if !indices.contains(q) {
                indices.append(q)
            }
        }
        circuit.append(.toffoli, to: indices)
    }

    static func generateRandomUnitary2x2(rng: inout SeededRandomGenerator) -> [[Complex<Double>]] {
        let theta = rng.nextDouble(in: 0 ..< Double.pi)
        let phi = rng.nextDouble(in: 0 ..< 2 * Double.pi)
        let lambda = rng.nextDouble(in: 0 ..< 2 * Double.pi)

        let cosHalf = cos(theta / 2)
        let sinHalf = sin(theta / 2)

        return [
            [
                Complex(cosHalf, 0),
                Complex(-sinHalf * cos(lambda), -sinHalf * sin(lambda)),
            ],
            [
                Complex(sinHalf * cos(phi), sinHalf * sin(phi)),
                Complex(cosHalf * cos(phi + lambda), cosHalf * sin(phi + lambda)),
            ],
        ]
    }
}

/// Test suite for quantum circuit unitarity invariants.
/// Validates U†U = I for randomly generated circuits,
/// ensuring gate composition preserves quantum mechanics correctness.
@Suite("Unitarity Property Tests")
struct UnitarityPropertyTests {
    private static let tolerance: Double = 1e-10

    @Test("Random 2-qubit circuits preserve unitarity", arguments: 1 ... 100)
    func twoQubitUnitarity(seed: Int) {
        var rng = SeededRandomGenerator(seed: UInt64(seed))
        let circuit = RandomCircuitGenerator.generate(
            qubits: 2,
            depth: rng.nextInt(in: 5 ... 20),
            rng: &rng,
        )

        let unitary = CircuitUnitary.unitary(for: circuit)
        let isUnitary = validateUnitarity(unitary)

        #expect(isUnitary, "Circuit with seed \(seed) failed unitarity: U†U ≠ I")
    }

    @Test("Random 3-qubit circuits preserve unitarity", arguments: 1 ... 100)
    func threeQubitUnitarity(seed: Int) {
        var rng = SeededRandomGenerator(seed: UInt64(seed))
        let circuit = RandomCircuitGenerator.generate(
            qubits: 3,
            depth: rng.nextInt(in: 5 ... 15),
            rng: &rng,
        )

        let unitary = CircuitUnitary.unitary(for: circuit)
        let isUnitary = validateUnitarity(unitary)

        #expect(isUnitary, "Circuit with seed \(seed) failed unitarity: U†U ≠ I")
    }

    @Test("Random 4-qubit circuits preserve unitarity", arguments: 1 ... 50)
    func fourQubitUnitarity(seed: Int) {
        var rng = SeededRandomGenerator(seed: UInt64(seed))
        let circuit = RandomCircuitGenerator.generate(
            qubits: 4,
            depth: rng.nextInt(in: 5 ... 12),
            rng: &rng,
        )

        let unitary = CircuitUnitary.unitary(for: circuit)
        let isUnitary = validateUnitarity(unitary)

        #expect(isUnitary, "Circuit with seed \(seed) failed unitarity: U†U ≠ I")
    }

    @Test("Random 5-qubit circuits preserve unitarity", arguments: 1 ... 30)
    func fiveQubitUnitarity(seed: Int) {
        var rng = SeededRandomGenerator(seed: UInt64(seed))
        let circuit = RandomCircuitGenerator.generate(
            qubits: 5,
            depth: rng.nextInt(in: 5 ... 10),
            rng: &rng,
        )

        let unitary = CircuitUnitary.unitary(for: circuit)
        let isUnitary = validateUnitarity(unitary)

        #expect(isUnitary, "Circuit with seed \(seed) failed unitarity: U†U ≠ I")
    }

    @Test("Deep circuits preserve unitarity", arguments: 1 ... 20)
    func deepCircuitUnitarity(seed: Int) {
        var rng = SeededRandomGenerator(seed: UInt64(seed))
        let circuit = RandomCircuitGenerator.generate(
            qubits: 3,
            depth: 50 + rng.nextInt(in: 0 ... 50),
            rng: &rng,
        )

        let unitary = CircuitUnitary.unitary(for: circuit)
        let isUnitary = validateUnitarity(unitary)

        #expect(isUnitary, "Deep circuit with seed \(seed) failed unitarity")
    }

    @Test("Custom unitary gates preserve unitarity", arguments: 1 ... 50)
    func customGateUnitarity(seed: Int) {
        var rng = SeededRandomGenerator(seed: UInt64(seed))
        let customMatrix = RandomCircuitGenerator.generateRandomUnitary2x2(rng: &rng)

        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.customSingleQubit(matrix: customMatrix), to: 0)
        circuit.append(.hadamard, to: 1)
        circuit.append(.cnot, to: [0, 1])

        let unitary = CircuitUnitary.unitary(for: circuit)
        let isUnitary = validateUnitarity(unitary)

        #expect(isUnitary, "Custom gate circuit with seed \(seed) failed unitarity")
    }

    private func validateUnitarity(_ matrix: [[Complex<Double>]]) -> Bool {
        let dim = matrix.count
        let conjugateTranspose = MatrixUtilities.hermitianConjugate(matrix)
        let product = MatrixUtilities.matrixMultiply(conjugateTranspose, matrix)

        for i in 0 ..< dim {
            for j in 0 ..< dim {
                let expected = i == j ? 1.0 : 0.0
                if abs(product[i][j].real - expected) > Self.tolerance {
                    return false
                }
                if abs(product[i][j].imaginary) > Self.tolerance {
                    return false
                }
            }
        }
        return true
    }
}

/// Test suite for state normalization invariants.
/// Validates Σ|cᵢ|² = 1 after circuit execution,
/// ensuring probability conservation through gate operations.
@Suite("Normalization Property Tests")
struct NormalizationPropertyTests {
    private static let tolerance: Double = 1e-10

    @Test("Random circuits preserve normalization", arguments: 1 ... 100)
    func circuitPreservesNormalization(seed: Int) {
        var rng = SeededRandomGenerator(seed: UInt64(seed))
        let qubits = rng.nextInt(in: 2 ... 5)
        let depth = rng.nextInt(in: 5 ... 30)

        let circuit = RandomCircuitGenerator.generate(
            qubits: qubits,
            depth: depth,
            rng: &rng,
        )

        let initialState = QuantumState(qubits: qubits)
        let finalState = circuit.execute(on: initialState)

        #expect(
            finalState.isNormalized(),
            "Circuit (qubits=\(qubits), depth=\(depth), seed=\(seed)) violated normalization",
        )
    }

    @Test("Random initial states remain normalized after execution", arguments: 1 ... 50)
    func randomInitialStateNormalization(seed: Int) {
        var rng = SeededRandomGenerator(seed: UInt64(seed))
        let qubits = rng.nextInt(in: 2 ... 4)
        let stateSize = 1 << qubits

        var amplitudes = [Complex<Double>]()
        var normSquared = 0.0
        for _ in 0 ..< stateSize {
            let real = rng.nextDouble(in: -1 ..< 1)
            let imag = rng.nextDouble(in: -1 ..< 1)
            amplitudes.append(Complex(real, imag))
            normSquared += real * real + imag * imag
        }

        let norm = sqrt(normSquared)
        for i in 0 ..< stateSize {
            amplitudes[i] = Complex(
                amplitudes[i].real / norm,
                amplitudes[i].imaginary / norm,
            )
        }

        let initialState = QuantumState(qubits: qubits, amplitudes: amplitudes)
        let circuit = RandomCircuitGenerator.generate(
            qubits: qubits,
            depth: rng.nextInt(in: 5 ... 20),
            rng: &rng,
        )

        let finalState = circuit.execute(on: initialState)

        #expect(
            finalState.isNormalized(),
            "Random initial state (seed=\(seed)) lost normalization",
        )
    }

    @Test("Sequential gate application preserves normalization", arguments: 1 ... 50)
    func sequentialGateNormalization(seed: Int) {
        var rng = SeededRandomGenerator(seed: UInt64(seed))
        let qubits = rng.nextInt(in: 2 ... 4)
        let depth = rng.nextInt(in: 10 ... 30)

        let circuit = RandomCircuitGenerator.generate(
            qubits: qubits,
            depth: depth,
            rng: &rng,
        )

        var state = QuantumState(qubits: qubits)

        for i in 0 ..< circuit.count {
            let operation = circuit.operations[i]
            state = GateApplication.apply(operation, state: state)

            #expect(
                state.isNormalized(),
                "Normalization violated at gate \(i) (seed=\(seed))",
            )
        }
    }
}

/// Test suite for circuit execution consistency.
/// Validates that different execution paths produce identical results,
/// ensuring implementation correctness across CPU and unitary-based evaluation.
@Suite("Execution Consistency Property Tests")
struct ExecutionConsistencyPropertyTests {
    private static let tolerance: Double = 1e-9

    @Test("Gate-by-gate matches unitary execution", arguments: 1 ... 50)
    func gateByGateMatchesUnitary(seed: Int) {
        var rng = SeededRandomGenerator(seed: UInt64(seed))
        let qubits = rng.nextInt(in: 2 ... 4)
        let depth = rng.nextInt(in: 3 ... 15)

        let circuit = RandomCircuitGenerator.generate(
            qubits: qubits,
            depth: depth,
            rng: &rng,
            includeToffoli: qubits >= 3,
        )

        let gateByGateResult = circuit.execute()
        let unitary = CircuitUnitary.unitary(for: circuit)
        let unitaryResult = applyUnitary(unitary, to: QuantumState(qubits: qubits))

        for i in 0 ..< (1 << qubits) {
            let diff = abs(gateByGateResult.amplitudes[i].real - unitaryResult.amplitudes[i].real)
                + abs(gateByGateResult.amplitudes[i].imaginary - unitaryResult.amplitudes[i].imaginary)

            #expect(
                diff < Self.tolerance,
                "Execution mismatch at index \(i) (seed=\(seed)): gate-by-gate vs unitary",
            )
        }
    }

    @Test("Circuit execution is deterministic", arguments: 1 ... 30)
    func executionDeterminism(seed: Int) {
        var rng = SeededRandomGenerator(seed: UInt64(seed))
        let qubits = rng.nextInt(in: 2 ... 4)
        let depth = rng.nextInt(in: 5 ... 20)

        let circuit = RandomCircuitGenerator.generate(
            qubits: qubits,
            depth: depth,
            rng: &rng,
        )

        let result1 = circuit.execute()
        let result2 = circuit.execute()

        for i in 0 ..< (1 << qubits) {
            #expect(
                result1.amplitudes[i] == result2.amplitudes[i],
                "Non-deterministic execution at index \(i) (seed=\(seed))",
            )
        }
    }

    private func applyUnitary(
        _ unitary: [[Complex<Double>]],
        to state: QuantumState,
    ) -> QuantumState {
        let dim = unitary.count
        var newAmplitudes = [Complex<Double>](repeating: .zero, count: dim)

        for i in 0 ..< dim {
            var sum = Complex<Double>.zero
            for j in 0 ..< dim {
                let product = Complex(
                    unitary[i][j].real * state.amplitudes[j].real
                        - unitary[i][j].imaginary * state.amplitudes[j].imaginary,
                    unitary[i][j].real * state.amplitudes[j].imaginary
                        + unitary[i][j].imaginary * state.amplitudes[j].real,
                )
                sum = Complex(sum.real + product.real, sum.imaginary + product.imaginary)
            }
            newAmplitudes[i] = sum
        }

        return QuantumState(qubits: state.qubits, amplitudes: newAmplitudes)
    }
}

/// Test suite for measurement invariants.
/// Validates Born rule probability distribution and
/// post-measurement state collapse properties.
@Suite("Measurement Property Tests")
struct MeasurementPropertyTests {
    @Test("Measurement probabilities sum to 1", arguments: 1 ... 50)
    func probabilitiesSumToOne(seed: Int) {
        var rng = SeededRandomGenerator(seed: UInt64(seed))
        let qubits = rng.nextInt(in: 2 ... 4)
        let depth = rng.nextInt(in: 5 ... 20)

        let circuit = RandomCircuitGenerator.generate(
            qubits: qubits,
            depth: depth,
            rng: &rng,
        )

        let state = circuit.execute()
        let probabilities = state.probabilities()
        let sum = probabilities.reduce(0.0, +)

        #expect(
            abs(sum - 1.0) < 1e-10,
            "Probabilities sum to \(sum) instead of 1.0 (seed=\(seed))",
        )
    }

    @Test("Measurement collapse produces basis state", arguments: 1 ... 50)
    func measurementCollapseToBasis(seed: Int) {
        var rng = SeededRandomGenerator(seed: UInt64(seed))
        let qubits = rng.nextInt(in: 2 ... 4)
        let depth = rng.nextInt(in: 5 ... 15)

        let circuit = RandomCircuitGenerator.generate(
            qubits: qubits,
            depth: depth,
            rng: &rng,
        )

        let state = circuit.execute()
        let result = Measurement.measure(state, seed: UInt64(seed))

        let outcomeProb = result.collapsedState.probability(of: result.outcome)
        #expect(
            abs(outcomeProb - 1.0) < 1e-10,
            "Collapsed state probability at outcome \(result.outcome) is \(outcomeProb) (seed=\(seed))",
        )

        for i in 0 ..< (1 << qubits) where i != result.outcome {
            let prob = result.collapsedState.probability(of: i)
            #expect(
                prob < 1e-10,
                "Non-zero probability \(prob) at index \(i) after collapse (seed=\(seed))",
            )
        }
    }

    @Test("Repeated measurement of collapsed state is deterministic", arguments: 1 ... 30)
    func collapsedStateMeasurementDeterministic(seed: Int) {
        var rng = SeededRandomGenerator(seed: UInt64(seed))
        let qubits = rng.nextInt(in: 2 ... 3)
        let depth = rng.nextInt(in: 5 ... 10)

        let circuit = RandomCircuitGenerator.generate(
            qubits: qubits,
            depth: depth,
            rng: &rng,
        )

        let state = circuit.execute()
        let firstResult = Measurement.measure(state, seed: UInt64(seed))

        for i in 0 ..< 10 {
            let repeatResult = Measurement.measure(
                firstResult.collapsedState,
                seed: UInt64(seed + 1000 + i),
            )
            #expect(
                repeatResult.outcome == firstResult.outcome,
                "Collapsed state gave different outcome on remeasurement (seed=\(seed))",
            )
        }
    }
}

/// Test suite for observable expectation value properties.
/// Validates linearity, bounds, and variance computation
/// for Hamiltonian expectation values.
@Suite("Observable Property Tests")
struct ObservablePropertyTests {
    @Test("Pauli Z expectation bounded by [-1, 1]", arguments: 1 ... 50)
    func pauliZExpectationBounded(seed: Int) {
        var rng = SeededRandomGenerator(seed: UInt64(seed))
        let qubits = rng.nextInt(in: 1 ... 4)
        let depth = rng.nextInt(in: 5 ... 20)

        let circuit = RandomCircuitGenerator.generate(
            qubits: qubits,
            depth: depth,
            rng: &rng,
        )

        let state = circuit.execute()
        let targetQubit = rng.nextInt(in: 0 ..< qubits)
        let observable = Observable.pauliZ(qubit: targetQubit)
        let expectation = observable.expectationValue(of: state)

        #expect(
            expectation >= -1.0 - 1e-10 && expectation <= 1.0 + 1e-10,
            "Pauli Z expectation \(expectation) out of bounds (seed=\(seed))",
        )
    }

    @Test("Observable linearity: c*O has c*⟨O⟩", arguments: 1 ... 30)
    func observableLinearity(seed: Int) {
        var rng = SeededRandomGenerator(seed: UInt64(seed))
        let qubits = rng.nextInt(in: 2 ... 3)
        let depth = rng.nextInt(in: 5 ... 15)

        let circuit = RandomCircuitGenerator.generate(
            qubits: qubits,
            depth: depth,
            rng: &rng,
        )

        let state = circuit.execute()
        let coefficient = rng.nextDouble(in: -5.0 ..< 5.0)

        let obs1 = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let obs2 = Observable(coefficient: coefficient, pauliString: PauliString(.z(0)))

        let exp1 = obs1.expectationValue(of: state)
        let exp2 = obs2.expectationValue(of: state)

        #expect(
            abs(exp2 - coefficient * exp1) < 1e-10,
            "Linearity violated: \(exp2) ≠ \(coefficient) * \(exp1) (seed=\(seed))",
        )
    }

    @Test("Variance is non-negative", arguments: 1 ... 30)
    func varianceNonNegative(seed: Int) {
        var rng = SeededRandomGenerator(seed: UInt64(seed))
        let qubits = rng.nextInt(in: 2 ... 3)
        let depth = rng.nextInt(in: 5 ... 15)

        let circuit = RandomCircuitGenerator.generate(
            qubits: qubits,
            depth: depth,
            rng: &rng,
        )

        let state = circuit.execute()
        let observable = Observable(terms: [
            (1.0, PauliString(.z(0))),
            (0.5, PauliString(.x(1))),
        ])

        let variance = observable.variance(of: state)

        #expect(
            variance >= -1e-10,
            "Variance \(variance) is negative (seed=\(seed))",
        )
    }
}

/// Test suite for quantum gate algebraic identities.
/// Validates mathematical properties like self-inverse gates,
/// commutation relations, and decomposition equivalences.
@Suite("Gate Identity Property Tests")
struct GateIdentityPropertyTests {
    private static let tolerance: Double = 1e-10

    @Test("Self-inverse gates: G² = I", arguments: 1 ... 20)
    func selfInverseGates(seed _: Int) {
        let selfInverseGates: [QuantumGate] = [
            .pauliX, .pauliY, .pauliZ, .hadamard, .cnot, .cz, .swap, .toffoli,
        ]

        for gate in selfInverseGates {
            let matrix = gate.matrix()
            let squared = MatrixUtilities.matrixMultiply(matrix, matrix)

            #expect(
                QuantumGate.isIdentityMatrix(squared, tolerance: Self.tolerance),
                "\(gate) is not self-inverse",
            )
        }
    }

    @Test("Rotation by 2π equals identity", arguments: 1 ... 10)
    func fullRotationIsIdentity(seed _: Int) {
        let rxFull = QuantumGate.rotationX(2 * .pi).matrix()
        let ryFull = QuantumGate.rotationY(2 * .pi).matrix()
        let rzFull = QuantumGate.rotationZ(2 * .pi).matrix()

        for matrix in [rxFull, ryFull, rzFull] {
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    let expectedMag = i == j ? 1.0 : 0.0
                    #expect(
                        abs(matrix[i][j].magnitude - expectedMag) < Self.tolerance,
                        "Full rotation magnitude mismatch",
                    )
                }
            }
        }
    }

    @Test("HZH = X (Hadamard conjugation)", arguments: 1 ... 5)
    func hadamardConjugation(seed _: Int) {
        let h = QuantumGate.hadamard.matrix()
        let z = QuantumGate.pauliZ.matrix()
        let x = QuantumGate.pauliX.matrix()

        let hz = MatrixUtilities.matrixMultiply(h, z)
        let hzh = MatrixUtilities.matrixMultiply(hz, h)

        #expect(
            QuantumGate.matricesEqual(hzh, x, tolerance: Self.tolerance),
            "HZH ≠ X",
        )
    }

    @Test("Pauli algebra: XY = iZ", arguments: 1 ... 5)
    func pauliAlgebra(seed _: Int) {
        let x = QuantumGate.pauliX.matrix()
        let y = QuantumGate.pauliY.matrix()

        let xy = MatrixUtilities.matrixMultiply(x, y)

        let iZ: [[Complex<Double>]] = [
            [Complex(0, 1), .zero],
            [.zero, Complex(0, -1)],
        ]

        #expect(
            QuantumGate.matricesEqual(xy, iZ, tolerance: Self.tolerance),
            "XY ≠ iZ",
        )
    }
}
