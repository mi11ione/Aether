// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Test suite for the Deutsch–Jozsa algorithm.
/// Validates constant vs balanced oracle detection with a single query,
/// including oracle helpers and measurement-based result interpretation.
@Suite("Deutsch-Jozsa Algorithm")
struct DeutschJozsaTests {
    @Test("Constant-0 oracle: All inputs measured as |0⟩")
    func testConstantZeroOracle() {
        let oracle = QuantumCircuit.constantZeroOracle()
        let circuit = QuantumCircuit.deutschJozsa(qubits: 3, oracle: oracle)
        let state = circuit.execute()

        let inputQubits = [0, 1, 2]
        #expect(state.allQubitsAreZero(inputQubits),
                "Constant-0 oracle should result in all input qubits measuring |0⟩")
    }

    @Test("Constant-1 oracle: All inputs measured as |0⟩")
    func testConstantOneOracle() {
        let oracle = QuantumCircuit.constantOneOracle()
        let circuit = QuantumCircuit.deutschJozsa(qubits: 3, oracle: oracle)
        let state = circuit.execute()

        let inputQubits = [0, 1, 2]
        #expect(state.allQubitsAreZero(inputQubits),
                "Constant-1 oracle should result in all input qubits measuring |0⟩")
    }

    @Test("Balanced parity oracle: At least one input is |1⟩")
    func testBalancedParityOracle() {
        let oracle = QuantumCircuit.balancedParityOracle()
        let circuit = QuantumCircuit.deutschJozsa(qubits: 3, oracle: oracle)
        let state = circuit.execute()

        let inputQubits = [0, 1, 2]
        let allZero = state.allQubitsAreZero(inputQubits)
        #expect(!allZero,
                "Balanced oracle should result in at least one input qubit measuring |1⟩")
    }

    @Test("Balanced first-bit oracle: At least one input is |1⟩")
    func testBalancedFirstBitOracle() {
        let oracle = QuantumCircuit.balancedFirstBitOracle()
        let circuit = QuantumCircuit.deutschJozsa(qubits: 3, oracle: oracle)
        let state = circuit.execute()

        let inputQubits = [0, 1, 2]
        let allZero = state.allQubitsAreZero(inputQubits)
        #expect(!allZero,
                "Balanced oracle should result in at least one input qubit measuring |1⟩")
    }

    @Test("Single qubit Deutsch-Jozsa with constant oracle")
    func singleQubitConstant() {
        let oracle = QuantumCircuit.constantZeroOracle()
        let circuit = QuantumCircuit.deutschJozsa(qubits: 1, oracle: oracle)
        let state = circuit.execute()

        #expect(state.allQubitsAreZero([0]),
                "Single-qubit constant oracle should measure |0⟩")
    }

    @Test("Deutsch-Jozsa with 5 input qubits")
    func largerInput() {
        let oracle = QuantumCircuit.constantZeroOracle()
        let circuit = QuantumCircuit.deutschJozsa(qubits: 5, oracle: oracle)
        let state = circuit.execute()

        let inputQubits = Array(0 ..< 5)
        #expect(state.allQubitsAreZero(inputQubits),
                "Constant oracle with 5 qubits should measure all |0⟩")
    }

    @Test("Balanced vs constant oracles produce different results")
    func balancedVsConstant() {
        let constantOracle = QuantumCircuit.constantZeroOracle()
        let constantCircuit = QuantumCircuit.deutschJozsa(
            qubits: 3,
            oracle: constantOracle,
        )
        let constantState = constantCircuit.execute()

        let balancedOracle = QuantumCircuit.balancedParityOracle()
        let balancedCircuit = QuantumCircuit.deutschJozsa(
            qubits: 3,
            oracle: balancedOracle,
        )
        let balancedState = balancedCircuit.execute()

        let inputQubits = [0, 1, 2]
        let constantAllZero = constantState.allQubitsAreZero(inputQubits)
        let balancedAllZero = balancedState.allQubitsAreZero(inputQubits)

        #expect(constantAllZero, "Constant oracle should give all zeros")
        #expect(!balancedAllZero, "Balanced oracle should give at least one 1")
    }
}

/// Test suite for the Bernstein–Vazirani algorithm.
/// Verifies single-query hidden string recovery using oracle construction
/// and Hadamard pre/post processing across varying input sizes.
@Suite("Bernstein-Vazirani Algorithm")
struct BernsteinVaziraniTests {
    @Test("Hidden string [1]: Single bit recovery")
    func singleBitHiddenString() {
        let hiddenString = [1]
        let oracle = QuantumCircuit.bernsteinVaziraniOracle(hiddenString: hiddenString)
        let circuit = QuantumCircuit.bernsteinVazirani(qubits: 1, oracle: oracle)
        let state = circuit.execute()

        let measured = state.measureQubits([0])
        #expect(measured == hiddenString,
                "BV should recover single-bit hidden string [1]")
    }

    @Test("Hidden string [0]: Zero string recovery")
    func zeroHiddenString() {
        let hiddenString = [0]
        let oracle = QuantumCircuit.bernsteinVaziraniOracle(hiddenString: hiddenString)
        let circuit = QuantumCircuit.bernsteinVazirani(qubits: 1, oracle: oracle)
        let state = circuit.execute()

        let measured = state.measureQubits([0])
        #expect(measured == hiddenString,
                "BV should recover hidden string [0]")
    }

    @Test("Hidden string [1,0,1]: 3-bit recovery")
    func threeBitHiddenString() {
        let hiddenString = [1, 0, 1]
        let oracle = QuantumCircuit.bernsteinVaziraniOracle(hiddenString: hiddenString)
        let circuit = QuantumCircuit.bernsteinVazirani(qubits: 3, oracle: oracle)
        let state = circuit.execute()

        let inputQubits = [0, 1, 2]
        let measured = state.measureQubits(inputQubits)
        #expect(measured == hiddenString,
                "BV should recover hidden string [1,0,1]")
    }

    @Test("Hidden string [1,1,1,1]: All ones")
    func allOnesHiddenString() {
        let hiddenString = [1, 1, 1, 1]
        let oracle = QuantumCircuit.bernsteinVaziraniOracle(hiddenString: hiddenString)
        let circuit = QuantumCircuit.bernsteinVazirani(qubits: 4, oracle: oracle)
        let state = circuit.execute()

        let inputQubits = [0, 1, 2, 3]
        let measured = state.measureQubits(inputQubits)
        #expect(measured == hiddenString,
                "BV should recover hidden string [1,1,1,1]")
    }

    @Test("Hidden string [0,0,0,0]: All zeros")
    func allZerosHiddenString() {
        let hiddenString = [0, 0, 0, 0]
        let oracle = QuantumCircuit.bernsteinVaziraniOracle(hiddenString: hiddenString)
        let circuit = QuantumCircuit.bernsteinVazirani(qubits: 4, oracle: oracle)
        let state = circuit.execute()

        let inputQubits = [0, 1, 2, 3]
        let measured = state.measureQubits(inputQubits)
        #expect(measured == hiddenString,
                "BV should recover hidden string [0,0,0,0]")
    }

    @Test("Hidden string [1,0,1,0,1]: Alternating pattern")
    func alternatingHiddenString() {
        let hiddenString = [1, 0, 1, 0, 1]
        let oracle = QuantumCircuit.bernsteinVaziraniOracle(hiddenString: hiddenString)
        let circuit = QuantumCircuit.bernsteinVazirani(qubits: 5, oracle: oracle)
        let state = circuit.execute()

        let inputQubits = [0, 1, 2, 3, 4]
        let measured = state.measureQubits(inputQubits)
        #expect(measured == hiddenString,
                "BV should recover hidden string [1,0,1,0,1]")
    }

    @Test("Hidden string [0,1,1,0,1,0]: 6-bit recovery")
    func sixBitHiddenString() {
        let hiddenString = [0, 1, 1, 0, 1, 0]
        let oracle = QuantumCircuit.bernsteinVaziraniOracle(hiddenString: hiddenString)
        let circuit = QuantumCircuit.bernsteinVazirani(qubits: 6, oracle: oracle)
        let state = circuit.execute()

        let inputQubits = Array(0 ..< 6)
        let measured = state.measureQubits(inputQubits)
        #expect(measured == hiddenString,
                "BV should recover hidden string [0,1,1,0,1,0]")
    }

    @Test("Hidden string: Random 8-bit pattern")
    func eightBitHiddenString() {
        let hiddenString = [1, 0, 0, 1, 1, 0, 1, 1]
        let oracle = QuantumCircuit.bernsteinVaziraniOracle(hiddenString: hiddenString)
        let circuit = QuantumCircuit.bernsteinVazirani(qubits: 8, oracle: oracle)
        let state = circuit.execute()

        let inputQubits = Array(0 ..< 8)
        let measured = state.measureQubits(inputQubits)
        #expect(measured == hiddenString,
                "BV should recover 8-bit hidden string")
    }
}

/// Test suite for Simon's algorithm iteration.
/// Ensures circuit structure prepares |+⟩^n|0⟩^n, applies the oracle, and
/// yields measurement outcomes orthogonal to the hidden period.
@Suite("Simon's Algorithm")
struct SimonTests {
    @Test("Period [1,1]: Orthogonality constraint y·s = 0")
    func period11() {
        let period = [1, 1]
        let oracle = QuantumCircuit.simonOracle(period: period)
        let circuit = QuantumCircuit.simonIteration(qubits: 2, oracle: oracle)
        let state = circuit.execute()

        let measured = state.measureQubits([0, 1])

        let dotProduct = (measured[0] + measured[1]) % 2
        #expect(dotProduct == 0,
                "Measured string should be orthogonal to period [1,1]")
    }

    @Test("Period [1,0]: Orthogonality y·s = 0")
    func period10() {
        let period = [1, 0]
        let oracle = QuantumCircuit.simonOracle(period: period)
        let circuit = QuantumCircuit.simonIteration(qubits: 2, oracle: oracle)
        let state = circuit.execute()

        let measured = state.measureQubits([0, 1])

        #expect(measured[0] == 0,
                "First bit should be 0 for period [1,0]")
    }

    @Test("Period [0,1]: Orthogonality y·s = 0")
    func period01() {
        let period = [0, 1]
        let oracle = QuantumCircuit.simonOracle(period: period)
        let circuit = QuantumCircuit.simonIteration(qubits: 2, oracle: oracle)
        let state = circuit.execute()

        let measured = state.measureQubits([0, 1])

        #expect(measured[1] == 0,
                "Second bit should be 0 for period [0,1]")
    }

    @Test("Period [1,0,1]: Three-qubit orthogonality")
    func period101() {
        let period = [1, 0, 1]
        let oracle = QuantumCircuit.simonOracle(period: period)
        let circuit = QuantumCircuit.simonIteration(qubits: 3, oracle: oracle)
        let state = circuit.execute()

        let measured = state.measureQubits([0, 1, 2])

        let dotProduct = (measured[0] + measured[2]) % 2
        #expect(dotProduct == 0,
                "Measured string should be orthogonal to period [1,0,1]")
    }

    @Test("Period [1,1,0]: Three-qubit orthogonality")
    func period110() {
        let period = [1, 1, 0]
        let oracle = QuantumCircuit.simonOracle(period: period)
        let circuit = QuantumCircuit.simonIteration(qubits: 3, oracle: oracle)
        let state = circuit.execute()

        let measured = state.measureQubits([0, 1, 2])

        let dotProduct = (measured[0] + measured[1]) % 2
        #expect(dotProduct == 0,
                "Measured string should be orthogonal to period [1,1,0]")
    }

    @Test("Period [1,1,1,1]: Four-qubit all-ones")
    func period1111() {
        let period = [1, 1, 1, 1]
        let oracle = QuantumCircuit.simonOracle(period: period)
        let circuit = QuantumCircuit.simonIteration(qubits: 4, oracle: oracle)
        let state = circuit.execute()

        let measured = state.measureQubits([0, 1, 2, 3])

        let dotProduct = measured.reduce(0, +) % 2
        #expect(dotProduct == 0,
                "Measured string should have even parity for period [1,1,1,1]")
    }

    @Test("Simon iteration produces normalized state")
    func simonProducesValidState() {
        let period = [1, 0, 1]
        let oracle = QuantumCircuit.simonOracle(period: period)
        let circuit = QuantumCircuit.simonIteration(qubits: 3, oracle: oracle)
        let state = circuit.execute()

        #expect(state.isNormalized(),
                "Simon circuit should produce normalized quantum state")
    }
}

/// Test suite for educational oracle builders.
/// Validates constant, balanced, and parameterized oracles used by
/// Deutsch–Jozsa, Bernstein–Vazirani, and Simon circuits.
@Suite("Oracle Builder")
struct OracleBuilderTests {
    @Test("Constant-zero oracle leaves output unchanged")
    func constantZeroOracleIdentity() {
        var circuit = QuantumCircuit(qubits: 2)
        let oracle = QuantumCircuit.constantZeroOracle()

        oracle([0], 1, &circuit)
        let state = circuit.execute()

        #expect(state.probability(of: 0) > 0.99,
                "Constant-zero oracle should not change state")
    }

    @Test("Constant-one oracle flips output")
    func constantOneOracleFlips() {
        var circuit = QuantumCircuit(qubits: 2)
        let oracle = QuantumCircuit.constantOneOracle()

        oracle([0], 1, &circuit)
        let state = circuit.execute()

        #expect(state.probability(of: 0b10) > 0.99,
                "Constant-one oracle should flip output to |1⟩")
    }

    @Test("Parity oracle computes XOR correctly")
    func parityOracleXOR() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 1)

        let oracle = QuantumCircuit.balancedParityOracle()
        oracle([0, 1], 2, &circuit)

        let state = circuit.execute()

        #expect(state.probability(of: 0b011) > 0.99,
                "Parity oracle should compute XOR correctly")
    }

    @Test("BV oracle [1,0] computes dot product")
    func bVOracleComputesDotProduct() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.pauliX, to: 0)

        let oracle = QuantumCircuit.bernsteinVaziraniOracle(hiddenString: [1, 0])
        oracle([0, 1], 2, &circuit)

        let state = circuit.execute()

        #expect(state.probability(of: 0b101) > 0.99,
                "BV oracle should compute dot product: [1,0]·[1,0]=1")
    }

    @Test("Simon oracle satisfies periodicity")
    func simonOraclePeriodicProperty() {
        let period = [1, 0]
        let oracle = QuantumCircuit.simonOracle(period: period)

        var circuit1 = QuantumCircuit(qubits: 3)
        oracle([0, 1], 2, &circuit1)
        let state1 = circuit1.execute()

        var circuit2 = QuantumCircuit(qubits: 3)
        circuit2.append(.pauliX, to: 0)
        oracle([0, 1], 2, &circuit2)
        let state2 = circuit2.execute()

        let output1IsCorrect = state1.probability(of: 0b000) > 0.99
        let output2IsCorrect = state2.probability(of: 0b001) > 0.99

        #expect(output1IsCorrect && output2IsCorrect,
                "Simon oracle f(00)=f(10)=0 for period [1,0]")
    }
}

/// Test suite for algorithm edge cases and validations.
/// Verifies measurement helpers, normalization preservation across
/// educational algorithms, and correctness of zero-state checks.
@Suite("Algorithms Edge Cases and Validation")
struct AlgorithmsEdgeCasesTests {
    @Test("measureQubits returns valid 0/1 values")
    func measureQubitValidation() {
        let state = QuantumState(qubits: 3)
        let measured = state.measureQubits([0, 1, 2])

        #expect(measured.count == 3, "Should measure 3 qubits")
        #expect(measured.allSatisfy { $0 == 0 || $0 == 1 },
                "Measurements should be 0 or 1")
    }

    @Test("measureQubits handles empty qubit array")
    func measureEmptyQubitArray() {
        let state = QuantumState(qubits: 3)
        let measured = state.measureQubits([])

        #expect(measured.isEmpty, "Measuring empty array should return empty array")
    }

    @Test("allQubitsAreZero detects non-zero qubits")
    func allQubitsAreZeroDetection() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.pauliX, to: 0)
        let state = circuit.execute()

        #expect(!state.allQubitsAreZero([0]),
                "Should detect qubit 0 is not zero")
        #expect(state.allQubitsAreZero([1]),
                "Should detect qubit 1 is zero")
    }

    @Test("DJ algorithm preserves normalization")
    func dJPreservesNormalization() {
        let oracle = QuantumCircuit.balancedParityOracle()
        let circuit = QuantumCircuit.deutschJozsa(qubits: 4, oracle: oracle)
        let state = circuit.execute()

        #expect(state.isNormalized(),
                "DJ algorithm should preserve state normalization")
    }

    @Test("BV algorithm preserves normalization")
    func bVPreservesNormalization() {
        let oracle = QuantumCircuit.bernsteinVaziraniOracle(hiddenString: [1, 0, 1, 1])
        let circuit = QuantumCircuit.bernsteinVazirani(qubits: 4, oracle: oracle)
        let state = circuit.execute()

        #expect(state.isNormalized(),
                "BV algorithm should preserve state normalization")
    }

    @Test("Simon algorithm preserves normalization")
    func simonPreservesNormalization() {
        let oracle = QuantumCircuit.simonOracle(period: [1, 1, 0])
        let circuit = QuantumCircuit.simonIteration(qubits: 3, oracle: oracle)
        let state = circuit.execute()

        #expect(state.isNormalized(),
                "Simon algorithm should preserve state normalization")
    }

    @Test("Balanced first-bit oracle handles empty input qubits")
    func balancedFirstBitOracleEmptyInput() {
        let oracle = QuantumCircuit.balancedFirstBitOracle()
        var circuit = QuantumCircuit(qubits: 1)

        oracle([], 0, &circuit)

        #expect(circuit.count == 0, "Oracle should not add gates when inputQubits is empty")
    }
}
