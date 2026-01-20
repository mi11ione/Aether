// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Tests MPSTruncationStatistics struct for tracking truncation information.
/// Validates error accumulation, counting, and maximum error tracking.
/// Ensures statistics correctly aggregate across multiple truncation operations.
@Suite("MPS Truncation Statistics")
struct MPSTruncationStatisticsTests {
    @Test("Zero statistics has all zeros")
    func zeroStatistics() {
        let stats = MPSTruncationStatistics.zero
        #expect(stats.truncationCount == 0, "Zero statistics should have count 0")
        #expect(abs(stats.cumulativeError) < 1e-10, "Zero statistics should have cumulative error 0")
        #expect(abs(stats.maxSingleError) < 1e-10, "Zero statistics should have max single error 0")
    }

    @Test("Adding error increments count")
    func addingErrorIncrementsCount() {
        let stats = MPSTruncationStatistics.zero
        let updated = stats.adding(error: 0.001)
        #expect(updated.truncationCount == 1, "Adding error should increment truncation count to 1")
    }

    @Test("Adding error accumulates cumulative error")
    func addingErrorAccumulatesCumulative() {
        var stats = MPSTruncationStatistics.zero
        stats = stats.adding(error: 0.001)
        stats = stats.adding(error: 0.002)
        stats = stats.adding(error: 0.003)
        #expect(abs(stats.cumulativeError - 0.006) < 1e-10, "Cumulative error should be sum of all errors")
        #expect(stats.truncationCount == 3, "Truncation count should be 3 after 3 additions")
    }

    @Test("Max single error tracks maximum")
    func maxSingleErrorTracksMaximum() {
        var stats = MPSTruncationStatistics.zero
        stats = stats.adding(error: 0.001)
        stats = stats.adding(error: 0.005)
        stats = stats.adding(error: 0.002)
        #expect(abs(stats.maxSingleError - 0.005) < 1e-10, "Max single error should be the largest error added")
    }

    @Test("Description format is correct")
    func descriptionFormat() {
        let stats = MPSTruncationStatistics.zero.adding(error: 0.001)
        let desc = stats.description
        #expect(desc.contains("count=1"), "Description should contain count")
        #expect(desc.contains("cumulative="), "Description should contain cumulative")
        #expect(desc.contains("max="), "Description should contain max")
    }

    @Test("Equatable conformance works")
    func equatableConformance() {
        let stats1 = MPSTruncationStatistics.zero.adding(error: 0.001)
        let stats2 = MPSTruncationStatistics.zero.adding(error: 0.001)
        let stats3 = MPSTruncationStatistics.zero.adding(error: 0.002)
        #expect(stats1 == stats2, "Identical statistics should be equal")
        #expect(stats1 != stats3, "Different statistics should not be equal")
    }
}

/// Tests ground state initialization of MatrixProductState.
/// Validates that |00...0> has amplitude 1 at index 0 and 0 elsewhere.
/// Ensures proper normalization and probability distribution for ground state.
@Suite("MPS Ground State Initialization")
struct MPSGroundStateTests {
    @Test("Ground state has amplitude 1 at index 0")
    func groundStateAmplitudeAtZero() {
        let mps = MatrixProductState(qubits: 4)
        let amp = mps.amplitude(of: 0)
        #expect(abs(amp.real - 1.0) < 1e-10, "Ground state amplitude at index 0 should be 1.0")
        #expect(abs(amp.imaginary) < 1e-10, "Ground state amplitude at index 0 should have zero imaginary part")
    }

    @Test("Ground state has amplitude 0 at other indices")
    func groundStateAmplitudeAtOthers() {
        let mps = MatrixProductState(qubits: 3)
        for i in 1 ..< 8 {
            let amp = mps.amplitude(of: i)
            #expect(abs(amp.real) < 1e-10, "Ground state amplitude at index \(i) should be 0")
            #expect(abs(amp.imaginary) < 1e-10, "Ground state amplitude at index \(i) should have zero imaginary part")
        }
    }

    @Test("Ground state is normalized")
    func groundStateNormalized() {
        let mps = MatrixProductState(qubits: 5)
        #expect(mps.isNormalized(), "Ground state should be normalized")
    }

    @Test("Ground state probability at index 0 is 1")
    func groundStateProbabilityAtZero() {
        let mps = MatrixProductState(qubits: 4)
        let prob = mps.probability(of: 0)
        #expect(abs(prob - 1.0) < 1e-10, "Ground state probability at index 0 should be 1.0")
    }

    @Test("Ground state with different qubit counts")
    func groundStateVariousQubitCounts() {
        for n in [1, 2, 4, 8, 10] {
            let mps = MatrixProductState(qubits: n)
            #expect(mps.qubits == n, "MPS should have \(n) qubits")
            let amp = mps.amplitude(of: 0)
            #expect(abs(amp.real - 1.0) < 1e-10, "Ground state amplitude at index 0 should be 1.0 for \(n) qubits")
        }
    }
}

/// Tests basis state initialization of MatrixProductState.
/// Validates that |k> has amplitude 1 at index k and 0 elsewhere.
/// Ensures correct handling of arbitrary computational basis states.
@Suite("MPS Basis State Initialization")
struct MPSBasisStateTests {
    @Test("Basis state |5> has amplitude 1 at index 5")
    func basisStateAmplitudeAtIndex() {
        let mps = MatrixProductState(qubits: 4, basisState: 5)
        let amp = mps.amplitude(of: 5)
        #expect(abs(amp.real - 1.0) < 1e-10, "Basis state |5> should have amplitude 1.0 at index 5")
        #expect(abs(amp.imaginary) < 1e-10, "Basis state |5> should have zero imaginary part at index 5")
    }

    @Test("Basis state |5> has amplitude 0 at other indices")
    func basisStateAmplitudeAtOthers() {
        let mps = MatrixProductState(qubits: 4, basisState: 5)
        for i in 0 ..< 16 where i != 5 {
            let amp = mps.amplitude(of: i)
            #expect(abs(amp.real) < 1e-10, "Basis state |5> should have amplitude 0 at index \(i)")
            #expect(abs(amp.imaginary) < 1e-10, "Basis state |5> should have zero imaginary part at index \(i)")
        }
    }

    @Test("Basis state |0> is same as ground state")
    func basisStateZeroSameAsGround() {
        let mpsGround = MatrixProductState(qubits: 3)
        let mpsBasis = MatrixProductState(qubits: 3, basisState: 0)
        for i in 0 ..< 8 {
            let ampGround = mpsGround.amplitude(of: i)
            let ampBasis = mpsBasis.amplitude(of: i)
            #expect(abs(ampGround.real - ampBasis.real) < 1e-10, "Ground and basis state |0> should have same amplitude at index \(i)")
        }
    }

    @Test("Basis state |15> (all ones) is correct")
    func basisStateAllOnes() {
        let mps = MatrixProductState(qubits: 4, basisState: 15)
        let amp = mps.amplitude(of: 15)
        #expect(abs(amp.real - 1.0) < 1e-10, "Basis state |15> should have amplitude 1.0 at index 15")
        let probZero = mps.probability(of: 0)
        #expect(abs(probZero) < 1e-10, "Basis state |15> should have probability 0 at index 0")
    }

    @Test("Basis state is normalized")
    func basisStateNormalized() {
        let mps = MatrixProductState(qubits: 5, basisState: 17)
        #expect(mps.isNormalized(), "Basis state should be normalized")
    }
}

/// Tests conversion from QuantumState to MatrixProductState.
/// Validates that conversion preserves amplitudes for small quantum systems.
/// Ensures normalization and entanglement structure are maintained during conversion.
@Suite("MPS From QuantumState Conversion")
struct MPSFromQuantumStateTests {
    @Test("Conversion from ground state preserves amplitudes")
    func conversionFromGroundState() {
        let state = QuantumState(qubits: 3)
        let mps = MatrixProductState(from: state)
        for i in 0 ..< 8 {
            let stateAmp = state.amplitude(of: i)
            let mpsAmp = mps.amplitude(of: i)
            #expect(abs(stateAmp.real - mpsAmp.real) < 1e-10, "Amplitude real part should match at index \(i)")
            #expect(abs(stateAmp.imaginary - mpsAmp.imaginary) < 1e-10, "Amplitude imaginary part should match at index \(i)")
        }
    }

    @Test("Conversion from Bell state preserves amplitudes")
    func conversionFromBellState() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes: [Complex<Double>] = [
            Complex(invSqrt2, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
            Complex(invSqrt2, 0.0),
        ]
        let state = QuantumState(qubits: 2, amplitudes: amplitudes)
        let mps = MatrixProductState(from: state, maxBondDimension: 16)

        let mpsAmp00 = mps.amplitude(of: 0)
        let mpsAmp11 = mps.amplitude(of: 3)
        #expect(abs(mpsAmp00.real - invSqrt2) < 1e-10, "Bell state amplitude at |00> should be 1/sqrt(2)")
        #expect(abs(mpsAmp11.real - invSqrt2) < 1e-10, "Bell state amplitude at |11> should be 1/sqrt(2)")
        #expect(abs(mps.amplitude(of: 1).real) < 1e-10, "Bell state amplitude at |01> should be 0")
        #expect(abs(mps.amplitude(of: 2).real) < 1e-10, "Bell state amplitude at |10> should be 0")
    }

    @Test("Conversion preserves normalization")
    func conversionPreservesNormalization() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes: [Complex<Double>] = [
            Complex(invSqrt2, 0.0),
            Complex(invSqrt2, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
        ]
        let state = QuantumState(qubits: 2, amplitudes: amplitudes)
        let mps = MatrixProductState(from: state)
        #expect(mps.isNormalized(), "Converted MPS should be normalized")
    }

    @Test("Conversion from uniform superposition")
    func conversionFromUniformSuperposition() {
        let n = 3
        let invSqrtN = 1.0 / sqrt(Double(1 << n))
        let amplitudes = [Complex<Double>](repeating: Complex(invSqrtN, 0.0), count: 1 << n)
        let state = QuantumState(qubits: n, amplitudes: amplitudes)
        let mps = MatrixProductState(from: state, maxBondDimension: 32)

        for i in 0 ..< (1 << n) {
            let mpsAmp = mps.amplitude(of: i)
            #expect(abs(mpsAmp.real - invSqrtN) < 1e-9, "Uniform superposition amplitude should be 1/sqrt(8) at index \(i)")
        }
    }

    @Test("Single qubit conversion works")
    func singleQubitConversion() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes: [Complex<Double>] = [Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0)]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)
        let mps = MatrixProductState(from: state)

        #expect(mps.qubits == 1, "MPS should have 1 qubit")
        #expect(abs(mps.amplitude(of: 0).real - invSqrt2) < 1e-10, "Single qubit |+> amplitude at |0> should be 1/sqrt(2)")
        #expect(abs(mps.amplitude(of: 1).real - invSqrt2) < 1e-10, "Single qubit |+> amplitude at |1> should be 1/sqrt(2)")
    }
}

/// Tests amplitude(of:) method of MatrixProductState.
/// Validates correct complex amplitude extraction for known quantum states.
/// Ensures accurate retrieval of both real and imaginary components.
@Suite("MPS Amplitude Calculation")
struct MPSAmplitudeTests {
    @Test("Amplitude calculation for product state")
    func amplitudeProductState() {
        let mps = MatrixProductState(qubits: 4, basisState: 10)
        let amp = mps.amplitude(of: 10)
        #expect(abs(amp.real - 1.0) < 1e-10, "Product state amplitude should be 1.0 at correct index")
        #expect(abs(amp.imaginary) < 1e-10, "Product state amplitude should have zero imaginary part")
    }

    @Test("Amplitude calculation returns complex values")
    func amplitudeReturnsComplex() {
        let amplitudes: [Complex<Double>] = [
            Complex(0.5, 0.5),
            Complex(0.5, -0.5),
        ]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)
        let mps = MatrixProductState(from: state)

        let mpsAmp0 = mps.amplitude(of: 0)
        let mpsAmp1 = mps.amplitude(of: 1)

        #expect(abs(mpsAmp0.real - state.amplitude(of: 0).real) < 1e-10, "Real part should match at index 0")
        #expect(abs(mpsAmp0.imaginary - state.amplitude(of: 0).imaginary) < 1e-10, "Imaginary part should match at index 0")
        #expect(abs(mpsAmp1.real - state.amplitude(of: 1).real) < 1e-10, "Real part should match at index 1")
        #expect(abs(mpsAmp1.imaginary - state.amplitude(of: 1).imaginary) < 1e-10, "Imaginary part should match at index 1")
    }
}

/// Tests probability(of:) method of MatrixProductState.
/// Validates Born rule probability calculation as magnitude squared.
/// Ensures probabilities sum to unity for normalized quantum states.
@Suite("MPS Probability Calculation")
struct MPSProbabilityTests {
    @Test("Probability of basis state is 1")
    func probabilityBasisState() {
        let mps = MatrixProductState(qubits: 4, basisState: 7)
        let prob = mps.probability(of: 7)
        #expect(abs(prob - 1.0) < 1e-10, "Probability of basis state should be 1.0")
    }

    @Test("Probability of other states is 0")
    func probabilityOtherStates() {
        let mps = MatrixProductState(qubits: 3, basisState: 3)
        for i in 0 ..< 8 where i != 3 {
            let prob = mps.probability(of: i)
            #expect(abs(prob) < 1e-10, "Probability should be 0 at index \(i)")
        }
    }

    @Test("Probability sum is 1 for superposition")
    func probabilitySumForSuperposition() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes: [Complex<Double>] = [
            Complex(invSqrt2, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
            Complex(invSqrt2, 0.0),
        ]
        let state = QuantumState(qubits: 2, amplitudes: amplitudes)
        let mps = MatrixProductState(from: state)

        var sum = 0.0
        for i in 0 ..< 4 {
            sum += mps.probability(of: i)
        }
        #expect(abs(sum - 1.0) < 1e-10, "Total probability should sum to 1.0")
    }

    @Test("Born rule: probability equals magnitude squared")
    func bornRuleProbability() {
        let amplitudes: [Complex<Double>] = [
            Complex(0.6, 0.0),
            Complex(0.8, 0.0),
        ]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)
        let mps = MatrixProductState(from: state)

        let prob0 = mps.probability(of: 0)
        let prob1 = mps.probability(of: 1)

        #expect(abs(prob0 - 0.36) < 1e-10, "Probability should be |0.6|^2 = 0.36")
        #expect(abs(prob1 - 0.64) < 1e-10, "Probability should be |0.8|^2 = 0.64")
    }
}

/// Tests normalization methods of MatrixProductState.
/// Validates isNormalized, normSquared, and normalize operations.
/// Ensures quantum states maintain proper normalization throughout operations.
@Suite("MPS Normalization")
struct MPSNormalizationTests {
    @Test("Ground state is normalized")
    func groundStateIsNormalized() {
        let mps = MatrixProductState(qubits: 5)
        #expect(mps.isNormalized(), "Ground state should be normalized")
    }

    @Test("Basis state is normalized")
    func basisStateIsNormalized() {
        let mps = MatrixProductState(qubits: 4, basisState: 11)
        #expect(mps.isNormalized(), "Basis state should be normalized")
    }

    @Test("normSquared returns 1 for normalized state")
    func normSquaredForNormalized() {
        let mps = MatrixProductState(qubits: 3)
        let norm = mps.normSquared()
        #expect(abs(norm - 1.0) < 1e-10, "Norm squared should be 1.0 for normalized state")
    }

    @Test("normSquared for converted Bell state")
    func normSquaredBellState() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes: [Complex<Double>] = [
            Complex(invSqrt2, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
            Complex(invSqrt2, 0.0),
        ]
        let state = QuantumState(qubits: 2, amplitudes: amplitudes)
        let mps = MatrixProductState(from: state)
        let norm = mps.normSquared()
        #expect(abs(norm - 1.0) < 1e-10, "Bell state norm squared should be 1.0")
    }

    @Test("normalize restores normalization")
    func normalizeRestoresNormalization() {
        let state = QuantumState(qubits: 2)
        var mps = MatrixProductState(from: state)
        mps.normalize()
        #expect(mps.isNormalized(), "State should be normalized after normalize()")
    }
}

/// Tests toQuantumState conversion from MatrixProductState.
/// Validates round-trip conversion preserves quantum state fidelity.
/// Ensures amplitude and normalization consistency after conversion.
@Suite("MPS To QuantumState Conversion")
struct MPSToQuantumStateTests {
    @Test("Round-trip conversion preserves ground state")
    func roundTripGroundState() {
        let mps = MatrixProductState(qubits: 3)
        let state = mps.toQuantumState()

        #expect(state.qubits == 3, "Converted state should have 3 qubits")
        #expect(abs(state.amplitude(of: 0).real - 1.0) < 1e-10, "Converted ground state should have amplitude 1 at index 0")
        for i in 1 ..< 8 {
            #expect(abs(state.amplitude(of: i).real) < 1e-10, "Converted ground state should have amplitude 0 at index \(i)")
        }
    }

    @Test("Round-trip conversion preserves basis state")
    func roundTripBasisState() {
        let mps = MatrixProductState(qubits: 4, basisState: 13)
        let state = mps.toQuantumState()

        #expect(abs(state.probability(of: 13) - 1.0) < 1e-10, "Round-trip should preserve basis state probability")
    }

    @Test("Round-trip conversion preserves Bell state")
    func roundTripBellState() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let originalAmps: [Complex<Double>] = [
            Complex(invSqrt2, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
            Complex(invSqrt2, 0.0),
        ]
        let originalState = QuantumState(qubits: 2, amplitudes: originalAmps)
        let mps = MatrixProductState(from: originalState)
        let roundTripState = mps.toQuantumState()

        for i in 0 ..< 4 {
            let origAmp = originalState.amplitude(of: i)
            let roundAmp = roundTripState.amplitude(of: i)
            #expect(abs(origAmp.real - roundAmp.real) < 1e-10, "Round-trip should preserve amplitude real part at index \(i)")
            #expect(abs(origAmp.imaginary - roundAmp.imaginary) < 1e-10, "Round-trip should preserve amplitude imaginary part at index \(i)")
        }
    }

    @Test("Round-trip preserves normalization")
    func roundTripPreservesNormalization() {
        let mps = MatrixProductState(qubits: 4, basisState: 5)
        let state = mps.toQuantumState()
        #expect(state.isNormalized(), "Round-trip converted state should be normalized")
    }
}

/// Tests expectation value calculation with PauliString operators.
/// Validates Z and X Pauli operator expectations on product states.
/// Ensures correct quantum mechanical expectation values for various bases.
@Suite("MPS Expectation Value PauliString")
struct MPSExpectationValuePauliStringTests {
    @Test("Z expectation on |0> is +1")
    func zExpectationOnZero() {
        let mps = MatrixProductState(qubits: 1)
        let z0 = PauliString(.z(0))
        let expectation = mps.expectationValue(of: z0)
        #expect(abs(expectation - 1.0) < 1e-10, "Z expectation on |0> should be +1")
    }

    @Test("Z expectation on |1> is -1")
    func zExpectationOnOne() {
        let mps = MatrixProductState(qubits: 1, basisState: 1)
        let z0 = PauliString(.z(0))
        let expectation = mps.expectationValue(of: z0)
        #expect(abs(expectation + 1.0) < 1e-10, "Z expectation on |1> should be -1")
    }

    @Test("ZZ expectation on |00> is +1")
    func zzExpectationOnZeroZero() {
        let mps = MatrixProductState(qubits: 2)
        let zz = PauliString(.z(0), .z(1))
        let expectation = mps.expectationValue(of: zz)
        #expect(abs(expectation - 1.0) < 1e-10, "ZZ expectation on |00> should be +1")
    }

    @Test("ZZ expectation on |01> is -1")
    func zzExpectationOnZeroOne() {
        let mps = MatrixProductState(qubits: 2, basisState: 1)
        let zz = PauliString(.z(0), .z(1))
        let expectation = mps.expectationValue(of: zz)
        #expect(abs(expectation + 1.0) < 1e-10, "ZZ expectation on |01> should be -1")
    }

    @Test("ZZ expectation on |11> is +1")
    func zzExpectationOnOneOne() {
        let mps = MatrixProductState(qubits: 2, basisState: 3)
        let zz = PauliString(.z(0), .z(1))
        let expectation = mps.expectationValue(of: zz)
        #expect(abs(expectation - 1.0) < 1e-10, "ZZ expectation on |11> should be +1")
    }

    @Test("X expectation on |0> is 0")
    func xExpectationOnZero() {
        let mps = MatrixProductState(qubits: 1)
        let x0 = PauliString(.x(0))
        let expectation = mps.expectationValue(of: x0)
        #expect(abs(expectation) < 1e-10, "X expectation on |0> should be 0")
    }

    @Test("X expectation on |+> is +1")
    func xExpectationOnPlus() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes: [Complex<Double>] = [Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0)]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)
        let mps = MatrixProductState(from: state)
        let x0 = PauliString(.x(0))
        let expectation = mps.expectationValue(of: x0)
        #expect(abs(expectation - 1.0) < 1e-10, "X expectation on |+> should be +1")
    }

    @Test("X expectation on |-> is -1")
    func xExpectationOnMinus() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes: [Complex<Double>] = [Complex(invSqrt2, 0.0), Complex(-invSqrt2, 0.0)]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)
        let mps = MatrixProductState(from: state)
        let x0 = PauliString(.x(0))
        let expectation = mps.expectationValue(of: x0)
        #expect(abs(expectation + 1.0) < 1e-10, "X expectation on |-> should be -1")
    }

    @Test("Identity Pauli string returns 1")
    func identityPauliString() {
        let mps = MatrixProductState(qubits: 3, basisState: 5)
        let identity = PauliString([])
        let expectation = mps.expectationValue(of: identity)
        #expect(abs(expectation - 1.0) < 1e-10, "Identity Pauli string expectation should be 1")
    }

    @Test("Multi-qubit Pauli string ZXZ")
    func multiQubitPauliString() {
        let mps = MatrixProductState(qubits: 3)
        let zxz = PauliString(.z(0), .x(1), .z(2))
        let expectation = mps.expectationValue(of: zxz)
        #expect(abs(expectation) < 1e-10, "ZXZ expectation on |000> should be 0 due to X on |0>")
    }
}

/// Tests expectation value calculation with Observable operators.
/// Validates Hamiltonian expectation values for quantum systems.
/// Ensures correct linear combination of Pauli term contributions.
@Suite("MPS Expectation Value Observable")
struct MPSExpectationValueObservableTests {
    @Test("Single Z observable")
    func singleZObservable() {
        let mps = MatrixProductState(qubits: 2)
        let observable = Observable(coefficient: 0.5, pauliString: PauliString(.z(0)))
        let expectation = mps.expectationValue(of: observable)
        #expect(abs(expectation - 0.5) < 1e-10, "Single Z observable expectation should be 0.5 * 1 = 0.5")
    }

    @Test("Sum of Z observables")
    func sumOfZObservables() {
        let mps = MatrixProductState(qubits: 2)
        let terms: PauliTerms = [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: 0.5, pauliString: PauliString(.z(1))),
        ]
        let observable = Observable(terms: terms)
        let expectation = mps.expectationValue(of: observable)
        #expect(abs(expectation - 1.0) < 1e-10, "Sum of Z expectations on |00> should be 0.5 + 0.5 = 1.0")
    }

    @Test("Ising-like Hamiltonian on ground state")
    func isingHamiltonianGroundState() {
        let mps = MatrixProductState(qubits: 2)
        let terms: PauliTerms = [
            (coefficient: -1.0, pauliString: PauliString(.z(0), .z(1))),
            (coefficient: -0.5, pauliString: PauliString(.x(0))),
            (coefficient: -0.5, pauliString: PauliString(.x(1))),
        ]
        let observable = Observable(terms: terms)
        let expectation = mps.expectationValue(of: observable)
        #expect(abs(expectation + 1.0) < 1e-10, "Ising Hamiltonian on |00>: -1*1 + 0 + 0 = -1")
    }

    @Test("Observable with negative coefficients")
    func observableNegativeCoefficients() {
        let mps = MatrixProductState(qubits: 1, basisState: 1)
        let terms: PauliTerms = [
            (coefficient: -2.0, pauliString: PauliString(.z(0))),
        ]
        let observable = Observable(terms: terms)
        let expectation = mps.expectationValue(of: observable)
        #expect(abs(expectation - 2.0) < 1e-10, "Observable with -2*Z on |1>: -2 * (-1) = 2")
    }

    @Test("Empty observable returns 0")
    func emptyObservable() {
        let mps = MatrixProductState(qubits: 2)
        let observable = Observable(terms: [])
        let expectation = mps.expectationValue(of: observable)
        #expect(abs(expectation) < 1e-10, "Empty observable expectation should be 0")
    }
}

/// Tests bond dimension properties of MatrixProductState.
/// Validates bondDimensions array and currentMaxBondDimension tracking.
/// Ensures correct entanglement characterization through bond dimensions.
@Suite("MPS Bond Dimension Properties")
struct MPSBondDimensionTests {
    @Test("Product state has bond dimension 1")
    func productStateBondDimension() {
        let mps = MatrixProductState(qubits: 5)
        let bonds = mps.bondDimensions
        #expect(bonds.count == 4, "5-qubit MPS should have 4 bond dimensions")
        for bond in bonds {
            #expect(bond == 1, "Product state should have bond dimension 1")
        }
    }

    @Test("currentMaxBondDimension for product state is 1")
    func currentMaxBondDimensionProductState() {
        let mps = MatrixProductState(qubits: 6)
        #expect(mps.currentMaxBondDimension == 1, "Product state should have max bond dimension 1")
    }

    @Test("Single qubit has empty bond dimensions")
    func singleQubitBondDimensions() {
        let mps = MatrixProductState(qubits: 1)
        #expect(mps.bondDimensions.isEmpty, "Single qubit MPS should have no bonds")
    }

    @Test("maxBondDimension property stores correctly")
    func maxBondDimensionProperty() {
        let mps = MatrixProductState(qubits: 4, maxBondDimension: 32)
        #expect(mps.maxBondDimension == 32, "maxBondDimension should be 32")
    }

    @Test("Bond dimension increases for entangled state")
    func bondDimensionEntangledState() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes: [Complex<Double>] = [
            Complex(invSqrt2, 0.0),
            Complex(0.0, 0.0),
            Complex(0.0, 0.0),
            Complex(invSqrt2, 0.0),
        ]
        let state = QuantumState(qubits: 2, amplitudes: amplitudes)
        let mps = MatrixProductState(from: state, maxBondDimension: 16)
        #expect(mps.currentMaxBondDimension >= 1, "Entangled state should have bond dimension >= 1")
    }
}

/// Tests memory usage calculation of MatrixProductState.
/// Validates memoryUsage property returns accurate byte counts.
/// Ensures memory scales appropriately with qubit count and bond dimensions.
@Suite("MPS Memory Usage")
struct MPSMemoryUsageTests {
    @Test("Memory usage is positive")
    func memoryUsagePositive() {
        let mps = MatrixProductState(qubits: 10)
        #expect(mps.memoryUsage > 0, "Memory usage should be positive")
    }

    @Test("Memory usage scales with qubits for product state")
    func memoryUsageScalesWithQubits() {
        let mps5 = MatrixProductState(qubits: 5)
        let mps10 = MatrixProductState(qubits: 10)
        #expect(mps10.memoryUsage > mps5.memoryUsage, "More qubits should use more memory")
    }

    @Test("Memory usage for product state is linear")
    func memoryUsageProductStateLinear() {
        let mps4 = MatrixProductState(qubits: 4)
        let mps8 = MatrixProductState(qubits: 8)
        let ratio = Double(mps8.memoryUsage) / Double(mps4.memoryUsage)
        #expect(abs(ratio - 2.0) < 0.1, "Product state memory should scale linearly: ratio should be ~2")
    }
}

/// Tests truncation statistics tracking in MatrixProductState operations.
/// Validates that truncation errors are recorded during MPS manipulations.
/// Ensures fresh and product states have minimal or zero truncation error.
@Suite("MPS Truncation Statistics Tracking")
struct MPSTruncationTrackingTests {
    @Test("Fresh MPS has zero truncation statistics")
    func freshMPSZeroTruncation() {
        let mps = MatrixProductState(qubits: 5)
        #expect(mps.truncationStatistics.truncationCount == 0, "Fresh MPS should have zero truncation count")
        #expect(abs(mps.truncationStatistics.cumulativeError) < 1e-15, "Fresh MPS should have zero cumulative error")
    }

    @Test("Basis state has zero truncation error")
    func basisStateZeroTruncation() {
        let mps = MatrixProductState(qubits: 6, basisState: 42)
        #expect(abs(mps.truncationStatistics.cumulativeError) < 1e-15, "Basis state should have zero truncation error")
    }

    @Test("Conversion from product state has minimal truncation")
    func conversionProductStateMinimalTruncation() {
        let state = QuantumState(qubits: 4)
        let mps = MatrixProductState(from: state)
        #expect(mps.truncationStatistics.cumulativeError < 1e-10, "Product state conversion should have minimal truncation error")
    }

    @Test("Truncation statistics is Sendable")
    func truncationStatisticsIsSendable() {
        let stats = MPSTruncationStatistics.zero
        _ = stats as Sendable
    }
}

/// Tests CustomStringConvertible implementation of MatrixProductState.
/// Validates description format contains required state information.
/// Ensures human-readable output includes qubits, bonds, and dimensions.
@Suite("MPS Description")
struct MPSDescriptionTests {
    @Test("Description contains qubits")
    func descriptionContainsQubits() {
        let mps = MatrixProductState(qubits: 5)
        #expect(mps.description.contains("qubits=5"), "Description should contain qubits count")
    }

    @Test("Description contains maxBond")
    func descriptionContainsMaxBond() {
        let mps = MatrixProductState(qubits: 3, maxBondDimension: 32)
        #expect(mps.description.contains("maxBond=32"), "Description should contain max bond dimension")
    }

    @Test("Description contains currentMaxBond")
    func descriptionContainsCurrentMaxBond() {
        let mps = MatrixProductState(qubits: 4)
        #expect(mps.description.contains("currentMaxBond="), "Description should contain current max bond")
    }

    @Test("Description contains bonds array")
    func descriptionContainsBonds() {
        let mps = MatrixProductState(qubits: 3)
        #expect(mps.description.contains("bonds=["), "Description should contain bonds array")
    }
}

/// Tests Equatable conformance of MatrixProductState.
/// Validates equality comparison between MPS instances.
/// Ensures states with same structure and amplitudes are considered equal.
@Suite("MPS Equatable")
struct MPSEquatableTests {
    @Test("Identical MPS are equal")
    func identicalMPSEqual() {
        let mps1 = MatrixProductState(qubits: 3)
        let mps2 = MatrixProductState(qubits: 3)
        #expect(mps1 == mps2, "Identical MPS should be equal")
    }

    @Test("Different qubit count MPS are not equal")
    func differentQubitCountNotEqual() {
        let mps1 = MatrixProductState(qubits: 3)
        let mps2 = MatrixProductState(qubits: 4)
        #expect(mps1 != mps2, "MPS with different qubit counts should not be equal")
    }

    @Test("Different basis states are not equal")
    func differentBasisStatesNotEqual() {
        let mps1 = MatrixProductState(qubits: 3, basisState: 0)
        let mps2 = MatrixProductState(qubits: 3, basisState: 5)
        #expect(mps1 != mps2, "MPS with different basis states should not be equal")
    }
}

/// Tests Sendable conformance of MatrixProductState.
/// Validates that MPS instances can be safely transferred across concurrency domains.
/// Ensures thread-safe usage in Swift structured concurrency contexts.
@Suite("MPS Sendable")
struct MPSSendableTests {
    @Test("MPS is Sendable")
    func mpsIsSendable() {
        let mps = MatrixProductState(qubits: 4)
        _ = mps as Sendable
    }
}

/// Tests Y Pauli expectation value calculation.
/// Validates Y-basis matrix generation in expectationValue(of:) method.
/// Ensures correct quantum mechanical expectation for Y Pauli operator.
@Suite("MPS Y Pauli Expectation Value")
struct MPSYPauliExpectationValueTests {
    @Test("Y Pauli expectation value on superposition state")
    func yPauliExpectationValue() {
        var mps = MatrixProductState(qubits: 1, maxBondDimension: 4)
        mps.applySingleQubitGate(.hadamard, to: 0)
        let yString = PauliString(.y(0))
        let expectation = mps.expectationValue(of: yString)
        #expect(abs(expectation) < 1e-10, "Y expectation on |+⟩ should be 0")
    }

    @Test("Y Pauli expectation value on |0⟩ state")
    func yPauliExpectationOnZero() {
        let mps = MatrixProductState(qubits: 1)
        let yString = PauliString(.y(0))
        let expectation = mps.expectationValue(of: yString)
        #expect(abs(expectation) < 1e-10, "Y expectation on |0⟩ should be 0")
    }

    @Test("Y Pauli expectation value on |1⟩ state")
    func yPauliExpectationOnOne() {
        let mps = MatrixProductState(qubits: 1, basisState: 1)
        let yString = PauliString(.y(0))
        let expectation = mps.expectationValue(of: yString)
        #expect(abs(expectation) < 1e-10, "Y expectation on |1⟩ should be 0")
    }

    @Test("Y Pauli expectation value on |i⟩ eigenstate")
    func yPauliExpectationOnPlusI() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes: [Complex<Double>] = [Complex(invSqrt2, 0.0), Complex(0.0, invSqrt2)]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)
        let mps = MatrixProductState(from: state)
        let yString = PauliString(.y(0))
        let expectation = mps.expectationValue(of: yString)
        #expect(abs(expectation - 1.0) < 1e-10, "Y expectation on |+i⟩ should be +1")
    }

    @Test("Y Pauli expectation value on |-i⟩ eigenstate")
    func yPauliExpectationOnMinusI() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let amplitudes: [Complex<Double>] = [Complex(invSqrt2, 0.0), Complex(0.0, -invSqrt2)]
        let state = QuantumState(qubits: 1, amplitudes: amplitudes)
        let mps = MatrixProductState(from: state)
        let yString = PauliString(.y(0))
        let expectation = mps.expectationValue(of: yString)
        #expect(abs(expectation + 1.0) < 1e-10, "Y expectation on |-i⟩ should be -1")
    }

    @Test("YY Pauli expectation value on multi-qubit state")
    func yyPauliExpectationMultiQubit() {
        let mps = MatrixProductState(qubits: 2)
        let yyString = PauliString(.y(0), .y(1))
        let expectation = mps.expectationValue(of: yyString)
        #expect(abs(expectation) < 1e-10, "YY expectation on |00⟩ should be 0")
    }
}

/// Tests internal tensor structure of MatrixProductState.
/// Validates tensor count, bond dimensions, and physical dimensions.
/// Ensures MPS maintains correct tensor network topology for quantum states.
@Suite("MPS Tensor Structure")
struct MPSTensorStructureTests {
    @Test("Tensor count equals qubit count")
    func tensorCountEqualsQubitCount() {
        let mps = MatrixProductState(qubits: 7)
        #expect(mps.tensors.count == 7, "MPS should have one tensor per qubit")
    }

    @Test("First tensor has left bond dimension 1")
    func firstTensorLeftBondDimension() {
        let mps = MatrixProductState(qubits: 5)
        #expect(mps.tensors[0].leftBondDimension == 1, "First tensor should have left bond dimension 1")
    }

    @Test("Last tensor has right bond dimension 1")
    func lastTensorRightBondDimension() {
        let mps = MatrixProductState(qubits: 5)
        #expect(mps.tensors[4].rightBondDimension == 1, "Last tensor should have right bond dimension 1")
    }

    @Test("All tensors have physical dimension 2")
    func allTensorsPhysicalDimension() {
        let mps = MatrixProductState(qubits: 6)
        for tensor in mps.tensors {
            #expect(tensor.physicalDimension == 2, "All tensors should have physical dimension 2")
        }
    }
}
