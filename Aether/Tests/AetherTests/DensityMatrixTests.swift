// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for DensityMatrix initialization.
/// Validates ground state construction, pure state projection,
/// and explicit element initialization with proper dimension handling.
@Suite("Density Matrix Initialization")
struct DensityMatrixInitializationTests {
    @Test("Initialize 1-qubit ground state |0⟩⟨0|")
    func initializeOneQubitGround() {
        let dm = DensityMatrix(qubits: 1)

        #expect(dm.qubits == 1, "Should have 1 qubit")
        #expect(dm.dimension == 2, "Dimension should be 2^1 = 2")
        #expect(dm.element(row: 0, col: 0) == .one, "ρ[0,0] should be 1")
        #expect(dm.element(row: 0, col: 1) == .zero, "ρ[0,1] should be 0")
        #expect(dm.element(row: 1, col: 0) == .zero, "ρ[1,0] should be 0")
        #expect(dm.element(row: 1, col: 1) == .zero, "ρ[1,1] should be 0")
    }

    @Test("Initialize 2-qubit ground state |00⟩⟨00|")
    func initializeTwoQubitGround() {
        let dm = DensityMatrix(qubits: 2)

        #expect(dm.qubits == 2, "Should have 2 qubits")
        #expect(dm.dimension == 4, "Dimension should be 2^2 = 4")
        #expect(dm.element(row: 0, col: 0) == .one, "ρ[0,0] should be 1")

        for i in 0 ..< 4 {
            for j in 0 ..< 4 where !(i == 0 && j == 0) {
                #expect(dm.element(row: i, col: j) == .zero, "Off-diagonal and non-|00⟩ elements should be zero")
            }
        }
    }

    @Test("Initialize from pure state |ψ⟩⟨ψ|")
    func initializeFromPureState() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let plus = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0), Complex(invSqrt2, 0),
        ])
        let dm = DensityMatrix(pureState: plus)

        #expect(dm.qubits == 1, "Should have 1 qubit")
        #expect(abs(dm.element(row: 0, col: 0).real - 0.5) < 1e-10, "ρ[0,0] should be 0.5")
        #expect(abs(dm.element(row: 0, col: 1).real - 0.5) < 1e-10, "ρ[0,1] should be 0.5 (coherence)")
        #expect(abs(dm.element(row: 1, col: 0).real - 0.5) < 1e-10, "ρ[1,0] should be 0.5 (coherence)")
        #expect(abs(dm.element(row: 1, col: 1).real - 0.5) < 1e-10, "ρ[1,1] should be 0.5")
    }

    @Test("Initialize from Bell state creates entangled density matrix")
    func initializeFromBellState() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])
        let dm = DensityMatrix(pureState: bell)

        #expect(dm.qubits == 2, "Should have 2 qubits")
        #expect(abs(dm.element(row: 0, col: 0).real - 0.5) < 1e-10, "ρ[00,00] should be 0.5")
        #expect(abs(dm.element(row: 0, col: 3).real - 0.5) < 1e-10, "ρ[00,11] should be 0.5 (entanglement)")
        #expect(abs(dm.element(row: 3, col: 0).real - 0.5) < 1e-10, "ρ[11,00] should be 0.5 (entanglement)")
        #expect(abs(dm.element(row: 3, col: 3).real - 0.5) < 1e-10, "ρ[11,11] should be 0.5")
    }

    @Test("Initialize from explicit elements")
    func initializeFromExplicitElements() {
        let elements: [Complex<Double>] = [
            Complex(0.5, 0), Complex(0.3, 0.1),
            Complex(0.3, -0.1), Complex(0.5, 0),
        ]
        let dm = DensityMatrix(qubits: 1, elements: elements)

        #expect(dm.element(row: 0, col: 0) == Complex(0.5, 0), "Should preserve explicit elements")
        #expect(dm.element(row: 0, col: 1) == Complex(0.3, 0.1), "Should preserve explicit elements")
    }

    @Test("Dimension scales as 2^n")
    func dimensionScaling() {
        for n in 1 ... 6 {
            let dm = DensityMatrix(qubits: n)
            #expect(dm.dimension == 1 << n, "Dimension should be 2^\(n)")
        }
    }
}

/// Test suite for DensityMatrix static factory methods.
/// Validates maximally mixed state construction with uniform diagonal,
/// and basis state projector |i⟩⟨i| construction with correct structure.
@Suite("Density Matrix Factory Methods")
struct DensityMatrixFactoryTests {
    @Test("Maximally mixed state has uniform diagonal")
    func maximallyMixedUniformDiagonal() {
        let dm = DensityMatrix.maximallyMixed(qubits: 2)
        let expectedDiag = 0.25

        for i in 0 ..< 4 {
            #expect(abs(dm.element(row: i, col: i).real - expectedDiag) < 1e-10,
                    "Diagonal element ρ[\(i),\(i)] should be 1/4")
        }
    }

    @Test("Maximally mixed state has zero off-diagonals")
    func maximallyMixedZeroOffDiagonal() {
        let dm = DensityMatrix.maximallyMixed(qubits: 2)

        for i in 0 ..< 4 {
            for j in 0 ..< 4 where i != j {
                #expect(dm.element(row: i, col: j) == .zero,
                        "Off-diagonal ρ[\(i),\(j)] should be zero")
            }
        }
    }

    @Test("Maximally mixed state purity equals 1/d")
    func maximallyMixedPurity() {
        for n in 1 ... 4 {
            let dm = DensityMatrix.maximallyMixed(qubits: n)
            let expectedPurity = 1.0 / Double(1 << n)
            #expect(abs(dm.purity() - expectedPurity) < 1e-10,
                    "Purity should be 1/2^\(n) = \(expectedPurity)")
        }
    }

    @Test("Basis state |i⟩⟨i| has correct structure")
    func basisStateStructure() {
        let dm = DensityMatrix.basis(qubits: 2, state: 0b10)

        #expect(dm.element(row: 2, col: 2) == .one, "ρ[2,2] should be 1 for |10⟩⟨10|")

        for i in 0 ..< 4 {
            for j in 0 ..< 4 where !(i == 2 && j == 2) {
                #expect(dm.element(row: i, col: j) == .zero,
                        "All elements except ρ[2,2] should be zero")
            }
        }
    }

    @Test("Basis state is pure")
    func basisStateIsPure() {
        for state in 0 ..< 4 {
            let dm = DensityMatrix.basis(qubits: 2, state: state)
            #expect(dm.isPure(), "Basis state |\(state)⟩⟨\(state)| should be pure")
        }
    }

    @Test("All basis states for given qubit count")
    func allBasisStates() {
        let n = 3
        let dim = 1 << n

        for state in 0 ..< dim {
            let dm = DensityMatrix.basis(qubits: n, state: state)
            #expect(dm.probability(of: state) == 1.0, "P(\(state)) should be 1.0")

            for other in 0 ..< dim where other != state {
                #expect(dm.probability(of: other) == 0.0, "P(\(other)) should be 0.0")
            }
        }
    }
}

/// Test suite for density matrix state properties.
/// Validates trace, purity, and Hermiticity computations that characterize
/// valid quantum states and distinguish pure from mixed states.
@Suite("Density Matrix Properties")
struct DensityMatrixPropertyTests {
    @Test("Ground state has trace 1")
    func groundStateTrace() {
        let dm = DensityMatrix(qubits: 2)
        #expect(abs(dm.trace() - 1.0) < 1e-10, "Trace should be 1.0")
    }

    @Test("Maximally mixed state has trace 1")
    func maximallyMixedTrace() {
        let dm = DensityMatrix.maximallyMixed(qubits: 3)
        #expect(abs(dm.trace() - 1.0) < 1e-10, "Trace should be 1.0")
    }

    @Test("Pure state from statevector has trace 1")
    func pureStateTrace() {
        let state = QuantumState(qubits: 2)
        let dm = DensityMatrix(pureState: state)
        #expect(abs(dm.trace() - 1.0) < 1e-10, "Trace should be 1.0")
    }

    @Test("Ground state is pure (purity = 1)")
    func groundStatePurity() {
        let dm = DensityMatrix(qubits: 2)
        #expect(abs(dm.purity() - 1.0) < 1e-10, "Purity should be 1.0 for pure state")
        #expect(dm.isPure(), "isPure() should return true")
    }

    @Test("Superposition state is pure")
    func superpositionStatePurity() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let state = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0), Complex(invSqrt2, 0),
        ])
        let dm = DensityMatrix(pureState: state)

        #expect(abs(dm.purity() - 1.0) < 1e-10, "Superposition is still pure")
        #expect(dm.isPure(), "isPure() should return true")
    }

    @Test("Maximally mixed state is not pure")
    func maximallyMixedNotPure() {
        let dm = DensityMatrix.maximallyMixed(qubits: 2)
        #expect(!dm.isPure(), "Maximally mixed state should not be pure")
        #expect(abs(dm.purity() - 0.25) < 1e-10, "Purity should be 1/4")
    }

    @Test("Purity bounds: 1/d ≤ Tr(ρ²) ≤ 1")
    func purityBounds() {
        let pure = DensityMatrix(qubits: 3)
        let mixed = DensityMatrix.maximallyMixed(qubits: 3)

        let minPurity = 1.0 / 8.0
        let maxPurity = 1.0

        #expect(pure.purity() <= maxPurity + 1e-10, "Purity should not exceed 1")
        #expect(mixed.purity() >= minPurity - 1e-10, "Purity should be at least 1/d")
    }

    @Test("Ground state is Hermitian")
    func groundStateHermitian() {
        let dm = DensityMatrix(qubits: 2)
        #expect(dm.isHermitian(), "Ground state should be Hermitian")
    }

    @Test("Maximally mixed state is Hermitian")
    func maximallyMixedHermitian() {
        let dm = DensityMatrix.maximallyMixed(qubits: 2)
        #expect(dm.isHermitian(), "Maximally mixed state should be Hermitian")
    }

    @Test("Pure state from superposition is Hermitian")
    func superpositionHermitian() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let state = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0), Complex(0, invSqrt2),
        ])
        let dm = DensityMatrix(pureState: state)

        #expect(dm.isHermitian(), "Pure state density matrix should be Hermitian")
    }

    @Test("Hermiticity: ρ[i,j] = ρ[j,i]*")
    func hermitianConjugateSymmetry() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let state = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(0, invSqrt2),
        ])
        let dm = DensityMatrix(pureState: state)

        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                let rhoIJ = dm.element(row: i, col: j)
                let rhoJI = dm.element(row: j, col: i)
                let diff = rhoIJ - rhoJI.conjugate
                #expect(diff.magnitudeSquared < 1e-20,
                        "ρ[\(i),\(j)] should equal ρ[\(j),\(i)]* for Hermitian matrix")
            }
        }
    }

    @Test("isTraceNormalized returns true for valid states")
    func traceNormalized() {
        let dm1 = DensityMatrix(qubits: 2)
        let dm2 = DensityMatrix.maximallyMixed(qubits: 2)

        #expect(dm1.isTraceNormalized(), "Ground state should be trace normalized")
        #expect(dm2.isTraceNormalized(), "Maximally mixed should be trace normalized")
    }

    @Test("Non-Hermitian matrix detected by isHermitian")
    func nonHermitianDetected() {
        var dm = DensityMatrix(qubits: 1)
        dm.setElement(row: 0, col: 1, to: Complex(0.3, 0.2))
        dm.setElement(row: 1, col: 0, to: Complex(0.3, 0.2))

        #expect(!dm.isHermitian(), "Matrix with ρ[0,1] ≠ ρ[1,0]* should not be Hermitian")
    }
}

/// Test suite for probability calculations on density matrices.
/// Validates Born rule P(i) = ρ[i,i] for computational basis measurements,
/// full probability distributions, and most probable state identification.
@Suite("Density Matrix Probabilities")
struct DensityMatrixProbabilityTests {
    @Test("Ground state probability distribution")
    func groundStateProbabilities() {
        let dm = DensityMatrix(qubits: 2)

        #expect(dm.probability(of: 0) == 1.0, "P(00) should be 1.0")
        #expect(dm.probability(of: 1) == 0.0, "P(01) should be 0.0")
        #expect(dm.probability(of: 2) == 0.0, "P(10) should be 0.0")
        #expect(dm.probability(of: 3) == 0.0, "P(11) should be 0.0")
    }

    @Test("Maximally mixed uniform probabilities")
    func maximallyMixedProbabilities() {
        let dm = DensityMatrix.maximallyMixed(qubits: 2)

        for i in 0 ..< 4 {
            #expect(abs(dm.probability(of: i) - 0.25) < 1e-10, "P(\(i)) should be 0.25")
        }
    }

    @Test("Bell state probabilities")
    func bellStateProbabilities() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])
        let dm = DensityMatrix(pureState: bell)

        #expect(abs(dm.probability(of: 0) - 0.5) < 1e-10, "P(00) should be 0.5")
        #expect(abs(dm.probability(of: 1)) < 1e-10, "P(01) should be 0.0")
        #expect(abs(dm.probability(of: 2)) < 1e-10, "P(10) should be 0.0")
        #expect(abs(dm.probability(of: 3) - 0.5) < 1e-10, "P(11) should be 0.5")
    }

    @Test("probabilities() returns full distribution")
    func fullProbabilityDistribution() {
        let dm = DensityMatrix.maximallyMixed(qubits: 3)
        let probs = dm.probabilities()

        #expect(probs.count == 8, "Should have 8 probabilities")

        let sum = probs.reduce(0.0, +)
        #expect(abs(sum - 1.0) < 1e-10, "Probabilities should sum to 1.0")

        for p in probs {
            #expect(abs(p - 0.125) < 1e-10, "Each probability should be 1/8")
        }
    }

    @Test("mostProbableState returns correct state")
    func mostProbableState() {
        let dm = DensityMatrix.basis(qubits: 3, state: 5)
        let (index, prob) = dm.mostProbableState()

        #expect(index == 5, "Most probable state should be 5")
        #expect(prob == 1.0, "Probability should be 1.0")
    }

    @Test("mostProbableState for uniform distribution")
    func mostProbableStateUniform() {
        let dm = DensityMatrix.maximallyMixed(qubits: 2)
        let (index, prob) = dm.mostProbableState()

        #expect(index >= 0 && index < 4, "Index should be valid")
        #expect(abs(prob - 0.25) < 1e-10, "Probability should be 0.25")
    }
}

/// Test suite for element access and mutation.
/// Validates get/set operations on density matrix elements,
/// and verifies row-major storage order for correct indexing.
@Suite("Density Matrix Element Access")
struct DensityMatrixElementAccessTests {
    @Test("Get element returns correct value")
    func getElement() {
        let dm = DensityMatrix(qubits: 1)

        #expect(dm.element(row: 0, col: 0) == .one, "ρ[0,0] should be 1")
        #expect(dm.element(row: 0, col: 1) == .zero, "ρ[0,1] should be 0")
        #expect(dm.element(row: 1, col: 0) == .zero, "ρ[1,0] should be 0")
        #expect(dm.element(row: 1, col: 1) == .zero, "ρ[1,1] should be 0")
    }

    @Test("Set element modifies correct value")
    func setElement() {
        var dm = DensityMatrix(qubits: 1)
        let newValue = Complex(0.3, 0.4)

        dm.setElement(row: 0, col: 1, to: newValue)

        #expect(dm.element(row: 0, col: 1) == newValue, "Modified element should match")
        #expect(dm.element(row: 0, col: 0) == .one, "Other elements should be unchanged")
    }

    @Test("Row-major storage order")
    func rowMajorOrder() {
        let elements: [Complex<Double>] = [
            Complex(1, 0), Complex(2, 0),
            Complex(3, 0), Complex(4, 0),
        ]
        let dm = DensityMatrix(qubits: 1, elements: elements)

        #expect(dm.element(row: 0, col: 0) == Complex(1, 0), "ρ[0,0] = element[0]")
        #expect(dm.element(row: 0, col: 1) == Complex(2, 0), "ρ[0,1] = element[1]")
        #expect(dm.element(row: 1, col: 0) == Complex(3, 0), "ρ[1,0] = element[2]")
        #expect(dm.element(row: 1, col: 1) == Complex(4, 0), "ρ[1,1] = element[3]")
    }
}

/// Test suite for partial trace operation.
/// Validates subsystem reduction Tr_B(ρ_AB) that reveals entanglement
/// by showing mixed reduced states for entangled systems.
@Suite("Density Matrix Partial Trace")
struct DensityMatrixPartialTraceTests {
    @Test("Partial trace of product state yields pure reduced state")
    func productStatePartialTrace() {
        let state = QuantumState(qubits: 2)
        let dm = DensityMatrix(pureState: state)
        let reduced = dm.partialTrace(over: [1])

        #expect(reduced.qubits == 1, "Reduced state should have 1 qubit")
        #expect(reduced.isPure(), "Reduced state of product state should be pure")
        #expect(reduced.probability(of: 0) == 1.0, "Should be |0⟩⟨0|")
    }

    @Test("Partial trace of Bell state yields maximally mixed")
    func bellStatePartialTrace() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])
        let dm = DensityMatrix(pureState: bell)

        let reduced = dm.partialTrace(over: [1])

        #expect(reduced.qubits == 1, "Reduced state should have 1 qubit")
        #expect(!reduced.isPure(), "Reduced Bell state should be mixed (entanglement signature)")
        #expect(abs(reduced.purity() - 0.5) < 1e-10, "Purity should be 0.5 (maximally mixed)")
        #expect(abs(reduced.probability(of: 0) - 0.5) < 1e-10, "P(0) should be 0.5")
        #expect(abs(reduced.probability(of: 1) - 0.5) < 1e-10, "P(1) should be 0.5")
    }

    @Test("Partial trace preserves trace")
    func partialTracePreservesTrace() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let state = QuantumState(qubits: 3, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, .zero,
            .zero, .zero, .zero, Complex(invSqrt2, 0),
        ])
        let dm = DensityMatrix(pureState: state)

        let reduced1 = dm.partialTrace(over: [0])
        let reduced2 = dm.partialTrace(over: [1])
        let reduced3 = dm.partialTrace(over: [2])
        let reduced12 = dm.partialTrace(over: [0, 1])

        #expect(abs(reduced1.trace() - 1.0) < 1e-10, "Trace should be preserved")
        #expect(abs(reduced2.trace() - 1.0) < 1e-10, "Trace should be preserved")
        #expect(abs(reduced3.trace() - 1.0) < 1e-10, "Trace should be preserved")
        #expect(abs(reduced12.trace() - 1.0) < 1e-10, "Trace should be preserved")
    }

    @Test("Partial trace over different qubits")
    func partialTraceSymmetry() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])
        let dm = DensityMatrix(pureState: bell)

        let reducedA = dm.partialTrace(over: [0])
        let reducedB = dm.partialTrace(over: [1])

        #expect(abs(reducedA.purity() - reducedB.purity()) < 1e-10,
                "Bell state has symmetric entanglement")
    }

    @Test("Partial trace of maximally mixed stays maximally mixed")
    func maximallyMixedPartialTrace() {
        let dm = DensityMatrix.maximallyMixed(qubits: 3)
        let reduced = dm.partialTrace(over: [1])

        #expect(reduced.qubits == 2, "Should have 2 qubits remaining")

        let expectedPurity = 0.25
        #expect(abs(reduced.purity() - expectedPurity) < 1e-10,
                "Reduced maximally mixed should still be maximally mixed")
    }

    @Test("Partial trace over multiple qubits")
    func partialTraceMultipleQubits() {
        let dm = DensityMatrix(qubits: 4)
        let reduced = dm.partialTrace(over: [1, 3])

        #expect(reduced.qubits == 2, "Should have 2 qubits remaining")
        #expect(reduced.isPure(), "Product state partial trace should be pure")
    }
}

/// Test suite for unitary gate application ρ -> UρU†.
/// Validates density matrix evolution under quantum gates,
/// preservation of trace, purity, and Hermiticity through unitaries.
@Suite("Density Matrix Gate Application")
struct DensityMatrixGateApplicationTests {
    @Test("Hadamard on |0⟩⟨0| creates |+⟩⟨+|")
    func hadamardOnGround() {
        let dm = DensityMatrix(qubits: 1)
        let result = dm.applying(.hadamard, to: 0)

        #expect(abs(result.element(row: 0, col: 0).real - 0.5) < 1e-10, "ρ[0,0] should be 0.5")
        #expect(abs(result.element(row: 0, col: 1).real - 0.5) < 1e-10, "ρ[0,1] should be 0.5")
        #expect(abs(result.element(row: 1, col: 0).real - 0.5) < 1e-10, "ρ[1,0] should be 0.5")
        #expect(abs(result.element(row: 1, col: 1).real - 0.5) < 1e-10, "ρ[1,1] should be 0.5")
    }

    @Test("Pauli X flips |0⟩⟨0| to |1⟩⟨1|")
    func pauliXFlip() {
        let dm = DensityMatrix(qubits: 1)
        let result = dm.applying(.pauliX, to: 0)

        #expect(result.element(row: 0, col: 0) == .zero, "ρ[0,0] should be 0")
        #expect(result.element(row: 1, col: 1) == .one, "ρ[1,1] should be 1")
    }

    @Test("Pauli Z preserves |0⟩⟨0|")
    func pauliZPreservesGround() {
        let dm = DensityMatrix(qubits: 1)
        let result = dm.applying(.pauliZ, to: 0)

        #expect(result.element(row: 0, col: 0) == .one, "ρ[0,0] should be 1")
        #expect(result.element(row: 1, col: 1) == .zero, "ρ[1,1] should be 0")
    }

    @Test("Gate application preserves trace")
    func gatePreservesTrace() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let state = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])
        let dm = DensityMatrix(pureState: state)

        let after = dm.applying(.hadamard, to: 0)

        #expect(abs(after.trace() - 1.0) < 1e-10, "Gate should preserve trace")
    }

    @Test("Gate application preserves purity")
    func gatePreservesPurity() {
        let dm = DensityMatrix(qubits: 2)
        let after = dm
            .applying(.hadamard, to: 0)
            .applying(.cnot, to: [0, 1])

        #expect(after.isPure(), "Unitary evolution should preserve purity")
    }

    @Test("Gate application preserves Hermiticity")
    func gatePreservesHermiticity() {
        let dm = DensityMatrix(qubits: 2)
        let after = dm
            .applying(.hadamard, to: 0)
            .applying(.pauliY, to: 1)

        #expect(after.isHermitian(), "Unitary evolution should preserve Hermiticity")
    }

    @Test("Two-qubit gate array syntax")
    func twoQubitGateArraySyntax() {
        let dm = DensityMatrix(qubits: 2)
        let result = dm.applying(.swap, to: [0, 1])

        #expect(result.probability(of: 0) == 1.0, "SWAP on |00⟩ should give |00⟩")
    }

    @Test("Sequence of gates")
    func sequenceOfGates() {
        let dm = DensityMatrix(qubits: 1)
        let result = dm
            .applying(.hadamard, to: 0)
            .applying(.pauliZ, to: 0)
            .applying(.hadamard, to: 0)

        #expect(abs(result.probability(of: 1) - 1.0) < 1e-10, "HZH = X, so should flip to |1⟩")
    }

    @Test("Gate on maximally mixed state")
    func gateOnMaximallyMixed() {
        let dm = DensityMatrix.maximallyMixed(qubits: 2)
        let after = dm.applying(.hadamard, to: 0)

        #expect(abs(after.purity() - dm.purity()) < 1e-10,
                "Unitary on maximally mixed should preserve purity")

        for i in 0 ..< 4 {
            #expect(abs(after.probability(of: i) - 0.25) < 1e-10,
                    "Maximally mixed stays uniform under unitaries")
        }
    }

    @Test("Toffoli gate via general gate path")
    func toffoliGateGeneralPath() {
        let dm = DensityMatrix.basis(qubits: 3, state: 0b110)
        let result = dm.applying(.toffoli, to: [0, 1, 2])

        #expect(abs(result.probability(of: 0b111) - 1.0) < 1e-10,
                "Toffoli with controls=11 should flip target: |110⟩ -> |111⟩")
        #expect(result.isPure(), "Toffoli should preserve purity")
        #expect(abs(result.trace() - 1.0) < 1e-10, "Toffoli should preserve trace")
    }

    @Test("Toffoli gate does not flip when control is 0")
    func toffoliNoFlipWhenControlZero() {
        let dm = DensityMatrix.basis(qubits: 3, state: 0b100)
        let result = dm.applying(.toffoli, to: [0, 1, 2])

        #expect(abs(result.probability(of: 0b100) - 1.0) < 1e-10,
                "Toffoli with control0=0 should not flip: |100⟩ -> |100⟩")
    }

    @Test("Toffoli on larger system with non-target qubits")
    func toffoliWithNonTargetQubits() {
        let dm = DensityMatrix.basis(qubits: 4, state: 0b1110)
        let result = dm.applying(.toffoli, to: [0, 1, 2])

        #expect(abs(result.probability(of: 0b1111) - 1.0) < 1e-10,
                "Toffoli should flip target q0 when controls q1,q2=11: |1110⟩ -> |1111⟩")
        #expect(result.isPure(), "Should preserve purity")
    }
}

/// Test suite for observable expectation values on density matrices.
/// Validates Tr(ρO) computation for Pauli observables including
/// single-qubit X, Y, Z and multi-qubit tensor products like ZZ.
@Suite("Density Matrix Expectation Values")
struct DensityMatrixExpectationValueTests {
    @Test("⟨Z⟩ = 1 for |0⟩⟨0|")
    func pauliZExpectationGround() {
        let dm = DensityMatrix(qubits: 1)
        let z = Observable.pauliZ(qubit: 0)

        let expectation = dm.expectationValue(of: z)

        #expect(abs(expectation - 1.0) < 1e-10, "⟨Z⟩ should be 1 for |0⟩")
    }

    @Test("⟨Z⟩ = -1 for |1⟩⟨1|")
    func pauliZExpectationExcited() {
        let dm = DensityMatrix.basis(qubits: 1, state: 1)
        let z = Observable.pauliZ(qubit: 0)

        let expectation = dm.expectationValue(of: z)

        #expect(abs(expectation + 1.0) < 1e-10, "⟨Z⟩ should be -1 for |1⟩")
    }

    @Test("⟨Z⟩ = 0 for maximally mixed")
    func pauliZExpectationMixed() {
        let dm = DensityMatrix.maximallyMixed(qubits: 1)
        let z = Observable.pauliZ(qubit: 0)

        let expectation = dm.expectationValue(of: z)

        #expect(abs(expectation) < 1e-10, "⟨Z⟩ should be 0 for maximally mixed")
    }

    @Test("⟨X⟩ = 1 for |+⟩⟨+|")
    func pauliXExpectationPlus() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let plus = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0), Complex(invSqrt2, 0),
        ])
        let dm = DensityMatrix(pureState: plus)
        let x = Observable.pauliX(qubit: 0)

        let expectation = dm.expectationValue(of: x)

        #expect(abs(expectation - 1.0) < 1e-10, "⟨X⟩ should be 1 for |+⟩")
    }

    @Test("⟨X⟩ = 0 for |0⟩⟨0|")
    func pauliXExpectationGround() {
        let dm = DensityMatrix(qubits: 1)
        let x = Observable.pauliX(qubit: 0)

        let expectation = dm.expectationValue(of: x)

        #expect(abs(expectation) < 1e-10, "⟨X⟩ should be 0 for |0⟩")
    }

    @Test("Multi-term observable expectation")
    func multiTermObservable() {
        let dm = DensityMatrix(qubits: 2)
        let observable = Observable(terms: [
            (1.0, PauliString(.z(0))),
            (1.0, PauliString(.z(1))),
        ])

        let expectation = dm.expectationValue(of: observable)

        #expect(abs(expectation - 2.0) < 1e-10, "⟨Z₀ + Z₁⟩ should be 2 for |00⟩")
    }

    @Test("ZZ expectation for Bell state")
    func zzExpectationBell() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])
        let dm = DensityMatrix(pureState: bell)
        let zz = Observable(terms: [(1.0, PauliString(.z(0), .z(1)))])

        let expectation = dm.expectationValue(of: zz)

        #expect(abs(expectation - 1.0) < 1e-10, "⟨Z₀Z₁⟩ should be 1 for |Φ⁺⟩")
    }
}

/// Test suite for Equatable conformance and string representation.
/// Validates equality comparison between density matrices and
/// verifies description contains relevant state information.
@Suite("Density Matrix Equality and Description")
struct DensityMatrixEqualityTests {
    @Test("Identical density matrices are equal")
    func identicalEqual() {
        let dm1 = DensityMatrix(qubits: 2)
        let dm2 = DensityMatrix(qubits: 2)

        #expect(dm1 == dm2, "Identical ground states should be equal")
    }

    @Test("Different qubit counts are not equal")
    func differentQubitsNotEqual() {
        let dm1 = DensityMatrix(qubits: 1)
        let dm2 = DensityMatrix(qubits: 2)

        #expect(dm1 != dm2, "Different qubit counts should not be equal")
    }

    @Test("Different states are not equal")
    func differentStatesNotEqual() {
        let dm1 = DensityMatrix(qubits: 1)
        let dm2 = DensityMatrix(qubits: 1).applying(.pauliX, to: 0)

        #expect(dm1 != dm2, "Different states should not be equal")
    }

    @Test("Description contains qubit count")
    func descriptionContainsQubits() {
        let dm = DensityMatrix(qubits: 2)
        #expect(dm.description.contains("2 qubits"), "Description should mention qubit count")
    }

    @Test("Description contains purity")
    func descriptionContainsPurity() {
        let dm = DensityMatrix(qubits: 1)
        #expect(dm.description.contains("purity"), "Description should mention purity")
    }

    @Test("Description shows near-zero for negligible diagonal")
    func descriptionNearZero() {
        let elements: [Complex<Double>] = [
            Complex(1e-8, 0), Complex(0.5, 0),
            Complex(0.5, 0), Complex(1e-8, 0),
        ]
        let dm = DensityMatrix(qubits: 1, elements: elements)

        #expect(dm.description.contains("near-zero"),
                "Description should show 'near-zero' when all diagonals below threshold")
    }
}

/// Test suite for edge cases and boundary conditions.
/// Validates single-qubit minimum, large systems, complex phases,
/// and numerical stability through gate sequences.
@Suite("Density Matrix Edge Cases")
struct DensityMatrixEdgeCaseTests {
    @Test("Single qubit minimum case")
    func singleQubitMinimum() {
        let dm = DensityMatrix(qubits: 1)

        #expect(dm.qubits == 1, "Should handle 1 qubit")
        #expect(dm.dimension == 2, "Dimension should be 2")
        #expect(dm.isPure(), "Should be pure")
        #expect(dm.isTraceNormalized(), "Should be trace normalized")
        #expect(dm.isHermitian(), "Should be Hermitian")
    }

    @Test("Maximum practical size (10 qubits)")
    func largeDensityMatrix() {
        let dm = DensityMatrix(qubits: 10)

        #expect(dm.qubits == 10, "Should handle 10 qubits")
        #expect(dm.dimension == 1024, "Dimension should be 1024")
        #expect(abs(dm.trace() - 1.0) < 1e-10, "Trace should be 1.0")
    }

    @Test("Complex amplitudes with phases")
    func complexPhases() {
        let state = QuantumState(qubits: 1, amplitudes: [
            Complex(1.0 / sqrt(2.0), 0),
            Complex(0, 1.0 / sqrt(2.0)),
        ])
        let dm = DensityMatrix(pureState: state)

        #expect(dm.isPure(), "State with complex phases should still be pure")
        #expect(dm.isHermitian(), "Should be Hermitian despite complex phases")

        #expect(abs(dm.element(row: 0, col: 1).imaginary + 0.5) < 1e-10,
                "Off-diagonal should have imaginary part")
    }

    @Test("Nearly zero coherences")
    func nearlyZeroCoherences() {
        var dm = DensityMatrix.maximallyMixed(qubits: 1)
        dm.setElement(row: 0, col: 1, to: Complex(1e-15, 0))
        dm.setElement(row: 1, col: 0, to: Complex(1e-15, 0))

        #expect(dm.isHermitian(tolerance: 1e-10), "Should be Hermitian within tolerance")
    }

    @Test("Gate sequence maintains valid state")
    func gateSequenceMaintainsValidity() {
        var dm = DensityMatrix(qubits: 3)

        for _ in 0 ..< 10 {
            dm = dm.applying(.hadamard, to: 0)
            dm = dm.applying(.cnot, to: [0, 1])
            dm = dm.applying(.rotationZ(.pi / 4), to: 2)
        }

        #expect(abs(dm.trace() - 1.0) < 1e-9, "Trace should be preserved through gate sequence")
        #expect(dm.isHermitian(tolerance: 1e-9), "Hermiticity should be preserved")
    }
}
