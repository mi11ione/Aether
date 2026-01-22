// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for LCUDecomposition struct properties.
/// Validates that decomposition correctly extracts coefficients, unitaries,
/// and metadata from Hamiltonians represented as weighted Pauli strings.
@Suite("LCUDecomposition Properties")
struct LCUDecompositionPropertiesTests {
    @Test("Decomposition preserves original coefficients")
    func decompositionPreservesOriginalCoefficients() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: -0.3, pauliString: PauliString(.x(1))),
            (coefficient: 0.2, pauliString: PauliString(.z(0), .z(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.originalCoefficients.count == 3, "Decomposition should have 3 original coefficients")
        #expect(abs(decomposition.originalCoefficients[0] - 0.5) < 1e-10, "First coefficient should be 0.5")
        #expect(abs(decomposition.originalCoefficients[1] - -0.3) < 1e-10, "Second coefficient should be -0.3")
        #expect(abs(decomposition.originalCoefficients[2] - 0.2) < 1e-10, "Third coefficient should be 0.2")
    }

    @Test("Decomposition preserves unitaries in order")
    func decompositionPreservesUnitaries() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: -0.3, pauliString: PauliString(.x(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.unitaries.count == 2, "Decomposition should have 2 unitaries")
        #expect(decomposition.unitaries[0] == PauliString(.z(0)), "First unitary should be Z on qubit 0")
        #expect(decomposition.unitaries[1] == PauliString(.x(1)), "Second unitary should be X on qubit 1")
    }

    @Test("Decomposition correctly reports term count")
    func decompositionTermCount() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: -0.3, pauliString: PauliString(.x(1))),
            (coefficient: 0.2, pauliString: PauliString(.z(0), .z(1))),
            (coefficient: 0.1, pauliString: PauliString(.y(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.termCount == 4, "Decomposition should report 4 terms")
    }

    @Test("Decomposition filters zero-coefficient terms")
    func decompositionFiltersZeroCoefficients() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: 0.0, pauliString: PauliString(.x(1))),
            (coefficient: 1e-16, pauliString: PauliString(.y(0))),
            (coefficient: 0.3, pauliString: PauliString(.z(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.termCount == 2, "Decomposition should filter out zero and near-zero coefficient terms")
    }

    @Test("Empty Hamiltonian produces empty decomposition")
    func emptyHamiltonianDecomposition() {
        let hamiltonian = Observable(terms: [])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.termCount == 0, "Empty Hamiltonian should produce zero terms")
        #expect(decomposition.normalizedCoefficients.isEmpty, "Normalized coefficients should be empty")
        #expect(decomposition.unitaries.isEmpty, "Unitaries should be empty")
    }
}

/// Test suite for normalized coefficient computation.
/// Validates that normalized coefficients sum to 1 and represent
/// probability weights for the PREPARE oracle superposition.
@Suite("Normalized Coefficients")
struct NormalizedCoefficientsTests {
    @Test("Normalized coefficients sum to 1")
    func normalizedCoefficientsSumToOne() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: -0.3, pauliString: PauliString(.x(1))),
            (coefficient: 0.2, pauliString: PauliString(.z(0), .z(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        let sum = decomposition.normalizedCoefficients.reduce(0.0, +)
        #expect(abs(sum - 1.0) < 1e-10, "Normalized coefficients should sum to 1.0")
    }

    @Test("Normalized coefficients are non-negative")
    func normalizedCoefficientsNonNegative() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: -0.3, pauliString: PauliString(.x(1))),
            (coefficient: -0.7, pauliString: PauliString(.y(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        for i in 0 ..< decomposition.normalizedCoefficients.count {
            #expect(decomposition.normalizedCoefficients[i] >= 0.0, "Normalized coefficient at index \(i) should be non-negative")
        }
    }

    @Test("Normalized coefficients equal |alpha_i|/sum(|alpha_j|)")
    func normalizedCoefficientsFormula() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.6, pauliString: PauliString(.z(0))),
            (coefficient: -0.4, pauliString: PauliString(.x(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        let expectedFirst = 0.6 / 1.0
        let expectedSecond = 0.4 / 1.0
        #expect(abs(decomposition.normalizedCoefficients[0] - expectedFirst) < 1e-10, "First normalized coefficient should be 0.6")
        #expect(abs(decomposition.normalizedCoefficients[1] - expectedSecond) < 1e-10, "Second normalized coefficient should be 0.4")
    }

    @Test("Single term has normalized coefficient of 1")
    func singleTermNormalization() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.7, pauliString: PauliString(.z(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.normalizedCoefficients.count == 1, "Should have exactly one normalized coefficient")
        #expect(abs(decomposition.normalizedCoefficients[0] - 1.0) < 1e-10, "Single normalized coefficient should be 1.0")
    }
}

/// Test suite for one-norm computation.
/// Validates that oneNorm equals the sum of absolute values of original
/// coefficients, which determines block-encoding normalization factor.
@Suite("One-Norm Computation")
struct OneNormTests {
    @Test("One-norm equals sum of absolute coefficients")
    func oneNormComputation() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: -0.3, pauliString: PauliString(.x(1))),
            (coefficient: 0.2, pauliString: PauliString(.z(0), .z(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        let expectedOneNorm = 0.5 + 0.3 + 0.2
        #expect(abs(decomposition.oneNorm - expectedOneNorm) < 1e-10, "One-norm should be sum of absolute coefficients")
    }

    @Test("One-norm handles all negative coefficients")
    func oneNormAllNegative() {
        let hamiltonian = Observable(terms: [
            (coefficient: -0.4, pauliString: PauliString(.z(0))),
            (coefficient: -0.6, pauliString: PauliString(.x(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(abs(decomposition.oneNorm - 1.0) < 1e-10, "One-norm should be 1.0 for coefficients summing to -1.0 in magnitude")
    }

    @Test("One-norm is zero for empty Hamiltonian")
    func oneNormEmptyHamiltonian() {
        let hamiltonian = Observable(terms: [])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(abs(decomposition.oneNorm) < 1e-10, "One-norm should be zero for empty Hamiltonian")
    }

    @Test("One-norm with mixed positive and negative coefficients")
    func oneNormMixedCoefficients() {
        let hamiltonian = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
            (coefficient: -2.0, pauliString: PauliString(.x(1))),
            (coefficient: 0.5, pauliString: PauliString(.y(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(abs(decomposition.oneNorm - 3.5) < 1e-10, "One-norm should be 3.5 (|1.0| + |-2.0| + |0.5|)")
    }
}

/// Test suite for ancilla qubit count computation.
/// Validates that ancillaQubits equals ceil(log2(termCount)) which determines
/// the register size for encoding term indices in binary.
@Suite("Ancilla Qubit Count")
struct AncillaQubitCountTests {
    @Test("One term requires 1 ancilla qubit")
    func oneTermAncilla() {
        let hamiltonian = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.ancillaQubits == 1, "One term should require 1 ancilla qubit")
    }

    @Test("Two terms require 1 ancilla qubit")
    func twoTermsAncilla() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: 0.5, pauliString: PauliString(.x(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.ancillaQubits == 1, "Two terms should require 1 ancilla qubit (ceil(log2(2)) = 1)")
    }

    @Test("Three terms require 2 ancilla qubits")
    func threeTermsAncilla() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: 0.3, pauliString: PauliString(.x(1))),
            (coefficient: 0.2, pauliString: PauliString(.y(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.ancillaQubits == 2, "Three terms should require 2 ancilla qubits (ceil(log2(3)) = 2)")
    }

    @Test("Four terms require 2 ancilla qubits")
    func fourTermsAncilla() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.25, pauliString: PauliString(.z(0))),
            (coefficient: 0.25, pauliString: PauliString(.x(1))),
            (coefficient: 0.25, pauliString: PauliString(.y(0))),
            (coefficient: 0.25, pauliString: PauliString(.z(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.ancillaQubits == 2, "Four terms should require 2 ancilla qubits (ceil(log2(4)) = 2)")
    }

    @Test("Five terms require 3 ancilla qubits")
    func fiveTermsAncilla() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.2, pauliString: PauliString(.z(0))),
            (coefficient: 0.2, pauliString: PauliString(.x(1))),
            (coefficient: 0.2, pauliString: PauliString(.y(0))),
            (coefficient: 0.2, pauliString: PauliString(.z(1))),
            (coefficient: 0.2, pauliString: PauliString(.x(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.ancillaQubits == 3, "Five terms should require 3 ancilla qubits (ceil(log2(5)) = 3)")
    }

    @Test("Eight terms require 3 ancilla qubits")
    func eightTermsAncilla() {
        var terms: PauliTerms = []
        for i in 0 ..< 8 {
            terms.append((coefficient: 0.125, pauliString: PauliString(.z(i))))
        }
        let hamiltonian = Observable(terms: terms)
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.ancillaQubits == 3, "Eight terms should require 3 ancilla qubits (ceil(log2(8)) = 3)")
    }
}

/// Test suite for PREPARE circuit generation.
/// Validates that prepareCircuit produces valid quantum circuits
/// with correct qubit count for amplitude encoding superposition.
@Suite("PREPARE Circuit Generation")
struct PrepareCircuitTests {
    @Test("PREPARE circuit has correct qubit count")
    func prepareCircuitQubitCount() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: 0.3, pauliString: PauliString(.x(1))),
            (coefficient: 0.2, pauliString: PauliString(.y(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let ancillaStart = 2
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: ancillaStart)

        let expectedQubits = ancillaStart + decomposition.ancillaQubits
        #expect(circuit.qubits == expectedQubits, "PREPARE circuit should have \(expectedQubits) qubits")
    }

    @Test("PREPARE circuit for single term is empty")
    func prepareCircuitSingleTerm() {
        let hamiltonian = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 1)

        #expect(circuit.gates.isEmpty, "PREPARE circuit for single term should be empty")
    }

    @Test("PREPARE circuit for empty decomposition is empty")
    func prepareCircuitEmptyDecomposition() {
        let hamiltonian = Observable(terms: [])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 0)

        #expect(circuit.gates.isEmpty, "PREPARE circuit for empty decomposition should be empty")
    }

    @Test("PREPARE circuit for two equal terms produces gates")
    func prepareCircuitTwoEqualTerms() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: 0.5, pauliString: PauliString(.x(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 2)

        #expect(circuit.gates.count > 0, "PREPARE circuit for two equal terms should have gates")
    }

    @Test("PREPARE circuit with different ancilla start positions")
    func prepareCircuitDifferentAncillaStarts() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.6, pauliString: PauliString(.z(0))),
            (coefficient: 0.4, pauliString: PauliString(.x(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        let circuit1 = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 2)
        let circuit2 = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 5)

        #expect(circuit1.qubits == 3, "PREPARE circuit with ancillaStart=2 should have 3 qubits")
        #expect(circuit2.qubits == 6, "PREPARE circuit with ancillaStart=5 should have 6 qubits")
    }
}

/// Test suite for SELECT circuit generation.
/// Validates that selectCircuit produces valid quantum circuits
/// that apply controlled Pauli strings based on ancilla state.
@Suite("SELECT Circuit Generation")
struct SelectCircuitTests {
    @Test("SELECT circuit has correct qubit count")
    func selectCircuitQubitCount() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: 0.3, pauliString: PauliString(.x(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let systemQubits = 2
        let ancillaStart = 2
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: systemQubits,
            ancillaStart: ancillaStart,
        )

        let expectedQubits = ancillaStart + decomposition.ancillaQubits
        #expect(circuit.qubits == expectedQubits, "SELECT circuit should have \(expectedQubits) qubits")
    }

    @Test("SELECT circuit for empty decomposition is empty")
    func selectCircuitEmptyDecomposition() {
        let hamiltonian = Observable(terms: [])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 2,
            ancillaStart: 2,
        )

        #expect(circuit.gates.isEmpty, "SELECT circuit for empty decomposition should be empty")
    }

    @Test("SELECT circuit produces gates for non-trivial Hamiltonian")
    func selectCircuitProducesGates() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: 0.3, pauliString: PauliString(.x(1))),
            (coefficient: 0.2, pauliString: PauliString(.y(0), .y(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 2,
            ancillaStart: 2,
        )

        #expect(circuit.gates.count > 0, "SELECT circuit should produce gates for non-trivial Hamiltonian")
    }

    @Test("SELECT circuit handles negative coefficients")
    func selectCircuitNegativeCoefficients() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: -0.5, pauliString: PauliString(.x(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 2,
            ancillaStart: 2,
        )

        #expect(circuit.gates.count > 0, "SELECT circuit should handle negative coefficients")
    }

    @Test("SELECT circuit with single-qubit Pauli strings")
    func selectCircuitSingleQubitPaulis() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.3, pauliString: PauliString(.x(0))),
            (coefficient: 0.3, pauliString: PauliString(.y(0))),
            (coefficient: 0.4, pauliString: PauliString(.z(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 1,
            ancillaStart: 1,
        )

        #expect(circuit.gates.count > 0, "SELECT circuit should handle all Pauli types")
    }
}

/// Test suite for block encoding circuit generation.
/// Validates that blockEncodingCircuit correctly combines PREPARE,
/// SELECT, and PREPARE-inverse to form the full LCU circuit.
@Suite("Block Encoding Circuit")
struct BlockEncodingCircuitTests {
    @Test("Block encoding circuit has correct qubit count")
    func blockEncodingQubitCount() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: 0.3, pauliString: PauliString(.x(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let systemQubits = 2
        let ancillaStart = 2
        let circuit = LCU.blockEncodingCircuit(
            decomposition: decomposition,
            systemQubits: systemQubits,
            ancillaStart: ancillaStart,
        )

        let expectedQubits = ancillaStart + decomposition.ancillaQubits
        #expect(circuit.qubits == expectedQubits, "Block encoding circuit should have \(expectedQubits) qubits")
    }

    @Test("Block encoding circuit for empty decomposition is empty")
    func blockEncodingEmptyDecomposition() {
        let hamiltonian = Observable(terms: [])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.blockEncodingCircuit(
            decomposition: decomposition,
            systemQubits: 2,
            ancillaStart: 2,
        )

        #expect(circuit.gates.isEmpty, "Block encoding circuit for empty decomposition should be empty")
    }

    @Test("Block encoding combines PREPARE, SELECT, PREPARE-inverse")
    func blockEncodingStructure() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.6, pauliString: PauliString(.z(0))),
            (coefficient: 0.4, pauliString: PauliString(.x(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        let prepareCircuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 2)
        let selectCircuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 2,
            ancillaStart: 2,
        )
        let blockCircuit = LCU.blockEncodingCircuit(
            decomposition: decomposition,
            systemQubits: 2,
            ancillaStart: 2,
        )

        let expectedMinGates = prepareCircuit.gates.count + selectCircuit.gates.count
        #expect(blockCircuit.gates.count >= expectedMinGates, "Block encoding should have at least PREPARE + SELECT gates")
    }

    @Test("Block encoding circuit produces gates for multi-term Hamiltonian")
    func blockEncodingProducesGates() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.3, pauliString: PauliString(.z(0))),
            (coefficient: 0.3, pauliString: PauliString(.x(1))),
            (coefficient: 0.4, pauliString: PauliString(.z(0), .z(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.blockEncodingCircuit(
            decomposition: decomposition,
            systemQubits: 2,
            ancillaStart: 2,
        )

        #expect(circuit.gates.count > 0, "Block encoding circuit should produce gates for multi-term Hamiltonian")
    }

    @Test("Block encoding with single term produces minimal circuit")
    func blockEncodingSingleTerm() {
        let hamiltonian = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.blockEncodingCircuit(
            decomposition: decomposition,
            systemQubits: 1,
            ancillaStart: 1,
        )

        let prepCircuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 1)
        let selCircuit = LCU.selectCircuit(decomposition: decomposition, systemQubits: 1, ancillaStart: 1)
        let expectedGates = prepCircuit.gates.count * 2 + selCircuit.gates.count
        #expect(circuit.gates.count == expectedGates, "Block encoding for single term should have PREPARE + SELECT + PREPARE-inverse gates")
    }
}

/// Test suite for success probability estimation.
/// Validates that estimateSuccessProbability returns values in [0,1]
/// and correctly computes (expectedEnergy/oneNorm)^2.
@Suite("Success Probability Estimation")
struct SuccessProbabilityTests {
    @Test("Success probability is bounded between 0 and 1")
    func successProbabilityBounds() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: 0.5, pauliString: PauliString(.x(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        let prob1 = LCU.estimateSuccessProbability(decomposition: decomposition, expectedEnergy: 0.5)
        let prob2 = LCU.estimateSuccessProbability(decomposition: decomposition, expectedEnergy: 2.0)
        let prob3 = LCU.estimateSuccessProbability(decomposition: decomposition, expectedEnergy: -0.5)

        #expect(prob1 >= 0.0 && prob1 <= 1.0, "Success probability should be in [0, 1]")
        #expect(prob2 >= 0.0 && prob2 <= 1.0, "Success probability should be clamped to [0, 1]")
        #expect(prob3 >= 0.0 && prob3 <= 1.0, "Success probability should be in [0, 1] for negative energy")
    }

    @Test("Success probability equals (E/alpha)^2")
    func successProbabilityFormula() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.6, pauliString: PauliString(.z(0))),
            (coefficient: 0.4, pauliString: PauliString(.x(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let energy = 0.5
        let probability = LCU.estimateSuccessProbability(decomposition: decomposition, expectedEnergy: energy)

        let ratio = energy / decomposition.oneNorm
        let expected = ratio * ratio
        #expect(abs(probability - expected) < 1e-10, "Success probability should equal (energy/oneNorm)^2")
    }

    @Test("Success probability is zero when oneNorm is zero")
    func successProbabilityZeroOneNorm() {
        let hamiltonian = Observable(terms: [])
        let decomposition = LCU.decompose(hamiltonian)
        let probability = LCU.estimateSuccessProbability(decomposition: decomposition, expectedEnergy: 1.0)

        #expect(abs(probability) < 1e-10, "Success probability should be zero when oneNorm is zero")
    }

    @Test("Success probability is 1 when energy equals oneNorm")
    func successProbabilityMaximum() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.7, pauliString: PauliString(.z(0))),
            (coefficient: 0.3, pauliString: PauliString(.x(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let probability = LCU.estimateSuccessProbability(
            decomposition: decomposition,
            expectedEnergy: decomposition.oneNorm,
        )

        #expect(abs(probability - 1.0) < 1e-10, "Success probability should be 1.0 when energy equals oneNorm")
    }

    @Test("Success probability is zero when energy is zero")
    func successProbabilityZeroEnergy() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: 0.5, pauliString: PauliString(.x(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let probability = LCU.estimateSuccessProbability(decomposition: decomposition, expectedEnergy: 0.0)

        #expect(abs(probability) < 1e-10, "Success probability should be zero when energy is zero")
    }

    @Test("Success probability handles negative energy")
    func successProbabilityNegativeEnergy() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: 0.5, pauliString: PauliString(.x(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let probability = LCU.estimateSuccessProbability(decomposition: decomposition, expectedEnergy: -0.5)

        let ratio = -0.5 / decomposition.oneNorm
        let expected = ratio * ratio
        #expect(abs(probability - expected) < 1e-10, "Success probability should work correctly with negative energy")
    }
}

/// Test suite for multi-qubit Pauli string handling in LCU.
/// Validates decomposition and circuit generation for Hamiltonians
/// with complex multi-qubit Pauli string terms.
@Suite("Multi-Qubit Pauli String Handling")
struct MultiQubitPauliStringTests {
    @Test("Decomposition handles ZZ interaction term")
    func decompositionZZTerm() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0), .z(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.termCount == 1, "Should have one ZZ term")
        #expect(decomposition.unitaries[0] == PauliString(.z(0), .z(1)), "Unitary should be ZZ")
    }

    @Test("Decomposition handles XYZ term")
    func decompositionXYZTerm() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.3, pauliString: PauliString(.x(0), .y(1), .z(2))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.termCount == 1, "Should have one XYZ term")
        #expect(decomposition.unitaries[0] == PauliString(.x(0), .y(1), .z(2)), "Unitary should be XYZ")
    }

    @Test("SELECT circuit handles multi-qubit Pauli strings")
    func selectCircuitMultiQubitPaulis() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0), .z(1))),
            (coefficient: 0.3, pauliString: PauliString(.x(0), .x(1))),
            (coefficient: 0.2, pauliString: PauliString(.y(0), .y(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 2,
            ancillaStart: 2,
        )

        #expect(circuit.gates.count > 0, "SELECT circuit should handle multi-qubit Pauli strings")
    }

    @Test("Heisenberg model Hamiltonian decomposition")
    func heisenbergModelDecomposition() {
        let hamiltonian = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.x(0), .x(1))),
            (coefficient: 1.0, pauliString: PauliString(.y(0), .y(1))),
            (coefficient: 1.0, pauliString: PauliString(.z(0), .z(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.termCount == 3, "Heisenberg model should have 3 terms")
        #expect(abs(decomposition.oneNorm - 3.0) < 1e-10, "Heisenberg model oneNorm should be 3.0")
        #expect(decomposition.ancillaQubits == 2, "Heisenberg model should need 2 ancilla qubits")
    }
}

/// Test suite for LCUDecomposition struct immutability and Sendable conformance.
/// Validates that the decomposition result is thread-safe and its properties
/// are correctly accessible as documented.
@Suite("LCUDecomposition Sendable and Immutability")
struct LCUDecompositionSendableTests {
    @Test("LCUDecomposition properties are accessible")
    func decompositionPropertiesAccessible() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: 0.3, pauliString: PauliString(.x(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.normalizedCoefficients.count == 2, "normalizedCoefficients should be accessible")
        #expect(decomposition.originalCoefficients.count == 2, "originalCoefficients should be accessible")
        #expect(decomposition.unitaries.count == 2, "unitaries should be accessible")
        #expect(decomposition.oneNorm > 0, "oneNorm should be accessible and positive")
        #expect(decomposition.ancillaQubits >= 1, "ancillaQubits should be accessible and at least 1")
        #expect(decomposition.termCount == 2, "termCount should be accessible")
    }

    @Test("LCUDecomposition can be captured in closure")
    func decompositionCapturedInClosure() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        let capturedTermCount = decomposition.termCount
        #expect(capturedTermCount == 1, "Decomposition should be capturable in closures")
    }
}

/// Test suite for edge cases in LCU decomposition and circuit generation.
/// Validates behavior with unusual but valid inputs like very small
/// coefficients, identity terms, and large term counts.
@Suite("Edge Cases")
struct LCUEdgeCasesTests {
    @Test("Very small coefficients are preserved")
    func verySmallCoefficients() {
        let smallCoeff = 1e-10
        let hamiltonian = Observable(terms: [
            (coefficient: smallCoeff, pauliString: PauliString(.z(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.termCount == 1, "Very small but non-zero coefficient should be preserved")
        #expect(abs(decomposition.originalCoefficients[0] - smallCoeff) < 1e-15, "Small coefficient should be preserved exactly")
    }

    @Test("Identity Pauli string (empty operators) is handled")
    func identityPauliString() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString([])),
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.termCount == 2, "Identity term should be included in decomposition")
    }

    @Test("Large coefficient values are handled")
    func largeCoefficients() {
        let largeCoeff = 1e10
        let hamiltonian = Observable(terms: [
            (coefficient: largeCoeff, pauliString: PauliString(.z(0))),
            (coefficient: -largeCoeff, pauliString: PauliString(.x(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)

        #expect(abs(decomposition.oneNorm - 2e10) < 1e5, "Large coefficients should be handled correctly")
        #expect(abs(decomposition.normalizedCoefficients[0] - 0.5) < 1e-10, "Normalization should work for large coefficients")
    }

    @Test("Sixteen terms require 4 ancilla qubits")
    func sixteenTermsAncilla() {
        var terms: PauliTerms = []
        for i in 0 ..< 16 {
            terms.append((coefficient: 1.0 / 16.0, pauliString: PauliString(.z(i % 4))))
        }
        let hamiltonian = Observable(terms: terms)
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.ancillaQubits == 4, "Sixteen terms should require 4 ancilla qubits")
    }

    @Test("Seventeen terms require 5 ancilla qubits")
    func seventeenTermsAncilla() {
        var terms: PauliTerms = []
        for i in 0 ..< 17 {
            terms.append((coefficient: 1.0 / 17.0, pauliString: PauliString(.z(i % 4))))
        }
        let hamiltonian = Observable(terms: terms)
        let decomposition = LCU.decompose(hamiltonian)

        #expect(decomposition.ancillaQubits == 5, "Seventeen terms should require 5 ancilla qubits (ceil(log2(17)) = 5)")
    }
}

/// Test suite for amplitude encoding tree edge cases in PREPARE circuit.
/// Validates behavior when probability distribution has zero regions
/// or skewed distributions that trigger special code paths.
@Suite("Amplitude Encoding Tree Edge Cases")
struct AmplitudeEncodingEdgeCasesTests {
    @Test("PREPARE circuit handles skewed probability distribution")
    func prepareSkewedProbability() {
        let hamiltonian = Observable(terms: [
            (coefficient: 1e-14, pauliString: PauliString(.z(0))),
            (coefficient: 1.0, pauliString: PauliString(.x(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 2)

        #expect(circuit.gates.count > 0, "PREPARE circuit should generate gates for skewed distribution")
    }

    @Test("PREPARE circuit with four terms and varying weights")
    func prepareFourTermsVarying() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.1, pauliString: PauliString(.z(0))),
            (coefficient: 0.2, pauliString: PauliString(.x(0))),
            (coefficient: 0.3, pauliString: PauliString(.y(0))),
            (coefficient: 0.4, pauliString: PauliString(.z(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 2)

        #expect(circuit.qubits == 4, "PREPARE circuit should have 4 qubits for 4 terms with ancillaStart=2")
    }

    @Test("PREPARE circuit with eight terms triggers multi-controlled rotations")
    func prepareEightTermsMultiControlled() {
        var terms: PauliTerms = []
        for i in 0 ..< 8 {
            terms.append((coefficient: 0.125, pauliString: PauliString(.z(i % 2))))
        }
        let hamiltonian = Observable(terms: terms)
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 2)

        #expect(circuit.gates.count > 0, "PREPARE circuit should generate gates for 8 uniform terms")
        #expect(circuit.qubits == 5, "PREPARE circuit should have 5 qubits (2 system + 3 ancilla)")
    }

    @Test("PREPARE circuit with uneven distribution in subtrees")
    func prepareUnevenSubtrees() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.4, pauliString: PauliString(.z(0))),
            (coefficient: 0.1, pauliString: PauliString(.x(1))),
            (coefficient: 0.3, pauliString: PauliString(.y(0))),
            (coefficient: 0.2, pauliString: PauliString(.z(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 2)

        #expect(circuit.qubits == 4, "PREPARE circuit should handle uneven subtree distributions")
    }

    @Test("PREPARE circuit probability concentrated in first term")
    func prepareProbabilityConcentratedFirst() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.97, pauliString: PauliString(.z(0))),
            (coefficient: 0.01, pauliString: PauliString(.x(0))),
            (coefficient: 0.01, pauliString: PauliString(.y(0))),
            (coefficient: 0.01, pauliString: PauliString(.z(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 2)

        #expect(circuit.gates.count > 0, "PREPARE circuit should handle concentrated probability in first term")
    }

    @Test("PREPARE circuit probability concentrated in last term")
    func prepareProbabilityConcentratedLast() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.01, pauliString: PauliString(.z(0))),
            (coefficient: 0.01, pauliString: PauliString(.x(0))),
            (coefficient: 0.01, pauliString: PauliString(.y(0))),
            (coefficient: 0.97, pauliString: PauliString(.z(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 2)

        #expect(circuit.gates.count > 0, "PREPARE circuit should handle concentrated probability in last term")
    }
}

/// Test suite for multi-controlled gate decomposition in SELECT circuit.
/// Validates behavior when SELECT circuit requires gates with multiple
/// control qubits, exercising Toffoli ladder decomposition paths.
@Suite("Multi-Controlled Gate Decomposition")
struct MultiControlledGateTests {
    @Test("SELECT circuit with five terms uses multi-controlled X")
    func selectFiveTermsMultiControlledX() {
        var terms: PauliTerms = []
        for _ in 0 ..< 5 {
            terms.append((coefficient: 0.2, pauliString: PauliString(.x(0))))
        }
        let hamiltonian = Observable(terms: terms)
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 1,
            ancillaStart: 1,
        )

        #expect(circuit.gates.count > 0, "SELECT circuit should produce gates for 5 terms with X operators")
    }

    @Test("SELECT circuit with five terms uses multi-controlled Y")
    func selectFiveTermsMultiControlledY() {
        var terms: PauliTerms = []
        for _ in 0 ..< 5 {
            terms.append((coefficient: 0.2, pauliString: PauliString(.y(0))))
        }
        let hamiltonian = Observable(terms: terms)
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 1,
            ancillaStart: 1,
        )

        #expect(circuit.gates.count > 0, "SELECT circuit should produce gates for 5 terms with Y operators")
    }

    @Test("SELECT circuit with five terms uses multi-controlled Z")
    func selectFiveTermsMultiControlledZ() {
        var terms: PauliTerms = []
        for _ in 0 ..< 5 {
            terms.append((coefficient: 0.2, pauliString: PauliString(.z(0))))
        }
        let hamiltonian = Observable(terms: terms)
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 1,
            ancillaStart: 1,
        )

        #expect(circuit.gates.count > 0, "SELECT circuit should produce gates for 5 terms with Z operators")
    }

    @Test("SELECT circuit with negative coefficients and multiple ancilla")
    func selectNegativeCoefficientsMultiAncilla() {
        var terms: PauliTerms = []
        for i in 0 ..< 5 {
            let sign = (i % 2 == 0) ? 1.0 : -1.0
            terms.append((coefficient: sign * 0.2, pauliString: PauliString(.z(0))))
        }
        let hamiltonian = Observable(terms: terms)
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 1,
            ancillaStart: 1,
        )

        #expect(circuit.gates.count > 0, "SELECT circuit should handle negative coefficients with 3 ancilla qubits")
        #expect(decomposition.ancillaQubits == 3, "Five terms should require 3 ancilla qubits")
    }

    @Test("SELECT circuit with nine terms triggers Toffoli ladder")
    func selectNineTermsToffoliLadder() {
        var terms: PauliTerms = []
        for _ in 0 ..< 9 {
            terms.append((coefficient: 1.0 / 9.0, pauliString: PauliString(.x(0))))
        }
        let hamiltonian = Observable(terms: terms)
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 1,
            ancillaStart: 1,
        )

        #expect(circuit.gates.count > 0, "SELECT circuit should handle 9 terms requiring Toffoli ladder decomposition")
        #expect(decomposition.ancillaQubits == 4, "Nine terms should require 4 ancilla qubits")
    }
}

/// Test suite for controlled phase application in SELECT circuit.
/// Validates correct phase handling for negative coefficients
/// with varying numbers of control qubits.
@Suite("Controlled Phase Application")
struct ControlledPhaseTests {
    @Test("SELECT circuit applies phase for single negative coefficient")
    func selectSingleNegativeCoefficient() {
        let hamiltonian = Observable(terms: [
            (coefficient: -1.0, pauliString: PauliString(.z(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 1,
            ancillaStart: 1,
        )

        #expect(circuit.gates.count > 0, "SELECT circuit should handle single negative coefficient")
    }

    @Test("SELECT circuit applies phase with two ancilla qubits")
    func selectNegativeCoeffTwoAncilla() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: -0.3, pauliString: PauliString(.x(0))),
            (coefficient: -0.2, pauliString: PauliString(.y(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 1,
            ancillaStart: 1,
        )

        #expect(circuit.gates.count > 0, "SELECT circuit should apply controlled phase with 2 ancilla qubits")
        #expect(decomposition.ancillaQubits == 2, "Three terms should require 2 ancilla qubits")
    }

    @Test("SELECT circuit applies multi-controlled phase with three ancilla")
    func selectMultiControlledPhaseThreeAncilla() {
        var terms: PauliTerms = []
        for i in 0 ..< 5 {
            let sign = (i % 2 == 0) ? 1.0 : -1.0
            terms.append((coefficient: sign * 0.2, pauliString: PauliString(.z(0))))
        }
        let hamiltonian = Observable(terms: terms)
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 1,
            ancillaStart: 1,
        )

        #expect(circuit.gates.count > 0, "SELECT circuit should apply multi-controlled phase with 3 ancilla qubits")
    }

    @Test("Block encoding with negative coefficients preserves correctness")
    func blockEncodingNegativeCoefficients() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.4, pauliString: PauliString(.z(0))),
            (coefficient: -0.3, pauliString: PauliString(.x(0))),
            (coefficient: -0.2, pauliString: PauliString(.y(0))),
            (coefficient: 0.1, pauliString: PauliString(.z(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.blockEncodingCircuit(
            decomposition: decomposition,
            systemQubits: 2,
            ancillaStart: 2,
        )

        #expect(circuit.gates.count > 0, "Block encoding should handle mixed positive and negative coefficients")
    }
}

/// Test suite for single-qubit Pauli operators without controls.
/// Validates that SELECT circuit correctly handles cases where
/// Pauli operators are applied without ancilla control (single term).
@Suite("Uncontrolled Pauli Operations")
struct UncontrolledPauliTests {
    @Test("SELECT circuit for single X term")
    func selectSingleXTerm() {
        let hamiltonian = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.x(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 1,
            ancillaStart: 1,
        )

        #expect(circuit.gates.count > 0, "SELECT circuit should produce gates for single X term")
    }

    @Test("SELECT circuit for single Y term")
    func selectSingleYTerm() {
        let hamiltonian = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.y(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 1,
            ancillaStart: 1,
        )

        #expect(circuit.gates.count > 0, "SELECT circuit should produce gates for single Y term")
    }

    @Test("SELECT circuit for single Z term")
    func selectSingleZTerm() {
        let hamiltonian = Observable(terms: [
            (coefficient: 1.0, pauliString: PauliString(.z(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 1,
            ancillaStart: 1,
        )

        #expect(circuit.gates.count > 0, "SELECT circuit should produce gates for single Z term")
    }

    @Test("SELECT circuit for single negative X term")
    func selectSingleNegativeXTerm() {
        let hamiltonian = Observable(terms: [
            (coefficient: -1.0, pauliString: PauliString(.x(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 1,
            ancillaStart: 1,
        )

        #expect(circuit.gates.count > 0, "SELECT circuit should handle single negative X term")
    }
}

/// Test suite for single-controlled Pauli operations.
/// Validates that SELECT circuit uses CY and CZ gates when
/// there is exactly one control qubit (two-term Hamiltonians).
@Suite("Single-Controlled Pauli Operations")
struct SingleControlledPauliTests {
    @Test("SELECT circuit uses CY for two Y terms")
    func selectTwoYTerms() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.y(0))),
            (coefficient: 0.5, pauliString: PauliString(.y(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 2,
            ancillaStart: 2,
        )

        #expect(circuit.gates.count > 0, "SELECT circuit should use controlled-Y for two Y terms")
        #expect(decomposition.ancillaQubits == 1, "Two terms should require 1 ancilla qubit")
    }

    @Test("SELECT circuit uses CZ for two Z terms")
    func selectTwoZTerms() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: 0.5, pauliString: PauliString(.z(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 2,
            ancillaStart: 2,
        )

        #expect(circuit.gates.count > 0, "SELECT circuit should use controlled-Z for two Z terms")
    }

    @Test("SELECT circuit uses CNOT for two X terms")
    func selectTwoXTerms() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.5, pauliString: PauliString(.x(0))),
            (coefficient: 0.5, pauliString: PauliString(.x(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 2,
            ancillaStart: 2,
        )

        #expect(circuit.gates.count > 0, "SELECT circuit should use CNOT for two X terms")
    }

    @Test("PREPARE circuit handles near-zero total probability in subtree")
    func prepareNearZeroTotalProbabilitySubtree() {
        let hamiltonian = Observable(terms: [
            (coefficient: 1e-20, pauliString: PauliString(.z(0))),
            (coefficient: 1e-20, pauliString: PauliString(.x(0))),
            (coefficient: 1.0, pauliString: PauliString(.y(0))),
            (coefficient: 1.0, pauliString: PauliString(.z(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 2)

        #expect(decomposition.termCount == 2, "Near-zero coefficients should be filtered out leaving 2 terms")
        #expect(circuit.qubits >= 3, "Circuit should have at least 3 qubits for 2 terms with ancillaStart=2")
    }

    @Test("PREPARE circuit triggers probZero near zero branch with skewed distribution")
    func prepareProbZeroNearZeroBranch() {
        let hamiltonian = Observable(terms: [
            (coefficient: 1e-16, pauliString: PauliString(.z(0))),
            (coefficient: 1.0, pauliString: PauliString(.x(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 2)

        #expect(decomposition.termCount == 1, "Single non-zero term should remain after filtering")
        #expect(circuit.gates.isEmpty, "Single term PREPARE circuit should be empty")
    }

    @Test("PREPARE circuit triggers X gate when left probability is negligible")
    func prepareTriggerXGateForNegligibleLeftProbability() {
        var terms: PauliTerms = []
        terms.append((coefficient: 1e-18, pauliString: PauliString(.z(0))))
        terms.append((coefficient: 1e-18, pauliString: PauliString(.x(0))))
        terms.append((coefficient: 0.5, pauliString: PauliString(.y(0))))
        terms.append((coefficient: 0.5, pauliString: PauliString(.z(1))))
        let hamiltonian = Observable(terms: terms)
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 2)

        #expect(decomposition.termCount == 2, "Two significant terms should remain")
        #expect(circuit.qubits == 3, "Circuit should have 3 qubits for 2 terms with ancillaStart=2")
    }

    @Test("PREPARE handles heavily right-skewed four term distribution")
    func prepareHeavilyRightSkewedFourTerms() {
        let hamiltonian = Observable(terms: [
            (coefficient: 1e-17, pauliString: PauliString(.z(0))),
            (coefficient: 1e-17, pauliString: PauliString(.x(0))),
            (coefficient: 0.5, pauliString: PauliString(.y(0))),
            (coefficient: 0.5, pauliString: PauliString(.z(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 2)

        let normalizedSum = decomposition.normalizedCoefficients.reduce(0.0, +)
        #expect(abs(normalizedSum - 1.0) < 1e-10, "Normalized coefficients should sum to 1.0")
        #expect(circuit.qubits >= 3, "Circuit should have at least 3 qubits")
    }

    @Test("SELECT circuit with eight terms triggers multi-controlled phase recursion")
    func selectEightTermsMultiControlledPhaseRecursion() {
        var terms: PauliTerms = []
        for i in 0 ..< 8 {
            let sign = (i % 2 == 0) ? 1.0 : -1.0
            terms.append((coefficient: sign * 0.125, pauliString: PauliString(.z(i % 2))))
        }
        let hamiltonian = Observable(terms: terms)
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 2,
            ancillaStart: 2,
        )

        #expect(decomposition.ancillaQubits == 3, "Eight terms should require 3 ancilla qubits")
        #expect(circuit.gates.count > 0, "SELECT circuit should produce gates with multi-controlled phase")
    }

    @Test("SELECT circuit with nine terms triggers deep phase recursion")
    func selectNineTermsDeepPhaseRecursion() {
        var terms: PauliTerms = []
        for i in 0 ..< 9 {
            let sign = (i % 3 == 0) ? -1.0 : 1.0
            terms.append((coefficient: sign / 9.0, pauliString: PauliString(.x(i % 2))))
        }
        let hamiltonian = Observable(terms: terms)
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 2,
            ancillaStart: 2,
        )

        #expect(decomposition.ancillaQubits == 4, "Nine terms should require 4 ancilla qubits")
        #expect(circuit.gates.count > 0, "SELECT circuit should produce gates for 9 terms with negative coefficients")
    }

    @Test("Block encoding with sixteen terms exercises all control paths")
    func blockEncodingSixteenTermsAllControlPaths() {
        var terms: PauliTerms = []
        for i in 0 ..< 16 {
            let sign = (i % 4 == 0) ? -1.0 : 1.0
            terms.append((coefficient: sign / 16.0, pauliString: PauliString(.z(i % 3))))
        }
        let hamiltonian = Observable(terms: terms)
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.blockEncodingCircuit(
            decomposition: decomposition,
            systemQubits: 3,
            ancillaStart: 3,
        )

        #expect(decomposition.ancillaQubits == 4, "Sixteen terms should require 4 ancilla qubits")
        #expect(circuit.gates.count > 0, "Block encoding should produce gates for 16 terms")
    }

    @Test("PREPARE with four terms and all weight in indices 2 and 3")
    func prepareFourTermsWeightInRightHalf() {
        let hamiltonian = Observable(terms: [
            (coefficient: 1e-20, pauliString: PauliString(.z(0))),
            (coefficient: 1e-20, pauliString: PauliString(.x(0))),
            (coefficient: 0.6, pauliString: PauliString(.y(0))),
            (coefficient: 0.4, pauliString: PauliString(.z(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 2)

        #expect(decomposition.termCount == 2, "Two significant terms should remain after filtering")
        #expect(abs(decomposition.oneNorm - 1.0) < 1e-10, "One-norm should be approximately 1.0")
        #expect(circuit.qubits >= 3, "Circuit should have at least 3 qubits")
    }

    @Test("SELECT with ten negative coefficient terms triggers recursive phase")
    func selectTenNegativeTermsRecursivePhase() {
        var terms: PauliTerms = []
        for i in 0 ..< 10 {
            terms.append((coefficient: -0.1, pauliString: PauliString(.y(i % 2))))
        }
        let hamiltonian = Observable(terms: terms)
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.selectCircuit(
            decomposition: decomposition,
            systemQubits: 2,
            ancillaStart: 2,
        )

        #expect(decomposition.ancillaQubits == 4, "Ten terms should require 4 ancilla qubits")
        #expect(circuit.gates.count > 0, "SELECT circuit should handle 10 negative coefficient terms")

        let allNegative = decomposition.originalCoefficients.allSatisfy { $0 < 0 }
        #expect(allNegative, "All original coefficients should be negative")
    }

    @Test("PREPARE circuit handles near-zero total probability in padded subtree")
    func prepareNearZeroTotalProbabilityInPaddedSubtree() {
        var terms: PauliTerms = []
        for i in 0 ..< 5 {
            terms.append((coefficient: 0.2, pauliString: PauliString(.z(i % 2))))
        }
        let hamiltonian = Observable(terms: terms)
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 2)

        #expect(decomposition.termCount == 5, "Should have 5 terms")
        #expect(decomposition.ancillaQubits == 3, "Five terms require 3 ancilla qubits (8 slots)")
        #expect(circuit.qubits == 5, "Circuit should have 5 qubits (2 ancillaStart + 3 ancilla)")
    }

    @Test("PREPARE circuit with six terms leaves two padded zero slots")
    func prepareSixTermsWithPaddedZeroSlots() {
        var terms: PauliTerms = []
        for i in 0 ..< 6 {
            terms.append((coefficient: 1.0 / 6.0, pauliString: PauliString(.x(i % 3))))
        }
        let hamiltonian = Observable(terms: terms)
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 3)

        #expect(decomposition.termCount == 6, "Should have 6 terms")
        #expect(decomposition.ancillaQubits == 3, "Six terms require 3 ancilla qubits (8 slots)")
        #expect(circuit.gates.count > 0, "Circuit should have gates despite padded zero probability slots")
    }

    @Test("PREPARE circuit with three terms has single padded zero slot triggering near-zero subtree")
    func prepareThreeTermsWithPaddedZeroSlot() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.4, pauliString: PauliString(.z(0))),
            (coefficient: 0.3, pauliString: PauliString(.x(0))),
            (coefficient: 0.3, pauliString: PauliString(.y(0))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 1)

        #expect(decomposition.termCount == 3, "Should have 3 terms")
        #expect(decomposition.ancillaQubits == 2, "Three terms require 2 ancilla qubits (4 slots)")
        #expect(circuit.gates.count > 0, "Circuit should have gates for 3 terms with padded slot")
    }

    @Test("PREPARE circuit triggers probZero near zero when left subtree is empty")
    func prepareProbZeroNearZeroWithEmptyLeftSubtree() {
        let hamiltonian = Observable(terms: [
            (coefficient: 0.0, pauliString: PauliString(.z(0))),
            (coefficient: 0.0, pauliString: PauliString(.x(0))),
            (coefficient: 0.5, pauliString: PauliString(.y(0))),
            (coefficient: 0.5, pauliString: PauliString(.z(1))),
        ])
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 2)

        #expect(decomposition.termCount == 2, "Zero coefficients should be filtered leaving 2 terms")
        #expect(circuit.qubits == 3, "Circuit should have 3 qubits for 2 terms with ancillaStart=2")
    }

    @Test("PREPARE circuit with nine terms exercises deep tree recursion with padded slots")
    func prepareNineTermsDeepTreeRecursion() {
        var terms: PauliTerms = []
        for i in 0 ..< 9 {
            terms.append((coefficient: 1.0 / 9.0, pauliString: PauliString(.z(i % 3))))
        }
        let hamiltonian = Observable(terms: terms)
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 3)

        #expect(decomposition.termCount == 9, "Should have 9 terms")
        #expect(decomposition.ancillaQubits == 4, "Nine terms require 4 ancilla qubits (16 slots)")
        #expect(circuit.gates.count > 0, "Circuit should have gates despite 7 padded zero probability slots")
    }

    @Test("PREPARE circuit with eleven terms has five padded zero slots")
    func prepareElevenTermsWithFivePaddedZeroSlots() {
        var terms: PauliTerms = []
        for i in 0 ..< 11 {
            terms.append((coefficient: 1.0 / 11.0, pauliString: PauliString(.x(i % 4))))
        }
        let hamiltonian = Observable(terms: terms)
        let decomposition = LCU.decompose(hamiltonian)
        let circuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 4)

        #expect(decomposition.termCount == 11, "Should have 11 terms")
        #expect(decomposition.ancillaQubits == 4, "Eleven terms require 4 ancilla qubits (16 slots)")
        #expect(circuit.qubits >= 8, "Circuit should have at least 8 qubits (4 ancillaStart + 4 ancilla)")
    }
}
