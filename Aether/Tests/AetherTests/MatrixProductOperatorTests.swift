// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Tests MPO initialization from Observable with weighted Pauli terms.
/// Validates correct tensor structure and site count for various observables.
/// Ensures MPO correctly represents Hamiltonian operators from Pauli decomposition.
@Suite("MPO Init Observable")
struct MPOInitObservableTests {
    @Test("Single Z term creates correct site count")
    func singleZTermSiteCount() {
        let observable = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let mpo = MatrixProductOperator(observable: observable)
        #expect(mpo.sites == 1, "Single qubit observable should create MPO with 1 site")
    }

    @Test("Two-qubit ZZ term creates correct site count")
    func twoQubitZZSiteCount() {
        let observable = Observable(coefficient: 1.0, pauliString: PauliString(.z(0), .z(1)))
        let mpo = MatrixProductOperator(observable: observable)
        #expect(mpo.sites == 2, "Two-qubit observable should create MPO with 2 sites")
    }

    @Test("Observable with gap in qubit indices creates correct site count")
    func observableWithGapSiteCount() {
        let observable = Observable(coefficient: 1.0, pauliString: PauliString(.z(0), .z(3)))
        let mpo = MatrixProductOperator(observable: observable)
        #expect(mpo.sites == 4, "Observable on qubits 0 and 3 should create MPO with 4 sites")
    }

    @Test("Multi-term observable creates correct site count")
    func multiTermObservableSiteCount() {
        let terms: PauliTerms = [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: 0.5, pauliString: PauliString(.z(1))),
            (coefficient: 1.0, pauliString: PauliString(.x(0), .x(1))),
        ]
        let observable = Observable(terms: terms)
        let mpo = MatrixProductOperator(observable: observable)
        #expect(mpo.sites == 2, "Multi-term observable on qubits 0,1 should create MPO with 2 sites")
    }

    @Test("Observable creates correct tensor count")
    func observableCreatesTensorCount() {
        let observable = Observable(coefficient: 1.0, pauliString: PauliString(.x(0), .y(1), .z(2)))
        let mpo = MatrixProductOperator(observable: observable)
        #expect(mpo.tensors.count == 3, "3-qubit observable should create MPO with 3 tensors")
    }
}

/// Tests MPO initialization from PauliString with explicit site count.
/// Validates tensor chain structure and boundary conditions for Pauli operators.
/// Ensures correct tensor dimensions for single Pauli string operators.
@Suite("MPO Init PauliString")
struct MPOInitPauliStringTests {
    @Test("Single Z Pauli string creates correct MPO")
    func singleZPauliString() {
        let pauliString = PauliString(.z(0))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 2)
        #expect(mpo.sites == 2, "MPO should have 2 sites as specified")
        #expect(mpo.tensors.count == 2, "MPO should have 2 tensors")
    }

    @Test("Identity on all sites has correct structure")
    func identityPauliString() {
        let pauliString = PauliString([])
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 3)
        #expect(mpo.sites == 3, "Identity MPO should have 3 sites")
        #expect(mpo.tensors.count == 3, "Identity MPO should have 3 tensors")
    }

    @Test("First tensor has left bond dimension 1")
    func firstTensorLeftBondDimension() {
        let pauliString = PauliString(.x(0), .z(1))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 3)
        #expect(mpo.tensors[0].leftBondDimension == 1, "First tensor should have left bond dimension 1")
    }

    @Test("Last tensor has right bond dimension 1")
    func lastTensorRightBondDimension() {
        let pauliString = PauliString(.x(0), .z(2))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 3)
        #expect(mpo.tensors[2].rightBondDimension == 1, "Last tensor should have right bond dimension 1")
    }

    @Test("XYZ Pauli string creates MPO with correct site count")
    func xyzPauliStringSiteCount() {
        let pauliString = PauliString(.x(0), .y(1), .z(2))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 4)
        #expect(mpo.sites == 4, "XYZ MPO with 4 sites should have 4 sites")
    }
}

/// Tests applying MPO to MPS preserves state norm approximately.
/// Validates MPO-MPS contraction maintains quantum state normalization.
/// Ensures truncation does not drastically affect state fidelity for simple cases.
@Suite("MPO Applying Norm Preservation")
struct MPOApplyingNormPreservationTests {
    @Test("Identity MPO preserves norm exactly")
    func identityMPOPreservesNorm() {
        let pauliString = PauliString([])
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 2)
        let mps = MatrixProductState(qubits: 2)
        let result = mpo.applying(to: mps, truncation: .maxBondDimension(16))
        let normSquared = result.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-10, "Identity MPO should preserve norm exactly")
    }

    @Test("Z operator on ground state preserves norm")
    func zOperatorPreservesNorm() {
        let pauliString = PauliString(.z(0))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 2)
        let mps = MatrixProductState(qubits: 2)
        let result = mpo.applying(to: mps, truncation: .maxBondDimension(16))
        let normSquared = result.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-10, "Z operator on |00> should preserve norm")
    }

    @Test("X operator on ground state preserves norm")
    func xOperatorPreservesNorm() {
        let pauliString = PauliString(.x(0))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 2)
        let mps = MatrixProductState(qubits: 2)
        let result = mpo.applying(to: mps, truncation: .maxBondDimension(16))
        let normSquared = result.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-10, "X operator on |00> should preserve norm")
    }

    @Test("Y operator on ground state preserves norm")
    func yOperatorPreservesNorm() {
        let pauliString = PauliString(.y(0))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 2)
        let mps = MatrixProductState(qubits: 2)
        let result = mpo.applying(to: mps, truncation: .maxBondDimension(16))
        let normSquared = result.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-10, "Y operator on |00> should preserve norm")
    }

    @Test("ZZ operator on ground state preserves norm")
    func zzOperatorPreservesNorm() {
        let pauliString = PauliString(.z(0), .z(1))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 2)
        let mps = MatrixProductState(qubits: 2)
        let result = mpo.applying(to: mps, truncation: .maxBondDimension(16))
        let normSquared = result.normSquared()
        #expect(abs(normSquared - 1.0) < 1e-10, "ZZ operator on |00> should preserve norm")
    }
}

/// Tests expectation value calculation for identity operator.
/// Validates that identity MPO gives expectation 1 for normalized states.
/// Ensures transfer matrix contraction produces correct boundary conditions.
@Suite("MPO Expectation Value Identity")
struct MPOExpectationValueIdentityTests {
    @Test("Identity expectation on ground state is 1")
    func identityExpectationGroundState() {
        let pauliString = PauliString([])
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 2)
        let mps = MatrixProductState(qubits: 2)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation - 1.0) < 1e-10, "Identity expectation on |00> should be 1")
    }

    @Test("Identity expectation on basis state is 1")
    func identityExpectationBasisState() {
        let pauliString = PauliString([])
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 3)
        let mps = MatrixProductState(qubits: 3, basisState: 5)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation - 1.0) < 1e-10, "Identity expectation on |101> should be 1")
    }

    @Test("Identity expectation on single qubit state is 1")
    func identityExpectationSingleQubit() {
        let pauliString = PauliString([])
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 1)
        let mps = MatrixProductState(qubits: 1)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation - 1.0) < 1e-10, "Identity expectation on single qubit |0> should be 1")
    }
}

/// Tests expectation value calculation for Z operator on known states.
/// Validates Z eigenvalue expectations for computational basis states.
/// Ensures MPO transfer matrix correctly computes Pauli Z expectations.
@Suite("MPO Expectation Value Z Operator")
struct MPOExpectationValueZOperatorTests {
    @Test("Z expectation on |0> is +1")
    func zExpectationOnZero() {
        let pauliString = PauliString(.z(0))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 1)
        let mps = MatrixProductState(qubits: 1)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation - 1.0) < 1e-10, "Z expectation on |0> should be +1")
    }

    @Test("Z expectation on |1> is -1")
    func zExpectationOnOne() {
        let pauliString = PauliString(.z(0))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 1)
        let mps = MatrixProductState(qubits: 1, basisState: 1)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation + 1.0) < 1e-10, "Z expectation on |1> should be -1")
    }

    @Test("Z0 expectation on |00> is +1")
    func z0ExpectationOnZeroZero() {
        let pauliString = PauliString(.z(0))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 2)
        let mps = MatrixProductState(qubits: 2)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation - 1.0) < 1e-10, "Z0 expectation on |00> should be +1")
    }

    @Test("Z1 expectation on |01> is +1")
    func z1ExpectationOnZeroOne() {
        let pauliString = PauliString(.z(1))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 2)
        let mps = MatrixProductState(qubits: 2, basisState: 1)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation - 1.0) < 1e-10, "Z1 expectation on |01> should be +1")
    }

    @Test("ZZ expectation on |00> is +1")
    func zzExpectationOnZeroZero() {
        let pauliString = PauliString(.z(0), .z(1))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 2)
        let mps = MatrixProductState(qubits: 2)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation - 1.0) < 1e-10, "ZZ expectation on |00> should be +1")
    }

    @Test("ZZ expectation on |01> is -1")
    func zzExpectationOnZeroOne() {
        let pauliString = PauliString(.z(0), .z(1))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 2)
        let mps = MatrixProductState(qubits: 2, basisState: 1)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation + 1.0) < 1e-10, "ZZ expectation on |01> should be -1")
    }

    @Test("ZZ expectation on |10> is -1")
    func zzExpectationOnOneZero() {
        let pauliString = PauliString(.z(0), .z(1))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 2)
        let mps = MatrixProductState(qubits: 2, basisState: 2)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation + 1.0) < 1e-10, "ZZ expectation on |10> should be -1")
    }

    @Test("ZZ expectation on |11> is +1")
    func zzExpectationOnOneOne() {
        let pauliString = PauliString(.z(0), .z(1))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 2)
        let mps = MatrixProductState(qubits: 2, basisState: 3)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation - 1.0) < 1e-10, "ZZ expectation on |11> should be +1")
    }
}

/// Tests MPO sites property returns correct qubit count.
/// Validates that sites matches the number of qubits in the operator chain.
/// Ensures proper system size determination from observables and Pauli strings.
@Suite("MPO Sites Property")
struct MPOSitesPropertyTests {
    @Test("Sites property matches observable qubit count")
    func sitesMatchesObservableQubitCount() {
        let observable = Observable(coefficient: 1.0, pauliString: PauliString(.z(0), .z(2)))
        let mpo = MatrixProductOperator(observable: observable)
        #expect(mpo.sites == 3, "Sites should be 3 for observable on qubits 0 and 2")
    }

    @Test("Sites property matches explicit site count")
    func sitesMatchesExplicitSiteCount() {
        let pauliString = PauliString(.x(0))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 5)
        #expect(mpo.sites == 5, "Sites should be 5 as explicitly specified")
    }

    @Test("Single site MPO has sites equal to 1")
    func singleSiteMPO() {
        let observable = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let mpo = MatrixProductOperator(observable: observable)
        #expect(mpo.sites == 1, "Single qubit observable should have 1 site")
    }
}

/// Tests MPO tensors property returns correct tensor array.
/// Validates tensor count, bond dimensions, and physical dimensions.
/// Ensures proper tensor network structure for quantum operators.
@Suite("MPO Tensors Property")
struct MPOTensorsPropertyTests {
    @Test("Tensors count equals sites")
    func tensorsCountEqualsSites() {
        let pauliString = PauliString(.z(0), .z(1), .z(2))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 4)
        #expect(mpo.tensors.count == 4, "Tensor count should equal site count")
    }

    @Test("All tensors have physical dimension 2")
    func allTensorsPhysicalDimension2() {
        let observable = Observable(coefficient: 1.0, pauliString: PauliString(.x(0), .y(1)))
        let mpo = MatrixProductOperator(observable: observable)
        for (index, tensor) in mpo.tensors.enumerated() {
            #expect(tensor.physicalDimension == 2, "Tensor at site \(index) should have physical dimension 2")
        }
    }

    @Test("First tensor left bond dimension is 1")
    func firstTensorLeftBond() {
        let observable = Observable(coefficient: 1.0, pauliString: PauliString(.z(0), .z(1), .z(2)))
        let mpo = MatrixProductOperator(observable: observable)
        #expect(mpo.tensors[0].leftBondDimension == 1, "First tensor left bond dimension should be 1")
    }

    @Test("Last tensor right bond dimension is 1")
    func lastTensorRightBond() {
        let observable = Observable(coefficient: 1.0, pauliString: PauliString(.z(0), .z(1), .z(2)))
        let mpo = MatrixProductOperator(observable: observable)
        #expect(mpo.tensors[2].rightBondDimension == 1, "Last tensor right bond dimension should be 1")
    }

    @Test("Single site MPO has correct tensor structure")
    func singleSiteTensorStructure() {
        let pauliString = PauliString(.z(0))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 1)
        #expect(mpo.tensors.count == 1, "Single site MPO should have 1 tensor")
        #expect(mpo.tensors[0].leftBondDimension == 1, "Single site tensor left bond should be 1")
        #expect(mpo.tensors[0].rightBondDimension == 1, "Single site tensor right bond should be 1")
    }
}

/// Tests MPO expectation value with Observable containing multiple terms.
/// Validates weighted sum of Pauli expectations for Hamiltonian operators.
/// Ensures correct energy calculations for simple quantum systems.
@Suite("MPO Observable Expectation")
struct MPOObservableExpectationTests {
    @Test("Sum of Z operators on ground state")
    func sumOfZOperatorsGroundState() {
        let terms: PauliTerms = [
            (coefficient: 0.5, pauliString: PauliString(.z(0))),
            (coefficient: 0.5, pauliString: PauliString(.z(1))),
        ]
        let observable = Observable(terms: terms)
        let mpo = MatrixProductOperator(observable: observable)
        let mps = MatrixProductState(qubits: 2)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation - 1.0) < 1e-10, "Sum of Z expectations on |00> should be 0.5+0.5=1.0")
    }

    @Test("Ising-like Hamiltonian on ground state")
    func isingHamiltonianGroundState() {
        let terms: PauliTerms = [
            (coefficient: -1.0, pauliString: PauliString(.z(0), .z(1))),
        ]
        let observable = Observable(terms: terms)
        let mpo = MatrixProductOperator(observable: observable)
        let mps = MatrixProductState(qubits: 2)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation + 1.0) < 1e-10, "Ising ZZ term on |00>: -1 * 1 = -1")
    }

    @Test("Weighted Z observable with coefficient")
    func weightedZObservable() {
        let observable = Observable(coefficient: 2.0, pauliString: PauliString(.z(0)))
        let mpo = MatrixProductOperator(observable: observable)
        let mps = MatrixProductState(qubits: 1, basisState: 1)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation + 2.0) < 1e-10, "2*Z on |1> should be 2*(-1) = -2")
    }
}

/// Tests MPO applying produces correct transformed states.
/// Validates operator action on MPS produces expected quantum states.
/// Ensures X operator flips basis states correctly via MPO application.
@Suite("MPO Applying State Transformation")
struct MPOApplyingStateTransformationTests {
    @Test("X operator flips |0> to |1>")
    func xOperatorFlipsZeroToOne() {
        let pauliString = PauliString(.x(0))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 1)
        let mps = MatrixProductState(qubits: 1)
        var result = mpo.applying(to: mps, truncation: .maxBondDimension(16))
        result.normalize()
        let prob0 = result.probability(of: 0)
        let prob1 = result.probability(of: 1)
        #expect(abs(prob0) < 1e-10, "X|0> should have zero probability at |0>")
        #expect(abs(prob1 - 1.0) < 1e-10, "X|0> should have probability 1 at |1>")
    }

    @Test("X operator flips |1> to |0>")
    func xOperatorFlipsOneToZero() {
        let pauliString = PauliString(.x(0))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 1)
        let mps = MatrixProductState(qubits: 1, basisState: 1)
        var result = mpo.applying(to: mps, truncation: .maxBondDimension(16))
        result.normalize()
        let prob0 = result.probability(of: 0)
        let prob1 = result.probability(of: 1)
        #expect(abs(prob0 - 1.0) < 1e-10, "X|1> should have probability 1 at |0>")
        #expect(abs(prob1) < 1e-10, "X|1> should have zero probability at |1>")
    }

    @Test("Z operator keeps |0> as |0>")
    func zOperatorKeepsZero() {
        let pauliString = PauliString(.z(0))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 1)
        let mps = MatrixProductState(qubits: 1)
        var result = mpo.applying(to: mps, truncation: .maxBondDimension(16))
        result.normalize()
        let prob0 = result.probability(of: 0)
        #expect(abs(prob0 - 1.0) < 1e-10, "Z|0> should remain |0>")
    }
}

/// Tests MPO Sendable conformance for concurrent usage.
/// Validates MatrixProductOperator can be safely shared across threads.
/// Ensures thread-safe tensor network operations in Swift concurrency.
@Suite("MPO Sendable Conformance")
struct MPOSendableConformanceTests {
    @Test("MPO is Sendable")
    func mpoIsSendable() {
        let observable = Observable(coefficient: 1.0, pauliString: PauliString(.z(0)))
        let mpo = MatrixProductOperator(observable: observable)
        _ = mpo as Sendable
    }
}

/// Tests MPO with three-qubit systems for larger operator chains.
/// Validates tensor network contraction for multi-site operators.
/// Ensures correct expectation values for extended quantum systems.
@Suite("MPO Three Qubit Systems")
struct MPOThreeQubitSystemsTests {
    @Test("ZZZ expectation on |000> is +1")
    func zzzExpectationOnAllZeros() {
        let pauliString = PauliString(.z(0), .z(1), .z(2))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 3)
        let mps = MatrixProductState(qubits: 3)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation - 1.0) < 1e-10, "ZZZ expectation on |000> should be +1")
    }

    @Test("ZZZ expectation on |001> is -1")
    func zzzExpectationOnZeroZeroOne() {
        let pauliString = PauliString(.z(0), .z(1), .z(2))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 3)
        let mps = MatrixProductState(qubits: 3, basisState: 1)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation + 1.0) < 1e-10, "ZZZ expectation on |001> should be -1")
    }

    @Test("ZZZ expectation on |111> is -1")
    func zzzExpectationOnAllOnes() {
        let pauliString = PauliString(.z(0), .z(1), .z(2))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 3)
        let mps = MatrixProductState(qubits: 3, basisState: 7)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation + 1.0) < 1e-10, "ZZZ expectation on |111> should be -1")
    }

    @Test("Three-qubit identity expectation is 1")
    func threeQubitIdentityExpectation() {
        let pauliString = PauliString([])
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 3)
        let mps = MatrixProductState(qubits: 3, basisState: 5)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation - 1.0) < 1e-10, "Identity expectation on |101> should be 1")
    }
}

/// Tests MPO X operator expectation values on various states.
/// Validates that X expectation is 0 on Z-basis states due to orthogonality.
/// Ensures correct off-diagonal operator behavior in MPO formulation.
@Suite("MPO X Operator Expectation")
struct MPOXOperatorExpectationTests {
    @Test("X expectation on |0> is 0")
    func xExpectationOnZero() {
        let pauliString = PauliString(.x(0))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 1)
        let mps = MatrixProductState(qubits: 1)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation) < 1e-10, "X expectation on |0> should be 0")
    }

    @Test("X expectation on |1> is 0")
    func xExpectationOnOne() {
        let pauliString = PauliString(.x(0))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 1)
        let mps = MatrixProductState(qubits: 1, basisState: 1)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation) < 1e-10, "X expectation on |1> should be 0")
    }

    @Test("XX expectation on |00> is 0")
    func xxExpectationOnZeroZero() {
        let pauliString = PauliString(.x(0), .x(1))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 2)
        let mps = MatrixProductState(qubits: 2)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation) < 1e-10, "XX expectation on |00> should be 0")
    }
}

/// Tests MPO with sparse Pauli operators (non-contiguous qubits).
/// Validates identity tensors are correctly inserted at intermediate sites.
/// Ensures proper handling of operators acting on distant qubits.
@Suite("MPO Sparse Pauli Operators")
struct MPOSparsePauliOperatorsTests {
    @Test("Z on qubit 2 with 4 sites creates correct structure")
    func zOnQubitTwoWithFourSites() {
        let pauliString = PauliString(.z(2))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 4)
        #expect(mpo.tensors.count == 4, "MPO should have 4 tensors")
        #expect(mpo.sites == 4, "MPO should have 4 sites")
    }

    @Test("Z0 Z3 expectation on |0000> is +1")
    func z0z3ExpectationOnAllZeros() {
        let pauliString = PauliString(.z(0), .z(3))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 4)
        let mps = MatrixProductState(qubits: 4)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation - 1.0) < 1e-10, "Z0 Z3 expectation on |0000> should be +1")
    }

    @Test("Z0 Z3 expectation on |1001> is +1")
    func z0z3ExpectationOnOneZeroZeroOne() {
        let pauliString = PauliString(.z(0), .z(3))
        let mpo = MatrixProductOperator(pauliString: pauliString, sites: 4)
        let mps = MatrixProductState(qubits: 4, basisState: 9)
        let expectation = mpo.expectationValue(bra: mps, ket: mps)
        #expect(abs(expectation - 1.0) < 1e-10, "Z0 Z3 expectation on |1001> should be (-1)*(-1) = +1")
    }
}
