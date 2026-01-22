// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for 1D Ising chain Hamiltonian construction.
/// Validates term counts, ZZ coupling coefficients, and X field terms
/// for both open and periodic boundary conditions.
@Suite("1D Ising Chain Hamiltonian")
struct IsingChainTests {
    @Test("Chain with 4 qubits produces correct term count (open boundary)")
    func chainOpenBoundaryTermCount() {
        let chain = IsingHamiltonian.chain(qubits: 4, J: 1.0, h: 0.5, periodic: false)
        let expectedZZTerms = 3
        let expectedXTerms = 4
        let expectedTotal = expectedZZTerms + expectedXTerms
        #expect(chain.terms.count == expectedTotal, "4-qubit open chain should have 7 terms (3 ZZ + 4 X)")
    }

    @Test("Chain with 4 qubits produces correct term count (periodic boundary)")
    func chainPeriodicBoundaryTermCount() {
        let chain = IsingHamiltonian.chain(qubits: 4, J: 1.0, h: 0.5, periodic: true)
        let expectedZZTerms = 4
        let expectedXTerms = 4
        let expectedTotal = expectedZZTerms + expectedXTerms
        #expect(chain.terms.count == expectedTotal, "4-qubit periodic chain should have 8 terms (4 ZZ + 4 X)")
    }

    @Test("Chain with 1 qubit produces only X term")
    func chainSingleQubit() {
        let chain = IsingHamiltonian.chain(qubits: 1, J: 1.0, h: 0.5, periodic: false)
        #expect(chain.terms.count == 1, "1-qubit chain should have exactly 1 X term")
        #expect(chain.terms[0].pauliString.operators.count == 1, "Single term should have one operator")
        #expect(chain.terms[0].pauliString.operators[0].basis == .x, "Single term should be X operator")
    }

    @Test("ZZ coupling coefficients are -J")
    func chainZZCouplingCoefficients() {
        let J = 2.5
        let chain = IsingHamiltonian.chain(qubits: 3, J: J, h: 0.5, periodic: false)
        let zzTerms = chain.terms.filter { $0.pauliString.operators.count == 2 }
        for term in zzTerms {
            #expect(abs(term.coefficient - -J) < 1e-10, "ZZ term coefficient should be -J = \(-J)")
        }
    }

    @Test("X field coefficients are -h")
    func chainXFieldCoefficients() {
        let h = 0.75
        let chain = IsingHamiltonian.chain(qubits: 3, J: 1.0, h: h, periodic: false)
        let xTerms = chain.terms.filter { $0.pauliString.operators.count == 1 }
        for term in xTerms {
            #expect(abs(term.coefficient - -h) < 1e-10, "X term coefficient should be -h = \(-h)")
        }
    }

    @Test("Periodic chain includes boundary term Z_{n-1}Z_0")
    func chainPeriodicBoundaryTerm() {
        let chain = IsingHamiltonian.chain(qubits: 4, J: 1.0, h: 0.5, periodic: true)
        let zzTerms = chain.terms.filter { $0.pauliString.operators.count == 2 }
        let hasBoundaryTerm = zzTerms.contains { term in
            let qubits = Set(term.pauliString.operators.map(\.qubit))
            return qubits.contains(0) && qubits.contains(3)
        }
        #expect(hasBoundaryTerm, "Periodic chain should contain Z_3Z_0 boundary term")
    }

    @Test("Open chain does not include boundary term")
    func chainOpenBoundaryNoBoundaryTerm() {
        let chain = IsingHamiltonian.chain(qubits: 4, J: 1.0, h: 0.5, periodic: false)
        let zzTerms = chain.terms.filter { $0.pauliString.operators.count == 2 }
        let hasBoundaryTerm = zzTerms.contains { term in
            let qubits = Set(term.pauliString.operators.map(\.qubit))
            return qubits.contains(0) && qubits.contains(3)
        }
        #expect(!hasBoundaryTerm, "Open chain should not contain Z_3Z_0 boundary term")
    }
}

/// Test suite for 2D Ising lattice Hamiltonian construction.
/// Validates grid topology, edge counts, and periodic torus topology
/// for rectangular qubit grids.
@Suite("2D Ising Lattice Hamiltonian")
struct IsingLatticeTests {
    @Test("2x3 lattice produces correct term count (open boundary)")
    func latticeOpenBoundaryTermCount() {
        let lattice = IsingHamiltonian.lattice(rows: 2, cols: 3, J: 1.0, h: 0.5, periodic: false)
        let horizontalEdges = 2 * (3 - 1)
        let verticalEdges = (2 - 1) * 3
        let xTerms = 2 * 3
        let expectedTotal = horizontalEdges + verticalEdges + xTerms
        #expect(lattice.terms.count == expectedTotal, "2x3 open lattice should have \(expectedTotal) terms")
    }

    @Test("3x3 lattice produces correct term count (periodic boundary)")
    func latticePeriodicBoundaryTermCount() {
        let lattice = IsingHamiltonian.lattice(rows: 3, cols: 3, J: 1.0, h: 0.5, periodic: true)
        let horizontalEdges = 3 * 3
        let verticalEdges = 3 * 3
        let xTerms = 3 * 3
        let expectedTotal = horizontalEdges + verticalEdges + xTerms
        #expect(lattice.terms.count == expectedTotal, "3x3 periodic lattice should have \(expectedTotal) terms")
    }

    @Test("1x1 lattice produces only single X term")
    func latticeSingleSite() {
        let lattice = IsingHamiltonian.lattice(rows: 1, cols: 1, J: 1.0, h: 0.5, periodic: false)
        #expect(lattice.terms.count == 1, "1x1 lattice should have exactly 1 X term")
    }

    @Test("Lattice qubit indexing follows row-major order")
    func latticeQubitIndexing() {
        let lattice = IsingHamiltonian.lattice(rows: 2, cols: 3, J: 1.0, h: 0.5, periodic: false)
        let xTerms = lattice.terms.filter { $0.pauliString.operators.count == 1 }
        let xQubits = xTerms.map { $0.pauliString.operators[0].qubit }.sorted()
        let expectedQubits = [0, 1, 2, 3, 4, 5]
        #expect(xQubits == expectedQubits, "X terms should cover qubits 0-5 in row-major order")
    }

    @Test("Periodic lattice includes horizontal wrap-around edges")
    func latticePeriodicHorizontalWrap() {
        let lattice = IsingHamiltonian.lattice(rows: 2, cols: 3, J: 1.0, h: 0.5, periodic: true)
        let zzTerms = lattice.terms.filter { $0.pauliString.operators.count == 2 }
        let hasWrapRow0 = zzTerms.contains { term in
            let qubits = Set(term.pauliString.operators.map(\.qubit))
            return qubits.contains(0) && qubits.contains(2)
        }
        let hasWrapRow1 = zzTerms.contains { term in
            let qubits = Set(term.pauliString.operators.map(\.qubit))
            return qubits.contains(3) && qubits.contains(5)
        }
        #expect(hasWrapRow0, "Periodic lattice should wrap row 0: Z_2Z_0")
        #expect(hasWrapRow1, "Periodic lattice should wrap row 1: Z_5Z_3")
    }

    @Test("Periodic lattice includes vertical wrap-around edges")
    func latticePeriodicVerticalWrap() {
        let lattice = IsingHamiltonian.lattice(rows: 2, cols: 3, J: 1.0, h: 0.5, periodic: true)
        let zzTerms = lattice.terms.filter { $0.pauliString.operators.count == 2 }
        let hasWrapCol0 = zzTerms.contains { term in
            let qubits = Set(term.pauliString.operators.map(\.qubit))
            return qubits.contains(0) && qubits.contains(3)
        }
        let hasWrapCol1 = zzTerms.contains { term in
            let qubits = Set(term.pauliString.operators.map(\.qubit))
            return qubits.contains(1) && qubits.contains(4)
        }
        #expect(hasWrapCol0, "Periodic lattice should wrap column 0: Z_3Z_0")
        #expect(hasWrapCol1, "Periodic lattice should wrap column 1: Z_4Z_1")
    }
}

/// Test suite for custom coupling Ising Hamiltonian construction.
/// Validates arbitrary graph topologies, non-uniform couplings, and
/// site-dependent transverse fields.
@Suite("Custom Coupling Ising Hamiltonian")
struct IsingFromCouplingsTests {
    @Test("Triangle graph produces correct term count")
    func fromCouplingsTriangleTermCount() {
        let couplings = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
        let fields: [Int: Double] = [0: 0.5, 1: 0.5, 2: 0.5]
        let hamiltonian = IsingHamiltonian.fromCouplings(zzCouplings: couplings, xFields: fields, qubits: 3)
        #expect(hamiltonian.terms.count == 6, "Triangle with 3 X fields should have 6 terms")
    }

    @Test("Non-uniform coupling strengths are preserved")
    func fromCouplingsNonUniformStrengths() {
        let couplings = [(0, 1, 1.5), (1, 2, 2.0)]
        let fields: [Int: Double] = [:]
        let hamiltonian = IsingHamiltonian.fromCouplings(zzCouplings: couplings, xFields: fields, qubits: 3)
        let coefficients = hamiltonian.terms.map(\.coefficient).sorted()
        #expect(abs(coefficients[0] - -2.0) < 1e-10, "Larger coupling should produce coefficient -2.0")
        #expect(abs(coefficients[1] - -1.5) < 1e-10, "Smaller coupling should produce coefficient -1.5")
    }

    @Test("Site-dependent fields are correctly assigned")
    func fromCouplingsSiteDependentFields() {
        let couplings: [(Int, Int, Double)] = []
        let fields: [Int: Double] = [0: 0.3, 1: 0.7, 2: 0.5]
        let hamiltonian = IsingHamiltonian.fromCouplings(zzCouplings: couplings, xFields: fields, qubits: 3)
        let xTerms = hamiltonian.terms.filter { $0.pauliString.operators.count == 1 }
        for term in xTerms {
            let qubit = term.pauliString.operators[0].qubit
            let expectedCoeff = -fields[qubit]!
            #expect(abs(term.coefficient - expectedCoeff) < 1e-10, "X term on qubit \(qubit) should have coefficient \(expectedCoeff)")
        }
    }

    @Test("Empty couplings and fields produce empty Hamiltonian")
    func fromCouplingsEmpty() {
        let hamiltonian = IsingHamiltonian.fromCouplings(zzCouplings: [], xFields: [:], qubits: 3)
        #expect(hamiltonian.terms.isEmpty, "Empty couplings and fields should produce empty Hamiltonian")
    }

    @Test("Star graph topology is correctly constructed")
    func fromCouplingsStarGraph() {
        let couplings = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)]
        let fields: [Int: Double] = [:]
        let hamiltonian = IsingHamiltonian.fromCouplings(zzCouplings: couplings, xFields: fields, qubits: 4)
        let zzTerms = hamiltonian.terms.filter { $0.pauliString.operators.count == 2 }
        #expect(zzTerms.count == 3, "Star graph should have 3 ZZ edges")
        for term in zzTerms {
            let qubits = term.pauliString.operators.map(\.qubit)
            #expect(qubits.contains(0), "All star edges should connect to central qubit 0")
        }
    }
}

/// Test suite for long-range Ising Hamiltonian with power-law decay.
/// Validates all-to-all coupling structure, distance-dependent strengths,
/// and limiting behavior for different alpha exponents.
@Suite("Long-Range Ising Hamiltonian")
struct IsingLongRangeTests {
    @Test("Long-range model produces O(n^2) ZZ terms")
    func longRangeTermCount() {
        let n = 5
        let hamiltonian = IsingHamiltonian.longRange(qubits: n, J0: 1.0, alpha: 1.0, h: 0.5)
        let expectedZZTerms = n * (n - 1) / 2
        let expectedXTerms = n
        let expectedTotal = expectedZZTerms + expectedXTerms
        #expect(hamiltonian.terms.count == expectedTotal, "5-qubit long-range should have \(expectedTotal) terms")
    }

    @Test("Power-law decay J(r) = J0/r^alpha is correct")
    func longRangePowerLawDecay() {
        let J0 = 2.0
        let alpha = 1.5
        let hamiltonian = IsingHamiltonian.longRange(qubits: 4, J0: J0, alpha: alpha, h: 0.0)
        let zzTerms = hamiltonian.terms.filter { $0.pauliString.operators.count == 2 }
        for term in zzTerms {
            let qubits = term.pauliString.operators.map(\.qubit).sorted()
            let distance = Double(qubits[1] - qubits[0])
            let expectedCoeff = -J0 / pow(distance, alpha)
            #expect(abs(term.coefficient - expectedCoeff) < 1e-10, "Coupling at distance \(distance) should be \(expectedCoeff)")
        }
    }

    @Test("Alpha = 0 produces uniform all-to-all coupling")
    func longRangeAlphaZeroUniform() {
        let J0 = 1.0
        let hamiltonian = IsingHamiltonian.longRange(qubits: 4, J0: J0, alpha: 0.0, h: 0.0)
        let zzTerms = hamiltonian.terms.filter { $0.pauliString.operators.count == 2 }
        for term in zzTerms {
            #expect(abs(term.coefficient - -J0) < 1e-10, "Alpha=0 should give uniform coupling -J0")
        }
    }

    @Test("Large alpha suppresses long-range interactions")
    func longRangeLargeAlpha() {
        let J0 = 1.0
        let alpha = 10.0
        let hamiltonian = IsingHamiltonian.longRange(qubits: 5, J0: J0, alpha: alpha, h: 0.0)
        let zzTerms = hamiltonian.terms.filter { $0.pauliString.operators.count == 2 }
        let nearestNeighborTerms = zzTerms.filter { term in
            let qubits = term.pauliString.operators.map(\.qubit).sorted()
            return qubits[1] - qubits[0] == 1
        }
        let longRangeTerms = zzTerms.filter { term in
            let qubits = term.pauliString.operators.map(\.qubit).sorted()
            return qubits[1] - qubits[0] > 1
        }
        let maxNNCoeff = nearestNeighborTerms.map { abs($0.coefficient) }.max()!
        let maxLRCoeff = longRangeTerms.map { abs($0.coefficient) }.max()!
        #expect(maxLRCoeff < maxNNCoeff / 100, "Long-range couplings should be suppressed by factor > 100 for alpha=10")
    }

    @Test("Coupling between qubits i and j with j-i=2 follows power law")
    func longRangeDistanceTwo() {
        let J0 = 1.0
        let alpha = 2.0
        let hamiltonian = IsingHamiltonian.longRange(qubits: 4, J0: J0, alpha: alpha, h: 0.0)
        let zzTerms = hamiltonian.terms.filter { $0.pauliString.operators.count == 2 }
        let distanceTwoTerms = zzTerms.filter { term in
            let qubits = term.pauliString.operators.map(\.qubit).sorted()
            return qubits[1] - qubits[0] == 2
        }
        let expectedCoeff = -J0 / pow(2.0, alpha)
        for term in distanceTwoTerms {
            #expect(abs(term.coefficient - expectedCoeff) < 1e-10, "Distance-2 coupling should be \(expectedCoeff)")
        }
    }
}

/// Test suite for random-field Ising Hamiltonian with site-dependent disorder.
/// Validates per-site transverse field values, ZZ coupling uniformity,
/// and boundary condition handling.
@Suite("Random-Field Ising Hamiltonian")
struct IsingRandomFieldTests {
    @Test("Random field Hamiltonian produces correct term count")
    func randomFieldTermCount() {
        let hValues = [0.3, 0.7, 0.5, 0.9]
        let hamiltonian = IsingHamiltonian.randomField(qubits: 4, J: 1.0, hValues: hValues, periodic: false)
        let expectedZZTerms = 3
        let expectedXTerms = 4
        #expect(hamiltonian.terms.count == expectedZZTerms + expectedXTerms, "4-qubit random field chain should have 7 terms")
    }

    @Test("Per-site field values are correctly assigned")
    func randomFieldSiteValues() {
        let hValues = [0.1, 0.2, 0.3, 0.4]
        let hamiltonian = IsingHamiltonian.randomField(qubits: 4, J: 1.0, hValues: hValues, periodic: false)
        let xTerms = hamiltonian.terms.filter { $0.pauliString.operators.count == 1 }
        for term in xTerms {
            let qubit = term.pauliString.operators[0].qubit
            let expectedCoeff = -hValues[qubit]
            #expect(abs(term.coefficient - expectedCoeff) < 1e-10, "X term on qubit \(qubit) should have coefficient \(expectedCoeff)")
        }
    }

    @Test("ZZ couplings remain uniform despite random fields")
    func randomFieldUniformCoupling() {
        let J = 1.5
        let hValues = [0.1, 0.9, 0.3, 0.7]
        let hamiltonian = IsingHamiltonian.randomField(qubits: 4, J: J, hValues: hValues, periodic: false)
        let zzTerms = hamiltonian.terms.filter { $0.pauliString.operators.count == 2 }
        for term in zzTerms {
            #expect(abs(term.coefficient - -J) < 1e-10, "ZZ coupling should be uniform -J = \(-J)")
        }
    }

    @Test("Periodic random field chain includes boundary ZZ term")
    func randomFieldPeriodicBoundary() {
        let hValues = [0.5, 0.5, 0.5]
        let hamiltonian = IsingHamiltonian.randomField(qubits: 3, J: 1.0, hValues: hValues, periodic: true)
        let zzTerms = hamiltonian.terms.filter { $0.pauliString.operators.count == 2 }
        let hasBoundaryTerm = zzTerms.contains { term in
            let qubits = Set(term.pauliString.operators.map(\.qubit))
            return qubits.contains(0) && qubits.contains(2)
        }
        #expect(hasBoundaryTerm, "Periodic random field chain should include Z_2Z_0 term")
    }

    @Test("Zero disorder reproduces uniform transverse field Ising")
    func randomFieldZeroDisorder() {
        let h = 0.5
        let hValues = [h, h, h, h]
        let randomField = IsingHamiltonian.randomField(qubits: 4, J: 1.0, hValues: hValues, periodic: false)
        let uniform = IsingHamiltonian.chain(qubits: 4, J: 1.0, h: h, periodic: false)
        #expect(randomField.terms.count == uniform.terms.count, "Zero disorder should match uniform chain term count")
    }
}

/// Test suite for Hermitian symmetry and physical consistency.
/// Validates that constructed Hamiltonians satisfy Hermiticity requirements
/// and produce real expectation values on quantum states.
@Suite("Hamiltonian Hermiticity and Symmetry")
struct IsingHermitianTests {
    @Test("Chain Hamiltonian produces real expectation value on computational basis state")
    func chainRealExpectationComputational() {
        let chain = IsingHamiltonian.chain(qubits: 3, J: 1.0, h: 0.5, periodic: false)
        let state = QuantumState(qubits: 3)
        let expectation = chain.expectationValue(of: state)
        #expect(expectation.isFinite, "Expectation value should be finite")
    }

    @Test("Lattice Hamiltonian produces real expectation value on superposition state")
    func latticeRealExpectationSuperposition() {
        let lattice = IsingHamiltonian.lattice(rows: 2, cols: 2, J: 1.0, h: 0.5, periodic: false)
        let sqrt2inv = 1.0 / sqrt(2.0)
        var amplitudes = [Complex<Double>](repeating: .zero, count: 16)
        amplitudes[0] = Complex(sqrt2inv, 0.0)
        amplitudes[15] = Complex(sqrt2inv, 0.0)
        let state = QuantumState(qubits: 4, amplitudes: amplitudes)
        let expectation = lattice.expectationValue(of: state)
        #expect(expectation.isFinite, "Expectation value should be finite")
    }

    @Test("All ZZ terms have Z basis operators on both qubits")
    func allZZTermsUseZBasis() {
        let chain = IsingHamiltonian.chain(qubits: 4, J: 1.0, h: 0.5, periodic: true)
        let zzTerms = chain.terms.filter { $0.pauliString.operators.count == 2 }
        for term in zzTerms {
            let allZ = term.pauliString.operators.allSatisfy { $0.basis == .z }
            #expect(allZ, "ZZ term should have Z basis on all operators")
        }
    }

    @Test("All X terms have X basis operator")
    func allXTermsUseXBasis() {
        let chain = IsingHamiltonian.chain(qubits: 4, J: 1.0, h: 0.5, periodic: true)
        let xTerms = chain.terms.filter { $0.pauliString.operators.count == 1 }
        for term in xTerms {
            #expect(term.pauliString.operators[0].basis == .x, "X term should have X basis operator")
        }
    }

    @Test("Hamiltonian coefficients are all real (implicit in PauliTerms)")
    func coefficientsAreReal() {
        let longRange = IsingHamiltonian.longRange(qubits: 4, J0: 1.0, alpha: 1.5, h: 0.5)
        for term in longRange.terms {
            #expect(term.coefficient.isFinite, "All coefficients should be finite real numbers")
        }
    }

    @Test("Ground state energy is bounded for ferromagnetic chain")
    func groundStateEnergyBounded() {
        let n = 3
        let J = 1.0
        let h = 0.0
        let chain = IsingHamiltonian.chain(qubits: n, J: J, h: h, periodic: false)
        var amplitudes = [Complex<Double>](repeating: .zero, count: 1 << n)
        amplitudes[0] = .one
        let allZeroState = QuantumState(qubits: n, amplitudes: amplitudes)
        let energy = chain.expectationValue(of: allZeroState)
        let expectedEnergy = -J * Double(n - 1)
        #expect(abs(energy - expectedEnergy) < 1e-10, "All-zero state energy should be \(expectedEnergy)")
    }
}

/// Test suite for edge cases and boundary behavior.
/// Validates minimum system sizes, parameter ranges, and special
/// limiting cases of the Ising model constructors.
@Suite("Ising Hamiltonian Edge Cases")
struct IsingEdgeCasesTests {
    @Test("Chain with 2 qubits periodic matches open for ZZ count")
    func twoQubitPeriodicVsOpen() {
        let open = IsingHamiltonian.chain(qubits: 2, J: 1.0, h: 0.5, periodic: false)
        let periodic = IsingHamiltonian.chain(qubits: 2, J: 1.0, h: 0.5, periodic: true)
        let openZZ = open.terms.count(where: { $0.pauliString.operators.count == 2 })
        let periodicZZ = periodic.terms.count(where: { $0.pauliString.operators.count == 2 })
        #expect(openZZ == 1, "2-qubit open chain should have 1 ZZ term")
        #expect(periodicZZ == 1, "2-qubit periodic chain should have 1 ZZ term (no duplicate)")
    }

    @Test("Lattice 1xN matches 1D chain structure")
    func lattice1xNMatchesChain() {
        let n = 4
        let lattice = IsingHamiltonian.lattice(rows: 1, cols: n, J: 1.0, h: 0.5, periodic: false)
        let chain = IsingHamiltonian.chain(qubits: n, J: 1.0, h: 0.5, periodic: false)
        #expect(lattice.terms.count == chain.terms.count, "1xN lattice should have same term count as N-qubit chain")
    }

    @Test("Zero coupling J=0 produces only X terms")
    func zeroCouplingOnlyXTerms() {
        let chain = IsingHamiltonian.chain(qubits: 4, J: 0.0, h: 0.5, periodic: true)
        let zzTerms = chain.terms.filter { term in
            term.pauliString.operators.count == 2 && abs(term.coefficient) > 1e-15
        }
        #expect(zzTerms.isEmpty, "J=0 should produce no significant ZZ terms")
    }

    @Test("Zero field h=0 produces only ZZ terms")
    func zeroFieldOnlyZZTerms() {
        let chain = IsingHamiltonian.chain(qubits: 4, J: 1.0, h: 0.0, periodic: false)
        let xTerms = chain.terms.filter { term in
            term.pauliString.operators.count == 1 && abs(term.coefficient) > 1e-15
        }
        #expect(xTerms.isEmpty, "h=0 should produce no significant X terms")
    }

    @Test("Negative coupling is antiferromagnetic")
    func negativeCouplingAntiferromagnetic() {
        let J = -1.0
        let chain = IsingHamiltonian.chain(qubits: 3, J: J, h: 0.0, periodic: false)
        let zzTerms = chain.terms.filter { $0.pauliString.operators.count == 2 }
        for term in zzTerms {
            #expect(term.coefficient > 0, "Antiferromagnetic coupling should give positive ZZ coefficients")
        }
    }

    @Test("Large qubit count produces expected scaling")
    func largeQubitScaling() {
        let n = 10
        let chain = IsingHamiltonian.chain(qubits: n, J: 1.0, h: 0.5, periodic: false)
        let expectedTerms = (n - 1) + n
        #expect(chain.terms.count == expectedTerms, "\(n)-qubit chain should have \(expectedTerms) terms")
    }

    @Test("Long-range with single qubit produces only X term")
    func longRangeSingleQubit() {
        let hamiltonian = IsingHamiltonian.longRange(qubits: 1, J0: 1.0, alpha: 1.0, h: 0.5)
        #expect(hamiltonian.terms.count == 1, "Single qubit long-range should have 1 X term")
        #expect(hamiltonian.terms[0].pauliString.operators[0].basis == .x, "Single term should be X operator")
    }

    @Test("Observable description is non-empty for constructed Hamiltonians")
    func observableDescriptionNonEmpty() {
        let chain = IsingHamiltonian.chain(qubits: 2, J: 1.0, h: 0.5, periodic: false)
        #expect(!chain.description.isEmpty, "Observable description should be non-empty")
        #expect(chain.description.contains("Observable"), "Description should contain 'Observable'")
    }
}
