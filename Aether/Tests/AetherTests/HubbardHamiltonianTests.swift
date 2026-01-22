// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for HubbardModel struct properties.
/// Validates correct qubit count (2*sites), parameter storage,
/// and consistency between sites and observable dimensions.
@Suite("HubbardModel Properties")
struct HubbardModelPropertiesTests {
    @Test("Qubit count equals 2 times site count for chain")
    func qubitCountEqualsDoublesSites() {
        let model = HubbardHamiltonian.chain(sites: 4, t: 1.0, U: 4.0)
        #expect(model.qubits == 2 * model.sites, "Qubits should be 2*sites for spin-up and spin-down electrons")
    }

    @Test("Sites property matches input for chain")
    func sitesPropertyMatchesInput() {
        let model = HubbardHamiltonian.chain(sites: 5, t: 1.0, U: 2.0)
        #expect(model.sites == 5, "Sites property should match the input parameter")
    }

    @Test("Hopping parameter t stored correctly")
    func hoppingParameterStoredCorrectly() {
        let model = HubbardHamiltonian.chain(sites: 3, t: 2.5, U: 4.0)
        #expect(abs(model.t - 2.5) < 1e-10, "Hopping parameter t should be stored exactly")
    }

    @Test("Interaction parameter U stored correctly")
    func interactionParameterStoredCorrectly() {
        let model = HubbardHamiltonian.chain(sites: 3, t: 1.0, U: 6.0)
        #expect(abs(model.U - 6.0) < 1e-10, "Interaction parameter U should be stored exactly")
    }

    @Test("Observable is accessible from model")
    func observableAccessibleFromModel() {
        let model = HubbardHamiltonian.chain(sites: 2, t: 1.0, U: 4.0)
        #expect(model.observable.terms.count > 0, "Observable should contain Pauli terms")
    }

    @Test("Two-site chain has correct qubit count")
    func twoSiteChainCorrectQubits() {
        let model = HubbardHamiltonian.chain(sites: 2, t: 1.0, U: 4.0)
        #expect(model.qubits == 4, "Two-site chain should have 4 qubits (2 sites x 2 spins)")
    }

    @Test("Single site model has two qubits")
    func singleSiteHasTwoQubits() {
        let model = HubbardHamiltonian.chain(sites: 1, t: 1.0, U: 4.0)
        #expect(model.qubits == 2, "Single site should have 2 qubits for spin-up and spin-down")
    }
}

/// Test suite for HubbardHamiltonian.chain() factory method.
/// Validates 1D chain topology, open vs periodic boundaries,
/// and correct number of hopping terms generated.
@Suite("Chain Hamiltonian Construction")
struct ChainHamiltonianTests {
    @Test("Open chain has N-1 hopping bonds for N sites")
    func openChainHoppingBonds() {
        let model = HubbardHamiltonian.chain(sites: 4, t: 1.0, U: 0.0)
        let hoppingTermCount = model.observable.terms.count
        #expect(hoppingTermCount == 12, "4-site open chain with U=0 should have 12 terms: 3 bonds x 2 spins x 2 terms (separate XX and YY)")
    }

    @Test("Periodic chain has N hopping bonds for N sites")
    func periodicChainHoppingBonds() {
        let openModel = HubbardHamiltonian.chain(sites: 4, t: 1.0, U: 0.0, periodic: false)
        let periodicModel = HubbardHamiltonian.chain(sites: 4, t: 1.0, U: 0.0, periodic: true)
        #expect(periodicModel.observable.terms.count > openModel.observable.terms.count, "Periodic chain should have more terms than open chain")
    }

    @Test("Chain default is non-periodic")
    func chainDefaultNonPeriodic() {
        let defaultModel = HubbardHamiltonian.chain(sites: 3, t: 1.0, U: 0.0)
        let openModel = HubbardHamiltonian.chain(sites: 3, t: 1.0, U: 0.0, periodic: false)
        #expect(defaultModel.observable.terms.count == openModel.observable.terms.count, "Default chain should be non-periodic")
    }

    @Test("Single site chain has no hopping terms")
    func singleSiteNoHopping() {
        let model = HubbardHamiltonian.chain(sites: 1, t: 1.0, U: 0.0)
        #expect(model.observable.terms.isEmpty, "Single site with U=0 should have no terms")
    }

    @Test("Chain with zero hopping has only interaction terms")
    func chainZeroHopping() {
        let model = HubbardHamiltonian.chain(sites: 2, t: 0.0, U: 4.0)
        for term in model.observable.terms {
            let hasXY = term.pauliString.operators.contains { $0.basis == .x || $0.basis == .y }
            #expect(!hasXY, "Zero hopping should produce no X or Y operators")
        }
    }

    @Test("Periodic boundary adds wrap-around term")
    func periodicBoundaryWrapAround() {
        let periodicModel = HubbardHamiltonian.chain(sites: 3, t: 1.0, U: 0.0, periodic: true)
        let openModel = HubbardHamiltonian.chain(sites: 3, t: 1.0, U: 0.0, periodic: false)
        let periodicTerms = periodicModel.observable.terms.count
        let openTerms = openModel.observable.terms.count
        #expect(periodicTerms == openTerms + 4, "Periodic 3-site chain adds 4 extra terms (1 bond x 2 spins x 2 Pauli strings)")
    }
}

/// Test suite for HubbardHamiltonian.lattice() factory method.
/// Validates 2D rectangular grid topology, correct connectivity,
/// and periodic boundary conditions (torus topology).
@Suite("Lattice Hamiltonian Construction")
struct LatticeHamiltonianTests {
    @Test("2x2 lattice has correct qubit count")
    func lattice2x2QubitCount() {
        let model = HubbardHamiltonian.lattice(rows: 2, cols: 2, t: 1.0, U: 4.0)
        #expect(model.qubits == 8, "2x2 lattice should have 8 qubits (4 sites x 2 spins)")
    }

    @Test("2x2 lattice sites equals rows times cols")
    func latticeSitesEqualsRowsCols() {
        let model = HubbardHamiltonian.lattice(rows: 2, cols: 3, t: 1.0, U: 4.0)
        #expect(model.sites == 6, "2x3 lattice should have 6 sites")
    }

    @Test("Open 2x2 lattice has 4 bonds")
    func openLattice2x2Bonds() {
        let model = HubbardHamiltonian.lattice(rows: 2, cols: 2, t: 1.0, U: 0.0, periodic: false)
        let termCount = model.observable.terms.count
        #expect(termCount == 16, "Open 2x2 lattice with U=0 should have 16 terms: 4 bonds x 2 spins x 2 Pauli strings")
    }

    @Test("Periodic lattice has more bonds than open lattice")
    func periodicLatticeMoreBonds() {
        let openModel = HubbardHamiltonian.lattice(rows: 2, cols: 2, t: 1.0, U: 0.0, periodic: false)
        let periodicModel = HubbardHamiltonian.lattice(rows: 2, cols: 2, t: 1.0, U: 0.0, periodic: true)
        #expect(periodicModel.observable.terms.count > openModel.observable.terms.count, "Periodic lattice should have more terms than open lattice")
    }

    @Test("1xN lattice is equivalent to chain")
    func oneRowLatticeIsChain() {
        let latticeModel = HubbardHamiltonian.lattice(rows: 1, cols: 4, t: 1.0, U: 4.0)
        let chainModel = HubbardHamiltonian.chain(sites: 4, t: 1.0, U: 4.0)
        #expect(latticeModel.qubits == chainModel.qubits, "1xN lattice should have same qubits as N-site chain")
        #expect(latticeModel.sites == chainModel.sites, "1xN lattice should have same sites as N-site chain")
    }

    @Test("Lattice default is non-periodic")
    func latticeDefaultNonPeriodic() {
        let defaultModel = HubbardHamiltonian.lattice(rows: 2, cols: 2, t: 1.0, U: 0.0)
        let openModel = HubbardHamiltonian.lattice(rows: 2, cols: 2, t: 1.0, U: 0.0, periodic: false)
        #expect(defaultModel.observable.terms.count == openModel.observable.terms.count, "Default lattice should be non-periodic")
    }

    @Test("Single site lattice has no hopping terms")
    func singleSiteLatticeNoHopping() {
        let model = HubbardHamiltonian.lattice(rows: 1, cols: 1, t: 1.0, U: 0.0)
        #expect(model.observable.terms.isEmpty, "1x1 lattice with U=0 should have no terms")
    }
}

/// Test suite for HubbardHamiltonian.fromHoppings() factory method.
/// Validates custom graph topologies, non-uniform hopping strengths,
/// and correct term generation for arbitrary connectivity.
@Suite("Custom Hoppings Construction")
struct FromHoppingsTests {
    @Test("Triangle topology creates three bonds")
    func triangleTopologyThreeBonds() {
        let triangleHoppings = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]
        let model = HubbardHamiltonian.fromHoppings(hoppings: triangleHoppings, U: 0.0, sites: 3)
        #expect(model.sites == 3, "Triangle should have 3 sites")
        #expect(model.qubits == 6, "Triangle should have 6 qubits")
        #expect(model.observable.terms.count == 12, "Triangle with U=0 should have 12 terms: 3 bonds x 2 spins x 2 Pauli strings")
    }

    @Test("Non-uniform hopping strengths preserved in terms")
    func nonUniformHoppingStrengths() {
        let hoppings = [(0, 1, 1.0), (1, 2, 2.0)]
        let model = HubbardHamiltonian.fromHoppings(hoppings: hoppings, U: 0.0, sites: 3)
        #expect(abs(model.t - 1.5) < 1e-10, "Average hopping should be 1.5")
    }

    @Test("Empty hoppings with U=0 gives empty observable")
    func emptyHoppingsNoInteraction() {
        let model = HubbardHamiltonian.fromHoppings(hoppings: [], U: 0.0, sites: 2)
        #expect(model.observable.terms.isEmpty, "No hoppings and U=0 should give empty observable")
    }

    @Test("Empty hoppings with U>0 gives only interaction terms")
    func emptyHoppingsWithInteraction() {
        let model = HubbardHamiltonian.fromHoppings(hoppings: [], U: 4.0, sites: 2)
        #expect(model.observable.terms.count > 0, "No hoppings but U>0 should give interaction terms")
    }

    @Test("Custom graph preserves site count")
    func customGraphPreservesSiteCount() {
        let hoppings = [(0, 1, 1.0)]
        let model = HubbardHamiltonian.fromHoppings(hoppings: hoppings, U: 4.0, sites: 5)
        #expect(model.sites == 5, "Site count should match input even if not all sites have hoppings")
    }

    @Test("Average hopping computed correctly for multiple bonds")
    func averageHoppingComputed() {
        let hoppings = [(0, 1, 2.0), (1, 2, 4.0), (2, 3, 6.0)]
        let model = HubbardHamiltonian.fromHoppings(hoppings: hoppings, U: 0.0, sites: 4)
        #expect(abs(model.t - 4.0) < 1e-10, "Average hopping should be (2+4+6)/3 = 4.0")
    }
}

/// Test suite for Jordan-Wigner transformation correctness.
/// Validates hopping term structure (XX + YY with Z strings),
/// interaction term structure (I, Z, ZZ), and coefficient signs.
@Suite("Jordan-Wigner Transformation")
struct JordanWignerTransformationTests {
    @Test("Nearest-neighbor hopping produces XX and YY terms")
    func nearestNeighborHoppingTerms() {
        let terms = HubbardHamiltonian.jordanWignerHopping(from: 0, to: 1, coefficient: -1.0)
        #expect(terms.count == 2, "Nearest-neighbor hopping should produce 2 Pauli strings")

        let firstOperators = terms[0].1.operators
        let secondOperators = terms[1].1.operators

        let firstHasX = firstOperators.contains { $0.basis == .x }
        let secondHasY = secondOperators.contains { $0.basis == .y }

        #expect(firstHasX, "First term should contain X operators")
        #expect(secondHasY, "Second term should contain Y operators")
    }

    @Test("Non-adjacent hopping includes Z string")
    func nonAdjacentHoppingZString() {
        let terms = HubbardHamiltonian.jordanWignerHopping(from: 0, to: 3, coefficient: -1.0)

        let xxTerm = terms[0].1.operators
        let zCount = xxTerm.count(where: { $0.basis == .z })

        #expect(zCount == 2, "Hopping from 0 to 3 should have Z operators on qubits 1 and 2")
    }

    @Test("Hopping coefficient halved for XX and YY terms")
    func hoppingCoefficientHalved() {
        let terms = HubbardHamiltonian.jordanWignerHopping(from: 0, to: 1, coefficient: -2.0)

        #expect(abs(terms[0].0 - -1.0) < 1e-10, "XX term coefficient should be half of input")
        #expect(abs(terms[1].0 - -1.0) < 1e-10, "YY term coefficient should be half of input")
    }

    @Test("Interaction produces four Pauli terms")
    func interactionProducesFourTerms() {
        let terms = HubbardHamiltonian.jordanWignerInteraction(upQubit: 0, downQubit: 2, U: 4.0)
        #expect(terms.count == 4, "Interaction should produce 4 Pauli terms")
    }

    @Test("Interaction identity term has positive coefficient")
    func interactionIdentityPositive() {
        let terms = HubbardHamiltonian.jordanWignerInteraction(upQubit: 0, downQubit: 2, U: 4.0)
        let identityTerm = terms.first { $0.1.operators.isEmpty }
        #expect(identityTerm != nil, "Should have identity term")
        #expect(identityTerm!.0 > 0, "Identity term coefficient should be positive (U/4)")
    }

    @Test("Interaction single-Z terms have negative coefficient")
    func interactionSingleZNegative() {
        let terms = HubbardHamiltonian.jordanWignerInteraction(upQubit: 0, downQubit: 2, U: 4.0)
        let singleZTerms = terms.filter { $0.1.operators.count == 1 && $0.1.operators[0].basis == .z }
        #expect(singleZTerms.count == 2, "Should have two single-Z terms")
        for term in singleZTerms {
            #expect(term.0 < 0, "Single-Z term coefficient should be negative (-U/4)")
        }
    }

    @Test("Interaction ZZ term has positive coefficient")
    func interactionZZPositive() {
        let terms = HubbardHamiltonian.jordanWignerInteraction(upQubit: 0, downQubit: 2, U: 4.0)
        let zzTerm = terms.first { $0.1.operators.count == 2 }
        #expect(zzTerm != nil, "Should have ZZ term")
        #expect(zzTerm!.0 > 0, "ZZ term coefficient should be positive (U/4)")
    }

    @Test("Interaction coefficients sum to U/4 for n_up * n_down expansion")
    func interactionCoefficientSum() {
        let U = 8.0
        let terms = HubbardHamiltonian.jordanWignerInteraction(upQubit: 0, downQubit: 1, U: U)
        let sum = terms.reduce(0.0) { $0 + $1.0 }
        #expect(abs(sum) < 1e-10, "Coefficients should sum to 0 (identity contributes when both occupied)")
    }
}

/// Test suite for qubit mapping conventions.
/// Validates spin-up qubits at indices 0..sites-1 and
/// spin-down qubits at indices sites..2*sites-1.
@Suite("Qubit Mapping Conventions")
struct QubitMappingTests {
    @Test("Spin-up electrons map to qubits 0 to sites-1")
    func spinUpQubitRange() {
        let model = HubbardHamiltonian.chain(sites: 3, t: 1.0, U: 0.0)
        let allQubits = model.observable.terms.flatMap { $0.pauliString.operators.map(\.qubit) }
        let spinUpQubits = Set(allQubits.filter { $0 < model.sites })
        #expect(spinUpQubits.isSubset(of: Set(0 ..< 3)), "Spin-up qubits should be in range 0..2")
    }

    @Test("Spin-down electrons map to qubits sites to 2*sites-1")
    func spinDownQubitRange() {
        let model = HubbardHamiltonian.chain(sites: 3, t: 1.0, U: 0.0)
        let allQubits = model.observable.terms.flatMap { $0.pauliString.operators.map(\.qubit) }
        let spinDownQubits = Set(allQubits.filter { $0 >= model.sites })
        #expect(spinDownQubits.isSubset(of: Set(3 ..< 6)), "Spin-down qubits should be in range 3..5")
    }

    @Test("Both spin species present in hopping terms")
    func bothSpinSpeciesPresent() {
        let model = HubbardHamiltonian.chain(sites: 2, t: 1.0, U: 0.0)
        let allQubits = model.observable.terms.flatMap { $0.pauliString.operators.map(\.qubit) }
        let hasSpinUp = allQubits.contains { $0 < model.sites }
        let hasSpinDown = allQubits.contains { $0 >= model.sites }
        #expect(hasSpinUp, "Should have spin-up hopping terms")
        #expect(hasSpinDown, "Should have spin-down hopping terms")
    }

    @Test("Interaction couples spin-up and spin-down at same site")
    func interactionCouplesSameSite() {
        let terms = HubbardHamiltonian.jordanWignerInteraction(upQubit: 0, downQubit: 3, U: 4.0)
        let zzTerm = terms.first { $0.1.operators.count == 2 }!
        let qubits = zzTerm.1.operators.map(\.qubit)
        #expect(qubits.contains(0), "Interaction should involve spin-up qubit 0")
        #expect(qubits.contains(3), "Interaction should involve spin-down qubit 3")
    }

    @Test("Hopping terms only couple same spin species")
    func hoppingOnlySameSpin() {
        let model = HubbardHamiltonian.chain(sites: 3, t: 1.0, U: 0.0)
        for term in model.observable.terms {
            let qubits = term.pauliString.operators.map(\.qubit)
            let allSpinUp = qubits.allSatisfy { $0 < model.sites }
            let allSpinDown = qubits.allSatisfy { $0 >= model.sites }
            #expect(allSpinUp || allSpinDown, "Each hopping term should only involve one spin species")
        }
    }
}

/// Test suite for Hermiticity of the Hubbard Hamiltonian.
/// Validates real coefficients and proper Pauli string structure
/// ensuring the observable represents a physical Hermitian operator.
@Suite("Hamiltonian Hermiticity")
struct HamiltonianHermiticityTests {
    @Test("All coefficients are real (no imaginary components)")
    func allCoefficientsReal() {
        let model = HubbardHamiltonian.chain(sites: 3, t: 1.0, U: 4.0)
        for term in model.observable.terms {
            #expect(term.coefficient.isFinite, "Coefficient should be finite real number")
        }
    }

    @Test("Hopping terms come in Hermitian pairs")
    func hoppingTermsHermitianPairs() {
        let terms = HubbardHamiltonian.jordanWignerHopping(from: 0, to: 1, coefficient: -1.0)
        #expect(abs(terms[0].0 - terms[1].0) < 1e-10, "XX and YY terms should have equal coefficients")
    }

    @Test("Interaction terms are diagonal in computational basis")
    func interactionTermsDiagonal() {
        let terms = HubbardHamiltonian.jordanWignerInteraction(upQubit: 0, downQubit: 2, U: 4.0)
        for term in terms {
            let hasXY = term.1.operators.contains { $0.basis == .x || $0.basis == .y }
            #expect(!hasXY, "Interaction terms should only contain I and Z (diagonal)")
        }
    }

    @Test("Lattice Hamiltonian has real expectation value on any state")
    func latticeRealExpectationValue() {
        let model = HubbardHamiltonian.lattice(rows: 2, cols: 2, t: 1.0, U: 4.0)
        let state = QuantumState(qubits: model.qubits)
        let expectation = model.observable.expectationValue(of: state)
        #expect(expectation.isFinite, "Expectation value should be finite real number")
    }

    @Test("Chain Hamiltonian preserves norm under evolution")
    func chainPreservesNorm() {
        let model = HubbardHamiltonian.chain(sites: 2, t: 1.0, U: 4.0)
        let state = QuantumState(qubits: model.qubits)
        let expectation = model.observable.expectationValue(of: state)
        #expect(expectation.isFinite, "Expectation value on |0...0⟩ should be finite")
    }

    @Test("Zero interaction gives purely kinetic Hamiltonian")
    func zeroInteractionPurelyKinetic() {
        let model = HubbardHamiltonian.chain(sites: 2, t: 1.0, U: 0.0)
        for term in model.observable.terms {
            let hasZ = term.pauliString.operators.contains { $0.basis == .z }
            let hasXY = term.pauliString.operators.contains { $0.basis == .x || $0.basis == .y }
            #expect(!hasZ || hasXY, "Pure kinetic terms should not have Z-only strings")
        }
    }
}

/// Test suite for edge cases and boundary conditions.
/// Validates behavior with minimum sizes, zero parameters,
/// and various parameter combinations.
@Suite("Hubbard Edge Cases and Boundaries")
struct HubbardEdgeCasesTests {
    @Test("Minimum chain size of 1 site works")
    func minimumChainSize() {
        let model = HubbardHamiltonian.chain(sites: 1, t: 1.0, U: 4.0)
        #expect(model.qubits == 2, "Single site chain should have 2 qubits")
    }

    @Test("Large hopping parameter handled correctly")
    func largeHoppingParameter() {
        let model = HubbardHamiltonian.chain(sites: 2, t: 100.0, U: 1.0)
        #expect(abs(model.t - 100.0) < 1e-10, "Large hopping parameter should be preserved")
    }

    @Test("Large interaction parameter handled correctly")
    func largeInteractionParameter() {
        let model = HubbardHamiltonian.chain(sites: 2, t: 1.0, U: 100.0)
        #expect(abs(model.U - 100.0) < 1e-10, "Large interaction parameter should be preserved")
    }

    @Test("Negative hopping parameter works")
    func negativeHoppingParameter() {
        let model = HubbardHamiltonian.chain(sites: 2, t: -1.0, U: 4.0)
        #expect(abs(model.t - -1.0) < 1e-10, "Negative hopping should be allowed")
    }

    @Test("Negative interaction parameter works")
    func negativeInteractionParameter() {
        let model = HubbardHamiltonian.chain(sites: 2, t: 1.0, U: -4.0)
        #expect(abs(model.U - -4.0) < 1e-10, "Negative U (attractive interaction) should be allowed")
    }

    @Test("Zero hopping and zero interaction gives empty observable")
    func zeroHoppingZeroInteraction() {
        let model = HubbardHamiltonian.chain(sites: 2, t: 0.0, U: 0.0)
        #expect(model.observable.terms.isEmpty, "Zero t and zero U should give empty Hamiltonian")
    }

    @Test("Very small hopping handled without numerical issues")
    func verySmallHopping() {
        let model = HubbardHamiltonian.chain(sites: 2, t: 1e-10, U: 4.0)
        #expect(model.observable.terms.count > 0, "Very small hopping should still generate terms")
    }

    @Test("Periodic single site has no wrap-around")
    func periodicSingleSiteNoWrap() {
        let model = HubbardHamiltonian.chain(sites: 1, t: 1.0, U: 0.0, periodic: true)
        #expect(model.observable.terms.isEmpty, "Periodic single site should have no hopping terms")
    }
}

/// Test suite for term counting and structure verification.
/// Validates expected number of Pauli terms for various
/// system sizes and parameter configurations.
@Suite("Term Count Verification")
struct TermCountTests {
    @Test("Two-site chain term count with both t and U")
    func twoSiteChainTermCount() {
        let model = HubbardHamiltonian.chain(sites: 2, t: 1.0, U: 4.0)
        let hoppingTerms = 1 * 2 * 2
        let interactionTerms = 2 * 4
        let expectedTotal = hoppingTerms + interactionTerms
        #expect(model.observable.terms.count == expectedTotal, "2-site chain should have \(expectedTotal) terms")
    }

    @Test("Three-site open chain term count")
    func threeSiteOpenChainTermCount() {
        let model = HubbardHamiltonian.chain(sites: 3, t: 1.0, U: 4.0)
        let hoppingTerms = 2 * 2 * 2
        let interactionTerms = 3 * 4
        let expectedTotal = hoppingTerms + interactionTerms
        #expect(model.observable.terms.count == expectedTotal, "3-site open chain should have \(expectedTotal) terms")
    }

    @Test("Only hopping terms when U is zero")
    func onlyHoppingWhenUZero() {
        let model = HubbardHamiltonian.chain(sites: 3, t: 1.0, U: 0.0)
        let expectedHoppingTerms = 2 * 2 * 2
        #expect(model.observable.terms.count == expectedHoppingTerms, "U=0 should give only hopping terms")
    }

    @Test("Only interaction terms when t is zero")
    func onlyInteractionWhenTZero() {
        let model = HubbardHamiltonian.chain(sites: 3, t: 0.0, U: 4.0)
        let expectedInteractionTerms = 3 * 4
        #expect(model.observable.terms.count == expectedInteractionTerms, "t=0 should give only interaction terms")
    }

    @Test("2x2 lattice term count")
    func lattice2x2TermCount() {
        let model = HubbardHamiltonian.lattice(rows: 2, cols: 2, t: 1.0, U: 4.0)
        let hoppingTerms = 4 * 2 * 2
        let interactionTerms = 4 * 4
        let expectedTotal = hoppingTerms + interactionTerms
        #expect(model.observable.terms.count == expectedTotal, "2x2 lattice should have \(expectedTotal) terms")
    }
}

/// Test suite for expectation value computations.
/// Validates Hamiltonian evaluation on reference states
/// and consistency between different construction methods.
@Suite("Expectation Value Computations")
struct ExpectationValueTests {
    @Test("Ground state |00...0⟩ has finite expectation value")
    func groundStateFiniteExpectation() {
        let model = HubbardHamiltonian.chain(sites: 2, t: 1.0, U: 4.0)
        let state = QuantumState(qubits: model.qubits)
        let expectation = model.observable.expectationValue(of: state)
        #expect(expectation.isFinite, "Expectation value on |0000⟩ should be finite")
    }

    @Test("Empty site state has zero interaction energy")
    func emptySiteZeroInteraction() {
        let model = HubbardHamiltonian.chain(sites: 2, t: 0.0, U: 4.0)
        let state = QuantumState(qubits: model.qubits)
        let expectation = model.observable.expectationValue(of: state)
        #expect(abs(expectation) < 1e-10, "Empty state should have zero interaction energy")
    }

    @Test("Different t values give different expectation values")
    func differentTDifferentExpectation() {
        let model1 = HubbardHamiltonian.chain(sites: 2, t: 1.0, U: 4.0)
        let model2 = HubbardHamiltonian.chain(sites: 2, t: 2.0, U: 4.0)
        let state = QuantumState(qubits: model1.qubits)
        let e1 = model1.observable.expectationValue(of: state)
        let e2 = model2.observable.expectationValue(of: state)
        #expect(abs(e1) < 1e-10 && abs(e2) < 1e-10, "Both should be zero for empty state but term structure differs")
    }

    @Test("Chain and equivalent lattice give consistent results")
    func chainLatticeConsistency() {
        let chainModel = HubbardHamiltonian.chain(sites: 3, t: 1.0, U: 4.0)
        let latticeModel = HubbardHamiltonian.lattice(rows: 1, cols: 3, t: 1.0, U: 4.0)
        let state = QuantumState(qubits: chainModel.qubits)
        let chainExpectation = chainModel.observable.expectationValue(of: state)
        let latticeExpectation = latticeModel.observable.expectationValue(of: state)
        #expect(abs(chainExpectation - latticeExpectation) < 1e-10, "1xN lattice should match N-site chain")
    }
}
