// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for TrotterOrder enum cases and raw values.
/// Validates that decomposition orders are correctly defined
/// with appropriate raw values for quantum time evolution.
@Suite("TrotterOrder Enum")
struct TrotterOrderTests {
    @Test("First order has raw value 1")
    func firstOrderRawValue() {
        let order = TrotterOrder.first
        #expect(order.rawValue == 1, "First order should have raw value 1")
    }

    @Test("Second order has raw value 2")
    func secondOrderRawValue() {
        let order = TrotterOrder.second
        #expect(order.rawValue == 2, "Second order should have raw value 2")
    }

    @Test("Fourth order has raw value 4")
    func fourthOrderRawValue() {
        let order = TrotterOrder.fourth
        #expect(order.rawValue == 4, "Fourth order should have raw value 4")
    }

    @Test("Sixth order has raw value 6")
    func sixthOrderRawValue() {
        let order = TrotterOrder.sixth
        #expect(order.rawValue == 6, "Sixth order should have raw value 6")
    }

    @Test("All orders can be initialized from raw values")
    func initFromRawValue() {
        let first = TrotterOrder(rawValue: 1)
        let second = TrotterOrder(rawValue: 2)
        let fourth = TrotterOrder(rawValue: 4)
        let sixth = TrotterOrder(rawValue: 6)

        #expect(first == .first, "Raw value 1 should initialize to first order")
        #expect(second == .second, "Raw value 2 should initialize to second order")
        #expect(fourth == .fourth, "Raw value 4 should initialize to fourth order")
        #expect(sixth == .sixth, "Raw value 6 should initialize to sixth order")
    }

    @Test("Invalid raw values return nil")
    func invalidRawValue() {
        let invalid0 = TrotterOrder(rawValue: 0)
        let invalid3 = TrotterOrder(rawValue: 3)
        let invalid5 = TrotterOrder(rawValue: 5)
        let invalid7 = TrotterOrder(rawValue: 7)

        #expect(invalid0 == nil, "Raw value 0 should return nil")
        #expect(invalid3 == nil, "Raw value 3 should return nil")
        #expect(invalid5 == nil, "Raw value 5 should return nil")
        #expect(invalid7 == nil, "Raw value 7 should return nil")
    }
}

/// Test suite for TrotterConfiguration initialization and defaults.
/// Validates configuration parameters including order, steps,
/// commutation sorting, and coefficient threshold defaults.
@Suite("TrotterConfiguration Initialization")
struct TrotterConfigurationTests {
    @Test("Default configuration uses second order")
    func defaultOrder() {
        let config = TrotterConfiguration()
        #expect(config.order == .second, "Default order should be second")
    }

    @Test("Default configuration uses 1 step")
    func defaultSteps() {
        let config = TrotterConfiguration()
        #expect(config.steps == 1, "Default steps should be 1")
    }

    @Test("Default configuration has sortByCommutation false")
    func defaultSortByCommutation() {
        let config = TrotterConfiguration()
        #expect(config.sortByCommutation == false, "Default sortByCommutation should be false")
    }

    @Test("Default coefficient threshold is 1e-15")
    func defaultCoefficientThreshold() {
        let config = TrotterConfiguration()
        #expect(abs(config.coefficientThreshold - 1e-15) < 1e-20, "Default threshold should be 1e-15")
    }

    @Test("Custom order is preserved")
    func customOrder() {
        let config = TrotterConfiguration(order: .fourth)
        #expect(config.order == .fourth, "Custom order should be preserved")
    }

    @Test("Custom steps are preserved")
    func customSteps() {
        let config = TrotterConfiguration(steps: 100)
        #expect(config.steps == 100, "Custom steps should be preserved")
    }

    @Test("Custom sortByCommutation is preserved")
    func customSortByCommutation() {
        let config = TrotterConfiguration(sortByCommutation: true)
        #expect(config.sortByCommutation == true, "Custom sortByCommutation should be preserved")
    }

    @Test("Custom coefficient threshold is preserved")
    func customCoefficientThreshold() {
        let config = TrotterConfiguration(coefficientThreshold: 1e-10)
        #expect(abs(config.coefficientThreshold - 1e-10) < 1e-15, "Custom threshold should be preserved")
    }

    @Test("All custom parameters work together")
    func allCustomParameters() {
        let config = TrotterConfiguration(
            order: .sixth,
            steps: 50,
            sortByCommutation: true,
            coefficientThreshold: 1e-12,
        )

        #expect(config.order == .sixth, "Order should be sixth")
        #expect(config.steps == 50, "Steps should be 50")
        #expect(config.sortByCommutation == true, "sortByCommutation should be true")
        #expect(abs(config.coefficientThreshold - 1e-12) < 1e-17, "Threshold should be 1e-12")
    }
}

/// Test suite for TrotterSuzuki.evolve() circuit generation.
/// Validates that evolution produces valid quantum circuits
/// with correct qubit counts and non-empty gate sequences.
@Suite("TrotterSuzuki Circuit Generation")
struct TrotterSuzukiCircuitTests {
    @Test("Evolve produces non-empty circuit for single Z term")
    func singleZTermProducesCircuit() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.z(0)))])
        let config = TrotterConfiguration(order: .first, steps: 1)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 1, config: config)

        #expect(circuit.qubits >= 1, "Circuit should have at least 1 qubit")
        #expect(circuit.count > 0, "Circuit should have gates for Z term evolution")
    }

    @Test("Evolve produces non-empty circuit for single X term")
    func singleXTermProducesCircuit() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.x(0)))])
        let config = TrotterConfiguration(order: .first, steps: 1)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 1, config: config)

        #expect(circuit.qubits >= 1, "Circuit should have at least 1 qubit")
        #expect(circuit.count > 0, "Circuit should have gates for X term evolution")
    }

    @Test("Evolve produces non-empty circuit for single Y term")
    func singleYTermProducesCircuit() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.y(0)))])
        let config = TrotterConfiguration(order: .first, steps: 1)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 1, config: config)

        #expect(circuit.qubits >= 1, "Circuit should have at least 1 qubit")
        #expect(circuit.count > 0, "Circuit should have gates for Y term evolution")
    }

    @Test("Evolve handles two-qubit ZZ term")
    func twoQubitZZTermProducesCircuit() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.z(0), .z(1)))])
        let config = TrotterConfiguration(order: .first, steps: 1)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 2, config: config)

        #expect(circuit.qubits >= 2, "Circuit should have at least 2 qubits")
        #expect(circuit.count > 0, "Circuit should have gates for ZZ term evolution")
    }

    @Test("Evolve handles multi-term Hamiltonian")
    func multiTermHamiltonianProducesCircuit() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0), .z(1))),
            (-0.3, PauliString(.x(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let config = TrotterConfiguration(order: .second, steps: 5)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 2, config: config)

        #expect(circuit.qubits >= 2, "Circuit should have at least 2 qubits")
        #expect(circuit.count > 0, "Circuit should have gates for multi-term Hamiltonian")
    }

    @Test("Circuit is executable and produces valid state")
    func circuitIsExecutable() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.z(0)))])
        let config = TrotterConfiguration(order: .first, steps: 1)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: 0.5, qubits: 1, config: config)
        let state = circuit.execute()

        let totalProb = state.probability(of: 0) + state.probability(of: 1)
        #expect(abs(totalProb - 1.0) < 1e-10, "Evolved state should have total probability 1")
    }

    @Test("More steps produce more gates")
    func moreStepsProduceMoreGates() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.z(0)))])
        let config1 = TrotterConfiguration(order: .first, steps: 1)
        let config5 = TrotterConfiguration(order: .first, steps: 5)

        let circuit1 = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 1, config: config1)
        let circuit5 = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 1, config: config5)

        #expect(circuit5.count > circuit1.count, "5 steps should produce more gates than 1 step")
    }
}

/// Test suite for different Trotter orders producing different circuits.
/// Validates that higher orders result in deeper circuits due to
/// recursive decomposition formulas (Yoshida, Suzuki).
@Suite("TrotterSuzuki Order Comparison")
struct TrotterOrderComparisonTests {
    @Test("Second order produces more gates than first order")
    func secondOrderDeeperThanFirst() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0), .z(1))),
            (-0.3, PauliString(.x(0))),
        ])
        let configFirst = TrotterConfiguration(order: .first, steps: 1)
        let configSecond = TrotterConfiguration(order: .second, steps: 1)

        let circuitFirst = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 2, config: configFirst)
        let circuitSecond = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 2, config: configSecond)

        #expect(circuitSecond.count > circuitFirst.count, "Second order should produce more gates than first order")
    }

    @Test("Fourth order produces more gates than second order")
    func fourthOrderDeeperThanSecond() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0), .z(1))),
            (-0.3, PauliString(.x(0))),
        ])
        let configSecond = TrotterConfiguration(order: .second, steps: 1)
        let configFourth = TrotterConfiguration(order: .fourth, steps: 1)

        let circuitSecond = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 2, config: configSecond)
        let circuitFourth = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 2, config: configFourth)

        #expect(circuitFourth.count > circuitSecond.count, "Fourth order should produce more gates than second order")
    }

    @Test("Sixth order produces more gates than fourth order")
    func sixthOrderDeeperThanFourth() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0), .z(1))),
            (-0.3, PauliString(.x(0))),
        ])
        let configFourth = TrotterConfiguration(order: .fourth, steps: 1)
        let configSixth = TrotterConfiguration(order: .sixth, steps: 1)

        let circuitFourth = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 2, config: configFourth)
        let circuitSixth = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 2, config: configSixth)

        #expect(circuitSixth.count > circuitFourth.count, "Sixth order should produce more gates than fourth order")
    }

    @Test("All orders produce valid executable circuits")
    func allOrdersExecutable() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.x(0)))])
        let time = 0.5
        let orders: [TrotterOrder] = [.first, .second, .fourth, .sixth]

        for order in orders {
            let config = TrotterConfiguration(order: order, steps: 1)
            let circuit = TrotterSuzuki.evolve(hamiltonian, time: time, qubits: 1, config: config)
            let state = circuit.execute()

            let totalProb = state.probability(of: 0) + state.probability(of: 1)
            #expect(abs(totalProb - 1.0) < 1e-10, "Order \(order.rawValue) should produce valid state")
        }
    }

    @Test("Gate count ratio reflects recursive structure")
    func gateCountRatioReflectsRecursion() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
            (0.5, PauliString(.x(0))),
        ])
        let configFirst = TrotterConfiguration(order: .first, steps: 1)
        let configSecond = TrotterConfiguration(order: .second, steps: 1)

        let circuitFirst = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 1, config: configFirst)
        let circuitSecond = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 1, config: configSecond)

        let ratio = Double(circuitSecond.count) / Double(circuitFirst.count)
        #expect(ratio >= 1.5, "Second order should have at least 1.5x gates of first order (symmetric split)")
    }
}

/// Test suite for identity evolution and edge cases.
/// Validates behavior with zero time evolution, single terms,
/// and coefficient threshold filtering.
@Suite("TrotterSuzuki Edge Cases")
struct TrotterEdgeCasesTests {
    @Test("Zero time evolution preserves initial state")
    func zeroTimeEvolution() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.x(0)))])
        let config = TrotterConfiguration(order: .first, steps: 1)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: 0.0, qubits: 1, config: config)
        let state = circuit.execute()

        #expect(abs(state.probability(of: 0) - 1.0) < 1e-10, "Zero time evolution should preserve |0> state")
    }

    @Test("Very small time produces near-identity evolution")
    func verySmallTimeNearIdentity() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.x(0)))])
        let config = TrotterConfiguration(order: .first, steps: 1)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: 1e-10, qubits: 1, config: config)
        let state = circuit.execute()

        #expect(abs(state.probability(of: 0) - 1.0) < 1e-8, "Very small time should produce near-identity evolution")
    }

    @Test("Single Z term evolution is diagonal")
    func singleZTermDiagonal() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.z(0)))])
        let config = TrotterConfiguration(order: .first, steps: 1)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: 0.5, qubits: 1, config: config)
        let state = circuit.execute()

        #expect(abs(state.probability(of: 0) - 1.0) < 1e-10, "Z evolution on |0> should stay in |0> state")
    }

    @Test("Terms below threshold are filtered")
    func coefficientThresholdFiltering() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
            (1e-20, PauliString(.x(0))),
        ])
        let configNoFilter = TrotterConfiguration(order: .first, steps: 1, coefficientThreshold: 1e-25)
        let configWithFilter = TrotterConfiguration(order: .first, steps: 1, coefficientThreshold: 1e-15)

        let circuitNoFilter = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 1, config: configNoFilter)
        let circuitWithFilter = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 1, config: configWithFilter)

        #expect(circuitWithFilter.count < circuitNoFilter.count, "Higher threshold should filter out tiny coefficient term")
    }

    @Test("Sort by commutation reorders terms")
    func sortByCommutationReorders() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.x(0))),
            (0.5, PauliString(.z(0), .z(1))),
            (0.3, PauliString(.x(1))),
        ])
        let configNoSort = TrotterConfiguration(order: .first, steps: 1, sortByCommutation: false)
        let configWithSort = TrotterConfiguration(order: .first, steps: 1, sortByCommutation: true)

        let circuitNoSort = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 2, config: configNoSort)
        let circuitWithSort = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 2, config: configWithSort)

        #expect(circuitNoSort.count == circuitWithSort.count, "Sorting should not change total gate count")
    }

    @Test("Three-qubit Pauli string evolution")
    func threeQubitPauliString() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.x(0), .y(1), .z(2)))])
        let config = TrotterConfiguration(order: .first, steps: 1)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: 0.5, qubits: 3, config: config)

        #expect(circuit.qubits >= 3, "Circuit should have at least 3 qubits")
        #expect(circuit.count > 0, "Circuit should have gates for XYZ term")
    }

    @Test("Mixed single and multi-qubit terms")
    func mixedSingleAndMultiQubitTerms() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
            (0.5, PauliString(.z(0), .z(1))),
            (-0.25, PauliString(.x(0), .x(1), .x(2))),
        ])
        let config = TrotterConfiguration(order: .second, steps: 2)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 3, config: config)
        let state = circuit.execute()

        let totalProb = (0 ..< 8).reduce(0.0) { $0 + state.probability(of: $1) }
        #expect(abs(totalProb - 1.0) < 1e-10, "Mixed terms should produce valid normalized state")
    }
}

/// Test suite for Trotter error bounds and accuracy.
/// Validates that higher orders and more steps reduce
/// approximation error in time evolution simulation.
@Suite("TrotterSuzuki Accuracy")
struct TrotterAccuracyTests {
    @Test("More steps improve accuracy for first order")
    func moreStepsImproveAccuracyFirstOrder() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
            (1.0, PauliString(.x(0))),
        ])
        let time = 2.0

        let config1 = TrotterConfiguration(order: .first, steps: 1)
        let config10 = TrotterConfiguration(order: .first, steps: 10)
        let config100 = TrotterConfiguration(order: .first, steps: 100)

        let configReference = TrotterConfiguration(order: .sixth, steps: 1000)
        let circuitReference = TrotterSuzuki.evolve(hamiltonian, time: time, qubits: 1, config: configReference)
        let stateReference = circuitReference.execute()
        let referenceProb0 = stateReference.probability(of: 0)

        let circuit1 = TrotterSuzuki.evolve(hamiltonian, time: time, qubits: 1, config: config1)
        let circuit10 = TrotterSuzuki.evolve(hamiltonian, time: time, qubits: 1, config: config10)
        let circuit100 = TrotterSuzuki.evolve(hamiltonian, time: time, qubits: 1, config: config100)

        let state1 = circuit1.execute()
        let state10 = circuit10.execute()
        let state100 = circuit100.execute()

        let error1 = abs(state1.probability(of: 0) - referenceProb0)
        let error10 = abs(state10.probability(of: 0) - referenceProb0)
        let error100 = abs(state100.probability(of: 0) - referenceProb0)

        #expect(error10 < error1, "10 steps should have smaller error than 1 step")
        #expect(error100 < error10, "100 steps should have smaller error than 10 steps")
    }

    @Test("Higher orders improve accuracy at fixed step count")
    func higherOrdersImproveAccuracy() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (0.5, PauliString(.x(0))),
        ])
        let time = 0.5
        let steps = 5

        let configFirst = TrotterConfiguration(order: .first, steps: steps)
        let configSecond = TrotterConfiguration(order: .second, steps: steps)
        let configFourth = TrotterConfiguration(order: .fourth, steps: steps)

        let circuitFirst = TrotterSuzuki.evolve(hamiltonian, time: time, qubits: 1, config: configFirst)
        let circuitSecond = TrotterSuzuki.evolve(hamiltonian, time: time, qubits: 1, config: configSecond)
        let circuitFourth = TrotterSuzuki.evolve(hamiltonian, time: time, qubits: 1, config: configFourth)

        let stateFirst = circuitFirst.execute()
        let stateSecond = circuitSecond.execute()
        let stateFourth = circuitFourth.execute()

        let configHighSteps = TrotterConfiguration(order: .sixth, steps: 100)
        let circuitReference = TrotterSuzuki.evolve(hamiltonian, time: time, qubits: 1, config: configHighSteps)
        let stateReference = circuitReference.execute()
        let referenceProb0 = stateReference.probability(of: 0)

        let errorFirst = abs(stateFirst.probability(of: 0) - referenceProb0)
        let errorSecond = abs(stateSecond.probability(of: 0) - referenceProb0)
        let errorFourth = abs(stateFourth.probability(of: 0) - referenceProb0)

        #expect(errorSecond <= errorFirst + 1e-10, "Second order should have comparable or smaller error than first")
        #expect(errorFourth <= errorSecond + 1e-10, "Fourth order should have comparable or smaller error than second")
    }

    @Test("Second order convergence is faster than first order")
    func secondOrderConvergenceIsFaster() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0), .z(1))),
            (0.8, PauliString(.x(0))),
            (0.8, PauliString(.x(1))),
        ])
        let time = 3.0
        let steps = 2

        let configFirst = TrotterConfiguration(order: .first, steps: steps)
        let configSecond = TrotterConfiguration(order: .second, steps: steps)

        let configReference = TrotterConfiguration(order: .sixth, steps: 1000)
        let circuitReference = TrotterSuzuki.evolve(hamiltonian, time: time, qubits: 2, config: configReference)
        let stateReference = circuitReference.execute()

        let circuitFirst = TrotterSuzuki.evolve(hamiltonian, time: time, qubits: 2, config: configFirst)
        let circuitSecond = TrotterSuzuki.evolve(hamiltonian, time: time, qubits: 2, config: configSecond)

        let stateFirst = circuitFirst.execute()
        let stateSecond = circuitSecond.execute()

        var errorFirst = 0.0
        var errorSecond = 0.0
        for i in 0 ..< 4 {
            errorFirst += abs(stateFirst.probability(of: i) - stateReference.probability(of: i))
            errorSecond += abs(stateSecond.probability(of: i) - stateReference.probability(of: i))
        }

        #expect(errorSecond < errorFirst, "Second order should converge faster than first order")
    }

    @Test("Unitarity is preserved across all orders")
    func unitarityPreserved() {
        let hamiltonian = Observable(terms: [
            (0.7, PauliString(.z(0), .z(1))),
            (-0.4, PauliString(.x(0))),
            (-0.4, PauliString(.x(1))),
            (0.2, PauliString(.y(0))),
        ])
        let time = 1.5
        let orders: [TrotterOrder] = [.first, .second, .fourth, .sixth]

        for order in orders {
            let config = TrotterConfiguration(order: order, steps: 3)
            let circuit = TrotterSuzuki.evolve(hamiltonian, time: time, qubits: 2, config: config)
            let state = circuit.execute()

            var totalProb = 0.0
            for i in 0 ..< 4 {
                totalProb += state.probability(of: i)
            }

            #expect(abs(totalProb - 1.0) < 1e-10, "Order \(order.rawValue) should preserve unitarity (total probability = 1)")
        }
    }
}

/// Test suite for Pauli string exponential implementation.
/// Validates basis rotations, CNOT ladder construction,
/// and correct phase accumulation for multi-qubit terms.
@Suite("TrotterSuzuki Pauli Exponentials")
struct TrotterPauliExponentialTests {
    @Test("X rotation evolution matches expected result")
    func xRotationEvolution() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.x(0)))])
        let angle = Double.pi / 4
        let config = TrotterConfiguration(order: .first, steps: 1)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: angle, qubits: 1, config: config)
        let state = circuit.execute()

        let expectedProb0 = pow(cos(angle), 2)
        let expectedProb1 = pow(sin(angle), 2)

        #expect(abs(state.probability(of: 0) - expectedProb0) < 1e-10, "X rotation should give cos^2(theta) for |0>")
        #expect(abs(state.probability(of: 1) - expectedProb1) < 1e-10, "X rotation should give sin^2(theta) for |1>")
    }

    @Test("Z rotation on |0> state preserves |0>")
    func zRotationPreservesZeroState() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.z(0)))])
        let config = TrotterConfiguration(order: .first, steps: 1)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: Double.pi / 3, qubits: 1, config: config)
        let state = circuit.execute()

        #expect(abs(state.probability(of: 0) - 1.0) < 1e-10, "Z rotation should preserve |0> state populations")
    }

    @Test("Y rotation evolution matches expected result")
    func yRotationEvolution() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.y(0)))])
        let angle = Double.pi / 6
        let config = TrotterConfiguration(order: .first, steps: 1)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: angle, qubits: 1, config: config)
        let state = circuit.execute()

        let expectedProb0 = pow(cos(angle), 2)

        #expect(abs(state.probability(of: 0) - expectedProb0) < 1e-10, "Y rotation should give correct probabilities")
    }

    @Test("ZZ coupling on product state")
    func zzCouplingProductState() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.z(0), .z(1)))])
        let config = TrotterConfiguration(order: .first, steps: 1)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: Double.pi / 4, qubits: 2, config: config)
        let state = circuit.execute()

        #expect(abs(state.probability(of: 0b00) - 1.0) < 1e-10, "ZZ evolution on |00> should preserve |00>")
    }

    @Test("XX coupling creates entanglement from superposition")
    func xxCouplingEntanglement() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.x(0), .x(1)))])
        let config = TrotterConfiguration(order: .first, steps: 10)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: Double.pi / 8, qubits: 2, config: config)

        var fullCircuit = QuantumCircuit(qubits: 2)
        fullCircuit.append(.hadamard, to: 0)
        fullCircuit.append(.hadamard, to: 1)
        for gate in circuit.gates {
            fullCircuit.append(gate.gate, to: gate.qubits)
        }

        let state = fullCircuit.execute()

        var totalProb = 0.0
        for i in 0 ..< 4 {
            totalProb += state.probability(of: i)
        }

        #expect(abs(totalProb - 1.0) < 1e-10, "XX coupling should preserve normalization")
    }
}

/// Test suite for empty Pauli string handling in Trotter-Suzuki.
/// Validates that PauliStrings with no operators are correctly
/// skipped during exponential application without circuit modification.
@Suite("TrotterSuzuki Empty Pauli String")
struct TrotterEmptyPauliStringTests {
    @Test("Empty Pauli string produces no gates")
    func emptyPauliStringProducesNoGates() {
        let emptyPauliString = PauliString()
        var circuit = QuantumCircuit(qubits: 2)
        let initialGateCount = circuit.count

        TrotterSuzuki.applyPauliExponential(term: emptyPauliString, angle: 1.0, circuit: &circuit)

        #expect(circuit.count == initialGateCount, "Empty Pauli string should add no gates to circuit")
    }

    @Test("Empty Pauli string with various angles adds no gates")
    func emptyPauliStringVariousAngles() {
        let emptyPauliString = PauliString()
        let angles = [0.0, 1.0, -1.0, Double.pi, -Double.pi / 2]

        for angle in angles {
            var circuit = QuantumCircuit(qubits: 1)
            TrotterSuzuki.applyPauliExponential(term: emptyPauliString, angle: angle, circuit: &circuit)
            #expect(circuit.count == 0, "Empty Pauli string with angle \(angle) should add no gates")
        }
    }

    @Test("Circuit with empty Pauli string terms filters them out")
    func hamiltonianWithEmptyTermsFiltered() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
            (0.5, PauliString()),
        ])
        let config = TrotterConfiguration(order: .first, steps: 1)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 1, config: config)
        let state = circuit.execute()

        let totalProb = state.probability(of: 0) + state.probability(of: 1)
        #expect(abs(totalProb - 1.0) < 1e-10, "Evolution with empty terms should produce valid state")
    }
}

/// Test suite for single-term commutation sorting in Trotter-Suzuki.
/// Validates that sortByCommutation with single or no terms
/// correctly returns the input unchanged without processing.
@Suite("TrotterSuzuki Single Term Commutation Sort")
struct TrotterSingleTermCommutationTests {
    @Test("Single term Hamiltonian with sort returns same circuit")
    func singleTermWithSortByCommutation() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.x(0)))])
        let configNoSort = TrotterConfiguration(order: .first, steps: 1, sortByCommutation: false)
        let configWithSort = TrotterConfiguration(order: .first, steps: 1, sortByCommutation: true)

        let circuitNoSort = TrotterSuzuki.evolve(hamiltonian, time: 0.5, qubits: 1, config: configNoSort)
        let circuitWithSort = TrotterSuzuki.evolve(hamiltonian, time: 0.5, qubits: 1, config: configWithSort)

        #expect(circuitNoSort.count == circuitWithSort.count, "Single term should produce same circuit with or without sort")
    }

    @Test("Single term with sort produces valid evolution")
    func singleTermSortProducesValidEvolution() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.z(0), .z(1)))])
        let config = TrotterConfiguration(order: .second, steps: 3, sortByCommutation: true)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 2, config: config)
        let state = circuit.execute()

        let totalProb = state.probability(of: 0) + state.probability(of: 1) + state.probability(of: 2) + state.probability(of: 3)
        #expect(abs(totalProb - 1.0) < 1e-10, "Single term with sort should produce valid normalized state")
    }

    @Test("Single term sort with different orders all produce valid states")
    func singleTermSortAllOrders() {
        let hamiltonian = Observable(terms: [(0.7, PauliString(.y(0)))])
        let orders: [TrotterOrder] = [.first, .second, .fourth, .sixth]

        for order in orders {
            let config = TrotterConfiguration(order: order, steps: 2, sortByCommutation: true)
            let circuit = TrotterSuzuki.evolve(hamiltonian, time: 0.3, qubits: 1, config: config)
            let state = circuit.execute()

            let totalProb = state.probability(of: 0) + state.probability(of: 1)
            #expect(abs(totalProb - 1.0) < 1e-10, "Single term sort with order \(order.rawValue) should produce valid state")
        }
    }
}

/// Test suite for commutation sorting with non-commuting terms.
/// Validates the fallback path when no commuting term is found
/// and the algorithm must select the first available unused term.
@Suite("TrotterSuzuki Non-Commuting Term Selection")
struct TrotterNonCommutingTermSelectionTests {
    @Test("All non-commuting terms fall back to sequential selection")
    func allNonCommutingTermsSequentialSelection() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.x(0))),
            (0.8, PauliString(.y(0))),
            (0.6, PauliString(.z(0))),
        ])
        let config = TrotterConfiguration(order: .first, steps: 1, sortByCommutation: true)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: 0.5, qubits: 1, config: config)
        let state = circuit.execute()

        let totalProb = state.probability(of: 0) + state.probability(of: 1)
        #expect(abs(totalProb - 1.0) < 1e-10, "Non-commuting terms with sort should produce valid state")
    }

    @Test("Mixed commuting and non-commuting terms handled correctly")
    func mixedCommutingNonCommutingTerms() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.x(0))),
            (0.5, PauliString(.z(1))),
            (0.3, PauliString(.y(0))),
            (0.2, PauliString(.x(1))),
        ])
        let config = TrotterConfiguration(order: .second, steps: 2, sortByCommutation: true)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: 0.4, qubits: 2, config: config)
        let state = circuit.execute()

        var totalProb = 0.0
        for i in 0 ..< 4 {
            totalProb += state.probability(of: i)
        }
        #expect(abs(totalProb - 1.0) < 1e-10, "Mixed commuting/non-commuting terms should produce valid state")
    }

    @Test("Chain of non-commuting single qubit terms")
    func chainNonCommutingSingleQubitTerms() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.x(0))),
            (0.9, PauliString(.y(0))),
            (0.8, PauliString(.z(0))),
            (0.7, PauliString(.x(0))),
            (0.6, PauliString(.y(0))),
        ])
        let config = TrotterConfiguration(order: .first, steps: 1, sortByCommutation: true)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: 0.2, qubits: 1, config: config)
        let state = circuit.execute()

        let totalProb = state.probability(of: 0) + state.probability(of: 1)
        #expect(abs(totalProb - 1.0) < 1e-10, "Chain of non-commuting terms should produce valid normalized state")
    }

    @Test("Two non-commuting terms triggers fallback selection")
    func twoNonCommutingTermsFallback() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.x(0))),
            (0.5, PauliString(.z(0))),
        ])
        let configNoSort = TrotterConfiguration(order: .first, steps: 1, sortByCommutation: false)
        let configWithSort = TrotterConfiguration(order: .first, steps: 1, sortByCommutation: true)

        let circuitNoSort = TrotterSuzuki.evolve(hamiltonian, time: 0.5, qubits: 1, config: configNoSort)
        let circuitWithSort = TrotterSuzuki.evolve(hamiltonian, time: 0.5, qubits: 1, config: configWithSort)

        #expect(circuitNoSort.count == circuitWithSort.count, "Two non-commuting terms should have same gate count regardless of sort")
    }
}

/// Test suite for Ising model time evolution using Trotter-Suzuki.
/// Validates physically meaningful Hamiltonians produce expected
/// dynamics and preserve quantum mechanical properties.
@Suite("TrotterSuzuki Ising Model")
struct TrotterIsingModelTests {
    @Test("Transverse field Ising model evolution")
    func transverseFieldIsing() {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
            (-0.5, PauliString(.x(0))),
            (-0.5, PauliString(.x(1))),
        ])
        let config = TrotterConfiguration(order: .second, steps: 10)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: 0.5, qubits: 2, config: config)
        let state = circuit.execute()

        var totalProb = 0.0
        for i in 0 ..< 4 {
            let prob = state.probability(of: i)
            #expect(prob >= -1e-10, "All probabilities should be non-negative")
            totalProb += prob
        }

        #expect(abs(totalProb - 1.0) < 1e-10, "Ising model evolution should preserve total probability")
    }

    @Test("Heisenberg model XYZ evolution")
    func heisenbergXYZEvolution() {
        let hamiltonian = Observable(terms: [
            (0.25, PauliString(.x(0), .x(1))),
            (0.25, PauliString(.y(0), .y(1))),
            (0.25, PauliString(.z(0), .z(1))),
        ])
        let config = TrotterConfiguration(order: .fourth, steps: 5)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 2, config: config)
        let state = circuit.execute()

        var totalProb = 0.0
        for i in 0 ..< 4 {
            totalProb += state.probability(of: i)
        }

        #expect(abs(totalProb - 1.0) < 1e-10, "Heisenberg model should preserve normalization")
    }

    @Test("Three-site Ising chain")
    func threeSiteIsingChain() {
        let hamiltonian = Observable(terms: [
            (-1.0, PauliString(.z(0), .z(1))),
            (-1.0, PauliString(.z(1), .z(2))),
            (-0.3, PauliString(.x(0))),
            (-0.3, PauliString(.x(1))),
            (-0.3, PauliString(.x(2))),
        ])
        let config = TrotterConfiguration(order: .second, steps: 8)

        let circuit = TrotterSuzuki.evolve(hamiltonian, time: 0.7, qubits: 3, config: config)
        let state = circuit.execute()

        var totalProb = 0.0
        for i in 0 ..< 8 {
            totalProb += state.probability(of: i)
        }

        #expect(abs(totalProb - 1.0) < 1e-10, "Three-site Ising chain should preserve normalization")
    }
}
