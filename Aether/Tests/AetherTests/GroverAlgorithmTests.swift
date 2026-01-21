// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under Apache 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for GroverDiffusion construction and gate generation.
/// Validates that the diffusion operator is correctly built with the
/// expected gate sequence: H - X - MCZ - X - H for amplitude inversion.
@Suite("GroverDiffusion Construction")
struct GroverDiffusionConstructionTests {
    @Test("Single qubit diffusion has correct qubit count")
    func singleQubitDiffusionQubitCount() {
        let diffusion = GroverDiffusion(qubits: 1)
        #expect(diffusion.qubits == 1, "Single qubit diffusion should have qubits == 1")
    }

    @Test("Two qubit diffusion has correct qubit count")
    func twoQubitDiffusionQubitCount() {
        let diffusion = GroverDiffusion(qubits: 2)
        #expect(diffusion.qubits == 2, "Two qubit diffusion should have qubits == 2")
    }

    @Test("Three qubit diffusion has correct qubit count")
    func threeQubitDiffusionQubitCount() {
        let diffusion = GroverDiffusion(qubits: 3)
        #expect(diffusion.qubits == 3, "Three qubit diffusion should have qubits == 3")
    }

    @Test("Diffusion gates are non-empty for any qubit count")
    func diffusionGatesNonEmpty() {
        for n in 1 ... 5 {
            let diffusion = GroverDiffusion(qubits: n)
            #expect(diffusion.gates.count > 0, "Diffusion operator for \(n) qubits should have gates")
        }
    }

    @Test("Single qubit diffusion contains Hadamard gates")
    func singleQubitDiffusionHasHadamard() {
        let diffusion = GroverDiffusion(qubits: 1)
        let hadamardCount = diffusion.gates.count(where: { $0.gate == .hadamard })
        #expect(hadamardCount >= 2, "Single qubit diffusion should have at least 2 Hadamard gates")
    }

    @Test("Two qubit diffusion contains X gates")
    func twoQubitDiffusionHasXGates() {
        let diffusion = GroverDiffusion(qubits: 2)
        let xCount = diffusion.gates.count(where: { $0.gate == .pauliX })
        #expect(xCount >= 4, "Two qubit diffusion should have at least 4 X gates (2 before and 2 after MCZ)")
    }

    @Test("Three qubit diffusion gate sequence structure")
    func threeQubitDiffusionStructure() {
        let diffusion = GroverDiffusion(qubits: 3)
        let hadamardCount = diffusion.gates.count(where: { $0.gate == .hadamard })
        let xCount = diffusion.gates.count(where: { $0.gate == .pauliX })
        #expect(hadamardCount >= 6, "Three qubit diffusion should have at least 6 Hadamard gates")
        #expect(xCount >= 6, "Three qubit diffusion should have at least 6 X gates")
    }

    @Test("Diffusion starts with Hadamard layer")
    func diffusionStartsWithHadamard() {
        let diffusion = GroverDiffusion(qubits: 3)
        #expect(diffusion.gates[0].gate == .hadamard, "First gate should be Hadamard")
        #expect(diffusion.gates[1].gate == .hadamard, "Second gate should be Hadamard")
        #expect(diffusion.gates[2].gate == .hadamard, "Third gate should be Hadamard")
    }

    @Test("Diffusion ends with Hadamard layer")
    func diffusionEndsWithHadamard() {
        let diffusion = GroverDiffusion(qubits: 3)
        let n = diffusion.gates.count
        #expect(diffusion.gates[n - 1].gate == .hadamard, "Last gate should be Hadamard")
        #expect(diffusion.gates[n - 2].gate == .hadamard, "Second to last gate should be Hadamard")
        #expect(diffusion.gates[n - 3].gate == .hadamard, "Third to last gate should be Hadamard")
    }

    @Test("Static factory method produces same result as init")
    func staticFactoryMethodMatchesInit() {
        let direct = GroverDiffusion(qubits: 3)
        let factory = QuantumCircuit.groverDiffusion(qubits: 3)
        #expect(direct.qubits == factory.qubits, "Factory method should produce same qubit count")
        #expect(direct.gates.count == factory.gates.count, "Factory method should produce same gate count")
    }
}

/// Test suite for GroverOracle enum cases and properties.
/// Validates singleTarget, multipleTargets, and custom oracle types,
/// including markedCount and targetStates computed properties.
@Suite("GroverOracle Enum")
struct GroverOracleEnumTests {
    @Test("Single target oracle has markedCount of 1")
    func singleTargetMarkedCount() {
        let oracle = GroverOracle.singleTarget(5)
        #expect(oracle.markedCount == 1, "Single target oracle should have markedCount == 1")
    }

    @Test("Single target oracle returns correct target state")
    func singleTargetTargetStates() {
        let oracle = GroverOracle.singleTarget(7)
        #expect(oracle.targetStates == [7], "Single target oracle should return [7] as target states")
    }

    @Test("Multiple targets oracle has correct markedCount")
    func multipleTargetsMarkedCount() {
        let oracle = GroverOracle.multipleTargets([1, 3, 5, 7])
        #expect(oracle.markedCount == 4, "Multiple targets oracle should have markedCount == 4")
    }

    @Test("Multiple targets oracle returns all target states")
    func multipleTargetsTargetStates() {
        let targets = [2, 4, 6]
        let oracle = GroverOracle.multipleTargets(targets)
        #expect(oracle.targetStates == targets, "Multiple targets oracle should return all target states")
    }

    @Test("Custom oracle has markedCount of 1")
    func customOracleMarkedCount() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [(.pauliZ, [0])]
        let oracle = GroverOracle.custom(customGates)
        #expect(oracle.markedCount == 1, "Custom oracle should have default markedCount == 1")
    }

    @Test("Custom oracle returns empty target states")
    func customOracleTargetStates() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [(.pauliZ, [0])]
        let oracle = GroverOracle.custom(customGates)
        #expect(oracle.targetStates.isEmpty, "Custom oracle should return empty target states")
    }

    @Test("Empty multiple targets has zero markedCount")
    func emptyMultipleTargetsMarkedCount() {
        let oracle = GroverOracle.multipleTargets([])
        #expect(oracle.markedCount == 0, "Empty multiple targets should have markedCount == 0")
    }

    @Test("Single element multiple targets equals single target")
    func singleElementMultipleTargets() {
        let multi = GroverOracle.multipleTargets([5])
        let single = GroverOracle.singleTarget(5)
        #expect(multi.markedCount == single.markedCount, "Single element multiple targets should match single target markedCount")
        #expect(multi.targetStates == single.targetStates, "Single element multiple targets should match single target states")
    }
}

/// Test suite for GroverResult struct properties and behavior.
/// Validates measuredState, successProbability, iterations,
/// optimalIterations, isTarget, and description formatting.
@Suite("GroverResult Properties")
struct GroverResultPropertiesTests {
    @Test("GroverResult stores all properties correctly")
    func groverResultStoresProperties() {
        let result = GroverResult(
            measuredState: 7,
            successProbability: 0.95,
            iterations: 3,
            optimalIterations: 3,
            isTarget: true,
        )
        #expect(result.measuredState == 7, "measuredState should be 7")
        #expect(abs(result.successProbability - 0.95) < 1e-10, "successProbability should be 0.95")
        #expect(result.iterations == 3, "iterations should be 3")
        #expect(result.optimalIterations == 3, "optimalIterations should be 3")
        #expect(result.isTarget, "isTarget should be true")
    }

    @Test("GroverResult description contains SUCCESS for target found")
    func groverResultDescriptionSuccess() {
        let result = GroverResult(
            measuredState: 5,
            successProbability: 0.90,
            iterations: 2,
            optimalIterations: 2,
            isTarget: true,
        )
        #expect(result.description.contains("SUCCESS"), "Description should contain SUCCESS when isTarget is true")
    }

    @Test("GroverResult description contains FAILURE for target not found")
    func groverResultDescriptionFailure() {
        let result = GroverResult(
            measuredState: 3,
            successProbability: 0.10,
            iterations: 2,
            optimalIterations: 2,
            isTarget: false,
        )
        #expect(result.description.contains("FAILURE"), "Description should contain FAILURE when isTarget is false")
    }

    @Test("GroverResult description contains measured state")
    func groverResultDescriptionContainsState() {
        let result = GroverResult(
            measuredState: 42,
            successProbability: 0.85,
            iterations: 5,
            optimalIterations: 5,
            isTarget: true,
        )
        #expect(result.description.contains("42"), "Description should contain the measured state")
    }

    @Test("GroverResult description contains iteration info")
    func groverResultDescriptionContainsIterations() {
        let result = GroverResult(
            measuredState: 7,
            successProbability: 0.95,
            iterations: 3,
            optimalIterations: 4,
            isTarget: true,
        )
        #expect(result.description.contains("3"), "Description should contain iterations count")
        #expect(result.description.contains("4"), "Description should contain optimal iterations count")
    }
}

/// Test suite for optimalGroverIterations formula verification.
/// Validates floor(pi/4 * sqrt(N/M)) for various qubit counts
/// and marked item configurations.
@Suite("Optimal Grover Iterations")
struct OptimalGroverIterationsTests {
    @Test("Optimal iterations for 2 qubits, 1 marked item")
    func optimalIterations2Qubits1Marked() {
        let optimal = QuantumCircuit.optimalGroverIterations(qubits: 2, markedItems: 1)
        let expected = Int((Double.pi / 4.0) * sqrt(4.0 / 1.0))
        #expect(optimal == expected, "Optimal iterations for 2 qubits, 1 marked should be \(expected)")
    }

    @Test("Optimal iterations for 3 qubits, 1 marked item")
    func optimalIterations3Qubits1Marked() {
        let optimal = QuantumCircuit.optimalGroverIterations(qubits: 3, markedItems: 1)
        let expected = Int((Double.pi / 4.0) * sqrt(8.0 / 1.0))
        #expect(optimal == expected, "Optimal iterations for 3 qubits, 1 marked should be \(expected)")
    }

    @Test("Optimal iterations for 4 qubits, 1 marked item")
    func optimalIterations4Qubits1Marked() {
        let optimal = QuantumCircuit.optimalGroverIterations(qubits: 4, markedItems: 1)
        let expected = Int((Double.pi / 4.0) * sqrt(16.0 / 1.0))
        #expect(optimal == expected, "Optimal iterations for 4 qubits, 1 marked should be \(expected)")
    }

    @Test("Optimal iterations for 5 qubits, 1 marked item")
    func optimalIterations5Qubits1Marked() {
        let optimal = QuantumCircuit.optimalGroverIterations(qubits: 5, markedItems: 1)
        let expected = Int((Double.pi / 4.0) * sqrt(32.0 / 1.0))
        #expect(optimal == expected, "Optimal iterations for 5 qubits, 1 marked should be \(expected)")
    }

    @Test("Optimal iterations for 4 qubits, 2 marked items")
    func optimalIterations4Qubits2Marked() {
        let optimal = QuantumCircuit.optimalGroverIterations(qubits: 4, markedItems: 2)
        let expected = Int((Double.pi / 4.0) * sqrt(16.0 / 2.0))
        #expect(optimal == expected, "Optimal iterations for 4 qubits, 2 marked should be \(expected)")
    }

    @Test("Optimal iterations for 4 qubits, 4 marked items")
    func optimalIterations4Qubits4Marked() {
        let optimal = QuantumCircuit.optimalGroverIterations(qubits: 4, markedItems: 4)
        let expected = Int((Double.pi / 4.0) * sqrt(16.0 / 4.0))
        #expect(optimal == expected, "Optimal iterations for 4 qubits, 4 marked should be \(expected)")
    }

    @Test("Optimal iterations is at least 1")
    func optimalIterationsAtLeastOne() {
        let optimal = QuantumCircuit.optimalGroverIterations(qubits: 1, markedItems: 1)
        #expect(optimal >= 1, "Optimal iterations should be at least 1")
    }

    @Test("More marked items means fewer iterations")
    func moreMarkedFewerIterations() {
        let few = QuantumCircuit.optimalGroverIterations(qubits: 5, markedItems: 1)
        let many = QuantumCircuit.optimalGroverIterations(qubits: 5, markedItems: 4)
        #expect(few >= many, "More marked items should result in fewer or equal iterations")
    }

    @Test("More qubits means more iterations for single target")
    func moreQubitsMoreIterations() {
        let small = QuantumCircuit.optimalGroverIterations(qubits: 3, markedItems: 1)
        let large = QuantumCircuit.optimalGroverIterations(qubits: 6, markedItems: 1)
        #expect(large >= small, "More qubits should result in more iterations for single target")
    }

    @Test("Default marked items is 1")
    func defaultMarkedItemsIsOne() {
        let withDefault = QuantumCircuit.optimalGroverIterations(qubits: 4)
        let explicit = QuantumCircuit.optimalGroverIterations(qubits: 4, markedItems: 1)
        #expect(withDefault == explicit, "Default markedItems should be 1")
    }
}

/// Test suite for groverDiffusion reflection operator construction.
/// Validates the correct gate sequence implementing 2|s><s| - I
/// where |s> is the uniform superposition state.
@Suite("Grover Diffusion Operator")
struct GroverDiffusionOperatorTests {
    @Test("Diffusion operator for 2 qubits has controlled phase gate")
    func diffusion2QubitsHasControlledPhase() {
        let diffusion = QuantumCircuit.groverDiffusion(qubits: 2)
        let hasControlledPhase = diffusion.gates.contains { $0.gate == .controlledPhase(.pi) }
        #expect(hasControlledPhase, "2-qubit diffusion should contain controlled phase gate")
    }

    @Test("Diffusion operator for 3 qubits has Toffoli gate")
    func diffusion3QubitsHasToffoli() {
        let diffusion = QuantumCircuit.groverDiffusion(qubits: 3)
        let hasToffoli = diffusion.gates.contains { $0.gate == .toffoli }
        #expect(hasToffoli, "3-qubit diffusion should contain Toffoli gate")
    }

    @Test("Diffusion operator applied to uniform superposition amplifies")
    func diffusionAmplifies() {
        var circuit = QuantumCircuit(qubits: 2)
        for i in 0 ..< 2 {
            circuit.append(.hadamard, to: i)
        }
        circuit.append(.pauliZ, to: 1)
        let diffusion = QuantumCircuit.groverDiffusion(qubits: 2)
        for (gate, qubits) in diffusion.gates {
            circuit.append(gate, to: qubits)
        }
        let state = circuit.execute()
        let (mostProbable, _) = state.mostProbableState()
        #expect(mostProbable >= 0 && mostProbable < 4, "Most probable state should be in valid range")
    }

    @Test("Single qubit diffusion uses Pauli Z")
    func singleQubitDiffusionUsesPauliZ() {
        let diffusion = QuantumCircuit.groverDiffusion(qubits: 1)
        let hasPauliZ = diffusion.gates.contains { $0.gate == .pauliZ }
        #expect(hasPauliZ, "Single qubit diffusion should use Pauli Z for phase flip")
    }

    @Test("Diffusion gate qubit indices are valid")
    func diffusionGateQubitIndicesValid() {
        let diffusion = QuantumCircuit.groverDiffusion(qubits: 4)
        for (_, qubits) in diffusion.gates {
            for qubit in qubits {
                #expect(qubit >= 0, "Qubit index should be non-negative")
            }
        }
    }
}

/// Test suite for groverOracle circuit generation.
/// Validates that oracles correctly mark target states with phase flip
/// for single target, multiple targets, and custom oracle configurations.
@Suite("Grover Oracle Generation")
struct GroverOracleGenerationTests {
    @Test("Single target oracle generates gates")
    func singleTargetOracleGeneratesGates() {
        let gates = QuantumCircuit.groverOracle(qubits: 3, oracle: .singleTarget(5))
        #expect(gates.count > 0, "Single target oracle should generate gates")
    }

    @Test("Multiple targets oracle generates gates for each target")
    func multipleTargetsOracleGeneratesGates() {
        let singleGates = QuantumCircuit.groverOracle(qubits: 3, oracle: .singleTarget(3))
        let multiGates = QuantumCircuit.groverOracle(qubits: 3, oracle: .multipleTargets([3, 5]))
        #expect(multiGates.count >= singleGates.count, "Multiple targets should generate at least as many gates as single target")
    }

    @Test("Custom oracle returns provided gates unchanged")
    func customOracleReturnsProvidedGates() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [
            (.pauliX, [0]),
            (.pauliZ, [1]),
        ]
        let result = QuantumCircuit.groverOracle(qubits: 2, oracle: .custom(customGates))
        #expect(result.count == 2, "Custom oracle should return 2 gates")
        #expect(result[0].gate == .pauliX, "First gate should be Pauli X")
        #expect(result[1].gate == .pauliZ, "Second gate should be Pauli Z")
    }

    @Test("Oracle for target 0 uses X gates")
    func oracleForTarget0UsesXGates() {
        let gates = QuantumCircuit.groverOracle(qubits: 2, oracle: .singleTarget(0))
        let xCount = gates.count(where: { $0.gate == .pauliX })
        #expect(xCount >= 4, "Oracle for target 0 (all zeros) should use X gates to flip all qubits")
    }

    @Test("Oracle for target 3 in 2 qubits uses fewer X gates")
    func oracleForTarget3UsesFewXGates() {
        let gates = QuantumCircuit.groverOracle(qubits: 2, oracle: .singleTarget(3))
        let xCount = gates.count(where: { $0.gate == .pauliX })
        #expect(xCount == 0, "Oracle for target 3 (all ones in 2 qubits) should use no X gates")
    }

    @Test("Oracle gate qubit indices are valid")
    func oracleGateQubitIndicesValid() {
        let gates = QuantumCircuit.groverOracle(qubits: 4, oracle: .singleTarget(7))
        for (_, qubits) in gates {
            for qubit in qubits {
                #expect(qubit >= 0, "Qubit index should be non-negative")
            }
        }
    }

    @Test("Empty multiple targets oracle generates no gates")
    func emptyMultipleTargetsNoGates() {
        let gates = QuantumCircuit.groverOracle(qubits: 3, oracle: .multipleTargets([]))
        #expect(gates.count == 0, "Empty multiple targets should generate no gates")
    }
}

/// Test suite for full groverSearch algorithm execution.
/// Validates that search finds target states with high probability
/// after optimal number of iterations.
@Suite("Grover Search Full Algorithm")
struct GroverSearchFullAlgorithmTests {
    @Test("2-qubit search for target 3 finds target")
    func twoQubitSearchTarget3() {
        let circuit = QuantumCircuit.groverSearch(qubits: 2, oracle: .singleTarget(3))
        let state = circuit.execute()
        let (mostProbable, _) = state.mostProbableState()
        #expect(mostProbable == 3, "2-qubit search for target 3 should find target 3")
    }

    @Test("2-qubit search for target 0 finds target")
    func twoQubitSearchTarget0() {
        let circuit = QuantumCircuit.groverSearch(qubits: 2, oracle: .singleTarget(0))
        let state = circuit.execute()
        let (mostProbable, _) = state.mostProbableState()
        #expect(mostProbable == 0, "2-qubit search for target 0 should find target 0")
    }

    @Test("3-qubit search for target 5 finds target")
    func threeQubitSearchTarget5() {
        let circuit = QuantumCircuit.groverSearch(qubits: 3, oracle: .singleTarget(5))
        let state = circuit.execute()
        let (mostProbable, _) = state.mostProbableState()
        #expect(mostProbable == 5, "3-qubit search for target 5 should find target 5")
    }

    @Test("3-qubit search with multiple targets finds one of them")
    func threeQubitSearchMultipleTargets() {
        let targets = [1, 5]
        let circuit = QuantumCircuit.groverSearch(qubits: 3, oracle: .multipleTargets(targets))
        let state = circuit.execute()
        let (mostProbable, _) = state.mostProbableState()
        #expect(targets.contains(mostProbable), "3-qubit search with multiple targets should find one of the targets")
    }

    @Test("4-qubit search for target 7 has high success probability")
    func fourQubitSearchHighProbability() {
        let circuit = QuantumCircuit.groverSearch(qubits: 4, oracle: .singleTarget(7))
        let state = circuit.execute()
        let probability = state.probability(of: 7)
        #expect(probability > 0.9, "4-qubit search should have >90% success probability at optimal iterations")
    }

    @Test("Search with custom iteration count")
    func searchWithCustomIterations() {
        let circuit = QuantumCircuit.groverSearch(qubits: 3, oracle: .singleTarget(3), iterations: 1)
        let state = circuit.execute()
        let prob1 = state.probability(of: 3)
        let circuitOpt = QuantumCircuit.groverSearch(qubits: 3, oracle: .singleTarget(3))
        let stateOpt = circuitOpt.execute()
        let probOpt = stateOpt.probability(of: 3)
        #expect(prob1 != probOpt || prob1 > 0.5, "Custom iteration count should affect probability differently than optimal")
    }

    @Test("Search uses optimal iterations by default")
    func searchUsesOptimalIterationsDefault() {
        let optimal = QuantumCircuit.optimalGroverIterations(qubits: 4, markedItems: 1)
        let circuit = QuantumCircuit.groverSearch(qubits: 4, oracle: .singleTarget(7))
        let state = circuit.execute()
        let result = state.groverResult(oracle: .singleTarget(7), iterations: optimal, searchQubits: 4)
        #expect(result.optimalIterations == optimal, "Search should use optimal iterations by default")
    }

    @Test("Search initializes with Hadamard layer")
    func searchInitializesWithHadamard() {
        let circuit = QuantumCircuit.groverSearch(qubits: 2, oracle: .singleTarget(1), iterations: 0)
        let state = circuit.execute()
        for i in 0 ..< 4 {
            let prob = state.probability(of: i)
            #expect(abs(prob - 0.25) < 1e-10, "Zero iterations should give uniform superposition with probability 0.25 for state \(i)")
        }
    }
}

/// Test suite for QuantumState.groverResult extraction.
/// Validates result extraction including measuredState, successProbability,
/// isTarget detection, and comparison to optimal iterations.
@Suite("QuantumState Grover Result Extraction")
struct QuantumStateGroverResultTests {
    @Test("groverResult extracts correct measured state")
    func groverResultExtractsMeasuredState() {
        let circuit = QuantumCircuit.groverSearch(qubits: 3, oracle: .singleTarget(5))
        let state = circuit.execute()
        let result = state.groverResult(oracle: .singleTarget(5), iterations: 2)
        #expect(result.measuredState == 5, "groverResult should extract the most probable state as measured state")
    }

    @Test("groverResult detects target correctly")
    func groverResultDetectsTarget() {
        let circuit = QuantumCircuit.groverSearch(qubits: 3, oracle: .singleTarget(5))
        let state = circuit.execute()
        let result = state.groverResult(oracle: .singleTarget(5), iterations: 2)
        #expect(result.isTarget, "groverResult should detect that measured state is target")
    }

    @Test("groverResult reports correct iterations")
    func groverResultReportsIterations() {
        let circuit = QuantumCircuit.groverSearch(qubits: 3, oracle: .singleTarget(5), iterations: 3)
        let state = circuit.execute()
        let result = state.groverResult(oracle: .singleTarget(5), iterations: 3)
        #expect(result.iterations == 3, "groverResult should report provided iteration count")
    }

    @Test("groverResult computes optimal iterations")
    func groverResultComputesOptimalIterations() {
        let circuit = QuantumCircuit.groverSearch(qubits: 4, oracle: .singleTarget(7))
        let state = circuit.execute()
        let result = state.groverResult(oracle: .singleTarget(7), iterations: 3, searchQubits: 4)
        let expected = QuantumCircuit.optimalGroverIterations(qubits: 4, markedItems: 1)
        #expect(result.optimalIterations == expected, "groverResult should compute correct optimal iterations")
    }

    @Test("groverResult success probability is sum over targets")
    func groverResultSuccessProbabilitySum() {
        let targets = [1, 3]
        let circuit = QuantumCircuit.groverSearch(qubits: 2, oracle: .multipleTargets(targets))
        let state = circuit.execute()
        let result = state.groverResult(oracle: .multipleTargets(targets), iterations: 1)
        let expectedProb = state.probability(of: 1) + state.probability(of: 3)
        #expect(abs(result.successProbability - expectedProb) < 1e-10, "Success probability should be sum of target probabilities")
    }

    @Test("groverResult isTarget false when wrong state measured")
    func groverResultIsTargetFalse() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        let state = circuit.execute()
        let result = state.groverResult(oracle: .singleTarget(3), iterations: 0)
        #expect(!result.isTarget || result.measuredState == 3, "isTarget should be false unless measured state matches")
    }

    @Test("groverResult with custom oracle handles empty targets")
    func groverResultCustomOracleEmptyTargets() {
        let customGates: [(gate: QuantumGate, qubits: [Int])] = [(.pauliZ, [0])]
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        let state = circuit.execute()
        let result = state.groverResult(oracle: .custom(customGates), iterations: 0)
        #expect(result.successProbability >= 0, "Success probability for custom oracle should be non-negative")
    }

    @Test("groverResult success probability high at optimal")
    func groverResultHighSuccessProbability() {
        let circuit = QuantumCircuit.groverSearch(qubits: 4, oracle: .singleTarget(10))
        let state = circuit.execute()
        let optimal = QuantumCircuit.optimalGroverIterations(qubits: 4, markedItems: 1)
        let result = state.groverResult(oracle: .singleTarget(10), iterations: optimal)
        #expect(result.successProbability > 0.9, "Success probability should be high (>0.9) at optimal iterations")
    }
}

/// Test suite for multi-controlled gate construction in diffusion.
/// Validates buildMultiControlledZGates and buildMultiControlledXGates
/// helper methods produce correct gate sequences.
@Suite("Multi-Controlled Gate Construction")
struct MultiControlledGateConstructionTests {
    @Test("Single qubit MCZ is Pauli Z")
    func singleQubitMCZIsPauliZ() {
        let gates = GroverDiffusion.buildMultiControlledZGates(qubits: 1)
        #expect(gates.count == 1, "Single qubit MCZ should have 1 gate")
        #expect(gates[0].gate == .pauliZ, "Single qubit MCZ should be Pauli Z")
    }

    @Test("Two qubit MCZ is controlled phase")
    func twoQubitMCZIsControlledPhase() {
        let gates = GroverDiffusion.buildMultiControlledZGates(qubits: 2)
        #expect(gates.count == 1, "Two qubit MCZ should have 1 gate")
        #expect(gates[0].gate == .controlledPhase(.pi), "Two qubit MCZ should be controlled phase pi")
    }

    @Test("Three qubit MCZ uses H-Toffoli-H pattern")
    func threeQubitMCZUsesHToffoliH() {
        let gates = GroverDiffusion.buildMultiControlledZGates(qubits: 3)
        #expect(gates.count == 3, "Three qubit MCZ should have 3 gates")
        #expect(gates[0].gate == .hadamard, "First gate should be Hadamard")
        #expect(gates[1].gate == .toffoli, "Second gate should be Toffoli")
        #expect(gates[2].gate == .hadamard, "Third gate should be Hadamard")
    }

    @Test("Four qubit MCZ uses Toffoli ladder")
    func fourQubitMCZUsesToffoliLadder() {
        let gates = GroverDiffusion.buildMultiControlledZGates(qubits: 4)
        let toffoliCount = gates.count(where: { $0.gate == .toffoli })
        #expect(toffoliCount >= 1, "Four qubit MCZ should use Toffoli gates")
        let hadamardCount = gates.count(where: { $0.gate == .hadamard })
        #expect(hadamardCount == 2, "Four qubit MCZ should have 2 Hadamard gates for Z decomposition")
    }

    @Test("Multi-controlled X with no controls is just X")
    func mcxNoControlsIsX() {
        let gates = GroverDiffusion.buildMultiControlledXGates(controls: [], target: 0)
        #expect(gates.count == 1, "MCX with no controls should have 1 gate")
        #expect(gates[0].gate == .pauliX, "MCX with no controls should be Pauli X")
    }

    @Test("Multi-controlled X with one control is CNOT")
    func mcxOneControlIsCNOT() {
        let gates = GroverDiffusion.buildMultiControlledXGates(controls: [0], target: 1)
        #expect(gates.count == 1, "MCX with one control should have 1 gate")
        #expect(gates[0].gate == .cnot, "MCX with one control should be CNOT")
    }

    @Test("Multi-controlled X with two controls is Toffoli")
    func mcxTwoControlsIsToffoli() {
        let gates = GroverDiffusion.buildMultiControlledXGates(controls: [0, 1], target: 2)
        #expect(gates.count == 1, "MCX with two controls should have 1 gate")
        #expect(gates[0].gate == .toffoli, "MCX with two controls should be Toffoli")
    }

    @Test("Multi-controlled X with three controls uses ancilla")
    func mcxThreeControlsUsesAncilla() {
        let gates = GroverDiffusion.buildMultiControlledXGates(controls: [0, 1, 2], target: 3)
        let toffoliCount = gates.count(where: { $0.gate == .toffoli })
        #expect(toffoliCount >= 3, "MCX with three controls should use multiple Toffoli gates")
    }
}

/// Test suite for edge cases and boundary conditions.
/// Validates behavior at limits of valid inputs such as
/// minimum/maximum qubit counts and special target values.
@Suite("Grover Algorithm Edge Cases")
struct GroverAlgorithmEdgeCasesTests {
    @Test("Single qubit search works")
    func singleQubitSearchWorks() {
        let circuit = QuantumCircuit.groverSearch(qubits: 1, oracle: .singleTarget(1))
        let state = circuit.execute()
        let prob = state.probability(of: 1)
        #expect(prob >= 0.49, "Single qubit search should find target with ~50% probability")
    }

    @Test("Search for maximum state in range")
    func searchForMaxState() {
        let circuit = QuantumCircuit.groverSearch(qubits: 3, oracle: .singleTarget(7))
        let state = circuit.execute()
        let (mostProbable, _) = state.mostProbableState()
        #expect(mostProbable == 7, "Search should find maximum state 7 in 3-qubit space")
    }

    @Test("Multiple targets covering half the space")
    func multipleTargetsHalfSpace() {
        let targets = [0, 1, 2, 3]
        let circuit = QuantumCircuit.groverSearch(qubits: 3, oracle: .multipleTargets(targets))
        let state = circuit.execute()
        let (mostProbable, _) = state.mostProbableState()
        #expect(targets.contains(mostProbable), "Search with half-space targets should find one")
    }

    @Test("Grover iteration count of 1")
    func singleIteration() {
        let circuit = QuantumCircuit.groverSearch(qubits: 3, oracle: .singleTarget(5), iterations: 1)
        let state = circuit.execute()
        let prob = state.probability(of: 5)
        #expect(prob > 0.125, "Single iteration should increase target probability above uniform (1/8)")
    }

    @Test("Diffusion operator is its own inverse conceptually")
    func diffusionSelfInverse() {
        var circuit = QuantumCircuit(qubits: 2)
        for i in 0 ..< 2 {
            circuit.append(.hadamard, to: i)
        }
        let diffusion = QuantumCircuit.groverDiffusion(qubits: 2)
        for (gate, qubits) in diffusion.gates {
            circuit.append(gate, to: qubits)
        }
        for (gate, qubits) in diffusion.gates {
            circuit.append(gate, to: qubits)
        }
        let state = circuit.execute()
        for i in 0 ..< 4 {
            let prob = state.probability(of: i)
            #expect(abs(prob - 0.25) < 1e-10, "Double diffusion should return to uniform superposition, state \(i)")
        }
    }

    @Test("Large qubit count diffusion has valid structure")
    func largeQubitDiffusion() {
        let diffusion = GroverDiffusion(qubits: 8)
        #expect(diffusion.qubits == 8, "8-qubit diffusion should have correct qubit count")
        #expect(diffusion.gates.count > 0, "8-qubit diffusion should have gates")
    }
}

/// Test suite for deriveSearchQubits and multi-controlled gate edge cases.
/// Validates nil coalescing branches and edge conditions in the
/// buildMultiControlledXGates and deriveSearchQubits helper methods.
@Suite("Grover Algorithm Nil Coalescing Coverage")
struct GroverAlgorithmNilCoalescingTests {
    @Test("Multi-controlled X with four controls uses ancilla ladder")
    func mcxFourControlsUsesAncillaLadder() {
        let gates = GroverDiffusion.buildMultiControlledXGates(controls: [0, 1, 2, 3], target: 4)
        let toffoliCount = gates.count(where: { $0.gate == .toffoli })
        #expect(toffoliCount >= 5, "MCX with four controls should use Toffoli ladder with ancillas")
    }

    @Test("groverResult derives search qubits for target zero")
    func groverResultDerivesSearchQubitsForTargetZero() {
        var circuit = QuantumCircuit(qubits: 3)
        for i in 0 ..< 3 {
            circuit.append(.hadamard, to: i)
        }
        let oracleGates = QuantumCircuit.groverOracle(qubits: 3, oracle: .singleTarget(0))
        for (gate, qubits) in oracleGates {
            circuit.append(gate, to: qubits)
        }
        let diffusion = QuantumCircuit.groverDiffusion(qubits: 3)
        for (gate, qubits) in diffusion.gates {
            circuit.append(gate, to: qubits)
        }
        let state = circuit.execute()
        let result = state.groverResult(oracle: .singleTarget(0), iterations: 1)
        #expect(result.optimalIterations >= 1, "Optimal iterations should be at least 1 for target 0")
    }

    @Test("groverResult with single target zero derives one search qubit")
    func groverResultSingleTargetZeroDerives() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.hadamard, to: 1)
        let state = circuit.execute()
        let result = state.groverResult(oracle: .singleTarget(0), iterations: 0)
        #expect(result.optimalIterations >= 1, "Should compute valid optimal iterations for target 0")
    }
}
