// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for BlockEncodingConfiguration properties.
/// Validates normalized coefficients, ancilla qubit count,
/// one-norm, and subnormalization factor storage.
@Suite("BlockEncodingConfiguration Properties")
struct BlockEncodingConfigurationTests {
    @Test("Normalized coefficients are stored correctly")
    func normalizedCoefficientsStored() {
        let coeffs = [0.5, 0.3, 0.2]
        let config = BlockEncodingConfiguration(
            normalizedCoefficients: coeffs,
            ancillaQubits: 2,
            oneNorm: 1.0,
            subnormalization: 1.0,
        )
        #expect(config.normalizedCoefficients.count == 3, "Should store 3 coefficients")
        #expect(abs(config.normalizedCoefficients[0] - 0.5) < 1e-10, "First coefficient should be 0.5")
        #expect(abs(config.normalizedCoefficients[1] - 0.3) < 1e-10, "Second coefficient should be 0.3")
        #expect(abs(config.normalizedCoefficients[2] - 0.2) < 1e-10, "Third coefficient should be 0.2")
    }

    @Test("Ancilla qubit count is stored correctly")
    func ancillaQubitsStored() {
        let config = BlockEncodingConfiguration(
            normalizedCoefficients: [0.6, 0.4],
            ancillaQubits: 3,
            oneNorm: 2.5,
            subnormalization: 0.9,
        )
        #expect(config.ancillaQubits == 3, "Ancilla qubits should be 3")
    }

    @Test("One-norm is stored correctly")
    func oneNormStored() {
        let config = BlockEncodingConfiguration(
            normalizedCoefficients: [1.0],
            ancillaQubits: 1,
            oneNorm: 5.5,
            subnormalization: 1.0,
        )
        #expect(abs(config.oneNorm - 5.5) < 1e-10, "One-norm should be 5.5")
    }

    @Test("Subnormalization factor is stored correctly")
    func subnormalizationStored() {
        let config = BlockEncodingConfiguration(
            normalizedCoefficients: [0.7, 0.3],
            ancillaQubits: 1,
            oneNorm: 1.0,
            subnormalization: 0.85,
        )
        #expect(abs(config.subnormalization - 0.85) < 1e-10, "Subnormalization should be 0.85")
    }

    @Test("Empty coefficients array is handled")
    func emptyCoefficientsHandled() {
        let config = BlockEncodingConfiguration(
            normalizedCoefficients: [],
            ancillaQubits: 0,
            oneNorm: 0.0,
            subnormalization: 1.0,
        )
        #expect(config.normalizedCoefficients.isEmpty, "Coefficients array should be empty")
    }
}

/// Test suite for BlockEncoding construction and circuits.
/// Validates Hamiltonian block encoding via LCU decomposition
/// and circuit generation for PREPARE/SELECT oracles.
@Suite("BlockEncoding Construction and Circuits")
struct BlockEncodingTests {
    @Test("BlockEncoding stores Hamiltonian correctly")
    func hamiltonianStored() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 2)
        #expect(encoding.hamiltonian.terms.count == 2, "Hamiltonian should have 2 terms")
    }

    @Test("BlockEncoding computes systemQubits correctly")
    func systemQubitsComputed() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0), .z(1), .z(2))),
        ])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 3)
        #expect(encoding.systemQubits == 3, "System qubits should be 3")
    }

    @Test("BlockEncoding computes totalQubits correctly")
    func totalQubitsComputed() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 2)
        let expectedTotal = encoding.systemQubits + encoding.configuration.ancillaQubits
        #expect(encoding.totalQubits == expectedTotal, "Total qubits should equal system + ancilla")
    }

    @Test("BlockEncoding configuration has valid one-norm")
    func configurationOneNormValid() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (0.3, PauliString(.x(1))),
        ])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 2)
        #expect(encoding.configuration.oneNorm > 0, "One-norm should be positive")
        #expect(abs(encoding.configuration.oneNorm - 0.8) < 1e-10, "One-norm should be 0.8")
    }

    @Test("prepareCircuit returns valid circuit")
    func prepareCircuitValid() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 2)
        let circuit = encoding.prepareCircuit()
        #expect(circuit.qubits >= encoding.systemQubits, "Circuit should have at least system qubits")
    }

    @Test("selectCircuit returns valid circuit")
    func selectCircuitValid() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 2)
        let circuit = encoding.selectCircuit()
        #expect(circuit.qubits >= encoding.systemQubits, "Circuit should have at least system qubits")
    }

    @Test("blockEncodingCircuit returns valid circuit")
    func blockEncodingCircuitValid() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 2)
        let circuit = encoding.blockEncodingCircuit()
        #expect(circuit.qubits >= encoding.totalQubits, "Block encoding circuit should have enough qubits")
    }

    @Test("Single term Hamiltonian encoding")
    func singleTermHamiltonian() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
        ])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 1)
        #expect(encoding.configuration.normalizedCoefficients.count >= 1, "Should have at least one coefficient")
    }
}

/// Test suite for QSPPolynomialTarget enum cases.
/// Validates time evolution, sign function, inverse function,
/// Chebyshev, and custom polynomial target specifications.
@Suite("QSPPolynomialTarget Enum Cases")
struct QSPPolynomialTargetTests {
    @Test("TimeEvolution target stores time correctly")
    func timeEvolutionStoresTime() {
        let target = QSPPolynomialTarget.timeEvolution(time: 2.5)
        if case let .timeEvolution(time) = target {
            #expect(abs(time - 2.5) < 1e-10, "Time should be 2.5")
        }
    }

    @Test("SignFunction target stores threshold correctly")
    func signFunctionStoresThreshold() {
        let target = QSPPolynomialTarget.signFunction(threshold: 0.3)
        if case let .signFunction(threshold) = target {
            #expect(abs(threshold - 0.3) < 1e-10, "Threshold should be 0.3")
        }
    }

    @Test("InverseFunction target stores condition correctly")
    func inverseFunctionStoresCondition() {
        let target = QSPPolynomialTarget.inverseFunction(condition: 10.0)
        if case let .inverseFunction(condition) = target {
            #expect(abs(condition - 10.0) < 1e-10, "Condition should be 10.0")
        }
    }

    @Test("Chebyshev target stores degree correctly")
    func chebyshevStoresDegree() {
        let target = QSPPolynomialTarget.chebyshev(degree: 5)
        if case let .chebyshev(degree) = target {
            #expect(degree == 5, "Degree should be 5")
        }
    }

    @Test("Custom target stores coefficients correctly")
    func customStoresCoefficients() {
        let coeffs = [0.1, 0.2, 0.3, 0.4]
        let target = QSPPolynomialTarget.custom(coefficients: coeffs)
        if case let .custom(coefficients) = target {
            #expect(coefficients.count == 4, "Should have 4 coefficients")
            #expect(abs(coefficients[0] - 0.1) < 1e-10, "First coefficient should be 0.1")
        }
    }

    @Test("QSPPolynomialTarget is Equatable")
    func equatableConformance() {
        let target1 = QSPPolynomialTarget.timeEvolution(time: 1.0)
        let target2 = QSPPolynomialTarget.timeEvolution(time: 1.0)
        let target3 = QSPPolynomialTarget.timeEvolution(time: 2.0)
        #expect(target1 == target2, "Same targets should be equal")
        #expect(target1 != target3, "Different targets should not be equal")
    }

    @Test("Different target types are not equal")
    func differentTypesNotEqual() {
        let timeTarget = QSPPolynomialTarget.timeEvolution(time: 1.0)
        let signTarget = QSPPolynomialTarget.signFunction(threshold: 1.0)
        #expect(timeTarget != signTarget, "Different target types should not be equal")
    }
}

/// Test suite for QSPPhaseAngles properties.
/// Validates phase angle storage, polynomial degree,
/// target function reference, and approximation error.
@Suite("QSPPhaseAngles Properties")
struct QSPPhaseAnglesTests {
    @Test("Phases array is stored correctly")
    func phasesStored() {
        let phases = [0.1, -0.2, 0.3, -0.4]
        let angles = QSPPhaseAngles(
            phases: phases,
            polynomialDegree: 3,
            targetFunction: .timeEvolution(time: 1.0),
            approximationError: 1e-6,
        )
        #expect(angles.phases.count == 4, "Should store 4 phases")
        #expect(abs(angles.phases[0] - 0.1) < 1e-10, "First phase should be 0.1")
        #expect(abs(angles.phases[1] - -0.2) < 1e-10, "Second phase should be -0.2")
    }

    @Test("Polynomial degree is stored correctly")
    func polynomialDegreeStored() {
        let angles = QSPPhaseAngles(
            phases: [0.5, -0.5],
            polynomialDegree: 1,
            targetFunction: .chebyshev(degree: 1),
            approximationError: 0.0,
        )
        #expect(angles.polynomialDegree == 1, "Polynomial degree should be 1")
    }

    @Test("Target function is stored correctly")
    func targetFunctionStored() {
        let target = QSPPolynomialTarget.signFunction(threshold: 0.5)
        let angles = QSPPhaseAngles(
            phases: [0.1, 0.2, 0.3],
            polynomialDegree: 2,
            targetFunction: target,
            approximationError: 1e-4,
        )
        #expect(angles.targetFunction == target, "Target function should match")
    }

    @Test("Approximation error is stored correctly")
    func approximationErrorStored() {
        let angles = QSPPhaseAngles(
            phases: [0.0],
            polynomialDegree: 0,
            targetFunction: .chebyshev(degree: 0),
            approximationError: 1e-8,
        )
        #expect(abs(angles.approximationError - 1e-8) < 1e-15, "Approximation error should be 1e-8")
    }

    @Test("Empty phases array is handled")
    func emptyPhasesHandled() {
        let angles = QSPPhaseAngles(
            phases: [],
            polynomialDegree: 0,
            targetFunction: .chebyshev(degree: 0),
            approximationError: 0.0,
        )
        #expect(angles.phases.isEmpty, "Phases array should be empty")
    }
}

/// Test suite for QuantumSignalProcessing.computePhaseAngles().
/// Validates phase angle computation for time evolution,
/// sign function, inverse function, and Chebyshev targets.
@Suite("QuantumSignalProcessing Phase Angle Computation")
struct QuantumSignalProcessingComputeTests {
    @Test("Time evolution phase angles have correct count")
    func timeEvolutionPhaseCount() {
        let phases = QuantumSignalProcessing.computePhaseAngles(
            for: .timeEvolution(time: 1.0),
            degree: 10,
            epsilon: 1e-6,
        )
        #expect(phases.phases.count == phases.polynomialDegree + 1, "Phase count should be degree + 1")
    }

    @Test("Time evolution stores correct target function")
    func timeEvolutionTargetFunction() {
        let phases = QuantumSignalProcessing.computePhaseAngles(
            for: .timeEvolution(time: 2.0),
            degree: 10,
            epsilon: 1e-6,
        )
        if case let .timeEvolution(time) = phases.targetFunction {
            #expect(abs(time - 2.0) < 1e-10, "Time should be 2.0")
        } else {
            #expect(Bool(false), "Target function should be timeEvolution")
        }
    }

    @Test("Sign function phase angles are computed")
    func signFunctionPhaseAngles() {
        let phases = QuantumSignalProcessing.computePhaseAngles(
            for: .signFunction(threshold: 0.5),
            degree: 20,
            epsilon: 1e-4,
        )
        #expect(phases.phases.count > 0, "Should have at least one phase angle")
        #expect(phases.polynomialDegree > 0, "Polynomial degree should be positive")
    }

    @Test("Inverse function phase angles are computed")
    func inverseFunctionPhaseAngles() {
        let phases = QuantumSignalProcessing.computePhaseAngles(
            for: .inverseFunction(condition: 5.0),
            degree: 30,
            epsilon: 1e-4,
        )
        #expect(phases.phases.count > 0, "Should have at least one phase angle")
    }

    @Test("Chebyshev phase angles have exact degree")
    func chebyshevExactDegree() {
        let phases = QuantumSignalProcessing.computePhaseAngles(
            for: .chebyshev(degree: 5),
            degree: 10,
            epsilon: 1e-6,
        )
        #expect(phases.polynomialDegree == 5, "Polynomial degree should be exactly 5")
        #expect(phases.phases.count == 6, "Phase count should be 6 for degree 5")
    }

    @Test("Custom coefficients phase angles are computed")
    func customCoefficientsPhaseAngles() {
        let coeffs = [0.5, 0.0, 0.3, 0.0, 0.2]
        let phases = QuantumSignalProcessing.computePhaseAngles(
            for: .custom(coefficients: coeffs),
            degree: 10,
            epsilon: 1e-6,
        )
        #expect(phases.polynomialDegree == 4, "Polynomial degree should be coefficients.count - 1")
    }

    @Test("Zero time evolution gives minimal polynomial")
    func zeroTimeEvolution() {
        let phases = QuantumSignalProcessing.computePhaseAngles(
            for: .timeEvolution(time: 0.0),
            degree: 10,
            epsilon: 1e-6,
        )
        #expect(phases.polynomialDegree >= 1, "Should have at least degree 1")
    }

    @Test("Negative time evolution is handled")
    func negativeTimeEvolution() {
        let phases = QuantumSignalProcessing.computePhaseAngles(
            for: .timeEvolution(time: -1.5),
            degree: 10,
            epsilon: 1e-6,
        )
        #expect(phases.phases.count > 0, "Should compute phases for negative time")
    }
}

/// Test suite for QuantumSignalProcessing circuit building.
/// Validates QSP circuit construction with walk operators
/// and controlled circuit generation for phase estimation.
@Suite("QuantumSignalProcessing Circuit Building")
struct QuantumSignalProcessingCircuitTests {
    @Test("buildCircuit creates valid circuit")
    func buildCircuitValid() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 2)
        let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
        let phases = QuantumSignalProcessing.computePhaseAngles(
            for: .timeEvolution(time: 1.0),
            degree: 5,
            epsilon: 1e-4,
        )
        let circuit = QuantumSignalProcessing.buildCircuit(
            walkOperator: walkOp,
            phaseAngles: phases,
            signalQubit: 2,
        )
        #expect(circuit.qubits >= 3, "Circuit should have at least 3 qubits")
    }

    @Test("buildCircuit with empty phases returns minimal circuit")
    func buildCircuitEmptyPhases() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.z(0)))])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 1)
        let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
        let emptyPhases = QSPPhaseAngles(
            phases: [],
            polynomialDegree: 0,
            targetFunction: .chebyshev(degree: 0),
            approximationError: 0.0,
        )
        let circuit = QuantumSignalProcessing.buildCircuit(
            walkOperator: walkOp,
            phaseAngles: emptyPhases,
            signalQubit: 1,
        )
        #expect(circuit.gates.count == 0, "Empty phases should produce empty circuit")
    }

    @Test("buildControlledCircuit creates valid circuit")
    func buildControlledCircuitValid() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 2)
        let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
        let phases = QuantumSignalProcessing.computePhaseAngles(
            for: .timeEvolution(time: 1.0),
            degree: 3,
            epsilon: 1e-4,
        )
        let circuit = QuantumSignalProcessing.buildControlledCircuit(
            walkOperator: walkOp,
            phaseAngles: phases,
            signalQubit: 3,
            controlQubit: 0,
        )
        #expect(circuit.qubits >= 4, "Controlled circuit should have at least 4 qubits")
    }

    @Test("buildControlledCircuit with empty phases returns minimal circuit")
    func buildControlledCircuitEmptyPhases() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.z(0)))])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 1)
        let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
        let emptyPhases = QSPPhaseAngles(
            phases: [],
            polynomialDegree: 0,
            targetFunction: .chebyshev(degree: 0),
            approximationError: 0.0,
        )
        let circuit = QuantumSignalProcessing.buildControlledCircuit(
            walkOperator: walkOp,
            phaseAngles: emptyPhases,
            signalQubit: 2,
            controlQubit: 0,
        )
        #expect(circuit.gates.count == 0, "Empty phases should produce empty controlled circuit")
    }
}

/// Test suite for QubitizedWalkOperator construction.
/// Validates walk operator creation from block encoding
/// with automatic and explicit qubit index assignment.
@Suite("QubitizedWalkOperator Construction")
struct QubitizedWalkOperatorConstructionTests {
    @Test("Walk operator stores blockEncoding correctly")
    func blockEncodingStored() {
        let hamiltonian = Observable(terms: [(0.5, PauliString(.z(0)))])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 1)
        let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
        #expect(walkOp.blockEncoding.systemQubits == 1, "Block encoding should have 1 system qubit")
    }

    @Test("Walk operator computes systemQubits automatically")
    func systemQubitsAutomatic() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 2)
        let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
        #expect(walkOp.systemQubits == [0, 1], "System qubits should be [0, 1]")
    }

    @Test("Walk operator computes ancillaQubits automatically")
    func ancillaQubitsAutomatic() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 2)
        let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
        let expectedAncillaStart = 2
        let numAncilla = encoding.configuration.ancillaQubits
        let expectedAncillas = Array(expectedAncillaStart ..< expectedAncillaStart + numAncilla)
        #expect(walkOp.ancillaQubits == expectedAncillas, "Ancilla qubits should start after system qubits")
    }

    @Test("Walk operator accepts explicit qubit assignments")
    func explicitQubitAssignments() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.z(0)))])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 1)
        let walkOp = QubitizedWalkOperator(
            blockEncoding: encoding,
            ancillaQubits: [5, 6],
            systemQubits: [0, 1, 2, 3],
        )
        #expect(walkOp.ancillaQubits == [5, 6], "Ancilla qubits should be [5, 6]")
        #expect(walkOp.systemQubits == [0, 1, 2, 3], "System qubits should be [0, 1, 2, 3]")
    }
}

/// Test suite for QubitizedWalkOperator circuit building.
/// Validates walk circuit and controlled walk circuit
/// generation including reflection operators.
@Suite("QubitizedWalkOperator Circuit Building")
struct QubitizedWalkOperatorCircuitTests {
    @Test("buildWalkCircuit returns valid circuit")
    func buildWalkCircuitValid() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 2)
        let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
        let circuit = walkOp.buildWalkCircuit()
        #expect(circuit.qubits >= encoding.totalQubits, "Walk circuit should have enough qubits")
        #expect(circuit.gates.count > 0, "Walk circuit should have gates")
    }

    @Test("buildControlledWalkCircuit returns valid circuit")
    func buildControlledWalkCircuitValid() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 2)
        let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
        let circuit = walkOp.buildControlledWalkCircuit(controlQubit: 0)
        #expect(circuit.qubits >= encoding.totalQubits, "Controlled circuit should have enough qubits")
        #expect(circuit.gates.count > 0, "Controlled circuit should have gates")
    }

    @Test("Controlled walk circuit includes control qubit")
    func controlledWalkIncludesControl() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.z(0)))])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 1)
        let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
        let controlQubit = 10
        let circuit = walkOp.buildControlledWalkCircuit(controlQubit: controlQubit)
        #expect(circuit.qubits >= controlQubit + 1, "Circuit should include control qubit")
    }
}

/// Test suite for QubitizationComplexity analysis methods.
/// Validates theoretical bounds, query complexity, and
/// speedup calculations over Trotter methods.
@Suite("QubitizationComplexity Analysis")
struct QubitizationComplexityTests {
    @Test("Time evolution analysis computes optimal queries")
    func timeEvolutionOptimalQueries() {
        let analysis = QubitizationComplexity.analyzeTimeEvolution(
            oneNorm: 5.0,
            time: 2.0,
            epsilon: 1e-6,
        )
        #expect(analysis.optimalQueries > 0, "Optimal queries should be positive")
        #expect(analysis.polynomialDegree > 0, "Polynomial degree should be positive")
    }

    @Test("Time evolution analysis computes success probability")
    func timeEvolutionSuccessProbability() {
        let analysis = QubitizationComplexity.analyzeTimeEvolution(
            oneNorm: 5.0,
            time: 2.0,
            epsilon: 1e-6,
        )
        #expect(analysis.successProbability > 0.99, "Success probability should be near 1")
        #expect(analysis.successProbability <= 1.0, "Success probability should not exceed 1")
    }

    @Test("Time evolution complexity scales with alpha*t")
    func timeEvolutionScalesWithAlphaT() {
        let analysis1 = QubitizationComplexity.analyzeTimeEvolution(
            oneNorm: 1.0,
            time: 1.0,
            epsilon: 1e-6,
        )
        let analysis2 = QubitizationComplexity.analyzeTimeEvolution(
            oneNorm: 2.0,
            time: 2.0,
            epsilon: 1e-6,
        )
        #expect(analysis2.optimalQueries > analysis1.optimalQueries, "Larger alpha*t should require more queries")
    }

    @Test("Eigenvalue estimation analysis computes queries")
    func eigenvalueEstimationQueries() {
        let analysis = QubitizationComplexity.analyzeEigenvalueEstimation(
            oneNorm: 5.0,
            precisionBits: 10,
            successProbability: 0.99,
        )
        let expectedQueries = (1 << 10) - 1
        #expect(analysis.optimalQueries == expectedQueries, "Queries should be 2^n - 1")
    }

    @Test("Eigenvalue estimation computes precision")
    func eigenvalueEstimationPrecision() {
        let analysis = QubitizationComplexity.analyzeEigenvalueEstimation(
            oneNorm: 5.0,
            precisionBits: 10,
            successProbability: 0.99,
        )
        #expect(analysis.eigenvaluePrecision > 0, "Eigenvalue precision should be positive")
    }

    @Test("Eigenvalue estimation computes ancilla qubits")
    func eigenvalueEstimationAncillaQubits() {
        let analysis = QubitizationComplexity.analyzeEigenvalueEstimation(
            oneNorm: 5.0,
            precisionBits: 10,
            successProbability: 0.99,
        )
        #expect(analysis.ancillaQubits >= 10, "Ancilla qubits should be at least precision bits")
    }

    @Test("Speedup over Trotter is positive")
    func speedupOverTrotterPositive() {
        let speedup = QubitizationComplexity.computeSpeedupOverTrotter(
            oneNorm: 10.0,
            time: 5.0,
            epsilon: 1e-8,
            trotterOrder: 2,
        )
        #expect(speedup > 0, "Speedup should be positive")
    }

    @Test("Higher Trotter order gives smaller speedup")
    func higherTrotterOrderSmallerSpeedup() {
        let speedup1 = QubitizationComplexity.computeSpeedupOverTrotter(
            oneNorm: 10.0,
            time: 5.0,
            epsilon: 1e-8,
            trotterOrder: 1,
        )
        let speedup4 = QubitizationComplexity.computeSpeedupOverTrotter(
            oneNorm: 10.0,
            time: 5.0,
            epsilon: 1e-8,
            trotterOrder: 4,
        )
        #expect(speedup1 > speedup4, "Higher Trotter order should reduce speedup advantage")
    }

    @Test("Small epsilon gives larger speedup")
    func smallEpsilonLargerSpeedup() {
        let speedupLargeEps = QubitizationComplexity.computeSpeedupOverTrotter(
            oneNorm: 10.0,
            time: 5.0,
            epsilon: 1e-4,
            trotterOrder: 2,
        )
        let speedupSmallEps = QubitizationComplexity.computeSpeedupOverTrotter(
            oneNorm: 10.0,
            time: 5.0,
            epsilon: 1e-10,
            trotterOrder: 2,
        )
        #expect(speedupSmallEps > speedupLargeEps, "Smaller epsilon should give larger speedup")
    }
}

/// Test suite for QubitizationCircuits factory methods.
/// Validates circuit construction for time evolution,
/// walk operators, and block encoding via convenience methods.
@Suite("QubitizationCircuits Factory Methods")
struct QubitizationCircuitsTests {
    @Test("buildTimeEvolutionCircuit creates valid circuit")
    func buildTimeEvolutionCircuitValid() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let circuit = QubitizationCircuits.buildTimeEvolutionCircuit(
            hamiltonian: hamiltonian,
            systemQubits: 2,
            time: 1.0,
            epsilon: 1e-4,
        )
        #expect(circuit.qubits >= 2, "Circuit should have at least system qubits")
        #expect(circuit.gates.count > 0, "Circuit should have gates")
    }

    @Test("buildWalkOperatorCircuit creates valid circuit")
    func buildWalkOperatorCircuitValid() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let circuit = QubitizationCircuits.buildWalkOperatorCircuit(
            hamiltonian: hamiltonian,
            systemQubits: 2,
        )
        #expect(circuit.qubits >= 2, "Circuit should have at least system qubits")
        #expect(circuit.gates.count > 0, "Circuit should have gates")
    }

    @Test("buildBlockEncodingCircuit creates valid circuit")
    func buildBlockEncodingCircuitValid() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let circuit = QubitizationCircuits.buildBlockEncodingCircuit(
            hamiltonian: hamiltonian,
            systemQubits: 2,
        )
        #expect(circuit.qubits >= 2, "Circuit should have at least system qubits")
    }

    @Test("estimateResources returns valid estimates")
    func estimateResourcesValid() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
            (0.2, PauliString(.z(0), .z(1))),
        ])
        let (totalQubits, gateCount, depthEstimate) = QubitizationCircuits.estimateResources(
            hamiltonian: hamiltonian,
            systemQubits: 2,
            time: 1.0,
            epsilon: 1e-4,
        )
        #expect(totalQubits >= 2, "Total qubits should be at least system qubits")
        #expect(gateCount > 0, "Gate count should be positive")
        #expect(depthEstimate > 0, "Depth estimate should be positive")
    }

    @Test("Resource estimates scale with time")
    func resourceEstimatesScaleWithTime() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let (_, gates1, _) = QubitizationCircuits.estimateResources(
            hamiltonian: hamiltonian,
            systemQubits: 2,
            time: 1.0,
            epsilon: 1e-4,
        )
        let (_, gates2, _) = QubitizationCircuits.estimateResources(
            hamiltonian: hamiltonian,
            systemQubits: 2,
            time: 5.0,
            epsilon: 1e-4,
        )
        #expect(gates2 > gates1, "Larger time should require more gates")
    }
}

/// Test suite for QuantumCircuit qubitization extensions.
/// Validates convenience factory methods for qubitized evolution,
/// walk operators, and block encoding on QuantumCircuit type.
@Suite("QuantumCircuit Qubitization Extensions")
struct QuantumCircuitQubitizationExtensionsTests {
    @Test("qubitizedEvolution creates valid circuit")
    func qubitizedEvolutionValid() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let circuit = QuantumCircuit.qubitizedEvolution(
            hamiltonian: hamiltonian,
            systemQubits: 2,
            time: 1.0,
            epsilon: 1e-4,
        )
        #expect(circuit.qubits >= 2, "Circuit should have at least system qubits")
    }

    @Test("qubitizedWalkOperator creates valid circuit")
    func qubitizedWalkOperatorValid() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let circuit = QuantumCircuit.qubitizedWalkOperator(
            hamiltonian: hamiltonian,
            systemQubits: 2,
        )
        #expect(circuit.qubits >= 2, "Circuit should have at least system qubits")
    }

    @Test("blockEncoding creates valid circuit")
    func blockEncodingValid() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let circuit = QuantumCircuit.blockEncoding(
            hamiltonian: hamiltonian,
            systemQubits: 2,
        )
        #expect(circuit.qubits >= 2, "Circuit should have at least system qubits")
    }
}

/// Test suite for optimal complexity scaling O(alpha*t + log(1/epsilon)).
/// Validates that qubitization achieves optimal query complexity
/// for Hamiltonian simulation as proven theoretically.
@Suite("Optimal Complexity Scaling")
struct OptimalComplexityScalingTests {
    @Test("Complexity scales linearly with alpha*t")
    func linearScalingWithAlphaT() {
        let epsilon = 1e-6
        let analysis1 = QubitizationComplexity.analyzeTimeEvolution(
            oneNorm: 100.0,
            time: 1.0,
            epsilon: epsilon,
        )
        let analysis2 = QubitizationComplexity.analyzeTimeEvolution(
            oneNorm: 100.0,
            time: 10.0,
            epsilon: epsilon,
        )
        let ratio = Double(analysis2.optimalQueries) / Double(analysis1.optimalQueries)
        #expect(ratio < 15.0, "Scaling should be approximately linear in time")
        #expect(ratio > 2.0, "Queries should increase with time")
    }

    @Test("Complexity scales logarithmically with 1/epsilon")
    func logarithmicScalingWithEpsilon() {
        let oneNorm = 1.0
        let time = 1.0
        let analysis1 = QubitizationComplexity.analyzeTimeEvolution(
            oneNorm: oneNorm,
            time: time,
            epsilon: 1e-4,
        )
        let analysis2 = QubitizationComplexity.analyzeTimeEvolution(
            oneNorm: oneNorm,
            time: time,
            epsilon: 1e-8,
        )
        let queryDiff = analysis2.optimalQueries - analysis1.optimalQueries
        let logDiff = log(1e-4 / 1e-8) / log(2.0)
        #expect(Double(queryDiff) < 2.0 * logDiff, "Scaling should be logarithmic in 1/epsilon")
    }

    @Test("Polynomial degree matches theoretical bound")
    func polynomialDegreeMatchesBound() {
        let oneNorm = 5.0
        let time = 2.0
        let epsilon = 1e-6
        let analysis = QubitizationComplexity.analyzeTimeEvolution(
            oneNorm: oneNorm,
            time: time,
            epsilon: epsilon,
        )
        let theoreticalBound = Int(ceil(oneNorm * time + log(1.0 / epsilon) / log(2.0)))
        #expect(analysis.polynomialDegree == theoreticalBound, "Polynomial degree should match O(alpha*t + log(1/epsilon))")
    }

    @Test("Queries equal polynomial degree")
    func queriesEqualDegree() {
        let analysis = QubitizationComplexity.analyzeTimeEvolution(
            oneNorm: 3.0,
            time: 4.0,
            epsilon: 1e-8,
        )
        #expect(analysis.optimalQueries == analysis.polynomialDegree, "Queries should equal polynomial degree")
    }

    @Test("Zero time gives minimal complexity")
    func zeroTimeMinimalComplexity() {
        let analysis = QubitizationComplexity.analyzeTimeEvolution(
            oneNorm: 10.0,
            time: 0.0,
            epsilon: 1e-6,
        )
        let logTerm = log(1.0 / 1e-6) / log(2.0)
        #expect(Double(analysis.optimalQueries) <= logTerm + 1, "Zero time should give complexity dominated by log term")
    }
}

/// Test suite for QubitizationEigenvalueResult properties.
/// Validates eigenvalue, phase, one-norm, confidence interval,
/// and walk operator call count storage in results.
@Suite("QubitizationEigenvalueResult Properties")
struct QubitizationEigenvalueResultTests {
    @Test("Eigenvalue is stored correctly")
    func eigenvalueStored() {
        let result = QubitizationEigenvalueResult(
            eigenvalue: -1.5,
            phase: 2.094,
            oneNorm: 3.0,
            confidenceInterval: (-1.6, -1.4),
            walkOperatorCalls: 100,
        )
        #expect(abs(result.eigenvalue - -1.5) < 1e-10, "Eigenvalue should be -1.5")
    }

    @Test("Phase is stored correctly")
    func phaseStored() {
        let result = QubitizationEigenvalueResult(
            eigenvalue: -0.5,
            phase: 1.047,
            oneNorm: 1.0,
            confidenceInterval: (-0.6, -0.4),
            walkOperatorCalls: 50,
        )
        #expect(abs(result.phase - 1.047) < 1e-10, "Phase should be 1.047")
    }

    @Test("One-norm is stored correctly")
    func oneNormStored() {
        let result = QubitizationEigenvalueResult(
            eigenvalue: 0.0,
            phase: 1.571,
            oneNorm: 2.5,
            confidenceInterval: (-0.1, 0.1),
            walkOperatorCalls: 255,
        )
        #expect(abs(result.oneNorm - 2.5) < 1e-10, "One-norm should be 2.5")
    }

    @Test("Confidence interval is stored correctly")
    func confidenceIntervalStored() {
        let result = QubitizationEigenvalueResult(
            eigenvalue: 1.0,
            phase: 0.0,
            oneNorm: 1.0,
            confidenceInterval: (0.9, 1.1),
            walkOperatorCalls: 127,
        )
        #expect(abs(result.confidenceInterval.lower - 0.9) < 1e-10, "Lower bound should be 0.9")
        #expect(abs(result.confidenceInterval.upper - 1.1) < 1e-10, "Upper bound should be 1.1")
    }

    @Test("Walk operator calls is stored correctly")
    func walkOperatorCallsStored() {
        let result = QubitizationEigenvalueResult(
            eigenvalue: 0.5,
            phase: 1.047,
            oneNorm: 1.0,
            confidenceInterval: (0.4, 0.6),
            walkOperatorCalls: 1023,
        )
        #expect(result.walkOperatorCalls == 1023, "Walk operator calls should be 1023")
    }
}

/// Test suite for QubitizationEvolutionResult properties.
/// Validates evolved state, time, epsilon, walk calls,
/// theoretical bound, and polynomial degree storage.
@Suite("QubitizationEvolutionResult Properties")
struct QubitizationEvolutionResultTests {
    @Test("Time is stored correctly")
    func timeStored() {
        let state = QuantumState(qubits: 2)
        let result = QubitizationEvolutionResult(
            evolvedState: state,
            time: 2.5,
            epsilon: 1e-6,
            walkOperatorCalls: 30,
            theoreticalBound: 35,
            polynomialDegree: 29,
        )
        #expect(abs(result.time - 2.5) < 1e-10, "Time should be 2.5")
    }

    @Test("Epsilon is stored correctly")
    func epsilonStored() {
        let state = QuantumState(qubits: 2)
        let result = QubitizationEvolutionResult(
            evolvedState: state,
            time: 1.0,
            epsilon: 1e-8,
            walkOperatorCalls: 40,
            theoreticalBound: 45,
            polynomialDegree: 39,
        )
        #expect(abs(result.epsilon - 1e-8) < 1e-15, "Epsilon should be 1e-8")
    }

    @Test("Walk operator calls is stored correctly")
    func walkOperatorCallsStored() {
        let state = QuantumState(qubits: 1)
        let result = QubitizationEvolutionResult(
            evolvedState: state,
            time: 1.0,
            epsilon: 1e-4,
            walkOperatorCalls: 15,
            theoreticalBound: 20,
            polynomialDegree: 14,
        )
        #expect(result.walkOperatorCalls == 15, "Walk operator calls should be 15")
    }

    @Test("Theoretical bound is stored correctly")
    func theoreticalBoundStored() {
        let state = QuantumState(qubits: 1)
        let result = QubitizationEvolutionResult(
            evolvedState: state,
            time: 1.0,
            epsilon: 1e-4,
            walkOperatorCalls: 15,
            theoreticalBound: 20,
            polynomialDegree: 14,
        )
        #expect(result.theoreticalBound == 20, "Theoretical bound should be 20")
    }

    @Test("Polynomial degree is stored correctly")
    func polynomialDegreeStored() {
        let state = QuantumState(qubits: 1)
        let result = QubitizationEvolutionResult(
            evolvedState: state,
            time: 1.0,
            epsilon: 1e-4,
            walkOperatorCalls: 15,
            theoreticalBound: 20,
            polynomialDegree: 14,
        )
        #expect(result.polynomialDegree == 14, "Polynomial degree should be 14")
    }

    @Test("Evolved state is stored correctly")
    func evolvedStateStored() {
        let state = QuantumState(qubits: 3)
        let result = QubitizationEvolutionResult(
            evolvedState: state,
            time: 1.0,
            epsilon: 1e-4,
            walkOperatorCalls: 10,
            theoreticalBound: 15,
            polynomialDegree: 9,
        )
        #expect(result.evolvedState.qubits == 3, "Evolved state should have 3 qubits")
    }
}

/// Test suite for Qubitization actor initialization.
/// Validates Hamiltonian storage, block encoding creation,
/// and walk operator construction during initialization.
@Suite("Qubitization Actor Initialization")
struct QubitizationInitializationTests {
    @Test("Qubitization stores Hamiltonian correctly")
    func hamiltonianStored() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 2)
        let storedTerms = await qubitization.hamiltonian.terms.count
        #expect(storedTerms == 2, "Hamiltonian should have 2 terms")
    }

    @Test("Qubitization creates block encoding")
    func blockEncodingCreated() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 2)
        let systemQubits = await qubitization.blockEncoding.systemQubits
        #expect(systemQubits == 2, "Block encoding should have 2 system qubits")
    }

    @Test("Qubitization creates walk operator")
    func walkOperatorCreated() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 2)
        let walkOpSystemQubits = await qubitization.walkOperator.systemQubits
        #expect(walkOpSystemQubits == [0, 1], "Walk operator should have correct system qubits")
    }

    @Test("Qubitization stores system qubits")
    func systemQubitsStored() async {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.z(0), .z(1), .z(2)))])
        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 3)
        let stored = await qubitization.systemQubits
        #expect(stored == 3, "System qubits should be 3")
    }
}

/// Test suite for Qubitization time evolution simulation.
/// Validates evolved state generation, complexity metrics,
/// and optimal query complexity achievement.
@Suite("Qubitization Time Evolution Simulation")
struct QubitizationSimulationTests {
    @Test("simulateEvolution returns valid result")
    func simulateEvolutionValid() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
        ])
        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 1)
        let initial = QuantumState(qubits: 1)
        let result = await qubitization.simulateEvolution(
            initialState: initial,
            time: 0.5,
            epsilon: 1e-3,
        )
        #expect(result.evolvedState.qubits >= 1, "Evolved state should have at least 1 qubit")
        #expect(result.walkOperatorCalls > 0, "Should have positive walk operator calls")
    }

    @Test("simulateEvolution stores time correctly")
    func simulateEvolutionStoresTime() async {
        let hamiltonian = Observable(terms: [(0.5, PauliString(.z(0)))])
        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 1)
        let initial = QuantumState(qubits: 1)
        let result = await qubitization.simulateEvolution(
            initialState: initial,
            time: 1.5,
            epsilon: 1e-3,
        )
        #expect(abs(result.time - 1.5) < 1e-10, "Time should be 1.5")
    }

    @Test("simulateEvolution stores epsilon correctly")
    func simulateEvolutionStoresEpsilon() async {
        let hamiltonian = Observable(terms: [(0.5, PauliString(.z(0)))])
        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 1)
        let initial = QuantumState(qubits: 1)
        let result = await qubitization.simulateEvolution(
            initialState: initial,
            time: 1.0,
            epsilon: 1e-5,
        )
        #expect(abs(result.epsilon - 1e-5) < 1e-12, "Epsilon should be 1e-5")
    }

    @Test("simulateEvolution computes theoretical bound")
    func simulateEvolutionTheoreticalBound() async {
        let hamiltonian = Observable(terms: [(0.5, PauliString(.z(0)))])
        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 1)
        let initial = QuantumState(qubits: 1)
        let result = await qubitization.simulateEvolution(
            initialState: initial,
            time: 1.0,
            epsilon: 1e-4,
        )
        #expect(result.theoreticalBound > 0, "Theoretical bound should be positive")
    }

    @Test("simulateEvolution walk calls within theoretical bound")
    func simulateEvolutionWithinBound() async {
        let hamiltonian = Observable(terms: [(0.5, PauliString(.z(0)))])
        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 1)
        let initial = QuantumState(qubits: 1)
        let result = await qubitization.simulateEvolution(
            initialState: initial,
            time: 1.0,
            epsilon: 1e-4,
        )
        #expect(result.walkOperatorCalls <= result.theoreticalBound + 5, "Walk calls should be near theoretical bound")
    }
}

/// Test suite for QSP phase conversion edge cases.
/// Validates Chebyshev to QSP phase conversion for single
/// phase polynomials and sparse coefficient arrays.
@Suite("QSP Phase Conversion Edge Cases")
struct QSPPhaseConversionEdgeCaseTests {
    @Test("Degree zero polynomial produces single phase")
    func degreeZeroSinglePhase() {
        let phases = QuantumSignalProcessing.computePhaseAngles(
            for: .chebyshev(degree: 0),
            degree: 10,
            epsilon: 1e-6,
        )
        #expect(phases.phases.count == 1, "Degree 0 should produce exactly 1 phase")
        #expect(phases.polynomialDegree == 0, "Polynomial degree should be 0")
    }

    @Test("Custom single coefficient produces single phase")
    func customSingleCoefficientSinglePhase() {
        let phases = QuantumSignalProcessing.computePhaseAngles(
            for: .custom(coefficients: [0.5]),
            degree: 10,
            epsilon: 1e-6,
        )
        #expect(phases.phases.count == 1, "Single coefficient should produce 1 phase")
        #expect(phases.polynomialDegree == 0, "Polynomial degree should be 0 for single coefficient")
    }

    @Test("Sparse coefficients with higher degree use zero padding")
    func sparseCoefficientsZeroPadding() {
        let phases = QuantumSignalProcessing.computePhaseAngles(
            for: .custom(coefficients: [0.1, 0.2]),
            degree: 10,
            epsilon: 1e-6,
        )
        #expect(phases.phases.count == 2, "Two coefficients should produce 2 phases")
        #expect(phases.polynomialDegree == 1, "Polynomial degree should be 1")
    }

    @Test("Higher degree Chebyshev with short coefficients uses zero")
    func higherDegreeChebyshevPadding() {
        let phases = QuantumSignalProcessing.computePhaseAngles(
            for: .chebyshev(degree: 5),
            degree: 10,
            epsilon: 1e-6,
        )
        #expect(phases.phases.count == 6, "Degree 5 Chebyshev should produce 6 phases")
    }

    @Test("Single zero coefficient produces single phase")
    func singleZeroCoefficient() {
        let phases = QuantumSignalProcessing.computePhaseAngles(
            for: .custom(coefficients: [0.0]),
            degree: 10,
            epsilon: 1e-6,
        )
        #expect(phases.polynomialDegree == 0, "Single zero coefficient should give degree 0")
        #expect(phases.phases.count == 1, "Should produce exactly 1 phase")
    }
}

/// Test suite for multi-controlled Z gate decomposition.
/// Validates reflection operator construction for walk
/// operators with varying numbers of ancilla qubits.
@Suite("Multi-Controlled Z Gate Decomposition")
struct MultiControlledZDecompositionTests {
    @Test("Walk circuit with many terms triggers multi-controlled Z")
    func walkCircuitManyTermsMultiControlledZ() {
        let hamiltonian = Observable(terms: [
            (0.1, PauliString(.z(0))),
            (0.1, PauliString(.x(0))),
            (0.1, PauliString(.y(0))),
            (0.1, PauliString(.z(1))),
            (0.1, PauliString(.x(1))),
            (0.1, PauliString(.y(1))),
            (0.1, PauliString(.z(0), .z(1))),
            (0.1, PauliString(.x(0), .x(1))),
        ])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 2)
        let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
        let circuit = walkOp.buildWalkCircuit()
        #expect(circuit.gates.count > 0, "Walk circuit should have gates for multi-term Hamiltonian")
        #expect(encoding.configuration.ancillaQubits >= 3, "Should have 3+ ancilla qubits for 8 terms")
    }

    @Test("Controlled walk with many ancillas uses Toffoli ladder")
    func controlledWalkManyAncillasToffoliLadder() {
        let hamiltonian = Observable(terms: [
            (0.1, PauliString(.z(0))),
            (0.1, PauliString(.x(0))),
            (0.1, PauliString(.y(0))),
            (0.1, PauliString(.z(1))),
            (0.1, PauliString(.x(1))),
            (0.1, PauliString(.y(1))),
            (0.1, PauliString(.z(0), .z(1))),
            (0.1, PauliString(.x(0), .x(1))),
        ])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 2)
        let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
        let circuit = walkOp.buildControlledWalkCircuit(controlQubit: 0)
        #expect(circuit.gates.count > 0, "Controlled walk should have gates")
    }

    @Test("Single ancilla qubit uses Z gate directly")
    func singleAncillaUsesZGate() {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
        ])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 1)
        let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
        let circuit = walkOp.buildWalkCircuit()
        #expect(circuit.gates.count > 0, "Walk circuit should have gates")
    }

    @Test("Two ancilla qubits uses CZ gate")
    func twoAncillasUsesCZGate() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (0.3, PauliString(.x(0))),
            (0.2, PauliString(.y(0))),
        ])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 1)
        let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
        let circuit = walkOp.buildWalkCircuit()
        #expect(circuit.gates.count > 0, "Walk circuit with 2 ancillas should have gates")
    }
}

/// Test suite for controlled gate decomposition edge cases.
/// Validates SWAP gate control, default 2-qubit handling,
/// and multi-qubit gate fallback decomposition.
@Suite("Controlled Gate Decomposition Edge Cases")
struct ControlledGateDecompositionEdgeCaseTests {
    @Test("Controlled walk circuit handles various gate types")
    func controlledWalkHandlesVariousGates() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 2)
        let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
        let circuit = walkOp.buildControlledWalkCircuit(controlQubit: 10)
        #expect(circuit.qubits >= 11, "Controlled circuit should include control qubit 10")
        #expect(circuit.gates.count > 0, "Controlled circuit should have gates")
    }

    @Test("Walk operator with explicit qubit assignment builds correctly")
    func explicitQubitAssignmentBuilds() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.z(0)))])
        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 1)
        let walkOp = QubitizedWalkOperator(
            blockEncoding: encoding,
            ancillaQubits: [3, 4, 5],
            systemQubits: [0, 1, 2],
        )
        let circuit = walkOp.buildWalkCircuit()
        #expect(circuit.gates.count > 0, "Walk circuit with explicit qubits should have gates")
    }
}

/// Test suite for state extension edge cases in Qubitization.
/// Validates state extension when state already has enough
/// qubits and when ancilla extension is needed.
@Suite("State Extension Edge Cases")
struct StateExtensionEdgeCaseTests {
    @Test("Simulation with already extended state preserves qubits")
    func simulationPreservesExtendedState() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
        ])
        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 1)
        let initial = QuantumState(qubits: 1)
        let result = await qubitization.simulateEvolution(
            initialState: initial,
            time: 0.1,
            epsilon: 1e-2,
        )
        #expect(result.evolvedState.qubits >= 1, "Evolved state should have at least 1 qubit")
    }

    @Test("Large initial state is handled correctly")
    func largeInitialStateHandled() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
        ])
        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 1)
        let initial = QuantumState(qubits: 1)
        let result = await qubitization.simulateEvolution(
            initialState: initial,
            time: 0.1,
            epsilon: 1e-2,
        )
        #expect(result.evolvedState.qubits == 1, "Should project back to system qubits")
    }

    @Test("State with sufficient qubits is not extended")
    func stateWithSufficientQubitsNotExtended() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
        ])
        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 1)
        let totalQubits = await qubitization.blockEncoding.totalQubits
        let largeState = QuantumState(qubits: totalQubits + 2)
        let result = await qubitization.simulateEvolution(
            initialState: largeState,
            time: 0.1,
            epsilon: 1e-2,
        )
        #expect(result.evolvedState.qubits >= 1, "Evolution should complete with large initial state")
    }
}

/// Test suite for chebyshevToQSPPhases edge cases.
/// Validates empty coefficient handling and coefficient
/// array padding when shorter than target degree.
@Suite("ChebyshevToQSPPhases Edge Cases")
struct ChebyshevToQSPPhasesEdgeCaseTests {
    @Test("Two coefficient array produces two phases")
    func twoCoefficientsProducesTwoPhases() {
        let phases = QuantumSignalProcessing.computePhaseAngles(
            for: .custom(coefficients: [0.5, 0.3]),
            degree: 5,
            epsilon: 1e-6,
        )
        #expect(phases.phases.count == 2, "Two coefficients should produce 2 phases")
        #expect(phases.polynomialDegree == 1, "Polynomial degree should be 1 for two coefficients")
    }

    @Test("Single coefficient produces single phase with acos transform")
    func singleCoefficientProducesSinglePhase() {
        let phases = QuantumSignalProcessing.computePhaseAngles(
            for: .custom(coefficients: [0.8]),
            degree: 4,
            epsilon: 1e-6,
        )
        #expect(phases.phases.count == 1, "Single coefficient should produce 1 phase")
        #expect(phases.polynomialDegree == 0, "Polynomial degree should be 0 for single coefficient")
        let expectedPhase = acos(0.8) / 2.0
        #expect(abs(phases.phases[0] - expectedPhase) < 1e-10, "Phase should be acos(coeff)/2")
    }

    @Test("Three or more coefficients uses full computation path")
    func threeOrMoreCoefficientsFullPath() {
        let phases = QuantumSignalProcessing.computePhaseAngles(
            for: .custom(coefficients: [0.1, 0.2, 0.3, 0.4]),
            degree: 10,
            epsilon: 1e-6,
        )
        #expect(phases.phases.count == 4, "Four coefficients should produce 4 phases")
        #expect(phases.polynomialDegree == 3, "Polynomial degree should be 3 for four coefficients")
    }
}

/// Test suite for Qubitization eigenvalue estimation.
/// Validates eigenvalue estimation via quantum phase
/// estimation on qubitized walk operator eigenstates.
@Suite("Qubitization Eigenvalue Estimation")
struct QubitizationEigenvalueEstimationTests {
    @Test("Eigenvalue estimation returns valid result")
    func eigenvalueEstimationReturnsValidResult() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
        ])
        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 1)
        let precisionBits = 4
        let blockEncodingTotalQubits = await qubitization.blockEncoding.totalQubits
        let requiredQubits = max(2 * precisionBits, precisionBits + blockEncodingTotalQubits)
        let eigenstate = QuantumState(qubits: requiredQubits)
        let result = await qubitization.estimateEigenvalue(
            eigenstate: eigenstate,
            precisionBits: precisionBits,
        )
        #expect(abs(result.oneNorm - 0.5) < 1e-10, "One-norm should be 0.5")
        #expect(result.walkOperatorCalls == 15, "Walk operator calls should be 2^4 - 1 = 15")
    }

    @Test("Eigenvalue estimation computes confidence interval")
    func eigenvalueEstimationComputesConfidenceInterval() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
        ])
        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 1)
        let precisionBits = 6
        let blockEncodingTotalQubits = await qubitization.blockEncoding.totalQubits
        let requiredQubits = max(2 * precisionBits, precisionBits + blockEncodingTotalQubits)
        let eigenstate = QuantumState(qubits: requiredQubits)
        let result = await qubitization.estimateEigenvalue(
            eigenstate: eigenstate,
            precisionBits: precisionBits,
        )
        #expect(result.confidenceInterval.lower <= result.eigenvalue, "Lower bound should be at most eigenvalue")
        #expect(result.confidenceInterval.upper >= result.eigenvalue, "Upper bound should be at least eigenvalue")
    }

    @Test("Eigenvalue estimation stores phase correctly")
    func eigenvalueEstimationStoresPhase() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
        ])
        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 1)
        let precisionBits = 5
        let blockEncodingTotalQubits = await qubitization.blockEncoding.totalQubits
        let requiredQubits = max(2 * precisionBits, precisionBits + blockEncodingTotalQubits)
        let eigenstate = QuantumState(qubits: requiredQubits)
        let result = await qubitization.estimateEigenvalue(
            eigenstate: eigenstate,
            precisionBits: precisionBits,
        )
        #expect(result.phase >= 0.0 || result.phase < 0.0, "Phase should be a valid number")
        #expect(result.walkOperatorCalls == 31, "Walk operator calls should be 2^5 - 1 = 31")
    }

    @Test("Eigenvalue estimation with oversized eigenstate skips extension")
    func eigenvalueEstimationWithOversizedEigenstate() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
        ])
        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 1)
        let precisionBits = 3
        let blockEncodingTotalQubits = await qubitization.blockEncoding.totalQubits
        let requiredQubits = max(2 * precisionBits, precisionBits + blockEncodingTotalQubits)
        let oversizedEigenstate = QuantumState(qubits: requiredQubits + 3)
        let result = await qubitization.estimateEigenvalue(
            eigenstate: oversizedEigenstate,
            precisionBits: precisionBits,
        )
        #expect(result.walkOperatorCalls == 7, "Walk operator calls should be 2^3 - 1 = 7")
        #expect(abs(result.oneNorm - 0.5) < 1e-10, "One-norm should be preserved")
    }
}

/// Validates that eigenvalue estimation correctly handles
/// 2-qubit CNOT and CZ gates in the controlled walk operator.
/// These arise from multi-qubit Pauli terms in the Hamiltonian.
@Suite("Eigenvalue Estimation Gate Decomposition")
struct EigenvalueEstimationGateDecompositionTests {
    @Test("Eigenvalue estimation handles XX Hamiltonian with 2-qubit gates")
    func eigenvalueEstimationXXHamiltonian() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.x(0), .x(1))),
        ])
        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 2)
        let eigenstate = await QuantumState(qubits: 2 + qubitization.blockEncoding.configuration.ancillaQubits + 2)
        let result = await qubitization.estimateEigenvalue(
            eigenstate: eigenstate,
            precisionBits: 2,
        )
        #expect(result.walkOperatorCalls > 0, "Should complete eigenvalue estimation with XX Hamiltonian")
    }

    @Test("Eigenvalue estimation handles ZX Hamiltonian with mixed gates")
    func eigenvalueEstimationZXHamiltonian() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0), .x(1))),
        ])
        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 2)
        let eigenstate = await QuantumState(qubits: 2 + qubitization.blockEncoding.configuration.ancillaQubits + 2)
        let result = await qubitization.estimateEigenvalue(
            eigenstate: eigenstate,
            precisionBits: 2,
        )
        #expect(result.walkOperatorCalls > 0, "Should complete eigenvalue estimation with ZX Hamiltonian")
    }

    @Test("Eigenvalue estimation handles YY Hamiltonian")
    func eigenvalueEstimationYYHamiltonian() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.y(0), .y(1))),
        ])
        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 2)
        let eigenstate = await QuantumState(qubits: 2 + qubitization.blockEncoding.configuration.ancillaQubits + 2)
        let result = await qubitization.estimateEigenvalue(
            eigenstate: eigenstate,
            precisionBits: 2,
        )
        #expect(result.walkOperatorCalls > 0, "Should complete eigenvalue estimation with YY Hamiltonian")
    }
}
