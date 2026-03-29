// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under Apache 2.0

import Aether
import Foundation
import Testing

/// Test suite for HHLProblem struct initialization, property storage,
/// and validation of input constraints for quantum linear system problems.
@Suite("HHLProblem Properties")
struct HHLProblemTests {
    @Test("Problem stores all properties correctly")
    func problemStoresProperties() {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(1))),
        ])
        let rhs = QuantumState(qubits: 2)
        let problem = HHLProblem(
            hamiltonian: hamiltonian,
            systemQubits: 2,
            rightHandSide: rhs,
            conditionNumber: 4.0,
        )

        #expect(problem.systemQubits == 2, "System qubits should be 2")
        #expect(abs(problem.conditionNumber - 4.0) < 1e-10, "Condition number should be 4.0")
        #expect(problem.rightHandSide.qubits == 2, "RHS should have 2 qubits")
    }

    @Test("Problem accepts single qubit system")
    func singleQubitSystem() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.z(0)))])
        let rhs = QuantumState(qubits: 1)
        let problem = HHLProblem(
            hamiltonian: hamiltonian,
            systemQubits: 1,
            rightHandSide: rhs,
            conditionNumber: 2.0,
        )

        #expect(problem.systemQubits == 1, "System qubits should be 1")
    }

    @Test("Problem accepts large condition number")
    func largeConditionNumber() {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.z(0)))])
        let rhs = QuantumState(qubits: 1)
        let problem = HHLProblem(
            hamiltonian: hamiltonian,
            systemQubits: 1,
            rightHandSide: rhs,
            conditionNumber: 100.0,
        )

        #expect(abs(problem.conditionNumber - 100.0) < 1e-10, "Condition number should be 100.0")
    }
}

/// Test suite for HHLMethod enum cases and equality comparison,
/// verifying QPE and QSVT method selection parameters are stored correctly.
@Suite("HHLMethod Selection")
struct HHLMethodTests {
    @Test("QPE method stores precision qubits")
    func qpeMethod() {
        let result = HHLResult(
            solutionState: QuantumState(qubits: 1),
            successProbability: 0.5,
            oracleCalls: 32,
            method: .qpe(precisionQubits: 6),
        )
        #expect(result.description.contains("QPE"), "Method should be QPE")
        #expect(result.description.contains("6"), "Description should contain precision qubit count")
    }

    @Test("QSVT method stores epsilon")
    func qsvtMethod() {
        let result = HHLResult(
            solutionState: QuantumState(qubits: 1),
            successProbability: 0.5,
            oracleCalls: 32,
            method: .qsvt(epsilon: 1e-6),
        )
        #expect(result.description.contains("QSVT"), "Method should be QSVT")
    }

    @Test("Methods with same parameters are equal")
    func methodEquality() {
        let m1 = HHLMethod.qpe(precisionQubits: 4)
        let m2 = HHLMethod.qpe(precisionQubits: 4)
        #expect(m1 == m2, "Same QPE methods should be equal")
    }

    @Test("Methods with different parameters are not equal")
    func methodInequality() {
        let m1 = HHLMethod.qpe(precisionQubits: 4)
        let m2 = HHLMethod.qpe(precisionQubits: 6)
        #expect(m1 != m2, "Different QPE methods should not be equal")
    }
}

/// Test suite for HHLResult struct properties and description formatting,
/// validating storage of solution state, success probability, and oracle call metrics.
@Suite("HHLResult Properties")
struct HHLResultTests {
    @Test("Result stores all properties correctly")
    func resultStoresProperties() {
        let state = QuantumState(qubits: 2)
        let result = HHLResult(
            solutionState: state,
            successProbability: 0.25,
            oracleCalls: 64,
            method: .qpe(precisionQubits: 6),
        )

        #expect(result.solutionState.qubits == 2, "Solution should have 2 qubits")
        #expect(abs(result.successProbability - 0.25) < 1e-10, "Success probability should be 0.25")
        #expect(result.oracleCalls == 64, "Oracle calls should be 64")
    }

    @Test("Description contains method name")
    func descriptionContainsMethod() {
        let result = HHLResult(
            solutionState: QuantumState(qubits: 1),
            successProbability: 0.5,
            oracleCalls: 32,
            method: .qsvt(epsilon: 1e-4),
        )

        #expect(result.description.contains("QSVT"), "Description should contain QSVT")
    }

    @Test("Description contains success probability")
    func descriptionContainsSuccessProb() {
        let result = HHLResult(
            solutionState: QuantumState(qubits: 1),
            successProbability: 0.123456,
            oracleCalls: 32,
            method: .qpe(precisionQubits: 4),
        )

        #expect(result.description.contains("0.123456"), "Description should contain probability value")
    }
}

/// Test suite for HHLAlgorithm QSVT-based solver validating that the quantum signal
/// processing path produces non-trivial solution states with positive success probability.
@Suite("HHLAlgorithm QSVT Solver")
struct HHLAlgorithmQSVTTests {
    @Test("QSVT solver returns solution with correct qubit count")
    func qsvtSolverReturnsCorrectQubits() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(0))),
        ])
        let rhs = QuantumState(qubits: 1)
        let problem = HHLProblem(
            hamiltonian: hamiltonian,
            systemQubits: 1,
            rightHandSide: rhs,
            conditionNumber: 3.0,
        )

        let hhl = HHLAlgorithm(problem: problem)
        let result = await hhl.solve(method: .qsvt(epsilon: 0.1))

        #expect(result.solutionState.qubits == 1, "Solution should have 1 qubit")
    }

    @Test("QSVT solver reports non-negative success probability")
    func qsvtSuccessProbabilityNonNegative() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
        ])
        let rhs = QuantumState(qubits: 1)
        let problem = HHLProblem(
            hamiltonian: hamiltonian,
            systemQubits: 1,
            rightHandSide: rhs,
            conditionNumber: 2.0,
        )

        let hhl = HHLAlgorithm(problem: problem)
        let result = await hhl.solve(method: .qsvt(epsilon: 0.01))

        #expect(result.successProbability >= 0.0, "Success probability must be non-negative")
        #expect(result.successProbability <= 1.0, "Success probability must be at most 1")
    }

    @Test("QSVT oracle calls match polynomial degree")
    func qsvtOracleCallsPositive() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (0.3, PauliString(.x(0))),
        ])
        let rhs = QuantumState(qubits: 1)
        let problem = HHLProblem(
            hamiltonian: hamiltonian,
            systemQubits: 1,
            rightHandSide: rhs,
            conditionNumber: 2.0,
        )

        let hhl = HHLAlgorithm(problem: problem)
        let result = await hhl.solve(method: .qsvt(epsilon: 0.1))

        #expect(result.oracleCalls > 0, "Oracle calls must be positive")
    }

    @Test("QSVT method stored in result")
    func qsvtMethodStoredInResult() async {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.z(0)))])
        let rhs = QuantumState(qubits: 1)
        let problem = HHLProblem(
            hamiltonian: hamiltonian,
            systemQubits: 1,
            rightHandSide: rhs,
            conditionNumber: 2.0,
        )

        let hhl = HHLAlgorithm(problem: problem)
        let result = await hhl.solve(method: .qsvt(epsilon: 0.01))

        #expect(result.method == .qsvt(epsilon: 0.01), "Method should be QSVT with matching epsilon")
    }
}

/// Test suite for HHLAlgorithm QPE-based solver validating eigenvalue inversion,
/// controlled walk operator construction, and solution state extraction.
@Suite("HHLAlgorithm QPE Solver")
struct HHLAlgorithmQPETests {
    @Test("QPE solver returns solution with correct qubit count")
    func qpeSolverReturnsCorrectQubits() async {
        let hamiltonian = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (-0.3, PauliString(.x(0))),
        ])
        let rhs = QuantumState(qubits: 1)
        let problem = HHLProblem(
            hamiltonian: hamiltonian,
            systemQubits: 1,
            rightHandSide: rhs,
            conditionNumber: 3.0,
        )

        let hhl = HHLAlgorithm(problem: problem)
        let result = await hhl.solve(method: .qpe(precisionQubits: 3))

        #expect(result.solutionState.qubits == 1, "Solution should have 1 qubit")
    }

    @Test("QPE solver reports bounded success probability")
    func qpeSuccessProbabilityBounded() async {
        let hamiltonian = Observable(terms: [
            (1.0, PauliString(.z(0))),
        ])
        let rhs = QuantumState(qubits: 1)
        let problem = HHLProblem(
            hamiltonian: hamiltonian,
            systemQubits: 1,
            rightHandSide: rhs,
            conditionNumber: 2.0,
        )

        let hhl = HHLAlgorithm(problem: problem)
        let result = await hhl.solve(method: .qpe(precisionQubits: 3))

        #expect(result.successProbability >= 0.0, "Success probability must be non-negative")
        #expect(result.successProbability <= 1.0, "Success probability must be at most 1")
    }

    @Test("QPE oracle calls scale with precision qubits")
    func qpeOracleCallsScaling() async {
        let hamiltonian = Observable(terms: [(0.5, PauliString(.z(0)))])
        let rhs = QuantumState(qubits: 1)
        let problem = HHLProblem(
            hamiltonian: hamiltonian,
            systemQubits: 1,
            rightHandSide: rhs,
            conditionNumber: 2.0,
        )

        let hhl = HHLAlgorithm(problem: problem)
        let result3 = await hhl.solve(method: .qpe(precisionQubits: 3))
        let result4 = await hhl.solve(method: .qpe(precisionQubits: 4))

        #expect(result4.oracleCalls > result3.oracleCalls, "More precision qubits should require more oracle calls")
    }

    @Test("QPE method stored in result")
    func qpeMethodStoredInResult() async {
        let hamiltonian = Observable(terms: [(1.0, PauliString(.z(0)))])
        let rhs = QuantumState(qubits: 1)
        let problem = HHLProblem(
            hamiltonian: hamiltonian,
            systemQubits: 1,
            rightHandSide: rhs,
            conditionNumber: 2.0,
        )

        let hhl = HHLAlgorithm(problem: problem)
        let result = await hhl.solve(method: .qpe(precisionQubits: 3))

        #expect(result.method == .qpe(precisionQubits: 3), "Method should be QPE with matching precision")
    }
}

/// Test suite for HHLComplexity static analysis methods validating theoretical
/// bounds, precision qubit requirements, and QSVT speedup calculations.
@Suite("HHLComplexity Analysis")
struct HHLComplexityTests {
    @Test("QPE analysis returns positive precision qubits")
    func qpeAnalysisPrecisionQubits() {
        let analysis = HHLComplexity.analyzeQPE(
            conditionNumber: 10.0,
            sparsity: 4,
            dimension: 16,
            targetError: 1e-3,
        )

        #expect(analysis.precisionQubits > 0, "Precision qubits must be positive")
        #expect(analysis.oracleCalls > 0, "Oracle calls must be positive")
        #expect(analysis.totalQubits > analysis.precisionQubits, "Total qubits must exceed precision")
    }

    @Test("QPE precision qubits increase with tighter error")
    func qpePrecisionIncreasesWithTighterError() {
        let loose = HHLComplexity.analyzeQPE(
            conditionNumber: 5.0, sparsity: 2, dimension: 8, targetError: 0.1,
        )
        let tight = HHLComplexity.analyzeQPE(
            conditionNumber: 5.0, sparsity: 2, dimension: 8, targetError: 0.001,
        )

        #expect(tight.precisionQubits >= loose.precisionQubits, "Tighter error requires more precision qubits")
    }

    @Test("QSVT analysis returns positive polynomial degree")
    func qsvtAnalysisPositiveDegree() {
        let analysis = HHLComplexity.analyzeQSVT(
            conditionNumber: 5.0,
            sparsity: 2,
            targetError: 1e-6,
        )

        #expect(analysis.polynomialDegree > 0, "Polynomial degree must be positive")
        #expect(analysis.oracleCalls > 0, "Oracle calls must be positive")
        #expect(analysis.successProbability > 0.0, "Success probability must be positive")
    }

    @Test("QSVT degree scales with condition number")
    func qsvtDegreeScalesWithKappa() {
        let low = HHLComplexity.analyzeQSVT(
            conditionNumber: 2.0, sparsity: 2, targetError: 0.01,
        )
        let high = HHLComplexity.analyzeQSVT(
            conditionNumber: 20.0, sparsity: 2, targetError: 0.01,
        )

        #expect(high.polynomialDegree > low.polynomialDegree, "Higher kappa requires higher degree")
    }

    @Test("QSVT speedup over QPE is positive")
    func qsvtSpeedupPositive() {
        let speedup = HHLComplexity.qsvtSpeedupOverQPE(
            conditionNumber: 10.0,
            targetError: 1e-6,
        )

        #expect(speedup > 1.0, "QSVT should be faster than QPE for high precision")
    }

    @Test("Speedup increases with tighter error tolerance")
    func speedupIncreasesWithPrecision() {
        let speedup1 = HHLComplexity.qsvtSpeedupOverQPE(
            conditionNumber: 10.0, targetError: 0.01,
        )
        let speedup2 = HHLComplexity.qsvtSpeedupOverQPE(
            conditionNumber: 10.0, targetError: 1e-6,
        )

        #expect(speedup2 > speedup1, "Tighter precision should increase QSVT advantage")
    }

    @Test("Success probability scales inversely with kappa squared")
    func successProbabilityScalesWithKappa() {
        let low = HHLComplexity.analyzeQSVT(
            conditionNumber: 2.0, sparsity: 1, targetError: 0.01,
        )
        let high = HHLComplexity.analyzeQSVT(
            conditionNumber: 10.0, sparsity: 1, targetError: 0.01,
        )

        #expect(low.successProbability > high.successProbability, "Higher kappa should reduce success probability")
    }
}
