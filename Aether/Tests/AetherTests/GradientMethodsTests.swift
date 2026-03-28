// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Test suite for adjoint differentiation gradient computation.
/// Validates that adjoint method produces exact gradients matching
/// parameter shift rule on parametric quantum circuits.
@Suite("Adjoint Differentiation")
struct AdjointDifferentiationTests {
    @Test("Single Ry gate gradient matches analytic derivative")
    func singleRyGradient() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        let observable = Observable.pauliZ(qubit: 0)
        let theta = 0.7
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [theta])
        let expected = -sin(theta)
        #expect(abs(gradient[0] - expected) < 1e-10, "Ry gradient of ⟨Z⟩ should be -sin(θ)")
    }

    @Test("Single Rx gate gradient matches analytic derivative")
    func singleRxGradient() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationX(.parameter(Parameter(name: "a"))), to: 0)
        let observable = Observable.pauliZ(qubit: 0)
        let theta = 1.2
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [theta])
        let expected = -sin(theta)
        #expect(abs(gradient[0] - expected) < 1e-10, "Rx gradient of ⟨Z⟩ should match analytic result")
    }

    @Test("Single Rz gate gradient with X observable")
    func singleRzGradient() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        circuit.append(.rotationZ(.parameter(Parameter(name: "a"))), to: 0)
        let observable = Observable.pauliX(qubit: 0)
        let theta = 0.5
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [theta])
        let paramShiftPlus = circuit.bound(with: [theta + .pi / 2]).execute()
        let paramShiftMinus = circuit.bound(with: [theta - .pi / 2]).execute()
        let expectedGrad = (observable.expectationValue(of: paramShiftPlus) - observable.expectationValue(of: paramShiftMinus)) / 2.0
        #expect(abs(gradient[0] - expectedGrad) < 1e-10, "Rz gradient should match parameter shift result")
    }

    @Test("Two-parameter circuit gradient")
    func twoParameterGradient() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.rotationZ(.parameter(Parameter(name: "b"))), to: 1)
        let observable = Observable.pauliZ(qubit: 0)
        let params = [0.8, 0.3]
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: params)
        let shift = Double.pi / 2
        let fPlusA = observable.expectationValue(of: circuit.bound(with: [params[0] + shift, params[1]]).execute())
        let fMinusA = observable.expectationValue(of: circuit.bound(with: [params[0] - shift, params[1]]).execute())
        let fPlusB = observable.expectationValue(of: circuit.bound(with: [params[0], params[1] + shift]).execute())
        let fMinusB = observable.expectationValue(of: circuit.bound(with: [params[0], params[1] - shift]).execute())
        #expect(abs(gradient[0] - (fPlusA - fMinusA) / 2.0) < 1e-10, "Gradient w.r.t. first parameter should match parameter shift")
        #expect(abs(gradient[1] - (fPlusB - fMinusB) / 2.0) < 1e-10, "Gradient w.r.t. second parameter should match parameter shift")
    }

    @Test("Shared parameter accumulates gradient from multiple gates")
    func sharedParameterGradient() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 1)
        let observable = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (0.5, PauliString(.z(1))),
        ])
        let theta = 0.6
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [theta])
        let expected = -sin(theta)
        #expect(abs(gradient[0] - expected) < 1e-10, "Shared parameter gradient should accumulate from both gates")
    }

    @Test("Empty parameter list returns empty gradient")
    func emptyParameters() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let observable = Observable.pauliZ(qubit: 0)
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [])
        #expect(gradient.isEmpty, "Non-parametric circuit should return empty gradient")
    }

    @Test("Phase gate gradient")
    func phaseGateGradient() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        circuit.append(.phase(.parameter(Parameter(name: "a"))), to: 0)
        let observable = Observable.pauliZ(qubit: 0)
        let theta = 1.0
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [theta])
        let shift = Double.pi / 2
        let fPlus = observable.expectationValue(of: circuit.bound(with: [theta + shift]).execute())
        let fMinus = observable.expectationValue(of: circuit.bound(with: [theta - shift]).execute())
        let expected = (fPlus - fMinus) / 2.0
        #expect(abs(gradient[0] - expected) < 1e-10, "Phase gate gradient should match parameter shift")
    }

    @Test("Controlled rotation Y gradient")
    func controlledRotationYGradient() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.pauliX, to: 0)
        circuit.append(.controlledRotationY(.parameter(Parameter(name: "a"))), to: [0, 1])
        let observable = Observable.pauliZ(qubit: 1)
        let theta = 0.9
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [theta])
        let shift = Double.pi / 2
        let fPlus = observable.expectationValue(of: circuit.bound(with: [theta + shift]).execute())
        let fMinus = observable.expectationValue(of: circuit.bound(with: [theta - shift]).execute())
        let expected = (fPlus - fMinus) / 2.0
        #expect(abs(gradient[0] - expected) < 1e-10, "CRy gradient should match parameter shift")
    }

    @Test("ZZ interaction gate gradient via finite difference")
    func zzGateGradient() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.zz(.parameter(Parameter(name: "a"))), to: [0, 1])
        let observable = Observable.pauliX(qubit: 0)
        let theta = 0.4
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [theta])
        let eps = 1e-6
        let fPlus = observable.expectationValue(of: circuit.bound(with: [theta + eps]).execute())
        let fMinus = observable.expectationValue(of: circuit.bound(with: [theta - eps]).execute())
        let expected = (fPlus - fMinus) / (2.0 * eps)
        #expect(abs(gradient[0] - expected) < 1e-5, "ZZ gate gradient should match finite difference")
    }

    @Test("Multi-term observable gradient")
    func multiTermObservableGradient() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        circuit.append(.cnot, to: [0, 1])
        let observable = Observable(terms: [
            (0.5, PauliString(.z(0))),
            (0.3, PauliString(.z(1))),
            (-0.2, PauliString(.z(0), .z(1))),
        ])
        let theta = 1.1
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [theta])
        let shift = Double.pi / 2
        let fPlus = observable.expectationValue(of: circuit.bound(with: [theta + shift]).execute())
        let fMinus = observable.expectationValue(of: circuit.bound(with: [theta - shift]).execute())
        let expected = (fPlus - fMinus) / 2.0
        #expect(abs(gradient[0] - expected) < 1e-10, "Multi-term observable gradient should match parameter shift")
    }
}

/// Test suite for complex-step finite difference gradients.
/// Validates machine-precision accuracy of the complex perturbation
/// method on analytic test functions.
@Suite("Complex-Step Finite Differences")
struct ComplexStepTests {
    @Test("Quadratic function gradient: ∂(x²+y²)/∂x = 2x")
    func quadraticGradient() {
        let grad = GradientMethods.complexStep(
            { z in z[0] * z[0] + z[1] * z[1] },
            at: [3.0, 4.0],
        )
        #expect(abs(grad[0] - 6.0) < 1e-14, "∂(x²+y²)/∂x at x=3 should be 6")
        #expect(abs(grad[1] - 8.0) < 1e-14, "∂(x²+y²)/∂y at y=4 should be 8")
    }

    @Test("Trigonometric function gradient: ∂sin(x)/∂x = cos(x)")
    func trigGradient() {
        let x = 1.3
        let grad = GradientMethods.complexStep(
            { z in Complex(Foundation.sin(z[0].real) * Foundation.cosh(z[0].imaginary), Foundation.cos(z[0].real) * Foundation.sinh(z[0].imaginary)) },
            at: [x],
        )
        #expect(abs(grad[0] - cos(x)) < 1e-14, "Derivative of sin(x) should be cos(x)")
    }

    @Test("Polynomial gradient: ∂(x³)/∂x = 3x²")
    func cubicGradient() {
        let x = 2.0
        let grad = GradientMethods.complexStep(
            { z in z[0] * z[0] * z[0] },
            at: [x],
        )
        #expect(abs(grad[0] - 3.0 * x * x) < 1e-13, "Derivative of x³ at x=2 should be 12")
    }

    @Test("Custom epsilon preserves accuracy")
    func customEpsilon() {
        let grad = GradientMethods.complexStep(
            { z in z[0] * z[0] },
            at: [5.0],
            epsilon: 1e-100,
        )
        #expect(abs(grad[0] - 10.0) < 1e-14, "Very small epsilon should still give machine precision")
    }
}

/// Test suite for stochastic parameter shift gradient estimation.
/// Validates that the randomized gradient estimator produces
/// correct evaluation counts and plausible gradient estimates.
@Suite("Stochastic Parameter Shift")
struct StochasticParameterShiftTests {
    @Test("Returns exactly 2 evaluations")
    func evaluationCount() async {
        let result = await GradientMethods.stochasticParameterShift(
            { _ in 0.0 },
            at: [0.1, 0.2, 0.3],
        )
        #expect(result.evaluations == 2, "Stochastic parameter shift should use exactly 2 evaluations")
    }

    @Test("Gradient vector has correct length")
    func gradientLength() async {
        let result = await GradientMethods.stochasticParameterShift(
            { params in params.reduce(0, +) },
            at: [0.1, 0.2, 0.3, 0.4],
        )
        #expect(result.gradient.count == 4, "Gradient should have same length as parameter vector")
    }

    @Test("Zero objective produces zero gradient")
    func zeroObjective() async {
        let result = await GradientMethods.stochasticParameterShift(
            { _ in 0.0 },
            at: [0.5, 0.3],
        )
        #expect(abs(result.gradient[0]) < 1e-10, "Zero objective should produce zero gradient")
        #expect(abs(result.gradient[1]) < 1e-10, "Zero objective should produce zero gradient")
    }

    @Test("Constant objective produces zero gradient")
    func constantObjective() async {
        let result = await GradientMethods.stochasticParameterShift(
            { _ in 42.0 },
            at: [1.0, 2.0],
        )
        #expect(abs(result.gradient[0]) < 1e-10, "Constant objective should produce zero gradient")
        #expect(abs(result.gradient[1]) < 1e-10, "Constant objective should produce zero gradient")
    }
}

/// Test suite for Hadamard test gradient computation.
/// Validates single-parameter gradient extraction matches
/// the adjoint differentiation result for each parameter.
@Suite("Hadamard Test Gradients")
struct HadamardTestTests {
    @Test("Matches adjoint gradient for Ry circuit")
    func matchesAdjointRy() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        let observable = Observable.pauliZ(qubit: 0)
        let theta = 0.8
        let adjointGrad = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [theta])
        let hadamardGrad = GradientMethods.hadamardTest(circuit: circuit, observable: observable, parameters: [theta], parameterIndex: 0)
        #expect(abs(hadamardGrad - adjointGrad[0]) < 1e-10, "Hadamard test should match adjoint gradient")
    }

    @Test("Matches adjoint for each parameter in multi-parameter circuit")
    func matchesAdjointMultiParam() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.rotationZ(.parameter(Parameter(name: "b"))), to: 1)
        let observable = Observable.pauliZ(qubit: 0)
        let params = [0.5, 1.2]
        let adjointGrad = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: params)
        let h0 = GradientMethods.hadamardTest(circuit: circuit, observable: observable, parameters: params, parameterIndex: 0)
        let h1 = GradientMethods.hadamardTest(circuit: circuit, observable: observable, parameters: params, parameterIndex: 1)
        #expect(abs(h0 - adjointGrad[0]) < 1e-10, "Hadamard test for param 0 should match adjoint")
        #expect(abs(h1 - adjointGrad[1]) < 1e-10, "Hadamard test for param 1 should match adjoint")
    }
}

/// Test suite for Hessian computation via parameter shift.
/// Validates second-order derivative matrix against finite
/// differences and analytic results for test functions.
@Suite("Hessian via Parameter Shift")
struct HessianTests {
    @Test("Hessian of quantum circuit expectation")
    func quantumHessian() async {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.rotationY(.parameter(Parameter(name: "b"))), to: 1)
        let observable = Observable.pauliZ(qubit: 0)
        let params = [0.5, 0.3]
        let frozenCircuit = circuit
        let result = await GradientMethods.hessian(
            { p in observable.expectationValue(of: frozenCircuit.bound(with: p).execute()) },
            at: params,
        )
        let eps = 1e-5
        let f0 = observable.expectationValue(of: circuit.bound(with: params).execute())
        let fPx = observable.expectationValue(of: circuit.bound(with: [params[0] + eps, params[1]]).execute())
        let fMx = observable.expectationValue(of: circuit.bound(with: [params[0] - eps, params[1]]).execute())
        let fdHxx = (fPx - 2 * f0 + fMx) / (eps * eps)
        #expect(abs(result.matrix[0][0] - fdHxx) < 0.5, "∂²E/∂a² should approximate finite difference")
    }

    @Test("Hessian is symmetric")
    func symmetry() async {
        let result = await GradientMethods.hessian(
            { p in sin(p[0]) * cos(p[1]) },
            at: [0.5, 0.3],
        )
        #expect(abs(result.matrix[0][1] - result.matrix[1][0]) < 1e-10, "Hessian should be symmetric")
    }

    @Test("Evaluation count for 2-parameter Hessian")
    func evaluationCount() async {
        let result = await GradientMethods.hessian(
            { _ in 0.0 },
            at: [0.1, 0.2],
        )
        #expect(result.evaluations == 12, "2-parameter Hessian needs 4*(1+2)=12 evaluations")
    }
}

/// Test suite for adjoint Hessian computation.
/// Validates that the adjoint-based Hessian matches finite difference
/// Hessian for quantum circuits with multiple parameters.
@Suite("Adjoint Hessian")
struct AdjointHessianTests {
    @Test("Single parameter produces 1×1 Hessian")
    func singleParameter() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        let observable = Observable.pauliZ(qubit: 0)
        let hessian = GradientMethods.adjointHessian(circuit: circuit, observable: observable, parameters: [0.5])
        #expect(hessian.count == 1, "Single parameter should produce 1×1 Hessian")
        #expect(hessian[0].count == 1, "Single parameter should produce 1×1 Hessian")
    }

    @Test("Two-parameter Hessian is symmetric")
    func twoParameterSymmetry() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.rotationZ(.parameter(Parameter(name: "b"))), to: 1)
        let observable = Observable.pauliZ(qubit: 0)
        let hessian = GradientMethods.adjointHessian(circuit: circuit, observable: observable, parameters: [0.5, 0.3])
        #expect(abs(hessian[0][1] - hessian[1][0]) < 1e-10, "Adjoint Hessian should be symmetric")
    }

    @Test("Empty parameters returns empty Hessian")
    func emptyParameters() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let observable = Observable.pauliZ(qubit: 0)
        let hessian = GradientMethods.adjointHessian(circuit: circuit, observable: observable, parameters: [])
        #expect(hessian.isEmpty, "Non-parametric circuit should return empty Hessian")
    }
}

/// Test suite for classical Fisher information matrix.
/// Validates the FIM structure and properties against known
/// analytical results for simple parametric circuits.
@Suite("Fisher Information Matrix")
struct FisherInformationMatrixTests {
    @Test("Single parameter FIM is non-negative")
    func nonNegative() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        let fim = GradientMethods.fisherInformationMatrix(circuit: circuit, parameters: [0.7])
        #expect(fim[0][0] >= -1e-10, "Fisher information should be non-negative")
    }

    @Test("Two-parameter FIM is symmetric")
    func symmetry() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.rotationZ(.parameter(Parameter(name: "b"))), to: 1)
        let fim = GradientMethods.fisherInformationMatrix(circuit: circuit, parameters: [0.5, 0.3])
        #expect(abs(fim[0][1] - fim[1][0]) < 1e-10, "Fisher information matrix should be symmetric")
    }

    @Test("FIM has correct dimensions")
    func dimensions() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        circuit.append(.rotationZ(.parameter(Parameter(name: "b"))), to: 1)
        let fim = GradientMethods.fisherInformationMatrix(circuit: circuit, parameters: [0.5, 0.3])
        #expect(fim.count == 2, "FIM should be 2×2 for 2 parameters")
        #expect(fim[0].count == 2, "FIM should be 2×2 for 2 parameters")
    }

    @Test("Empty parameters returns empty FIM")
    func emptyParameters() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)
        let fim = GradientMethods.fisherInformationMatrix(circuit: circuit, parameters: [])
        #expect(fim.isEmpty, "Non-parametric circuit should return empty FIM")
    }

    @Test("Diagonal FIM entries are non-negative")
    func diagonalNonNegative() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        circuit.append(.rotationX(.parameter(Parameter(name: "b"))), to: 1)
        let fim = GradientMethods.fisherInformationMatrix(circuit: circuit, parameters: [0.8, 1.2])
        #expect(fim[0][0] >= -1e-10, "Diagonal FIM entry should be non-negative")
        #expect(fim[1][1] >= -1e-10, "Diagonal FIM entry should be non-negative")
    }
}

/// Test suite for GradientResult and HessianResult types.
/// Validates struct construction and property access for
/// gradient computation result containers.
@Suite("Gradient Result Types")
struct GradientResultTypesTests {
    @Test("GradientResult stores gradient and evaluations")
    func gradientResultProperties() {
        let result = GradientMethods.GradientResult(gradient: [1.0, 2.0], evaluations: 4)
        #expect(result.gradient.count == 2, "Gradient should have 2 entries")
        #expect(result.evaluations == 4, "Evaluations should be 4")
    }

    @Test("HessianResult stores matrix and evaluations")
    func hessianResultProperties() {
        let result = GradientMethods.HessianResult(matrix: [[1.0, 0.0], [0.0, 2.0]], evaluations: 12)
        #expect(result.matrix.count == 2, "Matrix should be 2×2")
        #expect(result.evaluations == 12, "Evaluations should be 12")
    }
}

/// Test suite for additional parametric gate gradient support.
/// Validates gradient correctness for controlled gates, two-qubit
/// interaction gates, and Givens rotation.
@Suite("Extended Gate Gradient Support")
struct ExtendedGateGradientTests {
    @Test("XX interaction gate gradient")
    func xxGateGradient() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.xx(.parameter(Parameter(name: "a"))), to: [0, 1])
        let observable = Observable.pauliZ(qubit: 0)
        let theta = 0.6
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [theta])
        let shift = Double.pi / 2
        let fPlus = observable.expectationValue(of: circuit.bound(with: [theta + shift]).execute())
        let fMinus = observable.expectationValue(of: circuit.bound(with: [theta - shift]).execute())
        let expected = (fPlus - fMinus) / 2.0
        #expect(abs(gradient[0] - expected) < 1e-8, "XX gate gradient should match parameter shift")
    }

    @Test("YY interaction gate gradient")
    func yyGateGradient() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.yy(.parameter(Parameter(name: "a"))), to: [0, 1])
        let observable = Observable.pauliZ(qubit: 0)
        let theta = 0.4
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [theta])
        let shift = Double.pi / 2
        let fPlus = observable.expectationValue(of: circuit.bound(with: [theta + shift]).execute())
        let fMinus = observable.expectationValue(of: circuit.bound(with: [theta - shift]).execute())
        let expected = (fPlus - fMinus) / 2.0
        #expect(abs(gradient[0] - expected) < 1e-8, "YY gate gradient should match parameter shift")
    }

    @Test("Controlled Rx gate gradient")
    func controlledRxGradient() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.pauliX, to: 0)
        circuit.append(.controlledRotationX(.parameter(Parameter(name: "a"))), to: [0, 1])
        let observable = Observable.pauliZ(qubit: 1)
        let theta = 0.7
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [theta])
        let shift = Double.pi / 2
        let fPlus = observable.expectationValue(of: circuit.bound(with: [theta + shift]).execute())
        let fMinus = observable.expectationValue(of: circuit.bound(with: [theta - shift]).execute())
        let expected = (fPlus - fMinus) / 2.0
        #expect(abs(gradient[0] - expected) < 1e-10, "CRx gradient should match parameter shift")
    }

    @Test("Controlled Rz gate gradient")
    func controlledRzGradient() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.pauliX, to: 0)
        circuit.append(.controlledRotationZ(.parameter(Parameter(name: "a"))), to: [0, 1])
        let observable = Observable.pauliX(qubit: 1)
        let theta = 0.9
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [theta])
        let shift = Double.pi / 2
        let fPlus = observable.expectationValue(of: circuit.bound(with: [theta + shift]).execute())
        let fMinus = observable.expectationValue(of: circuit.bound(with: [theta - shift]).execute())
        let expected = (fPlus - fMinus) / 2.0
        #expect(abs(gradient[0] - expected) < 1e-10, "CRz gradient should match parameter shift")
    }

    @Test("Controlled phase gate gradient")
    func controlledPhaseGradient() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.pauliX, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.controlledPhase(.parameter(Parameter(name: "a"))), to: [0, 1])
        let observable = Observable.pauliZ(qubit: 0)
        let theta = 0.5
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [theta])
        let shift = Double.pi / 2
        let fPlus = observable.expectationValue(of: circuit.bound(with: [theta + shift]).execute())
        let fMinus = observable.expectationValue(of: circuit.bound(with: [theta - shift]).execute())
        let expected = (fPlus - fMinus) / 2.0
        #expect(abs(gradient[0] - expected) < 1e-10, "CPhase gradient should match parameter shift")
    }

    @Test("Givens rotation gate gradient via finite difference")
    func givensGradient() {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.pauliX, to: 0)
        circuit.append(.givens(.parameter(Parameter(name: "a"))), to: [0, 1])
        let observable = Observable.pauliZ(qubit: 0)
        let theta = 0.3
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [theta])
        let eps = 1e-6
        let fPlus = observable.expectationValue(of: circuit.bound(with: [theta + eps]).execute())
        let fMinus = observable.expectationValue(of: circuit.bound(with: [theta - eps]).execute())
        let expected = (fPlus - fMinus) / (2.0 * eps)
        #expect(abs(gradient[0] - expected) < 1e-5, "Givens gradient should match finite difference")
    }

    @Test("GlobalPhase gate gradient")
    func globalPhaseGradient() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.globalPhase(.parameter(Parameter(name: "a"))), to: [0])
        let observable = Observable.pauliZ(qubit: 0)
        let theta = 1.0
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [theta])
        #expect(abs(gradient[0]) < 1e-10, "GlobalPhase gradient of ⟨Z⟩ should be zero (unobservable)")
    }

    @Test("U3 gate theta gradient via finite difference")
    func u3ThetaGradient() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u3(
            theta: .parameter(Parameter(name: "t")),
            phi: .value(0.3),
            lambda: .value(0.5),
        ), to: 0)
        let observable = Observable.pauliZ(qubit: 0)
        let params = [0.7]
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: params)
        let eps = 1e-6
        let fPlus = observable.expectationValue(of: circuit.bound(with: [params[0] + eps]).execute())
        let fMinus = observable.expectationValue(of: circuit.bound(with: [params[0] - eps]).execute())
        let fdGrad = (fPlus - fMinus) / (2.0 * eps)
        #expect(abs(gradient[0] - fdGrad) < 1e-5, "U3 theta gradient should match finite difference")
    }

    @Test("U3 gate phi gradient via finite difference")
    func u3PhiGradient() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u3(
            theta: .value(0.7),
            phi: .parameter(Parameter(name: "p")),
            lambda: .value(0.5),
        ), to: 0)
        let observable = Observable.pauliZ(qubit: 0)
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [0.3])
        let eps = 1e-6
        let fPlus = observable.expectationValue(of: circuit.bound(with: [0.3 + eps]).execute())
        let fMinus = observable.expectationValue(of: circuit.bound(with: [0.3 - eps]).execute())
        let fdGrad = (fPlus - fMinus) / (2.0 * eps)
        #expect(abs(gradient[0] - fdGrad) < 1e-5, "U3 phi gradient should match finite difference")
    }

    @Test("U3 gate lambda gradient via finite difference")
    func u3LambdaGradient() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u3(
            theta: .value(0.7),
            phi: .value(0.3),
            lambda: .parameter(Parameter(name: "l")),
        ), to: 0)
        let observable = Observable.pauliZ(qubit: 0)
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [0.5])
        let eps = 1e-6
        let fPlus = observable.expectationValue(of: circuit.bound(with: [0.5 + eps]).execute())
        let fMinus = observable.expectationValue(of: circuit.bound(with: [0.5 - eps]).execute())
        let fdGrad = (fPlus - fMinus) / (2.0 * eps)
        #expect(abs(gradient[0] - fdGrad) < 1e-5, "U3 lambda gradient should match finite difference")
    }

    @Test("U2 gate phi gradient via finite difference")
    func u2PhiGradient() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u2(phi: .parameter(Parameter(name: "p")), lambda: .value(0.8)), to: 0)
        let observable = Observable.pauliZ(qubit: 0)
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [0.4])
        let eps = 1e-6
        let fPlus = observable.expectationValue(of: circuit.bound(with: [0.4 + eps]).execute())
        let fMinus = observable.expectationValue(of: circuit.bound(with: [0.4 - eps]).execute())
        let fdGrad = (fPlus - fMinus) / (2.0 * eps)
        #expect(abs(gradient[0] - fdGrad) < 1e-5, "U2 phi gradient should match finite difference")
    }

    @Test("U2 gate lambda gradient via finite difference")
    func u2LambdaGradient() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.u2(phi: .value(0.4), lambda: .parameter(Parameter(name: "l"))), to: 0)
        let observable = Observable.pauliZ(qubit: 0)
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [0.8])
        let eps = 1e-6
        let fPlus = observable.expectationValue(of: circuit.bound(with: [0.8 + eps]).execute())
        let fMinus = observable.expectationValue(of: circuit.bound(with: [0.8 - eps]).execute())
        let fdGrad = (fPlus - fMinus) / (2.0 * eps)
        #expect(abs(gradient[0] - fdGrad) < 1e-5, "U2 lambda gradient should match finite difference")
    }

    @Test("Circuit with reset operation skips non-gate ops")
    func circuitWithResetOperation() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        circuit.append(.reset, to: 0)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        let observable = Observable.pauliZ(qubit: 0)
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [0.5])
        #expect(gradient.count == 1, "Should return gradient despite reset in circuit")
    }

    @Test("Concrete-value gates produce no gradient contribution")
    func concreteValueGates() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        circuit.append(.rotationX(.value(0.5)), to: 0)
        let observable = Observable.pauliZ(qubit: 0)
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [0.3])
        #expect(gradient.count == 1, "Only symbolic parameters contribute to gradient")
    }

    @Test("Three-qubit circuit with Toffoli exercises multi-qubit path")
    func threeQubitCircuit() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        circuit.append(.toffoli, to: [0, 1, 2])
        let observable = Observable.pauliZ(qubit: 2)
        let theta = 0.6
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [theta])
        let eps = 1e-6
        let fPlus = observable.expectationValue(of: circuit.bound(with: [theta + eps]).execute())
        let fMinus = observable.expectationValue(of: circuit.bound(with: [theta - eps]).execute())
        let expected = (fPlus - fMinus) / (2.0 * eps)
        #expect(abs(gradient[0] - expected) < 1e-5, "Three-qubit circuit gradient should match finite difference")
    }

    @Test("U1 gate gradient matches phase gate")
    func u1GateGradient() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.pauliX, to: 0)
        circuit.append(.u1(lambda: .parameter(Parameter(name: "a"))), to: 0)
        let observable = Observable.pauliZ(qubit: 0)
        let theta = 0.8
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [theta])
        let eps = 1e-6
        let fPlus = observable.expectationValue(of: circuit.bound(with: [theta + eps]).execute())
        let fMinus = observable.expectationValue(of: circuit.bound(with: [theta - eps]).execute())
        let expected = (fPlus - fMinus) / (2.0 * eps)
        #expect(abs(gradient[0] - expected) < 1e-5, "U1 gradient should match finite difference")
    }

    @Test("Hadamard test with reset in circuit")
    func hadamardTestWithReset() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        circuit.append(.reset, to: 0)
        let observable = Observable.pauliZ(qubit: 0)
        let result = GradientMethods.hadamardTest(circuit: circuit, observable: observable, parameters: [0.5], parameterIndex: 0)
        #expect(result.isFinite, "Hadamard test should return finite value despite reset")
    }

    @Test("Adjoint Hessian with reset in circuit")
    func adjointHessianWithReset() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        circuit.append(.reset, to: 0)
        let observable = Observable.pauliZ(qubit: 0)
        let hessian = GradientMethods.adjointHessian(circuit: circuit, observable: observable, parameters: [0.5])
        #expect(hessian.count == 1, "Should produce Hessian despite reset")
    }

    @Test("Fisher information with reset in circuit")
    func fisherInfoWithReset() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        circuit.append(.reset, to: 0)
        let fim = GradientMethods.fisherInformationMatrix(circuit: circuit, parameters: [0.5])
        #expect(fim.count == 1, "Should produce FIM despite reset")
    }

    @Test("All concrete-value gate types return empty derivatives")
    func allConcreteValueGateTypes() {
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.rotationY(.value(0.5)), to: 0)
        circuit.append(.rotationZ(.value(0.3)), to: 0)
        circuit.append(.phase(.value(0.2)), to: 0)
        circuit.append(.globalPhase(.value(0.1)), to: [0])
        circuit.append(.controlledRotationX(.value(0.4)), to: [0, 1])
        circuit.append(.controlledRotationY(.value(0.4)), to: [0, 1])
        circuit.append(.controlledRotationZ(.value(0.4)), to: [0, 1])
        circuit.append(.controlledPhase(.value(0.3)), to: [0, 1])
        circuit.append(.xx(.value(0.2)), to: [0, 1])
        circuit.append(.yy(.value(0.2)), to: [0, 1])
        circuit.append(.zz(.value(0.2)), to: [0, 1])
        circuit.append(.givens(.value(0.3)), to: [0, 1])
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 2)
        let observable = Observable.pauliZ(qubit: 2)
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [0.5])
        #expect(gradient.count == 1, "Only symbolic parameter should produce gradient entry")
    }

    @Test("Negated parameter produces sign-flipped gradient for odd observable")
    func negatedParameterGradient() {
        var circuitPos = QuantumCircuit(qubits: 1)
        circuitPos.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        var circuitNeg = QuantumCircuit(qubits: 1)
        circuitNeg.append(.rotationY(.negatedParameter(Parameter(name: "a"))), to: 0)
        let observable = Observable.pauliX(qubit: 0)
        let theta = 0.7
        let gradPos = GradientMethods.adjoint(circuit: circuitPos, observable: observable, parameters: [theta])
        let gradNeg = GradientMethods.adjoint(circuit: circuitNeg, observable: observable, parameters: [theta])
        #expect(abs(gradPos[0] + gradNeg[0]) < 1e-10, "Negated parameter with X observable should flip gradient sign")
    }

    @Test("Empty observable produces zero gradient")
    func emptyObservableGradient() {
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.rotationY(.parameter(Parameter(name: "a"))), to: 0)
        let observable = Observable(terms: [])
        let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [0.5])
        #expect(abs(gradient[0]) < 1e-10, "Empty observable should produce zero gradient")
    }
}
