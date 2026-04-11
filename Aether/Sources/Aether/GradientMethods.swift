// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate
import Foundation

/// Gradient computation methods for variational quantum algorithms.
///
/// Provides multiple differentiation strategies for computing ∂⟨ψ(θ)|O|ψ(θ)⟩/∂θ, the gradient
/// of expectation values with respect to circuit parameters. Adjoint differentiation is the primary
/// method for simulation workloads, computing all parameter gradients in O(1) circuit evaluations
/// via reverse-mode accumulation on the statevector. Complex-step finite differences provide
/// machine-precision gradients when other methods are inapplicable. Stochastic parameter shift
/// offers O(1) circuit overhead per gradient estimate for SGD workflows with many parameters.
/// The Hadamard test computes gradients via ancilla-based controlled-unitary overlap.
/// Hessian and Fisher information matrix methods provide second-order curvature information
/// for Newton-type optimizers and statistical precision analysis.
///
/// **Example:**
/// ```swift
/// let circuit = QuantumCircuit(qubits: 2)
/// circuit.append(.rotationY(Parameter(name: "a")), to: 0)
/// circuit.append(.controlledNot, to: [0, 1])
/// let observable = Observable.pauliZ(qubit: 0)
/// let gradient = GradientMethods.adjoint(circuit: circuit, observable: observable, parameters: [0.5])
/// ```
///
/// - SeeAlso: ``Optimizer``
/// - SeeAlso: ``Observable``
/// - SeeAlso: ``QuantumCircuit``
public enum GradientMethods {
    // MARK: - Result Types

    /// Result of a gradient computation with evaluation metadata.
    ///
    /// Contains the computed gradient vector and the number of objective function evaluations
    /// consumed, enabling cost tracking across optimization iterations.
    ///
    /// **Example:**
    /// ```swift
    /// let result = await GradientMethods.stochasticParameterShift(objective, at: [0.1, 0.2])
    /// let grad = result.gradient
    /// let cost = result.evaluations
    /// ```
    ///
    /// - SeeAlso: ``GradientMethods``
    @frozen
    public struct GradientResult: Sendable {
        /// Gradient vector ∂E/∂θ with one entry per circuit parameter.
        public let gradient: [Double]

        /// Number of objective function evaluations consumed.
        public let evaluations: Int

        /// Create gradient result.
        public init(gradient: [Double], evaluations: Int) {
            self.gradient = gradient
            self.evaluations = evaluations
        }
    }

    /// Result of a Hessian computation with evaluation metadata.
    ///
    /// Contains the p×p Hessian matrix ∂²E/∂θᵢ∂θⱼ and the number of objective function
    /// evaluations consumed. The matrix is symmetric for smooth objectives.
    ///
    /// **Example:**
    /// ```swift
    /// let objective: @Sendable ([Double]) async -> Double = { p in p[0] * p[0] + p[1] * p[1] }
    /// let result = await GradientMethods.hessian(objective, at: [0.1, 0.2])
    /// let curvature = result.matrix[0][1]
    /// ```
    ///
    /// - SeeAlso: ``GradientMethods``
    @frozen
    public struct HessianResult: Sendable {
        /// Symmetric p×p Hessian matrix ∂²E/∂θᵢ∂θⱼ.
        public let matrix: [[Double]]

        /// Number of objective function evaluations consumed.
        public let evaluations: Int

        /// Create Hessian result.
        public init(matrix: [[Double]], evaluations: Int) {
            self.matrix = matrix
            self.evaluations = evaluations
        }
    }

    // MARK: - Adjoint Differentiation

    /// Compute all parameter gradients via adjoint (reverse-mode) differentiation.
    ///
    /// Performs one forward pass to obtain the final statevector, applies the observable, then
    /// accumulates gradients by reverse-iterating through circuit gates while recomputing intermediate
    /// states on-the-fly via inverse gate application. Each parametric gate contributes
    /// ∂E/∂θₖ = 2·Re(⟨λⱼ|∂Uⱼ/∂θₖ|φⱼ⟩) where |φⱼ⟩ is the forward state before gate j and |λⱼ⟩
    /// is the adjoint state propagated backward from O|ψ⟩. This achieves O(1) circuit evaluations
    /// for the full gradient regardless of parameter count, compared to O(2p) for parameter shift.
    /// The default gradient method for all simulation workloads where statevector access is available.
    ///
    /// - Parameters:
    ///   - circuit: Parameterized quantum circuit with symbolic parameters
    ///   - observable: Hermitian observable O for expectation value ⟨ψ|O|ψ⟩
    ///   - parameters: Concrete parameter values in circuit registration order
    /// - Returns: Gradient vector with one entry per circuit parameter
    /// - Precondition: parameters.count == circuit.parameterCount
    /// - Precondition: Circuit contains only unitary gate operations
    /// - Complexity: O(L·2ⁿ) time for L gates and n qubits, O(2ⁿ) memory
    /// - SeeAlso: ``stochasticParameterShift(_:at:shift:)`` for hardware execution
    /// - SeeAlso: ``adjointHessian(circuit:observable:parameters:)`` for second-order information
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.rotationY(Parameter(name: "a")), to: 0)
    /// circuit.append(.controlledNot, to: [0, 1])
    /// let obs = Observable.pauliZ(qubit: 0)
    /// let grad = GradientMethods.adjoint(circuit: circuit, observable: obs, parameters: [0.5])
    /// ```
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func adjoint(
        circuit: QuantumCircuit,
        observable: Observable,
        parameters: [Double],
    ) -> [Double] {
        let paramList = circuit.parameters
        let paramCount = paramList.count
        ValidationUtilities.validateArrayCount(parameters, expected: paramCount, name: "parameters")

        guard paramCount > 0 else { return [] }

        let qubitCount = circuit.qubits

        var paramNameToIndex: [String: Int] = [:]
        paramNameToIndex.reserveCapacity(paramCount)
        var parameterBindings: [String: Double] = [:]
        parameterBindings.reserveCapacity(paramCount)
        for i in 0 ..< paramCount {
            paramNameToIndex[paramList[i].name] = i
            parameterBindings[paramList[i].name] = parameters[i]
        }

        let boundCircuit = circuit.bound(with: parameters)
        let boundOps = boundCircuit.operations
        let originalOps = circuit.operations
        let opCount = boundOps.count

        var currentState = QuantumState(qubits: qubitCount)
        for i in 0 ..< opCount {
            currentState = GateApplication.apply(boundOps[i], state: currentState)
        }

        var lambda = applyObservableToAmplitudes(
            observable,
            amplitudes: currentState.amplitudes,
            qubitCount: qubitCount,
        )

        var gradient = [Double](unsafeUninitializedCapacity: paramCount) {
            buffer, count in
            buffer.initialize(repeating: 0.0)
            count = paramCount
        }

        var forwardAmps = currentState.amplitudes

        for j in stride(from: opCount - 1, through: 0, by: -1) {
            guard case let .gate(originalGate, qubits, _) = originalOps[j] else {
                continue
            }

            guard case let .gate(boundGate, boundQubits, _) = boundOps[j] else {
                continue
            }

            let inverseMatrix = boundGate.inverse.matrix()

            let stateBeforeAmps = applyGateMatrixToAmplitudes(
                inverseMatrix,
                qubits: boundQubits,
                amplitudes: forwardAmps,
                qubitCount: qubitCount,
            )

            let derivatives = computeGateDerivatives(
                originalGate: originalGate,
                qubits: qubits,
                stateAfterAmplitudes: forwardAmps,
                stateBeforeAmplitudes: stateBeforeAmps,
                paramNameToIndex: paramNameToIndex,
                parameterBindings: parameterBindings,
                qubitCount: qubitCount,
            )

            for entry in derivatives {
                let ip = complexInnerProduct(lambda, entry.amplitudes)
                gradient[entry.parameterIndex] += 2.0 * entry.signMultiplier * ip.real
            }

            lambda = applyGateMatrixToAmplitudes(
                inverseMatrix,
                qubits: boundQubits,
                amplitudes: lambda,
                qubitCount: qubitCount,
            )

            forwardAmps = stateBeforeAmps
        }

        return gradient
    }

    // MARK: - Complex-Step Finite Differences

    /// Compute gradient via complex-step finite differences with machine precision.
    ///
    /// Evaluates ∂f/∂θᵢ = Im[f(θ + iε·eᵢ)] / ε where ε is a tiny real number. Unlike standard
    /// finite differences, complex-step has zero truncation error at any step size because it
    /// avoids subtractive cancellation. The step size ε can be as small as 1e-200 while maintaining
    /// full machine precision (≈1e-15 relative error). Requires the objective function to accept
    /// complex-valued parameters and be analytically continuable to the complex plane.
    ///
    /// - Parameters:
    ///   - function: Complex-valued function accepting complex parameters
    ///   - parameters: Real parameter values at which to evaluate the gradient
    ///   - epsilon: Complex perturbation magnitude (default: 1e-20)
    /// - Returns: Gradient vector with machine-precision accuracy
    /// - Precondition: parameters is non-empty
    /// - Precondition: epsilon > 0
    /// - Complexity: O(p) function evaluations where p is the parameter count
    /// - SeeAlso: ``adjoint(circuit:observable:parameters:)`` for quantum circuit gradients
    ///
    /// **Example:**
    /// ```swift
    /// let grad = GradientMethods.complexStep(
    ///     { z in z[0] * z[0] + z[1] * z[1] },
    ///     at: [1.0, 2.0]
    /// )
    /// ```
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func complexStep(
        _ function: @Sendable ([Complex<Double>]) -> Complex<Double>,
        at parameters: [Double],
        epsilon: Double = 1e-20,
    ) -> [Double] {
        let paramCount = parameters.count
        ValidationUtilities.validateNonEmpty(parameters, name: "parameters")
        ValidationUtilities.validatePositiveDouble(epsilon, name: "epsilon")

        let inverseEpsilon = 1.0 / epsilon

        var complexParams = [Complex<Double>](unsafeUninitializedCapacity: paramCount) {
            buffer, count in
            for i in 0 ..< paramCount {
                buffer[i] = Complex(parameters[i], 0.0)
            }
            count = paramCount
        }

        return [Double](unsafeUninitializedCapacity: paramCount) { buffer, count in
            for i in 0 ..< paramCount {
                complexParams[i] = Complex(parameters[i], epsilon)
                let result = function(complexParams)
                buffer[i] = result.imaginary * inverseEpsilon
                complexParams[i] = Complex(parameters[i], 0.0)
            }
            count = paramCount
        }
    }

    // MARK: - Stochastic Parameter Shift

    /// Compute stochastic gradient estimate using random parameter shift direction.
    ///
    /// Generates a random Rademacher vector d ∈ {-1,+1}ᵖ and evaluates the objective at two
    /// shifted points θ ± (π/2)·d. The gradient estimator ĝ = [f(θ+s·d) - f(θ-s·d)] / (2·sin(s)) · d
    /// is unbiased with per-estimate variance O(p). For SGD/Adam workflows where noisy gradient
    /// estimates suffice, this achieves O(1) circuit evaluations per optimization step versus
    /// O(2p) for deterministic parameter shift. Advantage manifests when p > 10 parameters with
    /// SGD-style optimizers that tolerate gradient noise.
    ///
    /// - Parameters:
    ///   - objectiveFunction: Async objective function to differentiate
    ///   - parameters: Current parameter values
    ///   - shift: Parameter shift amount (default: π/2 for standard quantum gates)
    /// - Returns: Gradient result with stochastic estimate and evaluation count
    /// - Precondition: parameters is non-empty
    /// - Precondition: shift > 0
    /// - Complexity: O(1) objective evaluations regardless of parameter count
    /// - SeeAlso: ``adjoint(circuit:observable:parameters:)`` for exact gradients
    ///
    /// **Example:**
    /// ```swift
    /// let result = await GradientMethods.stochasticParameterShift(
    ///     { params in circuit.bound(with: params).execute().probability(of: 0) },
    ///     at: [0.5, 0.3, 0.1]
    /// )
    /// ```
    @_optimize(speed)
    @_eagerMove
    public static func stochasticParameterShift(
        _ objectiveFunction: @Sendable ([Double]) async -> Double,
        at parameters: [Double],
        shift: Double = .pi / 2,
    ) async -> GradientResult {
        let paramCount = parameters.count
        ValidationUtilities.validateNonEmpty(parameters, name: "parameters")
        ValidationUtilities.validatePositiveDouble(shift, name: "shift")

        let numBytes = (paramCount + 7) / 8
        var randomBytes = [UInt8](repeating: 0, count: numBytes)
        arc4random_buf(&randomBytes, numBytes)

        let direction = [Double](unsafeUninitializedCapacity: paramCount) { buffer, count in
            for i in 0 ..< paramCount {
                let byteIndex = i / 8
                let bitIndex = i % 8
                let bit = (randomBytes[byteIndex] >> bitIndex) & 1
                buffer[i] = 1.0 - 2.0 * Double(bit)
            }
            count = paramCount
        }

        var paramsPlus = [Double](unsafeUninitializedCapacity: paramCount) {
            _, count in
            count = paramCount
        }
        var paramsMinus = [Double](unsafeUninitializedCapacity: paramCount) {
            _, count in
            count = paramCount
        }
        var shiftVal = shift
        var negShift = -shift
        vDSP_vsmaD(direction, 1, &shiftVal, parameters, 1, &paramsPlus, 1, vDSP_Length(paramCount))
        vDSP_vsmaD(direction, 1, &negShift, parameters, 1, &paramsMinus, 1, vDSP_Length(paramCount))

        let valuePlus = await objectiveFunction(paramsPlus)
        let valueMinus = await objectiveFunction(paramsMinus)

        let scale = (valuePlus - valueMinus) / (2.0 * sin(shift))

        let gradient = [Double](unsafeUninitializedCapacity: paramCount) { buffer, count in
            var scaleVar = scale
            vDSP_vsmulD(direction, 1, &scaleVar, buffer.baseAddress!, 1, vDSP_Length(paramCount))
            count = paramCount
        }

        return GradientResult(gradient: gradient, evaluations: 2)
    }

    // MARK: - Hadamard Test

    /// Compute single-parameter gradient via Hadamard test formulation.
    ///
    /// Computes ∂⟨ψ|O|ψ⟩/∂θₖ using the controlled-unitary overlap interpretation. On quantum
    /// hardware, this uses a single ancilla qubit and controlled-U circuit to estimate the gradient
    /// as Re⟨0|U†∂U|0⟩. In simulation, computes the equivalent inner product directly from the
    /// statevector. More efficient than parameter shift when controlled gates are natively available,
    /// and extends naturally to higher-order derivatives via repeated Hadamard tests.
    ///
    /// - Parameters:
    ///   - circuit: Parameterized quantum circuit
    ///   - observable: Hermitian observable
    ///   - parameters: Concrete parameter values
    ///   - parameterIndex: Index of the parameter to differentiate
    /// - Returns: Scalar gradient component ∂E/∂θₖ
    /// - Precondition: parameters.count == circuit.parameterCount
    /// - Precondition: parameterIndex in 0..<circuit.parameterCount
    /// - Complexity: O(L·2ⁿ) time, O(2ⁿ) memory
    /// - SeeAlso: ``adjoint(circuit:observable:parameters:)`` for full gradient vector
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.rotationY(Parameter(name: "a")), to: 0)
    /// let obs = Observable.pauliZ(qubit: 0)
    /// let dEdA = GradientMethods.hadamardTest(circuit: circuit, observable: obs, parameters: [0.5], parameterIndex: 0)
    /// ```
    @_optimize(speed)
    @_eagerMove
    public static func hadamardTest(
        circuit: QuantumCircuit,
        observable: Observable,
        parameters: [Double],
        parameterIndex: Int,
    ) -> Double {
        let paramList = circuit.parameters
        let paramCount = paramList.count
        ValidationUtilities.validateArrayCount(parameters, expected: paramCount, name: "parameters")
        ValidationUtilities.validateIndexInBounds(parameterIndex, bound: paramCount, name: "parameterIndex")

        let qubitCount = circuit.qubits
        let boundCircuit = circuit.bound(with: parameters)
        let boundOps = boundCircuit.operations
        let originalOps = circuit.operations

        var paramNameToIndex: [String: Int] = [:]
        paramNameToIndex.reserveCapacity(paramCount)
        var parameterBindings: [String: Double] = [:]
        parameterBindings.reserveCapacity(paramCount)
        for i in 0 ..< paramCount {
            paramNameToIndex[paramList[i].name] = i
            parameterBindings[paramList[i].name] = parameters[i]
        }

        var forwardAmps: [[Complex<Double>]] = []
        forwardAmps.reserveCapacity(boundOps.count + 1)

        let stateSize = 1 << qubitCount
        let initialAmps = [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            buffer.initialize(repeating: .zero)
            buffer[0] = .one
            count = stateSize
        }
        forwardAmps.append(initialAmps)

        var currentState = QuantumState(qubits: qubitCount)
        for op in boundOps {
            currentState = GateApplication.apply(op, state: currentState)
            forwardAmps.append(currentState.amplitudes)
        }

        var lambda = applyObservableToAmplitudes(
            observable,
            amplitudes: forwardAmps[boundOps.count],
            qubitCount: qubitCount,
        )

        var gradientValue = 0.0
        let opCount = boundOps.count

        for j in stride(from: opCount - 1, through: 0, by: -1) {
            guard case let .gate(originalGate, qubits, _) = originalOps[j] else {
                continue
            }

            let derivatives = computeGateDerivatives(
                originalGate: originalGate,
                qubits: qubits,
                stateAfterAmplitudes: forwardAmps[j + 1],
                stateBeforeAmplitudes: forwardAmps[j],
                paramNameToIndex: paramNameToIndex,
                parameterBindings: parameterBindings,
                qubitCount: qubitCount,
            )

            for entry in derivatives where entry.parameterIndex == parameterIndex {
                let ip = complexInnerProduct(lambda, entry.amplitudes)
                gradientValue += 2.0 * entry.signMultiplier * ip.real
            }

            if case let .gate(boundGate, boundQubits, _) = boundOps[j] {
                lambda = applyGateMatrixToAmplitudes(
                    boundGate.inverse.matrix(),
                    qubits: boundQubits,
                    amplitudes: lambda,
                    qubitCount: qubitCount,
                )
            }
        }

        return gradientValue
    }

    // MARK: - Hessian via Parameter Shift

    /// Compute Hessian matrix via double parameter shift rule.
    ///
    /// Evaluates ∂²E/∂θᵢ∂θⱼ using the formula:
    /// [E(θᵢ+s,θⱼ+s) - E(θᵢ+s,θⱼ-s) - E(θᵢ-s,θⱼ+s) + E(θᵢ-s,θⱼ-s)] / (4·sin²(s)).
    /// Requires 4 evaluations per off-diagonal pair and 2 per diagonal, totaling
    /// 2p² + 2p evaluations for the full p×p matrix. The result is symmetric.
    ///
    /// - Parameters:
    ///   - objectiveFunction: Async objective function
    ///   - parameters: Current parameter values
    ///   - shift: Parameter shift amount (default: π/2)
    /// - Returns: Hessian result with p×p matrix and evaluation count
    /// - Precondition: parameters is non-empty
    /// - Precondition: shift > 0
    /// - Complexity: O(p²) objective evaluations
    /// - SeeAlso: ``adjointHessian(circuit:observable:parameters:)`` for O(p) circuit evaluations
    ///
    /// **Example:**
    /// ```swift
    /// let result = await GradientMethods.hessian(
    ///     { params in circuit.bound(with: params).execute().probability(of: 0) },
    ///     at: [0.5, 0.3]
    /// )
    /// ```
    @_optimize(speed)
    @_eagerMove
    public static func hessian(
        _ objectiveFunction: @Sendable ([Double]) async -> Double,
        at parameters: [Double],
        shift: Double = .pi / 2,
    ) async -> HessianResult {
        let p = parameters.count
        ValidationUtilities.validateNonEmpty(parameters, name: "parameters")
        ValidationUtilities.validatePositiveDouble(shift, name: "shift")

        let sinShift = sin(shift)
        let denominator = 4.0 * sinShift * sinShift
        let inverseDenom = 1.0 / denominator
        var evaluations = 0

        var matrix = [[Double]](unsafeUninitializedCapacity: p) {
            buffer, count in
            for i in 0 ..< p {
                buffer[i] = [Double](unsafeUninitializedCapacity: p) { inner, innerCount in
                    inner.initialize(repeating: 0.0)
                    innerCount = p
                }
            }
            count = p
        }

        var shifted = parameters

        for i in 0 ..< p {
            for j in i ..< p {
                for k in 0 ..< p {
                    shifted[k] = parameters[k]
                }
                shifted[i] += shift
                shifted[j] += shift
                let fPP = await objectiveFunction(shifted)

                for k in 0 ..< p {
                    shifted[k] = parameters[k]
                }
                shifted[i] += shift
                shifted[j] -= shift
                let fPN = await objectiveFunction(shifted)

                for k in 0 ..< p {
                    shifted[k] = parameters[k]
                }
                shifted[i] -= shift
                shifted[j] += shift
                let fNP = await objectiveFunction(shifted)

                for k in 0 ..< p {
                    shifted[k] = parameters[k]
                }
                shifted[i] -= shift
                shifted[j] -= shift
                let fNN = await objectiveFunction(shifted)

                let hessianEntry = (fPP - fPN - fNP + fNN) * inverseDenom
                matrix[i][j] = hessianEntry
                matrix[j][i] = hessianEntry
                evaluations += 4
            }
        }

        return HessianResult(matrix: matrix, evaluations: evaluations)
    }

    // MARK: - Adjoint Hessian

    /// Compute Hessian matrix via adjoint differentiation with O(p) backward passes.
    ///
    /// For each parameter θⱼ, performs a modified backward pass where the generator of gate j
    /// is applied to the forward state, creating a "derivative state" |∂ψ/∂θⱼ⟩. The Hessian
    /// entry H_{ij} = 2·Re(⟨∂ψ/∂θᵢ|O|∂ψ/∂θⱼ⟩) + 2·Re(⟨ψ|O|∂²ψ/∂θᵢ∂θⱼ⟩) is accumulated
    /// from the overlap of derivative states. Requires p backward passes for the full p×p matrix
    /// versus p² evaluations for the parameter shift approach.
    ///
    /// - Parameters:
    ///   - circuit: Parameterized quantum circuit
    ///   - observable: Hermitian observable
    ///   - parameters: Concrete parameter values
    /// - Returns: Symmetric p×p Hessian matrix
    /// - Precondition: parameters.count == circuit.parameterCount
    /// - Complexity: O(p·L·2ⁿ) time, O(L·2ⁿ) memory
    /// - SeeAlso: ``hessian(_:at:shift:)`` for black-box Hessian
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.rotationY(Parameter(name: "a")), to: 0)
    /// circuit.append(.rotationZ(Parameter(name: "b")), to: 1)
    /// let obs = Observable.pauliZ(qubit: 0)
    /// let H = GradientMethods.adjointHessian(circuit: circuit, observable: obs, parameters: [0.5, 0.3])
    /// ```
    @_optimize(speed)
    @_eagerMove
    public static func adjointHessian(
        circuit: QuantumCircuit,
        observable: Observable,
        parameters: [Double],
    ) -> [[Double]] {
        let paramList = circuit.parameters
        let paramCount = paramList.count
        ValidationUtilities.validateArrayCount(parameters, expected: paramCount, name: "parameters")

        guard paramCount > 0 else { return [] }

        var matrix = [[Double]](unsafeUninitializedCapacity: paramCount) {
            buffer, count in
            for i in 0 ..< paramCount {
                buffer[i] = [Double](unsafeUninitializedCapacity: paramCount) { inner, innerCount in
                    inner.initialize(repeating: 0.0)
                    innerCount = paramCount
                }
            }
            count = paramCount
        }

        let qubitCount = circuit.qubits
        let stateSize = 1 << qubitCount

        var paramNameToIndex: [String: Int] = [:]
        paramNameToIndex.reserveCapacity(paramCount)
        var parameterBindings: [String: Double] = [:]
        parameterBindings.reserveCapacity(paramCount)
        for i in 0 ..< paramCount {
            paramNameToIndex[paramList[i].name] = i
            parameterBindings[paramList[i].name] = parameters[i]
        }

        let boundCircuit = circuit.bound(with: parameters)
        let boundOps = boundCircuit.operations
        let originalOps = circuit.operations
        let opCount = boundOps.count

        var forwardStates: [[Complex<Double>]] = []
        forwardStates.reserveCapacity(opCount + 1)
        let initialAmps = [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            buffer.initialize(repeating: .zero)
            buffer[0] = .one
            count = stateSize
        }
        forwardStates.append(initialAmps)

        var currentState = QuantumState(qubits: qubitCount)
        for op in boundOps {
            currentState = GateApplication.apply(op, state: currentState)
            forwardStates.append(currentState.amplitudes)
        }

        var derivativeStates = [[Complex<Double>]](repeating: [], count: paramCount)

        for k in 0 ..< paramCount {
            var derivAmps = [Complex<Double>](unsafeUninitializedCapacity: stateSize) {
                buffer, count in
                buffer.initialize(repeating: .zero)
                count = stateSize
            }

            let targetName = paramList[k].name

            for j in 0 ..< opCount {
                guard case let .gate(originalGate, qubits, _) = originalOps[j] else {
                    continue
                }

                let derivatives = computeGateDerivatives(
                    originalGate: originalGate,
                    qubits: qubits,
                    stateAfterAmplitudes: forwardStates[j + 1],
                    stateBeforeAmplitudes: forwardStates[j],
                    paramNameToIndex: paramNameToIndex,
                    parameterBindings: parameterBindings,
                    qubitCount: qubitCount,
                )

                for entry in derivatives where paramList[entry.parameterIndex].name == targetName {
                    derivAmps = addAmplitudes(
                        derivAmps,
                        scaleAmplitudes(entry.amplitudes, by: Complex(entry.signMultiplier, 0.0)),
                    )
                }

                if case let .gate(boundGate, boundQubits, _) = boundOps[j] {
                    derivAmps = applyGateMatrixToAmplitudes(
                        boundGate.matrix(),
                        qubits: boundQubits,
                        amplitudes: derivAmps,
                        qubitCount: qubitCount,
                    )
                }
            }

            derivativeStates[k] = derivAmps
        }

        for i in 0 ..< paramCount {
            let oDerivI = applyObservableToAmplitudes(
                observable,
                amplitudes: derivativeStates[i],
                qubitCount: qubitCount,
            )
            for j in i ..< paramCount {
                let ip = complexInnerProduct(oDerivI, derivativeStates[j])
                let value = 2.0 * ip.real
                matrix[i][j] += value
                if i != j {
                    matrix[j][i] += value
                }
            }
        }

        return matrix
    }

    // MARK: - Classical Fisher Information Matrix

    /// Compute the classical Fisher information matrix from output probability gradients.
    ///
    /// Evaluates F_ij = Σₓ (∂log p(x|θ)/∂θᵢ)(∂log p(x|θ)/∂θⱼ) where p(x|θ) = |⟨x|ψ(θ)⟩|²
    /// are the Born rule measurement probabilities. Uses adjoint infrastructure to compute
    /// probability gradients ∂p(x)/∂θᵢ = 2·Re(⟨x|ψ⟩* · ⟨x|∂ψ/∂θᵢ⟩) for all basis states x.
    /// The Fisher information matrix provides the Cramér-Rao bound Var(θ̂) ≥ F⁻¹, the fundamental
    /// precision limit for parameter estimation, and connects to quantum Fisher information
    /// via measurement optimization.
    ///
    /// - Parameters:
    ///   - circuit: Parameterized quantum circuit
    ///   - parameters: Concrete parameter values
    /// - Returns: Symmetric p×p Fisher information matrix
    /// - Precondition: parameters.count == circuit.parameterCount
    /// - Complexity: O(p·L·2ⁿ) time, O(p·2ⁿ) memory for derivative states
    /// - SeeAlso: ``adjointHessian(circuit:observable:parameters:)`` for energy Hessian
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.rotationY(Parameter(name: "a")), to: 0)
    /// circuit.append(.controlledNot, to: [0, 1])
    /// let F = GradientMethods.fisherInformationMatrix(circuit: circuit, parameters: [0.5])
    /// ```
    @_optimize(speed)
    @_eagerMove
    public static func fisherInformationMatrix(
        circuit: QuantumCircuit,
        parameters: [Double],
    ) -> [[Double]] {
        let paramList = circuit.parameters
        let paramCount = paramList.count
        ValidationUtilities.validateArrayCount(parameters, expected: paramCount, name: "parameters")

        guard paramCount > 0 else { return [] }

        let qubitCount = circuit.qubits
        let stateSize = 1 << qubitCount

        var paramNameToIndex: [String: Int] = [:]
        paramNameToIndex.reserveCapacity(paramCount)
        var parameterBindings: [String: Double] = [:]
        parameterBindings.reserveCapacity(paramCount)
        for i in 0 ..< paramCount {
            paramNameToIndex[paramList[i].name] = i
            parameterBindings[paramList[i].name] = parameters[i]
        }

        let boundCircuit = circuit.bound(with: parameters)
        let boundOps = boundCircuit.operations
        let originalOps = circuit.operations
        let opCount = boundOps.count

        var forwardStates: [[Complex<Double>]] = []
        forwardStates.reserveCapacity(opCount + 1)
        let initialAmps = [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            buffer.initialize(repeating: .zero)
            buffer[0] = .one
            count = stateSize
        }
        forwardStates.append(initialAmps)

        var currentState = QuantumState(qubits: qubitCount)
        for op in boundOps {
            currentState = GateApplication.apply(op, state: currentState)
            forwardStates.append(currentState.amplitudes)
        }

        let finalAmps = forwardStates[opCount]

        var derivativeStates = [[Complex<Double>]](repeating: [], count: paramCount)

        for k in 0 ..< paramCount {
            var derivAmps = [Complex<Double>](unsafeUninitializedCapacity: stateSize) {
                buffer, count in
                buffer.initialize(repeating: .zero)
                count = stateSize
            }

            for j in 0 ..< opCount {
                guard case let .gate(originalGate, qubits, _) = originalOps[j] else {
                    continue
                }

                let derivatives = computeGateDerivatives(
                    originalGate: originalGate,
                    qubits: qubits,
                    stateAfterAmplitudes: forwardStates[j + 1],
                    stateBeforeAmplitudes: forwardStates[j],
                    paramNameToIndex: paramNameToIndex,
                    parameterBindings: parameterBindings,
                    qubitCount: qubitCount,
                )

                for entry in derivatives where entry.parameterIndex == k {
                    derivAmps = addAmplitudes(
                        derivAmps,
                        scaleAmplitudes(entry.amplitudes, by: Complex(entry.signMultiplier, 0.0)),
                    )
                }

                if case let .gate(boundGate, boundQubits, _) = boundOps[j] {
                    derivAmps = applyGateMatrixToAmplitudes(
                        boundGate.matrix(),
                        qubits: boundQubits,
                        amplitudes: derivAmps,
                        qubitCount: qubitCount,
                    )
                }
            }

            derivativeStates[k] = derivAmps
        }

        let probGradients = [[Double]](unsafeUninitializedCapacity: paramCount) { buffer, count in
            for k in 0 ..< paramCount {
                buffer[k] = [Double](unsafeUninitializedCapacity: stateSize) { inner, innerCount in
                    for x in 0 ..< stateSize {
                        let psiX = finalAmps[x]
                        let dPsiX = derivativeStates[k][x]
                        inner[x] = 2.0 * Double.fusedMultiplyAdd(psiX.real, dPsiX.real, psiX.imaginary * dPsiX.imaginary)
                    }
                    innerCount = stateSize
                }
            }
            count = paramCount
        }

        let probabilityFloor = GateApplication.probabilityFloor

        var fisherMatrix = [[Double]](unsafeUninitializedCapacity: paramCount) {
            buffer, count in
            for i in 0 ..< paramCount {
                buffer[i] = [Double](unsafeUninitializedCapacity: paramCount) { inner, innerCount in
                    inner.initialize(repeating: 0.0)
                    innerCount = paramCount
                }
            }
            count = paramCount
        }

        for x in 0 ..< stateSize {
            let px = finalAmps[x].magnitudeSquared
            guard px > probabilityFloor else { continue }
            let inversePx = 1.0 / px

            for i in 0 ..< paramCount {
                let dLogPi = probGradients[i][x] * inversePx
                for j in i ..< paramCount {
                    let dLogPj = probGradients[j][x] * inversePx
                    let contribution = Double.fusedMultiplyAdd(dLogPi, dLogPj, 0.0) * px
                    fisherMatrix[i][j] += contribution
                    if i != j {
                        fisherMatrix[j][i] += contribution
                    }
                }
            }
        }

        return fisherMatrix
    }

    /// Compute the Fubini-Study metric tensor (quantum geometric tensor real part)
    ///
    /// Evaluates g_ij = Re⟨∂ᵢψ|∂ⱼψ⟩ - ⟨∂ᵢψ|ψ⟩⟨ψ|∂ⱼψ⟩ where |∂ᵢψ⟩ is the derivative
    /// of the parameterized state with respect to parameter θᵢ. The Fubini-Study metric
    /// encodes the geometry of the quantum state manifold and is used by the quantum natural
    /// gradient optimizer to precondition gradient updates, achieving parameter-space
    /// covariance that accelerates convergence on variational energy landscapes.
    ///
    /// Uses adjoint differentiation infrastructure to compute derivative states |∂ᵢψ⟩
    /// for all parameters simultaneously in O(p·L·2ⁿ) time, then constructs the p×p
    /// metric tensor from inner products. The result is symmetric positive semi-definite.
    ///
    /// - Parameters:
    ///   - circuit: Parameterized quantum circuit
    ///   - parameters: Concrete parameter values
    /// - Returns: Symmetric p×p Fubini-Study metric tensor
    /// - Precondition: parameters.count == circuit.parameterCount
    /// - Complexity: O(p·L·2ⁿ + p²·2ⁿ) time, O(p·2ⁿ) memory for derivative states
    /// - SeeAlso: ``fisherInformationMatrix(circuit:parameters:)`` for classical Fisher information
    /// - SeeAlso: ``QuantumNaturalGradientOptimizer`` for usage context
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.rotationY(Parameter(name: "a")), to: 0)
    /// circuit.append(.controlledNot, to: [0, 1])
    /// let g = GradientMethods.fubiniStudyMetric(circuit: circuit, parameters: [0.5])
    /// ```
    @_optimize(speed)
    @_eagerMove
    public static func fubiniStudyMetric(
        circuit: QuantumCircuit,
        parameters: [Double],
    ) -> [[Double]] {
        let paramList = circuit.parameters
        let paramCount = paramList.count
        ValidationUtilities.validateArrayCount(parameters, expected: paramCount, name: "parameters")

        guard paramCount > 0 else { return [] }

        let qubitCount = circuit.qubits
        let stateSize = 1 << qubitCount

        var paramNameToIndex: [String: Int] = [:]
        paramNameToIndex.reserveCapacity(paramCount)
        var parameterBindings: [String: Double] = [:]
        parameterBindings.reserveCapacity(paramCount)
        for i in 0 ..< paramCount {
            paramNameToIndex[paramList[i].name] = i
            parameterBindings[paramList[i].name] = parameters[i]
        }

        let boundCircuit = circuit.bound(with: parameters)
        let boundOps = boundCircuit.operations
        let originalOps = circuit.operations
        let opCount = boundOps.count

        var forwardStates: [[Complex<Double>]] = []
        forwardStates.reserveCapacity(opCount + 1)
        let initialAmps = [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            buffer.initialize(repeating: .zero)
            buffer[0] = .one
            count = stateSize
        }
        forwardStates.append(initialAmps)

        var currentState = QuantumState(qubits: qubitCount)
        for op in boundOps {
            currentState = GateApplication.apply(op, state: currentState)
            forwardStates.append(currentState.amplitudes)
        }

        let finalAmps = forwardStates[opCount]

        var derivativeStates = [[Complex<Double>]](repeating: [], count: paramCount)

        for k in 0 ..< paramCount {
            var derivAmps = [Complex<Double>](unsafeUninitializedCapacity: stateSize) {
                buffer, count in
                buffer.initialize(repeating: .zero)
                count = stateSize
            }

            for j in 0 ..< opCount {
                guard case let .gate(originalGate, qubits, _) = originalOps[j] else {
                    continue
                }

                let derivatives = computeGateDerivatives(
                    originalGate: originalGate,
                    qubits: qubits,
                    stateAfterAmplitudes: forwardStates[j + 1],
                    stateBeforeAmplitudes: forwardStates[j],
                    paramNameToIndex: paramNameToIndex,
                    parameterBindings: parameterBindings,
                    qubitCount: qubitCount,
                )

                for entry in derivatives where entry.parameterIndex == k {
                    derivAmps = addAmplitudes(
                        derivAmps,
                        scaleAmplitudes(entry.amplitudes, by: Complex(entry.signMultiplier, 0.0)),
                    )
                }

                if case let .gate(boundGate, boundQubits, _) = boundOps[j] {
                    derivAmps = applyGateMatrixToAmplitudes(
                        boundGate.matrix(),
                        qubits: boundQubits,
                        amplitudes: derivAmps,
                        qubitCount: qubitCount,
                    )
                }
            }

            derivativeStates[k] = derivAmps
        }

        var metric = [[Double]](unsafeUninitializedCapacity: paramCount) {
            buffer, count in
            for i in 0 ..< paramCount {
                buffer[i] = [Double](unsafeUninitializedCapacity: paramCount) { inner, innerCount in
                    inner.initialize(repeating: 0.0)
                    innerCount = paramCount
                }
            }
            count = paramCount
        }

        let psiOverlaps = [Complex<Double>](unsafeUninitializedCapacity: paramCount) { buffer, count in
            for k in 0 ..< paramCount {
                buffer[k] = complexInnerProduct(finalAmps, derivativeStates[k])
            }
            count = paramCount
        }

        for i in 0 ..< paramCount {
            for j in i ..< paramCount {
                let derivOverlap = complexInnerProduct(derivativeStates[i], derivativeStates[j])
                let psiProduct = psiOverlaps[i].conjugate * psiOverlaps[j]
                let gij = derivOverlap.real - psiProduct.real

                metric[i][j] = gij
                if i != j {
                    metric[j][i] = gij
                }
            }
        }

        return metric
    }

    // MARK: - Private Helpers

    /// Derivative entry for a single parameter contribution from a gate.
    private struct DerivativeEntry {
        let parameterIndex: Int
        let signMultiplier: Double
        let amplitudes: [Complex<Double>]
    }

    /// Compute derivative amplitudes for all parameters of a gate.
    @_optimize(speed)
    @_eagerMove
    private static func computeGateDerivatives(
        originalGate: QuantumGate,
        qubits: [Int],
        stateAfterAmplitudes: [Complex<Double>],
        stateBeforeAmplitudes: [Complex<Double>],
        paramNameToIndex: [String: Int],
        parameterBindings: [String: Double],
        qubitCount: Int,
    ) -> [DerivativeEntry] {
        switch originalGate {
        case let .rotationX(paramVal):
            guard let entry = singleParamEntry(paramVal, paramNameToIndex: paramNameToIndex) else { return [] }
            let deriv = applyRotationDerivative(.x, qubit: qubits[0], amplitudes: stateAfterAmplitudes, qubitCount: qubitCount)
            return [DerivativeEntry(parameterIndex: entry.index, signMultiplier: entry.sign, amplitudes: deriv)]

        case let .rotationY(paramVal):
            guard let entry = singleParamEntry(paramVal, paramNameToIndex: paramNameToIndex) else { return [] }
            let deriv = applyRotationDerivative(.y, qubit: qubits[0], amplitudes: stateAfterAmplitudes, qubitCount: qubitCount)
            return [DerivativeEntry(parameterIndex: entry.index, signMultiplier: entry.sign, amplitudes: deriv)]

        case let .rotationZ(paramVal):
            guard let entry = singleParamEntry(paramVal, paramNameToIndex: paramNameToIndex) else { return [] }
            let deriv = applyRotationDerivative(.z, qubit: qubits[0], amplitudes: stateAfterAmplitudes, qubitCount: qubitCount)
            return [DerivativeEntry(parameterIndex: entry.index, signMultiplier: entry.sign, amplitudes: deriv)]

        case let .phase(paramVal), let .u1(lambda: paramVal):
            guard let entry = singleParamEntry(paramVal, paramNameToIndex: paramNameToIndex) else { return [] }
            let deriv = applyPhaseDerivative(qubit: qubits[0], amplitudes: stateAfterAmplitudes, qubitCount: qubitCount)
            return [DerivativeEntry(parameterIndex: entry.index, signMultiplier: entry.sign, amplitudes: deriv)]

        case let .globalPhase(paramVal):
            guard let entry = singleParamEntry(paramVal, paramNameToIndex: paramNameToIndex) else { return [] }
            let deriv = scaleAmplitudes(stateAfterAmplitudes, by: .i)
            return [DerivativeEntry(parameterIndex: entry.index, signMultiplier: entry.sign, amplitudes: deriv)]

        case let .controlledRotationX(paramVal):
            guard let entry = singleParamEntry(paramVal, paramNameToIndex: paramNameToIndex) else { return [] }
            let deriv = applyControlledRotationDerivative(.x, control: qubits[0], target: qubits[1], amplitudes: stateAfterAmplitudes, qubitCount: qubitCount)
            return [DerivativeEntry(parameterIndex: entry.index, signMultiplier: entry.sign, amplitudes: deriv)]

        case let .controlledRotationY(paramVal):
            guard let entry = singleParamEntry(paramVal, paramNameToIndex: paramNameToIndex) else { return [] }
            let deriv = applyControlledRotationDerivative(.y, control: qubits[0], target: qubits[1], amplitudes: stateAfterAmplitudes, qubitCount: qubitCount)
            return [DerivativeEntry(parameterIndex: entry.index, signMultiplier: entry.sign, amplitudes: deriv)]

        case let .controlledRotationZ(paramVal):
            guard let entry = singleParamEntry(paramVal, paramNameToIndex: paramNameToIndex) else { return [] }
            let deriv = applyControlledRotationDerivative(.z, control: qubits[0], target: qubits[1], amplitudes: stateAfterAmplitudes, qubitCount: qubitCount)
            return [DerivativeEntry(parameterIndex: entry.index, signMultiplier: entry.sign, amplitudes: deriv)]

        case let .controlledPhase(paramVal):
            guard let entry = singleParamEntry(paramVal, paramNameToIndex: paramNameToIndex) else { return [] }
            let deriv = applyControlledPhaseDerivative(control: qubits[0], target: qubits[1], amplitudes: stateAfterAmplitudes, qubitCount: qubitCount)
            return [DerivativeEntry(parameterIndex: entry.index, signMultiplier: entry.sign, amplitudes: deriv)]

        case let .xx(paramVal):
            guard let entry = singleParamEntry(paramVal, paramNameToIndex: paramNameToIndex) else { return [] }
            let deriv = applyTwoQubitPauliDerivative(.x, qubit0: qubits[0], qubit1: qubits[1], amplitudes: stateAfterAmplitudes, qubitCount: qubitCount)
            return [DerivativeEntry(parameterIndex: entry.index, signMultiplier: entry.sign, amplitudes: deriv)]

        case let .yy(paramVal):
            guard let entry = singleParamEntry(paramVal, paramNameToIndex: paramNameToIndex) else { return [] }
            let deriv = applyTwoQubitPauliDerivative(.y, qubit0: qubits[0], qubit1: qubits[1], amplitudes: stateAfterAmplitudes, qubitCount: qubitCount)
            return [DerivativeEntry(parameterIndex: entry.index, signMultiplier: entry.sign, amplitudes: deriv)]

        case let .zz(paramVal):
            guard let entry = singleParamEntry(paramVal, paramNameToIndex: paramNameToIndex) else { return [] }
            let deriv = applyTwoQubitPauliDerivative(.z, qubit0: qubits[0], qubit1: qubits[1], amplitudes: stateAfterAmplitudes, qubitCount: qubitCount)
            return [DerivativeEntry(parameterIndex: entry.index, signMultiplier: entry.sign, amplitudes: deriv)]

        case let .givens(paramVal):
            guard let entry = singleParamEntry(paramVal, paramNameToIndex: paramNameToIndex) else { return [] }
            let deriv = applyGivensDerivative(qubit0: qubits[0], qubit1: qubits[1], amplitudes: stateAfterAmplitudes, qubitCount: qubitCount)
            return [DerivativeEntry(parameterIndex: entry.index, signMultiplier: entry.sign, amplitudes: deriv)]

        case let .u2(phi, lambda):
            return computeU2Derivatives(
                phi: phi, lambda: lambda, qubits: qubits,
                stateBeforeAmplitudes: stateBeforeAmplitudes,
                paramNameToIndex: paramNameToIndex, bindings: parameterBindings, qubitCount: qubitCount,
            )

        case let .u3(theta, phi, lambda):
            return computeU3Derivatives(
                theta: theta, phi: phi, lambda: lambda, qubits: qubits,
                stateBeforeAmplitudes: stateBeforeAmplitudes,
                paramNameToIndex: paramNameToIndex, bindings: parameterBindings, qubitCount: qubitCount,
            )

        default:
            return []
        }
    }

    /// Extract parameter index and sign from a ParameterValue.
    @inline(__always)
    private static func singleParamEntry(
        _ paramVal: ParameterValue,
        paramNameToIndex: [String: Int],
    ) -> (index: Int, sign: Double)? {
        switch paramVal {
        case let .parameter(p):
            // Safe: paramNameToIndex is built from circuit.parameters which includes all gate parameters
            (paramNameToIndex[p.name]!, 1.0)
        case let .negatedParameter(p):
            // Safe: paramNameToIndex is built from circuit.parameters which includes all gate parameters
            (paramNameToIndex[p.name]!, -1.0)
        case .value, .expression:
            nil
        }
    }

    /// Apply rotation gate derivative: (∂U/∂θ)|φ⟩ = (-iσ/2)·U(θ)|φ⟩ = (-iσ/2)·|ψ_after⟩.
    @_optimize(speed)
    @_eagerMove
    private static func applyRotationDerivative(
        _ basis: PauliBasis,
        qubit: Int,
        amplitudes: [Complex<Double>],
        qubitCount: Int,
    ) -> [Complex<Double>] {
        let stateSize = 1 << qubitCount
        let bitMask = 1 << qubit
        let negIHalf = Complex<Double>(0.0, -0.5)

        return [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            switch basis {
            case .x:
                for i in 0 ..< stateSize where (i & bitMask) == 0 {
                    let j = i | bitMask
                    buffer[i] = negIHalf * amplitudes[j]
                    buffer[j] = negIHalf * amplitudes[i]
                }
            case .y:
                for i in 0 ..< stateSize where (i & bitMask) == 0 {
                    let j = i | bitMask
                    let halfReal = Complex<Double>(-0.5, 0.0)
                    buffer[i] = halfReal * amplitudes[j]
                    buffer[j] = Complex(0.5, 0.0) * amplitudes[i]
                }
            case .z:
                for i in 0 ..< stateSize where (i & bitMask) == 0 {
                    let j = i | bitMask
                    buffer[i] = negIHalf * amplitudes[i]
                    buffer[j] = Complex(0.0, 0.5) * amplitudes[j]
                }
            }
            count = stateSize
        }
    }

    /// Apply Phase gate derivative: (∂Phase/∂θ)|ψ_after⟩ projects |1⟩ and scales by i.
    @_optimize(speed)
    @_eagerMove
    private static func applyPhaseDerivative(
        qubit: Int,
        amplitudes: [Complex<Double>],
        qubitCount: Int,
    ) -> [Complex<Double>] {
        let stateSize = 1 << qubitCount
        let bitMask = 1 << qubit

        return [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            for i in 0 ..< stateSize {
                if (i & bitMask) != 0 {
                    buffer[i] = Complex(-amplitudes[i].imaginary, amplitudes[i].real)
                } else {
                    buffer[i] = .zero
                }
            }
            count = stateSize
        }
    }

    /// Apply controlled rotation derivative: zero for control=|0⟩, rotation derivative for control=|1⟩.
    @_optimize(speed)
    @_eagerMove
    private static func applyControlledRotationDerivative(
        _ basis: PauliBasis,
        control: Int,
        target: Int,
        amplitudes: [Complex<Double>],
        qubitCount: Int,
    ) -> [Complex<Double>] {
        let stateSize = 1 << qubitCount
        let controlMask = 1 << control
        let targetMask = 1 << target
        let negIHalf = Complex<Double>(0.0, -0.5)

        return [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            buffer.initialize(repeating: .zero)
            for i in 0 ..< stateSize where (i & controlMask) != 0 {
                let targetBit = i & targetMask
                switch basis {
                case .x:
                    let partner = i ^ targetMask
                    buffer[i] = negIHalf * amplitudes[partner]
                case .y:
                    let partner = i ^ targetMask
                    if targetBit == 0 {
                        buffer[i] = Complex(-0.5, 0.0) * amplitudes[partner]
                    } else {
                        buffer[i] = Complex(0.5, 0.0) * amplitudes[partner]
                    }
                case .z:
                    if targetBit == 0 {
                        buffer[i] = negIHalf * amplitudes[i]
                    } else {
                        buffer[i] = Complex(0.0, 0.5) * amplitudes[i]
                    }
                }
            }
            count = stateSize
        }
    }

    /// Apply controlled phase derivative: zero for control=|0⟩ or target=|0⟩, i for both=|1⟩.
    @_optimize(speed)
    @_eagerMove
    private static func applyControlledPhaseDerivative(
        control: Int,
        target: Int,
        amplitudes: [Complex<Double>],
        qubitCount: Int,
    ) -> [Complex<Double>] {
        let stateSize = 1 << qubitCount
        let bothMask = (1 << control) | (1 << target)

        return [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            for i in 0 ..< stateSize {
                if (i & bothMask) == bothMask {
                    buffer[i] = Complex(-amplitudes[i].imaginary, amplitudes[i].real)
                } else {
                    buffer[i] = .zero
                }
            }
            count = stateSize
        }
    }

    /// Apply two-qubit Pauli tensor derivative: (-i)·(σ⊗σ)·|ψ_after⟩.
    @_optimize(speed)
    @_eagerMove
    private static func applyTwoQubitPauliDerivative(
        _ basis: PauliBasis,
        qubit0: Int,
        qubit1: Int,
        amplitudes: [Complex<Double>],
        qubitCount: Int,
    ) -> [Complex<Double>] {
        var result = applySinglePauliToAmplitudes(basis, qubit: qubit0, amplitudes: amplitudes, qubitCount: qubitCount)
        result = applySinglePauliToAmplitudes(basis, qubit: qubit1, amplitudes: result, qubitCount: qubitCount)
        return scaleAmplitudes(result, by: Complex(0.0, -1.0))
    }

    /// Apply Givens rotation derivative using its generator (|10⟩⟨01| - |01⟩⟨10|).
    @_optimize(speed)
    @_eagerMove
    private static func applyGivensDerivative(
        qubit0: Int,
        qubit1: Int,
        amplitudes: [Complex<Double>],
        qubitCount: Int,
    ) -> [Complex<Double>] {
        let stateSize = 1 << qubitCount
        let mask0 = 1 << qubit0
        let mask1 = 1 << qubit1
        let bothMask = mask0 | mask1

        return [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            buffer.initialize(repeating: .zero)
            for i in 0 ..< stateSize where (i & bothMask) == 0 {
                let i01 = i | mask1
                let i10 = i | mask0
                buffer[i10] = amplitudes[i01]
                buffer[i01] = -amplitudes[i10]
            }
            count = stateSize
        }
    }

    /// Compute U2 derivatives for phi and lambda parameters.
    @_optimize(speed)
    @_eagerMove
    private static func computeU2Derivatives(
        phi: ParameterValue,
        lambda: ParameterValue,
        qubits: [Int],
        stateBeforeAmplitudes: [Complex<Double>],
        paramNameToIndex: [String: Int],
        bindings: [String: Double],
        qubitCount: Int,
    ) -> [DerivativeEntry] {
        var entries: [DerivativeEntry] = []
        entries.reserveCapacity(2)

        let phiVal = phi.evaluate(using: bindings)
        let lambdaVal = lambda.evaluate(using: bindings)
        let invSqrt2 = 1.0 / 2.0.squareRoot()

        if let phiEntry = singleParamEntry(phi, paramNameToIndex: paramNameToIndex) {
            let dUdPhi: [[Complex<Double>]] = [
                [.zero, .zero],
                [Complex(0.0, invSqrt2) * Complex(phase: phiVal), Complex(0.0, invSqrt2) * Complex(phase: phiVal + lambdaVal)],
            ]
            let deriv = applySingleQubitMatrixToAmplitudes(dUdPhi, qubit: qubits[0], amplitudes: stateBeforeAmplitudes, qubitCount: qubitCount)
            entries.append(DerivativeEntry(parameterIndex: phiEntry.index, signMultiplier: phiEntry.sign, amplitudes: deriv))
        }

        if let lambdaEntry = singleParamEntry(lambda, paramNameToIndex: paramNameToIndex) {
            let dUdLambda: [[Complex<Double>]] = [
                [.zero, Complex(0.0, -invSqrt2) * Complex(phase: lambdaVal)],
                [.zero, Complex(0.0, invSqrt2) * Complex(phase: phiVal + lambdaVal)],
            ]
            let deriv = applySingleQubitMatrixToAmplitudes(dUdLambda, qubit: qubits[0], amplitudes: stateBeforeAmplitudes, qubitCount: qubitCount)
            entries.append(DerivativeEntry(parameterIndex: lambdaEntry.index, signMultiplier: lambdaEntry.sign, amplitudes: deriv))
        }

        return entries
    }

    /// Compute U3 derivatives for theta, phi, and lambda parameters.
    @_optimize(speed)
    @_eagerMove
    private static func computeU3Derivatives(
        theta: ParameterValue,
        phi: ParameterValue,
        lambda: ParameterValue,
        qubits: [Int],
        stateBeforeAmplitudes: [Complex<Double>],
        paramNameToIndex: [String: Int],
        bindings: [String: Double],
        qubitCount: Int,
    ) -> [DerivativeEntry] {
        var entries: [DerivativeEntry] = []
        entries.reserveCapacity(3)

        let thetaVal = theta.evaluate(using: bindings)
        let phiVal = phi.evaluate(using: bindings)
        let lambdaVal = lambda.evaluate(using: bindings)
        let halfTheta = thetaVal / 2.0
        let cosHalf = cos(halfTheta)
        let sinHalf = sin(halfTheta)

        if let thetaEntry = singleParamEntry(theta, paramNameToIndex: paramNameToIndex) {
            let dUdTheta: [[Complex<Double>]] = [
                [Complex(-sinHalf / 2.0, 0.0), Complex(phase: lambdaVal) * Complex(-cosHalf / 2.0, 0.0)],
                [Complex(phase: phiVal) * Complex(cosHalf / 2.0, 0.0), Complex(phase: phiVal + lambdaVal) * Complex(-sinHalf / 2.0, 0.0)],
            ]
            let deriv = applySingleQubitMatrixToAmplitudes(dUdTheta, qubit: qubits[0], amplitudes: stateBeforeAmplitudes, qubitCount: qubitCount)
            entries.append(DerivativeEntry(parameterIndex: thetaEntry.index, signMultiplier: thetaEntry.sign, amplitudes: deriv))
        }

        if let phiEntry = singleParamEntry(phi, paramNameToIndex: paramNameToIndex) {
            let dUdPhi: [[Complex<Double>]] = [
                [.zero, .zero],
                [Complex(0.0, 1.0) * Complex(phase: phiVal) * Complex(sinHalf, 0.0), Complex(0.0, 1.0) * Complex(phase: phiVal + lambdaVal) * Complex(cosHalf, 0.0)],
            ]
            let deriv = applySingleQubitMatrixToAmplitudes(dUdPhi, qubit: qubits[0], amplitudes: stateBeforeAmplitudes, qubitCount: qubitCount)
            entries.append(DerivativeEntry(parameterIndex: phiEntry.index, signMultiplier: phiEntry.sign, amplitudes: deriv))
        }

        if let lambdaEntry = singleParamEntry(lambda, paramNameToIndex: paramNameToIndex) {
            let dUdLambda: [[Complex<Double>]] = [
                [.zero, Complex(0.0, -1.0) * Complex(phase: lambdaVal) * Complex(sinHalf, 0.0)],
                [.zero, Complex(0.0, 1.0) * Complex(phase: phiVal + lambdaVal) * Complex(cosHalf, 0.0)],
            ]
            let deriv = applySingleQubitMatrixToAmplitudes(dUdLambda, qubit: qubits[0], amplitudes: stateBeforeAmplitudes, qubitCount: qubitCount)
            entries.append(DerivativeEntry(parameterIndex: lambdaEntry.index, signMultiplier: lambdaEntry.sign, amplitudes: deriv))
        }

        return entries
    }

    /// Apply a single Pauli operator to amplitude array.
    @_optimize(speed)
    @_eagerMove
    private static func applySinglePauliToAmplitudes(
        _ basis: PauliBasis,
        qubit: Int,
        amplitudes: [Complex<Double>],
        qubitCount: Int,
    ) -> [Complex<Double>] {
        let stateSize = 1 << qubitCount
        let bitMask = 1 << qubit

        return [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            switch basis {
            case .x:
                for i in 0 ..< stateSize where (i & bitMask) == 0 {
                    let j = i | bitMask
                    buffer[i] = amplitudes[j]
                    buffer[j] = amplitudes[i]
                }
            case .y:
                for i in 0 ..< stateSize where (i & bitMask) == 0 {
                    let j = i | bitMask
                    buffer[i] = Complex(amplitudes[j].imaginary, -amplitudes[j].real)
                    buffer[j] = Complex(-amplitudes[i].imaginary, amplitudes[i].real)
                }
            case .z:
                for i in 0 ..< stateSize where (i & bitMask) == 0 {
                    let j = i | bitMask
                    buffer[i] = amplitudes[i]
                    buffer[j] = -amplitudes[j]
                }
            }
            count = stateSize
        }
    }

    /// Apply observable to amplitude array: O|ψ⟩ = Σᵢ cᵢ·Pᵢ|ψ⟩.
    @_optimize(speed)
    @_eagerMove
    private static func applyObservableToAmplitudes(
        _ observable: Observable,
        amplitudes: [Complex<Double>],
        qubitCount: Int,
    ) -> [Complex<Double>] {
        let stateSize = 1 << qubitCount
        guard !observable.terms.isEmpty else {
            return [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
                buffer.initialize(repeating: .zero)
                count = stateSize
            }
        }

        var result = [Complex<Double>](unsafeUninitializedCapacity: stateSize) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = stateSize
        }

        for term in observable.terms {
            var termAmps = amplitudes
            for op in term.pauliString.operators {
                termAmps = applySinglePauliToAmplitudes(op.basis, qubit: op.qubit, amplitudes: termAmps, qubitCount: qubitCount)
            }
            let coeff = Complex<Double>(term.coefficient, 0.0)
            for i in 0 ..< stateSize {
                result[i] = result[i] + coeff * termAmps[i]
            }
        }

        return result
    }

    /// Apply gate matrix to raw amplitude array.
    @_optimize(speed)
    @_eagerMove
    private static func applyGateMatrixToAmplitudes(
        _ matrix: [[Complex<Double>]],
        qubits: [Int],
        amplitudes: [Complex<Double>],
        qubitCount: Int,
    ) -> [Complex<Double>] {
        let gateSize = matrix.count
        if gateSize == 2, qubits.count == 1 {
            return applySingleQubitMatrixToAmplitudes(matrix, qubit: qubits[0], amplitudes: amplitudes, qubitCount: qubitCount)
        }
        if gateSize == 4, qubits.count == 2 {
            return applyTwoQubitMatrixToAmplitudes(matrix, qubit0: qubits[0], qubit1: qubits[1], amplitudes: amplitudes, qubitCount: qubitCount)
        }
        return applyMultiQubitMatrixToAmplitudes(matrix, qubits: qubits, amplitudes: amplitudes, qubitCount: qubitCount)
    }

    /// Apply 2×2 matrix to single qubit in amplitude array.
    @_optimize(speed)
    @_eagerMove
    private static func applySingleQubitMatrixToAmplitudes(
        _ matrix: [[Complex<Double>]],
        qubit: Int,
        amplitudes: [Complex<Double>],
        qubitCount: Int,
    ) -> [Complex<Double>] {
        let stateSize = 1 << qubitCount
        let bitMask = 1 << qubit
        let g00 = matrix[0][0], g01 = matrix[0][1]
        let g10 = matrix[1][0], g11 = matrix[1][1]

        return [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            for i in 0 ..< stateSize where (i & bitMask) == 0 {
                let j = i | bitMask
                let ci = amplitudes[i], cj = amplitudes[j]
                buffer[i] = g00 * ci + g01 * cj
                buffer[j] = g10 * ci + g11 * cj
            }
            count = stateSize
        }
    }

    /// Apply 4×4 matrix to two qubits in amplitude array.
    @_optimize(speed)
    @_eagerMove
    private static func applyTwoQubitMatrixToAmplitudes(
        _ matrix: [[Complex<Double>]],
        qubit0: Int,
        qubit1: Int,
        amplitudes: [Complex<Double>],
        qubitCount: Int,
    ) -> [Complex<Double>] {
        let stateSize = 1 << qubitCount
        let mask0 = 1 << qubit0, mask1 = 1 << qubit1
        let bothMask = mask0 | mask1

        return [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            for i in 0 ..< stateSize where (i & bothMask) == 0 {
                let i01 = i | mask1
                let i10 = i | mask0
                let i11 = i | bothMask
                let c0 = amplitudes[i], c1 = amplitudes[i01]
                let c2 = amplitudes[i10], c3 = amplitudes[i11]

                buffer[i] = matrix[0][0] * c0 + matrix[0][1] * c1 + matrix[0][2] * c2 + matrix[0][3] * c3
                buffer[i01] = matrix[1][0] * c0 + matrix[1][1] * c1 + matrix[1][2] * c2 + matrix[1][3] * c3
                buffer[i10] = matrix[2][0] * c0 + matrix[2][1] * c1 + matrix[2][2] * c2 + matrix[2][3] * c3
                buffer[i11] = matrix[3][0] * c0 + matrix[3][1] * c1 + matrix[3][2] * c2 + matrix[3][3] * c3
            }
            count = stateSize
        }
    }

    /// Apply arbitrary multi-qubit matrix to amplitude array.
    @_optimize(speed)
    @_eagerMove
    private static func applyMultiQubitMatrixToAmplitudes(
        _ matrix: [[Complex<Double>]],
        qubits: [Int],
        amplitudes: [Complex<Double>],
        qubitCount: Int,
    ) -> [Complex<Double>] {
        let stateSize = 1 << qubitCount
        let gateSize = matrix.count
        let targetCount = qubits.count
        let masks = qubits.map { 1 << $0 }

        return [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            buffer.initialize(repeating: .zero)
            for i in 0 ..< stateSize {
                var rowBits = 0
                for idx in 0 ..< targetCount {
                    if (i & masks[idx]) != 0 {
                        rowBits |= (1 << idx)
                    }
                }
                for col in 0 ..< gateSize {
                    let element = matrix[rowBits][col]
                    if element.real == 0.0, element.imaginary == 0.0 { continue }
                    var sourceIndex = i
                    for idx in 0 ..< targetCount {
                        let colBit = (col >> idx) & 1
                        if colBit == 1 {
                            sourceIndex |= masks[idx]
                        } else {
                            sourceIndex &= ~masks[idx]
                        }
                    }
                    buffer[i] = buffer[i] + element * amplitudes[sourceIndex]
                }
            }
            count = stateSize
        }
    }

    /// Compute complex inner product ⟨bra|ket⟩ = Σᵢ conj(braᵢ)·ketᵢ.
    @_optimize(speed)
    @_effects(readonly)
    private static func complexInnerProduct(
        _ bra: [Complex<Double>],
        _ ket: [Complex<Double>],
    ) -> Complex<Double> {
        let n = bra.count
        var realPart = 0.0
        var imagPart = 0.0

        for i in 0 ..< n {
            realPart = Double.fusedMultiplyAdd(bra[i].real, ket[i].real, realPart)
            realPart = Double.fusedMultiplyAdd(bra[i].imaginary, ket[i].imaginary, realPart)
            imagPart = Double.fusedMultiplyAdd(bra[i].real, ket[i].imaginary, imagPart)
            imagPart = Double.fusedMultiplyAdd(-bra[i].imaginary, ket[i].real, imagPart)
        }

        return Complex(realPart, imagPart)
    }

    /// Scale all amplitudes by a complex scalar.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func scaleAmplitudes(
        _ amplitudes: [Complex<Double>],
        by scalar: Complex<Double>,
    ) -> [Complex<Double>] {
        [Complex<Double>](unsafeUninitializedCapacity: amplitudes.count) { buffer, count in
            for i in 0 ..< amplitudes.count {
                buffer[i] = scalar * amplitudes[i]
            }
            count = amplitudes.count
        }
    }

    /// Element-wise addition of two amplitude arrays.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func addAmplitudes(
        _ lhs: [Complex<Double>],
        _ rhs: [Complex<Double>],
    ) -> [Complex<Double>] {
        let n = lhs.count
        return [Complex<Double>](unsafeUninitializedCapacity: n) { buffer, count in
            for i in 0 ..< n {
                buffer[i] = lhs[i] + rhs[i]
            }
            count = n
        }
    }
}
