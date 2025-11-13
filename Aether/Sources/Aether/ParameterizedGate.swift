// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Parameterized quantum gate for variational algorithms
///
/// Extends quantum gates to support symbolic parameters that can be bound at
/// execution time. Essential for variational quantum algorithms (VQE, QAOA)
/// where gate parameters are optimized by classical optimizers.
///
/// **Architecture**:
/// - Parameterized variants of rotation and phase gates (Rx, Ry, Rz, Phase, U1, U2, U3)
/// - Controlled parameterized gates (CRx, CRy, CRz, CPhase)
/// - Non-parameterized gates wrapped as `.concrete(QuantumGate)`
/// - Symbolic or concrete parameters via `ParameterExpression`
///
/// **Usage in variational algorithms**:
/// - VQE hardware-efficient ansatz: layers of Ry(θᵢ) + CNOT chains
/// - QAOA: exp(-iγH_p) as Rz(2γ) chains, exp(-iβH_m) as Rx(2β)
/// - Quantum ML: parameterized feature maps and variational classifiers
///
/// **Parameter binding workflow**:
/// 1. Build circuit with symbolic parameters
/// 2. Optimizer proposes parameter values
/// 3. Bind parameters to create concrete `QuantumCircuit`
/// 4. Execute circuit and measure expectation value
/// 5. Optimizer updates parameters based on measurement
///
/// Example:
/// ```swift
/// // Create parameterized gates
/// let theta = Parameter(name: "theta")
/// let phi = Parameter(name: "phi")
///
/// let ry = ParameterizedGate.rotationY(theta: .parameter(theta))
/// let rz = ParameterizedGate.rotationZ(theta: .parameter(phi))
///
/// // Mix parameterized and concrete gates
/// let h = ParameterizedGate.concrete(.hadamard)
/// let cnot = ParameterizedGate.concrete(.cnot(control: 0, target: 1))
///
/// // Build parameterized circuit
/// var circuit = ParameterizedQuantumCircuit(numQubits: 2)
/// circuit.append(gate: h, toQubit: 0)
/// circuit.append(gate: ry, toQubit: 0)
/// circuit.append(gate: cnot, qubits: [])
/// circuit.append(gate: rz, toQubit: 1)
///
/// // Bind parameters to execute
/// let bindings = ["theta": Double.pi / 4, "phi": Double.pi / 8]
/// let concrete = try circuit.bind(parameters: bindings)
/// let state = concrete.execute()
/// ```
@frozen
public enum ParameterizedGate: Equatable, Hashable, Sendable, CustomStringConvertible {
    // MARK: - Single-Qubit Parameterized Gates

    /// Phase gate with symbolic or concrete angle
    /// P(θ) = diag(1, e^(iθ))
    case phase(theta: ParameterExpression)

    /// Rotation around X-axis with symbolic or concrete angle
    /// Rx(θ) = [[cos(θ/2), -i·sin(θ/2)], [-i·sin(θ/2), cos(θ/2)]]
    case rotationX(theta: ParameterExpression)

    /// Rotation around Y-axis with symbolic or concrete angle
    /// Ry(θ) = [[cos(θ/2), -sin(θ/2)], [sin(θ/2), cos(θ/2)]]
    case rotationY(theta: ParameterExpression)

    /// Rotation around Z-axis with symbolic or concrete angle
    /// Rz(θ) = diag(e^(-iθ/2), e^(iθ/2))
    case rotationZ(theta: ParameterExpression)

    /// IBM U1 gate - single-parameter phase gate
    /// U1(λ) = diag(1, e^(iλ))
    case u1(lambda: ParameterExpression)

    /// IBM U2 gate - two-parameter gate creating superposition with phases
    /// U2(φ,λ) = Rz(φ)·Ry(π/2)·Rz(λ)
    case u2(phi: ParameterExpression, lambda: ParameterExpression)

    /// IBM U3 gate - three-parameter universal single-qubit gate
    /// U3(θ,φ,λ) most general single-qubit rotation
    case u3(theta: ParameterExpression, phi: ParameterExpression, lambda: ParameterExpression)

    // MARK: - Two-Qubit Parameterized Gates

    /// Controlled-Phase gate with symbolic or concrete angle
    /// Applies phase e^(iθ) when both qubits are |1⟩
    case controlledPhase(theta: ParameterExpression, control: Int, target: Int)

    /// Controlled rotation around X-axis
    /// Applies Rx(θ) to target when control is |1⟩
    case controlledRotationX(theta: ParameterExpression, control: Int, target: Int)

    /// Controlled rotation around Y-axis
    /// Applies Ry(θ) to target when control is |1⟩
    case controlledRotationY(theta: ParameterExpression, control: Int, target: Int)

    /// Controlled rotation around Z-axis
    /// Applies Rz(θ) to target when control is |1⟩
    case controlledRotationZ(theta: ParameterExpression, control: Int, target: Int)

    // MARK: - Non-Parameterized Gates

    /// Concrete quantum gate (non-parameterized)
    /// Wraps any non-parameterized gate from QuantumGate enum
    case concrete(QuantumGate)

    // MARK: - Gate Properties

    /// Number of qubits this gate operates on
    @inlinable
    public var qubitsRequired: Int {
        switch self {
        case .phase, .rotationX, .rotationY, .rotationZ, .u1, .u2, .u3: 1
        case .controlledPhase, .controlledRotationX, .controlledRotationY, .controlledRotationZ: 2
        case let .concrete(gate): gate.qubitsRequired
        }
    }

    /// Whether gate has symbolic parameters (unbound)
    @inlinable
    public var isParameterized: Bool { !parameters().isEmpty }

    /// Extract all symbolic parameters from gate
    /// - Returns: Set of symbolic parameters used in this gate
    @_optimize(speed)
    @inlinable
    @_effects(readonly)
    public func parameters() -> Set<Parameter> {
        var params = Set<Parameter>()

        switch self {
        case let .phase(theta),
             let .rotationX(theta),
             let .rotationY(theta),
             let .rotationZ(theta),
             let .u1(theta):
            if let p = theta.extractParameter() { params.insert(p) }

        case let .u2(phi, lambda):
            if let p = phi.extractParameter() { params.insert(p) }
            if let p = lambda.extractParameter() { params.insert(p) }

        case let .u3(theta, phi, lambda):
            if let p = theta.extractParameter() { params.insert(p) }
            if let p = phi.extractParameter() { params.insert(p) }
            if let p = lambda.extractParameter() { params.insert(p) }

        case let .controlledPhase(theta, _, _),
             let .controlledRotationX(theta, _, _),
             let .controlledRotationY(theta, _, _),
             let .controlledRotationZ(theta, _, _):
            if let p = theta.extractParameter() { params.insert(p) }

        case .concrete: break
        }

        return params
    }

    // MARK: - Parameter Binding

    /// Bind parameters to create concrete QuantumGate
    ///
    /// Substitutes all symbolic parameters with values from bindings dictionary.
    /// Produces executable `QuantumGate` that can be used in standard circuits.
    ///
    /// - Parameter bindings: Dictionary mapping parameter names to values
    /// - Returns: Concrete quantum gate with all parameters bound
    /// - Throws: ParameterError.unboundParameter if any parameter missing
    ///
    /// Example:
    /// ```swift
    /// let theta = Parameter(name: "theta")
    /// let gate = ParameterizedGate.rotationY(theta: .parameter(theta))
    ///
    /// let bindings = ["theta": Double.pi / 4]
    /// let concrete = try gate.bind(with: bindings)
    /// // concrete is QuantumGate.rotationY(theta: π/4)
    /// ```
    @_optimize(speed)
    @inlinable
    @_eagerMove
    public func bind(with bindings: [String: Double]) throws -> QuantumGate {
        switch self {
        case let .phase(theta):
            let value = try theta.evaluate(with: bindings)
            return .phase(theta: value)

        case let .rotationX(theta):
            let value = try theta.evaluate(with: bindings)
            return .rotationX(theta: value)

        case let .rotationY(theta):
            let value = try theta.evaluate(with: bindings)
            return .rotationY(theta: value)

        case let .rotationZ(theta):
            let value = try theta.evaluate(with: bindings)
            return .rotationZ(theta: value)

        case let .u1(lambda):
            let lambdaValue = try lambda.evaluate(with: bindings)
            return .u1(lambda: lambdaValue)

        case let .u2(phi, lambda):
            let phiValue = try phi.evaluate(with: bindings)
            let lambdaValue = try lambda.evaluate(with: bindings)
            return .u2(phi: phiValue, lambda: lambdaValue)

        case let .u3(theta, phi, lambda):
            let thetaValue = try theta.evaluate(with: bindings)
            let phiValue = try phi.evaluate(with: bindings)
            let lambdaValue = try lambda.evaluate(with: bindings)
            return .u3(theta: thetaValue, phi: phiValue, lambda: lambdaValue)

        case let .controlledPhase(theta, control, target):
            let value = try theta.evaluate(with: bindings)
            return .controlledPhase(theta: value, control: control, target: target)

        case let .controlledRotationX(theta, control, target):
            let value = try theta.evaluate(with: bindings)
            return .controlledRotationX(theta: value, control: control, target: target)

        case let .controlledRotationY(theta, control, target):
            let value = try theta.evaluate(with: bindings)
            return .controlledRotationY(theta: value, control: control, target: target)

        case let .controlledRotationZ(theta, control, target):
            let value = try theta.evaluate(with: bindings)
            return .controlledRotationZ(theta: value, control: control, target: target)

        case let .concrete(gate):
            return gate
        }
    }

    // MARK: - Validation

    /// Validate qubit indices for gate application
    ///
    /// Checks that all qubit indices are within bounds and distinct.
    /// Delegates to underlying QuantumGate validation for concrete gates.
    ///
    /// - Parameter maxAllowedQubit: Maximum allowed qubit index (inclusive)
    /// - Returns: True if all qubit indices are valid
    @inlinable
    @_effects(readonly)
    public func validateQubitIndices(maxAllowedQubit: Int) -> Bool {
        switch self {
        case .phase, .rotationX, .rotationY, .rotationZ, .u1, .u2, .u3: true

        case let .controlledPhase(_, control, target),
             let .controlledRotationX(_, control, target),
             let .controlledRotationY(_, control, target),
             let .controlledRotationZ(_, control, target):
            control != target &&
                control >= 0 && control <= maxAllowedQubit &&
                target >= 0 && target <= maxAllowedQubit

        case let .concrete(gate):
            gate.validateQubitIndices(maxAllowedQubit: maxAllowedQubit)
        }
    }

    // MARK: - CustomStringConvertible

    /// String representation of parameterized gate
    public var description: String {
        switch self {
        case let .phase(theta): "P(\(theta))"
        case let .rotationX(theta): "Rx(\(theta))"
        case let .rotationY(theta): "Ry(\(theta))"
        case let .rotationZ(theta): "Rz(\(theta))"
        case let .u1(lambda): "U1(\(lambda))"
        case let .u2(phi, lambda): "U2(\(phi), \(lambda))"
        case let .u3(theta, phi, lambda): "U3(\(theta), \(phi), \(lambda))"
        case let .controlledPhase(theta, control, target): "CP(\(theta), c:\(control), t:\(target))"
        case let .controlledRotationX(theta, control, target): "CRx(\(theta), c:\(control), t:\(target))"
        case let .controlledRotationY(theta, control, target): "CRy(\(theta), c:\(control), t:\(target))"
        case let .controlledRotationZ(theta, control, target): "CRz(\(theta), c:\(control), t:\(target))"
        case let .concrete(gate): gate.description
        }
    }
}
