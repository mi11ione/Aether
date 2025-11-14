// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Hardware-efficient ansatz builder for variational quantum algorithms
///
/// Constructs parameterized quantum circuits optimized for near-term quantum hardware.
/// This is the **PRIMARY ansatz** for general VQE problems, as specified in the roadmap.
///
/// **Design Philosophy:**
/// - Layers of single-qubit rotations followed by entangling gates
/// - Minimal circuit depth (NISQ-compatible)
/// - Works for any Hamiltonian without domain-specific structure
/// - Expressivity scales with depth: more layers = more parameters = better approximation
///
/// **Architecture:**
/// Each layer consists of:
/// 1. **Rotation layer**: Apply parameterized rotation gates to all qubits
/// 2. **Entangling layer**: Create entanglement via two-qubit gates (CNOT chain default)
///
/// **Trade-offs:**
/// - More layers → more expressivity but harder optimization (barren plateaus risk)
/// - More parameters → can represent more states but longer optimization time
/// - Circuit depth → coherence time constraints on real hardware
///
/// **Mathematical Form:**
/// For depth p, numQubits n:
/// - Total parameters: p × n (for single rotation gate per qubit per layer)
/// - Circuit: [Rotation layer] → [Entangling layer] → repeat p times
/// - Ansatz |ψ(θ)⟩ = U_p(θ_{p-1}) ... U_2(θ_1) U_1(θ_0) |0⟩^⊗n
///
/// **Example - 4-qubit, depth=2 ansatz:**
/// ```swift
/// // Creates parameterized circuit with 8 parameters (4 qubits × 2 layers)
/// let ansatz = HardwareEfficientAnsatz.create(numQubits: 4, depth: 2)
///
/// // Structure:
/// // Layer 0: Ry(θ0) on q0, Ry(θ1) on q1, Ry(θ2) on q2, Ry(θ3) on q3
/// //          CNOT(q0→q1), CNOT(q1→q2), CNOT(q2→q3)
/// // Layer 1: Ry(θ4) on q0, Ry(θ5) on q1, Ry(θ6) on q2, Ry(θ7) on q3
/// //          CNOT(q0→q1), CNOT(q1→q2), CNOT(q2→q3)
///
/// print(ansatz.parameterCount())  // 8
/// print(ansatz.gateCount())       // 20 gates (8 Ry + 6 CNOT × 2 layers)
///
/// // Use in VQE
/// let vqe = await VariationalQuantumEigensolver(
///     hamiltonian: hamiltonian,
///     ansatz: ansatz,
///     optimizer: NelderMeadOptimizer()
/// )
/// ```
///
/// **Customization:**
/// ```swift
/// // Use Rx rotations instead of Ry
/// let rxAnsatz = HardwareEfficientAnsatz.create(
///     numQubits: 4,
///     depth: 3,
///     rotationGates: .rx
/// )
///
/// // Full rotations (all three axes)
/// let fullAnsatz = HardwareEfficientAnsatz.create(
///     numQubits: 4,
///     depth: 2,
///     rotationGates: .full  // Rz-Ry-Rz per qubit (3× more parameters)
/// )
///
/// // Custom entangling pattern
/// let customAnsatz = HardwareEfficientAnsatz.create(
///     numQubits: 4,
///     depth: 2,
///     entanglingPattern: .circular  // Includes CNOT(q3→q0) for circular topology
/// )
/// ```
@frozen
public struct HardwareEfficientAnsatz {
    /// Rotation gate choices for single-qubit layers
    @frozen
    public enum RotationGateSet: Sendable {
        /// Ry rotations only (most common, 1 parameter per qubit per layer)
        case ry

        /// Rx rotations only (1 parameter per qubit per layer)
        case rx

        /// Rz rotations only (1 parameter per qubit per layer)
        case rz

        /// Full rotation: Rz-Ry-Rz sequence (3 parameters per qubit per layer)
        /// Most expressive but 3× more parameters
        case full

        @inlinable
        @_effects(readonly)
        func parametersPerQubit() -> Int {
            switch self {
            case .ry, .rx, .rz: 1
            case .full: 3
            }
        }
    }

    /// Entangling gate pattern for two-qubit layers
    @frozen
    public enum EntanglingPattern: Sendable {
        /// Linear chain: CNOT(i → i+1) for i=0..n-2
        case linear

        /// Circular chain: Linear + CNOT(n-1 → 0)
        case circular

        /// All-to-all: CNOT between every qubit pair (expensive, deep circuits)
        case allToAll
    }

    // MARK: - Primary Interface

    /// Create hardware-efficient ansatz
    ///
    /// Builds parameterized circuit with specified depth and rotation gates.
    /// Parameters are auto-named: "theta_{layer}_{qubit}_{axis}" for traceability.
    ///
    /// **Performance:**
    /// - Circuit construction: O(depth × numQubits)
    /// - Gate count: depth × (numQubits × rotations + (numQubits - 1) × CNOTs)
    /// - Parameter count: depth × numQubits × rotationGates.parametersPerQubit()
    ///
    /// - Parameters:
    ///   - numQubits: Number of qubits (1-30)
    ///   - depth: Number of ansatz layers (typically 1-10)
    ///   - rotationGates: Choice of rotation gates (default: Ry)
    ///   - entanglingPattern: Entangling gate topology (default: linear)
    /// - Returns: Parameterized quantum circuit ready for VQE
    ///
    /// Example:
    /// ```swift
    /// // Standard VQE ansatz for 6-qubit system
    /// let ansatz = HardwareEfficientAnsatz.create(numQubits: 6, depth: 3)
    ///
    /// // Use with VQE
    /// let initialParams = Array(repeating: 0.01, count: ansatz.parameterCount())
    /// let vqe = await VariationalQuantumEigensolver(
    ///     hamiltonian: hamiltonian,
    ///     ansatz: ansatz,
    ///     optimizer: NelderMeadOptimizer()
    /// )
    /// let result = try await vqe.run(initialParameters: initialParams)
    /// ```
    @_optimize(speed)
    @_eagerMove
    public static func create(
        numQubits: Int,
        depth: Int,
        rotationGates: RotationGateSet = .ry,
        entanglingPattern: EntanglingPattern = .linear
    ) -> ParameterizedQuantumCircuit {
        ValidationUtilities.validatePositiveQubits(numQubits)
        ValidationUtilities.validateMemoryLimit(numQubits)
        precondition(depth > 0, "depth must be positive")
        precondition(depth <= 100, "depth must be ≤ 100 (excessive circuit depth)")

        var circuit = ParameterizedQuantumCircuit(numQubits: numQubits)

        for layerIndex in 0 ..< depth {
            appendRotationLayer(
                to: &circuit,
                numQubits: numQubits,
                layerIndex: layerIndex,
                rotationGates: rotationGates
            )

            appendEntanglingLayer(
                to: &circuit,
                numQubits: numQubits,
                pattern: entanglingPattern
            )
        }

        return circuit
    }

    // MARK: - Layer Construction

    /// Append parameterized rotation layer to circuit
    ///
    /// Adds rotation gates to all qubits with unique symbolic parameters.
    /// Parameter naming: "theta_{layer}_{qubit}" for single rotations,
    /// "theta_{layer}_{qubit}_{axis}" for full rotations.
    ///
    /// - Parameters:
    ///   - circuit: Circuit to modify (inout for performance)
    ///   - numQubits: Number of qubits
    ///   - layerIndex: Current layer index (for parameter naming)
    ///   - rotationGates: Type of rotation gates to apply
    @_optimize(speed)
    private static func appendRotationLayer(
        to circuit: inout ParameterizedQuantumCircuit,
        numQubits: Int,
        layerIndex: Int,
        rotationGates: RotationGateSet
    ) {
        switch rotationGates {
        case .ry:
            for qubit in 0 ..< numQubits {
                let paramName = "theta_\(layerIndex)_\(qubit)"
                let param = Parameter(name: paramName)
                circuit.append(
                    gate: .rotationY(theta: .parameter(param)),
                    toQubit: qubit
                )
            }

        case .rx:
            for qubit in 0 ..< numQubits {
                let paramName = "theta_\(layerIndex)_\(qubit)"
                let param = Parameter(name: paramName)
                circuit.append(
                    gate: .rotationX(theta: .parameter(param)),
                    toQubit: qubit
                )
            }

        case .rz:
            for qubit in 0 ..< numQubits {
                let paramName = "theta_\(layerIndex)_\(qubit)"
                let param = Parameter(name: paramName)
                circuit.append(
                    gate: .rotationZ(theta: .parameter(param)),
                    toQubit: qubit
                )
            }

        case .full:
            for qubit in 0 ..< numQubits {
                let paramZ1 = Parameter(name: "theta_\(layerIndex)_\(qubit)_z1")
                let paramY = Parameter(name: "theta_\(layerIndex)_\(qubit)_y")
                let paramZ2 = Parameter(name: "theta_\(layerIndex)_\(qubit)_z2")

                circuit.append(gate: .rotationZ(theta: .parameter(paramZ1)), toQubit: qubit)
                circuit.append(gate: .rotationY(theta: .parameter(paramY)), toQubit: qubit)
                circuit.append(gate: .rotationZ(theta: .parameter(paramZ2)), toQubit: qubit)
            }
        }
    }

    /// Append entangling layer to circuit
    ///
    /// Adds two-qubit gates to create entanglement between qubits.
    /// Uses CNOT gates (standard for most hardware platforms).
    ///
    /// **Patterns:**
    /// - Linear: q0→q1, q1→q2, ..., q(n-2)→q(n-1)
    /// - Circular: Linear + q(n-1)→q0
    /// - All-to-all: Every pair (i→j) for i<j
    ///
    /// - Parameters:
    ///   - circuit: Circuit to modify
    ///   - numQubits: Number of qubits
    ///   - pattern: Entangling topology
    @_optimize(speed)
    private static func appendEntanglingLayer(
        to circuit: inout ParameterizedQuantumCircuit,
        numQubits: Int,
        pattern: EntanglingPattern
    ) {
        switch pattern {
        case .linear:
            for i in 0 ..< (numQubits - 1) {
                circuit.append(
                    gate: .concrete(.cnot(control: i, target: i + 1)),
                    qubits: []
                )
            }

        case .circular:
            for i in 0 ..< (numQubits - 1) {
                circuit.append(
                    gate: .concrete(.cnot(control: i, target: i + 1)),
                    qubits: []
                )
            }

            if numQubits >= 2 {
                circuit.append(
                    gate: .concrete(.cnot(control: numQubits - 1, target: 0)),
                    qubits: []
                )
            }

        case .allToAll:
            for i in 0 ..< numQubits {
                for j in (i + 1) ..< numQubits {
                    circuit.append(
                        gate: .concrete(.cnot(control: i, target: j)),
                        qubits: []
                    )
                }
            }
        }
    }
}
