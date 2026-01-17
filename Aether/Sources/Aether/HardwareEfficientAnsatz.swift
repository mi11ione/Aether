// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Parameterized quantum circuit optimized for variational algorithms on near-term hardware
///
/// Hardware-efficient ansatz minimizes circuit depth while maintaining expressivity through
/// layered rotations and entangling gates. Primary choice for VQE applications without
/// domain-specific structure (chemistry, optimization, condensed matter).
///
/// Each layer applies single-qubit rotations to all qubits followed by two-qubit entangling
/// gates. Depth controls expressivity: more layers represent more quantum states but increase
/// optimization difficulty and susceptibility to barren plateaus.
///
/// Construction requires O(depth * qubits) gates with parameter count equal to
/// depth * qubits * rotations.parametersPerQubit. Circuit depth scales as O(depth * qubits)
/// for linear/circular entanglement, or O(depth * qubits²) for all-to-all connectivity.
///
/// **Example:**
/// ```swift
/// let ansatz = HardwareEfficientAnsatz(qubits: 4, depth: 2)
/// let params = Array(repeating: 0.01, count: ansatz.parameterCount)
/// let state = ansatz.circuit.bound(with: params).execute()
/// ```
///
/// - SeeAlso: ``VQE``
/// - SeeAlso: ``QuantumCircuit``
/// - SeeAlso: ``Parameter``
@frozen
public struct HardwareEfficientAnsatz: Sendable {
    /// Single-qubit rotation gate choices affecting circuit expressivity and parameter count
    ///
    /// Rotation gates control how many parameters are needed per qubit per layer. Single-axis
    /// rotations (Rx, Ry, Rz) use one parameter each. Full rotations (Rz-Ry-Rz sequence) provide
    /// maximum single-qubit expressivity at the cost of 3x parameters. Ry rotations (default)
    /// balance expressivity and parameter efficiency; use full rotations when additional
    /// expressivity is needed and parameter count is acceptable.
    ///
    /// **Example:**
    /// ```swift
    /// let ry = HardwareEfficientAnsatz(qubits: 4, depth: 2, rotations: .ry)
    /// let full = HardwareEfficientAnsatz(qubits: 4, depth: 2, rotations: .full)
    /// print(ry.parameterCount, full.parameterCount)  // 8, 24
    /// ```
    @frozen
    public enum Rotations: Sendable {
        /// Ry rotations (1 parameter per qubit per layer)
        case ry

        /// Rx rotations (1 parameter per qubit per layer)
        case rx

        /// Rz rotations (1 parameter per qubit per layer)
        case rz

        /// Rz-Ry-Rz sequence (3 parameters per qubit per layer)
        case full

        /// Number of parameters required per qubit per layer
        @inlinable
        public var parametersPerQubit: Int {
            switch self {
            case .ry, .rx, .rz: 1
            case .full: 3
            }
        }
    }

    /// Two-qubit entangling gate patterns controlling connectivity and circuit depth
    ///
    /// Entangling patterns determine which qubits interact via CNOT gates. Linear and circular
    /// patterns create nearest-neighbor connectivity suitable for most quantum hardware. All-to-all
    /// creates full connectivity but produces deep circuits unsuitable for NISQ devices. Linear
    /// (default) provides hardware compatibility, circular adds one CNOT connecting first and
    /// last qubits, and all-to-all should be avoided except for small qubit counts.
    ///
    /// **Example:**
    /// ```swift
    /// let linear = HardwareEfficientAnsatz(qubits: 4, depth: 1, entanglement: .linear)
    /// let circular = HardwareEfficientAnsatz(qubits: 4, depth: 1, entanglement: .circular)
    /// print(linear.circuit.count, circular.circuit.count)  // 7, 8
    /// ```
    @frozen
    public enum Entanglement: Sendable {
        /// CNOT chain: i->(i+1) for i=0..n-2
        case linear

        /// Linear chain plus CNOT from last to first qubit
        case circular

        /// CNOT between every qubit pair (deep circuits, avoid for n>5)
        case allToAll
    }

    // MARK: - Properties

    /// Parameterized quantum circuit implementing this ansatz
    public let circuit: QuantumCircuit

    /// Number of rotation-entanglement layers
    public let depth: Int

    /// Rotation gate type used in each layer
    public let rotations: Rotations

    /// Two-qubit gate connectivity pattern
    public let entanglement: Entanglement

    /// Total parameters in circuit
    ///
    /// Computed as depth x qubits x parametersPerQubit. Use this count when
    /// initializing parameter vectors for optimization.
    ///
    /// - Complexity: O(1)
    /// - SeeAlso: ``Rotations/parametersPerQubit``
    public var parameterCount: Int {
        depth * circuit.qubits * rotations.parametersPerQubit
    }

    // MARK: - Initialization

    /// Creates hardware-efficient ansatz with specified configuration
    ///
    /// Constructs parameterized circuit by alternating rotation and entangling layers.
    /// Parameters are automatically named "theta_{layer}_{qubit}" for single rotations or
    /// "theta_{layer}_{qubit}_{axis}" for full rotations to enable gradient tracing.
    ///
    /// **Example:**
    /// ```swift
    /// let ansatz = HardwareEfficientAnsatz(qubits: 6, depth: 3)
    /// let params = Array(repeating: 0.01, count: ansatz.parameterCount)
    /// let circuit = ansatz.circuit.bound(with: params)
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: System size
    ///   - depth: Number of rotation-entanglement layer pairs
    ///   - rotations: Single-qubit gate choice (default: Ry)
    ///   - entanglement: Two-qubit connectivity (default: linear)
    ///
    /// - Complexity: O(depth x qubits) gates constructed
    /// - Precondition: `qubits` must be positive and ≤30, `depth` must be positive and ≤100
    @_optimize(speed)
    public init(
        qubits: Int,
        depth: Int,
        rotations: Rotations = .ry,
        entanglement: Entanglement = .linear,
    ) {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateMemoryLimit(qubits)
        ValidationUtilities.validatePositiveInt(depth, name: "depth")
        ValidationUtilities.validateUpperBound(depth, max: 100, name: "depth")

        var circuit = QuantumCircuit(qubits: qubits)

        for layerIndex in 0 ..< depth {
            Self.addRotations(
                to: &circuit,
                qubits: qubits,
                layer: layerIndex,
                type: rotations,
            )

            Self.addEntanglement(
                to: &circuit,
                qubits: qubits,
                pattern: entanglement,
            )
        }

        self.circuit = circuit
        self.depth = depth
        self.rotations = rotations
        self.entanglement = entanglement
    }

    // MARK: - Private Helpers

    @_optimize(speed)
    private static func addRotations(
        to circuit: inout QuantumCircuit,
        qubits: Int,
        layer: Int,
        type: Rotations,
    ) {
        let layerPrefix = "theta_\(layer)_"

        switch type {
        case .ry:
            for qubit in 0 ..< qubits {
                let param = Parameter(name: layerPrefix + String(qubit))
                circuit.append(.rotationY(.parameter(param)), to: qubit)
            }

        case .rx:
            for qubit in 0 ..< qubits {
                let param = Parameter(name: layerPrefix + String(qubit))
                circuit.append(.rotationX(.parameter(param)), to: qubit)
            }

        case .rz:
            for qubit in 0 ..< qubits {
                let param = Parameter(name: layerPrefix + String(qubit))
                circuit.append(.rotationZ(.parameter(param)), to: qubit)
            }

        case .full:
            for qubit in 0 ..< qubits {
                let qubitStr = String(qubit)
                let paramZ1 = Parameter(name: layerPrefix + qubitStr + "_z1")
                let paramY = Parameter(name: layerPrefix + qubitStr + "_y")
                let paramZ2 = Parameter(name: layerPrefix + qubitStr + "_z2")

                circuit.append(.rotationZ(.parameter(paramZ1)), to: qubit)
                circuit.append(.rotationY(.parameter(paramY)), to: qubit)
                circuit.append(.rotationZ(.parameter(paramZ2)), to: qubit)
            }
        }
    }

    @_optimize(speed)
    private static func addEntanglement(
        to circuit: inout QuantumCircuit,
        qubits: Int,
        pattern: Entanglement,
    ) {
        switch pattern {
        case .linear, .circular:
            for i in 0 ..< (qubits - 1) {
                circuit.append(.cnot, to: [i, i + 1])
            }
            if pattern == .circular, qubits >= 2 {
                circuit.append(.cnot, to: [qubits - 1, 0])
            }

        case .allToAll:
            for i in 0 ..< qubits {
                for j in (i + 1) ..< qubits {
                    circuit.append(.cnot, to: [i, j])
                }
            }
        }
    }
}
