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
/// **Performance:**
/// - Construction: O(depth x numQubits) gates
/// - Parameter count: depth x numQubits x rotations.parametersPerQubit
/// - Circuit depth: O(depth x numQubits) for linear/circular, O(depth x numQubits²) for all-to-all
///
/// **Example:**
/// ```swift
/// let ansatz = HardwareEfficientAnsatz(qubits: 4, depth: 2)
/// print(ansatz.parameterCount)  // 8 (4 qubits x 2 layers x 1 Ry per qubit)
///
/// let vqe = await VQE(
///     hamiltonian: hamiltonian,
///     ansatz: ansatz,
///     optimizer: COBYLAOptimizer()
/// )
/// let result = await vqe.run(initialParameters: Array(repeating: 0.01, count: ansatz.parameterCount))
/// ```
///
/// - SeeAlso: ``VQE``, ``QuantumCircuit``, ``Parameter``
public struct HardwareEfficientAnsatz: Sendable {
    /// Single-qubit rotation gate choices affecting circuit expressivity and parameter count
    ///
    /// Rotation gates control how many parameters are needed per qubit per layer. Single-axis
    /// rotations (Rx, Ry, Rz) use one parameter each. Full rotations (Rz-Ry-Rz sequence) provide
    /// maximum single-qubit expressivity at the cost of 3x parameters.
    ///
    /// **Typical usage:** Ry rotations (default) balance expressivity and parameter efficiency.
    /// Use full rotations when additional expressivity is needed and parameter count is acceptable.
    ///
    /// **Example:**
    /// ```swift
    /// let standard = HardwareEfficientAnsatz(qubits: 4, depth: 2, rotations: .ry)
    /// print(standard.parameterCount)  // 8
    ///
    /// let expressive = HardwareEfficientAnsatz(qubits: 4, depth: 2, rotations: .full)
    /// print(expressive.parameterCount)  // 24 (3x parameters)
    /// ```
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
    /// creates full connectivity but produces deep circuits unsuitable for NISQ devices.
    ///
    /// **Typical usage:** Linear (default) for hardware compatibility. Circular adds one CNOT
    /// connecting first and last qubits. Avoid all-to-all except for small qubit counts.
    ///
    /// **Example:**
    /// ```swift
    /// let linear = HardwareEfficientAnsatz(qubits: 4, depth: 1, entanglement: .linear)
    /// print(linear.circuit.count)  // 7 gates (4 Ry + 3 CNOT)
    ///
    /// let circular = HardwareEfficientAnsatz(qubits: 4, depth: 1, entanglement: .circular)
    /// print(circular.circuit.count)  // 8 gates (4 Ry + 4 CNOT)
    /// ```
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
    /// Computed as depth x numQubits x parametersPerQubit. Use this count when
    /// initializing parameter vectors for optimization.
    ///
    /// - Complexity: O(1)
    /// - SeeAlso: ``Rotations/parametersPerQubit``
    public var parameterCount: Int {
        depth * circuit.numQubits * rotations.parametersPerQubit
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
    ///   - numQubits: System size
    ///   - depth: Number of rotation-entanglement layer pairs
    ///   - rotations: Single-qubit gate choice (default: Ry)
    ///   - entanglement: Two-qubit connectivity (default: linear)
    ///
    /// - Complexity: O(depth x numQubits) gates constructed
    /// - Precondition: `numQubits` must be positive and ≤30, `depth` must be positive and ≤100
    @_optimize(speed)
    public init(
        qubits numQubits: Int,
        depth: Int,
        rotations: Rotations = .ry,
        entanglement: Entanglement = .linear
    ) {
        ValidationUtilities.validatePositiveQubits(numQubits)
        ValidationUtilities.validateMemoryLimit(numQubits)
        ValidationUtilities.validatePositiveInt(depth, name: "depth")
        ValidationUtilities.validateUpperBound(depth, max: 100, name: "depth")

        var circuit = QuantumCircuit(numQubits: numQubits)

        for layerIndex in 0 ..< depth {
            Self.addRotations(
                to: &circuit,
                qubits: numQubits,
                layer: layerIndex,
                type: rotations
            )

            Self.addEntanglement(
                to: &circuit,
                qubits: numQubits,
                pattern: entanglement
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
        type: Rotations
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
        pattern: Entanglement
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
