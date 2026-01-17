// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Internal QAOA ansatz circuit builder for combinatorial optimization.
///
/// Constructs parameterized quantum circuits for the Quantum Approximate Optimization
/// Algorithm (QAOA). The ansatz implements alternating cost and mixer layers applied
/// to an initial uniform superposition, producing the state:
///
/// |ψ(γ⃗,β⃗)⟩ = exp(-iβₚ₋₁H_m)exp(-iγₚ₋₁H_c)...exp(-iβ₀H_m)exp(-iγ₀H_c)|+⟩^⊗n
///
/// The circuit begins with Hadamard gates on all qubits to create |+⟩^⊗n, then
/// alternates between cost Hamiltonian evolution exp(-iγₖH_c) encoding the optimization
/// objective and mixer Hamiltonian evolution exp(-iβₖH_m) driving exploration. Each
/// Hamiltonian layer uses first-order Trotterization to decompose exp(-iθH) into
/// products of Pauli string exponentials.
///
/// Pauli string exponentiation exp(-iθP) for P = ⊗ⱼ Pⱼ converts non-Z Paulis to Z
/// basis via Hadamard (X) or Rx(-π/2) (Y), applies CNOT ladder to entangle target
/// qubits, executes Rz(2θ) on the final qubit, then reverses the CNOT chain and
/// basis rotations.
///
/// Direct use discouraged. Prefer ``QuantumCircuit/qaoa(cost:mixer:qubits:depth:)``.
///
/// - Complexity: Circuit depth O(p·|E|) and gate count ~3p|E| where p is depth and
///   |E| is the number of Hamiltonian terms. Parameter count is 2p.
///
/// - SeeAlso: ``QAOA``
/// - SeeAlso: ``QuantumCircuit/qaoa(cost:mixer:qubits:depth:)``
enum QAOAAnsatz {
    private static let maxDepth = 100
    private static let coefficientThreshold = 1e-15
    private static let halfPi = Double.pi / 2
    private static let negHalfPi = -Double.pi / 2

    /// Build QAOA ansatz circuit with alternating cost and mixer layers.
    ///
    /// Creates parameterized circuit implementing depth-p QAOA structure: initial
    /// uniform superposition followed by p alternating cost/mixer Hamiltonian
    /// evolution layers. Parameters named "gamma_0", "beta_0", ..., "gamma_{depth-1}",
    /// "beta_{depth-1}" for use with standard QAOA optimizers.
    ///
    /// **Example:**
    /// ```swift
    /// let cost = MaxCut.hamiltonian(edges: [(0,1), (1,2)])
    /// let mixer = MixerHamiltonian.x(qubits: 3)
    /// let ansatz = QAOAAnsatz.build(cost: cost, mixer: mixer, qubits: 3, depth: 2)
    /// ```
    ///
    /// - Parameters:
    ///   - cost: Cost Hamiltonian H_c encoding optimization objective
    ///   - mixer: Mixer Hamiltonian H_m for solution space exploration
    ///   - qubits: Number of qubits (1-30)
    ///   - depth: Number of alternating layers (1-100)
    /// - Returns: Parameterized quantum circuit ready for QAOA optimization
    ///
    /// - Complexity: O(depth · (|cost.terms| + |mixer.terms|)) construction time
    ///
    /// - Precondition: `qubits` in range 1...30, `depth` in range 1...100, both Hamiltonians must be non-empty
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    static func build(
        cost: Observable,
        mixer: Observable,
        qubits: Int,
        depth: Int,
    ) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateMemoryLimit(qubits)
        ValidationUtilities.validatePositiveInt(depth, name: "depth")
        ValidationUtilities.validateUpperBound(depth, max: maxDepth, name: "depth")
        ValidationUtilities.validateNonEmpty(cost.terms, name: "cost.terms")
        ValidationUtilities.validateNonEmpty(mixer.terms, name: "mixer.terms")

        var circuit = QuantumCircuit(qubits: qubits)

        for qubit in 0 ..< qubits {
            circuit.append(.hadamard, to: qubit)
        }

        for layer in 0 ..< depth {
            let gammaParam = Parameter(name: "gamma_\(layer)")
            applyTrotterLayer(
                to: &circuit,
                hamiltonian: cost,
                parameter: gammaParam,
            )

            let betaParam = Parameter(name: "beta_\(layer)")
            applyTrotterLayer(
                to: &circuit,
                hamiltonian: mixer,
                parameter: betaParam,
            )
        }

        return circuit
    }

    // MARK: - Hamiltonian Layer Construction

    /// Append Hamiltonian evolution layer exp(-iθH) using first-order Trotterization.
    @_optimize(speed)
    private static func applyTrotterLayer(
        to circuit: inout QuantumCircuit,
        hamiltonian: Observable,
        parameter: Parameter,
    ) {
        for (coefficient, pauliString) in hamiltonian.terms {
            guard !pauliString.operators.isEmpty, abs(coefficient) > coefficientThreshold else {
                continue
            }

            applyPauliExponential(
                to: &circuit,
                pauliString: pauliString,
                parameter: parameter,
                coefficient: coefficient,
            )
        }
    }

    /// Append Pauli string exponential exp(-iθcP) using CNOT ladder decomposition.
    @_optimize(speed)
    private static func applyPauliExponential(
        to circuit: inout QuantumCircuit,
        pauliString: PauliString,
        parameter: Parameter,
        coefficient: Double,
    ) {
        let operatorCount = pauliString.operators.count

        var targetQubits = [Int](unsafeUninitializedCapacity: operatorCount) { buffer, count in
            for i in 0 ..< operatorCount {
                let op = pauliString.operators[i]
                buffer[i] = op.qubit
                applyBasisRotation(to: &circuit, qubit: op.qubit, basis: op.basis, forward: true)
            }
            count = operatorCount
        }
        targetQubits.sort()

        let scaledCoeff = coefficient * 2.0
        let scaledParameter = Parameter(name: "\(parameter.name)_c_\(scaledCoeff)")

        if targetQubits.count == 1 {
            circuit.append(.rotationZ(.parameter(scaledParameter)), to: targetQubits[0])
        } else {
            for i in 0 ..< (targetQubits.count - 1) {
                circuit.append(.cnot, to: [targetQubits[i], targetQubits[i + 1]])
            }

            circuit.append(.rotationZ(.parameter(scaledParameter)), to: targetQubits[targetQubits.count - 1])

            for i in stride(from: targetQubits.count - 2, through: 0, by: -1) {
                circuit.append(.cnot, to: [targetQubits[i], targetQubits[i + 1]])
            }
        }

        for op in pauliString.operators {
            applyBasisRotation(to: &circuit, qubit: op.qubit, basis: op.basis, forward: false)
        }
    }

    /// Apply basis rotation for Pauli exponentiation.
    @inline(__always)
    private static func applyBasisRotation(
        to circuit: inout QuantumCircuit,
        qubit: Int,
        basis: PauliBasis,
        forward: Bool,
    ) {
        switch basis {
        case .z: break
        case .x: circuit.append(.hadamard, to: qubit)
        case .y: circuit.append(.rotationX(forward ? negHalfPi : halfPi), to: qubit)
        }
    }
}

/// Pre-computed QAOA parameter binding for O(n) parameter expansion.
///
/// Parses parameter names once at construction, enabling fast binding without repeated
/// string parsing in optimization loops. Expands base parameters [γ₀,β₀,...,γₚ₋₁,βₚ₋₁]
/// to all coefficient-scaled variants (e.g., "gamma_0_c_1.5" = γ₀ * 1.5).
///
/// Internal use by ``QAOA`` actor for efficient parameter binding during optimization.
///
/// - SeeAlso: ``QAOA``
struct QAOAParameterBinder: Sendable {
    private let parameterInfo: [(name: String, baseIndex: Int, coefficient: Double)]
    private let ansatz: QuantumCircuit

    /// Create binder with pre-computed parameter info.
    @_optimize(speed)
    @_effects(readonly)
    init(ansatz: QuantumCircuit) {
        self.ansatz = ansatz

        var info: [(name: String, baseIndex: Int, coefficient: Double)] = []
        info.reserveCapacity(ansatz.parameterCount)

        for param in ansatz.parameters {
            let paramName = param.name

            guard let coeffSeparatorRange = paramName.range(of: "_c_") else { continue }

            let baseName = String(paramName[..<coeffSeparatorRange.lowerBound])
            let coeffStr = String(paramName[coeffSeparatorRange.upperBound...])
            guard let coefficient = Double(coeffStr) else { continue }

            guard let underscoreRange = baseName.range(of: "_", options: .backwards) else { continue }
            let typeStr = String(baseName[..<underscoreRange.lowerBound])
            let layerStr = String(baseName[underscoreRange.upperBound...])
            guard let layer = Int(layerStr) else { continue }

            let isGamma = typeStr == "gamma"
            let baseIndex = isGamma ? (2 * layer) : (2 * layer + 1)
            info.append((name: paramName, baseIndex: baseIndex, coefficient: coefficient))
        }

        parameterInfo = info
    }

    /// Bind base parameters to ansatz with coefficient scaling.
    ///
    /// - Parameter baseParameters: Array of [γ₀, β₀, γ₁, β₁, ..., γₚ₋₁, βₚ₋₁]
    /// - Returns: Concrete quantum circuit with all parameters bound
    ///
    /// - Complexity: O(n) where n is the number of parameters (no string parsing)
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    func bind(baseParameters: [Double]) -> QuantumCircuit {
        var bindings: [String: Double] = Dictionary(minimumCapacity: parameterInfo.count)

        for info in parameterInfo {
            bindings[info.name] = baseParameters[info.baseIndex] * info.coefficient
        }

        return ansatz.binding(bindings)
    }
}

public extension QuantumCircuit {
    /// Create QAOA ansatz circuit for combinatorial optimization.
    ///
    /// Builds parameterized quantum circuit implementing the Quantum Approximate
    /// Optimization Algorithm (QAOA) ansatz for solving combinatorial optimization
    /// problems. The circuit alternates between cost Hamiltonian evolution encoding
    /// the objective and mixer Hamiltonian evolution driving exploration.
    ///
    /// **Example:**
    /// ```swift
    /// let cost = MaxCut.hamiltonian(edges: [(0,1), (1,2), (2,0)])
    /// let mixer = MixerHamiltonian.x(qubits: 3)
    /// let ansatz = QuantumCircuit.qaoa(cost: cost, mixer: mixer, qubits: 3, depth: 2)
    /// ```
    ///
    /// - Parameters:
    ///   - cost: Cost Hamiltonian H_c (e.g., ``MaxCut/hamiltonian(edges:)``)
    ///   - mixer: Mixer Hamiltonian H_m (e.g., ``MixerHamiltonian/x(qubits:)``)
    ///   - qubits: Number of qubits (1-30)
    ///   - depth: Number of alternating layers (1-10 typical)
    /// - Returns: Parameterized quantum circuit ready for QAOA optimization
    ///
    /// - Complexity: O(depth · (|cost.terms| + |mixer.terms|)) construction time
    ///
    /// - Precondition: `qubits` in range 1...30, `depth` in range 1...100, both Hamiltonians must be non-empty
    ///
    /// - SeeAlso: ``QAOA``
    /// - SeeAlso: ``MaxCut``
    /// - SeeAlso: ``MixerHamiltonian``
    @_eagerMove
    @_effects(readonly)
    static func qaoa(
        cost: Observable,
        mixer: Observable,
        qubits: Int,
        depth: Int,
    ) -> QuantumCircuit {
        QAOAAnsatz.build(cost: cost, mixer: mixer, qubits: qubits, depth: depth)
    }
}
