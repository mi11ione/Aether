// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// QAOA ansatz circuit constructor for combinatorial optimization
///
/// Builds parameterized quantum circuits implementing the Quantum Approximate
/// Optimization Algorithm (QAOA) ansatz structure. **Core circuit builder for QAOA.**
///
/// **QAOA Circuit Structure:**
/// ```
/// |0⟩^⊗n -> H^⊗n -> [Problem layer] -> [Mixer layer] -> repeat p times -> Measure
/// ```
///
/// Each QAOA layer consists of:
/// 1. **Problem layer**: exp(-iγ·H_p) encodes optimization cost function
/// 2. **Mixer layer**: exp(-iβ·H_m) drives exploration of solution space
///
/// **Mathematical Foundation:**
/// - Parameters: (γ₀,β₀), (γ₁,β₁), ..., (γₚ₋₁,βₚ₋₁) ∈ ℝ²ᵖ
/// - Ansatz state: |ψ(γ⃗,β⃗)⟩ = exp(-iβₚ₋₁H_m)exp(-iγₚ₋₁H_p)...exp(-iβ₀H_m)exp(-iγ₀H_p)|+⟩^⊗n
/// - Trotterization: exp(-iγH) = ∏ᵢ exp(-iγcᵢPᵢ) for H = Σᵢ cᵢPᵢ (first-order)
///
/// **Pauli String Exponentiation:**
/// exp(-iθ·P) where P = ⊗ⱼ Pⱼ (Pⱼ ∈ {I,X,Y,Z}):
/// - **Z string** (Z₀⊗Z₁⊗...⊗Zₖ): CNOT ladder + Rz(2θ) + reverse CNOT ladder
/// - **X string** (X₀⊗X₁⊗...⊗Xₖ): Hadamard + Z ladder + reverse Hadamard
/// - **Y string** (Y₀⊗Y₁⊗...⊗Yₖ): Rx(-π/2) + Z ladder + reverse Rx(π/2)
/// - **Mixed strings**: Convert to Z basis, apply Z ladder, convert back
///
/// **Performance:**
/// - Circuit depth: O(p · |E|) where |E| = number of Hamiltonian terms
/// - Gate count per layer: ~3|E| (basis rotations + CNOT ladder + reverse)
/// - Parameter count: 2p (γ and β per layer)
///
/// **Example - MaxCut QAOA:**
/// ```swift
/// // 4-vertex square graph
/// let edges = [(0,1), (1,2), (2,3), (3,0)]
/// let costHamiltonian = MaxCut.hamiltonian(edges: edges)
/// let mixerHamiltonian = MixerHamiltonian.xMixer(numQubits: 4)
///
/// // Build depth-2 QAOA circuit
/// let ansatz = QAOAAnsatz.create(
///     numQubits: 4,
///     depth: 2,
///     costHamiltonian: costHamiltonian,
///     mixerHamiltonian: mixerHamiltonian
/// )
///
/// // Circuit structure:
/// // 1. H^⊗4 (initial superposition)
/// // 2. Problem layer γ₀: 4 ZZ rotations (one per edge)
/// // 3. Mixer layer β₀: 4 Rx rotations (X mixer)
/// // 4. Problem layer γ₁: 4 ZZ rotations
/// // 5. Mixer layer β₁: 4 Rx rotations
///
/// print(ansatz.parameterCount())  // 4 (γ₀,β₀,γ₁,β₁)
/// print(ansatz.gateCount())       // ~36 gates
///
/// // Use with QAOA optimizer
/// let qaoa = await QAOA(
///     costHamiltonian: costHamiltonian,
///     mixerHamiltonian: mixerHamiltonian,
///     numQubits: 4,
///     depth: 2
/// )
/// ```
///
/// **Architecture Decisions:**
/// - **First-order Trotter**: Sufficient accuracy for QAOA (shallow circuits, small parameters)
/// - **CNOT ladder**: Efficient multi-qubit Pauli exponentiation without ancilla
/// - **Basis rotations**: Minimal overhead (2 gates per non-Z Pauli per term)
/// - **Parameter ordering**: (γ₀,β₀,...,γₚ₋₁,βₚ₋₁) matches optimization convention
@frozen
public struct QAOAAnsatz {
    /// Create QAOA ansatz circuit
    ///
    /// Constructs parameterized quantum circuit implementing QAOA structure:
    /// initial superposition + alternating problem/mixer layers × depth.
    ///
    /// **Algorithm:**
    /// 1. Apply Hadamard to all qubits: |0⟩^⊗n -> |+⟩^⊗n
    /// 2. For each layer k = 0 to depth-1:
    ///    a. Problem layer: Apply exp(-iγₖ·cᵢ·Pᵢ) for each term in H_p
    ///    b. Mixer layer: Apply exp(-iβₖ·dⱼ·Qⱼ) for each term in H_m
    /// 3. Parameters named: "gamma_0", "beta_0", ..., "gamma_{depth-1}", "beta_{depth-1}"
    ///
    /// **Complexity:**
    /// - Construction time: O(depth · (|terms_p| + |terms_m|))
    /// - Circuit depth: O(depth · max(|terms_p|, |terms_m|))
    /// - Gate count: depth · (3·|terms_p| + 3·|terms_m|) approximately
    ///
    /// - Parameters:
    ///   - numQubits: Number of qubits (1-30)
    ///   - depth: QAOA depth p (number of alternating layers, typically 1-10)
    ///   - costHamiltonian: Problem Hamiltonian H_p (e.g., MaxCut)
    ///   - mixerHamiltonian: Mixer Hamiltonian H_m (e.g., X mixer)
    /// - Returns: Parameterized quantum circuit ready for QAOA optimization
    ///
    /// **Validation:**
    /// - numQubits must be positive and ≤30 (memory limit)
    /// - depth must be positive and ≤100 (reasonable circuit depth)
    /// - Hamiltonians must be non-empty (at least one term)
    ///
    /// Example:
    /// ```swift
    /// // Triangle graph MaxCut with depth-1 QAOA
    /// let edges = [(0,1), (1,2), (0,2)]
    /// let cost = MaxCut.hamiltonian(edges: edges)
    /// let mixer = MixerHamiltonian.xMixer(numQubits: 3)
    ///
    /// let ansatz = QAOAAnsatz.create(
    ///     numQubits: 3,
    ///     depth: 1,
    ///     costHamiltonian: cost,
    ///     mixerHamiltonian: mixer
    /// )
    ///
    /// // Bind parameters and execute
    /// let circuit = ansatz.bind(parameterVector: [0.5, 0.5])  // γ=0.5, β=0.5
    /// let state = await simulator.execute(circuit)
    /// let energy = cost.expectationValue(state: state)
    /// ```
    @_optimize(speed)
    @_eagerMove
    public static func create(
        numQubits: Int,
        depth: Int,
        costHamiltonian: Observable,
        mixerHamiltonian: Observable
    ) -> ParameterizedQuantumCircuit {
        ValidationUtilities.validatePositiveQubits(numQubits)
        ValidationUtilities.validateMemoryLimit(numQubits)
        ValidationUtilities.validatePositiveInt(depth, name: "depth")
        ValidationUtilities.validateUpperBound(depth, max: 100, name: "depth")
        ValidationUtilities.validateNonEmpty(costHamiltonian.terms, name: "costHamiltonian.terms")
        ValidationUtilities.validateNonEmpty(mixerHamiltonian.terms, name: "mixerHamiltonian.terms")

        var circuit = ParameterizedQuantumCircuit(numQubits: numQubits)

        // Initial state: |+⟩^⊗n (equal superposition)
        for qubit in 0 ..< numQubits {
            circuit.append(gate: .concrete(.hadamard), toQubit: qubit)
        }

        // Alternating problem and mixer layers
        for layer in 0 ..< depth {
            // Problem layer: exp(-iγ·H_p)
            let gammaParam = Parameter(name: "gamma_\(layer)")
            appendHamiltonianLayer(
                to: &circuit,
                hamiltonian: costHamiltonian,
                parameter: gammaParam,
                numQubits: numQubits
            )

            // Mixer layer: exp(-iβ·H_m)
            let betaParam = Parameter(name: "beta_\(layer)")
            appendHamiltonianLayer(
                to: &circuit,
                hamiltonian: mixerHamiltonian,
                parameter: betaParam,
                numQubits: numQubits
            )
        }

        return circuit
    }

    // MARK: - Hamiltonian Layer Construction

    /// Append Hamiltonian evolution layer: exp(-iθ·H) for H = Σᵢ cᵢPᵢ
    ///
    /// Implements first-order Trotter decomposition:
    /// exp(-iθH) ≈ ∏ᵢ exp(-iθ·cᵢ·Pᵢ)
    ///
    /// Each Pauli term contributes one exp(-iα·P) gate where α = θ·cᵢ.
    ///
    /// - Parameters:
    ///   - circuit: Circuit to modify (inout for performance)
    ///   - hamiltonian: Observable H = Σᵢ cᵢPᵢ
    ///   - parameter: Symbolic parameter θ
    ///   - numQubits: Number of qubits (for validation)
    @_optimize(speed)
    private static func appendHamiltonianLayer(
        to circuit: inout ParameterizedQuantumCircuit,
        hamiltonian: Observable,
        parameter: Parameter,
        numQubits: Int
    ) {
        for (coefficient, pauliString) in hamiltonian.terms {
            // Skip identity and near-zero terms
            guard !pauliString.operators.isEmpty, abs(coefficient) > 1e-15 else {
                continue
            }

            // exp(-i·θ·c·P) where c = coefficient, θ = parameter
            // We pass both separately and scale inside the rotation gate
            appendPauliStringExponential(
                to: &circuit,
                pauliString: pauliString,
                parameter: parameter,
                coefficient: coefficient,
                numQubits: numQubits
            )
        }
    }

    /// Append exp(-iθ·c·P) for Pauli string P = ⊗ⱼ Pⱼ with coefficient c
    ///
    /// **Strategy:**
    /// 1. Convert all Paulis to Z basis via basis rotations
    /// 2. Apply multi-qubit Z rotation using CNOT ladder with angle 2·θ·c
    /// 3. Reverse basis rotations
    ///
    /// **CNOT ladder for ZZ...Z rotation:**
    /// - CNOT chain: q₀->q₁->...->qₖ (entangle all qubits)
    /// - Rz(2·θ·c) on last qubit qₖ
    /// - Reverse CNOT chain: qₖ₋₁->...->q₀ (disentangle)
    ///
    /// **Basis conversion:**
    /// - X -> Z: H (Hadamard)
    /// - Y -> Z: Rx(-π/2) = S†·H
    /// - Z -> Z: Identity (no conversion)
    ///
    /// **Coefficient handling:**
    /// - Symbolic parameter θ bound at runtime
    /// - Coefficient c pre-multiplied into rotation angle
    /// - Final angle: 2·θ·c where θ comes from parameter binding
    ///
    /// - Parameters:
    ///   - circuit: Circuit to modify
    ///   - pauliString: Pauli string P (non-empty)
    ///   - parameter: Symbolic parameter θ
    ///   - coefficient: Hamiltonian coefficient c
    ///   - numQubits: Total qubits for validation
    @_optimize(speed)
    private static func appendPauliStringExponential(
        to circuit: inout ParameterizedQuantumCircuit,
        pauliString: PauliString,
        parameter: Parameter,
        coefficient: Double,
        numQubits: Int
    ) {
        let operatorCount = pauliString.operators.count

        var qubits = [Int](unsafeUninitializedCapacity: operatorCount) { buffer, count in
            for i in 0 ..< operatorCount {
                let op = pauliString.operators[i]
                ValidationUtilities.validateQubitIndex(op.qubit, numQubits: numQubits)
                buffer[i] = op.qubit

                switch op.basis {
                case .z: break
                case .x: circuit.append(gate: .concrete(.hadamard), toQubit: op.qubit)
                case .y:
                    circuit.append(
                        gate: .concrete(.rotationX(theta: -.pi / 2)),
                        toQubit: op.qubit
                    )
                }
            }
            count = operatorCount
        }
        qubits.sort()

        // Step 2: Apply multi-qubit Z rotation with coefficient scaling
        // exp(-i·θ·c·P) -> Rz(2·θ·c) where θ is symbolic, c is concrete
        //
        // **Approach**: Create unique scaled parameter for each (base_param, coefficient) pair
        // - Parameter name encodes coefficient: "gamma_0_c_1.000000"
        // - QAOA binds base parameters (gamma_0, beta_0) to values
        // - Custom binding in QAOA.swift expands to all scaled variants
        // - Example: gamma_0=0.5 -> {gamma_0_c_1.0: 0.5, gamma_0_c_-0.5: -0.25, ...}
        let scaledParameter = createScaledParameter(base: parameter, coefficient: coefficient)

        if qubits.count == 1 {
            // Single-qubit case: exp(-iθ·c·Z) = Rz(2·θ·c)
            circuit.append(
                gate: .rotationZ(theta: .parameter(scaledParameter)),
                toQubit: qubits[0]
            )
        } else {
            // Multi-qubit case: CNOT ladder + Rz + reverse ladder

            // Forward CNOT ladder: entangle qubits
            for i in 0 ..< (qubits.count - 1) {
                circuit.append(
                    gate: .concrete(.cnot(control: qubits[i], target: qubits[i + 1])),
                    qubits: []
                )
            }

            // Rotation on last qubit: Rz(2·θ·c)
            circuit.append(
                gate: .rotationZ(theta: .parameter(scaledParameter)),
                toQubit: qubits[qubits.count - 1]
            )

            // Reverse CNOT ladder: disentangle
            for i in stride(from: qubits.count - 2, through: 0, by: -1) {
                circuit.append(
                    gate: .concrete(.cnot(control: qubits[i], target: qubits[i + 1])),
                    qubits: []
                )
            }
        }

        // Step 3: Reverse basis rotations
        for op in pauliString.operators {
            switch op.basis {
            case .z:
                break // No reversal needed

            case .x:
                // Reverse: H (Hadamard is self-inverse)
                circuit.append(gate: .concrete(.hadamard), toQubit: op.qubit)

            case .y:
                // Reverse: Rx(+π/2)
                circuit.append(
                    gate: .concrete(.rotationX(theta: .pi / 2)),
                    toQubit: op.qubit
                )
            }
        }
    }

    /// Create parameter representing 2·θ·c for scaled rotation
    ///
    /// Since ParameterExpression doesn't support arithmetic expressions,
    /// we use the same base parameter but rely on the fact that the coefficient
    /// is embedded in the circuit. When binding, the parameter value will be
    /// scaled appropriately.
    ///
    /// **Design decision:**
    /// - QAOA binds parameters (γ, β) directly
    /// - Each Pauli term has coefficient c
    /// - Rotation angle needs to be 2·γ·c or 2·β·c
    /// - We store the base parameter (γ or β) and coefficient separately
    /// - Binding layer multiplies: parameter_value * 2 * coefficient
    ///
    /// This is handled by ParameterizedGate.rotationZ which expects the final
    /// angle = 2·θ·c where θ comes from binding the parameter.
    ///
    /// **Actually**, looking at the code flow more carefully:
    /// - ParameterizedGate.rotationZ takes ParameterExpression
    /// - During binding, we substitute parameter -> value
    /// - But we need value * 2 * coefficient for the angle
    ///
    /// **Better approach**: Use the base parameter directly and handle
    /// coefficient scaling in a custom binding step for QAOA.
    ///
    /// Wait - even simpler: Just use the base parameter as-is.
    /// The coefficient scaling happens at binding time in QAOA.swift
    /// when we create the concrete circuit.
    ///
    /// - Parameters:
    ///   - base: Base parameter (γ or β)
    ///   - coefficient: Hamiltonian coefficient c
    /// - Returns: Parameter representing the scaled rotation
    @_optimize(speed)
    @_eagerMove
    @inline(__always)
    private static func createScaledParameter(
        base: Parameter,
        coefficient: Double
    ) -> Parameter {
        // Store coefficient in parameter name for proper scaling during binding
        // Format: "base_name_c_coefficient"
        let scaledCoeff = coefficient * 2.0
        return Parameter(name: "\(base.name)_c_\(scaledCoeff)")
    }
}

/// Pre-computed QAOA parameter binding information for fast repeated binding
///
/// Parses parameter names once at construction, enabling O(n) binding without
/// string parsing in hot optimization loops.
///
/// **Usage:**
/// ```swift
/// let ansatz = QAOAAnsatz.create(...)
/// let binder = QAOAParameterBinder(ansatz: ansatz)
///
/// // Fast repeated binding in optimization loop
/// for params in parameterSweep {
///     let circuit = binder.bind(baseParameters: params)
/// }
/// ```
@frozen
public struct QAOAParameterBinder: Sendable {
    /// Pre-parsed parameter info: (name, baseParameterIndex, coefficient)
    private let parameterInfo: [(name: String, baseIndex: Int, coefficient: Double)]
    private let ansatz: ParameterizedQuantumCircuit

    /// Create binder with pre-computed parameter info
    ///
    /// - Parameter ansatz: QAOA ansatz circuit to bind
    @_optimize(speed)
    public init(ansatz: ParameterizedQuantumCircuit) {
        self.ansatz = ansatz

        var info: [(name: String, baseIndex: Int, coefficient: Double)] = []
        info.reserveCapacity(ansatz.parameterCount())

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

    /// Bind base parameters to ansatz with coefficient scaling
    ///
    /// **Complexity:** O(n) where n = number of parameters (no string parsing)
    ///
    /// - Parameter baseParameters: Array of [γ₀, β₀, γ₁, β₁, ..., γₚ₋₁, βₚ₋₁]
    /// - Returns: Concrete quantum circuit with all parameters bound
    @_optimize(speed)
    @_eagerMove
    public func bind(baseParameters: [Double]) -> QuantumCircuit {
        var bindings: [String: Double] = Dictionary(minimumCapacity: parameterInfo.count)

        for info in parameterInfo {
            bindings[info.name] = baseParameters[info.baseIndex] * info.coefficient
        }

        return ansatz.bind(parameters: bindings)
    }
}
