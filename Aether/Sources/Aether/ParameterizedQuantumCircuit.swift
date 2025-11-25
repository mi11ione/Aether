// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Single parameterized gate operation in a circuit
@frozen
public struct ParameterizedGateOperation: Equatable, Sendable, CustomStringConvertible {
    public let gate: ParameterizedGate
    public let qubits: [Int]
    public let timestamp: Double?

    /// Create parameterized gate operation
    /// - Parameters:
    ///   - gate: Parameterized quantum gate
    ///   - qubits: Target qubit indices
    ///   - timestamp: Optional timestamp for animation/visualization
    public init(gate: ParameterizedGate, qubits: [Int], timestamp: Double? = nil) {
        self.gate = gate
        self.qubits = qubits
        self.timestamp = timestamp
    }

    /// String representation of operation
    @inlinable
    public var description: String {
        let qubitStr = qubits.isEmpty ? "" : " on qubits \(qubits)"
        if let ts = timestamp {
            return "\(gate)\(qubitStr) @ \(String(format: "%.2f", ts))s"
        }
        return "\(gate)\(qubitStr)"
    }
}

/// Parameterized quantum circuit for variational algorithms
///
/// Quantum circuit with symbolic parameters that can be bound at execution time.
/// Essential infrastructure for variational quantum algorithms (VQE, QAOA) where
/// circuit parameters are optimized by classical optimizers to minimize objective
/// functions like molecular ground state energies or combinatorial optimization costs.
///
/// **Architecture**:
/// - Ordered sequence of parameterized gate operations
/// - Auto-registration of parameters during circuit construction
/// - Maintains both ordered list (for vector interface) and set (for deduplication)
/// - Immutable parameter binding produces concrete `QuantumCircuit`
/// - Thread-safe via value semantics (struct with no shared mutable state)
///
/// **Parameter binding interfaces**:
/// 1. **Dictionary interface**: `bind(parameters: ["θ": 0.5, "φ": 1.2])`
///    - Named parameters, optimizer-friendly
///    - Clear parameter identification
///    - Type-safe with descriptive errors
///
/// 2. **Vector interface**: `bind(parameterVector: [0.5, 1.2])`
///    - Compatible with NumPy, SciPy, classical ML libraries
///    - Standard interface for gradient-based optimizers
///    - Parameters bound in registration order
///
/// **Gradient computation**:
/// - Parameter shift rule: ∂⟨H⟩/∂θᵢ = [⟨H⟩(θᵢ+π/2) - ⟨H⟩(θᵢ-π/2)]/2
/// - `generateShiftedCircuits()` creates θᵢ±shift circuits
/// - More accurate than finite differences for quantum circuits
/// - Enables gradient-based optimization (L-BFGS-B, Adam, natural gradient)
///
/// **VQE workflow example**:
/// ```swift
/// // 1. Build parameterized ansatz (hardware-efficient)
/// var circuit = ParameterizedQuantumCircuit(numQubits: 4)
/// let theta = Parameter(name: "theta")
/// let phi = Parameter(name: "phi")
///
/// // Layer 1: Ry rotations
/// for i in 0..<4 {
///     let param = Parameter(name: "theta_\(i)")
///     circuit.append(gate: .rotationY(theta: .parameter(param)), toQubit: i)
/// }
///
/// // Entangling layer: CNOT chain
/// for i in 0..<3 {
///     circuit.append(gate: .concrete(.cnot(control: i, target: i+1)), qubits: [])
/// }
///
/// // 2. Classical optimizer proposes parameters
/// var params = [0.1, 0.2, 0.3, 0.4]  // Initial guess
///
/// // 3. VQE optimization loop
/// for iteration in 0..<100 {
///     // Bind parameters and execute
///     let concreteCircuit = try circuit.bind(parameterVector: params)
///     let state = concreteCircuit.execute()
///
///     // Measure energy expectation ⟨ψ|H|ψ⟩
///     let energy = hamiltonian.expectation(in: state)
///
///     // Compute gradients via parameter shift
///     var gradients = [Double](repeating: 0, count: params.count)
///     for i in 0..<params.count {
///         let (plus, minus) = try circuit.generateShiftedCircuits(parameterIndex: i)
///         let statePlus = plus.execute()
///         let stateMinus = minus.execute()
///         let energyPlus = hamiltonian.expectation(in: statePlus)
///         let energyMinus = hamiltonian.expectation(in: stateMinus)
///         gradients[i] = (energyPlus - energyMinus) / 2.0
///     }
///
///     // Update parameters (gradient descent)
///     let learningRate = 0.1
///     for i in 0..<params.count {
///         params[i] -= learningRate * gradients[i]
///     }
///
///     print("Iteration \(iteration): E = \(energy)")
/// }
/// ```
///
/// **QAOA workflow example**:
/// ```swift
/// // Build QAOA circuit for MaxCut
/// var circuit = ParameterizedQuantumCircuit(numQubits: 4)
/// let gamma = Parameter(name: "gamma")
/// let beta = Parameter(name: "beta")
///
/// // Initialize superposition
/// for i in 0..<4 {
///     circuit.append(gate: .concrete(.hadamard), toQubit: i)
/// }
///
/// // Problem layer: exp(-iγH_p) with H_p = Σ (1-ZᵢZⱼ)/2 for edges
/// for (i, j) in edges {
///     circuit.append(gate: .concrete(.cnot(control: i, target: j)), qubits: [])
///     circuit.append(gate: .rotationZ(theta: .parameter(gamma)), toQubit: j)
///     circuit.append(gate: .concrete(.cnot(control: i, target: j)), qubits: [])
/// }
///
/// // Mixer layer: exp(-iβH_m) with H_m = Σ Xᵢ
/// for i in 0..<4 {
///     circuit.append(gate: .rotationX(theta: .parameter(beta)), toQubit: i)
/// }
///
/// // Optimize (γ, β) to maximize cut value
/// let bindings = ["gamma": 0.5, "beta": 1.2]
/// let concrete = try circuit.bind(parameters: bindings)
/// ```
@frozen
public struct ParameterizedQuantumCircuit: Equatable, Sendable, CustomStringConvertible {
    public private(set) var operations: [ParameterizedGateOperation]
    public private(set) var numQubits: Int

    /// Ordered list of parameters (for vector interface)
    /// Parameters registered in order of first appearance during circuit construction
    public private(set) var parameters: [Parameter]

    /// Set of parameter names for O(1) duplicate checking
    private var parameterSet: Set<String>

    // MARK: - Initialization

    /// Create empty parameterized quantum circuit
    /// - Parameter numQubits: Number of qubits (supports 1-30)
    public init(numQubits: Int) {
        ValidationUtilities.validatePositiveQubits(numQubits)
        ValidationUtilities.validateMemoryLimit(numQubits)
        self.numQubits = numQubits
        operations = []
        parameters = []
        parameterSet = Set()
    }

    /// Create circuit with predefined operations and parameters
    /// - Parameters:
    ///   - numQubits: Number of qubits
    ///   - operations: Initial gate operations
    ///   - parameters: Ordered parameter list
    init(numQubits: Int, operations: [ParameterizedGateOperation], parameters: [Parameter]) {
        ValidationUtilities.validatePositiveQubits(numQubits)
        self.numQubits = numQubits
        self.operations = operations
        self.parameters = parameters
        parameterSet = Set(parameters.map(\.name))
    }

    /// Reserve capacity for operations array to avoid reallocations
    /// - Parameter minimumCapacity: Minimum number of operations to reserve space for
    public mutating func reserveCapacity(_ minimumCapacity: Int) {
        operations.reserveCapacity(minimumCapacity)
    }

    // MARK: - Building Methods

    /// Append parameterized gate to circuit
    ///
    /// Adds gate operation and auto-registers any new symbolic parameters.
    /// Parameters are registered in order of first appearance for vector interface.
    /// Circuit auto-expands if gate references qubits beyond current size.
    ///
    /// - Parameters:
    ///   - gate: Parameterized quantum gate
    ///   - qubits: Target qubit indices
    ///   - timestamp: Optional timestamp for animation
    ///
    /// Example:
    /// ```swift
    /// var circuit = ParameterizedQuantumCircuit(numQubits: 2)
    /// let theta = Parameter(name: "theta")
    ///
    /// // Parameterized gate - auto-registers "theta"
    /// circuit.append(gate: .rotationY(theta: .parameter(theta)), qubits: [0])
    ///
    /// // Concrete gate - no parameters
    /// circuit.append(gate: .concrete(.hadamard), qubits: [1])
    ///
    /// // Controlled parameterized gate
    /// circuit.append(gate: .controlledRotationZ(
    ///     theta: .parameter(Parameter(name: "phi")),
    ///     control: 0,
    ///     target: 1
    /// ), qubits: [])
    ///
    /// print(circuit.parameters)  // [theta, phi] in registration order
    /// ```
    @_optimize(speed)
    public mutating func append(gate: ParameterizedGate, qubits: [Int], timestamp: Double? = nil) {
        ValidationUtilities.validateNonNegativeQubits(qubits)

        let maxQubit: Int = qubits.max() ?? -1
        if maxQubit >= numQubits {
            let newNumQubits: Int = maxQubit + 1
            ValidationUtilities.validateMemoryLimit(newNumQubits)
            numQubits = newNumQubits
        }

        registerParameters(from: gate)

        let operation = ParameterizedGateOperation(gate: gate, qubits: qubits, timestamp: timestamp)
        operations.append(operation)
    }

    /// Append single-qubit parameterized gate (convenience)
    /// - Parameters:
    ///   - gate: Parameterized gate
    ///   - qubit: Target qubit index
    ///   - timestamp: Optional timestamp
    public mutating func append(gate: ParameterizedGate, toQubit qubit: Int, timestamp: Double? = nil) {
        append(gate: gate, qubits: [qubit], timestamp: timestamp)
    }

    /// Register new parameters from gate
    /// Auto-registers symbolic parameters in order of first appearance
    @_optimize(speed)
    private mutating func registerParameters(from gate: ParameterizedGate) {
        for param in gate.parameters() {
            if !parameterSet.contains(param.name) {
                parameters.append(param)
                parameterSet.insert(param.name)
            }
        }
    }

    // MARK: - Querying

    /// Number of gate operations in circuit
    @inlinable
    @_effects(readonly)
    public func gateCount() -> Int { operations.count }

    /// Whether circuit is empty (no gates)
    @inlinable
    @_effects(readonly)
    public func isEmpty() -> Bool { operations.isEmpty }

    /// Number of distinct parameters in circuit
    @inlinable
    @_effects(readonly)
    public func parameterCount() -> Int { parameters.count }

    /// Get gate operation at index
    /// - Parameter index: Operation index
    /// - Returns: Parameterized gate operation
    @inlinable
    @_effects(readonly)
    public func operation(at index: Int) -> ParameterizedGateOperation {
        ValidationUtilities.validateIndexInBounds(index, bound: operations.count, name: "Index")
        return operations[index]
    }

    /// Find maximum qubit index used in circuit
    /// - Returns: Maximum qubit index, or numQubits-1 if no operations
    @_optimize(speed)
    @_effects(readonly)
    public func maxQubitUsed() -> Int {
        var maxQubit: Int = numQubits - 1

        for operation in operations {
            let gateMax: Int = switch operation.gate {
            case .phase, .rotationX, .rotationY, .rotationZ, .u1, .u2, .u3:
                operation.qubits.max() ?? -1

            case let .controlledPhase(_, control, target),
                 let .controlledRotationX(_, control, target),
                 let .controlledRotationY(_, control, target),
                 let .controlledRotationZ(_, control, target):
                max(control, target)

            case let .concrete(gate):
                maxQubitForConcreteGate(gate)
            }

            maxQubit = max(maxQubit, gateMax)
        }

        return maxQubit
    }

    /// Extract maximum qubit from concrete gate
    @_optimize(speed)
    @_effects(readonly)
    private func maxQubitForConcreteGate(_ gate: QuantumGate) -> Int {
        switch gate {
        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard,
             .phase, .sGate, .tGate, .rotationX, .rotationY, .rotationZ,
             .u1, .u2, .u3, .sx, .sy, .customSingleQubit: -1

        case let .cnot(control, target),
             let .cz(control, target),
             let .cy(control, target),
             let .ch(control, target),
             let .controlledPhase(_, control, target),
             let .controlledRotationX(_, control, target),
             let .controlledRotationY(_, control, target),
             let .controlledRotationZ(_, control, target),
             let .customTwoQubit(_, control, target): max(control, target)

        case let .swap(q1, q2), let .sqrtSwap(q1, q2): max(q1, q2)

        case let .toffoli(c1, c2, target): max(c1, c2, target)
        }
    }

    // MARK: - Validation

    /// Validate circuit structure and parameter references
    /// - Returns: True if circuit is valid
    @_optimize(speed)
    @_effects(readonly)
    public func validate() -> Bool {
        let maxAllowedQubit = 29

        for operation in operations {
            guard operation.qubits.allSatisfy({ $0 >= 0 && $0 < numQubits }) else {
                return false
            }

            guard operation.gate.validateQubitIndices(maxAllowedQubit: maxAllowedQubit) else {
                return false
            }
        }

        return true
    }

    // MARK: - Parameter Binding (Dictionary Interface)

    /// Bind parameters using name-value dictionary
    ///
    /// Substitutes all symbolic parameters with concrete values to produce
    /// executable `QuantumCircuit`. Validates that all parameters are bound
    /// and detects extra parameters (fail-fast for robustness).
    ///
    /// - Parameter bindings: Dictionary mapping parameter names to values
    /// - Returns: Concrete quantum circuit with all parameters bound
    /// - Throws: ParameterError if parameters missing, extra, or validation fails
    ///
    /// Example:
    /// ```swift
    /// var circuit = ParameterizedQuantumCircuit(numQubits: 2)
    /// circuit.append(gate: .rotationY(theta: .parameter(Parameter(name: "theta"))), toQubit: 0)
    /// circuit.append(gate: .rotationZ(theta: .parameter(Parameter(name: "phi"))), toQubit: 1)
    ///
    /// // Bind all parameters
    /// let bindings = ["theta": Double.pi / 4, "phi": Double.pi / 8]
    /// let concrete = try circuit.bind(parameters: bindings)
    ///
    /// // Execute concrete circuit
    /// let state = concrete.execute()
    ///
    /// // Missing parameter throws error
    /// do {
    ///     let _ = try circuit.bind(parameters: ["theta": 0.5])
    /// } catch ParameterError.unboundParameter(let name) {
    ///     print("Missing: \(name)")  // "Missing: phi"
    /// }
    ///
    /// // Extra parameters throw error (fail-fast)
    /// do {
    ///     let _ = try circuit.bind(parameters: [
    ///         "theta": 0.5, "phi": 1.0, "gamma": 2.0
    ///     ])
    /// } catch ParameterError.extraParameters(let names) {
    ///     print("Extra: \(names)")  // "Extra: [gamma]"
    /// }
    /// ```
    @_optimize(speed)
    @_eagerMove
    public func bind(parameters bindings: [String: Double]) throws -> QuantumCircuit {
        for param in parameters {
            if bindings[param.name] == nil {
                throw ParameterError.unboundParameter(param.name)
            }
        }

        var extraParams: [String] = []
        for key in bindings.keys {
            if !parameterSet.contains(key) {
                extraParams.append(key)
            }
        }
        if !extraParams.isEmpty {
            throw ParameterError.extraParameters(extraParams)
        }

        var concreteCircuit = QuantumCircuit(numQubits: numQubits)

        for operation in operations {
            let concreteGate = try operation.gate.bind(with: bindings)
            concreteCircuit.append(gate: concreteGate, qubits: operation.qubits, timestamp: operation.timestamp)
        }

        return concreteCircuit
    }

    // MARK: - Parameter Binding (Vector Interface)

    /// Bind parameters using numerical vector
    ///
    /// Vector interface compatible with NumPy, SciPy, and classical optimizers.
    /// Parameters bound in registration order (order of first appearance in circuit).
    /// Standard interface for gradient-based optimization algorithms.
    ///
    /// - Parameter parameterVector: Array of parameter values (length must match parameter count)
    /// - Returns: Concrete quantum circuit with all parameters bound
    /// - Throws: ParameterError.invalidVectorLength if length mismatch
    ///
    /// Example:
    /// ```swift
    /// var circuit = ParameterizedQuantumCircuit(numQubits: 3)
    ///
    /// // Parameters registered in order: theta_0, theta_1, theta_2
    /// for i in 0..<3 {
    ///     let param = Parameter(name: "theta_\(i)")
    ///     circuit.append(gate: .rotationY(theta: .parameter(param)), toQubit: i)
    /// }
    ///
    /// // Bind using vector (NumPy-style)
    /// let params: [Double] = [0.1, 0.2, 0.3]
    /// let concrete = try circuit.bind(parameterVector: params)
    ///
    /// // Wrong length throws error
    /// do {
    ///     let _ = try circuit.bind(parameterVector: [0.1, 0.2])
    /// } catch ParameterError.invalidVectorLength(let expected, let got) {
    ///     print("Expected \(expected), got \(got)")  // "Expected 3, got 2"
    /// }
    ///
    /// // Usage with optimizer
    /// func objectiveFunction(_ params: [Double]) throws -> Double {
    ///     let circuit = try buildAnsatz()
    ///     let concrete = try circuit.bind(parameterVector: params)
    ///     let state = concrete.execute()
    ///     return hamiltonian.expectation(in: state)
    /// }
    /// ```
    @_optimize(speed)
    @_eagerMove
    public func bind(parameterVector: [Double]) throws -> QuantumCircuit {
        let paramCount: Int = parameters.count
        guard parameterVector.count == paramCount else {
            throw ParameterError.invalidVectorLength(expected: paramCount, got: parameterVector.count)
        }

        let bindings = Dictionary(
            uniqueKeysWithValues: zip(parameters.lazy.map(\.name), parameterVector)
        )

        return try bind(parameters: bindings)
    }

    // MARK: - Gradient Computation Support

    /// Generate circuits for parameter shift rule gradient computation
    ///
    /// Creates two circuits with parameter shifted by ±shift for gradient evaluation.
    /// Implements parameter shift rule: ∂⟨H⟩/∂θᵢ = [⟨H⟩(θᵢ+s) - ⟨H⟩(θᵢ-s)]/(2sin(s))
    /// where s = π/2 gives: ∂⟨H⟩/∂θᵢ = [⟨H⟩(θᵢ+π/2) - ⟨H⟩(θᵢ-π/2)]/2
    ///
    /// More accurate than finite differences for quantum circuits. Enables
    /// gradient-based optimization algorithms (L-BFGS-B, Adam, natural gradient).
    ///
    /// - Parameters:
    ///   - parameterName: Name of parameter to shift
    ///   - baseBindings: Base parameter values (all parameters must be present)
    ///   - shift: Shift amount (default: π/2 for standard parameter shift rule)
    /// - Returns: Tuple of (plus, minus) concrete circuits with parameter shifted
    /// - Throws: ParameterError if parameter not found or bindings invalid
    ///
    /// Example:
    /// ```swift
    /// var circuit = ParameterizedQuantumCircuit(numQubits: 2)
    /// circuit.append(gate: .rotationY(theta: .parameter(Parameter(name: "theta"))), toQubit: 0)
    /// circuit.append(gate: .rotationZ(theta: .parameter(Parameter(name: "phi"))), toQubit: 1)
    ///
    /// let baseParams = ["theta": 0.5, "phi": 1.0]
    ///
    /// // Compute gradient ∂⟨H⟩/∂theta
    /// let (plus, minus) = try circuit.generateShiftedCircuits(
    ///     parameterName: "theta",
    ///     baseBindings: baseParams
    /// )
    ///
    /// let statePlus = plus.execute()
    /// let stateMinus = minus.execute()
    /// let energyPlus = hamiltonian.expectation(in: statePlus)
    /// let energyMinus = hamiltonian.expectation(in: stateMinus)
    ///
    /// let gradient = (energyPlus - energyMinus) / 2.0
    /// ```
    @_optimize(speed)
    @_eagerMove
    public func generateShiftedCircuits(
        parameterName: String,
        baseBindings: [String: Double],
        shift: Double = .pi / 2
    ) throws -> (plus: QuantumCircuit, minus: QuantumCircuit) {
        guard parameterSet.contains(parameterName) else {
            throw ParameterError.parameterNotFound(parameterName)
        }

        guard let baseValue = baseBindings[parameterName] else {
            throw ParameterError.unboundParameter(parameterName)
        }

        var plusBindings = baseBindings
        var minusBindings = baseBindings

        plusBindings[parameterName] = baseValue + shift
        minusBindings[parameterName] = baseValue - shift

        let plusCircuit = try bind(parameters: plusBindings)
        let minusCircuit = try bind(parameters: minusBindings)

        return (plus: plusCircuit, minus: minusCircuit)
    }

    /// Generate shifted circuits using parameter index (vector interface)
    ///
    /// Vector-based interface for parameter shifting compatible with array-based optimizers.
    ///
    /// - Parameters:
    ///   - parameterIndex: Index of parameter in registration order
    ///   - baseVector: Base parameter values (length must match parameter count)
    ///   - shift: Shift amount (default: π/2)
    /// - Returns: Tuple of (plus, minus) concrete circuits
    /// - Throws: ParameterError if index invalid or vector length wrong
    ///
    /// Example:
    /// ```swift
    /// var circuit = ParameterizedQuantumCircuit(numQubits: 2)
    /// for i in 0..<2 {
    ///     let param = Parameter(name: "theta_\(i)")
    ///     circuit.append(gate: .rotationY(theta: .parameter(param)), toQubit: i)
    /// }
    ///
    /// let baseParams: [Double] = [0.5, 1.0]
    ///
    /// // Compute gradient for parameter 0
    /// let (plus, minus) = try circuit.generateShiftedCircuits(
    ///     parameterIndex: 0,
    ///     baseVector: baseParams
    /// )
    /// ```
    @_optimize(speed)
    @_eagerMove
    public func generateShiftedCircuits(
        parameterIndex: Int,
        baseVector: [Double],
        shift: Double = .pi / 2
    ) throws -> (plus: QuantumCircuit, minus: QuantumCircuit) {
        let paramCount: Int = parameters.count
        guard parameterIndex >= 0, parameterIndex < paramCount else {
            throw ParameterError.parameterIndexOutOfBounds(index: parameterIndex, count: paramCount)
        }

        guard baseVector.count == paramCount else {
            throw ParameterError.invalidVectorLength(expected: paramCount, got: baseVector.count)
        }

        let baseBindings = Dictionary(
            uniqueKeysWithValues: zip(parameters.lazy.map(\.name), baseVector)
        )

        let paramName = parameters[parameterIndex].name
        return try generateShiftedCircuits(parameterName: paramName, baseBindings: baseBindings, shift: shift)
    }

    // MARK: - CustomStringConvertible

    /// String representation of parameterized circuit
    public var description: String {
        if operations.isEmpty {
            return "ParameterizedQuantumCircuit(\(numQubits) qubits, \(parameters.count) params, empty)"
        }

        var gateList = ""
        let gateLimit: Int = min(operations.count, 3)
        for i in 0 ..< gateLimit {
            if i > 0 { gateList += ", " }
            gateList += operations[i].description
        }
        let suffix = operations.count > 3 ? ", ..." : ""

        var paramList = ""
        let paramLimit: Int = min(parameters.count, 3)
        for i in 0 ..< paramLimit {
            if i > 0 { paramList += ", " }
            paramList += parameters[i].name
        }
        let paramSuffix = parameters.count > 3 ? ", ..." : ""

        return """
        ParameterizedQuantumCircuit(\(numQubits) qubits, \(operations.count) gates, \
        \(parameters.count) params: [\(paramList)\(paramSuffix)]): \(gateList)\(suffix)
        """
    }
}
