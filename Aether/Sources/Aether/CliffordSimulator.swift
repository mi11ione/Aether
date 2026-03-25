// Copyright (c) 2025-2026 Roman Zhuzhgov, Apache License 2.0

/// Asynchronous actor for efficient Clifford circuit simulation using stabilizer formalism.
///
/// Provides thread-safe execution of Clifford-only quantum circuits with O(n^2) gate complexity
/// instead of exponential statevector simulation. Supports batch sampling with millions of shots
/// through frame-based simulation that compiles the circuit once then samples efficiently.
///
/// **Example:**
/// ```swift
/// let simulator = CliffordSimulator()
/// var circuit = QuantumCircuit(qubits: 3)
/// circuit.append(.hadamard, to: 0)
/// circuit.append(.cnot, to: [0, 1])
/// let tableau = await simulator.execute(circuit)
/// ```
///
/// - SeeAlso: ``StabilizerTableau``
/// - SeeAlso: ``CliffordGateClassifier``
/// - SeeAlso: ``QuantumCircuit``
public actor CliffordSimulator {
    private static let twoPi = 2.0 * Double.pi
    private static let halfPi = Double.pi / 2.0
    private static let tolerance: Double = 1e-10

    /// Creates a new Clifford circuit simulator.
    ///
    /// Initializes an actor-isolated simulator for thread-safe Clifford circuit execution.
    ///
    /// **Example:**
    /// ```swift
    /// let simulator = CliffordSimulator()
    /// let circuit = QuantumCircuit(qubits: 2)
    /// let tableau = await simulator.execute(circuit)
    /// ```
    public init() {}

    /// Executes a Clifford circuit starting from the |0...0> ground state.
    ///
    /// Applies all gates in the circuit sequentially to transform the initial ground state
    /// tableau through Clifford operations. Only Clifford gates are supported; circuits
    /// containing non-Clifford gates (T, arbitrary rotations) will produce undefined results.
    ///
    /// **Example:**
    /// ```swift
    /// let simulator = CliffordSimulator()
    /// var circuit = QuantumCircuit(qubits: 3)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    /// let tableau = await simulator.execute(circuit)
    /// ```
    ///
    /// - Parameter circuit: Quantum circuit containing only Clifford gates
    /// - Returns: Final stabilizer tableau after applying all circuit operations
    /// - Complexity: O(n^2 * g / w) where n = qubits, g = gate count, w = 64 (word size)
    ///
    /// - SeeAlso: ``execute(_:from:)`` for execution from custom initial state
    @_optimize(speed)
    public func execute(_ circuit: QuantumCircuit) async -> StabilizerTableau {
        let qubits = determineQubitCount(circuit)
        let tableau = StabilizerTableau(qubits: qubits)
        return await execute(circuit, from: tableau)
    }

    /// Executes a Clifford circuit starting from a custom initial stabilizer state.
    ///
    /// Applies all gates in the circuit sequentially to transform the provided initial
    /// tableau through Clifford operations. The initial tableau must have sufficient qubits
    /// for all circuit operations.
    ///
    /// **Example:**
    /// ```swift
    /// let simulator = CliffordSimulator()
    /// var initial = StabilizerTableau(qubits: 2)
    /// initial.apply(.hadamard, to: 0)
    /// var circuit = QuantumCircuit(qubits: 2)
    /// let result = await simulator.execute(circuit, from: initial)
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit containing only Clifford gates
    ///   - from: Starting stabilizer tableau state
    /// - Returns: Final stabilizer tableau after applying all circuit operations
    /// - Precondition: Circuit contains only Clifford gates (no T or non-Clifford rotations).
    /// - Complexity: O(n^2 * g / w) where n = qubits, g = gate count, w = 64 (word size)
    @_optimize(speed)
    public func execute(_ circuit: QuantumCircuit, from initial: StabilizerTableau) async -> StabilizerTableau {
        validateCliffordCircuit(circuit)

        var tableau = initial

        for operation in circuit.operations {
            applyOperation(operation, to: &tableau)
        }

        return tableau
    }

    /// Samples measurement outcomes from a Clifford circuit with batch optimization.
    ///
    /// Performs efficient batch sampling by compiling the circuit once into a stabilizer
    /// tableau, then sampling multiple independent measurement outcomes. Uses frame-based
    /// simulation achieving O(shots * n / w) complexity after O(n^2) circuit compilation,
    /// enabling millions of shots efficiently.
    ///
    /// **Example:**
    /// ```swift
    /// let simulator = CliffordSimulator()
    /// var circuit = QuantumCircuit(qubits: 3)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    /// let samples = await simulator.sample(circuit, shots: 1000, seed: 42)
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit containing only Clifford gates
    ///   - shots: Number of measurement samples to collect (must be positive)
    ///   - seed: Optional seed for reproducible random results
    /// - Returns: Array of n-qubit measurement outcomes (each 0 to 2^n - 1)
    /// - Precondition: shots > 0
    /// - Complexity: O(n^2 * g / w + shots * n / w) where n = qubits, g = gates, w = 64
    @_optimize(speed)
    public func sample(_ circuit: QuantumCircuit, shots: Int, seed: UInt64?) async -> [Int] {
        ValidationUtilities.validatePositiveInt(shots, name: "shots")
        var tableau = await execute(circuit)
        return tableau.sample(shots: shots, seed: seed)
    }

    /// Returns the qubit count for the given circuit.
    @inline(__always)
    private func determineQubitCount(_ circuit: QuantumCircuit) -> Int {
        let maxQubit = circuit.highestQubitIndex
        return max(circuit.qubits, maxQubit + 1)
    }

    /// Validates that the circuit contains only Clifford gates.
    @inline(__always)
    private func validateCliffordCircuit(_ circuit: QuantumCircuit) {
        ValidationUtilities.validateCliffordCircuit(circuit)
    }

    /// Dispatches a circuit operation to the appropriate gate application.
    @inline(__always)
    @_optimize(speed)
    private func applyOperation(_ operation: CircuitOperation, to tableau: inout StabilizerTableau) {
        switch operation {
        case let .gate(gate, qubits, _):
            applyGate(gate, qubits: qubits, to: &tableau)
        case let .reset(qubit, _):
            _ = tableau.measure(qubit, seed: nil)
        case let .measure(qubit, _, _):
            _ = tableau.measure(qubit, seed: nil)
        }
    }

    /// Applies a single quantum gate to the stabilizer tableau.
    @inline(__always)
    @_optimize(speed)
    private func applyGate(_ gate: QuantumGate, qubits: [Int], to tableau: inout StabilizerTableau) {
        switch gate {
        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard, .sGate, .sx, .sy:
            tableau.apply(gate, to: qubits[0])
        case .cnot, .cz, .swap:
            tableau.apply(gate, to: qubits)
        case .cy:
            let target = qubits[1]
            tableau.apply(.sGate, to: target)
            tableau.apply(.sGate, to: target)
            tableau.apply(.sGate, to: target)
            tableau.apply(.cnot, to: qubits)
            tableau.apply(.sGate, to: target)
        case .ch:
            let target = qubits[1]
            tableau.apply(.sGate, to: target)
            tableau.apply(.hadamard, to: target)
            tableau.apply(.sGate, to: target)
            tableau.apply(.cnot, to: qubits)
            tableau.apply(.hadamard, to: target)
            tableau.apply(.sGate, to: target)
            tableau.apply(.cnot, to: qubits)
            tableau.apply(.sGate, to: target)
            tableau.apply(.sGate, to: target)
            tableau.apply(.sGate, to: target)
        case .iswap:
            let q0 = qubits[0]
            let q1 = qubits[1]
            tableau.apply(.swap, to: qubits)
            tableau.apply(.sGate, to: q0)
            tableau.apply(.sGate, to: q1)
            tableau.apply(.cz, to: qubits)
        case let .phase(angle):
            if case let .value(theta) = angle {
                applyCliffordPhase(theta, qubit: qubits[0], to: &tableau)
            }
        case let .controlled(innerGate, _):
            applyControlledClifford(innerGate, control: qubits[0], target: qubits[1], qubits: qubits, to: &tableau)
        default:
            break
        }
    }

    /// Applies a Clifford-compatible phase rotation to the tableau.
    @inline(__always)
    @_optimize(speed)
    private func applyCliffordPhase(_ theta: Double, qubit: Int, to tableau: inout StabilizerTableau) {
        let normalized = theta.truncatingRemainder(dividingBy: Self.twoPi)
        let adjusted = (normalized + Self.twoPi).truncatingRemainder(dividingBy: Self.twoPi)

        if abs(adjusted - Self.halfPi) < Self.tolerance {
            tableau.apply(.sGate, to: qubit)
        } else if abs(adjusted - .pi) < Self.tolerance {
            tableau.apply(.pauliZ, to: qubit)
        } else if abs(adjusted - 3.0 * Self.halfPi) < Self.tolerance {
            tableau.apply(.sGate, to: qubit)
            tableau.apply(.sGate, to: qubit)
            tableau.apply(.sGate, to: qubit)
        }
    }

    /// Decomposes and applies a controlled Clifford gate to the tableau.
    @inline(__always)
    @_optimize(speed)
    private func applyControlledClifford(_ innerGate: QuantumGate, control _: Int, target: Int, qubits: [Int], to tableau: inout StabilizerTableau) {
        switch innerGate {
        case .pauliX:
            tableau.apply(.cnot, to: qubits)
        case .pauliY:
            tableau.apply(.sGate, to: target)
            tableau.apply(.sGate, to: target)
            tableau.apply(.sGate, to: target)
            tableau.apply(.cnot, to: qubits)
            tableau.apply(.sGate, to: target)
        case .pauliZ:
            tableau.apply(.cz, to: qubits)
        case .sGate:
            tableau.apply(.cz, to: qubits)
            tableau.apply(.sGate, to: target)
        default:
            break
        }
    }
}
