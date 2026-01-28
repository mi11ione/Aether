// Copyright (c) 2025-2026 Roman Zhuzhgov, Apache License 2.0

import GameplayKit

/// Asynchronous actor for efficient Clifford circuit simulation using stabilizer formalism.
///
/// Provides thread-safe execution of Clifford-only quantum circuits with O(n^2) gate complexity
/// instead of exponential statevector simulation. Supports batch sampling with millions of shots
/// through frame-based simulation that compiles the circuit once then samples efficiently.
///
/// **Example:**
/// ```swift
/// let simulator = CliffordSimulator()
///
/// var circuit = QuantumCircuit(qubits: 100)
/// circuit.append(.hadamard, to: 0)
/// for i in 1..<100 {
///     circuit.append(.cnot, to: [0, i])
/// }
///
/// let tableau = await simulator.execute(circuit)
/// let samples = await simulator.sample(circuit, shots: 1_000_000, seed: 42)
/// ```
///
/// - SeeAlso: ``StabilizerTableau``
/// - SeeAlso: ``CliffordGateClassifier``
/// - SeeAlso: ``QuantumCircuit``
public actor CliffordSimulator {
    /// Creates a new Clifford circuit simulator.
    ///
    /// Initializes an actor-isolated simulator for thread-safe Clifford circuit execution.
    ///
    /// **Example:**
    /// ```swift
    /// let simulator = CliffordSimulator()
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
    /// circuit.append(.cnot, to: [0, 2])
    ///
    /// let tableau = await simulator.execute(circuit)
    /// let (p0, p1) = tableau.probability(of: 0, measuring: .z)
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
    ///
    /// var initial = StabilizerTableau(qubits: 2)
    /// initial.apply(.hadamard, to: 0)
    ///
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.cnot, to: [0, 1])
    ///
    /// let final = await simulator.execute(circuit, from: initial)
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit containing only Clifford gates
    ///   - from: Starting stabilizer tableau state
    /// - Returns: Final stabilizer tableau after applying all circuit operations
    /// - Complexity: O(n^2 * g / w) where n = qubits, g = gate count, w = 64 (word size)
    @_optimize(speed)
    public func execute(_ circuit: QuantumCircuit, from: StabilizerTableau) async -> StabilizerTableau {
        validateCliffordCircuit(circuit)

        var tableau = from

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
    ///
    /// var circuit = QuantumCircuit(qubits: 10)
    /// circuit.append(.hadamard, to: 0)
    /// for i in 1..<10 {
    ///     circuit.append(.cnot, to: [0, i])
    /// }
    ///
    /// let samples = await simulator.sample(circuit, shots: 1_000_000, seed: 42)
    /// let countZero = samples.filter { $0 == 0 }.count
    /// let countMax = samples.filter { $0 == 1023 }.count
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit containing only Clifford gates
    ///   - shots: Number of measurement samples to collect (must be positive)
    ///   - seed: Optional seed for reproducible random results
    /// - Returns: Array of n-qubit measurement outcomes (each 0 to 2^n - 1)
    /// - Complexity: O(n^2 * g / w + shots * n / w) where n = qubits, g = gates, w = 64
    @_optimize(speed)
    public func sample(_ circuit: QuantumCircuit, shots: Int, seed: UInt64?) async -> [Int] {
        ValidationUtilities.validatePositiveInt(shots, name: "shots")
        validateCliffordCircuit(circuit)

        let n = determineQubitCount(circuit)
        var tableau = StabilizerTableau(qubits: n)

        for operation in circuit.operations {
            applyOperation(operation, to: &tableau)
        }

        return tableau.sample(shots: shots, seed: seed)
    }

    @inline(__always)
    private func determineQubitCount(_ circuit: QuantumCircuit) -> Int {
        let maxQubit = circuit.highestQubitIndex
        return max(circuit.qubits, maxQubit + 1)
    }

    @inline(__always)
    private func validateCliffordCircuit(_ circuit: QuantumCircuit) {
        let analysis = CliffordGateClassifier.analyze(circuit)
        precondition(analysis.isClifford, "CliffordSimulator requires Clifford-only circuits (found \(analysis.tCount) T-equivalent gates)")
    }

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

    @inline(__always)
    @_optimize(speed)
    private func applyGate(_ gate: QuantumGate, qubits: [Int], to tableau: inout StabilizerTableau) {
        switch gate {
        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard, .sGate, .sx, .sy:
            tableau.apply(gate, to: qubits[0])
        case .cnot, .cz, .swap:
            tableau.apply(gate, to: qubits)
        case .cy:
            tableau.apply(.sGate, to: qubits[1])
            tableau.apply(.sGate, to: qubits[1])
            tableau.apply(.sGate, to: qubits[1])
            tableau.apply(.cnot, to: qubits)
            tableau.apply(.sGate, to: qubits[1])
        case .ch:
            tableau.apply(.sGate, to: qubits[1])
            tableau.apply(.hadamard, to: qubits[1])
            tableau.apply(.sGate, to: qubits[1])
            tableau.apply(.cnot, to: qubits)
            tableau.apply(.hadamard, to: qubits[1])
            tableau.apply(.sGate, to: qubits[1])
            tableau.apply(.cnot, to: qubits)
            tableau.apply(.sGate, to: qubits[1])
            tableau.apply(.sGate, to: qubits[1])
            tableau.apply(.sGate, to: qubits[1])
        case .iswap:
            tableau.apply(.swap, to: qubits)
            tableau.apply(.sGate, to: qubits[0])
            tableau.apply(.sGate, to: qubits[1])
            tableau.apply(.cz, to: qubits)
        case let .phase(angle):
            if case let .value(theta) = angle {
                applyCliffordPhase(theta, qubit: qubits[0], to: &tableau)
            }
        case let .controlled(innerGate, _):
            applyControlledClifford(innerGate, control: qubits[0], target: qubits[1], to: &tableau)
        default:
            break
        }
    }

    @inline(__always)
    private func applyCliffordPhase(_ theta: Double, qubit: Int, to tableau: inout StabilizerTableau) {
        let normalized = theta.truncatingRemainder(dividingBy: 2.0 * .pi)
        let adjusted = normalized < 0 ? normalized + 2.0 * .pi : normalized
        let tolerance = 1e-10

        if abs(adjusted - .pi / 2.0) < tolerance || abs(adjusted - 5.0 * .pi / 2.0) < tolerance {
            tableau.apply(.sGate, to: qubit)
        } else if abs(adjusted - .pi) < tolerance {
            tableau.apply(.pauliZ, to: qubit)
        } else if abs(adjusted - 3.0 * .pi / 2.0) < tolerance || abs(adjusted - 7.0 * .pi / 2.0) < tolerance {
            tableau.apply(.sGate, to: qubit)
            tableau.apply(.sGate, to: qubit)
            tableau.apply(.sGate, to: qubit)
        }
    }

    @inline(__always)
    private func applyControlledClifford(_ innerGate: QuantumGate, control: Int, target: Int, to tableau: inout StabilizerTableau) {
        switch innerGate {
        case .pauliX:
            tableau.apply(.cnot, to: [control, target])
        case .pauliY:
            tableau.apply(.sGate, to: target)
            tableau.apply(.sGate, to: target)
            tableau.apply(.sGate, to: target)
            tableau.apply(.cnot, to: [control, target])
            tableau.apply(.sGate, to: target)
        case .pauliZ:
            tableau.apply(.cz, to: [control, target])
        case .sGate:
            tableau.apply(.cz, to: [control, target])
            tableau.apply(.sGate, to: target)
        default:
            break
        }
    }
}
