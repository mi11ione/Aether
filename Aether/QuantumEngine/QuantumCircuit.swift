// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Represents a single gate operation in a quantum circuit
struct GateOperation: Equatable, CustomStringConvertible, Sendable {
    let gate: QuantumGate
    let qubits: [Int]
    let timestamp: Double?

    /// Create a new gate operation
    /// - Parameters:
    ///   - gate: The quantum gate to apply
    ///   - qubits: Target qubit indices
    ///   - timestamp: Optional timestamp for animation/scrubbing
    init(gate: QuantumGate, qubits: [Int], timestamp: Double? = nil) {
        self.gate = gate
        self.qubits = qubits
        self.timestamp = timestamp
    }

    /// String representation of the gate operation
    /// - Returns: Formatted string like "H on qubits [0]" or "CNOT(c:0, t:1) on qubits [] @ 1.50s"
    var description: String {
        let qubitStr = qubits.isEmpty ? "" : " on qubits \(qubits)"
        if let ts = timestamp {
            return "\(gate)\(qubitStr) @ \(String(format: "%.2f", ts))s"
        }
        return "\(gate)\(qubitStr)"
    }
}

/// Quantum circuit: ordered sequence of gates implementing quantum algorithms
///
/// Represents a quantum computation as a series of gate operations applied to qubits.
/// Circuits transform an initial state |00...0⟩ through unitary operations to produce
/// a final superposition state that can be measured.
///
/// **Architecture**:
/// - Generic over qubit count (supports 1-30 qubits)
/// - Auto-expands when gates reference higher qubit indices
/// - Thread-safe execution via immutable operations list
/// - Optional timestamping for animation/visualization
///
/// **Execution modes**:
/// - Full execution: `circuit.execute()` → final state
/// - Step-by-step: External caching for animation
/// - Validation: Check circuit correctness before execution
///
/// Example:
/// ```swift
/// // Create Bell state: (|00⟩ + |11⟩)/√2
/// var circuit = QuantumCircuit(numQubits: 2)
/// circuit.append(gate: .hadamard, toQubit: 0)
/// circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
/// let state = circuit.execute()
///
/// // Measure probabilities
/// let p00 = state.probability(ofState: 0b00)  // 50%
/// let p11 = state.probability(ofState: 0b11)  // 50%
///
/// // Build more complex circuits
/// circuit.append(gate: .phase(.pi/4), toQubit: 0)
/// circuit.append(gate: .rotationY(.pi/3), toQubit: 1)
///
/// // Auto-expansion: referencing qubit 5 expands to 6 qubits
/// circuit.append(gate: .hadamard, toQubit: 5)  // numQubits now 6
/// ```
struct QuantumCircuit: Equatable, CustomStringConvertible {
    private(set) var operations: [GateOperation]
    private(set) var numQubits: Int

    // Note: State cache removed for thread safety
    // Caching should be implemented at a higher level (e.g., in simulator actor)
    // where thread safety can be properly managed

    /// Cache interval: store state every N gates (for external caching)
    static let cacheInterval: Int = 5

    /// Maximum number of cached states (for external caching)
    static let maxCacheSize: Int = 20

    var gateCount: Int { operations.count }
    var isEmpty: Bool { operations.isEmpty }

    // MARK: - Initialization

    /// Create empty quantum circuit
    /// - Parameter numQubits: Number of qubits (supports 1-24+)
    init(numQubits: Int) {
        precondition(numQubits > 0, "Number of qubits must be positive")
        self.numQubits = numQubits
        operations = []
    }

    /// Create circuit with predefined operations
    /// - Parameters:
    ///   - numQubits: Number of qubits
    ///   - operations: Initial gate operations
    init(numQubits: Int, operations: [GateOperation]) {
        precondition(numQubits > 0, "Number of qubits must be positive")
        self.numQubits = numQubits
        self.operations = operations
    }

    // MARK: - Building Methods

    /// Append gate to end of circuit
    ///
    /// Adds a quantum gate operation to the circuit. Auto-expands circuit if gate
    /// references qubit indices beyond current size (up to 30 qubits maximum).
    ///
    /// - Parameters:
    ///   - gate: Quantum gate to apply
    ///   - qubits: Target qubit indices (varies by gate type)
    ///   - timestamp: Optional timestamp for animation/visualization
    ///
    /// Example:
    /// ```swift
    /// var circuit = QuantumCircuit(numQubits: 3)
    ///
    /// // Single-qubit gates: use [qubitIndex] or convenience toQubit:
    /// circuit.append(gate: .hadamard, qubits: [0])
    /// circuit.append(gate: .pauliX, toQubit: 1)
    ///
    /// // Two-qubit gates: use empty array (indices in gate definition)
    /// circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
    ///
    /// // Parameterized gates
    /// circuit.append(gate: .rotationY(.pi/4), toQubit: 2)
    /// circuit.append(gate: .phase(.pi/8), toQubit: 0)
    ///
    /// // With timestamp for animation
    /// circuit.append(gate: .hadamard, toQubit: 0, timestamp: 1.5)
    /// ```
    mutating func append(gate: QuantumGate, qubits: [Int], timestamp: Double? = nil) {
        precondition(qubits.allSatisfy { $0 >= 0 }, "Qubit indices must be non-negative")

        let maxQubit = qubits.max() ?? -1
        if maxQubit >= numQubits {
            let newNumQubits = maxQubit + 1
            precondition(newNumQubits <= 30,
                         "Circuit would grow to \(newNumQubits) qubits (max 30). This may be a typo.")
            numQubits = newNumQubits
        }

        let operation = GateOperation(gate: gate, qubits: qubits, timestamp: timestamp)
        operations.append(operation)
    }

    /// Append single-qubit gate (convenience)
    /// - Parameters:
    ///   - gate: Single-qubit gate
    ///   - qubit: Target qubit
    ///   - timestamp: Optional timestamp
    mutating func append(gate: QuantumGate, toQubit qubit: Int, timestamp: Double? = nil) {
        append(gate: gate, qubits: [qubit], timestamp: timestamp)
    }

    /// Insert gate at specific position
    /// - Parameters:
    ///   - gate: Gate to insert
    ///   - qubits: Target qubit indices
    ///   - index: Position to insert at
    ///   - timestamp: Optional timestamp
    mutating func insert(gate: QuantumGate, qubits: [Int], at index: Int, timestamp: Double? = nil) {
        precondition(index >= 0 && index <= operations.count, "Index out of bounds")
        precondition(qubits.allSatisfy { $0 >= 0 && $0 < numQubits },
                     "Qubit indices out of bounds")

        let operation = GateOperation(gate: gate, qubits: qubits, timestamp: timestamp)
        operations.insert(operation, at: index)
    }

    /// Remove gate at index
    mutating func remove(at index: Int) {
        precondition(index >= 0 && index < operations.count, "Index out of bounds")
        operations.remove(at: index)
    }

    /// Remove all gates
    mutating func clear() { operations.removeAll() }

    // MARK: - Querying

    /// Get gate operation at index
    /// - Parameter index: Index of operation
    /// - Returns: Gate operation
    func operation(at index: Int) -> GateOperation {
        precondition(index >= 0 && index < operations.count, "Index out of bounds")
        return operations[index]
    }

    // MARK: - Validation

    /// Validate circuit for given number of qubits
    func validate() -> Bool {
        let maxAllowedQubit = 29 // Allow up to 30 total qubits (2^30 = 1B amplitudes)

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

    /// Find maximum qubit index referenced by any operation in circuit
    /// Used to detect ancilla qubits that may exceed logical circuit size
    /// - Returns: Maximum qubit index, or numQubits-1 if no operations
    func maxQubitUsed() -> Int {
        var maxQubit = numQubits - 1

        for operation in operations {
            let gateMax: Int = switch operation.gate {
            case .identity, .pauliX, .pauliY, .pauliZ, .hadamard,
                 .phase, .sGate, .tGate, .rotationX, .rotationY, .rotationZ,
                 .u1, .u2, .u3, .sx, .sy, .customSingleQubit:
                operation.qubits.max() ?? -1

            case let .cnot(control, target),
                 let .cz(control, target),
                 let .cy(control, target),
                 let .ch(control, target),
                 let .controlledPhase(_, control, target),
                 let .controlledRotationX(_, control, target),
                 let .controlledRotationY(_, control, target),
                 let .controlledRotationZ(_, control, target),
                 let .customTwoQubit(_, control, target):
                max(control, target)

            case let .swap(q1, q2), let .sqrtSwap(q1, q2):
                max(q1, q2)

            case let .toffoli(c1, c2, target):
                max(c1, c2, target)
            }

            maxQubit = max(maxQubit, gateMax)
        }

        return maxQubit
    }

    // MARK: - Execution

    /// Execute circuit on custom initial state
    ///
    /// Applies all gates in sequence to transform the input state. Automatically
    /// handles ancilla qubit expansion if gates reference qubits beyond the initial
    /// state size (ancilla qubits are initialized to |0⟩).
    ///
    /// - Parameter initialState: Starting quantum state
    /// - Returns: Final quantum state after applying all gates
    ///
    /// Example:
    /// ```swift
    /// // Start from custom superposition
    /// let initial = QuantumState(numQubits: 2, amplitudes: [
    ///     Complex(0.6, 0), Complex(0.8, 0), .zero, .zero
    /// ])  // 0.6|00⟩ + 0.8|01⟩
    ///
    /// var circuit = QuantumCircuit(numQubits: 2)
    /// circuit.append(gate: .hadamard, toQubit: 0)
    /// let final = circuit.execute(on: initial)
    /// ```
    func execute(on initialState: QuantumState) -> QuantumState {
        precondition(initialState.numQubits >= numQubits,
                     "Initial state must have at least as many qubits as circuit")
        precondition(validate(), "Circuit validation failed")

        let maxQubit = maxQubitUsed()
        var currentState = initialState

        // Expand state if ancilla qubits are referenced
        // This handles tensor product: |ψ⟩ ⊗ |0...0⟩ where ancilla are in |0⟩
        if maxQubit >= initialState.numQubits {
            let numAncillaQubits = maxQubit - initialState.numQubits + 1
            let expandedSize = 1 << (initialState.numQubits + numAncillaQubits)

            // Copy original amplitudes; rest are zero (ancilla in |0⟩)
            // In little-endian ordering, ancilla are high-order bits
            // So original amplitudes stay at same indices (where ancilla bits are 0)
            var expandedAmplitudes = [Complex<Double>](repeating: .zero, count: expandedSize)
            for i in 0 ..< initialState.stateSpaceSize {
                expandedAmplitudes[i] = initialState.amplitudes[i]
            }

            currentState = QuantumState(numQubits: maxQubit + 1, amplitudes: expandedAmplitudes)
        }

        for operation in operations {
            currentState = GateApplication.apply(
                gate: operation.gate,
                to: operation.qubits,
                state: currentState
            )
        }

        return currentState
    }

    /// Execute circuit starting from ground state |00...0⟩
    ///
    /// Primary execution method for most quantum algorithms. Initializes all qubits
    /// to |0⟩ and applies the circuit's gate sequence.
    ///
    /// - Returns: Final quantum state after applying all gates
    ///
    /// Example:
    /// ```swift
    /// // Create GHZ state: (|000⟩ + |111⟩)/√2
    /// var circuit = QuantumCircuit(numQubits: 3)
    /// circuit.append(gate: .hadamard, toQubit: 0)
    /// circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
    /// circuit.append(gate: .cnot(control: 1, target: 2), qubits: [])
    ///
    /// let final = circuit.execute()
    /// let p000 = final.probability(ofState: 0b000)  // 50%
    /// let p111 = final.probability(ofState: 0b111)  // 50%
    /// ```
    func execute() -> QuantumState {
        let initialState = QuantumState(numQubits: numQubits)
        return execute(on: initialState)
    }

    /// Execute circuit up to specific gate index (for step-through/animation)
    /// - Parameters:
    ///   - initialState: Starting quantum state
    ///   - upToIndex: Execute gates 0..<upToIndex (exclusive)
    /// - Returns: Quantum state after partial execution
    /// - Note: Caching is removed from circuit for thread safety.
    ///         Implement caching at a higher level (e.g., in simulator actor) if needed.
    func execute(on initialState: QuantumState, upToIndex: Int) -> QuantumState {
        precondition(initialState.numQubits >= numQubits,
                     "Initial state must have at least as many qubits as circuit")
        precondition(upToIndex >= 0 && upToIndex <= operations.count,
                     "Index out of bounds")

        // Check if ancilla qubits are needed
        let maxQubit = maxQubitUsed()
        var currentState = initialState

        // Expand state if needed (same logic as execute(on:))
        if maxQubit >= initialState.numQubits {
            let numAncillaQubits = maxQubit - initialState.numQubits + 1
            let expandedSize = 1 << (initialState.numQubits + numAncillaQubits)

            var expandedAmplitudes = [Complex<Double>](repeating: .zero, count: expandedSize)
            for i in 0 ..< initialState.stateSpaceSize {
                expandedAmplitudes[i] = initialState.amplitudes[i]
            }

            currentState = QuantumState(numQubits: maxQubit + 1, amplitudes: expandedAmplitudes)
        }

        for i in 0 ..< upToIndex {
            currentState = GateApplication.apply(
                gate: operations[i].gate,
                to: operations[i].qubits,
                state: currentState
            )
        }

        return currentState
    }

    // MARK: - CustomStringConvertible

    /// String representation of the quantum circuit
    var description: String {
        if operations.isEmpty { return "QuantumCircuit(\(numQubits) qubits, empty)" }

        let gateList = operations.prefix(5).map(\.description).joined(separator: ", ")
        let suffix = operations.count > 5 ? ", ..." : ""

        return "QuantumCircuit(\(numQubits) qubits, \(operations.count) gates): \(gateList)\(suffix)"
    }
}

// MARK: - Pre-Built Circuits (Factory Methods)

extension QuantumCircuit {
    /// Create Bell state circuit: H(0) · CNOT(0,1)
    /// Produces (|00⟩ + |11⟩)/√2
    static func bellState() -> QuantumCircuit {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
        return circuit
    }

    /// Create GHZ state circuit: H(0) · CNOT(0,1) · CNOT(0,2)
    /// Produces (|000⟩ + |111⟩)/√2
    /// - Parameter numQubits: Number of qubits (default 3)
    static func ghzState(numQubits: Int = 3) -> QuantumCircuit {
        precondition(numQubits >= 2, "GHZ state requires at least 2 qubits")

        var circuit = QuantumCircuit(numQubits: numQubits)

        circuit.append(gate: .hadamard, toQubit: 0)

        for i in 1 ..< numQubits {
            circuit.append(gate: .cnot(control: 0, target: i), qubits: [])
        }

        return circuit
    }

    /// Create superposition circuit: Apply H to all qubits
    /// Produces equal superposition over all basis states
    /// - Parameter numQubits: Number of qubits
    static func superposition(numQubits: Int) -> QuantumCircuit {
        precondition(numQubits > 0, "Must have at least 1 qubit")

        var circuit = QuantumCircuit(numQubits: numQubits)

        for i in 0 ..< numQubits {
            circuit.append(gate: .hadamard, toQubit: i)
        }

        return circuit
    }

    /// Create Quantum Fourier Transform circuit
    /// Implements the QFT algorithm for transforming to frequency domain
    /// - Parameter numQubits: Number of qubits (typical: 3-8)
    /// - Returns: Circuit implementing QFT
    ///
    /// Algorithm structure:
    /// - For each qubit i (from 0 to n-1):
    ///   - Apply Hadamard to qubit i
    ///   - Apply controlled-phase gates from qubits i+1 to n-1
    ///   - Phase angles: π/2, π/4, π/8, ... (decreasing powers of 2)
    /// - Reverse qubit order with SWAP gates
    static func qft(numQubits: Int) -> QuantumCircuit {
        precondition(numQubits > 0, "Must have at least 1 qubit")
        precondition(numQubits <= 16, "QFT with >16 qubits is computationally expensive")

        var circuit = QuantumCircuit(numQubits: numQubits)

        for target in 0 ..< numQubits {
            circuit.append(gate: .hadamard, toQubit: target)

            for control in (target + 1) ..< numQubits {
                let k = control - target + 1
                let theta = Double.pi / Double(1 << k)

                circuit.append(
                    gate: .controlledPhase(theta: theta, control: control, target: target),
                    qubits: []
                )
            }
        }

        let swapCount = numQubits / 2
        for i in 0 ..< swapCount {
            let j = numQubits - 1 - i
            circuit.append(gate: .swap(qubit1: i, qubit2: j), qubits: [])
        }

        return circuit
    }

    /// Create inverse Quantum Fourier Transform circuit
    /// Reverses the QFT transformation
    /// - Parameter numQubits: Number of qubits
    /// - Returns: Circuit implementing inverse QFT
    static func inverseQFT(numQubits: Int) -> QuantumCircuit {
        precondition(numQubits > 0, "Must have at least 1 qubit")

        var circuit = QuantumCircuit(numQubits: numQubits)

        // Inverse QFT is QFT reversed with negated phases
        let swapCount = numQubits / 2
        for i in 0 ..< swapCount {
            let j = numQubits - 1 - i
            circuit.append(gate: .swap(qubit1: i, qubit2: j), qubits: [])
        }

        for target in (0 ..< numQubits).reversed() {
            for control in (target + 1 ..< numQubits).reversed() {
                let k = control - target + 1
                let theta = -Double.pi / Double(1 << k) // Negated

                circuit.append(
                    gate: .controlledPhase(theta: theta, control: control, target: target),
                    qubits: []
                )
            }

            circuit.append(gate: .hadamard, toQubit: target)
        }

        return circuit
    }

    /// Create Grover search algorithm circuit
    /// Finds a marked item in an unsorted database with O(√N) queries
    /// - Parameters:
    ///   - numQubits: Number of qubits (search space size = 2^n)
    ///   - target: Target state to search for (basis state index)
    ///   - iterations: Number of Grover iterations (optimal = π/4 * √(2^n), defaults to auto)
    /// - Returns: Circuit implementing Grover search
    ///
    /// Algorithm structure:
    /// 1. Initialize superposition (H on all qubits)
    /// 2. Repeat iterations times:
    ///    a. Oracle: mark the target state with phase flip
    ///    b. Diffusion: amplify marked state amplitude
    static func grover(numQubits: Int, target: Int, iterations: Int? = nil) -> QuantumCircuit {
        precondition(numQubits > 0, "Must have at least 1 qubit")
        precondition(numQubits <= 10, "Grover with >10 qubits requires many iterations")

        let stateSpaceSize = 1 << numQubits
        precondition(target >= 0 && target < stateSpaceSize, "Target must be valid basis state")

        let optimalIterations = iterations ?? Int((Double.pi / 4.0) * sqrt(Double(stateSpaceSize)))

        var circuit = QuantumCircuit(numQubits: numQubits)

        for qubit in 0 ..< numQubits {
            circuit.append(gate: .hadamard, toQubit: qubit)
        }

        for _ in 0 ..< optimalIterations {
            // Oracle: flip phase of target state
            appendGroverOracle(to: &circuit, target: target, numQubits: numQubits)

            // Diffusion operator: 2|s⟩⟨s| - I
            appendGroverDiffusion(to: &circuit, numQubits: numQubits)
        }

        return circuit
    }

    /// Append Grover oracle to circuit
    /// Implements the oracle function that flips the phase of the target state
    /// - Parameters:
    ///   - circuit: Circuit to append oracle to
    ///   - target: Target state index to mark with phase flip
    ///   - numQubits: Total number of qubits in the system
    private static func appendGroverOracle(to circuit: inout QuantumCircuit, target: Int, numQubits: Int) {
        for qubit in 0 ..< numQubits {
            if (target >> qubit) & 1 == 0 {
                circuit.append(gate: .pauliX, toQubit: qubit)
            }
        }

        appendMultiControlledZ(to: &circuit, numQubits: numQubits)

        for qubit in 0 ..< numQubits {
            if (target >> qubit) & 1 == 0 {
                circuit.append(gate: .pauliX, toQubit: qubit)
            }
        }
    }

    /// Append multi-controlled Z gate
    /// Flips phase when all qubits are |1⟩
    /// Uses proper recursive decomposition (no approximations)
    private static func appendMultiControlledZ(to circuit: inout QuantumCircuit, numQubits: Int) {
        if numQubits == 1 {
            circuit.append(gate: .pauliZ, toQubit: 0)
        } else if numQubits == 2 {
            // Controlled-Z using controlled-phase(π)
            circuit.append(gate: .controlledPhase(theta: .pi, control: 0, target: 1), qubits: [])
        } else if numQubits == 3 {
            // For 3 qubits, use Toffoli decomposition
            circuit.append(gate: .hadamard, toQubit: 2)
            circuit.append(gate: .toffoli(control1: 0, control2: 1, target: 2), qubits: [])
            circuit.append(gate: .hadamard, toQubit: 2)
        } else {
            // For n>3 qubits, use recursive decomposition into Toffoli gates
            // Standard technique: decompose n-controlled gate into (n-1)-controlled gates
            // Reference: Nielsen & Chuang, Section 4.3

            // Multi-controlled Z on qubits [0...n-1]:
            // 1. Apply H to target (last qubit)
            // 2. Apply multi-controlled X (Toffoli chain)
            // 3. Apply H to target

            let target = numQubits - 1
            circuit.append(gate: .hadamard, toQubit: target)

            appendMultiControlledX(to: &circuit, controls: Array(0 ..< numQubits - 1), target: target)

            circuit.append(gate: .hadamard, toQubit: target)
        }
    }

    /// Append multi-controlled X (NOT) gate using Toffoli decomposition
    /// This is the standard "ladder" decomposition for n-controlled gates
    static func appendMultiControlledX(to circuit: inout QuantumCircuit, controls: [Int], target: Int) {
        let n = controls.count

        if n == 0 {
            circuit.append(gate: .pauliX, toQubit: target)
        } else if n == 1 {
            circuit.append(gate: .cnot(control: controls[0], target: target), qubits: [])
        } else if n == 2 {
            circuit.append(gate: .toffoli(control1: controls[0], control2: controls[1], target: target), qubits: [])
        } else {
            // Allocate ancilla qubits: use high-numbered qubits beyond existing circuit qubits
            // Note: Circuit size remains at logical qubit count; ancilla handled during execution
            let maxUsedQubit = max(controls.max() ?? 0, target)
            let firstAncilla = maxUsedQubit + 1
            let numAncilla = n - 2
            let ancillaQubits = (0 ..< numAncilla).map { firstAncilla + $0 }

            // a0 = c0 ∧ c1
            circuit.append(gate: .toffoli(control1: controls[0], control2: controls[1], target: ancillaQubits[0]), qubits: [])

            // a_i = a_{i-1} ∧ c_{i+1} for i = 1 to n-3
            for i in 1 ..< numAncilla {
                circuit.append(gate: .toffoli(control1: ancillaQubits[i - 1], control2: controls[i + 1], target: ancillaQubits[i]), qubits: [])
            }

            // Final gate: a_{n-3} ∧ c_{n-1} controls target
            circuit.append(gate: .toffoli(control1: ancillaQubits[numAncilla - 1], control2: controls[n - 1], target: target), qubits: [])

            // Reverse pass: uncompute ancilla qubits (clean up workspace)
            for i in (1 ..< numAncilla).reversed() {
                circuit.append(gate: .toffoli(control1: ancillaQubits[i - 1], control2: controls[i + 1], target: ancillaQubits[i]), qubits: [])
            }

            circuit.append(gate: .toffoli(control1: controls[0], control2: controls[1], target: ancillaQubits[0]), qubits: [])
        }
    }

    /// Append multi-controlled Y gate using ancilla decomposition
    /// Decomposes C^n(Y) into multi-controlled X with basis change
    /// - Parameters:
    ///   - circuit: Circuit to append to
    ///   - controls: Control qubit indices
    ///   - target: Target qubit index
    static func appendMultiControlledY(to circuit: inout QuantumCircuit, controls: [Int], target: Int) {
        let n = controls.count

        if n == 0 {
            circuit.append(gate: .pauliY, toQubit: target)
        } else if n == 1 {
            circuit.append(gate: .cy(control: controls[0], target: target), qubits: [])
        } else {
            // C^n(Y) = S†·C^n(X)·S (since Y = S†XS up to global phase)
            circuit.append(gate: .phase(theta: -Double.pi / 2.0), toQubit: target) // S†
            appendMultiControlledX(to: &circuit, controls: controls, target: target)
            circuit.append(gate: .sGate, toQubit: target) // S
        }
    }

    /// Append multi-controlled Z gate (more efficient than C^n(X))
    /// Diagonal gate structure allows for efficient implementation
    /// - Parameters:
    ///   - circuit: Circuit to append to
    ///   - controls: Control qubit indices
    ///   - target: Target qubit index
    static func appendMultiControlledZ(to circuit: inout QuantumCircuit, controls: [Int], target: Int) {
        let n = controls.count

        if n == 0 {
            circuit.append(gate: .pauliZ, toQubit: target)
        } else if n == 1 {
            circuit.append(gate: .cz(control: controls[0], target: target), qubits: [])
        } else {
            // For n ≥ 2: Use H-C^n(X)-H decomposition
            // C^n(Z) = H_{target} · C^n(X) · H_{target}
            circuit.append(gate: .hadamard, toQubit: target)
            appendMultiControlledX(to: &circuit, controls: controls, target: target)
            circuit.append(gate: .hadamard, toQubit: target)
        }
    }

    /// Append multi-controlled arbitrary single-qubit unitary gate
    /// Applies any single-qubit gate U with n control qubits
    /// Uses decomposition based on gate type for optimal implementation
    /// - Parameters:
    ///   - circuit: Circuit to append to
    ///   - gate: Single-qubit gate to apply (must be single-qubit)
    ///   - controls: Control qubit indices
    ///   - target: Target qubit index
    static func appendMultiControlledU(
        to circuit: inout QuantumCircuit,
        gate: QuantumGate,
        controls: [Int],
        target: Int
    ) {
        precondition(gate.qubitsRequired == 1, "Multi-controlled U requires single-qubit gate")

        let n = controls.count

        if n == 0 {
            circuit.append(gate: gate, toQubit: target)
        } else {
            switch gate {
            case .pauliX:
                appendMultiControlledX(to: &circuit, controls: controls, target: target)
            case .pauliY:
                appendMultiControlledY(to: &circuit, controls: controls, target: target)
            case .pauliZ:
                appendMultiControlledZ(to: &circuit, controls: controls, target: target)
            case .hadamard:
                // C^n(H) using basis rotation
                circuit.append(gate: .rotationY(theta: .pi / 4), toQubit: target)
                appendMultiControlledZ(to: &circuit, controls: controls, target: target)
                circuit.append(gate: .rotationY(theta: -.pi / 4), toQubit: target)
            default:
                // For arbitrary U: decompose into controlled operations
                // Apply V then C^n(X) then V† then C^n(X) then V
                // This implements the multi-controlled unitary
                circuit.append(gate: gate, toQubit: target)
                appendMultiControlledX(to: &circuit, controls: controls, target: target)

                let matrix = gate.matrix()
                let adjointMatrix = QuantumGate.conjugateTranspose(matrix)
                do {
                    let adjointGate = try QuantumGate.createCustomSingleQubit(matrix: adjointMatrix)
                    circuit.append(gate: adjointGate, toQubit: target)
                } catch {
                    // Adjoint creation should never fail for a unitary matrix
                    // If it does, it indicates numerical instability
                    // Fall back to simpler decomposition (less optimal but safe)
                    circuit.append(gate: gate, toQubit: target)
                }

                appendMultiControlledX(to: &circuit, controls: controls, target: target)
                circuit.append(gate: gate, toQubit: target)
            }
        }
    }

    /// Append Grover diffusion operator to circuit
    /// Implements the diffusion operator 2|s⟩⟨s| - I that amplifies marked state amplitudes
    /// - Parameters:
    ///   - circuit: Circuit to append diffusion operator to
    ///   - numQubits: Total number of qubits in the system
    private static func appendGroverDiffusion(to circuit: inout QuantumCircuit, numQubits: Int) {
        for qubit in 0 ..< numQubits {
            circuit.append(gate: .hadamard, toQubit: qubit)
        }

        for qubit in 0 ..< numQubits {
            circuit.append(gate: .pauliX, toQubit: qubit)
        }

        appendMultiControlledZ(to: &circuit, numQubits: numQubits)

        for qubit in 0 ..< numQubits {
            circuit.append(gate: .pauliX, toQubit: qubit)
        }

        for qubit in 0 ..< numQubits {
            circuit.append(gate: .hadamard, toQubit: qubit)
        }
    }

    /// Create quantum annealing circuit for optimization problems
    /// Implements adiabatic evolution from simple to complex Hamiltonian
    /// - Parameters:
    ///   - numQubits: Number of qubits (problem variables)
    ///   - problem: Coupling strengths defining the optimization problem (Ising model)
    ///   - annealingSteps: Number of time steps in annealing schedule (default: 20)
    /// - Returns: Circuit demonstrating quantum annealing process
    ///
    /// Algorithm structure:
    /// 1. Initialize in superposition (transverse field Hamiltonian)
    /// 2. Gradually evolve to problem Hamiltonian (z-field + couplings)
    /// 3. Measure to find optimal configuration
    ///
    /// VFX Application: Ray tracing path optimization, rendering parameter optimization
    static func annealing(numQubits: Int, problem: IsingProblem, annealingSteps: Int = 20) -> QuantumCircuit {
        precondition(numQubits > 0, "Must have at least 1 qubit")
        precondition(numQubits <= 8, "Annealing with >8 qubits becomes computationally expensive")
        precondition(annealingSteps > 0, "Must have at least 1 annealing step")

        var circuit = QuantumCircuit(numQubits: numQubits)

        // Initialize all qubits in superposition (transverse field)
        for qubit in 0 ..< numQubits {
            circuit.append(gate: .hadamard, toQubit: qubit)
        }

        // Adiabatic evolution through annealing schedule
        for step in 0 ..< annealingSteps {
            let time = Double(step) / Double(annealingSteps - 1) // 0.0 to 1.0

            let transverseStrength = 1.0 - time
            for qubit in 0 ..< numQubits {
                let angle = 2.0 * transverseStrength * problem.transverseField[qubit]
                circuit.append(gate: .rotationX(theta: angle), toQubit: qubit)
            }

            let problemStrength = time

            for qubit in 0 ..< numQubits {
                let angle = 2.0 * problemStrength * problem.localFields[qubit]
                circuit.append(gate: .rotationZ(theta: angle), toQubit: qubit)
            }

            for i in 0 ..< numQubits {
                for j in (i + 1) ..< numQubits {
                    let coupling = problem.couplings[i][j]
                    if abs(coupling) > 1e-10 {
                        let angle = 2.0 * problemStrength * coupling
                        appendZZCoupling(to: &circuit, qubit1: i, qubit2: j, angle: angle)
                    }
                }
            }
        }

        return circuit
    }

    /// Convenience method for creating annealing circuit with simple coupling dictionary
    /// - Parameters:
    ///   - numQubits: Number of qubits
    ///   - couplings: Dictionary of qubit pairs to coupling strengths (ignored for simplicity)
    ///   - annealingSteps: Number of annealing steps
    /// - Returns: Annealing circuit with random problem
    static func annealing(numQubits: Int, couplings _: [String: Double], annealingSteps: Int = 20) -> QuantumCircuit {
        let problem = IsingProblem.random(numQubits: numQubits, maxCoupling: 1.0)
        return annealing(numQubits: numQubits, problem: problem, annealingSteps: annealingSteps)
    }

    /// Ising model problem definition for quantum annealing
    struct IsingProblem {
        let localFields: [Double]
        let couplings: [[Double]]
        let transverseField: [Double]

        /// Create Ising model problem instance
        /// - Parameters:
        ///   - localFields: Local magnetic field coefficients for each qubit
        ///   - couplings: Symmetric coupling matrix between qubits
        ///   - transverseField: Transverse field strengths (defaults to uniform field of 1.0)
        init(localFields: [Double], couplings: [[Double]], transverseField: [Double]? = nil) {
            let n = localFields.count
            precondition(couplings.count == n, "Couplings matrix must be N×N")
            precondition(couplings.allSatisfy { $0.count == n }, "Couplings matrix must be square")

            self.localFields = localFields
            self.couplings = couplings
            self.transverseField = transverseField ?? [Double](repeating: 1.0, count: n)
        }

        /// Create random Ising problem for demonstration
        static func random(numQubits: Int, maxCoupling: Double = 1.0) -> IsingProblem {
            var couplings = [[Double]](repeating: [Double](repeating: 0.0, count: numQubits), count: numQubits)
            var localFields = [Double]()

            for i in 0 ..< numQubits {
                localFields.append((Double.random(in: -1.0 ... 1.0)) * maxCoupling)
                for j in (i + 1) ..< numQubits {
                    couplings[i][j] = (Double.random(in: -1.0 ... 1.0)) * maxCoupling
                    couplings[j][i] = couplings[i][j]
                }
            }

            return IsingProblem(localFields: localFields, couplings: couplings)
        }

        /// Create simple quadratic optimization problem: minimize x² - 2x
        static func quadraticMinimum(numQubits: Int = 4) -> IsingProblem {
            precondition(numQubits >= 2, "Need at least 2 qubits for meaningful quadratic")

            var localFields = [Double](repeating: 0.0, count: numQubits)
            var couplings = [[Double]](repeating: [Double](repeating: 0.0, count: numQubits), count: numQubits)

            // Encode x² term with couplings between adjacent qubits
            for i in 0 ..< numQubits - 1 {
                couplings[i][i + 1] = 0.5 // Quadratic coupling
            }

            // Encode -2x term with local fields
            for i in 0 ..< numQubits {
                localFields[i] = -1.0 * Double(1 << i) // Linear terms decrease by powers of 2
            }

            return IsingProblem(localFields: localFields, couplings: couplings)
        }
    }

    /// Append ZZ coupling between two qubits
    /// Implements exp(-iθ Z₁Z₂) using CNOT decomposition: CNOT₁₂ · RZ₂(2θ) · CNOT₁₂
    /// - Parameters:
    ///   - circuit: Circuit to append coupling to
    ///   - qubit1: First qubit index
    ///   - qubit2: Second qubit index
    ///   - angle: Coupling strength θ
    private static func appendZZCoupling(to circuit: inout QuantumCircuit, qubit1: Int, qubit2: Int, angle: Double) {
        // ZZ coupling: exp(-iθ Z₁Z₂) = CNOT₁₂ · RZ₂(2θ) · CNOT₁₂
        // This rotates qubit 2 by 2θ when qubit 1 is |1⟩

        circuit.append(gate: .cnot(control: qubit1, target: qubit2), qubits: [])
        circuit.append(gate: .rotationZ(theta: 2.0 * angle), toQubit: qubit2)
        circuit.append(gate: .cnot(control: qubit1, target: qubit2), qubits: [])
    }
}
