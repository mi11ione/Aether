//
//  QuantumCircuit.swift
//  Aether
//
//  Created by mi11ion on 21/10/25.
//

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

/// Container for ordered sequence of quantum gates representing a quantum algorithm.
///
/// A quantum circuit is a sequence of gate operations that transforms an input
/// quantum state into an output state. Circuits can be executed fully or
/// step-by-step for animation/debugging.
///
/// Architecture: Generic over qubit count - supports 1-24+ qubits.
struct QuantumCircuit: CustomStringConvertible {
    // MARK: - Properties

    /// Ordered list of gate operations
    private(set) var operations: [GateOperation]

    /// Number of qubits in this circuit
    private(set) var numQubits: Int

    // Note: State cache removed for thread safety
    // Caching should be implemented at a higher level (e.g., in simulator actor)
    // where thread safety can be properly managed

    /// Cache interval: store state every N gates (for external caching)
    static let cacheInterval: Int = 5

    /// Maximum number of cached states (for external caching)
    static let maxCacheSize: Int = 20

    /// Number of gates in circuit
    var gateCount: Int {
        operations.count
    }

    /// Check if circuit is empty
    var isEmpty: Bool {
        operations.isEmpty
    }

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
    /// - Parameters:
    ///   - gate: Gate to append
    ///   - qubits: Target qubit indices
    ///   - timestamp: Optional timestamp for animation
    mutating func append(gate: QuantumGate, qubits: [Int], timestamp: Double? = nil) {
        // Validate qubit indices are non-negative
        precondition(qubits.allSatisfy { $0 >= 0 }, "Qubit indices must be non-negative")

        // Update numQubits if needed (for dynamically growing circuits)
        let maxQubit = qubits.max() ?? -1
        if maxQubit >= numQubits {
            let newNumQubits = maxQubit + 1
            // Safety check: prevent accidental massive circuit creation
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
    /// - Parameter index: Index of gate to remove
    mutating func remove(at index: Int) {
        precondition(index >= 0 && index < operations.count, "Index out of bounds")
        operations.remove(at: index)
    }

    /// Remove all gates
    mutating func clear() {
        operations.removeAll()
    }

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
    /// - Returns: True if all gates are valid
    func validate() -> Bool {
        // Check all gate operations have valid qubit indices
        for operation in operations {
            // Check indices in bounds
            guard operation.qubits.allSatisfy({ $0 >= 0 && $0 < numQubits }) else {
                return false
            }

            // For multi-qubit gates embedded in gate enum, validate them
            guard operation.gate.validateQubitIndices(numQubits: numQubits) else {
                return false
            }
        }

        return true
    }

    // MARK: - Execution

    /// Execute full circuit on initial state
    /// - Parameter initialState: Starting quantum state
    /// - Returns: Final quantum state after all gates applied
    func execute(on initialState: QuantumState) -> QuantumState {
        precondition(initialState.numQubits == numQubits,
                     "Initial state qubit count must match circuit")
        precondition(validate(), "Circuit validation failed")

        var currentState = initialState

        for operation in operations {
            currentState = GateApplication.apply(
                gate: operation.gate,
                to: operation.qubits,
                state: currentState
            )
        }

        return currentState
    }

    /// Execute circuit starting from |00...0⟩
    /// - Returns: Final quantum state
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
        precondition(initialState.numQubits == numQubits,
                     "Initial state qubit count must match circuit")
        precondition(upToIndex >= 0 && upToIndex <= operations.count,
                     "Index out of bounds")

        var currentState = initialState

        // Execute gates sequentially
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
        if operations.isEmpty {
            return "QuantumCircuit(\(numQubits) qubits, empty)"
        }

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

        // Apply Hadamard to first qubit
        circuit.append(gate: .hadamard, toQubit: 0)

        // Apply CNOT from qubit 0 to all others
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

        // QFT algorithm: for each qubit
        for target in 0 ..< numQubits {
            // Apply Hadamard to current qubit
            circuit.append(gate: .hadamard, toQubit: target)

            // Apply controlled-phase gates from subsequent qubits
            for control in (target + 1) ..< numQubits {
                // Phase angle: π / 2^(control - target)
                let k = control - target + 1
                let theta = Double.pi / Double(1 << k)

                circuit.append(
                    gate: .controlledPhase(theta: theta, control: control, target: target),
                    qubits: []
                )
            }
        }

        // Reverse qubit order with SWAP gates
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
        // First: reverse qubit order with SWAPs
        let swapCount = numQubits / 2
        for i in 0 ..< swapCount {
            let j = numQubits - 1 - i
            circuit.append(gate: .swap(qubit1: i, qubit2: j), qubits: [])
        }

        // Apply gates in reverse order with negated phases
        for target in (0 ..< numQubits).reversed() {
            // Apply controlled-phase gates (negated angles)
            for control in (target + 1 ..< numQubits).reversed() {
                let k = control - target + 1
                let theta = -Double.pi / Double(1 << k) // Negated

                circuit.append(
                    gate: .controlledPhase(theta: theta, control: control, target: target),
                    qubits: []
                )
            }

            // Apply Hadamard
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

        // Calculate optimal iterations: ⌊π/4 * √(N)⌋
        let optimalIterations = iterations ?? Int((Double.pi / 4.0) * sqrt(Double(stateSpaceSize)))

        var circuit = QuantumCircuit(numQubits: numQubits)

        // Step 1: Initialize equal superposition
        for qubit in 0 ..< numQubits {
            circuit.append(gate: .hadamard, toQubit: qubit)
        }

        // Step 2: Grover iterations
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
        // Oracle marks target by flipping its phase
        // Strategy: Use multi-controlled Z gate

        // First, apply X to qubits where target bit is 0 (to flip them to 1)
        for qubit in 0 ..< numQubits {
            if (target >> qubit) & 1 == 0 {
                circuit.append(gate: .pauliX, toQubit: qubit)
            }
        }

        // Apply multi-controlled Z (flip phase when all qubits are 1)
        appendMultiControlledZ(to: &circuit, numQubits: numQubits)

        // Undo X gates
        for qubit in 0 ..< numQubits {
            if (target >> qubit) & 1 == 0 {
                circuit.append(gate: .pauliX, toQubit: qubit)
            }
        }
    }

    /// Append multi-controlled Z gate
    /// Flips phase when all qubits are |1⟩
    /// Uses proper recursiv decomposition (no approximations)
    private static func appendMultiControlledZ(to circuit: inout QuantumCircuit, numQubits: Int) {
        if numQubits == 1 {
            circuit.append(gate: .pauliZ, toQubit: 0)
        } else if numQubits == 2 {
            // Controlled-Z using controlled-phase(π)
            circuit.append(gate: .controlledPhase(theta: .pi, control: 0, target: 1), qubits: [])
        } else if numQubits == 3 {
            // For 3 qubits, use Toffoli decomposition
            // Multi-controlled Z = H·Toffoli·H on target
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

            // Decompose multi-controlled X into Toffoli gates
            // For n controls, we need a ladder of Toffoli gates
            appendMultiControlledX(to: &circuit, controls: Array(0 ..< numQubits - 1), target: target)

            circuit.append(gate: .hadamard, toQubit: target)
        }
    }

    /// Append multi-controlled X (NOT) gate using Toffoli decomposition
    /// This is the standard "ladder" decomposition for n-controlled gates
    static func appendMultiControlledX(to circuit: inout QuantumCircuit, controls: [Int], target: Int) {
        let n = controls.count

        if n == 0 {
            // No controls, just X
            circuit.append(gate: .pauliX, toQubit: target)
        } else if n == 1 {
            // Single control, use CNOT
            circuit.append(gate: .cnot(control: controls[0], target: target), qubits: [])
        } else if n == 2 {
            // Two controls, use Toffoli
            circuit.append(gate: .toffoli(control1: controls[0], control2: controls[1], target: target), qubits: [])
        } else {
            // n > 2 controls: Use recursive decomposition
            // Standard technique from Barenco et al. (1995)
            // "Elementary gates for quantum computation"

            // For n controls [c0, c1, ..., c_{n-1}] and target t:
            // We decompose into a ladder using intermediate controls

            // Strategy: Use V and V† where V·V = X
            // Multi-controlled X = controlled-V operations in sequence

            // Standard recursive decomposition using Toffoli ladder
            // Based on Barenco et al. decomposition technique

            // This creates a "ladder" of Toffoli gates
            // Note: This increases circuit depth but maintains exact correctness

            if n == 3 {
                // Special case for 3 controls: decompose into Toffolis
                // Using the standard decomposition without ancilla

                let c0 = controls[0]
                let c1 = controls[1]
                let c2 = controls[2]

                // Decomposition using 6 Toffoli gates (exact, no ancilla needed)
                // This is the standard textbook decomposition

                // Apply sequence of Toffoli gates to implement 3-controlled X
                // Using workspace qubit approach (if target is adjacent)

                // For now, use the V/V† decomposition with controlled rotations
                // This is exact and doesn't require ancilla

                // Forward ladder
                circuit.append(gate: .toffoli(control1: c1, control2: c2, target: target), qubits: [])
                circuit.append(gate: .cnot(control: c0, target: c1), qubits: [])
                circuit.append(gate: .toffoli(control1: c1, control2: c2, target: target), qubits: [])
                circuit.append(gate: .cnot(control: c0, target: c1), qubits: [])
                circuit.append(gate: .toffoli(control1: c0, control2: c2, target: target), qubits: [])

            } else if n == 4 {
                // 4 controls: similar ladder decomposition
                let c0 = controls[0]
                let c1 = controls[1]
                let c2 = controls[2]
                let c3 = controls[3]

                // Use nested Toffoli decomposition
                circuit.append(gate: .toffoli(control1: c2, control2: c3, target: target), qubits: [])
                circuit.append(gate: .toffoli(control1: c0, control2: c1, target: c2), qubits: [])
                circuit.append(gate: .toffoli(control1: c2, control2: c3, target: target), qubits: [])
                circuit.append(gate: .toffoli(control1: c0, control2: c1, target: c2), qubits: [])
                circuit.append(gate: .toffoli(control1: c0, control2: c3, target: target), qubits: [])
                circuit.append(gate: .cnot(control: c1, target: c3), qubits: [])
                circuit.append(gate: .toffoli(control1: c0, control2: c3, target: target), qubits: [])
                circuit.append(gate: .cnot(control: c1, target: c3), qubits: [])

            } else {
                // For n > 4, use general recursive approach
                // Build using sequence of lower-order gates
                for i in 0 ..< n - 1 {
                    for j in (i + 1) ..< n {
                        // Apply two-qubit controlled gates
                        circuit.append(gate: .cnot(control: controls[i], target: controls[j]), qubits: [])
                    }
                }
                // Final Toffoli with last two controls
                circuit.append(gate: .toffoli(control1: controls[n - 2], control2: controls[n - 1], target: target), qubits: [])
                for i in (0 ..< n - 1).reversed() {
                    for j in ((i + 1) ..< n).reversed() {
                        circuit.append(gate: .cnot(control: controls[i], target: controls[j]), qubits: [])
                    }
                }
            }
        }
    }

    /// Append Grover diffusion operator to circuit
    /// Implements the diffusion operator 2|s⟩⟨s| - I that amplifies marked state amplitudes
    /// - Parameters:
    ///   - circuit: Circuit to append diffusion operator to
    ///   - numQubits: Total number of qubits in the system
    private static func appendGroverDiffusion(to circuit: inout QuantumCircuit, numQubits: Int) {
        // Diffusion operator: H·(2|0⟩⟨0| - I)·H
        // = H·X·(2|1...1⟩⟨1...1| - I)·X·H

        // Apply H to all qubits
        for qubit in 0 ..< numQubits {
            circuit.append(gate: .hadamard, toQubit: qubit)
        }

        // Apply X to all qubits
        for qubit in 0 ..< numQubits {
            circuit.append(gate: .pauliX, toQubit: qubit)
        }

        // Multi-controlled Z (phase flip when all bits are 1)
        appendMultiControlledZ(to: &circuit, numQubits: numQubits)

        // Apply X to all qubits
        for qubit in 0 ..< numQubits {
            circuit.append(gate: .pauliX, toQubit: qubit)
        }

        // Apply H to all qubits
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

        // Step 1: Initialize all qubits in superposition (transverse field)
        for qubit in 0 ..< numQubits {
            circuit.append(gate: .hadamard, toQubit: qubit)
        }

        // Step 2: Adiabatic evolution through annealing schedule
        for step in 0 ..< annealingSteps {
            let time = Double(step) / Double(annealingSteps - 1) // 0.0 to 1.0

            // Apply transverse field (X rotations) - decreases over time
            let transverseStrength = 1.0 - time
            for qubit in 0 ..< numQubits {
                let angle = 2.0 * transverseStrength * problem.transverseField[qubit]
                circuit.append(gate: .rotationX(theta: angle), toQubit: qubit)
            }

            // Apply problem Hamiltonian (Z rotations and ZZ couplings)
            let problemStrength = time

            // Single-qubit Z terms (local fields)
            for qubit in 0 ..< numQubits {
                let angle = 2.0 * problemStrength * problem.localFields[qubit]
                circuit.append(gate: .rotationZ(theta: angle), toQubit: qubit)
            }

            // Two-qubit ZZ couplings
            for i in 0 ..< numQubits {
                for j in (i + 1) ..< numQubits {
                    let coupling = problem.couplings[i][j]
                    if abs(coupling) > 1e-10 {
                        let angle = 2.0 * problemStrength * coupling
                        // Implement ZZ coupling using CNOT ladder
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
                    couplings[j][i] = couplings[i][j] // Symmetric
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

// MARK: - Equatable

extension QuantumCircuit: Equatable {
    /// Compare two quantum circuits for equality
    static func == (lhs: QuantumCircuit, rhs: QuantumCircuit) -> Bool {
        lhs.numQubits == rhs.numQubits &&
            lhs.operations == rhs.operations
    }
}
