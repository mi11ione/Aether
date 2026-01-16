// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Asynchronous quantum circuit executor with automatic GPU acceleration.
///
/// Actor-isolated simulator for non-blocking circuit execution with progress tracking. Uses
/// ``GateApplication`` (CPU) for circuits under 10 qubits and ``MetalGateApplication`` (GPU)
/// for larger circuits when Metal is available. Progress callbacks batch every 5 gates.
///
/// For synchronous execution without progress tracking, use ``QuantumCircuit/execute()`` directly.
///
/// **Example:**
/// ```swift
/// let simulator = QuantumSimulator()
/// var circuit = QuantumCircuit(qubits: 10)
/// circuit.append(.hadamard, to: 0)
/// circuit.append(.cnot, to: [0, 1])
///
/// // Basic execution
/// let state = await simulator.execute(circuit)
///
/// // With progress tracking
/// let result = await simulator.execute(circuit, progressHandler: { progress in
///     await MainActor.run {
///         progressBar.progress = progress
///     }
/// })
/// ```
///
/// - SeeAlso: ``QuantumCircuit/execute()``, ``MetalGateApplication``, ``GateApplication``
public actor QuantumSimulator {
    /// Execution progress information
    public struct Progress: Sendable {
        /// Number of gates executed so far
        public let executed: Int

        /// Total number of gates in circuit
        public let total: Int

        /// Execution progress as percentage in [0.0, 1.0]
        public var percentage: Double {
            total > 0 ? Double(executed) / Double(total) : 0.0
        }
    }

    /// Whether Metal GPU acceleration is enabled
    private let useMetalAcceleration: Bool

    /// Metal GPU backend (initialized if acceleration enabled)
    private var metalApplication: MetalGateApplication?

    /// Creates a quantum simulator with optional GPU acceleration
    ///
    /// Initializes actor-isolated simulator. If Metal acceleration is enabled,
    /// GPU will automatically be used for circuits with ≥10 qubits. Falls back
    /// to CPU for smaller circuits or if Metal is unavailable.
    ///
    /// **Example:**
    /// ```swift
    /// // With GPU acceleration (default)
    /// let simulator = await QuantumSimulator()
    ///
    /// // CPU only (disable GPU)
    /// let cpuSimulator = await QuantumSimulator(useMetalAcceleration: false)
    /// ```
    ///
    /// - Parameter useMetalAcceleration: Enable Metal GPU for circuits with ≥10 qubits (default: true)
    public init(useMetalAcceleration: Bool = true) {
        self.useMetalAcceleration = useMetalAcceleration
        if useMetalAcceleration {
            metalApplication = MetalGateApplication()
        }
    }

    // MARK: - Circuit Execution

    /// Executes quantum circuit asynchronously with optional progress tracking.
    ///
    /// Applies gate operations sequentially, using GPU for circuits with ≥10 qubits (after ancilla
    /// expansion) when Metal is enabled. Backend selection happens once at start; no mid-circuit
    /// switching. Progress callback fires every 5 gates or at completion with values in [0.0, 1.0].
    ///
    /// **Example:**
    /// ```swift
    /// let simulator = await QuantumSimulator()
    /// var circuit = QuantumCircuit(qubits: 10)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    ///
    /// // Basic execution
    /// let state = await simulator.execute(circuit)
    ///
    /// // With custom initial state
    /// let initial = QuantumState(qubits: 10)
    /// let result = await simulator.execute(circuit, from: initial)
    ///
    /// // With progress tracking
    /// let final = await simulator.execute(circuit, progressHandler: { progress in
    ///     print("Progress: \(Int(progress * 100))%")
    /// })
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit to execute
    ///   - initialState: Starting state (defaults to ground state |00...0⟩)
    ///   - progressHandler: Optional async callback receiving progress in [0.0, 1.0]
    /// - Returns: Final quantum state after applying all gates
    /// - Complexity: O(n x 2^q) where n = gate count, q = qubits (including ancilla)
    ///
    @_optimize(speed)
    @_eagerMove
    public func execute(
        _ circuit: QuantumCircuit,
        from initialState: QuantumState? = nil,
        progressHandler: (@isolated(any) @Sendable (Double) async -> Void)? = nil,
    ) async -> QuantumState {
        let operationCount: Int = circuit.count

        let startState: QuantumState = if let initialState {
            initialState
        } else {
            QuantumState(qubits: circuit.qubits)
        }

        let maxQubit: Int = circuit.highestQubitIndex
        var state = QuantumCircuit.expandStateForAncilla(startState, maxQubit: maxQubit)

        let useGPU: Bool = useMetalAcceleration && metalApplication != nil && state.qubits >= MetalGateApplication.minimumQubitCountForGPU
        let progressMultiplier: Double = operationCount > 0 ? 1.0 / Double(operationCount) : 0.0
        let lastIndex: Int = operationCount - 1
        let operations = circuit.gates

        let progressUpdateBatchSize = 5
        var progressCounter = 0

        for index in 0 ..< operationCount {
            let operation = operations[index]
            if useGPU {
                // Safety: useGPU is true only when metalApplication != nil (line 130)
                state = await metalApplication!.apply(operation.gate, to: operation.qubits, state: state)
            } else {
                state = GateApplication.apply(operation.gate, to: operation.qubits, state: state)
            }

            if let progressHandler {
                progressCounter += 1
                if progressCounter >= progressUpdateBatchSize || index == lastIndex {
                    progressCounter = 0
                    let executedGates = index + 1
                    await progressHandler(Double(executedGates) * progressMultiplier)
                }
            }
        }

        return state
    }
}
