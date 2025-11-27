// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Asynchronous quantum circuit executor with automatic GPU acceleration
///
/// Actor-isolated simulator for non-blocking circuit execution with progress tracking.
/// Automatically uses Metal GPU for circuits with ≥10 qubits when available.
///
/// Use ``QuantumSimulator`` when you need async execution for UI responsiveness or progress
/// monitoring. For synchronous execution, call ``QuantumCircuit/execute()`` directly (faster,
/// simpler API).
///
/// **Performance characteristics:**
/// - CPU execution: Uses ``GateApplication`` for n<10 qubits
/// - GPU execution: Uses ``MetalGateApplication`` for n≥10 qubits (faster for large states)
/// - Progress callbacks: Batched every 5 gates to minimize overhead
///
/// **Example:**
/// ```swift
/// let simulator = QuantumSimulator()
/// var circuit = QuantumCircuit(numQubits: 10)
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

    /// Executes quantum circuit asynchronously with optional progress tracking
    ///
    /// Applies all gate operations sequentially to evolve the quantum state.
    /// Automatically uses GPU acceleration for circuits with ≥10 qubits when Metal is enabled.
    /// Progress callbacks are batched every 5 gates to minimize overhead.
    ///
    /// **GPU decision**: Checks `numQubits ≥ 10` after ancilla expansion. Once decided,
    /// uses same backend for entire execution (no switching mid-circuit).
    ///
    /// **Progress updates**: Callback invoked every 5 gates or at completion. Values range
    /// from 0.0 (start) to 1.0 (complete).
    ///
    /// **Example:**
    /// ```swift
    /// let simulator = await QuantumSimulator()
    /// var circuit = QuantumCircuit(numQubits: 10)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    ///
    /// // Basic execution
    /// let state = await simulator.execute(circuit)
    ///
    /// // With custom initial state
    /// let initial = QuantumState(numQubits: 10)
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
        progressHandler: (@isolated(any) @Sendable (Double) async -> Void)? = nil
    ) async -> QuantumState {
        let operationCount: Int = circuit.gateCount

        let startState: QuantumState = if let initialState {
            initialState
        } else {
            QuantumState(numQubits: circuit.numQubits)
        }

        let maxQubit: Int = circuit.highestQubitIndex
        var state = QuantumCircuit.expandStateForAncilla(startState, maxQubit: maxQubit)

        let useGPU: Bool = useMetalAcceleration && metalApplication != nil && state.numQubits >= MetalGateApplication.minimumQubitCountForGPU
        let progressMultiplier: Double = operationCount > 0 ? 1.0 / Double(operationCount) : 0.0
        let lastIndex: Int = operationCount - 1
        let operations = circuit.gates

        let progressUpdateBatchSize = 5
        var progressCounter = 0

        for index in 0 ..< operationCount {
            let operation = operations[index]
            if useGPU {
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
