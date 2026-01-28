// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Asynchronous quantum circuit executor with automatic GPU acceleration.
///
/// Actor-isolated simulator for non-blocking circuit execution with progress tracking. Uses
/// ``GateApplication`` (CPU) for circuits below precision policy threshold and ``MetalGateApplication``
/// (GPU) for larger circuits when Metal is available. Progress callbacks batch every 5 gates.
///
/// Precision policy controls CPU/GPU threshold: `.fast` (≥10 qubits), `.balanced` (≥12 qubits),
/// `.accurate` (CPU-only). GPU uses Float32 (~1e-7 precision), CPU uses Float64 (~1e-15 precision).
///
/// For synchronous execution without progress tracking, use ``QuantumCircuit/execute()`` directly.
///
/// **Example:**
/// ```swift
/// let simulator = QuantumSimulator(precisionPolicy: .balanced)
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
/// - SeeAlso: ``QuantumCircuit/execute()``
/// - SeeAlso: ``MetalGateApplication``
/// - SeeAlso: ``GateApplication``
/// - SeeAlso: ``PrecisionPolicy``
public actor QuantumSimulator {
    /// Execution progress information
    @frozen public struct Progress: Sendable {
        /// Number of gates executed so far
        public let executed: Int

        /// Total number of gates in circuit
        public let total: Int

        /// Execution progress as percentage in [0.0, 1.0].
        ///
        /// - Returns: Progress value between 0.0 (not started) and 1.0 (complete)
        /// - Complexity: O(1)
        public var percentage: Double {
            total > 0 ? Double(executed) / Double(total) : 0.0
        }
    }

    /// Precision policy controlling GPU/CPU backend selection.
    public let precisionPolicy: PrecisionPolicy

    /// Whether Metal GPU acceleration is enabled (derived from precision policy).
    private let useMetalAcceleration: Bool

    /// Metal GPU backend (initialized if acceleration enabled)
    private var metalApplication: MetalGateApplication?

    /// Creates a quantum simulator with specified precision policy.
    ///
    /// Initializes actor-isolated simulator with precision-aware backend selection.
    /// GPU threshold varies by policy: `.fast` ≥10 qubits, `.balanced` ≥12 qubits,
    /// `.accurate` forces CPU-only execution for maximum precision.
    ///
    /// **Example:**
    /// ```swift
    /// // Fast mode with GPU acceleration (default)
    /// let simulator = QuantumSimulator()
    ///
    /// // Balanced mode with higher GPU threshold
    /// let balanced = QuantumSimulator(precisionPolicy: .balanced)
    ///
    /// // Accurate mode - CPU only for maximum precision
    /// let accurate = QuantumSimulator(precisionPolicy: .accurate)
    /// ```
    ///
    /// - Parameter precisionPolicy: Precision policy governing GPU threshold (default: `.fast`)
    /// - SeeAlso: ``PrecisionPolicy``
    public init(precisionPolicy: PrecisionPolicy = .fast) {
        self.precisionPolicy = precisionPolicy
        useMetalAcceleration = precisionPolicy.isGPUEnabled
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

        let useGPU: Bool = useMetalAcceleration && metalApplication != nil && PrecisionPolicy.shouldUseGPU(qubits: state.qubits, policy: precisionPolicy)
        let progressMultiplier: Double = operationCount > 0 ? 1.0 / Double(operationCount) : 0.0
        let lastIndex: Int = operationCount - 1
        let operations = circuit.operations

        let progressUpdateBatchSize = 5
        var progressCounter = 0

        for index in 0 ..< operationCount {
            let operation = operations[index]
            if useGPU {
                state = await metalApplication!.apply(operation, state: state)
            } else {
                state = GateApplication.apply(operation, state: state)
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
