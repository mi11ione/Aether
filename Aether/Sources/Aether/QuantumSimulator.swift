// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Async quantum simulator: thread-safe circuit execution with GPU acceleration
///
/// Actor-based quantum simulator providing asynchronous circuit execution
/// with automatic GPU acceleration and progress reporting.
/// Designed for UI applications and long-running quantum computations.
///
/// **Architecture**:
/// - Swift actor: Thread-safe, prevents data races
/// - Automatic GPU: Uses Metal for states ≥ 2^10 qubits (configurable)
/// - Progress tracking: Real-time execution monitoring
///
/// **GPU acceleration**:
/// - Automatic threshold: Switches to Metal when numQubits ≥ 10
/// - Significant speedup: 2-10x for large states (depends on hardware)
/// - Fallback: CPU implementation if Metal unavailable
///
/// **Use cases**:
/// - UI applications: Non-blocking circuit execution
/// - Progress bars: Real-time feedback for long computations
///
/// Example:
/// ```swift
/// // Basic async execution
/// let circuit = QuantumCircuit.bellPhiPlus()
/// let simulator = await QuantumSimulator()
/// let state = await simulator.execute(circuit)
///
/// // With progress updates (for UI)
/// let largeCircuit = QuantumCircuit(numQubits: 15)
/// // ... add many gates ...
///
/// let finalState = await simulator.execute(largeCircuit, progressHandler: { progress in
///     print("Progress: \(Int(progress * 100))%")
///     // Update UI progress bar on main thread
///     await MainActor.run {
///         progressBar.progress = progress
///     }
/// })
///
/// // Convenience: execute directly on circuit
/// let state2 = await circuit.executeAsync()
///
/// // GPU acceleration (automatic)
/// let gpuSimulator = await QuantumSimulator(useMetalAcceleration: true)
/// let bigCircuit = QuantumCircuit(numQubits: 12)  // Will use GPU
/// let result = await gpuSimulator.execute(bigCircuit)
/// ```
public actor QuantumSimulator {
    /// Current quantum state
    private var currentState: QuantumState?

    /// Total number of gates in current circuit
    private var totalGates = 0

    /// Number of gates executed so far
    private var executedGates = 0

    private let useMetalAcceleration: Bool
    private var metalApplication: MetalGateApplication?

    /// Create a new quantum simulator
    /// - Parameter useMetalAcceleration: Whether to use Metal GPU acceleration for large states
    public init(useMetalAcceleration: Bool = true) {
        self.useMetalAcceleration = useMetalAcceleration
        if useMetalAcceleration {
            metalApplication = MetalGateApplication()
        }
    }

    // MARK: - Circuit Execution

    /// Execute quantum circuit asynchronously
    /// - Parameters:
    ///   - circuit: Circuit to execute
    ///   - initialState: Optional initial state (defaults to |00...0⟩)
    ///   - progressHandler: Optional progress callback (0.0 to 1.0)
    /// - Returns: Final quantum state
    @_optimize(speed)
    @_eagerMove
    public func execute(
        _ circuit: QuantumCircuit,
        from initialState: QuantumState? = nil,
        progressHandler: (@isolated(any) @Sendable (Double) async -> Void)? = nil
    ) async -> QuantumState {
        let operationCount: Int = circuit.gateCount
        totalGates = operationCount
        executedGates = 0

        let startState: QuantumState = if let initialState {
            initialState
        } else {
            QuantumState(numQubits: circuit.numQubits)
        }

        let maxQubit: Int = circuit.maxQubitUsed()
        var state = QuantumCircuit.expandStateForAncilla(startState, maxQubit: maxQubit)
        currentState = state

        // Hoist GPU decision before loop - state.numQubits is constant after expansion
        let useGPU: Bool = useMetalAcceleration && metalApplication != nil && state.numQubits >= MetalGateApplication.gpuThreshold
        let progressMultiplier: Double = operationCount > 0 ? 1.0 / Double(operationCount) : 0.0
        let lastIndex: Int = operationCount - 1
        let operations = circuit.operations

        var progressCounter = 0

        for index in 0 ..< operationCount {
            let operation = operations[index]
            if useGPU {
                state = await metalApplication!.apply(gate: operation.gate, to: operation.qubits, state: state)
            } else {
                state = GateApplication.apply(gate: operation.gate, to: operation.qubits, state: state)
            }

            currentState = state
            executedGates = index + 1

            if let progressHandler {
                progressCounter += 1
                if progressCounter >= 5 || index == lastIndex {
                    progressCounter = 0
                    await progressHandler(Double(executedGates) * progressMultiplier)
                }
            }
        }

        return state
    }

    /// Get current execution progress
    /// - Returns: Tuple of (executed gates, total gates, percentage)
    @_effects(readonly)
    public func getProgress() -> (executed: Int, total: Int, percentage: Double) {
        let percentage = totalGates > 0 ? Double(executedGates) / Double(totalGates) : 0.0
        return (executedGates, totalGates, percentage)
    }

    /// Get current quantum state (if available)
    @_effects(readonly)
    public func getCurrentState() -> QuantumState? { currentState }
}

// MARK: - Convenience Extensions

public extension QuantumCircuit {
    /// Execute circuit asynchronously with optional progress updates
    /// - Parameter progressHandler: Optional callback for progress updates (0.0 to 1.0)
    /// - Returns: Final quantum state
    @_eagerMove
    func executeAsync(
        progressHandler: (@isolated(any) @Sendable (Double) async -> Void)? = nil
    ) async -> QuantumState {
        let simulator = QuantumSimulator()
        return await simulator.execute(self, progressHandler: progressHandler)
    }
}
