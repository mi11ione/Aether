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
/// let state = try await simulator.execute(circuit)
///
/// // With progress updates (for UI)
/// let largeCircuit = QuantumCircuit(numQubits: 15)
/// // ... add many gates ...
///
/// let finalState = try await simulator.execute(largeCircuit, progressHandler: { progress in
///     print("Progress: \(Int(progress * 100))%")
///     // Update UI progress bar on main thread
///     await MainActor.run {
///         progressBar.progress = progress
///     }
/// })
///
/// // Convenience: execute directly on circuit
/// let state2 = try await circuit.executeAsync()
///
/// // GPU acceleration (automatic)
/// let gpuSimulator = await QuantumSimulator(useMetalAcceleration: true)
/// let bigCircuit = QuantumCircuit(numQubits: 12)  // Will use GPU
/// let result = try await gpuSimulator.execute(bigCircuit)
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
    ) async throws -> QuantumState {
        totalGates = circuit.gateCount
        executedGates = 0

        let startState: QuantumState = if let initialState {
            initialState
        } else {
            QuantumState(numQubits: circuit.numQubits)
        }

        let maxQubit: Int = circuit.maxQubitUsed()
        var state = QuantumCircuit.expandStateForAncilla(startState, maxQubit: maxQubit)
        currentState = state

        for (index, operation) in circuit.operations.enumerated() {
            if useMetalAcceleration, let metal = metalApplication, state.numQubits >= MetalGateApplication.gpuThreshold {
                state = await metal.apply(gate: operation.gate, to: operation.qubits, state: state)
            } else {
                state = GateApplication.apply(gate: operation.gate, to: operation.qubits, state: state)
            }

            currentState = state
            executedGates = index + 1

            if let progressHandler, index % 5 == 0 || index == totalGates - 1 {
                let progress = Double(executedGates) / Double(totalGates)
                await progressHandler(progress)
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

// MARK: - Simulator Error

@frozen
public enum SimulatorError: Error, LocalizedError {
    case invalidCircuit
    case metalNotAvailable

    public var errorDescription: String? {
        switch self {
        case .invalidCircuit:
            "Circuit is invalid or incompatible"
        case .metalNotAvailable:
            "Metal acceleration is not available on this device"
        }
    }
}

// MARK: - Convenience Extensions

public extension QuantumCircuit {
    /// Execute circuit asynchronously
    /// - Returns: Final quantum state
    @_eagerMove
    func executeAsync() async throws -> QuantumState {
        let simulator = QuantumSimulator()
        return try await simulator.execute(self)
    }

    /// Execute circuit with progress updates
    /// - Parameter progressHandler: Called with progress updates
    /// - Returns: Final quantum state
    @_eagerMove
    func executeAsync(
        progressHandler: @isolated(any) @Sendable @escaping (Double) async -> Void
    ) async throws -> QuantumState {
        let simulator = QuantumSimulator()
        return try await simulator.execute(self, progressHandler: progressHandler)
    }
}
