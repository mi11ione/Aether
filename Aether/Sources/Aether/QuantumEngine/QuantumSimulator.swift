// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Async quantum simulator: thread-safe circuit execution with GPU acceleration
///
/// Actor-based quantum simulator providing asynchronous circuit execution with
/// automatic GPU acceleration, progress reporting, and cancellation support.
/// Designed for UI applications and long-running quantum computations.
///
/// **Architecture**:
/// - Swift actor: Thread-safe, prevents data races
/// - Automatic GPU: Uses Metal for states ≥ 2^10 qubits (configurable)
/// - Progress tracking: Real-time execution monitoring
/// - Cancellation: Cooperative task cancellation via Task.checkCancellation()
/// - Batch execution: Parallel circuit execution with task groups
///
/// **GPU acceleration**:
/// - Automatic threshold: Switches to Metal when numQubits ≥ 10
/// - Significant speedup: 2-10x for large states (depends on hardware)
/// - Fallback: CPU implementation if Metal unavailable
///
/// **Use cases**:
/// - UI applications: Non-blocking circuit execution
/// - Progress bars: Real-time feedback for long computations
/// - Parallel execution: Run multiple circuits concurrently
/// - Cancellable tasks: User can interrupt long operations
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
/// let finalState = try await simulator.executeWithProgress(largeCircuit) { progress in
///     print("Progress: \(Int(progress * 100))%")
///     // Update UI progress bar on main thread
///     await MainActor.run {
///         progressBar.progress = progress
///     }
/// }
///
/// // Cancellable execution
/// let task = Task {
///     try await simulator.execute(veryLongCircuit)
/// }
/// // Later: user cancels operation
/// task.cancel()
/// // Simulator will throw CancellationError
///
/// // Batch execution (parallel)
/// let circuits = [circuit1, circuit2, circuit3]
/// let results = try await simulator.executeBatch(circuits)
/// // All circuits run in parallel
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

    /// Whether simulator is currently executing a circuit
    private var isExecuting = false

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
    /// - Returns: Final quantum state
    /// - Throws: CancellationError if task is cancelled
    public func execute(_ circuit: QuantumCircuit, from initialState: QuantumState? = nil) async throws -> QuantumState {
        guard !isExecuting else { throw SimulatorError.alreadyExecuting }

        isExecuting = true
        totalGates = circuit.gateCount
        executedGates = 0

        defer { isExecuting = false }

        let startState: QuantumState = if let initialState {
            initialState
        } else {
            QuantumState(numQubits: circuit.numQubits)
        }

        let maxQubit: Int = circuit.maxQubitUsed()
        var state = QuantumCircuit.expandStateForAncilla(startState, maxQubit: maxQubit)
        currentState = state

        for (index, operation) in circuit.operations.enumerated() {
            try Task.checkCancellation()

            if useMetalAcceleration, let metal = metalApplication, state.numQubits >= MetalGateApplication.gpuThreshold {
                state = metal.apply(gate: operation.gate, to: operation.qubits, state: state)
            } else {
                state = GateApplication.apply(gate: operation.gate, to: operation.qubits, state: state)
            }

            currentState = state
            executedGates = index + 1

            // Yield periodically to allow UI updates
            if index % 10 == 0 {
                await Task.yield()
            }
        }

        return state
    }

    /// Execute circuit with progress updates
    /// - Parameters:
    ///   - circuit: Circuit to execute
    ///   - initialState: Optional initial state
    ///   - progressHandler: Called periodically with progress (0.0 to 1.0)
    /// - Returns: Final quantum state
    public func executeWithProgress(
        _ circuit: QuantumCircuit,
        from initialState: QuantumState? = nil,
        progressHandler: @isolated(any) @Sendable @escaping (Double) async -> Void
    ) async throws -> QuantumState {
        guard !isExecuting else {
            throw SimulatorError.alreadyExecuting
        }

        isExecuting = true
        totalGates = circuit.gateCount
        executedGates = 0

        defer {
            isExecuting = false
        }

        let startState: QuantumState = if let initialState {
            initialState
        } else {
            QuantumState(numQubits: circuit.numQubits)
        }

        let maxQubit: Int = circuit.maxQubitUsed()
        var state = QuantumCircuit.expandStateForAncilla(startState, maxQubit: maxQubit)
        currentState = state

        for (index, operation) in circuit.operations.enumerated() {
            try Task.checkCancellation()

            if useMetalAcceleration, let metal = metalApplication, state.numQubits >= MetalGateApplication.gpuThreshold {
                state = metal.apply(gate: operation.gate, to: operation.qubits, state: state)
            } else {
                state = GateApplication.apply(gate: operation.gate, to: operation.qubits, state: state)
            }

            currentState = state
            executedGates = index + 1

            if index % 5 == 0 || index == totalGates - 1 {
                let progress = Double(executedGates) / Double(totalGates)
                await progressHandler(progress)
            }

            await Task.yield()
        }

        return state
    }

    /// Get current execution progress
    /// - Returns: Tuple of (executed gates, total gates, percentage)
    public func getProgress() -> (executed: Int, total: Int, percentage: Double) {
        let percentage = totalGates > 0 ? Double(executedGates) / Double(totalGates) : 0.0
        return (executedGates, totalGates, percentage)
    }

    /// Get current quantum state (if available)
    public func getCurrentState() -> QuantumState? { currentState }

    /// Check if simulator is currently executing
    public func isCurrentlyExecuting() -> Bool { isExecuting }

    // MARK: - Batch Execution

    /// Execute multiple circuits in parallel
    /// - Parameters:
    ///   - circuits: Array of circuits to execute
    ///   - useMetalAcceleration: Whether to use Metal GPU acceleration
    /// - Returns: Array of final states
    public static func executeBatch(
        _ circuits: [QuantumCircuit],
        useMetalAcceleration: Bool = true
    ) async throws -> [QuantumState] {
        try await withThrowingTaskGroup(of: (Int, QuantumState).self) { group in
            for (index, circuit) in circuits.enumerated() {
                group.addTask {
                    let simulator = QuantumSimulator(useMetalAcceleration: useMetalAcceleration)
                    let result = try await simulator.execute(circuit)
                    return (index, result)
                }
            }

            var results: [QuantumState?] = Array(repeating: nil, count: circuits.count)
            for try await (index, state) in group {
                results[index] = state
            }

            return results.compactMap(\.self)
        }
    }
}

// MARK: - Simulator Error

public enum SimulatorError: Error, LocalizedError {
    case alreadyExecuting
    case invalidCircuit
    case metalNotAvailable

    public var errorDescription: String? {
        switch self {
        case .alreadyExecuting:
            "Simulator is already executing a circuit"
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
    func executeAsync() async throws -> QuantumState {
        let simulator = QuantumSimulator()
        return try await simulator.execute(self)
    }

    /// Execute circuit with progress updates
    /// - Parameter progressHandler: Called with progress updates
    /// - Returns: Final quantum state
    func executeAsync(
        progressHandler: @isolated(any) @Sendable @escaping (Double) async -> Void
    ) async throws -> QuantumState {
        let simulator = QuantumSimulator()
        return try await simulator.executeWithProgress(self, progressHandler: progressHandler)
    }
}
