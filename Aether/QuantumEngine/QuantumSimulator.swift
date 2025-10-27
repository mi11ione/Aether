//
//  QuantumSimulator.swift
//  Aether
//
//  Actor-based quantum simulator with async/await for background execution
//  Created by mi11ion on 21/10/25.
//

import Foundation

/// Thread-safe quantum simulator using Swift actors
/// Supports async execution with cancellation and progress reporting
actor QuantumSimulator {
    // MARK: - Properties

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

    // MARK: - Initialization

    /// Create a new quantum simulator
    /// - Parameter useMetalAcceleration: Whether to use Metal GPU acceleration for large states
    init(useMetalAcceleration: Bool = true) async {
        self.useMetalAcceleration = useMetalAcceleration
        if useMetalAcceleration {
            metalApplication = await MainActor.run { MetalGateApplication() }
        }
    }

    // MARK: - Circuit Execution

    /// Execute quantum circuit asynchronously
    /// - Parameters:
    ///   - circuit: Circuit to execute
    ///   - initialState: Optional initial state (defaults to |00...0⟩)
    /// - Returns: Final quantum state
    /// - Throws: CancellationError if task is cancelled
    func execute(_ circuit: QuantumCircuit, from initialState: QuantumState? = nil) async throws -> QuantumState {
        // Check if already executing
        guard !isExecuting else {
            throw SimulatorError.alreadyExecuting
        }

        isExecuting = true
        totalGates = await circuit.gateCount
        executedGates = 0

        defer {
            isExecuting = false
        }

        // Start with initial state
        let startState: QuantumState = if let initialState {
            initialState
        } else {
            await MainActor.run { QuantumState(numQubits: circuit.numQubits) }
        }
        var state = startState
        currentState = state

        // Execute gates with cancellation checks
        for (index, operation) in circuit.operations.enumerated() {
            // Check for cancellation
            try Task.checkCancellation()

            // Apply gate
            if useMetalAcceleration, let metal = metalApplication, await state.numQubits >= MetalGateApplication.gpuThreshold {
                state = await metal.apply(gate: operation.gate, to: operation.qubits, state: state)
            } else {
                state = await GateApplication.apply(gate: operation.gate, to: operation.qubits, state: state)
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
    func executeWithProgress(
        _ circuit: QuantumCircuit,
        from initialState: QuantumState? = nil,
        progressHandler: @Sendable @escaping (Double) async -> Void
    ) async throws -> QuantumState {
        guard !isExecuting else {
            throw SimulatorError.alreadyExecuting
        }

        isExecuting = true
        totalGates = await circuit.gateCount
        executedGates = 0

        defer {
            isExecuting = false
        }

        let startState: QuantumState = if let initialState {
            initialState
        } else {
            await MainActor.run { QuantumState(numQubits: circuit.numQubits) }
        }
        var state = startState
        currentState = state

        for (index, operation) in circuit.operations.enumerated() {
            try Task.checkCancellation()

            // Apply gate
            if useMetalAcceleration, let metal = metalApplication, await state.numQubits >= MetalGateApplication.gpuThreshold {
                state = await metal.apply(gate: operation.gate, to: operation.qubits, state: state)
            } else {
                state = await GateApplication.apply(gate: operation.gate, to: operation.qubits, state: state)
            }

            currentState = state
            executedGates = index + 1

            // Report progress every 5 gates
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
    func getProgress() -> (executed: Int, total: Int, percentage: Double) {
        let percentage = totalGates > 0 ? Double(executedGates) / Double(totalGates) : 0.0
        return (executedGates, totalGates, percentage)
    }

    /// Get current quantum state (if available)
    /// - Returns: Current state or nil if not executing
    func getCurrentState() -> QuantumState? {
        currentState
    }

    /// Check if simulator is currently executing
    /// - Returns: True if executing
    func isCurrentlyExecuting() -> Bool {
        isExecuting
    }

    // MARK: - Batch Execution

    /// Execute multiple circuits in parallel
    /// - Parameter circuits: Array of circuits to execute
    /// - Returns: Array of final states
    func executeBatch(_ circuits: [QuantumCircuit]) async throws -> [QuantumState] {
        try await withThrowingTaskGroup(of: (Int, QuantumState).self) { group in
            for (index, circuit) in circuits.enumerated() {
                group.addTask {
                    let simulator = await QuantumSimulator(useMetalAcceleration: self.useMetalAcceleration)
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

/// Errors that can occur during simulator operations
enum SimulatorError: Error, LocalizedError {
    case alreadyExecuting
    case invalidCircuit
    case metalNotAvailable

    var errorDescription: String? {
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

extension QuantumCircuit {
    /// Execute circuit asynchronously
    /// - Returns: Final quantum state
    func executeAsync() async throws -> QuantumState {
        let simulator = await QuantumSimulator()
        return try await simulator.execute(self)
    }

    /// Execute circuit with progress updates
    /// - Parameter progressHandler: Called with progress updates
    /// - Returns: Final quantum state
    func executeAsync(
        progressHandler: @Sendable @escaping (Double) async -> Void
    ) async throws -> QuantumState {
        let simulator = await QuantumSimulator()
        return try await simulator.executeWithProgress(self, progressHandler: progressHandler)
    }
}
