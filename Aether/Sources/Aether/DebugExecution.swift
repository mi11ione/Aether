// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Darwin

/// Snapshot of circuit state after applying an operation during step-through debugging.
///
/// Captures the operation index, the operation applied (nil for initial state), the resulting
/// quantum state, and the elapsed time in nanoseconds for applying the operation. Snapshots
/// form the execution trace enabling time-travel debugging and state inspection.
///
/// **Example:**
/// ```swift
/// var debug = DebugExecution(circuit: circuit)
/// let snapshot = debug.step()
/// print(snapshot.index)      // 0
/// print(snapshot.elapsedNs)  // Time to apply first gate
/// ```
///
/// - SeeAlso: ``DebugExecution``
/// - SeeAlso: ``QuantumState``
@frozen
public struct DebugSnapshot: Sendable, Equatable {
    /// Index of the operation in the circuit (0-based).
    public let index: Int

    /// The operation applied at this step, or nil for the initial state snapshot.
    public let operation: CircuitOperation?

    /// The quantum state after applying the operation.
    public let state: QuantumState

    /// Time in nanoseconds to apply the operation (0 for initial state).
    public let elapsedNs: UInt64
}

/// Breakdown of single-qubit amplitude probabilities and Bloch sphere coordinates.
///
/// Provides per-qubit analysis of the reduced density matrix by tracing out all other qubits.
/// The Bloch vector (x, y, z) represents the qubit state on the Bloch sphere where pure states
/// lie on the surface (|r| = 1) and mixed states lie inside (|r| < 1).
///
/// Bloch vector components are computed from the reduced density matrix rho:
/// - x = 2 * Re(rho_01) = 2 * Re(⟨0|rho|1⟩)
/// - y = -2 * Im(rho_01) = 2 * Im(rho_10) = Tr(rho * Y)
/// - z = p0 - p1 = rho_00 - rho_11
///
/// **Example:**
/// ```swift
/// let debug = DebugExecution(circuit: circuit)
/// _ = debug.step()
/// let breakdown = debug.amplitudes(qubit: 0)
/// print(breakdown.p0, breakdown.p1)
/// print(breakdown.blochVector)
/// ```
///
/// - SeeAlso: ``DebugExecution``
@frozen
public struct QubitAmplitudeBreakdown: Sendable {
    /// The qubit index this breakdown describes.
    public let qubit: Int

    /// Probability of measuring the qubit in state |0⟩.
    public let p0: Double

    /// Probability of measuring the qubit in state |1⟩.
    public let p1: Double

    /// Bloch sphere coordinates (x, y, z) for the reduced single-qubit state.
    public let blochVector: (x: Double, y: Double, z: Double)
}

/// Step-through circuit debugger with gate-by-gate execution, state snapshots, and timing.
///
/// Enables interactive circuit debugging by executing one operation at a time, recording
/// state snapshots and per-gate timing. Supports forward stepping, multi-step advancement,
/// reset to initial state, and per-qubit amplitude breakdown with Bloch vector computation.
///
/// **Example:**
/// ```swift
/// var circuit = QuantumCircuit(qubits: 2)
/// circuit.append(.hadamard, to: 0)
/// circuit.append(.cnot, to: [0, 1])
///
/// var debug = DebugExecution(circuit: circuit)
/// while !debug.isComplete {
///     let snapshot = debug.step()
///     print("Step \(snapshot.index): \(snapshot.elapsedNs) ns")
///     let q0 = debug.amplitudes(qubit: 0)
///     print("Qubit 0: p0=\(q0.p0), p1=\(q0.p1)")
/// }
/// debug.reset()
/// ```
///
/// - SeeAlso: ``DebugSnapshot``
/// - SeeAlso: ``QubitAmplitudeBreakdown``
/// - SeeAlso: ``QuantumCircuit``
@frozen
public struct DebugExecution: Sendable {
    /// The circuit being debugged.
    public let circuit: QuantumCircuit

    /// Current operation index (0 = before first operation, count = after last operation).
    public private(set) var currentIndex: Int

    /// Execution trace containing snapshots for each step taken.
    public private(set) var trace: [DebugSnapshot]

    @usableFromInline
    var internalState: QuantumState

    @usableFromInline
    let timebaseNumer: Double

    @usableFromInline
    let timebaseDenom: Double

    /// Creates a debug execution session for the specified circuit.
    ///
    /// Initializes the debugger at position 0 (before any operations) with an optional
    /// custom initial state. If no initial state is provided, uses the ground state |00...0⟩.
    /// The initial snapshot (index -1) is recorded in the trace.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// var debug = DebugExecution(circuit: circuit)
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: The quantum circuit to debug.
    ///   - initialState: Optional custom initial state (defaults to ground state).
    /// - Precondition: initialState.qubits must equal circuit.qubits if provided.
    public init(circuit: QuantumCircuit, initialState: QuantumState? = nil) {
        self.circuit = circuit
        currentIndex = 0

        if let state = initialState {
            ValidationUtilities.validateStateQubitCount(state, required: circuit.qubits)
            let maxQubit = circuit.highestQubitIndex
            internalState = QuantumCircuit.expandStateForAncilla(state, maxQubit: maxQubit)
        } else {
            let maxQubit = circuit.highestQubitIndex
            let groundState = QuantumState(qubits: circuit.qubits)
            internalState = QuantumCircuit.expandStateForAncilla(groundState, maxQubit: maxQubit)
        }

        var timebaseInfo = mach_timebase_info_data_t()
        mach_timebase_info(&timebaseInfo)
        timebaseNumer = Double(timebaseInfo.numer)
        timebaseDenom = Double(timebaseInfo.denom)

        let initialSnapshot = DebugSnapshot(
            index: -1,
            operation: nil,
            state: internalState,
            elapsedNs: 0,
        )
        trace = [initialSnapshot]
    }

    /// Executes the next operation and returns a snapshot of the resulting state.
    ///
    /// Applies the operation at `currentIndex`, measures execution time in nanoseconds,
    /// records the snapshot in the trace, and advances the index. If already complete,
    /// returns the final snapshot without modification.
    ///
    /// **Example:**
    /// ```swift
    /// var debug = DebugExecution(circuit: circuit)
    /// let snapshot = debug.step()
    /// print(snapshot.elapsedNs)  // Nanoseconds for this operation
    /// ```
    ///
    /// - Returns: Snapshot containing the operation, resulting state, and timing.
    /// - Complexity: O(2^n) where n is the number of qubits.
    @discardableResult
    public mutating func step() -> DebugSnapshot {
        guard currentIndex < circuit.operations.count else {
            return trace[trace.count - 1]
        }

        let operation = circuit.operations[currentIndex]

        let startTime = mach_absolute_time()
        internalState = GateApplication.apply(operation, state: internalState)
        let endTime = mach_absolute_time()

        let elapsedMach = Double(endTime - startTime)
        let elapsedNs = UInt64(elapsedMach * timebaseNumer / timebaseDenom)

        let snapshot = DebugSnapshot(
            index: currentIndex,
            operation: operation,
            state: internalState,
            elapsedNs: elapsedNs,
        )

        trace.append(snapshot)
        currentIndex += 1

        return snapshot
    }

    /// Executes multiple operations and returns a snapshot of the final state.
    ///
    /// Advances execution by the specified number of steps, stopping early if the circuit
    /// completes. Returns the snapshot from the last step executed.
    ///
    /// **Example:**
    /// ```swift
    /// var debug = DebugExecution(circuit: circuit)
    /// let snapshot = debug.step(count: 5)
    /// print(debug.currentIndex)  // Up to 5, or circuit.count if fewer operations
    /// ```
    ///
    /// - Parameter count: Number of operations to execute.
    /// - Returns: Snapshot from the final step executed.
    /// - Precondition: count must be positive.
    /// - Complexity: O(count * 2^n) where n is the number of qubits.
    @discardableResult
    public mutating func step(count: Int) -> DebugSnapshot {
        ValidationUtilities.validatePositiveInt(count, name: "Step count")

        var lastSnapshot = trace[trace.count - 1]
        for _ in 0 ..< count {
            guard !isComplete else { break }
            lastSnapshot = step()
        }
        return lastSnapshot
    }

    /// Resets the debugger to the initial state before any operations.
    ///
    /// Clears the execution trace except for the initial snapshot and resets the current
    /// index to 0. The internal state returns to the initial state provided at construction.
    ///
    /// **Example:**
    /// ```swift
    /// var debug = DebugExecution(circuit: circuit)
    /// _ = debug.step()
    /// _ = debug.step()
    /// debug.reset()
    /// print(debug.currentIndex)  // 0
    /// print(debug.trace.count)   // 1 (initial snapshot only)
    /// ```
    public mutating func reset() {
        currentIndex = 0

        let initialSnapshot = trace[0]
        internalState = initialSnapshot.state
        trace = [initialSnapshot]
    }

    /// Computes the amplitude breakdown for a single qubit in the current state.
    ///
    /// Calculates the reduced density matrix for the specified qubit by tracing out all
    /// other qubits, then extracts probabilities p0, p1 and Bloch vector coordinates.
    ///
    /// **Example:**
    /// ```swift
    /// var debug = DebugExecution(circuit: circuit)
    /// _ = debug.step()
    /// let breakdown = debug.amplitudes(qubit: 0)
    /// print("P(|0⟩) = \(breakdown.p0)")
    /// print("Bloch: (\(breakdown.blochVector.x), \(breakdown.blochVector.y), \(breakdown.blochVector.z))")
    /// ```
    ///
    /// - Parameter qubit: The qubit index to analyze.
    /// - Returns: Amplitude breakdown with probabilities and Bloch vector.
    /// - Precondition: qubit must be a valid index for the current state.
    /// - Complexity: O(2^n) where n is the number of qubits.
    public func amplitudes(qubit: Int) -> QubitAmplitudeBreakdown {
        ValidationUtilities.validateQubitIndex(qubit, qubits: internalState.qubits)

        let (p0, p1) = internalState.probabilities(for: qubit)

        let qubitMask = 1 << qubit
        var rho01Real = 0.0
        var rho01Imag = 0.0

        let stateSize = internalState.stateSpaceSize
        for i in 0 ..< stateSize where (i & qubitMask) == 0 {
            let j = i | qubitMask
            let amp0 = internalState.amplitudes[i]
            let amp1 = internalState.amplitudes[j]
            let conjAmp1Real = amp1.real
            let conjAmp1Imag = -amp1.imaginary
            rho01Real += amp0.real * conjAmp1Real - amp0.imaginary * conjAmp1Imag
            rho01Imag += amp0.real * conjAmp1Imag + amp0.imaginary * conjAmp1Real
        }

        let x = 2.0 * rho01Real
        let y = -2.0 * rho01Imag
        let z = p0 - p1

        return QubitAmplitudeBreakdown(
            qubit: qubit,
            p0: p0,
            p1: p1,
            blochVector: (x: x, y: y, z: z),
        )
    }

    /// Whether all circuit operations have been executed.
    ///
    /// Returns `true` when `currentIndex` equals the circuit's operation count.
    ///
    /// **Example:**
    /// ```swift
    /// var debug = DebugExecution(circuit: circuit)
    /// while !debug.isComplete {
    ///     _ = debug.step()
    /// }
    /// ```
    @inlinable
    public var isComplete: Bool { currentIndex >= circuit.operations.count }

    /// The current quantum state after all executed operations.
    ///
    /// Reflects the state after the most recently executed operation, or the initial
    /// state if no operations have been executed yet.
    ///
    /// **Example:**
    /// ```swift
    /// var debug = DebugExecution(circuit: circuit)
    /// _ = debug.step()
    /// let probs = debug.currentState.probabilities()
    /// ```
    @inlinable
    public var currentState: QuantumState { internalState }
}
