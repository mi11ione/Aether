// Copyright (c) 2025-2026 Roman Zhuzhgov, Apache License 2.0

/// Simulator backend type for quantum circuit execution.
///
/// Each backend offers different tradeoffs between simulation capability,
/// memory usage, and computational complexity based on circuit structure.
///
/// **Example:**
/// ```swift
/// let backend = SimulatorBackend.tableau
/// let extended = SimulatorBackend.extendedStabilizer(maxRank: 32)
/// let mps = SimulatorBackend.mps(bondDimension: 64)
/// ```
///
/// - SeeAlso: ``BackendDispatch``
@frozen public enum SimulatorBackend: Equatable, Sendable {
    /// Stabilizer tableau simulation for Clifford circuits.
    ///
    /// Achieves polynomial-time simulation O(n^2) per gate for circuits
    /// containing only Clifford gates. Supports 10M+ qubits.
    case tableau

    /// Extended stabilizer simulation for near-Clifford circuits.
    ///
    /// Complexity exponential only in T-count, not qubit count.
    /// Efficient when tCount <= maxRank.
    ///
    /// - Parameter maxRank: Maximum stabilizer rank for decomposition
    case extendedStabilizer(maxRank: Int)

    /// Full statevector simulation for exact computation.
    ///
    /// Stores all 2^n complex amplitudes. Memory: O(2^n).
    /// Practical for circuits with up to 25 qubits.
    case statevector

    /// Density matrix simulation for noisy circuits.
    ///
    /// Stores full 2^n x 2^n density matrix. Memory: O(4^n).
    /// Required for mixed states and noise channel simulation.
    case densityMatrix

    /// Matrix Product State simulation using tensor networks.
    ///
    /// Efficient for circuits with limited entanglement.
    /// Memory scales with bond dimension, not exponentially with qubits.
    ///
    /// - Parameter bondDimension: Maximum bond dimension for truncation
    case mps(bondDimension: Int)
}

/// Automatic backend selection and circuit execution dispatch.
///
/// Analyzes circuit composition to select the optimal simulation backend
/// without user intervention. Pure Clifford circuits with zero T-count use
/// tableau simulation for polynomial time performance, while circuits with
/// few non-Clifford gates (T-count up to 50) use extended stabilizer
/// decomposition. Small circuits with 25 or fewer qubits use statevector
/// for exact computation, and large entangled states fall back to MPS
/// tensor networks.
///
/// **Example:**
/// ```swift
/// var circuit = QuantumCircuit(qubits: 100)
/// circuit.append(.hadamard, to: 0)
/// circuit.append(.cnot, to: [0, 1])
///
/// let backend = BackendDispatch.selectBackend(for: circuit)
/// // Returns .tableau since circuit is pure Clifford
///
/// let state = await BackendDispatch.execute(circuit, backend: nil)
/// // Auto-selects optimal backend
/// ```
///
/// - SeeAlso: ``SimulatorBackend``
/// - SeeAlso: ``CliffordGateClassifier``
/// - SeeAlso: ``QuantumSimulator``
public enum BackendDispatch {
    /// Selects optimal simulation backend based on circuit analysis.
    ///
    /// Examines circuit structure to determine the most efficient backend.
    /// Pure Clifford circuits with zero T-count use tableau simulation,
    /// near-Clifford circuits with T-count up to 50 use extended stabilizer,
    /// small circuits with 25 or fewer qubits use exact statevector, and
    /// large circuits default to MPS tensor network.
    ///
    /// **Example:**
    /// ```swift
    /// var cliffordCircuit = QuantumCircuit(qubits: 1000)
    /// cliffordCircuit.append(.hadamard, to: 0)
    /// cliffordCircuit.append(.cnot, to: [0, 1])
    /// let backend1 = BackendDispatch.selectBackend(for: cliffordCircuit)
    /// // .tableau
    ///
    /// var tCircuit = QuantumCircuit(qubits: 50)
    /// tCircuit.append(.tGate, to: 0)
    /// let backend2 = BackendDispatch.selectBackend(for: tCircuit)
    /// // .extendedStabilizer(maxRank: 64)
    ///
    /// var smallCircuit = QuantumCircuit(qubits: 10)
    /// smallCircuit.append(.rotationY(.pi / 7), to: 0)
    /// let backend3 = BackendDispatch.selectBackend(for: smallCircuit)
    /// // .statevector
    /// ```
    ///
    /// - Parameter circuit: Quantum circuit to analyze
    /// - Returns: Optimal backend for the circuit characteristics
    /// - Complexity: O(n) where n is the number of operations in the circuit
    @inlinable
    @_effects(readonly)
    public static func selectBackend(for circuit: QuantumCircuit) -> SimulatorBackend {
        let analysis = CliffordGateClassifier.analyze(circuit)

        if analysis.isClifford {
            return .tableau
        }

        if analysis.tCount <= 50 {
            let maxRank = 1 << min(analysis.tCount, 6)
            return .extendedStabilizer(maxRank: maxRank)
        }

        if circuit.qubits <= 25 {
            return .statevector
        }

        let bondDimension = min(256, 1 << min(circuit.qubits / 2, 8))
        return .mps(bondDimension: bondDimension)
    }

    /// Executes circuit on specified or auto-selected backend.
    ///
    /// Dispatches circuit execution to the appropriate simulator based on
    /// backend type. When backend is nil, automatically selects optimal
    /// backend using ``selectBackend(for:)``.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 3)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    /// circuit.append(.cnot, to: [1, 2])
    ///
    /// let state1 = await BackendDispatch.execute(circuit, backend: nil)
    /// // Auto-selects backend
    ///
    /// let state2 = await BackendDispatch.execute(circuit, backend: .statevector)
    /// // Forces statevector simulation
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit to execute
    ///   - backend: Explicit backend selection, or nil for automatic
    /// - Returns: Final quantum state after circuit execution
    /// - Complexity: Depends on backend; O(n^2 g) for tableau, O(2^n g) for statevector
    /// - SeeAlso: ``selectBackend(for:)``
    @inlinable
    public static func execute(_ circuit: QuantumCircuit, backend: SimulatorBackend?) async -> QuantumState {
        let selectedBackend = backend ?? selectBackend(for: circuit)

        switch selectedBackend {
        case .tableau:
            return await executeTableau(circuit)

        case let .extendedStabilizer(maxRank):
            return await executeExtendedStabilizer(circuit, maxRank: maxRank)

        case .statevector:
            return circuit.execute()

        case .densityMatrix:
            return await executeDensityMatrix(circuit)

        case let .mps(bondDimension):
            return executeMPS(circuit, bondDimension: bondDimension)
        }
    }

    /// Executes the circuit using stabilizer tableau simulation.
    @usableFromInline
    static func executeTableau(_ circuit: QuantumCircuit) async -> QuantumState {
        guard circuit.qubits <= 20 else {
            return circuit.execute()
        }
        let simulator = CliffordSimulator()
        let tableau = await simulator.execute(circuit)
        return tableau.toQuantumState()
    }

    /// Executes the circuit using extended stabilizer simulation.
    @usableFromInline
    static func executeExtendedStabilizer(_ circuit: QuantumCircuit, maxRank: Int) async -> QuantumState {
        guard circuit.qubits <= 20 else {
            return circuit.execute()
        }
        let simulator = ExtendedStabilizerSimulator(maxRank: maxRank)
        let state = await simulator.execute(circuit)
        return state.toQuantumState()
    }

    /// Executes the circuit using density matrix simulation.
    @usableFromInline
    static func executeDensityMatrix(_ circuit: QuantumCircuit) async -> QuantumState {
        let simulator = DensityMatrixSimulator()
        let densityMatrix = await simulator.execute(circuit)
        return densityMatrix.toQuantumState()
    }

    /// Executes the circuit using matrix product state simulation.
    @usableFromInline
    @_optimize(speed)
    static func executeMPS(_ circuit: QuantumCircuit, bondDimension: Int) -> QuantumState {
        var mps = MatrixProductState(qubits: circuit.qubits, maxBondDimension: bondDimension)

        for operation in circuit.operations {
            switch operation {
            case let .gate(gate, qubits, _):
                MPSGateApplication.apply(gate, to: qubits, mps: &mps)
            case let .reset(qubit, _):
                MPSGateApplication.reset(qubit, mps: &mps)
            case let .measure(qubit, _, _):
                MPSGateApplication.measure(qubit, mps: &mps)
            }
        }

        return mps.toQuantumState()
    }
}
