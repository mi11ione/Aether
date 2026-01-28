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
/// without user intervention. Selection priorities:
/// 1. Pure Clifford (tCount = 0) uses tableau for polynomial time
/// 2. Few non-Clifford (tCount <= 50) uses extended stabilizer
/// 3. Small qubit count (<= 25) uses statevector for exactness
/// 4. Large entangled states use MPS tensor networks
///
/// **Example:**
/// ```swift
/// var circuit = QuantumCircuit(qubits: 100)
/// circuit.append(.hadamard, to: 0)
/// circuit.append(.cnot, to: [0, 1])
///
/// let backend = BackendDispatch.selectBackend(for: circuit, qubits: 100)
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
    /// Examines circuit structure to determine the most efficient backend:
    /// - Pure Clifford circuits (tCount = 0) use tableau simulation
    /// - Near-Clifford circuits (tCount <= 50) use extended stabilizer
    /// - Small circuits (<= 25 qubits) use exact statevector
    /// - Large circuits default to MPS tensor network
    ///
    /// **Example:**
    /// ```swift
    /// var cliffordCircuit = QuantumCircuit(qubits: 1000)
    /// cliffordCircuit.append(.hadamard, to: 0)
    /// cliffordCircuit.append(.cnot, to: [0, 1])
    /// let backend1 = BackendDispatch.selectBackend(for: cliffordCircuit, qubits: 1000)
    /// // .tableau
    ///
    /// var tCircuit = QuantumCircuit(qubits: 50)
    /// tCircuit.append(.tGate, to: 0)
    /// let backend2 = BackendDispatch.selectBackend(for: tCircuit, qubits: 50)
    /// // .extendedStabilizer(maxRank: 64)
    ///
    /// var smallCircuit = QuantumCircuit(qubits: 10)
    /// smallCircuit.append(.rotationY(.pi / 7), to: 0)
    /// let backend3 = BackendDispatch.selectBackend(for: smallCircuit, qubits: 10)
    /// // .statevector
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit to analyze
    ///   - qubits: Number of qubits in the system
    /// - Returns: Optimal backend for the circuit characteristics
    /// - Complexity: O(n) where n is the number of operations in the circuit
    @inlinable
    @_effects(readonly)
    public static func selectBackend(for circuit: QuantumCircuit, qubits: Int) -> SimulatorBackend {
        let analysis = CliffordGateClassifier.analyze(circuit)

        if analysis.isClifford {
            return .tableau
        }

        if analysis.tCount <= 50 {
            let maxRank = 1 << min(analysis.tCount, 6)
            return .extendedStabilizer(maxRank: maxRank)
        }

        if qubits <= 25 {
            return .statevector
        }

        let bondDimension = min(256, 1 << min(qubits / 2, 8))
        return .mps(bondDimension: bondDimension)
    }

    /// Executes circuit on specified or auto-selected backend.
    ///
    /// Dispatches circuit execution to the appropriate simulator based on
    /// backend type. When backend is nil, automatically selects optimal
    /// backend using ``selectBackend(for:qubits:)``.
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
    @inlinable
    public static func execute(_ circuit: QuantumCircuit, backend: SimulatorBackend?) async -> QuantumState {
        let selectedBackend = backend ?? selectBackend(for: circuit, qubits: circuit.qubits)

        switch selectedBackend {
        case .tableau:
            return executeTableau(circuit)

        case .extendedStabilizer:
            return executeExtendedStabilizer(circuit)

        case .statevector:
            return executeStatevector(circuit)

        case .densityMatrix:
            return await executeDensityMatrix(circuit)

        case let .mps(bondDimension):
            return executeMPS(circuit, bondDimension: bondDimension)
        }
    }

    @usableFromInline
    static func executeTableau(_ circuit: QuantumCircuit) -> QuantumState {
        circuit.execute()
    }

    @usableFromInline
    static func executeExtendedStabilizer(_ circuit: QuantumCircuit) -> QuantumState {
        circuit.execute()
    }

    @usableFromInline
    static func executeStatevector(_ circuit: QuantumCircuit) -> QuantumState {
        circuit.execute()
    }

    @usableFromInline
    static func executeDensityMatrix(_ circuit: QuantumCircuit) async -> QuantumState {
        let simulator = DensityMatrixSimulator()
        let densityMatrix = await simulator.execute(circuit)
        return densityMatrix.toQuantumState()
    }

    @usableFromInline
    static func executeMPS(_ circuit: QuantumCircuit, bondDimension: Int) -> QuantumState {
        var mps = MatrixProductState(qubits: circuit.qubits, maxBondDimension: bondDimension)

        for operation in circuit.operations {
            switch operation {
            case let .gate(gate, qubits, _):
                applyGateToMPS(gate, qubits: qubits, mps: &mps)
            case let .reset(qubit, _):
                MPSGateApplication.applySingleQubitGate(.pauliX, to: qubit, mps: &mps)
            case let .measure(qubit, _, _):
                MPSGateApplication.applySingleQubitGate(.pauliZ, to: qubit, mps: &mps)
            }
        }

        return mps.toQuantumState()
    }

    @usableFromInline
    static func applyGateToMPS(_ gate: QuantumGate, qubits: [Int], mps: inout MatrixProductState) {
        switch gate.qubitsRequired {
        case 1:
            MPSGateApplication.applySingleQubitGate(gate, to: qubits[0], mps: &mps)
        case 2:
            MPSGateApplication.applyTwoQubitGate(gate, control: qubits[0], target: qubits[1], mps: &mps)
        case 3:
            MPSGateApplication.applyToffoli(control1: qubits[0], control2: qubits[1], target: qubits[2], mps: &mps)
        default:
            preconditionFailure("Unsupported gate qubit count")
        }
    }
}
