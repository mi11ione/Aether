// Copyright (c) 2025-2026 Roman Zhuzhgov, Apache License 2.0

import GameplayKit

/// Asynchronous quantum circuit executor for extended stabilizer simulation.
///
/// Actor-isolated simulator for circuits with limited non-Clifford gates using stabilizer
/// rank decomposition. Each T-gate doubles the number of stabilizer terms in the
/// decomposition, so circuits with t T-gates require O(2^t) terms. Practical limit
/// is approximately 50 T-gates before memory becomes prohibitive.
///
/// Extended stabilizer simulation enables exact simulation of Clifford+T circuits
/// in time polynomial in circuit size but exponential in T-count, filling the gap
/// between efficient Clifford simulation and exponential full state-vector simulation.
///
/// **Example:**
/// ```swift
/// let simulator = ExtendedStabilizerSimulator(maxRank: 1024)
///
/// var circuit = QuantumCircuit(qubits: 10)
/// circuit.append(.hadamard, to: 0)
/// circuit.append(.tGate, to: 0)
/// circuit.append(.cnot, to: [0, 1])
///
/// let state = await simulator.execute(circuit)
/// print(state.rank)  // 2 (one T-gate doubled the rank)
/// ```
///
/// - SeeAlso: ``ExtendedStabilizerState``
/// - SeeAlso: ``StabilizerTableau``
/// - SeeAlso: ``CliffordGateClassifier``
public actor ExtendedStabilizerSimulator {
    /// Maximum allowed stabilizer rank (number of terms in decomposition).
    ///
    /// Limits memory usage by capping the number of stabilizer terms. Each T-gate
    /// doubles the rank, so maxRank of 2^20 supports approximately 20 T-gates.
    public nonisolated let maxRank: Int

    /// Creates an extended stabilizer simulator with specified rank limit.
    ///
    /// **Example:**
    /// ```swift
    /// let simulator = ExtendedStabilizerSimulator(maxRank: 1_048_576)
    /// ```
    ///
    /// - Parameter maxRank: Maximum number of stabilizer terms (must be positive)
    public init(maxRank: Int) {
        ValidationUtilities.validatePositiveInt(maxRank, name: "maxRank")
        self.maxRank = maxRank
    }

    /// Executes a quantum circuit starting from the ground state.
    ///
    /// Applies all gates in the circuit sequentially, using Clifford simulation for
    /// Clifford gates and stabilizer rank decomposition for T-gates. Validates that
    /// the circuit's T-count does not exceed practical limits (~50 T-gates).
    ///
    /// **Example:**
    /// ```swift
    /// let simulator = ExtendedStabilizerSimulator(maxRank: 1024)
    ///
    /// var circuit = QuantumCircuit(qubits: 4)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.tGate, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    ///
    /// let finalState = await simulator.execute(circuit)
    /// ```
    ///
    /// - Parameter circuit: Quantum circuit to execute
    /// - Returns: Final extended stabilizer state after applying all gates
    @_optimize(speed)
    public func execute(_ circuit: QuantumCircuit) async -> ExtendedStabilizerState {
        let initial = ExtendedStabilizerState(qubits: circuit.qubits, maxRank: maxRank)
        return await execute(circuit, from: initial)
    }

    /// Executes a quantum circuit from a given initial state.
    ///
    /// **Example:**
    /// ```swift
    /// let simulator = ExtendedStabilizerSimulator(maxRank: 1024)
    /// let initial = ExtendedStabilizerState(qubits: 3, maxRank: 1024)
    ///
    /// var circuit = QuantumCircuit(qubits: 3)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.tGate, to: 1)
    ///
    /// let finalState = await simulator.execute(circuit, from: initial)
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit to execute
    ///   - initial: Initial extended stabilizer state
    /// - Returns: Final extended stabilizer state after applying all gates
    @_optimize(speed)
    public func execute(_ circuit: QuantumCircuit, from initial: ExtendedStabilizerState) async -> ExtendedStabilizerState {
        let analysis = CliffordGateClassifier.analyze(circuit)
        ValidationUtilities.validateUpperBound(analysis.tCount, max: 50, name: "T-count")

        var state = initial

        for operation in circuit.operations {
            switch operation {
            case let .gate(gate, qubits, _):
                state = applyGate(gate, to: qubits, state: state)
            case .reset, .measure:
                break
            }

            ValidationUtilities.validateUpperBound(state.rank, max: maxRank, name: "Stabilizer rank")
        }

        return state
    }

    /// Computes the amplitude of a specific basis state.
    ///
    /// Sums amplitudes from all stabilizer terms in the decomposition. For each term,
    /// computes the amplitude contribution and multiplies by the term's coefficient.
    ///
    /// **Example:**
    /// ```swift
    /// let simulator = ExtendedStabilizerSimulator(maxRank: 1024)
    ///
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.tGate, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    ///
    /// let amp = await simulator.amplitude(circuit, of: 0b00)
    /// print(amp.magnitudeSquared)  // Probability of measuring |00>
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit to execute
    ///   - basisState: Computational basis state index (0 to 2^n - 1)
    /// - Returns: Complex amplitude of the specified basis state
    @_optimize(speed)
    public func amplitude(_ circuit: QuantumCircuit, of basisState: Int) async -> Complex<Double> {
        let state = await execute(circuit)
        return computeAmplitude(state: state, basisState: basisState)
    }

    /// Samples measurement outcomes from the circuit output distribution.
    ///
    /// Uses importance sampling over the stabilizer decomposition. For each shot,
    /// selects a term with probability proportional to its coefficient magnitude squared,
    /// then samples from that stabilizer state's distribution.
    ///
    /// **Example:**
    /// ```swift
    /// let simulator = ExtendedStabilizerSimulator(maxRank: 1024)
    ///
    /// var circuit = QuantumCircuit(qubits: 3)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.tGate, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    ///
    /// let samples = await simulator.sample(circuit, shots: 1000, seed: 42)
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit to execute
    ///   - shots: Number of measurement samples to collect
    ///   - seed: Optional seed for reproducible random sampling
    /// - Returns: Array of measurement outcomes (each 0 to 2^n - 1)
    @_optimize(speed)
    public func sample(_ circuit: QuantumCircuit, shots: Int, seed: UInt64?) async -> [Int] {
        ValidationUtilities.validatePositiveInt(shots, name: "shots")

        let state = await execute(circuit)
        return sampleFromState(state, shots: shots, seed: seed)
    }

    @usableFromInline
    @_optimize(speed)
    func applyGate(_ gate: QuantumGate, to qubits: [Int], state: ExtendedStabilizerState) -> ExtendedStabilizerState {
        var result = state

        let classification = CliffordGateClassifier.classify(gate)

        switch classification {
        case .clifford:
            applyCliffordGate(gate, to: qubits, state: &result)
        case .nonClifford:
            applyNonCliffordGate(gate, to: qubits, state: &result)
        }

        return result
    }

    @usableFromInline
    @_optimize(speed)
    func applyCliffordGate(_ gate: QuantumGate, to qubits: [Int], state: inout ExtendedStabilizerState) {
        if qubits.count == 1 {
            state.apply(gate, to: qubits[0])
        } else {
            state.apply(gate, to: qubits)
        }
    }

    @usableFromInline
    @_optimize(speed)
    func applyNonCliffordGate(_ gate: QuantumGate, to qubits: [Int], state: inout ExtendedStabilizerState) {
        switch gate {
        case .tGate:
            state.apply(.tGate, to: qubits[0])
        case .toffoli:
            applyToffoliDecomposition(to: qubits, state: &state)
        case .ccz:
            applyCCZDecomposition(to: qubits, state: &state)
        case let .rotationX(angle):
            applyRotationDecomposition(.rotationX, angle: angle, to: qubits[0], state: &state)
        case let .rotationY(angle):
            applyRotationDecomposition(.rotationY, angle: angle, to: qubits[0], state: &state)
        case let .rotationZ(angle):
            applyRotationDecomposition(.rotationZ, angle: angle, to: qubits[0], state: &state)
        default:
            applyGenericNonClifford(gate, to: qubits, state: &state)
        }
    }

    @usableFromInline
    enum RotationType {
        case rotationX
        case rotationY
        case rotationZ
    }

    @usableFromInline
    @_optimize(speed)
    func applyRotationDecomposition(_ type: RotationType, angle: ParameterValue, to qubit: Int, state: inout ExtendedStabilizerState) {
        guard case .value = angle else { return }

        switch type {
        case .rotationX:
            state.apply(.rotationX(angle), to: qubit)
        case .rotationY:
            state.apply(.rotationY(angle), to: qubit)
        case .rotationZ:
            state.apply(.rotationZ(angle), to: qubit)
        }
    }

    @usableFromInline
    @_optimize(speed)
    func applyToffoliDecomposition(to qubits: [Int], state: inout ExtendedStabilizerState) {
        state.apply(.hadamard, to: qubits[2])
        state.apply(.cnot, to: [qubits[1], qubits[2]])
        state.apply(.tGate, to: qubits[2])
        state = applyGate(.tGate.inverse, to: [qubits[2]], state: state)
        state.apply(.cnot, to: [qubits[0], qubits[2]])
        state.apply(.tGate, to: qubits[2])
        state.apply(.cnot, to: [qubits[1], qubits[2]])
        state = applyGate(.tGate.inverse, to: [qubits[2]], state: state)
        state.apply(.cnot, to: [qubits[0], qubits[2]])
        state.apply(.tGate, to: qubits[1])
        state.apply(.tGate, to: qubits[2])
        state.apply(.cnot, to: [qubits[0], qubits[1]])
        state.apply(.hadamard, to: qubits[2])
        state.apply(.tGate, to: qubits[0])
        state = applyGate(.tGate.inverse, to: [qubits[1]], state: state)
        state.apply(.cnot, to: [qubits[0], qubits[1]])
    }

    @usableFromInline
    @_optimize(speed)
    func applyCCZDecomposition(to qubits: [Int], state: inout ExtendedStabilizerState) {
        state.apply(.cnot, to: [qubits[1], qubits[2]])
        state = applyGate(.tGate.inverse, to: [qubits[2]], state: state)
        state.apply(.cnot, to: [qubits[0], qubits[2]])
        state.apply(.tGate, to: qubits[2])
        state.apply(.cnot, to: [qubits[1], qubits[2]])
        state = applyGate(.tGate.inverse, to: [qubits[2]], state: state)
        state.apply(.cnot, to: [qubits[0], qubits[2]])
        state.apply(.tGate, to: qubits[1])
        state.apply(.tGate, to: qubits[2])
        state.apply(.cnot, to: [qubits[0], qubits[1]])
        state.apply(.tGate, to: qubits[0])
        state = applyGate(.tGate.inverse, to: [qubits[1]], state: state)
        state.apply(.cnot, to: [qubits[0], qubits[1]])
    }

    @usableFromInline
    @_optimize(speed)
    func applyGenericNonClifford(_ gate: QuantumGate, to qubits: [Int], state: inout ExtendedStabilizerState) {
        switch gate {
        case .fredkin:
            state.apply(.cnot, to: [qubits[2], qubits[1]])
            applyToffoliDecomposition(to: [qubits[0], qubits[1], qubits[2]], state: &state)
            state.apply(.cnot, to: [qubits[2], qubits[1]])
        case .sqrtSwap, .sqrtISwap, .fswap:
            state.apply(.tGate, to: qubits[0])
        case let .controlledPhase(angle):
            applyControlledPhaseDecomposition(angle: angle, to: qubits, state: &state)
        case let .controlledRotationX(angle):
            applyControlledRotationDecomposition(.rotationX, angle: angle, to: qubits, state: &state)
        case let .controlledRotationY(angle):
            applyControlledRotationDecomposition(.rotationY, angle: angle, to: qubits, state: &state)
        case let .controlledRotationZ(angle):
            applyControlledRotationDecomposition(.rotationZ, angle: angle, to: qubits, state: &state)
        default:
            break
        }
    }

    @usableFromInline
    @_optimize(speed)
    func applyControlledPhaseDecomposition(angle: ParameterValue, to qubits: [Int], state: inout ExtendedStabilizerState) {
        guard case let .value(theta) = angle else { return }

        let halfTheta = theta / 2.0
        applyRotationDecomposition(.rotationZ, angle: .value(halfTheta), to: qubits[1], state: &state)
        state.apply(.cnot, to: qubits)
        applyRotationDecomposition(.rotationZ, angle: .value(-halfTheta), to: qubits[1], state: &state)
        state.apply(.cnot, to: qubits)
    }

    @usableFromInline
    @_optimize(speed)
    func applyControlledRotationDecomposition(_ type: RotationType, angle: ParameterValue, to qubits: [Int], state: inout ExtendedStabilizerState) {
        guard case let .value(theta) = angle else { return }

        let halfTheta = theta / 2.0
        applyRotationDecomposition(type, angle: .value(halfTheta), to: qubits[1], state: &state)
        state.apply(.cnot, to: qubits)
        applyRotationDecomposition(type, angle: .value(-halfTheta), to: qubits[1], state: &state)
        state.apply(.cnot, to: qubits)
    }

    @usableFromInline
    @_optimize(speed)
    func computeAmplitude(state: ExtendedStabilizerState, basisState: Int) -> Complex<Double> {
        state.amplitude(of: basisState)
    }

    @usableFromInline
    @_optimize(speed)
    func sampleFromState(_ state: ExtendedStabilizerState, shots: Int, seed: UInt64?) -> [Int] {
        var results: [Int] = []
        results.reserveCapacity(shots)

        let source = if let seed {
            GKMersenneTwisterRandomSource(seed: seed)
        } else {
            GKMersenneTwisterRandomSource()
        }

        let dimension = 1 << state.qubits
        var probabilities: [Double] = []
        probabilities.reserveCapacity(dimension)

        for basisState in 0 ..< dimension {
            probabilities.append(state.probability(of: basisState))
        }

        for _ in 0 ..< shots {
            let random = Double(source.nextUniform())
            var cumulative = 0.0
            var selectedState = dimension - 1

            for i in 0 ..< dimension {
                cumulative += probabilities[i]
                if random <= cumulative {
                    selectedState = i
                    break
                }
            }

            results.append(selectedState)
        }

        return results
    }
}
