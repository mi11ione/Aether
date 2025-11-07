// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// State preparation: efficient initialization of quantum states
///
/// Provides optimized methods for creating common quantum states beyond ground state |00...0⟩.
/// Supports both direct statevector construction and circuit-based preparation depending
/// on use case. Direct construction is faster for simulation; circuits are hardware-compatible.
///
/// **Supported states**:
/// - **Basis states**: |i⟩ for any i ∈ [0, 2^n-1] (computational basis)
/// - **Bell states**: All four maximally entangled EPR pairs
/// - **W states**: Symmetric n-qubit entanglement robust to particle loss
/// - **Dicke states**: Fixed Hamming weight superpositions for metrology
///
/// **Preparation strategies**:
/// - **Direct construction** (QuantumState extension): O(2^n) memory, fast initialization
/// - **Circuit-based** (QuantumCircuit extension): O(poly(n)) gates, hardware-compatible
///
/// **Usage patterns**:
/// - Simulation: Use direct construction for speed
/// - Hardware: Use circuit-based for gate decomposition
/// - Testing: Both approaches produce identical quantum states
///
/// Example:
/// ```swift
/// // Direct basis state preparation
/// let state5 = QuantumState.basisState(numQubits: 3, basisStateIndex: 5)  // |101⟩
///
/// // Circuit-based basis state
/// let circuit = QuantumCircuit.basisStateCircuit(numQubits: 3, basisStateIndex: 5)
/// let state = circuit.execute()  // Same as above
///
/// // Bell state preparation
/// let bellCircuit = QuantumCircuit.bellPhiPlus()
/// let bellState = bellCircuit.execute()  // (|00⟩ + |11⟩)/√2
///
/// // W state for 3 qubits
/// let w3 = QuantumCircuit.wState(numQubits: 3)
/// // (|100⟩ + |010⟩ + |001⟩)/√3
///
/// // Dicke state: exactly 2 ones among 4 qubits
/// let dicke = QuantumCircuit.dickeState(numQubits: 4, numOnes: 2)
/// // (|0011⟩ + |0101⟩ + |0110⟩ + |1001⟩ + |1010⟩ + |1100⟩)/√6
/// ```
extension QuantumState {
    // MARK: - Basis State Preparation

    /// Create computational basis state |i⟩ via direct construction
    ///
    /// Prepares state where only basis state |i⟩ has amplitude 1, all others zero.
    /// Uses binary representation with little-endian qubit ordering. More efficient
    /// than circuit-based preparation for simulation.
    ///
    /// **Binary encoding**: For index i, qubit k = (i >> k) & 1 (little-endian)
    ///
    /// - Parameters:
    ///   - numQubits: Number of qubits (n)
    ///   - basisStateIndex: Index i ∈ [0, 2^n-1]
    /// - Returns: Quantum state |i⟩
    ///
    /// Example:
    /// ```swift
    /// // Single qubit states
    /// let zero = QuantumState.basisState(numQubits: 1, basisStateIndex: 0)  // |0⟩
    /// let one = QuantumState.basisState(numQubits: 1, basisStateIndex: 1)   // |1⟩
    ///
    /// // Multi-qubit states
    /// let state5 = QuantumState.basisState(numQubits: 3, basisStateIndex: 5)
    /// // |101⟩ since 5 = 0b101 (qubit 0 = 1, qubit 1 = 0, qubit 2 = 1)
    ///
    /// let state7 = QuantumState.basisState(numQubits: 3, basisStateIndex: 7)
    /// // |111⟩ since 7 = 0b111
    ///
    /// // Verify probabilities
    /// let p5 = state5.probability(ofState: 5)  // 1.0
    /// let p0 = state5.probability(ofState: 0)  // 0.0
    /// ```
    ///
    /// **Complexity**:
    /// - Time: O(2^n) for amplitude array allocation, O(1) for initialization
    /// - Memory: O(2^n) for full statevector
    /// - Practical limit: n ≤ 30 qubits
    static func basisState(numQubits: Int, basisStateIndex: Int) -> QuantumState {
        precondition(numQubits > 0, "Number of qubits must be positive")
        precondition(numQubits <= 30, "Number of qubits too large (would exceed memory)")

        let stateSpaceSize = 1 << numQubits
        precondition(basisStateIndex >= 0 && basisStateIndex < stateSpaceSize,
                     "Basis state index \(basisStateIndex) out of bounds [0, \(stateSpaceSize - 1)]")

        var amplitudes = [Complex<Double>](repeating: .zero, count: stateSpaceSize)
        amplitudes[basisStateIndex] = .one

        return QuantumState(numQubits: numQubits, amplitudes: amplitudes)
    }
}

extension QuantumCircuit {
    // MARK: - Bell State Variants

    /// Create circuit for Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    ///
    /// Prepares maximally entangled EPR pair through Hadamard + CNOT.
    /// This is the canonical Bell state used in quantum teleportation,
    /// superdense coding, and quantum cryptography.
    ///
    /// **Construction**: H(0) · CNOT(0,1) on |00⟩
    ///
    /// **Properties**:
    /// - Maximal entanglement (entanglement entropy = 1 bit)
    /// - Perfect correlations: measuring qubit 0 → qubit 1 same result
    /// - CHSH inequality violation: demonstrates quantum nonlocality
    ///
    /// - Returns: 2-qubit circuit that creates |Φ⁺⟩ from ground state
    ///
    /// Example:
    /// ```swift
    /// let circuit = QuantumCircuit.bellPhiPlus()
    /// let state = circuit.execute()
    ///
    /// // Verify Bell state structure
    /// let p00 = state.probability(ofState: 0b00)  // 0.5
    /// let p11 = state.probability(ofState: 0b11)  // 0.5
    /// let p01 = state.probability(ofState: 0b01)  // 0.0
    /// let p10 = state.probability(ofState: 0b10)  // 0.0
    ///
    /// // Single-qubit marginals are maximally mixed
    /// let (p0_q0, p1_q0) = state.singleQubitProbabilities(qubit: 0)
    /// // p0_q0 = 0.5, p1_q0 = 0.5
    /// ```
    static func bellPhiPlus() -> QuantumCircuit {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
        return circuit
    }

    /// Create circuit for Bell state |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
    ///
    /// Maximally entangled state with relative negative phase between |00⟩ and |11⟩.
    /// Related to |Φ⁺⟩ by local Z gate (phase flip).
    ///
    /// **Construction**: H(0) · Z(0) · CNOT(0,1) on |00⟩
    ///
    /// - Returns: 2-qubit circuit that creates |Φ⁻⟩ from ground state
    ///
    /// Example:
    /// ```swift
    /// let circuit = QuantumCircuit.bellPhiMinus()
    /// let state = circuit.execute()
    /// let p00 = state.probability(ofState: 0b00)  // 0.5
    /// let p11 = state.probability(ofState: 0b11)  // 0.5
    /// ```
    static func bellPhiMinus() -> QuantumCircuit {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .pauliZ, toQubit: 0)
        circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
        return circuit
    }

    /// Create circuit for Bell state |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
    ///
    /// Maximally entangled state with perfect anti-correlations: measuring qubits
    /// yields opposite outcomes. Related to |Φ⁺⟩ by local X gate (bit flip on qubit 1).
    ///
    /// **Construction**: H(0) · X(1) · CNOT(0,1) on |00⟩
    ///
    /// - Returns: 2-qubit circuit that creates |Ψ⁺⟩ from ground state
    ///
    /// Example:
    /// ```swift
    /// let circuit = QuantumCircuit.bellPsiPlus()
    /// let state = circuit.execute()
    /// let p01 = state.probability(ofState: 0b01)  // 0.5
    /// let p10 = state.probability(ofState: 0b10)  // 0.5
    /// ```
    static func bellPsiPlus() -> QuantumCircuit {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .pauliX, toQubit: 1)
        circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
        return circuit
    }

    /// Create circuit for Bell state |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    ///
    /// Maximally entangled state combining anti-correlation with relative phase.
    /// This is the singlet state, important in quantum information and many-body physics.
    ///
    /// **Construction**: H(0) · Z(0) · X(1) · CNOT(0,1) on |00⟩
    ///
    /// - Returns: 2-qubit circuit that creates |Ψ⁻⟩ from ground state
    ///
    /// Example:
    /// ```swift
    /// let circuit = QuantumCircuit.bellPsiMinus()
    /// let state = circuit.execute()
    /// let p01 = state.probability(ofState: 0b01)  // 0.5
    /// let p10 = state.probability(ofState: 0b10)  // 0.5
    /// ```
    static func bellPsiMinus() -> QuantumCircuit {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .pauliZ, toQubit: 0)
        circuit.append(gate: .pauliX, toQubit: 1)
        circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
        return circuit
    }

    // MARK: - W State Preparation

    /// Create W state: symmetric n-qubit entanglement robust to particle loss
    ///
    /// W state is equal superposition over all computational basis states with exactly
    /// one |1⟩. Unlike GHZ states, W states remain entangled after measuring one qubit,
    /// making them robust to particle loss and useful for distributed quantum computing.
    ///
    /// **Mathematical definition**: |W_n⟩ = (|100...0⟩ + |010...0⟩ + ... + |00...01⟩)/√n
    ///
    /// **Properties**:
    /// - Symmetric under qubit permutation
    /// - Measuring one qubit → others remain entangled (unlike GHZ)
    /// - Hamming weight = 1 (exactly one qubit in |1⟩)
    /// - Different entanglement class than Bell/GHZ states
    ///
    /// **Algorithm**: Direct statevector construction
    /// - Enumerate all 2^n basis states, select those with weight 1
    /// - Assign amplitude 1/√n to each selected state
    ///
    /// - Parameter numQubits: Number of qubits (n ≥ 2)
    /// - Returns: W state |W_n⟩
    ///
    /// Example:
    /// ```swift
    /// // W state for 3 qubits
    /// let w3 = QuantumCircuit.wState(numQubits: 3)
    /// // |W_3⟩ = (|100⟩ + |010⟩ + |001⟩)/√3
    ///
    /// // Verify structure
    /// let p100 = w3.probability(ofState: 0b100)  // 1/3
    /// let p010 = w3.probability(ofState: 0b010)  // 1/3
    /// let p001 = w3.probability(ofState: 0b001)  // 1/3
    /// let p000 = w3.probability(ofState: 0b000)  // 0.0
    /// let p111 = w3.probability(ofState: 0b111)  // 0.0
    ///
    /// // W state for 4 qubits
    /// let w4 = QuantumCircuit.wState(numQubits: 4)
    /// // |W_4⟩ = (|1000⟩ + |0100⟩ + |0010⟩ + |0001⟩)/2
    /// ```
    ///
    /// **Complexity**:
    /// - Time: O(2^n) statevector enumeration
    /// - Memory: O(2^n) for full statevector
    /// - Practical limit: n ≤ 20 qubits
    static func wState(numQubits: Int) -> QuantumState {
        precondition(numQubits >= 2, "W state requires at least 2 qubits")
        precondition(numQubits <= 20, "W state with >20 qubits requires too much memory")

        let stateSpaceSize = 1 << numQubits
        var amplitudes = [Complex<Double>](repeating: .zero, count: stateSpaceSize)

        // W state is equal superposition over all states with exactly one |1⟩
        // These are states with Hamming weight 1: |100...0⟩, |010...0⟩, etc.
        let amplitude = Complex(1.0 / sqrt(Double(numQubits)), 0.0)

        for i in 0 ..< stateSpaceSize {
            if i.nonzeroBitCount == 1 {
                amplitudes[i] = amplitude
            }
        }

        return QuantumState(numQubits: numQubits, amplitudes: amplitudes)
    }

    // MARK: - Dicke State Preparation

    /// Create Dicke state: equal superposition with fixed Hamming weight
    ///
    /// Dicke state |D_k^n⟩ is uniform superposition over all n-qubit computational
    /// basis states with exactly k qubits in |1⟩. Generalizes W states (k=1) and
    /// appears in quantum metrology, collective spin models, and quantum sensing.
    ///
    /// **Mathematical definition**: |D_k^n⟩ = Σ|x⟩/√(C(n,k)) where |x| has k ones
    /// **Superposition size**: C(n,k) = n!/(k!(n-k)!) basis states
    ///
    /// **Applications**:
    /// - Quantum metrology: Heisenberg-limited sensing
    /// - Collective spins: Symmetric Dicke manifold
    /// - Quantum error correction: Certain code spaces
    ///
    /// **Algorithm**: Direct statevector construction
    /// - Count all basis states with Hamming weight k: C(n,k)
    /// - Assign amplitude 1/√(C(n,k)) to each
    ///
    /// - Parameters:
    ///   - numQubits: Total number of qubits (n)
    ///   - numOnes: Number of |1⟩ qubits (k), where 0 ≤ k ≤ n
    /// - Returns: Dicke state |D_k^n⟩
    ///
    /// Example:
    /// ```swift
    /// // Dicke state with 2 ones among 4 qubits
    /// let d24 = QuantumCircuit.dickeState(numQubits: 4, numOnes: 2)
    /// // |D_2^4⟩ = (|0011⟩+|0101⟩+|0110⟩+|1001⟩+|1010⟩+|1100⟩)/√6
    ///
    /// // Verify: C(4,2) = 6 states, each with probability 1/6
    /// let p0011 = d24.probability(ofState: 0b0011)  // 1/6 ≈ 0.167
    /// let p0000 = d24.probability(ofState: 0b0000)  // 0.0 (wrong weight)
    ///
    /// // Special cases
    /// let d04 = QuantumCircuit.dickeState(numQubits: 4, numOnes: 0)
    /// // |0000⟩ (ground state)
    ///
    /// let d14 = QuantumCircuit.dickeState(numQubits: 4, numOnes: 1)
    /// // |W_4⟩ = (|1000⟩+|0100⟩+|0010⟩+|0001⟩)/2
    ///
    /// let d44 = QuantumCircuit.dickeState(numQubits: 4, numOnes: 4)
    /// // |1111⟩ (all-ones state)
    /// ```
    ///
    /// **Complexity**:
    /// - Time: O(2^n) statevector enumeration
    /// - Memory: O(2^n) for full statevector
    /// - Practical limit: n ≤ 20 qubits
    static func dickeState(numQubits: Int, numOnes: Int) -> QuantumState {
        precondition(numQubits > 0, "Number of qubits must be positive")
        precondition(numQubits <= 20, "Dicke state with >20 qubits requires too much memory")
        precondition(numOnes >= 0 && numOnes <= numQubits,
                     "Number of ones must be between 0 and \(numQubits)")

        let stateSpaceSize = 1 << numQubits
        var amplitudes = [Complex<Double>](repeating: .zero, count: stateSpaceSize)

        var count = 0
        for i in 0 ..< stateSpaceSize {
            if i.nonzeroBitCount == numOnes {
                count += 1
            }
        }

        let amplitude = Complex(1.0 / sqrt(Double(count)), 0.0)

        for i in 0 ..< stateSpaceSize {
            if i.nonzeroBitCount == numOnes {
                amplitudes[i] = amplitude
            }
        }

        return QuantumState(numQubits: numQubits, amplitudes: amplitudes)
    }

    // MARK: - Basis State Preparation Circuit

    /// Create circuit for computational basis state |i⟩ (hardware-compatible)
    ///
    /// Generates gate sequence that transforms |00...0⟩ → |i⟩ using only X gates.
    /// More hardware-compatible than direct statevector construction. Uses binary
    /// representation: apply X to qubit k if bit k of i is 1.
    ///
    /// **Gate count**: O(n) gates (at most n X gates, not exponential)
    /// **Little-endian**: Qubit 0 corresponds to LSB of basisStateIndex
    ///
    /// - Parameters:
    ///   - numQubits: Number of qubits (n)
    ///   - basisStateIndex: Target basis state i ∈ [0, 2^n-1]
    /// - Returns: Circuit that prepares |i⟩ from ground state
    ///
    /// Example:
    /// ```swift
    /// // Prepare |101⟩ (index 5 = 0b101)
    /// let circuit = QuantumCircuit.basisStateCircuit(numQubits: 3, basisStateIndex: 5)
    /// // Applies: X(0), X(2) since bits 0 and 2 are set
    /// let state = circuit.execute()
    /// let p5 = state.probability(ofState: 5)  // 1.0
    ///
    /// // Prepare |1011⟩ (index 11 = 0b1011)
    /// let circuit2 = QuantumCircuit.basisStateCircuit(numQubits: 4, basisStateIndex: 11)
    /// // Applies: X(0), X(1), X(3)
    ///
    /// // Prepare |0000⟩ (index 0 = 0b0000)
    /// let circuit3 = QuantumCircuit.basisStateCircuit(numQubits: 4, basisStateIndex: 0)
    /// // Applies: no gates (already ground state)
    /// ```
    ///
    /// **Complexity**:
    /// - Gates: O(popcount(i)) ≤ O(n) X gates
    /// - Circuit depth: O(1) (all X gates commute)
    /// - Execution: O(2^n) for statevector simulation
    static func basisStateCircuit(numQubits: Int, basisStateIndex: Int) -> QuantumCircuit {
        precondition(numQubits > 0, "Number of qubits must be positive")
        precondition(numQubits <= 30, "Number of qubits too large")

        let stateSpaceSize = 1 << numQubits
        precondition(basisStateIndex >= 0 && basisStateIndex < stateSpaceSize,
                     "Basis state index \(basisStateIndex) out of bounds [0, \(stateSpaceSize - 1)]")

        var circuit = QuantumCircuit(numQubits: numQubits)

        for qubit in 0 ..< numQubits {
            if (basisStateIndex >> qubit) & 1 == 1 {
                circuit.append(gate: .pauliX, toQubit: qubit)
            }
        }

        return circuit
    }
}
