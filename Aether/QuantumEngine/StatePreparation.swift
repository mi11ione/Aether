// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// State preparation utilities for initializing quantum states beyond |00...0⟩
///
/// Provides efficient methods for preparing common quantum states:
/// - Basis states: |i⟩ for any computational basis state
/// - Bell states: All four maximally entangled two-qubit states
/// - W states: Symmetric entangled states robust to qubit loss
/// - Dicke states: Fixed Hamming weight superposition states
extension QuantumState {
    // MARK: - Basis State Preparation

    /// Create quantum state initialized to basis state |i⟩
    /// Uses binary representation: |i⟩ where i ∈ [0, 2^n-1]
    ///
    /// - Parameters:
    ///   - numQubits: Number of qubits
    ///   - basisStateIndex: Index of computational basis state (0 to 2^n-1)
    /// - Returns: Quantum state |i⟩ with amplitude 1 at index i, 0 elsewhere
    ///
    /// Example:
    /// - basisState(numQubits: 3, basisStateIndex: 5) creates |101⟩
    /// - Efficient: O(2^n) memory but O(1) initialization (single amplitude set)
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

    /// Create Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    /// Alias for bellState() to provide consistent naming with other Bell state variants
    /// Construction: H(0) · CNOT(0,1)
    /// - Returns: Circuit that creates |Φ⁺⟩ from |00⟩
    static func bellPhiPlus() -> QuantumCircuit {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
        return circuit
    }

    /// Create Bell state |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
    /// Construction: H(0) · Z(0) · CNOT(0,1)
    /// - Returns: Circuit that creates |Φ⁻⟩ from |00⟩
    static func bellPhiMinus() -> QuantumCircuit {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .pauliZ, toQubit: 0)
        circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
        return circuit
    }

    /// Create Bell state |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
    /// Construction: H(0) · X(1) · CNOT(0,1)
    /// - Returns: Circuit that creates |Ψ⁺⟩ from |00⟩
    static func bellPsiPlus() -> QuantumCircuit {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .pauliX, toQubit: 1)
        circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
        return circuit
    }

    /// Create Bell state |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    /// Construction: H(0) · Z(0) · X(1) · CNOT(0,1)
    /// - Returns: Circuit that creates |Ψ⁻⟩ from |00⟩
    static func bellPsiMinus() -> QuantumCircuit {
        var circuit = QuantumCircuit(numQubits: 2)
        circuit.append(gate: .hadamard, toQubit: 0)
        circuit.append(gate: .pauliZ, toQubit: 0)
        circuit.append(gate: .pauliX, toQubit: 1)
        circuit.append(gate: .cnot(control: 0, target: 1), qubits: [])
        return circuit
    }

    // MARK: - W State Preparation

    /// Create W state |W_n⟩ = (|100...0⟩ + |010...0⟩ + ... + |00...01⟩)/√n
    ///
    /// The W state is a symmetric entangled state robust to single-qubit loss.
    /// Unlike GHZ state, measuring one qubit in a W state leaves others entangled.
    ///
    /// Properties:
    /// - Equal superposition over all n-qubit states with exactly one |1⟩
    /// - Highly entangled but robust to particle loss
    /// - Different entanglement structure than GHZ state
    ///
    /// Algorithm: Direct state construction (statevector-based)
    /// - Enumerate all basis states with Hamming weight 1
    /// - Set equal amplitude 1/√n for each
    /// - Efficient and exact for simulation
    ///
    /// - Parameter numQubits: Number of qubits (n ≥ 2)
    /// - Returns: Quantum state |W_n⟩
    ///
    /// Example: |W_3⟩ = (|100⟩ + |010⟩ + |001⟩)/√3
    ///
    /// Complexity:
    /// - Time: O(2^n) to enumerate states
    /// - Memory: O(2^n) for state vector
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

    /// Create Dicke state |D_k^n⟩ with exactly k ones among n qubits
    ///
    /// Dicke state is equal superposition over all basis states with Hamming weight k.
    /// Example: |D_2^4⟩ = (|0011⟩+|0101⟩+|0110⟩+|1001⟩+|1010⟩+|1100⟩)/√6
    ///
    /// Applications:
    /// - Quantum metrology and sensing
    /// - Collective spin systems
    /// - Quantum error correction
    ///
    /// Algorithm: Direct state construction (statevector-based)
    /// - Enumerate all n-bit strings with k ones
    /// - Set equal amplitude 1/√(C(n,k)) for each
    /// - Efficient for small k or small (n-k)
    ///
    /// - Parameters:
    ///   - numQubits: Total number of qubits (n)
    ///   - numOnes: Number of qubits in |1⟩ state (k), where 0 ≤ k ≤ n
    /// - Returns: Quantum state |D_k^n⟩
    ///
    /// Complexity:
    /// - Time: O(2^n) to enumerate states
    /// - Memory: O(2^n) for state vector
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

    /// Create circuit that prepares computational basis state |i⟩ from |00...0⟩
    ///
    /// Uses binary representation of i to apply X gates:
    /// - If bit k of i is 1, apply X to qubit k
    /// - Result: |00...0⟩ → |i⟩ with O(n) gates (not O(2^n))
    ///
    /// - Parameters:
    ///   - numQubits: Number of qubits
    ///   - basisStateIndex: Target basis state index (0 to 2^n-1)
    /// - Returns: Circuit that prepares |i⟩
    ///
    /// Example:
    /// - basisStateCircuit(numQubits: 4, basisStateIndex: 11) prepares |1011⟩
    /// - Applies: X(0), X(1), X(3) (bits 0,1,3 are set in binary 1011)
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
