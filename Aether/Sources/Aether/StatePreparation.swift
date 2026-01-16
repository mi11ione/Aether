// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// State preparation utilities for quantum computing
///
/// Provides methods for creating common quantum states: computational basis states,
/// maximally entangled Bell pairs, symmetric W states, and fixed-weight Dicke states.
/// States can be prepared via direct construction (``QuantumState`` extensions, faster)
/// or circuit-based (``QuantumCircuit`` extensions, hardware-compatible).
///
/// **Example:**
/// ```swift
/// let basis = QuantumState.basis(qubits: 3, state: 5)  // |101⟩
/// let bell = QuantumCircuit.bellPhiPlus().execute()     // (|00⟩+|11⟩)/√2
/// let w = QuantumState.w(qubits: 3)                     // (|100⟩+|010⟩+|001⟩)/√3
/// let dicke = QuantumState.dicke(qubits: 4, ones: 2)    // 6-state superposition
/// ```
public extension QuantumState {
    // MARK: - Basis State Preparation

    /// Create computational basis state |i⟩ via direct construction
    ///
    /// Prepares state with only basis state |i⟩ having amplitude 1, all others zero.
    /// Uses little-endian binary encoding where qubit k corresponds to bit k of the index.
    /// More efficient than circuit-based preparation for simulation.
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (n)
    ///   - state: Basis state index i ∈ [0, 2^n-1]
    /// - Returns: Quantum state |i⟩
    ///
    /// **Example:**
    /// ```swift
    /// let zero = QuantumState.basis(qubits: 1, state: 0)   // |0⟩
    /// let one = QuantumState.basis(qubits: 1, state: 1)    // |1⟩
    /// let state5 = QuantumState.basis(qubits: 3, state: 5) // |101⟩ (5 = 0b101)
    /// ```
    ///
    /// - Precondition: `qubits` must be positive and ≤30, `state` must be in [0, 2^n-1]
    /// - Complexity: O(2^n) time and memory for statevector allocation
    /// - SeeAlso: ``QuantumCircuit/basis(qubits:state:)`` for hardware-compatible circuit version
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    static func basis(qubits: Int, state: Int) -> QuantumState {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateMemoryLimit(qubits)

        let stateSpaceSize = 1 << qubits
        ValidationUtilities.validateIndexInBounds(state, bound: stateSpaceSize, name: "Basis state index")

        let amplitudes = [Complex<Double>](unsafeUninitializedCapacity: stateSpaceSize) { buffer, count in
            buffer.initialize(repeating: .zero)
            buffer[state] = .one
            count = stateSpaceSize
        }

        return QuantumState(qubits: qubits, amplitudes: amplitudes)
    }

    // MARK: - W State Preparation

    /// Create W state with symmetric n-qubit entanglement
    ///
    /// Constructs equal superposition over all basis states with exactly one |1⟩:
    /// |W_n⟩ = (|100...0⟩ + |010...0⟩ + ... + |00...01⟩)/√n. Unlike GHZ states,
    /// W states remain entangled after measuring one qubit, making them robust to
    /// particle loss. Belongs to a different entanglement class than Bell/GHZ states.
    ///
    /// - Parameter qubits: Number of qubits (n ≥ 2, practical limit n ≤ 20)
    /// - Returns: W state |W_n⟩
    ///
    /// **Example:**
    /// ```swift
    /// let w3 = QuantumState.w(qubits: 3)
    /// // |W_3⟩ = (|100⟩ + |010⟩ + |001⟩)/√3
    /// let p100 = w3.probability(of: 0b100)  // 1/3
    /// ```
    ///
    /// - Precondition: `qubits` must be ≥2 and ≤20
    /// - Complexity: O(2^n) time and memory
    /// - SeeAlso: ``dicke(qubits:ones:)`` for generalization to arbitrary Hamming weight
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    static func w(qubits: Int) -> QuantumState {
        ValidationUtilities.validateMinimumQubits(qubits, min: 2, algorithmName: "W state")
        ValidationUtilities.validateAlgorithmQubitLimit(qubits, max: 20, algorithmName: "W state")

        let stateSpaceSize = 1 << qubits
        let amplitude = Complex<Double>(1.0 / sqrt(Double(qubits)), 0.0)

        let amplitudes = [Complex<Double>](unsafeUninitializedCapacity: stateSpaceSize) { buffer, count in
            buffer.initialize(repeating: .zero)
            for qubit in 0 ..< qubits {
                buffer[1 << qubit] = amplitude
            }
            count = stateSpaceSize
        }

        return QuantumState(qubits: qubits, amplitudes: amplitudes)
    }

    // MARK: - Dicke State Preparation

    /// Compute binomial coefficient C(n, k) = n! / (k! * (n-k)!)
    /// Uses multiplicative formula to avoid overflow: C(n,k) = ∏(i=1 to k) (n-k+i)/i
    @_optimize(speed)
    @_effects(readonly)
    private static func binomialCoefficient(_ n: Int, _ k: Int) -> Int {
        guard k > 0, k < n else { return 1 }

        let kOpt = min(k, n - k)

        var result = 1
        for i in 0 ..< kOpt {
            result *= (n - i)
            result /= (i + 1)
        }

        return result
    }

    /// Create Dicke state with fixed Hamming weight
    ///
    /// Constructs uniform superposition over all n-qubit basis states with exactly k ones:
    /// |D_k^n⟩ = Σ|x⟩/√C(n,k) where C(n,k) = n!/(k!(n-k)!) is the binomial coefficient.
    /// Generalizes W states (k=1) and appears in quantum metrology, collective spin models,
    /// and quantum error correction code spaces.
    ///
    /// - Parameters:
    ///   - qubits: Total number of qubits (n, practical limit n ≤ 20)
    ///   - ones: Number of |1⟩ qubits (k), where 0 ≤ k ≤ n
    /// - Returns: Dicke state |D_k^n⟩
    ///
    /// **Example:**
    /// ```swift
    /// let d24 = QuantumState.dicke(qubits: 4, ones: 2)
    /// // |D_2^4⟩ = (|0011⟩+|0101⟩+|0110⟩+|1001⟩+|1010⟩+|1100⟩)/√6
    /// let d14 = QuantumState.dicke(qubits: 4, ones: 1)  // Same as W_4
    /// ```
    ///
    /// - Precondition: `qubits` must be positive and ≤20, `ones` must be in [0, qubits]
    /// - Complexity: O(2^n) time and memory
    /// - Note: Uses enumeration optimization when C(n,k) < 2^(n-1) for sparse states
    /// - SeeAlso: ``w(qubits:)`` for the special case where `ones == 1`
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    static func dicke(qubits: Int, ones: Int) -> QuantumState {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateAlgorithmQubitLimit(qubits, max: 20, algorithmName: "Dicke state")
        ValidationUtilities.validateDickeParameters(ones, qubits: qubits)

        let stateSpaceSize = 1 << qubits
        let termCount = binomialCoefficient(qubits, ones)
        let amplitude = Complex<Double>(1.0 / sqrt(Double(termCount)), 0.0)

        var amplitudes = [Complex<Double>](unsafeUninitializedCapacity: stateSpaceSize) { buffer, count in
            buffer.initialize(repeating: .zero)
            count = stateSpaceSize
        }

        enumerateCombinations(n: qubits, k: ones) { state in
            amplitudes[state] = amplitude
        }

        return QuantumState(qubits: qubits, amplitudes: amplitudes)
    }

    /// Enumerate all n-bit integers with exactly k bits set
    /// Calls closure for each combination in lexicographic order
    /// Uses Gosper's hack for O(1) per combination
    @_optimize(speed)
    private static func enumerateCombinations(n: Int, k: Int, body: (Int) -> Void) {
        guard k > 0 else {
            body(0)
            return
        }
        guard k <= n else { return }

        var x = (1 << k) - 1
        let limit = 1 << n

        while x < limit {
            body(x)

            let u = x & -x
            let v = x + u
            x = v | (((v ^ x) >> u.trailingZeroBitCount) >> 2)
        }
    }
}

public extension QuantumCircuit {
    // MARK: - Bell State Variants

    /// Create circuit for Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    ///
    /// Prepares maximally entangled EPR pair via H(0)·CNOT(0,1) on ground state.
    /// This is the canonical Bell state used in quantum teleportation, superdense
    /// coding, and quantum cryptography. Exhibits perfect measurement correlations
    /// and violates the CHSH inequality, demonstrating quantum nonlocality.
    ///
    /// - Returns: 2-qubit circuit creating |Φ⁺⟩
    ///
    /// **Example:**
    /// ```swift
    /// let state = QuantumCircuit.bellPhiPlus().execute()
    /// let p00 = state.probability(of: 0b00)  // 0.5
    /// let p11 = state.probability(of: 0b11)  // 0.5
    /// ```
    ///
    /// - SeeAlso: ``bellPhiMinus()``
    /// - SeeAlso: ``bellPsiPlus()``
    /// - SeeAlso: ``bellPsiMinus()``
    @_eagerMove
    @_effects(readonly)
    static func bellPhiPlus() -> QuantumCircuit {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        return circuit
    }

    /// Create circuit for Bell state |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
    ///
    /// Maximally entangled state with negative relative phase between |00⟩ and |11⟩.
    /// Constructed via H(0)·Z(0)·CNOT(0,1), differing from |Φ⁺⟩ by local phase flip.
    ///
    /// - Returns: 2-qubit circuit creating |Φ⁻⟩
    ///
    /// **Example:**
    /// ```swift
    /// let state = QuantumCircuit.bellPhiMinus().execute()
    /// let p00 = state.probability(of: 0b00)  // 0.5
    /// ```
    ///
    /// - SeeAlso: ``bellPhiPlus()``
    /// - SeeAlso: ``bellPsiPlus()``
    /// - SeeAlso: ``bellPsiMinus()``
    @_eagerMove
    @_effects(readonly)
    static func bellPhiMinus() -> QuantumCircuit {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliZ, to: 0)
        circuit.append(.cnot, to: [0, 1])
        return circuit
    }

    /// Create circuit for Bell state |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
    ///
    /// Maximally entangled state exhibiting perfect anti-correlations: measuring
    /// qubits yields opposite outcomes. Constructed via H(0)·X(1)·CNOT(0,1),
    /// differing from |Φ⁺⟩ by local bit flip on qubit 1.
    ///
    /// - Returns: 2-qubit circuit creating |Ψ⁺⟩
    ///
    /// **Example:**
    /// ```swift
    /// let state = QuantumCircuit.bellPsiPlus().execute()
    /// let p01 = state.probability(of: 0b01)  // 0.5
    /// ```
    ///
    /// - SeeAlso: ``bellPhiPlus()``
    /// - SeeAlso: ``bellPhiMinus()``
    /// - SeeAlso: ``bellPsiMinus()``
    @_eagerMove
    @_effects(readonly)
    static func bellPsiPlus() -> QuantumCircuit {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.cnot, to: [0, 1])
        return circuit
    }

    /// Create circuit for Bell state |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    ///
    /// The singlet state, combining anti-correlation with negative relative phase.
    /// Constructed via H(0)·Z(0)·X(1)·CNOT(0,1). Important in quantum information
    /// and many-body physics as the unique antisymmetric two-qubit entangled state.
    ///
    /// - Returns: 2-qubit circuit creating |Ψ⁻⟩
    ///
    /// **Example:**
    /// ```swift
    /// let state = QuantumCircuit.bellPsiMinus().execute()
    /// let p01 = state.probability(of: 0b01)  // 0.5
    /// ```
    ///
    /// - SeeAlso: ``bellPhiPlus()``
    /// - SeeAlso: ``bellPhiMinus()``
    /// - SeeAlso: ``bellPsiPlus()``
    @_eagerMove
    @_effects(readonly)
    static func bellPsiMinus() -> QuantumCircuit {
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.pauliZ, to: 0)
        circuit.append(.pauliX, to: 1)
        circuit.append(.cnot, to: [0, 1])
        return circuit
    }

    // MARK: - Basis State Preparation Circuit

    /// Create circuit for computational basis state |i⟩ (hardware-compatible)
    ///
    /// Generates gate sequence transforming |00...0⟩ -> |i⟩ using only X gates.
    /// Applies X to qubit k if bit k of state index equals 1 (little-endian encoding).
    /// More hardware-compatible than direct construction, with circuit depth O(1)
    /// since all X gates commute and can execute in parallel.
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (n)
    ///   - state: Target basis state i ∈ [0, 2^n-1]
    /// - Returns: Circuit preparing |i⟩
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.basis(qubits: 3, state: 5)  // Applies X(0), X(2)
    /// let state = circuit.execute()  // |101⟩
    /// let p5 = state.probability(of: 5)  // 1.0
    /// ```
    ///
    /// - Precondition: `qubits` must be positive and ≤30, `state` must be in [0, 2^n-1]
    /// - Complexity: O(popcount(i)) gates where popcount ≤ n
    /// - SeeAlso: ``QuantumState/basis(qubits:state:)`` for direct statevector construction
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    static func basis(qubits: Int, state: Int) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateMemoryLimit(qubits)

        let stateSpaceSize = 1 << qubits
        ValidationUtilities.validateIndexInBounds(state, bound: stateSpaceSize, name: "Basis state index")

        var circuit = QuantumCircuit(qubits: qubits)

        for qubit in 0 ..< qubits {
            if (state >> qubit) & 1 == 1 {
                circuit.append(.pauliX, to: qubit)
            }
        }

        return circuit
    }
}
