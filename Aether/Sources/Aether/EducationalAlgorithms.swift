// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Educational quantum algorithms demonstrating quantum advantage.
///
/// Oracle-based algorithms that showcase exponential and polynomial speedups over
/// classical computation: Deutsch-Jozsa (constant vs balanced), Bernstein-Vazirani
/// (hidden string recovery), and Simon's algorithm (period finding).
///
/// - SeeAlso: ``QuantumCircuit``
/// - SeeAlso: ``QuantumState``
public extension QuantumCircuit {
    /// Oracle function that appends gates implementing a black-box function f.
    typealias Oracle = (_ inputQubits: [Int], _ outputQubit: Int, _ circuit: inout QuantumCircuit) -> Void

    /// Constructs a Deutsch-Jozsa circuit to determine if f is constant or balanced.
    ///
    /// Classically requires 2^(n-1)+1 queries in worst case; quantum requires exactly 1.
    /// After execution, measure input qubits: all |0⟩ indicates constant, any |1⟩ indicates balanced.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.deutschJozsa(qubits: 3, oracle: .balancedParityOracle)
    /// let state = circuit.execute()
    /// let isConstant = state.allQubitsAreZero([0, 1, 2])
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of input qubits (circuit uses n+1 total)
    ///   - oracle: Oracle implementing function f
    /// - Returns: Circuit configured for Deutsch-Jozsa algorithm
    /// - Precondition: qubits >= 1
    /// - Precondition: qubits <= 20
    /// - Complexity: O(n) gates where n = qubits
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func deutschJozsa(
        qubits: Int,
        oracle: Oracle,
    ) -> QuantumCircuit {
        ValidationUtilities.validateMinimumQubits(qubits, min: 1, algorithmName: "Deutsch-Jozsa")
        ValidationUtilities.validateAlgorithmQubitLimit(qubits, max: 20, algorithmName: "Deutsch-Jozsa")

        let totalQubits: Int = qubits + 1
        var circuit = QuantumCircuit(qubits: totalQubits)

        let inputQubits: [Int] = Array(0 ..< qubits)
        let outputQubit: Int = qubits

        circuit.append(.pauliX, to: outputQubit)

        for qubit in 0 ..< totalQubits {
            circuit.append(.hadamard, to: qubit)
        }

        oracle(inputQubits, outputQubit, &circuit)

        for qubit in inputQubits {
            circuit.append(.hadamard, to: qubit)
        }

        return circuit
    }

    /// Constructs a Bernstein-Vazirani circuit to recover hidden string a where f(x) = a·x mod 2.
    ///
    /// Classically requires n queries; quantum requires exactly 1. After execution, measuring
    /// input qubits directly reveals the hidden string.
    ///
    /// **Example:**
    /// ```swift
    /// let oracle = QuantumCircuit.bernsteinVaziraniOracle(hiddenString: [1, 0, 1])
    /// let circuit = QuantumCircuit.bernsteinVazirani(qubits: 3, oracle: oracle)
    /// let result = circuit.execute().measureQubits([0, 1, 2])  // [1, 0, 1]
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits for hidden string (circuit uses n+1 total)
    ///   - oracle: Oracle implementing f(x) = a·x
    /// - Returns: Circuit that reveals hidden string when measured
    /// - Precondition: qubits >= 1
    /// - Precondition: qubits <= 20
    /// - Complexity: O(n) gates where n = qubits
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func bernsteinVazirani(
        qubits: Int,
        oracle: Oracle,
    ) -> QuantumCircuit {
        ValidationUtilities.validateMinimumQubits(qubits, min: 1, algorithmName: "Bernstein-Vazirani")
        ValidationUtilities.validateAlgorithmQubitLimit(qubits, max: 20, algorithmName: "Bernstein-Vazirani")

        let totalQubits: Int = qubits + 1
        var circuit = QuantumCircuit(qubits: totalQubits)

        let inputQubits: [Int] = Array(0 ..< qubits)
        let outputQubit: Int = qubits

        circuit.append(.pauliX, to: outputQubit)
        for qubit in 0 ..< totalQubits {
            circuit.append(.hadamard, to: qubit)
        }

        oracle(inputQubits, outputQubit, &circuit)

        for qubit in inputQubits {
            circuit.append(.hadamard, to: qubit)
        }

        return circuit
    }

    /// Constructs one iteration of Simon's algorithm to find hidden period s where f(x) = f(x⊕s).
    ///
    /// Classically requires O(2^(n/2)) queries; quantum requires O(n). Each iteration yields
    /// a vector y satisfying y·s = 0 mod 2. Repeat n-1 times to collect linearly independent
    /// equations, then solve the linear system to recover s.
    ///
    /// **Example:**
    /// ```swift
    /// let oracle = QuantumCircuit.simonOracle(period: [1, 1])
    /// let circuit = QuantumCircuit.simonIteration(qubits: 2, oracle: oracle)
    /// let y = circuit.execute().measureQubits([0, 1])
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of input qubits (circuit uses 2n total)
    ///   - oracle: Oracle implementing f(x) = f(x⊕s)
    /// - Returns: Circuit for single Simon query iteration
    /// - Precondition: qubits >= 1
    /// - Precondition: qubits <= 15
    /// - Complexity: O(n) gates where n = qubits
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func simonIteration(
        qubits: Int,
        oracle: Oracle,
    ) -> QuantumCircuit {
        ValidationUtilities.validateMinimumQubits(qubits, min: 1, algorithmName: "Simon's algorithm")
        ValidationUtilities.validateAlgorithmQubitLimit(qubits, max: 15, algorithmName: "Simon's algorithm")

        let totalQubits: Int = qubits * 2
        var circuit = QuantumCircuit(qubits: totalQubits)

        let inputQubits: [Int] = Array(0 ..< qubits)
        let outputQubit: Int = qubits

        for qubit in inputQubits {
            circuit.append(.hadamard, to: qubit)
        }

        oracle(inputQubits, outputQubit, &circuit)

        for qubit in inputQubits {
            circuit.append(.hadamard, to: qubit)
        }

        return circuit
    }

    /// Constant-zero oracle for Deutsch-Jozsa: f(x) = 0 for all x.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.deutschJozsa(qubits: 3, oracle: .constantZeroOracle)
    /// ```
    @inlinable
    static var constantZeroOracle: Oracle {
        { _, _, _ in }
    }

    /// Constant-one oracle for Deutsch-Jozsa: f(x) = 1 for all x.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.deutschJozsa(qubits: 3, oracle: .constantOneOracle)
    /// ```
    @inlinable
    static var constantOneOracle: Oracle {
        { _, outputQubit, circuit in
            circuit.append(.pauliX, to: outputQubit)
        }
    }

    /// Balanced parity oracle for Deutsch-Jozsa: f(x) = x₀⊕x₁⊕...⊕xₙ₋₁.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.deutschJozsa(qubits: 3, oracle: .balancedParityOracle)
    /// ```
    @inlinable
    static var balancedParityOracle: Oracle {
        { inputQubits, outputQubit, circuit in
            for input in inputQubits {
                circuit.append(.cnot, to: [input, outputQubit])
            }
        }
    }

    /// Balanced first-bit oracle for Deutsch-Jozsa: f(x) = x₀.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.deutschJozsa(qubits: 3, oracle: .balancedFirstBitOracle)
    /// ```
    @inlinable
    static var balancedFirstBitOracle: Oracle {
        { inputQubits, outputQubit, circuit in
            guard let firstQubit = inputQubits.first else { return }
            circuit.append(.cnot, to: [firstQubit, outputQubit])
        }
    }

    /// Bernstein-Vazirani oracle for hidden string: f(x) = a·x mod 2.
    ///
    /// **Example:**
    /// ```swift
    /// let oracle = QuantumCircuit.bernsteinVaziraniOracle(hiddenString: [1, 0, 1])
    /// let circuit = QuantumCircuit.bernsteinVazirani(qubits: 3, oracle: oracle)
    /// ```
    ///
    /// - Parameter hiddenString: Binary array representing hidden string a
    /// - Precondition: All elements of hiddenString must be 0 or 1
    /// - Complexity: O(n) where n = hiddenString.count
    @_effects(readonly)
    @inlinable
    static func bernsteinVaziraniOracle(hiddenString: [Int]) -> Oracle {
        ValidationUtilities.validateBinaryArray(hiddenString, name: "Hidden string")

        return { inputQubits, outputQubit, circuit in
            ValidationUtilities.validateEqualCounts(inputQubits, hiddenString, name1: "input qubits", name2: "hidden string")

            for i in 0 ..< hiddenString.count {
                guard hiddenString[i] == 1 else { continue }
                circuit.append(.cnot, to: [inputQubits[i], outputQubit])
            }
        }
    }

    /// Simon oracle for hidden period: f(x) = f(x⊕s).
    ///
    /// **Example:**
    /// ```swift
    /// let oracle = QuantumCircuit.simonOracle(period: [1, 1])
    /// let circuit = QuantumCircuit.simonIteration(qubits: 2, oracle: oracle)
    /// ```
    ///
    /// - Parameter period: Binary array representing hidden period s (must be non-zero)
    /// - Precondition: All elements of period must be 0 or 1
    /// - Precondition: period must contain at least one 1
    /// - Complexity: O(n) where n = period.count
    @_effects(readonly)
    @inlinable
    static func simonOracle(period: [Int]) -> Oracle {
        ValidationUtilities.validateBinaryArray(period, name: "Period")
        ValidationUtilities.validateNonZeroBinary(period, name: "Period")

        return { inputQubits, outputQubit, circuit in
            ValidationUtilities.validateEqualCounts(inputQubits, period, name1: "input qubits", name2: "period")

            for i in 0 ..< period.count {
                guard period[i] == 0 else { continue }
                circuit.append(.cnot, to: [inputQubits[i], outputQubit])
            }
        }
    }
}

/// Measurement helpers for educational algorithm result extraction.
public extension QuantumState {
    /// Extracts bit values from the most probable state for specified qubits.
    ///
    /// Returns deterministic results based on highest-probability basis state.
    /// For probabilistic measurement with Born rule sampling, use ``Measurement``.
    ///
    /// **Example:**
    /// ```swift
    /// let state = QuantumCircuit.bell().execute()
    /// let bits = state.measureQubits([0, 1])  // [0, 0] or [1, 1]
    /// ```
    ///
    /// - Parameter indices: Indices of qubits to measure
    /// - Returns: Array of bit values (0 or 1) for each qubit
    /// - Precondition: All qubit indices must be in range [0, state.qubits)
    /// - Complexity: O(2^n) where n = state.qubits
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    @_eagerMove
    func measureQubits(_ indices: [Int]) -> [Int] {
        ValidationUtilities.validateOperationQubits(indices, numQubits: qubits)
        guard !indices.isEmpty else { return [] }

        let (maxIndex, _) = mostProbableState()

        return [Int](unsafeUninitializedCapacity: indices.count) { buffer, count in
            for i in 0 ..< indices.count {
                buffer[i] = BitUtilities.bit(maxIndex, qubit: indices[i])
            }
            count = indices.count
        }
    }

    /// Checks if all specified qubits are |0⟩ in the most probable state.
    ///
    /// Used for Deutsch-Jozsa result interpretation: all zeros indicates constant function.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.deutschJozsa(qubits: 3, oracle: .constantZeroOracle)
    /// let isConstant = circuit.execute().allQubitsAreZero([0, 1, 2])  // true
    /// ```
    ///
    /// - Parameter indices: Indices of qubits to check
    /// - Returns: True if all specified qubits are zero
    /// - Precondition: All qubit indices must be in range [0, state.qubits)
    /// - Complexity: O(2^n) where n = state.qubits
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    func allQubitsAreZero(_ indices: [Int]) -> Bool {
        ValidationUtilities.validateOperationQubits(indices, numQubits: qubits)
        let (maxIndex, _) = mostProbableState()

        for index in indices {
            if BitUtilities.bit(maxIndex, qubit: index) != 0 { return false }
        }
        return true
    }
}
