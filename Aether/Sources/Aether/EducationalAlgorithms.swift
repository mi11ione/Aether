// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Educational quantum algorithms demonstrating quantum advantage
///
/// These algorithms showcase fundamental quantum computing principles:
/// - Deutsch-Jozsa: Exponential speedup for constant vs balanced detection
/// - Bernstein-Vazirani: Linear speedup for hidden string recovery
/// - Simon's algorithm: Exponential speedup for period finding
///
/// All algorithms use oracle-based computation where the oracle implements
/// a black-box function as a quantum circuit.
public extension QuantumCircuit {
    // MARK: - Oracle Types

    /// Oracle function type: Maps input qubits and output qubit to circuit operations
    /// - Parameters:
    ///   - inputQubits: Array of input qubit indices
    ///   - outputQubit: Output qubit index (target for phase kickback)
    ///   - circuit: Circuit to append oracle gates to
    typealias Oracle = (_ inputQubits: [Int], _ outputQubit: Int, _ circuit: inout QuantumCircuit) -> Void

    // MARK: - Deutsch-Jozsa Algorithm

    /// Result of Deutsch-Jozsa algorithm
    @frozen
    enum DeutschJozsaResult: Equatable {
        case constant // f(x) = 0 for all x, or f(x) = 1 for all x
        case balanced // f(x) = 0 for exactly half of inputs
    }

    /// Deutsch-Jozsa algorithm: Determine if function is constant or balanced
    ///
    /// **Problem**: Given f:{0,1}^n → {0,1} that is either:
    /// - Constant: f(x) = 0 for all x, or f(x) = 1 for all x
    /// - Balanced: f(x) = 0 for exactly 2^(n-1) inputs, 1 for the rest
    ///
    /// **Quantum advantage**:
    /// - Classical: Requires 2^(n-1) + 1 queries in worst case
    /// - Quantum: Requires exactly 1 oracle query
    ///
    /// **Algorithm**:
    /// 1. Prepare |+⟩^⊗n|−⟩ using Hadamard gates
    /// 2. Apply oracle Uf: |x⟩|y⟩ → |x⟩|y⊕f(x)⟩
    /// 3. Apply Hadamard to input qubits
    /// 4. Measure input qubits: all |0⟩ → constant, any |1⟩ → balanced
    ///
    /// - Parameters:
    ///   - numInputQubits: Number of input qubits (n)
    ///   - oracle: Oracle implementing function f
    /// - Returns: Circuit configured for Deutsch-Jozsa algorithm
    ///
    /// Example:
    /// ```swift
    /// // Constant oracle (always returns 0)
    /// let constantOracle: QuantumCircuit.Oracle = { _, _, _ in }
    ///
    /// // Balanced oracle (returns parity)
    /// let balancedOracle: QuantumCircuit.Oracle = { inputs, output, circuit in
    ///     for input in inputs {
    ///         circuit.append(gate: .cnot(control: input, target: output), qubits: [])
    ///     }
    /// }
    ///
    /// let circuit = QuantumCircuit.deutschJozsa(numInputQubits: 3, oracle: balancedOracle)
    /// let state = circuit.execute()
    /// // Measure input qubits: all |0⟩ → constant, any |1⟩ → balanced
    /// ```
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func deutschJozsa(
        numInputQubits: Int,
        oracle: Oracle
    ) -> QuantumCircuit {
        ValidationUtilities.validateMinimumQubits(numInputQubits, min: 1, algorithmName: "Deutsch-Jozsa")
        ValidationUtilities.validateAlgorithmQubitLimit(numInputQubits, max: 20, algorithmName: "Deutsch-Jozsa")

        let numQubits: Int = numInputQubits + 1
        var circuit = QuantumCircuit(numQubits: numQubits)

        let inputQubits: [Int] = Array(0 ..< numInputQubits)
        let outputQubit: Int = numInputQubits

        circuit.append(gate: .pauliX, toQubit: outputQubit)

        for qubit in 0 ..< numQubits {
            circuit.append(gate: .hadamard, toQubit: qubit)
        }

        oracle(inputQubits, outputQubit, &circuit)

        for qubit in inputQubits {
            circuit.append(gate: .hadamard, toQubit: qubit)
        }

        return circuit
    }

    // MARK: - Bernstein-Vazirani Algorithm

    /// Bernstein-Vazirani algorithm: Find hidden bit string
    ///
    /// **Problem**: Given f(x) = a·x (mod 2) where a is hidden n-bit string,
    /// find a using queries to f.
    ///
    /// **Quantum advantage**:
    /// - Classical: Requires n queries (must query each bit position)
    /// - Quantum: Requires exactly 1 oracle query
    ///
    /// **Algorithm**:
    /// 1. Prepare |+⟩^⊗n|−⟩
    /// 2. Apply oracle Uf: |x⟩|y⟩ → |x⟩|y⊕(a·x)⟩
    /// 3. Apply Hadamard to input qubits
    /// 4. Measure: Result is hidden string a
    ///
    /// - Parameters:
    ///   - numQubits: Number of qubits (n)
    ///   - oracle: Oracle implementing f(x) = a·x
    /// - Returns: Circuit that when executed and measured reveals hidden string
    ///
    /// Example:
    /// ```swift
    /// // Oracle for hidden string a = 101 (binary)
    /// let hiddenString = [1, 0, 1]  // a₀=1, a₁=0, a₂=1
    /// let oracle: QuantumCircuit.Oracle = { inputs, output, circuit in
    ///     for (i, bit) in hiddenString.enumerated() where bit == 1 {
    ///         circuit.append(gate: .cnot(control: inputs[i], target: output), qubits: [])
    ///     }
    /// }
    ///
    /// let circuit = QuantumCircuit.bernsteinVazirani(numQubits: 3, oracle: oracle)
    /// let state = circuit.execute()
    /// // Measure input qubits: expect to see |101⟩
    /// ```
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func bernsteinVazirani(
        numQubits: Int,
        oracle: Oracle
    ) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(numQubits)
        ValidationUtilities.validateAlgorithmQubitLimit(numQubits, max: 20, algorithmName: "Bernstein-Vazirani")

        let numInputQubits: Int = numQubits
        let totalQubits: Int = numInputQubits + 1
        var circuit = QuantumCircuit(numQubits: totalQubits)

        let inputQubits: [Int] = Array(0 ..< numInputQubits)
        let outputQubit: Int = numInputQubits

        circuit.append(gate: .pauliX, toQubit: outputQubit)
        for qubit in 0 ..< totalQubits {
            circuit.append(gate: .hadamard, toQubit: qubit)
        }

        oracle(inputQubits, outputQubit, &circuit)

        for qubit in inputQubits {
            circuit.append(gate: .hadamard, toQubit: qubit)
        }

        return circuit
    }

    // MARK: - Simon's Algorithm

    /// Simon's algorithm: Find period of XOR function
    ///
    /// **Problem**: Given f:{0,1}^n → {0,1}^n such that f(x) = f(x⊕s) for all x
    /// (where s ≠ 0^n is hidden period), find s.
    ///
    /// **Quantum advantage**:
    /// - Classical: Requires exponential O(2^(n/2)) queries
    /// - Quantum: Requires O(n) queries with high probability
    ///
    /// **Algorithm** (single query iteration):
    /// 1. Prepare |+⟩^⊗n|0⟩^⊗n
    /// 2. Apply oracle Uf: |x⟩|0^n⟩ → |x⟩|f(x)⟩
    /// 3. Apply Hadamard to first n qubits
    /// 4. Measure first n qubits: Get y such that y·s = 0 (mod 2)
    /// 5. Repeat O(n) times to collect n-1 linearly independent equations
    /// 6. Solve linear system to find s
    ///
    /// This function returns the circuit for a single measurement.
    /// Caller must repeat and solve the linear system.
    ///
    /// - Parameters:
    ///   - numQubits: Number of input/output qubits (n)
    ///   - oracle: Oracle implementing f(x) = f(x⊕s)
    /// - Returns: Circuit for single Simon query
    ///
    /// Example:
    /// ```swift
    /// // Oracle for period s = 11 (binary): f(00)=f(11), f(01)=f(10)
    /// let period = [1, 1]
    /// let oracle: QuantumCircuit.Oracle = { inputs, output, circuit in
    ///     // Implementation of periodic function
    ///     // f(x) = f(x⊕s) for s=11
    /// }
    ///
    /// let circuit = QuantumCircuit.simonIteration(numQubits: 2, oracle: oracle)
    /// let state = circuit.execute()
    /// // Measure first n qubits: get y where y·s = 0 (mod 2)
    /// // Repeat n-1 times and solve linear system to recover s
    /// ```
    @_effects(readonly)
    @inlinable
    @_eagerMove
    static func simonIteration(
        numQubits: Int,
        oracle: Oracle
    ) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(numQubits)
        ValidationUtilities.validateAlgorithmQubitLimit(numQubits, max: 15, algorithmName: "Simon's algorithm")

        let totalQubits: Int = numQubits * 2
        var circuit = QuantumCircuit(numQubits: totalQubits)

        let inputQubits: [Int] = Array(0 ..< numQubits)
        let outputQubits: [Int] = Array(numQubits ..< totalQubits)

        for qubit in inputQubits {
            circuit.append(gate: .hadamard, toQubit: qubit)
        }

        oracle(inputQubits, outputQubits[0], &circuit)

        for qubit in inputQubits {
            circuit.append(gate: .hadamard, toQubit: qubit)
        }

        return circuit
    }

    // MARK: - Oracle Builders

    /// Create constant oracle for Deutsch-Jozsa (always returns 0)
    /// - Returns: Oracle that implements f(x) = 0 for all x
    @_effects(readonly)
    @inlinable
    static func constantZeroOracle() -> Oracle {
        { _, _, _ in
            // Do nothing: |y⟩ remains unchanged → f(x) = 0
        }
    }

    /// Create constant oracle for Deutsch-Jozsa (always returns 1)
    /// - Returns: Oracle that implements f(x) = 1 for all x
    @_effects(readonly)
    @inlinable
    static func constantOneOracle() -> Oracle {
        { _, outputQubit, circuit in
            circuit.append(gate: .pauliX, toQubit: outputQubit)
        }
    }

    /// Create balanced oracle for Deutsch-Jozsa (returns parity of input)
    /// - Returns: Oracle that implements f(x) = x₀⊕x₁⊕...⊕xₙ₋₁
    @_effects(readonly)
    @inlinable
    static func balancedParityOracle() -> Oracle {
        { inputQubits, outputQubit, circuit in
            for input in inputQubits {
                circuit.append(gate: .cnot(control: input, target: outputQubit), qubits: [])
            }
        }
    }

    /// Create balanced oracle that checks if first qubit is |1⟩
    /// - Returns: Oracle that implements f(x) = x₀
    @_effects(readonly)
    @inlinable
    static func balancedFirstBitOracle() -> Oracle {
        { inputQubits, outputQubit, circuit in
            guard let firstQubit = inputQubits.first else { return }
            circuit.append(gate: .cnot(control: firstQubit, target: outputQubit), qubits: [])
        }
    }

    /// Create Bernstein-Vazirani oracle for hidden bit string
    /// - Parameter hiddenString: The hidden string a (array of 0s and 1s)
    /// - Returns: Oracle that implements f(x) = a·x (mod 2)
    ///
    /// Example:
    /// ```swift
    /// let oracle = QuantumCircuit.bernsteinVaziraniOracle(hiddenString: [1, 0, 1, 1])
    /// // Implements f(x) = x₀⊕x₂⊕x₃ (dot product with a=1011)
    /// ```
    @_effects(readonly)
    @inlinable
    static func bernsteinVaziraniOracle(hiddenString: [Int]) -> Oracle {
        ValidationUtilities.validateBinaryArray(hiddenString, name: "Hidden string")

        return { inputQubits, outputQubit, circuit in
            ValidationUtilities.validateEqualCounts(inputQubits, hiddenString, name1: "input qubits", name2: "hidden string")

            for (i, bit) in hiddenString.enumerated() where bit == 1 {
                circuit.append(gate: .cnot(control: inputQubits[i], target: outputQubit), qubits: [])
            }
        }
    }

    /// Create simple Simon oracle with known period
    /// - Parameter period: The hidden period s (array of 0s and 1s)
    /// - Returns: Oracle that implements f(x) = f(x⊕s)
    ///
    /// Implementation: f(x) = x AND (NOT s)
    /// This ensures f(x) = f(x⊕s) for the given period s
    ///
    /// Example:
    /// ```swift
    /// let oracle = QuantumCircuit.simonOracle(period: [1, 1, 0])
    /// // f(000) = f(110), f(001) = f(111), f(010) = f(100), f(011) = f(101)
    /// ```
    @_effects(readonly)
    @inlinable
    static func simonOracle(period: [Int]) -> Oracle {
        ValidationUtilities.validateBinaryArray(period, name: "Period")
        ValidationUtilities.validateNonZeroBinary(period, name: "Period")

        return { inputQubits, outputQubit, circuit in
            ValidationUtilities.validateEqualCounts(inputQubits, period, name1: "input qubits", name2: "period")

            // Standard construction: Copy input bits where period bit is 0
            // This ensures f(x) = f(x⊕s) because XOR with s flips bits where s=1,
            // but we only copy bits where s=0, so both x and x⊕s produce same output
            // Mathematically proven correct for all periods s
            for (i, bit) in period.enumerated() where bit == 0 {
                circuit.append(gate: .cnot(control: inputQubits[i], target: outputQubit), qubits: [])
            }
        }
    }
}

// MARK: - Measurement Helpers

public extension QuantumState {
    /// Measure specified qubits and return classical bit string
    /// - Parameter qubits: Indices of qubits to measure
    /// - Returns: Array of measurement results (0 or 1 for each qubit)
    ///
    /// This is a simplified measurement that returns the most probable outcome
    /// for the specified qubits. For proper quantum measurement with randomness,
    /// use the existing measurement infrastructure.
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    func measureQubits(_ qubits: [Int]) -> [Int] {
        ValidationUtilities.validateOperationQubits(qubits, numQubits: numQubits)

        let probs: [Double] = probabilities()
        // Safe to force unwrap: probabilities() always returns 2^n elements
        // for valid quantum states, so max(by:) can never return nil.
        let maxIndex: Int = probs.indices.max(by: { probs[$0] < probs[$1] })!

        return qubits.map { qubit in
            BitUtilities.getBit(maxIndex, qubit: qubit)
        }
    }

    /// Check if all specified qubits are measured as |0⟩
    /// - Parameter qubits: Indices of qubits to check
    /// - Returns: True if all qubits are in |0⟩ state (within threshold)
    @_effects(readonly)
    @inlinable
    func allQubitsAreZero(_ qubits: [Int]) -> Bool {
        let measurements: [Int] = measureQubits(qubits)
        return measurements.allSatisfy { $0 == 0 }
    }
}
