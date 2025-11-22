// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Batch parameter binding extensions for efficient VQE gradient computation
///
/// Provides convenient methods to bind multiple parameter sets simultaneously,
/// enabling batched circuit evaluation via MPSBatchEvaluator. Eliminates boilerplate
/// for common VQE workflows like gradient computation and grid search.
///
/// **Use Cases:**
/// 1. **VQE gradients**: Bind all θᵢ±π/2 parameter sets in single call
/// 2. **Grid search**: Bind all grid point combinations at once
/// 3. **Population optimizers**: Bind entire population in batch
/// 4. **Hyperparameter tuning**: Test multiple configurations simultaneously
///
/// Example - VQE gradient computation:
/// ```swift
/// let ansatz = HardwareEfficientAnsatz.create(numQubits: 6, depth: 2)
/// let baseParams: [Double] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
///
/// // Generate all shifted parameter sets for gradient
/// let (plusVectors, minusVectors) = ansatz.generateGradientParameterVectors(
///     baseParameters: baseParams
/// )
///
/// // Bind all in batch (12 circuits total)
/// let plusCircuits = try ansatz.bindBatch(parameterVectors: plusVectors)
/// let minusCircuits = try ansatz.bindBatch(parameterVectors: minusVectors)
///
/// // Convert to unitaries and batch evaluate
/// let allCircuits = plusCircuits + minusCircuits
/// let unitaries = try allCircuits.map { try CircuitUnitary.computeUnitary(circuit: $0) }
///
/// let evaluator = await MPSBatchEvaluator()
/// let energies = try await evaluator.evaluateExpectationValues(
///     unitaries: unitaries,
///     initialState: QuantumState(numQubits: 6),
///     hamiltonian: hamiltonian
/// )
///
/// // Extract gradients
/// for i in 0..<baseParams.count {
///     let gradient = (energies[i] - energies[i + baseParams.count]) / 2.0
///     print("∂E/∂θ[\(i)] = \(gradient)")
/// }
/// ```
public extension ParameterizedQuantumCircuit {
    // MARK: - Batch Binding

    /// Bind multiple parameter vectors to produce batch of circuits
    ///
    /// Efficient batch binding for VQE gradient computation and grid search.
    /// Validates all parameter vectors have correct length before binding.
    ///
    /// **Performance:**
    /// - O(N × M) where N = batch size, M = gates per circuit
    /// - Sequential binding but with upfront validation
    /// - Amortized cost negligible compared to circuit execution
    ///
    /// - Parameter parameterVectors: Array of parameter value arrays
    /// - Returns: Array of concrete quantum circuits (one per parameter vector)
    /// - Throws: ParameterError if any vector has wrong length
    ///
    /// Example:
    /// ```swift
    /// let ansatz = ParameterizedQuantumCircuit(numQubits: 4)
    /// // ... build ansatz with parameters theta_0, theta_1, theta_2, theta_3
    ///
    /// let parameterSets: [[Double]] = [
    ///     [0.1, 0.2, 0.3, 0.4],
    ///     [0.5, 0.6, 0.7, 0.8],
    ///     [0.9, 1.0, 1.1, 1.2]
    /// ]
    ///
    /// let circuits = try ansatz.bindBatch(parameterVectors: parameterSets)
    /// // circuits[0] has parameters [0.1, 0.2, 0.3, 0.4]
    /// // circuits[1] has parameters [0.5, 0.6, 0.7, 0.8]
    /// // circuits[2] has parameters [0.9, 1.0, 1.1, 1.2]
    /// ```
    @_optimize(speed)
    @_eagerMove
    func bindBatch(parameterVectors: [[Double]]) throws -> [QuantumCircuit] {
        guard !parameterVectors.isEmpty else {
            return []
        }

        let expectedCount: Int = parameterCount()

        for (index, vector) in parameterVectors.enumerated() {
            guard vector.count == expectedCount else {
                throw ParameterError.invalidVectorLengthInBatch(
                    batchIndex: index,
                    expected: expectedCount,
                    got: vector.count
                )
            }
        }

        var circuits: [QuantumCircuit] = []
        circuits.reserveCapacity(parameterVectors.count)

        for vector in parameterVectors {
            let circuit: QuantumCircuit = try bind(parameterVector: vector)
            circuits.append(circuit)
        }

        return circuits
    }

    // MARK: - Gradient Helper Methods

    /// Generate parameter vectors for gradient computation via parameter shift rule
    ///
    /// Creates 2N parameter vectors for N parameters, with each parameter shifted
    /// by ±π/2 for gradient evaluation. Optimized for VQE gradient computation.
    ///
    /// **Parameter Shift Rule:**
    /// - ∂⟨H⟩/∂θᵢ = [⟨H⟩(θᵢ+π/2) - ⟨H⟩(θᵢ-π/2)] / 2
    /// - Requires 2N circuit evaluations for N parameters
    /// - More accurate than finite differences for quantum circuits
    ///
    /// **Output Format:**
    /// - plusVectors[i]: All parameters at base values except θᵢ → θᵢ + π/2
    /// - minusVectors[i]: All parameters at base values except θᵢ → θᵢ - π/2
    /// - Total vectors: 2N
    ///
    /// - Parameters:
    ///   - baseParameters: Base parameter values
    ///   - shift: Shift amount (default: π/2 for standard parameter shift)
    /// - Returns: Tuple of (plus vectors, minus vectors)
    /// - Throws: ParameterError if baseParameters length wrong
    ///
    /// Example:
    /// ```swift
    /// let ansatz = HardwareEfficientAnsatz.create(numQubits: 4, depth: 2)
    /// let baseParams: [Double] = Array(repeating: 0.1, count: ansatz.parameterCount())
    ///
    /// // Generate shifted parameter sets
    /// let (plusVectors, minusVectors) = try ansatz.generateGradientParameterVectors(
    ///     baseParameters: baseParams
    /// )
    ///
    /// // Bind all circuits (2N total)
    /// let plusCircuits = try ansatz.bindBatch(parameterVectors: plusVectors)
    /// let minusCircuits = try ansatz.bindBatch(parameterVectors: minusVectors)
    ///
    /// // Batch evaluate for gradients
    /// let allCircuits = plusCircuits + minusCircuits
    /// let unitaries = try allCircuits.map { try CircuitUnitary.computeUnitary(circuit: $0) }
    ///
    /// let evaluator = await MPSBatchEvaluator()
    /// let energies = try await evaluator.evaluateExpectationValues(
    ///     unitaries: unitaries,
    ///     initialState: QuantumState(numQubits: 4),
    ///     hamiltonian: hamiltonian
    /// )
    ///
    /// // Extract gradients
    /// let numParams = baseParams.count
    /// for i in 0..<numParams {
    ///     let gradient = (energies[i] - energies[i + numParams]) / 2.0
    ///     print("∂E/∂θ[\(i)] = \(gradient)")
    /// }
    /// ```
    @_optimize(speed)
    @_eagerMove
    func generateGradientParameterVectors(
        baseParameters: [Double],
        shift: Double = .pi / 2
    ) throws -> (plusVectors: [[Double]], minusVectors: [[Double]]) {
        let numParams: Int = parameterCount()

        guard baseParameters.count == numParams else {
            throw ParameterError.invalidVectorLength(expected: numParams, got: baseParameters.count)
        }

        var plusVectors: [[Double]] = []
        var minusVectors: [[Double]] = []

        plusVectors.reserveCapacity(numParams)
        minusVectors.reserveCapacity(numParams)

        for i in 0 ..< numParams {
            var plusVector = baseParameters
            plusVector[i] += shift

            var minusVector = baseParameters
            minusVector[i] -= shift

            plusVectors.append(plusVector)
            minusVectors.append(minusVector)
        }

        return (plusVectors: plusVectors, minusVectors: minusVectors)
    }

    /// Generate parameter vectors for grid search optimization
    ///
    /// Creates Cartesian product of parameter ranges for exhaustive grid search.
    /// Useful for QAOA parameter optimization over (γ,β) space.
    ///
    /// **Algorithm:**
    /// - Takes array of ranges (one per parameter)
    /// - Computes Cartesian product: all combinations of values
    /// - Example: 2 params with 10 values each → 100 parameter vectors
    ///
    /// **Warning:** Exponential growth in batch size
    /// - N parameters with M values each → M^N vectors
    /// - Practical limit: ~1000 vectors (GPU memory constraints)
    ///
    /// - Parameter ranges: Array of value arrays (one per parameter)
    /// - Returns: Array of parameter vectors (Cartesian product)
    /// - Throws: ParameterError if ranges count doesn't match parameter count
    ///
    /// Example:
    /// ```swift
    /// // QAOA with 2 parameters: gamma and beta
    /// let qaoaAnsatz = ParameterizedQuantumCircuit(numQubits: 6)
    /// // ... build QAOA circuit with gamma and beta parameters
    ///
    /// let gammaRange = stride(from: 0.0, through: .pi, by: .pi / 10)  // 11 values
    /// let betaRange = stride(from: 0.0, through: .pi, by: .pi / 10)   // 11 values
    ///
    /// let parameterVectors = try qaoaAnsatz.generateGridSearchVectors(
    ///     ranges: [Array(gammaRange), Array(betaRange)]
    /// )
    /// // 11 × 11 = 121 parameter vectors
    ///
    /// // Bind all and evaluate
    /// let circuits = try qaoaAnsatz.bindBatch(parameterVectors: parameterVectors)
    /// let unitaries = try circuits.map { try CircuitUnitary.computeUnitary(circuit: $0) }
    ///
    /// let evaluator = await MPSBatchEvaluator()
    /// let energies = try await evaluator.evaluateExpectationValues(
    ///     unitaries: unitaries,
    ///     initialState: QuantumState(numQubits: 6),
    ///     hamiltonian: maxCutHamiltonian
    /// )
    ///
    /// // Find optimal parameters
    /// let minIndex = energies.enumerated().min(by: { $0.element < $1.element })!.offset
    /// let optimalGamma = Array(gammaRange)[minIndex / 11]
    /// let optimalBeta = Array(betaRange)[minIndex % 11]
    /// ```
    @_optimize(speed)
    @_eagerMove
    func generateGridSearchVectors(ranges: [[Double]]) throws -> [[Double]] {
        let numParams: Int = parameterCount()

        guard ranges.count == numParams else {
            throw ParameterError.gridSearchRangeMismatch(
                expected: numParams,
                got: ranges.count
            )
        }

        for (index, range) in ranges.enumerated() {
            guard !range.isEmpty else {
                throw ParameterError.emptyGridSearchRange(parameterIndex: index)
            }
        }

        let totalCombinations: Int = ranges.reduce(1) { $0 * $1.count }
        var parameterVectors: [[Double]] = []
        parameterVectors.reserveCapacity(totalCombinations)

        var indices: [Int] = Array(repeating: 0, count: numParams)

        while true {
            var currentVector: [Double] = []
            currentVector.reserveCapacity(numParams)

            for (paramIndex, rangeIndex) in indices.enumerated() {
                currentVector.append(ranges[paramIndex][rangeIndex])
            }

            parameterVectors.append(currentVector)

            var incrementPosition: Int = numParams - 1
            while incrementPosition >= 0 {
                indices[incrementPosition] += 1

                if indices[incrementPosition] < ranges[incrementPosition].count {
                    break
                }

                indices[incrementPosition] = 0
                incrementPosition -= 1
            }

            if incrementPosition < 0 {
                break
            }
        }

        return parameterVectors
    }
}

// MARK: - Extended Error Cases

public extension ParameterError {
    /// Parameter vector has wrong length in batch binding
    static func invalidVectorLengthInBatch(batchIndex _: Int, expected: Int, got: Int) -> ParameterError {
        .invalidVectorLength(expected: expected, got: got)
    }

    /// Grid search range count doesn't match parameter count
    static func gridSearchRangeMismatch(expected: Int, got: Int) -> ParameterError {
        .invalidVectorLength(expected: expected, got: got)
    }

    /// Grid search range is empty for parameter
    static func emptyGridSearchRange(parameterIndex: Int) -> ParameterError {
        .unboundParameter("parameter_\(parameterIndex)")
    }
}
