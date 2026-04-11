// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Asynchronous density matrix simulator with configurable noise models.
///
/// Executes quantum circuits on density matrices with optional noise channels applied after
/// each gate. Supports all noise types: depolarizing, amplitude damping, phase damping,
/// bit flip, phase flip, and measurement errors. Essential for NISQ algorithm development
/// where noise significantly impacts algorithm performance.
///
/// Density matrices require O(4^n) memory, limiting simulation to approximately 14 qubits. For
/// noise-free simulation beyond 14 qubits, use ``QuantumSimulator`` with statevectors.
///
/// **Example:**
/// ```swift
/// // Create noisy simulator
/// let noise = NoiseModel.typicalNISQ
/// let simulator = DensityMatrixSimulator(noiseModel: noise)
///
/// // Execute circuit with noise
/// var circuit = QuantumCircuit(qubits: 2)
/// circuit.append(.hadamard, to: 0)
/// circuit.append(.cnot, to: [0, 1])
///
/// let result = await simulator.execute(circuit)
/// print(result.purity())  // < 1.0 due to noise
/// ```
///
/// - SeeAlso: ``DensityMatrix`` for mixed state representation
/// - SeeAlso: ``NoiseModel`` for noise configuration
/// - SeeAlso: ``QuantumSimulator`` for pure state simulation
public actor DensityMatrixSimulator {
    // MARK: - Properties

    /// Noise model applied during circuit execution.
    public let noiseModel: NoiseModel

    /// Progress tracking for current execution.
    public private(set) var progress: Progress

    // MARK: - Nested Types

    /// Execution progress information.
    @frozen
    public struct Progress: Sendable, Equatable {
        /// Number of gates executed so far.
        public let executed: Int

        /// Total number of gates in circuit.
        public let total: Int

        /// Execution progress as fraction in [0.0, 1.0].
        @inlinable
        public var percentage: Double {
            total > 0 ? Double(executed) / Double(total) : 0.0
        }

        /// Create progress snapshot with executed and total gate counts.
        @inlinable
        public init(executed: Int, total: Int) {
            self.executed = executed
            self.total = total
        }
    }

    // MARK: - Initialization

    /// Create density matrix simulator with specified noise model.
    ///
    /// **Example:**
    /// ```swift
    /// // Noise-free simulator
    /// let ideal = DensityMatrixSimulator()
    ///
    /// // Noisy simulator
    /// let noisy = DensityMatrixSimulator(
    ///     noiseModel: NoiseModel.typicalNISQ
    /// )
    /// ```
    ///
    /// - Parameter noiseModel: Noise configuration (default: ideal/no noise)
    public init(noiseModel: NoiseModel = .ideal) {
        self.noiseModel = noiseModel
        progress = Progress(executed: 0, total: 0)
    }

    // MARK: - Circuit Execution

    /// Execute quantum circuit on density matrix with noise.
    ///
    /// Applies each gate followed by the appropriate noise channel. Starts from ground state
    /// |0...0⟩⟨0...0| if no initial state provided.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    ///
    /// let result = await simulator.execute(circuit)
    /// ```
    ///
    /// - Parameter circuit: Quantum circuit to execute (must have concrete parameters)
    /// - Returns: Final density matrix after all gates and noise
    /// - Complexity: O(gates * 4^n)
    /// - Precondition: Circuit must have no symbolic parameters
    @_optimize(speed)
    @_eagerMove
    public func execute(_ circuit: QuantumCircuit) async -> DensityMatrix {
        let initial = DensityMatrix(qubits: circuit.qubits)
        return await execute(circuit, from: initial)
    }

    /// Execute quantum circuit on density matrix with custom initial state.
    ///
    /// When the noise model has idle noise configured, qubits not involved in each gate
    /// will experience T₁/T₂ decoherence proportional to the gate's execution time.
    ///
    /// **Example:**
    /// ```swift
    /// let initial = DensityMatrix.maximallyMixed(qubits: 2)
    /// let result = await simulator.execute(circuit, from: initial)
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit to execute
    ///   - initial: Initial density matrix state
    /// - Returns: Final density matrix after all gates and noise
    /// - Complexity: O(gates * 4^n)
    /// - Precondition: Circuit must have no symbolic parameters
    /// - Precondition: circuit.qubits == initial.qubits
    @_optimize(speed)
    @_eagerMove
    public func execute(
        _ circuit: QuantumCircuit,
        from initial: DensityMatrix,
    ) async -> DensityMatrix {
        ValidationUtilities.validateConcreteCircuit(circuit.parameterCount)
        ValidationUtilities.validateQubitCountsEqual(
            circuit.qubits,
            initial.qubits,
            name1: "Circuit qubits",
            name2: "initial state qubits",
        )

        let operations = circuit.operations
        let totalOps = operations.count
        progress = Progress(executed: 0, total: totalOps)

        var state = initial
        for (index, operation) in operations.enumerated() {
            state = state.applying(operation)

            if noiseModel.hasNoise, let gate = operation.gate {
                state = noiseModel.applyNoise(
                    after: gate,
                    targetQubits: operation.qubits,
                    to: state,
                    totalQubits: circuit.qubits,
                )
            }

            if index % 10 == 0 {
                progress = Progress(executed: index + 1, total: totalOps)
                await Task.yield()
            }
        }

        progress = Progress(executed: totalOps, total: totalOps)
        return state
    }

    /// Execute circuit and return expectation value of observable.
    ///
    /// Convenience method combining circuit execution with observable measurement.
    ///
    /// **Example:**
    /// ```swift
    /// let hamiltonian = Observable.pauliZ(qubit: 0)
    /// let energy = await simulator.expectationValue(circuit, observable: hamiltonian)
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit to execute
    ///   - observable: Observable to measure
    /// - Returns: Expectation value ⟨O⟩ = Tr(ρO)
    /// - Complexity: O(gates * 4^n + terms * 4^n)
    @_optimize(speed)
    public func expectationValue(
        _ circuit: QuantumCircuit,
        observable: Observable,
    ) async -> Double {
        let state = await execute(circuit)
        return state.expectationValue(of: observable)
    }

    /// Execute circuit from pure state initial condition.
    ///
    /// Convenience method that converts QuantumState to DensityMatrix before execution.
    ///
    /// **Example:**
    /// ```swift
    /// let initial = QuantumState(qubits: 2)
    /// let result = await simulator.execute(circuit, from: initial)
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit to execute
    ///   - pureState: Initial pure state |ψ⟩
    /// - Returns: Final density matrix (may be mixed due to noise)
    /// - Complexity: O(gates * 4^n)
    /// - Precondition: Circuit must have no symbolic parameters
    /// - Precondition: pureState.qubits == circuit.qubits
    @_optimize(speed)
    @_eagerMove
    public func execute(
        _ circuit: QuantumCircuit,
        from pureState: QuantumState,
    ) async -> DensityMatrix {
        let initial = DensityMatrix(pureState: pureState)
        return await execute(circuit, from: initial)
    }

    // MARK: - Measurement Sampling

    /// Sample measurement outcomes with noise.
    ///
    /// Executes circuit with noise, then samples from final probability distribution.
    /// Applies measurement error model if configured.
    ///
    /// **Example:**
    /// ```swift
    /// let outcomes = await simulator.sample(circuit, shots: 1000)
    /// // outcomes is array of 1000 basis state indices
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Circuit to execute
    ///   - shots: Number of measurement samples
    ///   - seed: Random seed for reproducibility (nil = random)
    /// - Returns: Array of measurement outcome indices
    /// - Complexity: O(gates * 4^n + shots * 2^n)
    /// - Precondition: shots > 0
    @_optimize(speed)
    public func sample(
        _ circuit: QuantumCircuit,
        shots: Int,
        seed: UInt64? = nil,
    ) async -> [Int] {
        ValidationUtilities.validatePositiveInt(shots, name: "Shots")

        let state = await execute(circuit)
        var probabilities = state.probabilities()

        if let measurementError = noiseModel.measurementError {
            probabilities = applyMeasurementError(
                probabilities: probabilities,
                error: measurementError,
                qubits: circuit.qubits,
            )
        }

        return sampleOutcomesFromDistribution(probabilities: probabilities, shots: shots, seed: seed)
    }

    /// Apply measurement error to probability distribution.
    @_optimize(speed)
    private func applyMeasurementError(
        probabilities: [Double],
        error: MeasurementErrorModel,
        qubits: Int,
    ) -> [Double] {
        applyMeasurementErrors(
            probabilities: probabilities,
            qubits: qubits,
            modelForQubit: { _ in error },
        )
    }

    // MARK: - Analysis Methods

    /// Compute fidelity between noisy and ideal execution.
    ///
    /// Fidelity measures how close the noisy state is to the ideal state:
    /// F(ρ, σ) = (Tr(√(√ρ σ √ρ)))² for general states.
    /// For pure target state |ψ⟩: F = ⟨ψ|ρ|ψ⟩.
    ///
    /// **Example:**
    /// ```swift
    /// let ideal = QuantumState(qubits: 2)
    /// let f = await simulator.fidelity(circuit, idealState: ideal)
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Circuit to execute
    ///   - idealState: Target pure state for comparison
    /// - Returns: Fidelity in [0, 1]
    /// - Complexity: O(gates * 4^n + 2^n)
    @_optimize(speed)
    public func fidelity(
        _ circuit: QuantumCircuit,
        idealState: QuantumState,
    ) async -> Double {
        let noisyState = await execute(circuit)
        return computePureStateFidelity(densityMatrix: noisyState, pureState: idealState)
    }

    /// Execute circuit multiple times and compute statistics.
    ///
    /// Useful for characterizing noise impact on algorithm performance.
    ///
    /// **Example:**
    /// ```swift
    /// let z = Observable.pauliZ(qubit: 0)
    /// let stats = await simulator.statisticalAnalysis(circuit, observable: z, repetitions: 100)
    /// print("Mean: \(stats.mean), Std: \(stats.std)")
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Circuit to execute
    ///   - observable: Observable to measure
    ///   - repetitions: Number of executions
    /// - Returns: Statistics (mean, std, min, max) of expectation values
    /// - Complexity: O(repetitions * gates * 4^n)
    /// - Precondition: repetitions > 0
    @_optimize(speed)
    public func statisticalAnalysis(
        _ circuit: QuantumCircuit,
        observable: Observable,
        repetitions: Int,
    ) async -> (mean: Double, std: Double, min: Double, max: Double) {
        ValidationUtilities.validatePositiveInt(repetitions, name: "Repetitions")

        var sum = 0.0
        var sumSq = 0.0
        var minVal = Double.infinity
        var maxVal = -Double.infinity

        for _ in 0 ..< repetitions {
            let v = await expectationValue(circuit, observable: observable)
            sum += v
            sumSq += v * v
            minVal = min(minVal, v)
            maxVal = max(maxVal, v)
        }

        let invN = 1.0 / Double(repetitions)
        let mean = sum * invN
        let variance = max(0, sumSq * invN - mean * mean)
        let std = sqrt(variance)

        return (mean, std, minVal, maxVal)
    }

    /// Compute total circuit execution time.
    ///
    /// Calculates the sum of gate durations assuming sequential execution.
    /// Useful for estimating decoherence impact and comparing circuit depths.
    ///
    /// **Example:**
    /// ```swift
    /// let time = simulator.circuitTime(circuit)
    /// print("Circuit duration: \(time) ns")
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Circuit to analyze
    ///   - timings: Gate timing model (default: IBM timings)
    /// - Returns: Total execution time in nanoseconds
    /// - Complexity: O(gates)
    @_effects(readonly)
    public nonisolated func circuitTime(
        _ circuit: QuantumCircuit,
        timings: GateTimingModel = .ibmDefault,
    ) -> Double {
        var totalTime = 0.0
        for operation in circuit.operations {
            if let gate = operation.gate {
                totalTime += timings.gateTime(for: gate.qubitsRequired)
            }
        }
        return totalTime
    }

    /// Estimate expected fidelity loss from decoherence.
    ///
    /// Provides a rough estimate based on T₁/T₂ times and circuit duration.
    /// Assumes exponential decay: F ≈ exp(-t/T₂).
    ///
    /// **Example:**
    /// ```swift
    /// let fidelity = simulator.estimateDecoherenceFidelity(
    ///     circuit, t1: 100_000, t2: 50_000
    /// )
    /// print("Expected fidelity: \(fidelity)")
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Circuit to analyze
    ///   - t1: T₁ time in nanoseconds
    ///   - t2: T₂ time in nanoseconds
    ///   - timings: Gate timing model
    /// - Returns: Estimated fidelity retention factor (0-1)
    /// - Complexity: O(gates)
    @_effects(readonly)
    public nonisolated func estimateDecoherenceFidelity(
        _ circuit: QuantumCircuit,
        t1: Double,
        t2: Double,
        timings: GateTimingModel = .ibmDefault,
    ) -> Double {
        let totalTime = circuitTime(circuit, timings: timings)
        let t1Decay = exp(-totalTime / t1)
        let t2Decay = exp(-totalTime / t2)
        return t1Decay * t2Decay
    }
}

// MARK: - Timing-Aware Density Matrix Simulator

/// Density matrix simulator with per-qubit noise from hardware profile.
///
/// Uses ``TimingAwareNoiseModel`` to apply qubit-specific error rates and
/// per-edge two-qubit gate errors. Provides more accurate NISQ simulation
/// than uniform noise models.
///
/// **Example:**
/// ```swift
/// let profile = HardwareNoiseProfile.ibmManila
/// let simulator = TimingAwareDensityMatrixSimulator(profile: profile)
///
/// let circuit = QuantumCircuit(qubits: 5)
/// // ... build circuit
///
/// let result = await simulator.execute(circuit)
/// ```
///
/// - SeeAlso: ``HardwareNoiseProfile`` for device characterization
/// - SeeAlso: ``DensityMatrixSimulator`` for uniform noise models
public actor TimingAwareDensityMatrixSimulator {
    /// Hardware profile with per-qubit parameters.
    public let profile: HardwareNoiseProfile

    /// Timing-aware noise model.
    private let noiseModel: TimingAwareNoiseModel

    /// Progress tracking.
    public private(set) var progress: DensityMatrixSimulator.Progress

    /// Create simulator from hardware profile.
    ///
    /// **Example:**
    /// ```swift
    /// let profile = HardwareNoiseProfile.ibmManila
    /// let simulator = TimingAwareDensityMatrixSimulator(profile: profile)
    /// ```
    ///
    /// - Parameter profile: Hardware noise profile
    public init(profile: HardwareNoiseProfile) {
        self.profile = profile
        noiseModel = TimingAwareNoiseModel(profile: profile)
        progress = DensityMatrixSimulator.Progress(executed: 0, total: 0)
    }

    /// Execute circuit with per-qubit noise and idle decoherence.
    ///
    /// **Example:**
    /// ```swift
    /// let result = await simulator.execute(circuit)
    /// print(result.purity())
    /// ```
    ///
    /// - Parameter circuit: Quantum circuit to execute
    /// - Returns: Final density matrix
    /// - Complexity: O(gates * 4^n * qubits)
    /// - Precondition: Circuit must have no symbolic parameters
    /// - Precondition: circuit.qubits ≤ profile.qubitCount
    @_optimize(speed)
    @_eagerMove
    public func execute(_ circuit: QuantumCircuit) async -> DensityMatrix {
        let initial = DensityMatrix(qubits: circuit.qubits)
        return await execute(circuit, from: initial)
    }

    /// Execute circuit from custom initial state.
    ///
    /// **Example:**
    /// ```swift
    /// let initial = DensityMatrix.maximallyMixed(qubits: 3)
    /// let result = await simulator.execute(circuit, from: initial)
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit to execute
    ///   - initial: Initial density matrix
    /// - Returns: Final density matrix
    /// - Precondition: Circuit must have no symbolic parameters
    /// - Precondition: circuit.qubits ≤ profile.qubitCount
    /// - Precondition: circuit.qubits == initial.qubits
    /// - Complexity: O(gates * 4^n * qubits)
    @_optimize(speed)
    @_eagerMove
    public func execute(
        _ circuit: QuantumCircuit,
        from initial: DensityMatrix,
    ) async -> DensityMatrix {
        ValidationUtilities.validateConcreteCircuit(circuit.parameterCount)
        ValidationUtilities.validateQubitCountWithinLimit(
            circuit.qubits,
            limit: profile.qubitCount,
            name: "Circuit qubits",
            limitName: "profile qubits",
        )
        ValidationUtilities.validateQubitCountsEqual(
            circuit.qubits,
            initial.qubits,
            name1: "Circuit qubits",
            name2: "initial state qubits",
        )

        let operations = circuit.operations
        let totalOps = operations.count
        progress = DensityMatrixSimulator.Progress(executed: 0, total: totalOps)

        var state = initial

        for (index, operation) in operations.enumerated() {
            state = state.applying(operation)

            if let gate = operation.gate {
                state = noiseModel.applyNoise(
                    after: gate,
                    targetQubits: operation.qubits,
                    to: state,
                )
            }

            if index % 10 == 0 {
                progress = DensityMatrixSimulator.Progress(executed: index + 1, total: totalOps)
                await Task.yield()
            }
        }

        progress = DensityMatrixSimulator.Progress(executed: totalOps, total: totalOps)
        return state
    }

    /// Execute circuit and return expectation value.
    ///
    /// **Example:**
    /// ```swift
    /// let z = Observable.pauliZ(qubit: 0)
    /// let energy = await simulator.expectationValue(circuit, observable: z)
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit to execute
    ///   - observable: Observable to measure
    /// - Returns: Expectation value Tr(ρO)
    /// - Complexity: O(gates * 4^n * qubits + terms * 4^n)
    @_optimize(speed)
    public func expectationValue(
        _ circuit: QuantumCircuit,
        observable: Observable,
    ) async -> Double {
        let state = await execute(circuit)
        return state.expectationValue(of: observable)
    }

    /// Sample measurement outcomes with per-qubit readout errors.
    ///
    /// **Example:**
    /// ```swift
    /// let outcomes = await simulator.sample(circuit, shots: 1000)
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Circuit to execute
    ///   - shots: Number of samples
    ///   - seed: Random seed
    /// - Returns: Array of measurement outcomes
    /// - Complexity: O(gates * 4^n * qubits + shots * log(2^n))
    /// - Precondition: shots > 0
    @_optimize(speed)
    public func sample(
        _ circuit: QuantumCircuit,
        shots: Int,
        seed: UInt64? = nil,
    ) async -> [Int] {
        ValidationUtilities.validatePositiveInt(shots, name: "Shots")

        let state = await execute(circuit)
        var probabilities = state.probabilities()

        let measurementModels = noiseModel.profile.measurementErrorModels()
        probabilities = applyPerQubitMeasurementError(
            probabilities: probabilities,
            models: measurementModels,
            qubits: circuit.qubits,
        )

        return sampleOutcomesFromDistribution(probabilities: probabilities, shots: shots, seed: seed)
    }

    /// Apply per-qubit measurement errors.
    @_optimize(speed)
    private func applyPerQubitMeasurementError(
        probabilities: [Double],
        models: [MeasurementErrorModel],
        qubits: Int,
    ) -> [Double] {
        applyMeasurementErrors(
            probabilities: probabilities,
            qubits: qubits,
            modelForQubit: { models[$0] },
        )
    }

    /// Compute fidelity between noisy and ideal execution.
    ///
    /// For pure target state |ψ⟩: F = ⟨ψ|ρ|ψ⟩.
    ///
    /// **Example:**
    /// ```swift
    /// let ideal = QuantumState(qubits: 2)
    /// let f = await simulator.fidelity(circuit, idealState: ideal)
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Circuit to execute
    ///   - idealState: Target pure state for comparison
    /// - Returns: Fidelity in [0, 1]
    /// - Complexity: O(gates * 4^n + 4^n)
    @_optimize(speed)
    public func fidelity(
        _ circuit: QuantumCircuit,
        idealState: QuantumState,
    ) async -> Double {
        let noisyState = await execute(circuit)
        return computePureStateFidelity(densityMatrix: noisyState, pureState: idealState)
    }

    /// Compute total circuit execution time from hardware profile timings.
    ///
    /// **Example:**
    /// ```swift
    /// let time = simulator.circuitTime(circuit)
    /// print("Estimated time: \(time) ns")
    /// ```
    ///
    /// - Parameter circuit: Circuit to analyze
    /// - Returns: Total execution time in nanoseconds
    /// - Complexity: O(gates)
    @_effects(readonly)
    public nonisolated func circuitTime(_ circuit: QuantumCircuit) -> Double {
        var totalTime = 0.0
        for operation in circuit.operations {
            if let gate = operation.gate {
                totalTime += profile.gateTimings.gateTime(for: gate.qubitsRequired)
            }
        }
        return totalTime
    }
}

// MARK: - Shared Helpers

/// Compute fidelity F = ⟨ψ|ρ|ψ⟩ between density matrix and pure state.
@_optimize(speed)
private func computePureStateFidelity(
    densityMatrix: DensityMatrix,
    pureState: QuantumState,
) -> Double {
    var result = 0.0
    let dim = densityMatrix.dimension

    for i in 0 ..< dim {
        let psiI = pureState.amplitudes[i].conjugate
        var rowSum = Complex<Double>.zero
        for j in 0 ..< dim {
            rowSum = rowSum + densityMatrix[row: i, col: j] * pureState.amplitudes[j]
        }
        result += (psiI * rowSum).real
    }

    return max(0, min(1, result))
}

/// Apply measurement errors per qubit using provided model lookup.
@_optimize(speed)
private func applyMeasurementErrors(
    probabilities: [Double],
    qubits: Int,
    modelForQubit: (Int) -> MeasurementErrorModel,
) -> [Double] {
    let dim = probabilities.count
    var noisyProbs = probabilities
    var newProbs = [Double](repeating: 0, count: dim)

    for qubit in 0 ..< qubits {
        let model = modelForQubit(qubit)
        let mask = 1 << qubit

        newProbs.withUnsafeMutableBufferPointer { buf in
            // Safety: buf.baseAddress is non-nil since dim > 0
            buf.baseAddress!.initialize(repeating: 0, count: dim)
        }

        for i in 0 ..< dim {
            let bit = (i >> qubit) & 1
            if bit == 0 {
                let partner = i | mask
                let p0 = noisyProbs[i]
                let p1 = noisyProbs[partner]
                let (noisyP0, noisyP1) = model.apply(to: (p0, p1))
                newProbs[i] = noisyP0
                newProbs[partner] = noisyP1
            }
        }

        swap(&noisyProbs, &newProbs)
    }

    return noisyProbs
}

/// Sample outcomes from probability distribution.
@_optimize(speed)
func sampleOutcomesFromDistribution(
    probabilities: [Double],
    shots: Int,
    seed: UInt64?,
) -> [Int] {
    var generator = Measurement.createRNG(seed: seed)

    var cdf = [Double](unsafeUninitializedCapacity: probabilities.count) {
        buffer, count in
        var cumulative = 0.0
        var compensation = 0.0
        for i in 0 ..< probabilities.count {
            let y = max(0, probabilities[i]) - compensation
            let t = cumulative + y
            compensation = (t - cumulative) - y
            cumulative = t
            buffer[i] = cumulative
        }
        count = probabilities.count
    }

    let total = cdf[probabilities.count - 1]
    if total > 0 {
        let invTotal = 1.0 / total
        for i in 0 ..< cdf.count {
            cdf[i] *= invTotal
        }
    }

    let cdfCount = cdf.count
    return [Int](unsafeUninitializedCapacity: shots) { buffer, count in
        for i in 0 ..< shots {
            let r = Double.random(in: 0 ..< 1, using: &generator)
            var lo = 0
            var hi = cdfCount - 1
            while lo < hi {
                let mid = (lo + hi) >> 1
                if r < cdf[mid] {
                    hi = mid
                } else {
                    lo = mid + 1
                }
            }
            buffer[i] = lo
        }
        count = shots
    }
}
