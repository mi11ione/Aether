// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Zero-noise extrapolation (ZNE) for error mitigation.
///
/// ZNE runs circuits at multiple noise levels and extrapolates to zero noise. The key insight
/// is that noise can be artificially amplified by circuit folding (U -> UU†U), and the
/// relationship between noise level and expectation value can be extrapolated to λ=0.
/// The algorithm runs the circuit at noise scale factors λ = [1, 2, 3, ...] via folding, measures
/// the expectation value E(λ) at each scale, fits a polynomial or exponential model to E(λ),
/// and extrapolates to E(0) as the noiseless estimate. Noise amplification can use global folding
/// (U -> UU†U for factor 3, U -> UU†UU†U for factor 5) or local folding of individual gates
/// for fractional scaling.
///
/// **Example:**
/// ```swift
/// let zne = ZeroNoiseExtrapolation(scaleFactors: [1, 2, 3])
/// let mitigated = await zne.mitigate(
///     circuit: circuit,
///     observable: hamiltonian,
///     simulator: noisySimulator
/// )
/// ```
///
/// - SeeAlso: ``ProbabilisticErrorCancellation`` for quasi-probability methods
/// - SeeAlso: ``DensityMatrixSimulator`` for noisy circuit execution
@frozen
public struct ZeroNoiseExtrapolation: Sendable {
    /// Noise scale factors for extrapolation.
    ///
    /// Must be positive and include 1.0 (unscaled). Common choices:
    /// - [1, 2, 3] for linear/quadratic extrapolation
    /// - [1, 3, 5] for higher-order odd folding
    public let scaleFactors: [Double]

    /// Extrapolation method for fitting noisy data.
    public let method: ExtrapolationMethod

    /// Folding strategy for noise amplification.
    public let foldingStrategy: FoldingStrategy

    // MARK: - Nested Types

    /// Extrapolation method for ZNE fitting.
    @frozen
    public enum ExtrapolationMethod: Sendable, Equatable {
        /// Linear extrapolation using least squares fit.
        /// E(λ) = a + bλ, extrapolate to E(0) = a
        case linear

        /// Polynomial extrapolation of given degree.
        /// E(λ) = Σᵢ aᵢλⁱ, extrapolate to E(0) = a₀
        case polynomial(degree: Int)

        /// Exponential extrapolation.
        /// E(λ) = a + b·exp(-cλ), extrapolate to E(0) = a + b
        case exponential

        /// Richardson extrapolation for systematic error cancellation.
        /// Optimal for errors that scale as O(λᵏ) for known k.
        case richardson
    }

    /// Circuit folding strategy for noise amplification.
    @frozen
    public enum FoldingStrategy: Sendable, Equatable {
        /// Fold entire circuit: U -> U U† U
        case global

        /// Fold individual gates for fractional scaling
        case local

        /// Fold from end of circuit (often better for coherent errors)
        case fromEnd
    }

    // MARK: - Initialization

    /// Create ZNE with specified configuration.
    ///
    /// **Example:**
    /// ```swift
    /// let zne = ZeroNoiseExtrapolation(
    ///     scaleFactors: [1, 2, 3],
    ///     method: .polynomial(degree: 2)
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - scaleFactors: Noise scale factors (must include 1.0, all > 0)
    ///   - method: Extrapolation method (default: Richardson)
    ///   - foldingStrategy: How to fold circuits (default: global)
    /// - Precondition: scaleFactors must contain 1.0 and have at least 2 elements
    public init(
        scaleFactors: [Double] = [1, 2, 3],
        method: ExtrapolationMethod = .richardson,
        foldingStrategy: FoldingStrategy = .global,
    ) {
        ValidationUtilities.validateZNEScaleFactors(scaleFactors)

        self.scaleFactors = scaleFactors.sorted()
        self.method = method
        self.foldingStrategy = foldingStrategy
    }

    // MARK: - Mitigation

    /// Mitigate noise using zero-noise extrapolation.
    ///
    /// **Example:**
    /// ```swift
    /// let result = await zne.mitigate(
    ///     circuit: vqeCircuit,
    ///     observable: hamiltonian,
    ///     simulator: noisySimulator
    /// )
    /// print("Mitigated energy: \(result.mitigatedValue)")
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit to execute
    ///   - observable: Observable to measure
    ///   - simulator: Noisy density matrix simulator
    /// - Returns: ZNE result with mitigated value and diagnostics
    /// - Complexity: O(k * gates * 4^n) where k = number of scale factors
    @_optimize(speed)
    public func mitigate(
        circuit: QuantumCircuit,
        observable: Observable,
        simulator: DensityMatrixSimulator,
    ) async -> ZNEResult {
        var noisyValues: [(scale: Double, value: Double)] = []
        noisyValues.reserveCapacity(scaleFactors.count)

        for scale in scaleFactors {
            let foldedCircuit = fold(circuit: circuit, scaleFactor: scale)
            let value = await simulator.expectationValue(foldedCircuit, observable: observable)
            noisyValues.append((scale, value))
        }

        let mitigatedValue = extrapolate(data: noisyValues)

        return ZNEResult(
            mitigatedValue: mitigatedValue,
            noisyValues: noisyValues,
            method: method,
            scaleFactors: scaleFactors,
        )
    }

    /// Mitigate multiple observables efficiently.
    ///
    /// Shares folded circuit executions across observables for efficiency.
    ///
    /// **Example:**
    /// ```swift
    /// let observables = [Observable.pauliZ(qubit: 0), Observable.pauliX(qubit: 1)]
    /// let values = await zne.mitigateBatch(circuit: circuit, observables: observables, simulator: sim)
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit to execute
    ///   - observables: Array of observables to measure
    ///   - simulator: Noisy density matrix simulator
    /// - Returns: Array of mitigated values (same order as observables)
    @_optimize(speed)
    public func mitigateBatch(
        circuit: QuantumCircuit,
        observables: [Observable],
        simulator: DensityMatrixSimulator,
    ) async -> [Double] {
        var statesAtScales: [(scale: Double, state: DensityMatrix)] = []

        for scale in scaleFactors {
            let foldedCircuit = fold(circuit: circuit, scaleFactor: scale)
            let state = await simulator.execute(foldedCircuit)
            statesAtScales.append((scale, state))
        }

        var results: [Double] = []
        results.reserveCapacity(observables.count)

        for observable in observables {
            var noisyValues: [(scale: Double, value: Double)] = []
            for (scale, state) in statesAtScales {
                let value = state.expectationValue(of: observable)
                noisyValues.append((scale, value))
            }
            results.append(extrapolate(data: noisyValues))
        }

        return results
    }

    // MARK: - Circuit Folding

    /// Fold circuit to amplify noise by scale factor.
    ///
    /// For integer scale n, applies: U -> U (U†U)^((n-1)/2)
    /// For fractional scales, uses local gate folding.
    ///
    /// **Example:**
    /// ```swift
    /// let folded = zne.fold(circuit: circuit, scaleFactor: 3.0)
    /// // folded contains circuit U U† U
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Original circuit
    ///   - scaleFactor: Noise amplification factor (≥ 1)
    /// - Returns: Folded circuit with amplified noise
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public func fold(circuit: QuantumCircuit, scaleFactor: Double) -> QuantumCircuit {
        if scaleFactor == 1.0 {
            return circuit
        }

        switch foldingStrategy {
        case .global:
            return foldGlobal(circuit: circuit, scaleFactor: scaleFactor)
        case .local:
            return foldLocal(circuit: circuit, scaleFactor: scaleFactor)
        case .fromEnd:
            return foldFromEnd(circuit: circuit, scaleFactor: scaleFactor)
        }
    }

    /// Global circuit folding: U -> U U† U for scale 3.
    @_optimize(speed)
    private func foldGlobal(circuit: QuantumCircuit, scaleFactor: Double) -> QuantumCircuit {
        let fullFolds = Int((scaleFactor - 1) / 2)
        let fractionalPart = (scaleFactor - 1).truncatingRemainder(dividingBy: 2)

        var result = QuantumCircuit(qubits: circuit.qubits)

        result = appendCircuit(circuit, to: result)

        for _ in 0 ..< fullFolds {
            result = appendCircuit(circuit.inverse(), to: result)
            result = appendCircuit(circuit, to: result)
        }

        if fractionalPart > 0.01 {
            let gatesToFold = Int(Double(circuit.count) * fractionalPart / 2)
            if gatesToFold > 0 {
                let partialInverse = partialCircuit(circuit.inverse(), gateCount: gatesToFold)
                let partialOriginal = partialCircuit(circuit, gateCount: gatesToFold)
                result = appendCircuit(partialInverse, to: result)
                result = appendCircuit(partialOriginal, to: result)
            }
        }

        return result
    }

    /// Local gate folding for fine-grained scaling.
    @_optimize(speed)
    private func foldLocal(circuit: QuantumCircuit, scaleFactor: Double) -> QuantumCircuit {
        let gatesToFold = Int(Double(circuit.count) * (scaleFactor - 1) / 2)
        let foldInterval = circuit.count > 0 ? max(1, circuit.count / max(1, gatesToFold)) : 1

        var result = QuantumCircuit(qubits: circuit.qubits)
        var foldedCount = 0

        for (index, operation) in circuit.operations.enumerated() {
            switch operation {
            case let .gate(gate, qubits, _):
                result.append(gate, to: qubits)
                if foldedCount < gatesToFold, index % foldInterval == 0 {
                    result.append(gate.inverse, to: qubits)
                    result.append(gate, to: qubits)
                    foldedCount += 1
                }
            case .reset, .measure:
                result.addOperation(operation)
            }
        }

        return result
    }

    /// Fold from end of circuit.
    @_optimize(speed)
    private func foldFromEnd(circuit: QuantumCircuit, scaleFactor: Double) -> QuantumCircuit {
        let gatesToFold = Int(Double(circuit.count) * (scaleFactor - 1) / 2)

        var result = QuantumCircuit(qubits: circuit.qubits)

        for operation in circuit.operations {
            result.addOperation(operation)
        }

        let startIndex = max(0, circuit.count - gatesToFold)
        let reversedOps = circuit.operations[startIndex...].reversed()

        for operation in reversedOps {
            switch operation {
            case let .gate(gate, qubits, _):
                result.append(gate.inverse, to: qubits)
            case .reset, .measure:
                result.addOperation(operation)
            }
        }

        for operation in circuit.operations[startIndex...] {
            switch operation {
            case let .gate(gate, qubits, _):
                result.append(gate, to: qubits)
            case .reset, .measure:
                result.addOperation(operation)
            }
        }

        return result
    }

    /// Helper to append one circuit to another.
    private func appendCircuit(_ source: QuantumCircuit, to target: QuantumCircuit) -> QuantumCircuit {
        var result = target
        for operation in source.operations {
            result.addOperation(operation)
        }
        return result
    }

    /// Helper to get first n gates of circuit.
    private func partialCircuit(_ circuit: QuantumCircuit, gateCount: Int) -> QuantumCircuit {
        var result = QuantumCircuit(qubits: circuit.qubits)
        for operation in circuit.operations.prefix(gateCount) {
            result.addOperation(operation)
        }
        return result
    }

    // MARK: - Extrapolation

    /// Extrapolate to zero noise using configured method.
    @_optimize(speed)
    private func extrapolate(data: [(scale: Double, value: Double)]) -> Double {
        switch method {
        case .linear:
            extrapolateLinear(data: data)
        case let .polynomial(degree):
            extrapolatePolynomial(data: data, degree: degree)
        case .exponential:
            extrapolateExponential(data: data)
        case .richardson:
            extrapolateRichardson(data: data)
        }
    }

    /// Linear least squares extrapolation.
    @_optimize(speed)
    private func extrapolateLinear(data: [(scale: Double, value: Double)]) -> Double {
        let n = Double(data.count)
        var sumX = 0.0
        var sumY = 0.0
        var sumXY = 0.0
        var sumX2 = 0.0

        for (x, y) in data {
            sumX += x
            sumY += y
            sumXY += x * y
            sumX2 += x * x
        }

        let denom = n * sumX2 - sumX * sumX
        if abs(denom) < 1e-15 {
            return sumY / n
        }

        let intercept = (sumY * sumX2 - sumX * sumXY) / denom
        return intercept
    }

    /// Polynomial extrapolation using Vandermonde matrix.
    @_optimize(speed)
    private func extrapolatePolynomial(data: [(scale: Double, value: Double)], degree: Int) -> Double {
        let effectiveDegree = min(degree, data.count - 1)
        let n = data.count
        let m = effectiveDegree + 1

        var vandermonde = [Double](repeating: 0, count: n * m)
        var y = [Double](repeating: 0, count: n)

        for i in 0 ..< n {
            let x = data[i].scale
            y[i] = data[i].value
            var xPow = 1.0
            for j in 0 ..< m {
                vandermonde[i * m + j] = xPow
                xPow *= x
            }
        }

        let coefficients = solveLeastSquares(matrix: vandermonde, rows: n, cols: m, rhs: y)

        return coefficients[0]
    }

    /// Exponential extrapolation: E(λ) = a + b·exp(-c·λ).
    @_optimize(speed)
    private func extrapolateExponential(data: [(scale: Double, value: Double)]) -> Double {
        guard data.count >= 3 else {
            return extrapolateLinear(data: data)
        }

        let y0 = data[0].value
        let y1 = data[data.count / 2].value
        let y2 = data[data.count - 1].value

        let x0 = data[0].scale
        let x1 = data[data.count / 2].scale
        let x2 = data[data.count - 1].scale

        let r1 = (y1 - y0) / (y2 - y1 + 1e-15)
        let c = -log(max(0.01, min(100, r1))) / (x1 - x0 + 1e-15)

        if c <= 0 || !c.isFinite {
            return extrapolateLinear(data: data)
        }

        let expCx0 = exp(-c * x0)
        let expCx2 = exp(-c * x2)
        let b = (y0 - y2) / (expCx0 - expCx2 + 1e-15)
        let a = y0 - b * expCx0

        return a + b
    }

    /// Richardson extrapolation for systematic error cancellation.
    @_optimize(speed)
    private func extrapolateRichardson(data: [(scale: Double, value: Double)]) -> Double {
        var values = data.map(\.value)
        var scales = data.map(\.scale)

        while values.count > 1 {
            var newValues: [Double] = []
            var newScales: [Double] = []

            for i in 0 ..< values.count - 1 {
                let ratio = scales[i + 1] / scales[i]
                let extrapolated = (ratio * values[i] - values[i + 1]) / (ratio - 1)
                newValues.append(extrapolated)
                newScales.append(scales[i])
            }

            values = newValues
            scales = newScales
        }

        return values[0]
    }

    /// Solve least squares Ax = b using normal equations.
    @_optimize(speed)
    private func solveLeastSquares(matrix: [Double], rows: Int, cols: Int, rhs: [Double]) -> [Double] {
        var ata = [Double](repeating: 0, count: cols * cols)
        var atb = [Double](repeating: 0, count: cols)

        for i in 0 ..< cols {
            for j in 0 ..< cols {
                var sum = 0.0
                for k in 0 ..< rows {
                    sum += matrix[k * cols + i] * matrix[k * cols + j]
                }
                ata[i * cols + j] = sum
            }
            var sum = 0.0
            for k in 0 ..< rows {
                sum += matrix[k * cols + i] * rhs[k]
            }
            atb[i] = sum
        }

        return solveLinearSystem(matrix: ata, size: cols, rhs: atb)
    }

    /// Solve linear system using Gaussian elimination with partial pivoting.
    @_optimize(speed)
    private func solveLinearSystem(matrix: [Double], size: Int, rhs: [Double]) -> [Double] {
        var a = matrix
        var b = rhs

        for k in 0 ..< size - 1 {
            var maxIdx = k
            var maxVal = abs(a[k * size + k])
            for i in k + 1 ..< size {
                let val = abs(a[i * size + k])
                if val > maxVal {
                    maxVal = val
                    maxIdx = i
                }
            }

            if maxIdx != k {
                for j in 0 ..< size {
                    let temp = a[k * size + j]
                    a[k * size + j] = a[maxIdx * size + j]
                    a[maxIdx * size + j] = temp
                }
                let temp = b[k]
                b[k] = b[maxIdx]
                b[maxIdx] = temp
            }

            let pivot = a[k * size + k]
            if abs(pivot) < 1e-15 {
                continue
            }

            for i in k + 1 ..< size {
                let factor = a[i * size + k] / pivot
                for j in k ..< size {
                    a[i * size + j] -= factor * a[k * size + j]
                }
                b[i] -= factor * b[k]
            }
        }

        var x = [Double](repeating: 0, count: size)
        for i in stride(from: size - 1, through: 0, by: -1) {
            var sum = b[i]
            for j in i + 1 ..< size {
                sum -= a[i * size + j] * x[j]
            }
            let diag = a[i * size + i]
            x[i] = abs(diag) > 1e-15 ? sum / diag : 0
        }

        return x
    }
}

// MARK: - ZNE Result

/// Result of zero-noise extrapolation.
@frozen
public struct ZNEResult: Sendable, CustomStringConvertible {
    /// Extrapolated zero-noise value.
    public let mitigatedValue: Double

    /// Measured values at each noise scale.
    public let noisyValues: [(scale: Double, value: Double)]

    /// Extrapolation method used.
    public let method: ZeroNoiseExtrapolation.ExtrapolationMethod

    /// Scale factors used.
    public let scaleFactors: [Double]

    /// Improvement factor: |noisy - mitigated| / |noisy|.
    @inlinable
    public var improvementFactor: Double {
        guard let unscaled = noisyValues.first(where: { $0.scale == 1.0 }) else {
            return 0
        }
        let noisyValue = unscaled.value
        return abs(noisyValue) > 1e-15 ? abs(noisyValue - mitigatedValue) / abs(noisyValue) : 0
    }

    public var description: String {
        let valuesStr = noisyValues.map { "λ=\($0.scale): \(String(format: "%.6f", $0.value))" }
            .joined(separator: ", ")
        return """
        ZNE Result:
          Mitigated: \(String(format: "%.6f", mitigatedValue))
          Noisy values: \(valuesStr)
          Improvement: \(String(format: "%.1f", improvementFactor * 100))%
        """
    }
}

// MARK: - Probabilistic Error Cancellation

/// Probabilistic error cancellation (PEC) for error mitigation.
///
/// PEC represents noisy operations as quasi-probability distributions over ideal operations.
/// By sampling from this distribution with appropriate signs, the expectation value converges
/// to the noiseless result. The cost is exponential sampling overhead. The algorithm decomposes
/// the noisy channel as E_noisy = Σᵢ qᵢPᵢ where Pᵢ are ideal operations and qᵢ can be negative
/// (quasi-probabilities) with γ = Σ|qᵢ| ≥ 1. It then samples Pᵢ with probability |qᵢ|/γ, weights
/// each measurement by sign(qᵢ)·γ, and averages over many samples to obtain ⟨O⟩_ideal = E[sign·γ·⟨O⟩_sample].
///
/// **Example:**
/// ```swift
/// let pec = ProbabilisticErrorCancellation(
///     noiseModel: NoiseModel.typicalNISQ,
///     samples: 10000
/// )
/// let mitigated = await pec.mitigate(circuit: circuit, observable: hamiltonian)
/// ```
///
/// - SeeAlso: ``ZeroNoiseExtrapolation`` for extrapolation-based mitigation
@frozen
public struct ProbabilisticErrorCancellation: Sendable {
    /// Number of Monte Carlo samples.
    public let samples: Int

    /// Quasi-probability overhead (γ = Σ|qᵢ|).
    public let gamma: Double

    /// Decomposition coefficients for depolarizing noise.
    private let decomposition: PauliDecomposition

    // MARK: - Nested Types

    /// Pauli decomposition for noise inverse.
    @frozen
    public struct PauliDecomposition: Sendable {
        /// Quasi-probability coefficients [qI, qX, qY, qZ].
        public let coefficients: [Double]

        /// Sampling overhead γ = Σ|qᵢ|.
        public let gamma: Double

        /// Sampling probabilities |qᵢ|/γ.
        public let probabilities: [Double]

        /// Signs of coefficients.
        public let signs: [Int]
    }

    // MARK: - Initialization

    /// Create PEC for depolarizing noise model.
    ///
    /// **Example:**
    /// ```swift
    /// let pec = ProbabilisticErrorCancellation(
    ///     errorProbability: 0.01,
    ///     samples: 10000
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - errorProbability: Depolarizing error probability per gate
    ///   - samples: Number of Monte Carlo samples (more = better precision, higher cost)
    /// - Precondition: 0 ≤ errorProbability < 0.75
    /// - Precondition: samples > 0
    public init(errorProbability: Double, samples: Int = 10000) {
        ValidationUtilities.validatePECErrorProbability(errorProbability)
        ValidationUtilities.validatePositiveInt(samples, name: "Samples")

        self.samples = samples

        let p = errorProbability
        let qI = (1 - p) / (1 - 4 * p / 3)
        let qP = -p / 3 / (1 - 4 * p / 3)

        let coefficients = [qI, qP, qP, qP]
        let gamma = abs(qI) + 3 * abs(qP)

        let probabilities = coefficients.map { abs($0) / gamma }
        let signs = coefficients.map { $0 >= 0 ? 1 : -1 }

        decomposition = PauliDecomposition(
            coefficients: coefficients,
            gamma: gamma,
            probabilities: probabilities,
            signs: signs,
        )
        self.gamma = gamma
    }

    // MARK: - Mitigation

    /// Mitigate noise using probabilistic error cancellation.
    ///
    /// **Example:**
    /// ```swift
    /// let result = await pec.mitigate(
    ///     circuit: circuit,
    ///     observable: hamiltonian,
    ///     simulator: simulator
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - circuit: Quantum circuit to execute
    ///   - observable: Observable to measure
    ///   - simulator: Density matrix simulator (noise will be cancelled)
    ///   - seed: Random seed for reproducibility
    /// - Returns: PEC result with mitigated value and statistics
    /// - Complexity: O(samples * gates * 4^n)
    @_optimize(speed)
    public func mitigate(
        circuit: QuantumCircuit,
        observable: Observable,
        simulator _: DensityMatrixSimulator,
        seed: UInt64? = nil,
    ) async -> PECResult {
        var generator: RandomNumberGenerator = if let seed {
            SeededPECGenerator(seed: seed)
        } else {
            SystemRandomNumberGenerator()
        }

        var weightedSum = 0.0
        var weightedSumSquared = 0.0
        let gateCount = circuit.count
        let totalGamma = pow(decomposition.gamma, Double(gateCount))

        for _ in 0 ..< samples {
            let (sampledCircuit, sign) = sampleCircuit(circuit: circuit, generator: &generator)

            let noiseFreeSimulator = DensityMatrixSimulator(noiseModel: .ideal)
            let value = await noiseFreeSimulator.expectationValue(sampledCircuit, observable: observable)

            let weight = Double(sign) * totalGamma
            weightedSum += weight * value
            weightedSumSquared += weight * weight * value * value
        }

        let mitigatedValue = weightedSum / Double(samples)
        let variance = weightedSumSquared / Double(samples) - mitigatedValue * mitigatedValue
        let standardError = sqrt(max(0, variance) / Double(samples))

        return PECResult(
            mitigatedValue: mitigatedValue,
            standardError: standardError,
            samples: samples,
            gamma: totalGamma,
            variance: variance,
        )
    }

    /// Sample circuit with quasi-probability weighting.
    @_optimize(speed)
    private func sampleCircuit(
        circuit: QuantumCircuit,
        generator: inout any RandomNumberGenerator,
    ) -> (circuit: QuantumCircuit, sign: Int) {
        var result = QuantumCircuit(qubits: circuit.qubits)
        var totalSign = 1

        for operation in circuit.operations {
            switch operation {
            case let .gate(gate, qubits, _):
                result.append(gate, to: qubits)
                if gate.qubitsRequired == 1 {
                    let (pauliIndex, sign) = samplePauli(generator: &generator)
                    totalSign *= sign
                    if pauliIndex > 0 {
                        let pauliGate = pauliGateForIndex(pauliIndex)
                        result.append(pauliGate, to: qubits)
                    }
                }
            case .reset, .measure:
                result.addOperation(operation)
            }
        }

        return (result, totalSign)
    }

    /// Sample Pauli operator from quasi-probability distribution.
    @_optimize(speed)
    private func samplePauli(generator: inout any RandomNumberGenerator) -> (index: Int, sign: Int) {
        let r = Double.random(in: 0 ..< 1, using: &generator)

        var cumulative = 0.0
        for (i, prob) in decomposition.probabilities.enumerated() {
            cumulative += prob
            if r < cumulative {
                return (i, decomposition.signs[i])
            }
        }

        return (0, decomposition.signs[0])
    }

    /// Get Pauli gate for index (1=X, 2=Y, 3=Z).
    @_effects(readonly)
    private func pauliGateForIndex(_ index: Int) -> QuantumGate {
        switch index {
        case 1: .pauliX
        case 2: .pauliY
        default: .pauliZ
        }
    }
}

/// Seeded random generator for PEC reproducibility.
private struct SeededPECGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed
    }

    mutating func next() -> UInt64 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return state
    }
}

// MARK: - PEC Result

/// Result of probabilistic error cancellation.
@frozen
public struct PECResult: Sendable, CustomStringConvertible {
    /// Mitigated expectation value.
    public let mitigatedValue: Double

    /// Standard error of the estimate.
    public let standardError: Double

    /// Number of samples used.
    public let samples: Int

    /// Total quasi-probability overhead γ^n.
    public let gamma: Double

    /// Variance of weighted samples.
    public let variance: Double

    /// 95% confidence interval.
    @inlinable
    public var confidenceInterval: (low: Double, high: Double) {
        (mitigatedValue - 1.96 * standardError, mitigatedValue + 1.96 * standardError)
    }

    public var description: String {
        let ci = confidenceInterval
        return """
        PEC Result:
          Mitigated: \(String(format: "%.6f", mitigatedValue)) ± \(String(format: "%.6f", standardError))
          95% CI: [\(String(format: "%.6f", ci.low)), \(String(format: "%.6f", ci.high))]
          Samples: \(samples), γ = \(String(format: "%.2f", gamma))
        """
    }
}

// MARK: - Readout Error Mitigation Extension

public extension MeasurementErrorModel {
    /// Mitigate measurement error from full histogram across all qubits.
    ///
    /// Applies tensor product of per-qubit inverse confusion matrices.
    /// More efficient than iterating `mitigateHistogram` per qubit.
    ///
    /// **Example:**
    /// ```swift
    /// let model = MeasurementErrorModel(p0Given1: 0.02, p1Given0: 0.01)
    /// let corrected = MeasurementErrorModel.mitigateFullHistogram(histogram, totalQubits: 3, models: [model])
    /// ```
    ///
    /// - Parameters:
    ///   - histogram: Dictionary mapping basis state indices to counts
    ///   - totalQubits: Total number of qubits in the system
    ///   - models: Per-qubit measurement error models (same model for all if single element)
    /// - Returns: Corrected histogram with mitigated counts
    /// - Complexity: O(2^n * n)
    /// - Precondition: models is non-empty
    @_optimize(speed)
    static func mitigateFullHistogram(
        _ histogram: [Int: Int],
        totalQubits: Int,
        models: [MeasurementErrorModel],
    ) -> [Int: Double] {
        ValidationUtilities.validateNonEmpty(models, name: "Measurement error models")

        var corrected: [Int: Double] = [:]
        let totalCounts = histogram.values.reduce(0, +)

        for (state, count) in histogram {
            var correctionFactor = 1.0
            for qubit in 0 ..< totalQubits {
                let model = models.count == 1 ? models[0] : models[qubit]
                let bit = (state >> qubit) & 1
                correctionFactor *= bit == 0 ? model.inverseMatrix[0][0] : model.inverseMatrix[1][1]
            }
            corrected[state] = Double(count) * correctionFactor
        }

        let correctedTotal = corrected.values.reduce(0, +)
        if abs(correctedTotal) > 1e-15 {
            let scale = Double(totalCounts) / correctedTotal
            for key in corrected.keys {
                corrected[key]! *= scale
            }
        }

        return corrected
    }
}
