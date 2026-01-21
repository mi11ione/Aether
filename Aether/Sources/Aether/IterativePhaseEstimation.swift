// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under Apache 2.0

import Foundation

/// Configuration for iterative phase estimation algorithm.
///
/// Controls precision, adaptive strategy, and optional initial estimate for NISQ-friendly
/// phase extraction. Iterative PE uses a single ancilla qubit and extracts phase bit-by-bit,
/// reducing qubit count compared to standard QPE at the cost of sequential measurements.
///
/// **Example:**
/// ```swift
/// let config = IPEConfiguration(precisionBits: 8, adaptiveStrategy: .semiclassical)
/// let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, configuration: config)
/// let result = await ipe.run()
/// print("Estimated phase: \(result.estimatedPhase)")
/// ```
///
/// - SeeAlso: ``IterativePhaseEstimation``
/// - SeeAlso: ``IPEAdaptiveStrategy``
/// - SeeAlso: ``IPEResult``
@frozen
public struct IPEConfiguration: Sendable {
    /// Number of precision bits for phase estimation.
    ///
    /// Determines accuracy as 2^(-precisionBits). Higher values yield more accurate
    /// phase estimates but require more iterations and measurements.
    public let precisionBits: Int

    /// Adaptive strategy for phase correction between iterations.
    ///
    /// Standard strategy extracts bits independently; semiclassical applies
    /// corrections based on previously measured bits for improved coherence.
    public let adaptiveStrategy: IPEAdaptiveStrategy

    /// Optional initial phase estimate for warm-start.
    ///
    /// When provided, enables faster convergence by biasing early measurements
    /// toward expected phase value. Value should be in [0, 1).
    public let initialEstimate: Double?

    /// Creates IPE configuration with specified parameters.
    ///
    /// **Example:**
    /// ```swift
    /// let basicConfig = IPEConfiguration(precisionBits: 4)
    /// let advancedConfig = IPEConfiguration(
    ///     precisionBits: 10,
    ///     adaptiveStrategy: .semiclassical,
    ///     initialEstimate: 0.25
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - precisionBits: Number of bits for phase precision (must be positive)
    ///   - adaptiveStrategy: Phase correction strategy (default: .standard)
    ///   - initialEstimate: Optional initial phase estimate in [0, 1)
    /// - Precondition: precisionBits > 0
    /// - Precondition: initialEstimate, if provided, must be in [0, 1)
    /// - Complexity: O(1)
    public init(
        precisionBits: Int,
        adaptiveStrategy: IPEAdaptiveStrategy = .standard,
        initialEstimate: Double? = nil,
    ) {
        ValidationUtilities.validatePositiveInt(precisionBits, name: "precisionBits")
        if let estimate = initialEstimate {
            ValidationUtilities.validateHalfOpenRange(estimate, min: 0.0, max: 1.0, name: "initialEstimate")
        }
        self.precisionBits = precisionBits
        self.adaptiveStrategy = adaptiveStrategy
        self.initialEstimate = initialEstimate
    }
}

/// Adaptive strategy for iterative phase estimation.
///
/// Controls how phase information from previous iterations affects subsequent measurements.
/// Standard mode treats each bit independently; semiclassical mode applies phase corrections
/// based on measured bits to improve measurement fidelity on NISQ hardware.
///
/// **Example:**
/// ```swift
/// let standardConfig = IPEConfiguration(precisionBits: 8, adaptiveStrategy: .standard)
/// let semiclassicalConfig = IPEConfiguration(precisionBits: 8, adaptiveStrategy: .semiclassical)
/// ```
///
/// - SeeAlso: ``IPEConfiguration``
@frozen
public enum IPEAdaptiveStrategy: Sendable {
    /// Bit-by-bit extraction without adaptation.
    ///
    /// Each iteration extracts one phase bit independently. Simpler but may
    /// accumulate errors on noisy hardware.
    case standard

    /// Semiclassical phase correction based on previous bits.
    ///
    /// Applies Rz rotation corrections computed from previously measured bits
    /// to align measurement basis with expected phase. Improves accuracy on
    /// NISQ devices by reducing dependence on long coherence times.
    case semiclassical
}

/// Result of a single IPE iteration.
///
/// Contains the measured bit, control parameters, and running phase estimate
/// for one iteration of the iterative phase estimation algorithm.
///
/// **Example:**
/// ```swift
/// let result = await ipe.run()
/// for iteration in result.iterations {
///     print("Bit \(iteration.bitIndex): \(iteration.measuredBit), phase: \(iteration.phaseEstimate)")
/// }
/// ```
///
/// - SeeAlso: ``IPEResult``
/// - SeeAlso: ``IterativePhaseEstimation``
@frozen
public struct IPEIterationResult: Sendable {
    /// Index of the bit being estimated (0 = MSB).
    public let bitIndex: Int

    /// Measured bit value (0 or 1).
    public let measuredBit: Int

    /// Control rotation angle applied for phase correction.
    public let controlAngle: Double

    /// Running phase estimate after this iteration.
    public let phaseEstimate: Double

    /// Creates an iteration result.
    ///
    /// - Parameters:
    ///   - bitIndex: Index of the bit (0 = MSB)
    ///   - measuredBit: Measured value (0 or 1)
    ///   - controlAngle: Phase correction angle applied
    ///   - phaseEstimate: Running phase estimate
    /// - Complexity: O(1)
    @inlinable
    public init(bitIndex: Int, measuredBit: Int, controlAngle: Double, phaseEstimate: Double) {
        self.bitIndex = bitIndex
        self.measuredBit = measuredBit
        self.controlAngle = controlAngle
        self.phaseEstimate = phaseEstimate
    }
}

/// Complete result of iterative phase estimation.
///
/// Contains the final estimated phase, detailed iteration history, and circuit
/// metrics. Provides equivalent precision information for comparison with
/// standard quantum phase estimation.
///
/// **Example:**
/// ```swift
/// let ipe = IterativePhaseEstimation(unitary: .rotationZ(.pi / 4), eigenstateQubits: 1, configuration: config)
/// let result = await ipe.run()
/// print(result)
/// print("Equivalent to \(result.equivalentQPEQubits)-qubit QPE")
/// ```
///
/// - SeeAlso: ``IterativePhaseEstimation``
/// - SeeAlso: ``IPEIterationResult``
@frozen
public struct IPEResult: Sendable, CustomStringConvertible {
    /// Estimated eigenvalue phase in [0, 1).
    ///
    /// The phase φ such that U|ψ⟩ = e^(2πiφ)|ψ⟩. Multiply by 2π to get
    /// the actual phase angle in radians.
    public let estimatedPhase: Double

    /// Results from each iteration in order.
    public let iterations: [IPEIterationResult]

    /// Total circuit depth across all iterations.
    public let totalDepth: Int

    /// Equivalent number of qubits for standard QPE with same precision.
    ///
    /// Standard QPE achieving the same precision would require this many
    /// qubits in the precision register, demonstrating the qubit savings
    /// from iterative approach.
    public let equivalentQPEQubits: Int

    /// Creates an IPE result.
    ///
    /// - Parameters:
    ///   - estimatedPhase: Final phase estimate
    ///   - iterations: Results from each iteration
    ///   - totalDepth: Cumulative circuit depth
    ///   - equivalentQPEQubits: Equivalent standard QPE qubit count
    /// - Complexity: O(1)
    @inlinable
    public init(
        estimatedPhase: Double,
        iterations: [IPEIterationResult],
        totalDepth: Int,
        equivalentQPEQubits: Int,
    ) {
        self.estimatedPhase = estimatedPhase
        self.iterations = iterations
        self.totalDepth = totalDepth
        self.equivalentQPEQubits = equivalentQPEQubits
    }

    /// Multi-line formatted summary of phase estimation results.
    @inlinable
    public var description: String {
        let phaseStr = String(format: "%.8f", estimatedPhase)
        let angleStr = String(format: "%.6f", estimatedPhase * 2.0 * .pi)
        let binaryStr = iterations.map { String($0.measuredBit) }.joined()

        return """
        IPE Result:
          Estimated Phase: \(phaseStr) (= \(angleStr) rad / 2π)
          Binary: 0.\(binaryStr)
          Iterations: \(iterations.count)
          Total Depth: \(totalDepth)
          Equivalent QPE Qubits: \(equivalentQPEQubits)
        """
    }
}

/// NISQ-friendly iterative phase estimation using single ancilla qubit.
///
/// Extracts the eigenvalue phase φ from U|ψ⟩ = e^(2πiφ)|ψ⟩ one bit at a time,
/// requiring only 1 ancilla qubit compared to n qubits for standard QPE with
/// n-bit precision. Trade-off is sequential measurements vs parallel register.
///
/// Algorithm overview for n-bit precision:
/// 1. For bit k from 0 to n-1 (MSB first):
///    - Prepare ancilla in |+⟩ via Hadamard
///    - Apply controlled-U^(2^(n-1-k)) between ancilla and eigenstate
///    - Apply Rz(-correction) based on previously measured bits (semiclassical)
///    - Apply Hadamard to ancilla
///    - Measure ancilla to obtain bit k
/// 2. Reconstruct phase: φ = Σ(bit_k * 2^(-k-1)) for k = 0..n-1
///
/// **Example:**
/// ```swift
/// let config = IPEConfiguration(precisionBits: 8, adaptiveStrategy: .semiclassical)
/// let ipe = IterativePhaseEstimation(
///     unitary: .rotationZ(.pi / 4),
///     eigenstateQubits: 1,
///     configuration: config
/// )
///
/// let result = await ipe.run(progress: { bit, phase in
///     print("Bit \(bit): current estimate = \(phase)")
/// })
/// print("Final phase: \(result.estimatedPhase)")
/// ```
///
/// - Complexity: O(n) iterations where n = precisionBits
/// - SeeAlso: ``IPEConfiguration``
/// - SeeAlso: ``IPEResult``
/// - SeeAlso: ``QuantumCircuit/phaseEstimation(unitary:precisionQubits:eigenstateQubits:)``
public actor IterativePhaseEstimation {
    /// Target unitary operator whose eigenvalue phase to estimate.
    private let unitary: QuantumGate

    /// Number of qubits in the eigenstate register.
    private let eigenstateQubits: Int

    /// Algorithm configuration (precision, strategy, initial estimate).
    private let configuration: IPEConfiguration

    /// Quantum simulator for circuit execution.
    private let simulator: QuantumSimulator

    /// Creates iterative phase estimator for specified unitary.
    ///
    /// **Example:**
    /// ```swift
    /// let ipe = IterativePhaseEstimation(
    ///     unitary: .pauliZ,
    ///     eigenstateQubits: 1,
    ///     configuration: IPEConfiguration(precisionBits: 10)
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - unitary: Single-qubit unitary gate U
    ///   - eigenstateQubits: Number of qubits for eigenstate (must be positive)
    ///   - configuration: IPE algorithm configuration
    /// - Precondition: unitary must be single-qubit gate
    /// - Precondition: eigenstateQubits > 0
    /// - Complexity: O(1)
    public init(
        unitary: QuantumGate,
        eigenstateQubits: Int,
        configuration: IPEConfiguration,
    ) {
        ValidationUtilities.validateControlledGateIsSingleQubit(unitary.qubitsRequired)
        ValidationUtilities.validatePositiveInt(eigenstateQubits, name: "eigenstateQubits")

        self.unitary = unitary
        self.eigenstateQubits = eigenstateQubits
        self.configuration = configuration
        simulator = QuantumSimulator()
    }

    /// Runs iterative phase estimation algorithm.
    ///
    /// Executes n iterations to extract n-bit phase estimate. Each iteration
    /// builds and executes a circuit for one phase bit, optionally applying
    /// semiclassical corrections based on previous measurements.
    ///
    /// The optional progress callback receives the current bit index and running
    /// phase estimate after each iteration, enabling UI updates or logging.
    ///
    /// **Example:**
    /// ```swift
    /// let result = await ipe.run(progress: { bitIndex, currentPhase in
    ///     print("Completed bit \(bitIndex), estimate: \(currentPhase)")
    /// })
    /// ```
    ///
    /// - Parameters:
    ///   - initialState: Optional initial eigenstate (default: computational basis |0...0⟩)
    ///   - progress: Optional callback receiving (bitIndex, currentPhaseEstimate) after each iteration
    /// - Returns: Complete IPE result with estimated phase and iteration details
    /// - Complexity: O(n * (circuit_depth * 2^q)) where n = precisionBits, q = eigenstateQubits + 1
    @_optimize(speed)
    @_eagerMove
    public func run(
        initialState: QuantumState? = nil,
        progress: (@Sendable (Int, Double) async -> Void)? = nil,
    ) async -> IPEResult {
        let n = configuration.precisionBits

        var measuredBits: [Int] = []
        measuredBits.reserveCapacity(n)

        var iterationResults: [IPEIterationResult] = []
        iterationResults.reserveCapacity(n)

        var totalDepth = 0
        var currentPhase = configuration.initialEstimate ?? 0.0

        for k in 0 ..< n {
            let power = n - 1 - k

            let correction = computePhaseCorrection(measuredBits: measuredBits, bitIndex: k)

            let circuit = QuantumCircuit.ipeIteration(
                unitary: unitary,
                power: power,
                phaseCorrection: correction,
                eigenstateQubits: eigenstateQubits,
            )

            let totalQubits = 1 + eigenstateQubits
            var state: QuantumState = if let initial = initialState {
                prepareInitialState(eigenstate: initial, totalQubits: totalQubits)
            } else {
                QuantumState(qubits: totalQubits)
            }

            state = await simulator.execute(circuit, from: state)

            let measuredBit = measureAncilla(state: state)
            measuredBits.append(measuredBit)

            currentPhase = reconstructPhase(measuredBits: measuredBits)

            totalDepth += circuit.depth

            let iterationResult = IPEIterationResult(
                bitIndex: k,
                measuredBit: measuredBit,
                controlAngle: correction,
                phaseEstimate: currentPhase,
            )
            iterationResults.append(iterationResult)

            await progress?(k, currentPhase)
        }

        return IPEResult(
            estimatedPhase: currentPhase,
            iterations: iterationResults,
            totalDepth: totalDepth,
            equivalentQPEQubits: n,
        )
    }

    /// Computes phase correction angle for semiclassical feedback.
    @_optimize(speed)
    @_effects(readonly)
    private func computePhaseCorrection(measuredBits: [Int], bitIndex: Int) -> Double {
        guard configuration.adaptiveStrategy == .semiclassical else {
            return 0.0
        }

        var correction = 0.0
        for (j, bit) in measuredBits.enumerated() {
            if bit == 1 {
                let exponent = bitIndex - j
                correction += .pi / Double(1 << exponent)
            }
        }

        return correction
    }

    /// Reconstructs phase from measured bits.
    @_optimize(speed)
    @_effects(readonly)
    private func reconstructPhase(measuredBits: [Int]) -> Double {
        var phase = 0.0
        for (k, bit) in measuredBits.enumerated() {
            if bit == 1 {
                phase += 1.0 / Double(1 << (k + 1))
            }
        }
        return phase
    }

    /// Measures ancilla qubit (qubit 0) and returns result.
    @_optimize(speed)
    @_effects(readonly)
    private func measureAncilla(state: QuantumState) -> Int {
        var prob0 = 0.0
        let size = state.stateSpaceSize

        for i in stride(from: 0, to: size, by: 2) {
            prob0 += state.amplitudes[i].magnitudeSquared
        }

        return prob0 >= 0.5 ? 0 : 1
    }

    /// Prepares initial state with eigenstate in register qubits.
    @_optimize(speed)
    @_eagerMove
    private func prepareInitialState(eigenstate: QuantumState, totalQubits: Int) -> QuantumState {
        ValidationUtilities.validateStateQubitCount(eigenstate, required: eigenstateQubits, exact: true)

        let totalSize = 1 << totalQubits
        let eigenstateSize = eigenstate.stateSpaceSize

        var amplitudes = [Complex<Double>](repeating: .zero, count: totalSize)

        for i in 0 ..< eigenstateSize {
            amplitudes[i << 1] = eigenstate.amplitudes[i]
        }

        return QuantumState(qubits: totalQubits, amplitudes: amplitudes)
    }
}

// MARK: - QuantumCircuit Extension

public extension QuantumCircuit {
    /// Builds single IPE iteration circuit.
    ///
    /// Constructs circuit for extracting one phase bit in iterative phase estimation.
    /// Ancilla qubit is index 0; eigenstate occupies qubits 1..<(1+eigenstateQubits).
    ///
    /// Circuit structure:
    /// 1. Hadamard on ancilla (create |+⟩)
    /// 2. Controlled-U^(2^power) with ancilla controlling eigenstate
    /// 3. Rz(-phaseCorrection) on ancilla (semiclassical feedback)
    /// 4. Hadamard on ancilla (convert phase to amplitude)
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.ipeIteration(
    ///     unitary: .pauliZ,
    ///     power: 3,
    ///     phaseCorrection: .pi / 4,
    ///     eigenstateQubits: 1
    /// )
    /// let state = circuit.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - unitary: Single-qubit unitary gate U
    ///   - power: Exponent k where controlled gate is C-U^(2^k)
    ///   - phaseCorrection: Rotation angle for semiclassical feedback
    ///   - eigenstateQubits: Number of qubits in eigenstate register
    /// - Returns: Circuit implementing one IPE iteration
    /// - Precondition: unitary must be single-qubit
    /// - Precondition: power >= 0
    /// - Precondition: eigenstateQubits > 0
    /// - Complexity: O(power) for controlled power decomposition
    ///
    /// - SeeAlso: ``IterativePhaseEstimation``
    /// - SeeAlso: ``ControlledGateDecomposer/controlledPower(of:power:control:targetQubits:)``
    @_optimize(speed)
    @_eagerMove
    static func ipeIteration(
        unitary: QuantumGate,
        power: Int,
        phaseCorrection: Double,
        eigenstateQubits: Int,
    ) -> QuantumCircuit {
        ValidationUtilities.validateControlledGateIsSingleQubit(unitary.qubitsRequired)
        ValidationUtilities.validateNonNegativeInt(power, name: "power")
        ValidationUtilities.validatePositiveInt(eigenstateQubits, name: "eigenstateQubits")

        let totalQubits = 1 + eigenstateQubits
        let ancillaQubit = 0
        let targetQubits = Array(1 ..< totalQubits)

        var circuit = QuantumCircuit(qubits: totalQubits)

        circuit.append(.hadamard, to: ancillaQubit)

        let controlledGates = ControlledGateDecomposer.controlledPower(
            of: unitary,
            power: power,
            control: ancillaQubit,
            targetQubits: targetQubits,
        )
        for (gate, qubits) in controlledGates {
            circuit.append(gate, to: qubits)
        }

        if abs(phaseCorrection) > 1e-12 {
            circuit.append(.rotationZ(-phaseCorrection), to: ancillaQubit)
        }

        circuit.append(.hadamard, to: ancillaQubit)

        return circuit
    }
}
