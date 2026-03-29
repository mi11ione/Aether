// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import GameplayKit

/// Instantaneous Quantum Polynomial (IQP) circuit sampling for computational advantage benchmarks.
///
/// Constructs and samples from IQP circuits of the form H^n D H^n |0>^n where D is a diagonal
/// unitary built from products of exp(i theta Z_i Z_j) two-qubit interactions and exp(i phi Z_k)
/// single-qubit phases. IQP circuits are classically hard to sample from under plausible
/// complexity-theoretic assumptions (collapse of the polynomial hierarchy), making them a
/// candidate for demonstrating quantum computational advantage.
///
/// Anti-concentration is verified via the collision probability C = sum_x p(x)^2 which measures
/// how spread out the output distribution is. For a distribution satisfying anti-concentration,
/// C <= c / 2^n for a small constant c, indicating that no single outcome dominates. The uniform
/// distribution achieves C = 1/2^n (the minimum), while a peaked distribution has C close to 1.
///
/// **Example:**
/// ```swift
/// let result = IQPSampling.evaluate(qubits: 4, shots: 10000, seed: 42)
/// let antiConcentrated = result.isAntiConcentrated
/// let collision = result.collisionProbability
/// ```
///
/// - SeeAlso: ``RandomCircuitSampling``
/// - SeeAlso: ``Measurement``
/// - SeeAlso: ``QuantumCircuit``
public enum IQPSampling {
    private static let antiConcentrationBound = 3.0

    /// Collision probability analysis result for anti-concentration verification.
    ///
    /// The collision probability C = sum_x (count(x)/S)^2 measures the concentration of the
    /// empirical output distribution from S samples. For anti-concentrated distributions
    /// arising from IQP circuits with random angles, C is bounded by a small multiple of
    /// 1/2^n. This struct reports the measured collision probability, the uniform reference
    /// threshold 1/2^n, and whether anti-concentration holds.
    ///
    /// **Example:**
    /// ```swift
    /// let result = IQPSampling.evaluate(qubits: 4, shots: 10000, seed: 42)
    /// let collision = result.collisionProbability
    /// let threshold = result.uniformThreshold
    /// ```
    ///
    /// - SeeAlso: ``IQPSampling/collisionProbability(outcomes:qubits:)``
    @frozen
    public struct CollisionResult: Sendable {
        /// Empirical collision probability: sum_x (count(x)/S)^2
        public let collisionProbability: Double

        /// Uniform distribution collision probability: 1/2^n
        public let uniformThreshold: Double

        /// Whether the distribution satisfies anti-concentration (collisionProbability <= bound/2^n)
        public let isAntiConcentrated: Bool
    }

    // MARK: - Circuit Generation

    /// Generates an IQP circuit with explicit single-qubit and two-qubit interaction angles.
    ///
    /// Constructs the circuit H^n D H^n where D = prod_k exp(i phi_k Z_k) prod_(i,j) exp(i theta_ij Z_i Z_j).
    /// The diagonal unitary D is decomposed into native gates: exp(i theta Z_i Z_j) becomes
    /// CNOT(i,j) Rz(-2 theta, j) CNOT(i,j), and exp(i phi Z_k) becomes Rz(-2 phi, k).
    ///
    /// **Example:**
    /// ```swift
    /// let singles = [0.5, 1.2, 0.8]
    /// let pairs: [(Int, Int, Double)] = [(0, 1, 0.7), (1, 2, 1.1)]
    /// let circuit = IQPSampling.generateCircuit(qubits: 3, singleAngles: singles, pairAngles: pairs)
    /// let state = circuit.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (minimum 2)
    ///   - singleAngles: Phase angles phi_k for single-qubit Z rotations, one per qubit
    ///   - pairAngles: Tuples (i, j, theta) for two-qubit ZZ interactions
    /// - Returns: IQP quantum circuit ready for execution
    /// - Precondition: `qubits` >= 2
    /// - Precondition: `singleAngles.count` == `qubits`
    /// - Precondition: All qubit indices in `pairAngles` are in 0..<qubits and i != j
    /// - Complexity: O(qubits + 3 * pairAngles.count) gate count
    ///
    /// - SeeAlso: ``generateCircuit(qubits:seed:)``
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func generateCircuit(
        qubits: Int,
        singleAngles: [Double],
        pairAngles: [(Int, Int, Double)],
    ) -> QuantumCircuit {
        ValidationUtilities.validateMinimumQubits(qubits, min: 2, algorithmName: "IQP sampling")
        ValidationUtilities.validateMemoryLimit(qubits)
        ValidationUtilities.validateArrayCount(singleAngles, expected: qubits, name: "singleAngles")

        for (qubitI, qubitJ, _) in pairAngles {
            ValidationUtilities.validateQubitIndex(qubitI, qubits: qubits)
            ValidationUtilities.validateQubitIndex(qubitJ, qubits: qubits)
            ValidationUtilities.validateUniqueQubits([qubitI, qubitJ])
        }

        var circuit = QuantumCircuit(qubits: qubits)

        for qubit in 0 ..< qubits {
            circuit.append(.hadamard, to: qubit)
        }

        for (qubitI, qubitJ, theta) in pairAngles {
            circuit.append(.cnot, to: [qubitI, qubitJ])
            circuit.append(.rotationZ(-2.0 * theta), to: qubitJ)
            circuit.append(.cnot, to: [qubitI, qubitJ])
        }

        for qubit in 0 ..< qubits {
            circuit.append(.rotationZ(-2.0 * singleAngles[qubit]), to: qubit)
        }

        for qubit in 0 ..< qubits {
            circuit.append(.hadamard, to: qubit)
        }

        return circuit
    }

    /// Generates an IQP circuit with random interaction angles.
    ///
    /// Constructs an IQP circuit H^n D H^n with uniformly random angles drawn from [0, 2 pi).
    /// All n(n-1)/2 qubit pairs receive ZZ interaction terms, and each qubit receives a single-Z
    /// phase rotation, producing a fully-connected IQP instance.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = IQPSampling.generateCircuit(qubits: 4, seed: 42)
    /// let state = circuit.execute()
    /// let probs = state.probabilities()
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (minimum 2)
    ///   - seed: Optional RNG seed for reproducible circuit generation
    /// - Returns: IQP quantum circuit with random angles
    /// - Precondition: `qubits` >= 2
    /// - Complexity: O(qubits^2) gate count from all-pairs ZZ interactions
    ///
    /// - SeeAlso: ``generateCircuit(qubits:singleAngles:pairAngles:)``
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func generateCircuit(qubits: Int, seed: UInt64? = nil) -> QuantumCircuit {
        ValidationUtilities.validateMinimumQubits(qubits, min: 2, algorithmName: "IQP sampling")
        ValidationUtilities.validateMemoryLimit(qubits)

        var rng: any RandomNumberGenerator = Measurement.createRNG(seed: seed)
        let twoPi = 2.0 * Double.pi

        let singleAngles = [Double](unsafeUninitializedCapacity: qubits) { buffer, count in
            for i in 0 ..< qubits {
                buffer[i] = Double.random(in: 0 ..< twoPi, using: &rng)
            }
            count = qubits
        }

        let pairCount = qubits * (qubits - 1) / 2
        var pairAngles = [(Int, Int, Double)]()
        pairAngles.reserveCapacity(pairCount)

        for i in 0 ..< qubits {
            for j in (i + 1) ..< qubits {
                let theta = Double.random(in: 0 ..< twoPi, using: &rng)
                pairAngles.append((i, j, theta))
            }
        }

        return generateCircuit(qubits: qubits, singleAngles: singleAngles, pairAngles: pairAngles)
    }

    // MARK: - Collision Probability

    /// Computes the empirical collision probability from measurement outcomes.
    ///
    /// The collision probability C = sum_x (f(x)/S)^2 where f(x) is the count of outcome x
    /// among S total samples. For a uniform distribution C = 1/2^n; for a delta distribution
    /// C = 1. Anti-concentration requires C <= c/2^n for a constant c close to 1, indicating
    /// the output is spread across exponentially many bitstrings.
    ///
    /// **Example:**
    /// ```swift
    /// let outcomes = Measurement.sample(circuit: iqpCircuit, shots: 10000, seed: 42)
    /// let result = IQPSampling.collisionProbability(outcomes: outcomes, qubits: 4)
    /// let collision = result.collisionProbability
    /// ```
    ///
    /// - Parameters:
    ///   - outcomes: Array of measurement outcomes (basis state indices)
    ///   - qubits: Number of qubits in the circuit
    /// - Returns: Collision result with empirical probability and anti-concentration check
    /// - Precondition: `outcomes` is non-empty
    /// - Precondition: `qubits` > 0
    /// - Complexity: O(outcomes.count) time, O(2^qubits) space for histogram
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public static func collisionProbability(outcomes: [Int], qubits: Int) -> CollisionResult {
        ValidationUtilities.validateNonEmpty(outcomes, name: "outcomes")
        ValidationUtilities.validatePositiveQubits(qubits)

        let counts = Measurement.histogram(outcomes: outcomes, qubits: qubits)
        let totalSamples = Double(outcomes.count)
        let totalSamplesSquared = totalSamples * totalSamples

        var collision = 0.0
        for count in counts {
            let frequency = Double(count)
            collision += frequency * frequency
        }
        collision /= totalSamplesSquared

        let stateSpaceSize = Double(1 << qubits)
        let uniformThreshold = 1.0 / stateSpaceSize

        return CollisionResult(
            collisionProbability: collision,
            uniformThreshold: uniformThreshold,
            isAntiConcentrated: collision <= antiConcentrationBound * uniformThreshold,
        )
    }

    // MARK: - Convenience Evaluation

    /// Generates a random IQP circuit, samples from it, and validates anti-concentration.
    ///
    /// Convenience method combining random IQP circuit generation, Born-rule sampling via
    /// statevector simulation, and collision probability computation with anti-concentration
    /// validation. Useful for quick experimental verification that IQP sampling produces
    /// sufficiently spread output distributions.
    ///
    /// **Example:**
    /// ```swift
    /// let result = IQPSampling.evaluate(qubits: 4, shots: 10000, seed: 42)
    /// let antiConcentrated = result.isAntiConcentrated
    /// let collision = result.collisionProbability
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (minimum 2)
    ///   - shots: Number of measurement samples
    ///   - seed: Optional RNG seed for reproducible results
    /// - Returns: Collision result with anti-concentration validation
    /// - Precondition: `qubits` >= 2
    /// - Precondition: `shots` > 0
    /// - Complexity: O(qubits^2 * 2^qubits + shots)
    ///
    /// - SeeAlso: ``generateCircuit(qubits:seed:)``
    /// - SeeAlso: ``collisionProbability(outcomes:qubits:)``
    @_optimize(speed)
    @_eagerMove
    public static func evaluate(qubits: Int, shots: Int, seed: UInt64? = nil) -> CollisionResult {
        ValidationUtilities.validatePositiveInt(shots, name: "shots")

        let circuit = generateCircuit(qubits: qubits, seed: seed)
        let outcomes = Measurement.sample(circuit: circuit, shots: shots, seed: seed)

        return collisionProbability(outcomes: outcomes, qubits: qubits)
    }
}
