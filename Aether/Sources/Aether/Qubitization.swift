// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Configuration for block encoding of Hamiltonians via LCU decomposition.
///
/// Encapsulates the normalized coefficients, ancilla qubit requirements, and normalization
/// parameters needed to construct the block-encoded Hamiltonian H/alpha. The subnormalization
/// factor accounts for the gap between the actual operator norm and the 1-norm bound.
///
/// - SeeAlso: ``BlockEncoding``
/// - SeeAlso: ``LCUDecomposition``
///
/// **Example:**
/// ```swift
/// let config = BlockEncodingConfiguration(
///     normalizedCoefficients: [0.5, 0.3, 0.2],
///     ancillaQubits: 2,
///     oneNorm: 1.0,
///     subnormalization: 1.0
/// )
/// print(config.ancillaQubits)
/// ```
@frozen
public struct BlockEncodingConfiguration: Sendable {
    /// Normalized coefficients alpha_i / alpha where alpha = sum(|alpha_j|).
    ///
    /// Each coefficient represents the probability weight for selecting the corresponding
    /// unitary in the PREPARE oracle superposition. Sum equals 1.0.
    public let normalizedCoefficients: [Double]

    /// Number of ancilla qubits required for the block encoding.
    ///
    /// Computed as ceil(log2(L)) where L is the number of terms in the LCU decomposition.
    public let ancillaQubits: Int

    /// 1-norm alpha = sum(|alpha_i|) of the Hamiltonian coefficients.
    ///
    /// Determines the normalization factor for the block-encoded Hamiltonian. The actual
    /// Hamiltonian is recovered by multiplying the block-encoded result by alpha.
    public let oneNorm: Double

    /// Subnormalization factor for the block encoding.
    ///
    /// Ratio between the operator norm and the 1-norm bound. For Pauli decompositions
    /// this is typically 1.0, but can be smaller for structured Hamiltonians.
    public let subnormalization: Double

    /// Creates a block encoding configuration with specified parameters.
    ///
    /// - Parameters:
    ///   - normalizedCoefficients: Probability weights summing to 1.0
    ///   - ancillaQubits: Number of ancilla qubits for PREPARE/SELECT
    ///   - oneNorm: 1-norm of original Hamiltonian coefficients
    ///   - subnormalization: Ratio of operator norm to 1-norm bound
    ///
    /// **Example:**
    /// ```swift
    /// let config = BlockEncodingConfiguration(
    ///     normalizedCoefficients: [0.6, 0.4],
    ///     ancillaQubits: 1,
    ///     oneNorm: 1.5,
    ///     subnormalization: 0.9
    /// )
    /// ```
    public init(
        normalizedCoefficients: [Double],
        ancillaQubits: Int,
        oneNorm: Double,
        subnormalization: Double,
    ) {
        self.normalizedCoefficients = normalizedCoefficients
        self.ancillaQubits = ancillaQubits
        self.oneNorm = oneNorm
        self.subnormalization = subnormalization
    }
}

/// Block encoding of a Hamiltonian for Qubitization.
///
/// Wraps an Observable Hamiltonian with its LCU decomposition and provides circuits for the
/// PREPARE and SELECT oracles. The block encoding embeds H/alpha in the top-left block of a
/// larger unitary operator, enabling optimal Hamiltonian simulation via quantum signal processing.
///
/// The PREPARE oracle maps |0⟩_a to a superposition encoding coefficient magnitudes,
/// SELECT applies controlled unitaries |i⟩_a|ψ⟩_s → |i⟩_aUᵢ|ψ⟩_s, and the combined
/// circuit (PREPARE†)(SELECT)(PREPARE) block-encodes H/alpha.
///
/// - SeeAlso: ``BlockEncodingConfiguration``
/// - SeeAlso: ``QubitizedWalkOperator``
/// - SeeAlso: ``LCU``
///
/// **Example:**
/// ```swift
/// let hamiltonian = Observable(terms: [
///     (0.5, PauliString(.z(0))),
///     (-0.3, PauliString(.x(1)))
/// ])
/// let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: 2)
/// let prepCircuit = encoding.prepareCircuit()
/// let selectCircuit = encoding.selectCircuit()
/// ```
@frozen
public struct BlockEncoding: Sendable {
    /// The Hamiltonian being block-encoded.
    public let hamiltonian: Observable

    /// Configuration parameters for the block encoding.
    public let configuration: BlockEncodingConfiguration

    /// Number of system qubits the Hamiltonian acts on.
    public let systemQubits: Int

    /// Total qubits required including ancillas.
    ///
    /// Equals systemQubits + configuration.ancillaQubits.
    @inlinable public var totalQubits: Int {
        systemQubits + configuration.ancillaQubits
    }

    /// LCU decomposition of the Hamiltonian.
    private let decomposition: LCUDecomposition

    /// Creates a block encoding for the given Hamiltonian.
    ///
    /// Performs LCU decomposition and computes the configuration parameters including
    /// normalized coefficients and ancilla qubit requirements.
    ///
    /// - Parameters:
    ///   - hamiltonian: Observable to block-encode
    ///   - systemQubits: Number of qubits in the system register
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
    /// let encoding = BlockEncoding(hamiltonian: H, systemQubits: 2)
    /// print(encoding.totalQubits)
    /// ```
    ///
    /// - Precondition: systemQubits > 0
    public init(hamiltonian: Observable, systemQubits: Int) {
        ValidationUtilities.validatePositiveQubits(systemQubits)

        self.hamiltonian = hamiltonian
        self.systemQubits = systemQubits
        decomposition = LCU.decompose(hamiltonian)

        configuration = BlockEncodingConfiguration(
            normalizedCoefficients: decomposition.normalizedCoefficients,
            ancillaQubits: decomposition.ancillaQubits,
            oneNorm: decomposition.oneNorm,
            subnormalization: 1.0,
        )
    }

    /// Builds the PREPARE oracle circuit.
    ///
    /// Creates a circuit that prepares the superposition state encoding coefficient magnitudes:
    /// |0⟩_a -> sum_i sqrt(|alpha_i|/alpha) |i⟩_a
    ///
    /// - Returns: Quantum circuit implementing the PREPARE oracle
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
    /// let encoding = BlockEncoding(hamiltonian: H, systemQubits: 2)
    /// let prepare = encoding.prepareCircuit()
    /// ```
    ///
    /// - Complexity: O(L) where L is the number of LCU terms
    @_optimize(speed)
    @_eagerMove
    public func prepareCircuit() -> QuantumCircuit {
        LCU.prepareCircuit(decomposition: decomposition, ancillaStart: systemQubits)
    }

    /// Builds the SELECT oracle circuit.
    ///
    /// Creates a circuit that applies controlled unitaries based on ancilla state:
    /// |i⟩_a |psi⟩_s -> |i⟩_a U_i |psi⟩_s
    ///
    /// - Returns: Quantum circuit implementing the SELECT oracle
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
    /// let encoding = BlockEncoding(hamiltonian: H, systemQubits: 2)
    /// let select = encoding.selectCircuit()
    /// ```
    ///
    /// - Complexity: O(L × n) where L is LCU terms and n is system qubits
    @_optimize(speed)
    @_eagerMove
    public func selectCircuit() -> QuantumCircuit {
        LCU.selectCircuit(
            decomposition: decomposition,
            ancillaStart: systemQubits,
        )
    }

    /// Builds the complete block encoding circuit.
    ///
    /// Constructs PREPARE^dagger * SELECT * PREPARE which embeds H/alpha in the (0,0) block.
    ///
    /// - Returns: Quantum circuit implementing the block-encoded Hamiltonian
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
    /// let encoding = BlockEncoding(hamiltonian: H, systemQubits: 2)
    /// let blockCircuit = encoding.blockEncodingCircuit()
    /// ```
    ///
    /// - Complexity: O(L × n) where L is LCU terms and n is system qubits
    @_optimize(speed)
    @_eagerMove
    public func blockEncodingCircuit() -> QuantumCircuit {
        LCU.blockEncodingCircuit(
            decomposition: decomposition,
            ancillaStart: systemQubits,
        )
    }
}

/// Target function for Quantum Signal Processing polynomial transformation.
///
/// Specifies the function to be approximated by QSP phase angles. Each target corresponds
/// to a different polynomial approximation with specific properties and applications.
///
/// - SeeAlso: ``QSPPhaseAngles``
/// - SeeAlso: ``QuantumSignalProcessing``
///
/// **Example:**
/// ```swift
/// let target = QSPPolynomialTarget.timeEvolution(time: 1.0)
/// let signTarget = QSPPolynomialTarget.signFunction(threshold: 0.5)
/// ```
@frozen
public enum QSPPolynomialTarget: Sendable, Equatable {
    /// Time evolution operator e^(-iHt) approximation.
    ///
    /// Uses Jacobi-Anger expansion in Chebyshev polynomials with Bessel function coefficients.
    /// Polynomial degree scales as O(alpha * t + log(1/epsilon)).
    case timeEvolution(time: Double)

    /// Sign function for eigenvalue threshold detection.
    ///
    /// Approximates sgn(x - threshold) for spectral gap amplification and ground state
    /// preparation. Useful for quantum linear algebra algorithms.
    case signFunction(threshold: Double)

    /// Matrix inversion (1/x) for quantum linear systems.
    ///
    /// Approximates 1/x with condition number kappa determining approximation quality.
    /// Core component of HHL algorithm and variants.
    case inverseFunction(condition: Double)

    /// Chebyshev polynomial T_n(x) of specified degree.
    ///
    /// Direct Chebyshev polynomial for testing and custom transformations.
    case chebyshev(degree: Int)

    /// Custom polynomial from explicit coefficients.
    ///
    /// Coefficients are in Chebyshev basis: sum_k c_k T_k(x).
    case custom(coefficients: [Double])
}

/// Quantum Signal Processing phase angles for polynomial transformation.
///
/// Encapsulates the computed phase angles phi that implement a polynomial transformation
/// P(x) on the block-encoded signal. The phases are applied alternating with signal
/// operator applications to build up the desired polynomial.
///
/// For a polynomial of degree d, exactly d+1 phase angles are required. The phases
/// satisfy specific symmetry constraints depending on whether d is even or odd.
///
/// - SeeAlso: ``QSPPolynomialTarget``
/// - SeeAlso: ``QuantumSignalProcessing``
///
/// **Example:**
/// ```swift
/// let phases = QSPPhaseAngles(
///     phases: [0.1, -0.2, 0.3],
///     polynomialDegree: 2,
///     targetFunction: .timeEvolution(time: 1.0),
///     approximationError: 1e-6
/// )
/// print(phases.phases.count)
/// ```
@frozen
public struct QSPPhaseAngles: Sendable {
    /// Phase angles phi_0, phi_1, ..., phi_d for degree-d polynomial.
    ///
    /// Applied alternating with signal operator: e^(i*phi_0*Z) W e^(i*phi_1*Z) W ... e^(i*phi_d*Z)
    public let phases: [Double]

    /// Degree of the implemented polynomial transformation.
    public let polynomialDegree: Int

    /// Target function being approximated.
    public let targetFunction: QSPPolynomialTarget

    /// Maximum approximation error |P(x) - f(x)| over x in [-1, 1].
    public let approximationError: Double

    /// Creates QSP phase angles with specified parameters.
    ///
    /// - Parameters:
    ///   - phases: Array of phase angles
    ///   - polynomialDegree: Degree of the polynomial
    ///   - targetFunction: Function being approximated
    ///   - approximationError: Maximum pointwise error
    ///
    /// **Example:**
    /// ```swift
    /// let angles = QSPPhaseAngles(
    ///     phases: [.pi/4, -.pi/4],
    ///     polynomialDegree: 1,
    ///     targetFunction: .chebyshev(degree: 1),
    ///     approximationError: 0.0
    /// )
    /// ```
    public init(
        phases: [Double],
        polynomialDegree: Int,
        targetFunction: QSPPolynomialTarget,
        approximationError: Double,
    ) {
        self.phases = phases
        self.polynomialDegree = polynomialDegree
        self.targetFunction = targetFunction
        self.approximationError = approximationError
    }
}

/// Quantum Signal Processing for polynomial transformations of block-encoded operators.
///
/// Provides algorithms to compute QSP phase angles and build circuits that implement
/// polynomial transformations P(H/alpha) on block-encoded Hamiltonians. Combined with
/// Qubitization, enables optimal query complexity O(alpha*t + log(1/epsilon)) for
/// Hamiltonian simulation.
///
/// The QSP framework represents arbitrary bounded polynomials as:
/// P(x) = sum_k c_k T_k(x)
/// where T_k are Chebyshev polynomials. The phase angles are computed via optimization
/// or closed-form solutions for special cases.
///
/// - SeeAlso: ``QSPPhaseAngles``
/// - SeeAlso: ``QSPPolynomialTarget``
/// - SeeAlso: ``Qubitization``
///
/// **Example:**
/// ```swift
/// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
/// let encoding = BlockEncoding(hamiltonian: H, systemQubits: 2)
/// let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
/// let phases = QuantumSignalProcessing.computePhaseAngles(
///     for: .timeEvolution(time: 1.0),
///     degree: 10,
///     epsilon: 1e-6
/// )
/// let circuit = QuantumSignalProcessing.buildCircuit(
///     walkOperator: walkOp,
///     phaseAngles: phases,
///     signalQubit: 0
/// )
/// ```
public enum QuantumSignalProcessing {
    private static let besselNearZeroThreshold = 1e-10
    private static let besselConvergenceEpsilon = 1e-15
    private static let besselMaxIterations = 200
    private static let qspMaxIterations = 3000
    private static let qspConvergenceThreshold = 1e-13
    /// Computes QSP phase angles for the specified target function.
    ///
    /// Determines the optimal polynomial degree and computes phase angles that approximate
    /// the target function to within epsilon error. For time evolution, uses Jacobi-Anger
    /// expansion with Bessel function coefficients truncated when |J_k(t)| < epsilon.
    ///
    /// - Parameters:
    ///   - target: Target function to approximate
    ///   - degree: Maximum polynomial degree (actual may be lower)
    ///   - epsilon: Target approximation error
    /// - Returns: QSP phase angles implementing the polynomial approximation
    ///
    /// **Example:**
    /// ```swift
    /// let phases = QuantumSignalProcessing.computePhaseAngles(
    ///     for: .timeEvolution(time: 2.0),
    ///     degree: 20,
    ///     epsilon: 1e-8
    /// )
    /// print(phases.polynomialDegree)
    /// ```
    ///
    /// - Complexity: O(degree)
    /// - Precondition: degree > 0
    /// - Precondition: epsilon > 0
    @_optimize(speed)
    @_effects(readonly)
    public static func computePhaseAngles(
        for target: QSPPolynomialTarget,
        degree: Int,
        epsilon: Double,
    ) -> QSPPhaseAngles {
        ValidationUtilities.validatePositiveInt(degree, name: "degree")
        ValidationUtilities.validatePositiveDouble(epsilon, name: "epsilon")

        switch target {
        case let .timeEvolution(time):
            return computeTimeEvolutionPhases(time: time, maxDegree: degree, epsilon: epsilon)

        case let .signFunction(threshold):
            return computeSignFunctionPhases(threshold: threshold, maxDegree: degree, epsilon: epsilon)

        case let .inverseFunction(condition):
            return computeInverseFunctionPhases(condition: condition, maxDegree: degree, epsilon: epsilon)

        case let .chebyshev(chebyshevDegree):
            return computeChebyshevPhases(degree: chebyshevDegree)

        case let .custom(coefficients):
            return computeCustomPhases(coefficients: coefficients, epsilon: epsilon)
        }
    }

    /// Builds QSP circuit applying polynomial transformation via walk operator.
    ///
    /// Constructs the circuit that implements P(W) where W is the qubitized walk operator
    /// and P is the polynomial encoded by the phase angles. The circuit alternates phase
    /// rotations on the signal qubit with walk operator applications.
    ///
    /// Circuit structure: e^(i*phi_0*Z) W e^(i*phi_1*Z) W ... W e^(i*phi_d*Z)
    ///
    /// - Parameters:
    ///   - walkOperator: Qubitized walk operator W
    ///   - phaseAngles: QSP phase angles for the polynomial
    ///   - signalQubit: Qubit for phase rotations (typically ancilla qubit 0)
    /// - Returns: Quantum circuit implementing the polynomial transformation
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
    /// let encoding = BlockEncoding(hamiltonian: H, systemQubits: 2)
    /// let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
    /// let phases = QuantumSignalProcessing.computePhaseAngles(for: .timeEvolution(time: 1.0), degree: 10, epsilon: 1e-6)
    /// let circuit = QuantumSignalProcessing.buildCircuit(
    ///     walkOperator: walkOp,
    ///     phaseAngles: phases,
    ///     signalQubit: 4
    /// )
    /// ```
    ///
    /// - Complexity: O(degree x walkCircuitSize)
    /// - Precondition: signalQubit >= 0
    @_optimize(speed)
    @_eagerMove
    public static func buildCircuit(
        walkOperator: QubitizedWalkOperator,
        phaseAngles: QSPPhaseAngles,
        signalQubit: Int,
    ) -> QuantumCircuit {
        ValidationUtilities.validateNonNegativeInt(signalQubit, name: "signalQubit")

        let walkCircuit = walkOperator.buildWalkCircuit()
        let totalQubits = max(walkCircuit.qubits, signalQubit + 1)

        var circuit = QuantumCircuit(qubits: totalQubits)

        let phaseValues = phaseAngles.phases
        let numPhases = phaseValues.count

        if numPhases == 0 {
            return circuit
        }

        circuit.append(.rotationZ(2.0 * phaseValues[0]), to: signalQubit)

        for i in 1 ..< numPhases {
            for op in walkCircuit.operations {
                circuit.append(op)
            }

            circuit.append(.rotationZ(2.0 * phaseValues[i]), to: signalQubit)
        }

        return circuit
    }

    /// Builds controlled QSP circuit for phase estimation applications.
    ///
    /// Creates a controlled version of the QSP polynomial transformation where all
    /// walk operator applications are controlled by an additional qubit.
    ///
    /// - Parameters:
    ///   - walkOperator: Qubitized walk operator W
    ///   - phaseAngles: QSP phase angles for the polynomial
    ///   - signalQubit: Qubit for phase rotations
    ///   - controlQubit: Control qubit for conditional execution
    /// - Returns: Controlled QSP circuit
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
    /// let encoding = BlockEncoding(hamiltonian: H, systemQubits: 2)
    /// let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
    /// let phases = QuantumSignalProcessing.computePhaseAngles(for: .timeEvolution(time: 1.0), degree: 10, epsilon: 1e-6)
    /// let controlled = QuantumSignalProcessing.buildControlledCircuit(
    ///     walkOperator: walkOp,
    ///     phaseAngles: phases,
    ///     signalQubit: 4,
    ///     controlQubit: 0
    /// )
    /// ```
    ///
    /// - Complexity: O(degree x controlledWalkSize)
    /// - Precondition: signalQubit >= 0
    /// - Precondition: controlQubit >= 0
    @_optimize(speed)
    @_eagerMove
    public static func buildControlledCircuit(
        walkOperator: QubitizedWalkOperator,
        phaseAngles: QSPPhaseAngles,
        signalQubit: Int,
        controlQubit: Int,
    ) -> QuantumCircuit {
        ValidationUtilities.validateNonNegativeInt(signalQubit, name: "signalQubit")
        ValidationUtilities.validateNonNegativeInt(controlQubit, name: "controlQubit")

        let controlledWalk = walkOperator.buildControlledWalkCircuit(controlQubit: controlQubit)
        let totalQubits = max(controlledWalk.qubits, max(signalQubit, controlQubit) + 1)

        var circuit = QuantumCircuit(qubits: totalQubits)

        let phaseValues = phaseAngles.phases
        let numPhases = phaseValues.count

        if numPhases == 0 {
            return circuit
        }

        circuit.append(.controlledRotationZ(2.0 * phaseValues[0]), to: [controlQubit, signalQubit])

        for i in 1 ..< numPhases {
            for op in controlledWalk.operations {
                circuit.append(op)
            }

            circuit.append(.controlledRotationZ(2.0 * phaseValues[i]), to: [controlQubit, signalQubit])
        }

        return circuit
    }

    /// Computes QSP phases for time evolution via Jacobi-Anger expansion.
    @_optimize(speed)
    @_effects(readonly)
    private static func computeTimeEvolutionPhases(
        time: Double,
        maxDegree: Int,
        epsilon: Double,
    ) -> QSPPhaseAngles {
        let absTime = abs(time)

        var effectiveDegree = 1
        for k in 1 ... maxDegree {
            let besselApprox = approximateBessel(order: k, argument: absTime)
            if abs(besselApprox) < epsilon {
                effectiveDegree = k - 1
                break
            }
            effectiveDegree = k
        }

        effectiveDegree = max(2, effectiveDegree)
        if effectiveDegree % 2 != 0 { effectiveDegree += 1 }

        var chebyshevCoeffs = [Double](unsafeUninitializedCapacity: effectiveDegree + 1) {
            buffer, count in
            buffer.initialize(repeating: 0.0)
            count = effectiveDegree + 1
        }

        chebyshevCoeffs[0] = approximateBessel(order: 0, argument: absTime)
        for m in 1 ... effectiveDegree / 2 {
            let sign = m % 2 == 0 ? 1.0 : -1.0
            chebyshevCoeffs[2 * m] = sign * 2.0 * approximateBessel(order: 2 * m, argument: absTime)
        }

        let phases = chebyshevToQSPPhases(coefficients: chebyshevCoeffs, targetDegree: effectiveDegree)

        var adjustedPhases = phases
        if time < 0 {
            for i in 0 ..< adjustedPhases.count {
                adjustedPhases[i] = -adjustedPhases[i]
            }
        }

        return QSPPhaseAngles(
            phases: adjustedPhases,
            polynomialDegree: effectiveDegree,
            targetFunction: .timeEvolution(time: time),
            approximationError: epsilon,
        )
    }

    /// Computes QSP phases for sign function approximation.
    @_optimize(speed)
    @_effects(readonly)
    private static func computeSignFunctionPhases(
        threshold: Double,
        maxDegree: Int,
        epsilon: Double,
    ) -> QSPPhaseAngles {
        let effectiveDegree = min(maxDegree, Int(ceil(log(2.0 / epsilon) / (1.0 - abs(threshold) + 0.01))))

        var chebyshevCoeffs = [Double](unsafeUninitializedCapacity: effectiveDegree + 1) {
            buffer, count in
            buffer.initialize(repeating: 0.0)
            count = effectiveDegree + 1
        }

        for k in stride(from: 1, through: effectiveDegree, by: 2) {
            let sign = ((k - 1) / 2) % 2 == 0 ? 1.0 : -1.0
            chebyshevCoeffs[k] = sign * 4.0 / (Double.pi * Double(k))
        }

        let phases = chebyshevToQSPPhases(coefficients: chebyshevCoeffs, targetDegree: effectiveDegree)

        return QSPPhaseAngles(
            phases: phases,
            polynomialDegree: effectiveDegree,
            targetFunction: .signFunction(threshold: threshold),
            approximationError: epsilon,
        )
    }

    /// Computes QSP phases for matrix inversion approximation.
    @_optimize(speed)
    @_effects(readonly)
    private static func computeInverseFunctionPhases(
        condition: Double,
        maxDegree: Int,
        epsilon: Double,
    ) -> QSPPhaseAngles {
        let kappa = max(1.0, condition)
        let effectiveDegree = min(maxDegree, Int(ceil(kappa * log(2.0 * kappa / epsilon))))
        let delta = 1.0 / kappa

        var chebyshevCoeffs = [Double](unsafeUninitializedCapacity: effectiveDegree + 1) {
            buffer, count in
            buffer.initialize(repeating: 0.0)
            count = effectiveDegree + 1
        }

        let numNodes = max(4 * effectiveDegree, 64)
        let invNodes = 2.0 / Double(numNodes)

        for k in stride(from: 1, through: effectiveDegree, by: 2) {
            var ck = 0.0
            for j in 0 ..< numNodes {
                let theta = Double.pi * (Double(j) + 0.5) / Double(numNodes)
                let x = cos(theta)
                let fx = abs(x) >= delta ? 1.0 / (kappa * x) : kappa * x
                ck += fx * cos(Double(k) * theta)
            }
            chebyshevCoeffs[k] = ck * invNodes
        }

        let phases = chebyshevToQSPPhases(coefficients: chebyshevCoeffs, targetDegree: effectiveDegree)

        return QSPPhaseAngles(
            phases: phases,
            polynomialDegree: effectiveDegree,
            targetFunction: .inverseFunction(condition: condition),
            approximationError: epsilon,
        )
    }

    /// Computes QSP phases for a Chebyshev polynomial of given degree.
    @_optimize(speed)
    @_effects(readonly)
    private static func computeChebyshevPhases(degree: Int) -> QSPPhaseAngles {
        var chebyshevCoeffs = [Double](unsafeUninitializedCapacity: degree + 1) {
            buffer, count in
            buffer.initialize(repeating: 0.0)
            count = degree + 1
        }
        chebyshevCoeffs[degree] = 1.0

        let phases = chebyshevToQSPPhases(coefficients: chebyshevCoeffs, targetDegree: degree)

        return QSPPhaseAngles(
            phases: phases,
            polynomialDegree: degree,
            targetFunction: .chebyshev(degree: degree),
            approximationError: 0.0,
        )
    }

    /// Computes QSP phases from custom Chebyshev coefficients.
    @_optimize(speed)
    @_effects(readonly)
    private static func computeCustomPhases(
        coefficients: [Double],
        epsilon: Double,
    ) -> QSPPhaseAngles {
        let degree = coefficients.count - 1

        let phases = chebyshevToQSPPhases(coefficients: coefficients, targetDegree: degree)

        return QSPPhaseAngles(
            phases: phases,
            polynomialDegree: degree,
            targetFunction: .custom(coefficients: coefficients),
            approximationError: epsilon,
        )
    }

    /// Converts Chebyshev coefficients to QSP phase angles via gradient optimization.
    @_optimize(speed)
    @_effects(readonly)
    private static func chebyshevToQSPPhases(
        coefficients: [Double],
        targetDegree: Int,
    ) -> [Double] {
        let numPhases = targetDegree + 1
        ValidationUtilities.validateArrayCount(coefficients, expected: numPhases, name: "coefficients")

        if numPhases == 1 {
            return [acos(min(1.0, max(-1.0, coefficients[0]))) / 2.0]
        }

        if numPhases == 2 {
            let c1 = min(1.0, max(-1.0, coefficients[1]))
            let halfAngle = acos(c1) / 2.0
            return [halfAngle, halfAngle]
        }

        let numSamples = max(2 * numPhases, 16)
        let invSamples = 1.0 / Double(numSamples)

        let sampleCosines = [Double](unsafeUninitializedCapacity: numSamples) {
            buffer, count in
            for j in 0 ..< numSamples {
                buffer[j] = cos(Double.pi * (2.0 * Double(j) + 1.0) / (2.0 * Double(numSamples)))
            }
            count = numSamples
        }

        let sampleSines = [Double](unsafeUninitializedCapacity: numSamples) {
            buffer, count in
            for j in 0 ..< numSamples {
                buffer[j] = sqrt(max(0.0, 1.0 - sampleCosines[j] * sampleCosines[j]))
            }
            count = numSamples
        }

        let targetValues = [Double](unsafeUninitializedCapacity: numSamples) {
            buffer, count in
            for j in 0 ..< numSamples {
                buffer[j] = evaluateChebyshevPolynomial(coefficients, at: sampleCosines[j])
            }
            count = numSamples
        }

        var phases = [Double](unsafeUninitializedCapacity: numPhases) {
            buffer, count in
            buffer.initialize(repeating: 0.0)
            count = numPhases
        }
        var gradient = [Double](unsafeUninitializedCapacity: numPhases) {
            buffer, count in
            buffer.initialize(repeating: 0.0)
            count = numPhases
        }
        var phaseEp = [Complex<Double>](unsafeUninitializedCapacity: numPhases) {
            buffer, count in
            buffer.initialize(repeating: .one)
            count = numPhases
        }
        var phaseEm = [Complex<Double>](unsafeUninitializedCapacity: numPhases) {
            buffer, count in
            buffer.initialize(repeating: .one)
            count = numPhases
        }
        var phaseDep = [Complex<Double>](unsafeUninitializedCapacity: numPhases) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = numPhases
        }
        var phaseDem = [Complex<Double>](unsafeUninitializedCapacity: numPhases) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = numPhases
        }
        var pf00 = [Complex<Double>](unsafeUninitializedCapacity: numPhases + 1) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = numPhases + 1
        }
        var pf01 = [Complex<Double>](unsafeUninitializedCapacity: numPhases + 1) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = numPhases + 1
        }

        var eta = 0.1
        var prevLoss = Double.infinity

        for _ in 0 ..< qspMaxIterations {
            for k in 0 ..< numPhases {
                gradient[k] = 0.0
            }
            var loss = 0.0

            for k in 0 ..< numPhases {
                let cp = cos(phases[k])
                let sp = sin(phases[k])
                phaseEp[k] = Complex(cp, -sp)
                phaseEm[k] = Complex(cp, sp)
                phaseDep[k] = Complex(-sp, -cp)
                phaseDem[k] = Complex(-sp, cp)
            }

            for j in 0 ..< numSamples {
                let ct = sampleCosines[j]
                let st = sampleSines[j]

                pf00[0] = .one
                pf01[0] = .zero

                for k in 0 ..< numPhases {
                    let ep = phaseEp[k]
                    let em = phaseEm[k]
                    let a00: Complex<Double>
                    let a01: Complex<Double>
                    let a10: Complex<Double>
                    let a11: Complex<Double>
                    if k == 0 {
                        a00 = ep; a01 = .zero; a10 = .zero; a11 = em
                    } else {
                        a00 = ep * ct; a01 = em * st; a10 = ep * -st; a11 = em * ct
                    }
                    pf00[k + 1] = pf00[k] * a00 + pf01[k] * a10
                    pf01[k + 1] = pf00[k] * a01 + pf01[k] * a11
                }

                let value = pf00[numPhases].real
                let residual = value - targetValues[j]
                loss += residual * residual

                var sf00: Complex<Double> = .one
                var sf10: Complex<Double> = .zero

                for k in stride(from: numPhases - 1, through: 0, by: -1) {
                    let dep = phaseDep[k]
                    let dem = phaseDem[k]
                    let da00: Complex<Double>
                    let da01: Complex<Double>
                    let da10: Complex<Double>
                    let da11: Complex<Double>
                    if k == 0 {
                        da00 = dep; da01 = .zero; da10 = .zero; da11 = dem
                    } else {
                        da00 = dep * ct; da01 = dem * st; da10 = dep * -st; da11 = dem * ct
                    }

                    let pd00 = pf00[k] * da00 + pf01[k] * da10
                    let pd01 = pf00[k] * da01 + pf01[k] * da11
                    let dm00 = pd00 * sf00 + pd01 * sf10
                    gradient[k] += 2.0 * residual * dm00.real

                    let ep = phaseEp[k]
                    let em = phaseEm[k]
                    let a00: Complex<Double>
                    let a01: Complex<Double>
                    let a10: Complex<Double>
                    let a11: Complex<Double>
                    if k == 0 {
                        a00 = ep; a01 = .zero; a10 = .zero; a11 = em
                    } else {
                        a00 = ep * ct; a01 = em * st; a10 = ep * -st; a11 = em * ct
                    }
                    let nsf00 = a00 * sf00 + a01 * sf10
                    let nsf10 = a10 * sf00 + a11 * sf10
                    sf00 = nsf00
                    sf10 = nsf10
                }
            }

            loss *= invSamples
            if loss < qspConvergenceThreshold { break }

            if loss > prevLoss { eta *= 0.5 }
            prevLoss = loss

            for k in 0 ..< numPhases {
                phases[k] -= eta * gradient[k] * invSamples
            }
        }

        return phases
    }

    /// Evaluates Chebyshev expansion via Clenshaw recurrence.
    @_optimize(speed)
    @_effects(readonly)
    @inline(__always)
    private static func evaluateChebyshevPolynomial(
        _ coefficients: [Double],
        at x: Double,
    ) -> Double {
        let n = coefficients.count
        if n <= 1 { return n == 0 ? 0.0 : coefficients[0] }

        var b1 = 0.0
        var b2 = 0.0
        let twox = 2.0 * x
        for k in stride(from: n - 1, through: 1, by: -1) {
            let temp = twox * b1 - b2 + coefficients[k]
            b2 = b1
            b1 = temp
        }
        return x * b1 - b2 + coefficients[0]
    }

    /// Approximates J_n(x) via power series or Hankel asymptotic expansion (DLMF 10.17).
    @_optimize(speed)
    @_effects(readonly)
    private static func approximateBessel(order: Int, argument: Double) -> Double {
        if argument < besselNearZeroThreshold {
            return order == 0 ? 1.0 : 0.0
        }

        if argument > Double(max(20, 2 * order)) {
            return besselAsymptotic(order: order, argument: argument)
        }

        let halfArg = argument / 2.0
        var term = 1.0
        if order > 0 {
            for k in 1 ... order {
                term *= halfArg / Double(k)
            }
        }

        var sum = term
        var currentTerm = term
        let argSquaredOver4 = -halfArg * halfArg

        for m in 1 ... besselMaxIterations {
            let reciprocal = 1.0 / (Double(m) * Double(m + order))
            currentTerm *= argSquaredOver4 * reciprocal
            sum += currentTerm
            if abs(currentTerm) < besselConvergenceEpsilon * abs(sum) {
                break
            }
        }

        return sum
    }

    /// Hankel asymptotic expansion of J_n(x) for large x (DLMF 10.17.3-4).
    @_optimize(speed)
    @_effects(readonly)
    private static func besselAsymptotic(order: Int, argument: Double) -> Double {
        let nu = Double(order)
        let mu = 4.0 * nu * nu
        let omega = argument - (nu * 0.5 + 0.25) * Double.pi

        var aPrev = 1.0
        var p = 1.0
        var q = 0.0
        let invArg = 1.0 / argument
        var invArgPow = 1.0

        for k in 1 ... 20 {
            let twoKMinus1 = Double(2 * k - 1)
            let aK = aPrev * (mu - twoKMinus1 * twoKMinus1) / (8.0 * Double(k))
            invArgPow *= invArg

            if k & 1 == 1 {
                let m = k >> 1
                let sign = (m & 1 == 0) ? 1.0 : -1.0
                q += sign * aK * invArgPow
            } else {
                let m = k >> 1
                let sign = (m & 1 == 0) ? 1.0 : -1.0
                p += sign * aK * invArgPow
            }

            if abs(aK * invArgPow) < 1e-16 * max(abs(p), abs(q), 1.0) { break }
            aPrev = aK
        }

        return sqrt(2.0 / (Double.pi * argument)) * (p * cos(omega) - q * sin(omega))
    }
}

/// Qubitized walk operator for Hamiltonian simulation.
///
/// Implements the walk operator W = R * SELECT * PREPARE^dagger where R = 2|0><0| - I is the
/// reflection on the ancilla register. The eigenvalues of W are e^(+/- i*arccos(lambda_j/alpha))
/// where lambda_j are eigenvalues of the Hamiltonian and alpha is the 1-norm.
///
/// The walk operator is the core component of Qubitization, replacing Trotter-based methods
/// with a single query per step, achieving optimal query complexity.
///
/// - SeeAlso: ``BlockEncoding``
/// - SeeAlso: ``Qubitization``
/// - SeeAlso: ``QuantumSignalProcessing``
///
/// **Example:**
/// ```swift
/// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
/// let encoding = BlockEncoding(hamiltonian: H, systemQubits: 2)
/// let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
/// let circuit = walkOp.buildWalkCircuit()
/// let controlledCircuit = walkOp.buildControlledWalkCircuit(controlQubit: 0)
/// ```
@frozen
public struct QubitizedWalkOperator: Sendable {
    /// Block encoding used to construct the walk operator.
    public let blockEncoding: BlockEncoding

    /// Ancilla qubit indices for the block encoding.
    public let ancillaQubits: [Int]

    /// System qubit indices for the Hamiltonian.
    public let systemQubits: [Int]

    /// Creates a qubitized walk operator from a block encoding.
    ///
    /// Initializes the walk operator with automatic qubit index assignment:
    /// system qubits are 0..<n and ancilla qubits are n..<(n+a).
    ///
    /// - Parameter blockEncoding: Block encoding of the Hamiltonian
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
    /// let encoding = BlockEncoding(hamiltonian: H, systemQubits: 2)
    /// let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
    /// print(walkOp.systemQubits)
    /// ```
    public init(blockEncoding: BlockEncoding) {
        self.blockEncoding = blockEncoding

        let numSystem = blockEncoding.systemQubits
        let numAncilla = blockEncoding.configuration.ancillaQubits

        systemQubits = Array(0 ..< numSystem)
        ancillaQubits = Array(numSystem ..< numSystem + numAncilla)
    }

    /// Creates a qubitized walk operator with explicit qubit assignments.
    ///
    /// - Parameters:
    ///   - blockEncoding: Block encoding of the Hamiltonian
    ///   - ancillaQubits: Explicit ancilla qubit indices
    ///   - systemQubits: Explicit system qubit indices
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
    /// let encoding = BlockEncoding(hamiltonian: H, systemQubits: 2)
    /// let walkOp = QubitizedWalkOperator(
    ///     blockEncoding: encoding,
    ///     ancillaQubits: [4, 5],
    ///     systemQubits: [0, 1, 2, 3]
    /// )
    /// ```
    public init(blockEncoding: BlockEncoding, ancillaQubits: [Int], systemQubits: [Int]) {
        self.blockEncoding = blockEncoding
        self.ancillaQubits = ancillaQubits
        self.systemQubits = systemQubits
    }

    /// Builds the walk operator circuit W = R * SELECT * PREPARE^dagger.
    ///
    /// Composes PREPARE inverse, SELECT controlled unitaries, and the ancilla
    /// reflection R = 2|0><0|_a ⊗ I_s - I.
    ///
    /// The reflection R is implemented as X^n * MCZ * X^n on the ancilla qubits.
    ///
    /// - Returns: Quantum circuit implementing the walk operator
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
    /// let encoding = BlockEncoding(hamiltonian: H, systemQubits: 2)
    /// let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
    /// let circuit = walkOp.buildWalkCircuit()
    /// ```
    ///
    /// - Complexity: O(L x n) where L is the number of LCU terms and n is the number of system qubits
    @_optimize(speed)
    @_eagerMove
    public func buildWalkCircuit() -> QuantumCircuit {
        let totalQubits = blockEncoding.totalQubits

        var circuit = QuantumCircuit(qubits: totalQubits)

        let prepareCircuit = blockEncoding.prepareCircuit()
        let prepareInverse = prepareCircuit.inversed()
        for op in prepareInverse.operations {
            circuit.append(op)
        }

        let selectCircuit = blockEncoding.selectCircuit()
        for op in selectCircuit.operations {
            circuit.append(op)
        }

        appendReflection(to: &circuit)

        return circuit
    }

    /// Builds controlled walk operator for phase estimation.
    ///
    /// Creates a version of the walk operator controlled by an additional qubit,
    /// essential for eigenvalue estimation via quantum phase estimation.
    ///
    /// - Parameter controlQubit: Control qubit index
    /// - Returns: Controlled walk operator circuit
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
    /// let encoding = BlockEncoding(hamiltonian: H, systemQubits: 2)
    /// let walkOp = QubitizedWalkOperator(blockEncoding: encoding)
    /// let controlled = walkOp.buildControlledWalkCircuit(controlQubit: 0)
    /// ```
    ///
    /// - Complexity: O(L x n) where L is the number of LCU terms
    /// - Precondition: controlQubit >= 0
    @_optimize(speed)
    @_eagerMove
    public func buildControlledWalkCircuit(controlQubit: Int) -> QuantumCircuit {
        ValidationUtilities.validateNonNegativeInt(controlQubit, name: "controlQubit")

        let totalQubits = max(blockEncoding.totalQubits, controlQubit + 1)

        var circuit = QuantumCircuit(qubits: totalQubits)

        let prepareCircuit = blockEncoding.prepareCircuit()
        let prepareInverse = prepareCircuit.inversed()
        for op in prepareInverse.operations {
            if case let .gate(g, qubits, _) = op {
                appendControlledGate(to: &circuit, gate: g, qubits: qubits, control: controlQubit)
            }
        }

        let selectCircuit = blockEncoding.selectCircuit()
        for op in selectCircuit.operations {
            if case let .gate(g, qubits, _) = op {
                appendControlledGate(to: &circuit, gate: g, qubits: qubits, control: controlQubit)
            }
        }

        appendControlledReflection(to: &circuit, controlQubit: controlQubit)

        return circuit
    }

    /// Appends the ancilla reflection R = 2|0><0| - I to the circuit.
    @_optimize(speed)
    private func appendReflection(to circuit: inout QuantumCircuit) {
        for qubit in ancillaQubits {
            circuit.append(.pauliX, to: qubit)
        }

        appendMultiControlledZ(to: &circuit, controlQubits: ancillaQubits)

        for qubit in ancillaQubits {
            circuit.append(.pauliX, to: qubit)
        }
    }

    /// Appends a controlled ancilla reflection to the circuit.
    @_optimize(speed)
    private func appendControlledReflection(to circuit: inout QuantumCircuit, controlQubit: Int) {
        circuit.append(.cnot, to: [controlQubit, ancillaQubits[0]])

        for qubit in ancillaQubits {
            circuit.append(.pauliX, to: qubit)
        }

        var allControls = [Int]()
        allControls.reserveCapacity(1 + ancillaQubits.count)
        allControls.append(controlQubit)
        allControls.append(contentsOf: ancillaQubits)
        appendMultiControlledZ(to: &circuit, controlQubits: allControls)

        for qubit in ancillaQubits {
            circuit.append(.pauliX, to: qubit)
        }

        circuit.append(.cnot, to: [controlQubit, ancillaQubits[0]])
    }

    /// Appends a multi-controlled Z gate decomposition to the circuit.
    @_optimize(speed)
    private func appendMultiControlledZ(to circuit: inout QuantumCircuit, controlQubits: [Int]) {
        let n = controlQubits.count

        if n == 1 {
            circuit.append(.pauliZ, to: controlQubits[0])
        } else if n == 2 {
            circuit.append(.cz, to: [controlQubits[0], controlQubits[1]])
        } else {
            let target = controlQubits[n - 1]
            let controls = Array(controlQubits.dropLast())

            circuit.append(.hadamard, to: target)

            let decomposition = ControlledGateDecomposer.toffoliLadder(
                gate: .pauliX,
                controls: controls,
                target: target,
            )
            for (gate, qubits) in decomposition {
                circuit.append(gate, to: qubits)
            }

            circuit.append(.hadamard, to: target)
        }
    }

    /// Appends a controlled version of a gate to the circuit.
    @_optimize(speed)
    private func appendControlledGate(
        to circuit: inout QuantumCircuit,
        gate: QuantumGate,
        qubits: [Int],
        control: Int,
    ) {
        if qubits.count == 1 {
            let decomposition = ControlledGateDecomposer.decompose(
                gate: gate,
                controls: [control],
                target: qubits[0],
            )
            for (g, q) in decomposition {
                circuit.append(g, to: q)
            }
        } else if qubits.count == 2 {
            switch gate {
            case .cnot:
                circuit.append(.toffoli, to: [control, qubits[0], qubits[1]])
            case .cz:
                circuit.append(.hadamard, to: qubits[1])
                circuit.append(.toffoli, to: [control, qubits[0], qubits[1]])
                circuit.append(.hadamard, to: qubits[1])
            default:
                for q in qubits {
                    circuit.append(.cnot, to: [control, q])
                }
            }
        } else {
            for q in qubits {
                circuit.append(.cnot, to: [control, q])
            }
        }
    }
}

/// Result from qubitized eigenvalue estimation.
///
/// Encapsulates the estimated Hamiltonian eigenvalue along with the walk operator phase,
/// confidence interval, and query complexity metrics. The eigenvalue is recovered from
/// the walk operator phase via lambda = alpha * cos(phase).
///
/// - SeeAlso: ``Qubitization``
/// - SeeAlso: ``QubitizedWalkOperator``
///
/// **Example:**
/// ```swift
/// let result = QubitizationEigenvalueResult(
///     eigenvalue: -1.5,
///     phase: 2.094,
///     oneNorm: 3.0,
///     confidenceInterval: (-1.6, -1.4),
///     walkOperatorCalls: 100
/// )
/// print(result.eigenvalue)
/// ```
@frozen
public struct QubitizationEigenvalueResult: Sendable {
    /// Estimated eigenvalue of the Hamiltonian.
    ///
    /// Computed as alpha * cos(phase) where alpha is the 1-norm and phase is the
    /// walk operator eigenvalue phase.
    public let eigenvalue: Double

    /// Phase of the walk operator eigenvalue e^(i*phase).
    ///
    /// Related to eigenvalue by eigenvalue = alpha * cos(phase).
    public let phase: Double

    /// 1-norm of the Hamiltonian used in block encoding.
    public let oneNorm: Double

    /// Confidence interval for the eigenvalue estimate.
    public let confidenceInterval: (lower: Double, upper: Double)

    /// Number of walk operator applications used.
    public let walkOperatorCalls: Int

    /// Creates an eigenvalue estimation result.
    ///
    /// - Parameters:
    ///   - eigenvalue: Estimated Hamiltonian eigenvalue
    ///   - phase: Walk operator phase
    ///   - oneNorm: Hamiltonian 1-norm
    ///   - confidenceInterval: Error bounds on eigenvalue
    ///   - walkOperatorCalls: Query complexity
    ///
    /// **Example:**
    /// ```swift
    /// let result = QubitizationEigenvalueResult(
    ///     eigenvalue: -0.5,
    ///     phase: 1.047,
    ///     oneNorm: 1.0,
    ///     confidenceInterval: (-0.55, -0.45),
    ///     walkOperatorCalls: 50
    /// )
    /// ```
    public init(
        eigenvalue: Double,
        phase: Double,
        oneNorm: Double,
        confidenceInterval: (lower: Double, upper: Double),
        walkOperatorCalls: Int,
    ) {
        self.eigenvalue = eigenvalue
        self.phase = phase
        self.oneNorm = oneNorm
        self.confidenceInterval = confidenceInterval
        self.walkOperatorCalls = walkOperatorCalls
    }
}

/// Result from qubitized time evolution simulation.
///
/// Contains the evolved quantum state along with simulation parameters and complexity
/// analysis. Compares actual query count against the theoretical optimal bound of
/// O(alpha*t + log(1/epsilon)).
///
/// - SeeAlso: ``Qubitization``
/// - SeeAlso: ``QuantumSignalProcessing``
///
/// **Example:**
/// ```swift
/// let result = QubitizationEvolutionResult(
///     evolvedState: QuantumState(qubits: 2),
///     time: 1.0,
///     epsilon: 1e-6,
///     walkOperatorCalls: 15,
///     theoreticalBound: 20,
///     polynomialDegree: 14
/// )
/// print(result.walkOperatorCalls)
/// ```
@frozen
public struct QubitizationEvolutionResult: Sendable {
    /// Final quantum state after time evolution.
    public let evolvedState: QuantumState

    /// Simulation time t in e^(-iHt).
    public let time: Double

    /// Target approximation error for the simulation.
    public let epsilon: Double

    /// Actual number of walk operator applications.
    public let walkOperatorCalls: Int

    /// Theoretical optimal bound O(alpha*t + log(1/epsilon)).
    public let theoreticalBound: Int

    /// Degree of the QSP polynomial used.
    public let polynomialDegree: Int

    /// Creates a time evolution result.
    ///
    /// - Parameters:
    ///   - evolvedState: Final state after evolution
    ///   - time: Evolution time
    ///   - epsilon: Target error
    ///   - walkOperatorCalls: Actual query count
    ///   - theoreticalBound: Optimal bound
    ///   - polynomialDegree: QSP polynomial degree
    ///
    /// **Example:**
    /// ```swift
    /// let result = QubitizationEvolutionResult(
    ///     evolvedState: QuantumState(qubits: 2),
    ///     time: 2.0,
    ///     epsilon: 1e-8,
    ///     walkOperatorCalls: 30,
    ///     theoreticalBound: 35,
    ///     polynomialDegree: 29
    /// )
    /// ```
    public init(
        evolvedState: QuantumState,
        time: Double,
        epsilon: Double,
        walkOperatorCalls: Int,
        theoreticalBound: Int,
        polynomialDegree: Int,
    ) {
        self.evolvedState = evolvedState
        self.time = time
        self.epsilon = epsilon
        self.walkOperatorCalls = walkOperatorCalls
        self.theoreticalBound = theoreticalBound
        self.polynomialDegree = polynomialDegree
    }
}

/// Qubitization engine for optimal Hamiltonian simulation.
///
/// Implements the state-of-the-art Qubitization technique for Hamiltonian simulation with
/// optimal query complexity O(alpha*t + log(1/epsilon)) where alpha is the 1-norm, t is
/// evolution time, and epsilon is target error. Combines block encoding via LCU, the
/// qubitized walk operator, and Quantum Signal Processing.
///
/// Qubitization achieves optimal complexity by block-encoding H/alpha via LCU decomposition,
/// constructing a walk operator W whose eigenvalues are e^(±i·arccos(λ/α)), and using QSP
/// to implement polynomial approximations of e^(-it·arccos(x)). This replaces Trotter-based
/// methods that have polynomial overhead in precision.
///
/// - SeeAlso: ``BlockEncoding``
/// - SeeAlso: ``QubitizedWalkOperator``
/// - SeeAlso: ``QuantumSignalProcessing``
///
/// **Example:**
/// ```swift
/// let hamiltonian = Observable(terms: [
///     (0.5, PauliString(.z(0))),
///     (-0.3, PauliString(.x(1))),
///     (0.2, PauliString(.z(0), .z(1)))
/// ])
///
/// let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: 2)
///
/// let initial = QuantumState(qubits: 2)
/// let result = await qubitization.simulateEvolution(
///     initialState: initial,
///     time: 1.0,
///     epsilon: 1e-6
/// )
/// print(result.walkOperatorCalls)
/// ```
public actor Qubitization {
    /// Hamiltonian being simulated.
    public let hamiltonian: Observable

    /// Block encoding of the Hamiltonian.
    public let blockEncoding: BlockEncoding

    /// Qubitized walk operator for the Hamiltonian.
    public let walkOperator: QubitizedWalkOperator

    /// Number of system qubits.
    public let systemQubits: Int

    private static let amplitudeThreshold = 1e-15

    /// Creates a Qubitization engine for the given Hamiltonian.
    ///
    /// Constructs the block encoding and walk operator for optimal Hamiltonian simulation.
    ///
    /// - Parameters:
    ///   - hamiltonian: Observable representing the Hamiltonian as weighted Pauli strings
    ///   - systemQubits: Number of qubits in the system register
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
    /// let qubitization = Qubitization(hamiltonian: H, systemQubits: 2)
    /// ```
    ///
    /// - Precondition: systemQubits > 0
    public init(hamiltonian: Observable, systemQubits: Int) {
        ValidationUtilities.validatePositiveQubits(systemQubits)

        self.hamiltonian = hamiltonian
        self.systemQubits = systemQubits
        blockEncoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: systemQubits)
        walkOperator = QubitizedWalkOperator(blockEncoding: blockEncoding)
    }

    /// Simulates time evolution with optimal query complexity O(alpha*t + log(1/epsilon)).
    ///
    /// Implements e^(-iHt) by computing QSP phase angles that approximate the time evolution
    /// operator and applying the resulting polynomial transformation to the walk operator.
    ///
    /// The algorithm computes polynomial degree d = O(alpha*t + log(1/epsilon)), finds QSP
    /// phase angles via Jacobi-Anger expansion, and executes the QSP circuit with d walk
    /// operator applications.
    ///
    /// - Parameters:
    ///   - initialState: Initial quantum state to evolve (system qubits only)
    ///   - time: Evolution time t
    ///   - epsilon: Target approximation error
    /// - Returns: Evolution result with final state and complexity metrics
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
    /// let qubitization = Qubitization(hamiltonian: H, systemQubits: 2)
    /// let initial = QuantumState(qubits: 2)
    /// let result = await qubitization.simulateEvolution(
    ///     initialState: initial,
    ///     time: 2.0,
    ///     epsilon: 1e-8
    /// )
    /// ```
    ///
    /// - Complexity: O(alpha x t + log(1/epsilon))
    /// - Precondition: epsilon > 0
    /// - Precondition: initialState must have exactly systemQubits qubits
    @_optimize(speed)
    public func simulateEvolution(
        initialState: QuantumState,
        time: Double,
        epsilon: Double,
    ) async -> QubitizationEvolutionResult {
        ValidationUtilities.validatePositiveDouble(epsilon, name: "epsilon")
        ValidationUtilities.validateStateQubitCount(initialState, required: systemQubits)

        let alpha = blockEncoding.configuration.oneNorm

        let theoreticalDegree = computeOptimalDegree(alpha: alpha, time: time, epsilon: epsilon)
        let maxDegree = max(theoreticalDegree, 1)

        let phaseAngles = QuantumSignalProcessing.computePhaseAngles(
            for: .timeEvolution(time: time),
            degree: maxDegree,
            epsilon: epsilon,
        )

        let signalQubit = systemQubits
        let qspCircuit = QuantumSignalProcessing.buildCircuit(
            walkOperator: walkOperator,
            phaseAngles: phaseAngles,
            signalQubit: signalQubit,
        )

        let totalQubits = blockEncoding.totalQubits
        var extendedInitial = extendState(initialState, toQubits: totalQubits)

        let prepareCircuit = blockEncoding.prepareCircuit()
        extendedInitial = prepareCircuit.execute(on: extendedInitial)

        let evolvedExtended = qspCircuit.execute(on: extendedInitial)

        let finalState = projectToSystemQubits(state: evolvedExtended)

        let actualDegree = phaseAngles.polynomialDegree
        let walkCalls = actualDegree

        let theoreticalBound = Int(ceil(alpha * abs(time) + log(1.0 / epsilon) / log(2.0)))

        return QubitizationEvolutionResult(
            evolvedState: finalState,
            time: time,
            epsilon: epsilon,
            walkOperatorCalls: walkCalls,
            theoreticalBound: theoreticalBound,
            polynomialDegree: actualDegree,
        )
    }

    /// Estimates a Hamiltonian eigenvalue via phase estimation on the walk operator.
    ///
    /// Uses quantum phase estimation to extract eigenvalue information from the walk
    /// operator. The walk operator eigenvalue e^(i*phi) relates to Hamiltonian eigenvalue
    /// lambda by lambda = alpha * cos(phi).
    ///
    /// - Parameters:
    ///   - eigenstate: Approximate eigenstate of the Hamiltonian
    ///   - precisionBits: Number of bits for phase estimation precision
    /// - Returns: Eigenvalue estimation result with confidence interval
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
    /// let qubitization = Qubitization(hamiltonian: H, systemQubits: 2)
    /// let groundState = QuantumState(qubits: 2)
    /// let result = await qubitization.estimateEigenvalue(
    ///     eigenstate: groundState,
    ///     precisionBits: 8
    /// )
    /// ```
    ///
    /// - Complexity: O(2^precisionBits x walkCircuitSize)
    /// - Precondition: precisionBits > 0
    /// - Precondition: eigenstate must have exactly systemQubits qubits
    @_optimize(speed)
    public func estimateEigenvalue(
        eigenstate: QuantumState,
        precisionBits: Int,
    ) async -> QubitizationEigenvalueResult {
        ValidationUtilities.validatePositiveInt(precisionBits, name: "precisionBits")
        ValidationUtilities.validateStateQubitCount(eigenstate, required: systemQubits)

        let alpha = blockEncoding.configuration.oneNorm

        let totalSystemQubits = blockEncoding.totalQubits
        let totalQubits = precisionBits + totalSystemQubits

        var circuit = QuantumCircuit(qubits: totalQubits)

        for i in 0 ..< precisionBits {
            circuit.append(.hadamard, to: i)
        }

        let extendedEigenstate = extendState(eigenstate, toQubits: totalSystemQubits)
        let prepareCircuit = blockEncoding.prepareCircuit()
        let preparedEigenstate = prepareCircuit.execute(on: extendedEigenstate)

        let basicWalkCircuit = walkOperator.buildWalkCircuit()

        for k in 0 ..< precisionBits {
            let controlQubit = precisionBits - 1 - k
            let power = 1 << k

            var controlledOps: [(QuantumGate, [Int])] = []
            for op in basicWalkCircuit.operations {
                if case let .gate(g, qubits, _) = op {
                    var shiftedQubits = [Int]()
                    shiftedQubits.reserveCapacity(qubits.count)
                    for q in qubits {
                        shiftedQubits.append(q + precisionBits)
                    }

                    if shiftedQubits.count == 1 {
                        let decomposition = ControlledGateDecomposer.decompose(
                            gate: g,
                            controls: [controlQubit],
                            target: shiftedQubits[0],
                        )
                        for (dg, dq) in decomposition {
                            controlledOps.append((dg, dq))
                        }
                    } else if shiftedQubits.count == 2 {
                        switch g {
                        case .cnot:
                            controlledOps.append((.toffoli, [controlQubit, shiftedQubits[0], shiftedQubits[1]]))
                        case .cz:
                            controlledOps.append((.hadamard, [shiftedQubits[1]]))
                            controlledOps.append((.toffoli, [controlQubit, shiftedQubits[0], shiftedQubits[1]]))
                            controlledOps.append((.hadamard, [shiftedQubits[1]]))
                        default:
                            for q in shiftedQubits {
                                controlledOps.append((.cnot, [controlQubit, q]))
                            }
                        }
                    }
                }
            }

            for _ in 0 ..< power {
                for (g, q) in controlledOps {
                    circuit.append(g, to: q)
                }
            }
        }

        let inverseQFT = QuantumCircuit.inverseQFT(qubits: precisionBits)
        for op in inverseQFT.operations {
            circuit.append(op)
        }

        var fullInitial = QuantumState(qubits: totalQubits)
        let stateLimit = min(preparedEigenstate.stateSpaceSize, fullInitial.stateSpaceSize)
        for i in 0 ..< stateLimit {
            let amplitude = preparedEigenstate.amplitude(of: i)
            if amplitude.magnitudeSquared > Self.amplitudeThreshold {
                fullInitial.setAmplitude(i, to: amplitude)
            }
        }
        fullInitial.normalize()

        let finalState = circuit.execute(on: fullInitial)

        let phaseResult = finalState.phaseEstimationResult(precisionQubits: precisionBits)
        let estimatedPhase = phaseResult.estimatedPhase * 2.0 * Double.pi

        let eigenvalue = alpha * cos(estimatedPhase)

        let precision = 2.0 * Double.pi / Double(1 << precisionBits)
        let eigenvaluePrecision = alpha * sin(abs(estimatedPhase) + precision / 2.0) * precision
        let confidenceInterval = (eigenvalue - eigenvaluePrecision, eigenvalue + eigenvaluePrecision)

        let walkCalls = (1 << precisionBits) - 1

        return QubitizationEigenvalueResult(
            eigenvalue: eigenvalue,
            phase: estimatedPhase,
            oneNorm: alpha,
            confidenceInterval: confidenceInterval,
            walkOperatorCalls: walkCalls,
        )
    }

    /// Computes the theoretical optimal polynomial degree for time evolution.
    @_optimize(speed)
    @_effects(readonly)
    private func computeOptimalDegree(alpha: Double, time: Double, epsilon: Double) -> Int {
        let scaledTime = alpha * abs(time)
        let logFactor = log(1.0 / epsilon) / log(2.0)
        let degree = Int(ceil(scaledTime + logFactor))
        return max(1, degree)
    }

    /// Extends a system state to include ancilla qubits in |0> state.
    @_optimize(speed)
    @_eagerMove
    private func extendState(_ state: QuantumState, toQubits totalQubits: Int) -> QuantumState {
        if state.qubits >= totalQubits {
            return state
        }

        let newSize = 1 << totalQubits
        let oldSize = state.stateSpaceSize

        let newAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: newSize) { buffer, count in
            for i in 0 ..< oldSize {
                buffer.initializeElement(at: i, to: state.amplitudes[i])
            }
            for i in oldSize ..< newSize {
                buffer.initializeElement(at: i, to: .zero)
            }
            count = newSize
        }

        return QuantumState(qubits: totalQubits, rawAmplitudes: newAmplitudes)
    }

    /// Post-selects extended state on ancilla=|0⟩ to recover system-qubit state.
    @_optimize(speed)
    @_eagerMove
    private func projectToSystemQubits(state: QuantumState) -> QuantumState {
        let systemQubits = systemQubits
        let systemSize = 1 << systemQubits

        let projectedAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: systemSize) {
            buffer, count in
            for i in 0 ..< systemSize {
                buffer[i] = state.amplitude(of: i)
            }
            count = systemSize
        }

        return QuantumState(qubits: systemQubits, rawAmplitudes: projectedAmplitudes)
    }
}

/// Complexity analysis for Qubitization algorithms.
///
/// Provides theoretical bounds and resource estimates for qubitized Hamiltonian simulation
/// and eigenvalue estimation. Used for algorithm design and benchmarking.
///
/// - SeeAlso: ``Qubitization``
///
/// **Example:**
/// ```swift
/// let analysis = QubitizationComplexity.analyzeTimeEvolution(
///     oneNorm: 10.0,
///     time: 5.0,
///     epsilon: 1e-10
/// )
/// print("Optimal queries: \(analysis.optimalQueries)")
/// ```
public enum QubitizationComplexity {
    /// Analyzes time evolution complexity.
    ///
    /// Computes theoretical bounds for e^(-iHt) simulation via Qubitization+QSP.
    ///
    /// - Parameters:
    ///   - oneNorm: 1-norm alpha of the Hamiltonian
    ///   - time: Evolution time t
    ///   - epsilon: Target error
    /// - Returns: Complexity analysis with query bounds
    ///
    /// **Example:**
    /// ```swift
    /// let analysis = QubitizationComplexity.analyzeTimeEvolution(
    ///     oneNorm: 5.0,
    ///     time: 2.0,
    ///     epsilon: 1e-6
    /// )
    /// ```
    @_effects(readonly)
    public static func analyzeTimeEvolution(
        oneNorm: Double,
        time: Double,
        epsilon: Double,
    ) -> (optimalQueries: Int, polynomialDegree: Int, successProbability: Double) {
        let scaledTime = oneNorm * abs(time)
        let logFactor = log(1.0 / epsilon) / log(2.0)

        let polynomialDegree = Int(ceil(scaledTime + logFactor))
        let optimalQueries = polynomialDegree

        let successProbability = 1.0 - epsilon

        return (optimalQueries, polynomialDegree, successProbability)
    }

    /// Analyzes eigenvalue estimation complexity.
    ///
    /// Computes theoretical bounds for phase estimation on the walk operator.
    ///
    /// - Parameters:
    ///   - oneNorm: 1-norm alpha of the Hamiltonian
    ///   - precisionBits: Number of precision qubits
    ///   - successProbability: Target success probability
    /// - Returns: Complexity analysis with query bounds
    ///
    /// **Example:**
    /// ```swift
    /// let analysis = QubitizationComplexity.analyzeEigenvalueEstimation(
    ///     oneNorm: 5.0,
    ///     precisionBits: 10,
    ///     successProbability: 0.99
    /// )
    /// ```
    @_effects(readonly)
    public static func analyzeEigenvalueEstimation(
        oneNorm: Double,
        precisionBits: Int,
        successProbability: Double,
    ) -> (optimalQueries: Int, eigenvaluePrecision: Double, ancillaQubits: Int) {
        let numPhaseEstimationStates = 1 << precisionBits
        let optimalQueries = numPhaseEstimationStates - 1

        let phasePrecision = 2.0 * Double.pi / Double(numPhaseEstimationStates)
        let eigenvaluePrecision = oneNorm * phasePrecision

        let repetitionsForSuccess = Int(ceil(log(1.0 / (1.0 - successProbability)) / log(2.0)))
        let ancillaQubits = precisionBits + repetitionsForSuccess

        return (optimalQueries, eigenvaluePrecision, ancillaQubits)
    }

    /// Computes the query complexity advantage over Trotter methods.
    ///
    /// Qubitization achieves O(alpha*t + log(1/epsilon)) vs Trotter's O((alpha*t)^(1+1/2k) / epsilon^(1/2k))
    /// for order-2k product formulas.
    ///
    /// - Parameters:
    ///   - oneNorm: 1-norm alpha of the Hamiltonian
    ///   - time: Evolution time
    ///   - epsilon: Target error
    ///   - trotterOrder: Order of Trotter-Suzuki decomposition (1, 2, or 4)
    /// - Returns: Speedup factor (Trotter queries / Qubitization queries)
    ///
    /// **Example:**
    /// ```swift
    /// let speedup = QubitizationComplexity.computeSpeedupOverTrotter(
    ///     oneNorm: 10.0,
    ///     time: 5.0,
    ///     epsilon: 1e-8,
    ///     trotterOrder: 2
    /// )
    /// print("Speedup: \(speedup)x")
    /// ```
    @_effects(readonly)
    public static func computeSpeedupOverTrotter(
        oneNorm: Double,
        time: Double,
        epsilon: Double,
        trotterOrder: Int,
    ) -> Double {
        let qubitizationQueries = Double(oneNorm * abs(time) + log(1.0 / epsilon) / log(2.0))

        let k = Double(trotterOrder)
        let exponent = 1.0 + 1.0 / (2.0 * k)
        let epsilonExponent = 1.0 / (2.0 * k)
        let trotterQueries = pow(oneNorm * abs(time), exponent) / pow(epsilon, epsilonExponent)

        return trotterQueries / qubitizationQueries
    }
}

/// Factory methods for constructing Qubitization circuits.
///
/// Provides convenience methods for building block encodings, walk operators,
/// and QSP circuits without manually managing intermediate structures.
///
/// - SeeAlso: ``Qubitization``
/// - SeeAlso: ``QuantumSignalProcessing``
///
/// **Example:**
/// ```swift
/// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
/// let circuit = QubitizationCircuits.buildTimeEvolutionCircuit(
///     hamiltonian: H, systemQubits: 2, time: 1.0, epsilon: 1e-6
/// )
/// ```
public enum QubitizationCircuits {
    /// Builds a complete time evolution circuit using Qubitization.
    ///
    /// Constructs the full circuit for e^(-iHt) including PREPARE, QSP sequence,
    /// and state preparation.
    ///
    /// - Parameters:
    ///   - hamiltonian: Observable to simulate
    ///   - systemQubits: Number of system qubits
    ///   - time: Evolution time
    ///   - epsilon: Target error
    /// - Returns: Complete evolution circuit
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
    /// let circuit = QubitizationCircuits.buildTimeEvolutionCircuit(
    ///     hamiltonian: H,
    ///     systemQubits: 2,
    ///     time: 1.0,
    ///     epsilon: 1e-6
    /// )
    /// ```
    ///
    /// - Precondition: systemQubits > 0
    /// - Precondition: epsilon > 0
    @_optimize(speed)
    @_eagerMove
    public static func buildTimeEvolutionCircuit(
        hamiltonian: Observable,
        systemQubits: Int,
        time: Double,
        epsilon: Double,
    ) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(systemQubits)
        ValidationUtilities.validatePositiveDouble(epsilon, name: "epsilon")

        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: systemQubits)
        let walkOp = QubitizedWalkOperator(blockEncoding: encoding)

        let alpha = encoding.configuration.oneNorm
        let degree = Int(ceil(alpha * abs(time) + log(1.0 / epsilon) / log(2.0)))

        let phases = QuantumSignalProcessing.computePhaseAngles(
            for: .timeEvolution(time: time),
            degree: max(1, degree),
            epsilon: epsilon,
        )

        let signalQubit = systemQubits

        var circuit = encoding.prepareCircuit()

        let qspCircuit = QuantumSignalProcessing.buildCircuit(
            walkOperator: walkOp,
            phaseAngles: phases,
            signalQubit: signalQubit,
        )

        for op in qspCircuit.operations {
            circuit.append(op)
        }

        return circuit
    }

    /// Builds a walk operator circuit from a Hamiltonian.
    ///
    /// Convenience method that constructs block encoding and walk operator
    /// in a single call.
    ///
    /// - Parameters:
    ///   - hamiltonian: Observable to encode
    ///   - systemQubits: Number of system qubits
    /// - Returns: Walk operator circuit
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
    /// let walkCircuit = QubitizationCircuits.buildWalkOperatorCircuit(
    ///     hamiltonian: H,
    ///     systemQubits: 2
    /// )
    /// ```
    ///
    /// - Precondition: systemQubits > 0
    @_optimize(speed)
    @_eagerMove
    public static func buildWalkOperatorCircuit(
        hamiltonian: Observable,
        systemQubits: Int,
    ) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(systemQubits)

        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: systemQubits)
        let walkOp = QubitizedWalkOperator(blockEncoding: encoding)

        return walkOp.buildWalkCircuit()
    }

    /// Builds a block encoding circuit from a Hamiltonian.
    ///
    /// Constructs the LCU-based block encoding U such that (⟨0|_a ⊗ I_s) U (|0⟩_a ⊗ I_s) = H/alpha.
    ///
    /// - Parameters:
    ///   - hamiltonian: Observable to encode
    ///   - systemQubits: Number of system qubits
    /// - Returns: Block encoding circuit
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
    /// let blockCircuit = QubitizationCircuits.buildBlockEncodingCircuit(
    ///     hamiltonian: H,
    ///     systemQubits: 2
    /// )
    /// ```
    ///
    /// - Precondition: systemQubits > 0
    @_optimize(speed)
    @_eagerMove
    public static func buildBlockEncodingCircuit(
        hamiltonian: Observable,
        systemQubits: Int,
    ) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(systemQubits)

        let encoding = BlockEncoding(hamiltonian: hamiltonian, systemQubits: systemQubits)
        return encoding.blockEncodingCircuit()
    }

    /// Estimates resources for qubitized time evolution.
    ///
    /// Computes qubit and gate counts without building the full circuit.
    ///
    /// - Parameters:
    ///   - hamiltonian: Observable to simulate
    ///   - systemQubits: Number of system qubits
    ///   - time: Evolution time
    ///   - epsilon: Target error
    /// - Returns: Resource estimates (total qubits, gate count, depth estimate)
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(0.5, PauliString(.z(0))), (-0.3, PauliString(.x(1)))])
    /// let (qubits, gates, depth) = QubitizationCircuits.estimateResources(
    ///     hamiltonian: H,
    ///     systemQubits: 4,
    ///     time: 2.0,
    ///     epsilon: 1e-8
    /// )
    /// ```
    ///
    /// - Complexity: O(L) where L is the number of Hamiltonian terms
    @_effects(readonly)
    public static func estimateResources(
        hamiltonian: Observable,
        systemQubits: Int,
        time: Double,
        epsilon: Double,
    ) -> (totalQubits: Int, gateCount: Int, depthEstimate: Int) {
        let decomposition = LCU.decompose(hamiltonian)
        let ancillaQubits = decomposition.ancillaQubits
        let totalQubits = systemQubits + ancillaQubits

        let termCount = decomposition.termCount

        let prepareGates = ancillaQubits * 4
        let selectGates = termCount * systemQubits * 3
        let reflectionGates = ancillaQubits * 3

        let walkGates = prepareGates + selectGates + reflectionGates

        let alpha = decomposition.oneNorm
        let degree = Int(ceil(alpha * abs(time) + log(1.0 / epsilon) / log(2.0)))
        let qspGates = degree * walkGates + (degree + 1) * 1

        let totalGates = prepareGates + qspGates

        let depthEstimate = degree * (ancillaQubits + termCount) + degree + 1

        return (totalQubits, totalGates, depthEstimate)
    }
}
