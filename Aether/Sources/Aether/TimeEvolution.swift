// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Result of time evolution simulation.
///
/// Contains the final quantum state after Hamiltonian time evolution along with metadata
/// about the simulation: evolution time, number of Trotter steps, error bounds, and circuit
/// statistics. Used by ``TimeEvolution/evolve(hamiltonian:initialState:time:method:)`` to
/// return both the computed state and quality metrics.
///
/// **Example:**
/// ```swift
/// let hamiltonian = Observable(terms: [
///     (0.5, PauliString(.z(0), .z(1))),
///     (-0.3, PauliString(.x(0)))
/// ])
/// let result = await TimeEvolution.evolve(
///     hamiltonian: hamiltonian,
///     initialState: .groundState(qubits: 2),
///     time: 1.0,
///     method: .trotterSuzuki(order: .second, steps: 10)
/// )
/// let energy = hamiltonian.expectationValue(of: result.finalState)
/// ```
///
/// - SeeAlso: ``TimeEvolution``
/// - SeeAlso: ``TimeEvolutionMethod``
@frozen
public struct TimeEvolutionResult: Sendable {
    /// Final quantum state after time evolution.
    public let finalState: QuantumState

    /// Total evolution time simulated.
    public let time: Double

    /// Number of Trotter steps used (for Trotter-Suzuki method).
    public let steps: Int

    /// Upper bound on simulation error from Trotter approximation.
    public let errorBound: Double

    /// Total number of quantum gates in the evolution circuit.
    public let gateCount: Int

    /// Circuit depth (longest path through circuit).
    public let circuitDepth: Int
}

/// MPS-based time evolution result.
///
/// Contains the final Matrix Product State after TEBD (Time-Evolving Block Decimation)
/// simulation along with truncation statistics tracking approximation errors from bond
/// dimension limiting. Used for simulating large systems with limited entanglement.
///
/// **Example:**
/// ```swift
/// let hamiltonian = Observable(terms: [
///     (-1.0, PauliString(.z(0), .z(1))),
///     (-0.5, PauliString(.x(0)))
/// ])
/// let mps = MatrixProductState(qubits: 50)
/// let result = await TimeEvolution.evolve(
///     hamiltonian: hamiltonian,
///     initialState: mps,
///     time: 1.0,
///     steps: 100,
///     maxBondDimension: 64
/// )
/// let finalEnergy = result.finalState.expectationValue(of: hamiltonian)
/// ```
///
/// - SeeAlso: ``TimeEvolution``
/// - SeeAlso: ``MatrixProductState``
/// - SeeAlso: ``MPSTruncationStatistics``
@frozen
public struct MPSTimeEvolutionResult: Sendable {
    /// Final MPS after time evolution.
    public let finalState: MatrixProductState

    /// Total evolution time simulated.
    public let time: Double

    /// Accumulated truncation error statistics from SVD truncations.
    public let truncationStatistics: MPSTruncationStatistics

    /// Maximum bond dimension reached during evolution.
    public let maxBondDimensionReached: Int
}

/// Evolution method selection.
///
/// Specifies which algorithm to use for Hamiltonian time evolution. Different methods
/// offer different trade-offs between accuracy, circuit depth, and qubit overhead.
/// Trotter-Suzuki is the most common choice for near-term devices.
///
/// **Example:**
/// ```swift
/// let trotter = TimeEvolutionMethod.trotterSuzuki(order: .fourth, steps: 20)
/// let mps = TimeEvolutionMethod.mps(maxBondDimension: 64, truncationThreshold: 1e-10)
/// ```
///
/// - SeeAlso: ``TimeEvolution``
/// - SeeAlso: ``TrotterOrder``
@frozen
public enum TimeEvolutionMethod: Sendable {
    /// Trotter-Suzuki product formula decomposition.
    ///
    /// Approximates exp(-iHt) as product of exponentials of individual Hamiltonian terms.
    /// Higher order formulas achieve better accuracy with fewer steps at the cost of
    /// deeper circuits per step.
    ///
    /// - Parameters:
    ///   - order: Decomposition order (first, second, fourth, or sixth)
    ///   - steps: Number of Trotter steps for time discretization
    case trotterSuzuki(order: TrotterOrder, steps: Int)

    /// Linear Combination of Unitaries.
    ///
    /// Implements Hamiltonian simulation via LCU oracle with ancilla qubits for
    /// amplitude encoding. Requires additional ancilla qubits but achieves better
    /// asymptotic scaling than Trotter for certain Hamiltonians.
    ///
    /// - Parameter ancillaQubits: Number of ancilla qubits for LCU encoding
    case lcu(ancillaQubits: Int)

    /// Qubitization with quantum signal processing.
    ///
    /// Optimal query complexity method using block-encoding and polynomial
    /// approximation via quantum signal processing. Best for fault-tolerant
    /// quantum computers with high-quality qubits.
    ///
    /// - Parameter polynomialDegree: Degree of Chebyshev polynomial approximation
    case qubitization(polynomialDegree: Int)

    /// Matrix Product State time evolution via TEBD.
    ///
    /// Tensor network method for simulating weakly-entangled states efficiently.
    /// Memory scales as O(n * chi^2) instead of O(2^n), enabling simulation of
    /// hundreds of qubits for 1D systems with area-law entanglement.
    ///
    /// - Parameters:
    ///   - maxBondDimension: Maximum allowed bond dimension (controls accuracy vs memory)
    ///   - truncationThreshold: SVD singular value threshold for truncation
    case mps(maxBondDimension: Int, truncationThreshold: Double)
}

/// Initial state specification.
///
/// Defines how to prepare the initial quantum state for time evolution. Supports
/// common quantum states (ground state, computational basis states) as well as
/// arbitrary pre-computed states.
///
/// **Example:**
/// ```swift
/// let ground = InitialState.groundState(qubits: 4)
/// let basis = InitialState.basisState(0b1010, qubits: 4)
/// let custom = InitialState.quantumState(myState)
/// ```
///
/// - SeeAlso: ``TimeEvolution``
@frozen
public enum InitialState: Sendable {
    /// All-zeros ground state |00...0⟩.
    ///
    /// - Parameter qubits: Number of qubits in the system
    case groundState(qubits: Int)

    /// Computational basis state |k⟩.
    ///
    /// - Parameters:
    ///   - basisIndex: Integer encoding the basis state (little-endian)
    ///   - qubits: Number of qubits in the system
    case basisState(Int, qubits: Int)

    /// Pre-constructed quantum state.
    ///
    /// - Parameter state: Existing QuantumState instance
    case quantumState(QuantumState)

    /// Matrix Product State for MPS evolution.
    ///
    /// - Parameter mps: Existing MatrixProductState instance
    case mps(MatrixProductState)
}

/// Unified time evolution API for Hamiltonian simulation.
///
/// Provides high-level methods for simulating quantum time evolution under a Hamiltonian
/// H, computing exp(-iHt)|ψ⟩. Orchestrates different evolution methods including Trotter-Suzuki
/// product formulas for near-term devices, LCU/qubitization for fault-tolerant simulation,
/// and TEBD for tensor network approaches.
///
/// The choice of method depends on the target hardware, required accuracy, and Hamiltonian
/// structure. Trotter-Suzuki is most practical for current NISQ devices while qubitization
/// achieves optimal query complexity for future fault-tolerant systems.
///
/// **Example:**
/// ```swift
/// let ising = Observable(terms: [
///     (-1.0, PauliString(.z(0), .z(1))),
///     (-1.0, PauliString(.z(1), .z(2))),
///     (-0.5, PauliString(.x(0))),
///     (-0.5, PauliString(.x(1))),
///     (-0.5, PauliString(.x(2)))
/// ])
///
/// let result = await TimeEvolution.evolve(
///     hamiltonian: ising,
///     initialState: .groundState(qubits: 3),
///     time: 2.0,
///     method: .trotterSuzuki(order: .fourth, steps: 20)
/// )
///
/// let error = TimeEvolution.estimateTrotterError(
///     hamiltonian: ising,
///     time: 2.0,
///     order: .fourth,
///     steps: 20
/// )
/// ```
///
/// - SeeAlso: ``TrotterSuzuki``
/// - SeeAlso: ``TimeEvolutionMethod``
/// - SeeAlso: ``TimeEvolutionResult``
public enum TimeEvolution {
    /// Numerical zero threshold for floating-point comparisons.
    private static let epsilon: Double = 1e-15

    /// Evolve a quantum state under Hamiltonian time evolution.
    ///
    /// Computes |ψ(t)⟩ = exp(-iHt)|ψ(0)⟩ using the specified evolution method. The method
    /// selection determines the circuit structure and approximation strategy. Trotter-Suzuki
    /// decomposes the evolution into a product of single-term exponentials, while LCU and
    /// qubitization use ancilla-based block encodings. This method is async to support
    /// Qubitization and MPS evolution which use actor-isolated operations.
    ///
    /// **Example:**
    /// ```swift
    /// let hamiltonian = Observable(terms: [
    ///     (0.5, PauliString(.z(0), .z(1))),
    ///     (-0.3, PauliString(.x(0))),
    ///     (-0.3, PauliString(.x(1)))
    /// ])
    ///
    /// let result = await TimeEvolution.evolve(
    ///     hamiltonian: hamiltonian,
    ///     initialState: .groundState(qubits: 2),
    ///     time: 1.0,
    ///     method: .trotterSuzuki(order: .second, steps: 10)
    /// )
    ///
    /// let finalEnergy = hamiltonian.expectationValue(of: result.finalState)
    /// ```
    ///
    /// - Parameters:
    ///   - hamiltonian: Observable representing the Hamiltonian as sum of Pauli strings
    ///   - initialState: Specification of the initial quantum state
    ///   - time: Total evolution time t in exp(-iHt)
    ///   - method: Evolution algorithm to use
    /// - Returns: TimeEvolutionResult containing final state and simulation metadata
    /// - Complexity: Depends on method; Trotter is O(steps * terms * 2^n)
    /// - Precondition: Hamiltonian must have at least one term
    /// - Precondition: Time must be non-negative
    @_optimize(speed)
    @_eagerMove
    public static func evolve(
        hamiltonian: Observable,
        initialState: InitialState,
        time: Double,
        method: TimeEvolutionMethod,
    ) async -> TimeEvolutionResult {
        ValidationUtilities.validateNonEmpty(hamiltonian.terms, name: "hamiltonian.terms")
        ValidationUtilities.validateNonNegativeDouble(time, name: "Evolution time")

        switch method {
        case let .trotterSuzuki(order, steps):
            return evolveTrotterSuzuki(
                hamiltonian: hamiltonian,
                initialState: initialState,
                time: time,
                order: order,
                steps: steps,
            )

        case let .lcu(ancillaQubits):
            return await evolveLCU(
                hamiltonian: hamiltonian,
                initialState: initialState,
                time: time,
                ancillaQubits: ancillaQubits,
            )

        case let .qubitization(polynomialDegree):
            return await evolveQubitization(
                hamiltonian: hamiltonian,
                initialState: initialState,
                time: time,
                polynomialDegree: polynomialDegree,
            )

        case let .mps(maxBondDimension, _):
            let mpsState: MatrixProductState = switch initialState {
            case let .groundState(qubits):
                MatrixProductState(qubits: qubits, maxBondDimension: maxBondDimension)
            case let .basisState(index, qubits):
                MatrixProductState(qubits: qubits, basisState: index, maxBondDimension: maxBondDimension)
            case let .quantumState(state):
                MatrixProductState(from: state, maxBondDimension: maxBondDimension)
            case let .mps(existingMPS):
                existingMPS
            }

            let defaultSteps = max(10, Int(time * 10))
            let mpsResult = await evolve(
                hamiltonian: hamiltonian,
                initialState: mpsState,
                time: time,
                steps: defaultSteps,
                maxBondDimension: maxBondDimension,
            )

            let finalQuantumState = mpsResult.finalState.toQuantumState()

            return TimeEvolutionResult(
                finalState: finalQuantumState,
                time: time,
                steps: defaultSteps,
                errorBound: mpsResult.truncationStatistics.cumulativeError,
                gateCount: 0,
                circuitDepth: 0,
            )
        }
    }

    /// Evolve a Matrix Product State under Hamiltonian time evolution using TEBD.
    ///
    /// Applies Time-Evolving Block Decimation (TEBD) to simulate exp(-iHt)|ψ⟩ directly
    /// on the MPS representation. Each Trotter step applies two-site gates followed by
    /// SVD truncation to limit bond dimension growth. Efficient for 1D systems with
    /// nearest-neighbor interactions where entanglement grows slowly.
    ///
    /// **Example:**
    /// ```swift
    /// let ising = Observable(terms: [
    ///     (-1.0, PauliString(.z(0), .z(1))),
    ///     (-1.0, PauliString(.z(1), .z(2))),
    ///     (-0.5, PauliString(.x(0))),
    ///     (-0.5, PauliString(.x(1))),
    ///     (-0.5, PauliString(.x(2)))
    /// ])
    ///
    /// let mps = MatrixProductState(qubits: 100)
    /// let result = await TimeEvolution.evolve(
    ///     hamiltonian: ising,
    ///     initialState: mps,
    ///     time: 1.0,
    ///     steps: 50,
    ///     maxBondDimension: 64
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - hamiltonian: Observable representing the Hamiltonian
    ///   - initialState: Initial MPS to evolve
    ///   - time: Total evolution time
    ///   - steps: Number of Trotter steps
    ///   - maxBondDimension: Maximum bond dimension for SVD truncation
    /// - Returns: ``MPSTimeEvolutionResult`` with final MPS and truncation statistics
    /// - Precondition: Hamiltonian must have at least one term
    /// - Precondition: Time must be non-negative
    /// - Precondition: Steps must be positive
    /// - Precondition: Max bond dimension must be positive
    /// - Complexity: O(steps * n * chi^3) where n = qubits, chi = bond dimension
    @_optimize(speed)
    @_eagerMove
    public static func evolve(
        hamiltonian: Observable,
        initialState: MatrixProductState,
        time: Double,
        steps: Int,
        maxBondDimension: Int,
    ) async -> MPSTimeEvolutionResult {
        ValidationUtilities.validateNonEmpty(hamiltonian.terms, name: "hamiltonian.terms")
        ValidationUtilities.validateNonNegativeDouble(time, name: "Evolution time")
        ValidationUtilities.validatePositiveInt(steps, name: "Steps")
        ValidationUtilities.validatePositiveInt(maxBondDimension, name: "Max bond dimension")

        let mpsEvolution = MPSTimeEvolution()

        let (twoSiteGate, singleSiteGates) = deriveTEBDGates(
            hamiltonian: hamiltonian,
            time: time,
            steps: steps,
            qubits: initialState.qubits,
        )

        let result = await mpsEvolution.evolveWithGate(
            mps: initialState,
            twoSiteGate: twoSiteGate,
            singleSiteGates: singleSiteGates,
            time: time,
            steps: steps,
            order: .second,
        )

        return MPSTimeEvolutionResult(
            finalState: result.finalState,
            time: result.time,
            truncationStatistics: result.truncationStatistics,
            maxBondDimensionReached: result.maxBondDimensionReached,
        )
    }

    /// Estimate Trotter decomposition error bound.
    ///
    /// Computes an upper bound on the operator norm error ||exp(-iHt) - S_p(t/n)^n|| based on
    /// rigorous Trotter error bounds. The error scales as O((αt)^(p+1)/n^p) where α is the
    /// Hamiltonian 1-norm, p is the Trotter order, and n is the number of steps.
    ///
    /// First-order error is O((αt)²/steps), second-order is O((αt)³/steps²),
    /// fourth-order is O((αt)⁵/steps⁴), and sixth-order is O((αt)⁷/steps⁶).
    ///
    /// **Example:**
    /// ```swift
    /// let hamiltonian = Observable(terms: [
    ///     (0.5, PauliString(.z(0), .z(1))),
    ///     (-0.3, PauliString(.x(0)))
    /// ])
    ///
    /// let error = TimeEvolution.estimateTrotterError(
    ///     hamiltonian: hamiltonian,
    ///     time: 1.0,
    ///     order: .second,
    ///     steps: 10
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - hamiltonian: Observable representing the Hamiltonian
    ///   - time: Total evolution time
    ///   - order: Trotter decomposition order
    ///   - steps: Number of Trotter steps
    /// - Returns: Upper bound on simulation error
    /// - Precondition: Hamiltonian must have at least one term
    /// - Precondition: Time must be non-negative
    /// - Precondition: Steps must be positive
    /// - Complexity: O(k) where k = number of Hamiltonian terms
    @_effects(readonly)
    public static func estimateTrotterError(
        hamiltonian: Observable,
        time: Double,
        order: TrotterOrder,
        steps: Int,
    ) -> Double {
        ValidationUtilities.validateNonEmpty(hamiltonian.terms, name: "hamiltonian.terms")
        ValidationUtilities.validateNonNegativeDouble(time, name: "Evolution time")
        ValidationUtilities.validatePositiveInt(steps, name: "Steps")

        let oneNorm = computeOneNorm(hamiltonian)
        let alphaT = oneNorm * time

        switch order {
        case .first:
            return alphaT * alphaT / Double(steps)

        case .second:
            let alphaT3 = alphaT * alphaT * alphaT
            let dSteps = Double(steps)
            return alphaT3 / (dSteps * dSteps)

        case .fourth:
            let alphaT5 = alphaT * alphaT * alphaT * alphaT * alphaT
            let dSteps = Double(steps)
            let steps2 = dSteps * dSteps
            return alphaT5 / (steps2 * steps2)

        case .sixth:
            let alphaT7 = alphaT * alphaT * alphaT * alphaT * alphaT * alphaT * alphaT
            let dSteps = Double(steps)
            let steps2 = dSteps * dSteps
            return alphaT7 / (steps2 * steps2 * steps2)
        }
    }

    /// Estimate query complexity for Hamiltonian simulation.
    ///
    /// Computes the expected number of queries to the Hamiltonian oracle required to
    /// achieve error at most epsilon. Query complexity varies significantly by method:
    ///
    /// Trotter-Suzuki scales as O(α²t²/ε) for first order, O((αt)^(3/2)/ε^(1/2)) for second,
    /// and O((αt)^(5/4)/ε^(1/4)) for fourth. Qubitization achieves optimal O(αt + log(1/ε))
    /// while LCU achieves O(αt log(αt/ε) / log log(αt/ε)).
    ///
    /// **Example:**
    /// ```swift
    /// let hamiltonian = Observable(terms: [
    ///     (0.5, PauliString(.z(0), .z(1))),
    ///     (-0.3, PauliString(.x(0)))
    /// ])
    ///
    /// let trotterQueries = TimeEvolution.estimateQueryComplexity(
    ///     hamiltonian: hamiltonian,
    ///     time: 10.0,
    ///     epsilon: 0.01,
    ///     method: .trotterSuzuki(order: .second, steps: 100)
    /// )
    ///
    /// let optimalQueries = TimeEvolution.estimateQueryComplexity(
    ///     hamiltonian: hamiltonian,
    ///     time: 10.0,
    ///     epsilon: 0.01,
    ///     method: .qubitization(polynomialDegree: 50)
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - hamiltonian: Observable representing the Hamiltonian
    ///   - time: Total evolution time
    ///   - epsilon: Target error tolerance
    ///   - method: Evolution method for complexity estimation
    /// - Returns: Estimated number of oracle queries
    /// - Precondition: Hamiltonian must have at least one term
    /// - Precondition: Time must be non-negative
    /// - Precondition: Error tolerance must be positive
    /// - Complexity: O(k) where k = number of Hamiltonian terms
    @_effects(readonly)
    public static func estimateQueryComplexity(
        hamiltonian: Observable,
        time: Double,
        epsilon: Double,
        method: TimeEvolutionMethod,
    ) -> Int {
        ValidationUtilities.validateNonEmpty(hamiltonian.terms, name: "hamiltonian.terms")
        ValidationUtilities.validateNonNegativeDouble(time, name: "Evolution time")
        ValidationUtilities.validatePositiveDouble(epsilon, name: "Error tolerance")

        let oneNorm = computeOneNorm(hamiltonian)
        let alphaT = oneNorm * time

        switch method {
        case let .trotterSuzuki(order, _):
            switch order {
            case .first:
                return Int(ceil(alphaT * alphaT / epsilon))

            case .second:
                let numerator = pow(alphaT, 1.5)
                let denominator = pow(epsilon, 0.5)
                return Int(ceil(numerator / denominator))

            case .fourth:
                let numerator = pow(alphaT, 1.25)
                let denominator = pow(epsilon, 0.25)
                return Int(ceil(numerator / denominator))

            case .sixth:
                let numerator = pow(alphaT, 7.0 / 6.0)
                let denominator = pow(epsilon, 1.0 / 6.0)
                return Int(ceil(numerator / denominator))
            }

        case .lcu:
            let logFactor = log(alphaT / epsilon)
            let logLogFactor = max(1.0, log(logFactor))
            return Int(ceil(alphaT * logFactor / logLogFactor))

        case .qubitization:
            return Int(ceil(alphaT + log(1.0 / epsilon)))

        case .mps:
            return Int(ceil(alphaT / epsilon))
        }
    }

    /// Performs Trotter-Suzuki time evolution returning result with error bound.
    @_optimize(speed)
    @_eagerMove
    private static func evolveTrotterSuzuki(
        hamiltonian: Observable,
        initialState: InitialState,
        time: Double,
        order: TrotterOrder,
        steps: Int,
    ) -> TimeEvolutionResult {
        let (state, qubits) = resolveInitialState(initialState)

        let config = TrotterConfiguration(order: order, steps: steps)
        let circuit = TrotterSuzuki.evolve(hamiltonian, time: time, qubits: qubits, config: config)

        let finalState = circuit.execute(on: state)

        let errorBound = estimateTrotterError(
            hamiltonian: hamiltonian,
            time: time,
            order: order,
            steps: steps,
        )

        return TimeEvolutionResult(
            finalState: finalState,
            time: time,
            steps: steps,
            errorBound: errorBound,
            gateCount: circuit.count,
            circuitDepth: circuit.depth,
        )
    }

    /// Computes the 1-norm of a Hamiltonian's coefficients.
    @_effects(readonly)
    private static func computeOneNorm(_ hamiltonian: Observable) -> Double {
        var sum = 0.0
        for term in hamiltonian.terms {
            sum += abs(term.coefficient)
        }
        return sum
    }

    /// Performs LCU-based Hamiltonian simulation with ancilla projection.
    @_optimize(speed)
    private static func evolveLCU(
        hamiltonian: Observable,
        initialState: InitialState,
        time: Double,
        ancillaQubits: Int,
    ) async -> TimeEvolutionResult {
        let (state, systemQubits) = resolveInitialState(initialState)

        let decomposition = LCU.decompose(hamiltonian)
        let totalQubits = systemQubits + max(decomposition.ancillaQubits, ancillaQubits)

        let circuit = LCU.blockEncodingCircuit(
            decomposition: decomposition,
            ancillaStart: systemQubits,
        )

        var extendedState = extendStateWithAncillas(state, totalQubits: totalQubits)
        extendedState = circuit.execute(on: extendedState)

        let projectedState = projectToSystemQubits(
            state: extendedState,
            systemQubits: systemQubits,
        )

        let oneNorm = decomposition.oneNorm
        let errorBound = oneNorm > epsilon ? 1.0 / oneNorm : 1.0

        return TimeEvolutionResult(
            finalState: projectedState,
            time: time,
            steps: 1,
            errorBound: errorBound,
            gateCount: circuit.count,
            circuitDepth: circuit.depth,
        )
    }

    /// Performs qubitization-based Hamiltonian simulation via quantum signal processing.
    @_optimize(speed)
    private static func evolveQubitization(
        hamiltonian: Observable,
        initialState: InitialState,
        time: Double,
        polynomialDegree: Int,
    ) async -> TimeEvolutionResult {
        let (state, systemQubits) = resolveInitialState(initialState)

        let epsilon = polynomialDegree > 0 ? pow(10.0, -Double(polynomialDegree) / 10.0) : 1e-6

        let qubitization = Qubitization(hamiltonian: hamiltonian, systemQubits: systemQubits)
        let result = await qubitization.simulateEvolution(
            initialState: state,
            time: time,
            epsilon: epsilon,
        )

        return TimeEvolutionResult(
            finalState: result.evolvedState,
            time: result.time,
            steps: result.walkOperatorCalls,
            errorBound: result.epsilon,
            gateCount: result.polynomialDegree,
            circuitDepth: result.theoreticalBound,
        )
    }

    /// Derives two-site and single-site TEBD gates from Hamiltonian terms.
    @_optimize(speed)
    @_effects(readonly)
    private static func deriveTEBDGates(
        hamiltonian: Observable,
        time: Double,
        steps: Int,
        qubits: Int,
    ) -> (twoSiteGate: [[Complex<Double>]], singleSiteGates: [[[Complex<Double>]]]) {
        let dt = time / Double(steps)

        var zzCoeff = 0.0
        var xxCoeff = 0.0
        var yyCoeff = 0.0
        var xCoeffs = [Double](repeating: 0.0, count: qubits)
        var zCoeffs = [Double](repeating: 0.0, count: qubits)

        for term in hamiltonian.terms {
            let pauliOps = term.pauliString.operators
            let coeff = term.coefficient

            if pauliOps.count == 2 {
                let op0 = pauliOps[0]
                let op1 = pauliOps[1]

                if abs(op0.qubit - op1.qubit) == 1 {
                    if op0.basis == .z, op1.basis == .z {
                        zzCoeff += coeff
                    } else if op0.basis == .x, op1.basis == .x {
                        xxCoeff += coeff
                    } else if op0.basis == .y, op1.basis == .y {
                        yyCoeff += coeff
                    }
                }
            } else if pauliOps.count == 1 {
                let op = pauliOps[0]
                if op.qubit < qubits {
                    if op.basis == .x {
                        xCoeffs[op.qubit] += coeff
                    } else if op.basis == .z {
                        zCoeffs[op.qubit] += coeff
                    }
                }
            }
        }

        let twoSiteGate: [[Complex<Double>]]
        if abs(xxCoeff) > epsilon, abs(yyCoeff) > epsilon, abs(zzCoeff) > epsilon {
            let avgXY = (xxCoeff + yyCoeff) / 2.0
            twoSiteGate = TEBDGates.heisenbergXXZ(angle: avgXY * dt, delta: zzCoeff / avgXY)
        } else if abs(zzCoeff) > epsilon {
            twoSiteGate = TEBDGates.zzEvolution(angle: zzCoeff * dt)
        } else if abs(xxCoeff) > epsilon {
            twoSiteGate = TEBDGates.xxEvolution(angle: xxCoeff * dt)
        } else if abs(yyCoeff) > epsilon {
            twoSiteGate = TEBDGates.yyEvolution(angle: yyCoeff * dt)
        } else {
            twoSiteGate = [
                [.one, .zero, .zero, .zero],
                [.zero, .one, .zero, .zero],
                [.zero, .zero, .one, .zero],
                [.zero, .zero, .zero, .one],
            ]
        }

        var singleSiteGates: [[[Complex<Double>]]] = []
        var hasSingleSite = false

        for site in 0 ..< qubits {
            if abs(xCoeffs[site]) > epsilon {
                hasSingleSite = true
                break
            }
            if abs(zCoeffs[site]) > epsilon {
                hasSingleSite = true
                break
            }
        }

        if hasSingleSite {
            singleSiteGates.reserveCapacity(qubits)
            for site in 0 ..< qubits {
                if abs(xCoeffs[site]) > epsilon {
                    singleSiteGates.append(TEBDGates.xEvolution(angle: xCoeffs[site] * dt))
                } else if abs(zCoeffs[site]) > epsilon {
                    singleSiteGates.append(TEBDGates.zEvolution(angle: zCoeffs[site] * dt))
                } else {
                    singleSiteGates.append([[.one, .zero], [.zero, .one]])
                }
            }
        }

        return (twoSiteGate, singleSiteGates)
    }

    /// Resolves an initial state specification into a concrete quantum state.
    private static func resolveInitialState(_ spec: InitialState) -> (state: QuantumState, qubits: Int) {
        switch spec {
        case let .groundState(qubits):
            return (QuantumState(qubits: qubits), qubits)

        case let .basisState(index, qubits):
            var basisState = QuantumState(qubits: qubits)
            basisState.setAmplitude(0, to: .zero)
            basisState.setAmplitude(index, to: .one)
            return (basisState, qubits)

        case let .quantumState(existingState):
            return (existingState, existingState.qubits)

        case let .mps(mpsState):
            return (mpsState.toQuantumState(), mpsState.qubits)
        }
    }

    /// Extends a quantum state with zero-initialized ancilla qubits.
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    private static func extendStateWithAncillas(_ state: QuantumState, totalQubits: Int) -> QuantumState {
        let newSize = 1 << totalQubits
        let oldSize = state.stateSpaceSize

        let newAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: newSize) { buffer, count in
            for i in 0 ..< oldSize {
                buffer.initializeElement(at: i, to: state.amplitudes[i])
            }
            if newSize > oldSize {
                // Safe: buffer.baseAddress! non-nil because newSize > 0
                buffer.baseAddress!.advanced(by: oldSize).initialize(repeating: .zero, count: newSize - oldSize)
            }
            count = newSize
        }

        return QuantumState(qubits: totalQubits, rawAmplitudes: newAmplitudes)
    }

    /// Post-selects extended state on ancilla=|0⟩ to recover system-qubit state.
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    private static func projectToSystemQubits(state: QuantumState, systemQubits: Int) -> QuantumState {
        let systemSize = 1 << systemQubits

        let projectedAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: systemSize) { buffer, count in
            for i in 0 ..< systemSize {
                buffer.initializeElement(at: i, to: state.amplitude(of: i))
            }
            count = systemSize
        }

        return QuantumState(qubits: systemQubits, rawAmplitudes: projectedAmplitudes)
    }
}
