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
/// let result = TimeEvolution.evolve(
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
/// let result = await TimeEvolution.evolveMPS(
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
/// let ground = InitialStateSpecification.groundState(qubits: 4)
/// let basis = InitialStateSpecification.basisState(0b1010, qubits: 4)
/// let custom = InitialStateSpecification.quantumState(myState)
/// ```
///
/// - SeeAlso: ``TimeEvolution``
@frozen
public enum InitialStateSpecification: Sendable {
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
        initialState: InitialStateSpecification,
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
            let mpsResult = await evolveMPS(
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
    /// let result = await TimeEvolution.evolveMPS(
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
    /// - Returns: MPSTimeEvolutionResult with final MPS and truncation statistics
    /// - Complexity: O(steps * n * chi^3) where n = qubits, chi = bond dimension
    @_optimize(speed)
    @_eagerMove
    public static func evolveMPS(
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
    /// Error bounds by order:
    /// - First order: O((αt)² / steps)
    /// - Second order: O((αt)³ / steps²)
    /// - Fourth order: O((αt)⁵ / steps⁴)
    /// - Sixth order: O((αt)⁷ / steps⁶)
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
            let steps2 = Double(steps * steps)
            return alphaT3 / steps2

        case .fourth:
            let alphaT5 = alphaT * alphaT * alphaT * alphaT * alphaT
            let steps4 = Double(steps * steps * steps * steps)
            return alphaT5 / steps4

        case .sixth:
            let alphaT7 = alphaT * alphaT * alphaT * alphaT * alphaT * alphaT * alphaT
            let steps6 = Double(steps * steps * steps * steps * steps * steps)
            return alphaT7 / steps6
        }
    }

    /// Estimate query complexity for Hamiltonian simulation.
    ///
    /// Computes the expected number of queries to the Hamiltonian oracle required to
    /// achieve error at most epsilon. Query complexity varies significantly by method:
    ///
    /// - Trotter-Suzuki (1st order): O(α²t²/ε)
    /// - Trotter-Suzuki (2nd order): O((αt)^(3/2)/ε^(1/2))
    /// - Trotter-Suzuki (4th order): O((αt)^(5/4)/ε^(1/4))
    /// - Qubitization: O(αt + log(1/ε)) - optimal
    /// - LCU: O(αt log(αt/ε) / log log(αt/ε))
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

    @_optimize(speed)
    private static func evolveTrotterSuzuki(
        hamiltonian: Observable,
        initialState: InitialStateSpecification,
        time: Double,
        order: TrotterOrder,
        steps: Int,
    ) -> TimeEvolutionResult {
        let state: QuantumState
        let qubits: Int

        switch initialState {
        case let .groundState(n):
            qubits = n
            state = QuantumState(qubits: n)

        case let .basisState(index, n):
            qubits = n
            var basisState = QuantumState(qubits: n)
            basisState.setAmplitude(0, to: .zero)
            basisState.setAmplitude(index, to: .one)
            state = basisState

        case let .quantumState(existingState):
            qubits = existingState.qubits
            state = existingState

        case let .mps(mpsState):
            qubits = mpsState.qubits
            state = mpsState.toQuantumState()
        }

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

    @_effects(readonly)
    private static func computeOneNorm(_ hamiltonian: Observable) -> Double {
        var sum = 0.0
        for term in hamiltonian.terms {
            sum += abs(term.coefficient)
        }
        return sum
    }

    @_optimize(speed)
    private static func evolveLCU(
        hamiltonian: Observable,
        initialState: InitialStateSpecification,
        time: Double,
        ancillaQubits: Int,
    ) async -> TimeEvolutionResult {
        let state: QuantumState
        let systemQubits: Int

        switch initialState {
        case let .groundState(n):
            systemQubits = n
            state = QuantumState(qubits: n)

        case let .basisState(index, n):
            systemQubits = n
            var basisState = QuantumState(qubits: n)
            basisState.setAmplitude(0, to: .zero)
            basisState.setAmplitude(index, to: .one)
            state = basisState

        case let .quantumState(existingState):
            systemQubits = existingState.qubits
            state = existingState

        case let .mps(mpsState):
            systemQubits = mpsState.qubits
            state = mpsState.toQuantumState()
        }

        let decomposition = LCU.decompose(hamiltonian)
        let totalQubits = systemQubits + max(decomposition.ancillaQubits, ancillaQubits)

        let circuit = LCU.blockEncodingCircuit(
            decomposition: decomposition,
            systemQubits: systemQubits,
            ancillaStart: systemQubits,
        )

        var extendedState = extendStateWithAncillas(state, totalQubits: totalQubits)
        extendedState = circuit.execute(on: extendedState)

        let projectedState = projectToSystemQubits(
            state: extendedState,
            systemQubits: systemQubits,
        )

        let oneNorm = decomposition.oneNorm
        let errorBound = oneNorm > 1e-15 ? 1.0 / oneNorm : 1.0

        return TimeEvolutionResult(
            finalState: projectedState,
            time: time,
            steps: 1,
            errorBound: errorBound,
            gateCount: circuit.count,
            circuitDepth: circuit.depth,
        )
    }

    @_optimize(speed)
    private static func evolveQubitization(
        hamiltonian: Observable,
        initialState: InitialStateSpecification,
        time: Double,
        polynomialDegree: Int,
    ) async -> TimeEvolutionResult {
        let state: QuantumState
        let systemQubits: Int

        switch initialState {
        case let .groundState(n):
            systemQubits = n
            state = QuantumState(qubits: n)

        case let .basisState(index, n):
            systemQubits = n
            var basisState = QuantumState(qubits: n)
            basisState.setAmplitude(0, to: .zero)
            basisState.setAmplitude(index, to: .one)
            state = basisState

        case let .quantumState(existingState):
            systemQubits = existingState.qubits
            state = existingState

        case let .mps(mpsState):
            systemQubits = mpsState.qubits
            state = mpsState.toQuantumState()
        }

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
        if abs(xxCoeff) > 1e-15, abs(yyCoeff) > 1e-15, abs(zzCoeff) > 1e-15 {
            let avgXY = (xxCoeff + yyCoeff) / 2.0
            twoSiteGate = TEBDGates.heisenbergXXZ(angle: avgXY * dt, delta: zzCoeff / avgXY)
        } else if abs(zzCoeff) > 1e-15 {
            twoSiteGate = TEBDGates.zzEvolution(angle: zzCoeff * dt)
        } else if abs(xxCoeff) > 1e-15 {
            twoSiteGate = TEBDGates.xxEvolution(angle: xxCoeff * dt)
        } else if abs(yyCoeff) > 1e-15 {
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
            if abs(xCoeffs[site]) > 1e-15 {
                hasSingleSite = true
                break
            }
            if abs(zCoeffs[site]) > 1e-15 {
                hasSingleSite = true
                break
            }
        }

        if hasSingleSite {
            singleSiteGates.reserveCapacity(qubits)
            for site in 0 ..< qubits {
                if abs(xCoeffs[site]) > 1e-15 {
                    singleSiteGates.append(TEBDGates.xEvolution(angle: xCoeffs[site] * dt))
                } else if abs(zCoeffs[site]) > 1e-15 {
                    singleSiteGates.append(TEBDGates.zEvolution(angle: zCoeffs[site] * dt))
                } else {
                    singleSiteGates.append([[.one, .zero], [.zero, .one]])
                }
            }
        }

        return (twoSiteGate, singleSiteGates)
    }

    @_optimize(speed)
    @_eagerMove
    private static func extendStateWithAncillas(_ state: QuantumState, totalQubits: Int) -> QuantumState {
        let newSize = 1 << totalQubits
        let oldSize = state.stateSpaceSize

        var newAmplitudes = [Complex<Double>](repeating: .zero, count: newSize)

        for i in 0 ..< oldSize {
            newAmplitudes[i] = state.amplitudes[i]
        }

        return QuantumState(qubits: totalQubits, amplitudes: newAmplitudes)
    }

    @_optimize(speed)
    @_eagerMove
    private static func projectToSystemQubits(state: QuantumState, systemQubits: Int) -> QuantumState {
        let systemSize = 1 << systemQubits
        let ancillaSize = state.stateSpaceSize / systemSize

        var projectedAmplitudes = [Complex<Double>](repeating: .zero, count: systemSize)

        for i in 0 ..< systemSize {
            var sumSquared = 0.0
            for a in 0 ..< ancillaSize {
                let fullIndex = i + a * systemSize
                if fullIndex < state.stateSpaceSize {
                    sumSquared += state.amplitude(of: fullIndex).magnitudeSquared
                }
            }
            let amplitude = state.amplitude(of: i)
            let norm = sqrt(sumSquared)
            if norm > 1e-15 {
                let phase = amplitude / Complex(sqrt(amplitude.magnitudeSquared), 0.0)
                projectedAmplitudes[i] = Complex(norm, 0.0) * phase
            }
        }

        return QuantumState(qubits: systemQubits, amplitudes: projectedAmplitudes)
    }
}
