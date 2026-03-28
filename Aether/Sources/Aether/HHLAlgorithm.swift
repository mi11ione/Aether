// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate
import Foundation

/// Specification of a quantum linear system problem Ax = b for the HHL algorithm.
///
/// Defines the Hermitian matrix A as an ``Observable`` (weighted sum of Pauli strings),
/// the right-hand side vector |b⟩ as a ``QuantumState``, and an estimate of the condition
/// number κ = λ_max/λ_min. The condition number controls both algorithm complexity and
/// the polynomial approximation quality for eigenvalue inversion.
///
/// The HHL algorithm outputs a quantum state |x⟩ proportional to A⁻¹|b⟩. Classical readout
/// of the full solution vector requires O(2^n) measurements where n is the number of system
/// qubits.
///
/// **Example:**
/// ```swift
/// let hamiltonian = Observable(terms: [
///     (0.5, PauliString(.z(0))),
///     (-0.3, PauliString(.x(1)))
/// ])
/// let rhs = QuantumState(qubits: 2)
/// let problem = HHLProblem(
///     hamiltonian: hamiltonian,
///     systemQubits: 2,
///     rightHandSide: rhs,
///     conditionNumber: 4.0
/// )
/// ```
///
/// - SeeAlso: ``HHLAlgorithm``
/// - SeeAlso: ``HHLResult``
@frozen
public struct HHLProblem: Sendable {
    /// Hermitian matrix A as a Hamiltonian (weighted sum of Pauli strings).
    ///
    /// The eigenvalues of A determine the solution. A must be Hermitian for HHL to produce
    /// correct results. The eigenvalue range [λ_min, λ_max] should be consistent with the
    /// provided condition number estimate.
    public let hamiltonian: Observable

    /// Number of qubits in the system register encoding the matrix dimension.
    ///
    /// The matrix A acts on a 2^systemQubits dimensional Hilbert space.
    public let systemQubits: Int

    /// Right-hand side vector |b⟩ encoded as a normalized quantum state.
    ///
    /// Must have exactly systemQubits qubits and be normalized. The HHL algorithm produces
    /// |x⟩ ∝ A⁻¹|b⟩ as output.
    public let rightHandSide: QuantumState

    /// Estimated condition number κ = λ_max/λ_min of the matrix A.
    ///
    /// Controls the polynomial degree for eigenvalue inversion and the success probability
    /// of the algorithm. Overestimating κ increases circuit depth but maintains correctness;
    /// underestimating can produce incorrect results.
    public let conditionNumber: Double

    /// Creates a quantum linear system problem.
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [(1.0, PauliString(.z(0)))])
    /// let b = QuantumState(qubits: 1)
    /// let problem = HHLProblem(
    ///     hamiltonian: H, systemQubits: 1,
    ///     rightHandSide: b, conditionNumber: 2.0
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - hamiltonian: Hermitian matrix A as Observable
    ///   - systemQubits: Number of system qubits
    ///   - rightHandSide: Right-hand side |b⟩
    ///   - conditionNumber: Estimated condition number κ
    /// - Precondition: systemQubits > 0
    /// - Precondition: systemQubits <= 8
    /// - Precondition: conditionNumber > 0
    /// - Complexity: O(1)
    @inlinable
    public init(
        hamiltonian: Observable,
        systemQubits: Int,
        rightHandSide: QuantumState,
        conditionNumber: Double,
    ) {
        ValidationUtilities.validatePositiveQubits(systemQubits)
        ValidationUtilities.validateAlgorithmQubitLimit(systemQubits, max: 8, algorithmName: "HHL")
        ValidationUtilities.validateStateQubitCount(rightHandSide, required: systemQubits)
        ValidationUtilities.validatePositiveDouble(conditionNumber, name: "conditionNumber")

        self.hamiltonian = hamiltonian
        self.systemQubits = systemQubits
        self.rightHandSide = rightHandSide
        self.conditionNumber = conditionNumber
    }
}

/// Method selection for the HHL quantum linear systems algorithm.
///
/// Two implementations are available with different performance characteristics.
/// The QPE-based method uses quantum phase estimation for eigenvalue extraction and is
/// primarily pedagogical. The QSVT-based method uses quantum signal processing for
/// direct eigenvalue inversion and achieves optimal query complexity.
///
/// **Example:**
/// ```swift
/// let qpeMethod = HHLMethod.qpe(precisionQubits: 6)
/// let qsvtMethod = HHLMethod.qsvt(epsilon: 1e-6)
/// ```
///
/// - SeeAlso: ``HHLAlgorithm``
@frozen
public enum HHLMethod: Sendable, Equatable {
    /// QPE-based HHL using eigenvalue estimation, controlled Ry rotation, and inverse QPE.
    ///
    /// Complexity: O(κ² s log(n)/ε) where s is sparsity, n is dimension, κ is condition
    /// number. The precision qubits determine eigenvalue resolution as ⌈log₂(κ/ε)⌉.
    case qpe(precisionQubits: Int)

    /// QSVT-based HHL using block encoding and quantum signal processing.
    ///
    /// Applies the inverse function polynomial 1/x via QSP phase angles to the
    /// block-encoded Hamiltonian. Complexity: O(κ s polylog(κ/ε)), exponentially better
    /// in precision and quadratically better in condition number than QPE-based.
    case qsvt(epsilon: Double)
}

/// Result of the HHL quantum linear systems algorithm.
///
/// Contains the solution quantum state |x⟩ ∝ A⁻¹|b⟩, the success probability of
/// post-selection on the ancilla qubit, the number of oracle calls made, and the
/// method used. The success probability determines how many repetitions are needed
/// to obtain the solution state on a real quantum device.
///
/// **Example:**
/// ```swift
/// let hhl = HHLAlgorithm(problem: problem)
/// let result = await hhl.solve(method: .qsvt(epsilon: 1e-4))
/// print(result.solutionState.qubits)
/// print(result.successProbability)
/// print(result.oracleCalls)
/// ```
///
/// - SeeAlso: ``HHLAlgorithm``
/// - SeeAlso: ``HHLProblem``
@frozen
public struct HHLResult: Sendable, CustomStringConvertible {
    /// Solution quantum state |x⟩ ∝ A⁻¹|b⟩.
    ///
    /// The normalized quantum state encoding the solution to the linear system.
    /// Classical extraction of the full solution vector requires O(2^n) measurements.
    public let solutionState: QuantumState

    /// Probability of successful post-selection on the ancilla qubit.
    ///
    /// On a real quantum device, the algorithm must be repeated O(1/successProbability)
    /// times to obtain the solution state. Scales as O(1/κ²) for condition number κ.
    public let successProbability: Double

    /// Number of oracle (walk operator) calls made during execution.
    ///
    /// For QPE-based: O(2^precisionQubits). For QSVT-based: equals the QSP polynomial degree.
    public let oracleCalls: Int

    /// Method used for this solve invocation.
    public let method: HHLMethod

    /// Creates an HHL result.
    ///
    /// **Example:**
    /// ```swift
    /// let result = HHLResult(
    ///     solutionState: QuantumState(qubits: 2),
    ///     successProbability: 0.25,
    ///     oracleCalls: 64,
    ///     method: .qpe(precisionQubits: 6)
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - solutionState: Normalized solution state
    ///   - successProbability: Post-selection success probability
    ///   - oracleCalls: Number of oracle invocations
    ///   - method: HHL method used
    /// - Complexity: O(1)
    @inlinable
    public init(
        solutionState: QuantumState,
        successProbability: Double,
        oracleCalls: Int,
        method: HHLMethod,
    ) {
        self.solutionState = solutionState
        self.successProbability = successProbability
        self.oracleCalls = oracleCalls
        self.method = method
    }

    /// Formatted description of the HHL result.
    ///
    /// **Example:**
    /// ```swift
    /// let result = HHLResult(
    ///     solutionState: QuantumState(qubits: 1),
    ///     successProbability: 0.5,
    ///     oracleCalls: 32,
    ///     method: .qsvt(epsilon: 1e-4)
    /// )
    /// print(result.description)
    /// ```
    @inlinable
    public var description: String {
        let methodStr = switch method {
        case let .qpe(n): "QPE(\(n) precision qubits)"
        case let .qsvt(eps): "QSVT(epsilon=\(String(format: "%.2e", eps)))"
        }
        return """
        HHL Result:
          Method: \(methodStr)
          Solution Qubits: \(solutionState.qubits)
          Success Probability: \(String(format: "%.6f", successProbability))
          Oracle Calls: \(oracleCalls)
        """
    }
}

/// Quantum linear systems solver implementing the Harrow-Hassidim-Lloyd (HHL) algorithm.
///
/// Solves Ax = b where A is a Hermitian matrix (given as an ``Observable``) and b is a
/// quantum state, producing |x⟩ ∝ A⁻¹|b⟩. Two implementations are available:
///
/// The QPE-based method (pedagogical) performs quantum phase estimation on the qubitized
/// walk operator to extract eigenvalues, applies controlled Ry rotations for eigenvalue
/// inversion, then uncomputes with inverse QPE. Complexity: O(κ² s log(n)/ε).
///
/// The QSVT-based method (production) block-encodes A via LCU decomposition and applies
/// the inverse function polynomial through quantum signal processing phase angles from
/// ``QSPPolynomialTarget/inverseFunction(condition:)``. This achieves optimal complexity
/// O(κ s polylog(κ/ε)), exponentially better in precision.
///
/// **Example:**
/// ```swift
/// let hamiltonian = Observable(terms: [
///     (0.5, PauliString(.z(0))),
///     (-0.3, PauliString(.x(1)))
/// ])
/// let rhs = QuantumState(qubits: 2)
/// let problem = HHLProblem(
///     hamiltonian: hamiltonian, systemQubits: 2,
///     rightHandSide: rhs, conditionNumber: 4.0
/// )
/// let hhl = HHLAlgorithm(problem: problem)
/// let result = await hhl.solve(method: .qsvt(epsilon: 1e-4))
/// print(result.successProbability)
/// ```
///
/// - Complexity: QPE: O(κ² s log(n)/ε), QSVT: O(κ s polylog(κ/ε))
/// - SeeAlso: ``HHLProblem``
/// - SeeAlso: ``HHLResult``
/// - SeeAlso: ``HHLMethod``
/// - SeeAlso: ``Qubitization``
public actor HHLAlgorithm {
    private static let amplitudeThreshold = 1e-15

    private let problem: HHLProblem
    private let blockEncoding: BlockEncoding
    private let walkOperator: QubitizedWalkOperator

    /// Creates an HHL solver for the given linear system problem.
    ///
    /// Constructs the block encoding and walk operator from the problem Hamiltonian,
    /// which are reused across multiple solve invocations with different methods.
    ///
    /// **Example:**
    /// ```swift
    /// let problem = HHLProblem(
    ///     hamiltonian: Observable(terms: [(1.0, PauliString(.z(0)))]),
    ///     systemQubits: 1, rightHandSide: QuantumState(qubits: 1),
    ///     conditionNumber: 2.0
    /// )
    /// let hhl = HHLAlgorithm(problem: problem)
    /// ```
    ///
    /// - Parameter problem: Linear system specification
    /// - Complexity: O(L) where L is the number of Hamiltonian terms
    public init(problem: HHLProblem) {
        self.problem = problem
        blockEncoding = BlockEncoding(hamiltonian: problem.hamiltonian, systemQubits: problem.systemQubits)
        walkOperator = QubitizedWalkOperator(blockEncoding: blockEncoding)
    }

    /// Solves the linear system Ax = b using the specified method.
    ///
    /// Dispatches to either QPE-based or QSVT-based implementation based on the method
    /// parameter. The QPE method provides a pedagogical implementation while QSVT achieves
    /// optimal query complexity.
    ///
    /// **Example:**
    /// ```swift
    /// let hhl = HHLAlgorithm(problem: problem)
    /// let result = await hhl.solve(method: .qsvt(epsilon: 1e-4), progress: { msg in
    ///     print(msg)
    /// })
    /// print(result.solutionState)
    /// ```
    ///
    /// - Parameters:
    ///   - method: Algorithm variant to use
    ///   - progress: Optional callback for status updates during execution
    /// - Returns: HHL result with solution state and metrics
    /// - Precondition: For QPE method, precisionQubits > 0 and <= 10
    /// - Precondition: For QSVT method, epsilon > 0
    /// - Complexity: QPE: O(κ² s log(n)/ε), QSVT: O(κ s polylog(κ/ε))
    @_optimize(speed)
    public func solve(
        method: HHLMethod,
        progress: (@Sendable (String) async -> Void)? = nil,
    ) async -> HHLResult {
        switch method {
        case let .qpe(precisionQubits):
            ValidationUtilities.validatePositiveInt(precisionQubits, name: "precisionQubits")
            ValidationUtilities.validateUpperBound(precisionQubits, max: 10, name: "precisionQubits")
            return await solveQPE(precisionQubits: precisionQubits, progress: progress)

        case let .qsvt(epsilon):
            ValidationUtilities.validatePositiveDouble(epsilon, name: "epsilon")
            return await solveQSVT(epsilon: epsilon, progress: progress)
        }
    }

    /// Solves using QPE-based eigenvalue inversion.
    @_optimize(speed)
    private func solveQPE(
        precisionQubits: Int,
        progress: (@Sendable (String) async -> Void)?,
    ) async -> HHLResult {
        await progress?("HHL QPE: building walk operator circuit")

        let n = precisionQubits
        let s = problem.systemQubits
        let alpha = blockEncoding.configuration.oneNorm
        let lcuAncillaCount = blockEncoding.configuration.ancillaQubits
        let totalBlockQubits = s + lcuAncillaCount
        let hhlAncillaQubit = n + totalBlockQubits
        let totalQubits = hhlAncillaQubit + 1

        let basicWalkCircuit = walkOperator.buildWalkCircuit()
        let controlledOpsPerQubit = buildControlledWalkOps(
            walkCircuit: basicWalkCircuit,
            precisionBits: n,
        )

        await progress?("HHL QPE: constructing QPE circuit with \(n) precision qubits")

        var qpeCircuit = QuantumCircuit(qubits: totalQubits)

        for i in 0 ..< n {
            qpeCircuit.append(.hadamard, to: i)
        }

        for k in 0 ..< n {
            let controlQubit = n - 1 - k
            let power = 1 << k
            let controlledOps = controlledOpsPerQubit[controlQubit]

            for _ in 0 ..< power {
                for (g, q) in controlledOps {
                    qpeCircuit.append(g, to: q)
                }
            }
        }

        let inverseQFTCircuit = QuantumCircuit.inverseQFT(qubits: n)
        for op in inverseQFTCircuit.operations {
            qpeCircuit.append(op)
        }

        await progress?("HHL QPE: preparing initial state")

        let fullInitial = prepareQPEInitialState(
            totalQubits: totalQubits,
            totalBlockQubits: totalBlockQubits,
        )

        await progress?("HHL QPE: executing QPE circuit")

        let stateAfterQPE = qpeCircuit.execute(on: fullInitial)

        await progress?("HHL QPE: applying eigenvalue inversion rotations")

        let stateAfterInversion = applyEigenvalueInversion(
            state: stateAfterQPE,
            precisionQubits: n,
            hhlAncillaQubit: hhlAncillaQubit,
            alpha: alpha,
        )

        await progress?("HHL QPE: executing inverse QPE")

        let inverseQPECircuit = qpeCircuit.inversed()
        let finalState = inverseQPECircuit.execute(on: stateAfterInversion)

        await progress?("HHL QPE: extracting solution state")

        let (solutionState, successProb) = extractQPESolution(
            state: finalState,
            precisionQubits: n,
            systemQubits: s,
            lcuAncillaQubits: lcuAncillaCount,
            hhlAncillaQubit: hhlAncillaQubit,
        )

        let oracleCalls = 2 * ((1 << n) - 1)

        await progress?(
            "HHL QPE complete: success probability = \(String(format: "%.6f", successProb))",
        )

        return HHLResult(
            solutionState: solutionState,
            successProbability: successProb,
            oracleCalls: oracleCalls,
            method: .qpe(precisionQubits: n),
        )
    }

    /// Solves using QSVT-based polynomial eigenvalue inversion.
    @_optimize(speed)
    private func solveQSVT(
        epsilon: Double,
        progress: (@Sendable (String) async -> Void)?,
    ) async -> HHLResult {
        await progress?("HHL QSVT: computing inverse function QSP phase angles")

        let kappa = problem.conditionNumber
        let degree = max(1, Int(ceil(kappa * Foundation.log(2.0 * kappa / epsilon))))

        let phaseAngles = QuantumSignalProcessing.computePhaseAngles(
            for: .inverseFunction(condition: kappa),
            degree: degree,
            epsilon: epsilon,
        )

        await progress?(
            "HHL QSVT: building QSP circuit with degree \(phaseAngles.polynomialDegree)",
        )

        let signalQubit = problem.systemQubits
        let qspCircuit = QuantumSignalProcessing.buildCircuit(
            walkOperator: walkOperator,
            phaseAngles: phaseAngles,
            signalQubit: signalQubit,
        )

        let totalBlockQubits = blockEncoding.totalQubits

        await progress?("HHL QSVT: preparing initial state")

        var extendedInitial = extendState(problem.rightHandSide, toQubits: totalBlockQubits)
        let prepareCircuit = blockEncoding.prepareCircuit()
        extendedInitial = prepareCircuit.execute(on: extendedInitial)

        await progress?("HHL QSVT: executing QSP circuit")

        let evolvedExtended = qspCircuit.execute(on: extendedInitial)

        let solutionState = projectToSystemQubits(state: evolvedExtended)
        let successProb = computeAncillaSuccessProbability(state: evolvedExtended)

        let actualDegree = phaseAngles.polynomialDegree

        await progress?(
            "HHL QSVT complete: success probability = \(String(format: "%.6f", successProb))",
        )

        return HHLResult(
            solutionState: solutionState,
            successProbability: successProb,
            oracleCalls: actualDegree,
            method: .qsvt(epsilon: epsilon),
        )
    }

    /// Builds controlled walk operator gate sequences for each precision qubit.
    @_optimize(speed)
    @_effects(readonly)
    private func buildControlledWalkOps(
        walkCircuit: QuantumCircuit,
        precisionBits: Int,
    ) -> [[(gate: QuantumGate, qubits: [Int])]] {
        var result = [[(gate: QuantumGate, qubits: [Int])]]()
        result.reserveCapacity(precisionBits)

        for controlQubit in 0 ..< precisionBits {
            var controlledOps = [(gate: QuantumGate, qubits: [Int])]()

            for op in walkCircuit.operations {
                if case let .gate(g, qubits, _) = op {
                    let shiftedQubits = [Int](unsafeUninitializedCapacity: qubits.count) {
                        buffer, count in
                        for i in 0 ..< qubits.count {
                            buffer[i] = qubits[i] + precisionBits
                        }
                        count = qubits.count
                    }

                    if shiftedQubits.count == 1 {
                        let decomposition = ControlledGateDecomposer.decompose(
                            gate: g,
                            controls: [controlQubit],
                            target: shiftedQubits[0],
                        )
                        for (dg, dq) in decomposition {
                            controlledOps.append((gate: dg, qubits: dq))
                        }
                    } else if g == .cnot {
                        controlledOps.append((
                            gate: .toffoli,
                            qubits: [controlQubit, shiftedQubits[0], shiftedQubits[1]],
                        ))
                    } else if g == .cz {
                        controlledOps.append((gate: .hadamard, qubits: [shiftedQubits[1]]))
                        controlledOps.append((
                            gate: .toffoli,
                            qubits: [controlQubit, shiftedQubits[0], shiftedQubits[1]],
                        ))
                        controlledOps.append((gate: .hadamard, qubits: [shiftedQubits[1]]))
                    }
                }
            }

            result.append(controlledOps)
        }

        return result
    }

    /// Prepares the initial state for QPE-based HHL.
    @_optimize(speed)
    @_eagerMove
    private func prepareQPEInitialState(
        totalQubits: Int,
        totalBlockQubits: Int,
    ) -> QuantumState {
        let extendedB = extendState(problem.rightHandSide, toQubits: totalBlockQubits)
        let preparedB = blockEncoding.prepareCircuit().execute(on: extendedB)

        var fullInitial = QuantumState(qubits: totalQubits)
        let stateLimit = min(preparedB.stateSpaceSize, fullInitial.stateSpaceSize)
        for i in 0 ..< stateLimit {
            let amplitude = preparedB.amplitude(of: i)
            if amplitude.magnitudeSquared > Self.amplitudeThreshold {
                fullInitial.setAmplitude(i, to: amplitude)
            }
        }
        fullInitial.normalize()

        return fullInitial
    }

    /// Applies eigenvalue inversion via Ry rotation directly on amplitudes.
    @_optimize(speed)
    @_eagerMove
    private func applyEigenvalueInversion(
        state: QuantumState,
        precisionQubits: Int,
        hhlAncillaQubit: Int,
        alpha: Double,
    ) -> QuantumState {
        let precisionStateSize = 1 << precisionQubits
        let precisionMask = precisionStateSize - 1
        let hhlAncillaBit = 1 << hhlAncillaQubit
        let inversionScale = alpha / problem.conditionNumber
        let eigenvalueThreshold = inversionScale * 0.01

        var amplitudes = state.amplitudes

        for basisIndex in stride(from: 0, to: state.stateSpaceSize, by: 1) {
            let ancillaBitValue = (basisIndex >> hhlAncillaQubit) & 1
            guard ancillaBitValue == 0 else { continue }

            let precisionIndex = basisIndex & precisionMask
            let phase = Double(precisionIndex) / Double(precisionStateSize)
            let eigenvalueEstimate = alpha * Foundation.cos(2.0 * .pi * phase)
            let absEigenvalue = abs(eigenvalueEstimate)

            guard absEigenvalue > eigenvalueThreshold else { continue }

            let ratio = min(1.0, inversionScale / absEigenvalue)
            let angle = 2.0 * Foundation.asin(ratio)
            let cosHalf = Foundation.cos(angle * 0.5)
            let sinHalf = Foundation.sin(angle * 0.5)

            let index0 = basisIndex
            let index1 = basisIndex | hhlAncillaBit

            let a0 = amplitudes[index0]
            let a1 = amplitudes[index1]

            amplitudes[index0] = Complex(
                cosHalf * a0.real - sinHalf * a1.real,
                cosHalf * a0.imaginary - sinHalf * a1.imaginary,
            )
            amplitudes[index1] = Complex(
                sinHalf * a0.real + cosHalf * a1.real,
                sinHalf * a0.imaginary + cosHalf * a1.imaginary,
            )
        }

        return QuantumState(qubits: state.qubits, amplitudes: amplitudes)
    }

    /// Extracts solution state from QPE-based HHL output.
    @_optimize(speed)
    @_eagerMove
    private func extractQPESolution(
        state: QuantumState,
        precisionQubits: Int,
        systemQubits: Int,
        lcuAncillaQubits: Int,
        hhlAncillaQubit: Int,
    ) -> (solutionState: QuantumState, successProbability: Double) {
        let precisionMask = (1 << precisionQubits) - 1
        let systemMask = (1 << systemQubits) - 1
        let lcuAncillaMask = (1 << lcuAncillaQubits) - 1
        let systemSize = 1 << systemQubits

        var successProb = 0.0
        for basisIndex in 0 ..< state.stateSpaceSize {
            if (basisIndex >> hhlAncillaQubit) & 1 == 1 {
                successProb += state.amplitude(of: basisIndex).magnitudeSquared
            }
        }

        var solutionAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: systemSize) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = systemSize
        }

        for basisIndex in 0 ..< state.stateSpaceSize {
            let hhlAncillaValue = (basisIndex >> hhlAncillaQubit) & 1
            guard hhlAncillaValue == 1 else { continue }

            let precisionValue = basisIndex & precisionMask
            guard precisionValue == 0 else { continue }

            let systemValue = (basisIndex >> precisionQubits) & systemMask
            let lcuAncillaValue = (basisIndex >> (precisionQubits + systemQubits)) & lcuAncillaMask
            guard lcuAncillaValue == 0 else { continue }

            let amp = state.amplitude(of: basisIndex)
            solutionAmplitudes[systemValue] = Complex(
                solutionAmplitudes[systemValue].real + amp.real,
                solutionAmplitudes[systemValue].imaginary + amp.imaginary,
            )
        }

        var solutionState = QuantumState(qubits: systemQubits, amplitudes: solutionAmplitudes)

        var normSq = 0.0
        for i in 0 ..< systemSize {
            normSq += solutionAmplitudes[i].magnitudeSquared
        }
        if normSq > Self.amplitudeThreshold {
            solutionState.normalize()
        }

        return (solutionState, successProb)
    }

    /// Extends a system state by adding ancilla qubits initialized to |0⟩.
    @_optimize(speed)
    @_eagerMove
    private func extendState(_ state: QuantumState, toQubits totalQubits: Int) -> QuantumState {
        let newSize = 1 << totalQubits
        let oldSize = state.stateSpaceSize

        let newAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: newSize) {
            buffer, count in
            for i in 0 ..< oldSize {
                buffer[i] = state.amplitudes[i]
            }
            for i in oldSize ..< newSize {
                buffer[i] = .zero
            }
            count = newSize
        }

        return QuantumState(qubits: totalQubits, amplitudes: newAmplitudes)
    }

    /// Projects extended state back to system qubits by partial trace over ancillas.
    @_optimize(speed)
    @_eagerMove
    private func projectToSystemQubits(state: QuantumState) -> QuantumState {
        let systemQubits = problem.systemQubits
        let systemSize = 1 << systemQubits
        let ancillaSize = state.stateSpaceSize / systemSize

        var projectedAmplitudes = [Complex<Double>](unsafeUninitializedCapacity: systemSize) {
            buffer, count in
            buffer.initialize(repeating: .zero)
            count = systemSize
        }

        for i in 0 ..< systemSize {
            var sumSquared = 0.0
            for a in 0 ..< ancillaSize {
                let fullIndex = i + a * systemSize
                if fullIndex < state.stateSpaceSize {
                    sumSquared += state.amplitude(of: fullIndex).magnitudeSquared
                }
            }
            let amplitude = state.amplitude(of: i)
            let norm = Foundation.sqrt(sumSquared)
            if norm > Self.amplitudeThreshold {
                let ampMag = Foundation.sqrt(max(amplitude.magnitudeSquared, Self.amplitudeThreshold))
                let invMag = 1.0 / ampMag
                projectedAmplitudes[i] = Complex(
                    norm * amplitude.real * invMag,
                    norm * amplitude.imaginary * invMag,
                )
            }
        }

        return QuantumState(qubits: systemQubits, amplitudes: projectedAmplitudes)
    }

    /// Computes post-selection success probability from extended state.
    @_optimize(speed)
    @_effects(readonly)
    private func computeAncillaSuccessProbability(state: QuantumState) -> Double {
        let systemSize = 1 << problem.systemQubits

        var successProb = 0.0
        for i in 0 ..< systemSize {
            successProb += state.amplitude(of: i).magnitudeSquared
        }

        return successProb
    }
}

/// Complexity analysis for HHL quantum linear systems algorithm.
///
/// Provides theoretical bounds and resource estimates for both QPE-based and QSVT-based
/// HHL implementations. Use for algorithm selection and circuit resource planning.
///
/// **Example:**
/// ```swift
/// let qpeAnalysis = HHLComplexity.analyzeQPE(
///     conditionNumber: 10.0, sparsity: 4,
///     dimension: 16, targetError: 1e-3
/// )
/// print("QPE precision qubits: \(qpeAnalysis.precisionQubits)")
/// print("QPE oracle calls: \(qpeAnalysis.oracleCalls)")
/// ```
///
/// - SeeAlso: ``HHLAlgorithm``
public enum HHLComplexity {
    /// Analyzes QPE-based HHL complexity.
    ///
    /// Computes required precision qubits as ⌈log₂(κ/ε)⌉ and total oracle calls
    /// as O(κ² s log(n)/ε) for the QPE-based implementation.
    ///
    /// **Example:**
    /// ```swift
    /// let analysis = HHLComplexity.analyzeQPE(
    ///     conditionNumber: 5.0, sparsity: 2,
    ///     dimension: 8, targetError: 0.01
    /// )
    /// print(analysis.precisionQubits)
    /// ```
    ///
    /// - Parameters:
    ///   - conditionNumber: Condition number κ of matrix A
    ///   - sparsity: Row sparsity s of matrix A
    ///   - dimension: Matrix dimension n = 2^systemQubits
    ///   - targetError: Target precision ε
    /// - Returns: Tuple with precision qubits, oracle calls, total qubits, and success probability
    /// - Precondition: conditionNumber > 0
    /// - Precondition: sparsity > 0
    /// - Precondition: dimension > 0
    /// - Precondition: targetError > 0
    /// - Complexity: O(1)
    @_effects(readonly)
    public static func analyzeQPE(
        conditionNumber: Double,
        sparsity: Int,
        dimension: Int,
        targetError: Double,
    ) -> (precisionQubits: Int, oracleCalls: Int, totalQubits: Int, successProbability: Double) {
        let kappa = conditionNumber
        let precisionQubits = max(1, Int(Foundation.ceil(Foundation.log2(kappa / targetError))))

        let s = Double(sparsity)
        let n = Double(dimension)
        let oracleCalls = Int(Foundation.ceil(kappa * kappa * s * Foundation.log2(n) / targetError))

        let systemQubits = max(1, Int(Foundation.ceil(Foundation.log2(n))))
        let ancillaQubits = max(1, Int(Foundation.ceil(Foundation.log2(s))))
        let totalQubits = precisionQubits + systemQubits + ancillaQubits + 1

        let successProbability = 1.0 / (kappa * kappa)

        return (precisionQubits, oracleCalls, totalQubits, successProbability)
    }

    /// Analyzes QSVT-based HHL complexity.
    ///
    /// Computes the QSP polynomial degree as O(κ log(1/ε)) and total oracle calls
    /// for the QSVT-based implementation with optimal query complexity.
    ///
    /// **Example:**
    /// ```swift
    /// let analysis = HHLComplexity.analyzeQSVT(
    ///     conditionNumber: 5.0, sparsity: 2,
    ///     targetError: 1e-6
    /// )
    /// print(analysis.polynomialDegree)
    /// ```
    ///
    /// - Parameters:
    ///   - conditionNumber: Condition number κ of matrix A
    ///   - sparsity: Row sparsity s of matrix A
    ///   - targetError: Target precision ε
    /// - Returns: Tuple with polynomial degree, oracle calls, and success probability
    /// - Precondition: conditionNumber > 0
    /// - Precondition: sparsity > 0
    /// - Precondition: targetError > 0
    /// - Complexity: O(1)
    @_effects(readonly)
    public static func analyzeQSVT(
        conditionNumber: Double,
        sparsity: Int,
        targetError: Double,
    ) -> (polynomialDegree: Int, oracleCalls: Int, successProbability: Double) {
        let kappa = conditionNumber
        let logFactor = Foundation.log(2.0 * kappa / targetError)
        let polynomialDegree = max(1, Int(Foundation.ceil(kappa * logFactor)))

        let s = Double(sparsity)
        let oracleCalls = Int(Foundation.ceil(kappa * s * logFactor))

        let successProbability = 1.0 / (kappa * kappa)

        return (polynomialDegree, oracleCalls, successProbability)
    }

    /// Computes the speedup factor of QSVT over QPE for HHL.
    ///
    /// QSVT achieves O(κ polylog(κ/ε)) vs QPE's O(κ²/ε), providing exponential
    /// improvement in precision and quadratic improvement in condition number.
    ///
    /// **Example:**
    /// ```swift
    /// let speedup = HHLComplexity.qsvtSpeedupOverQPE(
    ///     conditionNumber: 10.0, targetError: 1e-6
    /// )
    /// print("QSVT speedup: \(String(format: "%.1f", speedup))x")
    /// ```
    ///
    /// - Parameters:
    ///   - conditionNumber: Condition number κ
    ///   - targetError: Target precision ε
    /// - Returns: Ratio of QPE oracle calls to QSVT oracle calls
    /// - Precondition: conditionNumber > 0
    /// - Precondition: targetError > 0
    /// - Complexity: O(1)
    @_effects(readonly)
    public static func qsvtSpeedupOverQPE(
        conditionNumber: Double,
        targetError: Double,
    ) -> Double {
        let kappa = conditionNumber
        let qpeQueries = kappa * kappa / targetError
        let logFactor = Foundation.log(2.0 * kappa / targetError)
        let qsvtQueries = kappa * logFactor

        return qpeQueries / qsvtQueries
    }
}
