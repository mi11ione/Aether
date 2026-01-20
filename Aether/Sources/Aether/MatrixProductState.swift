// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Statistics tracking truncation errors accumulated during MPS operations.
///
/// When bond dimensions are truncated via SVD, small singular values are discarded introducing
/// controlled approximation error. This struct accumulates these errors across operations for
/// monitoring simulation fidelity. Total error is bounded by the sum of individual truncation
/// errors (triangle inequality).
///
/// **Example:**
/// ```swift
/// var stats = MPSTruncationStatistics.zero
/// stats = stats.adding(error: 0.001)
/// stats = stats.adding(error: 0.002)
/// print(stats.cumulativeError)  // 0.003
/// print(stats.maxSingleError)   // 0.002
/// ```
///
/// - SeeAlso: ``MatrixProductState``
@frozen
public struct MPSTruncationStatistics: Sendable, Equatable, CustomStringConvertible {
    /// Number of truncation operations performed
    public let truncationCount: Int

    /// Sum of all truncation errors (upper bound on total error)
    public let cumulativeError: Double

    /// Maximum single truncation error encountered
    public let maxSingleError: Double

    /// Human-readable description of truncation statistics.
    ///
    /// **Example:**
    /// ```swift
    /// let stats = MPSTruncationStatistics.zero.adding(error: 0.001)
    /// print(stats.description)  // "MPSTruncationStatistics(count=1, cumulative=0.001, max=0.001)"
    /// ```
    public var description: String {
        "MPSTruncationStatistics(count=\(truncationCount), cumulative=\(cumulativeError), max=\(maxSingleError))"
    }

    /// Zero truncation statistics for fresh MPS states.
    ///
    /// **Example:**
    /// ```swift
    /// let stats = MPSTruncationStatistics.zero
    /// print(stats.truncationCount)  // 0
    /// ```
    public static var zero: MPSTruncationStatistics {
        MPSTruncationStatistics(truncationCount: 0, cumulativeError: 0.0, maxSingleError: 0.0)
    }

    /// Creates new statistics by adding a truncation error.
    ///
    /// Increments truncation count, adds error to cumulative total, and updates
    /// max single error if this error exceeds the current maximum.
    ///
    /// **Example:**
    /// ```swift
    /// let stats = MPSTruncationStatistics.zero
    /// let updated = stats.adding(error: 0.001)
    /// print(updated.truncationCount)  // 1
    /// ```
    ///
    /// - Parameter error: Truncation error from SVD (sum of discarded squared singular values)
    /// - Returns: New statistics with error incorporated
    /// - Complexity: O(1)
    @_effects(readonly)
    public func adding(error: Double) -> MPSTruncationStatistics {
        MPSTruncationStatistics(
            truncationCount: truncationCount + 1,
            cumulativeError: cumulativeError + error,
            maxSingleError: max(maxSingleError, error),
        )
    }
}

/// Matrix Product State representation of quantum states for efficient simulation.
///
/// Represents an n-qubit quantum state as a chain of rank-3 tensors: |psi> = A[0] * A[1] * ... * A[n-1]
/// where each tensor A[k] has indices (left bond, physical, right bond). Memory scales as O(n * chi^2)
/// where chi is the maximum bond dimension, compared to O(2^n) for full statevector. This enables
/// simulation of weakly-entangled states with hundreds of qubits.
///
/// Bond dimension chi controls the trade-off between accuracy and efficiency. For product states chi=1
/// suffices. For entangled states, chi grows exponentially with entanglement entropy (area law for
/// ground states of gapped local Hamiltonians). Truncating to maxBondDimension introduces controlled
/// approximation tracked by ``truncationStatistics``.
///
/// **Example:**
/// ```swift
/// let mps = MatrixProductState(qubits: 100, maxBondDimension: 64)
/// let amplitude = mps.amplitude(of: 0)
/// print(mps.memoryUsage)
///
/// let bell = QuantumCircuit.bell().execute()
/// let mpsBell = MatrixProductState(from: bell, maxBondDimension: 16)
/// print(mpsBell.amplitude(of: 0b00))  // 1/sqrt(2)
/// ```
///
/// - SeeAlso: ``MPSTensor``
/// - SeeAlso: ``MPSTruncationStatistics``
@frozen
public struct MatrixProductState: Sendable, Equatable, CustomStringConvertible {
    /// Number of qubits in the system.
    ///
    /// **Example:**
    /// ```swift
    /// let mps = MatrixProductState(qubits: 10)
    /// print(mps.qubits)  // 10
    /// ```
    public let qubits: Int

    /// Maximum allowed bond dimension for truncation.
    ///
    /// Limits the growth of bond dimensions during entangling operations. Higher values
    /// increase accuracy but also memory and computation time.
    ///
    /// **Example:**
    /// ```swift
    /// let mps = MatrixProductState(qubits: 10, maxBondDimension: 32)
    /// print(mps.maxBondDimension)  // 32
    /// ```
    public let maxBondDimension: Int

    /// Array of MPS tensors forming the chain.
    ///
    /// Contains exactly ``qubits`` tensors, one per site. First tensor has left bond dimension 1,
    /// last tensor has right bond dimension 1 (boundary conditions).
    ///
    /// **Example:**
    /// ```swift
    /// let mps = MatrixProductState(qubits: 4)
    /// print(mps.tensors.count)  // 4
    /// ```
    public private(set) var tensors: [MPSTensor]

    /// Accumulated truncation error statistics.
    ///
    /// Tracks errors introduced by bond dimension truncation during operations.
    ///
    /// **Example:**
    /// ```swift
    /// let mps = MatrixProductState(qubits: 10)
    /// print(mps.truncationStatistics.cumulativeError)  // 0.0 for exact states
    /// ```
    public private(set) var truncationStatistics: MPSTruncationStatistics

    /// Creates MPS for ground state |00...0>.
    ///
    /// Initializes all tensors for the computational basis state with all qubits in |0>.
    /// Product states have bond dimension 1, requiring minimal memory.
    ///
    /// **Example:**
    /// ```swift
    /// let mps = MatrixProductState(qubits: 100)
    /// print(mps.amplitude(of: 0))  // (1.0, 0.0)
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (1 to 1000)
    ///   - maxBondDimension: Maximum bond dimension limit (default: 64)
    /// - Complexity: O(n) where n = qubits
    /// - Precondition: 1 <= qubits <= 1000
    public init(qubits: Int, maxBondDimension: Int = 64) {
        ValidationUtilities.validateMPSQubitCount(qubits)
        ValidationUtilities.validatePositiveInt(maxBondDimension, name: "Max bond dimension")

        self.qubits = qubits
        self.maxBondDimension = maxBondDimension
        truncationStatistics = .zero

        tensors = [MPSTensor](unsafeUninitializedCapacity: qubits) { buffer, count in
            for site in 0 ..< qubits {
                buffer[site] = MPSTensor.groundState(site: site, qubits: qubits, maxBondDimension: maxBondDimension)
            }
            count = qubits
        }
    }

    /// Creates MPS for computational basis state |k>.
    ///
    /// Initializes tensors for the basis state where the binary representation of k determines
    /// which qubits are in |1>. Uses little-endian ordering where bit 0 corresponds to qubit 0.
    ///
    /// **Example:**
    /// ```swift
    /// let mps = MatrixProductState(qubits: 4, basisState: 0b1010)
    /// print(mps.amplitude(of: 0b1010))  // (1.0, 0.0)
    /// print(mps.amplitude(of: 0b0000))  // (0.0, 0.0)
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits (1 to 1000)
    ///   - basisState: Integer encoding the basis state (0 to 2^qubits - 1)
    ///   - maxBondDimension: Maximum bond dimension limit (default: 64)
    /// - Complexity: O(n) where n = qubits
    /// - Precondition: 1 <= qubits <= 1000, 0 <= basisState < 2^qubits
    public init(qubits: Int, basisState: Int, maxBondDimension: Int = 64) {
        ValidationUtilities.validateMPSQubitCount(qubits)
        ValidationUtilities.validatePositiveInt(maxBondDimension, name: "Max bond dimension")
        ValidationUtilities.validateNonNegativeInt(basisState, name: "Basis state")

        self.qubits = qubits
        self.maxBondDimension = maxBondDimension
        truncationStatistics = .zero

        tensors = [MPSTensor](unsafeUninitializedCapacity: qubits) { buffer, count in
            for site in 0 ..< qubits {
                buffer[site] = MPSTensor.basisState(basisState, site: site, qubits: qubits, maxBondDimension: maxBondDimension)
            }
            count = qubits
        }
    }

    /// Creates MPS from a full statevector via successive SVD decomposition.
    ///
    /// Decomposes the 2^n amplitude vector into an MPS chain using SVD at each bond.
    /// Singular values below threshold are truncated to limit bond dimension. This is
    /// exact for states with low entanglement but approximate for highly entangled states.
    ///
    /// **Example:**
    /// ```swift
    /// let bell = QuantumCircuit.bell().execute()
    /// let mps = MatrixProductState(from: bell, maxBondDimension: 16)
    /// print(mps.amplitude(of: 0b00))  // ~0.707
    /// ```
    ///
    /// - Parameters:
    ///   - state: Full quantum statevector to decompose
    ///   - maxBondDimension: Maximum bond dimension (default: 64)
    /// - Complexity: O(n * chi^3) where chi = min(2^n, maxBondDimension)
    /// - Precondition: state.qubits <= 20 (memory limit for intermediate computations)
    public init(from state: QuantumState, maxBondDimension: Int = 64) {
        ValidationUtilities.validateMPSToStatevectorLimit(state.qubits)
        ValidationUtilities.validatePositiveInt(maxBondDimension, name: "Max bond dimension")

        qubits = state.qubits
        self.maxBondDimension = maxBondDimension
        var stats = MPSTruncationStatistics.zero

        if qubits == 1 {
            let elements: [Complex<Double>] = [state.amplitudes[0], state.amplitudes[1]]
            tensors = [MPSTensor(leftBondDimension: 1, rightBondDimension: 1, site: 0, elements: elements)]
            truncationStatistics = stats
            return
        }

        var mutableTensors = [MPSTensor]()
        mutableTensors.reserveCapacity(qubits)

        var remainder = state.amplitudes
        var leftDim = 1

        for site in 0 ..< qubits - 1 {
            let rightSize = 1 << (qubits - site - 1)
            let rows = leftDim * 2
            let cols = rightSize

            var matrix = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: cols), count: rows)
            for i in 0 ..< remainder.count {
                let row = i / rightSize
                let col = i % rightSize
                matrix[row][col] = remainder[i]
            }

            let (u, s, vt, truncError) = Self.truncatedSVD(matrix: matrix, maxRank: maxBondDimension)
            stats = stats.adding(error: truncError)

            let newRightDim = s.count
            let tensorElements = [Complex<Double>](unsafeUninitializedCapacity: leftDim * 2 * newRightDim) { buffer, count in
                for alpha in 0 ..< leftDim {
                    for physical in 0 ..< 2 {
                        for beta in 0 ..< newRightDim {
                            let rowIdx = alpha * 2 + physical
                            let flatIdx = alpha * (2 * newRightDim) + physical * newRightDim + beta
                            buffer[flatIdx] = u[rowIdx][beta]
                        }
                    }
                }
                count = leftDim * 2 * newRightDim
            }

            mutableTensors.append(MPSTensor(
                leftBondDimension: leftDim,
                rightBondDimension: newRightDim,
                site: site,
                elements: tensorElements,
            ))

            let newRemainder = [Complex<Double>](unsafeUninitializedCapacity: newRightDim * cols) { buffer, count in
                for i in 0 ..< newRightDim {
                    for j in 0 ..< cols {
                        buffer[i * cols + j] = s[i] * vt[i][j]
                    }
                }
                count = newRightDim * cols
            }

            remainder = newRemainder
            leftDim = newRightDim
        }

        let lastElements = [Complex<Double>](unsafeUninitializedCapacity: leftDim * 2) { buffer, count in
            for alpha in 0 ..< leftDim {
                buffer[alpha * 2] = remainder[alpha * 2]
                buffer[alpha * 2 + 1] = remainder[alpha * 2 + 1]
            }
            count = leftDim * 2
        }

        mutableTensors.append(MPSTensor(
            leftBondDimension: leftDim,
            rightBondDimension: 1,
            site: qubits - 1,
            elements: lastElements,
        ))

        tensors = mutableTensors
        truncationStatistics = stats
    }

    /// Computes the amplitude <basisState|psi> for a computational basis state.
    ///
    /// Contracts the MPS chain by selecting the physical index at each site according to
    /// the bits of basisState. Uses left-to-right contraction with vector-matrix products.
    ///
    /// **Example:**
    /// ```swift
    /// let mps = MatrixProductState(qubits: 4, basisState: 0b1010)
    /// let amp = mps.amplitude(of: 0b1010)  // (1.0, 0.0)
    /// let zero = mps.amplitude(of: 0b0000)  // (0.0, 0.0)
    /// ```
    ///
    /// - Parameter basisState: Integer encoding the basis state (little-endian)
    /// - Returns: Complex amplitude <basisState|psi>
    /// - Complexity: O(n * chi^2) where n = qubits, chi = max bond dimension
    /// - Precondition: 0 <= basisState < 2^qubits
    @_optimize(speed)
    @_effects(readonly)
    public func amplitude(of basisState: Int) -> Complex<Double> {
        ValidationUtilities.validateNonNegativeInt(basisState, name: "Basis state")

        var vector: [Complex<Double>] = [.one]

        for site in 0 ..< qubits {
            let bit = (basisState >> site) & 1
            let matrix = tensors[site].matrixForPhysicalIndex(bit)

            let newDim = matrix[0].count
            let newVector = [Complex<Double>](unsafeUninitializedCapacity: newDim) { buffer, count in
                for beta in 0 ..< newDim {
                    var sum: Complex<Double> = .zero
                    for alpha in 0 ..< vector.count {
                        sum = sum + vector[alpha] * matrix[alpha][beta]
                    }
                    buffer[beta] = sum
                }
                count = newDim
            }

            vector = newVector
        }

        return vector[0]
    }

    /// Computes the probability |<basisState|psi>|^2 for a computational basis state.
    ///
    /// **Example:**
    /// ```swift
    /// let mps = MatrixProductState(qubits: 4, basisState: 0b1010)
    /// let prob = mps.probability(of: 0b1010)  // 1.0
    /// ```
    ///
    /// - Parameter basisState: Integer encoding the basis state
    /// - Returns: Probability in [0, 1]
    /// - Complexity: O(n * chi^2)
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    public func probability(of basisState: Int) -> Double {
        amplitude(of: basisState).magnitudeSquared
    }

    /// Checks if the MPS is normalized within tolerance.
    ///
    /// **Example:**
    /// ```swift
    /// let mps = MatrixProductState(qubits: 4)
    /// print(mps.isNormalized())  // true
    /// ```
    ///
    /// - Returns: True if |<psi|psi> - 1| < 1e-10
    /// - Complexity: O(n * chi^3)
    @_effects(readonly)
    public func isNormalized() -> Bool {
        let norm = normSquared()
        return abs(norm - 1.0) < 1e-10
    }

    /// Computes the squared norm <psi|psi> using transfer matrix contraction.
    ///
    /// Contracts the MPS with its conjugate by building transfer matrices at each site
    /// and propagating from left to right.
    ///
    /// **Example:**
    /// ```swift
    /// let mps = MatrixProductState(qubits: 4)
    /// print(mps.normSquared())  // 1.0
    /// ```
    ///
    /// - Returns: Squared norm (should be 1.0 for normalized states)
    /// - Complexity: O(n * chi^3)
    @_optimize(speed)
    @_effects(readonly)
    public func normSquared() -> Double {
        var transfer: [[Complex<Double>]] = [[.one]]

        for site in 0 ..< qubits {
            let tensor = tensors[site]
            let leftDim = tensor.leftBondDimension
            let rightDim = tensor.rightBondDimension

            var newTransfer = [[Complex<Double>]](
                repeating: [Complex<Double>](repeating: .zero, count: rightDim),
                count: rightDim,
            )

            for alphaPrime in 0 ..< rightDim {
                for alpha in 0 ..< rightDim {
                    var sum: Complex<Double> = .zero
                    for betaPrime in 0 ..< leftDim {
                        for beta in 0 ..< leftDim {
                            for physical in 0 ..< 2 {
                                let aConj = tensor[beta, physical, alpha].conjugate
                                let a = tensor[betaPrime, physical, alphaPrime]
                                sum = sum + transfer[beta][betaPrime] * aConj * a
                            }
                        }
                    }
                    newTransfer[alpha][alphaPrime] = sum
                }
            }

            transfer = newTransfer
        }

        return transfer[0][0].real
    }

    /// Normalizes the MPS to have unit norm.
    ///
    /// Divides all elements of the first tensor by sqrt(normSquared()).
    ///
    /// **Example:**
    /// ```swift
    /// var mps = MatrixProductState(qubits: 4)
    /// mps.normalize()
    /// print(mps.isNormalized())  // true
    /// ```
    ///
    /// - Complexity: O(n * chi^3) for norm computation + O(chi^2) for rescaling
    public mutating func normalize() {
        let norm = normSquared()
        ValidationUtilities.validatePositiveDouble(norm, name: "MPS norm squared")

        let invNorm = 1.0 / sqrt(norm)
        let oldTensor = tensors[0]

        let newElements = [Complex<Double>](unsafeUninitializedCapacity: oldTensor.elements.count) { buffer, count in
            for i in 0 ..< oldTensor.elements.count {
                buffer[i] = oldTensor.elements[i] * invNorm
            }
            count = oldTensor.elements.count
        }

        tensors[0] = MPSTensor(
            leftBondDimension: oldTensor.leftBondDimension,
            rightBondDimension: oldTensor.rightBondDimension,
            site: 0,
            elements: newElements,
        )
    }

    /// Converts the MPS back to a full statevector.
    ///
    /// Reconstructs the 2^n amplitude vector by contracting all tensors. Only feasible
    /// for small systems due to exponential memory requirements.
    ///
    /// **Example:**
    /// ```swift
    /// let mps = MatrixProductState(qubits: 4, basisState: 5)
    /// let state = mps.toQuantumState()
    /// print(state.probability(of: 5))  // 1.0
    /// ```
    ///
    /// - Returns: Full quantum statevector
    /// - Complexity: O(2^n * n * chi^2)
    /// - Precondition: qubits <= 20
    @_eagerMove
    public func toQuantumState() -> QuantumState {
        ValidationUtilities.validateMPSToStatevectorLimit(qubits)

        let stateSpaceSize = 1 << qubits
        let amplitudes = [Complex<Double>](unsafeUninitializedCapacity: stateSpaceSize) { buffer, count in
            for basisState in 0 ..< stateSpaceSize {
                buffer[basisState] = amplitude(of: basisState)
            }
            count = stateSpaceSize
        }

        return QuantumState(qubits: qubits, amplitudes: amplitudes)
    }

    /// Computes expectation value <psi|P|psi> for a Pauli string operator.
    ///
    /// Uses transfer matrix method with Pauli insertions at relevant sites. For sites
    /// with identity, uses standard transfer matrix. For sites with X/Y/Z, inserts the
    /// Pauli matrix between conjugate and non-conjugate tensors.
    ///
    /// **Example:**
    /// ```swift
    /// let mps = MatrixProductState(qubits: 4)
    /// let zz = PauliString(.z(0), .z(1))
    /// let expectation = mps.expectationValue(of: zz)  // 1.0 for |0000>
    /// ```
    ///
    /// - Parameter pauliString: Pauli string operator
    /// - Returns: Real expectation value
    /// - Complexity: O(n * chi^3)
    @_optimize(speed)
    @_effects(readonly)
    public func expectationValue(of pauliString: PauliString) -> Double {
        var pauliMap = [Int: PauliBasis]()
        for op in pauliString.operators {
            ValidationUtilities.validateMPSSiteIndex(op.qubit, qubits: qubits)
            pauliMap[op.qubit] = op.basis
        }

        var transfer: [[Complex<Double>]] = [[.one]]

        for site in 0 ..< qubits {
            let tensor = tensors[site]
            let leftDim = tensor.leftBondDimension
            let rightDim = tensor.rightBondDimension

            let pauliAtSite = pauliMap[site]
            let pauliMatrix = Self.pauliMatrix(pauliAtSite)

            var newTransfer = [[Complex<Double>]](
                repeating: [Complex<Double>](repeating: .zero, count: rightDim),
                count: rightDim,
            )

            for alphaPrime in 0 ..< rightDim {
                for alpha in 0 ..< rightDim {
                    var sum: Complex<Double> = .zero
                    for betaPrime in 0 ..< leftDim {
                        for beta in 0 ..< leftDim {
                            for i in 0 ..< 2 {
                                for j in 0 ..< 2 {
                                    let aConj = tensor[beta, i, alpha].conjugate
                                    let a = tensor[betaPrime, j, alphaPrime]
                                    let pauli = pauliMatrix[i][j]
                                    sum = sum + transfer[beta][betaPrime] * aConj * pauli * a
                                }
                            }
                        }
                    }
                    newTransfer[alpha][alphaPrime] = sum
                }
            }

            transfer = newTransfer
        }

        return transfer[0][0].real
    }

    /// Computes expectation value <psi|O|psi> for a general observable.
    ///
    /// Sums weighted Pauli string expectation values: <O> = sum_i c_i <P_i>.
    ///
    /// **Example:**
    /// ```swift
    /// let mps = MatrixProductState(qubits: 2)
    /// let h = Observable(terms: [(0.5, PauliString(.z(0))), (0.5, PauliString(.z(1)))])
    /// let energy = mps.expectationValue(of: h)
    /// ```
    ///
    /// - Parameter observable: Observable as weighted sum of Pauli strings
    /// - Returns: Real expectation value
    /// - Complexity: O(k * n * chi^3) where k = number of terms
    @_optimize(speed)
    @_effects(readonly)
    public func expectationValue(of observable: Observable) -> Double {
        var total = 0.0
        for term in observable.terms {
            total += term.coefficient * expectationValue(of: term.pauliString)
        }
        return total
    }

    /// Array of current bond dimensions at each bond.
    ///
    /// Returns n-1 values for bonds between adjacent tensors. Bond k connects
    /// tensor k and tensor k+1.
    ///
    /// **Example:**
    /// ```swift
    /// let mps = MatrixProductState(qubits: 4)
    /// print(mps.bondDimensions)  // [1, 1, 1] for product state
    /// ```
    public var bondDimensions: [Int] {
        guard qubits > 1 else { return [] }
        return [Int](unsafeUninitializedCapacity: qubits - 1) { buffer, count in
            for i in 0 ..< qubits - 1 {
                buffer[i] = tensors[i].rightBondDimension
            }
            count = qubits - 1
        }
    }

    /// Maximum bond dimension currently used in the MPS.
    ///
    /// **Example:**
    /// ```swift
    /// let mps = MatrixProductState(qubits: 4)
    /// print(mps.currentMaxBondDimension)  // 1 for product state
    /// ```
    @inlinable
    public var currentMaxBondDimension: Int {
        var maxDim = 1
        for tensor in tensors {
            maxDim = max(maxDim, tensor.leftBondDimension)
            maxDim = max(maxDim, tensor.rightBondDimension)
        }
        return maxDim
    }

    /// Estimated memory usage in bytes.
    ///
    /// Counts complex numbers in all tensors, multiplied by 16 bytes per Complex<Double>.
    ///
    /// **Example:**
    /// ```swift
    /// let mps = MatrixProductState(qubits: 100)
    /// print(mps.memoryUsage)  // ~3200 bytes for product state
    /// ```
    public var memoryUsage: Int {
        var total = 0
        for tensor in tensors {
            total += tensor.elements.count * 16
        }
        return total
    }

    /// Human-readable description of the MPS.
    ///
    /// **Example:**
    /// ```swift
    /// let mps = MatrixProductState(qubits: 4)
    /// print(mps.description)
    /// ```
    public var description: String {
        let bonds = bondDimensions.map(String.init).joined(separator: ", ")
        return "MatrixProductState(qubits=\(qubits), maxBond=\(maxBondDimension), currentMaxBond=\(currentMaxBondDimension), bonds=[\(bonds)])"
    }

    @_optimize(speed)
    private static func truncatedSVD(
        matrix: [[Complex<Double>]],
        maxRank: Int,
    ) -> (u: [[Complex<Double>]], s: [Double], vt: [[Complex<Double>]], truncationError: Double) {
        let result = SVDDecomposition.decompose(matrix: matrix, truncation: .maxBondDimension(maxRank))
        return (result.u, result.singularValues, result.vDagger, result.truncationError)
    }

    @_effects(readonly)
    private static func pauliMatrix(_ basis: PauliBasis?) -> [[Complex<Double>]] {
        guard let basis else {
            return [[.one, .zero], [.zero, .one]]
        }

        switch basis {
        case .x:
            return [[.zero, .one], [.one, .zero]]
        case .y:
            return [[.zero, Complex(0, -1)], [Complex(0, 1), .zero]]
        case .z:
            return [[.one, .zero], [.zero, Complex(-1, 0)]]
        }
    }

    /// Updates tensor at specified site.
    mutating func updateTensor(at site: Int, with tensor: MPSTensor) {
        tensors[site] = tensor
    }

    /// Adds truncation error to statistics.
    mutating func addTruncationError(_ error: Double) {
        truncationStatistics = truncationStatistics.adding(error: error)
    }
}
