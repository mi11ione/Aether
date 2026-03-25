// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Generates the n-qubit Pauli basis as tensor products of single-qubit Paulis.
@_optimize(speed)
@_effects(readonly)
private func generatePauliBasis(qubits: Int) -> [[[Complex<Double>]]] {
    let singlePaulis: [[[Complex<Double>]]] = [
        [[.one, .zero], [.zero, .one]],
        [[.zero, .one], [.one, .zero]],
        [[.zero, Complex(0, -1)], [Complex(0, 1), .zero]],
        [[.one, .zero], [.zero, Complex(-1, 0)]],
    ]

    if qubits == 1 {
        return singlePaulis
    }

    var result: [[[Complex<Double>]]] = singlePaulis
    for _ in 1 ..< qubits {
        var newResult: [[[Complex<Double>]]] = []
        newResult.reserveCapacity(result.count * singlePaulis.count)
        for existing in result {
            for pauli in singlePaulis {
                newResult.append(MatrixUtilities.kroneckerProduct(existing, pauli))
            }
        }
        result = newResult
    }

    return result
}

/// Superoperator representation of a quantum channel in Liouville space.
///
/// The superoperator S acts on vectorized density matrices: vec(rho') = S * vec(rho),
/// where vec stacks matrix rows into a column vector. For a channel with Kraus operators {Ki},
/// the superoperator is S = sum_i Ki tensor conj(Ki). Dimension is d^2 x d^2 for d = 2^qubits.
///
/// Superoperators enable efficient composition of quantum channels via matrix multiplication
/// and analysis of channel fixed points as eigenvectors with eigenvalue 1.
///
/// **Example:**
/// ```swift
/// let channel = DepolarizingChannel(errorProbability: 0.1)
/// let superop = SuperoperatorRepresentation(channel: channel, qubits: 1)
/// let isTP = superop.isTracePreserving()
/// ```
///
/// - SeeAlso: ``ChoiMatrix`` for completely positive analysis
/// - SeeAlso: ``NoiseChannel`` for Kraus operator representation
@frozen public struct SuperoperatorRepresentation: Sendable {
    /// The superoperator matrix in row-major order.
    public let matrix: [[Complex<Double>]]

    /// Number of qubits the channel acts on.
    public let qubits: Int

    /// Dimension of the superoperator matrix (d^2 where d = 2^qubits).
    ///
    /// - Complexity: O(1)
    @inlinable public var dimension: Int { 1 << (2 * qubits) }

    /// Creates superoperator from a noise channel.
    ///
    /// Constructs S = sum_i Ki tensor conj(Ki) from Kraus operators.
    ///
    /// **Example:**
    /// ```swift
    /// let depol = DepolarizingChannel(errorProbability: 0.05)
    /// let superop = SuperoperatorRepresentation(channel: depol, qubits: 1)
    /// ```
    ///
    /// - Parameters:
    ///   - channel: Quantum noise channel with Kraus operators
    ///   - qubits: Number of qubits the channel acts on
    /// - Complexity: O(k * d^4) where k is number of Kraus operators, d = 2^qubits
    /// - Precondition: qubits >= 1
    @_optimize(speed)
    public init(channel: some NoiseChannel, qubits: Int) {
        ValidationUtilities.validatePositiveQubits(qubits)

        self.qubits = qubits
        let d = 1 << qubits
        let d2 = d * d

        var result = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: d2),
            count: d2,
        )

        for kraus in channel.krausOperators {
            let krausConj = Self.conjugateMatrix(kraus)
            let tensor = MatrixUtilities.kroneckerProduct(kraus, krausConj)
            for i in 0 ..< d2 {
                for j in 0 ..< d2 {
                    result[i][j] = result[i][j] + tensor[i][j]
                }
            }
        }

        matrix = result
    }

    /// Creates superoperator from Choi matrix via reshuffling.
    ///
    /// Converts Choi matrix J to superoperator S via index reshuffling:
    /// S[ab,cd] = J[ac,bd] where indices are decomposed in the computational basis.
    ///
    /// **Example:**
    /// ```swift
    /// let choi = ChoiMatrix(channel: DepolarizingChannel(errorProbability: 0.1), qubits: 1)
    /// let superop = SuperoperatorRepresentation(choi: choi)
    /// ```
    ///
    /// - Parameter choi: Choi matrix representation of the channel
    /// - Complexity: O(d^4) where d = 2^qubits
    @_optimize(speed)
    public init(choi: ChoiMatrix) {
        qubits = choi.qubits
        let d = 1 << qubits
        let d2 = d * d

        var result = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: d2),
            count: d2,
        )

        for a in 0 ..< d {
            for b in 0 ..< d {
                let row = a * d + b
                for c in 0 ..< d {
                    for dd in 0 ..< d {
                        let col = c * d + dd
                        let choiRow = a * d + c
                        let choiCol = b * d + dd
                        result[row][col] = choi.element(row: choiRow, col: choiCol)
                    }
                }
            }
        }

        matrix = result
    }

    /// Accesses superoperator matrix element at specified row and column.
    ///
    /// **Example:**
    /// ```swift
    /// let channel = DepolarizingChannel(errorProbability: 0.1)
    /// let superop = SuperoperatorRepresentation(channel: channel, qubits: 1)
    /// let value = superop.element(row: 0, col: 0)
    /// ```
    ///
    /// - Parameters:
    ///   - row: Row index (0 to dimension-1)
    ///   - col: Column index (0 to dimension-1)
    /// - Returns: Complex matrix element at (row, col)
    /// - Complexity: O(1)
    /// - Precondition: 0 <= row < dimension
    /// - Precondition: 0 <= col < dimension
    @inlinable
    @_effects(readonly)
    public func element(row: Int, col: Int) -> Complex<Double> {
        let dim = dimension
        ValidationUtilities.validateIndexInBounds(row, bound: dim, name: "Row")
        ValidationUtilities.validateIndexInBounds(col, bound: dim, name: "Column")
        return matrix[row][col]
    }

    /// Applies the superoperator to a density matrix.
    ///
    /// Computes rho' where vec(rho') = S * vec(rho). The density matrix is vectorized
    /// by stacking rows, multiplied by the superoperator, then reshaped back.
    ///
    /// **Example:**
    /// ```swift
    /// let superop = SuperoperatorRepresentation(channel: DepolarizingChannel(errorProbability: 0.1), qubits: 1)
    /// let rho = DensityMatrix(qubits: 1)
    /// let result = superop.apply(to: rho)
    /// ```
    ///
    /// - Parameter densityMatrix: Input density matrix
    /// - Returns: Transformed density matrix
    /// - Complexity: O(d^4) where d = 2^qubits
    /// - Precondition: densityMatrix.qubits == qubits
    @_optimize(speed)
    @_effects(readonly)
    public func apply(to densityMatrix: DensityMatrix) -> DensityMatrix {
        ValidationUtilities.validateQubitCountsEqual(densityMatrix.qubits, qubits, name1: "Input density matrix qubits", name2: "channel qubits")

        let d = 1 << qubits
        let d2 = d * d

        var vec = [Complex<Double>](repeating: .zero, count: d2)
        for i in 0 ..< d {
            for j in 0 ..< d {
                vec[i * d + j] = densityMatrix[row: i, col: j]
            }
        }

        var resultVec = [Complex<Double>](repeating: .zero, count: d2)
        for i in 0 ..< d2 {
            var sum = Complex<Double>.zero
            for j in 0 ..< d2 {
                sum = sum + matrix[i][j] * vec[j]
            }
            resultVec[i] = sum
        }

        return DensityMatrix(qubits: qubits, elements: resultVec)
    }

    /// Checks if the channel is trace-preserving.
    ///
    /// A channel is trace-preserving if sum_i Ki_dag Ki = I, which corresponds to
    /// the superoperator satisfying specific constraints on its structure.
    ///
    /// **Example:**
    /// ```swift
    /// let superop = SuperoperatorRepresentation(channel: DepolarizingChannel(errorProbability: 0.1), qubits: 1)
    /// let isTP = superop.isTracePreserving()
    /// ```
    ///
    /// - Parameter tolerance: Numerical tolerance for comparison (default 1e-10)
    /// - Returns: True if the channel is trace-preserving within tolerance
    /// - Complexity: O(d^4) where d = 2^qubits
    @_effects(readonly)
    public func isTracePreserving(tolerance: Double = 1e-10) -> Bool {
        let d = 1 << qubits

        for j in 0 ..< d {
            for k in 0 ..< d {
                var sum = Complex<Double>.zero
                for i in 0 ..< d {
                    let row = i * d + i
                    let col = j * d + k
                    sum = sum + matrix[row][col]
                }
                let expected: Complex<Double> = (j == k) ? .one : .zero
                let diff = sum - expected
                if diff.magnitudeSquared > tolerance * tolerance {
                    return false
                }
            }
        }
        return true
    }

    /// Composes this channel with another by matrix multiplication.
    ///
    /// Returns S_12 = S_2 * S_1, representing application of self first, then other.
    /// Channel composition in superoperator form is matrix multiplication.
    ///
    /// **Example:**
    /// ```swift
    /// let s1 = SuperoperatorRepresentation(channel: DepolarizingChannel(errorProbability: 0.1), qubits: 1)
    /// let s2 = SuperoperatorRepresentation(channel: AmplitudeDampingChannel(gamma: 0.05), qubits: 1)
    /// let composed = s1.composed(with: s2)
    /// ```
    ///
    /// - Parameter other: Superoperator to compose with (applied after self)
    /// - Returns: Composed superoperator representing sequential application
    /// - Complexity: O(d^6) where d = 2^qubits
    /// - Precondition: other.qubits == self.qubits
    @_optimize(speed)
    @_effects(readonly)
    public func composed(with other: SuperoperatorRepresentation) -> SuperoperatorRepresentation {
        ValidationUtilities.validateQubitCountsEqual(qubits, other.qubits, name1: "self.qubits", name2: "other.qubits")
        let product = MatrixUtilities.matrixMultiply(other.matrix, matrix)
        return SuperoperatorRepresentation(matrix: product, qubits: qubits)
    }

    /// Creates a superoperator from a precomputed matrix.
    private init(matrix: [[Complex<Double>]], qubits: Int) {
        self.matrix = matrix
        self.qubits = qubits
    }

    /// Returns the element-wise complex conjugate of a matrix.
    @inline(__always)
    @_optimize(speed)
    @_effects(readonly)
    private static func conjugateMatrix(_ m: [[Complex<Double>]]) -> [[Complex<Double>]] {
        m.map { row in row.map(\.conjugate) }
    }

    /// Accesses the superoperator matrix element at the given row and column.
    ///
    /// - Parameters:
    ///   - row: Row index in the superoperator matrix.
    ///   - col: Column index in the superoperator matrix.
    /// - Returns: The complex matrix element at the specified position.
    /// - Complexity: O(1)
    @inlinable public subscript(row row: Int, col col: Int) -> Complex<Double> {
        matrix[row][col]
    }
}

/// Choi matrix (Choi-Jamiolkowski) representation of a quantum channel.
///
/// The Choi matrix J(Phi) = (I tensor Phi)(|Omega><Omega|) where |Omega> = (1/sqrt(d)) sum_i |ii>
/// is the maximally entangled state. For Kraus operators {Ki}, J[ac,bd] = sum_i Ki[a,c] * conj(Ki[b,d]).
/// The Choi matrix is positive semidefinite iff the channel is completely positive.
///
/// **Example:**
/// ```swift
/// let channel = AmplitudeDampingChannel(gamma: 0.1)
/// let choi = ChoiMatrix(channel: channel, qubits: 1)
/// let isCP = choi.isCompletelyPositive()
/// let kraus = choi.krausOperators()
/// ```
///
/// - SeeAlso: ``SuperoperatorRepresentation`` for Liouville space representation
/// - SeeAlso: ``ChiMatrix`` for process matrix representation
@frozen public struct ChoiMatrix: Sendable {
    /// The Choi matrix in row-major order.
    public let matrix: [[Complex<Double>]]

    /// Number of qubits the channel acts on.
    public let qubits: Int

    /// Dimension of the Choi matrix (d^2 where d = 2^qubits).
    ///
    /// - Complexity: O(1)
    @inlinable public var dimension: Int { 1 << (2 * qubits) }

    /// Creates Choi matrix from a noise channel.
    ///
    /// Constructs J[ac,bd] = sum_i Ki[a,c] * conj(Ki[b,d]) from Kraus operators.
    ///
    /// **Example:**
    /// ```swift
    /// let amp = AmplitudeDampingChannel(gamma: 0.05)
    /// let choi = ChoiMatrix(channel: amp, qubits: 1)
    /// ```
    ///
    /// - Parameters:
    ///   - channel: Quantum noise channel with Kraus operators
    ///   - qubits: Number of qubits the channel acts on
    /// - Complexity: O(k * d^4) where k is number of Kraus operators, d = 2^qubits
    /// - Precondition: qubits >= 1
    @_optimize(speed)
    public init(channel: some NoiseChannel, qubits: Int) {
        ValidationUtilities.validatePositiveQubits(qubits)

        self.qubits = qubits
        let d = 1 << qubits
        let d2 = d * d

        var result = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: d2),
            count: d2,
        )

        for kraus in channel.krausOperators {
            for a in 0 ..< d {
                for c in 0 ..< d {
                    let rowIdx = a * d + c
                    let kAC = kraus[a][c]
                    for b in 0 ..< d {
                        for dd in 0 ..< d {
                            let colIdx = b * d + dd
                            let kBD = kraus[b][dd].conjugate
                            result[rowIdx][colIdx] = result[rowIdx][colIdx] + kAC * kBD
                        }
                    }
                }
            }
        }

        matrix = result
    }

    /// Creates Choi matrix from superoperator via reshuffling.
    ///
    /// Converts superoperator S to Choi matrix J via index reshuffling:
    /// J[ac,bd] = S[ab,cd] where indices are decomposed in the computational basis.
    ///
    /// **Example:**
    /// ```swift
    /// let superop = SuperoperatorRepresentation(channel: DepolarizingChannel(errorProbability: 0.1), qubits: 1)
    /// let choi = ChoiMatrix(superoperator: superop)
    /// ```
    ///
    /// - Parameter superoperator: Superoperator representation of the channel
    /// - Complexity: O(d^4) where d = 2^qubits
    @_optimize(speed)
    public init(superoperator: SuperoperatorRepresentation) {
        qubits = superoperator.qubits
        let d = 1 << qubits
        let d2 = d * d

        var result = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: d2),
            count: d2,
        )

        for a in 0 ..< d {
            for c in 0 ..< d {
                let choiRow = a * d + c
                for b in 0 ..< d {
                    for dd in 0 ..< d {
                        let choiCol = b * d + dd
                        let superRow = a * d + b
                        let superCol = c * d + dd
                        result[choiRow][choiCol] = superoperator.matrix[superRow][superCol]
                    }
                }
            }
        }

        matrix = result
    }

    /// Accesses Choi matrix element at specified row and column.
    ///
    /// **Example:**
    /// ```swift
    /// let channel = AmplitudeDampingChannel(gamma: 0.1)
    /// let choi = ChoiMatrix(channel: channel, qubits: 1)
    /// let value = choi.element(row: 0, col: 0)
    /// ```
    ///
    /// - Parameters:
    ///   - row: Row index (0 to dimension-1)
    ///   - col: Column index (0 to dimension-1)
    /// - Returns: Complex matrix element at (row, col)
    /// - Complexity: O(1)
    /// - Precondition: 0 <= row < dimension
    /// - Precondition: 0 <= col < dimension
    @inlinable
    @_effects(readonly)
    public func element(row: Int, col: Int) -> Complex<Double> {
        let dim = dimension
        ValidationUtilities.validateIndexInBounds(row, bound: dim, name: "Row")
        ValidationUtilities.validateIndexInBounds(col, bound: dim, name: "Column")
        return matrix[row][col]
    }

    /// Checks if the channel is completely positive.
    ///
    /// A channel is completely positive iff its Choi matrix is positive semidefinite,
    /// i.e., all eigenvalues are non-negative.
    ///
    /// **Example:**
    /// ```swift
    /// let choi = ChoiMatrix(channel: DepolarizingChannel(errorProbability: 0.1), qubits: 1)
    /// let isCP = choi.isCompletelyPositive()
    /// ```
    ///
    /// - Parameter tolerance: Numerical tolerance for eigenvalue comparison (default 1e-10)
    /// - Returns: True if the Choi matrix is positive semidefinite within tolerance
    /// - Complexity: O(d^6) for eigendecomposition where d = 2^qubits
    @_effects(readonly)
    public func isCompletelyPositive(tolerance: Double = 1e-10) -> Bool {
        let eigen = HermitianEigenDecomposition.decompose(matrix: matrix)
        return eigen.eigenvalues.allSatisfy { $0 >= -tolerance }
    }

    /// Checks if the channel is trace-preserving.
    ///
    /// A channel is trace-preserving if Tr_output(J) = I, where the partial trace
    /// is over the output system indices.
    ///
    /// **Example:**
    /// ```swift
    /// let choi = ChoiMatrix(channel: DepolarizingChannel(errorProbability: 0.1), qubits: 1)
    /// let isTP = choi.isTracePreserving()
    /// ```
    ///
    /// - Parameter tolerance: Numerical tolerance for comparison (default 1e-10)
    /// - Returns: True if the channel is trace-preserving within tolerance
    /// - Complexity: O(d^4) where d = 2^qubits
    @_effects(readonly)
    public func isTracePreserving(tolerance: Double = 1e-10) -> Bool {
        let d = 1 << qubits

        for i in 0 ..< d {
            for j in 0 ..< d {
                var sum = Complex<Double>.zero
                for k in 0 ..< d {
                    let row = i * d + k
                    let col = j * d + k
                    sum = sum + matrix[row][col]
                }
                let expected: Complex<Double> = (i == j) ? .one : .zero
                let diff = sum - expected
                if diff.magnitudeSquared > tolerance * tolerance {
                    return false
                }
            }
        }
        return true
    }

    /// Extracts Kraus operators from the Choi matrix via eigendecomposition.
    ///
    /// Decomposes J = sum_i lambda_i |v_i><v_i| and constructs Ki = sqrt(lambda_i) * reshape(v_i, d x d).
    /// Only eigenvectors with positive eigenvalues contribute to the Kraus representation.
    ///
    /// **Example:**
    /// ```swift
    /// let choi = ChoiMatrix(channel: DepolarizingChannel(errorProbability: 0.1), qubits: 1)
    /// let kraus = choi.krausOperators()
    /// ```
    ///
    /// - Returns: Array of Kraus operator matrices
    /// - Complexity: O(d^6) for eigendecomposition where d = 2^qubits
    @_optimize(speed)
    @_effects(readonly)
    public func krausOperators() -> [[[Complex<Double>]]] {
        let d = 1 << qubits
        let eigen = HermitianEigenDecomposition.decompose(matrix: matrix)

        var kraus: [[[Complex<Double>]]] = []
        kraus.reserveCapacity(eigen.eigenvalues.count)
        let tolerance = 1e-12

        for (idx, eigenvalue) in eigen.eigenvalues.enumerated() {
            if eigenvalue > tolerance {
                let sqrtLambda = Darwin.sqrt(eigenvalue)
                var k = [[Complex<Double>]](
                    repeating: [Complex<Double>](repeating: .zero, count: d),
                    count: d,
                )
                for a in 0 ..< d {
                    for c in 0 ..< d {
                        let vecIdx = a * d + c
                        k[a][c] = eigen.eigenvectors[idx][vecIdx] * sqrtLambda
                    }
                }
                kraus.append(k)
            }
        }

        return kraus
    }
}

/// Chi matrix (process matrix) representation of a quantum channel.
///
/// The Chi matrix expresses the channel in the Pauli basis: Phi(rho) = sum_{ij} chi_{ij} P_i rho P_j
/// where {P_i} is the normalized Pauli basis. The Chi matrix is related to the Choi matrix via
/// chi = W_dag * J * W where W is the Pauli-to-computational basis transformation.
///
/// **Example:**
/// ```swift
/// let channel = DepolarizingChannel(errorProbability: 0.1)
/// let chi = ChiMatrix(channel: channel, qubits: 1)
/// let elem = chi.element(row: 0, col: 0)
/// ```
///
/// - SeeAlso: ``ChoiMatrix`` for Choi-Jamiolkowski representation
/// - SeeAlso: ``PauliTransferMatrix`` for real-valued Pauli representation
@frozen public struct ChiMatrix: Sendable {
    /// The Chi matrix in row-major order.
    public let matrix: [[Complex<Double>]]

    /// Number of qubits the channel acts on.
    public let qubits: Int

    /// Dimension of the Chi matrix (d^2 where d = 2^qubits).
    ///
    /// - Complexity: O(1)
    @inlinable public var dimension: Int { 1 << (2 * qubits) }

    /// Creates Chi matrix from a noise channel.
    ///
    /// First constructs the Choi matrix, then transforms to Chi via the Pauli basis change.
    ///
    /// **Example:**
    /// ```swift
    /// let depol = DepolarizingChannel(errorProbability: 0.05)
    /// let chi = ChiMatrix(channel: depol, qubits: 1)
    /// ```
    ///
    /// - Parameters:
    ///   - channel: Quantum noise channel with Kraus operators
    ///   - qubits: Number of qubits the channel acts on
    /// - Complexity: O(d^6) where d = 2^qubits
    /// - Precondition: qubits >= 1
    public init(channel: some NoiseChannel, qubits: Int) {
        let choi = ChoiMatrix(channel: channel, qubits: qubits)
        self.init(choi: choi)
    }

    /// Creates Chi matrix from Choi matrix via Pauli basis transformation.
    ///
    /// Computes chi = W_dag * J * W where W transforms from Pauli to computational basis.
    /// W_{ij} = (1/d) * Tr(P_i * sigma_j) for normalized Pauli operators.
    ///
    /// **Example:**
    /// ```swift
    /// let choi = ChoiMatrix(channel: DepolarizingChannel(errorProbability: 0.1), qubits: 1)
    /// let chi = ChiMatrix(choi: choi)
    /// ```
    ///
    /// - Parameter choi: Choi matrix representation of the channel
    /// - Complexity: O(d^6) where d = 2^qubits
    @_optimize(speed)
    public init(choi: ChoiMatrix) {
        qubits = choi.qubits
        let d = 1 << qubits
        let d2 = d * d

        let paulis = generatePauliBasis(qubits: qubits)

        var w = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: d2),
            count: d2,
        )

        let normFactor = 1.0 / Double(d)
        for i in 0 ..< d2 {
            for j in 0 ..< d2 {
                let rowIdx = j / d
                let colIdx = j % d
                w[i][j] = paulis[i][rowIdx][colIdx] * normFactor
            }
        }

        let wDag = MatrixUtilities.hermitianConjugate(w)
        let temp = MatrixUtilities.matrixMultiply(wDag, choi.matrix)
        matrix = MatrixUtilities.matrixMultiply(temp, w)
    }

    /// Accesses Chi matrix element at specified row and column.
    ///
    /// **Example:**
    /// ```swift
    /// let channel = DepolarizingChannel(errorProbability: 0.1)
    /// let chi = ChiMatrix(channel: channel, qubits: 1)
    /// let value = chi.element(row: 0, col: 0)
    /// ```
    ///
    /// - Parameters:
    ///   - row: Row index (0 to d^2-1)
    ///   - col: Column index (0 to d^2-1)
    /// - Returns: Complex matrix element at (row, col)
    /// - Complexity: O(1)
    /// - Precondition: 0 <= row < d^2
    /// - Precondition: 0 <= col < d^2
    @inlinable
    @_effects(readonly)
    public func element(row: Int, col: Int) -> Complex<Double> {
        let d2 = 1 << (2 * qubits)
        ValidationUtilities.validateIndexInBounds(row, bound: d2, name: "Row")
        ValidationUtilities.validateIndexInBounds(col, bound: d2, name: "Column")
        return matrix[row][col]
    }
}

/// Pauli Transfer Matrix representation of a quantum channel.
///
/// The PTM R has elements R_{ij} = Tr(P_i * Phi(P_j)) / d where {P_i} is the normalized
/// Pauli basis. For CPTP maps, all entries are real. The PTM provides intuitive visualization
/// of how Pauli operators transform under the channel.
///
/// **Example:**
/// ```swift
/// let channel = PhaseDampingChannel(gamma: 0.1)
/// let ptm = PauliTransferMatrix(channel: channel, qubits: 1)
/// let isUnit = ptm.isUnital()
/// ```
///
/// - SeeAlso: ``ChiMatrix`` for process matrix representation
/// - SeeAlso: ``ChoiMatrix`` for Choi-Jamiolkowski representation
@frozen public struct PauliTransferMatrix: Sendable {
    /// The Pauli transfer matrix with real entries.
    public let matrix: [[Double]]

    /// Number of qubits the channel acts on.
    public let qubits: Int

    /// Creates Pauli transfer matrix from a noise channel.
    ///
    /// Computes R_{ij} = Tr(P_i * Phi(P_j)) / d by applying the channel to each
    /// Pauli operator and computing overlaps.
    ///
    /// **Example:**
    /// ```swift
    /// let phase = PhaseDampingChannel(gamma: 0.05)
    /// let ptm = PauliTransferMatrix(channel: phase, qubits: 1)
    /// ```
    ///
    /// - Parameters:
    ///   - channel: Quantum noise channel with Kraus operators
    ///   - qubits: Number of qubits the channel acts on
    /// - Complexity: O(d^6) where d = 2^qubits
    /// - Precondition: qubits >= 1
    @_optimize(speed)
    public init(channel: some NoiseChannel, qubits: Int) {
        ValidationUtilities.validatePositiveQubits(qubits)

        self.qubits = qubits
        let d = 1 << qubits
        let d2 = d * d

        let paulis = generatePauliBasis(qubits: qubits)
        let normFactor = 1.0 / Double(d)

        var result = [[Double]](
            repeating: [Double](repeating: 0.0, count: d2),
            count: d2,
        )

        let krausOps = channel.krausOperators
        let krausDags = krausOps.map { MatrixUtilities.hermitianConjugate($0) }

        var transformed = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: d),
            count: d,
        )
        for j in 0 ..< d2 {
            for a in 0 ..< d {
                for b in 0 ..< d {
                    transformed[a][b] = .zero
                }
            }

            for (kraus, kDag) in zip(krausOps, krausDags) {
                let kPj = MatrixUtilities.matrixMultiply(kraus, paulis[j])
                let kPjKdag = MatrixUtilities.matrixMultiply(kPj, kDag)
                for a in 0 ..< d {
                    for b in 0 ..< d {
                        transformed[a][b] = transformed[a][b] + kPjKdag[a][b]
                    }
                }
            }

            for i in 0 ..< d2 {
                var trace = Complex<Double>.zero
                for a in 0 ..< d {
                    for b in 0 ..< d {
                        trace = trace + paulis[i][a][b] * transformed[b][a]
                    }
                }
                result[i][j] = trace.real * normFactor
            }
        }

        matrix = result
    }

    /// Creates Pauli transfer matrix from Choi matrix.
    ///
    /// Converts from Choi representation by computing Pauli overlaps of the
    /// channel action on each Pauli basis element.
    ///
    /// **Example:**
    /// ```swift
    /// let choi = ChoiMatrix(channel: DepolarizingChannel(errorProbability: 0.1), qubits: 1)
    /// let ptm = PauliTransferMatrix(choi: choi)
    /// ```
    ///
    /// - Parameter choi: Choi matrix representation of the channel
    /// - Complexity: O(d^6) where d = 2^qubits
    @_optimize(speed)
    public init(choi: ChoiMatrix) {
        qubits = choi.qubits
        let d = 1 << qubits
        let d2 = d * d

        let paulis = generatePauliBasis(qubits: qubits)
        let normFactor = 1.0 / Double(d)

        var result = [[Double]](
            repeating: [Double](repeating: 0.0, count: d2),
            count: d2,
        )

        var transformed = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: d),
            count: d,
        )
        for j in 0 ..< d2 {
            for a in 0 ..< d {
                for b in 0 ..< d {
                    transformed[a][b] = .zero
                }
            }

            for a in 0 ..< d {
                for b in 0 ..< d {
                    var sum = Complex<Double>.zero
                    for c in 0 ..< d {
                        for dd in 0 ..< d {
                            let choiRow = a * d + c
                            let choiCol = b * d + dd
                            sum = sum + choi.matrix[choiRow][choiCol] * paulis[j][c][dd]
                        }
                    }
                    transformed[a][b] = sum
                }
            }

            for i in 0 ..< d2 {
                var trace = Complex<Double>.zero
                for a in 0 ..< d {
                    for b in 0 ..< d {
                        trace = trace + paulis[i][a][b] * transformed[b][a]
                    }
                }
                result[i][j] = trace.real * normFactor
            }
        }

        matrix = result
    }

    /// Accesses Pauli transfer matrix element at specified row and column.
    ///
    /// **Example:**
    /// ```swift
    /// let channel = PhaseDampingChannel(gamma: 0.1)
    /// let ptm = PauliTransferMatrix(channel: channel, qubits: 1)
    /// let value = ptm.element(row: 0, col: 0)
    /// ```
    ///
    /// - Parameters:
    ///   - row: Row index (0 to d^2-1)
    ///   - col: Column index (0 to d^2-1)
    /// - Returns: Real matrix element at (row, col)
    /// - Complexity: O(1)
    /// - Precondition: 0 <= row < d^2
    /// - Precondition: 0 <= col < d^2
    @inlinable
    @_effects(readonly)
    public func element(row: Int, col: Int) -> Double {
        let d2 = 1 << (2 * qubits)
        ValidationUtilities.validateIndexInBounds(row, bound: d2, name: "Row")
        ValidationUtilities.validateIndexInBounds(col, bound: d2, name: "Column")
        return matrix[row][col]
    }

    /// Checks if the channel is unital.
    ///
    /// A channel is unital if Phi(I) = I, which corresponds to the first column
    /// of the PTM being [1, 0, 0, ...]^T (identity maps to identity).
    ///
    /// **Example:**
    /// ```swift
    /// let ptm = PauliTransferMatrix(channel: DepolarizingChannel(errorProbability: 0.1), qubits: 1)
    /// let isUnit = ptm.isUnital()
    /// ```
    ///
    /// - Parameter tolerance: Numerical tolerance for comparison (default 1e-10)
    /// - Returns: True if the channel is unital within tolerance
    /// - Complexity: O(d^2) where d = 2^qubits
    @_effects(readonly)
    public func isUnital(tolerance: Double = 1e-10) -> Bool {
        let d2 = 1 << (2 * qubits)

        if Swift.abs(matrix[0][0] - 1.0) > tolerance {
            return false
        }

        for i in 1 ..< d2 {
            if Swift.abs(matrix[i][0]) > tolerance {
                return false
            }
        }

        return true
    }
}
