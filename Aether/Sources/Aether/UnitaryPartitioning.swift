// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// A group of Pauli terms measurable simultaneously via a unitary transformation.
///
/// Applies unitary U† before measurement, transforming all terms into diagonal operators
/// that share a common Z measurement basis. Enables single-shot measurement of multiple
/// non-commuting terms by diagonalizing them via the unitary transformation.
///
/// The unitary U is found such that U†PᵢU are diagonal for all Pauli terms Pᵢ in the partition.
/// This is achieved through eigendecomposition or variational optimization with hardware-efficient ansatz.
///
/// **Example**:
/// ```swift
/// let partitioner = UnitaryPartitioner()
/// let partitions = partitioner.partition(terms: hamiltonian.terms)
///
/// for partition in partitions {
///     let transformedState = applyUnitary(partition.unitaryMatrix, to: state)
///     let outcomes = measure(transformedState, basis: partition.measurementBasis)
///     let expectation = computeExpectation(partition.terms, outcomes: outcomes)
/// }
/// ```
///
/// - Complexity: O(n³) eigendecomposition or O(iterations x depth x 2^(2n)) variational optimization
/// - SeeAlso: ``UnitaryPartitioner``, ``QWCGroup``, ``Observable``
public struct UnitaryPartition: Sendable {
    /// Pauli terms with coefficients that can be measured simultaneously after applying the unitary transformation.
    public let terms: PauliTerms

    /// Unitary transformation matrix (2^n x 2^n) that diagonalizes the terms.
    ///
    /// After applying U†, all Pauli operators in the partition become (nearly) diagonal,
    /// sharing a common Z measurement basis.
    let unitaryMatrix: [[Complex<Double>]]

    /// Measurement basis for all terms in partition (all qubits measured in Z basis).
    ///
    /// Computed once at initialization for efficient repeated measurements.
    public let measurementBasis: [Int: PauliBasis]

    /// Number of qubits (derived from matrix dimension)
    private var numQubits: Int { unitaryMatrix.count.trailingZeroBitCount }

    /// Initialize with precomputed measurement basis
    init(terms: PauliTerms, unitaryMatrix: [[Complex<Double>]]) {
        self.terms = terms
        self.unitaryMatrix = unitaryMatrix

        var basis: [Int: PauliBasis] = [:]
        for term in terms {
            for op in term.pauliString.operators {
                basis[op.qubit] = .z
            }
        }
        measurementBasis = basis
    }

    /// Total weight of partition (sum of absolute coefficients).
    ///
    /// Used for partition ordering and shot allocation. Higher weight partitions
    /// contribute more to total expectation value and should receive more measurements.
    ///
    /// - Complexity: O(n) where n is number of terms in partition
    @inlinable
    public var weight: Double {
        terms.reduce(0.0) { $0 + abs($1.coefficient) }
    }
}

/// Reduces measurement overhead by finding unitary transformations that diagonalize Pauli terms.
///
/// Use this for large Hamiltonians where measurement dominates VQE runtime. QWC grouping alone
/// may leave many groups requiring separate circuits. Unitary partitioning further reduces groups
/// via variational optimization, cutting measurement cost at the expense of optimization overhead.
///
/// Starts with QWC groups, then greedily merges groups by finding unitary U that minimizes
/// the off-diagonal norm ||off-diagonal(U†PᵢU)||². Uses eigendecomposition when possible (fast, exact)
/// or variational ansatz optimization when needed (slower, approximate). Accepts partition if
/// off-diagonal norm falls below threshold (default 0.1).
///
/// The optimization uses LAPACK zheev for eigendecomposition (O(n³)) or L-BFGS-B with hardware-efficient
/// ansatz for variational approach (O(iterations x depth x 2^(2n))). Convergence occurs when gradient
/// norm drops below tolerance (default 1e-6) or max iterations reached (default 100).
///
/// **Example**:
/// ```swift
/// let config = UnitaryPartitioner.Config(
///     maxIterations: 200,
///     circuitDepth: 5,
///     diagonalityThreshold: 0.05
/// )
/// let partitioner = UnitaryPartitioner(config: config)
/// let partitions = partitioner.partition(terms: hamiltonian.terms)
/// print("Reduced \(hamiltonian.terms.count) terms to \(partitions.count) partitions")
/// ```
///
/// - Complexity: O(groups² x iterations x depth x 2^(2n)) worst-case variational optimization
/// - SeeAlso: ``UnitaryPartition``, ``QWCGrouper``, ``VQE``, ``Observable``
public struct UnitaryPartitioner {
    // MARK: - Configuration

    /// Configuration for unitary partitioning optimization.
    ///
    /// Balances optimization quality against runtime. More iterations and stricter thresholds
    /// improve diagonalization but increase cost. Default values work well for typical molecular
    /// Hamiltonians.
    ///
    /// **Example**:
    /// ```swift
    /// let aggressive = Config(maxIterations: 200, diagonalityThreshold: 0.05)
    /// let fast = Config(maxIterations: 50, diagonalityThreshold: 0.2)
    /// ```
    public struct Config: Sendable {
        /// Maximum number of L-BFGS-B iterations before giving up.
        public let maxIterations: Int

        /// Convergence tolerance for gradient norm in L-BFGS-B optimization.
        ///
        /// Optimization stops when ||∇C|| < tolerance. Default 1e-6 is standard for L-BFGS.
        public let convergenceTolerance: Double

        /// Number of layers in hardware-efficient variational ansatz.
        ///
        /// Each layer adds 3n parameters (n = number of qubits). Default 3 layers balances
        /// expressivity with optimization difficulty.
        public let circuitDepth: Int

        /// Whether to use adaptive layer depth (currently unused, reserved for future optimization).
        public let useAdaptiveDepth: Bool

        /// Maximum off-diagonal norm to accept partition (lower = stricter).
        ///
        /// Partitions with ||off-diagonal(U†HU)|| > threshold are rejected and fall back
        /// to QWC grouping. Default 0.1 accepts nearly-diagonal results.
        public let diagonalityThreshold: Double

        public init(
            maxIterations: Int = 100,
            convergenceTolerance: Double = 1e-6,
            circuitDepth: Int = 3,
            useAdaptiveDepth: Bool = true,
            diagonalityThreshold: Double = 0.1
        ) {
            self.maxIterations = maxIterations
            self.convergenceTolerance = convergenceTolerance
            self.circuitDepth = circuitDepth
            self.useAdaptiveDepth = useAdaptiveDepth
            self.diagonalityThreshold = diagonalityThreshold
        }
    }

    public let config: Config

    public init(config: Config = .init()) { self.config = config }

    // MARK: - Target Operator Construction

    /// Build target operator matrix from Pauli terms: H = Σᵢ cᵢ Pᵢ
    @_optimize(speed)
    @_eagerMove
    private func buildTargetOperator(
        terms: PauliTerms,
        numQubits: Int
    ) -> [[Complex<Double>]] {
        let dimension = 1 << numQubits
        var targetOperator = Array(repeating: Array(repeating: Complex<Double>.zero, count: dimension), count: dimension)

        for (coeff, pauliString) in terms {
            let pauliMatrix = pauliString.toMatrix(numQubits: numQubits)
            for i in 0 ..< dimension {
                for j in 0 ..< dimension {
                    targetOperator[i][j] += Complex(coeff) * pauliMatrix[i][j]
                }
            }
        }

        return targetOperator
    }

    // MARK: - Main Partitioning Algorithm

    /// Groups Pauli terms into partitions diagonalizable by unitary transformations.
    ///
    /// Starts with QWC groups, then greedily merges groups where optimization finds a diagonalizing
    /// unitary. Falls back to identity matrix if optimization fails (returns QWC group as-is).
    ///
    /// Computes QWC groups as baseline, then for each seed group attempts to merge remaining groups.
    /// Optimization tries eigendecomposition first (fast, exact), then variational approach if needed
    /// (slower, approximate). Accepts merge if off-diagonal norm falls below threshold. Non-mergeable
    /// groups remain as separate partitions.
    ///
    /// **Example**:
    /// ```swift
    /// let partitioner = UnitaryPartitioner()
    /// let partitions = partitioner.partition(terms: hamiltonian.terms)
    ///
    /// for (index, partition) in partitions.enumerated() {
    ///     print("Partition \(index): \(partition.terms.count) terms, weight \(partition.weight)")
    /// }
    /// ```
    ///
    /// - Parameter terms: Pauli terms with coefficients from ``Observable``
    /// - Returns: Unitary partitions (order depends on greedy merging, not sorted by weight)
    /// - Complexity: O(groups² x iterations x depth x 2^(2n)) worst-case, O(groups x n³) best-case
    /// - SeeAlso: ``UnitaryPartition``, ``Config``
    @_optimize(speed)
    @_eagerMove
    public func partition(terms: PauliTerms) -> [UnitaryPartition] {
        let numQubits = terms.map { $0.pauliString.operators.map(\.qubit).max() ?? 0 }.max().map { $0 + 1 } ?? 0
        let qwcGroups: [QWCGroup] = QWCGrouper.group(terms)
        var partitions: [UnitaryPartition] = []
        var remainingGroups: [QWCGroup] = qwcGroups

        while !remainingGroups.isEmpty {
            let seed: QWCGroup = remainingGroups.removeFirst()
            var currentTerms = seed.terms
            var lastUnitary: [[Complex<Double>]]?

            var i = 0
            while i < remainingGroups.count {
                let candidate: QWCGroup = remainingGroups[i]
                let mergedTerms: PauliTerms = currentTerms + candidate.terms

                if let unitary = findDiagonalizingUnitary(
                    terms: mergedTerms,
                    numQubits: numQubits
                ) {
                    currentTerms = mergedTerms
                    lastUnitary = unitary
                    remainingGroups.remove(at: i)
                } else {
                    i += 1
                }
            }

            if let unitary = lastUnitary ?? findDiagonalizingUnitary(terms: currentTerms, numQubits: numQubits) {
                partitions.append(UnitaryPartition(
                    terms: currentTerms,
                    unitaryMatrix: unitary
                ))
            } else {
                let identity: [[Complex<Double>]] = MatrixUtilities.identityMatrix(dimension: 1 << numQubits)
                partitions.append(UnitaryPartition(
                    terms: currentTerms,
                    unitaryMatrix: identity
                ))
            }
        }

        return partitions
    }

    // MARK: - Unitary Optimization

    /// Find unitary that diagonalizes given Pauli terms.
    ///
    /// - Parameters:
    ///   - terms: Pauli terms to diagonalize
    ///   - numQubits: Number of qubits
    /// - Returns: Unitary matrix if found, nil if optimization fails
    @_optimize(speed)
    @_eagerMove
    private func findDiagonalizingUnitary(
        terms: PauliTerms,
        numQubits: Int
    ) -> [[Complex<Double>]]? {
        let targetOperator = buildTargetOperator(terms: terms, numQubits: numQubits)

        if let (_, eigenvectors) = eigendecompose(targetOperator) {
            let offDiagNorm: Double = computeOffDiagonalNorm(
                operator: targetOperator,
                unitary: eigenvectors
            )

            if offDiagNorm < config.diagonalityThreshold {
                return eigenvectors
            }
        }

        return optimizeVariational(terms: terms, numQubits: numQubits)
    }

    /// Optimize unitary using variational ansatz.
    @_optimize(speed)
    @_eagerMove
    private func optimizeVariational(
        terms: PauliTerms,
        numQubits: Int
    ) -> [[Complex<Double>]]? {
        let pauliMatrices: [[[Complex<Double>]]] = terms.map { $0.pauliString.toMatrix(numQubits: numQubits) }

        let numParams: Int = parameterCount(numQubits: numQubits, depth: config.circuitDepth)
        let parameters = [Double](unsafeUninitializedCapacity: numParams) { buffer, count in
            for i in 0 ..< numParams {
                buffer[i] = Double.random(in: -Double.pi ... Double.pi)
            }
            count = numParams
        }

        let result: UnitaryOptimizationResult = lbfgsb(
            initialParameters: parameters,
            costFunction: { params in
                costFunctionCached(
                    parameters: params,
                    terms: terms,
                    pauliMatrices: pauliMatrices,
                    numQubits: numQubits
                )
            },
            gradientFunction: { params in
                gradientFunctionCached(
                    parameters: params,
                    terms: terms,
                    pauliMatrices: pauliMatrices,
                    numQubits: numQubits
                )
            },
            maxIterations: config.maxIterations,
            tolerance: config.convergenceTolerance
        )

        let unitary: [[Complex<Double>]] = buildVariationalUnitary(
            parameters: result.parameters,
            numQubits: numQubits,
            depth: config.circuitDepth
        )

        let targetOperator = buildTargetOperator(terms: terms, numQubits: numQubits)

        let offDiagNorm: Double = computeOffDiagonalNorm(operator: targetOperator, unitary: unitary)

        return offDiagNorm < config.diagonalityThreshold ? unitary : nil
    }

    // MARK: - Cost and Gradient Functions

    @_optimize(speed)
    private func costFunctionCached(
        parameters: [Double],
        terms: PauliTerms,
        pauliMatrices: [[[Complex<Double>]]],
        numQubits: Int
    ) -> Double {
        let unitary = buildVariationalUnitary(
            parameters: parameters,
            numQubits: numQubits,
            depth: config.circuitDepth
        )

        let dimension = 1 << numQubits
        var cost = 0.0

        for (index, term) in terms.enumerated() {
            let pauliMatrix = pauliMatrices[index]
            let conjugated = conjugateByUnitary(pauliMatrix, unitary: unitary)

            let coeffWeight = abs(term.coefficient)

            let interleaved = [Double](unsafeUninitializedCapacity: dimension * dimension * 2) { buffer, count in
                var idx = 0
                for i in 0 ..< dimension {
                    for j in 0 ..< dimension {
                        buffer[idx] = conjugated[i][j].real
                        buffer[idx + 1] = conjugated[i][j].imaginary
                        idx += 2
                    }
                }
                count = dimension * dimension * 2
            }

            var totalSumSq = 0.0
            vDSP_svesqD(interleaved, 1, &totalSumSq, vDSP_Length(interleaved.count))

            var diagSumSq = 0.0
            for i in 0 ..< dimension {
                diagSumSq += conjugated[i][i].magnitudeSquared
            }

            cost += coeffWeight * (totalSumSq - diagSumSq)
        }

        return cost
    }

    @_optimize(speed)
    @_eagerMove
    private func gradientFunctionCached(
        parameters: [Double],
        terms: PauliTerms,
        pauliMatrices: [[[Complex<Double>]]],
        numQubits: Int
    ) -> [Double] {
        let epsilon = 1e-7
        let paramCount = parameters.count

        let f0 = costFunctionCached(
            parameters: parameters,
            terms: terms,
            pauliMatrices: pauliMatrices,
            numQubits: numQubits
        )

        let gradient = [Double](unsafeUninitializedCapacity: paramCount) { buffer, count in
            for i in 0 ..< paramCount {
                var paramsPlus = parameters
                paramsPlus[i] += epsilon

                let fPlus = costFunctionCached(
                    parameters: paramsPlus,
                    terms: terms,
                    pauliMatrices: pauliMatrices,
                    numQubits: numQubits
                )
                buffer[i] = (fPlus - f0) / epsilon
            }
            count = paramCount
        }

        return gradient
    }

    // MARK: - Variational Ansatz

    /// Counts parameters in variational ansatz (depth x numQubits x 3).
    ///
    /// Each layer has 3 parameters per qubit (Rz-Ry-Rz Euler angles for U3 gate).
    ///
    /// - Complexity: O(1)
    @_effects(readonly)
    private func parameterCount(numQubits: Int, depth: Int) -> Int {
        depth * numQubits * 3
    }

    /// Constructs unitary matrix from variational ansatz parameters.
    ///
    /// Uses hardware-efficient ansatz with depth layers. Each layer applies U3 rotations (Rz-Ry-Rz Euler angles)
    /// to all qubits, then a linear CNOT chain for entanglement. Total parameters: depth x numQubits x 3.
    ///
    /// The ansatz structure repeats depth times: U3(θ, φ, λ) = Rz(λ) Ry(θ) Rz(φ) on each qubit, followed
    /// by CNOT ladder q₀->q₁, q₁->q₂, ..., qₙ₋₂->qₙ₋₁. CNOT matrices are precomputed once per qubit pair to
    /// avoid redundant construction.
    ///
    /// **Example**:
    /// ```swift
    /// let params = [0.5, 1.2, -0.3, 0.8, 0.1, 0.7]
    /// let unitary = buildVariationalUnitary(parameters: params, numQubits: 2, depth: 1)
    /// ```
    ///
    /// - Parameters:
    ///   - parameters: Rotation angles (length = depth x numQubits x 3)
    ///   - numQubits: Number of qubits
    ///   - depth: Number of ansatz layers
    /// - Returns: Unitary matrix (2^numQubits x 2^numQubits)
    /// - Complexity: O(depth x numQubits x 2^(2n)) for matrix composition
    /// - Precondition: `parameters.count == depth x numQubits x 3`
    @_optimize(speed)
    @_eagerMove
    private func buildVariationalUnitary(
        parameters: [Double],
        numQubits: Int,
        depth: Int
    ) -> [[Complex<Double>]] {
        let dimension = 1 << numQubits
        var unitary: [[Complex<Double>]] = MatrixUtilities.identityMatrix(dimension: dimension)

        let cnotMatrices: [[[Complex<Double>]]] = (0 ..< (numQubits - 1)).map { qubit in
            cnotMatrix(control: qubit, target: qubit + 1, numQubits: numQubits)
        }

        var paramIndex = 0

        for _ in 0 ..< depth {
            for qubit in 0 ..< numQubits {
                let theta: Double = parameters[paramIndex]
                let phi: Double = parameters[paramIndex + 1]
                let lambda: Double = parameters[paramIndex + 2]
                paramIndex += 3

                let rotation: [[Complex<Double>]] = singleQubitRotation(
                    qubit: qubit,
                    theta: theta,
                    phi: phi,
                    lambda: lambda,
                    numQubits: numQubits
                )

                unitary = MatrixUtilities.matrixMultiply(rotation, unitary)
            }

            for (index, cnot) in cnotMatrices.enumerated() where index < numQubits - 1 {
                unitary = MatrixUtilities.matrixMultiply(cnot, unitary)
            }
        }

        return unitary
    }

    // MARK: - Matrix Utilities

    @_optimize(speed)
    @_eagerMove
    private func conjugateByUnitary(
        _ matrix: [[Complex<Double>]],
        unitary: [[Complex<Double>]]
    ) -> [[Complex<Double>]] {
        let unitaryDagger: [[Complex<Double>]] = MatrixUtilities.hermitianConjugate(unitary)
        let temp: [[Complex<Double>]] = MatrixUtilities.matrixMultiply(unitaryDagger, matrix)
        return MatrixUtilities.matrixMultiply(temp, unitary)
    }

    @_optimize(speed)
    @_effects(readonly)
    private func computeOffDiagonalNorm(
        operator matrix: [[Complex<Double>]],
        unitary: [[Complex<Double>]]
    ) -> Double {
        let conjugated: [[Complex<Double>]] = conjugateByUnitary(matrix, unitary: unitary)
        let n: Int = conjugated.count

        var normSquared = 0.0
        for i in 0 ..< n {
            for j in 0 ..< n where i != j {
                normSquared += conjugated[i][j].magnitudeSquared
            }
        }

        return sqrt(normSquared)
    }

    @_optimize(speed)
    @_eagerMove
    private func singleQubitRotation(
        qubit: Int,
        theta: Double,
        phi: Double,
        lambda: Double,
        numQubits: Int
    ) -> [[Complex<Double>]] {
        let c: Double = cos(theta / 2)
        let s: Double = sin(theta / 2)

        let u3: [[Complex<Double>]] = [
            [Complex(c), Complex(-cos(lambda) * s, -sin(lambda) * s)],
            [Complex(cos(phi) * s, sin(phi) * s), Complex(cos(phi + lambda) * c, sin(phi + lambda) * c)],
        ]

        return embedSingleQubitGate(u3, qubit: qubit, numQubits: numQubits)
    }

    @_optimize(speed)
    @_eagerMove
    private func cnotMatrix(control: Int, target: Int, numQubits: Int) -> [[Complex<Double>]] {
        let dimension = 1 << numQubits
        var cnot: [[Complex<Double>]] = MatrixUtilities.identityMatrix(dimension: dimension)

        for basis in 0 ..< dimension {
            let controlBit: Int = BitUtilities.getBit(basis, qubit: control)
            if controlBit == 1 {
                let flippedBasis: Int = BitUtilities.flipBit(basis, qubit: target)
                if flippedBasis != basis {
                    cnot[basis][basis] = .zero
                    cnot[basis][flippedBasis] = .one
                }
            }
        }

        return cnot
    }

    @_optimize(speed)
    @_eagerMove
    private func embedSingleQubitGate(
        _ gate: [[Complex<Double>]],
        qubit: Int,
        numQubits: Int
    ) -> [[Complex<Double>]] {
        let dimension = 1 << numQubits
        var result = Array(repeating: Array(repeating: Complex<Double>.zero, count: dimension), count: dimension)

        for row in 0 ..< dimension {
            for col in 0 ..< dimension {
                let rowBit: Int = BitUtilities.getBit(row, qubit: qubit)
                let colBit: Int = BitUtilities.getBit(col, qubit: qubit)

                let rowRest: Int = BitUtilities.clearBit(row, qubit: qubit)
                let colRest: Int = BitUtilities.clearBit(col, qubit: qubit)

                if rowRest == colRest {
                    result[row][col] = gate[rowBit][colBit]
                }
            }
        }

        return result
    }

    // MARK: - Eigendecomposition

    /// Diagonalizes Hermitian matrix via eigendecomposition using LAPACK's zheev.
    ///
    /// For Hermitian operators, eigenvectors form a unitary basis that diagonalizes the operator.
    /// Given H, finds U such that U†HU = diag(λ₀, λ₁, ..., λₙ₋₁) where U has eigenvectors as columns.
    /// This is the exact solution for unitary partitioning when it succeeds.
    ///
    /// Returns nil on LAPACK errors (singular matrix, numerical instability, memory allocation failure).
    /// Algorithm assumes Hermiticity without validation. Caller should fall back to variational optimization
    /// on failure.
    ///
    /// **Example**:
    /// ```swift
    /// let hamiltonian = buildTargetOperator(terms: terms, numQubits: 3)
    /// if let (eigenvalues, eigenvectors) = eigendecompose(hamiltonian) {
    ///     let diagonal = conjugateByUnitary(hamiltonian, unitary: eigenvectors)
    /// }
    /// ```
    ///
    /// - Parameter matrix: Hermitian matrix (n x n) in row-major order
    /// - Returns: (eigenvalues sorted ascending, eigenvectors as columns) or nil if LAPACK fails
    /// - Complexity: O(n³) time, O(n²) space via LAPACK zheev
    /// - Precondition: Matrix must be Hermitian (no validation performed)
    @_optimize(speed)
    @_eagerMove
    private func eigendecompose(_ matrix: [[Complex<Double>]]) -> (eigenvalues: [Double], eigenvectors: [[Complex<Double>]])? {
        let n: Int = matrix.count

        var a = [Double](unsafeUninitializedCapacity: 2 * n * n) { buffer, count in
            for col in 0 ..< n {
                let colOffset = 2 * col * n
                for row in 0 ..< n {
                    let idx = colOffset + 2 * row
                    buffer[idx] = matrix[row][col].real
                    buffer[idx + 1] = matrix[row][col].imaginary
                }
            }
            count = 2 * n * n
        }

        var w = [Double](unsafeUninitializedCapacity: n) { _, count in count = n }

        var jobz = CChar(Character("V").asciiValue!)
        var uplo = CChar(Character("U").asciiValue!)
        var nn = __LAPACK_int(n)
        var lda = __LAPACK_int(n)
        var lwork = __LAPACK_int(-1)
        var info = __LAPACK_int(0)

        var rwork = [Double](repeating: 0.0, count: max(1, 3 * n - 2))

        var workQuery = [Double](repeating: 0.0, count: 2)

        let queryResult: __LAPACK_int = a.withUnsafeMutableBytes { aPtr in
            workQuery.withUnsafeMutableBytes { workPtr in
                w.withUnsafeMutableBufferPointer { wPtr in
                    rwork.withUnsafeMutableBufferPointer { rworkPtr in
                        zheev_(
                            &jobz, &uplo, &nn,
                            OpaquePointer(aPtr.baseAddress),
                            &lda,
                            wPtr.baseAddress,
                            OpaquePointer(workPtr.baseAddress)!,
                            &lwork,
                            rworkPtr.baseAddress,
                            &info
                        )
                        return info
                    }
                }
            }
        }

        guard queryResult == 0 else { return nil }

        let optimalWorkSize = Int(workQuery[0])
        guard optimalWorkSize > 0 else { return nil }

        lwork = __LAPACK_int(optimalWorkSize)
        var work = [Double](repeating: 0.0, count: 2 * optimalWorkSize)

        let computeResult: __LAPACK_int = a.withUnsafeMutableBytes { aPtr in
            work.withUnsafeMutableBytes { workPtr in
                w.withUnsafeMutableBufferPointer { wPtr in
                    rwork.withUnsafeMutableBufferPointer { rworkPtr in
                        zheev_(
                            &jobz, &uplo, &nn,
                            OpaquePointer(aPtr.baseAddress),
                            &lda,
                            wPtr.baseAddress,
                            OpaquePointer(workPtr.baseAddress)!,
                            &lwork,
                            rworkPtr.baseAddress,
                            &info
                        )
                        return info
                    }
                }
            }
        }

        guard computeResult == 0 else { return nil }

        let eigenvectors = [[Complex<Double>]](unsafeUninitializedCapacity: n) { rowBuffer, rowCount in
            for row in 0 ..< n {
                rowBuffer[row] = [Complex<Double>](unsafeUninitializedCapacity: n) { colBuffer, colCount in
                    for col in 0 ..< n {
                        let idx = 2 * (col * n + row)
                        colBuffer[col] = Complex(a[idx], a[idx + 1])
                    }
                    colCount = n
                }
            }
            rowCount = n
        }

        return (eigenvalues: w, eigenvectors: eigenvectors)
    }

    // MARK: - L-BFGS-B Optimizer

    public struct UnitaryOptimizationResult {
        public let parameters: [Double]
        public let finalCost: Double
        public let iterations: Int
        public let converged: Bool
    }

    /// Minimizes cost function using limited-memory quasi-Newton method (L-BFGS-B).
    ///
    /// Approximates Hessian via gradient history (last 10 steps), avoiding full storage of quasi-Newton
    /// methods. Uses two-loop recursion for search direction and backtracking line search with Wolfe
    /// conditions for step size. Converges when gradient norm drops below tolerance or max iterations
    /// reached. Line search may fail on non-descent directions.
    ///
    /// Faster convergence than gradient descent for smooth landscapes, but requires gradient evaluations
    /// which are costly for variational unitary optimization. Uses Wolfe constants c1=1e-4 (sufficient
    /// decrease) and c2=0.9 (curvature condition).
    ///
    /// **Example**:
    /// ```swift
    /// let result = lbfgsb(
    ///     initialParameters: randomAngles,
    ///     costFunction: { params in offDiagonalCost(params, terms: terms) },
    ///     gradientFunction: { params in finiteDifferenceGradient(params) },
    ///     maxIterations: 100,
    ///     tolerance: 1e-6
    /// )
    /// print("Converged: \(result.converged), iterations: \(result.iterations)")
    /// ```
    ///
    /// - Parameters:
    ///   - initialParameters: Starting point (typically random angles in [-π, π])
    ///   - costFunction: Objective to minimize (off-diagonal norm for unitary partitioning)
    ///   - gradientFunction: Gradient of objective (finite differences, epsilon=1e-7)
    ///   - maxIterations: Maximum iterations before giving up
    ///   - tolerance: Convergence threshold on gradient norm
    /// - Returns: Optimization result with final parameters, cost, iterations, convergence status
    /// - Complexity: O(iterations x n x cost_eval) where cost_eval = O(depth x 2^(2n)) for variational unitary
    @_optimize(speed)
    @_eagerMove
    private func lbfgsb(
        initialParameters: [Double],
        costFunction: ([Double]) -> Double,
        gradientFunction: ([Double]) -> [Double],
        maxIterations: Int,
        tolerance: Double
    ) -> UnitaryOptimizationResult {
        let c1 = 1e-4
        let c2 = 0.9

        var params = initialParameters
        var gradient = gradientFunction(params)
        var cost = costFunction(params)

        var sHistory: [[Double]] = []
        var yHistory: [[Double]] = []
        var rhoHistory: [Double] = []

        var iteration = 0
        var converged = false

        while iteration < maxIterations {
            var gradNormSq = 0.0
            vDSP_svesqD(gradient, 1, &gradNormSq, vDSP_Length(gradient.count))
            let gradNorm = sqrt(gradNormSq)

            if gradNorm < tolerance {
                converged = true
                break
            }

            let direction: [Double] = LBFGSBOptimizer.computeSearchDirection(
                gradient: gradient,
                sHistory: sHistory,
                yHistory: yHistory,
                rhoHistory: rhoHistory
            )

            guard let alpha = lineSearch(
                params: params,
                direction: direction,
                gradient: gradient,
                cost: cost,
                costFunction: costFunction,
                gradientFunction: gradientFunction,
                c1: c1,
                c2: c2
            ) else { break }

            let n = params.count
            let newParams = [Double](unsafeUninitializedCapacity: n) { buffer, count in
                var alphaVar = alpha
                vDSP_vsmaD(direction, 1, &alphaVar, params, 1, buffer.baseAddress!, 1, vDSP_Length(n))
                count = n
            }
            let newGradient: [Double] = gradientFunction(newParams)
            let newCost: Double = costFunction(newParams)

            let s = [Double](unsafeUninitializedCapacity: n) { buffer, count in
                vDSP_vsubD(params, 1, newParams, 1, buffer.baseAddress!, 1, vDSP_Length(n))
                count = n
            }
            let y = [Double](unsafeUninitializedCapacity: n) { buffer, count in
                vDSP_vsubD(gradient, 1, newGradient, 1, buffer.baseAddress!, 1, vDSP_Length(n))
                count = n
            }

            var ys = 0.0
            vDSP_dotprD(y, 1, s, 1, &ys, vDSP_Length(n))

            if ys > 1e-10 {
                let rho = 1.0 / ys

                sHistory.append(s)
                yHistory.append(y)
                rhoHistory.append(rho)
            }

            params = newParams
            gradient = newGradient
            cost = newCost
            iteration += 1
        }

        return UnitaryOptimizationResult(
            parameters: params,
            finalCost: cost,
            iterations: iteration,
            converged: converged
        )
    }

    /// Backtracking line search with Wolfe conditions.
    @_optimize(speed)
    private func lineSearch(
        params: [Double],
        direction: [Double],
        gradient: [Double],
        cost: Double,
        costFunction: ([Double]) -> Double,
        gradientFunction: ([Double]) -> [Double],
        c1: Double,
        c2: Double
    ) -> Double? {
        var alpha = 1.0
        let maxBacktrack = 20
        let rho = 0.5
        let n = params.count

        var dirGrad = 0.0
        vDSP_dotprD(direction, 1, gradient, 1, &dirGrad, vDSP_Length(n))

        guard dirGrad < 0 else { return nil }

        for _ in 0 ..< maxBacktrack {
            let newParams = [Double](unsafeUninitializedCapacity: n) { buffer, count in
                var alphaVar = alpha
                vDSP_vsmaD(direction, 1, &alphaVar, params, 1, buffer.baseAddress!, 1, vDSP_Length(n))
                count = n
            }
            let newCost: Double = costFunction(newParams)

            if newCost <= cost + c1 * alpha * dirGrad {
                let newGradient: [Double] = gradientFunction(newParams)
                var newDirGrad = 0.0
                vDSP_dotprD(direction, 1, newGradient, 1, &newDirGrad, vDSP_Length(n))

                if abs(newDirGrad) <= -c2 * dirGrad { return alpha }
            }

            alpha *= rho
        }

        return alpha > 1e-10 ? alpha : nil
    }
}

// MARK: - PauliString Matrix Extension

public extension PauliString {
    /// Compute column index and phase transformation for Pauli string applied to basis state
    ///
    /// Core algorithm for converting Pauli strings to matrix representations. Given a basis state
    /// (row index), computes the resulting state (column index) and accumulated phase after
    /// applying the Pauli string operator. Used by both sparse matrix construction (SparseHamiltonian)
    /// and dense matrix conversion (UnitaryPartitioning).
    ///
    /// **Pauli action on computational basis**:
    /// - **X**: Bit flip -> `col ^= (1 << qubit)`
    /// - **Y**: Bit flip + phase -> `col ^= (1 << qubit)`, `phase *= (bit==0 ? -i : i)`
    /// - **Z**: Phase only -> `phase *= (bit==0 ? +1 : -1)`
    /// - **I** (identity): No-op -> unchanged `col` and `phase`
    ///
    /// **Phase conventions**:
    /// - Y operator: Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
    /// - Z operator: Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
    /// - Combined via multiplication for multi-qubit tensors
    ///
    /// **Example**:
    /// ```swift
    /// let yPauli = PauliString(operators: [(qubit: 0, basis: .y)])
    /// let (col, phase) = yPauli.applyToRow(row: 0)
    /// // Y|0⟩ = i|1⟩ -> col = 1, phase = i
    /// ```
    ///
    /// - Parameter row: Input basis state index (0 to 2^numQubits - 1)
    /// - Returns: (column index, phase factor) representing P|row⟩ = phase * |col⟩
    @_optimize(speed)
    @inlinable
    func applyToRow(row: Int) -> (col: Int, phase: Complex<Double>) {
        var col: Int = row
        var phase = Complex<Double>.one

        for op in operators {
            let qubit = op.qubit
            let rowBit: Int = (row >> qubit) & 1
            let mask = BitUtilities.bitMask(qubit: qubit)

            switch op.basis {
            case .x:
                col ^= mask

            case .y:
                col ^= mask
                phase *= rowBit == 0 ? -Complex<Double>.i : Complex<Double>.i

            case .z:
                phase *= rowBit == 0 ? .one : -.one
            }
        }

        return (col, phase)
    }

    /// Converts Pauli string to dense matrix representation (2^n x 2^n).
    ///
    /// Applies ``applyToRow(row:)`` to all basis states to construct full matrix. Used by ``UnitaryPartitioner``
    /// to build target operators H = Σᵢ cᵢ Pᵢ for optimization. Result has exactly 2^numQubits non-zero entries
    /// (one per row).
    ///
    /// **Example**:
    /// ```swift
    /// let xGate = PauliString(operators: [(qubit: 0, basis: .x)])
    /// let matrix = xGate.toMatrix(numQubits: 1)
    /// ```
    ///
    /// - Parameter numQubits: Total number of qubits in system
    /// - Returns: Dense matrix representation (2^numQubits x 2^numQubits)
    /// - Complexity: O(2^(2n)) time and space
    /// - SeeAlso: ``applyToRow(row:)``
    @_optimize(speed)
    @_eagerMove
    func toMatrix(numQubits: Int) -> [[Complex<Double>]] {
        let dimension = 1 << numQubits
        var result = Array(repeating: Array(repeating: Complex<Double>.zero, count: dimension), count: dimension)

        for row in 0 ..< dimension {
            let (col, phase) = applyToRow(row: row)
            result[row][col] += phase
        }

        return result
    }
}
