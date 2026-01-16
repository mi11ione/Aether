// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Pauli terms diagonalizable by a common unitary transformation for simultaneous measurement.
///
/// Created by ``UnitaryPartitioner``. All terms share a Z measurement basis after applying
/// the diagonalizing unitary U†.
///
/// - SeeAlso: ``UnitaryPartitioner``
@frozen
public struct UnitaryPartition: Sendable {
    /// Pauli terms in this partition.
    public let terms: PauliTerms

    /// Diagonalizing unitary (2^n × 2^n).
    let unitaryMatrix: [[Complex<Double>]]

    /// Measurement basis per qubit (all Z after U†).
    public let measurementBasis: [Int: PauliBasis]

    private var qubits: Int { unitaryMatrix.count.trailingZeroBitCount }

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

    /// Sum of |coefficients| for shot allocation priority.
    @inlinable
    public var weight: Double {
        terms.reduce(0.0) { $0 + abs($1.coefficient) }
    }
}

/// Groups Pauli terms into partitions diagonalizable by unitary transformations.
///
/// Starts with QWC groups, then greedily merges using eigendecomposition (fast, exact) or
/// variational ansatz optimization (L-BFGS-B with hardware-efficient ansatz) when needed.
///
/// **Example:**
/// ```swift
/// let partitioner = UnitaryPartitioner()
/// let partitions = partitioner.partition(terms: observable.terms)
/// ```
///
/// - SeeAlso: ``UnitaryPartition``
/// - SeeAlso: ``QWCGrouper``
public struct UnitaryPartitioner {
    // MARK: - Configuration

    /// Variational optimization parameters.
    @frozen
    public struct Config: Sendable {
        /// L-BFGS-B iteration limit (default 100).
        public let maxIterations: Int

        /// Gradient norm convergence threshold (default 1e-6).
        public let convergenceTolerance: Double

        /// Ansatz layers, 3n parameters per layer (default 3).
        public let circuitDepth: Int

        /// Reserved for future use.
        public let useAdaptiveDepth: Bool

        /// Max off-diagonal norm to accept partition (default 0.1).
        public let diagonalityThreshold: Double

        public init(
            maxIterations: Int = 100,
            convergenceTolerance: Double = 1e-6,
            circuitDepth: Int = 3,
            useAdaptiveDepth: Bool = true,
            diagonalityThreshold: Double = 0.1,
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
        qubits: Int,
    ) -> [[Complex<Double>]] {
        let dimension = 1 << qubits
        var targetOperator = [[Complex<Double>]](unsafeUninitializedCapacity: dimension) { buffer, count in
            for i in 0 ..< dimension {
                buffer.initializeElement(at: i, to: [Complex<Double>](repeating: .zero, count: dimension))
            }
            count = dimension
        }

        for (coeff, pauliString) in terms {
            let pauliMatrix = pauliString.toMatrix(qubits: qubits)
            for i in 0 ..< dimension {
                for j in 0 ..< dimension {
                    targetOperator[i][j] += Complex(coeff) * pauliMatrix[i][j]
                }
            }
        }

        return targetOperator
    }

    // MARK: - Main Partitioning Algorithm

    /// Partitions terms into diagonalizable groups via greedy merging of QWC groups.
    ///
    /// **Example:**
    /// ```swift
    /// let partitions = UnitaryPartitioner().partition(terms: observable.terms)
    /// ```
    ///
    /// - Parameter terms: Pauli terms from ``Observable``
    /// - Returns: Diagonalizable partitions
    @_optimize(speed)
    @_eagerMove
    public func partition(terms: PauliTerms) -> [UnitaryPartition] {
        let qubits = terms.map { $0.pauliString.operators.map(\.qubit).max() ?? 0 }.max().map { $0 + 1 } ?? 0
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
                    qubits: qubits,
                ) {
                    currentTerms = mergedTerms
                    lastUnitary = unitary
                    remainingGroups.remove(at: i)
                } else {
                    i += 1
                }
            }

            if let unitary = lastUnitary ?? findDiagonalizingUnitary(terms: currentTerms, qubits: qubits) {
                partitions.append(UnitaryPartition(
                    terms: currentTerms,
                    unitaryMatrix: unitary,
                ))
            } else {
                let identity: [[Complex<Double>]] = MatrixUtilities.identityMatrix(dimension: 1 << qubits)
                partitions.append(UnitaryPartition(
                    terms: currentTerms,
                    unitaryMatrix: identity,
                ))
            }
        }

        return partitions
    }

    // MARK: - Unitary Optimization

    /// Finds diagonalizing unitary via eigendecomposition, falling back to variational optimization.
    @_optimize(speed)
    @_eagerMove
    private func findDiagonalizingUnitary(
        terms: PauliTerms,
        qubits: Int,
    ) -> [[Complex<Double>]]? {
        let targetOperator = buildTargetOperator(terms: terms, qubits: qubits)

        if let (_, eigenvectors) = eigendecompose(targetOperator) {
            let offDiagNorm: Double = computeOffDiagonalNorm(
                operator: targetOperator,
                unitary: eigenvectors,
            )

            if offDiagNorm < config.diagonalityThreshold {
                return eigenvectors
            }
        }

        return optimizeVariational(terms: terms, qubits: qubits)
    }

    /// Optimize unitary using variational ansatz.
    @_optimize(speed)
    @_eagerMove
    private func optimizeVariational(
        terms: PauliTerms,
        qubits: Int,
    ) -> [[Complex<Double>]]? {
        let pauliMatrices: [[[Complex<Double>]]] = terms.map { $0.pauliString.toMatrix(qubits: qubits) }

        let numParams: Int = parameterCount(qubits: qubits, depth: config.circuitDepth)
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
                    qubits: qubits,
                )
            },
            gradientFunction: { params in
                gradientFunctionCached(
                    parameters: params,
                    terms: terms,
                    pauliMatrices: pauliMatrices,
                    qubits: qubits,
                )
            },
            maxIterations: config.maxIterations,
            tolerance: config.convergenceTolerance,
        )

        let unitary: [[Complex<Double>]] = buildVariationalUnitary(
            parameters: result.parameters,
            qubits: qubits,
            depth: config.circuitDepth,
        )

        let targetOperator = buildTargetOperator(terms: terms, qubits: qubits)

        let offDiagNorm: Double = computeOffDiagonalNorm(operator: targetOperator, unitary: unitary)

        return offDiagNorm < config.diagonalityThreshold ? unitary : nil
    }

    // MARK: - Cost and Gradient Functions

    @_optimize(speed)
    private func costFunctionCached(
        parameters: [Double],
        terms: PauliTerms,
        pauliMatrices: [[[Complex<Double>]]],
        qubits: Int,
    ) -> Double {
        let unitary = buildVariationalUnitary(
            parameters: parameters,
            qubits: qubits,
            depth: config.circuitDepth,
        )

        let dimension = 1 << qubits
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
        qubits: Int,
    ) -> [Double] {
        let epsilon = 1e-7
        let paramCount = parameters.count

        let f0 = costFunctionCached(
            parameters: parameters,
            terms: terms,
            pauliMatrices: pauliMatrices,
            qubits: qubits,
        )

        var paramsPlus = parameters
        let gradient = [Double](unsafeUninitializedCapacity: paramCount) { buffer, count in
            for i in 0 ..< paramCount {
                paramsPlus[i] += epsilon

                let fPlus = costFunctionCached(
                    parameters: paramsPlus,
                    terms: terms,
                    pauliMatrices: pauliMatrices,
                    qubits: qubits,
                )
                buffer[i] = (fPlus - f0) / epsilon
                paramsPlus[i] -= epsilon
            }
            count = paramCount
        }

        return gradient
    }

    // MARK: - Variational Ansatz

    /// Parameter count: depth × qubits × 3 (U3 Euler angles per qubit per layer).
    @_effects(readonly)
    private func parameterCount(qubits: Int, depth: Int) -> Int {
        depth * qubits * 3
    }

    /// Builds unitary from hardware-efficient ansatz: [U3 rotations → CNOT ladder] × depth.
    @_optimize(speed)
    @_eagerMove
    private func buildVariationalUnitary(
        parameters: [Double],
        qubits: Int,
        depth: Int,
    ) -> [[Complex<Double>]] {
        let dimension = 1 << qubits
        var unitary: [[Complex<Double>]] = MatrixUtilities.identityMatrix(dimension: dimension)

        let cnotMatrices: [[[Complex<Double>]]] = (0 ..< (qubits - 1)).map { qubit in
            cnotMatrix(control: qubit, target: qubit + 1, qubits: qubits)
        }

        var paramIndex = 0

        for _ in 0 ..< depth {
            for qubit in 0 ..< qubits {
                let theta: Double = parameters[paramIndex]
                let phi: Double = parameters[paramIndex + 1]
                let lambda: Double = parameters[paramIndex + 2]
                paramIndex += 3

                let rotation: [[Complex<Double>]] = singleQubitRotation(
                    qubit: qubit,
                    theta: theta,
                    phi: phi,
                    lambda: lambda,
                    qubits: qubits,
                )

                unitary = MatrixUtilities.matrixMultiply(rotation, unitary)
            }

            for (index, cnot) in cnotMatrices.enumerated() where index < qubits - 1 {
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
        unitary: [[Complex<Double>]],
    ) -> [[Complex<Double>]] {
        let unitaryDagger: [[Complex<Double>]] = MatrixUtilities.hermitianConjugate(unitary)
        let temp: [[Complex<Double>]] = MatrixUtilities.matrixMultiply(unitaryDagger, matrix)
        return MatrixUtilities.matrixMultiply(temp, unitary)
    }

    @_optimize(speed)
    @_effects(readonly)
    private func computeOffDiagonalNorm(
        operator matrix: [[Complex<Double>]],
        unitary: [[Complex<Double>]],
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
        qubits: Int,
    ) -> [[Complex<Double>]] {
        let c: Double = cos(theta / 2)
        let s: Double = sin(theta / 2)

        let u3: [[Complex<Double>]] = [
            [Complex(c), Complex(-cos(lambda) * s, -sin(lambda) * s)],
            [Complex(cos(phi) * s, sin(phi) * s), Complex(cos(phi + lambda) * c, sin(phi + lambda) * c)],
        ]

        return embedSingleQubitGate(u3, qubit: qubit, qubits: qubits)
    }

    @_optimize(speed)
    @_eagerMove
    private func cnotMatrix(control: Int, target: Int, qubits: Int) -> [[Complex<Double>]] {
        let dimension = 1 << qubits
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
        qubits: Int,
    ) -> [[Complex<Double>]] {
        let dimension = 1 << qubits
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

    /// Diagonalizes Hermitian matrix via LAPACK zheev. Returns nil on failure.
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

        var rwork = [Double](unsafeUninitializedCapacity: max(1, 3 * n - 2)) { _, count in count = max(1, 3 * n - 2) }

        var workQuery = [Double](unsafeUninitializedCapacity: 2) { _, count in count = 2 }

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
                            &info,
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
        var work = [Double](unsafeUninitializedCapacity: 2 * optimalWorkSize) { _, count in count = 2 * optimalWorkSize }

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
                            &info,
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

    @frozen
    public struct UnitaryOptimizationResult {
        public let parameters: [Double]
        public let finalCost: Double
        public let iterations: Int
        public let converged: Bool
    }

    /// L-BFGS-B optimizer with Wolfe line search. Converges when ||∇|| < tolerance.
    @_optimize(speed)
    @_eagerMove
    private func lbfgsb(
        initialParameters: [Double],
        costFunction: ([Double]) -> Double,
        gradientFunction: ([Double]) -> [Double],
        maxIterations: Int,
        tolerance: Double,
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
                rhoHistory: rhoHistory,
            )

            guard let alpha = lineSearch(
                params: params,
                direction: direction,
                gradient: gradient,
                cost: cost,
                costFunction: costFunction,
                gradientFunction: gradientFunction,
                c1: c1,
                c2: c2,
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
            converged: converged,
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
        c2: Double,
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
    /// Computes P|row⟩ = phase × |col⟩ for Pauli string P acting on basis state |row⟩.
    ///
    /// X flips bit, Y flips bit with ±i phase, Z applies ±1 phase. Multi-qubit strings
    /// combine via tensor product (phases multiply).
    ///
    /// **Example:**
    /// ```swift
    /// let (col, phase) = PauliString([.y(0)]).applyToRow(row: 0)  // col=1, phase=i
    /// ```
    ///
    /// - Parameter row: Basis state index in [0, 2^n)
    /// - Returns: Resulting basis state index and accumulated phase
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

    /// Converts Pauli string to dense 2^n × 2^n matrix via ``applyToRow(row:)``.
    ///
    /// **Example:**
    /// ```swift
    /// let matrix = PauliString([.x(0)]).toMatrix(qubits: 1)
    /// ```
    ///
    /// - Parameter qubits: Total qubits in system
    /// - Returns: Dense matrix with 2^n non-zero entries (one per row)
    /// - Complexity: O(2^(2n))
    @_optimize(speed)
    @_eagerMove
    func toMatrix(qubits: Int) -> [[Complex<Double>]] {
        let dimension = 1 << qubits
        var result = Array(repeating: Array(repeating: Complex<Double>.zero, count: dimension), count: dimension)

        for row in 0 ..< dimension {
            let (col, phase) = applyToRow(row: row)
            result[row][col] += phase
        }

        return result
    }
}
