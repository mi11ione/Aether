// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate
import Foundation

/// A partition of Pauli terms grouped by a unitary transformation.
///
/// Terms in this partition can be measured simultaneously after applying
/// the unitary transformation U, which approximately diagonalizes all terms.
public struct UnitaryPartition: Sendable {
    /// Pauli strings with their coefficients
    public let terms: PauliTerms

    /// Unitary transformation matrix (2^n × 2^n) that diagonalizes the terms
    /// After applying U†, all Pauli operators in the partition become (nearly) diagonal
    public let unitaryMatrix: GateMatrix

    /// Number of qubits
    public let numQubits: Int

    /// Measurement basis after unitary transformation (computational basis)
    public var measurementBasis: MeasurementBasis {
        // After U†, measure in computational (Z) basis
        var basis: MeasurementBasis = [:]
        for term in terms {
            for op in term.pauliString.operators {
                basis[op.qubit] = .z
            }
        }
        return basis
    }

    /// Total weight (sum of absolute coefficients)
    public var weight: Double {
        terms.reduce(0.0) { $0 + abs($1.coefficient) }
    }
}

/// Finds optimal unitary transformations to group Pauli terms for measurement.
///
/// This is more powerful than QWC grouping: it finds a unitary U such that
/// U†PᵢU are all diagonal (or nearly diagonal), allowing simultaneous measurement
/// of terms that don't qubit-wise commute.
///
/// Algorithm:
/// 1. Start with QWC groups (baseline)
/// 2. For each group, try to merge with other groups by finding a unitary U
/// 3. Parameterize U as a quantum circuit (variational ansatz)
/// 4. Minimize: C(U) = Σᵢ wᵢ ||off-diagonal(U†PᵢU)||²
/// 5. Use L-BFGS-B for optimization
///
/// This can reduce 2000 terms → 50 QWC groups → 10-20 unitary partitions.
public struct UnitaryPartitioner {
    // MARK: - Configuration

    public struct Config: Sendable {
        /// Maximum number of optimization iterations
        public let maxIterations: Int

        /// Convergence tolerance for gradient norm
        public let tolerance: Double

        /// Number of layers in variational ansatz
        public let ansatzDepth: Int

        /// Whether to use adaptive layer depth
        public let adaptiveDepth: Bool

        /// Maximum off-diagonal norm to accept a partition (lower = stricter)
        public let maxOffDiagonalNorm: Double

        public static let `default` = Config(
            maxIterations: 100,
            tolerance: 1e-6,
            ansatzDepth: 3,
            adaptiveDepth: true,
            maxOffDiagonalNorm: 0.1
        )
    }

    public let config: Config

    public init(config: Config = .default) { self.config = config }

    // MARK: - Main Partitioning Algorithm

    /// Partition Pauli terms using unitary transformations.
    ///
    /// - Parameters:
    ///   - terms: Array of (coefficient, PauliString) pairs
    ///   - numQubits: Number of qubits
    /// - Returns: Array of unitary partitions
    ///
    /// Example:
    /// ```swift
    /// let partitioner = UnitaryPartitioner()
    /// let partitions = partitioner.partition(
    ///     terms: hamiltonian.terms,
    ///     numQubits: 10
    /// )
    /// print("Reduced \(hamiltonian.terms.count) terms to \(partitions.count) partitions")
    /// // Output: Reduced 2000 terms to 12 partitions (166× reduction)
    /// ```
    public func partition(
        terms: PauliTerms,
        numQubits: Int
    ) -> [UnitaryPartition] {
        let qwcGroups: [QWCGroup] = QWCGrouper.group(terms: terms)
        var partitions: [UnitaryPartition] = []
        var remainingGroups: [QWCGroup] = qwcGroups

        while !remainingGroups.isEmpty {
            let seed: QWCGroup = remainingGroups.removeFirst()
            var currentTerms = seed.terms
            var lastUnitary: GateMatrix?

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
                    unitaryMatrix: unitary,
                    numQubits: numQubits
                ))
            } else {
                let identity: GateMatrix = identityMatrix(dimension: 1 << numQubits)
                partitions.append(UnitaryPartition(
                    terms: currentTerms,
                    unitaryMatrix: identity,
                    numQubits: numQubits
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
    private func findDiagonalizingUnitary(
        terms: PauliTerms,
        numQubits: Int
    ) -> GateMatrix? {
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

        if let (_, eigenvectors) = eigendecompose(targetOperator) {
            let offDiagNorm: Double = computeOffDiagonalNorm(
                operator: targetOperator,
                unitary: eigenvectors
            )

            if offDiagNorm < config.maxOffDiagonalNorm {
                return eigenvectors
            }
        }

        return optimizeVariational(terms: terms, numQubits: numQubits)
    }

    /// Optimize unitary using variational ansatz.
    private func optimizeVariational(
        terms: PauliTerms,
        numQubits: Int
    ) -> GateMatrix? {
        let pauliMatrices: [GateMatrix] = terms.map { $0.pauliString.toMatrix(numQubits: numQubits) }

        let numParams: Int = variationalParameterCount(numQubits: numQubits, depth: config.ansatzDepth)
        var parameters: [Double] = Array(repeating: 0.0, count: numParams)

        for i in 0 ..< numParams {
            parameters[i] = Double.random(in: -Double.pi ... Double.pi)
        }

        let result: OptimizationResult = lbfgsb(
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
            tolerance: config.tolerance
        )

        let unitary: GateMatrix = buildVariationalUnitary(
            parameters: result.parameters,
            numQubits: numQubits,
            depth: config.ansatzDepth
        )

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

        let offDiagNorm: Double = computeOffDiagonalNorm(operator: targetOperator, unitary: unitary)

        return offDiagNorm < config.maxOffDiagonalNorm ? unitary : nil
    }

    // MARK: - Cost and Gradient Functions

    private func costFunctionCached(
        parameters: [Double],
        terms: PauliTerms,
        pauliMatrices: [GateMatrix],
        numQubits: Int
    ) -> Double {
        let unitary = buildVariationalUnitary(
            parameters: parameters,
            numQubits: numQubits,
            depth: config.ansatzDepth
        )

        let dimension = 1 << numQubits
        var cost = 0.0

        for (index, term) in terms.enumerated() {
            let pauliMatrix = pauliMatrices[index]
            let conjugated = conjugateByUnitary(pauliMatrix, unitary: unitary)

            for i in 0 ..< dimension {
                for j in 0 ..< dimension where i != j {
                    cost += abs(term.coefficient) * conjugated[i][j].magnitude * conjugated[i][j].magnitude
                }
            }
        }

        return cost
    }

    private func gradientFunctionCached(
        parameters: [Double],
        terms: PauliTerms,
        pauliMatrices: [GateMatrix],
        numQubits: Int
    ) -> [Double] {
        let epsilon = 1e-7
        var gradient = Array(repeating: 0.0, count: parameters.count)

        let f0 = costFunctionCached(
            parameters: parameters,
            terms: terms,
            pauliMatrices: pauliMatrices,
            numQubits: numQubits
        )

        for i in 0 ..< parameters.count {
            var paramsPlus = parameters
            paramsPlus[i] += epsilon

            let fPlus = costFunctionCached(
                parameters: paramsPlus,
                terms: terms,
                pauliMatrices: pauliMatrices,
                numQubits: numQubits
            )
            gradient[i] = (fPlus - f0) / epsilon
        }

        return gradient
    }

    // MARK: - Variational Ansatz

    /// Count parameters in variational ansatz.
    private func variationalParameterCount(numQubits: Int, depth: Int) -> Int {
        // Each layer: single-qubit rotations (3 params per qubit) + CNOT ladder
        depth * numQubits * 3
    }

    /// Build unitary from variational parameters.
    private func buildVariationalUnitary(
        parameters: [Double],
        numQubits: Int,
        depth: Int
    ) -> GateMatrix {
        let dimension = 1 << numQubits
        var unitary: GateMatrix = identityMatrix(dimension: dimension)

        var paramIndex = 0

        for _ in 0 ..< depth {
            for qubit in 0 ..< numQubits {
                let theta: Double = parameters[paramIndex]
                let phi: Double = parameters[paramIndex + 1]
                let lambda: Double = parameters[paramIndex + 2]
                paramIndex += 3

                let rotation: GateMatrix = singleQubitRotation(
                    qubit: qubit,
                    theta: theta,
                    phi: phi,
                    lambda: lambda,
                    numQubits: numQubits
                )

                unitary = matrixMultiply(rotation, unitary)
            }

            for qubit in 0 ..< (numQubits - 1) {
                let cnot: GateMatrix = cnotMatrix(control: qubit, target: qubit + 1, numQubits: numQubits)
                unitary = matrixMultiply(cnot, unitary)
            }
        }

        return unitary
    }

    // MARK: - Matrix Utilities

    private func identityMatrix(dimension: Int) -> GateMatrix {
        var matrix = Array(
            repeating: Array(repeating: Complex<Double>.zero, count: dimension),
            count: dimension
        )
        for i in 0 ..< dimension {
            matrix[i][i] = .one
        }
        return matrix
    }

    private func matrixMultiply(
        _ a: GateMatrix,
        _ b: GateMatrix
    ) -> GateMatrix {
        let n: Int = a.count

        // Fast path for small matrices
        if n <= 4 {
            var result = Array(repeating: Array(repeating: Complex<Double>.zero, count: n), count: n)
            for i in 0 ..< n {
                for j in 0 ..< n {
                    for k in 0 ..< n {
                        result[i][j] += a[i][k] * b[k][j]
                    }
                }
            }
            return result
        }

        // Use BLAS zgemm for larger matrices
        // Convert to interleaved format for BLAS (column-major)
        var aInterleaved = [Double](unsafeUninitializedCapacity: 2 * n * n) { buffer, count in
            for col in 0 ..< n {
                for row in 0 ..< n {
                    buffer[(col * n + row) * 2] = a[row][col].real
                    buffer[(col * n + row) * 2 + 1] = a[row][col].imaginary
                }
            }
            count = n * n * 2
        }

        var bInterleaved = [Double](unsafeUninitializedCapacity: 2 * n * n) { buffer, count in
            for col in 0 ..< n {
                for row in 0 ..< n {
                    buffer[(col * n + row) * 2] = b[row][col].real
                    buffer[(col * n + row) * 2 + 1] = b[row][col].imaginary
                }
            }
            count = n * n * 2
        }

        var cInterleaved = [Double](repeating: 0.0, count: 2 * n * n)

        var alpha: [Double] = [1.0, 0.0]
        var beta: [Double] = [0.0, 0.0]

        aInterleaved.withUnsafeMutableBufferPointer { aPtr in
            bInterleaved.withUnsafeMutableBufferPointer { bPtr in
                cInterleaved.withUnsafeMutableBufferPointer { cPtr in
                    alpha.withUnsafeMutableBufferPointer { alphaPtr in
                        beta.withUnsafeMutableBufferPointer { betaPtr in
                            cblas_zgemm(
                                CblasColMajor,
                                CblasNoTrans,
                                CblasNoTrans,
                                Int32(n), Int32(n), Int32(n),
                                OpaquePointer(alphaPtr.baseAddress)!,
                                OpaquePointer(aPtr.baseAddress), Int32(n),
                                OpaquePointer(bPtr.baseAddress), Int32(n),
                                OpaquePointer(betaPtr.baseAddress)!,
                                OpaquePointer(cPtr.baseAddress), Int32(n)
                            )
                        }
                    }
                }
            }
        }

        // Convert back to Complex<Double> matrix
        var result = Array(repeating: Array(repeating: Complex<Double>.zero, count: n), count: n)
        for col in 0 ..< n {
            for row in 0 ..< n {
                let idx = (col * n + row) * 2
                result[row][col] = Complex(cInterleaved[idx], cInterleaved[idx + 1])
            }
        }

        return result
    }

    private func conjugateByUnitary(
        _ matrix: GateMatrix,
        unitary: GateMatrix
    ) -> GateMatrix {
        let unitaryDagger: GateMatrix = hermitianConjugate(unitary)
        let temp: GateMatrix = matrixMultiply(unitaryDagger, matrix)
        return matrixMultiply(temp, unitary)
    }

    private func hermitianConjugate(_ matrix: GateMatrix) -> GateMatrix {
        let n: Int = matrix.count
        var result = Array(repeating: Array(repeating: Complex<Double>.zero, count: n), count: n)

        for i in 0 ..< n {
            for j in 0 ..< n {
                result[i][j] = matrix[j][i].conjugate
            }
        }

        return result
    }

    private func computeOffDiagonalNorm(
        operator matrix: GateMatrix,
        unitary: GateMatrix
    ) -> Double {
        let conjugated: GateMatrix = conjugateByUnitary(matrix, unitary: unitary)
        let n: Int = conjugated.count

        var norm = 0.0
        for i in 0 ..< n {
            for j in 0 ..< n where i != j {
                norm += conjugated[i][j].magnitude * conjugated[i][j].magnitude
            }
        }

        return sqrt(norm)
    }

    private func singleQubitRotation(
        qubit: Int,
        theta: Double,
        phi: Double,
        lambda: Double,
        numQubits: Int
    ) -> GateMatrix {
        let c: Double = cos(theta / 2)
        let s: Double = sin(theta / 2)

        let u3: GateMatrix = [
            [Complex(c), Complex(-cos(lambda) * s, -sin(lambda) * s)],
            [Complex(cos(phi) * s, sin(phi) * s), Complex(cos(phi + lambda) * c, sin(phi + lambda) * c)],
        ]

        return embedSingleQubitGate(u3, qubit: qubit, numQubits: numQubits)
    }

    private func cnotMatrix(control: Int, target: Int, numQubits: Int) -> GateMatrix {
        let dimension = 1 << numQubits
        var cnot: GateMatrix = identityMatrix(dimension: dimension)

        for basis in 0 ..< dimension {
            let controlBit: Int = (basis >> control) & 1
            if controlBit == 1 {
                let flippedBasis: Int = basis ^ (1 << target)
                if flippedBasis != basis {
                    cnot[basis][basis] = .zero
                    cnot[basis][flippedBasis] = .one
                }
            }
        }

        return cnot
    }

    private func embedSingleQubitGate(
        _ gate: GateMatrix,
        qubit: Int,
        numQubits: Int
    ) -> GateMatrix {
        let dimension = 1 << numQubits
        var result = Array(repeating: Array(repeating: Complex<Double>.zero, count: dimension), count: dimension)

        for row in 0 ..< dimension {
            for col in 0 ..< dimension {
                let rowBit: Int = (row >> qubit) & 1
                let colBit: Int = (col >> qubit) & 1

                let rowRest: Int = row & ~(1 << qubit)
                let colRest: Int = col & ~(1 << qubit)

                if rowRest == colRest {
                    result[row][col] = gate[rowBit][colBit]
                }
            }
        }

        return result
    }

    // MARK: - Eigendecomposition

    /// Compute eigendecomposition of Hermitian matrix using LAPACK's zheev.
    ///
    /// - Parameter matrix: Hermitian matrix (n × n)
    /// - Returns: Eigenvalues (real, sorted) and eigenvectors (columns), or nil if decomposition fails
    private func eigendecompose(_ matrix: GateMatrix) -> (eigenvalues: [Double], eigenvectors: GateMatrix)? {
        guard !matrix.isEmpty else { return nil }

        let n: Int = matrix.count

        var a = [Double]()
        a.reserveCapacity(2 * n * n)

        for col in 0 ..< n {
            for row in 0 ..< n {
                a.append(matrix[row][col].real)
                a.append(matrix[row][col].imaginary)
            }
        }

        var w = [Double](repeating: 0.0, count: n)

        var jobz = CChar(Character("V").asciiValue!) // Compute eigenvectors
        var uplo = CChar(Character("U").asciiValue!) // Upper triangle
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

        var eigenvectors = Array(
            repeating: Array(repeating: Complex<Double>.zero, count: n),
            count: n
        )

        for col in 0 ..< n {
            for row in 0 ..< n {
                let idx = 2 * (col * n + row)
                eigenvectors[row][col] = Complex(a[idx], a[idx + 1])
            }
        }

        return (eigenvalues: w, eigenvectors: eigenvectors)
    }

    // MARK: - L-BFGS-B Optimizer

    public struct OptimizationResult {
        public let parameters: [Double]
        public let finalCost: Double
        public let iterations: Int
        public let converged: Bool
    }

    /// L-BFGS optimizer (Limited-memory Broyden-Fletcher-Goldfarb-Shanno).
    ///
    /// Uses two-loop recursion to compute search directions with limited memory.
    /// Implements backtracking line search with Wolfe conditions.
    ///
    /// - Parameters:
    ///   - initialParameters: Starting point
    ///   - costFunction: Objective function to minimize
    ///   - gradientFunction: Gradient of objective
    ///   - maxIterations: Maximum iterations
    ///   - tolerance: Convergence tolerance on gradient norm
    /// - Returns: Optimization result with final parameters and convergence info
    private func lbfgsb(
        initialParameters: [Double],
        costFunction: ([Double]) -> Double,
        gradientFunction: ([Double]) -> [Double],
        maxIterations: Int,
        tolerance: Double
    ) -> OptimizationResult {
        let memorySize = 10 // L-BFGS memory parameter (m)
        let c1 = 1e-4 // Wolfe condition constant
        let c2 = 0.9 // Curvature condition constant

        var params = initialParameters
        var gradient = gradientFunction(params)
        var cost = costFunction(params)

        // History buffers for s_k and y_k
        var sHistory: [[Double]] = []
        var yHistory: [[Double]] = []
        var rhoHistory: [Double] = []

        var iteration = 0
        var converged = false

        while iteration < maxIterations {
            let gradNorm: Double = sqrt(gradient.reduce(0.0) { $0 + $1 * $1 })

            if gradNorm < tolerance {
                converged = true
                break
            }

            let direction: [Double] = computeSearchDirection(
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

            let newParams: [Double] = zip(params, direction).map { $0 + alpha * $1 }
            let newGradient: [Double] = gradientFunction(newParams)
            let newCost: Double = costFunction(newParams)

            let s: [Double] = zip(newParams, params).map { $0 - $1 }
            let y: [Double] = zip(newGradient, gradient).map { $0 - $1 }

            let ys: Double = zip(y, s).reduce(0.0) { $0 + $1.0 * $1.1 }

            if ys > 1e-10 {
                let rho = 1.0 / ys

                sHistory.append(s)
                yHistory.append(y)
                rhoHistory.append(rho)

                if sHistory.count > memorySize {
                    sHistory.removeFirst()
                    yHistory.removeFirst()
                    rhoHistory.removeFirst()
                }
            }

            params = newParams
            gradient = newGradient
            cost = newCost
            iteration += 1
        }

        return OptimizationResult(
            parameters: params,
            finalCost: cost,
            iterations: iteration,
            converged: converged
        )
    }

    /// Compute search direction using L-BFGS two-loop recursion.
    private func computeSearchDirection(
        gradient: [Double],
        sHistory: [[Double]],
        yHistory: [[Double]],
        rhoHistory: [Double]
    ) -> [Double] {
        guard !sHistory.isEmpty else { return gradient.map { -$0 } }

        let m: Int = sHistory.count
        var q = gradient
        var alpha = [Double](repeating: 0.0, count: m)

        for i in stride(from: m - 1, through: 0, by: -1) {
            let a: Double = rhoHistory[i] * zip(sHistory[i], q).reduce(0.0) { $0 + $1.0 * $1.1 }
            alpha[i] = a
            q = zip(q, yHistory[i]).map { $0 - a * $1 }
        }

        let lastS: [Double] = sHistory[m - 1]
        let lastY: [Double] = yHistory[m - 1]
        let sy: Double = zip(lastS, lastY).reduce(0.0) { $0 + $1.0 * $1.1 }
        let yy: Double = lastY.reduce(0.0) { $0 + $1 * $1 }
        let gamma: Double = sy / yy

        var r: [Double] = q.map { gamma * $0 }

        for i in 0 ..< m {
            let beta: Double = rhoHistory[i] * zip(yHistory[i], r).reduce(0.0) { $0 + $1.0 * $1.1 }
            r = zip(r, sHistory[i]).map { $0 + (alpha[i] - beta) * $1 }
        }

        return r.map { -$0 }
    }

    /// Backtracking line search with Wolfe conditions.
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

        let dirGrad: Double = zip(direction, gradient).reduce(0.0) { $0 + $1.0 * $1.1 }

        guard dirGrad < 0 else { return nil }

        for _ in 0 ..< maxBacktrack {
            let newParams: [Double] = zip(params, direction).map { $0 + alpha * $1 }
            let newCost: Double = costFunction(newParams)

            if newCost <= cost + c1 * alpha * dirGrad {
                let newGradient: [Double] = gradientFunction(newParams)
                let newDirGrad: Double = zip(direction, newGradient).reduce(0.0) { $0 + $1.0 * $1.1 }

                if abs(newDirGrad) <= -c2 * dirGrad { return alpha }
            }

            alpha *= rho
        }

        return alpha > 1e-10 ? alpha : nil
    }
}

// MARK: - PauliString Matrix Extension

public extension PauliString {
    /// Convert Pauli string to matrix representation.
    func toMatrix(numQubits: Int) -> GateMatrix {
        let dimension = 1 << numQubits
        var result = Array(repeating: Array(repeating: Complex<Double>.zero, count: dimension), count: dimension)

        var ops: MeasurementBasis = [:]
        for op in operators {
            ops[op.qubit] = op.basis
        }

        for row in 0 ..< dimension {
            var col: Int = row
            var phase = Complex<Double>.one

            for qubit in 0 ..< numQubits {
                let rowBit: Int = (row >> qubit) & 1

                if let basis = ops[qubit] {
                    switch basis {
                    case .x: col ^= (1 << qubit)

                    case .y:
                        col ^= (1 << qubit)
                        phase *= rowBit == 0 ? -Complex<Double>.i : Complex<Double>.i

                    case .z: phase *= rowBit == 0 ? .one : -.one
                    }
                }
                // If basis is nil (identity), do nothing
            }

            result[row][col] += phase
        }

        return result
    }
}
