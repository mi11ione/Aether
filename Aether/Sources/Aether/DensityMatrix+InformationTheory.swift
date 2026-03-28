// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Eigenvalue threshold below which contributions to entropy are treated as zero.
private let entropyEigenvalueThreshold: Double = 1e-15

/// Precomputed (sigma_y tensor sigma_y) matrix for Wootters concurrence formula.
private let sigmaYTensorSigmaY: [[Complex<Double>]] = [
    [.zero, .zero, .zero, Complex(-1, 0)],
    [.zero, .zero, Complex(1, 0), .zero],
    [.zero, Complex(1, 0), .zero, .zero],
    [Complex(-1, 0), .zero, .zero, .zero],
]

public extension DensityMatrix {
    /// Compute von Neumann entropy S(rho) = -Tr(rho log2 rho) via eigendecomposition.
    ///
    /// Diagonalizes the density matrix to obtain eigenvalues (populations), then computes
    /// Shannon entropy of the eigenvalue distribution. Pure states have S = 0 while the
    /// maximally mixed state of n qubits has S = n (maximum entropy). Negative eigenvalues
    /// from numerical noise are clamped to zero before entropy computation.
    ///
    /// **Example:**
    /// ```swift
    /// let pure = DensityMatrix(qubits: 1)
    /// pure.vonNeumannEntropy()  // 0.0
    ///
    /// let mixed = DensityMatrix.maximallyMixed(qubits: 2)
    /// mixed.vonNeumannEntropy()  // 2.0
    /// ```
    ///
    /// - Returns: Von Neumann entropy in bits, range [0, n] where n is qubit count
    /// - Complexity: O(d^3) where d = 2^n for eigendecomposition
    @_optimize(speed)
    @_effects(readonly)
    func vonNeumannEntropy() -> Double {
        let matrix = extractMatrix()
        let result = HermitianEigenDecomposition.decompose(matrix: matrix)
        var entropy = 0.0
        for eigenvalue in result.eigenvalues {
            let p = max(0.0, eigenvalue)
            if p < entropyEigenvalueThreshold { continue }
            entropy -= p * log2(p)
        }
        return entropy
    }

    /// Compute trace distance D(rho, sigma) = 0.5 * Tr(|rho - sigma|) between two density matrices.
    ///
    /// The trace distance measures distinguishability of quantum states, ranging from 0 (identical)
    /// to 1 (perfectly distinguishable orthogonal states). Computed by eigendecomposing the
    /// difference matrix and summing absolute eigenvalues. Invariant under unitary transformations
    /// and satisfies the triangle inequality.
    ///
    /// **Example:**
    /// ```swift
    /// let rho = DensityMatrix(qubits: 1)
    /// let sigma = DensityMatrix.basis(qubits: 1, state: 1)
    /// rho.traceDistance(to: sigma)  // 1.0 (orthogonal states)
    ///
    /// let same = DensityMatrix(qubits: 1)
    /// rho.traceDistance(to: same)  // 0.0 (identical states)
    /// ```
    ///
    /// - Parameter other: Second density matrix to compare
    /// - Returns: Trace distance in [0, 1]
    /// - Complexity: O(d^3) where d = 2^n for eigendecomposition
    /// - Precondition: Both density matrices must have the same number of qubits
    @_optimize(speed)
    @_effects(readonly)
    func traceDistance(to other: DensityMatrix) -> Double {
        ValidationUtilities.validateQubitCountsEqual(qubits, other.qubits, name1: "self", name2: "other")

        let dim = dimension
        let delta: [[Complex<Double>]] = (0 ..< dim).map { i in
            [Complex<Double>](unsafeUninitializedCapacity: dim) { buffer, count in
                for j in 0 ..< dim {
                    buffer[j] = self[row: i, col: j] - other[row: i, col: j]
                }
                count = dim
            }
        }

        let result = HermitianEigenDecomposition.decompose(matrix: delta)
        let absEigenvalues = vDSP.absolute(result.eigenvalues)
        let sum = vDSP.sum(absEigenvalues)
        return 0.5 * sum
    }

    /// Compute quantum state fidelity F(rho, sigma) = (Tr(sqrt(sqrt(rho) * sigma * sqrt(rho))))^2.
    ///
    /// Fidelity quantifies overlap between quantum states, ranging from 0 (orthogonal) to 1
    /// (identical). For pure states, simplifies to F = Tr(rho * sigma). The general algorithm
    /// constructs sqrt(rho) via eigendecomposition, forms the product matrix, and extracts
    /// fidelity from its eigenvalues. Satisfies Jozsa axioms: symmetry, bounded, and
    /// F(rho, sigma) = 1 iff rho = sigma.
    ///
    /// **Example:**
    /// ```swift
    /// let rho = DensityMatrix(qubits: 1)
    /// let sigma = DensityMatrix(qubits: 1)
    /// rho.fidelity(to: sigma)  // 1.0 (identical states)
    ///
    /// let ortho = DensityMatrix.basis(qubits: 1, state: 1)
    /// rho.fidelity(to: ortho)  // 0.0 (orthogonal states)
    /// ```
    ///
    /// - Parameter other: Second density matrix to compare
    /// - Returns: Fidelity in [0, 1]
    /// - Complexity: O(d^3) where d = 2^n for eigendecomposition and matrix multiplications
    /// - Precondition: Both density matrices must have the same number of qubits
    @_optimize(speed)
    @_effects(readonly)
    func fidelity(to other: DensityMatrix) -> Double {
        ValidationUtilities.validateQubitCountsEqual(qubits, other.qubits, name1: "self", name2: "other")

        let dim = dimension

        if isPure() {
            var f = 0.0
            for i in 0 ..< dim {
                for j in 0 ..< dim {
                    let prod = self[row: i, col: j] * other[row: j, col: i]
                    f += prod.real
                }
            }
            return max(0.0, min(1.0, f))
        }

        let sqrtRho = squareRootMatrix()

        let otherMatrix: [[Complex<Double>]] = (0 ..< dim).map { i in
            [Complex<Double>](unsafeUninitializedCapacity: dim) { buffer, count in
                for j in 0 ..< dim {
                    buffer[j] = other[row: i, col: j]
                }
                count = dim
            }
        }

        let sqrtRhoTimesSigma = MatrixUtilities.matrixMultiply(sqrtRho, otherMatrix)
        let m = MatrixUtilities.matrixMultiply(sqrtRhoTimesSigma, sqrtRho)

        let mEigen = HermitianEigenDecomposition.decompose(matrix: m)

        let clampedMu = vDSP.clip(mEigen.eigenvalues, to: 0.0 ... .greatestFiniteMagnitude)
        var sqrtMu = [Double](repeating: 0.0, count: clampedMu.count)
        vForce.sqrt(clampedMu, result: &sqrtMu)
        let sumSqrt = vDSP.sum(sqrtMu)

        return max(0.0, min(1.0, sumSqrt * sumSqrt))
    }

    /// Compute quantum mutual information I(A:B) = S(A) + S(B) - S(AB) between two subsystems.
    ///
    /// Mutual information quantifies total correlations (both classical and quantum) between
    /// subsystems A and B. Computed from von Neumann entropies of the reduced density matrices
    /// obtained via partial trace. Always non-negative by subadditivity of von Neumann entropy.
    /// For pure bipartite states, I(A:B) = 2*S(A) = 2*S(B).
    ///
    /// **Example:**
    /// ```swift
    /// let bell = QuantumState(qubits: 2, amplitudes: [
    ///     Complex(1/sqrt(2), 0), .zero, .zero, Complex(1/sqrt(2), 0)
    /// ])
    /// let dm = DensityMatrix(pureState: bell)
    /// dm.mutualInformation(between: [0], and: [1])  // 2.0
    /// ```
    ///
    /// - Parameters:
    ///   - subsystemA: Qubit indices for subsystem A
    ///   - subsystemB: Qubit indices for subsystem B
    /// - Returns: Mutual information in bits (non-negative)
    /// - Complexity: O(d^3) where d = 2^n for eigendecompositions
    /// - Precondition: subsystemA and subsystemB must be disjoint and non-empty
    /// - Precondition: All qubit indices must be valid
    @_optimize(speed)
    @_effects(readonly)
    func mutualInformation(between subsystemA: [Int], and subsystemB: [Int]) -> Double {
        ValidationUtilities.validateDisjointSubsystems(subsystemA, subsystemB)
        ValidationUtilities.validateNonEmpty(subsystemA, name: "subsystemA")
        ValidationUtilities.validateNonEmpty(subsystemB, name: "subsystemB")
        ValidationUtilities.validateOperationQubits(subsystemA, numQubits: qubits)
        ValidationUtilities.validateOperationQubits(subsystemB, numQubits: qubits)

        let complementOfA = (0 ..< qubits).filter { !subsystemA.contains($0) }
        let complementOfB = (0 ..< qubits).filter { !subsystemB.contains($0) }

        let sAB = vonNeumannEntropy()

        let rhoA = partialTrace(over: complementOfA)
        let sA = rhoA.vonNeumannEntropy()

        let rhoB = partialTrace(over: complementOfB)
        let sB = rhoB.vonNeumannEntropy()

        return sA + sB - sAB
    }

    /// Compute concurrence C(rho) for a two-qubit density matrix using the Wootters formula.
    ///
    /// Concurrence measures pairwise entanglement for two-qubit systems, ranging from 0
    /// (separable) to 1 (maximally entangled). Uses the Wootters formula: construct
    /// rho_tilde = (sigma_y tensor sigma_y) * rho* * (sigma_y tensor sigma_y), compute
    /// M = sqrt(rho) * rho_tilde * sqrt(rho), then C = max(0, sqrt(lambda_1) - sqrt(lambda_2)
    /// - sqrt(lambda_3) - sqrt(lambda_4)) where lambda_i are eigenvalues of M in descending order.
    ///
    /// **Example:**
    /// ```swift
    /// let bell = QuantumState(qubits: 2, amplitudes: [
    ///     Complex(1/sqrt(2), 0), .zero, .zero, Complex(1/sqrt(2), 0)
    /// ])
    /// let dm = DensityMatrix(pureState: bell)
    /// dm.concurrence()  // 1.0 (maximally entangled)
    ///
    /// let product = DensityMatrix(qubits: 2)
    /// product.concurrence()  // 0.0 (separable)
    /// ```
    ///
    /// - Returns: Concurrence in [0, 1]
    /// - Complexity: O(1) (fixed 4x4 matrix operations)
    /// - Precondition: Density matrix must be a two-qubit system (qubits == 2)
    @_optimize(speed)
    @_effects(readonly)
    func concurrence() -> Double {
        ValidationUtilities.validateQubitCountsEqual(qubits, 2, name1: "density matrix", name2: "concurrence requirement")

        let dim = dimension
        let rho = extractMatrix()

        let sqrtRho = squareRootMatrix()

        let syy = sigmaYTensorSigmaY

        let rhoConjugate: [[Complex<Double>]] = (0 ..< dim).map { i in
            [Complex<Double>](unsafeUninitializedCapacity: dim) { buffer, count in
                for j in 0 ..< dim {
                    buffer[j] = rho[i][j].conjugate
                }
                count = dim
            }
        }

        let syyTimesRhoConj = MatrixUtilities.matrixMultiply(syy, rhoConjugate)
        let rhoTilde = MatrixUtilities.matrixMultiply(syyTimesRhoConj, syy)

        let sqrtRhoTimesRhoTilde = MatrixUtilities.matrixMultiply(sqrtRho, rhoTilde)
        let m = MatrixUtilities.matrixMultiply(sqrtRhoTimesRhoTilde, sqrtRho)

        let mEigen = HermitianEigenDecomposition.decompose(matrix: m)

        var sqrtLambdas = [Double](unsafeUninitializedCapacity: dim) {
            buffer, count in
            for i in 0 ..< dim {
                buffer[i] = sqrt(max(0.0, mEigen.eigenvalues[i]))
            }
            count = dim
        }

        sqrtLambdas.sort(by: >)

        let c = sqrtLambdas[0] - sqrtLambdas[1] - sqrtLambdas[2] - sqrtLambdas[3]
        return max(0.0, c)
    }
}

// MARK: - Private Helpers

private extension DensityMatrix {
    /// Computes the matrix square root via eigendecomposition.
    @_effects(readonly)
    func squareRootMatrix() -> [[Complex<Double>]] {
        let dim = dimension
        let matrix = extractMatrix()
        let eigen = HermitianEigenDecomposition.decompose(matrix: matrix)

        let eigenvalues = eigen.eigenvalues
        let sqrtDiag = [Double](unsafeUninitializedCapacity: eigenvalues.count) { buffer, count in
            for i in 0 ..< eigenvalues.count {
                buffer[i] = sqrt(max(0.0, eigenvalues[i]))
            }
            count = eigenvalues.count
        }

        let u = eigen.eigenvectors
        let uDagger = MatrixUtilities.hermitianConjugate(u)

        let diagSqrt: [[Complex<Double>]] = (0 ..< dim).map { i in
            [Complex<Double>](unsafeUninitializedCapacity: dim) { buffer, count in
                buffer.initialize(repeating: .zero)
                buffer[i] = Complex(sqrtDiag[i], 0)
                count = dim
            }
        }

        let uTimesDiag = MatrixUtilities.matrixMultiply(u, diagSqrt)
        return MatrixUtilities.matrixMultiply(uTimesDiag, uDagger)
    }
}

// MARK: - Internal Helpers

extension DensityMatrix {
    /// Extract density matrix elements into a two-dimensional array via the public accessor.
    ///
    /// - Returns: Two-dimensional array representation of the density matrix
    /// - Complexity: O(d^2) where d = 2^n
    @_effects(readonly)
    func extractMatrix() -> [[Complex<Double>]] {
        let dim = dimension
        return (0 ..< dim).map { i in
            [Complex<Double>](unsafeUninitializedCapacity: dim) { buffer, count in
                for j in 0 ..< dim {
                    buffer[j] = self[row: i, col: j]
                }
                count = dim
            }
        }
    }
}
