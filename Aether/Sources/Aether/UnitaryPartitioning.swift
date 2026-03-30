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

    /// Diagonalizing unitary (2^n * 2^n).
    public let unitaryMatrix: [[Complex<Double>]]

    /// Measurement basis per qubit (all Z after U†).
    public let measurementBasis: [Int: PauliBasis]

    public init(terms: PauliTerms, unitaryMatrix: [[Complex<Double>]]) {
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
    ///
    /// **Example:**
    /// ```swift
    /// let w = partition.weight()
    /// ```
    ///
    /// - Returns: Total absolute coefficient weight
    /// - Complexity: O(n) where n is the number of terms
    /// - SeeAlso: ``UnitaryPartitioner``
    @inlinable
    public func weight() -> Double {
        terms.reduce(0.0) { $0 + abs($1.coefficient) }
    }
}

/// Groups Pauli terms into partitions diagonalizable by unitary transformations.
///
/// Starts with QWC groups, then greedily merges using eigendecomposition to find
/// diagonalizing unitaries that pass the off-diagonal norm threshold.
///
/// **Example:**
/// ```swift
/// let partitioner = UnitaryPartitioner(diagonalityThreshold: 0.1)
/// let partitions = partitioner.partition(terms: observable.terms)
/// ```
///
/// - SeeAlso: ``UnitaryPartition``
/// - SeeAlso: ``QWCGrouper``
@frozen
public struct UnitaryPartitioner: Sendable {
    /// Max off-diagonal norm to accept partition.
    public let diagonalityThreshold: Double

    /// Creates a partitioner with the given diagonality threshold.
    ///
    /// **Example:**
    /// ```swift
    /// let partitioner = UnitaryPartitioner(diagonalityThreshold: 0.1)
    /// ```
    ///
    /// - Parameter diagonalityThreshold: Max off-diagonal norm to accept partition (default: 0.1)
    public init(
        diagonalityThreshold: Double = 0.1,
    ) {
        self.diagonalityThreshold = diagonalityThreshold
    }

    /// Build target operator matrix from Pauli terms: H = Σᵢ cᵢ Pᵢ
    @_optimize(speed)
    @_eagerMove
    private func buildTargetOperator(
        terms: PauliTerms,
        qubits: Int,
        precomputedMatrices: [[[Complex<Double>]]]? = nil,
    ) -> [[Complex<Double>]] {
        let dimension = 1 << qubits
        var targetOperator = [[Complex<Double>]](unsafeUninitializedCapacity: dimension) {
            buffer, count in
            for i in 0 ..< dimension {
                buffer.initializeElement(at: i, to: [Complex<Double>](repeating: .zero, count: dimension))
            }
            count = dimension
        }

        for (index, (coeff, pauliString)) in terms.enumerated() {
            let pauliMatrix = precomputedMatrices?[index] ?? pauliString.matrix(qubits: qubits)
            for i in 0 ..< dimension {
                for j in 0 ..< dimension {
                    targetOperator[i][j] += Complex(coeff) * pauliMatrix[i][j]
                }
            }
        }

        return targetOperator
    }

    /// Partitions terms into diagonalizable groups via greedy merging of QWC groups.
    ///
    /// **Example:**
    /// ```swift
    /// let partitions = UnitaryPartitioner().partition(terms: observable.terms)
    /// ```
    ///
    /// - Parameter terms: Pauli terms from ``Observable``
    /// - Returns: Diagonalizable partitions
    /// - Complexity: O(g^2 * 2^(3n)) where g is the number of QWC groups and n is qubit count
    /// - SeeAlso: ``UnitaryPartition``
    @_optimize(speed)
    @_eagerMove
    public func partition(terms: PauliTerms) -> [UnitaryPartition] {
        var maxQubit = -1
        for term in terms {
            for op in term.pauliString.operators {
                if op.qubit > maxQubit { maxQubit = op.qubit }
            }
        }
        let qubits = maxQubit + 1
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
                    remainingGroups.swapAt(i, remainingGroups.count - 1)
                    remainingGroups.removeLast()
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

    /// Finds diagonalizing unitary via eigendecomposition.
    @_optimize(speed)
    @_eagerMove
    private func findDiagonalizingUnitary(
        terms: PauliTerms,
        qubits: Int,
    ) -> [[Complex<Double>]]? {
        let targetOperator = buildTargetOperator(terms: terms, qubits: qubits)

        let (_, eigenvectors) = eigendecompose(targetOperator)
        let offDiagNorm: Double = computeOffDiagonalNorm(
            operator: targetOperator,
            unitary: eigenvectors,
        )

        return offDiagNorm < diagonalityThreshold ? eigenvectors : nil
    }

    /// Compute U† M U for unitary similarity transformation.
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

    /// Frobenius norm of off-diagonal elements after unitary conjugation.
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

    /// Diagonalizes Hermitian matrix via LAPACK zheev.
    @_optimize(speed)
    @_eagerMove
    private func eigendecompose(_ matrix: [[Complex<Double>]]) -> (eigenvalues: [Double], eigenvectors: [[Complex<Double>]]) {
        let n: Int = matrix.count

        var a = [Double](unsafeUninitializedCapacity: 2 * n * n) {
            buffer, count in
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

        var w = [Double](unsafeUninitializedCapacity: n) {
            _, count in count = n
        }

        var jobz = CChar(Character("V").asciiValue!) // safe: ASCII literal
        var uplo = CChar(Character("U").asciiValue!) // safe: ASCII literal
        var nn = __LAPACK_int(n)
        var lda = __LAPACK_int(n)
        var lwork = __LAPACK_int(-1)
        var info = __LAPACK_int(0)

        var rwork = [Double](unsafeUninitializedCapacity: max(1, 3 * n - 2)) {
            _, count in count = max(1, 3 * n - 2)
        }

        var workQuery = [Double](unsafeUninitializedCapacity: 2) {
            _, count in count = 2
        }

        a.withUnsafeMutableBytes { aPtr in
            workQuery.withUnsafeMutableBytes { workPtr in
                w.withUnsafeMutableBufferPointer { wPtr in
                    rwork.withUnsafeMutableBufferPointer { rworkPtr in
                        zheev_(
                            &jobz, &uplo, &nn,
                            OpaquePointer(aPtr.baseAddress),
                            &lda,
                            wPtr.baseAddress,
                            OpaquePointer(workPtr.baseAddress)!, // safe: non-empty work buffer
                            &lwork,
                            rworkPtr.baseAddress,
                            &info,
                        )
                    }
                }
            }
        }

        let optimalWorkSize = Int(workQuery[0])

        lwork = __LAPACK_int(optimalWorkSize)
        var work = [Double](unsafeUninitializedCapacity: 2 * optimalWorkSize) {
            _, count in count = 2 * optimalWorkSize
        }

        a.withUnsafeMutableBytes { aPtr in
            work.withUnsafeMutableBytes { workPtr in
                w.withUnsafeMutableBufferPointer { wPtr in
                    rwork.withUnsafeMutableBufferPointer { rworkPtr in
                        zheev_(
                            &jobz, &uplo, &nn,
                            OpaquePointer(aPtr.baseAddress),
                            &lda,
                            wPtr.baseAddress,
                            OpaquePointer(workPtr.baseAddress)!, // safe: non-empty work buffer
                            &lwork,
                            rworkPtr.baseAddress,
                            &info,
                        )
                    }
                }
            }
        }

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
}

public extension PauliString {
    /// Computes P|row⟩ = phase * |col⟩ for Pauli string P acting on basis state |row⟩.
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
    /// - Complexity: O(k) where k is the number of Pauli operators in the string
    /// - SeeAlso: ``matrix(qubits:)``
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
                let ySign = 2.0 * Double(rowBit) - 1.0
                phase *= Complex(0.0, ySign)

            case .z:
                let zSign = 1.0 - 2.0 * Double(rowBit)
                phase *= Complex(zSign, 0.0)
            }
        }

        return (col, phase)
    }

    /// Dense 2^n * 2^n matrix representation via ``applyToRow(row:)``.
    ///
    /// **Example:**
    /// ```swift
    /// let matrix = PauliString([.x(0)]).matrix(qubits: 1)
    /// ```
    ///
    /// - Parameter qubits: Total qubits in system
    /// - Returns: Dense matrix with 2^n non-zero entries (one per row)
    /// - Complexity: O(2^(2n))
    /// - SeeAlso: ``applyToRow(row:)``
    @_optimize(speed)
    @_eagerMove
    func matrix(qubits: Int) -> [[Complex<Double>]] {
        let dimension = 1 << qubits
        var result = Array(repeating: Array(repeating: Complex<Double>.zero, count: dimension), count: dimension)

        for row in 0 ..< dimension {
            let (col, phase) = applyToRow(row: row)
            result[row][col] += phase
        }

        return result
    }
}
