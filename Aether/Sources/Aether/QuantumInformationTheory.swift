// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Result of Schmidt decomposition of a bipartite quantum state
///
/// Contains the Schmidt coefficients (singular values), left and right Schmidt basis vectors,
/// and derived entanglement properties. The original state can be reconstructed as
/// |psi> = sum_i lambda_i |a_i> tensor |b_i> where lambda_i are the coefficients,
/// |a_i> are the left basis vectors for subsystem A, and |b_i> are the right basis vectors
/// for subsystem B. The Schmidt rank indicates how many terms are needed in the decomposition,
/// with rank 1 indicating a product (unentangled) state.
///
/// **Example:**
/// ```swift
/// let bell = QuantumState(qubits: 2, amplitudes: [
///     Complex(1.0 / sqrt(2.0), 0), Complex(0, 0),
///     Complex(0, 0), Complex(1.0 / sqrt(2.0), 0)
/// ])
/// let result = QuantumInformationTheory.schmidtDecomposition(
///     state: bell, subsystemAQubits: [0]
/// )
/// let rank = result.schmidtRank
/// let entropy = result.entanglementEntropy
/// ```
@frozen public struct SchmidtDecompositionResult: Sendable {
    /// Schmidt coefficients (singular values) in descending order, all non-negative
    public let coefficients: [Double]

    /// Left Schmidt basis vectors |a_i> for subsystem A
    public let leftBasis: [[Complex<Double>]]

    /// Right Schmidt basis vectors |b_i> for subsystem B
    public let rightBasis: [[Complex<Double>]]

    /// Number of non-negligible Schmidt coefficients (above 1e-10 threshold)
    ///
    /// A Schmidt rank of 1 indicates a product state with no entanglement between subsystems.
    /// Higher ranks indicate entanglement, with the maximum rank being min(dimA, dimB).
    ///
    /// **Example:**
    /// ```swift
    /// let result = QuantumInformationTheory.schmidtDecomposition(
    ///     state: bellState, subsystemAQubits: [0]
    /// )
    /// let rank = result.schmidtRank
    /// ```
    @inlinable
    public var schmidtRank: Int {
        var count = 0
        for coefficient in coefficients {
            if coefficient > 1e-10 {
                count += 1
            }
        }
        return count
    }

    /// Von Neumann entanglement entropy S = -sum_i lambda_i^2 log2(lambda_i^2)
    ///
    /// Measures the degree of entanglement between subsystems A and B. Returns 0 for product
    /// states and log2(min(dimA, dimB)) for maximally entangled states. Computed from the
    /// Schmidt coefficient probability distribution.
    ///
    /// **Example:**
    /// ```swift
    /// let result = QuantumInformationTheory.schmidtDecomposition(
    ///     state: bellState, subsystemAQubits: [0]
    /// )
    /// let entropy = result.entanglementEntropy
    /// ```
    @inlinable
    public var entanglementEntropy: Double {
        QuantumInformationTheory.entropyFromProbabilities(coefficients.map { $0 * $0 })
    }
}

/// Quantum information theory computations for bipartite entanglement analysis
///
/// Provides Schmidt decomposition and entanglement entropy calculations for pure quantum
/// states partitioned into two subsystems. The Schmidt decomposition expresses any bipartite
/// pure state as |psi> = sum_i lambda_i |a_i> tensor |b_i>, revealing the entanglement
/// structure through the Schmidt coefficients lambda_i. The entanglement entropy
/// S = -sum_i lambda_i^2 log2(lambda_i^2) quantifies the degree of quantum correlations
/// between subsystems.
///
/// **Example:**
/// ```swift
/// let ghz = QuantumState(qubits: 3, amplitudes: [
///     Complex(1.0 / sqrt(2.0), 0), Complex(0, 0), Complex(0, 0), Complex(0, 0),
///     Complex(0, 0), Complex(0, 0), Complex(0, 0), Complex(1.0 / sqrt(2.0), 0)
/// ])
/// let entropy = QuantumInformationTheory.entanglementEntropy(
///     state: ghz, subsystemAQubits: [0]
/// )
/// ```
@frozen public enum QuantumInformationTheory {
    /// Perform Schmidt decomposition of a bipartite quantum state
    ///
    /// Decomposes the state vector into Schmidt form |psi> = sum_i lambda_i |a_i> tensor |b_i>
    /// by reshaping the amplitude vector into a matrix indexed by subsystem A and B basis states,
    /// then computing the singular value decomposition. The bit remapping extracts subsystem indices
    /// from the full basis state index using the specified qubit partitioning.
    ///
    /// - Parameters:
    ///   - state: Quantum state to decompose
    ///   - subsystemAQubits: Qubit indices defining subsystem A (complement forms subsystem B)
    /// - Returns: Schmidt decomposition result with coefficients, basis vectors, and entanglement measures
    /// - Precondition: subsystemAQubits must be a non-empty proper subset of valid qubit indices
    /// - Precondition: All qubit indices must be unique and within range
    /// - Complexity: O(min(dimA, dimB) * dimA * dimB) dominated by SVD computation
    ///
    /// **Example:**
    /// ```swift
    /// let bell = QuantumState(qubits: 2, amplitudes: [
    ///     Complex(1.0 / sqrt(2.0), 0), Complex(0, 0),
    ///     Complex(0, 0), Complex(1.0 / sqrt(2.0), 0)
    /// ])
    /// let result = QuantumInformationTheory.schmidtDecomposition(
    ///     state: bell, subsystemAQubits: [0]
    /// )
    /// ```
    @_optimize(speed)
    @_eagerMove
    public static func schmidtDecomposition(
        state: QuantumState,
        subsystemAQubits: [Int],
    ) -> SchmidtDecompositionResult {
        let aSet = Set(subsystemAQubits)
        let subsystemBQubits = (0 ..< state.qubits).filter { !aSet.contains($0) }

        ValidationUtilities.validateProperSubsystem(subsystemAQubits, totalQubits: state.qubits)
        ValidationUtilities.validateNonEmpty(subsystemAQubits, name: "subsystemAQubits")
        ValidationUtilities.validateUniqueQubits(subsystemAQubits)
        ValidationUtilities.validateOperationQubits(subsystemAQubits, numQubits: state.qubits)

        let dimA = 1 << subsystemAQubits.count
        let dimB = 1 << subsystemBQubits.count

        var matrix = [[Complex<Double>]](unsafeUninitializedCapacity: dimA) { buffer, count in
            for aIdx in 0 ..< dimA {
                buffer[aIdx] = [Complex<Double>](repeating: .zero, count: dimB)
            }
            count = dimA
        }

        for fullIndex in 0 ..< state.stateSpaceSize {
            var aIndex = 0
            for position in 0 ..< subsystemAQubits.count {
                let bit = BitUtilities.getBit(fullIndex, qubit: subsystemAQubits[position])
                aIndex |= (bit << position)
            }

            var bIndex = 0
            for position in 0 ..< subsystemBQubits.count {
                let bit = BitUtilities.getBit(fullIndex, qubit: subsystemBQubits[position])
                bIndex |= (bit << position)
            }

            matrix[aIndex][bIndex] = state.amplitudes[fullIndex]
        }

        let svdResult = SVDDecomposition.decompose(matrix: matrix)

        let schmidtCount = svdResult.singularValues.count

        let leftVectors = [[Complex<Double>]](unsafeUninitializedCapacity: schmidtCount) { buffer, count in
            for i in 0 ..< schmidtCount {
                buffer[i] = [Complex<Double>](unsafeUninitializedCapacity: dimA) { colBuffer, colCount in
                    for row in 0 ..< dimA {
                        colBuffer[row] = svdResult.u[row][i]
                    }
                    colCount = dimA
                }
            }
            count = schmidtCount
        }

        let rightVectors = [[Complex<Double>]](unsafeUninitializedCapacity: schmidtCount) { buffer, count in
            for i in 0 ..< schmidtCount {
                buffer[i] = [Complex<Double>](unsafeUninitializedCapacity: dimB) { colBuffer, colCount in
                    for col in 0 ..< dimB {
                        colBuffer[col] = svdResult.vDagger[i][col]
                    }
                    colCount = dimB
                }
            }
            count = schmidtCount
        }

        return SchmidtDecompositionResult(
            coefficients: svdResult.singularValues,
            leftBasis: leftVectors,
            rightBasis: rightVectors,
        )
    }

    /// Compute entanglement entropy of a quantum state for a given bipartition
    ///
    /// Convenience method that performs Schmidt decomposition and returns only the von Neumann
    /// entanglement entropy S = -sum_i lambda_i^2 log2(lambda_i^2). Values near 0 indicate
    /// a product state while log2(min(dimA, dimB)) indicates maximal entanglement.
    ///
    /// - Parameters:
    ///   - state: Quantum state to analyze
    ///   - subsystemAQubits: Qubit indices defining subsystem A
    /// - Returns: Entanglement entropy in bits (base-2 logarithm)
    /// - Precondition: subsystemAQubits must be a non-empty proper subset of valid qubit indices
    /// - Complexity: O(min(dimA, dimB) * dimA * dimB) dominated by SVD
    ///
    /// **Example:**
    /// ```swift
    /// let bell = QuantumState(qubits: 2, amplitudes: [
    ///     Complex(1.0 / sqrt(2.0), 0), Complex(0, 0),
    ///     Complex(0, 0), Complex(1.0 / sqrt(2.0), 0)
    /// ])
    /// let entropy = QuantumInformationTheory.entanglementEntropy(
    ///     state: bell, subsystemAQubits: [0]
    /// )
    /// ```
    @_optimize(speed)
    @_effects(readonly)
    public static func entanglementEntropy(
        state: QuantumState,
        subsystemAQubits: [Int],
    ) -> Double {
        let decomposition = schmidtDecomposition(state: state, subsystemAQubits: subsystemAQubits)
        var entropy = 0.0
        for lambda in decomposition.coefficients {
            let pSquared = lambda * lambda
            if pSquared < 1e-15 {
                continue
            }
            entropy -= pSquared * log2(pSquared)
        }
        return entropy
    }

    /// Compute Shannon entropy from a probability distribution using base-2 logarithm
    ///
    /// Calculates S = -sum_i p_i log2(p_i) with the convention that 0 log(0) = 0. Probabilities
    /// below 1e-15 are treated as zero to avoid numerical issues with logarithm of near-zero values.
    /// Used internally for entanglement entropy from Schmidt coefficient squares and externally
    /// by density matrix information theory computations.
    ///
    /// - Parameter probabilities: Array of probability values (should sum to 1)
    /// - Returns: Shannon entropy in bits (non-negative)
    /// - Complexity: O(n) where n is the number of probabilities
    ///
    /// **Example:**
    /// ```swift
    /// let uniform = [0.25, 0.25, 0.25, 0.25]
    /// let entropy = QuantumInformationTheory.entropyFromProbabilities(uniform)
    /// ```
    @_optimize(speed)
    @_effects(readonly)
    public static func entropyFromProbabilities(_ probabilities: [Double]) -> Double {
        var entropy = 0.0
        for p in probabilities {
            if p < 1e-15 {
                continue
            }
            entropy -= p * log2(p)
        }
        return entropy
    }
}
