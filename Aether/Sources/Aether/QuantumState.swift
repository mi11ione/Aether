// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Tolerance for normalization checks.
private let normalizationTolerance: Double = 1e-10
/// Minimum norm squared for safe normalization division.
private let normalizationMinimum: Double = 1e-15
/// Minimum amplitude magnitude squared for display.
private let displayThreshold: Double = 1e-6

/// Quantum state represented as complex amplitude vector |ψ⟩ = Σᵢ cᵢ|i⟩ in 2^n-dimensional Hilbert space.
///
/// Each computational basis state |i⟩ has complex amplitude cᵢ with measurement probability |cᵢ|² (Born rule).
/// Normalization constraint Σᵢ |cᵢ|² = 1 ensures probabilities sum to unity. Uses little-endian qubit ordering
/// where qubit 0 corresponds to the least significant bit in the basis state index.
///
/// Accelerate framework vectorization activates for states with 64+ amplitudes (6+ qubits), providing
/// hardware-accelerated probability calculations and normalization. Supports 1-30 qubits with 30 qubits
/// requiring approximately 8GB memory for the 2³⁰ complex amplitudes.
///
/// **Example:**
/// ```swift
/// // Ground state |00⟩
/// let ground = QuantumState(qubits: 2)
/// ground.probability(of: 0b00)  // 1.0
///
/// // Bell state (|00⟩ + |11⟩)/√2
/// let bell = QuantumState(qubits: 2, amplitudes: [
///     Complex(1/sqrt(2), 0), Complex(0, 0), Complex(0, 0), Complex(1/sqrt(2), 0)
/// ])
/// bell.probability(of: 0b00)  // 0.5
/// bell.probabilities(for: 0)  // (0.5, 0.5)
/// ```
///
/// - SeeAlso: ``QuantumCircuit`` for state evolution
/// - SeeAlso: ``Measurement`` for Born rule sampling
@frozen
public struct QuantumState: Equatable, CustomStringConvertible, Sendable {
    /// Array of complex amplitudes representing quantum state
    ///
    /// Contains 2^n complex coefficients for n-qubit system in computational basis ordering.
    /// Users can read amplitudes but not modify directly - use ``setAmplitude(_:to:)`` instead.
    ///
    /// **Example:**
    /// ```swift
    /// let state = QuantumState(qubits: 2)
    /// print(state.amplitudes.count)  // 4
    /// ```
    public private(set) var amplitudes: [Complex<Double>]

    /// Number of qubits in this quantum system
    ///
    /// **Example:**
    /// ```swift
    /// let state = QuantumState(qubits: 3)
    /// print(state.qubits)  // 3
    /// ```
    public let qubits: Int

    /// Size of state space (2^qubits)
    ///
    /// Convenience property for number of basis states. Equivalent to `1 << qubits`.
    ///
    /// **Example:**
    /// ```swift
    /// let state = QuantumState(qubits: 3)
    /// print(state.stateSpaceSize)  // 8
    /// ```
    ///
    /// - Complexity: O(1)
    public var stateSpaceSize: Int {
        1 << qubits
    }

    /// Initialize ground state: all qubits in |0⟩
    ///
    /// Creates computational basis state |00...0⟩ with amplitude 1.0 for state 0 and all other
    /// amplitudes zero. This is the default starting state for quantum circuits and algorithms.
    ///
    /// **Example:**
    /// ```swift
    /// let state = QuantumState(qubits: 2)  // |00⟩
    /// ```
    ///
    /// - Parameter qubits: Number of qubits (supports 1-30)
    /// - Complexity: O(2^n)
    /// - Precondition: 1 ≤ qubits ≤ 30
    public init(qubits: Int) {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateMemoryLimit(qubits)

        self.qubits = qubits
        let size = 1 << qubits

        amplitudes = [Complex<Double>](unsafeUninitializedCapacity: size) { buffer, count in
            buffer.initialize(repeating: .zero)
            buffer[0] = .one
            count = size
        }
    }

    /// Initialize quantum state from normalized amplitude vector.
    ///
    /// Creates arbitrary superposition state from provided complex amplitudes. Amplitudes must be
    /// approximately normalized (Σ|cᵢ|² ≈ 1 within 1e-6). Minor floating-point drift from
    /// unitary operations is corrected; grossly unnormalized vectors are rejected.
    ///
    /// **Example:**
    /// ```swift
    /// let plus = QuantumState(qubits: 1, amplitudes: [
    ///     Complex(1/sqrt(2), 0), Complex(1/sqrt(2), 0)  // (|0⟩ + |1⟩)/√2
    /// ])
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits
    ///   - amplitudes: Array of 2^n complex amplitudes (must be approximately normalized)
    /// - Complexity: O(2^n)
    /// - Precondition: amplitudes.count == 2^qubits
    /// - Precondition: Σ|cᵢ|² ≈ 1.0 (within 1e-6)
    /// - SeeAlso: ``isNormalized()``
    public init(qubits: Int, amplitudes: [Complex<Double>]) {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateAmplitudeCount(amplitudes, qubits: qubits)

        self.qubits = qubits
        self.amplitudes = amplitudes

        let sumSquared = computeNormSquared()
        ValidationUtilities.validateApproximateNormalization(sumSquared)

        if abs(sumSquared - 1.0) > normalizationTolerance {
            let invNorm = 1.0 / sqrt(sumSquared)
            self.amplitudes = [Complex<Double>](unsafeUninitializedCapacity: amplitudes.count) { buffer, count in
                for i in amplitudes.indices {
                    buffer[i] = amplitudes[i] * invNorm
                }
                count = amplitudes.count
            }
        }
    }

    /// Initialize quantum state from potentially unnormalized amplitude vector.
    ///
    /// Internal initializer for code paths that produce amplitudes with arbitrary norm
    /// (post-selection, representation conversion, trajectory evolution). Automatically
    /// normalizes any positive-norm vector. Rejects zero-norm vectors.
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits
    ///   - rawAmplitudes: Array of 2^n complex amplitudes (auto-normalizes)
    /// - Complexity: O(2^n)
    /// - Precondition: rawAmplitudes.count == 2^qubits
    /// - Precondition: Σ|cᵢ|² > 0 (non-zero norm)
    @usableFromInline
    init(qubits: Int, rawAmplitudes: [Complex<Double>]) {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateAmplitudeCount(rawAmplitudes, qubits: qubits)

        self.qubits = qubits
        amplitudes = rawAmplitudes

        let sumSquared = computeNormSquared()
        ValidationUtilities.validateNonZeroNorm(sumSquared)

        if abs(sumSquared - 1.0) > normalizationTolerance {
            let invNorm = 1.0 / sqrt(sumSquared)
            amplitudes = [Complex<Double>](unsafeUninitializedCapacity: rawAmplitudes.count) { buffer, count in
                for i in rawAmplitudes.indices {
                    buffer[i] = rawAmplitudes[i] * invNorm
                }
                count = rawAmplitudes.count
            }
        }
    }

    /// Initialize single-qubit computational basis state
    ///
    /// Creates |0⟩ or |1⟩ state for single qubit. Convenience initializer for the most common
    /// quantum states.
    ///
    /// **Example:**
    /// ```swift
    /// let zero = QuantumState(qubit: 0)  // |0⟩
    /// let one = QuantumState(qubit: 1)   // |1⟩
    /// ```
    ///
    /// - Parameter state: Either 0 (for |0⟩) or 1 (for |1⟩)
    /// - Complexity: O(1)
    /// - Precondition: state must be 0 or 1
    public init(qubit state: Int) {
        ValidationUtilities.validateBinaryValue(state, name: "Qubit state")

        qubits = 1
        if state == 0 { amplitudes = [.one, .zero] } else { amplitudes = [.zero, .one] }
    }

    /// Computes magnitude squared for all amplitudes using vDSP vectorization.
    @_effects(readonly)
    @_eagerMove
    private func computeMagnitudesSquaredVectorized() -> [Double] {
        let n: Int = amplitudes.count
        let interleavedAmps = [Double](unsafeUninitializedCapacity: n * 2) { buffer, count in
            for i in 0 ..< n {
                buffer[i * 2] = amplitudes[i].real
                buffer[i * 2 + 1] = amplitudes[i].imaginary
            }
            count = n * 2
        }

        return [Double](unsafeUninitializedCapacity: n) { magPtr, count in
            interleavedAmps.withUnsafeBufferPointer { interleavedPtr in
                // safety: interleavedAmps is non-empty (n >= 1 for valid quantum states)
                var splitComplex = DSPDoubleSplitComplex(
                    realp: UnsafeMutablePointer(mutating: interleavedPtr.baseAddress!),
                    imagp: UnsafeMutablePointer(mutating: interleavedPtr.baseAddress! + 1),
                )
                // safety: magPtr from unsafeUninitializedCapacity is always valid for count n
                vDSP_zvmagsD(&splitComplex, 2, magPtr.baseAddress!, 1, vDSP_Length(n))
            }
            count = n
        }
    }

    /// Calculate probability of measuring specific basis state (Born rule)
    ///
    /// Computes P(|i⟩) = |cᵢ|² where cᵢ is the amplitude of basis state |i⟩.
    /// This implements the Born rule from quantum mechanics: probability equals
    /// magnitude squared of the amplitude.
    ///
    /// **Example:**
    /// ```swift
    /// let bell = QuantumState(qubits: 2, amplitudes: [
    ///     Complex(1/sqrt(2), 0), Complex(0, 0), Complex(0, 0), Complex(1/sqrt(2), 0)
    /// ])
    /// bell.probability(of: 0b00)  // 0.5
    /// bell.probability(of: 0b11)  // 0.5
    /// ```
    ///
    /// - Parameter stateIndex: Index of basis state (0 to 2^n-1), little-endian qubit ordering
    /// - Returns: Probability P(i) = |cᵢ|² ∈ [0, 1]
    /// - Complexity: O(1)
    /// - Precondition: stateIndex must be in range [0, 2^n-1]
    /// - SeeAlso: ``probabilities()`` for full distribution
    @_effects(readonly)
    @inlinable
    public func probability(of stateIndex: Int) -> Double {
        ValidationUtilities.validateIndexInBounds(stateIndex, bound: stateSpaceSize, name: "Basis state index")
        return amplitudes[stateIndex].magnitudeSquared
    }

    /// Calculate full probability distribution over all basis states
    ///
    /// Returns complete probability vector [P(0), P(1), ..., P(2^n-1)] where P(i) = |cᵢ|².
    /// Automatically uses vectorized Accelerate framework for states with 64+ basis states.
    ///
    /// **Example:**
    /// ```swift
    /// let uniform = QuantumState(qubits: 2, amplitudes: [
    ///     Complex(0.5, 0), Complex(0.5, 0), Complex(0.5, 0), Complex(0.5, 0)
    /// ])
    /// uniform.probabilities()  // [0.25, 0.25, 0.25, 0.25]
    /// ```
    ///
    /// - Returns: Array of 2^n probabilities summing to 1.0
    /// - Complexity: O(2^n) with vectorization for large states
    /// - SeeAlso: ``probability(of:)`` for single state probability
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public func probabilities() -> [Double] {
        let n: Int = amplitudes.count
        if n >= 64 {
            return computeMagnitudesSquaredVectorized()
        } else {
            return [Double](unsafeUninitializedCapacity: n) { buffer, count in
                for i in 0 ..< n {
                    buffer[i] = amplitudes[i].magnitudeSquared
                }
                count = n
            }
        }
    }

    /// Find basis state with highest probability
    ///
    /// Computes maximum probability without allocating full probability array. Returns both the
    /// index and probability of the most likely measurement outcome.
    ///
    /// **Example:**
    /// ```swift
    /// let ghz = QuantumState(qubits: 3, amplitudes: [
    ///     Complex(1/sqrt(2), 0), Complex(0, 0), Complex(0, 0), Complex(0, 0),
    ///     Complex(0, 0), Complex(0, 0), Complex(0, 0), Complex(1/sqrt(2), 0)
    /// ])
    /// let (index, prob) = ghz.mostProbableState()  // (0 or 7, 0.5)
    /// ```
    ///
    /// - Returns: Tuple of (basisStateIndex, probability) for most probable state
    /// - Complexity: O(2^n)
    /// - SeeAlso: ``probabilities()`` for full distribution
    @_optimize(speed)
    @_effects(readonly)
    public func mostProbableState() -> (index: Int, probability: Double) {
        if stateSpaceSize >= 64 {
            let mags = computeMagnitudesSquaredVectorized()
            var maxValue = 0.0
            var maxIndex: vDSP_Length = 0
            vDSP_maxviD(mags, 1, &maxValue, &maxIndex, vDSP_Length(mags.count))
            return (Int(maxIndex), maxValue)
        }
        var maxIndex = 0
        var maxProb = amplitudes[0].magnitudeSquared

        for i in 1 ..< stateSpaceSize {
            let prob = amplitudes[i].magnitudeSquared
            if prob > maxProb {
                maxProb = prob
                maxIndex = i
            }
        }

        return (maxIndex, maxProb)
    }

    /// Calculate marginal probability distribution for single qubit
    ///
    /// Computes P(|0⟩) and P(|1⟩) for a specific qubit by summing over all basis states where
    /// that qubit has the desired value. Implements partial trace / marginalization over other qubits.
    ///
    /// **Example:**
    /// ```swift
    /// let bell = QuantumState(qubits: 2, amplitudes: [
    ///     Complex(1/sqrt(2), 0), Complex(0, 0), Complex(0, 0), Complex(1/sqrt(2), 0)
    /// ])
    /// let (p0, p1) = bell.probabilities(for: 0)  // (0.5, 0.5)
    /// ```
    ///
    /// - Parameter qubit: Qubit index to measure (0 to n-1)
    /// - Returns: Tuple (P(|0⟩), P(|1⟩)) for specified qubit
    /// - Complexity: O(2^n)
    /// - Precondition: qubit must be in range [0, n-1]
    /// - Note: P(qubit=0) = Σᵢ |cᵢ|² where bit_i(qubit) = 0
    /// - SeeAlso: ``probability(of:)`` for full basis state probabilities
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    public func probabilities(for qubit: Int) -> (p0: Double, p1: Double) {
        ValidationUtilities.validateIndexInBounds(qubit, bound: qubits, name: "Qubit index")

        var p0 = 0.0
        var p1 = 0.0

        let blockSize = 1 << qubit
        let doubleBlock: Int = blockSize << 1
        var baseIndex = 0

        while baseIndex < stateSpaceSize {
            for i in baseIndex ..< (baseIndex + blockSize) {
                p0 += amplitudes[i].magnitudeSquared
            }
            for i in (baseIndex + blockSize) ..< (baseIndex + doubleBlock) {
                p1 += amplitudes[i].magnitudeSquared
            }
            baseIndex += doubleBlock
        }

        return (p0, p1)
    }

    /// Computes sum of squared magnitudes for normalization checking.
    @_optimize(speed)
    @_effects(readonly)
    private func computeNormSquared() -> Double {
        if amplitudes.count >= 64 {
            let magnitudesSquared: [Double] = computeMagnitudesSquaredVectorized()
            var sum = 0.0
            magnitudesSquared.withUnsafeBufferPointer { ptr in
                // safety: magnitudesSquared is non-empty (amplitudes.count >= 64)
                vDSP_sveD(ptr.baseAddress!, 1, &sum, vDSP_Length(magnitudesSquared.count))
            }
            return sum
        } else {
            return amplitudes.reduce(0.0) { $0 + $1.magnitudeSquared }
        }
    }

    /// Check if quantum state satisfies normalization constraint
    ///
    /// Verifies that Σᵢ |cᵢ|² ≈ 1.0 within numerical tolerance (1e-10). All valid quantum states
    /// must be normalized for probabilities to sum to 1.
    ///
    /// **Example:**
    /// ```swift
    /// let state = QuantumState(qubits: 2)
    /// state.isNormalized()  // true
    /// ```
    ///
    /// - Returns: True if normalized within tolerance
    /// - Complexity: O(2^n) with vectorization for large states
    /// - SeeAlso: ``normalize()`` to enforce normalization
    @_effects(readonly)
    public func isNormalized() -> Bool {
        let sum: Double = computeNormSquared()
        return abs(sum - 1.0) < normalizationTolerance
    }

    /// Normalize quantum state to satisfy Σ|cᵢ|² = 1
    ///
    /// Divides all amplitudes by √(Σᵢ |cᵢ|²) to ensure normalization constraint.
    /// Required after operations that may denormalize the state.
    ///
    /// **Example:**
    /// ```swift
    /// var state = QuantumState(qubits: 2)
    /// state.normalize()  // Restores exact Σ|cᵢ|² = 1 after floating-point drift
    /// ```
    ///
    /// - Complexity: O(2^n) with vectorization for large states
    /// - Precondition: State norm must be positive (non-zero state)
    /// - SeeAlso: ``isNormalized()`` to check normalization status
    public mutating func normalize() {
        let sumSquared: Double = computeNormSquared()
        ValidationUtilities.validatePositiveDouble(sumSquared, name: "State norm squared")

        let invNorm = 1.0 / sqrt(sumSquared)
        let n: Int = amplitudes.count
        amplitudes = [Complex<Double>](unsafeUninitializedCapacity: n) { buffer, count in
            for i in 0 ..< n {
                buffer[i] = amplitudes[i] * invNorm
            }
            count = n
        }
    }

    /// Get complex amplitude of specific basis state
    ///
    /// Returns the coefficient cᵢ for basis state |i⟩. Use `probability(of:)` for Born rule
    /// probabilities. Direct amplitude access is useful for quantum algorithm analysis and debugging.
    ///
    /// **Example:**
    /// ```swift
    /// let plus = QuantumState(qubits: 1, amplitudes: [
    ///     Complex(1/sqrt(2), 0), Complex(1/sqrt(2), 0)
    /// ])
    /// plus.amplitude(of: 0)  // (1/√2, 0)
    /// ```
    ///
    /// - Parameter stateIndex: Index of basis state (0 to 2^n-1)
    /// - Returns: Complex amplitude cᵢ
    /// - Complexity: O(1)
    /// - Precondition: stateIndex must be in range [0, 2^n-1]
    /// - SeeAlso: ``setAmplitude(_:to:)`` for mutating amplitudes
    @_effects(readonly)
    @inlinable
    public func amplitude(of stateIndex: Int) -> Complex<Double> {
        ValidationUtilities.validateIndexInBounds(stateIndex, bound: stateSpaceSize, name: "Basis state index")
        return amplitudes[stateIndex]
    }

    /// Set complex amplitude of specific basis state (mutating)
    ///
    /// Directly modifies coefficient cᵢ for basis state |i⟩. May denormalize the state - call
    /// `normalize()` after if needed. Primarily used internally by gate application and for testing.
    ///
    /// **Example:**
    /// ```swift
    /// var state = QuantumState(qubits: 1)
    /// state.setAmplitude(0, to: Complex(1/sqrt(2), 0))
    /// state.setAmplitude(1, to: Complex(1/sqrt(2), 0))
    /// ```
    ///
    /// - Parameters:
    ///   - stateIndex: Index of basis state (0 to 2^n-1)
    ///   - amplitude: New complex amplitude cᵢ
    /// - Complexity: O(1)
    /// - Precondition: stateIndex must be in range [0, 2^n-1]
    /// - Note: May denormalize state - call ``normalize()`` after bulk modifications
    /// - SeeAlso: ``amplitude(of:)`` for reading amplitudes
    @inline(__always)
    public mutating func setAmplitude(_ stateIndex: Int, to amplitude: Complex<Double>) {
        ValidationUtilities.validateIndexInBounds(stateIndex, bound: stateSpaceSize, name: "Basis state index")
        amplitudes[stateIndex] = amplitude
    }

    /// String representation showing significant amplitudes
    ///
    /// Generates human-readable quantum state notation showing only basis states with probability
    /// above threshold (1e-6). Format: "QuantumState(n qubits): amplitude₁|basis₁⟩ + amplitude₂|basis₂⟩ + ..."
    ///
    /// **Example:**
    /// ```swift
    /// let bell = QuantumState(qubits: 2, amplitudes: [
    ///     Complex(1/sqrt(2), 0), Complex(0, 0), Complex(0, 0), Complex(1/sqrt(2), 0)
    /// ])
    /// print(bell)  // "QuantumState(2 qubits): 0.7071|00⟩ + 0.7071|11⟩"
    /// ```
    ///
    /// - Complexity: O(2^n)
    /// - Note: Filters amplitudes below 1e-6 threshold for readability
    public var description: String {
        let threshold = displayThreshold
        var terms: [String] = []
        terms.reserveCapacity(min(amplitudes.count, 16))

        for i in 0 ..< amplitudes.count {
            let magSq = amplitudes[i].magnitudeSquared
            if magSq > threshold {
                let ampStr = String(format: "%.4f", sqrt(magSq))
                let binaryStr = String(i, radix: 2)
                let paddedBinary = String(repeating: "0", count: max(0, qubits - binaryStr.count)) + binaryStr
                terms.append("\(ampStr)|\(paddedBinary)⟩")
            }
        }

        return terms.isEmpty
            ? "QuantumState(\(qubits) qubits, near-zero)"
            : "QuantumState(\(qubits) qubits): " + terms.joined(separator: " + ")
    }
}
