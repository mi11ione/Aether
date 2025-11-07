// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate
import Foundation

/// Errors that can occur during quantum state operations
enum QuantumStateError: Error, LocalizedError {
    case cannotNormalizeZeroState
    case invalidAmplitudes

    var errorDescription: String? {
        switch self {
        case .cannotNormalizeZeroState:
            "Cannot normalize quantum state with near-zero norm"
        case .invalidAmplitudes:
            "Quantum state contains invalid amplitudes (NaN or Inf)"
        }
    }
}

/// Quantum state: complex amplitude vector for n-qubit system
///
/// Represents quantum superposition as a statevector in 2^n-dimensional Hilbert space.
/// Each computational basis state |i⟩ has a complex amplitude cᵢ, with probabilities
/// given by |cᵢ|² (Born rule). Automatically handles normalization and validation.
///
/// **Mathematical representation**:
/// - State vector: |ψ⟩ = Σᵢ cᵢ|i⟩ where cᵢ ∈ ℂ, i ∈ [0, 2^n-1]
/// - Normalization constraint: Σᵢ |cᵢ|² = 1 (total probability = 1)
/// - Qubit ordering: Little-endian (qubit 0 is LSB in binary index)
/// - Hilbert space dimension: 2^n for n qubits
///
/// **Architecture**:
/// - Generic over qubit count (supports 1-30 qubits)
/// - Accelerate framework vectorization for large states (64+ basis states)
/// - Thread-safe (immutable operations, Sendable conformance)
/// - Auto-normalization in initializers with tolerance checking
///
/// **Usage patterns**:
/// - Ground state: Default initialization creates |00...0⟩
/// - Custom states: Provide amplitude vector (auto-normalizes if needed)
/// - Single qubits: Specialized initializer for |0⟩ or |1⟩
/// - Measurements: Born rule probabilities via `probability(ofState:)`
///
/// Example:
/// ```swift
/// // Ground state: |00⟩ = [1, 0, 0, 0]
/// let groundState = QuantumState(numQubits: 2)
/// let p00 = groundState.probability(ofState: 0b00)  // 1.0
///
/// // Bell state: (|00⟩ + |11⟩)/√2
/// let bellState = QuantumState(numQubits: 2, amplitudes: [
///     Complex(1/sqrt(2), 0),  // |00⟩
///     Complex(0, 0),          // |01⟩
///     Complex(0, 0),          // |10⟩
///     Complex(1/sqrt(2), 0)   // |11⟩
/// ])
/// let p00 = bellState.probability(ofState: 0b00)  // 0.5
/// let p11 = bellState.probability(ofState: 0b11)  // 0.5
///
/// // GHZ state: (|000⟩ + |111⟩)/√2
/// let ghz = QuantumState(numQubits: 3, amplitudes: [
///     Complex(1/sqrt(2), 0),  // |000⟩
///     Complex(0, 0),          // |001⟩
///     Complex(0, 0),          // |010⟩
///     Complex(0, 0),          // |011⟩
///     Complex(0, 0),          // |100⟩
///     Complex(0, 0),          // |101⟩
///     Complex(0, 0),          // |110⟩
///     Complex(1/sqrt(2), 0)   // |111⟩
/// ])
///
/// // Marginal qubit probabilities
/// let (p0, p1) = bellState.singleQubitProbabilities(qubit: 0)  // (0.5, 0.5)
///
/// // Full probability distribution
/// let allProbs = bellState.probabilities()  // [0.5, 0, 0, 0.5]
/// ```
struct QuantumState: Equatable, CustomStringConvertible, Sendable {
    // MARK: - Properties

    /// Array of complex amplitudes representing quantum state
    /// Length = 2^numQubits
    private(set) var amplitudes: [Complex<Double>]

    /// Number of qubits in this quantum system
    let numQubits: Int

    /// Size of state space (2^numQubits)
    var stateSpaceSize: Int { 1 << numQubits }

    // MARK: - Initialization

    /// Initialize ground state: all qubits in |0⟩
    ///
    /// Creates computational basis state |00...0⟩ with amplitude 1.0 for state 0
    /// and all other amplitudes zero. This is the default starting state for
    /// quantum circuits and algorithms.
    ///
    /// - Parameter numQubits: Number of qubits (supports 1-30)
    ///
    /// Example:
    /// ```swift
    /// let state1 = QuantumState(numQubits: 1)
    /// // |0⟩: amplitudes = [1, 0]
    ///
    /// let state2 = QuantumState(numQubits: 2)
    /// // |00⟩: amplitudes = [1, 0, 0, 0]
    ///
    /// let state3 = QuantumState(numQubits: 3)
    /// // |000⟩: amplitudes = [1, 0, 0, 0, 0, 0, 0, 0]
    /// ```
    init(numQubits: Int) {
        precondition(numQubits > 0, "Number of qubits must be positive")
        precondition(numQubits < 30, "Number of qubits too large (would exceed memory)")

        self.numQubits = numQubits
        let size = 1 << numQubits

        var amps = [Complex<Double>](repeating: .zero, count: size)
        amps[0] = .one
        amplitudes = amps
    }

    /// Initialize custom quantum state from amplitude vector
    ///
    /// Creates arbitrary superposition state from provided complex amplitudes.
    /// Automatically normalizes if amplitude vector is not already normalized
    /// (within tolerance). Useful for preparing specific quantum states.
    ///
    /// **Normalization**: If Σ|cᵢ|² ≠ 1, divides all amplitudes by √(Σ|cᵢ|²)
    ///
    /// - Parameters:
    ///   - numQubits: Number of qubits
    ///   - amplitudes: Array of 2^n complex amplitudes (auto-normalizes if needed)
    ///
    /// Example:
    /// ```swift
    /// // |+⟩ state: (|0⟩ + |1⟩)/√2
    /// let plus = QuantumState(numQubits: 1, amplitudes: [
    ///     Complex(1/sqrt(2), 0),
    ///     Complex(1/sqrt(2), 0)
    /// ])
    ///
    /// // |−⟩ state: (|0⟩ − |1⟩)/√2
    /// let minus = QuantumState(numQubits: 1, amplitudes: [
    ///     Complex(1/sqrt(2), 0),
    ///     Complex(-1/sqrt(2), 0)
    /// ])
    ///
    /// // W state: (|001⟩ + |010⟩ + |100⟩)/√3
    /// let w = QuantumState(numQubits: 3, amplitudes: [
    ///     Complex(0, 0),           // |000⟩
    ///     Complex(1/sqrt(3), 0),   // |001⟩
    ///     Complex(1/sqrt(3), 0),   // |010⟩
    ///     Complex(0, 0),           // |011⟩
    ///     Complex(1/sqrt(3), 0),   // |100⟩
    ///     Complex(0, 0),           // |101⟩
    ///     Complex(0, 0),           // |110⟩
    ///     Complex(0, 0)            // |111⟩
    /// ])
    ///
    /// // Auto-normalization: unnormalized input
    /// let state = QuantumState(numQubits: 1, amplitudes: [
    ///     Complex(1, 0),  // Will normalize to 1/√2
    ///     Complex(1, 0)   // Will normalize to 1/√2
    /// ])
    /// ```
    init(numQubits: Int, amplitudes: [Complex<Double>]) {
        precondition(numQubits > 0, "Number of qubits must be positive")
        precondition(amplitudes.count == (1 << numQubits),
                     "Amplitude array size must equal 2^numQubits")

        self.numQubits = numQubits
        self.amplitudes = amplitudes

        // Auto-normalize if needed (within reasonable tolerance)
        let sumSquared = amplitudes.reduce(0.0) { $0 + $1.magnitudeSquared }
        if abs(sumSquared - 1.0) > 1e-10,
           sumSquared > 1e-15
        {
            let norm = sqrt(sumSquared)
            self.amplitudes = amplitudes.map { $0 / norm }
        }
    }

    /// Initialize single-qubit computational basis state
    ///
    /// Creates |0⟩ or |1⟩ state for single qubit. Convenience initializer for
    /// the most common quantum states.
    ///
    /// - Parameter state: Either 0 (for |0⟩) or 1 (for |1⟩)
    ///
    /// Example:
    /// ```swift
    /// let zero = QuantumState(singleQubit: 0)
    /// // |0⟩: amplitudes = [1, 0]
    ///
    /// let one = QuantumState(singleQubit: 1)
    /// // |1⟩: amplitudes = [0, 1]
    /// ```
    init(singleQubit state: Int) {
        precondition(state == 0 || state == 1, "Single qubit state must be 0 or 1")

        numQubits = 1
        var amps = [Complex<Double>](repeating: .zero, count: 2)
        amps[state] = .one
        amplitudes = amps
    }

    /// Internal test-only initializer that bypasses validation
    /// Used to test validate() method with intentionally invalid states
    /// - Parameters:
    ///   - numQubits: Number of qubits
    ///   - amplitudes: Amplitudes array (can be wrong size)
    ///   - bypassValidation: Must be true to use this initializer
    init(numQubits: Int, amplitudes: [Complex<Double>], bypassValidation: Bool) {
        precondition(bypassValidation, "This initializer is for testing only")
        self.numQubits = numQubits
        self.amplitudes = amplitudes
    }

    /// Private helper: Compute magnitude squared for all amplitudes using Accelerate
    /// - Returns: Array of amplitude values
    private func computeMagnitudesSquaredVectorized() -> [Double] {
        var interleavedAmps = [Double]()
        interleavedAmps.reserveCapacity(amplitudes.count * 2)
        for amp in amplitudes {
            interleavedAmps.append(amp.real)
            interleavedAmps.append(amp.imaginary)
        }

        var magnitudesSquared = [Double](repeating: 0.0, count: amplitudes.count)
        interleavedAmps.withUnsafeBufferPointer { interleavedPtr in
            magnitudesSquared.withUnsafeMutableBufferPointer { magPtr in
                guard let interleavedBase = interleavedPtr.baseAddress,
                      let magBase = magPtr.baseAddress else { return }

                var splitComplex = DSPDoubleSplitComplex(
                    realp: UnsafeMutablePointer(mutating: interleavedBase),
                    imagp: UnsafeMutablePointer(mutating: interleavedBase + 1)
                )

                vDSP_zvmagsD(&splitComplex, 2, magBase, 1, vDSP_Length(amplitudes.count))
            }
        }

        return magnitudesSquared
    }

    /// Calculate probability of measuring specific basis state (Born rule)
    ///
    /// Computes P(|i⟩) = |cᵢ|² where cᵢ is the amplitude of basis state |i⟩.
    /// This implements the Born rule from quantum mechanics: probability equals
    /// magnitude squared of the amplitude.
    ///
    /// **Qubit indexing**: Uses little-endian (qubit 0 is LSB in state index)
    ///
    /// - Parameter stateIndex: Index of basis state (0 to 2^n-1)
    /// - Returns: Probability P(i) = |cᵢ|² ∈ [0, 1]
    ///
    /// Example:
    /// ```swift
    /// // Bell state: (|00⟩ + |11⟩)/√2
    /// let bell = QuantumState(numQubits: 2, amplitudes: [
    ///     Complex(1/sqrt(2), 0),  // |00⟩
    ///     Complex(0, 0),          // |01⟩
    ///     Complex(0, 0),          // |10⟩
    ///     Complex(1/sqrt(2), 0)   // |11⟩
    /// ])
    ///
    /// let p00 = bell.probability(ofState: 0b00)  // 0.5
    /// let p01 = bell.probability(ofState: 0b01)  // 0.0
    /// let p10 = bell.probability(ofState: 0b10)  // 0.0
    /// let p11 = bell.probability(ofState: 0b11)  // 0.5
    /// ```
    func probability(ofState stateIndex: Int) -> Double {
        precondition(stateIndex >= 0 && stateIndex < stateSpaceSize,
                     "State index out of bounds")
        return amplitudes[stateIndex].magnitudeSquared
    }

    /// Calculate full probability distribution over all basis states
    ///
    /// Returns complete probability vector [P(0), P(1), ..., P(2^n-1)] where
    /// P(i) = |cᵢ|². Automatically uses vectorized Accelerate framework for
    /// large states (64+ basis states) for optimal performance.
    ///
    /// - Returns: Array of 2^n probabilities summing to 1.0
    ///
    /// Example:
    /// ```swift
    /// // Uniform superposition: (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
    /// let uniform = QuantumState(numQubits: 2, amplitudes: [
    ///     Complex(0.5, 0),
    ///     Complex(0.5, 0),
    ///     Complex(0.5, 0),
    ///     Complex(0.5, 0)
    /// ])
    /// let probs = uniform.probabilities()
    /// // [0.25, 0.25, 0.25, 0.25]
    ///
    /// // GHZ state: (|000⟩ + |111⟩)/√2
    /// let ghz = QuantumState(numQubits: 3, amplitudes: [
    ///     Complex(1/sqrt(2), 0),  // |000⟩
    ///     Complex(0, 0),          // |001⟩
    ///     Complex(0, 0),          // |010⟩
    ///     Complex(0, 0),          // |011⟩
    ///     Complex(0, 0),          // |100⟩
    ///     Complex(0, 0),          // |101⟩
    ///     Complex(0, 0),          // |110⟩
    ///     Complex(1/sqrt(2), 0)   // |111⟩
    /// ])
    /// let ghzProbs = ghz.probabilities()
    /// // [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
    /// ```
    func probabilities() -> [Double] {
        if amplitudes.count >= 64 {
            computeMagnitudesSquaredVectorized()
        } else {
            amplitudes.map(\.magnitudeSquared)
        }
    }

    /// Calculate marginal probability distribution for single qubit
    ///
    /// Computes the probability of measuring |0⟩ or |1⟩ for a specific qubit
    /// by summing over all basis states where that qubit has the desired value.
    /// Implements partial trace / marginalization over other qubits.
    ///
    /// **Math**: P(qubit=0) = Σᵢ |cᵢ|² where bit_i(qubit) = 0
    ///
    /// - Parameter qubit: Qubit index to measure (0 to n-1)
    /// - Returns: Tuple (P(|0⟩), P(|1⟩)) for specified qubit
    ///
    /// Example:
    /// ```swift
    /// // Bell state: (|00⟩ + |11⟩)/√2
    /// let bell = QuantumState(numQubits: 2, amplitudes: [
    ///     Complex(1/sqrt(2), 0),  // |00⟩
    ///     Complex(0, 0),          // |01⟩
    ///     Complex(0, 0),          // |10⟩
    ///     Complex(1/sqrt(2), 0)   // |11⟩
    /// ])
    ///
    /// let (p0_q0, p1_q0) = bell.singleQubitProbabilities(qubit: 0)
    /// // p0_q0 = 0.5 (from |00⟩), p1_q0 = 0.5 (from |11⟩)
    ///
    /// let (p0_q1, p1_q1) = bell.singleQubitProbabilities(qubit: 1)
    /// // p0_q1 = 0.5 (from |00⟩), p1_q1 = 0.5 (from |11⟩)
    ///
    /// // Product state: |01⟩ (qubit 0 = |1⟩, qubit 1 = |0⟩)
    /// let product = QuantumState(numQubits: 2, amplitudes: [
    ///     Complex(0, 0),  // |00⟩
    ///     Complex(1, 0),  // |01⟩
    ///     Complex(0, 0),  // |10⟩
    ///     Complex(0, 0)   // |11⟩
    /// ])
    /// let (p0, p1) = product.singleQubitProbabilities(qubit: 0)
    /// // p0 = 0.0, p1 = 1.0
    /// ```
    func singleQubitProbabilities(qubit: Int) -> (p0: Double, p1: Double) {
        precondition(qubit >= 0 && qubit < numQubits, "Qubit index out of bounds")

        var p0 = 0.0
        var p1 = 0.0

        for i in 0 ..< stateSpaceSize {
            let prob = amplitudes[i].magnitudeSquared
            if getBit(index: i, qubit: qubit) == 0 {
                p0 += prob
            } else {
                p1 += prob
            }
        }

        return (p0, p1)
    }

    // MARK: - Normalization

    /// Check if quantum state satisfies normalization constraint
    ///
    /// Verifies that Σᵢ |cᵢ|² ≈ 1.0 within numerical tolerance (1e-10).
    /// All valid quantum states must be normalized for probabilities to sum to 1.
    /// Uses vectorized computation for large states.
    ///
    /// - Returns: True if normalized within tolerance
    ///
    /// Example:
    /// ```swift
    /// let state = QuantumState(numQubits: 2)
    /// state.isNormalized()  // true
    ///
    /// var custom = QuantumState(numQubits: 1, amplitudes: [
    ///     Complex(1/sqrt(2), 0),
    ///     Complex(1/sqrt(2), 0)
    /// ])
    /// custom.isNormalized()  // true (auto-normalized in init)
    /// ```
    func isNormalized() -> Bool {
        let sum: Double

        if amplitudes.count >= 64 {
            let magnitudesSquared = computeMagnitudesSquaredVectorized()
            sum = magnitudesSquared.reduce(0.0, +)
        } else {
            sum = amplitudes.reduce(0.0) { $0 + $1.magnitudeSquared }
        }

        return abs(sum - 1.0) < 1e-10
    }

    /// Normalize quantum state to satisfy Σ|cᵢ|² = 1
    ///
    /// Divides all amplitudes by √(Σᵢ |cᵢ|²) to ensure normalization constraint.
    /// Required after operations that may denormalize the state. Uses vectorized
    /// computation for large states.
    ///
    /// - Throws: QuantumStateError.cannotNormalizeZeroState if norm < 1e-15
    ///
    /// Example:
    /// ```swift
    /// var state = QuantumState(numQubits: 1, amplitudes: [
    ///     Complex(0, 0),
    ///     Complex(0, 0)
    /// ], bypassValidation: true)
    ///
    /// do {
    ///     try state.normalize()
    /// } catch QuantumStateError.cannotNormalizeZeroState {
    ///     print("Cannot normalize zero state")
    /// }
    ///
    /// // Valid normalization
    /// var unnormalized = QuantumState(numQubits: 1, amplitudes: [
    ///     Complex(3, 0),
    ///     Complex(4, 0)
    /// ])
    /// try unnormalized.normalize()
    /// // Now: [3/5, 4/5] since √(3² + 4²) = 5
    /// ```
    mutating func normalize() throws {
        let sumSquared: Double

        if amplitudes.count >= 64 {
            let magnitudesSquared = computeMagnitudesSquaredVectorized()
            sumSquared = magnitudesSquared.reduce(0.0, +)
        } else {
            sumSquared = amplitudes.reduce(0.0) { $0 + $1.magnitudeSquared }
        }

        guard sumSquared > 1e-15 else {
            throw QuantumStateError.cannotNormalizeZeroState
        }

        let norm = sqrt(sumSquared)
        amplitudes = amplitudes.map { $0 / norm }
    }

    // MARK: - State Access

    /// Get complex amplitude of specific basis state
    ///
    /// Returns the coefficient cᵢ for basis state |i⟩. Use `probability(ofState:)`
    /// for Born rule probabilities. Direct amplitude access is useful for
    /// quantum algorithm analysis and debugging.
    ///
    /// - Parameter stateIndex: Index of basis state (0 to 2^n-1)
    /// - Returns: Complex amplitude cᵢ
    ///
    /// Example:
    /// ```swift
    /// // |+⟩ state: (|0⟩ + |1⟩)/√2
    /// let plus = QuantumState(numQubits: 1, amplitudes: [
    ///     Complex(1/sqrt(2), 0),
    ///     Complex(1/sqrt(2), 0)
    /// ])
    ///
    /// let amp0 = plus.getAmplitude(ofState: 0)  // (1/√2, 0)
    /// let amp1 = plus.getAmplitude(ofState: 1)  // (1/√2, 0)
    /// ```
    func getAmplitude(ofState stateIndex: Int) -> Complex<Double> {
        precondition(stateIndex >= 0 && stateIndex < stateSpaceSize,
                     "State index out of bounds")
        return amplitudes[stateIndex]
    }

    /// Set complex amplitude of specific basis state (mutating)
    ///
    /// Directly modifies coefficient cᵢ for basis state |i⟩. Use with caution:
    /// may denormalize the state (call `normalize()` after). Primarily used
    /// internally by gate application and for testing.
    ///
    /// - Parameters:
    ///   - stateIndex: Index of basis state (0 to 2^n-1)
    ///   - amplitude: New complex amplitude cᵢ
    ///
    /// Example:
    /// ```swift
    /// var state = QuantumState(numQubits: 1)
    /// state.setAmplitude(ofState: 0, amplitude: Complex(1/sqrt(2), 0))
    /// state.setAmplitude(ofState: 1, amplitude: Complex(1/sqrt(2), 0))
    /// // Now state is |+⟩ = (|0⟩ + |1⟩)/√2
    /// ```
    mutating func setAmplitude(ofState stateIndex: Int, amplitude: Complex<Double>) {
        precondition(stateIndex >= 0 && stateIndex < stateSpaceSize,
                     "State index out of bounds")
        amplitudes[stateIndex] = amplitude
    }

    // MARK: - Qubit Indexing Utilities

    /// Extract bit value of specific qubit from state index
    ///
    /// Implements little-endian qubit ordering where qubit 0 is the least
    /// significant bit. Used internally for gate application and measurement.
    ///
    /// **Example**: For state index 5 (binary 101):
    /// - Qubit 0 = 1 (LSB)
    /// - Qubit 1 = 0
    /// - Qubit 2 = 1 (MSB)
    ///
    /// - Parameters:
    ///   - index: State index (0 to 2^n-1)
    ///   - qubit: Qubit position (0 to n-1)
    /// - Returns: 0 or 1
    ///
    /// Example:
    /// ```swift
    /// let state = QuantumState(numQubits: 3)
    /// let bit = state.getBit(index: 0b101, qubit: 0)  // 1 (LSB)
    /// let bit1 = state.getBit(index: 0b101, qubit: 1) // 0
    /// let bit2 = state.getBit(index: 0b101, qubit: 2) // 1 (MSB)
    /// ```
    func getBit(index: Int, qubit: Int) -> Int {
        (index >> qubit) & 1
    }

    /// Set bit value of specific qubit in state index
    /// - Parameters:
    ///   - index: Original state index
    ///   - qubit: Qubit position
    ///   - value: New bit value (0 or 1)
    /// - Returns: Modified state index
    func setBit(index: Int, qubit: Int, value: Int) -> Int {
        if value == 0 {
            index & ~(1 << qubit) // Clear bit
        } else {
            index | (1 << qubit) // Set bit
        }
    }

    /// Flip bit value of specific qubit in state index
    /// - Parameters:
    ///   - index: Original state index
    ///   - qubit: Qubit position
    /// - Returns: Modified state index with flipped bit
    func flipBit(index: Int, qubit: Int) -> Int {
        index ^ (1 << qubit)
    }

    // MARK: - Validation

    /// Validate all state invariants
    /// - Returns: True if state is valid
    func validate() -> Bool {
        // Check for NaN/Inf
        guard amplitudes.allSatisfy(\.isFinite) else { return false }
        guard isNormalized() else { return false }

        return true
    }

    // MARK: - CustomStringConvertible

    /// String representation showing significant amplitudes
    var description: String {
        // Show only non-negligible amplitudes
        let threshold = 1e-6
        var terms: [String] = []

        for (i, amp) in amplitudes.enumerated() {
            if amp.magnitudeSquared > threshold {
                let ampStr = String(format: "%.4f", amp.magnitude)
                let basisState = String(i, radix: 2)
                    .padding(toLength: numQubits, withPad: "0", startingAt: 0)
                terms.append("\(ampStr)|\(basisState)⟩")
            }
        }

        if terms.isEmpty {
            return "QuantumState(\(numQubits) qubits, near-zero)"
        }

        return "QuantumState(\(numQubits) qubits): " + terms.joined(separator: " + ")
    }
}
