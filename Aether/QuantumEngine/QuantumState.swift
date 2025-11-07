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

/// Represents an n-qubit quantum state as a complex amplitude vector.
///
/// State vector contains 2^n complex amplitudes representing superposition over
/// all computational basis states. Architecture is generic over qubit count -
/// supports 1 to 24+ qubits without hardcoding.
///
/// Mathematical basis:
/// - n-qubit system: 2^n dimensional Hilbert space
/// - State vector: |ψ⟩ = Σᵢ cᵢ|i⟩ where cᵢ ∈ ℂ, i ∈ [0, 2^n-1]
/// - Normalization: Σᵢ |cᵢ|² = 1 (total probability = 1)
/// - Qubit ordering: Little-endian (qubit 0 is LSB)
struct QuantumState: CustomStringConvertible, Sendable {
    // MARK: - Properties

    /// Array of complex amplitudes representing quantum state
    /// Length = 2^numQubits
    private(set) var amplitudes: [Complex<Double>]

    /// Number of qubits in this quantum system
    let numQubits: Int

    /// Size of state space (2^numQubits)
    var stateSpaceSize: Int { 1 << numQubits }

    // MARK: - Initialization

    /// Initialize quantum state with all qubits in |0⟩
    /// - Parameter numQubits: Number of qubits (supports 1-24+)
    init(numQubits: Int) {
        precondition(numQubits > 0, "Number of qubits must be positive")
        precondition(numQubits < 30, "Number of qubits too large (would exceed memory)")

        self.numQubits = numQubits
        let size = 1 << numQubits

        var amps = [Complex<Double>](repeating: .zero, count: size)
        amps[0] = .one
        amplitudes = amps
    }

    /// Initialize quantum state with custom amplitudes
    /// - Parameters:
    ///   - numQubits: Number of qubits
    ///   - amplitudes: Array of complex amplitudes (will auto-normalize if needed)
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

    /// Initialize single-qubit state
    /// - Parameter state: Either |0⟩ or |1⟩
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

    /// Calculate probability of measuring specific computational basis state
    /// - Parameter stateIndex: Index of basis state (0 to 2^n-1)
    /// - Returns: Probability P(i) = |cᵢ|² (Born rule)
    func probability(ofState stateIndex: Int) -> Double {
        precondition(stateIndex >= 0 && stateIndex < stateSpaceSize,
                     "State index out of bounds")
        return amplitudes[stateIndex].magnitudeSquared
    }

    /// Calculate probability distribution for all basis states
    /// - Returns: Array of probabilities [P(0), P(1), ..., P(2^n-1)]
    func probabilities() -> [Double] {
        if amplitudes.count >= 64 {
            computeMagnitudesSquaredVectorized()
        } else {
            amplitudes.map(\.magnitudeSquared)
        }
    }

    /// Calculate marginal probabilities for single qubit
    /// - Parameter qubit: Qubit index to measure
    /// - Returns: Tuple (P(|0⟩), P(|1⟩)) for this qubit
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

    /// Check if state is properly normalized
    /// - Returns: True if Σ|cᵢ|² ≈ 1.0 within tolerance
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

    /// Normalize the quantum state (mutating)
    /// Divides all amplitudes by √(Σ|cᵢ|²)
    /// - Throws: QuantumStateError.cannotNormalizeZeroState if state has near-zero norm
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

    /// Get amplitude of specific computational basis state
    /// - Parameter stateIndex: Index of basis state
    /// - Returns: Complex amplitude cᵢ
    func getAmplitude(ofState stateIndex: Int) -> Complex<Double> {
        precondition(stateIndex >= 0 && stateIndex < stateSpaceSize,
                     "State index out of bounds")
        return amplitudes[stateIndex]
    }

    /// Set amplitude of specific computational basis state (mutating)
    /// - Parameters:
    ///   - stateIndex: Index of basis state
    ///   - amplitude: New complex amplitude
    mutating func setAmplitude(ofState stateIndex: Int, amplitude: Complex<Double>) {
        precondition(stateIndex >= 0 && stateIndex < stateSpaceSize,
                     "State index out of bounds")
        amplitudes[stateIndex] = amplitude
    }

    // MARK: - Qubit Indexing Utilities

    /// Extract bit value of specific qubit from state index
    /// Uses little-endian ordering (qubit 0 is LSB)
    /// - Parameters:
    ///   - index: State index (0 to 2^n-1)
    ///   - qubit: Qubit position (0 to n-1)
    /// - Returns: 0 or 1
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

// MARK: - Equatable

extension QuantumState: Equatable {
    /// Compare two quantum states for equality
    /// - Parameters:
    ///   - lhs: Left-hand side quantum state
    ///   - rhs: Right-hand side quantum state
    /// - Returns: True if states have same qubit count and amplitudes (within tolerance)
    static func == (lhs: QuantumState, rhs: QuantumState) -> Bool {
        guard lhs.numQubits == rhs.numQubits else { return false }
        guard lhs.amplitudes.count == rhs.amplitudes.count else { return false }

        for (a, b) in zip(lhs.amplitudes, rhs.amplitudes) {
            if a != b { return false }
        }

        return true
    }
}
