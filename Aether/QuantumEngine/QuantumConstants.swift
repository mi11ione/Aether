//
//  QuantumConstants.swift
//  Aether
//
//  Physical and numerical constants for quantum computing simulation
//  Created by mi11ion on 23/10/25.
//

import Foundation

/// Global constants for quantum computing simulation
/// All values are based on standard quantum computing literature and numerical analysis
enum QuantumConstants {
    // MARK: - Numerical Tolerances

    /// Tolerance for comparing Double floating-point values
    /// Used for: state normalization, probability checks, unitarity verification
    nonisolated static let doubleTolerance: Double = 1e-10

    /// Tolerance for comparing Float floating-point values
    /// Used for: Metal GPU computations
    static let floatTolerance: Float = 1e-6

    /// Minimum magnitude for non-zero values (Double)
    /// Below this threshold, values are considered numerical zero
    static let doubleMinThreshold: Double = 1e-15

    /// Minimum magnitude for non-zero values (Float)
    static let floatMinThreshold: Float = 1e-10

    /// Display threshold for amplitudes
    /// Amplitudes smaller than this are omitted from string representations
    static let displayThreshold: Double = 1e-6

    /// Tolerance for probability sum validation
    /// Probabilities must sum to 1.0 within this tolerance
    static let probabilityTolerance: Double = 1e-6

    // MARK: - Performance Thresholds

    /// GPU acceleration threshold (qubits)
    /// States with >= this many qubits use Metal GPU acceleration
    /// States with < this many qubits use CPU (overhead not worth it)
    static let gpuAccelerationThreshold: Int = 10

    /// Maximum recommended qubits for Grover algorithm
    /// Beyond this, circuit depth becomes very large
    static let groverMaxQubits: Int = 10

    // MARK: - Circuit Caching

    /// State cache interval for circuit execution
    /// Cache state every N gates for smooth scrubbing
    static let stateCacheInterval: Int = 5

    /// Maximum cached states
    /// Limits memory usage for long circuits
    static let maxCachedStates: Int = 20

    // MARK: - Physical Constants

    /// Pi (high precision)
    static let pi: Double = .pi

    /// Planck constant (for future physical simulation extensions)
    /// Not currently used but available for audiovisual mapping
    static let planckConstant: Double = 6.62607015e-34 // J⋅Hz⁻¹
}
