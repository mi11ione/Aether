// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// User-controlled precision/performance trade-off for quantum simulation.
///
/// Provides three precision tiers balancing numerical accuracy against computational performance.
/// GPU acceleration uses Float32 internally (providing ~7 decimal digits of precision) while CPU
/// uses Float64 (providing ~16 decimal digits). The choice between backends affects both speed
/// and numerical stability, particularly for deep circuits and phase-sensitive algorithms.
///
/// Use `.fast` for general variational algorithms (VQE, QAOA), optimization loops, and initial
/// exploration where throughput matters more than precision (~1e-5 tolerance). Use `.balanced`
/// for production VQE runs and circuits with 50-200 gates where moderate precision (~1e-7) is
/// needed. Use `.accurate` for phase estimation, QFT-based algorithms, deep circuits (>200 gates),
/// and final verification runs requiring Float64 precision throughout (~1e-10 tolerance).
///
/// **Example:**
/// ```swift
/// let simulator = QuantumSimulator(precisionPolicy: .balanced)
/// let vqe = VQE(hamiltonian: h, ansatz: a, optimizer: o, precisionPolicy: .accurate)
/// ```
///
/// - SeeAlso: ``QuantumSimulator``
/// - SeeAlso: ``VQE``
/// - SeeAlso: ``QAOA``
/// - SeeAlso: ``SparseHamiltonian``
@frozen
public enum PrecisionPolicy: Sendable, CaseIterable, CustomStringConvertible {
    /// Maximum performance with Float32 GPU acceleration.
    ///
    /// Uses GPU for states with ≥10 qubits, Float32 precision throughout GPU paths.
    /// Acceptable deviation between GPU/CPU results: ~1e-5. Best for optimization
    /// loops, parameter sweeps, and general variational algorithms where throughput
    /// matters more than last-digit precision.
    case fast

    /// Balanced precision/performance trade-off.
    ///
    /// Raises GPU threshold to ≥12 qubits, reducing Float32 accumulation errors
    /// for medium-depth circuits. Tolerance ~1e-7. Recommended for production
    /// VQE runs and circuits with 50-200 gates.
    case balanced

    /// Maximum numerical precision with CPU-only execution.
    ///
    /// Forces Float64 CPU execution regardless of state size. Uses Accelerate
    /// framework for vectorized operations. Tolerance ~1e-10. Required for
    /// phase-sensitive algorithms (QPE, QFT), deep circuits (>200 gates), and
    /// final verification runs where numerical accuracy is paramount.
    case accurate

    // MARK: - Computed Properties

    /// Minimum qubit count for GPU acceleration under this policy.
    ///
    /// States with fewer qubits use CPU execution. Higher thresholds reduce
    /// Float32 precision loss at the cost of GPU acceleration opportunities.
    /// Returns 10 for `.fast`, 12 for `.balanced`, and `Int.max` for `.accurate`
    /// which effectively disables GPU entirely.
    @inlinable
    public var gpuQubitThreshold: Int {
        switch self {
        case .fast: 10
        case .balanced: 12
        case .accurate: Int.max
        }
    }

    /// Numerical tolerance for this precision level.
    ///
    /// Maximum acceptable deviation between computed results under different
    /// backends or repeated runs. Used for validation and convergence criteria.
    /// Returns 1e-5 for `.fast` (sufficient for optimization convergence),
    /// 1e-7 for `.balanced` (production quality), and 1e-10 for `.accurate`
    /// (matches Float64 machine epsilon regime).
    @inlinable
    public var tolerance: Double {
        switch self {
        case .fast: 1e-5
        case .balanced: 1e-7
        case .accurate: 1e-10
        }
    }

    /// Whether GPU acceleration is permitted under this policy.
    ///
    /// When false, forces CPU-only execution regardless of state size.
    /// GPU paths use Float32 precision; CPU paths use Float64. Returns
    /// true for `.fast` and `.balanced` (GPU enabled above threshold),
    /// false for `.accurate` (CPU-only for maximum precision).
    @inlinable
    public var isGPUEnabled: Bool {
        switch self {
        case .fast: true
        case .balanced: true
        case .accurate: false
        }
    }

    /// Human-readable description of precision policy semantics.
    @inlinable
    public var description: String {
        switch self {
        case .fast:
            "Fast (GPU threshold: 10 qubits, tolerance: 1e-5)"
        case .balanced:
            "Balanced (GPU threshold: 12 qubits, tolerance: 1e-7)"
        case .accurate:
            "Accurate (CPU-only, tolerance: 1e-10)"
        }
    }

    // MARK: - Static Properties

    /// Default precision policy for general use.
    ///
    /// Returns `.fast` which maximizes throughput while maintaining acceptable
    /// precision for most variational algorithms. Override with `.balanced` or
    /// `.accurate` for production runs or phase-sensitive algorithms.
    @inlinable
    public static var `default`: PrecisionPolicy { .fast }

    // MARK: - Static Methods

    /// Determines whether GPU should be used for given qubit count and policy.
    ///
    /// Combines qubit threshold check with policy GPU flag. Returns true only
    /// when policy permits GPU acceleration AND state size meets threshold.
    ///
    /// **Example:**
    /// ```swift
    /// let useGPU = PrecisionPolicy.shouldUseGPU(qubits: 15, policy: .balanced)
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits in quantum state
    ///   - policy: Precision policy governing backend selection
    /// - Returns: true if GPU should be used, false for CPU execution
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    @inline(__always)
    public static func shouldUseGPU(qubits: Int, policy: PrecisionPolicy) -> Bool {
        policy.isGPUEnabled && qubits >= policy.gpuQubitThreshold
    }
}
