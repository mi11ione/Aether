// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Configuration parameters for Density Matrix Renormalization Group (DMRG) algorithm.
///
/// Controls sweep count, convergence threshold, and subspace expansion settings for variational
/// ground state optimization of one-dimensional quantum systems. DMRG iteratively optimizes
/// Matrix Product State (MPS) tensors through left-to-right and right-to-left sweeps until
/// energy convergence.
///
/// Subspace expansion adds noise to prevent local minima trapping during early sweeps.
/// Noise strength typically decreases geometrically across sweeps (e.g., 1e-4 to 1e-8).
/// Disable subspace expansion for production runs after initial convergence.
///
/// **Example:**
/// ```swift
/// let config = DMRGConfiguration(maxSweeps: 30, convergenceThreshold: 1e-10)
/// let noiseConfig = DMRGConfiguration(subspaceExpansion: true, noiseStrength: 1e-4)
/// ```
///
/// - SeeAlso: ``DMRGSweepDirection``
@frozen
public struct DMRGConfiguration: Sendable, Equatable {
    /// Maximum number of DMRG sweeps before termination.
    ///
    /// Each sweep consists of one left-to-right pass followed by one right-to-left pass
    /// through all MPS tensors. Typical values range from 10-50 depending on system size
    /// and required precision.
    ///
    /// **Example:**
    /// ```swift
    /// let config = DMRGConfiguration(maxSweeps: 50)
    /// let optimized = DMRG(hamiltonian: mpo, maxBondDimension: 64, configuration: config)
    /// let result = await optimized.findGroundState(from: nil)
    /// ```
    public let maxSweeps: Int

    /// Energy convergence threshold for early termination.
    ///
    /// DMRG terminates when |E(sweep) - E(sweep-1)| < convergenceThreshold. Smaller values
    /// require more sweeps but yield higher precision. Typical values: 1e-6 to 1e-12.
    ///
    /// **Example:**
    /// ```swift
    /// let highPrecision = DMRGConfiguration(convergenceThreshold: 1e-12)
    /// ```
    public let convergenceThreshold: Double

    /// Enable subspace expansion for improved convergence.
    ///
    /// Subspace expansion adds controlled noise to MPS tensors during optimization,
    /// helping escape local minima. Recommended for initial sweeps with large bond
    /// dimensions. Disable for final high-precision sweeps.
    ///
    /// **Example:**
    /// ```swift
    /// let withExpansion = DMRGConfiguration(subspaceExpansion: true, noiseStrength: 1e-5)
    /// ```
    public let subspaceExpansion: Bool

    /// Noise strength for subspace expansion.
    ///
    /// Controls magnitude of random perturbations added during subspace expansion.
    /// Only effective when ``subspaceExpansion`` is true. Typical values: 1e-3 to 1e-6.
    /// Set to 0.0 to disable noise while keeping expansion enabled.
    ///
    /// **Example:**
    /// ```swift
    /// let config = DMRGConfiguration(subspaceExpansion: true, noiseStrength: 1e-4)
    /// ```
    public let noiseStrength: Double

    /// Creates DMRG configuration with specified parameters.
    ///
    /// Validates all parameters using preconditions: maxSweeps must be positive,
    /// convergenceThreshold must be positive, and noiseStrength must be non-negative.
    ///
    /// **Example:**
    /// ```swift
    /// let config = DMRGConfiguration(maxSweeps: 30, convergenceThreshold: 1e-10)
    /// let noiseConfig = DMRGConfiguration(subspaceExpansion: true, noiseStrength: 1e-5)
    /// let dmrg = DMRG(hamiltonian: mpo, maxBondDimension: 64, configuration: config)
    /// ```
    ///
    /// - Parameters:
    ///   - maxSweeps: Maximum sweep count (default: 20)
    ///   - convergenceThreshold: Energy convergence threshold (default: 1e-8)
    ///   - subspaceExpansion: Enable subspace expansion (default: false)
    ///   - noiseStrength: Noise magnitude for expansion (default: 0.0)
    /// - Complexity: O(1)
    public init(
        maxSweeps: Int = 20,
        convergenceThreshold: Double = 1e-8,
        subspaceExpansion: Bool = false,
        noiseStrength: Double = 0.0,
    ) {
        ValidationUtilities.validatePositiveInt(maxSweeps, name: "maxSweeps")
        ValidationUtilities.validatePositiveDouble(convergenceThreshold, name: "convergenceThreshold")
        ValidationUtilities.validateNonNegativeDouble(noiseStrength, name: "noiseStrength")

        self.maxSweeps = maxSweeps
        self.convergenceThreshold = convergenceThreshold
        self.subspaceExpansion = subspaceExpansion
        self.noiseStrength = noiseStrength
    }
}

/// Sweep direction for DMRG optimization pass.
///
/// DMRG alternates between left-to-right and right-to-left sweeps through the MPS chain.
/// Each direction optimizes tensors sequentially while maintaining canonical form.
///
/// **Example:**
/// ```swift
/// var direction = DMRGSweepDirection.leftToRight
/// for sweep in 0..<config.maxSweeps {
///     performSweep(direction: direction)
///     direction = (direction == .leftToRight) ? .rightToLeft : .leftToRight
/// }
/// ```
///
/// - SeeAlso: ``DMRGConfiguration``
@frozen
public enum DMRGSweepDirection: Sendable, Equatable {
    /// Sweep from left boundary to right boundary.
    ///
    /// Optimizes MPS tensors in order: site 0, site 1, ..., site N-1.
    /// Maintains right-canonical form during sweep.
    case leftToRight

    /// Sweep from right boundary to left boundary.
    ///
    /// Optimizes MPS tensors in order: site N-1, site N-2, ..., site 0.
    /// Maintains left-canonical form during sweep.
    case rightToLeft
}
