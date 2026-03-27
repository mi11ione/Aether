// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under Apache 2.0

import Foundation

/// Constraint types for parameter optimization
///
/// Encodes optimization constraints for variational parameters: bounded ranges for
/// box-constrained optimization, non-negative constraints for physical quantities
/// like probabilities, and periodic wrapping for angular parameters. Constraints
/// are applied during optimization to project parameter values into valid regions.
///
/// **Example:**
/// ```swift
/// let bounded = ParameterConstraint.bounded(min: 0.0, max: 2 * .pi)
/// let clamped = bounded.apply(to: 7.0)  // 6.283... (clamped to max)
///
/// let nonNeg = ParameterConstraint.nonNegative
/// let positive = nonNeg.apply(to: -0.5)  // 0.0
///
/// let periodic = ParameterConstraint.periodic(period: 2 * .pi)
/// let wrapped = periodic.apply(to: 3 * .pi)  // π (wrapped)
/// ```
///
/// - SeeAlso: ``Parameter``
/// - SeeAlso: ``ParameterValue``
@frozen
public enum ParameterConstraint: Equatable, Hashable, Sendable {
    /// Bounded range constraint [min, max]
    ///
    /// Clamps parameter values to the specified closed interval.
    /// Used for box-constrained optimization where parameters must stay within bounds.
    ///
    /// - Parameters:
    ///   - min: Minimum allowed value (inclusive)
    ///   - max: Maximum allowed value (inclusive)
    /// - Precondition: min < max
    case bounded(min: Double, max: Double)

    /// Non-negative constraint (value >= 0)
    ///
    /// Ensures parameter values are non-negative by taking max(0, value).
    /// Used for physical quantities that cannot be negative such as probabilities,
    /// decay rates, or amplitudes.
    case nonNegative

    /// Periodic wrapping constraint
    ///
    /// Wraps parameter values to [0, period) using modular arithmetic.
    /// Used for angular parameters where values outside the principal domain
    /// are equivalent to values within it.
    ///
    /// - Parameter period: Period length (must be positive)
    /// - Precondition: period > 0
    case periodic(period: Double)

    /// Apply constraint to a parameter value
    ///
    /// Enforces the constraint by transforming the value into the valid region.
    /// For bounded constraints, clamps to [min, max]. For non-negative, takes
    /// max(0, value). For periodic, wraps to [0, period) using fmod.
    ///
    /// **Example:**
    /// ```swift
    /// let bounded = ParameterConstraint.bounded(min: -1.0, max: 1.0)
    /// let clamped = bounded.apply(to: 2.5)  // 1.0
    ///
    /// let periodic = ParameterConstraint.periodic(period: 2 * .pi)
    /// let wrapped = periodic.apply(to: 3 * .pi)  // π
    /// ```
    ///
    /// - Parameter value: Parameter value to constrain
    /// - Returns: Constrained parameter value
    /// - Complexity: O(1)
    @_effects(readonly)
    @_optimize(speed)
    @inlinable
    public func apply(to value: Double) -> Double {
        switch self {
        case let .bounded(min, max):
            return Swift.min(Swift.max(value, min), max)
        case .nonNegative:
            return Swift.max(0.0, value)
        case let .periodic(period):
            let result = fmod(value, period)
            return result < 0.0 ? result + period : result
        }
    }

    /// Bounds for optimizer bound arrays
    ///
    /// Returns the explicit bounds for bounded constraints, enabling integration
    /// with optimizers that accept bound arrays. Returns nil for constraints
    /// that do not have finite bounds (non-negative, periodic).
    ///
    /// **Example:**
    /// ```swift
    /// let bounded = ParameterConstraint.bounded(min: 0.0, max: 1.0)
    /// let range = bounded.bounds  // (min: 0.0, max: 1.0)
    ///
    /// let nonNeg = ParameterConstraint.nonNegative
    /// let noBounds = nonNeg.bounds  // nil
    /// ```
    ///
    /// - Returns: Tuple of (min, max) for bounded constraints, nil otherwise
    /// - Complexity: O(1)
    @inlinable
    public var bounds: (min: Double, max: Double)? {
        switch self {
        case let .bounded(min, max):
            (min, max)
        case .nonNegative, .periodic:
            nil
        }
    }
}

public extension ParameterConstraint {
    /// Create a bounded constraint with validation
    ///
    /// Factory method that validates min < max before creating the constraint.
    ///
    /// **Example:**
    /// ```swift
    /// let constraint = ParameterConstraint.makeBounded(min: 0.0, max: 2 * .pi)
    /// ```
    ///
    /// - Parameters:
    ///   - min: Minimum allowed value (inclusive)
    ///   - max: Maximum allowed value (inclusive)
    /// - Returns: Bounded constraint
    /// - Precondition: min < max
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    static func makeBounded(min: Double, max: Double) -> ParameterConstraint {
        ValidationUtilities.validateOpenMinRange(max, min: min, max: .infinity, name: "Bounded constraint max")
        return .bounded(min: min, max: max)
    }

    /// Create a periodic constraint with validation
    ///
    /// Factory method that validates period > 0 before creating the constraint.
    ///
    /// **Example:**
    /// ```swift
    /// let constraint = ParameterConstraint.makePeriodic(period: 2 * .pi)
    /// ```
    ///
    /// - Parameter period: Period length (must be positive)
    /// - Returns: Periodic constraint
    /// - Precondition: period > 0
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    static func makePeriodic(period: Double) -> ParameterConstraint {
        ValidationUtilities.validatePositiveDouble(period, name: "Periodic constraint period")
        return .periodic(period: period)
    }
}
