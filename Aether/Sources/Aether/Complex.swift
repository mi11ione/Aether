// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Scalar type constraint for complex numbers with compile-time safe constants and math operations
public protocol ComplexScalar: BinaryFloatingPoint, Sendable {
    static var complexZero: Self { get }
    static var complexOne: Self { get }
    static var complexEpsilon: Self { get }
    static var complexDivisionThreshold: Self { get }

    static func complexSqrt(_ value: Self) -> Self
    static func complexCos(_ value: Self) -> Self
    static func complexSin(_ value: Self) -> Self
    static func complexAtan2(_ y: Self, _ x: Self) -> Self
    static func complexAbs(_ value: Self) -> Self
}

extension Double: ComplexScalar {
    @inlinable public static var complexZero: Double { 0.0 }
    @inlinable public static var complexOne: Double { 1.0 }
    @inlinable public static var complexEpsilon: Double { 1e-10 }
    @inlinable public static var complexDivisionThreshold: Double { 1e-15 }

    @inlinable public static func complexSqrt(_ value: Double) -> Double { Foundation.sqrt(value) }
    @inlinable public static func complexCos(_ value: Double) -> Double { Foundation.cos(value) }
    @inlinable public static func complexSin(_ value: Double) -> Double { Foundation.sin(value) }
    @inlinable public static func complexAtan2(_ y: Double, _ x: Double) -> Double { Foundation.atan2(y, x) }
    @inlinable public static func complexAbs(_ value: Double) -> Double { value < 0 ? -value : value }
}

extension Float: ComplexScalar {
    @inlinable public static var complexZero: Float { 0.0 }
    @inlinable public static var complexOne: Float { 1.0 }
    @inlinable public static var complexEpsilon: Float { 1e-6 }
    @inlinable public static var complexDivisionThreshold: Float { 1e-10 }

    @inlinable public static func complexSqrt(_ value: Float) -> Float { Foundation.sqrt(value) }
    @inlinable public static func complexCos(_ value: Float) -> Float { Foundation.cos(value) }
    @inlinable public static func complexSin(_ value: Float) -> Float { Foundation.sin(value) }
    @inlinable public static func complexAtan2(_ y: Float, _ x: Float) -> Float { Foundation.atan2(y, x) }
    @inlinable public static func complexAbs(_ value: Float) -> Float { value < 0 ? -value : value }
}

/// Generic complex number type for quantum computing
///
/// Provides complete complex arithmetic with support for Double and Float precision.
/// Implements polar/cartesian conversions, exponential form via Euler's formula,
/// and standard arithmetic operations required for quantum state manipulation.
///
/// **Mathematical representation**: z = a + bi where i² = -1
///
/// **Usage patterns**:
/// - Cartesian: Create from real and imaginary components
/// - Polar: Create from magnitude and phase using `fromPolar`
/// - Exponential: Create unit complex numbers using `exp` (Euler's formula)
///
/// Example:
/// ```swift
/// let z1 = Complex(3.0, 4.0)              // 3 + 4i
/// let z2 = Complex<Double>.fromPolar(r: 5.0, theta: .pi/4)  // Polar form
/// let z3 = Complex<Double>.exp(.pi/2)     // e^(iπ/2) = i
///
/// let sum = z1 + z2
/// let product = z1 * z2
/// let magnitude = z1.magnitude            // √(3² + 4²) = 5
/// let phase = z1.phase                    // atan2(4, 3)
/// let conj = z1.conjugate                 // 3 - 4i
/// ```
@frozen
public struct Complex<T: ComplexScalar>: Equatable, Hashable, CustomStringConvertible, ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral, AdditiveArithmetic, Sendable {
    // MARK: - Properties

    public let real: T
    public let imaginary: T

    // MARK: - Initialization

    /// Create complex number from cartesian coordinates
    ///
    /// - Parameters:
    ///   - real: Real component (a in a + bi)
    ///   - imaginary: Imaginary component (b in a + bi)
    ///
    /// Example:
    /// ```swift
    /// let z = Complex(3.0, 4.0)  // 3 + 4i
    /// ```
    @inlinable
    public init(_ real: T, _ imaginary: T) {
        self.real = real
        self.imaginary = imaginary
    }

    /// Create real number as complex (imaginary part = 0)
    ///
    /// - Parameter real: Real component
    ///
    /// Example:
    /// ```swift
    /// let z = Complex(5.0)  // 5 + 0i
    /// ```
    @inlinable
    public init(_ real: T) {
        self.real = real
        imaginary = T.complexZero
    }

    @inlinable
    public init(integerLiteral value: Int) {
        real = T(value)
        imaginary = T.complexZero
    }

    @inlinable
    public init(floatLiteral value: Double) {
        real = T(value)
        imaginary = T.complexZero
    }

    // MARK: - Static Constants

    @inlinable
    public static var zero: Complex<T> {
        Complex(T.complexZero, T.complexZero)
    }

    @inlinable
    public static var one: Complex<T> {
        Complex(T.complexOne, T.complexZero)
    }

    /// Imaginary unit where i² = -1
    ///
    /// Example:
    /// ```swift
    /// let i = Complex<Double>.i           // 0 + 1i
    /// let iPower2 = i * i                 // -1 + 0i
    /// let iPower4 = i * i * i * i         // 1 + 0i
    /// ```
    @inlinable
    public static var i: Complex<T> {
        Complex(T.complexZero, T.complexOne)
    }

    // MARK: - Computed Properties

    /// Complex conjugate: (a + bi)* = a - bi
    ///
    /// Used in quantum computing for computing probabilities and inner products.
    ///
    /// Example:
    /// ```swift
    /// let z = Complex(3.0, 4.0)
    /// let conj = z.conjugate              // 3 - 4i
    /// let prob = (z * z.conjugate).real   // |z|² = 25
    /// ```
    @_effects(readonly)
    @inlinable
    @_eagerMove
    public func conjugate() -> Complex<T> {
        Complex(real, -imaginary)
    }

    /// Magnitude squared: |z|² = a² + b²
    ///
    /// Optimized computation avoiding square root. Used for probabilities
    /// in quantum mechanics where |ψ|² represents probability.
    ///
    /// Example:
    /// ```swift
    /// let amplitude = Complex(0.6, 0.8)
    /// let probability = amplitude.magnitudeSquared  // 1.0
    /// ```
    @inlinable
    public var magnitudeSquared: T {
        real * real + imaginary * imaginary
    }

    /// Magnitude: |z| = √(a² + b²)
    ///
    /// Example:
    /// ```swift
    /// let z = Complex(3.0, 4.0)
    /// let r = z.magnitude                 // 5.0
    /// ```
    @_effects(readonly)
    @inlinable
    public func magnitude() -> T {
        T.complexSqrt(magnitudeSquared)
    }

    /// Phase/argument: arg(z) = atan2(b, a) ∈ (-π, π]
    ///
    /// Returns principal argument in radians. Used for polar representation.
    ///
    /// Example:
    /// ```swift
    /// let z = Complex(1.0, 1.0)
    /// let theta = z.phase                 // π/4 radians
    /// ```
    @_effects(readonly)
    @inlinable
    public func phase() -> T {
        T.complexAtan2(imaginary, real)
    }

    @inlinable
    public var isFinite: Bool {
        real.isFinite && imaginary.isFinite
    }

    // MARK: - Polar Conversions

    /// Create complex number from polar coordinates: z = r·e^(iθ)
    ///
    /// Uses Euler's formula: r·e^(iθ) = r·cos(θ) + r·i·sin(θ)
    ///
    /// - Parameters:
    ///   - r: Magnitude (radius)
    ///   - theta: Phase angle in radians
    /// - Returns: Complex number in cartesian form
    ///
    /// Example:
    /// ```swift
    /// let z = Complex<Double>.fromPolar(r: 2.0, theta: .pi/3)
    /// // Creates: 2·e^(iπ/3) = 1 + √3i
    /// ```
    @_effects(readonly)
    @inlinable
    @_eagerMove
    public static func fromPolar(r: T, theta: T) -> Complex<T> {
        Complex(r * T.complexCos(theta), r * T.complexSin(theta))
    }

    /// Convert to polar coordinates
    ///
    /// - Returns: Tuple of (magnitude, phase)
    ///
    /// Example:
    /// ```swift
    /// let z = Complex(3.0, 4.0)
    /// let (r, theta) = z.toPolar()
    /// // r = 5.0, theta = atan2(4, 3) ≈ 0.927 radians
    /// ```
    @_effects(readonly)
    @inlinable
    public func toPolar() -> (magnitude: T, phase: T) {
        (magnitude(), phase())
    }

    // MARK: - Exponential Form

    /// Euler's formula: e^(iθ) = cos(θ) + i·sin(θ)
    ///
    /// Creates unit complex number (|z| = 1) at given phase angle.
    /// Fundamental for quantum gate representation.
    ///
    /// - Parameter theta: Phase angle in radians
    /// - Returns: Unit complex number at angle θ
    ///
    /// Example:
    /// ```swift
    /// let i = Complex<Double>.exp(.pi/2)      // e^(iπ/2) = i
    /// let minusOne = Complex<Double>.exp(.pi) // e^(iπ) = -1
    /// ```
    @_effects(readonly)
    @inlinable
    @_eagerMove
    public static func exp(_ theta: T) -> Complex<T> {
        Complex(T.complexCos(theta), T.complexSin(theta))
    }

    // MARK: - CustomStringConvertible

    public var description: String {
        let epsilon = T.complexEpsilon

        if T.complexAbs(imaginary) < epsilon {
            return "\(real)"
        }
        if T.complexAbs(real) < epsilon {
            return "\(imaginary)i"
        }
        if imaginary >= 0 {
            return "\(real) + \(imaginary)i"
        } else {
            return "\(real) - \(T.complexAbs(imaginary))i"
        }
    }

    // MARK: - Equatable

    public static func == (lhs: Complex<T>, rhs: Complex<T>) -> Bool {
        let epsilon = T.complexEpsilon
        return T.complexAbs(lhs.real - rhs.real) < epsilon &&
            T.complexAbs(lhs.imaginary - rhs.imaginary) < epsilon
    }

    // MARK: - Hashable

    public func hash(into hasher: inout Hasher) {
        let quantized = (
            (real / T.complexEpsilon).rounded() * T.complexEpsilon,
            (imaginary / T.complexEpsilon).rounded() * T.complexEpsilon
        )
        hasher.combine(quantized.0)
        hasher.combine(quantized.1)
    }
}

// MARK: - Arithmetic Operators

@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_effects(readonly)
@inlinable
public func + <T>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    Complex(lhs.real + rhs.real, lhs.imaginary + rhs.imaginary)
}

@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_effects(readonly)
@inlinable
public func - <T>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    Complex(lhs.real - rhs.real, lhs.imaginary - rhs.imaginary)
}

@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_optimize(speed)
@_effects(readonly)
@inlinable
public func * <T>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    let real = lhs.real * rhs.real - lhs.imaginary * rhs.imaginary
    let imag = lhs.real * rhs.imaginary + lhs.imaginary * rhs.real
    return Complex(real, imag)
}

@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_optimize(speed)
@inlinable
public func / <T>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    let denominator = rhs.magnitudeSquared

    guard denominator > T.complexDivisionThreshold else {
        return Complex(T.nan, T.nan)
    }
    let real = (lhs.real * rhs.real + lhs.imaginary * rhs.imaginary) / denominator
    let imag = (lhs.imaginary * rhs.real - lhs.real * rhs.imaginary) / denominator
    return Complex(real, imag)
}

@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_effects(readonly)
@inlinable
public prefix func - <T>(z: Complex<T>) -> Complex<T> {
    Complex(-z.real, -z.imaginary)
}

// MARK: - Scalar Operations

@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_effects(readonly)
@inlinable
public func * <T>(scalar: T, z: Complex<T>) -> Complex<T> {
    Complex(scalar * z.real, scalar * z.imaginary)
}

@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_effects(readonly)
@inlinable
public func * <T>(z: Complex<T>, scalar: T) -> Complex<T> {
    Complex(z.real * scalar, z.imaginary * scalar)
}

@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_effects(readonly)
@inlinable
public func / <T>(z: Complex<T>, scalar: T) -> Complex<T> {
    Complex(z.real / scalar, z.imaginary / scalar)
}

// MARK: - Compound Assignment Operators

@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@inlinable
public func *= <T>(lhs: inout Complex<T>, rhs: Complex<T>) {
    lhs = lhs * rhs
}

@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@inlinable
public func /= <T>(lhs: inout Complex<T>, rhs: Complex<T>) {
    lhs = lhs / rhs
}

@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@inlinable
public func *= <T>(lhs: inout Complex<T>, scalar: T) {
    lhs = lhs * scalar
}

@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@inlinable
public func /= <T>(lhs: inout Complex<T>, scalar: T) {
    lhs = lhs / scalar
}
