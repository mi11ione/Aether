// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Scalar type constraint for complex numbers with compile-time safe constants and math operations.
///
/// Provides precision-specific constants and math operations for `Complex<T>`. Only Double and Float
/// conformances are supported. Users cannot conform custom types.
///
/// - Note: This protocol is internal implementation detail. Use `Complex<Double>` or `Complex<Float>`.
public protocol ComplexScalar: BinaryFloatingPoint, Sendable {
    static var zero: Self { get }
    static var one: Self { get }
    static var epsilon: Self { get }
    static var divisionThreshold: Self { get }

    static func squareRoot(of value: Self) -> Self
    static func cosine(of value: Self) -> Self
    static func sine(of value: Self) -> Self
    static func arctangent(y: Self, x: Self) -> Self
    static func absoluteValue(of value: Self) -> Self
    static func hypotenuse(x: Self, y: Self) -> Self
    static func fusedMultiplyAdd(_ a: Self, _ b: Self, _ c: Self) -> Self
}

extension Double: ComplexScalar {
    /// The additive identity.
    @inlinable public static var zero: Double {
        0.0
    }

    /// The multiplicative identity.
    @inlinable public static var one: Double {
        1.0
    }

    /// The comparison tolerance for Double precision.
    @inlinable public static var epsilon: Double {
        1e-10
    }

    /// The minimum denominator magnitude for safe division.
    @inlinable public static var divisionThreshold: Double {
        1e-15
    }

    /// Returns the square root.
    @inlinable public static func squareRoot(of value: Double) -> Double {
        Foundation.sqrt(value)
    }

    /// Returns the cosine of the angle.
    @inlinable public static func cosine(of value: Double) -> Double {
        Foundation.cos(value)
    }

    /// Returns the sine of the angle.
    @inlinable public static func sine(of value: Double) -> Double {
        Foundation.sin(value)
    }

    /// Returns the two-argument arctangent.
    @inlinable public static func arctangent(y: Double, x: Double) -> Double {
        Foundation.atan2(y, x)
    }

    /// Returns the absolute value.
    @inlinable public static func absoluteValue(of value: Double) -> Double {
        Double(bitPattern: value.bitPattern & 0x7FFF_FFFF_FFFF_FFFF)
    }

    /// Returns the hypotenuse from the two sides.
    @inlinable public static func hypotenuse(x: Double, y: Double) -> Double {
        Foundation.hypot(x, y)
    }

    /// Returns a fused multiply-add result.
    @inlinable public static func fusedMultiplyAdd(_ a: Double, _ b: Double, _ c: Double) -> Double {
        Foundation.fma(a, b, c)
    }
}

extension Float: ComplexScalar {
    /// The additive identity.
    @inlinable public static var zero: Float {
        0.0
    }

    /// The multiplicative identity.
    @inlinable public static var one: Float {
        1.0
    }

    /// The comparison tolerance for Float precision.
    @inlinable public static var epsilon: Float {
        1e-6
    }

    /// The minimum denominator magnitude for safe division.
    @inlinable public static var divisionThreshold: Float {
        1e-10
    }

    /// Returns the square root.
    @inlinable public static func squareRoot(of value: Float) -> Float {
        Foundation.sqrt(value)
    }

    /// Returns the cosine of the angle.
    @inlinable public static func cosine(of value: Float) -> Float {
        Foundation.cos(value)
    }

    /// Returns the sine of the angle.
    @inlinable public static func sine(of value: Float) -> Float {
        Foundation.sin(value)
    }

    /// Returns the two-argument arctangent.
    @inlinable public static func arctangent(y: Float, x: Float) -> Float {
        Foundation.atan2(y, x)
    }

    /// Returns the absolute value.
    @inlinable public static func absoluteValue(of value: Float) -> Float {
        Float(bitPattern: value.bitPattern & 0x7FFF_FFFF)
    }

    /// Returns the hypotenuse from the two sides.
    @inlinable public static func hypotenuse(x: Float, y: Float) -> Float {
        Foundation.hypot(x, y)
    }

    /// Returns a fused multiply-add result.
    @inlinable public static func fusedMultiplyAdd(_ a: Float, _ b: Float, _ c: Float) -> Float {
        Foundation.fma(a, b, c)
    }
}

/// Generic complex number type for quantum computing with type-safe arithmetic.
///
/// Represents complex numbers z = a + bi where i² = -1, supporting Double and Float precision.
/// Provides complete arithmetic operations, polar/cartesian conversions, and Euler's formula
/// for phase representation.
///
/// In quantum computing, complex amplitudes encode quantum state probabilities where |ψ|² represents
/// measurement probability (Born rule), and inner products use complex conjugation. Use
/// `Complex<Double>` for quantum state amplitudes (default) or `Complex<Float>` for GPU computations
/// where 7 decimal digits suffice. Equality uses epsilon-based comparison (1e-10 for Double, 1e-6
/// for Float) suitable for floating-point quantum computations. All operations are O(1), optimized
/// with fused multiply-add (FMA) for ``magnitudeSquared`` and division, specialized for Double and
/// Float at compile time.
///
/// **Example:**
/// ```swift
/// let amplitude = Complex(0.6, 0.8)
/// let probability = amplitude.magnitudeSquared  // 1.0 (normalized)
/// let phaseGate = Complex(magnitude: 1.0, phase: .pi/4)
/// let i = Complex<Double>(phase: .pi/2)  // i
/// let product = amplitude * amplitude.conjugate
/// ```
@frozen
public struct Complex<T: ComplexScalar>: Equatable, Hashable, CustomStringConvertible, ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral, AdditiveArithmetic, Sendable {
    // MARK: - Properties

    /// The real component.
    public let real: T
    /// The imaginary component.
    public let imaginary: T

    // MARK: - Initialization

    /// Creates a complex number from real and imaginary components: z = a + bi
    ///
    /// Primary initializer for explicit cartesian coordinates.
    ///
    /// - Parameters:
    ///   - real: Real component a
    ///   - imaginary: Imaginary component b
    ///
    /// **Example:**
    /// ```swift
    /// let amplitude = Complex(0.6, 0.8)
    /// let real = amplitude.real        // 0.6
    /// let imaginary = amplitude.imaginary  // 0.8
    /// ```
    @inlinable
    public init(_ real: T, _ imaginary: T) {
        self.real = real
        self.imaginary = imaginary
    }

    /// Creates a real number as complex with zero imaginary part
    ///
    /// - Parameter real: Real component
    ///
    /// **Example:**
    /// ```swift
    /// let eigenvalue = Complex(1.0)  // 1 + 0i
    /// let real = eigenvalue.real          // 1.0
    /// let imaginary = eigenvalue.imaginary  // 0.0
    /// ```
    @inlinable
    public init(_ real: T) {
        self.init(real, T.zero)
    }

    /// Integer literal support enabling `let z: Complex<Double> = 5`
    ///
    /// **Example:**
    /// ```swift
    /// let identity: Complex<Double> = 1
    /// let real = identity.real       // 1.0
    /// let imaginary = identity.imaginary  // 0.0
    /// ```
    @inlinable
    public init(integerLiteral value: Int) {
        self.init(T(value), T.zero)
    }

    /// Float literal support enabling `let z: Complex<Double> = 3.14`
    ///
    /// **Example:**
    /// ```swift
    /// let pi: Complex<Double> = 3.14159
    /// let real = pi.real       // 3.14159
    /// let imaginary = pi.imaginary  // 0.0
    /// ```
    @inlinable
    public init(floatLiteral value: Double) {
        self.init(T(value), T.zero)
    }

    // MARK: - Static Constants

    /// Additive identity: 0 + 0i
    ///
    /// **Example:**
    /// ```swift
    /// var z = Complex<Double>.zero
    /// z += Complex(3, 4)  // Now 3 + 4i
    /// let real = z.real   // 3.0
    /// ```
    @inlinable
    public static var zero: Complex<T> {
        Complex(T.zero, T.zero)
    }

    /// Multiplicative identity: 1 + 0i
    ///
    /// **Example:**
    /// ```swift
    /// let z = Complex(3.0, 4.0)
    /// let result = z * .one  // 3 + 4i (unchanged)
    /// let isUnchanged = result == z  // true
    /// ```
    @inlinable
    public static var one: Complex<T> {
        Complex(T.one, T.zero)
    }

    /// Imaginary unit where i² = -1
    ///
    /// **Example:**
    /// ```swift
    /// let i = Complex<Double>.i    // 0 + 1i
    /// let iPower2 = i * i          // -1 + 0i
    /// let iPower4 = i * i * i * i  // 1 + 0i
    /// ```
    @inlinable
    public static var i: Complex<T> {
        Complex(T.zero, T.one)
    }

    // MARK: - Computed Properties

    /// Complex conjugate: (a + bi)* = a - bi
    ///
    /// Flips the sign of the imaginary component. Essential for quantum inner products
    /// and probability calculations: ⟨ψ|φ⟩ and |ψ|² = ψ*ψ (Born rule).
    ///
    /// **Example:**
    /// ```swift
    /// let amplitude = Complex(0.6, 0.8)
    /// let conj = amplitude.conjugate      // 0.6 - 0.8i
    /// let prob = (amplitude * conj).real  // |amplitude|² = 1.0
    /// ```
    ///
    /// - Complexity: O(1)
    /// - SeeAlso: ``magnitudeSquared``
    @inlinable
    @_eagerMove
    public var conjugate: Complex<T> {
        Complex(real, -imaginary)
    }

    /// Magnitude squared: |z|² = a² + b²
    ///
    /// Efficient probability computation without square root. Uses fused multiply-add (FMA)
    /// for single rounding, maintaining precision. Directly represents Born rule probability
    /// in quantum mechanics: P(outcome) = |amplitude|².
    ///
    /// **Example:**
    /// ```swift
    /// let amplitude = Complex(0.6, 0.8)
    /// let probability = amplitude.magnitudeSquared  // 1.0
    /// let isNormalized = probability == 1.0  // true
    /// ```
    ///
    /// - Complexity: O(1)
    /// - SeeAlso: ``conjugate``, ``magnitude``
    @inlinable
    public var magnitudeSquared: T {
        T.fusedMultiplyAdd(real, real, imaginary * imaginary)
    }

    /// Magnitude (absolute value): |z| = √(a² + b²)
    ///
    /// Uses `hypot` for numerical stability, preventing overflow/underflow in intermediate
    /// calculations when real² + imaginary² exceeds representable range but result doesn't.
    ///
    /// **Example:**
    /// ```swift
    /// let z = Complex(3.0, 4.0)
    /// let r = z.magnitude         // 5.0
    /// let normalized = z / r      // Unit complex number
    /// ```
    ///
    /// - Complexity: O(1)
    /// - SeeAlso: ``magnitudeSquared``, ``phase``
    @inlinable
    @_eagerMove
    public var magnitude: T {
        T.hypotenuse(x: real, y: imaginary)
    }

    /// Phase angle (argument): arg(z) = atan2(b, a) ∈ (-π, π]
    ///
    /// Returns principal argument in radians. Combined with magnitude forms polar representation.
    /// Quantum gate phases are rotations in the complex plane: e^(iθ) = cos(θ) + i·sin(θ).
    ///
    /// **Example:**
    /// ```swift
    /// let z = Complex(1.0, 1.0)
    /// let angle = z.phase  // π/4 radians (45°)
    /// let magnitude = z.magnitude  // √2
    /// ```
    ///
    /// - Complexity: O(1)
    /// - SeeAlso: ``magnitude``, ``init(magnitude:phase:)``
    @inlinable
    @_eagerMove
    public var phase: T {
        T.arctangent(y: imaginary, x: real)
    }

    /// True if both real and imaginary components are finite (not NaN or Inf)
    ///
    /// Used for validation in quantum state construction and gate operations.
    ///
    /// **Example:**
    /// ```swift
    /// let valid = Complex(3.0, 4.0)
    /// let invalid = Complex(Double.nan, 1.0)
    /// print(valid.isFinite)    // true
    /// print(invalid.isFinite)  // false
    /// ```
    @inlinable
    public var isFinite: Bool {
        real.isFinite && imaginary.isFinite
    }

    // MARK: - Polar Initialization

    /// Creates a complex number from polar coordinates: z = r·e^(iθ)
    ///
    /// Converts polar (magnitude, phase) to cartesian (real, imaginary) using Euler's formula:
    /// r·e^(iθ) = r·cos(θ) + r·i·sin(θ). Natural representation for quantum gate rotations.
    ///
    /// - Parameters:
    ///   - magnitude: Magnitude (non-negative radius)
    ///   - phase: Phase angle in radians
    ///
    /// **Example:**
    /// ```swift
    /// let z = Complex(magnitude: 2.0, phase: .pi/3)
    /// let real = z.real        // 1.0
    /// let imaginary = z.imaginary  // √3
    /// ```
    ///
    /// - Complexity: O(1)
    /// - SeeAlso: ``magnitude``, ``phase``, ``init(phase:)``
    @inlinable
    public init(magnitude: T, phase: T) {
        self.init(magnitude * T.cosine(of: phase), magnitude * T.sine(of: phase))
    }

    // MARK: - Exponential Form

    /// Creates a unit complex number using Euler's formula: e^(iθ) = cos(θ) + i·sin(θ)
    ///
    /// Creates unit complex number (|z| = 1) on the unit circle at phase angle θ.
    /// Fundamental representation for quantum gate phases and rotations. All quantum
    /// gates with eigenvalues e^(±iθ) use this construction.
    ///
    /// - Parameter phase: Phase angle in radians
    ///
    /// **Example:**
    /// ```swift
    /// let i = Complex<Double>(phase: .pi/2)           // i
    /// let minusOne = Complex<Double>(phase: .pi)      // -1
    /// let phaseGate = Complex<Double>(phase: theta)   // e^(iθ)
    /// ```
    ///
    /// - Complexity: O(1)
    /// - SeeAlso: ``init(magnitude:phase:)``
    @inlinable
    public init(phase: T) {
        self.init(T.cosine(of: phase), T.sine(of: phase))
    }

    // MARK: - CustomStringConvertible

    /// Human-readable string representation with intelligent formatting
    ///
    /// Formats as "a", "bi", "a + bi", or "a - bi" depending on which components
    /// are significant (within epsilon threshold).
    ///
    /// **Example:**
    /// ```swift
    /// print(Complex(3, 4))    // "3.0 + 4.0i"
    /// print(Complex(5, 0))    // "5.0"
    /// print(Complex(0, -2))   // "-2.0i"
    /// ```
    public var description: String {
        let tolerance = T.epsilon

        if T.absoluteValue(of: imaginary) < tolerance {
            return "\(real)"
        }
        if T.absoluteValue(of: real) < tolerance {
            return "\(imaginary)i"
        }
        if imaginary >= 0 {
            return "\(real) + \(imaginary)i"
        } else {
            return "\(real) - \(T.absoluteValue(of: imaginary))i"
        }
    }

    // MARK: - Equatable

    /// Epsilon-quantized bucket equality for floating-point tolerance.
    ///
    /// Both components are quantized to epsilon-width bins via
    /// `Int64((value / epsilon).rounded())`; two values are equal when they fall in the
    /// same bin. Guarantees the Hashable contract: equal values always produce identical hashes.
    ///
    /// **Example:**
    /// ```swift
    /// let z1 = Complex(1.0, 0.0)
    /// let z2 = Complex(1.0 + 1e-11, 0.0)
    /// print(z1 == z2)  // true (same epsilon bucket)
    /// ```
    @_specialize(exported: true, where T == Double)
    @_specialize(exported: true, where T == Float)
    @_effects(readonly)
    @inlinable
    public static func == (lhs: Complex<T>, rhs: Complex<T>) -> Bool {
        let invEpsilon = T.one / T.epsilon
        return Int64((lhs.real * invEpsilon).rounded()) == Int64((rhs.real * invEpsilon).rounded()) &&
            Int64((lhs.imaginary * invEpsilon).rounded()) == Int64((rhs.imaginary * invEpsilon).rounded())
    }

    // MARK: - Hashable

    /// Epsilon-quantized hashing consistent with ``==(_:_:)`` bucket comparison.
    ///
    /// Both equality and hashing quantize components to the same epsilon-width
    /// bins via `Int64((component / epsilon).rounded())`, guaranteeing the
    /// Hashable contract: equal values always produce identical hashes.
    ///
    /// **Example:**
    /// ```swift
    /// var uniqueAmplitudes = Set<Complex<Double>>()
    /// uniqueAmplitudes.insert(Complex(0.707, 0.707))
    /// let count = uniqueAmplitudes.count  // 1
    /// ```
    @inlinable
    public func hash(into hasher: inout Hasher) {
        let invEpsilon = T.one / T.epsilon
        let realBucket = Int64((real * invEpsilon).rounded())
        let imaginaryBucket = Int64((imaginary * invEpsilon).rounded())
        hasher.combine(realBucket)
        hasher.combine(imaginaryBucket)
    }
}

// MARK: - Arithmetic Operators

/// Complex addition: (a+bi) + (c+di) = (a+c) + (b+d)i
/// - Complexity: O(1)
///
/// **Example:**
/// ```swift
/// let a = Complex(1.0, 2.0)
/// let b = Complex(3.0, 4.0)
/// let sum = a + b  // 4.0 + 6.0i
/// ```
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_effects(readonly)
@inlinable
public func + <T>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    Complex(lhs.real + rhs.real, lhs.imaginary + rhs.imaginary)
}

/// Complex subtraction: (a+bi) - (c+di) = (a-c) + (b-d)i
/// - Complexity: O(1)
///
/// **Example:**
/// ```swift
/// let a = Complex(3.0, 4.0)
/// let b = Complex(1.0, 2.0)
/// let diff = a - b  // 2.0 + 2.0i
/// ```
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_effects(readonly)
@inlinable
public func - <T>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    Complex(lhs.real - rhs.real, lhs.imaginary - rhs.imaginary)
}

/// Complex multiplication: (a+bi)·(c+di) = (ac-bd) + (ad+bc)i
/// - Complexity: O(1)
///
/// **Example:**
/// ```swift
/// let a = Complex(1.0, 2.0)
/// let b = Complex(3.0, 4.0)
/// let product = a * b  // -5.0 + 10.0i
/// ```
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_optimize(speed)
@_effects(readonly)
@inlinable
public func * <T>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    let real = T.fusedMultiplyAdd(lhs.real, rhs.real, -(lhs.imaginary * rhs.imaginary))
    let imaginary = T.fusedMultiplyAdd(lhs.real, rhs.imaginary, lhs.imaginary * rhs.real)
    return Complex(real, imaginary)
}

/// Complex division: (a+bi) / (c+di) = [(a+bi)·(c-di)] / (c²+d²)
/// - Complexity: O(1)
/// - Precondition: `rhs.magnitudeSquared > 0` (division by zero not allowed)
///
/// **Example:**
/// ```swift
/// let a = Complex(2.0, 4.0)
/// let b = Complex(1.0, 1.0)
/// let quotient = a / b  // 3.0 + 1.0i
/// ```
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_optimize(speed)
@_effects(readonly)
@inlinable
public func / <T>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    let denominator: T = rhs.magnitudeSquared
    ValidationUtilities.validateComplexDivisionByZero(denominator, threshold: T.divisionThreshold)

    let invDenom = T.one / denominator
    let real = T.fusedMultiplyAdd(lhs.real, rhs.real, lhs.imaginary * rhs.imaginary) * invDenom
    let imaginary = T.fusedMultiplyAdd(lhs.imaginary, rhs.real, -lhs.real * rhs.imaginary) * invDenom
    return Complex(real, imaginary)
}

/// Complex negation: -(a+bi) = -a - bi
/// - Complexity: O(1)
///
/// **Example:**
/// ```swift
/// let z = Complex(3.0, 4.0)
/// let neg = -z  // -3.0 - 4.0i
/// print(neg.real)  // -3.0
/// ```
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_effects(readonly)
@inlinable
public prefix func - <T>(z: Complex<T>) -> Complex<T> {
    Complex(-z.real, -z.imaginary)
}

// MARK: - Scalar Operations

/// Scalar multiplication: k·(a+bi) = ka + kbi
/// - Complexity: O(1)
///
/// **Example:**
/// ```swift
/// let z = Complex(1.0, 2.0)
/// let scaled = 3.0 * z  // 3.0 + 6.0i
/// print(scaled.imaginary)  // 6.0
/// ```
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_effects(readonly)
@inlinable
public func * <T>(scalar: T, z: Complex<T>) -> Complex<T> {
    Complex(scalar * z.real, scalar * z.imaginary)
}

/// Scalar multiplication: (a+bi)·k = ka + kbi
/// - Complexity: O(1)
///
/// **Example:**
/// ```swift
/// let z = Complex(1.0, 2.0)
/// let scaled = z * 3.0  // 3.0 + 6.0i
/// print(scaled.real)  // 3.0
/// ```
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_effects(readonly)
@inlinable
public func * <T>(z: Complex<T>, scalar: T) -> Complex<T> {
    Complex(z.real * scalar, z.imaginary * scalar)
}

/// Scalar division: (a+bi)/k = a/k + (b/k)i
/// - Complexity: O(1)
///
/// **Example:**
/// ```swift
/// let z = Complex(4.0, 6.0)
/// let halved = z / 2.0  // 2.0 + 3.0i
/// print(halved.real)  // 2.0
/// ```
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_effects(readonly)
@inlinable
public func / <T>(z: Complex<T>, scalar: T) -> Complex<T> {
    let inv = 1 / scalar
    return Complex(z.real * inv, z.imaginary * inv)
}

// MARK: - Compound Assignment Operators

/// Multiplies a complex number by another complex number in place.
///
/// **Example:**
/// ```swift
/// var z = Complex(1.0, 2.0)
/// z *= Complex(3.0, 4.0)
/// print(z)  // -5.0 + 10.0i
/// ```
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_optimize(speed)
@inlinable
public func *= <T>(lhs: inout Complex<T>, rhs: Complex<T>) {
    lhs = lhs * rhs
}

/// Divides a complex number by another complex number in place.
///
/// **Example:**
/// ```swift
/// var z = Complex(2.0, 4.0)
/// z /= Complex(1.0, 1.0)
/// print(z)  // 3.0 + 1.0i
/// ```
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_optimize(speed)
@inlinable
public func /= <T>(lhs: inout Complex<T>, rhs: Complex<T>) {
    lhs = lhs / rhs
}

/// Multiplies a complex number by a scalar in place.
///
/// **Example:**
/// ```swift
/// var z = Complex(1.0, 2.0)
/// z *= 3.0
/// print(z)  // 3.0 + 6.0i
/// ```
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@inlinable
public func *= <T>(lhs: inout Complex<T>, scalar: T) {
    lhs = lhs * scalar
}

/// Divides a complex number by a scalar in place.
///
/// **Example:**
/// ```swift
/// var z = Complex(4.0, 6.0)
/// z /= 2.0
/// print(z)  // 2.0 + 3.0i
/// ```
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@inlinable
public func /= <T>(lhs: inout Complex<T>, scalar: T) {
    lhs = lhs / scalar
}
