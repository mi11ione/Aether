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
    @inlinable public static var zero: Double { 0.0 }
    @inlinable public static var one: Double { 1.0 }
    @inlinable public static var epsilon: Double { 1e-10 }
    @inlinable public static var divisionThreshold: Double { 1e-15 }

    @inlinable public static func squareRoot(of value: Double) -> Double { Foundation.sqrt(value) }
    @inlinable public static func cosine(of value: Double) -> Double { Foundation.cos(value) }
    @inlinable public static func sine(of value: Double) -> Double { Foundation.sin(value) }
    @inlinable public static func arctangent(y: Double, x: Double) -> Double { Foundation.atan2(y, x) }
    @inlinable public static func absoluteValue(of value: Double) -> Double {
        Double(bitPattern: value.bitPattern & 0x7FFF_FFFF_FFFF_FFFF)
    }

    @inlinable public static func hypotenuse(x: Double, y: Double) -> Double { Foundation.hypot(x, y) }
    @inlinable public static func fusedMultiplyAdd(_ a: Double, _ b: Double, _ c: Double) -> Double { Foundation.fma(a, b, c) }
}

extension Float: ComplexScalar {
    @inlinable public static var zero: Float { 0.0 }
    @inlinable public static var one: Float { 1.0 }
    @inlinable public static var epsilon: Float { 1e-6 }
    @inlinable public static var divisionThreshold: Float { 1e-10 }

    @inlinable public static func squareRoot(of value: Float) -> Float { Foundation.sqrt(value) }
    @inlinable public static func cosine(of value: Float) -> Float { Foundation.cos(value) }
    @inlinable public static func sine(of value: Float) -> Float { Foundation.sin(value) }
    @inlinable public static func arctangent(y: Float, x: Float) -> Float { Foundation.atan2(y, x) }
    @inlinable public static func absoluteValue(of value: Float) -> Float {
        Float(bitPattern: value.bitPattern & 0x7FFF_FFFF)
    }

    @inlinable public static func hypotenuse(x: Float, y: Float) -> Float { Foundation.hypot(x, y) }
    @inlinable public static func fusedMultiplyAdd(_ a: Float, _ b: Float, _ c: Float) -> Float { Foundation.fma(a, b, c) }
}

/// Generic complex number type for quantum computing with type-safe arithmetic.
///
/// Represents complex numbers z = a + bi where i² = -1, supporting Double and Float precision.
/// Provides complete arithmetic operations, polar/cartesian conversions, and Euler's formula
/// for phase representation.
///
/// **Quantum computing context**: Complex amplitudes encode quantum state probabilities where
/// |ψ|² represents measurement probability (Born rule). Inner products use complex conjugation.
///
/// **Precision**: Use `Complex<Double>` for quantum state amplitudes (default). Use `Complex<Float>`
/// for GPU computations where 7 decimal digits suffice. Equality uses epsilon-based comparison
/// (1e-10 for Double, 1e-6 for Float) suitable for floating-point quantum computations.
///
/// **Performance**: All operations are O(1). Optimized with fused multiply-add (FMA) for
/// ``magnitudeSquared`` and division. Specialized for Double and Float at compile time.
///
/// **Example**:
/// ```swift
/// // Cartesian construction
/// let amplitude = Complex(0.6, 0.8)
/// let probability = amplitude.magnitudeSquared  // 1.0 (normalized)
///
/// // Polar construction for phase gates
/// let phaseGate = Complex(magnitude: 1.0, phase: .pi/4)
///
/// // Euler's formula for unit phases
/// let i = Complex<Double>(phase: .pi/2)  // i
///
/// // Arithmetic
/// let product = amplitude * amplitude.conjugate
/// print(product.real)  // 1.0 (Born rule probability)
/// ```
@frozen
public struct Complex<T: ComplexScalar>: Equatable, Hashable, CustomStringConvertible, ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral, AdditiveArithmetic, Sendable {
    // MARK: - Properties

    public let real: T
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
    /// **Example**:
    /// ```swift
    /// let amplitude = Complex(0.6, 0.8)
    /// let prob = amplitude.magnitudeSquared  // 1.0
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
    /// **Example**:
    /// ```swift
    /// let eigenvalue = Complex(1.0)  // 1 + 0i
    /// ```
    @inlinable
    public init(_ real: T) {
        self.real = real
        imaginary = T.zero
    }

    /// Integer literal support enabling `let z: Complex<Double> = 5`
    ///
    /// **Example**:
    /// ```swift
    /// let identity: Complex<Double> = 1
    /// ```
    @inlinable
    public init(integerLiteral value: Int) {
        real = T(value)
        imaginary = T.zero
    }

    /// Float literal support enabling `let z: Complex<Double> = 3.14`
    ///
    /// **Example**:
    /// ```swift
    /// let pi: Complex<Double> = 3.14159
    /// ```
    @inlinable
    public init(floatLiteral value: Double) {
        real = T(value)
        imaginary = T.zero
    }

    // MARK: - Static Constants

    /// Additive identity: 0 + 0i
    ///
    /// **Example**:
    /// ```swift
    /// var z = Complex<Double>.zero
    /// z += Complex(3, 4)  // Now 3 + 4i
    /// ```
    @inlinable
    public static var zero: Complex<T> {
        Complex(T.zero, T.zero)
    }

    /// Multiplicative identity: 1 + 0i
    ///
    /// **Example**:
    /// ```swift
    /// let z = Complex(3.0, 4.0)
    /// let result = z * .one  // 3 + 4i (unchanged)
    /// ```
    @inlinable
    public static var one: Complex<T> {
        Complex(T.one, T.zero)
    }

    /// Imaginary unit where i² = -1
    ///
    /// **Example**:
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
    /// **Example**:
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
    /// **Example**:
    /// ```swift
    /// let amplitude = Complex(0.6, 0.8)
    /// let probability = amplitude.magnitudeSquared  // 1.0
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
    /// **Example**:
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
    /// **Example**:
    /// ```swift
    /// let z = Complex(1.0, 1.0)
    /// let angle = z.phase  // π/4 radians (45°)
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
    /// **Example**:
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
    /// **Example**:
    /// ```swift
    /// let z = Complex(magnitude: 2.0, phase: .pi/3)
    /// // Creates: 2·e^(iπ/3) = 1 + √3i
    /// ```
    ///
    /// - Complexity: O(1)
    /// - SeeAlso: ``magnitude``, ``phase``, ``init(phase:)``
    @inlinable
    public init(magnitude: T, phase: T) {
        real = magnitude * T.cosine(of: phase)
        imaginary = magnitude * T.sine(of: phase)
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
    /// **Example**:
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
        real = T.cosine(of: phase)
        imaginary = T.sine(of: phase)
    }

    // MARK: - CustomStringConvertible

    /// Human-readable string representation with intelligent formatting
    ///
    /// Formats as "a", "bi", "a + bi", or "a - bi" depending on which components
    /// are significant (within epsilon threshold).
    ///
    /// **Example**:
    /// ```swift
    /// print(Complex(3, 4))    // "3.0 + 4.0i"
    /// print(Complex(5, 0))    // "5.0"
    /// print(Complex(0, -2))   // "-2.0i"
    /// ```
    public var description: String {
        let epsilon = T.epsilon

        if T.absoluteValue(of: imaginary) < epsilon {
            return "\(real)"
        }
        if T.absoluteValue(of: real) < epsilon {
            return "\(imaginary)i"
        }
        if imaginary >= 0 {
            return "\(real) + \(imaginary)i"
        } else {
            return "\(real) - \(T.absoluteValue(of: imaginary))i"
        }
    }

    // MARK: - Equatable

    /// Epsilon-based equality for floating-point tolerance
    ///
    /// Two complex numbers are equal if both real and imaginary components differ by less
    /// than epsilon (1e-10 for Double, 1e-6 for Float). Essential for quantum state comparisons
    /// where exact floating-point equality is unreliable.
    ///
    /// **Example**:
    /// ```swift
    /// let z1 = Complex(1.0, 0.0)
    /// let z2 = Complex(1.0 + 1e-11, 0.0)
    /// print(z1 == z2)  // true (within epsilon)
    /// ```
    public static func == (lhs: Complex<T>, rhs: Complex<T>) -> Bool {
        let epsilon = T.epsilon
        return T.absoluteValue(of: lhs.real - rhs.real) < epsilon &&
            T.absoluteValue(of: lhs.imaginary - rhs.imaginary) < epsilon
    }

    // MARK: - Hashable

    /// Epsilon-quantized hashing for Set and Dictionary compatibility
    ///
    /// Rounds components to epsilon boundaries before hashing, ensuring approximately
    /// equal values (per ``==(_:_:)`` definition) hash to same bucket. Enables use in Set
    /// and Dictionary collections without hash/equality contract violations.
    ///
    /// **Example**:
    /// ```swift
    /// var uniqueAmplitudes = Set<Complex<Double>>()
    /// uniqueAmplitudes.insert(Complex(0.707, 0.707))
    /// ```
    public func hash(into hasher: inout Hasher) {
        let realBucket = Int64((real / T.epsilon).rounded())
        let imagBucket = Int64((imaginary / T.epsilon).rounded())
        hasher.combine(realBucket)
        hasher.combine(imagBucket)
    }
}

// MARK: - Arithmetic Operators

/// Complex addition: (a+bi) + (c+di) = (a+c) + (b+d)i
/// - Complexity: O(1)
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_effects(readonly)
@inlinable
public func + <T>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    Complex(lhs.real + rhs.real, lhs.imaginary + rhs.imaginary)
}

/// Complex subtraction: (a+bi) - (c+di) = (a-c) + (b-d)i
/// - Complexity: O(1)
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_effects(readonly)
@inlinable
public func - <T>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    Complex(lhs.real - rhs.real, lhs.imaginary - rhs.imaginary)
}

/// Complex multiplication: (a+bi)·(c+di) = (ac-bd) + (ad+bc)i
/// - Complexity: O(1)
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

/// Complex division: (a+bi) / (c+di) = [(a+bi)·(c-di)] / (c²+d²)
/// - Complexity: O(1)
/// - Precondition: `rhs.magnitudeSquared > 0` (division by zero not allowed)
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_optimize(speed)
@inlinable
public func / <T>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    let denominator: T = rhs.magnitudeSquared
    ValidationUtilities.validateComplexDivisionByZero(denominator, threshold: T.divisionThreshold)

    let invDenom = T.one / denominator
    let real = T.fusedMultiplyAdd(lhs.real, rhs.real, lhs.imaginary * rhs.imaginary) * invDenom
    let imag = T.fusedMultiplyAdd(lhs.imaginary, rhs.real, -lhs.real * rhs.imaginary) * invDenom
    return Complex(real, imag)
}

/// Complex negation: -(a+bi) = -a - bi
/// - Complexity: O(1)
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
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_effects(readonly)
@inlinable
public func * <T>(scalar: T, z: Complex<T>) -> Complex<T> {
    Complex(scalar * z.real, scalar * z.imaginary)
}

/// Scalar multiplication: (a+bi)·k = ka + kbi
/// - Complexity: O(1)
@_specialize(exported: true, where T == Double)
@_specialize(exported: true, where T == Float)
@_effects(readonly)
@inlinable
public func * <T>(z: Complex<T>, scalar: T) -> Complex<T> {
    Complex(z.real * scalar, z.imaginary * scalar)
}

/// Scalar division: (a+bi)/k = a/k + (b/k)i
/// - Complexity: O(1)
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
