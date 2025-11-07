// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

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
struct Complex<T: BinaryFloatingPoint & Sendable>: Equatable, Hashable, CustomStringConvertible, ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral, AdditiveArithmetic, Sendable {
    // MARK: - Properties

    let real: T
    let imaginary: T

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
    init(_ real: T, _ imaginary: T) {
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
    init(_ real: T) {
        self.real = real
        if T.self == Double.self {
            imaginary = 0.0 as! T
        } else {
            imaginary = Float(0.0) as! T
        }
    }

    init(integerLiteral value: Int) {
        if T.self == Double.self {
            real = Double(value) as! T
            imaginary = 0.0 as! T
        } else if T.self == Float.self {
            real = Float(value) as! T
            imaginary = Float(0.0) as! T
        } else {
            preconditionFailure("Unsupported FloatingPoint type")
        }
    }

    init(floatLiteral value: Double) {
        if T.self == Double.self {
            real = value as! T
            imaginary = 0.0 as! T
        } else if T.self == Float.self {
            real = Float(value) as! T
            imaginary = Float(0.0) as! T
        } else {
            preconditionFailure("Unsupported FloatingPoint type")
        }
    }

    // MARK: - Static Constants

    static var zero: Complex<T> {
        let z: T = if T.self == Double.self {
            0.0 as! T
        } else {
            Float(0.0) as! T
        }
        return Complex(z, z)
    }

    static var one: Complex<T> {
        let o: T = if T.self == Double.self {
            1.0 as! T
        } else {
            Float(1.0) as! T
        }
        let z: T = if T.self == Double.self {
            0.0 as! T
        } else {
            Float(0.0) as! T
        }
        return Complex(o, z)
    }

    /// Imaginary unit where i² = -1
    ///
    /// Example:
    /// ```swift
    /// let i = Complex<Double>.i           // 0 + 1i
    /// let iPower2 = i * i                 // -1 + 0i
    /// let iPower4 = i * i * i * i         // 1 + 0i
    /// ```
    static var i: Complex<T> {
        let o: T = if T.self == Double.self {
            1.0 as! T
        } else {
            Float(1.0) as! T
        }
        let z: T = if T.self == Double.self {
            0.0 as! T
        } else {
            Float(0.0) as! T
        }
        return Complex(z, o)
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
    var conjugate: Complex<T> {
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
    var magnitudeSquared: T {
        real * real + imaginary * imaginary
    }

    /// Magnitude: |z| = √(a² + b²)
    ///
    /// Example:
    /// ```swift
    /// let z = Complex(3.0, 4.0)
    /// let r = z.magnitude                 // 5.0
    /// ```
    var magnitude: T {
        sqrt(magnitudeSquared)
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
    var phase: T {
        atan2(imaginary, real)
    }

    var isFinite: Bool {
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
    static func fromPolar(r: T, theta: T) -> Complex<T> {
        Complex(r * cos(theta), r * sin(theta))
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
    func toPolar() -> (magnitude: T, phase: T) {
        (magnitude, phase)
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
    static func exp(_ theta: T) -> Complex<T> {
        Complex(cos(theta), sin(theta))
    }

    // MARK: - CustomStringConvertible

    var description: String {
        let epsilon: T = if T.self == Double.self {
            1e-10 as! T
        } else {
            Float(1e-6) as! T
        }

        if abs(imaginary) < epsilon {
            return "\(real)"
        }
        if abs(real) < epsilon {
            return "\(imaginary)i"
        }
        if imaginary >= 0 {
            return "\(real) + \(imaginary)i"
        } else {
            return "\(real) - \(abs(imaginary))i"
        }
    }

    // MARK: - Equatable

    static func == (lhs: Complex<T>, rhs: Complex<T>) -> Bool {
        let epsilon: T = if T.self == Double.self {
            1e-10 as! T
        } else {
            Float(1e-6) as! T
        }
        return abs(lhs.real - rhs.real) < epsilon &&
            abs(lhs.imaginary - rhs.imaginary) < epsilon
    }
}

// MARK: - Arithmetic Operators

func + <T>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    Complex(lhs.real + rhs.real, lhs.imaginary + rhs.imaginary)
}

func - <T>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    Complex(lhs.real - rhs.real, lhs.imaginary - rhs.imaginary)
}

func * <T>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    let real = lhs.real * rhs.real - lhs.imaginary * rhs.imaginary
    let imag = lhs.real * rhs.imaginary + lhs.imaginary * rhs.real
    return Complex(real, imag)
}

func / <T>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    let denominator = rhs.magnitudeSquared
    let minThreshold: T = if T.self == Double.self {
        1e-15 as! T
    } else {
        Float(1e-10) as! T
    }

    guard denominator > minThreshold else {
        let nan: T = if T.self == Double.self {
            Double.nan as! T
        } else {
            Float.nan as! T
        }
        return Complex(nan, nan)
    }
    let real = (lhs.real * rhs.real + lhs.imaginary * rhs.imaginary) / denominator
    let imag = (lhs.imaginary * rhs.real - lhs.real * rhs.imaginary) / denominator
    return Complex(real, imag)
}

prefix func - <T>(z: Complex<T>) -> Complex<T> {
    Complex(-z.real, -z.imaginary)
}

// MARK: - Scalar Operations

func * <T>(scalar: T, z: Complex<T>) -> Complex<T> {
    Complex(scalar * z.real, scalar * z.imaginary)
}

func * <T>(z: Complex<T>, scalar: T) -> Complex<T> {
    Complex(z.real * scalar, z.imaginary * scalar)
}

func / <T>(z: Complex<T>, scalar: T) -> Complex<T> {
    Complex(z.real / scalar, z.imaginary / scalar)
}

// MARK: - Helper Functions

private func abs<T: FloatingPoint>(_ value: T) -> T {
    value < 0 ? -value : value
}

private func sqrt<T: FloatingPoint>(_ value: T) -> T where T: FloatingPoint {
    if let d = value as? Double {
        return Foundation.sqrt(d) as! T
    } else if let f = value as? Float {
        return Foundation.sqrt(f) as! T
    }
    preconditionFailure("Unsupported FloatingPoint type for sqrt")
}

private func cos<T: FloatingPoint>(_ value: T) -> T {
    if let d = value as? Double {
        return cos(d) as! T
    } else if let f = value as? Float {
        return cos(f) as! T
    }
    preconditionFailure("Unsupported FloatingPoint type for cos")
}

private func sin<T: FloatingPoint>(_ value: T) -> T {
    if let d = value as? Double {
        return sin(d) as! T
    } else if let f = value as? Float {
        return sin(f) as! T
    }
    preconditionFailure("Unsupported FloatingPoint type for sin")
}

private func atan2<T: FloatingPoint>(_ y: T, _ x: T) -> T {
    if let dy = y as? Double, let dx = x as? Double {
        return atan2(dy, dx) as! T
    } else if let fy = y as? Float, let fx = x as? Float {
        return atan2(fy, fx) as! T
    }
    preconditionFailure("Unsupported FloatingPoint type for atan2")
}

// MARK: - Type Aliases

typealias ComplexDouble = Complex<Double>
typealias ComplexFloat = Complex<Float>
