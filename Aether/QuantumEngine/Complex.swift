// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Generic complex number type supporting Float and Double precision.
/// Implements complete complex arithmetic including polar conversions and exponential form.
/// Type constraint ensures only Double and Float are supported for mathematical operations.
struct Complex<T: BinaryFloatingPoint & Sendable>: Equatable, Hashable, CustomStringConvertible, ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral, AdditiveArithmetic, Sendable {
    // MARK: - Properties

    let real: T
    let imaginary: T

    // MARK: - Initialization

    /// Create a complex number
    /// - Parameters:
    ///   - real: Real component
    ///   - imaginary: Imaginary component
    init(_ real: T, _ imaginary: T) {
        self.real = real
        self.imaginary = imaginary
    }

    /// Create a complex number from a real number (imaginary part = 0)
    /// - Parameter real: Real component
    init(_ real: T) {
        self.real = real
        if T.self == Double.self {
            imaginary = 0.0 as! T
        } else {
            imaginary = Float(0.0) as! T
        }
    }

    /// Create a complex number from an integer literal (ExpressibleByIntegerLiteral conformance)
    /// - Parameter value: Integer value (imaginary part = 0)
    init(integerLiteral value: Int) {
        // For generic FloatingPoint, we need explicit conversion
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

    /// Create a complex number from a float literal (ExpressibleByFloatLiteral conformance)
    /// - Parameter value: Float value (imaginary part = 0)
    init(floatLiteral value: Double) {
        // For generic FloatingPoint, we need explicit conversion
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

    /// Complex zero (0 + 0i)
    static var zero: Complex<T> {
        let z: T = if T.self == Double.self {
            0.0 as! T
        } else {
            Float(0.0) as! T
        }
        return Complex(z, z)
    }

    /// Complex one (1 + 0i)
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

    /// Imaginary unit (0 + 1i), where i² = -1
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
    var conjugate: Complex<T> {
        Complex(real, -imaginary)
    }

    /// Magnitude squared: |z|² = a² + b² (optimized, no sqrt)
    nonisolated var magnitudeSquared: T {
        real * real + imaginary * imaginary
    }

    /// Magnitude: |z| = √(a² + b²)
    var magnitude: T {
        sqrt(magnitudeSquared)
    }

    /// Phase/Argument: arg(z) = atan2(imaginary, real) ∈ (-π, π]
    /// - Returns: Principal argument in radians
    var phase: T {
        atan2(imaginary, real)
    }

    var isFinite: Bool {
        real.isFinite && imaginary.isFinite
    }

    // MARK: - Polar Conversions

    /// Create complex number from polar coordinates
    /// - Parameters:
    ///   - r: Magnitude
    ///   - theta: Phase angle
    static func fromPolar(r: T, theta: T) -> Complex<T> {
        Complex(r * cos(theta), r * sin(theta))
    }

    /// Convert to polar coordinates
    /// - Returns: Tuple of (magnitude, phase)
    func toPolar() -> (magnitude: T, phase: T) {
        (magnitude, phase)
    }

    // MARK: - Exponential Form

    /// Euler's formula: e^(iθ) = cos(θ) + i·sin(θ)
    static func exp(_ theta: T) -> Complex<T> {
        Complex(cos(theta), sin(theta))
    }

    // MARK: - CustomStringConvertible

    /// String representation of the complex number
    /// - Returns: Formatted string like "a + bi", "a - bi", or "bi"
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

/// Addition: (a + bi) + (c + di) = (a+c) + (b+d)i
func + <T>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    Complex(lhs.real + rhs.real, lhs.imaginary + rhs.imaginary)
}

/// Subtraction: (a + bi) - (c + di) = (a-c) + (b-d)i
func - <T>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    Complex(lhs.real - rhs.real, lhs.imaginary - rhs.imaginary)
}

/// Multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
func * <T>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    let real = lhs.real * rhs.real - lhs.imaginary * rhs.imaginary
    let imag = lhs.real * rhs.imaginary + lhs.imaginary * rhs.real
    return Complex(real, imag)
}

/// Division: (a + bi) / (c + di) = [(ac + bd) + (bc - ad)i] / (c² + d²)
func / <T>(lhs: Complex<T>, rhs: Complex<T>) -> Complex<T> {
    let denominator = rhs.magnitudeSquared
    let minThreshold: T = if T.self == Double.self {
        1e-15 as! T
    } else {
        Float(1e-10) as! T
    }

    guard denominator > minThreshold else {
        // Division by near-zero - return NaN complex number
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

/// Negation: -(a + bi) = (-a) + (-b)i
prefix func - <T>(z: Complex<T>) -> Complex<T> {
    Complex(-z.real, -z.imaginary)
}

// MARK: - Scalar Operations

/// Scalar multiplication: c · z
func * <T>(scalar: T, z: Complex<T>) -> Complex<T> {
    Complex(scalar * z.real, scalar * z.imaginary)
}

/// Scalar multiplication: z · c
func * <T>(z: Complex<T>, scalar: T) -> Complex<T> {
    Complex(z.real * scalar, z.imaginary * scalar)
}

/// Scalar division: z / c
func / <T>(z: Complex<T>, scalar: T) -> Complex<T> {
    Complex(z.real / scalar, z.imaginary / scalar)
}

// MARK: - Helper Functions

/// Absolute value function for generic FloatingPoint
private func abs<T: FloatingPoint>(_ value: T) -> T {
    value < 0 ? -value : value
}

/// Square root function wrapper for generic FloatingPoint types
/// - Parameter value: Input value
/// - Returns: Square root of the input
/// - Note: Uses Foundation's sqrt function with type-specific implementations
private func sqrt<T: FloatingPoint>(_ value: T) -> T where T: FloatingPoint {
    // For generic T, we need type-specific implementations
    if let d = value as? Double {
        return Foundation.sqrt(d) as! T
    } else if let f = value as? Float {
        return Foundation.sqrt(f) as! T
    }
    preconditionFailure("Unsupported FloatingPoint type for sqrt")
}

/// Cosine function wrapper for generic FloatingPoint types
/// - Parameter value: Angle in radians
/// - Returns: Cosine of the angle
private func cos<T: FloatingPoint>(_ value: T) -> T {
    if let d = value as? Double {
        return cos(d) as! T
    } else if let f = value as? Float {
        return cos(f) as! T
    }
    preconditionFailure("Unsupported FloatingPoint type for cos")
}

/// Sine function wrapper for generic FloatingPoint types
/// - Parameter value: Angle in radians
/// - Returns: Sine of the angle
private func sin<T: FloatingPoint>(_ value: T) -> T {
    if let d = value as? Double {
        return sin(d) as! T
    } else if let f = value as? Float {
        return sin(f) as! T
    }
    preconditionFailure("Unsupported FloatingPoint type for sin")
}

/// Arctangent function wrapper for generic FloatingPoint types
/// - Parameters:
///   - y: Y coordinate
///   - x: X coordinate
/// - Returns: Principal arctangent of y/x in radians
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
