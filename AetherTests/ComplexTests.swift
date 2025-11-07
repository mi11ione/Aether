// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for Complex<T> arithmetic operations.
/// Validates basic complex number algebra: addition, subtraction,
/// multiplication, division, and negation.
@Suite("Complex Number Arithmetic")
struct ComplexArithmeticTests {
    @Test("Addition: (2+3i) + (4+5i) = (6+8i)")
    func addition() {
        let a = Complex(2.0, 3.0)
        let b = Complex(4.0, 5.0)
        let result = a + b
        #expect(result == Complex(6.0, 8.0))
    }

    @Test("Subtraction: (5+7i) - (2+3i) = (3+4i)")
    func subtraction() {
        let a = Complex(5.0, 7.0)
        let b = Complex(2.0, 3.0)
        let result = a - b
        #expect(result == Complex(3.0, 4.0))
    }

    @Test("Multiplication: (2+3i)(4+5i) = -7+22i)")
    func multiplication() {
        let a = Complex(2.0, 3.0)
        let b = Complex(4.0, 5.0)
        let result = a * b
        #expect(result == Complex(-7.0, 22.0))
    }

    @Test("Division: (2+3i)/(4+5i) = (23+2i)/41")
    func division() {
        let a = Complex(2.0, 3.0)
        let b = Complex(4.0, 5.0)
        let result = a / b
        let expected = Complex(23.0 / 41.0, 2.0 / 41.0)
        #expect(abs(result.real - expected.real) < 1e-10)
        #expect(abs(result.imaginary - expected.imaginary) < 1e-10)
    }

    @Test("Negation: -(2+3i) = (-2-3i)")
    func negation() {
        let z = Complex(2.0, 3.0)
        let result = -z
        #expect(result == Complex(-2.0, -3.0))
    }
}

/// Test suite for fundamental complex number properties.
/// Validates conjugate, magnitude, phase, and mathematical identities
/// essential for quantum computing applications.
@Suite("Complex Number Properties")
struct ComplexPropertiesTests {
    @Test("i² = -1 (fundamental imaginary unit property)")
    func iSquaredEqualsMinusOne() {
        let i = Complex<Double>.i
        let result = i * i
        #expect(result == Complex(-1.0, 0.0))
    }

    @Test("Conjugate: (2+3i)* = (2-3i)")
    func conjugate() {
        let z = Complex(2.0, 3.0)
        let conjugate = z.conjugate
        #expect(conjugate == Complex(2.0, -3.0))
    }

    @Test("Conjugate property: z · z* = |z|²")
    func conjugateProperty() {
        let z = Complex(3.0, 4.0)
        let product = z * z.conjugate
        #expect(abs(product.real - z.magnitudeSquared) < 1e-10)
        #expect(abs(product.imaginary) < 1e-10)
    }

    @Test("Magnitude: |3+4i| = 5")
    func magnitude() {
        let z = Complex(3.0, 4.0)
        #expect(abs(z.magnitude - 5.0) < 1e-10)
    }

    @Test("Magnitude squared: |3+4i|² = 25")
    func magnitudeSquared() {
        let z = Complex(3.0, 4.0)
        #expect(abs(z.magnitudeSquared - 25.0) < 1e-10)
    }

    @Test("Phase: arg(1+i) = π/4")
    func phase() {
        let z = Complex(1.0, 1.0)
        #expect(abs(z.phase - .pi / 4.0) < 1e-10)
    }
}

/// Test suite for special complex values and identity constants.
/// Validates zero, one, and imaginary unit behavior in arithmetic operations.
@Suite("Special Values and Constants")
struct ComplexSpecialValuesTests {
    @Test("Zero operations: z + 0 = z, z - 0 = z, z · 0 = 0")
    func zeroOperations() {
        let zero = Complex<Double>.zero
        let z = Complex(3.0, 4.0)

        #expect(z + zero == z)
        #expect(z - zero == z)
        #expect(z * zero == zero)
    }

    @Test("One operations: z · 1 = z, z / 1 = z")
    func oneOperations() {
        let one = Complex<Double>.one
        let z = Complex(3.0, 4.0)

        #expect(z * one == z)
        #expect(z / one == z)
    }

    @Test("Imaginary unit: i = (0, 1)")
    func imaginaryUnit() {
        let i = Complex<Double>.i
        #expect(abs(i.real) < 1e-10)
        #expect(abs(i.imaginary - 1.0) < 1e-10)
    }
}

/// Test suite for Euler's identity and exponential form.
/// Validates e^(iθ) = cos(θ) + i·sin(θ) and related identities
/// critical for phase operations in quantum gates.
@Suite("Euler's Identity and Exponential Form")
struct ComplexEulerIdentityTests {
    @Test("Euler's identity: e^(iπ) + 1 = 0")
    func eulerIdentity() {
        let result = Complex<Double>.exp(.pi) + Complex.one
        #expect(abs(result.real) < 1e-10)
        #expect(abs(result.imaginary) < 1e-10)
    }

    @Test("Exponential at zero: e^(i0) = 1")
    func expZero() {
        let result = Complex<Double>.exp(0)
        #expect(result == Complex.one)
    }

    @Test("Exponential at π/2: e^(iπ/2) = i")
    func expPiOverTwo() {
        let result = Complex<Double>.exp(.pi / 2.0)
        #expect(abs(result.real) < 1e-10)
        #expect(abs(result.imaginary - 1.0) < 1e-10)
    }
}

/// Test suite for polar coordinate conversions.
/// Validates r·e^(iθ) ↔ (r, θ) round-trip conversions
/// used in phase visualization and quantum state analysis.
@Suite("Polar Coordinate Conversions")
struct ComplexPolarTests {
    @Test("Polar conversion round-trip preserves value")
    func polarConversionRoundTrip() {
        let z = Complex(3.0, 4.0)
        let (r, theta) = z.toPolar()
        let reconstructed = Complex<Double>.fromPolar(r: r, theta: theta)

        #expect(abs(reconstructed.real - z.real) < 1e-10)
        #expect(abs(reconstructed.imaginary - z.imaginary) < 1e-10)
    }
}

/// Test suite for mathematical identities and algebraic properties.
/// Validates commutativity, associativity, and other fundamental
/// algebraic properties required for numerical stability.
@Suite("Mathematical Identities")
struct ComplexMathematicalIdentitiesTests {
    @Test("Addition is commutative: a + b = b + a")
    func commutativityAddition() {
        let a = Complex(2.0, 3.0)
        let b = Complex(4.0, 5.0)
        #expect(a + b == b + a)
    }

    @Test("Multiplication is commutative: a · b = b · a")
    func commutativityMultiplication() {
        let a = Complex(2.0, 3.0)
        let b = Complex(4.0, 5.0)
        #expect(a * b == b * a)
    }

    @Test("Addition is associative: (a + b) + c = a + (b + c)")
    func associativityAddition() {
        let a = Complex(1.0, 2.0)
        let b = Complex(3.0, 4.0)
        let c = Complex(5.0, 6.0)
        #expect((a + b) + c == a + (b + c))
    }

    @Test("Multiplication is associative: (a · b) · c = a · (b · c)")
    func associativityMultiplication() {
        let a = Complex(1.0, 2.0)
        let b = Complex(3.0, 4.0)
        let c = Complex(5.0, 6.0)

        let left = (a * b) * c
        let right = a * (b * c)

        #expect(abs(left.real - right.real) < 1e-10)
        #expect(abs(left.imaginary - right.imaginary) < 1e-10)
    }

    @Test("Division inverse: z · (z / z) = z")
    func divisionInverse() {
        let z = Complex(2.0, 3.0)
        let result = z * (z / z)
        #expect(abs(result.real - z.real) < 1e-10)
        #expect(abs(result.imaginary - z.imaginary) < 1e-10)
    }
}

/// Test suite for numerical stability under edge conditions.
/// Validates behavior with very small/large numbers, division by near-zero,
/// and accumulated error bounds for quantum computing precision requirements.
@Suite("Numerical Stability")
struct ComplexNumericalStabilityTests {
    @Test("Very small numbers remain finite")
    func verySmallNumbers() {
        let small = Complex(1e-15, 1e-15)
        #expect(small.isFinite)
        let result = small + small
        #expect(result.isFinite)
    }

    @Test("Very large numbers remain finite")
    func veryLargeNumbers() {
        let large = Complex(1e15, 1e15)
        #expect(large.isFinite)
        let result = large * Complex(2.0, 0.0)
        #expect(result.isFinite)
    }

    @Test("Division by near-zero returns NaN")
    func divisionByNearZero() {
        let z = Complex(1.0, 0.0)
        let nearZero = Complex(1e-20, 0.0)
        let result = z / nearZero
        #expect(!result.isFinite)
    }

    @Test("Accumulated error remains within bounds")
    func accumulatedError() {
        var z = Complex(1.0, 0.0)
        let delta = Complex(0.01, 0.01)

        for _ in 0 ..< 100 {
            z = z + delta
        }

        let expected = Complex(2.0, 1.0)
        #expect(abs(z.real - expected.real) < 1e-10)
        #expect(abs(z.imaginary - expected.imaginary) < 1e-10)
    }

    @Test("Magnitude is never negative")
    func magnitudeNeverNegative() {
        let z = Complex(-3.0, -4.0)
        #expect(z.magnitude >= 0)
    }
}

/// Test suite for scalar operations with complex numbers.
/// Validates multiplication and division by real scalars,
/// essential for amplitude scaling in quantum state operations.
@Suite("Scalar Operations")
struct ComplexScalarTests {
    @Test("Scalar multiplication: 2.5 · (2+3i) = (5+7.5i)")
    func scalarMultiplication() {
        let z = Complex(2.0, 3.0)
        let scalar = 2.5
        let result = z * scalar
        #expect(result == Complex(5.0, 7.5))
    }

    @Test("Scalar multiplication is commutative: c · z = z · c")
    func scalarMultiplicationCommutative() {
        let z = Complex(2.0, 3.0)
        let scalar = 2.5
        #expect(z * scalar == scalar * z)
    }

    @Test("Scalar division: (4+6i) / 2 = (2+3i)")
    func scalarDivision() {
        let z = Complex(4.0, 6.0)
        let scalar = 2.0
        let result = z / scalar
        #expect(result == Complex(2.0, 3.0))
    }
}

/// Test suite for generic type system and protocol conformances.
/// Validates Complex<T> works correctly with Float and Double,
/// ensuring Metal GPU compatibility and CPU precision options.
@Suite("Type System and Generics")
struct ComplexTypeSystemTests {
    @Test("Generic over Float works correctly")
    func genericOverFloat() {
        let z = Complex<Float>(1.0, 2.0)
        let w = Complex<Float>(3.0, 4.0)
        let result = z + w
        #expect(abs(result.real - 4.0) < 1e-6)
        #expect(abs(result.imaginary - 6.0) < 1e-6)
    }

    @Test("Generic over Double works correctly")
    func genericOverDouble() {
        let z = Complex<Double>(1.0, 2.0)
        let w = Complex<Double>(3.0, 4.0)
        let result = z + w
        #expect(abs(result.real - 4.0) < 1e-10)
        #expect(abs(result.imaginary - 6.0) < 1e-10)
    }

    @Test("Static constants are accessible")
    func staticConstantsAccessible() {
        _ = Complex<Double>.zero
        _ = Complex<Double>.one
        _ = Complex<Double>.i
        // Test passes if constants can be accessed without error
        #expect(true)
    }
}

/// Test suite for string representation and CustomStringConvertible.
/// Validates human-readable output formats for debugging
/// and educational quantum computing interfaces.
@Suite("Complex String Representation")
struct ComplexStringRepresentationTests {
    @Test("Real-only number displays correctly")
    func descriptionRealOnly() {
        let z = Complex(3.0, 0.0)
        #expect(z.description.contains("3"))
    }

    @Test("Imaginary-only number displays with 'i'")
    func descriptionImaginaryOnly() {
        let z = Complex(0.0, 4.0)
        #expect(z.description.contains("i"))
    }

    @Test("Positive imaginary displays with '+'")
    func descriptionPositiveImaginary() {
        let z = Complex(2.0, 3.0)
        #expect(z.description.contains("+"))
    }

    @Test("Negative imaginary displays with '-'")
    func descriptionNegativeImaginary() {
        let z = Complex(2.0, -3.0)
        #expect(z.description.contains("-"))
    }
}

/// Test suite for Float precision to ensure full branch coverage.
/// Validates that all mathematical operations work correctly with Float type,
/// ensuring Metal GPU compatibility and hitting Float-specific code paths.
@Suite("Float Precision Coverage")
struct ComplexFloatCoverageTests {
    @Test("Float: Basic arithmetic operations")
    func floatArithmetic() {
        let a = Complex<Float>(2.0, 3.0)
        let b = Complex<Float>(4.0, 5.0)

        let sum = a + b
        #expect(abs(sum.real - 6.0) < 1e-6)
        #expect(abs(sum.imaginary - 8.0) < 1e-6)

        let diff = a - b
        #expect(abs(diff.real - -2.0) < 1e-6)

        let product = a * b
        #expect(abs(product.real - -7.0) < 1e-6)
    }

    @Test("Float: Division operation")
    func floatDivision() {
        let a = Complex<Float>(2.0, 3.0)
        let b = Complex<Float>(4.0, 5.0)
        let result = a / b

        #expect(abs(result.real - (23.0 / 41.0)) < 1e-5)
        #expect(abs(result.imaginary - (2.0 / 41.0)) < 1e-5)
    }

    @Test("Float: Division by near-zero")
    func floatDivisionByNearZero() {
        let z = Complex<Float>(1.0, 0.0)
        let nearZero = Complex<Float>(1e-15, 0.0)
        let result = z / nearZero

        #expect(!result.isFinite)
    }

    @Test("Float: Static constants")
    func floatStaticConstants() {
        let zero = Complex<Float>.zero
        let one = Complex<Float>.one
        let i = Complex<Float>.i

        #expect(abs(zero.real) < 1e-6)
        #expect(abs(zero.imaginary) < 1e-6)
        #expect(abs(one.real - 1.0) < 1e-6)
        #expect(abs(one.imaginary) < 1e-6)
        #expect(abs(i.real) < 1e-6)
        #expect(abs(i.imaginary - 1.0) < 1e-6)
    }

    @Test("Float: Equality comparison")
    func floatEquality() {
        let a = Complex<Float>(2.0, 3.0)
        let b = Complex<Float>(2.0, 3.0)
        let c = Complex<Float>(2.1, 3.0)

        #expect(a == b)
        #expect(a != c)
    }

    @Test("Float: Magnitude and phase")
    func floatMagnitudePhase() {
        let z = Complex<Float>(3.0, 4.0)

        #expect(abs(z.magnitude - 5.0) < 1e-5)
        #expect(abs(z.magnitudeSquared - 25.0) < 1e-5)

        let w = Complex<Float>(1.0, 1.0)
        #expect(abs(w.phase - Float.pi / 4.0) < 1e-5)
    }

    @Test("Float: Trigonometric functions (cos/sin)")
    func floatTrigonometricFunctions() {
        let theta = Float.pi / 4.0
        let result = Complex<Float>.exp(theta)

        #expect(abs(result.real - cos(theta)) < 1e-5)
        #expect(abs(result.imaginary - sin(theta)) < 1e-5)
    }

    @Test("Float: Polar conversion")
    func floatPolarConversion() {
        let z = Complex<Float>(3.0, 4.0)
        let (r, theta) = z.toPolar()
        let reconstructed = Complex<Float>.fromPolar(r: r, theta: theta)

        #expect(abs(reconstructed.real - z.real) < 1e-5)
        #expect(abs(reconstructed.imaginary - z.imaginary) < 1e-5)
    }

    @Test("Float: Square root function")
    func floatSquareRoot() {
        let z = Complex<Float>(9.0, 0.0)
        let mag = z.magnitude

        #expect(abs(mag - 9.0) < 1e-5)
    }

    @Test("Float: atan2 function")
    func floatAtan2() {
        let z = Complex<Float>(1.0, 1.0)
        let phase = z.phase

        #expect(abs(phase - Float.pi / 4.0) < 1e-5)
    }

    @Test("Float: Description formatting")
    func floatDescription() {
        let z1 = Complex<Float>(3.0, 0.0)
        #expect(z1.description.contains("3"))

        let z2 = Complex<Float>(0.0, 4.0)
        #expect(z2.description.contains("i"))

        let z3 = Complex<Float>(2.0, 3.0)
        #expect(z3.description.contains("+"))

        let z4 = Complex<Float>(2.0, -3.0)
        #expect(z4.description.contains("-"))
    }

    @Test("Float: Integer literal initialization")
    func floatIntegerLiteral() {
        let z: Complex<Float> = 5
        #expect(abs(z.real - 5.0) < 1e-6)
        #expect(abs(z.imaginary) < 1e-6)
    }

    @Test("Float: Float literal initialization")
    func floatFloatLiteral() {
        let z: Complex<Float> = 3.14
        #expect(abs(z.real - 3.14) < 1e-5)
        #expect(abs(z.imaginary) < 1e-6)
    }

    @Test("Float: Single parameter init")
    func floatSingleParamInit() {
        let z = Complex<Float>(2.5)
        #expect(abs(z.real - 2.5) < 1e-6)
        #expect(abs(z.imaginary) < 1e-6)
    }

    @Test("Double: Single parameter init")
    func doubleSingleParamInit() {
        let z = Complex<Double>(3.7)
        #expect(abs(z.real - 3.7) < 1e-10)
        #expect(abs(z.imaginary) < 1e-10)
    }

    @Test("Double: Integer literal initialization")
    func doubleIntegerLiteral() {
        let z: Complex<Double> = 7
        #expect(abs(z.real - 7.0) < 1e-10)
        #expect(abs(z.imaginary) < 1e-10)
    }

    @Test("Double: Float literal initialization")
    func doubleFloatLiteral() {
        let z: Complex<Double> = 2.71828
        #expect(abs(z.real - 2.71828) < 1e-10)
        #expect(abs(z.imaginary) < 1e-10)
    }
}
