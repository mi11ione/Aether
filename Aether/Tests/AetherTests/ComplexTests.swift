// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
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
        #expect(result == Complex(6.0, 8.0), "Addition should produce (6+8i)")
    }

    @Test("Subtraction: (5+7i) - (2+3i) = (3+4i)")
    func subtraction() {
        let a = Complex(5.0, 7.0)
        let b = Complex(2.0, 3.0)
        let result = a - b
        #expect(result == Complex(3.0, 4.0), "Subtraction should produce (3+4i)")
    }

    @Test("Multiplication: (2+3i)(4+5i) = -7+22i)")
    func multiplication() {
        let a = Complex(2.0, 3.0)
        let b = Complex(4.0, 5.0)
        let result = a * b
        #expect(result == Complex(-7.0, 22.0), "Multiplication should produce (-7+22i)")
    }

    @Test("Division: (2+3i)/(4+5i) = (23+2i)/41")
    func division() {
        let a = Complex(2.0, 3.0)
        let b = Complex(4.0, 5.0)
        let result = a / b
        let expected = Complex(23.0 / 41.0, 2.0 / 41.0)
        #expect(abs(result.real - expected.real) < 1e-10, "Real part of division should be approximately 23/41")
        #expect(abs(result.imaginary - expected.imaginary) < 1e-10, "Imaginary part of division should be approximately 2/41")
    }

    @Test("Negation: -(2+3i) = (-2-3i)")
    func negation() {
        let z = Complex(2.0, 3.0)
        let result = -z
        #expect(result == Complex(-2.0, -3.0), "Negation should produce (-2-3i)")
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
        #expect(result == Complex(-1.0, 0.0), "i squared should equal -1")
    }

    @Test("Conjugate: (2+3i)* = (2-3i)")
    func conjugate() {
        let z = Complex(2.0, 3.0)
        let conjugate = z.conjugate
        #expect(conjugate == Complex(2.0, -3.0), "Conjugate should negate the imaginary part")
    }

    @Test("Conjugate property: z · z* = |z|²")
    func conjugateProperty() {
        let z = Complex(3.0, 4.0)
        let product = z * z.conjugate
        #expect(abs(product.real - z.magnitudeSquared) < 1e-10, "Real part of z * z* should equal |z|²")
        #expect(abs(product.imaginary) < 1e-10, "Imaginary part of z * z* should be zero")
    }

    @Test("Magnitude: |3+4i| = 5")
    func magnitude() {
        let z = Complex(3.0, 4.0)
        #expect(abs(z.magnitude - 5.0) < 1e-10, "Magnitude of (3+4i) should be 5")
    }

    @Test("Magnitude squared: |3+4i|² = 25")
    func magnitudeSquared() {
        let z = Complex(3.0, 4.0)
        #expect(abs(z.magnitudeSquared - 25.0) < 1e-10, "Magnitude squared of (3+4i) should be 25")
    }

    @Test("Phase: arg(1+i) = π/4")
    func phase() {
        let z = Complex(1.0, 1.0)
        #expect(abs(z.phase - .pi / 4.0) < 1e-10, "Phase of (1+i) should be pi/4")
    }
}

/// Test suite for special complex values and identity constants.
/// Validates zero, one, and imaginary unit behavior in arithmetic operations.
/// Ensures protocol conformance members (zero, squareRoot) are accessible.
@Suite("Special Values and Constants")
struct ComplexSpecialValuesTests {
    @Test("Zero operations: z + 0 = z, z - 0 = z, z · 0 = 0")
    func zeroOperations() {
        let zero = Complex<Double>.zero
        let z = Complex(3.0, 4.0)

        #expect(z + zero == z, "Adding zero should not change the value")
        #expect(z - zero == z, "Subtracting zero should not change the value")
        #expect(z * zero == zero, "Multiplying by zero should produce zero")
    }

    @Test("One operations: z · 1 = z, z / 1 = z")
    func oneOperations() {
        let one = Complex<Double>.one
        let z = Complex(3.0, 4.0)

        #expect(z * one == z, "Multiplying by one should not change the value")
        #expect(z / one == z, "Dividing by one should not change the value")
    }

    @Test("Imaginary unit: i = (0, 1)")
    func imaginaryUnit() {
        let i = Complex<Double>.i
        #expect(abs(i.real) < 1e-10, "Real part of i should be zero")
        #expect(abs(i.imaginary - 1.0) < 1e-10, "Imaginary part of i should be 1")
    }

    @Test("Double.zero and Float.zero protocol members")
    func zeroProtocolMembers() {
        #expect(Double.zero == 0.0, "Double.zero should equal 0.0")
        #expect(Float.zero == 0.0, "Float.zero should equal 0.0")
    }

    @Test("Double.squareRoot and Float.squareRoot protocol members")
    func squareRootProtocolMembers() {
        #expect(abs(Double.squareRoot(of: 16.0) - 4.0) < 1e-10, "Double square root of 16 should be 4")
        #expect(abs(Float.squareRoot(of: 25.0) - 5.0) < 1e-6, "Float square root of 25 should be 5")
    }
}

/// Test suite for Euler's identity and exponential form.
/// Validates e^(iθ) = cos(θ) + i·sin(θ) and related identities
/// critical for phase operations in quantum gates.
@Suite("Euler's Identity and Exponential Form")
struct ComplexEulerIdentityTests {
    @Test("Euler's identity: e^(iπ) + 1 = 0")
    func eulerIdentity() {
        let result = Complex<Double>(phase: .pi) + Complex.one
        #expect(abs(result.real) < 1e-10, "Real part of e^(i*pi) + 1 should be zero")
        #expect(abs(result.imaginary) < 1e-10, "Imaginary part of e^(i*pi) + 1 should be zero")
    }

    @Test("Exponential at zero: e^(i0) = 1")
    func expZero() {
        let result = Complex<Double>(phase: 0)
        #expect(result == Complex.one, "e^(i*0) should equal 1")
    }

    @Test("Exponential at π/2: e^(iπ/2) = i")
    func expPiOverTwo() {
        let result = Complex<Double>(phase: .pi / 2.0)
        #expect(abs(result.real) < 1e-10, "Real part of e^(i*pi/2) should be zero")
        #expect(abs(result.imaginary - 1.0) < 1e-10, "Imaginary part of e^(i*pi/2) should be 1")
    }
}

/// Test suite for polar coordinate conversions.
/// Validates r·e^(iθ) <-> (r, θ) round-trip conversions
/// used in phase visualization and quantum state analysis.
@Suite("Polar Coordinate Conversions")
struct ComplexPolarTests {
    @Test("Polar conversion round-trip preserves value")
    func polarConversionRoundTrip() {
        let z = Complex(3.0, 4.0)
        let reconstructed = Complex<Double>(magnitude: z.magnitude, phase: z.phase)

        #expect(abs(reconstructed.real - z.real) < 1e-10, "Real part should survive polar round-trip")
        #expect(abs(reconstructed.imaginary - z.imaginary) < 1e-10, "Imaginary part should survive polar round-trip")
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
        #expect(a + b == b + a, "Addition should be commutative")
    }

    @Test("Multiplication is commutative: a · b = b · a")
    func commutativityMultiplication() {
        let a = Complex(2.0, 3.0)
        let b = Complex(4.0, 5.0)
        #expect(a * b == b * a, "Multiplication should be commutative")
    }

    @Test("Addition is associative: (a + b) + c = a + (b + c)")
    func associativityAddition() {
        let a = Complex(1.0, 2.0)
        let b = Complex(3.0, 4.0)
        let c = Complex(5.0, 6.0)
        #expect((a + b) + c == a + (b + c), "Addition should be associative")
    }

    @Test("Multiplication is associative: (a · b) · c = a · (b · c)")
    func associativityMultiplication() {
        let a = Complex(1.0, 2.0)
        let b = Complex(3.0, 4.0)
        let c = Complex(5.0, 6.0)

        let left = (a * b) * c
        let right = a * (b * c)

        #expect(abs(left.real - right.real) < 1e-10, "Real parts should match for associative multiplication")
        #expect(abs(left.imaginary - right.imaginary) < 1e-10, "Imaginary parts should match for associative multiplication")
    }

    @Test("Division inverse: (z / w) · w = z")
    func divisionInverse() {
        let z = Complex(2.0, 3.0)
        let w = Complex(4.0, 5.0)
        let result = (z / w) * w
        #expect(abs(result.real - z.real) < 1e-10, "Real part should recover after division inverse")
        #expect(abs(result.imaginary - z.imaginary) < 1e-10, "Imaginary part should recover after division inverse")
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
        #expect(small.isFinite, "Very small complex number should be finite")
        let result = small + small
        #expect(result.isFinite, "Sum of very small complex numbers should be finite")
    }

    @Test("Very large numbers remain finite")
    func veryLargeNumbers() {
        let large = Complex(1e15, 1e15)
        #expect(large.isFinite, "Very large complex number should be finite")
        let result = large * Complex(2.0, 0.0)
        #expect(result.isFinite, "Product of large complex numbers should be finite")
    }

    @Test("Accumulated error remains within bounds")
    func accumulatedError() {
        var z = Complex(1.0, 0.0)
        let delta = Complex(0.01, 0.01)

        for _ in 0 ..< 100 {
            z = z + delta
        }

        let expected = Complex(2.0, 1.0)
        #expect(abs(z.real - expected.real) < 1e-10, "Accumulated real part should be approximately 2.0")
        #expect(abs(z.imaginary - expected.imaginary) < 1e-10, "Accumulated imaginary part should be approximately 1.0")
    }

    @Test("Magnitude is never negative")
    func magnitudeNeverNegative() {
        let z = Complex(-3.0, -4.0)
        #expect(z.magnitude >= 0, "Magnitude should never be negative")
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
        #expect(result == Complex(5.0, 7.5), "Scalar multiplication should produce (5+7.5i)")
    }

    @Test("Scalar multiplication is commutative: c · z = z · c")
    func scalarMultiplicationCommutative() {
        let z = Complex(2.0, 3.0)
        let scalar = 2.5
        #expect(z * scalar == scalar * z, "Scalar multiplication should be commutative")
    }

    @Test("Scalar division: (4+6i) / 2 = (2+3i)")
    func scalarDivision() {
        let z = Complex(4.0, 6.0)
        let scalar = 2.0
        let result = z / scalar
        #expect(result == Complex(2.0, 3.0), "Scalar division should produce (2+3i)")
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
        #expect(abs(result.real - 4.0) < 1e-6, "Float real part should be approximately 4.0 after addition")
        #expect(abs(result.imaginary - 6.0) < 1e-6, "Float imaginary part should be approximately 6.0 after addition")
    }

    @Test("Generic over Double works correctly")
    func genericOverDouble() {
        let z = Complex<Double>(1.0, 2.0)
        let w = Complex<Double>(3.0, 4.0)
        let result = z + w
        #expect(abs(result.real - 4.0) < 1e-10, "Double real part should be approximately 4.0 after addition")
        #expect(abs(result.imaginary - 6.0) < 1e-10, "Double imaginary part should be approximately 6.0 after addition")
    }

    @Test("Static constants have correct values")
    func staticConstantsAccessible() {
        #expect(Complex<Double>.zero == Complex(0.0, 0.0), "Complex.zero should equal (0+0i)")
        #expect(Complex<Double>.one == Complex(1.0, 0.0), "Complex.one should equal (1+0i)")
        #expect(Complex<Double>.i == Complex(0.0, 1.0), "Complex.i should equal (0+1i)")
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
        #expect(z.description.contains("3"), "Description of real-only number should contain '3'")
    }

    @Test("Imaginary-only number displays with 'i'")
    func descriptionImaginaryOnly() {
        let z = Complex(0.0, 4.0)
        #expect(z.description.contains("i"), "Description of imaginary-only number should contain 'i'")
    }

    @Test("Positive imaginary displays with '+'")
    func descriptionPositiveImaginary() {
        let z = Complex(2.0, 3.0)
        #expect(z.description.contains("+"), "Description with positive imaginary should contain '+'")
    }

    @Test("Negative imaginary displays with '-'")
    func descriptionNegativeImaginary() {
        let z = Complex(2.0, -3.0)
        #expect(z.description.contains("-"), "Description with negative imaginary should contain '-'")
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
        #expect(abs(sum.real - 6.0) < 1e-6, "Float sum real part should be approximately 6.0")
        #expect(abs(sum.imaginary - 8.0) < 1e-6, "Float sum imaginary part should be approximately 8.0")

        let diff = a - b
        #expect(abs(diff.real - -2.0) < 1e-6, "Float difference real part should be approximately -2.0")

        let product = a * b
        #expect(abs(product.real - -7.0) < 1e-6, "Float product real part should be approximately -7.0")
    }

    @Test("Float: Division operation")
    func floatDivision() {
        let a = Complex<Float>(2.0, 3.0)
        let b = Complex<Float>(4.0, 5.0)
        let result = a / b

        #expect(abs(result.real - (23.0 / 41.0)) < 1e-6, "Float division real part should be approximately 23/41")
        #expect(abs(result.imaginary - (2.0 / 41.0)) < 1e-6, "Float division imaginary part should be approximately 2/41")
    }

    @Test("Float: Static constants")
    func floatStaticConstants() {
        let zero = Complex<Float>.zero
        let one = Complex<Float>.one
        let i = Complex<Float>.i

        #expect(abs(zero.real) < 1e-6, "Float zero real part should be approximately 0")
        #expect(abs(zero.imaginary) < 1e-6, "Float zero imaginary part should be approximately 0")
        #expect(abs(one.real - 1.0) < 1e-6, "Float one real part should be approximately 1")
        #expect(abs(one.imaginary) < 1e-6, "Float one imaginary part should be approximately 0")
        #expect(abs(i.real) < 1e-6, "Float i real part should be approximately 0")
        #expect(abs(i.imaginary - 1.0) < 1e-6, "Float i imaginary part should be approximately 1")
    }

    @Test("Float: Equality comparison")
    func floatEquality() {
        let a = Complex<Float>(2.0, 3.0)
        let b = Complex<Float>(2.0, 3.0)
        let c = Complex<Float>(2.1, 3.0)

        #expect(a == b, "Identical Float complex numbers should be equal")
        #expect(a != c, "Different Float complex numbers should not be equal")
    }

    @Test("Float: Magnitude and phase")
    func floatMagnitudePhase() {
        let z = Complex<Float>(3.0, 4.0)

        #expect(abs(z.magnitude - 5.0) < 1e-6, "Float magnitude of (3+4i) should be approximately 5")
        #expect(abs(z.magnitudeSquared - 25.0) < 1e-6, "Float magnitude squared of (3+4i) should be approximately 25")

        let w = Complex<Float>(1.0, 1.0)
        #expect(abs(w.phase - Float.pi / 4.0) < 1e-6, "Float phase of (1+i) should be approximately pi/4")
    }

    @Test("Float: Trigonometric functions (cos/sin)")
    func floatTrigonometricFunctions() {
        let theta = Float.pi / 4.0
        let result = Complex<Float>(phase: theta)

        #expect(abs(result.real - cos(theta)) < 1e-6, "Float real part should equal cos(theta)")
        #expect(abs(result.imaginary - sin(theta)) < 1e-6, "Float imaginary part should equal sin(theta)")
    }

    @Test("Float: Polar conversion")
    func floatPolarConversion() {
        let z = Complex<Float>(3.0, 4.0)
        let reconstructed = Complex<Float>(magnitude: z.magnitude, phase: z.phase)

        #expect(abs(reconstructed.real - z.real) < 1e-6, "Float real part should survive polar round-trip")
        #expect(abs(reconstructed.imaginary - z.imaginary) < 1e-6, "Float imaginary part should survive polar round-trip")
    }

    @Test("Float: Magnitude calculation")
    func floatMagnitude() {
        let z = Complex<Float>(9.0, 0.0)
        let mag = z.magnitude

        #expect(abs(mag - 9.0) < 1e-6, "Float magnitude of (9+0i) should be approximately 9")
    }

    @Test("Float: atan2 function")
    func floatAtan2() {
        let z = Complex<Float>(1.0, 1.0)
        let phase = z.phase

        #expect(abs(phase - Float.pi / 4.0) < 1e-6, "Float atan2-based phase of (1+i) should be approximately pi/4")
    }

    @Test("Float: Description formatting")
    func floatDescription() {
        let z1 = Complex<Float>(3.0, 0.0)
        #expect(z1.description.contains("3"), "Float real-only description should contain '3'")

        let z2 = Complex<Float>(0.0, 4.0)
        #expect(z2.description.contains("i"), "Float imaginary-only description should contain 'i'")

        let z3 = Complex<Float>(2.0, 3.0)
        #expect(z3.description.contains("+"), "Float positive imaginary description should contain '+'")

        let z4 = Complex<Float>(2.0, -3.0)
        #expect(z4.description.contains("-"), "Float negative imaginary description should contain '-'")
    }

    @Test("Float: Integer literal initialization")
    func floatIntegerLiteral() {
        let z: Complex<Float> = 5
        #expect(abs(z.real - 5.0) < 1e-6, "Float integer literal real part should be 5.0")
        #expect(abs(z.imaginary) < 1e-6, "Float integer literal imaginary part should be 0")
    }

    @Test("Float: Float literal initialization")
    func floatFloatLiteral() {
        let z: Complex<Float> = 3.14
        #expect(abs(z.real - 3.14) < 1e-6, "Float literal real part should be approximately 3.14")
        #expect(abs(z.imaginary) < 1e-6, "Float literal imaginary part should be 0")
    }

    @Test("Float: Single parameter init")
    func floatSingleParamInit() {
        let z = Complex<Float>(2.5)
        #expect(abs(z.real - 2.5) < 1e-6, "Float single-param init real part should be 2.5")
        #expect(abs(z.imaginary) < 1e-6, "Float single-param init imaginary part should be 0")
    }

    @Test("Double: Single parameter init")
    func doubleSingleParamInit() {
        let z = Complex<Double>(3.7)
        #expect(abs(z.real - 3.7) < 1e-10, "Double single-param init real part should be 3.7")
        #expect(abs(z.imaginary) < 1e-10, "Double single-param init imaginary part should be 0")
    }

    @Test("Double: Integer literal initialization")
    func doubleIntegerLiteral() {
        let z: Complex<Double> = 7
        #expect(abs(z.real - 7.0) < 1e-10, "Double integer literal real part should be 7.0")
        #expect(abs(z.imaginary) < 1e-10, "Double integer literal imaginary part should be 0")
    }

    @Test("Double: Float literal initialization")
    func doubleFloatLiteral() {
        let z: Complex<Double> = 2.71828
        #expect(abs(z.real - 2.71828) < 1e-10, "Double float literal real part should be approximately 2.71828")
        #expect(abs(z.imaginary) < 1e-10, "Double float literal imaginary part should be 0")
    }
}

/// Test suite for compound assignment operators (+=, -=, *=, /=).
/// Verifies they produce identical results to their binary counterparts
/// for complex-complex and complex-scalar operations across types.
@Suite("Compound Assignment Operators")
struct ComplexCompoundAssignmentOperatorTests {
    @Test("+= matches + (Double)")
    func plusEqualsMatchesPlusDouble() {
        var a = Complex<Double>(3.0, 4.0)
        let b = Complex<Double>(1.0, -2.0)
        let expected = a + b
        a += b
        #expect(a == expected, "+= should produce the same result as + for Double")
    }

    @Test("-= matches - (Double)")
    func minusEqualsMatchesMinusDouble() {
        var a = Complex<Double>(3.0, 4.0)
        let b = Complex<Double>(1.0, -2.0)
        let expected = a - b
        a -= b
        #expect(a == expected, "-= should produce the same result as - for Double")
    }

    @Test("*= matches * (complex-complex, Double)")
    func timesEqualsMatchesTimesComplexDouble() {
        var a = Complex<Double>(2.0, 3.0)
        let b = Complex<Double>(4.0, -1.0)
        let expected = a * b
        a *= b
        #expect(a == expected, "*= should produce the same result as * for complex Double")
    }

    @Test("/= matches / (complex-complex, Double)")
    func divideEqualsMatchesDivideComplexDouble() {
        var a = Complex<Double>(2.0, 3.0)
        let b = Complex<Double>(4.0, -1.0)
        let expected = a / b
        a /= b
        #expect(a == expected, "/= should produce the same result as / for complex Double")
    }

    @Test("*= matches scalar multiply (Double)")
    func timesEqualsMatchesScalarDouble() {
        var a = Complex<Double>(-1.5, 2.0)
        let s = 3.0
        let expected = a * s
        a *= s
        #expect(a == expected, "*= should produce the same result as scalar * for Double")
    }

    @Test("/= matches scalar divide (Double)")
    func divideEqualsMatchesScalarDouble() {
        var a = Complex<Double>(-1.5, 2.0)
        let s = 2.0
        let expected = a / s
        a /= s
        #expect(a == expected, "/= should produce the same result as scalar / for Double")
    }

    @Test("+= and *= scalar work for Float as well")
    func plusEqualsAndTimesEqualsFloat() {
        var a = Complex<Float>(2.0, 3.0)
        let b = Complex<Float>(-1.0, 0.5)
        let sumExpected = a + b
        a += b
        #expect(a == sumExpected, "+= should produce the same result as + for Float")

        let s: Float = 2.5
        let mulExpected = a * s
        a *= s
        #expect(a == mulExpected, "*= should produce the same result as scalar * for Float")
    }

    @Test("-= matches - (Float)")
    func minusEqualsMatchesMinusFloat() {
        var a = Complex<Float>(5.0, 7.0)
        let b = Complex<Float>(2.0, 3.0)
        let expected = a - b
        a -= b
        #expect(a == expected, "-= should produce the same result as - for Float")
    }

    @Test("/= matches / (complex-complex, Float)")
    func divideEqualsMatchesDivideComplexFloat() {
        var a = Complex<Float>(6.0, 8.0)
        let b = Complex<Float>(3.0, -4.0)
        let expected = a / b
        a /= b
        #expect(abs(a.real - expected.real) < 1e-6, "/= real part should match / for complex Float")
        #expect(abs(a.imaginary - expected.imaginary) < 1e-6, "/= imaginary part should match / for complex Float")
    }

    @Test("/= matches scalar divide (Float)")
    func divideEqualsMatchesScalarFloat() {
        var a = Complex<Float>(10.0, 15.0)
        let s: Float = 5.0
        let expected = a / s
        a /= s
        #expect(abs(a.real - expected.real) < 1e-6, "/= real part should match scalar / for Float")
        #expect(abs(a.imaginary - expected.imaginary) < 1e-6, "/= imaginary part should match scalar / for Float")
    }

    @Test("Real-only init works for Float")
    func realOnlyInitFloat() {
        let z = Complex<Float>(5.0)
        #expect(abs(z.real - 5.0) < 1e-6, "Real-only Float init real part should be 5.0")
        #expect(abs(z.imaginary - 0.0) < 1e-6, "Real-only Float init imaginary part should be 0")
    }

    @Test("Imaginary unit i works for Float")
    func imaginaryUnitFloat() {
        let i = Complex<Float>.i
        #expect(abs(i.real - 0.0) < 1e-6, "Float i real part should be 0")
        #expect(abs(i.imaginary - 1.0) < 1e-6, "Float i imaginary part should be 1")

        let iPower2 = i * i
        #expect(abs(iPower2.real - -1.0) < 1e-6, "Float i^2 real part should be -1")
        #expect(abs(iPower2.imaginary - 0.0) < 1e-6, "Float i^2 imaginary part should be 0")
    }

    @Test("Phase computation works for Float")
    func phaseComputationFloat() {
        let z = Complex<Float>(1.0, 1.0)
        let theta = z.phase
        let expectedPhase = Float.pi / 4.0
        #expect(abs(theta - expectedPhase) < 1e-6, "Float phase of (1+i) should be approximately pi/4")

        let zNegative = Complex<Float>(-1.0, 0.0)
        let phaseNeg = zNegative.phase
        #expect(abs(abs(phaseNeg) - Float.pi) < 1e-6, "Float phase of (-1+0i) should be approximately pi")
    }
}

/// Test suite for Hashable conformance with epsilon-based quantization.
/// Validates that complex numbers equal within epsilon have equal hashes,
/// enabling safe use in Sets and Dictionaries despite floating-point precision.
@Suite("Hashable with Epsilon Quantization")
struct ComplexHashableTests {
    @Test("Equal complex numbers have equal hashes (Double)")
    func equalNumbersEqualHashesDouble() {
        let z1 = Complex<Double>(3.0, 4.0)
        let z2 = Complex<Double>(3.0, 4.0)
        #expect(z1.hashValue == z2.hashValue, "Equal Double complex numbers should have equal hashes")
    }

    @Test("Numbers within epsilon have equal hashes (Double)")
    func withinEpsilonEqualHashesDouble() {
        let z1 = Complex<Double>(1.0, 2.0)
        let z2 = Complex<Double>(1.0 + 1e-11, 2.0 + 1e-11)
        #expect(z1 == z2, "Values should be equal within epsilon")
        #expect(z1.hashValue == z2.hashValue, "Equal values must have equal hashes")
    }

    @Test("Numbers beyond epsilon have different hashes (Double)")
    func beyondEpsilonDifferentHashesDouble() {
        let z1 = Complex<Double>(1.0, 2.0)
        let z2 = Complex<Double>(1.0 + 1e-8, 2.0)
        #expect(z1 != z2, "Values should be unequal beyond epsilon")
        #expect(z1.hashValue != z2.hashValue, "Unequal values should have different hashes")
    }

    @Test("Can be used in Set (Double)")
    func canBeUsedInSetDouble() {
        var set = Set<Complex<Double>>()
        let z1 = Complex<Double>(1.0, 2.0)
        let z2 = Complex<Double>(1.0 + 1e-11, 2.0)
        let z3 = Complex<Double>(3.0, 4.0)

        set.insert(z1)
        set.insert(z2)
        set.insert(z3)

        #expect(set.count == 2, "z1 and z2 are equal, so only 2 unique values")
        #expect(set.contains(z1), "Set should contain z1")
        #expect(set.contains(z2), "Set should contain z2")
        #expect(set.contains(z3), "Set should contain z3")
    }

    @Test("Can be used as Dictionary key (Double)")
    func canBeUsedAsDictionaryKeyDouble() {
        var dict: [Complex<Double>: String] = [:]
        let z1 = Complex<Double>(1.0, 2.0)
        let z2 = Complex<Double>(1.0 + 1e-11, 2.0)

        dict[z1] = "first"
        dict[z2] = "second"

        #expect(dict.count == 1, "z1 and z2 are equal, so only 1 key")
        #expect(dict[z1] == "second", "z2 overwrote z1's value")
        #expect(dict[z2] == "second", "Dictionary lookup with z2 should return 'second'")
    }

    @Test("Equal complex numbers have equal hashes (Float)")
    func equalNumbersEqualHashesFloat() {
        let z1 = Complex<Float>(3.0, 4.0)
        let z2 = Complex<Float>(3.0, 4.0)
        #expect(z1.hashValue == z2.hashValue, "Equal Float complex numbers should have equal hashes")
    }

    @Test("Numbers within epsilon have equal hashes (Float)")
    func withinEpsilonEqualHashesFloat() {
        let z1 = Complex<Float>(1.0, 2.0)
        let z2 = Complex<Float>(1.0 + 1e-7, 2.0 + 1e-7)
        #expect(z1 == z2, "Values should be equal within epsilon")
        #expect(z1.hashValue == z2.hashValue, "Equal values must have equal hashes")
    }

    @Test("Numbers beyond epsilon have different hashes (Float)")
    func beyondEpsilonDifferentHashesFloat() {
        let z1 = Complex<Float>(1.0, 2.0)
        let z2 = Complex<Float>(1.0 + 1e-4, 2.0)
        #expect(z1 != z2, "Values should be unequal beyond epsilon")
        #expect(z1.hashValue != z2.hashValue, "Unequal values should have different hashes")
    }

    @Test("Can be used in Set (Float)")
    func canBeUsedInSetFloat() {
        var set = Set<Complex<Float>>()
        let z1 = Complex<Float>(1.0, 2.0)
        let z2 = Complex<Float>(1.0 + 1e-7, 2.0)
        let z3 = Complex<Float>(3.0, 4.0)

        set.insert(z1)
        set.insert(z2)
        set.insert(z3)

        #expect(set.count == 2, "z1 and z2 are equal, so only 2 unique values")
        #expect(set.contains(z1), "Float Set should contain z1")
        #expect(set.contains(z2), "Float Set should contain z2")
        #expect(set.contains(z3), "Float Set should contain z3")
    }

    @Test("Zero has consistent hash")
    func zeroHasConsistentHash() {
        let zero1 = Complex<Double>.zero
        let zero2 = Complex<Double>(0.0, 0.0)
        let zero3 = Complex<Double>(1e-11, 1e-11)

        #expect(zero1.hashValue == zero2.hashValue, "Complex.zero and (0,0) should have equal hashes")
        #expect(zero1.hashValue == zero3.hashValue, "Near-zero quantizes to zero")
    }

    @Test("Quantization boundary consistency")
    func quantizationBoundaryConsistency() {
        let epsilon = 1e-10
        let z1 = Complex<Double>(epsilon, 0.0)
        let z2 = Complex<Double>(epsilon * 0.5, 0.0)

        #expect(z1 == z2, "Both should quantize to same grid point")
        #expect(z1.hashValue == z2.hashValue, "Values on same grid point should have equal hashes")
    }
}
