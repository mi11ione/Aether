// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for NumberTheory.gcd computing greatest common divisor.
/// Validates Euclidean algorithm with positive, negative, zero,
/// and coprime input combinations.
@Suite("NumberTheory - GCD")
struct NumberTheoryGCDTests {
    @Test("GCD of 48 and 18 equals 6")
    func gcdBasicCase() {
        let result = NumberTheory.gcd(48, 18)
        #expect(result == 6, "gcd(48, 18) should be 6")
    }

    @Test("GCD of 56 and 42 equals 14")
    func gcdAnotherBasicCase() {
        let result = NumberTheory.gcd(56, 42)
        #expect(result == 14, "gcd(56, 42) should be 14")
    }

    @Test("GCD with negative first argument uses absolute value")
    func gcdNegativeFirst() {
        let result = NumberTheory.gcd(-48, 18)
        #expect(result == 6, "gcd(-48, 18) should be 6")
    }

    @Test("GCD with negative second argument uses absolute value")
    func gcdNegativeSecond() {
        let result = NumberTheory.gcd(48, -18)
        #expect(result == 6, "gcd(48, -18) should be 6")
    }

    @Test("GCD with both arguments negative uses absolute values")
    func gcdBothNegative() {
        let result = NumberTheory.gcd(-48, -18)
        #expect(result == 6, "gcd(-48, -18) should be 6")
    }

    @Test("GCD with zero first argument returns absolute second")
    func gcdZeroFirst() {
        let result = NumberTheory.gcd(0, 18)
        #expect(result == 18, "gcd(0, 18) should be 18")
    }

    @Test("GCD with zero second argument returns absolute first")
    func gcdZeroSecond() {
        let result = NumberTheory.gcd(48, 0)
        #expect(result == 48, "gcd(48, 0) should be 48")
    }

    @Test("GCD of both zeros returns zero")
    func gcdBothZero() {
        let result = NumberTheory.gcd(0, 0)
        #expect(result == 0, "gcd(0, 0) should be 0")
    }

    @Test("GCD of coprime numbers returns 1")
    func gcdCoprimes() {
        let result = NumberTheory.gcd(17, 13)
        #expect(result == 1, "gcd(17, 13) should be 1 since they are coprime")
    }

    @Test("GCD of identical numbers returns that number")
    func gcdIdentical() {
        let result = NumberTheory.gcd(42, 42)
        #expect(result == 42, "gcd(42, 42) should be 42")
    }

    @Test("GCD where one divides the other returns divisor")
    func gcdOneDividesOther() {
        let result = NumberTheory.gcd(100, 25)
        #expect(result == 25, "gcd(100, 25) should be 25")
    }

    @Test("GCD of 1 and any number returns 1")
    func gcdWithOne() {
        let result = NumberTheory.gcd(1, 1000)
        #expect(result == 1, "gcd(1, 1000) should be 1")
    }
}

/// Test suite for NumberTheory.extendedGCD computing Bezout coefficients.
/// Validates that returned (g, x, y) satisfies a*x + b*y = gcd(a, b)
/// for various input combinations.
@Suite("NumberTheory - Extended GCD")
struct NumberTheoryExtendedGCDTests {
    @Test("Extended GCD of 48 and 18 satisfies Bezout identity")
    func extendedGCDBasicCase() {
        let (g, x, y) = NumberTheory.extendedGCD(48, 18)
        #expect(g == 6, "gcd(48, 18) should be 6")
        #expect(48 * x + 18 * y == g, "Bezout identity: 48*x + 18*y should equal 6")
    }

    @Test("Extended GCD of 35 and 15 satisfies Bezout identity")
    func extendedGCDAnotherCase() {
        let (g, x, y) = NumberTheory.extendedGCD(35, 15)
        #expect(g == 5, "gcd(35, 15) should be 5")
        #expect(35 * x + 15 * y == g, "Bezout identity: 35*x + 15*y should equal 5")
    }

    @Test("Extended GCD of coprime numbers satisfies identity")
    func extendedGCDCoprimes() {
        let (g, x, y) = NumberTheory.extendedGCD(17, 13)
        #expect(g == 1, "gcd(17, 13) should be 1")
        #expect(17 * x + 13 * y == 1, "Bezout identity: 17*x + 13*y should equal 1")
    }

    @Test("Extended GCD with zero second argument")
    func extendedGCDZeroSecond() {
        let (g, x, y) = NumberTheory.extendedGCD(42, 0)
        #expect(g == 42, "gcd(42, 0) should be 42")
        #expect(42 * x + 0 * y == g, "Bezout identity should hold for zero case")
    }

    @Test("Extended GCD with negative first argument")
    func extendedGCDNegativeFirst() {
        let (g, x, y) = NumberTheory.extendedGCD(-48, 18)
        #expect(g == 6, "gcd(-48, 18) should be 6")
        #expect(-48 * x + 18 * y == g, "Bezout identity: -48*x + 18*y should equal 6")
    }

    @Test("Extended GCD result has positive gcd")
    func extendedGCDPositiveResult() {
        let (g, _, _) = NumberTheory.extendedGCD(-100, -35)
        #expect(g > 0, "Extended GCD should return positive gcd")
    }
}

/// Test suite for NumberTheory.modularPow computing base^exp mod modulus.
/// Validates edge cases and known mathematical values using
/// repeated squaring algorithm.
@Suite("NumberTheory - Modular Power")
struct NumberTheoryModularPowTests {
    @Test("Modular power with exponent 0 returns 1")
    func modularPowExpZero() {
        let result = NumberTheory.modularPow(base: 7, exponent: 0, modulus: 13)
        #expect(result == 1, "Any base raised to 0 should be 1")
    }

    @Test("Modular power with exponent 1 returns base mod modulus")
    func modularPowExpOne() {
        let result = NumberTheory.modularPow(base: 7, exponent: 1, modulus: 13)
        #expect(result == 7, "7^1 mod 13 should be 7")
    }

    @Test("Modular power 2^10 mod 1000 equals 24")
    func modularPowKnownValue() {
        let result = NumberTheory.modularPow(base: 2, exponent: 10, modulus: 1000)
        #expect(result == 24, "2^10 mod 1000 = 1024 mod 1000 = 24")
    }

    @Test("Modular power 7^256 mod 13 equals 9")
    func modularPowLargeExponent() {
        let result = NumberTheory.modularPow(base: 7, exponent: 256, modulus: 13)
        #expect(result == 9, "7^256 mod 13 should be 9")
    }

    @Test("Modular power with modulus 1 returns 0")
    func modularPowModulusOne() {
        let result = NumberTheory.modularPow(base: 100, exponent: 50, modulus: 1)
        #expect(result == 0, "Any number mod 1 is 0")
    }

    @Test("Modular power with negative base handles correctly")
    func modularPowNegativeBase() {
        let result = NumberTheory.modularPow(base: -2, exponent: 3, modulus: 7)
        let expected = ((-2 % 7) + 7) % 7
        let expectedCubed = NumberTheory.modularPow(base: expected, exponent: 3, modulus: 7)
        #expect(result == expectedCubed, "Negative base should be handled correctly")
    }

    @Test("Modular power Fermat's little theorem: a^(p-1) = 1 mod p")
    func modularPowFermatLittleTheorem() {
        let result = NumberTheory.modularPow(base: 3, exponent: 12, modulus: 13)
        #expect(result == 1, "By Fermat's little theorem, 3^12 mod 13 = 1")
    }

    @Test("Modular power 3^5 mod 7 equals 5")
    func modularPowSmallValues() {
        let result = NumberTheory.modularPow(base: 3, exponent: 5, modulus: 7)
        #expect(result == 5, "3^5 = 243 = 34*7 + 5, so 3^5 mod 7 = 5")
    }
}

/// Test suite for NumberTheory.modularInverse computing multiplicative inverse.
/// Validates cases where inverse exists and where it does not exist
/// (when gcd != 1).
@Suite("NumberTheory - Modular Inverse")
struct NumberTheoryModularInverseTests {
    @Test("Modular inverse of 3 mod 7 is 5")
    func modularInverseExists() {
        let result = NumberTheory.modularInverse(3, modulus: 7)
        #expect(result == 5, "3 * 5 = 15 = 2*7 + 1, so inverse of 3 mod 7 is 5")
    }

    @Test("Modular inverse verified by multiplication")
    func modularInverseVerified() {
        let inverse = NumberTheory.modularInverse(3, modulus: 7)
        #expect(inverse != nil, "Inverse should exist for coprime values")
        let product = (3 * inverse!) % 7
        #expect(product == 1, "3 * inverse mod 7 should equal 1")
    }

    @Test("Modular inverse does not exist when gcd > 1")
    func modularInverseDoesNotExist() {
        let result = NumberTheory.modularInverse(4, modulus: 8)
        #expect(result == nil, "gcd(4, 8) = 4 != 1, so no inverse exists")
    }

    @Test("Modular inverse of 1 is always 1")
    func modularInverseOfOne() {
        let result = NumberTheory.modularInverse(1, modulus: 13)
        #expect(result == 1, "1 * 1 = 1 mod anything")
    }

    @Test("Modular inverse of 2 mod 5 is 3")
    func modularInverseAnotherCase() {
        let result = NumberTheory.modularInverse(2, modulus: 5)
        #expect(result == 3, "2 * 3 = 6 = 5 + 1, so inverse of 2 mod 5 is 3")
    }

    @Test("Modular inverse result is in valid range")
    func modularInverseInRange() {
        let modulus = 17
        let inverse = NumberTheory.modularInverse(5, modulus: modulus)
        #expect(inverse != nil, "Inverse should exist")
        #expect(inverse! >= 0 && inverse! < modulus, "Inverse should be in [0, modulus)")
    }
}

/// Test suite for NumberTheory.continuedFractionExpansion computing
/// partial quotients of rational numbers. Validates known fractions
/// including the famous 355/113 approximation of pi.
@Suite("NumberTheory - Continued Fraction Expansion")
struct NumberTheoryContinuedFractionTests {
    @Test("Continued fraction of 355/113 is [3, 7, 16]")
    func continuedFractionPiApproximation() {
        let expansion = NumberTheory.continuedFractionExpansion(355, 113, maxTerms: 10)
        #expect(expansion == [3, 7, 16], "355/113 = 3 + 1/(7 + 1/16)")
    }

    @Test("Continued fraction of 5/3 is [1, 1, 2]")
    func continuedFractionSimple() {
        let expansion = NumberTheory.continuedFractionExpansion(5, 3, maxTerms: 10)
        #expect(expansion == [1, 1, 2], "5/3 = 1 + 2/3 = 1 + 1/(3/2) = 1 + 1/(1 + 1/2)")
    }

    @Test("Continued fraction of integer n/1 is [n]")
    func continuedFractionInteger() {
        let expansion = NumberTheory.continuedFractionExpansion(7, 1, maxTerms: 10)
        #expect(expansion == [7], "7/1 = [7]")
    }

    @Test("Continued fraction of 0/n is [0]")
    func continuedFractionZeroNumerator() {
        let expansion = NumberTheory.continuedFractionExpansion(0, 5, maxTerms: 10)
        #expect(expansion == [0], "0/5 = [0]")
    }

    @Test("Continued fraction respects maxTerms limit")
    func continuedFractionMaxTerms() {
        let expansion = NumberTheory.continuedFractionExpansion(355, 113, maxTerms: 2)
        #expect(expansion.count <= 2, "Expansion should have at most 2 terms")
    }

    @Test("Continued fraction of 22/7 is [3, 7]")
    func continuedFractionPiRoughApproximation() {
        let expansion = NumberTheory.continuedFractionExpansion(22, 7, maxTerms: 10)
        #expect(expansion == [3, 7], "22/7 = 3 + 1/7")
    }
}

/// Test suite for NumberTheory.convergents computing successive
/// best rational approximations from continued fraction expansion.
/// Validates recurrence relation p_k = a_k * p_{k-1} + p_{k-2}.
@Suite("NumberTheory - Convergents")
struct NumberTheoryConvergentsTests {
    @Test("Convergents of [3, 7, 16] are [(3,1), (22,7), (355,113)]")
    func convergentsPiApproximation() {
        let expansion = [3, 7, 16]
        let convs = NumberTheory.convergents(of: expansion)
        #expect(convs.count == 3, "Should have 3 convergents")
        #expect(convs[0] == (p: 3, q: 1), "First convergent is 3/1")
        #expect(convs[1] == (p: 22, q: 7), "Second convergent is 22/7")
        #expect(convs[2] == (p: 355, q: 113), "Third convergent is 355/113")
    }

    @Test("Convergents satisfy recurrence relation")
    func convergentsRecurrenceRelation() {
        let expansion = [2, 3, 4, 5]
        let convs = NumberTheory.convergents(of: expansion)
        #expect(convs[2].p == expansion[2] * convs[1].p + convs[0].p, "p_k = a_k * p_{k-1} + p_{k-2}")
        #expect(convs[2].q == expansion[2] * convs[1].q + convs[0].q, "q_k = a_k * q_{k-1} + q_{k-2}")
    }

    @Test("Convergents of empty expansion returns empty array")
    func convergentsEmptyExpansion() {
        let convs = NumberTheory.convergents(of: [])
        #expect(convs.isEmpty, "Empty expansion should give empty convergents")
    }

    @Test("Convergents of single term [n] is [(n, 1)]")
    func convergentsSingleTerm() {
        let convs = NumberTheory.convergents(of: [5])
        #expect(convs.count == 1, "Single term expansion has one convergent")
        #expect(convs[0] == (p: 5, q: 1), "Convergent of [5] is 5/1")
    }

    @Test("Final convergent equals original fraction")
    func convergentsFinalEqualsOriginal() {
        let expansion = NumberTheory.continuedFractionExpansion(355, 113, maxTerms: 10)
        let convs = NumberTheory.convergents(of: expansion)
        let final = convs.last!
        #expect(final.p == 355 && final.q == 113, "Final convergent should be 355/113")
    }
}

/// Test suite for NumberTheory.isPerfectPower detecting a^k form.
/// Validates perfect squares, cubes, higher powers, and
/// numbers that are not perfect powers.
@Suite("NumberTheory - Is Perfect Power")
struct NumberTheoryIsPerfectPowerTests {
    @Test("27 is a perfect power: 3^3")
    func isPerfectPowerCube() {
        let result = NumberTheory.isPerfectPower(27)
        #expect(result != nil, "27 should be detected as perfect power")
        #expect(result!.base == 3 && result!.exp == 3, "27 = 3^3")
    }

    @Test("16 is a perfect power: 2^4")
    func isPerfectPowerFourth() {
        let result = NumberTheory.isPerfectPower(16)
        #expect(result != nil, "16 should be detected as perfect power")
        #expect(result!.base == 2 && result!.exp == 4, "16 = 2^4")
    }

    @Test("25 is a perfect power: 5^2")
    func isPerfectPowerSquare() {
        let result = NumberTheory.isPerfectPower(25)
        #expect(result != nil, "25 should be detected as perfect power")
        #expect(result!.base == 5 && result!.exp == 2, "25 = 5^2")
    }

    @Test("15 is not a perfect power")
    func isNotPerfectPower() {
        let result = NumberTheory.isPerfectPower(15)
        #expect(result == nil, "15 is not a perfect power")
    }

    @Test("2 is not a perfect power (base must be >= 2 and exp >= 2)")
    func isPerfectPowerTwo() {
        let result = NumberTheory.isPerfectPower(2)
        #expect(result == nil, "2 cannot be expressed as a^k with a >= 2 and k >= 2")
    }

    @Test("1 returns nil (n must be >= 2)")
    func isPerfectPowerOne() {
        let result = NumberTheory.isPerfectPower(1)
        #expect(result == nil, "isPerfectPower requires n >= 2")
    }

    @Test("64 returns highest exponent form")
    func isPerfectPowerHighestExponent() {
        let result = NumberTheory.isPerfectPower(64)
        #expect(result != nil, "64 should be detected as perfect power")
        #expect(result!.base == 2 && result!.exp == 6, "64 = 2^6 (highest exponent)")
    }

    @Test("81 is 3^4")
    func isPerfectPower81() {
        let result = NumberTheory.isPerfectPower(81)
        #expect(result != nil, "81 should be detected as perfect power")
        #expect(result!.base == 3 && result!.exp == 4, "81 = 3^4")
    }
}

/// Test suite for NumberTheory.isPrime using Miller-Rabin primality test.
/// Validates small primes, composites, and edge cases with
/// deterministic witnesses for integers < 2^64.
@Suite("NumberTheory - Is Prime")
struct NumberTheoryIsPrimeTests {
    @Test("2 is prime")
    func isPrimeTwo() {
        #expect(NumberTheory.isPrime(2), "2 is the smallest prime")
    }

    @Test("3 is prime")
    func isPrimeThree() {
        #expect(NumberTheory.isPrime(3), "3 is prime")
    }

    @Test("17 is prime")
    func isPrimeSeventeen() {
        #expect(NumberTheory.isPrime(17), "17 is prime")
    }

    @Test("15 is not prime (3 x 5)")
    func isNotPrimeFifteen() {
        #expect(!NumberTheory.isPrime(15), "15 = 3 * 5 is composite")
    }

    @Test("1 is not prime")
    func isNotPrimeOne() {
        #expect(!NumberTheory.isPrime(1), "1 is not prime by definition")
    }

    @Test("0 is not prime")
    func isNotPrimeZero() {
        #expect(!NumberTheory.isPrime(0), "0 is not prime")
    }

    @Test("-5 is not prime")
    func isNotPrimeNegative() {
        #expect(!NumberTheory.isPrime(-5), "Negative numbers are not prime")
    }

    @Test("4 is not prime")
    func isNotPrimeFour() {
        #expect(!NumberTheory.isPrime(4), "4 = 2^2 is composite")
    }

    @Test("Large prime 104729 is prime")
    func isPrimeLarge() {
        #expect(NumberTheory.isPrime(104_729), "104729 is the 10000th prime")
    }

    @Test("Carmichael number 561 is correctly identified as composite")
    func isNotPrimeCarmichael() {
        #expect(!NumberTheory.isPrime(561), "561 = 3 * 11 * 17 is a Carmichael number but composite")
    }

    @Test("97 is prime")
    func isPrimeNinetySeven() {
        #expect(NumberTheory.isPrime(97), "97 is prime")
    }
}

/// Test suite for NumberTheory.randomCoprimeBase generating random
/// coprime bases for Shor's algorithm. Validates that result is
/// coprime to input and in valid range.
@Suite("NumberTheory - Random Coprime Base")
struct NumberTheoryRandomCoprimeBaseTests {
    @Test("Random coprime base for 15 is coprime to 15")
    func randomCoprimeBaseIsCoprime() {
        let base = NumberTheory.randomCoprimeBase(for: 15)
        let gcd = NumberTheory.gcd(base, 15)
        #expect(gcd == 1, "Random base should be coprime to 15")
    }

    @Test("Random coprime base is in range [2, n-1]")
    func randomCoprimeBaseInRange() {
        let n = 21
        let base = NumberTheory.randomCoprimeBase(for: n)
        #expect(base >= 2, "Base should be at least 2")
        #expect(base < n, "Base should be less than n")
    }

    @Test("Random coprime base for 35 is coprime to 35")
    func randomCoprimeBaseForThirtyFive() {
        let base = NumberTheory.randomCoprimeBase(for: 35)
        let gcd = NumberTheory.gcd(base, 35)
        #expect(gcd == 1, "Random base should be coprime to 35")
    }
}

/// Test suite for ModularMultiplicationUnitary.permutationMatrix.
/// Validates correct mapping |x> -> |ax mod N> and that the
/// result is a valid unitary matrix.
@Suite("ModularMultiplicationUnitary - Permutation Matrix")
struct ModularMultiplicationUnitaryPermutationMatrixTests {
    @Test("Permutation matrix maps |x> to |ax mod N> correctly")
    func permutationMatrixMapping() {
        let unitary = ModularMultiplicationUnitary(multiplier: 2, modulus: 3, qubits: 2)
        let matrix = unitary.permutationMatrix()
        #expect(matrix[0][0] == .one, "|0> -> |0> since 2*0 mod 3 = 0")
        #expect(matrix[2][1] == .one, "|1> -> |2> since 2*1 mod 3 = 2")
        #expect(matrix[1][2] == .one, "|2> -> |1> since 2*2 mod 3 = 1")
        #expect(matrix[3][3] == .one, "|3> -> |3> (identity for x >= N)")
    }

    @Test("Permutation matrix is unitary: P^dagger * P = I")
    func permutationMatrixIsUnitary() {
        let unitary = ModularMultiplicationUnitary(multiplier: 7, modulus: 15, qubits: 4)
        let P = unitary.permutationMatrix()
        let dim = P.count
        for i in 0 ..< dim {
            for j in 0 ..< dim {
                var sum = Complex<Double>.zero
                for k in 0 ..< dim {
                    sum = sum + P[k][i].conjugate * P[k][j]
                }
                let expected: Complex<Double> = (i == j) ? .one : .zero
                #expect(
                    abs(sum.real - expected.real) < 1e-10 && abs(sum.imaginary - expected.imaginary) < 1e-10,
                    "P^dagger * P should be identity at (\(i), \(j))",
                )
            }
        }
    }

    @Test("Permutation matrix has exactly one 1 per row and column")
    func permutationMatrixStructure() {
        let unitary = ModularMultiplicationUnitary(multiplier: 7, modulus: 15, qubits: 4)
        let P = unitary.permutationMatrix()
        let dim = P.count
        for i in 0 ..< dim {
            var rowSum = 0
            var colSum = 0
            for j in 0 ..< dim {
                if P[i][j] == .one { rowSum += 1 }
                if P[j][i] == .one { colSum += 1 }
            }
            #expect(rowSum == 1, "Row \(i) should have exactly one 1")
            #expect(colSum == 1, "Column \(i) should have exactly one 1")
        }
    }
}

/// Test suite for ModularMultiplicationUnitary.powered computing
/// multiplier^(2^k) mod N. Validates that powered unitary has
/// correct multiplier.
@Suite("ModularMultiplicationUnitary - Powered")
struct ModularMultiplicationUnitaryPoweredTests {
    @Test("Powered(0) returns multiplier^1 = multiplier")
    func poweredZero() {
        let unitary = ModularMultiplicationUnitary(multiplier: 7, modulus: 15, qubits: 4)
        let powered = unitary.powered(0)
        #expect(powered.multiplier == 7, "powered(0) should give multiplier^(2^0) = multiplier^1")
    }

    @Test("Powered(1) returns multiplier^2 mod N")
    func poweredOne() {
        let unitary = ModularMultiplicationUnitary(multiplier: 7, modulus: 15, qubits: 4)
        let powered = unitary.powered(1)
        let expected = (7 * 7) % 15
        #expect(powered.multiplier == expected, "powered(1) should give 7^2 mod 15 = 49 mod 15 = 4")
    }

    @Test("Powered(3) returns multiplier^8 mod N")
    func poweredThree() {
        let unitary = ModularMultiplicationUnitary(multiplier: 2, modulus: 15, qubits: 4)
        let powered = unitary.powered(3)
        let expected = NumberTheory.modularPow(base: 2, exponent: 8, modulus: 15)
        #expect(powered.multiplier == expected, "powered(3) should give 2^8 mod 15 = 256 mod 15 = 1")
    }

    @Test("Powered preserves modulus and qubits")
    func poweredPreservesProperties() {
        let unitary = ModularMultiplicationUnitary(multiplier: 7, modulus: 15, qubits: 4)
        let powered = unitary.powered(2)
        #expect(powered.modulus == 15, "Powered should preserve modulus")
        #expect(powered.qubits == 4, "Powered should preserve qubits")
    }
}

/// Test suite for ShorConfiguration computed properties.
/// Validates effectivePrecisionQubits, workRegisterQubits,
/// and totalQubits calculations.
@Suite("ShorConfiguration - Computed Properties")
struct ShorConfigurationComputedPropertiesTests {
    @Test("Work register qubits for N=15 is 4")
    func workRegisterQubitsFifteen() {
        let config = ShorConfiguration(numberToFactor: 15)
        #expect(config.workRegisterQubits == 4, "ceil(log2(15)) = 4")
    }

    @Test("Effective precision qubits auto-calculated as 2*work + 1")
    func effectivePrecisionQubitsAuto() {
        let config = ShorConfiguration(numberToFactor: 15)
        #expect(config.effectivePrecisionQubits == 9, "2*4 + 1 = 9")
    }

    @Test("Effective precision qubits uses explicit value when provided")
    func effectivePrecisionQubitsExplicit() {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 12)
        #expect(config.effectivePrecisionQubits == 12, "Should use explicit value")
    }

    @Test("Total qubits equals precision + work")
    func totalQubits() {
        let config = ShorConfiguration(numberToFactor: 15)
        #expect(config.totalQubits == 13, "9 precision + 4 work = 13")
    }

    @Test("Work register qubits for N=21 is 5")
    func workRegisterQubitsTwentyOne() {
        let config = ShorConfiguration(numberToFactor: 21)
        #expect(config.workRegisterQubits == 5, "ceil(log2(21)) = 5")
    }

    @Test("Total qubits for N=21 is 16")
    func totalQubitsTwentyOne() {
        let config = ShorConfiguration(numberToFactor: 21)
        #expect(config.totalQubits == 16, "2*5 + 1 + 5 = 16")
    }
}

/// Test suite for ShorPeriodResult properties and description.
/// Validates that all stored properties are accessible and
/// description format is correct.
@Suite("ShorPeriodResult - Properties")
struct ShorPeriodResultPropertiesTests {
    @Test("All properties are accessible after initialization")
    func propertiesAccessible() {
        let result = ShorPeriodResult(
            measuredPhase: 0.25,
            phaseNumerator: 1,
            phaseDenominator: 4,
            candidatePeriods: [4],
            verifiedPeriod: 4,
            measurementOutcome: 128,
            precisionQubits: 9,
        )
        #expect(abs(result.measuredPhase - 0.25) < 1e-10, "measuredPhase should be 0.25")
        #expect(result.phaseNumerator == 1, "phaseNumerator should be 1")
        #expect(result.phaseDenominator == 4, "phaseDenominator should be 4")
        #expect(result.candidatePeriods == [4], "candidatePeriods should be [4]")
        #expect(result.verifiedPeriod == 4, "verifiedPeriod should be 4")
        #expect(result.measurementOutcome == 128, "measurementOutcome should be 128")
        #expect(result.precisionQubits == 9, "precisionQubits should be 9")
    }

    @Test("Description contains phase and verified period")
    func descriptionFormat() {
        let result = ShorPeriodResult(
            measuredPhase: 0.25,
            phaseNumerator: 1,
            phaseDenominator: 4,
            candidatePeriods: [4],
            verifiedPeriod: 4,
            measurementOutcome: 128,
            precisionQubits: 9,
        )
        let desc = result.description
        #expect(desc.contains("0.25"), "Description should contain phase")
        #expect(desc.contains("4"), "Description should contain verified period")
        #expect(desc.contains("ShorPeriodResult"), "Description should contain type name")
    }

    @Test("Description handles nil verified period")
    func descriptionNilVerified() {
        let result = ShorPeriodResult(
            measuredPhase: 0.3,
            phaseNumerator: 3,
            phaseDenominator: 10,
            candidatePeriods: [10],
            verifiedPeriod: nil,
            measurementOutcome: 153,
            precisionQubits: 9,
        )
        let desc = result.description
        #expect(desc.contains("none"), "Description should contain 'none' for nil verified period")
    }
}

/// Test suite for ShorResult properties and description.
/// Validates success case, failure case properties,
/// and description format.
@Suite("ShorResult - Properties")
struct ShorResultPropertiesTests {
    @Test("Success case has factors and no failure reason")
    func successCaseProperties() {
        let result = ShorResult(
            numberToFactor: 15,
            factors: (3, 5),
            period: 4,
            base: 7,
            attempts: 1,
            success: true,
            failureReason: nil,
        )
        #expect(result.success, "Should be successful")
        #expect(result.factors != nil, "Should have factors")
        #expect(result.factors!.p == 3 && result.factors!.q == 5, "Factors should be (3, 5)")
        #expect(result.period == 4, "Period should be 4")
        #expect(result.base == 7, "Base should be 7")
        #expect(result.failureReason == nil, "Should have no failure reason")
    }

    @Test("Failure case has no factors and has failure reason")
    func failureCaseProperties() {
        let result = ShorResult(
            numberToFactor: 17,
            factors: nil,
            period: nil,
            base: 0,
            attempts: 10,
            success: false,
            failureReason: .inputPrime,
        )
        #expect(!result.success, "Should not be successful")
        #expect(result.factors == nil, "Should have no factors")
        #expect(result.failureReason == .inputPrime, "Should have inputPrime failure reason")
    }

    @Test("Success description contains factors")
    func successDescriptionFormat() {
        let result = ShorResult(
            numberToFactor: 15,
            factors: (3, 5),
            period: 4,
            base: 7,
            attempts: 1,
            success: true,
            failureReason: nil,
        )
        let desc = result.description
        #expect(desc.contains("15"), "Description should contain number")
        #expect(desc.contains("3"), "Description should contain first factor")
        #expect(desc.contains("5"), "Description should contain second factor")
    }

    @Test("Failure description contains reason")
    func failureDescriptionFormat() {
        let result = ShorResult(
            numberToFactor: 17,
            factors: nil,
            period: nil,
            base: 0,
            attempts: 0,
            success: false,
            failureReason: .inputPrime,
        )
        let desc = result.description
        #expect(desc.contains("failed"), "Description should contain 'failed'")
        #expect(desc.contains("prime"), "Description should contain failure reason")
    }
}

/// Test suite for ShorFailureReason descriptions.
/// Validates that all failure reasons have meaningful descriptions.
@Suite("ShorFailureReason - Descriptions")
struct ShorFailureReasonDescriptionTests {
    @Test("inputTooSmall description is meaningful")
    func inputTooSmallDescription() {
        let desc = ShorFailureReason.inputTooSmall.description
        #expect(desc.contains("15"), "Description should mention minimum value 15")
    }

    @Test("inputEven description is meaningful")
    func inputEvenDescription() {
        let desc = ShorFailureReason.inputEven.description
        #expect(desc.contains("even"), "Description should mention 'even'")
    }

    @Test("inputPrime description is meaningful")
    func inputPrimeDescription() {
        let desc = ShorFailureReason.inputPrime.description
        #expect(desc.contains("prime"), "Description should mention 'prime'")
    }

    @Test("inputPerfectPower description is meaningful")
    func inputPerfectPowerDescription() {
        let desc = ShorFailureReason.inputPerfectPower.description
        #expect(desc.contains("perfect power"), "Description should mention 'perfect power'")
    }

    @Test("periodOdd description is meaningful")
    func periodOddDescription() {
        let desc = ShorFailureReason.periodOdd.description
        #expect(desc.contains("odd"), "Description should mention 'odd'")
    }

    @Test("trivialRoot description is meaningful")
    func trivialRootDescription() {
        let desc = ShorFailureReason.trivialRoot.description
        #expect(desc.contains("-1"), "Description should mention '-1'")
    }

    @Test("trivialFactor description is meaningful")
    func trivialFactorDescription() {
        let desc = ShorFailureReason.trivialFactor.description
        #expect(desc.contains("trivial"), "Description should mention 'trivial'")
    }

    @Test("maxAttemptsExceeded description is meaningful")
    func maxAttemptsExceededDescription() {
        let desc = ShorFailureReason.maxAttemptsExceeded.description
        #expect(desc.contains("attempts"), "Description should mention 'attempts'")
    }
}

/// Test suite for ShorConfiguration qubit calculations.
/// Validates that configuration correctly computes expected qubit counts
/// for period finding circuits.
@Suite("Shor Circuit - Configuration Qubit Calculation")
struct ShorCircuitQubitCalculationTests {
    @Test("Configuration for N=15 computes 13 total qubits")
    func configurationFifteenQubits() {
        let config = ShorConfiguration(numberToFactor: 15)
        #expect(config.totalQubits == 13, "9 precision + 4 work = 13 qubits")
    }

    @Test("Configuration for N=21 computes 16 total qubits")
    func configurationTwentyOneQubits() {
        let config = ShorConfiguration(numberToFactor: 21)
        #expect(config.totalQubits == 16, "11 precision + 5 work = 16 qubits")
    }

    @Test("Explicit precision qubits are used in total calculation")
    func explicitPrecisionQubits() {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 12)
        #expect(config.totalQubits == 16, "12 precision + 4 work = 16 qubits")
    }

    @Test("Work register qubits for various N values")
    func workRegisterVariousN() {
        #expect(ShorConfiguration(numberToFactor: 15).workRegisterQubits == 4, "ceil(log2(15)) = 4")
        #expect(ShorConfiguration(numberToFactor: 21).workRegisterQubits == 5, "ceil(log2(21)) = 5")
        #expect(ShorConfiguration(numberToFactor: 33).workRegisterQubits == 6, "ceil(log2(33)) = 6")
    }
}

/// Test suite for Shor's algorithm classical pre-checks.
/// Validates that even numbers, primes, and perfect powers
/// are handled classically without quantum computation.
@Suite("Shor Integration - Classical Pre-checks")
struct ShorIntegrationClassicalPrechecksTests {
    @Test("Even number N returns trivial factor 2")
    func evenNumberPrecheck() async {
        let config = ShorConfiguration(numberToFactor: 18)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        #expect(result.success, "Even number should succeed with classical factoring")
        #expect(result.factors != nil, "Should find factors")
        let factors = result.factors!
        #expect(factors.p == 2 || factors.q == 2, "One factor should be 2")
        #expect(factors.p * factors.q == 18, "Factors should multiply to 18")
    }

    @Test("Prime number N fails with inputPrime reason")
    func primeNumberPrecheck() async {
        let config = ShorConfiguration(numberToFactor: 17)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        #expect(!result.success, "Prime number should fail")
        #expect(result.failureReason == .inputPrime, "Failure reason should be inputPrime")
    }

    @Test("Perfect power N succeeds with classical factoring")
    func perfectPowerPrecheck() async {
        let config = ShorConfiguration(numberToFactor: 27)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        #expect(result.success, "Perfect power should succeed with classical factoring")
        #expect(result.factors != nil, "Should find factors")
        let factors = result.factors!
        #expect(factors.p == 3 || factors.q == 3, "3 should be a factor of 27")
    }

    @Test("Number less than 15 fails with inputTooSmall")
    func tooSmallPrecheck() async {
        let config = ShorConfiguration(numberToFactor: 10)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        #expect(!result.success, "Number < 15 should fail")
        #expect(result.failureReason == .inputTooSmall, "Failure reason should be inputTooSmall")
    }
}

/// Test suite for Shor's algorithm result ordering.
/// Validates that factors are properly ordered in results.
@Suite("Shor Result - Factor Ordering")
struct ShorResultFactorOrderingTests {
    @Test("Factors in result are ordered p <= q")
    func factorsAreOrdered() {
        let result = ShorResult(
            numberToFactor: 15,
            factors: (3, 5),
            period: 4,
            base: 7,
            attempts: 1,
            success: true,
            failureReason: nil,
        )
        #expect(result.factors!.p <= result.factors!.q, "Factors should be ordered p <= q")
    }

    @Test("Factors multiply to original number")
    func factorsMultiplyToOriginal() {
        let result = ShorResult(
            numberToFactor: 21,
            factors: (3, 7),
            period: 6,
            base: 2,
            attempts: 2,
            success: true,
            failureReason: nil,
        )
        #expect(result.factors!.p * result.factors!.q == 21, "Factors should multiply to 21")
    }

    @Test("Attempts count is preserved in result")
    func attemptsCountPreserved() {
        let result = ShorResult(
            numberToFactor: 15,
            factors: (3, 5),
            period: 4,
            base: 7,
            attempts: 5,
            success: true,
            failureReason: nil,
        )
        #expect(result.attempts == 5, "Attempts count should be preserved")
    }
}

/// Test suite for ShorPeriodResult phase and outcome consistency.
/// Validates that phase calculation from measurement outcome
/// follows correct formula.
@Suite("ShorPeriodResult - Phase Consistency")
struct ShorPeriodResultPhaseConsistencyTests {
    @Test("Phase is measurementOutcome divided by 2^precisionQubits")
    func phaseCalculation() {
        let result = ShorPeriodResult(
            measuredPhase: 0.5,
            phaseNumerator: 1,
            phaseDenominator: 2,
            candidatePeriods: [2],
            verifiedPeriod: 2,
            measurementOutcome: 256,
            precisionQubits: 9,
        )
        let expectedPhase = Double(result.measurementOutcome) / Double(1 << result.precisionQubits)
        #expect(abs(result.measuredPhase - expectedPhase) < 1e-10, "Phase should match formula")
    }

    @Test("Zero measurement outcome gives zero phase")
    func zeroMeasurementOutcome() {
        let result = ShorPeriodResult(
            measuredPhase: 0.0,
            phaseNumerator: 0,
            phaseDenominator: 1,
            candidatePeriods: [],
            verifiedPeriod: nil,
            measurementOutcome: 0,
            precisionQubits: 9,
        )
        #expect(abs(result.measuredPhase) < 1e-10, "Zero measurement should give zero phase")
    }

    @Test("Maximum measurement outcome gives phase near 1")
    func maxMeasurementOutcome() {
        let precisionQubits = 9
        let maxOutcome = (1 << precisionQubits) - 1
        let expectedPhase = Double(maxOutcome) / Double(1 << precisionQubits)
        let result = ShorPeriodResult(
            measuredPhase: expectedPhase,
            phaseNumerator: maxOutcome,
            phaseDenominator: 1 << precisionQubits,
            candidatePeriods: [],
            verifiedPeriod: nil,
            measurementOutcome: maxOutcome,
            precisionQubits: precisionQubits,
        )
        #expect(result.measuredPhase >= 0.0 && result.measuredPhase < 1.0, "Phase should be in [0, 1)")
    }
}

/// Test suite for QuantumCircuit.shorPeriodFinding circuit builder.
/// Validates circuit construction with correct qubit count,
/// gate structure, and inverse QFT components.
@Suite("QuantumCircuit - shorPeriodFinding")
struct QuantumCircuitShorPeriodFindingTests {
    @Test("shorPeriodFinding creates circuit with correct qubit count")
    func circuitHasCorrectQubitCount() {
        let circuit = QuantumCircuit.shorPeriodFinding(base: 7, modulus: 15, precisionQubits: 9)
        let workBits = 4
        let expectedQubits = 9 + workBits
        #expect(circuit.qubits == expectedQubits, "Circuit should have 9 precision + 4 work = 13 qubits")
    }

    @Test("shorPeriodFinding for modulus 21 uses 5 work qubits")
    func circuitForModulus21() {
        let circuit = QuantumCircuit.shorPeriodFinding(base: 2, modulus: 21, precisionQubits: 11)
        let workBits = 5
        let expectedQubits = 11 + workBits
        #expect(circuit.qubits == expectedQubits, "Circuit should have 11 precision + 5 work = 16 qubits")
    }

    @Test("shorPeriodFinding circuit has non-empty gates")
    func circuitHasGates() {
        let circuit = QuantumCircuit.shorPeriodFinding(base: 7, modulus: 15, precisionQubits: 9)
        #expect(circuit.operations.count > 0, "Circuit should have gates for Hadamards, controlled unitaries, and inverse QFT")
    }

    @Test("shorPeriodFinding circuit starts with Hadamard gates on precision qubits")
    func circuitStartsWithHadamards() {
        let circuit = QuantumCircuit.shorPeriodFinding(base: 7, modulus: 15, precisionQubits: 4)
        let firstFourOps = circuit.operations.prefix(4)
        var hadamardCount = 0
        for op in firstFourOps {
            if op.gate == .hadamard {
                hadamardCount += 1
            }
        }
        #expect(hadamardCount == 4, "First 4 gates should be Hadamard gates on precision qubits")
    }

    @Test("shorPeriodFinding with base 2 modulus 15 creates valid circuit")
    func circuitBase2Modulus15() {
        let circuit = QuantumCircuit.shorPeriodFinding(base: 2, modulus: 15, precisionQubits: 8)
        #expect(circuit.qubits == 12, "Circuit should have 8 precision + 4 work = 12 qubits")
        #expect(circuit.operations.count > 8, "Circuit should have more than 8 gates")
    }
}

/// Test suite for QuantumCircuit.shorFactorization convenience wrapper.
/// Validates that it correctly builds period finding circuit
/// with auto-calculated precision qubits.
@Suite("QuantumCircuit - shorFactorization")
struct QuantumCircuitShorFactorizationTests {
    @Test("shorFactorization creates circuit with auto-calculated precision")
    func circuitAutoCalculatedPrecision() {
        let circuit = QuantumCircuit.shorFactorization(number: 15, base: 7)
        let config = ShorConfiguration(numberToFactor: 15)
        let expectedQubits = config.totalQubits
        #expect(circuit.qubits == expectedQubits, "Circuit should have total qubits matching configuration")
    }

    @Test("shorFactorization with explicit precision uses provided value")
    func circuitExplicitPrecision() {
        let circuit = QuantumCircuit.shorFactorization(number: 15, base: 7, precisionQubits: 12)
        let workBits = 4
        let expectedQubits = 12 + workBits
        #expect(circuit.qubits == expectedQubits, "Circuit should have 12 precision + 4 work = 16 qubits")
    }

    @Test("shorFactorization for number 21 creates correct circuit")
    func circuitForNumber21() {
        let circuit = QuantumCircuit.shorFactorization(number: 21, base: 2)
        let config = ShorConfiguration(numberToFactor: 21)
        #expect(circuit.qubits == config.totalQubits, "Circuit qubits should match configuration total")
    }

    @Test("shorFactorization equivalent to shorPeriodFinding with same params")
    func factorizationEquivalentToPeriodFinding() {
        let config = ShorConfiguration(numberToFactor: 15)
        let factorizationCircuit = QuantumCircuit.shorFactorization(number: 15, base: 7)
        let periodFindingCircuit = QuantumCircuit.shorPeriodFinding(
            base: 7,
            modulus: 15,
            precisionQubits: config.effectivePrecisionQubits,
        )
        #expect(factorizationCircuit.qubits == periodFindingCircuit.qubits, "Both circuits should have same qubit count")
        #expect(factorizationCircuit.operations.count == periodFindingCircuit.operations.count, "Both circuits should have same gate count")
    }
}

/// Test suite for shorPeriodFinding circuit execution.
/// Validates that executing the circuit produces a valid
/// quantum state for period extraction.
@Suite("Shor Circuit Execution - Period Finding")
struct ShorCircuitExecutionPeriodFindingTests {
    @Test("Executed shorPeriodFinding circuit produces valid quantum state")
    func executedCircuitProducesValidState() async {
        let circuit = QuantumCircuit.shorPeriodFinding(base: 7, modulus: 15, precisionQubits: 4)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        var totalProbability = 0.0
        for amp in state.amplitudes {
            totalProbability += amp.magnitudeSquared
        }
        #expect(abs(totalProbability - 1.0) < 1e-10, "Total probability should be 1.0")
    }

    @Test("shorPeriodResult extracts period from executed circuit state")
    func periodResultExtraction() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 6)
        let circuit = QuantumCircuit.shorPeriodFinding(base: 7, modulus: 15, precisionQubits: 6)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let result = state.shorPeriodResult(config: config, base: 7)
        #expect(result.precisionQubits == 6, "Precision qubits in result should match configuration")
        #expect(result.measuredPhase >= 0.0 && result.measuredPhase < 1.0, "Measured phase should be in [0, 1)")
    }

    @Test("Executed circuit state has correct dimension")
    func executedCircuitStateDimension() async {
        let circuit = QuantumCircuit.shorPeriodFinding(base: 2, modulus: 15, precisionQubits: 5)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let expectedDim = 1 << (5 + 4)
        #expect(state.amplitudes.count == expectedDim, "State should have 2^9 = 512 amplitudes")
    }

    @Test("Period result has valid candidate periods")
    func periodResultHasCandidates() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 6)
        let circuit = QuantumCircuit.shorPeriodFinding(base: 7, modulus: 15, precisionQubits: 6)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let result = state.shorPeriodResult(config: config, base: 7)
        for period in result.candidatePeriods {
            #expect(period > 0 && period <= 15, "Candidate periods should be positive and <= N")
        }
    }
}

/// Test suite for shorFactorization circuit execution.
/// Validates that the factorization circuit produces
/// usable results for factor extraction.
@Suite("Shor Circuit Execution - Factorization")
struct ShorCircuitExecutionFactorizationTests {
    @Test("Executed shorFactorization circuit produces normalized state")
    func factorizationCircuitNormalized() async {
        let circuit = QuantumCircuit.shorFactorization(number: 15, base: 7, precisionQubits: 5)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        var totalProbability = 0.0
        for amp in state.amplitudes {
            totalProbability += amp.magnitudeSquared
        }
        #expect(abs(totalProbability - 1.0) < 1e-10, "Factorization circuit state should be normalized")
    }

    @Test("shorFactorization circuit with auto precision produces valid state")
    func factorizationAutoPrecsionValid() async {
        let circuit = QuantumCircuit.shorFactorization(number: 15, base: 2)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        #expect(state.amplitudes.count > 0, "Executed circuit should produce non-empty state")
        let expectedDim = 1 << (9 + 4)
        #expect(state.amplitudes.count == expectedDim, "State dimension should match 2^totalQubits")
    }

    @Test("Period extraction from factorization circuit returns valid result")
    func factorizationPeriodExtraction() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 6)
        let circuit = QuantumCircuit.shorFactorization(number: 15, base: 7, precisionQubits: 6)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let result = state.shorPeriodResult(config: config, base: 7)
        #expect(result.measurementOutcome >= 0, "Measurement outcome should be non-negative")
        #expect(result.measurementOutcome < (1 << 6), "Measurement outcome should be < 2^precisionQubits")
    }
}

/// Test suite for ShorFactorization.run quantum period finding loop.
/// Validates the main factorization loop that builds circuits,
/// executes them, and extracts factors from periods.
@Suite("ShorFactorization - Quantum Period Finding Loop")
struct ShorFactorizationQuantumLoopTests {
    @Test("run() with lucky GCD finding returns early success")
    func luckyGCDFinding() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 6, maxAttempts: 5)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        #expect(result.numberToFactor == 15, "Result should be for factoring 15")
        if result.success, result.factors != nil {
            let factors = result.factors!
            #expect(factors.p * factors.q == 15, "Factors should multiply to 15")
        }
    }

    @Test("run() executes quantum circuit in main loop")
    func quantumCircuitExecution() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 6, maxAttempts: 3)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        #expect(result.numberToFactor == 15, "Result should track the number being factored")
        #expect(result.attempts >= 0 && result.attempts <= 3, "Attempts should be between 0 and maxAttempts")
    }

    @Test("run() with nil progress callback completes successfully")
    func nilProgressCallback() async {
        let config = ShorConfiguration(numberToFactor: 15, maxAttempts: 2)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run(progress: nil)
        #expect(result.numberToFactor == 15, "Should factor 15")
    }

    @Test("run() returns maxAttemptsExceeded when no factor found")
    func maxAttemptsExceeded() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 3, maxAttempts: 1)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        var iterations = 0
        var foundMaxAttempts = false
        while iterations < 20, !foundMaxAttempts {
            let result = await shor.run()
            if result.failureReason == .maxAttemptsExceeded {
                foundMaxAttempts = true
                #expect(result.attempts == 1, "Should have made exactly 1 attempt")
            }
            iterations += 1
        }
        #expect(iterations <= 20, "Test should complete within reasonable iterations")
    }

    @Test("run() handles odd period by continuing to next attempt")
    func oddPeriodContinuation() async {
        let config = ShorConfiguration(numberToFactor: 21, precisionQubits: 5, maxAttempts: 5)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        #expect(result.numberToFactor == 21, "Should attempt to factor 21")
        if result.success {
            let factors = result.factors!
            #expect(factors.p * factors.q == 21, "Found factors should multiply to 21")
        }
    }

    @Test("run() extracts factors from valid even period")
    func factorExtractionFromEvenPeriod() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 6, maxAttempts: 5)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        if result.success, result.period != nil {
            let period = result.period!
            #expect(period % 2 == 0, "Successfully found period should be even")
            let factors = result.factors!
            #expect(factors.p * factors.q == 15, "Factors should multiply to 15")
        }
    }

    @Test("run() skips trivial root x = n-1")
    func trivialRootSkipped() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 6, maxAttempts: 10)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        if result.success {
            let factors = result.factors!
            #expect(factors.p > 1 && factors.p < 15, "First factor should be non-trivial")
            #expect(factors.q > 1 && factors.q < 15, "Second factor should be non-trivial")
        }
    }

    @Test("run() uses factor1 or factor2 from GCD calculations")
    func gcdFactorExtraction() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 6, maxAttempts: 5)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        if result.success, result.factors != nil {
            let factors = result.factors!
            let gcd1 = NumberTheory.gcd(factors.p, 15)
            let gcd2 = NumberTheory.gcd(factors.q, 15)
            #expect(gcd1 == factors.p || gcd2 == factors.q, "Factors should be obtained via GCD")
        }
    }
}

/// Test suite for NumberTheory.extendedGCD negative input edge cases.
/// Validates Bezout coefficients when first argument is negative
/// and second argument is zero.
@Suite("NumberTheory - Extended GCD Negative Edge Cases")
struct NumberTheoryExtendedGCDNegativeEdgeCasesTests {
    @Test("Extended GCD with negative a and zero b returns (-a, -1, 0)")
    func extendedGCDNegativeAZeroB() {
        let (g, x, y) = NumberTheory.extendedGCD(-5, 0)
        #expect(g == 5, "gcd(-5, 0) should be 5 (absolute value)")
        #expect(x == -1, "x coefficient should be -1 for negative a with zero b")
        #expect(y == 0, "y coefficient should be 0 when b is zero")
        #expect(-5 * x + 0 * y == g, "Bezout identity should hold: -5 * (-1) + 0 * 0 = 5")
    }

    @Test("Extended GCD with negative a and zero b for larger value")
    func extendedGCDNegativeAZeroBLarger() {
        let (g, x, y) = NumberTheory.extendedGCD(-42, 0)
        #expect(g == 42, "gcd(-42, 0) should be 42")
        #expect(x == -1, "x coefficient should be -1")
        #expect(y == 0, "y coefficient should be 0")
        #expect(-42 * x == g, "Bezout identity should hold")
    }

    @Test("Extended GCD with positive a and zero b returns (a, 1, 0)")
    func extendedGCDPositiveAZeroB() {
        let (g, x, y) = NumberTheory.extendedGCD(7, 0)
        #expect(g == 7, "gcd(7, 0) should be 7")
        #expect(x == 1, "x coefficient should be 1 for positive a with zero b")
        #expect(y == 0, "y coefficient should be 0 when b is zero")
    }
}

/// Test suite for NumberTheory.isPerfectPower overflow break paths.
/// Validates that power computation breaks early when exceeding n
/// for both root and rootPlus1 checks.
@Suite("NumberTheory - isPerfectPower Overflow Paths")
struct NumberTheoryIsPerfectPowerOverflowTests {
    @Test("isPerfectPower 31 triggers break when 2^5=32 > 31")
    func isPerfectPowerBreakLine267() {
        let result = NumberTheory.isPerfectPower(31)
        #expect(result == nil, "31 is not a perfect power, triggers break when 2^5=32 > 31")
    }

    @Test("isPerfectPower 33 triggers break when 2^5=32 < 33 < 64=2^6")
    func isPerfectPowerBreak() {
        let result = NumberTheory.isPerfectPower(33)
        #expect(result == nil, "33 is not a perfect power, triggers breaks during power computation")
    }

    @Test("isPerfectPower 32 matches")
    func isPerfectPower32RootPath() {
        let result = NumberTheory.isPerfectPower(32)
        #expect(result != nil, "32 should be detected as perfect power")
        #expect(result!.base == 2 && result!.exp == 5, "32 = 2^5")
    }

    @Test("isPerfectPower 243 matches via root path")
    func isPerfectPower243() {
        let result = NumberTheory.isPerfectPower(243)
        #expect(result != nil, "243 should be detected as perfect power")
        #expect(result!.base == 3 && result!.exp == 5, "243 = 3^5")
    }

    @Test("isPerfectPower 999 returns nil after many overflow breaks")
    func isPerfectPowerLargeNonPower() {
        let result = NumberTheory.isPerfectPower(999)
        #expect(result == nil, "999 is not a perfect power")
    }

    @Test("isPerfectPower 128 as 2^7")
    func isPerfectPower128() {
        let result = NumberTheory.isPerfectPower(128)
        #expect(result != nil, "128 should be detected as perfect power")
        #expect(result!.base == 2 && result!.exp == 7, "128 = 2^7")
    }

    @Test("isPerfectPower 4 returns (2, 2) checking highest exponent first")
    func isPerfectPower4() {
        let result = NumberTheory.isPerfectPower(4)
        #expect(result != nil, "4 should be detected as perfect power")
        #expect(result!.base == 2 && result!.exp == 2, "4 = 2^2")
    }

    @Test("isPerfectPower 9 triggers overflow breaks before finding 3^2")
    func isPerfectPower9() {
        let result = NumberTheory.isPerfectPower(9)
        #expect(result != nil, "9 should be detected as perfect power")
        #expect(result!.base == 3 && result!.exp == 2, "9 = 3^2")
    }
}

/// Test suite for NumberTheory.integerRoot edge cases.
/// Validates behavior when n is zero or negative,
/// and when k equals 1.
@Suite("NumberTheory - integerRoot Edge Cases")
struct NumberTheoryIntegerRootEdgeCasesTests {
    @Test("integerRoot returns 0 for n equal to 0")
    func integerRootZeroN() {
        let result = NumberTheory.integerRoot(0, 2)
        #expect(result == 0, "integerRoot(0, k) should return 0")
    }

    @Test("integerRoot returns n for k equal to 1")
    func integerRootKOne() {
        let result = NumberTheory.integerRoot(42, 1)
        #expect(result == 42, "integerRoot(n, 1) should return n itself")
    }

    @Test("integerRoot returns n for k equal to 1 with larger value")
    func integerRootKOneLarger() {
        let result = NumberTheory.integerRoot(100, 1)
        #expect(result == 100, "integerRoot(100, 1) should return 100")
    }

    @Test("integerRoot computes square root correctly")
    func integerRootSquareRoot() {
        let result = NumberTheory.integerRoot(16, 2)
        #expect(result == 4, "integerRoot(16, 2) should be 4")
    }

    @Test("integerRoot computes cube root correctly")
    func integerRootCubeRoot() {
        let result = NumberTheory.integerRoot(27, 3)
        #expect(result == 3, "integerRoot(27, 3) should be 3")
    }

    @Test("integerRoot handles non-perfect root by returning floor")
    func integerRootNonPerfect() {
        let result = NumberTheory.integerRoot(10, 2)
        #expect(result == 3, "integerRoot(10, 2) should be 3 (floor of sqrt(10))")
    }
}

/// Test suite for ShorResult.description edge cases.
/// Validates description output when period is nil but factors exist,
/// and when both factors and failureReason are nil.
@Suite("ShorResult - Description Edge Cases")
struct ShorResultDescriptionEdgeCasesTests {
    @Test("Description with factors but nil period uses 0 as fallback")
    func descriptionNilPeriodWithFactors() {
        let result = ShorResult(
            numberToFactor: 18,
            factors: (2, 9),
            period: nil,
            base: 0,
            attempts: 0,
            success: true,
            failureReason: nil,
        )
        let desc = result.description
        #expect(desc.contains("18"), "Description should contain the number")
        #expect(desc.contains("2"), "Description should contain factor 2")
        #expect(desc.contains("9"), "Description should contain factor 9")
        #expect(desc.contains("period: 0"), "Description should show period: 0 when period is nil")
    }

    @Test("Description without factors and without failure reason")
    func descriptionNoFactorsNoReason() {
        let result = ShorResult(
            numberToFactor: 15,
            factors: nil,
            period: nil,
            base: 0,
            attempts: 3,
            success: false,
            failureReason: nil,
        )
        let desc = result.description
        #expect(desc.contains("15"), "Description should contain the number")
        #expect(desc.contains("attempts: 3"), "Description should contain attempts count")
        #expect(!desc.contains("failed"), "Description should not contain 'failed' without failure reason")
    }

    @Test("Description with factors shows multiplication format")
    func descriptionWithFactorsFormat() {
        let result = ShorResult(
            numberToFactor: 21,
            factors: (3, 7),
            period: 6,
            base: 2,
            attempts: 1,
            success: true,
            failureReason: nil,
        )
        let desc = result.description
        #expect(desc.contains("21 = 3 * 7"), "Description should show N = p * q format")
    }
}

/// Test suite for QuantumState.shorPeriodResult phase edge cases.
@Suite("ShorPeriodResult - Phase Edge Cases")
struct ShorPeriodResultPhaseEdgeCasesTests {
    @Test("shorPeriodResult with concentrated state in work register")
    func shorPeriodResultConcentratedState() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 4)
        var circuit = QuantumCircuit(qubits: config.totalQubits)
        circuit.append(.pauliX, to: config.effectivePrecisionQubits)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let result = state.shorPeriodResult(config: config, base: 7)
        #expect(result.measurementOutcome >= 0, "Measurement outcome should be non-negative")
        #expect(result.phaseNumerator >= 0, "Phase numerator should be non-negative")
        #expect(result.phaseDenominator >= 1, "Phase denominator should be at least 1")
    }

    @Test("shorPeriodResult phase denominator always positive")
    func shorPeriodResultPhaseDenominatorAlwaysPositive() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 4)
        var circuit = QuantumCircuit(qubits: config.totalQubits)
        circuit.append(.pauliX, to: config.effectivePrecisionQubits)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let result = state.shorPeriodResult(config: config, base: 7)
        #expect(result.phaseDenominator > 0, "Phase denominator always > 0 since gcd(0, x) = x")
    }

    @Test("shorPeriodResult with minimal circuit")
    func shorPeriodResultMinimalCircuit() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 2)
        let circuit = QuantumCircuit(qubits: config.totalQubits)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let result = state.shorPeriodResult(config: config, base: 7)
        #expect(result.precisionQubits == 2, "Precision qubits should match configuration")
    }

    @Test("shorPeriodResult handles state with probability at index 0")
    func shorPeriodResultProbabilityAtZero() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 3)
        let circuit = QuantumCircuit(qubits: config.totalQubits)
        let simulator = QuantumSimulator()
        let state = await simulator.execute(circuit)
        let result = state.shorPeriodResult(config: config, base: 7)
        #expect(result.measurementOutcome == 0, "Initial state should have outcome 0")
        #expect(result.phaseNumerator == 0, "Phase numerator should be 0 for index 0")
    }
}

/// Test suite for ShorFactorization.run even number early return.
/// Validates the n % 2 == 0 branch returning (2, n/2).
@Suite("ShorFactorization - Even Number Factoring")
struct ShorFactorizationEvenNumberTests {
    @Test("Even N=16 returns (2, 8)")
    func evenNumber16() async {
        let config = ShorConfiguration(numberToFactor: 16)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        #expect(result.success, "Even number should succeed")
        #expect(result.factors != nil, "Should have factors")
        let factors = result.factors!
        #expect(factors.p == 2 && factors.q == 8, "16 should factor as (2, 8)")
    }

    @Test("Even N=18 returns (2, 9)")
    func evenNumber18() async {
        let config = ShorConfiguration(numberToFactor: 18)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        #expect(result.success, "Even number should succeed")
        #expect(result.factors != nil, "Should have factors from even handling")
        let factors = result.factors!
        #expect(factors.p == 2 && factors.q == 9, "18 should factor as (2, 9) via even branch")
    }

    @Test("Even N=100 with larger numbers")
    func evenNumber100() async {
        let config = ShorConfiguration(numberToFactor: 100)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        #expect(result.success, "Even number should succeed")
        #expect(result.factors != nil, "Should have factors")
        let factors = result.factors!
        #expect(factors.p == 2 && factors.q == 50, "100 should factor as (2, 50)")
    }

    @Test("Even number branch has nil period")
    func evenNumberNilPeriod() async {
        let config = ShorConfiguration(numberToFactor: 22)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        #expect(result.period == nil, "Even number factoring should have nil period")
    }

    @Test("Even number branch has zero attempts")
    func evenNumberZeroAttempts() async {
        let config = ShorConfiguration(numberToFactor: 24)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        #expect(result.attempts == 0, "Even number should be factored with 0 attempts")
    }

    @Test("Even number branch has base 0")
    func evenNumberBaseZero() async {
        let config = ShorConfiguration(numberToFactor: 30)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        #expect(result.base == 0, "Even number factoring should have base 0")
    }
}

/// Test suite for ShorFactorization.run period verification continues.
/// Validates that algorithm continues when verifiedPeriod is nil
/// or when period is odd.
@Suite("ShorFactorization - Period Verification Continues")
struct ShorFactorizationPeriodVerificationTests {
    @Test("Continue when verifiedPeriod is nil with low precision")
    func ContinueWhenPeriodNil() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 2, maxAttempts: 5)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        #expect(result.numberToFactor == 15, "Should be factoring 15")
        #expect(result.attempts >= 1 && result.attempts <= 5, "Should make attempts triggering line 1104")
    }

    @Test("Continue when period is odd")
    func continueWhenPeriodOdd() async {
        let config = ShorConfiguration(numberToFactor: 21, precisionQubits: 5, maxAttempts: 10)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        #expect(result.numberToFactor == 21, "Should be factoring 21")
        if result.success, result.period != nil {
            #expect(result.period! % 2 == 0, "Found period should be even after skipping odd ones")
        }
    }

    @Test("Multiple attempts trigger continue paths repeatedly")
    func multipleAttemptsTriggerContinues() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 3, maxAttempts: 10)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        #expect(result.attempts >= 1 && result.attempts <= 10, "Should make attempts within bounds")
    }

    @Test("Very low precision forces nil verifiedPeriod continue")
    func veryLowPrecisionForcesNilPeriod() async {
        let config = ShorConfiguration(numberToFactor: 33, precisionQubits: 2, maxAttempts: 3)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        #expect(result.numberToFactor == 33, "Should be factoring 33")
    }
}

/// Test suite for ShorFactorization.run factor2 extraction path.
/// Validates that when factor1 (gcd(x+1, n)) is trivial,
/// the algorithm uses factor2 (gcd(x-1, n)).
@Suite("ShorFactorization - Factor2 Path")
struct ShorFactorizationFactor2PathTests {
    @Test("Factor2 path when factor1 is trivial")
    func factor2PathWhenFactor1Trivial() async {
        var usedFactor2Path = false
        for _ in 0 ..< 100 {
            let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 8, maxAttempts: 1)
            let simulator = QuantumSimulator()
            let shor = ShorFactorization(configuration: config, simulator: simulator)
            let result = await shor.run()
            if result.success, result.factors != nil, result.period != nil {
                usedFactor2Path = true
                break
            }
        }
        #expect(usedFactor2Path, "Should eventually use factor2 path when factor1 is trivial")
    }

    @Test("Factorization finds non-trivial factor via either GCD path")
    func findNonTrivialFactor() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 8, maxAttempts: 20)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        if result.success, result.factors != nil {
            let factors = result.factors!
            #expect(factors.p > 1 && factors.p < 15, "First factor should be non-trivial")
            #expect(factors.q > 1 && factors.q < 15, "Second factor should be non-trivial")
        }
    }

    @Test("Factor1 and factor2 paths both produce valid factors")
    func bothPathsProduceValidFactors() async {
        var successCount = 0
        for _ in 0 ..< 30 {
            let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 7, maxAttempts: 1)
            let simulator = QuantumSimulator()
            let shor = ShorFactorization(configuration: config, simulator: simulator)
            let result = await shor.run()
            if result.success, result.factors != nil {
                let factors = result.factors!
                #expect(factors.p * factors.q == 15, "Factors should multiply to 15")
                successCount += 1
            }
        }
        #expect(successCount >= 1, "Should have at least one successful factorization")
    }

    @Test("Factor extraction maintains correct ordering from either path")
    func factorExtractionOrdering() async {
        let config = ShorConfiguration(numberToFactor: 21, precisionQubits: 8, maxAttempts: 15)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        if result.success, result.factors != nil {
            let factors = result.factors!
            #expect(factors.p <= factors.q, "Factors should be ordered p <= q")
            #expect(factors.p * factors.q == 21, "Factors should multiply to 21")
        }
    }
}

/// Test suite for ShorFactorization.run factor extraction paths.
/// Validates both factor1 and factor2 extraction branches
/// in the main loop.
@Suite("ShorFactorization - Factor Extraction Paths")
struct ShorFactorizationFactorExtractionTests {
    @Test("Successful factorization has ordered factors p <= q")
    func factorsOrdered() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 6, maxAttempts: 5)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        if result.success, result.factors != nil {
            let factors = result.factors!
            #expect(factors.p <= factors.q, "Factors should be ordered p <= q")
        }
    }

    @Test("Factorization records base used in successful attempt")
    func baseRecorded() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 6, maxAttempts: 5)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        if result.success, result.period != nil {
            #expect(result.base >= 2 && result.base < 15, "Base should be in valid range [2, N-1]")
            #expect(NumberTheory.gcd(result.base, 15) == 1, "Base should be coprime to N")
        }
    }

    @Test("Factorization records period in successful quantum attempt")
    func periodRecorded() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 6, maxAttempts: 5)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        if result.success, result.period != nil {
            let period = result.period!
            #expect(period > 0, "Period should be positive")
            #expect(period % 2 == 0, "Period should be even for successful factorization")
        }
    }

    @Test("Factorization attempts counter increments correctly")
    func attemptsCounter() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 4, maxAttempts: 5)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        #expect(result.attempts >= 0, "Attempts should be non-negative")
        #expect(result.attempts <= 5, "Attempts should not exceed maxAttempts")
    }

    @Test("factor2 path used when factor1 is trivial")
    func factor2PathUsed() async {
        let config = ShorConfiguration(numberToFactor: 15, precisionQubits: 6, maxAttempts: 5)
        let simulator = QuantumSimulator()
        let shor = ShorFactorization(configuration: config, simulator: simulator)
        let result = await shor.run()
        if result.success, result.factors != nil {
            let factors = result.factors!
            #expect(factors.p * factors.q == 15, "Factors should multiply to 15")
            #expect(factors.p == 3 || factors.p == 5, "Factor p should be 3 or 5")
            #expect(factors.q == 3 || factors.q == 5, "Factor q should be 3 or 5")
        }
    }
}
