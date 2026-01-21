// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Classical number theory algorithms for Shor's factorization algorithm.
///
/// Provides efficient implementations of essential number-theoretic operations including
/// GCD computation, modular exponentiation, continued fraction expansion, primality testing,
/// and perfect power detection. All methods are pure functions with no side effects.
///
/// **Example:**
/// ```swift
/// let gcd = NumberTheory.gcd(48, 18)  // 6
/// let power = NumberTheory.modularPow(base: 2, exponent: 10, modulus: 1000)  // 24
/// let inverse = NumberTheory.modularInverse(3, modulus: 7)  // 5 (since 3*5 = 15 = 1 mod 7)
/// let isPrime = NumberTheory.isPrime(17)  // true
/// ```
@frozen
public enum NumberTheory {
    /// Computes greatest common divisor using Euclidean algorithm.
    ///
    /// Handles negative inputs by taking absolute values. Returns 0 only when both inputs are 0.
    ///
    /// **Example:**
    /// ```swift
    /// let result = NumberTheory.gcd(48, 18)  // 6
    /// let negativeCase = NumberTheory.gcd(-12, 8)  // 4
    /// ```
    ///
    /// - Parameters:
    ///   - a: First integer
    ///   - b: Second integer
    /// - Returns: Greatest common divisor of a and b
    /// - Complexity: O(log(min(a, b)))
    @inlinable
    @_optimize(speed)
    @_effects(readonly)
    public static func gcd(_ a: Int, _ b: Int) -> Int {
        var x = a < 0 ? -a : a
        var y = b < 0 ? -b : b
        while y != 0 {
            let temp = y
            y = x % y
            x = temp
        }
        return x
    }

    /// Computes extended GCD returning Bezout coefficients.
    ///
    /// Returns (gcd, x, y) such that gcd = a*x + b*y (Bezout's identity).
    ///
    /// **Example:**
    /// ```swift
    /// let (g, x, y) = NumberTheory.extendedGCD(48, 18)
    /// // g = 6, x = -1, y = 3, since 48*(-1) + 18*3 = 6
    /// ```
    ///
    /// - Parameters:
    ///   - a: First integer
    ///   - b: Second integer
    /// - Returns: Tuple (gcd, x, y) where gcd = a*x + b*y
    /// - Complexity: O(log(min(a, b)))
    @inlinable
    @_optimize(speed)
    @_effects(readonly)
    public static func extendedGCD(_ a: Int, _ b: Int) -> (gcd: Int, x: Int, y: Int) {
        guard b != 0 else { return (a < 0 ? -a : a, a < 0 ? -1 : 1, 0) }

        var oldR = a, r = b
        var oldS = 1, s = 0
        var oldT = 0, t = 1

        while r != 0 {
            let quotient = oldR / r
            (oldR, r) = (r, oldR - quotient * r)
            (oldS, s) = (s, oldS - quotient * s)
            (oldT, t) = (t, oldT - quotient * t)
        }

        if oldR < 0 {
            return (-oldR, -oldS, -oldT)
        }
        return (oldR, oldS, oldT)
    }

    /// Computes modular exponentiation using repeated squaring.
    ///
    /// Efficiently computes base^exponent mod modulus in O(log exponent) time.
    ///
    /// **Example:**
    /// ```swift
    /// let result = NumberTheory.modularPow(base: 2, exponent: 10, modulus: 1000)  // 24
    /// let large = NumberTheory.modularPow(base: 7, exponent: 256, modulus: 13)  // 9
    /// ```
    ///
    /// - Parameters:
    ///   - base: Base value
    ///   - exponent: Exponent (must be non-negative)
    ///   - modulus: Modulus (must be positive)
    /// - Returns: base^exponent mod modulus
    /// - Complexity: O(log exponent)
    @inlinable
    @_optimize(speed)
    @_effects(readonly)
    public static func modularPow(base: Int, exponent: Int, modulus: Int) -> Int {
        ValidationUtilities.validateNonNegativeInt(exponent, name: "Exponent")
        ValidationUtilities.validatePositiveInt(modulus, name: "Modulus")

        guard modulus != 1 else { return 0 }
        guard exponent != 0 else { return 1 }

        var result = 1
        var b = base % modulus
        if b < 0 { b += modulus }
        var exp = exponent

        while exp > 0 {
            if exp & 1 == 1 {
                result = (result * b) % modulus
            }
            b = (b * b) % modulus
            exp >>= 1
        }

        return result
    }

    /// Computes modular multiplicative inverse using extended GCD.
    ///
    /// Returns the value x such that (a * x) mod modulus = 1, or nil if no inverse exists
    /// (when gcd(a, modulus) != 1).
    ///
    /// **Example:**
    /// ```swift
    /// let inv = NumberTheory.modularInverse(3, modulus: 7)  // 5 (since 3*5 = 15 = 1 mod 7)
    /// let noInv = NumberTheory.modularInverse(4, modulus: 8)  // nil (gcd(4,8) = 4 != 1)
    /// ```
    ///
    /// - Parameters:
    ///   - a: Value to invert
    ///   - modulus: Modulus (must be positive)
    /// - Returns: Modular inverse if it exists, nil otherwise
    /// - Complexity: O(log modulus)
    @inlinable
    @_optimize(speed)
    @_effects(readonly)
    public static func modularInverse(_ a: Int, modulus: Int) -> Int? {
        ValidationUtilities.validatePositiveInt(modulus, name: "Modulus")

        let (g, x, _) = extendedGCD(a, modulus)
        guard g == 1 else { return nil }

        var result = x % modulus
        if result < 0 { result += modulus }
        return result
    }

    /// Computes continued fraction expansion of a rational number.
    ///
    /// Returns the sequence of partial quotients [a_0; a_1, a_2, ...] such that
    /// numerator/denominator = a_0 + 1/(a_1 + 1/(a_2 + ...)).
    ///
    /// **Example:**
    /// ```swift
    /// let expansion = NumberTheory.continuedFractionExpansion(355, 113, maxTerms: 10)
    /// // [3, 7, 16] since 355/113 = 3 + 1/(7 + 1/16)
    /// ```
    ///
    /// - Parameters:
    ///   - numerator: Numerator of the fraction
    ///   - denominator: Denominator of the fraction (must be positive)
    ///   - maxTerms: Maximum number of terms to compute
    /// - Returns: Array of partial quotients
    /// - Complexity: O(log(min(numerator, denominator)))
    @inlinable
    @_optimize(speed)
    @_effects(readonly)
    public static func continuedFractionExpansion(_ numerator: Int, _ denominator: Int, maxTerms: Int) -> [Int] {
        ValidationUtilities.validatePositiveInt(denominator, name: "Denominator")
        ValidationUtilities.validatePositiveInt(maxTerms, name: "maxTerms")

        var result: [Int] = []
        result.reserveCapacity(maxTerms)

        var n = numerator
        var d = denominator

        while d != 0, result.count < maxTerms {
            let quotient = n / d
            result.append(quotient)
            let remainder = n - quotient * d
            n = d
            d = remainder
        }

        return result
    }

    /// Computes all convergents of a continued fraction expansion.
    ///
    /// Returns the sequence of fractions p_k/q_k that best approximate the original value.
    ///
    /// **Example:**
    /// ```swift
    /// let expansion = [3, 7, 16]
    /// let convs = NumberTheory.convergents(of: expansion)
    /// // [(3, 1), (22, 7), (355, 113)]
    /// ```
    ///
    /// - Parameter expansion: Continued fraction expansion as array of partial quotients
    /// - Returns: Array of (numerator, denominator) pairs for each convergent
    /// - Complexity: O(n) where n is the length of the expansion
    @inlinable
    @_optimize(speed)
    @_effects(readonly)
    public static func convergents(of expansion: [Int]) -> [(p: Int, q: Int)] {
        guard !expansion.isEmpty else { return [] }

        var result: [(p: Int, q: Int)] = []
        result.reserveCapacity(expansion.count)

        var pPrev2 = 0, pPrev1 = 1
        var qPrev2 = 1, qPrev1 = 0

        for a in expansion {
            let p = a * pPrev1 + pPrev2
            let q = a * qPrev1 + qPrev2
            result.append((p, q))
            pPrev2 = pPrev1
            pPrev1 = p
            qPrev2 = qPrev1
            qPrev1 = q
        }

        return result
    }

    /// Checks if n is a perfect power and returns the base and exponent.
    ///
    /// Determines if n = a^k for some integers a >= 2 and k >= 2.
    ///
    /// **Example:**
    /// ```swift
    /// let result = NumberTheory.isPerfectPower(27)  // (3, 3) since 27 = 3^3
    /// let notPower = NumberTheory.isPerfectPower(15)  // nil
    /// let square = NumberTheory.isPerfectPower(16)  // (2, 4) since 16 = 2^4
    /// ```
    ///
    /// - Parameter n: Integer to check (must be >= 2)
    /// - Returns: Tuple (base, exponent) if n is a perfect power, nil otherwise
    /// - Complexity: O(log^2(n))
    @inlinable
    @_optimize(speed)
    @_effects(readonly)
    public static func isPerfectPower(_ n: Int) -> (base: Int, exp: Int)? {
        guard n >= 2 else { return nil }

        let maxExp = Int(Foundation.log2(Double(n))) + 1

        for k in stride(from: maxExp, through: 2, by: -1) {
            let root = integerRoot(n, k)
            var power = 1
            for _ in 0 ..< k {
                power *= root
                if power > n { break }
            }
            if power == n {
                return (root, k)
            }
            let rootPlus1 = root + 1
            power = 1
            for _ in 0 ..< k {
                power *= rootPlus1
                if power > n { break }
            }
        }

        return nil
    }

    /// Computes integer k-th root using binary search.
    @usableFromInline
    @_optimize(speed)
    @_effects(readonly)
    static func integerRoot(_ n: Int, _ k: Int) -> Int {
        guard n > 0, k > 0 else { return 0 }
        guard k > 1 else { return n }

        var lo = 1
        var hi = Int(Foundation.pow(Double(n), 1.0 / Double(k))) + 2

        while lo < hi {
            let mid = lo + (hi - lo + 1) / 2
            var power = 1
            var overflow = false
            for _ in 0 ..< k {
                if power > n / mid {
                    overflow = true
                    break
                }
                power *= mid
            }
            if overflow || power > n {
                hi = mid - 1
            } else {
                lo = mid
            }
        }

        return lo
    }

    /// Tests primality using Miller-Rabin with deterministic witnesses for n < 2^64.
    ///
    /// Uses a fixed set of witnesses that guarantees correctness for all integers up to 2^64.
    ///
    /// **Example:**
    /// ```swift
    /// let prime = NumberTheory.isPrime(17)  // true
    /// let composite = NumberTheory.isPrime(15)  // false
    /// let largePrime = NumberTheory.isPrime(104729)  // true
    /// ```
    ///
    /// - Parameter n: Integer to test
    /// - Returns: True if n is prime, false otherwise
    /// - Complexity: O(k * log^3(n)) where k is the number of witnesses
    @inlinable
    @_optimize(speed)
    @_effects(readonly)
    public static func isPrime(_ n: Int) -> Bool {
        guard n > 1 else { return false }
        guard n > 3 else { return true }
        guard n % 2 != 0 else { return false }

        let witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

        var d = n - 1
        var r = 0
        while d % 2 == 0 {
            d /= 2
            r += 1
        }

        for a in witnesses {
            guard a < n else { continue }
            if !millerRabinWitness(a: a, d: d, n: n, r: r) {
                return false
            }
        }

        return true
    }

    /// Single Miller-Rabin witness test.
    @usableFromInline
    @_optimize(speed)
    @_effects(readonly)
    static func millerRabinWitness(a: Int, d: Int, n: Int, r: Int) -> Bool {
        var x = modularPow(base: a, exponent: d, modulus: n)

        if x == 1 || x == n - 1 {
            return true
        }

        for _ in 0 ..< r - 1 {
            x = (x * x) % n
            if x == n - 1 {
                return true
            }
        }

        return false
    }

    /// Generates a random coprime base for Shor's algorithm.
    ///
    /// Returns a random integer a in [2, n-1] such that gcd(a, n) = 1.
    ///
    /// **Example:**
    /// ```swift
    /// let base = NumberTheory.randomCoprimeBase(for: 15)
    /// // Could return 2, 4, 7, 8, 11, 13, 14 (all coprime to 15)
    /// ```
    ///
    /// - Parameter n: The modulus (must be >= 3)
    /// - Returns: Random coprime base in [2, n-1]
    /// - Complexity: O(log n) expected
    @inlinable
    @_optimize(speed)
    public static func randomCoprimeBase(for n: Int) -> Int {
        ValidationUtilities.validateLowerBound(n, min: 3, name: "n")

        while true {
            let a = Int.random(in: 2 ..< n)
            if gcd(a, n) == 1 {
                return a
            }
        }
    }
}

/// Reasons why Shor's algorithm may fail to find factors.
///
/// **Example:**
/// ```swift
/// let reason: ShorFailureReason = .periodOdd
/// print(reason)  // "Period r is odd, cannot compute a^(r/2)"
/// ```
@frozen
public enum ShorFailureReason: Sendable, CustomStringConvertible {
    case inputTooSmall
    case inputEven
    case inputPrime
    case inputPerfectPower
    case periodOdd
    case trivialRoot
    case trivialFactor
    case maxAttemptsExceeded

    /// Human-readable description of the failure reason.
    ///
    /// **Example:**
    /// ```swift
    /// print(ShorFailureReason.periodOdd.description)
    /// // "Period r is odd, cannot compute a^(r/2)"
    /// ```
    @inlinable
    public var description: String {
        switch self {
        case .inputTooSmall:
            "Input N must be at least 15"
        case .inputEven:
            "Input N is even (trivially factored)"
        case .inputPrime:
            "Input N is prime"
        case .inputPerfectPower:
            "Input N is a perfect power"
        case .periodOdd:
            "Period r is odd, cannot compute a^(r/2)"
        case .trivialRoot:
            "a^(r/2) = -1 (mod N), giving trivial factors"
        case .trivialFactor:
            "Only trivial factors found (1 or N)"
        case .maxAttemptsExceeded:
            "Maximum number of attempts exceeded"
        }
    }
}

/// Configuration for Shor's factorization algorithm.
///
/// Specifies the number to factor and controls precision/attempts parameters.
///
/// **Example:**
/// ```swift
/// let config = ShorConfiguration(numberToFactor: 15)
/// print(config.effectivePrecisionQubits)  // 9 (2*ceil(log2(15)) + 1)
/// print(config.workRegisterQubits)  // 4
/// print(config.totalQubits)  // 13
/// ```
@frozen
public struct ShorConfiguration: Sendable {
    /// The number to factor (must be odd composite >= 15).
    public let numberToFactor: Int

    /// Number of precision qubits (nil = auto-calculate as 2*ceil(log2(N)) + 1).
    public let precisionQubits: Int?

    /// Maximum number of factorization attempts before giving up.
    public let maxAttempts: Int

    /// Creates a Shor configuration.
    ///
    /// **Example:**
    /// ```swift
    /// let config = ShorConfiguration(numberToFactor: 21, maxAttempts: 20)
    /// ```
    ///
    /// - Parameters:
    ///   - numberToFactor: The number to factor
    ///   - precisionQubits: Optional precision qubits (defaults to auto-calculation)
    ///   - maxAttempts: Maximum attempts (default: 10)
    public init(numberToFactor: Int, precisionQubits: Int? = nil, maxAttempts: Int = 10) {
        self.numberToFactor = numberToFactor
        self.precisionQubits = precisionQubits
        self.maxAttempts = maxAttempts
    }

    /// Number of qubits needed for the work register (ceil(log2(N))).
    ///
    /// **Example:**
    /// ```swift
    /// let config = ShorConfiguration(numberToFactor: 15)
    /// print(config.workRegisterQubits)  // 4
    /// ```
    @inlinable
    public var workRegisterQubits: Int {
        var bits = 0
        var n = numberToFactor - 1
        while n > 0 {
            bits += 1
            n >>= 1
        }
        return bits
    }

    /// Effective precision qubits (explicit or auto-calculated).
    ///
    /// **Example:**
    /// ```swift
    /// let config = ShorConfiguration(numberToFactor: 15)
    /// print(config.effectivePrecisionQubits)  // 9
    /// ```
    @inlinable
    public var effectivePrecisionQubits: Int {
        if let explicit = precisionQubits {
            return explicit
        }
        return 2 * workRegisterQubits + 1
    }

    /// Total qubits needed (precision + work register).
    ///
    /// **Example:**
    /// ```swift
    /// let config = ShorConfiguration(numberToFactor: 15)
    /// print(config.totalQubits)  // 13
    /// ```
    @inlinable
    public var totalQubits: Int {
        effectivePrecisionQubits + workRegisterQubits
    }
}

/// Result of period finding phase in Shor's algorithm.
///
/// Contains the measured phase and candidate periods derived from continued fraction expansion.
///
/// **Example:**
/// ```swift
/// let result = ShorPeriodResult(
///     measuredPhase: 0.25,
///     phaseNumerator: 1,
///     phaseDenominator: 4,
///     candidatePeriods: [4],
///     verifiedPeriod: 4,
///     measurementOutcome: 128,
///     precisionQubits: 9
/// )
/// print(result)
/// ```
@frozen
public struct ShorPeriodResult: Sendable, CustomStringConvertible {
    /// Measured phase as a fraction of 2*pi.
    public let measuredPhase: Double

    /// Numerator of the phase fraction s/r.
    public let phaseNumerator: Int

    /// Denominator of the phase fraction s/r.
    public let phaseDenominator: Int

    /// Candidate periods derived from continued fraction convergents.
    public let candidatePeriods: [Int]

    /// Verified period (if any candidate satisfies a^r = 1 mod N).
    public let verifiedPeriod: Int?

    /// Raw measurement outcome from precision register.
    public let measurementOutcome: Int

    /// Number of precision qubits used.
    public let precisionQubits: Int

    /// Human-readable description of the period finding result.
    ///
    /// **Example:**
    /// ```swift
    /// let result = ShorPeriodResult(...)
    /// print(result)  // "ShorPeriodResult(phase: 0.25, candidates: [4], verified: 4)"
    /// ```
    @inlinable
    public var description: String {
        let verifiedStr = verifiedPeriod.map { String($0) } ?? "none"
        return "ShorPeriodResult(phase: \(measuredPhase), candidates: \(candidatePeriods), verified: \(verifiedStr))"
    }
}

/// Final result of Shor's factorization algorithm.
///
/// Contains the factors if found, along with diagnostic information about the algorithm execution.
///
/// **Example:**
/// ```swift
/// let result = ShorResult(
///     numberToFactor: 15,
///     factors: (3, 5),
///     period: 4,
///     base: 7,
///     attempts: 1,
///     success: true,
///     failureReason: nil
/// )
/// print(result)  // "ShorResult(15 = 3 * 5, period: 4, base: 7, attempts: 1)"
/// ```
@frozen
public struct ShorResult: Sendable, CustomStringConvertible {
    /// The number that was factored.
    public let numberToFactor: Int

    /// The non-trivial factors if found (p < q).
    public let factors: (p: Int, q: Int)?

    /// The period r such that a^r = 1 (mod N).
    public let period: Int?

    /// The base a used for period finding.
    public let base: Int

    /// Number of attempts made.
    public let attempts: Int

    /// Whether factorization was successful.
    public let success: Bool

    /// Reason for failure if unsuccessful.
    public let failureReason: ShorFailureReason?

    /// Human-readable description of the factorization result.
    ///
    /// **Example:**
    /// ```swift
    /// let result = ShorResult(...)
    /// print(result)
    /// ```
    @inlinable
    public var description: String {
        if let factors {
            return "ShorResult(\(numberToFactor) = \(factors.p) * \(factors.q), period: \(period ?? 0), base: \(base), attempts: \(attempts))"
        } else if let reason = failureReason {
            return "ShorResult(\(numberToFactor) failed: \(reason), attempts: \(attempts))"
        }
        return "ShorResult(\(numberToFactor), attempts: \(attempts))"
    }
}

/// Modular multiplication unitary operator for Shor's algorithm.
///
/// Implements the permutation matrix |x> -> |ax mod N> for x < N (identity for x >= N).
///
/// **Example:**
/// ```swift
/// let unitary = ModularMultiplicationUnitary(multiplier: 7, modulus: 15, qubits: 4)
/// let matrix = unitary.permutationMatrix()
/// // Matrix permutes |x> to |7x mod 15> for x < 15
/// ```
@frozen
public struct ModularMultiplicationUnitary: Sendable {
    /// The multiplier a in the operation |x> -> |ax mod N>.
    public let multiplier: Int

    /// The modulus N.
    public let modulus: Int

    /// Number of qubits needed to represent values up to N-1.
    public let qubits: Int

    /// Creates a modular multiplication unitary.
    ///
    /// **Example:**
    /// ```swift
    /// let unitary = ModularMultiplicationUnitary(multiplier: 7, modulus: 15, qubits: 4)
    /// ```
    ///
    /// - Parameters:
    ///   - multiplier: The multiplier a (must be coprime to modulus)
    ///   - modulus: The modulus N
    ///   - qubits: Number of qubits in the register
    public init(multiplier: Int, modulus: Int, qubits: Int) {
        self.multiplier = multiplier
        self.modulus = modulus
        self.qubits = qubits
    }

    /// Generates the unitary permutation matrix.
    ///
    /// For x in [0, modulus-1]: maps |x> to |(multiplier * x) mod modulus>
    /// For x >= modulus: identity mapping |x> -> |x>
    ///
    /// **Example:**
    /// ```swift
    /// let unitary = ModularMultiplicationUnitary(multiplier: 2, modulus: 3, qubits: 2)
    /// let matrix = unitary.permutationMatrix()
    /// // |0> -> |0>, |1> -> |2>, |2> -> |1>, |3> -> |3>
    /// ```
    ///
    /// - Returns: Unitary permutation matrix of dimension 2^qubits x 2^qubits
    /// - Complexity: O(4^qubits)
    @inlinable
    @_optimize(speed)
    @_effects(readonly)
    public func permutationMatrix() -> [[Complex<Double>]] {
        let dim = 1 << qubits
        var matrix = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: dim),
            count: dim,
        )

        for x in 0 ..< dim {
            let target: Int = if x < modulus {
                (multiplier * x) % modulus
            } else {
                x
            }
            matrix[target][x] = .one
        }

        return matrix
    }

    /// Creates a powered version of this unitary.
    ///
    /// Returns unitary for multiplication by (multiplier^(2^k)) mod modulus.
    ///
    /// **Example:**
    /// ```swift
    /// let unitary = ModularMultiplicationUnitary(multiplier: 2, modulus: 15, qubits: 4)
    /// let powered = unitary.powered(3)  // multiplication by 2^8 mod 15 = 1
    /// ```
    ///
    /// - Parameter k: Power of 2 exponent
    /// - Returns: New unitary for multiplication by multiplier^(2^k) mod modulus
    /// - Complexity: O(k) for computing new multiplier
    @inlinable
    @_optimize(speed)
    @_effects(readonly)
    public func powered(_ k: Int) -> ModularMultiplicationUnitary {
        let newMultiplier = NumberTheory.modularPow(base: multiplier, exponent: 1 << k, modulus: modulus)
        return ModularMultiplicationUnitary(multiplier: newMultiplier, modulus: modulus, qubits: qubits)
    }
}

public extension QuantumCircuit {
    /// Creates a period finding circuit for Shor's algorithm.
    ///
    /// Implements quantum phase estimation to find the period r of a^x mod N.
    ///
    /// Circuit structure:
    /// 1. Precision register: qubits 0..<precisionQubits (initialized to superposition)
    /// 2. Work register: qubits precisionQubits..<total (initialized to |1>)
    /// 3. Controlled modular multiplications
    /// 4. Inverse QFT on precision register
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.shorPeriodFinding(base: 7, modulus: 15, precisionQubits: 9)
    /// let state = circuit.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - base: Base a for modular exponentiation
    ///   - modulus: Modulus N
    ///   - precisionQubits: Number of precision qubits
    /// - Returns: Period finding quantum circuit
    /// - Complexity: O(precisionQubits * workQubits^2) gates
    @_optimize(speed)
    @_eagerMove
    static func shorPeriodFinding(base: Int, modulus: Int, precisionQubits: Int) -> QuantumCircuit {
        var workBits = 0
        var n = modulus - 1
        while n > 0 {
            workBits += 1
            n >>= 1
        }

        let totalQubits = precisionQubits + workBits
        var circuit = QuantumCircuit(qubits: totalQubits)

        for i in 0 ..< precisionQubits {
            circuit.append(.hadamard, to: i)
        }

        circuit.append(.pauliX, to: precisionQubits)

        for k in 0 ..< precisionQubits {
            let controlQubit = k
            let multiplier = NumberTheory.modularPow(base: base, exponent: 1 << k, modulus: modulus)

            let unitary = ModularMultiplicationUnitary(multiplier: multiplier, modulus: modulus, qubits: workBits)
            let permMatrix = unitary.permutationMatrix()

            let controlledDim = 2 * permMatrix.count
            var controlledMatrix = [[Complex<Double>]](
                repeating: [Complex<Double>](repeating: .zero, count: controlledDim),
                count: controlledDim,
            )

            let halfDim = permMatrix.count
            for i in 0 ..< halfDim {
                controlledMatrix[i][i] = .one
            }
            for i in 0 ..< halfDim {
                for j in 0 ..< halfDim {
                    controlledMatrix[halfDim + i][halfDim + j] = permMatrix[i][j]
                }
            }

            var allQubits: [Int] = []
            for w in 0 ..< workBits {
                allQubits.append(precisionQubits + w)
            }
            allQubits.append(controlQubit)

            circuit.append(.customUnitary(matrix: controlledMatrix), to: allQubits)
        }

        let inverseQFTOps = inverseQFTGatesForRange(start: 0, count: precisionQubits)
        for (gate, qubits) in inverseQFTOps {
            circuit.append(gate, to: qubits)
        }

        return circuit
    }

    /// Generates inverse QFT gates for a specific qubit range.
    @_optimize(speed)
    private static func inverseQFTGatesForRange(start: Int, count: Int) -> [(gate: QuantumGate, qubits: [Int])] {
        var gates: [(gate: QuantumGate, qubits: [Int])] = []

        let swapCount = count / 2
        for i in 0 ..< swapCount {
            let j = count - 1 - i
            gates.append((.swap, [start + i, start + j]))
        }

        for target in (0 ..< count).reversed() {
            for control in (target + 1 ..< count).reversed() {
                let k = control - target + 1
                let theta = -Double.pi / Double(1 << k)
                gates.append((.controlledPhase(theta), [start + control, start + target]))
            }
            gates.append((.hadamard, [start + target]))
        }

        return gates
    }

    /// Creates a complete Shor factorization circuit.
    ///
    /// Convenience wrapper that builds the period finding circuit with auto-calculated parameters.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.shorFactorization(number: 15, base: 7)
    /// let state = circuit.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - number: Number to factor
    ///   - base: Base for modular exponentiation
    ///   - precisionQubits: Optional precision qubits (auto-calculated if nil)
    /// - Returns: Period finding circuit
    @_eagerMove
    static func shorFactorization(number: Int, base: Int, precisionQubits: Int? = nil) -> QuantumCircuit {
        let config = ShorConfiguration(numberToFactor: number, precisionQubits: precisionQubits)
        return shorPeriodFinding(base: base, modulus: number, precisionQubits: config.effectivePrecisionQubits)
    }
}

public extension QuantumState {
    /// Extracts period finding result from quantum state after Shor circuit execution.
    ///
    /// Analyzes the precision register to find the measured phase and derives candidate
    /// periods using continued fraction expansion.
    ///
    /// **Example:**
    /// ```swift
    /// let circuit = QuantumCircuit.shorPeriodFinding(base: 7, modulus: 15, precisionQubits: 9)
    /// let state = circuit.execute()
    /// let config = ShorConfiguration(numberToFactor: 15)
    /// let result = state.shorPeriodResult(config: config, base: 7)
    /// print(result.verifiedPeriod)  // 4
    /// ```
    ///
    /// - Parameters:
    ///   - config: Shor configuration
    ///   - base: Base used in period finding
    /// - Returns: Period finding result with candidates and verified period
    /// - Complexity: O(2^precisionQubits) for state analysis
    @_optimize(speed)
    @_effects(readonly)
    func shorPeriodResult(config: ShorConfiguration, base: Int) -> ShorPeriodResult {
        let precisionQubits = config.effectivePrecisionQubits
        let precisionStateSize = 1 << precisionQubits

        var precisionProbabilities = [Double](repeating: 0.0, count: precisionStateSize)

        for basisIndex in 0 ..< stateSpaceSize {
            let precisionIndex = basisIndex % precisionStateSize
            let probability = amplitudes[basisIndex].magnitudeSquared
            precisionProbabilities[precisionIndex] += probability
        }

        var probWithIndex: [(prob: Double, index: Int)] = []
        for i in 0 ..< precisionStateSize {
            if precisionProbabilities[i] > 1e-6 {
                probWithIndex.append((precisionProbabilities[i], i))
            }
        }
        probWithIndex.sort { $0.prob > $1.prob }
        let topIndices = probWithIndex.prefix(10)

        var allCandidates: Set<Int> = []
        for (_, index) in topIndices {
            let expansion = NumberTheory.continuedFractionExpansion(index, precisionStateSize, maxTerms: 2 * precisionQubits)
            let convs = NumberTheory.convergents(of: expansion)
            for (_, q) in convs {
                if q > 0, q <= config.numberToFactor {
                    allCandidates.insert(q)
                }
            }
        }

        let candidatesWithDoubles = allCandidates.union(allCandidates.map { $0 * 2 }.filter { $0 <= config.numberToFactor })

        var verifiedPeriod: Int?
        for q in candidatesWithDoubles.sorted() {
            let check = NumberTheory.modularPow(base: base, exponent: q, modulus: config.numberToFactor)
            if check == 1 {
                verifiedPeriod = q
                break
            }
        }

        // Safety: topIndices can't be empty with normalized quantum state (probabilities sum to 1)
        let topIndex = topIndices.first!.index
        let measuredPhase = Double(topIndex) / Double(precisionStateSize)

        let phaseNum = topIndex
        let phaseDen = precisionStateSize
        let g = NumberTheory.gcd(phaseNum, phaseDen)

        return ShorPeriodResult(
            measuredPhase: measuredPhase,
            phaseNumerator: phaseNum / g,
            phaseDenominator: phaseDen / g,
            candidatePeriods: candidatesWithDoubles.sorted(),
            verifiedPeriod: verifiedPeriod,
            measurementOutcome: topIndex,
            precisionQubits: precisionQubits,
        )
    }
}

/// Actor for executing Shor's factorization algorithm.
///
/// Combines classical pre-processing, quantum period finding, and classical post-processing
/// to factor composite integers.
///
/// **Example:**
/// ```swift
/// let config = ShorConfiguration(numberToFactor: 15)
/// let simulator = QuantumSimulator()
/// let shor = ShorFactorization(configuration: config, simulator: simulator)
/// let result = await shor.run()
/// print(result)  // ShorResult(15 = 3 * 5, ...)
/// ```
public actor ShorFactorization {
    /// Configuration for the factorization.
    public let configuration: ShorConfiguration

    /// Quantum simulator for circuit execution.
    private let simulator: QuantumSimulator

    /// Creates a Shor factorization actor.
    ///
    /// **Example:**
    /// ```swift
    /// let config = ShorConfiguration(numberToFactor: 21)
    /// let simulator = QuantumSimulator()
    /// let shor = ShorFactorization(configuration: config, simulator: simulator)
    /// ```
    ///
    /// - Parameters:
    ///   - configuration: Factorization configuration
    ///   - simulator: Quantum simulator to use
    public init(configuration: ShorConfiguration, simulator: QuantumSimulator) {
        self.configuration = configuration
        self.simulator = simulator
    }

    /// Runs the Shor factorization algorithm.
    ///
    /// Algorithm:
    /// 1. Classical pre-checks (even, prime, perfect power)
    /// 2. Quantum period finding with random coprime bases
    /// 3. Classical factor extraction from period
    ///
    /// **Example:**
    /// ```swift
    /// let result = await shor.run { progress in
    ///     print("Progress: \(progress)")
    /// }
    /// if let factors = result.factors {
    ///     print("\(result.numberToFactor) = \(factors.p) * \(factors.q)")
    /// }
    /// ```
    ///
    /// - Parameter progress: Optional progress callback
    /// - Returns: Factorization result
    /// - Complexity: O(maxAttempts * quantum_circuit_execution)
    @_optimize(speed)
    public func run(progress: (@Sendable (String) async -> Void)? = nil) async -> ShorResult {
        let n = configuration.numberToFactor

        if n < 15 {
            return ShorResult(
                numberToFactor: n,
                factors: nil,
                period: nil,
                base: 0,
                attempts: 0,
                success: false,
                failureReason: .inputTooSmall,
            )
        }

        if n % 2 == 0 {
            let (p, q) = (2, n / 2)
            return ShorResult(
                numberToFactor: n,
                factors: (min(p, q), max(p, q)),
                period: nil,
                base: 0,
                attempts: 0,
                success: true,
                failureReason: nil,
            )
        }

        if NumberTheory.isPrime(n) {
            return ShorResult(
                numberToFactor: n,
                factors: nil,
                period: nil,
                base: 0,
                attempts: 0,
                success: false,
                failureReason: .inputPrime,
            )
        }

        if let (base, exp) = NumberTheory.isPerfectPower(n) {
            var factor = base
            for _ in 1 ..< exp - 1 {
                factor *= base
            }
            let other = n / base
            return ShorResult(
                numberToFactor: n,
                factors: (min(base, other), max(base, other)),
                period: nil,
                base: 0,
                attempts: 0,
                success: true,
                failureReason: nil,
            )
        }

        for attempt in 1 ... configuration.maxAttempts {
            await progress?("Attempt \(attempt)/\(configuration.maxAttempts)")

            let a = NumberTheory.randomCoprimeBase(for: n)

            let circuit = QuantumCircuit.shorPeriodFinding(
                base: a,
                modulus: n,
                precisionQubits: configuration.effectivePrecisionQubits,
            )

            let state = await simulator.execute(circuit)

            let periodResult = state.shorPeriodResult(config: configuration, base: a)

            guard let r = periodResult.verifiedPeriod, r > 0 else {
                continue
            }

            if r % 2 != 0 {
                continue
            }

            let x = NumberTheory.modularPow(base: a, exponent: r / 2, modulus: n)

            if x == n - 1 {
                continue
            }

            let factor1 = NumberTheory.gcd(x + 1, n)
            let factor2 = NumberTheory.gcd(x - 1, n)

            if factor1 > 1, factor1 < n {
                let other = n / factor1
                return ShorResult(
                    numberToFactor: n,
                    factors: (min(factor1, other), max(factor1, other)),
                    period: r,
                    base: a,
                    attempts: attempt,
                    success: true,
                    failureReason: nil,
                )
            }

            if factor2 > 1, factor2 < n {
                let other = n / factor2
                return ShorResult(
                    numberToFactor: n,
                    factors: (min(factor2, other), max(factor2, other)),
                    period: r,
                    base: a,
                    attempts: attempt,
                    success: true,
                    failureReason: nil,
                )
            }
        }

        return ShorResult(
            numberToFactor: n,
            factors: nil,
            period: nil,
            base: 0,
            attempts: configuration.maxAttempts,
            success: false,
            failureReason: .maxAttemptsExceeded,
        )
    }
}
