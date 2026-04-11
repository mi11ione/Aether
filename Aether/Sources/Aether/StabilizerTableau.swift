// Copyright (c) 2025-2026 Roman Zhuzhgov, Apache License 2.0

import GameplayKit

/// Stabilizer tableau data structure for efficient Clifford circuit simulation.
///
/// Represents an n-qubit stabilizer state using a 2n x (2n+1) binary matrix known as the
/// Aaronson-Gottesman tableau. The first n rows are destabilizers and the last n rows are
/// stabilizers. Each row encodes a Pauli operator: the first 2n bits represent X and Z
/// components, and the final bit is the phase (0 for +1, 1 for -1).
///
/// This representation enables O(n^2) time complexity per Clifford gate with SIMD
/// acceleration, supporting simulation of millions of qubits. Memory usage scales
/// as O(n^2) bits rather than the exponential O(2^n) of statevector simulation.
///
/// **Example:**
/// ```swift
/// var tableau = StabilizerTableau(qubits: 3)
/// tableau.apply(.hadamard, to: 0)
/// tableau.apply(.cnot, to: [0, 1])
/// let outcome = tableau.measure(0, seed: 42)
/// ```
///
/// - SeeAlso: ``QuantumGate``
/// - SeeAlso: ``PauliString``
@frozen public struct StabilizerTableau: Sendable, Equatable, CustomStringConvertible {
    @usableFromInline let n: Int
    @usableFromInline var tableau: ContiguousArray<UInt64>
    @usableFromInline let wordsPerRow: Int

    /// Number of qubits in this stabilizer state.
    ///
    /// **Example:**
    /// ```swift
    /// let tableau = StabilizerTableau(qubits: 5)
    /// let n = tableau.qubits
    /// print(n)  // 5
    /// ```
    public let qubits: Int

    /// Whether the current state is a valid stabilizer state.
    ///
    /// A stabilizer state is uniquely defined by n mutually commuting Pauli operators
    /// that stabilize it (have eigenvalue +1). This property validates the tableau structure.
    ///
    /// **Example:**
    /// ```swift
    /// let tableau = StabilizerTableau(qubits: 2)
    /// let valid = tableau.isStabilizerState
    /// print(valid)  // true
    /// ```
    @inlinable
    public var isStabilizerState: Bool {
        n > 0
    }

    /// Memory usage in bytes for the tableau storage.
    ///
    /// The tableau requires O(n^2) bits of storage, packed into UInt64 words.
    ///
    /// **Example:**
    /// ```swift
    /// let tableau = StabilizerTableau(qubits: 100)
    /// let bytes = tableau.memoryUsage
    /// print(bytes)  // ~5000 bytes
    /// ```
    @inlinable
    public var memoryUsage: Int {
        tableau.count * MemoryLayout<UInt64>.size
    }

    /// Creates a stabilizer tableau initialized to the |0...0> computational basis state.
    ///
    /// The initial state has destabilizers X_i and stabilizers Z_i for each qubit i,
    /// representing the ground state where all qubits are in |0>.
    ///
    /// **Example:**
    /// ```swift
    /// let tableau = StabilizerTableau(qubits: 3)
    /// print(tableau.qubits)  // 3
    /// print(tableau.isStabilizerState)  // true
    /// ```
    ///
    /// - Parameter qubits: Number of qubits (must be positive)
    /// - Precondition: `qubits` must be positive
    /// - Complexity: O(n^2)
    public init(qubits: Int) {
        ValidationUtilities.validatePositiveQubits(qubits)

        self.qubits = qubits
        n = qubits

        let totalBitsPerRow = 2 * n + 1
        wordsPerRow = (totalBitsPerRow + 63) / 64

        let totalWords = 2 * n * wordsPerRow
        tableau = ContiguousArray<UInt64>(repeating: 0, count: totalWords)

        for i in 0 ..< n {
            setX(row: i, qubit: i, value: true)
            setZ(row: n + i, qubit: i, value: true)
        }
    }

    /// Applies a single-qubit Clifford gate to the specified qubit.
    ///
    /// Supported gates: identity, pauliX, pauliY, pauliZ, hadamard, sGate, sx, sy.
    /// Each gate updates the tableau according to Clifford transformation rules.
    ///
    /// **Example:**
    /// ```swift
    /// var tableau = StabilizerTableau(qubits: 2)
    /// tableau.apply(.hadamard, to: 0)
    /// tableau.apply(.sGate, to: 1)
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Single-qubit Clifford gate to apply
    ///   - qubit: Target qubit index (0 to n-1)
    /// - Precondition: `qubit` must be in range 0 to n-1
    /// - Precondition: `gate` must be a single-qubit Clifford gate
    /// - Complexity: O(n)
    @inlinable
    @_optimize(speed)
    public mutating func apply(_ gate: QuantumGate, to qubit: Int) {
        ValidationUtilities.validateQubitIndex(qubit, qubits: n)

        switch gate {
        case .identity:
            break
        case .pauliX:
            applyPauliX(qubit)
        case .pauliY:
            applyPauliY(qubit)
        case .pauliZ:
            applyPauliZ(qubit)
        case .hadamard:
            applyHadamard(qubit)
        case .sGate:
            applyS(qubit)
        case .sx:
            applyHadamard(qubit)
            applyS(qubit)
            applyHadamard(qubit)
        case .sy:
            applyS(qubit)
            applyS(qubit)
            applyS(qubit)
            applyHadamard(qubit)
            applyS(qubit)
            applyHadamard(qubit)
        default:
            ValidationUtilities.validateSingleQubitCliffordGate(gate)
        }
    }

    /// Applies a multi-qubit Clifford gate to the specified qubits.
    ///
    /// Supported gates: cnot (control, target), cz (control, target), swap.
    ///
    /// **Example:**
    /// ```swift
    /// var tableau = StabilizerTableau(qubits: 3)
    /// tableau.apply(.hadamard, to: 0)
    /// tableau.apply(.cnot, to: [0, 1])  // Control: 0, Target: 1
    /// tableau.apply(.cz, to: [1, 2])
    /// ```
    ///
    /// - Parameters:
    ///   - gate: Multi-qubit Clifford gate to apply
    ///   - qubits: Target qubit indices (order depends on gate type)
    /// - Precondition: `qubits` must contain exactly 2 indices, each in range 0 to n-1
    /// - Precondition: `gate` must be a multi-qubit Clifford gate (cnot, cz, or swap)
    /// - Complexity: O(n)
    @inlinable
    @_optimize(speed)
    public mutating func apply(_ gate: QuantumGate, to qubits: [Int]) {
        switch gate {
        case .cnot:
            ValidationUtilities.validateArrayCount(qubits, expected: 2, name: "CNOT qubits")
            ValidationUtilities.validateQubitIndex(qubits[0], qubits: n)
            ValidationUtilities.validateQubitIndex(qubits[1], qubits: n)
            applyCNOT(control: qubits[0], target: qubits[1])
        case .cz:
            ValidationUtilities.validateArrayCount(qubits, expected: 2, name: "CZ qubits")
            ValidationUtilities.validateQubitIndex(qubits[0], qubits: n)
            ValidationUtilities.validateQubitIndex(qubits[1], qubits: n)
            applyCZ(qubit1: qubits[0], qubit2: qubits[1])
        case .swap:
            ValidationUtilities.validateArrayCount(qubits, expected: 2, name: "SWAP qubits")
            ValidationUtilities.validateQubitIndex(qubits[0], qubits: n)
            ValidationUtilities.validateQubitIndex(qubits[1], qubits: n)
            applyCNOT(control: qubits[0], target: qubits[1])
            applyCNOT(control: qubits[1], target: qubits[0])
            applyCNOT(control: qubits[0], target: qubits[1])
        default:
            ValidationUtilities.validateMultiQubitCliffordGate(gate)
        }
    }

    /// Computes the probability of measuring 0 or 1 for a qubit in the specified Pauli basis.
    ///
    /// For Z-basis measurement, returns (p0, p1) where p0 is probability of measuring |0>
    /// and p1 is probability of measuring |1>. For stabilizer states, one of these is
    /// always 0.0 or 0.5.
    ///
    /// **Example:**
    /// ```swift
    /// var tableau = StabilizerTableau(qubits: 2)
    /// tableau.apply(.hadamard, to: 0)
    /// let (p0, p1) = tableau.probability(of: 0, measuring: .z)
    /// // p0 = 0.5, p1 = 0.5 (superposition)
    /// ```
    ///
    /// - Parameters:
    ///   - qubit: Qubit index to query (0 to n-1)
    ///   - basis: Pauli basis for measurement
    /// - Returns: Tuple of (probability of 0 outcome, probability of 1 outcome)
    /// - Precondition: `qubit` must be in range 0 to n-1
    /// - Complexity: O(n)
    @_effects(readonly)
    @inlinable
    public func probability(of qubit: Int, measuring basis: PauliBasis) -> (p0: Double, p1: Double) {
        ValidationUtilities.validateQubitIndex(qubit, qubits: n)

        var tempTableau = self

        switch basis {
        case .x:
            tempTableau.applyHadamard(qubit)
        case .y:
            tempTableau.applyS(qubit)
            tempTableau.applyS(qubit)
            tempTableau.applyS(qubit)
            tempTableau.applyHadamard(qubit)
        case .z:
            break
        }

        for row in n ..< 2 * n {
            if tempTableau.getX(row: row, qubit: qubit) {
                return (0.5, 0.5)
            }
        }

        var phase = false
        for row in n ..< 2 * n {
            if tempTableau.getZ(row: row, qubit: qubit) {
                phase = phase != tempTableau.getPhase(row: row)
            }
        }

        if phase {
            return (0.0, 1.0)
        } else {
            return (1.0, 0.0)
        }
    }

    /// Performs a projective measurement on the specified qubit in the Z basis.
    ///
    /// If the measurement outcome is deterministic, returns that outcome. If random,
    /// uses the provided seed or system random to select the outcome, then updates
    /// the tableau to reflect the post-measurement state.
    ///
    /// **Example:**
    /// ```swift
    /// var tableau = StabilizerTableau(qubits: 2)
    /// tableau.apply(.hadamard, to: 0)
    /// let outcome = tableau.measure(0, seed: 42)
    /// // outcome is 0 or 1 (random)
    /// ```
    ///
    /// - Parameters:
    ///   - qubit: Qubit index to measure (0 to n-1)
    ///   - seed: Optional seed for reproducible random outcomes
    /// - Returns: Measurement outcome (0 or 1)
    /// - Precondition: `qubit` must be in range 0 to n-1
    /// - Complexity: O(n^2)
    @inlinable
    @_optimize(speed)
    public mutating func measure(_ qubit: Int, seed: UInt64?) -> Int {
        ValidationUtilities.validateQubitIndex(qubit, qubits: n)

        var anticommutingRow: Int? = nil
        for row in n ..< 2 * n {
            if getX(row: row, qubit: qubit) {
                anticommutingRow = row
                break
            }
        }

        guard let p = anticommutingRow else {
            var phase = false
            for row in n ..< 2 * n {
                if getZ(row: row, qubit: qubit) {
                    phase = phase != getPhase(row: row)
                }
            }
            return phase ? 1 : 0
        }

        for row in 0 ..< 2 * n where row != p {
            if getX(row: row, qubit: qubit) {
                rowMultiply(target: row, source: p)
            }
        }

        let destabRow = p - n
        copyRow(from: p, to: destabRow)

        clearRow(p)
        setZ(row: p, qubit: qubit, value: true)

        let random: Bool
        if let seed {
            let source = GKMersenneTwisterRandomSource(seed: seed)
            random = source.nextInt(upperBound: 2) == 1
        } else {
            random = Bool.random()
        }

        setPhase(row: p, value: random)

        for row in n ..< 2 * n where row != p {
            if getZ(row: row, qubit: qubit) {
                rowMultiply(target: row, source: p)
            }
        }

        return random ? 1 : 0
    }

    /// Projects qubit onto a specified measurement outcome without randomness.
    ///
    /// Unlike ``measure(_:seed:)`` which randomly selects an outcome for non-stabilized qubits,
    /// this method forces the collapse to `outcome`. Returns the coefficient scaling factor
    /// for the stabilizer rank decomposition, or nil if the outcome is inconsistent with a
    /// deterministic stabilizer eigenvalue.
    ///
    /// **Example:**
    /// ```swift
    /// var tableau = StabilizerTableau(qubits: 2)
    /// tableau.apply(.hadamard, to: 0)
    /// let scale = tableau.projectMeasurement(0, outcome: 0)
    /// // scale ≈ 0.707 (1/√2, random case forced to |0⟩)
    /// ```
    ///
    /// - Parameters:
    ///   - qubit: Qubit index to measure (0 to n-1)
    ///   - outcome: Desired measurement outcome (0 or 1)
    /// - Returns: Coefficient scaling factor (1.0 for deterministic match, 1/√2 for random),
    ///   or nil if deterministic outcome does not match
    /// - Precondition: `qubit` must be in range 0 to n-1
    /// - Precondition: `outcome` must be 0 or 1
    /// - Complexity: O(n^2)
    ///
    /// - SeeAlso: ``measure(_:seed:)``
    @inlinable
    @_optimize(speed)
    public mutating func projectMeasurement(_ qubit: Int, outcome: Int) -> Double? {
        ValidationUtilities.validateQubitIndex(qubit, qubits: n)
        ValidationUtilities.validateBinaryValue(outcome, name: "outcome")

        var anticommutingRow: Int? = nil
        for row in n ..< 2 * n {
            if getX(row: row, qubit: qubit) {
                anticommutingRow = row
                break
            }
        }

        guard let p = anticommutingRow else {
            var phase = false
            for row in n ..< 2 * n {
                if getZ(row: row, qubit: qubit) {
                    phase = phase != getPhase(row: row)
                }
            }
            let deterministicOutcome = phase ? 1 : 0
            return deterministicOutcome == outcome ? 1.0 : nil
        }

        for row in 0 ..< 2 * n where row != p {
            if getX(row: row, qubit: qubit) {
                rowMultiply(target: row, source: p)
            }
        }

        let destabRow = p - n
        copyRow(from: p, to: destabRow)
        clearRow(p)
        setZ(row: p, qubit: qubit, value: true)
        setPhase(row: p, value: outcome == 1)

        for row in n ..< 2 * n where row != p {
            if getZ(row: row, qubit: qubit) {
                rowMultiply(target: row, source: p)
            }
        }

        return 1.0 / 2.0.squareRoot()
    }

    /// Computes the amplitude of a specific computational basis state.
    ///
    /// For stabilizer states, amplitudes are either 0 or ±1/√(2^k) for some k.
    /// Returns nil if the amplitude computation would be too expensive (non-stabilizer).
    ///
    /// **Example:**
    /// ```swift
    /// let tableau = StabilizerTableau(qubits: 2)
    /// let amp = tableau.amplitude(of: 0)  // Amplitude of |00>
    /// // amp = Complex(1.0, 0.0)
    /// ```
    ///
    /// - Parameter basisState: Computational basis state index (0 to 2^n-1)
    /// - Returns: Complex amplitude or nil if not efficiently computable
    /// - Precondition: basisState >= 0 && basisState < 2^qubits
    /// - Complexity: O(n^3) worst case
    @_effects(readonly)
    @inlinable
    public func amplitude(of basisState: Int) -> Complex<Double>? {
        ValidationUtilities.validateNonNegativeInt(basisState, name: "Basis state")

        guard n <= 20 else { return nil }
        guard basisState < (1 << n) else { return nil }

        var tempTableau = self

        for i in 0 ..< n {
            let bit = (basisState >> i) & 1
            if bit == 1 {
                tempTableau.applyPauliX(i)
            }
        }

        var xGeneratorCount = 0
        for row in n ..< 2 * n {
            var hasX = false
            for q in 0 ..< n {
                if tempTableau.getX(row: row, qubit: q) {
                    hasX = true
                    break
                }
            }
            if hasX {
                xGeneratorCount += 1
                continue
            }

            if tempTableau.getPhase(row: row) {
                return .zero
            }
        }

        var phaseCount = 0
        for row in n ..< 2 * n {
            var hasX = false
            var xMatchesBasis = true
            for q in 0 ..< n {
                let bit = (basisState >> q) & 1
                let xBit = getX(row: row, qubit: q)
                if xBit {
                    hasX = true
                }
                if (bit == 1) != xBit {
                    xMatchesBasis = false
                }
            }
            if hasX, xMatchesBasis, tempTableau.getPhase(row: row) {
                phaseCount += 2
            }
        }

        let normFactor = 1.0 / Double(1 << xGeneratorCount).squareRoot()
        let phase = Double(phaseCount % 4) * Double.pi / 2.0

        return Complex<Double>(magnitude: normFactor, phase: phase)
    }

    /// Computes the expectation value of a Pauli string observable.
    ///
    /// For stabilizer states, the expectation value of any Pauli operator is
    /// either -1, 0, or +1.
    ///
    /// **Example:**
    /// ```swift
    /// var tableau = StabilizerTableau(qubits: 2)
    /// tableau.apply(.hadamard, to: 0)
    /// tableau.apply(.cnot, to: [0, 1])
    /// let zz = PauliString(.z(0), .z(1))
    /// let expectation = tableau.expectationValue(of: zz)
    /// // expectation = 1.0 (Bell state has ZZ = +1)
    /// ```
    ///
    /// - Parameter pauliString: Pauli string operator to measure
    /// - Returns: Expectation value (-1.0, 0.0, or +1.0)
    /// - Precondition: All qubit indices in `pauliString` must be in range 0 to n-1
    /// - Complexity: O(n^2)
    @_effects(readonly)
    @inlinable
    public func expectationValue(of pauliString: PauliString) -> Double {
        if pauliString.operators.isEmpty {
            return 1.0
        }

        for op in pauliString.operators {
            ValidationUtilities.validateQubitIndex(op.qubit, qubits: n)
        }

        for row in n ..< 2 * n {
            var anticommuteParity = false

            for op in pauliString.operators {
                let xBit = getX(row: row, qubit: op.qubit)
                let zBit = getZ(row: row, qubit: op.qubit)

                let anticommute: Bool = switch op.basis {
                case .x:
                    zBit
                case .y:
                    xBit != zBit
                case .z:
                    xBit
                }

                if anticommute {
                    anticommuteParity = !anticommuteParity
                }
            }

            if anticommuteParity {
                return 0.0
            }
        }

        var queryX = ContiguousArray<Bool>(repeating: false, count: n)
        var queryZ = ContiguousArray<Bool>(repeating: false, count: n)
        for op in pauliString.operators {
            switch op.basis {
            case .x:
                queryX[op.qubit] = true
            case .y:
                queryX[op.qubit] = true
                queryZ[op.qubit] = true
            case .z:
                queryZ[op.qubit] = true
            }
        }

        var phaseAccum = 0

        for row in n ..< 2 * n {
            var exactMatch = true
            for q in 0 ..< n {
                let sX = getX(row: row, qubit: q)
                let sZ = getZ(row: row, qubit: q)
                if sX != queryX[q] || sZ != queryZ[q] {
                    exactMatch = false
                    break
                }
            }
            if exactMatch {
                return getPhase(row: row) ? -1.0 : 1.0
            }
        }

        var usedRow = ContiguousArray<Bool>(repeating: false, count: n)

        for q in 0 ..< n {
            if !queryX[q] { continue }
            var pivotIdx = -1
            for ri in 0 ..< n {
                if usedRow[ri] { continue }
                if getX(row: n + ri, qubit: q) {
                    pivotIdx = ri
                    break
                }
            }
            if pivotIdx < 0 { continue }
            usedRow[pivotIdx] = true
            let row = n + pivotIdx
            for qi in 0 ..< n {
                let sX = getX(row: row, qubit: qi)
                let sZ = getZ(row: row, qubit: qi)
                phaseAccum += pauliMultPhase(x1: sX, z1: sZ, x2: queryX[qi], z2: queryZ[qi])
                queryX[qi] = queryX[qi] != sX
                queryZ[qi] = queryZ[qi] != sZ
            }
            if getPhase(row: row) { phaseAccum += 2 }
        }

        for q in 0 ..< n {
            if !queryZ[q] { continue }
            var pivotIdx = -1
            for ri in 0 ..< n {
                if usedRow[ri] { continue }
                if getZ(row: n + ri, qubit: q) {
                    pivotIdx = ri
                    break
                }
            }
            if pivotIdx < 0 { return 0.0 }
            usedRow[pivotIdx] = true
            let row = n + pivotIdx
            for qi in 0 ..< n {
                let sX = getX(row: row, qubit: qi)
                let sZ = getZ(row: row, qubit: qi)
                phaseAccum += pauliMultPhase(x1: sX, z1: sZ, x2: queryX[qi], z2: queryZ[qi])
                queryX[qi] = queryX[qi] != sX
                queryZ[qi] = queryZ[qi] != sZ
            }
            if getPhase(row: row) { phaseAccum += 2 }
        }

        let isIdentity = queryX.allSatisfy { !$0 } && queryZ.allSatisfy { !$0 }
        if !isIdentity {
            return 0.0
        }

        let finalPhase = phaseAccum % 4
        return (finalPhase == 2 || finalPhase == 3) ? -1.0 : 1.0
    }

    /// Samples multiple measurement outcomes from the stabilizer state.
    ///
    /// Performs the specified number of independent measurements, returning
    /// the full n-qubit bitstring for each shot.
    ///
    /// **Example:**
    /// ```swift
    /// var tableau = StabilizerTableau(qubits: 2)
    /// tableau.apply(.hadamard, to: 0)
    /// tableau.apply(.cnot, to: [0, 1])
    /// let samples = tableau.sample(shots: 1000, seed: 42)
    /// // samples contains 1000 outcomes, each 0 (for |00>) or 3 (for |11>)
    /// ```
    ///
    /// - Parameters:
    ///   - shots: Number of measurement samples to collect
    ///   - seed: Optional seed for reproducible results
    /// - Returns: Array of n-qubit measurement outcomes (each 0 to 2^n-1)
    /// - Precondition: `shots` must be positive
    /// - Complexity: O(shots * n^2)
    @_eagerMove
    @inlinable
    @_optimize(speed)
    public mutating func sample(shots: Int, seed: UInt64?) -> [Int] {
        ValidationUtilities.validatePositiveInt(shots, name: "shots")

        var results = [Int]()
        results.reserveCapacity(shots)

        let source: GKMersenneTwisterRandomSource? = if let seed {
            GKMersenneTwisterRandomSource(seed: seed)
        } else {
            nil
        }

        for _ in 0 ..< shots {
            var tempTableau = self
            var outcome = 0

            for qubit in 0 ..< n {
                let shotSeed: UInt64? = if let source {
                    UInt64(bitPattern: Int64(source.nextInt()))
                } else {
                    nil
                }
                let bit = tempTableau.measure(qubit, seed: shotSeed)
                outcome |= (bit << qubit)
            }

            results.append(outcome)
        }

        return results
    }

    /// Converts the stabilizer state to a full statevector representation.
    ///
    /// Computes all 2^n complex amplitudes from the stabilizer tableau by
    /// evaluating the amplitude of each computational basis state. The
    /// result is a normalized ``QuantumState`` suitable for statevector
    /// operations and ``BackendDispatch`` interoperability.
    ///
    /// **Example:**
    /// ```swift
    /// var tableau = StabilizerTableau(qubits: 2)
    /// tableau.apply(.hadamard, to: 0)
    /// tableau.apply(.cnot, to: [0, 1])
    /// let state = tableau.toQuantumState()
    /// ```
    ///
    /// - Returns: Full statevector quantum state
    /// - Precondition: Number of qubits must be 20 or fewer
    /// - Complexity: O(n^2 * 2^n) where n is the number of qubits
    /// - SeeAlso: ``amplitude(of:)``
    @_optimize(speed)
    @_effects(readonly)
    @_eagerMove
    public func toQuantumState() -> QuantumState {
        ValidationUtilities.validateStabilizerToStatevectorLimit(n)

        let stateSize = 1 << n
        let amplitudes = [Complex<Double>](unsafeUninitializedCapacity: stateSize) { buffer, count in
            for i in 0 ..< stateSize {
                buffer[i] = self.amplitude(of: i) ?? .zero
            }
            count = stateSize
        }
        return QuantumState(qubits: n, rawAmplitudes: amplitudes)
    }

    /// Human-readable description of the stabilizer tableau.
    ///
    /// **Example:**
    /// ```swift
    /// let tableau = StabilizerTableau(qubits: 2)
    /// print(tableau)
    /// // StabilizerTableau(2 qubits)
    /// ```
    public var description: String {
        "StabilizerTableau(\(n) qubits)"
    }

    /// Returns the word offset for a given tableau row.
    @inlinable
    @inline(__always)
    @_effects(readonly)
    func rowOffset(_ row: Int) -> Int {
        row * wordsPerRow
    }

    /// Reads the X component bit for a qubit in a tableau row.
    @inlinable
    @inline(__always)
    @_effects(readonly)
    func getX(row: Int, qubit: Int) -> Bool {
        let bitIndex = qubit
        let wordIndex = rowOffset(row) + bitIndex / 64
        let bitOffset = bitIndex % 64
        return (tableau[wordIndex] >> bitOffset) & 1 == 1
    }

    /// Sets the X component bit for a qubit in a tableau row.
    @inlinable
    @inline(__always)
    mutating func setX(row: Int, qubit: Int, value: Bool) {
        let bitIndex = qubit
        let wordIndex = rowOffset(row) + bitIndex / 64
        let bitOffset = bitIndex % 64
        let mask: UInt64 = 1 << bitOffset
        tableau[wordIndex] = (tableau[wordIndex] & ~mask) | (UInt64(value ? 1 : 0) << bitOffset)
    }

    /// Reads the Z component bit for a qubit in a tableau row.
    @inlinable
    @inline(__always)
    @_effects(readonly)
    func getZ(row: Int, qubit: Int) -> Bool {
        let bitIndex = n + qubit
        let wordIndex = rowOffset(row) + bitIndex / 64
        let bitOffset = bitIndex % 64
        return (tableau[wordIndex] >> bitOffset) & 1 == 1
    }

    /// Sets the Z component bit for a qubit in a tableau row.
    @inlinable
    @inline(__always)
    mutating func setZ(row: Int, qubit: Int, value: Bool) {
        let bitIndex = n + qubit
        let wordIndex = rowOffset(row) + bitIndex / 64
        let bitOffset = bitIndex % 64
        let mask: UInt64 = 1 << bitOffset
        tableau[wordIndex] = (tableau[wordIndex] & ~mask) | (UInt64(value ? 1 : 0) << bitOffset)
    }

    /// Reads the phase bit for a tableau row.
    @inlinable
    @inline(__always)
    @_effects(readonly)
    func getPhase(row: Int) -> Bool {
        let bitIndex = 2 * n
        let wordIndex = rowOffset(row) + bitIndex / 64
        let bitOffset = bitIndex % 64
        return (tableau[wordIndex] >> bitOffset) & 1 == 1
    }

    /// Sets the phase bit for a tableau row.
    @inlinable
    @inline(__always)
    mutating func setPhase(row: Int, value: Bool) {
        let bitIndex = 2 * n
        let wordIndex = rowOffset(row) + bitIndex / 64
        let bitOffset = bitIndex % 64
        let mask: UInt64 = 1 << bitOffset
        tableau[wordIndex] = (tableau[wordIndex] & ~mask) | (UInt64(value ? 1 : 0) << bitOffset)
    }

    /// Flips the phase bit for a tableau row.
    @inlinable
    @inline(__always)
    mutating func togglePhase(row: Int) {
        let bitIndex = 2 * n
        let wordIndex = rowOffset(row) + bitIndex / 64
        let bitOffset = bitIndex % 64
        tableau[wordIndex] ^= (1 << bitOffset)
    }

    /// Applies Hadamard gate transformation to all tableau rows for a qubit.
    @inlinable
    @_optimize(speed)
    mutating func applyHadamard(_ qubit: Int) {
        for row in 0 ..< 2 * n {
            let x = getX(row: row, qubit: qubit)
            let z = getZ(row: row, qubit: qubit)

            if x, z {
                togglePhase(row: row)
            }

            setX(row: row, qubit: qubit, value: z)
            setZ(row: row, qubit: qubit, value: x)
        }
    }

    /// Applies S gate transformation to all tableau rows for a qubit.
    @inlinable
    @_optimize(speed)
    mutating func applyS(_ qubit: Int) {
        for row in 0 ..< 2 * n {
            let x = getX(row: row, qubit: qubit)
            let z = getZ(row: row, qubit: qubit)

            if x, z {
                togglePhase(row: row)
            }

            setZ(row: row, qubit: qubit, value: x != z)
        }
    }

    /// Applies Pauli-X transformation to all tableau rows for a qubit.
    @inlinable
    @_optimize(speed)
    mutating func applyPauliX(_ qubit: Int) {
        for row in 0 ..< 2 * n {
            if getZ(row: row, qubit: qubit) {
                togglePhase(row: row)
            }
        }
    }

    /// Applies Pauli-Z transformation to all tableau rows for a qubit.
    @inlinable
    @_optimize(speed)
    mutating func applyPauliZ(_ qubit: Int) {
        for row in 0 ..< 2 * n {
            if getX(row: row, qubit: qubit) {
                togglePhase(row: row)
            }
        }
    }

    /// Applies Pauli-Y transformation to all tableau rows for a qubit.
    @inlinable
    @_optimize(speed)
    mutating func applyPauliY(_ qubit: Int) {
        for row in 0 ..< 2 * n {
            let x = getX(row: row, qubit: qubit)
            let z = getZ(row: row, qubit: qubit)
            if x != z {
                togglePhase(row: row)
            }
        }
    }

    /// Applies CNOT transformation to all tableau rows for control and target qubits.
    @inlinable
    @_optimize(speed)
    mutating func applyCNOT(control: Int, target: Int) {
        for row in 0 ..< 2 * n {
            let xc = getX(row: row, qubit: control)
            let zc = getZ(row: row, qubit: control)
            let xt = getX(row: row, qubit: target)
            let zt = getZ(row: row, qubit: target)

            if xc, zt, xt == zc {
                togglePhase(row: row)
            }

            setX(row: row, qubit: target, value: xt != xc)
            setZ(row: row, qubit: control, value: zc != zt)
        }
    }

    /// Applies CZ transformation to all tableau rows for two qubits.
    @inlinable
    @_optimize(speed)
    mutating func applyCZ(qubit1: Int, qubit2: Int) {
        for row in 0 ..< 2 * n {
            let x1 = getX(row: row, qubit: qubit1)
            let z1 = getZ(row: row, qubit: qubit1)
            let x2 = getX(row: row, qubit: qubit2)
            let z2 = getZ(row: row, qubit: qubit2)

            if x1, x2, z1 == z2 {
                togglePhase(row: row)
            }

            setZ(row: row, qubit: qubit1, value: z1 != x2)
            setZ(row: row, qubit: qubit2, value: z2 != x1)
        }
    }

    /// Multiplies source row into target row with phase tracking.
    @inlinable
    @_optimize(speed)
    mutating func rowMultiply(target: Int, source: Int) {
        var phaseAccum = 0

        for q in 0 ..< n {
            let x1 = getX(row: source, qubit: q)
            let z1 = getZ(row: source, qubit: q)
            let x2 = getX(row: target, qubit: q)
            let z2 = getZ(row: target, qubit: q)
            phaseAccum += pauliMultPhase(x1: x1, z1: z1, x2: x2, z2: z2)
        }

        let sourcePhase = getPhase(row: source) ? 2 : 0
        let targetPhase = getPhase(row: target) ? 2 : 0

        let srcOffset = rowOffset(source)
        let dstOffset = rowOffset(target)
        let simdCount = wordsPerRow / 4
        for i in 0 ..< simdCount {
            let idx = i * 4
            let a = SIMD4<UInt64>(
                tableau[dstOffset + idx],
                tableau[dstOffset + idx + 1],
                tableau[dstOffset + idx + 2],
                tableau[dstOffset + idx + 3],
            )
            let b = SIMD4<UInt64>(
                tableau[srcOffset + idx],
                tableau[srcOffset + idx + 1],
                tableau[srcOffset + idx + 2],
                tableau[srcOffset + idx + 3],
            )
            let result = a ^ b
            tableau[dstOffset + idx] = result[0]
            tableau[dstOffset + idx + 1] = result[1]
            tableau[dstOffset + idx + 2] = result[2]
            tableau[dstOffset + idx + 3] = result[3]
        }
        for i in (simdCount * 4) ..< wordsPerRow {
            tableau[dstOffset + i] ^= tableau[srcOffset + i]
        }

        let newPhase = (phaseAccum + sourcePhase + targetPhase) % 4

        setPhase(row: target, value: newPhase == 2)
    }

    /// Lookup table for Pauli product phase accumulation.
    @usableFromInline
    static let pauliPhaseLUT: ContiguousArray<Int> = [
        0, 0, 0, 0,
        0, 0, 1, 3,
        0, 3, 0, 1,
        0, 1, 3, 0,
    ]

    /// Returns phase contribution from multiplying two Pauli operators.
    @inlinable
    @inline(__always)
    @_effects(readonly)
    func pauliMultPhase(x1: Bool, z1: Bool, x2: Bool, z2: Bool) -> Int {
        let p1 = (x1 ? 1 : 0) | (z1 ? 2 : 0)
        let p2 = (x2 ? 1 : 0) | (z2 ? 2 : 0)
        return Self.pauliPhaseLUT[p1 * 4 + p2]
    }

    /// Copies all words from one tableau row to another.
    @inlinable
    @_optimize(speed)
    mutating func copyRow(from source: Int, to target: Int) {
        let srcOffset = rowOffset(source)
        let dstOffset = rowOffset(target)
        let simdCount = wordsPerRow / 4
        for i in 0 ..< simdCount {
            let offset = i * 4
            let a = SIMD4<UInt64>(
                tableau[srcOffset + offset],
                tableau[srcOffset + offset + 1],
                tableau[srcOffset + offset + 2],
                tableau[srcOffset + offset + 3],
            )
            tableau[dstOffset + offset] = a[0]
            tableau[dstOffset + offset + 1] = a[1]
            tableau[dstOffset + offset + 2] = a[2]
            tableau[dstOffset + offset + 3] = a[3]
        }
        for i in (simdCount * 4) ..< wordsPerRow {
            tableau[dstOffset + i] = tableau[srcOffset + i]
        }
    }

    /// Zeroes all words in a tableau row.
    @inlinable
    @_optimize(speed)
    mutating func clearRow(_ row: Int) {
        let offset = rowOffset(row)
        let simdCount = wordsPerRow / 4
        let zero = SIMD4<UInt64>(repeating: 0)
        for i in 0 ..< simdCount {
            let idx = i * 4
            tableau[offset + idx] = zero[0]
            tableau[offset + idx + 1] = zero[1]
            tableau[offset + idx + 2] = zero[2]
            tableau[offset + idx + 3] = zero[3]
        }
        for i in (simdCount * 4) ..< wordsPerRow {
            tableau[offset + i] = 0
        }
    }
}
