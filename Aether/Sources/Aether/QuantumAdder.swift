// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Quantum addition circuit primitives with variant selection based on ancilla budget.
///
/// Provides two adder variants optimized for different resource tradeoffs. The ripple-carry
/// variant (Cuccaro et al.) computes |a⟩|b⟩ → |a⟩|a+b⟩ in-place with a single ancilla at
/// O(n) depth using the MAJ-UMA gate pattern. The carry-lookahead variant computes
/// |a⟩|b⟩|0⟩ → |a⟩|b⟩|a+b⟩ out-of-place, preserving both inputs at the cost of O(n)
/// ancillas for carry computation via a prefix tree with Bennett-style cleanup.
///
/// Selection follows the ancilla budget: use ``Variant/carryLookahead`` when ancillas are
/// available and both inputs must be preserved, otherwise use ``Variant/rippleCarry`` for
/// minimal resource usage.
///
/// **Example:**
/// ```swift
/// let variant = QuantumAdder.optimalVariant(bits: 4, availableAncillas: 10)
/// let adder = QuantumAdder.circuit(variant, bits: 4)
/// let total = QuantumAdder.qubitCount(variant, bits: 4)
/// ```
///
/// - SeeAlso: ``QuantumComparator``
/// - SeeAlso: ``QuantumMultiplier``
/// - SeeAlso: ``QuantumCircuit/adder(bits:variant:)``
public enum QuantumAdder {
    /// Adder circuit variant controlling the depth-ancilla tradeoff.
    ///
    /// The two variants implement the same arithmetic operation (n-bit unsigned addition)
    /// with different resource profiles. ``rippleCarry`` overwrites the second operand
    /// with the sum using one ancilla; ``carryLookahead`` preserves both operands and
    /// writes the sum to a separate output register using O(n) ancillas.
    @frozen public enum Variant: Sendable {
        /// Carry-lookahead adder preserving both inputs (Draper et al. prefix tree).
        ///
        /// Computes |a⟩|b⟩|0⟩ → |a⟩|b⟩|a+b⟩ using a prefix tree for carry computation
        /// with O(n) ancillas cleaned via Bennett's trick. Both input registers are preserved.
        /// Use when both operands are needed after addition (e.g., modular arithmetic).
        case carryLookahead

        /// Ripple-carry adder with minimal ancilla usage (Cuccaro MAJ-UMA).
        ///
        /// Computes |a⟩|b⟩ → |a⟩|a+b⟩ in-place using the Majority-UnMajorityAdd gate
        /// pattern with a single ancilla qubit. O(n) depth from sequential carry propagation.
        /// Preferred when ancilla budget is exhausted or in-place operation is desired.
        case rippleCarry
    }

    /// Creates an adder circuit with auto-assigned qubit registers.
    ///
    /// Allocates qubit registers sequentially: a occupies [0, bits), b occupies [bits, 2·bits).
    /// For ``Variant/rippleCarry``, a single ancilla occupies qubit 2·bits and the sum overwrites
    /// register b. For ``Variant/carryLookahead``, the output occupies [2·bits, 3·bits) with
    /// internal ancillas above that, and both a and b are preserved.
    ///
    /// **Example:**
    /// ```swift
    /// let adder = QuantumAdder.circuit(.rippleCarry, bits: 4)
    /// let total = QuantumAdder.qubitCount(.rippleCarry, bits: 4)
    /// let state = adder.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - variant: Adder variant to use
    ///   - bits: Number of bits per operand (≥ 1)
    /// - Returns: Quantum circuit implementing n-bit addition
    /// - Precondition: bits ≥ 1
    /// - Precondition: Total qubit count ≤ 30
    /// - Complexity: O(n) gates for rippleCarry, O(n log n) for carryLookahead
    ///
    /// - SeeAlso: ``qubitCount(_:bits:)``
    /// - SeeAlso: ``resultQubits(_:bits:)``
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func circuit(_ variant: Variant, bits: Int) -> QuantumCircuit {
        ValidationUtilities.validatePositiveInt(bits, name: "bits")
        let total = qubitCount(variant, bits: bits)
        ValidationUtilities.validateUpperBound(total, max: 30, name: "total qubit count")

        let a = Array(0 ..< bits)
        let b = Array(bits ..< 2 * bits)

        switch variant {
        case .rippleCarry:
            let ancilla = 2 * bits
            return buildRippleCarry(a: a, b: b, ancilla: ancilla, totalQubits: total)
        case .carryLookahead:
            let out = Array(2 * bits ..< 3 * bits)
            let gen = Array(3 * bits ..< 4 * bits)
            let prop = Array(4 * bits ..< 5 * bits)
            return buildCarryLookahead(a: a, b: b, out: out, gen: gen, prop: prop, totalQubits: total)
        }
    }

    /// Creates an adder circuit with explicit qubit register assignments.
    ///
    /// For ``Variant/rippleCarry``, registers a and b must have equal length; ancilla and
    /// internal qubits are auto-allocated above the maximum index. The sum overwrites b.
    /// For ``Variant/carryLookahead``, both a and b are preserved and the sum is placed in
    /// auto-allocated output qubits.
    ///
    /// **Example:**
    /// ```swift
    /// let a = [0, 1, 2, 3]
    /// let adder = QuantumAdder.circuit(.rippleCarry, a: a, b: [4, 5, 6, 7])
    /// let state = adder.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - variant: Adder variant to use
    ///   - a: Qubit indices for first operand (LSB first)
    ///   - b: Qubit indices for second operand (LSB first)
    /// - Returns: Quantum circuit implementing n-bit addition
    /// - Precondition: a and b have equal non-zero length
    /// - Precondition: a and b registers must not overlap
    /// - Precondition: All qubit indices ≥ 0
    /// - Complexity: O(n) gates for rippleCarry, O(n log n) for carryLookahead
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func circuit(_ variant: Variant, a: [Int], b: [Int]) -> QuantumCircuit {
        ValidationUtilities.validateNonEmpty(a, name: "a")
        ValidationUtilities.validateEqualCounts(a, b, name1: "a", name2: "b")
        ValidationUtilities.validateNonNegativeQubits(a)
        ValidationUtilities.validateNonNegativeQubits(b)
        ValidationUtilities.validateUniqueQubits(a)
        ValidationUtilities.validateUniqueQubits(b)
        ValidationUtilities.validateDisjointRegisters(a, b, nameA: "a", nameB: "b")

        let bits = a.count
        let maxQubit = max(a.max()!, b.max()!) // safe: a, b validated non-empty

        switch variant {
        case .rippleCarry:
            let ancilla = maxQubit + 1
            let total = ancilla + 1
            ValidationUtilities.validateUpperBound(total, max: 30, name: "total qubit count")
            return buildRippleCarry(a: a, b: b, ancilla: ancilla, totalQubits: total)
        case .carryLookahead:
            let outStart = maxQubit + 1
            let out = Array(outStart ..< outStart + bits)
            let gen = Array(outStart + bits ..< outStart + 2 * bits)
            let prop = Array(outStart + 2 * bits ..< outStart + 3 * bits)
            let total = outStart + 3 * bits
            ValidationUtilities.validateUpperBound(total, max: 30, name: "total qubit count")
            return buildCarryLookahead(a: a, b: b, out: out, gen: gen, prop: prop, totalQubits: total)
        }
    }

    /// Creates a controlled adder circuit with auto-assigned registers.
    ///
    /// Every gate in the adder circuit is conditioned on the control qubit, making the
    /// entire addition conditional. When the control is |0⟩ all registers remain unchanged.
    /// Used in modular arithmetic for Shor's algorithm where addition is controlled on
    /// ancilla qubits.
    ///
    /// **Example:**
    /// ```swift
    /// let bits = 4
    /// let ctrlAdder = QuantumAdder.circuit(.rippleCarry, bits: bits, controlledBy: 12)
    /// let state = ctrlAdder.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - variant: Adder variant to use
    ///   - bits: Number of bits per operand (≥ 1)
    ///   - control: Control qubit index
    /// - Returns: Quantum circuit implementing controlled n-bit addition
    /// - Precondition: bits ≥ 1
    /// - Precondition: Total qubit count ≤ 30
    /// - Complexity: O(n) gates for rippleCarry, O(n log n) for carryLookahead
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func circuit(
        _ variant: Variant,
        bits: Int,
        controlledBy control: Int,
    ) -> QuantumCircuit {
        ValidationUtilities.validatePositiveInt(bits, name: "bits")
        ValidationUtilities.validateNonNegativeInt(control, name: "control")

        let a = Array(0 ..< bits)
        let b = Array(bits ..< 2 * bits)
        ValidationUtilities.validateQubitNotInRegisters(control, registers: [a, b], qubitName: "control")

        switch variant {
        case .rippleCarry:
            let ancilla = skipControl(2 * bits, control: control)
            var cccxAnc = ancilla + 1
            if cccxAnc == control { cccxAnc += 1 }
            let total = max(cccxAnc + 1, control + 1)
            ValidationUtilities.validateUpperBound(total, max: 30, name: "total qubit count")
            return buildControlledRippleCarry(
                a: a, b: b, ancilla: ancilla, control: control, cccxAncilla: cccxAnc, totalQubits: total,
            )
        case .carryLookahead:
            let out = allocateRegister(start: 2 * bits, count: bits, avoiding: control)
            let gen = allocateRegister(start: out.last! + 1, count: bits, avoiding: control) // safe: out has bits elements
            let prop = allocateRegister(start: gen.last! + 1, count: bits, avoiding: control) // safe: gen has bits elements
            var cccxAnc = prop.last! + 1 // safe: prop has bits elements
            if cccxAnc == control { cccxAnc += 1 }
            let total = max(cccxAnc + 1, control + 1)
            ValidationUtilities.validateUpperBound(total, max: 30, name: "total qubit count")
            return buildControlledCarryLookahead(
                a: a, b: b, out: out, gen: gen, prop: prop,
                control: control, cccxAncilla: cccxAnc, totalQubits: total,
            )
        }
    }

    /// Total qubit count required for the adder circuit.
    ///
    /// Returns the number of qubits including input registers, output register (for
    /// carryLookahead), and all ancillas. For ``Variant/rippleCarry``: 2n+1.
    /// For ``Variant/carryLookahead``: 5n.
    ///
    /// **Example:**
    /// ```swift
    /// let ripple = QuantumAdder.qubitCount(.rippleCarry, bits: 4)
    /// let cla = QuantumAdder.qubitCount(.carryLookahead, bits: 4)
    /// let ratio = Double(cla) / Double(ripple)
    /// ```
    ///
    /// - Parameters:
    ///   - variant: Adder variant
    ///   - bits: Number of bits per operand
    /// - Returns: Total qubits required
    /// - Precondition: bits ≥ 1
    /// - Complexity: O(1)
    @_effects(readonly)
    public static func qubitCount(_ variant: Variant, bits: Int) -> Int {
        ValidationUtilities.validatePositiveInt(bits, name: "bits")
        switch variant {
        case .rippleCarry: return 2 * bits + 1
        case .carryLookahead: return 5 * bits
        }
    }

    /// Number of ancilla qubits required beyond the input/output registers.
    ///
    /// For ``Variant/rippleCarry``: 1 (carry-in qubit). For ``Variant/carryLookahead``:
    /// 2n (generate and propagate registers, cleaned to zero after computation).
    ///
    /// **Example:**
    /// ```swift
    /// let ripple = QuantumAdder.ancillaCount(.rippleCarry, bits: 4)
    /// let cla = QuantumAdder.ancillaCount(.carryLookahead, bits: 4)
    /// let savings = cla - ripple
    /// ```
    ///
    /// - Parameters:
    ///   - variant: Adder variant
    ///   - bits: Number of bits per operand
    /// - Returns: Number of ancilla qubits
    /// - Precondition: bits ≥ 1
    /// - Complexity: O(1)
    @_effects(readonly)
    public static func ancillaCount(_ variant: Variant, bits: Int) -> Int {
        ValidationUtilities.validatePositiveInt(bits, name: "bits")
        switch variant {
        case .rippleCarry: return 1
        case .carryLookahead: return 2 * bits
        }
    }

    /// Selects the optimal adder variant based on available ancilla budget.
    ///
    /// Returns ``Variant/carryLookahead`` when at least 2n ancillas are available (plus n
    /// output qubits), otherwise returns ``Variant/rippleCarry``. The carry-lookahead
    /// variant preserves both inputs at the cost of more ancillas, while ripple-carry
    /// overwrites the second operand.
    ///
    /// **Example:**
    /// ```swift
    /// let v = QuantumAdder.optimalVariant(bits: 4, availableAncillas: 10)
    /// let circuit = QuantumAdder.circuit(v, bits: 4)
    /// let state = circuit.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - bits: Number of bits per operand
    ///   - availableAncillas: Number of available ancilla qubits
    /// - Returns: Recommended adder variant
    /// - Precondition: bits ≥ 1
    /// - Precondition: availableAncillas ≥ 0
    /// - Complexity: O(1)
    @_effects(readonly)
    public static func optimalVariant(bits: Int, availableAncillas: Int) -> Variant {
        ValidationUtilities.validatePositiveInt(bits, name: "bits")
        ValidationUtilities.validateNonNegativeInt(availableAncillas, name: "availableAncillas")
        let claNeeded = 3 * bits
        if availableAncillas >= claNeeded, 5 * bits <= 30 {
            return .carryLookahead
        }
        return .rippleCarry
    }

    /// Qubit indices holding the addition result for the given variant.
    ///
    /// For ``Variant/rippleCarry``, the result overwrites the b register [bits, 2·bits).
    /// For ``Variant/carryLookahead``, the result occupies the output register [2·bits, 3·bits).
    ///
    /// **Example:**
    /// ```swift
    /// let ripple = QuantumAdder.resultQubits(.rippleCarry, bits: 4)
    /// let cla = QuantumAdder.resultQubits(.carryLookahead, bits: 4)
    /// let overlap = Set(ripple).intersection(Set(cla))
    /// ```
    ///
    /// - Parameters:
    ///   - variant: Adder variant
    ///   - bits: Number of bits per operand
    /// - Returns: Array of qubit indices containing the sum (LSB first)
    /// - Precondition: bits ≥ 1
    /// - Complexity: O(n)
    @_effects(readonly)
    @_eagerMove
    public static func resultQubits(_ variant: Variant, bits: Int) -> [Int] {
        ValidationUtilities.validatePositiveInt(bits, name: "bits")
        switch variant {
        case .rippleCarry: return Array(bits ..< 2 * bits)
        case .carryLookahead: return Array(2 * bits ..< 3 * bits)
        }
    }

    // MARK: - Ripple-Carry Implementation (Cuccaro MAJ-UMA)

    /// Builds Cuccaro ripple-carry adder: |a⟩|b⟩|0⟩ → |a⟩|a+b⟩|0⟩.
    private static func buildRippleCarry(
        a: [Int], b: [Int], ancilla: Int, totalQubits: Int,
    ) -> QuantumCircuit {
        let n = a.count
        var circuit = QuantumCircuit(qubits: totalQubits)

        if n == 1 {
            circuit.append(.cnot, to: [a[0], b[0]])
            return circuit
        }

        appendMAJ(to: &circuit, x: ancilla, y: b[0], z: a[0])
        for i in 1 ..< n {
            appendMAJ(to: &circuit, x: a[i - 1], y: b[i], z: a[i])
        }

        for i in stride(from: n - 1, through: 1, by: -1) {
            appendUMA(to: &circuit, x: a[i - 1], y: b[i], z: a[i])
        }
        appendUMA(to: &circuit, x: ancilla, y: b[0], z: a[0])

        return circuit
    }

    /// Builds controlled Cuccaro ripple-carry adder.
    private static func buildControlledRippleCarry(
        a: [Int], b: [Int], ancilla: Int, control: Int, cccxAncilla: Int, totalQubits: Int,
    ) -> QuantumCircuit {
        let n = a.count
        var circuit = QuantumCircuit(qubits: totalQubits)

        if n == 1 {
            circuit.append(.toffoli, to: [control, a[0], b[0]])
            return circuit
        }

        appendControlledMAJ(to: &circuit, x: ancilla, y: b[0], z: a[0], control: control, cccxAncilla: cccxAncilla)
        for i in 1 ..< n {
            appendControlledMAJ(to: &circuit, x: a[i - 1], y: b[i], z: a[i], control: control, cccxAncilla: cccxAncilla)
        }

        for i in stride(from: n - 1, through: 1, by: -1) {
            appendControlledUMA(to: &circuit, x: a[i - 1], y: b[i], z: a[i], control: control, cccxAncilla: cccxAncilla)
        }
        appendControlledUMA(to: &circuit, x: ancilla, y: b[0], z: a[0], control: control, cccxAncilla: cccxAncilla)

        return circuit
    }

    // MARK: - Carry-Lookahead Implementation (Prefix Tree + Bennett)

    /// Builds carry-lookahead adder: |a⟩|b⟩|0⟩ → |a⟩|b⟩|a+b⟩ with ancilla cleanup.
    private static func buildCarryLookahead(
        a: [Int], b: [Int], out: [Int], gen: [Int], prop: [Int], totalQubits: Int,
    ) -> QuantumCircuit {
        let n = a.count
        var circuit = QuantumCircuit(qubits: totalQubits)

        if n == 1 {
            circuit.append(.cnot, to: [a[0], out[0]])
            circuit.append(.cnot, to: [b[0], out[0]])
            return circuit
        }

        appendForwardComputation(to: &circuit, a: a, b: b, gen: gen, prop: prop, n: n)
        appendSumComputation(to: &circuit, prop: prop, gen: gen, out: out, n: n)
        appendReverseComputation(to: &circuit, a: a, b: b, gen: gen, prop: prop, n: n)

        return circuit
    }

    /// Builds controlled carry-lookahead adder.
    private static func buildControlledCarryLookahead(
        a: [Int], b: [Int], out: [Int], gen: [Int], prop: [Int],
        control: Int, cccxAncilla: Int, totalQubits: Int,
    ) -> QuantumCircuit {
        let n = a.count
        var circuit = QuantumCircuit(qubits: totalQubits)

        if n == 1 {
            circuit.append(.toffoli, to: [control, a[0], out[0]])
            circuit.append(.toffoli, to: [control, b[0], out[0]])
            return circuit
        }

        appendControlledForward(to: &circuit, a: a, b: b, gen: gen, prop: prop, control: control, cccxAncilla: cccxAncilla, n: n)
        appendControlledSum(to: &circuit, prop: prop, gen: gen, out: out, control: control, n: n)
        appendControlledReverse(to: &circuit, a: a, b: b, gen: gen, prop: prop, control: control, cccxAncilla: cccxAncilla, n: n)

        return circuit
    }

    // MARK: - Forward/Reverse Computation Helpers

    /// Forward computation: generate, propagate, prefix tree.
    private static func appendForwardComputation(
        to circuit: inout QuantumCircuit,
        a: [Int], b: [Int], gen: [Int], prop: [Int], n: Int,
    ) {
        for i in 0 ..< n {
            circuit.append(.toffoli, to: [a[i], b[i], gen[i]])
        }
        for i in 0 ..< n {
            circuit.append(.cnot, to: [a[i], prop[i]])
            circuit.append(.cnot, to: [b[i], prop[i]])
        }
        for i in 1 ..< n {
            circuit.append(.toffoli, to: [prop[i], gen[i - 1], gen[i]])
        }
    }

    /// Sum computation: XOR propagate and carry into output.
    private static func appendSumComputation(
        to circuit: inout QuantumCircuit,
        prop: [Int], gen: [Int], out: [Int], n: Int,
    ) {
        for i in 0 ..< n {
            circuit.append(.cnot, to: [prop[i], out[i]])
        }
        for i in 1 ..< n {
            circuit.append(.cnot, to: [gen[i - 1], out[i]])
        }
    }

    /// Reverse computation: undo prefix tree, propagate, generate.
    private static func appendReverseComputation(
        to circuit: inout QuantumCircuit,
        a: [Int], b: [Int], gen: [Int], prop: [Int], n: Int,
    ) {
        for i in stride(from: n - 1, through: 1, by: -1) {
            circuit.append(.toffoli, to: [prop[i], gen[i - 1], gen[i]])
        }
        for i in 0 ..< n {
            circuit.append(.cnot, to: [b[i], prop[i]])
            circuit.append(.cnot, to: [a[i], prop[i]])
        }
        for i in 0 ..< n {
            circuit.append(.toffoli, to: [a[i], b[i], gen[i]])
        }
    }

    /// Controlled forward computation using manual CCCX decomposition.
    private static func appendControlledForward(
        to circuit: inout QuantumCircuit,
        a: [Int], b: [Int], gen: [Int], prop: [Int], control: Int, cccxAncilla: Int, n: Int,
    ) {
        for i in 0 ..< n {
            appendCCCX(to: &circuit, c0: control, c1: a[i], c2: b[i], target: gen[i], ancilla: cccxAncilla)
        }
        for i in 0 ..< n {
            circuit.append(.toffoli, to: [control, a[i], prop[i]])
            circuit.append(.toffoli, to: [control, b[i], prop[i]])
        }
        for i in 1 ..< n {
            appendCCCX(to: &circuit, c0: control, c1: prop[i], c2: gen[i - 1], target: gen[i], ancilla: cccxAncilla)
        }
    }

    /// Controlled sum computation.
    private static func appendControlledSum(
        to circuit: inout QuantumCircuit,
        prop: [Int], gen: [Int], out: [Int], control: Int, n: Int,
    ) {
        for i in 0 ..< n {
            circuit.append(.toffoli, to: [control, prop[i], out[i]])
        }
        for i in 1 ..< n {
            circuit.append(.toffoli, to: [control, gen[i - 1], out[i]])
        }
    }

    /// Controlled reverse computation using manual CCCX decomposition.
    private static func appendControlledReverse(
        to circuit: inout QuantumCircuit,
        a: [Int], b: [Int], gen: [Int], prop: [Int], control: Int, cccxAncilla: Int, n: Int,
    ) {
        for i in stride(from: n - 1, through: 1, by: -1) {
            appendCCCX(to: &circuit, c0: control, c1: prop[i], c2: gen[i - 1], target: gen[i], ancilla: cccxAncilla)
        }
        for i in 0 ..< n {
            circuit.append(.toffoli, to: [control, b[i], prop[i]])
            circuit.append(.toffoli, to: [control, a[i], prop[i]])
        }
        for i in 0 ..< n {
            appendCCCX(to: &circuit, c0: control, c1: a[i], c2: b[i], target: gen[i], ancilla: cccxAncilla)
        }
    }

    // MARK: - Gate Primitives

    /// Majority gate: MAJ(x, y, z) = CNOT(z,y); CNOT(z,x); Toffoli(x,y,z).
    private static func appendMAJ(to circuit: inout QuantumCircuit, x: Int, y: Int, z: Int) {
        circuit.append(.cnot, to: [z, y])
        circuit.append(.cnot, to: [z, x])
        circuit.append(.toffoli, to: [x, y, z])
    }

    /// UnMajority-Add gate: UMA(x, y, z) = Toffoli(x,y,z); CNOT(z,x); CNOT(x,y).
    private static func appendUMA(to circuit: inout QuantumCircuit, x: Int, y: Int, z: Int) {
        circuit.append(.toffoli, to: [x, y, z])
        circuit.append(.cnot, to: [z, x])
        circuit.append(.cnot, to: [x, y])
    }

    /// Controlled Majority gate using manual CCCX decomposition.
    private static func appendControlledMAJ(
        to circuit: inout QuantumCircuit, x: Int, y: Int, z: Int, control: Int, cccxAncilla: Int,
    ) {
        circuit.append(.toffoli, to: [control, z, y])
        circuit.append(.toffoli, to: [control, z, x])
        appendCCCX(to: &circuit, c0: control, c1: x, c2: y, target: z, ancilla: cccxAncilla)
    }

    /// Controlled UnMajority-Add gate using manual CCCX decomposition.
    private static func appendControlledUMA(
        to circuit: inout QuantumCircuit, x: Int, y: Int, z: Int, control: Int, cccxAncilla: Int,
    ) {
        appendCCCX(to: &circuit, c0: control, c1: x, c2: y, target: z, ancilla: cccxAncilla)
        circuit.append(.toffoli, to: [control, z, x])
        circuit.append(.toffoli, to: [control, x, y])
    }

    /// Manual 3-control X decomposition via Toffoli ladder with dedicated ancilla.
    private static func appendCCCX(
        to circuit: inout QuantumCircuit,
        c0: Int, c1: Int, c2: Int, target: Int, ancilla: Int,
    ) {
        circuit.append(.toffoli, to: [c0, c1, ancilla])
        circuit.append(.toffoli, to: [ancilla, c2, target])
        circuit.append(.toffoli, to: [c0, c1, ancilla])
    }

    // MARK: - Allocation Helpers

    /// Returns index, skipping over control if they collide.
    private static func skipControl(_ index: Int, control: Int) -> Int {
        index == control ? index + 1 : index
    }

    /// Allocates a contiguous register avoiding the control qubit.
    private static func allocateRegister(start: Int, count: Int, avoiding control: Int) -> [Int] {
        var result = [Int]()
        result.reserveCapacity(count)
        var idx = start
        for _ in 0 ..< count {
            if idx == control { idx += 1 }
            result.append(idx)
            idx += 1
        }
        return result
    }
}

public extension QuantumCircuit {
    /// Creates a quantum adder circuit with the specified variant and bit width.
    ///
    /// Convenience factory delegating to ``QuantumAdder/circuit(_:bits:)``. For
    /// ``QuantumAdder/Variant/rippleCarry``, the sum overwrites the b register. For
    /// ``QuantumAdder/Variant/carryLookahead``, the sum occupies a separate output register.
    ///
    /// **Example:**
    /// ```swift
    /// let adder = QuantumCircuit.adder(bits: 4, variant: .rippleCarry)
    /// let qubits = adder.qubits
    /// let state = adder.execute()
    /// ```
    ///
    /// - Parameters:
    ///   - bits: Number of bits per operand (≥ 1)
    ///   - variant: Adder variant (default: rippleCarry)
    /// - Returns: Quantum circuit implementing n-bit addition
    /// - Precondition: bits ≥ 1
    /// - Precondition: Total qubit count ≤ 30
    /// - Complexity: O(n) gates for rippleCarry, O(n log n) for carryLookahead
    ///
    /// - SeeAlso: ``QuantumAdder``
    @_eagerMove
    static func adder(bits: Int, variant: QuantumAdder.Variant = .rippleCarry) -> QuantumCircuit {
        QuantumAdder.circuit(variant, bits: bits)
    }
}
