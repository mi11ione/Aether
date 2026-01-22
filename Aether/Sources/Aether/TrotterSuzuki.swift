// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Trotter-Suzuki decomposition orders for time evolution approximation.
///
/// Defines the approximation order for product formula decomposition of the time evolution
/// operator exp(-iHt). Higher orders achieve better accuracy at the cost of increased circuit
/// depth. The error scaling is O(t^(p+1)) for order p, making higher orders advantageous for
/// longer evolution times or when high precision is required.
///
/// - ``first``: Simple product formula with O(t^2) error per step
/// - ``second``: Symmetric (Strang) splitting with O(t^3) error per step
/// - ``fourth``: Yoshida's formula with O(t^5) error per step
/// - ``sixth``: Suzuki's recursive formula with O(t^7) error per step
///
/// **Example:**
/// ```swift
/// let config = TrotterConfiguration(order: .fourth, steps: 10)
/// let circuit = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 4, config: config)
/// ```
///
/// - SeeAlso: ``TrotterConfiguration``
/// - SeeAlso: ``TrotterSuzuki``
@frozen
public enum TrotterOrder: Int, Sendable {
    /// First-order Trotter: S_1(t) = prod_k exp(-iH_k t), error O(t^2)
    case first = 1

    /// Second-order symmetric Trotter: S_2(t) = S_1(t/2)^dag S_1(t/2), error O(t^3)
    case second = 2

    /// Fourth-order Yoshida: recursive composition of S_2, error O(t^5)
    case fourth = 4

    /// Sixth-order Suzuki: recursive composition, error O(t^7)
    case sixth = 6
}

/// Configuration for Trotter decomposition.
///
/// Encapsulates all parameters controlling the Trotter-Suzuki approximation: decomposition
/// order, number of Trotter steps, term ordering strategy, and coefficient filtering threshold.
/// More steps reduce per-step error but increase circuit depth linearly.
///
/// **Example:**
/// ```swift
/// let config = TrotterConfiguration(
///     order: .second,
///     steps: 20,
///     sortByCommutation: true,
///     coefficientThreshold: 1e-12
/// )
/// let circuit = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 4, config: config)
/// ```
///
/// - SeeAlso: ``TrotterOrder``
/// - SeeAlso: ``TrotterSuzuki``
@frozen
public struct TrotterConfiguration: Sendable {
    /// Decomposition order determining approximation accuracy
    public let order: TrotterOrder

    /// Number of Trotter steps (total evolution time divided among steps)
    public let steps: Int

    /// When true, reorders terms to group commuting operators together
    public let sortByCommutation: Bool

    /// Terms with |coefficient| below this threshold are skipped
    public let coefficientThreshold: Double

    /// Creates a Trotter configuration with specified parameters.
    ///
    /// **Example:**
    /// ```swift
    /// let config = TrotterConfiguration(order: .fourth, steps: 10)
    /// ```
    ///
    /// - Parameters:
    ///   - order: Decomposition order (default: .second)
    ///   - steps: Number of Trotter steps (default: 1)
    ///   - sortByCommutation: Reorder terms by commutation (default: false)
    ///   - coefficientThreshold: Minimum coefficient magnitude (default: 1e-15)
    @inlinable
    public init(
        order: TrotterOrder = .second,
        steps: Int = 1,
        sortByCommutation: Bool = false,
        coefficientThreshold: Double = 1e-15,
    ) {
        self.order = order
        self.steps = steps
        self.sortByCommutation = sortByCommutation
        self.coefficientThreshold = coefficientThreshold
    }
}

/// Trotter-Suzuki decomposition for Hamiltonian time evolution.
///
/// Implements product formula approximations for the unitary time evolution operator
/// U(t) = exp(-iHt) where H is a Hamiltonian expressed as a sum of Pauli strings.
/// The decomposition breaks the exponential of a sum into a product of exponentials
/// of individual terms, enabling efficient quantum circuit implementation.
///
/// The approximation error depends on the order and step size:
/// - First-order: O(t^2/steps) - simple but least accurate
/// - Second-order: O(t^3/steps^2) - symmetric splitting, good balance
/// - Fourth-order: O(t^5/steps^4) - Yoshida's recursive formula
/// - Sixth-order: O(t^7/steps^6) - highest accuracy, deepest circuits
///
/// Each Pauli string exponential exp(-i theta P) is implemented using basis rotations
/// to transform non-Z Paulis to Z, a CNOT ladder to entangle qubits, an Rz rotation
/// encoding the angle, then the inverse CNOT ladder and basis rotations.
///
/// **Example:**
/// ```swift
/// let hamiltonian = Observable(terms: [
///     (0.5, PauliString(.z(0), .z(1))),
///     (-0.3, PauliString(.x(0))),
///     (-0.3, PauliString(.x(1)))
/// ])
/// let config = TrotterConfiguration(order: .second, steps: 10)
/// let circuit = TrotterSuzuki.evolve(hamiltonian, time: 1.0, qubits: 2, config: config)
/// let finalState = circuit.execute()
/// ```
///
/// - SeeAlso: ``TrotterConfiguration``
/// - SeeAlso: ``TrotterOrder``
/// - SeeAlso: ``Observable``
public enum TrotterSuzuki {
    private static let halfPi = Double.pi / 2
    private static let negHalfPi = -Double.pi / 2
    private static let maxSteps = 10000

    private static let yoshidaS = 1.0 / (4.0 - pow(4.0, 1.0 / 3.0))

    private static let suzukiS6 = 1.0 / (4.0 - pow(4.0, 1.0 / 5.0))

    /// Evolve a quantum state under Hamiltonian time evolution using Trotter-Suzuki decomposition.
    ///
    /// Constructs a quantum circuit approximating exp(-iHt) for the given Hamiltonian H and
    /// evolution time t. The approximation quality depends on the configuration: higher orders
    /// and more steps yield better accuracy at the cost of deeper circuits.
    ///
    /// **Example:**
    /// ```swift
    /// let ising = Observable(terms: [
    ///     (-1.0, PauliString(.z(0), .z(1))),
    ///     (-1.0, PauliString(.z(1), .z(2))),
    ///     (-0.5, PauliString(.x(0))),
    ///     (-0.5, PauliString(.x(1))),
    ///     (-0.5, PauliString(.x(2)))
    /// ])
    /// let config = TrotterConfiguration(order: .fourth, steps: 20)
    /// let circuit = TrotterSuzuki.evolve(ising, time: 2.0, qubits: 3, config: config)
    /// ```
    ///
    /// - Parameters:
    ///   - hamiltonian: Observable representing the Hamiltonian as sum of Pauli strings
    ///   - time: Total evolution time t in exp(-iHt)
    ///   - qubits: Number of qubits in the system
    ///   - config: Trotter decomposition configuration
    /// - Returns: Quantum circuit implementing the approximate time evolution
    /// - Complexity: O(steps * terms * order_factor) gates
    /// - Precondition: qubits > 0, steps > 0, hamiltonian non-empty
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func evolve(
        _ hamiltonian: Observable,
        time: Double,
        qubits: Int,
        config: TrotterConfiguration,
    ) -> QuantumCircuit {
        ValidationUtilities.validatePositiveQubits(qubits)
        ValidationUtilities.validateMemoryLimit(qubits)
        ValidationUtilities.validatePositiveInt(config.steps, name: "steps")
        ValidationUtilities.validateUpperBound(config.steps, max: maxSteps, name: "steps")
        ValidationUtilities.validateNonEmpty(hamiltonian.terms, name: "hamiltonian.terms")

        var circuit = QuantumCircuit(qubits: qubits)

        var terms = filterTerms(hamiltonian.terms, threshold: config.coefficientThreshold)

        if config.sortByCommutation {
            terms = sortTermsByCommutation(terms)
        }

        let stepSize = time / Double(config.steps)

        for _ in 0 ..< config.steps {
            switch config.order {
            case .first:
                firstOrderLayer(terms: terms, stepSize: stepSize, circuit: &circuit)
            case .second:
                secondOrderLayer(terms: terms, stepSize: stepSize, circuit: &circuit)
            case .fourth:
                fourthOrderLayer(terms: terms, stepSize: stepSize, circuit: &circuit)
            case .sixth:
                sixthOrderLayer(terms: terms, stepSize: stepSize, circuit: &circuit)
            }
        }

        return circuit
    }

    /// Apply first-order Trotter layer: prod_k exp(-i H_k * stepSize).
    ///
    /// Implements the simplest product formula by applying exponentials of each term
    /// sequentially in the given order. Error is O(stepSize^2) due to non-commutativity.
    ///
    /// - Parameters:
    ///   - terms: Hamiltonian terms as (coefficient, PauliString) pairs
    ///   - stepSize: Time step for this layer
    ///   - circuit: Circuit to append gates to
    /// - Complexity: O(terms) exponentials
    @_optimize(speed)
    static func firstOrderLayer(
        terms: [(Double, PauliString)],
        stepSize: Double,
        circuit: inout QuantumCircuit,
    ) {
        for (coefficient, pauliString) in terms {
            let angle = coefficient * stepSize
            applyPauliExponential(term: pauliString, angle: angle, circuit: &circuit)
        }
    }

    /// Apply second-order symmetric Trotter layer: S_2(t) = S_1(t/2)^dag S_1(t/2).
    ///
    /// Implements the Strang splitting by applying terms forward with half the time step,
    /// then backward with half the time step. The symmetry cancels first-order errors,
    /// yielding O(stepSize^3) error.
    ///
    /// - Parameters:
    ///   - terms: Hamiltonian terms as (coefficient, PauliString) pairs
    ///   - stepSize: Time step for this layer
    ///   - circuit: Circuit to append gates to
    /// - Complexity: O(2 * terms) exponentials
    @_optimize(speed)
    static func secondOrderLayer(
        terms: [(Double, PauliString)],
        stepSize: Double,
        circuit: inout QuantumCircuit,
    ) {
        let halfStep = stepSize / 2.0

        for (coefficient, pauliString) in terms {
            let angle = coefficient * halfStep
            applyPauliExponential(term: pauliString, angle: angle, circuit: &circuit)
        }

        for (coefficient, pauliString) in terms.reversed() {
            let angle = coefficient * halfStep
            applyPauliExponential(term: pauliString, angle: angle, circuit: &circuit)
        }
    }

    /// Apply fourth-order Yoshida layer using recursive composition.
    ///
    /// Implements Yoshida's formula: S_4(t) = S_2(s*t)^2 * S_2((1-4s)*t) * S_2(s*t)^2
    /// where s = 1/(4 - 4^(1/3)). This recursive composition cancels errors up to
    /// fourth order, yielding O(stepSize^5) error.
    ///
    /// - Parameters:
    ///   - terms: Hamiltonian terms as (coefficient, PauliString) pairs
    ///   - stepSize: Time step for this layer
    ///   - circuit: Circuit to append gates to
    /// - Complexity: O(10 * terms) exponentials
    @_optimize(speed)
    static func fourthOrderLayer(
        terms: [(Double, PauliString)],
        stepSize: Double,
        circuit: inout QuantumCircuit,
    ) {
        let s = yoshidaS
        let sStep = s * stepSize
        let centralStep = (1.0 - 4.0 * s) * stepSize

        secondOrderLayer(terms: terms, stepSize: sStep, circuit: &circuit)
        secondOrderLayer(terms: terms, stepSize: sStep, circuit: &circuit)
        secondOrderLayer(terms: terms, stepSize: centralStep, circuit: &circuit)
        secondOrderLayer(terms: terms, stepSize: sStep, circuit: &circuit)
        secondOrderLayer(terms: terms, stepSize: sStep, circuit: &circuit)
    }

    /// Apply sixth-order Suzuki layer using recursive composition.
    ///
    /// Implements Suzuki's sixth-order formula by recursively composing fourth-order
    /// integrators with coefficients s = 1/(4 - 4^(1/5)). The pattern mirrors the
    /// fourth-order construction: S_6(t) = S_4(s*t)^2 * S_4((1-4s)*t) * S_4(s*t)^2.
    /// This yields O(stepSize^7) error.
    ///
    /// - Parameters:
    ///   - terms: Hamiltonian terms as (coefficient, PauliString) pairs
    ///   - stepSize: Time step for this layer
    ///   - circuit: Circuit to append gates to
    /// - Complexity: O(50 * terms) exponentials
    @_optimize(speed)
    static func sixthOrderLayer(
        terms: [(Double, PauliString)],
        stepSize: Double,
        circuit: inout QuantumCircuit,
    ) {
        let s = suzukiS6
        let sStep = s * stepSize
        let centralStep = (1.0 - 4.0 * s) * stepSize

        fourthOrderLayer(terms: terms, stepSize: sStep, circuit: &circuit)
        fourthOrderLayer(terms: terms, stepSize: sStep, circuit: &circuit)
        fourthOrderLayer(terms: terms, stepSize: centralStep, circuit: &circuit)
        fourthOrderLayer(terms: terms, stepSize: sStep, circuit: &circuit)
        fourthOrderLayer(terms: terms, stepSize: sStep, circuit: &circuit)
    }

    /// Apply Pauli string exponential exp(-i * angle * P) to circuit.
    ///
    /// Implements the exponential of a Pauli string using the standard decomposition:
    /// 1. Rotate each qubit from its Pauli basis to Z basis (H for X, Rx(-pi/2) for Y)
    /// 2. Apply CNOT ladder to compute parity into the last qubit
    /// 3. Apply Rz(2*angle) to the last qubit
    /// 4. Reverse CNOT ladder
    /// 5. Reverse basis rotations
    ///
    /// - Parameters:
    ///   - term: Pauli string P to exponentiate
    ///   - angle: Rotation angle theta in exp(-i * theta * P)
    ///   - circuit: Circuit to append gates to
    /// - Complexity: O(qubits in term) gates
    @_optimize(speed)
    static func applyPauliExponential(
        term: PauliString,
        angle: Double,
        circuit: inout QuantumCircuit,
    ) {
        let operatorCount = term.operators.count

        guard operatorCount > 0 else { return }

        var targetQubits = [Int](unsafeUninitializedCapacity: operatorCount) { buffer, count in
            for i in 0 ..< operatorCount {
                let op = term.operators[i]
                buffer[i] = op.qubit
                applyBasisRotation(qubit: op.qubit, basis: op.basis, forward: true, circuit: &circuit)
            }
            count = operatorCount
        }
        targetQubits.sort()

        if targetQubits.count == 1 {
            circuit.append(.rotationZ(2.0 * angle), to: targetQubits[0])
        } else {
            for i in 0 ..< (targetQubits.count - 1) {
                circuit.append(.cnot, to: [targetQubits[i], targetQubits[i + 1]])
            }

            circuit.append(.rotationZ(2.0 * angle), to: targetQubits[targetQubits.count - 1])

            for i in stride(from: targetQubits.count - 2, through: 0, by: -1) {
                circuit.append(.cnot, to: [targetQubits[i], targetQubits[i + 1]])
            }
        }

        for op in term.operators {
            applyBasisRotation(qubit: op.qubit, basis: op.basis, forward: false, circuit: &circuit)
        }
    }

    /// Apply basis rotation for Pauli exponentiation.
    ///
    /// Transforms between Pauli eigenbasis and computational (Z) basis. For X basis,
    /// uses Hadamard. For Y basis, uses Rx(+-pi/2). Z basis requires no transformation.
    ///
    /// - Parameters:
    ///   - qubit: Target qubit index
    ///   - basis: Pauli basis to rotate from/to
    ///   - forward: If true, rotate from Pauli basis to Z; if false, rotate from Z to Pauli
    ///   - circuit: Circuit to append gates to
    @inline(__always)
    private static func applyBasisRotation(
        qubit: Int,
        basis: PauliBasis,
        forward: Bool,
        circuit: inout QuantumCircuit,
    ) {
        switch basis {
        case .z: break
        case .x: circuit.append(.hadamard, to: qubit)
        case .y: circuit.append(.rotationX(forward ? negHalfPi : halfPi), to: qubit)
        }
    }

    /// Filter terms by coefficient magnitude threshold.
    @_optimize(speed)
    @_effects(readonly)
    private static func filterTerms(
        _ terms: PauliTerms,
        threshold: Double,
    ) -> [(Double, PauliString)] {
        var filtered: [(Double, PauliString)] = []
        filtered.reserveCapacity(terms.count)

        for (coefficient, pauliString) in terms {
            guard !pauliString.operators.isEmpty, abs(coefficient) > threshold else {
                continue
            }
            filtered.append((coefficient, pauliString))
        }

        return filtered
    }

    /// Sort terms to group commuting operators together.
    ///
    /// Uses a greedy approach: starts with the first term, then repeatedly adds the next
    /// term that commutes with all previously added terms. Non-commuting terms are added
    /// at the end. This can reduce Trotter error by minimizing commutator contributions.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func sortTermsByCommutation(
        _ terms: [(Double, PauliString)],
    ) -> [(Double, PauliString)] {
        guard terms.count > 1 else { return terms }

        var sorted: [(Double, PauliString)] = []
        sorted.reserveCapacity(terms.count)

        let remaining = terms
        var used = [Bool](repeating: false, count: terms.count)

        sorted.append(remaining[0])
        used[0] = true

        for _ in 1 ..< terms.count {
            var bestIndex = -1

            for i in 0 ..< remaining.count {
                guard !used[i] else { continue }

                var commutesWithAll = true
                for (_, sortedTerm) in sorted {
                    if !PauliCommutation.commute(remaining[i].1, sortedTerm) {
                        commutesWithAll = false
                        break
                    }
                }

                if commutesWithAll {
                    bestIndex = i
                    break
                }
            }

            if bestIndex == -1 {
                for i in 0 ..< remaining.count where !used[i] {
                    bestIndex = i
                    break
                }
            }

            sorted.append(remaining[bestIndex])
            used[bestIndex] = true
        }

        return sorted
    }
}
