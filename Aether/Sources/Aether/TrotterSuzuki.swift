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
/// The available orders range from ``first`` (simple product formula, O(t^2) error per step)
/// through ``second`` (symmetric Strang splitting, O(t^3) error) and ``fourth`` (Yoshida's
/// recursive formula, O(t^5) error) to ``sixth`` (Suzuki's recursive formula, O(t^7) error).
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
///     isSortingByCommutation: true,
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
    public let isSortingByCommutation: Bool

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
    ///   - isSortingByCommutation: Reorder terms by commutation (default: false)
    ///   - coefficientThreshold: Minimum coefficient magnitude (default: 1e-15)
    @inlinable
    public init(
        order: TrotterOrder = .second,
        steps: Int = 1,
        isSortingByCommutation: Bool = false,
        coefficientThreshold: Double = 1e-15,
    ) {
        self.order = order
        self.steps = steps
        self.isSortingByCommutation = isSortingByCommutation
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
/// The approximation error depends on the order and step size. First-order gives
/// O(t^2/steps) error, second-order symmetric splitting gives O(t^3/steps^2),
/// fourth-order Yoshida gives O(t^5/steps^4), and sixth-order Suzuki achieves
/// O(t^7/steps^6) at the cost of the deepest circuits.
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
    /// Half of pi for basis rotation angles.
    private static let halfPi = Double.pi / 2
    /// Negative half of pi for basis rotation angles.
    private static let negHalfPi = -Double.pi / 2
    /// Upper bound on Trotter step count.
    private static let maxSteps = 10000

    /// Yoshida fourth-order composition coefficient s = 1/(4 - 4^(1/3)).
    private static let yoshidaS = 1.0 / (4.0 - pow(4.0, 1.0 / 3.0))

    /// Suzuki sixth-order composition coefficient s = 1/(4 - 4^(1/5)).
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
    /// - Precondition: `qubits > 0`
    /// - Precondition: `qubits` within memory limit
    /// - Precondition: `config.steps > 0`
    /// - Precondition: `config.steps <= 10000`
    /// - Precondition: `hamiltonian.terms` is non-empty
    @_optimize(speed)
    @_eagerMove
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

        if config.isSortingByCommutation {
            terms = sortTermsByCommutation(terms)
        }

        let stepSize = time / Double(config.steps)

        switch config.order {
        case .first:
            for _ in 0 ..< config.steps {
                firstOrderLayer(terms: terms, stepSize: stepSize, circuit: &circuit)
            }
        case .second:
            for _ in 0 ..< config.steps {
                secondOrderLayer(terms: terms, stepSize: stepSize, circuit: &circuit)
            }
        case .fourth:
            for _ in 0 ..< config.steps {
                fourthOrderLayer(terms: terms, stepSize: stepSize, circuit: &circuit)
            }
        case .sixth:
            for _ in 0 ..< config.steps {
                sixthOrderLayer(terms: terms, stepSize: stepSize, circuit: &circuit)
            }
        }

        return circuit
    }

    /// Apply first-order Trotter layer: prod_k exp(-i H_k * stepSize).
    @_optimize(speed)
    private static func firstOrderLayer(
        terms: PauliTerms,
        stepSize: Double,
        circuit: inout QuantumCircuit,
    ) {
        for (coefficient, pauliString) in terms {
            let angle = coefficient * stepSize
            applyPauliExponential(term: pauliString, angle: angle, circuit: &circuit)
        }
    }

    /// Apply second-order symmetric Trotter layer.
    @_optimize(speed)
    private static func secondOrderLayer(
        terms: PauliTerms,
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
    @_optimize(speed)
    private static func fourthOrderLayer(
        terms: PauliTerms,
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
    @_optimize(speed)
    private static func sixthOrderLayer(
        terms: PauliTerms,
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
    @_optimize(speed)
    private static func applyPauliExponential(
        term: PauliString,
        angle: Double,
        circuit: inout QuantumCircuit,
    ) {
        let operatorCount = term.operators.count

        guard operatorCount > 0 else { return }

        for op in term.operators {
            applyBasisRotation(qubit: op.qubit, basis: op.basis, forward: true, circuit: &circuit)
        }

        var targetQubits = [Int](unsafeUninitializedCapacity: operatorCount) {
            buffer, count in
            for i in 0 ..< operatorCount {
                buffer[i] = term.operators[i].qubit
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
    ) -> PauliTerms {
        var filtered: PauliTerms = []
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
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func sortTermsByCommutation(
        _ terms: PauliTerms,
    ) -> PauliTerms {
        guard terms.count > 1 else { return terms }

        var sorted: PauliTerms = []
        sorted.reserveCapacity(terms.count)

        let remaining = terms
        var used = [Bool](repeating: false, count: terms.count)

        sorted.append(remaining[0])
        used[0] = true
        var firstUnused = 1

        for _ in 1 ..< terms.count {
            var bestIndex = -1

            for i in firstUnused ..< remaining.count {
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
                bestIndex = firstUnused
            }

            sorted.append(remaining[bestIndex])
            used[bestIndex] = true
            while firstUnused < remaining.count, used[firstUnused] {
                firstUnused += 1
            }
        }

        return sorted
    }
}
