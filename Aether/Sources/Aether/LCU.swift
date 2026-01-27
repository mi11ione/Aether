// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Linear Combination of Unitaries (LCU) decomposition result.
///
/// Encapsulates the decomposition of a Hamiltonian H = Σᵢ αᵢUᵢ where Uᵢ are unitary operators
/// (Pauli strings). This representation is foundational for block-encoding methods and Qubitization,
/// enabling efficient Hamiltonian simulation with optimal query complexity. The 1-norm α = Σᵢ|αᵢ|
/// determines the success probability of LCU-based algorithms.
///
/// - SeeAlso: ``LCU``
/// - SeeAlso: ``Observable``
///
/// **Example:**
/// ```swift
/// let hamiltonian = Observable(terms: [
///     (0.5, PauliString(.z(0))),
///     (-0.3, PauliString(.x(1))),
///     (0.2, PauliString(.z(0), .z(1)))
/// ])
/// let decomposition = LCU.decompose(hamiltonian)
/// print(decomposition.oneNorm)  // 1.0
/// print(decomposition.termCount)  // 3
/// ```
@frozen
public struct LCUDecomposition: Sendable {
    /// Normalized coefficients αᵢ/α where α = Σⱼ|αⱼ| (sum to 1).
    ///
    /// Each coefficient represents the probability weight for selecting the corresponding
    /// unitary in the PREPARE oracle superposition.
    public let normalizedCoefficients: [Double]

    /// Original (unnormalized) coefficients from the Hamiltonian decomposition.
    ///
    /// Preserves sign information needed for proper phase handling in the SELECT oracle.
    public let originalCoefficients: [Double]

    /// Unitary operators Uᵢ (Pauli strings) comprising the LCU decomposition.
    ///
    /// Each Pauli string is a tensor product of single-qubit Pauli operators.
    public let unitaries: [PauliString]

    /// 1-norm α = Σᵢ|αᵢ| of the original coefficients.
    ///
    /// The 1-norm determines the normalization factor for the block-encoded Hamiltonian
    /// and affects the success probability of LCU-based algorithms.
    public let oneNorm: Double

    /// Number of ancilla qubits needed for PREPARE oracle (⌈log₂(L)⌉).
    ///
    /// Determines the register size for encoding term indices in binary.
    public let ancillaQubits: Int

    /// Number of terms L in the LCU decomposition.
    public let termCount: Int
}

/// Linear Combination of Unitaries for Hamiltonian simulation.
///
/// Provides algorithms for decomposing Hamiltonians into LCU form H = Σᵢ αᵢPᵢ where Pᵢ are
/// Pauli strings (unitary operators). The LCU decomposition enables block-encoding methods
/// where the Hamiltonian appears as a subblock of a larger unitary operator, foundational
/// for Qubitization and optimal quantum simulation algorithms.
///
/// The LCU circuit consists of three components:
/// - **PREPARE**: Creates superposition |ψ_prep⟩ = Σᵢ √(|αᵢ|/α)|i⟩ encoding coefficient magnitudes
/// - **SELECT**: Applies |i⟩|ψ⟩ -> |i⟩Pᵢ|ψ⟩ controlled on ancilla state
/// - **Block Encoding**: U_LCU = PREPARE† · SELECT · PREPARE contains H/α in top-left block
///
/// - SeeAlso: ``LCUDecomposition``
/// - SeeAlso: ``Observable``
///
/// **Example:**
/// ```swift
/// let hamiltonian = Observable(terms: [
///     (0.5, PauliString(.z(0))),
///     (-0.3, PauliString(.x(1)))
/// ])
/// let decomposition = LCU.decompose(hamiltonian)
/// let prepCircuit = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 2)
/// let selectCircuit = LCU.selectCircuit(decomposition: decomposition, systemQubits: 2, ancillaStart: 2)
/// ```
public enum LCU {
    /// Decompose Hamiltonian into LCU form H = Σᵢ αᵢPᵢ.
    ///
    /// Extracts Pauli string terms and coefficients from the Hamiltonian observable,
    /// computing the 1-norm and normalized coefficients for PREPARE oracle construction.
    /// Zero-coefficient terms are filtered out to optimize circuit size.
    ///
    /// - Parameter hamiltonian: Observable representing the Hamiltonian as weighted Pauli strings
    /// - Returns: LCU decomposition containing coefficients, unitaries, and metadata
    /// - Complexity: O(L) where L is the number of terms
    ///
    /// **Example:**
    /// ```swift
    /// let H = Observable(terms: [
    ///     (0.5, PauliString(.z(0))),
    ///     (-0.3, PauliString(.x(1))),
    ///     (0.2, PauliString(.z(0), .z(1)))
    /// ])
    /// let decomposition = LCU.decompose(H)
    /// print(decomposition.oneNorm)  // 1.0
    /// print(decomposition.ancillaQubits)  // 2 (⌈log₂(3)⌉)
    /// ```
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func decompose(_ hamiltonian: Observable) -> LCUDecomposition {
        var coefficients: [Double] = []
        var unitaries: [PauliString] = []

        coefficients.reserveCapacity(hamiltonian.terms.count)
        unitaries.reserveCapacity(hamiltonian.terms.count)

        var oneNorm = 0.0

        for term in hamiltonian.terms {
            let coeff = term.coefficient
            if abs(coeff) > 1e-15 {
                coefficients.append(coeff)
                unitaries.append(term.pauliString)
                oneNorm += abs(coeff)
            }
        }

        let termCount = coefficients.count
        let ancillaQubits = termCount > 1 ? ceilLog2(termCount) : 1

        var normalizedCoefficients: [Double] = []
        normalizedCoefficients.reserveCapacity(termCount)

        if oneNorm > 1e-15 {
            for coeff in coefficients {
                normalizedCoefficients.append(abs(coeff) / oneNorm)
            }
        }

        return LCUDecomposition(
            normalizedCoefficients: normalizedCoefficients,
            originalCoefficients: coefficients,
            unitaries: unitaries,
            oneNorm: oneNorm,
            ancillaQubits: ancillaQubits,
            termCount: termCount,
        )
    }

    /// Build PREPARE oracle circuit: |0⟩ -> Σᵢ √(|αᵢ|/α)|i⟩.
    ///
    /// Creates a superposition over ancilla qubits where each computational basis state |i⟩
    /// has amplitude √(|αᵢ|/α). Uses a binary tree of controlled-Ry rotations for amplitude
    /// encoding. The rotation angles are computed as θ = 2·arcsin(√(probability)).
    ///
    /// - Parameters:
    ///   - decomposition: LCU decomposition containing normalized coefficients
    ///   - ancillaStart: Starting qubit index for ancilla register
    /// - Returns: Quantum circuit implementing the PREPARE oracle
    /// - Complexity: O(L) gates where L is the number of terms
    ///
    /// **Example:**
    /// ```swift
    /// let decomposition = LCU.decompose(hamiltonian)
    /// let prepare = LCU.prepareCircuit(decomposition: decomposition, ancillaStart: 4)
    /// let state = prepare.execute()
    /// ```
    @_optimize(speed)
    @_eagerMove
    public static func prepareCircuit(
        decomposition: LCUDecomposition,
        ancillaStart: Int,
    ) -> QuantumCircuit {
        let numAncilla = decomposition.ancillaQubits
        let totalQubits = ancillaStart + numAncilla

        var circuit = QuantumCircuit(qubits: totalQubits)

        if decomposition.termCount == 0 {
            return circuit
        }

        if decomposition.termCount == 1 {
            return circuit
        }

        let numStates = 1 << numAncilla
        var probabilities = [Double](repeating: 0.0, count: numStates)

        for i in 0 ..< decomposition.termCount {
            probabilities[i] = decomposition.normalizedCoefficients[i]
        }

        buildAmplitudeEncodingTree(
            circuit: &circuit,
            probabilities: probabilities,
            ancillaStart: ancillaStart,
            numAncilla: numAncilla,
            startIndex: 0,
            endIndex: numStates,
            depth: 0,
            controls: [],
        )

        return circuit
    }

    /// Build SELECT oracle circuit: |i⟩|ψ⟩ -> |i⟩Uᵢ|ψ⟩.
    ///
    /// Applies the i-th Pauli string Uᵢ to the system register when the ancilla register
    /// is in state |i⟩. Uses multi-controlled gates with binary encoding of the index i,
    /// where control pattern matches the binary representation of each term index.
    ///
    /// - Parameters:
    ///   - decomposition: LCU decomposition containing Pauli string unitaries
    ///   - systemQubits: Number of qubits in the system register
    ///   - ancillaStart: Starting qubit index for ancilla register
    /// - Returns: Quantum circuit implementing the SELECT oracle
    /// - Complexity: O(L·n) gates where L is term count and n is system qubits
    ///
    /// **Example:**
    /// ```swift
    /// let decomposition = LCU.decompose(hamiltonian)
    /// let select = LCU.selectCircuit(
    ///     decomposition: decomposition,
    ///     systemQubits: 2,
    ///     ancillaStart: 2
    /// )
    /// ```
    @_optimize(speed)
    @_eagerMove
    public static func selectCircuit(
        decomposition: LCUDecomposition,
        systemQubits: Int,
        ancillaStart: Int,
    ) -> QuantumCircuit {
        let numAncilla = decomposition.ancillaQubits
        let totalQubits = ancillaStart + numAncilla

        var circuit = QuantumCircuit(qubits: totalQubits)

        if decomposition.termCount == 0 {
            return circuit
        }

        for termIndex in 0 ..< decomposition.termCount {
            let pauliString = decomposition.unitaries[termIndex]
            let coefficient = decomposition.originalCoefficients[termIndex]

            appendControlledPauliString(
                circuit: &circuit,
                pauliString: pauliString,
                termIndex: termIndex,
                numAncilla: numAncilla,
                ancillaStart: ancillaStart,
                systemQubits: systemQubits,
                isNegative: coefficient < 0,
            )
        }

        return circuit
    }

    /// Build complete LCU circuit: PREPARE† · SELECT · PREPARE.
    ///
    /// Constructs the full block-encoding circuit where the (0,0) block of the resulting
    /// unitary contains H/α (the Hamiltonian divided by its 1-norm). The circuit applies
    /// PREPARE to create coefficient superposition, SELECT to apply controlled unitaries,
    /// and PREPARE† to decode back to the ancilla ground state.
    ///
    /// - Parameters:
    ///   - decomposition: LCU decomposition of the Hamiltonian
    ///   - systemQubits: Number of qubits in the system register
    ///   - ancillaStart: Starting qubit index for ancilla register
    /// - Returns: Quantum circuit implementing the block-encoded Hamiltonian
    /// - Complexity: O(L·n) gates where L is term count and n is system qubits
    ///
    /// **Example:**
    /// ```swift
    /// let decomposition = LCU.decompose(hamiltonian)
    /// let blockCircuit = LCU.blockEncodingCircuit(
    ///     decomposition: decomposition,
    ///     systemQubits: 2,
    ///     ancillaStart: 2
    /// )
    /// let state = blockCircuit.execute()
    /// ```
    @_optimize(speed)
    @_eagerMove
    public static func blockEncodingCircuit(
        decomposition: LCUDecomposition,
        systemQubits: Int,
        ancillaStart: Int,
    ) -> QuantumCircuit {
        let numAncilla = decomposition.ancillaQubits
        let totalQubits = ancillaStart + numAncilla

        var circuit = QuantumCircuit(qubits: totalQubits)

        if decomposition.termCount == 0 {
            return circuit
        }

        let prepareCircuit = prepareCircuit(decomposition: decomposition, ancillaStart: ancillaStart)
        for op in prepareCircuit.operations {
            circuit.addOperation(op)
        }

        let selectCircuit = selectCircuit(
            decomposition: decomposition,
            systemQubits: systemQubits,
            ancillaStart: ancillaStart,
        )
        for op in selectCircuit.operations {
            circuit.addOperation(op)
        }

        let prepareInverse = prepareCircuit.inverse()
        for op in prepareInverse.operations {
            circuit.addOperation(op)
        }

        return circuit
    }

    /// Estimate success probability for LCU application.
    ///
    /// Computes the probability of successfully applying H/α to a quantum state,
    /// which is approximately (⟨ψ|H|ψ⟩/α)² for a normalized state |ψ⟩. This determines
    /// how many repetitions are needed on average for amplitude amplification.
    ///
    /// - Parameters:
    ///   - decomposition: LCU decomposition containing the 1-norm
    ///   - expectedEnergy: Expected value ⟨ψ|H|ψ⟩ of the Hamiltonian
    /// - Returns: Estimated success probability in range [0, 1]
    /// - Complexity: O(1)
    ///
    /// **Example:**
    /// ```swift
    /// let decomposition = LCU.decompose(hamiltonian)
    /// let state = circuit.execute()
    /// let energy = hamiltonian.expectationValue(of: state)
    /// let pSuccess = LCU.estimateSuccessProbability(
    ///     decomposition: decomposition,
    ///     expectedEnergy: energy
    /// )
    /// ```
    @_effects(readonly)
    public static func estimateSuccessProbability(
        decomposition: LCUDecomposition,
        expectedEnergy: Double,
    ) -> Double {
        guard decomposition.oneNorm > 1e-15 else {
            return 0.0
        }

        let ratio = expectedEnergy / decomposition.oneNorm
        let probability = ratio * ratio

        return min(max(probability, 0.0), 1.0)
    }

    @_optimize(speed)
    @_effects(readonly)
    private static func ceilLog2(_ n: Int) -> Int {
        var value = n - 1
        var bits = 0
        while value > 0 {
            bits += 1
            value >>= 1
        }
        return bits
    }

    @_optimize(speed)
    private static func buildAmplitudeEncodingTree(
        circuit: inout QuantumCircuit,
        probabilities: [Double],
        ancillaStart: Int,
        numAncilla: Int,
        startIndex: Int,
        endIndex: Int,
        depth: Int,
        controls: [(qubit: Int, value: Int)],
    ) {
        guard depth < numAncilla else { return }

        let midIndex = (startIndex + endIndex) / 2

        var probLeft = 0.0
        for i in startIndex ..< midIndex {
            probLeft += probabilities[i]
        }

        var probRight = 0.0
        for i in midIndex ..< endIndex {
            probRight += probabilities[i]
        }

        let totalProb = probLeft + probRight

        if totalProb < 1e-15 {
            return
        }

        let probZero = probLeft / totalProb

        if probZero > 1e-15, probZero < 1.0 - 1e-15 {
            let theta = 2.0 * acos(sqrt(probZero))
            let targetQubit = ancillaStart + depth

            if controls.isEmpty {
                circuit.append(.rotationY(theta), to: targetQubit)
            } else {
                appendControlledRy(
                    circuit: &circuit,
                    theta: theta,
                    target: targetQubit,
                    controls: controls,
                )
            }
        }

        var leftControls = controls
        leftControls.append((qubit: ancillaStart + depth, value: 0))

        buildAmplitudeEncodingTree(
            circuit: &circuit,
            probabilities: probabilities,
            ancillaStart: ancillaStart,
            numAncilla: numAncilla,
            startIndex: startIndex,
            endIndex: midIndex,
            depth: depth + 1,
            controls: leftControls,
        )

        var rightControls = controls
        rightControls.append((qubit: ancillaStart + depth, value: 1))

        buildAmplitudeEncodingTree(
            circuit: &circuit,
            probabilities: probabilities,
            ancillaStart: ancillaStart,
            numAncilla: numAncilla,
            startIndex: midIndex,
            endIndex: endIndex,
            depth: depth + 1,
            controls: rightControls,
        )
    }

    @_optimize(speed)
    private static func appendControlledRy(
        circuit: inout QuantumCircuit,
        theta: Double,
        target: Int,
        controls: [(qubit: Int, value: Int)],
    ) {
        for control in controls where control.value == 0 {
            circuit.append(.pauliX, to: control.qubit)
        }

        if controls.count == 1 {
            circuit.append(.controlledRotationY(theta), to: [controls[0].qubit, target])
        } else {
            let halfTheta = theta / 2.0
            let controlQubits = controls.map(\.qubit)

            circuit.append(.rotationY(halfTheta), to: target)
            appendMultiControlledX(circuit: &circuit, controls: controlQubits, target: target)
            circuit.append(.rotationY(-halfTheta), to: target)
            appendMultiControlledX(circuit: &circuit, controls: controlQubits, target: target)
        }

        for control in controls where control.value == 0 {
            circuit.append(.pauliX, to: control.qubit)
        }
    }

    @_optimize(speed)
    private static func appendMultiControlledX(
        circuit: inout QuantumCircuit,
        controls: [Int],
        target: Int,
    ) {
        let n = controls.count

        if n == 1 {
            circuit.append(.cnot, to: [controls[0], target])
        } else if n == 2 {
            circuit.append(.toffoli, to: [controls[0], controls[1], target])
        } else {
            let decomposition = ControlledGateDecomposer.toffoliLadderDecomposition(
                gate: .pauliX,
                controls: controls,
                target: target,
            )
            for (gate, qubits) in decomposition {
                circuit.append(gate, to: qubits)
            }
        }
    }

    @_optimize(speed)
    private static func appendControlledPauliString(
        circuit: inout QuantumCircuit,
        pauliString: PauliString,
        termIndex: Int,
        numAncilla: Int,
        ancillaStart: Int,
        systemQubits _: Int,
        isNegative: Bool,
    ) {
        var controls: [(qubit: Int, value: Int)] = []
        controls.reserveCapacity(numAncilla)

        for bit in 0 ..< numAncilla {
            let qubit = ancillaStart + bit
            let value = (termIndex >> bit) & 1
            controls.append((qubit: qubit, value: value))
        }

        for control in controls where control.value == 0 {
            circuit.append(.pauliX, to: control.qubit)
        }

        if isNegative {
            let controlQubits = controls.map(\.qubit)
            appendControlledPhase(
                circuit: &circuit,
                controls: controlQubits,
                phase: .pi,
            )
        }

        for op in pauliString.operators {
            let targetQubit = op.qubit
            let controlQubits = controls.map(\.qubit)

            switch op.basis {
            case .x:
                appendMultiControlledX(circuit: &circuit, controls: controlQubits, target: targetQubit)
            case .y:
                appendMultiControlledY(circuit: &circuit, controls: controlQubits, target: targetQubit)
            case .z:
                appendMultiControlledZ(circuit: &circuit, controls: controlQubits, target: targetQubit)
            }
        }

        for control in controls where control.value == 0 {
            circuit.append(.pauliX, to: control.qubit)
        }
    }

    @_optimize(speed)
    private static func appendMultiControlledY(
        circuit: inout QuantumCircuit,
        controls: [Int],
        target: Int,
    ) {
        let n = controls.count

        if n == 1 {
            circuit.append(.cy, to: [controls[0], target])
        } else {
            circuit.append(.phase(.value(-Double.pi / 2.0)), to: target)
            appendMultiControlledX(circuit: &circuit, controls: controls, target: target)
            circuit.append(.sGate, to: target)
        }
    }

    @_optimize(speed)
    private static func appendMultiControlledZ(
        circuit: inout QuantumCircuit,
        controls: [Int],
        target: Int,
    ) {
        let n = controls.count

        if n == 1 {
            circuit.append(.cz, to: [controls[0], target])
        } else {
            circuit.append(.hadamard, to: target)
            appendMultiControlledX(circuit: &circuit, controls: controls, target: target)
            circuit.append(.hadamard, to: target)
        }
    }

    @_optimize(speed)
    private static func appendControlledPhase(
        circuit: inout QuantumCircuit,
        controls: [Int],
        phase: Double,
    ) {
        let n = controls.count

        if n == 1 {
            circuit.append(.phase(.value(phase)), to: controls[0])
        } else {
            let target = controls[n - 1]
            let remainingControls = Array(controls.dropLast())

            circuit.append(.hadamard, to: target)

            let decomposition = ControlledGateDecomposer.toffoliLadderDecomposition(
                gate: .pauliX,
                controls: remainingControls,
                target: target,
            )

            let halfDecomp = decomposition.count / 2 + 1
            for i in 0 ..< halfDecomp {
                let (gate, qubits) = decomposition[i]
                circuit.append(gate, to: qubits)
            }

            circuit.append(.phase(.value(phase / 2.0)), to: target)

            for i in (0 ..< halfDecomp).reversed() {
                let (gate, qubits) = decomposition[i]
                circuit.append(gate, to: qubits)
            }

            circuit.append(.phase(.value(-phase / 2.0)), to: target)
            circuit.append(.hadamard, to: target)

            appendControlledPhase(circuit: &circuit, controls: remainingControls, phase: phase / 2.0)
        }
    }
}
