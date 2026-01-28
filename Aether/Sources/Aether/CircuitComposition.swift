// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Tensor product operator for composing quantum circuits in parallel across disjoint qubit registers.
///
/// - SeeAlso: ``QuantumCircuit``
infix operator ⊗: MultiplicationPrecedence

// MARK: - Circuit Composition Extension

public extension QuantumCircuit {
    /// Repeats the circuit a given number of times by concatenating its operations sequentially.
    ///
    /// Constructs a new circuit whose operation list is the original operations appended
    /// ``count`` times. Qubit count and labels are preserved from the source circuit.
    /// Useful for Trotterization steps and iterative amplitude amplification where the
    /// same unitary block is applied multiple times in succession.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// let tripled = circuit.repeated(3)
    /// ```
    ///
    /// - Parameter count: Number of repetitions (must be at least 1)
    /// - Returns: New circuit with operations repeated ``count`` times
    /// - Precondition: count >= 1
    /// - Complexity: O(n * count) where n is the number of operations in the original circuit
    ///
    /// - SeeAlso: ``power(_:)``
    @_optimize(speed)
    @_eagerMove
    func repeated(_ count: Int) -> QuantumCircuit {
        ValidationUtilities.validatePositiveInt(count, name: "count")

        let totalOps = operations.count * count
        var newOps = [CircuitOperation]()
        newOps.reserveCapacity(totalOps)

        for _ in 0 ..< count {
            newOps.append(contentsOf: operations)
        }

        var result = QuantumCircuit(qubits: qubits, operations: newOps)
        result.qubitLabels = qubitLabels
        return result
    }

    /// Raises the circuit unitary to a non-negative integer power.
    ///
    /// For small circuits (10 qubits or fewer) the full unitary matrix U is computed via
    /// ``CircuitUnitary`` and raised to the given exponent using O(log n) binary exponentiation
    /// through ``MatrixUtilities/matrixPower(_:exponent:)``, yielding a single custom unitary gate.
    /// For larger circuits or non-unitary circuits the method falls back to ``repeated(_:)``.
    /// Exponent zero returns an identity circuit (no operations), and exponent one returns the
    /// original circuit unchanged.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// let squared = circuit.power(2)
    /// ```
    ///
    /// - Parameter exponent: Non-negative integer power to raise the circuit to
    /// - Returns: New circuit implementing U^exponent
    /// - Precondition: exponent >= 0
    /// - Complexity: O(2^(3q) log(exponent)) for small circuits via matrix exponentiation, O(n * exponent) fallback
    ///
    /// - SeeAlso: ``repeated(_:)``
    /// - SeeAlso: ``MatrixUtilities/matrixPower(_:exponent:)``
    @_optimize(speed)
    @_eagerMove
    func power(_ exponent: Int) -> QuantumCircuit {
        ValidationUtilities.validateNonNegativeInt(exponent, name: "exponent")

        if exponent == 0 {
            var result = QuantumCircuit(qubits: qubits)
            result.qubitLabels = qubitLabels
            return result
        }

        if exponent == 1 {
            return self
        }

        let isUnitary = operations.allSatisfy(\.isUnitary)
        let useMatrixPath = isUnitary && qubits <= 10 && parameterCount == 0

        if useMatrixPath {
            let unitaryMatrix = CircuitUnitary.unitary(for: self)
            let powered = MatrixUtilities.matrixPower(unitaryMatrix, exponent: exponent)
            let allQubits = Array(0 ..< qubits)
            var result = QuantumCircuit(qubits: qubits)
            result.qubitLabels = qubitLabels

            if qubits == 1 {
                result.append(.customSingleQubit(matrix: powered), to: allQubits)
            } else if qubits == 2 {
                result.append(.customTwoQubit(matrix: powered), to: allQubits)
            } else {
                result.append(.customUnitary(matrix: powered), to: allQubits)
            }

            return result
        }

        return repeated(exponent)
    }

    /// Wraps every gate in the circuit with additional control qubits using decomposition.
    ///
    /// Produces a new circuit where each gate operation is conditioned on the specified control
    /// qubits all being in state |1>. The original circuit's qubit indices are shifted upward by
    /// the number of control qubits so that control qubits occupy the lowest indices and the
    /// original target qubits follow. Decomposition into native gates is handled by
    /// ``ControlledGateDecomposer``.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 1)
    /// circuit.append(.pauliX, to: 0)
    /// let controlled = circuit.controlled(by: [0])
    /// print(controlled.qubits)  // 2
    /// ```
    ///
    /// - Parameter controlQubits: Indices assigned to control qubits in the output circuit (must be non-empty)
    /// - Returns: New circuit with each gate conditioned on the control qubits
    /// - Precondition: controlQubits is non-empty
    /// - Precondition: Circuit contains only unitary operations (no reset or measurement)
    /// - Complexity: O(n * c) where n is operation count and c is number of control qubits
    ///
    /// - SeeAlso: ``ControlledGateDecomposer/decompose(gate:controls:target:)``
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    func controlled(by controlQubits: [Int]) -> QuantumCircuit {
        ValidationUtilities.validateNonEmpty(controlQubits, name: "controlQubits")
        ValidationUtilities.validateUnitaryCircuit(self)

        let numControls = controlQubits.count
        let newQubitCount = qubits + numControls
        var result = QuantumCircuit(qubits: newQubitCount)

        let controlIndices = Array(0 ..< numControls)

        for operation in operations {
            guard case let .gate(gate, targetQubits, _) = operation else { continue }

            let shiftedTargets = targetQubits.map { $0 + numControls }

            if gate.qubitsRequired == 1 {
                let decomposed = ControlledGateDecomposer.decompose(
                    gate: gate,
                    controls: controlIndices,
                    target: shiftedTargets[0],
                )
                for (decomposedGate, decomposedQubits) in decomposed {
                    result.append(decomposedGate, to: decomposedQubits)
                }
            } else {
                let gateMatrix = gate.matrix()
                let gateSize = gateMatrix.count
                let controlledDimension = 2 * gateSize

                var controlledMatrix = [[Complex<Double>]]()
                controlledMatrix.reserveCapacity(controlledDimension)

                for i in 0 ..< gateSize {
                    let row = [Complex<Double>](unsafeUninitializedCapacity: controlledDimension) { buffer, count in
                        for j in 0 ..< controlledDimension {
                            buffer[j] = .zero
                        }
                        buffer[i] = .one
                        count = controlledDimension
                    }
                    controlledMatrix.append(row)
                }

                for i in 0 ..< gateSize {
                    var row = [Complex<Double>](unsafeUninitializedCapacity: controlledDimension) { buffer, count in
                        for j in 0 ..< controlledDimension {
                            buffer[j] = .zero
                        }
                        count = controlledDimension
                    }
                    for j in 0 ..< gateSize {
                        row[gateSize + j] = gateMatrix[i][j]
                    }
                    controlledMatrix.append(row)
                }

                if numControls == 1 {
                    var allQubits = [controlIndices[0]]
                    allQubits.append(contentsOf: shiftedTargets)
                    result.append(.customUnitary(matrix: controlledMatrix), to: allQubits)
                } else {
                    var innerCircuit = QuantumCircuit(qubits: newQubitCount)
                    var innerQubits = [controlIndices[numControls - 1]]
                    innerQubits.append(contentsOf: shiftedTargets)
                    innerCircuit.append(.customUnitary(matrix: controlledMatrix), to: innerQubits)

                    let remainingControls = Array(controlIndices[0 ..< numControls - 1])
                    let innerControlled = innerCircuit.controlled(by: remainingControls)

                    for innerOp in innerControlled.operations {
                        result.addOperation(innerOp)
                    }
                }
            }
        }

        for (index, label) in qubitLabels {
            result.qubitLabels[index + numControls] = label
        }

        return result
    }
}

// MARK: - Series Composition Operator

/// Composes two quantum circuits in series by concatenating their operations sequentially.
///
/// The resulting circuit has qubit count equal to the maximum of the two input circuits.
/// All operations from the left-hand circuit execute first, followed by all operations from
/// the right-hand circuit. Qubit labels are merged with right-hand labels taking precedence
/// on index collision.
///
/// **Example:**
/// ```swift
/// var a = QuantumCircuit(qubits: 2)
/// a.append(.hadamard, to: 0)
/// var b = QuantumCircuit(qubits: 2)
/// b.append(.cnot, to: [0, 1])
/// let combined = a + b
/// print(combined.count)  // 2
/// ```
///
/// - Parameters:
///   - lhs: First circuit to execute
///   - rhs: Second circuit to execute after the first
/// - Returns: Combined circuit with operations from both inputs
/// - Complexity: O(m + n) where m and n are operation counts of the two circuits
///
/// - SeeAlso: ``QuantumCircuit``
@_optimize(speed)
public func + (lhs: QuantumCircuit, rhs: QuantumCircuit) -> QuantumCircuit {
    let maxQubits = max(lhs.qubits, rhs.qubits)
    let totalOps = lhs.operations.count + rhs.operations.count

    var newOps = [CircuitOperation]()
    newOps.reserveCapacity(totalOps)
    newOps.append(contentsOf: lhs.operations)
    newOps.append(contentsOf: rhs.operations)

    var result = QuantumCircuit(qubits: maxQubits, operations: newOps)

    for (index, label) in lhs.qubitLabels {
        result.qubitLabels[index] = label
    }
    for (index, label) in rhs.qubitLabels {
        result.qubitLabels[index] = label
    }

    return result
}

// MARK: - Tensor Product Operator

/// Composes two quantum circuits in parallel via tensor product across disjoint qubit registers.
///
/// The resulting circuit has qubit count equal to the sum of the two input circuits. Operations
/// from the left-hand circuit retain their original qubit indices. Operations from the right-hand
/// circuit have all qubit indices shifted upward by the left-hand circuit's qubit count so the
/// two circuits occupy non-overlapping registers. Qubit labels from the right-hand circuit are
/// similarly shifted.
///
/// **Example:**
/// ```swift
/// var a = QuantumCircuit(qubits: 1)
/// a.append(.hadamard, to: 0)
/// var b = QuantumCircuit(qubits: 1)
/// b.append(.pauliX, to: 0)
/// let tensor = a ⊗ b
/// print(tensor.qubits)  // 2
/// print(tensor.count)   // 2
/// ```
///
/// - Parameters:
///   - lhs: Circuit occupying qubits 0 ..< lhs.qubits
///   - rhs: Circuit occupying qubits lhs.qubits ..< lhs.qubits + rhs.qubits
/// - Returns: Combined circuit with disjoint qubit registers
/// - Complexity: O(m + n) where m and n are operation counts of the two circuits
///
/// - SeeAlso: ``QuantumCircuit``
@_optimize(speed)
public func ⊗ (lhs: QuantumCircuit, rhs: QuantumCircuit) -> QuantumCircuit {
    let totalQubits = lhs.qubits + rhs.qubits
    let shift = lhs.qubits
    let totalOps = lhs.operations.count + rhs.operations.count

    var newOps = [CircuitOperation]()
    newOps.reserveCapacity(totalOps)
    newOps.append(contentsOf: lhs.operations)

    for operation in rhs.operations {
        newOps.append(shiftOperation(operation, by: shift))
    }

    var result = QuantumCircuit(qubits: totalQubits, operations: newOps)

    for (index, label) in lhs.qubitLabels {
        result.qubitLabels[index] = label
    }
    for (index, label) in rhs.qubitLabels {
        result.qubitLabels[index + shift] = label
    }

    return result
}

/// Shifts all qubit indices in a circuit operation by the given offset.
///
/// - Parameters:
///   - operation: Circuit operation whose qubit indices to shift
///   - offset: Non-negative offset to add to each qubit index
/// - Returns: New operation with shifted qubit indices
/// - Complexity: O(k) where k is the number of qubits in the operation
@inline(__always)
private func shiftOperation(_ operation: CircuitOperation, by offset: Int) -> CircuitOperation {
    switch operation {
    case let .gate(gate, qubits, timestamp):
        let shifted = qubits.map { $0 + offset }
        return .gate(gate, qubits: shifted, timestamp: timestamp)
    case let .reset(qubit, timestamp):
        return .reset(qubit: qubit + offset, timestamp: timestamp)
    case let .measure(qubit, classicalBit, timestamp):
        return .measure(qubit: qubit + offset, classicalBit: classicalBit.map { $0 + offset }, timestamp: timestamp)
    }
}
