// Copyright (c) 2025-2026 Roman Zhuzhgov, Apache License 2.0

/// Gate classification utilities for Clifford simulation.
///
/// Classifies gates as Clifford or non-Clifford for backend dispatch optimization.
/// Clifford gates can be efficiently simulated on classical computers using stabilizer
/// formalism, while non-Clifford gates (primarily T gates) require full state vector
/// simulation.
///
/// **Example:**
/// ```swift
/// let classification = CliffordGateClassifier.classify(.hadamard)
/// // .clifford
///
/// let tClassification = CliffordGateClassifier.classify(.tGate)
/// // .nonClifford(tCount: 1)
///
/// var circuit = QuantumCircuit(qubits: 2)
/// circuit.append(.hadamard, to: 0)
/// circuit.append(.cnot, to: [0, 1])
/// let analysis = CliffordGateClassifier.analyze(circuit)
/// // (isClifford: true, tCount: 0)
/// ```
///
/// - SeeAlso: ``QuantumGate``, ``QuantumCircuit``
public enum CliffordGateClassifier {
    /// Standard tolerance for floating-point Clifford classification.
    private static let tolerance: Double = 1e-10
    /// Scaled tolerance for pi/8 multiple detection.
    private static let piEighthTolerance: Double = 1e-8

    /// Classification result for a ``QuantumGate``.
    ///
    /// Clifford gates form a finite group and can be efficiently simulated.
    /// Non-Clifford gates include T-count estimation for resource analysis.
    ///
    /// **Example:**
    /// ```swift
    /// let result = CliffordGateClassifier.classify(.tGate)
    /// switch result {
    /// case .clifford:
    ///     print("Efficient simulation possible")
    /// case .nonClifford(let tCount):
    ///     print("Requires \(tCount) T gates")
    /// }
    /// ```
    ///
    /// - SeeAlso: ``classify(_:)``
    @frozen
    public enum Classification: Equatable, Sendable {
        case clifford
        case nonClifford(tCount: Int)
    }

    /// Classifies a ``QuantumGate`` as Clifford or non-Clifford.
    ///
    /// Clifford gates include Pauli gates (X, Y, Z), Hadamard, S gate, phase gates
    /// at multiples of pi/2, CNOT, CZ, CY, SWAP, and iSWAP. Non-Clifford gates
    /// include T gate, arbitrary rotations, and Toffoli.
    ///
    /// **Example:**
    /// ```swift
    /// CliffordGateClassifier.classify(.hadamard)  // .clifford
    /// CliffordGateClassifier.classify(.tGate)     // .nonClifford(tCount: 1)
    /// CliffordGateClassifier.classify(.toffoli)   // .nonClifford(tCount: 7)
    /// ```
    ///
    /// - Parameter gate: The ``QuantumGate`` to classify
    /// - Returns: ``Classification`` indicating Clifford status and T-count if non-Clifford
    /// - Complexity: O(1) for most gates, O(n^2) for custom unitary matrices
    /// - SeeAlso: ``Classification``, ``isClifford(_:)``
    @inlinable
    @_effects(readonly)
    public static func classify(_ gate: QuantumGate) -> Classification {
        switch gate {
        case .identity, .pauliX, .pauliY, .pauliZ, .hadamard, .sGate, .sx, .sy:
            return .clifford

        case .cnot, .cz, .cy, .ch, .swap, .iswap:
            return .clifford

        case .tGate:
            return .nonClifford(tCount: 1)

        case .toffoli:
            return .nonClifford(tCount: 7)

        case .fredkin:
            return .nonClifford(tCount: 7)

        case .ccz:
            return .nonClifford(tCount: 7)

        case let .phase(angle):
            return classifyAngle(angle)

        case let .rotationX(theta):
            return classifyAngle(theta)

        case let .rotationY(theta):
            return classifyAngle(theta)

        case let .rotationZ(theta):
            return classifyAngle(theta)

        case let .globalPhase(phi):
            return classifyAngle(phi)

        case let .u1(lambda):
            return classifyAngle(lambda)

        case let .u2(phi, lambda):
            let phiClass = classifyAngle(phi)
            let lambdaClass = classifyAngle(lambda)
            return combineClassifications(phiClass, lambdaClass)

        case let .u3(theta, phi, lambda):
            let thetaClass = classifyAngle(theta)
            let phiClass = classifyAngle(phi)
            let lambdaClass = classifyAngle(lambda)
            return combineClassifications(combineClassifications(thetaClass, phiClass), lambdaClass)

        case let .controlledPhase(theta):
            return classifyAngle(theta)

        case let .controlledRotationX(theta):
            return classifyAngle(theta)

        case let .controlledRotationY(theta):
            return classifyAngle(theta)

        case let .controlledRotationZ(theta):
            return classifyAngle(theta)

        case .sqrtSwap, .sqrtISwap, .fswap:
            return .nonClifford(tCount: 1)

        case let .givens(theta):
            return classifyAngle(theta)

        case let .xx(theta):
            return classifyAngle(theta)

        case let .yy(theta):
            return classifyAngle(theta)

        case let .zz(theta):
            return classifyAngle(theta)

        case let .diagonal(phases):
            return classifyDiagonalPhases(phases)

        case let .multiplexor(unitaries):
            return classifyMultiplexor(unitaries)

        case let .controlled(innerGate, _):
            let innerClass = classify(innerGate)
            switch innerClass {
            case .clifford:
                return classifyControlledClifford(innerGate)
            case .nonClifford:
                return innerClass
            }

        case let .customSingleQubit(matrix):
            return classifyCustomMatrix(matrix)

        case let .customTwoQubit(matrix):
            return classifyCustomMatrix(matrix)

        case let .customUnitary(matrix):
            return classifyCustomMatrix(matrix)
        }
    }

    /// Checks if a ``QuantumGate`` is a Clifford gate.
    ///
    /// Convenience method that returns true if the gate classification is `.clifford`.
    ///
    /// **Example:**
    /// ```swift
    /// CliffordGateClassifier.isClifford(.hadamard)  // true
    /// CliffordGateClassifier.isClifford(.tGate)     // false
    /// ```
    ///
    /// - Parameter gate: The ``QuantumGate`` to check
    /// - Returns: True if the gate is a Clifford gate
    /// - Complexity: O(1) for most gates
    /// - SeeAlso: ``classify(_:)``
    @inlinable
    @_effects(readonly)
    public static func isClifford(_ gate: QuantumGate) -> Bool {
        switch classify(gate) {
        case .clifford:
            true
        case .nonClifford:
            false
        }
    }

    /// Analyzes a ``QuantumCircuit`` for Clifford structure and T-count.
    ///
    /// Examines all gates in the circuit to determine if the entire circuit
    /// consists only of Clifford gates, and counts the total T gates (or
    /// T-equivalent gates) for resource estimation.
    ///
    /// **Example:**
    /// ```swift
    /// var circuit = QuantumCircuit(qubits: 2)
    /// circuit.append(.hadamard, to: 0)
    /// circuit.append(.cnot, to: [0, 1])
    /// let result = CliffordGateClassifier.analyze(circuit)
    /// // (isClifford: true, tCount: 0)
    ///
    /// circuit.append(.tGate, to: 0)
    /// let result2 = CliffordGateClassifier.analyze(circuit)
    /// // (isClifford: false, tCount: 1)
    /// ```
    ///
    /// - Parameter circuit: The ``QuantumCircuit`` to analyze
    /// - Returns: Tuple containing Clifford status and total T-count
    /// - Complexity: O(n) where n is the number of operations in the circuit
    /// - SeeAlso: ``classify(_:)``, ``Classification``
    @inlinable
    @_effects(readonly)
    public static func analyze(_ circuit: QuantumCircuit) -> (isClifford: Bool, tCount: Int) {
        var totalTCount = 0
        var isAllClifford = true

        for operation in circuit.operations {
            switch operation {
            case let .gate(gate, _, _):
                let classification = classify(gate)
                switch classification {
                case .clifford:
                    break
                case let .nonClifford(tCount):
                    isAllClifford = false
                    totalTCount += tCount
                }
            case .reset, .measure:
                break
            }
        }

        return (isClifford: isAllClifford, tCount: totalTCount)
    }

    /// Classify a parameter value angle as Clifford or non-Clifford.
    @usableFromInline
    @_effects(readonly)
    static func classifyAngle(_ angle: ParameterValue) -> Classification {
        guard case let .value(theta) = angle else {
            return .nonClifford(tCount: 1)
        }
        return classifyAngleValue(theta)
    }

    /// Classify a concrete angle value by checking Clifford multiples and estimating T-count.
    @usableFromInline
    @_effects(readonly)
    static func classifyAngleValue(_ theta: Double) -> Classification {
        if isCliffordPhaseAngle(theta) { return .clifford }
        return estimateTCountForAngle(theta)
    }

    /// Check if an angle is a Clifford phase (multiple of pi/2).
    @inline(__always)
    @usableFromInline
    @_effects(readonly)
    static func isCliffordPhaseAngle(_ theta: Double) -> Bool {
        let normalized = normalizeAngle(theta)
        return abs(normalized) < tolerance
            || abs(normalized - (.pi * 0.5)) < tolerance
            || abs(normalized - .pi) < tolerance
            || abs(normalized - (.pi * 1.5)) < tolerance
    }

    /// Normalize an angle to the range [0, 2pi).
    @inline(__always)
    @usableFromInline
    @_effects(readonly)
    static func normalizeAngle(_ theta: Double) -> Double {
        let twoPi = 2.0 * Double.pi
        var normalized = theta.truncatingRemainder(dividingBy: twoPi)
        if normalized < 0 { normalized += twoPi }
        return normalized
    }

    /// Estimate T-gate count needed to synthesize a non-Clifford angle.
    @usableFromInline
    @_effects(readonly)
    static func estimateTCountForAngle(_ theta: Double) -> Classification {
        let normalized = normalizeAngle(theta)
        let quarterPi = Double.pi / 4.0
        if abs(normalized - quarterPi) < tolerance ||
            abs(normalized - 3.0 * quarterPi) < tolerance ||
            abs(normalized - 5.0 * quarterPi) < tolerance ||
            abs(normalized - 7.0 * quarterPi) < tolerance
        {
            return .nonClifford(tCount: 1)
        }
        let inv = normalized / (Double.pi * 0.125)
        let rounded = inv.rounded()
        if abs(inv - rounded) < piEighthTolerance {
            return .nonClifford(tCount: 2)
        }
        return .nonClifford(tCount: 3)
    }

    /// Combine two classifications by summing T-counts.
    @inline(__always)
    @usableFromInline
    @_effects(readonly)
    static func combineClassifications(_ a: Classification, _ b: Classification) -> Classification {
        switch (a, b) {
        case (.clifford, .clifford):
            .clifford
        case let (.clifford, .nonClifford(tCount)):
            .nonClifford(tCount: tCount)
        case let (.nonClifford(tCount), .clifford):
            .nonClifford(tCount: tCount)
        case let (.nonClifford(tCount1), .nonClifford(tCount2)):
            .nonClifford(tCount: tCount1 + tCount2)
        }
    }

    /// Classify a controlled gate based on its target gate classification.
    @usableFromInline
    @_effects(readonly)
    static func classifyControlledClifford(_ innerGate: QuantumGate) -> Classification {
        switch innerGate {
        case .pauliX, .pauliY, .pauliZ, .hadamard, .sGate:
            .clifford
        case let .phase(angle):
            classifyAngle(angle)
        default:
            .nonClifford(tCount: 1)
        }
    }

    /// Classify a gate defined by diagonal phase angles.
    @usableFromInline
    @_effects(readonly)
    static func classifyDiagonalPhases(_ phases: [Double]) -> Classification {
        var totalTCount = 0
        for phase in phases {
            let classification = classifyAngleValue(phase)
            switch classification {
            case .clifford:
                break
            case let .nonClifford(tCount):
                totalTCount += tCount
            }
        }
        if totalTCount == 0 {
            return .clifford
        }
        return .nonClifford(tCount: totalTCount)
    }

    /// Classify a multiplexed gate by analyzing its sub-unitary blocks.
    @usableFromInline
    @_effects(readonly)
    static func classifyMultiplexor(_ unitaries: [[[Complex<Double>]]]) -> Classification {
        var totalTCount = 0
        for unitary in unitaries {
            let classification = classifyCustomMatrix(unitary)
            switch classification {
            case .clifford:
                break
            case let .nonClifford(tCount):
                totalTCount += tCount
            }
        }
        if totalTCount == 0 {
            return .clifford
        }
        return .nonClifford(tCount: totalTCount)
    }

    /// Classify a custom gate by inspecting its unitary matrix.
    @usableFromInline
    @_effects(readonly)
    static func classifyCustomMatrix(_ matrix: [[Complex<Double>]]) -> Classification {
        if isCliffordMatrix(matrix) {
            return .clifford
        }
        let dimension = matrix.count
        let estimatedTCount = dimension > 2 ? dimension : 1
        return .nonClifford(tCount: estimatedTCount)
    }

    /// Check if a unitary matrix represents a Clifford operation.
    @usableFromInline
    @_effects(readonly)
    static func isCliffordMatrix(_ matrix: [[Complex<Double>]]) -> Bool {
        let toleranceSq = tolerance * tolerance
        for row in matrix {
            for element in row {
                let magSq = element.magnitudeSquared
                guard magSq < toleranceSq
                    || abs(magSq - 0.25) < tolerance
                    || abs(magSq - 0.5) < tolerance
                    || abs(magSq - 1.0) < tolerance
                else { return false }

                if magSq < toleranceSq { continue }

                let phase = normalizeAngle(element.phase)
                guard abs(phase) < tolerance
                    || abs(phase - (.pi * 0.5)) < tolerance
                    || abs(phase - .pi) < tolerance
                    || abs(phase - (.pi * 1.5)) < tolerance
                else { return false }
            }
        }
        return true
    }
}
