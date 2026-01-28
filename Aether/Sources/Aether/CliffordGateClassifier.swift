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
public enum CliffordGateClassifier {
    /// Classification result for a quantum gate.
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
    public enum Classification: Equatable, Sendable {
        case clifford
        case nonClifford(tCount: Int)
    }

    /// Classifies a quantum gate as Clifford or non-Clifford.
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
    /// - Parameter gate: The quantum gate to classify
    /// - Returns: Classification indicating Clifford status and T-count if non-Clifford
    /// - Complexity: O(1) for most gates, O(n^2) for custom unitary matrices
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
            return classifyPhaseAngle(angle)

        case let .rotationX(theta):
            return classifyRotationAngle(theta)

        case let .rotationY(theta):
            return classifyRotationAngle(theta)

        case let .rotationZ(theta):
            return classifyRotationAngle(theta)

        case let .globalPhase(phi):
            return classifyPhaseAngle(phi)

        case let .u1(lambda):
            return classifyPhaseAngle(lambda)

        case let .u2(phi, lambda):
            let phiClass = classifyPhaseAngle(phi)
            let lambdaClass = classifyPhaseAngle(lambda)
            return combineClassifications(phiClass, lambdaClass)

        case let .u3(theta, phi, lambda):
            let thetaClass = classifyRotationAngle(theta)
            let phiClass = classifyPhaseAngle(phi)
            let lambdaClass = classifyPhaseAngle(lambda)
            return combineClassifications(combineClassifications(thetaClass, phiClass), lambdaClass)

        case let .controlledPhase(theta):
            return classifyPhaseAngle(theta)

        case let .controlledRotationX(theta):
            return classifyRotationAngle(theta)

        case let .controlledRotationY(theta):
            return classifyRotationAngle(theta)

        case let .controlledRotationZ(theta):
            return classifyRotationAngle(theta)

        case .sqrtSwap, .sqrtISwap, .fswap:
            return .nonClifford(tCount: 1)

        case let .givens(theta):
            return classifyRotationAngle(theta)

        case let .xx(theta):
            return classifyRotationAngle(theta)

        case let .yy(theta):
            return classifyRotationAngle(theta)

        case let .zz(theta):
            return classifyRotationAngle(theta)

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

    /// Checks if a quantum gate is a Clifford gate.
    ///
    /// Convenience method that returns true if the gate classification is `.clifford`.
    ///
    /// **Example:**
    /// ```swift
    /// CliffordGateClassifier.isClifford(.hadamard)  // true
    /// CliffordGateClassifier.isClifford(.tGate)     // false
    /// ```
    ///
    /// - Parameter gate: The quantum gate to check
    /// - Returns: True if the gate is a Clifford gate
    /// - Complexity: O(1) for most gates
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

    /// Analyzes a quantum circuit for Clifford structure and T-count.
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
    /// - Parameter circuit: The quantum circuit to analyze
    /// - Returns: Tuple containing Clifford status and total T-count
    /// - Complexity: O(n) where n is the number of operations in the circuit
    @inlinable
    @_effects(readonly)
    public static func analyze(_ circuit: QuantumCircuit) -> (isClifford: Bool, tCount: Int) {
        var totalTCount = 0
        var allClifford = true

        for operation in circuit.operations {
            switch operation {
            case let .gate(gate, _, _):
                let classification = classify(gate)
                switch classification {
                case .clifford:
                    break
                case let .nonClifford(tCount):
                    allClifford = false
                    totalTCount += tCount
                }
            case .reset, .measure:
                break
            }
        }

        return (isClifford: allClifford, tCount: totalTCount)
    }

    @usableFromInline
    @_effects(readonly)
    static func classifyPhaseAngle(_ angle: ParameterValue) -> Classification {
        guard case let .value(theta) = angle else {
            return .nonClifford(tCount: 1)
        }
        return classifyAngleValue(theta, isCliffordAt: isCliffordPhaseAngle)
    }

    @usableFromInline
    @_effects(readonly)
    static func classifyRotationAngle(_ angle: ParameterValue) -> Classification {
        guard case let .value(theta) = angle else {
            return .nonClifford(tCount: 1)
        }
        return classifyAngleValue(theta, isCliffordAt: isCliffordRotationAngle)
    }

    @usableFromInline
    @_effects(readonly)
    static func classifyAngleValue(_ theta: Double, isCliffordAt: (Double) -> Bool) -> Classification {
        if isCliffordAt(theta) {
            return .clifford
        }
        return estimateTCountForAngle(theta)
    }

    @usableFromInline
    @_effects(readonly)
    static func isCliffordPhaseAngle(_ theta: Double) -> Bool {
        let normalized = normalizeAngle(theta)
        let tolerance = 1e-10
        let cliffordAngles: [Double] = [0.0, .pi / 2.0, .pi, 3.0 * .pi / 2.0]
        for clifford in cliffordAngles {
            if abs(normalized - clifford) < tolerance {
                return true
            }
        }
        return false
    }

    @usableFromInline
    @_effects(readonly)
    static func isCliffordRotationAngle(_ theta: Double) -> Bool {
        let normalized = normalizeAngle(theta)
        let tolerance = 1e-10
        let cliffordAngles: [Double] = [0.0, .pi / 2.0, .pi, 3.0 * .pi / 2.0]
        for clifford in cliffordAngles {
            if abs(normalized - clifford) < tolerance {
                return true
            }
        }
        return false
    }

    @usableFromInline
    @_effects(readonly)
    static func normalizeAngle(_ theta: Double) -> Double {
        var normalized = theta.truncatingRemainder(dividingBy: 2.0 * .pi)
        if normalized < 0 {
            normalized += 2.0 * .pi
        }
        return normalized
    }

    @usableFromInline
    @_effects(readonly)
    static func estimateTCountForAngle(_ theta: Double) -> Classification {
        let normalized = normalizeAngle(theta)
        let tolerance = 1e-10
        let quarterPi = Double.pi / 4.0
        if abs(normalized - quarterPi) < tolerance ||
            abs(normalized - 3.0 * quarterPi) < tolerance ||
            abs(normalized - 5.0 * quarterPi) < tolerance ||
            abs(normalized - 7.0 * quarterPi) < tolerance
        {
            return .nonClifford(tCount: 1)
        }
        let eighthPi = Double.pi / 8.0
        for k in 1 ... 15 {
            if abs(normalized - Double(k) * eighthPi) < tolerance {
                return .nonClifford(tCount: 2)
            }
        }
        return .nonClifford(tCount: 3)
    }

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

    @usableFromInline
    @_effects(readonly)
    static func classifyControlledClifford(_ innerGate: QuantumGate) -> Classification {
        switch innerGate {
        case .pauliX, .pauliY, .pauliZ, .hadamard, .sGate:
            .clifford
        case let .phase(angle):
            classifyPhaseAngle(angle)
        default:
            .nonClifford(tCount: 1)
        }
    }

    @usableFromInline
    @_effects(readonly)
    static func classifyDiagonalPhases(_ phases: [Double]) -> Classification {
        var totalTCount = 0
        for phase in phases {
            let classification = classifyAngleValue(phase, isCliffordAt: isCliffordPhaseAngle)
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

    @usableFromInline
    @_effects(readonly)
    static func isCliffordMatrix(_ matrix: [[Complex<Double>]]) -> Bool {
        let allowedMagnitudes: [Double] = [0.0, 0.5, 1.0, 0.5.squareRoot()]
        let tolerance = 1e-10
        for row in matrix {
            for element in row {
                let magnitude = element.magnitude
                var found = false
                for allowed in allowedMagnitudes {
                    if abs(magnitude - allowed) < tolerance {
                        found = true
                        break
                    }
                }
                if !found {
                    return false
                }
            }
        }
        let allowedPhases: [Double] = [0.0, .pi / 2.0, .pi, 3.0 * .pi / 2.0]
        for row in matrix {
            for element in row {
                if element.magnitude < tolerance {
                    continue
                }
                let phase = normalizeAngle(element.phase)
                var found = false
                for allowed in allowedPhases {
                    if abs(phase - allowed) < tolerance {
                        found = true
                        break
                    }
                }
                if !found {
                    return false
                }
            }
        }
        return true
    }
}
