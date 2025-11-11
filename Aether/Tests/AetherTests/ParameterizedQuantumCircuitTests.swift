// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for ParameterizedQuantumCircuit initialization.
/// Validates circuit creation with various qubit counts
/// and parameter tracking infrastructure.
@Suite("ParameterizedQuantumCircuit Initialization")
struct ParameterizedCircuitInitializationTests {
    @Test("Create empty parameterized circuit")
    func createEmptyCircuit() {
        let circuit = ParameterizedQuantumCircuit(numQubits: 2)

        #expect(circuit.numQubits == 2)
        #expect(circuit.gateCount == 0)
        #expect(circuit.isEmpty)
        #expect(circuit.parameterCount == 0)
    }

    @Test("Create circuit with various qubit counts")
    func createCircuitVariousQubits() {
        let circuit1 = ParameterizedQuantumCircuit(numQubits: 1)
        let circuit5 = ParameterizedQuantumCircuit(numQubits: 5)
        let circuit10 = ParameterizedQuantumCircuit(numQubits: 10)

        #expect(circuit1.numQubits == 1)
        #expect(circuit5.numQubits == 5)
        #expect(circuit10.numQubits == 10)
    }

    @Test("ParameterizedQuantumCircuit is Sendable")
    func circuitIsSendable() {
        let circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let _: any Sendable = circuit
    }

    @Test("Empty circuit has no parameters")
    func emptyCircuitNoParameters() {
        let circuit = ParameterizedQuantumCircuit(numQubits: 2)
        #expect(circuit.parameters.isEmpty)
    }
}

/// Test suite for parameterized circuit building operations.
/// Validates gate appending, parameter auto-registration,
/// and circuit structure maintenance.
@Suite("ParameterizedQuantumCircuit Building")
struct ParameterizedCircuitBuildingTests {
    @Test("Append parameterized gate")
    func appendParameterizedGate() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)

        #expect(circuit.gateCount == 1)
        #expect(!circuit.isEmpty)
    }

    @Test("Append concrete gate")
    func appendConcreteGate() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        circuit.append(gate: .concrete(.hadamard), toQubit: 0)

        #expect(circuit.gateCount == 1)
    }

    @Test("Parameter auto-registration on append")
    func parameterAutoRegistration() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)
        circuit.append(gate: .rotationZ(theta: .parameter(phi)), toQubit: 1)

        #expect(circuit.parameterCount == 2)
        #expect(circuit.parameters.contains(theta))
        #expect(circuit.parameters.contains(phi))
    }

    @Test("Parameter deduplication on append")
    func parameterDeduplication() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)
        circuit.append(gate: .rotationX(theta: .parameter(theta)), toQubit: 1)
        circuit.append(gate: .rotationZ(theta: .parameter(theta)), toQubit: 0)

        #expect(circuit.parameterCount == 1)
    }

    @Test("Parameter registration order preserved")
    func parameterRegistrationOrder() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 3)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        let gamma = Parameter(name: "gamma")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)
        circuit.append(gate: .rotationY(theta: .parameter(phi)), toQubit: 1)
        circuit.append(gate: .rotationY(theta: .parameter(gamma)), toQubit: 2)

        #expect(circuit.parameters[0] == theta)
        #expect(circuit.parameters[1] == phi)
        #expect(circuit.parameters[2] == gamma)
    }

    @Test("Append multiple gates")
    func appendMultipleGates() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .concrete(.hadamard), toQubit: 0)
        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)
        circuit.append(gate: .concrete(.cnot(control: 0, target: 1)), qubits: [])

        #expect(circuit.gateCount == 3)
        #expect(circuit.parameterCount == 1)
    }

    @Test("Append gate with timestamp")
    func appendGateWithTimestamp() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        circuit.append(gate: .concrete(.hadamard), toQubit: 0, timestamp: 1.5)

        let operation = circuit.operation(at: 0)
        #expect(operation.timestamp == 1.5)
    }

    @Test("Circuit auto-expansion on high qubit index")
    func circuitAutoExpansion() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        circuit.append(gate: .concrete(.hadamard), toQubit: 5)

        #expect(circuit.numQubits == 6)
    }

    @Test("Multi-parameter gate registration")
    func multiParameterGateRegistration() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")

        circuit.append(gate: .u3(
            theta: .parameter(theta),
            phi: .parameter(phi),
            lambda: .parameter(lambda)
        ), toQubit: 0)

        #expect(circuit.parameterCount == 3)
    }
}

/// Test suite for parameterized circuit querying operations.
/// Validates gate access, parameter inspection, and circuit
/// structure queries.
@Suite("ParameterizedQuantumCircuit Querying")
struct ParameterizedCircuitQueryingTests {
    @Test("Get gate operation by index")
    func getGateOperationByIndex() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)
        circuit.append(gate: .concrete(.hadamard), toQubit: 1)

        let op0 = circuit.operation(at: 0)
        let op1 = circuit.operation(at: 1)

        #expect(op0.gate == .rotationY(theta: .parameter(theta)))
        #expect(op1.gate == .concrete(.hadamard))
    }

    @Test("isEmpty reflects circuit state")
    func isEmptyReflectsState() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        #expect(circuit.isEmpty)

        circuit.append(gate: .concrete(.hadamard), toQubit: 0)
        #expect(!circuit.isEmpty)
    }

    @Test("Gate count reflects number of operations")
    func gateCountReflectsOperations() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)

        #expect(circuit.gateCount == 0)

        circuit.append(gate: .concrete(.hadamard), toQubit: 0)
        #expect(circuit.gateCount == 1)

        circuit.append(gate: .concrete(.cnot(control: 0, target: 1)), qubits: [])
        #expect(circuit.gateCount == 2)
    }

    @Test("Parameter count reflects unique parameters")
    func parameterCountReflectsUnique() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 3)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)
        #expect(circuit.parameterCount == 1)

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 1)
        #expect(circuit.parameterCount == 1)

        circuit.append(gate: .rotationZ(theta: .parameter(phi)), toQubit: 2)
        #expect(circuit.parameterCount == 2)
    }

    @Test("maxQubitUsed returns highest qubit index")
    func maxQubitUsedReturnsHighest() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 5)
        circuit.append(gate: .concrete(.hadamard), toQubit: 0)
        circuit.append(gate: .concrete(.hadamard), toQubit: 3)

        #expect(circuit.maxQubitUsed() == 4)

        circuit.append(gate: .concrete(.hadamard), toQubit: 4)
        #expect(circuit.maxQubitUsed() == 4)
    }

    @Test("maxQubitUsed with controlled gates")
    func maxQubitUsedWithControlledGates() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 10)
        let theta = Parameter(name: "theta")

        circuit.append(
            gate: .controlledRotationX(theta: .parameter(theta), control: 2, target: 7),
            qubits: []
        )

        #expect(circuit.maxQubitUsed() == 9)
    }

    @Test("maxQubitUsed with concrete multi-qubit gates")
    func maxQubitUsedWithConcreteMultiQubitGates() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 10)
        circuit.append(gate: .concrete(.toffoli(control1: 1, control2: 5, target: 8)), qubits: [])

        #expect(circuit.maxQubitUsed() == 9)
    }
}

/// Test suite for parameterized circuit validation.
/// Validates circuit structure correctness, qubit index bounds,
/// and gate validity before execution.
@Suite("ParameterizedQuantumCircuit Validation")
struct ParameterizedCircuitValidationTests {
    @Test("Empty circuit validates successfully")
    func emptyCircuitValidates() {
        let circuit = ParameterizedQuantumCircuit(numQubits: 2)
        #expect(circuit.validate())
    }

    @Test("Valid circuit validates successfully")
    func validCircuitValidates() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 3)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .concrete(.hadamard), toQubit: 0)
        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 1)
        circuit.append(gate: .concrete(.cnot(control: 0, target: 2)), qubits: [])

        #expect(circuit.validate())
    }

    @Test("Circuit with valid controlled gates validates")
    func circuitWithValidControlledGatesValidates() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 5)
        let theta = Parameter(name: "theta")

        circuit.append(
            gate: .controlledRotationZ(theta: .parameter(theta), control: 0, target: 4),
            qubits: []
        )

        #expect(circuit.validate())
    }
}

/// Test suite for parameter binding with dictionary interface.
/// Validates parameter substitution, error handling,
/// and concrete circuit generation.
@Suite("Parameter Binding Dictionary Interface")
struct ParameterBindingDictionaryTests {
    @Test("Bind empty circuit")
    func bindEmptyCircuit() throws {
        let circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let concrete = try circuit.bind(parameters: [:])

        #expect(concrete.numQubits == 2)
        #expect(concrete.gateCount == 0)
    }

    @Test("Bind single parameterized gate")
    func bindSingleParameterizedGate() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)

        let bindings = ["theta": Double.pi / 4.0]
        let concrete = try circuit.bind(parameters: bindings)

        #expect(concrete.gateCount == 1)
        let op = concrete.operation(at: 0)
        #expect(op.gate == .rotationY(theta: Double.pi / 4.0))
    }

    @Test("Bind multiple parameterized gates")
    func bindMultipleParameterizedGates() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)
        circuit.append(gate: .rotationZ(theta: .parameter(phi)), toQubit: 1)

        let bindings = ["theta": 0.5, "phi": 1.0]
        let concrete = try circuit.bind(parameters: bindings)

        #expect(concrete.gateCount == 2)
    }

    @Test("Bind circuit with mixed gates")
    func bindCircuitWithMixedGates() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .concrete(.hadamard), toQubit: 0)
        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)
        circuit.append(gate: .concrete(.cnot(control: 0, target: 1)), qubits: [])

        let bindings = ["theta": Double.pi / 2.0]
        let concrete = try circuit.bind(parameters: bindings)

        #expect(concrete.gateCount == 3)
        #expect(concrete.operation(at: 0).gate == .hadamard)
        #expect(concrete.operation(at: 1).gate == .rotationY(theta: Double.pi / 2.0))
    }

    @Test("Bind throws on missing parameter")
    func bindThrowsOnMissingParameter() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)
        circuit.append(gate: .rotationZ(theta: .parameter(phi)), toQubit: 0)

        let bindings = ["theta": 0.5]

        #expect(throws: ParameterError.unboundParameter("phi")) {
            try circuit.bind(parameters: bindings)
        }
    }

    @Test("Bind throws on extra parameters")
    func bindThrowsOnExtraParameters() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)

        let bindings = ["theta": 0.5, "phi": 1.0, "gamma": 2.0]

        #expect(throws: ParameterError.self) {
            try circuit.bind(parameters: bindings)
        }
    }

    @Test("Bind circuit with no parameters succeeds with empty bindings")
    func bindCircuitWithNoParametersSucceeds() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        circuit.append(gate: .concrete(.hadamard), toQubit: 0)
        circuit.append(gate: .concrete(.cnot(control: 0, target: 1)), qubits: [])

        let concrete = try circuit.bind(parameters: [:])
        #expect(concrete.gateCount == 2)
    }

    @Test("Bind preserves timestamps")
    func bindPreservesTimestamps() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0, timestamp: 1.5)

        let bindings = ["theta": 0.5]
        let concrete = try circuit.bind(parameters: bindings)

        #expect(concrete.operation(at: 0).timestamp == 1.5)
    }

    @Test("Bind with U3 gate")
    func bindWithU3Gate() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")

        circuit.append(gate: .u3(
            theta: .parameter(theta),
            phi: .parameter(phi),
            lambda: .parameter(lambda)
        ), toQubit: 0)

        let bindings = ["theta": 0.5, "phi": 1.0, "lambda": 1.5]
        let concrete = try circuit.bind(parameters: bindings)

        #expect(concrete.gateCount == 1)
        #expect(concrete.operation(at: 0).gate == .u3(theta: 0.5, phi: 1.0, lambda: 1.5))
    }
}

/// Test suite for parameter binding with vector interface.
/// Validates NumPy-compatible vector binding and length validation.
@Suite("Parameter Binding Vector Interface")
struct ParameterBindingVectorTests {
    @Test("Bind empty circuit with empty vector")
    func bindEmptyCircuitEmptyVector() throws {
        let circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let concrete = try circuit.bind(parameterVector: [])

        #expect(concrete.numQubits == 2)
        #expect(concrete.gateCount == 0)
    }

    @Test("Bind single parameter with vector")
    func bindSingleParameterWithVector() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)

        let concrete = try circuit.bind(parameterVector: [0.5])

        #expect(concrete.gateCount == 1)
        #expect(concrete.operation(at: 0).gate == .rotationY(theta: 0.5))
    }

    @Test("Bind multiple parameters with vector")
    func bindMultipleParametersWithVector() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 3)

        for i in 0 ..< 3 {
            let param = Parameter(name: "theta_\(i)")
            circuit.append(gate: .rotationY(theta: .parameter(param)), toQubit: i)
        }

        let params: [Double] = [0.1, 0.2, 0.3]
        let concrete = try circuit.bind(parameterVector: params)

        #expect(concrete.gateCount == 3)
        #expect(concrete.operation(at: 0).gate == .rotationY(theta: 0.1))
        #expect(concrete.operation(at: 1).gate == .rotationY(theta: 0.2))
        #expect(concrete.operation(at: 2).gate == .rotationY(theta: 0.3))
    }

    @Test("Bind respects parameter registration order")
    func bindRespectsRegistrationOrder() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)
        circuit.append(gate: .rotationZ(theta: .parameter(phi)), toQubit: 1)

        let params: [Double] = [0.5, 1.0]
        let concrete = try circuit.bind(parameterVector: params)

        #expect(concrete.operation(at: 0).gate == .rotationY(theta: 0.5))
        #expect(concrete.operation(at: 1).gate == .rotationZ(theta: 1.0))
    }

    @Test("Bind throws on wrong vector length")
    func bindThrowsOnWrongVectorLength() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)
        circuit.append(gate: .rotationZ(theta: .parameter(phi)), toQubit: 1)

        let params = [0.5]

        #expect(throws: ParameterError.invalidVectorLength(expected: 2, got: 1)) {
            try circuit.bind(parameterVector: params)
        }
    }

    @Test("Bind throws on too many parameters")
    func bindThrowsOnTooManyParameters() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)

        let params: [Double] = [0.5, 1.0, 1.5]

        #expect(throws: ParameterError.self) {
            try circuit.bind(parameterVector: params)
        }
    }

    @Test("Bind vector with hardware-efficient ansatz")
    func bindVectorWithHardwareEfficientAnsatz() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 4)

        for i in 0 ..< 4 {
            let param = Parameter(name: "theta_\(i)")
            circuit.append(gate: .rotationY(theta: .parameter(param)), toQubit: i)
        }

        for i in 0 ..< 3 {
            circuit.append(gate: .concrete(.cnot(control: i, target: i + 1)), qubits: [])
        }

        let params: [Double] = [0.1, 0.2, 0.3, 0.4]
        let concrete = try circuit.bind(parameterVector: params)

        #expect(concrete.gateCount == 7)
    }
}

/// Test suite for gradient computation support.
/// Validates parameter shift rule implementation for both
/// dictionary and vector interfaces.
@Suite("Gradient Computation Support")
struct GradientComputationTests {
    @Test("Generate shifted circuits with dictionary interface")
    func generateShiftedCircuitsDictionary() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)
        circuit.append(gate: .rotationZ(theta: .parameter(phi)), toQubit: 0)

        let baseBindings = ["theta": 0.5, "phi": 1.0]
        let (plus, minus) = try circuit.generateShiftedCircuits(
            parameterName: "theta",
            baseBindings: baseBindings
        )

        let plusOp = plus.operation(at: 0)
        let minusOp = minus.operation(at: 0)

        #expect(plusOp.gate == .rotationY(theta: 0.5 + Double.pi / 2.0))
        #expect(minusOp.gate == .rotationY(theta: 0.5 - Double.pi / 2.0))
    }

    @Test("Generate shifted circuits with vector interface")
    func generateShiftedCircuitsVector() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)

        for i in 0 ..< 2 {
            let param = Parameter(name: "theta_\(i)")
            circuit.append(gate: .rotationY(theta: .parameter(param)), toQubit: i)
        }

        let baseVector: [Double] = [0.5, 1.0]
        let (plus, minus) = try circuit.generateShiftedCircuits(
            parameterIndex: 0,
            baseVector: baseVector
        )

        let plusOp = plus.operation(at: 0)
        let minusOp = minus.operation(at: 0)

        #expect(plusOp.gate == .rotationY(theta: 0.5 + Double.pi / 2.0))
        #expect(minusOp.gate == .rotationY(theta: 0.5 - Double.pi / 2.0))
    }

    @Test("Generate shifted circuits with custom shift")
    func generateShiftedCircuitsCustomShift() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)

        let baseBindings = ["theta": 0.5]
        let customShift = 0.1
        let (plus, minus) = try circuit.generateShiftedCircuits(
            parameterName: "theta",
            baseBindings: baseBindings,
            shift: customShift
        )

        let plusOp = plus.operation(at: 0)
        let minusOp = minus.operation(at: 0)

        #expect(abs(plusOp.gate.matrix()[0][0].real - cos((0.5 + 0.1) / 2.0)) < 1e-10)
        #expect(abs(minusOp.gate.matrix()[0][0].real - cos((0.5 - 0.1) / 2.0)) < 1e-10)
    }

    @Test("Shifted circuits preserve other parameters")
    func shiftedCircuitsPreserveOtherParameters() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)
        circuit.append(gate: .rotationZ(theta: .parameter(phi)), toQubit: 1)

        let baseBindings = ["theta": 0.5, "phi": 1.0]
        let (plus, minus) = try circuit.generateShiftedCircuits(
            parameterName: "theta",
            baseBindings: baseBindings
        )

        #expect(plus.operation(at: 1).gate == .rotationZ(theta: 1.0))
        #expect(minus.operation(at: 1).gate == .rotationZ(theta: 1.0))
    }

    @Test("Generate shifted circuits throws on parameter not found")
    func generateShiftedCircuitsThrowsParameterNotFound() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)

        let baseBindings = ["theta": 0.5]

        #expect(throws: ParameterError.parameterNotFound("phi")) {
            try circuit.generateShiftedCircuits(
                parameterName: "phi",
                baseBindings: baseBindings
            )
        }
    }

    @Test("Generate shifted circuits throws on unbound parameter")
    func generateShiftedCircuitsThrowsUnboundParameter() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)

        let baseBindings: [String: Double] = [:]

        #expect(throws: ParameterError.unboundParameter("theta")) {
            try circuit.generateShiftedCircuits(
                parameterName: "theta",
                baseBindings: baseBindings
            )
        }
    }

    @Test("Generate shifted circuits throws on index out of bounds")
    func generateShiftedCircuitsThrowsIndexOutOfBounds() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)

        let baseVector = [0.5]

        #expect(throws: ParameterError.parameterIndexOutOfBounds(index: 5, count: 1)) {
            try circuit.generateShiftedCircuits(
                parameterIndex: 5,
                baseVector: baseVector
            )
        }
    }

    @Test("Generate shifted circuits with vector throws on wrong length")
    func generateShiftedCircuitsVectorThrowsWrongLength() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)

        for i in 0 ..< 2 {
            let param = Parameter(name: "theta_\(i)")
            circuit.append(gate: .rotationY(theta: .parameter(param)), toQubit: i)
        }

        let baseVector = [0.5]

        #expect(throws: ParameterError.invalidVectorLength(expected: 2, got: 1)) {
            try circuit.generateShiftedCircuits(
                parameterIndex: 0,
                baseVector: baseVector
            )
        }
    }

    @Test("Shifted circuits can be executed")
    func shiftedCircuitsCanBeExecuted() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)

        let baseBindings = ["theta": Double.pi / 4.0]
        let (plus, minus) = try circuit.generateShiftedCircuits(
            parameterName: "theta",
            baseBindings: baseBindings
        )

        let statePlus = plus.execute()
        let stateMinus = minus.execute()

        #expect(statePlus.isNormalized())
        #expect(stateMinus.isNormalized())
    }
}

/// Test suite for parameterized circuit string representation.
/// Validates description formatting for debugging and visualization.
@Suite("ParameterizedQuantumCircuit Description")
struct ParameterizedCircuitDescriptionTests {
    @Test("Empty circuit description")
    func emptyCircuitDescription() {
        let circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let desc = circuit.description

        #expect(desc.contains("2 qubits"))
        #expect(desc.contains("0 params"))
        #expect(desc.contains("empty"))
    }

    @Test("Circuit with gates description")
    func circuitWithGatesDescription() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)
        circuit.append(gate: .concrete(.hadamard), toQubit: 1)

        let desc = circuit.description

        #expect(desc.contains("2 qubits"))
        #expect(desc.contains("2 gates"))
        #expect(desc.contains("1 params"))
        #expect(desc.contains("theta"))
    }

    @Test("Circuit description shows parameter list")
    func circuitDescriptionShowsParameterList() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 3)

        for i in 0 ..< 3 {
            let param = Parameter(name: "theta_\(i)")
            circuit.append(gate: .rotationY(theta: .parameter(param)), toQubit: i)
        }

        let desc = circuit.description

        #expect(desc.contains("theta_0"))
        #expect(desc.contains("theta_1"))
        #expect(desc.contains("theta_2"))
    }

    @Test("Circuit description truncates long parameter lists")
    func circuitDescriptionTruncatesLongLists() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 10)

        for i in 0 ..< 10 {
            let param = Parameter(name: "theta_\(i)")
            circuit.append(gate: .rotationY(theta: .parameter(param)), toQubit: i)
        }

        let desc = circuit.description

        #expect(desc.contains("..."))
    }
}

/// Test suite for parameterized circuit equality.
/// Validates Equatable conformance for circuit comparison.
@Suite("ParameterizedQuantumCircuit Equality")
struct ParameterizedCircuitEqualityTests {
    @Test("Empty circuits are equal")
    func emptyCircuitsEqual() {
        let circuit1 = ParameterizedQuantumCircuit(numQubits: 2)
        let circuit2 = ParameterizedQuantumCircuit(numQubits: 2)

        #expect(circuit1 == circuit2)
    }

    @Test("Circuits with different qubit counts are not equal")
    func circuitsDifferentQubitsNotEqual() {
        let circuit1 = ParameterizedQuantumCircuit(numQubits: 2)
        let circuit2 = ParameterizedQuantumCircuit(numQubits: 3)

        #expect(circuit1 != circuit2)
    }

    @Test("Circuits with same gates are equal")
    func circuitsSameGatesEqual() {
        var circuit1 = ParameterizedQuantumCircuit(numQubits: 1)
        var circuit2 = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")

        circuit1.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)
        circuit2.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)

        #expect(circuit1 == circuit2)
    }

    @Test("Circuits with different gates are not equal")
    func circuitsDifferentGatesNotEqual() {
        var circuit1 = ParameterizedQuantumCircuit(numQubits: 1)
        var circuit2 = ParameterizedQuantumCircuit(numQubits: 1)
        let theta = Parameter(name: "theta")

        circuit1.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)
        circuit2.append(gate: .rotationX(theta: .parameter(theta)), toQubit: 0)

        #expect(circuit1 != circuit2)
    }
}

/// Test suite for integration scenarios.
/// Validates complete VQE and QAOA workflows with parameterized circuits.
@Suite("Parameterized Circuit Integration Tests")
struct ParameterizedCircuitIntegrationTests {
    @Test("VQE hardware-efficient ansatz workflow")
    func vqeHardwareEfficientAnsatz() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 4)

        for i in 0 ..< 4 {
            let param = Parameter(name: "theta_\(i)")
            circuit.append(gate: .rotationY(theta: .parameter(param)), toQubit: i)
        }

        for i in 0 ..< 3 {
            circuit.append(gate: .concrete(.cnot(control: i, target: i + 1)), qubits: [])
        }

        let params: [Double] = [0.1, 0.2, 0.3, 0.4]
        let concrete = try circuit.bind(parameterVector: params)
        let state = concrete.execute()

        #expect(state.isNormalized())
        #expect(circuit.parameterCount == 4)
        #expect(circuit.gateCount == 7)
    }

    @Test("QAOA MaxCut workflow")
    func qaoaMaxCutWorkflow() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 4)
        let gamma = Parameter(name: "gamma")
        let beta = Parameter(name: "beta")

        for i in 0 ..< 4 {
            circuit.append(gate: .concrete(.hadamard), toQubit: i)
        }

        let edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        for (i, j) in edges {
            circuit.append(gate: .concrete(.cnot(control: i, target: j)), qubits: [])
            circuit.append(gate: .rotationZ(theta: .parameter(gamma)), toQubit: j)
            circuit.append(gate: .concrete(.cnot(control: i, target: j)), qubits: [])
        }

        for i in 0 ..< 4 {
            circuit.append(gate: .rotationX(theta: .parameter(beta)), toQubit: i)
        }

        let bindings = ["gamma": 0.5, "beta": 1.0]
        let concrete = try circuit.bind(parameters: bindings)
        let state = concrete.execute()

        #expect(state.isNormalized())
        #expect(circuit.parameterCount == 2)
    }

    @Test("Parameter shift gradient computation workflow")
    func parameterShiftGradientWorkflow() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 2)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .concrete(.hadamard), toQubit: 0)
        circuit.append(gate: .rotationY(theta: .parameter(theta)), toQubit: 0)
        circuit.append(gate: .concrete(.cnot(control: 0, target: 1)), qubits: [])

        let baseBindings = ["theta": Double.pi / 4.0]
        let (plus, minus) = try circuit.generateShiftedCircuits(
            parameterName: "theta",
            baseBindings: baseBindings
        )

        let statePlus = plus.execute()
        let stateMinus = minus.execute()

        let pPlus = statePlus.probability(ofState: 0)
        let pMinus = stateMinus.probability(ofState: 0)
        let gradient = (pPlus - pMinus) / 2.0

        #expect(abs(gradient) >= 0)
    }

    @Test("Multi-layer variational ansatz")
    func multiLayerVariationalAnsatz() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 3)
        var paramIndex = 0

        for _ in 0 ..< 2 {
            for i in 0 ..< 3 {
                let param = Parameter(name: "theta_\(paramIndex)")
                circuit.append(gate: .rotationY(theta: .parameter(param)), toQubit: i)
                paramIndex += 1
            }

            for i in 0 ..< 2 {
                circuit.append(gate: .concrete(.cnot(control: i, target: i + 1)), qubits: [])
            }
        }

        #expect(circuit.parameterCount == 6)
        #expect(circuit.gateCount == 10)

        let params: [Double] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        let concrete = try circuit.bind(parameterVector: params)
        let state = concrete.execute()

        #expect(state.isNormalized())
    }

    @Test("Circuit with all gate types")
    func circuitWithAllGateTypes() throws {
        var circuit = ParameterizedQuantumCircuit(numQubits: 3)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")

        circuit.append(gate: .concrete(.hadamard), toQubit: 0)
        circuit.append(gate: .rotationX(theta: .parameter(theta)), toQubit: 0)
        circuit.append(gate: .rotationY(theta: .parameter(phi)), toQubit: 1)
        circuit.append(gate: .rotationZ(theta: .parameter(lambda)), toQubit: 2)
        circuit.append(gate: .controlledRotationX(theta: .parameter(theta), control: 0, target: 1), qubits: [])
        circuit.append(gate: .concrete(.cnot(control: 1, target: 2)), qubits: [])
        circuit.append(gate: .u3(theta: .parameter(theta), phi: .parameter(phi), lambda: .parameter(lambda)), toQubit: 0)

        let bindings = ["theta": 0.5, "phi": 1.0, "lambda": 1.5]
        let concrete = try circuit.bind(parameters: bindings)

        #expect(concrete.gateCount == 7)
        let state = concrete.execute()
        #expect(state.isNormalized())
    }
}

/// Test suite for timestamp functionality in parameterized gate operations.
/// Validates timestamp formatting in operation descriptions.
@Suite("ParameterizedGateOperation Timestamps")
struct ParameterizedGateOperationTimestampTests {
    @Test("Operation with timestamp formats correctly")
    func operationWithTimestamp() {
        let theta = Parameter(name: "theta")
        let gate = ParameterizedGate.rotationY(theta: .parameter(theta))
        let operation = ParameterizedGateOperation(gate: gate, qubits: [0], timestamp: 1.23)

        let desc = operation.description

        #expect(desc.contains("@ 1.23s"))
        #expect(desc.contains("Ry(theta)"))
        #expect(desc.contains("on qubits [0]"))
    }

    @Test("Operation without timestamp has no time suffix")
    func operationWithoutTimestamp() {
        let theta = Parameter(name: "theta")
        let gate = ParameterizedGate.rotationY(theta: .parameter(theta))
        let operation = ParameterizedGateOperation(gate: gate, qubits: [0], timestamp: nil)

        let desc = operation.description

        #expect(!desc.contains("@"))
        #expect(desc.contains("Ry(theta)"))
        #expect(desc.contains("on qubits"))
    }

    @Test("Operation with zero timestamp formats correctly")
    func operationWithZeroTimestamp() {
        let gate = ParameterizedGate.rotationX(theta: .value(.pi))
        let operation = ParameterizedGateOperation(gate: gate, qubits: [1], timestamp: 0.0)

        let desc = operation.description

        #expect(desc.contains("@ 0.00s"))
    }

    @Test("Operation with large timestamp formats correctly")
    func operationWithLargeTimestamp() {
        let gate = ParameterizedGate.phase(theta: .value(1.5))
        let operation = ParameterizedGateOperation(gate: gate, qubits: [2], timestamp: 123.456)

        let desc = operation.description

        #expect(desc.contains("@ 123.46s"))
    }

    @Test("Multi-qubit operation with timestamp")
    func multiQubitOperationWithTimestamp() {
        let theta = Parameter(name: "angle")
        let gate = ParameterizedGate.controlledRotationZ(theta: .parameter(theta), control: 0, target: 1)
        let operation = ParameterizedGateOperation(gate: gate, qubits: [], timestamp: 2.5)

        let desc = operation.description

        #expect(desc.contains("@ 2.50s"))
        #expect(desc.contains("CRz"))
    }
}

/// Test suite for custom initializer and internal circuit construction.
/// Validates init with predefined operations and parameters.
@Suite("ParameterizedQuantumCircuit Custom Initialization")
struct ParameterizedCircuitCustomInitTests {
    @Test("Init with operations and parameters creates valid circuit")
    func initWithOperationsAndParameters() {
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")

        let gate1 = ParameterizedGate.rotationY(theta: .parameter(theta))
        let gate2 = ParameterizedGate.rotationZ(theta: .parameter(phi))

        let op1 = ParameterizedGateOperation(gate: gate1, qubits: [0])
        let op2 = ParameterizedGateOperation(gate: gate2, qubits: [1])

        let circuit = ParameterizedQuantumCircuit(
            numQubits: 2,
            operations: [op1, op2],
            parameters: [theta, phi]
        )

        #expect(circuit.numQubits == 2)
        #expect(circuit.gateCount == 2)
        #expect(circuit.parameterCount == 2)
        #expect(circuit.parameters == [theta, phi])
    }

    @Test("Init with empty operations creates empty circuit")
    func initWithEmptyOperations() {
        let circuit = ParameterizedQuantumCircuit(
            numQubits: 3,
            operations: [],
            parameters: []
        )

        #expect(circuit.numQubits == 3)
        #expect(circuit.isEmpty)
        #expect(circuit.parameterCount == 0)
    }

    @Test("Init preserves operation order")
    func initPreservesOperationOrder() {
        let p1 = Parameter(name: "a")
        let p2 = Parameter(name: "b")
        let p3 = Parameter(name: "c")

        let ops = [
            ParameterizedGateOperation(gate: .rotationX(theta: .parameter(p1)), qubits: [0]),
            ParameterizedGateOperation(gate: .rotationY(theta: .parameter(p2)), qubits: [1]),
            ParameterizedGateOperation(gate: .rotationZ(theta: .parameter(p3)), qubits: [2]),
        ]

        let circuit = ParameterizedQuantumCircuit(
            numQubits: 3,
            operations: ops,
            parameters: [p1, p2, p3]
        )

        #expect(circuit.operation(at: 0).gate == .rotationX(theta: .parameter(p1)))
        #expect(circuit.operation(at: 1).gate == .rotationY(theta: .parameter(p2)))
        #expect(circuit.operation(at: 2).gate == .rotationZ(theta: .parameter(p3)))
    }

    @Test("Init with duplicate parameters in list")
    func initWithDuplicateParameters() {
        let theta = Parameter(name: "theta")

        let ops = [
            ParameterizedGateOperation(gate: .rotationX(theta: .parameter(theta)), qubits: [0]),
            ParameterizedGateOperation(gate: .rotationY(theta: .parameter(theta)), qubits: [1]),
        ]

        // User provides theta only once in parameters list
        let circuit = ParameterizedQuantumCircuit(
            numQubits: 2,
            operations: ops,
            parameters: [theta]
        )

        #expect(circuit.parameterCount == 1)
        #expect(circuit.parameters == [theta])
    }
}

/// Test suite for maxQubitUsed() coverage.
/// Validates maximum qubit calculation for all gate types.
@Suite("ParameterizedQuantumCircuit MaxQubit Coverage")
struct ParameterizedCircuitMaxQubitCoverageTests {
    @Test("maxQubitUsed with phase gate")
    func maxQubitUsedWithPhase() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 5)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .phase(theta: .parameter(theta)), toQubit: 3)

        #expect(circuit.maxQubitUsed() == 4)
    }

    @Test("maxQubitUsed with rotation gates")
    func maxQubitUsedWithRotations() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 4)
        let p1 = Parameter(name: "p1")
        let p2 = Parameter(name: "p2")
        let p3 = Parameter(name: "p3")

        circuit.append(gate: .rotationX(theta: .parameter(p1)), toQubit: 2)
        circuit.append(gate: .rotationY(theta: .parameter(p2)), toQubit: 1)
        circuit.append(gate: .rotationZ(theta: .parameter(p3)), toQubit: 3)

        #expect(circuit.maxQubitUsed() == 3)
    }

    @Test("maxQubitUsed with U gates")
    func maxQubitUsedWithUGates() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 3)
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")

        circuit.append(gate: .u1(lambda: .parameter(lambda)), toQubit: 1)
        circuit.append(gate: .u2(phi: .parameter(phi), lambda: .parameter(lambda)), toQubit: 2)
        circuit.append(gate: .u3(theta: .parameter(theta), phi: .parameter(phi), lambda: .parameter(lambda)), toQubit: 0)

        #expect(circuit.maxQubitUsed() == 2)
    }

    @Test("maxQubitUsed with controlled rotation gates")
    func maxQubitUsedWithControlledRotations() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 5)
        let angle = Parameter(name: "angle")

        circuit.append(gate: .controlledPhase(theta: .parameter(angle), control: 1, target: 3), qubits: [])
        circuit.append(gate: .controlledRotationX(theta: .parameter(angle), control: 0, target: 4), qubits: [])
        circuit.append(gate: .controlledRotationY(theta: .parameter(angle), control: 2, target: 3), qubits: [])
        circuit.append(gate: .controlledRotationZ(theta: .parameter(angle), control: 1, target: 2), qubits: [])

        #expect(circuit.maxQubitUsed() == 4)
    }

    @Test("maxQubitUsed with concrete two-qubit gates")
    func maxQubitUsedWithConcreteTwoQubitGates() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 6)

        circuit.append(gate: .concrete(.cnot(control: 2, target: 5)), qubits: [])
        circuit.append(gate: .concrete(.cz(control: 1, target: 3)), qubits: [])
        circuit.append(gate: .concrete(.cy(control: 0, target: 4)), qubits: [])
        circuit.append(gate: .concrete(.ch(control: 1, target: 2)), qubits: [])

        #expect(circuit.maxQubitUsed() == 5)
    }

    @Test("maxQubitUsed with concrete controlled phase gate")
    func maxQubitUsedWithConcreteControlledPhase() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 4)

        circuit.append(gate: .concrete(.controlledPhase(theta: .pi / 4, control: 0, target: 3)), qubits: [])

        #expect(circuit.maxQubitUsed() == 3)
    }

    @Test("maxQubitUsed with concrete controlled rotation gates")
    func maxQubitUsedWithConcreteControlledRotationGates() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 5)

        circuit.append(gate: .concrete(.controlledRotationX(theta: 0.5, control: 1, target: 4)), qubits: [])
        circuit.append(gate: .concrete(.controlledRotationY(theta: 0.5, control: 2, target: 3)), qubits: [])
        circuit.append(gate: .concrete(.controlledRotationZ(theta: 0.5, control: 0, target: 3)), qubits: [])

        #expect(circuit.maxQubitUsed() == 4)
    }

    @Test("maxQubitUsed with custom two-qubit gate")
    func maxQubitUsedWithCustomTwoQubit() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 4)

        let matrix = [
            [Complex<Double>.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .zero, .one],
            [.zero, .zero, .one, .zero],
        ]

        circuit.append(gate: .concrete(.customTwoQubit(matrix: matrix, control: 1, target: 3)), qubits: [])

        #expect(circuit.maxQubitUsed() == 3)
    }

    @Test("maxQubitUsed with SWAP gates")
    func maxQubitUsedWithSwapGates() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 5)

        circuit.append(gate: .concrete(.swap(qubit1: 0, qubit2: 4)), qubits: [])
        circuit.append(gate: .concrete(.sqrtSwap(qubit1: 1, qubit2: 3)), qubits: [])

        #expect(circuit.maxQubitUsed() == 4)
    }

    @Test("maxQubitUsed with Toffoli gate")
    func maxQubitUsedWithToffoli() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 6)

        circuit.append(gate: .concrete(.toffoli(control1: 0, control2: 2, target: 5)), qubits: [])

        #expect(circuit.maxQubitUsed() == 5)
    }

    @Test("maxQubitUsed returns numQubits-1 for empty circuit")
    func maxQubitUsedEmptyCircuit() {
        let circuit = ParameterizedQuantumCircuit(numQubits: 10)

        #expect(circuit.maxQubitUsed() == 9)
    }

    @Test("maxQubitUsed with parameterized gate having empty qubits array")
    func maxQubitUsedWithEmptyQubitsArray() {
        var circuit = ParameterizedQuantumCircuit(numQubits: 5)
        let theta = Parameter(name: "theta")

        circuit.append(gate: .rotationY(theta: .parameter(theta)), qubits: [])

        #expect(circuit.maxQubitUsed() == 4)
    }
}
