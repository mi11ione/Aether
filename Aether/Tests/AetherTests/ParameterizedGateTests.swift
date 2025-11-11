// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for ParameterizedGate properties.
/// Validates qubit requirements, parameterization detection,
/// and parameter extraction for all gate types.
@Suite("ParameterizedGate Properties")
struct ParameterizedGatePropertiesTests {
    @Test("Single-qubit parameterized gates require 1 qubit")
    func singleQubitGatesRequireOne() {
        let theta = ParameterExpression.value(0.5)

        #expect(ParameterizedGate.rotationX(theta: theta).qubitsRequired == 1)
        #expect(ParameterizedGate.rotationY(theta: theta).qubitsRequired == 1)
        #expect(ParameterizedGate.rotationZ(theta: theta).qubitsRequired == 1)
        #expect(ParameterizedGate.phase(theta: theta).qubitsRequired == 1)
        #expect(ParameterizedGate.u1(lambda: theta).qubitsRequired == 1)
        #expect(ParameterizedGate.u2(phi: theta, lambda: theta).qubitsRequired == 1)
        #expect(ParameterizedGate.u3(theta: theta, phi: theta, lambda: theta).qubitsRequired == 1)
    }

    @Test("Two-qubit parameterized gates require 2 qubits")
    func twoQubitGatesRequireTwo() {
        let theta = ParameterExpression.value(0.5)

        #expect(ParameterizedGate.controlledPhase(theta: theta, control: 0, target: 1).qubitsRequired == 2)
        #expect(ParameterizedGate.controlledRotationX(theta: theta, control: 0, target: 1).qubitsRequired == 2)
        #expect(ParameterizedGate.controlledRotationY(theta: theta, control: 0, target: 1).qubitsRequired == 2)
        #expect(ParameterizedGate.controlledRotationZ(theta: theta, control: 0, target: 1).qubitsRequired == 2)
    }

    @Test("Concrete gates have correct qubit requirements")
    func concreteGatesQubitRequirements() {
        let hadamard = ParameterizedGate.concrete(.hadamard)
        let cnot = ParameterizedGate.concrete(.cnot(control: 0, target: 1))
        let toffoli = ParameterizedGate.concrete(.toffoli(control1: 0, control2: 1, target: 2))

        #expect(hadamard.qubitsRequired == 1)
        #expect(cnot.qubitsRequired == 2)
        #expect(toffoli.qubitsRequired == 3)
    }

    @Test("Symbolic parameterized gates are parameterized")
    func symbolicGatesAreParameterized() {
        let param = Parameter(name: "theta")
        let expr = ParameterExpression.parameter(param)

        let gate = ParameterizedGate.rotationY(theta: expr)
        #expect(gate.isParameterized)
    }

    @Test("Concrete value gates are not parameterized")
    func concreteValueGatesNotParameterized() {
        let expr = ParameterExpression.value(0.5)
        let gate = ParameterizedGate.rotationY(theta: expr)

        #expect(!gate.isParameterized)
    }

    @Test("Concrete wrapped gates are not parameterized")
    func concreteWrappedGatesNotParameterized() {
        let gate = ParameterizedGate.concrete(.hadamard)
        #expect(!gate.isParameterized)
    }

    @Test("Extract parameters from single-parameter gates")
    func extractParametersFromSingleParameterGates() {
        let param = Parameter(name: "theta")
        let expr = ParameterExpression.parameter(param)

        let gates = [
            ParameterizedGate.rotationX(theta: expr),
            ParameterizedGate.rotationY(theta: expr),
            ParameterizedGate.rotationZ(theta: expr),
            ParameterizedGate.phase(theta: expr),
            ParameterizedGate.u1(lambda: expr),
        ]

        for gate in gates {
            let params = gate.parameters
            #expect(params.count == 1)
            #expect(params.contains(param))
        }
    }

    @Test("Extract parameters from multi-parameter gates")
    func extractParametersFromMultiParameterGates() {
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")

        let u2 = ParameterizedGate.u2(
            phi: .parameter(phi),
            lambda: .parameter(lambda)
        )
        #expect(u2.parameters.count == 2)
        #expect(u2.parameters.contains(phi))
        #expect(u2.parameters.contains(lambda))

        let u3 = ParameterizedGate.u3(
            theta: .parameter(theta),
            phi: .parameter(phi),
            lambda: .parameter(lambda)
        )
        #expect(u3.parameters.count == 3)
        #expect(u3.parameters.contains(theta))
        #expect(u3.parameters.contains(phi))
        #expect(u3.parameters.contains(lambda))
    }

    @Test("Extract parameters from controlled gates")
    func extractParametersFromControlledGates() {
        let param = Parameter(name: "theta")
        let expr = ParameterExpression.parameter(param)

        let gates = [
            ParameterizedGate.controlledPhase(theta: expr, control: 0, target: 1),
            ParameterizedGate.controlledRotationX(theta: expr, control: 0, target: 1),
            ParameterizedGate.controlledRotationY(theta: expr, control: 0, target: 1),
            ParameterizedGate.controlledRotationZ(theta: expr, control: 0, target: 1),
        ]

        for gate in gates {
            let params = gate.parameters
            #expect(params.count == 1)
            #expect(params.contains(param))
        }
    }

    @Test("Concrete gates have no parameters")
    func concreteGatesNoParameters() {
        let gate = ParameterizedGate.concrete(.hadamard)
        #expect(gate.parameters.isEmpty)
    }

    @Test("Mixed symbolic and concrete parameters")
    func mixedSymbolicAndConcreteParameters() {
        let theta = Parameter(name: "theta")
        let gate = ParameterizedGate.u3(
            theta: .parameter(theta),
            phi: .value(0.5),
            lambda: .value(1.0)
        )

        let params = gate.parameters
        #expect(params.count == 1)
        #expect(params.contains(theta))
    }
}

/// Test suite for ParameterizedGate binding.
/// Validates parameter substitution and concrete gate generation
/// for all parameterized gate types.
@Suite("ParameterizedGate Binding")
struct ParameterizedGateBindingTests {
    @Test("Bind single-qubit rotation gates")
    func bindSingleQubitRotations() throws {
        let param = Parameter(name: "theta")
        let expr = ParameterExpression.parameter(param)
        let bindings = ["theta": Double.pi / 4.0]

        let rx = ParameterizedGate.rotationX(theta: expr)
        let ry = ParameterizedGate.rotationY(theta: expr)
        let rz = ParameterizedGate.rotationZ(theta: expr)

        let boundRx = try rx.bind(with: bindings)
        let boundRy = try ry.bind(with: bindings)
        let boundRz = try rz.bind(with: bindings)

        #expect(boundRx == .rotationX(theta: Double.pi / 4.0))
        #expect(boundRy == .rotationY(theta: Double.pi / 4.0))
        #expect(boundRz == .rotationZ(theta: Double.pi / 4.0))
    }

    @Test("Bind phase gates")
    func bindPhaseGates() throws {
        let param = Parameter(name: "theta")
        let bindings = ["theta": 0.5]

        let phase = ParameterizedGate.phase(theta: .parameter(param))
        let u1 = ParameterizedGate.u1(lambda: .parameter(param))

        let boundPhase = try phase.bind(with: bindings)
        let boundU1 = try u1.bind(with: bindings)

        #expect(boundPhase == .phase(theta: 0.5))
        #expect(boundU1 == .u1(lambda: 0.5))
    }

    @Test("Bind U2 gate")
    func bindU2Gate() throws {
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")
        let bindings = ["phi": 0.5, "lambda": 1.0]

        let u2 = ParameterizedGate.u2(
            phi: .parameter(phi),
            lambda: .parameter(lambda)
        )

        let bound = try u2.bind(with: bindings)
        #expect(bound == .u2(phi: 0.5, lambda: 1.0))
    }

    @Test("Bind U3 gate")
    func bindU3Gate() throws {
        let theta = Parameter(name: "theta")
        let phi = Parameter(name: "phi")
        let lambda = Parameter(name: "lambda")
        let bindings = ["theta": 0.5, "phi": 1.0, "lambda": 1.5]

        let u3 = ParameterizedGate.u3(
            theta: .parameter(theta),
            phi: .parameter(phi),
            lambda: .parameter(lambda)
        )

        let bound = try u3.bind(with: bindings)
        #expect(bound == .u3(theta: 0.5, phi: 1.0, lambda: 1.5))
    }

    @Test("Bind controlled rotation gates")
    func bindControlledRotations() throws {
        let param = Parameter(name: "theta")
        let bindings = ["theta": Double.pi / 2.0]

        let crx = ParameterizedGate.controlledRotationX(theta: .parameter(param), control: 0, target: 1)
        let cry = ParameterizedGate.controlledRotationY(theta: .parameter(param), control: 0, target: 1)
        let crz = ParameterizedGate.controlledRotationZ(theta: .parameter(param), control: 0, target: 1)

        let boundCrx = try crx.bind(with: bindings)
        let boundCry = try cry.bind(with: bindings)
        let boundCrz = try crz.bind(with: bindings)

        #expect(boundCrx == .controlledRotationX(theta: Double.pi / 2.0, control: 0, target: 1))
        #expect(boundCry == .controlledRotationY(theta: Double.pi / 2.0, control: 0, target: 1))
        #expect(boundCrz == .controlledRotationZ(theta: Double.pi / 2.0, control: 0, target: 1))
    }

    @Test("Bind controlled phase gate")
    func bindControlledPhase() throws {
        let param = Parameter(name: "theta")
        let bindings = ["theta": Double.pi]

        let cphase = ParameterizedGate.controlledPhase(theta: .parameter(param), control: 0, target: 1)
        let bound = try cphase.bind(with: bindings)

        #expect(bound == .controlledPhase(theta: Double.pi, control: 0, target: 1))
    }

    @Test("Bind concrete gate returns same gate")
    func bindConcreteGateReturnsSame() throws {
        let gate = ParameterizedGate.concrete(.hadamard)
        let bound = try gate.bind(with: [:])

        #expect(bound == .hadamard)
    }

    @Test("Bind with missing parameter throws")
    func bindWithMissingParameterThrows() {
        let param = Parameter(name: "theta")
        let gate = ParameterizedGate.rotationY(theta: .parameter(param))
        let bindings: [String: Double] = [:]

        #expect(throws: ParameterError.self) {
            try gate.bind(with: bindings)
        }
    }

    @Test("Bind with concrete value succeeds without bindings")
    func bindConcreteValueWithoutBindings() throws {
        let gate = ParameterizedGate.rotationY(theta: .value(0.5))
        let bound = try gate.bind(with: [:])

        #expect(bound == .rotationY(theta: 0.5))
    }

    @Test("Bind preserves qubit indices in controlled gates")
    func bindPreservesQubitIndices() throws {
        let param = Parameter(name: "theta")
        let gate = ParameterizedGate.controlledRotationX(theta: .parameter(param), control: 2, target: 5)
        let bindings = ["theta": 1.0]

        let bound = try gate.bind(with: bindings)
        #expect(bound == .controlledRotationX(theta: 1.0, control: 2, target: 5))
    }
}

/// Test suite for ParameterizedGate validation.
/// Validates qubit index checking for all gate types,
/// ensuring circuit correctness before execution.
@Suite("ParameterizedGate Validation")
struct ParameterizedGateValidationTests {
    @Test("Single-qubit gates always valid")
    func singleQubitGatesAlwaysValid() {
        let theta = ParameterExpression.value(0.5)
        let gates = [
            ParameterizedGate.rotationX(theta: theta),
            ParameterizedGate.rotationY(theta: theta),
            ParameterizedGate.phase(theta: theta),
        ]

        for gate in gates {
            #expect(gate.validateQubitIndices(maxAllowedQubit: 10))
            #expect(gate.validateQubitIndices(maxAllowedQubit: 0))
        }
    }

    @Test("Two-qubit gates validate qubit indices")
    func twoQubitGatesValidateIndices() {
        let theta = ParameterExpression.value(0.5)
        let gate = ParameterizedGate.controlledRotationX(theta: theta, control: 0, target: 1)

        #expect(gate.validateQubitIndices(maxAllowedQubit: 5))
        #expect(gate.validateQubitIndices(maxAllowedQubit: 1))
        #expect(!gate.validateQubitIndices(maxAllowedQubit: 0))
    }

    @Test("Two-qubit gates reject same control and target")
    func twoQubitGatesRejectSameQubit() {
        let theta = ParameterExpression.value(0.5)
        let gate = ParameterizedGate.controlledRotationY(theta: theta, control: 2, target: 2)

        #expect(!gate.validateQubitIndices(maxAllowedQubit: 10))
    }

    @Test("Two-qubit gates reject negative indices")
    func twoQubitGatesRejectNegativeIndices() {
        let theta = ParameterExpression.value(0.5)
        let gate = ParameterizedGate.controlledPhase(theta: theta, control: -1, target: 1)

        #expect(!gate.validateQubitIndices(maxAllowedQubit: 10))
    }

    @Test("Concrete gates delegate validation")
    func concreteGatesDelegateValidation() {
        let cnot = ParameterizedGate.concrete(.cnot(control: 0, target: 1))
        let invalidCnot = ParameterizedGate.concrete(.cnot(control: 5, target: 1))

        #expect(cnot.validateQubitIndices(maxAllowedQubit: 5))
        #expect(!invalidCnot.validateQubitIndices(maxAllowedQubit: 2))
    }

    @Test("All controlled gate types validate")
    func allControlledGateTypesValidate() {
        let theta = ParameterExpression.value(0.5)
        let gates = [
            ParameterizedGate.controlledPhase(theta: theta, control: 0, target: 1),
            ParameterizedGate.controlledRotationX(theta: theta, control: 0, target: 1),
            ParameterizedGate.controlledRotationY(theta: theta, control: 0, target: 1),
            ParameterizedGate.controlledRotationZ(theta: theta, control: 0, target: 1),
        ]

        for gate in gates {
            #expect(gate.validateQubitIndices(maxAllowedQubit: 5))
            #expect(!gate.validateQubitIndices(maxAllowedQubit: 0))
        }
    }
}

/// Test suite for ParameterizedGate string representation.
/// Validates description formatting for all gate types,
/// essential for debugging and circuit visualization.
@Suite("ParameterizedGate Description")
struct ParameterizedGateDescriptionTests {
    @Test("Rotation gate descriptions")
    func rotationGateDescriptions() {
        let param = Parameter(name: "theta")
        let expr = ParameterExpression.parameter(param)

        let rx = ParameterizedGate.rotationX(theta: expr)
        let ry = ParameterizedGate.rotationY(theta: expr)
        let rz = ParameterizedGate.rotationZ(theta: expr)

        #expect(rx.description.contains("Rx"))
        #expect(rx.description.contains("theta"))
        #expect(ry.description.contains("Ry"))
        #expect(rz.description.contains("Rz"))
    }

    @Test("Phase gate descriptions")
    func phaseGateDescriptions() {
        let expr = ParameterExpression.value(0.5)

        let phase = ParameterizedGate.phase(theta: expr)
        let u1 = ParameterizedGate.u1(lambda: expr)

        #expect(phase.description.contains("P"))
        #expect(u1.description.contains("U1"))
    }

    @Test("U2 gate description")
    func u2GateDescription() {
        let gate = ParameterizedGate.u2(
            phi: .value(0.5),
            lambda: .value(1.0)
        )

        #expect(gate.description.contains("U2"))
        #expect(gate.description.contains("0.500"))
        #expect(gate.description.contains("1.000"))
    }

    @Test("U3 gate description")
    func u3GateDescription() {
        let gate = ParameterizedGate.u3(
            theta: .value(0.5),
            phi: .value(1.0),
            lambda: .value(1.5)
        )

        #expect(gate.description.contains("U3"))
    }

    @Test("Controlled gate descriptions include qubit indices")
    func controlledGateDescriptionsIncludeQubits() {
        let expr = ParameterExpression.value(0.5)
        let crx = ParameterizedGate.controlledRotationX(theta: expr, control: 0, target: 1)

        #expect(crx.description.contains("CRx"))
        #expect(crx.description.contains("c:0"))
        #expect(crx.description.contains("t:1"))
    }

    @Test("Concrete gate description delegates")
    func concreteGateDescriptionDelegates() {
        let gate = ParameterizedGate.concrete(.hadamard)
        #expect(gate.description == "H")
    }
}

/// Test suite for ParameterizedGate equality and hashing.
/// Validates Equatable and Hashable conformance for all gate types,
/// essential for parameter deduplication and circuit optimization.
@Suite("ParameterizedGate Equality and Hashing")
struct ParameterizedGateEqualityTests {
    @Test("Same parameterized gates are equal")
    func sameParameterizedGatesEqual() {
        let param = Parameter(name: "theta")
        let expr = ParameterExpression.parameter(param)

        let gate1 = ParameterizedGate.rotationY(theta: expr)
        let gate2 = ParameterizedGate.rotationY(theta: expr)

        #expect(gate1 == gate2)
    }

    @Test("Different parameterized gates are not equal")
    func differentParameterizedGatesNotEqual() {
        let param = Parameter(name: "theta")
        let expr = ParameterExpression.parameter(param)

        let gate1 = ParameterizedGate.rotationY(theta: expr)
        let gate2 = ParameterizedGate.rotationX(theta: expr)

        #expect(gate1 != gate2)
    }

    @Test("Gates with different parameters are not equal")
    func gatesWithDifferentParametersNotEqual() {
        let param1 = Parameter(name: "theta")
        let param2 = Parameter(name: "phi")

        let gate1 = ParameterizedGate.rotationY(theta: .parameter(param1))
        let gate2 = ParameterizedGate.rotationY(theta: .parameter(param2))

        #expect(gate1 != gate2)
    }

    @Test("Parameterized gates are hashable")
    func parameterizedGatesHashable() {
        let param = Parameter(name: "theta")
        let gate1 = ParameterizedGate.rotationY(theta: .parameter(param))
        let gate2 = ParameterizedGate.rotationX(theta: .parameter(param))

        var set = Set<ParameterizedGate>()
        set.insert(gate1)
        set.insert(gate2)

        #expect(set.count == 2)
    }

    @Test("ParameterizedGate is Sendable")
    func parameterizedGateIsSendable() {
        let gate = ParameterizedGate.rotationY(theta: .value(0.5))
        let _: any Sendable = gate
    }

    @Test("Concrete gates are equal")
    func concreteGatesEqual() {
        let gate1 = ParameterizedGate.concrete(.hadamard)
        let gate2 = ParameterizedGate.concrete(.hadamard)

        #expect(gate1 == gate2)
    }

    @Test("Controlled gates with different qubit indices are not equal")
    func controlledGatesDifferentQubitsNotEqual() {
        let expr = ParameterExpression.value(0.5)
        let gate1 = ParameterizedGate.controlledRotationX(theta: expr, control: 0, target: 1)
        let gate2 = ParameterizedGate.controlledRotationX(theta: expr, control: 1, target: 2)

        #expect(gate1 != gate2)
    }
}
