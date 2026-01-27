// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Validates CircuitOperation.reset gate access returns nil.
/// Ensures the gate property correctly distinguishes unitary gate
/// operations from non-unitary reset operations.
@Suite("Reset Gate Access")
struct ResetGateAccessTests {
    @Test("Reset operation gate property returns nil")
    func resetGateIsNil() {
        let op = CircuitOperation.reset(qubit: 0)
        #expect(op.gate == nil, "Reset operation should have no associated gate")
    }

    @Test("Gate operation gate property returns the gate")
    func gateOperationReturnsGate() {
        let op = CircuitOperation.gate(.hadamard, qubits: [0])
        #expect(op.gate == .hadamard, "Gate operation should return its wrapped QuantumGate")
    }
}

/// Validates CircuitOperation.reset timestamp extraction.
/// Ensures timestamp is correctly stored and retrieved for
/// both timed and untimed reset operations.
@Suite("Reset Timestamp Access")
struct ResetTimestampAccessTests {
    @Test("Reset with timestamp returns correct value")
    func resetWithTimestamp() {
        let op = CircuitOperation.reset(qubit: 0, timestamp: 2.5)
        #expect(op.timestamp == 2.5, "Reset timestamp should match the value provided at construction")
    }

    @Test("Reset without timestamp returns nil")
    func resetWithoutTimestamp() {
        let op = CircuitOperation.reset(qubit: 0)
        #expect(op.timestamp == nil, "Reset without explicit timestamp should return nil")
    }
}

/// Validates CircuitOperation.reset isParameterized returns false.
/// Ensures non-unitary operations correctly report they contain
/// no symbolic parameters requiring binding.
@Suite("Reset Parameterization")
struct ResetParameterizationTests {
    @Test("Reset is not parameterized")
    func resetIsNotParameterized() {
        let op = CircuitOperation.reset(qubit: 0)
        #expect(op.isParameterized == false, "Reset operations should never be parameterized")
    }

    @Test("Gate operation delegates isParameterized to underlying gate")
    func gateOperationIsParameterized() {
        let parameterized = CircuitOperation.gate(.rotationX(.parameter(Parameter(name: "theta"))), qubits: [0])
        #expect(parameterized.isParameterized, "Gate with symbolic parameter should report as parameterized")

        let concrete = CircuitOperation.gate(.hadamard, qubits: [0])
        #expect(!concrete.isParameterized, "Gate with no parameters should not report as parameterized")
    }
}

/// Validates CircuitOperation.reset unitarity reporting.
/// Ensures reset operations are correctly identified as
/// non-unitary irreversible transformations.
@Suite("Reset Unitarity")
struct ResetUnitarityTests {
    @Test("Reset is not unitary")
    func resetIsNotUnitary() {
        let op = CircuitOperation.reset(qubit: 0)
        #expect(op.isUnitary == false, "Reset operations are irreversible and should report as non-unitary")
    }

    @Test("Gate operation is unitary")
    func gateIsUnitary() {
        let op = CircuitOperation.gate(.hadamard, qubits: [0])
        #expect(op.isUnitary == true, "Gate operations should report as unitary")
    }
}

/// Validates CircuitOperation.reset bound(with:) returns self.
/// Ensures parameter binding on non-unitary operations is a
/// no-op that preserves the original operation unchanged.
@Suite("Reset Parameter Binding")
struct ResetParameterBindingTests {
    @Test("Binding parameters on reset returns equal operation")
    func resetBoundReturnsSelf() {
        let op = CircuitOperation.reset(qubit: 0)
        let bound = op.bound(with: ["x": 1.0])
        #expect(bound == op, "Reset bound(with:) should return an operation equal to the original")
    }
}

/// Validates CircuitOperation.reset description formatting.
/// Ensures human-readable string output contains expected
/// tokens for both timed and untimed reset operations.
@Suite("Reset Description")
struct ResetDescriptionTests {
    @Test("Reset description without timestamp contains reset keyword")
    func resetDescriptionContainsReset() {
        let op = CircuitOperation.reset(qubit: 0)
        let desc = op.description
        #expect(desc.contains("reset"), "Reset description should contain the word 'reset'")
        #expect(desc.contains("0"), "Reset description should contain the target qubit index")
    }

    @Test("Reset description with timestamp includes formatted time")
    func resetDescriptionWithTimestamp() {
        let op = CircuitOperation.reset(qubit: 3, timestamp: 1.5)
        let desc = op.description
        #expect(desc.contains("reset"), "Timed reset description should contain the word 'reset'")
        #expect(desc.contains("1.500"), "Timed reset description should contain formatted timestamp")
    }
}

/// Validates CircuitOperation.reset qubit access.
/// Ensures the qubits property returns a single-element array
/// containing the correct qubit index for reset operations.
@Suite("Reset Qubit Access")
struct ResetQubitAccessTests {
    @Test("Reset qubits returns single-element array")
    func resetQubits() {
        let op = CircuitOperation.reset(qubit: 0)
        #expect(op.qubits == [0], "Reset on qubit 0 should return qubits array [0]")
    }

    @Test("Reset qubits reflects construction qubit index")
    func resetQubitsReflectsIndex() {
        let op = CircuitOperation.reset(qubit: 5)
        #expect(op.qubits == [5], "Reset on qubit 5 should return qubits array [5]")
    }
}

/// Validates CircuitOperation.reset parameters returns empty set.
/// Ensures non-unitary operations correctly report no symbolic
/// parameters in their parameter extraction method.
@Suite("Reset Parameter Extraction")
struct ResetParameterExtractionTests {
    @Test("Reset parameters returns empty set")
    func resetParametersEmpty() {
        let op = CircuitOperation.reset(qubit: 0)
        let params = op.parameters()
        #expect(params.isEmpty, "Reset operations should return an empty parameter set")
    }
}

/// Validates NonUnitaryOperation equatable and hashable conformance.
/// Ensures reset enum cases compare equal to themselves and
/// produce consistent hash values for collection use.
@Suite("NonUnitaryOperation Equatable and Hashable")
struct NonUnitaryOperationEquatableHashableTests {
    @Test("NonUnitaryOperation reset equals itself")
    func resetEqualsItself() {
        let a = NonUnitaryOperation.reset
        let b = NonUnitaryOperation.reset
        #expect(a == b, "Two NonUnitaryOperation.reset values should be equal")
    }

    @Test("NonUnitaryOperation reset has consistent hash value")
    func resetHashConsistent() {
        let a = NonUnitaryOperation.reset
        let b = NonUnitaryOperation.reset
        #expect(a.hashValue == b.hashValue, "Equal NonUnitaryOperation values should produce identical hash values")
    }
}

/// Validates CircuitOperation equatable conformance for reset.
/// Ensures reset operations with matching qubit indices compare
/// equal and operations with different indices compare unequal.
@Suite("CircuitOperation Reset Equatable")
struct CircuitOperationResetEquatableTests {
    @Test("Same reset operations are equal")
    func sameResetsAreEqual() {
        let a = CircuitOperation.reset(qubit: 0)
        let b = CircuitOperation.reset(qubit: 0)
        #expect(a == b, "Reset operations on the same qubit should be equal")
    }

    @Test("Different reset operations are not equal")
    func differentResetsAreNotEqual() {
        let a = CircuitOperation.reset(qubit: 0)
        let b = CircuitOperation.reset(qubit: 1)
        #expect(a != b, "Reset operations on different qubits should not be equal")
    }
}
