// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Tests initialization behavior of MPSTensor.
/// Validates tensor creation with various bond dimensions and element counts.
/// Ensures elements are correctly stored and dimensions match specifications.
@Suite("MPSTensor - Initialization")
struct MPSTensorInitializationTests {
    @Test("Initialize tensor with valid dimensions and elements")
    func initializeWithValidDimensions() {
        let elements: [Complex<Double>] = [
            Complex(1, 0), Complex(0, 0),
            Complex(0, 0), Complex(1, 0),
        ]
        let tensor = MPSTensor(leftBondDimension: 1, rightBondDimension: 2, site: 0, elements: elements)

        #expect(
            tensor.leftBondDimension == 1,
            "Left bond dimension should be 1, got \(tensor.leftBondDimension)",
        )
        #expect(
            tensor.rightBondDimension == 2,
            "Right bond dimension should be 2, got \(tensor.rightBondDimension)",
        )
        #expect(
            tensor.physicalDimension == 2,
            "Physical dimension should always be 2 for qubits, got \(tensor.physicalDimension)",
        )
        #expect(
            tensor.site == 0,
            "Site index should be 0, got \(tensor.site)",
        )
        #expect(
            tensor.elements.count == 4,
            "Element count should be 4 (1*2*2), got \(tensor.elements.count)",
        )
    }

    @Test("Initialize tensor with larger bond dimensions")
    func initializeWithLargerBonds() {
        let leftBond = 4
        let rightBond = 8
        let expectedCount = leftBond * 2 * rightBond
        let elements = [Complex<Double>](repeating: .zero, count: expectedCount)
        let tensor = MPSTensor(leftBondDimension: leftBond, rightBondDimension: rightBond, site: 5, elements: elements)

        #expect(
            tensor.leftBondDimension == leftBond,
            "Left bond dimension should be \(leftBond), got \(tensor.leftBondDimension)",
        )
        #expect(
            tensor.rightBondDimension == rightBond,
            "Right bond dimension should be \(rightBond), got \(tensor.rightBondDimension)",
        )
        #expect(
            tensor.elements.count == expectedCount,
            "Element count should be \(expectedCount), got \(tensor.elements.count)",
        )
    }

    @Test("Tensor elements are correctly stored")
    func elementsCorrectlyStored() {
        let elements: [Complex<Double>] = [
            Complex(1, 2), Complex(3, 4),
        ]
        let tensor = MPSTensor(leftBondDimension: 1, rightBondDimension: 1, site: 0, elements: elements)

        #expect(
            tensor.elements[0] == Complex(1, 2),
            "First element should be (1,2), got \(tensor.elements[0])",
        )
        #expect(
            tensor.elements[1] == Complex(3, 4),
            "Second element should be (3,4), got \(tensor.elements[1])",
        )
    }
}

/// Tests ground state factory method of MPSTensor.
/// Validates creation of |0⟩ tensors at various sites.
/// Ensures correct boundary conditions and element values.
@Suite("MPSTensor - Ground State Factory")
struct MPSTensorGroundStateTests {
    @Test("Ground state at site 0 has correct elements")
    func groundStateAtSiteZero() {
        let tensor = MPSTensor.groundState(site: 0, qubits: 4, maxBondDimension: 16)

        #expect(
            tensor.leftBondDimension == 1,
            "Ground state left bond should be 1, got \(tensor.leftBondDimension)",
        )
        #expect(
            tensor.rightBondDimension == 1,
            "Ground state right bond should be 1, got \(tensor.rightBondDimension)",
        )
        #expect(
            tensor.site == 0,
            "Site should be 0, got \(tensor.site)",
        )

        let element00 = tensor[0, 0, 0]
        let element01 = tensor[0, 1, 0]

        #expect(
            abs(element00.real - 1.0) < 1e-10 && abs(element00.imaginary) < 1e-10,
            "A[0,0,0] should be 1.0 for |0⟩ state, got \(element00)",
        )
        #expect(
            abs(element01.real) < 1e-10 && abs(element01.imaginary) < 1e-10,
            "A[0,1,0] should be 0.0 for |0⟩ state, got \(element01)",
        )
    }

    @Test("Ground state at interior site has correct elements")
    func groundStateAtInteriorSite() {
        let tensor = MPSTensor.groundState(site: 2, qubits: 4, maxBondDimension: 16)

        #expect(
            tensor.site == 2,
            "Site should be 2, got \(tensor.site)",
        )

        let element00 = tensor[0, 0, 0]
        let element01 = tensor[0, 1, 0]

        #expect(
            abs(element00.real - 1.0) < 1e-10,
            "Interior ground state A[0,0,0] should be 1.0, got \(element00.real)",
        )
        #expect(
            abs(element01.real) < 1e-10,
            "Interior ground state A[0,1,0] should be 0.0, got \(element01.real)",
        )
    }

    @Test("Ground state at last site has correct boundary")
    func groundStateAtLastSite() {
        let tensor = MPSTensor.groundState(site: 3, qubits: 4, maxBondDimension: 16)

        #expect(
            tensor.rightBondDimension == 1,
            "Last site right bond should be 1 (boundary), got \(tensor.rightBondDimension)",
        )
        #expect(
            tensor.site == 3,
            "Site should be 3, got \(tensor.site)",
        )
    }

    @Test("Ground state for single qubit system")
    func groundStateSingleQubit() {
        let tensor = MPSTensor.groundState(site: 0, qubits: 1, maxBondDimension: 16)

        #expect(
            tensor.leftBondDimension == 1,
            "Single qubit left bond should be 1, got \(tensor.leftBondDimension)",
        )
        #expect(
            tensor.rightBondDimension == 1,
            "Single qubit right bond should be 1, got \(tensor.rightBondDimension)",
        )
        #expect(
            tensor.elements.count == 2,
            "Single qubit should have 2 elements, got \(tensor.elements.count)",
        )
    }
}

/// Tests basis state factory method of MPSTensor.
/// Validates creation of |0⟩ and |1⟩ tensors based on basis state encoding.
/// Ensures little-endian bit extraction produces correct local states.
@Suite("MPSTensor - Basis State Factory")
struct MPSTensorBasisStateTests {
    @Test("Basis state 0 at site 0 produces |0⟩")
    func basisStateZeroAtSiteZero() {
        let tensor = MPSTensor.basisState(0, site: 0, qubits: 4, maxBondDimension: 16)

        let element00 = tensor[0, 0, 0]
        let element01 = tensor[0, 1, 0]

        #expect(
            abs(element00.real - 1.0) < 1e-10,
            "Basis state 0 at site 0: A[0,0,0] should be 1.0, got \(element00.real)",
        )
        #expect(
            abs(element01.real) < 1e-10,
            "Basis state 0 at site 0: A[0,1,0] should be 0.0, got \(element01.real)",
        )
    }

    @Test("Basis state 1 at site 0 produces |1⟩ (little-endian)")
    func basisStateOneAtSiteZero() {
        let tensor = MPSTensor.basisState(1, site: 0, qubits: 4, maxBondDimension: 16)

        let element00 = tensor[0, 0, 0]
        let element01 = tensor[0, 1, 0]

        #expect(
            abs(element00.real) < 1e-10,
            "Basis state 1 at site 0: A[0,0,0] should be 0.0 (bit 0 is 1), got \(element00.real)",
        )
        #expect(
            abs(element01.real - 1.0) < 1e-10,
            "Basis state 1 at site 0: A[0,1,0] should be 1.0 (bit 0 is 1), got \(element01.real)",
        )
    }

    @Test("Basis state 0b1010 at site 1 produces |1⟩")
    func basisStateTenAtSiteOne() {
        let tensor = MPSTensor.basisState(0b1010, site: 1, qubits: 4, maxBondDimension: 16)

        let element00 = tensor[0, 0, 0]
        let element01 = tensor[0, 1, 0]

        #expect(
            abs(element00.real) < 1e-10,
            "Basis state 0b1010 at site 1: A[0,0,0] should be 0.0, got \(element00.real)",
        )
        #expect(
            abs(element01.real - 1.0) < 1e-10,
            "Basis state 0b1010 at site 1: A[0,1,0] should be 1.0, got \(element01.real)",
        )
    }

    @Test("Basis state 0b1010 at site 2 produces |0⟩")
    func basisStateTenAtSiteTwo() {
        let tensor = MPSTensor.basisState(0b1010, site: 2, qubits: 4, maxBondDimension: 16)

        let element00 = tensor[0, 0, 0]
        let element01 = tensor[0, 1, 0]

        #expect(
            abs(element00.real - 1.0) < 1e-10,
            "Basis state 0b1010 at site 2: A[0,0,0] should be 1.0, got \(element00.real)",
        )
        #expect(
            abs(element01.real) < 1e-10,
            "Basis state 0b1010 at site 2: A[0,1,0] should be 0.0, got \(element01.real)",
        )
    }

    @Test("Basis state 0b1010 at site 3 produces |1⟩")
    func basisStateTenAtSiteThree() {
        let tensor = MPSTensor.basisState(0b1010, site: 3, qubits: 4, maxBondDimension: 16)

        let element00 = tensor[0, 0, 0]
        let element01 = tensor[0, 1, 0]

        #expect(
            abs(element00.real) < 1e-10,
            "Basis state 0b1010 at site 3: A[0,0,0] should be 0.0, got \(element00.real)",
        )
        #expect(
            abs(element01.real - 1.0) < 1e-10,
            "Basis state 0b1010 at site 3: A[0,1,0] should be 1.0, got \(element01.real)",
        )
    }

    @Test("Basis state 0b1111 produces |1⟩ at all sites")
    func basisStateAllOnes() {
        for siteIdx in 0 ..< 4 {
            let tensor = MPSTensor.basisState(0b1111, site: siteIdx, qubits: 4, maxBondDimension: 16)

            let element00 = tensor[0, 0, 0]
            let element01 = tensor[0, 1, 0]

            #expect(
                abs(element00.real) < 1e-10,
                "Basis state 0b1111 at site \(siteIdx): A[0,0,0] should be 0.0, got \(element00.real)",
            )
            #expect(
                abs(element01.real - 1.0) < 1e-10,
                "Basis state 0b1111 at site \(siteIdx): A[0,1,0] should be 1.0, got \(element01.real)",
            )
        }
    }
}

/// Tests subscript element access of MPSTensor.
/// Validates correct retrieval using three-index notation.
/// Ensures flat indexing formula matches subscript behavior.
@Suite("MPSTensor - Subscript Access")
struct MPSTensorSubscriptTests {
    @Test("Subscript access for simple tensor")
    func subscriptSimpleTensor() {
        let elements: [Complex<Double>] = [
            Complex(1, 0), Complex(0, 0),
        ]
        let tensor = MPSTensor(leftBondDimension: 1, rightBondDimension: 1, site: 0, elements: elements)

        let val00 = tensor[0, 0, 0]
        let val01 = tensor[0, 1, 0]

        #expect(
            abs(val00.real - 1.0) < 1e-10,
            "tensor[0,0,0] should be 1.0, got \(val00.real)",
        )
        #expect(
            abs(val01.real) < 1e-10,
            "tensor[0,1,0] should be 0.0, got \(val01.real)",
        )
    }

    @Test("Subscript access with multiple right bond indices")
    func subscriptMultipleRightBond() {
        let elements: [Complex<Double>] = [
            Complex(1, 0), Complex(2, 0),
            Complex(3, 0), Complex(4, 0),
        ]
        let tensor = MPSTensor(leftBondDimension: 1, rightBondDimension: 2, site: 0, elements: elements)

        #expect(
            abs(tensor[0, 0, 0].real - 1.0) < 1e-10,
            "tensor[0,0,0] should be 1.0, got \(tensor[0, 0, 0].real)",
        )
        #expect(
            abs(tensor[0, 0, 1].real - 2.0) < 1e-10,
            "tensor[0,0,1] should be 2.0, got \(tensor[0, 0, 1].real)",
        )
        #expect(
            abs(tensor[0, 1, 0].real - 3.0) < 1e-10,
            "tensor[0,1,0] should be 3.0, got \(tensor[0, 1, 0].real)",
        )
        #expect(
            abs(tensor[0, 1, 1].real - 4.0) < 1e-10,
            "tensor[0,1,1] should be 4.0, got \(tensor[0, 1, 1].real)",
        )
    }

    @Test("Subscript access with multiple left bond indices")
    func subscriptMultipleLeftBond() {
        let elements: [Complex<Double>] = [
            Complex(1, 0), Complex(2, 0),
            Complex(3, 0), Complex(4, 0),
        ]
        let tensor = MPSTensor(leftBondDimension: 2, rightBondDimension: 1, site: 1, elements: elements)

        #expect(
            abs(tensor[0, 0, 0].real - 1.0) < 1e-10,
            "tensor[0,0,0] should be 1.0, got \(tensor[0, 0, 0].real)",
        )
        #expect(
            abs(tensor[0, 1, 0].real - 2.0) < 1e-10,
            "tensor[0,1,0] should be 2.0, got \(tensor[0, 1, 0].real)",
        )
        #expect(
            abs(tensor[1, 0, 0].real - 3.0) < 1e-10,
            "tensor[1,0,0] should be 3.0, got \(tensor[1, 0, 0].real)",
        )
        #expect(
            abs(tensor[1, 1, 0].real - 4.0) < 1e-10,
            "tensor[1,1,0] should be 4.0, got \(tensor[1, 1, 0].real)",
        )
    }

    @Test("Subscript access with complex elements")
    func subscriptComplexElements() {
        let elements: [Complex<Double>] = [
            Complex(1, 2), Complex(3, 4),
        ]
        let tensor = MPSTensor(leftBondDimension: 1, rightBondDimension: 1, site: 0, elements: elements)

        let val00 = tensor[0, 0, 0]
        let val01 = tensor[0, 1, 0]

        #expect(
            abs(val00.real - 1.0) < 1e-10 && abs(val00.imaginary - 2.0) < 1e-10,
            "tensor[0,0,0] should be (1,2), got (\(val00.real),\(val00.imaginary))",
        )
        #expect(
            abs(val01.real - 3.0) < 1e-10 && abs(val01.imaginary - 4.0) < 1e-10,
            "tensor[0,1,0] should be (3,4), got (\(val01.real),\(val01.imaginary))",
        )
    }
}

/// Tests left contraction operation of MPSTensor.
/// Validates v @ A produces correct result[i,beta] = Sum_alpha v[alpha] * A[alpha,i,beta].
/// Ensures contraction preserves complex coefficients and bond structure.
@Suite("MPSTensor - Contract Left")
struct MPSTensorContractLeftTests {
    @Test("Contract left with identity-like tensor")
    func contractLeftIdentityTensor() {
        let tensor = MPSTensor.groundState(site: 0, qubits: 4, maxBondDimension: 16)
        let leftVec: [Complex<Double>] = [.one]

        let result = tensor.contractLeft(with: leftVec)

        #expect(
            result.count == 2,
            "Contract left should produce 2 physical indices, got \(result.count)",
        )
        #expect(
            result[0].count == 1,
            "Each physical slice should have 1 right bond element, got \(result[0].count)",
        )

        #expect(
            abs(result[0][0].real - 1.0) < 1e-10,
            "result[0][0] should be 1.0 for ground state, got \(result[0][0].real)",
        )
        #expect(
            abs(result[1][0].real) < 1e-10,
            "result[1][0] should be 0.0 for ground state, got \(result[1][0].real)",
        )
    }

    @Test("Contract left with scaling vector")
    func contractLeftScalingVector() {
        let elements: [Complex<Double>] = [
            Complex(2, 0), Complex(3, 0),
        ]
        let tensor = MPSTensor(leftBondDimension: 1, rightBondDimension: 1, site: 0, elements: elements)
        let leftVec: [Complex<Double>] = [Complex(2, 0)]

        let result = tensor.contractLeft(with: leftVec)

        #expect(
            abs(result[0][0].real - 4.0) < 1e-10,
            "2 * 2 = 4, got \(result[0][0].real)",
        )
        #expect(
            abs(result[1][0].real - 6.0) < 1e-10,
            "2 * 3 = 6, got \(result[1][0].real)",
        )
    }

    @Test("Contract left with sum over left bond")
    func contractLeftSumOverBond() {
        let elements: [Complex<Double>] = [
            Complex(1, 0), Complex(2, 0),
            Complex(3, 0), Complex(4, 0),
        ]
        let tensor = MPSTensor(leftBondDimension: 2, rightBondDimension: 1, site: 1, elements: elements)
        let leftVec: [Complex<Double>] = [Complex(1, 0), Complex(1, 0)]
        let result = tensor.contractLeft(with: leftVec)

        #expect(
            abs(result[0][0].real - 4.0) < 1e-10,
            "Sum 1+3=4 for physical index 0, got \(result[0][0].real)",
        )
        #expect(
            abs(result[1][0].real - 6.0) < 1e-10,
            "Sum 2+4=6 for physical index 1, got \(result[1][0].real)",
        )
    }

    @Test("Contract left with complex coefficients")
    func contractLeftComplexCoefficients() {
        let elements: [Complex<Double>] = [
            Complex(1, 1), Complex(2, -1),
        ]
        let tensor = MPSTensor(leftBondDimension: 1, rightBondDimension: 1, site: 0, elements: elements)
        let leftVec: [Complex<Double>] = [Complex(0, 1)]
        let result = tensor.contractLeft(with: leftVec)

        #expect(
            abs(result[0][0].real - -1.0) < 1e-10 && abs(result[0][0].imaginary - 1.0) < 1e-10,
            "i*(1+i) = -1+i, got (\(result[0][0].real),\(result[0][0].imaginary))",
        )
        #expect(
            abs(result[1][0].real - 1.0) < 1e-10 && abs(result[1][0].imaginary - 2.0) < 1e-10,
            "i*(2-i) = 1+2i, got (\(result[1][0].real),\(result[1][0].imaginary))",
        )
    }

    @Test("Contract left preserves multiple right bond indices")
    func contractLeftMultipleRightBond() {
        let elements: [Complex<Double>] = [
            Complex(1, 0), Complex(2, 0),
            Complex(3, 0), Complex(4, 0),
        ]
        let tensor = MPSTensor(leftBondDimension: 1, rightBondDimension: 2, site: 0, elements: elements)
        let leftVec: [Complex<Double>] = [Complex(2, 0)]

        let result = tensor.contractLeft(with: leftVec)

        #expect(
            result[0].count == 2,
            "Should have 2 right bond indices, got \(result[0].count)",
        )
        #expect(
            abs(result[0][0].real - 2.0) < 1e-10,
            "2*1=2, got \(result[0][0].real)",
        )
        #expect(
            abs(result[0][1].real - 4.0) < 1e-10,
            "2*2=4, got \(result[0][1].real)",
        )
        #expect(
            abs(result[1][0].real - 6.0) < 1e-10,
            "2*3=6, got \(result[1][0].real)",
        )
        #expect(
            abs(result[1][1].real - 8.0) < 1e-10,
            "2*4=8, got \(result[1][1].real)",
        )
    }
}

/// Tests right contraction operation of MPSTensor.
/// Validates A @ v produces correct result[alpha,i] = Sum_beta A[alpha,i,beta] * v[beta].
/// Ensures contraction preserves complex coefficients and bond structure.
@Suite("MPSTensor - Contract Right")
struct MPSTensorContractRightTests {
    @Test("Contract right with identity-like tensor")
    func contractRightIdentityTensor() {
        let tensor = MPSTensor.groundState(site: 0, qubits: 4, maxBondDimension: 16)
        let rightVec: [Complex<Double>] = [.one]

        let result = tensor.contractRight(with: rightVec)

        #expect(
            result.count == 1,
            "Contract right should produce 1 left bond index, got \(result.count)",
        )
        #expect(
            result[0].count == 2,
            "Each left bond slice should have 2 physical elements, got \(result[0].count)",
        )

        #expect(
            abs(result[0][0].real - 1.0) < 1e-10,
            "result[0][0] should be 1.0 for ground state, got \(result[0][0].real)",
        )
        #expect(
            abs(result[0][1].real) < 1e-10,
            "result[0][1] should be 0.0 for ground state, got \(result[0][1].real)",
        )
    }

    @Test("Contract right with scaling vector")
    func contractRightScalingVector() {
        let elements: [Complex<Double>] = [
            Complex(2, 0), Complex(3, 0),
        ]
        let tensor = MPSTensor(leftBondDimension: 1, rightBondDimension: 1, site: 0, elements: elements)
        let rightVec: [Complex<Double>] = [Complex(2, 0)]

        let result = tensor.contractRight(with: rightVec)

        #expect(
            abs(result[0][0].real - 4.0) < 1e-10,
            "2 * 2 = 4, got \(result[0][0].real)",
        )
        #expect(
            abs(result[0][1].real - 6.0) < 1e-10,
            "3 * 2 = 6, got \(result[0][1].real)",
        )
    }

    @Test("Contract right with sum over right bond")
    func contractRightSumOverBond() {
        let elements: [Complex<Double>] = [
            Complex(1, 0), Complex(2, 0),
            Complex(3, 0), Complex(4, 0),
        ]
        let tensor = MPSTensor(leftBondDimension: 1, rightBondDimension: 2, site: 0, elements: elements)
        let rightVec: [Complex<Double>] = [Complex(1, 0), Complex(1, 0)]
        let result = tensor.contractRight(with: rightVec)

        #expect(
            abs(result[0][0].real - 3.0) < 1e-10,
            "Sum 1+2=3 for physical index 0, got \(result[0][0].real)",
        )
        #expect(
            abs(result[0][1].real - 7.0) < 1e-10,
            "Sum 3+4=7 for physical index 1, got \(result[0][1].real)",
        )
    }

    @Test("Contract right with complex coefficients")
    func contractRightComplexCoefficients() {
        let elements: [Complex<Double>] = [
            Complex(1, 1), Complex(2, -1),
        ]
        let tensor = MPSTensor(leftBondDimension: 1, rightBondDimension: 1, site: 0, elements: elements)
        let rightVec: [Complex<Double>] = [Complex(0, 1)]
        let result = tensor.contractRight(with: rightVec)

        #expect(
            abs(result[0][0].real - -1.0) < 1e-10 && abs(result[0][0].imaginary - 1.0) < 1e-10,
            "(1+i)*i = -1+i, got (\(result[0][0].real),\(result[0][0].imaginary))",
        )
        #expect(
            abs(result[0][1].real - 1.0) < 1e-10 && abs(result[0][1].imaginary - 2.0) < 1e-10,
            "(2-i)*i = 1+2i, got (\(result[0][1].real),\(result[0][1].imaginary))",
        )
    }

    @Test("Contract right preserves multiple left bond indices")
    func contractRightMultipleLeftBond() {
        let elements: [Complex<Double>] = [
            Complex(1, 0), Complex(2, 0),
            Complex(3, 0), Complex(4, 0),
        ]
        let tensor = MPSTensor(leftBondDimension: 2, rightBondDimension: 1, site: 1, elements: elements)
        let rightVec: [Complex<Double>] = [Complex(2, 0)]

        let result = tensor.contractRight(with: rightVec)

        #expect(
            result.count == 2,
            "Should have 2 left bond indices, got \(result.count)",
        )
        #expect(
            abs(result[0][0].real - 2.0) < 1e-10,
            "1*2=2, got \(result[0][0].real)",
        )
        #expect(
            abs(result[0][1].real - 4.0) < 1e-10,
            "2*2=4, got \(result[0][1].real)",
        )
        #expect(
            abs(result[1][0].real - 6.0) < 1e-10,
            "3*2=6, got \(result[1][0].real)",
        )
        #expect(
            abs(result[1][1].real - 8.0) < 1e-10,
            "4*2=8, got \(result[1][1].real)",
        )
    }
}

/// Tests slice extraction by physical index of MPSTensor.
/// Validates correct matrix A[alpha,beta] for fixed physical index i.
/// Ensures complex values are preserved in extracted slices.
@Suite("MPSTensor - Matrix For Physical Index")
struct MPSTensorMatrixSliceTests {
    @Test("Matrix for physical index 0 from ground state")
    func matrixPhysicalZeroGroundState() {
        let tensor = MPSTensor.groundState(site: 0, qubits: 4, maxBondDimension: 16)

        let matrix0 = tensor.matrixForPhysicalIndex(0)
        let matrix1 = tensor.matrixForPhysicalIndex(1)

        #expect(
            matrix0.count == 1,
            "Matrix should have 1 row (leftBond=1), got \(matrix0.count)",
        )
        #expect(
            matrix0[0].count == 1,
            "Matrix should have 1 column (rightBond=1), got \(matrix0[0].count)",
        )

        #expect(
            abs(matrix0[0][0].real - 1.0) < 1e-10,
            "Ground state matrix[0] should be [[1]], got \(matrix0[0][0].real)",
        )
        #expect(
            abs(matrix1[0][0].real) < 1e-10,
            "Ground state matrix[1] should be [[0]], got \(matrix1[0][0].real)",
        )
    }

    @Test("Matrix for physical index with larger bond dimensions")
    func matrixPhysicalLargerBonds() {
        var elements = [Complex<Double>](repeating: .zero, count: 12)
        for i in 0 ..< 12 {
            elements[i] = Complex(Double(i), 0)
        }
        let tensor = MPSTensor(leftBondDimension: 2, rightBondDimension: 3, site: 1, elements: elements)

        let matrix0 = tensor.matrixForPhysicalIndex(0)
        let matrix1 = tensor.matrixForPhysicalIndex(1)

        #expect(
            matrix0.count == 2,
            "Matrix should have 2 rows, got \(matrix0.count)",
        )
        #expect(
            matrix0[0].count == 3,
            "Matrix should have 3 columns, got \(matrix0[0].count)",
        )
        #expect(
            abs(matrix0[0][0].real - 0.0) < 1e-10,
            "matrix0[0][0] should be 0, got \(matrix0[0][0].real)",
        )
        #expect(
            abs(matrix0[0][2].real - 2.0) < 1e-10,
            "matrix0[0][2] should be 2, got \(matrix0[0][2].real)",
        )
        #expect(
            abs(matrix0[1][0].real - 6.0) < 1e-10,
            "matrix0[1][0] should be 6, got \(matrix0[1][0].real)",
        )
        #expect(
            abs(matrix1[0][0].real - 3.0) < 1e-10,
            "matrix1[0][0] should be 3, got \(matrix1[0][0].real)",
        )
        #expect(
            abs(matrix1[1][2].real - 11.0) < 1e-10,
            "matrix1[1][2] should be 11, got \(matrix1[1][2].real)",
        )
    }

    @Test("Matrix slice preserves complex values")
    func matrixSliceComplexValues() {
        let elements: [Complex<Double>] = [
            Complex(1, 2), Complex(3, 4),
        ]
        let tensor = MPSTensor(leftBondDimension: 1, rightBondDimension: 1, site: 0, elements: elements)

        let matrix0 = tensor.matrixForPhysicalIndex(0)
        let matrix1 = tensor.matrixForPhysicalIndex(1)

        #expect(
            abs(matrix0[0][0].real - 1.0) < 1e-10 && abs(matrix0[0][0].imaginary - 2.0) < 1e-10,
            "matrix0[0][0] should be (1,2), got (\(matrix0[0][0].real),\(matrix0[0][0].imaginary))",
        )
        #expect(
            abs(matrix1[0][0].real - 3.0) < 1e-10 && abs(matrix1[0][0].imaginary - 4.0) < 1e-10,
            "matrix1[0][0] should be (3,4), got (\(matrix1[0][0].real),\(matrix1[0][0].imaginary))",
        )
    }
}

/// Tests SVD reshaping operation of MPSTensor.
/// Validates correct matrix reshaping for both merge directions.
/// Ensures element ordering and total count are preserved after reshape.
@Suite("MPSTensor - Reshape For SVD")
struct MPSTensorReshapeTests {
    @Test("Reshape mergeLeft=true produces (leftBond*2) x rightBond matrix")
    func reshapeMergeLeftDimensions() {
        let elements: [Complex<Double>] = [
            Complex(1, 0), Complex(2, 0), Complex(3, 0),
            Complex(4, 0), Complex(5, 0), Complex(6, 0),
            Complex(7, 0), Complex(8, 0), Complex(9, 0),
            Complex(10, 0), Complex(11, 0), Complex(12, 0),
        ]
        let tensor = MPSTensor(leftBondDimension: 2, rightBondDimension: 3, site: 1, elements: elements)

        let matrix = tensor.reshapeForSVD(mergeLeft: true)

        #expect(
            matrix.count == 4,
            "mergeLeft=true should produce 4 rows (2*2), got \(matrix.count)",
        )
        #expect(
            matrix[0].count == 3,
            "mergeLeft=true should produce 3 columns, got \(matrix[0].count)",
        )
    }

    @Test("Reshape mergeLeft=false produces leftBond x (2*rightBond) matrix")
    func reshapeMergeRightDimensions() {
        let elements: [Complex<Double>] = [
            Complex(1, 0), Complex(2, 0), Complex(3, 0),
            Complex(4, 0), Complex(5, 0), Complex(6, 0),
            Complex(7, 0), Complex(8, 0), Complex(9, 0),
            Complex(10, 0), Complex(11, 0), Complex(12, 0),
        ]
        let tensor = MPSTensor(leftBondDimension: 2, rightBondDimension: 3, site: 1, elements: elements)

        let matrix = tensor.reshapeForSVD(mergeLeft: false)

        #expect(
            matrix.count == 2,
            "mergeLeft=false should produce 2 rows, got \(matrix.count)",
        )
        #expect(
            matrix[0].count == 6,
            "mergeLeft=false should produce 6 columns (2*3), got \(matrix[0].count)",
        )
    }

    @Test("Reshape mergeLeft=true element ordering")
    func reshapeMergeLeftOrdering() {
        let elements: [Complex<Double>] = [
            Complex(0, 0), Complex(1, 0), Complex(2, 0), Complex(3, 0),
            Complex(4, 0), Complex(5, 0), Complex(6, 0), Complex(7, 0),
        ]
        let tensor = MPSTensor(leftBondDimension: 2, rightBondDimension: 2, site: 1, elements: elements)
        let matrix = tensor.reshapeForSVD(mergeLeft: true)

        #expect(
            abs(matrix[0][0].real - 0.0) < 1e-10,
            "matrix[0][0] should be 0, got \(matrix[0][0].real)",
        )
        #expect(
            abs(matrix[0][1].real - 1.0) < 1e-10,
            "matrix[0][1] should be 1, got \(matrix[0][1].real)",
        )
        #expect(
            abs(matrix[1][0].real - 2.0) < 1e-10,
            "matrix[1][0] should be 2, got \(matrix[1][0].real)",
        )
        #expect(
            abs(matrix[2][0].real - 4.0) < 1e-10,
            "matrix[2][0] should be 4, got \(matrix[2][0].real)",
        )
        #expect(
            abs(matrix[3][1].real - 7.0) < 1e-10,
            "matrix[3][1] should be 7, got \(matrix[3][1].real)",
        )
    }

    @Test("Reshape mergeLeft=false element ordering")
    func reshapeMergeRightOrdering() {
        let elements: [Complex<Double>] = [
            Complex(0, 0), Complex(1, 0), Complex(2, 0), Complex(3, 0),
            Complex(4, 0), Complex(5, 0), Complex(6, 0), Complex(7, 0),
        ]
        let tensor = MPSTensor(leftBondDimension: 2, rightBondDimension: 2, site: 1, elements: elements)
        let matrix = tensor.reshapeForSVD(mergeLeft: false)

        #expect(
            abs(matrix[0][0].real - 0.0) < 1e-10,
            "matrix[0][0] should be 0, got \(matrix[0][0].real)",
        )
        #expect(
            abs(matrix[0][1].real - 1.0) < 1e-10,
            "matrix[0][1] should be 1, got \(matrix[0][1].real)",
        )
        #expect(
            abs(matrix[0][2].real - 2.0) < 1e-10,
            "matrix[0][2] should be 2, got \(matrix[0][2].real)",
        )
        #expect(
            abs(matrix[0][3].real - 3.0) < 1e-10,
            "matrix[0][3] should be 3, got \(matrix[0][3].real)",
        )
        #expect(
            abs(matrix[1][0].real - 4.0) < 1e-10,
            "matrix[1][0] should be 4, got \(matrix[1][0].real)",
        )
        #expect(
            abs(matrix[1][3].real - 7.0) < 1e-10,
            "matrix[1][3] should be 7, got \(matrix[1][3].real)",
        )
    }

    @Test("Reshape preserves all elements")
    func reshapePreservesAllElements() {
        let elements: [Complex<Double>] = [
            Complex(1, 1), Complex(2, 2), Complex(3, 3), Complex(4, 4),
        ]
        let tensor = MPSTensor(leftBondDimension: 1, rightBondDimension: 2, site: 0, elements: elements)

        let matrixLeft = tensor.reshapeForSVD(mergeLeft: true)
        let matrixRight = tensor.reshapeForSVD(mergeLeft: false)

        var countLeft = 0
        for row in matrixLeft {
            countLeft += row.count
        }
        var countRight = 0
        for row in matrixRight {
            countRight += row.count
        }

        #expect(
            countLeft == 4,
            "mergeLeft=true should preserve 4 elements, got \(countLeft)",
        )
        #expect(
            countRight == 4,
            "mergeLeft=false should preserve 4 elements, got \(countRight)",
        )
    }
}

/// Tests boundary conditions and edge cases of MPSTensor.
/// Validates correct behavior for boundary tensors and minimal configurations.
/// Ensures operations work correctly at system boundaries.
@Suite("MPSTensor - Edge Cases")
struct MPSTensorEdgeCasesTests {
    @Test("Left boundary tensor (leftBond=1)")
    func leftBoundaryTensor() {
        let tensor = MPSTensor.groundState(site: 0, qubits: 4, maxBondDimension: 16)

        #expect(
            tensor.leftBondDimension == 1,
            "First site must have leftBond=1 (boundary), got \(tensor.leftBondDimension)",
        )

        let contractResult = tensor.contractLeft(with: [.one])
        #expect(
            contractResult.count == 2,
            "Contract left on boundary tensor should work normally, got \(contractResult.count) physical indices",
        )
    }

    @Test("Right boundary tensor (rightBond=1)")
    func rightBoundaryTensor() {
        let tensor = MPSTensor.groundState(site: 3, qubits: 4, maxBondDimension: 16)

        #expect(
            tensor.rightBondDimension == 1,
            "Last site must have rightBond=1 (boundary), got \(tensor.rightBondDimension)",
        )

        let contractResult = tensor.contractRight(with: [.one])
        #expect(
            contractResult.count == 1,
            "Contract right on boundary tensor should work normally, got \(contractResult.count) left bond indices",
        )
    }

    @Test("Single qubit system (both boundaries)")
    func singleQubitBothBoundaries() {
        let tensor = MPSTensor.groundState(site: 0, qubits: 1, maxBondDimension: 16)

        #expect(
            tensor.leftBondDimension == 1,
            "Single qubit leftBond should be 1, got \(tensor.leftBondDimension)",
        )
        #expect(
            tensor.rightBondDimension == 1,
            "Single qubit rightBond should be 1, got \(tensor.rightBondDimension)",
        )
        #expect(
            tensor.elements.count == 2,
            "Single qubit should have 2 elements, got \(tensor.elements.count)",
        )
    }

    @Test("Minimal tensor (1x1 bonds)")
    func minimalTensor() {
        let elements: [Complex<Double>] = [.one, .zero]
        let tensor = MPSTensor(leftBondDimension: 1, rightBondDimension: 1, site: 0, elements: elements)

        let matrix = tensor.reshapeForSVD(mergeLeft: true)
        #expect(
            matrix.count == 2,
            "Minimal tensor mergeLeft should have 2 rows, got \(matrix.count)",
        )
        #expect(
            matrix[0].count == 1,
            "Minimal tensor mergeLeft should have 1 column, got \(matrix[0].count)",
        )
    }

    @Test("Large bond dimension tensor")
    func largeBondDimensionTensor() {
        let leftBond = 16
        let rightBond = 16
        let elementCount = leftBond * 2 * rightBond
        var elements = [Complex<Double>](repeating: .zero, count: elementCount)
        elements[0] = .one

        let tensor = MPSTensor(leftBondDimension: leftBond, rightBondDimension: rightBond, site: 5, elements: elements)

        #expect(
            tensor.elements.count == elementCount,
            "Large tensor should have \(elementCount) elements, got \(tensor.elements.count)",
        )

        let matrixLeft = tensor.reshapeForSVD(mergeLeft: true)
        #expect(
            matrixLeft.count == leftBond * 2,
            "mergeLeft should produce \(leftBond * 2) rows, got \(matrixLeft.count)",
        )
        #expect(
            matrixLeft[0].count == rightBond,
            "mergeLeft should produce \(rightBond) columns, got \(matrixLeft[0].count)",
        )
    }

    @Test("Equatable conformance")
    func equatableConformance() {
        let elements1: [Complex<Double>] = [.one, .zero]
        let elements2: [Complex<Double>] = [.one, .zero]
        let elements3: [Complex<Double>] = [.zero, .one]

        let tensor1 = MPSTensor(leftBondDimension: 1, rightBondDimension: 1, site: 0, elements: elements1)
        let tensor2 = MPSTensor(leftBondDimension: 1, rightBondDimension: 1, site: 0, elements: elements2)
        let tensor3 = MPSTensor(leftBondDimension: 1, rightBondDimension: 1, site: 0, elements: elements3)

        #expect(
            tensor1 == tensor2,
            "Tensors with identical elements should be equal",
        )
        #expect(
            tensor1 != tensor3,
            "Tensors with different elements should not be equal",
        )
    }

    @Test("Contract operations return correct types")
    func contractReturnTypes() {
        let tensor = MPSTensor.groundState(site: 1, qubits: 4, maxBondDimension: 16)

        let leftResult = tensor.contractLeft(with: [.one])
        let rightResult = tensor.contractRight(with: [.one])

        #expect(
            leftResult.count == 2,
            "contractLeft outer dimension should be physical (2), got \(leftResult.count)",
        )

        #expect(
            rightResult[0].count == 2,
            "contractRight inner dimension should be physical (2), got \(rightResult[0].count)",
        )
    }
}

/// Tests BLAS-optimized contraction paths for large tensors.
/// Validates that tensors with 64+ elements trigger BLAS optimizations.
/// Ensures correct results for both contractLeft and contractRight with large tensors.
@Suite("MPSTensor - BLAS Optimization Paths")
struct MPSTensorBLASOptimizationTests {
    @Test("Contract left with large tensor triggers BLAS path")
    func contractLeftLargeTensor() {
        let elements = (0 ..< 64).map { Complex(Double($0), 0.0) }
        let tensor = MPSTensor(leftBondDimension: 4, rightBondDimension: 8, site: 0, elements: elements)
        let leftVector = [Complex<Double>](repeating: Complex(1.0, 0.0), count: 4)
        let result = tensor.contractLeft(with: leftVector)
        #expect(result.count == 2, "Result should have 2 rows (physical dimension)")
        #expect(result[0].count == 8, "Result should have 8 columns (right bond)")
    }

    @Test("Contract right with large tensor triggers BLAS path")
    func contractRightLargeTensor() {
        let elements = (0 ..< 64).map { Complex(Double($0), 0.0) }
        let tensor = MPSTensor(leftBondDimension: 4, rightBondDimension: 8, site: 0, elements: elements)
        let rightVector = [Complex<Double>](repeating: Complex(1.0, 0.0), count: 8)
        let result = tensor.contractRight(with: rightVector)
        #expect(result.count == 4, "Result should have 4 rows (left bond)")
        #expect(result[0].count == 2, "Result should have 2 columns (physical dimension)")
    }

    @Test("Contract left large tensor produces non-zero output")
    func contractLeftLargeTensorCorrectness() {
        let leftBond = 4
        let rightBond = 16
        let physical = 2
        var elements = [Complex<Double>](repeating: .zero, count: leftBond * physical * rightBond)
        for alpha in 0 ..< leftBond {
            for i in 0 ..< physical {
                for beta in 0 ..< rightBond {
                    let idx = alpha * (physical * rightBond) + i * rightBond + beta
                    elements[idx] = Complex(Double(idx + 1), 0.0)
                }
            }
        }
        let tensor = MPSTensor(leftBondDimension: leftBond, rightBondDimension: rightBond, site: 1, elements: elements)
        let leftVec = [Complex<Double>](repeating: .one, count: leftBond)
        let result = tensor.contractLeft(with: leftVec)

        #expect(result.count == physical, "Contract left should have physical dimension rows")
        #expect(result[0].count == rightBond, "Contract left should have rightBond columns")

        var hasNonZero = false
        for i in 0 ..< physical {
            for beta in 0 ..< rightBond {
                if result[i][beta].magnitude > 1e-10 {
                    hasNonZero = true
                    break
                }
            }
        }
        #expect(hasNonZero, "Contract left with large tensor should produce non-zero result")
    }

    @Test("Contract right large tensor correctness with subscript verification")
    func contractRightLargeTensorCorrectness() {
        let leftBond = 4
        let rightBond = 16
        let physical = 2
        var elements = [Complex<Double>](repeating: .zero, count: leftBond * physical * rightBond)
        for alpha in 0 ..< leftBond {
            for i in 0 ..< physical {
                for beta in 0 ..< rightBond {
                    let idx = alpha * (physical * rightBond) + i * rightBond + beta
                    elements[idx] = Complex(Double(alpha * 100 + i * 10 + beta), 0.0)
                }
            }
        }
        let tensor = MPSTensor(leftBondDimension: leftBond, rightBondDimension: rightBond, site: 1, elements: elements)
        var rightVec = [Complex<Double>](repeating: .zero, count: rightBond)
        rightVec[0] = .one
        let result = tensor.contractRight(with: rightVec)

        for alpha in 0 ..< leftBond {
            for i in 0 ..< physical {
                let expected = tensor[alpha, i, 0]
                let diff = (result[alpha][i] - expected).magnitude
                #expect(
                    diff < 1e-10,
                    "Contract right result[\(alpha)][\(i)] should match tensor[\(alpha),\(i),0]",
                )
            }
        }
    }

    @Test("Contract left with very large tensor")
    func contractLeftVeryLargeTensor() {
        let leftBond = 8
        let rightBond = 32
        let physical = 2
        let elements = [Complex<Double>](repeating: Complex(1.0, 0.0), count: leftBond * physical * rightBond)
        let tensor = MPSTensor(leftBondDimension: leftBond, rightBondDimension: rightBond, site: 2, elements: elements)
        let leftVec = [Complex<Double>](repeating: Complex(1.0, 0.0), count: leftBond)
        let result = tensor.contractLeft(with: leftVec)

        for i in 0 ..< physical {
            for beta in 0 ..< rightBond {
                let expected = Complex(Double(leftBond), 0.0)
                let diff = (result[i][beta] - expected).magnitude
                #expect(
                    diff < 1e-10,
                    "Large tensor contract left result[\(i)][\(beta)] should be \(leftBond), got \(result[i][beta].real)",
                )
            }
        }
    }

    @Test("Contract right with very large tensor")
    func contractRightVeryLargeTensor() {
        let leftBond = 8
        let rightBond = 32
        let physical = 2
        let elements = [Complex<Double>](repeating: Complex(1.0, 0.0), count: leftBond * physical * rightBond)
        let tensor = MPSTensor(leftBondDimension: leftBond, rightBondDimension: rightBond, site: 2, elements: elements)
        let rightVec = [Complex<Double>](repeating: Complex(1.0, 0.0), count: rightBond)
        let result = tensor.contractRight(with: rightVec)

        for alpha in 0 ..< leftBond {
            for i in 0 ..< physical {
                let expected = Complex(Double(rightBond), 0.0)
                let diff = (result[alpha][i] - expected).magnitude
                #expect(
                    diff < 1e-10,
                    "Large tensor contract right result[\(alpha)][\(i)] should be \(rightBond), got \(result[alpha][i].real)",
                )
            }
        }
    }
}

/// Tests internal consistency of MPSTensor.
/// Validates that operations preserve mathematical relationships.
/// Ensures subscript, slice, and contraction operations are mutually consistent.
@Suite("MPSTensor - Consistency")
struct MPSTensorConsistencyTests {
    @Test("Ground state and basis state 0 are equivalent")
    func groundStateEqualsBasisStateZero() {
        let groundTensor = MPSTensor.groundState(site: 0, qubits: 4, maxBondDimension: 16)
        let basisTensor = MPSTensor.basisState(0, site: 0, qubits: 4, maxBondDimension: 16)

        #expect(
            groundTensor == basisTensor,
            "Ground state should equal basis state 0",
        )
    }

    @Test("Subscript access matches direct element access")
    func subscriptMatchesElements() {
        let elements: [Complex<Double>] = [
            Complex(0, 0), Complex(1, 0), Complex(2, 0), Complex(3, 0),
            Complex(4, 0), Complex(5, 0), Complex(6, 0), Complex(7, 0),
        ]
        let tensor = MPSTensor(leftBondDimension: 2, rightBondDimension: 2, site: 1, elements: elements)

        for alpha in 0 ..< 2 {
            for physical in 0 ..< 2 {
                for beta in 0 ..< 2 {
                    let flatIndex = alpha * (2 * 2) + physical * 2 + beta
                    let subscriptValue = tensor[alpha, physical, beta]
                    let directValue = tensor.elements[flatIndex]

                    #expect(
                        subscriptValue == directValue,
                        "Subscript [\(alpha),\(physical),\(beta)] should equal elements[\(flatIndex)]",
                    )
                }
            }
        }
    }

    @Test("Matrix slice matches subscript access")
    func matrixSliceMatchesSubscript() {
        let elements: [Complex<Double>] = [
            Complex(1, 0), Complex(2, 0), Complex(3, 0), Complex(4, 0),
            Complex(5, 0), Complex(6, 0), Complex(7, 0), Complex(8, 0),
        ]
        let tensor = MPSTensor(leftBondDimension: 2, rightBondDimension: 2, site: 1, elements: elements)

        for physical in 0 ..< 2 {
            let matrix = tensor.matrixForPhysicalIndex(physical)
            for alpha in 0 ..< 2 {
                for beta in 0 ..< 2 {
                    let matrixValue = matrix[alpha][beta]
                    let subscriptValue = tensor[alpha, physical, beta]

                    #expect(
                        matrixValue == subscriptValue,
                        "matrixForPhysicalIndex(\(physical))[\(alpha)][\(beta)] should match tensor[\(alpha),\(physical),\(beta)]",
                    )
                }
            }
        }
    }

    @Test("Contract left then sum equals trace-like operation")
    func contractLeftConsistency() {
        let elements: [Complex<Double>] = [
            Complex(1, 0), Complex(2, 0),
            Complex(3, 0), Complex(4, 0),
        ]
        let tensor = MPSTensor(leftBondDimension: 2, rightBondDimension: 1, site: 1, elements: elements)
        let leftVec: [Complex<Double>] = [Complex(1, 0), Complex(0, 0)]
        let result = tensor.contractLeft(with: leftVec)

        #expect(
            abs(result[0][0].real - 1.0) < 1e-10,
            "With leftVec=[1,0], result[0][0] should be 1, got \(result[0][0].real)",
        )
        #expect(
            abs(result[1][0].real - 2.0) < 1e-10,
            "With leftVec=[1,0], result[1][0] should be 2, got \(result[1][0].real)",
        )
    }

    @Test("Contract right then sum equals trace-like operation")
    func contractRightConsistency() {
        let elements: [Complex<Double>] = [
            Complex(1, 0), Complex(2, 0),
            Complex(3, 0), Complex(4, 0),
        ]
        let tensor = MPSTensor(leftBondDimension: 1, rightBondDimension: 2, site: 0, elements: elements)
        let rightVec: [Complex<Double>] = [Complex(1, 0), Complex(0, 0)]
        let result = tensor.contractRight(with: rightVec)

        #expect(
            abs(result[0][0].real - 1.0) < 1e-10,
            "With rightVec=[1,0], result[0][0] should be 1, got \(result[0][0].real)",
        )
        #expect(
            abs(result[0][1].real - 3.0) < 1e-10,
            "With rightVec=[1,0], result[0][1] should be 3, got \(result[0][1].real)",
        )
    }
}
