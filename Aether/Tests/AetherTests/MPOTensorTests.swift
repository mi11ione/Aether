// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Tests initialization behavior of MPOTensor.
/// Validates tensor creation with various bond dimensions and element counts.
/// Ensures elements are correctly stored and dimensions match specifications.
@Suite("MPOTensor - Initialization")
struct MPOTensorInitializationTests {
    @Test("Initialize tensor with valid dimensions and elements")
    func initializeWithValidDimensions() {
        let elements: [Complex<Double>] = [
            Complex(1, 0), Complex(0, 0),
            Complex(0, 0), Complex(1, 0),
        ]
        let tensor = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements)

        #expect(
            tensor.leftBondDimension == 1,
            "Left bond dimension should be 1, got \(tensor.leftBondDimension)",
        )
        #expect(
            tensor.rightBondDimension == 1,
            "Right bond dimension should be 1, got \(tensor.rightBondDimension)",
        )
        #expect(
            tensor.physicalDimension == 2,
            "Physical dimension should always be 2 for qubits, got \(tensor.physicalDimension)",
        )
        #expect(
            tensor.elements.count == 4,
            "Element count should be 4 (1*2*2*1), got \(tensor.elements.count)",
        )
    }

    @Test("Initialize tensor with larger bond dimensions")
    func initializeWithLargerBonds() {
        let leftBond = 3
        let rightBond = 4
        let expectedCount = leftBond * 2 * 2 * rightBond
        let elements = [Complex<Double>](repeating: .zero, count: expectedCount)
        let tensor = MPOTensor(leftBondDimension: leftBond, rightBondDimension: rightBond, elements: elements)

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
            Complex(5, 6), Complex(7, 8),
        ]
        let tensor = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements)

        #expect(
            tensor.elements[0] == Complex(1, 2),
            "First element should be (1,2), got \(tensor.elements[0])",
        )
        #expect(
            tensor.elements[1] == Complex(3, 4),
            "Second element should be (3,4), got \(tensor.elements[1])",
        )
        #expect(
            tensor.elements[2] == Complex(5, 6),
            "Third element should be (5,6), got \(tensor.elements[2])",
        )
        #expect(
            tensor.elements[3] == Complex(7, 8),
            "Fourth element should be (7,8), got \(tensor.elements[3])",
        )
    }

    @Test("Initialize identity-like MPO tensor")
    func initializeIdentityTensor() {
        let elements: [Complex<Double>] = [
            .one, .zero,
            .zero, .one,
        ]
        let tensor = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements)

        #expect(
            abs(tensor.elements[0].real - 1.0) < 1e-10 && abs(tensor.elements[0].imaginary) < 1e-10,
            "Identity element [0,0,0,0] should be 1, got \(tensor.elements[0])",
        )
        #expect(
            abs(tensor.elements[3].real - 1.0) < 1e-10 && abs(tensor.elements[3].imaginary) < 1e-10,
            "Identity element [0,1,1,0] should be 1, got \(tensor.elements[3])",
        )
    }
}

/// Tests subscript access for MPOTensor four-index notation.
/// Validates correct retrieval using [left, physIn, physOut, right] indices.
/// Ensures flat indexing formula matches subscript behavior.
@Suite("MPOTensor - Subscript Access")
struct MPOTensorSubscriptTests {
    @Test("Subscript access for simple tensor")
    func subscriptSimpleTensor() {
        let elements: [Complex<Double>] = [
            Complex(1, 0), Complex(2, 0),
            Complex(3, 0), Complex(4, 0),
        ]
        let tensor = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements)

        let val0000 = tensor[0, 0, 0, 0]
        let val0010 = tensor[0, 0, 1, 0]
        let val0100 = tensor[0, 1, 0, 0]
        let val0110 = tensor[0, 1, 1, 0]

        #expect(
            abs(val0000.real - 1.0) < 1e-10,
            "tensor[0,0,0,0] should be 1.0, got \(val0000.real)",
        )
        #expect(
            abs(val0010.real - 2.0) < 1e-10,
            "tensor[0,0,1,0] should be 2.0, got \(val0010.real)",
        )
        #expect(
            abs(val0100.real - 3.0) < 1e-10,
            "tensor[0,1,0,0] should be 3.0, got \(val0100.real)",
        )
        #expect(
            abs(val0110.real - 4.0) < 1e-10,
            "tensor[0,1,1,0] should be 4.0, got \(val0110.real)",
        )
    }

    @Test("Subscript access with multiple right bond indices")
    func subscriptMultipleRightBond() {
        let elements: [Complex<Double>] = [
            Complex(1, 0), Complex(2, 0),
            Complex(3, 0), Complex(4, 0),
            Complex(5, 0), Complex(6, 0),
            Complex(7, 0), Complex(8, 0),
        ]
        let tensor = MPOTensor(leftBondDimension: 1, rightBondDimension: 2, elements: elements)

        #expect(
            abs(tensor[0, 0, 0, 0].real - 1.0) < 1e-10,
            "tensor[0,0,0,0] should be 1.0, got \(tensor[0, 0, 0, 0].real)",
        )
        #expect(
            abs(tensor[0, 0, 0, 1].real - 2.0) < 1e-10,
            "tensor[0,0,0,1] should be 2.0, got \(tensor[0, 0, 0, 1].real)",
        )
        #expect(
            abs(tensor[0, 0, 1, 0].real - 3.0) < 1e-10,
            "tensor[0,0,1,0] should be 3.0, got \(tensor[0, 0, 1, 0].real)",
        )
        #expect(
            abs(tensor[0, 0, 1, 1].real - 4.0) < 1e-10,
            "tensor[0,0,1,1] should be 4.0, got \(tensor[0, 0, 1, 1].real)",
        )
        #expect(
            abs(tensor[0, 1, 0, 0].real - 5.0) < 1e-10,
            "tensor[0,1,0,0] should be 5.0, got \(tensor[0, 1, 0, 0].real)",
        )
        #expect(
            abs(tensor[0, 1, 1, 1].real - 8.0) < 1e-10,
            "tensor[0,1,1,1] should be 8.0, got \(tensor[0, 1, 1, 1].real)",
        )
    }

    @Test("Subscript access with multiple left bond indices")
    func subscriptMultipleLeftBond() {
        let elements: [Complex<Double>] = [
            Complex(1, 0), Complex(2, 0), Complex(3, 0), Complex(4, 0),
            Complex(5, 0), Complex(6, 0), Complex(7, 0), Complex(8, 0),
        ]
        let tensor = MPOTensor(leftBondDimension: 2, rightBondDimension: 1, elements: elements)

        #expect(
            abs(tensor[0, 0, 0, 0].real - 1.0) < 1e-10,
            "tensor[0,0,0,0] should be 1.0, got \(tensor[0, 0, 0, 0].real)",
        )
        #expect(
            abs(tensor[0, 1, 1, 0].real - 4.0) < 1e-10,
            "tensor[0,1,1,0] should be 4.0, got \(tensor[0, 1, 1, 0].real)",
        )
        #expect(
            abs(tensor[1, 0, 0, 0].real - 5.0) < 1e-10,
            "tensor[1,0,0,0] should be 5.0, got \(tensor[1, 0, 0, 0].real)",
        )
        #expect(
            abs(tensor[1, 1, 1, 0].real - 8.0) < 1e-10,
            "tensor[1,1,1,0] should be 8.0, got \(tensor[1, 1, 1, 0].real)",
        )
    }

    @Test("Subscript access with complex elements")
    func subscriptComplexElements() {
        let elements: [Complex<Double>] = [
            Complex(1, 2), Complex(3, 4),
            Complex(5, 6), Complex(7, 8),
        ]
        let tensor = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements)

        let val0000 = tensor[0, 0, 0, 0]
        let val0110 = tensor[0, 1, 1, 0]

        #expect(
            abs(val0000.real - 1.0) < 1e-10 && abs(val0000.imaginary - 2.0) < 1e-10,
            "tensor[0,0,0,0] should be (1,2), got (\(val0000.real),\(val0000.imaginary))",
        )
        #expect(
            abs(val0110.real - 7.0) < 1e-10 && abs(val0110.imaginary - 8.0) < 1e-10,
            "tensor[0,1,1,0] should be (7,8), got (\(val0110.real),\(val0110.imaginary))",
        )
    }

    @Test("Subscript access with all bond dimensions greater than one")
    func subscriptAllBondsLarge() {
        let leftBond = 2
        let rightBond = 3
        let d = 2
        let count = leftBond * d * d * rightBond
        var elements = [Complex<Double>](repeating: .zero, count: count)
        for i in 0 ..< count {
            elements[i] = Complex(Double(i), 0)
        }
        let tensor = MPOTensor(leftBondDimension: leftBond, rightBondDimension: rightBond, elements: elements)

        let val0000 = tensor[0, 0, 0, 0]
        let val0001 = tensor[0, 0, 0, 1]
        let val0012 = tensor[0, 0, 1, 2]
        let val1112 = tensor[1, 1, 1, 2]

        #expect(
            abs(val0000.real - 0.0) < 1e-10,
            "tensor[0,0,0,0] should be 0.0, got \(val0000.real)",
        )
        #expect(
            abs(val0001.real - 1.0) < 1e-10,
            "tensor[0,0,0,1] should be 1.0, got \(val0001.real)",
        )
        #expect(
            abs(val0012.real - 5.0) < 1e-10,
            "tensor[0,0,1,2] should be 5.0, got \(val0012.real)",
        )
        #expect(
            abs(val1112.real - 23.0) < 1e-10,
            "tensor[1,1,1,2] should be 23.0, got \(val1112.real)",
        )
    }
}

/// Tests row-major element storage order for MPOTensor.
/// Validates flat index formula: alpha * (d * d * rightBond) + physIn * (d * rightBond) + physOut * rightBond + beta.
/// Ensures correct mapping between multi-index and flattened storage.
@Suite("MPOTensor - Row-Major Storage Order")
struct MPOTensorStorageOrderTests {
    @Test("Verify row-major indexing formula for minimal tensor")
    func verifyRowMajorMinimal() {
        let elements: [Complex<Double>] = [
            Complex(0, 0), Complex(1, 0),
            Complex(2, 0), Complex(3, 0),
        ]
        let tensor = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements)

        let d = 2
        let rightBond = 1
        for physIn in 0 ..< d {
            for physOut in 0 ..< d {
                let flatIndex = physIn * (d * rightBond) + physOut * rightBond
                let subscriptVal = tensor[0, physIn, physOut, 0]
                let directVal = elements[flatIndex]

                #expect(
                    subscriptVal == directVal,
                    "tensor[0,\(physIn),\(physOut),0] should match elements[\(flatIndex)]",
                )
            }
        }
    }

    @Test("Verify row-major indexing formula with larger bonds")
    func verifyRowMajorLargerBonds() {
        let leftBond = 2
        let rightBond = 3
        let d = 2
        let count = leftBond * d * d * rightBond
        var elements = [Complex<Double>](repeating: .zero, count: count)
        for i in 0 ..< count {
            elements[i] = Complex(Double(i), Double(-i))
        }
        let tensor = MPOTensor(leftBondDimension: leftBond, rightBondDimension: rightBond, elements: elements)

        for alpha in 0 ..< leftBond {
            for physIn in 0 ..< d {
                for physOut in 0 ..< d {
                    for beta in 0 ..< rightBond {
                        let flatIndex = alpha * (d * d * rightBond) + physIn * (d * rightBond) + physOut * rightBond + beta
                        let subscriptVal = tensor[alpha, physIn, physOut, beta]
                        let directVal = elements[flatIndex]

                        #expect(
                            subscriptVal == directVal,
                            "tensor[\(alpha),\(physIn),\(physOut),\(beta)] should match elements[\(flatIndex)]",
                        )
                    }
                }
            }
        }
    }

    @Test("Element storage order matches documented formula")
    func elementStorageMatchesDocumentation() {
        let leftBond = 2
        let rightBond = 2
        let d = 2
        let count = leftBond * d * d * rightBond
        var elements = [Complex<Double>](repeating: .zero, count: count)
        for i in 0 ..< count {
            elements[i] = Complex(Double(i + 1), 0)
        }
        let tensor = MPOTensor(leftBondDimension: leftBond, rightBondDimension: rightBond, elements: elements)

        #expect(
            abs(tensor[0, 0, 0, 0].real - 1.0) < 1e-10,
            "First element tensor[0,0,0,0] should be 1, flatIndex=0",
        )
        #expect(
            abs(tensor[0, 0, 0, 1].real - 2.0) < 1e-10,
            "tensor[0,0,0,1] should be 2, flatIndex=1",
        )
        #expect(
            abs(tensor[0, 0, 1, 0].real - 3.0) < 1e-10,
            "tensor[0,0,1,0] should be 3, flatIndex=2",
        )
        #expect(
            abs(tensor[1, 0, 0, 0].real - 9.0) < 1e-10,
            "tensor[1,0,0,0] should be 9, flatIndex=8",
        )
        #expect(
            abs(tensor[1, 1, 1, 1].real - 16.0) < 1e-10,
            "Last element tensor[1,1,1,1] should be 16, flatIndex=15",
        )
    }

    @Test("Sequential elements map to correct indices")
    func sequentialElementsMapCorrectly() {
        let leftBond = 1
        let rightBond = 2
        let d = 2
        let count = leftBond * d * d * rightBond
        var elements = [Complex<Double>](repeating: .zero, count: count)
        for i in 0 ..< count {
            elements[i] = Complex(Double(100 + i), 0)
        }
        let tensor = MPOTensor(leftBondDimension: leftBond, rightBondDimension: rightBond, elements: elements)

        var idx = 0
        for physIn in 0 ..< d {
            for physOut in 0 ..< d {
                for beta in 0 ..< rightBond {
                    let expected = Complex(Double(100 + idx), 0.0)
                    let actual = tensor[0, physIn, physOut, beta]
                    #expect(
                        actual == expected,
                        "Sequential element \(idx) at [0,\(physIn),\(physOut),\(beta)] should be \(expected)",
                    )
                    idx += 1
                }
            }
        }
    }
}

/// Tests Equatable conformance of MPOTensor.
/// Validates equality comparison between tensors with same and different elements.
/// Ensures structural equality includes all dimensions and element values.
@Suite("MPOTensor - Equatable Conformance")
struct MPOTensorEquatableTests {
    @Test("Equal tensors with identical elements")
    func equalTensorsIdenticalElements() {
        let elements1: [Complex<Double>] = [.one, .zero, .zero, .one]
        let elements2: [Complex<Double>] = [.one, .zero, .zero, .one]

        let tensor1 = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements1)
        let tensor2 = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements2)

        #expect(
            tensor1 == tensor2,
            "Tensors with identical elements should be equal",
        )
    }

    @Test("Unequal tensors with different elements")
    func unequalTensorsDifferentElements() {
        let elements1: [Complex<Double>] = [.one, .zero, .zero, .one]
        let elements2: [Complex<Double>] = [.zero, .one, .one, .zero]

        let tensor1 = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements1)
        let tensor2 = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements2)

        #expect(
            tensor1 != tensor2,
            "Tensors with different elements should not be equal",
        )
    }

    @Test("Unequal tensors with different left bond dimensions")
    func unequalTensorsDifferentLeftBond() {
        let elements1: [Complex<Double>] = [.one, .zero, .zero, .one]
        let elements2: [Complex<Double>] = [.one, .zero, .zero, .one, .zero, .zero, .zero, .zero]

        let tensor1 = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements1)
        let tensor2 = MPOTensor(leftBondDimension: 2, rightBondDimension: 1, elements: elements2)

        #expect(
            tensor1 != tensor2,
            "Tensors with different left bond dimensions should not be equal",
        )
    }

    @Test("Unequal tensors with different right bond dimensions")
    func unequalTensorsDifferentRightBond() {
        let elements1: [Complex<Double>] = [.one, .zero, .zero, .one]
        let elements2: [Complex<Double>] = [.one, .zero, .zero, .zero, .zero, .one, .zero, .zero]

        let tensor1 = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements1)
        let tensor2 = MPOTensor(leftBondDimension: 1, rightBondDimension: 2, elements: elements2)

        #expect(
            tensor1 != tensor2,
            "Tensors with different right bond dimensions should not be equal",
        )
    }

    @Test("Equal tensors with complex elements")
    func equalTensorsComplexElements() {
        let elements1: [Complex<Double>] = [
            Complex(1, 2), Complex(3, 4),
            Complex(5, 6), Complex(7, 8),
        ]
        let elements2: [Complex<Double>] = [
            Complex(1, 2), Complex(3, 4),
            Complex(5, 6), Complex(7, 8),
        ]

        let tensor1 = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements1)
        let tensor2 = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements2)

        #expect(
            tensor1 == tensor2,
            "Tensors with identical complex elements should be equal",
        )
    }

    @Test("Unequal tensors with slightly different imaginary parts")
    func unequalTensorsSlightlyDifferent() {
        let elements1: [Complex<Double>] = [
            Complex(1, 2), Complex(3, 4),
            Complex(5, 6), Complex(7, 8),
        ]
        let elements2: [Complex<Double>] = [
            Complex(1, 2), Complex(3, 4),
            Complex(5, 6), Complex(7, 8.0001),
        ]

        let tensor1 = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements1)
        let tensor2 = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements2)

        #expect(
            tensor1 != tensor2,
            "Tensors with slightly different elements should not be equal",
        )
    }
}

/// Tests bond dimension properties of MPOTensor.
/// Validates leftBondDimension and rightBondDimension accessors.
/// Ensures element count matches dimension product.
@Suite("MPOTensor - Bond Dimension Properties")
struct MPOTensorBondDimensionTests {
    @Test("Minimal bond dimensions (1,1)")
    func minimalBondDimensions() {
        let elements: [Complex<Double>] = [.one, .zero, .zero, .one]
        let tensor = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements)

        #expect(
            tensor.leftBondDimension == 1,
            "Left bond dimension should be 1, got \(tensor.leftBondDimension)",
        )
        #expect(
            tensor.rightBondDimension == 1,
            "Right bond dimension should be 1, got \(tensor.rightBondDimension)",
        )
        #expect(
            tensor.physicalDimension == 2,
            "Physical dimension should be 2, got \(tensor.physicalDimension)",
        )
    }

    @Test("Asymmetric bond dimensions (3,5)")
    func asymmetricBondDimensions() {
        let leftBond = 3
        let rightBond = 5
        let d = 2
        let count = leftBond * d * d * rightBond
        let elements = [Complex<Double>](repeating: .zero, count: count)
        let tensor = MPOTensor(leftBondDimension: leftBond, rightBondDimension: rightBond, elements: elements)

        #expect(
            tensor.leftBondDimension == leftBond,
            "Left bond dimension should be \(leftBond), got \(tensor.leftBondDimension)",
        )
        #expect(
            tensor.rightBondDimension == rightBond,
            "Right bond dimension should be \(rightBond), got \(tensor.rightBondDimension)",
        )
    }

    @Test("Element count equals dimension product")
    func elementCountMatchesDimensionProduct() {
        let leftBond = 4
        let rightBond = 6
        let d = 2
        let expectedCount = leftBond * d * d * rightBond
        let elements = [Complex<Double>](repeating: .one, count: expectedCount)
        let tensor = MPOTensor(leftBondDimension: leftBond, rightBondDimension: rightBond, elements: elements)

        let actualCount = tensor.elements.count
        let computedCount = tensor.leftBondDimension * tensor.physicalDimension * tensor.physicalDimension * tensor.rightBondDimension

        #expect(
            actualCount == expectedCount,
            "Element count should be \(expectedCount), got \(actualCount)",
        )
        #expect(
            computedCount == expectedCount,
            "Computed dimension product should match element count",
        )
    }

    @Test("Large bond dimensions")
    func largeBondDimensions() {
        let leftBond = 16
        let rightBond = 16
        let d = 2
        let expectedCount = leftBond * d * d * rightBond
        let elements = [Complex<Double>](repeating: .zero, count: expectedCount)
        let tensor = MPOTensor(leftBondDimension: leftBond, rightBondDimension: rightBond, elements: elements)

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

    @Test("Boundary MPO tensor (leftBond=1, rightBond=3)")
    func boundaryMPOTensor() {
        let leftBond = 1
        let rightBond = 3
        let d = 2
        let expectedCount = leftBond * d * d * rightBond
        let elements = [Complex<Double>](repeating: Complex(0.5, 0.5), count: expectedCount)
        let tensor = MPOTensor(leftBondDimension: leftBond, rightBondDimension: rightBond, elements: elements)

        #expect(
            tensor.leftBondDimension == 1,
            "Boundary tensor left bond should be 1, got \(tensor.leftBondDimension)",
        )
        #expect(
            tensor.rightBondDimension == rightBond,
            "Boundary tensor right bond should be \(rightBond), got \(tensor.rightBondDimension)",
        )
    }

    @Test("Physical dimension is always 2 for qubits")
    func physicalDimensionAlwaysTwo() {
        let tensor1 = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: [.one, .zero, .zero, .one])
        let tensor2 = MPOTensor(leftBondDimension: 2, rightBondDimension: 3, elements: [Complex<Double>](repeating: .zero, count: 24))
        let tensor3 = MPOTensor(leftBondDimension: 5, rightBondDimension: 5, elements: [Complex<Double>](repeating: .zero, count: 100))

        #expect(
            tensor1.physicalDimension == 2,
            "Physical dimension should be 2 for any MPO tensor, got \(tensor1.physicalDimension)",
        )
        #expect(
            tensor2.physicalDimension == 2,
            "Physical dimension should be 2 for any MPO tensor, got \(tensor2.physicalDimension)",
        )
        #expect(
            tensor3.physicalDimension == 2,
            "Physical dimension should be 2 for any MPO tensor, got \(tensor3.physicalDimension)",
        )
    }
}

/// Tests internal consistency of MPOTensor operations.
/// Validates that subscript access matches direct element access.
/// Ensures tensor structure is self-consistent.
@Suite("MPOTensor - Consistency")
struct MPOTensorConsistencyTests {
    @Test("Subscript access matches direct element access")
    func subscriptMatchesElements() {
        let leftBond = 2
        let rightBond = 2
        let d = 2
        let count = leftBond * d * d * rightBond
        var elements = [Complex<Double>](repeating: .zero, count: count)
        for i in 0 ..< count {
            elements[i] = Complex(Double(i), Double(-i))
        }
        let tensor = MPOTensor(leftBondDimension: leftBond, rightBondDimension: rightBond, elements: elements)

        for alpha in 0 ..< leftBond {
            for physIn in 0 ..< d {
                for physOut in 0 ..< d {
                    for beta in 0 ..< rightBond {
                        let flatIndex = alpha * (d * d * rightBond) + physIn * (d * rightBond) + physOut * rightBond + beta
                        let subscriptValue = tensor[alpha, physIn, physOut, beta]
                        let directValue = tensor.elements[flatIndex]

                        #expect(
                            subscriptValue == directValue,
                            "Subscript [\(alpha),\(physIn),\(physOut),\(beta)] should equal elements[\(flatIndex)]",
                        )
                    }
                }
            }
        }
    }

    @Test("Tensor preserves all complex values")
    func tensorPreservesComplexValues() {
        let elements: [Complex<Double>] = [
            Complex(1.5, -2.3), Complex(0.0, 1.0),
            Complex(-1.0, 0.0), Complex(3.14159, 2.71828),
        ]
        let tensor = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements)

        for i in 0 ..< elements.count {
            let diff = (tensor.elements[i] - elements[i]).magnitude
            #expect(
                diff < 1e-10,
                "Element \(i) should be preserved exactly, diff=\(diff)",
            )
        }
    }

    @Test("Identity-like tensor has correct diagonal structure")
    func identityTensorDiagonalStructure() {
        let elements: [Complex<Double>] = [.one, .zero, .zero, .one]
        let tensor = MPOTensor(leftBondDimension: 1, rightBondDimension: 1, elements: elements)

        #expect(
            abs(tensor[0, 0, 0, 0].real - 1.0) < 1e-10,
            "Identity tensor[0,0,0,0] should be 1",
        )
        #expect(
            abs(tensor[0, 0, 1, 0].real) < 1e-10,
            "Identity tensor[0,0,1,0] should be 0",
        )
        #expect(
            abs(tensor[0, 1, 0, 0].real) < 1e-10,
            "Identity tensor[0,1,0,0] should be 0",
        )
        #expect(
            abs(tensor[0, 1, 1, 0].real - 1.0) < 1e-10,
            "Identity tensor[0,1,1,0] should be 1",
        )
    }

    @Test("Zero tensor has all zero elements")
    func zeroTensorAllZeros() {
        let count = 2 * 2 * 2 * 3
        let elements = [Complex<Double>](repeating: .zero, count: count)
        let tensor = MPOTensor(leftBondDimension: 2, rightBondDimension: 3, elements: elements)

        for alpha in 0 ..< 2 {
            for physIn in 0 ..< 2 {
                for physOut in 0 ..< 2 {
                    for beta in 0 ..< 3 {
                        let val = tensor[alpha, physIn, physOut, beta]
                        #expect(
                            val.magnitude < 1e-10,
                            "Zero tensor[\(alpha),\(physIn),\(physOut),\(beta)] should be 0, got \(val)",
                        )
                    }
                }
            }
        }
    }
}
