// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Tests leaf node creation via the .leaf() factory method.
/// Validates correct initialization of physical dimension, bond dimensions, and elements.
/// Ensures leaf nodes have empty child bond dimensions and proper element storage.
@Suite("TreeTensorNode - Leaf Node Creation")
struct TreeTensorNodeLeafCreationTests {
    @Test("Create leaf node with dimension 2")
    func createLeafNodeDimension2() {
        let elements: [Complex<Double>] = [.one, .zero]
        let node = TreeTensorNode.leaf(physicalDimension: 2, elements: elements)

        #expect(
            node.physicalDimension == 2,
            "Leaf node physical dimension should be 2, got \(String(describing: node.physicalDimension))",
        )
        #expect(
            node.childBondDimensions.isEmpty,
            "Leaf node should have empty childBondDimensions, got \(node.childBondDimensions)",
        )
        #expect(
            node.elements.count == 2,
            "Leaf node should have 2 elements, got \(node.elements.count)",
        )
    }

    @Test("Create leaf node with dimension 4")
    func createLeafNodeDimension4() {
        let elements: [Complex<Double>] = [.one, .zero, .zero, .zero]
        let node = TreeTensorNode.leaf(physicalDimension: 4, elements: elements)

        #expect(
            node.physicalDimension == 4,
            "Leaf node physical dimension should be 4, got \(String(describing: node.physicalDimension))",
        )
        #expect(
            node.childBondDimensions.isEmpty,
            "Leaf node should have empty childBondDimensions, got \(node.childBondDimensions)",
        )
        #expect(
            node.elements.count == 4,
            "Leaf node should have 4 elements, got \(node.elements.count)",
        )
    }

    @Test("Leaf node preserves complex elements")
    func leafNodePreservesComplexElements() {
        let elements: [Complex<Double>] = [Complex(0.707, 0.0), Complex(0.0, 0.707)]
        let node = TreeTensorNode.leaf(physicalDimension: 2, elements: elements)

        #expect(
            abs(node.elements[0].real - 0.707) < 1e-10 && abs(node.elements[0].imaginary) < 1e-10,
            "First element should be (0.707, 0), got (\(node.elements[0].real), \(node.elements[0].imaginary))",
        )
        #expect(
            abs(node.elements[1].real) < 1e-10 && abs(node.elements[1].imaginary - 0.707) < 1e-10,
            "Second element should be (0, 0.707), got (\(node.elements[1].real), \(node.elements[1].imaginary))",
        )
    }

    @Test("Create superposition leaf node")
    func createSuperpositionLeafNode() {
        let sqrtHalf = 1.0 / Double.squareRoot(2.0)()
        let elements: [Complex<Double>] = [Complex(sqrtHalf, 0.0), Complex(sqrtHalf, 0.0)]
        let node = TreeTensorNode.leaf(physicalDimension: 2, elements: elements)

        #expect(
            abs(node.elements[0].real - sqrtHalf) < 1e-10,
            "Superposition element 0 should be sqrt(1/2), got \(node.elements[0].real)",
        )
        #expect(
            abs(node.elements[1].real - sqrtHalf) < 1e-10,
            "Superposition element 1 should be sqrt(1/2), got \(node.elements[1].real)",
        )
    }
}

/// Tests internal node creation via the .internal() factory method.
/// Validates correct initialization of child bond dimensions and elements.
/// Ensures internal nodes have nil physical dimension and proper element storage.
@Suite("TreeTensorNode - Internal Node Creation")
struct TreeTensorNodeInternalCreationTests {
    @Test("Create internal node with 2x2 bond dimensions")
    func createInternalNode2x2() {
        let elements: [Complex<Double>] = [.one, .zero, .zero, .one]
        let node = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: elements)

        #expect(
            node.physicalDimension == nil,
            "Internal node should have nil physicalDimension, got \(String(describing: node.physicalDimension))",
        )
        #expect(
            node.childBondDimensions == [2, 2],
            "Internal node should have childBondDimensions [2, 2], got \(node.childBondDimensions)",
        )
        #expect(
            node.elements.count == 4,
            "Internal node with 2x2 bonds should have 4 elements, got \(node.elements.count)",
        )
    }

    @Test("Create internal node with 3x4 bond dimensions")
    func createInternalNode3x4() {
        let elements: [Complex<Double>] = Array(repeating: .zero, count: 12)
        let node = TreeTensorNode.internal(childBondDimensions: [3, 4], elements: elements)

        #expect(
            node.childBondDimensions == [3, 4],
            "Internal node should have childBondDimensions [3, 4], got \(node.childBondDimensions)",
        )
        #expect(
            node.elements.count == 12,
            "Internal node with 3x4 bonds should have 12 elements, got \(node.elements.count)",
        )
    }

    @Test("Create internal node with asymmetric bond dimensions")
    func createInternalNodeAsymmetric() {
        let elements: [Complex<Double>] = Array(repeating: Complex(1.0, 0.0), count: 8)
        let node = TreeTensorNode.internal(childBondDimensions: [2, 4], elements: elements)

        #expect(
            node.childBondDimensions[0] == 2,
            "First child bond dimension should be 2, got \(node.childBondDimensions[0])",
        )
        #expect(
            node.childBondDimensions[1] == 4,
            "Second child bond dimension should be 4, got \(node.childBondDimensions[1])",
        )
    }

    @Test("Internal node preserves complex elements")
    func internalNodePreservesComplexElements() {
        let elements: [Complex<Double>] = [
            Complex(1.0, 2.0), Complex(3.0, 4.0),
            Complex(5.0, 6.0), Complex(7.0, 8.0),
        ]
        let node = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: elements)

        #expect(
            abs(node.elements[0].real - 1.0) < 1e-10 && abs(node.elements[0].imaginary - 2.0) < 1e-10,
            "Element 0 should be (1, 2), got (\(node.elements[0].real), \(node.elements[0].imaginary))",
        )
        #expect(
            abs(node.elements[3].real - 7.0) < 1e-10 && abs(node.elements[3].imaginary - 8.0) < 1e-10,
            "Element 3 should be (7, 8), got (\(node.elements[3].real), \(node.elements[3].imaginary))",
        )
    }
}

/// Tests isLeaf and isInternal computed properties.
/// Validates correct identification of node types based on physicalDimension.
/// Ensures mutual exclusivity of isLeaf and isInternal properties.
@Suite("TreeTensorNode - Node Type Properties")
struct TreeTensorNodeTypePropertiesTests {
    @Test("Leaf node isLeaf returns true")
    func leafNodeIsLeafTrue() {
        let node = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])

        #expect(
            node.isLeaf,
            "Leaf node isLeaf should return true",
        )
    }

    @Test("Leaf node isInternal returns false")
    func leafNodeIsInternalFalse() {
        let node = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])

        #expect(
            !node.isInternal,
            "Leaf node isInternal should return false",
        )
    }

    @Test("Internal node isInternal returns true")
    func internalNodeIsInternalTrue() {
        let node = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.one, .zero, .zero, .one])

        #expect(
            node.isInternal,
            "Internal node isInternal should return true",
        )
    }

    @Test("Internal node isLeaf returns false")
    func internalNodeIsLeafFalse() {
        let node = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.one, .zero, .zero, .one])

        #expect(
            !node.isLeaf,
            "Internal node isLeaf should return false",
        )
    }

    @Test("Node types are mutually exclusive")
    func nodeTypesMutuallyExclusive() {
        let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        let internal1 = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.one, .zero, .zero, .one])

        #expect(
            leaf.isLeaf != leaf.isInternal,
            "isLeaf and isInternal should be mutually exclusive for leaf nodes",
        )
        #expect(
            internal1.isLeaf != internal1.isInternal,
            "isLeaf and isInternal should be mutually exclusive for internal nodes",
        )
    }
}

/// Tests elementCount computed property.
/// Validates correct count for both leaf and internal nodes.
/// Ensures elementCount matches the actual elements array count.
@Suite("TreeTensorNode - Element Count")
struct TreeTensorNodeElementCountTests {
    @Test("Leaf node elementCount equals physicalDimension")
    func leafNodeElementCount() {
        let node = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])

        #expect(
            node.elementCount == 2,
            "Leaf node with physicalDimension 2 should have elementCount 2, got \(node.elementCount)",
        )
    }

    @Test("Leaf node elementCount equals elements array count")
    func leafNodeElementCountMatchesArray() {
        let node = TreeTensorNode.leaf(physicalDimension: 4, elements: Array(repeating: .zero, count: 4))

        #expect(
            node.elementCount == node.elements.count,
            "elementCount should equal elements.count, got \(node.elementCount) vs \(node.elements.count)",
        )
    }

    @Test("Internal node elementCount equals product of bond dimensions")
    func internalNodeElementCount() {
        let node = TreeTensorNode.internal(childBondDimensions: [3, 4], elements: Array(repeating: .zero, count: 12))

        #expect(
            node.elementCount == 12,
            "Internal node with 3x4 bonds should have elementCount 12, got \(node.elementCount)",
        )
    }

    @Test("Internal node elementCount matches elements array count")
    func internalNodeElementCountMatchesArray() {
        let node = TreeTensorNode.internal(childBondDimensions: [2, 5], elements: Array(repeating: .one, count: 10))

        #expect(
            node.elementCount == node.elements.count,
            "elementCount should equal elements.count, got \(node.elementCount) vs \(node.elements.count)",
        )
    }
}

/// Tests subscript access for leaf nodes using physical index.
/// Validates correct element retrieval by physical index.
/// Ensures complex values are correctly accessed via subscript.
@Suite("TreeTensorNode - Leaf Subscript Access")
struct TreeTensorNodeLeafSubscriptTests {
    @Test("Subscript access for leaf node index 0")
    func leafSubscriptIndex0() {
        let node = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])

        let value = node[physical: 0]

        #expect(
            abs(value.real - 1.0) < 1e-10 && abs(value.imaginary) < 1e-10,
            "Leaf node[physical: 0] should be (1, 0), got (\(value.real), \(value.imaginary))",
        )
    }

    @Test("Subscript access for leaf node index 1")
    func leafSubscriptIndex1() {
        let node = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])

        let value = node[physical: 1]

        #expect(
            abs(value.real) < 1e-10 && abs(value.imaginary) < 1e-10,
            "Leaf node[physical: 1] should be (0, 0), got (\(value.real), \(value.imaginary))",
        )
    }

    @Test("Subscript access with complex values")
    func leafSubscriptComplexValues() {
        let elements: [Complex<Double>] = [Complex(1.0, 2.0), Complex(3.0, 4.0)]
        let node = TreeTensorNode.leaf(physicalDimension: 2, elements: elements)

        let val0 = node[physical: 0]
        let val1 = node[physical: 1]

        #expect(
            abs(val0.real - 1.0) < 1e-10 && abs(val0.imaginary - 2.0) < 1e-10,
            "node[physical: 0] should be (1, 2), got (\(val0.real), \(val0.imaginary))",
        )
        #expect(
            abs(val1.real - 3.0) < 1e-10 && abs(val1.imaginary - 4.0) < 1e-10,
            "node[physical: 1] should be (3, 4), got (\(val1.real), \(val1.imaginary))",
        )
    }

    @Test("Subscript access for higher physical dimension")
    func leafSubscriptHigherDimension() {
        let elements: [Complex<Double>] = [
            Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(2.0, 0.0), Complex(3.0, 0.0),
        ]
        let node = TreeTensorNode.leaf(physicalDimension: 4, elements: elements)

        for i in 0 ..< 4 {
            let value = node[physical: i]
            #expect(
                abs(value.real - Double(i)) < 1e-10,
                "node[physical: \(i)] should be \(i), got \(value.real)",
            )
        }
    }
}

/// Tests subscript access for internal nodes using child bond indices.
/// Validates correct element retrieval using row-major indexing.
/// Ensures complex values are correctly accessed via two-index subscript.
@Suite("TreeTensorNode - Internal Subscript Access")
struct TreeTensorNodeInternalSubscriptTests {
    @Test("Subscript access for internal node diagonal elements")
    func internalSubscriptDiagonal() {
        let elements: [Complex<Double>] = [.one, .zero, .zero, .one]
        let node = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: elements)

        let val00 = node[child0: 0, child1: 0]
        let val11 = node[child0: 1, child1: 1]

        #expect(
            abs(val00.real - 1.0) < 1e-10 && abs(val00.imaginary) < 1e-10,
            "node[child0: 0, child1: 0] should be (1, 0), got (\(val00.real), \(val00.imaginary))",
        )
        #expect(
            abs(val11.real - 1.0) < 1e-10 && abs(val11.imaginary) < 1e-10,
            "node[child0: 1, child1: 1] should be (1, 0), got (\(val11.real), \(val11.imaginary))",
        )
    }

    @Test("Subscript access for internal node off-diagonal elements")
    func internalSubscriptOffDiagonal() {
        let elements: [Complex<Double>] = [.one, .zero, .zero, .one]
        let node = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: elements)

        let val01 = node[child0: 0, child1: 1]
        let val10 = node[child0: 1, child1: 0]

        #expect(
            abs(val01.real) < 1e-10 && abs(val01.imaginary) < 1e-10,
            "node[child0: 0, child1: 1] should be (0, 0), got (\(val01.real), \(val01.imaginary))",
        )
        #expect(
            abs(val10.real) < 1e-10 && abs(val10.imaginary) < 1e-10,
            "node[child0: 1, child1: 0] should be (0, 0), got (\(val10.real), \(val10.imaginary))",
        )
    }

    @Test("Subscript access row-major ordering")
    func internalSubscriptRowMajorOrdering() {
        let elements: [Complex<Double>] = [
            Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(2.0, 0.0),
            Complex(3.0, 0.0), Complex(4.0, 0.0), Complex(5.0, 0.0),
        ]
        let node = TreeTensorNode.internal(childBondDimensions: [2, 3], elements: elements)

        #expect(
            abs(node[child0: 0, child1: 0].real - 0.0) < 1e-10,
            "node[child0: 0, child1: 0] should be 0, got \(node[child0: 0, child1: 0].real)",
        )
        #expect(
            abs(node[child0: 0, child1: 2].real - 2.0) < 1e-10,
            "node[child0: 0, child1: 2] should be 2, got \(node[child0: 0, child1: 2].real)",
        )
        #expect(
            abs(node[child0: 1, child1: 0].real - 3.0) < 1e-10,
            "node[child0: 1, child1: 0] should be 3, got \(node[child0: 1, child1: 0].real)",
        )
        #expect(
            abs(node[child0: 1, child1: 2].real - 5.0) < 1e-10,
            "node[child0: 1, child1: 2] should be 5, got \(node[child0: 1, child1: 2].real)",
        )
    }

    @Test("Subscript access with complex values")
    func internalSubscriptComplexValues() {
        let elements: [Complex<Double>] = [
            Complex(1.0, 2.0), Complex(3.0, 4.0),
            Complex(5.0, 6.0), Complex(7.0, 8.0),
        ]
        let node = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: elements)

        let val00 = node[child0: 0, child1: 0]
        let val11 = node[child0: 1, child1: 1]

        #expect(
            abs(val00.real - 1.0) < 1e-10 && abs(val00.imaginary - 2.0) < 1e-10,
            "node[child0: 0, child1: 0] should be (1, 2), got (\(val00.real), \(val00.imaginary))",
        )
        #expect(
            abs(val11.real - 7.0) < 1e-10 && abs(val11.imaginary - 8.0) < 1e-10,
            "node[child0: 1, child1: 1] should be (7, 8), got (\(val11.real), \(val11.imaginary))",
        )
    }
}

/// Tests Equatable conformance for TreeTensorNode.
/// Validates equality comparison for both leaf and internal nodes.
/// Ensures nodes with different properties are correctly identified as unequal.
@Suite("TreeTensorNode - Equatable Conformance")
struct TreeTensorNodeEquatableTests {
    @Test("Identical leaf nodes are equal")
    func identicalLeafNodesEqual() {
        let elements: [Complex<Double>] = [.one, .zero]
        let node1 = TreeTensorNode.leaf(physicalDimension: 2, elements: elements)
        let node2 = TreeTensorNode.leaf(physicalDimension: 2, elements: elements)

        #expect(
            node1 == node2,
            "Leaf nodes with identical elements should be equal",
        )
    }

    @Test("Leaf nodes with different elements are not equal")
    func leafNodesDifferentElementsNotEqual() {
        let node1 = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        let node2 = TreeTensorNode.leaf(physicalDimension: 2, elements: [.zero, .one])

        #expect(
            node1 != node2,
            "Leaf nodes with different elements should not be equal",
        )
    }

    @Test("Identical internal nodes are equal")
    func identicalInternalNodesEqual() {
        let elements: [Complex<Double>] = [.one, .zero, .zero, .one]
        let node1 = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: elements)
        let node2 = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: elements)

        #expect(
            node1 == node2,
            "Internal nodes with identical properties should be equal",
        )
    }

    @Test("Internal nodes with different bond dimensions are not equal")
    func internalNodesDifferentBondsNotEqual() {
        let node1 = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.one, .zero, .zero, .one])
        let node2 = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.zero, .one, .one, .zero])

        #expect(
            node1 != node2,
            "Internal nodes with different elements should not be equal",
        )
    }

    @Test("Leaf and internal nodes are not equal")
    func leafAndInternalNodesNotEqual() {
        let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        let internal1 = TreeTensorNode.internal(childBondDimensions: [1, 2], elements: [.one, .zero])

        #expect(
            leaf != internal1,
            "Leaf node and internal node should not be equal even with same elements",
        )
    }

    @Test("Nodes with different complex phases are not equal")
    func nodesWithDifferentPhasesNotEqual() {
        let elements1: [Complex<Double>] = [Complex(1.0, 0.0), Complex(0.0, 0.0)]
        let elements2: [Complex<Double>] = [Complex(0.0, 1.0), Complex(0.0, 0.0)]
        let node1 = TreeTensorNode.leaf(physicalDimension: 2, elements: elements1)
        let node2 = TreeTensorNode.leaf(physicalDimension: 2, elements: elements2)

        #expect(
            node1 != node2,
            "Nodes with different complex phases should not be equal",
        )
    }
}

/// Tests direct initializer for TreeTensorNode.
/// Validates correct construction for both leaf and internal configurations.
/// Ensures initializer produces equivalent nodes to factory methods.
@Suite("TreeTensorNode - Direct Initialization")
struct TreeTensorNodeDirectInitTests {
    @Test("Direct init for leaf node")
    func directInitLeafNode() {
        let elements: [Complex<Double>] = [.one, .zero]
        let node = TreeTensorNode(childBondDimensions: [], physicalDimension: 2, elements: elements)

        #expect(
            node.isLeaf,
            "Directly initialized node with physicalDimension should be leaf",
        )
        #expect(
            node.childBondDimensions.isEmpty,
            "Leaf node childBondDimensions should be empty, got \(node.childBondDimensions)",
        )
    }

    @Test("Direct init for internal node")
    func directInitInternalNode() {
        let elements: [Complex<Double>] = [.one, .zero, .zero, .one]
        let node = TreeTensorNode(childBondDimensions: [2, 2], physicalDimension: nil, elements: elements)

        #expect(
            node.isInternal,
            "Directly initialized node without physicalDimension should be internal",
        )
        #expect(
            node.childBondDimensions == [2, 2],
            "Internal node childBondDimensions should be [2, 2], got \(node.childBondDimensions)",
        )
    }

    @Test("Direct init equals factory method for leaf")
    func directInitEqualsFactoryLeaf() {
        let elements: [Complex<Double>] = [.one, .zero]
        let directNode = TreeTensorNode(childBondDimensions: [], physicalDimension: 2, elements: elements)
        let factoryNode = TreeTensorNode.leaf(physicalDimension: 2, elements: elements)

        #expect(
            directNode == factoryNode,
            "Direct init and factory method should produce equal leaf nodes",
        )
    }

    @Test("Direct init equals factory method for internal")
    func directInitEqualsFactoryInternal() {
        let elements: [Complex<Double>] = [.one, .zero, .zero, .one]
        let directNode = TreeTensorNode(childBondDimensions: [2, 2], physicalDimension: nil, elements: elements)
        let factoryNode = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: elements)

        #expect(
            directNode == factoryNode,
            "Direct init and factory method should produce equal internal nodes",
        )
    }
}

/// Tests consistency between subscript access and elements array.
/// Validates that subscript correctly indexes into the flattened elements.
/// Ensures row-major ordering is correctly implemented.
@Suite("TreeTensorNode - Subscript Consistency")
struct TreeTensorNodeSubscriptConsistencyTests {
    @Test("Leaf subscript matches direct element access")
    func leafSubscriptMatchesElements() {
        let elements: [Complex<Double>] = [Complex(1.0, 0.0), Complex(2.0, 0.0), Complex(3.0, 0.0)]
        let node = TreeTensorNode.leaf(physicalDimension: 3, elements: elements)

        for i in 0 ..< 3 {
            let subscriptValue = node[physical: i]
            let directValue = node.elements[i]
            #expect(
                subscriptValue == directValue,
                "node[physical: \(i)] should equal elements[\(i)]",
            )
        }
    }

    @Test("Internal subscript matches flat index formula")
    func internalSubscriptMatchesFlatIndex() {
        let dim0 = 3
        let dim1 = 4
        let elements: [Complex<Double>] = (0 ..< 12).map { Complex(Double($0), 0.0) }
        let node = TreeTensorNode.internal(childBondDimensions: [dim0, dim1], elements: elements)

        for i in 0 ..< dim0 {
            for j in 0 ..< dim1 {
                let flatIndex = i * dim1 + j
                let subscriptValue = node[child0: i, child1: j]
                let directValue = node.elements[flatIndex]
                #expect(
                    subscriptValue == directValue,
                    "node[child0: \(i), child1: \(j)] should equal elements[\(flatIndex)]",
                )
            }
        }
    }
}
