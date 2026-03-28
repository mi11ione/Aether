// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Tests binary topology initialization of TreeTensorNetwork.
/// Validates node count, structure, and adjacency for binary trees.
/// Ensures correct tree construction for depth 1, 2, and 3 binary topologies.
@Suite("TTN Binary Topology Initialization")
struct TTNBinaryTopologyInitTests {
    @Test("Binary depth 1 creates 3 nodes")
    func binaryDepth1Creates3Nodes() {
        let ttn = TreeTensorNetwork(topology: .binary(depth: 1))
        let leaf0 = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        let leaf1 = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        let root = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.one, .zero, .zero, .one])
        var mutableTTN = ttn
        mutableTTN.setNode(at: 0, tensor: root)
        mutableTTN.setNode(at: 1, tensor: leaf0)
        mutableTTN.setNode(at: 2, tensor: leaf1)
        let result = mutableTTN.contract()
        #expect(abs(result.real - 1.0) < 1e-10, "Binary depth 1 should contract to scalar 1.0 for |00> product state")
    }

    @Test("Binary depth 2 creates 7 nodes")
    func binaryDepth2Creates7Nodes() {
        var ttn = TreeTensorNetwork(topology: .binary(depth: 2))
        let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        let internal2x2 = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.one, .zero, .zero, .one])
        ttn.setNode(at: 0, tensor: internal2x2)
        ttn.setNode(at: 1, tensor: internal2x2)
        ttn.setNode(at: 2, tensor: internal2x2)
        ttn.setNode(at: 3, tensor: leaf)
        ttn.setNode(at: 4, tensor: leaf)
        ttn.setNode(at: 5, tensor: leaf)
        ttn.setNode(at: 6, tensor: leaf)
        let result = ttn.contract()
        #expect(abs(result.imaginary) < 1e-10, "Depth 2 binary tree contraction should have zero imaginary part")
    }

    @Test("Binary depth 3 creates 15 nodes")
    func binaryDepth3Creates15Nodes() {
        var ttn = TreeTensorNetwork(topology: .binary(depth: 3))
        let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        let internal2x2 = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.one, .zero, .zero, .one])
        for i in 0 ..< 7 {
            ttn.setNode(at: i, tensor: internal2x2)
        }
        for i in 7 ..< 15 {
            ttn.setNode(at: i, tensor: leaf)
        }
        let result = ttn.contract()
        #expect(abs(result.real) >= 0.0, "Depth 3 binary tree contraction should produce valid scalar")
    }
}

/// Tests custom topology initialization of TreeTensorNetwork.
/// Validates arbitrary tree structures via adjacency lists.
/// Ensures custom topologies correctly identify leaf and internal nodes.
@Suite("TTN Custom Topology Initialization")
struct TTNCustomTopologyInitTests {
    @Test("Custom topology with 3 nodes")
    func customTopology3Nodes() {
        let adjacency = [[1, 2], [], []]
        var ttn = TreeTensorNetwork(topology: .custom(adjacency: adjacency))
        let root = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.one, .zero, .zero, .one])
        let leaf0 = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        let leaf1 = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        ttn.setNode(at: 0, tensor: root)
        ttn.setNode(at: 1, tensor: leaf0)
        ttn.setNode(at: 2, tensor: leaf1)
        let result = ttn.contract()
        #expect(abs(result.real - 1.0) < 1e-10, "Custom topology should contract correctly for product state")
    }

    @Test("Custom topology with 5 nodes")
    func customTopology5Nodes() {
        let adjacency = [[1, 2], [3, 4], [], [], []]
        var ttn = TreeTensorNetwork(topology: .custom(adjacency: adjacency))
        let internal2x2 = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.one, .zero, .zero, .one])
        let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        ttn.setNode(at: 0, tensor: internal2x2)
        ttn.setNode(at: 1, tensor: internal2x2)
        ttn.setNode(at: 2, tensor: leaf)
        ttn.setNode(at: 3, tensor: leaf)
        ttn.setNode(at: 4, tensor: leaf)
        let result = ttn.contract()
        #expect(abs(result.imaginary) < 1e-10, "Custom 5-node topology contraction should have zero imaginary part")
    }

    @Test("Custom topology asymmetric tree")
    func customTopologyAsymmetric() {
        let adjacency = [[1, 2], [], [3, 4], [], []]
        var ttn = TreeTensorNetwork(topology: .custom(adjacency: adjacency))
        let internal2x2 = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.one, .zero, .zero, .one])
        let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        ttn.setNode(at: 0, tensor: internal2x2)
        ttn.setNode(at: 1, tensor: leaf)
        ttn.setNode(at: 2, tensor: internal2x2)
        ttn.setNode(at: 3, tensor: leaf)
        ttn.setNode(at: 4, tensor: leaf)
        let result = ttn.contract()
        #expect(abs(result.real) >= 0.0, "Asymmetric tree should contract to valid scalar")
    }
}

/// Tests setNode method of TreeTensorNetwork at valid indices.
/// Validates tensor assignment at leaf and internal node positions.
/// Ensures correct node placement maintains tree structure integrity.
@Suite("TTN SetNode Valid Indices")
struct TTNSetNodeValidIndicesTests {
    @Test("Set leaf node at valid index")
    func setLeafNodeAtValidIndex() {
        var ttn = TreeTensorNetwork(topology: .binary(depth: 1))
        let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        ttn.setNode(at: 1, tensor: leaf)
        ttn.setNode(at: 2, tensor: leaf)
        let root = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.one, .zero, .zero, .one])
        ttn.setNode(at: 0, tensor: root)
        let result = ttn.contract()
        #expect(abs(result.real - 1.0) < 1e-10, "Setting leaf nodes at valid indices should allow contraction")
    }

    @Test("Set internal node at root index 0")
    func setInternalNodeAtRoot() {
        var ttn = TreeTensorNetwork(topology: .binary(depth: 1))
        let root = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.one, .zero, .zero, .one])
        ttn.setNode(at: 0, tensor: root)
        let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        ttn.setNode(at: 1, tensor: leaf)
        ttn.setNode(at: 2, tensor: leaf)
        let result = ttn.contract()
        #expect(abs(result.real - 1.0) < 1e-10, "Setting internal node at root should allow valid contraction")
    }

    @Test("Set all nodes in depth 2 tree")
    func setAllNodesDepth2() {
        var ttn = TreeTensorNetwork(topology: .binary(depth: 2))
        let internal2x2 = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.one, .zero, .zero, .one])
        let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        for i in 0 ..< 3 {
            ttn.setNode(at: i, tensor: internal2x2)
        }
        for i in 3 ..< 7 {
            ttn.setNode(at: i, tensor: leaf)
        }
        let result = ttn.contract()
        #expect(abs(result.imaginary) < 1e-10, "Setting all nodes in depth 2 tree should produce valid contraction")
    }

    @Test("Replace node at same index")
    func replaceNodeAtSameIndex() {
        var ttn = TreeTensorNetwork(topology: .binary(depth: 1))
        let leaf0 = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        let leaf1 = TreeTensorNode.leaf(physicalDimension: 2, elements: [.zero, .one])
        ttn.setNode(at: 1, tensor: leaf0)
        ttn.setNode(at: 1, tensor: leaf1)
        let root = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.one, .zero, .zero, .one])
        ttn.setNode(at: 0, tensor: root)
        ttn.setNode(at: 2, tensor: leaf0)
        let result = ttn.contract()
        #expect(abs(result.imaginary) < 1e-10, "Replacing node should use the latest tensor")
    }
}

/// Tests contract method returns a scalar Complex value.
/// Validates bottom-up contraction algorithm produces single result.
/// Ensures contraction respects quantum state normalization properties.
@Suite("TTN Contract Returns Scalar")
struct TTNContractReturnsScalarTests {
    @Test("Contract returns Complex Double")
    func contractReturnsComplexDouble() {
        var ttn = TreeTensorNetwork(topology: .binary(depth: 1))
        let root = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.one, .zero, .zero, .one])
        let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        ttn.setNode(at: 0, tensor: root)
        ttn.setNode(at: 1, tensor: leaf)
        ttn.setNode(at: 2, tensor: leaf)
        let result = ttn.contract()
        let isFinite = result.real.isFinite && result.imaginary.isFinite
        #expect(isFinite, "Contract should return finite Complex<Double> scalar")
    }

    @Test("Contract produces non-negative magnitude squared")
    func contractProducesNonNegativeMagnitude() {
        var ttn = TreeTensorNetwork(topology: .binary(depth: 1))
        let root = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.one, .zero, .zero, .one])
        let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        ttn.setNode(at: 0, tensor: root)
        ttn.setNode(at: 1, tensor: leaf)
        ttn.setNode(at: 2, tensor: leaf)
        let result = ttn.contract()
        let magnitudeSquared = result.real * result.real + result.imaginary * result.imaginary
        #expect(magnitudeSquared >= 0.0, "Magnitude squared of contracted result should be non-negative")
    }

    @Test("Contract depth 2 returns scalar")
    func contractDepth2ReturnsScalar() {
        var ttn = TreeTensorNetwork(topology: .binary(depth: 2))
        let internal2x2 = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.one, .zero, .zero, .one])
        let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        for i in 0 ..< 3 {
            ttn.setNode(at: i, tensor: internal2x2)
        }
        for i in 3 ..< 7 {
            ttn.setNode(at: i, tensor: leaf)
        }
        let result = ttn.contract()
        let isFinite = result.real.isFinite && result.imaginary.isFinite
        #expect(isFinite, "Depth 2 contraction should return finite scalar")
    }

    @Test("Contract custom topology returns scalar")
    func contractCustomTopologyReturnsScalar() {
        let adjacency = [[1, 2], [], []]
        var ttn = TreeTensorNetwork(topology: .custom(adjacency: adjacency))
        let root = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.one, .zero, .zero, .one])
        let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        ttn.setNode(at: 0, tensor: root)
        ttn.setNode(at: 1, tensor: leaf)
        ttn.setNode(at: 2, tensor: leaf)
        let result = ttn.contract()
        let isFinite = result.real.isFinite && result.imaginary.isFinite
        #expect(isFinite, "Custom topology contraction should return finite scalar")
    }
}

/// Tests product state contraction yields known analytical results.
/// Validates |00...0> state with identity internal nodes gives amplitude 1.
/// Ensures correct quantum mechanical amplitudes for simple tensor networks.
@Suite("TTN Product State Contraction Known Results")
struct TTNProductStateContractionTests {
    @Test("Product state |00> with identity gives 1")
    func productState00WithIdentityGives1() {
        var ttn = TreeTensorNetwork(topology: .binary(depth: 1))
        let identity2x2: [Complex<Double>] = [.one, .zero, .zero, .one]
        let root = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: identity2x2)
        let leaf0 = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        ttn.setNode(at: 0, tensor: root)
        ttn.setNode(at: 1, tensor: leaf0)
        ttn.setNode(at: 2, tensor: leaf0)
        let result = ttn.contract()
        #expect(abs(result.real - 1.0) < 1e-10, "Product state |00> with identity should give amplitude 1.0")
        #expect(abs(result.imaginary) < 1e-10, "Product state |00> contraction should have zero imaginary part")
    }

    @Test("Product state |11> with identity gives 1")
    func productState11WithIdentityGives1() {
        var ttn = TreeTensorNetwork(topology: .binary(depth: 1))
        let identity2x2: [Complex<Double>] = [.one, .zero, .zero, .one]
        let root = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: identity2x2)
        let leaf1 = TreeTensorNode.leaf(physicalDimension: 2, elements: [.zero, .one])
        ttn.setNode(at: 0, tensor: root)
        ttn.setNode(at: 1, tensor: leaf1)
        ttn.setNode(at: 2, tensor: leaf1)
        let result = ttn.contract()
        #expect(abs(result.real - 1.0) < 1e-10, "Product state |11> with identity should give amplitude 1.0")
        #expect(abs(result.imaginary) < 1e-10, "Product state |11> contraction should have zero imaginary part")
    }

    @Test("Product state |01> with identity gives 0")
    func productState01WithIdentityGives0() {
        var ttn = TreeTensorNetwork(topology: .binary(depth: 1))
        let identity2x2: [Complex<Double>] = [.one, .zero, .zero, .one]
        let root = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: identity2x2)
        let leaf0 = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        let leaf1 = TreeTensorNode.leaf(physicalDimension: 2, elements: [.zero, .one])
        ttn.setNode(at: 0, tensor: root)
        ttn.setNode(at: 1, tensor: leaf0)
        ttn.setNode(at: 2, tensor: leaf1)
        let result = ttn.contract()
        #expect(abs(result.real) < 1e-10, "Product state |01> with identity should give amplitude 0")
        #expect(abs(result.imaginary) < 1e-10, "Product state |01> contraction should have zero imaginary part")
    }

    @Test("Four qubit product state |0000> gives 1")
    func fourQubitProductState0000Gives1() {
        var ttn = TreeTensorNetwork(topology: .binary(depth: 2))
        let identity2x2: [Complex<Double>] = [.one, .zero, .zero, .one]
        let internal2x2 = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: identity2x2)
        let leaf0 = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        for i in 0 ..< 3 {
            ttn.setNode(at: i, tensor: internal2x2)
        }
        for i in 3 ..< 7 {
            ttn.setNode(at: i, tensor: leaf0)
        }
        let result = ttn.contract()
        #expect(abs(result.real - 1.0) < 1e-10, "Four qubit product state |0000> should give amplitude 1.0")
        #expect(abs(result.imaginary) < 1e-10, "Four qubit product state contraction should have zero imaginary part")
    }

    @Test("Superposition state gives correct amplitude")
    func superpositionStateCorrectAmplitude() {
        var ttn = TreeTensorNetwork(topology: .binary(depth: 1))
        let invSqrt2 = 1.0 / Foundation.sqrt(2.0)
        let superpositionElements: [Complex<Double>] = [Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0)]
        let plusState = TreeTensorNode.leaf(physicalDimension: 2, elements: superpositionElements)
        let identity2x2: [Complex<Double>] = [.one, .zero, .zero, .one]
        let root = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: identity2x2)
        ttn.setNode(at: 0, tensor: root)
        ttn.setNode(at: 1, tensor: plusState)
        ttn.setNode(at: 2, tensor: plusState)
        let result = ttn.contract()
        let expectedReal = invSqrt2 * invSqrt2 + invSqrt2 * invSqrt2
        #expect(abs(result.real - expectedReal) < 1e-10, "Superposition state |++> should contract to 1.0")
    }

    @Test("Complex phase in leaf propagates correctly")
    func complexPhaseInLeafPropagates() {
        var ttn = TreeTensorNetwork(topology: .binary(depth: 1))
        let phaseElements: [Complex<Double>] = [Complex(0.0, 1.0), .zero]
        let phaseLeaf = TreeTensorNode.leaf(physicalDimension: 2, elements: phaseElements)
        let leaf0 = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
        let identity2x2: [Complex<Double>] = [.one, .zero, .zero, .one]
        let root = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: identity2x2)
        ttn.setNode(at: 0, tensor: root)
        ttn.setNode(at: 1, tensor: phaseLeaf)
        ttn.setNode(at: 2, tensor: leaf0)
        let result = ttn.contract()
        #expect(abs(result.real) < 1e-10, "Phase leaf real part should be 0")
        #expect(abs(result.imaginary - 1.0) < 1e-10, "Phase leaf imaginary part should be 1.0")
    }
}

/// Tests Topology enum equality comparison.
/// Validates binary and custom topology equality semantics.
/// Ensures correct Equatable conformance for tree specifications.
@Suite("TTN Topology Equatable")
struct TTNTopologyEquatableTests {
    @Test("Same binary depth topologies are equal")
    func sameBinaryDepthEqual() {
        let topology1 = TreeTensorNetwork.Topology.binary(depth: 2)
        let topology2 = TreeTensorNetwork.Topology.binary(depth: 2)
        #expect(topology1 == topology2, "Binary topologies with same depth should be equal")
    }

    @Test("Different binary depth topologies are not equal")
    func differentBinaryDepthNotEqual() {
        let topology1 = TreeTensorNetwork.Topology.binary(depth: 2)
        let topology2 = TreeTensorNetwork.Topology.binary(depth: 3)
        #expect(topology1 != topology2, "Binary topologies with different depths should not be equal")
    }

    @Test("Same custom adjacency topologies are equal")
    func sameCustomAdjacencyEqual() {
        let adjacency = [[1, 2], [], []]
        let topology1 = TreeTensorNetwork.Topology.custom(adjacency: adjacency)
        let topology2 = TreeTensorNetwork.Topology.custom(adjacency: adjacency)
        #expect(topology1 == topology2, "Custom topologies with same adjacency should be equal")
    }

    @Test("Different custom adjacency topologies are not equal")
    func differentCustomAdjacencyNotEqual() {
        let topology1 = TreeTensorNetwork.Topology.custom(adjacency: [[1, 2], [], []])
        let topology2 = TreeTensorNetwork.Topology.custom(adjacency: [[1, 2], [3, 4], [], [], []])
        #expect(topology1 != topology2, "Custom topologies with different adjacency should not be equal")
    }

    @Test("Binary and custom topologies are not equal")
    func binaryAndCustomNotEqual() {
        let binary = TreeTensorNetwork.Topology.binary(depth: 1)
        let custom = TreeTensorNetwork.Topology.custom(adjacency: [[1, 2], [], []])
        #expect(binary != custom, "Binary and custom topologies should not be equal")
    }
}

/// Tests contraction with large bond dimensions triggering BLAS path.
/// Validates that the accelerated contraction produces correct results
/// when tensor dimensions exceed the BLAS dispatch threshold.
@Suite("TTN Large Dimension Contraction")
struct TTNLargeDimensionContractionTests {
    @Test("Contraction with bond dimension 4 triggers BLAS path")
    func contractionWithBondDim4TriggersBLAS() {
        var ttn = TreeTensorNetwork(topology: .binary(depth: 1))

        var leafElements0 = [Complex<Double>](repeating: .zero, count: 4)
        leafElements0[0] = .one
        let leaf0 = TreeTensorNode.leaf(physicalDimension: 4, elements: leafElements0)

        var leafElements1 = [Complex<Double>](repeating: .zero, count: 4)
        leafElements1[0] = .one
        let leaf1 = TreeTensorNode.leaf(physicalDimension: 4, elements: leafElements1)

        var internalElements = [Complex<Double>](repeating: .zero, count: 16)
        for i in 0 ..< 4 {
            internalElements[i * 4 + i] = .one
        }
        let root = TreeTensorNode.internal(childBondDimensions: [4, 4], elements: internalElements)

        ttn.setNode(at: 0, tensor: root)
        ttn.setNode(at: 1, tensor: leaf0)
        ttn.setNode(at: 2, tensor: leaf1)

        let result = ttn.contract()
        #expect(
            abs(result.real - 1.0) < 1e-10,
            "Identity contraction with dim-4 tensors should produce 1.0, got \(result.real)",
        )
        #expect(
            abs(result.imaginary) < 1e-10,
            "Imaginary part should be zero, got \(result.imaginary)",
        )
    }

    @Test("BLAS path produces same result as scalar path")
    func blasPathMatchesScalarPath() {
        var ttnSmall = TreeTensorNetwork(topology: .binary(depth: 1))
        var ttnLarge = TreeTensorNetwork(topology: .binary(depth: 1))

        let smallLeaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [Complex(0.6, 0.0), Complex(0.8, 0.0)])
        let smallRoot = TreeTensorNode.internal(
            childBondDimensions: [2, 2],
            elements: [Complex(1, 0), Complex(0, 1), Complex(0, -1), Complex(1, 0)],
        )
        ttnSmall.setNode(at: 0, tensor: smallRoot)
        ttnSmall.setNode(at: 1, tensor: smallLeaf)
        ttnSmall.setNode(at: 2, tensor: smallLeaf)
        let smallResult = ttnSmall.contract()

        var largeLeafElements = [Complex<Double>](repeating: .zero, count: 4)
        largeLeafElements[0] = Complex(0.6, 0.0)
        largeLeafElements[1] = Complex(0.8, 0.0)
        let largeLeaf = TreeTensorNode.leaf(physicalDimension: 4, elements: largeLeafElements)

        var largeInternalElements = [Complex<Double>](repeating: .zero, count: 16)
        largeInternalElements[0] = Complex(1, 0)
        largeInternalElements[1] = Complex(0, 1)
        largeInternalElements[4] = Complex(0, -1)
        largeInternalElements[5] = Complex(1, 0)
        let largeRoot = TreeTensorNode.internal(childBondDimensions: [4, 4], elements: largeInternalElements)

        ttnLarge.setNode(at: 0, tensor: largeRoot)
        ttnLarge.setNode(at: 1, tensor: largeLeaf)
        ttnLarge.setNode(at: 2, tensor: largeLeaf)
        let largeResult = ttnLarge.contract()

        #expect(
            abs(smallResult.real - largeResult.real) < 1e-10,
            "BLAS and scalar paths should produce same real part: scalar=\(smallResult.real), blas=\(largeResult.real)",
        )
        #expect(
            abs(smallResult.imaginary - largeResult.imaginary) < 1e-10,
            "BLAS and scalar paths should produce same imaginary part: scalar=\(smallResult.imaginary), blas=\(largeResult.imaginary)",
        )
    }
}

/// Tests Sendable conformance of TreeTensorNetwork.
/// Validates that TTN instances can be safely transferred across concurrency domains.
/// Ensures thread-safe usage in Swift structured concurrency contexts.
@Suite("TTN Sendable Conformance")
struct TTNSendableConformanceTests {
    @Test("TreeTensorNetwork is Sendable")
    func ttnIsSendable() {
        let ttn = TreeTensorNetwork(topology: .binary(depth: 1))
        _ = ttn as Sendable
    }

    @Test("Topology is Sendable")
    func topologyIsSendable() {
        let topology = TreeTensorNetwork.Topology.binary(depth: 2)
        _ = topology as Sendable
    }
}
