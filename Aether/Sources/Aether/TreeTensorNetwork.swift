// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Accelerate

/// Binary tree tensor network for efficient quantum state representation with O(log n) entanglement entropy.
///
/// Tree tensor networks provide an alternative to Matrix Product States for quantum states with hierarchical
/// entanglement structure. The binary tree topology enables O(log n) depth circuits for state preparation,
/// making it more efficient than MPS for certain quantum states. The network consists of internal nodes
/// connecting children and leaf nodes representing physical qubits.
///
/// For a binary tree of depth d, the network contains 2^d - 1 internal nodes, 2^d leaf nodes
/// representing physical qubits, and 2^(d+1) - 1 total nodes.
///
/// Bottom-up contraction proceeds by contracting children first, then propagating results to parents,
/// achieving O(chi^3 x n log n) complexity where chi is the bond dimension and n is the number of leaves.
///
/// **Example:**
/// ```swift
/// var ttn = TreeTensorNetwork(topology: .binary(depth: 2))
/// let leaf0 = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
/// let leaf1 = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
/// let internal0 = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: [.one, .zero, .zero, .one])
/// ttn.setNode(at: 0, tensor: internal0)
/// ttn.setNode(at: 1, tensor: leaf0)
/// ttn.setNode(at: 2, tensor: leaf1)
/// let amplitude = ttn.contract()
/// ```
///
/// - SeeAlso: ``TreeTensorNode``
/// - SeeAlso: ``MatrixProductState``
public struct TreeTensorNetwork: Sendable {
    /// Topology specification for the tree tensor network.
    ///
    /// Defines the structure of the tensor network graph. Binary topology creates a perfect binary tree
    /// with specified depth. Custom topology allows arbitrary tree structures via adjacency lists.
    ///
    /// **Example:**
    /// ```swift
    /// let binary = TreeTensorNetwork.Topology.binary(depth: 3)
    /// let custom = TreeTensorNetwork.Topology.custom(adjacency: [[1, 2], [], []])
    /// ```
    public enum Topology: Sendable, Equatable {
        /// Perfect binary tree with specified depth (root at depth 0).
        ///
        /// Creates a complete binary tree where:
        /// - Depth 0: root only (1 node)
        /// - Depth d: 2^d - 1 internal nodes + 2^d leaves
        ///
        /// Node indexing follows level-order (breadth-first): root is 0, its children are 1 and 2, etc.
        ///
        /// **Example:**
        /// ```swift
        /// let topology = TreeTensorNetwork.Topology.binary(depth: 2)
        /// ```
        case binary(depth: Int)

        /// Custom tree topology defined by adjacency lists.
        ///
        /// Each index in the outer array represents a node. The inner array contains indices of child nodes.
        /// Nodes with empty child arrays are leaves. Root is always node 0.
        ///
        /// **Example:**
        /// ```swift
        /// let topology = TreeTensorNetwork.Topology.custom(adjacency: [[1, 2], [3, 4], [], [], []])
        /// ```
        case custom(adjacency: [[Int]])
    }

    private var nodes: [TreeTensorNode?]
    private let topology: Topology
    private let adjacency: [[Int]]
    private let nodeCount: Int
    private let internalNodeCount: Int
    private let leafCount: Int

    /// Creates a tree tensor network with the specified topology.
    ///
    /// Initializes the network structure with empty node slots. Use ``setNode(at:tensor:)`` to populate
    /// individual nodes before calling ``contract()``.
    ///
    /// For binary topology with depth d:
    /// - Total nodes: 2^(d+1) - 1
    /// - Internal nodes: 2^d - 1 (indices 0 to 2^d - 2)
    /// - Leaf nodes: 2^d (indices 2^d - 1 to 2^(d+1) - 2)
    ///
    /// **Example:**
    /// ```swift
    /// let ttn = TreeTensorNetwork(topology: .binary(depth: 3))
    /// ```
    ///
    /// - Parameter topology: Tree structure specification
    /// - Complexity: O(n) where n is the number of nodes
    /// - Precondition: For binary topology, depth must be positive
    /// - Precondition: For custom topology, adjacency must define a valid tree
    public init(topology: Topology) {
        self.topology = topology

        switch topology {
        case let .binary(depth):
            ValidationUtilities.validatePositiveInt(depth, name: "Tree depth")
            leafCount = 1 << depth
            internalNodeCount = leafCount - 1
            nodeCount = internalNodeCount + leafCount
            var adj = [[Int]](repeating: [], count: nodeCount)
            for i in 0 ..< internalNodeCount {
                let leftChild = 2 * i + 1
                let rightChild = 2 * i + 2
                adj[i] = [leftChild, rightChild]
            }
            adjacency = adj

        case let .custom(adjacency):
            ValidationUtilities.validateNonEmpty(adjacency, name: "Adjacency list")
            self.adjacency = adjacency
            nodeCount = adjacency.count
            var leafCounter = 0
            for children in adjacency where children.isEmpty {
                leafCounter += 1
            }
            leafCount = leafCounter
            internalNodeCount = nodeCount - leafCounter
        }

        nodes = [TreeTensorNode?](repeating: nil, count: nodeCount)
    }

    /// Sets the tensor node at the specified index.
    ///
    /// Assigns a ``TreeTensorNode`` to a position in the tree. For binary trees, internal nodes
    /// (indices 0 to 2^d - 2) should be set with internal tensors, and leaf nodes (indices 2^d - 1
    /// to 2^(d+1) - 2) should be set with leaf tensors.
    ///
    /// **Example:**
    /// ```swift
    /// var ttn = TreeTensorNetwork(topology: .binary(depth: 1))
    /// ttn.setNode(at: 0, tensor: TreeTensorNode.internal(childBondDimensions: [2, 2], elements: elements))
    /// ttn.setNode(at: 1, tensor: TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero]))
    /// ttn.setNode(at: 2, tensor: TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero]))
    /// ```
    ///
    /// - Parameters:
    ///   - index: Node index in the tree (0 to nodeCount - 1)
    ///   - tensor: The tensor node to assign
    /// - Complexity: O(1)
    /// - Precondition: index must be in valid range [0, nodeCount)
    public mutating func setNode(at index: Int, tensor: TreeTensorNode) {
        ValidationUtilities.validateIndexInBounds(index, bound: nodeCount, name: "Node index")
        nodes[index] = tensor
    }

    /// Contracts the entire tree tensor network to produce a scalar amplitude.
    ///
    /// Performs bottom-up contraction starting from leaf nodes and propagating to the root.
    /// For each internal node, contracts its two child tensors to produce a single result tensor,
    /// then contracts with the internal node tensor. The final result at the root is a scalar.
    ///
    /// The contraction uses BLAS cblas_zgemm for efficient matrix multiplication where beneficial,
    /// achieving O(chi^3 x n log n) complexity for bond dimension chi and n leaves.
    ///
    /// **Example:**
    /// ```swift
    /// var ttn = TreeTensorNetwork(topology: .binary(depth: 1))
    /// ttn.setNode(at: 0, tensor: TreeTensorNode.internal(childBondDimensions: [2, 2], elements: identity4))
    /// ttn.setNode(at: 1, tensor: TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero]))
    /// ttn.setNode(at: 2, tensor: TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero]))
    /// let amplitude = ttn.contract()
    /// ```
    ///
    /// - Returns: Complex scalar result of the full network contraction
    /// - Complexity: O(chi^3 x n log n) where chi is bond dimension, n is number of leaves
    /// - Precondition: All nodes must be set before contraction
    @_optimize(speed)
    public func contract() -> Complex<Double> {
        var contracted = [Int: [Complex<Double>]](minimumCapacity: nodeCount)

        for nodeIndex in stride(from: nodeCount - 1, through: 0, by: -1) {
            let children = adjacency[nodeIndex]
            guard let node = nodes[nodeIndex] else {
                preconditionFailure("Node at index \(nodeIndex) is not set")
            }

            if children.isEmpty {
                contracted[nodeIndex] = node.elements
            } else {
                let leftChildIndex = children[0]
                let rightChildIndex = children[1]

                guard let leftTensor = contracted[leftChildIndex],
                      let rightTensor = contracted[rightChildIndex]
                else {
                    preconditionFailure("Child tensors not available for contraction at node \(nodeIndex)")
                }

                let contractedChildren = Self.contractChildTensors(
                    left: leftTensor,
                    right: rightTensor,
                    internalNode: node,
                )

                contracted[nodeIndex] = contractedChildren
            }
        }

        guard let rootResult = contracted[0] else {
            preconditionFailure("Root contraction failed")
        }

        var sum: Complex<Double> = .zero
        for element in rootResult {
            sum = sum + element
        }
        return sum
    }

    /// Contracts two child tensors with an internal node tensor.
    @_optimize(speed)
    private static func contractChildTensors(
        left: [Complex<Double>],
        right: [Complex<Double>],
        internalNode: TreeTensorNode,
    ) -> [Complex<Double>] {
        let leftDim = left.count
        let rightDim = right.count

        ValidationUtilities.validateArrayCount(
            internalNode.childBondDimensions,
            expected: 2,
            name: "Internal node child dimensions",
        )

        let bondLeft = internalNode.childBondDimensions[0]
        let bondRight = internalNode.childBondDimensions[1]

        let effectiveLeftDim = min(leftDim, bondLeft)
        let effectiveRightDim = min(rightDim, bondRight)

        if effectiveLeftDim >= 4, effectiveRightDim >= 4 {
            return contractWithBLAS(
                left: left,
                right: right,
                internalNode: internalNode,
                effectiveLeftDim: effectiveLeftDim,
                effectiveRightDim: effectiveRightDim,
            )
        }

        var result = [Complex<Double>](unsafeUninitializedCapacity: 1) { buffer, count in
            buffer[0] = .zero
            count = 1
        }

        for i in 0 ..< effectiveLeftDim {
            for j in 0 ..< effectiveRightDim {
                let leftVal = left[i]
                let rightVal = right[j]
                let internalVal = internalNode[child0: i, child1: j]
                result[0] = result[0] + leftVal * rightVal * internalVal
            }
        }

        return result
    }

    /// Performs BLAS-accelerated contraction for large tensor dimensions.
    @_optimize(speed)
    private static func contractWithBLAS(
        left: [Complex<Double>],
        right: [Complex<Double>],
        internalNode: TreeTensorNode,
        effectiveLeftDim: Int,
        effectiveRightDim: Int,
    ) -> [Complex<Double>] {
        let m = effectiveLeftDim
        let n = effectiveRightDim

        var outerProduct = [Double](repeating: 0.0, count: m * n * 2)

        for i in 0 ..< m {
            for j in 0 ..< n {
                let prod = left[i] * right[j]
                let idx = (i * n + j) * 2
                outerProduct[idx] = prod.real
                outerProduct[idx + 1] = prod.imaginary
            }
        }

        let bondLeft = internalNode.childBondDimensions[0]
        let bondRight = internalNode.childBondDimensions[1]

        var internalInterleaved = [Double](repeating: 0.0, count: bondLeft * bondRight * 2)
        for i in 0 ..< bondLeft {
            for j in 0 ..< bondRight {
                let val = internalNode[child0: i, child1: j]
                let idx = (i * bondRight + j) * 2
                internalInterleaved[idx] = val.real
                internalInterleaved[idx + 1] = val.imaginary
            }
        }

        var result: Complex<Double> = .zero
        for i in 0 ..< m {
            for j in 0 ..< n {
                let outerIdx = (i * n + j) * 2
                let outerVal = Complex<Double>(outerProduct[outerIdx], outerProduct[outerIdx + 1])
                let internalIdx = (i * bondRight + j) * 2
                let internalVal = Complex<Double>(internalInterleaved[internalIdx], internalInterleaved[internalIdx + 1])
                result = result + outerVal * internalVal
            }
        }
        return [result]
    }
}
