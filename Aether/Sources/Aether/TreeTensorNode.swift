// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Node in a tree tensor network for log-depth entanglement representation.
///
/// Tree tensor networks provide an efficient representation for quantum states with hierarchical
/// entanglement structure. The tree topology enables O(log n) depth circuits for state preparation.
/// Leaf nodes have physical indices representing qubits, while internal nodes connect exactly two
/// children (binary tree structure). Elements are stored in row-major order over all indices.
///
/// For leaf nodes: tensor indices are [physical], so elements has physicalDimension elements.
/// For internal nodes: tensor indices are [child0Bond, child1Bond], so elements has
/// childBondDimensions[0] * childBondDimensions[1] elements.
///
/// **Example:**
/// ```swift
/// let leafNode = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
/// let internalNode = TreeTensorNode.internal(
///     childBondDimensions: [2, 2],
///     elements: [.one, .zero, .zero, .one]
/// )
/// ```
///
/// - SeeAlso: ``MPSTensor`` for linear tensor network representation
@frozen
public struct TreeTensorNode: Sendable, Equatable {
    /// Bond dimensions connecting to child nodes (empty for leaf nodes, exactly 2 for internal nodes)
    ///
    /// For leaf nodes, this array is empty as leaves have no children.
    /// For internal nodes in a binary tree, this array has exactly 2 elements representing
    /// the bond dimensions to the left and right children respectively.
    ///
    /// **Example:**
    /// ```swift
    /// let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
    /// print(leaf.childBondDimensions)  // []
    ///
    /// let internal = TreeTensorNode.internal(childBondDimensions: [4, 4], elements: elements)
    /// print(internal.childBondDimensions)  // [4, 4]
    /// ```
    public let childBondDimensions: [Int]

    /// Physical dimension for leaf nodes (nil for internal nodes)
    ///
    /// Leaf nodes represent physical qubits and have a non-nil physical dimension (typically 2
    /// for qubit basis states |0⟩ and |1⟩). Internal nodes have nil physical dimension as they
    /// only mediate entanglement between children.
    ///
    /// **Example:**
    /// ```swift
    /// let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
    /// print(leaf.physicalDimension)  // Optional(2)
    ///
    /// let internal = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: elements)
    /// print(internal.physicalDimension)  // nil
    /// ```
    public let physicalDimension: Int?

    /// Flattened tensor elements in row-major order
    ///
    /// For leaf nodes: elements indexed by physical index, count = physicalDimension.
    /// For internal nodes: elements indexed by [child0Bond, child1Bond] in row-major order,
    /// count = childBondDimensions[0] * childBondDimensions[1].
    ///
    /// **Example:**
    /// ```swift
    /// let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
    /// print(leaf.elements.count)  // 2
    ///
    /// let internal = TreeTensorNode.internal(childBondDimensions: [3, 4], elements: elements)
    /// print(internal.elements.count)  // 12
    /// ```
    public let elements: [Complex<Double>]

    /// Creates a tree tensor node with the specified parameters.
    ///
    /// Use this initializer for full control over node construction. For common cases,
    /// prefer the ``leaf(physicalDimension:elements:)`` and
    /// ``internal(childBondDimensions:elements:)`` factory methods.
    ///
    /// **Example:**
    /// ```swift
    /// let leafNode = TreeTensorNode(
    ///     childBondDimensions: [],
    ///     physicalDimension: 2,
    ///     elements: [.one, .zero]
    /// )
    /// let internalNode = TreeTensorNode(
    ///     childBondDimensions: [2, 2],
    ///     physicalDimension: nil,
    ///     elements: [.one, .zero, .zero, .one]
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - childBondDimensions: Bond dimensions to children (empty for leaf, exactly 2 for internal)
    ///   - physicalDimension: Physical dimension for leaf nodes (nil for internal nodes)
    ///   - elements: Flattened tensor elements in row-major order
    /// - Complexity: O(1)
    /// - Precondition: For leaf nodes, childBondDimensions must be empty and physicalDimension must be positive
    /// - Precondition: For internal nodes, childBondDimensions must have exactly 2 positive elements and physicalDimension must be nil
    /// - Precondition: Element count must match tensor dimensions
    public init(childBondDimensions: [Int], physicalDimension: Int?, elements: [Complex<Double>]) {
        if let physical = physicalDimension {
            ValidationUtilities.validatePositiveInt(physical, name: "Physical dimension")
            precondition(
                childBondDimensions.isEmpty,
                "Leaf nodes must have empty childBondDimensions (got \(childBondDimensions.count) children)",
            )
            ValidationUtilities.validateArrayCount(elements, expected: physical, name: "Leaf tensor elements")
        } else {
            ValidationUtilities.validateArrayCount(childBondDimensions, expected: 2, name: "Child bond dimensions")
            ValidationUtilities.validatePositiveInt(childBondDimensions[0], name: "Child 0 bond dimension")
            ValidationUtilities.validatePositiveInt(childBondDimensions[1], name: "Child 1 bond dimension")
            let expectedCount = childBondDimensions[0] * childBondDimensions[1]
            ValidationUtilities.validateArrayCount(elements, expected: expectedCount, name: "Internal tensor elements")
        }

        self.childBondDimensions = childBondDimensions
        self.physicalDimension = physicalDimension
        self.elements = elements
    }

    /// Creates a leaf node representing a physical qubit.
    ///
    /// Leaf nodes are the boundary of the tree tensor network and correspond to physical qubits.
    /// The physical dimension is typically 2 for qubit systems (|0⟩ and |1⟩ basis states).
    ///
    /// **Example:**
    /// ```swift
    /// let qubitZero = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
    /// let qubitOne = TreeTensorNode.leaf(physicalDimension: 2, elements: [.zero, .one])
    /// let superposition = TreeTensorNode.leaf(
    ///     physicalDimension: 2,
    ///     elements: [Complex(0.707, 0.0), Complex(0.707, 0.0)]
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - physicalDimension: Dimension of the physical Hilbert space (typically 2 for qubits)
    ///   - elements: Tensor elements indexed by physical index
    /// - Returns: A leaf tree tensor node
    /// - Complexity: O(1)
    /// - Precondition: physicalDimension must be positive
    /// - Precondition: elements.count must equal physicalDimension
    public static func leaf(physicalDimension: Int, elements: [Complex<Double>]) -> TreeTensorNode {
        TreeTensorNode(childBondDimensions: [], physicalDimension: physicalDimension, elements: elements)
    }

    /// Creates an internal node connecting two children.
    ///
    /// Internal nodes mediate entanglement between subtrees in the binary tree structure.
    /// They have no physical dimension and connect exactly two children via bond indices.
    ///
    /// **Example:**
    /// ```swift
    /// let identity = TreeTensorNode.internal(
    ///     childBondDimensions: [2, 2],
    ///     elements: [.one, .zero, .zero, .one]
    /// )
    /// let entangler = TreeTensorNode.internal(
    ///     childBondDimensions: [4, 4],
    ///     elements: bellStateElements
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - childBondDimensions: Array of exactly 2 bond dimensions [leftChildBond, rightChildBond]
    ///   - elements: Tensor elements in row-major order over [child0Bond, child1Bond]
    /// - Returns: An internal tree tensor node
    /// - Complexity: O(1)
    /// - Precondition: childBondDimensions must have exactly 2 positive elements
    /// - Precondition: elements.count must equal childBondDimensions[0] * childBondDimensions[1]
    public static func `internal`(childBondDimensions: [Int], elements: [Complex<Double>]) -> TreeTensorNode {
        TreeTensorNode(childBondDimensions: childBondDimensions, physicalDimension: nil, elements: elements)
    }

    /// Returns true if this node is a leaf node (has physical index).
    ///
    /// Leaf nodes represent physical qubits at the boundary of the tree tensor network.
    ///
    /// **Example:**
    /// ```swift
    /// let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
    /// print(leaf.isLeaf)  // true
    ///
    /// let internal = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: elements)
    /// print(internal.isLeaf)  // false
    /// ```
    ///
    /// - Complexity: O(1)
    @inlinable
    public var isLeaf: Bool {
        physicalDimension != nil
    }

    /// Returns true if this node is an internal node (connects children).
    ///
    /// Internal nodes mediate entanglement between subtrees without direct physical indices.
    ///
    /// **Example:**
    /// ```swift
    /// let internal = TreeTensorNode.internal(childBondDimensions: [2, 2], elements: elements)
    /// print(internal.isInternal)  // true
    ///
    /// let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
    /// print(leaf.isInternal)  // false
    /// ```
    ///
    /// - Complexity: O(1)
    @inlinable
    public var isInternal: Bool {
        physicalDimension == nil
    }

    /// Total number of tensor elements.
    ///
    /// For leaf nodes: equals physicalDimension.
    /// For internal nodes: equals childBondDimensions[0] * childBondDimensions[1].
    ///
    /// **Example:**
    /// ```swift
    /// let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
    /// print(leaf.elementCount)  // 2
    ///
    /// let internal = TreeTensorNode.internal(childBondDimensions: [3, 4], elements: elements)
    /// print(internal.elementCount)  // 12
    /// ```
    ///
    /// - Complexity: O(1)
    @inlinable
    public var elementCount: Int {
        elements.count
    }

    /// Accesses the element at the given physical index for leaf nodes.
    ///
    /// Provides direct access to tensor elements for leaf nodes using the physical index.
    ///
    /// **Example:**
    /// ```swift
    /// let leaf = TreeTensorNode.leaf(physicalDimension: 2, elements: [.one, .zero])
    /// let amplitude0 = leaf[physical: 0]  // Complex(1.0, 0.0)
    /// let amplitude1 = leaf[physical: 1]  // Complex(0.0, 0.0)
    /// ```
    ///
    /// - Parameter physical: Physical index (0 to physicalDimension-1)
    /// - Returns: Complex amplitude at the given physical index
    /// - Complexity: O(1)
    /// - Precondition: Node must be a leaf node
    /// - Precondition: physical must be in valid range
    @inlinable
    public subscript(physical physical: Int) -> Complex<Double> {
        precondition(isLeaf, "Physical index subscript requires leaf node")
        ValidationUtilities.validateIndexInBounds(physical, bound: physicalDimension!, name: "Physical index")
        return elements[physical]
    }

    /// Accesses the element at the given child bond indices for internal nodes.
    ///
    /// Provides direct access to tensor elements for internal nodes using row-major indexing
    /// over [child0Bond, child1Bond].
    ///
    /// **Example:**
    /// ```swift
    /// let identity = TreeTensorNode.internal(
    ///     childBondDimensions: [2, 2],
    ///     elements: [.one, .zero, .zero, .one]
    /// )
    /// let diag00 = identity[child0: 0, child1: 0]  // Complex(1.0, 0.0)
    /// let diag11 = identity[child0: 1, child1: 1]  // Complex(1.0, 0.0)
    /// let offDiag = identity[child0: 0, child1: 1] // Complex(0.0, 0.0)
    /// ```
    ///
    /// - Parameters:
    ///   - child0: First child bond index (0 to childBondDimensions[0]-1)
    ///   - child1: Second child bond index (0 to childBondDimensions[1]-1)
    /// - Returns: Complex amplitude at the given bond indices
    /// - Complexity: O(1)
    /// - Precondition: Node must be an internal node
    /// - Precondition: Both indices must be in valid ranges
    @inlinable
    public subscript(child0 child0: Int, child1 child1: Int) -> Complex<Double> {
        precondition(isInternal, "Child bond subscript requires internal node")
        ValidationUtilities.validateIndexInBounds(child0, bound: childBondDimensions[0], name: "Child 0 bond index")
        ValidationUtilities.validateIndexInBounds(child1, bound: childBondDimensions[1], name: "Child 1 bond index")
        let flatIndex = child0 * childBondDimensions[1] + child1
        return elements[flatIndex]
    }
}
