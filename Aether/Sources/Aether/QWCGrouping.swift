// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// A group of qubit-wise commuting Pauli strings that can be measured simultaneously.
///
/// Groups are produced by ``QWCGrouper/group(_:)`` via graph coloring. All Pauli strings within a group
/// satisfy the qubit-wise commutation criterion, enabling simultaneous measurement in a shared basis.
/// This reduces the number of measurement circuits needed for expectation value computation.
///
/// **Example:**
/// ```swift
/// let hamiltonian = Observable(terms: molecularTerms)
/// let groups = QWCGrouper.group(hamiltonian.terms)
/// for group in groups {
///     let expectation = measure(state, in: group.measurementBasis)
/// }
/// ```
///
/// - SeeAlso: ``QWCGrouper/group(_:)``
/// - SeeAlso: ``PauliCommutation/areQWC(_:_:)``
@frozen
public struct QWCGroup: Sendable {
    /// Pauli terms in this group with their coefficients.
    ///
    /// All terms are guaranteed to be qubit-wise commuting with each other.
    public let terms: PauliTerms

    /// The measurement basis for each qubit that appears in this group's Pauli strings.
    ///
    /// Dictionary is sparse: only includes qubits referenced by the group's terms.
    /// For qubits not in the dictionary, measurement basis is arbitrary (typically Z).
    public let measurementBasis: [Int: PauliBasis]

    /// Total weight of this group, computed as the sum of absolute coefficients.
    ///
    /// Used for variance-weighted shot allocation in measurement optimization.
    ///
    /// **Example:**
    /// ```swift
    /// let terms: PauliTerms = [(0.5, PauliString(.z(0))), (-0.3, PauliString(.z(1)))]
    /// let group = QWCGroup(terms: terms, measurementBasis: [0: .z, 1: .z])
    /// let w = group.weight
    /// ```
    ///
    /// - Complexity: O(n) where n is the number of terms in the group
    @inlinable
    public var weight: Double {
        terms.reduce(0.0) { $0 + $1.coefficient.magnitude }
    }

    /// Creates a qubit-wise commuting group.
    ///
    /// - Parameters:
    ///   - terms: Pauli terms in this group
    ///   - measurementBasis: The measurement basis for each qubit
    ///
    /// **Example:**
    /// ```swift
    /// let terms: PauliTerms = [(0.5, PauliString(.z(0))), (0.3, PauliString(.z(1)))]
    /// let basis: [Int: PauliBasis] = [0: .z, 1: .z]
    /// let group = QWCGroup(terms: terms, measurementBasis: basis)
    /// ```
    ///
    /// - Precondition: All terms must be qubit-wise commuting with each other
    public init(terms: PauliTerms, measurementBasis: [Int: PauliBasis]) {
        self.terms = terms
        self.measurementBasis = measurementBasis
    }
}

/// Groups Pauli terms using graph coloring for measurement optimization.
///
/// Qubit-wise commuting (QWC) Pauli strings can be measured simultaneously in a shared basis.
/// This grouping reduces the number of measurement circuits needed for Hamiltonian expectation values,
/// typically achieving reductions from thousands of terms to tens or hundreds of groups.
///
/// The grouper uses the DSATUR graph coloring algorithm on a conflict graph where nodes represent
/// Pauli strings and edges connect non-QWC pairs. The algorithm produces near-optimal groupings
/// with better performance than simple greedy methods.
///
/// **Example:**
/// ```swift
/// let hamiltonian = Observable(terms: molecularTerms)
/// let groups = QWCGrouper.group(hamiltonian.terms)
/// let stats = QWCGrouper.statistics(for: groups)
/// print(stats)
/// ```
///
/// - SeeAlso: ``QWCGroup``
/// - SeeAlso: ``PauliCommutation``
/// - SeeAlso: ``Observable``
public enum QWCGrouper {
    /// Groups Pauli terms into sets that can be measured simultaneously.
    ///
    /// Uses graph coloring to partition terms into qubit-wise commuting groups. Each group shares
    /// a measurement basis, enabling all terms in the group to be measured with a single circuit.
    /// Empty input returns empty array.
    ///
    /// **Example:**
    /// ```swift
    /// let hamiltonian = Observable(terms: molecularTerms)
    /// let groups = QWCGrouper.group(hamiltonian.terms)
    /// print("Reduced \(hamiltonian.terms.count) terms to \(groups.count) groups")
    /// ```
    ///
    /// - Parameter terms: Pauli terms to group
    /// - Returns: Array of QWC groups, each with a shared measurement basis
    /// - Complexity: O(n² + n log n) where n is the number of terms
    /// - SeeAlso: ``statistics(for:)``
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    public static func group(_ terms: PauliTerms) -> [QWCGroup] {
        guard !terms.isEmpty else { return [] }

        let graph: [[Int]] = buildConflictGraph(terms: terms)
        let (coloring, numColors) = colorGraphDSATUR(graph: graph)

        return buildGroups(terms: terms, coloring: coloring, numColors: numColors)
    }

    /// Build adjacency list where edges connect non-QWC term pairs.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func buildConflictGraph(terms: PauliTerms) -> [[Int]] {
        let n: Int = terms.count
        var adjacencyList: [[Int]] = Array(repeating: [Int](), count: n)

        for i in 0 ..< n {
            for j in (i + 1) ..< n {
                if !PauliCommutation.areQWC(terms[i].pauliString, terms[j].pauliString) {
                    adjacencyList[i].append(j)
                    adjacencyList[j].append(i)
                }
            }
        }

        return adjacencyList
    }

    /// Color graph using DSATUR heuristic for near-optimal chromatic number.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func colorGraphDSATUR(graph: [[Int]]) -> (colors: [Int], numColors: Int) {
        let n: Int = graph.count
        var colors: [Int] = Array(repeating: -1, count: n)
        var saturationDegree: [Int] = Array(repeating: 0, count: n)
        var neighborColorBits: [UInt64] = Array(repeating: 0, count: n)
        var maxColorUsed: Int = -1

        var heap = DSATURHeap(capacity: n)

        for v in 0 ..< n {
            heap.insert(saturation: 0, degree: graph[v].count, vertex: v)
        }

        while let selectedVertex = heap.extractMax() {
            let forbiddenBits: UInt64 = neighborColorBits[selectedVertex]
            let color: Int = forbiddenBits == UInt64.max ? UInt64.bitWidth : (~forbiddenBits).trailingZeroBitCount

            colors[selectedVertex] = color
            maxColorUsed = max(maxColorUsed, color)

            let colorBit: UInt64 = color < UInt64.bitWidth ? (1 << color) : 0

            for neighbor in graph[selectedVertex] where colors[neighbor] == -1 {
                if color < UInt64.bitWidth, neighborColorBits[neighbor] & colorBit == 0 {
                    neighborColorBits[neighbor] |= colorBit
                    saturationDegree[neighbor] += 1
                    heap.updatePriority(vertex: neighbor, newSaturation: saturationDegree[neighbor])
                }
            }
        }

        return (colors: colors, numColors: maxColorUsed + 1)
    }

    /// Priority queue for DSATUR vertex selection by saturation degree.
    private struct DSATURHeap {
        private var heap: [(saturation: Int, degree: Int, vertex: Int)]
        private var vertexToIndex: [Int]
        /// Creates a heap with pre-allocated capacity for the given number of vertices.
        init(capacity: Int) {
            heap = []
            heap.reserveCapacity(capacity)
            vertexToIndex = Array(repeating: -1, count: capacity)
        }

        /// Compares two entries by saturation degree then vertex degree.
        @inline(__always)
        private func compare(_ a: (saturation: Int, degree: Int, vertex: Int),
                             _ b: (saturation: Int, degree: Int, vertex: Int)) -> Bool
        {
            if a.saturation != b.saturation { return a.saturation > b.saturation }
            return a.degree > b.degree
        }

        /// Inserts a vertex with its saturation and degree into the heap.
        mutating func insert(saturation: Int, degree: Int, vertex: Int) {
            let index = heap.count
            heap.append((saturation: saturation, degree: degree, vertex: vertex))
            vertexToIndex[vertex] = index
            siftUp(index)
        }

        /// Extracts the vertex with highest saturation degree.
        mutating func extractMax() -> Int? {
            guard !heap.isEmpty else { return nil }
            let top = heap[0]
            removeTop()
            return top.vertex
        }

        /// Updates the saturation degree of a vertex and restores heap order.
        mutating func updatePriority(vertex: Int, newSaturation: Int) {
            let index = vertexToIndex[vertex]
            heap[index].saturation = newSaturation
            siftUp(index)
        }

        /// Removes the root element and restores heap property.
        @inline(__always)
        private mutating func removeTop() {
            let lastIndex = heap.count - 1
            if lastIndex > 0 {
                heap.swapAt(0, lastIndex)
                vertexToIndex[heap[0].vertex] = 0
            }
            vertexToIndex[heap[lastIndex].vertex] = -1
            heap.removeLast()
            if !heap.isEmpty {
                siftDown(0)
            }
        }

        /// Restores heap property by moving an element upward.
        @inline(__always)
        private mutating func siftUp(_ index: Int) {
            var i = index
            while i > 0 {
                let parent = (i - 1) / 2
                if compare(heap[i], heap[parent]) {
                    heap.swapAt(i, parent)
                    vertexToIndex[heap[i].vertex] = i
                    vertexToIndex[heap[parent].vertex] = parent
                    i = parent
                } else {
                    break
                }
            }
        }

        /// Restores heap property by moving an element downward.
        @inline(__always)
        private mutating func siftDown(_ index: Int) {
            var i = index
            let count = heap.count
            while true {
                let left = 2 * i + 1
                let right = 2 * i + 2
                var largest = i

                if left < count, compare(heap[left], heap[largest]) {
                    largest = left
                }
                if right < count, compare(heap[right], heap[largest]) {
                    largest = right
                }

                if largest != i {
                    heap.swapAt(i, largest)
                    vertexToIndex[heap[i].vertex] = i
                    vertexToIndex[heap[largest].vertex] = largest
                    i = largest
                } else {
                    break
                }
            }
        }
    }

    /// Convert coloring assignment to QWCGroup array with measurement bases.
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    private static func buildGroups(
        terms: PauliTerms,
        coloring: [Int],
        numColors: Int,
    ) -> [QWCGroup] {
        var colorToTerms: [PauliTerms] = Array(repeating: [], count: numColors)
        let avgSize = terms.count / max(numColors, 1)
        for i in 0 ..< numColors {
            colorToTerms[i].reserveCapacity(avgSize)
        }

        for index in 0 ..< coloring.count {
            colorToTerms[coloring[index]].append(terms[index])
        }

        var groups: [QWCGroup] = []
        groups.reserveCapacity(numColors)

        for groupTerms in colorToTerms where !groupTerms.isEmpty {
            let pauliStrings: [PauliString] = groupTerms.map(\.pauliString)
            let basis: [Int: PauliBasis] = PauliCommutation.measurementBasis(of: pauliStrings)! // Safety: groupTerms from same color class, guaranteed QWC by graph coloring

            groups.append(QWCGroup(terms: groupTerms, measurementBasis: basis))
        }

        return groups
    }

    /// Statistical summary of a QWC grouping.
    ///
    /// Provides metrics for analyzing grouping quality, including reduction factor and group size distribution.
    /// Use ``QWCGrouper/statistics(for:)`` to compute these statistics from a grouping.
    ///
    /// **Example:**
    /// ```swift
    /// let groups = QWCGrouper.group(hamiltonian.terms)
    /// let stats = QWCGrouper.statistics(for: groups)
    /// print("Reduction factor: \(stats.reductionFactor)x")
    /// ```
    ///
    /// - SeeAlso: ``QWCGrouper/statistics(for:)``
    @frozen
    public struct GroupingStats: Sendable {
        /// Total number of Pauli terms across all groups.
        public let numTerms: Int

        /// Number of QWC groups.
        public let numGroups: Int

        /// Reduction factor, computed as `numTerms / numGroups`.
        public let reductionFactor: Double

        /// Size of the largest group.
        public let largestGroupSize: Int

        /// Average group size.
        public let averageGroupSize: Double

        /// Size of each group.
        public let groupSizes: [Int]
    }
}

extension QWCGrouper.GroupingStats: CustomStringConvertible {
    /// Human-readable summary of grouping statistics.
    ///
    /// **Example:**
    /// ```swift
    /// let groups = QWCGrouper.group(hamiltonian.terms)
    /// let stats = QWCGrouper.statistics(for: groups)
    /// let summary = stats.description
    /// ```
    @inlinable
    public var description: String {
        """
        QWC Grouping Statistics:
        - Terms: \(numTerms)
        - Groups: \(numGroups)
        - Reduction: \(String(format: "%.1f", reductionFactor))x
        - Largest group: \(largestGroupSize) terms
        - Average group: \(String(format: "%.1f", averageGroupSize)) terms
        """
    }
}

public extension QWCGrouper {
    /// Computes statistical metrics for a QWC grouping.
    ///
    /// Analyzes a grouping to determine the number of terms, groups, reduction factor, and
    /// group size distribution. Returns default values for empty input.
    ///
    /// **Example:**
    /// ```swift
    /// let groups = QWCGrouper.group(hamiltonian.terms)
    /// let stats = QWCGrouper.statistics(for: groups)
    /// print(stats)
    /// ```
    ///
    /// - Parameter groups: QWC groups to analyze
    /// - Returns: Statistical summary including reduction factor and group sizes
    /// - Complexity: O(n) where n is the number of groups
    @_optimize(speed)
    @_eagerMove
    @_effects(readonly)
    static func statistics(for groups: [QWCGroup]) -> GroupingStats {
        let numGroups: Int = groups.count

        guard numGroups > 0 else {
            return GroupingStats(
                numTerms: 0,
                numGroups: 0,
                reductionFactor: 1.0,
                largestGroupSize: 0,
                averageGroupSize: 0.0,
                groupSizes: [],
            )
        }

        let groupSizes = [Int](unsafeUninitializedCapacity: numGroups) { buffer, count in
            count = numGroups
            for i in 0 ..< numGroups {
                buffer[i] = groups[i].terms.count
            }
        }

        var numTerms = 0
        var largestGroupSize = 0
        for size in groupSizes {
            numTerms += size
            largestGroupSize = max(largestGroupSize, size)
        }

        let ratio = Double(numTerms) / Double(numGroups)

        return GroupingStats(
            numTerms: numTerms,
            numGroups: numGroups,
            reductionFactor: ratio,
            largestGroupSize: largestGroupSize,
            averageGroupSize: ratio,
            groupSizes: groupSizes,
        )
    }
}
