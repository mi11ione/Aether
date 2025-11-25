// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// A group of qubit-wise commuting Pauli strings that can be measured simultaneously.
@frozen
public struct QWCGroup: Sendable {
    /// Pauli strings in this group with their coefficients
    public let terms: PauliTerms

    /// Measurement basis for each qubit
    /// Maps qubit index -> Pauli basis to measure in
    public let measurementBasis: MeasurementBasis

    /// Total weight of this group (sum of absolute coefficients)
    @inlinable
    @_effects(readonly)
    public func weight() -> Double {
        terms.reduce(0.0) { $0 + abs($1.coefficient) }
    }

    public init(terms: PauliTerms, measurementBasis: MeasurementBasis) {
        self.terms = terms
        self.measurementBasis = measurementBasis
    }
}

/// Groups Pauli strings using qubit-wise commutation (QWC) criterion.
///
/// This implementation uses graph coloring to find an optimal grouping of Pauli terms
/// such that terms in each group can be measured simultaneously.
///
/// Algorithm:
/// 1. Build conflict graph: nodes = Pauli strings, edges = non-QWC pairs
/// 2. Color graph using DSATUR (degree of saturation) algorithm
/// 3. Each color represents one measurement group
///
/// DSATUR is a greedy algorithm that provides near-optimal coloring:
/// - Select uncolored vertex with highest saturation degree (most distinct colors in neighbors)
/// - Assign smallest available color
/// - Ties broken by highest degree
///
/// For typical molecular Hamiltonians with 2000 terms, this reduces to ~50-200 groups.
public enum QWCGrouper {
    // MARK: - Main Grouping Algorithm

    /// Group Pauli terms by qubit-wise commutation.
    ///
    /// - Parameter terms: Array of (coefficient, PauliString) pairs
    /// - Returns: Array of QWC groups
    ///
    /// Example:
    /// ```swift
    /// let hamiltonian = Observable(terms: molecularTerms)  // 2000 terms
    /// let groups = QWCGrouper.group(terms: hamiltonian.terms)
    /// print("Reduced \(hamiltonian.terms.count) terms to \(groups.count) groups")
    /// // Output: Reduced 2000 terms to 48 groups (41× reduction)
    /// ```
    @_optimize(speed)
    @_eagerMove
    public static func group(terms: PauliTerms) -> [QWCGroup] {
        guard !terms.isEmpty else { return [] }

        let graph: [[Int]] = buildConflictGraph(terms: terms)
        let (coloring, numColors) = colorGraphDSATUR(graph: graph)

        return buildGroups(terms: terms, coloring: coloring, numColors: numColors)
    }

    // MARK: - Graph Construction

    /// Build conflict graph where edges represent non-QWC pairs.
    @_optimize(speed)
    @_eagerMove
    private static func buildConflictGraph(terms: PauliTerms) -> [[Int]] {
        let n: Int = terms.count
        var adjacencyList: [[Int]] = Array(repeating: [Int](), count: n)

        for i in 0 ..< n {
            for j in (i + 1) ..< n {
                if !PauliCommutation.qubitWiseCommute(terms[i].pauliString, terms[j].pauliString) {
                    adjacencyList[i].append(j)
                    adjacencyList[j].append(i)
                }
            }
        }

        return adjacencyList
    }

    // MARK: - Graph Coloring (DSATUR Algorithm)

    /// Color graph using DSATUR (Degree of Saturation) algorithm with priority queue.
    ///
    /// - Parameter graph: Adjacency list representation
    /// - Returns: Tuple of (colors array, number of colors used)
    ///
    /// DSATUR provides better coloring than simple greedy for most graphs.
    /// Time complexity: O(n log n) using priority queue for vertex selection
    @_optimize(speed)
    @_eagerMove
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
            let color: Int = forbiddenBits == UInt64.max ? 64 : (~forbiddenBits).trailingZeroBitCount

            colors[selectedVertex] = color
            if color > maxColorUsed { maxColorUsed = color }

            let colorBit: UInt64 = color < 64 ? (1 << color) : 0

            for neighbor in graph[selectedVertex] where colors[neighbor] == -1 {
                if color < 64, neighborColorBits[neighbor] & colorBit == 0 {
                    neighborColorBits[neighbor] |= colorBit
                    saturationDegree[neighbor] += 1
                    heap.updatePriority(vertex: neighbor, newSaturation: saturationDegree[neighbor])
                }
            }
        }

        return (colors: colors, numColors: maxColorUsed + 1)
    }

    // MARK: - Priority Queue for DSATUR

    /// Max-heap for DSATUR vertex selection with O(log n) operations
    private struct DSATURHeap {
        private var heap: [(saturation: Int, degree: Int, vertex: Int)]
        private var vertexToIndex: [Int]
        private var removed: [Bool]

        init(capacity: Int) {
            heap = []
            heap.reserveCapacity(capacity)
            vertexToIndex = Array(repeating: -1, count: capacity)
            removed = Array(repeating: false, count: capacity)
        }

        @inline(__always)
        private func compare(_ a: (saturation: Int, degree: Int, vertex: Int),
                             _ b: (saturation: Int, degree: Int, vertex: Int)) -> Bool
        {
            if a.saturation != b.saturation { return a.saturation > b.saturation }
            return a.degree > b.degree
        }

        mutating func insert(saturation: Int, degree: Int, vertex: Int) {
            let index = heap.count
            heap.append((saturation: saturation, degree: degree, vertex: vertex))
            vertexToIndex[vertex] = index
            siftUp(index)
        }

        mutating func extractMax() -> Int? {
            while !heap.isEmpty {
                let top = heap[0]
                if removed[top.vertex] {
                    removeTop()
                    continue
                }
                removed[top.vertex] = true
                removeTop()
                return top.vertex
            }
            return nil
        }

        mutating func updatePriority(vertex: Int, newSaturation: Int) {
            let index = vertexToIndex[vertex]
            guard index >= 0, index < heap.count, heap[index].vertex == vertex else { return }

            heap[index].saturation = newSaturation
            siftUp(index)
        }

        @inline(__always)
        private mutating func removeTop() {
            guard !heap.isEmpty else { return }
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

    // MARK: - Group Building

    /// Build QWC groups from coloring.
    @_optimize(speed)
    @_eagerMove
    private static func buildGroups(
        terms: PauliTerms,
        coloring: [Int],
        numColors: Int
    ) -> [QWCGroup] {
        var colorToTerms: [PauliTerms] = Array(repeating: [], count: numColors)

        for (index, color) in coloring.enumerated() {
            colorToTerms[color].append(terms[index])
        }

        var groups: [QWCGroup] = []
        groups.reserveCapacity(numColors)

        for groupTerms in colorToTerms where !groupTerms.isEmpty {
            let pauliStrings: [PauliString] = groupTerms.map(\.pauliString)

            // Safe: Graph coloring guarantees all strings in same color are QWC,
            // so measurementBasis always succeeds
            let basis: MeasurementBasis = PauliCommutation.measurementBasis(for: pauliStrings)!

            groups.append(QWCGroup(terms: groupTerms, measurementBasis: basis))
        }

        return groups
    }

    // MARK: - Statistics

    /// Compute grouping statistics.
    @frozen
    public struct GroupingStats {
        public let numTerms: Int
        public let numGroups: Int
        public let reductionFactor: Double
        public let largestGroupSize: Int
        public let averageGroupSize: Double
        public let groupSizes: [Int]

        @inlinable
        public var description: String {
            """
            QWC Grouping Statistics:
            - Terms: \(numTerms)
            - Groups: \(numGroups)
            - Reduction: \(String(format: "%.1f", reductionFactor))×
            - Largest group: \(largestGroupSize) terms
            - Average group: \(String(format: "%.1f", averageGroupSize)) terms
            """
        }
    }

    /// Compute statistics for a grouping.
    ///
    /// - Parameter groups: Array of QWC groups
    /// - Returns: Statistics about the grouping
    ///
    /// Example:
    /// ```swift
    /// let groups = QWCGrouper.group(terms: hamiltonian.terms)
    /// let stats = QWCGrouper.statistics(for: groups)
    /// print(stats.description)
    /// // QWC Grouping Statistics:
    /// // - Terms: 2000
    /// // - Groups: 48
    /// // - Reduction: 41.7×
    /// ```
    @_optimize(speed)
    @_eagerMove
    public static func statistics(for groups: [QWCGroup]) -> GroupingStats {
        let numGroups: Int = groups.count

        guard numGroups > 0 else {
            return GroupingStats(
                numTerms: 0,
                numGroups: 0,
                reductionFactor: 1.0,
                largestGroupSize: 0,
                averageGroupSize: 0.0,
                groupSizes: []
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
            if size > largestGroupSize { largestGroupSize = size }
        }

        let averageGroupSize = Double(numTerms) / Double(numGroups)
        let reductionFactor = Double(numTerms) / Double(numGroups)

        return GroupingStats(
            numTerms: numTerms,
            numGroups: numGroups,
            reductionFactor: reductionFactor,
            largestGroupSize: largestGroupSize,
            averageGroupSize: averageGroupSize,
            groupSizes: groupSizes
        )
    }
}
