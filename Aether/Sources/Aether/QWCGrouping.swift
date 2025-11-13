// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// A group of qubit-wise commuting Pauli strings that can be measured simultaneously.
@frozen
public struct QWCGroup: Sendable {
    /// Pauli strings in this group with their coefficients
    public let terms: PauliTerms

    /// Measurement basis for each qubit
    /// Maps qubit index → Pauli basis to measure in
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
        let coloring: [Int] = colorGraphDSATUR(graph: graph)

        return buildGroups(terms: terms, coloring: coloring)
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

    /// Color graph using DSATUR (Degree of Saturation) algorithm.
    ///
    /// - Parameter graph: Adjacency list representation
    /// - Returns: Array mapping vertex index to color
    ///
    /// DSATUR provides better coloring than simple greedy for most graphs.
    /// Time complexity: O(n²) where n = number of vertices
    @_optimize(speed)
    @_eagerMove
    private static func colorGraphDSATUR(graph: [[Int]]) -> [Int] {
        let n: Int = graph.count
        var colors: [Int] = Array(repeating: -1, count: n)
        var saturationDegree: [Int] = Array(repeating: 0, count: n)
        var neighborColors: [Set<Int>] = Array(repeating: Set<Int>(), count: n)

        for _ in 0 ..< n {
            var selectedVertex: Int = -1
            var maxSaturation: Int = -1
            var maxDegree: Int = -1

            for v in 0 ..< n where colors[v] == -1 {
                let saturation: Int = saturationDegree[v]
                let degree: Int = graph[v].count

                if saturation > maxSaturation ||
                    (saturation == maxSaturation && degree > maxDegree)
                {
                    selectedVertex = v
                    maxSaturation = saturation
                    maxDegree = degree
                }
            }

            let forbiddenColors: Set<Int> = neighborColors[selectedVertex]
            var color = 0
            while forbiddenColors.contains(color) {
                color += 1
            }

            colors[selectedVertex] = color

            for neighbor in graph[selectedVertex] where colors[neighbor] == -1 {
                if !neighborColors[neighbor].contains(color) {
                    neighborColors[neighbor].insert(color)
                    saturationDegree[neighbor] += 1
                }
            }
        }

        return colors
    }

    // MARK: - Group Building

    /// Build QWC groups from coloring.
    @_optimize(speed)
    @_eagerMove
    private static func buildGroups(
        terms: PauliTerms,
        coloring: [Int]
    ) -> [QWCGroup] {
        var colorToTerms: [Int: PauliTerms] = [:]

        for (index, color) in coloring.enumerated() {
            colorToTerms[color, default: []].append(terms[index])
        }

        var groups: [QWCGroup] = []

        for (_, groupTerms) in colorToTerms.sorted(by: { $0.key < $1.key }) {
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
        let groupSizes: [Int] = groups.map(\.terms.count)
        let numTerms: Int = groupSizes.reduce(0, +)
        let largestGroupSize: Int = groupSizes.max() ?? 0
        let averageGroupSize: Double = numGroups > 0 ? Double(numTerms) / Double(numGroups) : 0.0
        let reductionFactor: Double = numGroups > 0 ? Double(numTerms) / Double(numGroups) : 1.0

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
