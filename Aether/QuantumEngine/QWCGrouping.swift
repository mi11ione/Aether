// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// A group of qubit-wise commuting Pauli strings that can be measured simultaneously.
public struct QWCGroup: Sendable {
    /// Pauli strings in this group with their coefficients
    public let terms: PauliTerms

    /// Measurement basis for each qubit
    /// Maps qubit index → Pauli basis to measure in
    public let measurementBasis: MeasurementBasis

    /// Total weight of this group (sum of absolute coefficients)
    public var weight: Double {
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
    public static func group(terms: PauliTerms) -> [QWCGroup] {
        guard !terms.isEmpty else { return [] }

        let graph = buildConflictGraph(terms: terms)
        let coloring = colorGraphDSATUR(graph: graph)

        return buildGroups(terms: terms, coloring: coloring)
    }

    // MARK: - Graph Construction

    /// Build conflict graph where edges represent non-QWC pairs.
    private static func buildConflictGraph(terms: PauliTerms) -> [[Int]] {
        let n = terms.count
        var adjacencyList = Array(repeating: [Int](), count: n)

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
    private static func colorGraphDSATUR(graph: [[Int]]) -> [Int] {
        let n = graph.count
        var colors = Array(repeating: -1, count: n)
        var saturationDegree = Array(repeating: 0, count: n)
        var neighborColors = Array(repeating: Set<Int>(), count: n)

        for _ in 0 ..< n {
            var selectedVertex = -1
            var maxSaturation = -1
            var maxDegree = -1

            for v in 0 ..< n where colors[v] == -1 {
                let saturation = saturationDegree[v]
                let degree = graph[v].count

                if saturation > maxSaturation ||
                    (saturation == maxSaturation && degree > maxDegree)
                {
                    selectedVertex = v
                    maxSaturation = saturation
                    maxDegree = degree
                }
            }

            guard selectedVertex != -1 else { break }

            let forbiddenColors = neighborColors[selectedVertex]
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
    private static func buildGroups(
        terms: PauliTerms,
        coloring: [Int]
    ) -> [QWCGroup] {
        var colorToTerms: [Int: PauliTerms] = [:]

        for (index, color) in coloring.enumerated() {
            if colorToTerms[color] == nil { colorToTerms[color] = [] }
            colorToTerms[color]?.append(terms[index])
        }

        var groups: [QWCGroup] = []

        for (_, groupTerms) in colorToTerms.sorted(by: { $0.key < $1.key }) {
            let pauliStrings = groupTerms.map(\.pauliString)

            guard let basis = PauliCommutation.measurementBasis(for: pauliStrings) else {
                preconditionFailure("Invalid QWC group - measurement basis could not be determined. " +
                    "This indicates a bug in graph coloring or conflict detection.")
            }

            groups.append(QWCGroup(terms: groupTerms, measurementBasis: basis))
        }

        return groups
    }

    // MARK: - Statistics

    /// Compute grouping statistics.
    public struct GroupingStats {
        public let numTerms: Int
        public let numGroups: Int
        public let reductionFactor: Double
        public let largestGroupSize: Int
        public let averageGroupSize: Double
        public let groupSizes: [Int]

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
    public static func statistics(for groups: [QWCGroup]) -> GroupingStats {
        let numGroups = groups.count
        let groupSizes = groups.map(\.terms.count)
        let numTerms = groupSizes.reduce(0, +)
        let largestGroupSize = groupSizes.max() ?? 0
        let averageGroupSize = numGroups > 0 ? Double(numTerms) / Double(numGroups) : 0.0
        let reductionFactor = numGroups > 0 ? Double(numTerms) / Double(numGroups) : 1.0

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
