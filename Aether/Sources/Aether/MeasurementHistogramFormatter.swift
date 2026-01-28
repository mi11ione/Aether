// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Measurement outcome histogram renderer for text-based visualization of quantum measurement distributions.
///
/// Formats an array of per-basis-state counts into a horizontal bar chart string using full-block
/// Unicode characters (U+2588). Bars scale proportionally to ``barWidth``, entries can be sorted
/// by state index or count, and sparse distributions can be filtered via ``threshold``. When an
/// ``expected`` probability array is supplied, each line appends the theoretical count and a
/// chi-squared goodness-of-fit summary is appended at the bottom.
///
/// **Example:**
/// ```swift
/// let counts = [512, 0, 0, 488]
/// let text = MeasurementHistogramFormatter.render(counts, qubits: 2)
/// ```
///
/// - SeeAlso: ``Measurement/histogram(outcomes:qubits:)``
/// - SeeAlso: ``ChiSquaredResult``
public enum MeasurementHistogramFormatter: Sendable {
    /// Sort order for histogram entries.
    ///
    /// Determines whether rows appear in ascending basis-state index order or
    /// descending count order for quick identification of dominant outcomes.
    ///
    /// - SeeAlso: ``MeasurementHistogramFormatter/render(_:qubits:sort:barWidth:threshold:expected:)``
    @frozen public enum SortOrder: Sendable {
        case byState
        case byCount
    }

    /// Render a measurement outcome histogram as a multi-line text bar chart.
    ///
    /// Produces a human-readable histogram where each row shows a ket label, a proportional
    /// bar of full-block characters, the raw count, and the percentage of total shots. Rows
    /// with counts below ``threshold`` are omitted for clarity in sparse distributions. When
    /// ``expected`` probabilities are provided, each row appends the theoretical expected count
    /// and a chi-squared summary line is appended at the end.
    ///
    /// **Example:**
    /// ```swift
    /// let counts = [512, 0, 0, 488]
    /// let histogram = MeasurementHistogramFormatter.render(counts, qubits: 2, sort: .byCount)
    /// ```
    ///
    /// - Parameters:
    ///   - counts: Array of measurement counts with exactly 2^qubits entries where index i holds the count for basis state |i⟩
    ///   - qubits: Number of qubits determining the state space size and ket label width
    ///   - sort: Row ordering, either ascending by state index or descending by count
    ///   - barWidth: Maximum bar length in characters for the most frequent outcome
    ///   - threshold: Minimum count for a row to appear in the output
    ///   - expected: Optional array of probabilities for computing expected counts and chi-squared statistic
    /// - Returns: Multi-line histogram string ready for display
    /// - Complexity: O(2^n) where n is the number of qubits
    /// - Precondition: `qubits` must be positive.
    /// - Precondition: `barWidth` must be positive.
    /// - Precondition: `counts` must have exactly 2^qubits elements.
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    public static func render(
        _ counts: [Int],
        qubits: Int,
        sort: SortOrder = .byState,
        barWidth: Int = 40,
        threshold: Int = 0,
        expected: [Double]? = nil,
    ) -> String {
        ValidationUtilities.validatePositiveInt(qubits, name: "qubits")
        ValidationUtilities.validatePositiveInt(barWidth, name: "barWidth")
        ValidationUtilities.validateArrayCount(counts, expected: 1 << qubits, name: "counts")

        let stateCount = 1 << qubits
        let totalShots = computeTotalShots(counts)
        let maxCount = computeMaxCount(counts)

        var indices = buildFilteredIndices(counts, stateCount: stateCount, threshold: threshold)
        sortIndices(&indices, counts: counts, sort: sort)

        var result = String()
        result.reserveCapacity(indices.count * (qubits + barWidth + 40))

        for index in indices {
            let ket = formatKet(index, qubits: qubits)
            let bar = formatBar(counts[index], maxCount: maxCount, barWidth: barWidth)
            let padding = formatPadding(counts[index], maxCount: maxCount, barWidth: barWidth)
            let percentage = formatPercentage(counts[index], totalShots: totalShots)

            result += "|"
            result += ket
            result += "⟩ : "
            result += padding
            result += bar
            result += " "
            result += String(counts[index])
            result += " ("
            result += percentage
            result += "%)"

            if let expected {
                let expectedCount = expected[index] * Double(totalShots)
                result += " [exp: "
                result += String(Int(expectedCount.rounded()))
                result += "]"
            }

            result += "\n"
        }

        if let expected {
            let chiSquaredLine = formatChiSquared(counts, expected: expected, totalShots: totalShots)
            result += chiSquaredLine
            result += "\n"
        }

        if result.hasSuffix("\n") {
            result.removeLast()
        }

        return result
    }

    /// Compute total shots from counts array.
    @_effects(readonly)
    @inlinable
    static func computeTotalShots(_ counts: [Int]) -> Int {
        var total = 0
        for count in counts {
            total &+= count
        }
        return total
    }

    /// Compute maximum count from counts array.
    @_effects(readonly)
    @inlinable
    static func computeMaxCount(_ counts: [Int]) -> Int {
        var maxVal = 0
        for count in counts {
            if count > maxVal {
                maxVal = count
            }
        }
        return maxVal
    }

    /// Build array of state indices that pass the threshold filter.
    @_effects(readonly)
    @inlinable
    static func buildFilteredIndices(_ counts: [Int], stateCount: Int, threshold: Int) -> [Int] {
        var indices = [Int]()
        indices.reserveCapacity(stateCount)
        for i in 0 ..< stateCount {
            if counts[i] >= threshold {
                indices.append(i)
            }
        }
        return indices
    }

    /// Sort indices according to the specified sort order.
    @inlinable
    static func sortIndices(_ indices: inout [Int], counts: [Int], sort: SortOrder) {
        switch sort {
        case .byState:
            break
        case .byCount:
            indices.sort { counts[$0] > counts[$1] }
        }
    }

    /// Format a basis state index as a little-endian binary ket string.
    @_effects(readonly)
    @inlinable
    static func formatKet(_ index: Int, qubits: Int) -> String {
        var ket = String()
        ket.reserveCapacity(qubits)
        for bit in stride(from: qubits - 1, through: 0, by: -1) {
            ket += ((index >> bit) & 1) == 1 ? "1" : "0"
        }
        return ket
    }

    /// Format a bar string of full-block characters proportional to count.
    @_effects(readonly)
    @inlinable
    static func formatBar(_ count: Int, maxCount: Int, barWidth: Int) -> String {
        let length = barLength(count, maxCount: maxCount, barWidth: barWidth)
        var bar = String()
        bar.reserveCapacity(length * 3)
        for _ in 0 ..< length {
            bar += "\u{2588}"
        }
        return bar
    }

    /// Format padding spaces to align bars to the right edge.
    @_effects(readonly)
    @inlinable
    static func formatPadding(_ count: Int, maxCount: Int, barWidth: Int) -> String {
        let length = barLength(count, maxCount: maxCount, barWidth: barWidth)
        let paddingCount = barWidth - length
        var padding = String()
        padding.reserveCapacity(paddingCount)
        for _ in 0 ..< paddingCount {
            padding += " "
        }
        return padding
    }

    /// Compute the scaled bar length for a given count.
    @_effects(readonly)
    @inlinable
    static func barLength(_ count: Int, maxCount: Int, barWidth: Int) -> Int {
        guard maxCount > 0 else { return 0 }
        return (count * barWidth) / maxCount
    }

    /// Format count as a percentage string with two decimal places.
    @_effects(readonly)
    @inlinable
    static func formatPercentage(_ count: Int, totalShots: Int) -> String {
        guard totalShots > 0 else { return "0.00" }
        let pct = Double(count) / Double(totalShots) * 100.0
        return String(format: "%.2f", pct)
    }

    /// Format chi-squared summary line from observed counts and expected probabilities.
    @_effects(readonly)
    @inlinable
    static func formatChiSquared(_ counts: [Int], expected: [Double], totalShots: Int) -> String {
        var chiSq = 0.0
        var testedBins = 0

        for i in 0 ..< counts.count {
            let expectedCount = expected[i] * Double(totalShots)
            guard expectedCount > 0 else { continue }
            let diff = Double(counts[i]) - expectedCount
            chiSq += (diff * diff) / expectedCount
            testedBins += 1
        }

        let df = max(testedBins - 1, 0)
        return "χ² = " + String(format: "%.3f", chiSq) + ", df = " + String(df)
    }
}
