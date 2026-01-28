// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for single-qubit and basic histogram rendering.
/// Validates ket labels, bar characters, counts, and percentages
/// for the simplest possible quantum measurement distributions.
@Suite("Histogram Basic Rendering")
struct HistogramBasicRenderingTests {
    @Test("1-qubit histogram with counts [100, 0] shows |0> at 100%")
    func singleQubitDeterministic() {
        let output = MeasurementHistogramFormatter.render([100, 0], qubits: 1)
        let lines = output.split(separator: "\n")
        #expect(lines.count == 2, "1-qubit histogram should have 2 lines for 2 basis states")
        #expect(output.contains("|0\u{27E9}"), "Output should contain ket label |0>")
        #expect(output.contains("|1\u{27E9}"), "Output should contain ket label |1>")
        #expect(output.contains("100"), "Output should contain count 100")
        #expect(output.contains("100.00%"), "Output should contain 100.00% for state |0>")
        #expect(output.contains("0.00%"), "Output should contain 0.00% for state |1>")
        #expect(output.contains("\u{2588}"), "Output should contain bar block characters")
    }

    @Test("Bell state histogram with counts [512, 0, 0, 488] shows 2-qubit kets")
    func bellStateHistogram() {
        let output = MeasurementHistogramFormatter.render([512, 0, 0, 488], qubits: 2)
        let lines = output.split(separator: "\n")
        #expect(lines.count == 4, "2-qubit histogram should have 4 lines for 4 basis states")
        #expect(output.contains("|00\u{27E9}"), "Output should contain ket label |00>")
        #expect(output.contains("|01\u{27E9}"), "Output should contain ket label |01>")
        #expect(output.contains("|10\u{27E9}"), "Output should contain ket label |10>")
        #expect(output.contains("|11\u{27E9}"), "Output should contain ket label |11>")
        #expect(output.contains("512"), "Output should contain count 512")
        #expect(output.contains("488"), "Output should contain count 488")
    }
}

/// Test suite for sort order behavior of histogram entries.
/// Validates lexicographic ordering by state and descending ordering
/// by count to confirm both SortOrder cases work correctly.
@Suite("Histogram Sort Order")
struct HistogramSortOrderTests {
    @Test("Sort by state produces lexicographic ket order")
    func sortByStateIsLexicographic() {
        let counts = [100, 300, 200, 400]
        let output = MeasurementHistogramFormatter.render(counts, qubits: 2, sort: .byState)
        let lines = output.split(separator: "\n").map(String.init)
        #expect(lines[0].contains("|00\u{27E9}"), "First line should be |00> in byState order")
        #expect(lines[1].contains("|01\u{27E9}"), "Second line should be |01> in byState order")
        #expect(lines[2].contains("|10\u{27E9}"), "Third line should be |10> in byState order")
        #expect(lines[3].contains("|11\u{27E9}"), "Fourth line should be |11> in byState order")
    }

    @Test("Sort by count produces descending count order")
    func sortByCountIsDescending() {
        let counts = [100, 300, 200, 400]
        let output = MeasurementHistogramFormatter.render(counts, qubits: 2, sort: .byCount)
        let lines = output.split(separator: "\n").map(String.init)
        #expect(lines[0].contains("|11\u{27E9}"), "First line should be |11> (400) in byCount order")
        #expect(lines[1].contains("|01\u{27E9}"), "Second line should be |01> (300) in byCount order")
        #expect(lines[2].contains("|10\u{27E9}"), "Third line should be |10> (200) in byCount order")
        #expect(lines[3].contains("|00\u{27E9}"), "Fourth line should be |00> (100) in byCount order")
    }

    @Test("SortOrder enum has byState and byCount cases")
    func sortOrderEnumCases() {
        let byState = MeasurementHistogramFormatter.SortOrder.byState
        let byCount = MeasurementHistogramFormatter.SortOrder.byCount
        #expect(byState != byCount, "byState and byCount should be distinct cases")
    }
}

/// Test suite for custom bar width scaling in histogram rendering.
/// Validates that bar length scales proportionally to the specified
/// barWidth parameter and max count gets the full bar width.
@Suite("Histogram Bar Width")
struct HistogramBarWidthTests {
    @Test("Custom bar width 10 scales bars to at most 10 block characters")
    func customBarWidth() {
        let counts = [1000, 500]
        let output = MeasurementHistogramFormatter.render(counts, qubits: 1, barWidth: 10)
        let lines = output.split(separator: "\n").map(String.init)
        let blockChar: Character = "\u{2588}"
        let maxBlocks = lines[0].count(where: { $0 == blockChar })
        #expect(maxBlocks == 10, "Max count should produce exactly barWidth=10 blocks")
        let halfBlocks = lines[1].count(where: { $0 == blockChar })
        #expect(halfBlocks == 5, "Half-max count should produce barWidth/2=5 blocks")
    }

    @Test("Bar width 20 scales proportionally")
    func barWidth20() {
        let counts = [400, 100, 0, 200]
        let output = MeasurementHistogramFormatter.render(counts, qubits: 2, barWidth: 20)
        let lines = output.split(separator: "\n").map(String.init)
        let blockChar: Character = "\u{2588}"
        let state00Blocks = lines[0].count(where: { $0 == blockChar })
        let state11Blocks = lines[3].count(where: { $0 == blockChar })
        #expect(state00Blocks == 20, "Max count (400) should produce exactly 20 blocks")
        #expect(state11Blocks == 10, "Half-max count (200) should produce 10 blocks")
    }
}

/// Test suite for threshold filtering of histogram entries.
/// Validates that entries with counts below the threshold are omitted,
/// reducing clutter in sparse quantum measurement distributions.
@Suite("Histogram Threshold Filtering")
struct HistogramThresholdFilteringTests {
    @Test("Threshold filters entries below the specified count")
    func thresholdFiltersLowCounts() {
        let counts = [500, 5, 3, 492]
        let output = MeasurementHistogramFormatter.render(counts, qubits: 2, threshold: 10)
        let lines = output.split(separator: "\n")
        #expect(lines.count == 2, "Only 2 states above threshold=10 should appear")
        #expect(output.contains("|00\u{27E9}"), "State |00> (500) should appear above threshold")
        #expect(output.contains("|11\u{27E9}"), "State |11> (492) should appear above threshold")
        #expect(!output.contains("|01\u{27E9}"), "State |01> (5) should be filtered by threshold")
        #expect(!output.contains("|10\u{27E9}"), "State |10> (3) should be filtered by threshold")
    }

    @Test("Threshold of 0 includes all entries including zero counts")
    func thresholdZeroIncludesAll() {
        let counts = [100, 0, 0, 100]
        let output = MeasurementHistogramFormatter.render(counts, qubits: 2, threshold: 0)
        let lines = output.split(separator: "\n")
        #expect(lines.count == 4, "Threshold=0 should include all 4 states")
    }

    @Test("Threshold of 1 excludes zero-count entries")
    func thresholdOneExcludesZeroCounts() {
        let counts = [100, 0, 0, 100]
        let output = MeasurementHistogramFormatter.render(counts, qubits: 2, threshold: 1)
        let lines = output.split(separator: "\n")
        #expect(lines.count == 2, "Threshold=1 should exclude 2 zero-count states")
    }
}

/// Test suite for expected values overlay and chi-squared display.
/// Validates that [exp: N] annotations and chi-squared summary lines
/// appear when expected probabilities are provided.
@Suite("Histogram Expected Values and Chi-Squared")
struct HistogramExpectedValuesTests {
    @Test("Expected values overlay shows [exp: N] for each entry")
    func expectedValuesOverlay() {
        let counts = [500, 500]
        let expected = [0.5, 0.5]
        let output = MeasurementHistogramFormatter.render(counts, qubits: 1, expected: expected)
        #expect(output.contains("[exp: 500]"), "Output should contain expected count annotation [exp: 500]")
    }

    @Test("Chi-squared line appears when expected values are provided")
    func chiSquaredLineAppears() {
        let counts = [512, 0, 0, 488]
        let expected = [0.5, 0.0, 0.0, 0.5]
        let output = MeasurementHistogramFormatter.render(counts, qubits: 2, expected: expected)
        #expect(output.contains("\u{03C7}\u{00B2} ="), "Output should contain chi-squared symbol and value")
        #expect(output.contains("df ="), "Output should contain degrees of freedom")
    }

    @Test("Chi-squared is zero for perfect match")
    func chiSquaredZeroForPerfectMatch() {
        let counts = [500, 500]
        let expected = [0.5, 0.5]
        let output = MeasurementHistogramFormatter.render(counts, qubits: 1, expected: expected)
        #expect(output.contains("\u{03C7}\u{00B2} = 0.000"), "Chi-squared should be 0.000 for perfect match")
    }

    @Test("Chi-squared is non-zero for imperfect match")
    func chiSquaredNonZeroForMismatch() {
        let counts = [600, 400]
        let expected = [0.5, 0.5]
        let output = MeasurementHistogramFormatter.render(counts, qubits: 1, expected: expected)
        #expect(output.contains("\u{03C7}\u{00B2} ="), "Output should contain chi-squared for imperfect match")
        #expect(!output.contains("\u{03C7}\u{00B2} = 0.000"), "Chi-squared should not be 0.000 for mismatched counts")
    }

    @Test("Expected count rounds to nearest integer in display")
    func expectedCountRoundedDisplay() {
        let counts = [700, 300]
        let expected = [0.6, 0.4]
        let output = MeasurementHistogramFormatter.render(counts, qubits: 1, expected: expected)
        #expect(output.contains("[exp: 600]"), "0.6 * 1000 = 600 should appear as [exp: 600]")
        #expect(output.contains("[exp: 400]"), "0.4 * 1000 = 400 should appear as [exp: 400]")
    }

    @Test("No expected values means no [exp:] or chi-squared lines")
    func noExpectedValuesNoOverlay() {
        let counts = [500, 500]
        let output = MeasurementHistogramFormatter.render(counts, qubits: 1)
        #expect(!output.contains("[exp:"), "Output should not contain [exp:] when no expected values given")
        #expect(!output.contains("\u{03C7}\u{00B2}"), "Output should not contain chi-squared when no expected values given")
    }
}

/// Test suite for uniform distribution and all-zero edge cases.
/// Validates correct rendering when all counts are equal or when
/// every count is zero, testing degenerate quantum states.
@Suite("Histogram Edge Cases")
struct HistogramEdgeCasesTests {
    @Test("Uniform distribution renders all entries with equal percentages")
    func uniformDistribution() {
        let counts = [250, 250, 250, 250]
        let output = MeasurementHistogramFormatter.render(counts, qubits: 2)
        let lines = output.split(separator: "\n")
        #expect(lines.count == 4, "Uniform distribution should show all 4 states")
        for line in lines {
            #expect(line.contains("25.00%"), "Each state should show 25.00% in uniform distribution")
            #expect(line.contains("250"), "Each state should show count 250 in uniform distribution")
        }
    }

    @Test("All counts zero renders entries with 0.00% and no bars")
    func allCountsZero() {
        let counts = [0, 0]
        let output = MeasurementHistogramFormatter.render(counts, qubits: 1)
        let lines = output.split(separator: "\n")
        #expect(lines.count == 2, "All-zero histogram should still have 2 lines")
        for line in lines {
            #expect(line.contains("0.00%"), "All-zero histogram should show 0.00% for each entry")
            #expect(!line.contains("\u{2588}"), "All-zero histogram should have no bar characters")
        }
    }

    @Test("3-qubit histogram produces 8 entries with 3-bit ket labels")
    func threeQubitHistogram() {
        let counts = [100, 0, 0, 0, 0, 0, 0, 100]
        let output = MeasurementHistogramFormatter.render(counts, qubits: 3)
        let lines = output.split(separator: "\n")
        #expect(lines.count == 8, "3-qubit histogram should have 8 lines")
        #expect(output.contains("|000\u{27E9}"), "3-qubit histogram should contain |000> ket label")
        #expect(output.contains("|111\u{27E9}"), "3-qubit histogram should contain |111> ket label")
    }
}
