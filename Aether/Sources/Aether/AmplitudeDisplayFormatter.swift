// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Probability amplitude display formatter for quantum state visualization.
///
/// Renders a ``QuantumState`` as human-readable text showing per-basis-state probabilities,
/// phase angles, and complex amplitude components in ket notation. Filters insignificant
/// amplitudes below a configurable threshold, sorts by descending probability, and optionally
/// truncates output to a maximum entry count for compact display of large state spaces.
///
/// **Example:**
/// ```swift
/// let bell = QuantumState(qubits: 2, amplitudes: [
///     Complex(1/sqrt(2), 0), Complex(0, 0), Complex(0, 0), Complex(1/sqrt(2), 0)
/// ])
/// let text = AmplitudeDisplayFormatter.render(bell)
/// ```
///
/// - SeeAlso: ``QuantumState``
/// - SeeAlso: ``Complex``
public enum AmplitudeDisplayFormatter: Sendable {
    @frozen public enum PhaseUnit: Sendable {
        case radians
        case degrees
    }

    /// Render a quantum state as a formatted amplitude table.
    ///
    /// Produces a multi-line string with a header summarizing qubit count and significant state
    /// count, followed by one line per basis state showing probability (|amplitude|^2), phase
    /// angle (atan2 of imaginary over real), and the raw complex amplitude in parentheses. States
    /// are sorted by descending probability. Only states whose probability meets or exceeds the
    /// threshold are included. When maxEntries is less than the number of significant states,
    /// output is truncated and a summary line indicates how many additional states were omitted.
    ///
    /// - Precondition: `threshold` must be non-negative.
    /// - Precondition: `maxEntries` must be positive.
    /// - Parameters:
    ///   - state: Quantum state to render
    ///   - maxEntries: Maximum number of basis state lines to include (default unlimited)
    ///   - threshold: Minimum probability |amplitude|^2 for a state to appear (default 1e-6)
    ///   - phaseFormat: Unit for phase angle display, radians or degrees (default radians)
    /// - Returns: Formatted multi-line string representation of the quantum state amplitudes
    /// - Complexity: O(2^n log 2^n) where n is the qubit count, dominated by sorting
    @_optimize(speed)
    @_effects(readonly)
    @inlinable
    public static func render(
        _ state: QuantumState,
        maxEntries: Int = .max,
        threshold: Double = 1e-6,
        phaseFormat: PhaseUnit = .radians,
    ) -> String {
        ValidationUtilities.validateNonNegativeDouble(threshold, name: "Threshold")
        ValidationUtilities.validatePositiveInt(maxEntries, name: "maxEntries")

        let amplitudes = state.amplitudes
        let qubitCount = state.qubits
        let stateCount = amplitudes.count

        var significantEntries: [(index: Int, probability: Double, phase: Double, real: Double, imaginary: Double)] = []
        significantEntries.reserveCapacity(min(stateCount, 64))

        for i in 0 ..< stateCount {
            let amp = amplitudes[i]
            let prob = amp.magnitudeSquared
            if prob >= threshold {
                let ph = Foundation.atan2(amp.imaginary, amp.real)
                significantEntries.append((i, prob, ph, amp.real, amp.imaginary))
            }
        }

        significantEntries.sort { $0.probability > $1.probability }

        let significantCount = significantEntries.count
        let displayCount = min(maxEntries, significantCount)

        let header = if displayCount < significantCount {
            "Quantum State (\(qubitCount) qubits, showing top \(displayCount) of \(significantCount) significant):"
        } else {
            "Quantum State (\(qubitCount) qubits, \(significantCount) significant):"
        }

        let estimatedLineLength = qubitCount + 60
        var result = String()
        result.reserveCapacity(header.count + 1 + displayCount * estimatedLineLength + 32)

        result += header

        for entryIndex in 0 ..< displayCount {
            let entry = significantEntries[entryIndex]
            let binaryStr = formatBinaryString(entry.index, width: qubitCount)
            let probStr = formatFixed5(entry.probability)
            let phaseStr = formatPhase(entry.phase, unit: phaseFormat)
            let complexStr = formatComplex(entry.real, entry.imaginary)

            result += "\n"
            result += "|"
            result += binaryStr
            result += "⟩ : "
            result += probStr
            result += "  phase="
            result += phaseStr
            result += "  ("
            result += complexStr
            result += ")"
        }

        if displayCount < significantCount {
            let remaining = significantCount - displayCount
            result += "\n... and \(remaining) more states"
        }

        return result
    }

    /// Format an integer as zero-padded binary string of given width.
    @_effects(readonly)
    @inlinable
    static func formatBinaryString(_ value: Int, width: Int) -> String {
        var chars = [Character]()
        chars.reserveCapacity(width)
        for bit in stride(from: width - 1, through: 0, by: -1) {
            chars.append((value >> bit) & 1 == 1 ? "1" : "0")
        }
        return String(chars)
    }

    /// Format a Double to 5 decimal places.
    @_effects(readonly)
    @inlinable
    static func formatFixed5(_ value: Double) -> String {
        String(format: "%.5f", value)
    }

    /// Format phase angle with appropriate unit suffix.
    @_effects(readonly)
    @inlinable
    static func formatPhase(_ radians: Double, unit: PhaseUnit) -> String {
        switch unit {
        case .radians:
            return String(format: "%.4f", radians) + " rad"
        case .degrees:
            let degrees = radians * 180.0 / .pi
            return String(format: "%.2f", degrees) + "°"
        }
    }

    /// Format complex number as real+imag_i with 4 decimal places.
    @_effects(readonly)
    @inlinable
    static func formatComplex(_ real: Double, _ imaginary: Double) -> String {
        let realStr = String(format: "%.4f", real)
        let imagAbs = Foundation.fabs(imaginary)
        let imagStr = String(format: "%.4f", imagAbs)
        if imaginary >= 0 {
            return realStr + "+" + imagStr + "i"
        } else {
            return realStr + "-" + imagStr + "i"
        }
    }
}
