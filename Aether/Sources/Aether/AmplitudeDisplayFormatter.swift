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
    /// Unit for displaying phase angles in amplitude output.
    ///
    /// Controls whether phase values are rendered as radians (e.g., `0.7854 rad`) or
    /// degrees (e.g., `45.00°`). Radians are the default for mathematical precision,
    /// while degrees offer intuitive readability for common angles.
    ///
    /// **Example:**
    /// ```swift
    /// let state = QuantumState(qubits: 1, amplitudes: [
    ///     Complex(1/sqrt(2), 0), Complex(0, 1/sqrt(2))
    /// ])
    /// let radOutput = AmplitudeDisplayFormatter.render(state, phaseUnit: .radians)
    /// let degOutput = AmplitudeDisplayFormatter.render(state, phaseUnit: .degrees)
    /// ```
    ///
    /// - SeeAlso: ``AmplitudeDisplayFormatter/render(_:maxEntries:threshold:phaseUnit:)``
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
    /// **Example:**
    /// ```swift
    /// let bell = QuantumState(qubits: 2, amplitudes: [
    ///     Complex(1/sqrt(2), 0), Complex(0, 0), Complex(0, 0), Complex(1/sqrt(2), 0)
    /// ])
    /// let output = AmplitudeDisplayFormatter.render(bell, maxEntries: 10, threshold: 1e-4)
    /// print(output)
    /// ```
    ///
    /// - Precondition: `threshold` must be non-negative.
    /// - Precondition: `maxEntries` must be positive.
    /// - Parameters:
    ///   - state: Quantum state to render
    ///   - maxEntries: Maximum number of basis state lines to include (default unlimited)
    ///   - threshold: Minimum probability |amplitude|^2 for a state to appear (default 1e-6)
    ///   - phaseUnit: Unit for phase angle display, radians or degrees (default radians)
    /// - Returns: Formatted multi-line string representation of the quantum state amplitudes
    /// - Complexity: O(2^n log 2^n) where n is the qubit count, dominated by sorting
    @_optimize(speed)
    @_effects(readonly)
    @inlinable public static func render(
        _ state: QuantumState,
        maxEntries: Int = .max,
        threshold: Double = 1e-6,
        phaseUnit: PhaseUnit = .radians,
    ) -> String {
        ValidationUtilities.validateNonNegativeDouble(threshold, name: "Threshold")
        ValidationUtilities.validatePositiveInt(maxEntries, name: "maxEntries")

        let amplitudes = state.amplitudes
        let qubitCount = state.qubits
        let stateCount = amplitudes.count

        var significantEntries: [(index: Int, probability: Double, phase: Double, real: Double, imaginary: Double)] = []
        significantEntries.reserveCapacity(min(stateCount, 64))

        amplitudes.withUnsafeBufferPointer { ampBuffer in
            for i in 0 ..< stateCount {
                let amp = ampBuffer[i]
                let prob = amp.magnitudeSquared
                if prob >= threshold {
                    let ph = Foundation.atan2(amp.imaginary, amp.real)
                    significantEntries.append((i, prob, ph, amp.real, amp.imaginary))
                }
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

        var lines = [String]()
        lines.reserveCapacity(displayCount + 2)

        lines.append(header)

        for entryIndex in 0 ..< displayCount {
            let entry = significantEntries[entryIndex]
            let binaryStr = binaryString(for: entry.index, width: qubitCount)
            let probStr = formatProbability(entry.probability)
            let phaseStr = formatPhase(entry.phase, as: phaseUnit)
            let complexStr = formatComplex(real: entry.real, imaginary: entry.imaginary)

            lines.append("|\(binaryStr)⟩ : \(probStr)  phase=\(phaseStr)  (\(complexStr))")
        }

        if displayCount < significantCount {
            let remaining = significantCount - displayCount
            lines.append("... and \(remaining) more states")
        }

        return lines.joined(separator: "\n")
    }
}

extension AmplitudeDisplayFormatter {
    /// Character lookup table for binary digit formatting.
    @usableFromInline static let digitCharacters: [Character] = ["0", "1"]
    /// Conversion multiplier from radians to degrees.
    @usableFromInline static let radiansToDegrees = 180.0 / .pi
    /// Formats an integer as a fixed-width binary string.
    @_effects(readonly)
    @inlinable static func binaryString(for value: Int, width: Int) -> String {
        guard width > 0 else { return "" }
        let chars = [Character](unsafeUninitializedCapacity: width) { buffer, count in
            for i in 0 ..< width {
                let bit = (width - 1) - i
                buffer[i] = Self.digitCharacters[(value >> bit) & 1]
            }
            count = width
        }
        return String(chars)
    }

    /// Formats a probability value with 5 decimal places.
    @_effects(readonly)
    @inlinable static func formatProbability(_ value: Double) -> String {
        String(format: "%.5f", value)
    }

    /// Formats a phase angle in the specified unit.
    @_effects(readonly)
    @inlinable static func formatPhase(_ radians: Double, as unit: PhaseUnit) -> String {
        switch unit {
        case .radians:
            return String(format: "%.4f", radians) + " rad"
        case .degrees:
            let degrees = radians * Self.radiansToDegrees
            return String(format: "%.2f", degrees) + "°"
        }
    }

    /// Formats a complex number as "real±imag*i".
    @_effects(readonly)
    @inlinable static func formatComplex(real: Double, imaginary: Double) -> String {
        let realStr = String(format: "%.4f", real)
        let imagAbs = Foundation.fabs(imaginary)
        let imagStr = String(format: "%.4f", imagAbs)
        let sign: Character = imaginary >= 0 ? "+" : "-"
        return "\(realStr)\(sign)\(imagStr)i"
    }
}
