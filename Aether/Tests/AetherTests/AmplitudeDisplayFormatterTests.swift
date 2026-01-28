// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Validates rendering of single-qubit and multi-qubit states,
/// verifying ket labels, probability values, phase angles, and
/// header formatting in the amplitude display formatter output.
@Suite("Single and Multi-Qubit State Rendering")
struct AmplitudeDisplayFormatterBasicRenderingTests {
    @Test("Single qubit ground state renders one significant entry with |0> at probability 1.0")
    func singleQubitGroundState() {
        let state = QuantumState(qubits: 1)
        let output = AmplitudeDisplayFormatter.render(state)

        #expect(output.contains("|0⟩"), "Output should contain ket label |0>")
        #expect(output.contains("1.00000"), "Probability of |0> should be 1.00000")
        #expect(output.contains("1 qubits"), "Header should show 1 qubit")
        #expect(output.contains("1 significant"), "Header should show 1 significant state")
        #expect(!output.contains("|1⟩"), "Output should not contain |1> since its amplitude is zero")
    }

    @Test("Bell state renders two entries with equal probability 0.5")
    func bellStateRender() {
        let inv = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(inv, 0), Complex(0, 0), Complex(0, 0), Complex(inv, 0),
        ])
        let output = AmplitudeDisplayFormatter.render(bell)

        #expect(output.contains("|00⟩"), "Output should contain ket label |00>")
        #expect(output.contains("|11⟩"), "Output should contain ket label |11>")
        #expect(output.contains("0.50000"), "Both states should show probability 0.50000")
        #expect(output.contains("2 qubits"), "Header should show 2 qubits")
        #expect(output.contains("2 significant"), "Header should show 2 significant states")
    }

    @Test("Uniform 2-qubit superposition renders all four entries")
    func uniformSuperposition() {
        let amp = Complex(0.5, 0.0)
        let state = QuantumState(qubits: 2, amplitudes: [amp, amp, amp, amp])
        let output = AmplitudeDisplayFormatter.render(state)

        #expect(output.contains("|00⟩"), "Output should contain |00>")
        #expect(output.contains("|01⟩"), "Output should contain |01>")
        #expect(output.contains("|10⟩"), "Output should contain |10>")
        #expect(output.contains("|11⟩"), "Output should contain |11>")
        #expect(output.contains("4 significant"), "Header should show 4 significant states")
        #expect(output.contains("0.25000"), "Each state should have probability 0.25000")
    }
}

/// Validates threshold filtering behavior that omits amplitudes
/// below the configured probability threshold, and maxEntries
/// truncation that limits output to top-k entries.
@Suite("Threshold Filtering and MaxEntries Truncation")
struct AmplitudeDisplayFormatterFilteringTests {
    @Test("Amplitudes below threshold are omitted from output")
    func thresholdFiltering() {
        let state = QuantumState(qubits: 2, amplitudes: [
            Complex(0.999, 0.0), Complex(0.001, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0),
        ])
        let output = AmplitudeDisplayFormatter.render(state, threshold: 0.01)

        #expect(output.contains("|00⟩"), "State |00> with high probability should be present")
        #expect(!output.contains("|01⟩"), "State |01> below threshold should be omitted")
        #expect(output.contains("1 significant"), "Only 1 state should be significant at threshold 0.01")
    }

    @Test("maxEntries limits displayed entries and shows remaining count")
    func maxEntriesTopK() {
        let amp = Complex(0.5, 0.0)
        let state = QuantumState(qubits: 2, amplitudes: [amp, amp, amp, amp])
        let output = AmplitudeDisplayFormatter.render(state, maxEntries: 2)

        #expect(output.contains("showing top 2 of 4"), "Header should indicate truncation")
        #expect(output.contains("... and 2 more states"), "Footer should show remaining count")
    }

    @Test("maxEntries equal to significant count shows all without truncation message")
    func maxEntriesExactMatch() {
        let inv = 1.0 / sqrt(2.0)
        let state = QuantumState(qubits: 1, amplitudes: [
            Complex(inv, 0.0), Complex(inv, 0.0),
        ])
        let output = AmplitudeDisplayFormatter.render(state, maxEntries: 2)

        #expect(!output.contains("... and"), "No truncation message when maxEntries matches significant count")
        #expect(output.contains("2 significant"), "Header should show 2 significant states without truncation")
    }

    @Test("Zero threshold includes all non-zero amplitudes")
    func zeroThreshold() {
        let state = QuantumState(qubits: 1, amplitudes: [
            Complex(0.99, 0.0), Complex(0.01, 0.0),
        ])
        let output = AmplitudeDisplayFormatter.render(state, threshold: 0.0)

        #expect(output.contains("|0⟩"), "State |0> should be present with zero threshold")
        #expect(output.contains("|1⟩"), "State |1> should be present with zero threshold")
    }
}

/// Validates phase angle display in both radians and degrees,
/// including correct unit suffixes, angle values for known states,
/// and complex amplitude notation with real and imaginary parts.
@Suite("Phase Display and Complex Amplitude Formatting")
struct AmplitudeDisplayFormatterPhaseTests {
    @Test("Default phase format is radians with 'rad' suffix")
    func phaseRadiansDefault() {
        let state = QuantumState(qubits: 1)
        let output = AmplitudeDisplayFormatter.render(state)

        #expect(output.contains("rad"), "Default phase format should show 'rad' suffix")
        #expect(output.contains("phase="), "Output should contain phase= prefix")
    }

    @Test("Phase format degrees shows degree symbol")
    func phaseDegreesFormat() {
        let state = QuantumState(qubits: 1)
        let output = AmplitudeDisplayFormatter.render(state, phaseFormat: .degrees)

        #expect(output.contains("°"), "Degrees format should show degree symbol")
        #expect(!output.contains("rad"), "Degrees format should not show 'rad' suffix")
    }

    @Test("Pure real positive amplitude has zero phase in radians")
    func zeroPhaseRadians() {
        let state = QuantumState(qubits: 1)
        let output = AmplitudeDisplayFormatter.render(state)

        #expect(output.contains("phase=0.0000 rad"), "Real positive amplitude should have phase 0.0000 rad")
    }

    @Test("Pure real positive amplitude has 0.00 degrees phase")
    func zeroPhaseDegrees() {
        let state = QuantumState(qubits: 1)
        let output = AmplitudeDisplayFormatter.render(state, phaseFormat: .degrees)

        #expect(output.contains("phase=0.00°"), "Real positive amplitude should have phase 0.00 degrees")
    }

    @Test("Complex phase with imaginary component shows non-zero angle")
    func complexPhaseNonZero() {
        let state = QuantumState(qubits: 1, amplitudes: [
            Complex(0.0, 1.0), Complex(0.0, 0.0),
        ])
        let output = AmplitudeDisplayFormatter.render(state)

        #expect(output.contains("phase=1.5708 rad"), "Pure imaginary amplitude should have phase pi/2 ~ 1.5708 rad")
    }

    @Test("Complex phase in degrees shows 90 degrees for pure imaginary")
    func complexPhaseDegrees() {
        let state = QuantumState(qubits: 1, amplitudes: [
            Complex(0.0, 1.0), Complex(0.0, 0.0),
        ])
        let output = AmplitudeDisplayFormatter.render(state, phaseFormat: .degrees)

        #expect(output.contains("phase=90.00°"), "Pure imaginary amplitude should have phase 90.00 degrees")
    }

    @Test("Complex amplitude notation shows real and imaginary parts")
    func complexAmplitudeNotation() {
        let inv = 1.0 / sqrt(2.0)
        let state = QuantumState(qubits: 1, amplitudes: [
            Complex(inv, 0.0), Complex(0.0, inv),
        ])
        let output = AmplitudeDisplayFormatter.render(state)

        #expect(output.contains("(0.7071+0.0000i)"), "Real-only amplitude should show +0.0000i")
        #expect(output.contains("(0.0000+0.7071i)"), "Imaginary-only amplitude should show 0.0000+ real part")
    }

    @Test("Negative imaginary component shows minus sign in notation")
    func negativeImaginaryNotation() {
        let state = QuantumState(qubits: 1, amplitudes: [
            Complex(1.0 / sqrt(2.0), -1.0 / sqrt(2.0)), Complex(0.0, 0.0),
        ])
        let output = AmplitudeDisplayFormatter.render(state)

        #expect(output.contains("0.7071-0.7071i"), "Negative imaginary should show minus sign")
    }
}

/// Validates sorting behavior by descending probability, PhaseUnit
/// enum case coverage, and header content including qubit count
/// and significant state count display.
@Suite("Sorting, Header, and PhaseUnit Enum")
struct AmplitudeDisplayFormatterSortingAndEnumTests {
    @Test("Entries are sorted by descending probability")
    func sortedByDescendingProbability() {
        let state = QuantumState(qubits: 2, amplitudes: [
            Complex(0.2, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.9, 0.0),
        ])
        let output = AmplitudeDisplayFormatter.render(state)
        let lines = output.split(separator: "\n")

        let dataLines = lines.dropFirst()
        #expect(dataLines.count >= 2, "Should have at least 2 data lines for 2 significant states")

        let firstLine = String(dataLines.first!)
        let secondLine = String(dataLines.dropFirst().first!)
        #expect(firstLine.contains("|11⟩"), "First entry should be |11> with highest probability")
        #expect(secondLine.contains("|00⟩"), "Second entry should be |00> with lower probability")
    }

    @Test("Header shows correct qubit count for 3-qubit state")
    func headerQubitCount() {
        let state = QuantumState(qubits: 3)
        let output = AmplitudeDisplayFormatter.render(state)

        #expect(output.hasPrefix("Quantum State (3 qubits"), "Header should start with 'Quantum State (3 qubits'")
    }

    @Test("PhaseUnit.radians and .degrees are distinct enum cases")
    func phaseUnitEnumCases() {
        let radians = AmplitudeDisplayFormatter.PhaseUnit.radians
        let degrees = AmplitudeDisplayFormatter.PhaseUnit.degrees

        let state = QuantumState(qubits: 1)
        let radOutput = AmplitudeDisplayFormatter.render(state, phaseFormat: radians)
        let degOutput = AmplitudeDisplayFormatter.render(state, phaseFormat: degrees)

        #expect(radOutput.contains("rad"), "Radians output should contain 'rad'")
        #expect(degOutput.contains("°"), "Degrees output should contain degree symbol")
        #expect(!radOutput.contains("°"), "Radians output should not contain degree symbol")
        #expect(!degOutput.contains("rad"), "Degrees output should not contain 'rad'")
    }

    @Test("State with complex phases shows non-trivial phase angles")
    func stateWithComplexPhases() {
        let state = QuantumState(qubits: 2, amplitudes: [
            Complex(0.5, 0.5), Complex(0.5, -0.5), Complex(0.0, 0.0), Complex(0.0, 0.0),
        ])
        let output = AmplitudeDisplayFormatter.render(state)

        #expect(output.contains("|00⟩"), "State |00> should appear for amplitude (0.5+0.5i)")
        #expect(output.contains("|01⟩"), "State |01> should appear for amplitude (0.5-0.5i)")
        #expect(output.contains("phase=0.7854 rad"), "Phase of (0.5+0.5i) should be pi/4 ~ 0.7854 rad")
        #expect(output.contains("phase=-0.7854 rad"), "Phase of (0.5-0.5i) should be -pi/4 ~ -0.7854 rad")
    }

    @Test("Truncated header format differs from full header format")
    func truncatedVsFullHeader() {
        let amp = Complex(0.5, 0.0)
        let state = QuantumState(qubits: 2, amplitudes: [amp, amp, amp, amp])

        let fullOutput = AmplitudeDisplayFormatter.render(state)
        let truncOutput = AmplitudeDisplayFormatter.render(state, maxEntries: 1)

        #expect(fullOutput.contains("4 significant):"), "Full header should end with 'N significant):'")
        #expect(truncOutput.contains("showing top 1 of 4 significant):"), "Truncated header should show 'showing top K of N significant):'")
    }
}
