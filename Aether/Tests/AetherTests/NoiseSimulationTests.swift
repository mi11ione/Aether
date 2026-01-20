// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for Kraus operator completeness relation validation.
/// Validates Σᵢ Kᵢ†Kᵢ = I (trace preservation) for all noise channels.
/// Tests depolarizing, amplitude damping, phase damping, and bit/phase flip channels.
@Suite("Noise Channel Kraus Completeness")
struct NoiseChannelKrausCompletenessTests {
    @Test("Depolarizing channel satisfies Kraus completeness")
    func depolarizingCompleteness() {
        let channel = DepolarizingChannel(errorProbability: 0.1)
        var sum: [[Complex<Double>]] = [[.zero, .zero], [.zero, .zero]]
        for k in channel.krausOperators {
            let kDagger = MatrixUtilities.hermitianConjugate(k)
            let product = MatrixUtilities.matrixMultiply(kDagger, k)
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    sum[i][j] = sum[i][j] + product[i][j]
                }
            }
        }
        let tolerance = 1e-10
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                let expected: Complex<Double> = (i == j) ? .one : .zero
                let diff = sum[i][j] - expected
                #expect(diff.magnitudeSquared <= tolerance * tolerance, "Σᵢ Kᵢ†Kᵢ[\(i)][\(j)] should equal identity for depolarizing channel")
            }
        }
    }

    @Test("Amplitude damping channel satisfies Kraus completeness")
    func amplitudeDampingCompleteness() {
        let channel = AmplitudeDampingChannel(gamma: 0.2)
        var sum: [[Complex<Double>]] = [[.zero, .zero], [.zero, .zero]]
        for k in channel.krausOperators {
            let kDagger = MatrixUtilities.hermitianConjugate(k)
            let product = MatrixUtilities.matrixMultiply(kDagger, k)
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    sum[i][j] = sum[i][j] + product[i][j]
                }
            }
        }
        let tolerance = 1e-10
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                let expected: Complex<Double> = (i == j) ? .one : .zero
                let diff = sum[i][j] - expected
                #expect(diff.magnitudeSquared <= tolerance * tolerance, "Σᵢ Kᵢ†Kᵢ[\(i)][\(j)] should equal identity for amplitude damping")
            }
        }
    }

    @Test("Phase damping channel satisfies Kraus completeness")
    func phaseDampingCompleteness() {
        let channel = PhaseDampingChannel(gamma: 0.15)
        var sum: [[Complex<Double>]] = [[.zero, .zero], [.zero, .zero]]
        for k in channel.krausOperators {
            let kDagger = MatrixUtilities.hermitianConjugate(k)
            let product = MatrixUtilities.matrixMultiply(kDagger, k)
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    sum[i][j] = sum[i][j] + product[i][j]
                }
            }
        }
        let tolerance = 1e-10
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                let expected: Complex<Double> = (i == j) ? .one : .zero
                let diff = sum[i][j] - expected
                #expect(diff.magnitudeSquared <= tolerance * tolerance, "Σᵢ Kᵢ†Kᵢ[\(i)][\(j)] should equal identity for phase damping")
            }
        }
    }

    @Test("Bit flip channel satisfies Kraus completeness")
    func bitFlipCompleteness() {
        let channel = BitFlipChannel(errorProbability: 0.05)
        var sum: [[Complex<Double>]] = [[.zero, .zero], [.zero, .zero]]
        for k in channel.krausOperators {
            let kDagger = MatrixUtilities.hermitianConjugate(k)
            let product = MatrixUtilities.matrixMultiply(kDagger, k)
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    sum[i][j] = sum[i][j] + product[i][j]
                }
            }
        }
        let tolerance = 1e-10
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                let expected: Complex<Double> = (i == j) ? .one : .zero
                let diff = sum[i][j] - expected
                #expect(diff.magnitudeSquared <= tolerance * tolerance, "Σᵢ Kᵢ†Kᵢ[\(i)][\(j)] should equal identity for bit flip")
            }
        }
    }

    @Test("Phase flip channel satisfies Kraus completeness")
    func phaseFlipCompleteness() {
        let channel = PhaseFlipChannel(errorProbability: 0.05)
        var sum: [[Complex<Double>]] = [[.zero, .zero], [.zero, .zero]]
        for k in channel.krausOperators {
            let kDagger = MatrixUtilities.hermitianConjugate(k)
            let product = MatrixUtilities.matrixMultiply(kDagger, k)
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    sum[i][j] = sum[i][j] + product[i][j]
                }
            }
        }
        let tolerance = 1e-10
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                let expected: Complex<Double> = (i == j) ? .one : .zero
                let diff = sum[i][j] - expected
                #expect(diff.magnitudeSquared <= tolerance * tolerance, "Σᵢ Kᵢ†Kᵢ[\(i)][\(j)] should equal identity for phase flip")
            }
        }
    }

    @Test("Bit-phase flip channel satisfies Kraus completeness")
    func bitPhaseFlipCompleteness() {
        let channel = BitPhaseFlipChannel(errorProbability: 0.05)
        var sum: [[Complex<Double>]] = [[.zero, .zero], [.zero, .zero]]
        for k in channel.krausOperators {
            let kDagger = MatrixUtilities.hermitianConjugate(k)
            let product = MatrixUtilities.matrixMultiply(kDagger, k)
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    sum[i][j] = sum[i][j] + product[i][j]
                }
            }
        }
        let tolerance = 1e-10
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                let expected: Complex<Double> = (i == j) ? .one : .zero
                let diff = sum[i][j] - expected
                #expect(diff.magnitudeSquared <= tolerance * tolerance, "Σᵢ Kᵢ†Kᵢ[\(i)][\(j)] should equal identity for bit-phase flip")
            }
        }
    }

    @Test("Generalized amplitude damping satisfies Kraus completeness")
    func generalizedAmplitudeDampingCompleteness() {
        let channel = GeneralizedAmplitudeDampingChannel(gamma: 0.1, thermalPopulation: 0.2)
        var sum: [[Complex<Double>]] = [[.zero, .zero], [.zero, .zero]]
        for k in channel.krausOperators {
            let kDagger = MatrixUtilities.hermitianConjugate(k)
            let product = MatrixUtilities.matrixMultiply(kDagger, k)
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    sum[i][j] = sum[i][j] + product[i][j]
                }
            }
        }
        let tolerance = 1e-10
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                let expected: Complex<Double> = (i == j) ? .one : .zero
                let diff = sum[i][j] - expected
                #expect(diff.magnitudeSquared <= tolerance * tolerance, "Σᵢ Kᵢ†Kᵢ[\(i)][\(j)] should equal identity for generalized amplitude damping")
            }
        }
    }

    @Test("Two-qubit depolarizing satisfies Kraus completeness")
    func twoQubitDepolarizingCompleteness() {
        let channel = TwoQubitDepolarizingChannel(errorProbability: 0.05)
        var sum = [[Complex<Double>]](repeating: [Complex<Double>](repeating: .zero, count: 4), count: 4)
        for k in channel.krausOperators {
            let kDagger = MatrixUtilities.hermitianConjugate(k)
            let product = MatrixUtilities.matrixMultiply(kDagger, k)
            for i in 0 ..< 4 {
                for j in 0 ..< 4 {
                    sum[i][j] = sum[i][j] + product[i][j]
                }
            }
        }
        let tolerance = 1e-10
        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                let expected: Complex<Double> = (i == j) ? .one : .zero
                let diff = sum[i][j] - expected
                #expect(diff.magnitudeSquared <= tolerance * tolerance, "Σᵢ Kᵢ†Kᵢ[\(i)][\(j)] should equal identity for two-qubit depolarizing")
            }
        }
    }

    @Test("Kraus completeness holds for edge error probabilities")
    func edgeCaseCompleteness() {
        let zeroError = DepolarizingChannel(errorProbability: 0.0)
        let maxError = DepolarizingChannel(errorProbability: 0.75)
        let tolerance = 1e-10

        var sumZero: [[Complex<Double>]] = [[.zero, .zero], [.zero, .zero]]
        for k in zeroError.krausOperators {
            let kDagger = MatrixUtilities.hermitianConjugate(k)
            let product = MatrixUtilities.matrixMultiply(kDagger, k)
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    sumZero[i][j] = sumZero[i][j] + product[i][j]
                }
            }
        }
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                let expected: Complex<Double> = (i == j) ? .one : .zero
                let diff = sumZero[i][j] - expected
                #expect(diff.magnitudeSquared <= tolerance * tolerance, "Zero error Σᵢ Kᵢ†Kᵢ[\(i)][\(j)] should equal identity")
            }
        }

        var sumMax: [[Complex<Double>]] = [[.zero, .zero], [.zero, .zero]]
        for k in maxError.krausOperators {
            let kDagger = MatrixUtilities.hermitianConjugate(k)
            let product = MatrixUtilities.matrixMultiply(kDagger, k)
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    sumMax[i][j] = sumMax[i][j] + product[i][j]
                }
            }
        }
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                let expected: Complex<Double> = (i == j) ? .one : .zero
                let diff = sumMax[i][j] - expected
                #expect(diff.magnitudeSquared <= tolerance * tolerance, "Maximum error Σᵢ Kᵢ†Kᵢ[\(i)][\(j)] should equal identity")
            }
        }
    }
}

/// Test suite for depolarizing noise channel.
/// Validates ρ -> (1-p)ρ + (p/3)(XρX + YρY + ZρZ) behavior,
/// trace preservation, purity reduction, and maximally mixed limits.
@Suite("Depolarizing Channel")
struct DepolarizingChannelTests {
    @Test("Zero error probability preserves state exactly")
    func zeroErrorPreservesState() {
        let channel = DepolarizingChannel(errorProbability: 0.0)
        let dm = DensityMatrix(qubits: 1)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.purity() - 1.0) < 1e-10, "Zero error should preserve purity")
        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "State should be unchanged")
    }

    @Test("Maximum error (p=3/4) produces maximally mixed state")
    func maxErrorProducesMaximallyMixed() {
        let channel = DepolarizingChannel(errorProbability: 0.75)
        let dm = DensityMatrix(qubits: 1)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.purity() - 0.5) < 1e-10, "p=3/4 should give maximally mixed (purity=0.5)")
        #expect(abs(result.probability(of: 0) - 0.5) < 1e-10, "P(0) should be 0.5")
        #expect(abs(result.probability(of: 1) - 0.5) < 1e-10, "P(1) should be 0.5")
    }

    @Test("Depolarizing channel preserves trace")
    func preservesTrace() {
        let channel = DepolarizingChannel(errorProbability: 0.1)
        let invSqrt2 = 1.0 / sqrt(2.0)
        let plus = QuantumState(qubits: 1, amplitudes: [Complex(invSqrt2, 0), Complex(invSqrt2, 0)])
        let dm = DensityMatrix(pureState: plus)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace should be preserved")
    }

    @Test("Depolarizing reduces purity")
    func reducesPurity() {
        let channel = DepolarizingChannel(errorProbability: 0.1)
        let dm = DensityMatrix(qubits: 1)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(result.purity() < dm.purity(), "Depolarizing should reduce purity")
    }

    @Test("Depolarizing on specific qubit in multi-qubit system")
    func multiQubitTargeting() {
        let channel = DepolarizingChannel(errorProbability: 0.5)
        let dm = DensityMatrix(qubits: 2)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace preserved in 2-qubit system")

        let reduced = result.partialTrace(over: [1])
        #expect(reduced.purity() < 1.0, "Targeted qubit should be depolarized")
    }

    @Test("Depolarizing preserves maximally mixed state")
    func preservesMaximallyMixed() {
        let channel = DepolarizingChannel(errorProbability: 0.3)
        let dm = DensityMatrix.maximallyMixed(qubits: 1)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.purity() - 0.5) < 1e-10, "Maximally mixed should stay maximally mixed")
    }
}

/// Test suite for amplitude damping (T₁ decay).
/// Models |1⟩ -> |0⟩ relaxation with asymmetric decay where
/// ground state is unaffected and excited state decays to ground.
@Suite("Amplitude Damping Channel")
struct AmplitudeDampingChannelTests {
    @Test("Zero damping preserves state")
    func zeroDampingPreservesState() {
        let channel = AmplitudeDampingChannel(gamma: 0.0)
        let dm = DensityMatrix.basis(qubits: 1, state: 1)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.probability(of: 1) - 1.0) < 1e-10, "Zero damping should preserve |1⟩")
    }

    @Test("Complete damping (γ=1) decays |1⟩ to |0⟩")
    func completeDampingDecays() {
        let channel = AmplitudeDampingChannel(gamma: 1.0)
        let dm = DensityMatrix.basis(qubits: 1, state: 1)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "Complete damping should give |0⟩")
        #expect(abs(result.probability(of: 1)) < 1e-10, "|1⟩ probability should be zero")
    }

    @Test("Ground state is unaffected by amplitude damping")
    func groundStateUnaffected() {
        let channel = AmplitudeDampingChannel(gamma: 0.5)
        let dm = DensityMatrix(qubits: 1)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "|0⟩ should be unaffected")
        #expect(result.isPure(), "|0⟩ should remain pure")
    }

    @Test("Amplitude damping preserves trace")
    func preservesTrace() {
        let channel = AmplitudeDampingChannel(gamma: 0.3)
        let invSqrt2 = 1.0 / sqrt(2.0)
        let plus = QuantumState(qubits: 1, amplitudes: [Complex(invSqrt2, 0), Complex(invSqrt2, 0)])
        let dm = DensityMatrix(pureState: plus)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace should be preserved")
    }

    @Test("Amplitude damping reduces coherences")
    func reducesCoherences() {
        let channel = AmplitudeDampingChannel(gamma: 0.5)
        let invSqrt2 = 1.0 / sqrt(2.0)
        let plus = QuantumState(qubits: 1, amplitudes: [Complex(invSqrt2, 0), Complex(invSqrt2, 0)])
        let dm = DensityMatrix(pureState: plus)
        let result = channel.apply(to: dm, qubit: 0)

        let originalCoherence = dm.element(row: 0, col: 1).magnitude
        let resultCoherence = result.element(row: 0, col: 1).magnitude

        #expect(resultCoherence < originalCoherence, "Coherences should decay")
    }

    @Test("Partial damping gives intermediate populations")
    func partialDampingIntermediate() {
        let gamma = 0.4
        let channel = AmplitudeDampingChannel(gamma: gamma)
        let dm = DensityMatrix.basis(qubits: 1, state: 1)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.probability(of: 0) - gamma) < 1e-10, "P(0) should be γ")
        #expect(abs(result.probability(of: 1) - (1 - gamma)) < 1e-10, "P(1) should be 1-γ")
    }
}

/// Test suite for phase damping (T₂ dephasing).
/// Models loss of coherence without energy loss where diagonal
/// populations are preserved but off-diagonal coherences decay.
@Suite("Phase Damping Channel")
struct PhaseDampingChannelTests {
    @Test("Zero dephasing preserves state")
    func zeroDampingPreservesState() {
        let channel = PhaseDampingChannel(gamma: 0.0)
        let invSqrt2 = 1.0 / sqrt(2.0)
        let plus = QuantumState(qubits: 1, amplitudes: [Complex(invSqrt2, 0), Complex(invSqrt2, 0)])
        let dm = DensityMatrix(pureState: plus)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.purity() - 1.0) < 1e-10, "Zero dephasing should preserve purity")
    }

    @Test("Complete dephasing removes coherences")
    func completeDampingRemovesCoherences() {
        let channel = PhaseDampingChannel(gamma: 1.0)
        let invSqrt2 = 1.0 / sqrt(2.0)
        let plus = QuantumState(qubits: 1, amplitudes: [Complex(invSqrt2, 0), Complex(invSqrt2, 0)])
        let dm = DensityMatrix(pureState: plus)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(result.element(row: 0, col: 1).magnitude < 1e-10, "Off-diagonal should be zero")
        #expect(result.element(row: 1, col: 0).magnitude < 1e-10, "Off-diagonal should be zero")
    }

    @Test("Phase damping preserves populations")
    func preservesPopulations() {
        let channel = PhaseDampingChannel(gamma: 0.5)
        let invSqrt2 = 1.0 / sqrt(2.0)
        let plus = QuantumState(qubits: 1, amplitudes: [Complex(invSqrt2, 0), Complex(invSqrt2, 0)])
        let dm = DensityMatrix(pureState: plus)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.probability(of: 0) - 0.5) < 1e-10, "P(0) should be unchanged")
        #expect(abs(result.probability(of: 1) - 0.5) < 1e-10, "P(1) should be unchanged")
    }

    @Test("Computational basis states unaffected")
    func basisStatesUnaffected() {
        let channel = PhaseDampingChannel(gamma: 0.9)

        let dm0 = DensityMatrix(qubits: 1)
        let result0 = channel.apply(to: dm0, qubit: 0)
        #expect(result0.isPure(), "|0⟩ should remain pure")

        let dm1 = DensityMatrix.basis(qubits: 1, state: 1)
        let result1 = channel.apply(to: dm1, qubit: 0)
        #expect(result1.isPure(), "|1⟩ should remain pure")
    }

    @Test("Phase damping preserves trace")
    func preservesTrace() {
        let channel = PhaseDampingChannel(gamma: 0.4)
        let invSqrt2 = 1.0 / sqrt(2.0)
        let state = QuantumState(qubits: 1, amplitudes: [Complex(invSqrt2, 0), Complex(0, invSqrt2)])
        let dm = DensityMatrix(pureState: state)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace should be preserved")
    }
}

/// Test suite for bit flip and phase flip channels.
/// Validates Pauli X and Z error models with probabilistic
/// application and their combined bit-phase flip (Y error).
@Suite("Bit Flip and Phase Flip Channels")
struct BitPhaseFlipChannelTests {
    @Test("Bit flip with p=0 preserves state")
    func bitFlipZeroPreserves() {
        let channel = BitFlipChannel(errorProbability: 0.0)
        let dm = DensityMatrix(qubits: 1)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "State should be unchanged")
    }

    @Test("Bit flip with p=1 flips state")
    func bitFlipOneFlips() {
        let channel = BitFlipChannel(errorProbability: 1.0)
        let dm = DensityMatrix(qubits: 1)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.probability(of: 1) - 1.0) < 1e-10, "|0⟩ should flip to |1⟩")
    }

    @Test("Phase flip with p=0 preserves state")
    func phaseFlipZeroPreserves() {
        let channel = PhaseFlipChannel(errorProbability: 0.0)
        let invSqrt2 = 1.0 / sqrt(2.0)
        let plus = QuantumState(qubits: 1, amplitudes: [Complex(invSqrt2, 0), Complex(invSqrt2, 0)])
        let dm = DensityMatrix(pureState: plus)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.purity() - 1.0) < 1e-10, "State should be unchanged")
    }

    @Test("Phase flip preserves computational basis")
    func phaseFlipPreservesBasis() {
        let channel = PhaseFlipChannel(errorProbability: 0.5)
        let dm = DensityMatrix(qubits: 1)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "|0⟩ population unchanged")
    }

    @Test("Bit flip p=0.5 gives mixed state")
    func bitFlipHalfMixed() {
        let channel = BitFlipChannel(errorProbability: 0.5)
        let dm = DensityMatrix(qubits: 1)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.probability(of: 0) - 0.5) < 1e-10, "P(0) should be 0.5")
        #expect(abs(result.probability(of: 1) - 0.5) < 1e-10, "P(1) should be 0.5")
    }

    @Test("Bit-phase flip (Y error) combines effects")
    func bitPhaseFlipCombines() {
        let channel = BitPhaseFlipChannel(errorProbability: 1.0)
        let dm = DensityMatrix(qubits: 1)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.probability(of: 1) - 1.0) < 1e-10, "Y on |0⟩ gives |1⟩ (up to phase)")
    }
}

/// Test suite for generalized amplitude damping (finite temperature).
/// Models thermal equilibrium with both excitation and relaxation,
/// reducing to standard amplitude damping at zero temperature.
@Suite("Generalized Amplitude Damping Channel")
struct GeneralizedAmplitudeDampingTests {
    @Test("Zero temperature limit equals standard amplitude damping")
    func zeroTemperatureLimit() {
        let gamma = 0.3
        let gadChannel = GeneralizedAmplitudeDampingChannel(gamma: gamma, thermalPopulation: 0.0)
        let adChannel = AmplitudeDampingChannel(gamma: gamma)

        let dm = DensityMatrix.basis(qubits: 1, state: 1)
        let gadResult = gadChannel.apply(to: dm, qubit: 0)
        let adResult = adChannel.apply(to: dm, qubit: 0)

        #expect(abs(gadResult.probability(of: 0) - adResult.probability(of: 0)) < 1e-10,
                "Zero temperature GAD should match AD")
    }

    @Test("High temperature leads to maximally mixed")
    func highTemperatureLimit() {
        let channel = GeneralizedAmplitudeDampingChannel(gamma: 1.0, thermalPopulation: 0.5)
        let dm = DensityMatrix(qubits: 1)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.probability(of: 0) - 0.5) < 1e-10, "Infinite temperature gives 50/50")
        #expect(abs(result.probability(of: 1) - 0.5) < 1e-10, "Infinite temperature gives 50/50")
    }

    @Test("Preserves trace for all parameters")
    func preservesTrace() {
        let channel = GeneralizedAmplitudeDampingChannel(gamma: 0.4, thermalPopulation: 0.2)
        let invSqrt2 = 1.0 / sqrt(2.0)
        let state = QuantumState(qubits: 1, amplitudes: [Complex(invSqrt2, 0), Complex(invSqrt2, 0)])
        let dm = DensityMatrix(pureState: state)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace should be preserved")
    }
}

/// Test suite for two-qubit depolarizing noise.
/// Models correlated errors on two-qubit gates with uniform
/// distribution over all 16 two-qubit Pauli operators.
@Suite("Two-Qubit Depolarizing Channel")
struct TwoQubitDepolarizingTests {
    @Test("Zero error preserves state")
    func zeroErrorPreserves() {
        let channel = TwoQubitDepolarizingChannel(errorProbability: 0.0)
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])
        let dm = DensityMatrix(pureState: bell)
        let result = channel.apply(to: dm, qubits: [0, 1])

        #expect(abs(result.purity() - 1.0) < 1e-10, "Zero error should preserve purity")
    }

    @Test("Reduces purity of entangled state")
    func reducesPurity() {
        let channel = TwoQubitDepolarizingChannel(errorProbability: 0.1)
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])
        let dm = DensityMatrix(pureState: bell)
        let result = channel.apply(to: dm, qubits: [0, 1])

        #expect(result.purity() < dm.purity(), "Should reduce purity")
    }

    @Test("Two-qubit depolarizing preserves trace on 2-qubit system")
    func twoQubitDepolarizingPreservesTrace() {
        let channel = TwoQubitDepolarizingChannel(errorProbability: 0.05)
        let dm = DensityMatrix(qubits: 2)
        let result = channel.apply(to: dm, qubits: [0, 1])

        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace should be exactly preserved for 2-qubit system")
    }

    @Test("Two-qubit depolarizing preserves trace on larger system")
    func twoQubitDepolarizingPreservesTraceLargerSystem() {
        let channel = TwoQubitDepolarizingChannel(errorProbability: 0.05)
        let dm = DensityMatrix(qubits: 4)
        let result = channel.apply(to: dm, qubits: [1, 2])

        #expect(abs(result.trace() - 1.0) < 1e-9, "Trace should be preserved for 4-qubit system")
    }

    @Test("Preserves trace on 2-qubit system")
    func preservesTrace2Qubit() {
        let channel = TwoQubitDepolarizingChannel(errorProbability: 0.1)
        let dm = DensityMatrix(qubits: 2)
        let result = channel.apply(to: dm, qubits: [0, 1])

        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace should be exactly preserved")
    }

    @Test("Preserves trace on larger system")
    func preservesTraceLargerSystem() {
        let channel = TwoQubitDepolarizingChannel(errorProbability: 0.1)
        let dm = DensityMatrix(qubits: 4)
        let result = channel.apply(to: dm, qubits: [1, 2])

        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace should be preserved on 4-qubit system")
    }

    @Test("Preserves trace on 5-qubit system with non-adjacent qubits")
    func preservesTraceNonAdjacent() {
        let channel = TwoQubitDepolarizingChannel(errorProbability: 0.05)
        let dm = DensityMatrix(qubits: 5)
        let result = channel.apply(to: dm, qubits: [0, 3])

        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace should be preserved for non-adjacent qubits")
    }
}

/// Test suite for user-defined Kraus operators.
/// Validates custom noise channels with arbitrary Kraus operators
/// and equivalence to built-in channels when operators match.
@Suite("Custom Kraus Channel")
struct CustomKrausChannelTests {
    @Test("Identity Kraus operators preserve state")
    func identityKrausPreserves() {
        let identity: [[[Complex<Double>]]] = [
            [[.one, .zero], [.zero, .one]],
        ]
        let channel = CustomKrausChannel(krausOperators: identity)
        let dm = DensityMatrix(qubits: 1)
        let result = channel.apply(to: dm, qubit: 0)

        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "Identity should preserve state")
    }

    @Test("Equivalent to built-in depolarizing")
    func equivalentToDepolarizing() {
        let p = 0.2
        let sqrtOneMinusP = sqrt(1.0 - p)
        let sqrtPOver3 = sqrt(p / 3.0)

        let k0: [[Complex<Double>]] = [
            [Complex(sqrtOneMinusP, 0), .zero],
            [.zero, Complex(sqrtOneMinusP, 0)],
        ]
        let k1: [[Complex<Double>]] = [
            [.zero, Complex(sqrtPOver3, 0)],
            [Complex(sqrtPOver3, 0), .zero],
        ]
        let k2: [[Complex<Double>]] = [
            [.zero, Complex(0, -sqrtPOver3)],
            [Complex(0, sqrtPOver3), .zero],
        ]
        let k3: [[Complex<Double>]] = [
            [Complex(sqrtPOver3, 0), .zero],
            [.zero, Complex(-sqrtPOver3, 0)],
        ]

        let customChannel = CustomKrausChannel(krausOperators: [k0, k1, k2, k3])
        let builtinChannel = DepolarizingChannel(errorProbability: p)

        let dm = DensityMatrix(qubits: 1)
        let customResult = customChannel.apply(to: dm, qubit: 0)
        let builtinResult = builtinChannel.apply(to: dm, qubit: 0)

        #expect(abs(customResult.purity() - builtinResult.purity()) < 1e-10,
                "Custom should match built-in depolarizing")
    }
}

/// Test suite for readout confusion matrix.
/// Validates measurement error application, inverse matrix computation,
/// and round-trip mitigation through confusion matrix inversion.
@Suite("Measurement Error Model")
struct MeasurementErrorModelTests {
    @Test("Zero error gives identity confusion matrix")
    func zeroErrorIdentity() {
        let model = MeasurementErrorModel(p0Given1: 0.0, p1Given0: 0.0)

        #expect(model.confusionMatrix[0][0] == 1.0, "P(0|0) should be 1")
        #expect(model.confusionMatrix[1][1] == 1.0, "P(1|1) should be 1")
        #expect(model.confusionMatrix[0][1] == 0.0, "P(1|0) should be 0")
        #expect(model.confusionMatrix[1][0] == 0.0, "P(0|1) should be 0")
    }

    @Test("Apply error transforms probabilities")
    func applyErrorTransforms() {
        let model = MeasurementErrorModel(p0Given1: 0.1, p1Given0: 0.05)
        let (noisyP0, noisyP1) = model.applyError(to: (1.0, 0.0))

        #expect(abs(noisyP0 - 0.95) < 1e-10, "P(0) should be 1 - p1Given0 = 0.95")
        #expect(abs(noisyP1 - 0.05) < 1e-10, "P(1) should be p1Given0 = 0.05")
    }

    @Test("Inverse matrix satisfies M⁻¹M = I")
    func inverseMatrixProperty() {
        let model = MeasurementErrorModel(p0Given1: 0.15, p1Given0: 0.08)

        let a = model.confusionMatrix[0][0] * model.inverseMatrix[0][0] +
            model.confusionMatrix[0][1] * model.inverseMatrix[1][0]
        let d = model.confusionMatrix[1][0] * model.inverseMatrix[0][1] +
            model.confusionMatrix[1][1] * model.inverseMatrix[1][1]

        #expect(abs(a - 1.0) < 1e-10, "M⁻¹M should give identity diagonal")
        #expect(abs(d - 1.0) < 1e-10, "M⁻¹M should give identity diagonal")
    }

    @Test("Initialize from explicit confusion matrix")
    func initFromConfusionMatrix() {
        let matrix = [[0.98, 0.02], [0.03, 0.97]]
        let model = MeasurementErrorModel(confusionMatrix: matrix)

        #expect(model.confusionMatrix[0][0] == 0.98, "Should preserve matrix values")
        #expect(model.confusionMatrix[1][0] == 0.03, "Should preserve matrix values")
    }

    @Test("Mitigate histogram with sparse entries uses zero for missing partner")
    func mitigateHistogramSparseEntries() {
        let model = MeasurementErrorModel(p0Given1: 0.1, p1Given0: 0.05)

        let sparseHistogram = [0: 100]

        let corrected = model.mitigateHistogram(sparseHistogram, qubit: 0, totalQubits: 2)

        #expect(corrected[0] != nil, "State 0 should have corrected count")
        #expect(corrected[0]! > 0, "Corrected count should be positive")
    }

    @Test("Mitigate histogram handles asymmetric sparse data")
    func mitigateHistogramAsymmetricSparse() {
        let model = MeasurementErrorModel(p0Given1: 0.02, p1Given0: 0.01)

        let sparseHistogram: [Int: Int] = [0b00: 500, 0b11: 500]

        let corrected = model.mitigateHistogram(sparseHistogram, qubit: 0, totalQubits: 2)

        #expect(corrected.count >= 2, "Should have at least 2 corrected entries")
    }
}

/// Test suite for NoiseModel composition and factory methods.
/// Validates ideal, depolarizing, and NISQ presets with proper
/// channel selection based on gate arity and hardware profiles.
@Suite("Noise Model")
struct NoiseModelTests {
    @Test("Ideal noise model has no noise")
    func idealHasNoNoise() {
        let model = NoiseModel.ideal

        #expect(model.singleQubitNoise == nil, "Ideal should have no single-qubit noise")
        #expect(model.twoQubitNoise == nil, "Ideal should have no two-qubit noise")
        #expect(model.measurementError == nil, "Ideal should have no measurement error")
        #expect(!model.hasNoise, "hasNoise should be false")
    }

    @Test("Depolarizing factory creates correct channels")
    func depolarizingFactory() {
        let model = NoiseModel.depolarizing(singleQubitError: 0.001, twoQubitError: 0.01)

        #expect(model.singleQubitNoise != nil, "Should have single-qubit noise")
        #expect(model.twoQubitNoise != nil, "Should have two-qubit noise")
        #expect(model.hasNoise, "hasNoise should be true")
    }

    @Test("Typical NISQ preset has all noise components")
    func typicalNISQPreset() {
        let model = NoiseModel.typicalNISQ

        #expect(model.singleQubitNoise != nil, "Should have single-qubit noise")
        #expect(model.twoQubitNoise != nil, "Should have two-qubit noise")
        #expect(model.measurementError != nil, "Should have measurement error")
    }

    @Test("Amplitude damping factory uses T₁")
    func amplitudeDampingFactory() {
        let model = NoiseModel.amplitudeDamping(t1: 100_000)

        #expect(model.singleQubitNoise != nil, "Should have single-qubit noise")
    }

    @Test("Apply noise selects correct channel by gate arity")
    func applyNoiseSelectsChannel() {
        let model = NoiseModel.depolarizing(singleQubitError: 0.1, twoQubitError: 0.2)
        let dm = DensityMatrix(qubits: 2)

        let afterSingle = model.applyNoise(after: .hadamard, targetQubits: [0], to: dm)
        #expect(afterSingle.purity() < 1.0, "Single-qubit noise should reduce purity")

        let afterTwo = model.applyNoise(after: .cnot, targetQubits: [0, 1], to: dm)
        #expect(afterTwo.purity() < 1.0, "Two-qubit noise should reduce purity")
    }

    @Test("Apply noise returns unchanged matrix when no single-qubit noise configured")
    func applyNoiseNoSingleQubitNoise() {
        let model = NoiseModel(twoQubitNoise: TwoQubitDepolarizingChannel(errorProbability: 0.1))
        let dm = DensityMatrix(qubits: 2)

        let result = model.applyNoise(after: .hadamard, targetQubits: [0], to: dm)

        #expect(result == dm, "Matrix should be unchanged when no single-qubit noise")
        #expect(result.isPure(), "Purity should be preserved")
    }

    @Test("Apply noise returns unchanged matrix when no two-qubit noise configured")
    func applyNoiseNoTwoQubitNoise() {
        let model = NoiseModel(singleQubitNoise: DepolarizingChannel(errorProbability: 0.1))
        let dm = DensityMatrix(qubits: 2)

        let result = model.applyNoise(after: .cnot, targetQubits: [0, 1], to: dm)

        #expect(result == dm, "Matrix should be unchanged when no two-qubit noise")
        #expect(result.isPure(), "Purity should be preserved")
    }

    @Test("Apply noise returns unchanged matrix for three-qubit gate")
    func applyNoiseThreeQubitGate() {
        let model = NoiseModel.depolarizing(singleQubitError: 0.1, twoQubitError: 0.2)
        let dm = DensityMatrix(qubits: 3)

        let result = model.applyNoise(after: .toffoli, targetQubits: [0, 1, 2], to: dm)

        #expect(result == dm, "Matrix should be unchanged for 3-qubit gate (no 3-qubit noise)")
        #expect(result.isPure(), "Purity should be preserved")
    }

    @Test("Apply noise with idle returns result without idle processing when no idle config")
    func applyNoiseWithIdleNoConfig() {
        let model = NoiseModel.depolarizing(singleQubitError: 0.1, twoQubitError: 0.2)
        let dm = DensityMatrix(qubits: 3)

        let withIdle = model.applyNoiseWithIdle(after: .hadamard, targetQubits: [0], to: dm, totalQubits: 3)
        let withoutIdle = model.applyNoise(after: .hadamard, targetQubits: [0], to: dm)

        #expect(abs(withIdle.purity() - withoutIdle.purity()) < 1e-10,
                "Should behave same as applyNoise when no idle config")
    }

    @Test("From hardware profile creates noise model")
    func fromHardwareProfile() {
        let profile = HardwareNoiseProfile.ibmManila
        let model = NoiseModel.from(profile: profile)

        #expect(model.hasNoise, "Profile-derived model should have noise")
        #expect(model.singleQubitNoise != nil, "Should have single-qubit noise")
        #expect(model.twoQubitNoise != nil, "Should have two-qubit noise")
        #expect(model.measurementError != nil, "Should have measurement error")
    }
}

/// Test suite for idle qubit T₁/T₂ decoherence configuration.
/// Validates amplitude and phase damping gamma calculations from
/// relaxation times and per-qubit T₁/T₂ parameter handling.
@Suite("Idle Noise Configuration")
struct IdleNoiseConfigTests {
    @Test("Compute amplitude damping gamma from T₁")
    func amplitudeDampingGamma() {
        let config = IdleNoiseConfig(t1: 100_000, t2: 80000)
        let gateTime = 300.0

        let gamma = config.amplitudeDampingGamma(idleTime: gateTime)
        let expectedGamma = 1.0 - exp(-gateTime / 100_000)

        #expect(abs(gamma - expectedGamma) < 1e-10, "γ should be 1 - exp(-t/T₁)")
    }

    @Test("Compute phase damping gamma from T₂")
    func phaseDampingGamma() {
        let t1 = 100_000.0
        let t2 = 80000.0
        let config = IdleNoiseConfig(t1: t1, t2: t2)
        let gateTime = 300.0

        let gamma = config.phaseDampingGamma(idleTime: gateTime)
        let tPhi = 1.0 / (1.0 / t2 - 1.0 / (2.0 * t1))
        let expectedGamma = 1.0 - exp(-gateTime / tPhi)

        #expect(abs(gamma - expectedGamma) < 1e-10, "γ should use T_φ formula")
    }

    @Test("Per-qubit T₁/T₂ values")
    func perQubitValues() {
        let config = IdleNoiseConfig(
            perQubitT1: [100_000, 80000, 120_000],
            perQubitT2: [80000, 60000, 90000],
        )

        #expect(config.t1ForQubit(0) == 100_000, "Should return per-qubit T₁")
        #expect(config.t1ForQubit(1) == 80000, "Should return per-qubit T₁")
        #expect(config.t2ForQubit(2) == 90000, "Should return per-qubit T₂")
    }

    @Test("Creates valid damping channels")
    func createsDampingChannels() {
        let config = IdleNoiseConfig(t1: 100_000, t2: 80000)

        let adChannel = config.amplitudeDampingChannel(idleTime: 300)
        let pdChannel = config.phaseDampingChannel(idleTime: 300)

        #expect(adChannel.gamma >= 0 && adChannel.gamma <= 1, "AD gamma should be in [0,1]")
        #expect(pdChannel.gamma >= 0 && pdChannel.gamma <= 1, "PD gamma should be in [0,1]")
    }

    @Test("Phase damping gamma returns 0 when T₂ = 2T₁ (T₁-limited regime)")
    func phaseDampingGammaT1Limited() {
        let config = IdleNoiseConfig(t1: 100_000, t2: 200_000)
        let gamma = config.phaseDampingGamma(idleTime: 300)

        #expect(gamma == 0, "Phase damping gamma should be 0 when T₂ = 2T₁ (no pure dephasing)")
    }

    @Test("Phase damping gamma returns 0 for per-qubit T₁-limited regime")
    func phaseDampingGammaPerQubitT1Limited() {
        let config = IdleNoiseConfig(
            perQubitT1: [100_000, 50000],
            perQubitT2: [200_000, 100_000],
        )

        let gamma0 = config.phaseDampingGamma(idleTime: 300, qubit: 0)
        let gamma1 = config.phaseDampingGamma(idleTime: 300, qubit: 1)

        #expect(gamma0 == 0, "Qubit 0 should have gamma=0 when T₂ = 2T₁")
        #expect(gamma1 == 0, "Qubit 1 should have gamma=0 when T₂ = 2T₁")
    }
}

/// Test suite for gate duration parameters.
/// Validates IBM, Google Sycamore, and IonQ timing presets
/// with correct single-qubit, two-qubit, and multi-qubit times.
@Suite("Gate Timing Model")
struct GateTimingModelTests {
    @Test("IBM default timings")
    func ibmDefaultTimings() {
        let timing = GateTimingModel.ibmDefault

        #expect(timing.singleQubitGateTime == 35, "Single-qubit should be 35ns")
        #expect(timing.twoQubitGateTime == 300, "Two-qubit should be 300ns")
    }

    @Test("Google Sycamore timings")
    func googleSycamoreTimings() {
        let timing = GateTimingModel.googleSycamore

        #expect(timing.singleQubitGateTime == 25, "Single-qubit should be 25ns")
        #expect(timing.twoQubitGateTime == 32, "Two-qubit should be 32ns")
    }

    @Test("IonQ timings are slower")
    func ionQTimings() {
        let timing = GateTimingModel.ionQ

        #expect(timing.singleQubitGateTime == 10000, "Single-qubit should be 10μs")
        #expect(timing.twoQubitGateTime == 200_000, "Two-qubit should be 200μs")
    }

    @Test("Gate time lookup by arity")
    func gateTimeLookup() {
        let timing = GateTimingModel.ibmDefault

        #expect(timing.gateTime(for: 1) == 35, "1-qubit gate time")
        #expect(timing.gateTime(for: 2) == 300, "2-qubit gate time")
        #expect(timing.gateTime(for: 3) == 600, "3-qubit gate time")
    }

    @Test("Gate time default case for high arity")
    func gateTimeDefaultCase() {
        let timing = GateTimingModel.ibmDefault

        #expect(timing.gateTime(for: 4) == timing.threeQubitGateTime,
                "4-qubit should fall back to 3-qubit time")
        #expect(timing.gateTime(for: 5) == timing.threeQubitGateTime,
                "5-qubit should fall back to 3-qubit time")
        #expect(timing.gateTime(for: 10) == timing.threeQubitGateTime,
                "10-qubit should fall back to 3-qubit time")
    }

    @Test("Rigetti timing preset")
    func rigettiTimings() {
        let timing = GateTimingModel.rigetti

        #expect(timing.singleQubitGateTime == 40, "Single-qubit should be 40ns")
        #expect(timing.twoQubitGateTime == 180, "Two-qubit should be 180ns")
        #expect(timing.threeQubitGateTime == 400, "Three-qubit should be 400ns")
        #expect(timing.measurementTime == 800, "Measurement should be 800ns")
    }

    @Test("Custom gate timing model")
    func customGateTimingModel() {
        let timing = GateTimingModel(
            singleQubitGateTime: 50,
            twoQubitGateTime: 200,
            threeQubitGateTime: 500,
            measurementTime: 1500,
        )

        #expect(timing.singleQubitGateTime == 50, "Custom single-qubit time")
        #expect(timing.twoQubitGateTime == 200, "Custom two-qubit time")
        #expect(timing.threeQubitGateTime == 500, "Custom three-qubit time")
        #expect(timing.measurementTime == 1500, "Custom measurement time")
    }
}

/// Test suite for device-specific noise characterization.
/// Validates IBM Manila, Google Sycamore, and IonQ Harmony profiles
/// with qubit counts, connectivity graphs, and error parameters.
@Suite("Hardware Noise Profile")
struct HardwareNoiseProfileTests {
    @Test("IBM Manila profile has 5 qubits")
    func ibmManilaQubitCount() {
        let profile = HardwareNoiseProfile.ibmManila

        #expect(profile.qubitCount == 5, "Manila should have 5 qubits")
        #expect(profile.qubitParameters.count == 5, "Should have 5 qubit parameter sets")
    }

    @Test("Linear topology connectivity")
    func manilaConnectivity() {
        let profile = HardwareNoiseProfile.ibmManila

        #expect(profile.areConnected(0, 1), "0-1 should be connected")
        #expect(profile.areConnected(1, 2), "1-2 should be connected")
        #expect(!profile.areConnected(0, 2), "0-2 should not be connected")
        #expect(!profile.areConnected(0, 4), "0-4 should not be connected")
    }

    @Test("Edge parameters lookup")
    func edgeParametersLookup() {
        let profile = HardwareNoiseProfile.ibmManila

        let edge = profile.edgeParameters(q1: 0, q2: 1)
        #expect(edge != nil, "Should find edge 0-1")
        #expect(edge!.twoQubitErrorRate > 0, "Should have non-zero error rate")
    }

    @Test("Average error calculations")
    func averageErrors() {
        let profile = HardwareNoiseProfile.ibmManila

        #expect(profile.averageSingleQubitError > 0, "Average single-qubit error should be positive")
        #expect(profile.averageTwoQubitError > 0, "Average two-qubit error should be positive")
        #expect(profile.averageT1 > 0, "Average T₁ should be positive")
        #expect(profile.averageT2 > 0, "Average T₂ should be positive")
    }

    @Test("Google Sycamore 12-qubit grid")
    func sycamoreGrid() {
        let profile = HardwareNoiseProfile.googleSycamore12

        #expect(profile.qubitCount == 12, "Should have 12 qubits")
        #expect(profile.areConnected(0, 1), "Horizontal neighbors connected")
        #expect(profile.areConnected(0, 4), "Vertical neighbors connected")
    }

    @Test("IonQ all-to-all connectivity")
    func ionQConnectivity() {
        let profile = HardwareNoiseProfile.ionQHarmony

        #expect(profile.qubitCount == 11, "Should have 11 qubits")
        #expect(profile.areConnected(0, 10), "All pairs should be connected")
        #expect(profile.areConnected(3, 7), "All pairs should be connected")
    }

    @Test("Create single-qubit channel from profile")
    func singleQubitChannelFromProfile() {
        let profile = HardwareNoiseProfile.ibmManila
        let channel = profile.singleQubitChannel(for: 0)

        #expect(channel.errorProbability > 0, "Should have non-zero error probability")
    }

    @Test("Create two-qubit channel from profile")
    func twoQubitChannelFromProfile() {
        let profile = HardwareNoiseProfile.ibmManila
        let channel = profile.twoQubitChannel(for: 0, 1)

        #expect(channel.errorProbability > 0, "Should have non-zero error probability")
    }

    @Test("Linear chain factory")
    func linearChainFactory() {
        let profile = HardwareNoiseProfile.linearChain(qubits: 10)

        #expect(profile.qubitCount == 10, "Should have 10 qubits")
        #expect(profile.areConnected(4, 5), "Adjacent qubits connected")
        #expect(!profile.areConnected(4, 6), "Non-adjacent not connected")
    }

    @Test("Grid factory")
    func gridFactory() {
        let profile = HardwareNoiseProfile.grid(rows: 3, cols: 4)

        #expect(profile.qubitCount == 12, "Should have 3*4=12 qubits")
        #expect(profile.areConnected(0, 1), "Horizontal connected")
        #expect(profile.areConnected(0, 4), "Vertical connected")
    }

    @Test("IBM Quito T-shaped topology")
    func ibmQuitoTopology() {
        let profile = HardwareNoiseProfile.ibmQuito

        #expect(profile.qubitCount == 5, "Quito should have 5 qubits")
        #expect(profile.name.contains("Quito"), "Name should contain Quito")

        #expect(profile.areConnected(0, 2), "Qubit 0 connects to 2 in T-shape")
        #expect(profile.areConnected(1, 2), "Qubit 1 connects to 2")
        #expect(profile.areConnected(2, 3), "Qubit 2 connects to 3")
        #expect(profile.areConnected(3, 4), "Qubit 3 connects to 4")

        #expect(!profile.areConnected(0, 1), "Qubit 0 should not connect to 1")
        #expect(!profile.areConnected(0, 3), "Qubit 0 should not connect to 3")
    }

    @Test("Rigetti Aspen-8 octagonal topology")
    func rigettiAspen8Topology() {
        let profile = HardwareNoiseProfile.rigettiAspen8

        #expect(profile.qubitCount == 8, "Aspen-8 should have 8 qubits")
        #expect(profile.name.contains("Rigetti"), "Name should contain Rigetti")

        #expect(profile.areConnected(0, 1), "Ring connection 0-1")
        #expect(profile.areConnected(7, 0), "Ring wraps around 7-0")
        #expect(profile.areConnected(0, 4), "Cross connection 0-4")
        #expect(profile.areConnected(2, 6), "Cross connection 2-6")

        #expect(!profile.areConnected(1, 5), "No direct 1-5 connection")
    }

    @Test("Uniform parameters initializer")
    func uniformParametersInitializer() {
        let uniformParams = QubitNoiseParameters(
            t1: 100_000,
            t2: 80000,
            singleQubitErrorRate: 0.001,
            readoutError0Given1: 0.02,
            readoutError1Given0: 0.01,
        )
        let edges = [
            EdgeNoiseParameters(qubit1: 0, qubit2: 1, twoQubitErrorRate: 0.01),
            EdgeNoiseParameters(qubit1: 1, qubit2: 2, twoQubitErrorRate: 0.01),
        ]
        let profile = HardwareNoiseProfile(
            name: "Uniform Test",
            qubitCount: 3,
            uniformParameters: uniformParams,
            edges: edges,
        )

        #expect(profile.qubitCount == 3, "Should have 3 qubits")
        #expect(profile.qubitParameters.count == 3, "Should have 3 parameter sets")

        for i in 0 ..< 3 {
            #expect(profile.qubitParameters[i].t1 == 100_000, "Qubit \(i) T₁ should match uniform")
            #expect(profile.qubitParameters[i].singleQubitErrorRate == 0.001, "Qubit \(i) error rate should match")
        }
    }

    @Test("Average two-qubit error returns 0 for no edges")
    func averageTwoQubitErrorNoEdges() {
        let uniformParams = QubitNoiseParameters(
            t1: 100_000,
            t2: 80000,
            singleQubitErrorRate: 0.001,
            readoutError0Given1: 0.02,
            readoutError1Given0: 0.01,
        )
        let profile = HardwareNoiseProfile(
            name: "No Edges",
            qubitCount: 2,
            uniformParameters: uniformParams,
            edges: [],
        )

        #expect(profile.averageTwoQubitError == 0, "Average two-qubit error should be 0 with no edges")
        #expect(profile.edges.isEmpty, "Should have no edges")
    }

    @Test("Two-qubit channel fallback for unconnected qubits")
    func twoQubitChannelFallback() {
        let profile = HardwareNoiseProfile.ibmManila

        let connectedChannel = profile.twoQubitChannel(for: 0, 1)
        #expect(connectedChannel.errorProbability > 0, "Connected edge should have specific error")

        let unconnectedChannel = profile.twoQubitChannel(for: 0, 3)
        #expect(unconnectedChannel.errorProbability == profile.averageTwoQubitError,
                "Unconnected qubits should use average error rate")
    }
}

/// Test suite for per-qubit noise parameters.
/// Validates T₁/T₂ relaxation times, single-qubit error rates,
/// readout errors, and derived damping gamma calculations.
@Suite("Qubit Noise Parameters")
struct QubitNoiseParametersTests {
    @Test("Amplitude damping gamma calculation")
    func amplitudeDampingGammaCalc() {
        let params = QubitNoiseParameters(
            t1: 100_000,
            t2: 80000,
            singleQubitErrorRate: 0.001,
            readoutError0Given1: 0.02,
            readoutError1Given0: 0.01,
        )

        let gamma = params.amplitudeDampingGamma(gateTime: 35)
        let expected = 1.0 - exp(-35.0 / 100_000)

        #expect(abs(gamma - expected) < 1e-15, "Gamma calculation should match formula")
    }

    @Test("Phase damping gamma calculation")
    func phaseDampingGammaCalc() {
        let params = QubitNoiseParameters(
            t1: 100_000,
            t2: 80000,
            singleQubitErrorRate: 0.001,
            readoutError0Given1: 0.02,
            readoutError1Given0: 0.01,
        )

        let gamma = params.phaseDampingGamma(gateTime: 35)
        #expect(gamma >= 0, "Gamma should be non-negative")
    }

    @Test("Phase damping gamma returns 0 in T₁-limited regime")
    func phaseDampingGammaT1Limited() {
        let params = QubitNoiseParameters(
            t1: 100_000,
            t2: 200_000,
            singleQubitErrorRate: 0.001,
            readoutError0Given1: 0.02,
            readoutError1Given0: 0.01,
        )

        let gamma = params.phaseDampingGamma(gateTime: 35)

        #expect(gamma == 0, "Gamma should be 0 when T₂ = 2T₁ (T₁-limited, no pure dephasing)")
    }

    @Test("Measurement error model creation")
    func measurementErrorModelCreation() {
        let params = QubitNoiseParameters(
            t1: 100_000,
            t2: 80000,
            singleQubitErrorRate: 0.001,
            readoutError0Given1: 0.02,
            readoutError1Given0: 0.01,
        )

        let model = params.measurementErrorModel()

        #expect(model.confusionMatrix[1][0] == 0.02, "P(0|1) should be 0.02")
        #expect(model.confusionMatrix[0][1] == 0.01, "P(1|0) should be 0.01")
    }

    @Test("Optional frequency parameter")
    func optionalFrequencyParameter() {
        let paramsWithFreq = QubitNoiseParameters(
            t1: 100_000,
            t2: 80000,
            singleQubitErrorRate: 0.001,
            readoutError0Given1: 0.02,
            readoutError1Given0: 0.01,
            frequency: 5.0,
        )

        let paramsWithoutFreq = QubitNoiseParameters(
            t1: 100_000,
            t2: 80000,
            singleQubitErrorRate: 0.001,
            readoutError0Given1: 0.02,
            readoutError1Given0: 0.01,
        )

        #expect(paramsWithFreq.frequency == 5.0, "Should store frequency")
        #expect(paramsWithoutFreq.frequency == nil, "Should default to nil")
    }
}

/// Test suite for two-qubit edge noise parameters.
/// Validates qubit index sorting for canonical edge representation
/// and edge key generation for connectivity lookups.
@Suite("Edge Noise Parameters")
struct EdgeNoiseParametersTests {
    @Test("Qubit indices are sorted")
    func qubitIndicesSorted() {
        let edge = EdgeNoiseParameters(qubit1: 5, qubit2: 2, twoQubitErrorRate: 0.01)

        #expect(edge.qubit1 == 2, "qubit1 should be smaller")
        #expect(edge.qubit2 == 5, "qubit2 should be larger")
    }

    @Test("Edge key is canonical")
    func edgeKeyCanonical() {
        let edge1 = EdgeNoiseParameters(qubit1: 0, qubit2: 3, twoQubitErrorRate: 0.01)
        let edge2 = EdgeNoiseParameters(qubit1: 3, qubit2: 0, twoQubitErrorRate: 0.01)

        #expect(edge1.edgeKey == edge2.edgeKey, "Edge keys should be canonical")
    }
}

/// Test suite for per-qubit timing-aware noise.
/// Validates noise application with hardware profile parameters
/// including gate noise, idle decoherence, and measurement errors.
@Suite("Timing-Aware Noise Model")
struct TimingAwareNoiseModelTests {
    @Test("Creates from hardware profile")
    func createsFromProfile() {
        let profile = HardwareNoiseProfile.ibmManila
        let model = TimingAwareNoiseModel(profile: profile)

        #expect(model.profile.qubitCount == 5, "Should have 5 qubits")
    }

    @Test("Apply single-qubit noise")
    func applySingleQubitNoise() {
        let profile = HardwareNoiseProfile.ibmManila
        let model = TimingAwareNoiseModel(profile: profile)
        let dm = DensityMatrix(qubits: 5)

        let result = model.applySingleQubitNoise(qubit: 0, to: dm)

        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace should be preserved")
    }

    @Test("Apply idle noise to non-active qubits")
    func applyIdleNoise() {
        let profile = HardwareNoiseProfile.ibmManila
        let model = TimingAwareNoiseModel(profile: profile)
        let dm = DensityMatrix(qubits: 5)

        let result = model.applyIdleNoise(activeQubits: [0], gateTime: 300, to: dm)

        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace should be preserved")
    }

    @Test("Apply all noise combines gate and idle")
    func applyAllNoiseCombines() {
        let profile = HardwareNoiseProfile.ibmManila
        let model = TimingAwareNoiseModel(profile: profile)
        let dm = DensityMatrix(qubits: 5)

        let result = model.applyAllNoise(after: .hadamard, targetQubits: [0], to: dm)

        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace should be preserved")
    }

    @Test("Get measurement error models")
    func getMeasurementErrorModels() {
        let profile = HardwareNoiseProfile.ibmManila
        let model = TimingAwareNoiseModel(profile: profile)

        let models = model.measurementErrorModels()

        #expect(models.count == 5, "Should have 5 measurement error models")
    }

    @Test("Apply all noise handles two-qubit gate (CNOT)")
    func applyAllNoiseTwoQubitGate() {
        let profile = HardwareNoiseProfile.linearChain(qubits: 5)
        let model = TimingAwareNoiseModel(profile: profile)
        let dm = DensityMatrix(qubits: 5)

        let result = model.applyAllNoise(after: .cnot, targetQubits: [0, 1], to: dm)

        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace should be preserved")
    }

    @Test("Apply all noise handles three-qubit gate (Toffoli)")
    func applyAllNoiseThreeQubitGate() {
        let profile = HardwareNoiseProfile.linearChain(qubits: 5)
        let model = TimingAwareNoiseModel(profile: profile)
        let dm = DensityMatrix(qubits: 5)

        let result = model.applyAllNoise(after: .toffoli, targetQubits: [0, 1, 2], to: dm)

        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace should be preserved")
    }

    @Test("Apply all noise applies idle decoherence even for three-qubit gate")
    func applyAllNoiseThreeQubitWithIdle() {
        let profile = HardwareNoiseProfile.linearChain(qubits: 5)
        let model = TimingAwareNoiseModel(profile: profile)

        let dm = DensityMatrix.basis(qubits: 5, state: 0b11111)

        let result = model.applyAllNoise(after: .toffoli, targetQubits: [0, 1, 2], to: dm)

        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace should be preserved")
    }

    @Test("Apply two-qubit noise uses profile edge parameters")
    func applyTwoQubitNoiseUsesEdge() {
        let profile = HardwareNoiseProfile.ibmManila
        let model = TimingAwareNoiseModel(profile: profile)
        let dm = DensityMatrix(qubits: 5)

        let result = model.applyTwoQubitNoise(qubits: [0, 1], to: dm)

        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace should be preserved")
        #expect(result.purity() < 1.0, "Noise should reduce purity")
    }
}

/// Test suite for noisy circuit execution via DensityMatrixSimulator.
/// Validates ideal and noisy simulation, expectation values,
/// fidelity computation, circuit timing, and progress tracking.
@Suite("Density Matrix Simulator")
struct DensityMatrixSimulatorTests {
    @Test("Ideal simulator preserves purity")
    func idealPreservesPurity() async {
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let result = await simulator.execute(circuit)

        #expect(result.isPure(), "Ideal simulation should preserve purity")
    }

    @Test("Noisy simulator reduces purity")
    func noisyReducesPurity() async {
        let noise = NoiseModel.depolarizing(singleQubitError: 0.05, twoQubitError: 0.1)
        let simulator = DensityMatrixSimulator(noiseModel: noise)
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let result = await simulator.execute(circuit)

        #expect(!result.isPure(), "Noisy simulation should reduce purity")
    }

    @Test("Execute from custom initial state")
    func executeFromCustomInitial() async {
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        let initial = DensityMatrix.maximallyMixed(qubits: 2)
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)

        let result = await simulator.execute(circuit, from: initial)

        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace should be preserved")
    }

    @Test("Expectation value computation")
    func expectationValueComputation() async {
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        let z = Observable.pauliZ(qubit: 0)
        let expectation = await simulator.expectationValue(circuit, observable: z)

        #expect(abs(expectation) < 1e-10, "⟨Z⟩ should be 0 for |+⟩")
    }

    @Test("Fidelity computation")
    func fidelityComputation() async {
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        let invSqrt2 = 1.0 / sqrt(2.0)
        let idealState = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0), Complex(invSqrt2, 0),
        ])

        let fidelity = await simulator.fidelity(circuit, idealState: idealState)

        #expect(abs(fidelity - 1.0) < 1e-10, "Ideal simulation should have fidelity 1")
    }

    @Test("Circuit time calculation")
    func circuitTimeCalculation() {
        let simulator = DensityMatrixSimulator()
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])
        circuit.append(.pauliX, to: 1)

        let time = simulator.circuitTime(circuit)
        let expected = 35.0 + 300.0 + 35.0

        #expect(abs(time - expected) < 1e-10, "Time should sum gate durations")
    }

    @Test("Progress tracking")
    func progressTracking() async {
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        _ = await simulator.execute(circuit)
        let progress = await simulator.progress

        #expect(progress.executed == 2, "Should have executed 2 gates")
        #expect(progress.total == 2, "Should have 2 total gates")
        #expect(progress.percentage == 100.0, "Should be 100% complete")
    }

    @Test("Execute from pure state")
    func executeFromPureState() async {
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        let pureState = QuantumState(qubits: 2)
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)

        let result = await simulator.execute(circuit, from: pureState)

        #expect(abs(result.trace() - 1.0) < 1e-10, "Should preserve trace")
    }

    @Test("Execute with idle noise model")
    func executeWithIdleNoise() async {
        let noise = NoiseModel.typicalNISQWithIdle
        let simulator = DensityMatrixSimulator(noiseModel: noise)
        var circuit = QuantumCircuit(qubits: 3)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let result = await simulator.execute(circuit)

        #expect(abs(result.trace() - 1.0) < 0.02, "Trace should be approximately preserved with idle noise")
        #expect(!result.isPure(), "Noise should reduce purity")
    }

    @Test("Sample measurement outcomes")
    func sampleMeasurementOutcomes() async {
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.pauliX, to: 0)

        let outcomes = await simulator.sample(circuit, shots: 100, seed: 12345)

        #expect(outcomes.count == 100, "Should return requested number of samples")
        let correctCount = outcomes.count(where: { $0 == 1 })
        #expect(correctCount >= 95, "X|00⟩ = |01⟩ should predominantly give index 1")
    }

    @Test("Sample with measurement error")
    func sampleWithMeasurementError() async {
        let noise = NoiseModel.typicalNISQ
        let simulator = DensityMatrixSimulator(noiseModel: noise)
        let circuit = QuantumCircuit(qubits: 2)

        let outcomes = await simulator.sample(circuit, shots: 100, seed: 42)

        #expect(outcomes.count == 100, "Should return 100 samples")
    }

    @Test("Statistical analysis computes mean and std")
    func statisticalAnalysisComputation() async {
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        var circuit = QuantumCircuit(qubits: 1)
        circuit.append(.hadamard, to: 0)

        let z = Observable.pauliZ(qubit: 0)
        let stats = await simulator.statisticalAnalysis(circuit, observable: z, repetitions: 10)

        #expect(abs(stats.mean) < 1e-10, "Mean ⟨Z⟩ should be ~0 for |+⟩")
        #expect(stats.std < 1e-10, "Std should be ~0 for deterministic ideal simulation")
        #expect(stats.min <= stats.mean, "Min should be ≤ mean")
        #expect(stats.max >= stats.mean, "Max should be ≥ mean")
    }

    @Test("Estimated decoherence fidelity")
    func estimatedDecoherenceFidelityComputation() {
        let simulator = DensityMatrixSimulator()
        var circuit = QuantumCircuit(qubits: 2)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let fidelity = simulator.estimatedDecoherenceFidelity(
            circuit,
            t1: 100_000,
            t2: 80000,
        )

        #expect(fidelity > 0 && fidelity <= 1, "Fidelity should be in (0, 1]")
        #expect(fidelity < 1.0, "Non-zero circuit time should reduce fidelity")
    }

    @Test("Empty circuit progress is 100%")
    func emptyCircuitProgress() async {
        let simulator = DensityMatrixSimulator(noiseModel: .ideal)
        let circuit = QuantumCircuit(qubits: 2)

        _ = await simulator.execute(circuit)
        let progress = await simulator.progress

        #expect(progress.total == 0, "Empty circuit should have 0 total gates")
        #expect(progress.executed == 0, "Empty circuit should have 0 executed gates")
        #expect(progress.percentage == 100.0, "Empty circuit should be 100% complete")
    }
}

/// Test suite for per-qubit timing-aware simulation.
/// Validates TimingAwareDensityMatrixSimulator with hardware profiles
/// including per-qubit noise, readout errors, and circuit timing.
@Suite("Timing-Aware Density Matrix Simulator")
struct TimingAwareDensityMatrixSimulatorTests {
    @Test("Expectation value with per-qubit noise")
    func expectationValuePerQubit() async {
        let profile = HardwareNoiseProfile.ibmManila
        let simulator = TimingAwareDensityMatrixSimulator(profile: profile)
        var circuit = QuantumCircuit(qubits: 5)
        circuit.append(.hadamard, to: 0)

        let z = Observable.pauliZ(qubit: 0)
        let expectation = await simulator.expectationValue(circuit, observable: z)

        #expect(expectation.isFinite, "Expectation should be finite")
    }

    @Test("Sample with per-qubit readout errors")
    func samplePerQubitReadout() async {
        let profile = HardwareNoiseProfile.ibmManila
        let simulator = TimingAwareDensityMatrixSimulator(profile: profile)
        let circuit = QuantumCircuit(qubits: 5)

        let outcomes = await simulator.sample(circuit, shots: 100, seed: 42)

        #expect(outcomes.count == 100, "Should have 100 samples")
    }

    @Test("Fidelity with per-qubit noise")
    func fidelityPerQubit() async {
        let profile = HardwareNoiseProfile.ibmManila
        let simulator = TimingAwareDensityMatrixSimulator(profile: profile)
        var circuit = QuantumCircuit(qubits: 5)
        circuit.append(.pauliX, to: 0)

        let idealState = QuantumState.basis(qubits: 5, state: 1)
        let fidelity = await simulator.fidelity(circuit, idealState: idealState)

        #expect(fidelity > 0.9, "Fidelity should be high for simple circuit")
        #expect(fidelity < 1.0, "Fidelity should be reduced by noise")
    }

    @Test("Circuit time uses profile timings")
    func circuitTimeUsesProfile() {
        let profile = HardwareNoiseProfile.ibmManila
        let simulator = TimingAwareDensityMatrixSimulator(profile: profile)
        var circuit = QuantumCircuit(qubits: 5)
        circuit.append(.hadamard, to: 0)
        circuit.append(.cnot, to: [0, 1])

        let time = simulator.circuitTime(circuit)

        #expect(time == 35 + 300, "Should use IBM timings")
    }
}

/// Test suite for idle noise during gate execution.
/// Validates T₁/T₂ decoherence on idle qubits while other
/// qubits are executing gates, and profile-based configuration.
@Suite("Noise Model Idle Noise")
struct NoiseModelIdleNoiseTests {
    @Test("Apply noise with idle includes T₁/T₂")
    func applyNoiseWithIdleIncludesDecoherence() {
        let model = NoiseModel.typicalNISQWithIdle
        let dm = DensityMatrix(qubits: 3)

        let result = model.applyNoiseWithIdle(
            after: .hadamard,
            targetQubits: [0],
            to: dm,
            totalQubits: 3,
        )

        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace should be preserved")
    }

    @Test("Has idle noise flag")
    func hasIdleNoiseFlag() {
        let withIdle = NoiseModel.typicalNISQWithIdle
        let withoutIdle = NoiseModel.typicalNISQ

        #expect(withIdle.hasIdleNoise, "Should have idle noise")
        #expect(!withoutIdle.hasIdleNoise, "Should not have idle noise")
    }

    @Test("From profile with idle")
    func fromProfileWithIdle() {
        let profile = HardwareNoiseProfile.ibmManila
        let model = NoiseModel.fromWithIdle(profile: profile)

        #expect(model.hasIdleNoise, "Should have idle noise from profile")
        #expect(model.idleNoiseConfig != nil, "Idle config should be set")
    }
}
