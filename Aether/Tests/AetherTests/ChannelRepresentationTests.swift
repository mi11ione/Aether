// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for SuperoperatorRepresentation initialization and basic properties.
/// Validates construction from noise channels, matrix dimensions,
/// and element access via subscript for Liouville space representation.
@Suite("Superoperator Initialization")
struct SuperoperatorInitializationTests {
    @Test("Initialize from DepolarizingChannel creates correct dimension")
    func initializeFromDepolarizing() {
        let channel = DepolarizingChannel(errorProbability: 0.1)
        let superop = SuperoperatorRepresentation(channel: channel, qubits: 1)

        #expect(superop.qubits == 1, "Should have 1 qubit")
        #expect(superop.dimension == 4, "Dimension should be 2^(2*1) = 4")
        #expect(superop.matrix.count == 4, "Matrix should have 4 rows")
        #expect(superop.matrix[0].count == 4, "Matrix should have 4 columns")
    }

    @Test("Initialize from BitFlipChannel creates correct dimension")
    func initializeFromBitFlip() {
        let channel = BitFlipChannel(errorProbability: 0.05)
        let superop = SuperoperatorRepresentation(channel: channel, qubits: 1)

        #expect(superop.qubits == 1, "Should have 1 qubit")
        #expect(superop.dimension == 4, "Dimension should be 4 for single qubit")
    }

    @Test("Dimension property scales correctly")
    func dimensionScaling() {
        let channel = DepolarizingChannel(errorProbability: 0.1)
        let superop = SuperoperatorRepresentation(channel: channel, qubits: 1)

        #expect(superop.dimension == 4, "Dimension should be 2^(2*qubits) = 4 for 1 qubit")
    }

    @Test("Subscript access returns correct element")
    func subscriptAccess() {
        let channel = DepolarizingChannel(errorProbability: 0.1)
        let superop = SuperoperatorRepresentation(channel: channel, qubits: 1)

        let element = superop[row: 0, col: 0]

        #expect(element == superop.matrix[0][0], "Subscript should return same element as matrix access")
    }

    @Test("Identity channel has identity superoperator")
    func identityChannelSuperoperator() {
        let channel = DepolarizingChannel(errorProbability: 0.0)
        let superop = SuperoperatorRepresentation(channel: channel, qubits: 1)

        let tolerance = 1e-10
        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                let expected: Double = (i == j) ? 1.0 : 0.0
                let diff = (superop[row: i, col: j] - Complex(expected, 0)).magnitudeSquared
                #expect(diff < tolerance * tolerance, "Identity channel superoperator should be identity matrix at [\(i),\(j)]")
            }
        }
    }
}

/// Test suite for SuperoperatorRepresentation application to density matrices.
/// Validates vec(rho') = S * vec(rho) transformation and verifies
/// results match direct Kraus operator application.
@Suite("Superoperator Application")
struct SuperoperatorApplicationTests {
    @Test("Apply to ground state density matrix")
    func applyToGroundState() {
        let channel = DepolarizingChannel(errorProbability: 0.1)
        let superop = SuperoperatorRepresentation(channel: channel, qubits: 1)
        let dm = DensityMatrix(qubits: 1)

        let result = superop.apply(to: dm)

        #expect(result.qubits == 1, "Result should have 1 qubit")
        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace should be preserved")
    }

    @Test("Apply matches Kraus operator application")
    func applyMatchesKrausApplication() {
        let channel = BitFlipChannel(errorProbability: 0.2)
        let superop = SuperoperatorRepresentation(channel: channel, qubits: 1)
        let dm = DensityMatrix(qubits: 1)

        let superopResult = superop.apply(to: dm)
        let krausResult = channel.apply(to: dm, qubit: 0)

        let tolerance = 1e-10
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                let diff = (superopResult.element(row: i, col: j) - krausResult.element(row: i, col: j)).magnitudeSquared
                #expect(diff < tolerance * tolerance, "Superoperator application should match Kraus application at [\(i),\(j)]")
            }
        }
    }

    @Test("Apply to superposition state")
    func applyToSuperposition() {
        let channel = DepolarizingChannel(errorProbability: 0.1)
        let superop = SuperoperatorRepresentation(channel: channel, qubits: 1)
        let invSqrt2 = 1.0 / sqrt(2.0)
        let plus = QuantumState(qubits: 1, amplitudes: [Complex(invSqrt2, 0), Complex(invSqrt2, 0)])
        let dm = DensityMatrix(pureState: plus)

        let result = superop.apply(to: dm)

        #expect(abs(result.trace() - 1.0) < 1e-10, "Trace should be preserved for superposition state")
        #expect(result.isHermitian(), "Result should be Hermitian")
    }
}

/// Test suite for SuperoperatorRepresentation trace preservation and composition.
/// Validates isTracePreserving check and composed channel via matrix multiplication.
/// Ensures channel properties are preserved through superoperator operations.
@Suite("Superoperator Properties")
struct SuperoperatorPropertiesTests {
    @Test("DepolarizingChannel is trace preserving")
    func depolarizingIsTracePreserving() {
        let channel = DepolarizingChannel(errorProbability: 0.3)
        let superop = SuperoperatorRepresentation(channel: channel, qubits: 1)

        #expect(superop.isTracePreserving(), "Depolarizing channel should be trace preserving")
    }

    @Test("BitFlipChannel is trace preserving")
    func bitFlipIsTracePreserving() {
        let channel = BitFlipChannel(errorProbability: 0.15)
        let superop = SuperoperatorRepresentation(channel: channel, qubits: 1)

        #expect(superop.isTracePreserving(), "Bit flip channel should be trace preserving")
    }

    @Test("AmplitudeDampingChannel is trace preserving")
    func amplitudeDampingIsTracePreserving() {
        let channel = AmplitudeDampingChannel(gamma: 0.2)
        let superop = SuperoperatorRepresentation(channel: channel, qubits: 1)

        #expect(superop.isTracePreserving(), "Amplitude damping channel should be trace preserving")
    }

    @Test("Composed superoperators represent sequential channels")
    func composedSuperoperators() {
        let channel1 = DepolarizingChannel(errorProbability: 0.1)
        let channel2 = BitFlipChannel(errorProbability: 0.1)
        let s1 = SuperoperatorRepresentation(channel: channel1, qubits: 1)
        let s2 = SuperoperatorRepresentation(channel: channel2, qubits: 1)

        let composed = s1.composed(with: s2)
        let dm = DensityMatrix(qubits: 1)

        let composedResult = composed.apply(to: dm)
        let sequentialResult = s2.apply(to: s1.apply(to: dm))

        let tolerance = 1e-10
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                let diff = (composedResult.element(row: i, col: j) - sequentialResult.element(row: i, col: j)).magnitudeSquared
                #expect(diff < tolerance * tolerance, "Composed superoperator should match sequential application at [\(i),\(j)]")
            }
        }
    }

    @Test("Composed channel is trace preserving")
    func composedIsTracePreserving() {
        let channel1 = DepolarizingChannel(errorProbability: 0.1)
        let channel2 = BitFlipChannel(errorProbability: 0.2)
        let s1 = SuperoperatorRepresentation(channel: channel1, qubits: 1)
        let s2 = SuperoperatorRepresentation(channel: channel2, qubits: 1)

        let composed = s1.composed(with: s2)

        #expect(composed.isTracePreserving(), "Composition of trace-preserving channels should be trace preserving")
    }
}

/// Test suite for ChoiMatrix initialization and element access.
/// Validates construction from noise channels, matrix structure,
/// and Choi-Jamiolkowski correspondence for quantum channels.
@Suite("Choi Matrix Initialization")
struct ChoiMatrixInitializationTests {
    @Test("Initialize from DepolarizingChannel creates correct dimension")
    func initializeFromDepolarizing() {
        let channel = DepolarizingChannel(errorProbability: 0.1)
        let choi = ChoiMatrix(channel: channel, qubits: 1)

        #expect(choi.qubits == 1, "Should have 1 qubit")
        #expect(choi.dimension == 4, "Dimension should be 2^(2*1) = 4")
        #expect(choi.matrix.count == 4, "Matrix should have 4 rows")
        #expect(choi.matrix[0].count == 4, "Matrix should have 4 columns")
    }

    @Test("Initialize from BitFlipChannel")
    func initializeFromBitFlip() {
        let channel = BitFlipChannel(errorProbability: 0.1)
        let choi = ChoiMatrix(channel: channel, qubits: 1)

        #expect(choi.qubits == 1, "Should have 1 qubit")
        #expect(choi.dimension == 4, "Dimension should be 4")
    }

    @Test("Element access returns correct value")
    func elementAccess() {
        let channel = DepolarizingChannel(errorProbability: 0.1)
        let choi = ChoiMatrix(channel: channel, qubits: 1)

        let element = choi.element(row: 0, col: 0)

        #expect(element == choi.matrix[0][0], "Element access should return same value as matrix access")
    }

    @Test("Choi matrix dimension matches expected")
    func choiDimensionMatches() {
        let channel = DepolarizingChannel(errorProbability: 0.05)
        let choi = ChoiMatrix(channel: channel, qubits: 1)

        #expect(choi.dimension == 4, "Dimension should be d^2 = 4 for 1 qubit")
    }

    @Test("Identity channel Choi matrix is maximally entangled projector")
    func identityChannelChoi() {
        let channel = DepolarizingChannel(errorProbability: 0.0)
        let choi = ChoiMatrix(channel: channel, qubits: 1)

        let tolerance = 1e-10
        #expect(abs(choi.element(row: 0, col: 0).real - 1.0) < tolerance, "J[0,0] should be 1 for identity channel")
        #expect(abs(choi.element(row: 0, col: 3).real - 1.0) < tolerance, "J[0,3] should be 1 for identity channel")
        #expect(abs(choi.element(row: 3, col: 0).real - 1.0) < tolerance, "J[3,0] should be 1 for identity channel")
        #expect(abs(choi.element(row: 3, col: 3).real - 1.0) < tolerance, "J[3,3] should be 1 for identity channel")
    }
}

/// Test suite for ChoiMatrix trace preservation check.
/// Validates isTracePreserving via partial trace condition
/// for various quantum noise channels.
@Suite("Choi Matrix Properties")
struct ChoiMatrixPropertiesTests {
    @Test("DepolarizingChannel is trace preserving")
    func depolarizingIsTP() {
        let channel = DepolarizingChannel(errorProbability: 0.3)
        let choi = ChoiMatrix(channel: channel, qubits: 1)

        #expect(choi.isTracePreserving(), "Depolarizing channel should be trace preserving")
    }

    @Test("BitFlipChannel is trace preserving")
    func bitFlipIsTP() {
        let channel = BitFlipChannel(errorProbability: 0.2)
        let choi = ChoiMatrix(channel: channel, qubits: 1)

        #expect(choi.isTracePreserving(), "Bit flip channel should be trace preserving")
    }

    @Test("PhaseDampingChannel is trace preserving")
    func phaseDampingIsTP() {
        let channel = PhaseDampingChannel(gamma: 0.25)
        let choi = ChoiMatrix(channel: channel, qubits: 1)

        #expect(choi.isTracePreserving(tolerance: 1e-9), "Phase damping channel should be trace preserving")
    }

    @Test("PhaseFlipChannel is trace preserving")
    func phaseFlipIsTP() {
        let channel = PhaseFlipChannel(errorProbability: 0.2)
        let choi = ChoiMatrix(channel: channel, qubits: 1)

        #expect(choi.isTracePreserving(), "Phase flip channel should be trace preserving")
    }
}

/// Test suite for Choi matrix structure and Hermiticity.
/// Validates that Choi matrices are Hermitian for CPTP channels
/// and have correct structural properties.
@Suite("Choi Matrix Structure")
struct ChoiMatrixStructureTests {
    @Test("Choi matrix is Hermitian")
    func choiIsHermitian() {
        let channel = DepolarizingChannel(errorProbability: 0.2)
        let choi = ChoiMatrix(channel: channel, qubits: 1)

        let tolerance = 1e-10
        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                let diff = (choi.element(row: i, col: j) - choi.element(row: j, col: i).conjugate).magnitudeSquared
                #expect(diff < tolerance * tolerance, "Choi matrix should be Hermitian at [\(i),\(j)]")
            }
        }
    }

    @Test("BitFlip Choi matrix is Hermitian")
    func bitFlipChoiIsHermitian() {
        let channel = BitFlipChannel(errorProbability: 0.15)
        let choi = ChoiMatrix(channel: channel, qubits: 1)

        let tolerance = 1e-10
        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                let diff = (choi.element(row: i, col: j) - choi.element(row: j, col: i).conjugate).magnitudeSquared
                #expect(diff < tolerance * tolerance, "Choi matrix should be Hermitian at [\(i),\(j)]")
            }
        }
    }

    @Test("Choi trace equals dimension")
    func choiTraceEqualsDimension() {
        let channel = DepolarizingChannel(errorProbability: 0.3)
        let choi = ChoiMatrix(channel: channel, qubits: 1)

        var trace = Complex<Double>.zero
        for i in 0 ..< 4 {
            trace = trace + choi.element(row: i, col: i)
        }

        #expect(abs(trace.real - 2.0) < 1e-10, "Choi matrix trace should equal d = 2 for single qubit")
        #expect(abs(trace.imaginary) < 1e-10, "Choi matrix trace should be real")
    }
}

/// Test suite for ChiMatrix initialization and element access.
/// Validates construction from noise channels and Choi matrices,
/// and verifies correct Pauli basis transformation.
@Suite("Chi Matrix Initialization")
struct ChiMatrixInitializationTests {
    @Test("Initialize from DepolarizingChannel")
    func initializeFromDepolarizing() {
        let channel = DepolarizingChannel(errorProbability: 0.1)
        let chi = ChiMatrix(channel: channel, qubits: 1)

        #expect(chi.qubits == 1, "Should have 1 qubit")
        #expect(chi.dimension == 4, "Dimension should be 4 for single qubit")
        #expect(chi.matrix.count == 4, "Matrix should have 4 rows")
    }

    @Test("Initialize from Choi matrix")
    func initializeFromChoi() {
        let channel = BitFlipChannel(errorProbability: 0.15)
        let choi = ChoiMatrix(channel: channel, qubits: 1)
        let chi = ChiMatrix(choi: choi)

        #expect(chi.qubits == 1, "Should have 1 qubit")
        #expect(chi.dimension == 4, "Dimension should be 4")
    }

    @Test("Element access returns correct value")
    func elementAccess() {
        let channel = DepolarizingChannel(errorProbability: 0.2)
        let chi = ChiMatrix(channel: channel, qubits: 1)

        let element = chi.element(row: 0, col: 0)

        #expect(element == chi.matrix[0][0], "Element access should return same value as matrix access")
    }

    @Test("Dimension property matches matrix size")
    func dimensionProperty() {
        let channel = DepolarizingChannel(errorProbability: 0.1)
        let chi = ChiMatrix(channel: channel, qubits: 1)

        #expect(chi.dimension == chi.matrix.count, "Dimension should match matrix row count")
        #expect(chi.dimension == chi.matrix[0].count, "Dimension should match matrix column count")
    }

    @Test("Chi from channel equals Chi from Choi")
    func chiFromChannelEqualsChiFromChoi() {
        let channel = BitFlipChannel(errorProbability: 0.2)
        let chiDirect = ChiMatrix(channel: channel, qubits: 1)
        let choi = ChoiMatrix(channel: channel, qubits: 1)
        let chiFromChoi = ChiMatrix(choi: choi)

        let tolerance = 1e-10
        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                let diff = (chiDirect.element(row: i, col: j) - chiFromChoi.element(row: i, col: j)).magnitudeSquared
                #expect(diff < tolerance * tolerance, "Chi from channel should equal Chi from Choi at [\(i),\(j)]")
            }
        }
    }

    @Test("Chi matrix dimension matches expected")
    func chiDimensionMatches() {
        let channel = DepolarizingChannel(errorProbability: 0.05)
        let chi = ChiMatrix(channel: channel, qubits: 1)

        #expect(chi.dimension == 4, "Dimension should be d^2 = 4 for 1 qubit")
    }
}

/// Test suite for PauliTransferMatrix initialization and properties.
/// Validates construction from noise channels, real entries,
/// and isUnital check for channels preserving identity.
@Suite("Pauli Transfer Matrix Initialization")
struct PauliTransferMatrixInitializationTests {
    @Test("Initialize from DepolarizingChannel")
    func initializeFromDepolarizing() {
        let channel = DepolarizingChannel(errorProbability: 0.1)
        let ptm = PauliTransferMatrix(channel: channel, qubits: 1)

        #expect(ptm.qubits == 1, "Should have 1 qubit")
        #expect(ptm.matrix.count == 4, "Matrix should have 4 rows")
        #expect(ptm.matrix[0].count == 4, "Matrix should have 4 columns")
    }

    @Test("Initialize from BitFlipChannel")
    func initializeFromBitFlip() {
        let channel = BitFlipChannel(errorProbability: 0.2)
        let ptm = PauliTransferMatrix(channel: channel, qubits: 1)

        #expect(ptm.qubits == 1, "Should have 1 qubit")
    }

    @Test("Element access returns correct value")
    func elementAccess() {
        let channel = DepolarizingChannel(errorProbability: 0.1)
        let ptm = PauliTransferMatrix(channel: channel, qubits: 1)

        let element = ptm.element(row: 0, col: 0)

        #expect(element == ptm.matrix[0][0], "Element access should return same value as matrix access")
    }

    @Test("All entries are real")
    func allEntriesReal() {
        let channel = DepolarizingChannel(errorProbability: 0.2)
        let ptm = PauliTransferMatrix(channel: channel, qubits: 1)

        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                let value = ptm.element(row: i, col: j)
                #expect(value.isFinite, "PTM entry [\(i),\(j)] should be finite real number")
            }
        }
    }

    @Test("First row first column is 1 for CPTP channels")
    func firstElementIsOne() {
        let channel = DepolarizingChannel(errorProbability: 0.3)
        let ptm = PauliTransferMatrix(channel: channel, qubits: 1)

        #expect(abs(ptm.element(row: 0, col: 0) - 1.0) < 1e-10, "R[0,0] should be 1 for CPTP channels (trace preservation)")
    }
}

/// Test suite for PauliTransferMatrix unital property and Choi construction.
/// Validates isUnital check and alternative initialization from Choi matrix.
/// Verifies structural constraints of unital channels in PTM representation.
@Suite("Pauli Transfer Matrix Properties")
struct PauliTransferMatrixPropertiesTests {
    @Test("DepolarizingChannel is unital")
    func depolarizingIsUnital() {
        let channel = DepolarizingChannel(errorProbability: 0.3)
        let ptm = PauliTransferMatrix(channel: channel, qubits: 1)

        #expect(ptm.isUnital(), "Depolarizing channel should be unital")
    }

    @Test("BitFlipChannel is unital")
    func bitFlipIsUnital() {
        let channel = BitFlipChannel(errorProbability: 0.2)
        let ptm = PauliTransferMatrix(channel: channel, qubits: 1)

        #expect(ptm.isUnital(), "Bit flip channel should be unital")
    }

    @Test("AmplitudeDampingChannel is not unital")
    func amplitudeDampingNotUnital() {
        let channel = AmplitudeDampingChannel(gamma: 0.3)
        let ptm = PauliTransferMatrix(channel: channel, qubits: 1)

        #expect(!ptm.isUnital(), "Amplitude damping channel should not be unital")
    }

    @Test("Initialize from Choi matrix")
    func initializeFromChoi() {
        let channel = DepolarizingChannel(errorProbability: 0.2)
        let choi = ChoiMatrix(channel: channel, qubits: 1)
        let ptm = PauliTransferMatrix(choi: choi)

        #expect(ptm.qubits == 1, "Should have 1 qubit")
        #expect(ptm.isUnital(), "Depolarizing channel should be unital via Choi construction")
    }

    @Test("PTM from channel equals PTM from Choi")
    func ptmFromChannelEqualsPtmFromChoi() {
        let channel = BitFlipChannel(errorProbability: 0.15)
        let ptmDirect = PauliTransferMatrix(channel: channel, qubits: 1)
        let choi = ChoiMatrix(channel: channel, qubits: 1)
        let ptmFromChoi = PauliTransferMatrix(choi: choi)

        let tolerance = 1e-10
        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                let diff = abs(ptmDirect.element(row: i, col: j) - ptmFromChoi.element(row: i, col: j))
                #expect(diff < tolerance, "PTM from channel should equal PTM from Choi at [\(i),\(j)]")
            }
        }
    }

    @Test("Unital channel has first column [1,0,0,0]")
    func unitalFirstColumn() {
        let channel = DepolarizingChannel(errorProbability: 0.2)
        let ptm = PauliTransferMatrix(channel: channel, qubits: 1)

        let tolerance = 1e-10
        #expect(abs(ptm.element(row: 0, col: 0) - 1.0) < tolerance, "R[0,0] should be 1 for unital channel")
        #expect(abs(ptm.element(row: 1, col: 0)) < tolerance, "R[1,0] should be 0 for unital channel")
        #expect(abs(ptm.element(row: 2, col: 0)) < tolerance, "R[2,0] should be 0 for unital channel")
        #expect(abs(ptm.element(row: 3, col: 0)) < tolerance, "R[3,0] should be 0 for unital channel")
    }
}

/// Test suite for cross-conversions between representations.
/// Validates Superoperator<->Choi and Choi->Chi, Choi->PTM conversions
/// preserve channel information through round-trips.
@Suite("Channel Representation Conversions")
struct ChannelRepresentationConversionTests {
    @Test("Superoperator to Choi conversion")
    func superoperatorToChoi() {
        let channel = DepolarizingChannel(errorProbability: 0.2)
        let superop = SuperoperatorRepresentation(channel: channel, qubits: 1)
        let choiFromSuperop = ChoiMatrix(superoperator: superop)
        let choiDirect = ChoiMatrix(channel: channel, qubits: 1)

        let tolerance = 1e-10
        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                let diff = (choiFromSuperop.element(row: i, col: j) - choiDirect.element(row: i, col: j)).magnitudeSquared
                #expect(diff < tolerance * tolerance, "Choi from superoperator should match direct Choi at [\(i),\(j)]")
            }
        }
    }

    @Test("Choi to Superoperator conversion")
    func choiToSuperoperator() {
        let channel = BitFlipChannel(errorProbability: 0.15)
        let choi = ChoiMatrix(channel: channel, qubits: 1)
        let superopFromChoi = SuperoperatorRepresentation(choi: choi)
        let superopDirect = SuperoperatorRepresentation(channel: channel, qubits: 1)

        let tolerance = 1e-10
        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                let diff = (superopFromChoi[row: i, col: j] - superopDirect[row: i, col: j]).magnitudeSquared
                #expect(diff < tolerance * tolerance, "Superoperator from Choi should match direct superoperator at [\(i),\(j)]")
            }
        }
    }

    @Test("Round-trip Superoperator -> Choi -> Superoperator")
    func superoperatorChoiRoundTrip() {
        let channel = DepolarizingChannel(errorProbability: 0.25)
        let superop1 = SuperoperatorRepresentation(channel: channel, qubits: 1)
        let choi = ChoiMatrix(superoperator: superop1)
        let superop2 = SuperoperatorRepresentation(choi: choi)

        let tolerance = 1e-10
        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                let diff = (superop1[row: i, col: j] - superop2[row: i, col: j]).magnitudeSquared
                #expect(diff < tolerance * tolerance, "Round-trip should preserve superoperator at [\(i),\(j)]")
            }
        }
    }

    @Test("Round-trip Choi -> Superoperator -> Choi")
    func choiSuperoperatorRoundTrip() {
        let channel = BitFlipChannel(errorProbability: 0.2)
        let choi1 = ChoiMatrix(channel: channel, qubits: 1)
        let superop = SuperoperatorRepresentation(choi: choi1)
        let choi2 = ChoiMatrix(superoperator: superop)

        let tolerance = 1e-10
        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                let diff = (choi1.element(row: i, col: j) - choi2.element(row: i, col: j)).magnitudeSquared
                #expect(diff < tolerance * tolerance, "Round-trip should preserve Choi matrix at [\(i),\(j)]")
            }
        }
    }

    @Test("Converted Choi is Hermitian")
    func convertedChoiIsHermitian() {
        let channel = DepolarizingChannel(errorProbability: 0.3)
        let superop = SuperoperatorRepresentation(channel: channel, qubits: 1)
        let choiFromSuperop = ChoiMatrix(superoperator: superop)

        let tolerance = 1e-10
        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                let diff = (choiFromSuperop.element(row: i, col: j) - choiFromSuperop.element(row: j, col: i).conjugate).magnitudeSquared
                #expect(diff < tolerance * tolerance, "Converted Choi should be Hermitian at [\(i),\(j)]")
            }
        }
    }

    @Test("Converted Choi preserves trace preservation")
    func convertedChoiPreservesTP() {
        let channel = BitFlipChannel(errorProbability: 0.25)
        let superop = SuperoperatorRepresentation(channel: channel, qubits: 1)
        let choiFromSuperop = ChoiMatrix(superoperator: superop)

        #expect(choiFromSuperop.isTracePreserving(), "Choi from superoperator should be trace preserving")
    }

    @Test("Converted superoperator preserves trace preservation")
    func convertedSuperoperatorPreservesTP() {
        let channel = DepolarizingChannel(errorProbability: 0.2)
        let choi = ChoiMatrix(channel: channel, qubits: 1)
        let superopFromChoi = SuperoperatorRepresentation(choi: choi)

        #expect(superopFromChoi.isTracePreserving(), "Superoperator from Choi should be trace preserving")
    }
}

/// Test suite for depolarizing channel specific structure.
/// Validates known analytical properties of depolarizing noise
/// across all representations.
@Suite("Depolarizing Channel Structure")
struct DepolarizingChannelStructureTests {
    @Test("Depolarizing PTM has diagonal structure")
    func depolarizingPTMDiagonal() {
        let p = 0.3
        let channel = DepolarizingChannel(errorProbability: p)
        let ptm = PauliTransferMatrix(channel: channel, qubits: 1)

        let lambda = 1.0 - 4.0 * p / 3.0
        let tolerance = 1e-10

        #expect(abs(ptm.element(row: 0, col: 0) - 1.0) < tolerance, "R[0,0] should be 1")
        #expect(abs(ptm.element(row: 1, col: 1) - lambda) < tolerance, "R[1,1] should be 1-4p/3")
        #expect(abs(ptm.element(row: 2, col: 2) - lambda) < tolerance, "R[2,2] should be 1-4p/3")
        #expect(abs(ptm.element(row: 3, col: 3) - lambda) < tolerance, "R[3,3] should be 1-4p/3")
    }

    @Test("Depolarizing PTM off-diagonals are zero")
    func depolarizingPTMOffDiagonal() {
        let channel = DepolarizingChannel(errorProbability: 0.2)
        let ptm = PauliTransferMatrix(channel: channel, qubits: 1)

        let tolerance = 1e-10
        for i in 0 ..< 4 {
            for j in 0 ..< 4 where i != j {
                #expect(abs(ptm.element(row: i, col: j)) < tolerance, "Off-diagonal R[\(i),\(j)] should be zero for depolarizing")
            }
        }
    }

    @Test("Full depolarization gives identity PTM on identity row only")
    func fullDepolarizationPTM() {
        let channel = DepolarizingChannel(errorProbability: 0.75)
        let ptm = PauliTransferMatrix(channel: channel, qubits: 1)

        let tolerance = 1e-10
        #expect(abs(ptm.element(row: 0, col: 0) - 1.0) < tolerance, "R[0,0] should be 1 even at full depolarization")
        #expect(abs(ptm.element(row: 1, col: 1)) < tolerance, "R[1,1] should be 0 at full depolarization")
        #expect(abs(ptm.element(row: 2, col: 2)) < tolerance, "R[2,2] should be 0 at full depolarization")
        #expect(abs(ptm.element(row: 3, col: 3)) < tolerance, "R[3,3] should be 0 at full depolarization")
    }
}

/// Test suite for bit flip channel specific structure.
/// Validates known analytical properties of bit flip noise
/// including X-error structure in representations.
@Suite("Bit Flip Channel Structure")
struct BitFlipChannelStructureTests {
    @Test("Bit flip PTM preserves X, flips Y and Z")
    func bitFlipPTMStructure() {
        let p = 0.2
        let channel = BitFlipChannel(errorProbability: p)
        let ptm = PauliTransferMatrix(channel: channel, qubits: 1)

        let lambda = 1.0 - 2.0 * p
        let tolerance = 1e-10

        #expect(abs(ptm.element(row: 0, col: 0) - 1.0) < tolerance, "R[0,0] should be 1")
        #expect(abs(ptm.element(row: 1, col: 1) - 1.0) < tolerance, "R[1,1] should be 1 (X preserved)")
        #expect(abs(ptm.element(row: 2, col: 2) - lambda) < tolerance, "R[2,2] should be 1-2p (Y decays)")
        #expect(abs(ptm.element(row: 3, col: 3) - lambda) < tolerance, "R[3,3] should be 1-2p (Z decays)")
    }

    @Test("Bit flip is symmetric in Choi matrix")
    func bitFlipChoiSymmetric() {
        let channel = BitFlipChannel(errorProbability: 0.2)
        let choi = ChoiMatrix(channel: channel, qubits: 1)

        let tolerance = 1e-10
        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                let diff = (choi.element(row: i, col: j) - choi.element(row: j, col: i).conjugate).magnitudeSquared
                #expect(diff < tolerance * tolerance, "Choi matrix should be Hermitian at [\(i),\(j)]")
            }
        }
    }
}

/// Test suite for multi-qubit Pauli basis generation and representation dimensions.
/// Validates correct scaling of superoperator, Choi, Chi, and PTM dimensions.
/// Ensures two-qubit channels produce 16x16 matrices as expected.
@Suite("Multi-Qubit Pauli Basis Generation")
struct MultiQubitPauliBasisTests {
    @Test("Two-qubit superoperator generates 16 Pauli operators")
    func twoQubitSuperoperatorDimension() {
        let channel = TwoQubitDepolarizingChannel(errorProbability: 0.1)
        let superop = SuperoperatorRepresentation(channel: channel, qubits: 2)

        #expect(superop.qubits == 2, "Should have 2 qubits")
        #expect(superop.dimension == 16, "Dimension should be 2^(2*2) = 16 for 2 qubits")
        #expect(superop.matrix.count == 16, "Matrix should have 16 rows")
        #expect(superop.matrix[0].count == 16, "Matrix should have 16 columns")
    }

    @Test("Two-qubit Choi matrix dimension")
    func twoQubitChoiDimension() {
        let channel = TwoQubitDepolarizingChannel(errorProbability: 0.1)
        let choi = ChoiMatrix(channel: channel, qubits: 2)

        #expect(choi.qubits == 2, "Should have 2 qubits")
        #expect(choi.dimension == 16, "Dimension should be 16 for 2 qubits")
    }

    @Test("Two-qubit Chi matrix dimension")
    func twoQubitChiDimension() {
        let channel = TwoQubitDepolarizingChannel(errorProbability: 0.05)
        let chi = ChiMatrix(channel: channel, qubits: 2)

        #expect(chi.qubits == 2, "Should have 2 qubits")
        #expect(chi.dimension == 16, "Dimension should be 16 for 2 qubits")
    }

    @Test("Two-qubit PTM dimension")
    func twoQubitPTMDimension() {
        let channel = TwoQubitDepolarizingChannel(errorProbability: 0.1)
        let ptm = PauliTransferMatrix(channel: channel, qubits: 2)

        #expect(ptm.qubits == 2, "Should have 2 qubits")
        #expect(ptm.matrix.count == 16, "PTM should have 16 rows for 2 qubits")
        #expect(ptm.matrix[0].count == 16, "PTM should have 16 columns for 2 qubits")
    }
}

/// Test suite for trace preservation detection in valid CPTP channels.
/// All valid quantum channels with complete Kraus operators are trace-preserving by definition.
/// Validates that representations correctly identify valid channels as trace-preserving.
@Suite("Trace Preservation Detection")
struct TracePreservationDetectionTests {
    @Test("Superoperator confirms CustomKrausChannel is trace-preserving")
    func superoperatorCustomChannelTP() {
        let identity: [[Complex<Double>]] = [
            [Complex(1.0, 0), .zero],
            [.zero, Complex(1.0, 0)],
        ]
        let channel = CustomKrausChannel(krausOperators: [identity])
        let superop = SuperoperatorRepresentation(channel: channel, qubits: 1)

        #expect(superop.isTracePreserving(), "Valid CPTP channel should be trace-preserving")
    }

    @Test("Choi matrix confirms CustomKrausChannel is trace-preserving")
    func choiCustomChannelTP() {
        let identity: [[Complex<Double>]] = [
            [Complex(1.0, 0), .zero],
            [.zero, Complex(1.0, 0)],
        ]
        let channel = CustomKrausChannel(krausOperators: [identity])
        let choi = ChoiMatrix(channel: channel, qubits: 1)

        #expect(choi.isTracePreserving(), "Valid CPTP channel should be trace-preserving")
    }

    @Test("PTM confirms CustomKrausChannel identity is unital")
    func ptmCustomChannelUnital() {
        let identity: [[Complex<Double>]] = [
            [Complex(1.0, 0), .zero],
            [.zero, Complex(1.0, 0)],
        ]
        let channel = CustomKrausChannel(krausOperators: [identity])
        let ptm = PauliTransferMatrix(channel: channel, qubits: 1)

        #expect(ptm.isUnital(), "Identity channel should be unital")
    }
}

/// Validates complete positivity verification for Choi matrix representation.
/// Tests eigenvalue-based positivity checks with various quantum channels
/// including depolarizing, amplitude damping, and identity channels.
@Suite("Choi Matrix Complete Positivity")
struct ChoiMatrixCompletPositivityTests {
    @Test("Valid CPTP channel is completely positive")
    func depolarizingIsCP() {
        let channel = DepolarizingChannel(errorProbability: 0.3)
        let choi = ChoiMatrix(channel: channel, qubits: 1)

        #expect(choi.isCompletelyPositive(), "Depolarizing channel should be completely positive")
    }

    @Test("Amplitude damping is completely positive")
    func amplitudeDampingIsCP() {
        let channel = AmplitudeDampingChannel(gamma: 0.5)
        let choi = ChoiMatrix(channel: channel, qubits: 1)

        #expect(choi.isCompletelyPositive(), "Amplitude damping channel should be completely positive")
    }

    @Test("BitFlip channel is completely positive")
    func bitFlipIsCP() {
        let channel = BitFlipChannel(errorProbability: 0.2)
        let choi = ChoiMatrix(channel: channel, qubits: 1)

        #expect(choi.isCompletelyPositive(), "Bit flip channel should be completely positive")
    }
}

/// Validates Kraus operator extraction from Choi matrix representation.
/// Tests eigendecomposition-based extraction for standard quantum channels
/// and verifies the extracted operators reconstruct the original channel.
@Suite("Choi Matrix Kraus Extraction")
struct ChoiMatrixKrausExtractionTests {
    @Test("Extracted Kraus operators reconstruct channel")
    func krausOperatorsReconstructChannel() {
        let channel = DepolarizingChannel(errorProbability: 0.2)
        let choi = ChoiMatrix(channel: channel, qubits: 1)
        let extractedKraus = choi.krausOperators()

        #expect(extractedKraus.count > 0, "Should extract at least one Kraus operator")

        var reconstructedChoi = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: 4),
            count: 4,
        )

        for kraus in extractedKraus {
            for a in 0 ..< 2 {
                for c in 0 ..< 2 {
                    let rowIdx = a * 2 + c
                    for b in 0 ..< 2 {
                        for d in 0 ..< 2 {
                            let colIdx = b * 2 + d
                            reconstructedChoi[rowIdx][colIdx] = reconstructedChoi[rowIdx][colIdx] +
                                kraus[a][c] * kraus[b][d].conjugate
                        }
                    }
                }
            }
        }

        let tolerance = 1e-10
        for i in 0 ..< 4 {
            for j in 0 ..< 4 {
                let diff = (choi.element(row: i, col: j) - reconstructedChoi[i][j]).magnitudeSquared
                #expect(diff < tolerance * tolerance, "Reconstructed Choi should match original at [\(i),\(j)]")
            }
        }
    }

    @Test("Extracted Kraus operators from BitFlip reconstruct channel")
    func bitFlipKrausReconstruction() {
        let channel = BitFlipChannel(errorProbability: 0.15)
        let choi = ChoiMatrix(channel: channel, qubits: 1)
        let extractedKraus = choi.krausOperators()

        #expect(extractedKraus.count >= 1, "Should extract Kraus operators from bit flip channel")

        let dm = DensityMatrix(qubits: 1)
        var resultFromExtracted = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: 2),
            count: 2,
        )

        for kraus in extractedKraus {
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    for k in 0 ..< 2 {
                        for l in 0 ..< 2 {
                            resultFromExtracted[i][j] = resultFromExtracted[i][j] +
                                kraus[i][k] * dm.element(row: k, col: l) * kraus[j][l].conjugate
                        }
                    }
                }
            }
        }

        let directResult = channel.apply(to: dm, qubit: 0)

        let tolerance = 1e-10
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                let diff = (directResult.element(row: i, col: j) - resultFromExtracted[i][j]).magnitudeSquared
                #expect(diff < tolerance * tolerance, "Extracted Kraus should produce same result as original at [\(i),\(j)]")
            }
        }
    }

    @Test("Extracted Kraus operators satisfy trace preservation")
    func extractedKrausTracePreserving() {
        let channel = AmplitudeDampingChannel(gamma: 0.3)
        let choi = ChoiMatrix(channel: channel, qubits: 1)
        let extractedKraus = choi.krausOperators()

        var sumKdagK = [[Complex<Double>]](
            repeating: [Complex<Double>](repeating: .zero, count: 2),
            count: 2,
        )

        for kraus in extractedKraus {
            for i in 0 ..< 2 {
                for j in 0 ..< 2 {
                    for k in 0 ..< 2 {
                        sumKdagK[i][j] = sumKdagK[i][j] + kraus[k][i].conjugate * kraus[k][j]
                    }
                }
            }
        }

        let tolerance = 1e-10
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                let expected: Complex<Double> = (i == j) ? .one : .zero
                let diff = (sumKdagK[i][j] - expected).magnitudeSquared
                #expect(diff < tolerance * tolerance, "Sum of K†K should be identity at [\(i),\(j)]")
            }
        }
    }
}

/// Test suite for uncovered code paths in channel representations.
/// Covers SuperoperatorRepresentation.element method for direct element access.
@Suite("Channel Representation Uncovered Paths")
struct ChannelRepresentationUncoveredPathsTests {
    @Test("SuperoperatorRepresentation element method returns correct value")
    func superoperatorElementMethod() {
        let channel = DepolarizingChannel(errorProbability: 0.1)
        let superop = SuperoperatorRepresentation(channel: channel, qubits: 1)

        let elementValue = superop.element(row: 0, col: 0)

        #expect(elementValue == superop.matrix[0][0], "element(row:col:) should return the same value as direct matrix access")
    }
}
