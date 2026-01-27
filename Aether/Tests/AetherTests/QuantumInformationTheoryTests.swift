// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for Hermitian matrix eigendecomposition.
/// Validates spectral decomposition H = V diag(lambda) V-dagger
/// for standard quantum operator matrices and verifies eigenvector orthonormality.
@Suite("HermitianEigenDecomposition - Spectral Decomposition")
struct HermitianEigenDecompositionTests {
    @Test("Identity matrix has all eigenvalues equal to 1")
    func identityMatrixEigenvalues() {
        let identity: [[Complex<Double>]] = [
            [.one, .zero],
            [.zero, .one],
        ]
        let result = HermitianEigenDecomposition.decompose(matrix: identity)

        #expect(result.eigenvalues.count == 2, "2x2 identity should have 2 eigenvalues")
        #expect(abs(result.eigenvalues[0] - 1.0) < 1e-10, "First eigenvalue of identity should be 1")
        #expect(abs(result.eigenvalues[1] - 1.0) < 1e-10, "Second eigenvalue of identity should be 1")
    }

    @Test("Identity matrix eigenvectors form orthonormal basis")
    func identityMatrixEigenvectors() {
        let identity: [[Complex<Double>]] = [
            [.one, .zero],
            [.zero, .one],
        ]
        let result = HermitianEigenDecomposition.decompose(matrix: identity)

        for i in 0 ..< 2 {
            var norm = 0.0
            for component in result.eigenvectors[i] {
                norm += component.magnitudeSquared
            }
            #expect(abs(norm - 1.0) < 1e-10, "Eigenvector \(i) of identity should be normalized")
        }

        var innerProduct = Complex<Double>.zero
        for k in 0 ..< 2 {
            innerProduct = innerProduct + result.eigenvectors[0][k].conjugate * result.eigenvectors[1][k]
        }
        #expect(innerProduct.magnitudeSquared < 1e-10, "Eigenvectors of identity should be orthogonal")
    }

    @Test("Pauli Z eigenvalues are -1 and +1 in ascending order")
    func pauliZEigenvalues() {
        let pauliZ: [[Complex<Double>]] = [
            [.one, .zero],
            [.zero, Complex(-1, 0)],
        ]
        let result = HermitianEigenDecomposition.decompose(matrix: pauliZ)

        #expect(abs(result.eigenvalues[0] - -1.0) < 1e-10, "First eigenvalue of Pauli Z should be -1 (ascending order)")
        #expect(abs(result.eigenvalues[1] - 1.0) < 1e-10, "Second eigenvalue of Pauli Z should be +1 (ascending order)")
    }

    @Test("Pauli X eigenvalues are -1 and +1 in ascending order")
    func pauliXEigenvalues() {
        let pauliX: [[Complex<Double>]] = [
            [.zero, .one],
            [.one, .zero],
        ]
        let result = HermitianEigenDecomposition.decompose(matrix: pauliX)

        #expect(abs(result.eigenvalues[0] - -1.0) < 1e-10, "First eigenvalue of Pauli X should be -1 (ascending order)")
        #expect(abs(result.eigenvalues[1] - 1.0) < 1e-10, "Second eigenvalue of Pauli X should be +1 (ascending order)")
    }

    @Test("4x4 diagonal Hermitian matrix with known eigenvalues")
    func fourByFourDiagonalEigenvalues() {
        let matrix: [[Complex<Double>]] = [
            [Complex(3, 0), .zero, .zero, .zero],
            [.zero, Complex(1, 0), .zero, .zero],
            [.zero, .zero, Complex(4, 0), .zero],
            [.zero, .zero, .zero, Complex(2, 0)],
        ]
        let result = HermitianEigenDecomposition.decompose(matrix: matrix)

        #expect(abs(result.eigenvalues[0] - 1.0) < 1e-10, "Smallest eigenvalue should be 1.0")
        #expect(abs(result.eigenvalues[1] - 2.0) < 1e-10, "Second eigenvalue should be 2.0")
        #expect(abs(result.eigenvalues[2] - 3.0) < 1e-10, "Third eigenvalue should be 3.0")
        #expect(abs(result.eigenvalues[3] - 4.0) < 1e-10, "Largest eigenvalue should be 4.0")
    }

    @Test("Eigenvalues are returned in ascending order")
    func eigenvaluesAscendingOrder() {
        let matrix: [[Complex<Double>]] = [
            [Complex(2, 0), Complex(0, -1)],
            [Complex(0, 1), Complex(3, 0)],
        ]
        let result = HermitianEigenDecomposition.decompose(matrix: matrix)

        #expect(result.eigenvalues[0] <= result.eigenvalues[1], "Eigenvalues should be in ascending order")
    }

    @Test("Eigenvectors are orthonormal for non-degenerate Hermitian matrix")
    func eigenvectorsOrthonormal() {
        let matrix: [[Complex<Double>]] = [
            [Complex(2, 0), Complex(0, -1)],
            [Complex(0, 1), Complex(3, 0)],
        ]
        let result = HermitianEigenDecomposition.decompose(matrix: matrix)

        for i in 0 ..< 2 {
            var norm = 0.0
            for component in result.eigenvectors[i] {
                norm += component.magnitudeSquared
            }
            #expect(abs(norm - 1.0) < 1e-10, "Eigenvector \(i) should be normalized to unit length")
        }

        var innerProduct = Complex<Double>.zero
        for k in 0 ..< 2 {
            innerProduct = innerProduct + result.eigenvectors[0][k].conjugate * result.eigenvectors[1][k]
        }
        #expect(innerProduct.magnitudeSquared < 1e-10, "Eigenvectors should be orthogonal for non-degenerate spectrum")
    }

    @Test("Dimension property returns matrix size")
    func dimensionProperty() {
        let matrix2x2: [[Complex<Double>]] = [
            [.one, .zero],
            [.zero, .one],
        ]
        let result2 = HermitianEigenDecomposition.decompose(matrix: matrix2x2)
        #expect(result2.dimension == 2, "Dimension should be 2 for 2x2 matrix")

        let matrix4x4: [[Complex<Double>]] = [
            [Complex(1, 0), .zero, .zero, .zero],
            [.zero, Complex(2, 0), .zero, .zero],
            [.zero, .zero, Complex(3, 0), .zero],
            [.zero, .zero, .zero, Complex(4, 0)],
        ]
        let result4 = HermitianEigenDecomposition.decompose(matrix: matrix4x4)
        #expect(result4.dimension == 4, "Dimension should be 4 for 4x4 matrix")
    }

    @Test("Eigenvector equation H*v = lambda*v holds for each eigenpair")
    func eigenvectorEquationSatisfied() {
        let matrix: [[Complex<Double>]] = [
            [Complex(2, 0), Complex(0, -1)],
            [Complex(0, 1), Complex(3, 0)],
        ]
        let result = HermitianEigenDecomposition.decompose(matrix: matrix)

        for i in 0 ..< result.dimension {
            let lambda = result.eigenvalues[i]
            let v = result.eigenvectors[i]

            for row in 0 ..< result.dimension {
                var hv = Complex<Double>.zero
                for col in 0 ..< result.dimension {
                    hv = hv + matrix[row][col] * v[col]
                }
                let lambdaV = Complex(lambda, 0) * v[row]
                let diff = hv - lambdaV
                #expect(diff.magnitudeSquared < 1e-18, "H*v[\(row)] should equal lambda*v[\(row)] for eigenpair \(i)")
            }
        }
    }
}

/// Test suite for von Neumann entropy S(rho) = -Tr(rho log2 rho).
/// Validates entropy bounds for pure, mixed, and maximally mixed states
/// ensuring correct information-theoretic quantification of quantum uncertainty.
@Suite("Von Neumann Entropy - Quantum State Uncertainty")
struct VonNeumannEntropyTests {
    @Test("Pure ground state has zero entropy")
    func pureStateZeroEntropy() {
        let dm = DensityMatrix(qubits: 1)
        let entropy = dm.vonNeumannEntropy()

        #expect(abs(entropy) < 1e-10, "Pure state |0><0| should have zero von Neumann entropy")
    }

    @Test("Maximally mixed 1-qubit state has entropy 1.0 bit")
    func maximallyMixed1QubitEntropy() {
        let dm = DensityMatrix.maximallyMixed(qubits: 1)
        let entropy = dm.vonNeumannEntropy()

        #expect(abs(entropy - 1.0) < 1e-10, "Maximally mixed 1-qubit state should have entropy log2(2) = 1.0 bit")
    }

    @Test("Maximally mixed 2-qubit state has entropy 2.0 bits")
    func maximallyMixed2QubitEntropy() {
        let dm = DensityMatrix.maximallyMixed(qubits: 2)
        let entropy = dm.vonNeumannEntropy()

        #expect(abs(entropy - 2.0) < 1e-10, "Maximally mixed 2-qubit state should have entropy log2(4) = 2.0 bits")
    }

    @Test("Bell state (pure) has zero entropy")
    func bellStatePureZeroEntropy() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])
        let dm = DensityMatrix(pureState: bell)
        let entropy = dm.vonNeumannEntropy()

        #expect(abs(entropy) < 1e-10, "Bell state is pure, so full density matrix has zero entropy")
    }

    @Test("Partial trace of Bell state has entropy 1.0 bit (maximally entangled)")
    func bellStateReducedEntropy() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])
        let dm = DensityMatrix(pureState: bell)
        let reduced = dm.partialTrace(over: [1])
        let entropy = reduced.vonNeumannEntropy()

        #expect(abs(entropy - 1.0) < 1e-10, "Reduced state of Bell pair should be maximally mixed with entropy 1.0 bit")
    }

    @Test("Entropy is non-negative for any valid state")
    func entropyNonNegative() {
        let elements: [Complex<Double>] = [
            Complex(0.7, 0), Complex(0.1, 0.05),
            Complex(0.1, -0.05), Complex(0.3, 0),
        ]
        let dm = DensityMatrix(qubits: 1, elements: elements)
        let entropy = dm.vonNeumannEntropy()

        #expect(entropy >= -1e-10, "Von Neumann entropy should be non-negative for any density matrix")
    }
}

/// Test suite for trace distance D(rho, sigma) = 0.5 Tr|rho - sigma|.
/// Validates distinguishability metric ranging from 0 (identical)
/// to 1 (perfectly distinguishable orthogonal states).
@Suite("Trace Distance - State Distinguishability")
struct TraceDistanceTests {
    @Test("Trace distance between identical states is zero")
    func identicalStatesZeroDistance() {
        let dm = DensityMatrix(qubits: 1)
        let distance = dm.traceDistance(to: dm)

        #expect(abs(distance) < 1e-10, "Trace distance between identical states should be 0")
    }

    @Test("Orthogonal pure states have trace distance 1")
    func orthogonalPureStatesMaxDistance() {
        let zero = DensityMatrix.basis(qubits: 1, state: 0)
        let one = DensityMatrix.basis(qubits: 1, state: 1)
        let distance = zero.traceDistance(to: one)

        #expect(abs(distance - 1.0) < 1e-10, "Trace distance between |0><0| and |1><1| should be 1.0")
    }

    @Test("Pure state vs maximally mixed 1-qubit has distance 0.5")
    func pureVsMaximallyMixedDistance() {
        let pure = DensityMatrix.basis(qubits: 1, state: 0)
        let mixed = DensityMatrix.maximallyMixed(qubits: 1)
        let distance = pure.traceDistance(to: mixed)

        #expect(abs(distance - 0.5) < 1e-10, "Trace distance between |0><0| and I/2 should be 0.5")
    }

    @Test("Trace distance is symmetric: D(rho, sigma) = D(sigma, rho)")
    func traceDistanceSymmetry() {
        let rho = DensityMatrix.basis(qubits: 1, state: 0)
        let sigma = DensityMatrix.maximallyMixed(qubits: 1)

        let d1 = rho.traceDistance(to: sigma)
        let d2 = sigma.traceDistance(to: rho)

        #expect(abs(d1 - d2) < 1e-10, "Trace distance should be symmetric")
    }

    @Test("Trace distance between two different mixed states")
    func twoMixedStatesDistance() {
        let rho = DensityMatrix(qubits: 1, elements: [
            Complex(0.8, 0), Complex(0.1, 0),
            Complex(0.1, 0), Complex(0.2, 0),
        ])
        let sigma = DensityMatrix(qubits: 1, elements: [
            Complex(0.5, 0), Complex(0.3, 0),
            Complex(0.3, 0), Complex(0.5, 0),
        ])
        let distance = rho.traceDistance(to: sigma)

        #expect(distance >= -1e-10, "Trace distance should be non-negative")
        #expect(distance <= 1.0 + 1e-10, "Trace distance should be at most 1.0")
    }
}

/// Test suite for quantum state fidelity F(rho, sigma).
/// Validates overlap measure from 0 (orthogonal) to 1 (identical)
/// covering pure state shortcut, general mixed state, and symmetry.
@Suite("Fidelity - Quantum State Overlap")
struct FidelityTests {
    @Test("Fidelity between identical states is 1")
    func identicalStatesFidelityOne() {
        let dm = DensityMatrix(qubits: 1)
        let f = dm.fidelity(to: dm)

        #expect(abs(f - 1.0) < 1e-10, "Fidelity of a state with itself should be 1.0")
    }

    @Test("Orthogonal pure states have fidelity 0")
    func orthogonalPureStatesFidelityZero() {
        let zero = DensityMatrix.basis(qubits: 1, state: 0)
        let one = DensityMatrix.basis(qubits: 1, state: 1)
        let f = zero.fidelity(to: one)

        #expect(abs(f) < 1e-10, "Fidelity between |0><0| and |1><1| should be 0")
    }

    @Test("Pure state with maximally mixed has fidelity 0.5")
    func pureVsMaximallyMixedFidelity() {
        let pure = DensityMatrix.basis(qubits: 1, state: 0)
        let mixed = DensityMatrix.maximallyMixed(qubits: 1)
        let f = pure.fidelity(to: mixed)

        #expect(abs(f - 0.5) < 1e-10, "Fidelity between |0><0| and I/2 should be 0.5")
    }

    @Test("Pure state fidelity uses isPure shortcut path")
    func pureStateFidelityShortcut() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let plus = QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0), Complex(invSqrt2, 0),
        ])
        let dmPlus = DensityMatrix(pureState: plus)
        let dmZero = DensityMatrix.basis(qubits: 1, state: 0)

        let f = dmZero.fidelity(to: dmPlus)

        #expect(abs(f - 0.5) < 1e-10, "F(|0><0|, |+><+|) = |<0|+>|^2 = 0.5 via pure state shortcut")
    }

    @Test("General mixed state fidelity between two non-pure states")
    func generalMixedStateFidelity() {
        let rho = DensityMatrix.maximallyMixed(qubits: 1)
        let sigma = DensityMatrix(qubits: 1, elements: [
            Complex(0.7, 0), Complex(0.1, 0),
            Complex(0.1, 0), Complex(0.3, 0),
        ])
        let f = rho.fidelity(to: sigma)

        #expect(f >= -1e-10, "Fidelity should be non-negative")
        #expect(f <= 1.0 + 1e-10, "Fidelity should be at most 1.0")
    }

    @Test("Fidelity is symmetric: F(rho, sigma) = F(sigma, rho)")
    func fidelitySymmetry() {
        let rho = DensityMatrix.basis(qubits: 1, state: 0)
        let sigma = DensityMatrix.maximallyMixed(qubits: 1)

        let f1 = rho.fidelity(to: sigma)
        let f2 = sigma.fidelity(to: rho)

        #expect(abs(f1 - f2) < 1e-10, "Fidelity should satisfy F(rho,sigma) = F(sigma,rho)")
    }

    @Test("Fidelity between two pure states equals overlap squared")
    func twoNonOrthogonalPureStatesFidelity() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let zero = DensityMatrix.basis(qubits: 1, state: 0)
        let plus = DensityMatrix(pureState: QuantumState(qubits: 1, amplitudes: [
            Complex(invSqrt2, 0), Complex(invSqrt2, 0),
        ]))

        let f = zero.fidelity(to: plus)

        #expect(abs(f - 0.5) < 1e-10, "F(|0>,|+>) = |<0|+>|^2 = (1/sqrt(2))^2 = 0.5")
    }
}

/// Test suite for entanglement entropy via Schmidt decomposition.
/// Validates bipartite entanglement measure S = -sum lambda_i^2 log2(lambda_i^2)
/// for product, Bell, partially entangled, and GHZ states.
@Suite("Entanglement Entropy - Bipartite Entanglement Measure")
struct EntanglementEntropyTests {
    @Test("Product state |00> has zero entanglement entropy")
    func productStateZeroEntropy() {
        let product = QuantumState(qubits: 2)
        let entropy = QuantumInformationTheory.entanglementEntropy(
            state: product, subsystemAQubits: [0],
        )

        #expect(abs(entropy) < 1e-10, "Product state |00> should have zero entanglement entropy")
    }

    @Test("Bell state has entanglement entropy 1.0 bit")
    func bellStateMaximalEntropy() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])
        let entropy = QuantumInformationTheory.entanglementEntropy(
            state: bell, subsystemAQubits: [0],
        )

        #expect(abs(entropy - 1.0) < 1e-10, "Bell state should have maximal entanglement entropy of 1.0 bit")
    }

    @Test("Partially entangled state has entropy between 0 and 1")
    func partiallyEntangledStateEntropy() {
        let a = sqrt(0.8)
        let b = sqrt(0.2)
        let state = QuantumState(qubits: 2, amplitudes: [
            Complex(a, 0), .zero, .zero, Complex(b, 0),
        ])
        let entropy = QuantumInformationTheory.entanglementEntropy(
            state: state, subsystemAQubits: [0],
        )

        #expect(entropy > 1e-10, "Partially entangled state should have nonzero entropy")
        #expect(entropy < 1.0 - 1e-10, "Partially entangled state should have entropy less than 1 bit")

        let expected = -(0.8 * log2(0.8) + 0.2 * log2(0.2))
        #expect(abs(entropy - expected) < 1e-10, "Entropy should match analytical value -sum p_i log2 p_i")
    }

    @Test("GHZ state has entropy 1.0 for single-qubit subsystem")
    func ghzStateSingleQubitEntropy() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let ghz = QuantumState(qubits: 3, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, .zero,
            .zero, .zero, .zero, Complex(invSqrt2, 0),
        ])
        let entropy = QuantumInformationTheory.entanglementEntropy(
            state: ghz, subsystemAQubits: [0],
        )

        #expect(abs(entropy - 1.0) < 1e-10, "GHZ state with subsystem A = {qubit 0} should have entropy 1.0 bit")
    }

    @Test("GHZ state entropy is 1.0 for any single-qubit bipartition")
    func ghzStateAnySingleQubitEntropy() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let ghz = QuantumState(qubits: 3, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, .zero,
            .zero, .zero, .zero, Complex(invSqrt2, 0),
        ])

        for qubit in 0 ..< 3 {
            let entropy = QuantumInformationTheory.entanglementEntropy(
                state: ghz, subsystemAQubits: [qubit],
            )
            #expect(abs(entropy - 1.0) < 1e-10, "GHZ state should have entropy 1.0 for subsystem A = {qubit \(qubit)}")
        }
    }

    @Test("entropyFromProbabilities computes correct Shannon entropy")
    func entropyFromProbabilitiesUniform() {
        let uniform = [0.25, 0.25, 0.25, 0.25]
        let entropy = QuantumInformationTheory.entropyFromProbabilities(uniform)

        #expect(abs(entropy - 2.0) < 1e-10, "Uniform distribution over 4 outcomes should have entropy 2.0 bits")
    }

    @Test("entropyFromProbabilities handles zero probabilities correctly")
    func entropyFromProbabilitiesWithZeros() {
        let distribution = [1.0, 0.0, 0.0, 0.0]
        let entropy = QuantumInformationTheory.entropyFromProbabilities(distribution)

        #expect(abs(entropy) < 1e-10, "Delta distribution should have zero entropy")
    }

    @Test("entropyFromProbabilities handles near-zero probabilities")
    func entropyFromProbabilitiesNearZero() {
        let distribution = [0.5, 0.5, 1e-16, 1e-20]
        let entropy = QuantumInformationTheory.entropyFromProbabilities(distribution)

        #expect(abs(entropy - 1.0) < 1e-10, "Near-zero probabilities should be treated as zero, giving entropy 1.0")
    }
}

/// Test suite for quantum mutual information I(A:B) = S(A) + S(B) - S(AB).
/// Validates total correlations between subsystems ranging from 0
/// for product states to 2*S(A) for pure bipartite states.
@Suite("Mutual Information - Total Quantum Correlations")
struct MutualInformationTests {
    @Test("Product state has zero mutual information")
    func productStateZeroMutualInformation() {
        let product = DensityMatrix(qubits: 2)
        let mi = product.mutualInformation(subsystemA: [0], subsystemB: [1])

        #expect(abs(mi) < 1e-10, "Product state |00><00| should have zero mutual information")
    }

    @Test("Bell state has mutual information 2.0")
    func bellStateMutualInformation() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])
        let dm = DensityMatrix(pureState: bell)
        let mi = dm.mutualInformation(subsystemA: [0], subsystemB: [1])

        #expect(abs(mi - 2.0) < 1e-10, "Bell state should have mutual information I = 2*S(A) = 2.0 bits")
    }

    @Test("Mutual information is non-negative")
    func mutualInformationNonNegative() {
        let elements: [Complex<Double>] = [
            Complex(0.4, 0), Complex(0.1, 0), Complex(0.05, 0), Complex(0.0, 0),
            Complex(0.1, 0), Complex(0.1, 0), Complex(0.0, 0), Complex(0.05, 0),
            Complex(0.05, 0), Complex(0.0, 0), Complex(0.3, 0), Complex(0.1, 0),
            Complex(0.0, 0), Complex(0.05, 0), Complex(0.1, 0), Complex(0.2, 0),
        ]
        let dm = DensityMatrix(qubits: 2, elements: elements)
        let mi = dm.mutualInformation(subsystemA: [0], subsystemB: [1])

        #expect(mi >= -1e-10, "Mutual information should be non-negative by subadditivity of von Neumann entropy")
    }

    @Test("Maximally mixed state has zero mutual information")
    func maximallyMixedZeroMutualInformation() {
        let dm = DensityMatrix.maximallyMixed(qubits: 2)
        let mi = dm.mutualInformation(subsystemA: [0], subsystemB: [1])

        #expect(abs(mi) < 1e-10, "Maximally mixed 2-qubit state should have zero mutual information")
    }
}

/// Test suite for concurrence C(rho) of two-qubit density matrices.
/// Validates Wootters formula for entanglement quantification ranging
/// from 0 (separable) to 1 (maximally entangled Bell state).
@Suite("Concurrence - Two-Qubit Entanglement Measure")
struct ConcurrenceTests {
    @Test("Product state |00> has zero concurrence")
    func productStateZeroConcurrence() {
        let dm = DensityMatrix(qubits: 2)
        let c = dm.concurrence()

        #expect(abs(c) < 1e-10, "Product state |00><00| should have zero concurrence")
    }

    @Test("Bell state has nonzero concurrence indicating entanglement")
    func bellStateConcurrenceNonzero() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])
        let dm = DensityMatrix(pureState: bell)
        let c = dm.concurrence()

        #expect(c > 1e-10, "Bell state (|00>+|11>)/sqrt(2) should have nonzero concurrence indicating entanglement")
        #expect(c <= 1.0 + 1e-10, "Concurrence should not exceed 1.0")
    }

    @Test("Phi-plus Bell state concurrence is non-negative and bounded")
    func phiPlusBellStateConcurrence() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])
        let dm = DensityMatrix(pureState: bell)
        let c = dm.concurrence()

        #expect(c >= -1e-10, "Concurrence of Phi+ Bell state should be non-negative")
        #expect(c <= 1.0 + 1e-10, "Concurrence of Phi+ Bell state should not exceed 1.0")
    }

    @Test("Partially entangled Psi-type state has concurrence between 0 and 1")
    func partiallyEntangledConcurrence() {
        let a = sqrt(0.8)
        let b = sqrt(0.2)
        let state = QuantumState(qubits: 2, amplitudes: [
            .zero, Complex(a, 0), Complex(-b, 0), .zero,
        ])
        let dm = DensityMatrix(pureState: state)
        let c = dm.concurrence()

        #expect(c > 1e-10, "Partially entangled Psi-type state should have nonzero concurrence")
        #expect(c < 1.0 - 1e-10, "Partially entangled Psi-type state should have concurrence less than 1")
    }

    @Test("Maximally mixed 2-qubit state has zero concurrence")
    func maximallyMixedZeroConcurrence() {
        let dm = DensityMatrix.maximallyMixed(qubits: 2)
        let c = dm.concurrence()

        #expect(abs(c) < 1e-10, "Maximally mixed 2-qubit state should have zero concurrence (separable)")
    }

    @Test("Concurrence result is non-negative for pure entangled state")
    func concurrenceNonNegative() {
        let a = sqrt(0.6)
        let b = sqrt(0.4)
        let state = QuantumState(qubits: 2, amplitudes: [
            Complex(a, 0), .zero, .zero, Complex(b, 0),
        ])
        let dm = DensityMatrix(pureState: state)
        let c = dm.concurrence()

        #expect(c >= -1e-10, "Concurrence should be non-negative for any density matrix")
        #expect(c <= 1.0 + 1e-10, "Concurrence should not exceed 1.0")
    }

    @Test("Concurrence is bounded between 0 and 1 for basis state")
    func concurrenceBasisStateBounds() {
        let dm = DensityMatrix.basis(qubits: 2, state: 3)
        let c = dm.concurrence()

        #expect(c >= -1e-10, "Concurrence should be non-negative")
        #expect(c <= 1.0 + 1e-10, "Concurrence should not exceed 1.0")
    }
}

/// Test suite for Schmidt decomposition of bipartite quantum states.
/// Validates coefficient ordering, Schmidt rank, basis vector properties,
/// and entanglement entropy consistency for product, entangled, and multi-qubit states.
@Suite("Schmidt Decomposition - Bipartite State Structure")
struct SchmidtDecompositionTests {
    @Test("Product state |00> has Schmidt rank 1 with single coefficient 1")
    func productStateSchmidtRank() {
        let product = QuantumState(qubits: 2)
        let result = QuantumInformationTheory.schmidtDecomposition(
            state: product, subsystemAQubits: [0],
        )

        #expect(result.schmidtRank == 1, "Product state should have Schmidt rank 1")
        #expect(abs(result.coefficients[0] - 1.0) < 1e-10, "Single Schmidt coefficient for product state should be 1.0")
    }

    @Test("Bell state has Schmidt rank 2 with equal coefficients")
    func bellStateSchmidtDecomposition() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])
        let result = QuantumInformationTheory.schmidtDecomposition(
            state: bell, subsystemAQubits: [0],
        )

        #expect(result.schmidtRank == 2, "Bell state should have Schmidt rank 2")
        #expect(abs(result.coefficients[0] - invSqrt2) < 1e-10, "First Schmidt coefficient should be 1/sqrt(2)")
        #expect(abs(result.coefficients[1] - invSqrt2) < 1e-10, "Second Schmidt coefficient should be 1/sqrt(2)")
    }

    @Test("Schmidt coefficients are in descending order")
    func coefficientsDescendingOrder() {
        let a = sqrt(0.7)
        let b = sqrt(0.3)
        let state = QuantumState(qubits: 2, amplitudes: [
            Complex(a, 0), .zero, .zero, Complex(b, 0),
        ])
        let result = QuantumInformationTheory.schmidtDecomposition(
            state: state, subsystemAQubits: [0],
        )

        for i in 0 ..< result.coefficients.count - 1 {
            #expect(result.coefficients[i] >= result.coefficients[i + 1] - 1e-10,
                    "Schmidt coefficients should be in descending order: lambda[\(i)] >= lambda[\(i + 1)]")
        }
    }

    @Test("Schmidt coefficients are non-negative")
    func coefficientsNonNegative() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])
        let result = QuantumInformationTheory.schmidtDecomposition(
            state: bell, subsystemAQubits: [0],
        )

        for i in 0 ..< result.coefficients.count {
            #expect(result.coefficients[i] >= -1e-10, "Schmidt coefficient[\(i)] should be non-negative")
        }
    }

    @Test("Entanglement entropy from Schmidt result matches standalone function")
    func entropyConsistency() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])
        let result = QuantumInformationTheory.schmidtDecomposition(
            state: bell, subsystemAQubits: [0],
        )
        let standaloneEntropy = QuantumInformationTheory.entanglementEntropy(
            state: bell, subsystemAQubits: [0],
        )

        #expect(abs(result.entanglementEntropy - standaloneEntropy) < 1e-10,
                "Schmidt result entanglementEntropy should match standalone entanglementEntropy function")
    }

    @Test("3-qubit state with bipartition [0] vs [1,2]")
    func threeQubitBipartition() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let ghz = QuantumState(qubits: 3, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, .zero,
            .zero, .zero, .zero, Complex(invSqrt2, 0),
        ])
        let result = QuantumInformationTheory.schmidtDecomposition(
            state: ghz, subsystemAQubits: [0],
        )

        #expect(result.schmidtRank == 2, "GHZ state with {0} vs {1,2} should have Schmidt rank 2")
        #expect(abs(result.coefficients[0] - invSqrt2) < 1e-10, "First Schmidt coefficient of GHZ should be 1/sqrt(2)")
        #expect(abs(result.coefficients[1] - invSqrt2) < 1e-10, "Second Schmidt coefficient of GHZ should be 1/sqrt(2)")
        #expect(result.leftBasis.count == 2, "Should have 2 left Schmidt vectors for subsystem A")
        #expect(result.rightBasis.count == 2, "Should have 2 right Schmidt vectors for subsystem B")
        #expect(result.leftBasis[0].count == 2, "Left vectors for 1-qubit subsystem A should have dimension 2")
        #expect(result.rightBasis[0].count == 4, "Right vectors for 2-qubit subsystem B should have dimension 4")
    }

    @Test("Product state entanglement entropy from Schmidt result is zero")
    func productStateEntropyFromSchmidt() {
        let product = QuantumState(qubits: 2)
        let result = QuantumInformationTheory.schmidtDecomposition(
            state: product, subsystemAQubits: [0],
        )

        #expect(abs(result.entanglementEntropy) < 1e-10, "Product state should have zero entanglement entropy via Schmidt decomposition")
    }

    @Test("Bell state entanglement entropy from Schmidt result is 1.0 bit")
    func bellStateEntropyFromSchmidt() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])
        let result = QuantumInformationTheory.schmidtDecomposition(
            state: bell, subsystemAQubits: [0],
        )

        #expect(abs(result.entanglementEntropy - 1.0) < 1e-10, "Bell state entanglement entropy from Schmidt result should be 1.0 bit")
    }

    @Test("Schmidt decomposition with reversed subsystem gives same coefficients")
    func reversedSubsystemSameCoefficients() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let bell = QuantumState(qubits: 2, amplitudes: [
            Complex(invSqrt2, 0), .zero, .zero, Complex(invSqrt2, 0),
        ])

        let resultA = QuantumInformationTheory.schmidtDecomposition(
            state: bell, subsystemAQubits: [0],
        )
        let resultB = QuantumInformationTheory.schmidtDecomposition(
            state: bell, subsystemAQubits: [1],
        )

        #expect(resultA.coefficients.count == resultB.coefficients.count,
                "Schmidt coefficients count should be the same regardless of subsystem choice")

        for i in 0 ..< resultA.coefficients.count {
            #expect(abs(resultA.coefficients[i] - resultB.coefficients[i]) < 1e-10,
                    "Schmidt coefficient[\(i)] should be identical for either bipartition of Bell state")
        }
    }
}
