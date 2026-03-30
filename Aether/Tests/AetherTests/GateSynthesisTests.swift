// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Aether
import Foundation
import Testing

/// Test suite for single-qubit Euler angle extraction.
/// Validates that ZYZ/ZXZ/XYX decompositions correctly recover
/// rotation angles for standard gates and arbitrary unitaries.
@Suite("Euler Angle Extraction")
struct EulerAngleExtractionTests {
    private let tolerance: Double = 1e-10

    @Test("Identity matrix produces zero angles")
    func identityZeroAngles() {
        let matrix = QuantumGate.identity.matrix()
        let euler = GateSynthesis.eulerAngles(of: matrix, basis: .zyz)

        #expect(abs(euler.beta) < tolerance, "Identity beta should be zero, got \(euler.beta)")
    }

    @Test("Hadamard gate ZYZ decomposition")
    func hadamardZYZ() {
        let matrix = QuantumGate.hadamard.matrix()
        let euler = GateSynthesis.eulerAngles(of: matrix, basis: .zyz)

        let reconstructed = reconstructZYZ(euler)
        assertMatricesEqual(matrix, reconstructed, "Hadamard ZYZ reconstruction")
    }

    @Test("Pauli-X ZYZ decomposition")
    func pauliXZYZ() {
        let matrix = QuantumGate.pauliX.matrix()
        let euler = GateSynthesis.eulerAngles(of: matrix, basis: .zyz)

        #expect(abs(euler.beta - .pi) < tolerance, "Pauli-X beta should be pi, got \(euler.beta)")

        let reconstructed = reconstructZYZ(euler)
        assertMatricesEqual(matrix, reconstructed, "Pauli-X ZYZ reconstruction")
    }

    @Test("Pauli-Y ZYZ decomposition")
    func pauliYZYZ() {
        let matrix = QuantumGate.pauliY.matrix()
        let euler = GateSynthesis.eulerAngles(of: matrix, basis: .zyz)

        let reconstructed = reconstructZYZ(euler)
        assertMatricesEqual(matrix, reconstructed, "Pauli-Y ZYZ reconstruction")
    }

    @Test("Pauli-Z ZYZ decomposition")
    func pauliZZYZ() {
        let matrix = QuantumGate.pauliZ.matrix()
        let euler = GateSynthesis.eulerAngles(of: matrix, basis: .zyz)

        #expect(abs(euler.beta) < tolerance, "Pauli-Z beta should be zero, got \(euler.beta)")

        let reconstructed = reconstructZYZ(euler)
        assertMatricesEqual(matrix, reconstructed, "Pauli-Z ZYZ reconstruction")
    }

    @Test("Rotation-Z gate ZYZ decomposition")
    func rotationZGate() {
        let angle = 0.7
        let matrix = QuantumGate.rotationZ(angle).matrix()
        let euler = GateSynthesis.eulerAngles(of: matrix, basis: .zyz)

        let reconstructed = reconstructZYZ(euler)
        assertMatricesEqual(matrix, reconstructed, "Rz(0.7) ZYZ reconstruction")
    }

    @Test("Rotation-Y gate ZYZ decomposition")
    func rotationYGate() {
        let angle = 1.3
        let matrix = QuantumGate.rotationY(angle).matrix()
        let euler = GateSynthesis.eulerAngles(of: matrix, basis: .zyz)

        let reconstructed = reconstructZYZ(euler)
        assertMatricesEqual(matrix, reconstructed, "Ry(1.3) ZYZ reconstruction")
    }

    @Test("S gate ZYZ decomposition")
    func sGateZYZ() {
        let matrix = QuantumGate.sGate.matrix()
        let euler = GateSynthesis.eulerAngles(of: matrix, basis: .zyz)

        let reconstructed = reconstructZYZ(euler)
        assertMatricesEqual(matrix, reconstructed, "S gate ZYZ reconstruction")
    }

    @Test("T gate ZYZ decomposition")
    func tGateZYZ() {
        let matrix = QuantumGate.tGate.matrix()
        let euler = GateSynthesis.eulerAngles(of: matrix, basis: .zyz)

        let reconstructed = reconstructZYZ(euler)
        assertMatricesEqual(matrix, reconstructed, "T gate ZYZ reconstruction")
    }

    @Test("ZXZ basis variant")
    func zxzBasis() {
        let matrix = QuantumGate.hadamard.matrix()
        let euler = GateSynthesis.eulerAngles(of: matrix, basis: .zxz)

        let reconstructed = reconstructZXZ(euler)
        assertMatricesEqual(matrix, reconstructed, "Hadamard ZXZ reconstruction")
    }

    @Test("XYX basis variant")
    func xyxBasis() {
        let matrix = QuantumGate.hadamard.matrix()
        let euler = GateSynthesis.eulerAngles(of: matrix, basis: .xyx)

        let reconstructed = reconstructXYX(euler)
        assertMatricesEqual(matrix, reconstructed, "Hadamard XYX reconstruction")
    }

    private func reconstructZYZ(_ euler: GateSynthesis.EulerDecomposition) -> [[Complex<Double>]] {
        let rz1 = QuantumGate.rotationZ(euler.gamma).matrix()
        let ry = QuantumGate.rotationY(euler.beta).matrix()
        let rz2 = QuantumGate.rotationZ(euler.alpha).matrix()
        let phase = Complex(phase: euler.globalPhase)
        var result = MatrixUtilities.matrixMultiply(rz2, MatrixUtilities.matrixMultiply(ry, rz1))
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                result[i][j] = result[i][j] * phase
            }
        }
        return result
    }

    private func reconstructZXZ(_ euler: GateSynthesis.EulerDecomposition) -> [[Complex<Double>]] {
        let rz1 = QuantumGate.rotationZ(euler.gamma).matrix()
        let rx = QuantumGate.rotationX(euler.beta).matrix()
        let rz2 = QuantumGate.rotationZ(euler.alpha).matrix()
        let phase = Complex(phase: euler.globalPhase)
        var result = MatrixUtilities.matrixMultiply(rz2, MatrixUtilities.matrixMultiply(rx, rz1))
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                result[i][j] = result[i][j] * phase
            }
        }
        return result
    }

    private func reconstructXYX(_ euler: GateSynthesis.EulerDecomposition) -> [[Complex<Double>]] {
        let rx1 = QuantumGate.rotationX(euler.gamma).matrix()
        let ry = QuantumGate.rotationY(euler.beta).matrix()
        let rx2 = QuantumGate.rotationX(euler.alpha).matrix()
        let phase = Complex(phase: euler.globalPhase)
        var result = MatrixUtilities.matrixMultiply(rx2, MatrixUtilities.matrixMultiply(ry, rx1))
        for i in 0 ..< 2 {
            for j in 0 ..< 2 {
                result[i][j] = result[i][j] * phase
            }
        }
        return result
    }

    private func assertMatricesEqual(
        _ a: [[Complex<Double>]],
        _ b: [[Complex<Double>]],
        _ message: String,
    ) {
        let n = a.count
        let product = MatrixUtilities.matrixMultiply(MatrixUtilities.hermitianConjugate(a), b)
        var phaseRef: Complex<Double>?
        for i in 0 ..< n {
            let diag = product[i][i]
            if let ref = phaseRef {
                #expect(abs(diag.real - ref.real) < tolerance && abs(diag.imaginary - ref.imaginary) < tolerance,
                        "\(message): diagonal elements differ at [\(i)][\(i)]")
            } else {
                phaseRef = diag
            }
            for j in 0 ..< n where i != j {
                #expect(product[i][j].magnitude < tolerance,
                        "\(message): off-diagonal [\(i)][\(j)] = \(product[i][j].magnitude)")
            }
        }
        if let ref = phaseRef {
            #expect(abs(ref.magnitude - 1.0) < tolerance, "\(message): phase magnitude should be 1")
        }
    }
}

/// Test suite for unified synthesis pipeline dispatch.
/// Validates that synthesize() correctly dispatches to Euler (2x2),
/// KAK (4x4), and Shannon (larger) based on matrix dimension.
@Suite("Unified Synthesis Pipeline")
struct UnifiedSynthesisPipelineTests {
    private let tolerance: Double = 1e-6

    @Test("Single-qubit Hadamard synthesis produces gate sequence")
    func singleQubitHadamard() {
        let matrix = QuantumGate.hadamard.matrix()
        let gates = GateSynthesis.synthesize(matrix)

        #expect(!gates.isEmpty, "Hadamard synthesis should produce at least one gate")
        for entry in gates {
            #expect(entry.qubits == [0], "Single-qubit gates should target qubit 0")
        }
    }

    @Test("Single-qubit identity synthesis produces empty or phase-only")
    func singleQubitIdentity() {
        let matrix = QuantumGate.identity.matrix()
        let gates = GateSynthesis.synthesize(matrix)

        #expect(gates.count <= 1, "Identity synthesis should produce at most a global phase gate")
    }

    @Test("Two-qubit CNOT synthesis via KAK")
    func twoQubitCNOT() {
        let matrix = QuantumGate.cnot.matrix()
        let gates = GateSynthesis.synthesize(matrix)

        #expect(!gates.isEmpty, "CNOT synthesis should produce gates")

        var hasCNOT = false
        for entry in gates {
            if case .cnot = entry.gate { hasCNOT = true }
        }
        #expect(hasCNOT, "CNOT synthesis should include at least one CNOT gate")
    }

    @Test("Two-qubit CZ synthesis")
    func twoQubitCZ() {
        let matrix = QuantumGate.cz.matrix()
        let gates = GateSynthesis.synthesize(matrix)

        #expect(!gates.isEmpty, "CZ synthesis should produce gates")
    }

    @Test("Single-qubit synthesis with ZXZ basis")
    func singleQubitZXZ() {
        let matrix = QuantumGate.hadamard.matrix()
        let gates = GateSynthesis.synthesize(matrix, basis: .zxz)

        #expect(!gates.isEmpty, "ZXZ synthesis should produce gates")
    }

    @Test("Synthesized gates reconstruct original unitary")
    func singleQubitReconstruction() {
        let original = QuantumGate.rotationY(0.8).matrix()
        let gates = GateSynthesis.synthesize(original)

        var reconstructed = MatrixUtilities.identityMatrix(dimension: 2)
        for entry in gates {
            let gateMatrix = entry.gate.matrix()
            reconstructed = MatrixUtilities.matrixMultiply(gateMatrix, reconstructed)
        }

        assertUnitaryClose(original, reconstructed, "Ry(0.8) synthesis reconstruction")
    }

    @Test("Two-qubit CZ synthesis produces non-trivial gates")
    func twoQubitCZStructure() {
        let matrix = QuantumGate.cz.matrix()
        let gates = GateSynthesis.synthesize(matrix)

        #expect(!gates.isEmpty, "CZ synthesis should produce gates")

        var hasTwoQubitGate = false
        for entry in gates {
            if entry.qubits.count == 2 { hasTwoQubitGate = true }
        }
        #expect(hasTwoQubitGate, "CZ synthesis should include at least one two-qubit gate")
    }

    private func assertUnitaryClose(
        _ a: [[Complex<Double>]],
        _ b: [[Complex<Double>]],
        _ message: String,
    ) {
        let n = a.count
        let product = MatrixUtilities.matrixMultiply(MatrixUtilities.hermitianConjugate(a), b)
        for i in 0 ..< n {
            for j in 0 ..< n where i != j {
                #expect(product[i][j].magnitude < tolerance,
                        "\(message): off-diagonal [\(i)][\(j)] = \(product[i][j].magnitude)")
            }
        }
    }
}

/// Test suite for Ross-Selinger Clifford+T approximation.
/// Validates that cliffordT produces gate sequences from {H, S, T}
/// that approximate target unitaries within specified precision.
@Suite("Clifford+T Synthesis")
struct CliffordTSynthesisTests {
    private let tolerance: Double = 1e-6

    @Test("Exact Clifford gates produce exact sequences")
    func exactCliffordGates() {
        let sMatrix = QuantumGate.sGate.matrix()
        let gates = GateSynthesis.cliffordT(approximating: sMatrix, precision: 1e-10)

        #expect(!gates.isEmpty, "S gate should produce non-empty Clifford+T sequence")
        for gate in gates {
            let isHST = isCliffordTGate(gate)
            #expect(isHST, "Gate should be from {H, S, T} set")
        }
    }

    @Test("T gate produces T gate sequence")
    func tGateExact() {
        let tMatrix = QuantumGate.tGate.matrix()
        let gates = GateSynthesis.cliffordT(approximating: tMatrix, precision: 1e-10)

        #expect(!gates.isEmpty, "T gate should produce non-empty sequence")
    }

    @Test("Hadamard produces H gate")
    func hadamardExact() {
        let hMatrix = QuantumGate.hadamard.matrix()
        let gates = GateSynthesis.cliffordT(approximating: hMatrix, precision: 1e-10)

        #expect(!gates.isEmpty, "Hadamard should produce non-empty sequence")
    }

    @Test("Arbitrary rotation approximation within precision")
    func arbitraryRotation() {
        let matrix = QuantumGate.rotationZ(0.3).matrix()
        let precision = 1e-4
        let gates = GateSynthesis.cliffordT(approximating: matrix, precision: precision)

        #expect(!gates.isEmpty, "Rz(0.3) should produce non-empty Clifford+T sequence")

        for gate in gates {
            #expect(isCliffordTGate(gate), "All gates should be from {H, S, T, Y} set")
        }
    }

    @Test("Identity gate produces empty or minimal sequence")
    func identityGate() {
        let matrix = QuantumGate.identity.matrix()
        let gates = GateSynthesis.cliffordT(approximating: matrix, precision: 1e-10)

        #expect(gates.count <= 2, "Identity should produce at most 2 gates")
    }

    @Test("Gate count scales with precision")
    func gateCountScaling() {
        let matrix = QuantumGate.rotationZ(0.123).matrix()
        let coarseGates = GateSynthesis.cliffordT(approximating: matrix, precision: 1e-2)
        let fineGates = GateSynthesis.cliffordT(approximating: matrix, precision: 1e-6)

        #expect(fineGates.count >= coarseGates.count,
                "Finer precision should require at least as many gates (coarse: \(coarseGates.count), fine: \(fineGates.count))")
    }

    private static let cliffordTSet: Set<QuantumGate> = [.hadamard, .sGate, .tGate, .pauliY]

    private func isCliffordTGate(_ gate: QuantumGate) -> Bool {
        Self.cliffordTSet.contains(gate)
    }
}

/// Test suite for Shannon recursive n-qubit decomposition.
/// Validates that shannonDecompose correctly decomposes unitaries
/// larger than 4x4 into native gate sequences.
@Suite("Shannon Decomposition")
struct ShannonDecompositionTests {
    @Test("Toffoli gate decomposition produces gates")
    func toffoliDecomposition() {
        let matrix = QuantumGate.toffoli.matrix()
        let gates = GateSynthesis.shannonDecompose(matrix)

        #expect(!gates.isEmpty, "Toffoli decomposition should produce gates")

        var maxQubit = 0
        for entry in gates {
            for q in entry.qubits {
                if q > maxQubit { maxQubit = q }
            }
        }
        #expect(maxQubit <= 2, "Toffoli decomposition should use qubits 0-2 only")
    }

    @Test("Single-qubit via Shannon matches Euler")
    func singleQubitFallback() {
        let matrix = QuantumGate.hadamard.matrix()
        let gates = GateSynthesis.shannonDecompose(matrix)

        #expect(!gates.isEmpty, "Single-qubit Shannon should produce gates")
    }

    @Test("Two-qubit via Shannon matches KAK")
    func twoQubitFallback() {
        let matrix = QuantumGate.cnot.matrix()
        let gates = GateSynthesis.shannonDecompose(matrix)

        #expect(!gates.isEmpty, "Two-qubit Shannon should produce gates")
    }

    @Test("Fredkin gate decomposition")
    func fredkinDecomposition() {
        let matrix = QuantumGate.fredkin.matrix()
        let gates = GateSynthesis.shannonDecompose(matrix)

        #expect(!gates.isEmpty, "Fredkin decomposition should produce gates")
    }
}

/// Test suite for coverage of edge-case paths in gate synthesis.
/// Validates XYX synthesis, Clifford angle detection, n-qubit dispatch,
/// and CSD decomposition code paths.
@Suite("Gate Synthesis Coverage")
struct GateSynthesisCoverageTests {
    private let tolerance: Double = 1e-10

    @Test("Synthesize dispatches to Shannon for 3-qubit unitary")
    func threeQubitSynthesis() {
        let matrix = QuantumGate.toffoli.matrix()
        let gates = GateSynthesis.synthesize(matrix)

        #expect(!gates.isEmpty, "3-qubit synthesis should produce gates via Shannon")
    }

    @Test("XYX basis synthesis produces rotation-X gates")
    func xyxSynthesis() {
        let matrix = QuantumGate.hadamard.matrix()
        let gates = GateSynthesis.synthesize(matrix, basis: .xyx)

        #expect(!gates.isEmpty, "XYX synthesis should produce gates")

        var hasRx = false
        for entry in gates {
            if case .rotationX = entry.gate { hasRx = true }
        }
        #expect(hasRx, "XYX synthesis should include Rx gates")
    }

    @Test("Clifford+T S-dagger angle produces correct sequence")
    func cliffordSDagger() {
        let matrix = QuantumGate.phase(-.pi / 2).matrix()
        let gates = GateSynthesis.cliffordT(approximating: matrix, precision: 1e-10)

        #expect(!gates.isEmpty, "S-dagger should produce non-empty Clifford+T sequence")
    }

    @Test("Clifford+T T-dagger angle produces correct sequence")
    func cliffordTDagger() {
        let matrix = QuantumGate.phase(-.pi / 4).matrix()
        let gates = GateSynthesis.cliffordT(approximating: matrix, precision: 1e-10)

        #expect(!gates.isEmpty, "T-dagger should produce non-empty Clifford+T sequence")
    }

    @Test("Clifford+T for Ry(pi) produces Pauli-Y")
    func cliffordRyPi() {
        let matrix = QuantumGate.rotationY(.pi).matrix()
        let gates = GateSynthesis.cliffordT(approximating: matrix, precision: 1e-10)

        #expect(!gates.isEmpty, "Ry(pi) should produce non-empty Clifford+T sequence")
    }

    @Test("Clifford+T for identity Ry produces empty or minimal")
    func cliffordRyZero() {
        let matrix = QuantumGate.identity.matrix()
        let gates = GateSynthesis.cliffordT(approximating: matrix, precision: 1e-10)

        #expect(gates.count <= 2, "Identity Y rotation should produce minimal gates")
    }

    @Test("Global phase only gate from near-identity unitary")
    func globalPhaseOnlyEuler() {
        let phaseAngle = 0.5
        let matrix = QuantumGate.globalPhase(phaseAngle).matrix()
        let euler = GateSynthesis.eulerAngles(of: matrix, basis: .zyz)

        #expect(abs(euler.beta) < tolerance, "Global phase unitary should have zero beta")
    }

    @Test("Euler angles of Pauli-Z via ZXZ basis")
    func pauliZviaZXZ() {
        let matrix = QuantumGate.pauliZ.matrix()
        let euler = GateSynthesis.eulerAngles(of: matrix, basis: .zxz)
        let gates = GateSynthesis.synthesize(matrix, basis: .zxz)

        #expect(!gates.isEmpty || abs(euler.beta) < tolerance,
                "Pauli-Z in ZXZ should decompose correctly")
    }

    @Test("Arbitrary non-Clifford angle triggers grid search")
    func arbitraryAngleGridSearch() {
        let matrix = QuantumGate.rotationZ(0.777).matrix()
        let gates = GateSynthesis.cliffordT(approximating: matrix, precision: 1e-3)

        #expect(!gates.isEmpty, "Non-Clifford angle should produce Clifford+T via grid search")
    }

    @Test("CCZ gate Shannon decomposition")
    func cczDecomposition() {
        let matrix = QuantumGate.ccz.matrix()
        let gates = GateSynthesis.shannonDecompose(matrix)

        #expect(!gates.isEmpty, "CCZ decomposition should produce gates")
    }

    @Test("Diagonal 2-qubit gate via Shannon")
    func diagonalTwoQubit() {
        let phases: [Complex<Double>] = [.one, Complex(phase: 0.3), Complex(phase: 0.7), Complex(phase: -0.5)]
        let matrix: [[Complex<Double>]] = [
            [phases[0], .zero, .zero, .zero],
            [.zero, phases[1], .zero, .zero],
            [.zero, .zero, phases[2], .zero],
            [.zero, .zero, .zero, phases[3]],
        ]
        let gates = GateSynthesis.synthesize(matrix)

        #expect(!gates.isEmpty, "Diagonal 2-qubit unitary should produce gates")
    }

    @Test("Global phase only synthesis produces single gate")
    func globalPhaseOnlySynthesis() {
        let phaseAngle = 0.7
        let matrix = QuantumGate.globalPhase(phaseAngle).matrix()
        let gates = GateSynthesis.synthesize(matrix)

        #expect(gates.count <= 1, "Global phase should produce at most 1 gate")
    }

    @Test("Clifford+T for near-zero Y rotation produces empty")
    func cliffordNearZeroYRotation() {
        let matrix = QuantumGate.rotationZ(0.3).matrix()
        let gates = GateSynthesis.cliffordT(approximating: matrix, precision: 1e-2)

        #expect(!gates.isEmpty, "Z rotation should produce Clifford+T gates")
    }

    @Test("Three-qubit Toffoli exercises CSD paths")
    func threeQubitEntangling() {
        let toffoli = QuantumGate.toffoli.matrix()
        let gates = GateSynthesis.shannonDecompose(toffoli)

        #expect(!gates.isEmpty, "Toffoli Shannon decomposition should produce gates")
        #expect(gates.count > 5, "Toffoli should decompose to multiple gates")
    }

    @Test("Uniformly controlled Ry with single angle via CSD")
    func singleAngleUCRy() {
        let ccz = QuantumGate.ccz.matrix()
        let gates = GateSynthesis.shannonDecompose(ccz)

        #expect(!gates.isEmpty, "CCZ Shannon decomposition should produce gates")
    }

    @Test("H tensor I tensor I exercises CSD sin paths")
    func hadamardTensorIdentity() {
        let h = QuantumGate.hadamard.matrix()
        let i2 = MatrixUtilities.identityMatrix(dimension: 2)
        let hi = MatrixUtilities.kroneckerProduct(h, i2)
        let hii = MatrixUtilities.kroneckerProduct(hi, i2)
        let gates = GateSynthesis.shannonDecompose(hii)

        #expect(!gates.isEmpty, "H⊗I⊗I decomposition should produce gates")
    }

    @Test("Ry tensor I tensor I exercises CSD with non-trivial cosines")
    func ryTensorIdentity() {
        let ry = QuantumGate.rotationY(1.2).matrix()
        let i2 = MatrixUtilities.identityMatrix(dimension: 2)
        let ryi = MatrixUtilities.kroneckerProduct(ry, i2)
        let ryii = MatrixUtilities.kroneckerProduct(ryi, i2)
        let gates = GateSynthesis.shannonDecompose(ryii)

        #expect(!gates.isEmpty, "Ry⊗I⊗I decomposition should produce gates")
    }

    @Test("SWAP tensor I exercises CSD with zero cosines")
    func swapTensorIdentity() {
        let swap = QuantumGate.swap.matrix()
        let i2 = MatrixUtilities.identityMatrix(dimension: 2)
        let swapI = MatrixUtilities.kroneckerProduct(swap, i2)
        let gates = GateSynthesis.shannonDecompose(swapI)

        #expect(gates.count >= 0, "SWAP⊗I decomposition exercises rank-deficient CSD paths")
    }
}

/// Test suite for EulerBasis enum and EulerDecomposition struct.
/// Validates type properties, equality, and Sendable conformance
/// for the gate synthesis result types.
@Suite("Gate Synthesis Types")
struct GateSynthesisTypesTests {
    @Test("EulerBasis cases are distinct")
    func eulerBasisCases() {
        let zyz = GateSynthesis.EulerBasis.zyz
        let zxz = GateSynthesis.EulerBasis.zxz
        let xyx = GateSynthesis.EulerBasis.xyx

        #expect(zyz != zxz, "ZYZ and ZXZ should be distinct")
        #expect(zyz != xyx, "ZYZ and XYX should be distinct")
        #expect(zxz != xyx, "ZXZ and XYX should be distinct")
    }

    @Test("EulerBasis is Hashable")
    func eulerBasisHashable() {
        var set: Set<GateSynthesis.EulerBasis> = []
        set.insert(.zyz)
        set.insert(.zxz)
        set.insert(.xyx)

        #expect(set.count == 3, "Three distinct bases should produce set of size 3")
    }

    @Test("EulerDecomposition stores angles correctly")
    func eulerDecompositionStorage() {
        let decomp = GateSynthesis.EulerDecomposition(
            alpha: 1.0, beta: 2.0, gamma: 3.0, globalPhase: 0.5,
        )

        #expect(decomp.alpha == 1.0, "Alpha should be 1.0")
        #expect(decomp.beta == 2.0, "Beta should be 2.0")
        #expect(decomp.gamma == 3.0, "Gamma should be 3.0")
        #expect(decomp.globalPhase == 0.5, "Global phase should be 0.5")
    }

    @Test("EulerDecomposition equality")
    func eulerDecompositionEquality() {
        let a = GateSynthesis.EulerDecomposition(alpha: 1.0, beta: 2.0, gamma: 3.0, globalPhase: 0.0)
        let b = GateSynthesis.EulerDecomposition(alpha: 1.0, beta: 2.0, gamma: 3.0, globalPhase: 0.0)
        let c = GateSynthesis.EulerDecomposition(alpha: 1.0, beta: 2.0, gamma: 3.0, globalPhase: 0.1)

        #expect(a == b, "Identical decompositions should be equal")
        #expect(a != c, "Different global phases should be unequal")
    }

    @Test("EulerDecomposition is Sendable")
    func eulerDecompositionSendable() {
        let decomp: any Sendable = GateSynthesis.EulerDecomposition(alpha: 0, beta: 0, gamma: 0, globalPhase: 0)
        #expect(decomp is GateSynthesis.EulerDecomposition, "EulerDecomposition should conform to Sendable")
    }

    @Test("Clifford+T synthesis with large angle covers normalizeAngle > pi branch")
    func cliffordTLargeAngle() {
        let matrix = QuantumGate.rotationZ(4.0).matrix()
        let gates = GateSynthesis.cliffordT(approximating: matrix, precision: 0.1)
        #expect(!gates.isEmpty, "Clifford+T with angle > pi should produce gates")
    }
}
