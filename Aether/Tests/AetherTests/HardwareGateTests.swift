// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Foundation
import Testing

/// Test suite for iSWAP gate behavior and matrix correctness.
/// Validates iSWAP action on basis states, phase factors, and that
/// two iSWAP gates do not compose to identity.
@Suite("iSWAP Gate")
struct ISwapGateTests {
    @Test("iSWAP matrix matches expected 4x4 structure")
    func iSwapMatrixCorrect() {
        let gate = QuantumGate.iswap
        let matrix = gate.matrix()

        #expect(matrix[0][0] == Complex<Double>.one, "iSWAP[0][0] should be 1")
        #expect(matrix[0][1] == Complex<Double>.zero, "iSWAP[0][1] should be 0")
        #expect(matrix[0][2] == Complex<Double>.zero, "iSWAP[0][2] should be 0")
        #expect(matrix[0][3] == Complex<Double>.zero, "iSWAP[0][3] should be 0")

        #expect(matrix[1][0] == Complex<Double>.zero, "iSWAP[1][0] should be 0")
        #expect(matrix[1][1] == Complex<Double>.zero, "iSWAP[1][1] should be 0")
        #expect(matrix[1][2] == Complex<Double>.i, "iSWAP[1][2] should be i")
        #expect(matrix[1][3] == Complex<Double>.zero, "iSWAP[1][3] should be 0")

        #expect(matrix[2][0] == Complex<Double>.zero, "iSWAP[2][0] should be 0")
        #expect(matrix[2][1] == Complex<Double>.i, "iSWAP[2][1] should be i")
        #expect(matrix[2][2] == Complex<Double>.zero, "iSWAP[2][2] should be 0")
        #expect(matrix[2][3] == Complex<Double>.zero, "iSWAP[2][3] should be 0")

        #expect(matrix[3][0] == Complex<Double>.zero, "iSWAP[3][0] should be 0")
        #expect(matrix[3][1] == Complex<Double>.zero, "iSWAP[3][1] should be 0")
        #expect(matrix[3][2] == Complex<Double>.zero, "iSWAP[3][2] should be 0")
        #expect(matrix[3][3] == Complex<Double>.one, "iSWAP[3][3] should be 1")
    }

    @Test("iSWAP transforms |01⟩ to i|10⟩")
    func iSwapTransforms01To10WithPhase() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(1, to: .one)

        let result = GateApplication.apply(.iswap, to: [0, 1], state: state)

        #expect(abs(result.amplitude(of: 0).magnitudeSquared) < 1e-10, "Amplitude of |00⟩ should be zero after iSWAP on |01⟩")
        #expect(abs(result.amplitude(of: 1).magnitudeSquared) < 1e-10, "Amplitude of |01⟩ should be zero after iSWAP on |01⟩")
        #expect(abs(result.probability(of: 2) - 1.0) < 1e-10, "iSWAP should map |01⟩ to |10⟩ with unit probability")

        let amp10 = result.amplitude(of: 2)
        #expect(abs(amp10.real) < 1e-10, "Real part of |10⟩ amplitude should be zero (phase is i)")
        #expect(abs(amp10.imaginary - 1.0) < 1e-10, "Imaginary part of |10⟩ amplitude should be 1 (phase is i)")
    }

    @Test("iSWAP transforms |10⟩ to i|01⟩")
    func iSwapTransforms10To01WithPhase() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(2, to: .one)

        let result = GateApplication.apply(.iswap, to: [0, 1], state: state)

        #expect(abs(result.probability(of: 1) - 1.0) < 1e-10, "iSWAP should map |10⟩ to |01⟩ with unit probability")

        let amp01 = result.amplitude(of: 1)
        #expect(abs(amp01.real) < 1e-10, "Real part of |01⟩ amplitude should be zero (phase is i)")
        #expect(abs(amp01.imaginary - 1.0) < 1e-10, "Imaginary part of |01⟩ amplitude should be 1 (phase is i)")
    }

    @Test("iSWAP leaves |00⟩ unchanged")
    func iSwapLeaves00Unchanged() {
        let state = QuantumState(qubits: 2)

        let result = GateApplication.apply(.iswap, to: [0, 1], state: state)

        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "iSWAP should leave |00⟩ unchanged")
        #expect(result.amplitude(of: 0) == Complex<Double>.one, "Amplitude of |00⟩ should remain 1+0i")
    }

    @Test("iSWAP leaves |11⟩ unchanged")
    func iSwapLeaves11Unchanged() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(3, to: .one)

        let result = GateApplication.apply(.iswap, to: [0, 1], state: state)

        #expect(abs(result.probability(of: 3) - 1.0) < 1e-10, "iSWAP should leave |11⟩ unchanged")
        #expect(result.amplitude(of: 3) == Complex<Double>.one, "Amplitude of |11⟩ should remain 1+0i")
    }

    @Test("Two iSWAP gates do not equal identity")
    func twoISwapsNotIdentity() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(1, to: .one)

        let afterFirst = GateApplication.apply(.iswap, to: [0, 1], state: state)
        let afterSecond = GateApplication.apply(.iswap, to: [0, 1], state: afterFirst)

        let finalAmp = afterSecond.amplitude(of: 1)
        #expect(abs(finalAmp.real - -1.0) < 1e-10, "Two iSWAPs on |01⟩ should give -|01⟩ (real part -1)")
        #expect(abs(finalAmp.imaginary) < 1e-10, "Two iSWAPs on |01⟩ should give -|01⟩ (imaginary part 0)")

        let matrixISwap = QuantumGate.iswap.matrix()
        let matrixSquared = QuantumGate.matrixMultiply(matrixISwap, matrixISwap)
        #expect(!QuantumGate.isIdentityMatrix(matrixSquared), "iSWAP squared should not be identity matrix")
    }

    @Test("iSWAP matrix is unitary")
    func iSwapIsUnitary() {
        let matrix = QuantumGate.iswap.matrix()
        #expect(QuantumGate.isUnitary(matrix), "iSWAP matrix should be unitary")
    }
}

/// Test suite for sqrt(iSWAP) gate behavior and composition.
/// Validates that two sqrt(iSWAP) applications equal one iSWAP,
/// and that matrix elements are correct.
@Suite("sqrt(iSWAP) Gate")
struct SqrtISwapGateTests {
    @Test("Two sqrt(iSWAP) gates equal one iSWAP")
    func twoSqrtISwapEqualsISwap() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(1, to: .one)

        let afterSqrt1 = GateApplication.apply(.sqrtISwap, to: [0, 1], state: state)
        let afterSqrt2 = GateApplication.apply(.sqrtISwap, to: [0, 1], state: afterSqrt1)

        let afterISwap = GateApplication.apply(.iswap, to: [0, 1], state: state)

        for i in 0 ..< 4 {
            let sqrtResult = afterSqrt2.amplitude(of: i)
            let iswapResult = afterISwap.amplitude(of: i)
            #expect(abs(sqrtResult.real - iswapResult.real) < 1e-10, "Two sqrt(iSWAP) should match iSWAP for amplitude \(i) real part")
            #expect(abs(sqrtResult.imaginary - iswapResult.imaginary) < 1e-10, "Two sqrt(iSWAP) should match iSWAP for amplitude \(i) imaginary part")
        }
    }

    @Test("sqrt(iSWAP) matrix squared equals iSWAP matrix")
    func sqrtISwapMatrixSquaredEqualsISwap() {
        let sqrtISwapMatrix = QuantumGate.sqrtISwap.matrix()
        let sqrtSquared = QuantumGate.matrixMultiply(sqrtISwapMatrix, sqrtISwapMatrix)
        let iswapMatrix = QuantumGate.iswap.matrix()

        #expect(QuantumGate.matricesEqual(sqrtSquared, iswapMatrix), "sqrt(iSWAP) squared should equal iSWAP matrix")
    }

    @Test("sqrt(iSWAP) matrix elements are correct")
    func sqrtISwapMatrixElementsCorrect() {
        let matrix = QuantumGate.sqrtISwap.matrix()
        let invSqrt2 = 1.0 / sqrt(2.0)

        #expect(matrix[0][0] == Complex<Double>.one, "sqrt(iSWAP)[0][0] should be 1")
        #expect(matrix[3][3] == Complex<Double>.one, "sqrt(iSWAP)[3][3] should be 1")

        #expect(abs(matrix[1][1].real - invSqrt2) < 1e-10, "sqrt(iSWAP)[1][1] real should be 1/sqrt(2)")
        #expect(abs(matrix[1][1].imaginary) < 1e-10, "sqrt(iSWAP)[1][1] imaginary should be 0")

        #expect(abs(matrix[1][2].real) < 1e-10, "sqrt(iSWAP)[1][2] real should be 0")
        #expect(abs(matrix[1][2].imaginary - invSqrt2) < 1e-10, "sqrt(iSWAP)[1][2] imaginary should be 1/sqrt(2)")

        #expect(abs(matrix[2][1].real) < 1e-10, "sqrt(iSWAP)[2][1] real should be 0")
        #expect(abs(matrix[2][1].imaginary - invSqrt2) < 1e-10, "sqrt(iSWAP)[2][1] imaginary should be 1/sqrt(2)")

        #expect(abs(matrix[2][2].real - invSqrt2) < 1e-10, "sqrt(iSWAP)[2][2] real should be 1/sqrt(2)")
        #expect(abs(matrix[2][2].imaginary) < 1e-10, "sqrt(iSWAP)[2][2] imaginary should be 0")
    }

    @Test("sqrt(iSWAP) matrix is unitary")
    func sqrtISwapIsUnitary() {
        let matrix = QuantumGate.sqrtISwap.matrix()
        #expect(QuantumGate.isUnitary(matrix), "sqrt(iSWAP) matrix should be unitary")
    }
}

/// Test suite for FSWAP (fermionic swap) gate behavior.
/// Validates swap action on |01⟩ and |10⟩, negation of |11⟩,
/// invariance of |00⟩, and self-inverse property.
@Suite("FSWAP Gate")
struct FSwapGateTests {
    @Test("FSWAP swaps |01⟩ to |10⟩")
    func fswapSwaps01To10() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(1, to: .one)

        let result = GateApplication.apply(.fswap, to: [0, 1], state: state)

        #expect(abs(result.probability(of: 2) - 1.0) < 1e-10, "FSWAP should map |01⟩ to |10⟩")
        #expect(result.amplitude(of: 2) == Complex<Double>.one, "FSWAP on |01⟩ should give amplitude 1 for |10⟩")
    }

    @Test("FSWAP swaps |10⟩ to |01⟩")
    func fswapSwaps10To01() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(2, to: .one)

        let result = GateApplication.apply(.fswap, to: [0, 1], state: state)

        #expect(abs(result.probability(of: 1) - 1.0) < 1e-10, "FSWAP should map |10⟩ to |01⟩")
        #expect(result.amplitude(of: 1) == Complex<Double>.one, "FSWAP on |10⟩ should give amplitude 1 for |01⟩")
    }

    @Test("FSWAP negates |11⟩")
    func fswapNegates11() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(3, to: .one)

        let result = GateApplication.apply(.fswap, to: [0, 1], state: state)

        #expect(abs(result.probability(of: 3) - 1.0) < 1e-10, "FSWAP should keep |11⟩ in |11⟩ state")

        let amp11 = result.amplitude(of: 3)
        #expect(abs(amp11.real - -1.0) < 1e-10, "FSWAP should negate |11⟩ (real part should be -1)")
        #expect(abs(amp11.imaginary) < 1e-10, "FSWAP should negate |11⟩ (imaginary part should be 0)")
    }

    @Test("FSWAP leaves |00⟩ unchanged")
    func fswapLeaves00Unchanged() {
        let state = QuantumState(qubits: 2)

        let result = GateApplication.apply(.fswap, to: [0, 1], state: state)

        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "FSWAP should leave |00⟩ unchanged")
        #expect(result.amplitude(of: 0) == Complex<Double>.one, "FSWAP on |00⟩ should give amplitude 1 for |00⟩")
    }

    @Test("FSWAP squared equals identity (self-inverse)")
    func fswapSquaredIsIdentity() {
        let fswapMatrix = QuantumGate.fswap.matrix()
        let fswapSquared = QuantumGate.matrixMultiply(fswapMatrix, fswapMatrix)

        #expect(QuantumGate.isIdentityMatrix(fswapSquared), "FSWAP squared should be identity (self-inverse)")
    }

    @Test("FSWAP is Hermitian (self-adjoint)")
    func fswapIsHermitian() {
        #expect(QuantumGate.fswap.isHermitian, "FSWAP should be marked as Hermitian")
    }

    @Test("FSWAP matrix is unitary")
    func fswapIsUnitary() {
        let matrix = QuantumGate.fswap.matrix()
        #expect(QuantumGate.isUnitary(matrix), "FSWAP matrix should be unitary")
    }
}

/// Test suite for Fredkin (controlled-SWAP) gate behavior.
/// Validates control semantics, target swap when control is |1⟩,
/// and self-inverse property.
@Suite("Fredkin Gate")
struct FredkinGateTests {
    @Test("Fredkin leaves targets unchanged when control is 0")
    func fredkinControlZeroLeavesTargetsUnchanged() {
        var state = QuantumState(qubits: 3)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(2, to: .one)

        let result = GateApplication.apply(.fredkin, to: [0, 1, 2], state: state)

        #expect(abs(result.probability(of: 2) - 1.0) < 1e-10, "Fredkin with control=0 should not swap targets")
    }

    @Test("Fredkin swaps targets when control is 1 and targets differ")
    func fredkinSwapsTargetsWhenControlIsOne() {
        var state = QuantumState(qubits: 3)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(3, to: .one)

        let result = GateApplication.apply(.fredkin, to: [0, 1, 2], state: state)

        #expect(abs(result.probability(of: 5) - 1.0) < 1e-10, "Fredkin with control=1 should swap |011⟩ to |101⟩")
    }

    @Test("Fredkin swaps |101⟩ to |011⟩ when control is 1")
    func fredkinSwaps101To011() {
        var state = QuantumState(qubits: 3)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(5, to: .one)

        let result = GateApplication.apply(.fredkin, to: [0, 1, 2], state: state)

        #expect(abs(result.probability(of: 3) - 1.0) < 1e-10, "Fredkin with control=1 should swap |101⟩ to |011⟩")
    }

    @Test("Fredkin leaves |001⟩ unchanged (control=1, targets both 0)")
    func fredkinLeaves001Unchanged() {
        var state = QuantumState(qubits: 3)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(1, to: .one)

        let result = GateApplication.apply(.fredkin, to: [0, 1, 2], state: state)

        #expect(abs(result.probability(of: 1) - 1.0) < 1e-10, "Fredkin should leave |001⟩ unchanged (both targets are 0)")
    }

    @Test("Fredkin leaves |111⟩ unchanged (control=1, targets both 1)")
    func fredkinLeaves111Unchanged() {
        var state = QuantumState(qubits: 3)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(7, to: .one)

        let result = GateApplication.apply(.fredkin, to: [0, 1, 2], state: state)

        #expect(abs(result.probability(of: 7) - 1.0) < 1e-10, "Fredkin should leave |111⟩ unchanged (both targets are 1)")
    }

    @Test("Fredkin is self-inverse")
    func fredkinIsSelfInverse() {
        let fredkinMatrix = QuantumGate.fredkin.matrix()
        let fredkinSquared = QuantumGate.matrixMultiply(fredkinMatrix, fredkinMatrix)

        #expect(QuantumGate.isIdentityMatrix(fredkinSquared), "Fredkin squared should be identity (self-inverse)")
    }

    @Test("Fredkin is Hermitian")
    func fredkinIsHermitian() {
        #expect(QuantumGate.fredkin.isHermitian, "Fredkin should be marked as Hermitian")
    }

    @Test("Fredkin matrix is unitary")
    func fredkinIsUnitary() {
        let matrix = QuantumGate.fredkin.matrix()
        #expect(QuantumGate.isUnitary(matrix), "Fredkin matrix should be unitary")
    }
}

/// Test suite for Givens rotation gate behavior.
/// Validates identity at theta=0, full swap at theta=pi/2,
/// and inverse relationship between G(theta) and G(-theta).
@Suite("Givens Rotation Gate")
struct GivensGateTests {
    @Test("Givens(0) is identity on |01⟩-|10⟩ subspace")
    func givensZeroIsIdentity() {
        var state01 = QuantumState(qubits: 2)
        state01.setAmplitude(0, to: .zero)
        state01.setAmplitude(1, to: .one)

        let result01 = GateApplication.apply(.givens(0.0), to: [0, 1], state: state01)

        #expect(abs(result01.probability(of: 1) - 1.0) < 1e-10, "Givens(0) should leave |01⟩ unchanged")

        var state10 = QuantumState(qubits: 2)
        state10.setAmplitude(0, to: .zero)
        state10.setAmplitude(2, to: .one)

        let result10 = GateApplication.apply(.givens(0.0), to: [0, 1], state: state10)

        #expect(abs(result10.probability(of: 2) - 1.0) < 1e-10, "Givens(0) should leave |10⟩ unchanged")
    }

    @Test("Givens(pi/2) swaps |01⟩ and |10⟩")
    func givensPiOver2Swaps01And10() {
        var state01 = QuantumState(qubits: 2)
        state01.setAmplitude(0, to: .zero)
        state01.setAmplitude(1, to: .one)

        let result01 = GateApplication.apply(.givens(.pi / 2), to: [0, 1], state: state01)

        #expect(abs(result01.probability(of: 2) - 1.0) < 1e-10, "Givens(pi/2) should map |01⟩ to -|10⟩ with unit probability")

        let amp10FromState01 = result01.amplitude(of: 2)
        #expect(abs(amp10FromState01.real - -1.0) < 1e-10, "Givens(pi/2) on |01⟩ gives -|10⟩ (real part -1)")

        var state10 = QuantumState(qubits: 2)
        state10.setAmplitude(0, to: .zero)
        state10.setAmplitude(2, to: .one)

        let result10 = GateApplication.apply(.givens(.pi / 2), to: [0, 1], state: state10)

        #expect(abs(result10.probability(of: 1) - 1.0) < 1e-10, "Givens(pi/2) should map |10⟩ to |01⟩ with unit probability")

        let amp01 = result10.amplitude(of: 1)
        #expect(abs(amp01.real - 1.0) < 1e-10, "Givens(pi/2) on |10⟩ gives |01⟩ (real part 1)")
    }

    @Test("Givens(-theta) is inverse of Givens(theta)")
    func givensNegativeThetaIsInverse() {
        let theta = 0.7

        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: Complex(0.6, 0.0))
        state.setAmplitude(1, to: Complex(0.0, 0.8))

        let afterForward = GateApplication.apply(.givens(theta), to: [0, 1], state: state)
        let afterInverse = GateApplication.apply(.givens(-theta), to: [0, 1], state: afterForward)

        for i in 0 ..< 4 {
            let original = state.amplitude(of: i)
            let recovered = afterInverse.amplitude(of: i)
            #expect(abs(original.real - recovered.real) < 1e-10, "Givens(-theta) should invert Givens(theta) for amplitude \(i) real part")
            #expect(abs(original.imaginary - recovered.imaginary) < 1e-10, "Givens(-theta) should invert Givens(theta) for amplitude \(i) imaginary part")
        }
    }

    @Test("Givens gate leaves |00⟩ and |11⟩ unchanged")
    func givensLeaves00And11Unchanged() {
        let theta = 1.23

        let state00 = QuantumState(qubits: 2)
        let result00 = GateApplication.apply(.givens(theta), to: [0, 1], state: state00)
        #expect(abs(result00.probability(of: 0) - 1.0) < 1e-10, "Givens should leave |00⟩ unchanged")

        var state11 = QuantumState(qubits: 2)
        state11.setAmplitude(0, to: .zero)
        state11.setAmplitude(3, to: .one)
        let result11 = GateApplication.apply(.givens(theta), to: [0, 1], state: state11)
        #expect(abs(result11.probability(of: 3) - 1.0) < 1e-10, "Givens should leave |11⟩ unchanged")
    }

    @Test("Givens matrix is unitary")
    func givensIsUnitary() {
        let matrix = QuantumGate.givens(1.234).matrix()
        #expect(QuantumGate.isUnitary(matrix), "Givens matrix should be unitary")
    }
}

/// Test suite for XX (Molmer-Sorensen) interaction gate.
/// Validates identity at theta=0, entanglement creation at theta=pi/4,
/// and inverse relationship between XX(theta) and XX(-theta).
@Suite("XX (Molmer-Sorensen) Gate")
struct XXGateTests {
    @Test("XX(0) is identity")
    func xxZeroIsIdentity() {
        let xxZeroMatrix = QuantumGate.xx(0.0).matrix()

        #expect(QuantumGate.isIdentityMatrix(xxZeroMatrix), "XX(0) should be identity matrix")
    }

    @Test("XX(0) leaves all basis states unchanged")
    func xxZeroLeavesStatesUnchanged() {
        let state00 = QuantumState(qubits: 2)
        let result00 = GateApplication.apply(.xx(0.0), to: [0, 1], state: state00)
        #expect(abs(result00.probability(of: 0) - 1.0) < 1e-10, "XX(0) should leave |00⟩ unchanged")

        var state01 = QuantumState(qubits: 2)
        state01.setAmplitude(0, to: .zero)
        state01.setAmplitude(1, to: .one)
        let result01 = GateApplication.apply(.xx(0.0), to: [0, 1], state: state01)
        #expect(abs(result01.probability(of: 1) - 1.0) < 1e-10, "XX(0) should leave |01⟩ unchanged")

        var state10 = QuantumState(qubits: 2)
        state10.setAmplitude(0, to: .zero)
        state10.setAmplitude(2, to: .one)
        let result10 = GateApplication.apply(.xx(0.0), to: [0, 1], state: state10)
        #expect(abs(result10.probability(of: 2) - 1.0) < 1e-10, "XX(0) should leave |10⟩ unchanged")

        var state11 = QuantumState(qubits: 2)
        state11.setAmplitude(0, to: .zero)
        state11.setAmplitude(3, to: .one)
        let result11 = GateApplication.apply(.xx(0.0), to: [0, 1], state: state11)
        #expect(abs(result11.probability(of: 3) - 1.0) < 1e-10, "XX(0) should leave |11⟩ unchanged")
    }

    @Test("XX(pi/4) creates entanglement from |00⟩")
    func xxPiOver4CreatesEntanglement() {
        let state = QuantumState(qubits: 2)

        let result = GateApplication.apply(.xx(.pi / 4), to: [0, 1], state: state)

        let prob00 = result.probability(of: 0)
        let prob11 = result.probability(of: 3)

        #expect(prob00 > 0.4, "XX(pi/4) on |00⟩ should give significant probability for |00⟩")
        #expect(prob11 > 0.4, "XX(pi/4) on |00⟩ should give significant probability for |11⟩")
        #expect(abs(prob00 + prob11 - 1.0) < 1e-10, "XX(pi/4) on |00⟩ should only produce |00⟩ and |11⟩ components")
    }

    @Test("XX(-theta) is inverse of XX(theta)")
    func xxNegativeThetaIsInverse() {
        let theta = 0.567

        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: Complex(0.5, 0.0))
        state.setAmplitude(1, to: Complex(0.5, 0.0))
        state.setAmplitude(2, to: Complex(0.5, 0.0))
        state.setAmplitude(3, to: Complex(0.5, 0.0))

        let afterForward = GateApplication.apply(.xx(theta), to: [0, 1], state: state)
        let afterInverse = GateApplication.apply(.xx(-theta), to: [0, 1], state: afterForward)

        for i in 0 ..< 4 {
            let original = state.amplitude(of: i)
            let recovered = afterInverse.amplitude(of: i)
            #expect(abs(original.real - recovered.real) < 1e-10, "XX(-theta) should invert XX(theta) for amplitude \(i) real part")
            #expect(abs(original.imaginary - recovered.imaginary) < 1e-10, "XX(-theta) should invert XX(theta) for amplitude \(i) imaginary part")
        }
    }

    @Test("XX matrix composed with its inverse equals identity")
    func xxMatrixTimesInverseIsIdentity() {
        let theta = 1.234
        let xxMatrix = QuantumGate.xx(theta).matrix()
        let xxInverseMatrix = QuantumGate.xx(-theta).matrix()
        let product = QuantumGate.matrixMultiply(xxMatrix, xxInverseMatrix)

        #expect(QuantumGate.isIdentityMatrix(product), "XX(theta) * XX(-theta) should equal identity")
    }

    @Test("XX matrix is unitary")
    func xxIsUnitary() {
        let matrix = QuantumGate.xx(0.789).matrix()
        #expect(QuantumGate.isUnitary(matrix), "XX matrix should be unitary")
    }
}

/// Test suite for diagonal gate behavior and phase application.
/// Validates single-qubit Z equivalence, multi-qubit phase application,
/// and identity behavior when all phases are zero.
@Suite("Diagonal Gate")
struct DiagonalGateTests {
    @Test("Single-qubit diagonal [0, pi] equals Z gate")
    func singleQubitDiagonalEqualsZ() {
        let diagMatrix = QuantumGate.diagonal(phases: [0, .pi]).matrix()
        let zMatrix = QuantumGate.pauliZ.matrix()

        #expect(QuantumGate.matricesEqual(diagMatrix, zMatrix), "Diagonal([0, pi]) should equal Z gate matrix")
    }

    @Test("Diagonal [0, pi] action matches Z gate action")
    func diagonalZeroAndPiMatchesZAction() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        var state = QuantumState(qubits: 1)
        state.setAmplitude(0, to: Complex(invSqrt2, 0.0))
        state.setAmplitude(1, to: Complex(invSqrt2, 0.0))

        let diagResult = GateApplication.apply(.diagonal(phases: [0, .pi]), to: [0], state: state)
        let zResult = GateApplication.apply(.pauliZ, to: 0, state: state)

        for i in 0 ..< 2 {
            let diagAmp = diagResult.amplitude(of: i)
            let zAmp = zResult.amplitude(of: i)
            #expect(abs(diagAmp.real - zAmp.real) < 1e-10, "Diagonal([0,pi]) should match Z for amplitude \(i) real part")
            #expect(abs(diagAmp.imaginary - zAmp.imaginary) < 1e-10, "Diagonal([0,pi]) should match Z for amplitude \(i) imaginary part")
        }
    }

    @Test("Multi-qubit diagonal applies correct phases")
    func multiQubitDiagonalAppliesCorrectPhases() {
        let phases = [0.0, .pi / 4, .pi / 2, .pi]

        let invSqrt = 0.5
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: Complex(invSqrt, 0.0))
        state.setAmplitude(1, to: Complex(invSqrt, 0.0))
        state.setAmplitude(2, to: Complex(invSqrt, 0.0))
        state.setAmplitude(3, to: Complex(invSqrt, 0.0))

        let result = GateApplication.apply(.diagonal(phases: phases), to: [0, 1], state: state)

        for i in 0 ..< 4 {
            let expectedPhase = Complex<Double>(phase: phases[i])
            let expectedAmp = Complex(invSqrt, 0.0) * expectedPhase
            let actualAmp = result.amplitude(of: i)

            #expect(abs(actualAmp.real - expectedAmp.real) < 1e-10, "Diagonal phase \(i) real part should match expected")
            #expect(abs(actualAmp.imaginary - expectedAmp.imaginary) < 1e-10, "Diagonal phase \(i) imaginary part should match expected")
        }
    }

    @Test("Diagonal with all phases zero is identity")
    func diagonalAllZerosIsIdentity() {
        let diag = QuantumGate.diagonal(phases: [0, 0, 0, 0])
        let matrix = diag.matrix()

        #expect(QuantumGate.isIdentityMatrix(matrix), "Diagonal with all zero phases should be identity")
    }

    @Test("Diagonal gate preserves normalization")
    func diagonalPreservesNormalization() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: Complex(0.5, 0.0))
        state.setAmplitude(1, to: Complex(0.0, 0.5))
        state.setAmplitude(2, to: Complex(0.5, 0.0))
        state.setAmplitude(3, to: Complex(0.0, 0.5))

        let result = GateApplication.apply(.diagonal(phases: [0.1, 0.2, 0.3, 0.4]), to: [0, 1], state: state)

        #expect(result.isNormalized(), "Diagonal gate should preserve normalization")
    }

    @Test("Diagonal matrix is unitary")
    func diagonalIsUnitary() {
        let matrix = QuantumGate.diagonal(phases: [0.1, 0.2, 0.3, 0.4]).matrix()
        #expect(QuantumGate.isUnitary(matrix), "Diagonal matrix should be unitary")
    }
}

/// Test suite for multiplexor gate behavior and selection logic.
/// Validates single-unitary pass-through, multi-unitary control-based
/// selection, and matrix unitarity.
@Suite("Multiplexor Gate")
struct MultiplexorGateTests {
    @Test("Single unitary multiplexor applies that unitary")
    func singleUnitaryMultiplexor() {
        let xMatrix: [[Complex<Double>]] = [
            [.zero, .one],
            [.one, .zero],
        ]

        let state = QuantumState(qubits: 1)

        let multiplexorGate = QuantumGate.multiplexor(unitaries: [xMatrix])
        let result = GateApplication.apply(multiplexorGate, to: [0], state: state)

        #expect(abs(result.probability(of: 1) - 1.0) < 1e-10, "Single-unitary multiplexor should apply X gate (flip |0⟩ to |1⟩)")
    }

    @Test("Two-unitary multiplexor selects based on control qubit")
    func twoUnitaryMultiplexorSelectsBasedOnControl() {
        let identity: [[Complex<Double>]] = [
            [.one, .zero],
            [.zero, .one],
        ]
        let pauliX: [[Complex<Double>]] = [
            [.zero, .one],
            [.one, .zero],
        ]

        let multiplexorGate = QuantumGate.multiplexor(unitaries: [identity, pauliX])

        var stateControl0 = QuantumState(qubits: 2)
        stateControl0.setAmplitude(0, to: .one)
        let result0 = GateApplication.apply(multiplexorGate, to: [0, 1], state: stateControl0)
        #expect(abs(result0.probability(of: 0) - 1.0) < 1e-10, "Multiplexor with control=0 should apply identity (leave |00⟩ as |00⟩)")

        var stateControl1 = QuantumState(qubits: 2)
        stateControl1.setAmplitude(0, to: .zero)
        stateControl1.setAmplitude(2, to: .one)
        let result1 = GateApplication.apply(multiplexorGate, to: [0, 1], state: stateControl1)
        #expect(abs(result1.probability(of: 3) - 1.0) < 1e-10, "Multiplexor with control=1 should apply X (|10⟩ becomes |11⟩)")
    }

    @Test("Multiplexor with Hadamard creates superposition based on control")
    func multiplexorWithHadamardCreatesSuperposition() {
        let identity: [[Complex<Double>]] = [
            [.one, .zero],
            [.zero, .one],
        ]
        let invSqrt2 = 1.0 / sqrt(2.0)
        let hadamard: [[Complex<Double>]] = [
            [Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0)],
            [Complex(invSqrt2, 0.0), Complex(-invSqrt2, 0.0)],
        ]

        let multiplexorGate = QuantumGate.multiplexor(unitaries: [identity, hadamard])

        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(2, to: .one)

        let result = GateApplication.apply(multiplexorGate, to: [0, 1], state: state)

        #expect(abs(result.probability(of: 2) - 0.5) < 1e-10, "Multiplexor should apply H when control=1, giving |10⟩ with prob 0.5")
        #expect(abs(result.probability(of: 3) - 0.5) < 1e-10, "Multiplexor should apply H when control=1, giving |11⟩ with prob 0.5")
    }

    @Test("Multiplexor matrix is unitary")
    func multiplexorIsUnitary() {
        let identity: [[Complex<Double>]] = [
            [.one, .zero],
            [.zero, .one],
        ]
        let pauliZ: [[Complex<Double>]] = [
            [.one, .zero],
            [.zero, Complex(-1.0, 0.0)],
        ]

        let multiplexorGate = QuantumGate.multiplexor(unitaries: [identity, pauliZ])
        let matrix = multiplexorGate.matrix()

        #expect(QuantumGate.isUnitary(matrix), "Multiplexor matrix should be unitary")
    }

    @Test("Multiplexor with four unitaries selects correctly")
    func multiplexorFourUnitariesSelectsCorrectly() {
        let identity: [[Complex<Double>]] = [[.one, .zero], [.zero, .one]]
        let pauliX: [[Complex<Double>]] = [[.zero, .one], [.one, .zero]]
        let pauliY: [[Complex<Double>]] = [[.zero, -Complex.i], [Complex.i, .zero]]
        let pauliZ: [[Complex<Double>]] = [[.one, .zero], [.zero, Complex(-1.0, 0.0)]]

        let multiplexorGate = QuantumGate.multiplexor(unitaries: [identity, pauliX, pauliY, pauliZ])

        var stateForX = QuantumState(qubits: 3)
        stateForX.setAmplitude(0, to: .zero)
        stateForX.setAmplitude(2, to: .one)

        let resultX = GateApplication.apply(multiplexorGate, to: [0, 1, 2], state: stateForX)

        #expect(abs(resultX.probability(of: 3) - 1.0) < 1e-10, "Multiplexor with control=01 should apply X, flipping |010⟩ to |011⟩")

        var stateForZ = QuantumState(qubits: 3)
        stateForZ.setAmplitude(0, to: .zero)
        stateForZ.setAmplitude(7, to: .one)

        let resultZ = GateApplication.apply(multiplexorGate, to: [0, 1, 2], state: stateForZ)

        #expect(abs(resultZ.probability(of: 7) - 1.0) < 1e-10, "Multiplexor with control=11 should apply Z to |1⟩ target")

        let ampAfterZ = resultZ.amplitude(of: 7)
        #expect(abs(ampAfterZ.real - -1.0) < 1e-10, "Z gate on |1⟩ gives phase -1 (real part -1)")
    }
}

/// Test suite for hardware gate qubit requirements and descriptions.
/// Validates qubitsRequired property and human-readable descriptions
/// for all new hardware compatibility gates.
@Suite("Hardware Gate Properties")
struct HardwareGatePropertiesTests {
    @Test("All hardware gates have correct qubit requirements")
    func hardwareGatesQubitRequirements() {
        #expect(QuantumGate.iswap.qubitsRequired == 2, "iSWAP requires 2 qubits")
        #expect(QuantumGate.sqrtISwap.qubitsRequired == 2, "sqrt(iSWAP) requires 2 qubits")
        #expect(QuantumGate.fswap.qubitsRequired == 2, "FSWAP requires 2 qubits")
        #expect(QuantumGate.givens(0.5).qubitsRequired == 2, "Givens requires 2 qubits")
        #expect(QuantumGate.xx(0.5).qubitsRequired == 2, "XX requires 2 qubits")
        #expect(QuantumGate.fredkin.qubitsRequired == 3, "Fredkin requires 3 qubits")
        #expect(QuantumGate.diagonal(phases: [0, 0]).qubitsRequired == 1, "Diagonal(2 phases) requires 1 qubit")
        #expect(QuantumGate.diagonal(phases: [0, 0, 0, 0]).qubitsRequired == 2, "Diagonal(4 phases) requires 2 qubits")
    }

    @Test("Hardware gate descriptions are correct")
    func hardwareGateDescriptions() {
        #expect(QuantumGate.iswap.description == "iSWAP", "iSWAP description should be 'iSWAP'")
        #expect(QuantumGate.sqrtISwap.description == "√iSWAP", "sqrt(iSWAP) description should be '√iSWAP'")
        #expect(QuantumGate.fswap.description == "FSWAP", "FSWAP description should be 'FSWAP'")
        #expect(QuantumGate.fredkin.description == "Fredkin", "Fredkin description should be 'Fredkin'")
        #expect(QuantumGate.givens(1.5).description.contains("Givens"), "Givens description should contain 'Givens'")
        #expect(QuantumGate.xx(1.5).description.contains("XX"), "XX description should contain 'XX'")
        #expect(QuantumGate.diagonal(phases: [0, 0]).description.contains("Diagonal"), "Diagonal description should contain 'Diagonal'")
        #expect(QuantumGate.multiplexor(unitaries: [[[.one, .zero], [.zero, .one]]]).description.contains("Multiplexor"), "Multiplexor description should contain 'Multiplexor'")
    }

    @Test("All new hardware gate matrices are unitary")
    func allHardwareGatesUnitary() {
        let gates: [QuantumGate] = [
            .iswap,
            .sqrtISwap,
            .fswap,
            .givens(1.23),
            .xx(1.23),
            .fredkin,
            .diagonal(phases: [0.1, 0.2, 0.3, 0.4]),
            .multiplexor(unitaries: [
                [[.one, .zero], [.zero, .one]],
                [[.zero, .one], [.one, .zero]],
            ]),
        ]

        for gate in gates {
            let matrix = gate.matrix()
            #expect(QuantumGate.isUnitary(matrix), "Gate \(gate.description) should have unitary matrix")
        }
    }
}

/// Validates qubitsRequired computation for diagonal and multiplexor gates.
/// Tests that phase counts and unitary dimensions correctly determine qubit requirements.
/// Ensures quantum resource estimation is accurate for hardware synthesis.
@Suite("Diagonal and Multiplexor Qubit Requirements")
struct DiagonalMultiplexorQubitsRequiredTests {
    @Test("Diagonal gate with 4 phases requires 2 qubits")
    func diagonalFourPhasesRequiresTwoQubits() {
        let diagonal = QuantumGate.diagonal(phases: [0.0, 0.1, 0.2, 0.3])
        #expect(diagonal.qubitsRequired == 2, "Diagonal with 4 phases should require 2 qubits")
    }

    @Test("Diagonal gate with 8 phases requires 3 qubits")
    func diagonalEightPhasesRequiresThreeQubits() {
        let diagonal = QuantumGate.diagonal(phases: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        #expect(diagonal.qubitsRequired == 3, "Diagonal with 8 phases should require 3 qubits")
    }

    @Test("Multiplexor with 2 unitaries of 2x2 requires 2 qubits")
    func multiplexorTwoUnitariesRequiresTwoQubits() {
        let identity: [[Complex<Double>]] = [[.one, .zero], [.zero, .one]]
        let pauliX: [[Complex<Double>]] = [[.zero, .one], [.one, .zero]]
        let multiplexor = QuantumGate.multiplexor(unitaries: [identity, pauliX])
        #expect(multiplexor.qubitsRequired == 2, "Multiplexor with 2 unitaries of 2x2 should require 2 qubits (1 control + 1 target)")
    }

    @Test("Multiplexor with 4 unitaries of 2x2 requires 3 qubits")
    func multiplexorFourUnitariesRequiresThreeQubits() {
        let identity: [[Complex<Double>]] = [[.one, .zero], [.zero, .one]]
        let pauliX: [[Complex<Double>]] = [[.zero, .one], [.one, .zero]]
        let pauliY: [[Complex<Double>]] = [[.zero, -Complex.i], [Complex.i, .zero]]
        let pauliZ: [[Complex<Double>]] = [[.one, .zero], [.zero, Complex(-1.0, 0.0)]]
        let multiplexor = QuantumGate.multiplexor(unitaries: [identity, pauliX, pauliY, pauliZ])
        #expect(multiplexor.qubitsRequired == 3, "Multiplexor with 4 unitaries of 2x2 should require 3 qubits (2 control + 1 target)")
    }

    @Test("Multiplexor with 2 unitaries of 4x4 requires 3 qubits")
    func multiplexorTwoFourByFourRequiresThreeQubits() {
        let identity4: [[Complex<Double>]] = [
            [.one, .zero, .zero, .zero],
            [.zero, .one, .zero, .zero],
            [.zero, .zero, .one, .zero],
            [.zero, .zero, .zero, .one],
        ]
        let multiplexor = QuantumGate.multiplexor(unitaries: [identity4, identity4])
        #expect(multiplexor.qubitsRequired == 3, "Multiplexor with 2 unitaries of 4x4 should require 3 qubits (1 control + 2 target)")
    }
}

/// Validates symbolic parameter binding for givens and xx gates.
/// Tests that parameterized gates bind correctly and produce expected transformations.
/// Ensures variational quantum algorithms can use hardware-native gates.
@Suite("Givens and XX Parameter Binding")
struct GivensXXParameterBindingTests {
    @Test("Givens with symbolic parameter binds correctly")
    func givensSymbolicBinds() {
        let theta = Parameter(name: "theta")
        let gate = QuantumGate.givens(.parameter(theta))
        let bound = gate.bound(with: ["theta": .pi / 4])

        #expect(!bound.isParameterized, "Bound givens should not be parameterized")

        let boundMatrix = bound.matrix()
        let directMatrix = QuantumGate.givens(.pi / 4).matrix()

        #expect(QuantumGate.matricesEqual(boundMatrix, directMatrix), "Bound givens matrix should match direct construction")
    }

    @Test("Givens bound gate produces correct state transformation")
    func givensBoundProducesCorrectTransformation() {
        let theta = Parameter(name: "theta")
        let gate = QuantumGate.givens(.parameter(theta))
        let bound = gate.bound(with: ["theta": .pi / 2])

        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(1, to: .one)

        let result = GateApplication.apply(bound, to: [0, 1], state: state)

        #expect(abs(result.probability(of: 2) - 1.0) < 1e-10, "Bound givens(pi/2) should map |01⟩ to |10⟩")
    }

    @Test("XX with symbolic parameter binds correctly")
    func xxSymbolicBinds() {
        let theta = Parameter(name: "theta")
        let gate = QuantumGate.xx(.parameter(theta))
        let bound = gate.bound(with: ["theta": .pi / 4])

        #expect(!bound.isParameterized, "Bound XX should not be parameterized")

        let boundMatrix = bound.matrix()
        let directMatrix = QuantumGate.xx(.pi / 4).matrix()

        #expect(QuantumGate.matricesEqual(boundMatrix, directMatrix), "Bound XX matrix should match direct construction")
    }

    @Test("XX bound gate produces correct entanglement")
    func xxBoundProducesEntanglement() {
        let theta = Parameter(name: "theta")
        let gate = QuantumGate.xx(.parameter(theta))
        let bound = gate.bound(with: ["theta": .pi / 4])

        let state = QuantumState(qubits: 2)
        let result = GateApplication.apply(bound, to: [0, 1], state: state)

        let prob00 = result.probability(of: 0)
        let prob11 = result.probability(of: 3)

        #expect(prob00 > 0.4, "Bound XX(pi/4) on |00⟩ should give significant probability for |00⟩")
        #expect(prob11 > 0.4, "Bound XX(pi/4) on |00⟩ should give significant probability for |11⟩")
        #expect(abs(prob00 + prob11 - 1.0) < 1e-10, "Bound XX(pi/4) should only produce |00⟩ and |11⟩ components")
    }
}

/// Validates inverse property for all hardware gates.
/// Tests that gate inverse operations correctly recover original quantum states.
/// Ensures reversibility for quantum error correction and uncomputation.
@Suite("Hardware Gate Inverse Properties")
struct HardwareGateInverseTests {
    @Test("iSWAP inverse applied twice returns to original state")
    func iswapInverseTwiceReturnsOriginal() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(1, to: .one)

        let afterISwap = GateApplication.apply(.iswap, to: [0, 1], state: state)
        let afterInverse = GateApplication.apply(QuantumGate.iswap.inverse, to: [0, 1], state: afterISwap)

        for i in 0 ..< 4 {
            let original = state.amplitude(of: i)
            let recovered = afterInverse.amplitude(of: i)
            #expect(abs(original.real - recovered.real) < 1e-10, "iSWAP inverse should recover original state for amplitude \(i) real part")
            #expect(abs(original.imaginary - recovered.imaginary) < 1e-10, "iSWAP inverse should recover original state for amplitude \(i) imaginary part")
        }
    }

    @Test("sqrtISwap inverse times sqrtISwap equals identity")
    func sqrtISwapInverseTimesForwardIsIdentity() {
        let sqrtISwapMatrix = QuantumGate.sqrtISwap.matrix()
        let sqrtISwapInverseMatrix = QuantumGate.sqrtISwap.inverse.matrix()
        let product = QuantumGate.matrixMultiply(sqrtISwapInverseMatrix, sqrtISwapMatrix)

        #expect(QuantumGate.isIdentityMatrix(product), "sqrtISwap.inverse * sqrtISwap should equal identity")
    }

    @Test("sqrtISwap inverse recovers original state")
    func sqrtISwapInverseRecoversState() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: Complex(0.5, 0.0))
        state.setAmplitude(1, to: Complex(0.5, 0.0))
        state.setAmplitude(2, to: Complex(0.5, 0.0))
        state.setAmplitude(3, to: Complex(0.5, 0.0))

        let afterForward = GateApplication.apply(.sqrtISwap, to: [0, 1], state: state)
        let afterInverse = GateApplication.apply(QuantumGate.sqrtISwap.inverse, to: [0, 1], state: afterForward)

        for i in 0 ..< 4 {
            let original = state.amplitude(of: i)
            let recovered = afterInverse.amplitude(of: i)
            #expect(abs(original.real - recovered.real) < 1e-10, "sqrtISwap inverse should recover original for amplitude \(i) real part")
            #expect(abs(original.imaginary - recovered.imaginary) < 1e-10, "sqrtISwap inverse should recover original for amplitude \(i) imaginary part")
        }
    }

    @Test("Givens(theta) inverse equals Givens(-theta)")
    func givensInverseEqualsNegatedTheta() {
        let theta = 0.7
        let givensMatrix = QuantumGate.givens(theta).matrix()
        let inverseMatrix = QuantumGate.givens(theta).inverse.matrix()
        let negatedMatrix = QuantumGate.givens(-theta).matrix()

        #expect(QuantumGate.matricesEqual(inverseMatrix, negatedMatrix), "Givens(theta).inverse should equal Givens(-theta)")

        let product = QuantumGate.matrixMultiply(inverseMatrix, givensMatrix)
        #expect(QuantumGate.isIdentityMatrix(product), "Givens inverse times forward should equal identity")
    }

    @Test("XX(theta) inverse equals XX(-theta)")
    func xxInverseEqualsNegatedTheta() {
        let theta = 0.567
        let xxMatrix = QuantumGate.xx(theta).matrix()
        let inverseMatrix = QuantumGate.xx(theta).inverse.matrix()
        let negatedMatrix = QuantumGate.xx(-theta).matrix()

        #expect(QuantumGate.matricesEqual(inverseMatrix, negatedMatrix), "XX(theta).inverse should equal XX(-theta)")

        let product = QuantumGate.matrixMultiply(inverseMatrix, xxMatrix)
        #expect(QuantumGate.isIdentityMatrix(product), "XX inverse times forward should equal identity")
    }

    @Test("Diagonal inverse negates all phases")
    func diagonalInverseNegatesPhases() {
        let phases = [0.1, 0.2, 0.3, 0.4]
        let diagonal = QuantumGate.diagonal(phases: phases)
        let inverse = diagonal.inverse

        let diagonalMatrix = diagonal.matrix()
        let inverseMatrix = inverse.matrix()
        let product = QuantumGate.matrixMultiply(inverseMatrix, diagonalMatrix)

        #expect(QuantumGate.isIdentityMatrix(product), "Diagonal inverse times forward should equal identity")

        let negatedPhasesGate = QuantumGate.diagonal(phases: phases.map { -$0 })
        let negatedMatrix = negatedPhasesGate.matrix()

        #expect(QuantumGate.matricesEqual(inverseMatrix, negatedMatrix), "Diagonal inverse should equal diagonal with negated phases")
    }

    @Test("Multiplexor inverse is Hermitian conjugate of unitaries")
    func multiplexorInverseIsHermitianConjugate() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        let hadamard: [[Complex<Double>]] = [
            [Complex(invSqrt2, 0.0), Complex(invSqrt2, 0.0)],
            [Complex(invSqrt2, 0.0), Complex(-invSqrt2, 0.0)],
        ]
        let pauliY: [[Complex<Double>]] = [
            [.zero, -Complex.i],
            [Complex.i, .zero],
        ]

        let multiplexor = QuantumGate.multiplexor(unitaries: [hadamard, pauliY])
        let inverse = multiplexor.inverse

        let multiplexorMatrix = multiplexor.matrix()
        let inverseMatrix = inverse.matrix()
        let product = QuantumGate.matrixMultiply(inverseMatrix, multiplexorMatrix)

        #expect(QuantumGate.isIdentityMatrix(product), "Multiplexor inverse times forward should equal identity")
    }

    @Test("Multiplexor inverse recovers original state")
    func multiplexorInverseRecoversState() {
        let identity: [[Complex<Double>]] = [[.one, .zero], [.zero, .one]]
        let pauliX: [[Complex<Double>]] = [[.zero, .one], [.one, .zero]]

        let multiplexor = QuantumGate.multiplexor(unitaries: [identity, pauliX])
        let inverse = multiplexor.inverse

        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: Complex(0.5, 0.0))
        state.setAmplitude(1, to: Complex(0.5, 0.0))
        state.setAmplitude(2, to: Complex(0.5, 0.0))
        state.setAmplitude(3, to: Complex(0.5, 0.0))

        let afterForward = GateApplication.apply(multiplexor, to: [0, 1], state: state)
        let afterInverse = GateApplication.apply(inverse, to: [0, 1], state: afterForward)

        for i in 0 ..< 4 {
            let original = state.amplitude(of: i)
            let recovered = afterInverse.amplitude(of: i)
            #expect(abs(original.real - recovered.real) < 1e-10, "Multiplexor inverse should recover original for amplitude \(i) real part")
            #expect(abs(original.imaginary - recovered.imaginary) < 1e-10, "Multiplexor inverse should recover original for amplitude \(i) imaginary part")
        }
    }
}

/// Test suite for YY (Ising-YY) interaction gate matrix properties.
/// Validates unitarity, identity at theta=0, inverse relationship,
/// and correct matrix element structure for the RYY rotation gate.
@Suite("YY Gate Properties")
struct YYGatePropertiesTests {
    @Test("YY gate requires 2 qubits")
    func yyRequiresTwoQubits() {
        #expect(QuantumGate.yy(0.5).qubitsRequired == 2, "YY gate should require exactly 2 qubits for the two-body interaction")
    }

    @Test("YY gate matrix is unitary for multiple angles")
    func yyIsUnitaryForMultipleAngles() {
        let angles = [0.0, Double.pi / 4, Double.pi / 2, Double.pi]
        for angle in angles {
            let matrix = QuantumGate.yy(angle).matrix()
            #expect(QuantumGate.isUnitary(matrix), "YY(\(angle)) matrix should be unitary to preserve quantum state norms")
        }
    }

    @Test("YY(0) equals identity matrix")
    func yyZeroIsIdentity() {
        let matrix = QuantumGate.yy(0.0).matrix()
        #expect(QuantumGate.isIdentityMatrix(matrix), "YY(0) should equal the identity matrix since cos(0)=1 and sin(0)=0")
    }

    @Test("YY gate is not Hermitian")
    func yyIsNotHermitian() {
        #expect(!QuantumGate.yy(0.5).isHermitian, "YY gate with nonzero angle should not be Hermitian since it is a parameterized rotation")
    }

    @Test("YY inverse is YY(-theta)")
    func yyInverseIsNegatedTheta() {
        let theta = 1.234
        let yyMatrix = QuantumGate.yy(theta).matrix()
        let yyInverseMatrix = QuantumGate.yy(-theta).matrix()
        let product = QuantumGate.matrixMultiply(yyMatrix, yyInverseMatrix)
        #expect(QuantumGate.isIdentityMatrix(product), "YY(theta) * YY(-theta) should equal identity confirming inverse relationship")
    }

    @Test("YY matrix has correct structure for theta=pi/4")
    func yyMatrixStructureAtPiOverFour() {
        let theta = Double.pi / 4
        let matrix = QuantumGate.yy(theta).matrix()
        let cosTheta = cos(theta)
        let sinTheta = sin(theta)

        #expect(abs(matrix[0][0].real - cosTheta) < 1e-10, "YY(pi/4)[0][0] real part should be cos(pi/4)")
        #expect(abs(matrix[0][0].imaginary) < 1e-10, "YY(pi/4)[0][0] imaginary part should be zero")

        #expect(abs(matrix[0][1].real) < 1e-10, "YY(pi/4)[0][1] real part should be zero")
        #expect(abs(matrix[0][1].imaginary) < 1e-10, "YY(pi/4)[0][1] imaginary part should be zero")

        #expect(abs(matrix[0][2].real) < 1e-10, "YY(pi/4)[0][2] real part should be zero")
        #expect(abs(matrix[0][2].imaginary) < 1e-10, "YY(pi/4)[0][2] imaginary part should be zero")

        #expect(abs(matrix[0][3].real) < 1e-10, "YY(pi/4)[0][3] real part should be zero for i*sin(pi/4)")
        #expect(abs(matrix[0][3].imaginary - sinTheta) < 1e-10, "YY(pi/4)[0][3] imaginary part should be sin(pi/4)")

        #expect(abs(matrix[1][1].real - cosTheta) < 1e-10, "YY(pi/4)[1][1] real part should be cos(pi/4)")
        #expect(abs(matrix[1][1].imaginary) < 1e-10, "YY(pi/4)[1][1] imaginary part should be zero")

        #expect(abs(matrix[1][2].real) < 1e-10, "YY(pi/4)[1][2] real part should be zero for -i*sin(pi/4)")
        #expect(abs(matrix[1][2].imaginary - -sinTheta) < 1e-10, "YY(pi/4)[1][2] imaginary part should be -sin(pi/4)")

        #expect(abs(matrix[2][1].real) < 1e-10, "YY(pi/4)[2][1] real part should be zero for -i*sin(pi/4)")
        #expect(abs(matrix[2][1].imaginary - -sinTheta) < 1e-10, "YY(pi/4)[2][1] imaginary part should be -sin(pi/4)")

        #expect(abs(matrix[2][2].real - cosTheta) < 1e-10, "YY(pi/4)[2][2] real part should be cos(pi/4)")
        #expect(abs(matrix[2][2].imaginary) < 1e-10, "YY(pi/4)[2][2] imaginary part should be zero")

        #expect(abs(matrix[3][0].real) < 1e-10, "YY(pi/4)[3][0] real part should be zero for i*sin(pi/4)")
        #expect(abs(matrix[3][0].imaginary - sinTheta) < 1e-10, "YY(pi/4)[3][0] imaginary part should be sin(pi/4)")

        #expect(abs(matrix[3][3].real - cosTheta) < 1e-10, "YY(pi/4)[3][3] real part should be cos(pi/4)")
        #expect(abs(matrix[3][3].imaginary) < 1e-10, "YY(pi/4)[3][3] imaginary part should be zero")
    }
}

/// Test suite for YY (Ising-YY) interaction gate application to quantum states.
/// Validates identity behavior at theta=0, entanglement creation,
/// normalization preservation, and inverse recovery on multi-qubit states.
@Suite("YY Gate Application")
struct YYGateApplicationTests {
    @Test("YY(0) on |00⟩ gives |00⟩")
    func yyZeroOnZeroZeroIsIdentity() {
        let state = QuantumState(qubits: 2)
        let result = GateApplication.apply(.yy(0.0), to: [0, 1], state: state)

        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "YY(0) should leave |00⟩ unchanged since it acts as identity")
        #expect(abs(result.amplitude(of: 0).real - 1.0) < 1e-10, "YY(0) on |00⟩ should have amplitude 1 for |00⟩ real part")
        #expect(abs(result.amplitude(of: 0).imaginary) < 1e-10, "YY(0) on |00⟩ should have amplitude 0 for |00⟩ imaginary part")
    }

    @Test("YY(pi/2) on |00⟩ produces correct amplitudes")
    func yyPiOverTwoOnZeroZero() {
        let state = QuantumState(qubits: 2)
        let result = GateApplication.apply(.yy(.pi / 2), to: [0, 1], state: state)

        let cosVal = cos(Double.pi / 2)
        let sinVal = sin(Double.pi / 2)

        let amp00 = result.amplitude(of: 0)
        #expect(abs(amp00.real - cosVal) < 1e-10, "YY(pi/2) on |00⟩: |00⟩ amplitude real part should be cos(pi/2)")
        #expect(abs(amp00.imaginary) < 1e-10, "YY(pi/2) on |00⟩: |00⟩ amplitude imaginary part should be zero")

        let amp01 = result.amplitude(of: 1)
        #expect(abs(amp01.real) < 1e-10, "YY(pi/2) on |00⟩: |01⟩ amplitude real part should be zero")
        #expect(abs(amp01.imaginary) < 1e-10, "YY(pi/2) on |00⟩: |01⟩ amplitude imaginary part should be zero")

        let amp10 = result.amplitude(of: 2)
        #expect(abs(amp10.real) < 1e-10, "YY(pi/2) on |00⟩: |10⟩ amplitude real part should be zero")
        #expect(abs(amp10.imaginary) < 1e-10, "YY(pi/2) on |00⟩: |10⟩ amplitude imaginary part should be zero")

        let amp11 = result.amplitude(of: 3)
        #expect(abs(amp11.real) < 1e-10, "YY(pi/2) on |00⟩: |11⟩ amplitude real part should be zero for i*sin(pi/2)")
        #expect(abs(amp11.imaginary - sinVal) < 1e-10, "YY(pi/2) on |00⟩: |11⟩ amplitude imaginary part should be sin(pi/2)")
    }

    @Test("YY gate preserves state normalization")
    func yyPreservesNormalization() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: Complex(0.5, 0.0))
        state.setAmplitude(1, to: Complex(0.0, 0.5))
        state.setAmplitude(2, to: Complex(0.5, 0.0))
        state.setAmplitude(3, to: Complex(0.0, 0.5))

        let result = GateApplication.apply(.yy(0.789), to: [0, 1], state: state)

        #expect(result.isNormalized(), "YY gate should preserve quantum state normalization for any input state")
    }

    @Test("YY(theta) followed by YY(-theta) returns to original state")
    func yyForwardThenInverseRecoversState() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: Complex(0.5, 0.0))
        state.setAmplitude(1, to: Complex(0.5, 0.0))
        state.setAmplitude(2, to: Complex(0.5, 0.0))
        state.setAmplitude(3, to: Complex(0.5, 0.0))

        let theta = 0.567
        let afterForward = GateApplication.apply(.yy(theta), to: [0, 1], state: state)
        let afterInverse = GateApplication.apply(.yy(-theta), to: [0, 1], state: afterForward)

        for i in 0 ..< 4 {
            let original = state.amplitude(of: i)
            let recovered = afterInverse.amplitude(of: i)
            #expect(abs(original.real - recovered.real) < 1e-10, "YY(-theta) should invert YY(theta) for amplitude \(i) real part")
            #expect(abs(original.imaginary - recovered.imaginary) < 1e-10, "YY(-theta) should invert YY(theta) for amplitude \(i) imaginary part")
        }
    }

    @Test("YY on superposition state produces correct result")
    func yyOnSuperpositionState() {
        let invSqrt2 = 1.0 / sqrt(2.0)
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: Complex(invSqrt2, 0.0))
        state.setAmplitude(1, to: Complex(invSqrt2, 0.0))

        let theta = Double.pi / 4
        let result = GateApplication.apply(.yy(theta), to: [0, 1], state: state)

        let cosTheta = cos(theta)
        let sinTheta = sin(theta)

        let expectedAmp0 = Complex(invSqrt2 * cosTheta, 0.0)
        let expectedAmp1 = Complex(invSqrt2 * cosTheta, 0.0)
        let expectedAmp2 = Complex(0.0, -invSqrt2 * sinTheta)
        let expectedAmp3 = Complex(0.0, invSqrt2 * sinTheta)

        let amp0 = result.amplitude(of: 0)
        #expect(abs(amp0.real - expectedAmp0.real) < 1e-10, "YY(pi/4) on (|00⟩+|01⟩)/sqrt(2): |00⟩ real part should match cos component")
        #expect(abs(amp0.imaginary - expectedAmp0.imaginary) < 1e-10, "YY(pi/4) on (|00⟩+|01⟩)/sqrt(2): |00⟩ imaginary part should be zero")

        let amp1 = result.amplitude(of: 1)
        #expect(abs(amp1.real - expectedAmp1.real) < 1e-10, "YY(pi/4) on (|00⟩+|01⟩)/sqrt(2): |01⟩ real part should match cos component")
        #expect(abs(amp1.imaginary - expectedAmp1.imaginary) < 1e-10, "YY(pi/4) on (|00⟩+|01⟩)/sqrt(2): |01⟩ imaginary part should be zero")

        let amp2 = result.amplitude(of: 2)
        #expect(abs(amp2.real - expectedAmp2.real) < 1e-10, "YY(pi/4) on (|00⟩+|01⟩)/sqrt(2): |10⟩ real part should be zero")
        #expect(abs(amp2.imaginary - expectedAmp2.imaginary) < 1e-10, "YY(pi/4) on (|00⟩+|01⟩)/sqrt(2): |10⟩ imaginary part should match -sin component")

        let amp3 = result.amplitude(of: 3)
        #expect(abs(amp3.real - expectedAmp3.real) < 1e-10, "YY(pi/4) on (|00⟩+|01⟩)/sqrt(2): |11⟩ real part should be zero")
        #expect(abs(amp3.imaginary - expectedAmp3.imaginary) < 1e-10, "YY(pi/4) on (|00⟩+|01⟩)/sqrt(2): |11⟩ imaginary part should match sin component")
    }

    @Test("YY gate symmetry: same result regardless of qubit order assignment")
    func yyGateSymmetry() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: Complex(0.6, 0.0))
        state.setAmplitude(1, to: Complex(0.0, 0.4))
        state.setAmplitude(2, to: Complex(0.4, 0.0))
        state.setAmplitude(3, to: Complex(0.0, 0.6))

        let theta = 0.789
        let matrix = QuantumGate.yy(theta).matrix()

        let isSymmetric0103 = abs(matrix[0][3].imaginary - matrix[3][0].imaginary) < 1e-10
        #expect(isSymmetric0103, "YY matrix should have symmetric i*sin(theta) in [0][3] and [3][0] positions")

        let isSymmetric1221 = abs(matrix[1][2].imaginary - matrix[2][1].imaginary) < 1e-10
        #expect(isSymmetric1221, "YY matrix should have symmetric -i*sin(theta) in [1][2] and [2][1] positions")

        let diagonalsEqual = abs(matrix[0][0].real - matrix[1][1].real) < 1e-10
            && abs(matrix[1][1].real - matrix[2][2].real) < 1e-10
            && abs(matrix[2][2].real - matrix[3][3].real) < 1e-10
        #expect(diagonalsEqual, "YY matrix diagonal elements should all equal cos(theta) reflecting qubit exchange symmetry")
    }
}

/// Test suite for ZZ (Ising-ZZ) interaction gate matrix properties.
/// Validates unitarity, identity at theta=0, diagonal structure,
/// inverse relationship, and correct phase entries for the RZZ rotation gate.
@Suite("ZZ Gate Properties")
struct ZZGatePropertiesTests {
    @Test("ZZ gate requires 2 qubits")
    func zzRequiresTwoQubits() {
        #expect(QuantumGate.zz(0.5).qubitsRequired == 2, "ZZ gate should require exactly 2 qubits for the two-body interaction")
    }

    @Test("ZZ gate matrix is unitary for multiple angles")
    func zzIsUnitaryForMultipleAngles() {
        let angles = [0.0, Double.pi / 4, Double.pi / 2, Double.pi]
        for angle in angles {
            let matrix = QuantumGate.zz(angle).matrix()
            #expect(QuantumGate.isUnitary(matrix), "ZZ(\(angle)) matrix should be unitary to preserve quantum state norms")
        }
    }

    @Test("ZZ(0) equals identity matrix")
    func zzZeroIsIdentity() {
        let matrix = QuantumGate.zz(0.0).matrix()
        #expect(QuantumGate.isIdentityMatrix(matrix), "ZZ(0) should equal the identity matrix since e^(i*0)=1")
    }

    @Test("ZZ gate is not Hermitian")
    func zzIsNotHermitian() {
        #expect(!QuantumGate.zz(0.5).isHermitian, "ZZ gate with nonzero angle should not be Hermitian since it is a parameterized rotation")
    }

    @Test("ZZ gate is diagonal: all off-diagonal elements are zero")
    func zzIsDiagonal() {
        let theta = 0.789
        let matrix = QuantumGate.zz(theta).matrix()
        for row in 0 ..< 4 {
            for col in 0 ..< 4 {
                if row != col {
                    #expect(abs(matrix[row][col].real) < 1e-10, "ZZ(\(theta))[\(row)][\(col)] real part should be zero for off-diagonal element")
                    #expect(abs(matrix[row][col].imaginary) < 1e-10, "ZZ(\(theta))[\(row)][\(col)] imaginary part should be zero for off-diagonal element")
                }
            }
        }
    }

    @Test("ZZ inverse is ZZ(-theta)")
    func zzInverseIsNegatedTheta() {
        let theta = 1.234
        let zzMatrix = QuantumGate.zz(theta).matrix()
        let zzInverseMatrix = QuantumGate.zz(-theta).matrix()
        let product = QuantumGate.matrixMultiply(zzMatrix, zzInverseMatrix)
        #expect(QuantumGate.isIdentityMatrix(product), "ZZ(theta) * ZZ(-theta) should equal identity confirming inverse relationship")
    }

    @Test("ZZ matrix has correct diagonal: diag(e^(-itheta), e^(itheta), e^(itheta), e^(-itheta))")
    func zzDiagonalValuesAtPiOverFour() {
        let theta = Double.pi / 4
        let matrix = QuantumGate.zz(theta).matrix()

        let negPhaseReal = cos(-theta)
        let negPhaseImag = sin(-theta)
        let posPhaseReal = cos(theta)
        let posPhaseImag = sin(theta)

        #expect(abs(matrix[0][0].real - negPhaseReal) < 1e-10, "ZZ(pi/4)[0][0] real part should be cos(-pi/4) for e^(-i*pi/4)")
        #expect(abs(matrix[0][0].imaginary - negPhaseImag) < 1e-10, "ZZ(pi/4)[0][0] imaginary part should be sin(-pi/4) for e^(-i*pi/4)")

        #expect(abs(matrix[1][1].real - posPhaseReal) < 1e-10, "ZZ(pi/4)[1][1] real part should be cos(pi/4) for e^(i*pi/4)")
        #expect(abs(matrix[1][1].imaginary - posPhaseImag) < 1e-10, "ZZ(pi/4)[1][1] imaginary part should be sin(pi/4) for e^(i*pi/4)")

        #expect(abs(matrix[2][2].real - posPhaseReal) < 1e-10, "ZZ(pi/4)[2][2] real part should be cos(pi/4) for e^(i*pi/4)")
        #expect(abs(matrix[2][2].imaginary - posPhaseImag) < 1e-10, "ZZ(pi/4)[2][2] imaginary part should be sin(pi/4) for e^(i*pi/4)")

        #expect(abs(matrix[3][3].real - negPhaseReal) < 1e-10, "ZZ(pi/4)[3][3] real part should be cos(-pi/4) for e^(-i*pi/4)")
        #expect(abs(matrix[3][3].imaginary - negPhaseImag) < 1e-10, "ZZ(pi/4)[3][3] imaginary part should be sin(-pi/4) for e^(-i*pi/4)")
    }
}

/// Test suite for ZZ (Ising-ZZ) interaction gate application to quantum states.
/// Validates phase shifts on computational basis states, normalization preservation,
/// probability invariance for diagonal gates, and inverse recovery on arbitrary states.
@Suite("ZZ Gate Application")
struct ZZGateApplicationTests {
    @Test("ZZ(0) on |00⟩ gives |00⟩")
    func zzZeroOnZeroZeroIsIdentity() {
        let state = QuantumState(qubits: 2)
        let result = GateApplication.apply(.zz(0.0), to: [0, 1], state: state)

        #expect(abs(result.probability(of: 0) - 1.0) < 1e-10, "ZZ(0) should leave |00⟩ unchanged since it acts as identity")
        #expect(abs(result.amplitude(of: 0).real - 1.0) < 1e-10, "ZZ(0) on |00⟩ should have amplitude 1 for |00⟩ real part")
        #expect(abs(result.amplitude(of: 0).imaginary) < 1e-10, "ZZ(0) on |00⟩ should have amplitude 0 for |00⟩ imaginary part")
    }

    @Test("ZZ on |00⟩ applies e^(-itheta) phase")
    func zzOnZeroZeroAppliesNegativePhase() {
        let theta = Double.pi / 4
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: .one)

        let result = GateApplication.apply(.zz(theta), to: [0, 1], state: state)

        let amp = result.amplitude(of: 0)
        #expect(abs(amp.real - cos(-theta)) < 1e-10, "ZZ(pi/4) on |00⟩ should give e^(-i*pi/4) phase: real part cos(-pi/4)")
        #expect(abs(amp.imaginary - sin(-theta)) < 1e-10, "ZZ(pi/4) on |00⟩ should give e^(-i*pi/4) phase: imaginary part sin(-pi/4)")
    }

    @Test("ZZ on |01⟩ applies e^(itheta) phase")
    func zzOnZeroOneAppliesPositivePhase() {
        let theta = Double.pi / 4
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(1, to: .one)

        let result = GateApplication.apply(.zz(theta), to: [0, 1], state: state)

        let amp = result.amplitude(of: 1)
        #expect(abs(amp.real - cos(theta)) < 1e-10, "ZZ(pi/4) on |01⟩ should give e^(i*pi/4) phase: real part cos(pi/4)")
        #expect(abs(amp.imaginary - sin(theta)) < 1e-10, "ZZ(pi/4) on |01⟩ should give e^(i*pi/4) phase: imaginary part sin(pi/4)")
    }

    @Test("ZZ on |10⟩ applies e^(itheta) phase")
    func zzOnOneZeroAppliesPositivePhase() {
        let theta = Double.pi / 4
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(2, to: .one)

        let result = GateApplication.apply(.zz(theta), to: [0, 1], state: state)

        let amp = result.amplitude(of: 2)
        #expect(abs(amp.real - cos(theta)) < 1e-10, "ZZ(pi/4) on |10⟩ should give e^(i*pi/4) phase: real part cos(pi/4)")
        #expect(abs(amp.imaginary - sin(theta)) < 1e-10, "ZZ(pi/4) on |10⟩ should give e^(i*pi/4) phase: imaginary part sin(pi/4)")
    }

    @Test("ZZ on |11⟩ applies e^(-itheta) phase")
    func zzOnOneOneAppliesNegativePhase() {
        let theta = Double.pi / 4
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: .zero)
        state.setAmplitude(3, to: .one)

        let result = GateApplication.apply(.zz(theta), to: [0, 1], state: state)

        let amp = result.amplitude(of: 3)
        #expect(abs(amp.real - cos(-theta)) < 1e-10, "ZZ(pi/4) on |11⟩ should give e^(-i*pi/4) phase: real part cos(-pi/4)")
        #expect(abs(amp.imaginary - sin(-theta)) < 1e-10, "ZZ(pi/4) on |11⟩ should give e^(-i*pi/4) phase: imaginary part sin(-pi/4)")
    }

    @Test("ZZ preserves normalization")
    func zzPreservesNormalization() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: Complex(0.5, 0.0))
        state.setAmplitude(1, to: Complex(0.0, 0.5))
        state.setAmplitude(2, to: Complex(0.5, 0.0))
        state.setAmplitude(3, to: Complex(0.0, 0.5))

        let result = GateApplication.apply(.zz(0.789), to: [0, 1], state: state)

        #expect(result.isNormalized(), "ZZ gate should preserve quantum state normalization for any input state")
    }

    @Test("ZZ preserves computational basis probabilities (diagonal gate)")
    func zzPreservesBasisProbabilities() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: Complex(0.5, 0.0))
        state.setAmplitude(1, to: Complex(0.0, 0.5))
        state.setAmplitude(2, to: Complex(0.5, 0.0))
        state.setAmplitude(3, to: Complex(0.0, 0.5))

        let originalProbs = (0 ..< 4).map { state.probability(of: $0) }

        let result = GateApplication.apply(.zz(1.234), to: [0, 1], state: state)

        for i in 0 ..< 4 {
            let resultProb = result.probability(of: i)
            #expect(abs(resultProb - originalProbs[i]) < 1e-10, "ZZ diagonal gate should preserve probability of basis state \(i) since it only applies phases")
        }
    }

    @Test("ZZ(theta) followed by ZZ(-theta) returns to original state")
    func zzForwardThenInverseRecoversState() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: Complex(0.5, 0.0))
        state.setAmplitude(1, to: Complex(0.5, 0.0))
        state.setAmplitude(2, to: Complex(0.5, 0.0))
        state.setAmplitude(3, to: Complex(0.5, 0.0))

        let theta = 0.567
        let afterForward = GateApplication.apply(.zz(theta), to: [0, 1], state: state)
        let afterInverse = GateApplication.apply(.zz(-theta), to: [0, 1], state: afterForward)

        for i in 0 ..< 4 {
            let original = state.amplitude(of: i)
            let recovered = afterInverse.amplitude(of: i)
            #expect(abs(original.real - recovered.real) < 1e-10, "ZZ(-theta) should invert ZZ(theta) for amplitude \(i) real part")
            #expect(abs(original.imaginary - recovered.imaginary) < 1e-10, "ZZ(-theta) should invert ZZ(theta) for amplitude \(i) imaginary part")
        }
    }

    @Test("ZZ(pi) applies specific known phases")
    func zzPiAppliesKnownPhases() {
        var state = QuantumState(qubits: 2)
        state.setAmplitude(0, to: .one)
        let result00 = GateApplication.apply(.zz(.pi), to: [0, 1], state: state)
        let amp00 = result00.amplitude(of: 0)
        #expect(abs(amp00.real - cos(-Double.pi)) < 1e-10, "ZZ(pi) on |00⟩ should apply e^(-i*pi) phase: real part should be -1")
        #expect(abs(amp00.imaginary - sin(-Double.pi)) < 1e-10, "ZZ(pi) on |00⟩ should apply e^(-i*pi) phase: imaginary part should be ~0")

        var state01 = QuantumState(qubits: 2)
        state01.setAmplitude(0, to: .zero)
        state01.setAmplitude(1, to: .one)
        let result01 = GateApplication.apply(.zz(.pi), to: [0, 1], state: state01)
        let amp01 = result01.amplitude(of: 1)
        #expect(abs(amp01.real - cos(Double.pi)) < 1e-10, "ZZ(pi) on |01⟩ should apply e^(i*pi) phase: real part should be -1")
        #expect(abs(amp01.imaginary - sin(Double.pi)) < 1e-10, "ZZ(pi) on |01⟩ should apply e^(i*pi) phase: imaginary part should be ~0")

        var state11 = QuantumState(qubits: 2)
        state11.setAmplitude(0, to: .zero)
        state11.setAmplitude(3, to: .one)
        let result11 = GateApplication.apply(.zz(.pi), to: [0, 1], state: state11)
        let amp11 = result11.amplitude(of: 3)
        #expect(abs(amp11.real - cos(-Double.pi)) < 1e-10, "ZZ(pi) on |11⟩ should apply e^(-i*pi) phase: real part should be -1")
        #expect(abs(amp11.imaginary - sin(-Double.pi)) < 1e-10, "ZZ(pi) on |11⟩ should apply e^(-i*pi) phase: imaginary part should be ~0")
    }
}
