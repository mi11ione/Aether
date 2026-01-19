// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Quantum GPU compute shaders: Metal kernels for quantum gate application
///
/// Implements massively parallel quantum gate operations on Apple GPUs using Metal.
/// Each kernel processes quantum state amplitudes in parallel across thousands of GPU threads,
/// providing significant speedup over CPU for states with 10 or more qubits.
///
/// Single-qubit gates have each thread process one amplitude pair (i, j) where indices differ
/// only in target qubit. Two-qubit gates process amplitude quartets for states differing in
/// control/target qubits. CNOT and Toffoli gates process one amplitude per thread using
/// separate input/output buffers.
///
/// Thread grid size is 2^n for n-qubit states with threadgroups auto-sized to hardware limits.
/// Memory access patterns use coalesced reads/writes for optimal bandwidth. Complex amplitudes
/// use Float32 precision matching Swift Complex<Float> layout for zero-copy transfer.
///
/// Race conditions are avoided through unique index ownership for single/two-qubit gates
/// (in-place update safe) and separate input/output buffers for CNOT/Toffoli gates.
///
/// Kernels: applySingleQubitGate (2x2 matrix-vector), applyCNOT (conditional swap),
/// applyTwoQubitGate (4x4 matrix-vector), applyToffoli (double-controlled NOT),
/// csrSparseMatVec (sparse Hamiltonian multiply).
///
/// **Example:**
/// ```swift
/// let metalApp = MetalGateApplication()
/// let state = QuantumState(qubits: 12)
/// let newState = await metalApp.apply(gate: .hadamard, to: 0, state: state)
/// ```
///
/// All kernels preserve unitarity (U†U = I) and normalization (Σ|cᵢ|² = 1) through exact
/// matrix operations. Bit ordering uses little-endian qubit indexing (qubit 0 is LSB).

#include <metal_stdlib>
using namespace metal;

/// Complex number structure matching Swift Complex<Float>
/// Memory layout: 8 bytes (4 bytes real + 4 bytes imaginary)
/// Alignment: Matches Swift's @frozen Complex<Float> for zero-copy transfer
struct ComplexFloat {
    float real;
    float imaginary;
};

// MARK: - Complex Number Operations

/// Complex addition: (a + bi) + (c + di) = (a+c) + (b+d)i
/// Inline free function to avoid Metal address space issues with member functions
inline ComplexFloat complexAdd(ComplexFloat a, ComplexFloat b) {
    return ComplexFloat{a.real + b.real, a.imaginary + b.imaginary};
}

/// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
/// Uses fma() for better precision (single rounding) and potential speedup on FMA-capable GPUs
inline ComplexFloat complexMultiply(ComplexFloat a, ComplexFloat b) {
    return ComplexFloat{
        fma(a.real, b.real, -a.imaginary * b.imaginary),
        fma(a.real, b.imaginary, a.imaginary * b.real)
    };
}

/// Scalar multiplication: k(a + bi) = ka + kbi
/// Used for normalization and amplitude scaling
inline ComplexFloat complexScale(ComplexFloat a, float scalar) {
    return ComplexFloat{a.real * scalar, a.imaginary * scalar};
}

// MARK: - Kahan Summation for Improved Precision

/// Kahan summation accumulator for compensated floating-point addition.
///
/// Maintains running sum and compensation term to reduce accumulation error
/// from O(n·ε) to O(√n·ε) where n is the number of terms and ε is machine epsilon.
/// Critical for sparse matrix-vector multiplication where rows may have hundreds
/// of non-zero elements and naive summation loses significant precision.
///
/// Memory layout: 8 bytes (4 bytes sum + 4 bytes compensation)
struct KahanAccumulator {
    float sum;
    float compensation;
};

/// Kahan compensated addition: accumulator += value with error tracking.
///
/// Implements Kahan-Babushka-Neumaier algorithm for improved precision in
/// floating-point summation. Captures rounding error in compensation term
/// and feeds it back into subsequent additions. Reduces accumulation error
/// from O(n·ε) to O(√n·ε).
///
/// **Example:**
/// ```metal
/// KahanAccumulator acc = {0.0f, 0.0f};
/// for (uint i = 0; i < n; i++) {
///     kahanAdd(acc, values[i]);
/// }
/// float result = acc.sum + acc.compensation;
/// ```
///
/// - Parameters:
///   - acc: Kahan accumulator (modified in place)
///   - value: Value to add to accumulator
/// - Complexity: O(1) per addition, 4 FLOPs
inline void kahanAdd(thread KahanAccumulator& acc, float value) {
    float y = value - acc.compensation;
    float t = acc.sum + y;
    acc.compensation = (t - acc.sum) - y;
    acc.sum = t;
}

/// Complex Kahan accumulator for complex number summation.
///
/// Maintains separate Kahan accumulators for real and imaginary components,
/// providing compensated summation for both parts independently.
struct ComplexKahanAccumulator {
    KahanAccumulator real;
    KahanAccumulator imag;
};

/// Complex Kahan compensated addition for complex number summation.
///
/// Applies Kahan summation separately to real and imaginary components.
///
/// - Parameters:
///   - acc: Complex Kahan accumulator (modified in place)
///   - value: Complex value to add
/// - Complexity: O(1) per addition, 8 FLOPs
inline void complexKahanAdd(thread ComplexKahanAccumulator& acc, ComplexFloat value) {
    kahanAdd(acc.real, value.real);
    kahanAdd(acc.imag, value.imaginary);
}

/// Extracts final sum from complex Kahan accumulator including compensation.
///
/// - Parameter acc: Complex Kahan accumulator
/// - Returns: Final complex sum with compensation folded in
inline ComplexFloat complexKahanSum(ComplexKahanAccumulator acc) {
    return ComplexFloat{
        acc.real.sum + acc.real.compensation,
        acc.imag.sum + acc.imag.compensation
    };
}

// MARK: - Single-Qubit Gate Kernel

/// Apply single-qubit gate: parallel 2x2 matrix-vector multiplication
///
/// Transforms quantum state by applying 2x2 unitary matrix U to target qubit.
/// Each thread processes one amplitude pair (cᵢ, cⱼ) where i and j differ only
/// in the target qubit bit. Launches 2^(n-1) threads, each computing index pair
/// by inserting a 0 bit at targetQubit position, applying [cᵢ', cⱼ'] = U·[cᵢ, cⱼ],
/// and writing results in-place with no race conditions due to unique index ownership.
///
/// Index computation uses lowMask = (1 << targetQubit) - 1 to extract bits below
/// targetQubit, then i = (gid & lowMask) | ((gid & ~lowMask) << 1) shifts upper
/// bits left while keeping lower bits, and j = i | (1 << targetQubit) sets the
/// target bit to 1.
///
/// Each thread reads 2 amplitudes and writes 2 amplitudes. Adjacent threads access
/// nearby memory for coalesced bandwidth.
///
/// **Parameters:**:
/// - amplitudes: Input/output state vector (2^n complex amplitudes)
/// - targetQubit: Qubit index to apply gate to (0 to n-1)
/// - gateMatrix: 2x2 unitary matrix [g00, g01, g10, g11] row-major
/// - qubits: Total number of qubits (n)
/// - gid: Thread ID in grid (0 to 2^(n-1) - 1)
/// - Complexity: O(1) per thread, O(2^(n-1)) total threads
kernel void applySingleQubitGate(
    device ComplexFloat *amplitudes [[buffer(0)]],
    constant uint &targetQubit [[buffer(1)]],
    constant ComplexFloat *gateMatrix [[buffer(2)]],
    constant uint &qubits [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint numPairs = 1 << (qubits - 1);
    
    if (gid < numPairs) {
        const uint lowMask = (1 << targetQubit) - 1;
        const uint i = (gid & lowMask) | ((gid & ~lowMask) << 1);
        const uint j = i | (1 << targetQubit);
        
        ComplexFloat ci = amplitudes[i];
        ComplexFloat cj = amplitudes[j];

        ComplexFloat g00 = gateMatrix[0];
        ComplexFloat g01 = gateMatrix[1];
        ComplexFloat g10 = gateMatrix[2];
        ComplexFloat g11 = gateMatrix[3];
        
        ComplexFloat newCi = complexAdd(complexMultiply(g00, ci), complexMultiply(g01, cj));
        ComplexFloat newCj = complexAdd(complexMultiply(g10, ci), complexMultiply(g11, cj));
        
        amplitudes[i] = newCi;
        amplitudes[j] = newCj;
    }
}

// MARK: - CNOT Gate Kernel

/// Apply CNOT gate: optimized controlled-NOT implementation
///
/// Implements CNOT(control->target) with conditional amplitude swap, avoiding
/// complex arithmetic since CNOT only swaps amplitudes when control=1. Each
/// thread processes one amplitude: if control bit is 1, writes to flipped
/// target index; if 0, writes unchanged. Uses separate input/output buffers
/// to avoid read-after-write hazards.
///
/// Launches 2^n threads (one per amplitude). Each thread reads 1 amplitude
/// and writes 1 amplitude. Write pattern depends on qubit indices.
///
/// **Parameters:**
/// - amplitudes: Input state vector (read-only)
/// - controlQubit: Control qubit index (0 to n-1)
/// - targetQubit: Target qubit index (0 to n-1, ≠ control)
/// - qubits: Total number of qubits (n)
/// - outputAmplitudes: Output state vector (write-only)
/// - gid: Thread ID in grid (0 to 2^n - 1)
/// - Complexity: O(1) per thread, O(2^n) total threads
kernel void applyCNOT(
    device ComplexFloat *amplitudes [[buffer(0)]],
    constant uint &controlQubit [[buffer(1)]],
    constant uint &targetQubit [[buffer(2)]],
    constant uint &qubits [[buffer(3)]],
    device ComplexFloat *outputAmplitudes [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint stateSize = 1 << qubits;
    
    if (gid < stateSize) {
        const uint controlMask = 1 << controlQubit;
        const uint targetMask = 1 << targetQubit;
        
        if ((gid & controlMask) != 0) {
            const uint flipped = gid ^ targetMask;
            outputAmplitudes[flipped] = amplitudes[gid];
        } else {
            outputAmplitudes[gid] = amplitudes[gid];
        }
    }
}

// MARK: - Two-Qubit Gate Kernel

/// Apply two-qubit gate: parallel 4x4 matrix-vector multiplication
///
/// Transforms quantum state by applying 4x4 unitary matrix to control/target qubits.
/// Each thread processes one amplitude quartet (c00, c01, c10, c11) where indices
/// differ only in control and target bits. Launches 2^(n-2) threads, each computing
/// index quartet by inserting two 0 bits, applying [c'00, c'01, c'10, c'11]ᵀ = U·[c00, c01, c10, c11]ᵀ,
/// and writing in-place with unique index ownership.
///
/// Index computation sorts qubit positions (lo = min, hi = max), uses loMask to
/// extract bits below lo, inserts 0 at lo then hi to get base i00, then sets
/// control/target bits for i01, i10, i11.
///
/// Each thread reads 4 amplitudes and writes 4 amplitudes, performing 16 complex
/// multiplications and 12 complex additions.
///
/// **Parameters:**
/// - amplitudes: Input/output state vector (2^n complex amplitudes)
/// - controlQubit: Control qubit index (0 to n-1)
/// - targetQubit: Target qubit index (0 to n-1, ≠ control)
/// - gateMatrix: 4x4 unitary matrix (16 elements, row-major)
/// - qubits: Total number of qubits (n)
/// - gid: Thread ID in grid (0 to 2^(n-2) - 1)
/// - Complexity: O(1) per thread, O(2^(n-2)) total threads
kernel void applyTwoQubitGate(
    device ComplexFloat *amplitudes [[buffer(0)]],
    constant uint &controlQubit [[buffer(1)]],
    constant uint &targetQubit [[buffer(2)]],
    constant ComplexFloat *gateMatrix [[buffer(3)]],
    constant uint &qubits [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint numQuartets = 1 << (qubits - 2);
    
    if (gid < numQuartets) {
        const uint lo = min(controlQubit, targetQubit);
        const uint hi = max(controlQubit, targetQubit);
        
        const uint loMask = (1 << lo) - 1;
        const uint temp = (gid & loMask) | ((gid & ~loMask) << 1);
        
        const uint hiMask = (1 << hi) - 1;
        const uint i00 = (temp & hiMask) | ((temp & ~hiMask) << 1);
        
        const uint controlMask = 1 << controlQubit;
        const uint targetMask = 1 << targetQubit;
        
        const uint i01 = i00 | targetMask;
        const uint i10 = i00 | controlMask;
        const uint i11 = i00 | controlMask | targetMask;
        
        ComplexFloat c00 = amplitudes[i00];
        ComplexFloat c01 = amplitudes[i01];
        ComplexFloat c10 = amplitudes[i10];
        ComplexFloat c11 = amplitudes[i11];

        ComplexFloat new00 = complexAdd(
            complexAdd(complexMultiply(gateMatrix[0], c00), complexMultiply(gateMatrix[1], c01)),
            complexAdd(complexMultiply(gateMatrix[2], c10), complexMultiply(gateMatrix[3], c11))
        );
        ComplexFloat new01 = complexAdd(
            complexAdd(complexMultiply(gateMatrix[4], c00), complexMultiply(gateMatrix[5], c01)),
            complexAdd(complexMultiply(gateMatrix[6], c10), complexMultiply(gateMatrix[7], c11))
        );
        ComplexFloat new10 = complexAdd(
            complexAdd(complexMultiply(gateMatrix[8], c00), complexMultiply(gateMatrix[9], c01)),
            complexAdd(complexMultiply(gateMatrix[10], c10), complexMultiply(gateMatrix[11], c11))
        );
        ComplexFloat new11 = complexAdd(
            complexAdd(complexMultiply(gateMatrix[12], c00), complexMultiply(gateMatrix[13], c01)),
            complexAdd(complexMultiply(gateMatrix[14], c10), complexMultiply(gateMatrix[15], c11))
        );
        
        amplitudes[i00] = new00;
        amplitudes[i01] = new01;
        amplitudes[i10] = new10;
        amplitudes[i11] = new11;
    }
}

// MARK: - Toffoli Gate Kernel

/// Apply Toffoli gate: double-controlled NOT (CCNOT)
///
/// Implements three-qubit Toffoli gate that flips target qubit when both control
/// qubits are |1⟩. Essential for reversible classical logic and quantum error
/// correction, optimized to avoid 8x8 matrix multiplication. Each thread processes
/// one amplitude: if both control bits are 1, writes to flipped target index;
/// otherwise writes unchanged. Uses separate input/output buffers for race safety.
///
/// Launches 2^n threads (one per amplitude). Each thread reads 1 amplitude and
/// writes 1 amplitude. Approximately 25% of threads perform the swap operation.
///
/// **Parameters:**
/// - amplitudes: Input state vector (read-only)
/// - control1Qubit: First control qubit index (0 to n-1)
/// - control2Qubit: Second control qubit index (0 to n-1, ≠ control1)
/// - targetQubit: Target qubit index (0 to n-1, ≠ controls)
/// - qubits: Total number of qubits (n)
/// - outputAmplitudes: Output state vector (write-only)
/// - gid: Thread ID in grid (0 to 2^n - 1)
/// - Complexity: O(1) per thread, O(2^n) total threads
kernel void applyToffoli(
    device ComplexFloat *amplitudes [[buffer(0)]],
    constant uint &control1Qubit [[buffer(1)]],
    constant uint &control2Qubit [[buffer(2)]],
    constant uint &targetQubit [[buffer(3)]],
    constant uint &qubits [[buffer(4)]],
    device ComplexFloat *outputAmplitudes [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint stateSize = 1 << qubits;
    
    if (gid < stateSize) {
        const uint c1Mask = 1 << control1Qubit;
        const uint c2Mask = 1 << control2Qubit;
        const uint targetMask = 1 << targetQubit;
        
        if (((gid & c1Mask) != 0) && ((gid & c2Mask) != 0)) {
            const uint flipped = gid ^ targetMask;
            outputAmplitudes[flipped] = amplitudes[gid];
        } else {
            outputAmplitudes[gid] = amplitudes[gid];
        }
    }
}

// MARK: - CSR Sparse Matrix-Vector Multiply Kernel

/// CSR Sparse Matrix-Vector Multiply: output = H * input for Hamiltonian expectation values
///
/// Implements sparse matrix-vector multiplication using CSR (Compressed Sparse Row) format
/// with Kahan compensated summation for improved numerical precision. Each thread computes
/// one output element by accumulating the dot product of one sparse row with the input vector:
/// output[row] = Σⱼ H[row,j] * input[j], using Kahan summation to reduce accumulation error
/// from O(n·ε) to O(√n·ε) where n is the number of non-zeros per row.
///
/// Launches dimension threads (one per matrix row). Each thread reads O(nnz/dimension)
/// sparse elements on average, performing complex multiplication and Kahan accumulation
/// per non-zero. Kahan summation adds ~4 FLOPs per accumulation but dramatically improves
/// precision for rows with 10+ non-zero elements, typical in molecular Hamiltonians.
///
/// CSR format uses rowPointers[i] as the index where row i starts in columnIndices/values,
/// with rowPointers[i+1] - rowPointers[i] giving non-zeros in row i. columnIndices[k] holds
/// the column index and values[k] the complex value of the k-th non-zero element.
///
/// **Example:**
/// ```swift
/// let sparseH = SparseHamiltonian(observable: hamiltonian)
/// let energy = sparseH.expectationValue(state: state)
/// ```
///
/// **Parameters:**
/// - rowPointers: CSR row pointers [dimension+1]
/// - columnIndices: CSR column indices [nnz]
/// - values: Complex values [nnz]
/// - input: Input vector [dimension] quantum state |ψ⟩
/// - output: Output vector [dimension] result H|ψ⟩
/// - dimension: Matrix dimension (2^qubits)
/// - row: Thread ID (0 to dimension-1)
/// - Complexity: O(nnz/dimension) per thread, O(nnz) total
kernel void csrSparseMatVec(
    device const uint *rowPointers [[buffer(0)]],
    device const uint *columnIndices [[buffer(1)]],
    device const ComplexFloat *values [[buffer(2)]],
    device const ComplexFloat *input [[buffer(3)]],
    device ComplexFloat *output [[buffer(4)]],
    constant uint &dimension [[buffer(5)]],
    uint row [[thread_position_in_grid]]
) {
    if (row < dimension) {
        const uint start = rowPointers[row];
        const uint end = rowPointers[row + 1];

        ComplexKahanAccumulator acc = {{0.0f, 0.0f}, {0.0f, 0.0f}};

        for (uint k = start; k < end; k++) {
            const uint col = columnIndices[k];
            const ComplexFloat value = values[k];
            const ComplexFloat inputElement = input[col];
            const ComplexFloat product = complexMultiply(value, inputElement);

            complexKahanAdd(acc, product);
        }

        output[row] = complexKahanSum(acc);
    }
}
