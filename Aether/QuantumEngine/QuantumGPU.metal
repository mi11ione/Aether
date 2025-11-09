// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

/// Quantum GPU compute shaders: Metal kernels for quantum gate application
///
/// Implements massively parallel quantum gate operations on Apple GPUs using Metal.
/// Each kernel processes quantum state amplitudes in parallel across thousands of GPU threads,
/// achieving 2-10x speedup over CPU for states with ≥10 qubits (2^10 = 1024 amplitudes).
///
/// **GPU parallelization strategy**:
/// - Single-qubit gates: Each thread processes one amplitude pair (i, j) where i and j differ only in target qubit
/// - Two-qubit gates: Each thread processes one amplitude quartet (4 states differing in control/target qubits)
/// - CNOT/Toffoli: Each thread processes one amplitude, reads from input buffer, writes to output buffer
///
/// **Thread mapping**:
/// - Thread grid size: 2^n threads for n-qubit state (one per amplitude or per pair/quartet)
/// - Threadgroups: Auto-sized to GPU hardware limits (typically 256-1024 threads per group)
/// - Memory access: Coalesced reads/writes for optimal bandwidth utilization
///
/// **Complex number representation**:
/// - Float32 precision (complex amplitudes stored as (real, imaginary) pairs)
/// - Matches Swift Complex<Float> layout for zero-copy CPU-GPU transfer
/// - Operations: Complex addition, multiplication, scalar scaling
///
/// **Race condition avoidance**:
/// - Single-qubit/two-qubit: Each thread writes to unique indices (in-place update safe)
/// - CNOT/Toffoli: Separate input/output buffers prevent read-after-write hazards
///
/// **Kernel functions**:
/// 1. `applySingleQubitGate`: Parallel 2×2 matrix-vector multiplication for qubit pairs
/// 2. `applyCNOT`: Optimized controlled-NOT with conditional amplitude swap
/// 3. `applyTwoQubitGate`: General 4×4 matrix-vector for arbitrary two-qubit gates
/// 4. `applyToffoli`: Double-controlled NOT (three-qubit gate)
/// 5. `csrSparseMatVec`: CSR sparse matrix-vector multiply for Hamiltonian expectation values
///
/// Example usage (from Swift):
/// ```swift
/// // Metal shader automatically invoked by MetalGateApplication
/// let metalApp = MetalGateApplication()
/// let state = QuantumState(numQubits: 12)  // 4096 amplitudes
/// let newState = metalApp.apply(gate: .hadamard, to: [0], state: state)
/// // GPU processes 2048 amplitude pairs in parallel
/// ```
///
/// **Performance characteristics**:
/// - Latency: ~0.5-2ms per gate (includes buffer allocation + transfer)
/// - Throughput: ~10-50 GFLOPS depending on gate type and state size
/// - Speedup: 2-10x over CPU for n≥10 qubits (M1/M2/M3 chips)
/// - Memory bandwidth: ~200-400 GB/s on unified memory architecture
///
/// **Mathematical correctness**:
/// All kernels implement the same quantum mechanics as CPU GateApplication:
/// - Unitarity: U†U = I preserved through exact matrix operations
/// - Normalization: Σ|cᵢ|² = 1 maintained by unitary transformations
/// - Bit ordering: Little-endian qubit indexing (qubit 0 is LSB)

#include <metal_stdlib>
using namespace metal;

// Complex number structure matching Swift Complex<Float>
// Memory layout: 8 bytes (4 bytes real + 4 bytes imaginary)
// Alignment: Matches Swift's @frozen Complex<Float> for zero-copy transfer
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
/// Implements standard complex multiplication formula from complex analysis
inline ComplexFloat complexMultiply(ComplexFloat a, ComplexFloat b) {
    return ComplexFloat{
        a.real * b.real - a.imaginary * b.imaginary,
        a.real * b.imaginary + a.imaginary * b.real
    };
}

/// Scalar multiplication: k(a + bi) = ka + kbi
/// Used for normalization and amplitude scaling
inline ComplexFloat complexScale(ComplexFloat a, float scalar) {
    return ComplexFloat{a.real * scalar, a.imaginary * scalar};
}

// MARK: - Single-Qubit Gate Kernel

/// Apply single-qubit gate: parallel 2×2 matrix-vector multiplication
///
/// Transforms quantum state by applying 2×2 unitary matrix U to target qubit.
/// Each thread processes one amplitude pair (cᵢ, cⱼ) where i and j differ only
/// in the target qubit bit.
///
/// **Algorithm**:
/// - Thread gid processes indices i=gid (target bit=0) and j=i|(1<<target) (target bit=1)
/// - Applies matrix: [cᵢ', cⱼ'] = U · [cᵢ, cⱼ] where U = [[g00, g01], [g10, g11]]
/// - Writes results back in-place (no race conditions - unique indices per thread)
///
/// **Parallelization**:
/// - Threads: 2^n total, but only 2^(n-1) active (those with target bit = 0)
/// - Memory access: Each thread reads 2 amplitudes, writes 2 amplitudes
/// - Coalescing: Adjacent threads access nearby memory for optimal bandwidth
///
/// **Parameters**:
/// - amplitudes: Input/output state vector (2^n complex amplitudes)
/// - targetQubit: Qubit index to apply gate to (0 to n-1)
/// - gateMatrix: 2×2 unitary matrix [g00, g01, g10, g11] in row-major order
/// - numQubits: Total number of qubits (n)
/// - gid: Thread ID in grid (0 to 2^n - 1)
///
/// Example: Hadamard on qubit 0 of 2-qubit state
/// - State: [c00, c01, c10, c11]
/// - Thread 0: Processes (c00, c01) → applies H matrix
/// - Thread 2: Processes (c10, c11) → applies H matrix
/// - Threads 1,3: Inactive (target bit already set)
kernel void applySingleQubitGate(
    device ComplexFloat *amplitudes [[buffer(0)]],
    constant uint &targetQubit [[buffer(1)]],
    constant ComplexFloat *gateMatrix [[buffer(2)]],
    constant uint &numQubits [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint stateSize = 1 << numQubits;
    const uint bitMask = 1 << targetQubit;
    
    if ((gid & bitMask) == 0 && gid < stateSize) {
        const uint i = gid;
        const uint j = gid | bitMask;
        
        ComplexFloat ci = amplitudes[i];
        ComplexFloat cj = amplitudes[j];

        ComplexFloat g00 = gateMatrix[0];
        ComplexFloat g01 = gateMatrix[1];
        ComplexFloat g10 = gateMatrix[2];
        ComplexFloat g11 = gateMatrix[3];
        
        ComplexFloat newCi = complexAdd(complexMultiply(g00, ci), complexMultiply(g01, cj));
        ComplexFloat newCj = complexAdd(complexMultiply(g10, ci), complexMultiply(g11, cj));
        
        // Write back (no race conditions - each thread writes to unique indices)
        amplitudes[i] = newCi;
        amplitudes[j] = newCj;
    }
}

// MARK: - CNOT Gate Kernel

/// Apply CNOT gate: optimized controlled-NOT implementation
///
/// Implements CNOT(control→target) gate with conditional amplitude swap.
/// Much faster than general two-qubit gate matrix multiplication since
/// CNOT only swaps amplitudes when control=1, no complex arithmetic needed.
///
/// **Algorithm**:
/// - Each thread processes one amplitude at index gid
/// - If control bit is 1: write amplitude to flipped target index
/// - If control bit is 0: write amplitude unchanged
/// - Uses separate input/output buffers to avoid read-after-write hazards
///
/// **Parallelization**:
/// - Threads: 2^n (all active, one per amplitude)
/// - Memory access: Each thread reads 1 amplitude, writes 1 amplitude
/// - Write pattern: Non-coalesced (depends on qubit indices), but still fast
///
/// **Parameters**:
/// - amplitudes: Input state vector (read-only)
/// - controlQubit: Control qubit index (0 to n-1)
/// - targetQubit: Target qubit index (0 to n-1, ≠ control)
/// - numQubits: Total number of qubits (n)
/// - outputAmplitudes: Output state vector (write-only)
/// - gid: Thread ID in grid (0 to 2^n - 1)
///
/// Example: CNOT(0→1) on 2-qubit state
/// - Input: [c00, c01, c10, c11]
/// - Thread 0: control=0 → output[0] = input[0] (c00 unchanged)
/// - Thread 1: control=1 → output[3] = input[1] (c01 → c11)
/// - Thread 2: control=0 → output[2] = input[2] (c10 unchanged)
/// - Thread 3: control=1 → output[1] = input[3] (c11 → c01)
/// - Output: [c00, c11, c10, c01]
kernel void applyCNOT(
    device ComplexFloat *amplitudes [[buffer(0)]],
    constant uint &controlQubit [[buffer(1)]],
    constant uint &targetQubit [[buffer(2)]],
    constant uint &numQubits [[buffer(3)]],
    device ComplexFloat *outputAmplitudes [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint stateSize = 1 << numQubits;
    
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

/// Apply two-qubit gate: parallel 4×4 matrix-vector multiplication
///
/// Transforms quantum state by applying 4×4 unitary matrix to control/target qubits.
/// Each thread processes one amplitude quartet (c00, c01, c10, c11) where indices
/// differ only in the control and target qubit bits.
///
/// **Algorithm**:
/// - Thread gid processes indices where both control and target bits are 0
/// - Computes i00=gid, i01=gid|targetMask, i10=gid|controlMask, i11=gid|bothMask
/// - Applies 4×4 matrix: [c'00, c'01, c'10, c'11]ᵀ = U · [c00, c01, c10, c11]ᵀ
/// - Writes results back in-place (each thread owns 4 unique indices)
///
/// **Parallelization**:
/// - Threads: 2^n total, but only 2^(n-2) active (those with both bits = 0)
/// - Memory access: Each thread reads 4 amplitudes, writes 4 amplitudes
/// - Computation: 16 complex multiplications + 12 complex additions per thread
///
/// **Parameters**:
/// - amplitudes: Input/output state vector (2^n complex amplitudes)
/// - controlQubit: Control qubit index (0 to n-1)
/// - targetQubit: Target qubit index (0 to n-1, ≠ control)
/// - gateMatrix: 4×4 unitary matrix (16 elements, row-major order)
/// - numQubits: Total number of qubits (n)
/// - gid: Thread ID in grid (0 to 2^n - 1)
///
/// Example: CZ gate on qubits (0,1) of 3-qubit state
/// - Thread 0: Processes indices [0,1,2,3] (qubits 0,1 in all 4 combinations)
/// - Thread 4: Processes indices [4,5,6,7] (qubit 2=1, qubits 0,1 vary)
/// - Threads 1,2,3,5,6,7: Inactive (at least one target bit already set)
kernel void applyTwoQubitGate(
    device ComplexFloat *amplitudes [[buffer(0)]],
    constant uint &controlQubit [[buffer(1)]],
    constant uint &targetQubit [[buffer(2)]],
    constant ComplexFloat *gateMatrix [[buffer(3)]],
    constant uint &numQubits [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint stateSize = 1 << numQubits;
    const uint controlMask = 1 << controlQubit;
    const uint targetMask = 1 << targetQubit;
    const uint bothMask = controlMask | targetMask;
    
    if ((gid & bothMask) == 0 && gid < stateSize) {
        const uint i00 = gid;
        const uint i01 = gid | targetMask;
        const uint i10 = gid | controlMask;
        const uint i11 = gid | bothMask;
        
        ComplexFloat c00 = amplitudes[i00];
        ComplexFloat c01 = amplitudes[i01];
        ComplexFloat c10 = amplitudes[i10];
        ComplexFloat c11 = amplitudes[i11];
        
        // Apply 4x4 matrix
        // Matrix stored row-major: [row][col]
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
        
        // Write back
        amplitudes[i00] = new00;
        amplitudes[i01] = new01;
        amplitudes[i10] = new10;
        amplitudes[i11] = new11;
    }
}

// MARK: - Toffoli Gate Kernel

/// Apply Toffoli gate: double-controlled NOT (CCNOT)
///
/// Implements three-qubit Toffoli gate that flips target qubit if both control
/// qubits are |1⟩. Essential for reversible classical logic and quantum error
/// correction. Optimized special case avoiding 8×8 matrix multiplication.
///
/// **Algorithm**:
/// - Each thread processes one amplitude at index gid
/// - If both control bits are 1: write amplitude to flipped target index
/// - Otherwise: write amplitude unchanged (identity operation)
/// - Uses separate input/output buffers to prevent race conditions
///
/// **Parallelization**:
/// - Threads: 2^n (all active, one per amplitude)
/// - Memory access: Each thread reads 1 amplitude, writes 1 amplitude
/// - Conditional logic: Only ~25% of threads perform swap (when both controls = 1)
///
/// **Parameters**:
/// - amplitudes: Input state vector (read-only)
/// - control1Qubit: First control qubit index (0 to n-1)
/// - control2Qubit: Second control qubit index (0 to n-1, ≠ control1)
/// - targetQubit: Target qubit index (0 to n-1, ≠ control1, ≠ control2)
/// - numQubits: Total number of qubits (n)
/// - outputAmplitudes: Output state vector (write-only)
/// - gid: Thread ID in grid (0 to 2^n - 1)
///
/// Example: Toffoli(0,1→2) on 3-qubit state
/// - Input: [c000, c001, c010, c011, c100, c101, c110, c111]
/// - Thread 6 (0b110): Both controls=1, target=0 → output[7] = input[6]
/// - Thread 7 (0b111): Both controls=1, target=1 → output[6] = input[7]
/// - Other threads: Write unchanged (controls not both 1)
/// - Output: [c000, c001, c010, c011, c100, c101, c111, c110] (swap c110↔c111)
kernel void applyToffoli(
    device ComplexFloat *amplitudes [[buffer(0)]],
    constant uint &control1Qubit [[buffer(1)]],
    constant uint &control2Qubit [[buffer(2)]],
    constant uint &targetQubit [[buffer(3)]],
    constant uint &numQubits [[buffer(4)]],
    device ComplexFloat *outputAmplitudes [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint stateSize = 1 << numQubits;
    
    if (gid < stateSize) {
        const uint c1Mask = 1 << control1Qubit;
        const uint c2Mask = 1 << control2Qubit;
        const uint targetMask = 1 << targetQubit;
        
        if (((gid & c1Mask) != 0) && ((gid & c2Mask) != 0)) {
            // Both controls are 1: flip target
            const uint flipped = gid ^ targetMask;
            outputAmplitudes[flipped] = amplitudes[gid];
        } else {
            // Otherwise: identity
            outputAmplitudes[gid] = amplitudes[gid];
        }
    }
}

// MARK: - CSR Sparse Matrix-Vector Multiply Kernel

/// CSR Sparse Matrix-Vector Multiply: output = H * input (for Hamiltonian expectation values)
///
/// Implements sparse matrix-vector multiplication using CSR (Compressed Sparse Row) format
/// for quantum Hamiltonian expectation values. Each thread computes one output element
/// by accumulating the dot product of one sparse row with the input vector.
///
/// **Algorithm:**
/// - Each thread row: Computes output[row] = Σⱼ H[row,j] * input[j]
/// - CSR iteration: Loop from rowPointers[row] to rowPointers[row+1]-1
/// - For each k: Accumulate values[k] * input[columnIndices[k]]
///
/// **Parallelization:**
/// - Threads: dimension (one per matrix row)
/// - Memory access: Each thread reads O(nnz/dimension) sparse elements (average non-zeros per row)
/// - Computation: Complex multiplication + addition per non-zero element
///
/// **CSR Format:**
/// - rowPointers[i] = index in columnIndices/values where row i starts
/// - rowPointers[i+1] - rowPointers[i] = number of non-zeros in row i
/// - columnIndices[k] = column index of k-th non-zero element
/// - values[k] = complex value of k-th non-zero element
///
/// **Example:**
/// ```
/// Matrix: [[1+i, 0, 2], [0, 3, 0], [4-i, 0, 5]]
/// CSR:
///   rowPointers = [0, 2, 3, 5]
///   columnIndices = [0, 2, 1, 0, 2]
///   values = [1+i, 2+0i, 3+0i, 4-i, 5+0i]
///
/// Row 0: start=0, end=2 → H[0,0]*input[0] + H[0,2]*input[2]
/// Row 1: start=2, end=3 → H[1,1]*input[1]
/// Row 2: start=3, end=5 → H[2,0]*input[0] + H[2,2]*input[2]
/// ```
///
/// **Parameters:**
/// - rowPointers: CSR row pointers [dimension+1] - indices into columnIndices
/// - columnIndices: CSR column indices [nnz] - column index of each non-zero
/// - values: Complex values [nnz] - non-zero matrix elements
/// - input: Input vector [dimension] - quantum state |ψ⟩
/// - output: Output vector [dimension] - result H|ψ⟩ (write-only)
/// - dimension: Matrix dimension (2^numQubits)
/// - row: Thread ID (0 to dimension-1)
///
/// Example usage (from Swift):
/// ```swift
/// // Build CSR sparse Hamiltonian once
/// let sparseH = SparseHamiltonian(observable: hamiltonian)
///
/// // VQE iteration: compute ⟨ψ|H|ψ⟩ = ⟨ψ|(H|ψ⟩)
/// for iteration in 0..<100 {
///     let state = prepareAnsatz(parameters: params)
///     let energy = sparseH.expectationValue(state: state)
///     // GPU computes H|ψ⟩ in parallel, then CPU computes inner product
/// }
/// ```
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

        ComplexFloat sum = ComplexFloat{0.0, 0.0};

        for (uint k = start; k < end; k++) {
            const uint col = columnIndices[k];
            const ComplexFloat value = values[k];
            const ComplexFloat inputElement = input[col];

            sum = complexAdd(sum, complexMultiply(value, inputElement));
        }

        output[row] = sum;
    }
}
