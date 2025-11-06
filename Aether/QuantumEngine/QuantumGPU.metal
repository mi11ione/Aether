#include <metal_stdlib>
using namespace metal;

// Complex number structure matching Swift Complex<Float>
struct ComplexFloat {
    float real;
    float imaginary;
};

// Complex number operations (free functions to avoid address space issues)
inline ComplexFloat complexAdd(ComplexFloat a, ComplexFloat b) {
    return ComplexFloat{a.real + b.real, a.imaginary + b.imaginary};
}

inline ComplexFloat complexMultiply(ComplexFloat a, ComplexFloat b) {
    return ComplexFloat{
        a.real * b.real - a.imaginary * b.imaginary,
        a.real * b.imaginary + a.imaginary * b.real
    };
}

inline ComplexFloat complexScale(ComplexFloat a, float scalar) {
    return ComplexFloat{a.real * scalar, a.imaginary * scalar};
}

// Apply single-qubit gate to quantum state
// Each thread handles one amplitude pair
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

// CNOT gate application
// Faster than general two-qubit gate
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

// Apply two-qubit gate (general 4x4 matrix)
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

// Toffoli gate (CCNOT)
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
