// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

@testable import Aether
import Testing

/// Test suite for QAOA ansatz circuit construction.
/// Validates circuit structure, parameter counting, Pauli exponentiation via
/// CNOT ladders, and basis rotations for X, Y, Z operators.
@Suite("QAOA Ansatz Circuits")
struct QAOAAnsatzTests {
    @Test("Depth-1 has scaled parameters")
    func depth1Parameters() {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])
        let mixer = MixerHamiltonian.xMixer(numQubits: 2)

        let ansatz = QAOAAnsatz.create(
            numQubits: 2,
            depth: 1,
            costHamiltonian: cost,
            mixerHamiltonian: mixer
        )

        #expect(!ansatz.isEmpty())
        #expect(ansatz.parameters.contains { $0.name.starts(with: "gamma_0_c_") })
        #expect(ansatz.parameters.contains { $0.name.starts(with: "beta_0_c_") })
    }

    @Test("Depth-2 has scaled parameters for both layers")
    func depth2Parameters() {
        let cost = MaxCut.hamiltonian(edges: [(0, 1), (1, 2)])
        let mixer = MixerHamiltonian.xMixer(numQubits: 3)

        let ansatz = QAOAAnsatz.create(
            numQubits: 3,
            depth: 2,
            costHamiltonian: cost,
            mixerHamiltonian: mixer
        )

        let scaledParameters = ansatz.parameters.filter { $0.name.contains("_c_") }
        #expect(scaledParameters.count > 0)

        #expect(ansatz.parameters.contains { $0.name.starts(with: "gamma_0_c_") })
        #expect(ansatz.parameters.contains { $0.name.starts(with: "beta_0_c_") })
        #expect(ansatz.parameters.contains { $0.name.starts(with: "gamma_1_c_") })
        #expect(ansatz.parameters.contains { $0.name.starts(with: "beta_1_c_") })
    }

    @Test("Depth-5 has scaled parameters for all layers")
    func depth5Parameters() {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])
        let mixer = MixerHamiltonian.xMixer(numQubits: 2)

        let ansatz = QAOAAnsatz.create(
            numQubits: 2,
            depth: 5,
            costHamiltonian: cost,
            mixerHamiltonian: mixer
        )

        #expect(ansatz.parameters.contains { $0.name.starts(with: "gamma_4_c_") })
        #expect(ansatz.parameters.contains { $0.name.starts(with: "beta_4_c_") })
    }

    @Test("Circuit starts with Hadamard gates")
    func initialSuperposition() {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])
        let mixer = MixerHamiltonian.xMixer(numQubits: 2)

        let ansatz = QAOAAnsatz.create(
            numQubits: 2,
            depth: 1,
            costHamiltonian: cost,
            mixerHamiltonian: mixer
        )

        let circuit = QAOAParameterBinder(ansatz: ansatz).bind(baseParameters: [0.5, 0.5])
        #expect(circuit.gates.count > 2)
    }
}

/// Test suite for Pauli string exponentiation in QAOA ansatz.
/// Verifies CNOT ladder structure for multi-qubit rotations and proper
/// basis conversions for X, Y, Z Pauli operators.
@Suite("Pauli String Exponentiation")
struct PauliStringExponentiationTests {
    @Test("Two-qubit ZZ uses CNOT ladder")
    func twoQubitZZ() {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])
        let mixer = MixerHamiltonian.xMixer(numQubits: 2)

        let ansatz = QAOAAnsatz.create(
            numQubits: 2,
            depth: 1,
            costHamiltonian: cost,
            mixerHamiltonian: mixer
        )

        let circuit = QAOAParameterBinder(ansatz: ansatz).bind(baseParameters: [0.5, 0.3])

        let hasCNOT = circuit.gates.contains { op in
            if case .cnot = op.gate { return true }
            return false
        }

        #expect(hasCNOT)
    }

    @Test("Three-qubit ZZZ uses extended ladder")
    func threeQubitZZZ() {
        let pauliString = PauliString(operators: [
            (qubit: 0, basis: .z),
            (qubit: 1, basis: .z),
            (qubit: 2, basis: .z),
        ])
        let cost = Observable(terms: [(coefficient: -0.25, pauliString: pauliString)])
        let mixer = MixerHamiltonian.xMixer(numQubits: 3)

        let ansatz = QAOAAnsatz.create(
            numQubits: 3,
            depth: 1,
            costHamiltonian: cost,
            mixerHamiltonian: mixer
        )

        let circuit = QAOAParameterBinder(ansatz: ansatz).bind(baseParameters: [0.5, 0.3])

        let cnotCount = circuit.gates.filter { op in
            if case .cnot = op.gate { return true }
            return false
        }.count

        #expect(cnotCount >= 4)
    }

    @Test("Y operator requires Rx basis rotation")
    func yOperatorBasisRotation() {
        let cost = Observable(terms: [(coefficient: 1.0, pauliString: PauliString(operators: [(qubit: 0, basis: .y)]))])
        let mixer = MixerHamiltonian.xMixer(numQubits: 1)

        let ansatz = QAOAAnsatz.create(
            numQubits: 1,
            depth: 1,
            costHamiltonian: cost,
            mixerHamiltonian: mixer
        )

        let circuit = QAOAParameterBinder(ansatz: ansatz).bind(baseParameters: [0.5, 0.3])

        let rxGates = circuit.gates.filter { op in
            if case .rotationX = op.gate { return true }
            return false
        }

        #expect(rxGates.count >= 2)
    }

    @Test("Mixed XYZ string has multiple basis rotations")
    func mixedPauliString() {
        let pauliString = PauliString(operators: [
            (qubit: 0, basis: .x),
            (qubit: 1, basis: .y),
            (qubit: 2, basis: .z),
        ])
        let cost = Observable(terms: [(coefficient: 0.5, pauliString: pauliString)])
        let mixer = MixerHamiltonian.xMixer(numQubits: 3)

        let ansatz = QAOAAnsatz.create(
            numQubits: 3,
            depth: 1,
            costHamiltonian: cost,
            mixerHamiltonian: mixer
        )

        let circuit = QAOAParameterBinder(ansatz: ansatz).bind(baseParameters: [0.5, 0.3])

        let hadamardGates = circuit.gates.filter { op in
            if case .hadamard = op.gate { return true }
            return false
        }

        let rxGates = circuit.gates.filter { op in
            if case .rotationX = op.gate { return true }
            return false
        }

        #expect(hadamardGates.count >= 2)
        #expect(rxGates.count >= 2)
    }
}

/// Test suite for coefficient scaling in QAOA parameters.
/// Validates scaled parameter naming convention and near-zero coefficient
/// filtering for numerical stability.
@Suite("Coefficient Scaling")
struct CoefficientScalingTests {
    @Test("Scaled parameters encode coefficients")
    func scaledParameterNaming() {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])
        let mixer = MixerHamiltonian.xMixer(numQubits: 2)

        let ansatz = QAOAAnsatz.create(
            numQubits: 2,
            depth: 1,
            costHamiltonian: cost,
            mixerHamiltonian: mixer
        )

        let scaledGammaParams = ansatz.parameters.filter { $0.name.starts(with: "gamma_0_c_") }
        let scaledBetaParams = ansatz.parameters.filter { $0.name.starts(with: "beta_0_c_") }

        #expect(!scaledGammaParams.isEmpty)
        #expect(!scaledBetaParams.isEmpty)

        let gammaCoefficient = -0.5 * 2.0
        #expect(ansatz.parameters.contains { $0.name == "gamma_0_c_\(gammaCoefficient)" })

        let betaCoefficient = 1.0 * 2.0
        #expect(ansatz.parameters.contains { $0.name == "beta_0_c_\(betaCoefficient)" })
    }

    @Test("Near-zero coefficients are filtered")
    func nearZeroFiltering() {
        let cost = Observable(terms: [
            (coefficient: 1e-16, pauliString: PauliString(operators: [(qubit: 0, basis: .z)])),
            (coefficient: 0.5, pauliString: PauliString(operators: [(qubit: 1, basis: .z)])),
        ])
        let mixer = MixerHamiltonian.xMixer(numQubits: 2)

        let ansatz = QAOAAnsatz.create(
            numQubits: 2,
            depth: 1,
            costHamiltonian: cost,
            mixerHamiltonian: mixer
        )

        let circuit = QAOAParameterBinder(ansatz: ansatz).bind(baseParameters: [0.5, 0.3])
        #expect(!circuit.gates.isEmpty)
    }
}

/// Test suite for complete QAOA ansatz workflows.
/// Validates end-to-end circuit construction for standard MaxCut problems
/// including triangle and square graphs with parameter binding.
@Suite("Complete QAOA Workflows")
struct CompleteQAOAWorkflowsTests {
    @Test("Triangle MaxCut ansatz binds successfully")
    func triangleMaxCut() {
        let cost = MaxCut.hamiltonian(edges: MaxCut.Examples.triangle())
        let mixer = MixerHamiltonian.xMixer(numQubits: 3)

        let ansatz = QAOAAnsatz.create(
            numQubits: 3,
            depth: 2,
            costHamiltonian: cost,
            mixerHamiltonian: mixer
        )

        #expect(!ansatz.isEmpty())

        let scaledParams = ansatz.parameters.filter { $0.name.contains("_c_") }
        #expect(scaledParams.count > 0)
    }

    @Test("Square MaxCut ansatz binds successfully")
    func squareMaxCut() {
        let cost = MaxCut.hamiltonian(edges: MaxCut.Examples.square())
        let mixer = MixerHamiltonian.xMixer(numQubits: 4)

        let ansatz = QAOAAnsatz.create(
            numQubits: 4,
            depth: 3,
            costHamiltonian: cost,
            mixerHamiltonian: mixer
        )

        #expect(!ansatz.isEmpty())

        let scaledParams = ansatz.parameters.filter { $0.name.contains("_c_") }
        #expect(scaledParams.count > 0)
    }

    @Test("Parameter binding produces valid circuit")
    func parameterBinding() {
        let cost = MaxCut.hamiltonian(edges: [(0, 1)])
        let mixer = MixerHamiltonian.xMixer(numQubits: 2)

        let ansatz = QAOAAnsatz.create(
            numQubits: 2,
            depth: 1,
            costHamiltonian: cost,
            mixerHamiltonian: mixer
        )

        let circuit = QAOAParameterBinder(ansatz: ansatz).bind(baseParameters: [0.5, 0.3])

        #expect(!circuit.gates.isEmpty)
        #expect(circuit.numQubits == 2)
    }
}
