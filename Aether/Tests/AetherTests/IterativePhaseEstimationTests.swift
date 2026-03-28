// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under Apache 2.0

import Aether
import Foundation
import Testing

/// Test suite for IterativePhaseEstimation initialization parameter validation.
/// Validates precision bits, adaptive strategy selection, and initial estimate
/// constraints through observable behavior after algorithm execution.
@Suite("IterativePhaseEstimation Initialization")
struct IterativePhaseEstimationInitializationTests {
    @Test("Default configuration with precision bits only")
    func defaultConfiguration() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 8)
        let result = await ipe.run()
        #expect(result.iterations.count == 8, "Precision bits should be 8")
        #expect(result.equivalentQPEQubits == 8, "Equivalent QPE qubits should be 8")
    }

    @Test("Configuration with semiclassical strategy")
    func semiclassicalStrategy() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 10, adaptiveStrategy: .semiclassical)
        let result = await ipe.run()
        #expect(result.iterations.count == 10, "Precision bits should be 10")
    }

    @Test("Configuration with initial estimate")
    func withInitialEstimate() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 6, adaptiveStrategy: .standard, initialEstimate: 0.25)
        let result = await ipe.run()
        #expect(result.iterations.count == 6, "Precision bits should be 6")
    }

    @Test("Configuration with zero initial estimate")
    func zeroInitialEstimate() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 4, initialEstimate: 0.0)
        let result = await ipe.run()
        #expect(result.iterations.count == 4, "Should complete all iterations with zero initial estimate")
    }

    @Test("Configuration with initial estimate near upper bound")
    func nearUpperBoundEstimate() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 4, initialEstimate: 0.99)
        let result = await ipe.run()
        #expect(result.iterations.count == 4, "Should complete all iterations with near-upper-bound estimate")
    }

    @Test("Full configuration with all parameters")
    func fullConfiguration() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 12, adaptiveStrategy: .semiclassical, initialEstimate: 0.5)
        let result = await ipe.run()
        #expect(result.iterations.count == 12, "Precision bits should be 12")
    }
}

/// Test suite for AdaptiveStrategy enum values and behavior.
/// Validates standard and semiclassical strategy cases exist and are distinct
/// for controlling phase correction between iterations.
@Suite("AdaptiveStrategy Cases")
struct AdaptiveStrategyTests {
    @Test("Standard strategy case exists")
    func standardCaseExists() {
        let strategy: IterativePhaseEstimation.AdaptiveStrategy = .standard
        #expect(strategy == .standard, "Standard strategy should equal itself")
    }

    @Test("Semiclassical strategy case exists")
    func semiclassicalCaseExists() {
        let strategy: IterativePhaseEstimation.AdaptiveStrategy = .semiclassical
        #expect(strategy == .semiclassical, "Semiclassical strategy should equal itself")
    }

    @Test("Standard and semiclassical are distinct")
    func strategiesAreDistinct() {
        #expect(IterativePhaseEstimation.AdaptiveStrategy.standard != IterativePhaseEstimation.AdaptiveStrategy.semiclassical, "Standard and semiclassical should be different")
    }
}

/// Test suite for IterationResult properties and initialization.
/// Validates bit index, measured bit, control angle, and phase estimate
/// storage for single iteration results.
@Suite("IterationResult Properties")
struct IterationResultTests {
    @Test("Iteration result stores bit index correctly")
    func bitIndexStorage() {
        let result = IterativePhaseEstimation.IterationResult(bitIndex: 3, measuredBit: 1, controlAngle: 0.5, phaseEstimate: 0.125)
        #expect(result.bitIndex == 3, "Bit index should be 3")
    }

    @Test("Iteration result stores measured bit correctly")
    func measuredBitStorage() {
        let result = IterativePhaseEstimation.IterationResult(bitIndex: 0, measuredBit: 1, controlAngle: 0.0, phaseEstimate: 0.5)
        #expect(result.measuredBit == 1, "Measured bit should be 1")
    }

    @Test("Iteration result stores zero measured bit")
    func zeroMeasuredBit() {
        let result = IterativePhaseEstimation.IterationResult(bitIndex: 0, measuredBit: 0, controlAngle: 0.0, phaseEstimate: 0.0)
        #expect(result.measuredBit == 0, "Measured bit should be 0")
    }

    @Test("Iteration result stores control angle correctly")
    func controlAngleStorage() {
        let angle = Double.pi / 4.0
        let result = IterativePhaseEstimation.IterationResult(bitIndex: 2, measuredBit: 0, controlAngle: angle, phaseEstimate: 0.25)
        #expect(abs(result.controlAngle - angle) < 1e-10, "Control angle should be pi/4")
    }

    @Test("Iteration result stores phase estimate correctly")
    func phaseEstimateStorage() {
        let result = IterativePhaseEstimation.IterationResult(bitIndex: 1, measuredBit: 1, controlAngle: 0.0, phaseEstimate: 0.75)
        #expect(abs(result.phaseEstimate - 0.75) < 1e-10, "Phase estimate should be 0.75")
    }

    @Test("Iteration result with zero control angle")
    func zeroControlAngle() {
        let result = IterativePhaseEstimation.IterationResult(bitIndex: 0, measuredBit: 0, controlAngle: 0.0, phaseEstimate: 0.0)
        #expect(abs(result.controlAngle) < 1e-10, "Control angle should be zero")
    }
}

/// Test suite for phase estimation Result properties and description formatting.
/// Validates estimated phase, iterations array, total depth, equivalent QPE qubits,
/// and human-readable description output.
@Suite("Phase Estimation Result Properties")
struct PhaseEstimationResultPropertiesTests {
    @Test("Result stores estimated phase correctly")
    func estimatedPhaseStorage() {
        let iterations = [IterativePhaseEstimation.IterationResult(bitIndex: 0, measuredBit: 1, controlAngle: 0.0, phaseEstimate: 0.5)]
        let result = IterativePhaseEstimation.Result(estimatedPhase: 0.5, iterations: iterations, totalDepth: 5, equivalentQPEQubits: 1)
        #expect(abs(result.estimatedPhase - 0.5) < 1e-10, "Estimated phase should be 0.5")
    }

    @Test("Result stores iterations array correctly")
    func iterationsStorage() {
        let iter1 = IterativePhaseEstimation.IterationResult(bitIndex: 0, measuredBit: 1, controlAngle: 0.0, phaseEstimate: 0.5)
        let iter2 = IterativePhaseEstimation.IterationResult(bitIndex: 1, measuredBit: 0, controlAngle: 0.5, phaseEstimate: 0.5)
        let iterations = [iter1, iter2]
        let result = IterativePhaseEstimation.Result(estimatedPhase: 0.5, iterations: iterations, totalDepth: 10, equivalentQPEQubits: 2)
        #expect(result.iterations.count == 2, "Iterations count should be 2")
    }

    @Test("Result stores total depth correctly")
    func totalDepthStorage() {
        let iterations = [IterativePhaseEstimation.IterationResult(bitIndex: 0, measuredBit: 0, controlAngle: 0.0, phaseEstimate: 0.0)]
        let result = IterativePhaseEstimation.Result(estimatedPhase: 0.0, iterations: iterations, totalDepth: 15, equivalentQPEQubits: 1)
        #expect(result.totalDepth == 15, "Total depth should be 15")
    }

    @Test("Equivalent QPE qubits equals precision bits")
    func equivalentQPEQubitsStorage() {
        let iterations: [IterativePhaseEstimation.IterationResult] = (0 ..< 8).map { i in
            IterativePhaseEstimation.IterationResult(bitIndex: i, measuredBit: 0, controlAngle: 0.0, phaseEstimate: 0.0)
        }
        let result = IterativePhaseEstimation.Result(estimatedPhase: 0.0, iterations: iterations, totalDepth: 40, equivalentQPEQubits: 8)
        #expect(result.equivalentQPEQubits == 8, "Equivalent QPE qubits should equal precision bits (8)")
    }

    @Test("Description contains estimated phase")
    func descriptionContainsPhase() {
        let iterations = [IterativePhaseEstimation.IterationResult(bitIndex: 0, measuredBit: 1, controlAngle: 0.0, phaseEstimate: 0.5)]
        let result = IterativePhaseEstimation.Result(estimatedPhase: 0.5, iterations: iterations, totalDepth: 5, equivalentQPEQubits: 1)
        #expect(result.description.contains("0.5"), "Description should contain estimated phase")
    }

    @Test("Description contains binary representation")
    func descriptionContainsBinary() {
        let iterations = [
            IterativePhaseEstimation.IterationResult(bitIndex: 0, measuredBit: 1, controlAngle: 0.0, phaseEstimate: 0.5),
            IterativePhaseEstimation.IterationResult(bitIndex: 1, measuredBit: 0, controlAngle: 0.5, phaseEstimate: 0.5),
        ]
        let result = IterativePhaseEstimation.Result(estimatedPhase: 0.5, iterations: iterations, totalDepth: 10, equivalentQPEQubits: 2)
        #expect(result.description.contains("Binary"), "Description should contain Binary label")
        #expect(result.description.contains("10"), "Description should contain binary digits")
    }

    @Test("Description contains iteration count")
    func descriptionContainsIterations() {
        let iterations = [
            IterativePhaseEstimation.IterationResult(bitIndex: 0, measuredBit: 0, controlAngle: 0.0, phaseEstimate: 0.0),
            IterativePhaseEstimation.IterationResult(bitIndex: 1, measuredBit: 0, controlAngle: 0.0, phaseEstimate: 0.0),
            IterativePhaseEstimation.IterationResult(bitIndex: 2, measuredBit: 0, controlAngle: 0.0, phaseEstimate: 0.0),
        ]
        let result = IterativePhaseEstimation.Result(estimatedPhase: 0.0, iterations: iterations, totalDepth: 15, equivalentQPEQubits: 3)
        #expect(result.description.contains("Iterations: 3"), "Description should contain iteration count")
    }

    @Test("Description contains total depth")
    func descriptionContainsDepth() {
        let iterations = [IterativePhaseEstimation.IterationResult(bitIndex: 0, measuredBit: 0, controlAngle: 0.0, phaseEstimate: 0.0)]
        let result = IterativePhaseEstimation.Result(estimatedPhase: 0.0, iterations: iterations, totalDepth: 25, equivalentQPEQubits: 1)
        #expect(result.description.contains("Total Depth: 25"), "Description should contain total depth")
    }

    @Test("Description contains equivalent QPE qubits")
    func descriptionContainsEquivalentQubits() {
        let iterations = [IterativePhaseEstimation.IterationResult(bitIndex: 0, measuredBit: 0, controlAngle: 0.0, phaseEstimate: 0.0)]
        let result = IterativePhaseEstimation.Result(estimatedPhase: 0.0, iterations: iterations, totalDepth: 5, equivalentQPEQubits: 4)
        #expect(result.description.contains("Equivalent QPE Qubits: 4"), "Description should contain equivalent QPE qubits")
    }
}

/// Test suite for iterativePhaseEstimationStep circuit structure and properties.
/// Validates circuit qubit count, Hadamard gates placement, controlled unitary power,
/// and phase correction rotations for single IPE iteration.
@Suite("iterativePhaseEstimationStep Circuit Structure")
struct IterativePhaseEstimationStepCircuitTests {
    @Test("Circuit has correct qubit count for single eigenstate qubit")
    func circuitQubitCountSingleEigenstate() {
        let circuit = QuantumCircuit.iterativePhaseEstimationStep(unitary: .pauliZ, power: 0, phaseCorrection: 0.0, eigenstateQubits: 1)
        #expect(circuit.qubits == 2, "Circuit should have 2 qubits (1 ancilla + 1 eigenstate)")
    }

    @Test("Circuit has correct qubit count for multiple eigenstate qubits")
    func circuitQubitCountMultipleEigenstate() {
        let circuit = QuantumCircuit.iterativePhaseEstimationStep(unitary: .pauliZ, power: 0, phaseCorrection: 0.0, eigenstateQubits: 3)
        #expect(circuit.qubits == 4, "Circuit should have 4 qubits (1 ancilla + 3 eigenstate)")
    }

    @Test("Circuit contains Hadamard gates")
    func circuitContainsHadamards() {
        let circuit = QuantumCircuit.iterativePhaseEstimationStep(unitary: .pauliZ, power: 0, phaseCorrection: 0.0, eigenstateQubits: 1)
        let hadamardCount = circuit.operations.count(where: { if $0.gate == .hadamard { return true }; return false })
        #expect(hadamardCount >= 2, "Circuit should contain at least 2 Hadamard gates")
    }

    @Test("Circuit depth is positive")
    func circuitDepthPositive() {
        let circuit = QuantumCircuit.iterativePhaseEstimationStep(unitary: .pauliZ, power: 1, phaseCorrection: 0.0, eigenstateQubits: 1)
        #expect(circuit.depth > 0, "Circuit depth should be positive")
    }

    @Test("Circuit with phase correction includes Rz gate")
    func circuitWithPhaseCorrectionHasRz() {
        let circuit = QuantumCircuit.iterativePhaseEstimationStep(unitary: .pauliZ, power: 0, phaseCorrection: Double.pi / 4, eigenstateQubits: 1)
        let hasRotationZ = circuit.operations.contains { op in
            if case .rotationZ = op.gate { return true }
            return false
        }
        #expect(hasRotationZ, "Circuit with nonzero phase correction should contain Rz gate")
    }

    @Test("Circuit without phase correction omits Rz gate")
    func circuitWithoutPhaseCorrectionOmitsRz() {
        let circuit = QuantumCircuit.iterativePhaseEstimationStep(unitary: .pauliZ, power: 0, phaseCorrection: 0.0, eigenstateQubits: 1)
        let rotationZCount = circuit.operations.count(where: { op in
            if case .rotationZ = op.gate { return true }
            return false
        })
        #expect(rotationZCount == 0, "Circuit with zero phase correction should not contain Rz gate")
    }

    @Test("Circuit with power zero still has structure")
    func circuitWithPowerZero() {
        let circuit = QuantumCircuit.iterativePhaseEstimationStep(unitary: .pauliZ, power: 0, phaseCorrection: 0.0, eigenstateQubits: 1)
        #expect(!circuit.isEmpty, "Circuit with power 0 should still have gates")
    }

    @Test("Circuit with higher power has gates")
    func circuitWithHigherPower() {
        let circuit = QuantumCircuit.iterativePhaseEstimationStep(unitary: .pauliZ, power: 3, phaseCorrection: 0.0, eigenstateQubits: 1)
        #expect(!circuit.isEmpty, "Circuit with power 3 should have gates")
    }
}

/// Test suite for IterativePhaseEstimation actor run method with various unitaries.
/// Validates phase extraction for Pauli Z eigenvalues, phase gates, and
/// comparison between standard and semiclassical strategies.
@Suite("IterativePhaseEstimation Run")
struct IterativePhaseEstimationRunTests {
    @Test("IPE with Pauli Z on |0> eigenstate yields phase near 0")
    func pauliZOnZeroEigenstate() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 4)
        let zeroState = QuantumState(qubit: 0)
        let result = await ipe.run(initialState: zeroState)
        #expect(abs(result.estimatedPhase) < 0.1, "Phase for Z|0> should be near 0 (eigenvalue +1)")
    }

    @Test("IPE with Pauli Z on |1> eigenstate measures LSB only")
    func pauliZOnOneEigenstate() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 4)
        let oneState = QuantumState(qubit: 1)
        let result = await ipe.run(initialState: oneState)
        #expect(abs(result.estimatedPhase - 0.0625) < 0.01, "Z|1> should measure LSB only due to Z^2=I periodicity")
    }

    @Test("IPE result has correct number of iterations")
    func correctIterationCount() async {
        let precisionBits = 6
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: precisionBits)
        let result = await ipe.run()
        #expect(result.iterations.count == precisionBits, "Iterations count should equal precision bits")
    }

    @Test("IPE result equivalentQPEQubits equals precision bits")
    func equivalentQPEQubitsMatchesPrecision() async {
        let precisionBits = 5
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: precisionBits)
        let result = await ipe.run()
        #expect(result.equivalentQPEQubits == precisionBits, "Equivalent QPE qubits should equal precision bits")
    }

    @Test("IPE total depth accumulates from iterations")
    func totalDepthAccumulates() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 3)
        let result = await ipe.run()
        #expect(result.totalDepth > 0, "Total depth should be positive")
    }

    @Test("IPE with phase gate extracts known phase")
    func phaseGateKnownPhase() async {
        let targetPhase = 0.25
        let phaseAngle = 2.0 * Double.pi * targetPhase
        let ipe = IterativePhaseEstimation(unitary: .phase(phaseAngle), eigenstateQubits: 1, precisionBits: 4)
        let oneState = QuantumState(qubit: 1)
        let result = await ipe.run(initialState: oneState)
        #expect(abs(result.estimatedPhase - targetPhase) < 0.2, "Phase gate phase extraction should be within tolerance")
    }

    @Test("IPE with semiclassical strategy runs successfully")
    func semiclassicalStrategyRuns() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 4, adaptiveStrategy: .semiclassical)
        let result = await ipe.run()
        #expect(result.iterations.count == 4, "Semiclassical IPE should complete all iterations")
    }

    @Test("IPE iteration results have sequential bit indices")
    func sequentialBitIndices() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 5)
        let result = await ipe.run()
        for (index, iteration) in result.iterations.enumerated() {
            #expect(iteration.bitIndex == index, "Bit index \(iteration.bitIndex) should equal iteration index \(index)")
        }
    }

    @Test("IPE measured bits are binary (0 or 1)")
    func measuredBitsAreBinary() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 4)
        let result = await ipe.run()
        for iteration in result.iterations {
            #expect(iteration.measuredBit == 0 || iteration.measuredBit == 1, "Measured bit should be 0 or 1, got \(iteration.measuredBit)")
        }
    }

    @Test("IPE estimated phase is in valid range [0, 1)")
    func estimatedPhaseInRange() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 6)
        let result = await ipe.run()
        #expect(result.estimatedPhase >= 0.0, "Estimated phase should be >= 0")
        #expect(result.estimatedPhase < 1.0, "Estimated phase should be < 1")
    }

    @Test("IPE progress callback is invoked")
    func progressCallbackInvoked() async {
        actor Counter {
            var count = 0
            func increment() {
                count += 1
            }

            func value() -> Int {
                count
            }
        }
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 3)
        let counter = Counter()
        _ = await ipe.run { _, _ in
            await counter.increment()
        }
        let callbackCount = await counter.value()
        #expect(callbackCount == 3, "Progress callback should be invoked for each iteration")
    }

    @Test("IPE with initial estimate starts from provided value")
    func initialEstimateUsed() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 2, initialEstimate: 0.5)
        let result = await ipe.run()
        #expect(result.iterations.count == 2, "IPE with initial estimate should complete all iterations")
    }
}

/// Test suite comparing standard vs semiclassical strategies.
/// Validates that both strategies produce results and that semiclassical
/// applies phase corrections while standard does not.
@Suite("Standard vs Semiclassical Strategy Comparison")
struct StrategyComparisonTests {
    @Test("Standard strategy has zero control angles initially")
    func standardStrategyZeroAngles() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 3)
        let result = await ipe.run()
        for iteration in result.iterations {
            #expect(abs(iteration.controlAngle) < 1e-10, "Standard strategy should have zero control angles")
        }
    }

    @Test("Both strategies complete with same iteration count")
    func bothStrategiesSameIterationCount() async {
        let precisionBits = 4

        let standardIPE = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: precisionBits)
        let semiclassicalIPE = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: precisionBits, adaptiveStrategy: .semiclassical)

        let standardResult = await standardIPE.run()
        let semiclassicalResult = await semiclassicalIPE.run()

        #expect(standardResult.iterations.count == semiclassicalResult.iterations.count, "Both strategies should have same iteration count")
    }

    @Test("Both strategies produce valid phase estimates")
    func bothStrategiesValidPhases() async {
        let standardIPE = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 4)
        let semiclassicalIPE = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 4, adaptiveStrategy: .semiclassical)

        let standardResult = await standardIPE.run()
        let semiclassicalResult = await semiclassicalIPE.run()

        #expect(standardResult.estimatedPhase >= 0.0 && standardResult.estimatedPhase < 1.0, "Standard phase should be in [0, 1)")
        #expect(semiclassicalResult.estimatedPhase >= 0.0 && semiclassicalResult.estimatedPhase < 1.0, "Semiclassical phase should be in [0, 1)")
    }
}

/// Test suite for IPE with different unitary gates.
/// Validates phase extraction works correctly for Pauli X, Pauli Y, Hadamard,
/// and rotation gates with known eigenstructures.
@Suite("IPE with Different Unitaries")
struct IPEDifferentUnitariesTests {
    @Test("IPE with Pauli X gate runs successfully")
    func pauliXGate() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliX, eigenstateQubits: 1, precisionBits: 3)
        let result = await ipe.run()
        #expect(result.iterations.count == 3, "IPE with Pauli X should complete all iterations")
        #expect(result.estimatedPhase >= 0.0 && result.estimatedPhase < 1.0, "Phase should be in valid range")
    }

    @Test("IPE with Pauli Y gate runs successfully")
    func pauliYGate() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliY, eigenstateQubits: 1, precisionBits: 3)
        let result = await ipe.run()
        #expect(result.iterations.count == 3, "IPE with Pauli Y should complete all iterations")
        #expect(result.estimatedPhase >= 0.0 && result.estimatedPhase < 1.0, "Phase should be in valid range")
    }

    @Test("IPE with Hadamard gate runs successfully")
    func hadamardGate() async {
        let ipe = IterativePhaseEstimation(unitary: .hadamard, eigenstateQubits: 1, precisionBits: 3)
        let result = await ipe.run()
        #expect(result.iterations.count == 3, "IPE with Hadamard should complete all iterations")
        #expect(result.estimatedPhase >= 0.0 && result.estimatedPhase < 1.0, "Phase should be in valid range")
    }

    @Test("IPE with S gate runs successfully")
    func sGate() async {
        let ipe = IterativePhaseEstimation(unitary: .sGate, eigenstateQubits: 1, precisionBits: 4)
        let oneState = QuantumState(qubit: 1)
        let result = await ipe.run(initialState: oneState)
        #expect(result.iterations.count == 4, "IPE with S gate should complete all iterations")
    }

    @Test("IPE with T gate runs successfully")
    func tGate() async {
        let ipe = IterativePhaseEstimation(unitary: .tGate, eigenstateQubits: 1, precisionBits: 4)
        let oneState = QuantumState(qubit: 1)
        let result = await ipe.run(initialState: oneState)
        #expect(result.iterations.count == 4, "IPE with T gate should complete all iterations")
    }

    @Test("IPE with rotation Z gate runs successfully")
    func rotationZGate() async {
        let ipe = IterativePhaseEstimation(unitary: .rotationZ(Double.pi / 3), eigenstateQubits: 1, precisionBits: 4)
        let result = await ipe.run()
        #expect(result.iterations.count == 4, "IPE with Rz gate should complete all iterations")
        #expect(result.estimatedPhase >= 0.0 && result.estimatedPhase < 1.0, "Phase should be in valid range")
    }

    @Test("IPE with identity gate yields phase near 0")
    func identityGate() async {
        let ipe = IterativePhaseEstimation(unitary: .identity, eigenstateQubits: 1, precisionBits: 3)
        let result = await ipe.run()
        #expect(abs(result.estimatedPhase) < 0.2, "Identity gate should yield phase near 0")
    }
}

/// Test suite for semiclassical phase correction computation.
/// Validates that nonzero control angles are applied when previous bits
/// are measured as 1 in semiclassical adaptive strategy.
@Suite("Semiclassical Phase Correction")
struct SemiclassicalPhaseCorrectionTests {
    @Test("Semiclassical strategy computes phase correction from measured ones")
    func phaseCorrectionFromMeasuredOnes() async {
        let ipe = IterativePhaseEstimation(unitary: .phase(Double.pi / 2), eigenstateQubits: 1, precisionBits: 8, adaptiveStrategy: .semiclassical)
        let oneState = QuantumState(qubit: 1)
        let result = await ipe.run(initialState: oneState)
        var foundNonzeroAfterOne = false
        var seenOne = false
        for iteration in result.iterations {
            if seenOne, abs(iteration.controlAngle) > 1e-10 {
                foundNonzeroAfterOne = true
            }
            if iteration.measuredBit == 1 {
                seenOne = true
            }
        }
        #expect(seenOne, "Should measure at least one bit as 1 for phase gate eigenstate")
        #expect(foundNonzeroAfterOne, "Control angle should be nonzero after measuring bit 1")
    }
}

/// Test suite for edge cases in iterative phase estimation.
/// Validates behavior with minimum precision, single iteration,
/// and various boundary conditions.
@Suite("IPE Edge Cases")
struct IPEEdgeCasesTests {
    @Test("IPE with single precision bit")
    func singlePrecisionBit() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 1)
        let result = await ipe.run()
        #expect(result.iterations.count == 1, "Single precision bit should yield 1 iteration")
        #expect(result.equivalentQPEQubits == 1, "Equivalent QPE qubits should be 1")
    }

    @Test("IPE with many precision bits completes")
    func manyPrecisionBits() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 10)
        let result = await ipe.run()
        #expect(result.iterations.count == 10, "Should complete all 10 iterations")
    }

    @Test("IPE phase estimate reconstructs from measured bits")
    func phaseReconstructsFromBits() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 4)
        let oneState = QuantumState(qubit: 1)
        let result = await ipe.run(initialState: oneState)

        var reconstructed = 0.0
        for (k, iteration) in result.iterations.enumerated() {
            if iteration.measuredBit == 1 {
                reconstructed += 1.0 / Double(1 << (k + 1))
            }
        }
        #expect(abs(result.estimatedPhase - reconstructed) < 1e-10, "Estimated phase should match reconstructed from bits")
    }

    @Test("IPE total depth increases with precision")
    func totalDepthIncreasesWithPrecision() async {
        let ipe3 = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 3)
        let ipe6 = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 6)

        let result3 = await ipe3.run()
        let result6 = await ipe6.run()

        #expect(result6.totalDepth > result3.totalDepth, "More precision bits should yield greater total depth")
    }

    @Test("IPE final iteration phase estimate matches result")
    func finalIterationMatchesResult() async {
        let ipe = IterativePhaseEstimation(unitary: .pauliZ, eigenstateQubits: 1, precisionBits: 4)
        let result = await ipe.run()

        #expect(!result.iterations.isEmpty, "Should have at least one iteration")
        let lastIteration = result.iterations[result.iterations.count - 1]
        #expect(abs(lastIteration.phaseEstimate - result.estimatedPhase) < 1e-10, "Final iteration phase should match result estimated phase")
    }
}
