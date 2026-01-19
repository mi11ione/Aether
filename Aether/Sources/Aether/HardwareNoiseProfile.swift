// Copyright (c) 2025-2026 Roman Zhuzhgov
// Licensed under the Apache License, Version 2.0

import Foundation

/// Per-qubit noise parameters for realistic device modeling.
///
/// Captures qubit-specific coherence times and error rates that vary across a device.
/// Real quantum processors have significant variation in qubit quality - some qubits
/// have 100μs T₁ while neighbors may have 50μs. This heterogeneity significantly
/// impacts algorithm performance and optimal qubit mapping.
///
/// **Example:**
/// ```swift
/// let qubitParams = QubitNoiseParameters(
///     t1: 100_000,           // 100μs T₁
///     t2: 80_000,            // 80μs T₂
///     singleQubitErrorRate: 0.001,
///     readoutError0Given1: 0.02,
///     readoutError1Given0: 0.01
/// )
/// ```
///
/// - SeeAlso: ``HardwareNoiseProfile`` for full device characterization
@frozen
public struct QubitNoiseParameters: Sendable, Equatable {
    /// T₁ relaxation time in nanoseconds (energy decay).
    ///
    /// Typical values: 50,000-200,000 ns for superconducting qubits.
    public let t1: Double

    /// T₂ coherence time in nanoseconds (phase decay).
    ///
    /// Always satisfies T₂ ≤ 2T₁. Typical: 30,000-150,000 ns.
    public let t2: Double

    /// Single-qubit gate error rate (probability).
    ///
    /// Typical values: 0.0001-0.005 for superconducting qubits.
    public let singleQubitErrorRate: Double

    /// Readout error P(0|1): probability of measuring 0 when state is 1.
    public let readoutError0Given1: Double

    /// Readout error P(1|0): probability of measuring 1 when state is 0.
    public let readoutError1Given0: Double

    /// Qubit frequency in GHz (for crosstalk modeling).
    public let frequency: Double?

    /// Create qubit noise parameters.
    ///
    /// - Parameters:
    ///   - t1: T₁ relaxation time in nanoseconds
    ///   - t2: T₂ coherence time in nanoseconds (must be ≤ 2*T₁)
    ///   - singleQubitErrorRate: Gate error probability
    ///   - readoutError0Given1: P(measure 0 | prepared 1)
    ///   - readoutError1Given0: P(measure 1 | prepared 0)
    ///   - frequency: Qubit frequency in GHz (optional)
    /// - Precondition: t1 > 0
    /// - Precondition: t2 > 0
    /// - Precondition: t2 ≤ 2*t1
    /// - Precondition: singleQubitErrorRate ∈ [0, 1]
    /// - Precondition: readoutError0Given1 ∈ [0, 1]
    /// - Precondition: readoutError1Given0 ∈ [0, 1]
    public init(
        t1: Double,
        t2: Double,
        singleQubitErrorRate: Double,
        readoutError0Given1: Double,
        readoutError1Given0: Double,
        frequency: Double? = nil,
    ) {
        ValidationUtilities.validatePositiveDouble(t1, name: "T₁")
        ValidationUtilities.validatePositiveDouble(t2, name: "T₂")
        ValidationUtilities.validateT2Constraint(t2, t1: t1)
        ValidationUtilities.validateErrorProbability(singleQubitErrorRate, name: "Single-qubit error rate")
        ValidationUtilities.validateErrorProbability(readoutError0Given1, name: "Readout error P(0|1)")
        ValidationUtilities.validateErrorProbability(readoutError1Given0, name: "Readout error P(1|0)")

        self.t1 = t1
        self.t2 = t2
        self.singleQubitErrorRate = singleQubitErrorRate
        self.readoutError0Given1 = readoutError0Given1
        self.readoutError1Given0 = readoutError1Given0
        self.frequency = frequency
    }

    /// Compute amplitude damping parameter γ for given gate time.
    ///
    /// The amplitude damping parameter γ = 1 - exp(-t/T₁) determines the probability
    /// of energy decay during gate execution.
    ///
    /// **Example:**
    /// ```swift
    /// let params = QubitNoiseParameters(t1: 100_000, t2: 80_000,
    ///     singleQubitErrorRate: 0.001, readoutError0Given1: 0.02, readoutError1Given0: 0.01)
    /// let gamma = params.amplitudeDampingGamma(gateTime: 35)
    /// ```
    ///
    /// - Parameter gateTime: Gate duration in nanoseconds
    /// - Returns: Damping parameter γ ∈ [0, 1]
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    public func amplitudeDampingGamma(gateTime: Double) -> Double {
        1.0 - exp(-gateTime / t1)
    }

    /// Compute phase damping parameter for given gate time.
    ///
    /// Uses T₂ which combines T₁ decay and pure dephasing: 1/T₂ = 1/(2T₁) + 1/T_φ.
    /// Returns 0 if T_φ would be non-positive (pure T₁-limited regime).
    ///
    /// **Example:**
    /// ```swift
    /// let params = QubitNoiseParameters(t1: 100_000, t2: 80_000,
    ///     singleQubitErrorRate: 0.001, readoutError0Given1: 0.02, readoutError1Given0: 0.01)
    /// let gamma = params.phaseDampingGamma(gateTime: 35)
    /// ```
    ///
    /// - Parameter gateTime: Gate duration in nanoseconds
    /// - Returns: Phase damping parameter γ ∈ [0, 1]
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    public func phaseDampingGamma(gateTime: Double) -> Double {
        let tPhi = 1.0 / (1.0 / t2 - 1.0 / (2.0 * t1))
        if tPhi <= 0 || !tPhi.isFinite {
            return 0
        }
        return 1.0 - exp(-gateTime / tPhi)
    }

    /// Create measurement error model from readout parameters.
    ///
    /// **Example:**
    /// ```swift
    /// let params = QubitNoiseParameters(t1: 100_000, t2: 80_000,
    ///     singleQubitErrorRate: 0.001, readoutError0Given1: 0.02, readoutError1Given0: 0.01)
    /// let errorModel = params.measurementErrorModel()
    /// ```
    ///
    /// - Returns: Measurement error model with this qubit's readout error rates
    /// - Complexity: O(1)
    @_effects(readonly)
    @_eagerMove
    public func measurementErrorModel() -> MeasurementErrorModel {
        MeasurementErrorModel(p0Given1: readoutError0Given1, p1Given0: readoutError1Given0)
    }
}

// MARK: - Edge Noise Parameters

/// Two-qubit gate noise parameters for a specific qubit pair.
///
/// Two-qubit gate quality varies significantly with connectivity. Directly connected
/// qubits typically have 0.5-2% error, while SWAP-based routing adds overhead.
///
/// **Example:**
/// ```swift
/// let edge = EdgeNoiseParameters(
///     qubit1: 0,
///     qubit2: 1,
///     twoQubitErrorRate: 0.01,
///     gateTime: 300
/// )
/// ```
@frozen
public struct EdgeNoiseParameters: Sendable, Equatable, Hashable {
    /// First qubit index (always the smaller of the two).
    public let qubit1: Int

    /// Second qubit index (always the larger of the two).
    public let qubit2: Int

    /// Two-qubit gate error rate (probability).
    ///
    /// Typical values: 0.005-0.03 for superconducting qubits.
    public let twoQubitErrorRate: Double

    /// Two-qubit gate time in nanoseconds.
    ///
    /// Typical values: 100-500 ns for CZ/CNOT.
    public let gateTime: Double

    /// Create edge noise parameters.
    ///
    /// Qubit indices are automatically sorted so qubit1 < qubit2 for canonical representation.
    ///
    /// **Example:**
    /// ```swift
    /// let edge = EdgeNoiseParameters(qubit1: 3, qubit2: 1, twoQubitErrorRate: 0.01)
    /// // edge.qubit1 == 1, edge.qubit2 == 3 (sorted)
    /// ```
    ///
    /// - Parameters:
    ///   - qubit1: First qubit index
    ///   - qubit2: Second qubit index
    ///   - twoQubitErrorRate: Gate error probability
    ///   - gateTime: Gate duration in nanoseconds
    /// - Precondition: qubit1 ≠ qubit2
    /// - Precondition: twoQubitErrorRate ∈ [0, 1]
    /// - Precondition: gateTime > 0
    public init(qubit1: Int, qubit2: Int, twoQubitErrorRate: Double, gateTime: Double = 300) {
        ValidationUtilities.validateUniqueQubits([qubit1, qubit2])
        ValidationUtilities.validateErrorProbability(twoQubitErrorRate, name: "Two-qubit error rate")
        ValidationUtilities.validatePositiveDouble(gateTime, name: "Gate time")

        self.qubit1 = min(qubit1, qubit2)
        self.qubit2 = max(qubit1, qubit2)
        self.twoQubitErrorRate = twoQubitErrorRate
        self.gateTime = gateTime
    }

    /// Canonical edge identifier as sorted qubit pair (qubit1, qubit2).
    ///
    /// **Example:**
    /// ```swift
    /// let edge = EdgeNoiseParameters(qubit1: 0, qubit2: 1, twoQubitErrorRate: 0.01)
    /// let key = edge.edgeKey  // (0, 1)
    /// ```
    ///
    /// - Complexity: O(1)
    @inlinable
    public var edgeKey: (Int, Int) { (qubit1, qubit2) }
}

// MARK: - Hardware Noise Profile

/// Complete noise characterization for a quantum device.
///
/// Combines per-qubit noise parameters, two-qubit gate quality for each edge,
/// and connectivity topology. Enables realistic simulation matching specific
/// hardware like IBM, Google, or IonQ devices.
///
/// **Example:**
/// ```swift
/// // Create profile for 5-qubit device
/// let profile = HardwareNoiseProfile(
///     name: "Custom Device",
///     qubitCount: 5,
///     qubitParameters: qubitParams,
///     edges: edges,
///     gateTimings: GateTimingModel.ibmDefault
/// )
///
/// // Or use preset
/// let ibmProfile = HardwareNoiseProfile.ibmManila
/// ```
///
/// - SeeAlso: ``NoiseModel`` for applying profile to simulation
/// - SeeAlso: ``DensityMatrixSimulator`` for noisy execution
@frozen
public struct HardwareNoiseProfile: Sendable {
    /// Device name identifier.
    public let name: String

    /// Total number of qubits.
    public let qubitCount: Int

    /// Per-qubit noise parameters.
    public let qubitParameters: [QubitNoiseParameters]

    /// Two-qubit gate parameters for connected pairs.
    public let edges: [EdgeNoiseParameters]

    /// Gate timing model.
    public let gateTimings: GateTimingModel

    /// Connectivity graph as adjacency set.
    public let connectivity: Set<EdgeKey>

    /// Canonical edge identifier for O(1) connectivity lookup.
    ///
    /// Stores qubit pairs in sorted order (q1 < q2) for consistent hashing.
    ///
    /// **Example:**
    /// ```swift
    /// let key1 = HardwareNoiseProfile.EdgeKey(0, 1)
    /// let key2 = HardwareNoiseProfile.EdgeKey(1, 0)
    /// // key1 == key2 (both normalize to q1=0, q2=1)
    /// ```
    @frozen
    public struct EdgeKey: Hashable, Sendable {
        /// First qubit index (smaller).
        public let q1: Int

        /// Second qubit index (larger).
        public let q2: Int

        /// Create edge key with automatic sorting.
        public init(_ a: Int, _ b: Int) {
            q1 = min(a, b)
            q2 = max(a, b)
        }
    }

    // MARK: - Initialization

    /// Create hardware noise profile with full specification.
    ///
    /// **Example:**
    /// ```swift
    /// let profile = HardwareNoiseProfile(
    ///     name: "Test Device", qubitCount: 2,
    ///     qubitParameters: [qubit0Params, qubit1Params],
    ///     edges: [EdgeNoiseParameters(qubit1: 0, qubit2: 1, twoQubitErrorRate: 0.01)])
    /// ```
    ///
    /// - Parameters:
    ///   - name: Device identifier
    ///   - qubitCount: Total qubits
    ///   - qubitParameters: Per-qubit parameters (count must match qubitCount)
    ///   - edges: Two-qubit gate parameters for connected pairs
    ///   - gateTimings: Gate timing model
    /// - Precondition: qubitParameters.count == qubitCount
    /// - Complexity: O(E) where E is the number of edges
    public init(
        name: String,
        qubitCount: Int,
        qubitParameters: [QubitNoiseParameters],
        edges: [EdgeNoiseParameters],
        gateTimings: GateTimingModel = .ibmDefault,
    ) {
        ValidationUtilities.validateArrayCount(qubitParameters, expected: qubitCount, name: "qubitParameters")

        self.name = name
        self.qubitCount = qubitCount
        self.qubitParameters = qubitParameters
        self.edges = edges
        self.gateTimings = gateTimings

        var conn = Set<EdgeKey>()
        conn.reserveCapacity(edges.count)
        for edge in edges {
            conn.insert(EdgeKey(edge.qubit1, edge.qubit2))
        }
        connectivity = conn
    }

    /// Create profile with uniform parameters for all qubits.
    ///
    /// Convenience initializer for devices with similar qubit quality across all qubits.
    ///
    /// **Example:**
    /// ```swift
    /// let uniformParams = QubitNoiseParameters(t1: 100_000, t2: 80_000,
    ///     singleQubitErrorRate: 0.001, readoutError0Given1: 0.02, readoutError1Given0: 0.01)
    /// let profile = HardwareNoiseProfile(name: "Uniform", qubitCount: 5,
    ///     uniformParameters: uniformParams, edges: edgeList)
    /// ```
    ///
    /// - Parameters:
    ///   - name: Device identifier
    ///   - qubitCount: Total qubits
    ///   - uniformParameters: Parameters applied to all qubits
    ///   - edges: Two-qubit gate parameters
    ///   - gateTimings: Gate timing model
    /// - Complexity: O(Q + E) where Q is qubit count and E is edge count
    public init(
        name: String,
        qubitCount: Int,
        uniformParameters: QubitNoiseParameters,
        edges: [EdgeNoiseParameters],
        gateTimings: GateTimingModel = .ibmDefault,
    ) {
        self.init(
            name: name,
            qubitCount: qubitCount,
            qubitParameters: [QubitNoiseParameters](repeating: uniformParameters, count: qubitCount),
            edges: edges,
            gateTimings: gateTimings,
        )
    }

    // MARK: - Queries

    /// Check if two qubits are directly connected in the device topology.
    ///
    /// **Example:**
    /// ```swift
    /// let profile = HardwareNoiseProfile.ibmManila
    /// let connected = profile.areConnected(0, 1)  // true for linear topology
    /// ```
    ///
    /// - Parameters:
    ///   - q1: First qubit index
    ///   - q2: Second qubit index
    /// - Returns: True if qubits share a direct two-qubit gate connection
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    public func areConnected(_ q1: Int, _ q2: Int) -> Bool {
        connectivity.contains(EdgeKey(q1, q2))
    }

    /// Get edge parameters for a qubit pair.
    ///
    /// **Example:**
    /// ```swift
    /// let profile = HardwareNoiseProfile.ibmManila
    /// if let edge = profile.edgeParameters(q1: 0, q2: 1) {
    ///     print(edge.twoQubitErrorRate)
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - q1: First qubit index
    ///   - q2: Second qubit index
    /// - Returns: Edge parameters if connected, nil otherwise
    /// - Complexity: O(E) where E is the number of edges
    @_effects(readonly)
    public func edgeParameters(q1: Int, q2: Int) -> EdgeNoiseParameters? {
        let key = EdgeKey(q1, q2)
        return edges.first { EdgeKey($0.qubit1, $0.qubit2) == key }
    }

    /// Average single-qubit error rate across all qubits.
    ///
    /// **Example:**
    /// ```swift
    /// let profile = HardwareNoiseProfile.ibmManila
    /// let avgError = profile.averageSingleQubitError
    /// ```
    ///
    /// - Complexity: O(Q)
    @inlinable
    public var averageSingleQubitError: Double {
        qubitParameters.reduce(0) { $0 + $1.singleQubitErrorRate } / Double(qubitCount)
    }

    /// Average two-qubit error rate across all edges.
    ///
    /// **Example:**
    /// ```swift
    /// let profile = HardwareNoiseProfile.ibmManila
    /// let avgError = profile.averageTwoQubitError
    /// ```
    ///
    /// - Complexity: O(E)
    @inlinable
    public var averageTwoQubitError: Double {
        guard !edges.isEmpty else { return 0 }
        return edges.reduce(0) { $0 + $1.twoQubitErrorRate } / Double(edges.count)
    }

    /// Average T₁ relaxation time across all qubits in nanoseconds.
    ///
    /// **Example:**
    /// ```swift
    /// let profile = HardwareNoiseProfile.ibmManila
    /// let avgT1 = profile.averageT1
    /// ```
    ///
    /// - Complexity: O(Q)
    @inlinable
    public var averageT1: Double {
        qubitParameters.reduce(0) { $0 + $1.t1 } / Double(qubitCount)
    }

    /// Average T₂ coherence time across all qubits in nanoseconds.
    ///
    /// **Example:**
    /// ```swift
    /// let profile = HardwareNoiseProfile.ibmManila
    /// let avgT2 = profile.averageT2
    /// ```
    ///
    /// - Complexity: O(Q)
    @inlinable
    public var averageT2: Double {
        qubitParameters.reduce(0) { $0 + $1.t2 } / Double(qubitCount)
    }

    // MARK: - Noise Model Generation

    /// Create NoiseModel from this profile using average parameters.
    ///
    /// Converts hardware-specific characterization into a uniform noise model suitable
    /// for ``DensityMatrixSimulator``. For per-qubit noise, use ``singleQubitChannel(for:)``
    /// directly in custom simulation.
    ///
    /// **Example:**
    /// ```swift
    /// let profile = HardwareNoiseProfile.ibmManila
    /// let noiseModel = profile.toNoiseModel()
    /// ```
    ///
    /// - Returns: Noise model with averaged error rates
    /// - Complexity: O(Q)
    @_effects(readonly)
    @_eagerMove
    public func toNoiseModel() -> NoiseModel {
        let avgSingleError = averageSingleQubitError
        let avgTwoError = averageTwoQubitError

        let avgReadout0Given1 = qubitParameters.reduce(0) { $0 + $1.readoutError0Given1 } / Double(qubitCount)
        let avgReadout1Given0 = qubitParameters.reduce(0) { $0 + $1.readoutError1Given0 } / Double(qubitCount)

        return NoiseModel(
            singleQubitNoise: DepolarizingChannel(errorProbability: avgSingleError),
            twoQubitNoise: TwoQubitDepolarizingChannel(errorProbability: avgTwoError),
            measurementError: MeasurementErrorModel(p0Given1: avgReadout0Given1, p1Given0: avgReadout1Given0),
        )
    }

    /// Create depolarizing channel for a specific qubit.
    ///
    /// **Example:**
    /// ```swift
    /// let profile = HardwareNoiseProfile.ibmManila
    /// let channel = profile.singleQubitChannel(for: 0)
    /// ```
    ///
    /// - Parameter qubit: Qubit index
    /// - Returns: Depolarizing channel with this qubit's error rate
    /// - Precondition: 0 ≤ qubit < qubitCount
    /// - Complexity: O(1)
    @_effects(readonly)
    @_eagerMove
    public func singleQubitChannel(for qubit: Int) -> DepolarizingChannel {
        ValidationUtilities.validateQubitIndex(qubit, qubits: qubitCount)
        return DepolarizingChannel(errorProbability: qubitParameters[qubit].singleQubitErrorRate)
    }

    /// Create two-qubit depolarizing channel for a specific edge.
    ///
    /// If the edge is not in the connectivity graph, returns a channel with the
    /// average two-qubit error rate.
    ///
    /// **Example:**
    /// ```swift
    /// let profile = HardwareNoiseProfile.ibmManila
    /// let channel = profile.twoQubitChannel(for: 0, 1)
    /// ```
    ///
    /// - Parameters:
    ///   - q1: First qubit index
    ///   - q2: Second qubit index
    /// - Returns: Two-qubit depolarizing channel
    /// - Complexity: O(E)
    @_effects(readonly)
    @_eagerMove
    public func twoQubitChannel(for q1: Int, _ q2: Int) -> TwoQubitDepolarizingChannel {
        if let edge = edgeParameters(q1: q1, q2: q2) {
            return TwoQubitDepolarizingChannel(errorProbability: edge.twoQubitErrorRate)
        }
        return TwoQubitDepolarizingChannel(errorProbability: averageTwoQubitError)
    }

    /// Create measurement error models for all qubits.
    ///
    /// **Example:**
    /// ```swift
    /// let profile = HardwareNoiseProfile.ibmManila
    /// let models = profile.measurementErrorModels()
    /// ```
    ///
    /// - Returns: Array of measurement error models indexed by qubit
    /// - Complexity: O(Q)
    @_effects(readonly)
    @_eagerMove
    public func measurementErrorModels() -> [MeasurementErrorModel] {
        qubitParameters.map { $0.measurementErrorModel() }
    }
}

// MARK: - Gate Timing Model

/// Gate duration parameters for timing-aware noise simulation.
///
/// Models gate times for idle noise calculation. Qubits not involved in a gate
/// accumulate T₁/T₂ decay during the gate's execution time.
///
/// **Example:**
/// ```swift
/// let timing = GateTimingModel(
///     singleQubitGateTime: 35,    // 35 ns
///     twoQubitGateTime: 300,      // 300 ns
///     measurementTime: 1000       // 1 μs
/// )
/// ```
@frozen
public struct GateTimingModel: Sendable, Equatable {
    /// Single-qubit gate time in nanoseconds.
    public let singleQubitGateTime: Double

    /// Two-qubit gate time in nanoseconds.
    public let twoQubitGateTime: Double

    /// Three-qubit gate time in nanoseconds (Toffoli).
    public let threeQubitGateTime: Double

    /// Measurement time in nanoseconds.
    public let measurementTime: Double

    /// Create gate timing model.
    ///
    /// **Example:**
    /// ```swift
    /// let timing = GateTimingModel(
    ///     singleQubitGateTime: 35, twoQubitGateTime: 300, measurementTime: 1000)
    /// ```
    ///
    /// - Parameters:
    ///   - singleQubitGateTime: Single-qubit gate duration in ns
    ///   - twoQubitGateTime: Two-qubit gate duration in ns
    ///   - threeQubitGateTime: Three-qubit gate duration in ns
    ///   - measurementTime: Measurement duration in ns
    /// - Precondition: singleQubitGateTime > 0
    /// - Precondition: twoQubitGateTime > 0
    /// - Precondition: threeQubitGateTime > 0
    /// - Precondition: measurementTime > 0
    public init(
        singleQubitGateTime: Double = 35,
        twoQubitGateTime: Double = 300,
        threeQubitGateTime: Double = 600,
        measurementTime: Double = 1000,
    ) {
        ValidationUtilities.validatePositiveDouble(singleQubitGateTime, name: "Single-qubit gate time")
        ValidationUtilities.validatePositiveDouble(twoQubitGateTime, name: "Two-qubit gate time")
        ValidationUtilities.validatePositiveDouble(threeQubitGateTime, name: "Three-qubit gate time")
        ValidationUtilities.validatePositiveDouble(measurementTime, name: "Measurement time")

        self.singleQubitGateTime = singleQubitGateTime
        self.twoQubitGateTime = twoQubitGateTime
        self.threeQubitGateTime = threeQubitGateTime
        self.measurementTime = measurementTime
    }

    /// Get gate time for given qubit arity.
    ///
    /// **Example:**
    /// ```swift
    /// let timing = GateTimingModel.ibmDefault
    /// let cnot_time = timing.gateTime(for: 2)  // 300 ns
    /// ```
    ///
    /// - Parameter qubitsRequired: Number of qubits the gate operates on (1, 2, or 3)
    /// - Returns: Gate duration in nanoseconds
    /// - Complexity: O(1)
    @_effects(readonly)
    @inlinable
    public func gateTime(for qubitsRequired: Int) -> Double {
        switch qubitsRequired {
        case 1: singleQubitGateTime
        case 2: twoQubitGateTime
        case 3: threeQubitGateTime
        default: threeQubitGateTime
        }
    }

    // MARK: - Presets

    /// IBM superconducting qubit typical timings (35ns single, 300ns two-qubit).
    ///
    /// **Example:**
    /// ```swift
    /// let timing = GateTimingModel.ibmDefault
    /// let singleGateTime = timing.singleQubitGateTime  // 35 ns
    /// ```
    public static let ibmDefault = GateTimingModel(
        singleQubitGateTime: 35,
        twoQubitGateTime: 300,
        threeQubitGateTime: 600,
        measurementTime: 1000,
    )

    /// Google Sycamore typical timings (25ns single, 32ns two-qubit).
    ///
    /// **Example:**
    /// ```swift
    /// let timing = GateTimingModel.googleSycamore
    /// let twoQubitTime = timing.twoQubitGateTime  // 32 ns
    /// ```
    public static let googleSycamore = GateTimingModel(
        singleQubitGateTime: 25,
        twoQubitGateTime: 32,
        threeQubitGateTime: 100,
        measurementTime: 1000,
    )

    /// IonQ trapped ion typical timings (10μs single, 200μs two-qubit).
    ///
    /// **Example:**
    /// ```swift
    /// let timing = GateTimingModel.ionQ
    /// let singleGateTime = timing.singleQubitGateTime  // 10,000 ns
    /// ```
    public static let ionQ = GateTimingModel(
        singleQubitGateTime: 10000,
        twoQubitGateTime: 200_000,
        threeQubitGateTime: 600_000,
        measurementTime: 100_000,
    )

    /// Rigetti superconducting typical timings (40ns single, 180ns two-qubit).
    ///
    /// **Example:**
    /// ```swift
    /// let timing = GateTimingModel.rigetti
    /// let twoQubitTime = timing.twoQubitGateTime  // 180 ns
    /// ```
    public static let rigetti = GateTimingModel(
        singleQubitGateTime: 40,
        twoQubitGateTime: 180,
        threeQubitGateTime: 400,
        measurementTime: 800,
    )
}

// MARK: - Device Presets

public extension HardwareNoiseProfile {
    /// IBM Manila 5-qubit device approximation with linear topology 0-1-2-3-4.
    ///
    /// **Example:**
    /// ```swift
    /// let profile = HardwareNoiseProfile.ibmManila
    /// let noiseModel = profile.toNoiseModel()
    /// ```
    ///
    /// - Complexity: O(1)
    @_eagerMove
    static var ibmManila: HardwareNoiseProfile {
        let qubitParams = (0 ..< 5).map { _ in
            QubitNoiseParameters(
                t1: 100_000,
                t2: 80000,
                singleQubitErrorRate: 0.0003,
                readoutError0Given1: 0.02,
                readoutError1Given0: 0.01,
            )
        }

        let edges = [
            EdgeNoiseParameters(qubit1: 0, qubit2: 1, twoQubitErrorRate: 0.008),
            EdgeNoiseParameters(qubit1: 1, qubit2: 2, twoQubitErrorRate: 0.009),
            EdgeNoiseParameters(qubit1: 2, qubit2: 3, twoQubitErrorRate: 0.007),
            EdgeNoiseParameters(qubit1: 3, qubit2: 4, twoQubitErrorRate: 0.01),
        ]

        return HardwareNoiseProfile(
            name: "IBM Manila (approx)",
            qubitCount: 5,
            qubitParameters: qubitParams,
            edges: edges,
            gateTimings: .ibmDefault,
        )
    }

    /// IBM Quito 5-qubit device approximation with T-shaped topology (0 connects to 2, linear 1-2-3-4).
    ///
    /// **Example:**
    /// ```swift
    /// let profile = HardwareNoiseProfile.ibmQuito
    /// let connected = profile.areConnected(0, 2)  // true
    /// ```
    ///
    /// - Complexity: O(1)
    @_eagerMove
    static var ibmQuito: HardwareNoiseProfile {
        let qubitParams = [
            QubitNoiseParameters(t1: 95000, t2: 75000, singleQubitErrorRate: 0.0004,
                                 readoutError0Given1: 0.025, readoutError1Given0: 0.012),
            QubitNoiseParameters(t1: 110_000, t2: 90000, singleQubitErrorRate: 0.0003,
                                 readoutError0Given1: 0.018, readoutError1Given0: 0.008),
            QubitNoiseParameters(t1: 105_000, t2: 85000, singleQubitErrorRate: 0.0003,
                                 readoutError0Given1: 0.020, readoutError1Given0: 0.010),
            QubitNoiseParameters(t1: 100_000, t2: 80000, singleQubitErrorRate: 0.0005,
                                 readoutError0Given1: 0.022, readoutError1Given0: 0.011),
            QubitNoiseParameters(t1: 90000, t2: 70000, singleQubitErrorRate: 0.0004,
                                 readoutError0Given1: 0.024, readoutError1Given0: 0.013),
        ]

        let edges = [
            EdgeNoiseParameters(qubit1: 0, qubit2: 2, twoQubitErrorRate: 0.009),
            EdgeNoiseParameters(qubit1: 1, qubit2: 2, twoQubitErrorRate: 0.008),
            EdgeNoiseParameters(qubit1: 2, qubit2: 3, twoQubitErrorRate: 0.007),
            EdgeNoiseParameters(qubit1: 3, qubit2: 4, twoQubitErrorRate: 0.010),
        ]

        return HardwareNoiseProfile(
            name: "IBM Quito (approx)",
            qubitCount: 5,
            qubitParameters: qubitParams,
            edges: edges,
            gateTimings: .ibmDefault,
        )
    }

    /// Google Sycamore-like 12-qubit subset with 3*4 grid topology.
    ///
    /// **Example:**
    /// ```swift
    /// let profile = HardwareNoiseProfile.googleSycamore12
    /// let connected = profile.areConnected(0, 4)  // true (vertical neighbor)
    /// ```
    ///
    /// - Complexity: O(1)
    @_eagerMove
    static var googleSycamore12: HardwareNoiseProfile {
        let qubitParams = (0 ..< 12).map { _ in
            QubitNoiseParameters(
                t1: 15000,
                t2: 10000,
                singleQubitErrorRate: 0.001,
                readoutError0Given1: 0.03,
                readoutError1Given0: 0.01,
            )
        }

        var edges: [EdgeNoiseParameters] = []
        edges.reserveCapacity(17)

        for row in 0 ..< 3 {
            for col in 0 ..< 4 {
                let q = row * 4 + col
                if col < 3 {
                    edges.append(EdgeNoiseParameters(qubit1: q, qubit2: q + 1, twoQubitErrorRate: 0.006))
                }
                if row < 2 {
                    edges.append(EdgeNoiseParameters(qubit1: q, qubit2: q + 4, twoQubitErrorRate: 0.006))
                }
            }
        }

        return HardwareNoiseProfile(
            name: "Google Sycamore 12Q (approx)",
            qubitCount: 12,
            qubitParameters: qubitParams,
            edges: edges,
            gateTimings: .googleSycamore,
        )
    }

    /// IonQ Harmony 11-qubit device approximation with all-to-all connectivity.
    ///
    /// Trapped ion systems have full connectivity but slower gate times.
    ///
    /// **Example:**
    /// ```swift
    /// let profile = HardwareNoiseProfile.ionQHarmony
    /// let connected = profile.areConnected(0, 10)  // true (all-to-all)
    /// ```
    ///
    /// - Complexity: O(Q²)
    @_eagerMove
    static var ionQHarmony: HardwareNoiseProfile {
        let qubitParams = (0 ..< 11).map { _ in
            QubitNoiseParameters(
                t1: 10_000_000,
                t2: 1_000_000,
                singleQubitErrorRate: 0.0003,
                readoutError0Given1: 0.005,
                readoutError1Given0: 0.005,
            )
        }

        var edges: [EdgeNoiseParameters] = []
        edges.reserveCapacity(55)
        for i in 0 ..< 11 {
            for j in (i + 1) ..< 11 {
                edges.append(EdgeNoiseParameters(
                    qubit1: i,
                    qubit2: j,
                    twoQubitErrorRate: 0.01,
                    gateTime: 200_000,
                ))
            }
        }

        return HardwareNoiseProfile(
            name: "IonQ Harmony (approx)",
            qubitCount: 11,
            qubitParameters: qubitParams,
            edges: edges,
            gateTimings: .ionQ,
        )
    }

    /// Rigetti Aspen-M 8-qubit subset approximation with octagonal topology.
    ///
    /// **Example:**
    /// ```swift
    /// let profile = HardwareNoiseProfile.rigettiAspen8
    /// let connected = profile.areConnected(0, 7)  // true (ring)
    /// ```
    ///
    /// - Complexity: O(1)
    @_eagerMove
    static var rigettiAspen8: HardwareNoiseProfile {
        let qubitParams = (0 ..< 8).map { _ in
            QubitNoiseParameters(
                t1: 30000,
                t2: 20000,
                singleQubitErrorRate: 0.002,
                readoutError0Given1: 0.04,
                readoutError1Given0: 0.02,
            )
        }

        let edges = [
            EdgeNoiseParameters(qubit1: 0, qubit2: 1, twoQubitErrorRate: 0.02),
            EdgeNoiseParameters(qubit1: 1, qubit2: 2, twoQubitErrorRate: 0.02),
            EdgeNoiseParameters(qubit1: 2, qubit2: 3, twoQubitErrorRate: 0.02),
            EdgeNoiseParameters(qubit1: 3, qubit2: 4, twoQubitErrorRate: 0.02),
            EdgeNoiseParameters(qubit1: 4, qubit2: 5, twoQubitErrorRate: 0.02),
            EdgeNoiseParameters(qubit1: 5, qubit2: 6, twoQubitErrorRate: 0.02),
            EdgeNoiseParameters(qubit1: 6, qubit2: 7, twoQubitErrorRate: 0.02),
            EdgeNoiseParameters(qubit1: 7, qubit2: 0, twoQubitErrorRate: 0.02),
            EdgeNoiseParameters(qubit1: 0, qubit2: 4, twoQubitErrorRate: 0.025),
            EdgeNoiseParameters(qubit1: 2, qubit2: 6, twoQubitErrorRate: 0.025),
        ]

        return HardwareNoiseProfile(
            name: "Rigetti Aspen-M 8Q (approx)",
            qubitCount: 8,
            qubitParameters: qubitParams,
            edges: edges,
            gateTimings: .rigetti,
        )
    }

    /// Create custom linear chain topology with uniform parameters.
    ///
    /// **Example:**
    /// ```swift
    /// let profile = HardwareNoiseProfile.linearChain(qubits: 10, t1: 100_000, t2: 80_000)
    /// let connected = profile.areConnected(4, 5)  // true
    /// ```
    ///
    /// - Parameters:
    ///   - qubits: Number of qubits
    ///   - t1: T₁ time in ns
    ///   - t2: T₂ time in ns
    ///   - singleQubitError: Single-qubit gate error rate
    ///   - twoQubitError: Two-qubit gate error rate
    ///   - readoutError: Symmetric readout error (P(1|0) = readoutError * 0.5)
    /// - Returns: Linear chain hardware profile
    /// - Complexity: O(Q)
    @_eagerMove
    static func linearChain(
        qubits: Int,
        t1: Double = 100_000,
        t2: Double = 80000,
        singleQubitError: Double = 0.001,
        twoQubitError: Double = 0.01,
        readoutError: Double = 0.02,
    ) -> HardwareNoiseProfile {
        let qubitParams = (0 ..< qubits).map { _ in
            QubitNoiseParameters(
                t1: t1,
                t2: t2,
                singleQubitErrorRate: singleQubitError,
                readoutError0Given1: readoutError,
                readoutError1Given0: readoutError * 0.5,
            )
        }

        let edges = (0 ..< qubits - 1).map { i in
            EdgeNoiseParameters(qubit1: i, qubit2: i + 1, twoQubitErrorRate: twoQubitError)
        }

        return HardwareNoiseProfile(
            name: "Linear Chain (\(qubits)Q)",
            qubitCount: qubits,
            qubitParameters: qubitParams,
            edges: edges,
        )
    }

    /// Create custom grid topology with uniform parameters.
    ///
    /// Creates a rows * cols rectangular grid where each qubit connects to its
    /// horizontal and vertical neighbors.
    ///
    /// **Example:**
    /// ```swift
    /// let profile = HardwareNoiseProfile.grid(rows: 3, cols: 4)
    /// let connected = profile.areConnected(0, 4)  // true (vertical neighbor)
    /// ```
    ///
    /// - Parameters:
    ///   - rows: Number of rows
    ///   - cols: Number of columns
    ///   - t1: T₁ time in ns
    ///   - t2: T₂ time in ns
    ///   - singleQubitError: Single-qubit gate error rate
    ///   - twoQubitError: Two-qubit gate error rate
    ///   - readoutError: Symmetric readout error (P(1|0) = readoutError * 0.5)
    /// - Returns: Grid hardware profile
    /// - Complexity: O(rows * cols)
    @_eagerMove
    static func grid(
        rows: Int,
        cols: Int,
        t1: Double = 100_000,
        t2: Double = 80000,
        singleQubitError: Double = 0.001,
        twoQubitError: Double = 0.01,
        readoutError: Double = 0.02,
    ) -> HardwareNoiseProfile {
        let qubitCount = rows * cols

        let qubitParams = (0 ..< qubitCount).map { _ in
            QubitNoiseParameters(
                t1: t1,
                t2: t2,
                singleQubitErrorRate: singleQubitError,
                readoutError0Given1: readoutError,
                readoutError1Given0: readoutError * 0.5,
            )
        }

        let edgeCount = (rows - 1) * cols + rows * (cols - 1)
        var edges: [EdgeNoiseParameters] = []
        edges.reserveCapacity(edgeCount)

        for r in 0 ..< rows {
            for c in 0 ..< cols {
                let q = r * cols + c
                if c < cols - 1 {
                    edges.append(EdgeNoiseParameters(qubit1: q, qubit2: q + 1, twoQubitErrorRate: twoQubitError))
                }
                if r < rows - 1 {
                    edges.append(EdgeNoiseParameters(qubit1: q, qubit2: q + cols, twoQubitErrorRate: twoQubitError))
                }
            }
        }

        return HardwareNoiseProfile(
            name: "Grid (\(rows)*\(cols))",
            qubitCount: qubitCount,
            qubitParameters: qubitParams,
            edges: edges,
        )
    }
}
